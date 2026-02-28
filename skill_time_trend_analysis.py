"""
skill_time_trend_analysis.py

Compute statistically rigorous time trends for skills using
advanced_skills_with_dates.csv.  Uses scipy.stats.linregress for
slope, p-value, and R-squared, with Benjamini-Hochberg FDR correction
to control false discoveries across many skills.

Scientific methods (see SCIENTIFIC_METHODOLOGY.md §8):
    - Linear regression per skill (freq vs month_idx)
    - Benjamini-Hochberg q-values; trend_label: Emerging/Declining/Stable (q < 0.05)

Outputs:
    skill_time_trends.csv  (with slope, p_value, q_value, r_squared, trend_label)
    trend_stability_report.json  (optional, with --stability flag)
"""

import argparse
import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats

import config


def compute_trend_group(grp: pd.DataFrame) -> Tuple[float, float, float]:
    """Return (slope, p_value, r_squared) from linear regression."""
    x = grp["month_idx"].values.astype(float)
    y = grp["freq"].values.astype(float)

    if len(x) < 3:
        return 0.0, 1.0, 0.0

    result = sp_stats.linregress(x, y)
    return float(result.slope), float(result.pvalue), float(result.rvalue ** 2)


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Return q-values (FDR-adjusted p-values) using Benjamini-Hochberg."""
    n = len(p_values)
    if n == 0:
        return np.array([])

    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    q_values = np.empty(n)
    q_values[sorted_idx[-1]] = sorted_p[-1]

    for i in range(n - 2, -1, -1):
        rank = i + 1
        adjusted = sorted_p[i] * n / rank
        q_values[sorted_idx[i]] = min(adjusted, q_values[sorted_idx[i + 1]])

    return np.clip(q_values, 0, 1)


def label_trend_fdr(slope: float, q_value: float, alpha: float = 0.05) -> str:
    """Label trend using FDR-corrected q-value instead of raw p-value."""
    if q_value > alpha:
        return "Stable"
    return "Emerging" if slope > 0 else "Declining"


def compute_trends_for_df(df: pd.DataFrame, min_jobs: int) -> pd.DataFrame:
    """Core trend computation (reusable for stability analysis)."""
    df["year_month"] = df["job_date"].dt.to_period("M").astype(str)
    months_sorted = sorted(df["year_month"].unique())
    month_to_idx = {m: i for i, m in enumerate(months_sorted)}
    df["month_idx"] = df["year_month"].map(month_to_idx)

    grp = (
        df.groupby(["skill", "year_month", "month_idx"])
        .agg(freq=("job_id", "nunique"))
        .reset_index()
    )

    total_counts = grp.groupby("skill")["freq"].sum().reset_index()
    valid_skills = total_counts[total_counts["freq"] >= min_jobs]["skill"]
    grp = grp[grp["skill"].isin(valid_skills)].copy()

    if grp.empty:
        return pd.DataFrame()

    trend_rows = []
    for skill, g in grp.groupby("skill"):
        g_sorted = g.sort_values("month_idx")
        slope, p_value, r_squared = compute_trend_group(g_sorted)
        trend_rows.append({
            "skill": skill,
            "total_freq": g_sorted["freq"].sum(),
            "n_months": g_sorted["year_month"].nunique(),
            "slope": slope,
            "p_value": p_value,
            "r_squared": r_squared,
        })

    trends = pd.DataFrame(trend_rows)

    p_vals = trends["p_value"].values
    trends["q_value"] = benjamini_hochberg(p_vals)

    trends["trend_label"] = trends.apply(
        lambda r: label_trend_fdr(r["slope"], r["q_value"]), axis=1
    )

    return trends


def stability_analysis(df: pd.DataFrame, min_jobs_values: List[int],
                       top_n: int = 20, n_seeds: int = 3) -> dict:
    """Measure stability of top-N emerging skills across min_jobs and seeds."""
    results = []

    for mj in min_jobs_values:
        seed_sets = []
        for seed in range(n_seeds):
            df_sampled = df.sample(frac=0.9, random_state=seed).copy()
            trends = compute_trends_for_df(df_sampled, mj)
            if trends.empty:
                continue
            emerging = set(
                trends[trends["trend_label"] == "Emerging"]
                .nlargest(top_n, "slope")["skill"]
                .tolist()
            )
            seed_sets.append(emerging)

        if len(seed_sets) < 2:
            results.append({
                "min_jobs": mj, "n_seeds": len(seed_sets),
                "mean_jaccard": 0.0, "top_n": top_n,
            })
            continue

        jaccards = []
        for i in range(len(seed_sets)):
            for j in range(i + 1, len(seed_sets)):
                a, b = seed_sets[i], seed_sets[j]
                if a or b:
                    jacc = len(a & b) / len(a | b) if (a | b) else 0.0
                    jaccards.append(jacc)

        results.append({
            "min_jobs": mj,
            "n_seeds": len(seed_sets),
            "mean_jaccard": round(float(np.mean(jaccards)), 4) if jaccards else 0.0,
            "std_jaccard": round(float(np.std(jaccards)), 4) if jaccards else 0.0,
            "top_n": top_n,
        })

    return {"stability_by_min_jobs": results}


def main():
    parser = argparse.ArgumentParser(
        description="Analyze time trends of skills with FDR-controlled significance."
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(config.OUTPUT_DIR),
        help="Directory where advanced_skills_with_dates.csv is stored.",
    )
    parser.add_argument(
        "--input", type=str, default="advanced_skills_with_dates.csv",
        help="Input CSV (with job_date).",
    )
    parser.add_argument(
        "--output", type=str, default="skill_time_trends.csv",
        help="Output CSV file for skill trends.",
    )
    parser.add_argument(
        "--min_jobs", type=int, default=10,
        help="Minimum number of jobs mentioning a skill to include in trends.",
    )
    parser.add_argument(
        "--only_hard", action="store_true",
        help="If set, only analyze skills where type == 'hard'.",
    )
    parser.add_argument(
        "--fdr_alpha", type=float, default=0.05,
        help="FDR significance threshold (default: 0.05).",
    )
    parser.add_argument(
        "--stability", action="store_true",
        help="Run stability analysis across min_jobs settings and seeds.",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    in_path = out_dir / args.input

    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} not found")

    print(f"[INFO] Reading {in_path}")
    df = pd.read_csv(in_path)

    if "job_date" not in df.columns:
        raise ValueError("Input file must contain 'job_date' column. "
                         "Run enrich_with_dates.py first.")

    df["job_date"] = pd.to_datetime(df["job_date"], errors="coerce")
    df = df[df["job_date"].notna()].copy()

    if args.only_hard and "type" in df.columns:
        df = df[df["type"].astype(str).str.lower().isin(["hard", "both"])].copy()

    trends = compute_trends_for_df(df, args.min_jobs)

    if trends.empty:
        print("[WARN] No skills passed the min_jobs filter; nothing to analyze.")
        return

    n_emerging = (trends["trend_label"] == "Emerging").sum()
    n_declining = (trends["trend_label"] == "Declining").sum()
    n_stable = (trends["trend_label"] == "Stable").sum()
    print(f"[INFO] Trend summary (FDR alpha={args.fdr_alpha}): "
          f"Emerging={n_emerging}, Declining={n_declining}, Stable={n_stable}")
    print(f"[INFO] q-value stats: mean={trends['q_value'].mean():.4f}, "
          f"median={trends['q_value'].median():.4f}")

    out_path = out_dir / args.output
    trends.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved skill trends to {out_path}")

    if args.stability:
        print("[INFO] Running stability analysis...")
        min_jobs_values = [5, 10, 15, 20]
        stab = stability_analysis(df, min_jobs_values, top_n=20, n_seeds=3)
        stab_path = out_dir / "trend_stability_report.json"
        stab_path.write_text(json.dumps(stab, indent=2), encoding="utf-8")
        print(f"[INFO] Saved stability report to {stab_path}")
        for entry in stab["stability_by_min_jobs"]:
            print(f"  min_jobs={entry['min_jobs']}: "
                  f"Jaccard={entry['mean_jaccard']:.3f} +/- {entry.get('std_jaccard', 0):.3f}")

    try:
        top_emerging = trends[trends["trend_label"] == "Emerging"].nlargest(10, "slope")
        top_declining = trends[trends["trend_label"] == "Declining"].nsmallest(10, "slope")

        if not top_emerging.empty or not top_declining.empty:
            plt.figure(figsize=(10, 6))
            if not top_emerging.empty:
                sns.barplot(data=top_emerging, x="slope", y="skill",
                            label="Emerging", orient="h")
            if not top_declining.empty:
                sns.barplot(data=top_declining, x="slope", y="skill",
                            color="red", label="Declining", orient="h")
            plt.xlabel("Slope of monthly job count")
            plt.ylabel("Skill")
            plt.title("Top Emerging vs Declining Skills (FDR-controlled)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "skill_trend_barplot.png", dpi=300)
            plt.close()
            print("[INFO] Saved skill_trend_barplot.png")
    except Exception as e:
        print(f"[WARN] Could not create trend plot: {e}")


if __name__ == "__main__":
    main()
