"""
skill_time_trend_analysis.py

Compute statistically rigorous time trends for skills using
advanced_skills_with_dates.csv.  Uses scipy.stats.linregress for
slope, p-value, and R-squared.

Output:
    skill_time_trends.csv  (with slope, p_value, r_squared, trend_label)
"""

import argparse
from pathlib import Path
from typing import Tuple

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


def label_trend(slope: float, p_value: float, alpha: float = 0.05) -> str:
    """Label trend based on slope direction *and* statistical significance."""
    if p_value > alpha:
        return "Stable"
    return "Emerging" if slope > 0 else "Declining"


def main():
    parser = argparse.ArgumentParser(
        description="Analyze time trends of skills using advanced_skills_with_dates.csv."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(config.OUTPUT_DIR),
        help="Directory where advanced_skills_with_dates.csv is stored.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="advanced_skills_with_dates.csv",
        help="Input CSV (with job_date).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="skill_time_trends.csv",
        help="Output CSV file for skill trends.",
    )
    parser.add_argument(
        "--min_jobs",
        type=int,
        default=10,
        help="Minimum number of jobs mentioning a skill to include in trends.",
    )
    parser.add_argument(
        "--only_hard",
        action="store_true",
        help="If set, only analyze skills where type == 'hard'.",
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

    # Convert dates
    df["job_date"] = pd.to_datetime(df["job_date"], errors="coerce")
    df = df[df["job_date"].notna()].copy()

    # Optionally restrict to hard skills (include "Both" as hybrid skills are often curriculum-relevant)
    if args.only_hard and "type" in df.columns:
        df = df[df["type"].astype(str).str.lower().isin(["hard", "both"])].copy()

    # Month string and month index
    df["year_month"] = df["job_date"].dt.to_period("M").astype(str)

    # Create a sorted index for months
    months_sorted = sorted(df["year_month"].unique())
    month_to_idx = {m: i for i, m in enumerate(months_sorted)}
    df["month_idx"] = df["year_month"].map(month_to_idx)

    # Count frequency per skill per month
    grp = (
        df.groupby(["skill", "year_month", "month_idx"])
        .agg(freq=("job_id", "nunique"))
        .reset_index()
    )

    # Filter skills that appear in at least min_jobs postings
    total_counts = grp.groupby("skill")["freq"].sum().reset_index()
    valid_skills = total_counts[total_counts["freq"] >= args.min_jobs]["skill"]
    grp = grp[grp["skill"].isin(valid_skills)].copy()

    if grp.empty:
        print("[WARN] No skills passed the min_jobs filter; nothing to analyze.")
        return

    # Compute slope, p-value, R-squared per skill
    trend_rows = []
    for skill, g in grp.groupby("skill"):
        g_sorted = g.sort_values("month_idx")
        slope, p_value, r_squared = compute_trend_group(g_sorted)
        trend_rows.append(
            {
                "skill": skill,
                "total_freq": g_sorted["freq"].sum(),
                "n_months": g_sorted["year_month"].nunique(),
                "slope": slope,
                "p_value": p_value,
                "r_squared": r_squared,
            }
        )

    trends = pd.DataFrame(trend_rows)
    trends["trend_label"] = trends.apply(
        lambda r: label_trend(r["slope"], r["p_value"]), axis=1
    )

    out_path = out_dir / args.output
    trends.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved skill trends to {out_path}")

    # Optional quick plot: top emerging & declining skills
    try:
        # Top 10 emerging & declining by slope magnitude
        top_emerging = trends[trends["trend_label"] == "Emerging"].nlargest(10, "slope")
        top_declining = trends[trends["trend_label"] == "Declining"].nsmallest(
            10, "slope"
        )

        if not top_emerging.empty or not top_declining.empty:
            plt.figure(figsize=(10, 6))
            if not top_emerging.empty:
                sns.barplot(
                    data=top_emerging,
                    x="slope",
                    y="skill",
                    label="Emerging",
                    orient="h",
                )
            if not top_declining.empty:
                sns.barplot(
                    data=top_declining,
                    x="slope",
                    y="skill",
                    color="red",
                    label="Declining",
                    orient="h",
                )
            plt.xlabel("Slope of monthly job count")
            plt.ylabel("Skill")
            plt.title("Top Emerging vs Declining Skills")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "skill_trend_barplot.png", dpi=300)
            plt.close()
            print(f"[INFO] Saved skill_trend_barplot.png")
    except Exception as e:
        print(f"[WARN] Could not create trend plot: {e}")


if __name__ == "__main__":
    main()
