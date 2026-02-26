"""
recommendations.py

Generates ranked curriculum recommendations with evidence traces.

Each recommendation is a skill/knowledge gap with:
    - priority_score (composite of demand, trend, future_weight, coverage_gap)
    - evidence trace (supporting job_ids, trend stats, domain info)
    - curriculum gap status

Supports ablation: signals can be enabled/disabled individually.

Outputs:
    results/recommendations.csv
    results/recommendations_report.json

Evaluation (when gold labels are available):
    results/recommendation_evaluation_report.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config

LABELS_DIR = Path(config.PROJECT_ROOT) / "DATA" / "labels"


# ---------------------------------------------------------------------------
# Signal loaders
# ---------------------------------------------------------------------------

def load_skills(output_dir: Path) -> pd.DataFrame:
    for name in ["advanced_skills_human_filtered.csv", "verified_skills.csv",
                  "advanced_skills.csv"]:
        path = output_dir / name
        if path.exists():
            df = pd.read_csv(path)
            print(f"[INFO] Loaded skills from {name} ({len(df)} rows)")
            return df
    return pd.DataFrame()


def load_trends(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "skill_time_trends.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_future_weights(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "future_skill_weights.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_coverage(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "coverage_report.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Build recommendation table
# ---------------------------------------------------------------------------

def build_skill_demand(skills_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-skill demand statistics."""
    if skills_df.empty or "skill" not in skills_df.columns:
        return pd.DataFrame()

    agg = {"skill": "count"}
    if "confidence_score" in skills_df.columns:
        agg["confidence_score"] = "mean"

    grp = skills_df.groupby("skill").agg(**{
        "demand_freq": ("skill", "count"),
        "mean_confidence": ("confidence_score", "mean") if "confidence_score" in skills_df.columns else ("skill", "count"),
    }).reset_index()

    if "type" in skills_df.columns:
        type_map = skills_df.groupby("skill")["type"].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"
        ).reset_index()
        type_map.columns = ["skill", "skill_type"]
        grp = grp.merge(type_map, on="skill", how="left")

    if "bloom" in skills_df.columns:
        bloom_map = skills_df.groupby("skill")["bloom"].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"
        ).reset_index()
        bloom_map.columns = ["skill", "bloom_level"]
        grp = grp.merge(bloom_map, on="skill", how="left")

    if "job_id" in skills_df.columns:
        job_ids = skills_df.groupby("skill")["job_id"].apply(
            lambda x: list(x.unique()[:5])
        ).reset_index()
        job_ids.columns = ["skill", "example_job_ids"]
        grp = grp.merge(job_ids, on="skill", how="left")

    return grp


def compute_priority_scores(
    demand: pd.DataFrame,
    trends: pd.DataFrame,
    future_weights: pd.DataFrame,
    coverage: pd.DataFrame,
    use_trend: bool = True,
    use_future: bool = True,
    use_coverage: bool = True,
    use_validity: bool = True,
    w_demand: float = 0.30,
    w_trend: float = 0.25,
    w_future: float = 0.25,
    w_coverage: float = 0.20,
) -> pd.DataFrame:
    """Compute priority score per skill. Signals can be ablated."""
    if demand.empty:
        return pd.DataFrame()

    rec = demand.copy()
    skill_col = "skill"

    max_freq = rec["demand_freq"].max()
    rec["demand_norm"] = rec["demand_freq"] / max_freq if max_freq > 0 else 0.0

    # Trend signal
    rec["trend_score_norm"] = 0.0
    rec["trend_label"] = "Unknown"
    rec["trend_q_value"] = 1.0
    if use_trend and not trends.empty and skill_col in trends.columns:
        trend_map = trends.set_index(skill_col)
        for idx, row in rec.iterrows():
            s = row[skill_col]
            if s in trend_map.index:
                t = trend_map.loc[s]
                if isinstance(t, pd.DataFrame):
                    t = t.iloc[0]
                label = str(t.get("trend_label", "Stable"))
                q = float(t.get("q_value", t.get("p_value", 1.0)))
                slope = float(t.get("slope", 0))
                rec.at[idx, "trend_label"] = label
                rec.at[idx, "trend_q_value"] = q
                if label == "Emerging":
                    rec.at[idx, "trend_score_norm"] = min(1.0, abs(slope) / 0.5)
                elif label == "Declining":
                    rec.at[idx, "trend_score_norm"] = max(-1.0, -abs(slope) / 0.5)

    # Future weight signal
    rec["future_weight"] = 0.0
    rec["future_domain"] = ""
    rec["mapping_margin"] = 0.0
    if use_future and not future_weights.empty and skill_col in future_weights.columns:
        fw_map = future_weights.drop_duplicates(subset=[skill_col]).set_index(skill_col)
        for idx, row in rec.iterrows():
            s = row[skill_col]
            if s in fw_map.index:
                fw = fw_map.loc[s]
                if isinstance(fw, pd.DataFrame):
                    fw = fw.iloc[0]
                rec.at[idx, "future_weight"] = float(fw.get("future_weight", 0))
                rec.at[idx, "future_domain"] = str(fw.get("best_future_domain", ""))
                rec.at[idx, "mapping_margin"] = float(fw.get("mapping_margin", 0))

    fw_max = rec["future_weight"].abs().max()
    rec["future_norm"] = rec["future_weight"] / fw_max if fw_max > 0 else 0.0

    # Coverage gap signal (1.0 = not covered at all, 0.0 = fully covered)
    rec["coverage_gap"] = 1.0
    if use_coverage and not coverage.empty:
        if "coverage_percentage" in coverage.columns:
            mean_cov = coverage["coverage_percentage"].mean() / 100.0
            rec["coverage_gap"] = 1.0 - mean_cov

    # Validity signal (use mean_confidence as proxy)
    rec["validity_score"] = 0.5
    if use_validity and "mean_confidence" in rec.columns:
        rec["validity_score"] = rec["mean_confidence"].fillna(0.5)

    # Composite priority score
    rec["priority_score"] = (
        w_demand * rec["demand_norm"]
        + (w_trend * rec["trend_score_norm"].clip(-1, 1) if use_trend else 0)
        + (w_future * rec["future_norm"] if use_future else 0)
        + (w_coverage * rec["coverage_gap"] if use_coverage else 0)
    )

    rec = rec.sort_values("priority_score", ascending=False).reset_index(drop=True)
    rec.insert(0, "rank", range(1, len(rec) + 1))

    return rec


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def dcg(scores: List[float], k: int) -> float:
    result = 0.0
    for i, s in enumerate(scores[:k]):
        result += s / math.log2(i + 2)
    return result


def ndcg_at_k(relevance: List[float], k: int) -> float:
    actual = dcg(relevance, k)
    ideal = dcg(sorted(relevance, reverse=True), k)
    return actual / ideal if ideal > 0 else 0.0


def evaluate_recommendations(recs: pd.DataFrame, top_n: int = 20) -> dict:
    """Evaluate top-N recommendations if expert labels are available.

    Expects a column 'expert_priority' in recs (yes/no/partial) for top-N items.
    If not present, returns instructions for labeling.
    """
    if "expert_priority" not in recs.columns:
        return {
            "status": "no_expert_labels",
            "instructions": (
                "Add 'expert_priority' column to the top-N rows of "
                "recommendations.csv with values: yes, no, partial. "
                "Then re-run this script."
            ),
        }

    top = recs.head(top_n).copy()
    top["expert_priority"] = top["expert_priority"].astype(str).str.strip().str.lower()
    labeled = top[top["expert_priority"].isin(["yes", "no", "partial"])]

    if labeled.empty:
        return {"status": "no_labeled_items"}

    n = len(labeled)
    n_yes = (labeled["expert_priority"] == "yes").sum()
    n_partial = (labeled["expert_priority"] == "partial").sum()
    n_no = (labeled["expert_priority"] == "no").sum()

    precision = (n_yes + 0.5 * n_partial) / n if n > 0 else 0.0

    relevance = []
    for _, r in labeled.iterrows():
        ep = r["expert_priority"]
        relevance.append(1.0 if ep == "yes" else (0.5 if ep == "partial" else 0.0))

    ndcg = ndcg_at_k(relevance, top_n)

    return {
        "status": "ok",
        "top_n": top_n,
        "labeled_items": n,
        "n_yes": int(n_yes),
        "n_partial": int(n_partial),
        "n_no": int(n_no),
        "precision_at_n": round(precision, 4),
        "ndcg_at_n": round(ndcg, 4),
    }


def run_ablation(demand, trends, future_weights, coverage, top_n: int = 20) -> dict:
    """Run ablation study: remove one signal at a time."""
    variants = {
        "full": {"use_trend": True, "use_future": True, "use_coverage": True},
        "no_trend": {"use_trend": False, "use_future": True, "use_coverage": True},
        "no_future": {"use_trend": True, "use_future": False, "use_coverage": True},
        "no_coverage": {"use_trend": True, "use_future": True, "use_coverage": False},
        "demand_only": {"use_trend": False, "use_future": False, "use_coverage": False},
    }

    results = {}
    for name, flags in variants.items():
        recs = compute_priority_scores(demand, trends, future_weights, coverage, **flags)
        top_skills = recs.head(top_n)["skill"].tolist() if not recs.empty else []
        results[name] = {
            "top_skills": top_skills,
            "n_total": len(recs),
        }

    # Compute Jaccard overlaps between full and ablated
    full_set = set(results["full"]["top_skills"])
    for name in results:
        variant_set = set(results[name]["top_skills"])
        union = full_set | variant_set
        inter = full_set & variant_set
        results[name]["jaccard_vs_full"] = round(len(inter) / len(union), 4) if union else 1.0

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate ranked curriculum recommendations with evidence traces."
    )
    parser.add_argument("--output_dir", type=str, default=str(config.OUTPUT_DIR))
    parser.add_argument("--top_n", type=int, default=20,
                        help="Top-N recommendations to highlight (default: 20)")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study (remove signals one at a time)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate against expert labels in recommendations.csv")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    skills_df = load_skills(out_dir)
    trends = load_trends(out_dir)
    future_weights = load_future_weights(out_dir)
    coverage = load_coverage(out_dir)

    if skills_df.empty:
        print("[ERROR] No skills data found. Run the pipeline first.")
        return

    demand = build_skill_demand(skills_df)
    print(f"[INFO] Built demand table: {len(demand)} unique skills")

    if not trends.empty:
        print(f"[INFO] Loaded {len(trends)} trend records")
    if not future_weights.empty:
        print(f"[INFO] Loaded {len(future_weights)} future weight records")
    if not coverage.empty:
        print(f"[INFO] Loaded {len(coverage)} coverage records")

    recs = compute_priority_scores(demand, trends, future_weights, coverage)
    recs["expert_priority"] = ""

    rec_path = out_dir / "recommendations.csv"
    recs.to_csv(rec_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved {len(recs)} recommendations to {rec_path}")

    top = recs.head(args.top_n)
    print(f"\n[INFO] Top-{args.top_n} curriculum priorities:")
    for _, r in top.iterrows():
        print(f"  #{int(r['rank']):3d}  {r['skill'][:50]:50s}  "
              f"priority={r['priority_score']:.3f}  "
              f"demand={int(r['demand_freq'])}  "
              f"trend={r['trend_label']}  "
              f"future={r['future_weight']:.2f}")

    report = {
        "total_skills": len(recs),
        "top_n": args.top_n,
        "top_recommendations": top[[
            "rank", "skill", "priority_score", "demand_freq",
            "trend_label", "trend_q_value", "future_weight",
            "future_domain", "mapping_margin", "coverage_gap",
        ]].to_dict(orient="records"),
    }

    if args.ablation:
        print("\n[INFO] Running ablation study...")
        ablation = run_ablation(demand, trends, future_weights, coverage,
                                top_n=args.top_n)
        report["ablation"] = ablation
        for name, data in ablation.items():
            print(f"  {name}: Jaccard vs full = {data['jaccard_vs_full']:.3f}")

    if args.evaluate:
        existing = out_dir / "recommendations.csv"
        if existing.exists():
            eval_df = pd.read_csv(existing)
            eval_result = evaluate_recommendations(eval_df, top_n=args.top_n)
            report["evaluation"] = eval_result
            if eval_result.get("status") == "ok":
                print(f"\n[INFO] Evaluation: P@{args.top_n}={eval_result['precision_at_n']:.3f}, "
                      f"NDCG@{args.top_n}={eval_result['ndcg_at_n']:.3f}")
            else:
                print(f"[INFO] Evaluation: {eval_result.get('status', 'unknown')}")

    report_path = out_dir / "recommendations_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[INFO] Saved report to {report_path}")


if __name__ == "__main__":
    main()
