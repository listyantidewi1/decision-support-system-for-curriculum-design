"""
trend_score_sensitivity.py

Perturb future-domain trend_scores by ±0.2 and measure impact on the
top-20 recommendation ranking (Jaccard overlap vs baseline).

This validates that recommendation outputs are robust to uncertainty in
the expert-assigned trend_score values in future_domains.csv.

Outputs:
    results/trend_score_sensitivity_report.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config

# Reuse recommendation logic
import sys
sys.path.insert(0, str(Path(config.PROJECT_ROOT)))
from recommendations import (
    load_skills, load_trends, load_future_weights, load_coverage,
    build_skill_demand, compute_priority_scores,
)


def perturb_future_weights(fw: pd.DataFrame, delta: float) -> pd.DataFrame:
    """Shift all trend_score values by delta, clamp to [-1, 1], recompute future_weight."""
    fw = fw.copy()
    if "trend_score" not in fw.columns:
        return fw
    fw["trend_score"] = (fw["trend_score"] + delta).clip(-1, 1)
    if "similarity" in fw.columns:
        fw["future_weight"] = fw["similarity"] * fw["trend_score"]
    return fw


def main():
    parser = argparse.ArgumentParser(
        description="Trend-score sensitivity analysis for recommendations."
    )
    parser.add_argument("--output_dir", type=str, default=str(config.OUTPUT_DIR))
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    skills_df = load_skills(out_dir)
    trends = load_trends(out_dir)
    fw_base = load_future_weights(out_dir)
    coverage = load_coverage(out_dir)

    if skills_df.empty:
        print("[ERROR] No skills found. Run pipeline first.")
        return

    demand = build_skill_demand(skills_df)

    baseline_recs = compute_priority_scores(demand, trends, fw_base, coverage)
    baseline_top = set(baseline_recs.head(args.top_n)["skill"].tolist()) if not baseline_recs.empty else set()

    deltas = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    results = []

    for delta in deltas:
        fw_perturbed = perturb_future_weights(fw_base, delta)
        recs = compute_priority_scores(demand, trends, fw_perturbed, coverage)
        top_set = set(recs.head(args.top_n)["skill"].tolist()) if not recs.empty else set()

        inter = len(baseline_top & top_set)
        union = len(baseline_top | top_set)
        jaccard = round(inter / union, 4) if union else 1.0

        results.append({
            "delta": delta,
            "jaccard_vs_baseline": jaccard,
            "top_skills_overlap": inter,
            "top_skills_union": union,
        })

        print(f"  delta={delta:+.1f}: Jaccard={jaccard:.3f} "
              f"(overlap={inter}/{union})")

    jaccards = [r["jaccard_vs_baseline"] for r in results if r["delta"] != 0.0]
    min_jaccard = min(jaccards) if jaccards else 1.0

    report = {
        "top_n": args.top_n,
        "baseline_top_skills": sorted(baseline_top),
        "perturbations": results,
        "min_jaccard": round(min_jaccard, 4),
        "robust": min_jaccard >= 0.60,
        "interpretation": (
            "Robust: top-20 ranking is stable under ±0.2 trend_score perturbation"
            if min_jaccard >= 0.60
            else "Fragile: recommendations are sensitive to trend_score values"
        ),
    }

    out_path = out_dir / "trend_score_sensitivity_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[INFO] Saved report to {out_path}")
    print(f"[INFO] Min Jaccard = {min_jaccard:.3f} → {'ROBUST' if report['robust'] else 'FRAGILE'}")


if __name__ == "__main__":
    main()
