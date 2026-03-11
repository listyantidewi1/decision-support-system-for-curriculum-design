"""
trend_score_sensitivity.py

Sensitivity analysis for future-domain trend_score values.

Perturbs each domain's trend_score by ±0.2 (capped at [-1, 1]) and
recomputes top-20 recommendation rankings.  Jaccard overlap vs the
unperturbed baseline measures robustness.

Outputs:
    results/trend_score_sensitivity_report.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config

PERTURBATIONS = [-0.2, 0.0, +0.2]


def _load_recommendations_inputs(output_dir: Path):
    """Load the inputs needed to recompute priority scores."""
    from recommendations import (
        load_skills, load_trends, load_future_weights, load_coverage,
        build_skill_demand, compute_priority_scores,
    )
    skills_df = load_skills(output_dir)
    trends = load_trends(output_dir)
    future_weights = load_future_weights(output_dir)
    coverage = load_coverage(output_dir)
    demand = build_skill_demand(skills_df)
    return demand, trends, future_weights, coverage, compute_priority_scores


def main():
    parser = argparse.ArgumentParser(
        description="Trend score sensitivity analysis for domain taxonomy."
    )
    parser.add_argument("--output_dir", type=str, default=str(config.OUTPUT_DIR))
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    domains_path = Path(config.PROJECT_ROOT) / "future_domains.csv"
    if not domains_path.exists():
        print("[ERROR] future_domains.csv not found.")
        return

    domains_df = pd.read_csv(domains_path)
    print(f"[INFO] Loaded {len(domains_df)} domains from {domains_path}")

    demand, trends, fw_base, coverage, compute_fn = _load_recommendations_inputs(out_dir)
    if demand.empty:
        print("[ERROR] No skills data found. Run the pipeline first.")
        return

    baseline_recs = compute_fn(demand, trends, fw_base, coverage)
    baseline_top = set(baseline_recs.head(args.top_n)["skill"].tolist()) if not baseline_recs.empty else set()

    results = {"baseline_top_n": list(baseline_top), "perturbations": []}

    for delta in PERTURBATIONS:
        if delta == 0.0:
            label = "baseline"
            jaccard = 1.0
            results["perturbations"].append({
                "delta": delta, "label": label,
                "jaccard_vs_baseline": jaccard,
            })
            continue

        label = f"+{delta}" if delta > 0 else str(delta)
        perturbed_domains = domains_df.copy()
        perturbed_domains["trend_score"] = (
            perturbed_domains["trend_score"] + delta
        ).clip(-1.0, 1.0)

        perturbed_path = out_dir / "_temp_perturbed_domains.csv"
        perturbed_domains.to_csv(perturbed_path, index=False)

        fw_perturbed = fw_base.copy()
        if not fw_perturbed.empty and "best_future_domain" in fw_perturbed.columns:
            domain_scores = dict(zip(
                perturbed_domains["future_domain"].str.strip().str.lower(),
                perturbed_domains["trend_score"],
            ))
            for idx, row in fw_perturbed.iterrows():
                dom = str(row.get("best_future_domain", "")).strip().lower()
                sim = float(row.get("similarity", 0))
                new_ts = domain_scores.get(dom, 0)
                fw_perturbed.at[idx, "future_weight"] = sim * new_ts

        recs = compute_fn(demand, trends, fw_perturbed, coverage)
        top_set = set(recs.head(args.top_n)["skill"].tolist()) if not recs.empty else set()
        inter = len(baseline_top & top_set)
        union = len(baseline_top | top_set)
        jaccard = round(inter / union, 4) if union else 1.0

        results["perturbations"].append({
            "delta": delta, "label": label,
            "jaccard_vs_baseline": jaccard,
            "top_skills": list(top_set)[:args.top_n],
        })
        print(f"  delta={label}: Jaccard vs baseline = {jaccard:.3f}")

        perturbed_path.unlink(missing_ok=True)

    min_jaccard = min(p["jaccard_vs_baseline"] for p in results["perturbations"])
    results["min_jaccard"] = min_jaccard
    results["robust"] = min_jaccard >= 0.5

    out_path = out_dir / "trend_score_sensitivity_report.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Saved report to {out_path}")

    if results["robust"]:
        print("[INFO] Trend scores are ROBUST: Jaccard >= 0.5 under ±0.2 perturbation.")
    else:
        print("[WARN] Trend scores may be FRAGILE: Jaccard < 0.5 under ±0.2 perturbation.")


if __name__ == "__main__":
    main()
