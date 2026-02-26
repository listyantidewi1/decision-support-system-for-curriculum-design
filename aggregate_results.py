"""
aggregate_results.py

Aggregate multiple experimental runs of the advanced pipeline into
a single set of CSVs and reports for final analysis.

Aggregates:
    - CSV outputs (concatenated with run_id column)
    - JSON reports (collected into a per-run array)
    - Run metadata (collected for reproducibility audit)

Usage:
    python aggregate_results.py --run_dirs results_run1 results_run2 results_run3
"""

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd


CSV_FILES = [
    "advanced_skills.csv",
    "advanced_knowledge.csv",
    "coverage_report.csv",
    "comprehensive_analysis.csv",
    "model_comparison.csv",
    "future_skill_weights_dummy.csv",
    "future_skill_weights.csv",
    "verified_skills.csv",
    "advanced_skills_with_dates.csv",
    "skill_time_trends.csv",
    "recommendations.csv",
]

JSON_REPORTS = [
    "extraction_evaluation_report.json",
    "parameter_validation_report.json",
    "future_mapping_evaluation_report.json",
    "competency_evaluation_report.json",
    "recommendations_report.json",
    "trend_stability_report.json",
    "run_metadata.json",
]


def aggregate_csv(file_name: str, run_dirs: List[Path], output_dir: Path) -> None:
    frames = []
    for run_dir in run_dirs:
        src = run_dir / file_name
        if not src.exists():
            continue
        try:
            df = pd.read_csv(src)
            df["run_id"] = run_dir.name
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Failed to read {src}: {e}")

    if not frames:
        return

    combined = pd.concat(frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / file_name
    combined.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Aggregated {file_name} -> {out_path} ({len(combined)} rows)")


def aggregate_json(file_name: str, run_dirs: List[Path], output_dir: Path) -> None:
    collected = []
    for run_dir in run_dirs:
        src = run_dir / file_name
        if not src.exists():
            continue
        try:
            data = json.loads(src.read_text(encoding="utf-8"))
            data["_run_id"] = run_dir.name
            collected.append(data)
        except Exception as e:
            print(f"[WARN] Failed to read {src}: {e}")

    if not collected:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / file_name
    out_path.write_text(json.dumps(collected, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Aggregated {file_name} -> {out_path} ({len(collected)} runs)")


def compute_cross_run_summary(run_dirs: List[Path], output_dir: Path) -> None:
    """Compute summary statistics across runs for key metrics."""
    summary = {"n_runs": len(run_dirs), "runs": [r.name for r in run_dirs]}

    # Extraction metrics across runs
    extraction_reports = []
    for run_dir in run_dirs:
        path = run_dir / "extraction_evaluation_report.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                extraction_reports.append(data)
            except Exception:
                pass

    if extraction_reports:
        precisions = []
        for r in extraction_reports:
            skills = r.get("skills", {})
            if isinstance(skills, dict) and "by_source" in skills:
                for src in skills["by_source"]:
                    if src.get("source") == "all":
                        precisions.append(src.get("precision", 0))
        if precisions:
            import numpy as np
            summary["extraction_precision"] = {
                "mean": round(float(np.mean(precisions)), 4),
                "std": round(float(np.std(precisions)), 4),
                "values": precisions,
            }

    # Recommendation metrics across runs
    rec_reports = []
    for run_dir in run_dirs:
        path = run_dir / "recommendations_report.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                rec_reports.append(data)
            except Exception:
                pass

    if rec_reports:
        eval_results = [r.get("evaluation", {}) for r in rec_reports
                       if r.get("evaluation", {}).get("status") == "ok"]
        if eval_results:
            import numpy as np
            p_at_n = [e["precision_at_n"] for e in eval_results]
            ndcg = [e["ndcg_at_n"] for e in eval_results]
            summary["recommendation_precision"] = {
                "mean": round(float(np.mean(p_at_n)), 4),
                "std": round(float(np.std(p_at_n)), 4),
            }
            summary["recommendation_ndcg"] = {
                "mean": round(float(np.mean(ndcg)), 4),
                "std": round(float(np.std(ndcg)), 4),
            }

    out_path = output_dir / "cross_run_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Saved cross-run summary to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multiple experiment runs into a single results directory."
    )
    parser.add_argument(
        "--run_dirs", type=str, nargs="+", required=True,
        help="Run directories to aggregate (e.g. results_run1 results_run2)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results_aggregated",
        help="Output directory (default: results_aggregated)",
    )
    args = parser.parse_args()

    run_dirs = [Path(d) for d in args.run_dirs]
    for d in run_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Run directory not found: {d}")

    output_dir = Path(args.output_dir)

    print(f"[INFO] Aggregating {len(run_dirs)} runs:")
    for d in run_dirs:
        print(f"       - {d}")

    for fname in CSV_FILES:
        aggregate_csv(fname, run_dirs, output_dir)

    for fname in JSON_REPORTS:
        aggregate_json(fname, run_dirs, output_dir)

    compute_cross_run_summary(run_dirs, output_dir)

    print("[INFO] Aggregation complete.")


if __name__ == "__main__":
    main()
