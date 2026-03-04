"""
evaluate_future_mapping.py

Validates future-domain mapping accuracy against expert-labeled gold set.

Inputs:
    DATA/labels/gold_future_domain.csv  (with true_domain_id filled in)
    results/future_skill_weights.csv or future_skill_weights_dummy.csv

Outputs:
    results/future_mapping_evaluation_report.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

import config

LABELS_DIR = Path(config.PROJECT_ROOT) / "DATA" / "labels"


def load_gold(labels_dir: Path) -> pd.DataFrame:
    path = labels_dir / "gold_future_domain_merged.csv"
    if not path.exists():
        path = labels_dir / "gold_future_domain.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "true_domain_id" not in df.columns:
        return pd.DataFrame()
    df["true_domain_id"] = df["true_domain_id"].astype(str).str.strip().str.lower()
    df = df[~df["true_domain_id"].isin(["", "nan"])].copy()
    return df


def load_pipeline_mapping(output_dir: Path) -> pd.DataFrame:
    for name in ["future_skill_weights.csv", "future_skill_weights_dummy.csv"]:
        path = output_dir / name
        if path.exists():
            df = pd.read_csv(path)
            if "best_domain_id" in df.columns:
                return df
    return pd.DataFrame()


def evaluate(gold: pd.DataFrame, mapping: pd.DataFrame) -> dict:
    if gold.empty:
        return {"status": "no_gold_data"}

    item_col = "item_text"
    if item_col not in gold.columns:
        return {"status": "missing_item_text_column"}

    gold_lower = gold.copy()
    gold_lower["_item_key"] = gold_lower[item_col].astype(str).str.strip().str.lower()

    results = []
    for _, row in gold_lower.iterrows():
        key = row["_item_key"]
        item_type = str(row.get("item_type", "")).strip().lower()
        true_id = row["true_domain_id"]

        col = "skill" if item_type == "skill" else "knowledge"
        if col not in mapping.columns:
            col = "skill" if "skill" in mapping.columns else "knowledge"

        match = mapping[mapping[col].astype(str).str.strip().str.lower() == key]
        if match.empty:
            results.append({
                "item": key, "true_domain": true_id,
                "predicted_domain": "", "correct": False,
                "margin": 0.0, "status": "not_in_mapping",
            })
            continue

        pred_id = str(match.iloc[0].get("best_domain_id", "")).strip().lower()
        margin = float(match.iloc[0].get("mapping_margin", 0))

        is_correct = (true_id == pred_id)
        is_none = true_id in ("none", "unclear")

        results.append({
            "item": key, "true_domain": true_id,
            "predicted_domain": pred_id, "correct": is_correct,
            "is_none_or_unclear": is_none, "margin": round(margin, 4),
        })

    results_df = pd.DataFrame(results)

    # Ensure boolean: pandas can store as float if mixed/NaN, and ~ fails on float
    mask = (results_df["is_none_or_unclear"] == True)
    evaluable = results_df[~mask].copy()
    n_eval = len(evaluable)
    n_correct = evaluable["correct"].sum() if n_eval > 0 else 0
    top1_acc = n_correct / n_eval if n_eval > 0 else 0.0

    none_rate = mask.sum() / len(results_df) if len(results_df) > 0 else 0.0

    confusion = Counter()
    incorrect_mask = (evaluable["correct"] != True)
    for _, r in evaluable[incorrect_mask].iterrows():
        pair = f"{r['true_domain']} -> {r['predicted_domain']}"
        confusion[pair] += 1

    return {
        "status": "ok",
        "total_items": len(results_df),
        "evaluable_items": n_eval,
        "top1_accuracy": round(top1_acc, 4),
        "none_unclear_rate": round(none_rate, 4),
        "mean_margin": round(float(results_df["margin"].mean()), 4),
        "median_margin": round(float(results_df["margin"].median()), 4),
        "top_confusion_pairs": confusion.most_common(10),
        "per_item": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate future-domain mapping against gold labels."
    )
    parser.add_argument("--output_dir", type=str, default=str(config.OUTPUT_DIR))
    parser.add_argument("--labels_dir", type=str, default=str(LABELS_DIR))
    parser.add_argument("--output", type=str, default="future_mapping_evaluation_report.json")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    labels = Path(args.labels_dir)

    gold = load_gold(labels)
    if gold.empty:
        print("[WARN] No gold future-domain labels found. "
              "Export and label gold_future_domain.csv first.")
        report = {"status": "no_data"}
    else:
        print(f"[INFO] Loaded {len(gold)} gold domain labels")
        mapping = load_pipeline_mapping(out_dir)
        if mapping.empty:
            print("[WARN] No pipeline mapping found (future_skill_weights*.csv)")
            report = {"status": "no_mapping"}
        else:
            print(f"[INFO] Loaded {len(mapping)} mapping entries")
            report = evaluate(gold, mapping)
            if report.get("status") == "ok":
                print(f"  Top-1 accuracy: {report['top1_accuracy']:.3f}")
                print(f"  None/unclear rate: {report['none_unclear_rate']:.3f}")
                print(f"  Mean margin: {report['mean_margin']:.3f}")

    out_path = out_dir / args.output
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Saved report to {out_path}")


if __name__ == "__main__":
    main()
