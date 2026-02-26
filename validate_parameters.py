"""
validate_parameters.py

Validates pipeline scoring parameters against human review data.
Computes AUC-ROC for confidence as a predictor of human validity,
and suggests optimal thresholds using Youden's J statistic.

Inputs:
    - feedback_store/human_verified_skills.csv
    - config.OUTPUT_DIR/expert_review_skills.csv (with human_valid column)

Outputs:
    - parameter_validation_report.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config

FEEDBACK_DIR = Path(config.PROJECT_ROOT) / "feedback_store"


def load_reviewed_skills(output_dir: Path) -> pd.DataFrame:
    """Load expert_review_skills.csv with human_valid filled in."""
    path = output_dir / "expert_review_skills.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "human_valid" not in df.columns or "confidence_score" not in df.columns:
        return pd.DataFrame()
    df["human_valid"] = df["human_valid"].astype(str).str.strip().str.lower()
    df = df[df["human_valid"].isin(["valid", "invalid"])].copy()
    df["is_valid"] = (df["human_valid"] == "valid").astype(int)
    df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")
    df = df.dropna(subset=["confidence_score"])
    return df


def compute_auc_roc(y_true: np.ndarray, y_score: np.ndarray):
    """Compute AUC-ROC without sklearn dependency using trapezoidal rule."""
    desc_idx = np.argsort(y_score)[::-1]
    y_true = y_true[desc_idx]
    y_score = y_score[desc_idx]

    distinct_idxs = np.where(np.diff(y_score))[0]
    threshold_idxs = np.concatenate([distinct_idxs, [len(y_true) - 1]])

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = np.cumsum(1 - y_true)[threshold_idxs]

    total_pos = y_true.sum()
    total_neg = len(y_true) - total_pos
    if total_pos == 0 or total_neg == 0:
        return 0.5, [], [], []

    tpr = np.concatenate([[0], tps / total_pos])
    fpr = np.concatenate([[0], fps / total_neg])
    thresholds = y_score[threshold_idxs]

    auc = float(np.trapz(tpr, fpr))
    return auc, fpr.tolist(), tpr.tolist(), thresholds.tolist()


def youden_j_optimal(fpr_list, tpr_list, thresholds):
    """Find the threshold that maximizes Youden's J = TPR - FPR."""
    if not thresholds:
        return None
    fpr_arr = np.array(fpr_list[1:])
    tpr_arr = np.array(tpr_list[1:])
    j = tpr_arr - fpr_arr
    best_idx = int(np.argmax(j))
    return {
        "threshold": float(thresholds[best_idx]),
        "tpr": float(tpr_arr[best_idx]),
        "fpr": float(fpr_arr[best_idx]),
        "youden_j": float(j[best_idx]),
    }


def validate_column(df: pd.DataFrame, col: str) -> dict:
    """Compute AUC-ROC and optimal threshold for a single score column."""
    if col not in df.columns:
        return {"column": col, "status": "missing"}
    subset = df.dropna(subset=[col])
    if len(subset) < 10:
        return {"column": col, "status": "insufficient_data", "n": len(subset)}

    y_true = subset["is_valid"].values
    y_score = pd.to_numeric(subset[col], errors="coerce").fillna(0).values

    auc, fpr, tpr, thresholds = compute_auc_roc(y_true, y_score)
    optimal = youden_j_optimal(fpr, tpr, thresholds)
    corr = float(np.corrcoef(y_score, y_true)[0, 1]) if y_true.std() > 0 else 0.0

    return {
        "column": col,
        "n": len(subset),
        "n_valid": int(y_true.sum()),
        "n_invalid": int(len(y_true) - y_true.sum()),
        "auc_roc": round(auc, 4),
        "pearson_corr": round(corr, 4),
        "optimal_threshold": optimal,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate pipeline scoring parameters against human review data."
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(config.OUTPUT_DIR),
        help="Directory containing expert_review_skills.csv",
    )
    parser.add_argument(
        "--output", type=str, default="parameter_validation_report.json",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    df = load_reviewed_skills(out_dir)

    if df.empty:
        print("[WARN] No reviewed skills with human_valid found. "
              "Run the review workflow first.")
        report = {"status": "no_data"}
    else:
        print(f"[INFO] Loaded {len(df)} reviewed skills "
              f"({df['is_valid'].sum()} valid, "
              f"{(~df['is_valid'].astype(bool)).sum()} invalid)")

        score_cols = ["confidence_score", "semantic_density", "context_agreement"]
        results = [validate_column(df, c) for c in score_cols]

        report = {
            "status": "ok",
            "total_reviewed": len(df),
            "validations": results,
        }

        for r in results:
            if "auc_roc" in r:
                print(f"  {r['column']}: AUC-ROC={r['auc_roc']:.4f}, "
                      f"corr={r['pearson_corr']:.4f}")
                if r.get("optimal_threshold"):
                    opt = r["optimal_threshold"]
                    print(f"    Optimal threshold={opt['threshold']:.3f} "
                          f"(TPR={opt['tpr']:.3f}, FPR={opt['fpr']:.3f})")

    report_path = out_dir / args.output
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[INFO] Saved report to {report_path}")


if __name__ == "__main__":
    main()
