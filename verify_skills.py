"""
verify_skills.py

Reads advanced_skills.csv and assigns verification categories using a
multi-factor composite score (confidence, model agreement, frequency,
semantic density).

Threshold selection (in priority order):
  1. Calibrated threshold from feedback_store/calibrated_threshold.json
     (produced by validate_parameters.py against human labels)
  2. Percentile-based fallback: top 20% = HIGH, next 30% = MEDIUM

Outputs:
    verified_skills.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config

FEEDBACK_DIR = Path(config.PROJECT_ROOT) / "feedback_store"

# --- configurable weights ---------------------------------------------------

W_CONFIDENCE = 0.40
W_AGREEMENT = 0.30
W_FREQUENCY = 0.15
W_DENSITY = 0.15
FREQ_CAP = 20          # frequency above this is treated as 1.0
HIGH_PERCENTILE = 80   # top 20 %
MEDIUM_PERCENTILE = 50 # top 50 %


def load_calibrated_threshold() -> dict:
    """Load calibrated threshold if available."""
    path = FEEDBACK_DIR / "calibrated_threshold.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def compute_composite(df: pd.DataFrame) -> pd.Series:
    """Compute a composite verification score per row."""
    conf = pd.to_numeric(df["confidence_score"], errors="coerce").fillna(0.0)

    if "model_agreement_score" in df.columns:
        agree = pd.to_numeric(df["model_agreement_score"], errors="coerce").fillna(conf)
    else:
        agree = conf.copy()

    if "semantic_density" in df.columns:
        density = pd.to_numeric(df["semantic_density"], errors="coerce").fillna(0.5)
    else:
        density = pd.Series(0.5, index=df.index)

    freq = df["_freq"] if "_freq" in df.columns else pd.Series(1.0, index=df.index)
    freq_norm = (freq / FREQ_CAP).clip(upper=1.0)

    composite = (
        W_CONFIDENCE * conf
        + W_AGREEMENT * agree
        + W_FREQUENCY * freq_norm
        + W_DENSITY * density
    )
    return composite


def assign_levels_calibrated(composite: pd.Series,
                             cal_threshold: float) -> pd.Series:
    """Use a calibrated threshold from human validation.
    HIGH = above threshold, MEDIUM = above threshold * 0.75, else Low."""
    med_thresh = cal_threshold * 0.75

    def _level(val):
        if pd.isna(val):
            return "Unknown"
        if val >= cal_threshold:
            return "Verified_HIGH"
        if val >= med_thresh:
            return "Verified_MEDIUM"
        return "Low_Confidence"

    return composite.apply(_level)


def assign_levels_percentile(composite: pd.Series) -> pd.Series:
    """Percentile-based verification levels (fallback)."""
    valid = composite.dropna()
    if valid.empty:
        return pd.Series("Unknown", index=composite.index)

    high_thresh = float(np.percentile(valid, HIGH_PERCENTILE))
    med_thresh = float(np.percentile(valid, MEDIUM_PERCENTILE))

    def _level(val):
        if pd.isna(val):
            return "Unknown"
        if val >= high_thresh:
            return "Verified_HIGH"
        if val >= med_thresh:
            return "Verified_MEDIUM"
        return "Low_Confidence"

    return composite.apply(_level)


def main():
    parser = argparse.ArgumentParser(
        description="Assign multi-factor verification categories to advanced skills."
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(config.OUTPUT_DIR),
        help="Directory containing advanced_skills.csv",
    )
    parser.add_argument(
        "--input", type=str, default="advanced_skills.csv",
    )
    parser.add_argument(
        "--output", type=str, default="verified_skills.csv",
    )
    parser.add_argument(
        "--force_percentile", action="store_true",
        help="Force percentile-based thresholds even if calibrated threshold exists",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    input_path = output_dir / args.input
    output_path = output_dir / args.output

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"[INFO] Reading skills from {input_path}")
    df = pd.read_csv(input_path)

    if "confidence_score" not in df.columns:
        raise ValueError("advanced_skills.csv must contain 'confidence_score' column.")

    df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")

    if "skill" in df.columns:
        freq_map = df.groupby("skill").size()
        df["_freq"] = df["skill"].map(freq_map).fillna(1).astype(float)
    else:
        df["_freq"] = 1.0

    df["composite_score"] = compute_composite(df)

    cal = load_calibrated_threshold()
    if cal and "confidence_threshold" in cal and not args.force_percentile:
        cal_thresh = float(cal["confidence_threshold"])
        print(f"[INFO] Using calibrated threshold: {cal_thresh:.4f} "
              f"(AUC={cal.get('auc_roc', '?')}, Brier={cal.get('brier_score', '?')})")
        df["verification_level"] = assign_levels_calibrated(df["composite_score"], cal_thresh)
        df["threshold_source"] = "calibrated"
    else:
        print("[INFO] Using percentile-based thresholds (no calibrated threshold available)")
        df["verification_level"] = assign_levels_percentile(df["composite_score"])
        df["threshold_source"] = "percentile"

    df["is_verified"] = df["verification_level"].isin(
        ["Verified_HIGH", "Verified_MEDIUM"]
    )

    df = df.drop(columns=["_freq"], errors="ignore")

    print("[INFO] Verification distribution:")
    print(df["verification_level"].value_counts())
    print(f"[INFO] Composite score stats: "
          f"mean={df['composite_score'].mean():.3f}, "
          f"std={df['composite_score'].std():.3f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved verified skills to {output_path}")


if __name__ == "__main__":
    main()
