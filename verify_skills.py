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

DEFAULT_FEEDBACK_DIR = Path(config.PROJECT_ROOT) / "feedback_store"

# --- configurable weights ---------------------------------------------------

W_CONFIDENCE = 0.40
W_AGREEMENT = 0.30
W_FREQUENCY = 0.15
W_DENSITY = 0.15
FREQ_CAP = 20          # frequency above this is treated as 1.0
HIGH_PERCENTILE = 80   # top 20 %
MEDIUM_PERCENTILE = 50 # top 50 %
VAGUE_SKILL_PENALTY = 0.6  # multiplier for single-word hard skills (downrank vague skills)


def _is_vague_hard_skill(skill: str, skill_type: str) -> bool:
    """True if Hard skill with fewer than 2 words (e.g. 'design', 'propose')."""
    if not skill or not isinstance(skill, str):
        return False
    if str(skill_type).strip().lower() != "hard":
        return False
    words = str(skill).strip().split()
    return len(words) < 2


def load_calibrated_threshold(feedback_dir: Path) -> dict:
    """Load calibrated threshold if available."""
    path = feedback_dir / "calibrated_threshold.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_calibration_weights(output_dir: Path) -> dict | None:
    """Load Brier-based weights from parameter_validation_report.json if available.
    Map: confidence_score -> W_CONFIDENCE, semantic_density -> W_DENSITY,
    context_agreement -> W_AGREEMENT. W_FREQUENCY stays fixed (no Brier in report).
    Returns dict with keys W_CONFIDENCE, W_AGREEMENT, W_DENSITY, W_FREQUENCY, or None.
    """
    path = output_dir / "parameter_validation_report.json"
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "ok" or "validations" not in report:
            return None
        brier_map = {}
        for v in report["validations"]:
            if isinstance(v, dict) and "column" in v and "brier_score" in v:
                brier_map[v["column"]] = float(v["brier_score"])
        # Map validated columns to our signals
        w_conf = 1.0 / (1.0 + brier_map.get("confidence_score", 0.2))
        w_dens = 1.0 / (1.0 + brier_map.get("semantic_density", 0.2))
        w_agr = 1.0 / (1.0 + brier_map.get("context_agreement", 0.2))
        w_freq = 0.15  # Fixed (no Brier)
        total = w_conf + w_dens + w_agr + w_freq
        return {
            "W_CONFIDENCE": w_conf / total,
            "W_AGREEMENT": w_agr / total,
            "W_DENSITY": w_dens / total,
            "W_FREQUENCY": w_freq / total,
        }
    except Exception:
        return None


def compute_composite(df: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    """Compute a composite verification score per row.
    weights: optional dict with W_CONFIDENCE, W_AGREEMENT, W_DENSITY, W_FREQUENCY.
    If None, uses module-level constants.
    """
    w = weights or {}
    w_conf = w.get("W_CONFIDENCE", W_CONFIDENCE)
    w_agr = w.get("W_AGREEMENT", W_AGREEMENT)
    w_freq = w.get("W_FREQUENCY", W_FREQUENCY)
    w_dens = w.get("W_DENSITY", W_DENSITY)

    conf = pd.to_numeric(df["confidence_score"], errors="coerce").fillna(0.0)

    if "model_agreement_score" in df.columns:
        agree = pd.to_numeric(df["model_agreement_score"], errors="coerce").fillna(conf)
    elif "context_agreement" in df.columns:
        agree = pd.to_numeric(df["context_agreement"], errors="coerce").fillna(conf)
    else:
        agree = conf.copy()

    if "semantic_density" in df.columns:
        density = pd.to_numeric(df["semantic_density"], errors="coerce").fillna(0.5)
    else:
        density = pd.Series(0.5, index=df.index)

    freq = df["_freq"] if "_freq" in df.columns else pd.Series(1.0, index=df.index)
    freq_norm = (freq / FREQ_CAP).clip(upper=1.0)

    composite = (
        w_conf * conf
        + w_agr * agree
        + w_freq * freq_norm
        + w_dens * density
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
    parser.add_argument(
        "--feedback_dir",
        type=str,
        default=None,
        help="Feedback store directory (default: PROJECT_ROOT/feedback_store)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    feedback_dir = Path(args.feedback_dir) if args.feedback_dir else DEFAULT_FEEDBACK_DIR
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

    cal_weights = load_calibration_weights(output_dir)
    if cal_weights:
        print(f"[INFO] Using calibration-aware weights from parameter_validation_report.json")
    else:
        print("[INFO] Using fixed weights (parameter_validation_report.json not found or invalid)")
    df["composite_score"] = compute_composite(df, weights=cal_weights)

    # Apply vague-skill penalty for single-word hard skills
    if "skill" in df.columns and "type" in df.columns:
        vague_mask = df.apply(
            lambda r: _is_vague_hard_skill(str(r.get("skill", "")), str(r.get("type", ""))),
            axis=1,
        )
        if vague_mask.any():
            df.loc[vague_mask, "composite_score"] *= VAGUE_SKILL_PENALTY
            print(f"[INFO] Applied vague-skill penalty to {vague_mask.sum()} single-word hard skills")

    cal = load_calibrated_threshold(feedback_dir)
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
