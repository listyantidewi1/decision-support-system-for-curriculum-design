"""
evaluate_extraction.py

Evaluates extraction quality (precision, recall, F1) using a labeled gold set.
Computes metrics overall and per extraction source (BERT / GPT / Hybrid).

Inputs:
    DATA/labels/gold_skills.csv   (with is_correct filled in)
    DATA/labels/gold_knowledge.csv (with is_correct filled in)

Outputs:
    results/extraction_evaluation_report.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

import config

LABELS_DIR = Path(config.PROJECT_ROOT) / "DATA" / "labels"


def _load_labeled(path: Path, id_col: str = "gold_id") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "is_correct" not in df.columns:
        return pd.DataFrame()
    df["is_correct"] = df["is_correct"].astype(str).str.strip().str.lower()
    df = df[df["is_correct"].isin(["yes", "no"])].copy()
    df["is_correct_bin"] = (df["is_correct"] == "yes").astype(int)
    return df


def precision_recall_f1(y_true: List[int]) -> Dict[str, float]:
    """Compute precision, recall, F1 for extraction correctness.

    In the gold-set paradigm:
      - precision = fraction of extracted items that are correct
      - recall is approximated as the fraction of correct items
        (true recall requires knowing all true items, which we
        estimate from the gold set's positive rate)
    """
    n = len(y_true)
    if n == 0:
        return {"n": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = sum(y_true)
    fp = n - tp

    precision = tp / n if n > 0 else 0.0

    recall = tp / n if n > 0 else 0.0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "n": n,
        "n_correct": tp,
        "n_incorrect": fp,
        "precision": round(precision, 4),
        "recall_estimate": round(recall, 4),
        "f1": round(f1, 4),
    }


def wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return (round(max(0, lo), 4), round(min(1, hi), 4))


def evaluate_by_source(df: pd.DataFrame, source_col: str = "source") -> List[Dict]:
    results = []
    if source_col not in df.columns:
        metrics = precision_recall_f1(df["is_correct_bin"].tolist())
        metrics["source"] = "all"
        ci = wilson_ci(metrics["precision"], metrics["n"])
        metrics["precision_ci_95"] = list(ci)
        results.append(metrics)
        return results

    for src, grp in df.groupby(source_col):
        metrics = precision_recall_f1(grp["is_correct_bin"].tolist())
        metrics["source"] = str(src)
        ci = wilson_ci(metrics["precision"], metrics["n"])
        metrics["precision_ci_95"] = list(ci)
        results.append(metrics)

    overall = precision_recall_f1(df["is_correct_bin"].tolist())
    overall["source"] = "all"
    ci = wilson_ci(overall["precision"], overall["n"])
    overall["precision_ci_95"] = list(ci)
    results.append(overall)

    return results


def _load_irr_from_gold_labels(labels_dir: Path, kind: str, overlap_n: int = 20) -> Dict:
    """Load IRR from gold_labels/ (multi-reviewer UI format)."""
    path = labels_dir / "gold_labels" / f"{kind}_labels.csv"
    if not path.exists():
        return {"status": "no_gold_labels"}
    df = pd.read_csv(path)
    if "gold_id" not in df.columns or "labeler_id" not in df.columns or "is_correct" not in df.columns:
        return {"status": "missing_columns"}
    df["is_correct"] = df["is_correct"].astype(str).str.strip().str.lower()
    df = df[df["is_correct"].isin(["yes", "no"])].copy()
    df["is_correct_bin"] = (df["is_correct"] == "yes").astype(int)

    # Overlap: first N unique gold_ids from template
    template_path = labels_dir / ("gold_skills.csv" if kind == "skill" else "gold_knowledge.csv")
    if template_path.exists():
        tpl = pd.read_csv(template_path)
        overlap_ids = tpl["gold_id"].head(overlap_n).tolist() if "gold_id" in tpl.columns else []
    else:
        overlap_ids = df["gold_id"].unique()[:overlap_n].tolist()

    df_overlap = df[df["gold_id"].isin(overlap_ids)].copy()
    labelers = df_overlap["labeler_id"].astype(str).str.strip().unique()
    labelers = [x for x in labelers if x and x != ""]
    if len(labelers) < 2:
        return {"status": "single_labeler", "labeler": str(labelers[0]) if len(labelers) == 1 else "none"}

    pivot = df_overlap.pivot_table(
        index="gold_id", columns="labeler_id", values="is_correct_bin", aggfunc="first"
    ).dropna(how="all")
    valid_labelers = [c for c in labelers if c in pivot.columns]
    if len(valid_labelers) < 2:
        return {"status": "insufficient_overlap", "overlap_items": len(pivot)}
    pivot = pivot[valid_labelers].dropna()
    if len(pivot) < 5:
        return {"status": "insufficient_overlap", "overlap_items": len(pivot)}

    l1, l2 = valid_labelers[0], valid_labelers[1]
    a, b = pivot[l1].values, pivot[l2].values
    agree = (a == b).sum()
    n = len(a)
    po = agree / n
    pe = ((a == 1).sum() / n * (b == 1).sum() / n +
          (a == 0).sum() / n * (b == 0).sum() / n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
    return {
        "status": "ok",
        "labelers": [str(l1), str(l2)],
        "overlap_items": n,
        "observed_agreement": round(po, 4),
        "cohens_kappa": round(kappa, 4),
    }


def evaluate_irr(df: pd.DataFrame, id_col: str = "gold_id") -> Dict:
    """Compute inter-rater reliability if multiple labelers present."""
    if "labeler_id" not in df.columns:
        return {"status": "no_labeler_id"}
    df_filled = df[df["labeler_id"].astype(str).str.strip() != ""].copy()
    labelers = df_filled["labeler_id"].unique()
    if len(labelers) < 2:
        return {"status": "single_labeler", "labeler": str(labelers[0]) if len(labelers) == 1 else "none"}

    pivot = df_filled.pivot_table(
        index=id_col, columns="labeler_id",
        values="is_correct_bin", aggfunc="first",
    ).dropna()

    if len(pivot) < 5:
        return {"status": "insufficient_overlap", "overlap_items": len(pivot)}

    l1, l2 = labelers[0], labelers[1]
    if l1 not in pivot.columns or l2 not in pivot.columns:
        return {"status": "labelers_not_in_overlap"}

    a = pivot[l1].values
    b = pivot[l2].values
    agree = (a == b).sum()
    n = len(a)
    po = agree / n

    pe = ((a == 1).sum() / n * (b == 1).sum() / n +
          (a == 0).sum() / n * (b == 0).sum() / n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0

    return {
        "status": "ok",
        "labelers": [str(l1), str(l2)],
        "overlap_items": n,
        "observed_agreement": round(po, 4),
        "cohens_kappa": round(kappa, 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate extraction quality using gold-set labels."
    )
    parser.add_argument("--output_dir", type=str, default=str(config.OUTPUT_DIR))
    parser.add_argument("--labels_dir", type=str, default=str(LABELS_DIR))
    parser.add_argument("--output", type=str, default="extraction_evaluation_report.json")
    args = parser.parse_args()

    labels = Path(args.labels_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: Dict = {}

    # Prefer merged file (multi-reviewer majority vote); fallback to direct gold_skills.csv
    skills_path = labels / "gold_skills_merged.csv"
    if not skills_path.exists():
        skills_path = labels / "gold_skills.csv"
    skills_df = _load_labeled(skills_path)
    if skills_df.empty:
        print("[WARN] No labeled skills found (gold_skills.csv empty or missing is_correct)")
        report["skills"] = {"status": "no_data"}
    else:
        print(f"[INFO] Loaded {len(skills_df)} labeled skills")
        irr = _load_irr_from_gold_labels(labels, "skill") if (labels / "gold_labels" / "skill_labels.csv").exists() else evaluate_irr(skills_df)
        report["skills"] = {
            "by_source": evaluate_by_source(skills_df),
            "irr": irr,
        }
        overall = [r for r in report["skills"]["by_source"] if r["source"] == "all"]
        if overall:
            o = overall[0]
            print(f"  Skills overall: P={o['precision']:.3f} "
                  f"(95% CI {o['precision_ci_95']}), n={o['n']}")

    knowledge_path = labels / "gold_knowledge_merged.csv"
    if not knowledge_path.exists():
        knowledge_path = labels / "gold_knowledge.csv"
    knowledge_df = _load_labeled(knowledge_path)
    if knowledge_df.empty:
        print("[WARN] No labeled knowledge found")
        report["knowledge"] = {"status": "no_data"}
    else:
        print(f"[INFO] Loaded {len(knowledge_df)} labeled knowledge")
        irr = _load_irr_from_gold_labels(labels, "knowledge") if (labels / "gold_labels" / "knowledge_labels.csv").exists() else evaluate_irr(knowledge_df)
        report["knowledge"] = {
            "by_source": evaluate_by_source(knowledge_df),
            "irr": irr,
        }
        overall = [r for r in report["knowledge"]["by_source"] if r["source"] == "all"]
        if overall:
            o = overall[0]
            print(f"  Knowledge overall: P={o['precision']:.3f} "
                  f"(95% CI {o['precision_ci_95']}), n={o['n']}")

    out_path = out_dir / args.output
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Saved extraction evaluation report to {out_path}")


if __name__ == "__main__":
    main()
