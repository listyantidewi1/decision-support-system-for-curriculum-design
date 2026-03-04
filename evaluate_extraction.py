"""
evaluate_extraction.py

Evaluates extraction quality (precision, recall, F1) using a labeled gold set.
Computes metrics overall and per extraction source (BERT / LLM / Hybrid).

Scientific methods (see SCIENTIFIC_METHODOLOGY.md):
    - Binomial test: H0 precision=0.5 vs H1 precision>0.5
    - Effect sizes: odds ratio, risk difference
    - Wilson score CI for precision
    - Two-proportion z-test for pairwise source comparison
    - Bonferroni correction (per-source); Benjamini-Hochberg (pairwise)

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
from scipy import stats

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


def _binom_test_precision(n_correct: int, n: int, p0: float = 0.5) -> Tuple[float, bool]:
    """Test H0: precision = p0 vs H1: precision > p0. Returns (p_value, significant_at_005)."""
    if n == 0:
        return (1.0, False)
    try:
        bt = stats.binomtest(n_correct, n, p=p0, alternative="greater")
        return (float(bt.pvalue), bool(bt.pvalue < 0.05))
    except Exception:
        return (1.0, False)


def _effect_sizes(precision: float, p0: float = 0.5) -> Dict[str, float]:
    """Odds ratio vs chance and risk difference. For p0=0.5, odds_ratio = p/(1-p)."""
    risk_diff = round(precision - p0, 4)
    if precision <= 0:
        return {"odds_ratio": 0.0, "risk_difference": risk_diff}
    if precision >= 1:
        return {"odds_ratio": 999.0, "risk_difference": risk_diff}
    odds = precision / (1 - precision)
    odds_chance = p0 / (1 - p0) if p0 < 1 else 0
    odds_ratio = odds / odds_chance if odds_chance > 0 else odds
    return {"odds_ratio": round(min(999.0, odds_ratio), 4), "risk_difference": risk_diff}


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


def _add_binomial_and_effect(metrics: Dict, p0: float = 0.5) -> None:
    """Add binomial test, effect sizes, and recall_estimate key (already in metrics)."""
    n_correct = metrics.get("n_correct", 0)
    n = metrics.get("n", 0)
    precision = metrics.get("precision", 0.0)
    p_val, sig = _binom_test_precision(n_correct, n, p0)
    metrics["p_value_vs_chance"] = round(p_val, 6)
    metrics["significant_at_005"] = bool(sig)
    metrics.update(_effect_sizes(precision, p0))


def _two_proportion_z_test(n1: int, x1: int, n2: int, x2: int) -> float:
    """Two-proportion z-test (unpaired). Returns two-tailed p-value."""
    if n1 == 0 or n2 == 0:
        return 1.0
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    if p_pool <= 0 or p_pool >= 1:
        return 1.0
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0 if p1 != p2 else 1.0
    z = (p1 - p2) / se
    # Two-tailed: 2 * (1 - Phi(|z|)); Phi(x) = 0.5*(1+erf(x/sqrt(2)))
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def _benjamini_hochberg(p_values: List[float]) -> List[float]:
    """Benjamini-Hochberg adjusted p-values."""
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    for r, i in enumerate(order, start=1):
        adjusted[i] = min(1.0, p_values[i] * n / r)
    return adjusted


def _pairwise_comparisons(results: List[Dict]) -> List[Dict]:
    """Pairwise two-proportion z-tests between sources; BH correction."""
    by_source = {r["source"]: r for r in results if r.get("source") != "all"}
    sources = list(by_source.keys())
    if len(sources) < 2:
        return []

    pairs = []
    p_vals = []
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            a, b = sources[i], sources[j]
            ra, rb = by_source[a], by_source[b]
            n1, x1 = ra["n"], ra["n_correct"]
            n2, x2 = rb["n"], rb["n_correct"]
            p = _two_proportion_z_test(n1, x1, n2, x2)
            odds_a = (ra["precision"] / (1 - ra["precision"])) if ra["precision"] < 1 else 999
            odds_b = (rb["precision"] / (1 - rb["precision"])) if rb["precision"] < 1 else 999
            odds_ratio = odds_a / odds_b if odds_b > 0 else 0
            pairs.append({"A": a, "B": b, "p_value": p, "odds_ratio": round(odds_ratio, 4)})
            p_vals.append(p)

    adj = _benjamini_hochberg(p_vals)
    for i, p in enumerate(pairs):
        p["p_value"] = round(p["p_value"], 6)
        p["p_adjusted"] = round(adj[i], 6)
        p["significant_after_correction"] = bool(adj[i] < 0.05)
    return pairs


def _apply_multi_comparison_correction(results: List[Dict], alpha: float = 0.05) -> None:
    """Apply Bonferroni correction: alpha_adjusted = alpha / k for k tests vs chance."""
    by_source = [r for r in results if r.get("source") != "all"]
    if not by_source:
        return
    k = len(by_source)
    alpha_adj = alpha / k
    for r in by_source:
        p = r.get("p_value_vs_chance", 1.0)
        r["p_value_adjusted"] = round(min(1.0, p * k), 6)  # Bonferroni
        r["significant_after_correction"] = bool(p < alpha_adj)


def evaluate_by_source(df: pd.DataFrame, source_col: str = "source") -> List[Dict]:
    results = []
    if source_col not in df.columns:
        metrics = precision_recall_f1(df["is_correct_bin"].tolist())
        metrics["source"] = "all"
        ci = wilson_ci(metrics["precision"], metrics["n"])
        metrics["precision_ci_95"] = list(ci)
        _add_binomial_and_effect(metrics)
        results.append(metrics)
        return results

    for src, grp in df.groupby(source_col):
        metrics = precision_recall_f1(grp["is_correct_bin"].tolist())
        metrics["source"] = str(src)
        ci = wilson_ci(metrics["precision"], metrics["n"])
        metrics["precision_ci_95"] = list(ci)
        _add_binomial_and_effect(metrics)
        results.append(metrics)

    overall = precision_recall_f1(df["is_correct_bin"].tolist())
    overall["source"] = "all"
    ci = wilson_ci(overall["precision"], overall["n"])
    overall["precision_ci_95"] = list(ci)
    _add_binomial_and_effect(overall)
    results.append(overall)

    _apply_multi_comparison_correction(results)
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
        by_src = evaluate_by_source(skills_df)
        report["skills"] = {
            "by_source": by_src,
            "pairwise_comparisons": _pairwise_comparisons(by_src),
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
        by_src = evaluate_by_source(knowledge_df)
        report["knowledge"] = {
            "by_source": by_src,
            "pairwise_comparisons": _pairwise_comparisons(by_src),
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
