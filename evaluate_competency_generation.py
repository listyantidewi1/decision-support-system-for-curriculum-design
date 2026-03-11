"""
evaluate_competency_generation.py

Evaluates competency generation quality using human assessments from
feedback_store/competency_assessments.json.

Metrics:
    - Distribution stats (mean, median, std, quartiles, min, max)
    - Per-batch breakdown
    - 95 % confidence interval for mean quality (t-distribution)
    - Relevance breakdown
    - Notes analysis (count, common words)

Inputs:
    - config.OUTPUT_DIR/competency_proposals.json
    - feedback_store/competency_assessments.json

Outputs:
    - competency_evaluation_report.json
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import config
from scipy import stats

DEFAULT_FEEDBACK_DIR = Path(config.PROJECT_ROOT) / "feedback_store"

QUALITY_MAP = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    "poor": 1, "fair": 3, "good": 4, "excellent": 5,
}


def load_assessments(feedback_dir: Path) -> dict:
    path = feedback_dir / "competency_assessments.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_proposals(output_dir: Path) -> list:
    path = output_dir / "competency_proposals.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("competencies", [])


def _quality_to_score(q) -> float:
    s = str(q).lower().strip()
    if s in QUALITY_MAP:
        return QUALITY_MAP[s]
    # Handle float strings like "5.0"
    try:
        v = float(s)
        if 1 <= v <= 5:
            return int(round(v))
    except ValueError:
        pass
    return 0


def _discretize_quality(score: float) -> str:
    """Map quality score to category for IRR: 1-2=low, 3=mid, 4-5=high."""
    if score <= 0:
        return None
    if score <= 2:
        return "low"
    if score <= 3:
        return "mid"
    return "high"


def _get_quality_keys_from_assessment(a: dict) -> list[str]:
    """Extract quality-related keys: 'quality' or 'quality_<name>' (e.g. quality_alice)."""
    keys = []
    for k in a:
        if k == "quality":
            keys.append(k)
        elif re.match(r"quality_[a-zA-Z0-9_]+", k):
            keys.append(k)
    return keys


def _cohens_kappa(labels1: list, labels2: list) -> float | None:
    """Compute Cohen's Kappa for two rater label sequences. Returns None if invalid."""
    if len(labels1) != len(labels2) or len(labels1) == 0:
        return None
    n = len(labels1)
    # p_o = observed agreement
    agreements = sum(1 for a, b in zip(labels1, labels2) if a == b)
    p_o = agreements / n
    categories = sorted(set(labels1) | set(labels2))
    # p_e = expected agreement by chance
    c1, c2 = Counter(labels1), Counter(labels2)
    p_e = sum(c1.get(c, 0) / n * c2.get(c, 0) / n for c in categories)
    if p_e >= 1:
        return 1.0 if p_o >= 1 else None  # perfect agreement when single category
    return (p_o - p_e) / (1 - p_e)


def _t_critical_95(n: int) -> float:
    """Approximate t-critical value for 95 % CI (two-tailed)."""
    if n <= 1:
        return float("inf")
    lookup = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
              7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 15: 2.145,
              20: 2.093, 30: 2.045, 50: 2.009, 100: 1.984}
    if n in lookup:
        return lookup[n]
    for k in sorted(lookup):
        if k >= n:
            return lookup[k]
    return 1.96


def evaluate(assessments: dict, proposals: list) -> dict:
    if not assessments:
        return {
            "total_assessed": 0,
            "mean_quality": 0.0, "median_quality": 0.0,
            "std_quality": 0.0, "min_quality": 0, "max_quality": 0,
            "q1_quality": 0.0, "q3_quality": 0.0,
            "ci_95_lower": 0.0, "ci_95_upper": 0.0,
            "pct_quality_gte_good": 0.0,
            "pct_relevant_yes": 0.0,
            "pct_relevant_partial": 0.0,
            "pct_relevant_no": 0.0,
            "per_batch": [],
            "notes_count": 0,
            "notes_common_words": [],
            "sign_test_pvalue": None,
            "irr_kappa": None,
            "irr_n_overlap": 0,
            "irr_note": "no assessments",
            "kruskal_wallis_h": None,
            "kruskal_wallis_p": None,
            "n_batches": 0,
        }

    scores = []
    relevant_yes = relevant_partial = relevant_no = 0
    quality_gte_4 = 0
    batch_scores: dict = {}
    all_notes = []

    for cid, a in assessments.items():
        q = a.get("quality", "")
        r = str(a.get("relevant", "")).lower().strip()
        notes = str(a.get("notes", "")).strip()
        score = _quality_to_score(q)

        if score > 0:
            scores.append(score)
            if score >= 4:
                quality_gte_4 += 1

        if r == "yes":
            relevant_yes += 1
        elif r == "partial":
            relevant_partial += 1
        elif r == "no":
            relevant_no += 1

        if notes:
            all_notes.append(notes)

        bid = a.get("batch_id") or _find_batch(cid, proposals)
        if bid is not None:
            batch_scores.setdefault(bid, [])
            if score > 0:
                batch_scores[bid].append(score)

    n = len(assessments)
    scores_sorted = sorted(scores)

    if scores:
        mean_q = sum(scores) / len(scores)
        median_q = _median(scores_sorted)
        std_q = _std(scores, mean_q)
        q1 = _percentile(scores_sorted, 25)
        q3 = _percentile(scores_sorted, 75)
        t_crit = _t_critical_95(len(scores))
        margin = t_crit * std_q / math.sqrt(len(scores)) if len(scores) > 1 else 0
        ci_lo = mean_q - margin
        ci_hi = mean_q + margin
    else:
        mean_q = median_q = std_q = q1 = q3 = ci_lo = ci_hi = 0.0

    per_batch = []
    for bid in sorted(batch_scores):
        bs = batch_scores[bid]
        per_batch.append({
            "batch_id": bid,
            "n": len(bs),
            "mean_quality": round(sum(bs) / len(bs), 2) if bs else 0,
        })

    word_counter = Counter()
    stop = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "and",
            "in", "for", "on", "it", "this", "that", "with", "as", "but", "or"}
    for note in all_notes:
        for w in note.lower().split():
            w = w.strip(".,;:!?\"'()-")
            if len(w) > 2 and w not in stop:
                word_counter[w] += 1

    # --- Sign test: H0 median=3 vs H1 median>3 (binom: count>3, n=non-3 scores, p=0.5)
    non_3_scores = [s for s in scores if s != 3]
    count_gt_3 = sum(1 for s in scores if s > 3)
    n_non_3 = len(non_3_scores)
    if n_non_3 > 0:
        sign_result = stats.binomtest(count_gt_3, n_non_3, p=0.5, alternative="greater")
        sign_test_pvalue = round(sign_result.pvalue, 4)
    else:
        sign_test_pvalue = None  # all scores are 3, test undefined

    # --- Inter-rater reliability (Cohen's Kappa)
    irr_kappa = None
    irr_n_overlap = 0
    irr_note = None
    # Collect quality ratings per competency, keyed by rater
    rater_pairs_data: dict[str, list[tuple[str, float]]] = {}  # cid -> [(rater_key, score), ...]
    for cid, a in assessments.items():
        # Handle assessment as dict or list (e.g. multiple reviewers per cid)
        if isinstance(a, list):
            for item in a:
                if isinstance(item, dict):
                    qk_keys = _get_quality_keys_from_assessment(item)
                    if qk_keys:
                        for qk in qk_keys:
                            score = _quality_to_score(item.get(qk))
                            if score > 0:
                                rater_pairs_data.setdefault(cid, []).append((qk, score))
                    elif item.get("reviewer_id") is not None:
                        score = _quality_to_score(item.get("quality"))
                        if score > 0:
                            rater_pairs_data.setdefault(cid, []).append(
                                (f"reviewer_{item['reviewer_id']}", score)
                        )
        elif isinstance(a, dict):
            qk_keys = _get_quality_keys_from_assessment(a)
            if len(qk_keys) >= 2:
                for qk in qk_keys:
                    score = _quality_to_score(a.get(qk))
                    if score > 0:
                        rater_pairs_data.setdefault(cid, []).append((qk, score))
            elif a.get("reviewer_id") is not None:
                score = _quality_to_score(a.get("quality"))
                if score > 0:
                    rater_pairs_data.setdefault(cid, []).append(
                        (f"reviewer_{a['reviewer_id']}", score))

    # Build rater pairs: need at least 2 raters with overlap
    if rater_pairs_data:
        # Get unique rater ids
        all_raters = set()
        for pairs in rater_pairs_data.values():
            for rk, _ in pairs:
                all_raters.add(rk)
        # For quality_X/quality_Y in same assessment: each cid has 2+ ratings
        # For reviewer_id across assessments: need to group by base cid
        cids_with_multi = [cid for cid, pairs in rater_pairs_data.items() if len(pairs) >= 2]
        if len(all_raters) >= 2 and cids_with_multi:
            # Use first two raters found; build (rater1_labels, rater2_labels) for overlapping cids
            rater_keys = sorted(all_raters)[:2]
            r1_labels, r2_labels = [], []
            for cid in cids_with_multi:
                pairs = rater_pairs_data[cid]
                by_rater = {rk: s for rk, s in pairs}
                if rater_keys[0] in by_rater and rater_keys[1] in by_rater:
                    d1 = _discretize_quality(by_rater[rater_keys[0]])
                    d2 = _discretize_quality(by_rater[rater_keys[1]])
                    if d1 is not None and d2 is not None:
                        r1_labels.append(d1)
                        r2_labels.append(d2)
            if len(r1_labels) >= 2:
                kappa = _cohens_kappa(r1_labels, r2_labels)
                if kappa is not None:
                    irr_kappa = round(kappa, 4)
                    irr_n_overlap = len(r1_labels)
        else:
            irr_note = "single reviewer"
    else:
        irr_note = "single reviewer"

    # --- Kruskal-Wallis across batches
    kruskal_wallis_h = None
    kruskal_wallis_p = None
    n_batches = len(batch_scores)
    if n_batches >= 2:
        batch_lists = [batch_scores[bid] for bid in sorted(batch_scores) if batch_scores[bid]]
        if len(batch_lists) >= 2:
            kw_result = stats.kruskal(*batch_lists)
            kruskal_wallis_h = round(kw_result.statistic, 4)
            kruskal_wallis_p = round(kw_result.pvalue, 4)

    report = {
        "total_assessed": n,
        "mean_quality": round(mean_q, 3),
        "median_quality": round(median_q, 1),
        "std_quality": round(std_q, 3),
        "min_quality": min(scores) if scores else 0,
        "max_quality": max(scores) if scores else 0,
        "q1_quality": round(q1, 1),
        "q3_quality": round(q3, 1),
        "ci_95_lower": round(ci_lo, 3),
        "ci_95_upper": round(ci_hi, 3),
        "pct_quality_gte_good": round(quality_gte_4 / n * 100, 1) if n else 0.0,
        "pct_relevant_yes": round(relevant_yes / n * 100, 1) if n else 0.0,
        "pct_relevant_partial": round(relevant_partial / n * 100, 1) if n else 0.0,
        "pct_relevant_no": round(relevant_no / n * 100, 1) if n else 0.0,
        "per_batch": per_batch,
        "notes_count": len(all_notes),
        "notes_common_words": word_counter.most_common(15),
        "sign_test_pvalue": sign_test_pvalue,
        "irr_kappa": irr_kappa,
        "irr_n_overlap": irr_n_overlap,
        "irr_note": irr_note,
        "kruskal_wallis_h": kruskal_wallis_h,
        "kruskal_wallis_p": kruskal_wallis_p,
        "n_batches": n_batches,
    }
    return report


def _find_batch(cid: str, proposals: list):
    for c in proposals:
        key = f"{c.get('id', '')}_b{c.get('batch_id', 0)}"
        if key == cid:
            return c.get("batch_id")
    return None


def _median(sorted_vals):
    n = len(sorted_vals)
    if n == 0:
        return 0
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    return sorted_vals[mid]


def _percentile(sorted_vals, pct):
    if not sorted_vals:
        return 0
    k = (len(sorted_vals) - 1) * pct / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_vals) else f
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def _std(vals, mean):
    if len(vals) < 2:
        return 0.0
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate competency generation using human assessments."
    )
    parser.add_argument("--output_dir", type=str, default=str(config.OUTPUT_DIR))
    parser.add_argument("--output", type=str, default="competency_evaluation_report.json")
    parser.add_argument(
        "--feedback_dir",
        type=str,
        default=None,
        help="Feedback store directory (default: PROJECT_ROOT/feedback_store)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    feedback_dir = Path(args.feedback_dir) if args.feedback_dir else DEFAULT_FEEDBACK_DIR
    assessments = load_assessments(feedback_dir)
    proposals = load_proposals(out_dir)

    if not assessments:
        print("[WARN] No competency assessments found.")

    metrics = evaluate(assessments, proposals)

    if metrics["total_assessed"] > 0:
        print(f"[INFO] Assessed {metrics['total_assessed']} competencies")
        print(f"       Mean quality: {metrics['mean_quality']:.2f} "
              f"(95% CI: [{metrics['ci_95_lower']:.2f}, {metrics['ci_95_upper']:.2f}])")
        print(f"       Median: {metrics['median_quality']}, "
              f"Std: {metrics['std_quality']:.2f}, "
              f"Q1-Q3: [{metrics['q1_quality']}, {metrics['q3_quality']}]")
        print(f"       % quality >= good: {metrics['pct_quality_gte_good']:.1f}%")
        print(f"       % relevant=yes: {metrics['pct_relevant_yes']:.1f}%")
        if metrics["per_batch"]:
            print(f"       Per-batch: {len(metrics['per_batch'])} batches")

    out_path = out_dir / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[INFO] Saved report to {out_path}")


if __name__ == "__main__":
    main()
