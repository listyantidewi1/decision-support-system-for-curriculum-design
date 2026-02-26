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
from collections import Counter
from pathlib import Path

import config

FEEDBACK_DIR = Path(config.PROJECT_ROOT) / "feedback_store"

QUALITY_MAP = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    "poor": 1, "fair": 3, "good": 4, "excellent": 5,
}


def load_assessments() -> dict:
    path = FEEDBACK_DIR / "competency_assessments.json"
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
    return QUALITY_MAP.get(str(q).lower().strip(), 0)


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

    return {
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
    }


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
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    assessments = load_assessments()
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
