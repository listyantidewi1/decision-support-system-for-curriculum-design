"""
Build a scientific validation pack for a completed competency-v2 run.

The pack is designed for Sprint 5 of the research pipeline:

* reviewer-ready templates for competency, recommendation, cluster, and DACUM
  validation;
* gold-label templates for sentence relevance and Skill-LLM extraction spans;
* automatic metric computation when reviewers fill those templates;
* a pass/fail report that keeps official-readiness claims blocked until the
  required evidence exists.

No LLM calls are made here. The script only samples existing evidence and
computes deterministic metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_THRESHOLDS = {
    "min_reviewed_competencies": 20,
    "min_reviewed_recommendations": 20,
    "min_sentence_gold_rows": 100,
    "min_extraction_gold_rows": 100,
    "min_dacum_review_rows": 20,
    "competency_validity_mean_min": 4.0,
    "competency_actionability_mean_min": 4.0,
    "competency_hallucination_rate_max": 0.05,
    "recommendation_acceptance_rate_min": 0.80,
    "recommendation_usefulness_mean_min": 4.0,
    "sentence_relevance_f1_min": 0.85,
    "extraction_precision_min": 0.90,
    "extraction_type_accuracy_min": 0.90,
    "dacum_occupation_valid_rate_min": 0.85,
    "dacum_duty_revision_rate_max": 0.20,
}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_csv_template(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    overwrite: bool,
) -> str:
    if path.exists() and not overwrite:
        return "preserved"
    _write_csv(path, rows, fieldnames)
    return "written"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        out = float(value)
        if math.isnan(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def _rate(values: Iterable[bool | None]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return round(sum(1 for v in vals if v) / len(vals), 4)


def _yes_no(value: Any, *, partial_as_true: bool = False) -> bool | None:
    raw = str(value or "").strip().lower()
    if raw in {"yes", "y", "true", "1", "valid", "relevant", "accepted"}:
        return True
    if partial_as_true and raw in {"partial", "partially", "partly", "maybe"}:
        return True
    if raw in {"no", "n", "false", "0", "invalid", "irrelevant", "rejected"}:
        return False
    return None


def _compact(text: Any, max_chars: int = 320) -> str:
    raw = " ".join(str(text or "").split())
    if len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 3].rstrip() + "..."


def _run_tag(run_dir: Path, pipeline_report: dict[str, Any]) -> str:
    if pipeline_report.get("tag"):
        return str(pipeline_report["tag"])
    prefix = "competency_v2_pipeline_"
    return run_dir.name[len(prefix):] if run_dir.name.startswith(prefix) else run_dir.name


def _ranked_competencies(competencies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if any(c.get("class_rank") for c in competencies):
        return sorted(
            competencies,
            key=lambda c: (
                int(c.get("class_sort_group") if c.get("class_sort_group") is not None else 99),
                int(c.get("class_rank") or 999999),
                -float(c.get("job_prevalence") or 0),
                -int(c.get("n_source_jobs") or len(_as_list(c.get("source_job_ids")))),
                -float(c.get("grounding_score") or 0),
                str(c.get("id", "")),
            ),
        )
    return sorted(competencies, key=lambda c: (-float(c.get("priority_score") or 0), str(c.get("id", ""))))


def _binary_metrics(
    rows: list[dict[str, Any]],
    pred_col: str,
    gold_col: str,
) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    reviewed = 0
    for row in rows:
        pred = _yes_no(row.get(pred_col))
        gold = _yes_no(row.get(gold_col), partial_as_true=True)
        if pred is None or gold is None:
            continue
        reviewed += 1
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif not pred and not gold:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    accuracy = (tp + tn) / reviewed if reviewed else None
    return {
        "n_reviewed": reviewed,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
    }


def load_sentence_pool(phase1_dir: Path) -> list[dict[str, str]]:
    rows = _read_csv(phase1_dir / "job_sentences_mapping.csv")
    if rows:
        return rows
    pool: dict[str, dict[str, str]] = {}
    for filename in ["advanced_skills.csv", "advanced_knowledge.csv"]:
        for row in _read_csv(phase1_dir / filename):
            sid = row.get("sentence_id") or ""
            if sid and sid not in pool:
                pool[sid] = {
                    "sentence_id": sid,
                    "job_id": row.get("job_id", ""),
                    "sentence_text": row.get("sentence_text", ""),
                }
    return list(pool.values())


def build_sentence_relevance_template(
    competencies: list[dict[str, Any]],
    phase1_dir: Path,
    per_competency_evidence: int,
    per_competency_contrast: int,
) -> list[dict[str, Any]]:
    ranked = _ranked_competencies(competencies)
    sentence_pool = load_sentence_pool(phase1_dir)
    all_pool = sorted(sentence_pool, key=lambda r: (r.get("sentence_id", ""), r.get("sentence_text", "")))
    sentence_id_lookup = {r.get("sentence_id", ""): r for r in all_pool}

    rows: list[dict[str, Any]] = []
    for comp in ranked:
        comp_id = str(comp.get("id", ""))
        source_sentences = [str(s) for s in _as_list(comp.get("source_sentences")) if str(s).strip()]
        source_sentence_ids = [str(s) for s in _as_list(comp.get("source_sentence_ids")) if str(s).strip()]
        source_jobs = {str(j) for j in _as_list(comp.get("source_job_ids")) if str(j).strip()}
        for idx, sentence in enumerate(source_sentences[:per_competency_evidence], start=1):
            sid = source_sentence_ids[idx - 1] if idx - 1 < len(source_sentence_ids) else ""
            sentence_meta = sentence_id_lookup.get(sid, {})
            rows.append({
                "sample_id": f"{comp_id}_evidence_{idx:02d}",
                "sample_type": "system_evidence",
                "competency_id": comp_id,
                "statement": comp.get("statement", ""),
                "sentence_id": sid,
                "job_id": sentence_meta.get("job_id", ""),
                "sentence_text": sentence,
                "system_predicted_relevant_yes_no": "yes",
                "system_sentence_relevance_score": comp.get("sentence_relevance_score", ""),
                "gold_relevant_yes_no": "",
                "gold_relevance_1_5": "",
                "gold_required_revision": "",
                "reviewer_id": "",
                "comments": "",
            })

        contrast_candidates = [r for r in all_pool if r.get("job_id") not in source_jobs]
        if not contrast_candidates:
            contrast_candidates = all_pool
        stable_hash = int(hashlib.sha1(comp_id.encode("utf-8")).hexdigest()[:12], 16) if comp_id else 0
        offset = stable_hash % len(contrast_candidates) if contrast_candidates else 0
        for idx in range(min(per_competency_contrast, len(contrast_candidates))):
            row = contrast_candidates[(offset + idx) % len(contrast_candidates)]
            rows.append({
                "sample_id": f"{comp_id}_contrast_{idx + 1:02d}",
                "sample_type": "contrast_sentence",
                "competency_id": comp_id,
                "statement": comp.get("statement", ""),
                "sentence_id": row.get("sentence_id", ""),
                "job_id": row.get("job_id", ""),
                "sentence_text": row.get("sentence_text", ""),
                "system_predicted_relevant_yes_no": "no",
                "system_sentence_relevance_score": "",
                "gold_relevant_yes_no": "",
                "gold_relevance_1_5": "",
                "gold_required_revision": "",
                "reviewer_id": "",
                "comments": "",
            })
    return rows


def _phase1_extraction_rows(phase1_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for filename, text_col, extraction_type in [
        ("advanced_skills.csv", "skill", "skill"),
        ("advanced_knowledge.csv", "knowledge", "knowledge"),
    ]:
        for row in _read_csv(phase1_dir / filename):
            extracted = row.get(text_col, "")
            if not extracted:
                continue
            rows.append({
                "source_file": filename,
                "job_id": row.get("job_id", ""),
                "sentence_id": row.get("sentence_id", ""),
                "sentence_text": row.get("sentence_text", ""),
                "extracted_text": extracted,
                "extraction_type": extraction_type,
                "confidence_score": row.get("confidence_score", ""),
                "confidence_tier": row.get("confidence_tier", ""),
                "source": row.get("source", ""),
            })
    return rows


def build_extraction_gold_template(phase1_dir: Path, sample_size: int, seed: int) -> list[dict[str, Any]]:
    extraction_rows = _phase1_extraction_rows(phase1_dir)
    extraction_rows = sorted(
        extraction_rows,
        key=lambda r: (
            str(r.get("job_id", "")),
            str(r.get("sentence_id", "")),
            str(r.get("extraction_type", "")),
            str(r.get("extracted_text", "")),
        ),
    )
    if sample_size > 0 and len(extraction_rows) > sample_size:
        rng = random.Random(seed)
        extraction_rows = [extraction_rows[i] for i in sorted(rng.sample(range(len(extraction_rows)), sample_size))]

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(extraction_rows, start=1):
        rows.append({
            "sample_id": f"extract_{idx:05d}",
            **row,
            "system_predicted_extraction_yes_no": "yes",
            "gold_valid_extraction_yes_no": "",
            "gold_correct_type_yes_no": "",
            "gold_normalized_label": "",
            "gold_missing_spans_in_sentence": "",
            "reviewer_id": "",
            "comments": "",
        })
    return rows


def copy_review_templates(evidence_dir: Path, dacum_dir: Path, out_dir: Path, overwrite: bool) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    template_dir = out_dir / "reviewer_templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    for filename in [
        "expert_review_competencies.csv",
        "expert_review_clusters.csv",
        "expert_review_recommendations.csv",
    ]:
        src = evidence_dir / filename
        dest = template_dir / filename
        if src.exists():
            if dest.exists() and not overwrite:
                copied.append({"path": str(dest), "status": "preserved"})
            else:
                shutil.copy2(src, dest)
                copied.append({"path": str(dest), "status": "written"})
    dacum_src = dacum_dir / "dacum_expert_worksheet.csv"
    dacum_dest = template_dir / "dacum_expert_worksheet.csv"
    if dacum_src.exists():
        if dacum_dest.exists() and not overwrite:
            copied.append({"path": str(dacum_dest), "status": "preserved"})
        else:
            shutil.copy2(dacum_src, dacum_dest)
            copied.append({"path": str(dacum_dest), "status": "written"})
    return copied


def compute_competency_review_metrics(rows: list[dict[str, str]], thresholds: dict[str, float]) -> dict[str, Any]:
    reviewed = [
        r for r in rows
        if any(str(r.get(c, "")).strip() for c in [
            "validity_1_5", "relevance_yes_partial_no", "actionability_1_5", "hallucination_concern_yes_no"
        ])
    ]
    hallucination = [_yes_no(r.get("hallucination_concern_yes_no")) for r in reviewed]
    relevance_yes = [_yes_no(r.get("relevance_yes_partial_no")) for r in reviewed]
    relevance_yes_partial = [_yes_no(r.get("relevance_yes_partial_no"), partial_as_true=True) for r in reviewed]
    metrics = {
        "n_rows": len(rows),
        "n_reviewed": len(reviewed),
        "n_unique_competencies_reviewed": len({r.get("competency_id") for r in reviewed if r.get("competency_id")}),
        "validity_mean": _mean(_as_float(r.get("validity_1_5")) for r in reviewed),
        "grounding_clarity_mean": _mean(_as_float(r.get("grounding_clarity_1_5")) for r in reviewed),
        "rationale_clarity_mean": _mean(_as_float(r.get("rationale_clarity_1_5")) for r in reviewed),
        "actionability_mean": _mean(_as_float(r.get("actionability_1_5")) for r in reviewed),
        "relevance_yes_rate": _rate(relevance_yes),
        "relevance_yes_or_partial_rate": _rate(relevance_yes_partial),
        "hallucination_concern_rate": _rate(hallucination),
    }
    metrics["status"] = _metric_status(
        observed_n=metrics["n_reviewed"],
        min_n=int(thresholds["min_reviewed_competencies"]),
        checks=[
            (metrics["validity_mean"], ">=", thresholds["competency_validity_mean_min"], "validity_mean"),
            (metrics["actionability_mean"], ">=", thresholds["competency_actionability_mean_min"], "actionability_mean"),
            (metrics["hallucination_concern_rate"], "<=", thresholds["competency_hallucination_rate_max"], "hallucination_concern_rate"),
        ],
    )
    return metrics


def compute_recommendation_review_metrics(rows: list[dict[str, str]], thresholds: dict[str, float], top_n: int) -> dict[str, Any]:
    top_rows = [
        r for r in rows
        if int(str(r.get("rank") or "999999").split(".")[0]) <= top_n
    ]
    reviewed = [
        r for r in top_rows
        if any(str(r.get(c, "")).strip() for c in [
            "priority_label_yes_partial_no", "curriculum_usefulness_1_5", "ranking_reasonableness_1_5"
        ])
    ]
    accepted = [_yes_no(r.get("priority_label_yes_partial_no"), partial_as_true=True) for r in reviewed]
    metrics = {
        "top_n": top_n,
        "n_top_rows": len(top_rows),
        "n_reviewed": len(reviewed),
        "priority_acceptance_yes_or_partial_rate": _rate(accepted),
        "curriculum_usefulness_mean": _mean(_as_float(r.get("curriculum_usefulness_1_5")) for r in reviewed),
        "ranking_reasonableness_mean": _mean(_as_float(r.get("ranking_reasonableness_1_5")) for r in reviewed),
    }
    metrics["status"] = _metric_status(
        observed_n=metrics["n_reviewed"],
        min_n=int(thresholds["min_reviewed_recommendations"]),
        checks=[
            (
                metrics["priority_acceptance_yes_or_partial_rate"],
                ">=",
                thresholds["recommendation_acceptance_rate_min"],
                "priority_acceptance_yes_or_partial_rate",
            ),
            (
                metrics["curriculum_usefulness_mean"],
                ">=",
                thresholds["recommendation_usefulness_mean_min"],
                "curriculum_usefulness_mean",
            ),
        ],
    )
    return metrics


def compute_sentence_relevance_metrics(rows: list[dict[str, str]], thresholds: dict[str, float]) -> dict[str, Any]:
    metrics = _binary_metrics(rows, "system_predicted_relevant_yes_no", "gold_relevant_yes_no")
    metrics["status"] = _metric_status(
        observed_n=metrics["n_reviewed"],
        min_n=int(thresholds["min_sentence_gold_rows"]),
        checks=[(metrics["f1"], ">=", thresholds["sentence_relevance_f1_min"], "f1")],
    )
    return metrics


def compute_extraction_metrics(rows: list[dict[str, str]], thresholds: dict[str, float]) -> dict[str, Any]:
    reviewed = [r for r in rows if _yes_no(r.get("gold_valid_extraction_yes_no")) is not None]
    valid_flags = [_yes_no(r.get("gold_valid_extraction_yes_no")) for r in reviewed]
    type_rows = [r for r in reviewed if _yes_no(r.get("gold_correct_type_yes_no")) is not None]
    missing_total = 0
    for row in reviewed:
        missing = _as_float(row.get("gold_missing_spans_in_sentence"))
        if missing:
            missing_total += int(missing)
    tp = sum(1 for x in valid_flags if x)
    fp = sum(1 for x in valid_flags if x is False)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + missing_total) if (tp + missing_total) else None
    type_accuracy = _rate(_yes_no(r.get("gold_correct_type_yes_no")) for r in type_rows)
    metrics = {
        "n_rows": len(rows),
        "n_reviewed": len(reviewed),
        "true_positive_extractions": tp,
        "false_positive_extractions": fp,
        "gold_missing_spans_total": missing_total,
        "precision": round(precision, 4) if precision is not None else None,
        "recall_with_missing_span_counts": round(recall, 4) if recall is not None else None,
        "type_accuracy": type_accuracy,
    }
    metrics["status"] = _metric_status(
        observed_n=metrics["n_reviewed"],
        min_n=int(thresholds["min_extraction_gold_rows"]),
        checks=[
            (metrics["precision"], ">=", thresholds["extraction_precision_min"], "precision"),
            (metrics["type_accuracy"], ">=", thresholds["extraction_type_accuracy_min"], "type_accuracy"),
        ],
    )
    return metrics


def compute_dacum_metrics(rows: list[dict[str, str]], thresholds: dict[str, float]) -> dict[str, Any]:
    reviewed = [
        r for r in rows
        if any(str(r.get(c, "")).strip() for c in [
            "expert_occupation_match_valid_yes_partial_no",
            "expert_duty_area_revision",
            "expert_skkni_unit_selection",
            "expert_priority_1_5",
        ])
    ]
    occupation_valid = [
        _yes_no(r.get("expert_occupation_match_valid_yes_partial_no"), partial_as_true=True)
        for r in reviewed
    ]
    duty_revision_flags = [bool(str(r.get("expert_duty_area_revision", "")).strip()) for r in reviewed]
    metrics = {
        "n_rows": len(rows),
        "n_reviewed": len(reviewed),
        "occupation_valid_yes_or_partial_rate": _rate(occupation_valid),
        "duty_area_revision_rate": _rate(duty_revision_flags),
        "skkni_selection_rate": _rate(bool(str(r.get("expert_skkni_unit_selection", "")).strip()) for r in reviewed),
        "expert_priority_mean": _mean(_as_float(r.get("expert_priority_1_5")) for r in reviewed),
    }
    metrics["status"] = _metric_status(
        observed_n=metrics["n_reviewed"],
        min_n=int(thresholds["min_dacum_review_rows"]),
        checks=[
            (
                metrics["occupation_valid_yes_or_partial_rate"],
                ">=",
                thresholds["dacum_occupation_valid_rate_min"],
                "occupation_valid_yes_or_partial_rate",
            ),
            (
                metrics["duty_area_revision_rate"],
                "<=",
                thresholds["dacum_duty_revision_rate_max"],
                "duty_area_revision_rate",
            ),
        ],
    )
    return metrics


def _metric_status(observed_n: int, min_n: int, checks: list[tuple[Any, str, float, str]]) -> str:
    if observed_n <= 0:
        return "not_measured"
    if observed_n < min_n:
        return "insufficient_sample"
    for observed, operator, threshold, _name in checks:
        if observed is None:
            return "insufficient_labels"
        if operator == ">=" and observed < threshold:
            return "fail"
        if operator == "<=" and observed > threshold:
            return "fail"
    return "pass"


def build_official_readiness_report(
    pipeline_report: dict[str, Any],
    metrics: dict[str, Any],
    thresholds: dict[str, float],
    stability_report: dict[str, Any],
) -> dict[str, Any]:
    stages = pipeline_report.get("stages", {}) if isinstance(pipeline_report, dict) else {}
    corpus_status = (
        (stages.get("corpus_lineage_audit", {}) or {}).get("corpus_validity_status")
        or (stages.get("scientific_sprint1", {}) or {}).get("corpus_validity_status")
        or "needs_manual_review"
    )
    stability_status = stability_report.get("official_stability_gate_status") or "not_measured"
    gate_statuses = {
        "corpus_valid_for_indonesian_claim": "pass" if corpus_status == "valid_for_indonesian_claim" else "blocked",
        "competency_expert_review": metrics["competency_review"]["status"],
        "recommendation_expert_review": metrics["recommendation_review"]["status"],
        "sentence_relevance_gold": metrics["sentence_relevance_gold"]["status"],
        "extraction_quality_gold": metrics["extraction_quality_gold"]["status"],
        "dacum_cbc_expert_review": metrics["dacum_cbc_review"]["status"],
        "stability_validation": stability_status,
    }
    blockers = [name for name, status in gate_statuses.items() if status != "pass"]
    return {
        "schema_version": "official_readiness_v1",
        "thresholds": thresholds,
        "corpus_validity_status": corpus_status,
        "stability_ablation_report": stability_report or {},
        "gate_statuses": gate_statuses,
        "official_ready": not blockers,
        "blockers": blockers,
        "interpretation": (
            "Official-ready requires every gate to pass. Engineering or US-corpus runs "
            "must remain exploratory/review-candidate even when local validation metrics are strong."
        ),
    }


def write_protocol(out_dir: Path, thresholds: dict[str, float]) -> None:
    protocol = {
        "schema_version": "scientific_validation_protocol_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": thresholds,
        "label_values": {
            "yes_no": ["yes", "no"],
            "yes_partial_no": ["yes", "partial", "no"],
            "rating_1_5": [1, 2, 3, 4, 5],
        },
        "review_roles": [
            "domain_expert_curriculum",
            "industry_practitioner",
            "assessment_or_quality_assurance_reviewer",
        ],
        "minimum_scientific_claim": [
            "Corpus lineage must match the claim context.",
            "Expert reviewers must validate competency validity, relevance, and actionability.",
            "Gold labels must estimate sentence relevance and extraction precision.",
            "DACUM/CBC reviewers must validate occupation, duty, task, SKKNI, and Spektrum alignment.",
            "Stability experiments must pass before official-ready status is allowed.",
        ],
    }
    _write_json(out_dir / "validation_protocol.json", protocol)


def write_readme(out_dir: Path, run_tag: str, run_dir: Path, metrics: dict[str, Any], readiness: dict[str, Any]) -> None:
    text = f"""# Scientific Validation Pack: {run_tag}

Generated: {metrics.get('created_at')}

Source run:

```text
{run_dir}
```

## Purpose

This package converts pipeline outputs into validation materials and metrics.
It is the bridge between "the system generated recommendations" and "a
stakeholder can trust the recommendations for curriculum work."

## Files

| File | Purpose |
|---|---|
| `validation_protocol.json` | Thresholds, label values, review roles, and claim rules |
| `validation_metrics_report.json` | Automatic metrics computed from filled review/gold-label files |
| `official_readiness_report.json` | Gate-by-gate official-readiness status |
| `validation_pass_fail_report.md` | Human-readable pass/fail summary |
| `gold_templates/sentence_relevance_gold_template.csv` | Label system evidence and contrast sentence relevance |
| `gold_templates/extraction_span_gold_template.csv` | Label Skill-LLM skill/knowledge extraction correctness |
| `reviewer_templates/` | Competency, cluster, recommendation, and DACUM reviewer worksheets |

## Current Status

Official ready: `{readiness.get('official_ready')}`

Current blockers:

```text
{chr(10).join('- ' + b for b in readiness.get('blockers', []))}
```

## How To Use

1. Assign stable reviewer IDs.
2. Fill the CSV files in `reviewer_templates/` and `gold_templates/`.
3. Re-run this script against the same run directory.
4. Use `validation_metrics_report.json` and `validation_pass_fail_report.md`
   to decide which scientific gates passed.

By default, reruns preserve existing templates so filled labels are not lost.
Use `--overwrite-templates` only when you intentionally want a fresh blank
review packet.

Empty templates intentionally produce `not_measured` gates. That is not a
software failure; it is the honest scientific state before human validation.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def write_pass_fail_markdown(path: Path, metrics: dict[str, Any], readiness: dict[str, Any]) -> None:
    gate_lines = "\n".join(
        f"| {name} | {status} |"
        for name, status in readiness.get("gate_statuses", {}).items()
    )
    metric_lines = "\n".join(
        f"| {name} | {value.get('status')} | {value.get('n_reviewed', value.get('n_rows', ''))} |"
        for name, value in metrics.get("metric_groups", {}).items()
        if isinstance(value, dict)
    )
    text = f"""# Validation Pass/Fail Report

Generated: {metrics.get('created_at')}

## Official Readiness

Official ready: `{readiness.get('official_ready')}`

| Gate | Status |
|---|---|
{gate_lines}

## Metric Groups

| Metric group | Status | Reviewed rows |
|---|---|---:|
{metric_lines}

## Blockers

{chr(10).join('- ' + b for b in readiness.get('blockers', [])) or '- none'}

## Interpretation

This report blocks official claims unless every gate passes. A strong single
metric does not override corpus validity, expert review, DACUM alignment, or
stability requirements.
"""
    path.write_text(text, encoding="utf-8")


def build_validation_pack(
    run_dir: Path,
    output_dir: Path,
    evidence_dir: Path,
    dacum_dir: Path,
    stability_dir: Path,
    sentence_evidence_per_comp: int,
    sentence_contrast_per_comp: int,
    extraction_sample_size: int,
    seed: int,
    top_n: int,
    thresholds: dict[str, float],
    overwrite_templates: bool,
) -> dict[str, Any]:
    pipeline_report = _load_json(run_dir / "pipeline_report.json", {})
    competencies = _load_json(run_dir / "competencies.json", [])
    if not isinstance(competencies, list):
        raise ValueError("competencies.json must contain a list")
    phase1_dir = _resolve(pipeline_report.get("phase1_input", "")) if pipeline_report.get("phase1_input") else run_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    gold_dir = output_dir / "gold_templates"
    review_dir = output_dir / "reviewer_templates"
    gold_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    sentence_rows = build_sentence_relevance_template(
        competencies,
        phase1_dir=phase1_dir,
        per_competency_evidence=sentence_evidence_per_comp,
        per_competency_contrast=sentence_contrast_per_comp,
    )
    sentence_template_status = _write_csv_template(gold_dir / "sentence_relevance_gold_template.csv", sentence_rows, [
        "sample_id", "sample_type", "competency_id", "statement",
        "sentence_id", "job_id", "sentence_text",
        "system_predicted_relevant_yes_no", "system_sentence_relevance_score",
        "gold_relevant_yes_no", "gold_relevance_1_5", "gold_required_revision",
        "reviewer_id", "comments",
    ], overwrite_templates)

    extraction_rows = build_extraction_gold_template(
        phase1_dir=phase1_dir,
        sample_size=extraction_sample_size,
        seed=seed,
    )
    extraction_template_status = _write_csv_template(gold_dir / "extraction_span_gold_template.csv", extraction_rows, [
        "sample_id", "source_file", "job_id", "sentence_id", "sentence_text",
        "extracted_text", "extraction_type", "confidence_score",
        "confidence_tier", "source", "system_predicted_extraction_yes_no",
        "gold_valid_extraction_yes_no", "gold_correct_type_yes_no",
        "gold_normalized_label", "gold_missing_spans_in_sentence",
        "reviewer_id", "comments",
    ], overwrite_templates)

    copied_templates = copy_review_templates(evidence_dir, dacum_dir, output_dir, overwrite_templates)
    write_protocol(output_dir, thresholds)
    stability_report = _load_json(stability_dir / "stability_ablation_report.json", {})
    if not isinstance(stability_report, dict):
        stability_report = {}

    competency_rows = _read_csv(review_dir / "expert_review_competencies.csv")
    recommendation_rows = _read_csv(review_dir / "expert_review_recommendations.csv")
    sentence_gold_rows = _read_csv(gold_dir / "sentence_relevance_gold_template.csv")
    extraction_gold_rows = _read_csv(gold_dir / "extraction_span_gold_template.csv")
    dacum_rows = _read_csv(review_dir / "dacum_expert_worksheet.csv")

    metric_groups = {
        "competency_review": compute_competency_review_metrics(competency_rows, thresholds),
        "recommendation_review": compute_recommendation_review_metrics(recommendation_rows, thresholds, top_n=top_n),
        "sentence_relevance_gold": compute_sentence_relevance_metrics(sentence_gold_rows, thresholds),
        "extraction_quality_gold": compute_extraction_metrics(extraction_gold_rows, thresholds),
        "dacum_cbc_review": compute_dacum_metrics(dacum_rows, thresholds),
    }
    metrics = {
        "schema_version": "scientific_validation_metrics_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "evidence_dir": str(evidence_dir),
        "dacum_dir": str(dacum_dir),
        "stability_dir": str(stability_dir),
        "phase1_dir": str(phase1_dir),
        "templates": {
            "copied_reviewer_templates": copied_templates,
            "sentence_relevance_rows": len(sentence_rows),
            "sentence_relevance_template_status": sentence_template_status,
            "extraction_span_rows": len(extraction_rows),
            "extraction_span_template_status": extraction_template_status,
            "overwrite_templates": overwrite_templates,
        },
        "metric_groups": metric_groups,
        "label_status_distribution": dict(Counter(g["status"] for g in metric_groups.values())),
    }
    readiness = build_official_readiness_report(pipeline_report, metric_groups, thresholds, stability_report)

    _write_json(output_dir / "validation_metrics_report.json", metrics)
    _write_json(output_dir / "official_readiness_report.json", readiness)
    write_pass_fail_markdown(output_dir / "validation_pass_fail_report.md", metrics, readiness)
    write_readme(output_dir, _run_tag(run_dir, pipeline_report if isinstance(pipeline_report, dict) else {}), run_dir, metrics, readiness)
    return {
        "metrics": metrics,
        "readiness": readiness,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="results/competency_v2_pipeline_n1k_openrouter_gpt54")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--evidence-dir", default="")
    parser.add_argument("--dacum-dir", default="")
    parser.add_argument("--stability-dir", default="")
    parser.add_argument("--sentence-evidence-per-comp", type=int, default=3)
    parser.add_argument("--sentence-contrast-per-comp", type=int, default=1)
    parser.add_argument("--extraction-sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--overwrite-templates",
        action="store_true",
        help="Regenerate reviewer/gold templates even if they already exist. Default preserves filled labels.",
    )
    args = parser.parse_args()

    run_dir = _resolve(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    pipeline_report = _load_json(run_dir / "pipeline_report.json", {})
    tag = _run_tag(run_dir, pipeline_report if isinstance(pipeline_report, dict) else {})
    output_dir = _resolve(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "validation_packages" / tag
    evidence_dir = _resolve(args.evidence_dir) if args.evidence_dir else PROJECT_ROOT / "results" / "evidence_packages" / tag
    dacum_dir = _resolve(args.dacum_dir) if args.dacum_dir else PROJECT_ROOT / "results" / "dacum_cbc_packages" / tag
    stability_dir = _resolve(args.stability_dir) if args.stability_dir else PROJECT_ROOT / "results" / "stability_ablation" / tag

    result = build_validation_pack(
        run_dir=run_dir,
        output_dir=output_dir,
        evidence_dir=evidence_dir,
        dacum_dir=dacum_dir,
        stability_dir=stability_dir,
        sentence_evidence_per_comp=args.sentence_evidence_per_comp,
        sentence_contrast_per_comp=args.sentence_contrast_per_comp,
        extraction_sample_size=args.extraction_sample_size,
        seed=args.seed,
        top_n=args.top_n,
        thresholds=DEFAULT_THRESHOLDS,
        overwrite_templates=args.overwrite_templates,
    )
    metrics = result["metrics"]
    readiness = result["readiness"]
    print(f"[OK] Scientific validation pack written to: {output_dir}")
    print(f"     sentence relevance rows: {metrics['templates']['sentence_relevance_rows']}")
    print(f"     extraction span rows: {metrics['templates']['extraction_span_rows']}")
    print(f"     metric statuses: {metrics['label_status_distribution']}")
    print(f"     official ready: {readiness['official_ready']}")
    print(f"     blockers: {', '.join(readiness['blockers'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
