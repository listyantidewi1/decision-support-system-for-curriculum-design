"""
Build a research evidence package for one completed competency-v2 pipeline run.

The script reads a run directory such as:

    results/competency_v2_pipeline_n1k_v10d_with_okupasi/

and writes:

    results/evidence_packages/<run_tag>/
        automatic_evaluation.json
        provenance_audit.csv
        competency_quality_report.csv
        cluster_coherence_report.csv
        top_recommendation_audit.csv
        expert_review_competencies.csv
        expert_review_clusters.csv
        expert_review_recommendations.csv
        README.md

The package separates automatic evidence from human-review evidence. It does
not pretend expert-dependent gates are already satisfied.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _join_list(value: Any, limit: int | None = None) -> str:
    items = [str(v).strip() for v in _as_list(value) if str(v).strip()]
    if limit is not None:
        items = items[:limit]
    return " | ".join(items)


def _compact(text: Any, max_chars: int = 260) -> str:
    raw = " ".join(str(text or "").split())
    if len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 3].rstrip() + "..."


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return round(statistics.mean(vals), 4) if vals else 0.0


def _median(values: Iterable[float]) -> float:
    vals = list(values)
    return round(statistics.median(vals), 4) if vals else 0.0


def _min(values: Iterable[float]) -> float:
    vals = list(values)
    return round(min(vals), 4) if vals else 0.0


def _max(values: Iterable[float]) -> float:
    vals = list(values)
    return round(max(vals), 4) if vals else 0.0


def _distribution(values: Iterable[Any]) -> dict[str, int]:
    counts = Counter(str(v) for v in values if v not in (None, ""))
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _run_tag(run_dir: Path, pipeline_report: dict[str, Any]) -> str:
    if pipeline_report.get("tag"):
        return str(pipeline_report["tag"])
    name = run_dir.name
    prefix = "competency_v2_pipeline_"
    return name[len(prefix):] if name.startswith(prefix) else name


def _ranked_competencies(competencies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        competencies,
        key=lambda c: (
            -_as_float(c.get("priority_score")),
            -len(_as_list(c.get("source_job_ids"))),
            str(c.get("id", "")),
        ),
    )


def _competency_cluster_counts(competencies: list[dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for c in competencies:
        cid = c.get("cluster_id")
        if cid:
            counts[str(cid)] += 1
    return counts


def _failed_cluster_counts(failed: list[dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for c in failed:
        cid = c.get("cluster_id")
        if cid:
            counts[str(cid)] += 1
    return counts


def _provenance_complete(c: dict[str, Any]) -> bool:
    return (
        len(_as_list(c.get("related_skills"))) > 0
        and len(_as_list(c.get("contributing_item_ids"))) > 0
        and len(_as_list(c.get("source_job_ids"))) > 0
        and len(_as_list(c.get("source_sentences"))) > 0
    )


def build_automatic_evaluation(
    run_dir: Path,
    out_dir: Path,
    pipeline_report: dict[str, Any],
    competencies: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    batch_reasonings: list[dict[str, Any]],
    noise_audit: dict[str, Any],
    protocol_grounding_gate: float,
    engineering_grounding_gate: float,
) -> dict[str, Any]:
    g_scores = [_as_float(c.get("grounding_score")) for c in competencies]
    skill_g = [_as_float(c.get("skill_grounding_score")) for c in competencies]
    sent_g = [_as_float(c.get("sentence_relevance_score")) for c in competencies]
    priority = [_as_float(c.get("priority_score")) for c in competencies]
    future = [_as_float(c.get("future_weight")) for c in competencies]
    cohesion = [_as_float(c.get("cohesion_score")) for c in clusters]
    provenance_flags = [_provenance_complete(c) for c in competencies]

    n = len(competencies)
    top20 = _ranked_competencies(competencies)[:20]
    stages = pipeline_report.get("stages", {}) if isinstance(pipeline_report, dict) else {}
    phase2_1 = stages.get("phase2_1_clustering", {})
    phase2_2 = stages.get("phase2_2_generation", {})
    phase4 = stages.get("phase4_priority", {})

    gate_summary = {
        "sentence_relevance": {
            "status": "not_measured_in_evidence_package",
            "required_next_step": "Label sentence-relevance review CSV and compute precision/recall.",
        },
        "extraction_quality": {
            "status": "not_measured_in_evidence_package",
            "required_next_step": "Use SkillSpan metrics and production extraction gold labels.",
        },
        "cluster_coherence_automatic": {
            "status": "pass" if cohesion and _min(cohesion) >= 0.50 else "review",
            "criterion": "minimum cluster cohesion >= 0.50",
            "observed_min": _min(cohesion),
            "observed_mean": _mean(cohesion),
        },
        "competency_grounding_protocol": {
            "status": "pass" if g_scores and all(v >= protocol_grounding_gate for v in g_scores) else "fail_or_review",
            "criterion": f"all final competencies grounding_score >= {protocol_grounding_gate}",
            "observed_min": _min(g_scores),
            "observed_mean": _mean(g_scores),
        },
        "competency_grounding_engineering": {
            "status": "pass" if g_scores and all(v >= engineering_grounding_gate for v in g_scores) else "fail_or_review",
            "criterion": f"all final competencies grounding_score >= {engineering_grounding_gate}",
            "observed_min": _min(g_scores),
            "observed_mean": _mean(g_scores),
        },
        "provenance_completeness": {
            "status": "pass" if n and all(provenance_flags) else "fail",
            "criterion": "all final competencies have related skills, item IDs, source jobs, and source sentences",
            "n_complete": int(sum(1 for ok in provenance_flags if ok)),
            "n_final_competencies": n,
        },
        "recommendation_quality": {
            "status": "not_measured_until_expert_review",
            "required_next_step": "Label expert_review_recommendations.csv and compute P@20/NDCG@20.",
        },
        "production_stability": {
            "status": "not_measured_in_single_run",
            "required_next_step": "Run N/seed stability experiment and compute top-20 Jaccard.",
        },
    }

    return {
        "schema_version": "evidence_package_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "evidence_dir": str(out_dir),
        "run_tag": _run_tag(run_dir, pipeline_report),
        "pipeline_report": {
            "model": pipeline_report.get("model"),
            "phase1_input": pipeline_report.get("phase1_input"),
            "jobs_metadata": pipeline_report.get("jobs_metadata"),
            "total_runtime_seconds": pipeline_report.get("total_runtime_seconds"),
            "final_n_competencies": pipeline_report.get("final_n_competencies"),
        },
        "stage_summary": {
            "phase1_load": stages.get("phase1_load", {}),
            "phase2_1_clustering": phase2_1,
            "phase2_2_generation": phase2_2,
            "phase4_priority": phase4,
            "noise_audit": {
                "n_items_dropped_total": noise_audit.get("n_items_dropped_total", 0),
                "by_stage": noise_audit.get("by_stage", {}),
            },
        },
        "competency_summary": {
            "n_final": n,
            "n_failed_or_dropped": len(failed),
            "n_batch_reasonings": len(batch_reasonings),
            "grounding_score": {
                "mean": _mean(g_scores),
                "median": _median(g_scores),
                "min": _min(g_scores),
                "max": _max(g_scores),
            },
            "skill_grounding_score": {
                "mean": _mean(skill_g),
                "median": _median(skill_g),
                "min": _min(skill_g),
                "max": _max(skill_g),
            },
            "sentence_relevance_score": {
                "mean": _mean(sent_g),
                "median": _median(sent_g),
                "min": _min(sent_g),
                "max": _max(sent_g),
            },
            "priority_score": {
                "mean": _mean(priority),
                "median": _median(priority),
                "min": _min(priority),
                "max": _max(priority),
            },
            "future_weight": {
                "mean": _mean(future),
                "median": _median(future),
                "min": _min(future),
                "max": _max(future),
            },
            "kkni_distribution": _distribution(c.get("kkni_level") for c in competencies),
            "education_level_presence": {
                "n_with_education_data": sum(1 for c in competencies if c.get("education_levels_demanded")),
                "n_without_education_data": sum(1 for c in competencies if not c.get("education_levels_demanded")),
            },
            "occupation_mapping": {
                "n_mapped": sum(1 for c in competencies if _as_list(c.get("occupation_matches"))),
                "n_unmapped": sum(1 for c in competencies if not _as_list(c.get("occupation_matches"))),
            },
        },
        "cluster_summary": {
            "n_clusters": len(clusters),
            "cohesion_mean": _mean(cohesion),
            "cohesion_median": _median(cohesion),
            "cohesion_min": _min(cohesion),
            "method_distribution": _distribution(c.get("method") for c in clusters),
        },
        "top20_snapshot": [
            {
                "rank": i + 1,
                "id": c.get("id"),
                "priority_score": c.get("priority_score"),
                "demand_score": c.get("demand_score"),
                "grounding_score": c.get("grounding_score"),
                "future_weight": c.get("future_weight"),
                "kkni_level": c.get("kkni_level"),
                "statement": _compact(c.get("statement"), 180),
            }
            for i, c in enumerate(top20)
        ],
        "protocol_gate_summary": gate_summary,
        "limitations": [
            "Expert review has not been imported by this package builder.",
            "Sentence relevance and extraction correctness require separate labeled gold sets.",
            "Single-run evidence cannot establish production-volume stability.",
            "Generated results folders are local/regeneratable and are not committed to Git by default.",
        ],
    }


def build_provenance_rows(competencies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rank, c in enumerate(_ranked_competencies(competencies), start=1):
        source_sentences = _as_list(c.get("source_sentences"))
        rows.append({
            "rank": rank,
            "competency_id": c.get("id", ""),
            "statement": c.get("statement", ""),
            "priority_score": c.get("priority_score", ""),
            "demand_score": c.get("demand_score", ""),
            "future_weight": c.get("future_weight", ""),
            "grounding_score": c.get("grounding_score", ""),
            "skill_grounding_score": c.get("skill_grounding_score", ""),
            "sentence_relevance_score": c.get("sentence_relevance_score", ""),
            "n_related_skills": len(_as_list(c.get("related_skills"))),
            "n_contributing_item_ids": len(_as_list(c.get("contributing_item_ids"))),
            "n_source_jobs": len(_as_list(c.get("source_job_ids"))),
            "n_source_sentence_ids": len(_as_list(c.get("source_sentence_ids"))),
            "n_source_sentences": len(source_sentences),
            "provenance_complete": _provenance_complete(c),
            "related_skills": _join_list(c.get("related_skills")),
            "source_job_ids_sample": _join_list(c.get("source_job_ids"), limit=10),
            "source_sentences_sample": _join_list([_compact(s, 220) for s in source_sentences[:3]]),
        })
    return rows


def build_quality_rows(competencies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rank, c in enumerate(_ranked_competencies(competencies), start=1):
        evidence_concerns = _as_list(c.get("statement_evidence_concerns"))
        warning_concerns = [x for x in evidence_concerns if isinstance(x, dict) and x.get("severity") == "warning"]
        rows.append({
            "rank": rank,
            "competency_id": c.get("id", ""),
            "statement": c.get("statement", ""),
            "statement_chars": len(str(c.get("statement") or "")),
            "rationale_chars": len(str(c.get("rationale") or "")),
            "n_related_skills": len(_as_list(c.get("related_skills"))),
            "n_soft_skills": len(_as_list(c.get("soft_skills_required"))),
            "n_source_jobs": len(_as_list(c.get("source_job_ids"))),
            "n_source_sentences": len(_as_list(c.get("source_sentences"))),
            "grounding_score": c.get("grounding_score", ""),
            "grounding_passed": c.get("grounding_passed", ""),
            "grounding_method": c.get("grounding_method", ""),
            "n_statement_evidence_concerns": len(evidence_concerns),
            "n_statement_evidence_warnings": len(warning_concerns),
            "kkni_level": c.get("kkni_level", ""),
            "kkni_match_similarity": c.get("kkni_match_similarity", ""),
            "education_levels_demanded": json.dumps(c.get("education_levels_demanded", {}), ensure_ascii=False),
            "n_jobs_with_education": c.get("n_jobs_with_education", ""),
            "n_occupation_matches": len(_as_list(c.get("occupation_matches"))),
            "occupation_matches": _join_list(
                [
                    x.get("occupation_en") or x.get("occupation_id") or x
                    for x in _as_list(c.get("occupation_matches"))
                ],
                limit=5,
            ),
            "rationale": c.get("rationale", ""),
            "grounding_reasoning": c.get("grounding_reasoning", ""),
        })
    return rows


def build_cluster_rows(
    clusters: list[dict[str, Any]],
    competencies: list[dict[str, Any]],
    failed: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    final_counts = _competency_cluster_counts(competencies)
    failed_counts = _failed_cluster_counts(failed)
    rows = []
    for cluster in sorted(clusters, key=lambda c: (_as_float(c.get("cohesion_score")), str(c.get("id")))):
        items = _as_list(cluster.get("items"))
        rows.append({
            "cluster_id": cluster.get("id", ""),
            "summary_label": cluster.get("summary_label", ""),
            "method": cluster.get("method", ""),
            "stream": cluster.get("stream", ""),
            "cohesion_score": cluster.get("cohesion_score", ""),
            "cohesion_std": cluster.get("cohesion_std", ""),
            "n_items": cluster.get("n_items", ""),
            "n_unique_jobs": cluster.get("n_unique_jobs", ""),
            "n_skill_items": cluster.get("n_skill_items", ""),
            "n_knowledge_items": cluster.get("n_knowledge_items", ""),
            "n_final_competencies": final_counts[str(cluster.get("id", ""))],
            "n_failed_or_dropped_competencies": failed_counts[str(cluster.get("id", ""))],
            "top_terms": _join_list(cluster.get("top_terms")),
            "item_sample": _join_list(items, limit=20),
            "expert_reviewer_id": "",
            "expert_cluster_coherence_1_5": "",
            "expert_unrelated_items": "",
            "expert_comments": "",
        })
    return rows


def build_top_recommendation_rows(competencies: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    rows = []
    for rank, c in enumerate(_ranked_competencies(competencies)[:top_n], start=1):
        explanation = (
            f"priority={_as_float(c.get('priority_score')):.3f}; "
            f"demand={_as_float(c.get('demand_score')):.3f}; "
            f"grounding={_as_float(c.get('grounding_score')):.3f}; "
            f"future={_as_float(c.get('future_weight')):.3f}; "
            f"jobs={len(_as_list(c.get('source_job_ids')))}"
        )
        rows.append({
            "rank": rank,
            "competency_id": c.get("id", ""),
            "statement": c.get("statement", ""),
            "priority_score": c.get("priority_score", ""),
            "demand_score": c.get("demand_score", ""),
            "grounding_score": c.get("grounding_score", ""),
            "future_weight": c.get("future_weight", ""),
            "n_source_jobs": len(_as_list(c.get("source_job_ids"))),
            "n_source_sentences": len(_as_list(c.get("source_sentences"))),
            "kkni_level": c.get("kkni_level", ""),
            "education_levels_demanded": json.dumps(c.get("education_levels_demanded", {}), ensure_ascii=False),
            "occupation_matches": _join_list(
                [
                    x.get("occupation_en") or x.get("occupation_id") or x
                    for x in _as_list(c.get("occupation_matches"))
                ],
                limit=5,
            ),
            "score_explanation": explanation,
            "related_skills": _join_list(c.get("related_skills")),
            "evidence_sample": _join_list([_compact(s, 220) for s in _as_list(c.get("source_sentences"))[:3]]),
        })
    return rows


def build_expert_competency_rows(competencies: list[dict[str, Any]], top_n: int | None) -> list[dict[str, Any]]:
    selected = _ranked_competencies(competencies)
    if top_n is not None:
        selected = selected[:top_n]
    rows = []
    for rank, c in enumerate(selected, start=1):
        rows.append({
            "reviewer_id": "",
            "rank": rank,
            "competency_id": c.get("id", ""),
            "statement": c.get("statement", ""),
            "related_skills": _join_list(c.get("related_skills")),
            "rationale": c.get("rationale", ""),
            "source_sentences_sample": _join_list([_compact(s, 220) for s in _as_list(c.get("source_sentences"))[:5]]),
            "validity_1_5": "",
            "relevance_yes_partial_no": "",
            "grounding_clarity_1_5": "",
            "rationale_clarity_1_5": "",
            "actionability_1_5": "",
            "hallucination_concern_yes_no": "",
            "comments": "",
        })
    return rows


def build_expert_recommendation_rows(competencies: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    rows = []
    for rank, c in enumerate(_ranked_competencies(competencies)[:top_n], start=1):
        rows.append({
            "reviewer_id": "",
            "rank": rank,
            "competency_id": c.get("id", ""),
            "statement": c.get("statement", ""),
            "priority_score": c.get("priority_score", ""),
            "score_explanation": (
                f"demand={_as_float(c.get('demand_score')):.3f}; "
                f"grounding={_as_float(c.get('grounding_score')):.3f}; "
                f"future={_as_float(c.get('future_weight')):.3f}"
            ),
            "priority_label_yes_partial_no": "",
            "curriculum_usefulness_1_5": "",
            "ranking_reasonableness_1_5": "",
            "comments": "",
        })
    return rows


def write_readme(
    path: Path,
    run_tag: str,
    run_dir: Path,
    top_n: int,
    automatic_eval: dict[str, Any],
) -> None:
    summary = automatic_eval["competency_summary"]
    cluster = automatic_eval["cluster_summary"]
    gates = automatic_eval["protocol_gate_summary"]
    text = f"""# Evidence Package: {run_tag}

Generated: {automatic_eval["created_at"]}

Source run:

```text
{run_dir}
```

## What This Package Is

This folder is the research evidence package for one completed competency-v2
pipeline run. It separates automatic checks from human-review materials.

## Files

| File | Purpose |
|---|---|
| `automatic_evaluation.json` | Machine-readable summary of automatic evidence and pass/fail status |
| `provenance_audit.csv` | Per-competency traceability from recommendation to source evidence |
| `competency_quality_report.csv` | Grounding, text-quality, KKNI, education, occupation, and rationale audit |
| `cluster_coherence_report.csv` | Per-cluster cohesion and cluster-review worksheet |
| `top_recommendation_audit.csv` | Score breakdown and evidence sample for top {top_n} recommendations |
| `expert_review_competencies.csv` | Human-review form for competency validity and clarity |
| `expert_review_clusters.csv` | Human-review form for cluster coherence |
| `expert_review_recommendations.csv` | Human-review form for top recommendation usefulness |

## Automatic Snapshot

| Metric | Value |
|---|---:|
| Final competencies | {summary["n_final"]} |
| Failed or dropped competencies | {summary["n_failed_or_dropped"]} |
| Mean grounding score | {summary["grounding_score"]["mean"]} |
| Minimum grounding score | {summary["grounding_score"]["min"]} |
| Mean cluster cohesion | {cluster["cohesion_mean"]} |
| Minimum cluster cohesion | {cluster["cohesion_min"]} |

## Current Gate Status

| Gate | Status |
|---|---|
| Sentence relevance | {gates["sentence_relevance"]["status"]} |
| Extraction quality | {gates["extraction_quality"]["status"]} |
| Cluster coherence automatic | {gates["cluster_coherence_automatic"]["status"]} |
| Competency grounding, protocol gate | {gates["competency_grounding_protocol"]["status"]} |
| Competency grounding, engineering gate | {gates["competency_grounding_engineering"]["status"]} |
| Provenance completeness | {gates["provenance_completeness"]["status"]} |
| Recommendation quality | {gates["recommendation_quality"]["status"]} |
| Production stability | {gates["production_stability"]["status"]} |

## Reviewer Instructions

1. Use `expert_review_competencies.csv` to rate whether each generated
   competency is valid, useful, grounded, and actionable.
2. Use `expert_review_clusters.csv` to rate whether grouped skills/knowledge
   belong together.
3. Use `expert_review_recommendations.csv` to judge whether the top ranked
   recommendations deserve curriculum-design priority.
4. Keep `reviewer_id` stable for each reviewer. Do not change system-generated
   IDs or ranks.

## Interpretation

This package is not the final dissertation result by itself. It is the evidence
bundle that lets us decide whether this run is ready for expert validation,
larger-N stability experiments, or another pipeline revision.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="results/competency_v2_pipeline_n1k_v10d_with_okupasi",
        help="Completed v2 run directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Evidence package output directory. Defaults to results/evidence_packages/<run_tag>.",
    )
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--review-competencies", type=int, default=0,
                        help="Limit competency review CSV to top N. 0 means all final competencies.")
    parser.add_argument("--review-clusters", type=int, default=50,
                        help="Limit cluster review CSV to N lowest-cohesion clusters. 0 means all clusters.")
    parser.add_argument("--protocol-grounding-gate", type=float, default=0.80)
    parser.add_argument("--engineering-grounding-gate", type=float, default=0.50)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    pipeline_report = _load_json(run_dir / "pipeline_report.json", {})
    competencies = _load_json(run_dir / "competencies.json", [])
    failed = _load_json(run_dir / "competencies_failed.json", [])
    clusters = _load_json(run_dir / "clusters.json", [])
    batch_reasonings = _load_json(run_dir / "batch_reasonings.json", [])
    noise_audit = _load_json(run_dir / "noise_audit.json", {})

    if not isinstance(competencies, list):
        raise ValueError("competencies.json must contain a list")
    if not isinstance(clusters, list):
        raise ValueError("clusters.json must contain a list")

    tag = _run_tag(run_dir, pipeline_report)
    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "evidence_packages" / tag
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    automatic_eval = build_automatic_evaluation(
        run_dir=run_dir,
        out_dir=out_dir,
        pipeline_report=pipeline_report,
        competencies=competencies,
        failed=failed if isinstance(failed, list) else [],
        clusters=clusters,
        batch_reasonings=batch_reasonings if isinstance(batch_reasonings, list) else [],
        noise_audit=noise_audit if isinstance(noise_audit, dict) else {},
        protocol_grounding_gate=args.protocol_grounding_gate,
        engineering_grounding_gate=args.engineering_grounding_gate,
    )
    _write_json(out_dir / "automatic_evaluation.json", automatic_eval)

    provenance_rows = build_provenance_rows(competencies)
    _write_csv(out_dir / "provenance_audit.csv", provenance_rows, [
        "rank", "competency_id", "statement", "priority_score", "demand_score",
        "future_weight", "grounding_score", "skill_grounding_score",
        "sentence_relevance_score", "n_related_skills", "n_contributing_item_ids",
        "n_source_jobs", "n_source_sentence_ids", "n_source_sentences",
        "provenance_complete", "related_skills", "source_job_ids_sample",
        "source_sentences_sample",
    ])

    quality_rows = build_quality_rows(competencies)
    _write_csv(out_dir / "competency_quality_report.csv", quality_rows, [
        "rank", "competency_id", "statement", "statement_chars", "rationale_chars",
        "n_related_skills", "n_soft_skills", "n_source_jobs", "n_source_sentences",
        "grounding_score", "grounding_passed", "grounding_method",
        "n_statement_evidence_concerns", "n_statement_evidence_warnings",
        "kkni_level", "kkni_match_similarity", "education_levels_demanded",
        "n_jobs_with_education", "n_occupation_matches", "occupation_matches",
        "rationale", "grounding_reasoning",
    ])

    cluster_rows = build_cluster_rows(clusters, competencies, failed if isinstance(failed, list) else [])
    _write_csv(out_dir / "cluster_coherence_report.csv", cluster_rows, [
        "cluster_id", "summary_label", "method", "stream", "cohesion_score",
        "cohesion_std", "n_items", "n_unique_jobs", "n_skill_items",
        "n_knowledge_items", "n_final_competencies",
        "n_failed_or_dropped_competencies", "top_terms", "item_sample",
        "expert_reviewer_id", "expert_cluster_coherence_1_5",
        "expert_unrelated_items", "expert_comments",
    ])

    top_rows = build_top_recommendation_rows(competencies, args.top_n)
    _write_csv(out_dir / "top_recommendation_audit.csv", top_rows, [
        "rank", "competency_id", "statement", "priority_score", "demand_score",
        "grounding_score", "future_weight", "n_source_jobs", "n_source_sentences",
        "kkni_level", "education_levels_demanded", "occupation_matches",
        "score_explanation", "related_skills", "evidence_sample",
    ])

    review_top_n = None if args.review_competencies <= 0 else args.review_competencies
    expert_comp_rows = build_expert_competency_rows(competencies, review_top_n)
    _write_csv(out_dir / "expert_review_competencies.csv", expert_comp_rows, [
        "reviewer_id", "rank", "competency_id", "statement", "related_skills",
        "rationale", "source_sentences_sample", "validity_1_5",
        "relevance_yes_partial_no", "grounding_clarity_1_5",
        "rationale_clarity_1_5", "actionability_1_5",
        "hallucination_concern_yes_no", "comments",
    ])

    cluster_review_rows = sorted(cluster_rows, key=lambda r: _as_float(r.get("cohesion_score")))
    if args.review_clusters > 0:
        cluster_review_rows = cluster_review_rows[:args.review_clusters]
    _write_csv(out_dir / "expert_review_clusters.csv", cluster_review_rows, [
        "expert_reviewer_id", "cluster_id", "summary_label", "cohesion_score",
        "n_items", "n_unique_jobs", "n_final_competencies", "item_sample",
        "expert_cluster_coherence_1_5", "expert_unrelated_items", "expert_comments",
    ])

    expert_rec_rows = build_expert_recommendation_rows(competencies, args.top_n)
    _write_csv(out_dir / "expert_review_recommendations.csv", expert_rec_rows, [
        "reviewer_id", "rank", "competency_id", "statement", "priority_score",
        "score_explanation", "priority_label_yes_partial_no",
        "curriculum_usefulness_1_5", "ranking_reasonableness_1_5", "comments",
    ])

    write_readme(out_dir / "README.md", tag, run_dir, args.top_n, automatic_eval)

    print(f"[OK] Evidence package written to: {out_dir}")
    print(f"     Final competencies: {len(competencies)}")
    print(f"     Clusters: {len(clusters)}")
    print(f"     Expert competency rows: {len(expert_comp_rows)}")
    print(f"     Expert cluster rows: {len(cluster_review_rows)}")
    print(f"     Expert recommendation rows: {len(expert_rec_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

