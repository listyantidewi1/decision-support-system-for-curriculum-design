"""
Apply scientific-fix sprint 1 to a completed competency-v2 run.

Sprint 1 does not regenerate competencies. It adds an auditable
recommendation contract on top of a completed run:

* competency class: core technical, engineering process, employability support,
  or out-of-domain / needs review
* true job-prevalence denominator from jobs_metadata.csv
* support tier from source-job counts
* occupation mapping status
* official-core eligibility gates for a demand-first candidate list

The enriched competencies are written to competencies_sprint1_enriched.json and,
by default, merged additively back into competencies.json so the dashboard can
surface the fields.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPRINT_SCHEMA_VERSION = "scientific_sprint1_v1"

STATUS_VALID_ENGINEERING = "valid_for_engineering_test"
STATUS_VALID_INDONESIAN = "valid_for_indonesian_claim"
STATUS_INVALID = "invalid_or_incomplete"
STATUS_REVIEW = "needs_manual_review"

RECOMMENDATION_EXPLORATORY = "exploratory"
RECOMMENDATION_REVIEW = "review_candidate"
RECOMMENDATION_OFFICIAL = "official_ready"
RECOMMENDATION_NOT_FROM_CORPUS = "not_recommended_from_this_corpus"

LLM_FUTURE_SOURCE = "llm_suggested_not_evidence"
EVIDENCE_FUTURE_SOURCES = {
    "deterministic_domain_mapping",
    "external_trend_evidence",
    "official_reference_mapping",
}


CLASS_LABELS = {
    "core_technical": "Core technical",
    "engineering_process": "Engineering process / leadership",
    "employability_support": "Employability / support",
    "out_of_domain_or_needs_review": "Needs review / out-of-domain",
}

SUPPORT_TIER_LABELS = {
    "strong_support": "Strong support",
    "moderate_support": "Moderate support",
    "low_support": "Low support",
}

OCCUPATION_STATUS_LABELS = {
    "official_or_template": "Official/template",
    "community_only": "Community-only",
    "unmapped": "Unmapped",
}

TECH_TERMS = {
    "ai",
    "ai/ml",
    "android",
    "api",
    "apex",
    "application development",
    "architecture",
    "artificial intelligence",
    "automation",
    "azure",
    "azure devops",
    "back-end",
    "backend",
    "ci/cd",
    "cloud",
    "cloud computing",
    "computer engineering",
    "computer science",
    "computer vision",
    "data protection",
    "database",
    "databases",
    "devops",
    "distributed",
    "distributed computing",
    "distributed systems",
    "engineering foundations",
    "front-end",
    "frontend",
    "git",
    "github",
    "google cloud",
    "high-performance computing",
    "hpc",
    "iac",
    "image processing",
    "infrastructure as code",
    "ios",
    "machine learning",
    "mobile",
    "privacy",
    "regulatory",
    "salesforce",
    "security",
    "servicenow",
    "service cloud",
    "software",
    "software development",
    "software engineering",
    "sql",
    "sql server",
    "swift",
    "system architecture",
    "systems engineering",
    "web services",
}

CORE_TECH_ANCHOR_TERMS = {
    ".net",
    ".net core",
    "ai/ml",
    "android",
    "angular",
    "api",
    "apis",
    "artificial intelligence",
    "automated testing",
    "automation",
    "aws",
    "azure",
    "azure devops",
    "back-end",
    "backend",
    "business analytics",
    "business intelligence",
    "c#",
    "c++",
    "ci/cd",
    "cloud computing",
    "cloud infrastructure",
    "cloud platform",
    "cloud platforms",
    "computer science",
    "cybersecurity",
    "data analytics",
    "data engineering",
    "data ingestion",
    "data integration",
    "data pipeline",
    "data pipelines",
    "data transformation",
    "database",
    "databases",
    "debugging",
    "devops",
    "docker",
    "front-end",
    "frontend",
    "gcp",
    "generative ai",
    "google cloud",
    "iac",
    "infrastructure as code",
    "ios",
    "java",
    "javascript",
    "kubernetes",
    "large language models",
    "machine learning",
    "mlops",
    "mobile applications",
    "natural language processing",
    "nlp",
    "node.js",
    "power apps",
    "power bi",
    "python",
    "react",
    "rest api",
    "restful api",
    "security",
    "sharepoint",
    "software architecture",
    "software development",
    "software engineering",
    "sql",
    "swift",
    "terraform",
    "testing",
    "typescript",
}

SUPPORT_LOCK_TERMS = {
    "analytical",
    "analytical and problem-solving skills",
    "analytical skills",
    "analytical thinking",
    "attention to detail",
    "collaborate effectively",
    "collaboration",
    "communication",
    "communication skills",
    "cross-functional",
    "cross-functionally",
    "detail-oriented",
    "flexible professional conduct",
    "interpersonal",
    "problem solvers",
    "problem-solving",
    "problem-solving skills",
    "solve complex problems",
    "solving complex problems",
    "strong work ethic",
    "team player",
    "teamwork",
    "work cross-functionally",
    "work ethic",
    "work flexibly",
}

PROCESS_LOCK_TERMS = {
    "agile",
    "coaching",
    "delivery work",
    "engineering leadership",
    "lead engineering teams",
    "lead small teams",
    "lead teams",
    "mentoring",
    "project management",
    "scrum",
    "sprint planning",
    "technical leader",
    "technical leadership",
}

PROCESS_TERMS = {
    "agile",
    "architectural direction",
    "delivery cycle",
    "lead a small technical team",
    "lead reviews",
    "lead technical",
    "leadership",
    "project execution",
    "project management",
    "scrum",
    "sprint planning",
    "technical decision-making",
    "technical leader",
    "technical leadership",
}

SUPPORT_TERMS = {
    "analytical",
    "analytical skills",
    "attention to detail",
    "collaborate effectively",
    "collaboration",
    "cross-functional",
    "detail-oriented",
    "fast-paced",
    "manage multiple priorities",
    "manage work",
    "mentoring",
    "multitask",
    "problem-solving",
    "priorities",
    "prioritisation",
    "prioritize tasks",
    "team player",
    "teamwork",
    "time management",
}

OUT_OF_DOMAIN_TERMS = {
    "accounting",
    "cad",
    "child protection",
    "clinical governance",
    "budgeting",
    "drafting",
    "electrical engineering",
    "electrical engineer",
    "finance",
    "financial control",
    "internal audit",
    "mental capacity",
    "payroll",
    "prototyping",
    "safeguarding",
    "social work",
    "social worker",
    "soldering",
    "vulnerable adult",
    "vulnerable-adult",
}


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _text_blob(comp: dict[str, Any]) -> str:
    parts = [
        comp.get("statement") or "",
        " ".join(str(s) for s in _as_list(comp.get("related_skills"))),
    ]
    return " ".join(parts).lower()


def _contains_any(text: str, terms: set[str]) -> list[str]:
    matches: list[str] = []
    for term in terms:
        pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
        if re.search(pattern, text):
            matches.append(term)
    return sorted(matches)


def classify_competency(comp: dict[str, Any]) -> tuple[str, str]:
    text = _text_blob(comp)
    related_text = " ".join(str(s) for s in _as_list(comp.get("related_skills"))).lower()
    out_terms = _contains_any(text, OUT_OF_DOMAIN_TERMS)
    if out_terms:
        return (
            "out_of_domain_or_needs_review",
            f"Matched out-of-domain terms: {', '.join(out_terms[:5])}.",
        )

    anchor_terms = _contains_any(text, CORE_TECH_ANCHOR_TERMS)
    related_anchor_terms = _contains_any(related_text, CORE_TECH_ANCHOR_TERMS)
    support_lock_terms = _contains_any(text, SUPPORT_LOCK_TERMS)
    related_support_terms = _contains_any(related_text, SUPPORT_LOCK_TERMS)
    process_lock_terms = _contains_any(text, PROCESS_LOCK_TERMS)

    if process_lock_terms:
        return (
            "engineering_process",
            f"Matched delivery, project, or technical-leadership terms: {', '.join(process_lock_terms[:5])}.",
        )

    if related_support_terms and not related_anchor_terms:
        return (
            "employability_support",
            f"Related skills are generic employability/work-habit terms: {', '.join(related_support_terms[:5])}.",
        )

    process_terms = _contains_any(text, PROCESS_TERMS)
    if process_terms:
        return (
            "engineering_process",
            f"Matched delivery, project, or technical-leadership terms: {', '.join(process_terms[:5])}.",
        )

    support_terms = _contains_any(text, SUPPORT_TERMS)
    if support_lock_terms and not anchor_terms:
        return (
            "employability_support",
            f"Matched generic employability or work-habit terms without a technical artifact: {', '.join(support_lock_terms[:5])}.",
        )
    if support_terms and not anchor_terms:
        return (
            "employability_support",
            f"Matched generic employability or work-habit terms: {', '.join(support_terms[:5])}.",
        )

    if anchor_terms:
        return (
            "core_technical",
            f"Matched ICT/software technical anchors: {', '.join(anchor_terms[:5])}.",
        )

    if support_terms:
        return (
            "employability_support",
            f"Matched employability terms without a specific technical anchor: {', '.join(support_terms[:5])}.",
        )

    if comp.get("occupation_matches"):
        return (
            "core_technical",
            "Has an occupation mapping and no support/process/out-of-domain trigger.",
        )

    return (
        "out_of_domain_or_needs_review",
        "No specific ICT/software, process, or employability signal was detected.",
    )


def support_tier(n_source_jobs: int, strong_min_jobs: int, moderate_min_jobs: int) -> tuple[str, str]:
    if n_source_jobs >= strong_min_jobs:
        return "strong_support", f"Supported by at least {strong_min_jobs} unique source jobs."
    if n_source_jobs >= moderate_min_jobs:
        return "moderate_support", f"Supported by {moderate_min_jobs}-{strong_min_jobs - 1} unique source jobs."
    return "low_support", f"Supported by fewer than {moderate_min_jobs} unique source jobs."


def _is_official_occupation_match(match: dict[str, Any]) -> bool:
    source = str(match.get("source_type") or "")
    return bool(match.get("is_official")) or source.lower().startswith("template-based")


def occupation_mapping_status(comp: dict[str, Any]) -> str:
    matches = [m for m in _as_list(comp.get("occupation_matches")) if isinstance(m, dict)]
    if not matches:
        return "unmapped"
    if any(_is_official_occupation_match(m) for m in matches):
        return "official_or_template"
    return "community_only"


def load_job_denominator(jobs_metadata: Path | None) -> tuple[int, int]:
    if not jobs_metadata or not jobs_metadata.exists():
        return 0, 0

    unique_job_ids: set[str] = set()
    with jobs_metadata.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            jid = (row.get("job_id") or "").strip()
            if jid:
                unique_job_ids.add(jid)
    return len(unique_job_ids), sum(1 for _ in unique_job_ids)


def resolve_jobs_metadata(args_jobs_metadata: str, report: dict[str, Any]) -> Path | None:
    candidate = args_jobs_metadata or report.get("jobs_metadata") or ""
    if not candidate:
        stages = report.get("stages", {}) if isinstance(report, dict) else {}
        candidate = (stages.get("phase2_4_education", {}) or {}).get("jobs_metadata_path") or ""
    if not candidate:
        return None
    path = Path(candidate)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_corpus_context(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {
            "corpus_validity_status": STATUS_REVIEW,
            "corpus_country": "unknown",
            "corpus_language": "unknown",
            "corpus_lineage_report_path": "",
        }
    data = _load_json(path, {})
    if not isinstance(data, dict):
        data = {}
    return {
        "corpus_validity_status": data.get("corpus_validity_status") or STATUS_REVIEW,
        "corpus_country": data.get("corpus_country") or "unknown",
        "corpus_language": data.get("corpus_language") or "unknown",
        "corpus_lineage_report_path": str(path),
        "corpus_dataset_id": data.get("dataset_id") or "",
    }


def resolve_corpus_report(args_corpus_report: str, run_dir: Path, pipeline_report: dict[str, Any]) -> Path | None:
    candidate = args_corpus_report or ""
    if not candidate and isinstance(pipeline_report, dict):
        candidate = (
            pipeline_report.get("corpus_lineage_report")
            or ((pipeline_report.get("stages", {}) or {}).get("corpus_lineage_audit", {}) or {}).get("report_path")
            or ""
        )
    if not candidate:
        local = run_dir / "corpus_lineage_report.json"
        if local.exists():
            return local
        return None
    path = Path(candidate)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def official_core_blockers(
    comp: dict[str, Any],
    competency_class: str,
    n_source_jobs: int,
    grounding_score: float,
    min_jobs: int,
    min_grounding: float,
) -> list[str]:
    blockers: list[str] = []
    if competency_class != "core_technical":
        blockers.append(f"class={competency_class}")
    if n_source_jobs < min_jobs:
        blockers.append(f"source_jobs<{min_jobs}")
    if grounding_score < min_grounding:
        blockers.append(f"grounding<{min_grounding:.2f}")
    if occupation_mapping_status(comp) == "unmapped":
        blockers.append("occupation_unmapped")
    cluster_status = str(comp.get("cluster_validity_status") or "")
    if cluster_status in {"split_required", "mixed_domain", "support_process_cluster", "out_of_domain_cluster"}:
        blockers.append(f"cluster_{cluster_status}")
    theme_status = str(comp.get("theme_recommendation_status") or "")
    if theme_status == RECOMMENDATION_NOT_FROM_CORPUS:
        blockers.append("theme_not_recommended_from_this_corpus")
    return blockers


def official_readiness_blockers(
    *,
    corpus_status: str,
    occ_status: str,
    core_blockers: list[str],
    expert_review_passed: bool,
    stability_passed: bool,
) -> list[str]:
    blockers = list(core_blockers)
    if corpus_status == STATUS_VALID_ENGINEERING:
        blockers.append("corpus_valid_for_engineering_test_only")
    elif corpus_status != STATUS_VALID_INDONESIAN:
        blockers.append("corpus_lineage_not_valid_for_indonesian_claim")
    if occ_status != "official_or_template":
        blockers.append("occupation_mapping_not_official")
    if not expert_review_passed:
        blockers.append("expert_review_not_completed")
    if not stability_passed:
        blockers.append("stability_not_validated")
    return sorted(set(blockers))


def recommendation_status(
    *,
    corpus_status: str,
    core_blockers: list[str],
    readiness_blockers: list[str],
) -> str:
    if "theme_not_recommended_from_this_corpus" in core_blockers:
        return RECOMMENDATION_NOT_FROM_CORPUS
    if corpus_status not in {STATUS_VALID_ENGINEERING, STATUS_VALID_INDONESIAN}:
        return RECOMMENDATION_EXPLORATORY
    if core_blockers:
        return RECOMMENDATION_EXPLORATORY
    if corpus_status == STATUS_VALID_INDONESIAN and not readiness_blockers:
        return RECOMMENDATION_OFFICIAL
    return RECOMMENDATION_REVIEW


def scientific_flags(
    competency_class: str,
    tier: str,
    occ_status: str,
    blockers: list[str],
    comp: dict[str, Any] | None = None,
) -> list[str]:
    comp = comp or {}
    flags: list[str] = []
    if competency_class == "employability_support":
        flags.append("support_or_employability_signal")
    elif competency_class == "engineering_process":
        flags.append("engineering_process_or_leadership_signal")
    elif competency_class == "out_of_domain_or_needs_review":
        flags.append("out_of_domain_or_needs_review")
    if tier == "low_support":
        flags.append("low_job_support")
    if occ_status == "unmapped":
        flags.append("occupation_unmapped")
    elif occ_status == "community_only":
        flags.append("community_only_occupation_mapping")
    future_source = comp.get("future_weight_source") or LLM_FUTURE_SOURCE
    if future_source == LLM_FUTURE_SOURCE and _as_float(comp.get("future_weight")) > 0:
        flags.append("future_weight_llm_suggested_not_evidence")
    cluster_status = comp.get("cluster_validity_status") or ""
    if cluster_status:
        flags.append(f"cluster_{cluster_status}")
    theme_status = comp.get("theme_evidence_status") or ""
    if theme_status:
        flags.append(f"theme_{theme_status}")
    flags.extend(blockers)
    return sorted(set(flags))


def public_future_weight(comp: dict[str, Any]) -> float:
    """Future weight used in public sorting.

    The generated `future_weight` remains in the record for ablation and
    exploratory research, but it is not treated as demand evidence unless a
    deterministic or external-evidence source explicitly says so.
    """
    source = comp.get("future_weight_source") or LLM_FUTURE_SOURCE
    if source in EVIDENCE_FUTURE_SOURCES:
        return _as_float(comp.get("future_weight"))
    return 0.0


def enrich_competencies(
    competencies: list[dict[str, Any]],
    denominator: int,
    min_official_jobs: int,
    min_grounding: float,
    strong_min_jobs: int,
    moderate_min_jobs: int,
    corpus_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    corpus_context = corpus_context or {}
    corpus_status = corpus_context.get("corpus_validity_status") or STATUS_REVIEW
    corpus_country = corpus_context.get("corpus_country") or "unknown"
    corpus_language = corpus_context.get("corpus_language") or "unknown"
    corpus_dataset_id = corpus_context.get("corpus_dataset_id") or ""
    enriched: list[dict[str, Any]] = []
    for priority_rank, comp in enumerate(competencies, start=1):
        c = dict(comp)
        c.setdefault("future_weight_source", LLM_FUTURE_SOURCE)
        n_source_jobs = len(set(str(j) for j in _as_list(c.get("source_job_ids")) if str(j).strip()))
        prevalence = (n_source_jobs / denominator) if denominator else 0.0
        comp_class, class_reason = classify_competency(c)
        tier, tier_reason = support_tier(n_source_jobs, strong_min_jobs, moderate_min_jobs)
        occ_status = occupation_mapping_status(c)
        grounding = _as_float(c.get("grounding_score", c.get("grounding_score_preview", 0.0)))
        blockers = official_core_blockers(
            c,
            competency_class=comp_class,
            n_source_jobs=n_source_jobs,
            grounding_score=grounding,
            min_jobs=min_official_jobs,
            min_grounding=min_grounding,
        )
        official_eligible = not blockers
        readiness_blockers = official_readiness_blockers(
            corpus_status=corpus_status,
            occ_status=occ_status,
            core_blockers=blockers,
            expert_review_passed=bool(c.get("expert_review_passed")),
            stability_passed=bool(c.get("stability_passed")),
        )
        rec_status = recommendation_status(
            corpus_status=corpus_status,
            core_blockers=blockers,
            readiness_blockers=readiness_blockers,
        )

        c.update({
            "scientific_sprint1_schema_version": SPRINT_SCHEMA_VERSION,
            "pipeline_priority_rank": priority_rank,
            "corpus_dataset_id": corpus_dataset_id,
            "corpus_country": corpus_country,
            "corpus_language": corpus_language,
            "corpus_validity_status": corpus_status,
            "competency_class": comp_class,
            "competency_class_label": CLASS_LABELS[comp_class],
            "competency_class_reason": class_reason,
            "n_source_jobs": n_source_jobs,
            "eligible_job_denominator": denominator,
            "demand_denominator": denominator,
            "job_prevalence": round(prevalence, 6),
            "job_prevalence_pct": round(prevalence * 100, 3),
            "support_tier": tier,
            "support_tier_label": SUPPORT_TIER_LABELS[tier],
            "support_tier_reason": tier_reason,
            "occupation_mapping_status": occ_status,
            "occupation_mapping_label": OCCUPATION_STATUS_LABELS[occ_status],
            "official_core_eligible": official_eligible,
            "official_core_blockers": blockers,
            "recommendation_status": rec_status,
            "official_readiness_blockers": readiness_blockers,
            "scientific_flags": scientific_flags(comp_class, tier, occ_status, blockers, c),
        })
        enriched.append(c)

    official_sorted = sorted(
        [c for c in enriched if c.get("official_core_eligible")],
        key=lambda c: (
            -int(c.get("n_source_jobs") or 0),
            -_as_float(c.get("grounding_score", c.get("grounding_score_preview", 0.0))),
            -public_future_weight(c),
            str(c.get("id", "")),
        ),
    )
    rank_by_id = {c.get("id"): rank for rank, c in enumerate(official_sorted, start=1)}
    for c in enriched:
        c["official_core_rank"] = rank_by_id.get(c.get("id"))

    class_order = {
        "core_technical": 0,
        "engineering_process": 1,
        "employability_support": 2,
        "out_of_domain_or_needs_review": 3,
    }
    for cls in class_order:
        class_items = sorted(
            [c for c in enriched if c.get("competency_class") == cls],
            key=lambda c: (
                -_as_float(c.get("job_prevalence")),
                -int(c.get("n_source_jobs") or 0),
                -_as_float(c.get("grounding_score", c.get("grounding_score_preview", 0.0))),
                -public_future_weight(c),
                str(c.get("id", "")),
            ),
        )
        for rank, c in enumerate(class_items, start=1):
            c["class_rank"] = rank
            c["class_sort_group"] = class_order.get(cls, 99)
    return enriched


def class_aware_sort_key(comp: dict[str, Any]) -> tuple:
    return (
        int(comp.get("class_sort_group") if comp.get("class_sort_group") is not None else 99),
        int(comp.get("class_rank") or 999999),
        -_as_float(comp.get("job_prevalence")),
        -int(comp.get("n_source_jobs") or 0),
        -_as_float(comp.get("grounding_score", comp.get("grounding_score_preview", 0.0))),
        -public_future_weight(comp),
        -_as_float(comp.get("priority_score")),
        str(comp.get("id", "")),
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def csv_rows(competencies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for c in competencies:
        rows.append({
            "pipeline_priority_rank": c.get("pipeline_priority_rank"),
            "official_core_rank": c.get("official_core_rank") or "",
            "competency_id": c.get("id", ""),
            "competency_class": c.get("competency_class", ""),
            "competency_class_label": c.get("competency_class_label", ""),
            "support_tier": c.get("support_tier", ""),
            "support_tier_label": c.get("support_tier_label", ""),
            "official_core_eligible": c.get("official_core_eligible", False),
            "official_core_blockers": " | ".join(c.get("official_core_blockers") or []),
            "recommendation_status": c.get("recommendation_status", ""),
            "official_readiness_blockers": " | ".join(c.get("official_readiness_blockers") or []),
            "class_rank": c.get("class_rank", ""),
            "scientific_flags": " | ".join(c.get("scientific_flags") or []),
            "repair_decision": c.get("repair_decision", ""),
            "repair_flags": " | ".join(c.get("repair_flags") or []),
            "curriculum_dedup_group": c.get("curriculum_dedup_group", ""),
            "recovered_from_failed": c.get("recovered_from_failed", False),
            "source_failure_reason": c.get("source_failure_reason", ""),
            "n_source_jobs": c.get("n_source_jobs", ""),
            "eligible_job_denominator": c.get("eligible_job_denominator", ""),
            "demand_denominator": c.get("demand_denominator", ""),
            "job_prevalence_pct": c.get("job_prevalence_pct", ""),
            "corpus_country": c.get("corpus_country", ""),
            "corpus_language": c.get("corpus_language", ""),
            "corpus_validity_status": c.get("corpus_validity_status", ""),
            "grounding_score": c.get("grounding_score", ""),
            "future_weight": c.get("future_weight", ""),
            "future_weight_source": c.get("future_weight_source", LLM_FUTURE_SOURCE),
            "public_future_weight": public_future_weight(c),
            "priority_score": c.get("priority_score", ""),
            "demand_score": c.get("demand_score", ""),
            "occupation_mapping_status": c.get("occupation_mapping_status", ""),
            "occupation_mapping_label": c.get("occupation_mapping_label", ""),
            "cluster_validity_status": c.get("cluster_validity_status", ""),
            "cluster_validity_flags": " | ".join(c.get("cluster_validity_flags") or []),
            "technical_theme_id": c.get("technical_theme_id", ""),
            "technical_theme_label": c.get("technical_theme_label", ""),
            "theme_evidence_status": c.get("theme_evidence_status", ""),
            "theme_recommendation_status": c.get("theme_recommendation_status", ""),
            "not_recommended_reason": c.get("not_recommended_reason", ""),
            "statement": c.get("statement", ""),
            "related_skills": " | ".join(str(s) for s in _as_list(c.get("related_skills"))),
        })
    return rows


def build_report(
    run_dir: Path,
    jobs_metadata: Path | None,
    competencies: list[dict[str, Any]],
    denominator: int,
    min_official_jobs: int,
    min_grounding: float,
    strong_min_jobs: int,
    moderate_min_jobs: int,
    corpus_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    corpus_context = corpus_context or {}
    class_dist = Counter(c.get("competency_class") for c in competencies)
    tier_dist = Counter(c.get("support_tier") for c in competencies)
    occ_dist = Counter(c.get("occupation_mapping_status") for c in competencies)
    status_dist = Counter(c.get("recommendation_status") for c in competencies)
    blocker_dist: Counter = Counter()
    readiness_blocker_dist: Counter = Counter()
    flag_dist: Counter = Counter()
    repair_decision_dist: Counter = Counter()
    repair_flag_dist: Counter = Counter()
    for c in competencies:
        blocker_dist.update(c.get("official_core_blockers") or [])
        readiness_blocker_dist.update(c.get("official_readiness_blockers") or [])
        flag_dist.update(c.get("scientific_flags") or [])
        repair_decision_dist.update([c.get("repair_decision") or "unrepaired"])
        repair_flag_dist.update(c.get("repair_flags") or [])

    official = sorted(
        [c for c in competencies if c.get("official_core_eligible")],
        key=lambda c: int(c.get("official_core_rank") or 999999),
    )
    top_pipeline = competencies[:10]

    def _summary_row(c: dict[str, Any]) -> dict[str, Any]:
        return {
            "rank": c.get("pipeline_priority_rank"),
            "official_core_rank": c.get("official_core_rank"),
            "id": c.get("id"),
            "class": c.get("competency_class"),
            "jobs": c.get("n_source_jobs"),
            "prevalence_pct": c.get("job_prevalence_pct"),
            "grounding": c.get("grounding_score"),
            "future_weight": c.get("future_weight"),
            "future_weight_source": c.get("future_weight_source"),
            "public_future_weight": public_future_weight(c),
            "cluster_validity_status": c.get("cluster_validity_status"),
            "theme_evidence_status": c.get("theme_evidence_status"),
            "official_core_eligible": c.get("official_core_eligible"),
            "recommendation_status": c.get("recommendation_status"),
            "blockers": c.get("official_core_blockers"),
            "official_readiness_blockers": c.get("official_readiness_blockers"),
            "statement": c.get("statement"),
        }

    return {
        "schema_version": SPRINT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "jobs_metadata": str(jobs_metadata) if jobs_metadata else "",
        "eligible_job_denominator": denominator,
        "demand_denominator": denominator,
        "corpus_context": corpus_context,
        "gates": {
            "official_core_min_source_jobs": min_official_jobs,
            "official_core_min_grounding": min_grounding,
            "strong_support_min_jobs": strong_min_jobs,
            "moderate_support_min_jobs": moderate_min_jobs,
            "official_core_requires": [
                "competency_class == core_technical",
                f"n_source_jobs >= {min_official_jobs}",
                f"grounding_score >= {min_grounding}",
                "occupation_mapping_status != unmapped",
                "cluster_validity_status not in split_required/mixed_domain/support_process/out_of_domain",
                "theme_recommendation_status != not_recommended_from_this_corpus",
            ],
            "official_ready_requires": [
                "corpus_validity_status == valid_for_indonesian_claim",
                "occupation_mapping_status == official_or_template",
                "expert_review_passed == true",
                "stability_passed == true",
            ],
        },
        "distributions": {
            "competency_class": dict(class_dist),
            "support_tier": dict(tier_dist),
            "occupation_mapping_status": dict(occ_dist),
            "recommendation_status": dict(status_dist),
            "official_core_blockers": dict(blocker_dist),
            "official_readiness_blockers": dict(readiness_blocker_dist),
            "scientific_flags": dict(flag_dist),
            "repair_decision": dict(repair_decision_dist),
            "repair_flags": dict(repair_flag_dist),
        },
        "official_core_eligible_count": len(official),
        "official_core_candidate_ids": [c.get("id") for c in official],
        "top10_pipeline_priority": [_summary_row(c) for c in top_pipeline],
        "top10_official_core_demand_first": [_summary_row(c) for c in official[:10]],
        "interpretation": [
            "Pipeline priority remains available as the original demand/grounding/future ranking.",
            "Official-core candidates are a stricter subset intended for curriculum recommendation review.",
            "Recommendation status is capped by corpus validity; US/English engineering corpora cannot become official_ready.",
            "Support/process/out-of-domain recommendations are not discarded; they are surfaced as different recommendation classes.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="results/competency_v2_pipeline_n1k_openrouter_gpt54")
    parser.add_argument("--jobs-metadata", default="")
    parser.add_argument("--min-official-jobs", type=int, default=10)
    parser.add_argument("--min-grounding", type=float, default=0.80)
    parser.add_argument("--strong-min-jobs", type=int, default=10)
    parser.add_argument("--moderate-min-jobs", type=int, default=5)
    parser.add_argument("--corpus-report", default="",
                        help="Path to corpus_lineage_report.json. If omitted, tries the run directory and pipeline_report.")
    parser.add_argument("--no-inplace", action="store_true",
                        help="Do not merge enriched fields back into competencies.json.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    report_path = run_dir / "pipeline_report.json"
    pipeline_report = _load_json(report_path, {})
    competencies_path = run_dir / "competencies.json"
    competencies = _load_json(competencies_path, [])
    if not isinstance(competencies, list):
        raise ValueError("competencies.json must contain a list")

    jobs_metadata = resolve_jobs_metadata(args.jobs_metadata, pipeline_report)
    corpus_report = resolve_corpus_report(args.corpus_report, run_dir, pipeline_report)
    corpus_context = load_corpus_context(corpus_report)
    denominator, _n_jobs_scanned = load_job_denominator(jobs_metadata)
    if denominator <= 0:
        source_job_ids = {
            str(j)
            for c in competencies
            for j in _as_list(c.get("source_job_ids"))
            if str(j).strip()
        }
        denominator = len(source_job_ids)

    enriched = enrich_competencies(
        competencies,
        denominator=denominator,
        min_official_jobs=args.min_official_jobs,
        min_grounding=args.min_grounding,
        strong_min_jobs=args.strong_min_jobs,
        moderate_min_jobs=args.moderate_min_jobs,
        corpus_context=corpus_context,
    )

    report = build_report(
        run_dir=run_dir,
        jobs_metadata=jobs_metadata,
        competencies=enriched,
        denominator=denominator,
        min_official_jobs=args.min_official_jobs,
        min_grounding=args.min_grounding,
        strong_min_jobs=args.strong_min_jobs,
        moderate_min_jobs=args.moderate_min_jobs,
        corpus_context=corpus_context,
    )

    enriched_path = run_dir / "competencies_sprint1_enriched.json"
    _write_json(enriched_path, enriched)
    _write_json(run_dir / "scientific_fix_report.json", report)
    local_corpus_report = run_dir / "corpus_lineage_report.json"
    if corpus_report and corpus_report.exists():
        corpus_data = _load_json(corpus_report, {})
        if isinstance(corpus_data, dict) and corpus_data:
            _write_json(local_corpus_report, corpus_data)

    rows = csv_rows(enriched)
    fields = [
        "pipeline_priority_rank",
        "official_core_rank",
        "competency_id",
        "competency_class",
        "competency_class_label",
        "support_tier",
        "support_tier_label",
        "official_core_eligible",
        "official_core_blockers",
        "recommendation_status",
        "official_readiness_blockers",
        "class_rank",
        "scientific_flags",
        "repair_decision",
        "repair_flags",
        "curriculum_dedup_group",
        "recovered_from_failed",
        "source_failure_reason",
        "n_source_jobs",
        "eligible_job_denominator",
        "demand_denominator",
        "job_prevalence_pct",
        "corpus_country",
        "corpus_language",
        "corpus_validity_status",
        "grounding_score",
        "future_weight",
        "future_weight_source",
        "public_future_weight",
        "priority_score",
        "demand_score",
        "occupation_mapping_status",
        "occupation_mapping_label",
        "cluster_validity_status",
        "cluster_validity_flags",
        "technical_theme_id",
        "technical_theme_label",
        "theme_evidence_status",
        "theme_recommendation_status",
        "not_recommended_reason",
        "statement",
        "related_skills",
    ]
    write_csv(run_dir / "scientific_fix_competencies.csv", rows, fields)
    official_rows = [r for r in rows if str(r.get("official_core_eligible")).lower() == "true"]
    official_rows.sort(key=lambda r: int(r.get("official_core_rank") or 999999))
    write_csv(run_dir / "official_core_recommendations.csv", official_rows, fields)
    for class_name in CLASS_LABELS:
        class_rows = [r for r in rows if r.get("competency_class") == class_name]
        class_rows.sort(key=lambda r: int(r.get("class_rank") or 999999))
        write_csv(run_dir / f"ranked_{class_name}.csv", class_rows, fields)
        write_csv(run_dir / f"ranked_{class_name}_repaired.csv", class_rows, fields)

    if not args.no_inplace:
        backup_path = run_dir / "competencies_before_sprint1.json"
        if not backup_path.exists():
            _write_json(backup_path, competencies)
        _write_json(competencies_path, enriched)

        if isinstance(pipeline_report, dict):
            pipeline_report.setdefault("stages", {})["scientific_sprint1"] = {
                "schema_version": SPRINT_SCHEMA_VERSION,
                "eligible_job_denominator": denominator,
                "official_core_eligible_count": report["official_core_eligible_count"],
                "competency_class_distribution": report["distributions"]["competency_class"],
                "support_tier_distribution": report["distributions"]["support_tier"],
                "occupation_mapping_distribution": report["distributions"]["occupation_mapping_status"],
                "recommendation_status_distribution": report["distributions"]["recommendation_status"],
                "corpus_validity_status": corpus_context.get("corpus_validity_status"),
                "corpus_country": corpus_context.get("corpus_country"),
                "corpus_language": corpus_context.get("corpus_language"),
                "contract": "official-core candidates are a demand-first subset; support/process/out-of-domain items are surfaced separately",
            }
            if local_corpus_report.exists():
                pipeline_report["corpus_lineage_report"] = str(local_corpus_report)
                pipeline_report.setdefault("stages", {})["corpus_lineage_audit"] = {
                    "schema_version": corpus_context.get("schema_version", "corpus_lineage_audit_v1"),
                    "dataset_id": corpus_context.get("corpus_dataset_id"),
                    "report_path": str(local_corpus_report),
                    "corpus_validity_status": corpus_context.get("corpus_validity_status"),
                    "corpus_country": corpus_context.get("corpus_country"),
                    "corpus_language": corpus_context.get("corpus_language"),
                    "metadata_unique_job_ids": denominator,
                }
            _write_json(report_path, pipeline_report)

    print(f"[OK] Sprint 1 scientific fields written to: {enriched_path}")
    print(f"     Denominator: {denominator} eligible job postings")
    print(f"     Corpus status: {corpus_context.get('corpus_validity_status')} ({corpus_context.get('corpus_country')}/{corpus_context.get('corpus_language')})")
    print(f"     Official-core candidates: {report['official_core_eligible_count']} / {len(enriched)}")
    print(f"     Class distribution: {dict(report['distributions']['competency_class'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
