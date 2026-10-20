"""Read-only scientific audits for the competency v2 pipeline.

The functions in this module deliberately avoid mutating Phase-1 or v2
artifacts. They make the evidence chain observable so downstream ranking can
say "not recommended from this corpus" instead of forcing weak evidence into a
competency.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

from extraction_typing import (
    EXTRACTION_TYPES,
    classify_extraction_span,
    normalize_extraction_text,
)


SCIENTIFIC_AUDIT_SCHEMA_VERSION = "2026-06-11.evidence_first.v1"

CLUSTER_VALID = "valid_cluster"
CLUSTER_SPLIT = "split_required"
CLUSTER_MIXED = "mixed_domain"
CLUSTER_SUPPORT = "support_process_cluster"
CLUSTER_OUT = "out_of_domain_cluster"
CLUSTER_WEAK = "weak_low_frequency_theme"

THEME_STRONG = "strong_direct_evidence"
THEME_MODERATE = "moderate_direct_evidence"
THEME_WEAK = "weak_or_scattered_evidence"
THEME_TOOL_ONLY = "tool_mention_only"
THEME_NOT_DETECTED = "not_detected"
THEME_MIXED = "mixed_or_ambiguous"
THEME_OUT = "out_of_domain"
THEME_NOT_RECOMMENDED = "not_recommended_from_this_corpus"


THEME_PROBES: list[dict[str, Any]] = [
    {"id": "python", "label": "Python", "patterns": [r"\bpython\b"], "assessable": ["build", "script", "automate", "develop", "maintain", "data"]},
    {"id": "rest_api", "label": "REST/API", "patterns": [r"\brest\b", r"\bapi(s)?\b", r"\bgraphql\b"], "assessable": ["design", "develop", "integrate", "service", "backend"]},
    {"id": "kubernetes", "label": "Kubernetes", "patterns": [r"\bkubernetes\b", r"\bk8s\b"], "assessable": ["deploy", "operate", "orchestrate", "container"]},
    {"id": "data_pipelines", "label": "Data pipelines", "patterns": [r"\bdata pipeline", r"\betl\b", r"\bairflow\b", r"\bspark\b"], "assessable": ["build", "maintain", "process", "orchestrate"]},
    {"id": "testing", "label": "Automated testing", "patterns": [r"\btesting\b", r"\btest automation\b", r"\bpytest\b", r"\bjest\b", r"\bcypress\b", r"\bselenium\b"], "assessable": ["automate", "validate", "quality", "release"]},
    {"id": "debugging", "label": "Debugging/troubleshooting", "patterns": [r"\bdebug", r"\btroubleshoot", r"\broot cause"], "assessable": ["diagnose", "resolve", "investigate", "production"]},
    {"id": "cloud_iac", "label": "Cloud/IaC", "patterns": [r"\baws\b", r"\bazure\b", r"\bgcp\b", r"\bterraform\b", r"\binfrastructure as code\b", r"\biac\b"], "assessable": ["deploy", "provision", "operate", "infrastructure"]},
    {"id": "nlp_genai", "label": "NLP/GenAI", "patterns": [r"\bnlp\b", r"\bgenai\b", r"\bllm\b", r"\blarge language"], "assessable": ["build", "evaluate", "integrate", "model"]},
    {"id": "architecture_review", "label": "Architecture/design review", "patterns": [r"\barchitecture\b", r"\bdesign review\b", r"\bsystem design\b"], "assessable": ["review", "design", "evaluate", "tradeoff"]},
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys or ["metric", "value"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _iter_jsonl(path: Path | None) -> Iterable[dict[str, Any]]:
    if not path or not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value
            except json.JSONDecodeError:
                yield {"_parse_error": True, "raw_line": line[:300]}


def _job_id_from_sentence_id(sentence_id: str) -> str:
    sid = str(sentence_id or "").strip()
    if not sid:
        return ""
    return re.sub(r"_\d{1,6}$", "", sid)


def _job_id(row: dict[str, Any]) -> str:
    return str(row.get("job_id") or row.get("jobId") or row.get("id") or "").strip()


def _sentence_id(row: dict[str, Any]) -> str:
    return str(row.get("sentence_id") or row.get("sentenceId") or "").strip()


def _source_text(row: dict[str, Any]) -> str:
    for key in ("skill", "knowledge", "text", "span", "skill_span", "knowledge_span"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def audit_extraction_quality(phase1_dir: Path, output_dir: Path) -> dict[str, Any]:
    _ensure_dir(output_dir)
    records: list[dict[str, Any]] = []
    for filename, source_field, text_field in [
        ("advanced_skills.csv", "skill", "skill"),
        ("advanced_knowledge.csv", "knowledge", "knowledge"),
    ]:
        for row in _read_csv(phase1_dir / filename):
            text = row.get(text_field) or row.get("text") or ""
            category, flags = classify_extraction_span(
                text,
                source_field=source_field,
                sentence_text=row.get("sentence_text") or "",
            )
            records.append({
                "source_file": filename,
                "source_field": source_field,
                "job_id": _job_id(row),
                "sentence_id": _sentence_id(row),
                "raw_span": text,
                "normalized_span": normalize_extraction_text(text),
                "span_token_count": len(re.findall(r"[A-Za-z0-9+#.]+", str(text or ""))),
                "post_extraction_type": category,
                "flags": " | ".join(flags),
                "original_type": row.get("type") or source_field,
            })

    by_type = Counter(r["post_extraction_type"] for r in records)
    by_source = Counter((r["source_field"], r["post_extraction_type"]) for r in records)
    flag_counts: Counter[str] = Counter()
    for rec in records:
        flag_counts.update([f.strip() for f in str(rec["flags"]).split("|") if f.strip()])
    suspicious = [
        r for r in records
        if r["post_extraction_type"] not in {"hard_technical", "technical_process"}
    ]
    top_dropped = Counter(
        r["normalized_span"].lower() for r in suspicious if r["normalized_span"]
    ).most_common(50)
    summary_rows = [
        {"metric": "total_phase1_spans", "value": len(records)},
        *({"metric": f"type.{k}", "value": v} for k, v in sorted(by_type.items())),
        *({"metric": f"flag.{k}", "value": v} for k, v in sorted(flag_counts.items())),
    ]
    detail_rows = records
    _write_csv(output_dir / "extraction_quality_audit.csv", detail_rows, [
        "source_file", "source_field", "job_id", "sentence_id", "raw_span", "normalized_span",
        "span_token_count", "post_extraction_type", "flags", "original_type",
    ])
    report = {
        "schema_version": SCIENTIFIC_AUDIT_SCHEMA_VERSION,
        "phase1_dir": str(phase1_dir),
        "n_phase1_spans": len(records),
        "type_distribution": dict(by_type),
        "source_type_distribution": {f"{src}.{typ}": n for (src, typ), n in by_source.items()},
        "flag_distribution": dict(flag_counts),
        "n_not_hard_technical": sum(v for k, v in by_type.items() if k != "hard_technical"),
        "top_suspicious_spans": [{"span": span, "count": count} for span, count in top_dropped],
        "csv_path": str(output_dir / "extraction_quality_audit.csv"),
        "summary_rows": summary_rows,
    }
    _write_json(output_dir / "extraction_quality_audit.json", report)
    return report


def audit_sentence_filter(prepared_dir: Path, skill_llm_input: Path | None, output_dir: Path) -> dict[str, Any]:
    _ensure_dir(output_dir)
    sentence_files = [
        prepared_dir / "jobs_sentences.csv",
        prepared_dir / "job_sentences_full.csv",
        prepared_dir / "sentences.csv",
    ]
    filtered_files = [
        prepared_dir / "jobs_sentences_filtered.csv",
        prepared_dir / "skill_llm_sentences.csv",
        prepared_dir / "sentences_filtered.csv",
    ]
    prepared_rows: list[dict[str, str]] = []
    filtered_rows: list[dict[str, str]] = []
    prepared_file = ""
    filtered_file = ""
    for path in sentence_files:
        rows = _read_csv(path)
        if rows:
            prepared_rows = rows
            prepared_file = str(path)
            break
    for path in filtered_files:
        rows = _read_csv(path)
        if rows:
            filtered_rows = rows
            filtered_file = str(path)
            break

    input_rows = list(_iter_jsonl(skill_llm_input)) if skill_llm_input and skill_llm_input.exists() else []
    prepared_jobs = {_job_id(r) for r in prepared_rows if _job_id(r)}
    filtered_jobs = {_job_id(r) for r in filtered_rows if _job_id(r)}
    input_jobs = {
        _job_id(r) or _job_id_from_sentence_id(_sentence_id(r))
        for r in input_rows
        if (_job_id(r) or _sentence_id(r))
    }
    input_jobs.discard("")
    lost_prepared_to_filtered = sorted(prepared_jobs - filtered_jobs)
    lost_prepared_to_input = sorted(prepared_jobs - input_jobs) if input_jobs else []
    report = {
        "schema_version": SCIENTIFIC_AUDIT_SCHEMA_VERSION,
        "prepared_dir": str(prepared_dir),
        "prepared_sentence_file": prepared_file,
        "filtered_sentence_file": filtered_file,
        "skill_llm_input_path": str(skill_llm_input or ""),
        "prepared_sentence_count": len(prepared_rows),
        "prepared_unique_job_ids": len(prepared_jobs),
        "filtered_sentence_count": len(filtered_rows),
        "filtered_unique_job_ids": len(filtered_jobs),
        "skill_llm_input_records": len(input_rows),
        "skill_llm_input_unique_job_ids": len(input_jobs),
        "prepared_to_filtered_sentence_retention": _ratio(len(filtered_rows), len(prepared_rows)),
        "prepared_to_filtered_job_retention": _ratio(len(filtered_jobs), len(prepared_jobs)),
        "prepared_to_skill_llm_input_job_retention": _ratio(len(input_jobs), len(prepared_jobs)) if input_jobs else 0.0,
        "n_jobs_lost_all_sentences_prepared_to_filtered": len(lost_prepared_to_filtered),
        "n_jobs_lost_all_sentences_prepared_to_skill_llm_input": len(lost_prepared_to_input),
        "sample_jobs_lost_all_sentences": lost_prepared_to_filtered[:50],
    }
    rows = [
        {"stage": "prepared_sentences", "records": len(prepared_rows), "unique_jobs": len(prepared_jobs), "path": prepared_file},
        {"stage": "filtered_sentences", "records": len(filtered_rows), "unique_jobs": len(filtered_jobs), "path": filtered_file},
        {"stage": "skill_llm_input", "records": len(input_rows), "unique_jobs": len(input_jobs), "path": str(skill_llm_input or "")},
    ]
    _write_json(output_dir / "sentence_filter_audit.json", report)
    _write_csv(output_dir / "sentence_filter_audit.csv", rows, ["stage", "records", "unique_jobs", "path"])
    return report


def audit_domain_filter(phase1_dir: Path, output_dir: Path) -> dict[str, Any]:
    _ensure_dir(output_dir)
    rows = []
    for filename, source_field, text_field in [
        ("advanced_skills.csv", "skill", "skill"),
        ("advanced_knowledge.csv", "knowledge", "knowledge"),
    ]:
        for row in _read_csv(phase1_dir / filename):
            text = row.get(text_field) or row.get("text") or ""
            category, flags = classify_extraction_span(text, source_field=source_field)
            domain_status = "rpl_it_candidate"
            if category == "out_of_domain":
                domain_status = "out_of_domain_visible_signal"
            elif category in {"support_soft_skill", "personality_trait", "credential", "education_requirement", "invalid_fragment"}:
                domain_status = "domain_excluded"
            rows.append({
                "job_id": _job_id(row),
                "sentence_id": _sentence_id(row),
                "source_field": source_field,
                "span": text,
                "normalized_span": normalize_extraction_text(text),
                "post_extraction_type": category,
                "domain_status": domain_status,
                "flags": " | ".join(flags),
            })
    _write_csv(output_dir / "domain_filter_audit.csv", rows, [
        "job_id", "sentence_id", "source_field", "span", "normalized_span",
        "post_extraction_type", "domain_status", "flags",
    ])
    summary = {
        "schema_version": SCIENTIFIC_AUDIT_SCHEMA_VERSION,
        "domain_status_distribution": dict(Counter(r["domain_status"] for r in rows)),
        "post_extraction_type_distribution": dict(Counter(r["post_extraction_type"] for r in rows)),
        "csv_path": str(output_dir / "domain_filter_audit.csv"),
    }
    return summary


def _ratio(n: int | float, d: int | float) -> float:
    try:
        return round(float(n) / float(d), 6) if float(d) else 0.0
    except Exception:
        return 0.0


def _path(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _job_set_from_phase1(phase1_dir: Path) -> set[str]:
    ids: set[str] = set()
    for filename in ("advanced_skills.csv", "advanced_knowledge.csv"):
        for row in _read_csv(phase1_dir / filename):
            jid = _job_id(row)
            if jid:
                ids.add(jid)
    return ids


def _jsonl_profile(path: Path | None) -> dict[str, Any]:
    rows = list(_iter_jsonl(path)) if path and path.exists() else []
    jobs = {
        _job_id(row) or _job_id_from_sentence_id(_sentence_id(row))
        for row in rows
        if (_job_id(row) or _sentence_id(row))
    }
    jobs.discard("")
    parse_errors = sum(1 for row in rows if row.get("_parse_error"))
    extraction_records = 0
    skill_count = 0
    knowledge_count = 0
    raw_output_failures = 0
    for row in rows:
        skills = row.get("SKILL") or row.get("skills") or []
        knowledge = row.get("KNOWLEDGE") or row.get("knowledge") or []
        if isinstance(skills, list):
            skill_count += len(skills)
        if isinstance(knowledge, list):
            knowledge_count += len(knowledge)
        if skills or knowledge:
            extraction_records += 1
        if row.get("raw_output") and not skills and not knowledge:
            raw_output_failures += 1
    return {
        "records": len(rows),
        "unique_job_ids": len(jobs),
        "parse_errors": parse_errors,
        "records_with_extractions": extraction_records,
        "skill_spans": skill_count,
        "knowledge_spans": knowledge_count,
        "raw_output_parse_failures": raw_output_failures,
    }


def _date_anomaly_reasons(corpus_report: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    date_range = corpus_report.get("date_range") or {}
    min_date = str(date_range.get("min") or date_range.get("min_date") or "")
    max_date = str(date_range.get("max") or date_range.get("max_date") or "")
    for label, value in [("min", min_date), ("max", max_date)]:
        if not value:
            continue
        try:
            parsed = datetime.fromisoformat(value[:10]).date()
        except Exception:
            reasons.append(f"date_{label}_unparseable")
            continue
        if label == "min" and parsed < date(2000, 1, 1):
            reasons.append("date_min_before_2000")
        if label == "max" and parsed > date.today():
            reasons.append("date_max_after_today")
    return reasons


def build_stage_coverage_report(
    *,
    output_dir: Path,
    corpus_report: dict[str, Any] | None,
    prepared_dir: Path | None,
    phase1_dir: Path,
    skill_llm_input: Path | None,
    skill_llm_output: Path | None,
    competencies: list[dict[str, Any]],
    failed_competencies: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    _ensure_dir(output_dir)
    corpus_report = corpus_report or {}
    sentence_report = audit_sentence_filter(prepared_dir, skill_llm_input, output_dir) if prepared_dir else {}
    input_profile = _jsonl_profile(skill_llm_input)
    output_profile = _jsonl_profile(skill_llm_output)
    phase1_jobs = _job_set_from_phase1(phase1_dir)
    final_jobs = {
        str(j).strip()
        for comp in competencies
        for j in (comp.get("source_job_ids") or [])
        if str(j).strip()
    }
    failed_jobs = {
        str(j).strip()
        for comp in (failed_competencies or [])
        for j in (comp.get("source_job_ids") or [])
        if str(j).strip()
    }
    denominator = int(
        corpus_report.get("prepared_valid_description_unique_job_ids")
        or corpus_report.get("metadata_unique_job_ids")
        or 0
    )
    stages = [
        {"stage": "raw_jobs", "records": corpus_report.get("raw_row_count", ""), "unique_jobs": corpus_report.get("raw_unique_job_ids", ""), "denominator_role": "raw corpus"},
        {"stage": "metadata_jobs", "records": corpus_report.get("metadata_row_count", ""), "unique_jobs": corpus_report.get("metadata_unique_job_ids", ""), "denominator_role": "metadata denominator fallback"},
        {"stage": "prepared_sentences", "records": sentence_report.get("prepared_sentence_count", corpus_report.get("prepared_sentence_count", "")), "unique_jobs": sentence_report.get("prepared_unique_job_ids", corpus_report.get("prepared_sentence_unique_job_ids", "")), "denominator_role": "sentence preprocessing"},
        {"stage": "filtered_sentences", "records": sentence_report.get("filtered_sentence_count", ""), "unique_jobs": sentence_report.get("filtered_unique_job_ids", ""), "denominator_role": "Skill-LLM candidate sentences"},
        {"stage": "skill_llm_input", "records": input_profile.get("records", 0), "unique_jobs": input_profile.get("unique_job_ids", 0), "denominator_role": "LLM extraction input"},
        {"stage": "skill_llm_output", "records": output_profile.get("records", 0), "unique_jobs": output_profile.get("unique_job_ids", 0), "denominator_role": "LLM extraction output"},
        {"stage": "phase1_items", "records": "", "unique_jobs": len(phase1_jobs), "denominator_role": "cluster evidence input"},
        {"stage": "failed_competency_source_jobs", "records": len(failed_competencies or []), "unique_jobs": len(failed_jobs), "denominator_role": "dropped generated evidence"},
        {"stage": "final_competency_source_jobs", "records": len(competencies), "unique_jobs": len(final_jobs), "denominator_role": "final recommendation support"},
    ]
    blockers: list[str] = []
    if not phase1_jobs:
        blockers.append("missing_phase1_job_ids")
    if output_profile.get("records") and output_profile.get("records_with_extractions", 0) == 0:
        blockers.append("skill_llm_output_has_no_extractions")
    if denominator and len(phase1_jobs) / denominator < 0.80:
        blockers.append("phase1_job_coverage_below_0.80")
    if sentence_report.get("n_jobs_lost_all_sentences_prepared_to_filtered", 0):
        blockers.append("jobs_lost_all_sentences_in_filter")
    blockers.extend(_date_anomaly_reasons(corpus_report))
    report = {
        "schema_version": SCIENTIFIC_AUDIT_SCHEMA_VERSION,
        "labor_market_denominator": denominator,
        "denominator_policy": "prepared_valid_description_jobs_if_available_else_metadata_unique_jobs",
        "stage_rows": stages,
        "skill_llm_input_profile": input_profile,
        "skill_llm_output_profile": output_profile,
        "sentence_filter_summary": sentence_report,
        "final_source_job_coverage_of_denominator": _ratio(len(final_jobs), denominator),
        "phase1_job_coverage_of_denominator": _ratio(len(phase1_jobs), denominator),
        "coverage_blockers": sorted(set(blockers)),
        "date_sanity_reasons": _date_anomaly_reasons(corpus_report),
    }
    _write_csv(output_dir / "stage_coverage_report.csv", stages, ["stage", "records", "unique_jobs", "denominator_role"])
    _write_json(output_dir / "stage_coverage_report.json", report)
    return report


DOMAIN_FAMILIES: list[tuple[str, re.Pattern[str]]] = [
    ("cloud_iac", re.compile(r"\b(aws|azure|gcp|cloud|terraform|iac|infrastructure as code|kubernetes|docker)\b", re.I)),
    ("software_dev_api", re.compile(r"\b(api|rest|graphql|backend|microservice|software|programming|java\b|python|node|git)\b", re.I)),
    ("frontend_mobile_ui", re.compile(r"\b(frontend|front-end|react|angular|vue|mobile|ios|android|swift|kotlin|accessibility|ui|ux)\b", re.I)),
    ("data_ai", re.compile(r"\b(data|etl|pipeline|sql|database|tableau|power bi|machine learning|ai\b|nlp|genai|llm)\b", re.I)),
    ("security", re.compile(r"\b(security|oauth|saml|iam|vulnerability|compliance|devsecops)\b", re.I)),
    ("testing_devops", re.compile(r"\b(test|testing|qa|ci/cd|devops|debug|troubleshoot|observability|monitoring|sre)\b", re.I)),
    ("support_process", re.compile(r"\b(agile|scrum|leadership|mentor|communication|collaboration|stakeholder|problem solving|planning)\b", re.I)),
    ("business_platform", re.compile(r"\b(salesforce|sharepoint|power apps|crm|sap|user research)\b", re.I)),
    ("physical_engineering", re.compile(r"\b(cad|drafting|asic|bim|gis|meica|mechanical|electrical)\b", re.I)),
    ("clinical_social", re.compile(r"\b(clinical|safeguarding|social work|mental capacity)\b", re.I)),
]


def _family_hits(text: str) -> set[str]:
    return {family for family, pattern in DOMAIN_FAMILIES if pattern.search(text)}


def audit_cluster_validity(clusters: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    _ensure_dir(output_dir)
    rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    low_freq_rows: list[dict[str, Any]] = []
    for cluster in clusters:
        items = [str(x) for x in (cluster.get("items") or [])]
        blob = " ".join([str(cluster.get("summary_label") or ""), " ".join(cluster.get("top_terms") or []), " ".join(items)])
        families = sorted(_family_hits(blob))
        cohesion = float(cluster.get("cohesion_score") or 0.0)
        n_items = int(cluster.get("n_items") or len(items))
        n_jobs = int(cluster.get("n_unique_jobs") or 0)
        flags: list[str] = []
        if n_items >= 35:
            flags.append("large_cluster")
        if cohesion and cohesion < 0.62:
            flags.append("low_cohesion")
        if len(families) >= 4:
            flags.append("many_domain_families")
        if any(f in families for f in ("physical_engineering", "clinical_social")) and len(families) >= 2:
            flags.append("cross_domain_contamination")
        if n_jobs < 3:
            flags.append("weak_low_frequency")

        status = CLUSTER_VALID
        if "clinical_social" in families or ("physical_engineering" in families and not any(f in families for f in ("software_dev_api", "data_ai", "cloud_iac"))):
            status = CLUSTER_OUT
        elif families and set(families).issubset({"support_process"}):
            status = CLUSTER_SUPPORT
        elif "cross_domain_contamination" in flags:
            status = CLUSTER_MIXED
        elif "large_cluster" in flags or "low_cohesion" in flags or "many_domain_families" in flags:
            status = CLUSTER_SPLIT
        elif "weak_low_frequency" in flags:
            status = CLUSTER_WEAK

        row = {
            "cluster_id": cluster.get("id", ""),
            "summary_label": cluster.get("summary_label", ""),
            "cluster_validity_status": status,
            "cluster_validity_flags": " | ".join(sorted(set(flags))),
            "domain_families": " | ".join(families),
            "primary_domain_family": families[0] if families else "",
            "n_items": n_items,
            "n_unique_jobs": n_jobs,
            "cohesion_score": cohesion,
            "sample_items": " || ".join(items[:8]),
        }
        rows.append(row)
        if status in {CLUSTER_SPLIT, CLUSTER_MIXED}:
            split_rows.append(row)
        if status == CLUSTER_WEAK:
            low_freq_rows.append(row)

    status_dist = Counter(r["cluster_validity_status"] for r in rows)
    report = {
        "schema_version": SCIENTIFIC_AUDIT_SCHEMA_VERSION,
        "n_clusters": len(rows),
        "status_distribution": dict(status_dist),
        "n_split_candidates": len(split_rows),
        "n_low_frequency_themes": len(low_freq_rows),
        "split_candidate_ids": [r["cluster_id"] for r in split_rows],
        "cluster_validity_by_id": {
            r["cluster_id"]: {
                "cluster_validity_status": r["cluster_validity_status"],
                "cluster_validity_flags": [x.strip() for x in r["cluster_validity_flags"].split("|") if x.strip()],
                "domain_families": [x.strip() for x in r["domain_families"].split("|") if x.strip()],
                "primary_domain_family": r["primary_domain_family"],
            }
            for r in rows
        },
    }
    fields = [
        "cluster_id", "summary_label", "cluster_validity_status", "cluster_validity_flags",
        "domain_families", "primary_domain_family", "n_items", "n_unique_jobs",
        "cohesion_score", "sample_items",
    ]
    _write_csv(output_dir / "cluster_validity_report.csv", rows, fields)
    _write_csv(output_dir / "cluster_split_candidates.csv", split_rows, fields)
    _write_csv(output_dir / "low_frequency_theme_audit.csv", low_freq_rows, fields)
    _write_json(output_dir / "cluster_validity_report.json", report)
    return report


def annotate_cluster_validity(competencies: list[Any], cluster_report: dict[str, Any]) -> None:
    by_id = cluster_report.get("cluster_validity_by_id") or {}
    for comp in competencies:
        cluster_id = getattr(comp, "cluster_id", None)
        if cluster_id is None and isinstance(comp, dict):
            cluster_id = comp.get("cluster_id")
        audit = by_id.get(str(cluster_id or "")) or {}
        if not audit:
            continue
        if isinstance(comp, dict):
            comp["cluster_validity_status"] = audit.get("cluster_validity_status", "")
            comp["cluster_validity_flags"] = audit.get("cluster_validity_flags", [])
        else:
            comp.cluster_validity_status = audit.get("cluster_validity_status", "")
            comp.cluster_validity_flags = list(audit.get("cluster_validity_flags") or [])


def _theme_hits(theme: dict[str, Any], text: str) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in theme.get("patterns") or [])


def _has_assessable_activity(theme: dict[str, Any], text: str) -> bool:
    words = theme.get("assessable") or []
    return any(re.search(r"\b" + re.escape(word) + r"\b", text, re.I) for word in words)


def build_demand_theme_audit(
    *,
    output_dir: Path,
    competencies: list[dict[str, Any]],
    failed_competencies: list[dict[str, Any]] | None = None,
    clusters: list[dict[str, Any]] | None = None,
    denominator: int = 0,
) -> dict[str, Any]:
    _ensure_dir(output_dir)
    failed_competencies = failed_competencies or []
    clusters = clusters or []
    rows: list[dict[str, Any]] = []
    not_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    comp_theme_by_id: dict[str, dict[str, str]] = {}

    for theme in THEME_PROBES:
        final_jobs: set[str] = set()
        failed_jobs: set[str] = set()
        evidence_texts: list[str] = []
        matched_final_ids: list[str] = []
        matched_failed_ids: list[str] = []
        cluster_hits: list[str] = []
        assessable_hits = 0
        for comp in competencies:
            blob = " ".join([
                str(comp.get("statement") or ""),
                " ".join(str(s) for s in comp.get("related_skills") or []),
                " ".join(str(s) for s in comp.get("source_sentences") or []),
            ])
            if _theme_hits(theme, blob):
                matched_final_ids.append(str(comp.get("id") or ""))
                final_jobs.update(str(j) for j in comp.get("source_job_ids") or [] if str(j).strip())
                evidence_texts.extend([str(s) for s in comp.get("source_sentences") or []][:5])
                if _has_assessable_activity(theme, blob):
                    assessable_hits += 1
                comp_theme_by_id[str(comp.get("id") or "")] = {
                    "technical_theme_id": theme["id"],
                    "technical_theme_label": theme["label"],
                }
        for comp in failed_competencies:
            blob = " ".join([
                str(comp.get("statement") or ""),
                " ".join(str(s) for s in comp.get("related_skills") or []),
                " ".join(str(s) for s in comp.get("source_sentences") or []),
            ])
            if _theme_hits(theme, blob):
                matched_failed_ids.append(str(comp.get("id") or ""))
                failed_jobs.update(str(j) for j in comp.get("source_job_ids") or [] if str(j).strip())
                evidence_texts.extend([str(s) for s in comp.get("source_sentences") or []][:5])
                if _has_assessable_activity(theme, blob):
                    assessable_hits += 1
        for cluster in clusters:
            blob = " ".join([
                str(cluster.get("summary_label") or ""),
                " ".join(str(t) for t in cluster.get("top_terms") or []),
                " ".join(str(i) for i in cluster.get("items") or []),
            ])
            if _theme_hits(theme, blob):
                cluster_hits.append(str(cluster.get("id") or ""))

        distinct_jobs = len(final_jobs | failed_jobs)
        status = THEME_NOT_DETECTED
        if distinct_jobs >= 10 and assessable_hits:
            status = THEME_STRONG
        elif distinct_jobs >= 5 and assessable_hits:
            status = THEME_MODERATE
        elif distinct_jobs > 0 and assessable_hits:
            status = THEME_WEAK
        elif distinct_jobs > 0 or cluster_hits:
            status = THEME_TOOL_ONLY
        if any(f in theme["id"] for f in ("cad", "clinical")):
            status = THEME_OUT
        recommendation = "review_candidate" if status in {THEME_STRONG, THEME_MODERATE} and matched_final_ids else THEME_NOT_RECOMMENDED
        reason = ""
        if recommendation == THEME_NOT_RECOMMENDED:
            if status == THEME_NOT_DETECTED:
                reason = "theme_not_detected_in_this_corpus"
            elif status == THEME_TOOL_ONLY:
                reason = "tool_or_keyword_mentions_without_assessable_task_evidence"
            elif status == THEME_WEAK:
                reason = "insufficient_distinct_job_support"
            else:
                reason = "no_final_grounded_competency_for_theme"
        row = {
            "theme_id": theme["id"],
            "theme_label": theme["label"],
            "theme_evidence_status": status,
            "theme_recommendation_status": recommendation,
            "not_recommended_reason": reason,
            "distinct_source_jobs": distinct_jobs,
            "job_prevalence_pct": round((distinct_jobs / denominator * 100.0), 3) if denominator else 0.0,
            "matched_final_competency_ids": " | ".join(x for x in matched_final_ids if x),
            "matched_failed_competency_ids": " | ".join(x for x in matched_failed_ids if x),
            "matched_cluster_ids": " | ".join(cluster_hits),
            "assessable_evidence_hits": assessable_hits,
            "sample_evidence": " || ".join(evidence_texts[:4]),
        }
        rows.append(row)
        if recommendation == THEME_NOT_RECOMMENDED:
            not_rows.append(row)
        if status in {THEME_STRONG, THEME_MODERATE} and not matched_final_ids and matched_failed_ids:
            recovery_rows.append(row)

    for comp in competencies:
        cid = str(comp.get("id") or "")
        theme = comp_theme_by_id.get(cid)
        if not theme:
            continue
        row = next((r for r in rows if r["theme_id"] == theme["technical_theme_id"]), None)
        if row:
            comp["technical_theme_id"] = row["theme_id"]
            comp["technical_theme_label"] = row["theme_label"]
            comp["theme_evidence_status"] = row["theme_evidence_status"]
            comp["theme_recommendation_status"] = row["theme_recommendation_status"]
            comp["not_recommended_reason"] = row["not_recommended_reason"]

    fields = [
        "theme_id", "theme_label", "theme_evidence_status", "theme_recommendation_status",
        "not_recommended_reason", "distinct_source_jobs", "job_prevalence_pct",
        "matched_final_competency_ids", "matched_failed_competency_ids", "matched_cluster_ids",
        "assessable_evidence_hits", "sample_evidence",
    ]
    _write_csv(output_dir / "demand_theme_audit.csv", rows, fields)
    _write_csv(output_dir / "not_recommended_themes.csv", not_rows, fields)
    _write_csv(output_dir / "theme_recovery_candidates.csv", recovery_rows, fields)
    report = {
        "schema_version": SCIENTIFIC_AUDIT_SCHEMA_VERSION,
        "denominator": denominator,
        "n_themes": len(rows),
        "status_distribution": dict(Counter(r["theme_evidence_status"] for r in rows)),
        "recommendation_distribution": dict(Counter(r["theme_recommendation_status"] for r in rows)),
        "recovery_candidate_theme_ids": [r["theme_id"] for r in recovery_rows],
        "themes": rows,
    }
    _write_json(output_dir / "demand_theme_audit.json", report)
    _write_json(output_dir / "theme_recovery_audit_report.json", {
        "schema_version": SCIENTIFIC_AUDIT_SCHEMA_VERSION,
        "policy": "Recovery is allowed only for themes with distinct-job support and assessable task evidence.",
        "n_recovery_candidate_themes": len(recovery_rows),
        "recovery_candidate_themes": recovery_rows,
        "not_recommended_themes": not_rows,
    })
    return report
