"""Deterministic post-extraction typing for Skill-LLM spans.

This module is intentionally dependency-light so it can be reused by:
the Skill-LLM -> Phase-1 converter, read-only scientific audits, and the
runtime v2 pipeline gate. The rule set is conservative: a span must carry a
real technical/process anchor before it can enter technical generation.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


EXTRACTION_TYPES = (
    "hard_technical",
    "technical_process",
    "support_soft_skill",
    "personality_trait",
    "credential",
    "education_requirement",
    "out_of_domain",
    "invalid_fragment",
    "knowledge_domain",
)

TECHNICAL_GENERATION_TYPES = {"hard_technical", "technical_process"}
SKILL_PHASE1_TYPES = {"hard_technical", "technical_process", "support_soft_skill"}
KNOWLEDGE_PHASE1_TYPES = TECHNICAL_GENERATION_TYPES
EXCLUDED_EXTRACTION_TYPES = set(EXTRACTION_TYPES) - TECHNICAL_GENERATION_TYPES - {"support_soft_skill"}


_PUNCT_RE = re.compile(r"^[\s\"'`([{<]+|[\s\"'`)\]}>.,;:]+$")
_SPACE_RE = re.compile(r"\s+")
_GENERIC_FRAGMENT_RE = re.compile(
    r"\b(develop|build|planning|execution|execute|hands[- ]on|responsible for|ability to|"
    r"experience with|knowledge of|familiarity with|work on|work with|support|manage)\b",
    re.I,
)
_CREDENTIAL_RE = re.compile(
    r"\b(bachelor|master|degree|diploma|certification|certified|license|licence|"
    r"years? of experience|ph\.?d|bs/ms|b\.?s\.?|m\.?s\.?)\b",
    re.I,
)
_EDUCATION_RE = re.compile(r"\b(computer science|information systems|engineering degree|related field)\b", re.I)
_PERSONALITY_RE = re.compile(
    r"\b(self[- ]motivated|detail[- ]oriented|work ethic|professional drive|"
    r"fast[- ]paced|passion|curious|flexible|adaptable|proactive|ownership|"
    r"attention to detail)\b",
    re.I,
)
_SUPPORT_RE = re.compile(
    r"\b(communication|collaboration|teamwork|stakeholder|presentation|mentoring|"
    r"leadership|problem[- ]solving|analytical thinking|cross[- ]functional|"
    r"prioritization|time management)\b",
    re.I,
)
_PROCESS_RE = re.compile(
    r"\b(agile|scrum|kanban|ci/cd|devops|code review|design review|debugging|"
    r"troubleshooting|testing|test automation|observability|monitoring|"
    r"incident response|release management|requirements analysis)\b",
    re.I,
)
_OUT_OF_DOMAIN_RE = re.compile(
    r"\b(cad|drafting|solidworks|autocad|bim|gis|clinical|safeguarding|"
    r"mental capacity|social work|nursing|forklift|warehouse|plumbing|meica|asic)\b",
    re.I,
)
_TECH_RE = re.compile(
    r"\b(python|java\b|javascript|typescript|node\.?js|react|react native|angular|vue|"
    r"sql|postgres|mysql|mongodb|database|rest|api|graphql|microservice|backend|"
    r"frontend|aws|azure|google cloud|gcp|cloud|kubernetes|docker|terraform|"
    r"infrastructure as code|iac|linux|git|spark|airflow|etl|data pipeline|"
    r"machine learning|ml\b|ai\b|nlp|genai|llm|security|oauth|saml|testing|"
    r"selenium|cypress|jest|pytest|tableau|power bi|salesforce|sharepoint|"
    r"power apps|iot|mobile|ios|android|swift|kotlin)\b",
    re.I,
)


@dataclass(frozen=True)
class ExtractionTypingDecision:
    source_field: str
    raw_span: str
    normalized_span: str
    post_extraction_type: str
    flags: list[str]
    phase1_target: str
    skill_type: str
    hard_generation_eligible: bool


def normalize_extraction_text(text: str) -> str:
    text = _PUNCT_RE.sub("", str(text or ""))
    text = text.replace("RESTful APIs", "REST APIs")
    text = text.replace("Infrastructure as Code (IaC", "Infrastructure as Code")
    text = text.replace("Node.js.", "Node.js")
    text = text.replace("React Native.", "React Native")
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


def classify_extraction_span(text: str, *, source_field: str = "", sentence_text: str = "") -> tuple[str, list[str]]:
    raw = str(text or "")
    norm = normalize_extraction_text(raw)
    lower = norm.lower()
    flags: list[str] = []
    if not norm or len(norm) < 2:
        return "invalid_fragment", ["empty_or_too_short"]
    token_count = len(re.findall(r"[A-Za-z0-9+#.]+", norm))
    if raw != norm:
        flags.append("punctuation_normalized")
    if token_count >= 9:
        flags.append("long_sentence_fragment")
    if _GENERIC_FRAGMENT_RE.search(lower) and not _TECH_RE.search(lower):
        flags.append("generic_verb_or_fragment")
    if _CREDENTIAL_RE.search(lower):
        return "credential", sorted(set(flags + ["credential_or_experience_requirement"]))
    if _EDUCATION_RE.search(lower) and token_count <= 6:
        return "education_requirement", sorted(set(flags + ["education_field_requirement"]))
    if _PERSONALITY_RE.search(lower):
        return "personality_trait", sorted(set(flags + ["personality_or_trait"]))
    if _OUT_OF_DOMAIN_RE.search(lower):
        return "out_of_domain", sorted(set(flags + ["outside_rpl_it_domain"]))
    if _SUPPORT_RE.search(lower) and not _TECH_RE.search(lower):
        return "support_soft_skill", sorted(set(flags + ["support_soft_skill"]))
    if source_field == "knowledge" and not _TECH_RE.search(lower):
        return "knowledge_domain", sorted(set(flags + ["knowledge_without_technical_anchor"]))
    if _PROCESS_RE.search(lower):
        return "technical_process", sorted(set(flags + ["technical_process_or_practice"]))
    if _TECH_RE.search(lower):
        return "hard_technical", sorted(set(flags + ["technical_anchor"]))
    if token_count <= 1:
        return "invalid_fragment", sorted(set(flags + ["single_generic_or_unanchored_token"]))
    if token_count >= 9 or _GENERIC_FRAGMENT_RE.search(lower):
        return "invalid_fragment", sorted(set(flags + ["generic_or_long_unanchored_fragment"]))
    return "support_soft_skill", sorted(set(flags + ["unanchored_nontechnical_span"]))


def phase1_decision_for_span(
    text: str,
    *,
    source_field: str,
    sentence_text: str = "",
) -> ExtractionTypingDecision:
    category, flags = classify_extraction_span(text, source_field=source_field, sentence_text=sentence_text)
    normalized = normalize_extraction_text(text)
    phase1_target = "excluded"
    skill_type = ""
    hard_generation_eligible = category in TECHNICAL_GENERATION_TYPES
    if source_field == "skill":
        if category in TECHNICAL_GENERATION_TYPES:
            phase1_target = "advanced_skills"
            skill_type = "Hard"
        elif category == "support_soft_skill":
            phase1_target = "advanced_skills"
            skill_type = "Soft"
    elif source_field == "knowledge" and category in KNOWLEDGE_PHASE1_TYPES:
        phase1_target = "advanced_knowledge"
    return ExtractionTypingDecision(
        source_field=source_field,
        raw_span=str(text or ""),
        normalized_span=normalized,
        post_extraction_type=category,
        flags=list(flags),
        phase1_target=phase1_target,
        skill_type=skill_type,
        hard_generation_eligible=hard_generation_eligible,
    )


def extraction_flags_to_text(flags: Iterable[str]) -> str:
    return " | ".join(sorted({str(f).strip() for f in flags if str(f).strip()}))


def job_id_from_sentence_id(sentence_id: str) -> str:
    sid = str(sentence_id or "").strip()
    if not sid:
        return ""
    if "_" not in sid:
        return sid
    head, _sep, tail = sid.rpartition("_")
    return head if tail.isdigit() else sid


def apply_extraction_type_gate_to_items(items: list[Any]) -> tuple[list[Any], dict[str, Any], list[dict[str, Any]]]:
    """Classify already-loaded Phase-1 items and remove unsafe hard evidence.

    Old Phase-1 folders do not carry post_extraction_type columns, so this gate
    reclassifies missing rows at runtime. New folders can pass through with the
    same deterministic decision because the classifier is shared.
    """
    from pipeline import SkillType

    kept: list[Any] = []
    rows: list[dict[str, Any]] = []
    before_types: Counter[str] = Counter()
    after_types: Counter[str] = Counter()
    before_generation_types: Counter[str] = Counter()
    after_generation_types: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    affected_jobs: set[str] = set()

    for idx, item in enumerate(items):
        source_field = "skill" if hasattr(item, "type") else "knowledge"
        text = str(getattr(item, "text", "") or "")
        existing_category = str(getattr(item, "post_extraction_type", "") or "").strip()
        category, flags = classify_extraction_span(
            text,
            source_field=source_field,
            sentence_text=str(getattr(item, "sentence_text", "") or ""),
        )
        if existing_category in EXTRACTION_TYPES:
            category = existing_category
            existing_flags = str(getattr(item, "extraction_flags", "") or "")
            if existing_flags:
                flags = [x.strip() for x in existing_flags.split("|") if x.strip()]

        original_stream = "knowledge"
        if source_field == "skill":
            raw_type = getattr(getattr(item, "type", None), "value", getattr(item, "type", ""))
            original_stream = f"skill_{str(raw_type or '').lower() or 'unknown'}"
        generation_before = source_field == "knowledge" or original_stream == "skill_hard"
        if generation_before:
            before_generation_types[category] += 1
        before_types[category] += 1

        action = "dropped"
        final_stream = ""
        hard_generation_after = False
        if source_field == "skill":
            if category in TECHNICAL_GENERATION_TYPES:
                item.type = SkillType.HARD
                action = "kept"
                final_stream = "skill_hard"
                hard_generation_after = True
            elif category == "support_soft_skill":
                item.type = SkillType.SOFT
                action = "demoted_to_soft" if original_stream != "skill_soft" else "kept_soft"
                final_stream = "skill_soft"
            else:
                action = "dropped"
        else:
            if category in TECHNICAL_GENERATION_TYPES:
                action = "kept"
                final_stream = "knowledge"
                hard_generation_after = True
            else:
                action = "dropped"

        sid = str(getattr(item, "sentence_id", "") or "")
        job_id = str(getattr(item, "job_id", "") or "") or job_id_from_sentence_id(sid)
        normalized = normalize_extraction_text(text)
        setattr(item, "post_extraction_type", category)
        setattr(item, "extraction_flags", extraction_flags_to_text(flags))
        setattr(item, "normalized_span", normalized)
        setattr(item, "hard_generation_eligible", hard_generation_after)

        if action != "dropped":
            kept.append(item)
            after_types[category] += 1
            if hard_generation_after:
                after_generation_types[category] += 1
        else:
            if job_id:
                affected_jobs.add(job_id)
        if action.startswith("demoted") and job_id:
            affected_jobs.add(job_id)
        actions[action] += 1
        rows.append({
            "row_index": idx,
            "job_id": job_id,
            "sentence_id": sid,
            "source_field": source_field,
            "raw_span": text,
            "normalized_span": normalized,
            "post_extraction_type": category,
            "flags": extraction_flags_to_text(flags),
            "original_stream": original_stream,
            "final_stream": final_stream,
            "gate_action": action,
            "hard_generation_before": generation_before,
            "hard_generation_after": hard_generation_after,
            "sentence_text": str(getattr(item, "sentence_text", "") or ""),
        })

    unsafe_after = {
        k: v for k, v in after_generation_types.items()
        if k not in TECHNICAL_GENERATION_TYPES and v
    }
    report = {
        "schema_version": "2026-06-23.upstream_evidence_repair.v1",
        "n_items_before": len(items),
        "n_items_after": len(kept),
        "n_items_dropped": int(actions.get("dropped", 0)),
        "n_items_demoted_to_soft": int(actions.get("demoted_to_soft", 0)),
        "action_distribution": dict(actions),
        "post_extraction_type_distribution_before": dict(before_types),
        "post_extraction_type_distribution_after": dict(after_types),
        "hard_generation_type_distribution_before": dict(before_generation_types),
        "hard_generation_type_distribution_after": dict(after_generation_types),
        "unsafe_hard_generation_types_after": unsafe_after,
        "n_unsafe_hard_generation_items_after": sum(unsafe_after.values()),
        "affected_job_ids": sorted(affected_jobs),
        "n_affected_job_ids": len(affected_jobs),
    }
    return kept, report, rows


def write_extraction_type_gate_audit(output_dir: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "extraction_type_gate_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    fieldnames = [
        "row_index", "job_id", "sentence_id", "source_field", "raw_span",
        "normalized_span", "post_extraction_type", "flags", "original_stream",
        "final_stream", "gate_action", "hard_generation_before",
        "hard_generation_after", "sentence_text",
    ]
    with (output_dir / "extraction_type_gate_audit.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

