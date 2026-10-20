"""
Repair sprint stage for competency-v2 outputs.

This stage is intentionally placed after evidence repair and before final
ranking. It does three things:

1. Marks surviving competencies with repair audit fields.
2. Gives high-support technical failures a targeted LLM recovery chance.
3. Merges near-duplicate curriculum outcomes before public ranking.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from competency_v2_schema import CompetencyV2

logger = logging.getLogger(__name__)

REPAIR_STAGE_VERSION = "repair_sprint_v1"

_TOKEN_RE = re.compile(r"[a-z0-9+#./-]+")

_RECOVERY_EXCLUDE_TERMS = {
    "bachelor",
    "bachelor's",
    "bachelor’s",
    "master",
    "master's",
    "master’s",
    "degree",
    "communication",
    "communicate",
    "interpersonal",
    "attention to detail",
    "detail-oriented",
    "work ethic",
    "team player",
    "teamwork",
    "self-starter",
    "organised",
    "organized",
    "work under pressure",
    "fast-paced",
    "flexible",
}


@dataclass
class RepairStageConfig:
    denominator: int
    output_dir: Path
    enable_targeted_recovery: bool = True
    recovery_model: str = "openrouter:gpt-5.4-mini"
    request_timeout_seconds: int = 90
    temperature: float = 0.1
    max_tokens: int = 1400
    max_retries: int = 1
    max_recovery_candidates: int = 12
    min_recovery_source_jobs: int = 15
    min_recovery_prevalence: float = 0.01


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_dict(comp: CompetencyV2 | dict[str, Any]) -> dict[str, Any]:
    if isinstance(comp, CompetencyV2):
        return comp.to_dict()
    return dict(comp)


def _get(comp: CompetencyV2 | dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(comp, CompetencyV2):
        return getattr(comp, key, default)
    return comp.get(key, default)


def _set(comp: CompetencyV2, key: str, value: Any) -> None:
    setattr(comp, key, value)


def _append_flag(comp: CompetencyV2, flag: str) -> None:
    flags = list(comp.repair_flags or [])
    if flag not in flags:
        flags.append(flag)
    comp.repair_flags = sorted(flags)


def _source_job_count(comp: CompetencyV2 | dict[str, Any]) -> int:
    return len({str(j).strip() for j in _as_list(_get(comp, "source_job_ids")) if str(j).strip()})


def _tokens(text: str) -> set[str]:
    return {
        t
        for t in _TOKEN_RE.findall(str(text or "").lower())
        if len(t) >= 3
    }


def _text_blob(comp: CompetencyV2 | dict[str, Any]) -> str:
    parts = [
        str(_get(comp, "statement", "")),
        " ".join(str(s) for s in _as_list(_get(comp, "related_skills"))),
        str(_get(comp, "cluster_summary_label", "")),
    ]
    return " ".join(parts).lower()


def _failure_reason(comp: CompetencyV2 | dict[str, Any]) -> str:
    audit = _get(comp, "evidence_repair_audit") or {}
    if isinstance(audit, dict) and audit.get("drop_reason"):
        return str(audit.get("drop_reason"))
    return str(_get(comp, "dropped_reason", "") or _get(comp, "failure_reason", "") or "")


def _prevalence(comp: CompetencyV2 | dict[str, Any], denominator: int) -> float:
    return _source_job_count(comp) / denominator if denominator > 0 else 0.0


def _contains_excluded_recovery_term(text: str) -> bool:
    return any(term in text for term in _RECOVERY_EXCLUDE_TERMS)


def _looks_recoverable_technical(comp: CompetencyV2 | dict[str, Any]) -> tuple[bool, str]:
    from scripts.apply_scientific_sprint1 import classify_competency

    comp_class, reason = classify_competency(_as_dict(comp))
    if comp_class != "core_technical":
        return False, f"class={comp_class}"
    text = _text_blob(comp)
    if _contains_excluded_recovery_term(text):
        return False, "excluded_support_or_credential_terms"
    return True, reason


def _recovery_candidate_rows(
    failed: Sequence[CompetencyV2],
    config: RepairStageConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comp in failed:
        n_jobs = _source_job_count(comp)
        prevalence = _prevalence(comp, config.denominator)
        high_support = (
            n_jobs >= config.min_recovery_source_jobs
            or prevalence >= config.min_recovery_prevalence
        )
        technical, reason = _looks_recoverable_technical(comp)
        selected = bool(high_support and technical)
        rows.append({
            "selected": selected,
            "competency_id": comp.id,
            "cluster_id": comp.cluster_id,
            "cluster_summary_label": comp.cluster_summary_label,
            "source_job_count": n_jobs,
            "job_prevalence_pct": round(prevalence * 100, 3),
            "failure_reason": _failure_reason(comp),
            "selection_reason": reason if selected else ("not_high_support" if not high_support else reason),
            "statement": comp.statement,
            "related_skills": " | ".join(comp.related_skills or []),
        })
    rows.sort(key=lambda r: (not r["selected"], -int(r["source_job_count"]), r["competency_id"]))
    return rows


def select_recovery_candidates(
    failed: Sequence[CompetencyV2],
    config: RepairStageConfig,
) -> list[CompetencyV2]:
    selected_ids = {
        r["competency_id"]
        for r in _recovery_candidate_rows(failed, config)
        if r["selected"]
    }
    out = [c for c in failed if c.id in selected_ids]
    out.sort(key=lambda c: (-_source_job_count(c), c.id))
    return out[: config.max_recovery_candidates]


def _build_recovery_prompt(comp: CompetencyV2) -> str:
    skills = "\n".join(f"- {s}" for s in (comp.related_skills or []))
    sentences = "\n".join(
        f"{i + 1}. {s}" for i, s in enumerate((comp.source_sentences or [])[:12])
    )
    return f"""You are repairing one failed competency candidate.

The previous statement failed evidence grounding, so you must write a narrower
competency using ONLY the exact evidence below. Do not add products, languages,
frameworks, or duties that are absent from the evidence.

If the evidence is too mixed or too weak, return {{"recover": false, "reason": "..."}}
instead of forcing a competency.

FAILED STATEMENT:
{comp.statement}

FAILURE REASON:
{_failure_reason(comp)}

RELATED SKILL CANDIDATES:
{skills}

SOURCE SENTENCES:
{sentences}

Return JSON only:
{{
  "recover": true,
  "statement": "2-3 concise verb-led curriculum competency sentences.",
  "related_skills": ["copy only skills from RELATED SKILL CANDIDATES"],
  "rationale": "250-600 chars explaining why this narrower statement fits the evidence.",
  "kkni_level": 4
}}"""


def _parse_json(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _normal_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _make_recovered_competency(
    source: CompetencyV2,
    payload: dict[str, Any],
    model: str,
) -> Optional[CompetencyV2]:
    if payload.get("recover") is False:
        return None
    statement = str(payload.get("statement") or "").strip()
    if len(statement) < 40:
        return None
    allowed = {_normal_key(s): s for s in (source.related_skills or [])}
    skills: list[str] = []
    for skill in _as_list(payload.get("related_skills")):
        key = _normal_key(skill)
        if key in allowed and allowed[key] not in skills:
            skills.append(allowed[key])
    if not skills and len(source.related_skills or []) == 1:
        skills = list(source.related_skills or [])
    if not skills:
        return None
    rid = "comp_repair_" + hashlib.sha1(f"{source.id}|{statement}".encode("utf-8")).hexdigest()[:8]
    return CompetencyV2(
        id=rid,
        statement=statement,
        related_skills=skills,
        soft_skills_required=list(source.soft_skills_required or []),
        soft_skills_description=source.soft_skills_description or "",
        rationale=str(payload.get("rationale") or source.rationale or "")[:900],
        batch_reasoning_id=source.batch_reasoning_id,
        contributing_item_ids=list(source.contributing_item_ids or []),
        source_job_ids=list(source.source_job_ids or []),
        source_sentence_ids=list(source.source_sentence_ids or []),
        source_sentences=list(source.source_sentences or []),
        future_relevance=source.future_relevance,
        future_weight=float(source.future_weight or 0.0),
        empirical_trend=source.empirical_trend or "Stable",
        kkni_level=payload.get("kkni_level") or source.kkni_level,
        kkni_level_source="repair_llm_suggested",
        generated_at=datetime.now(timezone.utc).isoformat(),
        generator_version=f"{source.generator_version}+{REPAIR_STAGE_VERSION}",
        cluster_id=source.cluster_id,
        cluster_summary_label=source.cluster_summary_label,
        provider="jatevo" if model.startswith("jatevo:") else "openrouter",
        model=model,
        repair_decision="recovered",
        repair_flags=["targeted_llm_recovery"],
        recovered_from_failed=True,
        source_failure_reason=_failure_reason(source),
    )


def recover_with_llm(
    candidates: Sequence[CompetencyV2],
    config: RepairStageConfig,
) -> tuple[list[CompetencyV2], list[dict[str, Any]]]:
    if not config.enable_targeted_recovery or not candidates:
        return [], []

    from llm_client_router import get_client_for_model, strip_provider_prefix

    try:
        client, _provider = get_client_for_model(
            config.recovery_model,
            request_timeout=config.request_timeout_seconds,
        )
    except Exception as exc:  # pragma: no cover - live API dependent
        logger.warning("targeted recovery unavailable via %s: %s", config.recovery_model, exc)
        return [], [
            {
                "source_competency_id": comp.id,
                "cluster_id": comp.cluster_id,
                "source_job_count": _source_job_count(comp),
                "recovered_competency_id": "",
                "success": False,
                "error": str(exc),
                "latency_seconds": 0.0,
                "raw_response_preview": "",
            }
            for comp in candidates
        ]
    api_model_name = strip_provider_prefix(config.recovery_model)
    recovered: list[CompetencyV2] = []
    attempts: list[dict[str, Any]] = []

    for comp in candidates:
        prompt = _build_recovery_prompt(comp)
        raw_text = ""
        parsed: Optional[dict[str, Any]] = None
        error = ""
        latency = 0.0
        for attempt in range(config.max_retries + 1):
            started = time.time()
            try:
                resp = client.chat.completions.create(
                    model=api_model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a strict competency evidence repair assistant. Return JSON only.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
                latency += time.time() - started
                raw_text = resp.choices[0].message.content or ""
                parsed = _parse_json(raw_text)
                if parsed is not None:
                    break
                error = "unparseable_json"
            except Exception as exc:  # pragma: no cover - live API dependent
                latency += time.time() - started
                error = str(exc)
                logger.warning("targeted recovery failed for %s: %s", comp.id, exc)
            if attempt < config.max_retries:
                time.sleep(1.5 ** (attempt + 1))

        recovered_comp = _make_recovered_competency(comp, parsed or {}, config.recovery_model)
        if recovered_comp is not None:
            recovered.append(recovered_comp)
        attempts.append({
            "source_competency_id": comp.id,
            "cluster_id": comp.cluster_id,
            "source_job_count": _source_job_count(comp),
            "recovered_competency_id": recovered_comp.id if recovered_comp else "",
            "success": recovered_comp is not None,
            "error": error,
            "latency_seconds": round(latency, 3),
            "raw_response_preview": raw_text[:500],
        })
    return recovered, attempts


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    aa = set(a)
    bb = set(b)
    return len(aa & bb) / len(aa | bb) if aa or bb else 0.0


def _skill_overlap(a: CompetencyV2, b: CompetencyV2) -> int:
    return len({_normal_key(s) for s in (a.related_skills or [])} & {_normal_key(s) for s in (b.related_skills or [])})


def _class_for(comp: CompetencyV2) -> str:
    from scripts.apply_scientific_sprint1 import classify_competency

    cls, _reason = classify_competency(comp.to_dict())
    return cls


def _is_duplicate_outcome(a: CompetencyV2, b: CompetencyV2) -> bool:
    class_a = _class_for(a)
    class_b = _class_for(b)
    if class_a != class_b:
        return False
    job_overlap = _jaccard(a.source_job_ids, b.source_job_ids)
    skill_overlap = _skill_overlap(a, b)
    statement_overlap = _jaccard(_tokens(a.statement), _tokens(b.statement))

    if job_overlap >= 0.85 and (skill_overlap >= 1 or statement_overlap >= 0.08):
        return True
    if class_a in {"employability_support", "engineering_process"} and (
        statement_overlap >= 0.16 or job_overlap >= 0.30 or skill_overlap >= 1
    ):
        return True
    if class_a == "core_technical" and skill_overlap >= 2 and statement_overlap >= 0.16:
        return True
    return False


def _dedup_group_id(ids: Sequence[str]) -> str:
    key = "|".join(sorted(ids))
    return "dedup_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]


def _merge_into(survivor: CompetencyV2, absorbed: CompetencyV2, group_id: str) -> None:
    def union(existing: Sequence[Any], incoming: Sequence[Any]) -> list[Any]:
        out: list[Any] = []
        seen: set[str] = set()
        for value in list(existing or []) + list(incoming or []):
            key = json.dumps(value, sort_keys=True, ensure_ascii=False) if isinstance(value, dict) else str(value)
            if key in seen:
                continue
            seen.add(key)
            out.append(value)
        return out

    survivor.related_skills = union(survivor.related_skills, absorbed.related_skills)
    survivor.contributing_item_ids = union(survivor.contributing_item_ids, absorbed.contributing_item_ids)
    survivor.source_job_ids = union(survivor.source_job_ids, absorbed.source_job_ids)
    survivor.source_sentence_ids = union(survivor.source_sentence_ids, absorbed.source_sentence_ids)
    survivor.source_sentences = union(survivor.source_sentences, absorbed.source_sentences)[:12]
    survivor.merged_from = sorted(set((survivor.merged_from or []) + [absorbed.id] + list(absorbed.merged_from or [])))
    survivor.curriculum_dedup_group = group_id
    _append_flag(survivor, "merged_near_duplicate")
    absorbed.curriculum_dedup_group = group_id
    absorbed.repair_decision = "merged"
    _append_flag(absorbed, "merged_into_curriculum_duplicate")


def dedupe_curriculum_competencies(
    competencies: Sequence[CompetencyV2],
) -> tuple[list[CompetencyV2], list[CompetencyV2], list[dict[str, Any]]]:
    ordered = sorted(
        competencies,
        key=lambda c: (-_source_job_count(c), -float(c.grounding_score or 0.0), c.id),
    )
    used: set[str] = set()
    survivors: list[CompetencyV2] = []
    merged_out: list[CompetencyV2] = []
    groups: list[dict[str, Any]] = []

    for comp in ordered:
        if comp.id in used:
            continue
        group = [comp]
        used.add(comp.id)
        for other in ordered:
            if other.id in used:
                continue
            if _is_duplicate_outcome(comp, other):
                group.append(other)
                used.add(other.id)
        if len(group) > 1:
            group_id = _dedup_group_id([g.id for g in group])
            for absorbed in group[1:]:
                _merge_into(comp, absorbed, group_id)
                merged_out.append(absorbed)
            groups.append({
                "curriculum_dedup_group": group_id,
                "survivor_id": comp.id,
                "merged_ids": [g.id for g in group[1:]],
                "class": _class_for(comp),
            })
        survivors.append(comp)

    return survivors, merged_out, groups


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def run_repair_stage(
    competencies: Sequence[CompetencyV2],
    failed_evidence_repair: Sequence[CompetencyV2],
    config: RepairStageConfig,
    evidence_repair_config: Any,
) -> tuple[list[CompetencyV2], list[CompetencyV2], dict[str, Any]]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for comp in competencies:
        if not comp.repair_decision:
            comp.repair_decision = "kept"
        comp.repair_flags = sorted(set(comp.repair_flags or []))

    candidate_rows = _recovery_candidate_rows(failed_evidence_repair, config)
    selected = select_recovery_candidates(failed_evidence_repair, config)

    raw_recovered, recovery_attempts = recover_with_llm(selected, config)
    recovered_passing: list[CompetencyV2] = []
    recovered_failed: list[CompetencyV2] = []
    recovery_repair_report: dict[str, Any] = {}
    if raw_recovered:
        from competency_evidence_repair import repair_competencies

        recovered_passing, recovered_failed, er_report = repair_competencies(
            raw_recovered,
            evidence_repair_config,
        )
        recovery_repair_report = er_report.to_dict()
        for comp in recovered_passing:
            comp.repair_decision = "recovered"
            comp.recovered_from_failed = True
            _append_flag(comp, "passed_repair_after_targeted_recovery")
        for comp in recovered_failed:
            comp.repair_decision = "dropped"
            _append_flag(comp, "targeted_recovery_failed_grounding")

    combined = list(competencies) + recovered_passing
    deduped, merged_out, dedup_groups = dedupe_curriculum_competencies(combined)

    recovered_source_ids = sorted(
        {
            attempt["source_competency_id"]
            for attempt in recovery_attempts
            if attempt.get("success")
        }
    )
    dropped_high_support_rows = []
    recovered_ids = {c.id for c in recovered_passing}
    for row in candidate_rows:
        if not row["selected"]:
            continue
        attempt = next(
            (a for a in recovery_attempts if a.get("source_competency_id") == row["competency_id"]),
            {},
        )
        if attempt.get("recovered_competency_id") in recovered_ids:
            continue
        dropped_high_support_rows.append({
            **row,
            "recovery_attempted": bool(attempt),
            "recovery_error": attempt.get("error", ""),
            "recovered_competency_id": attempt.get("recovered_competency_id", ""),
        })

    report = {
        "schema_version": REPAIR_STAGE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_input_competencies": len(competencies),
        "n_failed_evidence_repair": len(failed_evidence_repair),
        "n_recovery_candidates": len(selected),
        "targeted_recovery_enabled": config.enable_targeted_recovery,
        "recovery_model": config.recovery_model if config.enable_targeted_recovery else "",
        "n_raw_recovered": len(raw_recovered),
        "n_recovered_passing": len(recovered_passing),
        "n_recovered_failed": len(recovered_failed),
        "n_after_recovery_before_dedup": len(combined),
        "n_dedup_merged": len(merged_out),
        "n_after_dedup": len(deduped),
        "recovered_source_ids": recovered_source_ids,
        "dedup_groups": dedup_groups,
        "recovery_attempts": recovery_attempts,
        "recovery_evidence_repair_report": recovery_repair_report,
        "config": {
            "min_recovery_source_jobs": config.min_recovery_source_jobs,
            "min_recovery_prevalence": config.min_recovery_prevalence,
            "max_recovery_candidates": config.max_recovery_candidates,
        },
    }

    _write_json(config.output_dir / "repair_audit_report.json", report)
    _write_json(config.output_dir / "recovered_competencies.json", [c.to_dict() for c in recovered_passing])
    _write_csv(
        config.output_dir / "repair_candidates.csv",
        candidate_rows,
        [
            "selected", "competency_id", "cluster_id", "cluster_summary_label",
            "source_job_count", "job_prevalence_pct", "failure_reason",
            "selection_reason", "statement", "related_skills",
        ],
    )
    _write_csv(
        config.output_dir / "dropped_high_support.csv",
        dropped_high_support_rows,
        [
            "selected", "competency_id", "cluster_id", "cluster_summary_label",
            "source_job_count", "job_prevalence_pct", "failure_reason",
            "selection_reason", "recovery_attempted", "recovery_error",
            "recovered_competency_id", "statement", "related_skills",
        ],
    )

    repair_failures = list(recovered_failed) + list(merged_out)
    return deduped, repair_failures, report
