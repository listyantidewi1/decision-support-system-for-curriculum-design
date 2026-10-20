"""
Build DACUM/CBC review worksheets for one competency-v2 run.

The package turns generated competencies into an auditable chain:

    occupation/skema
      -> duty area
      -> task evidence
      -> skills/knowledge
      -> competency statement
      -> curriculum recommendation

No LLM calls are made. Duty areas, Spektrum candidates, and SKKNI unit
candidates are deterministic first-pass labels for expert review, not official
classification.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_RUN = PROJECT_ROOT / "results" / "competency_v2_pipeline_n1k_openrouter_gpt54"
DEFAULT_SPEKTRUM = PROJECT_ROOT / "DATA" / "spektrum_keahlian" / "spektrum_keahlian.json"
DEFAULT_SPEKTRUM_MAPPING = PROJECT_ROOT / "DATA" / "spektrum_keahlian" / "spektrum_mapping.csv"
DEFAULT_SKKNI_UNITS = PROJECT_ROOT / "DATA" / "master_skkni" / "json" / "skkni.json"
DEFAULT_SKKNI_OCC = PROJECT_ROOT / "DATA" / "skema_okupasi_indonesia_with_desc.csv"
DEFAULT_COMMUNITY_OCC = PROJECT_ROOT / "DATA" / "skema_okupasi_community_additions.csv"

DUTY_KEYWORDS = [
    ("Software development", {"software", "program", "api", "application", "backend", "frontend", "web", "mobile", "debug", "code"}),
    ("Cloud and infrastructure operations", {"cloud", "infrastructure", "devops", "ci/cd", "deployment", "platform", "server", "container", "kubernetes"}),
    ("Data, AI, and analytics", {"data", "database", "sql", "analytics", "machine learning", "ai", "model", "business intelligence"}),
    ("Cybersecurity and privacy", {"security", "cyber", "privacy", "protection", "incident", "vulnerability", "risk"}),
    ("Systems integration and architecture", {"architecture", "system", "systems", "integration", "distributed", "engineering foundation"}),
    ("Quality assurance and testing", {"quality", "testing", "test", "qa", "validation", "review"}),
    ("Technical support and service", {"technical support", "it support", "user support", "service desk", "helpdesk", "troubleshoot", "maintenance", "repair"}),
    ("Technical leadership and delivery", {"lead", "leadership", "project", "agile", "scrum", "planning", "mentor"}),
    ("Professional effectiveness", {"collaboration", "communication", "attention", "detail", "problem-solving", "priorities"}),
]

SPEKTRUM_RULES = [
    ("4.1.2", {"game", "gaming", "unity", "unreal"}),
    ("4.3.1", {"security", "cyber", "forensic", "incident", "vulnerability", "network security"}),
    ("4.2.1", {"network", "infrastructure", "cloud", "devops", "container", "computer technician", "hardware", "server"}),
    ("4.1.3", {"data", "database", "sql", "system information", "analytics", "information system"}),
    ("4.1.1", {"software", "program", "api", "application", "web", "mobile", "backend", "frontend", "machine learning", "ai"}),
    ("10.1.1", {"ui", "ux", "visual", "design", "prototype"}),
]

TOKEN_RE = re.compile(r"[a-zA-Z0-9+#./-]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "into", "is", "it", "of", "on", "or", "the", "their", "to", "use",
    "using", "with", "within",
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


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _join(value: Any, limit: int | None = None) -> str:
    items = [str(v).strip() for v in _as_list(value) if str(v).strip()]
    if limit is not None:
        items = items[:limit]
    return " | ".join(items)


def _compact(text: Any, max_chars: int = 240) -> str:
    raw = " ".join(str(text or "").split())
    if len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 3].rstrip() + "..."


def _tokens(text: Any) -> set[str]:
    return {
        t.lower()
        for t in TOKEN_RE.findall(str(text or "").lower())
        if len(t) > 1 and t.lower() not in STOPWORDS
    }


def _blob(comp: dict[str, Any]) -> str:
    parts = [
        comp.get("statement") or "",
        " ".join(str(x) for x in _as_list(comp.get("related_skills"))),
        " ".join(str(m.get("occupation_en", "")) for m in _as_list(comp.get("occupation_matches")) if isinstance(m, dict)),
    ]
    return " ".join(parts).lower()


def _ranked(competencies: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def infer_duty_area(comp: dict[str, Any]) -> tuple[str, str]:
    text = _blob(comp)
    scores: list[tuple[int, str, set[str]]] = []
    for duty, keywords in DUTY_KEYWORDS:
        hits = {kw for kw in keywords if kw in text}
        scores.append((len(hits), duty, hits))
    scores.sort(key=lambda x: (-x[0], x[1]))
    if scores and scores[0][0] > 0:
        return scores[0][1], "matched: " + ", ".join(sorted(scores[0][2])[:8])
    if comp.get("competency_class") == "employability_support":
        return "Professional effectiveness", "fallback from competency_class=employability_support"
    if comp.get("competency_class") == "engineering_process":
        return "Technical leadership and delivery", "fallback from competency_class=engineering_process"
    return "Needs expert classification", "no deterministic duty match"


def flatten_spektrum(path: Path) -> dict[str, dict[str, str]]:
    data = _load_json(path, {})
    out: dict[str, dict[str, str]] = {}
    for bidang in data.get("bidang", []) if isinstance(data, dict) else []:
        for program in bidang.get("program", []) or []:
            for kons in program.get("konsentrasi", []) or []:
                code = str(kons.get("code") or "")
                if code:
                    out[code] = {
                        "spektrum_code": code,
                        "bidang": str(bidang.get("name") or ""),
                        "program": str(program.get("name") or ""),
                        "konsentrasi": str(kons.get("name") or ""),
                        "source": data.get("source", ""),
                        "version": data.get("version", ""),
                    }
    return out


def load_spektrum_mapping(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {str(r.get("spektrum_code") or ""): r for r in csv.DictReader(f)}


def infer_spektrum(comp: dict[str, Any], spektrum: dict[str, dict[str, str]]) -> tuple[dict[str, str], str]:
    text = _blob(comp)
    best_code = ""
    best_hits: set[str] = set()
    for code, keywords in SPEKTRUM_RULES:
        hits = {kw for kw in keywords if kw in text}
        if len(hits) > len(best_hits):
            best_code = code
            best_hits = hits
    if not best_code:
        best_code = "4.1.1" if comp.get("competency_class") == "core_technical" else ""
    if best_code and best_code in spektrum:
        reason = "matched: " + ", ".join(sorted(best_hits)[:8]) if best_hits else "fallback from competency class"
        return spektrum[best_code], reason
    return {}, "no deterministic Spektrum match"


def load_skkni_units(path: Path) -> list[dict[str, str]]:
    data = _load_json(path, {})
    rows: list[dict[str, str]] = []
    if not isinstance(data, dict):
        return rows
    for domain, units in data.items():
        for unit in units or []:
            rows.append({
                "skkni_domain": str(domain),
                "unit_code": str(unit.get("unit_code") or ""),
                "unit_title": str(unit.get("unit_title") or ""),
            })
    return rows


def skkni_unit_candidates(comp: dict[str, Any], units: list[dict[str, str]], top_n: int = 5) -> list[dict[str, Any]]:
    comp_tokens = _tokens(_blob(comp))
    if not comp_tokens:
        return []
    scored: list[dict[str, Any]] = []
    for unit in units:
        unit_tokens = _tokens(unit.get("unit_title"))
        if not unit_tokens:
            continue
        overlap = comp_tokens & unit_tokens
        if not overlap:
            continue
        score = len(overlap) / max(1, len(unit_tokens))
        scored.append({
            **unit,
            "token_overlap_score": round(score, 4),
            "matched_terms": " | ".join(sorted(overlap)[:12]),
        })
    scored.sort(key=lambda r: (-float(r["token_overlap_score"]), r["skkni_domain"], r["unit_code"]))
    return scored[:top_n]


def _is_official_match(match: dict[str, Any]) -> bool:
    source = str(match.get("source_type") or "").lower()
    return bool(match.get("is_official")) or source.startswith("template-based") or "kepmenaker" in source or "verbatim" in source


def mapping_status(comp: dict[str, Any]) -> tuple[str, list[str]]:
    matches = [m for m in _as_list(comp.get("occupation_matches")) if isinstance(m, dict)]
    blockers: list[str] = []
    if not matches:
        return "unmapped", ["occupation_unmapped"]
    if any(_is_official_match(m) for m in matches):
        if any(not _is_official_match(m) for m in matches):
            blockers.append("community_mapping_present_as_secondary")
        return "official_or_template", blockers
    return "community_only", ["occupation_mapping_not_official"]


def recommendation_use(comp: dict[str, Any], occ_status: str) -> tuple[str, list[str]]:
    blockers = list(comp.get("official_readiness_blockers") or [])
    if occ_status == "community_only" and "occupation_mapping_not_official" not in blockers:
        blockers.append("occupation_mapping_not_official")
    if occ_status == "unmapped" and "occupation_unmapped" not in blockers:
        blockers.append("occupation_unmapped")
    corpus_status = comp.get("corpus_validity_status") or "needs_manual_review"
    if corpus_status != "valid_for_indonesian_claim" and "corpus_not_valid_for_indonesian_claim" not in blockers:
        blockers.append("corpus_not_valid_for_indonesian_claim")
    if comp.get("recommendation_status") == "official_ready" and not blockers:
        return "curriculum_candidate_official_ready", []
    if comp.get("recommendation_status") == "review_candidate":
        return "curriculum_review_candidate", sorted(set(blockers))
    return "exploratory_or_supporting_signal", sorted(set(blockers))


def source_type_counts(path: Path, is_official: bool) -> Counter:
    counts: Counter = Counter()
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = "official" if is_official else "community_non_official"
            counts[key] += 1
            counts[str(row.get("Source Type") or "unknown")] += 1
    return counts


def build_rows(
    competencies: list[dict[str, Any]],
    spektrum: dict[str, dict[str, str]],
    spektrum_mapping: dict[str, dict[str, str]],
    skkni_units: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    chain_rows: list[dict[str, Any]] = []
    worksheet_rows: list[dict[str, Any]] = []
    occupation_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []

    for rank, comp in enumerate(_ranked(competencies), start=1):
        duty, duty_reason = infer_duty_area(comp)
        sp, sp_reason = infer_spektrum(comp, spektrum)
        mapping_extra = spektrum_mapping.get(sp.get("spektrum_code", ""), {}) if sp else {}
        occ_status, occ_blockers = mapping_status(comp)
        use_status, use_blockers = recommendation_use(comp, occ_status)
        matches = [m for m in _as_list(comp.get("occupation_matches")) if isinstance(m, dict)]
        official_matches = [m for m in matches if _is_official_match(m)]
        top_match = matches[0] if matches else {}
        top_official = official_matches[0] if official_matches else {}
        evidence = _as_list(comp.get("source_sentences"))[:5]
        skills = _as_list(comp.get("related_skills"))
        unit_candidates = skkni_unit_candidates(comp, skkni_units, top_n=5)

        chain = {
            "rank": rank,
            "competency_id": comp.get("id", ""),
            "recommendation_status": comp.get("recommendation_status", ""),
            "curriculum_use_status": use_status,
            "curriculum_readiness_blockers": " | ".join(use_blockers),
            "competency_class": comp.get("competency_class", ""),
            "statement": comp.get("statement", ""),
            "occupation_mapping_status": occ_status,
            "occupation_mapping_blockers": " | ".join(occ_blockers),
            "top_occupation_en": top_match.get("occupation_en", ""),
            "top_occupation_id": top_match.get("occupation_id", ""),
            "top_occupation_source_type": top_match.get("source_type", ""),
            "top_occupation_is_official": top_match.get("is_official", ""),
            "top_official_occupation_en": top_official.get("occupation_en", ""),
            "top_official_occupation_id": top_official.get("occupation_id", ""),
            "top_official_occupation_source_type": top_official.get("source_type", ""),
            "n_official_occupation_matches": len(official_matches),
            "n_occupation_matches": len(matches),
            "duty_area": duty,
            "duty_area_reason": duty_reason,
            "task_evidence_sample": _join([_compact(x, 220) for x in evidence], limit=5),
            "skills_knowledge": _join(skills, limit=20),
            "source_job_count": comp.get("n_source_jobs") or len(_as_list(comp.get("source_job_ids"))),
            "job_prevalence_pct": comp.get("job_prevalence_pct", ""),
            "grounding_score": comp.get("grounding_score", ""),
            "kkni_level": comp.get("kkni_level", ""),
            "spektrum_code": sp.get("spektrum_code", ""),
            "spektrum_bidang": sp.get("bidang", ""),
            "spektrum_program": sp.get("program", ""),
            "spektrum_konsentrasi": sp.get("konsentrasi", ""),
            "spektrum_reason": sp_reason,
            "spektrum_primary_domain_ids": mapping_extra.get("primary_domain_ids", ""),
            "skkni_unit_candidates": _join([f"{u['unit_code']} {u['unit_title']}" for u in unit_candidates], limit=5),
            "corpus_validity_status": comp.get("corpus_validity_status", ""),
        }
        chain_rows.append(chain)

        worksheet_rows.append({
            **chain,
            "expert_reviewer_id": "",
            "expert_occupation_match_valid_yes_partial_no": "",
            "expert_duty_area_revision": "",
            "expert_task_statement_revision": "",
            "expert_skkni_unit_selection": "",
            "expert_spektrum_code_revision": "",
            "expert_curriculum_module_recommendation": "",
            "expert_priority_1_5": "",
            "expert_comments": "",
        })

        for idx, match in enumerate(matches, start=1):
            occupation_rows.append({
                "competency_id": comp.get("id", ""),
                "match_rank": idx,
                "occupation_en": match.get("occupation_en", ""),
                "occupation_id": match.get("occupation_id", ""),
                "sector": match.get("sector", ""),
                "cosine": match.get("cosine", ""),
                "is_official": bool(_is_official_match(match)),
                "source_type": match.get("source_type", ""),
                "mapping_status": occ_status,
                "expert_valid_yes_partial_no": "",
                "expert_comments": "",
            })

        for idx, unit in enumerate(unit_candidates, start=1):
            unit_rows.append({
                "competency_id": comp.get("id", ""),
                "candidate_rank": idx,
                "skkni_domain": unit.get("skkni_domain", ""),
                "unit_code": unit.get("unit_code", ""),
                "unit_title": unit.get("unit_title", ""),
                "token_overlap_score": unit.get("token_overlap_score", ""),
                "matched_terms": unit.get("matched_terms", ""),
                "expert_selected_yes_no": "",
                "expert_comments": "",
            })

    return chain_rows, worksheet_rows, occupation_rows, unit_rows


def write_readme(path: Path, run_dir: Path, report: dict[str, Any]) -> None:
    text = f"""# DACUM/CBC Package: {report.get('run_tag')}

Generated: {report.get('created_at')}

Source run:

```text
{run_dir}
```

## Purpose

This package supports curriculum-development review using a DACUM/CBC chain:

```text
occupation/skema -> duty area -> task evidence -> skills/knowledge -> competency statement -> curriculum recommendation
```

The labels are deterministic draft labels for expert review. They are not
official by themselves.

## Files

| File | Purpose |
|---|---|
| `dacum_competency_chain.csv` | One row per competency with occupation, duty, task evidence, skills, SKKNI/Spektrum candidates, and readiness blockers |
| `dacum_expert_worksheet.csv` | Same chain plus reviewer columns for DACUM validation and curriculum-module decisions |
| `occupation_mapping_audit.csv` | One row per occupation match; distinguishes official/template vs community-only mappings |
| `skkni_unit_candidates.csv` | Deterministic SKKNI unit candidates by token overlap for reviewer selection |
| `spektrum_alignment.csv` | Spektrum Keahlian candidate per competency |
| `dacum_cbc_report.json` | Machine-readable summary, reference versions, and blocker distributions |

## Interpretation

Official curriculum claims require Indonesian corpus validity, official mapping,
expert validation, and stability validation. Community mappings are useful for
gap discovery but remain non-official until validated.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--spektrum-json", default=str(DEFAULT_SPEKTRUM))
    parser.add_argument("--spektrum-mapping", default=str(DEFAULT_SPEKTRUM_MAPPING))
    parser.add_argument("--skkni-units-json", default=str(DEFAULT_SKKNI_UNITS))
    parser.add_argument("--skkni-occupations-csv", default=str(DEFAULT_SKKNI_OCC))
    parser.add_argument("--community-occupations-csv", default=str(DEFAULT_COMMUNITY_OCC))
    args = parser.parse_args()

    run_dir = _resolve(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    pipeline_report = _load_json(run_dir / "pipeline_report.json", {})
    competencies = _load_json(run_dir / "competencies.json", [])
    if not isinstance(competencies, list):
        raise ValueError("competencies.json must contain a list")
    run_tag = pipeline_report.get("tag") or run_dir.name.replace("competency_v2_pipeline_", "")
    out_dir = _resolve(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "dacum_cbc_packages" / str(run_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    spektrum_path = _resolve(args.spektrum_json)
    spektrum_mapping_path = _resolve(args.spektrum_mapping)
    skkni_units_path = _resolve(args.skkni_units_json)
    skkni_occ_path = _resolve(args.skkni_occupations_csv)
    community_occ_path = _resolve(args.community_occupations_csv)

    spektrum = flatten_spektrum(spektrum_path)
    spektrum_mapping = load_spektrum_mapping(spektrum_mapping_path)
    skkni_units = load_skkni_units(skkni_units_path)
    chain_rows, worksheet_rows, occupation_rows, unit_rows = build_rows(
        competencies, spektrum, spektrum_mapping, skkni_units
    )

    chain_fields = [
        "rank", "competency_id", "recommendation_status", "curriculum_use_status",
        "curriculum_readiness_blockers", "competency_class", "statement",
        "occupation_mapping_status", "occupation_mapping_blockers",
        "top_occupation_en", "top_occupation_id", "top_occupation_source_type",
        "top_occupation_is_official", "top_official_occupation_en",
        "top_official_occupation_id", "top_official_occupation_source_type",
        "n_official_occupation_matches", "n_occupation_matches", "duty_area", "duty_area_reason",
        "task_evidence_sample", "skills_knowledge", "source_job_count",
        "job_prevalence_pct", "grounding_score", "kkni_level", "spektrum_code",
        "spektrum_bidang", "spektrum_program", "spektrum_konsentrasi",
        "spektrum_reason", "spektrum_primary_domain_ids",
        "skkni_unit_candidates", "corpus_validity_status",
    ]
    worksheet_fields = chain_fields + [
        "expert_reviewer_id", "expert_occupation_match_valid_yes_partial_no",
        "expert_duty_area_revision", "expert_task_statement_revision",
        "expert_skkni_unit_selection", "expert_spektrum_code_revision",
        "expert_curriculum_module_recommendation", "expert_priority_1_5",
        "expert_comments",
    ]
    _write_csv(out_dir / "dacum_competency_chain.csv", chain_rows, chain_fields)
    _write_csv(out_dir / "dacum_expert_worksheet.csv", worksheet_rows, worksheet_fields)
    _write_csv(out_dir / "occupation_mapping_audit.csv", occupation_rows, [
        "competency_id", "match_rank", "occupation_en", "occupation_id",
        "sector", "cosine", "is_official", "source_type", "mapping_status",
        "expert_valid_yes_partial_no", "expert_comments",
    ])
    _write_csv(out_dir / "skkni_unit_candidates.csv", unit_rows, [
        "competency_id", "candidate_rank", "skkni_domain", "unit_code",
        "unit_title", "token_overlap_score", "matched_terms",
        "expert_selected_yes_no", "expert_comments",
    ])
    _write_csv(out_dir / "spektrum_alignment.csv", [
        {
            "competency_id": r["competency_id"],
            "spektrum_code": r["spektrum_code"],
            "spektrum_bidang": r["spektrum_bidang"],
            "spektrum_program": r["spektrum_program"],
            "spektrum_konsentrasi": r["spektrum_konsentrasi"],
            "spektrum_reason": r["spektrum_reason"],
            "expert_valid_yes_partial_no": "",
            "expert_comments": "",
        }
        for r in chain_rows
    ], [
        "competency_id", "spektrum_code", "spektrum_bidang", "spektrum_program",
        "spektrum_konsentrasi", "spektrum_reason", "expert_valid_yes_partial_no",
        "expert_comments",
    ])

    blocker_counts: Counter = Counter()
    for r in chain_rows:
        blocker_counts.update(x for x in str(r.get("curriculum_readiness_blockers") or "").split(" | ") if x)
    report = {
        "schema_version": "dacum_cbc_package_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_tag": run_tag,
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "n_competencies": len(competencies),
        "n_dacum_rows": len(chain_rows),
        "n_occupation_matches": len(occupation_rows),
        "n_skkni_unit_candidates": len(unit_rows),
        "curriculum_use_status_distribution": dict(Counter(r["curriculum_use_status"] for r in chain_rows)),
        "occupation_mapping_status_distribution": dict(Counter(r["occupation_mapping_status"] for r in chain_rows)),
        "duty_area_distribution": dict(Counter(r["duty_area"] for r in chain_rows)),
        "spektrum_code_distribution": dict(Counter(r["spektrum_code"] for r in chain_rows if r["spektrum_code"])),
        "readiness_blocker_distribution": dict(blocker_counts),
        "reference_registry": {
            "skema_okupasi_official_csv": str(skkni_occ_path),
            "skema_okupasi_community_csv": str(community_occ_path),
            "skkni_units_json": str(skkni_units_path),
            "spektrum_json": str(spektrum_path),
            "spektrum_mapping_csv": str(spektrum_mapping_path),
            "spektrum_source": _load_json(spektrum_path, {}).get("source", ""),
            "spektrum_version": _load_json(spektrum_path, {}).get("version", ""),
            "occupation_source_counts": dict(
                source_type_counts(skkni_occ_path, True)
                + source_type_counts(community_occ_path, False)
            ),
        },
        "interpretation": [
            "DACUM duty areas and SKKNI/Spektrum candidates are draft deterministic labels for expert validation.",
            "Community-only occupation mappings are non-official and block official-ready use.",
            "Official curriculum recommendations still require Indonesian corpus validity, expert review, and stability validation.",
        ],
    }
    _write_json(out_dir / "dacum_cbc_report.json", report)
    write_readme(out_dir / "README.md", run_dir, report)

    print(f"[OK] DACUM/CBC package written to: {out_dir}")
    print(f"     competencies: {len(competencies)}")
    print(f"     curriculum use: {report['curriculum_use_status_distribution']}")
    print(f"     occupation mapping: {report['occupation_mapping_status_distribution']}")
    print(f"     duty areas: {report['duty_area_distribution']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
