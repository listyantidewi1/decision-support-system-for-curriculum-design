"""
Audit raw corpus lineage and corpus validity for competency-v2 runs.

The audit is deliberately read-only. It checks whether a prepared dataset and
Phase-1 extraction directory can be traced back to a raw job-posting CSV and to
Skill-LLM JSONL inputs/outputs. Outputs are written under:

    results/corpus_audits/<dataset_id>/

The audit does not certify recommendations as official. It only establishes
whether the corpus is complete enough for engineering tests or Indonesian
claim-making.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent

STATUS_VALID_ENGINEERING = "valid_for_engineering_test"
STATUS_VALID_INDONESIAN = "valid_for_indonesian_claim"
STATUS_INVALID = "invalid_or_incomplete"
STATUS_REVIEW = "needs_manual_review"


@dataclass
class CsvProfile:
    path: str
    exists: bool
    row_count: int = 0
    unique_job_ids: int = 0
    job_ids: set[str] | None = None
    missing_description_rate: float = 0.0
    missing_title_rate: float = 0.0
    duplicate_rate: float = 0.0
    date_min: str = ""
    date_max: str = ""
    site_distribution: dict[str, int] | None = None
    source_distribution: dict[str, int] | None = None
    country_distribution: dict[str, int] | None = None
    language_distribution: dict[str, int] | None = None


def _resolve(path: str | Path | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def _rel(path: Path | None) -> str:
    if not path:
        return ""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def _safe_ratio(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def _compact_distribution(counter: Counter, limit: int = 20) -> dict[str, int]:
    return {str(k): int(v) for k, v in counter.most_common(limit) if str(k).strip()}


def _standard_country(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "unknown"
    lower = raw.lower()
    aliases = {
        "united states": "US",
        "usa": "US",
        "u.s.": "US",
        "us": "US",
        "united kingdom": "GB",
        "uk": "GB",
        "great britain": "GB",
        "england": "GB",
        "gb": "GB",
        "australia": "AU",
        "au": "AU",
        "canada": "CA",
        "ca": "CA",
        "indonesia": "ID",
        "id": "ID",
    }
    if lower in aliases:
        return aliases[lower]
    if "indonesia" in lower:
        return "ID"
    if "amerika serikat" in lower or "united states" in lower:
        return "US"
    if "singapura" in lower or "singapore" in lower:
        return "SG"
    match = re.search(r",\s*([A-Z]{2})\s*$", raw)
    if match:
        return aliases.get(match.group(1).lower(), match.group(1).upper())
    return raw


def _infer_country(row: dict[str, Any]) -> str:
    for col in ("country", "job_country", "source_country", "search_country"):
        if row.get(col):
            return _standard_country(row.get(col))
    for col in ("location", "search_location"):
        if row.get(col):
            return _standard_country(row.get(col))
    url = str(row.get("job_url") or row.get("url") or "").lower()
    if ".co.id" in url or "/id/" in url:
        return "ID"
    if ".com" in url and "indeed" in url:
        return "US"
    return "unknown"


def _detect_language(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return "unknown"
    try:
        from langdetect import detect

        return detect(text)
    except Exception:
        lower = text.lower()
        id_markers = [" yang ", " dan ", " untuk ", " dengan ", " pengalaman ", " pendidikan "]
        if any(marker in f" {lower} " for marker in id_markers):
            return "id"
        if re.search(r"[a-zA-Z]", text):
            return "en"
        return "unknown"


def _job_id_from_sentence_id(sentence_id: str) -> str:
    sid = str(sentence_id or "").strip()
    if "_" not in sid:
        return sid
    head, _sep, tail = sid.rpartition("_")
    return head if tail.isdigit() else sid


def _row_job_id(row: dict[str, Any]) -> str:
    for col in ("job_id", "id", "jobId", "jobID"):
        value = str(row.get(col) or "").strip()
        if value:
            return value
    return ""


def _desc_text(row: dict[str, Any]) -> str:
    for col in ("description", "job_description", "text", "requirements"):
        value = str(row.get(col) or "").strip()
        if value:
            return value
    return ""


def _title_text(row: dict[str, Any]) -> str:
    for col in ("title", "job_title", "position"):
        value = str(row.get(col) or "").strip()
        if value:
            return value
    return ""


def _date_text(row: dict[str, Any]) -> str:
    for col in ("job_date", "date_posted", "posted_at", "date"):
        value = str(row.get(col) or "").strip()
        if value:
            return value[:10]
    return ""


def profile_csv(path: Path | None, *, sample_language: int = 500) -> CsvProfile:
    if not path or not path.exists():
        return CsvProfile(path=str(path or ""), exists=False, job_ids=set())

    job_ids: set[str] = set()
    site_counts: Counter = Counter()
    source_counts: Counter = Counter()
    country_counts: Counter = Counter()
    lang_counts: Counter = Counter()
    fingerprints: Counter = Counter()
    missing_desc = 0
    missing_title = 0
    dates: list[str] = []
    row_count = 0
    lang_seen = 0

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            jid = _row_job_id(row)
            if jid:
                job_ids.add(jid)

            desc = _desc_text(row)
            title = _title_text(row)
            company = str(row.get("company") or "").strip().lower()
            if not desc:
                missing_desc += 1
            if not title:
                missing_title += 1
            if title or company or desc:
                norm_desc = re.sub(r"\s+", " ", desc.lower().strip())[:500]
                fp = hashlib.md5(f"{title.lower()}||{company}||{norm_desc}".encode("utf-8")).hexdigest()
                fingerprints[fp] += 1

            site = str(row.get("site") or row.get("source_site") or "").strip() or "unknown"
            source = str(row.get("source") or row.get("extractor_source") or row.get("site") or "").strip() or "unknown"
            site_counts[site] += 1
            source_counts[source] += 1
            country_counts[_infer_country(row)] += 1

            date = _date_text(row)
            if date:
                dates.append(date)

            if lang_seen < sample_language:
                lang_source = desc or title
                if lang_source:
                    lang_counts[_detect_language(lang_source)] += 1
                    lang_seen += 1

    duplicates = sum(v - 1 for v in fingerprints.values() if v > 1)
    return CsvProfile(
        path=str(path),
        exists=True,
        row_count=row_count,
        unique_job_ids=len(job_ids),
        job_ids=job_ids,
        missing_description_rate=_safe_ratio(missing_desc, row_count),
        missing_title_rate=_safe_ratio(missing_title, row_count),
        duplicate_rate=_safe_ratio(duplicates, row_count),
        date_min=min(dates) if dates else "",
        date_max=max(dates) if dates else "",
        site_distribution=_compact_distribution(site_counts),
        source_distribution=_compact_distribution(source_counts),
        country_distribution=_compact_distribution(country_counts),
        language_distribution=_compact_distribution(lang_counts),
    )


def read_job_ids_from_csv(path: Path | None, columns: Iterable[str]) -> set[str]:
    if not path or not path.exists():
        return set()
    cols = list(columns)
    out: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in cols:
                value = str(row.get(col) or "").strip()
                if value:
                    out.add(value)
                    break
    return out


def read_phase1_job_ids(phase1_dir: Path | None) -> set[str]:
    if not phase1_dir or not phase1_dir.exists():
        return set()
    out: set[str] = set()
    for name in ("advanced_skills.csv", "advanced_knowledge.csv", "job_sentences_mapping.csv"):
        out.update(read_job_ids_from_csv(phase1_dir / name, ("job_id",)))
    return out


def read_jsonl_sentence_job_ids(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("sentence_id") or rec.get("id") or ""
            jid = rec.get("job_id") or _job_id_from_sentence_id(str(sid))
            if jid:
                out.add(str(jid))
    return out


def profile_prepared_sentences(prepared_dir: Path | None) -> dict[str, Any]:
    if not prepared_dir or not prepared_dir.exists():
        return {
            "prepared_sentence_path": "",
            "prepared_sentence_count": 0,
            "prepared_sentence_unique_job_ids": 0,
            "prepared_original_text_coverage": 0.0,
            "prepared_pipeline_text_coverage": 0.0,
            "prepared_english_text_coverage": 0.0,
            "prepared_source_language_distribution": {},
            "prepared_normalized_language_distribution": {},
            "prepared_translation_status_distribution": {},
        }
    path = prepared_dir / "job_sentences_full.csv"
    if not path.exists():
        path = prepared_dir / "job_sentences_for_pipeline.csv"
    if not path.exists():
        path = prepared_dir / "jobs_sentences.csv"
    if not path.exists():
        return {
            "prepared_sentence_path": "",
            "prepared_sentence_count": 0,
            "prepared_sentence_unique_job_ids": 0,
            "prepared_original_text_coverage": 0.0,
            "prepared_pipeline_text_coverage": 0.0,
            "prepared_english_text_coverage": 0.0,
            "prepared_source_language_distribution": {},
            "prepared_normalized_language_distribution": {},
            "prepared_translation_status_distribution": {},
        }

    n = 0
    job_ids: set[str] = set()
    original = 0
    pipeline = 0
    english = 0
    source_lang: Counter = Counter()
    normalized_lang: Counter = Counter()
    translation_status: Counter = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n += 1
            jid = str(row.get("job_id") or "").strip()
            if jid:
                job_ids.add(jid)
            original_text = (
                str(row.get("sentence_text_original") or "").strip()
                or str(row.get("sentence_raw") or "").strip()
                or str(row.get("sentence_clean") or "").strip()
            )
            pipeline_text = str(row.get("sentence_text") or row.get("sentence_en") or "").strip()
            english_text = str(row.get("sentence_text_en") or "").strip()
            if original_text:
                original += 1
            if pipeline_text:
                pipeline += 1
            if english_text:
                english += 1
            source_lang[str(row.get("source_language") or row.get("lang") or "unknown")] += 1
            normalized_lang[str(row.get("normalized_language") or "unknown")] += 1
            translation_status[str(row.get("translation_status") or "unknown")] += 1
    return {
        "prepared_sentence_path": _rel(path),
        "prepared_sentence_count": n,
        "prepared_sentence_unique_job_ids": len(job_ids),
        "prepared_original_text_coverage": _safe_ratio(original, n),
        "prepared_pipeline_text_coverage": _safe_ratio(pipeline, n),
        "prepared_english_text_coverage": _safe_ratio(english, n),
        "prepared_source_language_distribution": _compact_distribution(source_lang),
        "prepared_normalized_language_distribution": _compact_distribution(normalized_lang),
        "prepared_translation_status_distribution": _compact_distribution(translation_status),
    }


def overlap_summary(left_name: str, left: set[str], right_name: str, right: set[str]) -> dict[str, Any]:
    inter = left & right
    return {
        "left": left_name,
        "right": right_name,
        "left_count": len(left),
        "right_count": len(right),
        "overlap_count": len(inter),
        "left_coverage": _safe_ratio(len(inter), len(left)),
        "right_coverage": _safe_ratio(len(inter), len(right)),
        "left_only_count": len(left - right),
        "right_only_count": len(right - left),
    }


def _candidate_raw_csvs() -> list[Path]:
    base = PROJECT_ROOT / "DATA" / "preprocessing"
    if not base.exists():
        return []
    return sorted(p for p in base.glob("*.csv") if p.is_file())


def infer_raw_csv(metadata_ids: set[str], explicit_raw: Path | None) -> tuple[Path | None, list[dict[str, Any]]]:
    candidates = []
    if explicit_raw and explicit_raw.exists():
        candidates.append(explicit_raw)
    candidates.extend(p for p in _candidate_raw_csvs() if p not in candidates)

    scored: list[dict[str, Any]] = []
    for path in candidates:
        ids = read_job_ids_from_csv(path, ("job_id", "id"))
        inter = ids & metadata_ids
        scored.append({
            "path": str(path),
            "unique_job_ids": len(ids),
            "overlap_with_metadata": len(inter),
            "metadata_coverage": _safe_ratio(len(inter), len(metadata_ids)),
        })
    scored.sort(key=lambda r: (-r["metadata_coverage"], -r["overlap_with_metadata"], r["path"]))
    best = Path(scored[0]["path"]) if scored and scored[0]["overlap_with_metadata"] else explicit_raw
    return best, scored[:10]


def _top_share(dist: dict[str, int] | None) -> tuple[str, float]:
    if not dist:
        return "unknown", 0.0
    total = sum(dist.values())
    if total <= 0:
        return "unknown", 0.0
    key, value = max(dist.items(), key=lambda kv: kv[1])
    return str(key), value / total


def decide_corpus_status(
    *,
    raw_profile: CsvProfile,
    prepared_sentence_profile: dict[str, Any],
    metadata_ids: set[str],
    phase1_ids: set[str],
    raw_to_metadata: dict[str, Any],
    metadata_to_phase1: dict[str, Any],
) -> tuple[str, list[str], str, str]:
    reasons: list[str] = []
    country, country_share = _top_share(raw_profile.country_distribution)
    language, language_share = _top_share(raw_profile.language_distribution)
    country_label = country
    if country not in ("unknown", "ID") and country_share < 0.50:
        country_label = "mixed_non_indonesian"
    if not metadata_ids:
        return STATUS_INVALID, ["missing_jobs_metadata"], country_label, language
    if not phase1_ids:
        return STATUS_INVALID, ["missing_phase1_job_ids"], country_label, language
    if not raw_profile.exists:
        return STATUS_REVIEW, ["raw_csv_not_found_or_not_inferred"], "unknown", "unknown"

    raw_cov = float(raw_to_metadata.get("right_coverage") or 0.0)
    phase_cov = float(metadata_to_phase1.get("right_coverage") or 0.0)
    if raw_cov < 0.95:
        reasons.append("raw_to_metadata_overlap_below_0.95")
    if phase_cov < 0.80:
        reasons.append("metadata_to_phase1_overlap_below_0.80")
    if reasons:
        return STATUS_INVALID, reasons, country_label, language

    bilingual_reasons: list[str] = []
    if float(prepared_sentence_profile.get("prepared_original_text_coverage") or 0.0) < 0.95:
        bilingual_reasons.append("prepared_original_text_coverage_below_0.95")
    if float(prepared_sentence_profile.get("prepared_english_text_coverage") or 0.0) < 0.95:
        bilingual_reasons.append("prepared_english_text_coverage_below_0.95")

    if country == "ID" and country_share >= 0.80:
        if bilingual_reasons:
            return STATUS_REVIEW, ["indonesian_country_majority", *bilingual_reasons], country, language
        return STATUS_VALID_INDONESIAN, ["indonesian_country_majority"], country, language
    if language == "id" and language_share >= 0.80:
        if bilingual_reasons:
            return STATUS_REVIEW, ["indonesian_language_majority", *bilingual_reasons], country_label, language
        return STATUS_VALID_INDONESIAN, ["indonesian_language_majority"], country_label, language
    if language == "en" and language_share >= 0.80 and country != "ID":
        return STATUS_VALID_ENGINEERING, ["english_non_indonesian_engineering_corpus"], country_label, language
    if country not in ("unknown", "ID") and country_share >= 0.50:
        return STATUS_VALID_ENGINEERING, ["non_indonesian_country_majority"], country, language
    return STATUS_REVIEW, ["mixed_or_unknown_country_language_distribution"], country_label, language


def _metadata_education_coverage(path: Path | None) -> float:
    if not path or not path.exists():
        return 0.0
    total = 0
    known = 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if str(row.get("min_education_kkni") or row.get("education_labels") or "").strip():
                known += 1
    return _safe_ratio(known, total)


def _distribution_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_key, dist in [
        ("site", report.get("site_distribution")),
        ("source", report.get("source_distribution")),
        ("country", report.get("country_distribution")),
        ("language", report.get("language_distribution")),
        ("prepared_source_language", report.get("prepared_source_language_distribution")),
        ("prepared_normalized_language", report.get("prepared_normalized_language_distribution")),
        ("prepared_translation_status", report.get("prepared_translation_status_distribution")),
    ]:
        total = sum((dist or {}).values()) or 0
        for value, count in (dist or {}).items():
            rows.append({
                "group": group_key,
                "value": value,
                "count": count,
                "share": _safe_ratio(int(count), total),
            })
    return rows


def _quality_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    keys = [
        "raw_row_count",
        "raw_unique_job_ids",
        "metadata_unique_job_ids",
        "phase1_unique_job_ids",
        "prepared_sentence_count",
        "prepared_sentence_unique_job_ids",
        "prepared_original_text_coverage",
        "prepared_pipeline_text_coverage",
        "prepared_english_text_coverage",
        "skill_llm_input_unique_job_ids",
        "skill_llm_output_unique_job_ids",
        "duplicate_rate",
        "missing_description_rate",
        "missing_title_rate",
        "education_metadata_coverage",
        "corpus_validity_status",
    ]
    return [{"metric": key, "value": report.get(key, "")} for key in keys]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit_corpus(
    *,
    dataset_id: str,
    raw_csv: Path | None,
    prepared_dir: Path | None,
    phase1_dir: Path | None,
    skill_llm_input: Path | None,
    skill_llm_output: Path | None,
    jobs_metadata: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    prepared_dir = prepared_dir or (jobs_metadata.parent if jobs_metadata else None)
    jobs_metadata = jobs_metadata or ((prepared_dir / "jobs_metadata.csv") if prepared_dir else None)

    metadata_ids = read_job_ids_from_csv(jobs_metadata, ("job_id",))
    inferred_raw, raw_candidates = infer_raw_csv(metadata_ids, raw_csv)
    raw_profile = profile_csv(inferred_raw)
    prepared_sentence_profile = profile_prepared_sentences(prepared_dir)
    raw_ids = raw_profile.job_ids or set()
    phase1_ids = read_phase1_job_ids(phase1_dir)
    skill_input_ids = read_jsonl_sentence_job_ids(skill_llm_input)
    skill_output_ids = read_jsonl_sentence_job_ids(skill_llm_output)

    raw_to_metadata = overlap_summary("raw_csv", raw_ids, "jobs_metadata", metadata_ids)
    metadata_to_phase1 = overlap_summary("jobs_metadata", metadata_ids, "phase1", phase1_ids)
    metadata_to_skill_input = overlap_summary("jobs_metadata", metadata_ids, "skill_llm_input", skill_input_ids)
    skill_input_to_output = overlap_summary("skill_llm_input", skill_input_ids, "skill_llm_output", skill_output_ids)

    status, status_reasons, corpus_country, corpus_language = decide_corpus_status(
        raw_profile=raw_profile,
        prepared_sentence_profile=prepared_sentence_profile,
        metadata_ids=metadata_ids,
        phase1_ids=phase1_ids,
        raw_to_metadata=raw_to_metadata,
        metadata_to_phase1=metadata_to_phase1,
    )

    report = {
        "schema_version": "corpus_lineage_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "raw_csv_path": _rel(inferred_raw),
        "raw_csv_requested_path": _rel(raw_csv),
        "raw_candidate_matches": raw_candidates,
        "raw_row_count": raw_profile.row_count,
        "raw_unique_job_ids": raw_profile.unique_job_ids,
        "prepared_dir": _rel(prepared_dir),
        "phase1_dir": _rel(phase1_dir),
        **prepared_sentence_profile,
        "skill_llm_input_path": _rel(skill_llm_input),
        "skill_llm_output_path": _rel(skill_llm_output),
        "jobs_metadata_path": _rel(jobs_metadata),
        "metadata_unique_job_ids": len(metadata_ids),
        "phase1_unique_job_ids": len(phase1_ids),
        "skill_llm_input_unique_job_ids": len(skill_input_ids),
        "skill_llm_output_unique_job_ids": len(skill_output_ids),
        "job_id_overlap_raw_to_metadata": raw_to_metadata,
        "job_id_overlap_metadata_to_phase1": metadata_to_phase1,
        "job_id_overlap_metadata_to_skill_llm_input": metadata_to_skill_input,
        "job_id_overlap_skill_llm_input_to_output": skill_input_to_output,
        "language_distribution": raw_profile.language_distribution or {},
        "country_distribution": raw_profile.country_distribution or {},
        "site_distribution": raw_profile.site_distribution or {},
        "source_distribution": raw_profile.source_distribution or {},
        "date_range": {
            "min": raw_profile.date_min,
            "max": raw_profile.date_max,
        },
        "duplicate_rate": raw_profile.duplicate_rate,
        "missing_description_rate": raw_profile.missing_description_rate,
        "missing_title_rate": raw_profile.missing_title_rate,
        "education_metadata_coverage": _metadata_education_coverage(jobs_metadata),
        "corpus_country": corpus_country,
        "corpus_language": corpus_language,
        "corpus_validity_status": status,
        "corpus_validity_reasons": status_reasons,
    }

    coverage_rows = [
        raw_to_metadata,
        metadata_to_phase1,
        metadata_to_skill_input,
        skill_input_to_output,
    ]
    return report, coverage_rows, _distribution_rows(report), _quality_rows(report)


def write_audit_outputs(
    out_dir: Path,
    report: dict[str, Any],
    coverage_rows: list[dict[str, Any]],
    distribution_rows: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "corpus_lineage_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(out_dir / "job_id_coverage.csv", coverage_rows, [
        "left", "right", "left_count", "right_count", "overlap_count",
        "left_coverage", "right_coverage", "left_only_count", "right_only_count",
    ])
    _write_csv(out_dir / "source_distribution.csv", distribution_rows, [
        "group", "value", "count", "share",
    ])
    _write_csv(out_dir / "dedupe_and_quality_summary.csv", quality_rows, [
        "metric", "value",
    ])
    (out_dir / "corpus_lineage_report.md").write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    overlap = report.get("job_id_overlap_metadata_to_phase1", {})
    raw_overlap = report.get("job_id_overlap_raw_to_metadata", {})
    return f"""# Corpus Lineage Audit: {report.get('dataset_id')}

Generated: {report.get('created_at')}

## Status

| Field | Value |
|---|---|
| Corpus validity status | `{report.get('corpus_validity_status')}` |
| Reasons | {', '.join(report.get('corpus_validity_reasons') or [])} |
| Corpus country | `{report.get('corpus_country')}` |
| Corpus language | `{report.get('corpus_language')}` |

## Lineage

| Stage | Path |
|---|---|
| Raw CSV | `{report.get('raw_csv_path')}` |
| Prepared directory | `{report.get('prepared_dir')}` |
| Phase-1 directory | `{report.get('phase1_dir')}` |
| Prepared sentences | `{report.get('prepared_sentence_path')}` |
| Skill-LLM input | `{report.get('skill_llm_input_path')}` |
| Skill-LLM output | `{report.get('skill_llm_output_path')}` |
| Jobs metadata | `{report.get('jobs_metadata_path')}` |

## Coverage

| Link | Overlap |
|---|---:|
| Raw -> metadata | {raw_overlap.get('overlap_count', 0)} ({raw_overlap.get('right_coverage', 0):.2%} of metadata) |
| Metadata -> Phase-1 | {overlap.get('overlap_count', 0)} ({overlap.get('right_coverage', 0):.2%} of Phase-1) |

## Quality

| Metric | Value |
|---|---:|
| Raw rows | {report.get('raw_row_count')} |
| Raw unique jobs | {report.get('raw_unique_job_ids')} |
| Metadata unique jobs | {report.get('metadata_unique_job_ids')} |
| Phase-1 unique jobs | {report.get('phase1_unique_job_ids')} |
| Prepared sentences | {report.get('prepared_sentence_count')} |
| Original text coverage | {report.get('prepared_original_text_coverage')} |
| English-normalized text coverage | {report.get('prepared_english_text_coverage')} |
| Duplicate rate | {report.get('duplicate_rate')} |
| Missing description rate | {report.get('missing_description_rate')} |
| Missing title rate | {report.get('missing_title_rate')} |
| Education metadata coverage | {report.get('education_metadata_coverage')} |

## Interpretation

This audit establishes data lineage and corpus context only. It does not certify
that any recommendation is official-ready. Recommendation readiness must still
pass grounding, mapping, expert-review, and stability gates.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="n1k_openrouter_gpt54")
    parser.add_argument("--raw-csv", default="")
    parser.add_argument("--prepared-dir", default="DATA/preprocessing/data_prepared_n1k")
    parser.add_argument("--phase1-dir", default="DATA/preprocessing/phase1_n1k")
    parser.add_argument("--skill-llm-input", default="skill_llm_input_n1k.jsonl")
    parser.add_argument("--skill-llm-output", default="skill_llm_extractions.jsonl")
    parser.add_argument("--jobs-metadata", default="")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    dataset_id = args.dataset_id
    out_dir = _resolve(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "corpus_audits" / dataset_id
    report, coverage_rows, distribution_rows, quality_rows = audit_corpus(
        dataset_id=dataset_id,
        raw_csv=_resolve(args.raw_csv),
        prepared_dir=_resolve(args.prepared_dir),
        phase1_dir=_resolve(args.phase1_dir),
        skill_llm_input=_resolve(args.skill_llm_input),
        skill_llm_output=_resolve(args.skill_llm_output),
        jobs_metadata=_resolve(args.jobs_metadata),
    )
    write_audit_outputs(out_dir, report, coverage_rows, distribution_rows, quality_rows)
    print(f"[OK] Corpus audit written to: {out_dir}")
    print(f"     status={report['corpus_validity_status']}")
    print(f"     raw={report['raw_csv_path']}")
    print(f"     metadata_jobs={report['metadata_unique_job_ids']} phase1_jobs={report['phase1_unique_job_ids']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
