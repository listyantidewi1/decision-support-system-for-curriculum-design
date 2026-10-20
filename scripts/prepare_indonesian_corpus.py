"""
Prepare an Indonesian job corpus for the competency-v2 evidence pipeline.

This script is intentionally deterministic and does not call an LLM. It turns a
raw Indonesian job CSV into a traceable prepared directory:

    DATA/preprocessing/data_prepared_rpl_id_raw_2/
        jobs_metadata.csv
        job_sentences_full.csv
        job_sentences_for_pipeline.csv
        jobs_sentences.csv
        jobs_sentences_filtered.csv
        corpus_preparation_report.json

It also writes a Skill-LLM input JSONL by default. Original Indonesian evidence
is preserved in `sentence_text_original`; English-normalized text lives in
`sentence_text_en` when supplied by a translation map. Until then, pipeline text
falls back to the cleaned original and `translation_status` makes that explicit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RAW = PROJECT_ROOT / "DATA" / "preprocessing" / "jobs_rpl_id_raw_2.csv"
DEFAULT_OUT = PROJECT_ROOT / "DATA" / "preprocessing" / "data_prepared_rpl_id_raw_2"
DEFAULT_SKILL_INPUT = PROJECT_ROOT / "DATA" / "preprocessing" / "skill_llm_input_rpl_id_raw_2.jsonl"

BULLET_OR_PARAGRAPH_RE = re.compile(
    r"\n\s*(?:[-*•·]|\d+[\).])\s+|\n{2,}|\\n\s*(?:[-*•·]|\d+[\).])\s+|\\n{2,}"
)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z0-9A-ZÀ-ÖØ-Þ])")
BULLET_RE = re.compile(r"^[\s\-*•·\\/\d\).]+")

ID_MARKERS = {
    "yang", "dan", "untuk", "dengan", "mampu", "memiliki", "menguasai",
    "pengalaman", "pendidikan", "minimal", "bersedia", "bekerja", "dibawah",
    "tekanan", "lulusan", "jurusan", "bertanggung", "melakukan",
}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def clean_text(text: Any) -> str:
    s = _safe_text(text)
    s = s.replace("\r", " ").replace("\\r", " ")
    s = s.replace("\n", " ").replace("\\n", " ")
    s = s.replace("\xa0", " ")
    s = s.replace("*", "").replace("`", "")
    s = re.sub(r"[\\/]{2,}", " ", s)
    s = re.sub(r"\s*[/\\]\s*-\s*[/\\]?\s*", " ", s)
    s = BULLET_RE.sub("", s)
    s = re.sub(r"([!?.,;:])\1+", r"\1", s)
    s = re.sub(r"\(\s*\)", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip(" \"'")


def split_sentences(text: Any) -> list[str]:
    raw = _safe_text(text)
    if not raw.strip():
        return []
    parts = BULLET_OR_PARAGRAPH_RE.split(raw)
    out: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        subparts = SENT_SPLIT_RE.split(part)
        out.extend(s.strip() for s in subparts if s.strip())
    return out


def detect_language(text: str, detector: str = "heuristic") -> str:
    lower_tokens = set(re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ]+", f" {text.lower()} "))
    id_hits = len(lower_tokens & ID_MARKERS)
    if id_hits >= 1:
        return "id"
    if detector == "heuristic":
        return "en" if re.search(r"[a-zA-Z]", text or "") else "unknown"
    try:
        from langdetect import detect

        detected = detect(text)
    except Exception:
        detected = "unknown"
    return detected


def country_from_location(location: Any) -> str:
    loc = _safe_text(location).lower()
    if not loc:
        return "unknown"
    if loc == "id" or loc.endswith(", id") or "indonesia" in loc:
        return "ID"
    return "unknown"


def job_fingerprint(row: pd.Series) -> str:
    title = clean_text(row.get("title", "")).lower()
    company = clean_text(row.get("company", "")).lower()
    desc = re.sub(r"\s+", " ", clean_text(row.get("description", "")).lower())[:800]
    return hashlib.md5(f"{title}||{company}||{desc}".encode("utf-8")).hexdigest()


def load_translation_map(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}
    if path.suffix.lower() == ".jsonl":
        out: dict[str, str] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sid = str(rec.get("sentence_id") or "").strip()
                text = str(rec.get("sentence_text_en") or rec.get("translation") or rec.get("sentence_text") or "").strip()
                if sid and text:
                    out[sid] = text
        return out
    out = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = str(row.get("sentence_id") or "").strip()
            text = str(row.get("sentence_text_en") or row.get("translation") or "").strip()
            if sid and text:
                out[sid] = text
    return out


def _education_annotate(df: pd.DataFrame) -> None:
    try:
        from education_level_extractor import annotate_dataframe

        text_cols = [c for c in ("description",) if c in df.columns]
        if text_cols:
            annotate_dataframe(df, text_columns=text_cols)
            return
    except Exception as exc:
        print(f"[WARN] education-level annotation failed: {exc}")
    df["min_education_kkni"] = None
    df["education_labels"] = ""


def build_metadata(df: pd.DataFrame, raw_path: Path) -> pd.DataFrame:
    meta = pd.DataFrame()
    meta["job_id"] = df["job_id"]
    meta["raw_job_id"] = df["raw_job_id"]
    meta["raw_row_index"] = df["raw_row_index"]
    meta["title"] = df.get("title", "").fillna("").astype(str)
    meta["title_original"] = meta["title"]
    meta["company"] = df.get("company", "").fillna("").astype(str)
    meta["location"] = df.get("location", "").fillna("").astype(str)
    meta["country"] = meta["location"].apply(country_from_location)
    meta["corpus_country"] = "ID"
    meta["site"] = df.get("site", "").fillna("").astype(str)
    meta["source"] = meta["site"]
    meta["job_url"] = df.get("job_url", "").fillna("").astype(str)
    meta["job_date"] = pd.to_datetime(df.get("date_posted", pd.NaT), errors="coerce").dt.date.astype(str)
    meta["query_term"] = df.get("query_term", "").fillna("").astype(str)
    meta["query_family"] = df.get("query_family", "").fillna("").astype(str)
    meta["description_available"] = df.get("description", "").fillna("").astype(str).str.strip().str.len() > 0
    meta["description_language"] = df["_description_language"]
    meta["min_education_kkni"] = df.get("min_education_kkni", "")
    meta["education_labels"] = df.get("education_labels", "")
    meta["raw_csv_path"] = str(raw_path)
    return meta.drop_duplicates(subset=["job_id"])


def build_sentence_rows(
    df: pd.DataFrame,
    *,
    raw_path: Path,
    translation_map: dict[str, str],
    min_len: int,
    max_len: int,
    language_detector: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        job_id = row["job_id"]
        job_date = row.get("job_date", "")
        raw_sents = split_sentences(row.get("description", ""))
        sid = 0
        for raw_sent in raw_sents:
            clean = clean_text(raw_sent)
            if len(clean) < min_len or len(clean) > max_len:
                continue
            sid += 1
            sentence_id = f"{job_id}_{sid:04d}"
            lang = detect_language(clean, language_detector)
            if row.get("_description_language") == "id" and lang in {"en", "unknown"}:
                lang = "id"
            sentence_en = translation_map.get(sentence_id, "").strip()
            if sentence_en:
                sentence_text = sentence_en
                normalized_language = "en"
                translation_status = "translated_external"
            elif lang == "en":
                sentence_en = clean
                sentence_text = clean
                normalized_language = "en"
                translation_status = "already_english"
            else:
                sentence_text = clean
                normalized_language = lang or "unknown"
                translation_status = "original_not_translated"

            rows.append({
                "job_id": job_id,
                "job_date": job_date,
                "sentence_id": sentence_id,
                "sentence_raw": raw_sent,
                "sentence_text_original": clean,
                "sentence_clean": clean,
                "sentence_text_en": sentence_en,
                "sentence_text": sentence_text,
                "source_language": lang,
                "normalized_language": normalized_language,
                "translation_status": translation_status,
                "query_family": row.get("query_family", ""),
                "query_term": row.get("query_term", ""),
                "site": row.get("site", ""),
                "raw_csv_path": str(raw_path),
            })
    return pd.DataFrame(rows)


def write_skill_llm_input(path: Path, sentences: pd.DataFrame) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for _, row in sentences.iterrows():
            text = _safe_text(row.get("sentence_text")).strip()
            if not text:
                continue
            rec = {
                "sentence_id": row["sentence_id"],
                "job_id": row["job_id"],
                "sentence_text": text,
                "sentence_text_original": row.get("sentence_text_original", ""),
                "sentence_text_en": row.get("sentence_text_en", ""),
                "source_language": row.get("source_language", ""),
                "normalized_language": row.get("normalized_language", ""),
                "translation_status": row.get("translation_status", ""),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_RAW), help="Raw Indonesian job CSV.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT), help="Prepared output directory.")
    parser.add_argument("--skill-llm-output", default=str(DEFAULT_SKILL_INPUT), help="Skill-LLM input JSONL to write.")
    parser.add_argument("--translation-map", default="", help="Optional CSV/JSONL with sentence_id and sentence_text_en/translation.")
    parser.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True, help="Deduplicate jobs before sentence split.")
    parser.add_argument("--max-jobs", type=int, default=0, help="Optional cap for smoke tests.")
    parser.add_argument("--min-len", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=600)
    parser.add_argument(
        "--language-detector",
        choices=["heuristic", "langdetect"],
        default="heuristic",
        help="Use fast Indonesian-marker heuristic by default; langdetect is slower and noisy on short technical bullets.",
    )
    args = parser.parse_args()

    raw_path = _resolve(args.input)
    out_dir = _resolve(args.output_dir)
    skill_input_path = _resolve(args.skill_llm_output)
    translation_map_path = _resolve(args.translation_map) if args.translation_map else None
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_path)
    raw_row_count = len(df)
    df = df.reset_index(drop=False).rename(columns={"index": "raw_row_index"})
    if "id" in df.columns:
        df["raw_job_id"] = df["id"].fillna("").astype(str)
    else:
        df["raw_job_id"] = df["raw_row_index"].astype(str)
    df["job_id"] = df["raw_job_id"].where(df["raw_job_id"].str.len() > 0, df["raw_row_index"].astype(str))
    df["job_date"] = pd.to_datetime(df.get("date_posted", pd.NaT), errors="coerce").dt.date.astype(str)
    df["_description_language"] = df.get("description", "").fillna("").astype(str).apply(
        lambda s: detect_language(s[:2000], args.language_detector) if s.strip() else "missing"
    )

    if args.dedupe:
        before = len(df)
        df["_fingerprint"] = df.apply(job_fingerprint, axis=1)
        df = df.drop_duplicates(subset=["_fingerprint"], keep="first").drop(columns=["_fingerprint"])
        dedupe_removed = before - len(df)
    else:
        dedupe_removed = 0

    if args.max_jobs > 0:
        df = df.head(args.max_jobs).copy()

    _education_annotate(df)
    metadata = build_metadata(df, raw_path)
    metadata.to_csv(out_dir / "jobs_metadata.csv", index=False, encoding="utf-8-sig")

    translation_map = load_translation_map(translation_map_path)
    sentences = build_sentence_rows(
        df,
        raw_path=raw_path,
        translation_map=translation_map,
        min_len=args.min_len,
        max_len=args.max_len,
        language_detector=args.language_detector,
    )
    sentences.to_csv(out_dir / "job_sentences_full.csv", index=False, encoding="utf-8-sig")

    pipeline_cols = [
        "job_id", "job_date", "sentence_id", "sentence_text",
        "sentence_text_original", "sentence_text_en", "source_language",
        "normalized_language", "translation_status",
    ]
    pipeline_df = sentences[pipeline_cols].copy()
    pipeline_df.to_csv(out_dir / "job_sentences_for_pipeline.csv", index=False, encoding="utf-8-sig")
    pipeline_df.to_csv(out_dir / "jobs_sentences.csv", index=False, encoding="utf-8-sig")
    pipeline_df.to_csv(out_dir / "jobs_sentences_filtered.csv", index=False, encoding="utf-8-sig")
    n_skill_input = write_skill_llm_input(skill_input_path, pipeline_df)

    status_counts = Counter(sentences.get("translation_status", []))
    language_counts = Counter(sentences.get("source_language", []))
    query_counts = Counter(metadata.get("query_family", []))
    report = {
        "schema_version": "indonesian_corpus_preparation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_csv_path": str(raw_path),
        "output_dir": str(out_dir),
        "skill_llm_input_path": str(skill_input_path),
        "raw_row_count": int(raw_row_count),
        "prepared_job_count": int(len(metadata)),
        "dedupe_removed_jobs": int(dedupe_removed),
        "sentence_count": int(len(sentences)),
        "skill_llm_input_records": int(n_skill_input),
        "description_missing_rate": round(float((~metadata["description_available"]).mean()), 6) if len(metadata) else 0.0,
        "education_metadata_coverage": round(float(metadata["min_education_kkni"].notna().mean()), 6) if len(metadata) else 0.0,
        "source_language_distribution": dict(language_counts),
        "translation_status_distribution": dict(status_counts),
        "query_family_distribution": dict(query_counts),
        "readiness_notes": [
            "Original evidence is preserved in sentence_text_original.",
            "English-normalized text is available in sentence_text_en only when supplied by --translation-map or when source language is already English.",
            "Rows with translation_status=original_not_translated should not be treated as English-normalized evidence.",
            "This step prepares Skill-LLM input; Phase-1 evidence is not complete until Skill-LLM output is converted.",
        ],
    }
    write_report(out_dir / "corpus_preparation_report.json", report)

    print(f"[OK] Prepared Indonesian corpus: {out_dir}")
    print(f"     jobs: {len(metadata)} (dedupe removed {dedupe_removed})")
    print(f"     sentences: {len(sentences)}")
    print(f"     Skill-LLM input: {skill_input_path} ({n_skill_input} records)")
    print(f"     translation status: {dict(status_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
