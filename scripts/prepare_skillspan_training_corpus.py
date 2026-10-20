"""Prepare raw job CSVs for a SkillSpan-style annotation run.

The output is sentence-level JSONL. It is intentionally not GPT-annotated yet:
this stage only cleans, splits, scores, samples, and records provenance so the
paid annotation stage can be resumed and audited.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9A-ZÀ-ÖØ-Þ])")
BULLET_OR_PARAGRAPH_RE = re.compile(r"\n\s*(?:[-*•·]|\d+[.)])\s+|\n{2,}")
HTML_RE = re.compile(r"<[^>]+>")
MARKDOWN_RE = re.compile(r"[*_`#]+")
SPACE_RE = re.compile(r"\s+")

HEADER_RE = re.compile(
    r"^\s*(responsibilities|requirements?|qualifications?|skills?|knowledge|"
    r"kualifikasi|persyaratan|tanggung jawab|keahlian|kemampuan|pengalaman)\s*:?\s*$",
    re.I,
)
BOILERPLATE_RE = re.compile(
    r"\b(equal opportunity|privacy notice|apply now|benefits?|salary|transport|"
    r"lunch|holiday|medical insurance|job types?|jenis pekerjaan|gaji|"
    r"lamaran|candidate privacy|data protection notice)\b",
    re.I,
)
TECH_RE = re.compile(
    r"\b(python|java\b|javascript|typescript|php|c#|\.net|sql|mysql|postgres|"
    r"database|rdbms|api|rest|graphql|backend|frontend|react|angular|vue|"
    r"spring|django|node|aws|azure|gcp|cloud|docker|kubernetes|terraform|"
    r"linux|git|devops|ci/cd|testing|etl|pipeline|machine learning|ai\b|"
    r"nlp|security|html|css|jquery|android|ios|swift|kotlin|salesforce|"
    r"tableau|power bi)\b",
    re.I,
)
ID_TECH_RE = re.compile(
    r"\b(pemrograman|pemograman|basis data|database|rdbms|restful api|api|"
    r"framework|repository|menguasai|memahami|mengimplementasikan|"
    r"mendokumentasikan|pengujian|keamanan|jaringan|sistem informasi)\b",
    re.I,
)
SOFT_RE = re.compile(
    r"\b(communication|collaboration|teamwork|leadership|stakeholder|mentor|"
    r"problem[- ]solving|analytical|detail[- ]oriented|initiative|adaptable|"
    r"komunikasi|inisiatif|teliti|kerja sama|bekerja sama|kepemimpinan|"
    r"analitis|tekanan|mandiri|proaktif)\b",
    re.I,
)
REQ_RE = re.compile(
    r"\b(required|requirements?|qualification|experience|proficient|ability|"
    r"responsible|design|develop|build|implement|maintain|manage|test|"
    r"mampu|menguasai|memahami|memiliki|terbiasa|pengalaman|pendidikan|"
    r"bertanggung jawab|merancang|mengembangkan|membangun|menguji)\b",
    re.I,
)


def _path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


def clean_text(text: Any) -> str:
    s = str(text or "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\\+([\-&,#])", r"\1", s)
    s = HTML_RE.sub(" ", s)
    s = MARKDOWN_RE.sub("", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"https?://\S+", " ", s)
    lines = [SPACE_RE.sub(" ", line).strip() for line in s.split("\n")]
    s = "\n".join(line for line in lines if line)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


def split_sentences(text: str) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    parts = BULLET_OR_PARAGRAPH_RE.split(cleaned)
    out: list[str] = []
    for part in parts:
        part = SPACE_RE.sub(" ", part.replace("\n", " ")).strip(" -•·")
        if not part:
            continue
        if HEADER_RE.match(part):
            out.append(part)
            continue
        out.extend(s.strip(" -•·") for s in SENT_SPLIT_RE.split(part) if s.strip(" -•·"))
    return out


def score_sentence(sentence: str, *, language: str) -> tuple[float, list[str]]:
    s = sentence.strip()
    flags: list[str] = []
    score = 0.0
    n = len(s)
    if 35 <= n <= 420:
        score += 1.0
        flags.append("sane_length")
    elif 15 <= n <= 700:
        score += 0.25
        flags.append("borderline_length")
    else:
        score -= 1.5
        flags.append("bad_length")
    if TECH_RE.search(s):
        score += 2.0
        flags.append("technical_signal")
    if language == "id" and ID_TECH_RE.search(s):
        score += 1.25
        flags.append("indonesian_technical_signal")
    if SOFT_RE.search(s):
        score += 1.25
        flags.append("soft_skill_signal")
    if REQ_RE.search(s):
        score += 1.5
        flags.append("requirement_or_action_signal")
    if BOILERPLATE_RE.search(s):
        score -= 1.0
        flags.append("possible_boilerplate")
    if HEADER_RE.match(s):
        score -= 1.0
        flags.append("header_only")
    return round(score, 3), flags


def iter_sentence_records(
    csv_path: Path,
    *,
    dataset_id: str,
    language: str,
    job_id_col: str,
    title_col: str,
    description_col: str,
    source_col: str,
    min_len: int,
    max_len: int,
    min_score: float,
    max_sentences_per_job: int,
    max_jobs: int,
    chunksize: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_sentence_text: set[str] = set()
    counters: Counter[str] = Counter()
    jobs_seen = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        for row_idx, row in chunk.iterrows():
            if max_jobs and jobs_seen >= max_jobs:
                break
            desc = str(row.get(description_col) or "").strip()
            if not desc:
                counters["jobs_missing_description"] += 1
                continue
            jobs_seen += 1
            job_id = str(row.get(job_id_col) or row_idx).strip()
            title = str(row.get(title_col) or "").strip()
            source = str(row.get(source_col) or "").strip() if source_col in row else ""
            candidates = []
            for local_idx, sent in enumerate(split_sentences(desc), start=1):
                counters["raw_sentences"] += 1
                sent = SPACE_RE.sub(" ", sent).strip()
                if len(sent) < min_len or len(sent) > max_len:
                    counters["sentences_dropped_length"] += 1
                    continue
                score, flags = score_sentence(sent, language=language)
                if score < min_score:
                    counters["sentences_dropped_score"] += 1
                    continue
                dedupe_key = sent.lower()
                if dedupe_key in seen_sentence_text:
                    counters["sentences_dropped_duplicate"] += 1
                    continue
                candidates.append((score, local_idx, sent, flags))
            candidates.sort(key=lambda x: (-x[0], x[1]))
            if max_sentences_per_job > 0:
                candidates = candidates[:max_sentences_per_job]
            for kept_idx, (score, local_idx, sent, flags) in enumerate(candidates, start=1):
                seen_sentence_text.add(sent.lower())
                records.append({
                    "dataset_id": dataset_id,
                    "language": language,
                    "idx": jobs_seen,
                    "job_id": job_id,
                    "sentence_id": f"{job_id}_{local_idx:04d}",
                    "title": title,
                    "sentence_text": sent,
                    "source": source,
                    "source_file": str(csv_path),
                    "sentence_score": score,
                    "score_reasons": flags,
                    "rank_within_job": kept_idx,
                })
                counters["sentences_kept"] += 1
            if not candidates:
                counters["jobs_without_kept_sentence"] += 1
        if max_jobs and jobs_seen >= max_jobs:
            break
    counters["jobs_seen"] = jobs_seen
    return records, dict(counters)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw job CSV path.")
    parser.add_argument("--output-dir", required=True, help="Directory for prepared sentence files.")
    parser.add_argument("--dataset-id", required=True, help="Stable dataset id, e.g. us_akkasi_en or rpl_id.")
    parser.add_argument("--language", required=True, choices=["en", "id"], help="Primary corpus language.")
    parser.add_argument("--job-id-col", default="id")
    parser.add_argument("--title-col", default="title")
    parser.add_argument("--description-col", default="description")
    parser.add_argument("--source-col", default="site")
    parser.add_argument("--min-len", type=int, default=15)
    parser.add_argument("--max-len", type=int, default=700)
    parser.add_argument("--min-score", type=float, default=1.0)
    parser.add_argument("--max-sentences-per-job", type=int, default=12)
    parser.add_argument("--max-jobs", type=int, default=0, help="0 = no cap.")
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap after sentence ranking, 0 = no cap.")
    parser.add_argument("--chunksize", type=int, default=1000)
    args = parser.parse_args()

    input_path = _path(args.input)
    output_dir = _path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        print(f"[ERROR] input not found: {input_path}", file=sys.stderr)
        return 2

    records, summary = iter_sentence_records(
        input_path,
        dataset_id=args.dataset_id,
        language=args.language,
        job_id_col=args.job_id_col,
        title_col=args.title_col,
        description_col=args.description_col,
        source_col=args.source_col,
        min_len=args.min_len,
        max_len=args.max_len,
        min_score=args.min_score,
        max_sentences_per_job=args.max_sentences_per_job,
        max_jobs=args.max_jobs,
        chunksize=args.chunksize,
    )
    if args.max_records and args.max_records > 0:
        records = records[:args.max_records]
    summary.update({
        "schema_version": "2026-06-23.skillspan_ours.prepare.v1",
        "dataset_id": args.dataset_id,
        "language": args.language,
        "input": str(input_path),
        "output_dir": str(output_dir),
        "records_written": len(records),
        "max_jobs": args.max_jobs,
        "max_records": args.max_records,
        "max_sentences_per_job": args.max_sentences_per_job,
        "min_score": args.min_score,
    })

    write_jsonl(output_dir / "sentences.jsonl", records)
    write_csv(output_dir / "sentences.csv", records)
    (output_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[INFO] wrote {len(records)} sentence candidates to {output_dir / 'sentences.jsonl'}")
    print(f"[INFO] summary: {output_dir / 'prepare_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
