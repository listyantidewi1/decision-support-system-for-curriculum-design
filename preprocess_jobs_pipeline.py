"""
preprocess_jobs_pipeline.py

Unified preprocessing pipeline:

- Reads raw job postings from DATA/preprocessing/jobs_rpl_id_raw_2.csv
- Splits descriptions into sentences (with job_id + job_date)
- Cleans markdown/noise
- Detects language + (optional) translates to English
- Removes duplicate sentences (optional)
- Saves:
    DATA/preprocessing/data_prepared/jobs_metadata.csv
    DATA/preprocessing/data_prepared/job_sentences_full.csv
    DATA/preprocessing/data_prepared/job_sentences_for_pipeline.csv
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List

import pandas as pd
from langdetect import detect, LangDetectException

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import config
    _JOBS_SCRAPING_CSV = config.JOBS_SCRAPING_CSV
    _PREPROCESS_OUTPUT_DIR = config.PREPROCESS_OUTPUT_DIR
except ImportError:
    _JOBS_SCRAPING_CSV = Path("job_scraping/output/english_jobs.csv")
    _PREPROCESS_OUTPUT_DIR = Path("DATA/preprocessing/data_prepared")


# -------------------------------------------------------
# Sentence splitting
# -------------------------------------------------------

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
# Split on newline+bullet or paragraph breaks (keeps segments under ~128 BERT tokens)
BULLET_OR_PARAGRAPH_RE = re.compile(r"\n\s*[\-\*\u2022\u00b7•·]\s+|\n\n+")


def split_sentences(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    # First split on bullet boundaries and paragraph breaks
    parts = BULLET_OR_PARAGRAPH_RE.split(text)
    result = []
    for part in parts:
        part = part.replace("\r", " ").replace("\n", " ")
        sents = [s.strip() for s in SENT_SPLIT_RE.split(part) if s.strip()]
        result.extend(sents)
    return result


# -------------------------------------------------------
# Cleaning helpers
# -------------------------------------------------------

BULLET_RE = re.compile(r"^[\s\-\*\u2022\u00b7•·]+")  # -, *, •, etc.


def clean_sentence(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = text

    s = s.replace("\r", " ").replace("\n", " ").replace("\xa0", " ")
    s = s.replace("*", "").replace("`", "")

    # collapse repeated slashes
    s = re.sub(r"[\\/]{2,}", " ", s)

    # remove weird /- combinations
    s = re.sub(r"\s*[/\\]\s*-\s*[/\\]?\s*", " ", s)

    # remove bullet prefixes
    s = BULLET_RE.sub("", s)

    # collapse multi punctuation
    s = re.sub(r"([!?.,;:])\1+", r"\1", s)

    # clean empty parentheses
    s = re.sub(r"\(\s*\)", " ", s)

    s = re.sub(r"\s{2,}", " ", s)

    return s.strip(" \"'")


# -------------------------------------------------------
# Conservative sentence relevance rescue
# -------------------------------------------------------

REQUIREMENT_VERB_RE = re.compile(
    r"\b(required|requirement|must|should|responsibilit|qualifications?|"
    r"experience|proficien|develop|build|design|implement|maintain|manage|"
    r"deploy|test|debug|troubleshoot|analy[sz]e|integrate|configure|operate)\b",
    re.I,
)
TECH_SIGNAL_RE = re.compile(
    r"\b(python|java\b|javascript|typescript|sql|database|api|rest|backend|"
    r"frontend|react|node|aws|azure|gcp|cloud|docker|kubernetes|terraform|"
    r"linux|git|devops|ci/cd|testing|data|etl|pipeline|machine learning|"
    r"ai\b|nlp|security|salesforce|tableau|power bi)\b",
    re.I,
)
HEADER_ONLY_RE = re.compile(
    r"^\s*(about us|benefits?|compensation|salary|location|schedule|apply now|"
    r"job type|equal opportunity|privacy|company overview)\s*:?\s*$",
    re.I,
)


def score_sentence_for_relevance_rescue(text: str) -> tuple[float, str]:
    """Score dropped sentences when a job would otherwise lose all evidence."""
    s = str(text or "").strip()
    flags = []
    score = 0.0
    length = len(s)
    if 35 <= length <= 420:
        score += 1.0
        flags.append("sane_length")
    elif 15 <= length <= 600:
        score += 0.25
        flags.append("borderline_length")
    else:
        score -= 1.0
        flags.append("poor_length")
    if TECH_SIGNAL_RE.search(s):
        score += 2.0
        flags.append("technical_keyword")
    if REQUIREMENT_VERB_RE.search(s):
        score += 1.5
        flags.append("requirement_or_action_verb")
    if not HEADER_ONLY_RE.match(s):
        score += 0.75
        flags.append("non_header_text")
    else:
        score -= 2.0
        flags.append("header_only")
    if "," in s or ";" in s or " and " in s.lower():
        score += 0.25
        flags.append("contains_detail")
    return round(score, 3), " | ".join(flags)


def apply_min_sentence_coverage(
    pipe_df: pd.DataFrame,
    *,
    min_kept_sentences_per_job: int = 1,
    max_rescued_sentences_per_job: int = 3,
) -> pd.DataFrame:
    """Ensure a prepared job is not made invisible by the relevance filter."""
    df = pipe_df.copy()
    if "is_relevant" not in df.columns:
        df["is_relevant"] = True
    df["is_relevant_initial"] = df["is_relevant"].astype(bool)
    df["is_relevant_final"] = df["is_relevant_initial"]
    df["rescued_by_min_job_coverage"] = False
    df["relevance_decision"] = df["is_relevant_initial"].map(
        {True: "kept_by_relevance_filter", False: "dropped_by_relevance_filter"}
    )
    df["rescue_score"] = 0.0
    df["rescue_reason"] = ""

    min_keep = max(0, int(min_kept_sentences_per_job or 0))
    max_rescue = max(0, int(max_rescued_sentences_per_job or 0))
    if min_keep <= 0 or max_rescue <= 0 or df.empty:
        return df

    for _job_id, group in df.groupby("job_id", sort=False):
        kept = int(group["is_relevant_final"].sum())
        if kept >= min_keep:
            continue
        candidates = group[~group["is_relevant_final"]].copy()
        if candidates.empty:
            continue
        scored = []
        for idx, row in candidates.iterrows():
            score, reason = score_sentence_for_relevance_rescue(row.get("sentence_text", ""))
            scored.append((score, str(row.get("sentence_id", "")), idx, reason))
        scored.sort(key=lambda x: (-x[0], x[1]))
        rescue_needed = min(max_rescue, min_keep - kept, len(scored))
        for score, _sid, idx, reason in scored[:rescue_needed]:
            df.at[idx, "is_relevant_final"] = True
            df.at[idx, "rescued_by_min_job_coverage"] = True
            df.at[idx, "relevance_decision"] = "rescued_by_min_job_coverage"
            df.at[idx, "rescue_score"] = score
            df.at[idx, "rescue_reason"] = reason

    return df


def write_sentence_relevance_audits(df: pd.DataFrame, out_dir: Path) -> None:
    audit_df = df.copy()
    if "is_relevant_initial" not in audit_df.columns:
        audit_df["is_relevant_initial"] = True
    if "is_relevant_final" not in audit_df.columns:
        audit_df["is_relevant_final"] = True
    if "rescued_by_min_job_coverage" not in audit_df.columns:
        audit_df["rescued_by_min_job_coverage"] = False
    if "relevance_decision" not in audit_df.columns:
        audit_df["relevance_decision"] = "kept_without_filter"
    if "rescue_score" not in audit_df.columns:
        audit_df["rescue_score"] = 0.0
    if "rescue_reason" not in audit_df.columns:
        audit_df["rescue_reason"] = ""

    decision_cols = [
        "job_id", "job_date", "sentence_id", "sentence_text",
        "is_relevant_initial", "is_relevant_final",
        "rescued_by_min_job_coverage", "relevance_decision",
        "rescue_score", "rescue_reason",
    ]
    existing_decision_cols = [c for c in decision_cols if c in audit_df.columns]
    audit_df[existing_decision_cols].to_csv(
        out_dir / "sentence_relevance_decisions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    per_job = (
        audit_df.groupby("job_id")
        .agg(
            total=("sentence_text", "size"),
            initially_kept=("is_relevant_initial", "sum"),
            kept=("is_relevant_final", "sum"),
            rescued=("rescued_by_min_job_coverage", "sum"),
        )
        .reset_index()
    )
    per_job["lost_all_sentences_after_filter"] = per_job["kept"] == 0
    per_job.to_csv(out_dir / "sentence_relevance_per_job.csv", index=False, encoding="utf-8-sig")


# -------------------------------------------------------
# Language + Translation
# -------------------------------------------------------

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return "unknown"


class Translator:
    def __init__(self, model="Helsinki-NLP/opus-mt-mul-en"):
        from transformers import pipeline as hf_pipeline
        print(f"[INFO] Loading translation model: {model}", flush=True)
        self.pipe = hf_pipeline("translation", model=model)

    def translate_batch(self, texts: List[str]):
        out = self.pipe(texts, truncation=True, max_length=512)
        return [d["translation_text"] for d in out]


# -------------------------------------------------------
# Main pipeline
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Full preprocessing for raw jobs → pipeline-ready sentences")
    parser.add_argument(
        "--input",
        type=str,
        default=str(_JOBS_SCRAPING_CSV),
        help="Raw scraped job postings CSV (default: job_scraping/output/english_jobs.csv for 12mo data)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(_PREPROCESS_OUTPUT_DIR),
        help="Where to store metadata + cleaned sentence CSVs",
    )
    parser.add_argument("--translate", action="store_true", help="Translate non-English sentences → English")
    parser.add_argument("--dedupe", action="store_true", help="Remove duplicate cleaned sentences")
    parser.add_argument("--min_len", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=600,
        help="Max sentence length in chars (default 600 for JobBERT 128-token limit)")
    parser.add_argument("--translation_model", type=str, default="Helsinki-NLP/opus-mt-mul-en")
    parser.add_argument(
        "--no-relevance-filter",
        action="store_true",
        help="Skip the zero-shot LLM sentence relevance filter (Phase 1.2). "
             "When skipped, jobs_sentences_filtered.csv is a copy of the unfiltered "
             "sentences so downstream stages still find the file they expect.",
    )
    parser.add_argument(
        "--relevance-filter-model",
        type=str,
        default=None,
        help="OpenRouter model id for the relevance filter "
             "(default: sentence_relevance_filter.DEFAULT_MODEL)",
    )
    parser.add_argument(
        "--relevance-filter-batch-size",
        type=int,
        default=None,
        help="Sentences per LLM call for the relevance filter "
             "(default: sentence_relevance_filter.DEFAULT_BATCH_SIZE = 30; "
             "minimum recommended is 30 to amortize prompt overhead).",
    )
    parser.add_argument(
        "--min-kept-sentences-per-job",
        type=int,
        default=1,
        help="Minimum filtered sentences to retain per prepared job when the relevance filter would drop them all.",
    )
    parser.add_argument(
        "--max-rescued-sentences-per-job",
        type=int,
        default=3,
        help="Maximum dropped sentences to rescue per job to satisfy min-kept coverage.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    print(f"[INFO] Reading raw data: {input_path}", flush=True)
    df = pd.read_csv(input_path)

    if "description" not in df.columns:
        raise ValueError("Raw CSV must contain a 'description' column.")

    # -----------------------------
    # Normalize job_id + job_date
    # -----------------------------
    if "id" in df.columns:
        df["job_id"] = df["id"].astype(str)
    else:
        df = df.reset_index()
        df["job_id"] = df["index"].astype(str)

    df["job_date"] = pd.to_datetime(df.get("date_posted", pd.NaT))

    # --------------------------------------------------
    # Item 7: Job-level deduplication (before splitting)
    # --------------------------------------------------
    # Fingerprint each job by (title, company, normalised-description hash).
    # This removes re-posted jobs that inflate demand counts and trend slopes.
    if args.dedupe:
        import hashlib, re as _re
        def _job_fingerprint(row) -> str:
            title = str(row.get("title", "")).strip().lower()
            company = str(row.get("company", "")).strip().lower()
            desc_norm = _re.sub(r"\s+", " ", str(row.get("description", "")).lower().strip())[:500]
            raw = f"{title}||{company}||{desc_norm}"
            return hashlib.md5(raw.encode("utf-8")).hexdigest()

        before_jobs = len(df)
        df["_fp"] = df.apply(_job_fingerprint, axis=1)
        df = df.drop_duplicates(subset=["_fp"]).drop(columns=["_fp"])
        after_jobs = len(df)
        print(f"[INFO] Job-level dedup: {before_jobs} → {after_jobs} jobs (removed {before_jobs - after_jobs} duplicates)")

    # -----------------------------
    # Education-level extraction (KKNI)
    # Adds min_education_kkni and education_labels per job before saving metadata.
    # See education_level_extractor.py and kkni.py.
    # -----------------------------
    try:
        from education_level_extractor import annotate_dataframe as _edu_annotate
        text_cols = [c for c in ("description", "requirements") if c in df.columns]
        if text_cols:
            _edu_annotate(df, text_columns=text_cols)
            n_known = int(df["min_education_kkni"].notna().sum())
            print(f"[INFO] Education-level extraction: {n_known}/{len(df)} jobs tagged with KKNI level")
        else:
            df["min_education_kkni"] = None
            df["education_labels"] = ""
    except ImportError:
        print("[WARN] education_level_extractor not available; skipping KKNI annotation")
        df["min_education_kkni"] = None
        df["education_labels"] = ""

    # -----------------------------
    # Save job metadata
    # -----------------------------
    metadata_cols = [
        c for c in [
            "job_id", "title", "company", "location", "site", "job_url",
            "job_date", "min_education_kkni", "education_labels",
        ] if c in df.columns
    ]
    metadata = df[metadata_cols].drop_duplicates(subset=["job_id"] if "job_id" in metadata_cols else None)
    metadata.to_csv(out_dir / "jobs_metadata.csv", index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved jobs_metadata.csv", flush=True)

    # -----------------------------
    # Sentence splitting
    # -----------------------------
    print("[INFO] Splitting into sentences...")
    rows = []

    for _, row in df.iterrows():
        job_id = row["job_id"]
        job_date = row.get("job_date", pd.NaT)
        text = row["description"]

        sents = split_sentences(text)
        sid = 0
        for sent in sents:
            if len(sent) < args.min_len or len(sent) > args.max_len:
                continue
            sid += 1
            # sentence_id is zero-padded {job_id}_{sid:04d} so it sorts naturally
            # and stays unique across the whole corpus. Used for provenance traces
            # back from extracted skills/knowledge to source sentences.
            rows.append({
                "job_id": job_id,
                "job_date": job_date,
                "sentence_id": f"{job_id}_{sid:04d}",
                "sentence_raw": sent
            })

    sents_df = pd.DataFrame(rows)
    print(f"[INFO] Total split sentences: {len(sents_df)}")

    # -----------------------------
    # Cleaning
    # -----------------------------
    print("[INFO] Cleaning sentences...")
    sents_df["sentence_clean"] = sents_df["sentence_raw"].apply(clean_sentence)

    # -----------------------------
    # Language detection & Translation (optional)
    # -----------------------------
    if args.translate:
        print("[INFO] Detecting language...")
        sents_df["lang"] = sents_df["sentence_clean"].apply(detect_language)

        translator = Translator(args.translation_model)
        sents_df["sentence_en"] = sents_df["sentence_clean"]

        mask = (sents_df["lang"] != "en") & (sents_df["lang"] != "unknown")
        idxs = sents_df.index[mask]

        print(f"[INFO] Translating {len(idxs)} non-English sentences...")
        batch = 64
        for i in range(0, len(idxs), batch):
            b = idxs[i:i+batch]
            text_batch = sents_df.loc[b, "sentence_clean"].tolist()
            translated = translator.translate_batch(text_batch)
            sents_df.loc[b, "sentence_en"] = translated
    else:
        print("[INFO] Translation disabled. Using cleaned text only.")
        sents_df["sentence_en"] = sents_df["sentence_clean"]

    # -----------------------------
    # Dedupe (optional)
    # -----------------------------
    if args.dedupe:
        before = len(sents_df)
        sents_df = sents_df.drop_duplicates(subset=["sentence_en"])
        after = len(sents_df)
        print(f"[INFO] Removed duplicates: {before - after}")

    # -----------------------------
    # Save full + pipeline-ready
    # -----------------------------
    full_path = out_dir / "job_sentences_full.csv"
    sents_df.to_csv(full_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved: {full_path}")

    # Pipeline expects: job_id, sentence_id, sentence_text (and optionally job_date).
    # sentence_id is the provenance handle that downstream stages thread back to
    # advanced_skills.csv / advanced_knowledge.csv per pipeline-redesign-v2 Phase 1.1.
    pipe_df = sents_df[["job_id", "job_date", "sentence_id", "sentence_en"]].rename(
        columns={"sentence_en": "sentence_text"}
    )
    pipe_path = out_dir / "job_sentences_for_pipeline.csv"
    pipe_df.to_csv(pipe_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved pipeline-ready file: {pipe_path}")

    # Audit copy: jobs_sentences.csv = full unfiltered sentence set (preserved
    # for inspection / re-runs with a different filter). Pipeline input by
    # default is jobs_sentences_filtered.csv (next block).
    jobs_sent_path = out_dir / "jobs_sentences.csv"
    pipe_df.to_csv(jobs_sent_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved audit copy: {jobs_sent_path}")

    # -----------------------------
    # Sentence relevance filter (pipeline-redesign-v2 Phase 1.2, Req 4)
    # -----------------------------
    filtered_path = out_dir / "jobs_sentences_filtered.csv"
    if args.no_relevance_filter:
        print(
            "[INFO] --no-relevance-filter set; jobs_sentences_filtered.csv "
            "will be a copy of the unfiltered sentences."
        )
        pipe_df.to_csv(filtered_path, index=False, encoding="utf-8-sig")
        audit_df = pipe_df.copy()
        audit_df["is_relevant_initial"] = True
        audit_df["is_relevant_final"] = True
        audit_df["rescued_by_min_job_coverage"] = False
        audit_df["relevance_decision"] = "kept_filter_skipped"
        write_sentence_relevance_audits(audit_df, out_dir)
        print(f"[INFO] Saved: {filtered_path} (filter skipped — same as audit)")
    else:
        try:
            from sentence_relevance_filter import (
                classify_sentences,
                DEFAULT_BATCH_SIZE,
                DEFAULT_MODEL,
            )
        except ImportError as exc:
            print(
                f"[WARN] sentence_relevance_filter unavailable ({exc}); "
                f"skipping filter and copying unfiltered sentences to {filtered_path}."
            )
            pipe_df.to_csv(filtered_path, index=False, encoding="utf-8-sig")
            audit_df = pipe_df.copy()
            audit_df["is_relevant_initial"] = True
            audit_df["is_relevant_final"] = True
            audit_df["rescued_by_min_job_coverage"] = False
            audit_df["relevance_decision"] = "kept_filter_unavailable"
            write_sentence_relevance_audits(audit_df, out_dir)
        else:
            cache_path = out_dir / "sentence_relevance_cache.json"
            sentences_for_filter = pipe_df["sentence_text"].astype(str).tolist()
            print(
                f"[INFO] Running zero-shot relevance filter on "
                f"{len(sentences_for_filter)} sentences "
                f"(cache: {cache_path.name}; YES = relevant, NO = drop)..."
            )
            try:
                verdicts = classify_sentences(
                    sentences_for_filter,
                    cache_path=cache_path,
                    model=args.relevance_filter_model or DEFAULT_MODEL,
                    batch_size=args.relevance_filter_batch_size or DEFAULT_BATCH_SIZE,
                )
            except Exception as exc:
                print(
                    f"[WARN] Relevance filter failed ({exc}); falling back to "
                    f"unfiltered sentences for {filtered_path}."
                )
                pipe_df.to_csv(filtered_path, index=False, encoding="utf-8-sig")
                audit_df = pipe_df.copy()
                audit_df["is_relevant_initial"] = True
                audit_df["is_relevant_final"] = True
                audit_df["rescued_by_min_job_coverage"] = False
                audit_df["relevance_decision"] = "kept_filter_failed"
                write_sentence_relevance_audits(audit_df, out_dir)
            else:
                pipe_df = pipe_df.copy()
                pipe_df["is_relevant"] = verdicts
                pipe_df = apply_min_sentence_coverage(
                    pipe_df,
                    min_kept_sentences_per_job=args.min_kept_sentences_per_job,
                    max_rescued_sentences_per_job=args.max_rescued_sentences_per_job,
                )
                kept_df = pipe_df[pipe_df["is_relevant_final"]][
                    ["job_id", "job_date", "sentence_id", "sentence_text"]
                ]

                # Per-job kept / total log (Req 4.5)
                per_job = (
                    pipe_df.groupby("job_id")
                    .agg(
                        total=("sentence_text", "size"),
                        initially_kept=("is_relevant_initial", "sum"),
                        kept=("is_relevant_final", "sum"),
                        rescued=("rescued_by_min_job_coverage", "sum"),
                    )
                    .reset_index()
                )
                total_total = int(per_job["total"].sum())
                total_kept = int(per_job["kept"].sum())
                total_rescued = int(per_job["rescued"].sum())
                pct = (total_kept / total_total * 100.0) if total_total else 0.0
                print(
                    f"[INFO] Relevance filter kept {total_kept}/{total_total} "
                    f"sentences ({pct:.1f}%) across {len(per_job)} jobs "
                    f"({total_rescued} rescued by min-job coverage)."
                )

                kept_df.to_csv(filtered_path, index=False, encoding="utf-8-sig")
                print(f"[INFO] Saved: {filtered_path} (pipeline input — filtered)")

                write_sentence_relevance_audits(pipe_df, out_dir)
                print(f"[INFO] Saved relevance audit files in: {out_dir}")

    print("[INFO] Preprocessing completed successfully.", flush=True)


if __name__ == "__main__":
    main()
