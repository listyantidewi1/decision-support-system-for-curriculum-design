"""
Indonesian software engineering & game development job scraper for SMK (vocational school) graduates.
Single run: scrapes jobs from Indonesia, filters for SMK-only suitability, saves raw + SMK-filtered outputs.
"""
import csv
import html
import re
import time
from pathlib import Path
from typing import List, Dict

import pandas as pd
from jobspy import scrape_jobs

# ==============================
# 1. CONFIG
# ==============================

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
RAW_CSV = OUTPUT_DIR / "indonesian_jobs_raw.csv"
SMK_FILTERED_CSV = OUTPUT_DIR / "indonesian_jobs_smk.csv"

# Software engineering and game development job terms (aligned with English scraper)
# Indonesian + English terms for Indonesia job boards
JOB_TERMS: List[Dict[str, str]] = [
    # --- Software engineering (general) ---
    {"family": "software_developer", "term": "programmer"},
    {"family": "software_developer", "term": "junior programmer"},
    {"family": "software_developer", "term": "software developer"},
    {"family": "software_developer", "term": "software engineer"},
    {"family": "software_developer", "term": "developer aplikasi"},
    {"family": "software_developer", "term": "pengembang perangkat lunak"},
    # --- Web ---
    {"family": "web_developer", "term": "web developer"},
    {"family": "web_developer", "term": "front end developer"},
    {"family": "web_developer", "term": "backend developer"},
    {"family": "web_developer", "term": "full stack developer"},
    {"family": "web_developer", "term": "programmer web"},
    # --- Mobile ---
    {"family": "mobile_developer", "term": "mobile developer"},
    {"family": "mobile_developer", "term": "android developer"},
    {"family": "mobile_developer", "term": "flutter developer"},
    {"family": "mobile_developer", "term": "pengembang aplikasi mobile"},
    # --- Language-specific ---
    {"family": "software_developer", "term": "java developer"},
    {"family": "software_developer", "term": "python developer"},
    {"family": "software_developer", "term": "javascript developer"},
    # --- Game development ---
    {"family": "game_developer", "term": "game developer"},
    {"family": "game_developer", "term": "game programmer"},
    {"family": "game_developer", "term": "pengembang gim"},
    {"family": "game_developer", "term": "game artist"},
    {"family": "game_developer", "term": "game designer"},
    {"family": "game_developer", "term": "game tester"},
    {"family": "game_developer", "term": "level designer"},
    {"family": "game_developer", "term": "3d artist"},
    {"family": "game_developer", "term": "technical artist"},
    {"family": "game_developer", "term": "concept artist"},
    {"family": "game_developer", "term": "game animator"},
    {"family": "game_developer", "term": "sound designer"},
    {"family": "game_developer", "term": "game producer"},
    # --- UI/UX ---
    {"family": "ui_ux", "term": "ui ux designer"},
    {"family": "ui_ux", "term": "ui designer"},
    {"family": "ui_ux", "term": "ux designer"},
    {"family": "ui_ux", "term": "desainer ui ux"},
    # --- QA & testing ---
    {"family": "qa_tester", "term": "quality assurance"},
    {"family": "qa_tester", "term": "qa engineer"},
    {"family": "qa_tester", "term": "software tester"},
    {"family": "qa_tester", "term": "game tester"},
    # --- Database ---
    {"family": "database", "term": "database administrator"},
    {"family": "database", "term": "database engineer"},
    {"family": "database", "term": "basis data"},
    # --- DevOps ---
    {"family": "devops", "term": "devops engineer"},
]

SITE_NAMES = ["indeed", "linkedin", "google"]
LOCATION = "Indonesia"
COUNTRY_INDEED = "Indonesia"
RESULTS_WANTED_PER_TERM = 5000
HOURS_OLD = 24 * 365  # last 12 months
SLEEP_BETWEEN_CALLS = 15

# SMK only: vocational school (Sekolah Menengah Kejuruan) - jobs must mention SMK
# Require at least one SMK_SPECIFIC hit; supplementary terms (fresh grad, entry level) alone do not qualify
SMK_SPECIFIC_KEYWORDS = [
    "lulusan smk", "minimal smk", "pendidikan minimal smk", "pendidikan smk",
    "smk it", "smk rpl", "smk rekayasa perangkat lunak", "smk jurusan komputer",
    "smk jurusan it", "smk komputer", "smk sederajat", "smk atau setara",
    "smk/setara", "smk atau sederajat", "smk/sederajat", "smk/d3", "smk / d3",
    "rekayasa perangkat lunak", "smk tkj", "teknik komputer dan jaringan",
    "sekolah menengah kejuruan", "stm",
    "lulusan smk/sma", "minimal smk/sma", "smk/sma", "sma/smk",
]
SMK_INCLUDE_KEYWORDS = SMK_SPECIFIC_KEYWORDS + [
    "fresh graduate", "fresh-graduate", "freshgraduate", "entry level",
]

EXCLUDE_KEYWORDS_STRICT = [
    "minimal s1", "pendidikan minimal s1", "wajib s1", "harus s1", "sarjana s1",
    "gelar s1", "bachelor degree", "sarjana komputer", "s2", "magister",
    "desk collection", "electrical wiring", "plumbing", "nasabah kpr",
    "penagihan", "desk collector", "debt collector",
]


def clean_text(text) -> str:
    """Remove noise: emojis, control chars, HTML, repeated punctuation, excess whitespace."""
    if not isinstance(text, str) or (isinstance(text, float) and pd.isna(text)):
        return ""
    t = str(text)
    t = re.sub(r"[\x00-\x1f\x7f]", " ", t)
    t = re.sub(r"[\u200b-\u200d\ufeff]", "", t)
    t = re.sub(r"[\U00010000-\U0010ffff]", "", t)
    t = re.sub(r"[\U0001f300-\U0001f9ff\U00002600-\U000027bf]", "", t)
    t = html.unescape(t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"([^\w\s])\1{2,}", r"\1", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def clean_job_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clean_text to title, company, location, description."""
    df = df.copy()
    for col in ["title", "company", "location", "description"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).apply(clean_text)
    return df


# Must match at least one of these for "SMK/vocational graduate only"
SMK_SPECIFIC_KEYWORDS = [
    "lulusan smk", "minimal smk", "pendidikan smk", "smk it", "smk rpl",
    "smk rekayasa perangkat lunak", "smk jurusan komputer", "smk jurusan it",
    "smk komputer", "smk/d3", "smk / d3", "rekayasa perangkat lunak",
    "smk tkj", "teknik komputer dan jaringan", "sekolah menengah kejuruan",
    "stm", "lulusan smk/sma", "minimal smk/sma", "smk/sma", "sma/smk",
]


def annotate_smk_relevance(df: pd.DataFrame) -> pd.DataFrame:
    def _proc(row):
        title = (row.get("title") or "").lower()
        desc = (row.get("description") or "").lower()
        text = f"{title} {desc}"
        include_hits = [kw for kw in SMK_INCLUDE_KEYWORDS if kw in text]
        exclude_hits = [kw for kw in EXCLUDE_KEYWORDS_STRICT if kw in text]
        smk_specific_hits = [kw for kw in SMK_SPECIFIC_KEYWORDS if kw in text]
        # Require at least one SMK-specific keyword (vocational school) + no S1+ exclude
        is_smk = bool(smk_specific_hits) and not bool(exclude_hits)
        row["smk_include_hits"] = "; ".join(include_hits)
        row["smk_exclude_hits"] = "; ".join(exclude_hits)
        row["is_smk_level"] = is_smk
        return row

    return df.apply(_proc, axis=1)


def scrape_rpl_jobs_id() -> pd.DataFrame:
    all_jobs = []
    for cfg in JOB_TERMS:
        term = cfg["term"]
        family = cfg["family"]
        print(f"\n=== Scraping Indonesia | term='{term}' | family='{family}' ===")
        google_search_term = f"{term} lulusan SMK SMA lowongan pekerjaan di Indonesia"
        try:
            jobs_df = scrape_jobs(
                site_name=SITE_NAMES,
                search_term=term,
                google_search_term=google_search_term,
                location=LOCATION,
                country_indeed=COUNTRY_INDEED,
                results_wanted=RESULTS_WANTED_PER_TERM,
                hours_old=HOURS_OLD,
            )
        except Exception as e:
            print(f"[ERROR] term='{term}': {e}")
            time.sleep(SLEEP_BETWEEN_CALLS)
            continue

        if jobs_df.empty:
            print(f"[INFO] No jobs for term='{term}'")
        else:
            jobs_df["query_term"] = term
            jobs_df["query_family"] = family
            print(f"[OK] Retrieved {len(jobs_df)} jobs for term='{term}'")
            all_jobs.append(jobs_df)

        time.sleep(SLEEP_BETWEEN_CALLS)

    if not all_jobs:
        print("No jobs scraped at all.")
        return pd.DataFrame()

    combined = pd.concat(all_jobs, ignore_index=True)
    dedupe_cols = [c for c in ["job_url", "title", "company", "location"] if c in combined.columns]
    if dedupe_cols:
        before = len(combined)
        combined = combined.drop_duplicates(subset=dedupe_cols, keep="first")
        print(f"Deduplicated from {before} to {len(combined)} rows")
    return combined


def main():
    df = scrape_rpl_jobs_id()
    if df.empty:
        return

    print("Cleaning text (removing emojis, HTML, excess whitespace)...")
    df = clean_job_data(df)

    # All JobSpy attributes + our metadata; keep any extra columns JobSpy adds
    OUTPUT_COLS = [
        "id", "site", "job_url", "job_url_direct", "title", "company", "location",
        "date_posted", "job_type", "salary_source", "interval", "min_amount", "max_amount",
        "currency", "is_remote", "job_level", "job_function", "listing_type", "emails",
        "description", "company_industry", "company_url", "company_logo", "company_url_direct",
        "company_addresses", "company_num_employees", "company_revenue", "company_description",
        "skills", "experience_range", "company_rating", "company_reviews_count", "vacancy_count",
        "work_from_home_type",
        "query_term", "query_family",
    ]
    existing_cols = [c for c in OUTPUT_COLS if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + other_cols]

    df.to_csv(
        RAW_CSV,
        quoting=csv.QUOTE_NONNUMERIC,
        escapechar="\\",
        index=False,
        encoding="utf-8",
    )
    print(f"\n[1/2] Saved RAW dataset: {len(df)} rows -> {RAW_CSV.resolve()}")

    df = annotate_smk_relevance(df)
    smk_df = df[df["is_smk_level"]].reset_index(drop=True)
    # SMK output: same cols as raw + smk_include_hits, smk_exclude_hits, is_smk_level
    SMK_OUTPUT_COLS = OUTPUT_COLS + ["smk_include_hits", "smk_exclude_hits", "is_smk_level"]
    existing_smk = [c for c in SMK_OUTPUT_COLS if c in smk_df.columns]
    other_smk = [c for c in smk_df.columns if c not in existing_smk]
    smk_df = smk_df[existing_smk + other_smk]
    smk_df.to_csv(
        SMK_FILTERED_CSV,
        quoting=csv.QUOTE_NONNUMERIC,
        escapechar="\\",
        index=False,
        encoding="utf-8",
    )
    print(f"[2/2] Saved SMK-filtered dataset: {len(smk_df)} rows -> {SMK_FILTERED_CSV.resolve()}")
    print("\nDone.")


if __name__ == "__main__":
    main()
