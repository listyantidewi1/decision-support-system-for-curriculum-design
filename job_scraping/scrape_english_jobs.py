"""
English-only job scraper with post-processing.
Single run: scrapes jobs, filters by English, deduplicates, extracts sentences for NLP.
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
# 1. CONFIGURATION
# ==============================

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
JOBS_CSV = OUTPUT_DIR / "english_jobs.csv"
SENTENCES_CSV = OUTPUT_DIR / "english_job_sentences.csv"

# Software engineering and game development roles (comprehensive)
JOB_ROLES: List[str] = [
    # --- Software engineering (general) ---
    "Software engineer",
    "Software Developer",
    "Software developer",
    "Programmer",
    "Junior software engineer",
    "Junior developer",
    "Senior software engineer",
    "Staff software engineer",
    "Principal software engineer",
    "Application Developer",
    "Systems Developer",
    "Systems engineer",
    "System engineer",
    # --- Web & full stack ---
    "Web Developer",
    "Full Stack Developer",
    "Full stack engineer",
    "Frontend Engineer",
    "Frontend developer",
    "Front end developer",
    "Backend Engineer",
    "Backend developer",
    "Back end developer",
    "Frontend Web Developer",
    "Backend Web Developer",
    # --- Mobile ---
    "Mobile Developer",
    "Mobile applications developer",
    "iOS Developer",
    "Android Developer",
    # --- Language & platform specific ---
    "Java developer",
    "Python developer",
    "JavaScript developer",
    "C++ developer",
    ".NET developer",
    "Go developer",
    "Rust developer",
    "React developer",
    "Node.js developer",
    # --- Cloud, DevOps, SRE ---
    "Cloud architect",
    "Cloud Developer",
    "DevOps engineer",
    "Site reliability engineer",
    "Platform engineer",
    "Infrastructure engineer",
    # --- Data & ML ---
    "Data scientist",
    "Data engineer",
    "Data analyst",
    "Data Architect",
    "Data manager",
    "Data governance manager",
    "Data Modeler",
    "Data Visualization Specialist",
    "Data Warehouse Developer",
    "ETL Developer",
    "BI Developer",
    "BI Analyst",
    "Machine learning engineer",
    "Machine Learning Scientist",
    "Machine Learning Developer",
    "AI Engineer",
    "AI Developer",
    "AI ethics Specialist",
    # --- Database, security, network ---
    "Database administrator",
    "Database developer",
    "Database engineer",
    "Cyber security specialist",
    "Security Analyst",
    "Security engineer",
    "Network engineer",
    # --- Other engineering ---
    "BlockChain developer",
    "Big data architecture",
    "Business analyst",
    "Embedded software engineer",
    "Firmware engineer",
    "QA engineer",
    "Test engineer",
    "Automation engineer",
    "Solutions architect",
    "Software architect",
    "UX designer",
    # --- Game development ---
    "Game Developer",
    "Game developer",
    "Game Programmer",
    "Game programmer",
    "Gameplay Programmer",
    "Graphics Programmer",
    "Animation Programmer",
    "Game Designer",
    "Game designer",
    "Level Designer",
    "Narrative Designer",
    "Systems designer",
    "Game Artist",
    "Technical Artist",
    "Concept Artist",
    "Character Artist",
    "Environment Artist",
    "3D Modeler",
    "3D Artist",
    "VFX Artist",
    "UI Artist",
    "UI Designer",
    "Game Tester",
    "QA Tester",
    "Sound Designer",
    "Sound Engineer",
    "Music Composer",
    "Game Producer",
    "Tools programmer",
    "Engine programmer",
    "Physics programmer",
    "Shader programmer",
    "Combat designer",
    "Gameplay designer",
    "Rigging artist",
    "Lighting artist",
    "Game animator",
    "Live ops",
    # --- XR ---
    "AR Developer",
    "VR Developer",
    "XR Developer",
]

LOCATIONS: List[Dict[str, str]] = [
    {"label": "United States", "location": "United States", "country_indeed": "USA"},
    {"label": "United Kingdom", "location": "United Kingdom", "country_indeed": "UK"},
    {"label": "Canada", "location": "Canada", "country_indeed": "Canada"},
    {"label": "Australia", "location": "Australia", "country_indeed": "Australia"},
    {"label": "New Zealand", "location": "New Zealand", "country_indeed": "New Zealand"},
    {"label": "Ireland", "location": "Ireland", "country_indeed": "Ireland"},
    {"label": "Singapore", "location": "Singapore", "country_indeed": "Singapore"},
]

SITE_NAMES = ["indeed", "linkedin", "zip_recruiter", "google"]
RESULTS_WANTED_PER_QUERY = 5000
HOURS_OLD = 24 * 365  # last 12 months
SLEEP_BETWEEN_CALLS = 5

ENGLISH_COMMON_WORDS = {
    "the", "and", "for", "with", "you", "your", "will", "have", "from",
    "this", "that", "are", "as", "on", "in", "to", "of", "we", "our",
    "experience", "skills", "ability", "knowledge", "responsibilities",
    "requirements", "engineer", "developer", "team", "job", "role",
    "work", "software", "design",
}


def clean_text(text) -> str:
    """Remove noise: emojis, control chars, HTML, repeated punctuation, excess whitespace."""
    if not isinstance(text, str) or (isinstance(text, float) and pd.isna(text)):
        return ""
    t = str(text)
    # Control chars and non-printables
    t = re.sub(r"[\x00-\x1f\x7f]", " ", t)
    t = re.sub(r"[\u200b-\u200d\ufeff]", "", t)
    # Emojis (supplemental planes + common BMP emoji ranges)
    t = re.sub(r"[\U00010000-\U0010ffff]", "", t)
    t = re.sub(r"[\U0001f300-\U0001f9ff\U00002600-\U000027bf]", "", t)
    # HTML entities and tags
    t = html.unescape(t)
    t = re.sub(r"<[^>]+>", " ", t)
    # Collapse repeated punctuation (3+ -> 1), keep C++
    t = re.sub(r"([^\w\s])\1{2,}", r"\1", t)
    # Normalize whitespace
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def clean_job_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clean_text to title, company, location, description."""
    df = df.copy()
    for col in ["title", "company", "location", "description"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).apply(clean_text)
    return df


def is_likely_english(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip()
    if not t:
        return False
    ascii_chars = sum(1 for c in t if ord(c) < 128)
    if len(t) == 0:
        return False
    ascii_ratio = ascii_chars / len(t)
    lower = t.lower()
    hits = sum(1 for w in ENGLISH_COMMON_WORDS if w in lower)
    return ascii_ratio > 0.8 and hits >= 3


def fetch_jobs_for_role_and_location(role: str, loc_cfg: Dict[str, str]) -> pd.DataFrame:
    location = loc_cfg["location"]
    country_indeed = loc_cfg["country_indeed"]
    google_search_term = f"{role} jobs in {location}"
    print(f"\n=== Scraping: role='{role}' | location='{location}' ===")
    try:
        jobs_df = scrape_jobs(
            site_name=SITE_NAMES,
            search_term=role,
            google_search_term=google_search_term,
            location=location,
            results_wanted=RESULTS_WANTED_PER_QUERY,
            hours_old=HOURS_OLD,
            country_indeed=country_indeed,
        )
    except Exception as e:
        print(f"[ERROR] Failed for role='{role}', location='{location}': {e}")
        return pd.DataFrame()
    if jobs_df.empty:
        print(f"[INFO] No jobs returned for role='{role}', location='{location}'")
        return jobs_df
    jobs_df["query_role"] = role
    jobs_df["query_location_label"] = loc_cfg["label"]
    jobs_df["query_location_string"] = location
    print(f"[OK] Retrieved {len(jobs_df)} jobs for this (role, location)")
    return jobs_df


def extract_sentences(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.drop_duplicates(subset=["description"], keep="first").dropna(subset=["description"])
    print(f"Unique descriptions for sentence extraction: {len(df_clean)}")

    def clean_and_split(text):
        if not isinstance(text, str):
            return []
        text = re.sub(r"[\U00010000-\U0010ffff]", "", text)
        text = re.sub(r"([^\w\s])\1{2,}", r"\1", text)
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 2]

    all_sentences = []
    for desc in df_clean["description"]:
        all_sentences.extend(clean_and_split(desc))
    return pd.DataFrame(all_sentences, columns=["sentence"])


def main():
    all_jobs: List[pd.DataFrame] = []
    for role in JOB_ROLES:
        for loc_cfg in LOCATIONS:
            df = fetch_jobs_for_role_and_location(role, loc_cfg)
            if not df.empty:
                all_jobs.append(df)
            time.sleep(SLEEP_BETWEEN_CALLS)

    if not all_jobs:
        print("No jobs scraped at all.")
        return

    combined = pd.concat(all_jobs, ignore_index=True)
    dedupe_cols = [c for c in ["job_url", "title", "company", "location"] if c in combined.columns]
    if dedupe_cols:
        before = len(combined)
        combined = combined.drop_duplicates(subset=dedupe_cols, keep="first")
        print(f"Deduplicated from {before} to {len(combined)} rows")

    print("Cleaning text (removing emojis, HTML, excess whitespace)...")
    combined = clean_job_data(combined)

    title_col = combined["title"].fillna("").astype(str) if "title" in combined.columns else ""
    desc_col = combined["description"].fillna("").astype(str) if "description" in combined.columns else ""
    combined["is_english"] = (title_col + " " + desc_col).apply(is_likely_english)
    before_lang = len(combined)
    combined = combined[combined["is_english"]].reset_index(drop=True)
    print(f"Filtered by English: {before_lang} -> {len(combined)} rows")

    # All JobSpy attributes + our metadata; keep any extra columns JobSpy adds
    OUTPUT_COLS = [
        "id", "site", "job_url", "job_url_direct", "title", "company", "location",
        "date_posted", "job_type", "salary_source", "interval", "min_amount", "max_amount",
        "currency", "is_remote", "job_level", "job_function", "listing_type", "emails",
        "description", "company_industry", "company_url", "company_logo", "company_url_direct",
        "company_addresses", "company_num_employees", "company_revenue", "company_description",
        "skills", "experience_range", "company_rating", "company_reviews_count", "vacancy_count",
        "work_from_home_type",
        "query_role", "query_location_label", "query_location_string", "is_english",
    ]
    existing_cols = [c for c in OUTPUT_COLS if c in combined.columns]
    other_cols = [c for c in combined.columns if c not in existing_cols]
    combined = combined[existing_cols + other_cols]

    combined.to_csv(
        JOBS_CSV,
        quoting=csv.QUOTE_NONNUMERIC,
        escapechar="\\",
        index=False,
        encoding="utf-8",
    )
    print(f"\n[1/2] Saved {len(combined)} jobs -> {JOBS_CSV.resolve()}")

    print("\n[2/2] Extracting sentences for NLP...")
    sentences_df = extract_sentences(combined)
    sentences_df.to_csv(SENTENCES_CSV, index=False, encoding="utf-8")
    print(f"Saved {len(sentences_df)} sentences -> {SENTENCES_CSV.resolve()}")
    print("\nDone.")


if __name__ == "__main__":
    main()
