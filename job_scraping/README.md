# Job Scraping

Unified job scrapers for English and Indonesian IT jobs, with built-in post-processing.

**Default pipeline data source:** `output/english_jobs.csv` is the default input for the main pipeline (12 months of job postings; config.JOBS_SCRAPING_CSV).

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**English jobs** (US, UK, Canada, Australia, etc.) — scrapes, filters by English, deduplicates, extracts sentences for NLP:

```bash
python scrape_english_jobs.py
```

Outputs in `output/`:
- `english_jobs.csv` — filtered job listings (use this for pipeline; has job_id, description, date_posted)
- `english_job_sentences.csv` — flat sentence list (no job_id; not for pipeline)

## Integration with Main Pipeline

After scraping:

```bash
# From project root: one-step preprocess + pipeline
python run_with_job_scraping.py
```

Or manually:
```bash
python preprocess_jobs_pipeline.py --input job_scraping/output/english_jobs.csv
python pipeline.py --input_csv DATA/preprocessing/data_prepared/jobs_sentences.csv
```

Preprocess expects `description`, `id` (or index), `date_posted` columns (JobSpy format).

---

**Indonesian jobs** — RPL/SMK-level IT jobs from Indonesia:

```bash
python scrape_indonesian_jobs.py
```

Outputs in `output/`:
- `indonesian_jobs_raw.csv` — all scraped jobs
- `indonesian_jobs_smk.csv` — SMK-relevant jobs only

## Notes

- Use `english_jobs.csv` (not `english_job_sentences.csv`) for the pipeline — it has job_id and date_posted for per-job extraction and time-series analysis.

## Config

Both scrapers use:
- **Time range:** last 12 months
- **Results:** high limits (500/query English, 2000/term Indonesian)

Edit the top of each script to change `RESULTS_WANTED_*`, `HOURS_OLD`, or `SLEEP_BETWEEN_CALLS`.
