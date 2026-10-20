"""Ingest the March-2026 legacy scrapes as immutable batch-001 (plan A3).

Provenance is reconstructed post-hoc and the known deviations are recorded in the
manifest: the legacy scripts cleaned text BEFORE saving (violates raw-immutable)
and `date_posted` contains garbage values (1983/1990) that the governance layer
must reject with the date-sanity rule.

Run:  .venv-ml\\Scripts\\python.exe -m acquisition.ingest_legacy
"""
from __future__ import annotations

import sys

import pandas as pd

from .batch_store import REPO_ROOT, write_batch
from .schema import add_provenance

BATCH_ID = "batch-001"
# File mtimes: 2026-03-12 (english), 2026-03-12 (indonesian). Exact scrape start unknown.
ACQ_TS = "2026-03-12T00:00:00Z"

ENGLISH_CSV = REPO_ROOT / "job_scraping" / "output" / "english_jobs.csv"
INDONESIAN_CSV = REPO_ROOT / "job_scraping" / "output" / "indonesian_jobs_raw.csv"

COUNTRY_ISO = {
    "United States": "US", "United Kingdom": "GB", "Canada": "CA", "Australia": "AU",
    "New Zealand": "NZ", "Ireland": "IE", "Singapore": "SG",
}
# Singapore belongs to the ASEAN stratum; the other six are OECD.
STRATUM = {iso: ("asean" if iso == "SG" else "oecd") for iso in COUNTRY_ISO.values()}


def _read_csv(path):
    try:
        return pd.read_csv(path, dtype=str)
    except pd.errors.ParserError:
        return pd.read_csv(path, dtype=str, engine="python", escapechar="\\")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"Reading {ENGLISH_CSV.name} ...", flush=True)
    eng = _read_csv(ENGLISH_CSV)
    print(f"  {len(eng):,} rows", flush=True)
    eng_iso = eng["query_location_label"].map(COUNTRY_ISO).fillna("??")
    eng = add_provenance(
        eng,
        batch_id=BATCH_ID,
        stratum=eng_iso.map(STRATUM).fillna("unknown"),
        country_iso=eng_iso,
        acquisition_timestamp=ACQ_TS,
        acquisition_method=(
            "python-jobspy legacy script scrape_english_jobs.py "
            "(sites=indeed,linkedin,zip_recruiter,google; hours_old=8760; "
            "text pre-cleaned before save)"
        ),
        query_term=eng["query_role"],
        query_family="legacy_english_roles",
    )

    print(f"Reading {INDONESIAN_CSV.name} ...", flush=True)
    idn = _read_csv(INDONESIAN_CSV)
    print(f"  {len(idn):,} rows", flush=True)
    idn = add_provenance(
        idn,
        batch_id=BATCH_ID,
        stratum="indonesia",
        country_iso="ID",
        acquisition_timestamp=ACQ_TS,
        acquisition_method=(
            "python-jobspy legacy script scrape_indonesian_jobs.py "
            "(sites=indeed,linkedin,google; hours_old=8760; "
            "text pre-cleaned before save)"
        ),
        query_term=idn["query_term"],
        query_family=idn["query_family"],
    )

    combined = pd.concat([eng, idn], ignore_index=True)
    batch_dir = write_batch(
        combined,
        BATCH_ID,
        manifest_extra={
            "query_plan_version": "qp-v1-2026-03 (legacy, reconstructed)",
            "provenance_note": (
                "Reconstructed post-hoc on 2026-10-20 from job_scraping/output/*.csv "
                "(scraped ~2026-03-12). DEVIATIONS from the raw-immutable contract: "
                "(1) text fields were cleaned by the legacy scripts before saving, so "
                "description is NOT byte-raw for this batch only; (2) exact per-query "
                "acquisition timestamps were not recorded; (3) date_posted contains "
                "invalid values (e.g. 1983/1990) kept as-is at this layer — the "
                "governance date-sanity rule must exclude them from N_valid_metadata."
            ),
            "source_files": [str(ENGLISH_CSV), str(INDONESIAN_CSV)],
        },
    )
    print(f"\nWrote {len(combined):,} rows -> {batch_dir}", flush=True)
    by = combined.groupby(["stratum", "country_iso"]).size().sort_values(ascending=False)
    print(by.to_string(), flush=True)


if __name__ == "__main__":
    main()
