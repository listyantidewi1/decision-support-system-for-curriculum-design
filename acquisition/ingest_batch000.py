"""Ingest the Nov-Dec 2025 legacy scrapes (found in DATA/preprocessing) as batch-000.

Oldest known snapshot — predates batch-001 (March 2026) by ~3 months and extends the
temporal axis backwards. Provenance reconstructed post-hoc; deviations in the manifest.
Contains LinkedIn and Glassdoor rows collected before the SOURCE_AUDIT K1 decision;
whether they enter N_eligible is a governance/promotor decision, not an ingest one.

Run:  .venv-ml\\Scripts\\python.exe -m acquisition.ingest_batch000
"""
from __future__ import annotations

import sys

import pandas as pd

from .batch_store import REPO_ROOT, write_batch
from .schema import add_provenance

BATCH_ID = "batch-000"
BASE = REPO_ROOT / "DATA" / "preprocessing"

COUNTRY_ISO = {
    "United States": "US", "United Kingdom": "GB", "Canada": "CA", "Australia": "AU",
    "New Zealand": "NZ", "Ireland": "IE", "Singapore": "SG",
    "Germany": "DE", "France": "FR", "Netherlands": "NL", "Belgium": "BE",
    "Scotland": "GB",  # part of the UK
}


def _stratum(iso: pd.Series) -> pd.Series:
    return iso.map(lambda c: "asean" if c == "SG" else "oecd")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parts = []

    idn = pd.read_csv(BASE / "jobs_rpl_id_raw_2.csv", dtype=str)
    print(f"jobs_rpl_id_raw_2.csv: {len(idn):,} rows", flush=True)
    parts.append(add_provenance(
        idn,
        batch_id=BATCH_ID,
        stratum="indonesia",
        country_iso="ID",
        acquisition_timestamp="2025-12-05T00:00:00Z",
        acquisition_method=(
            "python-jobspy legacy scrape, Dec 2025 "
            "(file jobs_rpl_id_raw_2.csv; sites=indeed,linkedin; "
            "cleaning status unknown)"
        ),
        query_term=idn["query_term"],
        query_family=idn["query_family"],
    ))

    eng = pd.read_csv(BASE / "jobs_english_only_dataset.csv", dtype=str)
    print(f"jobs_english_only_dataset.csv: {len(eng):,} rows", flush=True)
    eng_iso = eng["query_location_label"].map(COUNTRY_ISO).fillna("??")
    parts.append(add_provenance(
        eng,
        batch_id=BATCH_ID,
        stratum=_stratum(eng_iso),
        country_iso=eng_iso,
        acquisition_timestamp="2025-12-02T00:00:00Z",
        acquisition_method=(
            "python-jobspy legacy scrape, Dec 2025 "
            "(file jobs_english_only_dataset.csv; sites=indeed,linkedin; "
            "English-filtered; cleaning status unknown)"
        ),
        query_term=eng["query_role"],
        query_family="legacy_english_roles",
    ))

    akk = pd.read_csv(BASE / "jobs_akkasi_style_dataset.csv", dtype=str)
    print(f"jobs_akkasi_style_dataset.csv: {len(akk):,} rows", flush=True)
    akk_iso = akk["query_location_label"].map(COUNTRY_ISO).fillna("??")
    parts.append(add_provenance(
        akk,
        batch_id=BATCH_ID,
        stratum=_stratum(akk_iso),
        country_iso=akk_iso,
        acquisition_timestamp="2025-11-29T00:00:00Z",
        acquisition_method=(
            "python-jobspy legacy scrape, Nov 2025 "
            "(file jobs_akkasi_style_dataset.csv; sites=indeed,linkedin,glassdoor; "
            "EU-focused; cleaning status unknown)"
        ),
        query_term=akk["query_role"],
        query_family="legacy_akkasi_roles",
    ))

    combined = pd.concat(parts, ignore_index=True)
    batch_dir = write_batch(
        combined,
        BATCH_ID,
        manifest_extra={
            "query_plan_version": "qp-v0-2025-11 (legacy, reconstructed)",
            "provenance_note": (
                "Reconstructed post-hoc on 2026-10-20 from DATA/preprocessing/*.csv "
                "(scraped Nov-Dec 2025; exact scrape scripts not preserved). Oldest "
                "known snapshot. DEVIATIONS: (1) per-query acquisition timestamps not "
                "recorded (file mtimes used); (2) text cleaning status unknown; "
                "(3) contains linkedin (and glassdoor) rows collected before the "
                "SOURCE_AUDIT K1 drop-LinkedIn recommendation - governance decides "
                "whether they enter N_eligible; (4) description null rates are high "
                "for linkedin/glassdoor rows (21-57% per file); (5) invalid "
                "date_posted values exist (min 1983/2017) - date-sanity rule applies."
            ),
            "source_files": [
                str(BASE / "jobs_rpl_id_raw_2.csv"),
                str(BASE / "jobs_english_only_dataset.csv"),
                str(BASE / "jobs_akkasi_style_dataset.csv"),
            ],
        },
    )
    print(f"\nWrote {len(combined):,} rows -> {batch_dir}", flush=True)
    print(combined.groupby(["stratum", "country_iso"]).size().sort_values(ascending=False).to_string(), flush=True)
    unknown = combined[combined["country_iso"] == "??"]
    if len(unknown):
        print(f"\nWARNING: {len(unknown)} rows with unmapped country labels:", flush=True)
        print(unknown["query_location_label"].value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
