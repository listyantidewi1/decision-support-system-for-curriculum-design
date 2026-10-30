"""Pull the Adzuna benchmark: demand-count matrix + flagged sample records.

Writes to DATA/corpus_benchmark/adzuna/ (SEPARATE from the extraction corpus in
DATA/corpus_raw/). Sample records carry description_truncated=true and are never
ingested into the five-tag extraction pipeline. See SOURCE_AUDIT K4.

Run:  .venv-ml\\Scripts\\python.exe -m acquisition.run_adzuna_benchmark
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from .adzuna_adapter import (ADAPTER_VERSION, COUNTRIES, load_adzuna_keys, count, search)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "DATA" / "corpus_benchmark" / "adzuna"

# Core RPL terms for the cross-check (kept small to respect the free quota).
TERMS = [
    "software engineer", "software developer", "programmer", "web developer",
    "backend developer", "frontend developer", "mobile developer", "devops engineer",
    "qa engineer", "data engineer", "game developer", "database administrator",
]
SAMPLE_PAGES = 2  # per country: 2 x 50 = up to 100 flagged sample records


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    app_id, app_key = load_adzuna_keys()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    count_rows, sample_rows = [], []
    n_calls = 0
    with requests.Session() as s:
        for cc, stratum in COUNTRIES.items():
            for term in TERMS:
                c = count(s, app_id, app_key, cc, term)
                n_calls += 1
                count_rows.append({"country": cc, "stratum": stratum, "term": term, "count": c})
                print(f"  count {cc}/{term}: {c}", flush=True)
                time.sleep(0.4)
            # one sample pull per country on the broadest term
            recs = search(s, app_id, app_key, cc, "software developer", pages=SAMPLE_PAGES)
            n_calls += SAMPLE_PAGES
            for j in recs:
                desc = j.get("description", "") or ""
                url = j.get("redirect_url", "") or ""
                sample_rows.append({
                    "source_record_id": "adz-" + hashlib.sha1(
                        (url or str(j.get("id"))).encode("utf-8")).hexdigest()[:16],
                    "country": cc, "stratum": stratum, "site": "adzuna",
                    "title": j.get("title"),
                    "company": (j.get("company") or {}).get("display_name"),
                    "location": (j.get("location") or {}).get("display_name"),
                    "date_posted": j.get("created"),
                    "description_truncated": desc,     # <= 500 chars, flagged by column name
                    "description_truncated_flag": True,
                    "category": (j.get("category") or {}).get("label"),
                    "salary_min": j.get("salary_min"), "salary_max": j.get("salary_max"),
                })
            print(f"  sample {cc}: {len(recs)} records", flush=True)

    counts_df = pd.DataFrame(count_rows)
    counts_df.to_csv(out_dir / "demand_counts.csv", index=False, encoding="utf-8")
    sample_df = pd.DataFrame(sample_rows).drop_duplicates("source_record_id")
    sample_df.astype(str).to_parquet(out_dir / "sample_records.parquet", index=False)

    by_stratum = counts_df.dropna(subset=["count"]).groupby("stratum")["count"].sum()
    manifest = {
        "created_utc": _utc(), "adapter_version": ADAPTER_VERSION,
        "purpose": "BENCHMARK / independent demand cross-check — NOT extraction corpus",
        "description_limitation": "Adzuna free API truncates descriptions to 500 chars.",
        "n_api_calls": n_calls, "terms": TERMS, "countries": list(COUNTRIES),
        "total_demand_count_by_stratum": {k: int(v) for k, v in by_stratum.items()},
        "n_sample_records": int(len(sample_df)),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nAdzuna benchmark -> {out_dir}")
    print(f"API calls used: {n_calls} (free tier ~250/day)")
    print("demand counts by stratum:", manifest["total_demand_count_by_stratum"])


if __name__ == "__main__":
    main()
