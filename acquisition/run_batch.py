"""Run one acquisition batch from a versioned query-plan config (plan A2/A3).

Run:  .venv-ml\\Scripts\\python.exe -m acquisition.run_batch --config acquisition/configs/batch-002.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .batch_store import DEFAULT_ROOT, write_batch
from .jobspy_adapter import JOBSPY_VERSION, polite_sleep, scrape_term
from .jobstreet_adapter import ADAPTER_VERSION as JOBSTREET_VERSION
from .jobstreet_adapter import scrape_term_jobstreet
from .kalibrr_adapter import ADAPTER_VERSION as KALIBRR_VERSION
from .kalibrr_adapter import scrape_term_kalibrr
from .glints_adapter import ADAPTER_VERSION as GLINTS_VERSION
from .glints_adapter import scrape_term_glints
from .kitalulus_adapter import ADAPTER_VERSION as KITALULUS_VERSION
from .kitalulus_adapter import scrape_term_kitalulus
from .schema import add_provenance

CHECKPOINT_EVERY = 25


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iter_queries(config: dict):
    for stratum in config["strata"]:
        for country in stratum["countries"]:
            for term in stratum["terms"]:
                yield stratum, country, term


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the query plan and exit")
    args = ap.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_root = Path(args.out_root) if args.out_root else DEFAULT_ROOT
    batch_id = config["batch_id"]
    defaults = config.get("defaults", {})
    sites = defaults.get("sites", ["indeed"])
    sleep_seconds = defaults.get("sleep_seconds", 5)
    hours_old = defaults.get("hours_old")

    queries = list(iter_queries(config))
    print(f"[{batch_id}] plan {config.get('query_plan_version')}: {len(queries)} queries, "
          f"sites={sites}, hours_old={hours_old}", flush=True)
    if args.dry_run:
        for stratum, country, term in queries:
            print(f"  {stratum['name']:>9} {country['iso']} {term['term']}")
        return

    records_path = out_root / batch_id / "records.parquet"
    if records_path.exists():
        raise SystemExit(f"{records_path} already exists — batches are immutable.")
    checkpoint = out_root / "_inflight" / f"{batch_id}.partial.parquet"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    query_log: list[dict] = []
    zero_streak = 0
    jobstreet_cache: dict = {}  # job detail fetched once per run across all terms
    for i, (stratum, country, term) in enumerate(queries, 1):
        if "jobstreet" in sites:
            df, log = scrape_term_jobstreet(
                term=term["term"],
                country_iso=country["iso"],
                results_wanted=stratum.get("results_wanted", 100),
                fetch_description=stratum.get("fetch_description",
                                              defaults.get("fetch_description", True)),
                detail_cache=jobstreet_cache,
            )
        elif "kalibrr" in sites:
            df, log = scrape_term_kalibrr(
                term=term["term"],
                country_iso=country["iso"],
                results_wanted=stratum.get("results_wanted", 100),
            )
        elif "glints" in sites:
            df, log = scrape_term_glints(
                term=term["term"],
                country_iso=country["iso"],
                results_wanted=stratum.get("results_wanted", 100),
                detail_cache=jobstreet_cache,  # reuse the one per-run detail cache
            )
        elif "kitalulus" in sites:
            df, log = scrape_term_kitalulus(
                term=term["term"],
                country_iso=country["iso"],
                results_wanted=stratum.get("results_wanted", 100),
                detail_cache=jobstreet_cache,  # reuse the one per-run detail cache
            )
        else:
            df, log = scrape_term(
                sites=sites,
                term=term["term"],
                location=country["location"],
                country_indeed=country["country_indeed"],
                results_wanted=stratum.get("results_wanted", 100),
                hours_old=stratum.get("hours_old", hours_old),
                fetch_description=stratum.get("fetch_description",
                                              defaults.get("fetch_description", False)),
            )
        log.update(stratum=stratum["name"], country_iso=country["iso"])
        query_log.append(log)
        status = log["error"] or f"{log['rows']} rows"
        print(f"[{i}/{len(queries)}] {stratum['name']}/{country['iso']} "
              f"'{term['term']}': {status} ({log['secs']}s)", flush=True)

        # jobspy catches connection errors internally and returns an empty frame,
        # so an outage is only visible as a run of zero-row queries.
        zero_streak = zero_streak + 1 if log["rows"] == 0 else 0
        if zero_streak == 3:
            print("WARNING: 3 consecutive zero-row queries — possible network/endpoint "
                  "outage; affected queries may need a supplementary batch.", flush=True)
        # Circuit breaker: a 403/429 streak means the platform has rate-banned us.
        # Continuing extends the ban; stop and write what we have (batch-002e lesson:
        # Jobstreet WAF banned after ~33 heavy terms / ~2,000 detail fetches).
        if zero_streak >= 3 and all(
            q["error"] and ("403" in q["error"] or "429" in q["error"])
            for q in query_log[-3:]
        ):
            print("ABORTING remaining queries: rate-ban detected (403/429 streak). "
                  "Re-run the missing terms as a supplementary batch after a cooldown.",
                  flush=True)
            break

        if len(df):
            frames.append(add_provenance(
                df,
                batch_id=batch_id,
                stratum=stratum["name"],
                country_iso=country["iso"],
                acquisition_timestamp=_utc_now_iso(),
                acquisition_method=(
                    f"{JOBSTREET_VERSION} (SEEK v5 search API + rendered detail state)"
                    if "jobstreet" in sites else
                    f"{KALIBRR_VERSION} (Kalibrr public job_board search API)"
                    if "kalibrr" in sites else
                    f"{GLINTS_VERSION} (Glints Next.js embedded state + detail)"
                    if "glints" in sites else
                    f"{KITALULUS_VERSION} (Kitalulus listing + JobPosting JSON-LD)"
                    if "kitalulus" in sites else
                    f"python-jobspy {JOBSPY_VERSION} via acquisition.run_batch "
                    f"(sites={','.join(sites)})"
                ),
                query_term=term["term"],
                query_family=term["family"],
            ))
        if frames and i % CHECKPOINT_EVERY == 0:
            pd.concat(frames, ignore_index=True).astype(str).to_parquet(checkpoint, index=False)
        polite_sleep(sleep_seconds)

    if not frames:
        raise SystemExit("No data acquired at all — batch not written.")

    combined = pd.concat(frames, ignore_index=True)
    n_errors = sum(1 for q in query_log if q["error"])
    batch_dir = write_batch(
        combined,
        batch_id,
        manifest_extra={
            "query_plan_version": config.get("query_plan_version"),
            "jobspy_version": JOBSPY_VERSION,
            "config": config,
            "n_queries": len(queries),
            "n_queries_failed": n_errors,
            "query_log": query_log,
            "notes": config.get("notes"),
        },
        out_root=out_root,
    )
    checkpoint.unlink(missing_ok=True)
    print(f"\n[{batch_id}] DONE: {len(combined):,} raw rows "
          f"({combined['source_record_id'].nunique():,} unique records), "
          f"{n_errors} failed queries -> {batch_dir}", flush=True)
    print(combined.groupby(["stratum", "country_iso"]).size().to_string(), flush=True)


if __name__ == "__main__":
    main()
