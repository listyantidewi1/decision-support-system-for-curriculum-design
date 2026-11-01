"""Single-GET health probe for the Jobstreet/SEEK v5 search endpoint.

Used by the acquisition cadence (docs/corpus/ACQUISITION_CADENCE.md §1) to decide,
before each run, whether the Cloudflare managed-challenge ban (403 "Just a moment…",
active since 2026-10-29) has lifted. It makes ONE search-API GET per region and no
detail fetches, so it will not deepen an active ban.

Run:  .venv-ml\\Scripts\\python.exe -m acquisition.probe_jobstreet          # ID only
      .venv-ml\\Scripts\\python.exe -m acquisition.probe_jobstreet --all     # ID+MY+SG+PH
(Do not add -I: isolated mode drops cwd from sys.path and breaks the -m import.)

Exit code 0 = ID reachable (HTTP 200); 1 = ID still blocked. (MY/SG/PH are advisory.)
"""
from __future__ import annotations

import argparse
import sys

import requests

from .jobstreet_adapter import HEADERS, SITES


def probe(region: str) -> tuple[int, int]:
    """Return (http_status, n_results) for one region; n_results is 0 unless HTTP 200."""
    site = SITES[region]
    r = requests.get(
        f"https://{site['host']}/api/jobsearch/v5/search",
        params={"siteKey": site["site_key"], "sourcesystem": "houston",
                "keywords": "software developer", "pageSize": 5, "page": 1},
        headers=HEADERS, timeout=25,
    )
    n = len((r.json().get("data") or [])) if r.status_code == 200 else 0
    return r.status_code, n


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="probe MY/SG/PH too (advisory)")
    args = ap.parse_args()

    regions = ["ID"] + (["MY", "SG", "PH"] if args.all else [])
    id_ok = False
    for region in regions:
        try:
            status, n = probe(region)
            verdict = "REACHABLE" if status == 200 else "BLOCKED"
            print(f"{region}: HTTP {status} ({n} results) -> {verdict}")
            if region == "ID":
                id_ok = status == 200
        except Exception as e:  # noqa: BLE001
            print(f"{region}: EXCEPTION {type(e).__name__}: {e}")
    raise SystemExit(0 if id_ok else 1)


if __name__ == "__main__":
    main()
