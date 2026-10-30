"""Adzuna official API adapter — BENCHMARK ONLY (SOURCE_AUDIT K4, 2026-10-30).

Descriptions are truncated to 500 chars by the free API, so Adzuna records are
NOT part of the five-tag extraction corpus. This adapter provides an independent,
legally-clean cross-check of demand counts and a flagged sample for the
representativeness audit. Keys live in api_keys/adzuna.txt (gitignored).
"""
from __future__ import annotations

import time
from pathlib import Path

import requests

ADAPTER_VERSION = "adzuna-benchmark-0.1.0"
BASE = "https://api.adzuna.com/v1/api/jobs"
KEYS_PATH = Path(__file__).resolve().parents[1] / "api_keys" / "adzuna.txt"

# Adzuna country code -> (our stratum). Adzuna supports neither Indonesia nor
# Ireland (ie returns no data), so neither appears here.
COUNTRIES = {
    "us": "oecd", "gb": "oecd", "ca": "oecd", "au": "oecd", "nz": "oecd",
    "de": "oecd", "fr": "oecd", "nl": "oecd",
    "sg": "asean",
}
RESULTS_PER_PAGE = 50
POLITE_DELAY = 0.4


def load_adzuna_keys() -> tuple[str, str]:
    creds = {}
    for line in KEYS_PATH.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sep = "=" if "=" in line else (":" if ":" in line else None)
        if sep:
            k, v = line.split(sep, 1)
            creds[k.strip().lower()] = v.strip()
    app_id = creds.get("app_id") or creds.get("id")
    app_key = creds.get("app_key") or creds.get("key")
    if not app_id or not app_key:
        raise ValueError(f"adzuna.txt must define app_id and app_key (got {sorted(creds)})")
    return app_id, app_key


def count(session: requests.Session, app_id: str, app_key: str,
          country: str, term: str) -> int | None:
    """One cheap call: total postings matching `term` in `country` (demand signal)."""
    r = session.get(f"{BASE}/{country}/search/1", timeout=25, params={
        "app_id": app_id, "app_key": app_key, "results_per_page": 1, "what": term})
    if r.status_code != 200:
        return None
    return r.json().get("count")


def search(session: requests.Session, app_id: str, app_key: str,
           country: str, term: str, pages: int = 1) -> list[dict]:
    """Paginated records (500-char truncated descriptions) for the sample table."""
    out = []
    for page in range(1, pages + 1):
        r = session.get(f"{BASE}/{country}/search/{page}", timeout=25, params={
            "app_id": app_id, "app_key": app_key,
            "results_per_page": RESULTS_PER_PAGE, "what": term})
        if r.status_code != 200:
            break
        results = r.json().get("results", [])
        if not results:
            break
        out.extend(results)
        time.sleep(POLITE_DELAY)
    return out
