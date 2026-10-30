"""Kalibrr adapter: public job-board search API (K3 source 3.7).

Returns full descriptions plus education_level and application_end_date
(→ last_seen, Tabel 3.10). Read-only, unauthenticated, paced. Descriptions
stored as returned (HTML) per the raw-immutable contract.
"""
from __future__ import annotations

import random
import time

import pandas as pd
import requests

ADAPTER_VERSION = "kalibrr-0.1.0"
SEARCH_URL = "https://www.kalibrr.com/kjs/job_board/search"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"),
    "Accept": "application/json",
}
PAGE_SIZE = 50
SEARCH_DELAY = (1.0, 2.2)

# Kalibrr country label (not ISO) expected by the API.
COUNTRY_LABEL = {"ID": "Indonesia", "PH": "Philippines"}


def _sleep() -> None:
    time.sleep(random.uniform(*SEARCH_DELAY))


def scrape_term_kalibrr(
    *,
    term: str,
    country_iso: str,
    results_wanted: int,
    **_ignored,
) -> tuple[pd.DataFrame, dict]:
    """scrape_term-compatible; paginates offset until results_wanted or exhausted."""
    label = COUNTRY_LABEL.get(country_iso, "Indonesia")
    log = {"term": term, "location": country_iso, "sites": ["kalibrr"]}
    t0 = time.time()
    rows: list[dict] = []
    try:
        with requests.Session() as session:
            offset = 0
            total = None
            while len(rows) < results_wanted:
                r = session.get(SEARCH_URL, headers=HEADERS, timeout=25, params={
                    "text": term, "country": label,
                    "limit": PAGE_SIZE, "offset": offset})
                r.raise_for_status()
                data = r.json()
                total = data.get("count") if total is None else total
                jobs = data.get("jobs") or []
                if not jobs:
                    break
                for j in jobs:
                    jid = str(j.get("id"))
                    company = j.get("company_name") or (j.get("company") or {}).get("name")
                    loc = j.get("google_location") or {}
                    rows.append({
                        "id": jid,
                        "site": "kalibrr",
                        "job_url": f"https://www.kalibrr.com/kjs/job/{jid}",
                        "title": j.get("name") or j.get("position"),
                        "company": company,
                        "location": (loc.get("address_components_json") or {}).get("city")
                                    or loc.get("formatted_address") or label,
                        "date_posted": j.get("activation_date"),
                        "description": j.get("description"),
                        "application_end_date": j.get("application_end_date"),
                        "education_level": j.get("education_level"),
                        "function": j.get("function"),
                        "base_salary": str(j.get("base_salary")),
                        "is_hybrid": j.get("is_hybrid"),
                        "adapter_version": ADAPTER_VERSION,
                    })
                offset += PAGE_SIZE
                if total is not None and offset >= total:
                    break
                _sleep()
        # Kalibrr can return the same job for different terms; de-dup within the call.
        df = pd.DataFrame(rows).drop_duplicates("id").head(results_wanted)
        log.update(rows=len(df), secs=round(time.time() - t0, 1),
                   total_available=total, error=None)
        return df, log
    except Exception as e:
        log.update(rows=0, secs=round(time.time() - t0, 1), error=f"{type(e).__name__}: {e}")
        return pd.DataFrame(), log
