"""Jobstreet (SEEK) adapter: public v5 search API + server-rendered job detail state.

Evaluation prototype per SOURCE_AUDIT decision K3 (2026-10-29). Endpoints are the
same ones the id.jobstreet.com web app calls; access is read-only, unauthenticated,
and paced (politeness, SOURCE_AUDIT §2). Descriptions are stored as returned
(raw HTML) per the raw-immutable contract.
"""
from __future__ import annotations

import json
import random
import re
import time

import pandas as pd
import requests

ADAPTER_VERSION = "jobstreet-0.1.0"

SITES = {
    "ID": {"host": "id.jobstreet.com", "site_key": "ID-Main", "job_path": "id/job"},
    "MY": {"host": "my.jobstreet.com", "site_key": "MY-Main", "job_path": "job"},
    "SG": {"host": "sg.jobstreet.com", "site_key": "SG-Main", "job_path": "job"},
    "PH": {"host": "ph.jobstreet.com", "site_key": "PH-Main", "job_path": "job"},
}

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"),
    "Accept": "application/json",
}

PAGE_SIZE = 30
SEARCH_DELAY = (0.8, 1.8)   # seconds between search pages
DETAIL_DELAY = (1.2, 2.6)   # seconds between detail fetches

_REDUX_RE = re.compile(r"window\.SEEK_REDUX_DATA\s*=\s*(\{.*)", re.DOTALL)


def _sleep(bounds: tuple[float, float]) -> None:
    time.sleep(random.uniform(*bounds))


def search(session: requests.Session, term: str, site: dict, results_wanted: int) -> list[dict]:
    """Paginate the v5 search API; returns raw job dicts as the API serves them."""
    jobs: list[dict] = []
    page = 1
    while len(jobs) < results_wanted:
        r = session.get(
            f"https://{site['host']}/api/jobsearch/v5/search",
            params={"siteKey": site["site_key"], "sourcesystem": "houston",
                    "keywords": term, "pageSize": PAGE_SIZE, "page": page},
            headers=HEADERS, timeout=25,
        )
        r.raise_for_status()
        data = r.json().get("data") or []
        if not data:
            break
        jobs.extend(data)
        page += 1
        _sleep(SEARCH_DELAY)
    return jobs[:results_wanted]


def fetch_detail(session: requests.Session, job_id: str, site: dict) -> dict | None:
    """Parse the server-rendered SEEK_REDUX_DATA for one job; None on any failure."""
    try:
        r = session.get(f"https://{site['host']}/{site['job_path']}/{job_id}",
                        headers={"User-Agent": HEADERS["User-Agent"]}, timeout=25)
        if r.status_code != 200:
            return None
        m = _REDUX_RE.search(r.text)
        if not m:
            return None
        blob = m.group(1).replace(":undefined", ":null")
        data, _ = json.JSONDecoder().raw_decode(blob)
        return ((data.get("jobdetails") or {}).get("result") or {}).get("job")
    except Exception:
        return None


def scrape_term_jobstreet(
    *,
    term: str,
    country_iso: str,
    results_wanted: int,
    fetch_description: bool = True,
    detail_cache: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """scrape_term-compatible entry point; never raises. `detail_cache` is shared
    across terms in one run so each job's detail page is fetched exactly once."""
    site = SITES[country_iso]
    cache = detail_cache if detail_cache is not None else {}
    log = {"term": term, "location": country_iso, "sites": ["jobstreet"]}
    t0 = time.time()
    try:
        with requests.Session() as session:
            found = search(session, term, site, results_wanted)
            rows = []
            n_detail_fail = 0
            for j in found:
                job_id = str(j.get("id"))
                detail = cache.get(job_id)
                if detail is None and fetch_description and job_id not in cache:
                    detail = fetch_detail(session, job_id, site)
                    cache[job_id] = detail
                    if detail is None:
                        n_detail_fail += 1
                    _sleep(DETAIL_DELAY)
                detail = detail or {}
                locations = j.get("locations") or []
                rows.append({
                    "id": job_id,
                    "site": "jobstreet",
                    "job_url": f"https://{site['host']}/{site['job_path']}/{job_id}",
                    "title": j.get("title"),
                    "company": j.get("companyName"),
                    "location": "; ".join(str(x.get("label")) for x in locations if x.get("label")),
                    "date_posted": j.get("listingDate"),
                    "description": detail.get("content"),
                    "teaser": j.get("teaser"),
                    "bullet_points": json.dumps(j.get("bulletPoints") or [], ensure_ascii=False),
                    "salary_label": j.get("salaryLabel"),
                    "work_types": json.dumps(j.get("workTypes") or [], ensure_ascii=False),
                    "classifications": json.dumps(j.get("classifications") or [], ensure_ascii=False),
                    "expires_at": detail.get("expiresAt"),
                    "is_verified": detail.get("isVerified"),
                    "adapter_version": ADAPTER_VERSION,
                })
        log.update(rows=len(rows), secs=round(time.time() - t0, 1),
                   detail_failures=n_detail_fail, error=None)
        return pd.DataFrame(rows), log
    except Exception as e:
        log.update(rows=0, secs=round(time.time() - t0, 1), error=f"{type(e).__name__}: {e}")
        return pd.DataFrame(), log
