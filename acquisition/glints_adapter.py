"""Glints adapter: Next.js embedded state search + detail (K3 source 3.6).

Search page embeds jobs in __NEXT_DATA__ pageProps.initialJobs.jobsInPage
(30/page, paginate via ?page=N while hasMore). The full description is only on
the detail page, as a Draft.js JSON blob at initialData.data.descriptionJsonString;
we reconstruct plain text from its blocks and ALSO keep the raw JSON. Detail is
fetched once per unique job. Same rate-limit exposure as Jobstreet — the
run_batch circuit breaker covers it; keep session term budgets modest.
"""
from __future__ import annotations

import json
import random
import re
import time

import pandas as pd
import requests

ADAPTER_VERSION = "glints-0.1.0"
HEADERS = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36")}
EXPLORE = "https://glints.com/id/opportunities/jobs/explore"
SEARCH_DELAY = (1.2, 2.5)
DETAIL_DELAY = (1.5, 3.0)
_NEXT_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.DOTALL)
COUNTRY = {"ID": "ID"}


def _sleep(bounds):
    time.sleep(random.uniform(*bounds))


def _next_data(session, url, params=None):
    r = session.get(url, params=params, headers=HEADERS, timeout=25)
    if r.status_code != 200:
        return None
    m = _NEXT_RE.search(r.text)
    return json.loads(m.group(1)) if m else None


def _draftjs_to_text(json_string: str) -> str:
    """Join Draft.js block texts into newline-separated plain text."""
    try:
        blocks = json.loads(json_string).get("blocks", [])
        return "\n".join(b.get("text", "") for b in blocks).strip()
    except Exception:
        return ""


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (title or "").lower()).strip("-")


def search(session, term, country_code, results_wanted):
    jobs, page = [], 1
    while len(jobs) < results_wanted:
        d = _next_data(session, EXPLORE,
                       {"keyword": term, "country": country_code, "page": page})
        if not d:
            break
        ij = d["props"]["pageProps"].get("initialJobs") or {}
        batch = ij.get("jobsInPage") or []
        if not batch:
            break
        jobs.extend(batch)
        if not ij.get("hasMore"):
            break
        page += 1
        _sleep(SEARCH_DELAY)
    return jobs[:results_wanted]


def fetch_description(session, job) -> tuple[str, str]:
    """Returns (plain_text, raw_json_string) for one job; ('','') on failure."""
    url = f"https://glints.com/id/opportunities/jobs/{_slug(job.get('title'))}/{job.get('id')}"
    d = _next_data(session, url)
    if not d:
        return "", ""
    data = (d["props"]["pageProps"].get("initialData") or {}).get("data") or {}
    raw = data.get("descriptionJsonString") or ""
    return _draftjs_to_text(raw), raw


def scrape_term_glints(*, term, country_iso, results_wanted,
                       fetch_description_flag=True, detail_cache=None, **_ignored):
    cc = COUNTRY.get(country_iso, "ID")
    cache = detail_cache if detail_cache is not None else {}
    log = {"term": term, "location": country_iso, "sites": ["glints"]}
    t0 = time.time()
    rows, n_detail_fail = [], 0
    try:
        with requests.Session() as session:
            found = search(session, term, cc, results_wanted)
            for j in found:
                jid = j.get("id")
                if fetch_description_flag and jid not in cache:
                    cache[jid] = fetch_description(session, j)
                    if not cache[jid][0]:
                        n_detail_fail += 1
                    _sleep(DETAIL_DELAY)
                desc, desc_raw = cache.get(jid, ("", ""))
                comp = j.get("company") or {}
                rows.append({
                    "id": jid,
                    "site": "glints",
                    "job_url": f"https://glints.com/id/opportunities/jobs/{_slug(j.get('title'))}/{jid}",
                    "title": j.get("title"),
                    "company": comp.get("name") if isinstance(comp, dict) else comp,
                    "location": j.get("city") or (j.get("location") or {}).get("name") if isinstance(j.get("location"), dict) else j.get("city"),
                    "date_posted": j.get("createdAt"),
                    "description": desc,
                    "description_json_raw": desc_raw,
                    "education_level": j.get("educationLevel"),
                    "min_years_exp": j.get("minYearsOfExperience"),
                    "max_years_exp": j.get("maxYearsOfExperience"),
                    "skills_listed": json.dumps(j.get("skills") or [], ensure_ascii=False),
                    "work_arrangement": j.get("workArrangementOption"),
                    "updated_at": j.get("updatedAt"),
                    "adapter_version": ADAPTER_VERSION,
                })
        df = pd.DataFrame(rows).drop_duplicates("id")
        log.update(rows=len(df), secs=round(time.time() - t0, 1),
                   detail_failures=n_detail_fail, error=None)
        return df, log
    except Exception as e:
        log.update(rows=0, secs=round(time.time() - t0, 1), error=f"{type(e).__name__}: {e}")
        return pd.DataFrame(), log
