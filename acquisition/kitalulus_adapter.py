"""Kitalulus adapter: public listing + JobPosting JSON-LD detail (K8 source).

Kitalulus (Indonesia) serves each job's detail page with a schema.org JobPosting
block in <script type="application/ld+json"> — machine-readable structured data
published for SEO, read as-is (no anti-bot evasion). It carries the full Tabel 3.10
payload incl. description, datePosted, validThrough, education/experience
requirements and salary.

ETHICS (K8, SOURCE_AUDIT): robots.txt `User-Agent: *` permits the public /lowongan
pages (only /auth/ and /my/ are disallowed, which we never touch). robots.txt also
carries a GPTBot-specific `Disallow: /`; we are not GPTBot and this corpus is for
research analysis, not model-training redistribution, but the signal is documented
transparently and volume is kept low. Read-only, unauthenticated, paced.

Limitation (v0.1): search returns the first listing page per keyword (~30 results);
broad terms may be truncated. RPL keyword searches are narrow, so coverage is high.
"""
from __future__ import annotations

import json
import random
import re
import time

import pandas as pd
import requests

ADAPTER_VERSION = "kitalulus-0.1.0"
BASE = "https://www.kitalulus.com"
SEARCH_URL = BASE + "/lowongan"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
}
SEARCH_DELAY = (1.0, 2.0)
DETAIL_DELAY = (1.2, 2.6)

_DETAIL_HREF = re.compile(r'href="(/lowongan/detail/[a-zA-Z0-9][a-zA-Z0-9/_\-]+)"')
_LDJSON = re.compile(r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', re.S)


def _sleep(bounds: tuple[float, float]) -> None:
    time.sleep(random.uniform(*bounds))


def _first(obj):
    return obj[0] if isinstance(obj, list) and obj else obj


def _flatten(v) -> str:
    """Readable scalar for a JSON-LD field that may be a str, list, or nested dict."""
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "; ".join(x for x in (_flatten(i) for i in v) if x)
    if isinstance(v, dict):
        for k in ("name", "credentialCategory", "monthsOfExperience", "value", "description"):
            if k in v and v[k] not in (None, ""):
                return str(v[k])
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _parse_jobposting(html: str) -> dict | None:
    for block in _LDJSON.findall(html):
        try:
            obj = json.loads(block)
        except Exception:
            continue
        for o in (obj if isinstance(obj, list) else [obj]):
            if isinstance(o, dict) and o.get("@type") == "JobPosting":
                return o
    return None


def _row_from_jobposting(o: dict, url: str, slug: str) -> dict:
    org = o.get("hiringOrganization") or {}
    loc = _first(o.get("jobLocation")) or {}
    addr = (loc.get("address") if isinstance(loc, dict) else {}) or {}
    city = addr.get("addressLocality") or addr.get("addressRegion")
    return {
        "id": slug,
        "site": "kitalulus",
        "job_url": url,
        "title": o.get("title"),
        "company": org.get("name") if isinstance(org, dict) else _flatten(org),
        "location": city or addr.get("addressCountry") or "Indonesia",
        "date_posted": o.get("datePosted"),
        "description": o.get("description"),            # raw HTML, immutable
        "valid_through": o.get("validThrough"),          # -> last_seen (Tabel 3.10)
        "employment_type": _flatten(o.get("employmentType")),
        "education_requirement": _flatten(o.get("educationRequirements")),
        "experience_requirement": _flatten(o.get("experienceRequirements")),
        "base_salary": json.dumps(o.get("baseSalary"), ensure_ascii=False)
                       if o.get("baseSalary") is not None else None,
        "identifier": _flatten(o.get("identifier")),
        "adapter_version": ADAPTER_VERSION,
    }


def scrape_term_kitalulus(
    *,
    term: str,
    country_iso: str,
    results_wanted: int,
    detail_cache: dict | None = None,
    **_ignored,
) -> tuple[pd.DataFrame, dict]:
    """scrape_term-compatible. Kitalulus is Indonesia-only."""
    log = {"term": term, "location": country_iso, "sites": ["kitalulus"]}
    t0 = time.time()
    if country_iso != "ID":
        log.update(rows=0, secs=0.0, error="kitalulus is Indonesia-only")
        return pd.DataFrame(), log
    cache = detail_cache if detail_cache is not None else {}
    rows: list[dict] = []
    try:
        with requests.Session() as session:
            session.headers.update(HEADERS)
            r = session.get(SEARCH_URL, params={"keyword": term}, timeout=25)
            r.raise_for_status()
            slugs: list[str] = []
            seen = set()
            for href in _DETAIL_HREF.findall(r.text):
                if href not in seen:
                    seen.add(href)
                    slugs.append(href)
            slugs = slugs[:results_wanted]
            _sleep(SEARCH_DELAY)

            for href in slugs:
                url = BASE + href
                slug = href.rsplit("/", 1)[-1]
                if url in cache:
                    if cache[url] is not None:
                        rows.append({**cache[url], "query_hit": term})
                    continue
                try:
                    dr = session.get(url, timeout=25)
                    jp = _parse_jobposting(dr.text) if dr.status_code == 200 else None
                except Exception:
                    jp = None
                row = _row_from_jobposting(jp, url, slug) if jp else None
                cache[url] = row
                if row is not None:
                    rows.append(row)
                _sleep(DETAIL_DELAY)

        df = pd.DataFrame(rows).drop_duplicates("id").head(results_wanted) if rows else pd.DataFrame()
        log.update(rows=len(df), secs=round(time.time() - t0, 1),
                   found_listings=len(slugs), error=None)
        return df, log
    except Exception as e:  # noqa: BLE001
        log.update(rows=0, secs=round(time.time() - t0, 1), error=f"{type(e).__name__}: {e}")
        return pd.DataFrame(), log
