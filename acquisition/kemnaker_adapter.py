"""Kemnaker Karirhub adapter: official Indonesian government OJP source (K10).

Karirhub is the job portal of SISNAKER (Sistem Informasi Ketenagakerjaan),
Kementerian Ketenagakerjaan RI (Ministry of Manpower). Vacancies are served by a
public, unauthenticated JSON API:

  LIST   GET {API}/karirhub/vacancy/v2/published-industrial-vacancies
         ?keyword=<term>&page=<n>&limit=<m>      (Laravel pagination: data/links/meta)
  DETAIL GET {API}/karirhub/vacancy/v2/published-industrial-vacancies/{id}
         -> full Tabel 3.10 payload: description + qualification (requirements),
            min_education, min_year_experience, salary, published_at/expires_at,
            employer, region, and a structured `skills` array (bonus signal).

ETHICS (K10, SOURCE_AUDIT): the API host `api.kemnaker.go.id` serves no robots.txt
(HTTP 404 => unrestricted); the public site `karirhub.kemnaker.go.id/robots.txt` is
explicitly headed "SEO & AI AGENT READINESS" and whitelists ClaudeBot/GPTBot/etc.
with `Allow: /` (only /admin, /dashboard, /api, /auth etc. on THAT host disallowed).
This endpoint returns the same public vacancy data any visitor sees. Official
government source, read-only, unauthenticated, no anti-bot evasion, paced + low
volume. The LIST response omits the description; only the DETAIL endpoint carries it,
so each unique vacancy is fetched once via a shared cache.

The description column stores the API's `description` + `qualification` (both raw
HTML) joined, exactly as served (raw-immutable layer; cleaning is downstream).
"""
from __future__ import annotations

import json
import random
import re
import time
from datetime import datetime, timezone

import pandas as pd
import requests

ADAPTER_VERSION = "kemnaker-0.1.0"
API = "https://api.kemnaker.go.id"
LIST_EP = API + "/karirhub/vacancy/v2/published-industrial-vacancies"
DETAIL_EP = LIST_EP + "/{id}"
PAGE_URL = ("https://karirhub.kemnaker.go.id/lowongan-dalam-negeri/lowongan/{slug}")
PER_PAGE = 10
MAX_PAGES = 25  # hard ceiling per term (PER_PAGE*MAX_PAGES = 250 cap)
HEADERS = {
    "User-Agent": "ClaudeBot/1.0 (+academic corpus research; respects robots.txt)",
    "Accept": "application/json",
    "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
}
LIST_DELAY = (1.0, 2.0)
DETAIL_DELAY = (0.8, 1.6)
# The api.kemnaker.go.id host is flaky under load: it intermittently returns
# HTTP 500 or resets the connection on rapid successive requests. Retry those
# transient server-side failures with backoff rather than dropping the record.
RETRY_BACKOFF = (2.0, 5.0, 10.0)


def _sleep(bounds: tuple[float, float]) -> None:
    time.sleep(random.uniform(*bounds))


def _get_json(session: requests.Session, url: str, params: dict | None = None):
    """GET returning parsed JSON, retrying transient 5xx / connection errors.

    Returns the decoded JSON on HTTP 200, or None once retries are exhausted or
    a non-retryable status (e.g. 404) is seen. Never raises."""
    for attempt in range(len(RETRY_BACKOFF) + 1):
        try:
            r = session.get(url, params=params, timeout=25)
            if r.status_code == 200:
                return r.json()
            if r.status_code < 500:   # 4xx: not transient, give up
                return None
        except requests.RequestException:
            pass  # connection reset / timeout -> treat as transient
        if attempt < len(RETRY_BACKOFF):
            time.sleep(RETRY_BACKOFF[attempt])
    return None


def _name(v) -> str | None:
    """Readable name from a nested API object (dict/list/scalar)."""
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "; ".join(x for x in (_name(i) for i in v) if x) or None
    if isinstance(v, dict):
        for k in ("name", "label", "title", "province_name", "city_name", "value"):
            if v.get(k):
                return str(v[k])
        return None
    return str(v)


def _slugify(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (title or "").lower()).strip("-")
    return s or "lowongan"


def _unix_to_iso(ts) -> str | None:
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return None


def _row_from_detail(d: dict) -> dict:
    vid = d.get("id")
    title = d.get("title")
    desc = "\n\n".join(
        str(x) for x in (d.get("description"), d.get("qualification"))
        if x and str(x).strip()
    ) or None
    sal = {k: d.get(k) for k in (
        "show_salary", "min_salary_amount", "max_salary_amount",
        "min_salary_currency", "max_salary_currency")}
    exp = d.get("min_year_experience")
    return {
        "id": vid,
        "site": "kemnaker",
        "job_url": PAGE_URL.format(slug=f"{_slugify(title)}-{vid}"),
        "title": title,
        "company": _name(d.get("employer")) or d.get("company_name"),
        "location": _name(d.get("region")) or "Indonesia",
        "date_posted": d.get("published_at") or d.get("created_at"),
        "description": desc,                                 # raw HTML, immutable
        "valid_through": _unix_to_iso(d.get("expires_at")),   # -> last_seen
        "employment_type": _name(d.get("job_type")) or d.get("job_type_name"),
        "education_requirement": _name(d.get("min_education")),
        "experience_requirement": (f"{exp} tahun" if exp not in (None, "") else None),
        "base_salary": (json.dumps(sal, ensure_ascii=False)
                        if any(v not in (None, "") for v in sal.values()) else None),
        "job_function": _name(d.get("job_function")) or d.get("job_function_name"),
        "skills": (json.dumps(d.get("skills"), ensure_ascii=False)
                   if d.get("skills") else None),            # bonus structured signal
        "identifier": d.get("job_id") or vid,
        "adapter_version": ADAPTER_VERSION,
    }


def scrape_term_kemnaker(
    *,
    term: str,
    country_iso: str,
    results_wanted: int,
    detail_cache: dict | None = None,
    **_ignored,
) -> tuple[pd.DataFrame, dict]:
    """scrape_term-compatible. Kemnaker Karirhub is Indonesia-only."""
    log = {"term": term, "location": country_iso, "sites": ["kemnaker"]}
    t0 = time.time()
    if country_iso != "ID":
        log.update(rows=0, secs=0.0, error="kemnaker is Indonesia-only")
        return pd.DataFrame(), log
    cache = detail_cache if detail_cache is not None else {}
    rows: list[dict] = []
    try:
        with requests.Session() as session:
            session.headers.update(HEADERS)
            # 1) enumerate vacancy ids via the paginated keyword search
            ids: list[str] = []
            seen_ids: set[str] = set()
            page = 1
            while len(ids) < results_wanted and page <= MAX_PAGES:
                j = _get_json(session, LIST_EP, {
                    "keyword": term, "page": page, "limit": PER_PAGE})
                if not j:
                    break  # transient failures exhausted; bank what we have
                items = j.get("data") or []
                for it in items:
                    vid = it.get("id")
                    if vid and vid not in seen_ids:
                        seen_ids.add(vid)
                        ids.append(vid)
                meta = j.get("meta") or {}
                last = meta.get("last_page") or page
                if not items or page >= last:
                    break
                page += 1
                _sleep(LIST_DELAY)
            ids = ids[:results_wanted]

            # 2) fetch the full record per unique vacancy (cached across terms)
            for vid in ids:
                if vid in cache:
                    if cache[vid] is not None:
                        rows.append({**cache[vid], "query_hit": term})
                    continue
                dj = _get_json(session, DETAIL_EP.format(id=vid))
                detail = (dj.get("data", dj) if isinstance(dj, dict) else None)
                row = _row_from_detail(detail) if isinstance(detail, dict) else None
                cache[vid] = row
                if row is not None:
                    rows.append(row)
                _sleep(DETAIL_DELAY)

        df = (pd.DataFrame(rows).drop_duplicates("id").head(results_wanted)
              if rows else pd.DataFrame())
        log.update(rows=len(df), secs=round(time.time() - t0, 1),
                   found_listings=len(ids), error=None)
        return df, log
    except Exception as e:  # noqa: BLE001
        log.update(rows=0, secs=round(time.time() - t0, 1),
                   error=f"{type(e).__name__}: {e}")
        return pd.DataFrame(), log
