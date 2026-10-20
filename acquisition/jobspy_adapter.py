"""JobSpy adapter: one polite, logged scrape per (term, country) — no cleaning."""
from __future__ import annotations

import random
import time

import pandas as pd
from jobspy import scrape_jobs

try:  # jobspy does not export __version__; read package metadata instead
    from importlib.metadata import version as _pkg_version
    JOBSPY_VERSION = _pkg_version("python-jobspy")
except Exception:  # pragma: no cover
    JOBSPY_VERSION = "unknown"

MAX_ATTEMPTS = 2
RETRY_SLEEP_SECONDS = 30


def scrape_term(
    *,
    sites: list[str],
    term: str,
    location: str,
    country_indeed: str,
    results_wanted: int,
    hours_old: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Fetch one query; returns (raw df, log entry). Retries once, never raises."""
    kwargs = dict(
        site_name=sites,
        search_term=term,
        location=location,
        country_indeed=country_indeed,
        results_wanted=results_wanted,
    )
    if hours_old:
        kwargs["hours_old"] = hours_old

    log = {"term": term, "location": location, "sites": sites}
    t0 = time.time()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            df = scrape_jobs(**kwargs)
            log.update(rows=int(len(df)), secs=round(time.time() - t0, 1), error=None)
            return df, log
        except Exception as e:
            log["error"] = f"{type(e).__name__}: {e}"
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_SLEEP_SECONDS)
    log.update(rows=0, secs=round(time.time() - t0, 1))
    return pd.DataFrame(), log


def polite_sleep(base_seconds: float) -> None:
    time.sleep(base_seconds + random.uniform(0, base_seconds))
