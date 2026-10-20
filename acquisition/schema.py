"""Tabel 3.10 provenance schema for the raw-immutable OJP layer.

Raw principle: source columns are stored exactly as fetched (no cleaning at this
layer); this module only ADDS provenance columns. Cleaning happens downstream and
produces derived versions linked by `source_record_id`.
"""
from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd

from . import HARNESS_VERSION

SCHEMA_VERSION = "tabel-3.10-v1"
# Fixed project salt: employer_hash must stay deterministic across batches so
# first_seen/last_seen and dedup work, while keeping employer names out of
# derived layers.
EMPLOYER_HASH_SALT = "ojp-rpl-2026"

# Query params that carry tracking noise, not record identity.
_TRACKING_PARAMS = re.compile(r"^(utm_|ref$|refid$|tk$|from$|vjs$|xkcb$|fccid$)", re.IGNORECASE)

PROVENANCE_COLUMNS = [
    "source_record_id",
    "batch_id",
    "stratum",
    "country_iso",
    "acquisition_timestamp",
    "acquisition_method",
    "harness_version",
    "schema_version",
    "query_term",
    "query_family",
    "employer_hash",
]


def normalize_url(url: str) -> str:
    """Canonicalize a job URL: lowercase scheme/host, drop fragment + tracking params."""
    if not isinstance(url, str) or not url.strip():
        return ""
    parts = urlsplit(url.strip())
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not _TRACKING_PARAMS.match(k)]
    return urlunsplit((
        parts.scheme.lower(),
        parts.netloc.lower(),
        parts.path.rstrip("/"),
        urlencode(sorted(query)),
        "",
    ))


def _sha1_16(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def make_source_record_id(row: pd.Series) -> str:
    """Hashed-URL record id (Tabel 3.10); falls back to content fields if URL missing."""
    url = normalize_url(row.get("job_url") or "")
    if url:
        return "u-" + _sha1_16(url)
    fallback = "|".join(str(row.get(c) or "") for c in
                        ("site", "title", "company", "location", "date_posted"))
    return "f-" + _sha1_16(fallback.lower())


def make_employer_hash(company: str) -> str:
    if not isinstance(company, str) or not company.strip():
        return ""
    normalized = re.sub(r"\s+", " ", company).strip().lower()
    return _sha1_16(f"{EMPLOYER_HASH_SALT}|{normalized}")


def add_provenance(
    df: pd.DataFrame,
    *,
    batch_id: str,
    stratum,
    country_iso,
    acquisition_timestamp: str,
    acquisition_method: str,
    query_term=None,
    query_family=None,
) -> pd.DataFrame:
    """Return a copy of `df` with the Tabel 3.10 provenance columns added.

    `stratum`, `country_iso`, `query_term`, `query_family` may be scalars or
    pandas Series aligned to `df` (legacy ingest maps them per row).
    """
    out = df.copy()
    out["source_record_id"] = out.apply(make_source_record_id, axis=1)
    out["employer_hash"] = out.get("company", pd.Series("", index=out.index)).map(make_employer_hash)
    out["batch_id"] = batch_id
    out["stratum"] = stratum
    out["country_iso"] = country_iso
    out["acquisition_timestamp"] = acquisition_timestamp
    out["acquisition_method"] = acquisition_method
    out["harness_version"] = HARNESS_VERSION
    out["schema_version"] = SCHEMA_VERSION
    if query_term is not None:
        out["query_term"] = query_term
    if query_family is not None:
        out["query_family"] = query_family
    return out
