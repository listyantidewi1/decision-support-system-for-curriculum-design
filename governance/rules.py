"""Deterministic Tabel 3.9 rule engine: flags per record, never deletes.

Domain logic (function over title, per the note under Tabel 3.9):
  1. exclude-title match with no include-title match and fewer than the general
     description-signal threshold -> EXCLUDE
  2. include-title match -> INCLUDE
  3. boundary-title match (data/AI-ML/cyber/cloud) -> INCLUDE only with enough
     distinct software-development signals in the description
  4. anything else -> INCLUDE only with enough description signals
"""
from __future__ import annotations

import hashlib
import json
import re
import warnings
from pathlib import Path

import pandas as pd

# Config patterns use groups for alternation, not capture; pandas warns either way.
warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def load_rules(name: str = "tabel_3_9_v1.json") -> dict:
    return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))


def _any_match(series: pd.Series, patterns: list[str]) -> pd.Series:
    out = pd.Series(False, index=series.index)
    for pat in patterns:
        out |= series.str.contains(pat, case=False, regex=True, na=False)
    return out


def _count_matches(series: pd.Series, patterns: list[str]) -> pd.Series:
    """Number of DISTINCT patterns that match each text."""
    counts = pd.Series(0, index=series.index)
    for pat in patterns:
        counts += series.str.contains(pat, case=False, regex=True, na=False).astype(int)
    return counts


def normalized_text_hash(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    norm = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


def apply_rules(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Return a flags frame aligned to df (adds no columns to df itself)."""
    flags = pd.DataFrame(index=df.index)
    title = df["title"].fillna("").astype(str)
    desc = df["description"].fillna("").astype(str)

    # --- Metadata (Tabel 3.9 Metadata) ---
    meta_ok = pd.Series(True, index=df.index)
    for col in rules["metadata_required"]:
        if col in df.columns:
            meta_ok &= df[col].notna() & (df[col].astype(str).str.strip() != "") \
                       & (df[col].astype(str) != "??") & (df[col].astype(str).str.lower() != "nan")
        else:
            meta_ok &= False
    flags["meta_valid"] = meta_ok

    # --- Period + date sanity (Tabel 3.9 Periode) ---
    posted = pd.to_datetime(df["date_posted"], errors="coerce")
    acquired = pd.to_datetime(df["acquisition_timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
    start = pd.Timestamp(rules["period"]["window_start"])
    end = pd.Timestamp(rules["period"]["window_end"])
    flags["date_in_window"] = (
        posted.notna() & (posted >= start) & (posted <= end)
        & (posted <= acquired + pd.Timedelta(days=2))
    )

    # --- Domain (Tabel 3.9 Domain okupasi + Okupasi berbatas) ---
    dom = rules["domain"]
    t_inc = _any_match(title, dom["include_title_patterns"])
    t_bnd = _any_match(title, dom["boundary_title_patterns"])
    t_exc = _any_match(title, dom["exclude_title_patterns"])
    d_sig = _count_matches(desc, dom["include_description_patterns"])
    flags["desc_signals"] = d_sig

    status = pd.Series("exclude_no_signal", index=df.index)
    status[d_sig >= dom["min_desc_signals_general"]] = "include_function"
    status[t_bnd & (d_sig >= dom["min_desc_signals_boundary"])] = "include_boundary"
    status[t_bnd & (d_sig < dom["min_desc_signals_boundary"])] = "exclude_boundary"
    status[t_inc] = "include_title"
    status[t_exc & ~t_inc & (d_sig < dom["min_desc_signals_general"])] = "exclude_title"
    flags["domain_status"] = status
    flags["domain_ok"] = status.str.startswith("include")

    # --- Content (Tabel 3.9 Konten) ---
    flags["content_ok"] = ~_any_match(title, rules["content"]["non_job_title_patterns"]) | t_inc

    # --- Conditional platforms (K5) ---
    site = df.get("site", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    flags["conditional_platform"] = site.isin(rules["conditional_platforms"]["platforms"])

    return flags


def detect_languages(texts: pd.Series, max_chars: int = 1200) -> pd.Series:
    """langdetect over a Series (deterministic seed); 'und' when detection fails."""
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0

    def _detect(t):
        if not isinstance(t, str) or len(t.strip()) < 30:
            return "und"
        try:
            return detect(t[:max_chars])
        except Exception:
            return "und"

    return texts.map(_detect)
