"""Draw the stratified gold-standard sentence sample (pilot 100 -> main 1,200).

Sampling design (purposive, documented in the sample manifest):
  * per-stratum quotas with an Indonesian + bilingual boost (STRATUM_QUOTA);
  * within a stratum, a minimum share of Indonesian / mixed-language sentences
    (MIN_LOCAL_LANG_SHARE) so rules 7.13 / 7.14 are exercised;
  * a deliberate minority of low-relevance sentences (BORDERLINE_SHARE) so the
    NON_RELEVANT / UNCERTAIN categories are real options during calibration;
  * at most MAX_PER_POSTING sentences from any one posting, so the sample spans
    many postings rather than clustering in a few long ones;
  * exact-duplicate sentence text removed, so boilerplate lines are not repeated.
All randomness is seeded (SEED) for reproducibility. The main sample is drawn
disjoint from the pilot.

Run:  .venv-ml\\Scripts\\python.exe -m gold_standard.sample [--main]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from governance.ladder import REPO_ROOT
from . import (ANNOTATOR_LANGUAGES, BORDERLINE_SHARE, GOLD_STANDARD_VERSION,
               GUIDELINE_CODE, MAIN_N, MIN_LOCAL_LANG_SHARE, PILOT_N, SEED,
               STRATUM_QUOTA)

GS_DIR = REPO_ROOT / "DATA" / "gold_standard"
POOL_PATH = GS_DIR / "sentence_pool.parquet"

RELEVANCE_FLOOR = 1.0   # score >= floor counts as content-bearing
MAX_PER_POSTING = 2

# Columns carried into the delivered sample (provenance + context + annotation aids).
SAMPLE_COLUMNS = [
    "sentence_id", "posting_id", "batch_id", "stratum", "country_iso", "site",
    "posting_language", "is_indonesian_sentence", "title", "date_posted", "quarter",
    "sent_index", "n_sents_in_posting", "previous_sentence", "sentence_text",
    "next_sentence", "char_len", "length_band", "relevance_score",
]


def _cap_per_posting(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Keep at most `cap` sentences per posting (seeded, order-independent)."""
    return (df.sample(frac=1.0, random_state=seed)   # shuffle so the kept rows vary
              .groupby("posting_id", sort=False)
              .head(cap))


def _spread_sample(df: pd.DataFrame, k: int, seed: int,
                   strata=("site", "length_band")) -> pd.DataFrame:
    """Sample k rows spread proportionally across the given strata, then trimmed."""
    if k <= 0 or df.empty:
        return df.iloc[0:0]
    if len(df) <= k:
        return df
    usable = [c for c in strata if c in df.columns]
    if not usable:
        return df.sample(n=k, random_state=seed)
    total, parts = len(df), []
    for _, grp in df.groupby(usable, dropna=False):
        alloc = max(1, round(k * len(grp) / total))
        parts.append(grp.sample(n=min(alloc, len(grp)), random_state=seed))
    out = pd.concat(parts)
    if len(out) > k:
        out = out.sample(n=k, random_state=seed)
    return out


def _take_stratum(df: pd.DataFrame, k: int, local_frac: float, seed: int) -> pd.DataFrame:
    """Select k sentences from one stratum's candidate pool honoring the local-
    language minimum and the borderline minority, capped per posting."""
    df = _cap_per_posting(df, MAX_PER_POSTING, seed)
    content = df[df["relevance_score"] >= RELEVANCE_FLOOR]
    borderline = df[df["relevance_score"] < RELEVANCE_FLOOR]

    n_border = min(len(borderline), round(k * BORDERLINE_SHARE))
    n_content = k - n_border
    loc = content[content["is_indonesian_sentence"]]
    oth = content[~content["is_indonesian_sentence"]]
    n_loc = min(len(loc), round(n_content * local_frac))
    n_oth = n_content - n_loc

    picks = [
        _spread_sample(loc, n_loc, seed),
        _spread_sample(oth, n_oth, seed),
        _spread_sample(borderline, n_border, seed),
    ]
    out = pd.concat([p for p in picks if not p.empty]) if any(
        not p.empty for p in picks) else df.iloc[0:0]

    # Top up from whatever remains in the stratum if quotas under-filled it.
    if len(out) < k:
        remaining = df.drop(index=out.index, errors="ignore")
        out = pd.concat([out, _spread_sample(remaining, k - len(out), seed)])
    return out.head(k)


def draw_sample(n: int, exclude_ids: set[str] | None = None,
                seed: int = SEED) -> pd.DataFrame:
    pool = pd.read_parquet(POOL_PATH)
    pool = pool.sort_values("sentence_id").drop_duplicates("sentence_text", keep="first")
    # Restrict to languages the human annotators can read (English + Indonesian).
    pool = pool[pool["posting_language"].isin(ANNOTATOR_LANGUAGES)]
    if exclude_ids:
        pool = pool[~pool["posting_id"].isin(exclude_ids)]

    parts = []
    for stratum, frac in STRATUM_QUOTA.items():
        k = round(n * frac)
        cand = pool[pool["stratum"] == stratum]
        parts.append(_take_stratum(cand, k, MIN_LOCAL_LANG_SHARE.get(stratum, 0.0), seed))
    sample = pd.concat(parts, ignore_index=True)

    # Correct any rounding drift to land exactly on n.
    if len(sample) > n:
        sample = sample.sample(n=n, random_state=seed).reset_index(drop=True)
    elif len(sample) < n:
        extra = (pool[~pool["sentence_id"].isin(sample["sentence_id"])]
                 .pipe(_spread_sample, n - len(sample), seed, ("stratum", "length_band")))
        sample = pd.concat([sample, extra], ignore_index=True)

    sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sample.insert(0, "sample_order", range(1, len(sample) + 1))
    return sample[["sample_order", *SAMPLE_COLUMNS]]


def _write(sample: pd.DataFrame, split: str) -> None:
    GS_DIR.mkdir(parents=True, exist_ok=True)
    pq = GS_DIR / f"{split}.parquet"
    csv = GS_DIR / f"{split}.csv"
    sample.to_parquet(pq, index=False)
    sample.to_csv(csv, index=False, encoding="utf-8")

    pool_hash = hashlib.sha1(POOL_PATH.read_bytes()).hexdigest()[:16]
    manifest = {
        "split": split,
        "n": int(len(sample)),
        "seed": SEED,
        "guideline": GUIDELINE_CODE,
        "gold_standard_version": GOLD_STANDARD_VERSION,
        "annotator_languages": list(ANNOTATOR_LANGUAGES),
        "pool_file": POOL_PATH.name,
        "pool_sha1_16": pool_hash,
        "relevance_floor": RELEVANCE_FLOOR,
        "max_per_posting": MAX_PER_POSTING,
        "stratum_quota": STRATUM_QUOTA,
        "min_local_lang_share": MIN_LOCAL_LANG_SHARE,
        "borderline_share": BORDERLINE_SHARE,
        "by_stratum": sample["stratum"].value_counts().to_dict(),
        "by_posting_language": sample["posting_language"].value_counts().to_dict(),
        "indonesian_sentence_count": int(sample["is_indonesian_sentence"].sum()),
        "by_site": sample["site"].value_counts().to_dict(),
        "by_length_band": sample["length_band"].value_counts().to_dict(),
        "borderline_count": int((sample["relevance_score"] < RELEVANCE_FLOOR).sum()),
        "distinct_postings": int(sample["posting_id"].nunique()),
    }
    (GS_DIR / f"{split}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{split}: {len(sample)} sentences, {manifest['distinct_postings']} postings", flush=True)
    print(json.dumps({k: manifest[k] for k in (
        "by_stratum", "by_posting_language", "indonesian_sentence_count",
        "by_length_band", "borderline_count")}, indent=2, ensure_ascii=False), flush=True)
    print(f"-> {pq}\n-> {csv}\n-> {GS_DIR / (split + '_manifest.json')}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main", action="store_true",
                    help="also draw the main 1,200 sample, disjoint from the pilot")
    args = ap.parse_args()

    pilot = draw_sample(PILOT_N)
    _write(pilot, "pilot_100")

    if args.main:
        pilot_postings = set(pilot["posting_id"])
        main = draw_sample(MAIN_N, exclude_ids=pilot_postings)
        _write(main, "main_1200")


if __name__ == "__main__":
    main()
