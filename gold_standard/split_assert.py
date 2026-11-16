"""Write-time leakage invariants for the group-aware split (I1-I17).

Pure and IO-free: every function takes the in-memory assignment frame plus the
cached artifacts and returns `{"passed": bool, "observed": ...}`. `split.py`
calls `assert_invariants` BEFORE it creates any file, and a failure RAISES
`InvariantFailure` rather than warning, so a leaking split can never reach disk.

Dev is included in the disjointness invariants although PROP:1192 / PROP:1436
name only train vs test. That is a deliberately STRICTER choice than the text,
recorded in the manifest as `dev_disjointness_is_stricter_than_proposal: true`,
not as a quotation.

FAIL-CLOSED DISCIPLINE
----------------------
An invariant that cannot fail is not evidence, and the adversarial review of
2026-11-16 proved two of these could not fail at all:

  * **I7** (the only all-pairs leakage check) SKIPPED any posting absent from
    the shingle cache. Deleting the sampled postings' rows from
    `shingles.parquet` therefore produced `n_pairs_compared: 0,
    max_cross_part_jaccard: 0.0, passed: true`, and the split was written with
    the manifest advertising 0.0 as evidence of disjointness (findings F04,
    F14). Absence is now a FAILURE, an EMPTY signature is a FAILURE, zero
    compared pairs is a FAILURE, and the cache must carry a fingerprint that
    matches the signatures actually handed in.
  * **I16** (the invariant written specifically to stop "a target reported as
    achieved") was a key-PRESENCE check: it only asked whether a dict carrying
    `target` also carried `achieved`. A manifest whose `target` was copied from
    the achieved assignment sailed through it, and did (findings F07, F10, F15).
    It now compares VALUES and requires every `target` to trace to the
    REGISTERED config, so a target derived from the assignment is rejected
    whatever the producer does.

The same question - "can this pass vacuously on absent, empty or degenerate
input?" - was then put to every other invariant in this module. The audit, with
the fail-closed guard each one grew, is recorded in `VACUITY_AUDIT` at the
bottom of this module so it can be read, cited and tested rather than trusted.

HONEST SCOPE OF THE I7 MEASUREMENT
----------------------------------
I7 compares the signatures in the cache it is handed, and nothing else. It
cannot see past the window those signatures were built under: while the leakage
window was the first 2,500 cleaned characters, 69% of real pool postings were
longer than the window and 126 real template pairs at full-text Jaccard >=
TAU_BUILD were invisible to both the group map and this invariant (finding F01,
fixed in `textsig` by removing the window, not by raising it). I7 therefore
reads the window out of the cache's own declared parameters and publishes it as
`signature_window_chars` / `measures_full_text` / `signature_scope`, so a
prefix measurement can never again be reported as a statement about postings.
The earlier docstring claim that I7 "measures the quantity that actually
matters" is withdrawn: it measures the quantity the cache contains, and it now
refuses to pass unless the cache proves which quantity that is.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np
import pandas as pd

from . import (MAX_GROUP_SHARE, MIN_SHARED_SHINGLES, MIN_TEST_SENTENCES,
               SENT_NEAR_DUP_JACCARD, SHINGLE_WORDS, SPLIT_TARGETS,
               TAU_BUILD)
from .textsig import jaccard, overlap_and_jaccard, signature_fingerprint

# `textsig` grew a full cache-fingerprint API (window + cleaner + shingle width
# + posting-id set + per-signature digests, each digesting itself). Use it when
# it is there: it verifies the PARAMETERS a content digest alone cannot see, so
# a cache built under the old 2,500-character window can never be reused under
# the full-text window. Its absence never relaxes I7 - the content digest below
# is checked either way - so the guard only affects how much detail the
# mismatch message carries.
try:  # pragma: no cover - exercised by whichever textsig is installed
    from .textsig import LEAKAGE_WINDOW_CHARS as _LEAKAGE_WINDOW_CHARS
    from .textsig import verify_cache_fingerprint as _verify_cache_fingerprint
except ImportError:  # pragma: no cover
    _LEAKAGE_WINDOW_CHARS = "undeclared"
    _verify_cache_fingerprint = None

PARTS = ("train", "dev", "test")
ALL_PARTS = ("train", "dev", "test", "excluded")
KNOWN_REGIMES = ("grouped_random", "temporal", "cross_country")

# Columns a blind test artifact may carry. Anything else (a tag, a span, an
# adjudication verdict) means the split layer has touched the labels.
TEST_BLIND_ALLOWED_COLUMNS = frozenset({
    "sentence_id", "posting_id", "group_id", "part", "stratum", "country_iso",
    "site", "posting_language", "is_indonesian_sentence", "title", "date_posted",
    "quarter", "sent_index", "n_sents_in_posting", "previous_sentence",
    "sentence_text", "next_sentence", "char_len", "length_band",
    "relevance_score", "sample_order", "occupation_group", "country_language",
})
# A label-ish column NAME. The five tag codes are matched as whole words
# (`sh`, `SH_spans`) rather than as bare prefixes: `^sh` also matched
# `short_text` and `shingles`, which would refuse a legitimate write on a
# column that carries no label at all.
LABEL_LIKE = re.compile(
    r"(?i)^(label|labels|tag|tags|span|spans|annotation|annotations|adjudic|"
    r"gold|verdict|relevance$|has_5tag|(sh|ss|kn|tt|wb)(\W|_|$))")

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")

# A hash column is a STRING column, so the missing marker arrives as any of
# these. `""` is the documented one; the rest are what pandas and parquet
# round-trips produce. Treating "nan" as a VALUE would let every employerless
# posting share one hash (a false straddle); treating it as a hash of its own
# would hide a real one. Both are wrong, so they are normalised to missing and
# COUNTED, never silently dropped.
MISSING_STRINGS = frozenset({"", "nan", "none", "null", "na", "<na>", "nat",
                             "<nan>", "n/a"})
_TRUE_STRINGS = frozenset({"true", "t", "yes", "y", "1"})
_FALSE_STRINGS = frozenset({"false", "f", "no", "n", "0"})

# ---------------------------------------------------------------------------
# The shingle-cache fingerprint contract (I7).
#
# `split_assert` is pure: it cannot re-read the raw descriptions, so it cannot
# tell by itself whether the cache it was handed is the one the group map was
# built from. The GROUP LAYER must therefore declare it. The declaration is
# looked for under any of the field names below, in `split.manifest.provenance`
# first and then in `split.manifest.group_layer` (which is
# `leakage_groups_manifest.json` verbatim), and its value must equal
# `shingle_cache_fingerprint(shingles)` computed over the cache as loaded.
# ---------------------------------------------------------------------------
SHINGLE_FINGERPRINT_FIELDS = ("shingles_fingerprint", "shingle_cache_fingerprint",
                              "shingles_cache_fingerprint")
SHINGLE_SOURCE_FIELDS = ("shingles_source", "shingle_source")
# The full fingerprint BLOCK (`textsig.cache_fingerprint`), when the producer
# records one: it verifies the window, the cleaner and the shingle width too.
SHINGLE_BLOCK_FIELDS = ("signature_cache", "shingle_cache", "cache_fingerprint_block")
SHINGLE_FINGERPRINT_ALGORITHM = (
    "sha256 over '<posting_id>\\t<textsig.signature_fingerprint(signature)>\\n' "
    "lines, posting ids sorted ascending; see "
    "gold_standard.split_assert.shingle_cache_fingerprint")

REQUIRED_QUARANTINE_KEYS = ("sentence_ids", "posting_ids", "group_ids")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class InvariantFailure(RuntimeError):
    """Raised instead of writing a split that violates a leakage invariant."""


# ----------------------------------------------------------------- primitives
def _norm_sentence(text: object) -> str:
    t = unicodedata.normalize("NFKC", str(text)).casefold()
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", t)).strip()


def _clean(value: object) -> str:
    """A string column's value, with every missing marker mapped to ""."""
    if value is None:
        return ""
    try:
        if not isinstance(value, str) and pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    return "" if s.casefold() in MISSING_STRINGS else s


def _clean_series(series: pd.Series) -> pd.Series:
    return series.map(_clean)


def _strict_bool(values) -> tuple[np.ndarray, list[str]]:
    """Booleans without `astype(bool)`'s lies.

    `pd.Series.astype(bool)` maps NaN and the STRING "False" to True, which made
    I9 pass on both (a latent hazard the review flagged). Here a missing value
    is False and is reported, and an uninterpretable value is False and is
    reported, so neither can be read as verified.
    """
    out, bad = [], []
    for v in values:
        if isinstance(v, (bool, np.bool_)):
            out.append(bool(v))
            continue
        if isinstance(v, (int, np.integer)) and int(v) in (0, 1):
            out.append(bool(int(v)))
            continue
        s = _clean(v)
        if s == "":
            out.append(False)
            bad.append(repr(v))
            continue
        low = s.casefold()
        if low in _TRUE_STRINGS:
            out.append(True)
        elif low in _FALSE_STRINGS:
            out.append(False)
        else:
            out.append(False)
            bad.append(repr(v))
    return np.array(out, dtype=bool), bad


def _missing_columns(df: pd.DataFrame, columns) -> list[str]:
    return [c for c in columns if c not in getattr(df, "columns", ())]


def _assigned(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["part"].isin(PARTS)]


def _part_sizes(df: pd.DataFrame) -> dict[str, int]:
    return {p: int((df["part"] == p).sum()) for p in PARTS}


def _by_part(df: pd.DataFrame, column: str) -> dict[str, set]:
    """Non-missing values of `column` per part.

    Missing values are EXCLUDED here and counted by the caller: a null id is a
    broken join, not a value, and silently dropping it is how a disjointness
    check goes vacuous (`i2` passed on a frame whose every `group_id` was NaN).
    Every caller that uses this on an id column also calls `_id_problems`.
    """
    out = {}
    for p in PARTS:
        vals = _clean_series(df.loc[df["part"] == p, column])
        out[p] = set(vals[vals != ""].tolist())
    return out


def _id_problems(df: pd.DataFrame, column: str) -> dict:
    """Null / blank ids among the ASSIGNED rows, which make a check vacuous."""
    assigned = _assigned(df)
    vals = _clean_series(assigned[column])
    bad = assigned.loc[(vals == "").values, "sentence_id"] if "sentence_id" in assigned \
        else assigned.loc[(vals == "").values].index.to_series()
    return {"n_missing": int((vals == "").sum()),
            "examples": [str(v) for v in list(bad)[:5]]}


def _pairwise_overlaps(sets: dict[str, set]) -> list[dict]:
    out = []
    names = list(PARTS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = sets[a] & sets[b]
            if shared:
                out.append({"parts": [a, b], "n_shared": len(shared),
                            "examples": sorted(map(str, shared))[:5]})
    return out


def _comparable_parts(sets: dict[str, set]) -> int:
    return sum(1 for p in PARTS if sets.get(p))


def _ok(passed: bool, **observed) -> dict:
    return {"passed": bool(passed), **observed}


def shingle_cache_fingerprint(shingles: dict) -> str:
    """Stable digest of a whole shingle cache, for cache-staleness detection.

    This is the value the GROUP LAYER must record (see
    `SHINGLE_FINGERPRINT_FIELDS`) and the value I7 recomputes from the cache it
    is handed. It covers both halves of the staleness question: which postings
    the cache claims to describe, and what each signature actually contains. A
    sidecar with the right posting ids and garbage signatures - the attack that
    silently disabled the template half of the group map and left every
    invariant green - changes this digest.
    """
    h = hashlib.sha256()
    for pid in sorted(map(str, shingles)):
        arr = np.asarray(shingles[pid], dtype=np.uint64)
        h.update(f"{pid}\t{signature_fingerprint(arr)}\n".encode("utf-8"))
    return h.hexdigest()


def _first_field(meta: dict, fields) -> tuple[str | None, object]:
    for field in fields:
        if isinstance(meta, dict) and meta.get(field) not in (None, ""):
            return field, meta[field]
    return None, None


def resolve_shingle_meta(provenance: dict | None, group_layer: dict | None) -> dict:
    """The group layer's declaration about the shingle cache, from either place.

    `split.py` already copies `leakage_groups_manifest.json` into
    `manifest.group_layer` verbatim, so the group layer can satisfy the I7
    fingerprint contract without `split.py` changing at all; a field in
    `manifest.provenance` wins if both are present.
    """
    meta: dict = {}
    for source_name, source in (("group_layer", group_layer),
                                ("provenance", provenance)):
        if not isinstance(source, dict):
            continue
        field, value = _first_field(source, SHINGLE_FINGERPRINT_FIELDS)
        if field:
            meta["fingerprint"] = str(value)
            meta["fingerprint_field"] = f"{source_name}.{field}"
        field, value = _first_field(source, SHINGLE_SOURCE_FIELDS)
        if field:
            meta["source"] = str(value)
            meta["source_field"] = f"{source_name}.{field}"
        field, value = _first_field(source, SHINGLE_BLOCK_FIELDS)
        if field and isinstance(value, dict):
            meta["block"] = value
            meta["block_field"] = f"{source_name}.{field}"
        for key in ("shingle_words", "hash_fn", "text_pipeline"):
            if source.get(key) is not None:
                meta[key] = source[key]
        # The leakage WINDOW. `None` is a real value (the whole cleaned text),
        # so presence of the key matters, not truthiness.
        params = source.get("signature_params")
        if isinstance(params, dict) and "window_chars" in params:
            meta["window_chars"] = params["window_chars"]
        elif "window_chars" in source:
            meta["window_chars"] = source["window_chars"]
        elif "shingle_max_chars" in source:
            meta["window_chars"] = source["shingle_max_chars"]
    block = meta.get("block")
    if isinstance(block, dict):
        params = block.get("signature_params")
        if isinstance(params, dict) and "window_chars" in params:
            meta["window_chars"] = params["window_chars"]
        if block.get("content_sha256") and not meta.get("fingerprint"):
            meta["fingerprint"] = str(block["content_sha256"])
            meta["fingerprint_field"] = f"{meta.get('block_field')}.content_sha256"
    return meta


# ------------------------------------------------------------------ invariants
def i1_accounting(df: pd.DataFrame, input_ids: list[str],
                  allowed_reasons: tuple[str, ...]) -> dict:
    """Every input sentence lands in exactly one part, with a legal reason.

    Fail-closed: an EMPTY frame or an empty `input_ids` list used to satisfy
    "the union equals the input exactly" trivially, so both are failures - a
    split of nothing is not a split.
    """
    missing_cols = _missing_columns(df, ("sentence_id", "part", "exclusion_reason"))
    if missing_cols:
        return _ok(False, reason="missing columns", missing_columns=missing_cols)
    if len(df) == 0 or not len(input_ids):
        return _ok(False, reason="empty assignment frame or empty input id list: "
                                 "the accounting identity would hold vacuously",
                   n_rows=int(len(df)), n_input=len(input_ids))
    sid = _clean_series(df["sentence_id"])
    n_blank_ids = int((sid == "").sum())
    parts = set(df["part"].astype(str).unique())
    bad_parts = sorted(parts - set(ALL_PARTS))
    dupes = df["sentence_id"][df["sentence_id"].duplicated()].tolist()
    missing = sorted(set(map(str, input_ids)) - set(sid))
    extra = sorted(set(sid) - set(map(str, input_ids)) - {""})
    exc = df[df["part"] == "excluded"]
    bad_reason = sorted(set(_clean_series(exc["exclusion_reason"])) - set(allowed_reasons))
    assigned = _assigned(df)
    stray = int((_clean_series(assigned["exclusion_reason"]) != "").sum())
    sizes = _part_sizes(df)
    empty_parts = [p for p in PARTS if sizes[p] == 0]
    return _ok(not bad_parts and not dupes and not missing and not extra
               and not bad_reason and stray == 0 and n_blank_ids == 0
               and not empty_parts,
               n_rows=int(len(df)), n_input=len(input_ids),
               unknown_parts=bad_parts, duplicate_sentence_ids=dupes[:5],
               missing_sentence_ids=missing[:5], unexpected_sentence_ids=extra[:5],
               blank_sentence_ids=n_blank_ids,
               unknown_exclusion_reasons=bad_reason,
               assigned_rows_carrying_a_reason=stray,
               empty_parts=empty_parts,
               by_part={p: int((df["part"] == p).sum()) for p in ALL_PARTS})


def _disjointness(df: pd.DataFrame, column: str, *, require_ids: bool = True) -> dict:
    missing_cols = _missing_columns(df, ("part", column))
    if missing_cols:
        return _ok(False, reason="missing columns", missing_columns=missing_cols)
    sets = _by_part(df, column)
    ov = _pairwise_overlaps(sets)
    ids = _id_problems(df, column) if require_ids else {"n_missing": 0, "examples": []}
    comparable = _comparable_parts(sets)
    return _ok(not ov and ids["n_missing"] == 0 and comparable >= 2,
               column=column, overlaps=ov,
               n_missing_values=ids["n_missing"],
               missing_value_examples=ids["examples"],
               n_distinct_by_part={p: len(sets[p]) for p in PARTS},
               n_parts_with_values=comparable,
               vacuity_guard=("fewer than two parts carry a value: disjointness "
                              "would hold trivially" if comparable < 2 else None))


def i2_group_disjoint(df: pd.DataFrame) -> dict:
    """Primary invariant. A null `group_id` is a BROKEN JOIN, never a value:
    `.dropna()` used to swallow it, so a frame whose every group_id was NaN
    passed. Missing ids now fail."""
    return _disjointness(df, "group_id")


def i3_posting_disjoint(df: pd.DataFrame) -> dict:
    return _disjointness(df, "posting_id")


def _hash_coverage(df: pd.DataFrame, column: str) -> dict:
    assigned = _assigned(df)
    vals = _clean_series(assigned[column])
    per_part = {}
    for p in PARTS:
        v = _clean_series(df.loc[df["part"] == p, column])
        per_part[p] = {"n_rows": int(len(v)), "n_missing": int((v == "").sum())}
    return {"n_assigned": int(len(vals)), "n_nonempty": int((vals != "").sum()),
            "by_part": per_part}


def i4_employer_exact_disjoint(df: pd.DataFrame) -> dict:
    """Employer-exact disjointness, with the coverage it is claimed over.

    Two fail-closed guards. (1) `""`, None and every parquet missing marker are
    normalised to missing: `astype(str) != ""` let NaN through as the literal
    value "nan", which both fires falsely and hides a real hash. (2) If NO
    assigned row carries a hash the claim is vacuous, so it fails. Per-posting
    missing hashes are legitimate (the contract confines those postings to
    train and I9 proves dev/test are clean), so they are COUNTED here, not
    failed - the count is what the overclaim "employer-disjoint" needs beside it.
    """
    missing_cols = _missing_columns(df, ("part", "employer_hash"))
    if missing_cols:
        return _ok(False, reason="missing columns", missing_columns=missing_cols)
    sets = _by_part(df, "employer_hash")
    ov = _pairwise_overlaps(sets)
    cov = _hash_coverage(df, "employer_hash")
    comparable = _comparable_parts(sets)
    dead = cov["n_nonempty"] == 0
    return _ok(not ov and not dead and comparable >= 2,
               overlaps=ov, coverage=cov,
               n_parts_with_values=comparable,
               vacuity_guard=("no assigned row carries an employer_hash: the "
                              "disjointness claim is vacuous" if dead else
                              ("fewer than two parts carry an employer_hash"
                               if comparable < 2 else None)))


def i5_employer_loose_disjoint(df: pd.DataFrame) -> dict:
    """Employer-loose disjointness.

    The loose hash is legitimately empty for a company whose whole name is legal
    form, and (finding F03, owned by posting_keys) for every residual under 3
    characters - 230 real pool postings. Those empties are therefore counted,
    not failed; `loose_coverage_share` is published so "loose atomicity across
    every boundary" can never again be claimed over a column that is mostly
    empty. A column with NO non-empty value at all is a dead relation and fails.
    """
    missing_cols = _missing_columns(df, ("part", "employer_hash_loose"))
    if missing_cols:
        return _ok(False, reason="missing columns", missing_columns=missing_cols)
    sets = _by_part(df, "employer_hash_loose")
    ov = _pairwise_overlaps(sets)
    cov = _hash_coverage(df, "employer_hash_loose")
    comparable = _comparable_parts(sets)
    dead = cov["n_nonempty"] == 0
    share = (cov["n_nonempty"] / cov["n_assigned"]) if cov["n_assigned"] else 0.0
    return _ok(not ov and not dead and comparable >= 2,
               overlaps=ov, coverage=cov,
               loose_coverage_share=round(share, 6),
               n_parts_with_values=comparable,
               vacuity_guard=("no assigned row carries an employer_hash_loose: "
                              "the loose relation is switched off" if dead else
                              ("fewer than two parts carry a loose hash"
                               if comparable < 2 else None)))


def i6_template_cluster_disjoint(df: pd.DataFrame,
                                 column: str = "template_cluster_t030") -> dict:
    res = _disjointness(df, column)
    res["note"] = ("t050/t070 are logically implied by monotonicity and are "
                   "reported in the manifest, never asserted")
    return res


def i7_direct_pairwise_jaccard(df: pd.DataFrame, shingles: dict,
                               tau: float = TAU_BUILD, *,
                               shingle_meta: dict | None = None,
                               require_fingerprint: bool = True) -> dict:
    """Re-derive the maximum DIRECT cross-part template similarity.

    Independent of the union-find on purpose: "no cluster straddles" is not the
    same statement as "no similar pair straddles" (a component can contain
    postings whose mutual Jaccard is far below tau, and a two-hop bridge can run
    through a posting that is not in the pool at all).

    FAIL-CLOSED. This invariant used to `continue` past any posting the cache did
    not cover and past any empty signature, and it reported `passed: True` with
    `n_pairs_compared: 0`. Deleting the sampled postings' rows from
    `shingles.parquet` therefore turned the architecture's main guarantee into a
    pass, with `max_cross_part_jaccard: 0.0` published as evidence of
    disjointness. Now:

      * a sampled posting ABSENT from the cache fails (incomplete cache);
      * a sampled posting with an EMPTY signature fails (it cannot be compared,
        and an all-empty cache is exactly what the stale-sidecar attack plants);
      * zero compared pairs fails, however it arose;
      * the cache must carry a fingerprint matching the signatures handed in,
        so a cache swapped out after the group map was built is caught here
        rather than trusted;
      * a cache built under a FINITE character window fails, because a Jaccard
        over prefixes is not the leakage claim (finding F01) - the window the
        signatures were built under is read from the cache's own declared
        parameters, not assumed.

    What it does NOT measure: whatever the cache leaves out. The scope is
    reported in `signature_scope` / `measures_full_text` / `signature_window_
    chars` and must be carried into any claim made from
    `max_cross_part_jaccard`.
    """
    missing_cols = _missing_columns(df, ("part", "posting_id"))
    if missing_cols:
        return _ok(False, reason="missing columns", missing_columns=missing_cols)
    shingles = shingles or {}
    meta = dict(shingle_meta or {})

    per_part, blank_ids = {}, 0
    for p in PARTS:
        vals = _clean_series(df.loc[df["part"] == p, "posting_id"])
        blank_ids += int((vals == "").sum())
        per_part[p] = sorted(set(vals[vals != ""].tolist()))

    cache = {str(k): np.asarray(v, dtype=np.uint64) for k, v in shingles.items()}
    sampled = sorted({pid for p in PARTS for pid in per_part[p]})
    absent = [pid for pid in sampled if pid not in cache]
    empty = [pid for pid in sampled if pid in cache and cache[pid].size == 0]
    usable = {pid for pid in sampled if pid in cache and cache[pid].size > 0}

    worst, worst_pair, n_pairs, violations = 0.0, None, 0, []
    worst_shared, worst_shared_pair = 0, None
    names = list(PARTS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for pa in per_part[a]:
                if pa not in usable:
                    continue
                sa = cache[pa]
                for pb in per_part[b]:
                    if pb not in usable:
                        continue
                    n_pairs += 1
                    # The grouping rule is "ratio OR absolute floor", so the
                    # invariant must test BOTH. Asserting only the ratio would
                    # let exactly the pairs the floor was added for straddle a
                    # boundary while this gate reported green. One intersection
                    # pass yields both numbers.
                    n_shared, j = overlap_and_jaccard(sa, cache[pb])
                    if j > worst:
                        worst, worst_pair = j, (pa, pb, a, b)
                    if n_shared > worst_shared:
                        worst_shared, worst_shared_pair = n_shared, (pa, pb, a, b)
                    if j >= tau or n_shared >= MIN_SHARED_SHINGLES:
                        violations.append({"a": pa, "b": pb, "parts": [a, b],
                                           "jaccard": round(j, 4),
                                           "n_shared": n_shared,
                                           "rule": ("ratio" if j >= tau
                                                    else "overlap_floor")})

    computed_fp = shingle_cache_fingerprint(cache)
    declared_fp = meta.get("fingerprint")
    if declared_fp is None:
        fp_status = "undeclared"
    elif str(declared_fp) == computed_fp:
        fp_status = "verified"
    else:
        fp_status = "mismatch"
    fp_ok = fp_status == "verified" or (fp_status == "undeclared"
                                        and not require_fingerprint)
    # The stronger check, when the producer records the whole block: it also
    # verifies the window, the cleaner and the shingle width the signatures were
    # built under, which a content digest cannot see.
    block_problems: list[str] = []
    block = meta.get("block")
    if isinstance(block, dict) and _verify_cache_fingerprint is not None:
        try:
            block_problems = list(_verify_cache_fingerprint(block, cache))
        except Exception as exc:  # noqa: BLE001 - a broken block is a mismatch
            block_problems = [f"the declared fingerprint block could not be "
                              f"verified: {type(exc).__name__}: {exc}"]

    window = meta.get("window_chars", _LEAKAGE_WINDOW_CHARS)
    full_text = window in (None, 0) or (
        isinstance(window, (int, float)) and window >= 10 ** 8)
    window_declared = "window_chars" in meta
    comparable_parts = sum(1 for p in PARTS if per_part[p])
    vacuous = n_pairs == 0

    reasons = []
    if absent:
        reasons.append(f"{len(absent)} sampled posting(s) are ABSENT from the "
                       f"shingle cache, so the only all-pairs leakage check "
                       f"cannot see them")
    if empty:
        reasons.append(f"{len(empty)} sampled posting(s) have an EMPTY signature "
                       f"in the cache, so they are uncomparable")
    if blank_ids:
        reasons.append(f"{blank_ids} assigned row(s) carry a blank posting_id")
    if vacuous:
        reasons.append("no cross-part pair was compared at all: a reported "
                       "max_cross_part_jaccard of 0.0 would be meaningless")
    if comparable_parts < 2:
        reasons.append("fewer than two parts hold a posting")
    if fp_status == "mismatch":
        reasons.append(
            f"shingle cache fingerprint MISMATCH: {meta.get('fingerprint_field')} "
            f"declares {declared_fp} but the cache handed in digests to "
            f"{computed_fp}. The cache is not the one the group map was built "
            f"from, so neither the group map nor this check can be trusted")
    if window_declared and not full_text:
        reasons.append(
            f"the signature cache was built under a FINITE window "
            f"({window} characters), so this is a Jaccard over prefixes: 69% of "
            f"real pool postings are longer than 2,500 cleaned characters and "
            f"126 real template pairs at full-text Jaccard >= TAU_BUILD were "
            f"invisible to it (finding F01). textsig.LEAKAGE_WINDOW_CHARS is "
            f"{_LEAKAGE_WINDOW_CHARS!r}; rebuild the cache under that window "
            f"rather than publishing a prefix measurement as a leakage claim")
    if block_problems:
        reasons.append(
            "the declared signature-cache fingerprint block does not describe "
            f"the cache handed in: {block_problems[:5]}")
    if fp_status == "undeclared" and require_fingerprint:
        reasons.append(
            "no shingle cache fingerprint is declared. The group layer must "
            f"record one of {list(SHINGLE_FINGERPRINT_FIELDS)} in "
            "leakage_groups_manifest.json (copied verbatim into "
            "split.manifest.group_layer) or in split.manifest.provenance, "
            f"computed as {SHINGLE_FINGERPRINT_ALGORITHM}. Until it does, an "
            "un-fingerprinted cache cannot be distinguished from a stale one "
            f"and this invariant refuses to pass. Expected value for the cache "
            f"handed in: {computed_fp}")

    return _ok(not violations and not reasons and fp_ok,
               tau=tau, n_pairs_compared=n_pairs,
               max_cross_part_jaccard=round(worst, 4),
               max_pair=list(worst_pair) if worst_pair else None,
               min_shared_shingles=int(MIN_SHARED_SHINGLES),
               max_cross_part_n_shared=worst_shared,
               max_shared_pair=list(worst_shared_pair) if worst_shared_pair else None,
               violations=violations[:10],
               n_sampled_postings=len(sampled),
               n_cache_rows=len(cache),
               n_postings_without_signature=len(absent) + len(empty),
               postings_absent_from_cache=absent[:10],
               postings_with_empty_signature=empty[:10],
               blank_posting_ids=blank_ids,
               cache_fingerprint_status=fp_status,
               cache_fingerprint_computed=computed_fp,
               cache_fingerprint_declared=declared_fp,
               cache_fingerprint_field=meta.get("fingerprint_field"),
               cache_fingerprint_algorithm=SHINGLE_FINGERPRINT_ALGORITHM,
               cache_fingerprint_block_problems=block_problems[:5],
               cache_fingerprint_block_field=meta.get("block_field"),
               shingle_source=meta.get("source"),
               signature_window_chars=window,
               signature_window_declared=window_declared,
               shingle_words=meta.get("shingle_words", SHINGLE_WORDS),
               measures_full_text=bool(full_text),
               signature_scope=("full cleaned text" if full_text else
                                f"first {window} characters of cleaned text — "
                                f"max_cross_part_jaccard is a PREFIX measurement "
                                f"and bounds nothing past that cut"),
               failure_reasons=reasons)


def i8_sentence_text_disjoint(df: pd.DataFrame,
                              threshold: float = SENT_NEAR_DUP_JACCARD) -> dict:
    """Exact, normalised and near-duplicate sentence text across parts.

    The one leakage channel neither employer nor template grouping sees:
    identical boilerplate requirement lines shared by unrelated employers.
    `sample.py` already drops exact duplicates pool-wide, so the normalised and
    near-duplicate forms are what can still bite.

    Fail-closed: a blank or missing `sentence_text` on an assigned row fails (it
    cannot be compared, and `str(NaN)` would otherwise be compared as the token
    "nan"), and zero compared pairs fails.
    """
    missing_cols = _missing_columns(df, ("part", "sentence_id", "sentence_text"))
    if missing_cols:
        return _ok(False, reason="missing columns", missing_columns=missing_cols)
    assigned = _assigned(df).copy()
    assigned["_text"] = _clean_series(assigned["sentence_text"])
    blank = assigned.loc[assigned["_text"] == "", "sentence_id"].tolist()

    exact = _pairwise_overlaps(_by_part(assigned, "_text"))
    norm = assigned.assign(_n=[_norm_sentence(t) for t in assigned["_text"]])
    norm_ov = _pairwise_overlaps(_by_part(norm, "_n"))

    toks = {p: [(sid, set(str(t).casefold().split()))
                for sid, t in zip(assigned.loc[assigned["part"] == p, "sentence_id"],
                                  assigned.loc[assigned["part"] == p, "_text"])
                if str(t).strip()]
            for p in PARTS}
    near, worst, n_pairs = [], 0.0, 0
    names = list(PARTS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for sid_a, ta in toks[a]:
                if not ta:
                    continue
                for sid_b, tb in toks[b]:
                    if not tb:
                        continue
                    n_pairs += 1
                    j = len(ta & tb) / len(ta | tb)
                    worst = max(worst, j)
                    if j >= threshold:
                        near.append({"a": sid_a, "b": sid_b, "parts": [a, b],
                                     "jaccard": round(j, 4)})
    return _ok(not exact and not norm_ov and not near and not blank and n_pairs > 0,
               threshold=threshold, exact_overlaps=exact,
               normalised_overlaps=norm_ov,
               n_near_duplicate_pairs=len(near), near_examples=near[:10],
               n_pairs_compared=n_pairs,
               n_blank_sentence_texts=len(blank),
               blank_examples=[str(s) for s in blank[:5]],
               max_cross_part_sentence_jaccard=round(worst, 4),
               vacuity_guard=("no cross-part sentence pair was compared"
                              if n_pairs == 0 else None))


def i9_dev_test_verifiable(df: pd.DataFrame) -> dict:
    """Every dev and test posting's employer is verifiable.

    Fail-closed on three counts the old version missed: an EMPTY dev/test makes
    the share meaningless (it returned `passed: True, share: None`); `employer_
    known` is coerced strictly, because `astype(bool)` read both NaN and the
    string "False" as True; and the contract's actual words - "every dev and
    test posting has a NON-EMPTY employer_hash" - are now checked against the
    hash column itself rather than only against the boolean beside it.
    """
    missing_cols = _missing_columns(df, ("part", "sentence_id", "employer_known"))
    if missing_cols:
        return _ok(False, reason="missing columns", missing_columns=missing_cols)
    sub = df[df["part"].isin(("dev", "test"))]
    if sub.empty:
        return _ok(False, reason="dev and test are both EMPTY: a verifiability "
                                 "share of 1.0 over no rows is vacuous",
                   share=None, n=0)
    known, non_boolean = _strict_bool(sub["employer_known"].tolist())
    if "employer_hash" in sub.columns:
        hashed = (_clean_series(sub["employer_hash"]) != "").to_numpy()
    else:
        hashed = np.ones(len(sub), dtype=bool)
    verifiable = known & hashed
    share = float(verifiable.mean())
    offenders = sub.loc[~verifiable, "sentence_id"].tolist()
    missing_hash = sub.loc[~hashed, "sentence_id"].tolist()
    return _ok(share == 1.0 and not non_boolean,
               dev_test_employer_verifiable_share=round(share, 6),
               n=int(len(sub)), n_unverifiable=len(offenders),
               examples=[str(s) for s in offenders[:5]],
               n_dev_test_rows_without_employer_hash=len(missing_hash),
               rows_without_hash_examples=[str(s) for s in missing_hash[:5]],
               non_boolean_employer_known_values=sorted(set(non_boolean))[:5])


def i10_quarantine(df: pd.DataFrame, pilot: dict, practice: dict, *,
                   require_nonempty: bool = True) -> dict:
    """No pilot_100 / practice_10 sentence, posting or group in any part.

    Fail-closed on the reference sets themselves. `split.quarantine_sets`
    returns three EMPTY sets when `pilot_100.parquet` is missing, and this
    invariant then compared the assignment against nothing and passed - the
    quarantine claim is only as strong as the id lists it is given, so an empty
    reference is a failure, not a clean bill of health.

    SCOPE: "any part" means the three ASSIGNED parts. Contract §6.5.1 forces a
    group containing a pilot/practice posting to `excluded` with reason
    `pilot_group` / `practice_group`, so scanning the excluded bucket too made
    that mandated forcing step fail this invariant by construction - the
    exclusion reason was dead code and the step was unreachable end to end. The
    contract's I10 wording ("in any part of any regime") and its §6.5.1 forcing
    rule contradict each other if `excluded` counts as a part; §6.5.1 wins,
    because a quarantined sentence that is excluded has left the benchmark.
    """
    problems, hits = [], {}
    assigned = df[df["part"].astype(str).isin(PARTS)] if "part" in df.columns else df
    for name, ref in (("pilot_100", pilot), ("practice_10", practice)):
        if not isinstance(ref, dict):
            problems.append(f"{name}: quarantine reference is not a mapping")
            continue
        absent_keys = [k for k in REQUIRED_QUARANTINE_KEYS if k not in ref]
        if absent_keys:
            problems.append(f"{name}: quarantine reference is missing "
                            f"{absent_keys}")
        if require_nonempty and not any(ref.get(k) for k in REQUIRED_QUARANTINE_KEYS):
            problems.append(
                f"{name}: every quarantine id set is EMPTY, so the quarantine "
                f"claim is vacuous. This is what a missing {name}.parquet looks "
                f"like from in here, because split.quarantine_sets returns three "
                f"empty sets for an absent file. Fix it upstream: draw "
                f"{name} into the gold-standard directory before splitting "
                f"(contract §11 run order), or make quarantine_sets REFUSE a "
                f"gs_dir that has no {name}.parquet. Passing an empty reference "
                f"is not evidence that the quarantine held")
        for key, values in ref.items():
            column = {"sentence_ids": "sentence_id", "posting_ids": "posting_id",
                      "group_ids": "group_id"}.get(key)
            if column is None or column not in assigned.columns:
                continue
            bad = sorted(set(_clean_series(assigned[column])) & set(map(str, values)))
            if bad:
                hits[f"{name}.{key}"] = bad[:5]
    return _ok(not hits and not problems, hits=hits, reference_problems=problems,
               scope="train/dev/test only; the `excluded` bucket is where "
                     "contract §6.5.1 PUTS a quarantined group",
               n_assigned_rows_scanned=int(len(assigned)),
               n_rows_outside_the_assigned_parts=int(len(df) - len(assigned)),
               reference_sizes={
                   name: {k: len(ref.get(k, ())) for k in REQUIRED_QUARANTINE_KEYS}
                   for name, ref in (("pilot_100", pilot), ("practice_10", practice))
                   if isinstance(ref, dict)})


def i11_oversize_groups(df: pd.DataFrame, targets: dict) -> dict:
    """No group in dev or test exceeds MAX_GROUP_SHARE of that part's quota.

    Vacuous at weight <= 2 by design (the contract keeps it so a future looser
    cap fails loudly), but it was vacuous for a second, undesigned reason: a
    missing or zero quota made `if quota and ...` skip the part entirely. A
    quota of 0 is now a failure, and the limit is evaluated against the larger
    of the quota and the part's realised size so a shrunken regime capacity
    cannot quietly relax it.
    """
    missing_cols = _missing_columns(df, ("part", "group_id"))
    if missing_cols:
        return _ok(False, reason="missing columns", missing_columns=missing_cols)
    offenders, bad_quota, limits = [], [], {}
    sizes = _part_sizes(df)
    for part in ("dev", "test"):
        quota = int((targets or {}).get(part, 0) or 0)
        if quota <= 0:
            bad_quota.append(part)
        basis = max(quota, sizes[part])
        limit = MAX_GROUP_SHARE * basis
        limits[part] = {"quota": quota, "achieved": sizes[part],
                        "basis": basis, "limit": limit}
        counts = _clean_series(df.loc[df["part"] == part, "group_id"]).value_counts()
        for gid, n in counts.items():
            if basis and n > limit:
                offenders.append({"part": part, "group_id": gid,
                                  "n_sentences": int(n), "limit": limit})
    return _ok(not offenders and not bad_quota, max_group_share=MAX_GROUP_SHARE,
               offenders=offenders, parts_without_a_quota=bad_quota, limits=limits)


def i12_exact_sizes(df: pd.DataFrame, targets: dict, exact: bool, *,
                    registered_targets: dict | None = None) -> dict:
    """Achieved sizes against the allocator's capacity, and against the register.

    Fail-closed additions: an EMPTY part fails in either mode (no regime can
    claim a split with an empty train, dev or test), and in exact mode a
    targets object that sums to 0 fails instead of making "all deltas are zero"
    true by arithmetic. The registered numbers are reported next to the achieved
    ones in both modes, because the soft regimes' capacity is NOT a target - the
    gap between capacity and register is I16's business, and this is where the
    numbers to compare are visible.
    """
    missing_cols = _missing_columns(df, ("part",))
    if missing_cols:
        return _ok(False, reason="missing columns", missing_columns=missing_cols)
    achieved = _part_sizes(df)
    empty_parts = [p for p in PARTS if achieved[p] == 0]
    reg = {p: int((registered_targets or {}).get(p, 0)) for p in PARTS} \
        if registered_targets else None
    reg_delta = {p: achieved[p] - reg[p] for p in PARTS} if reg else None
    if exact:
        tgt = {p: int((targets or {}).get(p, 0) or 0) for p in PARTS}
        deltas = {p: achieved[p] - tgt[p] for p in PARTS}
        return _ok(all(v == 0 for v in deltas.values()) and sum(tgt.values()) > 0
                   and not empty_parts,
                   exact=True, targets=tgt, achieved=achieved, delta=deltas,
                   empty_parts=empty_parts,
                   registered_targets=reg, registered_delta=reg_delta,
                   capacity_sums_to_zero=sum(tgt.values()) == 0)
    underpowered = achieved["test"] < MIN_TEST_SENTENCES
    return _ok(not empty_parts, exact=False, achieved=achieved,
               min_test_sentences=MIN_TEST_SENTENCES, underpowered=underpowered,
               empty_parts=empty_parts,
               registered_targets=reg, registered_delta=reg_delta,
               note="regime-determined sizes: a date cut and a country boundary "
                    "cannot be bent to a target, so no exactness is claimed. An "
                    "EMPTY part is still a failure, and the shortfall against "
                    "the registered targets is reported, never absorbed")


def i13_determinism(df: pd.DataFrame) -> dict:
    """Record the assignment digest - and refuse to digest a broken frame.

    This used to return `passed: True` unconditionally, i.e. it was a recorder
    wearing an invariant's name. A digest over an empty frame, over duplicated
    sentence ids or over null triples is a stable hash of nothing useful, so
    each of those now fails.
    """
    missing_cols = _missing_columns(df, ("sentence_id", "part", "group_id"))
    if missing_cols:
        return _ok(False, reason="missing columns", missing_columns=missing_cols)
    sid = _clean_series(df["sentence_id"])
    part = _clean_series(df["part"])
    gid = _clean_series(df["group_id"])
    n_blank = int(((sid == "") | (part == "")).sum())
    # An excluded row legitimately carries no group when the group map dropped
    # it; a row in a PART must have one.
    assigned_mask = df["part"].isin(PARTS).to_numpy()
    n_blank_group = int(((gid == "") & assigned_mask).sum())
    n_dupes = int(sid.duplicated().sum())
    rows = sorted(f"{a}\t{b}\t{c}" for a, b, c in zip(sid, part, gid))
    return _ok(len(df) > 0 and n_blank == 0 and n_blank_group == 0 and n_dupes == 0,
               n_rows=int(len(df)), n_blank_fields=n_blank,
               n_assigned_rows_without_group=n_blank_group,
               n_duplicate_sentence_ids=n_dupes,
               assignment_sha256=hashlib.sha256(
                   "\n".join(rows).encode("utf-8")).hexdigest())


def i14_provenance(provenance: dict, *, group_layer: dict | None = None) -> dict:
    """Provenance binding, with the values validated and not merely present.

    Added fail-closed checks: every `*_sha256` must look like a sha256, the raw
    batch list must be a non-empty list of identified, digested batches,
    `pool_append_stable` must be RECORDED (the contract requires the field, and
    `.get(..., False)` used to invent it), and the shingle cache's source and
    fingerprint must be declared somewhere - finding F05: the group layer
    accepted an un-fingerprinted sidecar and nothing anywhere named the source
    of the signatures the whole template layer rests on.
    """
    provenance = provenance or {}
    required = ("sample_file", "sample_sha256", "pool_file", "pool_sha256",
                "posting_keys_file", "posting_keys_sha256", "leakage_groups_file",
                "leakage_groups_sha256", "shingles_file", "shingles_sha256",
                "flags_file", "raw_batches")
    missing = [k for k in required if not provenance.get(k)]
    bad_digests = [k for k in required if k.endswith("_sha256")
                   and provenance.get(k)
                   and not _SHA256_RE.match(str(provenance[k]).lower())]
    batches = provenance.get("raw_batches")
    batch_problems = []
    rows_recorded = False
    if not isinstance(batches, list) or not batches:
        batch_problems.append("raw_batches is empty or not a list")
    else:
        rows_recorded = all(isinstance(b, dict) and b.get("rows") is not None
                            for b in batches)
        for i, b in enumerate(batches):
            if not isinstance(b, dict):
                batch_problems.append(f"raw_batches[{i}] is not an object")
                continue
            if not (b.get("batch") or b.get("batch_id")):
                batch_problems.append(f"raw_batches[{i}] has no batch id")
            if not _SHA256_RE.match(str(b.get("sha256", "")).lower()):
                batch_problems.append(f"raw_batches[{i}] has no valid sha256")
    append_stable_recorded = "pool_append_stable" in provenance
    meta = resolve_shingle_meta(provenance, group_layer)
    shingle_problems = []
    if not meta.get("fingerprint"):
        shingle_problems.append(
            f"no shingle cache fingerprint under any of "
            f"{list(SHINGLE_FINGERPRINT_FIELDS)} in provenance or group_layer")
    if not meta.get("source"):
        shingle_problems.append(
            f"no shingle source under any of {list(SHINGLE_SOURCE_FIELDS)}: a "
            f"sidecar and a raw-batch rebuild are not interchangeable and the "
            f"manifest must say which one produced the signatures")
    return _ok(not missing and not bad_digests and not batch_problems
               and append_stable_recorded and not shingle_problems,
               missing=missing, malformed_digests=bad_digests,
               raw_batch_problems=batch_problems[:5],
               n_raw_batches=len(batches) if isinstance(batches, list) else 0,
               raw_batch_rows_recorded=rows_recorded,
               pool_append_stable_recorded=append_stable_recorded,
               pool_append_stable=provenance.get("pool_append_stable"),
               shingle_declaration_problems=shingle_problems,
               shingles_source=meta.get("source"),
               shingles_fingerprint_field=meta.get("fingerprint_field"),
               note=("contract §8 also asks for a row count per raw batch; the "
                     "producer records batch+sha256 only, which is reported as "
                     "raw_batch_rows_recorded rather than failed"))


def i15_regime(df: pd.DataFrame, regime: str, regime_meta: dict) -> dict:
    """Regime-specific ordering / boundary constraints.

    Fail-closed additions: an UNKNOWN regime name fails instead of returning a
    free pass; `temporal` with no derived cut, with a stratum the cuts do not
    cover, or with an unparseable date among the assigned rows fails (the old
    `errors="coerce"` silently turned a bad date into NaT, which `max()` then
    ignored); `cross_country` with no hold-out country, or whose hold-out
    country appears in no test row, fails - `df.country_iso == None` matched
    nothing and passed.
    """
    regime = str(regime)
    if regime not in KNOWN_REGIMES:
        return _ok(False, regime=regime, known_regimes=list(KNOWN_REGIMES),
                   reason="unknown regime: no regime check could be selected, "
                          "so this invariant would pass without checking "
                          "anything")
    regime_meta = regime_meta or {}
    if regime == "temporal":
        missing_cols = _missing_columns(df, ("part", "stratum", "date_posted_utc"))
        if missing_cols:
            return _ok(False, reason="missing columns", missing_columns=missing_cols)
        cuts = regime_meta.get("cut_date", {}) or {}
        bad, problems = [], []
        if not cuts:
            problems.append("no per-stratum cut was recorded: the temporal "
                            "ordering claim cannot be checked")
        assigned = _assigned(df)
        strata = sorted(set(map(str, assigned["stratum"])))
        uncovered = [s for s in strata if s not in {str(k) for k in cuts}]
        if uncovered:
            problems.append(f"strata present in the assignment but absent from "
                            f"the cut map: {uncovered}")
        parsed = pd.to_datetime(assigned["date_posted_utc"], utc=True,
                                errors="coerce")
        n_unparseable = int(parsed.isna().sum())
        if n_unparseable:
            problems.append(f"{n_unparseable} assigned row(s) carry an "
                            f"unparseable date, which the cut comparison would "
                            f"silently ignore")
        early = df[df["part"].isin(("train", "dev"))]
        late = df[df["part"] == "test"]
        for stratum, cut in sorted(cuts.items()):
            cut_ts = pd.Timestamp(cut)
            e = pd.to_datetime(early.loc[early["stratum"].astype(str) == str(stratum),
                                         "date_posted_utc"], utc=True, errors="coerce")
            t = pd.to_datetime(late.loc[late["stratum"].astype(str) == str(stratum),
                                        "date_posted_utc"], utc=True, errors="coerce")
            if len(e) and e.max() >= cut_ts:
                bad.append({"stratum": stratum, "problem": "train/dev holds a post-cut date",
                            "max_early": str(e.max()), "cut": str(cut_ts)})
            if len(t) and t.min() < cut_ts:
                bad.append({"stratum": stratum, "problem": "test holds a pre-cut date",
                            "min_late": str(t.min()), "cut": str(cut_ts)})
        spanning = sorted(set(_by_part(df, "group_id")["test"])
                          & (set(_by_part(df, "group_id")["train"])
                             | set(_by_part(df, "group_id")["dev"])))
        return _ok(not bad and not spanning and not problems,
                   cuts={str(k): str(v) for k, v in cuts.items()},
                   violations=bad, groups_spanning_cut=spanning[:5],
                   n_unparseable_dates=n_unparseable, problems=problems)
    if regime == "grouped_random":
        return _ok(True, regime=regime,
                   note="no regime-specific ordering constraint: the primary "
                        "regime's guarantees are I1-I14 and I16-I17")
    # cross_country
    missing_cols = _missing_columns(df, ("part", "country_iso", "group_id",
                                         "sentence_id"))
    if missing_cols:
        return _ok(False, reason="missing columns", missing_columns=missing_cols)
    iso = _clean(regime_meta.get("holdout_country"))
    problems = []
    if not iso:
        problems.append("no hold-out country recorded: `country_iso == None` "
                        "matches no row, so every boundary check below would "
                        "pass without checking anything")
    assigned = df[df["part"].isin(("train", "dev"))]
    country = _clean_series(df["country_iso"])
    leaked = sorted(df.loc[(country == iso) & df["part"].isin(("train", "dev")),
                           "sentence_id"].astype(str)) if iso else []
    n_test_holdout = int(((country == iso) & (df["part"] == "test")).sum()) if iso else 0
    if iso and n_test_holdout == 0:
        problems.append(f"no test row carries the hold-out country {iso}: the "
                        f"external test does not contain the country it is for")
    # The group-level rule is stated over the POOL, not over the draw. Use a
    # pool-wide map when the producer supplies one (documented field
    # `pool_group_countries`), and say so plainly when it does not.
    pool_map = regime_meta.get("pool_group_countries")
    if isinstance(pool_map, dict) and pool_map:
        groups_with_iso = {str(g) for g, isos in pool_map.items()
                           if iso in {_clean(v) for v in (isos or ())}}
        scope = "pool-wide group->country map supplied by the producer"
    else:
        groups_with_iso = set(df.loc[country == iso, "group_id"].astype(str)) if iso else set()
        scope = ("SAMPLED postings only: a leakage group whose held-out-country "
                 "posting was not drawn is invisible to this check, so it is "
                 "NARROWER than the manifest's holdout_rule_text (finding F09, "
                 "owned by split.py). Supply regime_meta['pool_group_countries'] "
                 "to close it")
    bridging = sorted(groups_with_iso & set(assigned["group_id"].astype(str)))
    return _ok(not leaked and not bridging and not problems,
               holdout_country=iso, leaked_sentences=leaked[:5],
               bridging_groups=bridging[:5],
               n_test_sentences_from_holdout_country=n_test_holdout,
               group_rule_scope=scope, problems=problems)


# ------------------------------------------------------------------------ I16
def _is_num(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) \
        and not isinstance(v, bool)


def _delta_problems(node: dict, path: str) -> list[str]:
    """`delta` must equal `achieved - target`, scalar or per-value."""
    out = []
    t, a, d = node.get("target"), node.get("achieved"), node.get("delta")
    if d is None:
        return out
    if _is_num(t) and _is_num(a) and _is_num(d):
        if abs((a - t) - d) > 1e-9:
            out.append(f"{path}: delta {d} != achieved {a} - target {t}")
    elif isinstance(t, dict) and isinstance(a, dict) and isinstance(d, dict):
        for k in sorted(set(t) | set(a) | set(d)):
            tv, av, dv = t.get(k, 0), a.get(k, 0), d.get(k, 0)
            if _is_num(tv) and _is_num(av) and _is_num(dv) and abs((av - tv) - dv) > 1e-9:
                out.append(f"{path}.{k}: delta {dv} != achieved {av} - target {tv}")
    elif isinstance(t, dict) != isinstance(a, dict):
        out.append(f"{path}: target and achieved have different shapes")
    return out


def i16_reporting(manifest: dict) -> dict:
    """Every target is MEASURED against an achieved value and TRACES to the
    registered config.

    A target reported as achieved is the exact class of bug this project already
    hit once, so it is a WRITE-TIME error, not a style note. The first version of
    this invariant only asked whether a dict carrying `target` also carried
    `achieved`; since the producer always built both keys in the same
    comprehension it could not fail, and it did not fail on the real defect -
    five written splits reported `delta: 0`, `exact_sizes_met: true` and
    `unmet_targets: []` for a test set 28 sentences short of the registered 240,
    because `target` had been assigned from the achieved assignment.

    What it now checks:
      a. `registry.body.targets` exists - without the registered numbers nothing
         here is verifiable, so its absence is a failure;
      b. `sizes[part].target` EQUALS the registered target, which is what makes
         the target's provenance the registry and not the assignment. The
         producer may report the allocator's capacity, but only under a
         different name (`regime_capacity`);
      c. `delta == achieved - target` everywhere, scalar or per-value;
      d. `targets` / `achieved` top-level blocks agree with `sizes`;
      e. `exact_sizes_met` is computed against the REGISTERED targets, and is
         null - not true - wherever `exactness_claimed` is false;
      f. `unmet_targets` names every part whose achieved differs from its
         registered target;
      g. every `marginals.<axis>.<part>.target` column closes to the part total
         it was apportioned from, and that total is the part's realised size, so
         an arithmetically unmeetable target column cannot be published as the
         baseline for a `delta` or a `tv_distance`.
    """
    if not isinstance(manifest, dict) or not manifest:
        return _ok(False, reason="empty manifest: nothing to check")
    problems: list[str] = []
    missing: list[str] = []

    def walk(node, path):
        if isinstance(node, dict):
            if "target" in node:
                if "achieved" not in node:
                    missing.append(path)
                else:
                    problems.extend(_delta_problems(node, path))
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    for key in ("targets", "achieved", "sizes", "marginals", "temporal",
                "cross_country", "verifiability", "tag_support", "power",
                "stratification"):
        if key in manifest:
            walk(manifest[key], key)

    registered = (((manifest.get("registry") or {}).get("body") or {})
                  .get("targets"))
    if not isinstance(registered, dict) or not registered:
        problems.append(
            "registry.body.targets is absent: the registered targets are the "
            "only thing a reported target can be traced to, so a manifest "
            "without them cannot be verified")
        registered = {}
    registered = {p: registered.get(p) for p in PARTS if p in registered}

    sizes = manifest.get("sizes") or {}
    achieved_block = manifest.get("achieved") or {}
    targets_block = manifest.get("targets") or {}
    if not sizes:
        problems.append("sizes block is absent")
    capacity = ((manifest.get("regime_capacity") or {}).get("capacity")
                if isinstance(manifest.get("regime_capacity"), dict) else None) or {}

    circular, achieved_sizes = [], {}
    for part in PARTS:
        block = sizes.get(part)
        if not isinstance(block, dict):
            problems.append(f"sizes.{part} is absent")
            continue
        tgt, ach = block.get("target"), block.get("achieved")
        achieved_sizes[part] = ach
        reg = registered.get(part)
        if reg is None:
            continue
        if not _is_num(tgt):
            problems.append(f"sizes.{part}.target is not a number: {tgt!r}")
        elif int(tgt) != int(reg):
            problems.append(
                f"sizes.{part}.target is {tgt} but the REGISTERED target is "
                f"{reg}: a target must come from the registry, never from the "
                f"assignment or from the allocator's capacity")
            if _is_num(ach) and int(tgt) == int(ach):
                circular.append(part)
                problems.append(
                    f"sizes.{part}.target EQUALS sizes.{part}.achieved ({ach}) "
                    f"while the registered target is {reg}: the target was "
                    f"derived from the achieved assignment, so its delta is 0 "
                    f"by construction")
            if capacity and _is_num(capacity.get(part)) and int(tgt) == int(capacity[part]):
                problems.append(
                    f"sizes.{part}.target equals regime_capacity.capacity."
                    f"{part} ({capacity[part]}): the allocator's capacity is "
                    f"not a target and must be reported under its own name")
        if part in targets_block and _is_num(targets_block[part]) \
                and int(targets_block[part]) != int(reg):
            problems.append(
                f"targets.{part} is {targets_block[part]} but the registered "
                f"target is {reg}")
        if part in achieved_block and _is_num(achieved_block[part]) and _is_num(ach) \
                and int(achieved_block[part]) != int(ach):
            problems.append(
                f"achieved.{part} ({achieved_block[part]}) disagrees with "
                f"sizes.{part}.achieved ({ach})")

    exactness_claimed = manifest.get("exactness_claimed")
    has_esm = "exact_sizes_met" in manifest
    esm = manifest.get("exact_sizes_met")
    met_against_register = all(
        _is_num(achieved_sizes.get(p)) and _is_num(registered.get(p))
        and int(achieved_sizes[p]) == int(registered[p]) for p in registered) \
        if registered else None
    if not has_esm:
        problems.append("exact_sizes_met is absent")
    elif exactness_claimed is False:
        if esm is not None:
            problems.append(
                f"exact_sizes_met is {esm!r} while exactness_claimed is false: a "
                f"regime that claims no exactness must report null, not a "
                f"verdict on sizes it never promised")
    elif exactness_claimed is True and met_against_register is not None \
            and bool(esm) != bool(met_against_register):
        problems.append(
            f"exact_sizes_met is {esm!r} but measured against the REGISTERED "
            f"targets it is {met_against_register}")

    unmet = manifest.get("unmet_targets")
    if unmet is None:
        problems.append("unmet_targets is absent")
    else:
        text = [str(s) for s in unmet]
        for part in registered:
            ach, reg = achieved_sizes.get(part), registered[part]
            if _is_num(ach) and _is_num(reg) and int(ach) != int(reg) \
                    and not any(s.strip().startswith(part) or f"{part}:" in s
                                for s in text):
                problems.append(
                    f"{part} achieved {ach} against a registered {reg} but "
                    f"unmet_targets does not name it")

    # --- marginals: the target column must close to the part total it came from
    column_problems, skipped = [], []
    marginals = manifest.get("marginals") or {}
    for axis, per_part in (marginals.items() if isinstance(marginals, dict) else ()):
        if not isinstance(per_part, dict):
            continue
        for part, block in per_part.items():
            if not isinstance(block, dict) or not isinstance(block.get("target"), dict):
                continue
            if block.get("not_applicable"):
                skipped.append(f"marginals.{axis}.{part}")
                continue
            col = sum(v for v in block["target"].values() if _is_num(v))
            basis = block.get("apportioned_from_part_total")
            ach_size = achieved_sizes.get(part)
            if basis is None:
                basis = capacity.get(part, ach_size)
            if _is_num(basis) and col != int(basis):
                column_problems.append(
                    f"marginals.{axis}.{part}.target sums to {col} against a "
                    f"part total of {int(basis)}: the column is arithmetically "
                    f"unmeetable, so its delta and tv_distance measure rounding "
                    f"residue")
            if _is_num(basis) and _is_num(ach_size) and int(basis) != int(ach_size):
                column_problems.append(
                    f"marginals.{axis}.{part} was apportioned from {int(basis)} "
                    f"but the part holds {int(ach_size)} rows")
            if isinstance(block.get("achieved"), dict) \
                    and set(block["achieved"]) != set(block["target"]):
                column_problems.append(
                    f"marginals.{axis}.{part}: target and achieved cover "
                    f"different value sets")

    t_keys = set(targets_block)
    a_keys = set(achieved_block)
    orphan = sorted(t_keys - a_keys) if t_keys else []
    if orphan:
        problems.append(f"targets keys without an achieved counterpart: {orphan}")
    return _ok(not missing and not problems and not column_problems,
               targets_without_achieved=missing,
               orphan_target_keys=orphan,
               registered_targets=registered,
               achieved_sizes=achieved_sizes,
               parts_whose_target_came_from_the_assignment=circular,
               exact_sizes_met_against_register=met_against_register,
               problems=problems[:20], n_problems=len(problems),
               marginal_column_problems=column_problems[:20],
               n_marginal_column_problems=len(column_problems),
               marginal_blocks_not_applicable=skipped)


def i17_no_labels(frames: dict, *, required_frames: tuple[str, ...] = ()) -> dict:
    """No file written by split.py carries a label column.

    Fail-closed additions: an EMPTY `frames` mapping fails (there is nothing to
    clear), any frame named in `required_frames` must be present, and a
    zero-row `test_blind` fails - "the blind test carries no labels" is not a
    claim worth making about no rows.
    """
    frames = frames or {}
    offenders, empties = {}, []
    if not frames:
        return _ok(False, reason="no frames handed in: the no-labels claim would "
                                 "be made over nothing")
    absent = [name for name in required_frames if name not in frames]
    for name, frame in frames.items():
        columns = list(getattr(frame, "columns", []))
        bad = [c for c in columns
               if c not in TEST_BLIND_ALLOWED_COLUMNS and LABEL_LIKE.match(str(c))]
        if name.startswith("test_blind"):
            bad = sorted(set(bad) | (set(columns) - TEST_BLIND_ALLOWED_COLUMNS))
            if len(frame) == 0:
                empties.append(name)
        if bad:
            offenders[name] = sorted(bad)
    return _ok(not offenders and not absent and not empties,
               offenders=offenders, missing_required_frames=absent,
               empty_blind_frames=empties,
               n_rows={n: int(len(f)) for n, f in frames.items()},
               allowlist=sorted(TEST_BLIND_ALLOWED_COLUMNS))


# -------------------------------------------------------------------- driver
def run_invariants(df: pd.DataFrame, *, input_ids: list[str], targets: dict,
                   exact: bool, shingles: dict,
                   pilot: dict, practice: dict,
                   allowed_reasons: tuple[str, ...], provenance: dict,
                   regime: str, regime_meta: dict, manifest: dict,
                   frames: dict,
                   registered_targets: dict | None = None,
                   shingle_meta: dict | None = None,
                   require_shingle_fingerprint: bool = True,
                   required_frames: tuple[str, ...] = ("train", "dev", "test_blind"),
                   ) -> dict:
    """Run I1-I17 over one computed assignment.

    `registered_targets` and `shingle_meta` are derived from the manifest when
    the caller does not pass them: the manifest already carries
    `registry.body.targets` and a verbatim copy of `leakage_groups_manifest.json`
    under `group_layer`, so the gate can reach both without `split.py` growing
    new arguments.
    """
    manifest = manifest or {}
    group_layer = manifest.get("group_layer") if isinstance(manifest, dict) else None
    if registered_targets is None:
        registered_targets = (((manifest.get("registry") or {}).get("body") or {})
                              .get("targets"))
    if shingle_meta is None:
        shingle_meta = resolve_shingle_meta(provenance, group_layer)
    return {
        "I1_accounting": i1_accounting(df, input_ids, allowed_reasons),
        "I2_group_disjoint": i2_group_disjoint(df),
        "I3_posting_disjoint": i3_posting_disjoint(df),
        "I4_employer_exact_disjoint": i4_employer_exact_disjoint(df),
        "I5_employer_loose_disjoint": i5_employer_loose_disjoint(df),
        "I6_template_cluster_disjoint": i6_template_cluster_disjoint(df),
        "I7_direct_pairwise_jaccard": i7_direct_pairwise_jaccard(
            df, shingles, shingle_meta=shingle_meta,
            require_fingerprint=require_shingle_fingerprint),
        "I8_sentence_text_disjoint": i8_sentence_text_disjoint(df),
        "I9_dev_test_verifiable": i9_dev_test_verifiable(df),
        "I10_quarantine": i10_quarantine(df, pilot, practice),
        "I11_oversize_groups": i11_oversize_groups(df, targets),
        "I12_exact_sizes": i12_exact_sizes(df, targets, exact,
                                           registered_targets=registered_targets),
        "I13_determinism": i13_determinism(df),
        "I14_provenance": i14_provenance(provenance, group_layer=group_layer),
        "I15_regime": i15_regime(df, regime, regime_meta),
        "I16_reporting": i16_reporting(manifest),
        "I17_no_labels": i17_no_labels(frames, required_frames=required_frames),
    }


def assert_invariants(results: dict) -> dict:
    failed = {k: v for k, v in results.items() if not v.get("passed")}
    if failed:
        import json
        raise InvariantFailure(
            "refusing to write the split — "
            f"{len(failed)} leakage invariant(s) failed: {sorted(failed)}\n"
            + json.dumps(failed, indent=2, default=str))
    return results


# ---------------------------------------------------------------- the audit
# Can this invariant pass vacuously on ABSENT, EMPTY or DEGENERATE input? Asked
# of every invariant in this module, with the answer each one now carries.
# `tests/test_split_assert.py` holds a positive control per row.
VACUITY_AUDIT = {
    "I1_accounting": "WAS vacuous on an empty frame / empty input_ids (the "
                     "union identity held over nothing) and on a blank "
                     "sentence_id. Now fails on all three, and on an empty part.",
    "I2_group_disjoint": "WAS vacuous when group_id was null everywhere "
                         "(_by_part dropped NaN). Now a missing group_id on an "
                         "assigned row fails, as does having fewer than two "
                         "parts with any value.",
    "I3_posting_disjoint": "Same disease, same fix as I2.",
    "I4_employer_exact_disjoint": "WAS vacuous if no row carried a hash, and "
                                  "treated NaN as the literal value 'nan'. Now "
                                  "sentinels normalise to missing, coverage is "
                                  "published, and a dead column fails. "
                                  "Per-posting empties stay legal by contract "
                                  "(confined to train; I9 covers dev/test).",
    "I5_employer_loose_disjoint": "Same, and the empties are EXPECTED here "
                                  "(finding F03), so they are counted and the "
                                  "coverage share published rather than failed. "
                                  "A column with no non-empty value fails.",
    "I6_template_cluster_disjoint": "WAS vacuous on an all-null cluster column; "
                                    "now fails. A missing column already failed.",
    "I7_direct_pairwise_jaccard": "WAS the worst case: absent and empty "
                                  "signatures were SKIPPED and zero compared "
                                  "pairs passed. Now absence, emptiness, zero "
                                  "pairs, a blank posting_id and an "
                                  "undeclared/mismatched cache fingerprint all "
                                  "fail. Truncation scope is reported, not "
                                  "claimed away.",
    "I8_sentence_text_disjoint": "WAS vacuous on an empty assigned frame and "
                                 "skipped blank text. Now blank text fails and "
                                 "zero compared pairs fails.",
    "I9_dev_test_verifiable": "WAS vacuous when dev+test were empty "
                              "(passed with share=None) and astype(bool) read "
                              "NaN and 'False' as True. Now strict coercion, an "
                              "empty dev/test fails, and the employer_hash "
                              "column is checked as the contract words it.",
    "I10_quarantine": "WAS vacuous when the quarantine reference sets were "
                      "empty, which is exactly what a missing pilot_100.parquet "
                      "produces. Now an empty or malformed reference fails.",
    "I11_oversize_groups": "WAS skipped entirely when a part's quota was 0 or "
                           "absent. Now that fails, and the limit is measured "
                           "against max(quota, achieved).",
    "I12_exact_sizes": "The soft branch returned passed=True unconditionally, "
                       "by contract (sizes are regime-determined). Kept, but an "
                       "EMPTY part now fails in both branches, a zero-sum "
                       "capacity fails in exact mode, and the registered "
                       "shortfall is reported here instead of being invisible.",
    "I13_determinism": "WAS a recorder that always passed. Now an empty frame, "
                       "duplicate sentence ids, blank fields or an assigned row "
                       "without a group all fail; the digest is unchanged.",
    "I14_provenance": "WAS a truthiness check on key presence. Now digests must "
                      "look like sha256, raw_batches must be a non-empty list "
                      "of identified digested batches, pool_append_stable must "
                      "be RECORDED, and the shingle cache's source and "
                      "fingerprint must be declared.",
    "I15_regime": "WAS a free pass for an unknown regime name, for a temporal "
                  "regime with no cut map, for unparseable dates (coerced to "
                  "NaT and ignored by max()) and for cross_country with no "
                  "hold-out country. All four now fail. The group-level "
                  "cross-country rule is still evaluated over the SAMPLED "
                  "postings unless the producer supplies "
                  "regime_meta['pool_group_countries']; the scope is reported "
                  "in group_rule_scope (finding F09, split.py's to close).",
    "I16_reporting": "WAS a key-presence check that could not fail. Now "
                     "compares values, requires every target to equal the "
                     "REGISTERED target, rejects a target equal to the achieved "
                     "assignment or to the allocator capacity, checks every "
                     "delta, requires unmet_targets to name every shortfall, "
                     "and requires every marginals target column to close to "
                     "the part total it was apportioned from.",
    "I17_no_labels": "WAS vacuous on an empty frames mapping and could not see "
                     "a frame that was never handed in. Now an empty mapping, a "
                     "missing required frame and a zero-row blind test all fail.",
}

__all__ = [
    "InvariantFailure", "PARTS", "ALL_PARTS", "KNOWN_REGIMES",
    "TEST_BLIND_ALLOWED_COLUMNS", "MISSING_STRINGS",
    "SHINGLE_FINGERPRINT_FIELDS", "SHINGLE_SOURCE_FIELDS",
    "SHINGLE_FINGERPRINT_ALGORITHM", "REQUIRED_QUARANTINE_KEYS",
    "VACUITY_AUDIT", "shingle_cache_fingerprint", "resolve_shingle_meta",
    "run_invariants", "assert_invariants", "i1_accounting", "i2_group_disjoint",
    "i3_posting_disjoint", "i4_employer_exact_disjoint",
    "i5_employer_loose_disjoint", "i6_template_cluster_disjoint",
    "i7_direct_pairwise_jaccard", "i8_sentence_text_disjoint",
    "i9_dev_test_verifiable", "i10_quarantine", "i11_oversize_groups",
    "i12_exact_sizes", "i13_determinism", "i14_provenance", "i15_regime",
    "i16_reporting", "i17_no_labels", "SPLIT_TARGETS",
]
