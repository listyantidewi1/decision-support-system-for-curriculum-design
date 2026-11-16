"""Leakage groups: employer (exact + loose) UNION template -> one atomic group.

The proposal's only hard clause for the split layer is a prohibition (PROP:1192,
PROP:1436): postings from the same employer or the same "atau sangat mirip"
(or very similar) template must not be spread across train and test. "Same
employer" is an equivalence relation, but "similar template" is NOT transitive,
so a split can only be made safe by assigning the TRANSITIVE CLOSURE of the
union of the three relations as one atomic unit:

  (a) equal non-empty employer_hash          (exact lower-cased company string)
  (b) equal non-empty employer_hash_loose    (legal-form-insensitive)
  (c) cleaned-text shingle Jaccard >= TAU_BUILD, over the WHOLE cleaned
      description (see `textsig` and finding F01 below)

Scope is the POOL (18,667 postings), not all 85,740 eligible. For (a) and (b)
this is exactly equivalent — restricting the nodes of an equivalence relation
cannot change the partition of the remaining nodes — and the only thing global
scope would add is two-hop template bridges through postings that can never be
sampled.

WHAT I7 IS AND IS NOT (corrected overclaim). This docstring used to say that
invariant I7 in `split_assert` closes the two-hop residual and is "a strictly
stronger guarantee than closure". That was false in two ways and the adversarial
review reproduced both. I7 re-derives the maximum direct cross-part Jaccard
from THE SAME signature cache this module builds, so it shares every parameter
choice with the union-find: when the cache was truncated at 2,500 characters I7
measured a prefix too, and when the cache was stale or incomplete I7 silently
skipped the postings it did not cover. I7 is a CONSISTENCY CHECK on the
cache -> group-map -> assignment chain, not an independent leakage oracle, and
it is honest only because the cache is now (i) taken over the full cleaned text
and (ii) fingerprinted, so a cache that does not match the parameters the group
map was built under is REFUSED rather than quietly used. The genuine residual —
a two-hop template bridge through a NON-POOL posting — is not covered by either
layer and is carried as a stated limitation, not as a closed hole.

FINDING F01 (FATAL), fixed here. The template relation used to be computed over
the first 2,500 cleaned characters. On the real pool that left 126 posting pairs
whose full cleaned-text Jaccard is >= 0.30 in DIFFERENT leakage groups (worst
case full 0.7374 / truncated 0.2814, two ITALDESIGN postings), free to straddle
train and test with every invariant green. `textsig.LEAKAGE_WINDOW_CHARS` is now
None (the whole description); the chosen window is recorded in the manifest and
fingerprinted, so a cache built under one window can never be reused under
another.

FINDING F17 (edge-set completeness), fixed here. Candidate generation used
`MinHashLSH(threshold=0.25, num_perm=128)`, which datasketch resolves to 42
bands of 3, so its recall is below 1.0. Measured on the real pool:

    candidate generator                     pairs found   recall   time
    MinHashLSH, 2,500-char signatures        4,799/5,319  0.9022   (shipped)
    MinHashLSH, full-text signatures        11,272/13,430 0.8393   380.8 s
    exact inverted index, full-text         13,430/13,430 1.0000     8.2 s

with zero false positives in every case (exact verification). 16 of the real
cross-group pairs were pure recall misses rather than truncation misses. The
LSH pass is therefore replaced by an EXACT inverted-index pass
(`template_pairs`), which is complete at every tau by construction and 46x
faster on this corpus. The edge set is no longer a lower bound on its own
threshold; tau itself is still a chosen floor, and employer hashes are still
exact-string hashes, so grouping remains a lower bound on true relatedness for
those reasons only.

Honest claim wording, fixed in the manifest: "no employer (exact or loose hash)
and no template pair at full-cleaned-text Jaccard >= 0.30 straddles any two
parts" — with the residuals named: tau is a chosen floor, employer hashes miss
abbreviations and rebrands, and two-hop bridges through non-pool postings are
not grouped.

Run:  .venv-ml\\Scripts\\python.exe -m gold_standard.groups
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from governance.ladder import RAW_ROOT
from . import (LSH_THRESHOLD, MINHASH_NUM_PERM, SEED, SHINGLE_MAX_CHARS,
               MIN_SHARED_SHINGLES, SHINGLE_WORDS, SPLIT_SPEC_VERSION,
               TAU_ASSERT, TAU_BUILD)
from .posting_keys import (GS_DIR, _raw_batch_paths, _sha256,
                           is_missing_hash)
from .textsig import (HASH_FN, LEAKAGE_WINDOW_CHARS, cache_fingerprint,
                      params_fingerprint, signature_fingerprint,
                      signature_params, text_pipeline, verify_cache_fingerprint,
                      window_label)

# Refuse to write a group map that cannot be split: a component bigger than this
# share of a stratum's pool postings would make a 20% test share impossible.
MAX_COMPONENT_SHARE_OF_STRATUM = 0.05
# A single LOOSE edge is the weakest evidence in the union; refuse to let one
# merge two already-large components (a runaway normalisation collapse).
MAX_LOOSE_MERGE_SHARE = 0.02
# Audit only: cleaned-text pairs this similar would have been caught by the
# ladder's 0.85 dedup had the ladder run on cleaned rather than raw text.
LADDER_DEDUP_THRESHOLD = 0.85
# How many postings of an otherwise-verified sidecar are re-shingled from the
# RAW layer before the sidecar is believed. A fingerprint proves the cache is
# self-consistent and was written under the right parameters; only the raw text
# proves the signatures describe the right postings. 256 of 18,667 costs ~20 s
# (the raw read dominates) against ~220 s for a full rebuild. Stated honestly,
# not oversold: the audit catches a WHOLESALE forgery with certainty — which is
# the attack actually reproduced, all 18,667 rows replaced — and a single
# tampered row only with probability 256/18,667 = 1.4%. A single tampered row
# inside a sidecar whose manifest was forged to match it is NOT caught; that
# residual is stated in the manifest and is why the raw layer stays
# authoritative (delete the sidecar to force a full rebuild).
SIDECAR_AUDIT_SAMPLE = 256
# Measured MinHashLSH recall at TAU_BUILD on the real pool, recorded in the
# manifest as the reason candidate generation is now exact (finding F17).
# 0.9022 = 4,799 of 5,319 exact pairs under the shipped 2,500-char window;
# 0.8393 = 11,272 of 13,430 exact pairs under full-text signatures.
SUPERSEDED_LSH_MEASURED_RECALL = {"at_2500_char_window": 0.9022,
                                  "at_full_text": 0.8393}


class GroupFeasibilityError(RuntimeError):
    """Raised instead of writing a group map that cannot carry a 60/20/20 split."""


class ShingleCacheError(RuntimeError):
    """Raised instead of using a signature cache whose fingerprint does not match.

    The whole template half of the leakage guarantee rests on this cache, so an
    unverifiable cache is a hard stop at every point where there is no
    authoritative source to fall back to (finding F05/F14: a sidecar with the
    right posting ids and garbage signatures produced 0 template edges, 456
    groups instead of 418, and a written split with all invariants green).
    """


def _cache_manifest_path(cache_path: Path) -> Path:
    cache_path = Path(cache_path)
    return cache_path.with_name(cache_path.stem + "_manifest.json")


# --------------------------------------------------------------------- shingles
def _load_descriptions(raw_root: Path, posting_ids: set[str]) -> pd.DataFrame:
    """First-sighting description text for the given postings, from the raw layer."""
    frames = []
    for path in _raw_batch_paths(raw_root):
        frame = pd.read_parquet(
            path, columns=["source_record_id", "acquisition_timestamp", "description"])
        frames.append(frame[frame["source_record_id"].isin(posting_ids)])
    if not frames:
        raise FileNotFoundError(f"no batch-*/records.parquet under {raw_root}")
    raw = pd.concat(frames, ignore_index=True)
    return (raw.sort_values(["acquisition_timestamp", "source_record_id"], kind="stable")
               .drop_duplicates("source_record_id"))


def _shingle_from_raw(raw_root: Path, posting_ids: set[str],
                      window_chars: int | None) -> dict[str, np.ndarray]:
    from .build_pool import _html_to_text
    from .textsig import shingles_from_clean_text

    desc = _load_descriptions(raw_root, posting_ids)
    found = set(desc["source_record_id"].astype(str))
    if found != set(map(str, posting_ids)):
        missing = set(map(str, posting_ids)) - found
        raise GroupFeasibilityError(
            f"{len(missing)} pool postings have no description in the raw layer "
            f"(stale sidecar or missing batch); first few: {sorted(missing)[:5]}")
    return {str(rec.source_record_id): shingles_from_clean_text(
                _html_to_text(rec.description), max_chars=window_chars)
            for rec in desc.itertuples(index=False)}


def _audit_sample(posting_ids: list[str], n: int) -> list[str]:
    """Deterministic audit subset (seeded, so two runs audit the same postings)."""
    ids = sorted(map(str, posting_ids))
    if len(ids) <= n:
        return ids
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(ids), size=n, replace=False)
    return sorted(ids[i] for i in idx)


def _try_sidecar(sidecar: Path, want: set[str], raw_root: Path,
                 window_chars: int | None, verbose: bool) -> tuple[dict | None, dict]:
    """Accept `build_pool`'s signature sidecar only if it can be VERIFIED.

    Four gates, in cost order: the sidecar exists; it carries a fingerprint
    manifest that matches this code's parameters, its own posting-id set and its
    own contents; it covers every pool posting; and a seeded audit sample
    re-shingled from the raw layer reproduces its signatures. Posting-id
    coverage, which used to be the ONLY gate (`want.issubset(have)`), is now the
    weakest of the four and is never sufficient on its own.
    """
    sidecar = Path(sidecar)
    report: dict = {"sidecar_file": sidecar.name, "accepted": False, "reasons": []}
    if not sidecar.exists():
        report["reasons"].append("the sidecar does not exist")
        return None, report
    report["sidecar_sha256"] = _sha256(sidecar)

    man_path = _cache_manifest_path(sidecar)
    declared = None
    if man_path.exists():
        try:
            man = json.loads(man_path.read_text(encoding="utf-8"))
            declared = man.get("signature_cache", man)
        except (ValueError, OSError) as exc:
            report["reasons"].append(f"{man_path.name} is unreadable: {exc}")
    else:
        report["reasons"].append(
            f"{man_path.name} is absent, so the sidecar carries no signature "
            f"fingerprint: the window, the cleaner and the shingle width it was "
            f"written under are unverifiable")
    if declared is None:
        return None, report
    report["sidecar_manifest_file"] = man_path.name

    side = pd.read_parquet(sidecar)
    have = {str(r.posting_id): np.asarray(r.shingles, dtype=np.uint64)
            for r in side.itertuples(index=False)}
    report["sidecar_n_postings"] = len(have)
    problems = verify_cache_fingerprint(declared, have, window_chars=window_chars,
                                        shingle_words=SHINGLE_WORDS,
                                        require_ids=want)
    report["reasons"].extend(problems)
    if problems:
        return None, report

    audit = _audit_sample(sorted(want), SIDECAR_AUDIT_SAMPLE)
    report["audit_n"] = len(audit)
    try:
        fresh = _shingle_from_raw(raw_root, set(audit), window_chars)
    except (GroupFeasibilityError, FileNotFoundError) as exc:
        report["reasons"].append(f"the raw audit could not run: {exc}")
        return None, report
    bad = [pid for pid in audit
           if signature_fingerprint(fresh[pid]) != signature_fingerprint(have[pid])]
    report["audit_mismatches"] = len(bad)
    if bad:
        report["reasons"].append(
            f"{len(bad)} of {len(audit)} audited postings re-shingle to a "
            f"different signature than the sidecar records; first few: {bad[:5]}")
        return None, report

    report["accepted"] = True
    if verbose:
        print(f"  sidecar {sidecar.name}: fingerprint matches and {len(audit)} "
              f"audited postings re-shingle identically -> accepted", flush=True)
    return {pid: have[pid] for pid in sorted(want)}, report


def build_shingles(posting_ids: list[str], raw_root: Path, out_dir: Path,
                   sidecar: Path | None = None, verbose: bool = True,
                   window_chars: int | None = LEAKAGE_WINDOW_CHARS,
                   ) -> tuple[dict[str, np.ndarray], dict]:
    """Signature cache for the pool postings, with its fingerprint.

    Returns `(cache, meta)`. `meta` is the provenance block the group manifest
    records: which source produced the signatures, why the sidecar was refused
    if it was, and the cache fingerprint a consumer must match.
    """
    want = set(map(str, posting_ids))
    cache, sidecar_report = None, {"sidecar_file": None, "accepted": False,
                                   "reasons": ["no sidecar path was given"]}
    if sidecar:
        cache, sidecar_report = _try_sidecar(Path(sidecar), want, raw_root,
                                             window_chars, verbose)
    source = f"sidecar:{sidecar_report.get('sidecar_file')}" if cache else "raw_batches"
    if cache is None:
        if sidecar and verbose:
            print(f"  sidecar refused ({'; '.join(sidecar_report['reasons'])[:300]})"
                  f" -> recomputing every signature from the raw batches", flush=True)
        t0 = time.perf_counter()
        fresh = _shingle_from_raw(raw_root, want, window_chars)
        cache = {pid: fresh[pid] for pid in sorted(want)}
        if verbose:
            print(f"  shingled {len(cache):,} postings from the raw layer in "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)

    # Deterministic row order, so the parquet is byte-stable across runs.
    cache = {pid: cache[pid] for pid in sorted(cache)}
    fp = cache_fingerprint(cache, window_chars=window_chars,
                           shingle_words=SHINGLE_WORDS)
    counts = np.array([v.size for v in cache.values()], dtype=np.int64)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "shingles.parquet"
    pd.DataFrame({
        "posting_id": list(cache.keys()),
        "n_shingles": [int(v.size) for v in cache.values()],
        "shingle_sha1_16": [signature_fingerprint(v) for v in cache.values()],
        "shingles": [v.tolist() for v in cache.values()],
    }).to_parquet(path, index=False)

    meta = {
        "shingles_file": path.name,
        "shingles_source": source,
        "shingles_fingerprint": fp["content_sha256"],
        "signature_cache": fp,
        "sidecar_decision": sidecar_report,
        "n_postings": int(len(cache)),
        "n_postings_without_signature": int((counts == 0).sum()),
        "mean_shingles_per_posting": round(float(counts.mean()), 1) if counts.size else 0.0,
        "median_shingles_per_posting": float(np.median(counts)) if counts.size else 0.0,
        "max_shingles_per_posting": int(counts.max()) if counts.size else 0,
        "total_shingles": int(counts.sum()),
    }
    (_cache_manifest_path(path)).write_text(
        json.dumps({**meta, "created_utc": datetime.now(timezone.utc)
                    .isoformat(timespec="seconds")},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    if verbose:
        print(f"  shingles ({source}): {len(cache):,} postings, "
              f"{meta['mean_shingles_per_posting']} mean shingles, "
              f"fingerprint {fp['content_sha256'][:16]}... -> {path}", flush=True)
    return cache, meta


def load_shingles(path: Path, *, verify: bool = True,
                  window_chars: int | None = LEAKAGE_WINDOW_CHARS,
                  require_ids=None) -> dict[str, np.ndarray]:
    """Load the signature cache, REFUSING it unless its fingerprint matches.

    Fail-closed on purpose, and this is the only place downstream consumers
    (`split.py` -> I7) touch the cache. Before this check, three separate real
    attacks went through silently: deleting the sampled postings' rows turned I7
    into `n_pairs_compared: 0, max_cross_part_jaccard: 0.0, passed: true`; a
    sidecar of garbage signatures produced 0 template edges with every invariant
    green; and a cache built under the old 2,500-char window would be
    indistinguishable from one built under the full-text window.

    `verify=False` exists only for callers that have just built the cache in the
    same process and have already fingerprinted it.
    """
    path = Path(path)
    frame = pd.read_parquet(path)
    cache = {str(r.posting_id): np.asarray(r.shingles, dtype=np.uint64)
             for r in frame.itertuples(index=False)}
    if not verify:
        return cache
    man_path = _cache_manifest_path(path)
    if not man_path.exists():
        raise ShingleCacheError(
            f"{path.name} has no fingerprint manifest ({man_path.name}): a "
            f"signature cache cannot be trusted on its posting ids alone — the "
            f"window, the cleaner, the shingle width and the signature contents "
            f"are all unverifiable. Rerun gold_standard.groups to rebuild both.")
    try:
        declared = json.loads(man_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ShingleCacheError(f"{man_path.name} is unreadable: {exc}") from exc
    problems = verify_cache_fingerprint(declared.get("signature_cache", declared),
                                        cache, window_chars=window_chars,
                                        shingle_words=SHINGLE_WORDS,
                                        require_ids=require_ids)
    if problems:
        raise ShingleCacheError(
            f"{path.name} does not match the signature contract recorded in "
            f"{man_path.name}; refusing to use it:\n  - "
            + "\n  - ".join(problems)
            + "\nRerun gold_standard.groups so the cache, the group map and the "
              "manifests are rebuilt together.")
    return cache


# ------------------------------------------------------------- template pairing
def template_pairs(shingles: dict[str, np.ndarray], min_jaccard: float,
                   verbose: bool = True, chunk: int = 2000,
                   min_shared: int | None = None,
                   ) -> tuple[pd.DataFrame, dict]:
    """EXACT similar pairs at `min_jaccard` OR `min_shared`. Complete, order-independent.

    TWO rules, because a ratio alone is not a template test. Jaccard is
    |A n B| / |A u B|, so a long posting DILUTES a large shared block: two ads
    can share an unbounded amount of verbatim text and still fall under tau
    purely by being long. Measured on the real pool, 220 pairs shared 10 or more
    IDENTICAL sampleable sentences while sitting below tau and therefore going
    ungrouped (review finding on groups.py:606), and removing the old
    2,500-character window diluted 26 pairs the windowed rule had caught. An
    absolute floor on |A n B| closes both: a pair qualifies at
    `jaccard >= min_jaccard` OR `n_shared >= min_shared`.

    The completeness proof below is unaffected: both rules require
    |A n B| >= 1, and the enumeration covers every pair sharing any shingle.

    Completeness proof (this is why it replaces MinHashLSH, finding F17):
    Jaccard(A, B) >= tau > 0 requires |A n B| >= 1, so every qualifying pair
    shares at least one shingle; a shingle whose document frequency is 1
    contributes to no off-diagonal intersection, so dropping those columns is
    lossless; and the sparse product of the remaining incidence matrix with its
    transpose yields |A n B| for EVERY pair that shares any shingle. The
    enumeration is therefore exhaustive, and exact Jaccard on the uint64 sets
    then decides each pair, so the edge set has neither false positives nor
    false negatives at `min_jaccard`.

    Cost on the real pool (18,667 postings, full cleaned text, 10.2M signature
    entries): 898,816 shared shingles, 3.72M non-zeros, 4.8 s to index and 4.0 s
    to enumerate. The MinHashLSH pass it replaces took 208 s and had measured
    recall 0.9022 at tau = 0.30.
    """
    from scipy import sparse

    t0 = time.perf_counter()
    ids = sorted(shingles)
    sizes = np.array([int(np.asarray(shingles[p]).size) for p in ids], dtype=np.int64)
    n_empty = int((sizes == 0).sum())
    if sizes.sum() == 0:
        if verbose:
            print("  no signatures at all -> no template pairs", flush=True)
        return (pd.DataFrame(columns=["a", "b", "jaccard", "n_shared"]),
                {"method": "exact-sparse-overlap", "n_postings": len(ids),
                 "n_postings_without_signature": n_empty, "n_shared_shingles": 0,
                 "nnz": 0, "n_pairs_verified": 0, "seconds": 0.0})

    flat = np.concatenate([np.asarray(shingles[p], dtype=np.uint64) for p in ids])
    rows = np.repeat(np.arange(len(ids), dtype=np.int64), sizes)
    _, col = np.unique(flat, return_inverse=True)
    col = np.asarray(col, dtype=np.int64).ravel()
    dfs = np.bincount(col)
    keep = dfs[col] > 1                       # df == 1 cannot create a pair
    rows_k = rows[keep]
    _, col_k = np.unique(col[keep], return_inverse=True)
    col_k = np.asarray(col_k, dtype=np.int64).ravel()
    n_cols = int(col_k.max()) + 1 if col_k.size else 1
    X = sparse.csr_matrix((np.ones(col_k.size, dtype=np.int32), (rows_k, col_k)),
                          shape=(len(ids), n_cols))
    Xt = X.T.tocsr()
    work = int((dfs[dfs > 1].astype(np.float64) ** 2).sum())
    if verbose:
        print(f"  exact overlap index: {len(ids):,} postings, {n_cols:,} shared "
              f"shingles, {col_k.size:,} entries ({time.perf_counter() - t0:.1f}s)",
              flush=True)

    id_arr = np.asarray(ids, dtype=object)
    parts, n_candidates = [], 0
    for start in range(0, len(ids), max(int(chunk), 1)):
        stop = min(start + max(int(chunk), 1), len(ids))
        c = (X[start:stop] @ Xt).tocoo()
        r, cc, inter = c.row + start, c.col, c.data.astype(np.float64)
        upper = cc > r                        # each unordered pair exactly once
        r, cc, inter = r[upper], cc[upper], inter[upper]
        n_candidates += int(r.size)
        union = sizes[r] + sizes[cc] - inter
        j = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        hit = j >= min_jaccard
        if min_shared is not None:
            # OR, not AND: the floor exists precisely to catch pairs the ratio
            # misses, so it must widen the edge set, never narrow it.
            hit = hit | (inter >= float(min_shared))
        if hit.any():
            parts.append(pd.DataFrame({"a": id_arr[r[hit]], "b": id_arr[cc[hit]],
                                       "jaccard": j[hit],
                                       "n_shared": inter[hit].astype(np.int64)}))
        if verbose and stop % (max(int(chunk), 1) * 5) == 0:
            print(f"  exact overlap {stop:,}/{len(ids):,} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    pairs = (pd.concat(parts, ignore_index=True) if parts
             else pd.DataFrame(columns=["a", "b", "jaccard", "n_shared"]))
    pairs["jaccard"] = pairs["jaccard"].astype(float)
    pairs["n_shared"] = pairs["n_shared"].astype("int64")
    pairs = pairs.sort_values(["a", "b"], kind="stable").reset_index(drop=True)
    meta = {
        "method": "exact-sparse-overlap",
        "complete_at_min_jaccard": True,
        "completeness_basis": (
            "every pair at Jaccard >= tau shares at least one shingle; shingles "
            "with document frequency 1 cannot contribute to an off-diagonal "
            "intersection, so every remaining candidate pair is enumerated and "
            "verified by exact Jaccard — no false positives and no false "
            "negatives at this threshold"),
        "min_jaccard": float(min_jaccard),
        "min_shared_shingles": None if min_shared is None else int(min_shared),
        "n_postings": len(ids),
        "n_postings_without_signature": n_empty,
        "n_shared_shingles": int(n_cols),
        "nnz": int(col_k.size),
        "n_candidate_pairs_sharing_a_shingle": int(n_candidates),
        "overlap_work_sum_df_squared": work,
        "n_pairs_verified": int(len(pairs)),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    if verbose:
        print(f"  exact template pairs >= {min_jaccard}: {len(pairs):,} "
              f"({meta['seconds']}s, {n_candidates:,} candidate pairs)", flush=True)
    return pairs, meta


# ------------------------------------------------------------------- union-find
class _UF:
    def __init__(self, nodes: list[str]) -> None:
        self.parent = {n: n for n in nodes}
        self.size = {n: 1 for n in nodes}

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:       # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> tuple[str, str] | None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return None
        # deterministic: larger component wins, ties by lexicographic root
        if (self.size[ra], rb) < (self.size[rb], ra):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return ra, rb


def _components(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, str]:
    uf = _UF(nodes)
    for a, b in edges:
        uf.union(a, b)
    members: dict[str, list[str]] = {}
    for n in nodes:
        members.setdefault(uf.find(n), []).append(n)
    out = {}
    for group in members.values():
        label = min(group)
        for n in group:
            out[n] = label
    return out


def _equality_edges(keys: pd.DataFrame, column: str) -> list[tuple[str, str]]:
    """Star edges inside every non-empty value class (n-1 edges, not n^2).

    MISSING is decided by `posting_keys.is_missing_hash`, not by `!= ""`. The
    string comparison was correct for every hash `posting_keys` writes today, but
    an object-dtype column holding the literal `"nan"` (a parquet/CSV round trip
    is enough) would have been kept as a VALUE and collapsed every such posting
    into ONE employer group — the exact failure the module forbids. One shared
    predicate, so the two layers cannot drift apart.
    """
    edges = []
    sub = keys[~keys[column].map(is_missing_hash)]
    for _, grp in sub.groupby(column, sort=True):
        ids = sorted(grp["posting_id"].astype(str))
        edges.extend((ids[0], other) for other in ids[1:])
    return edges


# ------------------------------------------------------------------------ build
def _raw_batch_provenance(raw_root: Path) -> list[dict]:
    """Batch id, ROW COUNT and sha256 per raw batch (contract §8 asks for rows)."""
    import pyarrow.parquet as pq

    out = []
    for p in _raw_batch_paths(raw_root):
        try:
            rows = int(pq.ParquetFile(p).metadata.num_rows)
        except Exception:                                   # pragma: no cover
            rows = None
        out.append({"batch": p.parent.name, "rows": rows, "sha256": _sha256(p)})
    return out


def build_groups(raw_root: Path = RAW_ROOT,
                 keys_path: Path | None = None,
                 pool_path: Path | None = None,
                 out_dir: Path = GS_DIR,
                 shingle_sidecar: Path | None = None,
                 verbose: bool = True,
                 window_chars: int | None = LEAKAGE_WINDOW_CHARS) -> pd.DataFrame:
    out_dir = Path(out_dir)
    keys_path = Path(keys_path) if keys_path else out_dir / "posting_keys.parquet"
    pool_path = Path(pool_path) if pool_path else out_dir / "sentence_pool.parquet"
    if shingle_sidecar is None:
        shingle_sidecar = out_dir / "posting_shingles.parquet"

    keys = pd.read_parquet(keys_path)
    pool_ids = sorted(set(pd.read_parquet(pool_path, columns=["posting_id"])["posting_id"]))
    keys = keys[keys["posting_id"].isin(pool_ids)].copy()
    missing = set(pool_ids) - set(keys["posting_id"])
    if missing:
        raise GroupFeasibilityError(
            f"{len(missing)} pool postings absent from {keys_path.name} — the sidecar "
            f"is stale; rerun gold_standard.posting_keys. First few: {sorted(missing)[:5]}")
    keys = keys.sort_values("posting_id", kind="stable").reset_index(drop=True)
    nodes = [str(p) for p in keys["posting_id"]]
    if verbose:
        print(f"group nodes (pool postings): {len(nodes):,}", flush=True)
        print(f"signature window: {window_label(window_chars)}", flush=True)

    shingles, shingle_meta = build_shingles(nodes, raw_root, out_dir,
                                            shingle_sidecar, verbose, window_chars)
    if set(shingles) != set(nodes):
        raise ShingleCacheError(
            f"the signature cache covers {len(shingles):,} postings but the group "
            f"map has {len(nodes):,} nodes; refusing to build a group map over a "
            f"cache that does not describe it")

    if verbose:
        print("template pairs (exact inverted index) ...", flush=True)
    pairs, pair_meta = template_pairs(shingles, min(TAU_ASSERT), verbose,
                                      min_shared=MIN_SHARED_SHINGLES)
    pairs.to_parquet(out_dir / "template_pairs.parquet", index=False)
    n_ladder_blind = int((pairs["jaccard"] >= LADDER_DEDUP_THRESHOLD).sum())

    # Per-tau template clusters, each computed independently (reported; only the
    # TAU_BUILD one is ever asserted, because higher-tau edge sets are subsets).
    cluster_cols = {}
    comp_sizes_per_tau = {}
    for tau in TAU_ASSERT:
        # The floor is part of the RULE, not of tau, so it applies at every
        # reported tau; without it these columns would describe a rule the
        # grouping does not use.
        edges = [(r.a, r.b) for r in pairs.itertuples(index=False)
                 if r.jaccard >= tau or int(r.n_shared) >= MIN_SHARED_SHINGLES]
        comp = _components(nodes, edges)
        col = f"template_cluster_t{int(round(tau * 100)):03d}"
        cluster_cols[col] = [comp[n] for n in nodes]
        sizes = pd.Series(list(comp.values())).value_counts()
        comp_sizes_per_tau[f"{tau:.2f}"] = {
            "n_verified_pairs": len(edges),
            "n_multi_posting_components": int((sizes > 1).sum()),
            "postings_in_multi_components": int(sizes[sizes > 1].sum()),
            "largest_component": int(sizes.max()) if len(sizes) else 0,
        }

    # ---- the union itself: employer-exact, employer-loose, template >= TAU_BUILD
    emp_edges = _equality_edges(keys, "employer_hash")
    loose_edges = _equality_edges(keys, "employer_hash_loose")
    # BOTH rules: the ratio, and the absolute-overlap floor that catches the
    # large shared block a long posting dilutes below tau. Ratio-only left
    # 16,522 real pairs sharing >=10 identical sampleable sentences ungrouped.
    tmpl_edges = sorted((r.a, r.b) for r in pairs.itertuples(index=False)
                        if r.jaccard >= TAU_BUILD
                        or int(r.n_shared) >= MIN_SHARED_SHINGLES)
    n_tmpl_by_ratio = int((pairs["jaccard"] >= TAU_BUILD).sum())
    n_tmpl_by_floor_only = int(((pairs["jaccard"] < TAU_BUILD)
                                & (pairs["n_shared"] >= MIN_SHARED_SHINGLES)).sum())

    uf = _UF(nodes)
    provenance: dict[str, set[str]] = {}
    for a, b in sorted(emp_edges):
        if uf.union(a, b):
            provenance.setdefault(uf.find(a), set()).add("employer_exact")
    for a, b in tmpl_edges:
        if uf.union(a, b):
            provenance.setdefault(uf.find(a), set()).add(f"template_j{TAU_BUILD:.2f}")

    # Loose edges applied LAST and guarded: they are the weakest evidence, so a
    # single one must not be allowed to collapse two already-large components.
    loose_only_roots: set[str] = set()
    loose_merges = 0
    largest_loose_merge = 0
    cap = MAX_LOOSE_MERGE_SHARE * len(nodes)
    for a, b in sorted(loose_edges):
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue
        combined = uf.size[ra] + uf.size[rb]
        largest_loose_merge = max(largest_loose_merge, combined)
        if combined > cap:
            raise GroupFeasibilityError(
                f"a single LOOSE employer edge ({a}, {b}) would merge components of "
                f"{uf.size[ra]} and {uf.size[rb]} postings ({combined} > "
                f"{cap:.1f} = {MAX_LOOSE_MERGE_SHARE:.0%} of {len(nodes)} pool "
                f"postings). Refusing to write: inspect the loose normalisation.")
        merged = uf.union(a, b)
        if merged:
            loose_merges += 1
            provenance.setdefault(uf.find(a), set()).add("employer_loose")
            loose_only_roots.add(uf.find(a))

    members: dict[str, list[str]] = {}
    for n in nodes:
        members.setdefault(uf.find(n), []).append(n)
    group_of, group_sha, prov_of = {}, {}, {}
    for root, group in members.items():
        label = "g:" + min(group)
        sha = hashlib.sha1("\n".join(sorted(group)).encode("utf-8")).hexdigest()[:16]
        prov = ",".join(sorted(provenance.get(root, {"singleton"}))) or "singleton"
        for n in group:
            group_of[n], group_sha[n], prov_of[n] = label, sha, prov

    # `employer_loose_status` rides along when posting_keys wrote it: it is the
    # only way an auditor can see which loose edges rest on a 1-2 character
    # residual (conservative for leakage, wrong for any per-employer count) and
    # which postings have no loose hash at all.
    _cols = ["posting_id", "stratum", "country_iso", "site", "employer_hash",
             "employer_hash_loose", "employer_known"]
    if "employer_loose_status" in keys.columns:
        _cols.append("employer_loose_status")
    out = keys[_cols].copy()
    out["group_id"] = [group_of[n] for n in nodes]
    out["group_members_sha1"] = [group_sha[n] for n in nodes]
    out["edge_provenance"] = [prov_of[n] for n in nodes]
    for col, values in cluster_cols.items():
        out[col] = values
    gsize = out.groupby("group_id", sort=False)["posting_id"].transform("size")
    out["n_postings_in_group"] = gsize.astype(int)
    verifiable = out.groupby("group_id", sort=False)["employer_known"].transform("all")
    out["group_employer_verifiable"] = verifiable.astype(bool)
    out = out.sort_values("posting_id", kind="stable").reset_index(drop=True)

    # ---- the guarantee, re-derived from the edge set rather than asserted
    # Every template edge at TAU_BUILD must be INSIDE one group. This is cheap
    # (it walks the edge list, not all pairs) and it is the statement the
    # manifest's claim_wording makes, so it is checked before the write.
    straddling = [(a, b) for a, b in tmpl_edges if group_of[a] != group_of[b]]
    if straddling:
        raise GroupFeasibilityError(
            f"{len(straddling)} template edges at Jaccard >= {TAU_BUILD} connect "
            f"postings in different groups, so the closure is broken; first few: "
            f"{straddling[:5]}")

    # ---- feasibility guards: refuse, never warn
    per_stratum_largest = {}
    for st, grp in out.groupby("stratum", sort=True):
        sizes = grp["group_id"].value_counts()
        largest = int(sizes.max()) if len(sizes) else 0
        per_stratum_largest[str(st)] = {
            "pool_postings": int(len(grp)), "largest_group": largest,
            "share": round(largest / max(len(grp), 1), 4),
            "guard_share": MAX_COMPONENT_SHARE_OF_STRATUM,
            "headroom_postings": int(MAX_COMPONENT_SHARE_OF_STRATUM * len(grp)) - largest}
        if largest > MAX_COMPONENT_SHARE_OF_STRATUM * len(grp):
            raise GroupFeasibilityError(
                f"stratum '{st}': largest leakage group holds {largest} of "
                f"{len(grp)} pool postings ({largest / len(grp):.1%} > "
                f"{MAX_COMPONENT_SHARE_OF_STRATUM:.0%}); a 20% test share is not "
                f"safely reachable. Refusing to write the group map.")

    sizes_all = out["group_id"].value_counts()
    manifest = {
        "split_spec_version": SPLIT_SPEC_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "node_scope": "sentence_pool postings only (see module docstring)",
        "tau_build": TAU_BUILD,
        "tau_asserted": [min(TAU_ASSERT)],
        "tau_reported": list(TAU_ASSERT),
        "min_shared_shingles": int(MIN_SHARED_SHINGLES),
        "template_rule": (
            f"jaccard >= {TAU_BUILD} OR n_shared >= {MIN_SHARED_SHINGLES} "
            "shared 5-word shingles. The floor exists because Jaccard is a "
            "ratio: a long posting dilutes a large shared block, and the "
            "ratio-only rule left 16,522 real pool pairs sharing >=10 "
            "byte-identical sampleable sentences in DIFFERENT groups."),
        "n_template_edges_by_ratio": n_tmpl_by_ratio,
        "n_template_edges_by_floor_only": n_tmpl_by_floor_only,
        # ---- the signature contract (finding F01 / F05 / F14)
        "shingle_window_chars": (None if window_chars is None else int(window_chars)),
        "shingle_window": window_label(window_chars),
        "shingle_max_chars": (None if window_chars is None else int(window_chars)),
        "shingle_max_chars_note": (
            "null means NO character window: the whole cleaned description is "
            "shingled. The old value 2500 was inherited from "
            "governance/configs/tabel_3_9_v1.json's raw-text near-duplicate "
            f"config (kept as textsig.LADDER_PARITY_WINDOW_CHARS={SHINGLE_MAX_CHARS} "
            "for ladder parity only) and left 126 real pairs at full-text "
            "Jaccard >= 0.30 in different groups — finding F01."),
        "shingle_words": SHINGLE_WORDS,
        "hash_fn": HASH_FN,
        "text_pipeline": text_pipeline(window_chars, SHINGLE_WORDS),
        "signature_params": signature_params(window_chars, SHINGLE_WORDS),
        "signature_params_fingerprint": params_fingerprint(
            signature_params(window_chars, SHINGLE_WORDS)),
        "shingles_source": shingle_meta["shingles_source"],
        "shingles_file": shingle_meta["shingles_file"],
        "shingles_fingerprint": shingle_meta["shingles_fingerprint"],
        "shingles_fingerprint_algorithm": (
            "sha256 over '<posting_id>\\t<textsig.signature_fingerprint("
            "signature)>\\n' lines, posting ids sorted ascending"),
        "signature_cache": shingle_meta["signature_cache"],
        "shingle_sidecar_decision": shingle_meta["sidecar_decision"],
        "shingle_sidecar_policy": (
            f"build_pool's posting_shingles.parquet is used ONLY if it carries a "
            f"matching fingerprint manifest, covers every pool posting, and "
            f"{SIDECAR_AUDIT_SAMPLE} seeded postings re-shingle identically from "
            f"the raw layer; posting-id coverage alone is never enough (finding "
            f"F05/F14). Stated residual: a single tampered row inside a sidecar "
            f"whose manifest was forged to match it is caught only with "
            f"probability {SIDECAR_AUDIT_SAMPLE}/n_pool, so the raw layer remains "
            f"authoritative — delete the sidecar to force a full rebuild."),
        "n_postings_without_signature": shingle_meta["n_postings_without_signature"],
        "mean_shingles_per_posting": shingle_meta["mean_shingles_per_posting"],
        "median_shingles_per_posting": shingle_meta["median_shingles_per_posting"],
        "max_shingles_per_posting": shingle_meta["max_shingles_per_posting"],
        # ---- candidate generation (finding F17)
        "candidate_generation": {
            **pair_meta,
            "template_edge_recall_at_tau_build": 1.0,
            "recall_basis": "exact enumeration, not an estimate — see "
                            "candidate_generation.completeness_basis",
            "superseded_method": {
                "method": "MinHashLSH + exact verification",
                "lsh_threshold": LSH_THRESHOLD,
                "num_perm": MINHASH_NUM_PERM,
                "lsh_bands_x_rows": "42 x 3 (datasketch's choice at 0.25/128)",
                "measured_recall_at_tau_0_30": SUPERSEDED_LSH_MEASURED_RECALL,
                "measured_how": (
                    "the shipped template_pairs.parquet held 4,799 of the 5,319 "
                    "pairs exact enumeration finds at Jaccard >= 0.30 under the "
                    "same 2,500-char window (recall 0.9022); re-running the same "
                    "MinHashLSH over full-text signatures produced 11,272 of "
                    "13,430 (recall 0.8393) in 380.8 s against 8.2 s for the "
                    "exact pass. Zero false positives either way, because every "
                    "candidate was Jaccard-verified. 16 of the real cross-group "
                    "pairs were recall misses, not truncation misses (F17)"),
            },
        },
        # kept at the top level for manifest-schema continuity; no longer used
        "lsh_threshold": LSH_THRESHOLD,
        "num_perm": MINHASH_NUM_PERM,
        "lsh_note": ("lsh_threshold / num_perm are recorded for continuity only; "
                     "candidate generation is exact and uses neither"),
        "n_nodes": len(nodes),
        "n_employer_exact_edges": len(emp_edges),
        "n_employer_loose_edges": len(loose_edges),
        "employer_loose_status_counts": (
            {str(k): int(v) for k, v in
             keys["employer_loose_status"].value_counts().items()}
            if "employer_loose_status" in keys.columns else None),
        "n_postings_without_a_loose_hash": int(
            keys["employer_hash_loose"].map(is_missing_hash).sum()),
        "loose_merge_count": loose_merges,
        "largest_loose_merge_postings": largest_loose_merge,
        "loose_merge_cap_postings": round(cap, 1),
        "n_groups_created_by_loose_only": len(loose_only_roots),
        "n_template_edges_at_tau_build": len(tmpl_edges),
        "n_template_edges_straddling_groups": len(straddling),
        "per_tau": comp_sizes_per_tau,
        "n_pairs_jaccard_ge_085_cleaned": n_ladder_blind,
        "ladder_dedup_note": (
            "The ladder's 0.85 near-dup pass ran on RAW text and on its own "
            "2500-char window; this layer shingles the FULL CLEANED text, so "
            "these pairs are residual near-duplicates the ladder could not see. "
            "0 eligible rows carry near_dup_minhash."),
        "n_groups": int(len(sizes_all)),
        "n_singleton_groups": int((sizes_all == 1).sum()),
        "largest_group_postings": int(sizes_all.max()) if len(sizes_all) else 0,
        "largest_group_share_of_pool": round(
            float(sizes_all.max()) / max(len(nodes), 1), 4) if len(sizes_all) else 0.0,
        "per_stratum_largest_group": per_stratum_largest,
        "n_groups_employer_unverifiable": int(
            out.loc[~out["group_employer_verifiable"], "group_id"].nunique()),
        "provenance": {
            "pool_file": pool_path.name, "pool_sha256": _sha256(pool_path),
            "posting_keys_file": keys_path.name, "posting_keys_sha256": _sha256(keys_path),
            "raw_batches": _raw_batch_provenance(raw_root),
        },
        "claim_wording": (
            "no employer (exact or loose hash) and no template pair at "
            f"FULL-cleaned-text Jaccard >= {TAU_BUILD} straddles any two parts. "
            "The template edge set is EXACT at that threshold (every pair sharing "
            "a shingle is enumerated and verified), so it is no longer a lower "
            "bound on its own threshold. Residuals that remain, and must be "
            f"stated: tau = {TAU_BUILD} is a chosen floor, so a true template "
            "pair at 0.25 is not grouped; employer_hash and employer_hash_loose "
            "are string hashes, so abbreviations, misspellings, subsidiaries and "
            "rebrands are missed; and a two-hop template bridge whose middle "
            "posting is NOT in the pool is not grouped, because the node set is "
            "the pool."),
        "claim_wording_history": [
            "v1 (superseded, inaccurate): 'no employer ... and no template "
            "cluster at cleaned-text Jaccard >= 0.3 straddles any two parts' — "
            "the signatures covered only the first 2,500 cleaned characters, so "
            "126 real pairs at full-text Jaccard >= 0.30 sat in different groups "
            "(finding F01), and MinHashLSH recall 0.9022 dropped a further 16 "
            "(finding F17)."],
    }
    out.to_parquet(out_dir / "leakage_groups.parquet", index=False)
    (out_dir / "leakage_groups_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if verbose:
        print(f"groups: {manifest['n_groups']:,} "
              f"({manifest['n_singleton_groups']:,} singleton), largest "
              f"{manifest['largest_group_postings']} postings "
              f"({manifest['largest_group_share_of_pool']:.2%} of pool)", flush=True)
        for st, d in sorted(per_stratum_largest.items()):
            flag = "  <-- WATCH" if d["share"] > 0.8 * MAX_COMPONENT_SHARE_OF_STRATUM else ""
            print(f"  {st}: largest {d['largest_group']}/{d['pool_postings']} "
                  f"= {d['share']:.2%} (guard {MAX_COMPONENT_SHARE_OF_STRATUM:.0%}, "
                  f"headroom {d['headroom_postings']} postings){flag}", flush=True)
        print(json.dumps(comp_sizes_per_tau, indent=2), flush=True)
        print(f"-> {out_dir / 'leakage_groups.parquet'}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(GS_DIR))
    ap.add_argument("--window-chars", default=None,
                    help="signature window in CLEANED characters; omit for the "
                         "full description (the only value the leakage layer "
                         "should ever use). Recorded and fingerprinted.")
    args = ap.parse_args()
    window = None if args.window_chars in (None, "", "none", "full") else int(args.window_chars)
    build_groups(out_dir=Path(args.out_dir), window_chars=window)


if __name__ == "__main__":
    main()
