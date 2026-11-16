"""Positive controls for the write-time invariant layer (`gold_standard.split_assert`).

Standalone runner (the repo has no pytest):

    .venv-ml\\Scripts\\python.exe tests\\test_split_assert.py

WHY THIS FILE EXISTS. The adversarial review of the split layer reproduced two
invariants that could not fail at all - I7, which skipped every posting the
shingle cache did not cover, and I16, which only checked that a `target` key had
an `achieved` key beside it - and it singled out the existing I7 control for
using two ~120-character strings with a hand-supplied shingle dict, so that it
exercised neither the truncation blind spot nor the cache-staleness path.

Everything here therefore runs against a REAL-SHAPED assignment: 240 postings /
480 sentences over 3 strata, 6 countries, 4 sites and 14 months, with
descriptions whose median cleaned length is above the 2,500-character shingle
cut (the real pool's median is 3,705), signatures computed by the production
`textsig` pipeline rather than hand-written, and a manifest built in the shape
`split.py` actually writes, marginals and all. Each invariant gets a control
that makes it FIRE on that data, plus the negative control that the clean frame
passes - a gate that always fails is as useless as one that never does.

No DATA/ path is read and no file is written.
"""
from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_standard import SHINGLE_MAX_CHARS, TAU_BUILD                   # noqa: E402
from gold_standard import split_assert as sa                             # noqa: E402
from gold_standard import textsig                                        # noqa: E402
from gold_standard.textsig import jaccard, shingles_from_clean_text      # noqa: E402

SEED = 20261116
N_POSTINGS = 240
AXES = ("stratum", "country_iso", "country_language", "posting_language",
        "site", "quarter", "occupation_group", "length_band")
# What the allocator aims at; the registered targets used by the manifest
# helpers are the baseline frame's REALISED sizes (`_registered()`), so an
# honest manifest is honest by construction and a dishonest one is built
# explicitly in the test that needs it.
TARGET_SHAPE = {"train": 288, "dev": 96, "test": 96}   # 60/20/20 of 480

_STATE: dict = {}


# --------------------------------------------------------------- the corpus
def _vocab(rng: np.random.Generator) -> list[str]:
    """A big vocabulary on purpose: unrelated postings must NOT look like
    template clones, or every pair trips I7 and the controls prove nothing."""
    base = [f"t{i:04d}" for i in range(4000)]
    rng.shuffle(base)
    return base


def _description(rng: np.random.Generator, vocab: list[str], n_chars: int) -> str:
    """A posting-length description: unique body plus a short shared header.

    Real cleaned descriptions in the pool run 3,705 characters at the median and
    7,004 at q0.90, which is the whole point of the truncation finding, so the
    control corpus is built at that scale rather than at sentence scale.
    """
    header = ("about the role we are a growing technology company and we are "
              "looking for an engineer to join the product team ")
    idx = rng.choice(len(vocab), size=max(n_chars // 6, 120), replace=False)
    body = " ".join(vocab[int(i)] for i in idx)
    return (header + body)[:n_chars]


def _sentence(rng: np.random.Generator, vocab: list[str], pid: str, k: int) -> str:
    idx = rng.choice(len(vocab), size=rng.integers(6, 24), replace=False)
    words = " ".join(vocab[int(i)] for i in idx)
    return f"the candidate {pid[-6:]} item {k} should {words}"


def _largest_remainder(total: int, weights: dict) -> dict:
    """Integer apportionment whose parts sum EXACTLY to `total`."""
    s = sum(weights.values())
    if s <= 0 or total <= 0:
        return {k: 0 for k in weights}
    raw = {k: total * v / s for k, v in weights.items()}
    out = {k: int(np.floor(v)) for k, v in raw.items()}
    rest = total - sum(out.values())
    order = sorted(weights, key=lambda k: (-(raw[k] - out[k]), str(k)))
    for k in order[:rest]:
        out[k] += 1
    return out


def _build_corpus() -> dict:
    rng = np.random.default_rng(SEED)
    vocab = _vocab(rng)
    strata = (["indonesia"] * 108 + ["asean"] * 48 + ["oecd"] * 84)
    countries = {"indonesia": ["ID"], "asean": ["SG", "MY"], "oecd": ["US", "GB", "DE"]}
    sites = {"indonesia": ["jobstreet", "indeed", "glints"],
             "asean": ["jobstreet", "indeed"], "oecd": ["indeed", "linkedin"]}
    languages = {"indonesia": ["id", "en"], "asean": ["en"], "oecd": ["en"]}
    occupations = ["software_developer", "web_developer", "system_analysis",
                   "devops", "UNRESOLVED"]

    # Employers (hence leakage groups) of 1 or 2 postings, WITHIN a stratum:
    # a group that spans strata cannot be placed by a per-stratum temporal cut,
    # which is the production regime's straddler problem and not what these
    # controls are about.
    emp_of: list[int] = [-1] * N_POSTINGS
    emp_id = 0
    for stratum in ("indonesia", "asean", "oecd"):
        idxs = [i for i in range(N_POSTINGS) if strata[i] == stratum]
        j = 0
        while j < len(idxs):
            if j + 1 < len(idxs) and emp_id % 3:
                emp_of[idxs[j]] = emp_of[idxs[j + 1]] = emp_id
                j += 2
            else:
                emp_of[idxs[j]] = emp_id
                j += 1
            emp_id += 1

    postings, employers = [], []
    for i in range(N_POSTINGS):
        stratum = strata[i]
        emp = emp_of[i]
        employers.append(emp)
        postings.append({
            "posting_id": f"u-{hashlib.sha1(f'p{i}'.encode()).hexdigest()[:16]}",
            "employer_idx": emp,
            "stratum": stratum,
            "country_iso": countries[stratum][i % len(countries[stratum])],
            "site": sites[stratum][i % len(sites[stratum])],
            "posting_language": languages[stratum][i % len(languages[stratum])],
            "occupation_group": occupations[i % len(occupations)],
            # 14 months of postings, one distinct timestamp each
            "date_posted_utc": pd.Timestamp("2025-11-29", tz="UTC")
            + pd.Timedelta(days=int(i * 1.7), hours=int(i % 11)),
            "n_chars": int(rng.integers(2600, 8200)),
        })
    for p in postings:
        p["description"] = _description(rng, vocab, p["n_chars"])
        p["quarter"] = f"{p['date_posted_utc'].year}Q{(p['date_posted_utc'].month - 1) // 3 + 1}"

    # --- planted TRUNCATION pair: two different employers, two different
    # groups, each with a unique 2,700-char prefix and then the SAME 5,000-char
    # template block. Full-text Jaccard is high, first-2,500-char Jaccard is ~0.
    shared_block = " ".join(f"s{i:04d}" for i in range(900))
    for slot, pid_i in (("A", 0), ("B", 1)):
        p = postings[pid_i]
        prefix = " ".join(f"{slot}{i:04d}" for i in range(520))
        p["description"] = (prefix + " " + shared_block)
        p["n_chars"] = len(p["description"])
    postings[0]["employer_idx"] = 900          # distinct employers on purpose
    postings[1]["employer_idx"] = 901

    emp_hash = {e: hashlib.sha1(f"emp{e}".encode()).hexdigest()[:16]
                for e in {p["employer_idx"] for p in postings}}
    loose_hash = {e: (hashlib.sha1(f"loose{e}".encode()).hexdigest()[:16]
                      if e % 7 else "")      # ~14% legitimately empty (F03)
                  for e in {p["employer_idx"] for p in postings}}
    for p in postings:
        e = p["employer_idx"]
        p["employer_hash"] = emp_hash[e] if e % 23 else ""   # ~4% employerless
        p["employer_hash_loose"] = loose_hash[e] if p["employer_hash"] else ""
        p["employer_known"] = bool(p["employer_hash"])
        p["group_id"] = "g:" + min(q["posting_id"] for q in postings
                                   if q["employer_idx"] == e) \
            if p["employer_hash"] else "g:" + p["posting_id"]
        p["template_cluster_t030"] = p["posting_id"]

    # --- group-atomic part assignment, employerless groups confined to train
    groups: dict[str, list[dict]] = {}
    for p in postings:
        groups.setdefault(p["group_id"], []).append(p)
    order = sorted(groups)
    rng2 = np.random.default_rng(SEED + 1)
    rng2.shuffle(order)
    want = dict(TARGET_SHAPE)
    part_of, placed = {}, {"train": 0, "dev": 0, "test": 0}
    planted_groups = {p["group_id"] for p in postings[:2]}
    forced_train = [g for g in order
                    if not all(p["employer_known"] for p in groups[g])
                    or g in planted_groups]
    for g in forced_train:
        part_of[g] = "train"
        placed["train"] += 2 * len(groups[g])
    for g in order:
        if g in part_of:
            continue
        w = 2 * len(groups[g])
        for part in ("test", "dev", "train"):
            if placed[part] + w <= want[part]:
                part_of[g] = part
                placed[part] += w
                break
        else:
            part_of[g] = "train"
            placed["train"] += w

    rows = []
    for p in postings:
        for k in range(2):
            rows.append({
                "sentence_id": f"{p['posting_id']}#{k:03d}",
                "posting_id": p["posting_id"],
                "group_id": p["group_id"],
                "part": part_of[p["group_id"]],
                "exclusion_reason": "",
                "employer_hash": p["employer_hash"],
                "employer_hash_loose": p["employer_hash_loose"],
                "employer_known": p["employer_known"],
                "template_cluster_t030": p["template_cluster_t030"],
                "sentence_text": _sentence(rng, vocab, p["posting_id"], k),
                "stratum": p["stratum"],
                "country_iso": p["country_iso"],
                "country_language": f"{p['country_iso']}_{p['posting_language']}",
                "posting_language": p["posting_language"],
                "site": p["site"],
                "quarter": p["quarter"],
                "occupation_group": p["occupation_group"],
                "date_posted_utc": p["date_posted_utc"],
                "date_parsed": True,
                "length_band": "short" if p["n_chars"] < 4000 else "long",
            })
    df = pd.DataFrame(rows)
    texts = {p["posting_id"]: p["description"] for p in postings}
    # `textsig`'s leakage window is now the WHOLE cleaned text (finding F01), so
    # the truncated cache is built explicitly at the ladder-parity window - it is
    # the cache the shipped artifacts were built with and the one the truncation
    # control needs.
    cache_trunc = {pid: shingles_from_clean_text(t, max_chars=SHINGLE_MAX_CHARS)
                   for pid, t in texts.items()}
    cache_full = {pid: shingles_from_clean_text(t) for pid, t in texts.items()}
    return {"df": df, "postings": postings, "texts": texts,
            "cache_trunc": cache_trunc, "cache_full": cache_full,
            "registered": {p: int((df["part"] == p).sum()) for p in sa.PARTS},
            "planted_pair": (postings[0]["posting_id"], postings[1]["posting_id"])}


def _corpus() -> dict:
    if "corpus" not in _STATE:
        _STATE["corpus"] = _build_corpus()
    return _STATE["corpus"]


def _frame() -> pd.DataFrame:
    return _corpus()["df"].copy(deep=True)


def _registered() -> dict:
    """The baseline frame's realised sizes, used as the REGISTERED targets."""
    return dict(_corpus()["registered"])


def _cache() -> dict:
    """The production-shaped cache: signatures over the FULL cleaned text."""
    return dict(_corpus()["cache_full"])


def _cache_trunc() -> dict:
    """The pre-F01 cache: signatures over the first 2,500 cleaned characters."""
    return dict(_corpus()["cache_trunc"])


def _meta(cache: dict, *, window=None) -> dict:
    """The group layer's declaration, as `resolve_shingle_meta` returns it."""
    return {"fingerprint": sa.shingle_cache_fingerprint(cache),
            "fingerprint_field": "group_layer.shingles_fingerprint",
            "source": "raw_batches", "source_field": "group_layer.shingles_source",
            "block": textsig.cache_fingerprint(cache, window_chars=window),
            "block_field": "group_layer.signature_cache",
            "window_chars": window, "shingle_words": 5}


def _i7(df, cache, *, meta=None, require=True):
    return sa.i7_direct_pairwise_jaccard(
        df, cache, shingle_meta=meta if meta is not None else _meta(cache),
        require_fingerprint=require)


def _plant_pair_across_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Put the planted template pair in train and test, in different groups."""
    a, b = _corpus()["planted_pair"]
    df.loc[df["posting_id"] == a, "part"] = "train"
    df.loc[df["posting_id"] == b, "part"] = "test"
    return df


# ------------------------------------------------------------- provenance etc
def _sha(tag: str) -> str:
    return hashlib.sha256(tag.encode()).hexdigest()


def _provenance() -> dict:
    return {
        "sample_file": "main_1200.parquet", "sample_sha256": _sha("sample"),
        "pool_file": "sentence_pool.parquet", "pool_sha256": _sha("pool"),
        "posting_keys_file": "posting_keys.parquet",
        "posting_keys_sha256": _sha("keys"),
        "leakage_groups_file": "leakage_groups.parquet",
        "leakage_groups_sha256": _sha("groups"),
        "shingles_file": "shingles.parquet", "shingles_sha256": _sha("shingles"),
        "flags_file": "record_flags_20261115T164444Z.parquet",
        "raw_batches": [{"batch": f"batch-{i:03d}", "rows": 1000 + i,
                         "sha256": _sha(f"batch{i}")} for i in range(25)],
        "pool_append_stable": False,
    }


def _group_layer(cache: dict, *, window=None) -> dict:
    """`leakage_groups_manifest.json`'s shingle block, as groups.py writes it."""
    block = textsig.cache_fingerprint(cache, window_chars=window)
    return {"tau_build": TAU_BUILD, "shingle_words": 5,
            "hash_fn": textsig.HASH_FN, "shingles_source": "raw_batches",
            "shingles_file": "shingles.parquet",
            "signature_params": block["signature_params"],
            "signature_cache": block,
            "shingles_fingerprint": block["content_sha256"]}


def _quarantine() -> tuple[dict, dict]:
    pilot = {"sentence_ids": {f"pilot#{i}" for i in range(100)},
             "posting_ids": {f"u-pilot{i}" for i in range(50)},
             "group_ids": {f"g:u-pilot{i}" for i in range(50)}}
    practice = {"sentence_ids": {f"prac#{i}" for i in range(10)},
                "posting_ids": {f"u-prac{i}" for i in range(5)},
                "group_ids": {f"g:u-prac{i}" for i in range(5)}}
    return pilot, practice


def _marginals(df: pd.DataFrame, part_totals: dict, *, close: bool = True,
               not_applicable=()) -> dict:
    """The manifest's marginals block, in `split.marginals_block`'s shape.

    `close=True` apportions each part's total across that axis's values with a
    two-stage largest remainder, so the target column sums EXACTLY to the part
    total. `close=False` reproduces the defect the review measured: one
    independent rounding per axis VALUE, so the column misses the part size.
    """
    out = {}
    assigned = df[df["part"].isin(sa.PARTS)]
    for axis in AXES:
        per_part = {}
        shares = assigned[axis].astype(str).value_counts().to_dict()
        for part in sa.PARTS:
            ach = df.loc[df["part"] == part, axis].astype(str).value_counts().to_dict()
            total = int(part_totals.get(part, 0))
            if close:
                tgt = _largest_remainder(total, shares)
            else:
                n = max(len(assigned), 1)
                tgt = {v: int(round(total * c / n)) + 1 for v, c in shares.items()}
            values = sorted(set(ach) | set(tgt))
            target = {v: int(tgt.get(v, 0)) for v in values}
            achieved = {v: int(ach.get(v, 0)) for v in values}
            block = {"target": target, "achieved": achieved,
                     "delta": {v: achieved[v] - target[v] for v in values},
                     "tv_distance": 0.0,
                     "target_total": int(sum(target.values())),
                     "achieved_total": int(sum(achieved.values())),
                     "apportioned_from_part_total": total,
                     "target_column_closes": sum(target.values()) == total}
            if axis in not_applicable:
                block["not_applicable"] = True
                block["not_applicable_reason"] = "the regime partitions on this axis"
            per_part[part] = block
        out[axis] = per_part
    return out


def _manifest(df: pd.DataFrame, *, exact: bool = True, registered: dict | None = None,
              capacity: dict | None = None, close_columns: bool = True) -> dict:
    """A manifest in the shape `split.py` writes, honest by construction."""
    registered = dict(registered or _registered())
    achieved = {p: int((df["part"] == p).sum()) for p in sa.PARTS}
    capacity = dict(capacity or achieved)
    sizes = {p: {"target": registered[p], "achieved": achieved[p],
                 "delta": achieved[p] - registered[p]} for p in sa.PARTS}
    unmet = [f"{p}: registered target {v['target']} achieved {v['achieved']} "
             f"(delta {v['delta']:+d})" for p, v in sizes.items() if v["delta"]]
    cache = _cache()
    return {
        "identity": {"regime_id": "grouped_random", "created_utc": "2026-11-16T00:00:00Z"},
        "registry": {"registry_entry_sha256": _sha("entry"), "frozen": True,
                     "body": {"targets": registered, "runnable": True}},
        "provenance": _provenance(),
        "group_layer": _group_layer(cache),
        "targets": dict(registered),
        "achieved": achieved,
        "sizes": sizes,
        "regime_capacity": {"capacity": capacity,
                            "equals_registered_targets": capacity == registered},
        "exact_sizes_met": (all(v["delta"] == 0 for v in sizes.values())
                            if exact else None),
        "exactness_claimed": exact,
        "marginals": _marginals(df, capacity, close=close_columns),
        "unmet_targets": unmet,
    }


def _frames(df: pd.DataFrame) -> dict:
    blind = ["sentence_id", "posting_id", "group_id", "part", "stratum",
             "country_iso", "site", "posting_language", "quarter",
             "sentence_text", "occupation_group", "country_language",
             "length_band"]
    return {"assignment": df[["sentence_id", "posting_id", "group_id", "part",
                              "exclusion_reason"]].copy(),
            "train": df[df["part"] == "train"].copy(),
            "dev": df[df["part"] == "dev"].copy(),
            "test_blind": df.loc[df["part"] == "test", blind].copy()}


def _run(df: pd.DataFrame, *, cache=None, meta=None, regime="grouped_random",
         regime_meta=None, manifest=None, exact=True) -> dict:
    cache = _cache() if cache is None else cache
    pilot, practice = _quarantine()
    manifest = _manifest(df, exact=exact) if manifest is None else manifest
    return sa.run_invariants(
        df, input_ids=df["sentence_id"].tolist(),
        targets={p: int((df["part"] == p).sum()) for p in sa.PARTS},
        exact=exact, shingles=cache, pilot=pilot, practice=practice,
        allowed_reasons=("pilot_group", "practice_group", "unverifiable_employer",
                         "date_unparseable", "straddles_temporal_cut_dropped",
                         "heldout_country_group"),
        provenance=_provenance(), regime=regime, regime_meta=regime_meta or {},
        manifest=manifest, frames=_frames(df),
        shingle_meta=meta if meta is not None else _meta(cache))


# =========================================================== the control data
def test_the_control_corpus_is_real_shaped():
    """Guard the guard: if the fixture shrinks back to toy scale, the controls
    below stop proving what they claim, so the shape is asserted."""
    c = _corpus()
    df = c["df"]
    lengths = pd.Series({p: len(t) for p, t in c["texts"].items()})
    assert len(df) == 480 and df["posting_id"].nunique() == N_POSTINGS
    assert lengths.median() > SHINGLE_MAX_CHARS, lengths.median()
    assert (lengths > SHINGLE_MAX_CHARS).mean() > 0.6
    assert df["stratum"].nunique() == 3 and df["country_iso"].nunique() >= 5
    assert df["site"].nunique() >= 4 and df["quarter"].nunique() >= 4
    sig = pd.Series({p: len(s) for p, s in c["cache_full"].items()})
    assert sig.median() > 300, sig.median()
    # the parts are non-empty and group-atomic in the clean baseline
    assert all((df["part"] == p).sum() > 0 for p in sa.PARTS)


def test_clean_real_shaped_split_passes_every_invariant():
    """The negative control. A gate that cannot pass is not a gate."""
    res = _run(_frame())
    failed = {k: v for k, v in res.items() if not v["passed"]}
    assert not failed, {k: v.get("failure_reasons") or v.get("problems") or v
                        for k, v in failed.items()}
    assert len(res) == 17


# ======================================================================= I7
def test_i7_fires_on_a_real_shaped_template_straddle():
    """A template bridge whose shared block starts after character 2,600, over a
    real-shaped corpus - not two ~120-character strings with a hand-written
    shingle dict, which is what the review found the old control doing."""
    df = _plant_pair_across_parts(_frame())
    full = _cache()
    a, b = _corpus()["planted_pair"]
    assert jaccard(full[a], full[b]) >= TAU_BUILD
    assert min(len(_corpus()["texts"][a]), len(_corpus()["texts"][b])) > 8000
    res = _i7(df, full)
    assert not res["passed"] and res["violations"]
    assert res["n_pairs_compared"] > 1000
    assert res["measures_full_text"] is True
    assert {res["violations"][0]["a"], res["violations"][0]["b"]} == {a, b}


def test_i7_refuses_a_prefix_cache_and_names_the_blind_spot():
    """The blind spot the review called FATAL, reproduced as a control.

    The planted pair is a template pair over the full cleaned text and is
    INVISIBLE over the first 2,500 characters: measured here, 0.46 against 0.0.
    I7 cannot find such a pair in a truncated cache, so it no longer accepts
    one - a Jaccard over prefixes is not the leakage claim, and `textsig`
    removed the window for exactly this reason.
    """
    c = _corpus()
    a, b = c["planted_pair"]
    full_j = jaccard(c["cache_full"][a], c["cache_full"][b])
    trunc_j = jaccard(c["cache_trunc"][a], c["cache_trunc"][b])
    assert full_j >= TAU_BUILD > trunc_j, (full_j, trunc_j)
    assert trunc_j == 0.0

    df = _plant_pair_across_parts(_frame())
    trunc_cache = _cache_trunc()
    res = _i7(df, trunc_cache, meta=_meta(trunc_cache, window=SHINGLE_MAX_CHARS))
    # the straddle is invisible in that cache ...
    assert not res["violations"]
    # ... and the cache itself is refused, with the window named
    assert not res["passed"]
    assert res["measures_full_text"] is False
    assert "PREFIX" in res["signature_scope"]
    assert res["signature_window_chars"] == SHINGLE_MAX_CHARS
    assert any("FINITE window" in r for r in res["failure_reasons"])
    # textsig's own parameter check agrees, independently of the window rule:
    # a cache built under one window cannot be reused under another
    assert any("window" in p for p in res["cache_fingerprint_block_problems"])
    # and the same frame against the full-text cache fires on the pair itself
    assert _i7(df, _cache())["violations"]


def test_i7_refuses_a_hand_edited_fingerprint_block():
    """The fingerprint block digests its own fields, so an operator who edits
    the manifest to make a stale cache look current is caught."""
    df = _frame()
    cache = _cache()
    meta = _meta(cache)
    forged = copy.deepcopy(meta)
    forged["block"]["n_postings"] = 999
    res = _i7(df, cache, meta=forged)
    assert not res["passed"]
    assert res["cache_fingerprint_block_problems"]


def test_i7_fails_closed_when_the_cache_does_not_cover_the_sample():
    """The reproduction from finding F04, as a test.

    Removing the sampled postings' rows from the cache used to give
    `passed: True, n_pairs_compared: 0, max_cross_part_jaccard: 0.0`, and the
    split was written with that 0.0 published as evidence of disjointness.
    """
    df = _frame()
    cache = _cache()
    victims = sorted(df.loc[df["part"] == "test", "posting_id"].unique())[:5]
    for pid in victims:
        del cache[pid]
    res = _i7(df, cache)
    assert not res["passed"]
    assert set(res["postings_absent_from_cache"]) & set(victims)
    assert res["n_postings_without_signature"] >= len(victims)

    # the extreme form: every sampled posting removed -> zero pairs
    empty_cache = {k: v for k, v in _cache().items()
                   if k not in set(df["posting_id"])}
    res2 = _i7(df, empty_cache)
    assert not res2["passed"] and res2["n_pairs_compared"] == 0
    assert any("ABSENT" in r for r in res2["failure_reasons"])


def test_i7_fails_closed_on_empty_signatures():
    """The all-empty sidecar attack: right posting ids, no signatures."""
    df = _frame()
    cache = {pid: np.zeros(0, dtype=np.uint64) for pid in _cache()}
    res = _i7(df, cache, meta=_meta(cache))
    assert not res["passed"] and res["n_pairs_compared"] == 0
    assert res["postings_with_empty_signature"]
    assert any("EMPTY" in r for r in res["failure_reasons"])


def test_i7_fails_closed_on_a_stale_cache_fingerprint():
    """The staleness path: the group map was built from one cache, the split is
    being checked against another. The ids line up; the signatures do not."""
    df = _frame()
    good = _cache()
    declared = _meta(good)                      # fingerprint of the REAL cache
    garbage = {pid: shingles_from_clean_text(f"unrelated planted text {pid} " * 40)
               for pid in good}
    res = sa.i7_direct_pairwise_jaccard(df, garbage, shingle_meta=declared)
    assert not res["passed"]
    assert res["cache_fingerprint_status"] == "mismatch"
    assert any("MISMATCH" in r for r in res["failure_reasons"])
    # ... and the honest pairing passes
    assert sa.i7_direct_pairwise_jaccard(df, good, shingle_meta=declared)["passed"]


def test_i7_fails_closed_when_no_fingerprint_is_declared():
    df = _frame()
    res = sa.i7_direct_pairwise_jaccard(df, _cache(), shingle_meta={})
    assert not res["passed"]
    assert res["cache_fingerprint_status"] == "undeclared"
    assert res["cache_fingerprint_computed"]
    assert any("fingerprint" in r for r in res["failure_reasons"])


def test_i7_fails_on_a_blank_posting_id():
    df = _frame()
    df.loc[df.index[0], "posting_id"] = None
    res = _i7(df, _cache())
    assert not res["passed"] and res["blank_posting_ids"] == 1


def test_shingle_cache_fingerprint_is_stable_and_sensitive():
    cache = _cache()
    assert sa.shingle_cache_fingerprint(cache) == sa.shingle_cache_fingerprint(dict(cache))
    dropped = {k: v for k, v in cache.items() if k != sorted(cache)[0]}
    assert sa.shingle_cache_fingerprint(dropped) != sa.shingle_cache_fingerprint(cache)
    mutated = dict(cache)
    first = sorted(cache)[0]
    mutated[first] = np.array(sorted(set(cache[first].tolist()) | {12345}),
                              dtype=np.uint64)
    assert sa.shingle_cache_fingerprint(mutated) != sa.shingle_cache_fingerprint(cache)


def test_content_digest_is_byte_identical_to_textsigs():
    """The value the group layer RECORDS must be the value I7 RECOMPUTES.
    `textsig.cache_content_fingerprint` and
    `split_assert.shingle_cache_fingerprint` are two implementations of one
    definition, so they are asserted equal here as well as in the group
    layer's own suite."""
    cache = _cache()
    assert sa.shingle_cache_fingerprint(cache) == \
        textsig.cache_content_fingerprint(cache)
    assert sa.shingle_cache_fingerprint(cache) == \
        textsig.cache_fingerprint(cache)["content_sha256"]


def test_resolve_shingle_meta_reads_the_group_layer_manifest():
    cache = _cache()
    meta = sa.resolve_shingle_meta(_provenance(), _group_layer(cache))
    assert meta["fingerprint"] == sa.shingle_cache_fingerprint(cache)
    assert meta["fingerprint_field"] == "group_layer.shingles_fingerprint"
    assert meta["source"] == "raw_batches"
    assert sa.resolve_shingle_meta({}, {}) == {}


# ====================================================================== I16
def test_i16_passes_on_an_honest_manifest():
    res = sa.i16_reporting(_manifest(_frame()))
    assert res["passed"], res["problems"] + res["marginal_column_problems"]


def test_i16_fires_when_the_target_is_the_achieved_assignment():
    """The defect the invariant was written for, in the mirror form the review
    reproduced: registered 288/96/96, a regime that placed 287/96/84, and a
    manifest whose `target` is the placement - so every delta is 0,
    `exact_sizes_met` is true and `unmet_targets` is empty."""
    df = _frame()
    m = _manifest(df, exact=False, registered=_registered())
    achieved = {"train": 300, "dev": 96, "test": 84}
    m["achieved"] = achieved
    m["sizes"] = {p: {"target": achieved[p], "achieved": achieved[p], "delta": 0}
                  for p in sa.PARTS}
    m["targets"] = dict(achieved)
    m["exact_sizes_met"] = True
    m["unmet_targets"] = []
    res = sa.i16_reporting(m)
    assert not res["passed"]
    assert set(res["parts_whose_target_came_from_the_assignment"]) == {"train", "test"}
    joined = " ".join(res["problems"])
    assert "derived from the achieved assignment" in joined
    assert "exactness_claimed is false" in joined
    assert "unmet_targets does not name it" in joined


def test_i16_fires_when_the_target_is_the_allocator_capacity():
    df = _frame()
    capacity = {"train": 280, "dev": 96, "test": 90}
    m = _manifest(df, exact=True, capacity=capacity)
    m["sizes"] = {p: {"target": capacity[p],
                      "achieved": m["sizes"][p]["achieved"],
                      "delta": m["sizes"][p]["achieved"] - capacity[p]}
                  for p in sa.PARTS}
    res = sa.i16_reporting(m)
    assert not res["passed"]
    assert any("regime_capacity" in p for p in res["problems"])


def test_i16_fires_on_an_inconsistent_delta():
    m = _manifest(_frame())
    m["sizes"]["test"]["delta"] = 0
    m["sizes"]["test"]["achieved"] = m["sizes"]["test"]["target"] + 7
    res = sa.i16_reporting(m)
    assert not res["passed"]
    assert any("delta" in p for p in res["problems"])


def test_i16_fires_on_a_target_without_an_achieved_counterpart():
    m = _manifest(_frame())
    del m["sizes"]["dev"]["achieved"]
    res = sa.i16_reporting(m)
    assert not res["passed"] and "sizes.dev" in res["targets_without_achieved"]


def test_i16_fires_when_a_marginal_target_column_does_not_close():
    """Finding F08: one independent largest-remainder call per axis VALUE, so
    the per-part target column does not sum to the part size and every reported
    delta and tv_distance carries the residue."""
    df = _frame()
    m = _manifest(df, close_columns=False)
    res = sa.i16_reporting(m)
    assert not res["passed"]
    assert res["n_marginal_column_problems"] > 0
    assert any("arithmetically unmeetable" in p
               for p in res["marginal_column_problems"])


def test_i16_accepts_a_marginal_block_marked_not_applicable():
    df = _frame()
    m = _manifest(df)
    m["marginals"] = _marginals(df, {p: int((df["part"] == p).sum()) for p in sa.PARTS},
                                close=False, not_applicable=("quarter",))
    res = sa.i16_reporting(m)
    # the quarter blocks are skipped (and recorded); the others still fire
    assert "marginals.quarter.test" in res["marginal_blocks_not_applicable"]
    assert not any(".quarter." in p for p in res["marginal_column_problems"])


def test_i16_fires_without_the_registered_targets():
    m = _manifest(_frame())
    m["registry"]["body"].pop("targets")
    res = sa.i16_reporting(m)
    assert not res["passed"]
    assert any("registry.body.targets" in p for p in res["problems"])
    assert not sa.i16_reporting({})["passed"]


def test_i16_fires_when_exact_sizes_met_contradicts_the_register():
    df = _frame()
    m = _manifest(df, exact=True, registered={"train": 300, "dev": 96, "test": 96})
    m["exact_sizes_met"] = True
    res = sa.i16_reporting(m)
    assert not res["passed"]
    assert res["exact_sizes_met_against_register"] is False


# ======================================================== I1-I6, I8-I15, I17
def test_i1_fires_on_a_missing_row_a_duplicate_and_a_bad_reason():
    df = _frame()
    ids = df["sentence_id"].tolist()
    assert sa.i1_accounting(df, ids, ("date_unparseable",))["passed"]
    assert not sa.i1_accounting(df.iloc[1:], ids, ("date_unparseable",))["passed"]
    dupe = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert not sa.i1_accounting(dupe, ids, ("date_unparseable",))["passed"]
    bad = df.copy()
    bad.loc[bad.index[0], "part"] = "excluded"
    bad.loc[bad.index[0], "exclusion_reason"] = "because_i_said_so"
    assert not sa.i1_accounting(bad, ids, ("date_unparseable",))["passed"]
    stray = df.copy()
    stray.loc[stray.index[0], "exclusion_reason"] = "date_unparseable"
    assert not sa.i1_accounting(stray, ids, ("date_unparseable",))["passed"]


def test_i1_fails_closed_on_an_empty_frame_and_an_empty_input_list():
    df = _frame()
    assert not sa.i1_accounting(df.iloc[0:0], [], ("date_unparseable",))["passed"]
    assert not sa.i1_accounting(df, [], ("date_unparseable",))["passed"]
    empty_part = df.copy()
    empty_part.loc[empty_part["part"] == "dev", "part"] = "train"
    res = sa.i1_accounting(empty_part, df["sentence_id"].tolist(),
                           ("date_unparseable",))
    assert not res["passed"] and res["empty_parts"] == ["dev"]


def test_i2_and_i3_fire_on_a_real_group_straddle_and_on_null_ids():
    df = _frame()
    victim = df.loc[df["part"] == "test"].iloc[0]
    donor = df.loc[df["part"] == "train"].iloc[0]
    df.loc[df["sentence_id"] == victim["sentence_id"], "group_id"] = donor["group_id"]
    assert not sa.i2_group_disjoint(df)["passed"]
    df2 = _frame()
    df2.loc[df2["sentence_id"] == victim["sentence_id"], "posting_id"] = donor["posting_id"]
    assert not sa.i3_posting_disjoint(df2)["passed"]

    nulled = _frame()
    nulled["group_id"] = np.nan
    res = sa.i2_group_disjoint(nulled)
    assert not res["passed"] and res["n_missing_values"] == len(nulled)
    assert res["vacuity_guard"]


def test_i4_and_i5_fire_on_employer_straddles_and_on_dead_columns():
    df = _frame()
    victim = df.loc[df["part"] == "test"].iloc[0]["sentence_id"]
    donor = df.loc[(df["part"] == "train") & (df["employer_hash"] != "")].iloc[0]
    df.loc[df["sentence_id"] == victim, "employer_hash"] = donor["employer_hash"]
    assert not sa.i4_employer_exact_disjoint(df)["passed"]

    df2 = _frame()
    loose_donor = df2.loc[(df2["part"] == "train")
                          & (df2["employer_hash_loose"] != "")].iloc[0]
    df2.loc[df2["sentence_id"] == victim, "employer_hash_loose"] = \
        loose_donor["employer_hash_loose"]
    assert not sa.i5_employer_loose_disjoint(df2)["passed"]

    # a NaN hash is MISSING, not the shared value "nan" (astype(str) let it
    # through as a value before)
    nan_frame = _frame()
    nan_frame["employer_hash"] = np.nan
    res = sa.i4_employer_exact_disjoint(nan_frame)
    assert not res["passed"] and res["coverage"]["n_nonempty"] == 0
    assert res["vacuity_guard"]
    dead_loose = _frame()
    dead_loose["employer_hash_loose"] = ""
    res5 = sa.i5_employer_loose_disjoint(dead_loose)
    assert not res5["passed"] and res5["loose_coverage_share"] == 0.0


def test_i5_reports_the_coverage_its_claim_is_made_over():
    res = sa.i5_employer_loose_disjoint(_frame())
    assert res["passed"]
    # the fixture plants legitimately empty loose hashes (finding F03), so the
    # claim must be published with its coverage rather than as "every boundary"
    assert 0.0 < res["loose_coverage_share"] < 1.0


def test_i6_fires_on_a_cluster_straddle_and_on_a_missing_column():
    df = _frame()
    victim = df.loc[df["part"] == "test"].iloc[0]["sentence_id"]
    donor = df.loc[df["part"] == "train"].iloc[0]["template_cluster_t030"]
    df.loc[df["sentence_id"] == victim, "template_cluster_t030"] = donor
    assert not sa.i6_template_cluster_disjoint(df)["passed"]
    assert not sa.i6_template_cluster_disjoint(
        _frame().drop(columns=["template_cluster_t030"]))["passed"]
    nulled = _frame()
    nulled["template_cluster_t030"] = None
    assert not sa.i6_template_cluster_disjoint(nulled)["passed"]


def test_i8_fires_on_exact_normalised_and_near_duplicate_sentences():
    df = _frame()
    t_idx = df.index[df["part"] == "test"][0]
    train_text = df.loc[df.index[df["part"] == "train"][0], "sentence_text"]
    df.loc[t_idx, "sentence_text"] = train_text
    assert not sa.i8_sentence_text_disjoint(df)["passed"]

    df2 = _frame()
    df2.loc[t_idx, "sentence_text"] = train_text.upper() + " !!!"
    res = sa.i8_sentence_text_disjoint(df2)
    assert not res["passed"] and res["normalised_overlaps"]

    df3 = _frame()
    df3.loc[t_idx, "sentence_text"] = train_text + " tambahan"
    res3 = sa.i8_sentence_text_disjoint(df3)
    assert not res3["passed"] and res3["n_near_duplicate_pairs"] >= 1

    blank = _frame()
    blank.loc[t_idx, "sentence_text"] = None
    assert not sa.i8_sentence_text_disjoint(blank)["passed"]


def test_i9_fires_on_unverifiable_rows_and_on_the_dtype_traps():
    df = _frame()
    t_idx = df.index[df["part"] == "test"][0]
    df.loc[t_idx, "employer_known"] = False
    res = sa.i9_dev_test_verifiable(df)
    assert not res["passed"] and res["dev_test_employer_verifiable_share"] < 1.0

    # astype(bool) used to read BOTH of these as True
    nan_frame = _frame()
    nan_frame["employer_known"] = nan_frame["employer_known"].astype(object)
    nan_frame.loc[t_idx, "employer_known"] = np.nan
    res_nan = sa.i9_dev_test_verifiable(nan_frame)
    assert not res_nan["passed"] and res_nan["non_boolean_employer_known_values"]

    str_frame = _frame()
    str_frame["employer_known"] = str_frame["employer_known"].astype(object)
    str_frame.loc[t_idx, "employer_known"] = "False"
    assert not sa.i9_dev_test_verifiable(str_frame)["passed"]

    # the contract's own words: a dev/test posting must have a NON-EMPTY hash
    hash_frame = _frame()
    hash_frame.loc[t_idx, "employer_hash"] = ""
    res_hash = sa.i9_dev_test_verifiable(hash_frame)
    assert not res_hash["passed"]
    assert res_hash["n_dev_test_rows_without_employer_hash"] == 1

    # and an empty dev+test is not a 1.0 share, it is nothing to measure
    train_only = _frame()
    train_only["part"] = "train"
    assert not sa.i9_dev_test_verifiable(train_only)["passed"]


def test_i10_fires_on_a_quarantine_hit_and_on_an_empty_reference():
    df = _frame()
    pilot, practice = _quarantine()
    assert sa.i10_quarantine(df, pilot, practice)["passed"]
    leaked = df.copy()
    leaked.loc[leaked.index[0], "sentence_id"] = sorted(pilot["sentence_ids"])[0]
    assert not sa.i10_quarantine(leaked, pilot, practice)["passed"]
    group_leak = df.copy()
    group_leak.loc[group_leak.index[0], "group_id"] = sorted(pilot["group_ids"])[0]
    assert not sa.i10_quarantine(group_leak, pilot, practice)["passed"]

    empty = {"sentence_ids": set(), "posting_ids": set(), "group_ids": set()}
    res = sa.i10_quarantine(df, empty, practice)
    assert not res["passed"], "an empty quarantine reference is a vacuous pass"
    assert any("EMPTY" in p for p in res["reference_problems"])
    assert not sa.i10_quarantine(df, {"sentence_ids": {"x"}}, practice)["passed"]


def test_i11_fires_on_an_oversize_group_and_on_a_missing_quota():
    df = _frame()
    targets = {p: int((df["part"] == p).sum()) for p in sa.PARTS}
    assert sa.i11_oversize_groups(df, targets)["passed"]
    big = df.copy()
    test_rows = big.index[big["part"] == "test"]
    big.loc[test_rows[:40], "group_id"] = "g:one-huge-group"
    res = sa.i11_oversize_groups(big, targets)
    assert not res["passed"] and res["offenders"]
    res0 = sa.i11_oversize_groups(df, {"train": 288})
    assert not res0["passed"] and res0["parts_without_a_quota"] == ["dev", "test"]


def test_i12_fires_on_size_misses_empty_parts_and_a_zero_capacity():
    df = _frame()
    targets = {p: int((df["part"] == p).sum()) for p in sa.PARTS}
    assert sa.i12_exact_sizes(df, targets, True)["passed"]
    assert not sa.i12_exact_sizes(df, {"train": 1, "dev": 1, "test": 1}, True)["passed"]
    assert not sa.i12_exact_sizes(df, {}, True)["passed"]
    merged = df.copy()
    merged.loc[merged["part"] == "test", "part"] = "train"
    assert not sa.i12_exact_sizes(merged, targets, False)["passed"]
    soft = sa.i12_exact_sizes(df, targets, False, registered_targets=_registered())
    assert soft["passed"] and soft["registered_delta"] is not None
    assert soft["underpowered"] is True   # 96 test sentences < MIN_TEST_SENTENCES


def test_i13_fires_on_duplicates_blanks_and_an_empty_frame():
    df = _frame()
    res = sa.i13_determinism(df)
    assert res["passed"] and len(res["assignment_sha256"]) == 64
    assert sa.i13_determinism(_frame())["assignment_sha256"] == res["assignment_sha256"]
    assert not sa.i13_determinism(df.iloc[0:0])["passed"]
    dupe = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert not sa.i13_determinism(dupe)["passed"]
    blank = df.copy()
    blank.loc[blank.index[0], "group_id"] = ""
    assert not sa.i13_determinism(blank)["passed"]


def test_i14_fires_on_malformed_provenance_and_an_undeclared_cache():
    prov, gl = _provenance(), _group_layer(_cache())
    assert sa.i14_provenance(prov, group_layer=gl)["passed"]
    assert not sa.i14_provenance({}, group_layer=gl)["passed"]
    bad = copy.deepcopy(prov)
    bad["pool_sha256"] = "not-a-digest"
    assert not sa.i14_provenance(bad, group_layer=gl)["passed"]
    no_batches = copy.deepcopy(prov)
    no_batches["raw_batches"] = []
    assert not sa.i14_provenance(no_batches, group_layer=gl)["passed"]
    bad_batch = copy.deepcopy(prov)
    bad_batch["raw_batches"][3] = {"batch": "batch-003"}
    assert not sa.i14_provenance(bad_batch, group_layer=gl)["passed"]
    no_stable = copy.deepcopy(prov)
    no_stable.pop("pool_append_stable")
    res = sa.i14_provenance(no_stable, group_layer=gl)
    assert not res["passed"] and res["pool_append_stable_recorded"] is False
    # the F05 gap: no shingle source, no fingerprint anywhere
    res2 = sa.i14_provenance(prov, group_layer={})
    assert not res2["passed"] and len(res2["shingle_declaration_problems"]) == 2


def _temporal_frame() -> tuple[pd.DataFrame, dict]:
    """A real-shaped temporal split: per-stratum cut, group-atomic, late->test.

    Every posting in a group is given the group's latest date, which is what the
    production regime achieves by dropping a straddler's post-cut sentences; it
    keeps the control frame a legal temporal split so the invariant has
    something true to confirm before the violations are planted into it.
    """
    df = _frame()
    gmax = df.groupby("group_id")["date_posted_utc"].transform("max")
    df["date_posted_utc"] = gmax
    df["part"] = "train"
    cuts = {}
    for stratum, grp in df.groupby("stratum", sort=True):
        dates = grp.groupby("group_id")["date_posted_utc"].max().sort_values()
        n_late = max(1, int(round(0.2 * len(dates))))
        late = set(dates.index[-n_late:])
        in_stratum = df["stratum"] == stratum
        df.loc[in_stratum & df["group_id"].isin(late), "part"] = "test"
        cuts[stratum] = pd.Timestamp(
            df.loc[in_stratum & (df["part"] == "test"), "date_posted_utc"].min())
        early = [g for g in dates.index if g not in late]
        for g in early[:max(1, len(early) // 5)]:
            df.loc[in_stratum & (df["group_id"] == g), "part"] = "dev"
    return df, {"cut_date": {k: str(v) for k, v in cuts.items()}}


def test_i15_temporal_fires_on_a_post_cut_row_and_a_missing_cut_map():
    df, meta = _temporal_frame()
    res = sa.i15_regime(df, "temporal", meta)
    assert res["passed"], res
    leaked = df.copy()
    late_date = leaked.loc[leaked["part"] == "test", "date_posted_utc"].max()
    train_idx = leaked.index[leaked["part"] == "train"][0]
    leaked.loc[train_idx, "date_posted_utc"] = late_date
    assert not sa.i15_regime(leaked, "temporal", meta)["passed"]

    res_nocuts = sa.i15_regime(df, "temporal", {})
    assert not res_nocuts["passed"] and res_nocuts["problems"]
    partial = {"cut_date": {"indonesia": meta["cut_date"]["indonesia"]}}
    assert not sa.i15_regime(df, "temporal", partial)["passed"]

    nat = df.copy()
    nat["date_posted_utc"] = nat["date_posted_utc"].astype(object)
    nat.loc[nat.index[0], "date_posted_utc"] = "not-a-date"
    res_nat = sa.i15_regime(nat, "temporal", meta)
    assert not res_nat["passed"] and res_nat["n_unparseable_dates"] == 1


def _cross_country_frame(iso: str = "SG") -> tuple[pd.DataFrame, dict]:
    df = _frame()
    holdout_groups = set(df.loc[df["country_iso"] == iso, "group_id"])
    df["part"] = "train"
    df.loc[df["group_id"].isin(holdout_groups), "part"] = "test"
    rest = sorted(set(df.loc[df["part"] == "train", "group_id"]))
    for g in rest[:len(rest) // 5]:
        df.loc[df["group_id"] == g, "part"] = "dev"
    return df, {"holdout_country": iso}


def test_i15_cross_country_fires_on_a_bridge_and_a_missing_holdout():
    df, meta = _cross_country_frame()
    res = sa.i15_regime(df, "cross_country", meta)
    assert res["passed"], res
    assert res["n_test_sentences_from_holdout_country"] > 0
    assert "SAMPLED postings only" in res["group_rule_scope"]

    leaked = df.copy()
    sg_group = sorted(set(leaked.loc[leaked["country_iso"] == "SG", "group_id"]))[0]
    leaked.loc[leaked["group_id"] == sg_group, "part"] = "train"
    assert not sa.i15_regime(leaked, "cross_country", meta)["passed"]

    assert not sa.i15_regime(df, "cross_country", {})["passed"]
    assert not sa.i15_regime(df, "cross_country", {"holdout_country": "ZZ"})["passed"]
    assert not sa.i15_regime(df, "no_such_regime", {})["passed"]
    assert sa.i15_regime(_frame(), "grouped_random", {})["passed"]


def test_i15_cross_country_uses_a_pool_wide_group_map_when_given_one():
    """Finding F09: the group rule is stated over the POOL, and a group whose
    held-out-country posting was never drawn is invisible to the sampled view."""
    df, meta = _cross_country_frame()
    train_group = sorted(set(df.loc[df["part"] == "train", "group_id"]))[0]
    assert sa.i15_regime(df, "cross_country", meta)["passed"]
    pool_meta = dict(meta)
    pool_meta["pool_group_countries"] = {train_group: ["SG", "MY"]}
    res = sa.i15_regime(df, "cross_country", pool_meta)
    assert not res["passed"] and train_group in res["bridging_groups"]
    assert "pool-wide" in res["group_rule_scope"]


def test_i17_fires_on_a_label_column_a_missing_frame_and_an_empty_blind_test():
    df = _frame()
    frames = _frames(df)
    assert sa.i17_no_labels(frames, required_frames=("train", "dev", "test_blind"))["passed"]
    leaked = copy.copy(frames)
    leaked["test_blind"] = frames["test_blind"].assign(adjudicated_spans="[]")
    assert not sa.i17_no_labels(leaked)["passed"]
    labelled_train = copy.copy(frames)
    labelled_train["train"] = frames["train"].assign(gold_tag="SH")
    assert not sa.i17_no_labels(labelled_train)["passed"]
    assert not sa.i17_no_labels({})["passed"]
    assert not sa.i17_no_labels(frames, required_frames=("train", "dev", "test_blind",
                                                         "train_720"))["passed"]
    empty_blind = copy.copy(frames)
    empty_blind["test_blind"] = frames["test_blind"].iloc[0:0]
    assert not sa.i17_no_labels(empty_blind)["passed"]


# ================================================================== the gate
def test_run_invariants_and_assert_invariants_refuse_the_whole_write():
    """End-to-end over the gate split.py actually calls: a planted straddle plus
    a stale cache must raise, and the message must name the invariants."""
    df = _plant_pair_across_parts(_frame())
    full = _cache()
    res = _run(df, cache=full)
    assert not res["I7_direct_pairwise_jaccard"]["passed"]
    try:
        sa.assert_invariants(res)
    except sa.InvariantFailure as exc:
        assert "I7_direct_pairwise_jaccard" in str(exc)
    else:
        raise AssertionError("assert_invariants did not raise")

    # an incomplete cache must also stop the write, not pass vacuously
    cache = _cache()
    for pid in sorted(set(_frame()["posting_id"]))[:3]:
        del cache[pid]
    res2 = _run(_frame(), cache=cache, meta=_meta(cache))
    assert not res2["I7_direct_pairwise_jaccard"]["passed"]
    try:
        sa.assert_invariants(res2)
    except sa.InvariantFailure:
        pass
    else:
        raise AssertionError("an incomplete shingle cache was allowed through")


def test_run_invariants_derives_the_fingerprint_from_the_manifest():
    """The group layer can satisfy the I7 contract without split.py changing:
    `manifest.group_layer` is `leakage_groups_manifest.json` verbatim."""
    df = _frame()
    pilot, practice = _quarantine()
    manifest = _manifest(df)
    res = sa.run_invariants(
        df, input_ids=df["sentence_id"].tolist(),
        targets={p: int((df["part"] == p).sum()) for p in sa.PARTS},
        exact=True, shingles=_cache(), pilot=pilot, practice=practice,
        allowed_reasons=("date_unparseable",), provenance=_provenance(),
        regime="grouped_random", regime_meta={}, manifest=manifest,
        frames=_frames(df))
    i7 = res["I7_direct_pairwise_jaccard"]
    assert i7["cache_fingerprint_status"] == "verified"
    assert i7["cache_fingerprint_field"] == "group_layer.shingles_fingerprint"
    assert i7["passed"]
    # ... and without it the gate refuses rather than trusting the cache
    stripped = copy.deepcopy(manifest)
    stripped["group_layer"] = {}
    res2 = sa.run_invariants(
        df, input_ids=df["sentence_id"].tolist(),
        targets={p: int((df["part"] == p).sum()) for p in sa.PARTS},
        exact=True, shingles=_cache(), pilot=pilot, practice=practice,
        allowed_reasons=("date_unparseable",), provenance=_provenance(),
        regime="grouped_random", regime_meta={}, manifest=stripped,
        frames=_frames(df))
    assert res2["I7_direct_pairwise_jaccard"]["cache_fingerprint_status"] == "undeclared"
    assert not res2["I7_direct_pairwise_jaccard"]["passed"]
    assert not res2["I14_provenance"]["passed"]


def test_every_invariant_is_audited_and_has_a_control():
    """The audit is part of the deliverable: every invariant run by the gate
    must appear in VACUITY_AUDIT, and every invariant must have a control in
    this file that makes it fire."""
    df = _frame()
    res = _run(df)
    assert set(res) == set(sa.VACUITY_AUDIT), \
        set(res).symmetric_difference(sa.VACUITY_AUDIT)
    source = Path(__file__).read_text(encoding="utf-8")
    for key in res:
        n = key.split("_")[0].lower()          # i1, i2, ... i17
        assert f"sa.{n}_" in source or f"_{n}(" in source, f"no control for {key}"


def _run_standalone() -> int:
    fns = {n: f for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)}
    passed, failed = 0, []
    for name, fn in fns.items():
        try:
            fn()
            passed += 1
            print(f"  PASS {name}", flush=True)
        except Exception as e:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL {name}: {type(e).__name__}: {e}", flush=True)
    print(f"\n{passed}/{len(fns)} passed"
          + (f", {len(failed)} FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
