"""Tests for the TEMPLATE-SIGNATURE layer: gold_standard/textsig.py + groups.py.

Standalone runner (the repo has no pytest), in the style of tests/test_iaa.py:

    .venv-ml\\Scripts\\python.exe tests\\test_split_groups.py

These are positive controls for the two defects the adversarial review called
fatal and serious in this layer, written so they FAIL against the old code:

  * F01 — the leakage signature covered only the first 2,500 cleaned characters,
    so a template bridge living past that point was invisible to both the group
    map and to invariant I7. `test_template_bridge_past_char_2500_*` plants
    exactly that bridge and asserts it is grouped; the companion
    `..._is_invisible_under_the_old_2500_window` asserts the SAME corpus is
    mis-grouped when the window is re-applied, so the test cannot pass for the
    wrong reason and the old suite's ~120-character I7 control is replaced by
    one that can actually fail.
  * F05 / F14 — a signature cache was trusted on posting-id coverage alone.
    Every tamper the adversaries reproduced (deleted rows, garbage signatures
    under the right ids, a cache built under a different window, a hand-edited
    manifest, an un-fingerprinted sidecar) is asserted to be REFUSED here.
  * F17 — MinHashLSH recall below 1.0 dropped candidate pairs already above
    TAU_BUILD. `test_template_pairs_is_exact_against_brute_force` pins the
    replacement against O(n^2) ground truth.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_standard import SHINGLE_WORDS, TAU_BUILD                   # noqa: E402
from gold_standard import groups as G                                # noqa: E402
from gold_standard import split_assert as sa                         # noqa: E402
from gold_standard import textsig as T                               # noqa: E402


# ----------------------------------------------------------------- fixtures
def _words(tag: str, n: int) -> str:
    """`n` distinct 8-character words, so character length is predictable."""
    return " ".join(f"{tag}{i:05d}" for i in range(n))


# A shared template block that starts AFTER character 2,500 of both postings.
_UNIQUE_PREFIX_WORDS = 340           # ~2,720 chars, so the cut lands inside it
_SHARED_BLOCK_WORDS = 340
_SHARED_BLOCK = _words("zz", _SHARED_BLOCK_WORDS)


def _bridge_pair() -> tuple[str, str]:
    a = _words("aa", _UNIQUE_PREFIX_WORDS) + " " + _SHARED_BLOCK
    b = _words("bb", _UNIQUE_PREFIX_WORDS) + " " + _SHARED_BLOCK
    return a, b


def _mini_corpus(root: Path, postings: list[dict]) -> tuple[Path, Path]:
    """Smallest inputs `groups.build_groups` needs: a raw batch, keys, a pool."""
    raw_root = root / "DATA" / "corpus_raw"
    gs = root / "DATA" / "gold_standard"
    (raw_root / "batch-000").mkdir(parents=True, exist_ok=True)
    gs.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "source_record_id": p["pid"],
        "acquisition_timestamp": "2026-11-01T00:00:00Z",
        "description": p.get("description", ""),
    } for p in postings]).to_parquet(raw_root / "batch-000" / "records.parquet",
                                     index=False)
    pd.DataFrame([{
        "posting_id": p["pid"],
        "stratum": p.get("stratum", "asean"),
        "country_iso": p.get("country_iso", "SG"),
        "site": p.get("site", "jobstreet"),
        "employer_hash": p.get("employer_hash", ""),
        "employer_hash_loose": p.get("employer_hash_loose", ""),
        "employer_known": bool(p.get("employer_known", False)),
    } for p in postings]).to_parquet(gs / "posting_keys.parquet", index=False)
    pd.DataFrame({"posting_id": [p["pid"] for p in postings]}).to_parquet(
        gs / "sentence_pool.parquet", index=False)
    return raw_root, gs


def _build(root: Path, postings: list[dict], **kw) -> tuple[pd.DataFrame, dict, Path]:
    raw_root, gs = _mini_corpus(root, postings)
    frame = G.build_groups(raw_root=raw_root, keys_path=gs / "posting_keys.parquet",
                           pool_path=gs / "sentence_pool.parquet", out_dir=gs,
                           verbose=False, **kw)
    man = json.loads((gs / "leakage_groups_manifest.json").read_text(encoding="utf-8"))
    return frame, man, gs


def _filler(n: int, tag: str = "fl") -> list[dict]:
    """`n` mutually unrelated postings, so a stratum is big enough to be legal."""
    return [{"pid": f"u-{tag}{i:04d}",
             "description": _words(f"{tag}{i:04d}", 60),
             "employer_hash": f"h{tag}{i:04d}", "employer_known": True}
            for i in range(n)]


# --------------------------------------------------- F01: the window is gone
def test_leakage_window_is_the_whole_description():
    assert T.LEAKAGE_WINDOW_CHARS is None, (
        "the leakage signature must cover the whole cleaned description; a "
        "finite window is finding F01")
    assert T.LADDER_PARITY_WINDOW_CHARS == 2500, (
        "the ladder-parity value must stay available and unchanged")
    assert "full cleaned text" in T.TEXT_PIPELINE


def test_long_text_signature_strictly_contains_the_truncated_one():
    long_text = _words("aa", 400) + " " + _words("bb", 400)
    # cut on a word boundary: a mid-word cut invents a token that exists in no
    # full-text shingle, which would make the containment claim false for a
    # reason that has nothing to do with the window.
    cut = long_text.index(" ", 2500)
    full = T.shingles_from_clean_text(long_text)
    trunc = T.shingles_from_clean_text(long_text, max_chars=cut)
    assert trunc.size < full.size
    assert set(trunc.tolist()) <= set(full.tolist())


def test_the_planted_bridge_really_is_a_past_2500_bridge():
    """The fixture must be the defect's shape, not an easier one."""
    a, b = _bridge_pair()
    assert len(a) > 2500 and len(b) > 2500
    full = T.jaccard(T.shingles_from_clean_text(a), T.shingles_from_clean_text(b))
    trunc = T.jaccard(T.shingles_from_clean_text(a, max_chars=2500),
                      T.shingles_from_clean_text(b, max_chars=2500))
    assert full >= TAU_BUILD, f"bridge must be a template pair by the rule, got {full}"
    assert trunc < TAU_BUILD, f"bridge must be invisible at 2500 chars, got {trunc}"


def test_template_bridge_past_char_2500_is_grouped():
    root = Path(tempfile.mkdtemp(prefix="grp_bridge_"))
    try:
        a, b = _bridge_pair()
        postings = _filler(60) + [
            {"pid": "u-bridgeA", "description": a,
             "employer_hash": "empA", "employer_hash_loose": "looseA",
             "employer_known": True},
            {"pid": "u-bridgeB", "description": b,
             "employer_hash": "empB", "employer_hash_loose": "looseB",
             "employer_known": True}]
        frame, man, _ = _build(root, postings)
        gof = dict(zip(frame["posting_id"], frame["group_id"]))
        assert gof["u-bridgeA"] == gof["u-bridgeB"], (
            "a template pair whose shared block starts after char 2,500 must be "
            "in ONE leakage group (finding F01)")
        assert man["n_template_edges_at_tau_build"] >= 1
        assert man["n_template_edges_straddling_groups"] == 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_template_bridge_is_invisible_under_the_old_2500_window():
    """Negative control: the SAME corpus, mis-grouped by the old window.

    Without this, the test above could pass for a reason unrelated to the
    window. It also proves the defect was real rather than theoretical.
    """
    root = Path(tempfile.mkdtemp(prefix="grp_bridge_old_"))
    try:
        a, b = _bridge_pair()
        postings = _filler(60) + [
            {"pid": "u-bridgeA", "description": a, "employer_hash": "empA",
             "employer_hash_loose": "looseA", "employer_known": True},
            {"pid": "u-bridgeB", "description": b, "employer_hash": "empB",
             "employer_hash_loose": "looseB", "employer_known": True}]
        frame, man, _ = _build(root, postings, window_chars=2500)
        gof = dict(zip(frame["posting_id"], frame["group_id"]))
        assert gof["u-bridgeA"] != gof["u-bridgeB"], (
            "the 2,500-char window is supposed to MISS this pair; if it does "
            "not, the fixture no longer reproduces finding F01")
        assert man["shingle_window_chars"] == 2500
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _group_layer_declaration(shingles: dict, window_chars=None) -> dict:
    """The fields this module writes into leakage_groups_manifest.json.

    Built by hand here so the test exercises the interop contract between the
    group layer's manifest and `split_assert.resolve_shingle_meta` / I7.
    """
    return {"shingles_fingerprint": T.cache_content_fingerprint(shingles),
            "shingles_source": "raw_batches",
            "signature_cache": T.cache_fingerprint(shingles,
                                                   window_chars=window_chars),
            "shingle_words": SHINGLE_WORDS,
            "text_pipeline": T.text_pipeline(window_chars)}


def test_i7_fires_on_a_past_2500_bridge():
    """The I7 positive control the old suite could not express.

    tests/test_split.py's I7 control uses two ~120-character strings, so the
    truncation blind spot could never be exercised by it. Here the same bridge
    is measured twice, through the real manifest -> resolve_shingle_meta -> I7
    path: at full text I7 SEES the straddle; over the 2,500-char signatures the
    straddle is invisible to the measurement (max < tau, no violation listed),
    which is precisely finding F01.
    """
    a, b = _bridge_pair()
    df = pd.DataFrame([{"posting_id": "u-A", "part": "train"},
                       {"posting_id": "u-B", "part": "test"}])

    full = {"u-A": T.shingles_from_clean_text(a),
            "u-B": T.shingles_from_clean_text(b)}
    meta = sa.resolve_shingle_meta(None, _group_layer_declaration(full))
    assert meta["fingerprint"] == sa.shingle_cache_fingerprint(full)
    res = sa.i7_direct_pairwise_jaccard(df, full, shingle_meta=meta)
    assert res["passed"] is False, "I7 must fire on a full-text template straddle"
    assert res["max_cross_part_jaccard"] >= TAU_BUILD
    assert res["violations"], "the straddling pair must be named"

    trunc = {"u-A": T.shingles_from_clean_text(a, max_chars=2500),
             "u-B": T.shingles_from_clean_text(b, max_chars=2500)}
    meta25 = sa.resolve_shingle_meta(
        None, _group_layer_declaration(trunc, window_chars=2500))
    res25 = sa.i7_direct_pairwise_jaccard(df, trunc, shingle_meta=meta25)
    assert res25["max_cross_part_jaccard"] < TAU_BUILD, (
        "the 2,500-char signatures must NOT see this straddle; if they do, the "
        "fixture no longer reproduces finding F01")
    assert res25["violations"] == []


# ----------------------------------------- F17: exactness of the edge set
def _brute_force_pairs(shingles: dict, tau: float) -> set[tuple[str, str]]:
    ids = sorted(shingles)
    out = set()
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if T.jaccard(shingles[a], shingles[b]) >= tau:
                out.add((a, b))
    return out


def test_template_pairs_is_exact_against_brute_force():
    rng = np.random.default_rng(7)
    vocab = [f"w{i:04d}" for i in range(400)]
    shingles = {}
    for i in range(120):
        k = int(rng.integers(30, 120))
        words = [vocab[int(v)] for v in rng.integers(0, len(vocab), k)]
        shingles[f"p{i:03d}"] = T.shingles_from_clean_text(" ".join(words))
    # a few deliberate near-clones so the ground truth is not empty
    for i in range(0, 9, 3):
        src = sorted(shingles)[i]
        shingles[f"clone{i}"] = shingles[src]
    shingles["empty"] = np.zeros(0, dtype=np.uint64)
    for tau in (0.10, 0.30, 0.60, 0.95):
        got, meta = G.template_pairs(shingles, tau, verbose=False)
        got_set = {(r.a, r.b) for r in got.itertuples(index=False)}
        assert got_set == _brute_force_pairs(shingles, tau), (
            f"exact enumeration disagrees with brute force at tau={tau}")
        assert meta["complete_at_min_jaccard"] is True
        assert meta["n_postings_without_signature"] == 1


def test_manifest_claims_exact_recall_and_names_the_superseded_method():
    root = Path(tempfile.mkdtemp(prefix="grp_recall_"))
    try:
        _, man, _ = _build(root, _filler(40))
        cg = man["candidate_generation"]
        assert cg["method"] == "exact-sparse-overlap"
        assert cg["template_edge_recall_at_tau_build"] == 1.0
        assert cg["complete_at_min_jaccard"] is True
        assert "completeness_basis" in cg
        sup = cg["superseded_method"]
        assert sup["measured_recall_at_tau_0_30"]["at_full_text"] == 0.8393
        assert sup["measured_recall_at_tau_0_30"]["at_2500_char_window"] == 0.9022
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ------------------------------- F05 / F14: the cache must be fingerprinted
def _tiny_cache() -> dict:
    return {"p1": T.shingles_from_clean_text("one two three four five six seven"),
            "p2": T.shingles_from_clean_text("one two three four five six eight"),
            "p3": np.zeros(0, dtype=np.uint64)}


def test_content_fingerprint_matches_split_assert_exactly():
    """The group layer must RECORD the value I7 RECOMPUTES."""
    cache = _tiny_cache()
    assert T.cache_content_fingerprint(cache) == sa.shingle_cache_fingerprint(cache)


def test_a_matching_fingerprint_verifies():
    cache = _tiny_cache()
    assert T.verify_cache_fingerprint(T.cache_fingerprint(cache), cache) == []


def test_missing_fingerprint_is_a_refusal_not_a_pass():
    cache = _tiny_cache()
    for declared in (None, {}, {"signature_params": T.signature_params()}):
        problems = T.verify_cache_fingerprint(declared, cache)
        assert problems, f"an un-fingerprinted cache must be refused: {declared!r}"


def test_fingerprint_refuses_a_different_window():
    cache = _tiny_cache()
    at_2500 = T.cache_fingerprint(cache, window_chars=2500)
    problems = T.verify_cache_fingerprint(at_2500, cache)      # code wants full text
    assert problems, "a cache built under another window must be refused"
    assert any("window" in p for p in problems), problems


def test_fingerprint_refuses_garbage_signatures_under_the_right_ids():
    """The reproduced attack: right posting ids, unrelated signatures."""
    cache = _tiny_cache()
    declared = T.cache_fingerprint(cache)
    forged = {k: T.shingles_from_clean_text(f"garbage text for {k} and nothing else")
              for k in cache}
    problems = T.verify_cache_fingerprint(declared, forged)
    assert problems and any("signatures themselves differ" in p for p in problems)


def test_fingerprint_refuses_deleted_rows():
    """The reproduced attack that turned I7 into n_pairs_compared = 0."""
    cache = _tiny_cache()
    declared = T.cache_fingerprint(cache)
    problems = T.verify_cache_fingerprint(
        declared, {k: v for k, v in cache.items() if k != "p2"})
    assert problems and any("different posting set" in p for p in problems)


def test_fingerprint_refuses_a_hand_edited_manifest():
    cache = _tiny_cache()
    declared = dict(T.cache_fingerprint(cache))
    declared["n_postings"] = 999
    problems = T.verify_cache_fingerprint(declared, cache)
    assert any("digest its own fields" in p for p in problems), problems


def test_fingerprint_reports_uncovered_required_postings():
    cache = _tiny_cache()
    declared = T.cache_fingerprint(cache)
    problems = T.verify_cache_fingerprint(declared, cache,
                                          require_ids=["p1", "p9"])
    assert any("no signature in the cache" in p for p in problems), problems


# --------------------------------------- the cache on disk, through groups.py
def test_build_shingles_writes_a_fingerprint_manifest_and_load_verifies():
    root = Path(tempfile.mkdtemp(prefix="grp_cache_"))
    try:
        _, man, gs = _build(root, _filler(30))
        cache_man = gs / "shingles_manifest.json"
        assert cache_man.exists()
        body = json.loads(cache_man.read_text(encoding="utf-8"))
        assert body["signature_cache"]["signature_params"]["window_chars"] is None
        loaded = G.load_shingles(gs / "shingles.parquet")
        assert len(loaded) == 30
        # the manifest the SPLIT layer reads carries the same fingerprint
        assert man["shingles_fingerprint"] == sa.shingle_cache_fingerprint(loaded)
        assert sa.resolve_shingle_meta(None, man)["fingerprint"] == \
            man["shingles_fingerprint"]
        assert man["shingles_source"] == "raw_batches"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_load_shingles_refuses_a_cache_with_no_manifest():
    root = Path(tempfile.mkdtemp(prefix="grp_nomani_"))
    try:
        _, _, gs = _build(root, _filler(20))
        (gs / "shingles_manifest.json").unlink()
        try:
            G.load_shingles(gs / "shingles.parquet")
        except G.ShingleCacheError as exc:
            assert "fingerprint manifest" in str(exc)
        else:
            raise AssertionError("an un-fingerprinted cache must be refused")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_load_shingles_refuses_deleted_rows_and_tampered_signatures():
    root = Path(tempfile.mkdtemp(prefix="grp_tamper_"))
    try:
        _, _, gs = _build(root, _filler(20))
        path = gs / "shingles.parquet"
        good = pd.read_parquet(path)

        good.iloc[3:].to_parquet(path, index=False)           # deleted rows
        try:
            G.load_shingles(path)
        except G.ShingleCacheError as exc:
            assert "different posting set" in str(exc)
        else:
            raise AssertionError("row deletion must be refused")

        junk = good.copy()
        junk["shingles"] = [[1, 2, 3]] * len(junk)            # garbage content
        junk.to_parquet(path, index=False)
        try:
            G.load_shingles(path)
        except G.ShingleCacheError as exc:
            assert "signatures themselves differ" in str(exc)
        else:
            raise AssertionError("garbage signatures must be refused")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_load_shingles_refuses_a_cache_built_under_the_old_window():
    root = Path(tempfile.mkdtemp(prefix="grp_oldwin_"))
    try:
        postings = _filler(20) + [
            {"pid": "u-long", "description": _words("aa", 500),
             "employer_hash": "hlong", "employer_known": True}]
        _, man, gs = _build(root, postings, window_chars=2500)
        assert man["shingle_window_chars"] == 2500
        try:
            G.load_shingles(gs / "shingles.parquet")          # default = full text
        except G.ShingleCacheError as exc:
            assert "window" in str(exc)
        else:
            raise AssertionError(
                "a cache built at 2,500 chars must not be reusable at full text")
        # ... and it loads fine when the consumer declares the same window
        assert len(G.load_shingles(gs / "shingles.parquet", window_chars=2500)) == 21
    finally:
        shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------- the build_pool sidecar (F05/F14)
def _write_sidecar(gs: Path, cache: dict, with_manifest: bool,
                   window_chars=None) -> None:
    pd.DataFrame({
        "posting_id": sorted(cache),
        "n_shingles": [int(np.asarray(cache[p]).size) for p in sorted(cache)],
        "shingle_sha1_16": [T.signature_fingerprint(np.asarray(cache[p], dtype=np.uint64))
                            for p in sorted(cache)],
        "shingles": [list(map(int, np.asarray(cache[p]))) for p in sorted(cache)],
    }).to_parquet(gs / "posting_shingles.parquet", index=False)
    if with_manifest:
        (gs / "posting_shingles_manifest.json").write_text(json.dumps({
            "signature_cache": T.cache_fingerprint(cache, window_chars=window_chars),
        }, indent=2), encoding="utf-8")


def test_unfingerprinted_sidecar_is_refused_and_the_raw_layer_wins():
    """The F05/F14 attack: right posting ids, garbage signatures, no fingerprint."""
    root = Path(tempfile.mkdtemp(prefix="grp_side1_"))
    try:
        a, b = _bridge_pair()
        postings = _filler(60) + [
            {"pid": "u-bridgeA", "description": a, "employer_hash": "empA",
             "employer_known": True},
            {"pid": "u-bridgeB", "description": b, "employer_hash": "empB",
             "employer_known": True}]
        raw_root, gs = _mini_corpus(root, postings)
        forged = {p["pid"]: T.shingles_from_clean_text(f"junk for {p['pid']} only")
                  for p in postings}
        _write_sidecar(gs, forged, with_manifest=False)
        G.build_groups(raw_root=raw_root, keys_path=gs / "posting_keys.parquet",
                       pool_path=gs / "sentence_pool.parquet", out_dir=gs,
                       verbose=False)
        man = json.loads((gs / "leakage_groups_manifest.json").read_text(encoding="utf-8"))
        assert man["shingles_source"] == "raw_batches", (
            "an un-fingerprinted sidecar must never be used")
        assert man["shingle_sidecar_decision"]["accepted"] is False
        assert any("fingerprint" in r for r in man["shingle_sidecar_decision"]["reasons"])
        # and the real edge survives, which is what the attack destroyed
        assert man["n_template_edges_at_tau_build"] >= 1
        lg = pd.read_parquet(gs / "leakage_groups.parquet")
        gof = dict(zip(lg["posting_id"], lg["group_id"]))
        assert gof["u-bridgeA"] == gof["u-bridgeB"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_sidecar_with_a_forged_fingerprint_is_caught_by_the_raw_audit():
    """A self-consistent fingerprint is still not enough: the raw text decides."""
    root = Path(tempfile.mkdtemp(prefix="grp_side2_"))
    try:
        postings = _filler(40)
        raw_root, gs = _mini_corpus(root, postings)
        forged = {p["pid"]: T.shingles_from_clean_text(f"junk for {p['pid']} only")
                  for p in postings}
        _write_sidecar(gs, forged, with_manifest=True)       # fingerprint MATCHES itself
        assert T.verify_cache_fingerprint(
            T.cache_fingerprint(forged), forged) == []       # ... provably
        G.build_groups(raw_root=raw_root, keys_path=gs / "posting_keys.parquet",
                       pool_path=gs / "sentence_pool.parquet", out_dir=gs,
                       verbose=False)
        man = json.loads((gs / "leakage_groups_manifest.json").read_text(encoding="utf-8"))
        decision = man["shingle_sidecar_decision"]
        assert man["shingles_source"] == "raw_batches"
        assert decision["accepted"] is False
        assert decision["audit_mismatches"] > 0
        assert any("re-shingle to a different signature" in r
                   for r in decision["reasons"])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_a_genuine_sidecar_is_accepted_and_recorded():
    """The fast path must still exist, and must say so in the manifest."""
    root = Path(tempfile.mkdtemp(prefix="grp_side3_"))
    try:
        postings = _filler(40)
        raw_root, gs = _mini_corpus(root, postings)
        from gold_standard.build_pool import _html_to_text
        genuine = {p["pid"]: T.shingles_from_clean_text(
            _html_to_text(p["description"])) for p in postings}
        _write_sidecar(gs, genuine, with_manifest=True)
        G.build_groups(raw_root=raw_root, keys_path=gs / "posting_keys.parquet",
                       pool_path=gs / "sentence_pool.parquet", out_dir=gs,
                       verbose=False)
        man = json.loads((gs / "leakage_groups_manifest.json").read_text(encoding="utf-8"))
        assert man["shingles_source"] == "sidecar:posting_shingles.parquet"
        assert man["shingle_sidecar_decision"]["accepted"] is True
        assert man["shingle_sidecar_decision"]["audit_mismatches"] == 0
        # a sidecar written under the OLD window is refused even though genuine
        _write_sidecar(gs, {k: T.shingles_from_clean_text(
            _html_to_text(p["description"]), max_chars=2500)
            for k, p in zip(sorted(genuine), sorted(postings, key=lambda d: d["pid"]))},
            with_manifest=True, window_chars=2500)
        G.build_groups(raw_root=raw_root, keys_path=gs / "posting_keys.parquet",
                       pool_path=gs / "sentence_pool.parquet", out_dir=gs,
                       verbose=False)
        man2 = json.loads((gs / "leakage_groups_manifest.json").read_text(encoding="utf-8"))
        assert man2["shingles_source"] == "raw_batches"
        assert any("window" in r for r in man2["shingle_sidecar_decision"]["reasons"])
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ------------------------------------------------- manifest + guards + determinism
def test_manifest_records_the_window_and_fixes_the_claim_wording():
    root = Path(tempfile.mkdtemp(prefix="grp_man_"))
    try:
        _, man, _ = _build(root, _filler(30))
        assert man["shingle_window_chars"] is None
        assert man["shingle_window"] == "full cleaned text (no character window)"
        assert man["signature_params"]["window_chars"] is None
        assert man["signature_params_fingerprint"] == T.params_fingerprint()
        assert man["signature_cache"]["cache_fingerprint"]
        assert "FULL-cleaned-text" in man["claim_wording"]
        # the residuals must be named, not implied
        for residual in ("chosen floor", "two-hop", "string hashes"):
            assert residual in man["claim_wording"], residual
        assert man["claim_wording_history"], "the superseded claim must be recorded"
        assert man["n_postings_without_signature"] == 0
        # contract §8: a row count per raw batch
        assert all(b.get("rows") is not None
                   for b in man["provenance"]["raw_batches"])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_description_less_posting_is_counted_not_hidden():
    root = Path(tempfile.mkdtemp(prefix="grp_empty_"))
    try:
        postings = _filler(20) + [{"pid": "u-blank", "description": "",
                                   "employer_hash": "hblank", "employer_known": True}]
        _, man, _ = _build(root, postings)
        assert man["n_postings_without_signature"] == 1
        assert man["candidate_generation"]["n_postings_without_signature"] == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_oversize_component_guard_actually_refuses():
    """The 5%-per-stratum guard has never fired in this repo; make it fire."""
    root = Path(tempfile.mkdtemp(prefix="grp_guard_"))
    try:
        shared = _words("ss", 200)
        postings = [{"pid": f"u-g{i:03d}", "description": shared,
                     "employer_hash": f"h{i:03d}", "employer_known": True}
                    for i in range(3)]
        postings += _filler(17)                      # 20 postings, one group of 3
        try:
            _build(root, postings)
        except G.GroupFeasibilityError as exc:
            assert "largest leakage group" in str(exc)
        else:
            raise AssertionError("a 15%-of-stratum component must be refused")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_two_builds_of_the_same_corpus_are_identical():
    roots = [Path(tempfile.mkdtemp(prefix=f"grp_det{i}_")) for i in (1, 2)]
    try:
        a, b = _bridge_pair()
        postings = _filler(60) + [
            {"pid": "u-bridgeA", "description": a, "employer_hash": "empA",
             "employer_known": True},
            {"pid": "u-bridgeB", "description": b, "employer_hash": "empB",
             "employer_known": True}]
        out = [_build(r, postings) for r in roots]
        f0, m0, _ = out[0]
        f1, m1, _ = out[1]
        assert m0["shingles_fingerprint"] == m1["shingles_fingerprint"]
        assert m0["signature_cache"]["cache_fingerprint"] == \
            m1["signature_cache"]["cache_fingerprint"]
        assert f0.drop(columns=[]).equals(f1.drop(columns=[]))
    finally:
        for r in roots:
            shutil.rmtree(r, ignore_errors=True)


def test_shingle_width_and_hash_are_part_of_the_fingerprint():
    cache = _tiny_cache()
    assert T.params_fingerprint(shingle_words=SHINGLE_WORDS) != \
        T.params_fingerprint(shingle_words=SHINGLE_WORDS + 1)
    declared = T.cache_fingerprint(cache, shingle_words=SHINGLE_WORDS + 1)
    assert T.verify_cache_fingerprint(declared, cache), (
        "a cache built with a different shingle width must be refused")


def _run_standalone() -> int:
    """Plain-python runner (the repo has no pytest). Pytest-compatible too."""
    fns = {n: f for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)}
    passed, failed = 0, []
    for name, fn in fns.items():
        try:
            fn()
            passed += 1
            print(f"  PASS {name}")
        except Exception as e:                                       # noqa: BLE001
            failed.append(name)
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed"
          + (f", {len(failed)} FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
