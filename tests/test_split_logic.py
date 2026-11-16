"""Allocator / regime / fail-closed-write tests for gold_standard.split.

Standalone runner in the style of tests/test_iaa.py (the repo has no pytest):

    .venv-ml\\Scripts\\python.exe tests\\test_split_logic.py

This file covers the four defects that made the SPLITTER's own logic wrong rather
than its reporting, each with the property the fix has to hold, not just the
symptom:

* the PRIMARY regime could not be written at all. In `grouped_random` with
  exact=True the allocator's capacity was the registered 720/240/240 while the
  placeable weight was the sample minus exclusions, so ANY excluded sentence made
  precondition P2 raise — making `pilot_group`, `practice_group` and
  `date_unparseable`, all three of them in the contract's own closed set of
  reasons, dead letters in the regime that produces the reported benchmark.
  Tested here both as a pure function (`_exact_capacity`) and end to end through
  `run_split` on a sample that really does contain an unparseable-date group.
* P2 itself must still FIRE when a caller hands the allocator a capacity that
  ignores exclusions. The fix is in the capacity argument, not in the assertion,
  and there is a test below whose only job is to prove the assertion was not
  quietly softened.
* the cross-country hold-out rule was evaluated over the DRAWN SAMPLE, so a
  leakage group whose held-out-country posting was not drawn stayed in train and
  the regime's own invariant, checking the same restricted view, could not see it.
* re-running split.py over a sealed split deleted the adjudicated test labels
  (rmtree + os.replace over the whole directory). It must refuse, and a forced
  re-seal must carry the human work forward instead of discarding it.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402

from gold_standard import MAX_SENT_PER_GROUP, SPLIT_TARGETS          # noqa: E402
from gold_standard import groups as groups_mod                       # noqa: E402
from gold_standard import posting_keys as keys_mod                   # noqa: E402
from gold_standard import sample as sample_mod                       # noqa: E402
from gold_standard import seal as seal_mod                           # noqa: E402
from gold_standard import split as split_mod                         # noqa: E402
from gold_standard import synth_fixture                              # noqa: E402

FIXTURE_TARGETS = {"train": 120, "dev": 40, "test": 40}
FIXTURE_SAMPLE_N = 200
PILOT_N, PRACTICE_N = 20, 5
HOLDOUT_RULE = {"min_pool_postings": 40, "min_employers": 20,
                "min_candidate_sentences": 200}

_STATE: dict = {}


# ------------------------------------------------------------------- fixture
def _fixture() -> dict:
    """Synthetic corpus -> keys -> groups -> pilot/practice/main draws -> registry.

    The pilot and practice draws are written even though no test below reads them:
    I10's quarantine claim is vacuous without them, so a fixture that skips them
    is not the pipeline the contract's run order describes (§11).
    """
    if _STATE:
        return _STATE
    root = Path(tempfile.mkdtemp(prefix="splitlogic_"))
    meta = synth_fixture.build_fixture(root, n_postings=420, sents_per_posting=5,
                                       holdout_extra_sentences=3)
    gs = root / "DATA" / "gold_standard"
    raw_root = root / "DATA" / "corpus_raw"
    keys_mod.build_posting_keys(raw_root=raw_root, flags_path=Path(meta["flags"]),
                                pool_path=gs / "sentence_pool.parquet", out_dir=gs,
                                verbose=False)
    groups_mod.build_groups(raw_root=raw_root, keys_path=gs / "posting_keys.parquet",
                            pool_path=gs / "sentence_pool.parquet", out_dir=gs,
                            shingle_sidecar=gs / "no_sidecar.parquet", verbose=False)
    sample_mod.GS_DIR = gs
    sample_mod.POOL_PATH = gs / "sentence_pool.parquet"

    pilot = sample_mod.draw_sample(PILOT_N)
    pilot.to_parquet(gs / "pilot_100.parquet", index=False)
    used = set(pilot["posting_id"])
    practice = sample_mod.draw_sample(PRACTICE_N, exclude_ids=used)
    practice.to_parquet(gs / "practice_10.parquet", index=False)
    used |= set(practice["posting_id"])

    gmap = sample_mod._load_group_map()
    quarantined = {gmap[p] for p in used if p in gmap}
    draw = sample_mod.draw_sample(FIXTURE_SAMPLE_N, exclude_ids=used,
                                  group_cap=MAX_SENT_PER_GROUP,
                                  exclude_groups=quarantined)
    assert len(draw) == FIXTURE_SAMPLE_N, f"fixture draw is {len(draw)}"
    draw.to_parquet(gs / "main_200.parquet", index=False)
    registry = synth_fixture.write_registry(gs / "registry.json", FIXTURE_TARGETS)
    _STATE.update({"root": root, "gs": gs, "raw_root": raw_root, "meta": meta,
                   "registry": registry, "sample": draw})
    return _STATE


def _gs_copy() -> Path:
    """An independent copy of the fixture gold-standard dir, so a test that
    tampers with an input cannot leak into another test."""
    st = _fixture()
    dst = Path(tempfile.mkdtemp(prefix="splitlogic_gs_")) / "gold_standard"
    shutil.copytree(st["gs"], dst)
    shutil.rmtree(dst / "splits", ignore_errors=True)
    return dst


def _run(gs: Path, regime: str = "grouped_random", **kw) -> dict:
    st = _fixture()
    params = {"gs_dir": gs, "raw_root": st["raw_root"], "sample_name": "main_200",
              "registry_path": st["registry"], "verbose": False}
    params.update(kw)
    return split_mod.run_split(regime, "CONFIG_SMALL", **params)


def _groups_frame(weights: list[int]) -> list[dict]:
    return [{"group_id": f"g{i:05d}", "weight": int(w), "axis_counts": {}}
            for i, w in enumerate(weights)]


def _frame_for(weights: dict[str, int]) -> pd.DataFrame:
    rows = []
    for gid, w in weights.items():
        for k in range(w):
            rows.append({"sentence_id": f"{gid}#{k}", "group_id": gid})
    return pd.DataFrame(rows)


# ------------------------------------------- P2 / exact-capacity (pure function)
def test_exact_capacity_is_the_register_when_nothing_is_excluded():
    df = _frame_for({f"g{i}": 1 for i in range(1200)})
    realised, info = split_mod._exact_capacity(
        dict(SPLIT_TARGETS), df, {}, {}, set(), verbose=False)
    assert realised == {"train": 720, "dev": 240, "test": 240}, realised
    assert info["shortfall_total"] == 0
    assert info["n_assignable"] == 1200


def test_exact_capacity_charges_an_excluded_group_to_train_only():
    """The whole point of the fix: dev and test keep their registered sizes, so
    the held-out n and every figure in the power block stay pre-registered."""
    df = _frame_for({f"g{i}": 1 for i in range(1198)} | {"gx": 2})
    forced = {"gx": "excluded"}
    reason = {"gx": "date_unparseable"}
    realised, info = split_mod._exact_capacity(
        dict(SPLIT_TARGETS), df, forced, reason, set(), verbose=False)
    assert realised == {"train": 718, "dev": 240, "test": 240}, realised
    assert sum(realised.values()) == info["n_assignable"] == 1198
    assert info["shortfall_total"] == 2
    assert info["shortfall_by_part"] == {"dev": 0, "test": 0, "train": -2}
    assert info["excluded_groups"] == [
        {"group_id": "gx", "weight": 2, "reason": "date_unparseable"}]


def test_exact_capacity_absorbs_an_under_filled_draw():
    """sample.py tolerates an under-filled draw and prints the achieved n; the
    exact regime refused it outright, by exactly the missing weight."""
    df = _frame_for({f"g{i}": 1 for i in range(1197)})
    realised, info = split_mod._exact_capacity(
        dict(SPLIT_TARGETS), df, {}, {}, set(), verbose=False)
    assert realised == {"train": 717, "dev": 240, "test": 240}, realised
    assert info["shortfall_total"] == 3 and info["excluded_weight"] == 0


def test_exact_capacity_counts_dropped_sentences_too():
    df = _frame_for({f"g{i}": 1 for i in range(1200)})
    dropped = {"g0#0", "g1#0", "g2#0"}
    realised, info = split_mod._exact_capacity(
        dict(SPLIT_TARGETS), df, {}, {}, dropped, verbose=False)
    assert info["dropped_weight"] == 3
    assert realised["train"] == 717 and realised["dev"] == 240


def test_exact_capacity_refuses_a_shortfall_train_cannot_absorb():
    df = _frame_for({"gbig": 400, "grest": 50})
    forced, reason = {"gbig": "excluded"}, {"gbig": "pilot_group"}
    try:
        split_mod._exact_capacity({"train": 300, "dev": 100, "test": 100}, df,
                                  forced, reason, set(), verbose=False)
    except split_mod.SplitRefusal as exc:
        assert "more than the registered train target" in str(exc), str(exc)
        assert "pilot_group" in str(exc)
    else:
        raise AssertionError("a shortfall larger than train must be refused, not "
                             "silently absorbed into a negative capacity")


def test_exact_capacity_refuses_a_sample_bigger_than_the_register():
    df = _frame_for({f"g{i}": 1 for i in range(1205)})
    try:
        split_mod._exact_capacity(dict(SPLIT_TARGETS), df, {}, {}, set(),
                                  verbose=False)
    except split_mod.SplitRefusal as exc:
        assert "will not choose which" in str(exc), str(exc)
    else:
        raise AssertionError("exact mode must not silently discard sentences")


def test_p2_still_fires_when_the_capacity_ignores_exclusions():
    """NOT a weakened invariant. Hand `allocate` the old, wrong capacity (the
    register) with an excluded group present and P2 must still raise — and must
    now name the group and its reason instead of blaming the group map."""
    groups = _groups_frame([2] * 600)
    forced = {"g00000": "excluded"}
    try:
        split_mod.allocate(groups, dict(SPLIT_TARGETS), {}, forced=forced, seed=1,
                           regime_id="t", exact=True,
                           exclusion_reasons={"g00000": "date_unparseable"})
    except split_mod.SplitRefusal as exc:
        text = str(exc)
        assert "P2 violated" in text
        assert "g00000" in text and "date_unparseable" in text, text
        assert "assignable weight" in text, text
    else:
        raise AssertionError("P2 must still refuse a capacity that ignores "
                             "exclusions")


def test_p2_message_says_under_filled_when_there_are_no_exclusions():
    try:
        split_mod.allocate(_groups_frame([2] * 599), dict(SPLIT_TARGETS), {},
                           forced={}, seed=1, regime_id="t", exact=True)
    except split_mod.SplitRefusal as exc:
        assert "P2 violated" in str(exc)
        assert "under-filled" in str(exc), str(exc)
    else:
        raise AssertionError("short total weight must be refused")


def test_realised_capacity_still_lands_exactly_and_parity_holds():
    """An odd realised capacity is legal: P4's parity correction handles it. The
    allocator must hit the realised numbers exactly, not approximately."""
    groups = _groups_frame([2] * 598 + [1] * 3)      # 1,199 placeable
    res = split_mod.allocate(groups, {"train": 719, "dev": 240, "test": 240}, {},
                             forced={}, seed=5, regime_id="t", exact=True)
    sizes = {p: 0 for p in ("train", "dev", "test")}
    by_id = {g["group_id"]: g for g in groups}
    for gid, part in res["group_part"].items():
        sizes[part] += by_id[gid]["weight"]
    assert sizes == {"train": 719, "dev": 240, "test": 240}, sizes
    assert len(res["parity_corrections"]) == 1, res["parity_corrections"]


# ---------------------------------------------------- cross-country group scope
def test_pool_group_countries_sees_postings_that_were_never_drawn():
    pool_groups = pd.DataFrame([
        {"posting_id": "p1", "group_id": "gA", "country_iso": "SG"},   # not drawn
        {"posting_id": "p2", "group_id": "gA", "country_iso": "MY"},   # drawn
        {"posting_id": "p3", "group_id": "gB", "country_iso": "MY"},
    ])
    out = split_mod._pool_group_countries(pool_groups, {"gA", "gB"})
    assert out == {"gA": ["MY", "SG"], "gB": ["MY"]}, out
    assert "SG" in out["gA"], ("a group whose SG posting was not drawn must still "
                               "be known to touch SG")
    assert split_mod._pool_group_countries(pool_groups, {"gB"}) == {"gB": ["MY"]}


def test_cross_country_purity_block_measures_what_it_cannot_claim():
    df = pd.DataFrame([
        {"part": "test", "country_iso": "US"},
        {"part": "test", "country_iso": "US"},
        {"part": "test", "country_iso": "DE"},
        {"part": "train", "country_iso": "DE"},
        {"part": "dev", "country_iso": "ID"},
    ])
    block = split_mod.cross_country_purity_block(df, "US")
    assert block["country_purity"] == "NOT claimed"
    assert block["test_country_composition"] == {"DE": 1, "US": 2}
    assert block["n_test_sentences_from_other_countries"] == 1
    assert block["countries_on_both_sides"] == ["DE"]
    assert block["n_train_dev_sentences_from_countries_also_in_test"] == 1
    assert block["share_test_from_holdout_country"] == round(2 / 3, 4)


# ------------------------------------------------------ sealed-split protection
def _fake_sealed_split(labels: bool = True, materialised: bool = True,
                       opened: str | None = None) -> Path:
    out = Path(tempfile.mkdtemp(prefix="sealed_")) / "grouped_random_v1"
    out.mkdir(parents=True)
    (out / "manifest.json").write_text("{}", encoding="utf-8")
    (out / "SEAL.json").write_text(json.dumps({
        "labels_materialised": materialised, "opened_utc": opened,
        "test_labels_sha256": "deadbeef" if materialised else None,
        "assignment_sha256": "abc"}), encoding="utf-8")
    if labels:
        (out / "labels").mkdir()
        (out / "labels" / "test_labels.parquet").write_bytes(b"not-really-parquet")
    return out


def test_sealed_state_reports_labels_and_receipts_as_protected():
    out = _fake_sealed_split()
    state = split_mod._sealed_state(out)
    assert state["protected"] is True
    assert state["label_files"] == ["test_labels.parquet"]
    assert any("labels/" in e for e in state["evidence"])
    assert any("labels_materialised" in e for e in state["evidence"])

    opened = _fake_sealed_split(labels=False, materialised=False,
                                opened="2026-11-16T00:00:00+00:00")
    assert split_mod._sealed_state(opened)["protected"] is True

    plain = _fake_sealed_split(labels=False, materialised=False)
    assert split_mod._sealed_state(plain)["protected"] is False
    assert split_mod._sealed_state(plain / "nope")["protected"] is False


def test_sealed_state_treats_an_unparseable_seal_as_protected():
    out = _fake_sealed_split(labels=False, materialised=False)
    (out / "SEAL.json").write_text("{not json", encoding="utf-8")
    state = split_mod._sealed_state(out)
    assert state["seal_unreadable"] is True and state["protected"] is True


def test_carry_superseded_copies_labels_and_the_old_receipt():
    out = _fake_sealed_split()
    tmp = Path(tempfile.mkdtemp(prefix="carry_"))
    carried = split_mod._carry_superseded(out, tmp, "20261116T000000Z")
    assert carried == ["SEAL_superseded_20261116T000000Z.json",
                       "labels_superseded_20261116T000000Z"], carried
    assert (tmp / "labels_superseded_20261116T000000Z"
            / "test_labels.parquet").exists()
    assert not (tmp / "labels").exists(), ("the carried labels must NOT be "
                                           "reinstated as the new split's labels")


def test_a_carried_forward_archive_is_itself_protected_and_chains():
    """Otherwise the carry only postpones the loss: force-reseal once, then an
    ordinary re-run rmtrees the directory with the archive inside it."""
    out = _fake_sealed_split(labels=False, materialised=False)
    (out / "labels_superseded_20261101T000000Z").mkdir()
    (out / "labels_superseded_20261101T000000Z" / "test_labels.parquet"
     ).write_bytes(b"older-adjudication")
    state = split_mod._sealed_state(out)
    assert state["protected"] is True
    assert state["superseded_archives"] == ["labels_superseded_20261101T000000Z"]

    tmp = Path(tempfile.mkdtemp(prefix="carry2_"))
    carried = split_mod._carry_superseded(out, tmp, "20261116T010101Z")
    assert "labels_superseded_20261101T000000Z" in carried, carried
    assert (tmp / "labels_superseded_20261101T000000Z"
            / "test_labels.parquet").read_bytes() == b"older-adjudication"


def test_force_reseal_without_a_written_reason_is_refused():
    try:
        split_mod.run_split("grouped_random", "CONFIG_A",
                            gs_dir=Path(tempfile.mkdtemp(prefix="empty_")),
                            force_reseal=True, verbose=False)
    except split_mod.SplitRefusal as exc:
        assert "requires a written reason" in str(exc), str(exc)
    else:
        raise AssertionError("a forced re-seal needs a logged reason")


# ------------------------------------------------------------------- input gate
def test_missing_posting_keys_manifest_is_a_refusal_not_a_filenotfounderror():
    """It was read unguarded for the flags provenance, so a missing one surfaced
    as a bare pathlib FileNotFoundError from the middle of the run."""
    gs = _gs_copy()
    (gs / "posting_keys_manifest.json").unlink()
    try:
        _run(gs)
    except split_mod.SplitRefusal as exc:
        assert "posting_keys_manifest.json" in str(exc)
        assert "gold_standard.posting_keys" in str(exc)
    except FileNotFoundError as exc:                      # the defect itself
        raise AssertionError(f"bare FileNotFoundError instead of a refusal: {exc}")
    else:
        raise AssertionError("a missing input manifest must be refused")
    finally:
        shutil.rmtree(gs.parent, ignore_errors=True)


# ----------------------------------------------------------------- end to end
def _make_one_group_date_unparseable(gs: Path) -> tuple[str, int]:
    """Make one group that IS in the drawn sample carry an unparseable date.

    Corrupting the posting key (rather than the sample) is the same path the real
    corpus takes: `posting_keys.date_parsed` is False for the one unparseable row
    in 85,740, and base_forced excludes its whole group.
    """
    sample = pd.read_parquet(gs / "main_200.parquet", columns=["posting_id"])
    glg = pd.read_parquet(gs / "leakage_groups.parquet",
                          columns=["posting_id", "group_id",
                                   "group_employer_verifiable"])
    joined = sample.merge(glg, on="posting_id", how="left")
    ok = joined[joined["group_employer_verifiable"].astype(bool)]
    gid = sorted(ok["group_id"].astype(str))[0]
    postings = sorted(set(glg.loc[glg["group_id"].astype(str) == gid, "posting_id"]))
    weight = int((joined["group_id"].astype(str) == gid).sum())

    keys = pd.read_parquet(gs / "posting_keys.parquet")
    mask = keys["posting_id"].isin(postings)
    keys.loc[mask, "date_parsed"] = False
    keys.loc[mask, "date_posted_utc"] = pd.NaT
    keys.to_parquet(gs / "posting_keys.parquet", index=False)
    return gid, weight


def test_grouped_random_writes_with_an_excluded_group_and_charges_train():
    """THE defect: with exact=True the primary regime raised P2 on any exclusion,
    so the benchmark the dissertation reports could not be built at all."""
    gs = _gs_copy()
    try:
        gid, weight = _make_one_group_date_unparseable(gs)
        assert weight >= 1
        res = _run(gs)
        man = res["manifest"]
        achieved = {p: int((res["assignment"]["part"] == p).sum())
                    for p in ("train", "dev", "test")}
        assert achieved["dev"] == FIXTURE_TARGETS["dev"], achieved
        assert achieved["test"] == FIXTURE_TARGETS["test"], achieved
        assert achieved["train"] == FIXTURE_TARGETS["train"] - weight, achieved
        reasons = man["exclusions"]["by_reason"]
        assert reasons.get("date_unparseable") == weight, reasons
        assert man["invariants"]["I12_exact_sizes"]["passed"] is True
        assert man["invariants"]["I1_accounting"]["passed"] is True
        cap = man.get("exact_capacity") or man.get("regime", {}).get("exact_capacity")
        if cap is None:                 # manifest surfacing is another agent's
            cap = res.get("regime_meta", {}).get("exact_capacity")
        assert (gs / "splits" / "grouped_random_v1" / "assignment.parquet").exists()
    finally:
        shutil.rmtree(gs.parent, ignore_errors=True)


def test_pilot_group_collision_is_excluded_and_the_capacity_absorbs_it():
    """`pilot_group` was equally unreachable in the exact regime: P2 raised before
    the forcing rule could ever matter.

    Driven through load_inputs -> quarantine_sets -> base_forced ->
    _exact_capacity -> allocate rather than through run_split, because
    split_assert.i10_quarantine currently scans the WHOLE assignment frame
    including the `excluded` rows, so a pilot-group collision fails I10 by
    construction and the contract's own `pilot_group` reason stays unreachable
    for a reason outside this agent's files (reported as a cross-file request).
    The allocator layer this test does cover is the part that was broken.
    """
    gs = _gs_copy()
    try:
        sample = pd.read_parquet(gs / "main_200.parquet")
        pilot = pd.read_parquet(gs / "pilot_100.parquet")
        victim = sorted(set(sample["posting_id"]))[0]
        weight = int((sample["posting_id"] == victim).sum())
        pd.concat([pilot, sample[sample["posting_id"] == victim].head(1)],
                  ignore_index=True).to_parquet(gs / "pilot_100.parquet",
                                                index=False)

        df = split_mod.load_inputs(gs / "main_200.parquet",
                                   gs / "leakage_groups.parquet",
                                   gs / "posting_keys.parquet")
        gframe = pd.read_parquet(gs / "leakage_groups.parquet",
                                 columns=["posting_id", "group_id"])
        quarantine = split_mod.quarantine_sets(gs, gframe)
        forced, reason = split_mod.base_forced(df, quarantine)
        gid = str(df.loc[df["posting_id"] == victim, "group_id"].iloc[0])
        assert forced.get(gid) == "excluded" and reason[gid] == "pilot_group"

        realised, info = split_mod._exact_capacity(
            dict(FIXTURE_TARGETS), df, forced, reason, set(), verbose=False)
        assert realised["dev"] == 40 and realised["test"] == 40, realised
        assert realised["train"] == 120 - info["excluded_weight"], realised
        assert info["excluded_weight"] >= weight

        work = df.copy()
        result = split_mod.allocate(
            split_mod._group_rows(work), realised,
            split_mod.axis_targets(work, realised), forced, seed=1,
            regime_id="grouped_random_v1", exact=True, exclusion_reasons=reason)
        part = {p: 0 for p in ("train", "dev", "test")}
        for g, p in result["group_part"].items():
            if p in part:
                part[p] += int((work["group_id"].astype(str) == g).sum())
        assert part == realised, (part, realised)
    finally:
        shutil.rmtree(gs.parent, ignore_errors=True)


def test_cross_country_moves_groups_whose_holdout_posting_was_never_drawn():
    gs = _gs_copy()
    try:
        chosen = split_mod.derive_holdouts(gs, HOLDOUT_RULE)
        assert chosen, "the fixture must resolve at least one hold-out country"
        iso = sorted(chosen.items())[0][1]
        res = _run(gs, "cross_country", holdout_country=iso,
                   holdout_country_derived=True, holdout_rule=HOLDOUT_RULE)
        assign = res["assignment"]

        pool = pd.read_parquet(gs / "sentence_pool.parquet",
                               columns=["posting_id", "country_iso"]
                               ).drop_duplicates("posting_id")
        glg = pd.read_parquet(gs / "leakage_groups.parquet",
                              columns=["posting_id", "group_id"])
        pg = glg.merge(pool, on="posting_id", how="left")
        pool_touching = set(pg.loc[pg["country_iso"].astype(str) == iso,
                                   "group_id"].astype(str))
        trdev = set(assign.loc[assign["part"].isin(("train", "dev")),
                               "group_id"].astype(str))
        assert not (pool_touching & trdev), sorted(pool_touching & trdev)[:5]

        drawn_only = set(assign.loc[
            assign["posting_id"].isin(
                pool.loc[pool["country_iso"].astype(str) == iso, "posting_id"]),
            "group_id"].astype(str))
        in_sample = pool_touching & set(assign["group_id"].astype(str))
        assert drawn_only <= in_sample
        man = res["manifest"]["cross_country"]
        assert man.get("holdout_group_scope") == "pool", man.get(
            "holdout_group_scope")
        assert man["n_groups_touching_holdout_in_pool"] >= len(in_sample)
    finally:
        shutil.rmtree(gs.parent, ignore_errors=True)


def test_rerunning_over_a_sealed_split_refuses_and_keeps_the_labels():
    """The data-destruction bug: rmtree(out_dir) + os.replace removed labels/ and
    reset SEAL.json, silently, on an ordinary re-run."""
    gs = _gs_copy()
    try:
        _run(gs)
        splits_dir = gs / "splits"
        blind = pd.read_parquet(splits_dir / "grouped_random_v1"
                                / "test_blind.parquet")
        gold = pd.DataFrame([{"sentence_id": s, "span_start": 0, "span_end": 4,
                              "tag": "SH", "verdict": "RESOLVED"}
                             for s in blind["sentence_id"]])
        seal_mod.materialise_labels("grouped_random_v1", gold,
                                    splits_dir=splits_dir)
        labels = splits_dir / "grouped_random_v1" / "labels" / "test_labels.parquet"
        assert labels.exists()
        before = labels.read_bytes()
        seal_before = json.loads((splits_dir / "grouped_random_v1" / "SEAL.json")
                                 .read_text(encoding="utf-8"))
        assert seal_before["labels_materialised"] is True

        try:
            _run(gs)
        except split_mod.SplitRefusal as exc:
            assert "SEALED" in str(exc) and "force-reseal" in str(exc), str(exc)
        else:
            raise AssertionError("re-running over a sealed split must be refused")

        assert labels.exists(), "the adjudicated labels were destroyed"
        assert labels.read_bytes() == before
        seal_after = json.loads((splits_dir / "grouped_random_v1" / "SEAL.json")
                                .read_text(encoding="utf-8"))
        assert seal_after["labels_materialised"] is True
        assert seal_after["test_labels_sha256"] == seal_before["test_labels_sha256"]
    finally:
        shutil.rmtree(gs.parent, ignore_errors=True)


def test_force_reseal_carries_the_old_labels_forward_instead_of_deleting_them():
    gs = _gs_copy()
    try:
        _run(gs)
        splits_dir = gs / "splits"
        blind = pd.read_parquet(splits_dir / "grouped_random_v1"
                                / "test_blind.parquet")
        gold = pd.DataFrame([{"sentence_id": s, "span_start": 0, "span_end": 4,
                              "tag": "SH", "verdict": "RESOLVED"}
                             for s in blind["sentence_id"]])
        seal_mod.materialise_labels("grouped_random_v1", gold,
                                    splits_dir=splits_dir)
        before = (splits_dir / "grouped_random_v1" / "labels"
                  / "test_labels.parquet").read_bytes()

        _run(gs, force_reseal=True,
             force_reseal_reason="unit test: deliberate recompute")
        out = splits_dir / "grouped_random_v1"
        carried = sorted(p.name for p in out.glob("labels_superseded_*"))
        assert len(carried) == 1, sorted(p.name for p in out.iterdir())
        kept = (out / carried[0] / "test_labels.parquet")
        assert kept.exists() and kept.read_bytes() == before
        assert not (out / "labels").exists(), (
            "old labels must not be reinstated as the new split's labels")
        seal = json.loads((out / "SEAL.json").read_text(encoding="utf-8"))
        assert seal["labels_materialised"] is False
        sup = seal["superseded_previous_seal"]
        assert sup["reason"] == "unit test: deliberate recompute"
        assert sup["previous_seal"]["labels_materialised"] is True
        assert carried[0] in sup["carried_files"]
    finally:
        shutil.rmtree(gs.parent, ignore_errors=True)


def test_an_unsealed_rerun_is_still_allowed_and_deterministic():
    gs = _gs_copy()
    try:
        first = _run(gs)
        second = _run(gs)
        assert (first["manifest"]["assignment"]["assignment_sha256"]
                == second["manifest"]["assignment"]["assignment_sha256"])
    finally:
        shutil.rmtree(gs.parent, ignore_errors=True)


def _run_standalone() -> int:
    fns = {n: f for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)}
    passed, failed = 0, []
    for name, fn in fns.items():
        try:
            fn()
            passed += 1
            print(f"  PASS {name}", flush=True)
        except Exception as exc:                                     # noqa: BLE001
            failed.append(name)
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
    print(f"\n{passed}/{len(fns)} passed"
          + (f", {len(failed)} FAILED: {failed}" if failed else ""))
    if _STATE.get("root"):
        shutil.rmtree(_STATE["root"], ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
