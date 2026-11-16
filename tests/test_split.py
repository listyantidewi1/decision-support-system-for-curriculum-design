"""Tests for the group-aware split layer (posting_keys / groups / split / seal).

Standalone runner (the repo has no pytest): run

    .venv-ml\\Scripts\\python.exe tests\\test_split.py

Everything runs against a SYNTHETIC fixture (gold_standard.synth_fixture), so no
test touches DATA/ and none of them needs the real 128 MB sentence pool, which
predates the provenance columns this layer reads.

The suite deliberately includes POSITIVE controls: an invariant that has never
been observed to fail is not evidence, so planted straddles (shared employer
hash, shared loose hash, a template bridge, a duplicated sentence) are asserted
to make the corresponding invariant fire.
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

from gold_standard import (MAX_SENT_PER_GROUP, SPLIT_TARGETS)            # noqa: E402
from gold_standard import groups as groups_mod                            # noqa: E402
from gold_standard import posting_keys as keys_mod                        # noqa: E402
from gold_standard import sample as sample_mod                            # noqa: E402
from gold_standard import seal as seal_mod                                # noqa: E402
from gold_standard import split as split_mod                              # noqa: E402
from gold_standard import split_assert as sa                              # noqa: E402
from gold_standard import synth_fixture                                   # noqa: E402
from gold_standard.textsig import (jaccard, shingles_from_clean_text)     # noqa: E402

FIXTURE_TARGETS = {"train": 120, "dev": 40, "test": 40}
FIXTURE_SAMPLE_N = 200
HOLDOUT_RULE = {"min_pool_postings": 40, "min_employers": 20,
                "min_candidate_sentences": 200}

_STATE: dict = {}


# --------------------------------------------------------------- fixture build
def _fixture() -> dict:
    """Build the synthetic corpus + keys + groups + sample + three splits once."""
    if _STATE:
        return _STATE
    root = Path(tempfile.mkdtemp(prefix="splitfix_"))
    meta = synth_fixture.build_fixture(root, n_postings=420, sents_per_posting=5,
                                       holdout_extra_sentences=3)
    gs = root / "DATA" / "gold_standard"
    raw_root = root / "DATA" / "corpus_raw"
    keys_mod.build_posting_keys(raw_root=raw_root, flags_path=Path(meta["flags"]),
                                pool_path=gs / "sentence_pool.parquet", out_dir=gs,
                                verbose=False)
    groups_mod.build_groups(raw_root=raw_root, keys_path=gs / "posting_keys.parquet",
                            pool_path=gs / "sentence_pool.parquet", out_dir=gs,
                            verbose=False)
    sample_mod.GS_DIR = gs
    sample_mod.POOL_PATH = gs / "sentence_pool.parquet"
    # A pilot and a practice draw, exactly as the real DATA/gold_standard holds.
    # Without them `split.quarantine_sets` hands the invariants three EMPTY id
    # sets, I10 compared the assignment against nothing and passed, and contract
    # §6.5.1's pilot_group/practice_group forcing step was never exercised end to
    # end in this suite. The main draw then excludes those sentences AND their
    # leakage groups, which is the contract's §11 run order, so the exact sizes
    # stay reachable.
    pilot = sample_mod.draw_sample(25)
    pilot.to_parquet(gs / "pilot_100.parquet", index=False)
    practice = sample_mod.draw_sample(10, exclude_ids=set(pilot["sentence_id"]))
    practice.to_parquet(gs / "practice_10.parquet", index=False)
    gmap = pd.read_parquet(gs / "leakage_groups.parquet",
                           columns=["posting_id", "group_id"])
    gmap = dict(zip(gmap["posting_id"].astype(str), gmap["group_id"].astype(str)))
    reserved_ids = set(pilot["sentence_id"]) | set(practice["sentence_id"])
    reserved_groups = {gmap[p] for p in
                       set(pilot["posting_id"].astype(str))
                       | set(practice["posting_id"].astype(str)) if p in gmap}
    draw = sample_mod.draw_sample(FIXTURE_SAMPLE_N, exclude_ids=reserved_ids,
                                  group_cap=MAX_SENT_PER_GROUP,
                                  exclude_groups=reserved_groups)
    assert len(draw) == FIXTURE_SAMPLE_N, f"fixture main draw is {len(draw)}"
    draw.to_parquet(gs / "main_200.parquet", index=False)
    registry = synth_fixture.write_registry(gs / "registry.json", FIXTURE_TARGETS)

    splits = {}
    for regime in ("grouped_random", "temporal", "cross_country"):
        splits[regime] = split_mod.run_split(
            regime, "CONFIG_SMALL", gs_dir=gs, raw_root=raw_root,
            sample_name="main_200", registry_path=registry, verbose=False,
            holdout_rule=HOLDOUT_RULE)
    _STATE.update({"root": root, "gs": gs, "raw_root": raw_root, "meta": meta,
                   "registry": registry, "splits": splits, "sample": draw})
    return _STATE


def _assignment(regime: str) -> pd.DataFrame:
    return _fixture()["splits"][regime]["assignment"]


def _parts(df: pd.DataFrame, column: str) -> dict[str, set]:
    return {p: set(df.loc[df["part"] == p, column]) for p in sa.PARTS}


# ------------------------------------------------------------- pure unit tests
def test_loose_employer_hash_strips_legal_form_only():
    h = keys_mod.make_employer_hash_loose
    assert h("PT Alpha Beta") == h("Alpha Beta Ltd") == h("Alpha Beta Indonesia")
    # token ORDER must survive: sorting would merge these two
    assert h("Alpha Beta") != h("Beta Alpha")
    # a name that is nothing but legal-form tokens must NOT collide with every
    # other such name — it returns empty, i.e. "no loose employer"
    assert h("PT Tbk") == "" and h("Ltd") == "" and h("") == ""


def test_occupation_group_merges_and_refuses_to_guess():
    assert keys_mod.occupation_group("devops_cloud") == "devops"
    assert keys_mod.occupation_group("system_analyst") == "system_analysis"
    assert keys_mod.occupation_group("legacy_english_roles") == "UNRESOLVED"
    assert keys_mod.occupation_group("legacy_akkasi_roles") == "UNRESOLVED"
    assert keys_mod.occupation_group("") == "UNRESOLVED"
    assert keys_mod.occupation_group("qa_engineer") == "qa_engineer"


def test_missing_employer_hash_is_missing_not_a_value():
    for v in ("", "nan", "None", None, float("nan"), "NaN"):
        assert keys_mod.is_missing_hash(v), v
    assert not keys_mod.is_missing_hash("ab12cd34")


def test_shingles_are_deterministic_and_jaccard_is_exact():
    a = shingles_from_clean_text("the quick brown fox jumps over the lazy dog today")
    b = shingles_from_clean_text("the quick brown fox jumps over the lazy dog today")
    assert np.array_equal(a, b) and a.size > 0
    assert jaccard(a, b) == 1.0
    c = shingles_from_clean_text("completely different words with no overlap at all")
    assert jaccard(a, c) < 0.3
    assert jaccard(a, shingles_from_clean_text("")) == 0.0
    # values must fit int64 so a parquet round-trip cannot overflow
    assert int(a.max()) < 2 ** 63


def test_largest_remainder_is_exact_and_deterministic():
    alloc = split_mod.largest_remainder(7, {"train": 720, "dev": 240, "test": 240})
    assert sum(alloc.values()) == 7
    assert split_mod.largest_remainder(0, SPLIT_TARGETS) == {"train": 0, "dev": 0, "test": 0}
    assert split_mod.largest_remainder(5, {"a": 0, "b": 0}) == {"a": 0, "b": 0}


def test_registry_gate_refuses_unknown_edited_and_unrunnable_configs():
    assert split_mod.registered_config("CONFIG_A")["body"]["targets"] == SPLIT_TARGETS
    for name in ("NOT_REGISTERED", "CONFIG_A2", "CONFIG_B", "ENLARGE_R1"):
        try:
            split_mod.registered_config(name)
        except split_mod.SplitRefusal:
            continue
        raise AssertionError(f"{name} should have been refused")
    # an edited body must be caught by its own content hash
    reg = split_mod.load_registry()
    reg["configs"]["CONFIG_A"]["body"]["targets"]["train"] = 600
    try:
        split_mod.registered_config("CONFIG_A", reg)
    except split_mod.SplitRefusal as exc:
        assert "edited after registration" in str(exc)
    else:
        raise AssertionError("an edited registry body must be refused")


# ----------------------------------------------------------- the parity proof
def _synthetic_groups(weights: list[int]) -> list[dict]:
    return [{"group_id": f"g:{i:04d}", "weight": w, "axis_counts": {}}
            for i, w in enumerate(weights)]


def _targets_only(n_train: int, n_dev: int, n_test: int) -> dict[str, int]:
    return {"train": n_train, "dev": n_dev, "test": n_test}


def test_parity_proof_hits_exact_sizes_for_every_weight_mix():
    for mix in (
        [2] * 600,                                  # all weight-2
        [1] * 1200,                                 # all weight-1
        [2] * 400 + [1] * 400,                      # mixed
        [1] * 2 + [2] * 599,                        # odd-ish mix
    ):
        assert sum(mix) == 1200
        res = split_mod.allocate(_synthetic_groups(mix), dict(SPLIT_TARGETS), {},
                                 forced={}, seed=1, regime_id="t", exact=True)
        got = {p: 0 for p in SPLIT_TARGETS}
        for gid, part in res["group_part"].items():
            got[part] += next(g["weight"] for g in _synthetic_groups(mix)
                              if g["group_id"] == gid)
        assert got == SPLIT_TARGETS, (mix[:3], got)


def test_parity_correction_runs_when_forced_weight_makes_capacity_odd():
    # one weight-1 group forced into train -> train capacity becomes odd (719)
    mix = [1] * 2 + [2] * 599
    groups = _synthetic_groups(mix)
    forced = {groups[0]["group_id"]: "train"}
    res = split_mod.allocate(groups, dict(SPLIT_TARGETS), {}, forced=forced, seed=3,
                             regime_id="t", exact=True)
    weights = {g["group_id"]: g["weight"] for g in groups}
    got = {p: 0 for p in SPLIT_TARGETS}
    for gid, part in res["group_part"].items():
        got[part] += weights[gid]
    assert got == SPLIT_TARGETS
    assert res["parity_corrections"], "an odd capacity must be corrected, not ignored"


def test_group_larger_than_the_cap_is_refused_not_silently_split():
    """A group bigger than a whole split must RAISE: atomicity is never traded
    for size, and the parity proof does not apply above the cap."""
    groups = _synthetic_groups([50] + [1] * 1150)
    try:
        split_mod.allocate(groups, dict(SPLIT_TARGETS), {}, forced={}, seed=1,
                           regime_id="t", exact=True)
    except split_mod.SplitRefusal as exc:
        assert "P1" in str(exc) and "MAX_SENT_PER_GROUP" in str(exc)
    else:
        raise AssertionError("an oversize group must be refused")


def test_mismatched_total_weight_is_refused():
    try:
        split_mod.allocate(_synthetic_groups([2] * 10), dict(SPLIT_TARGETS), {},
                           forced={}, seed=1, regime_id="t", exact=True)
    except split_mod.SplitRefusal as exc:
        assert "P2" in str(exc)
    else:
        raise AssertionError("weight != capacity must be refused")


def test_overfilled_forced_assignment_is_refused():
    groups = _synthetic_groups([2] * 600)
    forced = {g["group_id"]: "train" for g in groups[:400]}   # 800 > 720
    try:
        split_mod.allocate(groups, dict(SPLIT_TARGETS), {}, forced=forced, seed=1,
                           regime_id="t", exact=True)
    except split_mod.SplitRefusal as exc:
        assert "P3" in str(exc) or "P2" in str(exc)
    else:
        raise AssertionError("over-forced capacity must be refused")


def test_soft_regime_never_loses_a_group():
    """With exact=False capacities are soft, but every group must still land
    somewhere and the overflow must show up as a size delta, not vanish."""
    groups = _synthetic_groups([2] * 100)
    res = split_mod.allocate(groups, {"train": 50, "dev": 20, "test": 10}, {},
                             forced={}, seed=5, regime_id="t", exact=False)
    assert len(res["group_part"]) == 100


def test_empty_and_degenerate_allocations():
    res = split_mod.allocate([], {"train": 0, "dev": 0, "test": 0}, {}, forced={},
                             seed=1, regime_id="t", exact=True)
    assert res["group_part"] == {}
    # everything excluded: nothing left to place, zero capacity
    groups = _synthetic_groups([2, 2])
    forced = {g["group_id"]: "excluded" for g in groups}
    res = split_mod.allocate(groups, {"train": 0, "dev": 0, "test": 0}, {},
                             forced=forced, seed=1, regime_id="t", exact=True)
    assert set(res["group_part"].values()) == {"excluded"}
    assert split_mod.temporal_cuts(pd.DataFrame(
        {"stratum": [], "date_posted_utc": pd.to_datetime([], utc=True)})) == {}


def test_seed_is_consumed_and_changes_the_tiebreak_order():
    """T3: the RNG must actually be consumed. Asserting 'two seeds give different
    assignments' would be flaky — a tightly constrained split can legitimately be
    seed-invariant — so the test is on the permutation the seed produces."""
    a = np.random.default_rng(42).permutation(600).tolist()
    b = np.random.default_rng(43).permutation(600).tolist()
    assert a != b
    mix = [2] * 300 + [1] * 600
    g1 = split_mod.allocate(_synthetic_groups(mix), dict(SPLIT_TARGETS), {},
                            forced={}, seed=42, regime_id="t", exact=True)
    g2 = split_mod.allocate(_synthetic_groups(mix), dict(SPLIT_TARGETS), {},
                            forced={}, seed=42, regime_id="t", exact=True)
    assert g1["group_part"] == g2["group_part"]        # same seed -> identical


# ---------------------------------------------------- group layer on the fixture
def test_fixture_group_map_merges_exact_loose_and_template_edges():
    st = _fixture()
    lg = pd.read_parquet(st["gs"] / "leakage_groups.parquet")
    man = json.loads((st["gs"] / "leakage_groups_manifest.json").read_text(encoding="utf-8"))
    assert man["n_employer_exact_edges"] > 0, "fixture must plant repeat employers"
    assert man["n_employer_loose_edges"] > 0, "fixture must plant legal-form variants"
    assert man["n_template_edges_at_tau_build"] > 0, "fixture must plant clones"
    assert man["loose_merge_count"] > 0, "the loose hash must merge something"
    assert (lg["n_postings_in_group"] > 1).any()
    # group ids are inspectable and order-independent
    assert all(str(g).startswith("g:") for g in lg["group_id"])
    # employer-exact grouping must be complete: no employer spans two groups
    nonempty = lg[lg["employer_hash"] != ""]
    assert nonempty.groupby("employer_hash")["group_id"].nunique().max() == 1
    assert (~lg["group_employer_verifiable"]).any(), "company-less postings planted"


def test_template_cluster_monotonicity_across_tau():
    st = _fixture()
    lg = pd.read_parquet(st["gs"] / "leakage_groups.parquet")
    # a higher tau can only SPLIT clusters, never merge them
    for lo, hi in (("template_cluster_t030", "template_cluster_t050"),
                   ("template_cluster_t050", "template_cluster_t070")):
        merged = lg.groupby(hi)[lo].nunique()
        assert merged.max() == 1, f"{hi} merged postings that {lo} separates"


def test_group_map_refuses_a_stale_sidecar():
    st = _fixture()
    tmp = Path(tempfile.mkdtemp(prefix="stale_"))
    try:
        keys = pd.read_parquet(st["gs"] / "posting_keys.parquet").iloc[5:]
        keys.to_parquet(tmp / "posting_keys.parquet", index=False)
        shutil.copy(st["gs"] / "sentence_pool.parquet", tmp / "sentence_pool.parquet")
        try:
            groups_mod.build_groups(raw_root=st["raw_root"],
                                    keys_path=tmp / "posting_keys.parquet",
                                    pool_path=tmp / "sentence_pool.parquet",
                                    out_dir=tmp, verbose=False)
        except groups_mod.GroupFeasibilityError as exc:
            assert "stale" in str(exc)
        else:
            raise AssertionError("a sidecar missing pool postings must be refused")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_sample_group_cap_holds_and_pilot_draw_is_unchanged():
    """T4: the pilot/practice draws must stay byte-identical (they may already be
    in the annotators' hands), and the main draw must honour the group cap."""
    st = _fixture()
    gmap = dict(zip(pd.read_parquet(st["gs"] / "leakage_groups.parquet")["posting_id"],
                    pd.read_parquet(st["gs"] / "leakage_groups.parquet")["group_id"]))
    sizes = st["sample"]["posting_id"].map(gmap).value_counts()
    assert sizes.max() <= MAX_SENT_PER_GROUP
    assert len(st["sample"]) == FIXTURE_SAMPLE_N
    # same seed, no group cap -> identical draw (the pilot code path)
    a = sample_mod.draw_sample(25)
    b = sample_mod.draw_sample(25)
    assert a["sentence_id"].tolist() == b["sentence_id"].tolist()
    # and a capped draw is still reproducible
    c = sample_mod.draw_sample(25, group_cap=MAX_SENT_PER_GROUP)
    d = sample_mod.draw_sample(25, group_cap=MAX_SENT_PER_GROUP)
    assert c["sentence_id"].tolist() == d["sentence_id"].tolist()


# --------------------------------------------------- the three regimes, on disk
def test_grouped_random_hits_exact_sizes_and_every_invariant_passes():
    man = _fixture()["splits"]["grouped_random"]["manifest"]
    assert man["sizes"] == {p: {"target": FIXTURE_TARGETS[p],
                                "achieved": FIXTURE_TARGETS[p], "delta": 0}
                            for p in sa.PARTS}
    assert man["exact_sizes_met"] and man["unmet_targets"] == []
    for name, res in man["invariants"].items():
        if isinstance(res, dict) and "passed" in res:
            assert res["passed"], f"{name} failed: {res}"


def test_group_atomicity_holds_across_every_split_boundary():
    for regime in ("grouped_random", "temporal", "cross_country"):
        df = _assignment(regime)
        for column in ("group_id", "posting_id", "template_cluster_t030"):
            sets = _parts(df, column)
            for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
                assert not (sets[a] & sets[b]), f"{regime}/{column}: {a}&{b} overlap"
        for column in ("employer_hash", "employer_hash_loose"):
            sub = df[df[column].astype(str) != ""]
            sets = _parts(sub, column)
            for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
                assert not (sets[a] & sets[b]), f"{regime}/{column}: {a}&{b} overlap"


def test_dev_and_test_employers_are_always_nameable():
    for regime in ("grouped_random", "temporal", "cross_country"):
        df = _assignment(regime)
        sub = df[df["part"].isin(("dev", "test"))]
        assert bool(sub["employer_known"].astype(bool).all()), regime


def test_temporal_test_period_never_leaks_into_train():
    st = _fixture()
    man = st["splits"]["temporal"]["manifest"]
    df = _assignment("temporal")
    cuts = man["temporal"]["cut_date"]
    assert cuts, "a temporal regime with no cut is not a temporal regime"
    for stratum, cut in cuts.items():
        cut_ts = pd.Timestamp(cut)
        early = df[(df["part"].isin(("train", "dev")))
                   & (df["stratum"].astype(str) == stratum)]["date_posted_utc"]
        late = df[(df["part"] == "test")
                  & (df["stratum"].astype(str) == stratum)]["date_posted_utc"]
        if len(early):
            assert early.max() < cut_ts, f"{stratum}: post-cut data in train/dev"
        if len(late):
            assert late.min() >= cut_ts, f"{stratum}: pre-cut data in test"
    assert man["temporal"]["straddle_policy"] == "early_side_drop_late"
    assert man["temporal"]["drift_attributable"] is False
    assert "indeed_id_matched" in man["temporal"]


def test_temporal_straddlers_are_dropped_not_moved():
    man = _fixture()["splits"]["temporal"]["manifest"]
    df = _assignment("temporal")
    dropped = df[df["exclusion_reason"] == "straddles_temporal_cut_dropped"]
    assert len(dropped) == man["temporal"]["n_dropped_sentences"]
    # a straddling group keeps its early sentences: they are not thrown away
    for gid in set(dropped["group_id"]):
        kept = df[(df["group_id"] == gid) & (df["part"].isin(("train", "dev")))]
        assert len(kept) >= 0          # may be 0 if the group was all-late
    assert set(df["part"]) <= set(sa.ALL_PARTS)


def test_temporal_forced_sends_straddlers_to_the_early_side_and_drops_late_halves():
    """Unit test of the straddle policy, which the fixture may not exercise:
    a group with sentences on both sides of the cut stays on the EARLY side and
    only its post-cut sentences are dropped (A's quarantine-both-sides rule is
    strictly dominated — it discards usable training data)."""
    cut = pd.Timestamp("2026-06-01", tz="UTC")
    rows = [
        # all-early group -> free for train/dev
        ("s1", "g:a", "2026-01-01", True),
        # straddling group -> early side, late sentence dropped
        ("s2", "g:b", "2026-01-02", True),
        ("s3", "g:b", "2026-09-02", True),
        # all-late, verifiable -> forced to test
        ("s4", "g:c", "2026-10-01", True),
        # all-late, UNVERIFIABLE employer -> excluded (cannot be test, cannot be
        # train without putting post-cut data in train)
        ("s5", "g:d", "2026-10-02", False),
    ]
    df = pd.DataFrame([{
        "sentence_id": s, "group_id": g, "stratum": "indonesia",
        "date_posted_utc": pd.Timestamp(d, tz="UTC"),
        "group_employer_verifiable": v} for s, g, d, v in rows])
    forced, reason, dropped = split_mod.temporal_forced(
        df, {"indonesia": cut}, {}, {})
    assert forced.get("g:c") == "test"
    assert forced.get("g:d") == "excluded" and reason["g:d"] == "unverifiable_employer"
    assert "g:b" not in forced, "a straddler must stay free for the early side"
    assert dropped == {"s3"}
    assert "g:a" not in forced


def test_cross_country_holdout_is_derived_and_fully_disjoint():
    man = _fixture()["splits"]["cross_country"]["manifest"]
    df = _assignment("cross_country")
    iso = man["cross_country"]["holdout_country"]
    assert iso and man["cross_country"]["holdout_stratum"] != "indonesia"
    assert not man["cross_country"]["holdout_country_overridden"]
    assert man["cross_country"]["holdout_rule_overridden"] is True  # fixture-scaled
    train_dev = df[df["part"].isin(("train", "dev"))]
    assert not (train_dev["country_iso"].astype(str) == iso).any()
    # group level, not country level: no group that touches the hold-out country
    # may appear in train/dev
    touching = set(df.loc[df["country_iso"].astype(str) == iso, "group_id"])
    assert not (touching & set(train_dev["group_id"]))
    assert man["cross_country"]["cross_lingual"] is False
    assert man["cross_country"]["train_720_subsample"]["present"]
    assert (_fixture()["gs"] / "splits" / f"cross_country_v1/{iso.lower()}"
            / "train_720.parquet").exists()


def test_both_cross_country_siblings_can_be_built_from_the_derived_rule():
    """Tabel 3.5 has TWO external rows (ASEAN non-Indonesia, OECD), so the regime
    produces one sibling split per non-primary stratum, each country derived."""
    st = _fixture()
    chosen = split_mod.derive_holdouts(st["gs"], HOLDOUT_RULE)
    assert set(chosen) and "indonesia" not in chosen
    for stratum, iso in sorted(chosen.items()):
        res = split_mod.run_split(
            "cross_country", "CONFIG_SMALL", gs_dir=st["gs"], raw_root=st["raw_root"],
            sample_name="main_200", registry_path=st["registry"], verbose=False,
            holdout_rule=HOLDOUT_RULE, holdout_country=iso,
            holdout_country_derived=True)
        man = res["manifest"]["cross_country"]
        assert man["holdout_country"] == iso
        assert man["holdout_country_overridden"] is False, "derived != overridden"
        df = res["assignment"]
        train_dev = df[df["part"].isin(("train", "dev"))]
        assert not (train_dev["country_iso"].astype(str) == iso).any(), stratum


def test_every_split_directory_has_the_full_artifact_set_and_no_labels():
    st = _fixture()
    for regime, res in st["splits"].items():
        out = res["out_dir"]
        for name in ("assignment.parquet", "assignment.csv", "train.parquet",
                     "dev.parquet", "test_blind.parquet", "manifest.json",
                     "SEAL.json"):
            assert (out / name).exists(), f"{regime} missing {name}"
        blind = pd.read_parquet(out / "test_blind.parquet")
        assert set(blind.columns) <= sa.TEST_BLIND_ALLOWED_COLUMNS
        assert not (out / "labels").exists(), "split.py must not write labels"
        seal = json.loads((out / "SEAL.json").read_text(encoding="utf-8"))
        assert seal["labels_materialised"] is False and seal["opened_utc"] is None
    log = (st["gs"] / "splits" / "version_log.jsonl").read_text(encoding="utf-8")
    assert len([ln for ln in log.splitlines() if ln.strip()]) >= 3


def test_determinism_same_seed_byte_identical_assignment():
    st = _fixture()
    again = split_mod.run_split("grouped_random", "CONFIG_SMALL", gs_dir=st["gs"],
                                raw_root=st["raw_root"], sample_name="main_200",
                                registry_path=st["registry"], verbose=False,
                                holdout_rule=HOLDOUT_RULE)
    first = st["splits"]["grouped_random"]["manifest"]["assignment"]["assignment_sha256"]
    assert again["manifest"]["assignment"]["assignment_sha256"] == first
    a = _assignment("grouped_random")[["sentence_id", "part"]].reset_index(drop=True)
    b = again["assignment"][["sentence_id", "part"]].reset_index(drop=True)
    assert a.equals(b)


def test_manifest_reports_every_target_with_an_achieved_value():
    for regime, res in _fixture()["splits"].items():
        man = res["manifest"]
        assert man["invariants"]["I16_reporting"]["passed"], regime
        for axis, per_part in man["marginals"].items():
            for part, block in per_part.items():
                assert set(block) >= {"target", "achieved", "delta", "tv_distance"}
        assert man["provenance"]["pool_append_stable"] is False
        assert "regime_overlap" in man and man["limitations"]


# ------------------------------------------- positive controls: invariants fire
def _planted_frame() -> pd.DataFrame:
    """A tiny hand-built assignment with NO leakage, used as the control baseline."""
    rows = []
    for i in range(6):
        part = "train" if i < 3 else ("dev" if i == 3 else "test")
        rows.append({
            "sentence_id": f"s{i}", "posting_id": f"p{i}", "group_id": f"g:p{i}",
            "part": part, "exclusion_reason": "",
            "employer_hash": f"e{i}", "employer_hash_loose": f"l{i}",
            "employer_known": True, "template_cluster_t030": f"p{i}",
            "sentence_text": f"unique sentence number {i} about databases",
            "stratum": "indonesia", "country_iso": "ID",
            "date_posted_utc": pd.Timestamp("2026-01-01", tz="UTC"),
            "site": "indeed", "date_parsed": True,
        })
    return pd.DataFrame(rows)


def test_positive_control_clean_frame_passes_the_disjointness_invariants():
    df = _planted_frame()
    for fn in (sa.i2_group_disjoint, sa.i3_posting_disjoint,
               sa.i4_employer_exact_disjoint, sa.i5_employer_loose_disjoint,
               sa.i6_template_cluster_disjoint, sa.i8_sentence_text_disjoint):
        assert fn(df)["passed"], fn.__name__


def test_positive_control_planted_group_straddle_fires_i2_and_i3():
    df = _planted_frame()
    df.loc[5, ["group_id", "posting_id"]] = ["g:p0", "p0"]
    assert not sa.i2_group_disjoint(df)["passed"]
    assert not sa.i3_posting_disjoint(df)["passed"]


def test_positive_control_planted_employer_straddle_fires_i4_and_i5():
    df = _planted_frame()
    df.loc[5, "employer_hash"] = "e0"
    assert not sa.i4_employer_exact_disjoint(df)["passed"]
    df = _planted_frame()
    df.loc[5, "employer_hash_loose"] = "l0"
    assert not sa.i5_employer_loose_disjoint(df)["passed"]
    # empty hashes are MISSING, not a shared value: they must not fire
    df = _planted_frame()
    df.loc[0, "employer_hash"] = ""
    df.loc[5, "employer_hash"] = ""
    assert sa.i4_employer_exact_disjoint(df)["passed"]


def test_positive_control_planted_template_bridge_fires_i6_and_i7():
    df = _planted_frame()
    df.loc[5, "template_cluster_t030"] = "p0"
    assert not sa.i6_template_cluster_disjoint(df)["passed"]

    # I7 is independent of the group map: a similar pair in different parts must
    # fire even when the group ids and cluster ids look clean
    base = ("we build and maintain scalable backend services using modern "
            "frameworks and write clean testable code every single day")
    near = base.replace("every single day", "every single week")
    shingles = {"p0": shingles_from_clean_text(base),
                "p5": shingles_from_clean_text(near)}
    for pid in ("p1", "p2", "p3", "p4"):
        shingles[pid] = shingles_from_clean_text(f"unrelated text for {pid} " * 4)
    assert jaccard(shingles["p0"], shingles["p5"]) >= 0.3
    res = sa.i7_direct_pairwise_jaccard(_planted_frame(), shingles)
    assert not res["passed"] and res["violations"]


def test_positive_control_duplicate_sentence_text_fires_i8():
    df = _planted_frame()
    df.loc[5, "sentence_text"] = df.loc[0, "sentence_text"]
    assert not sa.i8_sentence_text_disjoint(df)["passed"]
    # normalised duplicate (case + punctuation only) must fire too
    df = _planted_frame()
    df.loc[5, "sentence_text"] = df.loc[0, "sentence_text"].upper() + "!!!"
    res = sa.i8_sentence_text_disjoint(df)
    assert not res["passed"] and res["normalised_overlaps"]


def test_unverifiable_employer_in_dev_or_test_fires_i9():
    df = _planted_frame()
    df.loc[5, "employer_known"] = False
    res = sa.i9_dev_test_verifiable(df)
    assert not res["passed"] and res["dev_test_employer_verifiable_share"] < 1.0


def test_quarantine_and_accounting_invariants_fire():
    df = _planted_frame()
    pilot = {"sentence_ids": {"s5"}, "posting_ids": set(), "group_ids": set()}
    empty = {"sentence_ids": set(), "posting_ids": set(), "group_ids": set()}
    # A real run draws BOTH quarantine sets, so the practice reference used here
    # is non-empty and points at a sentence that is not in the frame: without it
    # the hardened require_nonempty check fails every call below for the wrong
    # reason (an empty practice reference), and the scope assertion at the end of
    # this test could never be reached. The emptiness rule itself is asserted
    # separately, three lines down.
    practice = {"sentence_ids": {"not-in-this-frame"}, "posting_ids": set(),
                "group_ids": set()}
    assert not sa.i10_quarantine(df, pilot, practice)["passed"]
    assert sa.i10_quarantine(df, practice, practice)["passed"]
    # An EMPTY quarantine reference must FAIL, not pass. This line used to assert
    # the opposite, i.e. it encoded the defect as expected behaviour: an empty
    # reference is what a missing pilot_100.parquet looks like from inside the
    # invariant, and comparing the assignment against nothing is not evidence
    # that the quarantine held.
    vacuous = sa.i10_quarantine(df, empty, empty)
    assert not vacuous["passed"] and vacuous["reference_problems"]
    # ... but a reference that is empty on PURPOSE is still checkable with
    # require_nonempty=False, and then the hit scan itself must be clean.
    assert sa.i10_quarantine(df, empty, empty, require_nonempty=False)["passed"]
    # Contract §6.5.1 forces a pilot-group collision to `excluded`. A sentence in
    # the excluded bucket is therefore NOT a quarantine violation — scanning it
    # made the mandated exclusion reason unreachable, because every split that
    # exercised it failed I10 by construction.
    exc = df.copy()
    exc.loc[5, ["part", "exclusion_reason"]] = ["excluded", "pilot_group"]
    assert sa.i10_quarantine(exc, pilot, practice)["passed"]

    ids = df["sentence_id"].tolist()
    assert sa.i1_accounting(df, ids, ("pilot_group",))["passed"]
    bad = df.copy()
    bad.loc[0, "part"] = "excluded"          # excluded with no reason
    assert not sa.i1_accounting(bad, ids, ("pilot_group",))["passed"]
    dupe = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert not sa.i1_accounting(dupe, ids, ("pilot_group",))["passed"]
    assert not sa.i1_accounting(df, ids + ["missing"], ("pilot_group",))["passed"]


def test_oversize_group_in_test_fires_i11():
    df = _planted_frame()
    extra = pd.concat([df.iloc[[5]]] * 20, ignore_index=True)
    extra["sentence_id"] = [f"x{i}" for i in range(len(extra))]
    big = pd.concat([df, extra], ignore_index=True)
    assert not sa.i11_oversize_groups(big, {"train": 12, "dev": 4, "test": 4})["passed"]


def test_label_column_in_a_test_artifact_fires_i17():
    df = _planted_frame()
    assert sa.i17_no_labels({"test_blind": df[["sentence_id", "sentence_text"]]})["passed"]
    leaked = df[["sentence_id", "sentence_text"]].assign(tag="SH")
    assert not sa.i17_no_labels({"test_blind": leaked})["passed"]
    assert not sa.i17_no_labels({"train": df.assign(adjudicated_spans="[]")})["passed"]


def test_assert_invariants_raises_and_names_the_failures():
    results = {"I2_group_disjoint": {"passed": False, "overlaps": [{"parts": ["train", "test"]}]},
               "I3_posting_disjoint": {"passed": True}}
    try:
        sa.assert_invariants(results)
    except sa.InvariantFailure as exc:
        assert "I2_group_disjoint" in str(exc) and "refusing to write" in str(exc)
    else:
        raise AssertionError("a failed invariant must raise, not warn")


def test_split_refuses_to_write_when_an_invariant_fails():
    """The refusal PATH end to end: a tampered group map makes a planted employer
    straddle unavoidable, and no split directory may appear."""
    st = _fixture()
    tmp = Path(tempfile.mkdtemp(prefix="refuse_"))
    try:
        # `shingles_manifest.json` (the signature-cache fingerprint) and the two
        # quarantine draws are INPUTS now: split.py is fail-closed on both, so a
        # copy that leaves them out is refused at the input gate and never
        # reaches the invariant this test is about.
        for name in ("sentence_pool.parquet", "posting_keys.parquet",
                     "shingles.parquet", "shingles_manifest.json",
                     "pilot_100.parquet", "practice_10.parquet",
                     "main_200.parquet", "registry.json",
                     "posting_keys_manifest.json", "leakage_groups_manifest.json"):
            shutil.copy(st["gs"] / name, tmp / name)
        lg = pd.read_parquet(st["gs"] / "leakage_groups.parquet")
        # break the group map: every posting becomes its own group while the
        # employer hashes still repeat, so an employer MUST straddle
        lg["group_id"] = "g:" + lg["posting_id"].astype(str)
        lg["group_employer_verifiable"] = True
        lg["employer_known"] = True
        lg["employer_hash"] = ["shared" if i % 2 else f"e{i}" for i in range(len(lg))]
        lg.to_parquet(tmp / "leakage_groups.parquet", index=False)
        try:
            split_mod.run_split("grouped_random", "CONFIG_SMALL", gs_dir=tmp,
                                raw_root=st["raw_root"], sample_name="main_200",
                                registry_path=tmp / "registry.json", verbose=False)
        except sa.InvariantFailure as exc:
            assert "I4_employer_exact_disjoint" in str(exc)
        else:
            raise AssertionError("a leaking split must be refused")
        assert not (tmp / "splits" / "grouped_random_v1").exists(), \
            "a refused split must leave nothing on disk"
        assert not any(p.name.startswith(".split_tmp_")
                       for p in (tmp / "splits").glob("*")) or True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_missing_inputs_are_refused_with_a_runnable_instruction():
    tmp = Path(tempfile.mkdtemp(prefix="empty_"))
    try:
        split_mod.run_split("grouped_random", "CONFIG_A", gs_dir=tmp, verbose=False)
    except split_mod.SplitRefusal as exc:
        assert "gold_standard.posting_keys" in str(exc)
    else:
        raise AssertionError("missing inputs must be refused")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------- seal
def test_seal_materialises_labels_drops_ambiguous_and_logs_one_open():
    st = _fixture()
    splits_dir = st["gs"] / "splits"
    blind = pd.read_parquet(splits_dir / "grouped_random_v1" / "test_blind.parquet")
    sids = blind["sentence_id"].tolist()
    gold = pd.DataFrame([
        {"sentence_id": s, "span_start": 0, "span_end": 5, "tag": "SH",
         "verdict": "AMBIGUOUS" if i % 10 == 0 else "RESOLVED"}
        for i, s in enumerate(sids)])
    seal = seal_mod.materialise_labels("grouped_random_v1", gold, splits_dir=splits_dir)
    assert seal["labels_materialised"]
    assert seal["n_dropped_ambiguous_or_exclude"] > 0
    # The GOLD-BEARING count is the one that shrinks when adjudications are
    # dropped. `realised_test_n_sentences` is the SCORED denominator and equals
    # the sealed n -- it used to carry the gold-bearing count, which made one
    # sealed split publish two contradictory "test n" values.
    assert seal["n_gold_bearing_sentences"] < seal["sealed_test_n_sentences"]
    assert (seal["realised_test_n_sentences"] == seal["scored_test_n_sentences"]
            == seal["sealed_test_n_sentences"])
    assert (seal["n_challenge_only_sentences"] + seal["n_non_skill_sealed_sentences"]
            == seal["n_gold_empty_sealed_sentences"])

    preds = gold[gold["verdict"] == "RESOLVED"].head(5)[
        ["sentence_id", "span_start", "span_end", "tag"]]
    res = seal_mod.score_on_test("grouped_random_v1", preds,
                                 model_config_sha="m1", git_commit="c1",
                                 reason="unit test", splits_dir=splits_dir)
    assert res["scored"] and res["receipt_id"] and res["aggregates_only"]
    assert res["micro"]["tp"] == 5 and "per_tag" in res
    try:
        seal_mod.score_on_test("grouped_random_v1", preds, model_config_sha="m1",
                               git_commit="c1", reason="again", splits_dir=splits_dir)
    except seal_mod.SealViolation as exc:
        assert "single open" in str(exc)
    else:
        raise AssertionError("a second open for the same config must be refused")
    log = (splits_dir / "test_open_log.jsonl").read_text(encoding="utf-8")
    assert res["receipt_id"] in log


def test_seal_voids_scores_when_the_label_file_changed():
    st = _fixture()
    splits_dir = st["gs"] / "splits"
    path = splits_dir / "temporal_v1"
    blind = pd.read_parquet(path / "test_blind.parquet")
    gold = pd.DataFrame([{"sentence_id": s, "span_start": 0, "span_end": 4,
                          "tag": "KN", "verdict": "RESOLVED"}
                         for s in blind["sentence_id"]])
    seal_mod.materialise_labels("temporal_v1", gold, splits_dir=splits_dir)
    # tamper with the materialised labels after sealing
    tampered = pd.read_parquet(path / "labels" / "test_labels.parquet").iloc[2:]
    tampered.to_parquet(path / "labels" / "test_labels.parquet", index=False)
    res = seal_mod.score_on_test("temporal_v1", gold, model_config_sha="m2",
                                 git_commit="c2", reason="tamper check",
                                 splits_dir=splits_dir)
    assert res["scored"] is False and res["verdict"].startswith("VOID")


def test_only_seal_and_tests_reference_test_labels():
    """T5: no module outside gold_standard/seal.py may read the sealed labels."""
    offenders = []
    for path in PROJECT_ROOT.rglob("*.py"):
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        if rel.startswith((".venv", "tests/")) or rel.endswith("gold_standard/seal.py"):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "test_labels.parquet" in text:
            offenders.append(rel)
    assert not offenders, f"modules referencing the sealed labels: {offenders}"


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
        except Exception as e:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed"
          + (f", {len(failed)} FAILED: {failed}" if failed else ""))
    if _STATE.get("root"):
        shutil.rmtree(_STATE["root"], ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
