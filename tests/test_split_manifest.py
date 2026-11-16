"""Tests for the SPLIT MANIFEST: targets, achieved, statistics and prose.

Standalone runner (the repo has no pytest):

    .venv-ml\\Scripts\\python.exe tests\\test_split_manifest.py

Scope. This file covers the manifest / reporting / statistics layer of
`gold_standard.split` only — the allocator and the leakage invariants are
covered by `tests/test_split.py`. Every test here is written against a defect
that was REPRODUCED on real data by the adversarial review, so each one is a
regression test with a known failing predecessor, named in the test's docstring:

  * the target of a regime-determined split taken FROM the achieved assignment
    (so every delta was 0, `exact_sizes_met` true and `unmet_targets` empty
    while the test set was 212 against a registered 240);
  * per-axis targets apportioned per axis VALUE, so the target column did not
    close to the part size and every delta / tv_distance carried the residue;
  * the temporal confound report covering four axes and omitting
    `occupation_group` (TV 0.4741) and `country_iso`;
  * a hardcoded `claimability_note` ("~0.05 F1") beside a computed MDE of 0.186;
  * `n_cells_final` reporting the number of STRATA;
  * `n_employers_spanning_boundary` summing boolean ROWS;
  * hardcoded corpus numbers in `year_stratification.evidence` and
    `limitations`;
  * a resolved cell that is absent from a part and can never be listed;
  * the registered temporal quantile never reported as an achieved share;
  * the verifiability block counting posting-level `employer_known` while the
    forcing rule acts on group-level verifiability.

Everything runs against the synthetic fixture (`gold_standard.synth_fixture`) or
on hand-built frames, so no test touches DATA/.
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

from gold_standard import (MAX_SENT_PER_GROUP, MIN_CELL_SENTENCES,      # noqa: E402
                          MIN_TEST_SENTENCES, TEMPORAL_QUANTILE)
from gold_standard import groups as groups_mod                          # noqa: E402
from gold_standard import posting_keys as keys_mod                      # noqa: E402
from gold_standard import sample as sample_mod                          # noqa: E402
from gold_standard import split as split_mod                            # noqa: E402
from gold_standard import synth_fixture                                 # noqa: E402
from gold_standard.split_assert import PARTS                            # noqa: E402

FIXTURE_TARGETS = {"train": 120, "dev": 40, "test": 40}
FIXTURE_SAMPLE_N = 200
HOLDOUT_RULE = {"min_pool_postings": 40, "min_employers": 20,
                "min_candidate_sentences": 200}
REGIMES = ("grouped_random", "temporal", "cross_country")

_STATE: dict = {}


# --------------------------------------------------------------- fixture build
def _fixture() -> dict:
    """Synthetic corpus + keys + groups + sample + the three splits, built once."""
    if _STATE:
        return _STATE
    root = Path(tempfile.mkdtemp(prefix="splitman_"))
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
    # A pilot and a practice draw, as the real DATA/gold_standard holds, so the
    # quarantine reference sets this fixture hands the invariants are NOT empty
    # (an empty reference makes the quarantine claim vacuous). The main draw then
    # excludes those sentences AND their leakage groups, exactly as the contract
    # specifies for the main draw, so the exact sizes are still reachable.
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
    for regime in REGIMES:
        splits[regime] = split_mod.run_split(
            regime, "CONFIG_SMALL", gs_dir=gs, raw_root=raw_root,
            sample_name="main_200", registry_path=registry, verbose=False,
            holdout_rule=HOLDOUT_RULE)
    _STATE.update({"root": root, "gs": gs, "raw_root": raw_root, "meta": meta,
                   "registry": registry, "splits": splits, "sample": draw})
    return _STATE


def _manifest(regime: str) -> dict:
    return _fixture()["splits"][regime]["manifest"]


# ----------------------------------------------------- 1. target provenance
def test_sizes_target_is_the_registered_number_not_the_assignment():
    """Predecessor: `soft_targets['test'] = forced_test_weight`, i.e. the target
    WAS the achieved count, so delta was 0 by construction in every regime."""
    for regime in REGIMES:
        man = _manifest(regime)
        registered = man["registry"]["body"]["targets"]
        assert man["targets"] == {p: int(registered[p]) for p in PARTS}, regime
        for part in PARTS:
            assert man["sizes"][part]["target"] == int(registered[part]), \
                f"{regime}/{part}: target {man['sizes'][part]['target']} is not the " \
                f"registered {registered[part]}"
            assert (man["sizes"][part]["delta"]
                    == man["sizes"][part]["achieved"]
                    - man["sizes"][part]["target"]), f"{regime}/{part}"
        assert "split_registry" in man["targets_source"]


def test_regime_determined_shortfall_is_visible_as_a_nonzero_delta():
    """The fixture's soft regimes genuinely miss the registered sizes (the date
    cut and the country boundary cannot be bent), so the manifest must SAY so:
    a non-zero delta, a loud unmet_targets line, exact_sizes_met null."""
    loud = 0
    for regime in ("temporal", "cross_country"):
        man = _manifest(regime)
        assert man["exactness_claimed"] is False, regime
        assert man["exact_sizes_met"] is None, \
            f"{regime}: exact_sizes_met must be null where no exactness is claimed"
        missed = [p for p in PARTS if man["sizes"][p]["delta"] != 0]
        if missed:
            loud += 1
            assert man["unmet_targets"], \
                f"{regime} missed {missed} but unmet_targets is empty"
            for part in missed:
                assert any(line.startswith(f"{part}:")
                           for line in man["unmet_targets"]), f"{regime}/{part}"
        # the allocator's capacity object is published, separately and labelled
        cap = man["regime_capacity"]
        assert set(PARTS) <= set(cap["capacity"])
        assert cap["equals_registered_targets"] == (
            cap["capacity"] == man["targets"])
    assert loud, "neither soft regime missed a registered size — the test is vacuous"


def test_grouped_random_still_reports_the_registered_sizes_as_met():
    man = _manifest("grouped_random")
    assert man["exactness_claimed"] is True
    assert man["exact_sizes_met"] is True
    assert [man["sizes"][p]["delta"] for p in PARTS] == [0, 0, 0]
    assert man["unmet_targets"] == []
    # and the capacity arithmetic that got there is auditable in the artifact
    cap = man["regime_capacity"]["exact_capacity"]
    assert cap["registered"] == man["targets"]
    assert cap["realised_capacity"] == man["regime_capacity"]["capacity"]
    assert cap["n_assignable"] == sum(man["achieved"][p] for p in PARTS)


def test_achieved_is_recounted_from_the_written_artifacts():
    """`achieved` must be a recount of the files on disk, not an allocator
    return value."""
    for regime in REGIMES:
        res = _fixture()["splits"][regime]
        man, out = res["manifest"], Path(res["out_dir"])
        on_disk = {
            "train": len(pd.read_parquet(out / "train.parquet")),
            "dev": len(pd.read_parquet(out / "dev.parquet")),
            "test": len(pd.read_parquet(out / "test_blind.parquet")),
        }
        assert man["achieved"] == on_disk, f"{regime}: {man['achieved']} != {on_disk}"
        assignment = pd.read_parquet(out / "assignment.parquet")
        for part in PARTS:
            assert int((assignment["part"] == part).sum()) == on_disk[part], \
                f"{regime}/{part}"
        assert "recounted" in man["achieved_source"]


def test_underpowered_is_surfaced_next_to_sizes():
    """Predecessor: `underpowered` lived only inside invariants.I12 while the
    sizes block beside it reported delta 0."""
    for regime in REGIMES:
        man = _manifest(regime)
        block = man["underpowered"]
        assert block["n_test_sentences"] == man["achieved"]["test"]
        assert block["min_test_sentences"] == MIN_TEST_SENTENCES
        assert block["test_underpowered"] == (
            man["achieved"]["test"] < MIN_TEST_SENTENCES), regime
    # the fixture test sets are small on purpose, so the flag must be ON
    assert _manifest("grouped_random")["underpowered"]["test_underpowered"]


# ------------------------------------------- 2. apportionment closes its column
def test_axis_target_columns_close_to_the_part_total():
    """Two-stage largest remainder. Predecessor: one `largest_remainder` call per
    axis VALUE, so each value rounded in isolation and the column did not close
    (country_iso.dev summed to 243 for a 240-row part)."""
    rng = np.random.default_rng(7)
    values = [f"v{i}" for i in range(7)]
    df = pd.DataFrame({
        "stratum": rng.choice(["indonesia", "asean", "oecd"], 1200),
        "country_iso": rng.choice(values, 1200),
        "quarter": rng.choice(["2026Q1", "2026Q2", "2026Q3"], 1200),
    })
    for part_totals in ({"train": 720, "dev": 240, "test": 240},
                        {"train": 719, "dev": 240, "test": 212},
                        {"train": 1, "dev": 0, "test": 3}):
        tgt = split_mod.axis_targets(df, part_totals,
                                     axes=("stratum", "country_iso", "quarter"))
        for axis, per_part in tgt.items():
            for part, column in per_part.items():
                assert sum(column.values()) == part_totals[part], \
                    f"{axis}.{part} column sums to {sum(column.values())} for a " \
                    f"{part_totals[part]}-sentence part"


def test_axis_targets_are_empty_when_there_is_nothing_to_apportion():
    df = pd.DataFrame({"stratum": []})
    tgt = split_mod.axis_targets(df, {"train": 10, "dev": 0, "test": 0},
                                 axes=("stratum",))
    assert tgt == {"stratum": {"train": {}, "dev": {}, "test": {}}}


def test_every_marginal_column_closes_in_every_written_manifest():
    for regime in REGIMES:
        man = _manifest(regime)
        basis = man["marginals_basis"]
        assert basis["part_totals"] == man["achieved"], regime
        for axis, per_part in man["marginals"].items():
            for part, block in per_part.items():
                total = basis["part_totals"][part]
                assert block["target_total"] == sum(block["target"].values())
                assert block["target_total"] == total, \
                    f"{regime}/{axis}/{part}: target column {block['target_total']}" \
                    f" != part total {total}"
                assert block["apportioned_from_part_total"] == total
                assert block["target_column_closes"] is True
                assert block["achieved_total"] == man["achieved"][part]


def test_marginals_are_apportioned_over_assignable_rows_only():
    """Predecessor: targets were apportioned over `work` (the draw minus the
    temporal drops) while further groups were then EXCLUDED, so the whole column
    was offset by the excluded weight."""
    for regime in REGIMES:
        man = _manifest(regime)
        n_assignable = man["marginals_basis"]["n_assignable_rows"]
        assigned = sum(man["achieved"][p] for p in PARTS)
        assert n_assignable == assigned, \
            f"{regime}: apportioned over {n_assignable} rows but {assigned} were " \
            f"assigned"


def test_regime_partitioned_axes_are_marked_not_applicable():
    temporal = _manifest("temporal")["marginals"]
    assert temporal["quarter"]["test"].get("not_applicable") is True
    assert "partitions" in temporal["quarter"]["test"]["not_applicable_reason"]
    assert "not_applicable" not in temporal["stratum"]["test"]
    cross = _manifest("cross_country")["marginals"]
    assert cross["country_iso"]["test"].get("not_applicable") is True
    plain = _manifest("grouped_random")["marginals"]
    for axis, per_part in plain.items():
        for part, block in per_part.items():
            assert "not_applicable" not in block, f"{axis}/{part}"
    cost = _manifest("temporal")["allocation_cost"]
    assert cost["regime_partitioned_axes_excluded"] == ["quarter"]
    assert (cost["final_weighted_abs_deviation_excl_regime_axes"]
            <= cost["final_weighted_abs_deviation"])


# -------------------------------------------------- 3. temporal confound report
def _temporal_frame(occupation_late: str = "UNRESOLVED") -> pd.DataFrame:
    """Early/late frame with a PLANTED occupation_group confound: every late row
    carries a family that never occurs early."""
    rows = []
    for i in range(40):
        rows.append({
            "sentence_id": f"e{i}", "part": "train" if i % 4 else "dev",
            "stratum": "indonesia", "country_iso": "ID", "country_language": "ID_id",
            "posting_language": "id", "is_indonesian_sentence": True,
            "site": "indeed", "quarter": "2026Q1", "occupation_group": "UNRESOLVED",
            "length_band": "mid",
            "date_posted_utc": pd.Timestamp("2026-01-05", tz="UTC"),
        })
    for i in range(10):
        rows.append({
            "sentence_id": f"l{i}", "part": "test",
            "stratum": "indonesia", "country_iso": "ID", "country_language": "ID_id",
            "posting_language": "id", "is_indonesian_sentence": True,
            "site": "jobstreet", "quarter": "2026Q4",
            "occupation_group": "software_developer", "length_band": "mid",
            "date_posted_utc": pd.Timestamp("2026-11-05", tz="UTC"),
        })
    df = pd.DataFrame(rows)
    df["date_posted_utc"] = pd.to_datetime(df["date_posted_utc"], utc=True)
    return df


def test_temporal_confound_report_covers_occupation_group_and_country_iso():
    """Predecessor: the report iterated (site, posting_language, quarter,
    stratum) only, and set `confounded` from that subset — so a 0.47 TV distance
    on occupation_group was invisible in the artifact built to disclose
    confounds."""
    df = _temporal_frame()
    cuts = {"indonesia": pd.Timestamp("2026-11-01", tz="UTC")}
    block = split_mod.temporal_confound_block(df, cuts, set())
    report = block["temporal_confound_report"]
    for axis in ("occupation_group", "country_iso", "site", "posting_language",
                 "quarter", "stratum", "length_band"):
        assert axis in report, f"{axis} missing from the confound report"
    occ = report["occupation_group"]
    assert occ["tv_distance"] == 1.0, occ
    assert occ["over_threshold"] is True
    assert occ["early_only_categories"] == ["UNRESOLVED"]
    assert occ["late_only_categories"] == ["software_developer"]
    assert block["confounded"] is True
    assert "occupation_group" in block["confound_axes_over_threshold"]
    assert block["drift_attributable"] is False
    assert "occupation" in block["evidence"]
    assert "2,707" not in block["evidence"] and "Kitalulus" not in block["evidence"]


def test_temporal_confound_report_in_the_written_manifest_covers_every_axis():
    block = _manifest("temporal")["temporal"]
    report = block["temporal_confound_report"]
    axes_in_manifest = set(block["confound_axes_covered"])
    assert axes_in_manifest == set(report)
    assert {"occupation_group", "country_iso"} <= axes_in_manifest
    assert "recomputed" in block["evidence_source"]


def test_temporal_per_stratum_late_share_is_reported_against_the_quantile():
    """Predecessor: the regime registered a 0.80 quantile as its target and the
    manifest reported only the derived cut date, so a stratum below the
    registered share was unreported and unmet_targets stayed empty."""
    df = _temporal_frame()
    cuts = {"indonesia": pd.Timestamp("2026-11-01", tz="UTC")}
    block = split_mod.temporal_confound_block(df, cuts, set())
    per = block["per_stratum"]["indonesia"]
    assert per["quantile_target"] == round(1.0 - TEMPORAL_QUANTILE, 4)
    assert per["achieved_late_share"] == 0.2      # 10 test of 50 drawn
    assert per["n_test_sentences"] == 10 and per["n_drawn_sentences"] == 50
    assert per["delta"] == 0.0
    assert block["per_stratum_unmet"] == []

    # now starve the late side: 5 of 50 is a miss and must be reported as one
    short = df[~df["sentence_id"].isin({f"l{i}" for i in range(5)})].copy()
    block2 = split_mod.temporal_confound_block(short, cuts, set())
    per2 = block2["per_stratum"]["indonesia"]
    assert per2["achieved_late_share"] < per2["quantile_target"]
    assert per2["delta"] < 0
    assert any("temporal late share [indonesia]" in line
               for line in block2["per_stratum_unmet"])


def test_temporal_manifest_appends_a_missed_quantile_to_unmet_targets():
    man = _manifest("temporal")
    for line in man["temporal"]["per_stratum_unmet"]:
        assert line in man["unmet_targets"], line
    for stratum, per in man["temporal"]["per_stratum"].items():
        assert (per["delta"] < 0) == any(
            f"temporal late share [{stratum}]" in line
            for line in man["temporal"]["per_stratum_unmet"]), stratum


# ----------------------------------------------------------------- 4. power
def test_claimability_note_is_built_from_the_computed_mde():
    """Predecessor: a constant string ('nothing below ~0.05 F1 ... ~1.5x wider
    still') emitted verbatim next to a computed MDE of 0.1856."""
    notes = {}
    for n_test in (40, 240):
        df = pd.DataFrame({
            "part": ["test"] * n_test + ["train"] * 10,
            "stratum": ["indonesia"] * (n_test // 2)
                       + ["oecd"] * (n_test - n_test // 2) + ["indonesia"] * 10,
        })
        block = split_mod.power_block(df)
        mde = block["paired_mde_80pct_power"]
        note = block["claimability_note"]
        notes[n_test] = note
        assert f"{mde:.3f}" in note, (mde, note)
        assert f"n_test={n_test}" in note
        assert "~0.05" not in note
        ratio = (block["indonesia_only"]["ci_width_at_0_5"]
                 / block["expected_micro_f1_ci_width_at_0_5"])
        assert f"{ratio:.2f}x" in note, (ratio, note)
    assert notes[40] != notes[240], "the note did not respond to the test size"


def test_claimability_note_survives_an_empty_test_set():
    df = pd.DataFrame({"part": ["train", "train"], "stratum": ["oecd", "oecd"]})
    block = split_mod.power_block(df)
    assert block["paired_mde_80pct_power"] is None
    assert "nothing is claimable" in block["claimability_note"]


# --------------------------------------------------- 5. stratification counting
def _cell_frame() -> pd.DataFrame:
    """3 strata x several cells; one BIG cell deliberately absent from test."""
    rows = []

    def add(n, stratum, occ, quarter, site, parts):
        for i in range(n):
            rows.append({"sentence_id": f"{stratum}-{occ}-{quarter}-{site}-{i}",
                         "part": parts[i % len(parts)], "stratum": stratum,
                         "occupation_group": occ,
                         "occupation_group_coarse": "other",
                         "quarter": quarter, "site": site,
                         "date_posted_utc": pd.Timestamp("2026-03-01", tz="UTC")})
    # resolved at L0 but NEVER in test
    add(15, "oecd", "web_developer", "2026Q3", "indeed", ["train", "dev"])
    # resolved at L0, all three parts present
    add(30, "indonesia", "software_developer", "2026Q1", "jobstreet",
        ["train", "dev", "test"])
    add(20, "indonesia", "software_developer", "2026Q2", "indeed",
        ["train", "dev", "test"])
    add(12, "indonesia", "devops", "2026Q1", "glints", ["train", "dev", "test"])
    # small cells that must collapse
    add(4, "asean", "devops", "2026Q2", "glints", ["train"])
    add(3, "asean", "devops", "2026Q3", "glints", ["train", "test"])
    df = pd.DataFrame(rows)
    df["date_posted_utc"] = pd.to_datetime(df["date_posted_utc"], utc=True)
    return df


def test_n_cells_final_counts_cells_not_strata():
    """Predecessor: `n_cells_final` was len(set(strata)) = 3, published next to
    `n_unfilled_cells` in the hundreds."""
    block = split_mod.stratification_block(_cell_frame())
    assert block["n_strata"] == 3
    # four L0 cells clear MIN_CELL_SENTENCES; the asean residual terminates at L4
    assert block["n_cells_final_by_level"] == {"L0": 4, "L1": 0, "L2": 0,
                                               "L3": 0, "L4": 1}
    assert block["n_cells_final"] == 5
    assert block["n_cells_final"] > block["n_strata"], block["n_cells_final"]
    assert sum(block["n_cells_final_by_level"].values()) == block["n_cells_final"]


def test_resolution_shares_sum_to_one_and_the_terminal_level_is_reachable():
    """Predecessor: the residual rows at L4 were measured against the residual's
    own cell sizes, so L4 was 0.0 and the shares summed to 0.935."""
    block = split_mod.stratification_block(_cell_frame())
    assert abs(block["resolution_shares_sum"] - 1.0) < 1e-6, \
        block["achieved_stratum_resolution"]
    assert block["achieved_stratum_resolution"]["L4"] > 0.0


def test_a_resolved_cell_missing_from_a_part_is_listed():
    """Predecessor: `unfilled_cells` walks only cells BELOW
    MIN_CELL_SENTENCES, so a cell big enough to count as resolved could be
    entirely absent from a part and appear nowhere."""
    block = split_mod.stratification_block(_cell_frame())
    listed = {tuple(e["cell"]): e for e in block["resolved_but_missing_from_part"]}
    key = ("oecd", "web_developer", "2026Q3", "indeed")
    assert key in listed, listed
    assert listed[key]["missing_from_part"] == ["test"]
    assert listed[key]["sample_n"] >= MIN_CELL_SENTENCES
    assert block["n_resolved_but_missing_from_part"] == len(
        block["resolved_but_missing_from_part"])
    # and the small cells are still reported as unfilled
    assert block["n_unfilled_cells"] >= 1
    for entry in block["unfilled_cells"]:
        assert entry["sample_n"] < MIN_CELL_SENTENCES


def test_year_stratification_verdict_and_evidence_are_derived():
    """Predecessor: verdict 'NOT ACHIEVED (indonesia)' and an evidence string
    citing '~79% 2026' and '89 eligible postings at or before 2024' were
    constants, contradicted by the counts in the same block."""
    one_year = _cell_frame()
    block = split_mod.stratification_block(one_year)
    year = block["year_stratification"]
    assert year["verdict"].startswith("NOT ACHIEVED")
    for stratum in ("indonesia", "asean", "oecd"):
        assert stratum in year["verdict"]
        assert year["per_stratum_detail"][stratum]["stratified"] is False
    assert "89 eligible" not in year["evidence"] and "79%" not in year["evidence"]
    assert "modal year" in year["evidence"]
    assert year["observed_date_window"][0].startswith("2026-03-01")

    # two well-populated years in one stratum -> that stratum IS stratified
    two = _cell_frame()
    extra = two[two["stratum"] == "indonesia"].copy()
    extra["sentence_id"] = extra["sentence_id"] + "-2025"
    extra["date_posted_utc"] = pd.Timestamp("2025-03-01", tz="UTC")
    two = pd.concat([two, extra], ignore_index=True)
    block2 = split_mod.stratification_block(two)
    year2 = block2["year_stratification"]
    assert year2["per_stratum_detail"]["indonesia"]["stratified"] is True
    assert "indonesia" not in year2["verdict"]
    assert year2["per_stratum_detail"]["indonesia"]["n_years_at_or_above_min_cell"] == 2


def test_written_manifest_year_verdict_matches_its_own_counts():
    for regime in REGIMES:
        year = _manifest(regime)["stratification"]["year_stratification"]
        failing = [st for st, d in year["per_stratum_detail"].items()
                   if not d["stratified"]]
        if failing:
            assert year["verdict"].startswith("NOT ACHIEVED")
            for st in failing:
                assert st in year["verdict"], (regime, st)
        else:
            assert year["verdict"] == "ACHIEVED", regime
        for st, detail in year["per_stratum_detail"].items():
            counts = year["per_stratum_year_counts"][st]
            assert detail["n_assigned_sentences"] == sum(counts.values())
            assert detail["n_distinct_years"] == len(counts)


# ------------------------------------------------------- 6. cross-country count
def test_n_employers_spanning_boundary_counts_employers_not_rows():
    """The review's own counter-example: three US rows of ONE employer that also
    posts in GB. The old boolean row-sum returned 3; the answer is 1."""
    df = pd.DataFrame({
        "country_iso": ["US", "US", "US", "GB", "GB"],
        "employer_hash": ["e1", "e1", "e1", "e1", "e2"],
    })
    assert split_mod._n_employers_spanning(df, "US") == 1
    # two distinct spanning employers, many rows each
    df2 = pd.DataFrame({
        "country_iso": ["US"] * 5 + ["GB"] * 4,
        "employer_hash": ["e1", "e1", "e2", "e2", "e3", "e1", "e2", "e2", "e9"],
    })
    assert split_mod._n_employers_spanning(df2, "US") == 2
    # a missing hash is never an employer
    df3 = pd.DataFrame({"country_iso": ["US", "GB", "US", "GB"],
                        "employer_hash": ["", "", "nan", "nan"]})
    assert split_mod._n_employers_spanning(df3, "US") == 0


def test_cross_country_manifest_reports_the_employer_count_with_its_scope():
    man = _manifest("cross_country")["cross_country"]
    iso = man["holdout_country"]
    df = _fixture()["splits"]["cross_country"]["assignment"]
    assert man["n_employers_spanning_boundary"] == \
        split_mod._n_employers_spanning(df, iso)
    assert "DISTINCT" in man["n_employers_spanning_boundary_scope"]


def test_cross_country_manifest_reports_scope_and_test_purity():
    """The hold-out rule's scope and the external test's achieved country mix are
    computed; they must also be REPORTED, or a reader infers purity from the
    regime's name."""
    man = _manifest("cross_country")["cross_country"]
    assert man["holdout_group_scope"] == "pool"
    for key in ("n_groups_touching_holdout_in_pool",
                "n_groups_touching_holdout_in_sample",
                "n_groups_recovered_by_pool_scope"):
        assert isinstance(man[key], int), key
    assert (man["n_groups_touching_holdout_in_sample"]
            <= man["n_groups_touching_holdout_in_pool"])
    purity = man["purity"]
    assert purity["country_purity"] == "NOT claimed"
    composition = purity["test_country_composition"]
    achieved_test = _manifest("cross_country")["achieved"]["test"]
    assert sum(composition.values()) == achieved_test
    assert (purity["n_test_sentences_from_holdout_country"]
            == composition.get(man["holdout_country"], 0))


# ------------------------------------------------------------ 7. verifiability
def test_verifiability_counts_group_level_confinement_too():
    """Predecessor: the block counted posting-level ~employer_known while the
    forcing rule acts on group-level `group_employer_verifiable`, so sentences
    confined to train only through a group-mate were never counted (82 reported
    against 85 confined on the real draw)."""
    df = pd.DataFrame({
        "sentence_id": ["s1", "s2", "s3", "s4"],
        "employer_known": [True, True, False, True],
        "group_employer_verifiable": [True, False, False, True],
        "stratum": ["indonesia"] * 4,
        "site": ["indeed", "jobstreet", "jobstreet", "indeed"],
    })
    block = split_mod._verifiability_block(df)
    assert block["n_unknown_employer_sentences"] == 1
    assert block["n_sentences_confined_by_group_unverifiability"] == 2
    assert block["n_sentences_confined_only_via_a_group_mate"] == 1
    assert block["sentences_confined_only_via_a_group_mate"] == ["s2"]
    assert block["by_site"] == {"jobstreet": 1}
    assert block["by_site_group_level"] == {"jobstreet": 2}


def test_written_manifest_verifiability_is_at_least_the_posting_level_count():
    for regime in REGIMES:
        block = _manifest(regime)["verifiability"]
        assert (block["n_sentences_confined_by_group_unverifiability"]
                >= block["n_unknown_employer_sentences"]), regime
        assert "group level" in block["counting_note"] or \
               "group_employer_verifiable" in block["counting_note"]


# --------------------------------------------------------------- 8. limitations
def test_limitations_carry_no_stale_hardcoded_corpus_numbers():
    """Predecessor: seven constant strings, including '290 eligible Indonesian
    postings (5.07%; 14.6% of eligible Jobstreet ID)', emitted verbatim on a
    420-posting fixture they do not describe."""
    keys_manifest = json.loads(
        (_fixture()["gs"] / "posting_keys_manifest.json").read_text(encoding="utf-8"))
    missing = keys_manifest.get("missing_employer_by_stratum") or {}
    for regime in REGIMES:
        lim = _manifest(regime)["limitations"]
        assert lim, regime
        text = " || ".join(lim)
        for stale in ("290 eligible", "5.07%", "14.6%", "~79% 2026",
                      "89 eligible postings"):
            assert stale not in text, f"{regime}: stale constant '{stale}'"
        # the employer-coverage figure is the one read back from the artifact
        for stratum, counts in missing.items():
            n = int((counts or {}).get("missing_employer", 0))
            if n:
                assert f"{stratum} {n}/" in text, (regime, stratum, n)
        assert "EDITORIAL NOTE" in text, \
            f"{regime}: the fixed literature anchor must be labelled as such"
        assert any(line.startswith("power: ") for line in lim), regime


# ------------------------------------------------------------- 9. input guards
def test_a_missing_posting_keys_manifest_is_refused_not_a_traceback():
    """Predecessor: posting_keys_manifest.json was read unguarded in the middle
    of manifest construction, so a missing one produced a bare
    FileNotFoundError instead of the SplitRefusal every sibling input gets."""
    st = _fixture()
    root = Path(tempfile.mkdtemp(prefix="splitman_noman_"))
    try:
        gs = root / "gold_standard"
        shutil.copytree(st["gs"], gs)
        shutil.rmtree(gs / "splits", ignore_errors=True)
        (gs / "posting_keys_manifest.json").unlink()
        try:
            split_mod.run_split("grouped_random", "CONFIG_SMALL", gs_dir=gs,
                                raw_root=st["raw_root"], sample_name="main_200",
                                registry_path=gs / "registry.json", verbose=False)
        except split_mod.SplitRefusal as exc:
            assert "posting_keys_manifest.json" in str(exc)
            assert "gold_standard.posting_keys" in str(exc)
        else:
            raise AssertionError("a missing posting_keys_manifest.json was accepted")
        assert not (gs / "splits" / "grouped_random_v1").exists(), \
            "the refusal still left a split on disk"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_recount_mismatch_between_artifacts_and_assignment_is_refused():
    """The achieved recount is cross-checked against the assignment frame, and a
    disagreement refuses the write rather than publishing either number."""
    st = _fixture()
    original = split_mod._part_frame

    def _short_train(df, part, columns):
        frame = original(df, part, columns)
        return frame.iloc[1:] if part == "train" else frame

    root = Path(tempfile.mkdtemp(prefix="splitman_recount_"))
    try:
        gs = root / "gold_standard"
        shutil.copytree(st["gs"], gs)
        shutil.rmtree(gs / "splits", ignore_errors=True)
        split_mod._part_frame = _short_train
        try:
            split_mod.run_split("grouped_random", "CONFIG_SMALL", gs_dir=gs,
                                raw_root=st["raw_root"], sample_name="main_200",
                                registry_path=gs / "registry.json", verbose=False)
        except split_mod.SplitRefusal as exc:
            assert "recount mismatch" in str(exc)
        else:
            raise AssertionError("a part artifact short of the assignment was written")
        finally:
            split_mod._part_frame = original
    finally:
        split_mod._part_frame = original
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------- 10. manifest self-consistency
def test_every_target_block_has_an_achieved_counterpart_with_a_true_delta():
    """I16's intent, checked on VALUES here rather than on key presence."""
    for regime in REGIMES:
        man = _manifest(regime)
        for part, block in man["sizes"].items():
            assert block["delta"] == block["achieved"] - block["target"], \
                f"{regime}/{part}"
        for axis, per_part in man["marginals"].items():
            for part, block in per_part.items():
                for value, delta in block["delta"].items():
                    assert delta == (block["achieved"].get(value, 0)
                                     - block["target"].get(value, 0)), \
                        f"{regime}/{axis}/{part}/{value}"
                assert sum(block["achieved"].values()) == man["achieved"][part]


def test_version_log_carries_the_same_unmet_targets_as_the_manifest():
    st = _fixture()
    lines = [json.loads(line) for line in
             (st["gs"] / "splits" / "version_log.jsonl")
             .read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {entry["split_id"]: entry for entry in lines}
    for regime in REGIMES:
        man = _manifest(regime)
        entry = by_id[man["identity"]["split_id"]]
        assert entry["unmet_targets"] == man["unmet_targets"], regime
        assert entry["sizes"] == man["achieved"], regime


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
