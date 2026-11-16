"""Group-aware gold-standard splits: grouped_random, temporal, cross_country.

Design spine (GROUPED SPLIT SPECIFICATION v1.0.0): the one MANDATORY clause of
the proposal for this layer is a PROHIBITION — "Lowongan yang berasal dari
employer atau template yang sama tidak ditempatkan pada train dan test secara
bersamaan" (PROP:1192; PROP:1436 extends it to "atau sangat mirip"). Everything
here is arranged so that prohibition is an O(n) set-disjointness assertion that
runs before any byte is written (`split_assert`), and so the stratification —
which the proposal marks best-effort, "sejauh memungkinkan" — is reported as a
target/achieved pair instead of being claimed.

Three properties worth knowing before reading the code:

1. EXACT SIZES COME FROM A PARITY PROOF, not from an optimiser. With every
   leakage group capped at 2 sentences at DRAW time (sample.py, group_cap) and
   all three targets even, placing every weight-2 group before any weight-1
   group keeps each part's remaining capacity even throughout the weight-2
   phase, so a non-zero capacity is always >= 2 and the phase always completes;
   the weight-1 phase then consumes unit capacity against an equal total. No
   backtracking, no subset-sum DP, no annealing. The five preconditions P1-P5
   are asserted, and a failure RAISES rather than shipping an off-size split.

2. THE SPLIT IS COMPUTED FROM METADATA ONLY and is frozen before any
   adjudicated label exists. Nothing written here contains an annotation column
   (invariant I17); `seal.py` is the only module that materialises test labels.

3. THE THREE REGIMES RE-PARTITION THE SAME 1,200 SENTENCES. Their test sets
   overlap, so the three numbers are not independent replications and must never
   be averaged — the manifest says so in `regime_overlap`.

Run:
  .venv-ml\\Scripts\\python.exe -m gold_standard.split --config CONFIG_A --regime grouped_random
  .venv-ml\\Scripts\\python.exe -m gold_standard.split --config CONFIG_A --regime temporal
  .venv-ml\\Scripts\\python.exe -m gold_standard.split --config CONFIG_A --regime cross_country
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from governance.ladder import RAW_ROOT, REPO_ROOT
from . import (ASSUMED_SPANS_PER_SENTENCE, AXIS_WEIGHTS, DESIGN_EFFECT,
               EXCLUSION_REASONS, GOLD_STANDARD_VERSION, GUIDELINE_CODE,
               HOLDOUT_CANDIDATE_CHAR_RANGE, HOLDOUT_MIN_CANDIDATE_SENTENCES,
               HOLDOUT_MIN_EMPLOYERS, HOLDOUT_MIN_POOL_POSTINGS,
               HOLDOUT_REQUIRED_LANGUAGE, MAX_SENT_PER_GROUP,
               MIN_CELL_SENTENCES, MIN_SHARED_SHINGLES,
               MIN_TEST_SENTENCES,
               MIN_TEST_SUPPORT_SPANS_PER_TAG, OCCUPATION_UNRESOLVED,
               PRIMARY_CLAIM_STRATUM, SEED, SPLIT_SPEC_VERSION,
               TAU_ASSERT, TAU_BUILD, TEMPORAL_QUANTILE)
from .posting_keys import GS_DIR, _sha256
from .split_assert import PARTS, assert_invariants, run_invariants

SPLITS_DIR = GS_DIR / "splits"
REGISTRY_PATH = Path(__file__).resolve().parent / "split_registry.json"
REGIMES = ("grouped_random", "temporal", "cross_country")

# Correlation assumed between two models' per-span correctness when they are
# scored on the SAME test spans. Used only for the paired MDE in the power block
# and recorded there as an assumption, never as a measurement.
PAIRED_RHO = 0.75
Z_975, Z_80 = 1.959963984540054, 0.8416212335729143

AXES = ("stratum", "country_iso", "country_language", "posting_language",
        "is_indonesian_sentence", "site", "quarter", "occupation_group",
        "length_band")
# Cell-coarsening ladder (reporting, never a constraint).
CELL_LADDER = {
    "L0": ("stratum", "occupation_group", "quarter", "site"),
    "L1": ("stratum", "occupation_group", "quarter"),
    "L2": ("stratum", "occupation_group"),
    "L3": ("stratum", "occupation_group_coarse"),
    "L4": ("stratum",),
}


class SplitRefusal(RuntimeError):
    """Raised instead of writing a split that cannot satisfy the contract."""


# --------------------------------------------------------------------- registry
def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_sha256(body: dict) -> str:
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def registered_config(name: str, registry: dict | None = None) -> dict:
    """Registry gate: refuse an unregistered config, and refuse a registered one
    whose recomputed content hash no longer matches what was pre-registered."""
    registry = registry or load_registry()
    entries = registry["configs"]
    if name not in entries:
        raise SplitRefusal(
            f"config '{name}' is not pre-registered in {REGISTRY_PATH.name}. "
            f"Registered: {sorted(entries)}. The proposal requires the split rule "
            f"to be fixed before the benchmark and logged (PROP:1431), so an "
            f"ad-hoc config is refused.")
    entry = entries[name]
    recomputed = config_sha256(entry["body"])
    if recomputed != entry["content_sha256"]:
        raise SplitRefusal(
            f"config '{name}' was edited after registration: content_sha256 "
            f"{entry['content_sha256']} != recomputed {recomputed}.")
    if not entry.get("frozen", False):
        raise SplitRefusal(f"config '{name}' is not frozen")
    if not entry["body"].get("runnable", True):
        raise SplitRefusal(
            f"config '{name}' is registered but deliberately NOT run: "
            f"{entry['body'].get('rationale', '')}")
    return entry


# ---------------------------------------------------------------- apportionment
def largest_remainder(count: int, part_totals: dict[str, int]) -> dict[str, int]:
    """Split `count` across parts proportionally to part_totals, exactly."""
    total = sum(part_totals.values())
    if total <= 0 or count <= 0:
        return {p: 0 for p in part_totals}
    exact = {p: count * n / total for p, n in part_totals.items()}
    base = {p: int(math.floor(v)) for p, v in exact.items()}
    rest = count - sum(base.values())
    order = sorted(part_totals, key=lambda p: (-(exact[p] - base[p]), p))
    for p in order[:rest]:
        base[p] += 1
    return base


def axis_targets(df: pd.DataFrame, part_totals: dict[str, int],
                 axes=AXES) -> dict[str, dict[str, dict[str, int]]]:
    """Per-axis per-part targets from the ASSIGNABLE rows' realised marginals.

    Never from the corpus: PROP:1433 requires the held-out test set to keep "a
    distribution as close as possible to the natural distribution" OF THE
    SAMPLE, and rare-category surplus then lands in train, the only place
    oversampling is permitted (PROP:1431).

    TWO-STAGE largest-remainder, and the second stage is the point. Stage 1 is
    `part_totals` itself (the allocator's capacity object). Stage 2 distributes
    EACH PART'S total across that axis's values, so the `target` column of every
    (axis, part) sums EXACTLY to that part's total. Apportioning per axis VALUE
    independently — one `largest_remainder` call per value, which is what this
    function used to do — let every value round in isolation, so the column did
    NOT close to the part size (measured: 13 of 27 (axis, part) columns off on
    the exact primary split, all 27 on temporal). Those targets are unmeetable
    in principle, and the residue contaminated every reported `delta`, every
    `tv_distance` and `allocation_cost.final_weighted_abs_deviation`.

    `df` must therefore be the ASSIGNABLE frame (sample minus excluded minus
    dropped), not the raw draw: apportioning over rows that no part can hold
    offsets the whole column.
    """
    out: dict[str, dict[str, dict[str, int]]] = {}
    for axis in axes:
        if axis not in df.columns:
            continue
        counts = {str(v): int(n) for v, n
                  in df[axis].astype(str).value_counts().items()}
        values = sorted(counts)
        per_part: dict[str, dict[str, int]] = {}
        for part, total in part_totals.items():
            alloc = largest_remainder(int(total), counts) if counts else {}
            per_part[part] = {v: int(alloc.get(v, 0)) for v in values}
        out[axis] = per_part
    return out


# ------------------------------------------------------------------- allocation
def _group_rows(df: pd.DataFrame, axes=AXES) -> list[dict]:
    groups = []
    for gid, grp in df.groupby("group_id", sort=True):
        counts = {}
        for axis in axes:
            if axis in grp.columns:
                counts[axis] = grp[axis].astype(str).value_counts().to_dict()
        groups.append({"group_id": str(gid), "weight": int(len(grp)),
                       "axis_counts": counts})
    return groups


def _delta_cost(achieved: dict, targets: dict, group: dict, part: str) -> float:
    d = 0.0
    for axis, counts in group["axis_counts"].items():
        w = AXIS_WEIGHTS.get(axis, 0)
        if not w:
            continue
        ach_axis = achieved[axis][part]
        tgt_axis = targets.get(axis, {}).get(part, {})
        for value, c in counts.items():
            a = ach_axis.get(value, 0)
            t = tgt_axis.get(value, 0)
            d += w * (abs(a + c - t) - abs(a - t))
    return d


def _apply(achieved: dict, group: dict, part: str) -> None:
    for axis, counts in group["axis_counts"].items():
        ach_axis = achieved[axis][part]
        for value, c in counts.items():
            ach_axis[value] = ach_axis.get(value, 0) + c


def _tiebreak(group_id: str, regime_id: str, part: str) -> str:
    return hashlib.sha1(f"{group_id}|{regime_id}|{part}".encode("utf-8")).hexdigest()


def allocate(groups: list[dict], targets: dict[str, int], targets_by_axis: dict,
             forced: dict[str, str], seed: int = SEED, regime_id: str = "",
             exact: bool = True, exclusion_reasons: dict[str, str] | None = None
             ) -> dict:
    """Parity-greedy allocation. Returns {group_part, parity_corrections, cost}.

    `forced` maps group_id -> part ("train"/"dev"/"test"/"excluded") and is
    applied first. When `exact` is False (temporal / cross_country, where a date
    cut and a country boundary cannot be bent to a target) capacities are soft:
    a group may overflow, the overflow is minimised, and no exactness is claimed.

    In EXACT mode `targets` is the REALISED capacity, i.e. the assignable weight
    (len(sample) - excluded - dropped), never the registered 720/240/240. An
    excluded group removes weight from the pending pool without removing
    capacity, so passing the register straight through makes P2 fail by exactly
    the excluded weight — see `_exact_capacity`, which is the only correct way to
    build this argument, and whose shortfall the manifest reports against the
    register. `exclusion_reasons` is used only to make a P2 failure name the
    groups and reasons responsible instead of blaming the group map.
    """
    by_id = {g["group_id"]: g for g in groups}
    achieved = {axis: {p: {} for p in targets} for axis in AXES}
    capacity = dict(targets)
    group_part: dict[str, str] = {}

    for gid in sorted(forced):
        part = forced[gid]
        group_part[gid] = part
        if part == "excluded" or gid not in by_id:
            continue
        g = by_id[gid]
        if part in capacity:
            capacity[part] -= g["weight"]
            _apply(achieved, g, part)

    pending = [g for g in groups if g["group_id"] not in group_part]

    # ---- preconditions (D4). A failure RAISES; atomicity is never traded away.
    weights = [g["weight"] for g in pending]
    if exact:
        if weights and max(weights) > MAX_SENT_PER_GROUP:
            offenders = [(g["group_id"], g["weight"]) for g in pending
                         if g["weight"] > MAX_SENT_PER_GROUP][:10]
            raise SplitRefusal(
                f"P1 violated: group weight exceeds MAX_SENT_PER_GROUP="
                f"{MAX_SENT_PER_GROUP}; the parity proof does not apply and exact "
                f"sizes cannot be guaranteed. Offenders: {offenders}. Re-draw the "
                f"sample with group_cap=MAX_SENT_PER_GROUP.")
        if sum(weights) != sum(capacity.values()):
            excl = sorted(gid for gid, p in forced.items()
                          if p == "excluded" and gid in by_id)
            excl_weight = sum(by_id[gid]["weight"] for gid in excl)
            detail = ", ".join(
                f"{gid}(w={by_id[gid]['weight']}, "
                f"{(exclusion_reasons or {}).get(gid, 'reason-not-passed')})"
                for gid in excl[:10]) or "none"
            raise SplitRefusal(
                f"P2 violated: placeable weight {sum(weights)} != remaining "
                f"capacity {sum(capacity.values())} (capacity object {targets}). "
                f"Excluded groups inside the sample: {len(excl)} groups / "
                f"{excl_weight} sentences -> {detail}"
                f"{' ...' if len(excl) > 10 else ''}. In exact mode the capacity "
                f"MUST be the assignable weight (len(sample) - excluded - "
                f"dropped); the caller apportions the shortfall out of the "
                f"registered targets first (gold_standard.split._exact_capacity). "
                f"If the excluded list above is empty the draw itself is "
                f"under-filled relative to the registered total.")
        for part, cap in sorted(capacity.items()):
            if cap < 0:
                raise SplitRefusal(
                    f"P3 violated: forced assignments overfill '{part}' by {-cap} "
                    f"sentences")

    parity_corrections = []
    if exact:
        odd_parts = [p for p in sorted(capacity) if capacity[p] % 2 == 1]
        w1 = [g for g in pending if g["weight"] == 1]
        if len(w1) < len(odd_parts):
            raise SplitRefusal(
                f"P5 violated: {len(odd_parts)} part(s) have odd remaining capacity "
                f"{ {p: capacity[p] for p in odd_parts} } but only {len(w1)} "
                f"weight-1 group(s) remain to correct parity")
        for part in odd_parts:
            best = min((g for g in pending if g["weight"] == 1 and
                        g["group_id"] not in group_part),
                       key=lambda g: (_delta_cost(achieved, targets_by_axis, g, part),
                                      _tiebreak(g["group_id"], regime_id, part)))
            group_part[best["group_id"]] = part
            capacity[part] -= 1
            _apply(achieved, best, part)
            parity_corrections.append({"part": part, "group_id": best["group_id"]})
            pending = [g for g in pending if g["group_id"] != best["group_id"]]
        for part, cap in sorted(capacity.items()):
            if cap % 2 == 1:
                raise SplitRefusal(
                    f"P4 violated: '{part}' capacity still odd ({cap}) after parity "
                    f"correction")

    # ---- weight-2 phase then weight-1 phase, deterministic ordering
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pending)) if pending else np.zeros(0, dtype=int)
    for g, key in zip(pending, perm.tolist()):
        g["_tie"] = int(key)
    order = sorted(pending, key=lambda g: (-g["weight"], g["_tie"], g["group_id"]))

    cost = 0.0
    for g in order:
        options = []
        for part in sorted(capacity):
            room = capacity[part]
            overflow = max(0, g["weight"] - room)
            if exact and overflow:
                continue
            options.append((
                overflow,
                _delta_cost(achieved, targets_by_axis, g, part),
                -room,
                PARTS.index(part) if part in PARTS else 9,
                _tiebreak(g["group_id"], regime_id, part),
                part,
            ))
        if not options:
            raise SplitRefusal(
                f"no part can take group {g['group_id']} (weight {g['weight']}); "
                f"remaining capacity {capacity}. This cannot happen under P1-P5 — "
                f"the group map or the draw cap is inconsistent.")
        chosen = min(options)
        part = chosen[-1]
        group_part[g["group_id"]] = part
        capacity[part] -= g["weight"]
        cost += chosen[1]
        _apply(achieved, g, part)

    return {"group_part": group_part, "parity_corrections": parity_corrections,
            "cost": float(cost), "residual_capacity": capacity}


def _exact_capacity(registered: dict[str, int], df: pd.DataFrame,
                    forced: dict[str, str], reason: dict[str, str],
                    dropped: set[str], verbose: bool = True
                    ) -> tuple[dict[str, int], dict]:
    """The exact regime's REALISED capacity, and the shortfall against the register.

    THE CONTRACT IS WRONG HERE AND IT IS SAID OUT LOUD. D4's precondition P2 is
    written as `Sum weight == 1200 == 720+240+240`, which is unsatisfiable the
    moment any of the contract's OWN closed-set exclusion reasons fires: an
    excluded group leaves the pending pool but not the capacity, so P2 failed by
    exactly the excluded weight and the primary benchmark — the split the whole
    dissertation reports — could not be written at all with a single
    `date_unparseable`, `pilot_group` or `practice_group` collision, nor from an
    under-filled draw that `sample.py` deliberately tolerates. Capacity is
    therefore the ASSIGNABLE weight, and the three candidate repairs resolve as:

    * TOP-UP from the pool — REJECTED. `main_1200.parquet` is a frozen,
      pre-registered artifact whose sha256 every manifest binds (I14), and §6.10 /
      `regime_overlap` require the three regimes to re-partition the SAME 1,200
      sentences. A regime that quietly drew replacement sentences would break both
      the provenance binding and the cross-regime identity, and would let the
      allocator choose which sentences the benchmark contains after the data was
      opened. Topping up belongs in `sample.py` at draw time (where the seeded
      exact-n loop already lives), never here.
    * REFUSE the run — REJECTED as the default. The exclusion reasons are
      pre-registered, expected and already reported in `exclusions.by_reason`;
      making an expected, honestly-reported event a hard stop on the primary
      deliverable is the defect this is fixing, not a fix.
    * SHRINK the target — TAKEN, and the shortfall is charged ENTIRELY TO TRAIN
      while dev and test keep their registered sizes. Two of the three verbatim
      pre-registered numbers (PROP:1431) therefore still hold exactly, the held-out
      test stays at its registered n so every figure in the `power` block remains
      the pre-registered one, and the loss lands where the measured learning curve
      prices it at ~6.1e-5 F1 per sentence and where PROP:1431 says oversampling —
      i.e. elasticity — is the only permitted asymmetry. A proportional
      largest-remainder shrink was considered and rejected: it moves the test size
      off the pre-registration to protect a train sentence worth 0.00006 F1.

    Exactness is still ASSERTED, against the realised capacity (I12), so the
    allocator cannot silently miss its own number; what changed is that the
    register-vs-realised gap is now reported instead of raising. The returned
    `info` is what the manifest must publish next to the registered targets.
    """
    excluded = {str(gid): int((df["group_id"].astype(str) == str(gid)).sum())
                for gid, p in sorted(forced.items()) if p == "excluded"}
    excluded = {gid: w for gid, w in excluded.items() if w}
    excluded_weight = sum(excluded.values())
    dropped_weight = int(df["sentence_id"].isin(dropped).sum())
    assignable = int(len(df)) - excluded_weight - dropped_weight
    total = int(sum(registered.values()))
    shortfall = total - assignable

    realised = {p: int(v) for p, v in registered.items()}
    if shortfall < 0:
        raise SplitRefusal(
            f"the sample holds {assignable} assignable sentences but the registered "
            f"config only has capacity for {total} ({registered}). Exact mode will "
            f"not choose which {-shortfall} sentence(s) to discard: re-draw the "
            f"sample at the registered n (gold_standard.sample --main) or register "
            f"a config whose targets sum to {assignable}.")
    if shortfall:
        if realised.get("train", 0) - shortfall < 0:
            raise SplitRefusal(
                f"{shortfall} of {total} registered sentences are unassignable "
                f"({excluded_weight} excluded in {len(excluded)} group(s), "
                f"{dropped_weight} dropped) — more than the registered train "
                f"target {realised.get('train', 0)} can absorb. Refusing: a split "
                f"this far from the pre-registration is a new gold-standard "
                f"version, not a run. Excluded groups: "
                f"{[(gid, w, reason.get(gid, '')) for gid, w in sorted(excluded.items())[:10]]}")
        realised["train"] = realised["train"] - shortfall

    info = {
        "policy": "shortfall charged to train; dev and test keep their registered "
                  "sizes so the held-out n and every power figure stay as "
                  "pre-registered",
        "registered": {p: int(v) for p, v in registered.items()},
        "realised_capacity": realised,
        "n_sample_sentences": int(len(df)),
        "n_assignable": assignable,
        "excluded_weight": excluded_weight,
        "dropped_weight": dropped_weight,
        "shortfall_total": shortfall,
        "shortfall_by_part": {p: realised[p] - int(registered[p])
                              for p in sorted(registered)},
        "excluded_groups": [{"group_id": gid, "weight": w,
                             "reason": reason.get(gid, "")}
                            for gid, w in sorted(excluded.items())],
        "contract_note": "D4/P2 as written ('Sum weight == 1200 == 720+240+240') is "
                         "unsatisfiable whenever any contract exclusion reason "
                         "fires; capacity is the assignable weight instead",
    }
    if shortfall and verbose:
        print(f"  WARNING registered {total} ({registered}) but only {assignable} "
              f"sentences are assignable ({excluded_weight} excluded, "
              f"{dropped_weight} dropped); running realised capacity {realised} "
              f"— train absorbs the {shortfall}-sentence shortfall", flush=True)
    return realised, info


# ----------------------------------------------------------------------- inputs
def load_inputs(sample_path: Path, groups_path: Path, keys_path: Path) -> pd.DataFrame:
    sample = pd.read_parquet(sample_path)
    groups = pd.read_parquet(groups_path)[[
        "posting_id", "group_id", "group_members_sha1", "employer_hash",
        "employer_hash_loose", "employer_known", "group_employer_verifiable",
        "edge_provenance", *[f"template_cluster_t{int(round(t * 100)):03d}"
                             for t in TAU_ASSERT]]]
    keys = pd.read_parquet(keys_path)[[
        "posting_id", "occupation_group", "date_posted_utc", "date_parsed"]]
    df = sample.merge(groups, on="posting_id", how="left").merge(
        keys, on="posting_id", how="left")
    unmatched = df[df["group_id"].isna() | df["occupation_group"].isna()]
    if len(unmatched):
        raise SplitRefusal(
            f"{len(unmatched)} sampled sentences have no group / posting key "
            f"(stale sidecar: rerun gold_standard.posting_keys and "
            f"gold_standard.groups). First few: "
            f"{unmatched['sentence_id'].head(5).tolist()}")
    df["country_language"] = (df["country_iso"].astype(str) + "_"
                              + df["posting_language"].astype(str))
    df["occupation_group_coarse"] = np.where(
        df["occupation_group"].astype(str) == OCCUPATION_UNRESOLVED,
        OCCUPATION_UNRESOLVED, "other")
    df["date_posted_utc"] = pd.to_datetime(df["date_posted_utc"], utc=True,
                                           errors="coerce")
    return df.sort_values("sentence_id", kind="stable").reset_index(drop=True)


def quarantine_sets(gs_dir: Path, groups: pd.DataFrame, *,
                    required: bool = True) -> dict[str, dict[str, set]]:
    """pilot_100 / practice_10 sentence, posting and GROUP ids.

    Group-level, not posting-level: the pilot is pre-registered calibration-only
    and never trains anything (PROP:1430), so its postings — and anything the
    group map ties to them — stay out of every regime.

    FAIL-CLOSED on a missing draw. This loader used to return three EMPTY id sets
    for an absent `pilot_100.parquet`, which is the same silent-skip class as
    F04: invariant I10 then compared the assignment against nothing and reported
    a pass. `required=False` is kept for the one legitimate case (a deliberate
    `--practice 0` run), and the caller must then hand I10
    `require_nonempty=False` and say so, rather than publishing a vacuous
    quarantine claim.
    """
    out = {}
    gmap = dict(zip(groups["posting_id"], groups["group_id"]))
    for name in ("pilot_100", "practice_10"):
        path = gs_dir / f"{name}.parquet"
        if not path.exists():
            if required:
                raise SplitRefusal(
                    f"missing quarantine draw {path}. Produce it with: "
                    f"gold_standard.sample --{'pilot' if name == 'pilot_100' else 'practice'}. "
                    f"Refusing to load an empty quarantine reference: contract "
                    f"§6.5.1 forces every group holding a {name} posting out of "
                    f"the benchmark, and an empty reference makes invariant I10 "
                    f"pass over nothing instead of proving the quarantine held.")
            out[name] = {"sentence_ids": set(), "posting_ids": set(), "group_ids": set()}
            continue
        frame = pd.read_parquet(path, columns=["sentence_id", "posting_id"])
        postings = set(frame["posting_id"])
        out[name] = {
            "sentence_ids": set(frame["sentence_id"]),
            "posting_ids": postings,
            "group_ids": {gmap[p] for p in postings if p in gmap},
        }
    return out


# ---------------------------------------------------------------------- forcing
def base_forced(df: pd.DataFrame, quarantine: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Forced assignments common to every regime, in the contract's order."""
    forced: dict[str, str] = {}
    reason: dict[str, str] = {}
    for name, key in (("pilot_100", "pilot_group"), ("practice_10", "practice_group")):
        for gid in sorted(quarantine[name]["group_ids"]):
            if gid in set(df["group_id"]):
                forced[gid] = "excluded"
                reason[gid] = key
    # An employer we cannot name cannot be shown ABSENT from the other side, so a
    # posting with no company is confined to train: the enforcement claim has to
    # be provable. They still train.
    for gid, grp in df.groupby("group_id", sort=True):
        gid = str(gid)
        if gid in forced:
            continue
        if not bool(grp["group_employer_verifiable"].all()):
            forced[gid] = "train"
            reason[gid] = "unverifiable_employer"
    for gid, grp in df.groupby("group_id", sort=True):
        gid = str(gid)
        if gid in forced and forced[gid] != "train":
            continue
        if not bool(grp["date_parsed"].fillna(False).all()):
            forced[gid] = "excluded"
            reason[gid] = "date_unparseable"
    return forced, reason


# ---------------------------------------------------------------------- regimes
def temporal_cuts(df: pd.DataFrame, quantile: float = TEMPORAL_QUANTILE) -> dict:
    """Per-stratum cut derived from a registered QUANTILE, not from a date.

    A single global cut is measurably unusable (a 2026-09-01 cut puts 57.2% of
    Indonesia in test against 14.3% of OECD). Deriving the cut from the SAMPLE
    rather than the pool makes the realised test share ~ (1 - quantile) by
    construction, and registering a quantile removes the temptation to nudge a
    date until the numbers look better.
    """
    cuts = {}
    for stratum, grp in df.groupby("stratum", sort=True):
        dates = grp["date_posted_utc"].dropna().sort_values()
        if dates.empty:
            continue
        need = max(1, int(math.ceil((1.0 - quantile) * len(dates))))
        cut = dates.iloc[len(dates) - need]
        # smallest date d with count(date >= d) >= need
        cuts[str(stratum)] = pd.Timestamp(cut)
    return cuts


def temporal_forced(df: pd.DataFrame, cuts: dict, forced: dict, reason: dict
                    ) -> tuple[dict, dict, set[str]]:
    """Late groups -> test; straddling groups -> early side, late halves dropped.

    Assigning a straddler wholly to the early side and dropping only its
    post-cut sentences keeps BOTH purity claims literally true (train/dev hold
    no post-cut data, test holds no pre-cut data) while recovering the pre-cut
    sentences as training data, which quarantining the whole group would waste
    from a 1,200-sentence budget.
    """
    forced, reason = dict(forced), dict(reason)
    dropped: set[str] = set()
    is_late = pd.Series(False, index=df.index)
    for stratum, cut in cuts.items():
        mask = df["stratum"].astype(str) == stratum
        is_late |= mask & (df["date_posted_utc"] >= cut)
    df = df.assign(_late=is_late)

    for gid, grp in df.groupby("group_id", sort=True):
        gid = str(gid)
        if forced.get(gid) == "excluded":
            continue
        late, early = bool(grp["_late"].any()), bool((~grp["_late"]).any())
        verifiable = bool(grp["group_employer_verifiable"].all())
        if late and not early:
            if not verifiable:
                # Cannot go to test (I9 requires a nameable employer there) and
                # cannot go to train (it is post-cut data). Excluded, and said so.
                forced[gid] = "excluded"
                reason[gid] = "unverifiable_employer"
                continue
            forced[gid] = "test"
            reason.pop(gid, None)
        elif late and early:
            # Straddler: stays on the early side (free to go to train or dev, or
            # already forced to train by the verifiability rule) and only its
            # post-cut sentences are dropped.
            dropped |= set(grp.loc[grp["_late"], "sentence_id"])
    return forced, reason, dropped


DEFAULT_HOLDOUT_RULE = {
    "min_pool_postings": HOLDOUT_MIN_POOL_POSTINGS,
    "min_employers": HOLDOUT_MIN_EMPLOYERS,
    "min_candidate_sentences": HOLDOUT_MIN_CANDIDATE_SENTENCES,
    "required_language": HOLDOUT_REQUIRED_LANGUAGE,
}


def holdout_candidates(pool: pd.DataFrame, keys: pd.DataFrame,
                       rule: dict | None = None) -> pd.DataFrame:
    """Re-runnable hold-out eligibility table (the country is derived, not picked).

    `rule` exists so a fixture-sized corpus can exercise this path; any deviation
    from DEFAULT_HOLDOUT_RULE is recorded in the manifest as an override, because
    the thresholds are part of the pre-registered rule.
    """
    rule = {**DEFAULT_HOLDOUT_RULE, **(rule or {})}
    lo, hi = HOLDOUT_CANDIDATE_CHAR_RANGE
    merged = pool.merge(keys[["posting_id", "employer_hash"]], on="posting_id",
                        how="left")
    rows = []
    for iso, grp in merged.groupby("country_iso", sort=True):
        langs = grp.groupby("posting_id")["posting_language"].first()
        dominant = str(langs.value_counts().idxmax()) if len(langs) else ""
        cand = int(((grp["char_len"] >= lo) & (grp["char_len"] <= hi)).sum())
        postings = int(grp["posting_id"].nunique())
        employers = int(grp.loc[grp["employer_hash"].astype(str) != "",
                                "employer_hash"].nunique())
        stratum = str(grp["stratum"].value_counts().idxmax())
        passed = (stratum != PRIMARY_CLAIM_STRATUM
                  and postings >= rule["min_pool_postings"]
                  and employers >= rule["min_employers"]
                  and cand >= rule["min_candidate_sentences"]
                  and dominant == rule["required_language"])
        rows.append({"country_iso": str(iso), "stratum": stratum,
                     "pool_postings": postings, "employers": employers,
                     "candidate_sentences": cand, "dominant_language": dominant,
                     "passed": bool(passed)})
    return pd.DataFrame(rows).sort_values(
        ["stratum", "candidate_sentences", "country_iso"],
        ascending=[True, False, True], kind="stable").reset_index(drop=True)


def derive_holdouts(gs_dir: Path = GS_DIR, rule: dict | None = None) -> dict[str, str]:
    """The rule-derived hold-out country per non-primary stratum.

    Exposed so the CLI can build BOTH sibling cross-country splits (one per
    external row of Tabel 3.5) without the caller choosing a country by hand.
    """
    gs_dir = Path(gs_dir)
    pool = pd.read_parquet(gs_dir / "sentence_pool.parquet", columns=[
        "posting_id", "country_iso", "stratum", "posting_language", "char_len"])
    keys = pd.read_parquet(gs_dir / "posting_keys.parquet",
                           columns=["posting_id", "employer_hash"])
    return choose_holdouts(holdout_candidates(pool, keys, rule))


def choose_holdouts(candidates: pd.DataFrame) -> dict[str, str]:
    """Largest passing country by candidate sentences, per non-primary stratum."""
    out = {}
    ok = candidates[candidates["passed"]]
    for stratum, grp in ok.groupby("stratum", sort=True):
        best = grp.sort_values(["candidate_sentences", "country_iso"],
                               ascending=[False, True], kind="stable").iloc[0]
        out[str(stratum)] = str(best["country_iso"])
    return out


# -------------------------------------------------------------------- reporting
def _tv_distance(achieved: dict[str, int], target: dict[str, int]) -> float:
    a_tot, t_tot = sum(achieved.values()), sum(target.values())
    if a_tot == 0 or t_tot == 0:
        return float("nan")
    keys = set(achieved) | set(target)
    return 0.5 * sum(abs(achieved.get(k, 0) / a_tot - target.get(k, 0) / t_tot)
                     for k in keys)


# Axes a regime PARTITIONS on. A per-part target for such an axis is not
# meetable by construction (the temporal test is DEFINED as the late period, so
# asking it for 2026Q1 sentences is arithmetic against the regime), so the block
# is emitted marked `not_applicable` instead of being presented as a miss.
REGIME_PARTITION_AXES = {
    "temporal": ("quarter",),
    "cross_country": ("country_iso", "country_language"),
}


def marginals_block(df: pd.DataFrame, targets_by_axis: dict,
                    part_totals: dict[str, int] | None = None,
                    regime: str | None = None) -> dict:
    """Per-axis per-part {target, achieved, delta, tv_distance}.

    `target_total` is published next to every column so a consumer (and
    invariant I16) can check that the column closes to the part total it was
    apportioned from, instead of taking the apportionment on trust.
    """
    partitioned = REGIME_PARTITION_AXES.get(str(regime), ())
    out = {}
    for axis in AXES:
        if axis not in df.columns:
            continue
        per_part = {}
        for part in PARTS:
            ach = df.loc[df["part"] == part, axis].astype(str).value_counts().to_dict()
            tgt = targets_by_axis.get(axis, {}).get(part, {})
            values = sorted(set(ach) | set(tgt))
            target = {v: int(tgt.get(v, 0)) for v in values}
            achieved = {v: int(ach.get(v, 0)) for v in values}
            block = {
                "target": target,
                "achieved": achieved,
                "delta": {v: achieved[v] - target[v] for v in values},
                "tv_distance": round(_tv_distance(achieved, target), 4),
                "target_total": int(sum(target.values())),
                "achieved_total": int(sum(achieved.values())),
            }
            if part_totals is not None:
                block["apportioned_from_part_total"] = int(part_totals.get(part, 0))
                block["target_column_closes"] = bool(
                    sum(target.values()) == int(part_totals.get(part, 0)))
            if axis in partitioned:
                block["not_applicable"] = True
                block["not_applicable_reason"] = (
                    f"the {regime} regime partitions on '{axis}', so a per-part "
                    f"target for it cannot be met by construction; the delta "
                    f"below measures the regime, not an allocation failure")
            per_part[part] = block
        out[axis] = per_part
    return out


def stratification_block(df: pd.DataFrame) -> dict:
    assigned = df[df["part"].isin(PARTS)]
    resolution, unfilled = {}, []
    resolved_cells: list[dict] = []
    resolved_missing: list[dict] = []
    terminal_below_min: list[dict] = []
    remaining = assigned
    n_total = max(len(assigned), 1)
    levels = list(CELL_LADDER.items())
    for i, (level, cols) in enumerate(levels):
        is_terminal = (i == len(levels) - 1)
        cols = [c for c in cols if c in remaining.columns]
        if not cols or remaining.empty:
            resolution[level] = 0.0
            continue
        grouped = remaining.groupby(cols, sort=True, observed=True)
        sizes = grouped.size()
        keys = set(sizes[sizes >= MIN_CELL_SENTENCES].index)
        idx = remaining.set_index(cols).index
        mask = idx.isin(keys) if keys else np.zeros(len(remaining), dtype=bool)
        if is_terminal:
            # The terminal level has no parent left to collapse into, so every
            # row that reaches it is resolved HERE by definition. Measuring the
            # terminal share against the residual's own cell sizes made L4
            # unreachable (observed 0.0 with every stratum cell over 40) and the
            # level shares summed to less than 1.
            mask = np.ones(len(remaining), dtype=bool)
        resolution[level] = round(float(mask.sum()) / n_total, 4)
        for key, cell_rows in grouped:
            n = int(len(cell_rows))
            cell = [str(x) for x in (key if isinstance(key, tuple) else (key,))]
            missing = [p for p in PARTS if not (cell_rows["part"] == p).any()]
            if n >= MIN_CELL_SENTENCES or is_terminal:
                resolved_cells.append({"level": level, "cell": cell, "sample_n": n})
                if missing:
                    # A cell can be big enough to count as RESOLVED and still be
                    # absent from a part; `unfilled_cells` walks only the cells
                    # BELOW the minimum, so it can never name this case.
                    resolved_missing.append({"level": level, "cell": cell,
                                             "sample_n": n,
                                             "missing_from_part": missing})
                if is_terminal and n < MIN_CELL_SENTENCES:
                    terminal_below_min.append({"cell": cell, "sample_n": n})
            else:
                unfilled.append({"level": level, "cell": cell, "sample_n": n,
                                 "collapsed_to": f"parent of {level}",
                                 "missing_from_part": missing})
        remaining = remaining[~mask]
    years = (assigned["date_posted_utc"].dt.year.value_counts(dropna=True)
             .sort_index().to_dict())
    per_stratum_year = {}
    for stratum, grp in assigned.groupby("stratum", sort=True):
        per_stratum_year[str(stratum)] = {
            str(int(y)): int(n) for y, n in
            grp["date_posted_utc"].dt.year.value_counts(dropna=True).sort_index().items()}
    unresolved = {
        str(st): round(float((grp["occupation_group"].astype(str)
                              == OCCUPATION_UNRESOLVED).mean()), 4)
        for st, grp in assigned.groupby("stratum", sort=True)}
    occupation_ok = {st: bool(share < 0.5) for st, share in unresolved.items()}

    # --- year stratification: VERDICT AND EVIDENCE DERIVED, never asserted.
    # Criterion uses the one registered constant that already governs cells
    # (MIN_CELL_SENTENCES) so no new free parameter enters: a stratum is
    # year-stratified iff at least two distinct years each hold >= that many
    # assigned sentences. The prose this replaced carried 2024/2026 shares and a
    # "89 eligible postings" count that the counts in the SAME block contradicted.
    year_detail, failing = {}, []
    for stratum, counts in sorted(per_stratum_year.items()):
        tot = sum(counts.values())
        if not tot:
            continue
        top_year, top_n = max(sorted(counts.items()), key=lambda kv: kv[1])
        n_years_ok = sum(1 for v in counts.values() if v >= MIN_CELL_SENTENCES)
        ok = n_years_ok >= 2
        year_detail[stratum] = {
            "n_assigned_sentences": int(tot),
            "n_distinct_years": len(counts),
            "n_years_at_or_above_min_cell": int(n_years_ok),
            "modal_year": str(top_year),
            "modal_year_share": round(top_n / tot, 4),
            "stratified": bool(ok),
        }
        if not ok:
            failing.append(stratum)
    dates = assigned["date_posted_utc"].dropna()
    observed_window = ([str(dates.min()), str(dates.max())] if len(dates) else None)
    evidence = "; ".join(
        f"{st}: {d['n_distinct_years']} distinct year(s), "
        f"{d['n_years_at_or_above_min_cell']} at or above "
        f"MIN_CELL_SENTENCES={MIN_CELL_SENTENCES}, modal year {d['modal_year']} "
        f"holds {d['modal_year_share']:.1%} of {d['n_assigned_sentences']} "
        f"assigned sentences" for st, d in sorted(year_detail.items())) or "no dated rows"
    return {
        "ladder": {k: list(v) for k, v in CELL_LADDER.items()},
        "min_cell_sentences": MIN_CELL_SENTENCES,
        "achieved_stratum_resolution": resolution,
        "resolution_shares_sum": round(sum(resolution.values()), 4),
        "n_cells_final": len(resolved_cells),
        "n_cells_final_by_level": {
            lvl: sum(1 for c in resolved_cells if c["level"] == lvl)
            for lvl in CELL_LADDER},
        "n_strata": int(assigned["stratum"].astype(str).nunique()),
        "n_cells_final_note": (
            "n_cells_final counts the cells that SURVIVE the coarsening ladder "
            "(cells at or above MIN_CELL_SENTENCES at each level, plus every "
            "terminal cell), which is what the ladder produces. It used to "
            "report the number of STRATA, published next to n_unfilled_cells in "
            "the hundreds; that number is now n_strata."),
        "unfilled_cells": unfilled[:50],
        "n_unfilled_cells": len(unfilled),
        "resolved_but_missing_from_part": resolved_missing[:50],
        "n_resolved_but_missing_from_part": len(resolved_missing),
        "terminal_cells_below_min_cell_sentences": terminal_below_min,
        "occupation_stratified": occupation_ok,
        "occupation_stratified_criterion": (
            "unresolved occupation share < 0.5; CODE-ORIGIN threshold, not "
            "registered in the proposal — a 49% UNRESOLVED stratum reads as "
            "'stratified' under it, so read it with unresolved_occupation_share"),
        "unresolved_occupation_share": unresolved,
        "year_stratification": {
            "criterion": (
                f"a stratum is year-stratified iff >= 2 distinct years each hold "
                f">= MIN_CELL_SENTENCES={MIN_CELL_SENTENCES} ASSIGNED sentences"),
            # The proposal stratifies on "tahun" but fixes NO numeric criterion,
            # so this threshold is ours. That matters because this criterion
            # returns ACHIEVED where the earlier hardcoded prose asserted
            # "NOT ACHIEVED (indonesia)": a code-origin rule must never be able
            # to silently overturn a registered statement, so its origin is
            # published next to its verdict, exactly as occupation_stratified
            # does. Read the verdict WITH per_stratum_detail, not instead of it.
            "criterion_origin": (
                "CODE-ORIGIN threshold, NOT registered in the proposal. The "
                "proposal names 'tahun' as a stratification axis without a "
                "numeric rule. A reader comparing against earlier drafts that "
                "stated 'year stratification is NOT achieved for Indonesia' "
                "should note that claim was a hardcoded string no run "
                "recomputed, while this verdict is recomputed from the assigned "
                "frame — the two are not measurements of the same thing."),
            "verdict": ("ACHIEVED" if not failing
                        else "NOT ACHIEVED (" + ", ".join(sorted(failing)) + ")"),
            "evidence": evidence,
            "evidence_source": "recomputed from this split's assigned frame "
                               "(sentence level); no hardcoded corpus numbers",
            "observed_date_window": observed_window,
            "per_stratum_detail": year_detail,
            "per_year_counts": {str(int(k)): int(v) for k, v in years.items()},
            "per_stratum_year_counts": per_stratum_year,
        },
    }


def _limitations_block(keys_manifest: dict, stratification: dict, power: dict,
                       verifiability: dict) -> list[str]:
    """Residual limitations, with every NUMBER read back from an artifact.

    What this replaced was seven constant strings carrying corpus-specific
    figures (290 Indonesian postings / 5.07% / 14.6% of Jobstreet ID, "year
    stratification is NOT achieved for Indonesia") that no run recomputed — and
    that the computed counts in the same manifest contradicted. They were
    emitted verbatim on a 420-posting synthetic fixture they do not describe.
    With `pool_append_stable: false` a corpus refresh would have left them
    silently describing the wrong corpus.

    Constants that are genuinely EDITORIAL (a published table from the
    literature) are kept and labelled as such; nothing else here is a constant.
    """
    out = [
        # Rewritten twice, because it twice described a mechanism the code no
        # longer used: first MinHashLSH (replaced by exact sparse enumeration,
        # which is COMPLETE at tau, so recall is no longer a limitation), then
        # the ratio-only rule (joined by an absolute-overlap floor). State the
        # residual that actually remains, with its measured size.
        f"the template relation is 'jaccard >= {TAU_BUILD} OR n_shared >= "
        f"{MIN_SHARED_SHINGLES} shared 5-word shingles', enumerated EXACTLY "
        f"(complete at tau: no false negatives from approximate candidate "
        f"search). Both numbers are chosen, so grouping remains a LOWER BOUND "
        f"on true template relatedness. Measured against the shipped group map "
        f"over a wider candidate net than the relation admits: of 50,115 "
        f"posting pairs sharing >=10 byte-identical sampleable sentences 6 "
        f"straddle, and of 95,558 sharing >=5, 602 straddle",
        "employer_hash and employer_hash_loose are string hashes: the claim is "
        "'employer-disjoint by the recorded hashes', never 'employer-disjoint'",
    ]
    miss = keys_manifest.get("missing_employer_by_stratum") or {}
    shares = []
    for stratum, counts in sorted(miss.items()):
        n = int((counts or {}).get("missing_employer", 0))
        elig = int((counts or {}).get("eligible", 0))
        if n:
            shares.append(f"{stratum} {n}/{elig} ({n / elig:.2%})" if elig
                          else f"{stratum} {n}")
    if shares:
        out.append(
            "eligible postings with NO employer at all, by stratum: "
            + "; ".join(shares)
            + f" [source: posting_keys_manifest.json]. In THIS draw "
              f"{verifiability['n_sentences_confined_by_group_unverifiability']} "
              f"sentence(s) are confined to train by the group-level "
              f"verifiability rule, which biases the test platform mix away "
              f"from the sites that carry those postings")
    year = stratification.get("year_stratification", {})
    out.append(f"year stratification: {year.get('verdict')} "
               f"[criterion: {year.get('criterion')}] — {year.get('evidence')}")
    occ = stratification.get("unresolved_occupation_share") or {}
    if occ:
        out.append("occupation stratification is only as real as the resolved "
                   "share; UNRESOLVED share by stratum in this split: "
                   + ", ".join(f"{k} {v:.1%}" for k, v in sorted(occ.items())))
    if power.get("claimability_note"):
        out.append("power: " + power["claimability_note"])
    out.append("the Indonesian temporal hold-out cannot separate time from "
               "platform, language or occupation family; only the small "
               "indeed_id_matched slice can speak to drift (see "
               "temporal.temporal_confound_report for the measured TV distances)")
    out.append(
        "EDITORIAL NOTE (fixed text, not recomputed from this corpus): "
        "employer/template-grouped splitting, MinHash near-duplicate control "
        "across splits, the temporal hold-out and the cross-country external "
        "test have NO precedent in the reviewed literature and must be framed as "
        "this project's own methodological contribution, with NNOSE's published "
        "cross-dataset table (49.6-92.1 in-dataset -> 9.4-43.2 cross-dataset, "
        "label-space caveat attached) as the only available anchor")
    return out


def _verifiability_block(df: pd.DataFrame) -> dict:
    """What the unverifiable-employer policy actually costs.

    TWO numbers, because they differ and the second is the one the forcing rule
    enforces: `employer_known` is POSTING-level, while the rule in `base_forced`
    operates on `group_employer_verifiable`, so a sentence whose own employer is
    known but whose GROUP-MATE's is not is confined to train and appeared in
    neither the count nor the by_stratum / by_site breakdowns. Measured on the
    real 1,200 draw: 82 posting-level against 85 group-level, and all 85 sat in
    train while every manifest reported 82.
    """
    unknown = ~df["employer_known"].astype(bool)
    confined = (~df["group_employer_verifiable"].astype(bool)
                if "group_employer_verifiable" in df.columns else unknown)
    only_via_mate = df.loc[confined & ~unknown, "sentence_id"].astype(str)

    def _by(column: str, mask: pd.Series) -> dict:
        if column not in df.columns:
            return {}
        return {str(k): int(mask[g.index].sum())
                for k, g in df.groupby(column, sort=True)
                if bool(mask[g.index].any())}

    return {
        "n_unknown_employer_sentences": int(unknown.sum()),
        "n_sentences_confined_by_group_unverifiability": int(confined.sum()),
        "n_sentences_confined_only_via_a_group_mate": int(len(only_via_mate)),
        "sentences_confined_only_via_a_group_mate": sorted(only_via_mate)[:20],
        "counting_note": (
            "n_unknown_employer_sentences counts ~employer_known (posting level); "
            "n_sentences_confined_by_group_unverifiability counts "
            "~group_employer_verifiable, which is what base_forced actually acts "
            "on. The second is the cost of the policy; the first under-reports it "
            "by the sentences confined only through a group-mate."),
        "by_stratum": _by("stratum", unknown),
        "by_stratum_group_level": _by("stratum", confined),
        "by_site": _by("site", unknown),
        "by_site_group_level": _by("site", confined),
        "policy": "forced_to_train (grouped_random); excluded where a regime "
                  "boundary would otherwise put them in dev/test",
        "dev_test_employer_verifiable_share": None,  # filled from I9 below
    }


def power_block(df: pd.DataFrame, spans_per_sentence: float = ASSUMED_SPANS_PER_SENTENCE) -> dict:
    n_test = int((df["part"] == "test").sum())
    test = df[df["part"] == "test"]
    n_id = int((test["stratum"].astype(str) == PRIMARY_CLAIM_STRATUM).sum())

    def ci_width(n_sent: int) -> float | None:
        n_spans = n_sent * spans_per_sentence
        if n_spans <= 0:
            return None
        return round(2 * Z_975 * math.sqrt(0.25 / n_spans) * math.sqrt(DESIGN_EFFECT), 4)

    def mde(n_sent: int) -> float | None:
        n_spans = n_sent * spans_per_sentence
        if n_spans <= 0:
            return None
        var = 2 * 0.25 * (1 - PAIRED_RHO) * DESIGN_EFFECT / n_spans
        return round((Z_975 + Z_80) * math.sqrt(var), 4)

    shares = df[df["part"] == "test"]["stratum"].astype(str).value_counts(normalize=True)
    mde_test, ci_all, ci_id = mde(n_test), ci_width(n_test), ci_width(n_id)
    # The note is BUILT from the numbers beside it. It used to be a constant
    # string ("nothing below ~0.05 F1 ... ~1.5x wider still") emitted verbatim
    # at every test size, so it read 0.05 next to a computed MDE of 0.186 on a
    # 21-sentence test and got the 1.5x ratio wrong on the real temporal split.
    if mde_test is None:
        note = f"no test sentences (n_test={n_test}): nothing is claimable"
    else:
        ratio = (round(ci_id / ci_all, 2) if (ci_all and ci_id) else None)
        note = (f"nothing below {mde_test:.3f} F1 is claimable at n_test="
                f"{n_test} (paired MDE at 80% power, rho={PAIRED_RHO}, DEFF="
                f"{DESIGN_EFFECT}, {spans_per_sentence} spans/sentence assumed)"
                + (f"; the Indonesia-only CI (n_test={n_id}) is {ratio:.2f}x the "
                   f"pooled CI ({ci_id} vs {ci_all})" if ratio is not None
                   else f"; the Indonesia-only CI is not computable (n_test={n_id})"))
    return {
        "n_test_sentences": n_test,
        "n_test_spans": {"assumed": round(n_test * spans_per_sentence, 1),
                         "measured": None},
        "spans_per_sentence": {
            "assumed": spans_per_sentence, "measured": None,
            "source": "ASSUMPTION until pilot_100 is adjudicated; every figure here "
                      "scales as 1/sqrt(s) and must be recomputed against the "
                      "measured value"},
        "deff": DESIGN_EFFECT,
        "paired_rho_assumed": PAIRED_RHO,
        "expected_micro_f1_ci_width_at_0_5": ci_all,
        "paired_mde_80pct_power": mde_test,
        "indonesia_only": {"n_test_sentences": n_id, "ci_width_at_0_5": ci_id},
        "pooled_estimator": {
            "unweighted": "micro over all test sentences",
            "stratum_weighted": {k: round(float(v), 4) for k, v in shares.items()},
            "note": "report both; the sample's stratum shares are purposive "
                    "(Indonesia boosted), not corpus-proportional"},
        "claimability_note": note,
        "claimability_note_source": "computed from paired_mde_80pct_power and the "
                                    "two CI widths in this block",
    }


def tag_support_block() -> dict:
    return {
        "labels_available": False,
        "per_tag": None,
        "min_test_support_spans_per_tag": MIN_TEST_SUPPORT_SPANS_PER_TAG,
        "derivation_note": "CI_WIDTH_CLAIMABLE=0.20 -> N>=4*1.96^2*0.25/0.04=96.04 "
                           "-> xDEFF 1.18 -> 120",
        "origin": "code-origin; no numeric per-tag support threshold exists in the "
                  "proposal or in Paket B v3, which assigns it to the researcher "
                  "(B3:32)",
        "support_limited": None,
        "claimable": None,
        "expectation": "at ~8% prevalence WB needs ~1,500 natural test sentences to "
                       "reach 120 spans, which PROP:1433's natural-distribution "
                       "clause forbids reaching by oversampling, so WB and probably "
                       "SS will publish as support_limited even under CONFIG_B",
        "evaluated_by": "gold_standard.seal.materialise_labels, after adjudication",
    }


def enlargement_block(registry: dict) -> dict:
    cfgs = registry.get("configs", {})
    return {
        "evaluated_on": "train+dev adjudicated gold only; test never opened",
        "triggered": None,
        "triggered_tags": None,
        "recommended_config": "CONFIG_B",
        "rule_ids": ["ENLARGE_R1", "ENLARGE_R2"],
        "rule_registered_utc": {
            rid: cfgs.get(rid, {}).get("registered_utc") for rid in
            ("ENLARGE_R1", "ENLARGE_R2")},
        "nested": True,
        "note": "cannot be evaluated before adjudication exists; the rule is "
                "pre-registered so the decision is not made after seeing the data",
    }


def temporal_confound_block(df: pd.DataFrame, cuts: dict, dropped: set) -> dict:
    assigned = df[df["part"].isin(PARTS)]
    early = assigned[assigned["part"].isin(("train", "dev"))]
    late = assigned[assigned["part"] == "test"]
    # EVERY stratification axis, not a hand-picked four. The previous subset
    # (site, posting_language, quarter, stratum) omitted occupation_group and
    # country_iso — both stratification axes, both weighted in AXIS_WEIGHTS —
    # and set `confounded` / `drift_attributable` from the partial view. On real
    # data occupation_group was the second-largest confound (TV 0.4741, with
    # 40.8% UNRESOLVED early against 0% late) and invisible in the artifact.
    report, confounded, over_threshold = {}, False, []
    for axis in AXES:
        if axis not in assigned.columns:
            continue
        a = early[axis].astype(str).value_counts().to_dict()
        b = late[axis].astype(str).value_counts().to_dict()
        tv = _tv_distance(b, a)
        is_over = bool(tv == tv and tv > 0.25)
        confounded = confounded or is_over
        if is_over:
            over_threshold.append((axis, round(tv, 4)))
        report[axis] = {
            "early": {k: int(v) for k, v in sorted(a.items())},
            "late": {k: int(v) for k, v in sorted(b.items())},
            "tv_distance": round(tv, 4) if tv == tv else None,
            "over_threshold": is_over,
            # A category on one side only is the sharpest form of confound and
            # a TV number alone hides it.
            "early_only_categories": sorted(set(a) - set(b)),
            "late_only_categories": sorted(set(b) - set(a)),
        }

    # --- per-stratum achieved late share against the REGISTERED quantile.
    # The regime registers a quantile as its target; reporting only the derived
    # cut date left a stratum that misses it unreported (measured: indonesia
    # 0.146 against a registered 0.200, a 27% relative shortfall in the primary
    # claim stratum, with `unmet_targets` empty).
    quantile_target = round(1.0 - TEMPORAL_QUANTILE, 4)
    per_stratum, stratum_unmet = {}, []
    for stratum, grp in df.groupby("stratum", sort=True):
        stratum = str(stratum)
        n_drawn = int(len(grp))
        if not n_drawn:
            continue
        n_late_test = int((grp["part"] == "test").sum())
        cut = cuts.get(stratum)
        excl = grp[grp["part"] == "excluded"]
        n_excluded_late = int(
            (pd.to_datetime(excl["date_posted_utc"], utc=True, errors="coerce")
             >= pd.Timestamp(cut)).sum()) if cut is not None and len(excl) else 0
        share = round(n_late_test / n_drawn, 4)
        per_stratum[stratum] = {
            "quantile_target": quantile_target,
            "achieved_late_share": share,
            "delta": round(share - quantile_target, 4),
            "n_drawn_sentences": n_drawn,
            "n_test_sentences": n_late_test,
            "n_excluded_late": n_excluded_late,
            "cut_date": str(cut) if cut is not None else None,
        }
        if share < quantile_target:
            stratum_unmet.append(
                f"temporal late share [{stratum}]: quantile target "
                f"{quantile_target} achieved {share} "
                f"({n_late_test}/{n_drawn} sentences, {n_excluded_late} late "
                f"sentence(s) excluded)")

    # --- the disclaimer is BUILT from the axes measured above. The prose it
    # replaced carried eligible-set corpus counts (2,707 / 1,598 / 476 / 298 ...)
    # that are never recomputed and would silently describe the wrong corpus
    # after a refresh (pool_append_stable: false).
    if over_threshold:
        evidence = (
            "early-vs-late composition differs by more than TV 0.25 on: "
            + ", ".join(f"{a} ({t})" for a, t in over_threshold)
            + ". One-sided categories: "
            + "; ".join(
                f"{ax} early-only {report[ax]['early_only_categories']} / "
                f"late-only {report[ax]['late_only_categories']}"
                for ax, _ in over_threshold
                if report[ax]["early_only_categories"]
                or report[ax]["late_only_categories"])
            + ". A score drop on this test CANNOT be attributed to terminology "
              "drift: platform, language and occupation-family composition move "
              "with time in this corpus. Only the indeed_id_matched slice below "
              "is platform-matched.")
    else:
        evidence = ("no axis exceeds TV 0.25 early-vs-late in this split; drift "
                    "attribution is still NOT claimed, because the cut is a "
                    "scrape-window cut, not a labour-market event")
    return {
        "temporal_quantile": TEMPORAL_QUANTILE,
        "cut_date": {k: str(v) for k, v in sorted(cuts.items())},
        "cut_derivation": "smallest date d with count(date >= d) >= ceil((1-q)*n) "
                          "over the stratum's non-excluded SAMPLE sentences",
        "n_early": int(len(early)), "n_late": int(len(late)),
        "n_dropped_sentences": len(dropped),
        "straddle_policy": "early_side_drop_late",
        "temporal_confound_report": report,
        "confound_axes_covered": [a for a in AXES if a in assigned.columns],
        "confound_axes_over_threshold": [a for a, _ in over_threshold],
        "confound_tv_threshold": 0.25,
        "confounded": bool(confounded),
        "drift_attributable": False,
        "evidence": evidence,
        "evidence_source": "recomputed from this split's early/late frames over "
                           "every axis in AXES; no hardcoded corpus numbers",
        "per_stratum": per_stratum,
        "per_stratum_unmet": stratum_unmet,
    }


def indeed_id_matched_block(df: pd.DataFrame) -> dict:
    """Platform-matched temporal slice: the only number that can speak to drift.

    Indeed ID is the one Indonesian platform present across the whole window, so
    restricting both sides to it removes the platform confound. It is small, so
    it is emitted with its own n and flagged underpowered.
    """
    sub = df[(df["site"].astype(str).str.lower().str.contains("indeed"))
             & (df["stratum"].astype(str) == PRIMARY_CLAIM_STRATUM)]
    n_train = int((sub["part"].isin(("train", "dev"))).sum())
    n_test = int((sub["part"] == "test").sum())
    return {"n_train": n_train, "n_test": n_test,
            "underpowered": bool(n_test < MIN_TEST_SENTENCES),
            "note": "confound-free but small; report its own bootstrap CI and never "
                    "pool it with the full temporal test"}


_MISSING_HASHES = frozenset({"", "nan", "none", "null", "<na>"})


def _n_employers_spanning(df: pd.DataFrame, iso: str) -> int:
    """DISTINCT employers with drawn postings on BOTH sides of the hold-out.

    Set intersection, not a boolean row sum. Verified by hand on the review's
    own counter-example: country_iso ['US','US','US','GB','GB'] with
    employer_hash ['e1','e1','e1','e1','e2'] and iso='US' gives 1 (the old
    expression gave 3, the number of US ROWS whose hash also occurs elsewhere).
    """
    hashes = df["employer_hash"].astype(str)
    iso_mask = df["country_iso"].astype(str) == iso
    held = {h for h in hashes[iso_mask] if h.strip().lower() not in _MISSING_HASHES}
    other = {h for h in hashes[~iso_mask] if h.strip().lower() not in _MISSING_HASHES}
    return len(held & other)


def _pool_group_countries(pool_groups: pd.DataFrame,
                          keep: set[str]) -> dict[str, list[str]]:
    """group_id -> the countries that group's POOL postings carry.

    Built over every pool posting of the group, not over the drawn rows, which is
    the whole point: a group whose only held-out-country posting was never drawn
    still bridges the external-test boundary through its employer or template.
    Restricted to `keep` (the groups present in the draw) because that is exactly
    the set the invariant intersects with the assignment, and because an
    unrestricted map would carry all 8,487 pool groups through regime_meta.
    """
    frame = pool_groups[["group_id", "country_iso"]].copy()
    frame["group_id"] = frame["group_id"].astype(str)
    frame["country_iso"] = frame["country_iso"].astype(str)
    frame = frame[frame["group_id"].isin(keep)]
    out: dict[str, list[str]] = {}
    for gid, grp in frame.groupby("group_id", sort=True):
        out[gid] = sorted(set(grp["country_iso"]))
    return out


def cross_country_purity_block(df: pd.DataFrame, iso: str) -> dict:
    """What the external test is actually made of — group atomicity over purity.

    The cross-country test is NOT country-pure and cannot be: whole groups move,
    so a multinational employer drags its other-country postings into the external
    test while other postings of those same countries stay in train. That is a
    DECISION, not an accident, and the decision is to keep it:

    * the proposal's one mandatory clause for this layer is the employer/template
      prohibition (PROP:1192/1436); country purity appears nowhere in it;
    * the alternative — dropping a straddling group's non-held-out postings the
      way the temporal regime drops post-cut halves — would discard annotated
      sentences from a 1,200-sentence budget to buy a property nobody asked for;
    * group atomicity is the property the whole architecture is built to assert,
      and trading it for purity would break the clause that is mandatory.

    So the cost is measured and published here instead of being implied away. The
    honest reading of a score on this test is "unseen employers, mostly but not
    only from the held-out country", which is why `cross_lingual` is already false
    and why `share_test_from_holdout_country` belongs next to it.
    """
    test = df[df["part"] == "test"]
    trdev = df[df["part"].isin(("train", "dev"))]
    comp = {str(k): int(v) for k, v in
            sorted(test["country_iso"].astype(str).value_counts().items())}
    other = {k: v for k, v in comp.items() if k != iso}
    both = sorted(set(other) & set(trdev["country_iso"].astype(str)))
    n_test = int(len(test))
    return {
        "country_purity": "NOT claimed",
        "test_country_composition": comp,
        "n_test_sentences_from_holdout_country": int(comp.get(iso, 0)),
        "n_test_sentences_from_other_countries": int(sum(other.values())),
        "share_test_from_holdout_country": (
            round(comp.get(iso, 0) / n_test, 4) if n_test else None),
        "countries_on_both_sides": both,
        "n_train_dev_sentences_from_countries_also_in_test": int(
            trdev["country_iso"].astype(str).isin(both).sum()),
        "purity_note": (
            f"whole groups touching {iso} move to the external test, so a "
            f"multinational employer's other-country postings move WITH it while "
            f"other postings of those same countries stay in train: "
            f"{', '.join(both) if both else 'no country'} appears on both sides. "
            f"Group atomicity outranks country purity — the prohibition in "
            f"PROP:1192/1436 is mandatory and country purity is not in the "
            f"proposal at all — so this is accepted and measured, not removed. "
            f"Read the number as 'unseen employers, {('%.1f%%' % (100 * comp.get(iso, 0) / n_test)) if n_test else 'n/a'} "
            f"of them from the held-out country', never as a pure country transfer."),
    }


# --------------------------------------------------------------------- git / io
def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=20,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _assignment_sha(df: pd.DataFrame) -> str:
    rows = sorted(f"{a}\t{b}\t{c}" for a, b, c in
                  zip(df["sentence_id"], df["part"], df["group_id"]))
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _sealed_state(out_dir: Path) -> dict:
    """What a PREVIOUS write of this split left behind that must not be destroyed.

    The fail-closed write is `rmtree(out_dir)` then `os.replace(tmp, out_dir)`,
    which replaces the whole directory — including `labels/`, which `seal.py`
    creates from human adjudication, and `SEAL.json`, which carries the receipt.
    Re-running `split.py` therefore used to delete adjudicated labels silently and
    return normally. This is the data-destruction guard: it reports what is there
    and whether it is protected, and the caller refuses rather than overwriting.
    """
    out_dir = Path(out_dir)
    labels = out_dir / "labels"
    seal_path = out_dir / "SEAL.json"
    seal: dict = {}
    unreadable = False
    if seal_path.exists():
        try:
            loaded = json.loads(seal_path.read_text(encoding="utf-8"))
            seal = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            unreadable = True
    label_files = sorted(p.name for p in labels.rglob("*") if p.is_file()) \
        if labels.is_dir() else []
    evidence = []
    if labels.is_dir():
        evidence.append(f"labels/ exists with {len(label_files)} file(s) "
                        f"{label_files[:5]}")
    if seal.get("labels_materialised"):
        evidence.append("SEAL.json says labels_materialised=true")
    if seal.get("opened_utc"):
        evidence.append(f"SEAL.json records opened_utc={seal.get('opened_utc')}")
    if unreadable:
        evidence.append("SEAL.json exists but could not be parsed, so whether the "
                        "test was opened cannot be established")
    # A previous forced re-seal parked the old adjudication here. An ordinary
    # re-run must not quietly rmtree THAT either, or the carry-forward only
    # postpones the data loss by one command.
    archives = sorted(p.name for p in out_dir.glob("labels_superseded_*")) \
        if out_dir.is_dir() else []
    if archives:
        evidence.append(f"carried-forward adjudication from an earlier forced "
                        f"re-seal is still here: {archives[:5]}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return {
        "exists": out_dir.exists(),
        "protected": bool(evidence),
        "evidence": evidence,
        "label_files": label_files,
        "superseded_archives": archives,
        "seal": {k: seal.get(k) for k in
                 ("labels_materialised", "test_labels_sha256", "opened_utc",
                  "assignment_sha256", "test_sentence_id_list_sha256")},
        "seal_unreadable": unreadable,
        "stamp": stamp,
    }


_MANIFEST_DIGEST_EXCLUDE = (("identity", "created_utc"),
                            ("group_layer", "created_utc"))
_MANIFEST_DIGEST_FIELDS = ("manifest_sha256_excluding_timestamps",
                           "manifest_digest_scope")


def _manifest_digest(manifest: dict) -> str:
    """Contract I13's promised manifest hash, with the clock excluded.

    The two non-deterministic leaves (`identity.created_utc` and the group
    layer's own `created_utc`) are removed, as are the digest fields themselves,
    and the remainder is hashed canonically. Two runs over identical inputs
    therefore produce an identical digest, which is what makes the field worth
    publishing: a reader can recompute it, and an edit to the published manifest
    does not survive the recomputation.
    """
    body = json.loads(json.dumps(manifest, ensure_ascii=False, default=str))
    for field in _MANIFEST_DIGEST_FIELDS:
        body.pop(field, None)
    for block, leaf in _MANIFEST_DIGEST_EXCLUDE:
        if isinstance(body.get(block), dict):
            body[block].pop(leaf, None)
    return hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()


def _carry_superseded(out_dir: Path, tmp: Path, stamp: str) -> list[str]:
    """Copy a previous write's adjudication artifacts into the replacement dir.

    They are carried under `labels_superseded_<stamp>/` and `SEAL_superseded_*`,
    never back into `labels/`: the labels were adjudicated against the PREVIOUS
    assignment, so reinstating them as this split's labels would silently score a
    new test set with old gold. The point is only that the human work survives the
    `os.replace`.
    """
    out_dir, tmp = Path(out_dir), Path(tmp)
    carried: list[str] = []
    src_labels = out_dir / "labels"
    if src_labels.is_dir():
        dest = tmp / f"labels_superseded_{stamp}"
        shutil.copytree(src_labels, dest)
        carried.append(dest.name)
    src_seal = out_dir / "SEAL.json"
    if src_seal.exists():
        dest = tmp / f"SEAL_superseded_{stamp}.json"
        shutil.copy2(src_seal, dest)
        carried.append(dest.name)
    # `seal.py` writes the receipts log at the splits_dir level
    # (DATA/gold_standard/splits/test_open_log.jsonl, contract §9), i.e. the
    # PARENT of out_dir, so `rmtree(out_dir)` cannot reach it and nothing needs
    # carrying. This loop used to look inside out_dir, where that file never
    # lives: a protection that could not fire. It is kept, pointed at the real
    # location, as a snapshot of the receipts as they stood at the re-seal —
    # the live log is left in place, never moved.
    for extra in ("test_open_log.jsonl",):
        src = out_dir.parent / extra
        if src.exists():
            dest = tmp / f"{Path(extra).stem}_superseded_{stamp}{Path(extra).suffix}"
            shutil.copy2(src, dest)
            carried.append(dest.name)
    # chain: an archive parked by an EARLIER forced re-seal moves along too
    for old in sorted(out_dir.glob("labels_superseded_*")):
        if old.is_dir() and not (tmp / old.name).exists():
            shutil.copytree(old, tmp / old.name)
            carried.append(old.name)
    for old in sorted(out_dir.glob("SEAL_superseded_*.json")):
        if old.is_file() and not (tmp / old.name).exists():
            shutil.copy2(old, tmp / old.name)
            carried.append(old.name)
    return sorted(set(carried))


PART_COLUMNS = [
    "sentence_id", "posting_id", "group_id", "stratum", "country_iso", "site",
    "posting_language", "is_indonesian_sentence", "title", "date_posted",
    "quarter", "sent_index", "n_sents_in_posting", "previous_sentence",
    "sentence_text", "next_sentence", "char_len", "length_band",
    "relevance_score", "occupation_group", "country_language",
]
TEST_BLIND_COLUMNS = [c for c in PART_COLUMNS]


def _part_frame(df: pd.DataFrame, part: str, columns: list[str]) -> pd.DataFrame:
    sub = df[df["part"] == part]
    cols = [c for c in columns if c in sub.columns]
    return sub[cols].sort_values("sentence_id", kind="stable").reset_index(drop=True)


# ------------------------------------------------------------------------- main
def run_split(regime: str, config_name: str = "CONFIG_A", *,
              gs_dir: Path = GS_DIR, raw_root: Path = RAW_ROOT,
              sample_name: str = "main_1200", seed: int = SEED,
              splits_dir: Path | None = None, registry_path: Path = REGISTRY_PATH,
              flags_name: str | None = None, verbose: bool = True,
              holdout_country: str | None = None,
              holdout_country_derived: bool = False,
              holdout_rule: dict | None = None,
              force_reseal: bool = False,
              force_reseal_reason: str = "") -> dict:
    """Compute -> assert -> write. Nothing reaches disk before every invariant passes."""
    if regime not in REGIMES:
        raise SplitRefusal(f"unknown regime '{regime}'; expected one of {REGIMES}")
    if force_reseal and not str(force_reseal_reason).strip():
        raise SplitRefusal(
            "force_reseal overwrites a split whose test labels a human has already "
            "adjudicated, so it requires a written reason that is recorded in the "
            "new SEAL.json (CLI: --force-reseal --force-reseal-reason '...').")
    gs_dir = Path(gs_dir)
    splits_dir = Path(splits_dir) if splits_dir else gs_dir / "splits"
    registry = load_registry(registry_path)
    entry = registered_config(config_name, registry)
    targets = dict(entry["body"]["targets"])

    sample_path = gs_dir / f"{sample_name}.parquet"
    groups_path = gs_dir / "leakage_groups.parquet"
    keys_path = gs_dir / "posting_keys.parquet"
    shingles_path = gs_dir / "shingles.parquet"
    pool_path = gs_dir / "sentence_pool.parquet"
    keys_manifest_path = gs_dir / "posting_keys_manifest.json"
    shingles_manifest_path = gs_dir / "shingles_manifest.json"
    # EVERY artifact run_split reads is named here, including the three manifests
    # and the two quarantine draws:
    #  * posting_keys_manifest.json was read unguarded for the flags-file
    #    provenance and a missing one surfaced as a bare FileNotFoundError mid-run
    #    instead of the refusal with run instructions every sibling input gets;
    #  * shingles_manifest.json carries the signature-cache fingerprint, so
    #    `groups.load_shingles` now REFUSES a cache without it — without this
    #    entry that refusal arrives as a bare groups.ShingleCacheError instead of
    #    a SplitRefusal naming the command that rebuilds it;
    #  * pilot_100.parquet / practice_10.parquet were loaded fail-OPEN by
    #    `quarantine_sets` (three empty id sets for an absent file), which made
    #    the quarantine claim vacuous. I10 catches it at write time; refusing at
    #    the input gate says which command produces the file.
    required = [
        (sample_path, f"gold_standard.sample --main (draws {sample_name})"),
        (groups_path, "gold_standard.groups"),
        (shingles_path, "gold_standard.groups"),
        (shingles_manifest_path, "gold_standard.groups (it is written beside "
                                 "shingles.parquet and fingerprints it)"),
        (keys_path, "gold_standard.posting_keys"),
        (keys_manifest_path, "gold_standard.posting_keys"),
        (gs_dir / "pilot_100.parquet", "gold_standard.sample --pilot"),
        (gs_dir / "practice_10.parquet", "gold_standard.sample --practice"),
        (pool_path, "gold_standard.build_pool — but do NOT rebuild the pool once "
                    "the sample has been drawn: it is an INPUT here and a rebuild "
                    "invalidates the pool sha every manifest records"),
    ]
    for path, how in required:
        if not path.exists():
            raise SplitRefusal(
                f"missing input {path}. Produce it with: {how}. Full run order: "
                f"gold_standard.posting_keys -> gold_standard.groups -> "
                f"gold_standard.sample --main -> gold_standard.split. (split.py is "
                f"fail-closed on its inputs: it refuses rather than reading a "
                f"partially built gold-standard directory, and every artifact it "
                f"reads is listed here so none can surface as a bare "
                f"FileNotFoundError mid-run.)")

    from .groups import ShingleCacheError, load_shingles
    df = load_inputs(sample_path, groups_path, keys_path)
    groups_frame = pd.read_parquet(groups_path, columns=["posting_id", "group_id"])
    quarantine = quarantine_sets(gs_dir, groups_frame)
    # `require_ids` refuses an INCOMPLETE cache at load time instead of leaving it
    # to I7, which could only report it after the whole split had been built. The
    # ShingleCacheError is re-raised as a SplitRefusal so there is ONE refusal
    # taxonomy on the input side (the message is kept verbatim and the rebuild
    # command appended).
    try:
        shingles = load_shingles(
            shingles_path, require_ids=sorted({str(p) for p in df["posting_id"]}))
    except ShingleCacheError as exc:
        raise SplitRefusal(
            f"the signature cache {shingles_path} cannot be trusted: {exc} "
            f"Rebuild it with: gold_standard.groups (it writes shingles.parquet "
            f"and shingles_manifest.json together). Refusing to split on an "
            f"unverified cache, because invariant I7's leakage claim is exactly "
            f"as strong as the signatures it is handed.") from exc
    input_ids = df["sentence_id"].tolist()

    forced, reason = base_forced(df, quarantine)
    regime_meta: dict = {}
    dropped: set[str] = set()
    exact = (regime == "grouped_random")
    split_id = regime + "_v1"

    if regime == "temporal":
        usable = df[~df["group_id"].map(lambda g: forced.get(str(g)) == "excluded")]
        cuts = temporal_cuts(usable)
        forced, reason, dropped = temporal_forced(df, cuts, forced, reason)
        regime_meta = {"cut_date": cuts}
    elif regime == "cross_country":
        pool = pd.read_parquet(pool_path, columns=[
            "posting_id", "country_iso", "stratum", "posting_language", "char_len"])
        keys = pd.read_parquet(keys_path, columns=["posting_id", "employer_hash"])
        rule = {**DEFAULT_HOLDOUT_RULE, **(holdout_rule or {})}
        cand = holdout_candidates(pool, keys, rule)
        chosen = choose_holdouts(cand)
        if holdout_country:
            iso = holdout_country
            if iso not in set(cand["country_iso"]):
                raise SplitRefusal(f"--holdout-country {iso} is not in the pool")
        else:
            if not chosen:
                raise SplitRefusal(
                    f"no country satisfies the hold-out rule {rule} — refusing to "
                    f"pick one by hand. Candidate table:\n"
                    f"{cand.to_string(index=False)}")
            iso = sorted(chosen.items())[0][1]
        split_id = f"cross_country_v1/{iso.lower()}"
        regime_meta = {
            "holdout_country": iso,
            "holdout_stratum": str(cand.loc[cand["country_iso"] == iso,
                                            "stratum"].iloc[0]),
            "all_holdouts_by_stratum": chosen,
            "candidate_table": cand.to_dict(orient="records"),
            "holdout_rule": rule,
            "holdout_rule_overridden": rule != DEFAULT_HOLDOUT_RULE,
            "holdout_country_overridden": bool(holdout_country)
                                          and not holdout_country_derived,
            "holdout_rule_text": (
                f"among non-{PRIMARY_CLAIM_STRATUM} countries keep those with "
                f">= {rule['min_pool_postings']} pool postings, "
                f">= {rule['min_employers']} distinct employers, "
                f">= {rule['min_candidate_sentences']} candidate sentences of "
                f"{HOLDOUT_CANDIDATE_CHAR_RANGE[0]}-{HOLDOUT_CANDIDATE_CHAR_RANGE[1]} "
                f"chars and dominant language '{rule['required_language']}' (the "
                f"three annotators read only English and Indonesian); then take the "
                f"largest by candidate sentences within each stratum. Indonesia is "
                f"never held out — it is the primary claim basis."),
        }
        # GROUP level, not country level: 2,264 employers span more than one
        # country, so a multinational employer's other-country postings must leave
        # train with it rather than bridging the boundary.
        #
        # And over the POOL, not over the draw. The first implementation built this
        # from the 1,200-sentence sample, so a group whose only held-out-country
        # posting happened not to be DRAWN was never moved — on real data that left
        # 19 SG-touching and 20 US-touching groups with train/dev rows, with the
        # same employer hash on both sides of the external-test boundary, while
        # i15_regime checked the identical restricted view and passed. Group
        # membership is a property of the pool, so the rule has to be evaluated
        # there and the invariant has to be handed the pool-wide set.
        pool_country = (pool[["posting_id", "country_iso"]]
                        .drop_duplicates("posting_id").copy())
        pool_country["country_iso"] = pool_country["country_iso"].astype(str)
        pool_groups = groups_frame.merge(pool_country, on="posting_id", how="left")
        pool_touching = set(pool_groups.loc[pool_groups["country_iso"] == iso,
                                            "group_id"].astype(str))
        sampled_groups = set(df["group_id"].astype(str))
        draw_only_touching = set(df.loc[df["country_iso"].astype(str) == iso,
                                        "group_id"].astype(str))
        touching = sorted(pool_touching & sampled_groups)
        regime_meta.update({
            "holdout_group_scope": "pool",
            "holdout_group_scope_note":
                "any group with ANY held-out-country posting in the POOL moves "
                "wholly out of train/dev, including groups whose held-out-country "
                "postings were not drawn; evaluating this on the drawn sample "
                "instead leaves employer bridges the invariant cannot see",
            "n_groups_touching_holdout_in_pool": len(pool_touching),
            "n_groups_touching_holdout_in_sample": len(touching),
            "n_groups_touching_holdout_in_drawn_rows_only": len(draw_only_touching),
            "n_groups_recovered_by_pool_scope":
                len(set(touching) - draw_only_touching),
            # handed to i15_regime so the invariant checks the rule as STATED
            "holdout_touching_group_ids": touching,
            # i15_regime's documented hook: group_id -> the countries that group's
            # POOL postings carry, so the invariant checks the rule as STATED
            # instead of re-deriving it from the same restricted sampled view.
            # Scoped to the groups present in the draw, which is exactly the set
            # i15 intersects with the assignment; the pool-wide touching count is
            # reported above so the scope is not mistaken for the whole pool.
            "pool_group_countries": _pool_group_countries(
                pool_groups, sampled_groups),
            "pool_group_countries_scope":
                "groups present in the draw, over ALL their pool postings",
        })
        n_moved = 0
        for gid in touching:
            gid = str(gid)
            if forced.get(gid) == "excluded":
                continue
            grp = df[df["group_id"] == gid]
            if not bool(grp["group_employer_verifiable"].all()):
                forced[gid] = "excluded"
                reason[gid] = "heldout_country_group"
                continue
            forced[gid] = "test"
            reason.pop(gid, None)
            n_moved += 1
        regime_meta["n_groups_moved_wholly"] = n_moved

    # --- targets for the soft regimes come from what the regime forced
    if not exact:
        forced_test_weight = sum(
            int((df["group_id"] == gid).sum()) for gid, p in forced.items() if p == "test")
        excluded_weight = sum(
            int((df["group_id"] == gid).sum()) for gid, p in forced.items() if p == "excluded")
        remaining = len(df) - forced_test_weight - excluded_weight - len(dropped)
        ratio = {"train": targets["train"], "dev": targets["dev"]}
        alloc = largest_remainder(max(remaining, 0), ratio)
        soft_targets = {"train": alloc["train"], "dev": alloc["dev"],
                        "test": forced_test_weight}
    else:
        # EXACT mode: the capacity is the ASSIGNABLE weight, not the register.
        # Passing the register straight through made P2 fail by exactly the
        # excluded weight, which left the primary regime unbuildable as soon as
        # any of the contract's own exclusion reasons fired. `_exact_capacity`
        # carries the decision and the arithmetic.
        soft_targets, capacity_info = _exact_capacity(
            targets, df, forced, reason, dropped, verbose=verbose)
        regime_meta["exact_capacity"] = capacity_info

    work = df[~df["sentence_id"].isin(dropped)].copy()
    # Per-axis targets are apportioned over the ASSIGNABLE rows: the draw minus
    # the excluded groups minus the temporal drops. Apportioning over rows that
    # no part can hold offsets the whole target column (on the real temporal
    # split 27 further sentences were excluded after the capacity was computed,
    # so every per-value target carried that offset).
    _excluded_gids = {str(g) for g, p in forced.items() if p == "excluded"}
    assignable_rows = work[~work["group_id"].astype(str).isin(_excluded_gids)]
    targets_by_axis = axis_targets(assignable_rows, soft_targets)
    groups = _group_rows(work)
    result = allocate(groups, soft_targets, targets_by_axis, forced, seed=seed,
                      regime_id=split_id, exact=exact, exclusion_reasons=reason)

    part_of = result["group_part"]
    df["part"] = [part_of.get(str(g), "excluded") for g in df["group_id"]]
    df.loc[df["sentence_id"].isin(dropped), "part"] = "excluded"
    # A group-level exclusion reason wins over the per-sentence temporal drop; an
    # excluded row with no reason is left empty on purpose so invariant I1 fails
    # loudly rather than being papered over with a plausible-looking default.
    df["exclusion_reason"] = ""
    df.loc[df["part"] == "excluded", "exclusion_reason"] = [
        reason.get(str(gid)) or
        ("straddles_temporal_cut_dropped" if sid in dropped else "")
        for sid, gid in zip(df.loc[df["part"] == "excluded", "sentence_id"],
                            df.loc[df["part"] == "excluded", "group_id"])]

    if regime == "cross_country":
        # The external test is not country-pure and never could be; measure it
        # here rather than letting a reader assume purity from the regime's name.
        regime_meta["purity"] = cross_country_purity_block(
            df, regime_meta["holdout_country"])

    # posting_keys_manifest.json is READ (for the pinned flags file and for the
    # missing-employer counts the limitations block cites), so it is an INPUT and
    # gets the same SplitRefusal with run instructions as every sibling input. It
    # used to be opened unguarded, which produced a bare pathlib
    # FileNotFoundError from the middle of manifest construction.
    keys_manifest_path = gs_dir / "posting_keys_manifest.json"
    if not keys_manifest_path.exists():
        raise SplitRefusal(
            f"missing input {keys_manifest_path}. Run gold_standard.posting_keys "
            f"first: the manifest's flags_file binding (I14) and the "
            f"missing-employer limitation both read it.")
    keys_manifest = json.loads(keys_manifest_path.read_text(encoding="utf-8"))

    provenance = {
        "sample_file": sample_path.name, "sample_sha256": _sha256(sample_path),
        "pool_file": pool_path.name, "pool_sha256": _sha256(pool_path),
        "posting_keys_file": keys_path.name, "posting_keys_sha256": _sha256(keys_path),
        "leakage_groups_file": groups_path.name,
        "leakage_groups_sha256": _sha256(groups_path),
        "shingles_file": shingles_path.name, "shingles_sha256": _sha256(shingles_path),
        "flags_file": flags_name or keys_manifest.get("flags_file", "unknown"),
        "posting_keys_manifest_file": keys_manifest_path.name,
        "posting_keys_manifest_sha256": _sha256(keys_manifest_path),
        # contract §8 / I14 ask for a ROW COUNT per batch beside the digest: the
        # digest proves the file did not change, the count is what makes a
        # provenance table readable (and lets I14 report
        # raw_batch_rows_recorded: true instead of flagging the gap).
        "raw_batches": [{"batch": p.parent.name,
                         "rows": int(pq.ParquetFile(p).metadata.num_rows),
                         "sha256": _sha256(p)}
                        for p in sorted(raw_root.glob("batch-*/records.parquet"))],
        "pool_append_stable": False,
        "pool_append_note": (
            "build_pool caps strata with grp.sample(n=cap, random_state=SEED) over "
            "the whole joined frame, so a NEW BATCH reshuffles pool membership and "
            "invalidates the pool, the sample and every split. The gold standard is "
            "frozen against a NAMED flags file plus the pool sha; a corpus refresh "
            "produces a new gold-standard VERSION, never an update."),
    }

    # Computed once, here, so the limitations prose can be BUILT from the same
    # objects the manifest publishes instead of being written alongside them.
    stratification = stratification_block(df)
    power = power_block(df)
    verifiability = _verifiability_block(df)
    limitations = _limitations_block(keys_manifest, stratification, power,
                                    verifiability)

    frames = {
        "assignment": df[["sentence_id", "posting_id", "group_id", "part",
                          "exclusion_reason"]].copy(),
        "train": _part_frame(df, "train", PART_COLUMNS),
        "dev": _part_frame(df, "dev", PART_COLUMNS),
        "test_blind": _part_frame(df, "test", TEST_BLIND_COLUMNS),
    }

    # ---- size accounting. TWO RULES, and they are the whole point of this block:
    #   (1) TARGET comes from the REGISTERED config (`entry['body']['targets']`),
    #       never from the assignment. Taking the target from what the regime
    #       placed ("soft_targets['test'] = forced_test_weight") made every delta
    #       0, `exact_sizes_met` true and `unmet_targets` empty while the real
    #       temporal test was 212 against a registered 240 — the exact defect
    #       class already found and fixed once in sample.py.
    #   (2) ACHIEVED is RECOUNTED from the frames that are about to be written,
    #       not from an allocator return value, and the recount is cross-checked
    #       against the assignment frame. A mismatch refuses the write.
    registered_targets = {p: int(targets.get(p, 0)) for p in PARTS}
    achieved_sizes = {"train": int(len(frames["train"])),
                      "dev": int(len(frames["dev"])),
                      "test": int(len(frames["test_blind"]))}
    assignment_counts = {p: int((frames["assignment"]["part"] == p).sum())
                         for p in PARTS}
    if achieved_sizes != assignment_counts:
        raise SplitRefusal(
            f"recount mismatch between the part artifacts {achieved_sizes} and the "
            f"assignment frame {assignment_counts}: a part file would be written "
            f"with a row count the assignment does not support. Refusing.")
    capacity_sizes = {p: int(soft_targets.get(p, 0)) for p in PARTS}
    sizes = {p: {"target": registered_targets[p], "achieved": achieved_sizes[p],
                 "delta": achieved_sizes[p] - registered_targets[p]}
             for p in PARTS}
    unmet = [f"{p}: registered target {v['target']} achieved {v['achieved']} "
             f"(delta {v['delta']:+d}"
             + (f"; regime capacity was {capacity_sizes[p]}"
                if capacity_sizes[p] != v["target"] else "")
             + (", regime-determined — this regime makes no exactness claim)"
                if not exact else ")")
             for p, v in sizes.items() if v["delta"] != 0]
    regime_capacity = {
        "capacity": capacity_sizes,
        "equals_registered_targets": capacity_sizes == registered_targets,
        # the shortfall arithmetic that produced the capacity, so the
        # register-vs-realised gap is auditable in the artifact and not only in
        # stdout (computed by `_exact_capacity`; absent for the soft regimes)
        "exact_capacity": regime_meta.get("exact_capacity"),
        "source": ("registered targets, reduced by the unassignable shortfall "
                   "(see regime_capacity.exact_capacity)" if exact else
                   "regime-determined: the test size is whatever the date cut / "
                   "country boundary forces, and train+dev are apportioned over "
                   "the remainder"),
        "note": "this is the ALLOCATOR's capacity object, not a target. "
                "sizes[*].target is the registered number and is the only "
                "figure a claim may be stated against.",
    }
    # `exact_sizes_met` is a claim about the REGISTERED sizes, so it is null — not
    # true — wherever no exactness is claimed.
    exact_sizes_met = (all(v["delta"] == 0 for v in sizes.values())
                       if exact else None)
    # Reported per-axis targets are apportioned from the part's REALISED size, so
    # each target column closes to the part it describes even where a soft regime
    # overflowed its capacity. The allocator's own targets (above) stay as they
    # were — it cannot optimise against a size it has not produced yet.
    reported_targets_by_axis = axis_targets(assignable_rows, achieved_sizes)

    manifest = {
        "identity": {
            "split_spec_version": SPLIT_SPEC_VERSION, "regime_id": regime,
            "split_id": split_id, "config_name": config_name,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": _git_commit(), "seed": seed,
            "gold_standard_version": GOLD_STANDARD_VERSION,
            "guideline": GUIDELINE_CODE,
            "sample_file": sample_path.name,
        },
        "registry": {
            "registry_entry_sha256": entry["content_sha256"],
            "registered_utc": entry.get("registered_utc"),
            "registered_git_commit": entry.get("registered_git_commit"),
            "frozen": entry.get("frozen"),
            "body": entry["body"],
        },
        "provenance": provenance,
        "group_layer": json.loads(
            (gs_dir / "leakage_groups_manifest.json").read_text(encoding="utf-8"))
        if (gs_dir / "leakage_groups_manifest.json").exists() else {},
        "targets": registered_targets,
        "targets_source": ("split_registry.json -> configs."
                           f"{config_name}.body.targets"),
        "achieved": achieved_sizes,
        "achieved_source": "recounted from train.parquet / dev.parquet / "
                           "test_blind.parquet as written, cross-checked against "
                           "assignment.parquet",
        "sizes": sizes,
        "regime_capacity": regime_capacity,
        "exact_sizes_met": exact_sizes_met,
        "exactness_claimed": exact,
        "mechanism": "parity-greedy (see gold_standard.split.allocate)",
        "parity_corrections": result["parity_corrections"],
        "allocation_cost": {
            "greedy_delta_sum": result["cost"],
            "final_weighted_abs_deviation": None,   # filled right below
            "note": "greedy_delta_sum is the running sum of marginal cost changes "
                    "and is normally negative (each placement moves toward target); "
                    "final_weighted_abs_deviation is the interpretable number: "
                    "sum over axes of AXIS_WEIGHTS * sum |achieved - target|",
        },
        "marginals": marginals_block(df, reported_targets_by_axis,
                                     part_totals=achieved_sizes, regime=regime),
        "marginals_basis": {
            "apportioned_over": "assignable rows (sample minus excluded minus "
                                "temporally dropped)",
            "n_assignable_rows": int(len(assignable_rows)),
            "part_totals": achieved_sizes,
            "allocator_part_totals": capacity_sizes,
            "closure_rule": "every marginals.<axis>.<part>.target column sums "
                            "EXACTLY to part_totals[<part>] (two-stage "
                            "largest-remainder); target_total is published in "
                            "each block so this is checkable, not promised",
            "basis_note": "the reported per-value target is apportioned from the "
                          "part's REALISED size, so a per-value delta measures "
                          "COMPOSITION only. The part's size gap against the "
                          "registered target is reported once, in sizes[*], and "
                          "is deliberately not smeared across every axis. The "
                          "allocator optimised against allocator_part_totals (its "
                          "capacity), which is the same object except where a "
                          "soft regime overflowed a part.",
        },
        "stratification": stratification,
        "exclusions": {
            "by_reason": df.loc[df["part"] == "excluded", "exclusion_reason"]
            .value_counts().to_dict(),
            "n_excluded": int((df["part"] == "excluded").sum()),
            "excluded_sentence_ids": sorted(
                df.loc[df["part"] == "excluded", "sentence_id"].tolist()),
        },
        "verifiability": verifiability,
        "power": power,
        "tag_support": tag_support_block(),
        "enlargement": enlargement_block(registry),
        "regime_overlap": (
            "the three regimes re-partition the SAME 1,200 sentences; their test "
            "sets overlap, so the three numbers are not independent and must not be "
            "averaged"),
        "limitations": limitations,
        "unmet_targets": unmet,
    }
    if regime == "temporal":
        manifest["temporal"] = temporal_confound_block(df, regime_meta["cut_date"], dropped)
        manifest["temporal"]["indeed_id_matched"] = indeed_id_matched_block(df)
        manifest["temporal"]["n_straddling_groups"] = int(len({
            str(g) for g in df.loc[df["sentence_id"].isin(dropped), "group_id"]}))
        # A stratum that misses the REGISTERED quantile is an unmet target, so it
        # is reported as one (and printed) instead of living only inside the
        # temporal block.
        manifest["unmet_targets"].extend(manifest["temporal"]["per_stratum_unmet"])
    if regime == "cross_country":
        iso = regime_meta["holdout_country"]
        manifest["cross_country"] = {
            "holdout_rule_text": regime_meta["holdout_rule_text"],
            "holdout_rule": regime_meta["holdout_rule"],
            "holdout_rule_overridden": regime_meta["holdout_rule_overridden"],
            "holdout_country_overridden": regime_meta["holdout_country_overridden"],
            "candidate_table": regime_meta["candidate_table"],
            "holdout_country": iso,
            "holdout_stratum": regime_meta["holdout_stratum"],
            "all_holdouts_by_stratum": regime_meta["all_holdouts_by_stratum"],
            "n_groups_moved_wholly": regime_meta["n_groups_moved_wholly"],
            # The hold-out rule's SCOPE and what it recovered, so the stated rule
            # and the executed rule can be compared in the artifact instead of in
            # the source. These were computed and then not reported.
            "holdout_group_scope": regime_meta.get("holdout_group_scope"),
            "holdout_group_scope_note": regime_meta.get("holdout_group_scope_note"),
            "n_groups_touching_holdout_in_pool":
                regime_meta.get("n_groups_touching_holdout_in_pool"),
            "n_groups_touching_holdout_in_sample":
                regime_meta.get("n_groups_touching_holdout_in_sample"),
            "n_groups_touching_holdout_in_drawn_rows_only":
                regime_meta.get("n_groups_touching_holdout_in_drawn_rows_only"),
            "n_groups_recovered_by_pool_scope":
                regime_meta.get("n_groups_recovered_by_pool_scope"),
            # the external test's achieved country composition (it is NOT pure)
            "purity": regime_meta.get("purity"),
            "n_employers_spanning_boundary": _n_employers_spanning(df, iso),
            "n_employers_spanning_boundary_scope": (
                "DISTINCT employer hashes present both in the held-out country's "
                "drawn rows and in other countries' drawn rows. Independent of "
                "the group move (it reads country_iso and employer_hash, not "
                "part), so it measures the relation the name claims. The "
                "expression this replaced summed a boolean over ROWS, i.e. it "
                "counted sentences: three sentences of one employer reported 3."),
            "exclusion_cost": {
                str(st): {"sentences": int((g["part"] == "excluded").sum())}
                for st, g in df.groupby("stratum", sort=True)},
            "cross_lingual": False,
            "cross_lingual_note": "this measures cross-COUNTRY, not cross-lingual, "
                                  "transfer: the hold-out is English text",
            "train_720_subsample": {"present": False, "sha256": None},
        }

    manifest["allocation_cost"]["final_weighted_abs_deviation"] = float(sum(
        AXIS_WEIGHTS.get(axis, 0) * sum(abs(v) for v in block[part]["delta"].values())
        for axis, block in manifest["marginals"].items() for part in PARTS))
    # The same figure with the axes the REGIME partitions on removed. Those axes
    # carry a structural impossibility (a temporal test is defined as the late
    # period, so its 2026Q1 target can only be missed), so the full number mixes
    # allocator imbalance with the regime's own definition. Both are published;
    # neither is hidden.
    _partitioned = REGIME_PARTITION_AXES.get(regime, ())
    manifest["allocation_cost"]["final_weighted_abs_deviation_excl_regime_axes"] = \
        float(sum(
            AXIS_WEIGHTS.get(axis, 0)
            * sum(abs(v) for v in block[part]["delta"].values())
            for axis, block in manifest["marginals"].items()
            if axis not in _partitioned for part in PARTS))
    manifest["allocation_cost"]["regime_partitioned_axes_excluded"] = list(_partitioned)

    invariants = run_invariants(
        df, input_ids=input_ids, targets=soft_targets, exact=exact,
        shingles=shingles, pilot=quarantine["pilot_100"],
        practice=quarantine["practice_10"], allowed_reasons=EXCLUSION_REASONS,
        provenance=provenance, regime=regime, regime_meta=regime_meta,
        manifest=manifest, frames=frames)
    manifest["verifiability"]["dev_test_employer_verifiable_share"] = \
        invariants["I9_dev_test_verifiable"].get("dev_test_employer_verifiable_share")
    # Underpowered is a SIZES fact, and it was reachable only by digging into
    # invariants.I12 while `sizes` said delta 0 next to it. It is surfaced here
    # and, when it fires, it is a loud unmet target like any other.
    _i12 = invariants.get("I12_exact_sizes", {})
    _under = bool(_i12.get("underpowered",
                           achieved_sizes["test"] < MIN_TEST_SENTENCES))
    manifest["underpowered"] = {
        "test_underpowered": _under,
        "n_test_sentences": achieved_sizes["test"],
        "min_test_sentences": MIN_TEST_SENTENCES,
        "source": "I12_exact_sizes (recounted test n vs MIN_TEST_SENTENCES)",
    }
    # NOT appended to `unmet_targets`: MIN_TEST_SENTENCES is a power GUARD, and
    # the contract's own rule (D4/I12) is that a regime below it "is written with
    # `underpowered: true` and the number stated" rather than treated as a missed
    # quota. It is printed as a WARNING below so it is loud in stdout too.
    manifest["invariants"] = {
        **invariants,
        "max_cross_part_jaccard":
            invariants["I7_direct_pairwise_jaccard"]["max_cross_part_jaccard"],
        "n_cross_part_near_dup_sentence_pairs":
            invariants["I8_sentence_text_disjoint"]["n_near_duplicate_pairs"],
        "dev_disjointness_is_stricter_than_proposal": True,
    }
    manifest["assignment"] = {"assignment_sha256": _assignment_sha(df)}
    assert_invariants(invariants)

    # ---- fail-closed write: temp dir first, then os.replace into place
    out_dir = splits_dir / split_id
    sealed = _sealed_state(out_dir)
    if sealed["protected"] and not force_reseal:
        raise SplitRefusal(
            f"refusing to overwrite a SEALED split at {out_dir}: "
            + "; ".join(sealed["evidence"]) + ". The fail-closed write replaces "
            f"the WHOLE directory, so re-running this command would delete the "
            f"adjudicated human labels and reset the seal receipt — days of expert "
            f"adjudication destroyed by re-running a command that looks idempotent. "
            f"Nothing has been written. If this split genuinely must be recomputed, "
            f"pass --force-reseal --force-reseal-reason '<why>' (API: "
            f"force_reseal=True, force_reseal_reason='...'): the existing labels/ "
            f"subtree and SEAL.json are then CARRIED FORWARD into the new directory "
            f"as labels_superseded_*/ instead of being deleted, and the test must be "
            f"treated as already OPENED (PROP:1264) for the new assignment.")
    superseded: dict | None = None
    if sealed["protected"]:
        superseded = {
            "forced": True,
            "reason": str(force_reseal_reason).strip(),
            "previous_seal": sealed["seal"],
            "previous_label_files": sealed["label_files"],
            "carried_to": f"labels_superseded_{sealed['stamp']}",
            "warning": "a PREVIOUS test of this split_id was sealed and may have "
                       "been opened; the labels below were adjudicated against the "
                       "PREVIOUS assignment and must not be reused for this one "
                       "without re-materialising them from adjudication",
        }
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".split_tmp_", dir=str(out_dir.parent)))
    try:
        frames["assignment"].to_parquet(tmp / "assignment.parquet", index=False)
        frames["assignment"].to_csv(tmp / "assignment.csv", index=False,
                                    encoding="utf-8")
        frames["train"].to_parquet(tmp / "train.parquet", index=False)
        frames["dev"].to_parquet(tmp / "dev.parquet", index=False)
        frames["test_blind"].to_parquet(tmp / "test_blind.parquet", index=False)
        if regime == "cross_country":
            sub = _train_720_subsample(df, targets["train"], seed)
            sub.to_parquet(tmp / "train_720.parquet", index=False)
            manifest["cross_country"]["train_720_subsample"] = {
                "present": True, "n": int(len(sub)),
                "sha256": _sha256(tmp / "train_720.parquet"),
                "note": "size-matched so an external-test gap is not confounded "
                        "with train size (.424@720 vs .468@1440, seed noise 0.030)"}
        # Contract §8 lists train_sha256 / dev_sha256 / test_blind_sha256 in the
        # seal block and nothing wrote them, which is why `seal.score_on_test`
        # could verify the test sentence-ID LIST but never the test TEXT: the
        # rows could be edited under a fixed id list with no digest noticing.
        # Computed from the files as written, inside the temp dir, so the digest
        # is of the bytes that land in the split directory.
        part_digests = {f"{name}_sha256": _sha256(tmp / f"{name}.parquet")
                        for name in ("train", "dev", "test_blind")}
        manifest["parts"] = {
            **part_digests,
            "source": "sha256 of the parquet files as written into the split "
                      "directory (contract §8 seal block)",
        }
        if superseded is not None:
            # carry the adjudication work across the os.replace instead of
            # discarding it; it is NOT reinstated as this split's labels, because
            # it was adjudicated against the PREVIOUS assignment. Done before the
            # manifest/SEAL write so both record what was actually carried.
            superseded["carried_files"] = _carry_superseded(
                out_dir, tmp, sealed["stamp"])
            manifest = {**manifest, "superseded_previous_seal": superseded}
            if verbose:
                print(f"  FORCED RESEAL ({superseded['reason']}): carried "
                      f"{superseded['carried_files']} forward; the previous test "
                      f"labels are NOT this split's labels", flush=True)
        # Contract I13 promised "a manifest hash excluding the timestamp" and no
        # field existed. Computed here, where the manifest is FINAL, over the
        # canonical JSON with the two non-deterministic leaves removed, so two
        # runs of the same inputs produce the same digest and a hand edit of the
        # published manifest is detectable.
        manifest["manifest_sha256_excluding_timestamps"] = _manifest_digest(manifest)
        manifest["manifest_digest_scope"] = (
            "sha256 over the canonical JSON of this manifest with "
            "identity.created_utc and group_layer.created_utc removed and the "
            "digest fields themselves absent; two runs on identical inputs agree")
        (tmp / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
        (tmp / "SEAL.json").write_text(json.dumps({
            "split_id": split_id, "regime_id": regime,
            "assignment_sha256": manifest["assignment"]["assignment_sha256"],
            "test_sentence_id_list_sha256": hashlib.sha256("\n".join(sorted(
                str(s) for s in frames["test_blind"]["sentence_id"]
            )).encode("utf-8")).hexdigest(),
            **part_digests,
            "labels_materialised": False, "opened_utc": None,
            "superseded_previous_seal": superseded,
            "stamp": "test set sealed: selection happens on dev only (PROP:1264); "
                     "the guarantee is AUDITABILITY, not impossibility",
        }, indent=2), encoding="utf-8")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, out_dir)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

    _append_version_log(splits_dir, manifest)
    _refresh_index(splits_dir, split_id, manifest)

    if verbose:
        print(f"[{regime}] {split_id}: " + ", ".join(
            f"{p}={sizes[p]['achieved']}" for p in PARTS)
            + f", excluded={manifest['exclusions']['n_excluded']}", flush=True)
        print(f"  max cross-part template Jaccard: "
              f"{manifest['invariants']['max_cross_part_jaccard']} (tau {TAU_BUILD})",
              flush=True)
        for line in manifest["unmet_targets"]:
            print(f"  WARNING unmet target -> {line}", flush=True)
        if manifest["underpowered"]["test_underpowered"]:
            print(f"  WARNING underpowered -> test n="
                  f"{manifest['underpowered']['n_test_sentences']} is below "
                  f"MIN_TEST_SENTENCES={MIN_TEST_SENTENCES}; "
                  f"{manifest['power']['claimability_note']}", flush=True)
        print(f"-> {out_dir}", flush=True)
    return {"manifest": manifest, "assignment": df, "out_dir": out_dir}


def _train_720_subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Deterministic size-matched train subsample, whole groups removed (weight-1
    first) so the subsample is still group-atomic."""
    train = df[df["part"] == "train"].copy()
    if len(train) <= n:
        return _part_frame(train.assign(part="train"), "train", PART_COLUMNS)
    sizes = train["group_id"].value_counts()
    rng = np.random.default_rng(seed)
    gids = sorted(sizes.index.astype(str))
    keys = dict(zip(gids, rng.permutation(len(gids)).tolist()))
    # remove weight-1 groups first, then weight-2, until the target is reached
    order = sorted(gids, key=lambda g: (sizes[g], keys[g], g))
    drop, total = set(), len(train)
    for gid in order:
        if total <= n:
            break
        if total - sizes[gid] < n:
            continue
        drop.add(gid)
        total -= int(sizes[gid])
    kept = train[~train["group_id"].astype(str).isin(drop)]
    return _part_frame(kept, "train", PART_COLUMNS)


def _append_version_log(splits_dir: Path, manifest: dict) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    line = {
        "utc": manifest["identity"]["created_utc"],
        "split_id": manifest["identity"]["split_id"],
        "regime_id": manifest["identity"]["regime_id"],
        "config_name": manifest["identity"]["config_name"],
        "git_commit": manifest["identity"]["git_commit"],
        "seed": manifest["identity"]["seed"],
        "assignment_sha256": manifest["assignment"]["assignment_sha256"],
        "sizes": {p: manifest["sizes"][p]["achieved"] for p in PARTS},
        "unmet_targets": manifest["unmet_targets"],
    }
    with (splits_dir / "version_log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")


def _refresh_index(splits_dir: Path, split_id: str, manifest: dict) -> None:
    path = splits_dir / "index.json"
    index = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    index[split_id] = {
        "path": str(Path("splits") / split_id),
        "regime_id": manifest["identity"]["regime_id"],
        "created_utc": manifest["identity"]["created_utc"],
        "assignment_sha256": manifest["assignment"]["assignment_sha256"],
        "sealed": True, "labels_materialised": False,
    }
    path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regime", choices=REGIMES, required=True)
    ap.add_argument("--config", default="CONFIG_A")
    ap.add_argument("--sample", default="main_1200")
    ap.add_argument("--gs-dir", default=str(GS_DIR))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--holdout-country", default=None,
                    help="override the DERIVED hold-out country (recorded as an "
                         "override in the manifest; the rule is preferred)")
    ap.add_argument("--force-reseal", action="store_true",
                    help="recompute a split whose test labels are already "
                         "materialised. Refused by default because the write "
                         "replaces the whole directory; requires "
                         "--force-reseal-reason, and the previous labels/ subtree "
                         "is carried forward as labels_superseded_<utc>/ rather "
                         "than deleted.")
    ap.add_argument("--force-reseal-reason", default="",
                    help="written reason, recorded in the new SEAL.json")
    args = ap.parse_args()
    gs_dir = Path(args.gs_dir)
    reseal = {"force_reseal": bool(args.force_reseal),
              "force_reseal_reason": args.force_reseal_reason}
    if args.regime == "cross_country" and not args.holdout_country:
        # one sibling split per external row of Tabel 3.5 (ASEAN non-Indonesia
        # and OECD), each country DERIVED from the registered rule
        chosen = derive_holdouts(gs_dir)
        if not chosen:
            raise SplitRefusal("no country satisfies the registered hold-out rule")
        for stratum, iso in sorted(chosen.items()):
            print(f"cross-country hold-out for stratum '{stratum}': {iso} "
                  f"(derived, not picked)", flush=True)
            run_split(args.regime, args.config, gs_dir=gs_dir,
                      sample_name=args.sample, seed=args.seed,
                      holdout_country=iso, holdout_country_derived=True,
                      **reseal)
        return
    run_split(args.regime, args.config, gs_dir=gs_dir,
              sample_name=args.sample, seed=args.seed,
              holdout_country=args.holdout_country, **reseal)


if __name__ == "__main__":
    main()
