"""Test-set seal: label materialisation and single-open receipts.

PROP:1264 fixes the rule this module enforces: selection happens on the
development set and the development time window, and "test set serta jendela
hold-out temporal hanya dibuka satu kali untuk konfigurasi terpilih" — the test
set and the temporal hold-out window are opened ONCE, for the selected
configuration. PROP:1442 adds that the held-out gold test set is used for
neither prompt refinement, example selection nor model weighting.

`gold_standard.split` writes no label column anywhere (invariant I17). This is
the ONLY module that writes `labels/test_labels.parquet`, and every published
test metric carries a `receipt_id` that resolves in `test_open_log.jsonl`.

The guarantee is AUDITABILITY, not impossibility: nothing stops a researcher
opening the parquet by hand. Claim exactly that and no more.

Items adjudicated AMBIGUOUS / EXCLUDE are dropped from the strict gold and
WRITTEN to `labels/test_challenge_set.parquet` (B3:413). They are never
backfilled into the strict gold, so the GOLD-BEARING sentence count is <= 240 --
but the scored test n is the sealed n (see SCORING DENOMINATOR below), not the
gold-bearing count.

CHALLENGE SET (the second thing this module got wrong, so it is also written
down). Dropping an adjudicated span from the gold does not make its sentence a
non-skill sentence. Under the sealed-denominator rule a sentence whose only
adjudicated spans were dropped stays a scoring unit with an EMPTY gold set, so a
model reproducing exactly the span the human adjudicators could not decide on is
charged a false positive it had no way to avoid -- and published precision is
deflated in proportion to the adjudication's own ambiguity rate, which is a
property of the annotation process, not of the model. Counting those rows is not
enough: the scorer needs them ON DISK to tell "the annotators found no skill
here" from "the annotators could not decide". So `materialise_labels` writes
them, and `score_on_test` publishes BOTH numbers -- `micro` (strict: a predicted
challenge span is a false positive) and `micro_challenge_neutral` (challenge
spans neither credited nor penalised). `micro` remains the primary metric, since
that is the rule `MATCH_RULE_TEXT` was registered with; the gap between the two
IS the ambiguity cost, published as `challenge_cost_precision_delta` rather than
left invisible.

SCORING DENOMINATOR (the rule this module got wrong once, so it is written
down). The scoring population is EVERY sentence in `test_blind.parquet`, not
every sentence that carries a gold span. A sealed test sentence with no gold
span is a real scoring unit whose gold set is empty, so a span predicted on it
is a false positive and must enter `n_pred`. Dropping those units -- which is
what iterating the gold frame does -- deletes exactly the errors that the
NON_RELEVANT / `has_5tag = TIDAK` part of the sample design exists to measure
(`sample.py` draws a BORDERLINE_SHARE minority precisely so that a non-skill
sentence is a live option), and it reports a model that hallucinates on every
non-skill sentence as if it had not. `n_test_sentences` is therefore the sealed
n, the bootstrap resamples sealed SENTENCES, and the gold-bearing subset is
reported separately as `n_gold_bearing_sentences`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import TAGS
from .posting_keys import GS_DIR

SPLITS_DIR = GS_DIR / "splits"
DROP_VERDICTS = ("AMBIGUOUS", "EXCLUDE")
LABEL_COLUMNS = ["sentence_id", "span_start", "span_end", "tag", "verdict"]
SPAN_COLUMNS = ["sentence_id", "span_start", "span_end", "tag"]
BOOTSTRAP_ITERS = 2000
BOOTSTRAP_SEED = 20261115


class SealViolation(RuntimeError):
    """Raised instead of returning a test number that cannot be trusted."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _split_dir(split_id: str, splits_dir: Path = SPLITS_DIR) -> Path:
    path = Path(splits_dir) / split_id
    if not path.exists():
        raise SealViolation(f"no such split: {path}")
    return path


def _sealed_sentence_ids(path: Path, seal: dict) -> list[str]:
    r"""The sealed test sentence ids, re-verified against SEAL.json's digest.

    `split.py` writes `test_sentence_id_list_sha256` over
    "\n".join(sorted(test_blind.sentence_id)). Recomputing it here means the
    SCORING DENOMINATOR itself is sealed: editing `test_blind.parquet` to add or
    remove sentences can no longer change the reported n or the bootstrap
    population without the open being refused.
    """
    # Contract §8's `test_blind_sha256` is now written by `split.py`, so the TEXT
    # is verifiable too, not only the id list. The id list is the weaker claim:
    # rows could be edited under a fixed set of ids and nothing would notice. The
    # file digest is checked FIRST and only when the seal carries it, so a split
    # sealed by an older writer still scores (and says which guarantee it has).
    blind_path = path / "test_blind.parquet"
    registered_file = seal.get("test_blind_sha256")
    if registered_file:
        observed_file = hashlib.sha256(
            blind_path.read_bytes()).hexdigest()
        if observed_file != registered_file:
            raise SealViolation(
                f"{path.name}: test_blind.parquet does not match the sealed file "
                f"digest (sha256 {observed_file} != sealed {registered_file}); the "
                f"sealed test TEXT has changed since the split was written, so no "
                f"number computed on it is the pre-registered one")
    frame = pd.read_parquet(blind_path, columns=["sentence_id"])
    ids = sorted(str(s) for s in frame["sentence_id"])
    registered = seal.get("test_sentence_id_list_sha256")
    if not registered:
        raise SealViolation(
            f"{path.name}: SEAL.json carries no test_sentence_id_list_sha256, so "
            f"the sealed sentence list cannot be verified; re-write the split with "
            f"gold_standard.split (it stamps that digest) before scoring")
    observed = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    if observed != registered:
        raise SealViolation(
            f"{path.name}: test_blind.parquet's sentence id list no longer matches "
            f"the sealed digest (sha256 {observed} != sealed {registered}); the "
            f"scoring denominator is not the one that was pre-registered")
    if len(set(ids)) != len(ids):
        n_dupe = len(ids) - len(set(ids))
        seen, dupes = set(), []
        for i in ids:
            if i in seen and i not in dupes:
                dupes.append(i)
            seen.add(i)
        raise SealViolation(
            f"{path.name}: test_blind.parquet repeats sentence ids {dupes[:5]} "
            f"({n_dupe} duplicate rows), which would weight those sentences "
            f"twice in every test metric")
    return ids


# ------------------------------------------------------------------- labels
def materialise_labels(split_id: str, adjudicated_gold: pd.DataFrame,
                       splits_dir: Path = SPLITS_DIR) -> dict:
    """Write labels/test_labels.parquet for the sealed test sentences only.

    `adjudicated_gold` must carry at least sentence_id, span_start, span_end,
    tag and verdict. Rows whose verdict is AMBIGUOUS/EXCLUDE are dropped from
    the strict test set (B3:413) and counted, never silently kept.

    Exact-duplicate (sentence_id, span_start, span_end, tag) rows are collapsed
    and counted: two identical adjudicated rows cannot be two distinct gold
    spans, and leaving them in would inflate `n_gold` and deflate recall.
    """
    path = _split_dir(split_id, splits_dir)
    seal = json.loads((path / "SEAL.json").read_text(encoding="utf-8"))
    sealed_ids = _sealed_sentence_ids(path, seal)
    missing = [c for c in LABEL_COLUMNS if c not in adjudicated_gold.columns]
    if missing:
        raise SealViolation(f"adjudicated gold is missing columns {missing}")

    sealed_set = set(sealed_ids)
    gold = adjudicated_gold[adjudicated_gold["sentence_id"].astype(str).isin(sealed_set)]
    dropped = gold[gold["verdict"].astype(str).str.upper().isin(DROP_VERDICTS)]
    strict = gold[~gold["verdict"].astype(str).str.upper().isin(DROP_VERDICTS)]
    strict = strict[LABEL_COLUMNS].sort_values(
        ["sentence_id", "span_start", "span_end", "tag"], kind="stable"
    ).reset_index(drop=True)
    n_before = len(strict)
    strict = strict.drop_duplicates(subset=SPAN_COLUMNS, keep="first").reset_index(drop=True)
    n_dupes = n_before - len(strict)

    labels_dir = path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    out = labels_dir / "test_labels.parquet"
    strict.to_parquet(out, index=False)

    # ---- the challenge set is WRITTEN, not merely counted.
    # Without this file on disk the scorer cannot distinguish a sealed sentence
    # the annotators judged non-skill from one they could not decide, and every
    # prediction reproducing an adjudicator's own AMBIGUOUS/EXCLUDE span is
    # charged a false positive with nothing in the result to show it. Writing it
    # is what makes `micro_challenge_neutral` computable at scoring time.
    challenge = dropped[LABEL_COLUMNS].sort_values(
        ["sentence_id", "span_start", "span_end", "tag"], kind="stable"
    ).reset_index(drop=True)
    n_chal_before = len(challenge)
    challenge = challenge.drop_duplicates(subset=SPAN_COLUMNS,
                                          keep="first").reset_index(drop=True)
    n_chal_dupes = n_chal_before - len(challenge)
    chal_out = labels_dir / "test_challenge_set.parquet"
    challenge.to_parquet(chal_out, index=False)

    gold_bearing = int(strict["sentence_id"].nunique())
    gold_ids = set(strict["sentence_id"].astype(str))
    chal_ids = set(challenge["sentence_id"].astype(str))
    # A sealed sentence is CHALLENGE-ONLY when every adjudicated span on it was
    # dropped: its empty gold set means "undecided", not "no skill here".
    challenge_only = sorted(chal_ids - gold_ids)
    non_skill = [sid for sid in sealed_ids
                 if sid not in gold_ids and sid not in chal_ids]
    n_empty = len(sealed_ids) - gold_bearing
    if len(challenge_only) + len(non_skill) != n_empty:
        raise SealViolation(
            f"{path.name}: the gold-empty sealed sentences do not partition into "
            f"challenge-only ({len(challenge_only)}) + non-skill "
            f"({len(non_skill)}) = {len(challenge_only) + len(non_skill)}, but "
            f"{n_empty} sealed sentences carry no strict gold span; the challenge "
            f"set and the strict labels disagree about which sentences were "
            f"adjudicated, so neither metric would mean what it says")
    seal.update({
        "labels_materialised": True,
        "test_labels_sha256": _sha256_file(out),
        "test_challenge_set_sha256": _sha256_file(chal_out),
        # The SCORED denominator is the sealed n (see the module docstring).
        # `realised_test_n_sentences` used to carry the GOLD-BEARING count while
        # score_on_test reported the sealed n, so one sealed split published two
        # contradictory "test n" values and the smaller one had the name a reader
        # would cite. The gold-bearing subset keeps its own field and is not the
        # test n.
        "realised_test_n_sentences": len(sealed_ids),
        "scored_test_n_sentences": len(sealed_ids),
        "sealed_test_n_sentences": len(sealed_ids),
        "n_gold_bearing_sentences": gold_bearing,
        "n_gold_empty_sealed_sentences": n_empty,
        "n_challenge_only_sentences": len(challenge_only),
        "n_non_skill_sealed_sentences": len(non_skill),
        "n_spans": int(len(strict)),
        "n_challenge_spans": int(len(challenge)),
        "n_dropped_ambiguous_or_exclude": int(len(dropped)),
        "n_duplicate_span_rows_dropped": int(n_dupes),
        "n_duplicate_challenge_span_rows_dropped": int(n_chal_dupes),
        "challenge_set_note": "AMBIGUOUS / EXCLUDE items are dropped from the "
                              "strict gold and WRITTEN to "
                              "labels/test_challenge_set.parquet (B3:413); they "
                              "are NEVER backfilled into the strict gold. "
                              "score_on_test reads that file, so a predicted span "
                              "matching an adjudicator's own undecided span is "
                              "reported separately instead of silently counting "
                              "as a false positive",
        "scoring_denominator_note": "every SEALED test sentence is a scoring unit, "
                                    "including the n_gold_empty_sealed_sentences that "
                                    "carry no gold span: a span predicted on one of "
                                    "those is a false positive, not a non-event. "
                                    "Those empty-gold units split into "
                                    "n_non_skill_sealed_sentences (the annotators "
                                    "found no skill) and n_challenge_only_sentences "
                                    "(the annotators could not decide), which are "
                                    "NOT the same evidence",
        "materialised_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    (path / "SEAL.json").write_text(json.dumps(seal, indent=2), encoding="utf-8")
    print(f"test labels: {seal['n_spans']} spans over "
          f"{seal['n_gold_bearing_sentences']} of {seal['sealed_test_n_sentences']} "
          f"sealed sentences ({seal['n_dropped_ambiguous_or_exclude']} dropped"
          + (f", {n_dupes} duplicate rows collapsed" if n_dupes else "")
          + f"); scored test n is the sealed {seal['scored_test_n_sentences']}, of "
            f"which {seal['n_non_skill_sealed_sentences']} are non-skill and "
            f"{seal['n_challenge_only_sentences']} challenge-only "
            f"({seal['n_challenge_spans']} challenge spans -> {chal_out.name})",
          flush=True)
    return seal


# ------------------------------------------------------------------- scoring
# The span-F1 definition is NOT re-invented here. Per-sentence tp is
# gold_standard.iaa._greedy_match (maximum-cardinality one-to-one matching, so
# nested/overlapping spans cannot be double-counted) in its SkillSpan-comparable
# EXACT mode (label_sensitive=True, exact=True), and P/R/F1 come from
# gold_standard.iaa._prf, including its NaN convention: a unit with neither a
# gold nor a predicted span has nothing to score, so P/R/F1 are undefined (NaN,
# rendered 'n/a') rather than a flattering 1.0. The CI uses iaa._percentile_ci,
# which drops undefined resamples and refuses a CI built on fewer than 20 of
# them. Importing rather than reimplementing is deliberate: two incompatible
# span-F1s in one repo would make the IAA ceiling and the model score
# non-comparable, and the IAA ceiling is the number the model score is read
# against.
from .iaa import Span as _Span                                      # noqa: E402
# The PUBLIC names: these three are the cross-module API (see iaa.py and
# gold_standard/metrics.py). They are the same objects as the underscored
# ones — tests/test_metrics.py pins the identity.
from .iaa import greedy_match as _greedy_match                      # noqa: E402
from .iaa import percentile_ci as _percentile_ci                    # noqa: E402
from .iaa import prf as _prf                                        # noqa: E402

MATCH_MODE = dict(label_sensitive=True, exact=True)
MATCH_RULE_TEXT = ("exact span + type match, micro-averaged over ALL sealed test "
                   "sentences (gold-empty ones included, so a span predicted on a "
                   "non-skill sentence is a false positive); tp and P/R/F1 are "
                   "gold_standard.iaa._greedy_match / _prf in iaa's 'exact' mode, "
                   "the same definition as the inter-annotator ceiling (comparable "
                   "with SkillSpan, NNOSE, Skill-LLM, SAP-IE)")


def _spans_by_sid(frame: pd.DataFrame, sealed: set[str], *,
                  what: str) -> tuple[dict[str, list], dict]:
    """sentence_id -> list[Span], with an EMPTY LIST for every sealed sentence.

    The empty lists are the whole point of this function: a sealed sentence that
    carries no gold span must still be a scoring unit, otherwise a prediction on
    it is silently not an error.

    Malformed rows are refused, never skipped: silently dropping a malformed
    PREDICTION would delete a false positive and flatter the model.
    """
    missing = [c for c in SPAN_COLUMNS if c not in frame.columns]
    if missing:
        raise SealViolation(f"{what} frame is missing columns {missing}")
    by_sid: dict[str, list] = {sid: [] for sid in sealed}
    stats = {"n_rows": int(len(frame)), "n_rows_inside_sealed_test": 0,
             "n_rows_outside_sealed_test": 0, "n_sentences_outside_sealed_test": 0,
             "n_duplicate_span_rows": 0}
    outside: set[str] = set()
    bad: list[str] = []
    seen: set[tuple] = set()
    for r in frame.itertuples(index=False):
        sid = str(r.sentence_id)
        try:
            start, end = int(r.span_start), int(r.span_end)
        except (TypeError, ValueError):
            bad.append(f"{sid}: non-integer offsets ({r.span_start!r},{r.span_end!r})")
            continue
        tag = str(r.tag)
        if end <= start:
            bad.append(f"{sid}: empty/reversed span [{start},{end})")
            continue
        if tag not in TAGS:
            bad.append(f"{sid}: tag {tag!r} is outside the five-tag set {sorted(TAGS)}")
            continue
        if sid not in by_sid:
            outside.add(sid)
            stats["n_rows_outside_sealed_test"] += 1
            continue
        key = (sid, start, end, tag)
        if key in seen:
            stats["n_duplicate_span_rows"] += 1
        seen.add(key)
        by_sid[sid].append(_Span(start, end, tag))
        stats["n_rows_inside_sealed_test"] += 1
    if bad:
        raise SealViolation(
            f"{what} frame has {len(bad)} unusable span rows; they are refused "
            f"rather than skipped, because skipping a malformed prediction would "
            f"delete a false positive. First few: {bad[:5]}")
    stats["n_sentences_outside_sealed_test"] = len(outside)
    return by_sid, stats


def _contribs(gold_by_sid: dict, pred_by_sid: dict, sealed_ids,
              tag: str | None = None) -> dict[str, tuple[int, int, int]]:
    """Per-SEALED-SENTENCE (tp, n_gold, n_pred), one entry per sealed sentence."""
    out: dict[str, tuple[int, int, int]] = {}
    for sid in sealed_ids:
        g = gold_by_sid.get(sid, [])
        p = pred_by_sid.get(sid, [])
        if tag is not None:
            g = [s for s in g if s.label == tag]
            p = [s for s in p if s.label == tag]
        out[sid] = (_greedy_match(g, p, **MATCH_MODE), len(g), len(p))
    return out


def _r4(x: float) -> float:
    return round(float(x), 4)


def _micro(contribs: dict[str, tuple[int, int, int]]) -> dict:
    """Micro P/R/F1 via iaa._prf; its n_a is gold and its n_b is predicted."""
    pr = _prf(sum(c[0] for c in contribs.values()),
              sum(c[1] for c in contribs.values()),
              sum(c[2] for c in contribs.values()))
    return {"tp": pr["tp"], "n_gold": pr["n_a"], "n_pred": pr["n_b"],
            "precision": _r4(pr["precision"]), "recall": _r4(pr["recall"]),
            "f1": _r4(pr["f1"])}


def _bootstrap_ci(contribs: dict, iters: int = BOOTSTRAP_ITERS,
                  seed: int = BOOTSTRAP_SEED) -> tuple[float, float, int]:
    """95% CI for micro span-F1, resampling SEALED SENTENCES with replacement.

    The resampling unit is the sentence (the unit the split allocated and the
    unit DESIGN_EFFECT is argued over), never the gold span, and the population
    is every sealed sentence including the gold-empty ones -- the same population
    the point estimate is computed on, which is what makes the interval cover it.
    """
    sids = sorted(contribs)
    if len(sids) < 2:
        return float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    n = len(sids)
    vals = []
    for _ in range(iters):
        tp = n_gold = n_pred = 0
        for i in rng.integers(0, n, n):
            t, g, p = contribs[sids[i]]
            tp += t
            n_gold += g
            n_pred += p
        vals.append(_prf(tp, n_gold, n_pred)["f1"])
    return _percentile_ci(vals)


def score_on_test(split_id: str, predictions: pd.DataFrame, *,
                  model_config_sha: str, git_commit: str, reason: str,
                  force_reopen: bool = False, splits_dir: Path = SPLITS_DIR) -> dict:
    """Open the sealed test ONCE for one frozen configuration and return
    AGGREGATES only (micro/macro exact span+type F1 with bootstrap CI and per-tag
    support) — never per-sentence errors, which would let the test set steer the
    next iteration.

    Every sealed test sentence is a scoring unit. A sentence with no gold span
    contributes (tp=0, n_gold=0, n_pred=len(predicted)), so hallucinated spans on
    non-skill sentences cost precision instead of disappearing.
    """
    path = _split_dir(split_id, splits_dir)
    seal = json.loads((path / "SEAL.json").read_text(encoding="utf-8"))
    labels = path / "labels" / "test_labels.parquet"
    if not seal.get("labels_materialised") or not labels.exists():
        raise SealViolation(
            f"{split_id}: test labels have not been materialised; run "
            f"materialise_labels after adjudication")
    if _sha256_file(labels) != seal.get("test_labels_sha256"):
        return {"verdict": "VOID — test_labels.parquet changed after it was sealed "
                           f"(sha256 {_sha256_file(labels)} != sealed "
                           f"{seal.get('test_labels_sha256')}). Every test number "
                           "from this split is void; no score is computed.",
                "scored": False}

    # ---- the challenge set must be on disk whenever the adjudication dropped
    # anything. Refusing here rather than scoring without it is deliberate: the
    # strict metric is still computable, but the AMBIGUOUS/EXCLUDE deflation of
    # precision would then be invisible and unquantifiable, which is exactly the
    # defect this file now exists to prevent.
    challenge_path = path / "labels" / "test_challenge_set.parquet"
    n_dropped = int(seal.get("n_dropped_ambiguous_or_exclude") or 0)
    if not challenge_path.exists() and n_dropped:
        raise SealViolation(
            f"{split_id}: SEAL.json records {n_dropped} adjudicated span(s) dropped "
            f"as AMBIGUOUS/EXCLUDE, but labels/test_challenge_set.parquet does not "
            f"exist. Without it the scorer cannot tell a sentence the annotators "
            f"judged non-skill from one they could not decide, so every prediction "
            f"reproducing an adjudicator's undecided span would be charged as a "
            f"false positive with nothing in the result to show it. Re-run "
            f"materialise_labels (it writes that file) before scoring.")
    if (challenge_path.exists()
            and _sha256_file(challenge_path) != seal.get("test_challenge_set_sha256")):
        return {"verdict": "VOID — test_challenge_set.parquet changed after it was "
                           f"sealed (sha256 {_sha256_file(challenge_path)} != sealed "
                           f"{seal.get('test_challenge_set_sha256')}). The set of "
                           "spans exempt from the false-positive count is not the "
                           "sealed one, so no score is computed.",
                "scored": False}

    log_path = Path(splits_dir) / "test_open_log.jsonl"
    prior = []
    if log_path.exists():
        prior = [json.loads(line) for line in
                 log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    key = (split_id, model_config_sha)
    reopened = [r for r in prior
                if (r.get("split_id"), r.get("model_config_sha")) == key]
    if reopened and not force_reopen:
        raise SealViolation(
            f"{split_id} has already been opened for model config "
            f"{model_config_sha} (receipt {reopened[0]['receipt_id']}). PROP:1264 "
            f"allows a single open per selected configuration; pass "
            f"force_reopen=True WITH a written reason, which is also logged.")

    # ---- the scoring population: every SEALED test sentence, not every gold row
    sealed_ids = _sealed_sentence_ids(path, seal)
    sealed_set = set(sealed_ids)
    gold_by_sid, gold_stats = _spans_by_sid(pd.read_parquet(labels), sealed_set,
                                            what="sealed gold label")
    if gold_stats["n_rows_outside_sealed_test"]:
        raise SealViolation(
            f"{split_id}: test_labels.parquet carries "
            f"{gold_stats['n_rows_outside_sealed_test']} span rows for "
            f"{gold_stats['n_sentences_outside_sealed_test']} sentences that are "
            f"not in the sealed test; the labels must be a subset of the seal")
    pred_by_sid, pred_stats = _spans_by_sid(predictions, sealed_set,
                                            what="prediction")
    if pred_stats["n_rows"] and not pred_stats["n_rows_inside_sealed_test"]:
        raise SealViolation(
            f"{split_id}: none of the {pred_stats['n_rows']} predicted spans falls "
            f"on a sealed test sentence ({pred_stats['n_sentences_outside_sealed_test']} "
            f"unknown sentence ids). That is an id-space mismatch, not a score of 0; "
            f"predictions must be keyed by the sentence_id of test_blind.parquet")
    if pred_stats["n_rows_outside_sealed_test"]:
        print(f"WARNING: {pred_stats['n_rows_outside_sealed_test']} predicted spans on "
              f"{pred_stats['n_sentences_outside_sealed_test']} sentence ids outside "
              f"the sealed test are IGNORED (they are not test sentences)", flush=True)

    chal_by_sid: dict[str, list] = {sid: [] for sid in sealed_set}
    if challenge_path.exists():
        chal_by_sid, chal_stats = _spans_by_sid(
            pd.read_parquet(challenge_path), sealed_set, what="challenge-set")
        if chal_stats["n_rows_outside_sealed_test"]:
            raise SealViolation(
                f"{split_id}: test_challenge_set.parquet carries "
                f"{chal_stats['n_rows_outside_sealed_test']} span rows for "
                f"{chal_stats['n_sentences_outside_sealed_test']} sentences that are "
                f"not in the sealed test; the challenge set must be a subset of the "
                f"seal, like the labels")

    gold_keys = {(sid, s.start, s.end, s.label)
                 for sid, spans in gold_by_sid.items() for s in spans}
    challenge_keys = {(sid, s.start, s.end, s.label)
                      for sid, spans in chal_by_sid.items() for s in spans}
    contradictory = sorted(gold_keys & challenge_keys)
    if contradictory:
        raise SealViolation(
            f"{split_id}: {len(contradictory)} span(s) are in BOTH the strict gold "
            f"and the challenge set, i.e. adjudicated as resolved and as "
            f"undecided at once. A span cannot be both credited and exempt. First "
            f"few: {contradictory[:5]}")

    contribs = _contribs(gold_by_sid, pred_by_sid, sealed_ids)
    # Challenge-neutral prediction set: a prediction that reproduces EXACTLY an
    # adjudicator's dropped span is neither credited (it is not gold) nor
    # penalised (the gold standard does not assert it is wrong -- it asserts the
    # humans could not agree). Removing only exact matches is the narrowest
    # defensible exemption: a span that merely overlaps a challenge span is still
    # a false positive.
    pred_neutral = {sid: [s for s in spans
                          if (sid, s.start, s.end, s.label) not in challenge_keys]
                    for sid, spans in pred_by_sid.items()}
    n_pred_on_challenge = sum(len(pred_by_sid[sid]) - len(pred_neutral[sid])
                              for sid in pred_by_sid)
    contribs_neutral = _contribs(gold_by_sid, pred_neutral, sealed_ids)
    gold_bearing = [sid for sid in sealed_ids if gold_by_sid.get(sid)]
    gold_empty = [sid for sid in sealed_ids if not gold_by_sid.get(sid)]
    # The two kinds of empty-gold unit are NOT the same evidence.
    challenge_only = [sid for sid in gold_empty if chal_by_sid.get(sid)]
    non_skill = [sid for sid in gold_empty if not chal_by_sid.get(sid)]
    # tags = gold tags UNION predicted tags, so a tag the model emits but the
    # gold never carries is scored (precision 0) instead of being invisible to
    # macro-F1 while micro-F1 pays for it.
    tags = sorted({s.label for spans in gold_by_sid.values() for s in spans}
                  | {s.label for spans in pred_by_sid.values() for s in spans})
    per_tag = {}
    for tag in tags:
        m = _micro(_contribs(gold_by_sid, pred_by_sid, sealed_ids, tag=tag))
        m["support_spans"] = m["n_gold"]
        per_tag[tag] = m

    from . import MIN_TEST_SUPPORT_SPANS_PER_TAG
    micro = _micro(contribs)
    lo, hi, n_ci = _bootstrap_ci(contribs)
    micro_neutral = _micro(contribs_neutral)
    nlo, nhi, n_nci = _bootstrap_ci(contribs_neutral)

    def _gap(neutral: float, strict: float):
        """neutral - strict, or None when either side is undefined (NaN)."""
        if neutral != neutral or strict != strict:
            return None
        return _r4(float(neutral) - float(strict))
    defined = [per_tag[t]["f1"] for t in tags if per_tag[t]["f1"] == per_tag[t]["f1"]]
    gold_tag_f1 = [per_tag[t]["f1"] for t in tags
                   if per_tag[t]["n_gold"] and per_tag[t]["f1"] == per_tag[t]["f1"]]
    receipt_id = str(uuid.uuid4())
    result = {
        "scored": True,
        "receipt_id": receipt_id,
        "split_id": split_id,
        "match_rule": MATCH_RULE_TEXT,
        "scoring_unit": "sealed test sentence (every row of test_blind.parquet)",
        "micro": micro,
        "micro_f1_ci95": [_r4(lo), _r4(hi)],
        "micro_f1_ci95_defined_resamples": n_ci,
        "micro_f1_ci95_iters": BOOTSTRAP_ITERS,
        # `micro` is the PRIMARY metric: it is the rule MATCH_RULE_TEXT was
        # registered with, and it is the conservative one. `micro_challenge_neutral`
        # exempts predictions that exactly reproduce a span the adjudicators
        # dropped as AMBIGUOUS/EXCLUDE. Report both; the gap is a property of the
        # annotation, not of the model, and citing either alone hides that.
        "primary_metric": "micro",
        "micro_challenge_neutral": micro_neutral,
        "micro_challenge_neutral_f1_ci95": [_r4(nlo), _r4(nhi)],
        "micro_challenge_neutral_f1_ci95_defined_resamples": n_nci,
        "challenge_cost_precision_delta": _gap(micro_neutral["precision"],
                                               micro["precision"]),
        "challenge_cost_f1_delta": _gap(micro_neutral["f1"], micro["f1"]),
        "challenge_neutral_rule": ("predictions whose (sentence_id, span_start, "
                                   "span_end, tag) exactly matches a row of "
                                   "labels/test_challenge_set.parquet are removed "
                                   "from n_pred: not credited as true positives, "
                                   "not charged as false positives. Overlapping "
                                   "but non-identical spans are still charged"),
        "macro_f1": _r4(float(np.mean(defined))) if defined else None,
        "macro_f1_tags": tags,
        "macro_f1_gold_bearing_tags_only": (_r4(float(np.mean(gold_tag_f1)))
                                            if gold_tag_f1 else None),
        "per_tag": per_tag,
        "support_limited_tags": [t for t in tags
                                 if per_tag[t]["support_spans"] < MIN_TEST_SUPPORT_SPANS_PER_TAG],
        "min_test_support_spans_per_tag": MIN_TEST_SUPPORT_SPANS_PER_TAG,
        "n_test_sentences": len(sealed_ids),
        "n_gold_bearing_sentences": len(gold_bearing),
        "n_gold_empty_sentences": len(gold_empty),
        "n_challenge_only_sentences": len(challenge_only),
        "n_non_skill_sentences": len(non_skill),
        "n_challenge_spans": len(challenge_keys),
        "n_sentences_with_predictions": sum(1 for sid in sealed_ids if pred_by_sid[sid]),
        "n_pred_spans_on_gold_empty_sentences":
            sum(len(pred_by_sid[sid]) for sid in gold_empty),
        "n_pred_spans_on_challenge_only_sentences":
            sum(len(pred_by_sid[sid]) for sid in challenge_only),
        "n_pred_spans_on_non_skill_sentences":
            sum(len(pred_by_sid[sid]) for sid in non_skill),
        "n_pred_spans_matching_challenge_spans": int(n_pred_on_challenge),
        "n_false_positive_spans": micro["n_pred"] - micro["tp"],
        "n_false_negative_spans": micro["n_gold"] - micro["tp"],
        "gold": {k: v for k, v in gold_stats.items()
                 if k not in ("n_rows_outside_sealed_test",
                              "n_sentences_outside_sealed_test")},
        "predictions": pred_stats,
        "aggregates_only": True,
        "reopened": bool(reopened),
    }
    receipt = {
        "receipt_id": receipt_id,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split_id": split_id,
        "config_sha": seal.get("assignment_sha256"),
        "model_config_sha": model_config_sha,
        "git_commit": git_commit,
        "reason": reason,
        "forced_reopen": bool(force_reopen and reopened),
        # null, not NaN: the log is strict JSON. f1 is undefined only when the
        # whole sealed test has neither a gold nor a predicted span.
        "micro_f1": micro["f1"] if micro["f1"] == micro["f1"] else None,
        "micro_f1_challenge_neutral": (micro_neutral["f1"]
                                       if micro_neutral["f1"] == micro_neutral["f1"]
                                       else None),
        "n_test_sentences": len(sealed_ids),
        "n_gold_bearing_sentences": len(gold_bearing),
        "n_pred_spans_matching_challenge_spans": int(n_pred_on_challenge),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(receipt, ensure_ascii=False) + "\n")
    seal["opened_utc"] = receipt["utc"]
    seal.setdefault("receipts", []).append(receipt_id)
    seal["test_open_log_path"] = str(log_path)
    (path / "SEAL.json").write_text(json.dumps(seal, indent=2), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-id", required=True)
    ap.add_argument("--list-receipts", action="store_true")
    args = ap.parse_args()
    path = _split_dir(args.split_id)
    seal = json.loads((path / "SEAL.json").read_text(encoding="utf-8"))
    print(json.dumps(seal, indent=2))


if __name__ == "__main__":
    main()
