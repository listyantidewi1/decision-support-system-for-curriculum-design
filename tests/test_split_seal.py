"""Tests for gold_standard.seal — the SCORING DENOMINATOR above all.

Standalone runner (the repo has no pytest): run

    .venv-ml\\Scripts\\python.exe tests\\test_split_seal.py

The centre of this file is finding F02: `score_on_test` used to build its
scoring units from the gold frame, so every sealed test sentence carrying NO
gold span was dropped from the denominator and every false positive on a
non-skill sentence vanished. `_old_score_gold_only` below is the pre-fix
algorithm, kept verbatim as a NEGATIVE CONTROL: each test that asserts the
model is now punished also asserts that the old code scored the same inputs
falsely well, so the regression cannot come back silently.

No test touches DATA/. A sealed split is faked at the level seal.py consumes it
(test_blind.parquet + SEAL.json, whose test_sentence_id_list_sha256 is computed
exactly as gold_standard.split writes it), so these tests need neither the real
pool nor a fixture build.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_standard import seal as seal_mod                                # noqa: E402
from gold_standard.iaa import _greedy_match, _prf                         # noqa: E402

SPLIT_ID = "grouped_random_v1"


# --------------------------------------------------------------- fake sealed split
def _sealed_split(sids: list[str]) -> tuple[Path, Path]:
    """A minimal sealed split directory. Returns (splits_dir, split_dir)."""
    root = Path(tempfile.mkdtemp(prefix="sealtest_"))
    splits = root / "splits"
    d = splits / SPLIT_ID
    d.mkdir(parents=True)
    pd.DataFrame({"sentence_id": sids,
                  "sentence_text": [f"text of {s}" for s in sids]}).to_parquet(
        d / "test_blind.parquet", index=False)
    (d / "SEAL.json").write_text(json.dumps({
        "split_id": SPLIT_ID, "regime_id": "grouped_random",
        "assignment_sha256": "a" * 64,
        # byte-for-byte the expression gold_standard/split.py uses
        "test_sentence_id_list_sha256": hashlib.sha256(
            "\n".join(sorted(sids)).encode("utf-8")).hexdigest(),
        "labels_materialised": False, "opened_utc": None,
    }, indent=2), encoding="utf-8")
    return splits, d


def _gold(rows: list[tuple[str, int, int, str]], verdict: str = "ACCEPT") -> pd.DataFrame:
    return pd.DataFrame([{"sentence_id": s, "span_start": a, "span_end": b,
                          "tag": t, "verdict": verdict} for s, a, b, t in rows],
                        columns=seal_mod.LABEL_COLUMNS)


def _preds(rows: list[tuple[str, int, int, str]]) -> pd.DataFrame:
    return pd.DataFrame([{"sentence_id": s, "span_start": a, "span_end": b, "tag": t}
                         for s, a, b, t in rows], columns=seal_mod.SPAN_COLUMNS)


# ----------------------------------------------------- the pre-fix algorithm
def _old_score_gold_only(gold: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """seal.score_on_test's scoring core BEFORE the F02 fix, verbatim.

    Kept so every "the model is punished now" assertion is paired with proof
    that the old code did not punish it. Do not repair this function.
    """
    def exact_sets(frame):
        out: dict[str, set] = {}
        for r in frame.itertuples(index=False):
            out.setdefault(str(r.sentence_id), set()).add(
                (int(r.span_start), int(r.span_end), str(r.tag)))
        return out

    g_by, p_by = exact_sets(gold), exact_sets(pred)
    contribs = {}
    for sid, g in g_by.items():                     # <-- the defect: gold only
        p = p_by.get(sid, set())
        contribs[sid] = (len(g & p), len(g), len(p))
    tp = sum(c[0] for c in contribs.values())
    n_gold = sum(c[1] for c in contribs.values())
    n_pred = sum(c[2] for c in contribs.values())
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gold if n_gold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "n_gold": n_gold, "n_pred": n_pred, "precision": prec,
            "recall": rec, "f1": f1, "n_test_sentences": len(contribs)}


# ------------------------------------------------------------------ F02 itself
def test_f02_four_sentences_one_gold_three_hallucinations():
    """The adversary's exact reproduction: 4 sealed sentences, 1 gold span, 3
    spurious predictions. Truth is P 0.25 / R 1.0 / F1 0.40 over n=4."""
    sids = ["s1", "s2", "s3", "s4"]
    splits, _ = _sealed_split(sids)
    try:
        gold = _gold([("s1", 0, 5, "SH")])
        preds = _preds([("s1", 0, 5, "SH"), ("s2", 0, 4, "SH"),
                        ("s3", 1, 6, "KN"), ("s4", 2, 7, "SS")])
        seal_mod.materialise_labels(SPLIT_ID, gold, splits_dir=splits)
        res = seal_mod.score_on_test(SPLIT_ID, preds, model_config_sha="m1",
                                     git_commit="c1", reason="F02",
                                     splits_dir=splits)
        assert res["micro"] == {"tp": 1, "n_gold": 1, "n_pred": 4,
                               "precision": 0.25, "recall": 1.0, "f1": 0.4}, res["micro"]
        assert res["n_test_sentences"] == 4
        assert res["n_gold_bearing_sentences"] == 1
        assert res["n_gold_empty_sentences"] == 3
        assert res["n_pred_spans_on_gold_empty_sentences"] == 3
        assert res["n_false_positive_spans"] == 3

        # NEGATIVE CONTROL: the pre-fix code on the same inputs.
        old = _old_score_gold_only(
            pd.read_parquet(splits / SPLIT_ID / "labels" / "test_labels.parquet"), preds)
        assert old["f1"] == 1.0 and old["precision"] == 1.0
        assert old["n_pred"] == 1 and old["n_test_sentences"] == 1
    finally:
        shutil.rmtree(splits.parent, ignore_errors=True)


def test_hallucinating_on_every_gold_empty_sentence_is_punished():
    """A model that is perfect on the gold-bearing sentences and invents one span
    on EVERY gold-empty sentence must lose precision, not score 1.0.

    80 of 100 sealed sentences carry no gold span (the normal case for a 5-tag
    span task with a NON_RELEVANT category, which is why sample.py draws a
    BORDERLINE_SHARE minority at all)."""
    sids = [f"s{i:03d}" for i in range(100)]
    splits, _ = _sealed_split(sids)
    try:
        gold_rows = [(s, 0, 5, "SH") for s in sids[:20]]
        pred_rows = gold_rows[:]                       # all 20 true positives
        pred_rows += [(s, 0, 3, "SH") for s in sids[20:]]   # 80 false positives
        gold = _gold([(s, a, b, t) for s, a, b, t in gold_rows])
        preds = _preds(pred_rows)
        info = seal_mod.materialise_labels(SPLIT_ID, gold, splits_dir=splits)
        assert info["sealed_test_n_sentences"] == 100
        assert info["n_gold_bearing_sentences"] == 20
        assert info["n_gold_empty_sealed_sentences"] == 80

        res = seal_mod.score_on_test(SPLIT_ID, preds, model_config_sha="halluc",
                                     git_commit="c1", reason="F02 punishment",
                                     splits_dir=splits)
        m = res["micro"]
        assert (m["tp"], m["n_gold"], m["n_pred"]) == (20, 20, 100), m
        assert m["precision"] == 0.2 and m["recall"] == 1.0
        assert m["f1"] == round(2 * 0.2 * 1.0 / 1.2, 4) == 0.3333, m
        assert res["n_test_sentences"] == 100
        assert res["n_gold_empty_sentences"] == 80
        assert res["n_pred_spans_on_gold_empty_sentences"] == 80
        # the CI must not reach the inflated value either
        lo, hi = res["micro_f1_ci95"]
        assert lo < m["f1"] < hi < 0.6, res["micro_f1_ci95"]

        # NEGATIVE CONTROL: the same model scored a perfect 1.0 under the old code.
        old = _old_score_gold_only(
            pd.read_parquet(splits / SPLIT_ID / "labels" / "test_labels.parquet"), preds)
        assert old["f1"] == 1.0 and old["n_pred"] == 20
        assert old["n_test_sentences"] == 20, "old code's n was the gold-bearing n"
    finally:
        shutil.rmtree(splits.parent, ignore_errors=True)


def test_n_test_sentences_is_the_sealed_n_and_is_logged():
    sids = [f"s{i:02d}" for i in range(30)]
    splits, d = _sealed_split(sids)
    try:
        gold = _gold([(s, 0, 4, "KN") for s in sids[:3]])
        seal_mod.materialise_labels(SPLIT_ID, gold, splits_dir=splits)
        res = seal_mod.score_on_test(SPLIT_ID, _preds([("s00", 0, 4, "KN")]),
                                     model_config_sha="m1", git_commit="c1",
                                     reason="n check", splits_dir=splits)
        assert res["n_test_sentences"] == 30 and res["n_gold_bearing_sentences"] == 3
        receipt = json.loads((splits / "test_open_log.jsonl").read_text(
            encoding="utf-8").splitlines()[-1])
        assert receipt["n_test_sentences"] == 30
        assert receipt["receipt_id"] == res["receipt_id"]
        seal = json.loads((d / "SEAL.json").read_text(encoding="utf-8"))
        assert seal["sealed_test_n_sentences"] == 30
    finally:
        shutil.rmtree(splits.parent, ignore_errors=True)


# ------------------------------------------------------------------- bootstrap
def test_bootstrap_resamples_sentences_including_the_gold_empty_ones():
    """The resampling unit must be the sealed SENTENCE, not the gold span.

    A span-resampling bootstrap cannot see a sentence with no gold span, so its
    interval would sit at 1.0 for the second contribs dict below. Sentence
    resampling must drag it down to the point estimate."""
    gold_bearing = {f"g{i}": (1, 1, 1) for i in range(20)}
    lo, hi, n = seal_mod._bootstrap_ci(gold_bearing)
    assert (lo, hi) == (1.0, 1.0) and n == seal_mod.BOOTSTRAP_ITERS

    with_fps = dict(gold_bearing)
    with_fps.update({f"e{i}": (0, 0, 1) for i in range(80)})
    lo2, hi2, n2 = seal_mod._bootstrap_ci(with_fps)
    point = _prf(20, 20, 100)["f1"]
    assert n2 == seal_mod.BOOTSTRAP_ITERS
    assert lo2 < point < hi2, (lo2, point, hi2)
    assert hi2 < 0.6, hi2


def test_bootstrap_is_deterministic_and_undefined_below_two_units():
    one = {"s1": (1, 1, 1)}
    lo, hi, n = seal_mod._bootstrap_ci(one)
    assert math.isnan(lo) and math.isnan(hi) and n == 0
    c = {f"s{i}": (i % 2, 1, 1) for i in range(40)}
    assert seal_mod._bootstrap_ci(c) == seal_mod._bootstrap_ci(c)


# --------------------------------------------------- consistency with iaa.py
def test_micro_is_iaa_greedy_match_plus_iaa_prf():
    """seal must not define a second span-F1. Recompute the whole metric with
    iaa._greedy_match / iaa._prf over the same per-sentence span lists and
    require the identical numbers, including on nested predictions where a
    set-intersection count and a maximum matching could disagree."""
    sids = ["s1", "s2", "s3"]
    splits, _ = _sealed_split(sids)
    try:
        gold_rows = [("s1", 0, 5, "SH"), ("s1", 3, 9, "SH"), ("s2", 0, 4, "KN")]
        pred_rows = [("s1", 0, 5, "SH"), ("s1", 3, 9, "KN"), ("s1", 6, 9, "SH"),
                     ("s3", 2, 7, "SS")]
        seal_mod.materialise_labels(SPLIT_ID, _gold(gold_rows), splits_dir=splits)
        res = seal_mod.score_on_test(SPLIT_ID, _preds(pred_rows),
                                     model_config_sha="m1", git_commit="c1",
                                     reason="iaa parity", splits_dir=splits)

        from gold_standard.iaa import Span
        g_by = {s: [] for s in sids}
        p_by = {s: [] for s in sids}
        for s, a, b, t in gold_rows:
            g_by[s].append(Span(a, b, t))
        for s, a, b, t in pred_rows:
            p_by[s].append(Span(a, b, t))
        tp = n_g = n_p = 0
        for s in sids:                       # EVERY sealed sentence, as seal does
            tp += _greedy_match(g_by[s], p_by[s], label_sensitive=True, exact=True)
            n_g += len(g_by[s])
            n_p += len(p_by[s])
        expect = _prf(tp, n_g, n_p)
        # only (0,5,SH) matches: (3,9,SH) vs (3,9,KN) is a label miss and
        # (6,9,SH) is a boundary miss, so exact+type tp is 1, not 2.
        assert res["micro"]["tp"] == expect["tp"] == 1
        assert res["micro"]["n_gold"] == expect["n_a"] == 3
        assert res["micro"]["n_pred"] == expect["n_b"] == 4
        assert res["micro"]["precision"] == 0.25
        assert res["micro"]["f1"] == round(expect["f1"], 4) == 0.2857
    finally:
        shutil.rmtree(splits.parent, ignore_errors=True)


def test_empty_test_is_nan_not_a_flattering_one():
    """iaa's NaN convention is inherited: with no gold span and no prediction
    there is nothing to score, so P/R/F1 are undefined rather than 1.0 (which
    the old code's `tp/n_pred if n_pred else 0.0` would have rendered as 0.0,
    an equally false precise claim)."""
    sids = ["s1", "s2"]
    splits, _ = _sealed_split(sids)
    try:
        info = seal_mod.materialise_labels(
            SPLIT_ID, _gold([("s1", 0, 5, "SH")], verdict="AMBIGUOUS"),
            splits_dir=splits)
        assert info["n_spans"] == 0 and info["n_dropped_ambiguous_or_exclude"] == 1
        res = seal_mod.score_on_test(SPLIT_ID, _preds([]), model_config_sha="m1",
                                     git_commit="c1", reason="nan convention",
                                     splits_dir=splits)
        m = res["micro"]
        assert m["tp"] == 0 and m["n_gold"] == 0 and m["n_pred"] == 0
        assert math.isnan(m["precision"]) and math.isnan(m["recall"])
        assert math.isnan(m["f1"])
        assert res["macro_f1"] is None and res["per_tag"] == {}
        receipt = json.loads((splits / "test_open_log.jsonl").read_text(
            encoding="utf-8").splitlines()[-1])
        assert receipt["micro_f1"] is None, "the log must stay strict JSON"
    finally:
        shutil.rmtree(splits.parent, ignore_errors=True)


def test_a_tag_the_model_invents_is_scored_not_invisible():
    sids = ["s1", "s2"]
    splits, _ = _sealed_split(sids)
    try:
        seal_mod.materialise_labels(SPLIT_ID, _gold([("s1", 0, 5, "SH")]),
                                    splits_dir=splits)
        res = seal_mod.score_on_test(SPLIT_ID,
                                     _preds([("s1", 0, 5, "SH"), ("s2", 0, 5, "WB")]),
                                     model_config_sha="m1", git_commit="c1",
                                     reason="tag union", splits_dir=splits)
        assert set(res["per_tag"]) == {"SH", "WB"}
        assert res["per_tag"]["WB"] == {"tp": 0, "n_gold": 0, "n_pred": 1,
                                       "precision": 0.0, "recall": 0.0, "f1": 0.0,
                                       "support_spans": 0}
        assert res["macro_f1"] == 0.5
        assert res["macro_f1_gold_bearing_tags_only"] == 1.0
    finally:
        shutil.rmtree(splits.parent, ignore_errors=True)


# ------------------------------------------------- the denominator is sealed too
def test_editing_test_blind_after_sealing_is_refused():
    """Positive control for the new guard: if the sealed sentence list could be
    edited, the fix would be undone by deleting the gold-empty rows."""
    sids = [f"s{i}" for i in range(6)]
    splits, d = _sealed_split(sids)
    try:
        seal_mod.materialise_labels(SPLIT_ID, _gold([("s0", 0, 5, "SH")]),
                                    splits_dir=splits)
        blind = pd.read_parquet(d / "test_blind.parquet")
        blind[blind["sentence_id"] == "s0"].to_parquet(d / "test_blind.parquet",
                                                       index=False)
        try:
            seal_mod.score_on_test(SPLIT_ID, _preds([("s0", 0, 5, "SH")]),
                                   model_config_sha="m1", git_commit="c1",
                                   reason="shrunk denominator", splits_dir=splits)
        except seal_mod.SealViolation as exc:
            assert "sealed digest" in str(exc)
        else:
            raise AssertionError("a shrunken sealed sentence list must be refused")
        assert not (splits / "test_open_log.jsonl").exists(), \
            "a refused open must not burn the single open"
    finally:
        shutil.rmtree(splits.parent, ignore_errors=True)


def test_predictions_in_the_wrong_id_space_are_refused_not_scored_zero():
    sids = ["s1", "s2"]
    splits, _ = _sealed_split(sids)
    try:
        seal_mod.materialise_labels(SPLIT_ID, _gold([("s1", 0, 5, "SH")]),
                                    splits_dir=splits)
        try:
            seal_mod.score_on_test(SPLIT_ID, _preds([("train-9", 0, 5, "SH")]),
                                   model_config_sha="m1", git_commit="c1",
                                   reason="wrong ids", splits_dir=splits)
        except seal_mod.SealViolation as exc:
            assert "id-space mismatch" in str(exc)
        else:
            raise AssertionError("predictions keyed outside the sealed test must refuse")
    finally:
        shutil.rmtree(splits.parent, ignore_errors=True)


def test_malformed_prediction_rows_are_refused_not_skipped():
    """Skipping a malformed predicted span would delete a false positive."""
    sids = ["s1", "s2"]
    splits, _ = _sealed_split(sids)
    try:
        seal_mod.materialise_labels(SPLIT_ID, _gold([("s1", 0, 5, "SH")]),
                                    splits_dir=splits)
        for bad, needle in [([("s2", 7, 3, "SH")], "empty/reversed"),
                            ([("s2", 0, 4, "SKILL_HARD")], "five-tag set")]:
            try:
                seal_mod.score_on_test(SPLIT_ID, _preds(bad), model_config_sha="m1",
                                       git_commit="c1", reason="malformed",
                                       splits_dir=splits)
            except seal_mod.SealViolation as exc:
                assert needle in str(exc), str(exc)
            else:
                raise AssertionError(f"malformed prediction ({needle}) must refuse")
    finally:
        shutil.rmtree(splits.parent, ignore_errors=True)


def test_labels_outside_the_sealed_test_are_refused():
    sids = ["s1", "s2"]
    splits, d = _sealed_split(sids)
    try:
        seal_mod.materialise_labels(SPLIT_ID, _gold([("s1", 0, 5, "SH")]),
                                    splits_dir=splits)
        labels = d / "labels" / "test_labels.parquet"
        frame = pd.read_parquet(labels)
        frame.loc[0, "sentence_id"] = "dev-1"
        frame.to_parquet(labels, index=False)
        seal = json.loads((d / "SEAL.json").read_text(encoding="utf-8"))
        seal["test_labels_sha256"] = seal_mod._sha256_file(labels)   # re-seal the file
        (d / "SEAL.json").write_text(json.dumps(seal, indent=2), encoding="utf-8")
        try:
            seal_mod.score_on_test(SPLIT_ID, _preds([("s1", 0, 5, "SH")]),
                                   model_config_sha="m1", git_commit="c1",
                                   reason="alien label", splits_dir=splits)
        except seal_mod.SealViolation as exc:
            assert "subset of the seal" in str(exc)
        else:
            raise AssertionError("gold outside the sealed test must be refused")
    finally:
        shutil.rmtree(splits.parent, ignore_errors=True)


# ------------------------------------------------------------- seal discipline
def test_duplicate_gold_rows_are_collapsed_and_counted():
    splits, _ = _sealed_split(["s1", "s2"])
    try:
        info = seal_mod.materialise_labels(
            SPLIT_ID, _gold([("s1", 0, 5, "SH"), ("s1", 0, 5, "SH")]),
            splits_dir=splits)
        assert info["n_spans"] == 1 and info["n_duplicate_span_rows_dropped"] == 1
    finally:
        shutil.rmtree(splits.parent, ignore_errors=True)


def test_single_open_per_config_still_refuses_and_logs():
    splits, _ = _sealed_split(["s1", "s2", "s3"])
    try:
        seal_mod.materialise_labels(SPLIT_ID, _gold([("s1", 0, 5, "SH")]),
                                    splits_dir=splits)
        preds = _preds([("s1", 0, 5, "SH")])
        first = seal_mod.score_on_test(SPLIT_ID, preds, model_config_sha="m1",
                                       git_commit="c1", reason="first",
                                       splits_dir=splits)
        assert first["scored"] and first["aggregates_only"]
        try:
            seal_mod.score_on_test(SPLIT_ID, preds, model_config_sha="m1",
                                   git_commit="c1", reason="again", splits_dir=splits)
        except seal_mod.SealViolation as exc:
            assert "single open" in str(exc)
        else:
            raise AssertionError("a second open for the same config must be refused")
        forced = seal_mod.score_on_test(SPLIT_ID, preds, model_config_sha="m1",
                                        git_commit="c1", reason="documented reopen",
                                        force_reopen=True, splits_dir=splits)
        assert forced["reopened"] is True
        lines = (splits / "test_open_log.jsonl").read_text(
            encoding="utf-8").strip().splitlines()
        assert len(lines) == 2 and json.loads(lines[1])["forced_reopen"] is True
        # no per-sentence error list may ever leave this function
        assert not any(k for k in forced if "sentence_id" in k or k == "errors")
    finally:
        shutil.rmtree(splits.parent, ignore_errors=True)


def test_void_when_the_label_file_changed_after_sealing():
    splits, d = _sealed_split(["s1", "s2", "s3"])
    try:
        seal_mod.materialise_labels(
            SPLIT_ID, _gold([("s1", 0, 5, "SH"), ("s2", 0, 4, "KN")]),
            splits_dir=splits)
        labels = d / "labels" / "test_labels.parquet"
        pd.read_parquet(labels).iloc[1:].to_parquet(labels, index=False)
        res = seal_mod.score_on_test(SPLIT_ID, _preds([("s1", 0, 5, "SH")]),
                                     model_config_sha="m1", git_commit="c1",
                                     reason="tamper", splits_dir=splits)
        assert res["scored"] is False and res["verdict"].startswith("VOID")
    finally:
        shutil.rmtree(splits.parent, ignore_errors=True)


def test_unmaterialised_labels_cannot_be_scored():
    splits, _ = _sealed_split(["s1", "s2"])
    try:
        try:
            seal_mod.score_on_test(SPLIT_ID, _preds([("s1", 0, 5, "SH")]),
                                   model_config_sha="m1", git_commit="c1",
                                   reason="too early", splits_dir=splits)
        except seal_mod.SealViolation as exc:
            assert "materialise_labels" in str(exc)
        else:
            raise AssertionError("scoring before materialisation must be refused")
    finally:
        shutil.rmtree(splits.parent, ignore_errors=True)


# ------------------------------------------------------- challenge set (B3:413)
# A sealed sentence whose only adjudicated spans were dropped as AMBIGUOUS /
# EXCLUDE still belongs in the scoring population (F02), but its empty gold set
# means "the annotators could not decide", NOT "there is no skill here". Scoring
# the two alike charged a model a false positive for reproducing exactly the span
# the human adjudicators marked undecided, and deflated published precision in
# proportion to the adjudication's own ambiguity rate. These tests pin both
# halves of the repair: the dropped rows are WRITTEN to disk, and the scorer
# publishes a challenge-neutral metric beside the strict one.
_TRAP_GOLD = [("s1", 0, 5, "SH", "ACCEPT"), ("s2", 0, 5, "KN", "ACCEPT"),
              ("s3", 0, 5, "SH", "AMBIGUOUS"), ("s4", 0, 5, "SH", "EXCLUDE")]


def _mixed_gold() -> pd.DataFrame:
    return pd.DataFrame([{"sentence_id": s, "span_start": a, "span_end": b,
                          "tag": t, "verdict": v}
                         for s, a, b, t, v in _TRAP_GOLD],
                        columns=seal_mod.LABEL_COLUMNS)


def test_challenge_set_is_written_to_disk_not_merely_counted():
    """SEAL.json used to CLAIM the dropped rows were 'kept for qualitative
    analysis (B3:413)' while materialise_labels discarded them."""
    splits, d = _sealed_split([f"s{i}" for i in range(1, 11)])
    try:
        info = seal_mod.materialise_labels(SPLIT_ID, _mixed_gold(), splits_dir=splits)
        chal = d / "labels" / "test_challenge_set.parquet"
        assert chal.exists(), "the challenge set is claimed to be kept; keep it"
        assert len(info["test_challenge_set_sha256"]) == 64
        rows = pd.read_parquet(chal)
        assert len(rows) == 2 and info["n_challenge_spans"] == 2
        assert set(rows["verdict"].str.upper()) == {"AMBIGUOUS", "EXCLUDE"}
        assert sorted(rows["sentence_id"]) == ["s3", "s4"]
        # and they are NOT in the strict gold
        strict = pd.read_parquet(d / "labels" / "test_labels.parquet")
        assert sorted(strict["sentence_id"]) == ["s1", "s2"]
    finally:
        shutil.rmtree(splits.parent, ignore_errors=True)


def test_empty_gold_units_partition_into_challenge_only_and_non_skill():
    splits, d = _sealed_split([f"s{i}" for i in range(1, 11)])
    try:
        info = seal_mod.materialise_labels(SPLIT_ID, _mixed_gold(), splits_dir=splits)
        assert info["n_gold_empty_sealed_sentences"] == 8
        assert info["n_challenge_only_sentences"] == 2      # s3, s4
        assert info["n_non_skill_sealed_sentences"] == 6    # s5..s10
        res = seal_mod.score_on_test(SPLIT_ID, _preds([("s1", 0, 5, "SH")]),
                                     model_config_sha="m1", git_commit="c1",
                                     reason="partition", splits_dir=splits)
        assert res["n_gold_empty_sentences"] == 8
        assert res["n_challenge_only_sentences"] == 2
        assert res["n_non_skill_sentences"] == 6
        assert (res["n_challenge_only_sentences"] + res["n_non_skill_sentences"]
                == res["n_gold_empty_sentences"])
    finally:
        shutil.rmtree(splits.parent, ignore_errors=True)


def test_model_reproducing_the_adjudication_is_not_punished():
    """The recorded repro, verbatim: a model that agrees with EVERY adjudicated
    span scored precision 0.5 because two of those spans were the adjudicators'
    own AMBIGUOUS and EXCLUDE items."""
    splits, _ = _sealed_split([f"s{i}" for i in range(1, 11)])
    try:
        seal_mod.materialise_labels(SPLIT_ID, _mixed_gold(), splits_dir=splits)
        preds = _preds([("s1", 0, 5, "SH"), ("s2", 0, 5, "KN"),
                        ("s3", 0, 5, "SH"), ("s4", 0, 5, "SH")])
        res = seal_mod.score_on_test(SPLIT_ID, preds, model_config_sha="m1",
                                     git_commit="c1", reason="trap",
                                     splits_dir=splits)
        # the strict metric is UNCHANGED -- the primary, registered rule did not
        # silently move
        assert res["primary_metric"] == "micro"
        assert res["micro"]["precision"] == 0.5
        assert res["micro"]["n_pred"] == 4 and res["micro"]["tp"] == 2
        # and the challenge-neutral metric shows the model was in fact perfect
        assert res["micro_challenge_neutral"]["precision"] == 1.0
        assert res["micro_challenge_neutral"]["f1"] == 1.0
        assert res["micro_challenge_neutral"]["n_pred"] == 2
        # the ambiguity cost is PUBLISHED, not left for the reader to infer
        assert res["challenge_cost_precision_delta"] == 0.5
        assert res["n_pred_spans_matching_challenge_spans"] == 2
        # F02 is intact: the denominator is still the sealed n
        assert res["n_test_sentences"] == 10
        assert len(res["micro_challenge_neutral_f1_ci95"]) == 2
    finally:
        shutil.rmtree(splits.parent, ignore_errors=True)


def test_challenge_exemption_is_not_a_general_amnesty():
    """NEGATIVE CONTROL. The exemption covers EXACT reproductions of a dropped
    span and nothing else: hallucinations elsewhere, and spans that merely
    overlap a challenge span, are still false positives under both metrics."""
    splits, _ = _sealed_split([f"s{i}" for i in range(1, 11)])
    try:
        seal_mod.materialise_labels(SPLIT_ID, _mixed_gold(), splits_dir=splits)
        bad = _preds([("s1", 0, 5, "SH"),    # correct
                      ("s5", 0, 5, "SH"),    # hallucination on a non-skill unit
                      ("s6", 0, 5, "KN"),    # hallucination on a non-skill unit
                      ("s3", 1, 4, "SH")])   # OVERLAPS the challenge span, inexact
        res = seal_mod.score_on_test(SPLIT_ID, bad, model_config_sha="m2",
                                     git_commit="c1", reason="amnesty control",
                                     splits_dir=splits)
        assert res["n_pred_spans_matching_challenge_spans"] == 0
        assert res["micro"]["n_pred"] == 4
        assert res["micro_challenge_neutral"]["n_pred"] == 4
        assert res["micro"] == res["micro_challenge_neutral"]
        assert res["challenge_cost_precision_delta"] == 0.0
        assert res["n_false_positive_spans"] == 3
    finally:
        shutil.rmtree(splits.parent, ignore_errors=True)


def test_wrong_tag_on_a_challenge_span_is_still_a_false_positive():
    """The exemption is label-sensitive: predicting the challenge span's offsets
    with a DIFFERENT tag is a real error, not an undecided item."""
    splits, _ = _sealed_split([f"s{i}" for i in range(1, 11)])
    try:
        seal_mod.materialise_labels(SPLIT_ID, _mixed_gold(), splits_dir=splits)
        res = seal_mod.score_on_test(SPLIT_ID, _preds([("s3", 0, 5, "KN")]),
                                     model_config_sha="m3", git_commit="c1",
                                     reason="tag control", splits_dir=splits)
        assert res["n_pred_spans_matching_challenge_spans"] == 0
        assert res["micro_challenge_neutral"]["n_pred"] == 1
        assert res["n_false_positive_spans"] == 1
    finally:
        shutil.rmtree(splits.parent, ignore_errors=True)


def test_missing_challenge_file_is_refused_not_scored():
    """Fail closed: scoring without the file would recreate the trap invisibly."""
    splits, d = _sealed_split([f"s{i}" for i in range(1, 11)])
    try:
        seal_mod.materialise_labels(SPLIT_ID, _mixed_gold(), splits_dir=splits)
        (d / "labels" / "test_challenge_set.parquet").unlink()
        try:
            seal_mod.score_on_test(SPLIT_ID, _preds([("s1", 0, 5, "SH")]),
                                   model_config_sha="m1", git_commit="c1",
                                   reason="missing", splits_dir=splits)
        except seal_mod.SealViolation as exc:
            assert "test_challenge_set.parquet does not" in str(exc)
            assert "materialise_labels" in str(exc)
        else:
            raise AssertionError("scoring without the challenge set must be refused")
    finally:
        shutil.rmtree(splits.parent, ignore_errors=True)


def test_tampered_challenge_file_voids_the_score():
    """The exempt set is sealed like the gold: editing it after the fact must not
    be able to buy precision."""
    splits, d = _sealed_split([f"s{i}" for i in range(1, 11)])
    try:
        seal_mod.materialise_labels(SPLIT_ID, _mixed_gold(), splits_dir=splits)
        # forge a challenge set that exempts a hallucination on s5
        pd.DataFrame([{"sentence_id": "s5", "span_start": 0, "span_end": 5,
                       "tag": "SH", "verdict": "AMBIGUOUS"}],
                     columns=seal_mod.LABEL_COLUMNS).to_parquet(
            d / "labels" / "test_challenge_set.parquet", index=False)
        res = seal_mod.score_on_test(SPLIT_ID, _preds([("s5", 0, 5, "SH")]),
                                     model_config_sha="m1", git_commit="c1",
                                     reason="tamper", splits_dir=splits)
        assert res["scored"] is False
        assert "VOID" in res["verdict"] and "test_challenge_set" in res["verdict"]
    finally:
        shutil.rmtree(splits.parent, ignore_errors=True)


def test_span_in_both_gold_and_challenge_set_is_refused():
    """A span cannot be adjudicated resolved AND undecided; crediting and
    exempting the same span would double-count it."""
    splits, d = _sealed_split([f"s{i}" for i in range(1, 11)])
    try:
        seal_mod.materialise_labels(SPLIT_ID, _mixed_gold(), splits_dir=splits)
        chal = d / "labels" / "test_challenge_set.parquet"
        rows = pd.read_parquet(chal)
        rows = pd.concat([rows, pd.DataFrame([
            {"sentence_id": "s1", "span_start": 0, "span_end": 5, "tag": "SH",
             "verdict": "AMBIGUOUS"}])], ignore_index=True)
        rows.to_parquet(chal, index=False)
        seal = json.loads((d / "SEAL.json").read_text(encoding="utf-8"))
        seal["test_challenge_set_sha256"] = seal_mod._sha256_file(chal)
        (d / "SEAL.json").write_text(json.dumps(seal, indent=2), encoding="utf-8")
        try:
            seal_mod.score_on_test(SPLIT_ID, _preds([("s1", 0, 5, "SH")]),
                                   model_config_sha="m1", git_commit="c1",
                                   reason="contradiction", splits_dir=splits)
        except seal_mod.SealViolation as exc:
            assert "BOTH the strict gold and the challenge set" in str(exc)
        else:
            raise AssertionError("a span in gold AND challenge must be refused")
    finally:
        shutil.rmtree(splits.parent, ignore_errors=True)


def test_clean_adjudication_needs_no_challenge_file_and_both_metrics_agree():
    """When nothing was dropped, the two metrics must be identical -- the repair
    must not perturb the ordinary case."""
    splits, d = _sealed_split([f"s{i}" for i in range(1, 11)])
    try:
        info = seal_mod.materialise_labels(
            SPLIT_ID, _gold([("s1", 0, 5, "SH"), ("s2", 0, 5, "KN")]),
            splits_dir=splits)
        assert info["n_dropped_ambiguous_or_exclude"] == 0
        assert info["n_challenge_spans"] == 0
        assert info["n_challenge_only_sentences"] == 0
        assert info["n_non_skill_sealed_sentences"] == 8
        res = seal_mod.score_on_test(SPLIT_ID, _preds([("s1", 0, 5, "SH"),
                                                       ("s7", 0, 5, "SH")]),
                                     model_config_sha="m1", git_commit="c1",
                                     reason="clean", splits_dir=splits)
        assert res["micro"] == res["micro_challenge_neutral"]
        assert res["challenge_cost_precision_delta"] == 0.0
        assert res["n_pred_spans_matching_challenge_spans"] == 0
        assert res["micro"]["precision"] == 0.5   # the s7 hallucination still costs
    finally:
        shutil.rmtree(splits.parent, ignore_errors=True)


def test_receipt_logs_both_f1s():
    splits, _ = _sealed_split([f"s{i}" for i in range(1, 11)])
    try:
        seal_mod.materialise_labels(SPLIT_ID, _mixed_gold(), splits_dir=splits)
        preds = _preds([("s1", 0, 5, "SH"), ("s2", 0, 5, "KN"),
                        ("s3", 0, 5, "SH"), ("s4", 0, 5, "SH")])
        seal_mod.score_on_test(SPLIT_ID, preds, model_config_sha="m1",
                               git_commit="c1", reason="receipt",
                               splits_dir=splits)
        log = [json.loads(ln) for ln in
               (splits / "test_open_log.jsonl").read_text(encoding="utf-8").splitlines()
               if ln.strip()]
        assert len(log) == 1
        assert log[0]["micro_f1"] == 0.6667
        assert log[0]["micro_f1_challenge_neutral"] == 1.0
        assert log[0]["n_pred_spans_matching_challenge_spans"] == 2
    finally:
        shutil.rmtree(splits.parent, ignore_errors=True)


def test_old_conflated_rule_is_the_negative_control():
    """Pin the pre-fix behaviour so the conflation cannot return silently: with
    the challenge spans NOT exempt, the model that matched every adjudicated span
    scores 0.5 precision. That number is what the repair exists to qualify."""
    splits, _ = _sealed_split([f"s{i}" for i in range(1, 11)])
    try:
        seal_mod.materialise_labels(SPLIT_ID, _mixed_gold(), splits_dir=splits)
        preds = _preds([("s1", 0, 5, "SH"), ("s2", 0, 5, "KN"),
                        ("s3", 0, 5, "SH"), ("s4", 0, 5, "SH")])
        res = seal_mod.score_on_test(SPLIT_ID, preds, model_config_sha="m1",
                                     git_commit="c1", reason="control",
                                     splits_dir=splits)
        strict_p = res["micro"]["precision"]
        neutral_p = res["micro_challenge_neutral"]["precision"]
        assert strict_p == 0.5 and neutral_p == 1.0, (
            "if these are equal the challenge set is no longer being read")
        assert neutral_p > strict_p
    finally:
        shutil.rmtree(splits.parent, ignore_errors=True)


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
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
