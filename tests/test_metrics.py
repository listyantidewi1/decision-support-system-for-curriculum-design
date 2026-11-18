"""Tests for gold_standard.metrics — the shared evaluation engine.

Standalone runner (the repo has no pytest):

    .venv-ml\\Scripts\\python.exe tests\\test_metrics.py

Every expected value below is hand-computed in the comment beside it, so a test
that passes is checking arithmetic against arithmetic done independently, not
the module against itself.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_standard import iaa, metrics, seal                              # noqa: E402
from gold_standard.metrics import (                                        # noqa: E402
    Span, bootstrap_f1_ci, confusion_matrix, contribs, error_taxonomy,
    match_pairs, mcnemar_exact, mcnemar_recall_strict, micro,
    paired_bootstrap_f1, per_tag, wilson_ci)


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def _raises(exc, fn, *a, **k) -> bool:
    try:
        fn(*a, **k)
    except exc:
        return True
    return False


# ------------------------------------------------- one kernel, not two
def test_kernels_are_the_iaa_objects_not_copies():
    """The whole point of the module: no second span-F1 in the repo."""
    assert metrics.prf is iaa.prf is iaa._prf
    assert metrics.greedy_match is iaa.greedy_match is iaa._greedy_match
    assert metrics.percentile_ci is iaa.percentile_ci is iaa._percentile_ci
    assert metrics.Span is iaa.Span
    # and the sealed-test scorer uses those same objects
    assert seal._prf is iaa._prf
    assert seal._greedy_match is iaa._greedy_match
    assert seal._percentile_ci is iaa._percentile_ci


def test_partial_mode_is_iaa_relaxed_rule():
    assert metrics.MATCH_MODES["strict"] == {"label_sensitive": True, "exact": True}
    assert metrics.MATCH_MODES["partial"] == {"label_sensitive": True, "exact": False}
    assert _raises(ValueError, contribs, {}, {}, ["s"], "fuzzy")


def test_match_pairs_cardinality_equals_greedy_match_on_random_configs():
    """match_pairs is the same matcher, returning pairs instead of a count."""
    rng = random.Random(7)
    for _ in range(300):
        def spans():
            out = []
            for _ in range(rng.randint(0, 6)):
                s = rng.randint(0, 30)
                out.append(Span(s, s + rng.randint(1, 8), rng.choice(["SH", "TT", "KN"])))
            return out
        a, b = spans(), spans()
        for ls in (True, False):
            for ex in (True, False):
                pairs = match_pairs(a, b, lambda x, y: iaa._eligible(x, y, ls, ex))
                assert len(pairs) == iaa.greedy_match(a, b, ls, ex), (a, b, ls, ex)
                assert len({i for i, _ in pairs}) == len(pairs)      # one-to-one
                assert len({j for _, j in pairs}) == len(pairs)


# ----------------------------------------------------------------- scoring
REF = {"s1": [Span(0, 5, "SH"), Span(10, 15, "TT")]}
HYP = {"s1": [Span(0, 5, "SH"), Span(10, 16, "TT"), Span(20, 25, "KN")]}


def test_strict_micro_hand_computed():
    # exact+label: only (0,5,SH) matches. tp=1, n_ref=2, n_hyp=3
    # P = 1/3, R = 1/2, F1 = 2*(1/3)*(1/2)/(1/3+1/2) = (1/3)/(5/6) = 0.4
    m = micro(contribs(REF, HYP, ["s1"], "strict"))
    assert (m["tp"], m["n_ref"], m["n_hyp"]) == (1, 2, 3)
    assert _close(m["precision"], 1 / 3) and _close(m["recall"], 0.5)
    assert _close(m["f1"], 0.4)


def test_partial_micro_hand_computed():
    # (10,15) vs (10,16): inter 5, union 6, IoU .833 >= .5 -> matches. tp=2
    # P = 2/3, R = 1, F1 = 2*(2/3)/(5/3) = 0.8
    m = micro(contribs(REF, HYP, ["s1"], "partial"))
    assert m["tp"] == 2 and _close(m["f1"], 0.8)


def test_population_is_taken_literally_so_reference_empty_sentences_count():
    """F02 guard: a false positive on a reference-empty sentence must cost."""
    ref = {"s1": [Span(0, 5, "SH")]}
    hyp = {"s1": [Span(0, 5, "SH")], "s2": [Span(0, 3, "TT")]}
    on_ref_only = micro(contribs(ref, hyp, ["s1"]))
    on_population = micro(contribs(ref, hyp, ["s1", "s2"]))
    assert _close(on_ref_only["precision"], 1.0)
    assert _close(on_population["precision"], 0.5)        # 1 tp of 2 predicted
    assert on_population["n_sentences"] == 2


def test_nothing_on_either_side_is_nan_not_one():
    m = micro(contribs({}, {}, ["s1", "s2"]))
    assert m["f1"] != m["f1"] and m["precision"] != m["precision"]


def test_per_tag_macro_averages_only_defined_tags():
    # SH: ref 1, hyp 1, tp 1 -> F1 1.0
    # TT: ref 1, hyp 1 (different bounds) -> tp 0 -> F1 0.0
    # KN: ref 0, hyp 1 -> P 0, R 0 -> F1 0.0 (defined: one side produced a span)
    # SS, WB: neither side -> NaN, excluded
    # macro = (1.0 + 0.0 + 0.0) / 3 = 0.3333
    out = per_tag(REF, HYP, ["s1"], "strict")
    assert _close(out["by_tag"]["SH"]["f1"], 1.0)
    assert _close(out["by_tag"]["TT"]["f1"], 0.0)
    assert _close(out["by_tag"]["KN"]["f1"], 0.0)
    assert out["n_tags_defined"] == 3
    assert out["tags_undefined"] == ["SS", "WB"]
    assert _close(out["macro_f1"], 1 / 3)


def test_per_tag_macro_is_nan_when_no_tag_is_defined():
    out = per_tag({}, {}, ["s1"])
    assert out["macro_f1"] != out["macro_f1"] and out["n_tags_defined"] == 0


# ---------------------------------------------------------------- Wilson
def test_wilson_known_values():
    z2 = 1.959963984540054 ** 2
    # k=0, n=10: lower 0, upper = z^2 / (n + z^2) = 3.8415 / 13.8415 = 0.27753
    lo, hi = wilson_ci(0, 10)
    assert _close(lo, 0.0) and _close(hi, z2 / (10 + z2), 1e-9)
    assert abs(hi - 0.27753) < 1e-4
    # k=10, n=10: mirror image
    lo, hi = wilson_ci(10, 10)
    assert _close(hi, 1.0) and abs(lo - 0.72247) < 1e-4
    # k=5, n=10: centre .5; half = z*sqrt(.025 + z^2/400)/(1 + z^2/10) = 0.26340
    lo, hi = wilson_ci(5, 10)
    assert abs(lo - 0.23659) < 1e-4 and abs(hi - 0.76341) < 1e-4


def test_wilson_stays_inside_unit_interval_and_contains_the_point():
    for n in (1, 3, 17, 240, 1200):
        for k in range(0, n + 1, max(1, n // 7)):
            lo, hi = wilson_ci(k, n)
            assert 0.0 <= lo <= k / n <= hi <= 1.0, (k, n, lo, hi)


def test_wilson_narrows_with_n_and_rejects_nonsense():
    w100 = wilson_ci(50, 100)
    w1200 = wilson_ci(600, 1200)
    assert (w1200[1] - w1200[0]) < (w100[1] - w100[0])
    lo, hi = wilson_ci(0, 0)
    assert lo != lo and hi != hi                         # NaN, not a made-up interval
    assert _raises(ValueError, wilson_ci, 5, 3)
    assert _raises(ValueError, wilson_ci, -1, 3)
    assert _raises(ValueError, wilson_ci, 1, 3, 1.5)


# --------------------------------------------------------------- McNemar
def test_mcnemar_exact_hand_computed():
    # b=5, c=0: p = 2 * C(5,0)/2^5 = 2/32 = 0.0625
    assert _close(mcnemar_exact(5, 0)["p_value"], 0.0625)
    # b=8, c=2: tail = (C(10,0)+C(10,1)+C(10,2))/1024 = 56/1024; p = 0.109375
    assert _close(mcnemar_exact(8, 2)["p_value"], 0.109375)
    # symmetric in b and c
    assert _close(mcnemar_exact(2, 8)["p_value"], 0.109375)
    # balanced discordance: capped at 1
    assert _close(mcnemar_exact(3, 3)["p_value"], 1.0)
    # nothing discordant: nothing to test
    out = mcnemar_exact(0, 0)
    assert out["p_value"] == 1.0 and out["n_discordant"] == 0
    assert _raises(ValueError, mcnemar_exact, -1, 2)


def test_mcnemar_recall_counts_reference_spans():
    ref = {"s1": [Span(0, 5, "SH"), Span(10, 15, "TT"), Span(20, 25, "KN")],
           "s2": [Span(0, 4, "WB")]}
    a = {"s1": [Span(0, 5, "SH"), Span(10, 15, "TT")], "s2": [Span(0, 4, "WB")]}
    b = {"s1": [Span(0, 5, "SH"), Span(20, 25, "KN"), Span(40, 45, "SS")]}
    # (0,5,SH): both. (10,15,TT): A only -> b. (20,25,KN): B only -> c.
    # (0,4,WB): A only -> b.  B's extra (40,45,SS) is a false positive: invisible.
    out = mcnemar_recall_strict(ref, a, b, ["s1", "s2"])
    assert (out["both"], out["b"], out["c"], out["neither"]) == (1, 2, 1, 0)
    assert out["n_reference_spans"] == 4
    # label must match under the strict rule
    wrong_label = {"s1": [Span(0, 5, "SS")]}
    out = mcnemar_recall_strict(ref, wrong_label, {}, ["s1"])
    assert out["b"] == 0 and out["neither"] == 3


# ------------------------------------------------------- error analysis
def test_confusion_matrix_hand_computed():
    ref = {"s1": [Span(0, 5, "SH"), Span(10, 15, "TT"), Span(20, 25, "KN")]}
    hyp = {"s1": [Span(0, 5, "SS"), Span(10, 15, "TT"), Span(30, 35, "WB")]}
    cm = confusion_matrix(ref, hyp, ["s1"])["matrix"]
    assert cm["SH"]["SS"] == 1                 # same bounds, wrong category
    assert cm["TT"]["TT"] == 1
    assert cm["KN"]["MISSED"] == 1
    assert cm["SPURIOUS"]["WB"] == 1
    total = sum(v for row in cm.values() for v in row.values())
    assert total == 4                           # 2 pairs + 1 missed + 1 spurious


def test_error_taxonomy_classes_and_conservation():
    ref = {"s1": [Span(0, 5, "SH"),        # exact + label           -> correct
                  Span(10, 15, "TT"),      # exact, hyp says KN      -> wrong_category
                  Span(20, 30, "KN"),      # overlap, same label     -> boundary_error
                  Span(40, 50, "SS"),      # overlap, different label-> boundary_and_category
                  Span(60, 65, "WB")]}     # nothing near            -> missed
    hyp = {"s1": [Span(0, 5, "SH"), Span(10, 15, "KN"), Span(22, 30, "KN"),
                  Span(40, 46, "WB"), Span(80, 85, "TT")]}   # last: over_extraction
    out = error_taxonomy(ref, hyp, ["s1"])
    c = out["counts"]
    assert c == {"correct": 1, "wrong_category": 1, "boundary_error": 1,
                 "boundary_and_category": 1, "missed": 1, "over_extraction": 1}
    # every reference span is in exactly one class, and so is every hypothesis span
    assert (c["correct"] + c["wrong_category"] + c["boundary_error"]
            + c["boundary_and_category"] + c["missed"]) == out["n_reference_spans"] == 5
    assert (c["correct"] + c["wrong_category"] + c["boundary_error"]
            + c["boundary_and_category"] + c["over_extraction"]) == out["n_hypothesis_spans"] == 5
    assert "unsupported_extraction" in out["not_computed"]


def test_error_taxonomy_conserves_on_random_configs():
    rng = random.Random(11)
    for _ in range(200):
        def spans():
            return [Span(s, s + rng.randint(1, 9), rng.choice(["SH", "TT", "KN", "SS"]))
                    for s in (rng.randint(0, 40) for _ in range(rng.randint(0, 6)))]
        ref, hyp = {"x": spans()}, {"x": spans()}
        out = error_taxonomy(ref, hyp, ["x"])
        c = out["counts"]
        matched = (c["correct"] + c["wrong_category"] + c["boundary_error"]
                   + c["boundary_and_category"])
        assert matched + c["missed"] == len(ref["x"])
        assert matched + c["over_extraction"] == len(hyp["x"])


# -------------------------------------------------------------- bootstrap
def _toy_contribs(n: int, tp: int, n_ref: int, n_hyp: int) -> dict:
    return {f"s{i:03d}": (tp, n_ref, n_hyp) for i in range(n)}


def test_bootstrap_ci_matches_the_sealed_test_scorer():
    """Same seed, same resampling order -> the identical interval seal publishes."""
    rng = random.Random(3)
    c = {f"s{i:03d}": (t, t + rng.randint(0, 2), t + rng.randint(0, 2))
         for i, t in enumerate(rng.randint(0, 3) for _ in range(80))}
    ours = bootstrap_f1_ci(c, iters=seal.BOOTSTRAP_ITERS, seed=seal.BOOTSTRAP_SEED)
    assert ours == seal._bootstrap_ci(c)
    assert ours[2] == seal.BOOTSTRAP_ITERS


def test_bootstrap_is_deterministic_and_degenerate_cases_are_nan():
    c = _toy_contribs(30, 1, 2, 2)
    assert bootstrap_f1_ci(c, iters=200, seed=5) == bootstrap_f1_ci(c, iters=200, seed=5)
    lo, hi, n = bootstrap_f1_ci({"only": (1, 1, 1)}, iters=200)
    assert lo != lo and n == 0                    # one unit: no interval


def test_cluster_bootstrap_is_wider_when_sentences_are_dependent():
    """20 postings x 5 sentences that all share their posting's outcome. Treating
    the 100 sentences as independent understates the uncertainty."""
    c, cluster_of = {}, {}
    for g in range(20):
        good = g % 2 == 0
        for j in range(5):
            sid = f"g{g:02d}-{j}"
            c[sid] = (1, 1, 1) if good else (0, 1, 1)
            cluster_of[sid] = f"g{g:02d}"
    lo_s, hi_s, _ = bootstrap_f1_ci(c, iters=1000, seed=1)
    lo_c, hi_c, _ = bootstrap_f1_ci(c, iters=1000, seed=1, cluster_of=cluster_of)
    assert (hi_c - lo_c) > 1.5 * (hi_s - lo_s), ((lo_s, hi_s), (lo_c, hi_c))
    del cluster_of["g00-0"]
    assert _raises(KeyError, bootstrap_f1_ci, c, 100, 1, cluster_of)


def test_paired_bootstrap_detects_a_real_difference():
    a = _toy_contribs(60, 1, 1, 1)                # perfect
    b = _toy_contribs(60, 0, 1, 1)                # always wrong
    out = paired_bootstrap_f1(a, b, iters=500, seed=2)
    assert _close(out["delta"], 1.0)
    assert out["ci95_low"] > 0 and out["p_two_sided"] < 0.01
    rev = paired_bootstrap_f1(b, a, iters=500, seed=2)
    assert _close(rev["delta"], -1.0) and rev["ci95_high"] < 0


def test_paired_bootstrap_on_identical_systems_claims_nothing():
    a = _toy_contribs(60, 1, 2, 2)
    out = paired_bootstrap_f1(a, dict(a), iters=500, seed=2)
    assert _close(out["delta"], 0.0)
    assert out["ci95_low"] <= 0.0 <= out["ci95_high"]
    assert _close(out["p_two_sided"], 1.0)


def test_paired_bootstrap_is_paired_noise_in_difficulty_cancels():
    """A beats B by one span on every 4th sentence; sentence difficulty varies a
    lot. Unpaired intervals overlap heavily, the paired delta still excludes 0."""
    rng = random.Random(9)
    a, b = {}, {}
    for i in range(200):
        n_ref = rng.randint(1, 6)
        hit = rng.randint(0, n_ref - 1)
        sid = f"s{i:03d}"
        b[sid] = (hit, n_ref, n_ref)
        a[sid] = (hit + (1 if i % 4 == 0 else 0), n_ref, n_ref)
    out = paired_bootstrap_f1(a, b, iters=1000, seed=4)
    assert out["delta"] > 0 and out["ci95_low"] > 0
    ci_a, ci_b = bootstrap_f1_ci(a, 1000, 4), bootstrap_f1_ci(b, 1000, 4)
    assert ci_b[1] > ci_a[0], "expected the unpaired intervals to overlap"


def test_paired_bootstrap_refuses_different_populations():
    a = _toy_contribs(10, 1, 1, 1)
    b = dict(a)
    b.pop("s000")
    assert _raises(ValueError, paired_bootstrap_f1, a, b)


def test_module_has_no_private_span_f1_of_its_own():
    """Negative control on the design: fail if someone pastes a kernel in here."""
    src = (PROJECT_ROOT / "gold_standard" / "metrics.py").read_text(encoding="utf-8")
    assert "def prf(" not in src and "def _prf(" not in src
    assert "def greedy_match(" not in src and "def _greedy_match(" not in src
    assert "from .iaa import" in src
    assert math.isnan(metrics.prf(0, 0, 0)["f1"])


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
