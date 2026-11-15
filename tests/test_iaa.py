"""Tests for gold_standard.iaa — metric kernels against hand-computed values,
loaders, span-F1, token labeling, and end-to-end IAA + adjudication."""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_standard.iaa import (  # noqa: E402
    Span, SentAnn, build_adjudication, compute_iaa, cohen_kappa, fleiss_kappa,
    krippendorff_alpha, load_json, load_form_a1_csv, span_level_iaa, _greedy_match,
    _token_labels, _prf, _anns_from_anotasi_df, interpret,
)


# --------------------------------------------------------------- metric kernels
def test_fleiss_perfect_two_categories():
    # two items, each unanimously a different category -> kappa 1.0
    assert fleiss_kappa(np.array([[3, 0], [0, 3]])) == 1.0


def test_fleiss_zero_hand_computed():
    # table engineered so P_bar == P_e -> kappa == 0 (see test docstring math)
    k = fleiss_kappa(np.array([[2, 1], [1, 2], [3, 0]]))
    assert abs(k) < 1e-9


def test_fleiss_zero_variance_is_undefined_not_perfect():
    """Zero marginal variance => kappa is 0/0, i.e. UNDEFINED.

    This previously asserted == 1.0, pinning in a convention that let the report
    print 'Fleiss κ = 1.000 (almost perfect)' for data in which every annotator
    used a single category — a perfect-reliability claim from data that cannot
    demonstrate any. scikit-learn returns nan for the equivalent Cohen case."""
    assert math.isnan(fleiss_kappa(np.array([[3], [3]])))
    assert math.isnan(fleiss_kappa(np.array([[3, 0]] * 40)))
    assert interpret(fleiss_kappa(np.array([[3, 0]] * 40))) == "undefined"


def test_krippendorff_and_cohen_zero_variance_undefined():
    assert math.isnan(krippendorff_alpha([["YA"] * 3] * 40))
    # must NOT override sklearn's nan with a fabricated 1.0
    assert math.isnan(cohen_kappa(["YA"] * 5, ["YA"] * 5))


def test_landis_koch_boundaries_are_inclusive():
    # exact band edges previously landed one band too high
    assert interpret(0.20) == "slight"
    assert interpret(0.40) == "fair"
    assert interpret(0.60) == "moderate"
    assert interpret(0.80) == "substantial"
    assert interpret(0.81) == "almost perfect"


def test_fleiss_unequal_raters_is_nan():
    assert math.isnan(fleiss_kappa(np.array([[3, 0], [2, 0]])))


def test_krippendorff_hand_computed():
    # units [a,a],[a,b],[b,b] -> alpha = 1 - 5*2/18 = 0.4444...
    a = krippendorff_alpha([["a", "a"], ["a", "b"], ["b", "b"]])
    assert abs(a - (1 - 10 / 18)) < 1e-9


def test_krippendorff_perfect():
    assert krippendorff_alpha([["a", "a"], ["b", "b"], ["a", "a"]]) == 1.0


def test_krippendorff_ignores_singletons():
    # a unit with a single rating contributes nothing and must not crash
    a = krippendorff_alpha([["a", "a"], ["b"], ["b", "b"]])
    assert a == 1.0


def test_cohen_kappa_perfect_and_runs():
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    # disagreement produces a finite, <1 value
    k = cohen_kappa(["a", "b", "a", "b"], ["a", "a", "a", "b"])
    assert k < 1.0


# ------------------------------------------------------------------- span level
def test_greedy_match_exact_vs_overlap():
    a = [Span(0, 5, "SH"), Span(10, 15, "TT")]
    b = [Span(0, 5, "SH"), Span(10, 16, "TT")]
    assert _greedy_match(a, b, label_sensitive=True, exact=True) == 1      # only (0,5)
    assert _greedy_match(a, b, label_sensitive=True, exact=False) == 2     # iou(10,15/10,16)=.83


def test_greedy_match_label_sensitivity():
    a = [Span(0, 5, "SH")]
    b = [Span(0, 5, "KN")]
    assert _greedy_match(a, b, label_sensitive=True, exact=True) == 0      # label differs
    assert _greedy_match(a, b, label_sensitive=False, exact=True) == 1     # boundary only


def test_prf_f1_symmetry_and_values():
    pr = _prf(tp=1, n_a=2, n_b=2)
    assert pr["precision"] == 0.5 and pr["recall"] == 0.5 and pr["f1"] == 0.5


def test_span_level_iaa_structure():
    annset = {
        "AV": {"s1": SentAnn("RELEVANT", "YA", [Span(0, 5, "SH")])},
        "AI": {"s1": SentAnn("RELEVANT", "YA", [Span(0, 5, "SH")])},
        "AD": {"s1": SentAnn("RELEVANT", "YA", [Span(0, 6, "SH")])},
    }
    r = span_level_iaa(annset)
    # AV==AI exact; AD overlaps -> exact micro F1 < overlap micro F1
    assert r["modes"]["exact"]["micro"]["f1"] < r["modes"]["overlap_iou50"]["micro"]["f1"]
    assert r["modes"]["overlap_iou50"]["micro"]["f1"] == 1.0


# ------------------------------------------------------------------ token level
def test_token_labels_cover_and_O():
    # "build REST API" -> span over "REST API" tags those two, "build" stays O
    labels = _token_labels("build REST API", [Span(6, 14, "SH")])
    assert labels == ["O", "SH", "SH"]


def test_token_labels_smallest_span_wins_on_overlap():
    # overlapping spans: the shorter (more specific) one wins the shared token
    labels = _token_labels("Python Django framework",
                           [Span(0, 23, "KN"), Span(0, 6, "TT")])
    assert labels[0] == "TT"           # "Python" covered by both -> smallest (TT) wins


# ---------------------------------------------------------------------- loaders
def test_load_json_roundtrip(tmp_path=None):
    tmp_path = Path(tmp_path or tempfile.mkdtemp())
    data = {"_meta": {"annotator": "AV"},
            "s1": {"relevance": "RELEVANT", "has_5tag": "YA",
                   "spans": [{"span_text": "SQL", "start_char": 0, "end_char": 3,
                              "label": "TT", "confidence": 4}]},
            "s2": {"relevance": "NON_RELEVANT", "has_5tag": "TIDAK", "spans": []}}
    p = tmp_path / "pilot_100_AV.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    loaded = load_json(p)
    assert set(loaded) == {"s1", "s2"}           # _meta dropped
    assert loaded["s1"].relevance == "RELEVANT"
    assert loaded["s1"].spans == [Span(0, 3, "TT")]
    assert loaded["s2"].spans == []


def test_load_form_a1_csv(tmp_path=None):
    tmp_path = Path(tmp_path or tempfile.mkdtemp())
    import pandas as pd
    rows = [
        {"sentence_id": "s1", "relevance": "RELEVANT", "has_5tag": "YA",
         "span_text": "SQL", "start_char": 0, "end_char": 3, "label": "TT"},
        {"sentence_id": "s1", "relevance": "RELEVANT", "has_5tag": "YA",
         "span_text": "query", "start_char": 4, "end_char": 9, "label": "SH"},
        {"sentence_id": "s2", "relevance": "NON_RELEVANT", "has_5tag": "TIDAK",
         "span_text": "", "start_char": "", "end_char": "", "label": ""},
    ]
    p = tmp_path / "pilot_100_AV_form_a1.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    loaded = load_form_a1_csv(p)
    assert len(loaded["s1"].spans) == 2
    assert loaded["s2"].spans == []


# --------------------------------------------------------------------- end to end
def _three_coder_set():
    return {
        "AV": {"s1": SentAnn("RELEVANT", "YA", [Span(0, 6, "TT")]),
               "s2": SentAnn("NON_RELEVANT", "TIDAK", []),
               "s3": SentAnn("RELEVANT", "YA", [Span(0, 5, "SH")])},
        "AI": {"s1": SentAnn("RELEVANT", "YA", [Span(0, 6, "TT")]),
               "s2": SentAnn("NON_RELEVANT", "TIDAK", []),
               "s3": SentAnn("UNCERTAIN", "TIDAK", [])},          # disagrees on s3
        "AD": {"s1": SentAnn("RELEVANT", "YA", [Span(0, 6, "TT")]),
               "s2": SentAnn("NON_RELEVANT", "TIDAK", []),
               "s3": SentAnn("RELEVANT", "YA", [Span(0, 5, "SH")])},
    }


def test_compute_iaa_runs_end_to_end():
    annset = _three_coder_set()
    text = {"s1": "Python is used", "s2": "Bachelor degree required",
            "s3": "write code daily"}
    rep = compute_iaa(annset, text)
    assert rep["sentence_level"]["relevance"]["n_complete_cases"] == 3
    # s1/s2 unanimous, s3 2-of-3 RELEVANT -> relevance not perfect but positive
    assert rep["span_level"]["modes"]["exact"]["micro"]["f1"] > 0
    assert rep["token_level"]["n_tokens"] > 0


def test_adjudication_separates_unanimous_and_disputed():
    annset = _three_coder_set()
    text = {"s1": "Python is used", "s2": "Bachelor degree required",
            "s3": "write code daily"}
    adj = build_adjudication(annset, text)
    gold = adj["provisional_gold"]
    assert gold["s1"]["resolution"] == "unanimous"
    assert gold["s2"]["resolution"] == "unanimous"
    assert gold["s3"]["resolution"] in ("majority", "DISPUTE")   # AI dissents
    disputed_ids = {q["sentence_id"] for q in adj["queue"]}
    assert "s3" in disputed_ids and "s1" not in disputed_ids
    # unanimous TT span on s1 is promoted to provisional gold
    assert gold["s1"]["spans"] == [{"start": 0, "end": 6, "label": "TT"}]


# ----------------------------------------------- regression tests for review fixes
def test_greedy_match_is_maximum_cardinality_staircase():
    """The case that actually discriminates a correct cost encoding from a broken one.

    A 2x2 configuration CANNOT detect the defect (it passes either way), which is
    why the original version of this test certified a broken implementation. The
    staircase below is the minimal realisable counterexample: A_i == B_{i+1}, so
    four IDENTICAL pairs exist (IoU 1.0, cost -2 each = -8) while the true
    5-matching A_i-B_i is all IoU 0.5 (cost -1.5 each = -7.5) and the leftover
    pair A[4]/B[0] is disjoint. Under the old -(1+IoU) weighting the optimiser
    preferred the 4-matching and reported tp=4 (F1 0.80) instead of tp=5 (F1 1.0).
    """
    A = [Span(3, 6, "SH"), Span(4, 7, "SH"), Span(5, 8, "SH"),
         Span(6, 9, "SH"), Span(7, 10, "SH")]
    B = [Span(2, 5, "SH"), Span(3, 6, "SH"), Span(4, 7, "SH"),
         Span(5, 8, "SH"), Span(6, 9, "SH")]
    assert _greedy_match(A, B, label_sensitive=True, exact=False) == 5


def test_greedy_match_equals_oracle_on_random_configs():
    """Property test against scipy's maximum-cardinality matching, over dense
    overlapping configurations (the regime where a weighting bug shows up)."""
    import random

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching

    from gold_standard.iaa import _eligible

    def oracle(a, b):
        M = np.array([[_eligible(x, y, True, False) is not None for y in b]
                      for x in a], np.int8)
        if M.size == 0:
            return 0
        return int((maximum_bipartite_matching(csr_matrix(M), perm_type="column") >= 0).sum())

    rng = random.Random(20261115)
    for _ in range(600):
        n, m = rng.randrange(1, 7), rng.randrange(1, 7)
        # narrow widths + unit steps reproduce the staircase regime
        A = [Span(s := rng.randrange(0, 12), s + rng.choice([3, 4, 6]), "SH") for _ in range(n)]
        B = [Span(s := rng.randrange(0, 12), s + rng.choice([3, 4, 6]), "SH") for _ in range(m)]
        assert _greedy_match(A, B, True, False) == oracle(A, B), (
            [(s.start, s.end) for s in A], [(s.start, s.end) for s in B])


def test_prf_both_empty_is_nan_not_one():
    pr = _prf(0, 0, 0)
    assert math.isnan(pr["precision"]) and math.isnan(pr["recall"]) and math.isnan(pr["f1"])


def test_per_tag_f1_is_nan_for_unused_tag():
    annset = {
        "AV": {"s1": SentAnn("RELEVANT", "YA", [Span(0, 6, "TT")])},
        "AI": {"s1": SentAnn("RELEVANT", "YA", [Span(0, 6, "TT")])},
        "AD": {"s1": SentAnn("RELEVANT", "YA", [Span(0, 6, "TT")])},
    }
    per_tag = span_level_iaa(annset)["per_tag_overlap_f1"]
    assert math.isnan(per_tag["WB"]["f1"])          # nobody used WB -> n/a, not 1.0
    assert per_tag["TT"]["f1"] == 1.0


def test_token_label_single_char_glued_to_punctuation():
    # 'C,' is one token [0,2); the stored TT span is [0,1). max-overlap keeps it TT
    assert _token_labels("C, Python", [Span(0, 1, "TT")]) == ["TT", "O"]


def test_adjudication_gates_has_5tag_disagreement():
    annset = {
        "AV": {"s1": SentAnn("RELEVANT", "YA", [])},
        "AI": {"s1": SentAnn("RELEVANT", "YA", [])},
        "AD": {"s1": SentAnn("RELEVANT", "TIDAK", [])},   # dissents on has_5tag only
    }
    adj = build_adjudication(annset, {"s1": "..."})
    assert adj["provisional_gold"]["s1"]["resolution"] != "unanimous"
    assert "s1" in {q["sentence_id"] for q in adj["queue"]}
    assert adj["queue"][0]["has_5tag_agreement"] == "majority"


def test_adjudication_non_relevant_forces_tidak():
    annset = {
        "AV": {"s1": SentAnn("NON_RELEVANT", "YA", [])},   # inconsistent has_5tag
        "AI": {"s1": SentAnn("NON_RELEVANT", "YA", [])},
        "AD": {"s1": SentAnn("RELEVANT", "TIDAK", [])},
    }
    gold = build_adjudication(annset, {"s1": "..."})["provisional_gold"]["s1"]
    assert gold["relevance"] == "NON_RELEVANT" and gold["has_5tag"] == "TIDAK"


def test_adjudication_span_unanimity_requires_exact_boundaries():
    text = {"s1": "x" * 20}
    identical = {c: {"s1": SentAnn("RELEVANT", "YA", [Span(0, 10, "SH")])}
                 for c in ("AV", "AI", "AD")}
    g = build_adjudication(identical, text)["provisional_gold"]["s1"]
    assert g["resolution"] == "unanimous" and g["spans"] == [{"start": 0, "end": 10, "label": "SH"}]

    # same label, overlapping but non-identical boundaries -> BOUND dispute, queued,
    # and NO fabricated gold span wider than any annotator marked.
    boundary = {"AV": {"s1": SentAnn("RELEVANT", "YA", [Span(0, 10, "SH")])},
                "AI": {"s1": SentAnn("RELEVANT", "YA", [Span(0, 10, "SH")])},
                "AD": {"s1": SentAnn("RELEVANT", "YA", [Span(2, 12, "SH")])}}
    adj = build_adjudication(boundary, text)
    gb = adj["provisional_gold"]["s1"]
    assert gb["resolution"] != "unanimous" and gb["spans"] == []
    assert adj["queue"][0]["span_disputes"][0]["type"] == "BOUND"


def test_adjudication_transitive_cluster_not_unanimous():
    # single-link chains AV-AI-AD, but AV vs AD IoU<0.5 -> must not be 'unanimous'
    text = {"s1": "x" * 20}
    annset = {"AV": {"s1": SentAnn("RELEVANT", "YA", [Span(0, 10, "SH")])},
              "AI": {"s1": SentAnn("RELEVANT", "YA", [Span(2, 12, "SH")])},
              "AD": {"s1": SentAnn("RELEVANT", "YA", [Span(4, 14, "SH")])}}
    g = build_adjudication(annset, text)["provisional_gold"]["s1"]
    assert g["resolution"] != "unanimous" and g["spans"] == []


def test_csv_loader_drops_unjudged_filler_rows(tmp_path=None):
    tmp_path = Path(tmp_path or tempfile.mkdtemp())
    import pandas as pd
    rows = [
        {"sentence_id": "s1", "relevance": "RELEVANT", "has_5tag": "YA",
         "span_text": "SQL", "start_char": 0, "end_char": 3, "label": "TT"},
        {"sentence_id": "s2", "relevance": "", "has_5tag": "",      # unjudged filler
         "span_text": "", "start_char": "", "end_char": "", "label": ""},
    ]
    p = tmp_path / "pilot_100_AV_form_a1.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    loaded = load_form_a1_csv(p)
    assert set(loaded) == {"s1"}                     # s2 filler dropped (JSON parity)


def test_xlsx_parse_captures_overflow_and_reports_unresolved():
    import pandas as pd
    cols = ["row_type", "sentence_id", "sentence_text", "relevance", "has_5tag",
            "span_text", "occurrence_no", "start_char", "end_char", "label"]
    rows = [
        ["SENTENCE", "s1", "build REST API with SQL", "RELEVANT", "YA", "", "", "", "", ""],
        ["SPAN", "s1", "", "", "", "REST API", "1", "", "", "SH"],
        ["SPAN", "s1", "", "", "", "SQL", "1", "", "", "TT"],
        ["", "", "", "", "", "build", "1", "", "", "SH"],       # inserted overflow row
        ["SPAN", "s1", "", "", "", "Python", "1", "", "", "TT"],  # not in text -> unresolved
        ["SENTENCE", "s2", "nothing judged here", "", "", "", "", "", "", ""],  # unjudged
    ]
    df = pd.DataFrame(rows, columns=cols)
    anns, problems = _anns_from_anotasi_df(df)
    assert len(anns["s1"].spans) == 3                # REST API, SQL, + inserted 'build'
    assert "s2" not in anns                          # unjudged -> dropped
    assert any(p["span_text"] == "Python" for p in problems)


# ------------------------------- second review round (prior-model remediation)
def test_blank_has_5tag_is_not_counted_as_agreement():
    """A coder who never answered has_5tag must not make the item 'unanimous'."""
    annset = {"AV": {"s1": SentAnn("RELEVANT", "YA", [])},
              "AI": {"s1": SentAnn("RELEVANT", "YA", [])},
              "AD": {"s1": SentAnn("RELEVANT", "", [])}}      # blank, not answered
    adj = build_adjudication(annset, {"s1": "x"})
    assert adj["provisional_gold"]["s1"]["resolution"] != "unanimous"
    assert "s1" in {q["sentence_id"] for q in adj["queue"]}


def test_unanimous_nested_span_survives_an_extra_overlapping_span():
    """A nested/extra span from ONE coder must not destroy a span all three
    endorsed identically (the guideline's own nested SH/TT + OP pattern)."""
    text = {"s1": "x" * 30}
    shared = Span(0, 20, "SH")
    annset = {"AV": {"s1": SentAnn("RELEVANT", "YA", [shared])},
              "AI": {"s1": SentAnn("RELEVANT", "YA", [shared])},
              "AD": {"s1": SentAnn("RELEVANT", "YA", [shared, Span(2, 18, "TT")])}}
    g = build_adjudication(annset, text)["provisional_gold"]["s1"]
    assert {"start": 0, "end": 20, "label": "SH"} in g["spans"]


def test_non_relevant_with_spans_is_flagged_not_silently_fixed():
    annset = {c: {"s1": SentAnn("NON_RELEVANT", "YA", [Span(0, 4, "SH")])}
              for c in ("AV", "AI", "AD")}
    adj = build_adjudication(annset, {"s1": "xxxx"})
    g = adj["provisional_gold"]["s1"]
    assert g["has_5tag"] == "TIDAK" and g["spans"] == []
    assert g["guideline_conflicts"]                      # recorded, not silent
    assert g["resolution"] != "unanimous"                # and surfaced for review
    assert "s1" in {q["sentence_id"] for q in adj["queue"]}


def test_out_of_schema_value_does_not_count_as_a_judgement():
    annset = {"AV": {"s1": SentAnn("RELEVANT", "YA", [])},
              "AI": {"s1": SentAnn("RELEVANT", "YA", [])},
              "AD": {"s1": SentAnn("MAYBE", "YA", [])}}       # not in RELEVANCE
    adj = build_adjudication(annset, {"s1": "x"})
    assert adj["provisional_gold"]["s1"]["resolution"] != "unanimous"


def test_sentence_with_spans_but_no_relevance_is_not_dropped():
    annset = {c: {"s1": SentAnn("", "", [Span(0, 4, "SH")])} for c in ("AV", "AI", "AD")}
    adj = build_adjudication(annset, {"s1": "xxxx"})
    assert "s1" in adj["provisional_gold"]               # previously vanished entirely


def test_macro_f1_survives_one_undefined_pair():
    a = {"s1": SentAnn("RELEVANT", "YA", [Span(0, 5, "SH")])}
    annset = {"AV": dict(a),
              "AI": {"s1": SentAnn("RELEVANT", "TIDAK", [])},
              "AD": {"s1": SentAnn("RELEVANT", "TIDAK", [])}}
    m = span_level_iaa(annset)["modes"]["exact"]
    assert not math.isnan(m["macro_f1"])                # nanmean, not mean
    assert m["n_pairs_defined"] == 2 and m["n_pairs_total"] == 3


def test_token_level_undefined_when_no_spans_anywhere():
    from gold_standard.iaa import token_level_iaa
    annset = {c: {f"s{i}": SentAnn("RELEVANT", "TIDAK", []) for i in range(5)}
              for c in ("AV", "AI", "AD")}
    tl = token_level_iaa(annset, {f"s{i}": "satu dua tiga" for i in range(5)})
    assert tl["no_spans_anywhere"] is True
    assert math.isnan(tl["fleiss_kappa"]) and math.isnan(tl["krippendorff_alpha"])


def test_span_text_does_not_bind_inside_a_longer_word():
    from gold_standard.iaa import _resolve_span
    text = "Experience with REST APIs is required"
    span, ok, reason = _resolve_span(text, "API", 1, "", "", "TT")
    assert not ok, f"'API' must not bind inside 'APIs' (got {span})"


def test_researcher_offsets_rejected_when_they_do_not_match_span_text():
    from gold_standard.iaa import _resolve_span
    text = "membangun REST API dengan Docker"
    # 'Kubernetes' is absent, so the fallback is reached; the supplied offsets
    # point at 'membangun', which is not the typed span -> must be rejected
    # rather than fabricating a span over unrelated characters.
    span, ok, reason = _resolve_span(text, "Kubernetes", 1, 0, 9, "TT")
    assert not ok and "do not contain" in reason
    # and accepted when they DO contain it
    span2, ok2, reason2 = _resolve_span(text, "Kubernetes", 1, 26, 32, "TT")
    assert not ok2            # text[26:32] == 'Docker' != 'Kubernetes'


def test_xlsx_offsets_match_the_app_trim_rule():
    from gold_standard.iaa import _resolve_span
    text = "menguasai Python, Docker"
    span, ok, _ = _resolve_span(text, "Python,", 1, "", "", "TT")
    assert ok and (span.start, span.end) == (10, 16)     # trailing ',' trimmed


def test_inserted_row_with_stray_sentence_text_does_not_steal_spans():
    import pandas as pd
    cols = ["row_type", "sentence_id", "sentence_text", "relevance", "has_5tag",
            "span_text", "occurrence_no", "start_char", "end_char", "label"]
    rows = [
        ["SENTENCE", "s1", "build REST API with SQL", "RELEVANT", "YA", "", "", "", "", ""],
        ["SPAN", "s1", "", "", "", "REST API", "1", "", "", "SH"],
        # autofilled sentence_text on an inserted span row, no sentence_id
        ["", "", "build REST API with SQL", "", "", "SQL", "1", "", "", "TT"],
    ]
    anns, problems = _anns_from_anotasi_df(pd.DataFrame(rows, columns=cols))
    assert len(anns["s1"].spans) == 2        # both spans stay with s1
    assert "" not in anns                    # no bogus empty-id sentence


def test_duplicate_sentence_blocks_merge_instead_of_overwriting():
    import pandas as pd
    cols = ["row_type", "sentence_id", "sentence_text", "relevance", "has_5tag",
            "span_text", "occurrence_no", "start_char", "end_char", "label"]
    rows = [
        ["SENTENCE", "s1", "build REST API with SQL", "RELEVANT", "YA", "", "", "", "", ""],
        ["SPAN", "s1", "", "", "", "REST API", "1", "", "", "SH"],
        ["SENTENCE", "s1", "build REST API with SQL", "RELEVANT", "YA", "", "", "", "", ""],
        ["SPAN", "s1", "", "", "", "SQL", "1", "", "", "TT"],
    ]
    anns, problems = _anns_from_anotasi_df(pd.DataFrame(rows, columns=cols))
    assert len(anns["s1"].spans) == 2        # earlier block's span not discarded
    assert any("merged" in p["reason"] for p in problems)


def test_has_5tag_only_judgement_kept_by_csv_loader(tmp_path=None):
    tmp_path = Path(tmp_path or tempfile.mkdtemp())
    import pandas as pd
    rows = [{"sentence_id": "s1", "relevance": "", "has_5tag": "TIDAK",
             "span_text": "", "start_char": "", "end_char": "", "label": ""}]
    p = tmp_path / "pilot_100_AV_form_a1.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    assert set(load_form_a1_csv(p)) == {"s1"}   # JSON loader keeps it, so must this


def _run_standalone() -> int:
    """Plain-python runner (the repo has no pytest). Pytest-compatible too: under
    pytest these same test_* functions are collected normally."""
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
