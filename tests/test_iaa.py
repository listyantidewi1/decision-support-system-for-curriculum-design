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
    _token_labels, _prf, _anns_from_anotasi_df,
)


# --------------------------------------------------------------- metric kernels
def test_fleiss_perfect_two_categories():
    # two items, each unanimously a different category -> kappa 1.0
    assert fleiss_kappa(np.array([[3, 0], [0, 3]])) == 1.0


def test_fleiss_zero_hand_computed():
    # table engineered so P_bar == P_e -> kappa == 0 (see test docstring math)
    k = fleiss_kappa(np.array([[2, 1], [1, 2], [3, 0]]))
    assert abs(k) < 1e-9


def test_fleiss_single_category_is_guarded():
    # everyone picks the one category -> P_e == 1; perfect -> 1.0 (not div-by-zero)
    assert fleiss_kappa(np.array([[3], [3]])) == 1.0


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
def test_greedy_match_is_maximum_cardinality():
    # nested/overlapping spans: highest-IoU-first greedy would match only 1;
    # maximum-cardinality matching finds 2 (a0-b1 and a1-b0).
    a = [Span(0, 10, "SH"), Span(0, 8, "SH")]
    b = [Span(0, 9, "SH"), Span(4, 10, "SH")]
    assert _greedy_match(a, b, label_sensitive=True, exact=False) == 2


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
    anns, unresolved = _anns_from_anotasi_df(df)
    assert len(anns["s1"].spans) == 3                # REST API, SQL, + inserted 'build'
    assert "s2" not in anns                          # unjudged -> dropped
    assert ("s1", "Python") in unresolved


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
