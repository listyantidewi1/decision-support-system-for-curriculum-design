"""Tests for the steps that follow annotation: the go/no-go check, the
calibration report, adjudication and gold finalisation, and the stand-in
annotators used to rehearse them.

Standalone runner (the repo has no pytest):

    .venv-ml\\Scripts\\python.exe tests\\test_after_annotation.py

Everything here runs on a four-sentence annotation set built in the test. No
file under DATA/ is read or written.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_standard import adjudicate as adj                                # noqa: E402
from gold_standard import calibration, gate, seal                          # noqa: E402
from gold_standard.iaa import SentAnn, Span                                # noqa: E402

TEXT = {
    "t-1": "Develop REST APIs using Python and Docker.",
    "t-2": "Good communication skills and Python.",
    "t-3": "We are a fast-growing company.",
    "t-4": "Design database schemas daily.",
}


def _sp(sid: str, text: str, tag: str, nth: int = 0) -> Span:
    i = -1
    for _ in range(nth + 1):
        i = TEXT[sid].index(text, i + 1)
    return Span(i, i + len(text), tag)


def _annset() -> dict:
    """t-1: 'Develop REST APIs' SH unanimous; Python TT unanimous; Docker TT by AV, AI only.
    t-2: boundary dispute (communication / communication skills, 2 vs 1); Python label
         dispute (TT, TT, KN).
    t-3: relevance 2 NON_RELEVANT vs 1 RELEVANT, no spans.
    t-4: fully unanimous."""
    def a(rel, tag, spans):
        return SentAnn(rel, tag, spans)
    dev, py, dock = _sp("t-1", "Develop REST APIs", "SH"), _sp("t-1", "Python", "TT"), \
        _sp("t-1", "Docker", "TT")
    com, coms = _sp("t-2", "communication", "SS"), _sp("t-2", "communication skills", "SS")
    design = _sp("t-4", "Design database schemas", "SH")
    return {
        "AV": {"t-1": a("RELEVANT", "YA", [dev, py, dock]),
               "t-2": a("RELEVANT", "YA", [com, _sp("t-2", "Python", "TT")]),
               "t-3": a("NON_RELEVANT", "TIDAK", []),
               "t-4": a("RELEVANT", "YA", [design])},
        "AI": {"t-1": a("RELEVANT", "YA", [dev, py, dock]),
               "t-2": a("RELEVANT", "YA", [com, _sp("t-2", "Python", "TT")]),
               "t-3": a("NON_RELEVANT", "TIDAK", []),
               "t-4": a("RELEVANT", "YA", [design])},
        "AD": {"t-1": a("RELEVANT", "YA", [dev, py]),
               "t-2": a("RELEVANT", "YA", [coms, _sp("t-2", "Python", "KN")]),
               "t-3": a("RELEVANT", "TIDAK", []),
               "t-4": a("RELEVANT", "YA", [design])},
    }


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    return None


def _fill(path: Path, sentences: dict | None = None, spans: dict | None = None,
          extra_rows: list | None = None) -> None:
    """Write decisions into a worksheet. `spans`: dispute kind keyword or id -> dict."""
    from openpyxl import load_workbook
    wb = load_workbook(path)
    ws = wb["sentences"]
    head = [c.value for c in ws[1]]
    for row in ws.iter_rows(min_row=2):
        sid = row[0].value
        for col, val in (sentences or {}).get(sid, {}).items():
            row[head.index(col)].value = val
    sp = wb["spans"]
    head = [c.value for c in sp[1]]
    for row in sp.iter_rows(min_row=2):
        key = f"{row[head.index('sentence_id')].value}|{row[head.index('kind')].value}"
        for col, val in (spans or {}).get(key, {}).items():
            row[head.index(col)].value = val
    for r in extra_rows or []:
        sp.append([r.get(h, "") for h in head])
    wb.save(path)


KEY_DOCKER = "t-1|not everyone marked it"
KEY_COMM = "t-2|same tag, different boundaries"
KEY_PY = "t-2|same span, different tag"
ALL_SENT = {"t-3": {"FINAL relevance": "NON_RELEVANT", "FINAL five-tag": "TIDAK"}}
ALL_SPANS = {KEY_DOCKER: {"DECISION": "GOLD"},                       # take the suggestion
             KEY_COMM: {"DECISION": "GOLD", "final text": "communication",
                        "final tag": "SS", "reason": "BOUND"},
             KEY_PY: {"DECISION": "AMBIGUOUS", "reason": "LABEL"}}


def _worksheet(td: Path, annset=None) -> Path:
    path = td / "w.xlsx"
    adj.make_worksheet(annset or _annset(), TEXT, path, "testsplit")
    return path


# ------------------------------------------------------------------ the gate
def _report(rel=0.8, tag=0.8, tok=0.7, n=100, complete=100, annotators=3, **prov):
    return {"split": "pilot_100", "annotators": ["AV", "AI", "AD"][:annotators],
            "sentence_level": {
                "relevance": {"pabak": rel, "fleiss_kappa": 0.5, "n_complete_cases": complete,
                              "n_sentences_any": n},
                "has_5tag": {"pabak": tag, "fleiss_kappa": 0.5, "n_complete_cases": complete,
                             "n_sentences_any": n}},
            "token_level": {"fleiss_kappa": tok},
            "span_level": {"modes": {"exact": {"micro": {"f1": 0.5}},
                                     "overlap_iou50": {"micro": {"f1": 0.7}}}},
            "provenance": {"ids_verified_against_sample": True, "input_quarantined": None,
                           **prov}}


def test_gate_verdicts_follow_the_bands():
    assert gate.evaluate(_report())["verdict"] == gate.GO
    assert gate.evaluate(_report(rel=0.60, tag=0.60, tok=0.60))["verdict"] == gate.GO
    assert gate.evaluate(_report(tok=0.59))["verdict"] == gate.RECALIBRATE
    assert gate.evaluate(_report(rel=0.40))["verdict"] == gate.RECALIBRATE
    assert gate.evaluate(_report(tag=0.39))["verdict"] == gate.STOP
    # one low measure outweighs two good ones
    assert gate.evaluate(_report(rel=0.95, tag=0.95, tok=0.10))["verdict"] == gate.STOP


def test_gate_is_not_ready_when_the_pilot_is_incomplete():
    r = gate.evaluate(_report(complete=90), n_expected=100)
    assert r["verdict"] == gate.NOT_READY and r["problems"]
    assert gate.evaluate(_report(complete=95), n_expected=100)["verdict"] == gate.GO
    assert gate.evaluate(_report(annotators=2))["verdict"] == gate.NOT_READY


def test_gate_treats_an_undefined_figure_as_not_good_enough_to_go():
    r = gate.evaluate(_report(tok=float("nan")))
    assert r["verdict"] == gate.RECALIBRATE
    assert [c["band"] for c in r["checks"]][-1] == "undefined"
    assert "n/a" in gate.to_text(r)


def test_gate_flags_synthetic_input_and_reports_span_f1_only_as_context():
    r = gate.evaluate(_report(input_quarantined="x/_synthetic"))
    assert r["input_is_synthetic"] is True and "rehearsal" in gate.to_text(r)
    assert gate.evaluate(_report())["input_is_synthetic"] is False
    assert r["context_not_gating"] == {"span_exact_f1": 0.5, "span_overlap_f1": 0.7}
    # span F1 must not move the verdict
    low_f1 = _report()
    low_f1["span_level"]["modes"]["exact"]["micro"]["f1"] = 0.01
    assert gate.evaluate(low_f1)["verdict"] == gate.GO


def test_gate_thresholds_match_the_written_rule():
    """The code and docs/experiments/PREREG_pilot_gate.md must say the same thing."""
    doc = (PROJECT_ROOT / "docs" / "experiments" / "PREREG_pilot_gate.md").read_text("utf-8")
    assert f"{gate.GO_AT:.2f} or higher" in doc
    assert f"below {gate.STOP_BELOW:.2f}" in doc
    assert f"fewer than {gate.MIN_COMPLETE_SHARE:.0%}" in doc
    for verdict in (gate.GO, gate.RECALIBRATE, gate.STOP, gate.NOT_READY):
        assert verdict in doc
    # the status in the code may only say "registered" once the document does
    draft = "NOT YET APPROVED" in doc
    assert ("PROPOSED" in gate.RULE_STATUS) == draft, (
        "the rule's status in gate.py and in the pre-registration document disagree")


# --------------------------------------------------------------- calibration
def _queue_report():
    return {"split": "testsplit", "annotators": ["AV", "AI", "AD"], "guideline": "G",
            "created_utc": "T", "sentence_level": {}, "span_level": {}, "token_level": {},
            "provenance": {"ids_verified_against_sample": True, "input_quarantined": "q"},
            "adjudication_queue": [
                {"sentence_id": "t-2", "sentence_text": TEXT["t-2"],
                 "relevance_agreement": "unanimous", "has_5tag_agreement": "unanimous",
                 "n_span_disputes": 2, "guideline_conflicts": [],
                 "span_disputes": [
                     {"type": "BOUND", "coders": ["AD", "AI", "AV"], "labels": {"SS": 3},
                      "spans": [{"coder": "AV", "start": 5, "end": 18, "label": "SS"},
                                {"coder": "AI", "start": 5, "end": 18, "label": "SS"},
                                {"coder": "AD", "start": 5, "end": 25, "label": "SS"}]},
                     {"type": "LABEL", "coders": ["AD", "AI", "AV"],
                      "labels": {"TT": 2, "KN": 1},
                      "spans": [{"coder": "AV", "start": 30, "end": 36, "label": "TT"},
                                {"coder": "AI", "start": 30, "end": 36, "label": "TT"},
                                {"coder": "AD", "start": 30, "end": 36, "label": "KN"}]}]},
                {"sentence_id": "t-1", "sentence_text": TEXT["t-1"],
                 "relevance_agreement": "majority", "has_5tag_agreement": "unanimous",
                 "n_span_disputes": 1, "guideline_conflicts": [],
                 "span_disputes": [
                     {"type": "PRESENCE", "coders": ["AV"], "labels": {"TT": 1},
                      "spans": [{"coder": "AV", "start": 35, "end": 41, "label": "TT"}]}]}]}


def test_bound_pattern_names_how_boundaries_differ():
    s = lambda a, b: {"start": a, "end": b}                                  # noqa: E731
    assert calibration.bound_pattern([s(0, 5), s(0, 9)]) == "same start, different end"
    assert calibration.bound_pattern([s(0, 9), s(4, 9)]) == "same end, different start"
    assert calibration.bound_pattern([s(0, 9), s(2, 7)]) == "one span inside another"
    assert calibration.bound_pattern([s(0, 5), s(3, 9)]) == "shifted or split"


def test_calibration_counts_kinds_confusions_and_who_marked_alone():
    raw = {"AV": {"t-1": {"relevance": "RELEVANT", "has_5tag": "YA", "flag": "AL",
                          "spans": [{"label": "TT", "confidence": 4, "flag": "AB"},
                                    {"label": "SH", "confidence": 2}]}},
           "AI": {"t-1": {"relevance": "RELEVANT", "has_5tag": "YA", "spans": []}},
           "AD": {"_meta": {}, "t-1": {"relevance": "NON_RELEVANT", "has_5tag": "TIDAK",
                                       "spans": []}}}
    a = calibration.analyse(_queue_report(), raw)
    assert a["dispute_types"] == {"BOUND": 1, "LABEL": 1, "PRESENCE": 1}
    assert a["tag_confusion"] == {"KN / TT": 1}
    assert a["bound_patterns"] == {"same start, different end": 1}
    av = a["per_annotator"]["AV"]
    assert av["marked_alone"] == {"TT": 1} and av["n_spans"] == 2
    assert av["mean_confidence"] == 3.0 and av["flags"] == {"AL": 1, "AB": 1}
    assert a["per_annotator"]["AD"]["n_sentences"] == 1          # _meta is not a sentence
    assert any("KN / TT" in x for x in a["agenda"])
    assert any("Relevance was not unanimous" in x for x in a["agenda"])


def test_calibration_page_is_bannered_when_synthetic_and_escapes_text():
    rep = _queue_report()
    rep["adjudication_queue"][0]["sentence_text"] = "<script>alert(1)</script>"
    page = calibration.build(rep, {}, {"t-2": "<script>alert(1)</script>"})
    assert "REHEARSAL ON SYNTHETIC" in page
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page
    rep["provenance"]["input_quarantined"] = None
    assert "REHEARSAL ON SYNTHETIC" not in calibration.build(rep, {}, TEXT)


# -------------------------------------------------------------- adjudication
def test_worksheet_lists_only_what_is_disputed():
    with tempfile.TemporaryDirectory() as td:
        from openpyxl import load_workbook
        path = _worksheet(Path(td))
        wb = load_workbook(path)
        sent = list(wb["sentences"].iter_rows(values_only=True))
        spans = list(wb["spans"].iter_rows(values_only=True))
        assert [r[0] for r in sent[1:]] == ["t-3"]               # only the relevance split
        assert sent[1][5] == "NON_RELEVANT"                      # the 2-of-3 suggestion
        kinds = sorted(f"{r[1]}|{r[3]}" for r in spans[1:])
        assert kinds == sorted([KEY_DOCKER, KEY_COMM, KEY_PY])
        by = {f"{r[1]}|{r[3]}": r for r in spans[1:]}
        assert by[KEY_DOCKER][7:9] == ("Docker", "TT")           # AV and AI agreed exactly
        assert by[KEY_COMM][7:9] == ("communication", "SS")
        assert by[KEY_PY][7] == "Python" and by[KEY_PY][8] == "TT"
        assert all(not r[9] for r in spans[1:])                  # DECISION starts empty
        assert adj.fingerprint(_annset()) in str(list(wb["read me"].iter_rows(values_only=True)))


def test_finalise_builds_the_gold_from_agreement_plus_decisions():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        path = _worksheet(td)
        _fill(path, ALL_SENT, ALL_SPANS)
        m = adj.finalise(_annset(), TEXT, path, td / "gold", "1.0", "testsplit")
        import pandas as pd
        spans = pd.read_parquet(td / "gold" / "gold_spans.parquet")
        sents = pd.read_parquet(td / "gold" / "gold_sentences.parquet")
        gold = {(r.sentence_id, TEXT[r.sentence_id][r.span_start:r.span_end], r.tag, r.source)
                for r in spans[spans.verdict == "GOLD"].itertuples()}
        assert gold == {("t-1", "Develop REST APIs", "SH", "unanimous"),
                        ("t-1", "Python", "TT", "unanimous"),
                        ("t-1", "Docker", "TT", "suggestion accepted"),
                        ("t-2", "communication", "SS", "adjudicated"),
                        ("t-4", "Design database schemas", "SH", "unanimous")}
        amb = spans[spans.verdict == "AMBIGUOUS"]
        assert sorted(amb.tag) == ["KN", "TT"]                   # both variants kept
        assert dict(zip(sents.sentence_id, sents.relevance))["t-3"] == "NON_RELEVANT"
        assert m["n_spans_by_verdict"] == {"GOLD": 5, "AMBIGUOUS": 2}
        assert m["n_decisions_taken_from_suggestions"] == 1 and m["accept_suggestions"] is False
        assert m["n_disputes"] == 3 and m["gold_version"] == "1.0"
        # the table is what the sealed-test scorer consumes
        assert all(c in spans.columns for c in seal.LABEL_COLUMNS)
        assert set(spans.verdict) - {"GOLD"} <= set(seal.DROP_VERDICTS)


def test_nothing_is_written_while_a_row_is_unresolved():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        path = _worksheet(td)
        _fill(path, ALL_SENT, {KEY_COMM: ALL_SPANS[KEY_COMM]})   # two rows left blank
        e = _raises(adj.AdjudicationError, adj.finalise, _annset(), TEXT, path, td / "gold",
                    "1.0", "testsplit")
        assert e is not None and "DECISION" in str(e) and str(e).count("spans row") == 2
        assert not (td / "gold").exists()
        # blank sentence decisions are caught too
        path2 = td / "w2.xlsx"
        adj.make_worksheet(_annset(), TEXT, path2, "testsplit")
        _fill(path2, {}, ALL_SPANS)
        e = _raises(adj.AdjudicationError, adj.finalise, _annset(), TEXT, path2, td / "g2",
                    "1.0", "testsplit")
        assert e is not None and "FINAL relevance" in str(e)


def test_accepting_suggestions_is_explicit_and_recorded():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        path = _worksheet(td)
        _fill(path, {}, {KEY_PY: {"DECISION": "EXCLUDE"}})        # everything else blank
        m = adj.finalise(_annset(), TEXT, path, td / "gold", "1.0", "testsplit",
                         accept_suggestions=True)
        assert m["accept_suggestions"] is True
        # t-3 relevance + five-tag, Docker, communication
        assert m["n_decisions_taken_from_suggestions"] == 4
        assert m["n_spans_by_verdict"] == {"GOLD": 5, "EXCLUDE": 2}


def test_final_text_must_be_in_the_sentence_and_unambiguous():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for n, (final, expect) in enumerate([
                ({"final text": "communications", "final tag": "SS"}, "is not in the sentence"),
                ({"final text": "communication", "final tag": "SKILL"}, "final tag"),
                ({"final text": "communication", "final tag": "SS", "final start": "0"},
                 "at position 0"),
                ({"final text": "communication", "final tag": "SS", "reason": "TYPO"},
                 "reason")]):
            path = td / f"w{n}.xlsx"
            adj.make_worksheet(_annset(), TEXT, path, "testsplit")
            spans = dict(ALL_SPANS)
            spans[KEY_COMM] = {"DECISION": "GOLD", **final}
            _fill(path, ALL_SENT, spans)
            e = _raises(adj.AdjudicationError, adj.finalise, _annset(), TEXT, path,
                        td / f"g{n}", "1.0", "testsplit")
            assert e is not None and expect in str(e), (final, str(e))
    assert adj._locate("Python and Python", "Python", "", None).startswith("'Python' occurs 2")
    assert adj._locate("Python and Python", "Python", "", (9, 17)) == (11, 17)
    assert adj._locate("Python and Python", "Python", "11", None) == (11, 17)


def test_guideline_rules_are_enforced_on_the_result():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # a NON_RELEVANT sentence cannot keep a gold span (guideline 3.2)
        path = _worksheet(td)
        _fill(path, ALL_SENT, ALL_SPANS, extra_rows=[
            {"sentence_id": "t-3", "kind": "added by the adjudicators", "DECISION": "GOLD",
             "final text": "company", "final tag": "KN"}])
        e = _raises(adj.AdjudicationError, adj.finalise, _annset(), TEXT, path, td / "g1",
                    "1.0", "testsplit")
        assert e is not None and "NON_RELEVANT but has a gold span" in str(e)
        # two overlapping gold spans with the same tag (guideline 6.9)
        path = td / "w2.xlsx"
        adj.make_worksheet(_annset(), TEXT, path, "testsplit")
        _fill(path, ALL_SENT, ALL_SPANS, extra_rows=[
            {"sentence_id": "t-2", "kind": "added by the adjudicators", "DECISION": "GOLD",
             "final text": "communication skills", "final tag": "SS"}])
        e = _raises(adj.AdjudicationError, adj.finalise, _annset(), TEXT, path, td / "g2",
                    "1.0", "testsplit")
        assert e is not None and "overlap" in str(e)


def test_added_row_can_create_a_second_gold_span():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        path = _worksheet(td)
        _fill(path, ALL_SENT, ALL_SPANS, extra_rows=[
            {"sentence_id": "t-2", "kind": "added by the adjudicators", "DECISION": "GOLD",
             "final text": "Good", "final tag": "WB"}])
        m = adj.finalise(_annset(), TEXT, path, td / "gold", "1.0", "testsplit")
        assert m["n_spans_by_verdict"]["GOLD"] == 6
        assert m["n_gold_spans_by_tag"]["WB"] == 1


def test_stale_worksheet_and_existing_version_are_refused():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        path = _worksheet(td)
        _fill(path, ALL_SENT, ALL_SPANS)
        changed = _annset()
        changed["AD"]["t-4"] = SentAnn("RELEVANT", "YA", [])     # an annotation changed
        e = _raises(adj.AdjudicationError, adj.finalise, changed, TEXT, path, td / "gold",
                    "1.0", "testsplit")
        assert e is not None and "different annotation files" in str(e)
        adj.finalise(_annset(), TEXT, path, td / "gold", "1.0", "testsplit")
        e = _raises(adj.AdjudicationError, adj.finalise, _annset(), TEXT, path, td / "gold",
                    "1.0", "testsplit")
        assert e is not None and "already exists" in str(e)


def test_unanimous_spans_survive_a_non_relevant_majority():
    """The provisional gold empties the span list when the majority says
    NON_RELEVANT; if the adjudicators overrule that, the agreed spans must still be
    there. They are recomputed from the annotations, not read from that list."""
    s = _annset()
    for c in ("AV", "AI"):
        s[c]["t-4"] = SentAnn("NON_RELEVANT", "YA", s[c]["t-4"].spans)
    assert adj.unanimous_spans(s, "t-4") == [(0, 23, "SH")]
    _d, a = adj.disputes_of(s, TEXT)
    assert a["gold"]["t-4"]["spans"] == []                       # the provisional list
    assert a["gold"]["t-4"]["guideline_conflicts"]


def test_synthetic_annotations_cannot_be_finalised_into_the_evidence_tree():
    gs = adj.GS_DIR
    assert _raises(SystemExit, adj._refuse_evidence_tree, "x/_synthetic", gs / "gold" / "v1")
    assert _raises(SystemExit, adj._refuse_evidence_tree, "marker", gs / "iaa" / "w.xlsx")
    assert _raises(SystemExit, adj._refuse_evidence_tree, "marker",
                   gs / "_synthetic" / "rehearsal" / "gold") is None
    assert _raises(SystemExit, adj._refuse_evidence_tree, None, gs / "gold" / "v1") is None
    with tempfile.TemporaryDirectory() as td:
        assert _raises(SystemExit, adj._refuse_evidence_tree, "marker", Path(td) / "x") is None


# --------------------------------------------------- protecting drawn samples
def test_a_sample_people_have_annotated_cannot_be_redrawn_silently():
    """Drawing the main sample re-draws the pilot first. If the sentence pool was
    rebuilt in between, that would replace the pilot under the annotators' feet."""
    import pandas as pd
    from gold_standard import sample as sample_mod
    with tempfile.TemporaryDirectory() as td:
        pq = Path(td) / "pilot_100.parquet"
        same = pd.DataFrame({"sentence_id": ["a#1", "b#2", "c#3"]})
        other = pd.DataFrame({"sentence_id": ["a#1", "b#2", "z#9"]})
        assert sample_mod.existing_draw_blocks(pq, same, "pilot_100", set()) == "write"
        same.to_parquet(pq, index=False)
        assert sample_mod.existing_draw_blocks(pq, same, "pilot_100", set()) == "keep"
        e = _raises(SystemExit, sample_mod.existing_draw_blocks, pq, other, "pilot_100", set())
        assert e is not None and "refusing to overwrite" in str(e) and "--redraw" in str(e)
        # a different ORDER is a different draw too: sample_order is part of the sample
        reordered = pd.DataFrame({"sentence_id": ["c#3", "b#2", "a#1"]})
        assert _raises(SystemExit, sample_mod.existing_draw_blocks, pq, reordered,
                       "pilot_100", set()) is not None
        # only an explicit redraw may replace it
        assert sample_mod.existing_draw_blocks(pq, other, "pilot_100", {"pilot_100"}) == "write"
        assert _raises(SystemExit, sample_mod.existing_draw_blocks, pq, other, "pilot_100",
                       {"main_1200"}) is not None


# ------------------------------------------------------- stand-in annotators
def test_stand_in_annotators_are_marked_and_confined_to_a_synthetic_folder():
    from labeling import rehearsal
    models = ["fam-a/m", "fam-b/m", "fam-c/m"]
    sents = [{"sentence_id": k, "sentence_text": v} for k, v in TEXT.items()]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        run_dir = td / "run"
        run_dir.mkdir()
        (run_dir / "summary.json").write_text(json.dumps({
            "request_sha256": "abc", "run_config": {"models": models},
            "prompt": {"prompt_version": "v0"}}), encoding="utf-8")
        lines = []
        for m in models:
            for s in sents:
                spans = [{"text": "Python", "label": "TT"}] if "Python" in s["sentence_text"] \
                    else []
                raw = "not json" if (m == models[2] and s["sentence_id"] == "t-1") \
                    else json.dumps({"spans": spans})
                lines.append(json.dumps({"model": m, "sentence_id": s["sentence_id"],
                                         "request_sha256": "abc", "ok": True, "raw": raw}))
        (run_dir / "raw.jsonl").write_text("\n".join(lines), encoding="utf-8")

        assert _raises(ValueError, rehearsal.write, run_dir, "testsplit", sents, td / "plain")
        out = rehearsal.write(run_dir, "testsplit", sents, td / "_synthetic" / "ann")
        assert (out / rehearsal.MARKER).exists()
        files = sorted(p.name for p in out.glob("*.json"))
        assert files == ["testsplit_AD.json", "testsplit_AI.json", "testsplit_AV.json"]
        av = json.loads((out / "testsplit_AV.json").read_text("utf-8"))
        ad = json.loads((out / "testsplit_AD.json").read_text("utf-8"))
        assert av["_meta"]["SIMULATED"] is True and av["_meta"]["model"] == models[0]
        assert av["t-1"]["relevance"] == "RELEVANT" and av["t-1"]["has_5tag"] == "YA"
        assert av["t-1"]["spans"][0]["span_text"] == "Python"
        assert av["t-3"]["has_5tag"] == "TIDAK" and av["t-3"]["spans"] == []
        assert av["t-3"]["relevance"] in ("RELEVANT", "NON_RELEVANT")
        assert ad["t-1"]["spans"] == [] and "invalid" in ad["t-1"]["note"]
        # deterministic: the seeded coin gives the same answer every time
        again = rehearsal.simulate(run_dir, sents)
        assert again["AV"]["t-3"]["relevance"] == av["t-3"]["relevance"]
        assert math.isfinite(len(json.dumps(again)))


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
