"""Tests for labeling/ — the deterministic half of the multi-LLM ensemble.

Standalone runner (the repo has no pytest):

    .venv-ml\\Scripts\\python.exe tests\\test_labeling.py

No model is called anywhere in this suite: every "model output" is a literal
written here. The rules under test are quoted from proposal §3.4.3 in
labeling/consensus.py; each test names the clause it pins.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_standard import metrics                                          # noqa: E402
from gold_standard.iaa import _trim_offsets                                # noqa: E402
from labeling import NO_VOTE, NONE                                         # noqa: E402
from labeling.align import align, span_iou                                 # noqa: E402
from labeling.consensus import (                                           # noqa: E402
    ABSTAIN, ACCEPTED, BOUNDARY_DISAGREEMENT, CONFLICT, MAJORITY, NONE_MAJORITY,
    UNANIMOUS, ConsensusConfig, accepted_spans_by_sid, decide_sentence, summarise)
from labeling.schema import (INVALID, UNSUPPORTED, VALID, VerifiedSpan,    # noqa: E402
                             normalise_label)
from labeling.verify import (VerifyPolicy, summarise_verification,         # noqa: E402
                             verify_output)

M = ("model-a", "model-b", "model-c")
CFG = ConsensusConfig(models=M)

S_K8S = "Configure Kubernetes clusters."
S_TEST = "Must have experience in automated testing, Jest, and Python."
S_TWICE = "Python developers who teach Python."


def _raises(exc, fn, *a, **k) -> bool:
    try:
        fn(*a, **k)
    except exc:
        return True
    return False


def _span(sentence: str, text: str, label: str, occurrence: int | None = None) -> dict:
    d = {"text": text, "label": label}
    if occurrence is not None:
        d["occurrence"] = occurrence
    return d


def _vo(model: str, sentence: str, spans, policy=None, sid: str = "s1"):
    return verify_output(model, sid, sentence, json.dumps({"spans": spans}), policy)


def _codes(out) -> list[str]:
    return [r.code for r in out.rejections]


# =================================================================== verifier
def test_valid_offsets_and_text_are_accepted():
    i = S_TEST.index("automated testing")
    out = _vo("m", S_TEST, [{"text": "automated testing", "label": "SH",
                              "start": i, "end": i + 17}])
    assert out.status == VALID and not out.rejections
    s = out.spans[0]
    assert (s.start, s.end, s.label, s.text) == (i, i + 17, "SH", "automated testing")
    assert s.anchor == "offsets_and_text"


def test_text_only_is_anchored_when_unique():
    out = _vo("m", S_TEST, [_span(S_TEST, "Jest", "TT")])
    s = out.spans[0]
    assert S_TEST[s.start:s.end] == "Jest" and s.anchor == "text_unique"


def test_ambiguous_text_is_rejected_not_guessed():
    """No guessing between occurrences."""
    out = _vo("m", S_TWICE, [_span(S_TWICE, "Python", "TT")])
    assert not out.spans and _codes(out) == ["UNSUPPORTED_AMBIGUOUS_TEXT"]
    ok = _vo("m", S_TWICE, [_span(S_TWICE, "Python", "TT", occurrence=2)])
    assert ok.spans[0].start == S_TWICE.rindex("Python")
    assert ok.spans[0].anchor == "text_occurrence"
    bad = _vo("m", S_TWICE, [_span(S_TWICE, "Python", "TT", occurrence=3)])
    assert _codes(bad) == ["UNSUPPORTED_AMBIGUOUS_TEXT"]


def test_hallucinated_text_is_unsupported():
    """'keberadaan span pada teks sumber'."""
    out = _vo("m", S_TEST, [_span(S_TEST, "Kubernetes", "TT")])
    assert not out.spans and out.status == VALID
    r = out.rejections[0]
    assert (r.code, r.category) == ("UNSUPPORTED_TEXT_NOT_FOUND", UNSUPPORTED)


def test_case_is_not_forgiven():
    out = _vo("m", S_TEST, [_span(S_TEST, "jest", "TT")])
    assert _codes(out) == ["UNSUPPORTED_TEXT_NOT_FOUND"]


def test_offset_mismatch_is_unsupported_by_default_and_repairable_only_on_request():
    """'kesesuaian posisi karakter' — and no silent repair."""
    item = {"text": "Jest", "label": "TT", "start": 0, "end": 4}      # S_TEST[0:4] == "Must"
    strict = _vo("m", S_TEST, [item])
    assert _codes(strict) == ["UNSUPPORTED_OFFSET_MISMATCH"] and not strict.repairs
    lenient = _vo("m", S_TEST, [item], VerifyPolicy(reanchor_on_offset_mismatch=True))
    assert lenient.spans[0].text == "Jest"
    assert lenient.spans[0].anchor == "reanchored_by_text"
    assert lenient.repairs == ["reanchored_by_text"]


def test_bad_offsets_are_invalid():
    n = len(S_TEST)
    for start, end in [(-1, 4), (5, 5), (7, 3), (0, n + 1), (1.0, 4), (True, 4), ("0", 4)]:
        out = _vo("m", S_TEST, [{"label": "TT", "start": start, "end": end}])
        assert _codes(out) == ["INVALID_OFFSETS"], (start, end, _codes(out))
    half = _vo("m", S_TEST, [{"label": "TT", "start": 3}])
    assert _codes(half) == ["INVALID_OFFSETS"]


def test_invalid_json_and_schema_invalidate_the_whole_output():
    """'validitas struktur JSON'."""
    for raw in ["not json", '{"spans": [', "", "[1, 2"]:
        out = verify_output("m", "s1", S_TEST, raw)
        assert out.status == INVALID and _codes(out) == ["INVALID_JSON"], raw
        assert out.can_vote_none is False
    for raw in ["[]", '{"span": []}', '{"spans": {}}', '"x"', "null", "3"]:
        out = verify_output("m", "s1", S_TEST, raw)
        assert out.status == INVALID and _codes(out) == ["INVALID_SCHEMA"], raw


def test_code_fence_is_invalid_json_unless_stripping_is_switched_on():
    raw = '```json\n{"spans": [{"text": "Jest", "label": "TT"}]}\n```'
    strict = verify_output("m", "s1", S_TEST, raw)
    assert strict.status == INVALID and _codes(strict) == ["INVALID_JSON"]
    lenient = verify_output("m", "s1", S_TEST, raw, VerifyPolicy(strip_code_fence=True))
    assert lenient.status == VALID and lenient.spans[0].text == "Jest"
    assert lenient.repairs == ["stripped_code_fence"]


def test_already_parsed_output_is_accepted():
    out = verify_output("m", "s1", S_TEST, {"spans": [{"text": "Jest", "label": "TT"}]})
    assert out.status == VALID and out.spans[0].text == "Jest"


def test_label_vocabulary_is_closed():
    """'penggunaan label yang diperbolehkan'."""
    assert normalise_label("SH") == "SH"
    assert normalise_label("skill_hard") == "SH"
    assert normalise_label("skill-hard") == "SH"          # the proposal's spelling
    assert normalise_label(" Tool Technology ") == "TT"
    assert normalise_label("work-behavior") == "WB"
    for bad in ["SKILL", "HARD_SKILL", "O", "", None, 3, "NONE", "SH,TT"]:
        assert normalise_label(bad) is None, bad
    out = _vo("m", S_TEST, [_span(S_TEST, "Jest", "TECHNOLOGY")])
    assert _codes(out) == ["INVALID_LABEL"] and out.rejections[0].category == INVALID


def test_malformed_items_are_logged_not_fatal():
    out = _vo("m", S_TEST, ["Jest", None, {"label": "TT"}, {"text": 5, "label": "TT"},
                            _span(S_TEST, "Jest", "TT")])
    assert [s.text for s in out.spans] == ["Jest"]
    assert _codes(out) == ["INVALID_ITEM"] * 4
    assert out.status == VALID and out.can_vote_none is False


def test_edges_are_trimmed_with_the_annotator_ui_rule():
    """A model is not charged a boundary error the UI would have removed."""
    i = S_TEST.index("automated testing")
    out = _vo("m", S_TEST, [{"label": "SH", "start": i - 1, "end": i + 18}])  # " ...testing,"
    s = out.spans[0]
    assert s.text == "automated testing" and "TRIMMED" in s.flags
    assert (s.start, s.end) == _trim_offsets(S_TEST, i - 1, i + 18)
    only_punct = _vo("m", S_TEST, [{"label": "SH", "start": i + 17, "end": i + 19}])
    assert _codes(only_punct) == ["INVALID_EMPTY_SPAN"]


def test_human_app_field_names_are_understood():
    i = S_TEST.index("Jest")
    out = _vo("m", S_TEST, [{"span_text": "Jest", "label": "TT",
                              "start_char": i, "end_char": i + 4}])
    assert out.spans[0].text == "Jest"


def test_exact_duplicate_is_noted_and_harmless():
    out = _vo("m", S_TEST, [_span(S_TEST, "Jest", "TT"), _span(S_TEST, "Jest", "TT")])
    assert len(out.spans) == 1 and _codes(out) == ["DUPLICATE_SPAN"]
    assert out.can_vote_none is True


def test_one_span_with_two_labels_is_rejected_entirely():
    out = _vo("m", S_TEST, [_span(S_TEST, "Jest", "TT"), _span(S_TEST, "Jest", "KN")])
    assert not out.spans
    assert _codes(out) == ["INVALID_CONFLICTING_LABELS"] * 2


def test_guideline_rule_6_9_nested_spans_with_different_function_are_kept():
    """The validated guideline's own example must survive the verifier."""
    out = _vo("m", S_K8S, [_span(S_K8S, "Configure Kubernetes clusters", "SH"),
                           _span(S_K8S, "Kubernetes", "TT")])
    assert [(s.text, s.label) for s in out.spans] == [
        ("Configure Kubernetes clusters", "SH"), ("Kubernetes", "TT")]
    assert all("OVERLAP_POTENTIAL" in s.flags for s in out.spans)
    assert not out.rejections


def test_same_label_overlap_is_rejected_and_forbid_all_rejects_every_overlap():
    same = _vo("m", S_K8S, [_span(S_K8S, "Configure Kubernetes clusters", "SH"),
                            _span(S_K8S, "Configure Kubernetes", "SH")])
    assert not same.spans and _codes(same) == ["INVALID_SAME_LABEL_OVERLAP"] * 2
    strict = _vo("m", S_K8S, [_span(S_K8S, "Configure Kubernetes clusters", "SH"),
                              _span(S_K8S, "Kubernetes", "TT")],
                 VerifyPolicy(overlap="forbid_all"))
    assert not strict.spans and _codes(strict) == ["INVALID_OVERLAP"] * 2
    assert _raises(ValueError, VerifyPolicy, overlap="whatever")


def test_empty_answer_is_a_valid_none():
    out = _vo("m", S_TEST, [])
    assert out.status == VALID and not out.spans and out.can_vote_none is True


def test_verifier_never_raises_and_every_kept_span_is_in_the_source():
    """Fuzz: whatever a model emits, the verifier returns a verdict, and a kept
    span's text is always exactly the source slice at its offsets."""
    rng = random.Random(20261118)
    junk = [None, 1, -3, 2.5, True, "", "Jest", "x" * 50, [], {}, [1], {"a": 1}]
    labels = ["SH", "TT", "KN", "nope", None, 7, "skill-soft"]
    for _ in range(600):
        items = []
        for _ in range(rng.randint(0, 5)):
            kind = rng.random()
            if kind < 0.25:
                items.append(rng.choice(junk))
                continue
            a, b = rng.randint(-2, len(S_TEST) + 2), rng.randint(-2, len(S_TEST) + 2)
            item = {"label": rng.choice(labels)}
            if rng.random() < 0.7:
                item["start"], item["end"] = a, b
            if rng.random() < 0.6:
                item["text"] = rng.choice([S_TEST[max(0, a):max(0, b)], "Jest", "Python",
                                           "ghost", rng.choice(junk)])
            items.append(item)
        raw = rng.choice([json.dumps({"spans": items}), json.dumps(items), "{bad",
                          {"spans": items}, rng.choice(junk)])
        for policy in (VerifyPolicy(), VerifyPolicy(overlap="forbid_all",
                                                    reanchor_on_offset_mismatch=True,
                                                    strip_code_fence=True)):
            out = verify_output("m", "s1", S_TEST, raw, policy)
            assert out.status in (VALID, INVALID)
            for s in out.spans:
                assert 0 <= s.start < s.end <= len(S_TEST)
                assert S_TEST[s.start:s.end] == s.text
                assert s.label in ("SH", "SS", "KN", "TT", "WB")
            if out.status == INVALID:
                assert not out.spans


def test_verification_summary_rates_and_denominators():
    outs = [
        _vo("a", S_TEST, [_span(S_TEST, "Jest", "TT"), _span(S_TEST, "ghost", "TT")]),
        verify_output("a", "s2", S_TEST, "not json"),
        _vo("b", S_TEST, [_span(S_TEST, "Jest", "TT")]),
    ]
    rep = summarise_verification(outs)
    assert rep["a"]["json_valid"]["k"] == 1 and rep["a"]["json_valid"]["n"] == 2
    assert rep["a"]["span_unsupported"]["k"] == 1 and rep["a"]["span_unsupported"]["n"] == 2
    assert rep["b"]["json_valid"]["rate"] == 1.0
    lo, hi = rep["a"]["json_valid"]["wilson95"]
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0


# ================================================================== alignment
def _vs(start: int, end: int, label: str = "SH") -> VerifiedSpan:
    return VerifiedSpan(start, end, label, "x" * (end - start), "test")


def _out(model: str, spans, status: str = VALID, sid: str = "s1"):
    from labeling.schema import VerifiedOutput
    return VerifiedOutput(model=model, sentence_id=sid, status=status, spans=list(spans))


def test_exact_tier_groups_identical_boundaries_whatever_the_label():
    cands = align([_out("a", [_vs(0, 5, "SH")]), _out("b", [_vs(0, 5, "TT")]),
                   _out("c", [_vs(20, 25, "KN")])], "x" * 40)
    assert [sorted(c.members) for c in cands] == [["a", "b"], ["c"]]
    assert cands[0].exact


def test_relaxed_tier_merges_at_the_threshold_and_not_below():
    # (0,10) vs (0,5): inter 5, union 10, IoU exactly 0.5 -> merged at 0.5
    at = align([_out("a", [_vs(0, 10)]), _out("b", [_vs(0, 5)])], "x" * 40, 0.5)
    assert len(at) == 1 and not at[0].exact
    # (0,10) vs (0,4): IoU 0.4 -> separate
    below = align([_out("a", [_vs(0, 10)]), _out("b", [_vs(0, 4)])], "x" * 40, 0.5)
    assert len(below) == 2
    # threshold 1.0 means exact only
    exact_only = align([_out("a", [_vs(0, 10)]), _out("b", [_vs(0, 9)])], "x" * 40, 1.0)
    assert len(exact_only) == 2


def test_complete_linkage_does_not_chain():
    """A~B and B~C clear 0.5 but A~C is 0.25: that is not three models agreeing."""
    a, b, c = _vs(0, 10), _vs(3, 13), _vs(6, 16)
    assert span_iou(a, b) > 0.5 and span_iou(b, c) > 0.5 and span_iou(a, c) == 0.25
    cands = align([_out("a", [a]), _out("b", [b]), _out("c", [c])], "x" * 40, 0.5)
    assert sorted(len(x.members) for x in cands) == [1, 2]
    for cand in cands:
        assert cand.min_pairwise_iou("x" * 40, "char") >= 0.5


def test_a_candidate_never_holds_two_spans_from_one_model():
    """Nested spans from one model are two claims, even when both overlap a
    third model's span enough to merge."""
    cands = align([_out("a", [_vs(0, 10, "SH"), _vs(0, 8, "TT")]),
                   _out("b", [_vs(0, 9, "SH")])], "x" * 40, 0.5)
    for c in cands:
        assert len(c.members) == len(set(c.members))
    assert sum(len(c.members) for c in cands) == 3
    assert len(cands) == 2


def test_alignment_does_not_depend_on_input_order():
    rng = random.Random(5)
    for _ in range(100):
        outs = [_out(m, {(s, s + rng.randint(2, 9)): None
                         for s in (rng.randint(0, 25) for _ in range(rng.randint(0, 4)))})
                for m in M]
        outs = [_out(o.model, [_vs(s, e) for (s, e) in o.spans]) for o in outs]
        base = [c._key() for c in align(outs, "x" * 40, 0.5)]
        shuffled = outs[:]
        rng.shuffle(shuffled)
        assert [c._key() for c in align(shuffled, "x" * 40, 0.5)] == base


def test_token_unit_differs_from_char_unit():
    s = "requires automated testing skills"
    a = VerifiedSpan(9, 26, "SH", "automated testing", "test")
    b = VerifiedSpan(19, 26, "SH", "testing", "test")
    assert abs(span_iou(a, b, s, "char") - 7 / 17) < 1e-9          # 0.41
    assert span_iou(a, b, s, "token") == 0.5                        # 1 of 2 tokens
    outs = [_out("a", [a]), _out("b", [b])]
    assert len(align(outs, s, 0.5, "char")) == 2
    assert len(align(outs, s, 0.5, "token")) == 1
    assert _raises(ValueError, align, outs, s, 0.5, "word")
    assert _raises(ValueError, align, outs, s, 0.0)


def test_invalid_outputs_contribute_no_spans():
    cands = align([_out("a", [_vs(0, 5)]), _out("b", [_vs(0, 5)], status=INVALID)], "x" * 9)
    assert [sorted(c.members) for c in cands] == [["a"]]


# ================================================================== consensus
def _decide(sentence: str, per_model: dict, cfg: ConsensusConfig = CFG):
    """per_model: model -> list of raw span dicts, or a raw string, or absent."""
    outs = []
    for m, v in per_model.items():
        raw = v if isinstance(v, str) else json.dumps({"spans": v})
        outs.append(verify_output(m, "s1", sentence, raw, cfg.verify_policy))
    return decide_sentence("s1", sentence, outs, cfg)


JEST = _span(S_TEST, "Jest", "TT")


def test_config_requires_three_distinct_models_and_computes_two_thirds():
    assert _raises(ValueError, ConsensusConfig, models=("a", "b"))
    assert _raises(ValueError, ConsensusConfig, models=("a", "a", "b"))
    assert _raises(ValueError, ConsensusConfig, models=M, iou_threshold=0)
    # ceil(2n/3)
    want = {3: 2, 4: 3, 5: 4, 6: 4, 7: 5, 9: 6}
    for n, need in want.items():
        cfg = ConsensusConfig(models=tuple(f"m{i}" for i in range(n)))
        assert cfg.majority_votes == need, (n, cfg.majority_votes)
        assert cfg.majority_votes / n >= 2 / 3 > (cfg.majority_votes - 1) / n


def test_unanimous_same_label_and_boundaries_is_accepted():
    """'seluruh LLM memberikan label dan batas span yang sama' -> auto-accept."""
    r = _decide(S_TEST, {m: [JEST] for m in M})
    c = r.candidates[0]
    assert (c.status, c.disposition) == (UNANIMOUS, ACCEPTED)
    assert (c.label, c.text) == ("TT", "Jest") and S_TEST[c.start:c.end] == "Jest"
    assert c.consensus_ratio == 1.0 and c.disagreement_score == 0.0
    assert c.boundary == "exact_all"


def test_majority_is_not_accepted_until_the_rule_is_validated_on_dev():
    """'hanya apabila aturan mayoritas tersebut sebelumnya menunjukkan tingkat
    ketepatan yang memadai pada development gold standard'."""
    votes = {"model-a": [JEST], "model-b": [JEST], "model-c": []}
    c = _decide(S_TEST, votes).candidates[0]
    assert (c.status, c.disposition) == (MAJORITY, ABSTAIN)
    assert c.reason == "majority_rule_not_yet_validated_on_dev"
    assert c.votes == {"model-a": "TT", "model-b": "TT", "model-c": NONE}
    assert abs(c.consensus_ratio - 2 / 3) < 1e-12
    # entropy of (2/3, 1/3) over ln 3 = 0.63651 / 1.09861 = 0.57938
    assert abs(c.disagreement_score - 0.57938) < 1e-4
    on = ConsensusConfig(models=M, accept_majority=True)
    c2 = _decide(S_TEST, votes, on).candidates[0]
    assert (c2.status, c2.disposition) == (MAJORITY, ACCEPTED)


def test_majority_is_off_by_default_in_the_source():
    """Negative control: the fail-closed default must not be flipped quietly."""
    assert ConsensusConfig(models=M).accept_majority is False
    src = (PROJECT_ROOT / "labeling" / "consensus.py").read_text(encoding="utf-8")
    assert "accept_majority: bool = False" in src


def test_same_label_but_one_boundary_differs_is_majority_not_unanimous():
    """Unanimity needs the SAME boundaries, not merely aligned ones."""
    wide = _span(S_TEST, "automated testing", "SH")
    i = S_TEST.index("automated testing")
    narrow = {"label": "SH", "start": i, "end": i + 15}          # "automated testi"
    r = _decide(S_TEST, {"model-a": [wide], "model-b": [wide], "model-c": [narrow]})
    assert len(r.candidates) == 1
    c = r.candidates[0]
    assert c.status == MAJORITY and c.boundary == "exact_majority"
    assert c.text == "automated testing" and c.vote_counts == {"SH": 3}
    assert len(c.variants) == 3


def test_label_agreed_but_boundaries_not_is_boundary_disagreement():
    """'boundary disagreement ... tidak akan dipaksakan untuk memperoleh label'."""
    i = S_TEST.index("automated testing")
    a = {"label": "SH", "start": i, "end": i + 17}
    b = {"label": "SH", "start": i, "end": i + 13}               # IoU 13/17 = .76
    r = _decide(S_TEST, {"model-a": [a], "model-b": [b], "model-c": []},
                ConsensusConfig(models=M, accept_majority=True))
    c = r.candidates[0]
    assert (c.status, c.disposition) == (BOUNDARY_DISAGREEMENT, ABSTAIN)
    assert c.label == "SH" and c.start is None and c.text is None
    assert not r.accepted


def test_output_supported_by_one_model_is_not_forced():
    """'keluaran yang hanya didukung satu model'."""
    c = _decide(S_TEST, {"model-a": [JEST], "model-b": [], "model-c": []}).candidates[0]
    assert (c.status, c.disposition) == (NONE_MAJORITY, ABSTAIN)
    assert c.vote_counts == {NONE: 2, "TT": 1}


def test_label_split_is_conflict():
    """'perbedaan label yang tidak dapat diselesaikan'."""
    kn = _span(S_TEST, "Jest", "KN")
    two = _decide(S_TEST, {"model-a": [JEST], "model-b": [kn], "model-c": []}).candidates[0]
    assert (two.status, two.reason) == (CONFLICT, "label_split")
    three = _decide(S_TEST, {"model-a": [JEST], "model-b": [kn],
                             "model-c": [_span(S_TEST, "Jest", "SH")]}).candidates[0]
    assert three.status == CONFLICT and three.disposition == ABSTAIN
    assert abs(three.consensus_ratio - 1 / 3) < 1e-12
    assert abs(three.disagreement_score - 1.0) < 1e-12           # no two agree


def test_two_agree_one_disagrees_on_label_is_majority():
    kn = _span(S_TEST, "Jest", "KN")
    c = _decide(S_TEST, {"model-a": [JEST], "model-b": [JEST], "model-c": [kn]}).candidates[0]
    assert c.status == MAJORITY and c.label == "TT" and c.boundary == "exact_all"


def test_a_failed_model_never_makes_the_others_unanimous():
    """The denominator is all configured LLMs."""
    for broken in ("not json", None):
        votes = {"model-a": [JEST], "model-b": [JEST]}
        if broken is not None:
            votes["model-c"] = broken
        r = _decide(S_TEST, votes)                       # None -> model-c absent
        c = r.candidates[0]
        assert c.status == MAJORITY, (broken, c.status)
        assert c.votes["model-c"] == NO_VOTE
        assert r.sentence_status == "OK" and r.n_valid_outputs == 2
    assert r.model_status["model-c"]["rejection_codes"] == ["MISSING_OUTPUT"]


def test_too_few_valid_outputs_abstains_the_sentence():
    r = _decide(S_TEST, {"model-a": [JEST], "model-b": "{bad", "model-c": "{bad"})
    assert r.sentence_status == "ABSTAIN_INSUFFICIENT_VALID_OUTPUTS"
    assert r.candidates[0].status == CONFLICT
    assert r.candidates[0].reason == "too_few_valid_votes" and not r.accepted


def test_a_model_with_a_rejected_span_does_not_vote_none():
    """Its silence is not agreement: we threw away something it claimed."""
    ghost = _span(S_TEST, "Kubernetes", "TT")                    # not in the sentence
    tainted = _decide(S_TEST, {"model-a": [JEST], "model-b": [], "model-c": [ghost]})
    c = tainted.candidates[0]
    assert c.votes == {"model-a": "TT", "model-b": NONE, "model-c": NO_VOTE}
    assert c.status == CONFLICT                                  # not NONE_MAJORITY
    clean = _decide(S_TEST, {"model-a": [JEST], "model-b": [], "model-c": []})
    assert clean.candidates[0].status == NONE_MAJORITY


def test_no_vote_rule_never_changes_what_is_accepted():
    """The README's claim, as a property: treating a tainted model's silence as
    NO_VOTE instead of NONE cannot add or remove an accepted span."""
    from labeling.schema import Rejection, VerifiedOutput
    rng = random.Random(77)
    cfg = ConsensusConfig(models=M, accept_majority=True)
    pool = [(0, 4), (0, 5), (2, 6), (10, 14), (10, 15), (20, 26)]
    for _ in range(400):
        tainted, cleaned = [], []
        for m in M:
            if rng.random() < 0.15:
                bad = VerifiedOutput(m, "s1", INVALID, rejections=[
                    Rejection("INVALID_JSON", INVALID, "output", "x")])
                tainted.append(bad)
                cleaned.append(bad)
                continue
            chosen = {}
            for b in rng.sample(pool, rng.randint(0, 3)):
                if not any(max(b[0], o[0]) < min(b[1], o[1]) for o in chosen):
                    chosen[b] = rng.choice(["SH", "TT"])
            spans = [VerifiedSpan(s, e, lab, "x" * (e - s), "t") for (s, e), lab in chosen.items()]
            rej = ([Rejection("UNSUPPORTED_TEXT_NOT_FOUND", UNSUPPORTED, "span", "x")]
                   if rng.random() < 0.4 else [])
            tainted.append(VerifiedOutput(m, "s1", VALID, spans=spans, rejections=rej))
            cleaned.append(VerifiedOutput(m, "s1", VALID, spans=spans))
        a = decide_sentence("s1", "x" * 40, tainted, cfg)
        b = decide_sentence("s1", "x" * 40, cleaned, cfg)
        key = lambda r: sorted((c.start, c.end, c.label, c.status) for c in r.accepted)  # noqa: E731
        assert key(a) == key(b)


def test_empty_sentences_report_how_sure_the_ensemble_is():
    assert _decide(S_TEST, {m: [] for m in M}).empty_consensus == "UNANIMOUS_NONE"
    assert _decide(S_TEST, {"model-a": [], "model-b": [],
                            "model-c": "{bad"}).empty_consensus == "MAJORITY_NONE"
    assert _decide(S_TEST, {"model-a": [], "model-b": "{bad",
                            "model-c": "{bad"}).empty_consensus == "UNDETERMINED"
    assert _decide(S_TEST, {m: [JEST] for m in M}).empty_consensus is None


def test_guideline_nested_example_end_to_end():
    spans = [_span(S_K8S, "Configure Kubernetes clusters", "SH"),
             _span(S_K8S, "Kubernetes", "TT")]
    r = _decide(S_K8S, {m: spans for m in M})
    got = [(c.text, c.label, c.status) for c in r.candidates]
    assert got == [("Configure Kubernetes clusters", "SH", UNANIMOUS),
                   ("Kubernetes", "TT", UNANIMOUS)]
    assert all("OVERLAP_POTENTIAL" in c.flags for c in r.candidates)
    assert len(r.accepted) == 2


def test_consensus_rejects_outputs_it_cannot_account_for():
    ok = [verify_output(m, "s1", S_TEST, '{"spans": []}') for m in M]
    stranger = verify_output("model-z", "s1", S_TEST, '{"spans": []}')
    assert _raises(ValueError, decide_sentence, "s1", S_TEST, ok[:2] + [stranger], CFG)
    assert _raises(ValueError, decide_sentence, "s1", S_TEST, ok + [ok[0]], CFG)
    other = verify_output("model-c", "s2", S_TEST, '{"spans": []}')
    assert _raises(ValueError, decide_sentence, "s1", S_TEST, ok[:2] + [other], CFG)


def test_accepted_spans_always_come_from_the_source():
    """End-to-end fuzz from raw JSON: nothing accepted is ever invented text."""
    rng = random.Random(404)
    words = ["Python", "Jest", "automated testing", "experience", "testing", "ghost", "Must"]
    cfg = ConsensusConfig(models=M, accept_majority=True)
    for _ in range(300):
        per = {}
        for m in M:
            if rng.random() < 0.1:
                per[m] = "{broken"
                continue
            per[m] = [{"text": rng.choice(words), "label": rng.choice(["SH", "TT", "KN", "zz"])}
                      for _ in range(rng.randint(0, 3))]
        r = _decide(S_TEST, per, cfg)
        for c in r.accepted:
            assert S_TEST[c.start:c.end] == c.text and c.label in ("SH", "TT", "KN")
            assert sum(1 for v in c.votes.values() if v == c.label) >= cfg.majority_votes
        json.dumps(r.to_dict())                                # serialisable


def test_bridge_to_metrics_gives_acceptance_precision():
    """Accepted spans scored against a reference with the shared engine."""
    i_j, i_p = S_TEST.index("Jest"), S_TEST.index("Python")
    results = [
        decide_sentence("s1", S_TEST, [verify_output(
            m, "s1", S_TEST, json.dumps({"spans": [JEST, _span(S_TEST, "Python", "KN")]}))
            for m in M], CFG),
        decide_sentence("s2", S_TEST, [verify_output(m, "s2", S_TEST, '{"spans": []}')
                                       for m in M], CFG),
    ]
    hyp = accepted_spans_by_sid(results)
    assert set(hyp) == {"s1", "s2"} and hyp["s2"] == []        # population not narrowed
    gold = {"s1": [metrics.Span(i_j, i_j + 4, "TT"), metrics.Span(i_p, i_p + 6, "TT")]}
    m = metrics.micro(metrics.contribs(gold, hyp, ["s1", "s2"]))
    # accepted: Jest/TT (right) and Python/KN (wrong tag) -> precision 1/2, recall 1/2
    assert (m["tp"], m["n_ref"], m["n_hyp"]) == (1, 2, 2)
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_summary_counts_and_intervals():
    results = [
        _decide(S_TEST, {m: [JEST] for m in M}),                                # UNANIMOUS
        _decide(S_TEST, {"model-a": [JEST], "model-b": [JEST], "model-c": []}), # MAJORITY
        _decide(S_TEST, {"model-a": [JEST], "model-b": [], "model-c": []}),     # NONE_MAJ
        _decide(S_TEST, {m: [] for m in M}),                                    # empty
    ]
    s = summarise(results)
    assert s["n_sentences"] == 4 and s["n_candidates"] == 3
    assert s["status_rates"][UNANIMOUS]["k"] == 1
    assert s["status_rates"][MAJORITY]["k"] == 1
    assert s["status_rates"][NONE_MAJORITY]["k"] == 1
    assert s["acceptance_rate"]["k"] == 1 and s["abstention_rate"]["k"] == 2
    assert s["sentences_with_no_candidate"] == {"UNANIMOUS_NONE": 1}
    lo, hi = s["acceptance_rate"]["wilson95"]
    assert 0.0 <= lo <= 1 / 3 <= hi <= 1.0
    empty = summarise([])
    assert math.isnan(empty["acceptance_rate"]["rate"])


def test_every_result_carries_the_frozen_config():
    r = _decide(S_TEST, {m: [JEST] for m in M})
    cfg = r.config
    assert cfg["models"] == list(M) and cfg["majority_votes"] == 2
    assert cfg["iou_threshold"] == 0.5 and cfg["iou_unit"] == "char"
    assert cfg["accept_majority"] is False
    assert cfg["verify_policy"]["overlap"] == "allow_cross_label"


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
