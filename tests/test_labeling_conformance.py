"""Tests for labeling.conformance (the synthetic rule-conformance set) and for
prompt versioning.

Standalone runner (the repo has no pytest):

    .venv-ml\\Scripts\\python.exe tests\\test_labeling_conformance.py

No API calls. What is checked is that the set is well-formed, that its scorer
fails the things it is meant to fail, and that the set cannot be confused with
the gold standard or leak into a prompt.
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_standard import TAGS                                             # noqa: E402
from gold_standard.metrics import Span                                     # noqa: E402
from labeling import conformance as conf                                   # noqa: E402
from labeling import prompts                                               # noqa: E402

ITEMS = conf.ITEMS
BY_SENT = {it.sentence: it for it in ITEMS}


def _raises(exc, fn, *a, **k) -> bool:
    try:
        fn(*a, **k)
    except exc:
        return True
    return False


def _perfect(item) -> list[Span]:
    return [Span(s, e, t) for s, e, t in item.expected]


def _span(item, text: str, tag: str) -> Span:
    i = item.sentence.index(text)
    return Span(i, i + len(text), tag)


# ------------------------------------------------------------ the set itself
def test_set_is_well_formed():
    assert len(ITEMS) >= 100
    assert len({it.item_id for it in ITEMS}) == len(ITEMS)
    assert len({it.sentence for it in ITEMS}) == len(ITEMS)
    for it in ITEMS:
        assert it.origin in ("guideline", "variant") and it.lang in ("en", "id", "mixed")
        for s, e, t in it.expected + it.forbidden:
            assert 0 <= s < e <= len(it.sentence) and t in TAGS, it.item_id
        # no expected span has edges the verifier would trim away
        for s, e, _ in it.expected:
            text = it.sentence[s:e]
            assert text == text.strip() and text[-1] not in ",;:.", (it.item_id, text)
        assert not (set(it.expected) & set(it.forbidden)), it.item_id
        if not it.expected:
            assert it.closed, f"{it.item_id}: a no-span item must be closed"


def test_ambiguous_span_text_is_refused_at_build_time():
    assert _raises(ValueError, conf._locate, "Java and JavaScript.", "Java")
    assert _raises(ValueError, conf._locate, "Python only.", "Docker")
    assert conf._locate("Python only.", "Python") == (0, 6)


def test_set_cannot_be_mistaken_for_the_gold_standard():
    """Ids, location and wording all say synthetic."""
    for it in ITEMS:
        assert it.item_id.startswith("conf-")
        assert not re.match(r"^(u|in)-[0-9a-f]{16}", it.item_id)
    gs = (PROJECT_ROOT / "DATA" / "gold_standard").resolve()
    assert gs not in [conf.OUT_DIR.resolve(), *conf.OUT_DIR.resolve().parents]
    assert "researcher" in conf.REVIEW_STATUS and "2026-11-18" in conf.REVIEW_STATUS
    doc = conf.__doc__
    assert "SYNTHETIC" in doc and "not gold" in doc and "rank the models" in doc


def test_every_major_rule_family_is_covered_in_both_languages():
    rules = {it.rule for it in ITEMS}
    for needed in ("6.2", "6.3", "6.5", "6.8", "7.8", "7.9", "7.10", "5 SS/WB",
                   "5 SH/KN", conf.RULING):
        assert needed in rules, needed
    assert sum(1 for it in ITEMS if it.lang in ("id", "mixed")) >= 20
    assert sum(1 for it in ITEMS if it.rule == "6.2") >= 10        # the verb rule
    assert {t for it in ITEMS for _, _, t in it.expected} == set(TAGS)


def test_variant_items_are_not_examples_in_any_prompt():
    """A variant measures the RULE. If its sentence sat in the prompt it would
    measure whether the model can copy an example."""
    for version in prompts.PROMPT_VERSIONS:
        text = prompts.system_prompt(version)
        for it in ITEMS:
            if it.origin == "variant":
                assert it.sentence.rstrip(".") not in text, (version, it.item_id)


# ---------------------------------------------------------------- the scorer
def test_perfect_answers_pass_every_item():
    out = conf.score({it.item_id: _perfect(it) for it in ITEMS})
    assert out["passed"] == out["n_items"] == len(ITEMS) and out["rate"] == 1.0
    assert out["expected_spans_found"] == out["expected_spans_total"]
    assert sum(v["n"] for v in out["by_rule"].values()) == len(ITEMS)
    assert sum(v["n"] for v in out["by_origin"].values()) == len(ITEMS)


def test_dropping_the_verb_fails_the_verb_rule():
    """The weakness seen in the smoke test must be visible to the scorer."""
    it = BY_SENT["Design database schemas."]
    r = conf.score_item(it, [_span(it, "database schemas", "SH")])
    assert r["pass"] is False
    assert r["missing"] == [("Design database schemas", "SH")]
    it = BY_SENT["Mampu merancang skema basis data."]
    assert conf.score_item(it, [_span(it, "skema basis data", "SH")])["pass"] is False
    assert conf.score_item(it, _perfect(it))["pass"] is True


def test_keeping_the_lead_in_or_the_wrong_tag_fails():
    it = BY_SENT["Able to implement authentication."]
    assert conf.score_item(it, [_span(it, "Able to implement authentication", "SH")])["pass"] is False
    assert conf.score_item(it, [_span(it, "implement authentication", "KN")])["pass"] is False
    assert conf.score_item(it, [_span(it, "implement authentication", "SH")])["pass"] is True


def test_forbidden_span_fails_even_when_the_expected_one_is_there():
    it = BY_SENT["You will work from user flows through wireframes and prototypes."]
    both = _perfect(it) + [_span(it, "wireframes", "SH")]
    r = conf.score_item(it, both)
    assert r["pass"] is False and r["forbidden_hit"] == [("wireframes", "SH")]
    assert conf.score_item(it, _perfect(it))["pass"] is True


def test_closed_items_fail_on_any_extra_span_and_open_items_do_not():
    closed = BY_SENT["Bachelor of Computer Science required."]
    assert closed.closed and not closed.expected
    assert conf.score_item(closed, [])["pass"] is True
    r = conf.score_item(closed, [_span(closed, "Computer Science", "KN")])
    assert r["pass"] is False and r["extra"] == [("Computer Science", "KN")]

    open_item = BY_SENT["Configure Kubernetes clusters."]
    assert not open_item.closed
    nested = _perfect(open_item) + [_span(open_item, "Kubernetes", "TT")]
    r = conf.score_item(open_item, nested)                 # guideline 6.9: allowed
    assert r["pass"] is True and r["extra"] == [("Kubernetes", "TT")]


def test_list_must_be_split_into_separate_spans():
    it = BY_SENT["Python, Java, and Go."]
    one_span = [_span(it, "Python, Java, and Go", "TT")]
    assert conf.score_item(it, one_span)["pass"] is False
    assert conf.score_item(it, _perfect(it))["pass"] is True


def test_invalid_output_fails_even_an_item_that_expects_nothing():
    """A model that returned garbage did not correctly decide 'no tag'."""
    it = BY_SENT["Fluent in English."]
    ok = conf.score({it.item_id: []}, [it], {it.item_id: True})
    bad = conf.score({}, [it], {it.item_id: False})
    assert ok["passed"] == 1 and bad["passed"] == 0
    assert bad["items"][0]["invalid_output"] is True


def test_model_that_returns_nothing_passes_only_the_no_span_items():
    out = conf.score({})
    assert out["passed"] == sum(1 for it in ITEMS if not it.expected)
    assert out["expected_spans_found"] == 0


def test_review_workbook_has_one_row_per_item_and_room_for_a_verdict():
    from openpyxl import load_workbook
    with tempfile.TemporaryDirectory() as td:
        path = conf.export_review(Path(td) / "review.xlsx")
        wb = load_workbook(path)
        ws = wb["items"]
        rows = list(ws.iter_rows(values_only=True))
        assert len(rows) == len(ITEMS) + 1
        head = rows[0]
        assert "OK? (Y / N)" in head and "sentence" in head
        assert all(r[head.index("OK? (Y / N)")] in ("", None) for r in rows[1:])
        assert rows[1][0] == "conf-001"
        assert "read me" in wb.sheetnames


# ----------------------------------------------------------- prompt versions
def test_v0_prompt_is_frozen():
    """Finished runs are keyed on this hash. Editing v0 in place would orphan
    them and silently change what 'v0' means; make a new version instead."""
    assert prompts.prompt_sha256(prompts.PROMPT_VERSION).startswith("d03786b2")
    assert prompts.DEFAULT_PROMPT_VERSION == prompts.PROMPT_VERSION


def test_v1_is_v0_plus_one_section_and_has_its_own_identity():
    v0, v1 = prompts.system_prompt(), prompts.system_prompt(prompts.PROMPT_V1)
    assert v1 != v0 and v1.replace(prompts.V1_ADDENDUM, "") == v0
    assert v1.index("Clarifications on SKILL_HARD") < v1.index("## Output format")
    assert "researcher ruling, 2026-11-18" in v1 and "KN" in prompts.V1_ADDENDUM
    assert prompts.prompt_sha256(prompts.PROMPT_V1) != prompts.prompt_sha256()
    assert prompts.prompt_manifest(prompts.PROMPT_V1)["prompt_version"] == prompts.PROMPT_V1
    assert _raises(ValueError, prompts.system_prompt, "v9")
    msgs = prompts.build_messages("x", version=prompts.PROMPT_V1)
    assert msgs[0]["content"] == v1


def test_prompt_version_is_part_of_the_request_identity():
    from labeling.run import RunConfig
    prices = {"a/x": {"usd_per_m_in": 1, "usd_per_m_out": 1}}
    base = dict(models=("a/x",), prices=prices, max_usd=1.0)
    a = RunConfig(**base)
    b = RunConfig(**base, prompt_version=prompts.PROMPT_V1)
    assert a.request_sha256() != b.request_sha256()


# ------------------------------------------------------------ SkillSpan probe
def test_probe_locates_gold_spans_only_when_unambiguous():
    from labeling import skillspan_probe as sp
    assert sp.locate_gold("Build and test APIs.", "test APIs") == (10, 19)
    # twice in the sentence, context pins it down
    s = "Python developers who teach Python daily."
    assert sp.locate_gold(s, "Python", "teach Python daily") == (28, 34)
    # twice and no usable context -> refused, never guessed
    assert sp.locate_gold(s, "Python", "") is None
    assert sp.locate_gold(s, "Kotlin", "x") is None
    row = {"messages": [{}, {"content": s}, {"content":
           '{"SKILL": [{"skill_span": "Python", "context": ""}], "KNOWLEDGE": []}'}]}
    assert sp.parse_row(row) is None                       # whole sentence dropped


def test_probe_collapses_five_tags_to_two_and_scores_head_retention():
    from labeling import skillspan_probe as sp
    assert sp.COLLAPSE == {"SH": "SKILL", "SS": "SKILL", "WB": "SKILL",
                           "KN": "KNOWLEDGE", "TT": "KNOWLEDGE"}
    gold = [Span(0, 20, "SKILL"), Span(30, 40, "SKILL"), Span(50, 60, "SKILL"),
            Span(70, 80, "SKILL"), Span(90, 95, "SKILL")]
    hyp = sp.collapse([Span(0, 20, "SH"),       # exact
                       Span(28, 38, "SS"),      # starts before the gold: head kept
                       Span(54, 60, "WB"),      # starts after: head dropped
                       Span(72, 80, "TT")])     # overlaps only as KNOWLEDGE
    assert sp.head_retention(gold, hyp) == {"exact": 1, "head_kept": 1, "head_dropped": 1,
                                            "only_as_knowledge": 1, "missed": 1}


def test_probe_scorer_tells_a_perfect_model_from_a_verb_dropping_one():
    from labeling import skillspan_probe as sp
    probe = [{"sentence_id": f"ss-{i}", "sentence_text": "Design database schemas daily.",
              "gold": [Span(0, 23, "SKILL")]} for i in range(10)]
    perfect = {p["sentence_id"]: [Span(0, 23, "SH")] for p in probe}
    nouns = {p["sentence_id"]: [Span(7, 23, "SH")] for p in probe}        # "database schemas"
    a, b = sp.score_model(probe, perfect), sp.score_model(probe, nouns)
    assert a["head_kept_rate"]["rate"] == 1.0 and a["strict"]["micro"]["f1"] == 1.0
    assert b["head_kept_rate"]["rate"] == 0.0 and b["strict"]["micro"]["f1"] == 0.0
    assert b["partial"]["micro"]["f1"] == 1.0              # IoU 16/23 = .70: partial credit
    assert b["skill_span_outcomes"]["head_dropped"] == 10


def test_probe_sample_is_seeded_and_never_written_into_the_gold_tree():
    from labeling import skillspan_probe as sp
    if not sp.SOURCE.exists():
        print("    (skipped: SkillSpan-derived source not present)")
        return
    a, b = sp.build_probe(), sp.build_probe()
    assert [p["sentence_id"] for p in a] == [p["sentence_id"] for p in b]
    assert len(a) == sp.PROBE_N and all(p["sentence_id"].startswith("ss-dev-") for p in a)
    for p in a:
        assert any(s.label == "SKILL" for s in p["gold"])
        for s in p["gold"]:
            assert 0 <= s.start < s.end <= len(p["sentence_text"])


# ------------------------------------------- SkillSpan sentences, guideline tags
def test_any_tag_forbidden_means_the_span_must_not_be_tagged_at_all():
    """Used for 'fluent in Danish and English': a language gets NO tag (7.8)."""
    it = conf.Item("x-1", "", "en", "test", "Good communication and fluent in English.",
                   ((5, 18, "SS"),), ((33, 40, conf.ANY_TAG),))
    assert it.sentence[5:18] == "communication" and it.sentence[33:40] == "English"
    ok = [Span(5, 18, "SS")]
    assert conf.score_item(it, ok)["pass"] is True
    for tag in ("KN", "TT", "SS"):
        r = conf.score_item(it, ok + [Span(33, 40, tag)])
        assert r["pass"] is False and r["forbidden_hit"] == [("English", conf.ANY_TAG)], tag


def test_guideline_annotated_skillspan_file_is_validated_on_load():
    import json
    from labeling import skillspan_guideline as g
    if not g.SOURCE.exists():
        print("    (skipped: SkillSpan-derived source not present)")
        return
    sid, sentence = next((k, v) for k, v in g._source_sentences().items() if len(v) > 40)
    word = next(w for w in sentence.split() if sentence.count(w) == 1 and len(w) > 3)

    def load(items):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.json"
            p.write_text(json.dumps({"review_status": "DRAFT", "items": items}), encoding="utf-8")
            return g.load_items(p)

    items, meta = load([{"sentence_id": sid, "expected": [{"text": word, "tag": "TT"}],
                         "forbidden": [{"text": word}]}])
    it = items[0]
    assert it.sentence == sentence and it.sentence[it.expected[0][0]:it.expected[0][1]] == word
    assert it.forbidden[0][2] == conf.ANY_TAG and meta["review_status"] == "DRAFT"
    # a span that is not in the sentence, an unknown tag, an unknown id, an open empty item
    bad = [
        [{"sentence_id": sid, "expected": [{"text": "zzz-not-there", "tag": "TT"}]}],
        [{"sentence_id": sid, "expected": [{"text": word, "tag": "SKILL"}]}],
        [{"sentence_id": "ss-dev-99999", "expected": [{"text": word, "tag": "TT"}]}],
        [{"sentence_id": sid, "expected": []}],
    ]
    for items in bad:
        assert _raises((ValueError, KeyError), load, items), items


def test_guideline_annotated_set_stays_local_and_says_it_is_a_draft():
    from labeling import skillspan_guideline as g
    gs = (PROJECT_ROOT / "DATA" / "gold_standard").resolve()
    assert gs not in [g.SET_DIR.resolve(), *g.SET_DIR.resolve().parents]
    doc = g.__doc__
    flat = " ".join(doc.split())
    assert "DRAFTED by an LLM" in flat and "not gold" in flat
    assert "never for ranking models" in flat
    assert len(g.CONVENTIONS) >= 5
    if g.ANNOTATIONS.exists() and g.SOURCE.exists():
        items, meta = g.load_items()
        assert len(items) >= 30 and "review_status" in meta
        for it in items:
            assert it.item_id.startswith("ss-dev-")


# ------------------------------------------------------ prompt v2 and batches
def test_v1_and_v2_prompts_are_frozen_too():
    """Finished runs are keyed on these hashes; a new idea is a new version."""
    assert prompts.prompt_sha256(prompts.PROMPT_V1).startswith("ca6397c8")
    assert prompts.prompt_sha256(prompts.PROMPT_V2).startswith("3f2e95c2")


def test_v2_is_v1_plus_the_boundary_conventions():
    v1, v2 = prompts.system_prompt(prompts.PROMPT_V1), prompts.system_prompt(prompts.PROMPT_V2)
    assert v2.replace(prompts.V2_ADDENDUM, "") == v1
    assert v2.index("Clarifications on SKILL_HARD") < v2.index("Span boundary conventions")
    assert v2.index("Span boundary conventions") < v2.index("## Output format")
    assert prompts.PROMPT_VERSIONS == (prompts.PROMPT_VERSION, prompts.PROMPT_V1,
                                       prompts.PROMPT_V2)
    for needle in ("TOO LONG", "lead-in noun", "trailing generic", "Coordinated verbs"):
        assert needle in prompts.V2_ADDENDUM, needle


def test_no_prompt_quotes_a_sentence_of_either_skillspan_batch():
    """Batch 2 is only a fair test of prompt v2 if v2 never saw its sentences."""
    from labeling import skillspan_guideline as g
    if not g.SOURCE.exists():
        print("    (skipped: SkillSpan-derived source not present)")
        return
    for batch in (1, 2):
        path = g.batch_path(batch)
        if not path.exists():
            continue
        items, _ = g.load_items(path)
        for version in prompts.PROMPT_VERSIONS:
            text = prompts.system_prompt(version)
            for it in items:
                core = it.sentence.strip("* ").rstrip(". ")
                assert core not in text, (batch, version, it.item_id)
    # and the v2 section does not quote a multi-word expected span of batch 2
    if g.batch_path(2).exists():
        items, _ = g.load_items(g.batch_path(2))
        for it in items:
            for a, b, _ in it.expected:
                span = it.sentence[a:b]
                if len(span.split()) >= 3:
                    assert span not in prompts.V2_ADDENDUM, (it.item_id, span)


def test_skillspan_batches_are_separate_files_and_do_not_overlap():
    from labeling import skillspan_guideline as g
    assert g.batch_path(1) == g.ANNOTATIONS
    assert g.batch_path(2).name == "annotations_batch2.json"
    assert g.batch_path(2).parent == g.SET_DIR
    if g.SOURCE.exists() and g.batch_path(1).exists() and g.batch_path(2).exists():
        a, _ = g.load_items(g.batch_path(1))
        b, meta = g.load_items(g.batch_path(2))
        assert not ({i.item_id for i in a} & {i.item_id for i in b})
        assert "3f2e95c2" in meta["drafted"], "batch 2 must record the prompt it was drawn after"


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
