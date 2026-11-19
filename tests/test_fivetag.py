"""Tests for fivetag/ — the extraction conditions E0–E6 and the classifiers D1–D3.

Standalone runner (the repo has no pytest):

    .venv-ml\\Scripts\\python.exe tests\\test_fivetag.py

No network, no API key, no spend, no download. The encoder tests train a tiny
randomly-initialised model on SYNTHETIC sentences for a few seconds; they prove
the training loop learns and round-trips through disk, not that any model is
good. The LLM tests use a fake client.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fivetag import TAG_CODES, synthetic                                   # noqa: E402
from fivetag import lexicon as lex                                         # noqa: E402
from fivetag.base import Extractor, Pred, make_pred, to_spans              # noqa: E402
from fivetag.e0_rules import E0Rules, tokenize                             # noqa: E402
from fivetag.evaluate import compare_full, score_full, score_partial      # noqa: E402
from fivetag.typing_d import (ABSTAIN, CONFLICT, CREDENTIAL, EXPERIENCE,   # noqa: E402
                              D1LexiconRegexPriority, D2TaxonomyAnchoredContext,
                              D3Consensus, TypingItem, evaluate_typing)

E0 = E0Rules()


def _got(sentence: str) -> set[tuple[str, str]]:
    return {(p.text, p.label) for p in E0.extract(sentence)}


# --------------------------------------------------------------------- lexicon
def test_lexicon_loads_every_file_and_has_a_stable_hash():
    lx = lex.load("v1")
    assert len(lx.sha256) == 64 and lx.version == "v1"
    assert lx.sha256 == lex.lexicon_sha256(lex.LEXICON_ROOT / "v1")
    for file, section in [("technologies", ""), ("soft_skills", ""), ("work_behaviors", ""),
                          ("knowledge", "heads"), ("knowledge", "terms"), ("knowledge", "groups"),
                          ("actions", "verbs"), ("actions", "activity-nouns"),
                          ("actions", "soft-verbs"), ("boundaries", "lead-ins"),
                          ("boundaries", "stops"), ("boundaries", "function-words"),
                          ("boundaries", "title-heads")]:
        assert lx.entries(file, section), f"{file}:{section} is empty"


def test_lexicon_hash_ignores_line_endings_but_not_content():
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a", Path(tmp) / "b"
        for folder, nl in ((a, b"\n"), (b, b"\r\n")):
            folder.mkdir()
            for name in lex.FILES:
                (folder / f"{name}.txt").write_bytes(b"one" + nl + b"two" + nl)
        assert lex.lexicon_sha256(a) == lex.lexicon_sha256(b)
        (b / "actions.txt").write_bytes(b"one\r\nthree\r\n")
        assert lex.lexicon_sha256(a) != lex.lexicon_sha256(b)


def test_lexicon_parse_sections_and_case_sensitive_mark():
    got = lex.parse("# c\nPython\nGo !cs\n[verbs]\nbuild building\n")
    assert [e.text for e in got[""]] == ["Python", "Go"]
    assert got[""][1].case_sensitive and not got[""][0].case_sensitive
    assert got["verbs"][0].text == "build building"


def test_stops_keep_to_with_when_inside_the_span():
    stops = lex.load("v1").words("boundaries", "stops")
    assert {"using", "for", "in"} <= stops
    assert not ({"to", "with", "when"} & stops)


def test_missing_lexicon_version_is_an_error():
    try:
        lex.load("v999")
    except FileNotFoundError:
        return
    raise AssertionError("a missing lexicon version must not load")


# -------------------------------------------------------------------------- E0
def test_e0_is_an_extractor_and_is_deterministic():
    assert isinstance(E0, Extractor)
    s = "Develop REST APIs using Spring Boot and PostgreSQL."
    assert E0.extract(s) == E0.extract(s)
    m = E0.manifest()
    assert m["lexicon_sha256"] == lex.load("v1").sha256 and m["rules_version"]


def test_e0_offsets_always_match_the_text():
    for row in synthetic.generate(150, seed=7):
        for p in E0.extract(row["text"]):
            assert row["text"][p.start:p.end] == p.text and p.label in TAG_CODES


def test_e0_empty_and_boilerplate_sentences_give_nothing():
    assert E0.extract("") == [] and E0.extract("   ") == []
    assert E0.extract("Our office is located in Jakarta.") == []


def test_e0_technology_is_not_matched_inside_a_longer_word():
    assert ("Java", "TT") not in _got("Experience with JavaScript.")
    assert ("JavaScript", "TT") in _got("Experience with JavaScript.")
    assert ("Go", "TT") not in _got("We go to the office.")          # case-sensitive entry
    assert ("Go", "TT") in _got("Python, Java, and Go.")


def test_e0_list_items_are_separate_spans():
    assert {("Python", "TT"), ("Java", "TT"), ("Docker", "TT")} <= _got("Python, Java, Docker.")


def test_e0_version_number_stays_with_the_technology_but_years_do_not():
    assert ("Java 17", "TT") in _got("Experience with Java 17, Vue 3, and Python 3.11.")
    assert ("Python 3.11", "TT") in _got("Experience with Java 17, Vue 3, and Python 3.11.")
    got = _got("Experience with Python 3 years minimum.")
    assert ("Python", "TT") in got and ("Python 3", "TT") not in got


def test_e0_full_name_with_acronym_is_one_span():
    assert ("Amazon Web Services (AWS)", "TT") in _got("Experience with Amazon Web Services (AWS).")


def test_e0_action_is_cut_before_the_technology():
    got = _got("Build web applications using React.")
    assert ("Build web applications", "SH") in got and ("React", "TT") in got
    got = _got("Membangun aplikasi web menggunakan Laravel.")
    assert ("Membangun aplikasi web", "SH") in got and ("Laravel", "TT") in got


def test_e0_technology_may_sit_inside_an_action_but_knowledge_may_not():
    got = _got("Configure Kubernetes clusters.")
    assert ("Configure Kubernetes clusters", "SH") in got and ("Kubernetes", "TT") in got
    got = _got("Deploy to cloud platforms.")
    assert ("Deploy to cloud platforms", "SH") in got
    assert not any(label == "KN" for _, label in got)


def test_e0_generic_group_is_knowledge_never_a_technology():
    got = _got("Familiarity with NoSQL databases.")
    assert ("NoSQL databases", "KN") in got and ("NoSQL databases", "TT") not in got


def test_e0_ing_form_before_a_knowledge_head_is_a_topic_not_an_action():
    got = _got("Understanding of testing principles.")
    assert ("testing principles", "KN") in got and not any(label == "SH" for _, label in got)


def test_e0_lead_in_words_stay_outside_the_span():
    assert ("Agile principles", "KN") in _got("Understanding Agile principles.")
    assert ("automated testing", "SH") in _got("Experience in automated testing.")
    assert ("communication", "SS") in _got("Strong communication skills.")


def test_e0_job_title_swallows_a_technology_but_not_a_trait():
    assert _got("We are hiring a Backend Developer.") == set()
    assert ("Java", "TT") not in _got("Experience as a Java developer.")
    assert ("detail-oriented", "WB") in _got("We are looking for a detail-oriented developer.")


def test_e0_coordinated_verbs_share_one_span():
    assert ("Design, develop and test web applications", "SH") in _got(
        "Design, develop and test web applications.")


def test_e0_verb_whose_object_is_only_technology_names_is_not_an_action():
    got = _got("Experience in writing Python Kotlin Laravel")
    assert not any(label == "SH" for _, label in got)
    assert {("Python", "TT"), ("Kotlin", "TT"), ("Laravel", "TT")} <= got


def test_e0_same_tag_spans_never_overlap():
    for row in synthetic.generate(200, seed=3):
        ps = E0.extract(row["text"])
        for i, a in enumerate(ps):
            for b in ps[i + 1:]:
                if a.label == b.label:
                    assert a.end <= b.start or b.end <= a.start, (row["text"], a, b)


def test_e0_scores_separate_dictionary_hits_from_patterns():
    by = {p.text: p for p in E0.extract("Build web applications using React.")}
    assert by["React"].score == 1.0 and by["React"].rule.startswith("dict:")
    assert by["Build web applications"].score < 1.0


def test_tokenize_keeps_technology_punctuation_and_records_breaks():
    toks = tokenize("Node.js, C++ and (AWS).")
    assert [t.low for t in toks] == ["node.js", "c++", "and", "aws"]
    assert toks[0].breaks and toks[3].opens and toks[3].ends_clause


# ------------------------------------------------------------------ evaluation
def test_score_full_is_perfect_on_the_reference_and_counts_false_positives():
    rows = synthetic.generate(60, seed=5)
    sents, ref = synthetic.as_reference(rows)
    s = score_full(ref, ref, bootstrap=False)
    assert s["strict"]["f1"] == 1.0 and s["partial"]["f1"] == 1.0
    empty_sid = next(k for k, v in ref.items() if not v)
    hyp = dict(ref)
    hyp[empty_sid] = [make_pred(sents[empty_sid], 0, 3, "TT")]
    s2 = score_full(ref, hyp, bootstrap=False)
    assert s2["strict"]["precision"] < 1.0 and s2["strict"]["recall"] == 1.0


def test_score_full_uses_the_shared_metrics_engine():
    from gold_standard import metrics
    rows = synthetic.generate(80, seed=9)
    sents, ref = synthetic.as_reference(rows)
    hyp = {k: to_spans(E0.extract(v)) for k, v in sents.items()}
    s = score_full(ref, hyp, bootstrap=False)
    direct = metrics.micro(metrics.contribs(ref, hyp, list(ref), "strict"))
    assert s["strict"]["f1"] == direct["f1"] and s["strict"]["tp"] == direct["tp"]


def test_compare_full_reports_the_paired_tests():
    rows = synthetic.generate(60, seed=11)
    sents, ref = synthetic.as_reference(rows)
    a = {k: E0.extract(v) for k, v in sents.items()}
    out = compare_full(ref, a, ref)
    assert "paired_bootstrap_f1" in out and "mcnemar_recall_strict" in out


def test_score_partial_matches_the_conformance_scorer():
    from labeling.conformance import ITEMS, score
    preds = {it.item_id: E0.extract(it.sentence) for it in ITEMS}
    mine = score_partial(ITEMS, preds)
    theirs = score(preds, ITEMS)
    assert mine["passed"] == theirs["passed"]
    assert mine["expected_spans_found"] == theirs["expected_spans_found"]
    assert sum(v["found"] for v in mine["expected_found_by_tag"].values()) \
        == mine["expected_spans_found"]


def test_e0_floor_on_the_conformance_set_has_not_silently_dropped():
    """A regression guard, not a result: the lexicon was written with this set
    in view, so the number is optimistic by construction."""
    from labeling.conformance import ITEMS
    s = score_partial(ITEMS, {it.item_id: E0.extract(it.sentence) for it in ITEMS})
    assert s["expected_spans_found"] >= 120, s["expected_spans_found"]
    assert s["expected_found_by_tag"]["TT"]["found"] == s["expected_found_by_tag"]["TT"]["n"]


# ------------------------------------------------------------------- synthetic
def test_synthetic_is_seeded_marked_and_consistent():
    a, b = synthetic.generate(50, seed=1), synthetic.generate(50, seed=1)
    assert a == b and a != synthetic.generate(50, seed=2)
    for r in a:
        assert r["synthetic"] is True and r["sentence_id"].startswith("syn-")
        for s in r["spans"]:
            assert r["text"][s["start"]:s["end"]].strip() == r["text"][s["start"]:s["end"]]
            assert s["label"] in TAG_CODES
    assert {s["label"] for r in synthetic.generate(300, seed=1) for s in r["spans"]} \
        == set(TAG_CODES)


def test_synthetic_contains_nested_technology_and_long_tail_terms():
    rows = synthetic.generate(400, seed=4)
    nested = any(a["label"] == "SH" and b["label"] == "TT" and a["start"] <= b["start"]
                 and b["end"] <= a["end"] for r in rows for a in r["spans"] for b in r["spans"])
    assert nested
    known = lex.load("v1").words("technologies")
    assert any(r["text"][s["start"]:s["end"]].lower() not in known
               for r in rows for s in r["spans"] if s["label"] == "TT")


def test_synthetic_refuses_the_gold_standard_tree_and_unmarked_rows():
    rows = synthetic.generate(3, seed=0)
    for target, bad_rows in ((synthetic.REPO_ROOT / "DATA" / "gold_standard" / "x.jsonl", rows),
                             (Path(tempfile.gettempdir()) / "x.jsonl", [{"text": "a"}])):
        try:
            synthetic.write_jsonl(bad_rows, target)
        except ValueError:
            continue
        raise AssertionError(f"write_jsonl accepted {target}")
    with tempfile.TemporaryDirectory() as tmp:
        p = synthetic.write_jsonl(rows, Path(tmp) / "s.jsonl")
        assert "SYNTHETIC" in p.read_text(encoding="utf-8").splitlines()[0]
        assert synthetic.read_jsonl(p) == rows


# ---------------------------------------------------------------- D1 / D2 / D3
D1 = D1LexiconRegexPriority()
D2 = D2TaxonomyAnchoredContext(D1.lexicon)
D3 = D3Consensus(D1, D2)


def test_d1_examples_from_the_proposal():
    assert D1.classify("minimum 3 years of experience").label == EXPERIENCE
    assert D1.classify("5+ years").label == EXPERIENCE
    assert D1.classify("Bachelor degree in Computer Science").label == CREDENTIAL
    assert D1.classify("Docker").label == "TT"
    assert D1.classify("teamwork").label == "SS"
    assert D1.classify("detail-oriented").label == "WB"
    assert D1.classify("zzz qqq").label == ABSTAIN


def test_d1_priority_table_decides_when_several_rules_match():
    d = D1.classify("Develop REST APIs with Spring Boot")
    assert d.label == "SH" and d.rule == "starts:hard_verb" and len(d.fired) >= 1
    d = D1.classify("design patterns")              # a term AND a verb-led shape
    assert d.label == "KN" and d.rule == "whole:knowledge_term"
    assert "starts:hard_verb" in d.fired            # the losing rule is still recorded
    rules = [r for r, _ in D1.PRIORITY]
    assert rules.index("regex:experience") < rules.index("whole:technology")


def test_d1_case_sensitive_technology_names():
    assert D1.classify("Go").label == "TT"
    assert D1.classify("go").label == ABSTAIN


def test_d2_anchor_methods_exact_normalised_acronym_fuzzy():
    assert D2.anchor("PostgreSQL") == ("TECHNOLOGY", "exact")
    assert D2.anchor("node js") == ("TECHNOLOGY", "normalised")
    assert D2.anchor("Kubernetis") == ("TECHNOLOGY", "fuzzy")
    assert D2.anchor("GCP")[0] == "TECHNOLOGY"
    assert D2.anchor("zzzqqq") == ("NONE", "")
    assert D2.anchor("Got") == ("NONE", "")           # short names are never fuzzy-matched


def test_d2_context_decides_an_unknown_term():
    s = "Experience using Zentrix in production."
    a = s.index("Zentrix")
    assert D2.classify("Zentrix", s, a, a + 7).label == "TT"
    s = "Memahami konsep containerization."
    a = s.index("containerization")
    d = D2.classify("containerization", s, a, a + 16)
    assert d.label == "KN" and d.cue == "UNDERSTAND"
    assert D2.classify("Zentrix").label == ABSTAIN          # no sentence, no cue


def test_d2_known_technology_stays_a_technology_in_an_understanding_context():
    s = "Knowledge of MongoDB is a plus."
    a = s.index("MongoDB")
    assert D2.classify("MongoDB", s, a, a + 7).label == "TT"


def test_d3_agreement_conflict_and_abstention():
    assert D3.classify("Docker").label == "TT"
    assert D3.classify("design patterns").label == "KN"
    s = "Understanding of unit testing."            # D1 reads the span, D2 also the context
    a = s.index("unit testing")
    d = D3.classify("unit testing", s, a, a + 12)
    assert d.label == CONFLICT and d.conflict_kind == "disagree" and len(d.parts) == 2
    assert {p.label for p in d.parts} == {"SH", "KN"}
    s = "Experience using Zentrix in production."
    a = s.index("Zentrix")
    d = D3.classify("Zentrix", s, a, a + 7)
    assert d.label == CONFLICT and d.conflict_kind == "one_abstained"
    assert D3.classify("zzz qqq").label == ABSTAIN


def test_typing_models_are_deterministic_and_describe_themselves():
    for m in (D1, D2, D3):
        assert m.classify("Develop REST APIs") == m.classify("Develop REST APIs")
        assert m.manifest()["typing_version"]
    assert D2.manifest()["fuzzy_threshold"] == 0.90


def test_evaluate_typing_counts_abstention_and_conflict_separately():
    s = "Experience using Zentrix and Docker with good teamwork."
    items = [TypingItem(s, s.index(t), s.index(t) + len(t), lab)
             for t, lab in [("Zentrix", "TT"), ("Docker", "TT"), ("teamwork", "SS")]]
    r1, r3 = evaluate_typing(D1, items), evaluate_typing(D3, items)
    assert r1["abstention_rate"] == 1 / 3 and r1["conflict_rate"] == 0
    assert r3["conflict_rate"] == 1 / 3 and r3["conflict_kinds"]["one_abstained"] == 1
    assert r1["by_class"]["TT"]["recall"] == 0.5 and r1["by_class"]["TT"]["precision"] == 1.0
    assert r1["accuracy_decided"] == 1.0


# ---------------------------------------------------------------- LLM: E4 / E5
from fivetag import llm as L                                               # noqa: E402

SENT = "Build web applications using React and React."


def test_e4_takes_output_as_returned_and_counts_what_is_wrong():
    raw = json.dumps({"spans": [
        {"text": "Build web applications", "label": "SKILL_HARD"},
        {"text": "React", "label": "TT", "occurrence": 2},
        {"text": "Vue", "label": "TT"},                       # not in the sentence
        {"text": "React", "label": "FRAMEWORK"},              # bad label
        {"label": "TT"}]})                                    # missing text
    preds, st = L.parse_unverified(raw, SENT)
    assert {(p.text, p.label, p.start) for p in preds} == {
        ("Build web applications", "SH", 0), ("React", "TT", SENT.rindex("React"))}
    assert st["json_valid"] and not st["schema_ok"]
    assert (st["n_returned"], st["n_unsupported"], st["n_bad_label"], st["n_missing_field"]) \
        == (5, 1, 1, 1)


def test_e4_invalid_json_and_empty_output():
    for raw in (None, "", "I think the answer is React", '{"spans": "none"}'):
        preds, st = L.parse_unverified(raw, SENT)
        assert preds == [] and not st["schema_ok"]
    assert L.parse_unverified("```json\n{\"spans\": []}\n```", SENT)[1]["schema_ok"]
    summary = L.summarise_structure([L.parse_unverified("x", SENT)[1],
                                     L.parse_unverified('{"spans": []}', SENT)[1]])
    assert summary["json_validity"] == 0.5


def test_e5_keeps_only_verified_spans():
    raw = json.dumps({"spans": [{"text": "Build web applications", "label": "SH"},
                                {"text": "Vue", "label": "TT"}]})
    preds, st = L.parse_verified(raw, SENT, "s1", "m")
    assert [(p.text, p.label) for p in preds] == [("Build web applications", "SH")]
    assert st["rejections"]


def test_retriever_is_deterministic_never_returns_the_query_and_refuses_test_rows():
    pool = synthetic.generate(80, seed=21)
    r = L.Retriever(pool)
    q = pool[5]
    top = r.top_k(q["text"], 4, exclude_id=q["sentence_id"])
    assert len(top) == 4 and top == r.top_k(q["text"], 4, exclude_id=q["sentence_id"])
    assert all(t["text"] != q["text"] for t in top)
    assert len(r.manifest()["pool_sha256"]) == 64
    try:
        L.Retriever(pool + [{"sentence_id": "t", "text": "x", "spans": [], "split": "test"}])
    except ValueError:
        return
    raise AssertionError("a sealed test row was accepted into the retrieval pool")


def test_e5_messages_put_examples_before_the_sentence():
    pool = synthetic.generate(30, seed=22)
    demos = [r for r in pool if r["spans"]][:2]
    msgs = L.build_e5_messages("Experience with Docker.", demos)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert "Experience with Docker." in msgs[-1]["content"]
    answer = json.loads(msgs[2]["content"])
    assert {s["text"] for s in answer["spans"]} == {
        demos[0]["text"][s["start"]:s["end"]] for s in demos[0]["spans"]}


class _Completions:
    def __init__(self, calls):
        self.calls = calls

    def create(self, **kw):
        self.calls.append(kw)
        text = kw["messages"][-1]["content"]
        spans = [{"text": "Docker", "label": "TT"}] if "Docker" in text else []
        return {"choices": [{"message": {"content": json.dumps({"spans": spans})},
                             "finish_reason": "stop"}],
                "model": kw["model"], "usage": {"prompt_tokens": 100, "completion_tokens": 10}}


class _Client:
    def __init__(self, calls):
        self.chat = type("Chat", (), {"completions": _Completions(calls)})()


PRICE = {"usd_per_m_in": 1.0, "usd_per_m_out": 2.0}
SENTS = {"s1": "Experience with Docker.", "s2": "Our office is located in Jakarta."}


def test_run_llm_caches_caps_spending_and_stays_out_of_the_gold_tree():
    from labeling.run import BudgetError
    calls: list = []
    with tempfile.TemporaryDirectory() as tmp:
        kw = dict(condition="E4", model="fam/model", prices=PRICE, max_usd=1.0,
                  out_dir=Path(tmp), client_factory=lambda m: _Client(calls), log=lambda *_: None)
        out = L.run_llm(SENTS, **kw)
        assert [(p.text, p.label) for p in out["preds"]["s1"]] == [("Docker", "TT")]
        assert out["preds"]["s2"] == [] and out["info"]["n_calls_made"] == 2
        assert out["info"]["structure"]["json_validity"] == 1.0
        again = L.run_llm(SENTS, **kw)
        assert again["info"]["n_calls_made"] == 0 and len(calls) == 2
        try:
            L.run_llm(SENTS, **{**kw, "max_usd": 1e-9, "max_tokens": 601})
        except BudgetError:
            pass
        else:
            raise AssertionError("the spending cap did not stop the run")
        assert len(calls) == 2
    try:
        L.run_llm(SENTS, condition="E4", model="m", prices=PRICE, max_usd=1.0,
                  out_dir=L.REPO_ROOT / "DATA" / "gold_standard" / "x",
                  client_factory=lambda m: _Client(calls), log=lambda *_: None)
    except ValueError:
        return
    raise AssertionError("model output was allowed under DATA/gold_standard/")


def test_run_llm_e5_needs_a_retriever_and_sends_examples():
    calls: list = []
    pool = synthetic.generate(40, seed=23)
    with tempfile.TemporaryDirectory() as tmp:
        kw = dict(model="fam/model", prices=PRICE, max_usd=1.0, out_dir=Path(tmp),
                  client_factory=lambda m: _Client(calls), log=lambda *_: None)
        try:
            L.run_llm(SENTS, condition="E5", **kw)
        except ValueError:
            pass
        else:
            raise AssertionError("E5 ran without a retriever")
        out = L.run_llm(SENTS, condition="E5", retriever=L.Retriever(pool), k=3, **kw)
        assert len(calls[0]["messages"]) == 1 + 2 * 3 + 1
        assert out["info"]["retrieval"]["k"] == 3
        assert [(p.text, p.rule) for p in out["preds"]["s1"]] == [("Docker", "verified")]


def test_from_stored_run_reads_only_the_current_request():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "summary.json").write_text(json.dumps({"request_sha256": "new"}), encoding="utf-8")
        recs = [{"model": "m", "sentence_id": "s1", "request_sha256": "old", "ok": True,
                 "raw": json.dumps({"spans": [{"text": "Experience", "label": "SH"}]})},
                {"model": "m", "sentence_id": "s1", "request_sha256": "new", "ok": True,
                 "raw": json.dumps({"spans": [{"text": "Docker", "label": "TT"},
                                              {"text": "Vue", "label": "TT"}]})}]
        (d / "raw.jsonl").write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
        preds, info = L.from_stored_run(d, "m", SENTS)
        assert [(p.text, p.label) for p in preds["s1"]] == [("Docker", "TT")]
        assert info["n_missing"] == 1 and info["structure"]["unsupported_span_rate"] == 0.5
        verified, _ = L.from_stored_run(d, "m", SENTS, verified=True)
        assert [(p.text, p.rule) for p in verified["s1"]] == [("Docker", "verified")]


# -------------------------------------------------------------------------- E6
from fivetag.router import Route, Router, RouterConfig, TRIGGERS, summarise   # noqa: E402


class _Fixed:
    """An 'LLM' that answers from a table; raises KeyError for anything else."""
    name = "fixed"

    def __init__(self, table):
        self.table, self.asked = table, []

    def extract(self, sentence):
        self.asked.append(sentence)
        return [make_pred(sentence, sentence.index(t), sentence.index(t) + len(t), lab,
                          source="fixed") for t, lab in self.table[sentence]]

    def manifest(self):
        return {"extractor": self.name}


def test_router_keeps_a_clean_dictionary_sentence_on_the_nlp_path():
    llm = _Fixed({})
    r = Router(E0, llm).route("Experience with Docker.")
    assert r.path == "nlp" and not r.escalated and llm.asked == []
    assert [(p.text, p.label) for p in r.preds] == [("Docker", "TT")]


def test_router_escalates_on_each_trigger():
    router = Router(E0, None)
    cases = {
        "a_uncertain_confidence": "Build web applications.",
        "c_coordination_or_complex": "Docker and Jenkins or Kubernetes.",
        "d_long_tail_term": "Experience with Zentrix.",
        "e_relation_unclear": "Docker deploy.",
    }
    for trigger, sentence in cases.items():
        r = router.route(sentence)
        assert trigger in r.triggers, (trigger, sentence, r.triggers)
        assert r.path == "nlp-fallback"            # no LLM given: recorded, not hidden

    class Other:
        name = "other"

        def extract(self, s):
            return [make_pred(s, s.index("Docker"), s.index("Docker") + 6, "KN")]

        def manifest(self):
            return {}
    r = Router(E0, None, secondary=Other()).route("Experience with Docker.")
    assert "b_label_disagreement" in r.triggers


def test_router_uses_the_llm_answer_when_escalated_and_can_ablate_a_trigger():
    s = "Experience with Zentrix."
    llm = _Fixed({s: [("Zentrix", "TT")]})
    r = Router(E0, llm).route(s)
    assert r.path == "llm" and [(p.text, p.label) for p in r.preds] == [("Zentrix", "TT")]
    assert r.nlp_preds == [] and llm.asked == [s]
    off = RouterConfig(triggers=tuple(t for t in TRIGGERS if t != "d_long_tail_term"))
    assert Router(E0, llm, config=off).route(s).path == "nlp"


def test_router_merge_policy_union_keeps_dictionary_hits():
    s = "Experience with Docker and Zentrix."
    llm = _Fixed({s: [("Zentrix", "TT")]})
    assert {p.text for p in Router(E0, llm).route(s).preds} == {"Zentrix"}
    union = Router(E0, llm, config=RouterConfig(merge="union_dictionary")).route(s)
    assert {p.text for p in union.preds} == {"Zentrix", "Docker"}


def test_router_summary_and_manifest():
    router = Router(E0, None)
    routes = [router.route(s) for s in ("Experience with Docker.", "Experience with Zentrix.",
                                        "Build web applications.")]
    s = summarise(routes)
    assert s["n_sentences"] == 3 and s["n_escalated"] == 2 and s["llm_unavailable"] == 2
    assert s["by_trigger"]["d_long_tail_term"] == 1
    m = router.manifest()
    assert "not yet tuned" in m["thresholds_status"] and m["primary"]["extractor"] == "E0"
    assert isinstance(routes[0], Route)


# -------------------------------------------------------------------------- E3
def test_gliner_adapter_maps_labels_and_reports_a_missing_package():
    from fivetag.gliner_adapter import LABEL_PROMPTS, GlinerExtractor

    class Fake:
        def predict_entities(self, text, labels, threshold=0.5, flat_ner=False):
            assert set(labels) == set(LABEL_PROMPTS.values())
            return [{"start": 16, "end": 22, "text": "Docker", "label": LABEL_PROMPTS["TT"],
                     "score": 0.91},
                    {"start": 11, "end": 22, "text": "with Docker", "label": LABEL_PROMPTS["TT"],
                     "score": 0.40}]          # same tag, overlapping, lower score: dropped
    ex = GlinerExtractor(model=Fake())
    (p,) = ex.extract("Experience with Docker.")
    assert (p.text, p.label, p.source) == ("Docker", "TT", "E3") and abs(p.score - 0.91) < 1e-9
    assert set(LABEL_PROMPTS) == set(TAG_CODES) and ex.manifest()["labels_version"]

    class Odd:
        def predict_entities(self, *a, **k):
            return [{"span": "Docker"}]
    try:
        GlinerExtractor(model=Odd()).extract("Experience with Docker.")
    except RuntimeError:
        return
    raise AssertionError("an unexpected GLiNER entity shape went unnoticed")


# --------------------------------------------------------------------- E1 / E2
from fivetag import encoder as enc                                         # noqa: E402


def test_bio_coding_round_trips_including_a_nested_technology():
    text = "Configure Kubernetes clusters using Docker."
    spans = [{"start": 0, "end": 29, "label": "SH"}, {"start": 10, "end": 20, "label": "TT"},
             {"start": 36, "end": 42, "label": "TT"}]
    offsets = [(0, 0), (0, 9), (10, 20), (21, 29), (30, 35), (36, 42), (42, 43), (0, 0)]
    special = [1, 0, 0, 0, 0, 0, 0, 1]
    labels = enc.encode_labels(offsets, special, spans)
    sh, tt = TAG_CODES.index("SH"), TAG_CODES.index("TT")
    assert [row[sh] for row in labels] == [enc.IGNORE, enc.B, enc.I, enc.I, enc.O, enc.O, enc.O,
                                           enc.IGNORE]
    assert [row[tt] for row in labels][1:7] == [enc.O, enc.B, enc.O, enc.O, enc.B, enc.O]
    back = enc.decode_labels(text, offsets, special, labels)
    assert {(p.start, p.end, p.label) for p in back} == {(s["start"], s["end"], s["label"])
                                                         for s in spans}


def test_decode_repairs_an_i_without_a_b():
    text = "Use Docker daily."
    offsets, special = [(0, 3), (4, 10), (11, 16)], [0, 0, 0]
    tags = [[enc.O] * 5 for _ in offsets]
    tags[1][TAG_CODES.index("TT")] = enc.I
    (p,) = enc.decode_labels(text, offsets, special, tags)
    assert (p.text, p.label) == ("Docker", "TT")


def test_encoder_refuses_a_sealed_test_row_and_a_bad_span():
    for rows in ([{"sentence_id": "x", "text": "abc", "spans": [], "split": "test"}],
                 [{"sentence_id": "x", "text": "abc",
                   "spans": [{"start": 1, "end": 9, "label": "TT"}]}]):
        try:
            enc.check_rows(rows)
        except ValueError:
            continue
        raise AssertionError(f"check_rows accepted {rows}")


_TRAINED: dict = {}


def _trained():
    """One tiny training run shared by the encoder tests (a few seconds on a CPU)."""
    if not _TRAINED:
        from fivetag.tiny import tiny_encoder, tiny_tokenizer
        tr, dv = synthetic.generate(300, seed=31), synthetic.generate(100, seed=32)
        tok = tiny_tokenizer([r["text"] for r in tr + dv])
        cfg = enc.EncoderConfig(model_name="tiny-random", epochs=6, batch_size=32, lr=3e-3,
                                head_lr=3e-3, seed=0)
        tmp = Path(tempfile.mkdtemp())
        res = enc.train(tr, dv, cfg, tmp, tokenizer=tok, encoder=tiny_encoder(len(tok)),
                        log=lambda *_: None)
        _TRAINED.update(res=res, dir=tmp, dev=dv, cfg=cfg, tok=tok)
    return _TRAINED


def test_encoder_training_loop_learns_on_synthetic_sentences():
    log = _trained()["res"]["log"]
    hist = log["history"]
    assert hist[-1]["train_loss"] < hist[0]["train_loss"] * 0.5
    assert log["best_dev_strict_f1"] > 0.5, log["best_dev_strict_f1"]
    assert log["synthetic_training_data"] is True


def test_encoder_saves_and_loads_to_the_same_predictions():
    t = _trained()
    sents, _ = synthetic.as_reference(t["dev"][:40])
    before = enc.EncoderExtractor(t["res"]["model"], t["tok"], t["cfg"]).extract_many(sents)
    loaded = enc.EncoderExtractor.load(t["dir"])
    after = loaded.extract_many(sents)
    assert {k: [(p.start, p.end, p.label) for p in v] for k, v in before.items()} \
        == {k: [(p.start, p.end, p.label) for p in v] for k, v in after.items()}
    assert isinstance(loaded, Extractor) and loaded.name == "E1"
    assert loaded.manifest()["synthetic_training_data"] is True
    for sid, ps in after.items():
        for p in ps:
            assert sents[sid][p.start:p.end] == p.text and 0.0 < p.score <= 1.0


def test_dapt_runs_and_its_checkpoint_feeds_e2():
    import torch
    from transformers import BertConfig, BertForMaskedLM
    from fivetag import dapt
    from fivetag.tiny import tiny_tokenizer
    rows = synthetic.generate(60, seed=41)
    texts = [r["text"] for r in rows]
    tok = tiny_tokenizer(texts)
    torch.manual_seed(0)
    mlm = BertForMaskedLM(BertConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=1,
                                     num_attention_heads=2, intermediate_size=64,
                                     max_position_embeddings=128))
    with tempfile.TemporaryDirectory() as tmp:
        src, banned = Path(tmp) / "s.txt", Path(tmp) / "test.txt"
        src.write_text("\n".join(texts), encoding="utf-8")
        banned.write_text(texts[0] + "\n", encoding="utf-8")
        kept, removed = dapt.load_sentences(src, banned)
        assert removed >= 1 and texts[0] not in kept
        rec = dapt.run_dapt(kept, Path(tmp) / "ckpt", model_name="tiny", epochs=2, batch_size=16,
                            lr=1e-3, tokenizer=tok, model=mlm, log=lambda *_: None)
        assert len(rec["history"]) == 2 and (Path(tmp) / "ckpt" / "dapt_log.json").is_file()
        from transformers import AutoModel, AutoTokenizer
        cfg = enc.EncoderConfig(model_name=str(Path(tmp) / "ckpt"), condition="E2", epochs=1,
                                batch_size=16, lr=1e-3)
        res = enc.train(rows[:32], rows[32:], cfg, log=lambda *_: None,
                        tokenizer=AutoTokenizer.from_pretrained(Path(tmp) / "ckpt"),
                        encoder=AutoModel.from_pretrained(Path(tmp) / "ckpt"))
        assert res["log"]["config"]["condition"] == "E2"


# ------------------------------------------------------------------- benchmark
def test_benchmark_compare_scores_every_condition_against_a_baseline():
    from fivetag.benchmark import compare, format_table
    rows = synthetic.generate(80, seed=51)
    sents, ref = synthetic.as_reference(rows)
    out = compare(ref, {"E0": {k: E0.extract(v) for k, v in sents.items()}, "perfect": ref},
                  baseline="E0")
    by = {r["condition"]: r for r in out["rows"]}
    assert by["perfect"]["strict_f1"] == 1.0 and by["E0"]["strict_f1"] < 1.0
    assert list(out["paired"]) == ["E0 vs perfect"]
    assert "condition" in format_table(out["rows"])


def test_nothing_in_fivetag_imports_legacy_or_writes_under_gold_standard():
    for path in (PROJECT_ROOT / "fivetag").glob("*.py"):
        src = path.read_text(encoding="utf-8")
        assert "from legacy" not in src and "import legacy" not in src, path.name
    assert "gold_standard" not in str(synthetic.OUT_DIR) and "gold_standard" not in str(L.RUNS_DIR)


# ---------------------------------------------------------- SkillSpan rehearsal
def _skillspan_file(folder: Path, name: str, records: list[tuple[str, dict]]) -> None:
    with (folder / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
        for text, answer in records:
            fh.write(json.dumps({"messages": [
                {"role": "system", "content": "x"}, {"role": "user", "content": text},
                {"role": "assistant", "content": json.dumps(answer)}]}) + "\n")


def test_skillspan_conversion_places_spans_by_context_and_counts_what_it_drops():
    from fivetag import skillspan_rehearsal as R
    text = "Use Java daily and teach Java to juniors with unit testing skills"
    answer = {"SKILL": [{"skill_span": "teach Java to juniors", "context": "and teach Java to juniors with"},
                        {"skill_span": "teach Java", "context": "teach Java to"},      # overlaps: dropped
                        {"skill_span": "missing phrase", "context": "nowhere"}],
              "KNOWLEDGE": [{"skill_span": "Java", "context": "teach Java to"},        # the 2nd "Java"
                            {"skill_span": "Java", "context": "Use Java daily"}]}
    with tempfile.TemporaryDirectory() as tmp:
        _skillspan_file(Path(tmp), "train", [(text, answer), ("Nothing here", {"SKILL": [], "KNOWLEDGE": []})])
        with (Path(tmp) / "train.jsonl").open("a", encoding="utf-8") as fh:
            fh.write("{not json}\n")
        rows, stats = R.load_part("train", Path(tmp))
    assert stats == {"n_sentences": 2, "n_spans": 3, "n_not_found": 1, "n_bad_records": 1,
                     "n_same_tag_overlap_dropped": 1}
    got = {(text[s["start"]:s["end"]], s["start"], s["label"]) for s in rows[0]["spans"]}
    assert got == {("teach Java to juniors", text.index("teach"), "SH"),
                   ("Java", text.rindex("Java"), "KN"), ("Java", text.index("Java"), "KN")}
    assert rows[1]["spans"] == [] and "split" not in rows[0]
    enc.check_rows(rows)


def test_skillspan_rehearsal_says_what_it_is_and_never_calls_itself_a_result():
    from fivetag import skillspan_rehearsal as R
    assert R.LABEL_MAP == {"SKILL": "SH", "KNOWLEDGE": "KN"}
    assert "Not a study result" in R.NOTICE
    fake = {"notice": R.NOTICE, "tiny_toy_model": True, "model_name": "tiny-random", "epochs": 1,
            "device": "cpu", "train_seconds": 6.0,
            "conversion": {"train": {"n_spans": 1, "n_not_found": 0,
                                     "n_same_tag_overlap_dropped": 0}}}
    part = {"strict": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "f1_ci95": [0.4, 0.6]},
            "partial": {"f1": 0.6}, "strict_f1_by_label": {"SKILL (SH layer)": 0.5},
            "spurious_spans_in_untrained_layers": 0}
    text = R.to_text({**fake, "dev": part, "skillspan_test": part})
    assert text.startswith("REHEARSAL") and "TOY MODEL" in text and "SkillSpan test" in text


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
