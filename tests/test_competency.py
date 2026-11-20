"""Tests for competency/ — the layers from five-tag evidence to competencies.

Standalone runner (the repo has no pytest):

    .venv-ml\\Scripts\\python.exe tests\\test_competency.py

No embedding model, no network, no spend: grouping is given a simple stand-in
clusterer, and the KKNI / okupasi step is switched off. What is tested is the
evidence model, the attachment of technologies, the statement and its grounding
check, the status contract, and the guards.
"""
from __future__ import annotations

import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from competency import (GROUP_STATUS, STATUS_CANDIDATE, STATUS_EXPLORATORY,   # noqa: E402
                        STATUS_READY, VALID)
from competency import evidence as E                                          # noqa: E402
from competency import statement as S                                         # noqa: E402
from competency.contract import ContractConfig, StudyFacts, decide            # noqa: E402
from competency.grouping import Group, GroupingConfig, group_evidence         # noqa: E402
from competency.pipeline import run                                           # noqa: E402
from fivetag.base import make_pred                                            # noqa: E402

TEXTS = ["Develop REST APIs using Spring Boot.", "Develop REST APIs with Java 17.",
         "Understanding of data structures.", "Good teamwork and detail-oriented.",
         "Our office is in Jakarta."]


def _sentences(n_postings=6):
    out = {}
    for p in range(n_postings):
        for i, t in enumerate(TEXTS):
            out[f"u-{p:016x}#{i:03d}"] = {"text": t, "posting_id": f"u-{p:016x}"}
    return out


SPANS = {0: [("Develop REST APIs", "SH"), ("Spring Boot", "TT")],
         1: [("Develop REST APIs", "SH"), ("Java 17", "TT")],
         2: [("data structures", "KN")],
         3: [("teamwork", "SS"), ("detail-oriented", "WB")], 4: []}


def _evidence(n_postings=6):
    sents = _sentences(n_postings)
    preds = {}
    for sid, row in sents.items():
        i = int(sid.split("#")[1])
        preds[sid] = [make_pred(row["text"], row["text"].index(t), row["text"].index(t) + len(t),
                                tag, source="test") for t, tag in SPANS[i]]
    return E.build_evidence(preds, sents, source="test"), sents


def _key_clusterer(items, cfg):
    """Stand-in for the real clusterer: one cluster per distinct phrase."""
    groups = defaultdict(list)
    for it in items:
        groups[it.text.lower()].append(it)
    clusters = []
    for n, (text, members) in enumerate(sorted(groups.items())):
        soft = getattr(getattr(members[0], "type", None), "value", "") == "soft"
        clusters.append(SimpleNamespace(
            id=f"cluster_{'s' if soft else 'h'}{n:04d}", stream="soft_skill" if soft else
            "hard_plus_knowledge", items=members, summary_label=text, top_terms=text.split(),
            cohesion_score=0.9))
    return clusters, {"stand_in": True}


# -------------------------------------------------------------------- evidence
def test_evidence_keeps_sentence_and_posting_and_refuses_a_span_that_does_not_fit():
    ev, sents = _evidence(2)
    assert len(ev) == 2 * 7
    e = ev[0]
    assert e.sentence_text[e.start:e.end] == e.text and e.posting_id == E.posting_of(e.sentence_id)
    assert e.evidence_id == f"{e.sentence_id}:{e.start}-{e.end}:{e.tag}"
    bad = {next(iter(sents)): [SimpleNamespace(start=5, end=999, label="TT")]}
    try:
        E.build_evidence(bad, sents)
    except ValueError:
        return
    raise AssertionError("a span outside its sentence became evidence")


def test_canonical_key_merges_versions_for_technologies_only():
    assert E.canonical_key("Java 17", "TT") == E.canonical_key("java", "TT") == "java"
    assert E.canonical_key("Amazon Web Services (AWS)", "TT") == "amazon web services"
    assert E.canonical_key("Develop  REST APIs.", "SH") == "develop rest apis"
    assert E.canonical_key("Level 3", "KN") == "level 3"


def test_profiles_and_denominators_count_postings_not_mentions():
    ev, sents = _evidence(4)
    den = E.denominators(sents)
    assert den == {"n_sentences": 20, "n_postings": 4}
    tech = E.profile(ev, "TT", den["n_postings"])
    assert {r["key"]: r["n_postings"] for r in tech} == {"java": 4, "spring boot": 4}
    wb = E.profile(ev, "WB", den["n_postings"])
    assert wb[0]["text"] == "detail-oriented" and wb[0]["share_of_postings"] == 1.0
    assert E.by_tag(ev) == {"SH": 8, "SS": 4, "KN": 4, "TT": 8, "WB": 4}


# -------------------------------------------------------------------- grouping
def test_competencies_are_built_from_hard_skills_only():
    ev, sents = _evidence(6)
    groups, report = group_evidence(ev, 6, clusterer=_key_clusterer)
    assert {g.label for g in groups} == {"develop rest apis"}          # nothing from KN/SS/TT/WB
    assert {g.stream for g in groups} == {"technical"}
    for g in groups:
        assert set(g.tag_counts) == {"SH"}
    assert report["n_evidence_groupable"] == 12 == report["n_evidence_grouped"]
    assert GroupingConfig().competency_tags == ("SH",)
    assert GroupingConfig().min_postings_per_phrase == 1
    # the earlier design is still reachable, for an ablation
    old, _ = group_evidence(ev, 6, GroupingConfig(competency_tags=("SH", "KN", "SS")),
                            clusterer=_key_clusterer)
    assert {g.label for g in old} == {"develop rest apis", "data structures", "teamwork"}


def test_each_competency_lists_its_soft_skills_behaviours_and_knowledge_with_share_and_lift():
    ev, sents = _evidence(6)
    # two more postings that ask ONLY for teamwork: it becomes less specific to the competency
    extra, preds = {}, {}
    for p in (20, 21):
        sid = f"u-{p:016x}#000"
        extra[sid] = {"text": "Good teamwork.", "posting_id": f"u-{p:016x}"}
        preds[sid] = [make_pred("Good teamwork.", 5, 13, "SS")]
    ev = ev + E.build_evidence(preds, extra)
    (g,) = group_evidence(ev, 8, clusterer=_key_clusterer)[0]
    assert g.soft_skills == [{"text": "teamwork", "n_postings": 6, "share": 1.0, "lift": 1.0}]
    assert g.work_behaviours == [{"text": "detail-oriented", "n_postings": 6, "share": 1.0,
                                  "lift": 1.33}]
    assert g.knowledge == [{"text": "data structures", "n_postings": 6, "share": 1.0,
                            "lift": 1.33}]
    assert g.knowledge_context == []                 # never in the same SENTENCE as the action
    assert {t["text"] for t in g.technologies} == {"Spring Boot", "Java 17"}


def test_technology_is_attached_by_sharing_a_sentence():
    ev, _ = _evidence(6)
    groups, _ = group_evidence(ev, 6, clusterer=_key_clusterer)
    by = {g.label: g for g in groups}
    assert {t["text"] for t in by["develop rest apis"].technologies} == {"Spring Boot", "Java 17"}
    assert by["develop rest apis"].technologies[0]["n_postings"] == 6
    g = by["develop rest apis"]
    assert g.n_evidence == 12 and g.n_postings == 6 and g.share_of_postings == 1.0
    assert g.samples and g.samples[0]["sentence_text"] in TEXTS


def test_posting_groups_count_employers_or_templates_once():
    ev, _ = _evidence(6)
    same_employer = {f"u-{p:016x}": "employer-A" for p in range(5)}
    groups, _ = group_evidence(ev, 6, clusterer=_key_clusterer, posting_group=same_employer)
    g = next(g for g in groups if g.label == "develop rest apis")
    assert g.n_postings == 6 and g.n_posting_groups == 2


def test_every_group_gets_one_of_the_six_statuses_of_the_proposal():
    ev, _ = _evidence(6)
    groups, report = group_evidence(ev, 6, clusterer=_key_clusterer)
    assert all(g.status in GROUP_STATUS.values() for g in groups)
    assert sum(report["status_distribution"].values()) == len(groups)
    few, _ = group_evidence(_evidence(2)[0], 2, clusterer=_key_clusterer)
    assert {g.status for g in few} == {"berfrekuensi_rendah"}
    assert len(GROUP_STATUS) == 6 and GROUP_STATUS["valid_cluster"] == VALID


# ------------------------------------------------------------------- statement
def _group(members, technologies=(), stream="technical", knowledge=()):
    return Group("g1", stream, members[0]["text"], [], list(members), [],
                 sum(m["n_evidence"] for m in members), max(m["n_postings"] for m in members),
                 1.0, max(m["n_postings"] for m in members), 0.9, {},
                 [{"text": t, "n_postings": 3} for t in technologies],
                 [{"text": k, "n_postings": 3} for k in knowledge], status=VALID)


def _m(text, tag, n=5):
    return {"text": text, "tag": tag, "n_evidence": n, "n_postings": n}


def test_deterministic_statement_is_a_verb_and_its_object_without_tool_names():
    g = _group([_m("Develop REST APIs", "SH", 6), _m("developing REST APIs", "SH", 2)],
               ["Spring Boot", "Java"])
    st = S.deterministic_statement(g)
    assert st.text == "Develop REST APIs" and st.technologies == []
    # tool names can be put back for an ablation, never by default
    assert S.deterministic_statement(g, max_technologies=3).text == \
        "Develop REST APIs using Spring Boot and Java"
    assert (st.verb, st.krathwohl, st.language, st.verb_source) == ("develop", "create", "en",
                                                                    "evidence")
    assert st.grounding == 1.0 and st.ungrounded == []


def test_a_knowledge_term_that_starts_like_a_verb_is_not_read_as_an_action():
    st = S.deterministic_statement(_group([_m("design patterns", "KN")], ["Java"]))
    assert st.text == "Explain design patterns" and st.verb_source == "default"
    # a default verb is the method's wording: its level is not reported as demanded
    assert st.krathwohl == "not stated" and st.krathwohl_source == "none"
    assert st.technologies == []


def test_default_verbs_follow_the_tag_and_the_language():
    assert S.deterministic_statement(_group([_m("unit testing", "SH")], ["Jest"])).text \
        == "Perform unit testing"
    assert S.deterministic_statement(_group([_m("teamwork", "SS")], stream="soft")).text \
        == "Demonstrate teamwork"
    st = S.deterministic_statement(_group([_m("Membangun aplikasi web", "SH")], ["Laravel"]))
    assert st.text == "Membangun aplikasi web" and st.language == "id"
    # a one-word label takes its language from the sentences it came from
    g = _group([_m("pemrograman", "SH")])
    g.samples = [{"sentence_text": "Menguasai pemrograman dan memiliki pengalaman kerja."}]
    assert S.deterministic_statement(g).text == "Melakukan pemrograman"
    g.samples = [{"sentence_text": "Experience with programming and the team."}]
    g.members[0]["text"] = "programming"
    assert S.deterministic_statement(g).text == "Perform programming"
    assert S.deterministic_statement(_group([_m("struktur data", "KN")])).text \
        == "Menjelaskan struktur data"


def test_noun_dominated_group_is_not_spoken_for_by_a_rare_verb_phrase():
    st = S.deterministic_statement(_group([_m("code reviews", "SH", 50),
                                           _m("Design RESTful APIs", "SH", 3)]))
    assert st.text == "Perform code reviews" and st.verb_source == "default"


def test_grounding_counts_what_is_not_in_the_evidence():
    g = _group([_m("Develop REST APIs", "SH")], ["Spring Boot"])
    score, missing = S.grounding("Develop REST APIs using Spring Boot", g)
    assert score == 1.0 and missing == []
    score, missing = S.grounding("Develop scalable REST APIs on Kubernetes", g)
    assert missing == ["scalable", "kubernetes"] and abs(score - 3 / 5) < 1e-9


def test_llm_statement_is_checked_and_falls_back_when_it_adds_things():
    g = _group([_m("Develop REST APIs", "SH", 6)], ["Spring Boot"])
    good = json.dumps({"statement": "Develop REST APIs using Spring Boot", "verb": "develop",
                       "object": "REST APIs", "technologies": ["Spring Boot"]})
    st = S.llm_statement(g, lambda messages: good)
    assert st.method == "llm" and st.grounding == 1.0
    for raw, why in [
            (json.dumps({"statement": "Develop REST APIs using Kubernetes",
                         "technologies": ["Kubernetes"]}), "technologies not in the evidence"),
            (json.dumps({"statement": "Responsible for REST APIs", "technologies": []}),
             "allowed operational verb"),
            (json.dumps({"statement": "Develop secure cloud-native microservices platforms",
                         "technologies": []}), "grounding"),
            ("I think the statement is ...", "not JSON")]:
        st = S.llm_statement(g, lambda messages, raw=raw: raw)
        assert st.method == "deterministic-v1", raw
        assert any(why in n for n in st.notes), (why, st.notes)
    msgs = S.build_llm_messages(g, "en")
    payload = json.loads(msgs[1]["content"])
    assert "develop" in payload["allowed_verbs"] and "mengembangkan" not in payload["allowed_verbs"]


def test_llm_statement_may_name_at_most_three_technologies():
    g = _group([_m("Develop REST APIs", "SH", 6)], ["Java", "Python", "Docker", "AWS", "Git"])
    many = json.dumps({"statement": "Develop REST APIs using Java, Python, Docker, AWS and Git",
                       "technologies": ["Java", "Python", "Docker", "AWS", "Git"]})
    st = S.llm_statement(g, lambda messages: many)
    assert st.method == "deterministic-v1" and any("more than 3" in n for n in st.notes)
    hidden = json.dumps({"statement": "Develop REST APIs using Java, Python, Docker and AWS",
                         "technologies": ["Java"]})          # under-declared: counted in the text
    assert S.llm_statement(g, lambda messages: hidden).method == "deterministic-v1"
    three = json.dumps({"statement": "Develop REST APIs using Java, Python and Docker",
                        "technologies": ["Java", "Python", "Docker"]})
    assert S.llm_statement(g, lambda messages: three).method == "llm"
    assert "at most three technologies" in S.SYSTEM


def test_kkni_step_proposes_a_level_for_every_statement_and_flags_confidence():
    import numpy as np
    from competency import mapping as MP

    class FakeEmbedder:
        def encode(self, texts, **kw):
            # two-dimensional toy space: "design" statements point one way, others the other
            return np.array([[1.0, 0.0] if "Design" in t else [0.6, 0.8] for t in texts])
    import competency_kkni_labeller as KL
    original = KL._get_descriptor_embeddings
    KL._get_descriptor_embeddings = lambda model, emb, allowed: (
        [3, 4, 5], np.array([[0.0, 1.0], [0.6, 0.8], [1.0, 0.0]]))
    try:
        comps = [{"statement": {"text": "Design database schemas"}, "evidence": {"members": []}},
                 {"statement": {"text": "Perform unit testing"}, "evidence": {"members": []}}]
        report = MP.map_kkni(comps, embedder=FakeEmbedder())
    finally:
        KL._get_descriptor_embeddings = original
    assert [c["kkni"]["level"] for c in comps] == [5, 4]
    assert all(c["kkni"]["confident"] for c in comps) and report["n_confident"] == 2
    assert report["n_evaluated"] == 2 and set(comps[0]["kkni"]["similarity_by_level"]) == {3, 4, 5}
    assert comps[0]["kkni"]["source"] == MP.KKNI_STEP_VERSION


def test_verb_table_is_versioned_and_marked_unreviewed():
    assert "unreviewed" in S.VERB_TABLE_VERSION
    assert set(S.VERBS.values()) <= {"remember", "understand", "apply", "analyze", "evaluate",
                                     "create"}
    assert S.base_verb("Developing") == "develop" and S.base_verb("banana") is None


# ------------------------------------------- verb + knowledge, and the term map
def test_a_phrase_is_split_into_its_verbs_and_its_object():
    f = S.split_verb_object
    assert f("Develop machine learning algorithms") == (["develop"], "machine learning algorithms")
    assert f("design, develop and test web apps") == (["design", "develop", "test"], "web apps")
    assert f("unit testing") == ([], "unit testing")
    assert f("Develop and maintain") == ([], "Develop and maintain")


def test_a_competency_is_a_verb_plus_the_knowledge_its_phrases_act_on():
    from competency import structure as ST
    sents, preds = {}, {}

    def add(p, i, text, spans):
        sid = f"u-{p:016x}#{i:03d}"
        sents[sid] = {"text": text, "posting_id": f"u-{p:016x}"}
        preds[sid] = [make_pred(text, text.index(t), text.index(t) + len(t), tag) for t, tag in spans]
    for p in range(4):                       # the knowledge term stands alone in 4 postings
        add(p, 0, "Knowledge of machine learning algorithms.", [("machine learning algorithms", "KN")])
    for p in range(3):
        add(p, 1, "Develop machine learning algorithms in Python.",
            [("Develop machine learning algorithms", "SH"), ("Python", "TT")])
    for p in range(3, 6):
        add(p, 1, "Design and evaluate scalable machine learning algorithms for vision.",
            [("Design and evaluate scalable machine learning algorithms for vision", "SH")])
    add(6, 1, "Develop new features.", [("Develop new features", "SH")])
    ev = E.build_evidence(preds, sents)
    vocab = ST.knowledge_vocabulary(ev, ST.StructureConfig())
    assert set(vocab) == {"machine learning algorithms"}
    assert ST.knowledge_in("scalable machine learning algorithms for vision", vocab) == \
        "machine learning algorithms"
    assert ST.knowledge_in("new features", vocab) is None
    # the noun phrase after the verb, cut at the first preposition
    assert ST.object_chunk("scalable machine learning algorithms for vision", False) == \
        ["scalable", "machine", "learning", "algorithms"]
    assert ST.object_chunk("the clean, maintainable and efficient code", False) == \
        ["clean", "maintainable", "and", "efficient", "code"]
    assert ST.object_chunk("aplikasi web dan mobile", True) == ["aplikasi", "web"]

    def one_cluster(items, cfg):
        return [SimpleNamespace(id="cluster_h0000", stream="hard_plus_knowledge", items=list(items),
                                summary_label="ml", top_terms=["ml"], cohesion_score=0.9)], {}
    (g,), _ = group_evidence(ev, 7, clusterer=one_cluster)
    rep = ST.annotate([g], ev)
    assert [(v["verb"], v["n_postings"]) for v in g.verbs] == [("develop", 4), ("design", 3),
                                                              ("evaluate", 3)]
    assert g.verbs[0]["krathwohl"] == "create"
    # "algorithms" is the shared head; it is lengthened while more than half of
    # its postings keep the longer form, and stops before "scalable" (3 of 6)
    k = g.knowledge_objects[0]
    assert (k["text"], k["n_postings"]) == ("machine learning algorithms", 6)
    assert k["verbs"] == ["design", "develop", "evaluate"] and abs(k["share"] - 6 / 7) < 1e-3
    assert k["tagged_knowledge"] is True
    assert len(g.knowledge_objects) == 1            # "features" is in one posting only
    assert rep["n_units_containing_a_term_tagged_knowledge"] == 6
    assert rep["n_units_with_a_verb"] == 7 and rep["n_groups_with_a_knowledge_object"] == 1
    assert rep["n_units_whose_verb_is_read_from_a_noun"] == 0
    # the statement: verb + that knowledge; no tool is named
    st = S.deterministic_statement(g)
    assert st.text == "Design machine learning algorithms" and st.object_source == "knowledge"
    assert st.grounding == 1.0 and "6 of the group's 7 postings" in st.notes[0]


def test_cognitive_level_is_read_from_the_postings_verbs_never_from_a_default_verb():
    from competency import structure as ST
    cfg = ST.StructureConfig()
    # the researcher's case: a group named by the noun "pemrograman", whose
    # postings write "membangun" and "menulis" (create), a few "memelihara" (apply)
    verbs = {"membangun": {f"p{i}" for i in range(6)}, "menulis": {f"p{i}" for i in range(4, 9)},
             "memelihara": {"p0", "p20"}}
    cog = ST.cognitive_profile(verbs, 40, cfg)
    assert (cog["level"], cog["code"]) == ("create", "C6") and cog["n_postings_with_a_verb"] == 10
    assert cog["share"] == 0.9 and cog["share_of_group_postings"] == 0.25
    assert [(r["code"], r["n_postings"]) for r in cog["levels"]] == [("C6", 9), ("C3", 2)]
    assert cog["highest"]["code"] == "C6"
    # the highest level needs a fifth of the verb-using postings behind it
    mostly_apply = {"memelihara": {f"p{i}" for i in range(9)}, "membangun": {"p9"}}
    cog2 = ST.cognitive_profile(mostly_apply, 10, cfg)
    assert cog2["code"] == "C3" and cog2["highest"]["code"] == "C3"
    mostly_apply["membangun"] = {"p8", "p9"}
    assert ST.cognitive_profile(mostly_apply, 10, cfg)["highest"]["code"] == "C6"
    none = ST.cognitive_profile({"membangun": {"p1"}}, 30, cfg)
    assert none["level"] is None and "too few" in none["basis"]
    assert ST.BLOOM_CODE == {"remember": "C1", "understand": "C2", "apply": "C3",
                             "analyze": "C4", "evaluate": "C5", "create": "C6"}
    # the statement of that noun-led group keeps its default verb, but its level
    # is the one the postings ask for, and the difference is written down
    g = _group([_m("pemrograman", "SH", 40), _m("Membangun aplikasi", "SH", 3)])
    g.samples = [{"sentence_text": "Menguasai pemrograman dan memiliki pengalaman."}]
    g.cognitive = cog
    st = S.deterministic_statement(g)
    assert st.text == "Melakukan pemrograman" and st.verb_source == "default"
    assert st.krathwohl == "create" and st.krathwohl_source == "postings"
    assert any("ask for 'create'" in n for n in st.notes)
    # with a verb-led group, the verb on the demanded level is the one chosen
    g2 = _group([_m("Maintain websites", "SH", 5), _m("Build websites", "SH", 4)])
    g2.knowledge_objects = [{"text": "websites", "n_postings": 9, "share": 1.0,
                             "verbs": ["maintain", "build"], "tagged_knowledge": False}]
    g2.cognitive = {"level": "create", "code": "C6"}
    assert S.deterministic_statement(g2).text == "Build websites"



def test_generator_statements_of_an_earlier_run_are_reused_without_a_new_call():
    ev, sents = _evidence(6)

    def fake_generate(groups, clusters_by_id, **kw):
        g = groups[0]
        st = S.Statement("Develop REST APIs for internal services.", "develop", "", [], "create",
                         "en", "v2-generator", "evidence", 0.7)
        return {"statements": {g.group_id: (st, {"kkni_level": 5, "rationale": "r"})},
                "info": {"n_groups_sent": 1, "n_statements": 1, "model": "m"}}

    def must_not_be_called(*a, **k):
        raise AssertionError("the generator was called although statements were to be reused")
    common = dict(evidence_source="test", mapping=False, clusterer=_key_clusterer,
                  log=lambda *_: None)
    with tempfile.TemporaryDirectory() as tmp:
        first = run(ev, sents, Path(tmp) / "a", **common,
                    v2_generator={"max_groups": 5, "max_usd": 1.0, "generate": fake_generate})
        again = run(ev, sents, Path(tmp) / "b", **common,
                    v2_generator={"reuse_from": first["out_dir"], "generate": must_not_be_called})
        comps = json.loads((Path(again["out_dir"]) / "competencies.json").read_text("utf-8"))
        assert comps[0]["family"]["description"]["text"] == "Develop REST APIs for internal services."
        assert comps[0]["cognitive"]["code"] == "C6"
        info = again["v2_generator"]
        assert info["new_calls"] == 0 and info["n_reused"] == 1 and info["reused_from_run"] == "a"
        assert info["model"] == "m"
        # reuse AND new calls together: the family already described is skipped
        got = {}

        def second_generate(groups, clusters_by_id, **kw):
            got.update(skip=set(kw["skip"]))
            return {"statements": {}, "info": {"n_groups_sent": 0, "n_statements": 0}}
        both = run(ev, sents, Path(tmp) / "d", **common,
                   v2_generator={"reuse_from": first["out_dir"], "max_groups": 5, "max_usd": 1.0,
                                 "generate": second_generate})
        assert got["skip"] == {comps[0]["family"]["id"]} and both["v2_generator"]["n_reused"] == 1
        # a run made before families existed (statement stored on the competency) is read too
        old_style = Path(tmp) / "old"
        old_style.mkdir()
        c = dict(comps[0])
        c["statement"] = {**c["statement"], "method": "v2-generator", "text": "Old paragraph."}
        c["v2"] = {"rationale": "old"}
        c["evidence"] = {**c["evidence"], "group_id": c["family"]["id"]}
        (old_style / "competencies.json").write_text(json.dumps([c]), encoding="utf-8")
        (old_style / "report.json").write_text(json.dumps({"v2_generator": {"model": "x"}}),
                                               encoding="utf-8")
        from_old = run(ev, sents, Path(tmp) / "e", **common,
                       v2_generator={"reuse_from": str(old_style)})
        e = json.loads((Path(from_old["out_dir"]) / "competencies.json").read_text("utf-8"))
        assert e[0]["family"]["description"]["text"] == "Old paragraph."
        # a family made of other evidence does not inherit a description
        fewer = [x for x in ev if x.posting_id != ev[0].posting_id]
        third = run(fewer, sents, Path(tmp) / "c", **common,
                    v2_generator={"reuse_from": first["out_dir"]})
        assert third["v2_generator"]["n_reused"] == 0
        assert third["v2_generator"]["n_not_reused_because_the_group_changed"] == 1
        assert again["cognitive"]["by_level"] == {"C6": 1}


def test_a_group_gives_one_competency_per_cognitive_level_with_its_own_demand():
    """The researcher's case: "apply web programming" and "develop a website" are
    two competencies on two levels, even when they are one group of meaning."""
    from competency import structure as ST
    sents, preds = {}, {}

    def add(p, text, spans):
        sid = f"u-{p:016x}#000"
        sents[sid] = {"text": text, "posting_id": f"u-{p:016x}"}
        preds[sid] = [make_pred(text, text.index(t), text.index(t) + len(t), tag) for t, tag in spans]
    for p in range(5):
        add(p, "Develop websites with React and teamwork.",
            [("Develop websites", "SH"), ("React", "TT"), ("teamwork", "SS")])
    for p in range(5, 8):
        add(p, "Apply web programming using PHP.", [("Apply web programming", "SH"), ("PHP", "TT")])
    for p in range(8, 10):
        add(p, "Design and maintain websites.", [("Design and maintain websites", "SH")])
    for p in range(10, 14):
        add(p, "Strong web programming.", [("web programming", "SH")])
    add(14, "Evaluate websites.", [("Evaluate websites", "SH")])          # one posting only
    ev = E.build_evidence(preds, sents)

    def one_cluster(items, cfg):
        return [SimpleNamespace(id="cluster_h0000", stream="hard_plus_knowledge", items=list(items),
                                summary_label="web", top_terms=["web"], cohesion_score=0.9)], {}
    (family,), _ = group_evidence(ev, 15, clusterer=one_cluster)
    family.status = VALID
    parts, rep = ST.split_by_level([family], ev, 15)
    by = {v.group_id: v for v in parts}
    # "web programming" names its work with a noun made from a verb (to program):
    # it belongs to the level of that verb, not to a part without a level
    assert set(by) == {"cluster_h0000@C6", "cluster_h0000@C3"}
    create, apply_ = by["cluster_h0000@C6"], by["cluster_h0000@C3"]
    # "design and maintain websites" is in BOTH parts, through its verb of each level
    assert (create.n_postings, apply_.n_postings) == (11, 5)
    assert S.deterministic_statement(create).text == "Develop websites"
    assert S.deterministic_statement(apply_).text == "Apply web programming"
    assert {v["verb"] for v in apply_.verbs} == {"apply", "maintain"}
    assert {v["verb"] for v in create.verbs} == {"develop", "design", "program"}
    assert create.cognitive["read_from"] == {"verb": 7, "noun made from a verb": 4}
    # each part is described by its own postings: tools, soft skills
    assert [t["text"] for t in create.technologies] == ["React"]
    assert [t["text"] for t in apply_.technologies] == ["PHP"]
    assert [x["text"] for x in create.soft_skills] == ["teamwork"] and apply_.soft_skills == []
    assert create.cognitive["share_of_family_postings"] == round(11 / 15, 4)
    assert create.cognitive["n_family_postings_with_a_verb"] == 15
    assert all(v.family_id == "cluster_h0000" and v.status == VALID for v in parts)
    # the level asked for by one posting makes no competency, and is counted
    assert rep["n_units_on_a_level_too_small_to_stand"] == 1
    assert rep["by_level"] == {"C6": 1, "C3": 1}
    assert rep["n_families_with_more_than_one_level"] == 1


def test_a_phrase_that_names_its_work_with_a_noun_is_read_through_that_nouns_verb():
    r = S.read_phrase
    assert (r("web application development").how, r("web application development").verbs) == \
        ("deverbal", ("develop",))
    assert r("web application development").modifier == "web application"
    assert r("Perform code reviews").how == "light+deverbal" and r("Perform code reviews").verbs == ("review",)
    assert r("pengembangan aplikasi web").obj == "aplikasi web" and r("pengembangan aplikasi web").indonesian
    assert r("development of scalable web apps").obj == "scalable web apps"
    assert r("Melakukan pengujian").verbs == ("menguji",)
    assert r("programming languages").how == "none"          # names no work
    assert r("perform tasks").how == "verb" and r("perform tasks").verbs == ("perform",)
    assert r("Develop REST APIs").how == "verb"


def test_perform_development_is_a_develop_competency_on_the_level_of_develop():
    """The researcher's correction: "Perform web application development" and
    "Perform mobile development" look like C3 (perform) but the work is
    developing; the competencies are "Develop web applications" and "Develop
    mobile applications"."""
    from competency import structure as ST
    sents, preds = {}, {}
    n = [0]

    def add(text, span, tag="SH", times=1):
        for _ in range(times):
            n[0] += 1
            sid = f"u-{n[0]:016x}#000"
            sents[sid] = {"text": text, "posting_id": f"u-{n[0]:016x}"}
            preds[sid] = [make_pred(text, text.index(span), text.index(span) + len(span), tag)]
    add("Experience in web application development.", "web application development", times=6)
    add("Develop web applications.", "Develop web applications", times=3)
    add("Experience in mobile development.", "mobile development", times=4)
    add("Develop mobile applications.", "Develop mobile applications", times=5)
    add("Strong unit testing.", "unit testing", times=5)
    add("Perform code reviews.", "Perform code reviews", times=3)
    add("Review code.", "Review code", times=2)
    add("Menguasai pemrograman.", "pemrograman", times=3)
    add("Knows programming languages.", "programming languages", times=2)
    ev = E.build_evidence(preds, sents)
    reader = ST.Reader(ev, ST.StructureConfig())
    assert reader.row("web application development") == (("develop",), ["web", "applications"],
                                                         False, "confirmed")
    assert reader.row("mobile development") == (("develop",), ["mobile", "applications"], False,
                                                "completed")
    assert reader.row("unit testing")[1:] == ([], False, "unconfirmed")
    assert reader.row("Perform code reviews") == (("review",), ["code"], False, "confirmed")
    assert reader.row("programming languages")[3] == "none"

    def cluster(prefix, ids):
        return SimpleNamespace(id=prefix, stream="hard_plus_knowledge", items=ids,
                               summary_label=prefix, top_terms=[prefix], cohesion_score=0.9)

    def by_topic(items, cfg):
        topics = {"web": [], "mobile": [], "testing": [], "review": [], "id": [], "lang": []}
        for it in items:
            t = it.text.lower()
            key = ("web" if "web" in t else "mobile" if "mobile" in t else "testing" if "unit" in t
                   else "review" if "review" in t else "id" if "pemrograman" in t else "lang")
            topics[key].append(it)
        return [cluster(k, v) for k, v in topics.items() if v], {}
    families, _ = group_evidence(ev, n[0], clusterer=by_topic)
    for f in families:
        f.status = VALID
    parts, rep = ST.split_by_level(families, ev, n[0])
    got = {p.group_id: (S.deterministic_statement(p), p) for p in parts}
    st, part = got["web@C6"]
    assert st.text == "Develop web applications" and st.krathwohl == "create" and part.n_postings == 9
    assert got["mobile@C6"][0].text == "Develop mobile applications"
    # no object could be read for "unit testing": the noun phrase is kept, the level
    # is that of "to test", and the statement says "perform" is not the level
    st, part = got["testing@C5"]
    assert st.text == "Perform unit testing" and st.krathwohl == "evaluate"
    assert st.verb_source == "light verb" and st.krathwohl_source == "deverbal noun"
    assert "not of 'perform'" in st.notes[0] and st.grounding == 1.0
    assert got["review@C5"][0].text == "Review code" and got["review@C5"][1].n_postings == 5
    st, _ = got["id@C6"]
    assert st.text == "Melakukan pemrograman" and st.krathwohl == "create"
    # a noun phrase that names no work stays without a level
    assert got["lang@none"][0].krathwohl == "not stated"
    assert set(got) == {"web@C6", "mobile@C6", "testing@C5", "review@C5", "id@C6", "lang@none"}
    how = rep["units_with_a_level_by_how_they_were_read"]
    assert how["deverbal, object confirmed"] == 6 and how["deverbal, object completed"] == 4
    assert how["deverbal, object unconfirmed"] == 5 and how["light+deverbal, object confirmed"] == 3


def test_families_with_nearly_the_same_centre_are_merged_into_one():
    import numpy as np
    sents, preds = {}, {}
    n = [0]

    def add(text, times):
        for _ in range(times):
            n[0] += 1
            sid = f"u-{n[0]:016x}#000"
            sents[sid] = {"text": text + ".", "posting_id": f"u-{n[0]:016x}"}
            preds[sid] = [make_pred(text + ".", 0, len(text), "SH")]
    add("REST API development", 5)
    add("Integrate REST APIs", 3)
    add("Maintain databases", 4)
    ev = E.build_evidence(preds, sents)

    def fake_embed(texts):
        table = {"rest api development": [1.0, 0.0], "integrate rest apis": [0.98, 0.2],
                 "maintain databases": [0.0, 1.0]}
        v = np.array([table[t.lower()] for t in texts], dtype=float)
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    apart, rep0 = group_evidence(ev, n[0], clusterer=_key_clusterer)
    assert len(apart) == 3 and rep0["merge"]["n_families_merged_away"] == 0
    merged, rep = group_evidence(ev, n[0], GroupingConfig(merge_threshold=0.90),
                                 clusterer=_key_clusterer, merge_embed=fake_embed)
    assert len(merged) == 2 and rep["merge"] == {
        "merge_threshold": 0.90, "n_families_before": 3, "n_merged_sets": 1,
        "n_families_merged_away": 1, "n_families_after": 2}
    api = next(g for g in merged if g.merged_from)
    assert api.n_postings == 8 and {m["text"] for m in api.members} == {
        "REST API development", "Integrate REST APIs"}
    assert len(api.merged_from) == 1 and 0.8 < api.cohesion <= 0.9      # lower than either part
    assert next(g for g in merged if not g.merged_from).n_postings == 4
    # a stricter threshold leaves them apart: every pair must be that close
    strict, _ = group_evidence(ev, n[0], GroupingConfig(merge_threshold=0.99),
                               clusterer=_key_clusterer, merge_embed=fake_embed)
    assert len(strict) == 3


def test_counts_carry_intervals_and_a_small_family_is_pulled_towards_the_common_mix():
    from competency import smoothing as SM
    d = SM.demand(42, 1000)
    assert d["share"] == 0.042 and d["ci95"][0] < 0.042 < d["ci95"][1]
    assert d["ci95"][1] - d["ci95"][0] < 0.03
    assert SM.demand(0, 0)["share"] == 0.0
    counts = {f"big{i}": {"C6": 50 + i, "C3": 50 - i} for i in range(10)}
    counts["small"] = {"C6": 3}                       # 3 postings, all on one level
    mix = SM.pool_level_mix(counts)
    assert mix["strength"] > 0 and abs(mix["mix"]["C6"] - 0.5) < 0.05
    small = mix["shares"]["small"]["C6"]
    assert small["raw"] == 1.0 and 0.5 < small["pooled"] < 1.0       # pulled towards the mix
    big = mix["shares"]["big0"]["C6"]
    assert abs(big["pooled"] - big["raw"]) < abs(small["pooled"] - small["raw"])
    assert SM.pool_level_mix({}) == {"strength": None, "mix": {}, "shares": {}}
    postings = [f"p{i}" for i in range(1000)]
    items = {"a": set(postings[:300]), "b": set(postings[295:590]), "c": set(postings[990:995])}
    ranks = SM.bootstrap_ranks(items, postings)
    assert [ranks[k]["rank"] for k in "abc"] == [1, 2, 3]
    assert ranks["a"]["rank_interval90"] == [1, 2] and ranks["b"]["rank_interval90"] == [1, 2]
    assert ranks["c"]["rank_interval90"] == [3, 3]           # far behind: its rank is certain
    assert SM.bootstrap_ranks(items, postings) == ranks      # a fixed seed
    assert SM.bootstrap_ranks({}, postings) == {}


def test_every_verb_the_extractor_can_open_a_hard_skill_with_has_a_sourced_level():
    from fivetag.lexicon import load
    table = S.load_verb_table()
    for version in ("v1", "v2", "v3"):
        for line in load(version).lines("actions", "verbs"):
            assert line[0] in table, f"{line[0]} (lexicon {version}) has no row in the verb table"
    assert len(table) >= 200 and S.VERB_TABLE_VERSION == "verbs-v2.3-sourced-unreviewed"
    assert {r["level"] for r in table.values()} == set(S.LEVELS)
    assert {r["basis"] for r in table.values()} == {"AK", "LIST", "DEV", "ID", "SOFT", "DEFAULT",
                                                    "NOM"}
    # a verb met only through its noun never opens a phrase: "code reviews" is not "to code"
    assert S.base_verb("code") is None and S.base_verb("program") is None
    assert S.VERBS["program"] == "create"
    for noun, row in S.load_nominals().items():
        assert row["verb"] in table and table[row["verb"]]["lang"] == row["lang"], noun
    # the framework's own placements
    assert table["test"]["level"] == "evaluate" and table["test"]["process"] == "checking"
    assert table["design"]["process"] == "planning" and table["implement"]["level"] == "apply"
    # a verb that lists place on several levels says so
    assert "apply" in table["develop"]["also"] and "remember" in table["define"]["also"]
    assert table["membangun"]["lang"] == "id" and "membangun" in S.INDONESIAN_VERBS
    # verbs the lexicon does not inflect are still recognised in their common forms
    assert S.base_verb("validating") == "validate" and S.base_verb("prototyped") == "prototype"
    sources = (PROJECT_ROOT / "competency" / "verbs" / "SOURCES.md").read_text(encoding="utf-8")
    assert "Krathwohl" in sources and "Stanny" in sources and "unreviewed" in sources.lower()
    assert len(S.verb_table_sha256()) == 64


def test_shared_object_is_found_across_differently_worded_phrases():
    from competency import structure as ST
    cfg = ST.StructureConfig()
    rows = [(f"p{i}", ["build"], ST.object_chunk(o, False), False) for i, o in enumerate([
        "scalable REST APIs for internal services", "secure REST APIs", "REST APIs",
        "REST APIs with Spring Boot", "GraphQL APIs", "the next generation of platforms",
        "clean, maintainable and efficient code", "efficient code"])]
    rows += [("p20", ["membangun"], ST.object_chunk("aplikasi web dan mobile", True), True),
             ("p21", ["mengembangkan"], ST.object_chunk("aplikasi web berbasis Laravel", True), True)]
    found = ST.shared_objects(rows, 10, cfg)
    assert [(k["text"], k["n_postings"]) for k in found] == [
        ("REST APIs", 4), ("efficient code", 2), ("aplikasi web", 2)]
    assert found[0]["verbs"] == ["build"] and found[2]["verbs"] == ["membangun", "mengembangkan"]
    assert all(k["tagged_knowledge"] is False for k in found)


def test_term_map_groups_terms_that_are_asked_for_together():
    from competency import termmap as TM
    rows = []
    stacks = [(["React", "TypeScript", "Next.js", "CSS"], "TT"), (["Python", "pandas", "SQL", "Spark"], "TT")]
    for s, (terms, tag) in enumerate(stacks):
        for p in range(30):
            for t in terms:
                rows.append((f"p{s}-{p}", tag, t.lower(), t))
            rows.append((f"p{s}-{p}", "KN", "web standards" if s == 0 else "data modelling",
                         "web standards" if s == 0 else "data modelling"))
    rows.append(("p0-0", "TT", "cobol", "COBOL"))                # too rare: left out
    rows.append(("p0-0", "SS", "teamwork", "teamwork"))          # not a tag of the map
    m = TM.build(rows, 60, TM.TermMapConfig(n_clusters=2))
    assert m["n_terms"] == 10 and len(m["clusters"]) == 2
    by = {t["term"]: t for t in m["terms"]}
    assert "COBOL" not in by and "teamwork" not in by
    assert len({by[t]["cluster"] for t in ("React", "TypeScript", "Next.js", "CSS", "web standards")}) == 1
    assert len({by[t]["cluster"] for t in ("Python", "pandas", "SQL", "Spark", "data modelling")}) == 1
    assert by["React"]["cluster"] != by["Python"]["cluster"]
    assert by["React"]["n_postings"] == 30 and by["React"]["share"] == 0.5
    assert {c["n_knowledge"] for c in m["clusters"]} == {1} and "notice" in m
    again = TM.build(rows, 60, TM.TermMapConfig(n_clusters=2))
    assert again["terms"] == m["terms"]                          # a fixed seed: the same drawing
    tiny = TM.build(rows[:3], 60, TM.TermMapConfig(min_postings=1))
    assert tiny["n_terms"] <= 5 and len(tiny["clusters"]) == 1   # too few terms to cluster


# -------------------------------------------------------------------- contract
def test_contract_statuses_and_their_blockers():
    ok = dict(group_status=VALID, n_postings=10, grounding=0.9, okupasi_mapped=True)
    assert decide(**ok)["status"] == STATUS_CANDIDATE
    assert decide(**ok)["blockers"] == []
    for change, blocker in [({"group_status": "perlu_dipecah"}, "group_status=perlu_dipecah"),
                            ({"n_postings": 2}, "postings<5"),
                            ({"grounding": 0.5}, "grounding<0.80"),
                            ({"okupasi_mapped": False}, "okupasi_unmapped"),
                            ({"okupasi_mapped": None}, "okupasi_mapping_not_run")]:
        d = decide(**{**ok, **change})
        assert d["status"] == STATUS_EXPLORATORY and blocker in d["blockers"], change


def test_kkni_is_information_only_and_never_changes_status_or_order():
    import inspect
    from competency import contract as CT
    from competency.delta import priority_order
    assert "kkni" not in inspect.signature(CT.decide).parameters
    base = {"statement": {"text": "Develop APIs"}, "n_postings": 9, "share_of_postings": 0.1,
            "contract": {"status": STATUS_CANDIDATE, "blockers": []}}
    a = [{**base, "competency_id": "a", "kkni": {"level": 3, "confident": True}},
         {**base, "competency_id": "b", "kkni": {"level": 7, "confident": True}},
         {**base, "competency_id": "c", "kkni": None}]
    b = [{**c, "kkni": {"level": 5, "confident": False}} for c in a]
    assert [c["competency_id"] for c in priority_order(a)] == \
           [c["competency_id"] for c in priority_order(b)]


def test_a_generator_statement_is_judged_by_the_generators_evidence_filter():
    ok = dict(group_status=VALID, n_postings=10, grounding=0.1, okupasi_mapped=True)
    assert decide(**ok)["status"] == STATUS_EXPLORATORY          # lexical check: too abstract
    d = decide(**ok, grounding_passed=True)
    assert d["status"] == STATUS_CANDIDATE and d["grounding_check"] == "generator_evidence_filter"
    d = decide(**{**ok, "grounding": 0.95}, grounding_passed=False)
    assert d["status"] == STATUS_EXPLORATORY
    assert d["blockers"] == ["statement_evidence_filter_failed"]
    assert decide(**{**ok, "grounding": 0.95})["grounding_check"] == "lexical>=0.80"


def test_two_families_that_give_the_same_statement_on_the_same_level_become_one():
    from competency import structure as ST
    sents, preds = {}, {}
    n = [0]

    def add(text, span, times):
        for _ in range(times):
            n[0] += 1
            sid = f"u-{n[0]:016x}#000"
            sents[sid] = {"text": text, "posting_id": f"u-{n[0]:016x}"}
            preds[sid] = [make_pred(text, text.index(span), text.index(span) + len(span), "SH")]
    add("Develop web applications.", "Develop web applications", 4)
    add("Experience in web application development.", "web application development", 5)
    add("Maintain databases.", "Maintain databases", 3)
    ev = E.build_evidence(preds, sents)

    def by_wording(items, cfg):                       # the clusterer splits by wording
        verb = [i for i in items if i.text.startswith("Develop")]
        noun = [i for i in items if i.text.startswith("web")]
        db = [i for i in items if i.text.startswith("Maintain")]
        return [SimpleNamespace(id=k, stream="hard_plus_knowledge", items=v, summary_label=k,
                                top_terms=[k], cohesion_score=0.9)
                for k, v in (("verb", verb), ("noun", noun), ("db", db))], {}
    families, _ = group_evidence(ev, n[0], clusterer=by_wording)
    for f in families:
        f.status = VALID
    parts, _ = ST.split_by_level(families, ev, n[0])
    assert sorted(S.deterministic_statement(p).text for p in parts if p.level == "create") == [
        "Develop web applications", "Develop web applications"]
    merged, rep = ST.merge_same_statement(families, parts, ev, n[0])
    assert rep["n_families_merged_away"] == 1 and rep["n_merged_sets"] == 1
    assert len(merged) == 2
    web = next(g for g in merged if g.merged_from)
    assert web.group_id == "noun" and web.merged_from == ["verb"] and web.n_postings == 9
    parts2, _ = ST.split_by_level(merged, ev, n[0])
    assert [(S.deterministic_statement(p).text, p.n_postings) for p in parts2
            if p.family_id == "noun"] == [("Develop web applications", 9)]
    # nothing to merge: the families come back unchanged
    same, rep = ST.merge_same_statement(merged, parts2, ev, n[0])
    assert rep["n_families_merged_away"] == 0 and len(same) == 2
    # a statement that names no specific work links nothing ("Develop solutions")
    assert ST.names_specific_work("Develop web applications")
    assert ST.names_specific_work("Review code") and ST.names_specific_work("Perform unit testing")
    for generic in ("Develop solutions", "Troubleshoot issues", "Build applications",
                    "Support development", "Melakukan pengembangan", "Deliver"):
        assert not ST.names_specific_work(generic), generic


def test_a_fragment_statement_cannot_be_a_candidate():
    ok = dict(group_status=VALID, n_postings=10, grounding=0.9, okupasi_mapped=True)
    d = decide(**ok, fragment=True)
    assert d["status"] == STATUS_EXPLORATORY and d["blockers"] == ["statement_is_a_fragment"]
    assert decide(**ok)["status"] == STATUS_CANDIDATE


def test_formal_review_is_never_reached_without_the_three_human_facts():
    ok = dict(group_status=VALID, n_postings=10, grounding=0.9, okupasi_mapped=True)
    d = decide(**ok)
    assert len(d["blockers_to_formal_review"]) == 3
    two = StudyFacts(corpus_valid_for_indonesian_claim=True, expert_review_passed=True)
    assert decide(**ok, facts=two)["status"] == STATUS_CANDIDATE
    full = StudyFacts(True, True, True)
    assert decide(**ok, facts=full)["status"] == STATUS_READY
    assert decide(**{**ok, "n_postings": 1}, facts=full)["status"] == STATUS_EXPLORATORY
    assert "not yet agreed" in ContractConfig().thresholds_status


# -------------------------------------------------------------------- pipeline
def test_pipeline_writes_an_evidence_package_and_says_it_is_an_engineering_run():
    ev, sents = _evidence(6)
    with tempfile.TemporaryDirectory() as tmp:
        report = run(ev, sents, Path(tmp) / "r", evidence_source="test", mapping=False,
                     clusterer=_key_clusterer, log=lambda *_: None)
        out = Path(report["out_dir"])
        assert {p.name for p in out.iterdir()} == {
            "competencies.json", "competencies.csv", "groups.json", "report.json",
            "evidence.parquet", "term_map.json", "family_descriptions.json",
            "technology_profile.csv", "work_behaviour_profile.csv"}
        comps = json.loads((out / "competencies.json").read_text(encoding="utf-8"))
        assert report["notice"].startswith("ENGINEERING RUN")
        assert report["n_evidence_by_tag"]["TT"] == 12 and report["n_competencies"] == len(comps) == 1
        assert [s["text"] for s in comps[0]["evidence"]["soft_skills"]] == ["teamwork"]
        assert [s["text"] for s in comps[0]["evidence"]["work_behaviours"]] == ["detail-oriented"]
        assert [s["text"] for s in comps[0]["evidence"]["knowledge"]] == ["data structures"]
        header = (out / "competencies.csv").read_text("utf-8-sig").splitlines()[0]
        assert "soft_skills" in header and "work_behaviours" in header and "knowledge" in header
        c = next(c for c in comps if c["statement"]["verb"] == "develop")
        assert c["statement"]["text"] == "Develop REST APIs"
        assert [v["verb"] for v in c["evidence"]["verbs"]] == ["develop"]
        assert {t["text"] for t in c["evidence"]["technologies"]} == {"Java 17", "Spring Boot"}
        assert (out / "term_map.json").is_file() and report["term_map"]["n_terms"] == 3
        assert c["contract"]["status"] == STATUS_EXPLORATORY
        assert c["contract"]["blockers"] == ["okupasi_mapping_not_run"]
        assert len(c["evidence"]["evidence_ids"]) == 12 and c["evidence"]["samples"]
        # the one group has one level, so it gives one competency of its family
        assert c["competency_id"] == "comp-cluster_h0001@C6" or c["competency_id"].endswith("@C6")
        assert c["cognitive"]["code"] == "C6" and c["cognitive"]["share_of_family_postings"] == 1.0
        assert c["family"]["competencies"] == [{
            "competency_id": c["competency_id"], "code": "C6", "level": "create",
            "statement": "Develop REST APIs", "n_postings": 6}]
        assert c["family"]["description"] is None and report["n_families"] == 1
        # every count says how sure it is
        assert c["demand"]["n_postings"] == 6 and c["demand"]["share"] == 1.0
        assert c["demand"]["ci95"][0] < 1.0 and c["demand"]["rank"] == 1
        assert c["family"]["rank"] == 1 and c["family"]["rank_interval90"] == [1, 1]
        assert report["smoothing"]["bootstrap_rounds"] == 200 and report["zone_filter"] is None
        assert report["levels"]["by_level"] == {"C6": 1}
        assert "detail-oriented" in (out / "work_behaviour_profile.csv").read_text("utf-8-sig")
        assert report["not_built_yet"]


def test_pipeline_refuses_the_gold_standard_tree():
    from competency.pipeline import GS_DIR
    ev, sents = _evidence(2)
    try:
        run(ev, sents, GS_DIR / "_competency_should_not_exist", evidence_source="t",
            mapping=False, clusterer=_key_clusterer, log=lambda *_: None)
    except ValueError:
        assert not (GS_DIR / "_competency_should_not_exist").exists()
        return
    raise AssertionError("competency output was allowed under DATA/gold_standard/")


def test_package_does_not_import_legacy():
    for path in (PROJECT_ROOT / "competency").glob("*.py"):
        src = path.read_text(encoding="utf-8")
        assert "from legacy" not in src and "import legacy" not in src, path.name


# -------------------------------------------------------------------- temporal
from competency import temporal as T                                          # noqa: E402


def _months(n_per_window=300, windows=4):
    """Postings p{w}_{i}, window w covering six months from January 2025."""
    return {f"p{w}_{i}": 2025 * 12 + 6 * w + (i % 6) for w in range(windows)
            for i in range(n_per_window)}


def _item(share_by_window, n_per_window=300):
    return {f"p{w}_{i}" for w, s in enumerate(share_by_window)
            for i in range(int(round(s * n_per_window)))}


def test_month_index_and_calendar_windows():
    assert T.month_index("2026-03-14 10:00:00+00:00") == 2026 * 12 + 2
    assert T.month_index("") is None and T.month_index("14/03/2026") is None
    windows = T.make_windows(_months(), T.TemporalConfig())
    assert [(w["start"], w["end"], w["n_postings"]) for w in windows] == [
        ("2025-01", "2025-06", 300), ("2025-07", "2025-12", 300),
        ("2026-01", "2026-06", 300), ("2026-07", "2026-12", 300)]
    thin = dict(_months())
    thin.update({f"x{i}": 2024 * 12 + 7 for i in range(10)})
    w = T.make_windows(thin, T.TemporalConfig())
    assert w[0]["n_postings"] == 10 and not w[0]["included"] and w[1]["included"]


def test_share_slope_volatility_and_direction():
    items = {"rising": _item([0.10, 0.20, 0.30, 0.40]), "flat": _item([0.30, 0.30, 0.30, 0.30]),
             "jumpy": _item([0.05, 0.40, 0.05, 0.40]), "rare": {"p0_0", "p1_0"}}
    r = T.analyse(items, _months(), T.TemporalConfig(top_n=2))
    by = {x["item"]: x for x in r["items"]}
    assert "rare" not in by and r["n_items_too_rare"] == 1
    assert by["rising"]["shares"] == [0.1, 0.2, 0.3, 0.4]
    assert abs(by["rising"]["slope_per_window"] - 0.10) < 1e-9 and by["rising"]["direction"] == "naik"
    assert by["flat"]["direction"] == "datar" and by["flat"]["volatility"] == "rendah"
    assert by["jumpy"]["volatility"] == "tinggi" and by["jumpy"]["top_n_changes"] == 3
    assert by["rising"]["trend"] == "menaik" and by["flat"]["trend"] == "tidak_terdeteksi"
    assert r["adequacy"] == "memadai" and len(r["top_n_turnover"]) == 3
    assert "cannot tell" in r["trend_test_note"]


def test_too_few_windows_is_exploratory_and_tests_nothing():
    r = T.analyse({"a": _item([0.2, 0.4], 300)}, _months(windows=2), T.TemporalConfig())
    assert r["adequacy"] == "eksploratif"
    assert r["items"][0]["p_value"] is None and r["items"][0]["trend"] == "tidak_cukup_data"


def test_benjamini_hochberg_and_percentile_rank():
    q = T.benjamini_hochberg([0.01, 0.04, None, 0.03])
    assert q[2] is None and abs(q[0] - 0.03) < 1e-12 and abs(q[1] - 0.04) < 1e-12
    assert abs(q[3] - 0.04) < 1e-12
    assert T._percentile_rank([5, 1, 3]) == [1.0, 0.0, 0.5]
    assert T._percentile_rank([2, 2]) == [0.5, 0.5]


def test_persistence_weight_combines_prevalence_trend_and_prior_and_never_a_status():
    items = {"python": _item([0.40, 0.42, 0.44, 0.46]), "oldtool": _item([0.30, 0.20, 0.12, 0.08])}
    prior = {"python": (1.0, "AI & Machine Learning")}
    r = T.analyse(items, _months(), prior=prior)
    by = {x["item"]: x for x in r["items"]}
    assert by["python"]["prior_matched"] and by["oldtool"]["prior"] == 0.5
    assert by["python"]["persistence_weight"] > by["oldtool"]["persistence_weight"]
    cfg = T.TemporalConfig()
    assert abs(cfg.weight_prevalence + cfg.weight_trend + cfg.weight_prior - 1.0) < 1e-12
    assert "status" not in by["python"] and "starting values" in cfg.thresholds_status


def test_a_volatile_item_is_recommended_at_its_parent_node():
    items = {"flutter": _item([0.05, 0.40, 0.05, 0.40]), "sql": _item([0.3, 0.3, 0.3, 0.3])}
    parent = {"flutter": ("0613", "software and applications development and analysis"),
              "sql": ("0612", "database and network design and administration")}
    by = {x["item"]: x for x in T.analyse(items, _months(), parent=parent)["items"]}
    assert by["flutter"]["recommend_at"] == {
        "level": "parent", "node": "software and applications development and analysis",
        "code": "0613", "example": "flutter"}
    assert by["sql"]["recommend_at"]["level"] == "specific" and by["sql"]["recommend_at"]["anchored"]
    agg = T.aggregate([(0.8, 0.1), (0.2, 0.3)])
    assert agg["max"] == 0.8 and abs(agg["prevalence_weighted_mean"] - 0.35) < 1e-12
    assert T.aggregate([]) == {"max": None, "prevalence_weighted_mean": None}


def test_temporal_output_is_written_outside_the_gold_tree():
    r = T.analyse({"a": _item([0.2, 0.3, 0.4, 0.5])}, _months())
    with tempfile.TemporaryDirectory() as tmp:
        p = T.write(r, Path(tmp))
        assert json.loads(p.read_text(encoding="utf-8"))["items"][0]["item"] == "a"
        assert "share 2025-01..2025-06" in (Path(tmp) / "temporal.csv").read_text("utf-8-sig")
    try:
        T.write(r, T.GS_DIR / "_temporal_should_not_exist")
    except ValueError:
        assert not (T.GS_DIR / "_temporal_should_not_exist").exists()
        return
    raise AssertionError("temporal output was allowed under DATA/gold_standard/")


# ------------------------------------------------- bridge to the v2 generator
def test_v2_bridge_refuses_before_spending_past_the_cap():
    from competency import v2_generator as V2
    assert V2.worst_case_usd(["x" * 2500], "s" * 250, max_tokens=2000, retries=2) > 0.02
    ev, _ = _evidence(6)
    keep: dict = {}

    def clusterer(items, cfg):
        clusters, rep = _key_clusterer(items, cfg)
        for c in clusters:
            c.n_unique_jobs, c.n_items, c.cohesion_std = 6, len(c.items), 0.0
            c.n_skill_items = c.n_knowledge_items = 0
        return clusters, rep
    groups, _ = group_evidence(ev, 6, clusterer=clusterer, keep_clusters=keep)
    assert set(keep) == {g.group_id for g in groups}
    for g in groups:
        g.status = VALID
    try:
        V2.generate(groups, keep, max_groups=5, max_usd=0.0001, log=lambda *_: None)
    except V2.GeneratorBudgetError:
        return
    raise AssertionError("the v2 generator bridge did not enforce its cap")



def test_the_generator_describes_the_family_and_the_competencies_keep_their_own_statement():
    ev, sents = _evidence(6)
    seen = {}

    def fake_generate(groups, clusters_by_id, **kw):
        seen.update(only=kw["only"], skip=kw["skip"])
        g = next(x for x in groups if x.label == "develop rest apis")
        st = S.Statement("Develop REST APIs for internal services. Apply versioning.", "develop",
                         "", [], "create", "en", "v2-generator", "evidence", 0.7)
        return {"statements": {g.group_id: (st, {"kkni_level": 5, "rationale": "r",
                                                 "tools_named_in_statement": ["Java"]})},
                "info": {"n_groups_sent": 1, "n_statements": 1, "model": "m"}}
    with tempfile.TemporaryDirectory() as tmp:
        rep = run(ev, sents, Path(tmp) / "r", evidence_source="test", mapping=False,
                  clusterer=_key_clusterer, log=lambda *_: None,
                  v2_generator={"max_groups": 5, "max_usd": 1.0, "generate": fake_generate})
        comps = json.loads((Path(rep["out_dir"]) / "competencies.json").read_text("utf-8"))
        saved = json.loads((Path(rep["out_dir"]) / "family_descriptions.json").read_text("utf-8"))
    (c,) = comps
    # the competency of the level keeps its verb + knowledge statement ...
    assert c["statement"]["text"] == "Develop REST APIs" and c["statement"]["method"] == \
        "deterministic-v1"
    # ... and the generator's paragraph describes the family it belongs to
    d = c["family"]["description"]
    assert d["text"].startswith("Develop REST APIs for internal services")
    assert d["rationale"] == "r" and d["tools_named"] == ["Java"] and "blends" in d["note"]
    # the model's KKNI proposal is not used (it did not discriminate: EXP-026, EXP-028)
    assert c["kkni"] is None
    assert list(saved) == [c["family"]["id"]] and len(saved[c["family"]["id"]]["evidence_ids"]) == 12
    info = rep["v2_generator"]
    assert info["n_new"] == 1 and info["n_reused"] == 0 and info["new_calls"] == 1
    assert rep["n_families_described_by_the_generator"] == 1
    assert seen == {"only": None, "skip": set()}


# ------------------------------------------ two-stage grouping, rank stability
def test_two_stage_grouping_clusters_frequent_phrases_and_attaches_rare_ones():
    import numpy as np
    sents, preds = {}, {}

    def add(posting, text, span, tag):
        sid = f"u-{posting:016x}#000"
        sents[sid] = {"text": text, "posting_id": f"u-{posting:016x}"}
        preds[sid] = [make_pred(text, text.index(span), text.index(span) + len(span), tag)]
    for p in range(4):                                             # frequent: 4 postings
        add(p, "Develop REST APIs.", "Develop REST APIs", "SH")
    add(10, "Develop scalable REST APIs.", "Develop scalable REST APIs", "SH")   # rare, close
    add(11, "Water the office plants.", "Water the office plants", "SH")         # rare, far
    ev = E.build_evidence(preds, sents)

    def clusterer(items, cfg):
        assert {it.text for it in items} == {"Develop REST APIs"}      # only the frequent phrase
        c = SimpleNamespace(id="cluster_h0001", stream="hard_plus_knowledge", items=list(items),
                            summary_label="develop rest apis", top_terms=["rest", "apis"],
                            cohesion_score=0.9, centroid_embedding=np.array([1.0, 0.0]))
        return [c], {"stand_in": True}

    def embed(texts):
        return np.array([[0.95, 0.31] if "REST" in t else [0.0, 1.0] for t in texts])
    cfg = GroupingConfig(core_min_postings=2, attach_threshold=0.75)
    groups, report = group_evidence(ev, 6, cfg, clusterer=clusterer, embed=embed)
    (g,) = groups
    assert {m["text"] for m in g.members} == {"Develop REST APIs", "Develop scalable REST APIs"}
    assert [(m["text"], m["attached"]) for m in g.members] == [
        ("Develop REST APIs", False), ("Develop scalable REST APIs", True)]   # core first
    assert g.n_postings == 5 and g.n_evidence == 5
    two = report["two_stage"]
    assert (two["n_tail_phrases"], two["n_phrases_attached"], two["n_evidence_attached"]) == (2, 1, 1)
    assert two["n_evidence_in_stage_1"] == 4 and report["n_evidence_left_ungrouped"] == 1
    one_stage, r1 = group_evidence(ev, 6, GroupingConfig(), clusterer=_key_clusterer)
    assert r1["two_stage"] is None and len(one_stage) == 3


def test_rank_stability_compares_consecutive_windows():
    same = T.rank_stability([[0.5, 0.5], [0.3, 0.3], [0.1, 0.1], [0.05, 0.05]], top_ks=(2,))
    assert same["pairs"][0]["spearman"] == 1.0 and same["pairs"][0]["top_k_jaccard"] == {"2": 1.0}
    flipped = T.rank_stability([[0.5, 0.05], [0.3, 0.1], [0.1, 0.3], [0.05, 0.5]], top_ks=(2,))
    assert flipped["pairs"][0]["spearman"] == -1.0
    assert flipped["pairs"][0]["top_k_jaccard"] == {"2": 0.0} and flipped["mean_spearman"] == -1.0
    assert T.rank_stability([[0.1], [0.2], [0.3]])["pairs"] == []          # one window: nothing
    three = T.rank_stability([[0.3, 0.2, 0.1], [0.2, 0.3, 0.2], [0.1, 0.1, 0.3]], top_ks=(1, 9))
    assert len(three["pairs"]) == 2 and "9" not in three["pairs"][0]["top_k_jaccard"]


# ------------------------------------------- coverage, delta, versioned profile
from competency import coverage as CV                                         # noqa: E402
from competency import delta as DL                                            # noqa: E402
from competency import profile as PF                                          # noqa: E402


class _TopicEmbedder:
    """Toy embedding: one axis per topic word; texts sharing words are similar."""
    AXES = ["api", "database", "testing", "teamwork", "network"]

    def encode(self, texts, **kw):
        import numpy as np
        out = []
        for t in texts:
            v = np.array([1.0 if a in t.lower() else 0.0 for a in self.AXES]) + 1e-3
            out.append(v / np.linalg.norm(v))
        return np.array(out)


def _comp(cid, text, members, status=STATUS_CANDIDATE, n=10, blockers=()):
    return {"competency_id": cid, "stream": "technical", "n_postings": n,
            "share_of_postings": n / 100, "n_posting_groups": n,
            "statement": {"text": text, "language": "en", "krathwohl": "apply"},
            "contract": {"status": status, "blockers": list(blockers)},
            "kkni": None, "okupasi": [],
            "evidence": {"members": [{"text": m, "tag": "SH", "n_postings": n, "n_evidence": n}
                                     for m in members],
                         "technologies": [], "knowledge_context": [], "samples": []}}


UNITS = [{"ref_id": "U1", "title": "Develop API", "text": "Develop API", "group": "x"},
         {"ref_id": "U2", "title": "Database and network administration",
          "text": "database network", "group": "x"}]


def test_coverage_proposes_one_of_four_statuses_and_never_validates_itself():
    comps = [_comp("c1", "Develop REST API", ["develop api"]),
             _comp("c2", "Design database schemas", ["database design"]),
             _comp("c3", "Perform unit testing", ["unit testing"]),
             _comp("c4", "Demonstrate teamwork", ["teamwork"], status=STATUS_EXPLORATORY)]
    s = CV.audit(comps, UNITS, "SKKNI", embedder=_TopicEmbedder())
    got = {c["competency_id"]: c["coverage"]["SKKNI"]["proposed"] for c in comps}
    # c2 shares one of the two topics of unit U2: cosine 0.71, above the 0.65 line
    assert got == {"c1": CV.COVERED, "c2": CV.COVERED, "c3": CV.UPDATE_INPUT, "c4": CV.MISSING}
    assert 0.65 <= comps[1]["coverage"]["SKKNI"]["best_similarity"] < 0.75
    assert sum(s["counts"].values()) == 4 and s["validated_by_expert"] is False
    for c in comps:
        cov = c["coverage"]["SKKNI"]
        assert cov["validated_by_expert"] is False and cov["matches"][0]["ref_id"] in ("U1", "U2")
    assert set(CV.COVERAGE_STATUSES) == {"tercakup_jelas", "tercakup_sebagian", "belum_tercakup",
                                         "masukan_pembaruan"}
    assert "not calibrated" in CV.CoverageConfig().thresholds_status


def test_school_curriculum_never_gets_the_update_input_status_and_empty_reference_is_safe():
    comps = [_comp("c3", "Perform unit testing", ["unit testing"])]
    CV.audit(comps, UNITS, "kurikulum", embedder=_TopicEmbedder(), official=False)
    assert comps[0]["coverage"]["kurikulum"]["proposed"] == CV.MISSING
    s = CV.audit(comps, [], "kosong", embedder=_TopicEmbedder())
    assert comps[0]["coverage"]["kosong"]["proposed"] == CV.MISSING and s["n_reference_units"] == 0
    assert CV.load_skkni_units() and CV.load_skkni_units()[0]["title"]


def _with_cov(c, proposed, sim=0.5):
    c["coverage"] = {"K": {"proposed": proposed, "best_similarity": sim,
                           "validated_by_expert": False}}
    return c


def test_five_delta_types_follow_the_proposal():
    add = _with_cov(_comp("a", "x", ["a"]), CV.MISSING, 0.2)
    monitor = _with_cov(_comp("m", "x", ["m"], status=STATUS_EXPLORATORY), CV.MISSING, 0.2)
    deepen = _with_cov(_comp("d", "x", ["d"]), CV.PARTIAL)
    keep = _with_cov(_comp("k", "x", ["k"]), CV.COVERED, 0.8)
    level = _with_cov(_comp("l", "x", ["l"]), CV.COVERED, 0.8)
    level["kkni"] = {"level": 5, "confident": True}
    unsure = _with_cov(_comp("u", "x", ["u"]), CV.COVERED, 0.8)
    unsure["kkni"] = {"level": 5, "confident": False}
    reduce = _with_cov(_comp("r", "x", ["r"]), CV.COVERED, 0.8)
    reduce["temporal"] = {"shares": [0.30, 0.22, 0.15, 0.09], "trend": "tidak_terdeteksi"}
    wobble = _with_cov(_comp("w", "x", ["w"]), CV.COVERED, 0.8)
    wobble["temporal"] = {"shares": [0.30, 0.22, 0.25, 0.09]}
    comps = [add, monitor, deepen, keep, level, unsure, reduce, wobble]
    s = DL.apply(comps, "K", curriculum_kkni_level=2)
    got = {c["competency_id"]: c["delta"]["type"] for c in comps}
    assert got == {"a": DL.ADD, "m": DL.MONITOR, "d": DL.DEEPEN, "k": DL.KEEP,
                   "l": DL.REVIEW_LEVEL, "u": DL.KEEP, "r": DL.REDUCE, "w": DL.KEEP}
    assert s["counts"][DL.REDUCE] == 1 and set(DL.DELTA_TYPES) <= set(s["counts"])
    assert not DL.falls_in_every_window([0.3, 0.2]) and DL.falls_in_every_window([0.3, 0.2, 0.1])


def test_reducing_emphasis_is_capped_at_exploratory_and_weight_only_orders():
    reduce = _with_cov(_comp("r", "x", ["r"], status=STATUS_CANDIDATE), CV.COVERED, 0.8)
    reduce["temporal"] = {"shares": [0.3, 0.2, 0.1], "persistence_weight": 0.99}
    keep = _with_cov(_comp("k", "x", ["k"], status=STATUS_CANDIDATE), CV.COVERED, 0.8)
    keep["temporal"] = {"shares": [0.1, 0.2, 0.3], "persistence_weight": 0.10}
    heavy = _with_cov(_comp("h", "x", ["h"], status=STATUS_EXPLORATORY), CV.PARTIAL)
    heavy["temporal"] = {"shares": [0.1, 0.2, 0.3], "persistence_weight": 1.0}
    comps = [heavy, reduce, keep]
    DL.apply(comps, "K")
    assert reduce["delta"]["status_cap"] == STATUS_EXPLORATORY
    assert reduce["delta"]["effective_status"] == STATUS_EXPLORATORY
    assert reduce["contract"]["status"] == STATUS_CANDIDATE          # the contract is untouched
    # a higher weight never lifts an exploratory item above a candidate
    assert [c["competency_id"] for c in DL.priority_order(comps)] == ["k", "h", "r"]
    try:
        DL.decide(_comp("z", "x", ["z"]), "K")
    except KeyError:
        return
    raise AssertionError("a delta was produced without a coverage audit")


def _run_dir(tmp, comps, name="run1"):
    d = Path(tmp) / name
    d.mkdir()
    (d / "competencies.json").write_text(json.dumps(comps), encoding="utf-8")
    (d / "report.json").write_text(json.dumps({
        "evidence_source": "test", "denominators": {"n_postings": 100, "n_sentences": 900},
        "n_evidence": 50, "n_evidence_by_tag": {"SH": 50},
        "grouping": {"n_groups": 3, "status_distribution": {"sahih": 3}, "n_evidence_grouped": 50,
                     "n_evidence_left_ungrouped": 0},
        "statement_method": "deterministic-v1", "verb_table_version": "v", "grounding_version": "g",
        "contract": {}}), encoding="utf-8")
    return d


def test_profile_is_written_once_drafts_are_marked_and_the_changelog_names_every_move():
    with tempfile.TemporaryDirectory() as tmp:
        v1 = [_comp("c-a", "Develop REST API", ["develop api", "rest api"], STATUS_EXPLORATORY,
                    blockers=["postings<5"]),
              _comp("c-b", "Design database schemas", ["database design"], STATUS_CANDIDATE),
              _comp("c-c", "Perform unit testing", ["unit testing"], STATUS_CANDIDATE)]
        p1 = PF.build(_run_dir(tmp, v1), "2026-1", profiles_dir=Path(tmp) / "profiles")
        assert p1["identity"]["kind"] == "drafts" and "DRAFT" in p1["identity"]["notice"]
        assert p1["changelog"]["previous_version"] is None
        out1 = Path(p1["out_dir"])
        assert out1.parent.name == "drafts" and (out1 / "profile.md").read_text("utf-8").startswith(
            "# Profil Kompetensi RPL 2026-1")
        try:
            PF.build(_run_dir(tmp, v1, "again"), "2026-1", profiles_dir=Path(tmp) / "profiles")
        except PF.ProfileError:
            pass
        else:
            raise AssertionError("an existing profile version was overwritten")
        v2 = [_comp("x-1", "Develop REST API", ["develop api", "rest api"], STATUS_CANDIDATE),   # rose
              _comp("x-2", "Design database schemas", ["database design"], STATUS_EXPLORATORY,
                    blockers=["okupasi_unmapped"]),                                               # fell
              _comp("x-3", "Deploy containers", ["deploy containers"], STATUS_CANDIDATE)]        # entered
        p2 = PF.build(_run_dir(tmp, v2, "run2"), "2026-2", previous=out1 / "profile.json",
                      valid_for_release=True, profiles_dir=Path(tmp) / "profiles")
        log = p2["changelog"]
        assert log["previous_version"] == "2026-1" and p2["identity"]["kind"] == "releases"
        assert [r["statement"] for r in log["naik"]] == ["Develop REST API"]
        assert "postings<5" in log["naik"][0]["reason"]
        assert [r["statement"] for r in log["turun"]] == ["Design database schemas"]
        assert "okupasi_unmapped" in log["turun"][0]["reason"]
        assert [r["statement"] for r in log["masuk"]] == ["Deploy containers"]
        assert [r["statement"] for r in log["keluar"]] == ["Perform unit testing"]
        assert p2["summary"]["by_status"] == {STATUS_CANDIDATE: 2, STATUS_EXPLORATORY: 1}
        md = (Path(p2["out_dir"]) / "profile.md").read_text(encoding="utf-8")
        assert "1 naik, 1 turun, 1 masuk, 1 keluar" in md and "DRAFT" not in md


# ------------------------------------------------------- end-to-end pipelines
from competency import endtoend as EE                                         # noqa: E402


class _FakeLLM:
    """One fake model for the three jobs: extraction, adjudication, statements."""

    def __init__(self):
        self.calls = {"extract": 0, "adjudicate": 0, "statement": 0}
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kw):
        system, last = kw["messages"][0]["content"], kw["messages"][-1]["content"]
        if "competency statement" in system:
            self.calls["statement"] += 1
            payload = json.loads(last)
            phrase = payload["phrases_from_postings"][0]["text"]
            content = json.dumps({"statement": phrase, "technologies": []})
        elif "decide the tag" in system:
            self.calls["adjudicate"] += 1
            content = json.dumps({"label": json.loads(last)["candidates"][0]})
        else:
            self.calls["extract"] += 1
            spans = [{"text": t, "label": tag} for t, tag in (
                ("Develop REST APIs", "SH"), ("Spring Boot", "TT"), ("Java 17", "TT"),
                ("data structures", "KN"), ("teamwork", "SS"), ("detail-oriented", "WB"))
                if t in last]
            content = json.dumps({"spans": spans})
        return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "model": kw["model"], "usage": {"prompt_tokens": 50, "completion_tokens": 10}}


def test_five_pipelines_run_on_the_same_postings_and_differ_as_defined():
    assert list(EE.SPECS) == ["P1", "P2", "P3", "P3-tanpa-verifier", "P3-tanpa-persistensi"]
    assert EE.SPECS["P1"].extraction == "rules" and EE.SPECS["P1"].statements == "deterministic"
    assert EE.SPECS["P3-tanpa-verifier"].verifier is False
    assert EE.SPECS["P3-tanpa-persistensi"].persistence is False
    sents = _sentences(6)
    fake = _FakeLLM()
    with tempfile.TemporaryDirectory() as tmp:
        r = EE.run_all(sents, Path(tmp) / "e2e", max_usd=1.0, client_factory=lambda m: fake,
                       clusterer=_key_clusterer, mapping=False, posting_month=None,
                       log=lambda *_: None)
        by = {row["pipeline"]: row for row in r["rows"]}
        assert set(by) == set(EE.SPECS)
        assert by["P1"]["new_cost_usd"] == 0 and "n_sentences_sent_to_llm" not in by["P1"]["extraction"]
        assert by["P2"]["extraction"]["n_sentences_sent_to_llm"] == len(sents)
        assert by["P3"]["extraction"]["n_sentences_sent_to_llm"] <= len(sents)
        # answers paid for by P2 are reused by the P3 variants: no sentence is asked twice
        assert fake.calls["extract"] <= len(sents)
        assert by["P3"]["new_cost_usd"] <= by["P2"]["new_cost_usd"]
        assert all(row["n_competencies"] >= 1 for row in r["rows"])
        assert 0 < r["total_charged_usd"] <= 1.0 and r["notice"].startswith("REHEARSAL")
        assert (Path(tmp) / "e2e" / "comparison.json").is_file()
        assert "P1 vs P2" in r["top20_overlap"] and "expert appropriateness" in r["not_measured"]
        text = EE.to_text(r)
        assert "pipeline" in text and "P3-tanpa-verifier" in text


def test_conflicts_are_left_out_without_an_llm_and_adjudicated_with_one():
    s = "Understanding of unit testing."
    sents = {f"u-{p:016x}#000": {"text": s, "posting_id": f"u-{p:016x}"} for p in range(3)}
    preds = {sid: [make_pred(s, s.index("unit testing"), s.index("unit testing") + 12, "SH")]
             for sid in sents}
    ev = E.build_evidence(preds, sents)
    kept, info = EE.resolve_conflicts(ev, EE.SPECS["P1"])
    assert kept == [] and info["n_conflicts"] == 3 and "left out" in info["resolution"]
    kept, info = EE.resolve_conflicts(ev, EE.SPECS["P3"], adjudicate=lambda m: '{"label": "KN"}')
    assert [e.tag for e in kept] == ["KN"] * 3 and info["n_tag_changed"] == 3
    assert all(e.evidence_id.endswith(":KN") for e in kept)
    kept, info = EE.resolve_conflicts(ev, EE.SPECS["P2"])
    assert len(kept) == 3 and info == {"used": False}


def test_unchecked_statement_is_used_but_its_problems_are_recorded():
    g = _group([_m("Develop REST APIs", "SH", 6)], ["Spring Boot"])
    raw = json.dumps({"statement": "Develop REST APIs using Kubernetes", "technologies": ["Kubernetes"]})
    st = S.unchecked_llm_statement(g, lambda messages: raw)
    assert st.method == "llm-unchecked" and st.text == "Develop REST APIs using Kubernetes"
    assert st.grounding < 1.0 and any("would have been rejected" in n for n in st.notes)
    assert S.llm_statement(g, lambda messages: raw).method == "deterministic-v1"


# --------------------------------------------------------------------- figures
def test_figures_are_drawn_from_files_and_carry_their_status_note():
    from competency import figures as F
    with tempfile.TemporaryDirectory() as tmp:
        a = F.hbar(["a", "b"], [3, 1], "t", Path(tmp) / "a.png", "REHEARSAL · not a finding")
        b = F.lines(["w1", "w2", "w3"], {"x": [0.1, 0.2, 0.3], "y": [0.3, 0.2, 0.1]}, "t",
                    Path(tmp) / "b.png", "note", subtitle="s")
        c = F.grouped_bars(["g1", "g2"], {"v0": [1, 2], "v1": [2, 3]}, "t", Path(tmp) / "c.png",
                           "note", reference=(3, "all"))
        d = F.interval_dots(["m"], [0.9], [(0.85, 0.94)], "t", Path(tmp) / "d.png", "note")
        assert all(p.is_file() and p.stat().st_size > 2000 for p in (a, b, c, d))
        ev, sents = _evidence(6)
        rep = run(ev, sents, Path(tmp) / "r", evidence_source="test", mapping=False,
                  clusterer=_key_clusterer, log=lambda *_: None)
        made = F.figures_for_run(Path(rep["out_dir"]))
        names = {p.name for p in made}
        assert {"evidence_by_tag.png", "group_status.png", "contract_status.png"} <= names
    assert list(F.TAG_COLOR) == ["SH", "SS", "KN", "TT", "WB"] and len(set(F.SERIES)) == 8
    src = (PROJECT_ROOT / "competency" / "figures.py").read_text(encoding="utf-8")
    assert "twinx" not in src                                # never two y-scales on one chart


# ------------------------------------------------------------------ lexicon v3
def test_lexicon_v3_adds_traceable_esco_entries_sorted_into_technology_and_knowledge():
    from fivetag import build_lexicon_v3 as B3
    from fivetag import lexicon as lex
    v2, v3 = lex.load("v2"), lex.load("v3")
    assert v2.words("technologies") < v3.words("technologies")
    assert v2.words("knowledge", "terms") < v3.words("knowledge", "terms")
    assert v2.words("knowledge", "groups") == v3.words("knowledge", "groups")
    assert v2.sha256 != v3.sha256
    folder = lex.LEXICON_ROOT / "v3"
    assert "European Commission" in (folder / "SOURCES.md").read_text(encoding="utf-8")
    assert (folder / "ESCO_DERIVATION.tsv").read_text("utf-8").startswith("esco_uri")
    assert "parent_code" in (folder / "ESCO_ICT.tsv").read_text("utf-8").splitlines()[0]
    assert B3.split_title("Python (computer programming)") == ("Python", "computer programming")
    assert B3.is_name("Oracle Relational Database") and B3.is_name("JavaScript")
    for generic in ("database quality standards", "ICT network routing", "CAD software",
                    "computer science", "NoSQL"):
        assert not B3.is_name(generic), generic
    row = {"title": "database quality standards", "alt_en": [], "uri": "u", "parent_code": "0612"}
    assert B3.derive(row) == [("knowledge", "database quality standards",
                               "title, common-noun phrase")]
    for noise in ("ml", "saas", "less", "computer science"):
        assert noise not in v3.words("technologies") and noise not in v3.words("knowledge", "terms")
    from competency import temporal as TT
    assert TT.esco_parents("v3")                              # the taxonomy the laddering uses


# ------------------------------------------------------------------ lexicon v2
def test_lexicon_v2_is_v1_plus_traceable_onet_entries():
    from fivetag import build_lexicon_v2 as B
    from fivetag import lexicon as lex
    v1, v2 = lex.load("v1"), lex.load("v2")
    assert v1.sha256 != v2.sha256
    t1, t2 = v1.words("technologies"), v2.words("technologies")
    assert t1 < t2 and len(t2) - len(t1) >= 100
    for name in ("soft_skills", "work_behaviors"):
        assert v1.words(name) == v2.words(name)
    assert {"ec2", "power automate", "active directory"} <= t2
    for generic in ("web services", "analytics", "cloud", "database", "operating system",
                    "web application framework", "nosql"):
        assert generic not in t2, generic
    folder = lex.LEXICON_ROOT / "v2"
    assert "CC BY 4.0" in (folder / "SOURCES.md").read_text(encoding="utf-8")
    trace = (folder / "ONET_DERIVATION.tsv").read_text(encoding="utf-8").splitlines()
    assert trace[0].startswith("onet_example") and len(trace) > 150
    assert [e for e, _ in B.derive("Cascading style sheets CSS")] == ["Cascading style sheets",
                                                                      "CSS !cs"]
    assert [e for e, _ in B.derive("Microsoft Power BI")] == ["Microsoft Power BI", "Power BI"]
    assert [e for e, _ in B.derive("Atlassian JIRA")] == ["Atlassian JIRA", "JIRA !cs"]
    assert B.derive("C") == [] and B.is_generic_kind("Web application framework")
    assert not B.is_generic_kind("Cascading style sheets CSS")


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
