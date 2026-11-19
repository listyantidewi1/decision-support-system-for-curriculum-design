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
def test_groups_are_built_from_sh_kn_ss_and_never_from_technology_or_behaviour():
    ev, sents = _evidence(6)
    groups, report = group_evidence(ev, 6, clusterer=_key_clusterer)
    assert {g.label for g in groups} == {"develop rest apis", "data structures", "teamwork"}
    assert {g.stream for g in groups} == {"technical", "soft"}
    for g in groups:
        assert set(g.tag_counts) <= {"SH", "KN", "SS"}
    assert report["n_evidence_groupable"] == 24 == report["n_evidence_grouped"]
    assert GroupingConfig().min_postings_per_phrase == 1


def test_technology_is_attached_by_sharing_a_sentence():
    ev, _ = _evidence(6)
    groups, _ = group_evidence(ev, 6, clusterer=_key_clusterer)
    by = {g.label: g for g in groups}
    assert {t["text"] for t in by["develop rest apis"].technologies} == {"Spring Boot", "Java 17"}
    assert by["develop rest apis"].technologies[0]["n_postings"] == 6
    assert by["data structures"].technologies == [] and by["teamwork"].technologies == []
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
                 [{"text": k, "n_postings": 3} for k in knowledge], VALID)


def _m(text, tag, n=5):
    return {"text": text, "tag": tag, "n_evidence": n, "n_postings": n}


def test_deterministic_statement_uses_the_postings_own_verb_object_and_technologies():
    st = S.deterministic_statement(_group([_m("Develop REST APIs", "SH", 6),
                                           _m("developing REST APIs", "SH", 2)],
                                          ["Spring Boot", "Java"]))
    assert st.text == "Develop REST APIs using Spring Boot and Java"
    assert (st.verb, st.krathwohl, st.language, st.verb_source) == ("develop", "create", "en",
                                                                    "evidence")
    assert st.grounding == 1.0 and st.ungrounded == []


def test_a_knowledge_term_that_starts_like_a_verb_is_not_read_as_an_action():
    st = S.deterministic_statement(_group([_m("design patterns", "KN")], ["Java"]))
    assert st.text == "Explain design patterns" and st.verb_source == "default"
    assert st.krathwohl == "understand" and st.technologies == []


def test_default_verbs_follow_the_tag_and_the_language():
    assert S.deterministic_statement(_group([_m("unit testing", "SH")], ["Jest"])).text \
        == "Perform unit testing using Jest"
    assert S.deterministic_statement(_group([_m("teamwork", "SS")], stream="soft")).text \
        == "Demonstrate teamwork"
    st = S.deterministic_statement(_group([_m("Membangun aplikasi web", "SH")], ["Laravel"]))
    assert st.text == "Membangun aplikasi web menggunakan Laravel" and st.language == "id"
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


def test_verb_table_is_versioned_and_marked_unreviewed():
    assert "unreviewed" in S.VERB_TABLE_VERSION
    assert set(S.VERBS.values()) <= {"remember", "understand", "apply", "analyze", "evaluate",
                                     "create"}
    assert S.base_verb("Developing") == "develop" and S.base_verb("banana") is None


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
            "evidence.parquet",
            "technology_profile.csv", "work_behaviour_profile.csv"}
        comps = json.loads((out / "competencies.json").read_text(encoding="utf-8"))
        assert report["notice"].startswith("ENGINEERING RUN")
        assert report["n_evidence_by_tag"]["TT"] == 12 and report["n_competencies"] == len(comps) == 3
        c = next(c for c in comps if c["statement"]["verb"] == "develop")
        assert c["statement"]["text"] == "Develop REST APIs using Java 17 and Spring Boot"
        assert c["contract"]["status"] == STATUS_EXPLORATORY
        assert c["contract"]["blockers"] == ["okupasi_mapping_not_run"]
        assert len(c["evidence"]["evidence_ids"]) == 12 and c["evidence"]["samples"]
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
