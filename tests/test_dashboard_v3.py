"""Tests for dashboard_v3/ — both dashboards are run headless on invented data.

Standalone runner (the repo has no pytest):

    .venv-ml\\Scripts\\python.exe tests\\test_dashboard_v3.py

Streamlit's own test harness runs each page as a script. COMPETENCY_DATA_ROOT
points the dashboards at a temporary folder, so nothing under DATA/ is read or
written. No embedding model, no network.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest                                   # noqa: E402

from competency import profile as PF                                       # noqa: E402

PUBLIC = str(PROJECT_ROOT / "dashboard_v3" / "public.py")
ADMIN = str(PROJECT_ROOT / "dashboard_v3" / "admin.py")


def _comp(cid, text, status, n, members):
    return {"competency_id": cid, "stream": "technical", "n_postings": n,
            "share_of_postings": n / 100, "n_posting_groups": n, "group_status": "sahih",
            "n_evidence": n,
            "statement": {"text": text, "verb": text.split()[0].lower(), "krathwohl": "apply",
                          "language": "en", "grounding": 1.0, "notes": []},
            "contract": {"status": status, "blockers": [] if status != "eksploratif" else ["postings<5"]},
            "kkni": {"level": 4, "confident": False, "similarity": 0.3},
            "cognitive": {"level": "create", "code": "C6", "share": 0.9,
                          "n_postings_with_a_verb": 10, "share_of_group_postings": 0.8,
                          "highest": {"level": "create", "code": "C6", "share": 0.9},
                          "levels": [{"level": "create", "code": "C6", "n_postings": 9,
                                      "share": 0.9},
                                     {"level": "apply", "code": "C3", "n_postings": 2,
                                      "share": 0.2}]},
            "okupasi": [{"occupation_en": "Junior Programmer"}],
            "coverage": {"SKKNI": {"proposed": "tercakup_sebagian", "best_similarity": 0.5,
                                   "validated_by_expert": False,
                                   "matches": [{"ref_id": "J.1", "title": "Develop software",
                                                "group": "x", "similarity": 0.5}]}},
            "temporal": {"shares": [0.1, 0.2, 0.3], "direction": "naik", "volatility": "sedang",
                         "trend": "tidak_terdeteksi", "persistence_weight": 0.6},
            "evidence": {"group_id": cid, "label": text,
                         "members": [{"text": m, "tag": "SH", "n_postings": n, "n_evidence": n}
                                     for m in members],
                         "technologies": [{"text": "Docker", "n_postings": 3}],
                         "knowledge_context": [], "tag_counts": {"SH": n}, "cohesion": 0.8,
                         "evidence_ids": [], "samples": [{"evidence_id": "e", "text": members[0],
                                                          "tag": "SH",
                                                          "sentence_text": "An invented sentence."}]}}


def _env():
    """A temporary data root with one run and (optionally) one draft profile."""
    root = Path(tempfile.mkdtemp(prefix="dash_"))
    run = root / "runs" / "run_a"
    run.mkdir(parents=True)
    comps = [_comp("c1", "Develop REST APIs", "kandidat_tinjauan", 12, ["develop rest apis"]),
             _comp("c2", "Perform unit testing", "eksploratif", 3, ["unit testing"])]
    (run / "competencies.json").write_text(json.dumps(comps), encoding="utf-8")
    (run / "groups.json").write_text(json.dumps([
        {"group_id": "g1", "status": "sahih", "label": "develop rest apis", "n_postings": 12,
         "cohesion": 0.8, "status_flags": [], "members": comps[0]["evidence"]["members"]},
        {"group_id": "g2", "status": "berfrekuensi_rendah", "label": "x", "n_postings": 2,
         "cohesion": 0.7, "status_flags": ["weak_low_frequency"],
         "members": comps[1]["evidence"]["members"]}]), encoding="utf-8")
    (run / "report.json").write_text(json.dumps({
        "notice": "ENGINEERING RUN: test", "evidence_source": "test", "seconds": 1.0,
        "denominators": {"n_postings": 100, "n_sentences": 900}, "n_evidence": 15,
        "n_evidence_by_tag": {"SH": 15}, "n_competencies": 2,
        "status_distribution": {"kandidat_tinjauan": 1, "eksploratif": 1},
        "grouping": {"n_groups": 2, "status_distribution": {"sahih": 1, "berfrekuensi_rendah": 1},
                     "n_evidence_grouped": 15, "n_evidence_left_ungrouped": 0},
        "statement_method": "deterministic-v1", "verb_table_version": "v", "grounding_version": "g",
        "contract": {"min_postings": 5}}), encoding="utf-8")
    os.environ["COMPETENCY_DATA_ROOT"] = str(root)
    return root, run


def _ok(at) -> None:
    assert not at.exception, [e.value for e in at.exception]


def test_public_dashboard_says_so_when_no_profile_exists():
    _env()
    at = AppTest.from_file(PUBLIC).run(timeout=60)
    _ok(at)
    assert any("Belum ada profil" in i.value for i in at.info)


def test_public_dashboard_shows_a_draft_with_its_warning_and_the_evidence():
    root, run = _env()
    PF.build(run, "2026-1", profiles_dir=root / "profiles")
    at = AppTest.from_file(PUBLIC).run(timeout=60)
    _ok(at)
    assert at.title[0].value == "Profil Kompetensi RPL 2026-1"
    assert any("DRAFT" in w.value and "belum boleh dikutip" in w.value for w in at.warning)
    assert [m.value for m in at.metric][:4] == ["100", "15", "2", "2"]
    text = " ".join(m.value for m in at.markdown)
    assert "Develop REST APIs" in text and "Frasa dari lowongan" in text
    # the level is the one the postings' verbs ask for, with the Bloom code
    assert "Tingkat kognitif yang diminta lowongan: **C6 mencipta**" in text
    assert "90% dari 10 lowongan yang memakai kata kerja" in text
    assert any("An invented sentence." in c.value for c in at.caption)
    assert any("Versi pertama" in i.value for i in at.info)


def test_public_dashboard_draws_the_cluster_map_of_knowledge_and_technology():
    from competency import termmap as TM
    from dashboard_v3 import termmap_view as TV
    root, run = _env()
    rows = []
    for s, terms in enumerate([["React", "TypeScript", "CSS"], ["Python", "pandas", "Docker"]]):
        for p in range(12):
            rows += [(f"p{s}-{p}", "TT", t.lower(), t) for t in terms]
            rows.append((f"p{s}-{p}", "KN", f"topic {s}", f"topic {s}"))
    tm = TM.build(rows, 24, TM.TermMapConfig(n_clusters=2))
    (run / "term_map.json").write_text(json.dumps(tm), encoding="utf-8")
    PF.build(run, "2026-1", profiles_dir=root / "profiles")
    assert (root / "profiles" / "drafts" / "2026-1" / "term_map.json").is_file()
    at = AppTest.from_file(PUBLIC).run(timeout=90)
    _ok(at)
    assert any("8 dari 8 istilah" in c.value and "2 kelompok" in c.value for c in at.caption)
    assert any(s.value == "Kelompok" for s in at.subheader)
    # one cluster in focus: its terms, and the competencies that use them
    at.selectbox(key="termmap_cluster").select(
        next(t["cluster"] for t in tm["terms"] if t["term"] == "Docker")).run(timeout=90)
    _ok(at)
    assert any("Kompetensi yang memakai istilah kelompok ini" in m.value for m in at.markdown)
    # a cluster keeps its colour when a filter removes other clusters' points
    df, domain, colours = TV.frame(tm)
    assert len(domain) == 2 and colours == TV.PALETTE[:2]
    hit = TV.related_competencies([{"statement": {"text": "X"}, "n_postings": 3,
                                    "evidence": {"technologies": [{"text": "docker"}]}}], {"Docker"})
    assert hit == [{"Pernyataan kompetensi": "X", "Lowongan": 3, "Istilah kelompok ini": "docker"}]


def test_public_dashboard_shows_related_competencies_per_level_and_filters_by_level():
    root, run = _env()
    comps = json.loads((run / "competencies.json").read_text(encoding="utf-8"))
    rel = [{"competency_id": "c1", "code": "C6", "level": "create",
            "statement": "Develop websites", "n_postings": 12},
           {"competency_id": "c2", "code": "C3", "level": "apply",
            "statement": "Apply web programming", "n_postings": 3}]
    for c, r in zip(comps, rel):
        c["statement"]["text"] = r["statement"]
        c["cognitive"] = {"level": r["level"], "code": r["code"], "n_postings": r["n_postings"],
                          "share_of_family_postings": r["n_postings"] / 14,
                          "n_family_postings": 14, "n_family_postings_with_a_verb": 14}
        c["family"] = {"id": "f1", "label": "web", "n_postings": 14, "competencies": rel,
                       "description": {"text": "A paragraph for the whole family.",
                                       "method": "v2-generator"}}
    (run / "competencies.json").write_text(json.dumps(comps), encoding="utf-8")
    PF.build(run, "2026-1", profiles_dir=root / "profiles")
    at = AppTest.from_file(PUBLIC).run(timeout=90)
    _ok(at)
    text = " ".join(m.value for m in at.markdown)
    assert "Tingkat kognitif: **C6 mencipta**" in text and "dari 14 lowongan rumpun" in text
    assert "Kompetensi serumpun, per tingkat kognitif" in text
    assert "A paragraph for the whole family." in text
    # an institution keeps only the levels it teaches to
    at.sidebar.multiselect[2].set_value(["C3"]).run(timeout=90)
    _ok(at)
    assert any("1 dari 2 rekomendasi" in c.value for c in at.caption)
    assert "Tingkat kognitif: **C3 menerapkan**" in " ".join(m.value for m in at.markdown)


def test_public_filters_narrow_the_list():
    root, run = _env()
    PF.build(run, "2026-1", profiles_dir=root / "profiles")
    at = AppTest.from_file(PUBLIC).run(timeout=60)
    at.text_input[0].set_value("unit testing").run(timeout=60)
    _ok(at)
    assert any(c.value.startswith("1 dari 2 rekomendasi") for c in at.caption)


def test_admin_pages_render_and_a_profile_can_be_published_as_a_draft():
    root, run = _env()
    at = AppTest.from_file(ADMIN).run(timeout=60)
    _ok(at)
    assert any("ENGINEERING RUN" in w.value for w in at.warning)
    for page in ("Audit kelompok", "Telaah pernyataan", "Log eksperimen", "Koleksi lowongan",
                 "Telaah konflik D1/D2"):
        at.sidebar.radio[0].set_value(page).run(timeout=60)
        _ok(at)
    at.sidebar.radio[0].set_value("Terbitkan profil").run(timeout=60)
    at.text_input[0].set_value("2026-9").run(timeout=60)
    assert at.checkbox[0].value is False                    # release is never the default
    next(b for b in at.button if b.label == "Terbitkan").click().run(timeout=60)
    _ok(at)
    assert (root / "profiles" / "drafts" / "2026-9" / "profile.json").is_file()
    assert not (root / "profiles" / "releases").exists()


def test_admin_saves_statement_reviews_next_to_the_run():
    root, run = _env()
    at = AppTest.from_file(ADMIN).run(timeout=60)
    at.sidebar.radio[0].set_value("Telaah pernyataan").run(timeout=60)
    next(b for b in at.button if b.label == "Simpan telaah").click().run(timeout=60)
    _ok(at)
    saved = (run / "review_statements.csv").read_text(encoding="utf-8-sig")
    assert "Develop REST APIs" in saved and "keputusan" in saved.splitlines()[0]


def test_dashboards_cannot_collect_pay_or_touch_the_gold_standard():
    for name in ("public.py", "admin.py", "common.py"):
        src = (PROJECT_ROOT / "dashboard_v3" / name).read_text(encoding="utf-8")
        for banned in ("capped_caller", "run_llm", "call_model", "gold_standard", "legacy",
                       '"--go"', "'--go'", "requests.", "pipeline_orchestrator"):
            assert banned not in src, (name, banned)
    admin = (PROJECT_ROOT / "dashboard_v3" / "admin.py").read_text(encoding="utf-8")
    assert '"acquisition.monthly", "status"' in admin


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
