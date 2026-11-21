"""Tests for competency.qualifiers (how well / which standards / outcome, read
from a competency's own postings) and competency.formulate (competency
statements formulated the way SKKNI unit titles are, with SKKNI wording only
where a competency IS an SKKNI unit on the same level).
Every sentence, phrase and SKKNI unit here is invented; no model is called.

Run:  .venv-ml\\Scripts\\python.exe tests\\test_formulate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from competency import formulate as F                                         # noqa: E402
from competency import qualifiers as Q                                        # noqa: E402


def _competency(level="create", code="C6", statement="Develop web applications", n=10):
    return {
        "competency_id": f"comp-f1@{code}", "n_postings": n,
        "statement": {"text": statement, "object": "web applications"},
        "contract": {"status": "kandidat_tinjauan"},
        "cognitive": {"level": level, "code": code},
        "evidence": {
            "verbs": [{"verb": "develop", "n_postings": 8}, {"verb": "build", "n_postings": 4}],
            "knowledge_objects": [{"text": "web applications", "n_postings": 9, "share": 0.9}],
            "technologies": [{"text": "React", "n_postings": 6}, {"text": "Node.js", "n_postings": 3},
                             {"text": "SQL", "n_postings": 3}],
            "knowledge": [], "work_behaviours": [], "soft_skills": [],
            "members": [{"text": "Develop scalable web applications"}],
            "samples": [{"sentence_text": "Develop scalable, secure web applications following "
                                         "best practices to improve performance."}]}}


def _rows():
    s1 = "Develop scalable, secure web applications following best practices to improve performance."
    s2 = "Build secure web applications according to OWASP guidelines."
    s3 = "Develop web applications. Our office offers a scalable career."
    rows = []
    for p in range(3):
        rows.append({"text": "Develop scalable, secure web applications", "posting_id": f"p{p}",
                     "sentence_text": s1, "start": 0,
                     "end": len("Develop scalable, secure web applications")})
    for p in range(3, 5):
        rows.append({"text": "Build secure web applications", "posting_id": f"p{p}",
                     "sentence_text": s2, "start": 0, "end": len("Build secure web applications")})
    rows.append({"text": "Develop web applications", "posting_id": "p5", "sentence_text": s3,
                 "start": 0, "end": len("Develop web applications")})
    return rows


# ------------------------------------------------------------------ qualifiers
def test_the_profile_reads_how_well_which_standards_and_what_outcome_from_the_postings():
    prof = Q.profile(_competency(), _rows())
    q = {r["text"]: (r["n_postings"], r["iso25010"]) for r in prof["qualities"]}
    assert q == {"secure": (5, "security"), "scalable": (3, "performance efficiency")}
    s = {r["text"]: r["n_postings"] for r in prof["standards"]}
    assert s["best practices"] == 3 and s["OWASP"] == 2
    assert [(r["text"], r["n_postings"]) for r in prof["outcomes"]] == [("improve performance", 3)]
    assert "career" not in json.dumps(prof["qualities"])        # after the end of the sentence
    own = Q.profile(_competency(statement="Perform code reviews"),
                    [{"text": "Perform code reviews", "posting_id": "p1", "start": 0, "end": 20,
                      "sentence_text": "Perform code reviews."},
                     {"text": "Perform code reviews", "posting_id": "p2", "start": 0, "end": 20,
                      "sentence_text": "Perform code reviews."}])
    assert own["standards"] == []                                # it only restates the competency


# --------------------------------------------------------------------- slots
def test_qualifiers_enter_a_statement_only_when_enough_postings_ask_for_them():
    prof = {"standards": [
        {"id": "S1", "text": "following: company guidelines", "n_postings": 9, "share": 0.30},
        {"id": "S2", "text": "REST principles", "n_postings": 6, "share": 0.20},
        {"id": "S3", "text": "best practices", "n_postings": 4, "share": 0.13},
        {"id": "S4", "text": "OWASP", "n_postings": 1, "share": 0.03}],
        "qualities": [
            {"id": "Q1", "text": "high-quality", "n_postings": 12, "share": 0.4, "iso25010": "general"},
            {"id": "Q2", "text": "scalable", "n_postings": 8, "share": 0.27,
             "iso25010": "performance efficiency"},
            {"id": "Q3", "text": "scalability", "n_postings": 5, "share": 0.17,
             "iso25010": "performance efficiency"},
            {"id": "Q4", "text": "secure", "n_postings": 5, "share": 0.17, "iso25010": "security"},
            {"id": "Q5", "text": "reliable", "n_postings": 4, "share": 0.13, "iso25010": "reliability"}],
        "outcomes": [{"id": "R1", "text": "ensure code quality", "n_postings": 2, "share": 0.07},
                     {"id": "R2", "text": "improve performance", "n_postings": 6, "share": 0.2}]}
    got = F.choose_qualifiers(prof)
    # named standards first; "high-quality" is too vague; one word per characteristic;
    # an outcome needs 5 postings (R1 has 2): the first outcome that has them enters
    assert [(q["id"], q["kind"]) for q in got] == [("S2", "standard"), ("S3", "standard"),
                                                  ("Q2", "quality"), ("Q4", "quality"),
                                                  ("R2", "outcome")]
    prof["outcomes"] = [prof["outcomes"][0]]
    assert [q["kind"] for q in F.choose_qualifiers(prof)].count("outcome") == 0
    strict = F.choose_qualifiers(prof, F.FormulateConfig(min_share=0.25, max_standards=1))
    assert [q["id"] for q in strict] == ["S1", "Q2"]           # the generic phrase is all there is


def test_the_writer_is_given_the_verbs_on_the_level_or_the_neutral_verbs_without_one():
    idn, en = F.allowed_verbs("create")
    assert "mengembangkan" in idn and "membangun" in idn and "memelihara" not in idn
    assert "develop" in en and "maintain" not in en
    assert F.allowed_verbs(None) == (["melakukan", "melaksanakan"], ["perform", "carry out"])


# ------------------------------------------------------------ SKKNI candidates
UNITS = [
    {"code": "J.620100.005.01", "title": "Mengimplementasikan User Interface",
     "description": "Unit ini menentukan kompetensi untuk mengimplementasikan user interface.",
     "elements": [{"title": "Menerapkan rancangan user interface"}], "decree_year": "2016",
     "source": "Software Development Pemrograman.pdf"},
    {"code": "J.620100.032.01", "title": "Menerapkan Code Review",
     "description": "Unit ini menentukan kompetensi untuk menerapkan code review.",
     "elements": [{"title": "Melakukan code review"}], "decree_year": "2016",
     "source": "Software Development Pemrograman.pdf"},
    {"code": "J.62SAD00.001.1", "title": "Mengembangkan Aplikasi Web",
     "description": "Unit ini berkaitan dengan pengembangan aplikasi web.", "elements": [],
     "decree_year": "2017", "source": "Software Requirement Analysis and Design.pdf"},
]


def _embed(texts):
    """A stand-in embedder: bag of a few words, enough to rank the invented units."""
    vocab = ["interface", "review", "web", "aplikasi", "applications", "code", "kode"]
    return [[t.lower().count(w) + 0.01 for w in vocab] for t in texts]


def test_the_nearest_skkni_units_are_found_by_their_title_description_and_elements():
    idx = F.SkkniIndex(UNITS, embed=_embed)
    top = idx.nearest("Develop web applications. web applications. aplikasi web", 2)
    assert top[0][0]["code"] == "J.62SAD00.001.1" and len(top) == 2
    assert F.candidate_rows(top)[0]["judul"] == "Mengembangkan Aplikasi Web"


def _answer(**over):
    a = {"pernyataan": "Mengembangkan aplikasi web yang aman sesuai best practices",
         "statement_en": "Develop secure web applications following best practices",
         "kualifikasi_dipakai": ["S1", "Q1"], "alasan_tanpa_kualifikasi": "",
         "skkni": {"kode": None, "hubungan": "tidak", "alasan": ""}, "pernyataan_skkni": None}
    a.update(over)
    return json.dumps(a)


def test_the_checker_accepts_a_grounded_statement_and_names_every_problem_of_a_bad_one():
    prof = Q.profile(_competency(), _rows())
    chosen = F.choose_qualifiers(prof, F.FormulateConfig(min_share=0.2, min_postings=3,
                                                         max_qualities=1, max_standards=1))
    assert [q["text"] for q in chosen] == ["best practices", "secure"]
    ids = {q["text"]: q["id"] for q in chosen}
    cands = F.SkkniIndex(UNITS, embed=_embed).nearest("web aplikasi", 3)
    good = _answer(kualifikasi_dipakai=[ids["best practices"], ids["secure"]])
    a, problems = F.check(good, prof, chosen, cands)
    assert problems == [] and a is not None
    bad = _answer(pernyataan="Memelihara aplikasi web dengan React dalam 3 hari",   # C3, tool, number
                  statement_en="Develop scalable web applications using OWASP",     # quality, standard
                  kualifikasi_dipakai=["S9"])
    a, problems = F.check(bad, prof, chosen, cands)
    text = " | ".join(problems)
    assert a is None
    assert "does not open with a verb on the level 'create'" in text
    assert "names the tool 'React'" in text and "states the number 3" in text
    assert "adds the quality 'scalable'" in text and "adds the standard 'OWASP'" in text
    assert "['S9']" in text
    a, problems = F.check(_answer(pernyataan="Mengembangkan aplikasi web yang skalabel"), prof,
                          chosen, cands)
    assert any("adds the quality 'skalabel'" in p for p in problems)          # Indonesian too
    # ordinary English words have an Indonesian form; established terms may stay
    a, problems = F.check(_answer(pernyataan="Mengembangkan web applications yang aman sesuai "
                                             "best practices"), prof, chosen, cands)
    assert any("mixes English words" in p and "applications" in p for p in problems)
    a, problems = F.check(_answer(pernyataan="mengembangkan user interface aplikasi web yang aman "
                                             "sesuai best practices",
                                  kualifikasi_dipakai=[ids["best practices"], ids["secure"]]),
                          prof, chosen, cands)
    assert problems == [] and a["pernyataan"].startswith("Mengembangkan")      # capitalised
    # slips in the SKKNI fields do not cost the statement: the link takes the cautious reading
    q = [ids["best practices"], ids["secure"]]
    outside = _answer(skkni={"kode": "J.000000.001.01", "hubungan": "sama", "alasan": ""},
                      pernyataan_skkni="Mengembangkan Aplikasi Web", kualifikasi_dipakai=q)
    a, problems = F.check(outside, prof, chosen, cands)
    assert problems == [] and a["skkni"]["hubungan"] == "tidak" and a["skkni"]["kode"] is None
    assert "tanpa kode kandidat" in a["skkni"]["catatan"][0]
    nearest = _answer(skkni={"kode": "J.62SAD00.001.1", "hubungan": "tidak", "alasan": "terdekat"},
                      kualifikasi_dipakai=q)
    a, problems = F.check(nearest, prof, chosen, cands)
    assert problems == [] and a["skkni"]["kode"] is None and "dibuang" in a["skkni"]["catatan"][0]
    no_code = _answer(skkni={"kode": None, "hubungan": "sebagian", "alasan": ""}, kualifikasi_dipakai=q)
    a, problems = F.check(no_code, prof, chosen, cands)
    assert problems == [] and a["skkni"]["hubungan"] == "tidak"
    reworded = _answer(skkni={"kode": "J.62SAD00.001.1", "hubungan": "sama", "alasan": "sama"},
                       pernyataan_skkni="Membangun aplikasi berbasis web",
                       pernyataan_skkni_en="Develop web applications")
    assert any("must begin with the SKKNI title" in p for p in F.check(reworded, prof, chosen, cands)[1])
    glued = _answer(skkni={"kode": "J.62SAD00.001.1", "hubungan": "sama", "alasan": "sama"},
                    pernyataan_skkni="Mengembangkan Aplikasi Web yang aman",
                    pernyataan_skkni_en="Develop secure web applications",
                    kualifikasi_dipakai=[ids["best practices"], ids["secure"]])
    assert any("no quality may be appended" in p for p in F.check(glued, prof, chosen, cands)[1])
    standard = _answer(skkni={"kode": "J.62SAD00.001.1", "hubungan": "sama", "alasan": "sama"},
                       pernyataan_skkni="Mengembangkan Aplikasi Web sesuai best practices",
                       pernyataan_skkni_en="Develop web applications following best practices",
                       kualifikasi_dipakai=[ids["best practices"], ids["secure"]])
    assert F.check(standard, prof, chosen, cands)[1] == []
    no_en = _answer(skkni={"kode": "J.62SAD00.001.1", "hubungan": "sama", "alasan": "sama"},
                    pernyataan_skkni="Mengembangkan Aplikasi Web")
    assert any("needs pernyataan_skkni_en" in p for p in F.check(no_en, prof, chosen, cands)[1])
    # the postings' own verb, not a stock verb of the same level
    swapped = _answer(pernyataan="Merancang aplikasi web yang aman sesuai best practices",
                      statement_en="Design secure web applications following best practices",
                      kualifikasi_dipakai=[ids["best practices"], ids["secure"]])
    assert any("does not use a verb the postings use" in p
               for p in F.check(swapped, prof, chosen, cands)[1])
    assert F.check("not json", prof, chosen, cands) == (None, ["the answer is not JSON"])


def test_a_competency_without_a_level_keeps_a_neutral_verb():
    prof = Q.profile(_competency(level=None, code=None, statement="Perform web programming"), _rows())
    cands = F.SkkniIndex(UNITS, embed=_embed).nearest("web", 2)
    ok = _answer(pernyataan="Melakukan pemrograman web", statement_en="Perform web programming",
                 kualifikasi_dipakai=[])
    assert F.check(ok, prof, [], cands)[1] == []
    _, problems = F.check(_answer(kualifikasi_dipakai=[]), prof, [], cands)
    assert any("must open with melakukan or melaksanakan" in p for p in problems)
    # a named practice keeps its English name
    trunk = _answer(pernyataan="Melakukan trunk-based development",
                    statement_en="Perform trunk-based development", kualifikasi_dipakai=[])
    assert F.check(trunk, prof, [], cands)[1] == []


def test_skkni_wording_is_used_only_for_the_same_competency_on_the_same_level():
    cands = F.SkkniIndex(UNITS, embed=_embed).nearest("web aplikasi review code interface", 3)
    prof = {"level": "create", "statement": "Develop web applications"}
    same = json.loads(_answer(skkni={"kode": "J.62SAD00.001.1", "hubungan": "sama", "alasan": "x"},
                              pernyataan_skkni="Mengembangkan Aplikasi Web",
                              pernyataan_skkni_en="Develop web applications"))
    w = F.decide_wording(same, prof, cands)
    assert w["pernyataan"] == "Mengembangkan aplikasi web"         # the title, in sentence case
    assert w["statement_en"] == "Develop web applications"                 # ITS English
    assert w["sumber_rumusan"] == "SKKNI J.62SAD00.001.1" and w["skkni"]["tingkat_sama"]
    with_std = json.loads(_answer(skkni={"kode": "J.62SAD00.001.1", "hubungan": "sama", "alasan": "x"},
                                  pernyataan_skkni="Mengembangkan Aplikasi Web sesuai best practices",
                                  pernyataan_skkni_en="Develop web applications following best practices"))
    w = F.decide_wording(with_std, prof, cands)
    assert w["pernyataan"] == "Mengembangkan aplikasi web sesuai best practices"
    assert w["sumber_rumusan"] == "SKKNI J.62SAD00.001.1 + pasar"
    assert F.sentence_case("Melaksanakan Pengujian Unit Program dalam alur CI/CD dan API") == \
        "Melaksanakan pengujian unit program dalam alur CI/CD dan API"
    assert F.sentence_case("memantau sistem IoT") == "Memantau sistem IoT"
    assert w["pernyataan_pasar"].startswith("Mengembangkan aplikasi web")
    # the market asks to REVIEW code (C5); the standard's title applies code review (C3)
    review = {"level": "evaluate", "statement": "Review code"}
    a = json.loads(_answer(pernyataan="Menelaah kode program", statement_en="Review code",
                           skkni={"kode": "J.620100.032.01", "hubungan": "sama", "alasan": "x"},
                           pernyataan_skkni="Menerapkan Code Review"))
    w = F.decide_wording(a, review, cands)
    assert w["pernyataan"] == "Menelaah kode program" and w["sumber_rumusan"] == "pasar"
    assert w["skkni"]["kode_tingkat_judul"] == "C3" and w["skkni"]["tingkat_sama"] is False
    part = json.loads(_answer(skkni={"kode": "J.620100.005.01", "hubungan": "sebagian", "alasan": "x"}))
    w = F.decide_wording(part, prof, cands)
    assert w["sumber_rumusan"] == "pasar" and w["skkni"]["hubungan"] == "sebagian"


def _plain(**over):
    return _answer(pernyataan="Mengembangkan aplikasi web",
                   statement_en="Develop web applications", kualifikasi_dipakai=[], **over)


def test_a_statement_is_written_checked_retried_once_and_a_general_one_is_flagged():
    c = _competency()
    idx = F.SkkniIndex(UNITS, embed=_embed)
    # first answer: wrong level; second: correct, and it adds no quality or standard,
    # because with min_share 0.9 none was given
    answers = [_answer(pernyataan="Memelihara aplikasi web", kualifikasi_dipakai=[]), _plain()]
    seen = []

    def call(messages):
        seen.append(messages)
        return answers.pop(0)
    cfg = F.FormulateConfig(min_share=0.9)                    # no qualifier reaches 90%
    r = F.formulate_one(c, _rows(), idx, call, cfg)
    assert r["accepted"] and len(r["attempts"]) == 2 and "DITOLAK" in seen[1][1]["content"]
    payload = json.loads(seen[0][1]["content"])
    assert payload["contoh_gaya_skkni"][0] == "Menggunakan struktur data"
    assert payload["tingkat_kognitif"] == "C6 create" and payload["kualifikasi"] == []
    assert r["pernyataan"] == "Mengembangkan aplikasi web" and r["kualifikasi"] == []
    assert any("adds the quality 'secure'" in p for p in r["attempts"][0]["problems"])
    assert r["sumber_rumusan"] == "pasar" and r["kandidat_skkni"][0]["kode"] == "J.62SAD00.001.1"
    general = F.formulate_one({**c, "statement": {"text": "Develop applications",
                                                  "object": "applications"}},
                              _rows(), idx, lambda m: _plain(), cfg)
    assert any(f.startswith("terlalu_umum") for f in general["flags"])
    # a general object stays general when the postings add a quality ("solusi yang skalabel")
    with_q = F.formulate_one({**c, "statement": {"text": "Develop applications",
                                                 "object": "applications"}},
                             _rows(), idx, lambda m: _plain(), F.FormulateConfig(min_share=0.2))
    assert any(f.startswith("terlalu_umum") for f in with_q["flags"])
    never = F.formulate_one(c, _rows(), idx, lambda m: "{}", cfg)
    assert not never["accepted"] and never["problems"] and len(never["attempts"]) == 2


def test_skkni_wording_needs_a_second_model_to_agree_that_the_competency_is_the_same():
    c = _competency()
    idx = F.SkkniIndex(UNITS, embed=_embed)
    cfg = F.FormulateConfig(min_share=0.9)
    same = _plain(skkni={"kode": "J.62SAD00.001.1", "hubungan": "sama", "alasan": "x"},
                  pernyataan_skkni="Mengembangkan Aplikasi Web",
                  pernyataan_skkni_en="Develop web applications")
    seen = []

    def confirm_with(answer):
        def confirm(messages):
            seen.append(messages)
            return answer
        return confirm
    ok = F.formulate_one(c, _rows(), idx, lambda m: same, cfg,
                         confirm=confirm_with('{"hubungan": "sama", "alasan": "y"}'))
    assert ok["sumber_rumusan"] == "SKKNI J.62SAD00.001.1" and ok["skkni"]["dikonfirmasi"]
    # the second model sees the competency and the one unit, nothing the writer decided
    payload = json.loads(seen[0][1]["content"])
    assert set(payload) == {"kompetensi", "unit_skkni"} and payload["unit_skkni"]["kode"] == "J.62SAD00.001.1"
    assert "kemiripan" not in payload["unit_skkni"] and "pernyataan" not in payload["kompetensi"]
    # not confirmed: the market's statement stays and the relation becomes 'sebagian'
    no = F.formulate_one(c, _rows(), idx, lambda m: same, cfg,
                         confirm=confirm_with('{"hubungan": "sebagian", "alasan": "scope"}'))
    assert no["sumber_rumusan"] == "pasar" and no["pernyataan"] == "Mengembangkan aplikasi web"
    assert no["skkni"]["hubungan"] == "sebagian" and no["skkni"]["hubungan_penulis"] == "sama"
    assert no["statement_en"] == "Develop web applications" and "statement_en_pasar" not in no
    # an unreadable answer, or no second model at all, confirms nothing
    garbage = F.formulate_one(c, _rows(), idx, lambda m: same, cfg, confirm=confirm_with("not json"))
    assert garbage["skkni"]["hubungan"] == "sebagian" and garbage["skkni"]["konfirmasi"]["masalah"]
    alone = F.formulate_one(c, _rows(), idx, lambda m: same, cfg)
    assert alone["sumber_rumusan"] == "pasar" and alone["skkni"]["dikonfirmasi"] is False
    # only a 'sama' is sent for confirmation
    part = _plain(skkni={"kode": "J.62SAD00.001.1", "hubungan": "sebagian", "alasan": "x"})

    def never(messages):
        raise AssertionError("a 'sebagian' must not be sent for confirmation")
    r = F.formulate_one(c, _rows(), idx, lambda m: part, cfg, confirm=never)
    assert r["skkni"]["hubungan"] == "sebagian" and "dikonfirmasi" not in r["skkni"]
    # one definition of 'sama' for the writer and the confirmation
    assert F.SAME_RULE in F.SYSTEM and F.SAME_RULE in F.CONFIRM_SYSTEM
    # an answer cut off by the token allowance is asked once more; twice is not a confirmation
    answers = ["", '{"hubungan": "sama", "alasan": "y"}']
    again = F.formulate_one(c, _rows(), idx, lambda m: same, cfg, confirm=lambda m: answers.pop(0))
    assert again["skkni"]["dikonfirmasi"] and again["skkni"]["konfirmasi"]["percobaan"] == 2
    twice = F.formulate_one(c, _rows(), idx, lambda m: same, cfg, confirm=lambda m: "")
    assert not twice["skkni"]["dikonfirmasi"] and twice["skkni"]["konfirmasi"]["percobaan"] == 2
    assert again["version"] == F.FORMULATE_VERSION == twice["version"]


def test_a_slip_in_the_skkni_fields_is_asked_again_and_the_first_answer_kept_if_the_second_is_worse():
    c = _competency()
    idx = F.SkkniIndex(UNITS, embed=_embed)
    cfg = F.FormulateConfig(min_share=0.9)
    slip = _plain(skkni={"kode": "J.62SAD00.001.1", "hubungan": "tidak", "alasan": "terdekat"})
    seen = []

    def answers(*xs):
        xs = list(xs)

        def call(messages):
            seen.append(messages)
            return xs.pop(0)
        return call
    r = F.formulate_one(c, _rows(), idx, answers(slip, _plain()), cfg)
    assert r["accepted"] and r["attempts"][0]["disimpan"] and "catatan_skkni" not in r
    assert "must be null when hubungan is 'tidak'" in seen[1][1]["content"]
    worse = _answer(pernyataan="Memelihara aplikasi web", kualifikasi_dipakai=[])   # wrong level
    r = F.formulate_one(c, _rows(), idx, answers(slip, worse), cfg)
    assert r["accepted"] and r["attempts"][1]["dipakai_jawaban"] == 1
    assert r["pernyataan"] == "Mengembangkan aplikasi web" and "dibuang" in r["catatan_skkni"][0]
    assert r["skkni"] is None                                       # 'tidak': no link is shown


def test_a_fitting_qualifier_is_written_and_what_is_not_work_or_mixed_is_marked():
    """The researcher's decisions of 2026-11-21: a statement names the standard or
    quality its postings ask for; what is not work, or mixes works, is marked."""
    prof = Q.profile(_competency(), _rows())
    chosen = F.choose_qualifiers(prof, F.FormulateConfig(min_share=0.2, min_postings=3,
                                                         max_qualities=1, max_standards=1))
    cands = F.SkkniIndex(UNITS, embed=_embed).nearest("web aplikasi", 3)
    ids = {q["text"]: q["id"] for q in chosen}
    bare = _plain()                                            # qualifiers offered, none used
    assert any("none is used" in p for p in F.check(bare, prof, chosen, cands)[1])
    why = _plain(alasan_tanpa_kualifikasi="mutu dan standar tidak menerangkan objek ini")
    assert F.check(why, prof, chosen, cands)[1] == []
    # a qualifier listed as used must be written, in English or in Indonesian
    claimed = _answer(pernyataan="Mengembangkan aplikasi web", statement_en="Develop web applications",
                      kualifikasi_dipakai=[ids["secure"]])                  # "aman" is not there
    assert any("does not say it" in p for p in F.check(claimed, prof, chosen, cands)[1])
    as_std = _answer(pernyataan="Mengembangkan aplikasi web sesuai prinsip aman",
                     statement_en="Develop web applications according to secure principles",
                     kualifikasi_dipakai=[ids["secure"]])
    assert any("writes a quality as if it were a standard" in p
               for p in F.check(as_std, prof, chosen, cands)[1])
    written = _answer(pernyataan="Mengembangkan aplikasi web yang aman", kualifikasi_dipakai=[ids["secure"]],
                      statement_en="Develop secure web applications")
    assert F.check(written, prof, chosen, cands)[1] == []
    std = {"id": "S9", "kind": "standard", "text": "design patterns"}
    assert F._says(std, "Merancang aplikasi sesuai pola desain") and not F._says(std, "Merancang aplikasi")
    assert F._says({"id": "O1", "kind": "outcome", "text": "improve performance"},
                   "Mengoptimalkan kueri untuk meningkatkan kinerja")
    # an Indonesian rendering of a standard the slots did not give is refused too
    added = _answer(pernyataan="Mengembangkan aplikasi web sesuai pola desain",
                    statement_en="Develop web applications", kualifikasi_dipakai=[],
                    alasan_tanpa_kualifikasi="x")
    assert any("adds the standard 'design patterns'" in p for p in F.check(added, prof, chosen, cands)[1])
    # MARKS: a focused second model, shown only the evidence, marks what is not work or mixed
    c = _competency()
    idx = F.SkkniIndex(UNITS, embed=_embed)
    cfg = F.FormulateConfig(min_share=0.9)
    seen = []

    def marker(answer):
        def mark(messages):
            seen.append(messages)
            return answer
        return mark

    def never(messages):
        raise AssertionError("what is not work needs no confirmation of an SKKNI unit")
    linked = _plain(skkni={"kode": "J.62SAD00.001.1", "hubungan": "sama", "alasan": "x"},
                    pernyataan_skkni="Mengembangkan Aplikasi Web",
                    pernyataan_skkni_en="Develop web applications")
    r = F.formulate_one(c, _rows(), idx, lambda m: linked, cfg, confirm=never,
                        mark=marker('{"bukan_pekerjaan": true, "keluarga_campuran": false, '
                                    '"alasan": "slogan perusahaan"}'))
    # marked, not left out: the statement is written; what is not work keeps no SKKNI link
    assert r["accepted"] and r["pernyataan"] == "Mengembangkan aplikasi web" and r["skkni"] is None
    assert r["sumber_rumusan"] == "pasar" and "tidak dipakai" in r["catatan_skkni"][-1]
    assert "bukan_pekerjaan: slogan perusahaan" in r["flags"] and r["tanda"]["bukan_pekerjaan"]
    payload = json.loads(seen[0][1]["content"])
    assert "pernyataan" not in payload and "kandidat_skkni" not in payload
    assert 0 < payload["porsi_objek_utama"] <= 1 and payload["contoh_kalimat"]
    mixed = F.formulate_one(c, _rows(), idx, lambda m: _plain(), cfg,
                            mark=marker('{"bukan_pekerjaan": false, "keluarga_campuran": true, '
                                        '"alasan": "RPA hanya 7 dari 101 lowongan"}'))
    assert "keluarga_campuran: RPA hanya 7 dari 101 lowongan" in mixed["flags"]
    assert mixed["version"] == "formulate-v5"
    # an unreadable mark marks nothing and says so; a refused competency is marked as well
    lost = F.formulate_one(c, _rows(), idx, lambda m: "{}", cfg, mark=marker("not json"))
    assert not lost["accepted"] and lost["tanda"]["masalah"] and lost["tanda"]["percobaan"] == 2
    assert not any(f.startswith(("bukan_pekerjaan", "keluarga_campuran")) for f in lost["flags"])
    # the writer writes; marking is not its task
    assert "bukan_pekerjaan" not in F.SYSTEM and "bukan_pekerjaan" in F.MARK_SYSTEM
    # one employer's sentence repeated across postings is marked without a model
    same = [{"text": "Support families", "posting_id": f"p{i}", "start": 0, "end": 16,
             "sentence_text": "We support our employees and their families."} for i in range(6)]
    assert F.repeated_sentences(same, F.FormulateConfig()).startswith("kalimat_berulang: 6 lowongan")
    assert F.repeated_sentences(_rows(), F.FormulateConfig()) is None       # 6 postings, 3 sentences
    assert F.repeated_sentences(same[:4], F.FormulateConfig()) is None      # too few postings
    assert "alasan_tanpa_kualifikasi" in F.SYSTEM
    # the English of an SKKNI title is in sentence case, like every other statement
    same_unit = json.loads(_answer(skkni={"kode": "J.62SAD00.001.1", "hubungan": "sama", "alasan": "x"},
                                   pernyataan_skkni="Mengembangkan Aplikasi Web",
                                   pernyataan_skkni_en="Develop Web Applications"))
    w = F.decide_wording(same_unit, {"level": "create", "statement": "Develop web applications"}, cands)
    assert w["statement_en"] == "Develop web applications"


def test_a_statement_is_carried_over_only_when_the_formulation_input_is_the_same():
    """--carry-from (EXP-035): after a re-run of the chain, a competency whose input to
    the writer did not change keeps its statement instead of being paid for again."""
    prof = Q.profile(_competency(), _rows())
    chosen = F.choose_qualifiers(prof)
    fp = F.input_fingerprint(prof, chosen)
    assert fp == F.input_fingerprint(Q.profile(_competency(), _rows()), F.choose_qualifiers(prof))
    assert len(fp) == 16
    longer = _rows() + [{"text": "Write clean, efficient web applications code", "posting_id": "p9",
                         "sentence_text": "Write clean, efficient web applications code.",
                         "start": 0, "end": 44}]
    c = _competency()
    c["evidence"] = {**c["evidence"], "members": c["evidence"]["members"]
                     + [{"text": "Write clean, efficient web applications code"}]}
    other = Q.profile(c, longer)
    assert F.input_fingerprint(other, F.choose_qualifiers(other)) != fp       # a new phrase
    lower = Q.profile(_competency(level="apply", code="C3"), _rows())
    assert F.input_fingerprint(lower, F.choose_qualifiers(lower)) != fp       # another level
    # the loose fingerprint keeps only what the statement is made of: a new supporting
    # phrase does not change it, another level or other qualifiers do
    loose = F.input_fingerprint(prof, chosen, strict=False)
    assert F.input_fingerprint(other, F.choose_qualifiers(other), strict=False) == loose \
        or F.choose_qualifiers(other) != chosen
    assert F.input_fingerprint(lower, F.choose_qualifiers(lower), strict=False) != loose


def test_the_checker_reads_the_work_before_its_qualifiers_and_the_glossary_verbs():
    from competency.statement import read_phrase
    # the qualifier clause is not the work: 'according to ...' hid 'testing' (C5) behind 'perform'
    assert F.levels_of("Perform API testing according to REST principles") == ["evaluate"]
    assert F.levels_of("Mengembangkan aplikasi web yang aman sesuai best practices") == ["create"]
    assert F.work_part("Melaksanakan pengujian unit program dalam alur CI/CD") == \
        "Melaksanakan pengujian unit program"
    # Indonesian activity nouns of the noun list v1.1
    assert read_phrase("Melakukan peninjauan kode").verbs == ("meninjau",)
    assert F.levels_of("Melakukan peninjauan kode sesuai best practices") == ["evaluate"]
    assert F.levels_of("Melakukan pendokumentasian desain") == ["apply"]
    # 'deliver' has no Indonesian verb of its own: the glossary's 'menerapkan' (C3) counts as it
    text = "Deliver reliable systems"
    rows = [{"text": text, "posting_id": f"p{i}", "sentence_text": text + " to our clients.",
             "start": 0, "end": len(text)} for i in range(4)]
    c = _competency(level="apply", code="C3", statement="Deliver systems")
    c["evidence"] = {**c["evidence"], "verbs": [{"verb": "deliver", "n_postings": 4}]}
    prof = Q.profile(c, rows)
    assert F.main_verbs(prof) == ["deliver"]
    cands = F.SkkniIndex(UNITS, embed=_embed).nearest("sistem", 2)
    ok = _answer(pernyataan="Menerapkan sistem", statement_en="Implement systems", kualifikasi_dipakai=[])
    assert F.check(ok, prof, [], cands)[1] == []
    swapped = _answer(pernyataan="Mengoperasikan sistem", statement_en="Operate systems",
                      kualifikasi_dipakai=[])
    assert any("does not use a verb the postings use" in p for p in F.check(swapped, prof, [], cands)[1])
    # a statement may name its own work with a practice term: "code review" for "Review code"
    assert Q.restates("code review", "Review code") and not Q.restates("best practices", "Review code")
    text = "Review code"
    rows = [{"text": text, "posting_id": f"p{i}", "sentence_text": "Perform code review weekly.",
             "start": 0, "end": len(text)} for i in range(4)]
    c = _competency(level="evaluate", code="C5", statement="Review code")
    c["evidence"] = {**c["evidence"], "verbs": [{"verb": "review", "n_postings": 4}]}
    prof = Q.profile(c, rows)
    review = _answer(pernyataan="Melakukan code review", statement_en="Perform code review",
                     kualifikasi_dipakai=[])
    assert F.check(review, prof, [], cands)[1] == []


def test_a_fragment_is_not_formulated_and_costs_no_call():
    from competency.statement import is_fragment
    for text in ("Perform Design, develop", "Write clean", "Design scalable",
                 "Perform deliver secure", "Melakukan Mengembangkan, memelihara", "Perform",
                 "Perform building, documenting", "Perform developing, implementing",
                 "Perform Creates, implements, optimizes, debugs, refactors",
                 "Perform deliver integrated", "Perform building and operating, monitoring",
                 "Perform deliver", "Melakukan melakukan debug", "Perform developing test",
                 "Deliver high-quality"):
        assert is_fragment(text), text
    for text in ("Develop web applications", "Perform support releases", "Review code",
                 "Perform debugging", "Perform refactoring", "Perform test plans",
                 "Perform debugging, troubleshooting", "Perform Testing and Debugging",
                 "Translate business/feature requirements", "Monitor tools",
                 "Maintain code quality", "Improve reliability", "Perform create prototypes",
                 "Melakukan pemrograman", "Perform unit testing", "Design scalable systems"):
        assert not is_fragment(text), text
    c = {**_competency(), "statement": {"text": "Perform Design, develop", "object": ""}}

    def no_call(messages):
        raise AssertionError("a fragment must not be sent to the model")
    r = F.formulate_one(c, _rows(), F.SkkniIndex(UNITS, embed=_embed), no_call)
    assert r["skipped"] and not r["accepted"] and r["flags"] == ["fragmen"]


def test_a_sample_is_spread_over_the_levels():
    comps = [{"competency_id": f"c{i}", "n_postings": 100 - i,
              "statement": {"text": "Develop APIs" if i != 3 else "Perform Design, develop"},
              "contract": {"status": "kandidat_tinjauan"},
              "cognitive": {"code": ["C6", "C3", "C5", None][i % 4]}} for i in range(40)]
    s = F.pick_sample(comps, 8, 1, "kandidat_tinjauan")
    assert len(s) == 8 and {c["cognitive"]["code"] for c in s} == {"C6", "C3", "C5", None}
    assert "c3" not in {c["competency_id"] for c in s}                    # a fragment is not sampled
    assert F.pick_sample(comps, 8, 1, "kandidat_tinjauan") == s                    # seeded


def _run_standalone() -> int:
    tests = sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f))
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS {name}")
        except Exception as exc:                                            # noqa: BLE001
            failed.append(name)
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed"
          + (f", {len(failed)} FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
