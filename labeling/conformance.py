"""Guideline conformance set — SYNTHETIC sentences with rule-determined answers.

    .venv-ml\\Scripts\\python.exe -m labeling.conformance export-review
    .venv-ml\\Scripts\\python.exe -m labeling.conformance run --max-usd 0.60 --prompt-version v1-sh-action-and-noun-lists

Why it exists. Until the human gold standard is annotated there is nothing to
score a prompt change against: more agreement between models is not more
correctness (in the first smoke test all three models agreed that "wireframes"
is SKILL_HARD). This set gives a number now. Each item is a short sentence built
to exercise ONE rule of the validated guideline (Paket B v3.0), and its expected
annotation follows from that rule.

What it is NOT, stated here because this project has already had synthetic
numbers mistaken for evidence once:

  * It is not the pilot and not gold. Nobody annotated it; it lives outside
    DATA/gold_standard/ and its ids start with `conf-`.
  * It says nothing about accuracy on real job postings. Synthetic sentences are
    cleaner than real ones.
  * It must not be used to rank the models against each other. The items were
    drafted with an LLM's help, which can favour that model family. Its valid
    use is WITHIN a model: did prompt v1 follow a rule better than prompt v0?
  * The expected answers were reviewed and approved as a whole by the researcher
    on 2026-11-18. That makes them her reading of the guideline; it does not
    make the sentences any less synthetic.

`origin` matters when reading results. "guideline" items are the guideline's own
example sentences, many of which also appear in the prompt, so they measure
whether a model applies an example it was shown. "variant" items are new
sentences for the same rule and measure whether the rule generalises.

Scoring is strict and per item. An item PASSES when every expected span is
present with identical boundaries and tag, no `forbidden` span is present, and —
for `closed` items, where the guideline says nothing else is taggable — there is
no other span at all. On open items extra spans do not fail the item; they are
counted separately.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from gold_standard import TAGS

CONFORMANCE_VERSION = "v0"
# The researcher read the exported workbook on 2026-11-18 and approved the set
# as a whole ("the xlsx looks good for me"); no item was changed. She also
# confirmed the reading behind prompt v1: an activity noun ("automated testing",
# "unit testing") can be SKILL_HARD without a verb, a thing or artefact cannot.
REVIEW_STATUS = "reviewed and approved as a whole by the researcher, 2026-11-18"

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "DATA" / "labeling" / "conformance"

RULING = "R-2026-11-18"        # researcher ruling: one verb + list of nouns -> KN
ANY_TAG = "*"                  # in `forbidden`: the span must not carry any tag


@dataclass(frozen=True)
class Item:
    item_id: str
    rule: str
    lang: str                                  # en | id | mixed
    origin: str                                # guideline | variant
    sentence: str
    expected: tuple[tuple[int, int, str], ...]         # (start, end, tag)
    forbidden: tuple[tuple[int, int, str], ...] = ()
    closed: bool = False
    note: str = ""

    def expected_text(self) -> list[tuple[str, str]]:
        return [(self.sentence[s:e], t) for s, e, t in self.expected]


def _locate(sentence: str, text: str) -> tuple[int, int]:
    n = sentence.count(text)
    if n != 1:
        raise ValueError(f"{text!r} occurs {n} times in {sentence!r}; conformance "
                         f"items must be unambiguous")
    i = sentence.index(text)
    return i, i + len(text)


def _whole(sentence: str) -> str:
    """The sentence without its final full stop — 'seluruh frasa' in the guideline."""
    return sentence[:-1] if sentence.endswith(".") else sentence


_RAW: list[tuple] = []


def _add(rule, lang, origin, sentence, expected, *, forbidden=(), closed=False, note=""):
    _RAW.append((rule, lang, origin, sentence, tuple(expected), tuple(forbidden), closed, note))


G, V = "guideline", "variant"
W = _whole

# ---- §2 / §4: the guideline's own examples ---------------------------------
_add("2.1", "en", G, "Develop backend services using Python.",
     [("Develop backend services", "SH"), ("Python", "TT")])
_add("4.1", "en", G, "Develop REST APIs for internal services.", [("Develop REST APIs", "SH")])
_add("4.2", "en", G, "Communicate technical issues to non-technical stakeholders.",
     [(W("Communicate technical issues to non-technical stakeholders."), "SS")])
_add("4.3", "en", G, "Strong understanding of relational database concepts.",
     [("relational database concepts", "KN")])
_add("4.4", "en", G, "Experience with Docker and Kubernetes.",
     [("Docker", "TT"), ("Kubernetes", "TT")])
_add("4.5", "en", G, "We are looking for a detail-oriented developer.",
     [("detail-oriented", "WB")])

# ---- §5: commonly confused pairs -------------------------------------------
_add("5 SH/KN", "en", G, "Understanding object-oriented programming principles.",
     [("object-oriented programming principles", "KN")])
_add("5 SH/KN", "en", G,
     "Apply object-oriented programming principles when designing applications.",
     [(W("Apply object-oriented programming principles when designing applications."), "SH")])
_add("5 TT/KN", "en", G, "Strong knowledge of Python.", [("Python", "TT")])
_add("5 TT/KN", "en", G, "Strong knowledge of software architecture principles.",
     [("software architecture principles", "KN")])
_add("5 SS/WB", "en", G, "Collaborate with product and QA teams.",
     [(W("Collaborate with product and QA teams."), "SS")])
_add("5 SS/WB", "en", G, "Proactive and detail-oriented.",
     [("Proactive", "WB"), ("detail-oriented", "WB")])
_add("5 SS/SH", "en", G, "Present technical designs to stakeholders.",
     [(W("Present technical designs to stakeholders."), "SS")])
_add("5 SS/SH", "en", G, "Facilitate Scrum ceremonies.", [("Facilitate Scrum ceremonies", "SH")])

# ---- §6: span boundaries ----------------------------------------------------
_add("6.1", "en", G, "Must have experience in automated testing.", [("automated testing", "SH")])
_add("6.2", "en", G, "Design database schemas.", [("Design database schemas", "SH")])
_add("6.5", "en", G, "Python, Java, and Go.",
     [("Python", "TT"), ("Java", "TT"), ("Go", "TT")], closed=True)
_add("6.6", "en", V, "Experience with Amazon Web Services (AWS).",
     [("Amazon Web Services (AWS)", "TT")],
     note="the phrase is the guideline's; the carrier sentence is new")
_add("6.7", "en", G, "Python, Docker, and React.",
     [("Python", "TT"), ("Docker", "TT"), ("React", "TT")], closed=True)
_add("6.8", "en", G, "Build web applications using React.",
     [("Build web applications", "SH"), ("React", "TT")])
_add("6.9", "en", G, "Configure Kubernetes clusters.", [("Configure Kubernetes clusters", "SH")],
     note="a nested 'Kubernetes' TT is allowed by 6.9 and neither required nor penalised")
_add("6.10", "en", V, "Experience with Java 17, Vue 3, and Python 3.11.",
     [("Java 17", "TT"), ("Vue 3", "TT"), ("Python 3.11", "TT")])

# ---- §7: special cases ------------------------------------------------------
_add("7.1", "en", G, "Python, Java, Docker.",
     [("Python", "TT"), ("Java", "TT"), ("Docker", "TT")], closed=True)
_add("7.2", "en", G, "Understanding Agile principles.", [("Agile principles", "KN")])
_add("7.3", "en", V, "Experience with MySQL, PostgreSQL, and MongoDB.",
     [("MySQL", "TT"), ("PostgreSQL", "TT"), ("MongoDB", "TT")])
_add("7.4", "en", V, "Understanding of testing principles.", [("testing principles", "KN")])
_add("7.4", "en", G, "Write automated tests using Selenium.",
     [("Write automated tests", "SH"), ("Selenium", "TT")])
_add("7.5", "en", V, "Experience with AWS and Jenkins.", [("AWS", "TT"), ("Jenkins", "TT")])
_add("7.5", "en", V, "Knowledge of cloud architecture principles.",
     [("cloud architecture principles", "KN")])
_add("7.5", "en", V, "Deploy applications to cloud environments.",
     [(W("Deploy applications to cloud environments."), "SH")])
_add("7.5", "en", V, "Build CI/CD pipelines.", [("Build CI/CD pipelines", "SH")])
_add("7.6", "en", V, "Understanding of security principles.", [("security principles", "KN")])
_add("7.6", "en", V, "Perform vulnerability assessment.",
     [("Perform vulnerability assessment", "SH")])
_add("7.7", "en", V, "Experience with TensorFlow and Python.",
     [("TensorFlow", "TT"), ("Python", "TT")])
_add("7.7", "en", V, "Understanding of machine learning principles.",
     [("machine learning principles", "KN")])
_add("7.7", "en", V, "Train machine learning models.", [("Train machine learning models", "SH")])

# 7.8 — requirements that get NO tag
_add("7.8", "en", G, "Minimum three years of experience and a bachelor's degree.", [], closed=True)
_add("7.8", "en", V, "Bachelor of Computer Science required.", [], closed=True)
_add("7.8", "en", V, "Fluent in English.", [], closed=True)
_add("7.8", "id", V, "Pendidikan minimal S1.", [], closed=True)
_add("7.8", "id", V, "Usia maksimal 27 tahun.", [], closed=True)
_add("7.8", "id", V, "Penempatan di Jakarta.", [], closed=True)
_add("7.8", "id", V, "Memiliki pengalaman kerja minimal tiga tahun.", [], closed=True)

# 7.9 — generic technology groups are not TT
_add("7.9", "en", G, "Familiarity with NoSQL databases.", [("NoSQL databases", "KN")],
     forbidden=[("NoSQL databases", "TT")])
_add("7.9", "en", G, "Deploy to cloud platforms.", [("Deploy to cloud platforms", "SH")],
     forbidden=[("cloud platforms", "TT")])
_add("7.9", "en", V, "Familiarity with frontend frameworks.", [("frontend frameworks", "KN")],
     forbidden=[("frontend frameworks", "TT")])

# 7.10 — job titles get no tag
_add("7.10", "en", V, "We are hiring a Backend Developer.", [], closed=True)
_add("7.10", "id", V, "Dibutuhkan segera QA Engineer.", [], closed=True)

_add("7.11", "en", G, "No prior experience with Kubernetes is required.", [("Kubernetes", "TT")])
_add("7.12", "en", G, "Experience with Docker is preferred.", [("Docker", "TT")])
_add("7.13", "mixed", G, "Mampu melakukan debugging dan unit testing menggunakan Jest.",
     [("melakukan debugging", "SH"), ("unit testing", "SH"), ("Jest", "TT")])

# 7.15 — general abilities vs traits
_add("7.15", "en", V, "Strong problem solving and analytical thinking.",
     [("problem solving", "SS"), ("analytical thinking", "SS")])
_add("7.15", "en", V, "Critical thinking is required.", [("Critical thinking", "SS")])
_add("7.15", "en", V, "We want someone curious and resilient.",
     [("curious", "WB"), ("resilient", "WB")])
_add("7.15", "en", V, "An ownership mindset is essential.", [("ownership mindset", "WB")])

# 3.1 — a non-relevant sentence carries nothing
_add("3.1", "en", G,
     "We are one of the fastest-growing technology companies in Southeast Asia.", [], closed=True)

# ---- researcher ruling 2026-11-18: one verb + a list of nouns -> KN ----------
_add(RULING, "en", V, "You will work from user flows through wireframes and prototypes.",
     [("user flows", "KN"), ("wireframes", "KN"), ("prototypes", "KN")],
     forbidden=[("wireframes", "SH"), ("prototypes", "SH"), ("user flows", "SH")],
     note="needs the researcher's confirmation: this applies her ruling on sentence 6")
_add(RULING, "en", V, "You will work with design systems and component libraries.",
     [("design systems", "KN"), ("component libraries", "KN")],
     forbidden=[("design systems", "SH"), ("component libraries", "SH")],
     note="needs the researcher's confirmation")
_add(RULING, "id", V, "Anda akan bekerja dengan wireframe dan prototipe.",
     [("wireframe", "KN"), ("prototipe", "KN")],
     forbidden=[("wireframe", "SH"), ("prototipe", "SH")],
     note="needs the researcher's confirmation")

# ---- variants: 6.2 keep the verb (the weakness seen in the smoke test) ------
for _s in ["Write SQL queries.", "Implement authentication.", "Perform code review.",
           "Develop mobile applications.", "Maintain legacy systems."]:
    _add("6.2", "en", V, _s, [(W(_s), "SH")])
for _s, _span in [
        ("Mampu menulis kueri SQL.", "menulis kueri SQL"),
        ("Mampu melakukan pengujian otomatis.", "melakukan pengujian otomatis"),
        ("Mampu merancang skema basis data.", "merancang skema basis data"),
        ("Mampu mengimplementasikan autentikasi.", "mengimplementasikan autentikasi"),
        ("Mampu melakukan deployment aplikasi.", "melakukan deployment aplikasi"),
        ("Mampu membangun CI/CD pipeline.", "membangun CI/CD pipeline"),
        ("Mampu melakukan code review.", "melakukan code review")]:
    _add("6.2", "id", V, _s, [(_span, "SH")],
         note="verb kept (6.2), lead-in 'Mampu' left out (6.3)")

# ---- variants: 6.3 leave out the lead-in ------------------------------------
for _s, _span in [
        ("Able to implement authentication.", "implement authentication"),
        ("Required to build CI/CD pipelines.", "build CI/CD pipelines"),
        ("Should be able to perform code review.", "perform code review"),
        ("Must be able to design database schemas.", "design database schemas")]:
    _add("6.3", "en", V, _s, [(_span, "SH")])
for _s, _span, _tag in [
        ("Proficiency in JavaScript.", "JavaScript", "TT"),
        ("Familiar with Git.", "Git", "TT"),
        ("Experience in Laravel.", "Laravel", "TT"),
        ("Strong knowledge of data structures.", "data structures", "KN"),
        ("Understanding of containerization concepts.", "containerization concepts", "KN")]:
    _add("6.3", "en", V, _s, [(_span, _tag)])
for _s, _span, _tag in [
        ("Menguasai Laravel.", "Laravel", "TT"),
        ("Berpengalaman dalam React.", "React", "TT"),
        ("Memahami konsep microservices.", "konsep microservices", "KN"),
        ("Memahami prinsip pemrograman berorientasi objek.",
         "prinsip pemrograman berorientasi objek", "KN")]:
    _add("6.3", "id", V, _s, [(_span, _tag)])

# ---- variants: 6.5 list items are separate spans ----------------------------
_add("6.5", "en", V, "Experience with React, Laravel, and Spring Boot.",
     [("React", "TT"), ("Laravel", "TT"), ("Spring Boot", "TT")])
_add("6.5", "en", V, "Knowledge of Git or Docker.", [("Git", "TT"), ("Docker", "TT")])
_add("6.5", "id", V, "Menguasai Python, Kotlin, dan Go.",
     [("Python", "TT"), ("Kotlin", "TT"), ("Go", "TT")])
_add("6.5", "id", V, "Memahami struktur data dan algoritma.",
     [("struktur data", "KN"), ("algoritma", "KN")])

# ---- variants: 6.6 full name with its acronym is one span -------------------
_add("6.6", "en", V, "Experience with Google Cloud Platform (GCP).",
     [("Google Cloud Platform (GCP)", "TT")])

# ---- variants: 6.8 separate the action from the technology ------------------
_add("6.8", "en", V, "Develop REST APIs using Spring Boot.",
     [("Develop REST APIs", "SH"), ("Spring Boot", "TT")])
_add("6.8", "en", V, "Write SQL queries in PostgreSQL.",
     [("Write SQL queries", "SH"), ("PostgreSQL", "TT")])
_add("6.8", "id", V, "Membangun aplikasi web menggunakan Laravel.",
     [("Membangun aplikasi web", "SH"), ("Laravel", "TT")])
_add("6.8", "en", V, "Train machine learning models using TensorFlow.",
     [("Train machine learning models", "SH"), ("TensorFlow", "TT")])

# ---- variants: SS vs WB ------------------------------------------------------
_add("5 SS/WB", "en", V, "Coordinate with the QA team.", [(W("Coordinate with the QA team."), "SS")])
_add("5 SS/WB", "en", V, "Self-motivated and proactive.",
     [("Self-motivated", "WB"), ("proactive", "WB")])
_add("5 SS/WB", "id", V, "Teliti dan bertanggung jawab.",
     [("Teliti", "WB"), ("bertanggung jawab", "WB")])
_add("5 SS/WB", "id", V, "Disiplin, proaktif, dan mau belajar.",
     [("Disiplin", "WB"), ("proaktif", "WB"), ("mau belajar", "WB")])
_add("5 SS/WB", "id", V, "Mampu berkomunikasi dengan klien.",
     [("berkomunikasi dengan klien", "SS")])
_add("6.1", "en", V, "Good time management.", [("time management", "SS")])

# ---- variants: more of the no-tag and technology cases ----------------------
_add("7.8", "en", V, "Salary is negotiable.", [], closed=True)
_add("7.8", "id", V, "Bersedia ditempatkan di Surabaya.", [], closed=True)
_add("7.10", "en", V, "The role reports to the Engineering Manager.", [], closed=True)
_add("7.11", "en", V, "Experience with Azure is not required.", [("Azure", "TT")])
_add("7.12", "en", V, "Knowledge of MongoDB is a plus.", [("MongoDB", "TT")])
_add("7.12", "id", V, "Pengalaman dengan Docker menjadi nilai tambah.", [("Docker", "TT")])
_add("7.2", "en", V, "Understanding of Scrum principles.", [("Scrum principles", "KN")])
_add("6.10", "en", V, "Experience with Angular 17.", [("Angular 17", "TT")])
_add("6.4", "en", V, "Experience in automated testing.", [("automated testing", "SH")],
     note="6.4: must not shrink to 'testing'")
_add("6.4", "en", V, "Knowledge of relational database concepts.",
     [("relational database concepts", "KN")], note="6.4: must not shrink to 'database'")


def build_items() -> list[Item]:
    items = []
    for n, (rule, lang, origin, sentence, expected, forbidden, closed, note) in enumerate(_RAW, 1):
        exp = tuple((*_locate(sentence, t), tag) for t, tag in expected)
        forb = tuple((*_locate(sentence, t), tag) for t, tag in forbidden)
        for _, _, tag in exp + forb:
            if tag not in TAGS:
                raise ValueError(f"unknown tag {tag!r} in item {n}")
        items.append(Item(f"conf-{n:03d}", rule, lang, origin, sentence, exp, forb, closed, note))
    return items


ITEMS: list[Item] = build_items()


def as_sentences(items: list[Item] | None = None) -> list[dict]:
    """The items in the shape `labeling.run.run` takes. No context sentences."""
    return [{"sentence_id": it.item_id, "sentence_text": it.sentence,
             "previous_sentence": "", "next_sentence": ""} for it in (items or ITEMS)]


def score_item(item: Item, spans) -> dict:
    """`spans`: verified spans of ONE model for this item (anything with
    .start/.end/.label). Returns the verdict and what went wrong."""
    got = {(s.start, s.end, s.label) for s in spans}
    bounds = {(s.start, s.end) for s in spans}
    missing = [e for e in item.expected if e not in got]
    # a forbidden tag of ANY_TAG means "this stretch must not be tagged at all"
    hit_forbidden = [f for f in item.forbidden
                     if f in got or (f[2] == ANY_TAG and (f[0], f[1]) in bounds)]
    extra = sorted(got - set(item.expected))
    passed = (not missing and not hit_forbidden and not (item.closed and extra))
    return {"item_id": item.item_id, "pass": passed,
            "missing": [(item.sentence[s:e], t) for s, e, t in missing],
            "forbidden_hit": [(item.sentence[s:e], t) for s, e, t in hit_forbidden],
            "extra": [(item.sentence[s:e], t) for s, e, t in extra],
            "n_expected": len(item.expected), "n_expected_found": len(item.expected) - len(missing)}


def score(spans_by_item: dict[str, list], items: list[Item] | None = None,
          output_valid: dict[str, bool] | None = None) -> dict:
    """Pass rates for one model: overall, by rule, by origin and by language.

    An item with no entry in `spans_by_item` counts as a model that returned
    nothing for it. `output_valid[item_id] is False` marks an invalid or missing
    response, which FAILS the item even when the item expects no span — a
    model that crashed did not correctly decide "no tag".
    """
    from gold_standard.metrics import wilson_ci
    items = items or ITEMS
    rows = []
    for it in items:
        r = score_item(it, spans_by_item.get(it.item_id, []))
        if output_valid is not None and output_valid.get(it.item_id) is False:
            r["pass"] = False
            r["invalid_output"] = True
        r.update(rule=it.rule, origin=it.origin, lang=it.lang)
        rows.append(r)

    def agg(key):
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r["pass"])
        out = {}
        for k in sorted(groups):
            v = groups[k]
            lo, hi = wilson_ci(sum(v), len(v))
            out[k] = {"passed": sum(v), "n": len(v), "rate": sum(v) / len(v),
                      "wilson95": [lo, hi]}
        return out

    k, n = sum(r["pass"] for r in rows), len(rows)
    lo, hi = wilson_ci(k, n)
    return {"conformance_version": CONFORMANCE_VERSION, "review_status": REVIEW_STATUS,
            "n_items": n, "passed": k, "rate": k / n if n else float("nan"),
            "wilson95": [lo, hi],
            "expected_spans_found": sum(r["n_expected_found"] for r in rows),
            "expected_spans_total": sum(r["n_expected"] for r in rows),
            "by_rule": agg("rule"), "by_origin": agg("origin"), "by_lang": agg("lang"),
            "items": rows}


def score_run(run_dir: Path, items: list[Item] | None = None) -> dict:
    """Score every model of a finished run. Free: re-reads raw.jsonl only."""
    from .verify import verify_output
    items = items or ITEMS
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    sha = summary["request_sha256"]
    by_id = {it.item_id: it for it in items}
    raw: dict[tuple[str, str], dict] = {}
    for line in (run_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("request_sha256") == sha and r["sentence_id"] in by_id:
                raw[(r["model"], r["sentence_id"])] = r
    out = {"prompt_version": summary["prompt"]["prompt_version"],
           "request_sha256": sha, "models": {}}
    for model in summary["run_config"]["models"]:
        spans, valid = {}, {}
        for it in items:
            rec = raw.get((model, it.item_id))
            if not rec or not rec.get("ok"):
                valid[it.item_id] = False
                continue
            vo = verify_output(model, it.item_id, it.sentence, rec.get("raw") or "")
            valid[it.item_id] = vo.status == "VALID"
            spans[it.item_id] = vo.spans
        out["models"][model] = score(spans, items, valid)
    return out


def export_review(path: Path | None = None, items: list[Item] | None = None,
                  readme_lines: list[str] | None = None) -> Path:
    """Write the review workbook the researcher signs off on.

    With `items` and `readme_lines` it serves any item list in the same shape
    (used by `labeling.skillspan_guideline`)."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    path = Path(path) if path else OUT_DIR / f"conformance_{CONFORMANCE_VERSION}_REVIEW.xlsx"
    items = ITEMS if items is None else items
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "items"
    head = ["id", "rule", "language", "origin", "sentence", "expected spans  (text [TAG])",
            "must NOT appear", "nothing else may be tagged?", "note",
            "OK? (Y / N)", "your correction or comment"]
    ws.append(head)
    for c in ws[1]:
        c.font = Font(bold=True)
    # Real text can carry control characters that Excel refuses. They are removed
    # for DISPLAY only; offsets and scoring use the untouched sentence.
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

    def clean(v):
        return ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v

    for it in items:
        ws.append([clean(v) for v in [
            it.item_id, it.rule, it.lang, it.origin, it.sentence,
            "\n".join(f"{t} [{tag}]" for t, tag in it.expected_text()) or "(no span)",
            "\n".join(f"{it.sentence[s:e]} [{t}]" for s, e, t in it.forbidden),
            "yes" if it.closed else "", it.note, "", ""]])
    for col, width in zip("ABCDEFGHIJK", (10, 14, 9, 10, 62, 44, 26, 12, 40, 10, 44)):
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    info = wb.create_sheet("read me")
    for line in readme_lines if readme_lines is not None else [
        f"Guideline conformance set {CONFORMANCE_VERSION} — {len(ITEMS)} SYNTHETIC items.",
        "",
        "These sentences were written to test one guideline rule each. They are not from",
        "the corpus, not the pilot and not gold. Please check that every expected answer is",
        "what guideline B v3.0 requires, and mark column J with Y or N.",
        "",
        "origin = guideline : the sentence is one of the guideline's own examples.",
        "origin = variant   : a new sentence for the same rule; these need your eye most.",
        "",
        f"Rule {RULING} is your ruling of 2026-11-18 (one verb governing a list of nouns:",
        "tag the nouns as KNOWLEDGE). Please confirm those three items in particular.",
        "",
        "Not included, because the guideline does not settle them:",
        "  - coordinated verbs sharing one object ('Creating, compiling, and debugging programs')",
        "  - guideline 2.3's example 'Develop Python applications' -> 'Develop applications' +",
        "    'Python': the first span is not contiguous, so it cannot be written as offsets.",
    ]:
        info.append([line])
    info.column_dimensions["A"].width = 100
    wb.save(path)
    return path


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export-review", help="write the review workbook")
    rp = sub.add_parser("run", help="label the set with the ensemble and score it")
    rp.add_argument("--max-usd", type=float, required=True)
    rp.add_argument("--prompt-version", required=True)
    rp.add_argument("--reasoning", default="none")
    # Short sentences, reasoning off: answers are well under 400 tokens. The
    # budget pre-flight assumes every call uses all of max_tokens AND every
    # retry, so a tight max_tokens is what keeps the worst case realistic.
    rp.add_argument("--max-tokens", type=int, default=400)
    rp.add_argument("--retries", type=int, default=1)
    sp = sub.add_parser("score", help="re-score a finished run (no API calls)")
    sp.add_argument("run_dir")
    args = ap.parse_args()

    if args.cmd == "export-review":
        print(f"-> {export_review()}")
        return
    if args.cmd == "run":
        from .run import DEFAULT_MODELS, RUNS_DIR, BudgetError, RunConfig, run
        reasoning = None if args.reasoning == "default" else {"effort": args.reasoning}
        cfg = RunConfig(models=tuple(DEFAULT_MODELS), prices=DEFAULT_MODELS,
                        max_usd=args.max_usd, reasoning=reasoning,
                        max_tokens=args.max_tokens, retries=args.retries,
                        prompt_version=args.prompt_version)
        run_dir = RUNS_DIR / f"conformance_{CONFORMANCE_VERSION}_{cfg.request_sha256()[:8]}"
        try:
            summary = run(as_sentences(), cfg, run_dir)
        except BudgetError as exc:
            raise SystemExit(f"BUDGET: {exc}")
        print(f"spent ${summary['cost']['charged_usd_total']:.4f}")
    else:
        run_dir = Path(args.run_dir)
    result = score_run(run_dir)
    (run_dir / "conformance_score.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"prompt {result['prompt_version']}   ({REVIEW_STATUS})")
    for model, s in result["models"].items():
        print(f"  {model:30} {s['passed']}/{s['n_items']} items pass  "
              f"(guideline {s['by_origin'].get('guideline', {}).get('passed', 0)}/"
              f"{s['by_origin'].get('guideline', {}).get('n', 0)}, variant "
              f"{s['by_origin'].get('variant', {}).get('passed', 0)}/"
              f"{s['by_origin'].get('variant', {}).get('n', 0)})")
    print(f"-> {run_dir / 'conformance_score.json'}")


if __name__ == "__main__":
    main()
