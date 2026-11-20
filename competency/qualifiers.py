"""What a competency's own postings say about HOW WELL the work is done, WHICH
STANDARDS apply and WHAT OUTCOME it serves.

The promotors (2026-11-20): a task title such as "develop web application" is a
duty; a competency says how well it is done, which skills or standards apply
and what outcome it achieves. Postings often say exactly that next to the work
they ask for ("develop scalable, secure web applications following best
practices to improve performance"). This module reads it, for each competency,
from each of its phrases and the words after the phrase in the same sentence:

  qualities  quality words ("scalable", "secure", "maintainable", "aman"), each
             with the product-quality characteristic of ISO/IEC 25010 it belongs
             to (the developer's mapping; check the standard before citing it);
  standards  practices and standards named ("best practices", "coding
             standards", "SOLID", "OWASP", "REST", "sesuai SOP") and any
             "following / according to / sesuai ..." phrase; a standard that
             only restates the competency ("code review" for "Review code") is
             left out;
  outcomes   purpose clauses ("to improve performance", "untuk memastikan ...");

next to what the competency already carries (verbs, the knowledge acted on,
tools, knowledge asked for alongside, work behaviours, soft skills). Every item
has an id (Q1, S1, R1, V1, O1, T1, K1, B1) and the number of the competency's
postings behind it. Free; no model is called. `competency.formulate` turns the
items into the "how well / which standards / outcome" parts of a statement.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


QUALIFIERS_VERSION = "qualifiers-v1"
WINDOW_AFTER_SPAN = 160              # characters after a phrase still read as about it

# quality word -> ISO/IEC 25010 product-quality characteristic (2011 names).
# The developer's mapping; "general" = quality without a characteristic.
QUALITY_TERMS: dict[str, str] = {
    **{w: "performance efficiency" for w in (
        "scalable", "scalability", "high-performance", "high performance", "high-performing",
        "performant", "efficient", "efficiently", "fast", "low-latency", "low latency",
        "optimized", "optimised", "high-traffic", "high traffic", "high-throughput",
        "large-scale", "large scale", "efisien", "skalabel", "cepat")},
    **{w: "reliability" for w in (
        "reliable", "reliability", "robust", "stable", "resilient", "fault-tolerant",
        "highly available", "high-availability", "high availability", "dependable",
        "bug-free", "error-free", "andal", "stabil")},
    **{w: "security" for w in ("secure", "securely", "aman")},
    **{w: "maintainability" for w in (
        "maintainable", "maintainability", "clean", "readable", "modular", "reusable",
        "testable", "well-tested", "well tested", "well-documented", "well documented",
        "well-structured", "well structured", "extensible", "mudah dipelihara", "rapi")},
    **{w: "usability" for w in (
        "user-friendly", "user friendly", "intuitive", "accessible", "usable",
        "user-centric", "user-centered", "user-centred", "mudah digunakan")},
    **{w: "portability" for w in (
        "responsive", "cross-platform", "cross platform", "portable", "mobile-friendly",
        "responsif")},
    **{w: "compatibility" for w in ("interoperable", "compatible")},
    **{w: "functional suitability" for w in ("accurate", "accurately", "correct", "akurat")},
    **{w: "general" for w in (
        "high-quality", "high quality", "quality", "production-ready", "production ready",
        "production-grade", "enterprise-grade", "world-class", "berkualitas")},
}
_QUALITY = re.compile(r"(?<![\w-])(" + "|".join(sorted(map(re.escape, QUALITY_TERMS), key=len,
                                                       reverse=True)) + r")(?![\w-])", re.I)

NAMED_STANDARDS: list[tuple[str, str]] = [
    ("best practices", r"best[- ]practices?"),
    ("coding standards", r"coding (?:standards?|guidelines?|conventions?)"),
    ("design patterns", r"design patterns?"),
    ("SOLID principles", r"\bSOLID\b"),
    ("clean code", r"clean[- ]code"),
    ("test-driven development (TDD)", r"\bTDD\b|test[- ]driven"),
    ("behaviour-driven development (BDD)", r"\bBDD\b|behaviou?r[- ]driven"),
    ("OWASP", r"\bOWASP\b"),
    ("accessibility standards (WCAG)", r"\bWCAG\b|accessibility (?:standards?|guidelines?)"),
    ("REST principles", r"\bREST(?:ful)?\b"),
    ("object-oriented principles", r"object[- ]oriented|\bOOP\b"),
    ("MVC", r"\bMVC\b"),
    ("CI/CD", r"\bCI\s*/\s*CD\b|continuous (?:integration|delivery|deployment)"),
    ("Agile / Scrum", r"\bAgile\b|\bScrum\b"),
    ("code review", r"code reviews?"),
    ("unit tests", r"unit[- ]tests?|unit[- ]testing"),
    ("security standards", r"security (?:standards?|best[- ]practices?|guidelines?|policies)"),
    ("industry standards", r"industry[- ]standards?"),
    ("ISO/IEC 27001", r"ISO(?:/IEC)?[ -]?27001"),
    ("PCI DSS", r"PCI[- ]?DSS"),
    ("GDPR", r"\bGDPR\b"),
    ("SDLC", r"\bSDLC\b|software development life ?cycle"),
    ("documentation standards", r"documentation standards?"),
    ("company standards", r"(?:company|organi[sz]ational|internal|team) (?:standards?|guidelines?)"),
    ("SOP / sesuai standar", r"\bSOP\b|sesuai (?:dengan )?(?:standar|prosedur)"),
]
_NAMED = [(name, re.compile(rx, re.I if not name.startswith(("SOLID", "OWASP", "MVC", "REST"))
                            else 0)) for name, rx in NAMED_STANDARDS]
_FOLLOWING = re.compile(
    r"\b(?:following|according to|in accordance with|adhering to|in line with|compliant with|"
    r"conforming to|sesuai(?: dengan)?|mengikuti|berdasarkan)\s+((?:[A-Za-z0-9/&+.-]+\s?){1,5})",
    re.I)
_OUTCOME = re.compile(
    r"\b(?:to|in order to|so that|untuk|agar|guna|sehingga)\s+"
    r"(ensure|improve|increase|reduce|enhance|optimi[sz]e|meet|achieve|maintain|minimi[sz]e|"
    r"maximi[sz]e|guarantee|boost|accelerate|streamline|memastikan|meningkatkan|mengurangi|"
    r"memenuhi|mencapai|menjaga|mengoptimalkan|mempercepat)\s+((?:[A-Za-z0-9/&+-]+\s?){1,4})",
    re.I)
_STOP_TAIL = {"and", "or", "the", "a", "an", "of", "to", "for", "with", "in", "on", "our", "your",
              "dan", "atau", "yang", "untuk", "dengan", "di"}


def _phrase(words: str) -> str:
    out = [w for w in words.strip().split()]
    while out and out[-1].lower().strip(".,;:") in _STOP_TAIL:
        out.pop()
    return " ".join(out).strip(" .,;:").lower()


@dataclass(frozen=True)
class QualifierConfig:
    min_postings: int = 2             # an item of the profile needs at least this many postings
    max_items: int = 8
    version: str = QUALIFIERS_VERSION


def restates(name: str, statement: str) -> bool:
    """A "standard" that only restates the competency is not a standard of it:
    "code review" for the competency "Review code". The formulation checker uses
    the same test, so a statement may name its own work with that term."""
    own = {w.lower()[:4] for w in re.findall(r"[A-Za-z]{3,}", statement)}
    words = [w.lower()[:4] for w in re.findall(r"[A-Za-z]{3,}", name.replace("following: ", ""))]
    return bool(words) and all(w in own for w in words)


def profile(competency: dict, evidence_rows: Sequence[dict], cfg: QualifierConfig | None = None) -> dict:
    """What a competency's own postings say about it, item by item, each with
    an id and its postings. `evidence_rows` are the competency's evidence
    units: dicts with text, posting_id, sentence_text, start, end."""
    cfg = cfg or QualifierConfig()
    q_post, s_post, r_post = defaultdict(set), defaultdict(set), defaultdict(set)
    q_char = {}
    for e in evidence_rows:
        sent = e.get("sentence_text") or ""
        start, end = int(e.get("start", 0) or 0), int(e.get("end", 0) or 0)
        window = sent[start:min(len(sent), end + WINDOW_AFTER_SPAN)] if sent else e["text"]
        cut = re.search(r"[.;!?]\s", window[end - start:]) if end > start else None
        if cut:
            window = window[:end - start + cut.start()]
        pid = e["posting_id"]
        for m in _QUALITY.finditer(window):
            word = m.group(1).lower()
            q_post[word].add(pid)
            q_char[word] = QUALITY_TERMS.get(word, "general")
        for name, rx in _NAMED:
            if rx.search(window):
                s_post[name].add(pid)
        for m in _FOLLOWING.finditer(window):
            p = _phrase(m.group(1))
            if p and not any(rx.search(p) for _, rx in _NAMED):
                s_post["following: " + p].add(pid)
        for m in _OUTCOME.finditer(window):
            p = _phrase(m.group(1) + " " + m.group(2))
            if len(p.split()) >= 2:
                r_post[p].add(pid)
    n = max(1, competency["n_postings"])
    for name in list(s_post):
        if restates(name, competency["statement"]["text"]):
            del s_post[name]

    def rows(prefix, table, extra=None):
        items = sorted(((k, len(v)) for k, v in table.items() if len(v) >= cfg.min_postings),
                       key=lambda kv: (-kv[1], kv[0]))[:cfg.max_items]
        return [{"id": f"{prefix}{i}", "text": k, "n_postings": c, "share": round(c / n, 4),
                 **(extra(k) if extra else {})} for i, (k, c) in enumerate(items, 1)]
    ev = competency["evidence"]

    def listed(prefix, key, fields=("text", "n_postings")):
        return [{"id": f"{prefix}{i}", **{f: r.get(f) for f in fields if f in r}}
                for i, r in enumerate(ev.get(key) or [], 1)][:cfg.max_items]
    out = {
        "competency_id": competency["competency_id"],
        "statement": competency["statement"]["text"],
        "level": competency["cognitive"].get("level"), "code": competency["cognitive"].get("code"),
        "n_postings": competency["n_postings"],
        "verbs": [{"id": f"V{i}", "text": v["verb"], "n_postings": v["n_postings"]}
                  for i, v in enumerate(ev.get("verbs") or [], 1)][:cfg.max_items],
        "objects": listed("O", "knowledge_objects"),
        "qualities": rows("Q", q_post, lambda k: {"iso25010": q_char[k]}),
        "standards": rows("S", s_post),
        "outcomes": rows("R", r_post),
        "tools": listed("T", "technologies"),
        "knowledge": listed("K", "knowledge", ("text", "n_postings", "share", "lift")),
        "work_behaviours": listed("B", "work_behaviours", ("text", "n_postings", "share", "lift")),
        "soft_skills": listed("SS", "soft_skills", ("text", "n_postings", "share", "lift")),
        "example_phrases": [m["text"] for m in ev.get("members", [])[:12]],
        "example_sentences": [s["sentence_text"] for s in ev.get("samples", [])[:5]],
    }
    return out



def load_run(run_dir: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    """The competencies of a finished run and, for each, its evidence rows."""
    import pandas as pd
    run_dir = Path(run_dir)
    comps = json.loads((run_dir / "competencies.json").read_text(encoding="utf-8"))
    df = pd.read_parquet(run_dir / "evidence.parquet",
                         columns=["evidence_id", "text", "posting_id", "sentence_text", "start", "end"])
    rows = {r["evidence_id"]: r for r in df.to_dict("records")}
    return comps, {c["competency_id"]: [rows[i] for i in c["evidence"]["evidence_ids"] if i in rows]
                   for c in comps}
