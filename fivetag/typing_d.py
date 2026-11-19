"""D1, D2, D3 — the three deterministic classifiers of proposal §3.3.1.

They answer one question about a span that has ALREADY been extracted: which
class is it? "Deterministic" means what the proposal says it means: the output
is fixed by rules, word lists, thresholds and priorities of a named version.

  D1  LRP   Lexicon-Regex-Priority. Looks at the span text only. Every rule that
            matches is recorded; a priority table picks the answer.
  D2  TACD  Taxonomy-Anchored Context Decision. First anchors the span to the
            taxonomy (exact, normalised, acronym, frozen-threshold fuzzy match),
            then reads cue words around it in the sentence, then looks the
            combination up in a decision table.
  D3        Deterministic consensus. D1 and D2 agree -> accepted. They differ ->
            CONFLICT, passed on; never resolved by a silent automatic choice.

Classes: the five tags, plus EXPERIENCE and CREDENTIAL (named by the proposal),
plus ABSTAIN when no rule applies. D3 adds CONFLICT.

The "taxonomy" is, for now, the versioned lexicon. The plan also names ESCO and
O*NET lists; they plug in as further anchor sources without changing D2's logic.

`extraction_typing.py` at the repo root is the pre-approval v1 classifier with a
different class set. It still feeds the v1 competency stages and is left as it
is; this module is the D1 the plan asks for, written against the five tags.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable, Sequence

from . import TAG_CODES
from .lexicon import DEFAULT_VERSION, Lexicon, load

TYPING_VERSION = "typing-v1"

EXPERIENCE = "EXPERIENCE"
CREDENTIAL = "CREDENTIAL"
ABSTAIN = "ABSTAIN"
CONFLICT = "CONFLICT"
CLASSES: tuple[str, ...] = (*TAG_CODES, EXPERIENCE, CREDENTIAL)

# Frozen. Changing it is a new TYPING_VERSION.
FUZZY_THRESHOLD = 0.90
FUZZY_MIN_LENGTH = 5            # never fuzzy-match very short names ("Go", "Git", "C#")
CUE_WINDOW = 4                  # tokens of context read on each side

_EXPERIENCE_RE = re.compile(
    r"\b(?:\d+\s*\+?|\d+\s*[-–]\s*\d+|one|two|three|four|five|satu|dua|tiga|empat|lima)\s*"
    r"(?:\+\s*)?(?:years?|yrs?|tahun|bulan|months?)\b"
    r"|\bfresh\s*graduates?\b|\bminimum\s+\d+\b|\bminimal\s+\d+\b", re.I)
_CREDENTIAL_RE = re.compile(
    r"\b(?:bachelor(?:'s)?|master(?:'s)?|ph\.?d|doctorate|diploma|degree|sarjana|magister|"
    r"S1|S2|D3|D4|SMK|SMA|ijazah|lulusan|certified|certification|certificate|"
    r"sertifikasi|sertifikat|gelar|GPA|IPK)\b", re.I)

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#.'\-]*")


def _words(text: str) -> list[str]:
    return [w.rstrip(".").lower() for w in _WORD.findall(text) if w.rstrip(".")]


def normalise(text: str) -> str:
    """Case, punctuation and spacing folded away: 'Node.JS' == 'nodejs' == 'node js'."""
    return re.sub(r"[^a-z0-9+#]", "", text.lower())


@dataclass(frozen=True)
class Decision:
    label: str                          # one of CLASSES, or ABSTAIN / CONFLICT
    model: str                          # D1 | D2 | D3
    rule: str = ""                      # the rule or table row that decided
    fired: tuple[str, ...] = ()         # every rule that matched (D1)
    anchor: str = ""                    # D2: anchor type
    anchor_how: str = ""                # D2: exact | normalised | acronym | fuzzy
    cue: str = ""                       # D2: context cue class
    parts: tuple = ()                   # D3: the two decisions it combined
    conflict_kind: str = ""             # D3: disagree | one_abstained

    @property
    def decided(self) -> bool:
        return self.label in CLASSES


class _Shared:
    """What D1 and D2 both read from the lexicon."""

    def __init__(self, lexicon: Lexicon | None = None, version: str = DEFAULT_VERSION):
        lx = self.lexicon = lexicon or load(version)
        self.tech = lx.words("technologies")
        self.tech_cs = {e.text for e in lx.entries("technologies") if e.case_sensitive}
        self.soft = lx.words("soft_skills")
        self.behavior = lx.words("work_behaviors")
        self.kn_terms = lx.words("knowledge", "terms")
        self.kn_groups = lx.words("knowledge", "groups")
        self.heads = lx.words("knowledge", "heads")
        self.heads_first = lx.words("knowledge", "heads-first")
        self.activity = lx.words("actions", "activity-nouns")
        self.suffix = lx.words("actions", "activity-suffix")
        self.hard_verbs = {f for line in lx.lines("actions", "verbs") for f in line}
        self.soft_verbs = {f for line in lx.lines("actions", "soft-verbs") for f in line}

    def is_tech(self, text: str) -> bool:
        """Whole text is a technology entry, allowing a version number after it."""
        t = text.strip()
        base = re.sub(r"\s+v?\d+(?:\.\d+)*\+?$", "", t)
        base = re.sub(r"\s*\([^)]*\)$", "", base)
        for cand in {t, base}:
            if cand.lower() in self.tech:
                cs = [e for e in self.tech_cs if e.lower() == cand.lower()]
                if not cs or cand in cs:
                    return True
        return False


class D1LexiconRegexPriority(_Shared):
    """D1 (LRP). Priority = position in PRIORITY; the first matching rule decides."""
    name = "D1"

    # (rule id, class). Order IS the priority table.
    PRIORITY: tuple[tuple[str, str], ...] = (
        ("regex:experience", EXPERIENCE),
        ("regex:credential", CREDENTIAL),
        # a whole-span dictionary entry outranks a shape: "design patterns" is
        # the known term, not the verb "design" with an object
        ("whole:technology", "TT"),
        ("whole:knowledge_term", "KN"),
        ("whole:knowledge_group", "KN"),
        ("whole:activity_noun", "SH"),
        ("starts:hard_verb", "SH"),
        ("starts:soft_verb", "SS"),
        ("ends:knowledge_head", "KN"),
        ("starts:knowledge_head", "KN"),
        ("ends:activity_suffix", "SH"),
        ("whole:work_behavior", "WB"),
        ("whole:soft_skill", "SS"),
        ("contains:work_behavior", "WB"),
        ("contains:soft_skill", "SS"),
        ("contains:technology", "TT"),
    )

    def _fired(self, text: str) -> list[str]:
        low, words = text.strip().lower(), _words(text)
        if not words:
            return []
        out = []
        if _EXPERIENCE_RE.search(text):
            out.append("regex:experience")
        if _CREDENTIAL_RE.search(text):
            out.append("regex:credential")
        if len(words) >= 2 and words[0] in self.hard_verbs:
            out.append("starts:hard_verb")
        if len(words) >= 2 and words[0] in self.soft_verbs:
            out.append("starts:soft_verb")
        if self.is_tech(text):
            out.append("whole:technology")
        if low in self.kn_terms:
            out.append("whole:knowledge_term")
        if low in self.kn_groups:
            out.append("whole:knowledge_group")
        if len(words) >= 2 and words[-1] in self.heads:
            out.append("ends:knowledge_head")
        if len(words) >= 2 and words[0] in self.heads_first:
            out.append("starts:knowledge_head")
        if low in self.activity:
            out.append("whole:activity_noun")
        if len(words) >= 2 and words[-1] in self.suffix:
            out.append("ends:activity_suffix")
        if low in self.behavior:
            out.append("whole:work_behavior")
        if low in self.soft:
            out.append("whole:soft_skill")
        joined = " " + " ".join(words) + " "
        if any(f" {b} " in joined for b in self.behavior):
            out.append("contains:work_behavior")
        if any(f" {s} " in joined for s in self.soft):
            out.append("contains:soft_skill")
        if len(words) <= 3 and any(w in self.tech and not self._cs_only(w, text) for w in words):
            out.append("contains:technology")
        return out

    def _cs_only(self, word: str, text: str) -> bool:
        cs = [e for e in self.tech_cs if e.lower() == word]
        return bool(cs) and not any(re.search(rf"(?<![A-Za-z0-9]){re.escape(e)}(?![A-Za-z0-9])", text)
                                    for e in cs)

    def classify(self, text: str, sentence: str = "", start: int | None = None,
                 end: int | None = None) -> Decision:
        fired = self._fired(text)
        for rule, label in self.PRIORITY:
            if rule in fired:
                return Decision(label, self.name, rule, tuple(fired))
        return Decision(ABSTAIN, self.name, "no rule matched")

    def manifest(self) -> dict:
        return {"model": self.name, "typing_version": TYPING_VERSION,
                "priority_table": [list(p) for p in self.PRIORITY], **self.lexicon.manifest()}


class D2TaxonomyAnchoredContext(_Shared):
    """D2 (TACD): anchor to the taxonomy, read the context, look up the table."""
    name = "D2"

    # cue classes, read from the words just before the span (lower-cased)
    CUES: dict[str, tuple[str, ...]] = {
        "UNDERSTAND": ("understanding of", "understanding", "understand", "knowledge of",
                       "familiar with", "familiarity with", "concept of", "concepts of",
                       "principles of", "theory of", "memahami", "pemahaman", "pengetahuan",
                       "konsep", "prinsip", "mengerti"),
        "USE": ("using", "use", "with", "in", "on", "via", "proficient in", "proficiency in",
                "experience with", "experience in", "hands-on", "menggunakan", "dengan",
                "menguasai", "pengalaman dengan", "berpengalaman dalam"),
        "TRAIT": ("be", "is", "are", "being", "someone", "person", "attitude", "mindset",
                  "bersikap", "pribadi", "memiliki sikap", "sikap"),
        "ABILITY": ("ability to", "able to", "skills", "skill", "kemampuan", "mampu",
                    "keterampilan"),
    }
    CUE_ORDER = ("UNDERSTAND", "ABILITY", "TRAIT", "USE")

    # Decision table. First matching row wins. '*' matches anything.
    # (shape, anchor, cue) -> class
    TABLE: tuple[tuple[str, str, str, str], ...] = (
        ("experience", "*", "*", EXPERIENCE),
        ("credential", "*", "*", CREDENTIAL),
        ("*", "TECHNOLOGY", "*", "TT"),
        ("*", "KNOWLEDGE_TERM", "*", "KN"),
        ("*", "KNOWLEDGE_GROUP", "*", "KN"),
        ("hard_verb_phrase", "*", "*", "SH"),
        ("soft_verb_phrase", "*", "*", "SS"),
        ("head_phrase", "*", "*", "KN"),
        ("*", "ACTIVITY", "UNDERSTAND", "KN"),
        ("*", "ACTIVITY", "*", "SH"),
        ("activity_suffix", "*", "UNDERSTAND", "KN"),
        ("activity_suffix", "*", "*", "SH"),
        ("*", "WORK_BEHAVIOR", "*", "WB"),
        ("*", "SOFT_SKILL", "*", "SS"),
        ("*", "NONE", "UNDERSTAND", "KN"),
        ("name_like", "NONE", "USE", "TT"),
        ("*", "NONE", "TRAIT", "WB"),
    )

    def __init__(self, lexicon: Lexicon | None = None, version: str = DEFAULT_VERSION):
        super().__init__(lexicon, version)
        self._anchors: list[tuple[str, dict[str, str]]] = []
        for kind, words in (("TECHNOLOGY", self.tech), ("KNOWLEDGE_TERM", self.kn_terms),
                            ("KNOWLEDGE_GROUP", self.kn_groups), ("ACTIVITY", self.activity),
                            ("WORK_BEHAVIOR", self.behavior), ("SOFT_SKILL", self.soft)):
            self._anchors.append((kind, {normalise(w): w for w in words}))
        self._acronyms: dict[str, str] = {}
        for kind, words in (("TECHNOLOGY", self.tech), ("KNOWLEDGE_TERM", self.kn_terms)):
            for w in sorted(words):
                parts = re.split(r"[\s\-]+", w)
                if len(parts) >= 2:
                    self._acronyms.setdefault("".join(p[0] for p in parts if p), kind)

    # ---- step 1: anchor
    def anchor(self, text: str) -> tuple[str, str]:
        """(anchor type, how). Deterministic: fixed order of sources and methods."""
        t = re.sub(r"\s+v?\d+(?:\.\d+)*\+?$", "", text.strip())
        low, norm = t.lower(), normalise(t)
        if not norm:
            return "NONE", ""
        if self.is_tech(text):
            return "TECHNOLOGY", "exact"
        for kind, table in self._anchors:
            if kind != "TECHNOLOGY" and low in table.values():
                return kind, "exact"
        for kind, table in self._anchors:
            if norm in table:
                if kind == "TECHNOLOGY" and any(e.lower() == table[norm] for e in self.tech_cs):
                    continue                       # case-sensitive names need the exact form
                return kind, "normalised"
        if t.isupper() and 2 <= len(t) <= 6 and t.lower() in self._acronyms:
            return self._acronyms[t.lower()], "acronym"
        if len(norm) >= FUZZY_MIN_LENGTH:
            best = (0.0, "", "")
            for kind, table in self._anchors:
                for key in sorted(table):
                    if len(key) < FUZZY_MIN_LENGTH or abs(len(key) - len(norm)) > 2:
                        continue
                    r = SequenceMatcher(None, norm, key, autojunk=False).ratio()
                    if r > best[0]:
                        best = (r, kind, key)
            if best[0] >= FUZZY_THRESHOLD:
                return best[1], "fuzzy"
        return "NONE", ""

    # ---- step 2: context cue
    def cue(self, sentence: str, start: int | None, end: int | None) -> str:
        if not sentence or start is None or end is None:
            return "NONE"
        before = " ".join(_words(sentence[:start])[-CUE_WINDOW:])
        after = " ".join(_words(sentence[end:])[:2])
        for cls in self.CUE_ORDER:
            for phrase in self.CUES[cls]:
                if cls == "ABILITY" and phrase in ("skills", "skill", "keterampilan"):
                    if after.startswith(phrase):
                        return cls
                    continue
                if before == phrase or before.endswith(" " + phrase):
                    return cls
        return "NONE"

    # ---- step 3: shape of the span itself
    def shape(self, text: str) -> str:
        words = _words(text)
        if _EXPERIENCE_RE.search(text):
            return "experience"
        if _CREDENTIAL_RE.search(text):
            return "credential"
        if len(words) >= 2 and words[0] in self.hard_verbs:
            return "hard_verb_phrase"
        if len(words) >= 2 and words[0] in self.soft_verbs:
            return "soft_verb_phrase"
        if len(words) >= 2 and (words[-1] in self.heads or words[0] in self.heads_first):
            return "head_phrase"
        if len(words) >= 2 and words[-1] in self.suffix:
            return "activity_suffix"
        raw = text.strip().split()
        if 1 <= len(raw) <= 3 and all(w[:1].isupper() or any(c.isdigit() for c in w)
                                      or any(c in ".+#" for c in w[1:]) for w in raw):
            return "name_like"
        return "other"

    def classify(self, text: str, sentence: str = "", start: int | None = None,
                 end: int | None = None) -> Decision:
        if not _words(text):
            return Decision(ABSTAIN, self.name, "empty span")
        shape = self.shape(text)
        anchor, how = self.anchor(text)
        cue = self.cue(sentence, start, end)
        for n, (s, a, c, label) in enumerate(self.TABLE):
            if s in ("*", shape) and a in ("*", anchor) and c in ("*", cue):
                return Decision(label, self.name, f"row {n}: {s}/{a}/{c}", anchor=anchor,
                                anchor_how=how, cue=cue)
        return Decision(ABSTAIN, self.name, f"no row for {shape}/{anchor}/{cue}",
                        anchor=anchor, anchor_how=how, cue=cue)

    def manifest(self) -> dict:
        return {"model": self.name, "typing_version": TYPING_VERSION,
                "fuzzy_threshold": FUZZY_THRESHOLD, "fuzzy_min_length": FUZZY_MIN_LENGTH,
                "cue_window": CUE_WINDOW, "decision_table": [list(r) for r in self.TABLE],
                **self.lexicon.manifest()}


class D3Consensus:
    """D3: accept only what D1 and D2 both say.

    A difference is a CONFLICT and is passed on. One model abstaining while the
    other decides is also a difference — recorded with `conflict_kind`, so the
    two situations can be counted separately. Both abstaining is an ABSTAIN.

    Where a conflict goes next (LLM adjudication or human review, "sesuai
    tingkat risikonya") is NOT decided here. It is decided by the registered
    risk rule in `fivetag.conflict_routing`.
    """
    name = "D3"

    def __init__(self, d1: D1LexiconRegexPriority | None = None,
                 d2: D2TaxonomyAnchoredContext | None = None):
        self.d1 = d1 or D1LexiconRegexPriority()
        self.d2 = d2 or D2TaxonomyAnchoredContext(self.d1.lexicon)

    def classify(self, text: str, sentence: str = "", start: int | None = None,
                 end: int | None = None) -> Decision:
        a = self.d1.classify(text, sentence, start, end)
        b = self.d2.classify(text, sentence, start, end)
        if a.label == b.label:
            return Decision(a.label, self.name, "D1 and D2 agree" if a.decided
                            else "D1 and D2 both abstain", parts=(a, b))
        kind = "disagree" if a.decided and b.decided else "one_abstained"
        return Decision(CONFLICT, self.name, f"D1={a.label} D2={b.label}", parts=(a, b),
                        conflict_kind=kind)

    def manifest(self) -> dict:
        return {"model": self.name, "typing_version": TYPING_VERSION,
                "d1": self.d1.manifest(), "d2": self.d2.manifest()}


# ------------------------------------------------------------------ evaluation
@dataclass
class TypingItem:
    """A span with known boundaries and a reference class."""
    sentence: str
    start: int
    end: int
    label: str
    item_id: str = ""

    @property
    def text(self) -> str:
        return self.sentence[self.start:self.end]


def evaluate_typing(model, items: Sequence[TypingItem], classes: Iterable[str] = TAG_CODES) -> dict:
    """The measures §3.3.1 names for D1-D3: macro-F1 per class, class-specific
    precision / recall, conflict rate, abstention rate, processing time, and
    the confusion counts an error taxonomy is built from.

    Precision of a class counts only DECIDED predictions; an abstention or a
    conflict is not a wrong answer, it is a non-answer, and it lowers recall.
    """
    classes = list(classes)
    t0 = time.perf_counter()
    decisions = [model.classify(it.text, it.sentence, it.start, it.end) for it in items]
    dt = time.perf_counter() - t0
    n = len(items)
    confusion: dict[str, dict[str, int]] = {}
    for it, d in zip(items, decisions):
        confusion.setdefault(it.label, {}).setdefault(d.label, 0)
        confusion[it.label][d.label] += 1
    by_class, f1s = {}, []
    for c in classes:
        tp = sum(1 for it, d in zip(items, decisions) if it.label == c and d.label == c)
        n_ref = sum(1 for it in items if it.label == c)
        n_hyp = sum(1 for d in decisions if d.label == c)
        p = tp / n_hyp if n_hyp else float("nan")
        r = tp / n_ref if n_ref else float("nan")
        f = (2 * p * r / (p + r)) if (n_ref and n_hyp and (p + r) > 0) else (
            0.0 if (n_ref or n_hyp) else float("nan"))
        by_class[c] = {"tp": tp, "n_ref": n_ref, "n_hyp": n_hyp, "precision": p,
                       "recall": r, "f1": f}
        if f == f:
            f1s.append(f)
    n_abstain = sum(1 for d in decisions if d.label == ABSTAIN)
    n_conflict = sum(1 for d in decisions if d.label == CONFLICT)
    n_decided = n - n_abstain - n_conflict
    correct = sum(1 for it, d in zip(items, decisions) if it.label == d.label)
    return {
        "model": model.name, "n": n,
        "macro_f1": sum(f1s) / len(f1s) if f1s else float("nan"),
        "n_classes_defined": len(f1s),
        "accuracy_all": correct / n if n else float("nan"),
        "accuracy_decided": correct / n_decided if n_decided else float("nan"),
        "abstention_rate": n_abstain / n if n else float("nan"),
        "conflict_rate": n_conflict / n if n else float("nan"),
        "conflict_kinds": {k: sum(1 for d in decisions if d.conflict_kind == k)
                           for k in ("disagree", "one_abstained")},
        "by_class": by_class, "confusion": confusion,
        "seconds": dt, "spans_per_second": n / dt if dt > 0 else float("inf"),
        "decisions": decisions,
    }


def items_from_reference(items) -> list[TypingItem]:
    """Conformance-style items -> one typing item per expected span."""
    return [TypingItem(it.sentence, s, e, tag, f"{it.item_id}:{s}-{e}")
            for it in items for s, e, tag in it.expected]


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from .evaluate import development_sets
    d1 = D1LexiconRegexPriority()
    d2 = D2TaxonomyAnchoredContext(d1.lexicon)
    models = [d1, d2, D3Consensus(d1, d2)]
    for name, items in development_sets().items():
        if not items:
            continue
        ti = items_from_reference(items)
        print(f"\n{name} — {len(ti)} reference spans")
        for m in models:
            r = evaluate_typing(m, ti)
            print(f"  {m.name}: macro-F1 {r['macro_f1']:.3f}   correct {r['accuracy_all']:.1%}"
                  f"   of decided {r['accuracy_decided']:.1%}   abstain {r['abstention_rate']:.1%}"
                  f"   conflict {r['conflict_rate']:.1%}")


if __name__ == "__main__":
    main()
