"""E0 — rule / dictionary extraction: "baseline paling transparan dan berbiaya rendah".

No model, no training, no network. Every span is produced by a named rule over a
versioned lexicon (`fivetag.lexicon`), so each prediction can be explained in one
line and reproduced from the manifest.

The rules implement the annotation guideline (AG-RPL-LIMA-v3.0) as far as word
lists and patterns can:

  dict        an entry of a lexicon                       TT, SS, WB, KN, SH
  version     a technology keeps its version number       6.10  "Java 17"
  acronym     full name + bracketed acronym is one span   6.6   "Amazon Web Services (AWS)"
  verb-led    action verb(s) + object, cut before         6.2, 6.8, 4.1
              "using / for / in / ..."                    "Develop REST APIs"
  soft-verb   the same for interpersonal verbs            "Collaborate with QA teams"
  head        <modifiers> + principles / concepts / ...   7.2   "Agile principles"
  suffix      <modifiers> + development                   7.4   "web development"
  job titles  technologies inside a job title are dropped 7.10

and resolve overlaps the way the guideline does: the longest span of a tag wins
(6.4), a technology may sit inside an action phrase (6.9), nothing else may.

`score` is the rule's fixed confidence, not a probability: 1.0 for a dictionary
hit, lower for a pattern. The router (E6) reads it for trigger (a).

A rule baseline is expected to be weak on anything not in its lists. That is
the point of having it: it is the floor the other conditions are read against.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .base import Pred, dedupe, make_pred
from .lexicon import DEFAULT_VERSION, Lexicon, load

RULES_VERSION = "e0-rules-v2"

SCORE_DICT = 1.0
SCORE_ACTIVITY = 0.9
SCORE_HEAD = 0.7
SCORE_VERB = 0.6
SCORE_SUFFIX = 0.6

MAX_OBJECT_TOKENS = 8
MAX_MODIFIERS = 4
# generic nouns that may follow "<x> development" without making it something else
GENERIC_TAIL = {"experience", "skills", "skill", "expertise", "background", "knowledge"}

CONJ = {"and", "or", "dan", "atau", "serta", "&"}
# words after which a base-form verb really is a verb ("to develop", "will design")
BASE_CONTEXT = {"to", "will", "can", "must", "should", "you'll", "we'll", "you", "we", "they",
                "also", "then", "help", "mampu", "dapat", "bisa", "akan", "harus"}
# ... and after which an -ing form is ("experience in developing")
ING_CONTEXT = BASE_CONTEXT | {"for", "in", "of", "with", "by", "on", "experience",
                              "including", "like", "while", "when"}

# Rules v2 (EXP-035). In v1 a comma ended a verb's object, which cut two frequent
# patterns before their noun: a list of qualities ("Write clean, efficient,
# maintainable, and well-tested code" -> "Write clean") and a list of verbs that
# goes on with a verb the lexicon lacks ("Design, develop, and manage databases"
# -> "Design, develop"). 97 competencies of run e0_pool_all_v8 were such cut phrases.
# The quality words a listed object may open with; the most frequent were taken
# from the words that ended those cut phrases.
QUALITY_MODIFIERS = {
    "clean", "scalable", "secure", "high-quality", "robust", "reliable", "responsive",
    "high-performance", "high-performing", "performant", "intuitive", "efficient", "modular",
    "quality", "fast", "flexible", "large-scale", "user-centric", "user-centered",
    "user-centred", "seamless", "reusable", "user-friendly", "available", "highly-available",
    "maintainable", "clear", "testable", "readable", "well-tested", "well-documented",
    "well-structured", "well-designed", "well-architected", "resilient", "fault-tolerant",
    "extensible", "stable", "accessible", "interactive", "dynamic", "lightweight", "elegant",
    "consistent", "simple", "concise", "idiomatic", "compliant", "cost-effective",
    "cost-efficient", "cloud-native", "production-ready", "production-grade",
    "enterprise-grade", "mission-critical", "data-driven", "real-time", "cross-platform",
    "mobile-friendly", "pixel-perfect", "beautiful", "engaging", "innovative", "modern",
    "optimized", "optimised", "automated", "distributed", "low-latency", "end-to-end",
    "full-stack", "custom", "complex", "critical", "new", "existing", "efficient", "secure",
    "safe", "smooth", "visually", "appealing", "polished", "delightful", "inclusive"}
# noun endings: a word with one is not taken for an unlisted verb in a verb list
_NOUNISH = ("tion", "sion", "ment", "ness", "ity", "ance", "ence", "ship", "ware", "ics",
            "ism", "ist", "or", "er")
# an Indonesian verb: the meN- or ber- prefix ("mengevaluasi", "mengelola", "berkolaborasi")
_ID_VERB = re.compile(r"^(?:me|ber)[a-z]{4,}$")


def _plain(word: str) -> str:
    return word.replace("‑", "-").replace("‐", "-")


_TOKEN = re.compile(r"\S+")
_OPEN = "([{\"'“‘"
_CLOSE = ".,;:!?)]}\"'”’"
_CLAUSE_END = set(".;:!?")
_VERSION = re.compile(r"\s+v?\d+(?:\.\d+)*\+?(?![A-Za-z0-9])"
                      r"(?!\s*\+?\s*(?:years?|yrs?|tahun|bulan|months?)\b)", re.I)
_PAREN = re.compile(r"\s*\(([A-Za-z0-9.+#/\- ]{1,40})\)")


@dataclass
class Tok:
    start: int
    end: int
    low: str
    punct: bool = False            # a bullet or dash on its own
    opens: bool = False            # had an opening bracket before it
    trail: str = ""                # closing punctuation stripped from its end

    @property
    def breaks(self) -> bool:
        return bool(self.trail) and any(c in ".,;:!?)" for c in self.trail)

    @property
    def ends_clause(self) -> bool:
        return any(c in _CLAUSE_END for c in self.trail)


def tokenize(sentence: str) -> list[Tok]:
    toks = []
    for m in _TOKEN.finditer(sentence):
        s, e = m.span()
        opens = False
        while s < e and sentence[s] in _OPEN:
            opens = opens or sentence[s] in "(["
            s += 1
        trail_start = e
        while trail_start > s and sentence[trail_start - 1] in _CLOSE:
            trail_start -= 1
        # keep "C++"-style and "(AWS)"-style characters out of this: only closers
        trail = sentence[trail_start:e]
        e = trail_start
        if s >= e or not any(c.isalnum() for c in sentence[s:e]):
            toks.append(Tok(m.start(), m.end(), sentence[m.start():m.end()], punct=True))
            continue
        toks.append(Tok(s, e, sentence[s:e].lower().replace("\u2019", "'"), opens=opens,
                        trail=trail))
    return toks


@dataclass(frozen=True)
class _Cand:
    start: int
    end: int
    label: str
    score: float
    rule: str
    obj_start: int = -1            # verb-led only: where the object begins

    @property
    def length(self) -> int:
        return self.end - self.start

    def inside(self, other: "_Cand") -> bool:
        return other.start <= self.start and self.end <= other.end

    def overlaps(self, other: "_Cand") -> bool:
        return self.start < other.end and other.start < self.end


def _is_acronym_of(short: str, long: str) -> bool:
    initials = "".join(w[0] for w in re.split(r"[\s\-]+", long) if w and w[0].isalnum())
    return len(short) >= 2 and short.replace(".", "").lower() == initials.lower()


class E0Rules:
    """The E0 condition. Stateless after construction; safe to reuse."""
    name = "E0"

    def __init__(self, lexicon: Lexicon | None = None, version: str = DEFAULT_VERSION):
        lx = self.lexicon = lexicon or load(version)
        self._dict = [
            ("TT", lx.patterns("technologies"), SCORE_DICT, "dict:technology"),
            ("SS", lx.patterns("soft_skills"), SCORE_DICT, "dict:soft_skill"),
            ("WB", lx.patterns("work_behaviors"), SCORE_DICT, "dict:work_behavior"),
            ("KN", lx.patterns("knowledge", "terms"), SCORE_DICT, "dict:knowledge_term"),
            ("KN", lx.patterns("knowledge", "groups"), SCORE_DICT, "dict:knowledge_group"),
            ("SH", lx.patterns("actions", "activity-nouns"), SCORE_ACTIVITY, "dict:activity_noun"),
        ]
        self._known = (lx.words("technologies") | lx.words("knowledge", "terms"))
        self._hard = self._verb_forms(lx.lines("actions", "verbs"))
        self._soft = self._verb_forms(lx.lines("actions", "soft-verbs"))
        self._any_verb = set(self._hard) | set(self._soft)
        self._stops = lx.words("boundaries", "stops")
        self._function = lx.words("boundaries", "function-words")
        lead = lx.words("boundaries", "lead-ins")
        self._lead_words = {w.split()[0] for w in lead} | {w for w in lead if " " not in w}
        self._heads = lx.words("knowledge", "heads")
        self._heads_first = lx.words("knowledge", "heads-first")
        self._suffix = lx.words("actions", "activity-suffix")
        self._titles = lx.words("boundaries", "title-heads")

    @staticmethod
    def _verb_forms(lines: list[list[str]]) -> dict[str, str]:
        """form -> kind: 'base', 'ing', 's', or 'any' (single-form, Indonesian)."""
        out: dict[str, str] = {}
        for forms in lines:
            if len(forms) == 1:
                out[forms[0]] = "any"
                continue
            base_forms = {forms[0]} | {f for f in forms if not f.endswith(("ing", "s"))}
            for f in forms:
                out[f] = "base" if f in base_forms else ("ing" if f.endswith("ing") else "s")
        return out

    # ------------------------------------------------------------ dictionary
    def _dictionary(self, sentence: str) -> list[_Cand]:
        out = []
        for label, patterns, score, rule in self._dict:
            for pat in patterns:
                for m in pat.finditer(sentence):
                    s, e = m.span()
                    r = rule
                    if label == "TT":
                        v = _VERSION.match(sentence, e)
                        if v:
                            e, r = v.end(), rule + "+version"
                    if rule in ("dict:technology", "dict:knowledge_term"):
                        p = _PAREN.match(sentence, e)
                        if p:
                            inner, outer = p.group(1).strip(), sentence[s:e]
                            if (_is_acronym_of(inner, outer) or _is_acronym_of(outer, inner)
                                    or inner.lower() in self._known):
                                e, r = p.end(), r + "+acronym"
                    out.append(_Cand(s, e, label, score, r))
        return out

    # -------------------------------------------------------------- verb-led
    def _start_ok(self, toks: list[Tok], i: int, kind: str, last_end: int) -> bool:
        if i == 0 or kind == "any":
            return True
        prev = toks[i - 1]
        if prev.punct or prev.ends_clause:
            return True
        if prev.low.endswith("ly") and not prev.breaks and i >= 2 and kind != "s":
            return toks[i - 2].low in BASE_CONTEXT      # "will proactively automate"
        if prev.breaks:                               # a comma: only as a coordination
            return i - 1 == last_end
        if kind == "s":
            return False
        if prev.low in CONJ:
            return i - 2 == last_end
        if prev.low in BASE_CONTEXT:
            return True
        return kind == "ing" and prev.low in ING_CONTEXT

    @staticmethod
    def _quality(toks: list[Tok], x: int) -> bool:
        """A quality word, or an intensifier in front of one ("highly scalable")."""
        if _plain(toks[x].low) in QUALITY_MODIFIERS:
            return True
        return (toks[x].low.endswith("ly") and not toks[x].breaks and x + 1 < len(toks)
                and _plain(toks[x + 1].low) in QUALITY_MODIFIERS)

    def _past_qualities(self, toks: list[Tok], first: int, k: int) -> int | None:
        """Rules v2. The object toks[first..k] ends in a comma. When it is only quality
        words ("Write clean, efficient, maintainable, and well-tested code"), the index
        of the first word after the listed qualities (where the noun starts); None when
        the object is not such a list or no noun follows (then the comma ends it, as in v1)."""
        n = len(toks)
        if not all(self._quality(toks, x) for x in range(first, k + 1)):
            return None
        m = k + 1
        for _ in range(10):
            if m >= n:
                return None
            tk = toks[m]
            if tk.punct or tk.opens or tk.low in self._stops:
                return None
            if tk.low in CONJ:
                # "clean, and test it": a verb after the conjunction starts a new action
                if m + 1 < n and toks[m + 1].low in self._any_verb \
                        and not self._quality(toks, m + 1):
                    return None
                m += 1
                continue
            if self._quality(toks, m):
                if tk.breaks and tk.trail != ",":    # the clause ends on a quality: no noun
                    return None
                m += 1
                continue
            return m
        return None

    def _verb_like(self, tk: Tok, kind: str) -> bool:
        """Can this word be one more verb of a coordinated verb list of this form?"""
        w = tk.low
        if tk.punct or tk.opens or not w.isalpha() or w in self._stops or w in self._function \
                or w in self._known or w in self._heads:
            return False
        if w in self._any_verb:
            return True
        if kind == "ing":
            return w.endswith("ing") and len(w) > 5
        if kind == "any":                             # Indonesian
            return bool(_ID_VERB.match(w))
        if kind == "base":
            return (len(w) >= 3 and not w.endswith(("ing", "s"))
                    and not any(w.endswith(e) and len(w) > len(e) + 2 for e in _NOUNISH))
        return False

    def _verb_series(self, toks: list[Tok], j: int, kind: str) -> int | None:
        """Rules v2. toks[j], a verb of the list, ends with a comma and the next word is
        not a verb of the lexicon. When the list goes on with verb-like words and ends
        with a conjunction and a verb ("design, develop, and manage databases";
        "designing, developing, extending, and supporting custom tools"; "merancang,
        mengimplementasikan, dan mengevaluasi proses"), the index of its last verb."""
        n, m = len(toks), j + 1
        for _ in range(6):
            final = m < n and toks[m].low in CONJ
            if final:
                m += 1
            if m >= n or not self._verb_like(toks[m], kind):
                return None
            if final:
                return None if toks[m].breaks else m   # its object must follow
            if toks[m].trail == ",":                   # "..., extending, ..."
                m += 1
                continue
            if not toks[m].trail and m + 1 < n and toks[m + 1].low in CONJ:   # "extend and ship"
                m += 1
                continue
            return None
        return None

    def _verb_led(self, sentence: str, toks: list[Tok], forms: dict[str, str], label: str,
                  rule: str, verb_tokens: set[int], activity_starts: set[int]) -> list[_Cand]:
        out, n, i, last_end = [], len(toks), 0, -9
        while i < n:
            t = toks[i]
            if t.punct or t.low not in forms or not self._start_ok(toks, i, forms[t.low], last_end):
                i += 1
                continue
            j = i                                     # the verb group: "design, develop and test"
            j_known = None                            # v2: where the group of known verbs ended
            while True:
                nxt = j + 1
                if nxt < n and toks[j].trail in ("", ",") and toks[nxt].low in CONJ \
                        and not toks[nxt].breaks and nxt + 1 < n and toks[nxt + 1].low in forms:
                    j = nxt + 1
                    continue
                if nxt < n and toks[j].trail == "," and toks[nxt].low in forms:
                    j = nxt
                    continue
                if toks[j].trail == ",":                  # v2: the list goes on with
                    last = self._verb_series(toks, j, forms[t.low])   # verbs not listed
                    if last is not None:
                        j_known, j = j, last
                break
            end = None
            if not toks[j].breaks:
                k, limit = j + 1, j + MAX_OBJECT_TOKENS
                while k < n and k <= limit:
                    tk = toks[k]
                    if tk.punct or tk.opens or tk.low in self._stops:
                        break
                    if (tk.low in CONJ or tk.low == "to") and k + 1 < n and (
                            toks[k + 1].low in self._any_verb
                            or toks[k + 1].start in activity_starts
                            or (toks[k + 1].low.endswith("ly") and k + 2 < n
                                and toks[k + 2].low in self._any_verb)):
                        break
                    end = k
                    if tk.breaks:
                        # v2: a comma inside a list of qualities does not end the object
                        head = self._past_qualities(toks, j + 1, k) if tk.trail == "," else None
                        if head is None:
                            break
                        limit += head - k                 # the list does not count
                        k = head
                        continue
                    k += 1
                while end is not None and end > j and (toks[end].low in self._function
                                                       or toks[end].low in CONJ):
                    end -= 1
                if end is not None and end <= j:
                    end = None
            if end is None and j_known is not None:
                # v2's longer verb list found no object: a list of verbs is no hard skill
                # ("apply, and believe", "build, solve, and operate"), so back to v1
                j = j_known
            if end is None and j > i:
                end = j                               # "developing and deploying": verbs alone
            if (end is not None and forms[t.low] == "ing" and j == i
                    and toks[end].low in self._heads):
                end = None                            # "testing principles" is a topic (7.4)
            if end is None:
                i = j + 1
                continue
            verb_tokens.update(range(i, j + 1))
            obj = toks[j + 1].start if end > j else toks[end].end
            out.append(_Cand(t.start, toks[end].end, label, SCORE_VERB, rule, obj))
            last_end = end
            i = end + 1
        return out

    # ---------------------------------------------------------- head patterns
    def _modifiers_before(self, toks: list[Tok], h: int, verb_tokens: set[int]) -> int:
        """Index of the first modifier token of the phrase whose head is toks[h]."""
        first = h
        for k in range(h - 1, max(-1, h - 1 - MAX_MODIFIERS), -1):
            tk = toks[k]
            if (tk.punct or tk.breaks or k in verb_tokens or tk.low in self._function
                    or tk.low in CONJ or tk.low in self._lead_words or tk.low in self._stops
                    or tk.low in self._suffix):
                break
            first = k
        return first

    def _head_patterns(self, toks: list[Tok], verb_tokens: set[int]) -> list[_Cand]:
        out, n = [], len(toks)
        for h, t in enumerate(toks):
            if t.punct:
                continue
            if t.low in self._heads:
                first = self._modifiers_before(toks, h, verb_tokens)
                if first < h and not toks[h].opens:
                    out.append(_Cand(toks[first].start, t.end, "KN", SCORE_HEAD, "head:knowledge"))
            if t.low in self._suffix:
                nxt = toks[h + 1] if h + 1 < n else None
                closed = (nxt is None or t.breaks or nxt.punct or nxt.low in self._function
                          or nxt.low in self._stops or nxt.low in CONJ
                          or nxt.low in GENERIC_TAIL)
                first = self._modifiers_before(toks, h, verb_tokens)
                if closed and first < h:
                    out.append(_Cand(toks[first].start, t.end, "SH", SCORE_SUFFIX,
                                     "suffix:activity"))
            if t.low in self._heads_first and not t.breaks:
                last = h
                for k in range(h + 1, min(n, h + 1 + MAX_MODIFIERS - 1)):
                    tk = toks[k]
                    if tk.punct or tk.opens or tk.low in self._function or tk.low in CONJ \
                            or tk.low in self._stops:
                        break
                    last = k
                    if tk.breaks:
                        break
                if last > h:
                    out.append(_Cand(t.start, toks[last].end, "KN", SCORE_HEAD,
                                     "head-first:knowledge"))
        return out

    # ------------------------------------------------------------ job titles
    def _title_regions(self, toks: list[Tok], verb_tokens: set[int]) -> list[tuple[int, int]]:
        out = []
        for h, t in enumerate(toks):
            if h == 0 or t.punct or t.low not in self._titles or h in verb_tokens:
                continue
            first = h
            for k in range(h - 1, max(-1, h - 4), -1):
                tk = toks[k]
                if tk.punct or tk.breaks or tk.low in self._function or tk.low in CONJ \
                        or k in verb_tokens:
                    break
                first = k
            out.append((toks[first].start, t.end))
        return out

    # ------------------------------------------------------------- resolving
    @staticmethod
    def _resolve(sentence: str, cands: list[_Cand],
                 titles: list[tuple[int, int]]) -> list[_Cand]:
        # 7.10: a dictionary hit inside a job title is part of the title, not a requirement
        cands = [c for c in cands
                 if not (c.rule.startswith("dict:") and c.label in ("TT", "KN")
                         and any(a <= c.start and c.end <= b for a, b in titles))]
        # 6.8: a verb whose whole object is technology names is "using X", not an action
        tech = sorted((c for c in cands if c.label == "TT"), key=lambda c: c.start)

        def only_technology(c: _Cand) -> bool:
            pos = c.obj_start
            for t in tech:
                if t.end <= pos:
                    continue
                if t.start > pos and sentence[pos:t.start].strip(" ,/&"):
                    break
                pos = max(pos, t.end)
            return pos >= c.end
        cands = [c for c in cands
                 if not (c.rule == "verb-led" and 0 <= c.obj_start < c.end
                         and only_technology(c))]
        # an "action phrase" that is really a dictionary term ("design patterns") is the term
        terms = [c for c in cands if c.rule.startswith("dict:") and c.label in ("KN", "TT")]
        cands = [c for c in cands
                 if not (c.label == "SH" and not c.rule.startswith("dict:")
                         and any(c.inside(t) for t in terms))]
        # 6.4: within one tag the longest span wins
        kept: list[_Cand] = []
        for c in sorted(cands, key=lambda c: (-c.length, -c.score, c.start, c.label)):
            if not any(k.label == c.label and k.overlaps(c) for k in kept):
                kept.append(c)
        # 6.9: only a technology may sit inside an action / soft-skill phrase
        phrases = [k for k in kept if k.rule in ("verb-led", "soft-verb")]
        kept = [c for c in kept
                if c.label == "TT" or c in phrases
                or not any(c.inside(p) and c is not p for p in phrases)]
        # different tags crossing without containment: keep the surer, then the longer
        out: list[_Cand] = []
        for c in sorted(kept, key=lambda c: (-c.score, -c.length, c.start, c.label)):
            crossing = [k for k in out if k.label != c.label and k.overlaps(c)
                        and not c.inside(k) and not k.inside(c)]
            same_extent = [k for k in out if k.label != c.label
                           and (k.start, k.end) == (c.start, c.end)]
            if not crossing and not same_extent:
                out.append(c)
        return out

    # ----------------------------------------------------------------- public
    def extract(self, sentence: str) -> list[Pred]:
        if not sentence or not sentence.strip():
            return []
        toks = tokenize(sentence)
        verb_tokens: set[int] = set()
        cands = self._dictionary(sentence)
        acts = {c.start for c in cands if c.rule == "dict:activity_noun"}
        cands += self._verb_led(sentence, toks, self._hard, "SH", "verb-led", verb_tokens, acts)
        cands += self._verb_led(sentence, toks, self._soft, "SS", "soft-verb", verb_tokens, acts)
        cands += self._head_patterns(toks, verb_tokens)
        final = self._resolve(sentence, cands, self._title_regions(toks, verb_tokens))
        return dedupe(make_pred(sentence, c.start, c.end, c.label, score=c.score,
                                source=self.name, rule=c.rule) for c in final)

    def manifest(self) -> dict:
        return {"extractor": self.name, "rules_version": RULES_VERSION,
                **self.lexicon.manifest()}
