"""E6 — the proposed framework: NLP extraction, a conditional LLM router, a verifier.

Proposal §3.3.1, "escalation-based processing": the NLP path is the default; a
sentence goes to the LLM only when at least one named trigger fires.

  (a) a model confidence lies in the uncertain band
  (b) two extraction approaches give different labels
  (c) the sentence contains coordination or a complex mention
  (d) a long-tail or new term is not found in the dictionary / taxonomy
  (e) the relation between action, object and technology cannot be read from
      the spans alone
  (f) a competency statement must be formulated from several evidence spans —
      this one belongs to the competency stage, after extraction, and is NOT
      evaluated here.

Every decision is recorded with the triggers that caused it, so the escalation
rate, the share of each trigger and the cost of the LLM path can be reported,
and the router can be ablated one trigger at a time.

The thresholds in `RouterConfig` are STARTING VALUES. They have not been tuned,
because there is no development set yet. They must be fixed on the development
split and written into the configuration registry before the test split is
opened; changing them afterwards is a new router version.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from .base import Extractor, Pred, dedupe
from .lexicon import DEFAULT_VERSION, Lexicon, load
from .typing_d import CONFLICT, D3Consensus

ROUTER_VERSION = "router-v1"
TRIGGERS = ("a_uncertain_confidence", "b_label_disagreement", "c_coordination_or_complex",
            "d_long_tail_term", "e_relation_unclear")
_CONJ = re.compile(r"(?<![A-Za-z])(?:and|or|dan|atau|serta)(?![A-Za-z])|[/&]", re.I)
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+#.\-]*")


@dataclass(frozen=True)
class RouterConfig:
    uncertain_low: float = 0.40           # (a): low <= score < high is "uncertain"
    uncertain_high: float = 0.75
    disagreement_iou: float = 0.5         # (b): spans this similar are "the same span"
    complex_min_conjunctions: int = 2     # (c)
    triggers: tuple[str, ...] = TRIGGERS  # switch one off to ablate it
    merge: str = "llm_replaces"           # or "union_dictionary": keep dictionary hits too
    version: str = ROUTER_VERSION


@dataclass
class Route:
    sentence: str
    preds: list[Pred]
    path: str                             # nlp | llm | nlp-fallback
    triggers: tuple[str, ...] = ()
    details: dict = field(default_factory=dict)
    nlp_preds: list[Pred] = field(default_factory=list)
    llm_preds: list[Pred] | None = None

    @property
    def escalated(self) -> bool:
        return bool(self.triggers)


def _iou(a: Pred, b: Pred) -> float:
    inter = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = (a.end - a.start) + (b.end - b.start) - inter
    return inter / union if union else 0.0


class Router:
    """E6. `primary` is the NLP extractor (E0 today; E1/E2/E3 once trained).
    `secondary` is an optional second NLP extractor for trigger (b).
    `llm` is the LLM path (E5): anything with `.extract(sentence)`."""
    name = "E6"

    def __init__(self, primary: Extractor, llm: Extractor | None = None, *,
                 secondary: Extractor | None = None, typing: D3Consensus | None = None,
                 config: RouterConfig | None = None, lexicon: Lexicon | None = None):
        self.primary, self.llm, self.secondary = primary, llm, secondary
        self.config = config or RouterConfig()
        self.lexicon = lexicon or load(DEFAULT_VERSION)
        self.typing = typing if typing is not None else D3Consensus()
        lx = self.lexicon
        self._verbs = ({f for line in lx.lines("actions", "verbs") for f in line}
                       | {f for line in lx.lines("actions", "soft-verbs") for f in line})
        self._vocab: set[str] = set()
        for (_file, _section), entries in lx.sections.items():
            for e in entries:
                self._vocab.update(w.lower() for w in _TOKEN.findall(e.text))

    # ------------------------------------------------------------- triggers
    def _a(self, preds: Sequence[Pred]) -> list:
        c = self.config
        return [(p.text, p.label, round(p.score, 3)) for p in preds
                if c.uncertain_low <= p.score < c.uncertain_high]

    def _b(self, sentence: str, preds: Sequence[Pred], other: Sequence[Pred] | None) -> list:
        out = []
        if other is not None:
            for p in preds:
                for q in other:
                    if p.label != q.label and _iou(p, q) >= self.config.disagreement_iou:
                        out.append((p.text, p.label, q.text, q.label))
        for p in preds:
            d = self.typing.classify(p.text, sentence, p.start, p.end)
            if d.label == CONFLICT and d.conflict_kind == "disagree":
                out.append((p.text, p.label, "D1/D2 conflict", d.rule))
            elif d.decided and d.label != p.label:
                out.append((p.text, p.label, "typing", d.label))
        return out

    def _c(self, sentence: str, preds: Sequence[Pred]) -> list:
        out = []
        n = len(_CONJ.findall(sentence))
        if n >= self.config.complex_min_conjunctions:
            out.append(("conjunctions", n))
        out += [("coordinated mention", p.text) for p in preds
                if p.label != "TT" and _CONJ.search(p.text)]
        return out

    def _d(self, sentence: str, preds: Sequence[Pred]) -> list:
        out, first = [], True
        for m in _TOKEN.finditer(sentence):
            tok = m.group().rstrip(".")
            is_first, first = first, False
            name_like = (any(c.isdigit() for c in tok) or any(c in ".+#" for c in tok[1:])
                         or any(c.isupper() for c in tok[1:])
                         or (tok[:1].isupper() and not is_first))
            if not name_like or len(tok) < 2 or tok.lower() in self._vocab:
                continue
            if any(p.start <= m.start() and m.start() + len(tok) <= p.end for p in preds):
                continue
            out.append(tok)
        return out

    def _e(self, sentence: str, preds: Sequence[Pred]) -> list:
        if not any(p.label == "TT" for p in preds):
            return []
        phrases = [p for p in preds if p.label in ("SH", "SS")]
        return [m.group().rstrip(".") for m in _TOKEN.finditer(sentence)
                if m.group().rstrip(".").lower() in self._verbs
                and not any(p.start <= m.start() < p.end for p in phrases)]

    def triggers_for(self, sentence: str, preds: Sequence[Pred],
                     other: Sequence[Pred] | None = None) -> dict[str, list]:
        found = {"a_uncertain_confidence": self._a(preds),
                 "b_label_disagreement": self._b(sentence, preds, other),
                 "c_coordination_or_complex": self._c(sentence, preds),
                 "d_long_tail_term": self._d(sentence, preds),
                 "e_relation_unclear": self._e(sentence, preds)}
        return {k: v for k, v in found.items() if v and k in self.config.triggers}

    # ---------------------------------------------------------------- route
    def route(self, sentence: str) -> Route:
        if not sentence or not sentence.strip():
            return Route(sentence, [], "nlp")
        nlp = self.primary.extract(sentence)
        other = self.secondary.extract(sentence) if self.secondary is not None else None
        fired = self.triggers_for(sentence, nlp, other)
        if not fired:
            return Route(sentence, nlp, "nlp", nlp_preds=nlp)
        names = tuple(fired)
        llm_preds = None
        if self.llm is not None:
            try:
                llm_preds = self.llm.extract(sentence)
            except KeyError:
                llm_preds = None
        if llm_preds is None:
            return Route(sentence, nlp, "nlp-fallback", names, fired, nlp)
        # the verifier's guarantee, re-checked here: every span is in the sentence
        llm_preds = [p for p in llm_preds if sentence[p.start:p.end] == p.text]
        final = list(llm_preds)
        if self.config.merge == "union_dictionary":
            final += [p for p in nlp if p.rule.startswith("dict:")
                      and not any(q.label == p.label and _iou(p, q) > 0 for q in llm_preds)]
        return Route(sentence, dedupe(final), "llm", names, fired, nlp, llm_preds)

    def extract(self, sentence: str) -> list[Pred]:
        return self.route(sentence).preds

    def manifest(self) -> dict:
        c = self.config
        return {"extractor": self.name, "router_version": c.version,
                "thresholds": {"uncertain_low": c.uncertain_low,
                               "uncertain_high": c.uncertain_high,
                               "disagreement_iou": c.disagreement_iou,
                               "complex_min_conjunctions": c.complex_min_conjunctions},
                "thresholds_status": "starting values, not yet tuned on a development set",
                "triggers": list(c.triggers), "merge": c.merge,
                "primary": self.primary.manifest(),
                "secondary": self.secondary.manifest() if self.secondary else None,
                "llm": self.llm.manifest() if self.llm else None,
                "typing": self.typing.manifest()}


def summarise(routes: Sequence[Route]) -> dict:
    """Escalation rate and what caused it — the router's own Tabel 3.7 rows."""
    n = len(routes)
    trig = Counter(t for r in routes for t in r.triggers)
    paths = Counter(r.path for r in routes)
    esc = sum(r.escalated for r in routes)
    return {"n_sentences": n, "n_escalated": esc,
            "escalation_rate": esc / n if n else float("nan"),
            "paths": dict(paths),
            "llm_unavailable": paths.get("nlp-fallback", 0),
            "by_trigger": {t: trig.get(t, 0) for t in TRIGGERS},
            "single_trigger_only": dict(Counter(r.triggers[0] for r in routes
                                                if len(r.triggers) == 1))}
