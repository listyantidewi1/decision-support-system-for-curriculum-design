"""The typed evidence table: one row per extracted span, never detached from its
sentence and its posting.

Every later number (how many postings ask for something, which technologies go
with which action) is counted from this table, so a competency can always be
traced back to the sentences that produced it.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

from gold_standard import TAGS

_VERSION_TAIL = re.compile(r"\s+v?\d+(?:\.\d+)*\+?$")
_PAREN_TAIL = re.compile(r"\s*\([^)]*\)$")
_EDGE = " \t\r\n.,;:!?\"'`()[]{}*-–•"


def posting_of(sentence_id: str) -> str:
    """`u-aaaaaaaaaaaaaaaa#004` -> `u-aaaaaaaaaaaaaaaa`."""
    return sentence_id.split("#", 1)[0]


def canonical_key(text: str, tag: str) -> str:
    """The form under which variants of one thing are counted together.
    For a technology the version number and a bracketed acronym are dropped:
    'Java 17' and 'Java' are the same technology, 'Python 3.11' and 'python' too."""
    t = " ".join(text.split())
    if tag == "TT":                       # before the edges are trimmed: ")" is an edge
        t = _PAREN_TAIL.sub("", t.rstrip(" .,;:"))
        t = _VERSION_TAIL.sub("", t)
    return t.strip(_EDGE).lower()


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    sentence_id: str
    posting_id: str
    tag: str
    start: int
    end: int
    text: str
    key: str
    score: float
    source: str
    sentence_text: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_evidence(preds_by_sid: Mapping[str, Sequence], sentences: Mapping[str, Mapping],
                   source: str = "") -> list[Evidence]:
    """`preds_by_sid`: anything with .start/.end/.label per sentence id.
    `sentences[sid]` must give `text`; `posting_id` is read from it or from the id.

    A span whose offsets do not reproduce inside its sentence is refused: an
    evidence unit that cannot be shown in its source is not evidence.
    """
    out = []
    for sid in sorted(preds_by_sid):
        if sid not in sentences:
            raise KeyError(f"no sentence for {sid}")
        text = sentences[sid]["text"]
        posting = sentences[sid].get("posting_id") or posting_of(sid)
        for p in preds_by_sid[sid]:
            if p.label not in TAGS or not (0 <= p.start < p.end <= len(text)):
                raise ValueError(f"{sid}: span {p.start}:{p.end} {p.label} does not fit")
            span = text[p.start:p.end]
            key = canonical_key(span, p.label)
            if not key:
                continue
            out.append(Evidence(f"{sid}:{p.start}-{p.end}:{p.label}", sid, posting, p.label,
                                p.start, p.end, span, key, float(getattr(p, "score", 1.0)),
                                source or getattr(p, "source", ""), text))
    return out


def from_extractor(extractor, sentences: Mapping[str, Mapping]) -> list[Evidence]:
    preds = {sid: extractor.extract(row["text"]) for sid, row in sentences.items()}
    return build_evidence(preds, sentences, source=getattr(extractor, "name", ""))


def from_ensemble_run(run_dir, sentences: Mapping[str, Mapping],
                      accept_majority: bool = False) -> list[Evidence]:
    """Evidence from the spans a stored multi-LLM run accepted automatically."""
    from labeling.consensus import ConsensusConfig, accepted_spans_by_sid, decide_sentence
    from labeling.versus_gold import load_run
    text = {sid: row["text"] for sid, row in sentences.items()}
    models, outs, _ = load_run(run_dir, text)
    cfg = ConsensusConfig(models=tuple(models), accept_majority=accept_majority)
    decided = [decide_sentence(sid, text[sid], [outs[m][sid] for m in models if sid in outs[m]],
                               cfg) for sid in text]
    return build_evidence(accepted_spans_by_sid(decided), sentences,
                          source="ensemble:" + ("majority" if accept_majority else "unanimous"))


def denominators(sentences: Mapping[str, Mapping]) -> dict:
    postings = {row.get("posting_id") or posting_of(sid) for sid, row in sentences.items()}
    return {"n_sentences": len(sentences), "n_postings": len(postings)}


def by_tag(evidence: Iterable[Evidence]) -> dict[str, int]:
    out = {t: 0 for t in TAGS}
    for e in evidence:
        out[e.tag] += 1
    return out


def profile(evidence: Iterable[Evidence], tag: str, n_postings: int, top: int = 50) -> list[dict]:
    """How many postings mention each thing of one tag — used for the
    technology profile and the work-behaviour profile."""
    postings: dict[str, set] = {}
    surface: dict[str, dict[str, int]] = {}
    for e in evidence:
        if e.tag != tag:
            continue
        postings.setdefault(e.key, set()).add(e.posting_id)
        surface.setdefault(e.key, {}).setdefault(e.text, 0)
        surface[e.key][e.text] += 1
    rows = [{"key": k, "text": max(sorted(surface[k]), key=lambda s: surface[k][s]),
             "n_postings": len(v), "share_of_postings": len(v) / n_postings if n_postings else 0.0}
            for k, v in postings.items()]
    return sorted(rows, key=lambda r: (-r["n_postings"], r["key"]))[:top]
