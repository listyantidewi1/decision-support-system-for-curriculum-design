"""Align verified spans across models into candidate entities.

Proposal §3.4.3: "Penyelarasan span dilakukan secara bertingkat. Exact match
digunakan apabila batas awal dan akhir dua prediksi adalah identik. Apabila
terdapat perbedaan kecil pada batas span, kandidat dapat disejajarkan
menggunakan token overlap atau Intersection over Union (IoU) dengan ambang yang
ditetapkan pada development set."

Two tiers, in that order:

  1. EXACT   spans with identical boundaries form one candidate.
  2. RELAXED candidates are merged when their boundaries overlap enough.

Alignment is label-blind on purpose: it only decides which spans are "about the
same stretch of text". Whether the models then agree on the tag is the vote in
`consensus.py`.

Two design decisions the proposal leaves open, made here and reported in every
output so they can be revisited on the development set:

  * A candidate holds AT MOST ONE span per model. Two spans from one model are
    two claims, never one.
  * Merging uses COMPLETE linkage: every pair of spans in a candidate must clear
    the threshold. Single linkage would chain A~B and B~C into one candidate
    even when A and C barely touch, and report that as three models agreeing.

Nothing here creates text. A candidate is a set of source offsets; "penyelarasan
semantik tidak digunakan untuk mengubah teks hasil ekstraksi menjadi teks baru".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from .schema import VALID, VerifiedOutput, VerifiedSpan

IOU_UNITS = ("char", "token")
_TOKEN_RE = re.compile(r"\S+")          # the tokenizer the annotator UI uses


def span_iou(a: VerifiedSpan, b: VerifiedSpan, sentence: str = "",
             unit: str = "char") -> float:
    """Intersection over union of two spans, over characters or tokens.

    `char` is the rule `gold_standard.iaa` uses for the relaxed human-agreement
    figure, so with the default a pair of model spans is "aligned" under exactly
    the definition by which two experts' spans "partially agree".
    """
    if unit == "char":
        inter = max(0, min(a.end, b.end) - max(a.start, b.start))
        union = (a.end - a.start) + (b.end - b.start) - inter
        return inter / union if union > 0 else 0.0
    if unit == "token":
        def toks(s: VerifiedSpan) -> set[int]:
            return {i for i, m in enumerate(_TOKEN_RE.finditer(sentence))
                    if m.start() < s.end and m.end() > s.start}
        ta, tb = toks(a), toks(b)
        union = len(ta | tb)
        return len(ta & tb) / union if union else 0.0
    raise ValueError(f"unit must be one of {IOU_UNITS}, got {unit!r}")


@dataclass
class Candidate:
    """One candidate entity: at most one verified span per model."""
    members: dict[str, VerifiedSpan] = field(default_factory=dict)

    @property
    def start(self) -> int:
        return min(s.start for s in self.members.values())

    @property
    def end(self) -> int:
        return max(s.end for s in self.members.values())

    @property
    def exact(self) -> bool:
        return len({s.bounds for s in self.members.values()}) == 1

    def min_pairwise_iou(self, sentence: str, unit: str) -> float:
        spans = list(self.members.values())
        if len(spans) < 2:
            return 1.0
        return min(span_iou(spans[i], spans[j], sentence, unit)
                   for i in range(len(spans)) for j in range(i + 1, len(spans)))

    def _key(self) -> tuple:
        return tuple(sorted((s.start, s.end, m) for m, s in self.members.items()))


def align(outputs: Sequence[VerifiedOutput], sentence: str,
          iou_threshold: float = 0.5, unit: str = "char") -> list[Candidate]:
    """Group the verified spans of all models into candidates.

    Deterministic: the result does not depend on the order of `outputs` or of
    the spans within them.
    """
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")
    if unit not in IOU_UNITS:
        raise ValueError(f"unit must be one of {IOU_UNITS}, got {unit!r}")
    models = [o.model for o in outputs]
    if len(set(models)) != len(models):
        raise ValueError(f"more than one output for the same model: {sorted(models)}")

    # tier 1 — identical boundaries
    by_bounds: dict[tuple[int, int], Candidate] = {}
    for o in outputs:
        if o.status != VALID:
            continue
        for s in o.spans:
            cand = by_bounds.setdefault(s.bounds, Candidate())
            if o.model in cand.members:
                # the verifier rejects one span carrying two labels, so this
                # means verify_output was bypassed
                raise ValueError(
                    f"model {o.model!r} has two spans at {s.bounds}; outputs must "
                    f"come from labeling.verify.verify_output")
            cand.members[o.model] = s
    clusters = sorted(by_bounds.values(), key=Candidate._key)

    # tier 2 — complete-linkage agglomeration under the threshold
    def linkage(a: Candidate, b: Candidate) -> float | None:
        if set(a.members) & set(b.members):
            return None                                   # one span per model
        score = min(span_iou(x, y, sentence, unit)
                    for x in a.members.values() for y in b.members.values())
        return score if score >= iou_threshold else None

    while True:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                score = linkage(clusters[i], clusters[j])
                if score is None:
                    continue
                # highest IoU first; ties broken by position, so the outcome is
                # a function of the spans alone
                rank = (-score, clusters[i]._key(), clusters[j]._key())
                if best is None or rank < best[0]:
                    best = (rank, i, j)
        if best is None:
            break
        _, i, j = best
        merged = Candidate({**clusters[i].members, **clusters[j].members})
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [merged]
        clusters.sort(key=Candidate._key)

    return sorted(clusters, key=lambda c: (c.start, c.end, c._key()))
