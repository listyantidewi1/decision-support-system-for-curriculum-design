"""The one prediction type and the one interface every extractor implements."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Protocol, runtime_checkable

from gold_standard.iaa import Span

from . import TAG_CODES


@dataclass(frozen=True)
class Pred:
    """One predicted span. Offsets are characters into the sentence, end exclusive."""
    start: int
    end: int
    label: str
    text: str
    score: float = 1.0          # the extractor's own confidence, 0..1
    source: str = ""            # which extractor produced it (E0, E1, ...)
    rule: str = ""              # which rule / path inside that extractor

    def __post_init__(self):
        if self.label not in TAG_CODES:
            raise ValueError(f"unknown tag {self.label!r}")
        if not (0 <= self.start < self.end):
            raise ValueError(f"bad offsets {self.start}:{self.end}")

    @property
    def bounds(self) -> tuple[int, int]:
        return (self.start, self.end)

    def to_dict(self) -> dict:
        return asdict(self)


def make_pred(sentence: str, start: int, end: int, label: str, *, score: float = 1.0,
              source: str = "", rule: str = "") -> Pred:
    """A Pred whose text is read from the sentence, never supplied separately."""
    if not (0 <= start < end <= len(sentence)):
        raise ValueError(f"offsets {start}:{end} outside a sentence of {len(sentence)} chars")
    return Pred(start, end, label, sentence[start:end], score, source, rule)


def to_spans(preds: Iterable) -> list[Span]:
    """Anything with .start/.end/.label -> the span type the metrics engine takes."""
    return [Span(p.start, p.end, p.label) for p in preds]


@runtime_checkable
class Extractor(Protocol):
    """An extraction condition. `extract` must be deterministic for a fixed
    manifest: same sentence, same output."""
    name: str

    def extract(self, sentence: str) -> list[Pred]: ...

    def manifest(self) -> dict:
        """Everything needed to reproduce the output: versions and hashes."""
        ...


def dedupe(preds: Iterable[Pred]) -> list[Pred]:
    """Drop exact duplicates (same bounds and tag), keep the highest score; sorted."""
    best: dict[tuple[int, int, str], Pred] = {}
    for p in preds:
        k = (p.start, p.end, p.label)
        if k not in best or p.score > best[k].score:
            best[k] = p
    return sorted(best.values(), key=lambda p: (p.start, p.end, p.label))
