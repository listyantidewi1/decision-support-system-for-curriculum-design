"""The output contract between a labelling model and the verifier.

A model is asked for exactly this JSON object for one sentence:

    {"spans": [
        {"text": "automated testing", "label": "SH", "start": 24, "end": 41},
        {"text": "Jest",              "label": "TT"}
    ]}

  * `label`  one of the five tags, as a code (SH) or a full name (SKILL_HARD).
  * `text`   the span copied verbatim from the sentence.
  * `start`/`end`  character offsets into the sentence, end exclusive. Optional
             when `text` is given: the verifier then locates the text itself,
             and needs `occurrence` (1-based) if the text appears more than once.
  * `{"spans": []}` is a valid answer and means "no entity in this sentence".

Nothing a model writes is trusted. `labeling.verify` turns this into
`VerifiedSpan`s whose offsets were checked against the source, or into logged
rejections.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from gold_standard import TAGS

ALLOWED_LABELS: tuple[str, ...] = tuple(TAGS)          # ("SH","SS","KN","TT","WB")

# Rejection categories named by the proposal: a prediction that fails the
# deterministic check "dicatat sebagai INVALID atau UNSUPPORTED".
INVALID = "INVALID"              # malformed: bad JSON, bad schema, bad label, bad offsets
UNSUPPORTED = "UNSUPPORTED"      # well-formed but not backed by the source text
NOTE = "NOTE"                    # logged, harmless (an exact duplicate)

VALID = "VALID"


def _label_key(raw: str) -> str:
    return re.sub(r"[\s\-]+", "_", raw.strip()).upper()


# Deterministic, closed vocabulary. The proposal writes the tags as
# "skill-hard, skill-soft, knowledge, tool-technology, work-behavior"; the
# guideline uses SKILL_HARD ... and the codes SH ... . All three spellings map
# to the code; anything else is an INVALID label — never a best guess.
_LABEL_ALIASES: dict[str, str] = {}
for _code, _full in TAGS.items():
    _LABEL_ALIASES[_label_key(_code)] = _code
    _LABEL_ALIASES[_label_key(_full)] = _code


def normalise_label(raw) -> str | None:
    """Tag code for `raw`, or None when it is not one of the five tags."""
    if not isinstance(raw, str):
        return None
    return _LABEL_ALIASES.get(_label_key(raw))


@dataclass(frozen=True)
class VerifiedSpan:
    """A span whose offsets were checked against the source sentence."""
    start: int
    end: int
    label: str
    text: str                      # == sentence[start:end], always
    anchor: str                    # how the offsets were established
    flags: tuple[str, ...] = ()

    @property
    def bounds(self) -> tuple[int, int]:
        return (self.start, self.end)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["flags"] = list(self.flags)
        return d


@dataclass(frozen=True)
class Rejection:
    """One thing the verifier refused, kept for the INVALID / UNSUPPORTED log."""
    code: str                      # e.g. UNSUPPORTED_TEXT_NOT_FOUND
    category: str                  # INVALID | UNSUPPORTED | NOTE
    level: str                     # "output" (whole response) | "span"
    detail: str
    item: str = ""                 # the offending fragment, truncated
    # A rejected span means we no longer know everything this model claimed
    # about the sentence, so its silence elsewhere cannot be read as "NONE".
    blocks_none: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VerifiedOutput:
    """One model's verified answer for one sentence."""
    model: str
    sentence_id: str
    status: str                                  # VALID | INVALID (whole output)
    spans: list[VerifiedSpan] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)

    @property
    def can_vote_none(self) -> bool:
        """True when this model's silence on a candidate is a real NONE vote.

        It must have produced a valid output AND had no span thrown away. A
        model whose span was rejected did claim something we could not use; to
        count it as agreeing "nothing here" would manufacture consensus out of a
        failure, so it casts no vote on candidates it has no valid span for.
        """
        return self.status == VALID and not any(r.blocks_none for r in self.rejections)

    def to_dict(self) -> dict:
        return {"model": self.model, "sentence_id": self.sentence_id,
                "status": self.status,
                "spans": [s.to_dict() for s in self.spans],
                "rejections": [r.to_dict() for r in self.rejections],
                "repairs": list(self.repairs)}
