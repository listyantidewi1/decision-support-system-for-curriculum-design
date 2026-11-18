"""Consensus engine — the six-way vote and the acceptance rule.

Proposal §3.4.3, which this file follows clause by clause:

  "Untuk setiap kandidat, setiap LLM memberikan salah satu dari enam kemungkinan
   keputusan, yaitu lima label penelitian atau NONE."
        -> every model votes one of SH/SS/KN/TT/WB or NONE on every candidate.

  "Unanimous consensus terjadi apabila seluruh LLM memberikan label dan batas
   span yang sama."
        -> UNANIMOUS: all n models, same label, identical boundaries.

  "Majority consensus terjadi apabila sekurang-kurangnya dua pertiga LLM
   mendukung keputusan yang sama."
        -> MAJORITY: at least ceil(2n/3) of the n CONFIGURED models. The
           denominator is all models, not just the ones whose output survived:
           a model that failed verification does not make the others unanimous.

  "Conflict terjadi apabila tidak terdapat mayoritas yang memadai, terdapat
   perbedaan label yang tidak dapat diselesaikan, atau terdapat konflik batas
   span yang substansial."
        -> CONFLICT, and BOUNDARY_DISAGREEMENT when the label reaches the
           two-thirds bar but no single pair of boundaries does.

  "Hasil unanimous consensus yang lolos verifikasi deterministik dapat diterima
   secara otomatis ... Hasil majority consensus dapat diterima sebagai label
   otomatis hanya apabila aturan mayoritas tersebut sebelumnya menunjukkan
   tingkat ketepatan yang memadai pada development gold standard."
        -> UNANIMOUS is ACCEPTED. MAJORITY is ACCEPTED only when
           `accept_majority` is switched on, and that switch defaults to OFF:
           it may be turned on only after the majority rule's precision has been
           measured on the development gold.

  "Kasus CONFLICT, boundary disagreement, keluaran yang hanya didukung satu
   model, dan keluaran yang tidak lolos pemeriksaan bukti tidak akan dipaksakan
   untuk memperoleh label."
        -> everything else ABSTAINs, with the reason recorded. Nothing is forced.

Decisions the proposal leaves open, made here (see labeling/README.md):

  * A model casts NO_VOTE — neither a tag nor NONE — on candidates it has no
    valid span for, if its output was invalid or any of its spans was rejected.
    This can only lower agreement; it cannot change which spans are accepted.
  * `consensus_ratio` and `disagreement_score` are named by the proposal but not
    given formulas. The definitions used are in `_vote_stats`.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Sequence

from . import LABELING_VERSION, NO_VOTE, NONE
from .align import IOU_UNITS, Candidate, align
from .schema import ALLOWED_LABELS, INVALID, VALID, Rejection, VerifiedOutput
from .verify import VerifyPolicy

# candidate statuses
UNANIMOUS = "UNANIMOUS"
MAJORITY = "MAJORITY"
BOUNDARY_DISAGREEMENT = "BOUNDARY_DISAGREEMENT"
NONE_MAJORITY = "NONE_MAJORITY"
CONFLICT = "CONFLICT"
STATUSES = (UNANIMOUS, MAJORITY, BOUNDARY_DISAGREEMENT, NONE_MAJORITY, CONFLICT)

# dispositions
ACCEPTED = "ACCEPTED"
ABSTAIN = "ABSTAIN"

MIN_MODELS = 3          # "sekurang-kurangnya tiga LLM"
_N_OUTCOMES = len(ALLOWED_LABELS) + 2      # five tags + NONE + NO_VOTE


@dataclass(frozen=True)
class ConsensusConfig:
    """The frozen ensemble configuration. Everything that can change a decision
    lives here and is written into every result, because the held-out gold test
    may be opened only "setelah konfigurasi pelabelan multi-LLM dibekukan".

    `iou_threshold` 0.5 / `iou_unit` "char" are PROVISIONAL: they equal the rule
    behind the relaxed human-agreement figure, and the proposal requires the
    threshold to be set on the development gold, which does not exist yet.
    """
    models: tuple[str, ...]
    iou_threshold: float = 0.5
    iou_unit: str = "char"
    accept_majority: bool = False
    verify_policy: VerifyPolicy = field(default_factory=VerifyPolicy)

    def __post_init__(self):
        object.__setattr__(self, "models", tuple(self.models))
        if len(self.models) < MIN_MODELS:
            raise ValueError(
                f"the proposal requires at least {MIN_MODELS} LLMs; got {len(self.models)}")
        if len(set(self.models)) != len(self.models):
            raise ValueError(f"duplicate model ids: {self.models}")
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in (0, 1], got {self.iou_threshold}")
        if self.iou_unit not in IOU_UNITS:
            raise ValueError(f"iou_unit must be one of {IOU_UNITS}")

    @property
    def n_models(self) -> int:
        return len(self.models)

    @property
    def majority_votes(self) -> int:
        """ceil(2n/3): the fewest models that make "at least two-thirds"."""
        return -(-2 * self.n_models // 3)

    def to_dict(self) -> dict:
        return {"labeling_version": LABELING_VERSION, "models": list(self.models),
                "n_models": self.n_models, "majority_votes": self.majority_votes,
                "iou_threshold": self.iou_threshold, "iou_unit": self.iou_unit,
                "accept_majority": self.accept_majority,
                "verify_policy": self.verify_policy.to_dict()}


@dataclass
class CandidateDecision:
    status: str
    disposition: str
    reason: str
    label: str | None                 # the agreed tag, when there is one
    start: int | None                 # the agreed boundaries, when there are any
    end: int | None
    text: str | None                  # == sentence[start:end]
    votes: dict[str, str]             # model -> tag | NONE | NO_VOTE
    vote_counts: dict[str, int]
    consensus_ratio: float
    disagreement_score: float
    boundary: str                     # exact_all | exact_majority | relaxed | n/a
    min_pairwise_iou: float
    variants: list[dict]              # every member span, for the reviewer
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SentenceConsensus:
    sentence_id: str
    sentence_status: str              # OK | ABSTAIN_INSUFFICIENT_VALID_OUTPUTS
    empty_consensus: str | None       # set only when no model proposed any span
    n_valid_outputs: int
    candidates: list[CandidateDecision]
    model_status: dict[str, dict]
    config: dict

    @property
    def accepted(self) -> list[CandidateDecision]:
        return [c for c in self.candidates if c.disposition == ACCEPTED]

    def to_dict(self) -> dict:
        return {"sentence_id": self.sentence_id, "sentence_status": self.sentence_status,
                "empty_consensus": self.empty_consensus,
                "n_valid_outputs": self.n_valid_outputs,
                "candidates": [c.to_dict() for c in self.candidates],
                "model_status": self.model_status, "config": self.config}


def _vote_stats(votes: dict[str, str], n: int) -> tuple[dict[str, int], float, float]:
    """(counts, consensus_ratio, disagreement_score).

    consensus_ratio    share of ALL n models behind the single most-voted
                       decision among the five tags and NONE. NO_VOTE is not a
                       decision and cannot be the consensus.
    disagreement_score Shannon entropy of the n votes over the seven outcomes
                       (five tags, NONE, NO_VOTE), divided by its maximum
                       ln(min(n, 7)). 0 when every model did the same thing,
                       1 when no two did. NO_VOTE is counted as an outcome here
                       because a model that failed verification is itself a
                       sign the sentence is hard.
    """
    counts = Counter(votes.values())
    decided = {k: v for k, v in counts.items() if k != NO_VOTE}
    ratio = (max(decided.values()) / n) if decided else 0.0
    ent = -sum((c / n) * math.log(c / n) for c in counts.values() if c)
    cap = math.log(min(n, _N_OUTCOMES))
    return dict(sorted(counts.items())), ratio, (ent / cap if cap > 0 else 0.0)


def _decide_candidate(cand: Candidate, sentence: str,
                      by_model: dict[str, VerifiedOutput],
                      cfg: ConsensusConfig) -> CandidateDecision:
    n, need = cfg.n_models, cfg.majority_votes
    votes: dict[str, str] = {}
    for m in cfg.models:
        if m in cand.members:
            votes[m] = cand.members[m].label
        elif by_model[m].can_vote_none:
            votes[m] = NONE
        else:
            votes[m] = NO_VOTE
    counts, ratio, disagreement = _vote_stats(votes, n)

    label_counts = Counter(s.label for s in cand.members.values())
    top_label, top_votes = max(sorted(label_counts.items()), key=lambda kv: kv[1])
    supporters = {m: s for m, s in cand.members.items() if s.label == top_label}
    bound_counts = Counter(s.bounds for s in supporters.values())
    top_bounds, top_bound_votes = max(sorted(bound_counts.items()), key=lambda kv: kv[1])

    variants = [{"model": m, **cand.members[m].to_dict()} for m in cfg.models
                if m in cand.members]
    flags = sorted({f for s in cand.members.values() for f in s.flags})
    common = dict(votes=votes, vote_counts=counts, consensus_ratio=ratio,
                  disagreement_score=disagreement,
                  min_pairwise_iou=cand.min_pairwise_iou(sentence, cfg.iou_unit),
                  variants=variants, flags=flags)

    def agreed(status, disposition, reason, boundary):
        s, e = top_bounds
        return CandidateDecision(status, disposition, reason, top_label, s, e,
                                 sentence[s:e], boundary=boundary, **common)

    def undecided(status, reason, label=None):
        return CandidateDecision(status, ABSTAIN, reason, label, None, None, None,
                                 boundary="n/a" if label is None else "relaxed", **common)

    if top_votes == n and len(bound_counts) == 1:
        return agreed(UNANIMOUS, ACCEPTED, "unanimous_and_verified", "exact_all")

    if top_votes >= need:
        if top_bound_votes >= need:
            boundary = "exact_all" if len(bound_counts) == 1 else "exact_majority"
            if cfg.accept_majority:
                return agreed(MAJORITY, ACCEPTED, "majority_rule_validated_on_dev", boundary)
            return agreed(MAJORITY, ABSTAIN, "majority_rule_not_yet_validated_on_dev",
                          boundary)
        return undecided(BOUNDARY_DISAGREEMENT, "label_agreed_boundaries_not", top_label)

    if counts.get(NONE, 0) >= need:
        return undecided(NONE_MAJORITY, "supported_by_a_minority_of_models")

    if len(cand.members) >= need:
        return undecided(CONFLICT, "label_split")
    if n - counts.get(NO_VOTE, 0) < need:
        return undecided(CONFLICT, "too_few_valid_votes")
    return undecided(CONFLICT, "no_two_thirds_majority")


def decide_sentence(sentence_id: str, sentence: str,
                    outputs: Sequence[VerifiedOutput],
                    cfg: ConsensusConfig) -> SentenceConsensus:
    """Run alignment and the vote for one sentence.

    Every configured model must be accounted for. One that returned nothing is
    recorded as an INVALID output (MISSING_OUTPUT) and casts no vote — it is
    never silently dropped from the denominator.
    """
    by_model: dict[str, VerifiedOutput] = {}
    for o in outputs:
        if o.model not in cfg.models:
            raise ValueError(f"output from {o.model!r}, which is not in the configured "
                             f"ensemble {cfg.models}")
        if o.model in by_model:
            raise ValueError(f"two outputs for model {o.model!r} on {sentence_id!r}")
        if o.sentence_id != sentence_id:
            raise ValueError(f"output for sentence {o.sentence_id!r} passed to the "
                             f"consensus for {sentence_id!r}")
        by_model[o.model] = o
    for m in cfg.models:
        if m not in by_model:
            by_model[m] = VerifiedOutput(
                model=m, sentence_id=sentence_id, status=INVALID,
                rejections=[Rejection("MISSING_OUTPUT", INVALID, "output",
                                      "the model returned no output for this sentence")])

    ordered = [by_model[m] for m in cfg.models]
    cands = align(ordered, sentence, cfg.iou_threshold, cfg.iou_unit)
    decisions = [_decide_candidate(c, sentence, by_model, cfg) for c in cands]

    n_valid = sum(1 for o in ordered if o.status == VALID)
    status = "OK" if n_valid >= cfg.majority_votes else "ABSTAIN_INSUFFICIENT_VALID_OUTPUTS"

    empty = None
    if not cands:
        clean = sum(1 for o in ordered if o.can_vote_none)
        if clean == cfg.n_models:
            empty = "UNANIMOUS_NONE"
        elif clean >= cfg.majority_votes:
            empty = "MAJORITY_NONE"
        else:
            empty = "UNDETERMINED"

    model_status = {
        m: {"status": by_model[m].status, "n_spans": len(by_model[m].spans),
            "n_rejections": len(by_model[m].rejections),
            "can_vote_none": by_model[m].can_vote_none,
            "rejection_codes": sorted({r.code for r in by_model[m].rejections})}
        for m in cfg.models}
    return SentenceConsensus(sentence_id, status, empty, n_valid, decisions,
                             model_status, cfg.to_dict())


def accepted_spans_by_sid(results: Sequence[SentenceConsensus]) -> dict[str, list]:
    """Accepted spans as `gold_standard.metrics.Span`s, keyed by sentence id.

    This is the bridge to scoring: `metrics.micro(metrics.contribs(gold, this,
    sids))` gives the automatic-acceptance precision the proposal asks for, with
    the same matcher the gold standard itself is measured by. Every sentence is
    present, including those with nothing accepted, so the scoring population is
    not narrowed to the sentences where the ensemble spoke.
    """
    from gold_standard.metrics import Span
    return {r.sentence_id: [Span(c.start, c.end, c.label) for c in r.accepted]
            for r in results}


def summarise(results: Sequence[SentenceConsensus]) -> dict:
    """Corpus-level rates named in the proposal, each with a Wilson interval."""
    from gold_standard.metrics import wilson_ci

    def rate(k: int, n: int) -> dict:
        lo, hi = wilson_ci(k, n)
        return {"k": k, "n": n, "rate": (k / n) if n else float("nan"),
                "wilson95": [lo, hi]}

    cands = [c for r in results for c in r.candidates]
    n = len(cands)
    by_status = Counter(c.status for c in cands)
    by_reason = Counter(c.reason for c in cands if c.disposition == ABSTAIN)
    n_sent = len(results)
    return {
        "n_sentences": n_sent,
        "n_candidates": n,
        "status_rates": {s: rate(by_status.get(s, 0), n) for s in STATUSES},
        "acceptance_rate": rate(sum(1 for c in cands if c.disposition == ACCEPTED), n),
        "abstention_rate": rate(sum(1 for c in cands if c.disposition == ABSTAIN), n),
        "abstain_reasons": dict(sorted(by_reason.items())),
        "mean_consensus_ratio": (sum(c.consensus_ratio for c in cands) / n) if n
        else float("nan"),
        "mean_disagreement_score": (sum(c.disagreement_score for c in cands) / n) if n
        else float("nan"),
        "sentences_abstained_for_too_few_valid_outputs": rate(
            sum(1 for r in results if r.sentence_status != "OK"), n_sent),
        "sentences_with_no_candidate": dict(sorted(Counter(
            r.empty_consensus for r in results if r.empty_consensus).items())),
    }
