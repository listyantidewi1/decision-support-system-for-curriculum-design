"""The recommendation contract: which status a competency may carry, and why.

Proposal §3.3.1: every competency gets "status kelayakan (eksploratif, kandidat
tinjauan, atau 'siap ditelaah formal') sesuai syarat keabsahan yang dipenuhi".

  eksploratif            something blocks it: the group is not valid, too few
                         postings, the statement is not grounded, no okupasi.
  kandidat tinjauan      nothing in the evidence blocks it. It is worth an
                         expert's time.
  siap ditelaah formal   as above AND the corpus is valid for the Indonesian
                         claim AND experts reviewed it AND it is stable across
                         runs. The pipeline can never reach this status by
                         itself: the last three are facts about the study, set
                         by a person, and they default to "not yet".

What does NOT enter the status: the KKNI level. The researcher's instruction
(2026-11-19): KKNI is additional information, there to tell the reader at which
level a competency sits. It never blocks, raises or orders a competency. A test
holds this.

Which grounding check applies depends on who wrote the statement. A rule
statement is short and made of the group's own words, so the share of its words
found in the evidence is the check (`min_grounding`). A statement from the
earlier LLM generator abstracts on purpose and scores low on that measure; for
it the check is the generator's own evidence filter (a product or method it
names must be named by the sources), passed as `grounding_passed`. Which check
was used is written into the decision.

Every decision lists its blockers, so the status can be explained and argued
with. The two thresholds are STARTING VALUES carried over from the pre-approval
pipeline; they are to be fixed with the promotor and written into the
configuration registry before any profile is released.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import STATUS_CANDIDATE, STATUS_EXPLORATORY, STATUS_READY, VALID


@dataclass(frozen=True)
class ContractConfig:
    min_postings: int = 5
    min_grounding: float = 0.80
    thresholds_status: str = "starting values, not yet agreed"
    version: str = "contract-v1"


@dataclass(frozen=True)
class StudyFacts:
    """Set by a person, never inferred from the data."""
    corpus_valid_for_indonesian_claim: bool = False
    expert_review_passed: bool = False
    stability_validated: bool = False


def decide(*, group_status: str, n_postings: int, grounding: float,
           okupasi_mapped: bool | None, cfg: ContractConfig | None = None,
           facts: StudyFacts | None = None, grounding_passed: bool | None = None,
           fragment: bool = False) -> dict:
    """`okupasi_mapped` is None when the mapping step was not run; that blocks
    too, because an unmapped competency was never shown to map anywhere.
    `grounding_passed`, when given, is the verdict of the statement writer's own
    evidence filter and replaces the lexical threshold."""
    cfg, facts = cfg or ContractConfig(), facts or StudyFacts()
    core = []
    if group_status != VALID:
        core.append(f"group_status={group_status or 'unknown'}")
    if n_postings < cfg.min_postings:
        core.append(f"postings<{cfg.min_postings}")
    if fragment:
        # the phrases were cut before their object: they name no work (EXP-030)
        core.append("statement_is_a_fragment")
    if grounding_passed is None:
        check = f"lexical>={cfg.min_grounding:.2f}"
        if grounding < cfg.min_grounding:
            core.append(f"grounding<{cfg.min_grounding:.2f}")
    else:
        check = "generator_evidence_filter"
        if not grounding_passed:
            core.append("statement_evidence_filter_failed")
    if okupasi_mapped is None:
        core.append("okupasi_mapping_not_run")
    elif not okupasi_mapped:
        core.append("okupasi_unmapped")
    readiness = list(core)
    if not facts.corpus_valid_for_indonesian_claim:
        readiness.append("corpus_not_yet_valid_for_indonesian_claim")
    if not facts.expert_review_passed:
        readiness.append("expert_review_not_completed")
    if not facts.stability_validated:
        readiness.append("stability_not_validated")
    if core:
        status = STATUS_EXPLORATORY
    elif readiness:
        status = STATUS_CANDIDATE
    else:
        status = STATUS_READY
    return {"status": status, "blockers": core, "grounding_check": check,
            "blockers_to_formal_review": readiness, "contract_version": cfg.version}
