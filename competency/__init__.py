"""Competency layers on five-tag evidence (proposal §3.3.1, plan Workstream E).

From spans to competencies, in the order the proposal gives:

    evidence   every extracted span becomes one typed evidence unit that keeps
               its sentence and posting                      competency.evidence
    grouping   evidence is grouped by meaning; every group gets one of the six
               validity statuses                             competency.grouping
    statement  one assessable statement per valid group, opening with an
               operational verb, checked against its evidence
                                                             competency.statement
    mapping    KKNI level and okupasi                        competency.mapping
    contract   eksploratif / kandidat tinjauan / siap ditelaah formal, with
               the reasons                                   competency.contract
    pipeline   the whole chain, writing an evidence package  competency.pipeline

How the five tags are used — the design decision of this package:

  * SKILL_HARD and KNOWLEDGE are what competencies are built from.
  * SKILL_SOFT is grouped separately, as supporting competencies.
  * TOOL_TECHNOLOGY never forms a competency by itself. The proposal says a
    technology name is "bukti teknologi yang masih harus dikaitkan dengan
    tindakan kerja": it is attached to the groups whose evidence it shares a
    sentence with, and appears in the statement as context.
  * WORK_BEHAVIOR is reported as a profile of expected dispositions, not turned
    into competencies ("sifat pribadi" is kept apart).

Reused, not rewritten: the clusterer (`clustering/`), the cluster validity
audit (`pipeline_scientific_audits`), the KKNI labeller and the okupasi mapper.
What is new is the typed evidence model in front of them and the grounded
statement and contract behind them.

Not built here yet: the gap audit against official references, the rolling
time windows (persistence, volatility), the delta recommendations and the
versioned profile. The education-level distribution is also not built.
"""
from __future__ import annotations

COMPETENCY_VERSION = "0.1.0"

# The six group statuses of §3.3.1, keyed by the audit module's own codes.
GROUP_STATUS = {
    "valid_cluster": "sahih",
    "split_required": "perlu_dipecah",
    "mixed_domain": "bercampur_domain",
    "support_process_cluster": "penampung_proses_penunjang",
    "out_of_domain_cluster": "di_luar_bidang",
    "weak_low_frequency_theme": "berfrekuensi_rendah",
}
VALID = "sahih"

STATUS_EXPLORATORY = "eksploratif"
STATUS_CANDIDATE = "kandidat_tinjauan"
STATUS_READY = "siap_ditelaah_formal"
