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

What a competency is — decided by the researcher on 2026-11-20:

    a VERB plus a KNOWLEDGE. "Develop machine learning algorithms": the verb
    "develop" is placed on the taxonomy by the verb table, "machine learning
    algorithms" is the knowledge it acts on. The statement is those two and
    nothing else: it names no tool ("Perform programming", not "Perform
    programming using Python, PHP and JavaScript"). For each competency the
    most demanded tools and technologies, soft skills and work behaviours are
    LISTED beside it.                                    competency.structure

How the five tags are used — decided by the researcher on 2026-11-19:

  * SKILL_HARD is what competencies are built from. A competency is a group of
    hard-skill phrases that mean the same work. Its verbs and its knowledge are
    read from inside those phrases.
  * Everything else DESCRIBES a competency:
      TOOL_TECHNOLOGY  the tools mostly used for it: those named in the same
                       sentence as its phrases. (The proposal: a technology is
                       "bukti teknologi yang masih harus dikaitkan dengan
                       tindakan kerja".)
      KNOWLEDGE        (a term tagged KNOWLEDGE because it stands alone in a
                       posting) other knowledge asked for in the same postings.
                       This list is the developer's reading. The knowledge a
                       competency IS about is the object of its verb, above.
      SKILL_SOFT and   what is expected of the person who has it: those asked
      WORK_BEHAVIOR    for in the same postings, each with its share of the
                       competency's postings and its lift over postings in
                       general.
  This is also the shape of an Indonesian competency standard (SKKNI): the unit
  of competency, then the knowledge, the work attitude and the equipment needed.

  An earlier version of this package (same day) built competencies from
  hard-skill AND knowledge phrases and grouped soft skills as competencies of
  their own. Runs made with it are marked in the experiment log.

Reused, not rewritten: the clusterer (`clustering/`), the cluster validity
audit (`pipeline_scientific_audits`), the KKNI labeller and the okupasi mapper.
What is new is the typed evidence model in front of them and the grounded
statement and contract behind them.

Also here: `termmap` (a map of knowledge terms and technologies grouped by what
is asked for together, shown in the public dashboard), `temporal`, `coverage`,
`delta`, `profile`, `endtoend`, `figures`. Not built: the education-level
distribution.
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
