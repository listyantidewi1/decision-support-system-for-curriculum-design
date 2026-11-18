"""Multi-LLM ensemble labelling — Workstream C of docs/IMPLEMENTATION_PLAN.md.

Implements proposal §3.4.3 (Gambar 3.8): at least three LLMs label the same
sentence independently; every output passes a deterministic verifier; the
surviving spans are aligned across models; a consensus rule decides what may be
accepted automatically and what must abstain.

This package currently holds the DETERMINISTIC half — no model is called here:

    schema.py     the output contract and the label vocabulary
    verify.py     JSON / schema / span-exists-in-source checks; INVALID and
                  UNSUPPORTED log
    align.py      exact, then IoU alignment of spans across models
    consensus.py  the six-way vote (five tags + NONE) and the acceptance rule

The prompt pack, the model runner and the silver store come next and depend on
this half, not the other way round. See labeling/README.md for the design
decisions, including the ones the proposal leaves open and the one place the
plan and the validated guideline disagree.
"""
from __future__ import annotations

LABELING_VERSION = "0.1.0"

# A model's decision on one candidate span is one of the five tags, NONE, or —
# when its output for the sentence could not be trusted — no vote at all.
NONE = "NONE"
NO_VOTE = "NO_VOTE"
