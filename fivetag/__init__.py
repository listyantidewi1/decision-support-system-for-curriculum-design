"""Five-tag extraction and typing models (proposal §3.3.1–3.3.3; plan Workstream D).

One package for every condition the approved proposal compares:

    E0  rule / dictionary extraction            fivetag.e0_rules
    E1  multilingual encoder + BIO              fivetag.encoder
    E2  E1 after domain-adaptive pretraining    fivetag.encoder (same code, other checkpoint)
    E3  GLiNER open-type extraction             fivetag.gliner_adapter
    E4  prompted LLM, JSON schema, no verifier  fivetag.llm
    E5  LLM + retrieval + deterministic verifier fivetag.llm
    E6  NLP extraction + conditional LLM router fivetag.router

    D1  lexicon-regex-priority classifier       fivetag.typing_d
    D2  taxonomy-anchored context decision      fivetag.typing_d
    D3  deterministic consensus (CONFLICT)      fivetag.typing_d

Every extractor returns `fivetag.base.Pred` objects with character offsets into
the sentence it was given, so all of them are scored by the one shared engine
(`gold_standard.metrics`) and can be compared pairwise.

Nothing here has been trained or selected on the gold standard: at the time of
writing (2026-11-19) no gold standard exists. Numbers produced before it does
come from development instruments (`labeling.conformance`,
`labeling.skillspan_guideline`, `fivetag.synthetic`) and are engineering checks,
not results.
"""
from __future__ import annotations

from gold_standard import TAGS

FIVETAG_VERSION = "0.1.0"
TAG_CODES: tuple[str, ...] = tuple(TAGS)
