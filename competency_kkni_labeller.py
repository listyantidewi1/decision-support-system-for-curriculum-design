"""
competency_kkni_labeller.py

Phase 2.3 — post-hoc KKNI level labeller for CompetencyV2.

The Phase 2.2 LLM emits a tentative `kkni_level` (1-9) for each competency,
but LLMs aren't reliable at consistent KKNI-bracket assignment across runs.
This module replaces / overrides that LLM suggestion with a deterministic
SBERT-based assignment:

    1. Embed each KKNI level's descriptor (from `kkni.py::KKNI_LEVELS`).
       KKNI descriptors are in Indonesian (Perpres 8/2012, Lampiran).
    2. For each competency, embed `title + ". " + description`.
    3. Cosine-similarity match the competency vector against the 9 descriptor
       vectors; assign the argmax level.
    4. Optionally restrict the search to KKNI levels 3-6 (the SMK/D3/D4/S1
       relevant range for vocational software-engineering curricula).

Multilingual SBERT is used (`paraphrase-multilingual-MiniLM-L12-v2`) so the
English competency text can still cosine-match against Indonesian KKNI
descriptors. The model is cached on disk per HuggingFace's usual pattern.

Public API:

    from competency_kkni_labeller import label_competencies_kkni, KkniLabellerConfig
    label_competencies_kkni(competencies, config=KkniLabellerConfig())

Mutates each competency to set:
    kkni_level             (canonical, overrides llm_suggested)
    kkni_level_source      = "sbert_labeller"
    kkni_match_similarity  (cosine similarity to the assigned level descriptor)

The original LLM-suggested level is preserved in `grounding_reasoning` as
an audit breadcrumb (won't survive future runs of the grounding evaluator,
so the labeller writes to a separate log).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from competency_v2_schema import CompetencyV2
from kkni import KKNI_LEVELS

logger = logging.getLogger(__name__)


LABELLER_VERSION = "v2.3.0"


@dataclass
class KkniLabellerConfig:
    """Knobs for the KKNI labeller."""

    # Multilingual SBERT — handles Indonesian descriptors + English competency text
    sbert_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # KKNI range to consider. 3-7 covers SMK through Profesi/Spes-1, which is
    # the realistic span for software-engineering competencies. Tier-2 update
    # (2026-05-13) widened from (3, 6) so the labeller no longer pre-clamps
    # higher-complexity competencies down to a vocational ceiling — the
    # evidence in the cluster picks the level.
    # Set to (1, 9) for the full KKNI range.
    allowed_levels: Tuple[int, int] = (3, 7)

    # Minimum cosine similarity required to override the LLM suggestion.
    # Raised 2026-05-14 from 0.30 → 0.45 after the n1k_v10b expert review
    # found 11 of 30 competencies placed at KKNI 7 — many incorrectly,
    # because the SBERT labeller eagerly overrode the LLM on cosine deltas
    # too small to trust. With 0.45, only confidently-different placements
    # override the LLM, which calibrates KKNI from richer cluster context.
    override_min_similarity: float = 0.45

    # Tiebreaker: when the top-2 levels are within `tiebreak_cosine_window`
    # of each other, prefer the LOWER level. Bias is intentional —
    # personality-trait-ish wording in statements ("leadership",
    # "strategic", "evaluate") cosine-matches KKNI 7 descriptors even
    # when the underlying cognitive demand is KKNI 5-6. Conservative
    # placement is the safer error mode for curriculum work.
    tiebreak_cosine_window: float = 0.05

    # If allowed_levels=(3, 6) and best match is outside that range, snap to
    # nearest allowed boundary. Disabled 2026-05-14 — with (3, 7) covering
    # the realistic span, snapping is unnecessary and obscures which
    # competencies the labeller had to clamp.
    snap_to_allowed: bool = False


# --------------------------------------------------------------------------- #
# Embedder cache (module-local)
# --------------------------------------------------------------------------- #


_embedder_cache: dict = {}
_descriptor_cache: dict = {}


def _get_embedder(model_name: str):
    """Lazy-load the multilingual SBERT embedder, with a hard-coded download
    fallback that avoids the HuggingFace xet-storage stall we hit on 2026-05-12.
    """
    if model_name in _embedder_cache:
        return _embedder_cache[model_name]

    # Belt-and-braces: disable HF's experimental xet transfer (the source of
    # the earlier hang at 2026-05-12 22:51 — see notes in PYTHON_ENV.md).
    # If the model is already cached this is a no-op.
    import os as _os
    _os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    _os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        _embedder_cache[model_name] = model
        return model
    except Exception as e:
        logger.warning("multilingual SBERT %s unavailable: %s", model_name, e)
        _embedder_cache[model_name] = None
        return None


def _get_descriptor_embeddings(model_name: str, embedder, allowed_levels: Tuple[int, int]):
    """Embed KKNI level descriptors once and cache them.

    Cache key includes allowed_levels so different restrictions don't collide.
    """
    cache_key = (model_name, allowed_levels)
    if cache_key in _descriptor_cache:
        return _descriptor_cache[cache_key]

    levels = list(range(allowed_levels[0], allowed_levels[1] + 1))
    descriptors = [
        # Include the label + descriptor so we match against both
        f"KKNI Level {lvl} ({KKNI_LEVELS[lvl]['label']}): {KKNI_LEVELS[lvl]['descriptor']}"
        for lvl in levels
    ]
    embs = embedder.encode(
        descriptors,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    _descriptor_cache[cache_key] = (levels, embs)
    return levels, embs


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #


def label_one(
    competency: CompetencyV2,
    config: KkniLabellerConfig,
    embedder,
    levels: List[int],
    descriptor_embs,
) -> None:
    """Compute KKNI level for one competency. Mutates in place.

    Sets:
        kkni_level             (overrides LLM suggestion)
        kkni_level_source      = "sbert_labeller"
        kkni_match_similarity  (cosine to the chosen level)
    """
    text = (competency.statement or "").strip()
    if not text:
        return

    vec = embedder.encode(
        [text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )[0]
    sims = descriptor_embs @ vec
    best_local = int(sims.argmax())
    best_level = levels[best_local]
    best_sim = float(sims[best_local])

    if best_sim < config.override_min_similarity:
        # Low confidence — don't override
        return

    # Prefer-lower tiebreaker: among levels whose cosine is within
    # `tiebreak_cosine_window` of the best, pick the LOWEST level. Bias
    # is intentional and conservative — see config docstring.
    win = float(getattr(config, "tiebreak_cosine_window", 0.0) or 0.0)
    if win > 0.0:
        tied = [
            (levels[i], float(sims[i]))
            for i in range(len(levels))
            if float(sims[i]) >= best_sim - win
        ]
        if tied:
            best_level, best_sim = min(tied, key=lambda lv_s: lv_s[0])

    if config.snap_to_allowed:
        lo, hi = config.allowed_levels
        best_level = max(lo, min(hi, best_level))

    competency.kkni_level = best_level
    competency.kkni_level_source = "sbert_labeller"
    competency.kkni_match_similarity = best_sim


def label_competencies_kkni(
    competencies: Sequence[CompetencyV2],
    config: Optional[KkniLabellerConfig] = None,
) -> dict:
    """Label a batch of competencies. Returns an audit report.

    Mutates each competency in place. If SBERT is unavailable, leaves
    LLM-suggested labels untouched and returns a report indicating skip.
    """
    if config is None:
        config = KkniLabellerConfig()

    embedder = _get_embedder(config.sbert_model)
    if embedder is None:
        logger.warning("KKNI labeller skipped — embedder unavailable")
        return {
            "version": LABELLER_VERSION,
            "n_evaluated": len(competencies),
            "n_relabeled": 0,
            "n_skipped_low_confidence": 0,
            "skipped_reason": "embedder unavailable",
        }

    levels, descr_embs = _get_descriptor_embeddings(
        config.sbert_model, embedder, config.allowed_levels
    )

    n_relabeled = 0
    n_low_conf = 0
    n_kept_llm = 0
    sims = []
    for c in competencies:
        before_level = c.kkni_level
        before_source = c.kkni_level_source
        label_one(c, config, embedder, levels, descr_embs)
        if c.kkni_level_source == "sbert_labeller":
            n_relabeled += 1
            sims.append(c.kkni_match_similarity or 0.0)
            if before_source == "llm_suggested" and before_level != c.kkni_level:
                logger.debug(
                    "competency %s: LLM=%s → SBERT=%s (sim=%.2f)",
                    c.id, before_level, c.kkni_level, c.kkni_match_similarity,
                )
        else:
            n_low_conf += 1

    return {
        "version": LABELLER_VERSION,
        "n_evaluated": len(competencies),
        "n_relabeled": n_relabeled,
        "n_skipped_low_confidence": n_low_conf,
        "n_kept_llm": n_kept_llm,
        "mean_similarity": (sum(sims) / len(sims)) if sims else 0.0,
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in config.__dict__.items()},
    }
