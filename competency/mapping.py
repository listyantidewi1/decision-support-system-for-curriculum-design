"""KKNI level and okupasi for each competency — thin adapters over the existing
labeller and mapper, which both read the statement text.

Both load a sentence-embedding model, so they are imported only when called and
can be switched off (`pipeline --no-mapping`).

KKNI is a label of cognitive complexity. It informs the reader; it never
decides whether a competency is kept, and market demand is never fed into it.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence


KKNI_STEP_VERSION = "kkni-nearest-descriptor-v1"


def shared_embedder():
    """The multilingual sentence-embedding model, loaded once and shared by the
    KKNI step and the coverage audit."""
    from competency_kkni_labeller import KkniLabellerConfig, _get_embedder
    return _get_embedder(KkniLabellerConfig().sbert_model)


def map_kkni(competencies: Sequence[dict], embedder=None) -> dict:
    """Proposes a KKNI level for EVERY competency and says how sure it is.

    The existing labeller (`competency_kkni_labeller.label_competencies_kkni`)
    was built to correct a level an LLM had already suggested: it stays silent
    unless its similarity passes a threshold. On this chain nothing suggests a
    level first, so it assigned one to 17 of 445 statements (experiment log,
    EXP-017). This step keeps the labeller's own model, level descriptors,
    range and prefer-the-lower-level tie-break, but always reports the nearest
    level together with its similarity and a `confident` flag (the labeller's
    own threshold). A level that is not confident is a starting point for an
    expert, not a placement.

    The text compared with the KKNI descriptors is the statement followed by the
    group's most frequent phrases, so that a short statement is not judged on
    four words alone.
    """
    from competency_kkni_labeller import (KkniLabellerConfig, _get_descriptor_embeddings,
                                          _get_embedder)
    cfg = KkniLabellerConfig()
    embedder = embedder or _get_embedder(cfg.sbert_model)
    if embedder is None or not competencies:
        for c in competencies:
            c["kkni"] = {"level": None, "similarity": None, "confident": False,
                         "source": "not run: embedding model unavailable"}
        return {"version": KKNI_STEP_VERSION, "n_evaluated": len(competencies), "n_confident": 0,
                "skipped_reason": "embedder unavailable" if embedder is None else ""}
    levels, descriptors = _get_descriptor_embeddings(cfg.sbert_model, embedder,
                                                     cfg.allowed_levels)
    texts = [c["statement"]["text"] + ". " + "; ".join(
        m["text"] for m in c["evidence"]["members"][:5]) for c in competencies]
    vecs = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=False)
    sims = vecs @ descriptors.T
    window = float(getattr(cfg, "tiebreak_cosine_window", 0.0) or 0.0)
    n_conf, dist = 0, {}
    for c, row in zip(competencies, sims):
        best = float(row.max())
        tied = [(levels[i], float(row[i])) for i in range(len(levels)) if float(row[i]) >= best - window]
        level, sim = min(tied, key=lambda t: t[0])
        confident = sim >= cfg.override_min_similarity
        n_conf += confident
        dist[level] = dist.get(level, 0) + 1
        c["kkni"] = {"level": int(level), "similarity": round(sim, 4), "confident": bool(confident),
                     "source": KKNI_STEP_VERSION,
                     "similarity_by_level": {int(l): round(float(s), 4) for l, s in zip(levels, row)}}
    return {"version": KKNI_STEP_VERSION, "n_evaluated": len(competencies),
            "n_confident": int(n_conf), "confidence_threshold": cfg.override_min_similarity,
            "level_distribution": {int(k): v for k, v in sorted(dist.items())},
            "allowed_levels": list(cfg.allowed_levels)}


def map_okupasi(competencies: Sequence[dict]) -> dict:
    """Adds `okupasi`: the nearest occupations of the Indonesian occupation map."""
    from occupation_mapper import map_competencies
    shells = [{"statement": c["statement"]["text"],
               "related_skills": [m["text"] for m in c["evidence"]["members"][:10]]}
              for c in competencies]
    map_competencies(shells)
    for c, s in zip(competencies, shells):
        c["okupasi"] = [{k: m.get(k) for k in ("occupation_en", "occupation_id", "sector",
                                               "cosine", "is_official") if k in m}
                        for m in s.get("occupation_matches") or []]
    return {"n_mapped": sum(1 for c in competencies if c["okupasi"]),
            "n_unmapped": sum(1 for c in competencies if not c["okupasi"])}
