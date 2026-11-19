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


def map_kkni(competencies: Sequence[dict]) -> dict:
    """Adds kkni_level / kkni_level_source / kkni_match_similarity where the
    labeller is confident; leaves them None otherwise."""
    from competency_kkni_labeller import label_competencies_kkni
    shells = [SimpleNamespace(statement=c["statement"]["text"], kkni_level=None,
                              kkni_level_source=None, kkni_match_similarity=None)
              for c in competencies]
    report = label_competencies_kkni(shells)
    for c, s in zip(competencies, shells):
        c["kkni"] = {"level": s.kkni_level, "source": s.kkni_level_source,
                     "similarity": s.kkni_match_similarity}
    return report if isinstance(report, dict) else {}


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
