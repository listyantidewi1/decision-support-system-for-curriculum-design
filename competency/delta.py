"""Delta recommendations: what to do to a curriculum document (proposal §3.3.6).

When a curriculum has been uploaded and its coverage audited, each competency
is stated as one of five proposed actions on THAT document:

  menambah                not covered, and the evidence is strong
  memperdalam             partly covered
  meninjau_ulang_jenjang  covered, but the KKNI level asked for by the market
                          differs from the level the curriculum is written for
  mempertahankan          clearly covered and still in demand
  mengurangi_penekanan    covered, and demand has fallen in every window

and, when none applies, no action:

  (pemantauan)            not covered but the evidence is only exploratory —
                          Tabel 3.8 limits such a signal to monitoring

Two safeguards from the proposal are built in, not left to the reader:

  * "mengurangi penekanan" never exceeds the status eksploratif without expert
    review: falling demand in postings does not mean a competency stopped
    mattering. The delta carries `status_cap = eksploratif`.
  * a delta is a PROPOSAL. Coverage it rests on is itself only proposed until an
    expert validates it; the delta says so.

"Demand has fallen in every window" is taken literally: the share is strictly
lower in each window than in the one before, over at least three windows. A
statistical trend test is reported beside it when the time analysis has one.

Ordering: among recommendations with the same status, the persistence weight
decides the order. It never changes a status.
"""
from __future__ import annotations

from typing import Sequence

from . import STATUS_CANDIDATE, STATUS_EXPLORATORY, STATUS_READY
from .coverage import COVERED, MISSING, PARTIAL, UPDATE_INPUT

ADD, DEEPEN, REVIEW_LEVEL, KEEP, REDUCE = ("menambah", "memperdalam", "meninjau_ulang_jenjang",
                                           "mempertahankan", "mengurangi_penekanan")
DELTA_TYPES = (ADD, DEEPEN, REVIEW_LEVEL, KEEP, REDUCE)
MONITOR = "pemantauan"
DELTA_VERSION = "delta-v1"
_RANK = {STATUS_EXPLORATORY: 0, STATUS_CANDIDATE: 1, STATUS_READY: 2}


def falls_in_every_window(shares: Sequence[float] | None, min_windows: int = 3) -> bool:
    return bool(shares) and len(shares) >= min_windows and all(
        b < a for a, b in zip(shares, shares[1:]))


def decide(c: dict, reference_name: str, curriculum_kkni_level: int | None = None) -> dict:
    """The delta for one competency against one audited curriculum."""
    cov = (c.get("coverage") or {}).get(reference_name)
    if cov is None:
        raise KeyError(f"coverage against {reference_name!r} has not been audited")
    status = c.get("contract", {}).get("status", STATUS_EXPLORATORY)
    temporal = c.get("temporal") or {}
    shares = temporal.get("shares")
    falling = falls_in_every_window(shares)
    kkni = c.get("kkni") or {}
    level_differs = (curriculum_kkni_level is not None and kkni.get("confident")
                     and kkni.get("level") is not None
                     and int(kkni["level"]) != int(curriculum_kkni_level))
    proposed, basis, cap = cov["proposed"], [], None
    if proposed in (MISSING, UPDATE_INPUT):
        if _RANK.get(status, 0) >= 1:
            delta = ADD
            basis.append(f"not covered (best similarity {cov['best_similarity']}); "
                         f"evidence status {status}")
        else:
            delta = MONITOR
            basis.append("not covered, but the evidence is exploratory: monitoring only")
    elif proposed == PARTIAL:
        delta = DEEPEN
        basis.append(f"partly covered (best similarity {cov['best_similarity']})")
    elif falling:
        delta, cap = REDUCE, STATUS_EXPLORATORY
        basis.append(f"covered; share fell in every one of {len(shares)} windows: {shares}")
        basis.append("capped at eksploratif until an expert reviews it")
    elif level_differs:
        delta = REVIEW_LEVEL
        basis.append(f"covered; KKNI level proposed from the postings is {kkni['level']}, the "
                     f"curriculum is written for level {curriculum_kkni_level}")
    else:
        delta = KEEP
        basis.append(f"clearly covered (best similarity {cov['best_similarity']})"
                     + ("" if shares else "; demand over time not analysed"))
    if curriculum_kkni_level is None and proposed == COVERED and delta == KEEP:
        basis.append("level not compared: no curriculum KKNI level was given")
    effective = status if cap is None or _RANK.get(status, 0) <= _RANK[cap] else cap
    return {"type": delta, "reference": reference_name, "basis": basis, "status_cap": cap,
            "effective_status": effective, "delta_version": DELTA_VERSION,
            "coverage_validated_by_expert": bool(cov.get("validated_by_expert")),
            "trend_test": temporal.get("trend")}


def apply(competencies: Sequence[dict], reference_name: str,
          curriculum_kkni_level: int | None = None) -> dict:
    counts = {t: 0 for t in (*DELTA_TYPES, MONITOR)}
    for c in competencies:
        c["delta"] = decide(c, reference_name, curriculum_kkni_level)
        counts[c["delta"]["type"]] += 1
    return {"reference": reference_name, "delta_version": DELTA_VERSION,
            "curriculum_kkni_level": curriculum_kkni_level, "counts": counts}


def priority_order(competencies: Sequence[dict]) -> list[dict]:
    """Status first; inside a status, the persistence weight; then postings.
    The weight orders, it never promotes."""
    def key(c):
        status = (c.get("delta") or {}).get("effective_status") or c["contract"]["status"]
        weight = (c.get("temporal") or {}).get("persistence_weight")
        return (-_RANK.get(status, 0), -(weight if weight is not None else -1.0),
                -c.get("n_postings", 0), c["competency_id"])
    return sorted(competencies, key=key)
