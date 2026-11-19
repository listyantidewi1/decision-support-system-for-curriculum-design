"""Coverage audit: is a competency already in the reference documents? (proposal §3.3.5)

Two kinds of reference are compared with the same method:

  official references   the SKKNI unit titles in `DATA/master_skkni/json/skkni.json`
  a curriculum document any text or PDF a school uploads (parsed by the existing
                        `curriculum_coverage.parse_curriculum_text` / `_pdf`)

For every competency the audit PROPOSES one of the four statuses of the
proposal and shows the reference units it rests on:

  tercakup_jelas      a reference unit says nearly the same thing
  tercakup_sebagian   a reference unit is related but does not say the same
  belum_tercakup      nothing close
  masukan_pembaruan   not covered, AND the competency's evidence is strong
                      enough to be worth an expert's time ("berpotensi menjadi
                      masukan pembaruan")

The proposal is explicit that this status "tidak ditentukan oleh model secara
sepihak, melainkan melalui evidence package dan validasi ahli". So every result
is marked `proposed` and `validated_by_expert: false`, and the nearest units
are listed with their similarity for the expert to judge.

Method: cosine similarity between sentence embeddings of the competency (its
statement and most frequent phrases) and of each reference unit, with a
multilingual model because statements and references mix Indonesian and
English. The two thresholds are the ones the pre-approval coverage module
used (0.65, 0.45). They are STARTING VALUES and have not been calibrated
against expert judgements.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import STATUS_CANDIDATE, STATUS_READY
from .grouping import MULTILINGUAL_EMBEDDER

REPO_ROOT = Path(__file__).resolve().parents[1]
SKKNI_JSON = REPO_ROOT / "DATA" / "master_skkni" / "json" / "skkni.json"

COVERED, PARTIAL, MISSING, UPDATE_INPUT = ("tercakup_jelas", "tercakup_sebagian",
                                           "belum_tercakup", "masukan_pembaruan")
COVERAGE_STATUSES = (COVERED, PARTIAL, MISSING, UPDATE_INPUT)


@dataclass(frozen=True)
class CoverageConfig:
    covered_at: float = 0.65
    partial_at: float = 0.45
    embedder_model: str = MULTILINGUAL_EMBEDDER
    top_matches: int = 3
    thresholds_status: str = "starting values, not calibrated against expert judgement"
    version: str = "coverage-v1"


def load_skkni_units(path: Path = SKKNI_JSON) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    units = []
    for domain, rows in data.items():
        for r in rows if isinstance(rows, list) else []:
            title = (r.get("unit_title") or "").strip()
            if title:
                units.append({"ref_id": r.get("unit_code") or f"{domain}:{len(units)}",
                              "title": title, "text": title, "group": domain})
    return units


def units_from_curriculum_text(text: str) -> list[dict]:
    """Sections of an uploaded curriculum, as reference units."""
    from curriculum_coverage import parse_curriculum_text
    return [{"ref_id": s.section_id, "title": s.title, "text": s.text, "group": "curriculum"}
            for s in parse_curriculum_text(text)]


def units_from_curriculum_pdf(pdf_bytes: bytes) -> list[dict]:
    from curriculum_coverage import parse_curriculum_pdf
    return [{"ref_id": s.section_id, "title": s.title, "text": s.text, "group": "curriculum"}
            for s in parse_curriculum_pdf(pdf_bytes)]


def _embedder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def competency_text(c: dict) -> str:
    return c["statement"]["text"] + ". " + "; ".join(
        m["text"] for m in c["evidence"]["members"][:5])


def audit(competencies: Sequence[dict], units: Sequence[dict], reference_name: str,
          cfg: CoverageConfig | None = None, embedder=None, official: bool = True) -> dict:
    """Adds `c["coverage"][reference_name]` to every competency and returns a
    summary. `official=True` enables the fourth status (input for updating the
    reference); for a school's own curriculum the uncovered ones stay
    `belum_tercakup` and the delta step decides what to propose."""
    cfg = cfg or CoverageConfig()
    summary = {"reference": reference_name, "n_reference_units": len(units),
               "coverage_version": cfg.version, "thresholds": [cfg.partial_at, cfg.covered_at],
               "thresholds_status": cfg.thresholds_status,
               "validated_by_expert": False, "counts": {s: 0 for s in COVERAGE_STATUSES}}
    if not competencies:
        return summary
    if not units:
        for c in competencies:
            c.setdefault("coverage", {})[reference_name] = {
                "proposed": MISSING, "best_similarity": None, "matches": [],
                "validated_by_expert": False, "note": "the reference has no units"}
            summary["counts"][MISSING] += 1
        return summary
    embedder = embedder or _embedder(cfg.embedder_model)
    kw = dict(normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    ref = embedder.encode([u["text"] for u in units], **kw)
    vec = embedder.encode([competency_text(c) for c in competencies], **kw)
    sims = vec @ ref.T
    for c, row in zip(competencies, sims):
        order = sorted(range(len(units)), key=lambda i: (-float(row[i]), i))[:cfg.top_matches]
        best = float(row[order[0]])
        if best >= cfg.covered_at:
            status = COVERED
        elif best >= cfg.partial_at:
            status = PARTIAL
        elif official and c.get("contract", {}).get("status") in (STATUS_CANDIDATE, STATUS_READY):
            status = UPDATE_INPUT
        else:
            status = MISSING
        c.setdefault("coverage", {})[reference_name] = {
            "proposed": status, "best_similarity": round(best, 4),
            "matches": [{"ref_id": units[i]["ref_id"], "title": units[i]["title"],
                         "group": units[i].get("group", ""),
                         "similarity": round(float(row[i]), 4)} for i in order],
            "validated_by_expert": False}
        summary["counts"][status] += 1
    return summary
