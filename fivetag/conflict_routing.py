"""Where a D1/D2 conflict goes — the risk rule.

Proposal §3.3.1: when the two deterministic classifiers disagree, the case is
labelled CONFLICT and "diteruskan ke mekanisme adjudication/LLM atau human
review sesuai tingkat risikonya", so that a conflict is never closed by an
unexplained automatic choice. The proposal names no risk rule. The researcher
chose this one on 2026-11-19; it is registered in
`docs/experiments/PREREG_d3_conflict_routing.md`, and a test keeps that
document and the constants below identical.

    Every conflict is first decided by the LLM adjudicator. Then:

    HIGH   the span is evidence under a competency that is about to be
           recommended (status "kandidat tinjauan" or "siap ditelaah formal").
           A human reviews it. The LLM's answer is shown as a suggestion.
    AUDIT  a fixed-size random sample of all other conflicts. A human reviews
           it too. Its only purpose is to MEASURE how often the LLM adjudicator
           is right, so the unreviewed rest can be trusted to a known degree.
    LOW    everything else. The LLM's decision stands.

    A conflict also goes to a human when the LLM's answer cannot be used: not
    valid JSON, not one of the allowed labels, or a label that neither
    classifier proposed.

The human effort therefore sits where a wrong tag could reach a curriculum
recommendation, and the automatic part is not taken on faith.

"One classifier decided, the other abstained" is counted as a conflict, as in
`typing_d.D3Consensus`, and is reported separately from a true disagreement.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

from . import TAG_CODES
from .typing_d import CONFLICT, D3Consensus

ROUTING_VERSION = "d3-routing-v1"
HIGH_RISK_STATUSES = ("kandidat_tinjauan", "siap_ditelaah_formal")
AUDIT_SAMPLE_N = 150            # starting value: see the registration for why 150
AUDIT_SEED = 20261119
NONE = "NONE"                   # the adjudicator may say "not one of the five tags"

HIGH, AUDIT, LOW = "high", "audit", "low"

# When may the adjudicator's unreviewed decisions be trusted? Registered
# 2026-11-19 as an addition to the routing rule, agreed by the promotor, before
# any audit sample has been reviewed. Judged on the LOWER bound of the Wilson
# 95% interval of the audit accuracy, never on the observed share.
ACCEPTANCE_VERSION = "d3-acceptance-v1"
ACCEPT_LOWER_BOUND = 0.80           # with 150 audited: at least 130 correct
BORDERLINE_LOWER_BOUND = 0.667      # with 150 audited: 112 to 129 correct
ACCEPT, BORDERLINE, REJECT = "accept", "borderline", "reject"


@dataclass(frozen=True)
class RoutingConfig:
    high_risk_statuses: tuple[str, ...] = HIGH_RISK_STATUSES
    audit_sample_n: int = AUDIT_SAMPLE_N
    seed: int = AUDIT_SEED
    version: str = ROUTING_VERSION


@dataclass
class Conflict:
    evidence_id: str
    text: str
    sentence_text: str
    start: int
    end: int
    extractor_tag: str
    d1: str
    d2: str
    kind: str                         # disagree | one_abstained
    competency_id: str = ""
    competency_status: str = ""
    risk: str = LOW
    human_review: bool = False
    reasons: list[str] = field(default_factory=list)
    llm_label: str = ""
    llm_status: str = ""              # "" (not asked) | ok | unusable
    final_label: str = ""             # filled by the LLM (low risk) or by a human

    def to_dict(self) -> dict:
        return asdict(self)


def find_conflicts(evidence: Sequence, d3: D3Consensus | None = None) -> list[Conflict]:
    """`evidence`: anything with evidence_id, text, sentence_text, start, end, tag."""
    d3 = d3 or D3Consensus()
    out = []
    for e in evidence:
        d = d3.classify(e.text, e.sentence_text, e.start, e.end)
        if d.label != CONFLICT:
            continue
        a, b = d.parts
        out.append(Conflict(e.evidence_id, e.text, e.sentence_text, e.start, e.end, e.tag,
                            a.label, b.label, d.conflict_kind))
    return out


def _rank(seed: int, evidence_id: str) -> str:
    return hashlib.sha256(f"{seed}|{evidence_id}".encode("utf-8")).hexdigest()


def assign_risk(conflicts: Sequence[Conflict],
                competency_of: Mapping[str, tuple[str, str]] | None = None,
                cfg: RoutingConfig | None = None) -> None:
    """Sets risk and human_review on every conflict, in place.

    `competency_of[evidence_id] = (competency_id, status)`. The audit sample is
    the `audit_sample_n` non-high conflicts with the smallest seeded hash, so
    it is reproducible and does not depend on the order of the input.
    """
    cfg, competency_of = cfg or RoutingConfig(), competency_of or {}
    rest = []
    for c in conflicts:
        c.competency_id, c.competency_status = competency_of.get(c.evidence_id, ("", ""))
        c.reasons = [r for r in c.reasons if r.startswith("LLM")]
        if c.competency_status in cfg.high_risk_statuses:
            c.risk, c.human_review = HIGH, True
            c.reasons.append(f"evidence of a competency with status {c.competency_status}")
        else:
            c.risk, c.human_review = LOW, False
            rest.append(c)
    for c in sorted(rest, key=lambda c: _rank(cfg.seed, c.evidence_id))[:cfg.audit_sample_n]:
        c.risk, c.human_review = AUDIT, True
        c.reasons.append("audit sample: measures the LLM adjudicator")


SYSTEM = (
    "You decide the tag of ONE phrase from a job posting. Tags: SH = a technical work action "
    "(what the person does); SS = an interpersonal or self-management ability; KN = a concept, "
    "principle, method or domain to be understood; TT = a specific named tool, language, "
    "framework, platform or standard; WB = an attitude or personal disposition; NONE = none of "
    "these. Read the phrase in its sentence. Answer with JSON only: {\"label\": \"SH\"}")


def build_messages(c: Conflict, with_candidates: bool = True) -> list[dict]:
    payload = {"sentence": c.sentence_text, "phrase": c.text}
    if with_candidates:
        payload["candidates"] = sorted({x for x in (c.d1, c.d2) if x in TAG_CODES})
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def parse_label(raw: str | None) -> str | None:
    """The adjudicator's label, or None when the answer cannot be used."""
    try:
        obj = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip()))
    except (ValueError, TypeError):
        return None
    label = obj.get("label") if isinstance(obj, dict) else None
    label = label.strip().upper() if isinstance(label, str) else None
    return label if label in (*TAG_CODES, NONE) else None


def evaluate_adjudicator(items: Sequence, call: Callable[[list[dict]], str | None],
                         d3: D3Consensus | None = None) -> dict:
    """How often does a model give the reference tag? For CHOOSING the adjudicator.

    `items`: spans with a known tag (anything with sentence, start, end, label,
    item_id; `typing_d.TypingItem` fits). Two questions are asked:

      all spans     the phrase in its sentence, no candidates shown. Measures
                    the model's own ability to tag a span.
      conflicts     only the spans where D1 and D2 differ, with the two
                    candidates shown, exactly as in real use.

    Returns per-item correctness for both, so two models can be compared on
    the same items with McNemar's test.
    """
    d3 = d3 or D3Consensus()
    out = {"all": {}, "conflicts": {}, "unusable": {"all": 0, "conflicts": 0}}
    for it in items:
        text = it.sentence[it.start:it.end]
        d = d3.classify(text, it.sentence, it.start, it.end)
        a, b = (d.parts[0].label, d.parts[1].label) if d.parts else ("", "")
        c = Conflict(it.item_id, text, it.sentence, it.start, it.end, "", a, b, d.conflict_kind)
        label = parse_label(call(build_messages(c, with_candidates=False)))
        out["unusable"]["all"] += label is None
        out["all"][it.item_id] = label == it.label
        if d.label == CONFLICT:
            label = parse_label(call(build_messages(c, with_candidates=True)))
            out["unusable"]["conflicts"] += label is None
            out["conflicts"][it.item_id] = label == it.label
    return out


def adjudicate_with_llm(conflicts: Sequence[Conflict], call: Callable[[list[dict]], str | None]
                        ) -> dict:
    """Ask the adjudicator about every conflict. `call(messages) -> raw text`.

    An answer that cannot be used, or that names a label neither classifier
    proposed, sends the conflict to a human whatever its risk was. Only a LOW
    conflict with a usable answer gets its final label here.
    """
    for c in conflicts:
        label = parse_label(call(build_messages(c)))
        if label is None:
            c.llm_status, c.llm_label = "unusable", ""
            c.human_review = True
            c.reasons.append("LLM answer unusable")
            continue
        c.llm_status, c.llm_label = "ok", label
        candidates = {x for x in (c.d1, c.d2) if x in TAG_CODES}
        if label != NONE and label not in candidates:
            c.human_review = True
            c.reasons.append("LLM chose a label neither classifier proposed")
        if not c.human_review:
            c.final_label = label
    return summarise(conflicts)


def summarise(conflicts: Sequence[Conflict], n_evidence: int | None = None) -> dict:
    n = len(conflicts)
    out = {"routing_version": ROUTING_VERSION, "n_conflicts": n,
           "by_kind": dict(Counter(c.kind for c in conflicts)),
           "by_risk": dict(Counter(c.risk for c in conflicts)),
           "n_for_human_review": sum(c.human_review for c in conflicts),
           "n_decided_by_llm_alone": sum(1 for c in conflicts
                                         if c.final_label and not c.human_review),
           "n_llm_asked": sum(1 for c in conflicts if c.llm_status),
           "n_llm_unusable": sum(1 for c in conflicts if c.llm_status == "unusable"),
           "by_pair": dict(Counter(f"D1={c.d1} D2={c.d2}" for c in conflicts).most_common(12))}
    if n_evidence:
        out["conflict_rate"] = n / n_evidence
    return out


def audit_accuracy(conflicts: Sequence[Conflict]) -> dict:
    """After the human has filled the audit sample: how often was the LLM right?
    Uses only AUDIT conflicts, because the HIGH ones are not a random sample."""
    from gold_standard.metrics import wilson_ci
    rows = [c for c in conflicts if c.risk == AUDIT and c.final_label and c.llm_status == "ok"]
    k = sum(1 for c in rows if c.llm_label == c.final_label)
    lo, hi = wilson_ci(k, len(rows))
    return {"n": len(rows), "agree": k, "accuracy": k / len(rows) if rows else float("nan"),
            "wilson95": [lo, hi]}


def acceptance(audit: dict) -> dict:
    """The registered verdict on an `audit_accuracy` result.

      accept      lower bound >= 0.80: the LLM's decisions on low-risk conflicts stand.
      borderline  lower bound >= 0.667: a second audit sample of the same size is
                  reviewed and this rule is applied once more to both samples
                  pooled. (Another model may be tried instead; it is a new
                  adjudicator and needs its own audit.)
      reject      otherwise: the LLM adjudicator is not used. Low-risk conflicts
                  stay CONFLICT and are left out of the evidence.
    """
    n, lower = audit["n"], (audit["wilson95"][0] if audit["n"] else float("nan"))
    if not n:
        verdict = "no audit yet"
    elif lower >= ACCEPT_LOWER_BOUND:
        verdict = ACCEPT
    elif lower >= BORDERLINE_LOWER_BOUND:
        verdict = BORDERLINE
    else:
        verdict = REJECT
    return {"verdict": verdict, "acceptance_version": ACCEPTANCE_VERSION, "n_audited": n,
            "n_agree": audit["agree"], "accuracy": audit["accuracy"], "lower_bound_95": lower,
            "accept_at": ACCEPT_LOWER_BOUND, "borderline_at": BORDERLINE_LOWER_BOUND}


def apply_rejection(conflicts: Sequence[Conflict]) -> int:
    """After a REJECT verdict: withdraw every decision the LLM made alone. Those
    conflicts stay unresolved. Returns how many were withdrawn."""
    n = 0
    for c in conflicts:
        if c.final_label and not c.human_review:
            c.final_label = ""
            c.reasons.append("LLM adjudicator rejected by the audit: left unresolved")
            n += 1
    return n


def write_worksheet(conflicts: Sequence[Conflict], path: Path) -> Path:
    """The human reviewer's sheet: one row per conflict that needs a person."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["evidence_id", "risk", "why", "sentence", "phrase", "extractor_tag",
                    "D1", "D2", "LLM_suggestion", "competency_id",
                    "FINAL (SH/SS/KN/TT/WB/NONE)", "note"])
        for c in sorted((c for c in conflicts if c.human_review),
                        key=lambda c: (c.risk != HIGH, c.competency_id, c.evidence_id)):
            w.writerow([c.evidence_id, c.risk, "; ".join(c.reasons), c.sentence_text, c.text,
                        c.extractor_tag, c.d1, c.d2, c.llm_label, c.competency_id, "", ""])
    return path


def route_run(run_dir: Path, cfg: RoutingConfig | None = None) -> dict:
    """Find and route the conflicts of a finished `competency.pipeline` run.
    Free: D1 and D2 are rules. The LLM adjudicator is NOT called here."""
    import pandas as pd
    run_dir = Path(run_dir)
    ev = pd.read_parquet(run_dir / "evidence.parquet")
    comps = json.loads((run_dir / "competencies.json").read_text(encoding="utf-8"))
    competency_of = {eid: (c["competency_id"], c["contract"]["status"])
                     for c in comps for eid in c["evidence"]["evidence_ids"]}
    conflicts = find_conflicts(list(ev.itertuples(index=False)))
    assign_risk(conflicts, competency_of, cfg)
    summary = summarise(conflicts, n_evidence=len(ev))
    sheet = write_worksheet(conflicts, run_dir / "d3_conflicts_for_human_review.csv")
    (run_dir / "d3_conflicts.json").write_text(json.dumps(
        {"summary": summary, "conflicts": [c.to_dict() for c in conflicts]},
        indent=2, ensure_ascii=False), encoding="utf-8")
    summary["worksheet"] = str(sheet)
    return summary


def main() -> None:
    import argparse
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="a DATA/competency/runs/<tag> folder")
    args = ap.parse_args()
    s = route_run(Path(args.run_dir))
    print(f"{s['n_conflicts']} conflicts ({s.get('conflict_rate', 0):.1%} of the evidence): "
          f"{s['by_kind']}")
    print(f"by risk: {s['by_risk']}")
    print(f"for a human: {s['n_for_human_review']}   left to the LLM: "
          f"{s['n_conflicts'] - s['n_for_human_review']}")
    print(f"most frequent: {s['by_pair']}")
    print(f"-> {s['worksheet']}")


if __name__ == "__main__":
    main()
