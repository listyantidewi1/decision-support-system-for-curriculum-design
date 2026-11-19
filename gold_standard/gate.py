"""Pilot go/no-go check — reads an agreement report and applies a written rule.

    .venv-ml\\Scripts\\python.exe -m gold_standard.gate <iaa_report.json>

The rule lives in docs/experiments/PREREG_pilot_gate.md. It has to be fixed
BEFORE any expert saves a pilot annotation, so the decision to go on to the main
1,200 sentences cannot be shaped by having seen the result.

**Status of the rule: see `RULE_STATUS` below.** It was registered on 2026-11-19,
before any pilot annotation existed, and may not change once one does.

What the rule looks at, and why these three measures:

  * relevance and has_5tag are judged per sentence. Their kappa is sensitive to
    prevalence — the pilot sample is deliberately skewed, and at fixed annotator
    accuracy kappa falls as one category dominates — so the gate reads PABAK
    (prevalence- and bias-adjusted kappa), which is a function of observed
    agreement only. Kappa is still reported beside it.
  * the token-level Fleiss kappa is the span-level counterpart with a chance
    baseline, and is the figure comparable to SkillSpan's published 0.70-0.75.

What the gate deliberately does not do: it makes no reliability CLAIM. With 100
sentences the 95% interval on these figures is about 0.3 wide, so the pilot can
tell "broadly fine" from "something is wrong", not more. The reliability figure
the dissertation reports comes from the main 1,200.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

# ---- the rule (mirror of docs/experiments/PREREG_pilot_gate.md) ---------------
RULE_VERSION = "pilot-gate-v1"
RULE_STATUS = "registered 2026-11-19, approved by the researcher"
GO_AT = 0.60            # at or above: fine
STOP_BELOW = 0.40       # below: the guideline or the training needs rework
MIN_COMPLETE_SHARE = 0.95

GO = "GO"
RECALIBRATE = "RECALIBRATE"
STOP = "STOP"
NOT_READY = "NOT READY"

MEANING = {
    GO: "Hold the calibration session, freeze the guideline, and go on to the main "
        "1,200 sentences.",
    RECALIBRATE: "Hold the calibration session and revise the guideline wording where "
                 "the disagreements cluster. Then have the three annotators do a fresh "
                 "set of 30 sentences and run this check again before the main 1,200.",
    STOP: "Do not start the main 1,200. Agreement is low enough that the guideline or "
          "the annotator training needs rework first; discuss with your promotor.",
    NOT_READY: "The pilot is not complete. Finish the annotation before reading any "
               "agreement figure.",
}


def _num(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _band(value: float) -> str:
    if value != value:
        return "undefined"
    if value >= GO_AT:
        return "ok"
    if value >= STOP_BELOW:
        return "borderline"
    return "low"


def evaluate(report: dict, n_expected: int | None = None) -> dict:
    """Apply the rule to an agreement report (the JSON `gold_standard.iaa` writes)."""
    sl = report.get("sentence_level", {})
    rel, tag = sl.get("relevance", {}), sl.get("has_5tag", {})
    tok = report.get("token_level", {})
    n_annotators = len(report.get("annotators", []))

    n_complete = min(int(rel.get("n_complete_cases", 0) or 0),
                     int(tag.get("n_complete_cases", 0) or 0))
    n_any = int(rel.get("n_sentences_any", 0) or 0)
    n_expected = int(n_expected) if n_expected else n_any
    complete_share = (n_complete / n_expected) if n_expected else 0.0

    checks = []
    for name, value, kappa in (
            ("relevance (PABAK)", _num(rel.get("pabak")), _num(rel.get("fleiss_kappa"))),
            ("has_5tag (PABAK)", _num(tag.get("pabak")), _num(tag.get("fleiss_kappa"))),
            ("spans, token level (Fleiss kappa)", _num(tok.get("fleiss_kappa")), None)):
        checks.append({"measure": name, "value": value, "band": _band(value),
                       "kappa_for_reference": kappa})

    problems = []
    if n_annotators < 3:
        problems.append(f"only {n_annotators} annotator(s) in the report; the design needs 3")
    if complete_share < MIN_COMPLETE_SHARE:
        problems.append(f"{n_complete} of {n_expected} sentences were judged by all "
                        f"annotators ({complete_share:.0%}); the rule needs "
                        f"{MIN_COMPLETE_SHARE:.0%}")

    bands = [c["band"] for c in checks]
    if problems:
        verdict = NOT_READY
    elif "low" in bands:
        verdict = STOP
    elif "borderline" in bands or "undefined" in bands:
        verdict = RECALIBRATE
    else:
        verdict = GO

    prov = report.get("provenance") or {}
    synthetic = bool(prov.get("input_quarantined")) or not prov.get(
        "ids_verified_against_sample", True)
    span = report.get("span_level", {}).get("modes", {})
    return {
        "rule_version": RULE_VERSION,
        "rule_status": RULE_STATUS,
        "thresholds": {"go_at": GO_AT, "stop_below": STOP_BELOW,
                       "min_complete_share": MIN_COMPLETE_SHARE},
        "split": report.get("split"),
        "verdict": verdict,
        "meaning": MEANING[verdict],
        "checks": checks,
        "problems": problems,
        "n_complete": n_complete, "n_expected": n_expected,
        "complete_share": complete_share,
        # reported for context, not part of the rule: there is no agreed
        # benchmark for inter-annotator span F1 under a five-tag scheme
        "context_not_gating": {
            "span_exact_f1": _num((span.get("exact", {}).get("micro", {}) or {}).get("f1")),
            "span_overlap_f1": _num((span.get("overlap_iou50", {}).get("micro", {})
                                     or {}).get("f1")),
        },
        "input_is_synthetic": synthetic,
        "evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def to_text(result: dict) -> str:
    lines = [f"PILOT GO/NO-GO CHECK   rule {result['rule_version']}  "
             f"({result['rule_status']})"]
    if result["input_is_synthetic"]:
        lines.append("*** SYNTHETIC OR UNVERIFIED INPUT - this is a rehearsal, "
                     "not a decision ***")
    lines.append("")
    lines.append(f"  sentences judged by all annotators: {result['n_complete']} of "
                 f"{result['n_expected']}")
    for c in result["checks"]:
        v = "n/a" if c["value"] != c["value"] else f"{c['value']:.2f}"
        lines.append(f"  {c['measure']:36} {v:>5}   {c['band']}")
    ctx = result["context_not_gating"]
    lines.append(f"  (for context, not part of the rule: exact span F1 "
                 f"{ctx['span_exact_f1']:.2f}, overlap span F1 {ctx['span_overlap_f1']:.2f})")
    for p in result["problems"]:
        lines.append(f"  PROBLEM: {p}")
    lines += ["", f"VERDICT: {result['verdict']}", result["meaning"]]
    return "\n".join(lines)


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", help="the iaa_<split>_<stamp>.json file")
    ap.add_argument("--n-expected", type=int, default=None,
                    help="how many sentences the split holds (default: read the sample)")
    args = ap.parse_args()
    path = Path(args.report)
    report = json.loads(path.read_text(encoding="utf-8"))
    n_expected = args.n_expected
    if n_expected is None:
        from .iaa import _text_by_sid
        n_expected = len(_text_by_sid(report.get("split", ""))) or None
    result = evaluate(report, n_expected)
    out = path.with_name(path.name.replace("iaa_", "gate_", 1))
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(to_text(result))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
