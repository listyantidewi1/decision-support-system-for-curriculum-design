"""SkillSpan probe — does a model cut the verb off a real, human-marked skill?

    .venv-ml\\Scripts\\python.exe -m labeling.skillspan_probe sample
    .venv-ml\\Scripts\\python.exe -m labeling.skillspan_probe run --max-usd 0.60 --prompt-version v0-guideline-only

The conformance set (`labeling.conformance`) is synthetic. This is its
real-text complement: sentences from the SkillSpan-derived data already in the
repo (Zhang et al., NAACL 2022), whose SKILL and KNOWLEDGE spans were marked by
human annotators on real job postings. Human SKILL spans there keep their verb
("build a meaningful engineering discipline", "designing architecture"), which
is exactly the behaviour the first smoke test suggested the models lack.

What it can and cannot show:

  * SkillSpan has TWO labels, this study has five. A SkillSpan SKILL may be a
    hard skill, a soft skill or a work behaviour; a KNOWLEDGE may be knowledge or
    a technology. So five-tag correctness is NOT measurable here. Model output is
    collapsed (SH/SS/WB -> SKILL, KN/TT -> KNOWLEDGE) before scoring.
  * SkillSpan was annotated under a different guideline. Its boundary habits
    differ from Paket B v3.0, so the absolute F1 is not an accuracy figure for
    this study. The useful readings are relative — the same model under prompt
    v0 versus v1 — and the head-retention breakdown below.
  * The data is English only.

Head retention: for every human SKILL span that a model overlaps with a
skill-type span, did the model's span start at or before the human one (the
head, usually the verb, is kept) or after it (the head was dropped)?

Outputs stay in the gitignored DATA/labeling/. Whether SkillSpan-derived text may
be redistributed is an open licence question (CLAUDE.md section 6), so no
SkillSpan sentence and no run output is written into a tracked file. (The repo
already tracks the SkillSpan-derived dataset itself; the docs and the
conventions list quote a few short phrases from it as examples, no more.)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from gold_standard.metrics import Span

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "baseline_versions" / "skill_llm" / "datasets" / "dev.jsonl"
PROBE_SEED = 20261118
PROBE_N = 100
MAX_CHARS = 300

COLLAPSE = {"SH": "SKILL", "SS": "SKILL", "WB": "SKILL", "KN": "KNOWLEDGE", "TT": "KNOWLEDGE"}
TWO_TAGS = ["SKILL", "KNOWLEDGE"]


def locate_gold(sentence: str, span_text: str, context: str = "") -> tuple[int, int] | None:
    """Offsets of a gold span, or None when it cannot be placed unambiguously.

    The source gives the span text and a few words of context, not offsets. A
    unique occurrence is taken directly; otherwise the context must pin it down.
    Ambiguous spans return None and the sentence is left out of the probe —
    guessing an occurrence would put a made-up boundary into the reference.
    """
    if not span_text:
        return None
    if sentence.count(span_text) == 1:
        i = sentence.index(span_text)
        return i, i + len(span_text)
    if context and sentence.count(context) == 1 and context.count(span_text) == 1:
        i = sentence.index(context) + context.index(span_text)
        return i, i + len(span_text)
    return None


def parse_row(row: dict) -> tuple[str, list[Span]] | None:
    """(sentence, gold spans) for one source row, or None if unusable."""
    msgs = row.get("messages") or []
    if len(msgs) < 3:
        return None
    sentence = msgs[1].get("content") or ""
    try:
        gold = json.loads(msgs[2].get("content") or "")
    except json.JSONDecodeError:
        return None
    spans = []
    for label in TWO_TAGS:
        for g in gold.get(label, []) or []:
            loc = locate_gold(sentence, (g or {}).get("skill_span", ""),
                              (g or {}).get("context", ""))
            if loc is None:
                return None
            spans.append(Span(loc[0], loc[1], label))
    return sentence, spans


def build_probe(source: Path = SOURCE, n: int = PROBE_N, seed: int = PROBE_SEED) -> list[dict]:
    """A seeded sample of sentences that carry at least one human SKILL span."""
    pool = []
    with Path(source).open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            parsed = parse_row(json.loads(line))
            if parsed is None:
                continue
            sentence, spans = parsed
            if not (20 <= len(sentence) <= MAX_CHARS):
                continue
            if not any(s.label == "SKILL" for s in spans):
                continue
            pool.append({"sentence_id": f"ss-dev-{i:05d}", "sentence_text": sentence,
                         "previous_sentence": "", "next_sentence": "", "gold": spans})
    rng = random.Random(seed)
    rng.shuffle(pool)
    return sorted(pool[:n], key=lambda d: d["sentence_id"])


def collapse(spans) -> list[Span]:
    """Five-tag verified spans -> the two SkillSpan labels (deduplicated)."""
    out = {(s.start, s.end, COLLAPSE[s.label]) for s in spans if s.label in COLLAPSE}
    return [Span(a, b, lab) for a, b, lab in sorted(out)]


def head_retention(gold: list[Span], hyp: list[Span]) -> dict:
    """What happened to each human SKILL span (hyp already collapsed)."""
    out = {"exact": 0, "head_kept": 0, "head_dropped": 0, "only_as_knowledge": 0,
           "missed": 0}
    for g in gold:
        if g.label != "SKILL":
            continue
        over = [h for h in hyp if max(g.start, h.start) < min(g.end, h.end)]
        skillish = [h for h in over if h.label == "SKILL"]
        if any(h.start == g.start and h.end == g.end for h in skillish):
            out["exact"] += 1
        elif skillish:
            best = max(skillish, key=lambda h: min(g.end, h.end) - max(g.start, h.start))
            out["head_kept" if best.start <= g.start else "head_dropped"] += 1
        elif over:
            out["only_as_knowledge"] += 1
        else:
            out["missed"] += 1
    return out


def score_model(probe: list[dict], spans_by_sid: dict[str, list]) -> dict:
    from gold_standard import metrics
    from gold_standard.metrics import wilson_ci
    sids = [p["sentence_id"] for p in probe]
    gold = {p["sentence_id"]: p["gold"] for p in probe}
    hyp = {sid: collapse(spans_by_sid.get(sid, [])) for sid in sids}
    totals = {"exact": 0, "head_kept": 0, "head_dropped": 0, "only_as_knowledge": 0,
              "missed": 0}
    for sid in sids:
        for k, v in head_retention(gold[sid], hyp[sid]).items():
            totals[k] += v
    touched = totals["exact"] + totals["head_kept"] + totals["head_dropped"]
    kept = totals["exact"] + totals["head_kept"]
    lo, hi = wilson_ci(kept, touched)
    out = {"n_sentences": len(sids),
           "n_gold_skill_spans": sum(totals.values()),
           "skill_span_outcomes": totals,
           "head_kept_rate": {"k": kept, "n": touched,
                              "rate": kept / touched if touched else float("nan"),
                              "wilson95": [lo, hi],
                              "meaning": "of human SKILL spans the model overlapped with a "
                                         "skill-type span, the share whose head it kept"}}
    for mode in ("strict", "partial"):
        pt = metrics.per_tag(gold, hyp, sids, mode, TWO_TAGS)
        out[mode] = {"micro": metrics.micro(metrics.contribs(gold, hyp, sids, mode)),
                     "SKILL_f1": pt["by_tag"]["SKILL"]["f1"],
                     "KNOWLEDGE_f1": pt["by_tag"]["KNOWLEDGE"]["f1"]}
    return out


def score_run(run_dir: Path, probe: list[dict] | None = None) -> dict:
    from .verify import verify_output
    probe = probe or build_probe()
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    sha = summary["request_sha256"]
    text = {p["sentence_id"]: p["sentence_text"] for p in probe}
    raw = {}
    for line in (run_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("request_sha256") == sha and r["sentence_id"] in text:
                raw[(r["model"], r["sentence_id"])] = r
    out = {"prompt_version": summary["prompt"]["prompt_version"], "request_sha256": sha,
           "probe_seed": PROBE_SEED, "models": {}}
    for model in summary["run_config"]["models"]:
        spans = {}
        for sid, sentence in text.items():
            rec = raw.get((model, sid))
            if rec and rec.get("ok"):
                spans[sid] = verify_output(model, sid, sentence, rec.get("raw") or "").spans
        out["models"][model] = score_model(probe, spans)
    return out


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample", help="show the probe's composition (no API calls)")
    rp = sub.add_parser("run")
    rp.add_argument("--max-usd", type=float, required=True)
    rp.add_argument("--prompt-version", required=True)
    rp.add_argument("--reasoning", default="none")
    # Short sentences, reasoning off: answers are well under 400 tokens. The
    # budget pre-flight assumes every call uses all of max_tokens AND every
    # retry, so a tight max_tokens is what keeps the worst case realistic.
    rp.add_argument("--max-tokens", type=int, default=400)
    rp.add_argument("--retries", type=int, default=1)
    sp = sub.add_parser("score")
    sp.add_argument("run_dir")
    args = ap.parse_args()

    probe = build_probe()
    if args.cmd == "sample":
        n_sk = sum(1 for p in probe for s in p["gold"] if s.label == "SKILL")
        n_kn = sum(1 for p in probe for s in p["gold"] if s.label == "KNOWLEDGE")
        print(f"{len(probe)} sentences, {n_sk} human SKILL spans, {n_kn} human "
              f"KNOWLEDGE spans (seed {PROBE_SEED})")
        return
    if args.cmd == "run":
        from .run import DEFAULT_MODELS, RUNS_DIR, BudgetError, RunConfig, run
        reasoning = None if args.reasoning == "default" else {"effort": args.reasoning}
        cfg = RunConfig(models=tuple(DEFAULT_MODELS), prices=DEFAULT_MODELS,
                        max_usd=args.max_usd, reasoning=reasoning,
                        max_tokens=args.max_tokens, retries=args.retries,
                        prompt_version=args.prompt_version)
        run_dir = RUNS_DIR / f"skillspan_probe_{cfg.request_sha256()[:8]}"
        try:
            summary = run([{k: v for k, v in p.items() if k != "gold"} for p in probe],
                          cfg, run_dir)
        except BudgetError as exc:
            raise SystemExit(f"BUDGET: {exc}")
        print(f"spent ${summary['cost']['charged_usd_total']:.4f}")
    else:
        run_dir = Path(args.run_dir)
    result = score_run(run_dir, probe)
    (run_dir / "skillspan_score.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"prompt {result['prompt_version']}")
    for model, s in result["models"].items():
        h, o = s["head_kept_rate"], s["skill_span_outcomes"]
        print(f"  {model:30} head kept {h['k']}/{h['n']}  outcomes {o}  "
              f"strict F1 {s['strict']['micro']['f1']:.3f}  partial F1 "
              f"{s['partial']['micro']['f1']:.3f}")
    print(f"-> {run_dir / 'skillspan_score.json'}")


if __name__ == "__main__":
    main()
