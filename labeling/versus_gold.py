"""The LLMs against the experts' gold — the comparison the pilot exists to make.

Given a finished ensemble run and an adjudicated gold version over the SAME
sentences, this scores, with the shared metrics engine:

  * every model on its own (after the verifier),
  * the ensemble's automatically accepted spans (unanimous, and with
    majority acceptance switched on),
  * each pair of models against each other (paired bootstrap on F1, McNemar on
    recall),

and reports strict / partial span-F1, per-tag and macro F1, the confusion
matrix and the error taxonomy. It costs nothing: it re-reads the stored answers.

    python -m labeling.versus_gold DATA/labeling/runs/pilot_100_<id> ^
        --gold-dir DATA/gold_standard/gold/pilot_100/v1

What the numbers are for (proposal §3.4.3): deciding whether automatic
acceptance is precise enough to label the wider corpus. What they are NOT for:
the sealed test split. PROP:1442 forbids using the held-out test set for prompt
refinement, example selection or model weighting, so this command refuses any
gold sentence that belongs to a sealed test part.

A gold made in a rehearsal (under `_synthetic/`) can be used to try the command;
the output then says SYNTHETIC on every line that carries a number.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gold_standard import metrics
from gold_standard.iaa import Span, _quarantine_marker

from .consensus import ConsensusConfig, accepted_spans_by_sid, decide_sentence
from .verify import verify_output

REPO_ROOT = Path(__file__).resolve().parents[1]
GS_DIR = REPO_ROOT / "DATA" / "gold_standard"
SPLITS_DIR = GS_DIR / "splits"
GOLD_VERDICT = "GOLD"


class VersusGoldError(RuntimeError):
    pass


def sealed_test_ids(splits_dir: Path = SPLITS_DIR) -> set[str]:
    out: set[str] = set()
    if splits_dir.is_dir():
        for p in splits_dir.rglob("test_blind.parquet"):
            out |= set(pd.read_parquet(p, columns=["sentence_id"])["sentence_id"].astype(str))
    return out


def load_gold(gold_dir: Path, sample: str | None = None,
              gs_dir: Path = GS_DIR) -> tuple[dict[str, str], dict[str, list[Span]], dict]:
    """(text by sentence id, GOLD spans by sentence id, info). Every adjudicated
    sentence is a scoring unit, including those with no gold span.

    The gold files hold ids, offsets and tags, not the sentences; the text is
    read from the sample the gold was made on (named in the gold's MANIFEST).
    """
    gold_dir = Path(gold_dir)
    sents = pd.read_parquet(gold_dir / "gold_sentences.parquet")
    spans = pd.read_parquet(gold_dir / "gold_spans.parquet")
    if sample is None:
        manifest = gold_dir / "MANIFEST.json"
        if manifest.is_file():
            sample = json.loads(manifest.read_text(encoding="utf-8")).get("split")
    sample_path = Path(gs_dir) / f"{sample}.parquet"
    if not sample or not sample_path.is_file():
        raise VersusGoldError(f"cannot find the sample the gold was made on ({sample_path}); "
                              f"name it with --sample")
    frame = pd.read_parquet(sample_path, columns=["sentence_id", "sentence_text"])
    all_text = dict(zip(frame["sentence_id"].astype(str), frame["sentence_text"].astype(str)))
    unknown = [sid for sid in sents["sentence_id"].astype(str) if sid not in all_text]
    if unknown:
        raise VersusGoldError(f"{len(unknown)} gold sentences are not in sample {sample} "
                              f"(first: {unknown[:3]})")
    text = {sid: all_text[sid] for sid in sents["sentence_id"].astype(str)}
    ref: dict[str, list[Span]] = {sid: [] for sid in text}
    n_challenge = 0
    for r in spans.itertuples(index=False):
        sid = str(r.sentence_id)
        if sid not in ref:
            raise VersusGoldError(f"gold span on unknown sentence {sid}")
        if str(r.verdict).upper() != GOLD_VERDICT:
            n_challenge += 1
            continue
        if not (0 <= int(r.span_start) < int(r.span_end) <= len(text[sid])):
            raise VersusGoldError(f"gold span {r.span_start}:{r.span_end} does not fit "
                                  f"sentence {sid}: the gold and the sample disagree")
        ref[sid].append(Span(int(r.span_start), int(r.span_end), str(r.tag)))
    marker = _quarantine_marker(gold_dir.resolve())
    return text, ref, {"gold_dir": str(gold_dir), "n_sentences": len(text),
                       "n_gold_spans": sum(len(v) for v in ref.values()),
                       "n_challenge_spans_not_scored": n_challenge,
                       "synthetic": bool(marker), "synthetic_marker": marker}


def load_run(run_dir: Path, text: dict[str, str]) -> tuple[list[str], dict, dict]:
    """(models, verified outputs by model and sentence id, run summary)."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    sha, models = summary["request_sha256"], list(summary["run_config"]["models"])
    raw = {}
    for line in (run_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("request_sha256") == sha:
                raw[(r["model"], r["sentence_id"])] = r
    outs: dict[str, dict] = {m: {} for m in models}
    for m in models:
        for sid, sentence in text.items():
            rec = raw.get((m, sid))
            if rec is not None and rec.get("ok"):
                outs[m][sid] = verify_output(m, sid, sentence, rec.get("raw") or "")
    return models, outs, summary


def _score(ref, hyp, sids) -> dict:
    out = {}
    for mode in ("strict", "partial"):
        c = metrics.contribs(ref, hyp, sids, mode)
        m = metrics.micro(c)
        m["f1_ci95"] = metrics.bootstrap_f1_ci(c)
        m["per_tag"] = metrics.per_tag(ref, hyp, sids, mode)
        out[mode] = m
    out["confusion_matrix"] = metrics.confusion_matrix(ref, hyp, sids)
    out["error_taxonomy"] = metrics.error_taxonomy(ref, hyp, sids)
    return out


def compare(run_dir: Path, gold_dir: Path, *, splits_dir: Path = SPLITS_DIR,
            sample: str | None = None, gs_dir: Path = GS_DIR) -> dict:
    text, ref, gold_info = load_gold(gold_dir, sample, gs_dir)
    leaked = sorted(set(text) & sealed_test_ids(splits_dir))
    if leaked:
        raise VersusGoldError(
            f"{len(leaked)} of these sentences are in a sealed test part (first: "
            f"{leaked[:3]}). The test split is not used to compare or tune the LLMs.")
    models, outs, summary = load_run(run_dir, text)
    covered = [sid for sid in text if all(sid in outs[m] for m in models)]
    missing = {m: sum(1 for sid in text if sid not in outs[m]) for m in models}
    if not covered:
        raise VersusGoldError("the run has no answers for these gold sentences: the run "
                              "and the gold are over different sentences")
    sids = list(text)                       # scored on ALL gold sentences: a missing
    result = {"gold": gold_info,            # answer is an empty answer, not a skipped one
              "run": {"run_dir": str(run_dir), "prompt_version": summary["prompt"]["prompt_version"],
                      "request_sha256": summary["request_sha256"], "models": models,
                      "n_sentences_without_answer": missing},
              "n_sentences": len(sids), "models": {}, "ensemble": {}, "pairs": {}}
    hyp = {m: {sid: [Span(s.start, s.end, s.label) for s in outs[m][sid].spans]
               if sid in outs[m] else [] for sid in sids} for m in models}
    for m in models:
        result["models"][m] = _score(ref, hyp[m], sids)
    for name, majority in (("unanimous_only", False), ("with_majority", True)):
        cfg = ConsensusConfig(models=tuple(models), accept_majority=majority)
        decided = [decide_sentence(sid, text[sid], [outs[m][sid] for m in models
                                                   if sid in outs[m]], cfg) for sid in sids]
        accepted = accepted_spans_by_sid(decided)
        s = _score(ref, accepted, sids)
        k, n = s["strict"]["tp"], s["strict"]["n_hyp"]
        s["automatic_acceptance_precision"] = {
            "k": k, "n": n, "rate": k / n if n else float("nan"),
            "wilson95": list(metrics.wilson_ci(k, n))}
        result["ensemble"][name] = s
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            result["pairs"][f"{a} vs {b}"] = {
                "paired_bootstrap_f1_strict": metrics.paired_bootstrap_f1(
                    metrics.contribs(ref, hyp[a], sids, "strict"),
                    metrics.contribs(ref, hyp[b], sids, "strict")),
                "mcnemar_recall_strict": metrics.mcnemar_recall_strict(ref, hyp[a], hyp[b], sids)}
    return result


def to_text(result: dict) -> str:
    g, tag = result["gold"], ""
    lines = []
    if g["synthetic"]:
        tag = "  [SYNTHETIC]"
        lines.append("SYNTHETIC GOLD (rehearsal): these numbers check the command. "
                     "They are not a result.")
    lines.append(f"{result['n_sentences']} sentences, {g['n_gold_spans']} gold spans; "
                 f"prompt {result['run']['prompt_version']}")
    head = f"{'':<34}{'P':>7}{'R':>7}{'F1':>7}{'F1 95% CI':>16}{'partF1':>8}{'macroF1':>9}"
    lines += ["", head, "-" * len(head)]

    def line(name, s):
        st, ci = s["strict"], s["strict"]["f1_ci95"]
        ci_txt = f"{ci[0]:.2f}–{ci[1]:.2f}" if isinstance(ci, (list, tuple)) else str(ci)
        return (f"{name:<34}{st['precision']:>7.3f}{st['recall']:>7.3f}{st['f1']:>7.3f}"
                f"{ci_txt:>16}{s['partial']['f1']:>8.3f}{st['per_tag']['macro_f1']:>9.3f}{tag}")
    for m, s in result["models"].items():
        lines.append(line(m, s))
    for name, s in result["ensemble"].items():
        lines.append(line(f"ensemble, {name.replace('_', ' ')}", s))
    lines.append("")
    for name, s in result["ensemble"].items():
        a = s["automatic_acceptance_precision"]
        lines.append(f"automatic acceptance ({name.replace('_', ' ')}): {a['k']}/{a['n']} "
                     f"accepted spans are gold = {a['rate']:.1%} "
                     f"(95% CI {a['wilson95'][0]:.1%}–{a['wilson95'][1]:.1%}){tag}")
    miss = {m: n for m, n in result["run"]["n_sentences_without_answer"].items() if n}
    if miss:
        lines.append(f"sentences with no usable answer (scored as empty): {miss}")
    if g["n_challenge_spans_not_scored"]:
        lines.append(f"{g['n_challenge_spans_not_scored']} AMBIGUOUS/EXCLUDE spans are not "
                     f"part of the gold and were not scored")
    return "\n".join(lines)


def _ci_as_list(x):
    if isinstance(x, float) and x != x:
        return None
    if isinstance(x, dict):
        return {str(k): _ci_as_list(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_ci_as_list(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        try:
            return _ci_as_list(x.item())
        except (ValueError, AttributeError):
            return str(x)
    return x


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--gold-dir", required=True)
    ap.add_argument("--sample", default=None,
                    help="sample the gold was made on (default: read from the gold's MANIFEST)")
    args = ap.parse_args()
    try:
        result = compare(Path(args.run_dir), Path(args.gold_dir), sample=args.sample)
    except VersusGoldError as exc:
        raise SystemExit(f"REFUSED: {exc}")
    print(to_text(result))
    name = Path(args.gold_dir).name + ("_SYNTHETIC" if result["gold"]["synthetic"] else "")
    out = Path(args.run_dir) / f"versus_gold_{name}.json"
    out.write_text(json.dumps(_ci_as_list(result), indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
