"""Dress rehearsal of E1 on SkillSpan — real text, real encoder, NOT a study result.

Why this exists: the experts' gold standard is weeks away, and the first time
the encoder meets real job-posting text should not be the day the real
experiment starts. SkillSpan (Zhang et al., 2022) is public, expert-annotated
job-posting text that is already in this repository. Training E1 on it shows
whether the whole route works — the pretrained encoder, the GPU, the layered
BIO heads, the scoring — and how long it takes.

What it cannot show. SkillSpan is English only and has two labels, SKILL and
KNOWLEDGE, defined by its own guideline. They are mapped here onto two of the
five tag layers (SKILL -> SH, KNOWLEDGE -> KN) only so the unchanged E1 code can
be used; SkillSpan's SKILL also covers what this study calls soft skills and
work behaviour. So the score says the pipeline learns from real annotations. It
says nothing about five-tag accuracy on Indonesian postings, and it is not one
of the numbers of the E0-E6 comparison.

"test" below means SkillSpan's own test file. It has nothing to do with the
study's sealed test split, which does not exist yet and which this script never
touches.

    # on Kaggle (GPU), after cloning the repository:
    python -m fivetag.skillspan_rehearsal --model xlm-roberta-base --epochs 6 --out /kaggle/working/e1_skillspan

    # on a laptop, to check the wiring in a minute (toy encoder, no download):
    python -m fivetag.skillspan_rehearsal --tiny --limit 600 --epochs 3 --out DATA/fivetag/rehearsal/tiny
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .encoder import EncoderConfig, predict_rows, train
from .evaluate import _json_safe, score_full
from .synthetic import as_reference

REPO_ROOT = Path(__file__).resolve().parents[1]
# the bundle was unpacked one level too deep; the doubled path is the tracked one
CANDIDATE_DIRS = (REPO_ROOT / "baseline_versions" / "skill_llm" / "datasets",
                  REPO_ROOT / "baseline_versions" / "skill_llm" / "skill_llm" / "datasets")
LABEL_MAP = {"SKILL": "SH", "KNOWLEDGE": "KN"}
NOTICE = ("REHEARSAL on SkillSpan (English, two labels mapped onto SH and KN). "
          "It checks that the pipeline learns from real annotations. Not a study result.")


def data_dir() -> Path:
    for d in CANDIDATE_DIRS:
        if (d / "train.jsonl").is_file():
            return d
    raise FileNotFoundError(f"SkillSpan files not found in {[str(d) for d in CANDIDATE_DIRS]}")


def _locate(text: str, span: str, context: str, taken: set[tuple[int, int]]) -> int:
    """Start offset of `span`, using its context to choose among repeats; -1 if absent."""
    if context and span in context:
        c = text.find(context)
        if c >= 0:
            start = c + context.find(span)
            if (start, start + len(span)) not in taken:
                return start
    start = text.find(span)
    while start >= 0 and (start, start + len(span)) in taken:
        start = text.find(span, start + 1)
    return start


def load_part(name: str, folder: Path | None = None) -> tuple[list[dict], dict]:
    """(rows in the encoder's format, conversion statistics)."""
    path = (folder or data_dir()) / f"{name}.jsonl"
    rows, stats = [], {"n_sentences": 0, "n_spans": 0, "n_not_found": 0, "n_bad_records": 0,
                       "n_same_tag_overlap_dropped": 0}
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            try:
                msgs = json.loads(line)["messages"]
                text, answer = msgs[1]["content"], json.loads(msgs[2]["content"])
            except (KeyError, IndexError, ValueError, TypeError):
                stats["n_bad_records"] += 1
                continue
            spans, taken = [], set()
            for key, tag in LABEL_MAP.items():
                kept: list[tuple[int, int]] = []
                for entry in answer.get(key, []):
                    if isinstance(entry, dict):
                        span, context = entry.get("skill_span") or "", entry.get("context") or ""
                    else:
                        span, context = str(entry), ""
                    start = _locate(text, span, context, taken) if span else -1
                    if start < 0:
                        stats["n_not_found"] += 1
                        continue
                    a, b = start, start + len(span)
                    if any(a < y and x < b for x, y in kept):      # one BIO layer per tag
                        stats["n_same_tag_overlap_dropped"] += 1
                        continue
                    kept.append((a, b))
                    taken.add((a, b))
                    spans.append({"start": a, "end": b, "label": tag})
            rows.append({"sentence_id": f"skillspan-{name}-{i:05d}", "text": text,
                         "spans": sorted(spans, key=lambda s: (s["start"], s["end"])),
                         "source": "skillspan", "part": name})
            stats["n_sentences"] += 1
            stats["n_spans"] += len(spans)
    return rows, stats


def _score(model, tokenizer, rows, cfg, device) -> dict:
    t0 = time.perf_counter()
    preds = predict_rows(model, tokenizer, rows, cfg, device)
    dt = time.perf_counter() - t0
    _, ref = as_reference(rows)
    s = score_full(ref, preds)
    by = s["strict"]["per_tag"]["by_tag"]
    return {"n_sentences": len(rows), "n_ref_spans": s["n_ref"], "n_pred_spans": s["n_hyp"],
            "strict": {k: s["strict"][k] for k in ("precision", "recall", "f1", "f1_ci95")},
            "partial": {k: s["partial"][k] for k in ("precision", "recall", "f1")},
            "strict_f1_by_label": {"SKILL (SH layer)": by["SH"]["f1"],
                                   "KNOWLEDGE (KN layer)": by["KN"]["f1"]},
            "spurious_spans_in_untrained_layers": sum(
                1 for ps in preds.values() for p in ps if p.label not in LABEL_MAP.values()),
            "error_taxonomy": s["error_taxonomy"],
            "seconds": dt, "sentences_per_second": len(rows) / dt if dt else None}


def run(out_dir: Path, *, model_name: str = "xlm-roberta-base", epochs: int = 6,
        batch_size: int = 16, lr: float = 3e-5, seed: int = 42, limit: int | None = None,
        tiny: bool = False, log=print) -> dict:
    import torch
    parts, conversion = {}, {}
    for name in ("train", "dev", "test"):
        parts[name], conversion[name] = load_part(name)
        if limit:
            parts[name] = parts[name][: limit if name == "train" else max(50, limit // 3)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(NOTICE)
    log(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    log("sentences: " + ", ".join(f"{k} {len(v)}" for k, v in parts.items()))
    tokenizer = encoder = None
    if tiny:
        from .tiny import tiny_encoder, tiny_tokenizer
        tokenizer = tiny_tokenizer([r["text"] for rows in parts.values() for r in rows])
        encoder = tiny_encoder(len(tokenizer), seed)
        model_name, lr = "tiny-random", 3e-3
    cfg = EncoderConfig(model_name=model_name, condition="E1", epochs=epochs,
                        batch_size=batch_size, lr=lr, seed=seed)
    t0 = time.perf_counter()
    result = train(parts["train"], parts["dev"], cfg, Path(out_dir), tokenizer=tokenizer,
                   encoder=encoder, device=device, log=log)
    train_seconds = time.perf_counter() - t0
    model, tokenizer = result["model"], result["tokenizer"]
    report = {
        "notice": NOTICE, "model_name": model_name, "tiny_toy_model": tiny,
        "limit": limit, "device": device, "epochs": epochs, "batch_size": batch_size,
        "lr": lr, "seed": seed, "label_map": LABEL_MAP, "conversion": conversion,
        "train_seconds": train_seconds, "history": result["log"]["history"],
        "dev": _score(model, tokenizer, parts["dev"], cfg, device),
        "skillspan_test": _score(model, tokenizer, parts["test"], cfg, device),
    }
    out_dir = Path(out_dir)
    (out_dir / "skillspan_rehearsal_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    return report


def to_text(report: dict) -> str:
    lines = [report["notice"], ""]
    if report["tiny_toy_model"]:
        lines.append("TOY MODEL (randomly initialised, word-level): this run only checks the "
                     "wiring. Its scores mean nothing.")
    lines.append(f"model {report['model_name']}, {report['epochs']} epochs on {report['device']}, "
                 f"training took {report['train_seconds'] / 60:.1f} minutes")
    for name, key in (("dev", "dev"), ("SkillSpan test", "skillspan_test")):
        s = report[key]
        ci = s["strict"]["f1_ci95"]
        lines.append(f"{name:<15} strict P {s['strict']['precision']:.3f}  R "
                     f"{s['strict']['recall']:.3f}  F1 {s['strict']['f1']:.3f} "
                     f"(95% CI {ci[0]:.3f}-{ci[1]:.3f})   partial F1 {s['partial']['f1']:.3f}")
        lines.append(f"{'':<15} by label: " + ", ".join(
            f"{k} {v:.3f}" for k, v in s["strict_f1_by_label"].items())
            + f";  spans in untrained layers: {s['spurious_spans_in_untrained_layers']}")
    c = report["conversion"]["train"]
    lines.append(f"conversion (train): {c['n_spans']} spans placed, {c['n_not_found']} not "
                 f"found in their sentence, {c['n_same_tag_overlap_dropped']} same-label overlaps "
                 f"dropped")
    return "\n".join(lines)


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="use only this many training sentences")
    ap.add_argument("--tiny", action="store_true", help="toy encoder, no download (wiring check)")
    args = ap.parse_args()
    report = run(Path(args.out), model_name=args.model, epochs=args.epochs,
                 batch_size=args.batch_size, lr=args.lr, seed=args.seed, limit=args.limit,
                 tiny=args.tiny)
    print()
    print(to_text(report))
    print(f"\nreport: {Path(args.out) / 'skillspan_rehearsal_report.json'}")


if __name__ == "__main__":
    main()
