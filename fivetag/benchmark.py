"""One table for every extraction condition, on one reference.

    python -m fivetag.benchmark synthetic            # code check, free, ~1 minute
    python -m fivetag.benchmark gold --rows dev.jsonl --train train.jsonl ...   (later)

`synthetic` runs what can run today without spending anything:

    E0   rules / dictionary
    E1   a TINY randomly-initialised encoder trained on synthetic sentences
    E6   router: E0 first, a stand-in "LLM" on escalation

and prints the same columns the real benchmark will have. The stand-in LLM is an
ORACLE that returns the reference with a fixed share of spans removed — it
exists so the router's plumbing and accounting can be checked. Its score is
meaningless by construction.

    Everything this prints under `synthetic` is a check that the code runs.
    None of it is a research result.

`compare` scores stored predictions of several conditions against a real
reference and adds the paired tests; it is what Workstream D will call once the
gold standard exists.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Mapping, Sequence

from .base import Pred, make_pred
from .evaluate import _json_safe, compare_full, score_full

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "DATA" / "fivetag" / "benchmarks"


def row(name: str, score: dict, seconds: float, n: int, extra: dict | None = None) -> dict:
    return {"condition": name,
            "strict_p": score["strict"]["precision"], "strict_r": score["strict"]["recall"],
            "strict_f1": score["strict"]["f1"], "partial_f1": score["partial"]["f1"],
            "macro_f1": score["strict"]["per_tag"]["macro_f1"],
            "sec_per_1000_sentences": (1000 * seconds / n) if n else float("nan"),
            **(extra or {})}


def format_table(rows: Sequence[dict]) -> str:
    head = f"{'condition':<26}{'P':>7}{'R':>7}{'F1':>7}{'partF1':>8}{'macroF1':>9}{'s/1000':>9}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r['condition']:<26}{r['strict_p']:>7.3f}{r['strict_r']:>7.3f}"
                     f"{r['strict_f1']:>7.3f}{r['partial_f1']:>8.3f}{r['macro_f1']:>9.3f}"
                     f"{r['sec_per_1000_sentences']:>9.2f}")
    return "\n".join(lines)


class OracleWithNoise:
    """Stand-in for the LLM path in the synthetic check: the reference, minus a
    fixed share of spans. NOT a model."""
    name = "oracle-stand-in"

    def __init__(self, rows: Sequence[dict], drop: float = 0.1, seed: int = 0):
        rng = random.Random(seed)
        self._by_text = {}
        for r in rows:
            keep = [s for s in r["spans"] if rng.random() >= drop]
            self._by_text[r["text"]] = [make_pred(r["text"], s["start"], s["end"], s["label"],
                                                  source="oracle", rule="stand-in") for s in keep]

    def extract(self, sentence: str) -> list[Pred]:
        if sentence not in self._by_text:
            raise KeyError(sentence)
        return list(self._by_text[sentence])

    def manifest(self) -> dict:
        return {"extractor": self.name, "note": "reference with spans removed; not a model"}


def run_synthetic(n_train: int = 600, n_dev: int = 200, epochs: int = 8, seed: int = 0,
                  log=print) -> dict:
    from . import synthetic
    from .e0_rules import E0Rules
    from .encoder import EncoderConfig, EncoderExtractor, train
    from .router import Router, summarise
    from .tiny import tiny_encoder, tiny_tokenizer

    train_rows = synthetic.generate(n_train, seed=seed + 1)
    dev_rows = synthetic.generate(n_dev, seed=seed + 2)
    sents, ref = synthetic.as_reference(dev_rows)
    rows, preds_by = [], {}

    def timed(fn):
        t0 = time.perf_counter()
        out = fn()
        return out, time.perf_counter() - t0

    e0 = E0Rules()
    p0, dt = timed(lambda: {k: e0.extract(v) for k, v in sents.items()})
    preds_by["E0"] = p0
    rows.append(row("E0 rules/dictionary", score_full(ref, p0, bootstrap=False), dt, len(sents)))

    tok = tiny_tokenizer([r["text"] for r in train_rows + dev_rows])
    cfg = EncoderConfig(model_name="tiny-random", epochs=epochs, batch_size=32, lr=3e-3,
                        head_lr=3e-3, seed=seed)
    result = train(train_rows, dev_rows, cfg, tokenizer=tok, encoder=tiny_encoder(len(tok), seed),
                   log=lambda *_: None)
    e1 = EncoderExtractor(result["model"], tok, cfg, result["log"])
    p1, dt = timed(lambda: e1.extract_many(sents))
    preds_by["E1"] = p1
    rows.append(row("E1 tiny encoder (toy)", score_full(ref, p1, bootstrap=False), dt, len(sents)))

    router = Router(e0, OracleWithNoise(dev_rows, seed=seed), secondary=e1)
    routes, dt = timed(lambda: {k: router.route(v) for k, v in sents.items()})
    p6 = {k: r.preds for k, r in routes.items()}
    preds_by["E6"] = p6
    summary = summarise(list(routes.values()))
    rows.append(row("E6 router (oracle LLM)", score_full(ref, p6, bootstrap=False), dt,
                    len(sents), {"escalation_rate": summary["escalation_rate"]}))
    return {"notice": synthetic.NOTICE, "n_train": n_train, "n_dev": n_dev, "rows": rows,
            "router": summary,
            "paired_E0_vs_E1": compare_full(ref, p0, p1)}


def compare(ref_by_sid: Mapping[str, Sequence], preds_by_condition: Mapping[str, Mapping],
            seconds: Mapping[str, float] | None = None, baseline: str | None = None) -> dict:
    """Score stored predictions of several conditions on one reference, and test
    each against `baseline` (paired bootstrap on F1, McNemar on recall)."""
    sids = list(ref_by_sid)
    out = {"n_sentences": len(sids), "rows": [], "scores": {}, "paired": {}}
    for name, preds in preds_by_condition.items():
        s = score_full(ref_by_sid, preds, sids)
        out["scores"][name] = s
        out["rows"].append(row(name, s, (seconds or {}).get(name, float("nan")), len(sids)))
    if baseline is not None:
        for name, preds in preds_by_condition.items():
            if name != baseline:
                out["paired"][f"{baseline} vs {name}"] = compare_full(
                    ref_by_sid, preds_by_condition[baseline], preds, sids)
    return out


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["synthetic"])
    ap.add_argument("--json", help="write the full result here")
    args = ap.parse_args()
    result = run_synthetic()
    print(result["notice"])
    print(f"trained on {result['n_train']} synthetic sentences, scored on {result['n_dev']}\n")
    print(format_table(result["rows"]))
    r = result["router"]
    print(f"\nrouter: {r['n_escalated']}/{r['n_sentences']} sentences escalated "
          f"({r['escalation_rate']:.0%});  by trigger: "
          + ", ".join(f"{k[0]}={v}" for k, v in r["by_trigger"].items()))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(_json_safe(result), indent=2, ensure_ascii=False,
                                              default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
