"""Domain-adaptive pretraining for E2: continue masked-language-model training of
the encoder on UNLABELLED job-posting sentences, then fine-tune it exactly as E1.

The plan calls E2 "a condition, not an assumption": whether adaptation helps is
what the E1-versus-E2 comparison measures. So this script changes one thing —
the checkpoint E2 starts from — and nothing else.

    python -m fivetag.dapt --sentences corpus_sentences.txt --out runs/dapt-xlmr \\
                           --model xlm-roberta-base --epochs 3
    python -m fivetag.encoder train --model runs/dapt-xlmr --condition E2 ...

`--sentences` is a text file, one sentence per line. Leakage rule: it must NOT
contain sentences of the sealed test split. Pass their file with
`--exclude` and they are removed (exact text match) before training; the count
removed is written to the log.

Meant for a GPU. On the CPU it is only good for the smoke test.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path


def load_sentences(path: Path, exclude: Path | None = None) -> tuple[list[str], int]:
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    removed = 0
    if exclude is not None:
        banned = {ln.strip() for ln in Path(exclude).read_text(encoding="utf-8").splitlines()}
        kept = [ln for ln in lines if ln not in banned]
        removed, lines = len(lines) - len(kept), kept
    return lines, removed


def run_dapt(sentences: list[str], out_dir: Path, *, model_name: str = "xlm-roberta-base",
             epochs: int = 3, batch_size: int = 16, lr: float = 5e-5, max_length: int = 128,
             mlm_probability: float = 0.15, seed: int = 42, tokenizer=None, model=None,
             device: str | None = None, extra_log: dict | None = None, log=print) -> dict:
    import torch
    from transformers import DataCollatorForLanguageModeling
    random.seed(seed)
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if tokenizer is None or model is None:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        model = model or AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)
    collate = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=mlm_probability)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    order, rng, history = list(sentences), random.Random(seed), []
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(order)
        total, n = 0.0, 0
        for i in range(0, len(order), batch_size):
            enc = tokenizer(order[i:i + batch_size], truncation=True, max_length=max_length)
            batch = collate([{"input_ids": ids} for ids in enc["input_ids"]])
            out = model(**{k: v.to(device) for k, v in batch.items()})
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            total, n = total + out.loss.item(), n + 1
        history.append({"epoch": epoch, "mlm_loss": total / max(n, 1)})
        log(f"epoch {epoch}  masked-LM loss {total / max(n, 1):.4f}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    record = {"base_model": model_name, "n_sentences": len(sentences), "epochs": epochs,
              "batch_size": batch_size, "lr": lr, "mlm_probability": mlm_probability,
              "seed": seed, "device": device, "seconds": time.perf_counter() - t0,
              "history": history, **(extra_log or {})}
    (out_dir / "dapt_log.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--exclude", help="file of sentences that must not be trained on (test split)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    sentences, removed = load_sentences(Path(args.sentences),
                                        Path(args.exclude) if args.exclude else None)
    print(f"{len(sentences)} sentences ({removed} removed by --exclude)")
    run_dapt(sentences, Path(args.out), model_name=args.model, epochs=args.epochs,
             batch_size=args.batch_size, lr=args.lr, seed=args.seed,
             extra_log={"excluded_test_sentences": removed,
                        "exclude_file_given": bool(args.exclude)})


if __name__ == "__main__":
    main()
