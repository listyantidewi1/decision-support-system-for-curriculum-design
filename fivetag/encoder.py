"""E1 / E2 — multilingual encoder + BIO sequence labelling.

E1 is the main supervised baseline: a pretrained multilingual encoder
(XLM-RoBERTa by default) fine-tuned to tag tokens. E2 is the SAME code started
from a checkpoint that was first adapted to job-posting language
(`fivetag.dapt`); the only difference between the two conditions is the
checkpoint, which is what makes the comparison a test of adaptation.

Layered BIO. The guideline lets a technology sit inside an action phrase
("Configure Kubernetes clusters": SH, with "Kubernetes" TT inside — rule 6.9).
One flat BIO sequence cannot represent that. So each of the five tags gets its
own B/I/O layer over the same encoder: five small heads, one shared body.
Within a tag spans never overlap (rule 6.4), so each layer is ordinary BIO.

Sizes this module is built for: training on a GPU (Kaggle) with the real gold
standard; the CPU is enough only for the smoke test in `tests/test_fivetag.py`,
which trains a tiny randomly-initialised encoder on synthetic sentences to
prove the loop learns.

Training rows (JSON lines; `fivetag.synthetic.read_jsonl` reads them):

    {"sentence_id": "...", "text": "...", "spans": [{"start": 0, "end": 6, "label": "TT"}]}

    python -m fivetag.encoder train --train train.jsonl --dev dev.jsonl --out runs/e1
    python -m fivetag.encoder train ... --model path/to/dapt-checkpoint --condition E2

The sealed test split is never an input here: a row marked `"split": "test"`
is refused.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from . import FIVETAG_VERSION, TAG_CODES
from .base import Pred, dedupe, make_pred

O, B, I = 0, 1, 2
IGNORE = -100
N_TAGS = len(TAG_CODES)


@dataclass
class EncoderConfig:
    model_name: str = "xlm-roberta-base"
    condition: str = "E1"                 # E1 | E2
    max_length: int = 128
    epochs: int = 10
    batch_size: int = 16
    lr: float = 3e-5                      # encoder body
    head_lr: float = 1e-3                 # the five BIO heads
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout: float = 0.1
    seed: int = 42
    scheme: str = "layered-BIO"


def check_rows(rows: Sequence[dict]) -> None:
    for r in rows:
        if r.get("split") == "test":
            raise ValueError(f"{r.get('sentence_id')}: a sealed test row was given to the "
                             f"encoder; the test split is opened once, for the final "
                             f"configuration only")
        n = len(r["text"])
        for s in r.get("spans", ()):
            if s["label"] not in TAG_CODES or not (0 <= s["start"] < s["end"] <= n):
                raise ValueError(f"{r.get('sentence_id')}: bad span {s}")


# ------------------------------------------------------------------ BIO coding
def encode_labels(offsets: Sequence[tuple[int, int]], special: Sequence[int],
                  spans: Sequence[dict]) -> list[list[int]]:
    """Token labels, shape [n_tokens][5]. A token is in a span when their
    character ranges overlap; the first such token is B, the rest I."""
    labels = [[IGNORE] * N_TAGS if sp else [O] * N_TAGS for sp in special]
    for s in sorted(spans, key=lambda s: (s["start"], s["end"])):
        layer, first = TAG_CODES.index(s["label"]), True
        for t, (a, b) in enumerate(offsets):
            if special[t] or a == b:
                continue
            if a < s["end"] and b > s["start"]:
                labels[t][layer] = B if first else I
                first = False
    return labels


def decode_labels(sentence: str, offsets: Sequence[tuple[int, int]], special: Sequence[int],
                  tags: Sequence[Sequence[int]], conf: Sequence[Sequence[float]] | None = None,
                  source: str = "E1") -> list[Pred]:
    """Token labels -> spans. An I with no B before it starts a span (the usual
    lenient repair); the score is the mean token confidence of the span."""
    out = []
    for layer, tag in enumerate(TAG_CODES):
        cur: list[int] = []

        def flush():
            if not cur:
                return
            a, b = offsets[cur[0]][0], offsets[cur[-1]][1]
            while a < b and sentence[a].isspace():
                a += 1
            while b > a and sentence[b - 1].isspace():
                b -= 1
            if a < b:
                sc = (sum(conf[t][layer] for t in cur) / len(cur)) if conf is not None else 1.0
                out.append(make_pred(sentence, a, b, tag, score=float(sc), source=source,
                                     rule="bio"))
            cur.clear()

        for t in range(len(offsets)):
            if special[t] or offsets[t][0] == offsets[t][1]:
                continue
            v = tags[t][layer]
            if v == B:
                flush()
                cur.append(t)
            elif v == I:
                cur.append(t)
            else:
                flush()
        flush()
    return dedupe(out)


# ----------------------------------------------------------------------- model
def _torch():
    import torch
    return torch


def build_model(encoder, dropout: float):
    torch = _torch()

    class LayeredBio(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.dropout = torch.nn.Dropout(dropout)
            self.head = torch.nn.Linear(encoder.config.hidden_size, N_TAGS * 3)

        def forward(self, input_ids, attention_mask):
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            return self.head(self.dropout(h)).view(h.shape[0], h.shape[1], N_TAGS, 3)

    return LayeredBio()


def _batches(rows, tokenizer, cfg: EncoderConfig, with_labels: bool):
    torch = _torch()
    for i in range(0, len(rows), cfg.batch_size):
        chunk = rows[i:i + cfg.batch_size]
        enc = tokenizer([r["text"] for r in chunk], truncation=True, max_length=cfg.max_length,
                        padding=True, return_offsets_mapping=True,
                        return_special_tokens_mask=True)
        item = {"rows": chunk, "offsets": enc["offset_mapping"],
                "special": [[int(s or not m) for s, m in zip(sp, am)]
                            for sp, am in zip(enc["special_tokens_mask"], enc["attention_mask"])],
                "input_ids": torch.tensor(enc["input_ids"]),
                "attention_mask": torch.tensor(enc["attention_mask"])}
        if with_labels:
            item["labels"] = torch.tensor([
                encode_labels(off, sp, r.get("spans", ()))
                for off, sp, r in zip(item["offsets"], item["special"], chunk)])
        yield item


def predict_rows(model, tokenizer, rows: Sequence[dict], cfg: EncoderConfig,
                 device: str = "cpu") -> dict[str, list[Pred]]:
    torch = _torch()
    model.eval()
    out: dict[str, list[Pred]] = {}
    with torch.no_grad():
        for b in _batches(list(rows), tokenizer, cfg, with_labels=False):
            logits = model(b["input_ids"].to(device), b["attention_mask"].to(device))
            prob = torch.softmax(logits, dim=-1)
            conf, tags = prob.max(dim=-1)
            for k, r in enumerate(b["rows"]):
                out[r["sentence_id"]] = decode_labels(
                    r["text"], b["offsets"][k], b["special"][k], tags[k].tolist(),
                    conf[k].tolist(), source=cfg.condition)
    return out


def _dev_f1(model, tokenizer, rows, cfg, device) -> dict:
    from .evaluate import score_full
    from .synthetic import as_reference
    _, ref = as_reference(list(rows))
    s = score_full(ref, predict_rows(model, tokenizer, rows, cfg, device), bootstrap=False)
    return {"strict_f1": s["strict"]["f1"], "partial_f1": s["partial"]["f1"],
            "strict_macro_f1": s["strict"]["per_tag"]["macro_f1"]}


def train(train_rows: Sequence[dict], dev_rows: Sequence[dict], cfg: EncoderConfig,
          out_dir: Path | None = None, *, tokenizer=None, encoder=None, device: str | None = None,
          log=print) -> dict:
    """Fine-tune; keep the epoch with the best strict span-F1 on dev.

    Returns {"model", "tokenizer", "log"}. `tokenizer` / `encoder` may be passed
    in (the smoke test does, to avoid any download); otherwise they are loaded
    from `cfg.model_name`.
    """
    torch = _torch()
    from transformers import get_linear_schedule_with_warmup
    check_rows(train_rows)
    check_rows(dev_rows)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if tokenizer is None or encoder is None:
        from transformers import AutoModel, AutoTokenizer
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(cfg.model_name)
        encoder = encoder or AutoModel.from_pretrained(cfg.model_name)
    model = build_model(encoder, cfg.dropout).to(device)
    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": cfg.lr},
        {"params": model.head.parameters(), "lr": cfg.head_lr}], weight_decay=cfg.weight_decay)
    steps = max(1, cfg.epochs * ((len(train_rows) + cfg.batch_size - 1) // cfg.batch_size))
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * steps), steps)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
    order = list(train_rows)
    rng = random.Random(cfg.seed)
    history, best, best_state = [], -1.0, None
    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        rng.shuffle(order)
        total, n = 0.0, 0
        for b in _batches(order, tokenizer, cfg, with_labels=True):
            logits = model(b["input_ids"].to(device), b["attention_mask"].to(device))
            loss = loss_fn(logits.reshape(-1, 3), b["labels"].to(device).reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total, n = total + loss.item(), n + 1
        dev = _dev_f1(model, tokenizer, dev_rows, cfg, device) if dev_rows else {}
        history.append({"epoch": epoch, "train_loss": total / max(n, 1), **dev})
        f1 = dev.get("strict_f1", float("nan"))
        key = f1 if f1 == f1 else -total            # no dev: keep the last epoch
        if key > best or best_state is None:
            best, best_state = key, {k: v.detach().cpu().clone()
                                     for k, v in model.state_dict().items()}
        log(f"epoch {epoch:>2}  loss {total / max(n, 1):.4f}  dev strict-F1 {f1:.3f}")
    model.load_state_dict(best_state)
    record = {"fivetag_version": FIVETAG_VERSION, "config": asdict(cfg),
              "n_train": len(train_rows), "n_dev": len(dev_rows), "device": device,
              "seconds": time.perf_counter() - t0, "history": history,
              "best_dev_strict_f1": best if best >= 0 else None,
              "synthetic_training_data": bool(train_rows) and all(
                  r.get("synthetic") for r in train_rows)}
    if out_dir is not None:
        save(model, tokenizer, cfg, record, Path(out_dir))
    return {"model": model, "tokenizer": tokenizer, "log": record}


def save(model, tokenizer, cfg: EncoderConfig, record: dict, out_dir: Path) -> None:
    torch = _torch()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(out_dir / "encoder")
    tokenizer.save_pretrained(out_dir / "tokenizer")
    torch.save(model.head.state_dict(), out_dir / "head.pt")
    (out_dir / "training_log.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


class EncoderExtractor:
    """A trained E1 / E2 model behind the common `Extractor` interface."""

    def __init__(self, model, tokenizer, cfg: EncoderConfig, record: dict | None = None,
                 device: str = "cpu"):
        self.model, self.tokenizer, self.cfg = model.to(device), tokenizer, cfg
        self.record, self.device, self.name = record or {}, device, cfg.condition

    @classmethod
    def load(cls, model_dir: Path, device: str = "cpu") -> "EncoderExtractor":
        torch = _torch()
        from transformers import AutoModel, AutoTokenizer
        model_dir = Path(model_dir)
        record = json.loads((model_dir / "training_log.json").read_text(encoding="utf-8"))
        cfg = EncoderConfig(**record["config"])
        model = build_model(AutoModel.from_pretrained(model_dir / "encoder"), cfg.dropout)
        model.head.load_state_dict(torch.load(model_dir / "head.pt", map_location=device))
        return cls(model, AutoTokenizer.from_pretrained(model_dir / "tokenizer"), cfg, record,
                   device)

    def extract(self, sentence: str) -> list[Pred]:
        if not sentence or not sentence.strip():
            return []
        return predict_rows(self.model, self.tokenizer,
                            [{"sentence_id": "_", "text": sentence}], self.cfg, self.device)["_"]

    def extract_many(self, sentences: dict[str, str]) -> dict[str, list[Pred]]:
        rows = [{"sentence_id": k, "text": v} for k, v in sentences.items()]
        return predict_rows(self.model, self.tokenizer, rows, self.cfg, self.device)

    def manifest(self) -> dict:
        return {"extractor": self.name, "fivetag_version": FIVETAG_VERSION,
                "config": asdict(self.cfg),
                "best_dev_strict_f1": self.record.get("best_dev_strict_f1"),
                "synthetic_training_data": self.record.get("synthetic_training_data")}


def main() -> None:
    from .synthetic import read_jsonl
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    tp = sub.add_parser("train")
    tp.add_argument("--train", required=True)
    tp.add_argument("--dev", required=True)
    tp.add_argument("--out", required=True)
    tp.add_argument("--model", default=EncoderConfig.model_name)
    tp.add_argument("--condition", choices=["E1", "E2"], default="E1")
    tp.add_argument("--epochs", type=int, default=EncoderConfig.epochs)
    tp.add_argument("--batch-size", type=int, default=EncoderConfig.batch_size)
    tp.add_argument("--lr", type=float, default=EncoderConfig.lr)
    tp.add_argument("--seed", type=int, default=EncoderConfig.seed)
    pp = sub.add_parser("predict")
    pp.add_argument("--model-dir", required=True)
    pp.add_argument("--input", required=True, help="JSON lines with sentence_id and text")
    pp.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "train":
        cfg = EncoderConfig(model_name=args.model, condition=args.condition, epochs=args.epochs,
                            batch_size=args.batch_size, lr=args.lr, seed=args.seed)
        result = train(read_jsonl(args.train), read_jsonl(args.dev), cfg, Path(args.out))
        print(f"best dev strict-F1: {result['log']['best_dev_strict_f1']}  ->  {args.out}")
    else:
        ex = EncoderExtractor.load(Path(args.model_dir))
        rows = read_jsonl(args.input)
        preds = ex.extract_many({r["sentence_id"]: r["text"] for r in rows})
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            for sid, ps in preds.items():
                fh.write(json.dumps({"sentence_id": sid, "spans": [p.to_dict() for p in ps]},
                                    ensure_ascii=False) + "\n")
        print(f"{len(preds)} sentences -> {args.out}")


if __name__ == "__main__":
    main()
