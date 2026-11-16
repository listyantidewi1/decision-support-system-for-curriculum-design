"""How does span-F1 scale with the number of annotated sentences?

The literature reviewed for this project reports NO learning curve for skill span
extraction — not one paper evaluates below ~3,700 training items. This measures the
curve on SkillSpan (Zhang et al., NAACL 2022), the closest public analogue, so the
annotation budget for the Indonesian five-tag gold standard can be chosen from a
measurement instead of a guess.

Design
  * Task: BIO span tagging with TWO entity types (SKILL, KNOWLEDGE) = 5 classes.
    SkillSpan ships two parallel BIO sequences, so overlapping spans are
    representable; where a token carries both, KNOWLEDGE wins (~3% of train spans
    overlap, so the choice is immaterial to the curve) and the count is reported.
  * Metric: EXACT span match on (start, end, type), scored with _prf imported from
    gold_standard.iaa — deliberately the same F1 implementation the IAA uses, so
    model scores and inter-annotator scores are on one scale.
  * Subsample sizes x seeds, fixed epochs, evaluated on SkillSpan's own test split.
  * Per-type F1 is reported separately: SKILL spans are longer and more abstract
    than KNOWLEDGE, so whether SKILL saturates later is the finding that transfers
    to the SH/SS/WB-vs-TT question.

Caveats to carry into any write-up: English, 2 types (not 5), distilbert rather
than a multilingual encoder. This estimates the SHAPE of the curve — where returns
flatten — not the absolute level attainable on Indonesian.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

sys.path.insert(0, r"D:\Projects\skill-extraction")
from gold_standard.iaa import _prf  # same F1 definition as the IAA report

MODEL = os.environ.get("LC_MODEL", "distilbert-base-uncased")
MAX_LEN = 128
BATCH = 16
LR = 5e-5
EPOCHS = int(os.environ.get("LC_EPOCHS", "8"))   # same budget at every size
SIZES = [180, 360, 720, 1440, 2880, 4830]
SEEDS = [13, 42]
EVAL_N = int(os.environ.get("LC_EVAL_N", "1500"))
TYPES = ["SKILL", "KNOWLEDGE"]
LABELS = ["O"] + [f"{p}-{t}" for t in TYPES for p in ("B", "I")]
L2I = {l: i for i, l in enumerate(LABELS)}


def load(split: str):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("jjzha/skillspan", f"{split}.json", repo_type="dataset")
    rows = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def merge_tags(r) -> tuple[list[str], int]:
    """One BIO sequence over two types; KNOWLEDGE wins a contested token."""
    sk, kn = r["tags_skill"], r["tags_knowledge"]
    out, clashes = [], 0
    for a, b in zip(sk, kn):
        if a != "O" and b != "O":
            clashes += 1
            out.append(f"{b[0]}-KNOWLEDGE")
        elif b != "O":
            out.append(f"{b[0]}-KNOWLEDGE")
        elif a != "O":
            out.append(f"{a[0]}-SKILL")
        else:
            out.append("O")
    return out, clashes


def spans_of(tags: list[str]) -> set:
    """Decode BIO into {(start, end_exclusive, type)}."""
    out, start, typ = set(), None, None
    for i, t in enumerate(tags):
        if t.startswith("B-"):
            if start is not None:
                out.add((start, i, typ))
            start, typ = i, t[2:]
        elif t.startswith("I-") and start is not None and t[2:] == typ:
            continue
        else:
            if start is not None:
                out.add((start, i, typ))
            start, typ = None, None
    if start is not None:
        out.add((start, len(tags), typ))
    return out


class DS(Dataset):
    def __init__(self, rows, tok):
        self.ex = []
        for r in rows:
            tags, _ = merge_tags(r)
            enc = tok(r["tokens"], is_split_into_words=True, truncation=True,
                      max_length=MAX_LEN)
            wids = enc.word_ids()
            lab, prev = [], None
            for w in wids:
                if w is None:
                    lab.append(-100)
                elif w != prev:
                    lab.append(L2I.get(tags[w], 0))
                else:
                    lab.append(-100)          # score first subword only
                prev = w
            self.ex.append({"input_ids": enc["input_ids"],
                            "attention_mask": enc["attention_mask"],
                            "labels": lab, "word_ids": wids, "gold": tags})

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    pad = lambda x, v: x + [v] * (n - len(x))
    return {
        "input_ids": torch.tensor([pad(b["input_ids"], 0) for b in batch]),
        "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
        "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
        "_raw": batch,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    agg = {t: [0, 0, 0] for t in TYPES}        # tp, n_gold, n_pred
    tot = [0, 0, 0]
    for b in loader:
        logits = model(input_ids=b["input_ids"].to(device),
                       attention_mask=b["attention_mask"].to(device)).logits
        pred = logits.argmax(-1).cpu().numpy()
        for k, raw in enumerate(b["_raw"]):
            wids, gold = raw["word_ids"], raw["gold"]
            ptags = ["O"] * len(gold)
            prev = None
            for pos, w in enumerate(wids):
                if w is not None and w != prev and w < len(ptags):
                    ptags[w] = LABELS[pred[k][pos]]
                prev = w
            g, p = spans_of(gold), spans_of(ptags)
            tot[0] += len(g & p); tot[1] += len(g); tot[2] += len(p)
            for t in TYPES:
                gt = {s for s in g if s[2] == t}
                pt = {s for s in p if s[2] == t}
                agg[t][0] += len(gt & pt); agg[t][1] += len(gt); agg[t][2] += len(pt)
    res = {"overall": _prf(tot[0], tot[1], tot[2])}
    for t in TYPES:
        res[t] = _prf(agg[t][0], agg[t][1], agg[t][2])
    return res


def run(train_rows, test_ds, size, seed, tok, device):
    random.Random(seed).shuffle(train_rows := list(train_rows))
    sub = train_rows[:size]
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL, num_labels=len(LABELS)).to(device)
    dl = DataLoader(DS(sub, tok), batch_size=BATCH, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    t0 = time.time()
    for _ in range(EPOCHS):
        for b in dl:
            opt.zero_grad()
            out = model(input_ids=b["input_ids"].to(device),
                        attention_mask=b["attention_mask"].to(device),
                        labels=b["labels"].to(device))
            out.loss.backward(); opt.step()
    secs = time.time() - t0
    res = evaluate(model, DataLoader(test_ds, batch_size=64, collate_fn=collate), device)
    del model
    return res, secs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(os.cpu_count() or 8)
    tok = AutoTokenizer.from_pretrained(MODEL)
    train = load("train")
    dev = load("dev")
    random.Random(7).shuffle(dev)
    test = dev[:EVAL_N]      # held-out DEV subsample; SkillSpan test stays sealed
    clash = sum(merge_tags(r)[1] for r in train)
    print(f"model={MODEL} device={device} epochs={EPOCHS}", flush=True)
    print(f"train={len(train)} eval(dev subsample)={len(test)} contested_tokens={clash}", flush=True)
    gold_spans = sum(len(spans_of(merge_tags(r)[0])) for r in train)
    print(f"train gold spans={gold_spans} ({gold_spans/len(train):.2f}/sentence)", flush=True)
    test_ds = DS(test, tok)
    out = []
    for size in SIZES:
        if size > len(train):
            continue
        for seed in SEEDS:
            res, secs = run(train, test_ds, size, seed, tok, device)
            row = {"size": size, "seed": seed, "secs": round(secs, 1),
                   "f1": res["overall"]["f1"],
                   "f1_skill": res["SKILL"]["f1"],
                   "f1_knowledge": res["KNOWLEDGE"]["f1"],
                   "n_gold": res["overall"]["n_gold"] if "n_gold" in res["overall"] else res["overall"]["n_a"]}
            out.append(row)
            print(f"  size={size:>5} seed={seed} f1={row['f1']:.4f} "
                  f"skill={row['f1_skill']:.4f} know={row['f1_knowledge']:.4f} "
                  f"({secs/60:.1f} min)", flush=True)
            Path(sys.argv[1]).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n=== LEARNING CURVE (mean over seeds) ===", flush=True)
    print(f"{'sentences':>10}{'overall F1':>12}{'SKILL':>9}{'KNOWLEDGE':>11}", flush=True)
    for size in SIZES:
        rs = [r for r in out if r["size"] == size]
        if rs:
            m = lambda k: float(np.mean([r[k] for r in rs]))
            print(f"{size:>10}{m('f1'):>12.4f}{m('f1_skill'):>9.4f}{m('f1_knowledge'):>11.4f}", flush=True)


if __name__ == "__main__":
    main()
