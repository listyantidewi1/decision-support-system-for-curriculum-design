"""Fill the two gaps in the learning curve: the 4,800 top point and a third
seed at the sizes where the plateau claim is contested (1440, 2880).

Run from an EMPTY directory so the scratchpad inspect.py cannot shadow stdlib.
"""
import json, sys
from pathlib import Path

import lc_base as L

# (size, seed) cells still missing. 4800 = full SkillSpan train (the earlier run
# asked for 4830 and the `size > len(train)` guard silently dropped it).
CELLS = [
    (4800, 13), (4800, 42),
    (1440, 7), (2880, 7), (4800, 7),
]

def main():
    import torch, random
    from transformers import AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import os
    torch.set_num_threads(os.cpu_count() or 8)
    tok = AutoTokenizer.from_pretrained(L.MODEL)
    train = L.load("train")
    dev = L.load("dev")
    random.Random(7).shuffle(dev)
    test = dev[:L.EVAL_N]
    print(f"model={L.MODEL} device={device} epochs={L.EPOCHS}", flush=True)
    print(f"train={len(train)} eval={len(test)}", flush=True)
    test_ds = L.DS(test, tok)
    out = []
    for size, seed in CELLS:
        size = min(size, len(train))
        res, secs = L.run(train, test_ds, size, seed, tok, device)
        row = {"size": size, "seed": seed, "secs": round(secs, 1),
               "f1": res["overall"]["f1"],
               "f1_skill": res["SKILL"]["f1"],
               "f1_knowledge": res["KNOWLEDGE"]["f1"]}
        out.append(row)
        print(f"  size={size:>5} seed={seed} f1={row['f1']:.4f} "
              f"skill={row['f1_skill']:.4f} know={row['f1_knowledge']:.4f} "
              f"({secs/60:.1f} min)", flush=True)
        Path(sys.argv[1]).write_text(json.dumps(out, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
