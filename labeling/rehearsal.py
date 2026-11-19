"""Three LLMs as stand-in annotators, to REHEARSE what happens after annotation.

    .venv-ml\\Scripts\\python.exe -m labeling.rehearsal --run-dir DATA\\labeling\\runs\\pilot_100_<id> --split pilot_100

When the three experts finish the pilot, their work has to go through the
agreement report, the calibration report, the go/no-go check, adjudication and
gold finalisation. That chain should not meet real data for the first time on
the day it matters. This module turns one ensemble run into three annotator
files in exactly the format the annotation page saves, so the whole chain can be
run end to end beforehand on the real pilot sentences.

EVERYTHING THIS WRITES IS SYNTHETIC. No expert judged it:

  * the "annotators" AV / AI / AD are three language models;
  * relevance and has_5tag are not model judgements at all — the models are only
    asked for spans, so those two fields are filled by a fixed rule (a span means
    RELEVANT + YA; no span means TIDAK and a seeded coin decides RELEVANT or
    NON_RELEVANT) purely so the sentence-level code paths get exercised;
  * confidence and flags are constants.

Agreement numbers computed from these files describe three models and a coin.
They are not reliability evidence and must never be reported. The files are
written under DATA/gold_standard/_synthetic/, which `gold_standard.iaa` refuses
to read unless its output is redirected away from the evidence directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from gold_standard import ANNOTATORS

from .verify import verify_output

REPO_ROOT = Path(__file__).resolve().parents[1]
REHEARSAL_ROOT = REPO_ROOT / "DATA" / "gold_standard" / "_synthetic" / "rehearsal"
SEED = 20261119
MARKER = "SYNTHETIC_DO_NOT_CITE.md"


def _coin(seed: int, model: str, sid: str, percent: int) -> bool:
    h = hashlib.sha256(f"{seed}|{model}|{sid}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 100 < percent


def simulate(run_dir: Path, sentences: list[dict], seed: int = SEED) -> dict[str, dict]:
    """{annotator code: {sentence id: record}} from one ensemble run."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    models = summary["run_config"]["models"]
    sha = summary["request_sha256"]
    codes = list(ANNOTATORS)
    if len(models) != len(codes):
        raise ValueError(f"need exactly {len(codes)} models to stand in for "
                         f"{codes}; the run has {len(models)}")
    raw = {}
    for line in (run_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("request_sha256") == sha:
                raw[(r["model"], r["sentence_id"])] = r
    out: dict[str, dict] = {}
    for code, model in zip(codes, models):
        recs = {"_meta": {"SIMULATED": True, "stands_in_for": code, "model": model,
                          "prompt_version": summary["prompt"]["prompt_version"],
                          "run": run_dir.name,
                          "warning": "synthetic rehearsal data - not an expert's work"}}
        for s in sentences:
            sid, text = s["sentence_id"], s["sentence_text"]
            rec = raw.get((model, sid))
            spans, note = [], "SIMULATED"
            if rec and rec.get("ok"):
                vo = verify_output(model, sid, text, rec.get("raw") or "")
                if vo.status != "VALID":
                    note = "SIMULATED (the model's answer was invalid; treated as no span)"
                spans = [{"span_text": v.text, "start_char": v.start, "end_char": v.end,
                          "label": v.label, "confidence": 3, "flag": "",
                          "parent_span_id": "", "note": ""} for v in vo.spans]
            else:
                note = "SIMULATED (the model returned nothing; treated as no span)"
            if spans:
                relevance, has = "RELEVANT", "YA"
            else:
                has = "TIDAK"
                relevance = "NON_RELEVANT" if _coin(seed, model, sid, 60) else "RELEVANT"
            recs[sid] = {"relevance": relevance, "has_5tag": has, "spans": spans,
                         "flag": "", "note": note}
        out[code] = recs
    return out


def write(run_dir: Path, split: str, sentences: list[dict],
          out_dir: Path | None = None, seed: int = SEED) -> Path:
    out_dir = Path(out_dir) if out_dir else REHEARSAL_ROOT / split / "annotations"
    resolved = out_dir.resolve()
    if "_synthetic" not in resolved.parts:
        raise ValueError(
            f"refusing to write simulated annotators to {resolved}: the directory must "
            f"sit under a '_synthetic' folder, so the agreement code can never mistake "
            f"it for expert work")
    out_dir.mkdir(parents=True, exist_ok=True)
    sim = simulate(run_dir, sentences, seed)
    for code, recs in sim.items():
        (out_dir / f"{split}_{code}.json").write_text(
            json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (out_dir / MARKER).write_text(
        "# Synthetic rehearsal data - not evidence, not citable\n\n"
        f"Written {stamp} by `labeling.rehearsal` from the ensemble run "
        f"`{Path(run_dir).name}`.\n\n"
        "The three 'annotators' here are language models, and their relevance and\n"
        "has_5tag values come from a fixed rule and a seeded coin. Nothing in this\n"
        "directory was judged by an expert. It exists only so the steps that follow\n"
        "annotation can be rehearsed before the real pilot data arrives.\n",
        encoding="utf-8")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    from .run import load_split
    out = write(Path(args.run_dir), args.split, load_split(args.split), args.out_dir)
    print(f"simulated annotators AV / AI / AD written to {out}")
    print("SYNTHETIC - rehearsal only. Run the agreement step with --out-dir pointing "
          "somewhere disposable.")


if __name__ == "__main__":
    main()
