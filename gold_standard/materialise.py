"""After adjudication: seal the test labels and hand train/dev to the models — one command.

The split itself (`gold_standard.split`) is computed from metadata BEFORE any
label exists and contains no annotation. Once the experts' gold is final
(`gold_standard.adjudicate finalise`), two things remain, and this does both:

  1. Test part  -> `seal.materialise_labels` writes the sealed test labels. From
     then on they are opened only through `seal.score_on_test`, once, for the
     chosen configuration.
  2. Train and dev parts -> written as JSON lines for the models
     (`fivetag.encoder`, the E5 retrieval pool). The test part is NEVER
     exported: this command fails if a test sentence would reach the export.

    python -m gold_standard.materialise --split-id grouped_random_v1 ^
        --gold-dir DATA/gold_standard/gold/main_1200/v1

Output: `DATA/fivetag/gold/<split id>/<gold version>/{train,dev}.jsonl` and a
MANIFEST.json with the counts and file hashes. `DATA/` is gitignored; the rows
contain corpus sentences and stay local.

Only spans with verdict GOLD are exported as labels. A train/dev sentence that
also has an AMBIGUOUS or EXCLUDE span is kept and flagged
(`n_challenge_spans`), not dropped: whether to train on it is the experiment's
decision, and it should be a visible one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from . import TAGS
from . import seal as seal_mod

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_ROOT = REPO_ROOT / "DATA" / "fivetag" / "gold"
GOLD_VERDICT = "GOLD"


class MaterialiseError(RuntimeError):
    """The gold and the split do not fit together; nothing was written."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _part(split_dir: Path, name: str) -> pd.DataFrame:
    path = split_dir / f"{name}.parquet"
    if not path.is_file():
        raise MaterialiseError(f"{path} is missing: run gold_standard.split first")
    return pd.read_parquet(path)


def build_rows(part: pd.DataFrame, part_name: str, gold: pd.DataFrame,
               text_by_sid: dict[str, str]) -> list[dict]:
    by_sid: dict[str, list] = {}
    for r in gold.itertuples(index=False):
        by_sid.setdefault(str(r.sentence_id), []).append(r)
    rows = []
    for sid in part["sentence_id"].astype(str):
        if sid not in text_by_sid:
            raise MaterialiseError(f"no sentence text for {sid}")
        text, spans, challenge = text_by_sid[sid], [], 0
        for r in by_sid.get(sid, ()):
            a, b, tag = int(r.span_start), int(r.span_end), str(r.tag)
            if str(r.verdict).upper() != GOLD_VERDICT:
                challenge += 1
                continue
            if tag not in TAGS or not (0 <= a < b <= len(text)):
                raise MaterialiseError(f"{sid}: gold span {a}:{b} {tag} does not fit its "
                                       f"sentence ({len(text)} chars)")
            spans.append({"start": a, "end": b, "label": tag})
        spans = sorted({(s["start"], s["end"], s["label"]) for s in spans})
        rows.append({"sentence_id": sid, "text": text, "split": part_name,
                     "spans": [{"start": a, "end": b, "label": t} for a, b, t in spans],
                     "n_challenge_spans": challenge})
    return rows


def materialise(split_id: str, gold_dir: Path, *, splits_dir: Path = seal_mod.SPLITS_DIR,
                export_root: Path = EXPORT_ROOT, text_by_sid: dict[str, str] | None = None,
                gold_version: str | None = None) -> dict:
    gold_dir = Path(gold_dir)
    gold_path = gold_dir / "gold_spans.parquet"
    if not gold_path.is_file():
        raise MaterialiseError(f"{gold_path} not found: finalise the adjudication first")
    gold = pd.read_parquet(gold_path)
    missing = [c for c in seal_mod.LABEL_COLUMNS if c not in gold.columns]
    if missing:
        raise MaterialiseError(f"gold is missing columns {missing}")
    from .iaa import _quarantine_marker
    synthetic = bool(_quarantine_marker(gold_dir.resolve()))
    version = gold_version or gold_dir.name

    split_dir = seal_mod._split_dir(split_id, splits_dir)
    parts = {name: _part(split_dir, name) for name in ("train", "dev", "test_blind")}
    ids = {name: set(df["sentence_id"].astype(str)) for name, df in parts.items()}
    for a, b in (("train", "dev"), ("train", "test_blind"), ("dev", "test_blind")):
        both = ids[a] & ids[b]
        if both:
            raise MaterialiseError(f"{len(both)} sentences are in both {a} and {b}")
    everything = ids["train"] | ids["dev"] | ids["test_blind"]
    foreign = sorted(set(gold["sentence_id"].astype(str)) - everything)
    if foreign:
        raise MaterialiseError(
            f"{len(foreign)} gold sentences are not in split {split_id} (first: "
            f"{foreign[:3]}). The gold and the split were made from different samples.")

    text = dict(text_by_sid or {})
    for name in ("train", "dev"):
        if "sentence_text" in parts[name].columns:
            text.update(zip(parts[name]["sentence_id"].astype(str),
                            parts[name]["sentence_text"].astype(str)))
    rows = {name: build_rows(parts[name], name, gold, text) for name in ("train", "dev")}
    leaked = [r["sentence_id"] for name in rows for r in rows[name]
              if r["sentence_id"] in ids["test_blind"]]
    if leaked:
        raise MaterialiseError(f"test sentences reached the export: {leaked[:3]}")

    out_dir = Path(export_root) / split_id / version
    if (REPO_ROOT / "DATA" / "gold_standard").resolve() in out_dir.resolve().parents:
        raise MaterialiseError("the model-facing export must not live under DATA/gold_standard/")

    # ---- all checks passed: now write. Test labels first, through the seal.
    seal_info = seal_mod.materialise_labels(split_id, gold, splits_dir=splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("train", "dev"):
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for r in rows[name]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        files[path.name] = _sha256(path)
    manifest = {
        "split_id": split_id, "gold_version": version, "gold_dir": str(gold_dir),
        "gold_spans_sha256": _sha256(gold_path), "synthetic_gold": synthetic,
        "n_sentences": {"train": len(rows["train"]), "dev": len(rows["dev"]),
                        "test_sealed": len(ids["test_blind"])},
        "n_gold_spans": {name: sum(len(r["spans"]) for r in rows[name])
                         for name in ("train", "dev")},
        "n_sentences_with_challenge_spans": {
            name: sum(1 for r in rows[name] if r["n_challenge_spans"])
            for name in ("train", "dev")},
        "test_exported": False,
        "test_labels": {k: seal_info.get(k) for k in (
            "n_spans", "n_challenge_spans", "n_gold_bearing_sentences",
            "sealed_test_n_sentences") if isinstance(seal_info, dict)},
        "files": files,
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    manifest["out_dir"] = str(out_dir)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-id", required=True, help="e.g. grouped_random_v1")
    ap.add_argument("--gold-dir", required=True,
                    help="folder written by gold_standard.adjudicate finalise")
    ap.add_argument("--sample", default="main_1200",
                    help="sample whose parquet supplies sentence text if the split lacks it")
    args = ap.parse_args()
    text = {}
    sample_path = seal_mod.GS_DIR / f"{args.sample}.parquet"
    if sample_path.is_file():
        df = pd.read_parquet(sample_path, columns=["sentence_id", "sentence_text"])
        text = dict(zip(df["sentence_id"].astype(str), df["sentence_text"].astype(str)))
    try:
        m = materialise(args.split_id, Path(args.gold_dir), text_by_sid=text)
    except (MaterialiseError, seal_mod.SealViolation) as exc:
        raise SystemExit(f"REFUSED: {exc}")
    n = m["n_sentences"]
    print(f"Test labels sealed for {n['test_sealed']} sentences (not exported).")
    print(f"Train: {n['train']} sentences, {m['n_gold_spans']['train']} gold spans")
    print(f"Dev:   {n['dev']} sentences, {m['n_gold_spans']['dev']} gold spans")
    if m["synthetic_gold"]:
        print("NOTE: this gold is SYNTHETIC (rehearsal). Not a result.")
    print(f"-> {m['out_dir']}")


if __name__ == "__main__":
    main()
