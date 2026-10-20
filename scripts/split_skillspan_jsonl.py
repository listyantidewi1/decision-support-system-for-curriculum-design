"""Split a SkillSpan-style JSONL file into train/dev/test by job_id.

Grouping by job_id prevents sentences from the same posting leaking across
training and evaluation splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def group_key(row: dict[str, Any]) -> str:
    key = str(row.get("job_id") or "").strip()
    if key:
        return key
    sid = str(row.get("sentence_id") or "").strip()
    if sid:
        return sid.rsplit("_", 1)[0]
    return hashlib.sha1(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for label, field in [
            ("hard_skill", "tags_hard_skill"),
            ("soft_skill", "tags_soft_skill"),
            ("knowledge", "tags_knowledge"),
        ]:
            counts[f"{label}_tokens"] += sum(1 for tag in row.get(field, []) if tag != "O")
            counts[f"{label}_spans"] += sum(1 for tag in row.get(field, []) if tag == "B")
    return dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total_ratio = args.train_ratio + args.dev_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print("[ERROR] train/dev/test ratios must sum to 1.0", file=sys.stderr)
        return 2
    input_path = _path(args.input_jsonl)
    output_dir = _path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(input_path)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)

    keys = list(groups)
    rng = random.Random(args.seed)
    rng.shuffle(keys)
    n = len(keys)
    n_train = int(round(n * args.train_ratio))
    n_dev = int(round(n * args.dev_ratio))
    split_keys = {
        "train": set(keys[:n_train]),
        "dev": set(keys[n_train : n_train + n_dev]),
        "test": set(keys[n_train + n_dev :]),
    }
    split_rows = {
        name: [row for key in keys if key in chosen for row in groups[key]]
        for name, chosen in split_keys.items()
    }
    for name, out_rows in split_rows.items():
        write_jsonl(output_dir / f"{name}.jsonl", out_rows)

    report = {
        "schema_version": "2026-06-23.skillspan_ours.split.v1",
        "input": str(input_path),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "n_records": len(rows),
        "n_groups": len(groups),
        "splits": {
            name: {
                "records": len(out_rows),
                "groups": len(split_keys[name]),
                **label_counts(out_rows),
            }
            for name, out_rows in split_rows.items()
        },
    }
    (output_dir / "split_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[INFO] wrote train/dev/test split to {output_dir}")
    print(f"[INFO] report: {output_dir / 'split_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

