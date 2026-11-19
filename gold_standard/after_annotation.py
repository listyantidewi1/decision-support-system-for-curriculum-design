"""One command for the day the annotators finish.

    .venv-ml\\Scripts\\python.exe -m gold_standard.after_annotation --split pilot_100

It runs, in order, everything that turns three annotators' saved work into the
material the researcher needs next:

  1. the agreement report            (gold_standard.iaa)
  2. the go/no-go check              (gold_standard.gate)
  3. the calibration report          (gold_standard.calibration) - a web page
  4. the adjudication worksheet      (gold_standard.adjudicate)  - a spreadsheet

and then says, in plain words, what to do with them.

By default it reads the annotation files from DATA/gold_standard/annotations/
(where the collector's export is unzipped) and writes beside the other evidence
in DATA/gold_standard/iaa/. For a rehearsal on synthetic annotators, give
--ann-dir and --out-dir; step 1 refuses to mix synthetic input into the evidence
directory, and the later steps write next to whatever step 1 produced.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import adjudicate, calibration, gate

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(split: str, ann_dir: Path | None = None, out_dir: Path | None = None,
        log=print) -> dict:
    from .iaa import OUT_DIR
    target = Path(out_dir) if out_dir else OUT_DIR
    before = set(target.glob(f"iaa_{split}_*.json")) if target.is_dir() else set()

    cmd = [sys.executable, "-m", "gold_standard.iaa", "--split", split]
    if ann_dir:
        cmd += ["--ann-dir", str(ann_dir)]
    if out_dir:
        cmd += ["--out-dir", str(out_dir)]
    log("STEP 1 of 4 - agreement report")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise SystemExit("the agreement step stopped:\n" + (proc.stderr or proc.stdout).strip())
    new = sorted(set(target.glob(f"iaa_{split}_*.json")) - before)
    if not new:
        raise SystemExit(f"the agreement step wrote no report into {target}")
    report_path = new[-1]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    log(f"   -> {report_path}")

    log("STEP 2 of 4 - go/no-go check")
    from .iaa import _text_by_sid
    n_expected = len(_text_by_sid(split)) or None
    result = gate.evaluate(report, n_expected)
    gate_path = report_path.with_name(report_path.name.replace("iaa_", "gate_", 1))
    gate_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log(gate.to_text(result))

    log("\nSTEP 3 of 4 - calibration report")
    cal_path = calibration.write(report_path)
    log(f"   -> {cal_path}")

    log("STEP 4 of 4 - adjudication worksheet")
    stamp = report.get("created_utc", "")
    sheet = report_path.with_name(f"adjudication_{split}_{stamp}_WORKSHEET.xlsx")
    annset, text, resolved, quarantined = adjudicate._load(split, ann_dir)
    adjudicate._refuse_evidence_tree(quarantined, sheet)
    info = adjudicate.make_worksheet(annset, text, sheet, split)
    log(f"   -> {sheet}  ({info['n_sentence_rows']} sentence rows, "
        f"{info['n_span_rows']} span rows)")

    log("\nWHAT TO DO NEXT")
    if result["input_is_synthetic"]:
        log("  This was a rehearsal on synthetic data. Nothing here is a result.")
    log(f"  1. Read the verdict above: {result['verdict']}.")
    log("  2. Open the calibration report (the .html file) in a browser and use it as "
        "the agenda for the session with the three annotators.")
    log("  3. During adjudication, fill in the worksheet (the .xlsx file).")
    log("  4. Then run:  python -m gold_standard.adjudicate finalise --split "
        f"{split} --worksheet <the worksheet> --version 1.0")
    return {"report": str(report_path), "gate": str(gate_path), "calibration": str(cal_path),
            "worksheet": str(sheet), "verdict": result["verdict"]}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True)
    ap.add_argument("--ann-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    run(args.split, Path(args.ann_dir) if args.ann_dir else None,
        Path(args.out_dir) if args.out_dir else None)


if __name__ == "__main__":
    main()
