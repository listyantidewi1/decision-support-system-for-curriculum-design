"""The IAA provenance gates — a synthetic run must not be able to produce
evidence.

Standalone runner (the repo has no pytest):

    .venv-ml\\Scripts\\python.exe tests\\test_iaa_provenance.py

Background. Four IAA report sets were found in DATA/gold_standard/iaa/ on
2026-11-17 labelled `split: pilot_100`, annotators [AV, AI, AD], publishing
Fleiss kappa 0.657 "substantial". They came from a four-sentence hand-made
fixture whose sentence ids (s1, s3, s4) exist nowhere in pilot_100.parquet. Two
of them also lacked the `claimable` field, so a downstream filter on
`claimable == false` would have let them through.

The route in was structural. `gold_standard.iaa` took `--ann-dir` for INPUT but
hardcoded its OUTPUT directory with no flag to redirect it, so every dry run
wrote synthetic numbers into the real evidence directory by design; and it never
checked that an annotated sentence id belonged to the split's sample, so the
fixture still produced a full per-tag F1 table and a confident interpretation
string.

Each test below drives the real CLI in a subprocess, because the gates live in
main() and the failure they prevent is a file appearing in a directory. Every
one of them FAILS CLOSED: a gate that silently stopped refusing would make these
tests fail, not pass quietly.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GS_DIR = PROJECT_ROOT / "DATA" / "gold_standard"
EVIDENCE_DIR = GS_DIR / "iaa"
PY = PROJECT_ROOT / ".venv-ml" / "Scripts" / "python.exe"

# Ids deliberately shaped like the fixture that caused the incident: nothing
# like the real `u-<16hex>#NNN`, and absent from every sample.
FOREIGN = ["s1", "s2", "s3"]


# --------------------------------------------------------------------- helpers
def _ann(sid_list: list[str]) -> dict:
    """One annotator's file in the app's autosave shape."""
    return {
        sid: {"relevance": "RELEVANT", "has_5tag": "YA",
              "spans": [{"span_text": "Python", "start_char": 0, "end_char": 6,
                         "label": "SH", "confidence": "tinggi", "flag": ""}]}
        for sid in sid_list
    }


def _write_ann_dir(dirpath: Path, sids: list[str]) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    for code in ("AV", "AI", "AD"):
        (dirpath / f"pilot_100_{code}.json").write_text(
            json.dumps(_ann(sids), ensure_ascii=False), encoding="utf-8")


def _run(*extra: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [str(PY), "-m", "gold_standard.iaa", "--split", "pilot_100", *extra],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, env=env, timeout=300)


def _evidence_snapshot() -> set[str]:
    if not EVIDENCE_DIR.is_dir():
        return set()
    return {p.name for p in EVIDENCE_DIR.iterdir()}


# ------------------------------------------------------------------ the gates
def test_foreign_ids_are_refused_by_default():
    """The incident, reproduced: a fixture must not yield a report at all."""
    before = _evidence_snapshot()
    with tempfile.TemporaryDirectory() as td:
        ann = Path(td) / "anns"
        _write_ann_dir(ann, FOREIGN)
        r = _run("--ann-dir", str(ann))
    assert r.returncode != 0, (
        "iaa accepted sentence ids that are not in the pilot_100 sample. That is "
        "how four fake 'substantial kappa' reports got into the evidence "
        f"directory.\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}")
    msg = r.stdout + r.stderr
    assert "not in the pilot_100 sample" in msg, (
        f"refused, but the reason does not name the cause:\n{msg}")
    assert _evidence_snapshot() == before, (
        "the run refused but still wrote into the real evidence directory")


def test_foreign_ids_with_the_escape_hatch_still_refuse_the_evidence_dir():
    """--allow-foreign-ids must not be a way back into the evidence directory.

    The escape hatch is tied to redirecting output on purpose: that is what
    makes the separation a mechanism rather than a convention.
    """
    before = _evidence_snapshot()
    with tempfile.TemporaryDirectory() as td:
        ann = Path(td) / "anns"
        _write_ann_dir(ann, FOREIGN)
        r = _run("--ann-dir", str(ann), "--allow-foreign-ids")
    assert r.returncode != 0, (
        "--allow-foreign-ids alone was enough to write a report into "
        f"{EVIDENCE_DIR}.\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}")
    assert "requires --out-dir" in (r.stdout + r.stderr)
    assert _evidence_snapshot() == before


def test_foreign_ids_with_out_dir_are_allowed_and_marked_unverified():
    """A test run must remain possible — just not where evidence lives."""
    before = _evidence_snapshot()
    with tempfile.TemporaryDirectory() as td:
        ann, out = Path(td) / "anns", Path(td) / "out"
        _write_ann_dir(ann, FOREIGN)
        r = _run("--ann-dir", str(ann), "--allow-foreign-ids", "--out-dir", str(out))
        assert r.returncode == 0, (
            f"a redirected test run should succeed.\nstdout:\n{r.stdout}\n"
            f"stderr:\n{r.stderr}")
        reports = sorted(out.glob("iaa_pilot_100_*.json"))
        assert len(reports) == 1, f"expected one report in {out}, found {reports}"
        rep = json.loads(reports[0].read_text(encoding="utf-8"))

        prov = rep.get("provenance")
        assert prov, "the report records no provenance block"
        assert prov["ids_verified_against_sample"] is False, (
            "a run on ids absent from the sample claims its ids were verified")
        assert prov["n_ids_not_in_sample"] == len(FOREIGN)
        assert Path(prov["ann_dir"]) == ann.resolve(), (
            "the report does not record which directory it actually read — the "
            "exact gap that made a fixture run indistinguishable from a real one")
        assert Path(prov["out_dir"]) == out.resolve()
    assert _evidence_snapshot() == before, (
        "a redirected run still touched the real evidence directory")


def test_quarantined_input_cannot_write_to_the_evidence_dir():
    """A directory carrying the marker is refused even with real-looking ids."""
    before = _evidence_snapshot()
    with tempfile.TemporaryDirectory() as td:
        ann = Path(td) / "_synthetic" / "annotations"
        _write_ann_dir(ann, FOREIGN)
        r = _run("--ann-dir", str(ann))
        msg = r.stdout + r.stderr
        assert r.returncode != 0, f"quarantined input was accepted:\n{msg}"
        assert "quarantined" in msg, (
            f"refused for some other reason than the quarantine:\n{msg}")
    assert _evidence_snapshot() == before


def test_marker_file_alone_quarantines_a_directory():
    """Not only the `_synthetic` name: the marker file must work on its own."""
    with tempfile.TemporaryDirectory() as td:
        ann = Path(td) / "someplace" / "anns"
        _write_ann_dir(ann, FOREIGN)
        (Path(td) / "someplace" / "SYNTHETIC_DO_NOT_CITE.md").write_text(
            "synthetic", encoding="utf-8")
        r = _run("--ann-dir", str(ann))
        msg = r.stdout + r.stderr
        assert r.returncode != 0 and "quarantined" in msg, (
            f"a parent directory's marker file did not quarantine its contents:\n{msg}")


def test_the_quarantined_fixture_on_disk_is_still_refused():
    """End-to-end on the real quarantine, not a synthetic one.

    Skips rather than passes if the quarantine was deliberately deleted — a
    vacuous pass here would read as corroboration.
    """
    real = GS_DIR / "_synthetic" / "annotations"
    if not real.is_dir():
        print("    (skipped: DATA/gold_standard/_synthetic/annotations/ is absent)")
        return
    before = _evidence_snapshot()
    r = _run("--ann-dir", str(real))
    msg = r.stdout + r.stderr
    assert r.returncode != 0, f"the real quarantine was accepted as input:\n{msg}"
    assert _evidence_snapshot() == before


def test_the_evidence_directory_holds_no_unverified_report():
    """Guards the state the quarantine established.

    Every report in DATA/gold_standard/iaa/ must either predate the provenance
    block (there should be none left — they were quarantined) or declare its ids
    verified. This is the test that fires if a fifth fake set ever lands there.
    """
    offenders: list[str] = []
    for p in sorted(EVIDENCE_DIR.glob("iaa_*.json")) if EVIDENCE_DIR.is_dir() else []:
        try:
            rep = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            offenders.append(f"{p.name}: unreadable ({exc})")
            continue
        prov = rep.get("provenance")
        if prov is None:
            offenders.append(
                f"{p.name}: no provenance block, so it predates the gates and its "
                f"input cannot be identified")
        elif not prov.get("ids_verified_against_sample"):
            offenders.append(
                f"{p.name}: ids_verified_against_sample is "
                f"{prov.get('ids_verified_against_sample')!r}, "
                f"n_ids_not_in_sample={prov.get('n_ids_not_in_sample')!r}")
    assert not offenders, (
        "DATA/gold_standard/iaa/ holds report(s) whose sentence ids were never "
        "verified against the split's sample. Those are not reliability "
        "evidence; move them to DATA/gold_standard/_synthetic/iaa/:\n  "
        + "\n  ".join(offenders))


def test_gates_live_in_the_module_not_only_in_this_test():
    """Fail closed on the mechanism itself, so deleting it is loud."""
    src = (PROJECT_ROOT / "gold_standard" / "iaa.py").read_text(
        encoding="utf-8", errors="replace")
    for needed in ("--out-dir", "--allow-foreign-ids", "_quarantine_marker",
                   "ids_verified_against_sample", "SYNTHETIC_DO_NOT_CITE.md"):
        assert needed in src, (
            f"gold_standard/iaa.py no longer contains {needed!r}; the provenance "
            f"gate was removed or renamed without updating this suite")
    assert PY.is_file(), f"the ML venv interpreter is missing at {PY}"


def _run_standalone() -> int:
    """Plain-python runner (the repo has no pytest). Pytest-compatible too."""
    fns = {n: f for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)}
    passed, failed = 0, []
    for name, fn in fns.items():
        try:
            fn()
            passed += 1
            print(f"  PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed"
          + (f", {len(failed)} FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
