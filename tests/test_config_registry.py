"""Tests for governance.config_registry — the registry must match the code.

Standalone runner (the repo has no pytest):

    .venv-ml\\Scripts\\python.exe tests\\test_config_registry.py

If `test_committed_registry_matches_the_code` fails, a version or threshold was
changed without regenerating the registry:

    .venv-ml\\Scripts\\python.exe -m governance.config_registry
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from governance import config_registry as R                                # noqa: E402

ROWS = R.collect()


def test_committed_registry_matches_the_code():
    assert R.check(), "docs/CONFIG_REGISTRY.md is out of date: regenerate it"


def test_every_entry_has_a_status_a_value_and_an_existing_location():
    assert len(ROWS) >= 50
    for r in ROWS:
        assert r["status"] in (R.REGISTERED, R.DECIDED, R.DESIGN, R.STARTING), r
        assert r["name"] and r["value"] != "", r
        target = PROJECT_ROOT / r["where"]
        assert target.exists(), f"{r['name']}: {r['where']} does not exist"


def test_registered_entries_point_at_a_registration_document():
    registered = [r for r in ROWS if r["status"] == R.REGISTERED]
    assert len(registered) >= 8
    for r in registered:
        assert r["where"].startswith("docs/experiments/"), r
        assert "PREREG" in r["where"] or "split_spec" in r["where"], r


def test_starting_values_are_listed_and_include_the_known_open_ones():
    names = " | ".join(r["name"] for r in ROWS if r["status"] == R.STARTING)
    for known in ("router thresholds", "contract", "coverage thresholds", "verb table",
                  "time analysis settings", "adjudicator model", "KKNI step"):
        assert known in names, known
    doc = R.to_markdown(ROWS)
    assert "## Still to be fixed" in doc and doc.count("**starting value**") >= 10


def test_registry_reads_values_from_the_code_not_from_itself():
    from fivetag import conflict_routing as CR
    from gold_standard import gate
    by = {r["name"]: r["value"] for r in ROWS}
    assert by["pilot gate: go at / stop below"] == f"[{gate.GO_AT}, {gate.STOP_BELOW}]"
    assert str(CR.AUDIT_SAMPLE_N) in by["conflict routing: audit sample size / seed"]
    assert len(by["lexicon v1 (SHA-256)"]) == 64


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
