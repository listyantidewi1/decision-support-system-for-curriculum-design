"""The `legacy/` boundary — enforced, not aspirational.

Standalone runner (the repo has no pytest):

    .venv-ml\\Scripts\\python.exe tests\\test_legacy_boundary.py

`legacy/` holds pre-approval code kept for provenance (see legacy/README.md and
CLAUDE.md §7). The boundary decays the first time someone is in a hurry, so it
is checked mechanically rather than documented and hoped for.

Every check here FAILS CLOSED: if `legacy/` disappears, if its README goes
missing, or if the scan finds no files to inspect, the test fails rather than
passing vacuously. A boundary test that silently passes on an empty input set
is worse than no test, because it reads as corroboration.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LEGACY = PROJECT_ROOT / "legacy"
EXCLUDE_DIRS = {".git", ".venv", ".venv-ml", "__pycache__", "node_modules",
                ".pytest_cache", "DATA", "cache", "results"}


# --------------------------------------------------------------------- helpers
def _tracked_py() -> list[Path]:
    """Tracked .py files, so untracked scratch files cannot fail the build."""
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=PROJECT_ROOT,
                         capture_output=True, text=True, check=True).stdout
    paths = []
    for line in out.splitlines():
        if not line.strip():
            continue
        p = PROJECT_ROOT / line
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        paths.append(p)
    return paths


def _rel(p: Path) -> str:
    return p.relative_to(PROJECT_ROOT).as_posix()


def _legacy_py() -> list[Path]:
    return [p for p in _tracked_py() if _rel(p).startswith("legacy/")]


def _live_py() -> list[Path]:
    return [p for p in _tracked_py() if not _rel(p).startswith("legacy/")]


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


# ----------------------------------------------------------------- the quarantine
def test_legacy_directory_exists_with_a_readme():
    """Fail closed: no legacy/ means the other checks below are vacuous."""
    assert LEGACY.is_dir(), (
        "legacy/ does not exist. If the quarantine was deliberately removed, "
        "delete this test file in the same commit and say why; do not leave a "
        "boundary test that cannot fail.")
    readme = LEGACY / "README.md"
    assert readme.is_file(), "legacy/README.md is missing"
    body = _read(readme)
    assert len(body) > 500, "legacy/README.md must explain what is here and why"
    assert "not a specification" in body, (
        "legacy/README.md must state that it is not a specification — that is "
        "the whole point of the directory")


def test_scan_population_is_not_empty():
    """Fail closed on the scan itself, not just on its findings."""
    legacy, live = _legacy_py(), _live_py()
    assert legacy, "no tracked .py files under legacy/ — nothing to enforce"
    assert len(live) > 50, (
        f"only {len(live)} live .py files found; git ls-files probably failed, "
        f"so a clean result here would mean nothing")


# ------------------------------------------------------- the boundary itself
def test_live_code_does_not_import_legacy_modules():
    """The load-bearing check: no live module may import a legacy one.

    Legacy files sit at import-visible names (`legacy/skill_normalizer.py` is
    importable as `skill_normalizer` if legacy/ is ever put on sys.path, and as
    `legacy.skill_normalizer` directly), so both spellings are refused.
    """
    legacy = _legacy_py()
    names = {p.stem for p in legacy if p.stem != "__init__"}
    dotted = {_rel(p)[:-3].replace("/", ".") for p in legacy}

    violations: list[str] = []
    for src in _live_py():
        txt = _read(src)
        for n in sorted(names):
            if re.search(rf"(?:^|\n)\s*(?:from\s+{re.escape(n)}\s+import"
                         rf"|import\s+{re.escape(n)}\b)", txt):
                violations.append(f"{_rel(src)} imports legacy module {n!r}")
        for d in sorted(dotted):
            if re.search(rf"(?:^|\n)\s*(?:from\s+{re.escape(d)}\s+import"
                         rf"|import\s+{re.escape(d)}\b)", txt):
                violations.append(f"{_rel(src)} imports {d!r}")

    assert not violations, (
        "live code must not import from legacy/ — port it deliberately under "
        "docs/IMPLEMENTATION_PLAN.md instead:\n  " + "\n  ".join(violations))


def test_live_code_does_not_reference_legacy_paths():
    """Catch the subprocess/string route, which an import scan would miss.

    This repo launches scripts by path and by `-m` as well as by import (see
    CLAUDE.md §4), so a path reference is as real a dependency as an import.
    """
    legacy_rels = [_rel(p) for p in _legacy_py()]
    legacy_rels += [_rel(p) for p in LEGACY.rglob("*.bat")] if LEGACY.is_dir() else []
    violations = []
    for src in _live_py():
        if _rel(src) == "tests/test_legacy_boundary.py":
            continue
        txt = _read(src)
        for rel in legacy_rels:
            if rel in txt:
                violations.append(f"{_rel(src)} references {rel}")
    assert not violations, (
        "live code references a path inside legacy/:\n  " + "\n  ".join(violations))


def test_legacy_modules_are_not_named_by_the_implementation_plan():
    """The plan is the scope authority; anything it names is LIVE by definition.

    If this fires, the quarantine is wrong, not the plan.
    """
    plan = PROJECT_ROOT / "docs" / "IMPLEMENTATION_PLAN.md"
    assert plan.is_file(), "docs/IMPLEMENTATION_PLAN.md is missing"
    txt = _read(plan)
    named = [p.name for p in _legacy_py() if p.name in txt]
    assert not named, (
        f"docs/IMPLEMENTATION_PLAN.md names these quarantined files, which "
        f"makes them LIVE: {named}. Move them back out of legacy/.")


def test_multitask_model_stayed_at_the_repo_root():
    """A named regression guard for the silent-failure trap.

    `pipeline.py:41-50` imports multitask_model AND config inside one
    try/except ImportError with a DummyConfig fallback, so moving this file
    does not raise — the pipeline runs on empty label maps and emits
    plausible-looking garbage behind a single 'Warning:' line. Only
    `train_jobbert_research.py` (quarantined) and `pipeline.py` import it, so
    it is the obvious thing for a future tidy-up to move. It must not move
    while pipeline.py still exists.
    """
    root_copy = PROJECT_ROOT / "multitask_model.py"
    pipeline = PROJECT_ROOT / "pipeline.py"
    if not pipeline.is_file():
        return  # pipeline.py retired; the trap is gone with it
    assert root_copy.is_file(), (
        "multitask_model.py has left the repo root while pipeline.py still "
        "imports it. That import is wrapped in a try/except ImportError with "
        "a DummyConfig fallback, so this does NOT crash — it produces output "
        "from empty label maps. Restore it, or retire pipeline.py.")
    assert not (LEGACY / "multitask_model.py").is_file(), (
        "multitask_model.py was quarantined; see the docstring above")


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
