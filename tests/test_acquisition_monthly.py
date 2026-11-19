"""Tests for acquisition.monthly — the schedule, with NO network.

Standalone runner (the repo has no pytest):

    .venv-ml\\Scripts\\python.exe tests\\test_acquisition_monthly.py

Everything runs on temporary folders with a fake runner in place of the real
batch runner, so no request leaves the machine. What is under test is the
schedule and its limits: session size, the 24-hour rest, what counts as due,
and that a blocked platform stops instead of being retried.
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from acquisition import monthly as M                                       # noqa: E402

NOW = datetime(2026, 12, 10, 3, 0, tzinfo=timezone.utc)
LIMITS = {"detail_fetch_max_queries_per_session": 25, "detail_fetch_cooldown_hours": 24,
          "max_hours_old": 2160}
ID = [{"iso": "ID", "location": "Indonesia"}]


def _env(n_terms=60, sources=None):
    """(config_dir, store_root, roster) in a fresh temp folder."""
    root = Path(tempfile.mkdtemp(prefix="monthly_"))
    cfg_dir, store = root / "configs", root / "store"
    cfg_dir.mkdir()
    store.mkdir()
    terms = [{"term": f"term {i}", "family": "f"} for i in range(n_terms)]
    (cfg_dir / "tpl.json").write_text(json.dumps({
        "batch_id": "tpl", "defaults": {"sites": ["x"]},
        "strata": [{"name": "indonesia", "results_wanted": 150, "countries": ID,
                    "terms": terms}]}), encoding="utf-8")
    roster = {"roster_version": "t-v1", "limits": LIMITS, "sources": sources or [
        {"key": "indeed-id", "site": "indeed", "stratum": "indonesia", "cadence_days": 28,
         "detail_fetch": False, "templates": [{"config": "tpl.json", "stratum": "indonesia"}]},
        {"key": "glints-id", "site": "glints", "stratum": "indonesia", "cadence_days": 28,
         "detail_fetch": True, "templates": [{"config": "tpl.json", "stratum": "indonesia"}]}]}
    return cfg_dir, store, roster


def _batch(store: Path, batch_id: str, site: str, when: datetime, n=10, logged=None):
    d = store / batch_id
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "batch_id": batch_id, "created_utc": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_queries": n, "query_log": [{}] * (n if logged is None else logged),
        "config": {"defaults": {"sites": [site]}, "strata": [{"name": "indonesia"}]}}),
        encoding="utf-8")


def _plan(cfg_dir, store, roster, now=NOW):
    return M.build_plan(roster, M.batch_history(store), now, config_dir=cfg_dir,
                        store_root=store)


def test_the_shipped_roster_loads_and_points_at_existing_templates():
    roster = M.load_roster()
    assert roster["limits"]["detail_fetch_max_queries_per_session"] <= 25
    assert roster["limits"]["detail_fetch_cooldown_hours"] >= 24
    for src in roster["sources"]:
        strata = M.template_strata(src)
        assert strata and all(s["terms"] for s in strata), src["key"]
        for session in M.sessions_for(src, strata, roster["limits"]):
            if src.get("detail_fetch"):
                assert M.n_queries(session) <= 25, src["key"]
    by = {s["key"]: s for s in roster["sources"]}
    assert by["jobstreet-id"]["detail_fetch"] and by["jobstreet-id"]["needs_probe"]
    assert by["glints-id"]["detail_fetch"]


def test_detail_fetch_sources_are_cut_into_sessions_of_at_most_25_queries():
    cfg_dir, store, roster = _env(n_terms=60)
    plan = _plan(cfg_dir, store, roster)
    glints = [e for e in plan if e["source"] == "glints-id"]
    assert [e["n_queries"] for e in glints] == [25, 25, 10]
    assert [e["batch_id"] for e in glints] == [f"batch-202612-glints-id-s{i}" for i in (1, 2, 3)]
    all_terms = [t["term"] for e in glints for s in e["config"]["strata"] for t in s["terms"]]
    assert len(all_terms) == 60 == len(set(all_terms))
    indeed = [e for e in plan if e["source"] == "indeed-id"]
    assert len(indeed) == 1 and indeed[0]["n_queries"] == 60
    assert indeed[0]["batch_id"] == "batch-202612-indeed-id"


def test_only_one_session_of_a_detail_platform_is_ready_per_day():
    cfg_dir, store, roster = _env()
    glints = [e for e in _plan(cfg_dir, store, roster) if e["source"] == "glints-id"]
    assert [e["state"] for e in glints] == ["ready", "cooling_down", "cooling_down"]


def test_the_24_hour_rest_is_counted_from_the_last_batch_on_that_platform():
    cfg_dir, store, roster = _env()
    _batch(store, "batch-202612-glints-id-s1", "glints", NOW - timedelta(hours=5))
    glints = [e for e in _plan(cfg_dir, store, roster) if e["source"] == "glints-id"]
    assert [e["state"] for e in glints] == ["done", "cooling_down", "cooling_down"]
    assert glints[1]["until"] == NOW + timedelta(hours=19)
    later = [e for e in _plan(cfg_dir, store, roster, NOW + timedelta(hours=20))
             if e["source"] == "glints-id"]
    assert [e["state"] for e in later] == ["done", "ready", "cooling_down"]


def test_a_source_collected_recently_is_not_due_and_becomes_due_after_its_cadence():
    cfg_dir, store, roster = _env()
    _batch(store, "batch-old", "indeed", NOW - timedelta(days=10))
    (e,) = [e for e in _plan(cfg_dir, store, roster) if e["source"] == "indeed-id"]
    assert e["state"] == "not_due" and e["until"] == NOW + timedelta(days=18)
    (e,) = [e for e in _plan(cfg_dir, store, roster, NOW + timedelta(days=19))
            if e["source"] == "indeed-id"]
    assert e["state"] == "ready"
    assert e["config"]["defaults"]["hours_old"] == 29 * 24 + 48      # covers the whole gap


def test_hours_old_is_capped_and_a_disabled_source_is_left_out():
    cfg_dir, store, roster = _env()
    _batch(store, "batch-old", "indeed", NOW - timedelta(days=300))
    roster["sources"][1]["enabled"] = False
    plan = _plan(cfg_dir, store, roster)
    assert [e["source"] for e in plan] == ["indeed-id"]
    assert plan[0]["config"]["defaults"]["hours_old"] == LIMITS["max_hours_old"]


def test_preview_makes_no_real_run_and_writes_the_config_once():
    cfg_dir, store, roster = _env()
    plan = _plan(cfg_dir, store, roster)
    calls = []
    monthly_dir = cfg_dir / "monthly"
    res = M.execute(plan, go=False, runner=lambda cmd, log: calls.append(cmd) or 0,
                    monthly_dir=monthly_dir, store_root=store, out=lambda *_: None)
    assert res["dry_run"] and len(calls) == 2
    assert all("--dry-run" in c for c in calls) and "ladder_exit_code" not in res
    path = monthly_dir / "batch-202612-indeed-id.json"
    first = path.read_text(encoding="utf-8")
    plan[0]["config"]["notes"] = "changed"
    M.write_config(plan[0], monthly_dir)
    assert path.read_text(encoding="utf-8") == first


def test_go_runs_in_order_then_the_ladder_and_stops_a_platform_that_refuses():
    src = [{"key": "jobstreet-id", "site": "jobstreet", "stratum": "indonesia",
            "cadence_days": 28, "detail_fetch": True, "needs_probe": True,
            "templates": [{"config": "tpl.json", "stratum": "indonesia"}]},
           {"key": "indeed-id", "site": "indeed", "stratum": "indonesia", "cadence_days": 28,
            "detail_fetch": False, "templates": [{"config": "tpl.json", "stratum": "indonesia"}]}]
    cfg_dir, store, roster = _env(n_terms=10, sources=src)
    plan = _plan(cfg_dir, store, roster)
    calls = []

    def runner(cmd, log):
        calls.append(cmd[2])
        if cmd[2] == "acquisition.probe_jobstreet":
            return 1                                    # blocked
        if cmd[2] == "acquisition.run_batch":
            batch_id = Path(cmd[4]).stem
            _batch(store, batch_id, "indeed", NOW, n=10, logged=4)   # stopped early
        return 0
    res = M.execute(plan, go=True, runner=runner, monthly_dir=cfg_dir / "monthly",
                    store_root=store, out=lambda *_: None)
    assert calls == ["acquisition.probe_jobstreet", "acquisition.run_batch", "governance.ladder"]
    assert res["skipped"] == [("batch-202612-jobstreet-id", "probe failed: platform blocked")]
    assert res["ran"] == ["batch-202612-indeed-id"]
    assert res["stopped"] == [("batch-202612-indeed-id", "4/10 queries")]


def test_a_failed_batch_stops_that_platform_and_no_ladder_runs_without_data():
    cfg_dir, store, roster = _env(n_terms=60)
    plan = _plan(cfg_dir, store, roster)
    calls = []
    res = M.execute(plan, go=True, runner=lambda cmd, log: calls.append(cmd[2]) or 3,
                    monthly_dir=cfg_dir / "monthly", store_root=store, out=lambda *_: None)
    assert res["ran"] == [] and len(res["stopped"]) == 2
    assert "governance.ladder" not in calls


def test_the_module_adds_no_fetching_code_of_its_own():
    src = (PROJECT_ROOT / "acquisition" / "monthly.py").read_text(encoding="utf-8")
    for word in ("requests", "httpx", "urllib", "selenium", "playwright", "proxy",
                 "user_agent", "User-Agent"):
        assert word not in src, word
    assert '"acquisition.run_batch"' in src and "--go" in src


def test_status_text_is_plain():
    cfg_dir, store, roster = _env()
    text = M.format_plan(_plan(cfg_dir, store, roster), NOW)
    assert "READY" in text and "waiting (24 h rest)" in text and "can run now" in text


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
