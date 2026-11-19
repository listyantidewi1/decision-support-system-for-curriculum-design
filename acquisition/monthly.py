"""The monthly collection as one command.

A 36-month corpus cannot be scraped backwards; it is accumulated forward, one
snapshot at a time (docs/corpus/ACQUISITION_CADENCE.md). This turns that
runbook's monthly procedure into something that can be run without help:

    python -m acquisition.monthly status       what is due, what is cooling down
    python -m acquisition.monthly run          SHOW what would be collected (no network)
    python -m acquisition.monthly run --go     collect it, then recompute the ladder

`run` without `--go` touches nothing outside this computer. With `--go` it runs
the due sources one after another through `acquisition.run_batch` — the same
runner, adapters, pacing and circuit breakers as every batch so far. This file
adds a schedule on top; it adds no new way of fetching anything.

The limits it enforces are the standing ones:

  * a platform whose detail pages are fetched (Jobstreet, Glints) gets at most
    25 queries per session and at least 24 hours between sessions. A source
    with more terms is cut into sessions, so running the command once a day
    works through them. This is pacing to stay within what the platform
    tolerates, not a way around its protections: when a platform answers 403 or
    429 the runner stops, and so does this.
  * Jobstreet is probed first with a single search request; if it is blocked,
    it is skipped and reported.
  * one platform at a time; raw batches are immutable.
  * only sources listed in `configs/monthly_roster.json` are collected. Adding
    a source needs its audit and sign-off first (SOURCE_AUDIT.md).

What it does NOT do: commit anything, or update the to-do list and notes. After
a run, see section 4 steps 6-8 of the cadence runbook.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .batch_store import DEFAULT_ROOT

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "acquisition" / "configs"
ROSTER_PATH = CONFIG_DIR / "monthly_roster.json"
MONTHLY_DIR = CONFIG_DIR / "monthly"
OPS_LOG_DIR = Path(r"D:\Projects\_skill-extraction-ops") / "acquisition"


class RosterError(ValueError):
    pass


def load_roster(path: Path = ROSTER_PATH) -> dict:
    roster = json.loads(Path(path).read_text(encoding="utf-8"))
    keys = [s["key"] for s in roster["sources"]]
    if len(set(keys)) != len(keys):
        raise RosterError("duplicate source keys in the roster")
    for s in roster["sources"]:
        for field in ("site", "stratum", "cadence_days", "templates"):
            if field not in s:
                raise RosterError(f"{s['key']}: missing {field}")
    return roster


def _parse_utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def batch_history(store_root: Path = DEFAULT_ROOT) -> list[dict]:
    """Every batch whose manifest records its config: id, time, site, strata."""
    out = []
    for manifest in sorted(Path(store_root).glob("*/manifest.json")):
        m = json.loads(manifest.read_text(encoding="utf-8"))
        cfg = m.get("config") or {}
        sites = (cfg.get("defaults") or {}).get("sites") or []
        if not sites or not m.get("created_utc"):
            continue
        out.append({"batch_id": m.get("batch_id", manifest.parent.name),
                    "created": _parse_utc(m["created_utc"]), "sites": list(sites),
                    "strata": [s.get("name") for s in cfg.get("strata", [])],
                    "n_queries": m.get("n_queries"),
                    "n_queries_logged": len(m.get("query_log") or []),
                    "n_queries_failed": m.get("n_queries_failed")})
    return out


def last_run(history: list[dict], site: str, stratum: str | None = None) -> datetime | None:
    times = [h["created"] for h in history
             if site in h["sites"] and (stratum is None or stratum in h["strata"])]
    return max(times) if times else None


def template_strata(source: dict, config_dir: Path = CONFIG_DIR) -> list[dict]:
    """The strata to query: copied from the template configs, terms of strata
    with the same countries merged without duplicates."""
    merged: dict[tuple, dict] = {}
    for t in source["templates"]:
        cfg = json.loads((Path(config_dir) / t["config"]).read_text(encoding="utf-8"))
        found = [s for s in cfg["strata"] if s["name"] == t["stratum"]]
        if not found:
            raise RosterError(f"{source['key']}: {t['config']} has no stratum {t['stratum']}")
        for s in found:
            key = tuple(c["iso"] for c in s["countries"])
            tgt = merged.setdefault(key, {"name": s["name"], "countries": s["countries"],
                                          "results_wanted": source.get(
                                              "results_wanted", s.get("results_wanted", 100)),
                                          "terms": []})
            seen = {x["term"] for x in tgt["terms"]}
            tgt["terms"] += [x for x in s["terms"] if x["term"] not in seen
                             and not seen.add(x["term"])]
    return list(merged.values())


def sessions_for(source: dict, strata: list[dict], limits: dict) -> list[list[dict]]:
    """One session = one batch. A detail-fetch platform is cut so that no
    session makes more than the allowed number of queries."""
    if not source.get("detail_fetch"):
        return [strata]
    cap = int(limits["detail_fetch_max_queries_per_session"])
    sessions, current, used = [], [], 0
    for s in strata:
        per_term = len(s["countries"])
        if per_term > cap:
            raise RosterError(f"{source['key']}: one term over {per_term} countries already "
                              f"exceeds the session limit of {cap} queries")
        terms = list(s["terms"])
        while terms:
            room = (cap - used) // per_term
            if room == 0:
                sessions.append(current)
                current, used = [], 0
                continue
            take, terms = terms[:room], terms[room:]
            current.append({**s, "terms": take})
            used += len(take) * per_term
    if current:
        sessions.append(current)
    return sessions


def n_queries(strata: list[dict]) -> int:
    return sum(len(s["terms"]) * len(s["countries"]) for s in strata)


def build_plan(roster: dict, history: list[dict], now: datetime, month: str | None = None,
               config_dir: Path = CONFIG_DIR, store_root: Path = DEFAULT_ROOT) -> list[dict]:
    """One entry per session of every enabled source, with its state:
    done | ready | cooling_down | not_due."""
    month = month or now.strftime("%Y%m")
    limits = roster["limits"]
    cooldown = timedelta(hours=float(limits["detail_fetch_cooldown_hours"]))
    plan, queued_site_today = [], set()
    for src in roster["sources"]:
        if not src.get("enabled", True):
            continue
        strata = template_strata(src, config_dir)
        sessions = sessions_for(src, strata, limits)
        ids = [f"batch-{month}-{src['key']}" + (f"-s{i}" if len(sessions) > 1 else "")
               for i in range(1, len(sessions) + 1)]
        done = [(Path(store_root) / b / "manifest.json").is_file() for b in ids]
        prev = last_run([h for h in history if h["batch_id"] not in ids],
                        src["site"], src["stratum"])
        age_days = (now - prev).total_seconds() / 86400 if prev else None
        cycle_open = any(done) or prev is None or age_days >= src["cadence_days"]
        site_last = last_run(history, src["site"])
        hours_old = src["cadence_days"] * 24
        if prev is not None:
            hours_old = max(hours_old, int((now - prev).total_seconds() / 3600) + 48)
        hours_old = min(hours_old, int(limits["max_hours_old"]))
        for i, (batch_id, session, is_done) in enumerate(zip(ids, sessions, done)):
            state, until = "ready", None
            if is_done:
                state = "done"
            elif not cycle_open:
                state = "not_due"
                until = prev + timedelta(days=src["cadence_days"])
            elif src.get("detail_fetch"):
                if site_last is not None and now - site_last < cooldown:
                    state, until = "cooling_down", site_last + cooldown
                elif src["site"] in queued_site_today:
                    state, until = "cooling_down", now + cooldown
                else:
                    queued_site_today.add(src["site"])
            plan.append({
                "source": src["key"], "site": src["site"], "stratum": src["stratum"],
                "batch_id": batch_id, "session": i + 1, "n_sessions": len(sessions),
                "n_queries": n_queries(session), "state": state, "until": until,
                "last_run": prev, "needs_probe": bool(src.get("needs_probe")),
                "detail_fetch": bool(src.get("detail_fetch")),
                "config": {
                    "batch_id": batch_id,
                    "query_plan_version": f"{roster['roster_version']}/{month}",
                    "notes": (f"Monthly forward-accumulation batch for {src['key']}, session "
                              f"{i + 1} of {len(sessions)}, planned by acquisition.monthly "
                              f"on {now.date().isoformat()}."),
                    "defaults": {"sites": [src["site"]], "hours_old": hours_old,
                                 "sleep_seconds": src.get("sleep_seconds", 5),
                                 "fetch_description": True},
                    "strata": session}})
    return plan


def write_config(entry: dict, monthly_dir: Path = MONTHLY_DIR) -> Path:
    """Write the session's config once. An existing file is kept as it is, so a
    session that has been planned is not silently re-planned with other terms."""
    path = Path(monthly_dir) / f"{entry['batch_id']}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entry["config"], indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    return path


def format_plan(plan: list[dict], now: datetime) -> str:
    words = {"done": "done", "ready": "READY", "cooling_down": "waiting (24 h rest)",
             "not_due": "not due yet"}
    lines = [f"{'source':<14}{'session':>8}{'queries':>9}  {'last collected':<16}{'state'}"]
    for e in plan:
        last = e["last_run"].strftime("%Y-%m-%d") if e["last_run"] else "never"
        state = words[e["state"]]
        if e["until"] is not None and e["state"] != "done":
            state += f" until {e['until'].astimezone().strftime('%Y-%m-%d %H:%M')}"
        lines.append(f"{e['source']:<14}{e['session']:>4}/{e['n_sessions']:<3}{e['n_queries']:>9}"
                     f"  {last:<16}{state}")
    ready = [e for e in plan if e["state"] == "ready"]
    lines.append("")
    lines.append(f"{len(ready)} session(s) can run now, {sum(e['n_queries'] for e in ready)} "
                 f"search queries in total." if ready else "Nothing to collect right now.")
    return "\n".join(lines)


def _python() -> str:
    return sys.executable


def _run(cmd: list[str], log_path: Path | None) -> int:
    """Run a step, echoing its output and keeping a copy in the log file."""
    fh = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = log_path.open("a", encoding="utf-8")
    try:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace")
        for line in proc.stdout:
            print("    " + line.rstrip())
            if fh:
                fh.write(line)
        return proc.wait()
    finally:
        if fh:
            fh.close()


def execute(plan: list[dict], *, go: bool, only: str | None = None, runner=_run,
            monthly_dir: Path = MONTHLY_DIR, store_root: Path = DEFAULT_ROOT,
            log_dir: Path | None = None, out=print) -> dict:
    """Run every READY session in roster order. Returns what happened."""
    result = {"ran": [], "skipped": [], "stopped": [], "dry_run": not go}
    stopped_sites: set[str] = set()
    for e in plan:
        if e["state"] != "ready" or (only and e["source"] != only):
            continue
        if e["site"] in stopped_sites:
            result["skipped"].append((e["batch_id"], "platform stopped earlier in this run"))
            continue
        cfg = write_config(e, monthly_dir)
        log = (log_dir / f"{e['batch_id']}.log") if (go and log_dir) else None
        out(f"\n== {e['batch_id']}  ({e['n_queries']} queries on {e['site']})")
        if not go:
            runner([_python(), "-m", "acquisition.run_batch", "--config", str(cfg), "--dry-run"],
                   None)
            result["ran"].append(e["batch_id"])
            continue
        if e["needs_probe"]:
            if runner([_python(), "-m", "acquisition.probe_jobstreet"], log) != 0:
                out(f"   {e['site']} is not reachable (blocked). Skipped; try again another day.")
                result["skipped"].append((e["batch_id"], "probe failed: platform blocked"))
                stopped_sites.add(e["site"])
                continue
        code = runner([_python(), "-m", "acquisition.run_batch", "--config", str(cfg)], log)
        manifest = Path(store_root) / e["batch_id"] / "manifest.json"
        if code != 0 or not manifest.is_file():
            out(f"   The batch did not finish (exit code {code}). Nothing else is run on "
                f"{e['site']} now.")
            result["stopped"].append((e["batch_id"], f"exit code {code}"))
            stopped_sites.add(e["site"])
            continue
        m = json.loads(manifest.read_text(encoding="utf-8"))
        made, planned = len(m.get("query_log") or []), m.get("n_queries") or 0
        result["ran"].append(e["batch_id"])
        if made < planned:
            out(f"   Stopped early: {made} of {planned} queries were made (the platform "
                f"started refusing, or the connection failed). Leave {e['site']} alone for "
                f"24 hours.")
            result["stopped"].append((e["batch_id"], f"{made}/{planned} queries"))
            stopped_sites.add(e["site"])
    if go and result["ran"]:
        out("\n== recomputing the corpus ladder")
        result["ladder_exit_code"] = runner([_python(), "-m", "governance.ladder"],
                                            (log_dir / "ladder.log") if log_dir else None)
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["status", "run"])
    ap.add_argument("--go", action="store_true",
                    help="really collect (without it, `run` only shows the plan)")
    ap.add_argument("--only", help="run a single source, e.g. indeed-id")
    ap.add_argument("--month", help="YYYYMM (default: this month)")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    roster = load_roster()
    plan = build_plan(roster, batch_history(), now, args.month)
    print(format_plan(plan, now))
    if args.command == "status":
        return
    if not args.go:
        print("\nThis is a preview: nothing is fetched. Add --go to collect.")
    log_dir = OPS_LOG_DIR if OPS_LOG_DIR.parent.is_dir() else REPO_ROOT / "logs" / "acquisition"
    result = execute(plan, go=args.go, only=args.only, log_dir=log_dir)
    if args.go:
        print(f"\nCollected: {len(result['ran'])} batch(es). "
              f"Skipped: {len(result['skipped'])}. Stopped early: {len(result['stopped'])}.")
        for batch_id, why in result["skipped"] + result["stopped"]:
            print(f"  {batch_id}: {why}")
        waiting = [e for e in plan if e["state"] == "cooling_down"]
        if waiting:
            print(f"{len(waiting)} session(s) are still waiting for their 24-hour rest: run "
                  f"this command again tomorrow.")
        print(f"Logs: {log_dir}")
        print("Then: check the ladder by platform and stratum, and update TODO.md "
              "(cadence runbook, section 4, steps 6-8).")


if __name__ == "__main__":
    main()
