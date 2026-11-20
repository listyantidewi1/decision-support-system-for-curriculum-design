"""The versioned competency profile and its change log (proposal §3.3.7).

A release such as "Profil Kompetensi RPL 2026-2" carries, as the proposal
lists them: the version identity, a summary of the corpus and its
denominators, the recommendations with their status and delta type, and a
change log against the previous version naming what rose, fell, entered or
left the list, with reasons.

Rules this module enforces:

  * A version is written ONCE. Publishing over an existing version is refused;
    a correction is a new version.
  * A profile built from an engineering run says so in its first lines and in
    its metadata, and is written under `drafts/`, not `releases/`. Only a run
    marked as valid for release (by the caller, a person's decision) goes to
    `releases/`.
  * Competencies are matched across versions by what they are made of, not by
    their internal id (cluster ids change between runs): two competencies are
    the same when their member phrases overlap enough, or their statements are
    identical.

Output: `profile.json` (everything, for the dashboards) and `profile.md` (for
a reader).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

from . import COMPETENCY_VERSION, STATUS_CANDIDATE, STATUS_EXPLORATORY, STATUS_READY
from .delta import priority_order

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILES_DIR = REPO_ROOT / "DATA" / "competency" / "profiles"
PROFILE_VERSION = "profile-v1"
MATCH_JACCARD = 0.5
_RANK = {STATUS_EXPLORATORY: 0, STATUS_CANDIDATE: 1, STATUS_READY: 2}
_STATUS_ID = {STATUS_EXPLORATORY: "Eksploratif", STATUS_CANDIDATE: "Kandidat tinjauan",
              STATUS_READY: "Siap ditelaah formal"}


class ProfileError(RuntimeError):
    pass


def _member_keys(c: dict) -> set[str]:
    return {m["text"].lower() for m in c["evidence"]["members"]}


def match(old: Sequence[dict], new: Sequence[dict]) -> dict[str, str]:
    """new competency_id -> old competency_id. Each old one is used at most once;
    the best overlaps are paired first."""
    pairs = []
    for n in new:
        nk, ns = _member_keys(n), n["statement"]["text"].lower()
        for o in old:
            ok = _member_keys(o)
            j = len(nk & ok) / len(nk | ok) if nk | ok else 0.0
            if o["statement"]["text"].lower() == ns:
                j = max(j, 1.0)
            if j >= MATCH_JACCARD:
                pairs.append((-j, n["competency_id"], o["competency_id"]))
    out, used = {}, set()
    for _, n_id, o_id in sorted(pairs):
        if n_id not in out and o_id not in used:
            out[n_id] = o_id
            used.add(o_id)
    return out


def changelog(old_profile: dict | None, new: Sequence[dict]) -> dict:
    """What rose, fell, entered or left, with the reason read from the blockers."""
    if old_profile is None:
        return {"previous_version": None, "note": "first version: no change log",
                "naik": [], "turun": [], "masuk": [c["competency_id"] for c in new], "keluar": []}
    old = old_profile["recommendations"]
    pairing = match(old, new)
    old_by = {c["competency_id"]: c for c in old}
    naik, turun, masuk = [], [], []
    for c in new:
        o = old_by.get(pairing.get(c["competency_id"], ""))
        if o is None:
            masuk.append({"competency_id": c["competency_id"], "statement": c["statement"]["text"],
                          "status": c["contract"]["status"],
                          "reason": "no competency in the previous version is made of the same evidence"})
            continue
        a, b = o["contract"]["status"], c["contract"]["status"]
        if _RANK[b] == _RANK[a]:
            continue
        gone = sorted(set(o["contract"]["blockers"]) - set(c["contract"]["blockers"]))
        came = sorted(set(c["contract"]["blockers"]) - set(o["contract"]["blockers"]))
        row = {"competency_id": c["competency_id"], "previous_id": o["competency_id"],
               "statement": c["statement"]["text"], "from": a, "to": b,
               "reason": ("no longer blocked by: " + ", ".join(gone) if _RANK[b] > _RANK[a]
                          else "now blocked by: " + ", ".join(came)) or "status changed"}
        (naik if _RANK[b] > _RANK[a] else turun).append(row)
    kept = set(pairing.values())
    keluar = [{"competency_id": o["competency_id"], "statement": o["statement"]["text"],
               "status": o["contract"]["status"],
               "reason": "no competency in this version is made of the same evidence"}
              for o in old if o["competency_id"] not in kept]
    return {"previous_version": old_profile["identity"]["version"], "naik": naik, "turun": turun,
            "masuk": masuk, "keluar": keluar}


def build(run_dir: Path, version: str, *, previous: Path | None = None,
          valid_for_release: bool = False, profiles_dir: Path = PROFILES_DIR,
          title: str = "Profil Kompetensi RPL") -> dict:
    """Publish a profile from a finished `competency.pipeline` run."""
    run_dir = Path(run_dir)
    comps = json.loads((run_dir / "competencies.json").read_text(encoding="utf-8"))
    # statements formulated in the SKKNI style (competency.formulate), when the run has them
    formulated: dict = {}
    if (run_dir / "formulated.json").is_file():
        formulated = {k: v for k, v in json.loads((run_dir / "formulated.json").read_text(
            encoding="utf-8"))["statements"].items() if v.get("accepted")}
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    kind = "releases" if valid_for_release else "drafts"
    out_dir = Path(profiles_dir) / kind / version
    if out_dir.exists():
        raise ProfileError(f"{out_dir} already exists. A version is written once; publish a "
                           f"correction as a new version.")
    old = json.loads(Path(previous).read_text(encoding="utf-8")) if previous else None
    ordered = priority_order(comps)
    recs = []
    for rank, c in enumerate(ordered, 1):
        recs.append({
            "rank": rank, "competency_id": c["competency_id"], "statement": c["statement"],
            "stream": c["stream"], "contract": c["contract"], "delta": c.get("delta"),
            "cognitive": c.get("cognitive"), "family": c.get("family"),
            "formulated": formulated.get(c["competency_id"]),
            "coverage": c.get("coverage"), "kkni": c.get("kkni"), "okupasi": c.get("okupasi"),
            "temporal": c.get("temporal"), "n_postings": c["n_postings"],
            "n_posting_groups": c.get("n_posting_groups"),
            "share_of_postings": c["share_of_postings"],
            "evidence": {k: c["evidence"].get(k, []) for k in (
                "members", "verbs", "knowledge_objects", "technologies", "knowledge_context",
                "knowledge", "soft_skills", "work_behaviours", "samples")}})
    counts: dict[str, int] = {}
    for r in recs:
        counts[r["contract"]["status"]] = counts.get(r["contract"]["status"], 0) + 1
    profile = {
        "identity": {"title": f"{title} {version}", "version": version, "kind": kind,
                     "valid_for_release": valid_for_release,
                     "created": time.strftime("%Y-%m-%d"), "profile_version": PROFILE_VERSION,
                     "competency_version": COMPETENCY_VERSION, "source_run": run_dir.name,
                     "notice": "" if valid_for_release else
                     "DRAFT built from an engineering run. Not a released profile; not a finding."},
        "corpus": {"evidence_source": report["evidence_source"],
                   "denominators": report["denominators"],
                   "n_evidence": report["n_evidence"],
                   "n_evidence_by_tag": report["n_evidence_by_tag"],
                   "grouping": {k: report["grouping"][k] for k in (
                       "n_groups", "status_distribution", "n_evidence_grouped",
                       "n_evidence_left_ungrouped")}},
        "method": {k: report.get(k) for k in ("statement_method", "verb_table_version",
                                               "grounding_version", "contract", "pipeline")},
        "summary": {"n_recommendations": len(recs), "by_status": counts,
                    "by_delta": _count(recs, lambda r: (r["delta"] or {}).get("type"))},
        "recommendations": recs,
        "changelog": changelog(old, ordered),
    }
    out_dir.mkdir(parents=True)
    (out_dir / "profile.json").write_text(json.dumps(profile, indent=1, ensure_ascii=False,
                                                     default=str), encoding="utf-8")
    (out_dir / "profile.md").write_text(to_markdown(profile), encoding="utf-8")
    if (run_dir / "term_map.json").is_file():     # the cluster map travels with the profile
        (out_dir / "term_map.json").write_bytes((run_dir / "term_map.json").read_bytes())
    profile["out_dir"] = str(out_dir)
    return profile


def _count(rows, key) -> dict:
    out: dict[str, int] = {}
    for r in rows:
        k = key(r)
        if k:
            out[k] = out.get(k, 0) + 1
    return out


def to_markdown(profile: dict, top: int = 40) -> str:
    i, c, s, log = profile["identity"], profile["corpus"], profile["summary"], profile["changelog"]
    lines = [f"# {i['title']}", ""]
    if i["notice"]:
        lines += [f"> **{i['notice']}**", ""]
    lines += [f"Dibuat: {i['created']} · sumber: run `{i['source_run']}` · {i['profile_version']}",
              "", "## Korpus dan denominator", "",
              f"- Sumber bukti: {c['evidence_source']}",
              f"- Lowongan: {c['denominators']['n_postings']} · kalimat: "
              f"{c['denominators']['n_sentences']} · unit bukti: {c['n_evidence']}",
              f"- Kelompok bukti: {c['grouping']['n_groups']} "
              f"({c['grouping']['status_distribution']})", "",
              "## Ringkasan", "",
              "| Status kelayakan | Jumlah |", "|---|---|"]
    lines += [f"| {_STATUS_ID.get(k, k)} | {v} |" for k, v in sorted(
        s["by_status"].items(), key=lambda kv: -_RANK.get(kv[0], 0))]
    if s["by_delta"]:
        lines += ["", "| Jenis rekomendasi delta | Jumlah |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in sorted(s["by_delta"].items())]
    lines += ["", f"## Rekomendasi (teratas {min(top, len(profile['recommendations']))})", "",
              "| # | Pernyataan kompetensi | Status | Delta | Lowongan | KKNI |", "|---|---|---|---|---|---|"]
    for r in profile["recommendations"][:top]:
        kkni = r["kkni"] or {}
        level = f"{kkni.get('level')}{'' if kkni.get('confident') else ' (?)'}" if kkni.get(
            "level") else ""
        lines.append(f"| {r['rank']} | {r['statement']['text']} | "
                     f"{_STATUS_ID.get(r['contract']['status'], r['contract']['status'])} | "
                     f"{(r['delta'] or {}).get('type', '')} | {r['n_postings']} | {level} |")
    lines += ["", "## Catatan perubahan", ""]
    if log["previous_version"] is None:
        lines.append("Versi pertama: belum ada catatan perubahan.")
    else:
        lines.append(f"Dibandingkan dengan versi {log['previous_version']}: "
                     f"{len(log['naik'])} naik, {len(log['turun'])} turun, "
                     f"{len(log['masuk'])} masuk, {len(log['keluar'])} keluar.")
        for name in ("naik", "turun", "masuk", "keluar"):
            for row in log[name][:15]:
                move = f" ({row['from']} → {row['to']})" if "from" in row else ""
                lines.append(f"- **{name}**{move}: {row['statement']} — {row['reason']}")
    lines += ["", "KKNI bertanda (?) adalah usulan dengan keyakinan rendah dan menunggu telaah ahli.",
              ""]
    return "\n".join(lines)
