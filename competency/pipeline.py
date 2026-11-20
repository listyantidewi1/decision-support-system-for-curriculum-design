"""The competency chain as one command: evidence -> groups -> statements ->
KKNI and okupasi -> status, written as an evidence package.

    # P1-style run (rules only, no LLM, free) on a sample of the sentence pool
    python -m competency.pipeline e0 --n-sentences 20000 --tag e0_pool20k

    # the same, Indonesian sentences only
    python -m competency.pipeline e0 --n-sentences 20000 --indonesian-only --tag e0_id

    # from the spans a stored multi-LLM run accepted
    python -m competency.pipeline ensemble --run-dir DATA/labeling/runs/pilot_100_<id> --split pilot_100

Output, under DATA/competency/runs/<tag>/ (local only; it quotes corpus
sentences):

    evidence.parquet    the typed evidence table every count is made from
    competencies.json   every competency with its statement, status, blockers,
                        KKNI, okupasi and its evidence package
    competencies.csv    the same, one line each, to read in a spreadsheet
    groups.json         every group, including those that did not become a
                        competency, with its status
    technology_profile.csv, work_behaviour_profile.csv
    report.json         counts at every step, versions, the denominators

    Until the gold standard and the model comparison exist, a run of this
    command is an ENGINEERING RUN: it shows the chain works on real text. The
    evidence comes from an extractor whose accuracy has not been measured, so
    the competencies it prints are not findings.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Mapping, Sequence

from . import COMPETENCY_VERSION, VALID
from .contract import ContractConfig, StudyFacts, decide
from .evidence import Evidence, by_tag, denominators, from_ensemble_run, from_extractor, profile
from .grouping import GroupingConfig, group_evidence
from .statement import (GROUNDING_VERSION, VERB_TABLE_VERSION, VERBS,
                        deterministic_statement, llm_statement, unchecked_llm_statement)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "DATA" / "competency" / "runs"
GS_DIR = REPO_ROOT / "DATA" / "gold_standard"
ENGINEERING_NOTICE = ("ENGINEERING RUN: the chain works on real text. The evidence comes from "
                      "an extractor whose accuracy has not been measured against the gold "
                      "standard, so these competencies are not findings.")


def run(evidence: Sequence[Evidence], sentences: Mapping[str, Mapping], out_dir: Path, *,
        evidence_source: str, grouping_cfg: GroupingConfig | None = None,
        contract_cfg: ContractConfig | None = None, facts: StudyFacts | None = None,
        clusterer: Callable | None = None, mapping: bool = True,
        llm_call: Callable | None = None, statements_for: Sequence[str] = (VALID,),
        posting_group: dict[str, str] | None = None,
        posting_month: Mapping[str, int] | None = None, temporal_cfg=None,
        use_persistence: bool = True, coverage: bool = True,
        curriculum_units: Sequence[dict] | None = None, curriculum_name: str = "kurikulum",
        curriculum_kkni_level: int | None = None, check_statements: bool = True,
        pipeline_name: str = "", embedder=None, v2_generator: dict | None = None,
        log=print) -> dict:
    out_dir = Path(out_dir).resolve()
    if GS_DIR.resolve() in out_dir.parents:
        raise ValueError("competency output must not be written under DATA/gold_standard/")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    den = denominators(sentences)
    contract_cfg = contract_cfg or ContractConfig()
    log(f"{len(evidence)} evidence units from {den['n_sentences']} sentences, "
        f"{den['n_postings']} postings")
    clusters_by_id: dict = {}
    groups, grouping_report = group_evidence(evidence, den["n_postings"], grouping_cfg, clusterer,
                                             posting_group, keep_clusters=clusters_by_id)
    from . import structure as ST
    structure_report = ST.annotate(groups, evidence)
    # One competency per cognitive level of each group (the group is the family).
    variants, split_report = ST.split_by_level(groups, evidence, den["n_postings"], grouping_cfg,
                                               posting_group)
    # The earlier LLM generator writes ONE description per family: it blends the
    # levels on purpose, so it describes the family and is not a competency of
    # one level. Descriptions already paid for are taken over, not bought again.
    v2_statements, v2_info = {}, None
    if v2_generator:
        from . import v2_generator as V2
        from .statement import Statement
        import dataclasses
        by_group = {g.group_id: g for g in groups}
        names = {f.name for f in dataclasses.fields(Statement)}
        reused = skipped = 0
        old_info: dict = {}
        if v2_generator.get("reuse_from"):
            src = Path(v2_generator["reuse_from"])
            if (src / "family_descriptions.json").is_file():
                rows = [(k, r["statement"], r["v2"], r["evidence_ids"]) for k, r in json.loads(
                    (src / "family_descriptions.json").read_text(encoding="utf-8")).items()]
            else:                            # runs made before families existed
                rows = [(c["evidence"]["group_id"], c["statement"], c["v2"],
                         c["evidence"]["evidence_ids"])
                        for c in json.loads((src / "competencies.json").read_text(encoding="utf-8"))
                        if c["statement"].get("method") == "v2-generator"]
            for gid, st_dict, extras, evidence_ids in rows:
                g = by_group.get(gid)
                # only for a family made of exactly the same evidence
                if g is None or sorted(g.evidence_ids) != sorted(evidence_ids):
                    skipped += 1
                    continue
                v2_statements[gid] = (Statement(**{k: v for k, v in st_dict.items() if k in names}),
                                      extras)
                reused += 1
            old_info = json.loads((src / "report.json").read_text(encoding="utf-8")).get(
                "v2_generator") or {}
            log(f"family descriptions reused from {src.name}: {reused} (not reused: {skipped})")
        new_info: dict = {}
        if v2_generator.get("max_groups"):
            soft = [r["text"] for r in profile(evidence, "SS", den["n_postings"], top=30)]
            produced = (v2_generator.get("generate") or V2.generate)(
                groups, clusters_by_id, max_groups=v2_generator["max_groups"],
                max_usd=v2_generator["max_usd"], soft_skills=soft, log=log,
                only=v2_generator.get("only"), skip=set(v2_statements))
            v2_statements.update(produced["statements"])
            new_info = produced["info"]
        v2_info = {**{k: v for k, v in old_info.items() if k in ("model", "generator_version",
                                                                 "bridge_version")},
                   **new_info,
                   "reused_from_run": (Path(v2_generator["reuse_from"]).name
                                       if v2_generator.get("reuse_from") else None),
                   "n_reused": reused, "n_not_reused_because_the_group_changed": skipped,
                   "n_new": len(new_info and produced["statements"] or {}),
                   "new_calls": new_info.get("n_groups_sent", 0),
                   "n_statements": len(v2_statements)}
    den["n_posting_groups"] = (len({posting_group.get(r.get("posting_id") or s.split("#")[0],
                                                      r.get("posting_id") or s.split("#")[0])
                                    for s, r in sentences.items()})
                               if posting_group else None)
    log(f"{len(groups)} groups: {grouping_report['status_distribution']}")

    family_of = {g.group_id: g for g in groups}
    competencies = []
    for g in variants:
        if g.status not in statements_for:
            continue
        extras = None
        if llm_call and not check_statements:
            st = unchecked_llm_statement(g, llm_call)      # ablation: no deterministic verifier
        elif llm_call:
            st = llm_statement(g, llm_call, contract_cfg.min_grounding)
        else:
            st = deterministic_statement(g)
        competencies.append({
            "competency_id": "comp-" + g.group_id, "statement": st.to_dict(),
            "stream": g.stream, "group_status": g.status,
            "n_postings": g.n_postings, "share_of_postings": g.share_of_postings,
            "n_posting_groups": g.n_posting_groups,
            "n_evidence": g.n_evidence, "kkni": None, "okupasi": None, "v2": extras,
            "cognitive": g.cognitive,
            "family": {"id": g.family_id, "label": family_of[g.family_id].label,
                       "n_postings": family_of[g.family_id].n_postings},
            "evidence": {"group_id": g.group_id, "label": g.label, "members": g.members,
                         "technologies": g.technologies,
                         "knowledge_context": g.knowledge_context,
                         "soft_skills": g.soft_skills, "work_behaviours": g.work_behaviours,
                         "knowledge": g.knowledge,
                         "verbs": g.verbs, "knowledge_objects": g.knowledge_objects,
                         "tag_counts": g.tag_counts, "cohesion": g.cohesion,
                         "evidence_ids": g.evidence_ids, "samples": g.samples}})
    # every competency names its relatives: the other levels of the same family
    relatives: dict[str, list[dict]] = {}
    for c in competencies:
        relatives.setdefault(c["family"]["id"], []).append(
            {"competency_id": c["competency_id"], "code": c["cognitive"]["code"],
             "level": c["cognitive"]["level"], "statement": c["statement"]["text"],
             "n_postings": c["n_postings"]})
    for c in competencies:
        fam = c["family"]
        fam["competencies"] = relatives[fam["id"]]
        described = v2_statements.get(fam["id"])
        fam["description"] = None if described is None else {
            "text": described[0].text, "method": "v2-generator",
            "rationale": (described[1] or {}).get("rationale"),
            "tools_named": (described[1] or {}).get("tools_named_in_statement", []),
            "note": "written by the earlier LLM generator for the whole family; it blends "
                    "the levels and is not a competency of one level"}
    mapping_report: dict = {"run": bool(mapping and competencies)}
    if mapping and competencies:
        from .mapping import map_kkni, map_okupasi
        mapping_report["kkni"] = map_kkni(competencies)
        mapping_report["okupasi"] = map_okupasi(competencies)
    for c in competencies:
        # The earlier design, kept: the LLM proposes a KKNI level from cognitive
        # complexity; the embedding step overrides it only when it is confident.
        llm_level = (c.get("v2") or {}).get("kkni_level")
        if llm_level and c.get("kkni") is not None:
            k = c["kkni"]
            k["nearest_descriptor_level"] = k.get("level")
            if not k.get("confident"):
                k["level"], k["source"] = int(llm_level), "llm_suggested"
            k["llm_level"] = int(llm_level)
        elif llm_level:
            c["kkni"] = {"level": int(llm_level), "source": "llm_suggested", "confident": False,
                         "llm_level": int(llm_level)}
    for c in competencies:
        mapped = None if c["okupasi"] is None else bool(c["okupasi"])
        c["contract"] = decide(group_status=c["group_status"], n_postings=c["n_postings"],
                               grounding=c["statement"]["grounding"], okupasi_mapped=mapped,
                               cfg=contract_cfg, facts=facts,
                               # a v2 statement reached here only by passing the generator's
                               # own evidence filter; the ones it dropped are in the report
                               grounding_passed=True if c.get("v2") else None)
    # ---- time analysis of each competency inside this run's own postings
    temporal_report: dict = {"run": False}
    if posting_month and competencies:
        from . import temporal as T
        by_evidence = {e.evidence_id: e.posting_id for e in evidence}
        run_postings = {(r.get("posting_id") or s.split("#")[0]) for s, r in sentences.items()}
        months = {p: m for p, m in posting_month.items() if p in run_postings}
        items = {c["competency_id"]: {by_evidence[i] for i in c["evidence"]["evidence_ids"]}
                 for c in competencies}
        t = T.analyse(items, months, temporal_cfg)
        rows = {r["item"]: r for r in t["items"]}
        for c in competencies:
            r = rows.get(c["competency_id"])
            c["temporal"] = None if r is None else {
                k: r[k] for k in ("shares", "current_share", "relative_slope", "direction",
                                  "trend", "q_value", "volatility", "cv", "persistence_weight")}
        temporal_report = {"run": True, **{k: t[k] for k in (
            "adequacy", "adequacy_reason", "trend_test_note", "windows", "n_items_analysed",
            "n_items_too_rare", "trend_counts", "volatility_counts", "top_n_turnover")},
            "rank_stability": T.rank_stability([r["shares"] for r in t["items"]]),
            "persistence_module_used_for_order": use_persistence}

    # ---- coverage against the official references, and against a curriculum
    coverage_report: dict = {"run": False}
    if coverage and mapping and competencies:
        from . import coverage as CV
        from .mapping import shared_embedder
        emb = embedder or shared_embedder()
        coverage_report = {"run": True,
                           "SKKNI": CV.audit(competencies, CV.load_skkni_units(), "SKKNI",
                                             embedder=emb, official=True)}
        if curriculum_units is not None:
            from . import delta as DL
            coverage_report[curriculum_name] = CV.audit(
                competencies, curriculum_units, curriculum_name, embedder=emb, official=False)
            coverage_report["delta"] = DL.apply(competencies, curriculum_name,
                                                curriculum_kkni_level)
    from .delta import priority_order
    if use_persistence:
        competencies = priority_order(competencies)
    else:
        competencies.sort(key=lambda c: (-c["n_postings"], c["competency_id"]))

    status_counts: dict[str, int] = {}
    for c in competencies:
        status_counts[c["contract"]["status"]] = status_counts.get(c["contract"]["status"], 0) + 1
    report = {
        "notice": ENGINEERING_NOTICE, "competency_version": COMPETENCY_VERSION,
        "evidence_source": evidence_source, "denominators": den,
        "n_evidence": len(evidence), "n_evidence_by_tag": by_tag(evidence),
        "grouping": grouping_report,
        "n_competencies": len(competencies), "status_distribution": status_counts,
        "statement_method": ("v2 generator for the most frequent groups, rule statements for "
                             "the rest" if v2_generator else
                             "llm with deterministic fallback" if llm_call else "deterministic-v1"),
        "v2_generator": v2_info, "structure": structure_report, "levels": split_report,
        "n_families": len(groups),
        "n_families_described_by_the_generator": len(
            {c["family"]["id"] for c in competencies if c["family"]["description"]}),
        "cognitive": {
            "n_with_a_level_read_from_the_postings": sum(
                1 for c in competencies if (c["cognitive"] or {}).get("level")),
            "n_without": sum(1 for c in competencies if not (c["cognitive"] or {}).get("level")),
            "by_level": dict(Counter((c["cognitive"] or {}).get("code") or "none"
                                     for c in competencies)),
            "n_rule_statements_whose_verb_is_on_another_level": sum(
                1 for c in competencies
                if c["statement"]["method"] == "deterministic-v1"
                and (c["cognitive"] or {}).get("level")
                and VERBS.get(c["statement"]["verb"]) != c["cognitive"]["level"])},
        "n_statements_with_default_verb": sum(
            1 for c in competencies if c["statement"]["verb_source"] == "default"),
        "verb_table_version": VERB_TABLE_VERSION, "grounding_version": GROUNDING_VERSION,
        "contract": {"min_postings": contract_cfg.min_postings,
                     "min_grounding": contract_cfg.min_grounding,
                     "thresholds_status": contract_cfg.thresholds_status},
        "mapping": mapping_report, "temporal": temporal_report, "coverage": coverage_report,
        "pipeline": pipeline_name, "statements_checked": check_statements,
        "n_llm_statements_rejected": sum(
            1 for c in competencies
            if any(n.startswith("LLM statement rejected") for n in c["statement"]["notes"])),
        "seconds": time.perf_counter() - t0,
        "not_built_yet": ["education-level distribution"],
    }

    def dump(name, obj):
        (out_dir / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str),
                                    encoding="utf-8")
    import pandas as pd
    pd.DataFrame([e.to_dict() for e in evidence]).to_parquet(out_dir / "evidence.parquet",
                                                             index=False)
    dump("competencies.json", competencies)
    dump("family_descriptions.json", {
        gid: {"statement": st.to_dict(), "v2": extras,
              "evidence_ids": sorted(family_of[gid].evidence_ids)}
        for gid, (st, extras) in v2_statements.items() if gid in family_of})
    dump("groups.json", [g.to_dict() for g in groups])
    dump("report.json", report)
    with (out_dir / "competencies.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["competency_id", "statement", "language", "cognitive_level", "family",
                    "postings_of_family", "share_of_family_postings", "levels_in_family",
                    "status", "blockers",
                    "n_postings", "n_employer_or_template_groups", "share_of_postings",
                    "grounding", "kkni_level", "kkni_confident", "okupasi_top",
                    "object_support", "verbs", "knowledge_acted_on", "technologies", "knowledge",
                    "soft_skills", "work_behaviours", "group_status", "stream"])
        for c in competencies:
            ok = (c["okupasi"] or [{}])[0].get("occupation_en", "") if c["okupasi"] else ""
            w.writerow([c["competency_id"], c["statement"]["text"], c["statement"]["language"],
                        " ".join(filter(None, [c["cognitive"]["code"], c["cognitive"]["level"]]))
                        or "not stated by the postings",
                        c["family"]["id"], c["family"]["n_postings"],
                        c["cognitive"]["share_of_family_postings"],
                        "; ".join(f"{r['code'] or 'none'} {r['n_postings']}"
                                  for r in c["family"]["competencies"]),
                        c["contract"]["status"],
                        " | ".join(c["contract"]["blockers"]), c["n_postings"],
                        c["n_posting_groups"],
                        f"{c['share_of_postings']:.4f}", f"{c['statement']['grounding']:.2f}",
                        (c["kkni"] or {}).get("level") or "",
                        "yes" if (c["kkni"] or {}).get("confident") else "no", ok,
                        (f"{c['evidence']['knowledge_objects'][0]['share']:.2f}"
                         if c["statement"].get("object_source") == "knowledge"
                         and c["evidence"]["knowledge_objects"] else ""),
                        ", ".join(f"{t['verb']} ({t['n_postings']})"
                                  for t in c["evidence"]["verbs"]),
                        ", ".join(f"{t['text']} ({t['n_postings']})"
                                  for t in c["evidence"]["knowledge_objects"]),
                        ", ".join(f"{t['text']} ({t['n_postings']})"
                                  for t in c["evidence"]["technologies"]),
                        ", ".join(f"{t['text']} ({t['share']:.0%})"
                                  for t in c["evidence"]["knowledge"]),
                        ", ".join(f"{t['text']} ({t['share']:.0%})"
                                  for t in c["evidence"]["soft_skills"]),
                        ", ".join(f"{t['text']} ({t['share']:.0%})"
                                  for t in c["evidence"]["work_behaviours"]),
                        c["group_status"], c["stream"]])
    for name, tag in (("technology_profile.csv", "TT"), ("work_behaviour_profile.csv", "WB")):
        with (out_dir / name).open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["text", "n_postings", "share_of_postings"])
            for r in profile(evidence, tag, den["n_postings"], top=200):
                w.writerow([r["text"], r["n_postings"], f"{r['share_of_postings']:.4f}"])
    # the cluster map of knowledge terms and technologies: a view for the
    # reader. A failure here must not lose a finished (possibly paid) run.
    try:
        from . import termmap as TM
        tm = TM.from_evidence(evidence, den["n_postings"])
        tm["source_run"], tm["evidence_source"] = out_dir.name, evidence_source
        dump("term_map.json", tm)
        report["term_map"] = {"version": tm["version"], "n_terms": tm["n_terms"],
                              "n_clusters": len(tm["clusters"])}
    except Exception as exc:                                    # noqa: BLE001
        log(f"TERM MAP NOT WRITTEN: {type(exc).__name__}: {exc}")
        report["term_map"] = {"error": f"{type(exc).__name__}: {exc}"}
    dump("report.json", report)
    report["out_dir"] = str(out_dir)
    return report


def _v2_options(args) -> dict | None:
    if args.statements != "v2":
        return None
    out: dict = {"reuse_from": args.reuse_v2_from}
    if args.max_usd > 0:                       # no cap given = no new call
        out.update(max_groups=args.v2_max_groups, max_usd=args.max_usd)
    if args.v2_families_from:
        comps = json.loads((Path(args.v2_families_from) / "competencies.json").read_text(
            encoding="utf-8"))
        out["only"] = {(c.get("family") or {}).get("id") or c["evidence"]["group_id"]
                       for c in comps if c["contract"]["status"] == "kandidat_tinjauan"}
    return out


def pool_sentences(n: int | None, seed: int = 7, indonesian_only: bool = False,
                   gs_dir: Path = GS_DIR, stratum: str | None = None) -> dict[str, dict]:
    """`n` WHOLE POSTINGS' worth of sentences would be the right unit for
    shares; a sentence sample is used here only to exercise the chain. Postings
    are sampled, and all their pool sentences taken, until n is reached."""
    import pandas as pd
    df = pd.read_parquet(gs_dir / "sentence_pool.parquet",
                         columns=["sentence_id", "posting_id", "sentence_text", "country_iso",
                                  "is_indonesian_sentence", "stratum"])
    if indonesian_only:
        df = df[df["is_indonesian_sentence"]]
    if stratum:
        df = df[df["stratum"] == stratum]
    if n is None:                                   # the whole pool (or the whole stratum)
        return {r.sentence_id: {"text": r.sentence_text, "posting_id": r.posting_id,
                                "country_iso": r.country_iso}
                for r in df.itertuples(index=False)}
    postings = df["posting_id"].drop_duplicates().sample(frac=1.0, random_state=seed)
    sizes = df.groupby("posting_id").size()
    keep, total = [], 0
    for pid in postings:
        keep.append(pid)
        total += int(sizes[pid])
        if total >= n:
            break
    sub = df[df["posting_id"].isin(set(keep))]
    return {r.sentence_id: {"text": r.sentence_text, "posting_id": r.posting_id,
                            "country_iso": r.country_iso}
            for r in sub.itertuples(index=False)}


def to_text(report: dict, competencies: list[dict], top: int = 15) -> str:
    lines = [report["notice"], "",
             f"evidence: {report['n_evidence']} units {report['n_evidence_by_tag']} from "
             f"{report['denominators']['n_postings']} postings",
             f"groups: {report['grouping']['n_groups']}  {report['grouping']['status_distribution']}",
             f"competencies: {report['n_competencies']}  {report['status_distribution']}", ""]
    for c in competencies[:top]:
        lines.append(f"  [{c['contract']['status']:<17}] {c['n_postings']:>4} postings "
                     f"({c['n_posting_groups']:>3} employers/templates)  "
                     f"{c['statement']['text'][:95]}")
    return "\n".join(lines)


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e0 = sub.add_parser("e0", help="rule extractor on a sample of the sentence pool (free)")
    e0.add_argument("--n-sentences", type=int, default=20000)
    e0.add_argument("--all", action="store_true", help="every posting of the pool, not a sample")
    e0.add_argument("--stratum", default=None, help="indonesia | asean | oecd")
    e0.add_argument("--core-min-postings", type=int, default=None,
                    help="two-stage grouping: cluster phrases seen in at least this many "
                         "postings, attach the rest (default 1 for a sample, 3 with --all)")
    e0.add_argument("--attach-threshold", type=float, default=GroupingConfig.attach_threshold)
    e0.add_argument("--indonesian-only", action="store_true")
    e0.add_argument("--lexicon", default="v2")
    e0.add_argument("--seed", type=int, default=7)
    en = sub.add_parser("ensemble", help="spans accepted by a stored multi-LLM run")
    en.add_argument("--run-dir", required=True)
    en.add_argument("--split", required=True)
    en.add_argument("--accept-majority", action="store_true")
    for p in (e0, en):
        p.add_argument("--statements", choices=["rules", "v2"], default="rules",
                       help="v2 = the earlier LLM generator (costs credit; needs --max-usd)")
        p.add_argument("--v2-max-groups", type=int, default=20)
        p.add_argument("--reuse-v2-from", default=None,
                       help="run folder whose generator statements are taken over (no new call)")
        p.add_argument("--v2-families-from", default=None,
                       help="run folder: new descriptions only for families that have a "
                            "'kandidat tinjauan' competency there")
        p.add_argument("--max-usd", type=float, default=0.0)
        p.add_argument("--tag", required=True)
        p.add_argument("--no-mapping", action="store_true")
        p.add_argument("--min-postings", type=int, default=ContractConfig.min_postings)
    args = ap.parse_args()

    if args.cmd == "e0":
        from fivetag.e0_rules import E0Rules
        sentences = pool_sentences(None if args.all else args.n_sentences, args.seed,
                                   args.indonesian_only, stratum=args.stratum)
        extractor = E0Rules(version=args.lexicon)
        evidence = from_extractor(extractor, sentences)
        core = args.core_min_postings or (3 if args.all else 1)
        grouping_cfg = GroupingConfig(core_min_postings=core,
                                      attach_threshold=args.attach_threshold)
        source = (f"E0 rules, lexicon {args.lexicon}, "
                  + ("whole pool" if args.all else f"pool sample seed {args.seed}")
                  + (f", stratum {args.stratum}" if args.stratum else ""))
    else:
        import pandas as pd
        df = pd.read_parquet(GS_DIR / f"{args.split}.parquet")
        sentences = {r.sentence_id: {"text": r.sentence_text, "posting_id": r.posting_id}
                     for r in df.itertuples(index=False)}
        evidence = from_ensemble_run(Path(args.run_dir), sentences, args.accept_majority)
        source = f"multi-LLM run {Path(args.run_dir).name}, split {args.split}"
        grouping_cfg = None
    posting_group = None
    groups_path = GS_DIR / "leakage_groups.parquet"
    if groups_path.is_file():
        import pandas as pd
        lg = pd.read_parquet(groups_path, columns=["posting_id", "group_id"])
        wanted = {r["posting_id"] for r in sentences.values()}
        lg = lg[lg["posting_id"].isin(wanted)]
        posting_group = dict(zip(lg["posting_id"].astype(str), lg["group_id"].astype(str)))
    from .temporal import posting_months
    report = run(evidence, sentences, RUNS_DIR / args.tag, evidence_source=source,
                 contract_cfg=ContractConfig(min_postings=args.min_postings),
                 grouping_cfg=grouping_cfg,
                 mapping=not args.no_mapping, posting_group=posting_group,
                 v2_generator=_v2_options(args),
                 posting_month=posting_months(), pipeline_name="P1" if args.cmd == "e0" else "")
    competencies = json.loads((Path(report["out_dir"]) / "competencies.json").read_text("utf-8"))
    print()
    print(to_text(report, competencies))
    print(f"\n-> {report['out_dir']}")


if __name__ == "__main__":
    main()
