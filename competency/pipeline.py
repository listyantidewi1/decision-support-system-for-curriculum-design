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
from pathlib import Path
from typing import Callable, Mapping, Sequence

from . import COMPETENCY_VERSION, VALID
from .contract import ContractConfig, StudyFacts, decide
from .evidence import Evidence, by_tag, denominators, from_ensemble_run, from_extractor, profile
from .grouping import GroupingConfig, group_evidence
from .statement import (GROUNDING_VERSION, VERB_TABLE_VERSION, deterministic_statement,
                        llm_statement, unchecked_llm_statement)

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
        pipeline_name: str = "", embedder=None, log=print) -> dict:
    out_dir = Path(out_dir).resolve()
    if GS_DIR.resolve() in out_dir.parents:
        raise ValueError("competency output must not be written under DATA/gold_standard/")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    den = denominators(sentences)
    contract_cfg = contract_cfg or ContractConfig()
    log(f"{len(evidence)} evidence units from {den['n_sentences']} sentences, "
        f"{den['n_postings']} postings")
    groups, grouping_report = group_evidence(evidence, den["n_postings"], grouping_cfg, clusterer,
                                             posting_group)
    den["n_posting_groups"] = (len({posting_group.get(r.get("posting_id") or s.split("#")[0],
                                                      r.get("posting_id") or s.split("#")[0])
                                    for s, r in sentences.items()})
                               if posting_group else None)
    log(f"{len(groups)} groups: {grouping_report['status_distribution']}")

    competencies = []
    for g in groups:
        if g.status not in statements_for:
            continue
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
            "n_evidence": g.n_evidence, "kkni": None, "okupasi": None,
            "evidence": {"group_id": g.group_id, "label": g.label, "members": g.members,
                         "technologies": g.technologies,
                         "knowledge_context": g.knowledge_context,
                         "tag_counts": g.tag_counts, "cohesion": g.cohesion,
                         "evidence_ids": g.evidence_ids, "samples": g.samples}})
    mapping_report: dict = {"run": bool(mapping and competencies)}
    if mapping and competencies:
        from .mapping import map_kkni, map_okupasi
        mapping_report["kkni"] = map_kkni(competencies)
        mapping_report["okupasi"] = map_okupasi(competencies)
    for c in competencies:
        mapped = None if c["okupasi"] is None else bool(c["okupasi"])
        c["contract"] = decide(group_status=c["group_status"], n_postings=c["n_postings"],
                               grounding=c["statement"]["grounding"], okupasi_mapped=mapped,
                               cfg=contract_cfg, facts=facts)
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
        "statement_method": "llm with deterministic fallback" if llm_call else "deterministic-v1",
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
    dump("groups.json", [g.to_dict() for g in groups])
    dump("report.json", report)
    with (out_dir / "competencies.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["competency_id", "statement", "language", "krathwohl", "status", "blockers",
                    "n_postings", "n_employer_or_template_groups", "share_of_postings",
                    "grounding", "kkni_level", "kkni_confident", "okupasi_top",
                    "technologies", "group_status", "stream"])
        for c in competencies:
            ok = (c["okupasi"] or [{}])[0].get("occupation_en", "") if c["okupasi"] else ""
            w.writerow([c["competency_id"], c["statement"]["text"], c["statement"]["language"],
                        c["statement"]["krathwohl"], c["contract"]["status"],
                        " | ".join(c["contract"]["blockers"]), c["n_postings"],
                        c["n_posting_groups"],
                        f"{c['share_of_postings']:.4f}", f"{c['statement']['grounding']:.2f}",
                        (c["kkni"] or {}).get("level") or "",
                        "yes" if (c["kkni"] or {}).get("confident") else "no", ok,
                        ", ".join(t["text"] for t in c["evidence"]["technologies"]),
                        c["group_status"], c["stream"]])
    for name, tag in (("technology_profile.csv", "TT"), ("work_behaviour_profile.csv", "WB")):
        with (out_dir / name).open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["text", "n_postings", "share_of_postings"])
            for r in profile(evidence, tag, den["n_postings"], top=200):
                w.writerow([r["text"], r["n_postings"], f"{r['share_of_postings']:.4f}"])
    report["out_dir"] = str(out_dir)
    return report


def pool_sentences(n: int, seed: int = 7, indonesian_only: bool = False,
                   gs_dir: Path = GS_DIR) -> dict[str, dict]:
    """`n` WHOLE POSTINGS' worth of sentences would be the right unit for
    shares; a sentence sample is used here only to exercise the chain. Postings
    are sampled, and all their pool sentences taken, until n is reached."""
    import pandas as pd
    df = pd.read_parquet(gs_dir / "sentence_pool.parquet",
                         columns=["sentence_id", "posting_id", "sentence_text", "country_iso",
                                  "is_indonesian_sentence"])
    if indonesian_only:
        df = df[df["is_indonesian_sentence"]]
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
    e0.add_argument("--indonesian-only", action="store_true")
    e0.add_argument("--lexicon", default="v2")
    e0.add_argument("--seed", type=int, default=7)
    en = sub.add_parser("ensemble", help="spans accepted by a stored multi-LLM run")
    en.add_argument("--run-dir", required=True)
    en.add_argument("--split", required=True)
    en.add_argument("--accept-majority", action="store_true")
    for p in (e0, en):
        p.add_argument("--tag", required=True)
        p.add_argument("--no-mapping", action="store_true")
        p.add_argument("--min-postings", type=int, default=ContractConfig.min_postings)
    args = ap.parse_args()

    if args.cmd == "e0":
        from fivetag.e0_rules import E0Rules
        sentences = pool_sentences(args.n_sentences, args.seed, args.indonesian_only)
        extractor = E0Rules(version=args.lexicon)
        evidence = from_extractor(extractor, sentences)
        source = f"E0 rules, lexicon {args.lexicon}, pool sample seed {args.seed}"
    else:
        import pandas as pd
        df = pd.read_parquet(GS_DIR / f"{args.split}.parquet")
        sentences = {r.sentence_id: {"text": r.sentence_text, "posting_id": r.posting_id}
                     for r in df.itertuples(index=False)}
        evidence = from_ensemble_run(Path(args.run_dir), sentences, args.accept_majority)
        source = f"multi-LLM run {Path(args.run_dir).name}, split {args.split}"
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
                 mapping=not args.no_mapping, posting_group=posting_group,
                 posting_month=posting_months(), pipeline_name="P1" if args.cmd == "e0" else "")
    competencies = json.loads((Path(report["out_dir"]) / "competencies.json").read_text("utf-8"))
    print()
    print(to_text(report, competencies))
    print(f"\n-> {report['out_dir']}")


if __name__ == "__main__":
    main()
