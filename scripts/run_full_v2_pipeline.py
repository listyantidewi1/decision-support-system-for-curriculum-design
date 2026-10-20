"""
run_full_v2_pipeline.py

End-to-end driver for the v2 pipeline. Reads Phase 1 output, runs every
v2 stage in sequence, and emits a dashboard-ready directory:

    Phase 1 (advanced_skills.csv + advanced_knowledge.csv)
        ↓
    Phase 2.1: skill_clusterer.cluster_skills        (cohesion >= 0.55)
        ↓
    Phase 2.2: competency_generator_v2               (openrouter:gpt-5.4-mini default)
        ↓
    Phase 2.5: grounding gate (>= 0.80)              (auto in generate_competencies_v2)
        ↓
    Phase 2.5b: text-quality post-check              (catches "religious" bug class)
        ↓
    Phase 2.3: KKNI labeller (SBERT)                 (overrides LLM kkni_level)
        ↓
    Phase 2.4: education aggregator                  (joins jobs_metadata.csv)
        ↓
    Output: results/competency_v2_pipeline_<tag>/
        competencies.json           — final list (passing all gates)
        competencies_failed.json    — dropped competencies + reasons
        batch_reasonings.json
        clusters.json
        pipeline_report.json        — aggregate metrics from every stage

Usage:
    python scripts/run_full_v2_pipeline.py
        [--phase1-input results.old2]
        [--jobs-metadata DATA/preprocessing/data_prepared_n1k/jobs_metadata.csv]
        [--model openrouter:gpt-5.4-mini]
        [--tag latest]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.test_clustering_on_real_data import load_real_items

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger("v2_pipeline")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-input", default="results.old2",
                        help="dir containing advanced_skills.csv + advanced_knowledge.csv")
    parser.add_argument("--jobs-metadata",
                        default="DATA/preprocessing/data_prepared_n1k/jobs_metadata.csv")
    parser.add_argument("--model", default="openrouter:gpt-5.4-mini")
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--cohesion-threshold", type=float, default=0.55)
    parser.add_argument("--grounding-threshold", type=float, default=0.80)
    parser.add_argument("--skip-text-quality", action="store_true")
    parser.add_argument("--skip-evidence-repair", action="store_true")
    parser.add_argument("--evidence-repair-no-sbert", action="store_true")
    parser.add_argument("--evidence-repair-statement-threshold", type=float, default=0.45)
    parser.add_argument("--evidence-repair-skill-threshold", type=float, default=0.45)
    parser.add_argument("--evidence-repair-min-skills", type=int, default=2)
    parser.add_argument("--evidence-repair-max-sentences", type=int, default=8)
    parser.add_argument("--skip-repair-stage", action="store_true",
                        help="Skip the precision/redundancy/recall repair stage.")
    parser.add_argument("--skip-scientific-audits", action="store_true",
                        help="Skip read-only stage/extraction/sentence/cluster/theme audits.")
    parser.add_argument("--no-enforce-extraction-typing", action="store_true",
                        help="Ablation only: skip the deterministic post-extraction typing gate before clustering.")
    parser.add_argument("--skip-research-packages", action="store_true",
                        help="Skip automatic evidence, DACUM/CBC, validation, and figures packages.")
    parser.add_argument("--skip-targeted-recovery", action="store_true",
                        help="Run deterministic repair/dedup only; do not call the LLM for high-support failed clusters.")
    parser.add_argument("--targeted-recovery-model", default="openrouter:gpt-5.4-mini",
                        help="Model used only for focused repair of high-support failed technical clusters.")
    parser.add_argument("--max-recovery-candidates", type=int, default=12,
                        help="Maximum high-support failed competencies to send to targeted recovery.")
    parser.add_argument("--skip-kkni-labeller", action="store_true")
    parser.add_argument("--skip-education", action="store_true")
    parser.add_argument(
        "--enable-role-context", action="store_true",
        help="Phase 2.1 Tier 2: per-item role-context embedding prefix "
             "(skill in role '{job_title}': ...). Requires jobs_metadata.",
    )
    parser.add_argument(
        "--enable-cluster-refinement", action="store_true",
        help="Phase 2.1 v8 sprint: LLM-as-arbiter cluster refinement pass "
             "after agglomerative recovery. Re-arranges cluster members and "
             "noise items based on LLM verdicts; commits moves only when "
             "cluster cohesion improves monotonically.",
    )
    parser.add_argument(
        "--refinement-model", default="openrouter:gpt-5.4-mini",
        help="Model for the cluster-refinement arbiter (routed via llm_client_router).",
    )
    parser.add_argument(
        "--enable-it-relevance-filter", action="store_true",
        help="v9 sprint: classify each source job + each source sentence as "
             "IT/software-engineering content (yes/no via LLM). Drops items "
             "whose job or sentence is non-IT BEFORE clustering. Targets "
             "the v8.1 climbing/pizza-assembly contamination failure mode.",
    )
    parser.add_argument(
        "--raw-jobs-csv", default="job_scraping/output/english_jobs.csv",
        help="Path to the raw scraped job-postings CSV (with `id` and "
             "`description` columns). Used by the IT-relevance filter for "
             "job-level classification.",
    )
    parser.add_argument(
        "--skip-corpus-audit", action="store_true",
        help="Skip the read-only corpus lineage/validity audit. Not recommended "
             "for research or stakeholder-facing runs.",
    )
    parser.add_argument(
        "--corpus-dataset-id", default="",
        help="Dataset id for the corpus audit. Defaults to --tag.",
    )
    parser.add_argument(
        "--corpus-raw-csv", default="",
        help="Raw job CSV for lineage audit. When omitted, the audit infers the "
             "best DATA/preprocessing/*.csv match by job_id overlap.",
    )
    parser.add_argument(
        "--prepared-dir", default="",
        help="Prepared data directory for lineage audit. Defaults to the parent "
             "of --jobs-metadata.",
    )
    parser.add_argument(
        "--skill-llm-input", default="skill_llm_input_n1k.jsonl",
        help="Skill-LLM JSONL input used for Kaggle inference, if available.",
    )
    parser.add_argument(
        "--skill-llm-output", default="skill_llm_extractions.jsonl",
        help="Skill-LLM JSONL output downloaded from Kaggle, if available.",
    )
    parser.add_argument(
        "--corpus-audit-output-dir", default="",
        help="Where to write the corpus audit. Defaults to results/corpus_audits/<dataset_id>.",
    )
    parser.add_argument(
        "--translate-output-to-id", action="store_true",
        help="v9 sprint: after all phases complete, translate each "
             "competency's user-facing fields (title, description, "
             "rationale, soft_skills_description, related_skills, "
             "soft_skills_required) into Bahasa Indonesia and persist "
             "as {field}_id keys. Powers the dashboard language toggle.",
    )
    parser.add_argument(
        "--translation-model", default="openrouter:gpt-5.4-mini",
        help="Model for EN→ID competency translation. Default is the "
             "same OpenRouter model already used for generation.",
    )
    parser.add_argument(
        "--translate-input-sentences", action="store_true",
        help="v9 sprint: detect language of each loaded item's "
             "sentence_text and translate Indonesian ones to English "
             "BEFORE downstream filters and clustering. Keeps the "
             "original Indonesian text in `sentence_text_original` for "
             "provenance. No-op when items are already English.",
    )
    # Occupation (SKKNI) mapping defaults ON 2026-05-14 — curriculum
    # designers consistently need the Indonesian occupational-standard
    # anchor on every competency, and v10c shipped without it because the
    # opt-in flag was missing from the re-run command. The skema okupasi
    # mapping is now part of the standard pipeline contract; pass
    # --disable-occupation-mapping to skip for ablation runs.
    parser.add_argument(
        "--enable-occupation-mapping", action=argparse.BooleanOptionalAction,
        default=True,
        help="Map each competency to the top-K Indonesian SKKNI occupations "
             "(skema okupasi) via multilingual SBERT cosine. Populates the "
             "`occupation_matches` field used by the dashboard's occupation "
             "filter + per-competency chips. Default: enabled. Use "
             "--no-enable-occupation-mapping to skip.",
    )
    parser.add_argument(
        "--skkni-csv", default="DATA/skema_okupasi_indonesia_with_desc.csv",
        help="Path to the enriched SKKNI CSV (with English Description "
             "column). Defaults to DATA/skema_okupasi_indonesia_with_desc.csv.",
    )
    parser.add_argument(
        "--occupation-top-k", type=int, default=3,
        help="Number of top-K occupation matches to keep per competency.",
    )
    parser.add_argument(
        "--occupation-min-cosine", type=float, default=0.45,
        help="Minimum cosine for a SKKNI occupation match to be retained. "
             "Below this, the match is discarded (a competency with no "
             "matches above threshold gets an empty occupation_matches list, "
             "which flags it as not anchored in any Indonesian occupational "
             "standard).",
    )
    parser.add_argument(
        "--occupation-sectors", default="",
        help="Comma-separated list of SKKNI sectors to consider for "
             "occupation mapping. Default is the ICT sector only "
             "(Teknologi Informasi dan Komunikasi). Pass an empty string "
             "(default) to use the built-in IT-only whitelist; pass 'all' "
             "to disable filtering; or pass a specific list like "
             "'Teknologi Informasi dan Komunikasi,Industri Kreatif dan Komunikasi' "
             "for IT + creative industries.",
    )
    parser.add_argument(
        "--allow-invalid-cluster-generation",
        action="store_true",
        help="Ablation only: allow split/mixed/support/out-of-domain clusters to reach competency generation.",
    )
    args = parser.parse_args()

    in_dir = (PROJECT_ROOT / args.phase1_input) if not Path(args.phase1_input).is_absolute() else Path(args.phase1_input)
    jobs_meta = (PROJECT_ROOT / args.jobs_metadata) if not Path(args.jobs_metadata).is_absolute() else Path(args.jobs_metadata)
    out_dir = PROJECT_ROOT / "results" / f"competency_v2_pipeline_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir = Path(args.prepared_dir) if args.prepared_dir else jobs_meta.parent
    if not prepared_dir.is_absolute():
        prepared_dir = PROJECT_ROOT / prepared_dir
    raw_csv = Path(args.corpus_raw_csv) if args.corpus_raw_csv else None
    if raw_csv is not None and not raw_csv.is_absolute():
        raw_csv = PROJECT_ROOT / raw_csv
    skill_llm_input = Path(args.skill_llm_input) if args.skill_llm_input else None
    if skill_llm_input is not None and not skill_llm_input.is_absolute():
        skill_llm_input = PROJECT_ROOT / skill_llm_input
    skill_llm_output = Path(args.skill_llm_output) if args.skill_llm_output else None
    if skill_llm_output is not None and not skill_llm_output.is_absolute():
        skill_llm_output = PROJECT_ROOT / skill_llm_output

    pipeline_report = {
        "tag": args.tag,
        "model": args.model,
        "phase1_input": str(in_dir),
        "jobs_metadata": str(jobs_meta),
        "stages": {},
    }
    t_total = time.time()

    # -------------------- Sprint 0: corpus lineage / validity audit --------------------
    corpus_report = {}
    corpus_report_path = None
    if not args.skip_corpus_audit:
        try:
            from scripts.audit_corpus_lineage import audit_corpus, write_audit_outputs

            dataset_id = args.corpus_dataset_id or args.tag
            audit_out_dir = (
                PROJECT_ROOT / "results" / "corpus_audits" / dataset_id
                if not args.corpus_audit_output_dir
                else Path(args.corpus_audit_output_dir)
            )
            if not audit_out_dir.is_absolute():
                audit_out_dir = PROJECT_ROOT / audit_out_dir

            corpus_report, coverage_rows, distribution_rows, quality_rows = audit_corpus(
                dataset_id=dataset_id,
                raw_csv=raw_csv,
                prepared_dir=prepared_dir,
                phase1_dir=in_dir,
                skill_llm_input=skill_llm_input,
                skill_llm_output=skill_llm_output,
                jobs_metadata=jobs_meta,
            )
            write_audit_outputs(audit_out_dir, corpus_report, coverage_rows, distribution_rows, quality_rows)
            corpus_report_path = audit_out_dir / "corpus_lineage_report.json"
            pipeline_report["corpus_lineage_report"] = str(corpus_report_path)
            pipeline_report["stages"]["corpus_lineage_audit"] = {
                "schema_version": corpus_report.get("schema_version"),
                "dataset_id": dataset_id,
                "report_path": str(corpus_report_path),
                "corpus_validity_status": corpus_report.get("corpus_validity_status"),
                "corpus_country": corpus_report.get("corpus_country"),
                "corpus_language": corpus_report.get("corpus_language"),
                "metadata_unique_job_ids": corpus_report.get("metadata_unique_job_ids"),
                "phase1_unique_job_ids": corpus_report.get("phase1_unique_job_ids"),
            }
            logger.info(
                "Corpus audit: status=%s country=%s language=%s report=%s",
                corpus_report.get("corpus_validity_status"),
                corpus_report.get("corpus_country"),
                corpus_report.get("corpus_language"),
                corpus_report_path,
            )
        except Exception as e:
            logger.warning("Corpus audit failed; recommendations will be exploratory unless audited later (%s)", e)
            pipeline_report["stages"]["corpus_lineage_audit"] = {
                "skipped": False,
                "failed": True,
                "error": str(e),
            }
    else:
        pipeline_report["stages"]["corpus_lineage_audit"] = {"skipped": True}

    # -------------------- Phase 1 load --------------------
    logger.info("PHASE 1: loading SkillItem + KnowledgeItem from %s", in_dir)
    items = load_real_items(
        in_dir,
        jobs_metadata_csv=jobs_meta if jobs_meta.exists() else None,
    )
    pipeline_report["stages"]["phase1_load"] = {
        "n_items": len(items),
        "n_skills": sum(1 for it in items if hasattr(it, "type")),
        "n_knowledge": sum(1 for it in items if not hasattr(it, "type")),
    }

    # -------------------- Phase 1.4: ID→EN translation (v9 sprint) --------------------
    scientific_audit_reports: dict = {}
    cluster_validity_report: dict = {}
    if not args.skip_scientific_audits:
        try:
            from pipeline_scientific_audits import (
                audit_domain_filter,
                audit_extraction_quality,
                audit_sentence_filter,
            )

            extraction_audit = audit_extraction_quality(in_dir, out_dir)
            sentence_filter_audit = audit_sentence_filter(prepared_dir, skill_llm_input, out_dir)
            domain_filter_audit = audit_domain_filter(in_dir, out_dir)
            scientific_audit_reports.update({
                "extraction_quality": extraction_audit,
                "sentence_filter": sentence_filter_audit,
                "domain_filter": domain_filter_audit,
            })
            pipeline_report["stages"]["scientific_audits_phase1"] = {
                "schema_version": extraction_audit.get("schema_version"),
                "extraction_quality_report": str(out_dir / "extraction_quality_audit.json"),
                "sentence_filter_report": str(out_dir / "sentence_filter_audit.json"),
                "domain_filter_audit": str(out_dir / "domain_filter_audit.csv"),
                "n_phase1_spans": extraction_audit.get("n_phase1_spans"),
                "n_not_hard_technical": extraction_audit.get("n_not_hard_technical"),
                "jobs_lost_all_sentences": sentence_filter_audit.get("n_jobs_lost_all_sentences_prepared_to_filtered"),
            }
        except Exception as e:
            logger.warning("Scientific Phase-1 audits failed; continuing pipeline (%s)", e)
            pipeline_report["stages"]["scientific_audits_phase1"] = {
                "failed": True,
                "error": str(e),
            }
    else:
        pipeline_report["stages"]["scientific_audits_phase1"] = {"skipped": True}

    # -------------------- Phase 1.3b: deterministic extraction typing gate --------------------
    # Protects both new and legacy Phase-1 folders. New converter outputs carry
    # post_extraction_type columns, while old folders are classified here.
    if not args.no_enforce_extraction_typing:
        from extraction_typing import (
            apply_extraction_type_gate_to_items,
            write_extraction_type_gate_audit,
        )

        before_items = len(items)
        items, extraction_gate_report, extraction_gate_rows = apply_extraction_type_gate_to_items(items)
        write_extraction_type_gate_audit(out_dir, extraction_gate_report, extraction_gate_rows)
        pipeline_report["stages"]["phase1_3b_extraction_type_gate"] = {
            "schema_version": extraction_gate_report.get("schema_version"),
            "report_path": str(out_dir / "extraction_type_gate_audit.json"),
            "csv_path": str(out_dir / "extraction_type_gate_audit.csv"),
            "n_items_before": before_items,
            "n_items_after": extraction_gate_report.get("n_items_after"),
            "n_items_dropped": extraction_gate_report.get("n_items_dropped"),
            "n_items_demoted_to_soft": extraction_gate_report.get("n_items_demoted_to_soft"),
            "n_unsafe_hard_generation_items_after": extraction_gate_report.get("n_unsafe_hard_generation_items_after"),
            "unsafe_hard_generation_types_after": extraction_gate_report.get("unsafe_hard_generation_types_after", {}),
        }
        logger.info(
            "Phase 1.3b: extraction type gate kept %d/%d items, dropped %d, demoted %d to soft",
            len(items),
            before_items,
            int(extraction_gate_report.get("n_items_dropped") or 0),
            int(extraction_gate_report.get("n_items_demoted_to_soft") or 0),
        )
    else:
        pipeline_report["stages"]["phase1_3b_extraction_type_gate"] = {
            "skipped": True,
            "reason": "--no-enforce-extraction-typing",
        }

    # When the loaded items have Indonesian sentence_text, translate each
    # unique sentence to English so downstream filters / Skill-LLM (if
    # invoked) / SBERT operate on English content. Original Indonesian is
    # preserved as `sentence_text_original` for provenance display.
    if args.translate_input_sentences and items:
        from translator import langdetect_lang, translate_to_english
        unique_texts: dict = {}
        for it in items:
            stxt = (getattr(it, "sentence_text", "") or "").strip()
            if stxt:
                unique_texts.setdefault(stxt, []).append(it)
        translated_map: dict = {}
        n_id, n_translated = 0, 0
        for stxt in unique_texts:
            lang = langdetect_lang(stxt)
            if lang == "id":
                n_id += 1
                en = translate_to_english(stxt)
                if en and en != stxt:
                    translated_map[stxt] = en
                    n_translated += 1
        if translated_map:
            for stxt, en in translated_map.items():
                for it in unique_texts[stxt]:
                    setattr(it, "sentence_text_original", stxt)
                    it.sentence_text = en
        logger.info(
            "Phase 1.4: ID→EN translation — %d unique sentences scanned, "
            "%d detected ID, %d translated",
            len(unique_texts), n_id, n_translated,
        )
        pipeline_report["stages"]["phase1_4_translation"] = {
            "n_unique_sentences": len(unique_texts),
            "n_detected_indonesian": n_id,
            "n_translated": n_translated,
        }

    # -------------------- Phase 1.5: IT-relevance gate (v9 sprint) --------------------
    # Drops items whose source JOB is not IT/software-engineering, then drops
    # items whose source SENTENCE is not IT content. Targets the v8.1
    # climbing/pizza-assembly contamination failure where boilerplate
    # physical-requirements text reached the clusterer.
    if args.enable_it_relevance_filter:
        from pipeline_it_filter import apply_it_relevance_filter
        raw_jobs_path = (PROJECT_ROOT / args.raw_jobs_csv) if not Path(args.raw_jobs_csv).is_absolute() else Path(args.raw_jobs_csv)
        items, it_audit = apply_it_relevance_filter(
            items,
            raw_jobs_csv=raw_jobs_path if raw_jobs_path.exists() else None,
            cache_dir=PROJECT_ROOT / "cache" / "it_relevance_filter",
        )
        pipeline_report["stages"]["phase1_5_it_relevance"] = it_audit["summary"]
        (out_dir / "it_relevance_audit.json").write_text(
            json.dumps(it_audit, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        logger.info(
            "Phase 1.5: IT-relevance gate kept %d / %d items (jobs %d/%d, sentences %d/%d)",
            it_audit["summary"]["n_items_kept"], it_audit["summary"]["n_items_in"],
            it_audit["summary"]["n_jobs_it"], it_audit["summary"]["n_jobs_total"],
            it_audit["summary"]["n_sentences_it"], it_audit["summary"]["n_sentences_total"],
        )

    # -------------------- Phase 2.1: clustering --------------------
    logger.info(
        "PHASE 2.1: clustering (cohesion >= %.2f, role_context=%s, refinement=%s)",
        args.cohesion_threshold, args.enable_role_context, args.enable_cluster_refinement,
    )
    from clustering import ClusteringConfig, cluster_skills
    cl_cfg = ClusteringConfig(
        cohesion_threshold=args.cohesion_threshold,
        enable_role_context=args.enable_role_context,
        enable_cluster_refinement=args.enable_cluster_refinement,
        refinement_model=args.refinement_model,
    )
    clusters, cl_report = cluster_skills(items, config=cl_cfg)
    n_hpk = sum(1 for c in clusters if c.stream == "hard_plus_knowledge")
    logger.info("Phase 2.1: %d clusters total (%d hard+knowledge for Phase 2.2)", len(clusters), n_hpk)
    pipeline_report["stages"]["phase2_1_clustering"] = cl_report.to_dict()

    cluster_dicts = [c.to_dict() for c in clusters]
    if not args.skip_scientific_audits:
        try:
            from pipeline_scientific_audits import audit_cluster_validity

            cluster_validity_report = audit_cluster_validity(cluster_dicts, out_dir)
            for cd in cluster_dicts:
                audit = (cluster_validity_report.get("cluster_validity_by_id") or {}).get(cd.get("id"), {})
                cd.update({
                    "cluster_validity_status": audit.get("cluster_validity_status", ""),
                    "cluster_validity_flags": audit.get("cluster_validity_flags", []),
                    "domain_families": audit.get("domain_families", []),
                    "primary_domain_family": audit.get("primary_domain_family", ""),
                })
            pipeline_report["stages"]["phase2_1_cluster_validity"] = {
                "schema_version": cluster_validity_report.get("schema_version"),
                "report_path": str(out_dir / "cluster_validity_report.json"),
                "status_distribution": cluster_validity_report.get("status_distribution", {}),
                "n_split_candidates": cluster_validity_report.get("n_split_candidates", 0),
            }
        except Exception as e:
            logger.warning("Cluster validity audit failed; continuing pipeline (%s)", e)
            pipeline_report["stages"]["phase2_1_cluster_validity"] = {
                "failed": True,
                "error": str(e),
            }
    else:
        pipeline_report["stages"]["phase2_1_cluster_validity"] = {"skipped": True}

    clusters_for_generation = list(clusters)
    if cluster_validity_report and not args.allow_invalid_cluster_generation:
        validity_by_id = cluster_validity_report.get("cluster_validity_by_id") or {}
        kept_clusters = []
        gate_rows = []
        for c, cd in zip(clusters, cluster_dicts):
            cluster_id = str(cd.get("id") or "")
            status = (validity_by_id.get(cluster_id) or {}).get("cluster_validity_status", "")
            sent = status == "valid_cluster"
            decision = "sent_to_generation" if sent else "blocked_from_generation"
            reason = "" if sent else f"cluster_validity_status={status or 'missing'}"
            if sent:
                kept_clusters.append(c)
            cd["generation_gate_decision"] = decision
            cd["generation_gate_reason"] = reason
            gate_rows.append({
                "cluster_id": cluster_id,
                "summary_label": cd.get("summary_label", ""),
                "stream": cd.get("stream", ""),
                "cluster_validity_status": status,
                "cluster_validity_flags": " | ".join(cd.get("cluster_validity_flags") or []),
                "n_items": cd.get("n_items", 0),
                "n_unique_jobs": cd.get("n_unique_jobs", 0),
                "sent_to_generation": sent,
                "gate_decision": decision,
                "gate_reason": reason,
            })
        clusters_for_generation = kept_clusters
        with (out_dir / "cluster_generation_gate.csv").open("w", encoding="utf-8-sig", newline="") as f:
            fields = [
                "cluster_id", "summary_label", "stream", "cluster_validity_status",
                "cluster_validity_flags", "n_items", "n_unique_jobs",
                "sent_to_generation", "gate_decision", "gate_reason",
            ]
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(gate_rows)
        blocked_rows = [r for r in gate_rows if not r["sent_to_generation"]]
        blocked_by_status: dict[str, int] = {}
        for row in blocked_rows:
            status = row["cluster_validity_status"] or "missing"
            blocked_by_status[status] = blocked_by_status.get(status, 0) + 1
        cluster_gate_report = {
            "schema_version": "2026-06-23.upstream_evidence_repair.v1",
            "policy": "only valid_cluster reaches competency generation",
            "allow_invalid_cluster_generation": False,
            "n_clusters_total": len(gate_rows),
            "n_sent_to_generation": len(clusters_for_generation),
            "n_blocked_from_generation": len(blocked_rows),
            "blocked_by_status": blocked_by_status,
            "blocked_cluster_ids": [r["cluster_id"] for r in blocked_rows],
            "csv_path": str(out_dir / "cluster_generation_gate.csv"),
        }
        (out_dir / "cluster_generation_gate.json").write_text(
            json.dumps(cluster_gate_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        pipeline_report["stages"]["phase2_1b_cluster_generation_gate"] = {
            "schema_version": cluster_gate_report["schema_version"],
            "report_path": str(out_dir / "cluster_generation_gate.json"),
            "csv_path": str(out_dir / "cluster_generation_gate.csv"),
            "n_clusters_total": cluster_gate_report["n_clusters_total"],
            "n_sent_to_generation": cluster_gate_report["n_sent_to_generation"],
            "n_blocked_from_generation": cluster_gate_report["n_blocked_from_generation"],
            "blocked_by_status": blocked_by_status,
        }
        logger.info(
            "Phase 2.1b: cluster generation gate sent %d/%d clusters, blocked %d",
            len(clusters_for_generation),
            len(gate_rows),
            len(blocked_rows),
        )
    elif args.allow_invalid_cluster_generation:
        for cd in cluster_dicts:
            cd["generation_gate_decision"] = "sent_to_generation"
            cd["generation_gate_reason"] = "--allow-invalid-cluster-generation"
        pipeline_report["stages"]["phase2_1b_cluster_generation_gate"] = {
            "skipped": True,
            "reason": "--allow-invalid-cluster-generation",
        }
    else:
        pipeline_report["stages"]["phase2_1b_cluster_generation_gate"] = {
            "skipped": True,
            "reason": "cluster_validity_report_unavailable",
        }

    # Persist clusters
    (out_dir / "clusters.json").write_text(
        json.dumps(cluster_dicts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # v8.1: persist the full noise audit alongside clusters.json so the
    # dashboard's Pipeline-audit page can load it as a separate file.
    if cl_report.noise_audit:
        (out_dir / "noise_audit.json").write_text(
            json.dumps(cl_report.noise_audit, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "Phase 2.1: noise_audit.json written (%d items dropped, by stage: %s)",
            cl_report.noise_audit.get("n_items_dropped_total", 0),
            cl_report.noise_audit.get("by_stage", {}),
        )

    # -------------------- Phase 2.2: competency generation + 2.5 grounding gate --------------------
    logger.info("PHASE 2.2: generating competencies via %s (Phase 2.5 grounding gate auto-applied)", args.model)
    from competency_v2_schema import GeneratorConfig
    from competency_generator_v2 import generate_competencies_v2
    g_cfg = GeneratorConfig(model=args.model)

    # v8 sprint Phase 2.A: load jobs_metadata for role distribution in the
    # generator prompt. Education-stage demand stays in Phase 2.4 only —
    # NOT passed to the generator (KKNI placement must be cognitive-complexity
    # only).
    job_titles_for_generator: dict = {}
    if jobs_meta.exists():
        import csv as _csv
        try:
            with open(jobs_meta, encoding="utf-8-sig") as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    jid = (row.get("job_id") or "").strip()
                    title = (row.get("title") or "").strip()
                    if jid and title:
                        job_titles_for_generator[jid] = title
            logger.info(
                "Phase 2.2: loaded %d job_id->title mappings for role distribution",
                len(job_titles_for_generator),
            )
        except Exception as e:
            logger.warning("Phase 2.2: jobs_metadata load failed (%s); proceeding without role distribution", e)

    competencies, batch_reasonings, hallucination_dropped = generate_competencies_v2(
        clusters=clusters_for_generation,
        config=g_cfg,
        apply_grounding_gate=True,
        job_titles=job_titles_for_generator or None,
    )
    logger.info(
        "Phase 2.2: %d competencies passed anti-hallucination filter "
        "(%d dropped for fabrication)",
        len(competencies), len(hallucination_dropped),
    )
    pipeline_report["stages"]["phase2_2_generation"] = {
        "n_competencies": len(competencies),
        "n_batch_reasonings": len(batch_reasonings),
        "n_hallucination_dropped": len(hallucination_dropped),
        "model": args.model,
        "n_clusters_input": len(clusters_for_generation),
        "contract": "demand-rank + anti-hallucination filter (was: grounding gate, until 2026-05-13b)",
    }
    if cluster_validity_report and competencies:
        try:
            from pipeline_scientific_audits import annotate_cluster_validity

            annotate_cluster_validity(competencies, cluster_validity_report)
            annotate_cluster_validity(hallucination_dropped, cluster_validity_report)
        except Exception as e:
            logger.warning("Could not annotate generation outputs with cluster validity (%s)", e)

    # -------------------- Phase 2.5b: text-quality post-check --------------------
    failed_text_quality = []
    failed_evidence_repair = []
    failed_repair_stage = []
    repair_stage_report = {}
    recovered_source_ids: set[str] = set()
    er_cfg = None
    if not args.skip_evidence_repair and competencies:
        logger.info("PHASE 2.6: evidence linker / repair pass")
        from competency_evidence_repair import EvidenceRepairConfig, repair_competencies
        er_cfg = EvidenceRepairConfig(
            disable_sbert=args.evidence_repair_no_sbert,
            statement_threshold=args.evidence_repair_statement_threshold,
            skill_threshold=args.evidence_repair_skill_threshold,
            min_supported_skills=args.evidence_repair_min_skills,
            max_supporting_sentences=args.evidence_repair_max_sentences,
            final_grounding_threshold=args.grounding_threshold,
        )
        competencies, failed_evidence_repair, er_report = repair_competencies(
            competencies, er_cfg
        )
        pipeline_report["stages"]["phase2_6_evidence_repair"] = er_report.to_dict()
        logger.info(
            "Phase 2.6: evidence repair kept %d, dropped %d; mean grounding %.3f -> %.3f",
            er_report.n_repaired,
            er_report.n_dropped,
            er_report.grounding_mean_before,
            er_report.grounding_mean_after,
        )
        if cluster_validity_report:
            try:
                from pipeline_scientific_audits import annotate_cluster_validity

                annotate_cluster_validity(competencies, cluster_validity_report)
                annotate_cluster_validity(failed_evidence_repair, cluster_validity_report)
            except Exception as e:
                logger.warning("Could not annotate evidence-repair outputs with cluster validity (%s)", e)

    if not args.skip_repair_stage and competencies and er_cfg is not None:
        logger.info("PHASE 2.7: precision/redundancy/recall repair stage")
        from competency_repair import RepairStageConfig, run_repair_stage
        denominator_for_repair = int(corpus_report.get("metadata_unique_job_ids") or 0)
        if denominator_for_repair <= 0 and jobs_meta.exists():
            from scripts.apply_scientific_sprint1 import load_job_denominator
            denominator_for_repair, _ = load_job_denominator(jobs_meta)
        if denominator_for_repair <= 0:
            denominator_for_repair = len({jid for c in competencies for jid in c.source_job_ids})
        repair_cfg = RepairStageConfig(
            denominator=denominator_for_repair,
            output_dir=out_dir,
            enable_targeted_recovery=not args.skip_targeted_recovery,
            recovery_model=args.targeted_recovery_model,
            max_recovery_candidates=args.max_recovery_candidates,
        )
        competencies, failed_repair_stage, repair_stage_report = run_repair_stage(
            competencies=competencies,
            failed_evidence_repair=failed_evidence_repair,
            config=repair_cfg,
            evidence_repair_config=er_cfg,
        )
        if cluster_validity_report:
            try:
                from pipeline_scientific_audits import annotate_cluster_validity

                annotate_cluster_validity(competencies, cluster_validity_report)
                annotate_cluster_validity(failed_repair_stage, cluster_validity_report)
            except Exception as e:
                logger.warning("Could not annotate repair-stage outputs with cluster validity (%s)", e)
        recovered_source_ids = set(repair_stage_report.get("recovered_source_ids") or [])
        pipeline_report["stages"]["phase2_7_repair"] = {
            "schema_version": repair_stage_report.get("schema_version"),
            "n_recovery_candidates": repair_stage_report.get("n_recovery_candidates"),
            "targeted_recovery_enabled": repair_stage_report.get("targeted_recovery_enabled"),
            "recovery_model": repair_stage_report.get("recovery_model"),
            "n_recovered_passing": repair_stage_report.get("n_recovered_passing"),
            "n_dedup_merged": repair_stage_report.get("n_dedup_merged"),
            "n_after_dedup": repair_stage_report.get("n_after_dedup"),
            "report_path": str(out_dir / "repair_audit_report.json"),
        }
        logger.info(
            "Phase 2.7: recovered %d competencies, merged %d duplicates, kept %d after repair",
            int(repair_stage_report.get("n_recovered_passing") or 0),
            int(repair_stage_report.get("n_dedup_merged") or 0),
            len(competencies),
        )

    if not args.skip_text_quality and competencies:
        logger.info("PHASE 2.5b: text-quality post-check")
        from competency_text_quality import TextQualityConfig, check_text_quality
        tq_passing, tq_failing, tq_report = check_text_quality(competencies, TextQualityConfig())
        if tq_failing:
            logger.warning(
                "Phase 2.5b: dropping %d competencies for text-quality violations",
                len(tq_failing),
            )
            for c in tq_failing:
                first_sent = (c.statement or "").split(".", 1)[0].strip()[:80]
                logger.warning("  - %s: %s", c.id, first_sent)
        competencies = tq_passing
        failed_text_quality = tq_failing
        pipeline_report["stages"]["phase2_5b_text_quality"] = {
            "n_passed": tq_report["n_passed"],
            "n_failed": tq_report["n_failed"],
        }

    # -------------------- Phase 2.3: KKNI labeller --------------------
    if not args.skip_kkni_labeller and competencies:
        logger.info("PHASE 2.3: KKNI labeller (multilingual SBERT)")
        from competency_kkni_labeller import KkniLabellerConfig, label_competencies_kkni
        kkni_report = label_competencies_kkni(competencies, KkniLabellerConfig())
        pipeline_report["stages"]["phase2_3_kkni"] = kkni_report
        logger.info("Phase 2.3: %d/%d relabelled", kkni_report["n_relabeled"], kkni_report["n_evaluated"])

    # -------------------- Phase 2.4: education aggregator --------------------
    if not args.skip_education and competencies:
        if jobs_meta.exists():
            logger.info("PHASE 2.4: education-level demand aggregation")
            from competency_education_aggregator import aggregate_education_demand
            edu_report = aggregate_education_demand(competencies, jobs_meta)
            pipeline_report["stages"]["phase2_4_education"] = edu_report
            logger.info("Phase 2.4: %d/%d competencies have education data",
                        edu_report["n_with_education_data"], edu_report["n_evaluated"])
        else:
            logger.warning("PHASE 2.4 skipped — jobs_metadata not found at %s", jobs_meta)
            pipeline_report["stages"]["phase2_4_education"] = {"skipped": True, "reason": "no jobs_metadata.csv"}

    # -------------------- Phase 2.4b — SKKNI occupation mapping (v9 sprint) --------------------
    # Map each surviving competency to the top-K Indonesian occupations via
    # multilingual SBERT cosine. Populates the `occupation_matches` field on
    # each CompetencyV2; an empty list means the competency does not anchor
    # to any Indonesian occupational standard.
    if args.enable_occupation_mapping and competencies:
        skkni_path = (PROJECT_ROOT / args.skkni_csv) if not Path(args.skkni_csv).is_absolute() else Path(args.skkni_csv)
        if not skkni_path.exists():
            logger.warning(
                "PHASE 2.4b skipped — SKKNI CSV not found at %s", skkni_path,
            )
            pipeline_report["stages"]["phase2_4b_occupations"] = {
                "skipped": True, "reason": "no SKKNI CSV",
            }
        else:
            logger.info("PHASE 2.4b: mapping competencies to SKKNI occupations")
            from occupation_mapper import load_occupations, map_competencies
            occupations = load_occupations(skkni_path)
            # CompetencyV2 instances expose `occupation_matches` as an attribute,
            # but `map_competencies` operates on dicts. Round-trip via the dict
            # representation and mirror the result back onto the instances.
            comp_dicts_for_mapping = [c.to_dict() for c in competencies]
            # Parse sector override
            from occupation_mapper import NO_SECTOR_FILTER
            sectors_arg = (args.occupation_sectors or "").strip()
            if sectors_arg.lower() == "all":
                sector_kw = NO_SECTOR_FILTER
            elif sectors_arg:
                sector_kw = [s.strip() for s in sectors_arg.split(",") if s.strip()]
            else:
                sector_kw = None  # use module default whitelist
            map_competencies(
                comp_dicts_for_mapping,
                occupations=occupations,
                top_k=args.occupation_top_k,
                min_cosine=args.occupation_min_cosine,
                relevant_sectors=sector_kw,
            )
            for c, d in zip(competencies, comp_dicts_for_mapping):
                c.occupation_matches = d.get("occupation_matches", []) or []
            n_mapped = sum(1 for c in competencies if c.occupation_matches)
            pipeline_report["stages"]["phase2_4b_occupations"] = {
                "n_competencies": len(competencies),
                "n_mapped": n_mapped,
                "n_unmapped": len(competencies) - n_mapped,
                "top_k": args.occupation_top_k,
                "min_cosine": args.occupation_min_cosine,
                "skkni_n_occupations": len(occupations),
            }
            logger.info(
                "Phase 2.4b: %d / %d competencies mapped to >=1 occupation",
                n_mapped, len(competencies),
            )

    # -------------------- Phase 4 — priority / demand scoring + final cap --------------------
    # Computed AFTER Phase 2.4 so every competency has its final grounding +
    # demand + future_weight values. The priority_score formula tracks Req 9.6.
    #
    # 2026-05-13b contract change: this stage is now the SOLE place where the
    # final cap is applied. The pipeline is "rank by priority_score, take
    # top N" — not "grounding-gate to 20, then sort". This means high-demand
    # topics like Java/Kotlin (25 jobs) and Databases reach the user even
    # when their grounding score is moderate, because the anti-hallucination
    # filter (Phase 2.2 final step) already removed actual fabrications.
    rank_excluded: list = []
    competency_dicts: list = []
    scientific_report = {}
    if competencies:
        from scripts.apply_scientific_sprint1 import (
            build_report as _build_scientific_report,
            class_aware_sort_key,
            csv_rows as _scientific_csv_rows,
            enrich_competencies,
            load_job_denominator,
            write_csv as _scientific_write_csv,
        )

        max_n_jobs = max((len(c.source_job_ids) for c in competencies), default=1) or 1
        denominator = int(corpus_report.get("metadata_unique_job_ids") or 0)
        if denominator <= 0:
            denominator, _ = load_job_denominator(jobs_meta if jobs_meta.exists() else None)
        if denominator <= 0:
            denominator = len({jid for c in competencies for jid in c.source_job_ids})

        for c in competencies:
            c.future_weight_source = c.future_weight_source or "llm_suggested_not_evidence"
            n_jobs = len(c.source_job_ids)
            c.demand_score = n_jobs / max_n_jobs if max_n_jobs > 0 else 0.0
            grounding = float(c.grounding_score or 0.0)
            fw = float(c.future_weight or 0.0)
            c.priority_score = round(
                0.40 * c.demand_score + 0.30 * grounding + 0.30 * fw, 4
            )

        priority_sorted = sorted(
            competencies,
            key=lambda c: (-c.priority_score, -len(c.source_job_ids), c.id),
        )
        n_before_cap = len(priority_sorted)
        all_competency_dicts = [c.to_dict() for c in priority_sorted]
        corpus_context = {
            "corpus_validity_status": corpus_report.get("corpus_validity_status", "needs_manual_review"),
            "corpus_country": corpus_report.get("corpus_country", "unknown"),
            "corpus_language": corpus_report.get("corpus_language", "unknown"),
            "corpus_dataset_id": corpus_report.get("dataset_id", args.corpus_dataset_id or args.tag),
            "corpus_lineage_report_path": str(corpus_report_path or ""),
        }
        if not args.skip_scientific_audits:
            try:
                from pipeline_scientific_audits import build_demand_theme_audit

                failed_for_theme_audit = [
                    *(c.to_dict() for c in hallucination_dropped),
                    *(c.to_dict() for c in failed_text_quality),
                    *(c.to_dict() for c in failed_evidence_repair),
                    *(c.to_dict() for c in failed_repair_stage),
                ]
                demand_theme_report = build_demand_theme_audit(
                    output_dir=out_dir,
                    competencies=all_competency_dicts,
                    failed_competencies=failed_for_theme_audit,
                    clusters=cluster_dicts,
                    denominator=denominator,
                )
                pipeline_report["stages"]["demand_theme_audit"] = {
                    "schema_version": demand_theme_report.get("schema_version"),
                    "report_path": str(out_dir / "demand_theme_audit.json"),
                    "status_distribution": demand_theme_report.get("status_distribution", {}),
                    "recommendation_distribution": demand_theme_report.get("recommendation_distribution", {}),
                    "recovery_candidate_theme_ids": demand_theme_report.get("recovery_candidate_theme_ids", []),
                }
            except Exception as e:
                logger.warning("Demand-theme audit failed; continuing pipeline (%s)", e)
                pipeline_report["stages"]["demand_theme_audit"] = {
                    "failed": True,
                    "error": str(e),
                }
        else:
            pipeline_report["stages"]["demand_theme_audit"] = {"skipped": True}
        enriched_all = enrich_competencies(
            all_competency_dicts,
            denominator=denominator,
            min_official_jobs=10,
            min_grounding=args.grounding_threshold,
            strong_min_jobs=10,
            moderate_min_jobs=5,
            corpus_context=corpus_context,
        )
        class_sorted = sorted(enriched_all, key=class_aware_sort_key)
        cap = g_cfg.final_competency_cap
        kept_dicts = class_sorted[:cap]
        excluded_dicts = class_sorted[cap:]
        comp_by_id = {c.id: c for c in competencies}
        kept_ids = {d.get("id") for d in kept_dicts}
        rank_excluded = [comp_by_id[d.get("id")] for d in excluded_dicts if d.get("id") in comp_by_id]
        competencies = [comp_by_id[d.get("id")] for d in kept_dicts if d.get("id") in comp_by_id]
        competency_dicts = kept_dicts

        scientific_report = _build_scientific_report(
            run_dir=out_dir,
            jobs_metadata=jobs_meta if jobs_meta.exists() else None,
            competencies=competency_dicts,
            denominator=denominator,
            min_official_jobs=10,
            min_grounding=args.grounding_threshold,
            strong_min_jobs=10,
            moderate_min_jobs=5,
            corpus_context=corpus_context,
        )
        scientific_rows = _scientific_csv_rows(competency_dicts)
        scientific_fields = [
            "pipeline_priority_rank", "official_core_rank", "competency_id",
            "competency_class", "competency_class_label", "support_tier",
            "support_tier_label", "official_core_eligible", "official_core_blockers",
            "recommendation_status", "official_readiness_blockers", "class_rank",
            "scientific_flags", "n_source_jobs", "eligible_job_denominator",
            "repair_decision", "repair_flags", "curriculum_dedup_group",
            "recovered_from_failed", "source_failure_reason",
            "demand_denominator", "job_prevalence_pct", "corpus_country",
            "corpus_language", "corpus_validity_status", "grounding_score",
            "future_weight", "future_weight_source", "public_future_weight",
            "priority_score", "demand_score",
            "occupation_mapping_status", "occupation_mapping_label", "statement",
            "cluster_validity_status", "cluster_validity_flags",
            "technical_theme_id", "technical_theme_label", "theme_evidence_status",
            "theme_recommendation_status", "not_recommended_reason",
            "related_skills",
        ]
        _scientific_write_csv(out_dir / "scientific_fix_competencies.csv", scientific_rows, scientific_fields)
        official_rows = [r for r in scientific_rows if str(r.get("official_core_eligible")).lower() == "true"]
        official_rows.sort(key=lambda r: int(r.get("official_core_rank") or 999999))
        _scientific_write_csv(out_dir / "official_core_recommendations.csv", official_rows, scientific_fields)
        for class_name in ("core_technical", "engineering_process", "employability_support", "out_of_domain_or_needs_review"):
            class_rows = [r for r in scientific_rows if r.get("competency_class") == class_name]
            class_rows.sort(key=lambda r: int(r.get("class_rank") or 999999))
            _scientific_write_csv(out_dir / f"ranked_{class_name}.csv", class_rows, scientific_fields)
            _scientific_write_csv(out_dir / f"ranked_{class_name}_repaired.csv", class_rows, scientific_fields)

        pipeline_report["stages"]["phase4_priority"] = {
            "max_n_unique_jobs": int(max_n_jobs),
            "demand_denominator": int(denominator),
            "legacy_formula": "0.40 * relative_demand + 0.30 * grounding + 0.30 * future_weight",
            "final_sort": "class-aware demand-first",
            "future_weight_policy": "future_weight is retained as an LLM-suggested research signal unless future_weight_source is deterministic/external evidence",
            "final_sort_order": "competency_class -> class_rank -> job_prevalence -> n_source_jobs -> grounding_score -> public_future_weight",
            "n_before_cap": n_before_cap,
            "final_competency_cap": cap,
            "n_after_cap": len(competencies),
            "n_excluded_by_cap": len(rank_excluded),
            "recommendation_status_distribution": scientific_report.get("distributions", {}).get("recommendation_status", {}),
            "competency_class_distribution": scientific_report.get("distributions", {}).get("competency_class", {}),
        }
        pipeline_report["stages"]["scientific_recommendation_contract"] = {
            "schema_version": scientific_report.get("schema_version"),
            "report_path": str(out_dir / "scientific_fix_report.json"),
            "corpus_validity_status": corpus_context.get("corpus_validity_status"),
            "official_core_eligible_count": scientific_report.get("official_core_eligible_count"),
            "recommendation_status_distribution": scientific_report.get("distributions", {}).get("recommendation_status", {}),
        }
        logger.info(
            "Phase 4: class-aware demand-first ranking kept %d/%d competencies "
            "(denominator=%d, corpus_status=%s)",
            len(competencies), n_before_cap, denominator, corpus_context.get("corpus_validity_status"),
        )

    # -------------------- v9: EN→ID translation (optional) --------------------
    if not competency_dicts:
        competency_dicts = [c.to_dict() for c in competencies]
    if args.translate_output_to_id and competency_dicts:
        logger.info(
            "Phase v9: translating %d competencies EN→ID via %s",
            len(competency_dicts), args.translation_model,
        )
        from translator import translate_competency_to_indonesian
        competency_dicts = [
            translate_competency_to_indonesian(c, model=args.translation_model)
            for c in competency_dicts
        ]
        # Translate the BatchReasoning text too so users browsing in ID see
        # the reasoning in their language.
        from translator import translate_to_indonesian as _tr_id
        for br in batch_reasonings:
            br_text = getattr(br, "batch_reasoning", "") or ""
            if br_text:
                # Mutate the BatchReasoning so its to_dict() picks it up.
                # Add the *_id field via a per-instance attribute.
                br.batch_reasoning_id = _tr_id(br_text, model=args.translation_model)
        pipeline_report["stages"]["phase_v9_translation"] = {
            "model": args.translation_model,
            "n_competencies_translated": len(competency_dicts),
            "n_batch_reasonings_translated": sum(
                1 for br in batch_reasonings if getattr(br, "batch_reasoning_id", None)
            ),
        }

    # -------------------- Persist --------------------
    (out_dir / "competencies.json").write_text(
        json.dumps(competency_dicts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if scientific_report:
        (out_dir / "scientific_fix_report.json").write_text(
            json.dumps(scientific_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if corpus_report:
        (out_dir / "corpus_lineage_report.json").write_text(
            json.dumps(corpus_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    # competencies_failed.json: records EVERY drop with its reason. The 2026-
    # 05-13b honesty audit found the old version only saved text-quality
    # failures (0 entries at v9, despite 30 grounding-gate drops + 18
    # invariant-violation drops). With the contract switch, drops happen at:
    #   1. anti-hallucination filter (Phase 2.2 final step) — fabrications
    #   2. text-quality post-check (Phase 2.5b) — language / length / etc.
    #   3. priority-rank cap (Phase 4) — fell below the top-N cut
    # All three groups are now serialised here so reviewers can audit the
    # full pipeline behaviour and adjust thresholds with provenance.
    failed_entries = []
    for c in hallucination_dropped:
        d = c.to_dict()
        d["dropped_reason"] = "anti_hallucination_filter"
        d["dropped_detail"] = {
            "warnings": [w for w in (c.statement_evidence_concerns or []) if w.get("severity") == "warning"],
        }
        failed_entries.append(d)
    for c in failed_text_quality:
        d = c.to_dict()
        d["dropped_reason"] = "text_quality_post_check"
        failed_entries.append(d)
    for c in failed_evidence_repair:
        d = c.to_dict()
        d["dropped_reason"] = "evidence_repair_gate"
        d["dropped_detail"] = c.evidence_repair_audit
        if c.id in recovered_source_ids:
            d["dropped_detail"] = dict(d.get("dropped_detail") or {})
            d["dropped_detail"]["recovered_by_repair_stage"] = True
        failed_entries.append(d)
    for c in failed_repair_stage:
        d = c.to_dict()
        d["dropped_reason"] = (
            "repair_stage_merged"
            if d.get("repair_decision") == "merged"
            else "repair_stage_failed"
        )
        d["dropped_detail"] = {
            "repair_decision": d.get("repair_decision"),
            "repair_flags": d.get("repair_flags") or [],
            "curriculum_dedup_group": d.get("curriculum_dedup_group", ""),
        }
        failed_entries.append(d)
    for c in rank_excluded:
        d = c.to_dict()
        d["dropped_reason"] = "priority_rank_cap"
        d["dropped_detail"] = {
            "priority_score": d.get("priority_score", 0.0),
            "cap": g_cfg.final_competency_cap,
            "note": "Excluded by final top-N cut; not a quality failure.",
        }
        failed_entries.append(d)
    (out_dir / "competencies_failed.json").write_text(
        json.dumps(failed_entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not args.skip_scientific_audits:
        try:
            from pipeline_scientific_audits import build_stage_coverage_report

            stage_coverage_report = build_stage_coverage_report(
                output_dir=out_dir,
                corpus_report=corpus_report,
                prepared_dir=prepared_dir,
                phase1_dir=in_dir,
                skill_llm_input=skill_llm_input,
                skill_llm_output=skill_llm_output,
                competencies=competency_dicts,
                failed_competencies=failed_entries,
            )
            pipeline_report["stages"]["stage_coverage_audit"] = {
                "schema_version": stage_coverage_report.get("schema_version"),
                "report_path": str(out_dir / "stage_coverage_report.json"),
                "labor_market_denominator": stage_coverage_report.get("labor_market_denominator"),
                "phase1_job_coverage_of_denominator": stage_coverage_report.get("phase1_job_coverage_of_denominator"),
                "final_source_job_coverage_of_denominator": stage_coverage_report.get("final_source_job_coverage_of_denominator"),
                "coverage_blockers": stage_coverage_report.get("coverage_blockers", []),
            }
        except Exception as e:
            logger.warning("Stage coverage audit failed; continuing final persist (%s)", e)
            pipeline_report["stages"]["stage_coverage_audit"] = {
                "failed": True,
                "error": str(e),
            }
    else:
        pipeline_report["stages"]["stage_coverage_audit"] = {"skipped": True}
    (out_dir / "batch_reasonings.json").write_text(
        json.dumps([br.to_dict() for br in batch_reasonings], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pipeline_report["total_runtime_seconds"] = round(time.time() - t_total, 2)
    pipeline_report["final_n_competencies"] = len(competencies)

    (out_dir / "pipeline_report.json").write_text(
        json.dumps(pipeline_report, indent=2),
        encoding="utf-8",
    )

    if not args.skip_research_packages:
        package_commands = [
            [
                sys.executable, "scripts/build_evidence_package.py",
                "--run-dir", str(out_dir),
            ],
            [
                sys.executable, "scripts/build_dacum_cbc_package.py",
                "--run-dir", str(out_dir),
            ],
            [
                sys.executable, "scripts/build_scientific_validation_pack.py",
                "--run-dir", str(out_dir),
            ],
            [
                sys.executable, "scripts/plot_competency_v2_run.py",
                "--run-dir", str(out_dir),
            ],
        ]
        package_results = []
        for cmd in package_commands:
            try:
                completed = subprocess.run(
                    cmd,
                    cwd=PROJECT_ROOT,
                    text=True,
                    capture_output=True,
                    timeout=240,
                    check=False,
                )
                package_results.append({
                    "command": " ".join(cmd),
                    "returncode": completed.returncode,
                    "stdout_tail": (completed.stdout or "")[-1200:],
                    "stderr_tail": (completed.stderr or "")[-1200:],
                })
                if completed.returncode != 0:
                    logger.warning(
                        "Research package command failed (%s): %s",
                        completed.returncode,
                        " ".join(cmd),
                    )
            except Exception as e:
                package_results.append({
                    "command": " ".join(cmd),
                    "failed": True,
                    "error": str(e),
                })
                logger.warning("Research package command failed (%s): %s", " ".join(cmd), e)
        pipeline_report["stages"]["research_packages"] = {
            "evidence_package": str(PROJECT_ROOT / "results" / "evidence_packages" / args.tag),
            "dacum_cbc_package": str(PROJECT_ROOT / "results" / "dacum_cbc_packages" / args.tag),
            "validation_package": str(PROJECT_ROOT / "results" / "scientific_validation" / args.tag),
            "figures_package": str(PROJECT_ROOT / "results" / "figures" / args.tag),
            "commands": package_results,
        }
        (out_dir / "pipeline_report.json").write_text(
            json.dumps(pipeline_report, indent=2),
            encoding="utf-8",
        )
    else:
        pipeline_report["stages"]["research_packages"] = {"skipped": True}
        (out_dir / "pipeline_report.json").write_text(
            json.dumps(pipeline_report, indent=2),
            encoding="utf-8",
        )

    print()
    print("=" * 80)
    print(f"v2 pipeline complete — {len(competencies)} final competencies in {pipeline_report['total_runtime_seconds']:.1f}s")
    print(f"  Output: {out_dir}")
    print(f"  Launch dashboard:")
    print(f"    streamlit run dashboard_v2/app.py -- --run-dir {out_dir.relative_to(PROJECT_ROOT)}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
