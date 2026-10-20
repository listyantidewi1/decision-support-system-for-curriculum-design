"""
dashboard_v2/app.py

Phase 2.6 — minimal public UI for the v2 competency pipeline.

Streamlit single-page app that:
  * lists all competencies from a v2 run directory (competencies.json)
  * supports filter by KKNI level, future_weight, grounding score, and search
  * shows full provenance chain (source sentences + jobs) per competency
  * exposes the per-competency rationale with a "Read more" toggle
  * exposes the cluster's batch_reasoning with a "View reasoning" toggle
  * shows education-level demand distribution (Phase 2.4)

Run:
    streamlit run dashboard_v2/app.py -- --run-dir results/competency_v2_live_gpt54mini

Or, from the project root:
    streamlit run dashboard_v2/app.py

Environment variables (optional override):
    COMPETENCY_V2_RUN_DIR  — directory with competencies.json + batch_reasonings.json + clusters.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import streamlit as st

# Add project root to sys.path so kkni etc. import cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# CLI args
# --------------------------------------------------------------------------- #


def _resolve_run_dir() -> Path:
    # 1. CLI flag --run-dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None)
    # streamlit may pass other args we don't care about
    args, _ = parser.parse_known_args(sys.argv[1:])

    rd = args.run_dir or os.environ.get("COMPETENCY_V2_RUN_DIR")
    if rd:
        return Path(rd).resolve()

    # 2. Auto-pick the most recent results/competency_v2_* directory that
    #    has a competencies.json file.
    candidates = []
    results_root = PROJECT_ROOT / "results"
    if results_root.exists():
        for d in results_root.iterdir():
            if not d.is_dir():
                continue
            name = d.name
            if not (name.startswith("competency_v2_pipeline_")
                    or name.startswith("competency_v2_live")):
                continue
            comp_file = d / "competencies.json"
            if comp_file.exists():
                candidates.append((d, comp_file.stat().st_mtime))
    # Prefer competency_v2_pipeline_* (full pipeline output) over competency_v2_live*
    # (single-stage / model-comparison runs) when timestamps are close — but
    # primarily order by recency. Take the newest competencies.json.
    candidates.sort(key=lambda t: t[1], reverse=True)
    if candidates:
        return candidates[0][0].resolve()
    return (PROJECT_ROOT / "results" / "competency_v2_pipeline_e2e_v2_real_sentences").resolve()


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def load_run(run_dir: str) -> dict:
    """Load competencies + batch_reasonings + clusters from a run directory."""
    rd = Path(run_dir)
    out = {"run_dir": str(rd), "competencies": [], "batch_reasonings": [], "clusters": []}
    cp = rd / "competencies.json"
    if cp.exists():
        out["competencies"] = json.loads(cp.read_text(encoding="utf-8"))
    bp = rd / "batch_reasonings.json"
    if bp.exists():
        out["batch_reasonings"] = json.loads(bp.read_text(encoding="utf-8"))
    cl = rd / "clusters.json"
    if cl.exists():
        out["clusters"] = json.loads(cl.read_text(encoding="utf-8"))
    return out


@st.cache_data(show_spinner=False)
def load_job_titles(run_dir: str) -> dict:
    """Load {job_id: title} from this run's jobs_metadata.csv (path is recorded
    in pipeline_report.json). Returns empty dict when not available — the
    role-shape panel then falls back to showing raw job_ids.
    """
    import csv as _csv
    rd = Path(run_dir)
    pr = rd / "pipeline_report.json"
    if not pr.exists():
        return {}
    try:
        report = json.loads(pr.read_text(encoding="utf-8"))
    except Exception:
        return {}
    jm_path = report.get("jobs_metadata") or ""
    if not jm_path:
        return {}
    p = Path(jm_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return {}
    out: dict = {}
    try:
        with open(p, encoding="utf-8-sig") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                jid = (row.get("job_id") or "").strip()
                title = (row.get("title") or "").strip()
                if jid and title:
                    out[jid] = title
    except Exception:
        return {}
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


# Role-normalisation: hoisted to `role_normalization` module (Phase 2.A of v8
# sprint) so the Phase 2.2 generator can share the same buckets the dashboard
# uses. See role_normalization.py for the table.
from role_normalization import normalize_role as _normalize_role  # noqa: F401


def _stage_chip(stage: str, frac: float) -> str:
    return f"`{stage}` {frac*100:.0f}%"


def _kkni_color(level: Optional[int]) -> str:
    if level is None:
        return "#888888"
    palette = {
        3: "#1f77b4", 4: "#2ca02c", 5: "#ff7f0e", 6: "#d62728",
        7: "#9467bd", 8: "#8c564b", 9: "#e377c2",
    }
    return palette.get(int(level), "#888888")


# Brief cognitive-complexity descriptor per KKNI level (v8 sprint Phase 3).
# Mirrors the Bloom-style anchors in the generator's KKNI_BRIEF — kept short
# enough to fit in a metadata-cell caption. KKNI = cognitive complexity of
# the work itself, independent of education-stage demand.
_KKNI_LEVEL_DESCRIPTORS = {
    3: "operational; follows procedures",
    4: "applies standard methods; minor diagnostic decisions",
    5: "analyses + selects methods; supervises routine",
    6: "designs novel-within-standard; leads small teams",
    7: "complex novel problems; multidisciplinary; leadership",
    8: "research; original contribution",
    9: "research; original contribution",
}


def _kkni_descriptor(level: Optional[int]) -> str:
    if level is None:
        return ""
    try:
        return _KKNI_LEVEL_DESCRIPTORS.get(int(level), "")
    except (TypeError, ValueError):
        return ""


# v9 sprint: target-stage-fit support. Each stage maps to a KKNI range; the
# fit chip is based on the delta between the competency's KKNI and the
# stage's max KKNI level. KKNI and education-stage demand stay independent
# labels — this is a UX hint for curriculum designers, not a gate.
_TARGET_STAGE_KKNI_RANGE = {
    "SMK":      (2, 3),
    "D3":       (4, 5),
    "D4 / S1":  (6, 6),
    "S2":       (7, 8),
}


def _target_stage_fit(target_stage: str, kkni_level: Optional[int]) -> Optional[dict]:
    """Return a fit-chip descriptor for this competency given the user's
    selected target stage. Returns None when target_stage is 'Any' / kkni
    is missing."""
    if not target_stage or target_stage == "Any" or kkni_level is None:
        return None
    rng = _TARGET_STAGE_KKNI_RANGE.get(target_stage)
    if not rng:
        return None
    lo, hi = rng
    try:
        k = int(kkni_level)
    except (TypeError, ValueError):
        return None
    if lo <= k <= hi:
        return {"severity": "match", "delta": 0, "msg": f"Matches your {target_stage} target stage"}
    if k == hi + 1:
        return {"severity": "info", "delta": 1, "msg": f"Slightly above your {target_stage} target stage"}
    if k > hi + 1:
        return {"severity": "warning", "delta": k - hi,
                "msg": f"{k - hi} levels above your {target_stage} target stage — consider whether students can reach this complexity"}
    if k < lo:
        return {"severity": "below", "delta": k - lo,
                "msg": f"Below your {target_stage} target stage — may already be covered at a lower level"}
    return None


def _stages_for_kkni(level: Optional[int]) -> list:
    """Map a KKNI level back to the friendly Indonesian education stages it
    represents. Uses the canonical `kkni.STAGE_TO_KKNI` mapping reversed.
    e.g., level 5 -> ['D3'], level 6 -> ['D4', 'S1'], level 3 -> ['SMK'].
    """
    if level is None:
        return []
    try:
        from kkni import STAGE_TO_KKNI
    except Exception:
        return []
    out = []
    for stage, levels in STAGE_TO_KKNI.items():
        if int(level) in levels:
            out.append(stage)
    # Prefer the SMK-D3-D4-S1 ordering that vocational stakeholders use
    preferred_order = ["SMK", "D1", "D2", "D3", "D4", "S1", "Profesi", "S2", "S3", "SMA"]
    out.sort(key=lambda s: preferred_order.index(s) if s in preferred_order else 99)
    return out


def _grounding_color(score: float) -> str:
    if score >= 0.95: return "#1a9850"  # green
    if score >= 0.80: return "#91cf60"  # lighter green
    if score >= 0.60: return "#fc8d59"  # orange
    return "#d73027"                     # red


def _grounding_tier(score: float) -> tuple:
    """Return (icon, label, color) for the grounding transparency badge.

    Replaces the legacy color-only display 2026-05-13b after the contract
    switch from grounding-gate to demand-rank. Grounding is now an evidence-
    strength LABEL, not a gate — users see the tier and can filter, but
    high-demand low-grounding competencies still surface so the user can
    decide whether the demand signal is enough on its own.
    """
    if score >= 0.60:
        return "✅", "well grounded", "#1a9850"
    if score >= 0.30:
        return "⚠️", "partial evidence", "#fc8d59"
    return "🚧", "thin evidence", "#d73027"


def _statement(comp: dict, use_indonesian: bool) -> str:
    """Return the competency statement, with read-side back-compat for v1
    JSONs that stored title + description separately."""
    s = _localize(comp, "statement", use_indonesian)
    if isinstance(s, str) and s.strip():
        return s
    # Legacy path
    t = _localize(comp, "title", use_indonesian) or comp.get("title", "")
    d = _localize(comp, "description", use_indonesian) or comp.get("description", "")
    if t and d:
        return f"{t}. {d}"
    return (t or d or "").strip()


def _statement_first_sentence(comp: dict, use_indonesian: bool, max_chars: int = 110) -> str:
    """Short label for the browse list — first sentence of the statement,
    truncated. Used in place of the legacy `title`."""
    full = _statement(comp, use_indonesian)
    if not full:
        return "(no statement)"
    first = full.split(".", 1)[0].strip()
    return (first[:max_chars] + "…") if len(first) > max_chars else first


def _bar(fraction: float, color: str, label: str = "") -> str:
    pct = max(0, min(100, int(round(fraction * 100))))
    return f"""
<div style='background:#eee; border-radius:4px; height:8px; overflow:hidden; margin-top:2px;'>
  <div style='background:{color}; width:{pct}%; height:100%'></div>
</div>
<div style='font-size:11px; color:#666;'>{label}</div>
"""


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


_CORPUS_STATUS_LABELS = {
    "valid_for_engineering_test": "Engineering/test corpus",
    "valid_for_indonesian_claim": "Valid Indonesian corpus",
    "invalid_or_incomplete": "Invalid or incomplete lineage",
    "needs_manual_review": "Needs manual corpus review",
}


_RECOMMENDATION_STATUS_LABELS = {
    "exploratory": "Exploratory",
    "review_candidate": "Review candidate",
    "official_ready": "Official ready",
    "not_recommended_from_this_corpus": "Not recommended from this corpus",
}


_COMPETENCY_CLASS_LABELS = {
    "core_technical": "Core technical",
    "engineering_process": "Engineering process / leadership",
    "employability_support": "Employability / support",
    "out_of_domain_or_needs_review": "Needs review / out-of-domain",
}


def _competency_class_label(comp: dict) -> str:
    value = comp.get("competency_class_label") or comp.get("competency_class") or "Unclassified"
    return _COMPETENCY_CLASS_LABELS.get(str(value), str(value))


def _support_tier_label(comp: dict) -> str:
    return str(comp.get("support_tier_label") or comp.get("support_tier") or "Not tiered")


def _recommendation_status_label(comp: dict) -> str:
    status = str(comp.get("recommendation_status") or "exploratory")
    return _RECOMMENDATION_STATUS_LABELS.get(status, status)


def _corpus_status_label(status: str) -> str:
    return _CORPUS_STATUS_LABELS.get(str(status or "needs_manual_review"), str(status or "needs_manual_review"))


def _job_prevalence_pct(comp: dict) -> float:
    if comp.get("job_prevalence_pct") is not None:
        return _safe_float(comp.get("job_prevalence_pct"))
    prevalence = _safe_float(comp.get("job_prevalence"))
    return prevalence * 100.0 if prevalence else 0.0


def _public_future_weight(comp: dict) -> float:
    source = comp.get("future_weight_source") or "llm_suggested_not_evidence"
    if source in {"deterministic_domain_mapping", "external_trend_evidence", "official_reference_mapping"}:
        return _safe_float(comp.get("future_weight"))
    return 0.0


def _load_run_json(run_dir: Path, filename: str, default):
    path = run_dir / filename
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _has_sprint1_fields(comps: list) -> bool:
    return any(c.get("scientific_sprint1_schema_version") for c in comps or [])


def _load_corpus_report(run_dir: Path) -> dict:
    local = run_dir / "corpus_lineage_report.json"
    if local.exists():
        try:
            return json.loads(local.read_text(encoding="utf-8"))
        except Exception:
            return {}
    pr = run_dir / "pipeline_report.json"
    if not pr.exists():
        return {}
    try:
        pipeline_report = json.loads(pr.read_text(encoding="utf-8"))
    except Exception:
        return {}
    candidate = (
        pipeline_report.get("corpus_lineage_report")
        or ((pipeline_report.get("stages", {}) or {}).get("corpus_lineage_audit", {}) or {}).get("report_path")
        or ""
    )
    if not candidate:
        return {}
    path = Path(candidate)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _corpus_context(comps: list, corpus_report: dict | None = None) -> dict:
    corpus_report = corpus_report or {}
    first = comps[0] if comps else {}
    return {
        "dataset_id": corpus_report.get("dataset_id") or first.get("corpus_dataset_id") or "",
        "status": corpus_report.get("corpus_validity_status") or first.get("corpus_validity_status") or "needs_manual_review",
        "country": corpus_report.get("corpus_country") or first.get("corpus_country") or "unknown",
        "language": corpus_report.get("corpus_language") or first.get("corpus_language") or "unknown",
        "metadata_jobs": corpus_report.get("metadata_unique_job_ids") or first.get("demand_denominator") or first.get("eligible_job_denominator") or "",
        "phase1_jobs": corpus_report.get("phase1_unique_job_ids") or "",
    }


def _render_corpus_notice(comps: list, corpus_report: dict | None = None) -> None:
    ctx = _corpus_context(comps, corpus_report)
    status = ctx["status"]
    label = _corpus_status_label(status)
    detail = (
        f"{label}; country={ctx['country']}; language={ctx['language']}; "
        f"denominator={ctx['metadata_jobs'] or 'unknown'} postings."
    )
    if status == "valid_for_indonesian_claim":
        st.success(f"Corpus validity: {detail}")
    elif status == "valid_for_engineering_test":
        st.warning(
            f"Corpus validity: {detail} This run is useful for engineering validation, "
            "but it should not be presented as Indonesian curriculum demand."
        )
    elif status == "invalid_or_incomplete":
        st.error(f"Corpus validity: {detail} Demand claims should be blocked until lineage is fixed.")
    else:
        st.warning(f"Corpus validity: {detail} Treat recommendations as exploratory until audited.")


def _ranked_competencies(comps: list) -> list:
    if any(c.get("class_rank") for c in comps or []):
        return sorted(
            comps or [],
            key=lambda c: (
                int(c.get("class_sort_group") if c.get("class_sort_group") is not None else 99),
                int(c.get("class_rank") or 999999),
                -_job_prevalence_pct(c),
                -int(c.get("n_source_jobs") or len(c.get("source_job_ids") or [])),
                -_safe_float(c.get("grounding_score", c.get("grounding_score_preview", 0.0))),
                -_public_future_weight(c),
                -_safe_float(c.get("priority_score")),
                str(c.get("id", "")),
            ),
        )
    return sorted(
        comps or [],
        key=lambda c: (
            -_safe_float(c.get("priority_score")),
            -len(c.get("source_job_ids") or []),
            str(c.get("id", "")),
        ),
    )


def _competency_chart_rows(comps: list) -> list:
    rows = []
    for rank, c in enumerate(_ranked_competencies(comps), 1):
        gs = _safe_float(c.get("grounding_score", c.get("grounding_score_preview", 0.0)))
        _tier_icon, tier_label, _tier_color = _grounding_tier(gs)
        rows.append({
            "rank": rank,
            "competency_id": c.get("id", ""),
            "statement": _statement_first_sentence(c, False, max_chars=86),
            "priority_score": _safe_float(c.get("priority_score")),
            "demand_score": _safe_float(c.get("demand_score")),
            "future_weight": _safe_float(c.get("future_weight")),
            "future_weight_source": c.get("future_weight_source") or "llm_suggested_not_evidence",
            "public_future_weight": _public_future_weight(c),
            "grounding_score": gs,
            "grounding_tier": tier_label,
            "kkni_level": c.get("kkni_level"),
            "n_source_jobs": len(c.get("source_job_ids") or []),
            "n_source_jobs_sprint1": int(c.get("n_source_jobs") or len(c.get("source_job_ids") or [])),
            "eligible_job_denominator": c.get("eligible_job_denominator"),
            "job_prevalence_pct": _job_prevalence_pct(c),
            "n_related_skills": len(c.get("related_skills") or []),
            "n_occupation_matches": len(c.get("occupation_matches") or []),
            "competency_class": _competency_class_label(c),
            "support_tier": _support_tier_label(c),
            "recommendation_status": _recommendation_status_label(c),
            "repair_decision": c.get("repair_decision") or "",
            "repair_flags": ", ".join(c.get("repair_flags") or []),
            "curriculum_dedup_group": c.get("curriculum_dedup_group") or "",
            "recovered_from_failed": bool(c.get("recovered_from_failed")),
            "corpus_validity_status": c.get("corpus_validity_status") or "",
            "corpus_country": c.get("corpus_country") or "",
            "corpus_language": c.get("corpus_language") or "",
            "official_core_eligible": bool(c.get("official_core_eligible")),
            "official_core_rank": c.get("official_core_rank"),
            "occupation_mapping_status": c.get("occupation_mapping_label") or c.get("occupation_mapping_status") or "",
            "cluster_validity_status": c.get("cluster_validity_status") or "",
            "cluster_validity_flags": ", ".join(c.get("cluster_validity_flags") or []),
            "technical_theme_label": c.get("technical_theme_label") or "",
            "theme_evidence_status": c.get("theme_evidence_status") or "",
            "theme_recommendation_status": c.get("theme_recommendation_status") or "",
            "not_recommended_reason": c.get("not_recommended_reason") or "",
        })
    return rows


def _education_totals(comps: list) -> dict:
    totals: dict = {}
    for c in comps or []:
        edu = c.get("education_levels_demanded") or {}
        if not isinstance(edu, dict):
            continue
        for stage, frac in edu.items():
            totals[stage] = totals.get(stage, 0.0) + _safe_float(frac)
    return totals


def _occupation_counts(comps: list) -> dict:
    counts: dict = {}
    for c in comps or []:
        for m in c.get("occupation_matches") or []:
            name = m.get("occupation_en") or m.get("occupation_id") or ""
            if name:
                counts[name] = counts.get(name, 0) + 1
    return counts


def _render_market_insights_page(comps: list, run_dir: Path, corpus_report: dict | None = None):
    """Decision-oriented charts for curriculum designers."""
    st.markdown("# Market insights")
    st.caption(
        "High-level view of what the final recommendation set says about "
        "market demand, future relevance, KKNI complexity, education-stage "
        "demand, and Indonesian occupation mapping."
    )
    if not comps:
        st.info("No competencies found for this run.")
        return
    _render_corpus_notice(comps, corpus_report)

    import pandas as pd
    import matplotlib.pyplot as plt

    rows = _competency_chart_rows(comps)
    df = pd.DataFrame(rows)
    has_sprint1 = _has_sprint1_fields(comps)

    stage_coverage = _load_run_json(run_dir, "stage_coverage_report.json", {})
    cluster_validity = _load_run_json(run_dir, "cluster_validity_report.json", {})
    demand_theme = _load_run_json(run_dir, "demand_theme_audit.json", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final competencies", len(comps))
    if has_sprint1:
        c2.metric("Core candidates", f"{int(df['official_core_eligible'].sum())}/{len(df)}")
        c3.metric("Mean prevalence", f"{df['job_prevalence_pct'].mean():.2f}%")
        c4.metric("Mapped to SKKNI", f"{int((df['n_occupation_matches'] > 0).sum())}/{len(df)}")
    else:
        c2.metric("Mapped to SKKNI", f"{int((df['n_occupation_matches'] > 0).sum())}/{len(df)}")
        c3.metric("Mean grounding", f"{df['grounding_score'].mean():.2f}")
        c4.metric("Mean demand", f"{df['demand_score'].mean():.2f}")

    if stage_coverage or cluster_validity or demand_theme:
        st.markdown("## Scientific audit snapshot")
        audit_cols = st.columns(4)
        audit_cols[0].metric(
            "Denominator",
            stage_coverage.get("labor_market_denominator", df.get("eligible_job_denominator", [""])[0] if not df.empty else ""),
        )
        audit_cols[1].metric(
            "Coverage blockers",
            len(stage_coverage.get("coverage_blockers") or []),
        )
        audit_cols[2].metric(
            "Split/mixed clusters",
            int(cluster_validity.get("n_split_candidates") or 0),
        )
        audit_cols[3].metric(
            "Not-recommended themes",
            int((demand_theme.get("recommendation_distribution") or {}).get("not_recommended_from_this_corpus", 0)),
        )
        blockers = stage_coverage.get("coverage_blockers") or []
        if blockers:
            st.warning("Coverage blockers: " + ", ".join(str(b) for b in blockers))
        with st.expander("Show audit distributions"):
            if cluster_validity:
                st.write("Cluster validity")
                st.json(cluster_validity.get("status_distribution", {}), expanded=False)
            if demand_theme:
                st.write("Demand theme evidence")
                st.json(demand_theme.get("status_distribution", {}), expanded=False)

    if has_sprint1:
        st.markdown("## Recommendation contract")
        st.caption(
            "Recommendations are class-aware and status-capped by corpus validity. "
            "The original priority score remains visible as a research baseline, "
            "while the stricter core technical list is demand-first."
        )
        class_col, support_col, status_col = st.columns(3)
        with class_col:
            class_counts = df["competency_class"].value_counts()
            st.bar_chart(class_counts.rename("competencies"), height=280, color="#4477aa")
        with support_col:
            support_counts = df["support_tier"].value_counts()
            st.bar_chart(support_counts.rename("competencies"), height=280, color="#66aa55")
        with status_col:
            status_counts = df["recommendation_status"].value_counts()
            st.bar_chart(status_counts.rename("competencies"), height=280, color="#cc9944")

        core = df[df["official_core_eligible"]].copy()
        if not core.empty:
            core = core.sort_values(["official_core_rank", "rank"])
            st.markdown("## Core technical candidates")
            st.caption(
                "Demand-first subset passing sprint-1 gates: core technical class, "
                "minimum source-job support, grounding threshold, and at least one occupation mapping."
            )
            st.dataframe(
                core[[
                    "official_core_rank", "rank", "statement", "n_source_jobs_sprint1",
                    "job_prevalence_pct", "grounding_score", "public_future_weight",
                    "future_weight_source", "occupation_mapping_status", "repair_decision",
                    "cluster_validity_status", "theme_evidence_status",
                ]].rename(columns={
                    "official_core_rank": "core_rank",
                    "rank": "pipeline_rank",
                    "n_source_jobs_sprint1": "source_jobs",
                    "job_prevalence_pct": "prevalence_pct",
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("No competencies passed the sprint-1 core technical candidate gates.")

    st.markdown("## Top recommendations")
    st.caption(
        "Ranked by the public recommendation contract when available. The priority "
        "score is retained as the legacy demand/grounding/future baseline."
    )
    top = df.head(15).copy()
    if not top.empty:
        chart = top[["statement", "priority_score"]].set_index("statement").sort_values("priority_score")
        st.bar_chart(chart, height=440, color="#4477aa")
        with st.expander("Show score table"):
            st.dataframe(
                top[[
                    "rank", "statement", "priority_score", "demand_score",
                    "future_weight", "future_weight_source", "public_future_weight",
                    "grounding_score", "kkni_level",
                    "n_source_jobs", "job_prevalence_pct", "competency_class",
                    "support_tier", "recommendation_status", "repair_decision",
                    "recovered_from_failed", "official_core_eligible", "n_occupation_matches",
                    "cluster_validity_status", "theme_evidence_status", "not_recommended_reason",
                ]],
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("## Demand vs future relevance")
    st.caption(
        "Bubble size reflects number of source jobs; color reflects grounding score. "
        "The y-axis uses public future weight, so LLM-suggested future scores appear as 0 unless backed by deterministic or external evidence."
    )
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    sizes = (df["n_source_jobs"].clip(lower=1) * 9).tolist()
    scatter = ax.scatter(
        df["demand_score"], df["public_future_weight"],
        s=sizes,
        c=df["grounding_score"],
        cmap="viridis",
        alpha=0.75,
        edgecolor="#333333",
        linewidth=0.4,
    )
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Demand score")
    ax.set_ylabel("Public future relevance")
    ax.grid(True, alpha=0.22)
    for _, row in df.head(8).iterrows():
        ax.annotate(str(int(row["rank"])), (row["demand_score"], row["public_future_weight"]),
                    color="white", fontsize=8, ha="center", va="center")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Grounding score")
    st.pyplot(fig, clear_figure=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("## KKNI complexity")
        kkni = df["kkni_level"].dropna().astype(int).value_counts().sort_index()
        if not kkni.empty:
            st.bar_chart(kkni.rename("competencies"), height=300, color="#66aa55")
        else:
            st.info("No KKNI labels available.")

    with col_b:
        st.markdown("## Evidence strength")
        grounding_bins = pd.cut(
            df["grounding_score"],
            bins=[-0.01, 0.30, 0.60, 0.80, 1.01],
            labels=["thin <0.30", "partial 0.30-0.60", "strong 0.60-0.80", "gate >=0.80"],
        )
        st.bar_chart(grounding_bins.value_counts().sort_index().rename("competencies"), height=300, color="#aa7744")

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("## Education-stage demand")
        edu = _education_totals(comps)
        if edu:
            preferred = ["SMK", "SMA", "D1", "D2", "D3", "D4", "S1", "Profesi", "S2", "S3"]
            edu_df = pd.DataFrame(
                [{"stage": k, "summed_share": v} for k, v in edu.items()]
            ).sort_values(
                "stage",
                key=lambda s: s.map(lambda x: preferred.index(x) if x in preferred else 99),
            )
            st.bar_chart(edu_df.set_index("stage"), height=320, color="#4477aa")
        else:
            st.info("No education demand data available.")

    with col_d:
        st.markdown("## SKKNI occupation coverage")
        occ = _occupation_counts(comps)
        mapped = int((df["n_occupation_matches"] > 0).sum())
        st.metric("Competencies with occupation match", f"{mapped}/{len(df)}")
        if occ:
            occ_df = pd.DataFrame(
                [{"occupation": k, "matches": v} for k, v in occ.items()]
            ).sort_values("matches", ascending=False).head(10)
            st.bar_chart(occ_df.set_index("occupation"), height=320, color="#7755aa")
        else:
            st.info("No SKKNI occupation matches available.")

    figs_dir = PROJECT_ROOT / "results" / "figures" / run_dir.name.replace("competency_v2_pipeline_", "") / "dashboard"
    if figs_dir.exists():
        with st.expander("Static dashboard figures saved for this run"):
            for p in sorted(figs_dir.glob("*.png")):
                st.image(str(p), caption=p.name, use_container_width=True)


def _render_pipeline_charts(run_dir: Path, pipeline_report: dict, noise_audit: dict):
    """Compact research/audit charts for the Pipeline audit page."""
    import pandas as pd
    import matplotlib.pyplot as plt

    st.markdown("## Run charts")
    stages = pipeline_report.get("stages", {}) if isinstance(pipeline_report, dict) else {}
    p1 = stages.get("phase1_load", {}) or {}
    p21 = stages.get("phase2_1_clustering", {}) or {}
    p22 = stages.get("phase2_2_generation", {}) or {}
    p26 = stages.get("phase2_6_evidence_repair", {}) or {}
    p4 = stages.get("phase4_priority", {}) or {}

    funnel = [
        ("Extracted items", p1.get("n_items")),
        ("Canonical items", p21.get("n_items_canonical")),
        ("Frequency-filtered items", p21.get("n_items_after_frequency_filter")),
        ("Clusters", (p21.get("n_clusters_hdbscan", 0) or 0) + (p21.get("n_clusters_recovered", 0) or 0) + (p21.get("n_clusters_split", 0) or 0)),
        ("Generated competencies", p22.get("n_competencies")),
    ]
    if p26:
        funnel.append(("Evidence-repaired", p26.get("n_repaired")))
    funnel.append(("Final recommendations", pipeline_report.get("final_n_competencies") or p4.get("n_after_cap")))
    funnel = [(label, int(value)) for label, value in funnel if value is not None]

    if funnel:
        st.markdown("### Pipeline funnel")
        funnel_df = pd.DataFrame(funnel, columns=["stage", "count"]).set_index("stage")
        st.bar_chart(funnel_df, height=300, color="#4477aa")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Drop reasons")
        counts = {}
        for stage, n in (noise_audit.get("by_stage", {}) or {}).items():
            counts[f"Phase 2.1: {stage}"] = int(n)
        failed_path = run_dir / "competencies_failed.json"
        if failed_path.exists():
            try:
                failed = json.loads(failed_path.read_text(encoding="utf-8"))
                for item in failed or []:
                    reason = item.get("dropped_reason")
                    if reason:
                        counts[f"Competency: {reason}"] = counts.get(f"Competency: {reason}", 0) + 1
            except Exception:
                pass
        if counts:
            drop_df = pd.DataFrame(
                [{"reason": k, "count": v} for k, v in counts.items()]
            ).sort_values("count", ascending=False).head(12)
            st.bar_chart(drop_df.set_index("reason"), height=360, color="#888888")
        else:
            st.info("No drop-reason data available.")

    with col2:
        st.markdown("### Evidence repair")
        if p26:
            repair_df = pd.DataFrame([
                {"metric": "mean before", "score": _safe_float(p26.get("grounding_mean_before"))},
                {"metric": "mean after", "score": _safe_float(p26.get("grounding_mean_after"))},
                {"metric": "min before", "score": _safe_float(p26.get("grounding_min_before"))},
                {"metric": "min after", "score": _safe_float(p26.get("grounding_min_after"))},
            ]).set_index("metric")
            st.bar_chart(repair_df, height=280, color="#228833")
            st.caption(
                f"Repaired {p26.get('n_repaired', 0)} of {p26.get('n_input', 0)} "
                f"candidates; dropped {p26.get('n_dropped', 0)}."
            )
        else:
            st.info("No evidence repair stage found in this run.")

    clusters_path = run_dir / "clusters.json"
    if clusters_path.exists():
        try:
            clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
            scores = [_safe_float(c.get("cohesion_score")) for c in clusters if c.get("cohesion_score") is not None]
            if scores:
                st.markdown("### Cluster cohesion distribution")
                fig, ax = plt.subplots(figsize=(8, 4.2))
                ax.hist(scores, bins=14, color="#66aa55", edgecolor="white")
                ax.axvline(0.55, color="#cc3311", linestyle="--", linewidth=1.2, label="threshold 0.55")
                ax.set_xlabel("Cohesion score")
                ax.set_ylabel("Clusters")
                ax.legend()
                st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.warning(f"Could not render cluster cohesion chart: {e}")


# --------------------------------------------------------------------------- #
# Detail panel
# --------------------------------------------------------------------------- #


def _localize(comp: dict, field: str, use_indonesian: bool) -> str:
    """Return the Indonesian translation of a competency field when
    available + requested; otherwise the English original. Used by every
    rendered string in the detail view (v9 sprint language toggle)."""
    if use_indonesian:
        v = comp.get(f"{field}_id")
        if isinstance(v, str) and v:
            return v
    return comp.get(field, "") or ""


def _localize_list(comp: dict, field: str, use_indonesian: bool) -> list:
    """Same as `_localize` but for list-valued fields (related_skills,
    soft_skills_required). Returns the English original when Indonesian
    translations are missing or partial."""
    if use_indonesian:
        v = comp.get(f"{field}_id")
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            return v
    return comp.get(field, []) or []


def _render_detail(
    comp: dict,
    br_lookup: dict,
    cluster_lookup: dict,
    job_titles: dict = None,
    target_stage: str = "Any",
    use_indonesian: bool = False,
):
    statement = _statement(comp, use_indonesian) or "(no statement)"
    # Use the FIRST sentence of the statement as the H2 heading so it
    # stays compact; the rest of the statement is rendered below as the
    # canonical body text. This replaces the legacy title+description
    # split: there is no "Description:" anymore, the statement IS the
    # competency.
    heading = statement.split(".", 1)[0].strip() or statement[:110]
    st.markdown(f"## {heading}")

    # Metadata row 1: Competency complexity / Labour-market demand /
    # Future weight / Source jobs. Splitting KKNI complexity from
    # education-stage demand is the v8 sprint Phase 3 methodological fix —
    # the two are INDEPENDENT labels and must not be presented as one
    # compound chip.
    cols = st.columns(4)
    kkni = comp.get("kkni_level")
    kkni_descr = _kkni_descriptor(kkni)
    cols[0].markdown(
        f"**Competency complexity**<br>"
        f"<span style='color:{_kkni_color(kkni)}; font-size:24px; font-weight:bold'>KKNI {kkni or '?'}</span>"
        f"<br><span style='font-size:11px; color:#666'>{kkni_descr or '—'}</span>"
        f"<br><span style='font-size:10px; color:#888'>source: {comp.get('kkni_level_source','?')}</span>",
        unsafe_allow_html=True,
    )
    # Labour-market demand: surface the dominant education stage(s) from
    # Phase 2.4 right next to the KKNI complexity cell so the user can see
    # both facts side-by-side. INDEPENDENT of KKNI.
    edu = comp.get("education_levels_demanded") or {}
    if edu:
        top_two = sorted(edu.items(), key=lambda x: -x[1])[:2]
        demand_line = " · ".join(f"{stage} {frac*100:.0f}%" for stage, frac in top_two)
        n_disclosed = comp.get("n_jobs_with_education", 0)
        n_total = len(comp.get("source_job_ids", []))
        demand_caption = f"from {n_disclosed}/{n_total} postings disclosing"
    else:
        demand_line = "—"
        demand_caption = "no education data"
    cols[1].markdown(
        f"**Labour-market demand**<br>"
        f"<span style='font-size:18px; font-weight:bold'>{demand_line}</span>"
        f"<br><span style='font-size:11px; color:#666'>{demand_caption}</span>"
        f"<br><span style='font-size:10px; color:#888'>independent of KKNI</span>",
        unsafe_allow_html=True,
    )
    fg = comp.get("future_weight", 0.0)
    fg_source = comp.get("future_weight_source") or "llm_suggested_not_evidence"
    public_fg = _public_future_weight(comp)
    cols[2].markdown(
        f"**Future weight**<br>"
        f"<span style='font-size:24px; font-weight:bold'>{public_fg:.2f}</span>"
        f"<br><span style='font-size:11px; color:#666'>raw {fg:.2f} | {comp.get('empirical_trend','?')}</span>"
        f"<br><span style='font-size:10px; color:#888'>{fg_source}</span>",
        unsafe_allow_html=True,
    )
    nj = len(comp.get("source_job_ids", []))
    if comp.get("n_source_jobs") is not None:
        nj = int(comp.get("n_source_jobs") or nj)
    demand = float(comp.get("demand_score") or 0.0)
    denominator = comp.get("eligible_job_denominator")
    prevalence_pct = _job_prevalence_pct(comp)
    if denominator:
        demand_caption = f"{prevalence_pct:.2f}% of {int(denominator)} postings"
    elif demand > 0:
        top_pct = max(1, int(round((1 - demand) * 100)))
        demand_caption = f"top {top_pct}% by demand"
    else:
        demand_caption = f"{len(comp.get('source_sentences',[]))} unique sentences"
    cols[3].markdown(
        f"**Source jobs**<br>"
        f"<span style='font-size:24px; font-weight:bold'>{nj}</span>"
        f"<br><span style='font-size:11px; color:#666'>{demand_caption}</span>",
        unsafe_allow_html=True,
    )

    # Grounding badge — now a TRANSPARENCY LABEL, not a gate (2026-05-13b
    # contract change). Three tiers: well-grounded / partial / thin. Users
    # see the tier and can filter via the sidebar, but the pipeline does not
    # drop low-grounding competencies — it drops fabrications via the
    # anti-hallucination filter instead, which preserves the demand signal.
    gs = comp.get("grounding_score", comp.get("grounding_score_preview", 0.0))
    tier_icon, tier_label, tier_color = _grounding_tier(gs)
    skill_g = comp.get("skill_grounding_score")
    sent_r = comp.get("sentence_relevance_score")
    if skill_g is not None and sent_r is not None and (skill_g != gs or sent_r != gs):
        breakdown = f"skill {float(skill_g):.2f} · sentence {float(sent_r):.2f}"
    else:
        breakdown = f"method: {comp.get('grounding_method','?')}"
    st.markdown(
        f"<span style='font-size:14px; color:#444'>**Evidence**: "
        f"<span style='color:{tier_color}; font-weight:bold'>{tier_icon} {tier_label}</span> "
        f"<span style='font-size:11px; color:#666'>· grounding {gs:.2f} · {breakdown}</span></span>",
        unsafe_allow_html=True,
    )

    if comp.get("scientific_sprint1_schema_version"):
        class_label = _competency_class_label(comp)
        support_label = _support_tier_label(comp)
        status_label = _recommendation_status_label(comp)
        occ_label = comp.get("occupation_mapping_label") or comp.get("occupation_mapping_status") or "unknown"
        if comp.get("official_core_eligible"):
            st.success(
                f"Sprint-1 core candidate #{comp.get('official_core_rank')}: "
                f"{class_label}; {support_label}; {prevalence_pct:.2f}% prevalence; "
                f"occupation mapping: {occ_label}."
            )
        else:
            blockers = comp.get("official_core_blockers") or []
            blocker_text = ", ".join(blockers) if blockers else "not selected by sprint-1 gates"
            st.caption(
                f"Sprint-1 class: {class_label}; support: {support_label}; "
                f"prevalence: {prevalence_pct:.2f}%; occupation mapping: {occ_label}. "
                f"Not in core candidate list: {blocker_text}."
            )
        readiness_blockers = comp.get("official_readiness_blockers") or []
        if comp.get("recommendation_status") == "official_ready":
            st.success("Recommendation status: official ready.")
        elif readiness_blockers:
            st.warning(
                f"Recommendation status: {status_label}. Official-readiness blockers: "
                f"{', '.join(readiness_blockers)}."
            )
        else:
            st.info(f"Recommendation status: {status_label}.")

    if comp.get("cluster_validity_status") or comp.get("theme_evidence_status"):
        audit_bits = []
        if comp.get("cluster_validity_status"):
            audit_bits.append(f"cluster: {comp.get('cluster_validity_status')}")
        if comp.get("technical_theme_label") or comp.get("theme_evidence_status"):
            theme = comp.get("technical_theme_label") or "theme"
            audit_bits.append(f"{theme}: {comp.get('theme_evidence_status') or 'unclassified'}")
        if comp.get("not_recommended_reason"):
            audit_bits.append(f"reason: {comp.get('not_recommended_reason')}")
        st.caption("Scientific audit: " + " | ".join(audit_bits))

    if comp.get("repair_decision") or comp.get("repair_flags"):
        repair_decision = comp.get("repair_decision") or "unmarked"
        repair_flags = comp.get("repair_flags") or []
        dedup_group = comp.get("curriculum_dedup_group") or ""
        recovered = bool(comp.get("recovered_from_failed"))
        repair_bits = [f"decision: {repair_decision}"]
        if recovered:
            repair_bits.append("recovered from failed high-support evidence")
        if dedup_group:
            repair_bits.append(f"dedup group: {dedup_group}")
        if comp.get("source_failure_reason"):
            repair_bits.append(f"source failure: {comp.get('source_failure_reason')}")
        st.caption("Repair audit: " + "; ".join(repair_bits) + ".")
        if repair_flags:
            st.caption("Repair flags: " + ", ".join(str(f) for f in repair_flags))

    # Caption explaining the independence of KKNI complexity vs labour-market
    # demand. Shown once per detail page, right under the metadata row.
    st.caption(
        "ℹ️ **Competency complexity (KKNI)** describes the cognitive demand of "
        "the work itself. **Labour-market demand** describes the education stage "
        "employers currently request. These are independent labels — a D3 "
        "institution can teach S2 competencies; a KKNI-7 competency can be in "
        "demand at SMK or D3. They do not influence each other."
    )

    # v9 sprint: target-stage fit chip. Shown only when user picked a stage
    # in the sidebar. Doesn't gate anything — pure UX hint.
    fit = _target_stage_fit(target_stage, comp.get("kkni_level"))
    if fit:
        sev = fit["severity"]
        if sev == "warning":
            st.warning(f"⚠️ {fit['msg']}")
        elif sev == "info":
            st.info(f"ℹ️ {fit['msg']}")
        elif sev == "match":
            st.success(f"✅ {fit['msg']}")
        else:  # below
            st.caption(f"⬇ {fit['msg']}")

    # Full statement (verb-led, 2-4 sentences). Replaces the legacy
    # "Description:" label which was a redundant duplicate of the title.
    st.markdown(statement)

    # v9.10: SKKNI occupation matches. Each chip = one Indonesian
    # occupational standard the competency maps to (top-K, cosine >=
    # threshold). An empty list means no SKKNI ICT match — surfaced as a
    # caption so users can see the honesty.
    occ_matches = comp.get("occupation_matches") or []
    if occ_matches:
        st.markdown("### Maps to (Indonesian occupational standards)")
        chips = []
        any_community = False
        for m in occ_matches:
            name = m.get("occupation_en", "?")
            score = m.get("cosine", 0.0)
            # Mark community-defined entries with a dagger (†) so the chip
            # carries its provenance even before the user expands details.
            marker = "" if m.get("is_official", True) else " †"
            if not m.get("is_official", True):
                any_community = True
            chips.append(f"`{name}{marker}` ({score:.2f})")
        st.markdown(" · ".join(chips))
        if any_community:
            st.caption(
                "† = Community-defined occupation (not an official "
                "Kepmenaker SKKNI). Added to cover gaps in the current ICT "
                "catalogue — see file `DATA/skema_okupasi_community_additions.csv` "
                "for the full list."
            )
        with st.expander("Why these occupations? ▾"):
            for m in occ_matches:
                official_tag = "" if m.get("is_official", True) else " *(community-defined, non-official)*"
                st.markdown(
                    f"**{m.get('occupation_en','?')}**{official_tag} — "
                    f"_{m.get('occupation_id','')}_  "
                    f"(cosine {m.get('cosine', 0.0):.3f})"
                )
                desc = m.get("description_en", "")
                if desc:
                    st.caption(desc)
    else:
        st.caption(
            "ℹ️ No SKKNI ICT occupation match above the threshold — this "
            "competency may not anchor to a current Indonesian occupational "
            "standard (could be cross-occupational, leadership-level, or a "
            "skill the SKKNI catalogue doesn't yet cover, e.g. cloud "
            "architect, ML engineer, frontend specialist)."
        )

    # Statement evidence concerns (anti-hallucination audit). After the
    # 2026-05-13b contract change, severity="warning" entries cause the
    # competency to be DROPPED upstream — so anything reaching the dashboard
    # is "info" only (a term that appears in the statement but with fewer
    # source-sentence hits than the threshold prefers). Reads from the new
    # `statement_evidence_concerns` field; falls back to the legacy
    # `title_evidence_concerns` for older run directories.
    concerns = (
        comp.get("statement_evidence_concerns")
        or comp.get("title_evidence_concerns")
        or []
    )
    if concerns:
        infos = [c for c in concerns if c.get("severity") == "info"]
        warnings = [c for c in concerns if c.get("severity") == "warning"]
        if warnings:
            # Should only appear for legacy run dirs (pre-contract-switch).
            for c in warnings:
                st.warning(
                    f"⚠️ Statement mentions **{c['term']}** but it appears in "
                    f"only **{c['literal_hits']} of {c['n_sentences']}** source sentences "
                    f"(threshold {c['min_required']}). Likely fabrication."
                )
        if infos:
            terms = ", ".join(f"`{c['term']}` ({c['literal_hits']}/{c['n_sentences']})" for c in infos)
            st.caption(f"ℹ️ Limited literal evidence for: {terms}")

    # Related skills
    st.markdown("### Related skills")
    rs = _localize_list(comp, "related_skills", use_indonesian)
    if rs:
        st.markdown("\n".join(f"- {s}" for s in rs))
    else:
        st.write("_no related skills_")

    # Soft skills
    soft = _localize_list(comp, "soft_skills_required", use_indonesian)
    if soft:
        st.markdown("### Soft skills required")
        st.markdown(", ".join(f"`{s}`" for s in soft))
        sd = _localize(comp, "soft_skills_description", use_indonesian)
        if sd:
            st.caption(sd)

    # Education-level demand (Phase 2.4)
    edu = comp.get("education_levels_demanded") or {}
    if edu:
        st.markdown("### Education-level demand")
        st.caption(
            f"From {comp.get('n_jobs_with_education', 0)} of {len(comp.get('source_job_ids', []))} "
            f"source job postings that disclosed education requirements"
        )
        for stage, frac in sorted(edu.items(), key=lambda x: -x[1]):
            st.markdown(f"`{stage}` — {frac*100:.0f}%")
            st.markdown(_bar(frac, "#4477aa"), unsafe_allow_html=True)

    # Role-shape diversity (Phase 2.1 Tier 2). Shows the histogram of source
    # job_titles bucketed into broad role families — auditing whether the
    # competency draws from a narrow role band (e.g., only DevOps engineers)
    # or spans broadly. Job posting titles are unique per posting, so we
    # normalize into role buckets via _normalize_role.
    if job_titles:
        src_jobs = comp.get("source_job_ids") or []
        role_counts: dict = {}
        unknown = 0
        sample_titles_per_role: dict = {}
        for j in src_jobs:
            t = job_titles.get(j) or ""
            if not t:
                unknown += 1
                continue
            role = _normalize_role(t)
            role_counts[role] = role_counts.get(role, 0) + 1
            sample_titles_per_role.setdefault(role, []).append(t)
        if role_counts:
            st.markdown("### Source role distribution")
            n_resolved = sum(role_counts.values())
            uniq_roles = len(role_counts)
            st.caption(
                f"{n_resolved} of {len(src_jobs)} source postings resolved to a job title, "
                f"bucketed into {uniq_roles} role families"
                + (f" — {unknown} unmatched" if unknown else "")
            )
            shown = sorted(role_counts.items(), key=lambda x: -x[1])
            max_n = shown[0][1] if shown else 1
            for role, cnt in shown:
                frac_share = cnt / n_resolved if n_resolved else 0.0
                frac_bar = cnt / max_n if max_n else 0.0
                st.markdown(f"`{role}` — {cnt} ({frac_share*100:.0f}%)")
                st.markdown(_bar(frac_bar, "#aa7744"), unsafe_allow_html=True)
            # Show 2-3 example titles per top role to keep the panel auditable
            with st.expander("Example postings per role ▾"):
                for role, _ in shown[:5]:
                    examples = sample_titles_per_role.get(role, [])[:3]
                    if examples:
                        st.markdown(f"**{role}**")
                        for ex in examples:
                            st.caption(f"• {ex}")

    # Rationale (read-more toggle)
    rationale = _localize(comp, "rationale", use_indonesian)
    st.markdown("### Why this competency? (LLM rationale)")
    if rationale:
        teaser = rationale[:200] + ("…" if len(rationale) > 200 else "")
        st.caption(teaser)
        if len(rationale) > 200:
            with st.expander("Read more ▾"):
                st.write(rationale)
    else:
        st.write("_no rationale_")

    # Batch reasoning (public via toggle, per resolved Phase 2.2 design Q#2)
    br_id = comp.get("batch_reasoning_id") or ""
    br = br_lookup.get(br_id)
    if br:
        with st.expander("How the LLM grouped these skills (batch reasoning) ▾"):
            st.caption(
                f"Model: `{br.get('model','?')}` ({br.get('provider','?')}) — "
                f"latency {br.get('latency_seconds',0):.1f}s, "
                f"competencies generated from this cluster: {br.get('n_competencies_out','?')}"
            )
            st.write(br.get("batch_reasoning") or "_(no batch_reasoning recorded)_")

    # Cluster context. Streamlit forbids nested expanders, so the inner
    # "show items" is gated by a checkbox instead.
    cid = comp.get("cluster_id") or ""
    cl = cluster_lookup.get(cid)
    if cl:
        with st.expander("Source cluster details ▾"):
            st.caption(
                f"Cluster `{cid}` (stream: {cl.get('stream','?')}, method: {cl.get('method','?')}) — "
                f"{cl.get('n_items','?')} items, cohesion {cl.get('cohesion_score','?'):.3f}"
                if isinstance(cl.get('cohesion_score'), (int, float))
                else f"Cluster `{cid}`"
            )
            st.write(f"**Heuristic label:** {cl.get('summary_label','?')}")
            if cl.get("top_terms"):
                st.write(f"**Top terms:** {', '.join(cl['top_terms'])}")
            if cl.get("items"):
                # Checkbox key includes cid + comp id so multiple competencies
                # from the same cluster get independent toggles.
                show_items = st.checkbox(
                    f"Show all {len(cl['items'])} cluster items",
                    key=f"show_items_{comp.get('id','')}_{cid}",
                )
                if show_items:
                    for it in cl["items"][:50]:
                        st.markdown(f"- {it}")
                    if len(cl["items"]) > 50:
                        st.caption(f"... and {len(cl['items']) - 50} more")

    # Provenance chain
    sents = comp.get("source_sentences") or []
    sids = comp.get("source_sentence_ids") or []
    job_ids = comp.get("source_job_ids") or []
    with st.expander(f"Provenance — {nj} jobs / {len(sents)} sentences ▾"):
        # If every "sentence" matches one of the related_skills verbatim, warn
        # the user that we're showing skill texts as a stand-in (happens with
        # legacy data that didn't preserve full sentence_text).
        related = {(s or "").strip().lower() for s in (comp.get("related_skills") or [])}
        skill_proxies = sum(
            1 for s in sents if (s or "").strip().lower() in related
        )
        if sents and skill_proxies == len(sents):
            st.caption(
                "ℹ️ This competency's source data didn't preserve full sentence "
                "text — showing the extracted skill phrases as a stand-in. Re-run "
                "the pipeline on Phase 1 output that includes `sentence_text` "
                "(e.g., from the offline Skill-LLM batch) for verbatim provenance."
            )
        # Render
        for i, s in enumerate(sents, 1):
            st.markdown(f"{i}. _{s}_")
        # And the unique job-ID list for full traceability
        if job_ids:
            st.caption(f"Source jobs: " + ", ".join(f"`{j}`" for j in job_ids[:25]) +
                       (f" + {len(job_ids) - 25} more" if len(job_ids) > 25 else ""))

    # --- User ratings (Phase 2.6 sub-deliverable) ---
    _render_rating_block(comp)


def _render_rating_block(comp: dict):
    """Light-signup-gated rating widget + public aggregate display.

    Per locked design:
      - rating does NOT affect priority_score / ranking (research signal only)
      - anonymous browse is free; rating requires light signup
      - one rating per (user, competency); overwrites on re-submit
    """
    from user_ratings import (
        VALID_ROLES,
        find_or_create_user,
        get_aggregate,
        get_user_rating,
        submit_rating,
    )

    st.markdown("### User ratings")

    comp_id = comp.get("id", "")
    agg = get_aggregate(comp_id)

    if agg.rating_count == 0:
        st.caption("No ratings yet. Be the first — your feedback helps us improve future versions.")
    else:
        cols = st.columns(3)
        cols[0].metric(
            "Average",
            f"{agg.rating_mean:.1f} / 5",
            f"σ {agg.rating_std:.2f}" if agg.rating_count > 1 else None,
        )
        cols[1].metric("Ratings", str(agg.rating_count))
        # Distribution
        dist_str = " · ".join(
            f"{k}★:{agg.distribution[k]}" for k in range(5, 0, -1)
        )
        cols[2].caption(dist_str)

        if agg.by_role:
            role_summary = " | ".join(
                f"{role}: {info['mean']:.1f} (n={info['count']})"
                for role, info in sorted(agg.by_role.items())
            )
            st.caption(f"By role: {role_summary}")

    # --- Signup / rating widget ---
    if "rating_user" not in st.session_state:
        st.session_state["rating_user"] = None

    if st.session_state["rating_user"] is None:
        with st.expander("Sign in to rate (one-time, takes 10 seconds) ▾"):
            with st.form(key=f"signup_form_{comp_id}"):
                email = st.text_input("Email", placeholder="you@example.com")
                role = st.selectbox("Your role", options=list(VALID_ROLES))
                signup_ok = st.form_submit_button("Sign in")
            if signup_ok and email:
                try:
                    user = find_or_create_user(email, role)
                    st.session_state["rating_user"] = user
                    st.rerun()
                except Exception as e:
                    st.error(f"Sign-in failed: {e}")
    else:
        user = st.session_state["rating_user"]
        existing = get_user_rating(user["user_id"], comp_id)
        default_rating = int(existing["rating"]) if existing else 4
        default_text = existing["feedback_text"] if existing else ""

        with st.form(key=f"rate_form_{comp_id}"):
            st.caption(f"Signed in as `{user['role']}`. Ratings are aggregated; identity is never shown publicly.")
            r = st.slider(
                "Your rating", min_value=1, max_value=5,
                value=default_rating, step=1,
            )
            txt = st.text_area(
                "Optional feedback (≤ 2000 chars)",
                value=default_text, max_chars=2000, height=80,
                placeholder="Anything you'd like the curriculum team to know about this competency?",
            )
            cols = st.columns([1, 1, 3])
            submit = cols[0].form_submit_button("Submit rating")
            sign_out = cols[1].form_submit_button("Sign out")
        if submit:
            try:
                submit_rating(
                    user_id=user["user_id"],
                    role=user["role"],
                    competency_id=comp_id,
                    rating=int(r),
                    feedback_text=txt,
                    pipeline_run_tag=comp.get("generator_version", ""),
                )
                st.success(
                    "Thanks! Your rating is recorded. "
                    "It will not change the live ranking but helps us improve future versions."
                )
                st.rerun()
            except Exception as e:
                st.error(f"Could not save rating: {e}")
        if sign_out:
            st.session_state["rating_user"] = None
            st.rerun()


# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #


def _render_pipeline_audit_page(run_dir: Path):
    """Pipeline-audit page (v8.1) — surfaces refinement moves and the full
    noise audit so curriculum designers can see which items were extracted
    but didn't reach a competency, and why.
    """
    st.markdown("# Pipeline audit")
    st.caption(
        "Transparency view: every item the extractor surfaced that did NOT "
        "reach a competency, plus every refinement move the LLM arbiter made. "
        "Helps spot extractor noise, generic-skill pollution, and "
        "cluster-membership decisions you might want to second-guess."
    )

    # Load the three audit sources
    pr_path = run_dir / "pipeline_report.json"
    na_path = run_dir / "noise_audit.json"
    it_path = run_dir / "it_relevance_audit.json"  # v9 sprint
    pipeline_report: dict = {}
    noise_audit: dict = {}
    it_audit: dict = {}
    try:
        if pr_path.exists():
            pipeline_report = json.loads(pr_path.read_text(encoding="utf-8"))
    except Exception as e:
        st.warning(f"Could not load pipeline_report.json: {e}")
    try:
        if na_path.exists():
            noise_audit = json.loads(na_path.read_text(encoding="utf-8"))
    except Exception as e:
        st.warning(f"Could not load noise_audit.json: {e}")
    try:
        if it_path.exists():
            it_audit = json.loads(it_path.read_text(encoding="utf-8"))
    except Exception as e:
        st.warning(f"Could not load it_relevance_audit.json: {e}")

    corpus_report = _load_corpus_report(run_dir)
    _render_corpus_notice([], corpus_report)
    if corpus_report:
        st.markdown("## Corpus lineage")
        cols = st.columns(4)
        cols[0].metric("Raw rows", corpus_report.get("raw_row_count", 0))
        cols[1].metric("Metadata jobs", corpus_report.get("metadata_unique_job_ids", 0))
        cols[2].metric("Phase-1 jobs", corpus_report.get("phase1_unique_job_ids", 0))
        cols[3].metric("Duplicates", f"{_safe_float(corpus_report.get('duplicate_rate')):.2%}")
        st.caption(
            f"Raw CSV: `{corpus_report.get('raw_csv_path') or 'unknown'}` | "
            f"Prepared: `{corpus_report.get('prepared_dir') or 'unknown'}` | "
            f"Skill-LLM output: `{corpus_report.get('skill_llm_output_path') or 'unknown'}`"
        )
        with st.expander("Show job-ID coverage checks"):
            coverage = [
                corpus_report.get("job_id_overlap_raw_to_metadata") or {},
                corpus_report.get("job_id_overlap_metadata_to_phase1") or {},
                corpus_report.get("job_id_overlap_metadata_to_skill_llm_input") or {},
                corpus_report.get("job_id_overlap_skill_llm_input_to_output") or {},
            ]
            st.dataframe(coverage, use_container_width=True, hide_index=True)

    stage_coverage = _load_run_json(run_dir, "stage_coverage_report.json", {})
    extraction_quality = _load_run_json(run_dir, "extraction_quality_audit.json", {})
    cluster_validity = _load_run_json(run_dir, "cluster_validity_report.json", {})
    demand_theme = _load_run_json(run_dir, "demand_theme_audit.json", {})
    if stage_coverage or extraction_quality or cluster_validity or demand_theme:
        st.markdown("## Evidence-first scientific audits")
        cols = st.columns(4)
        cols[0].metric("Labor denominator", stage_coverage.get("labor_market_denominator", 0))
        cols[1].metric("Non-hard extraction spans", extraction_quality.get("n_not_hard_technical", 0))
        cols[2].metric("Split candidates", cluster_validity.get("n_split_candidates", 0))
        cols[3].metric(
            "Theme recovery candidates",
            len(demand_theme.get("recovery_candidate_theme_ids") or []),
        )
        if stage_coverage.get("coverage_blockers"):
            st.warning("Coverage blockers: " + ", ".join(stage_coverage.get("coverage_blockers") or []))
        with st.expander("Stage coverage rows"):
            st.dataframe(stage_coverage.get("stage_rows") or [], use_container_width=True, hide_index=True)
        with st.expander("Extraction type distribution"):
            st.json(extraction_quality.get("type_distribution") or {}, expanded=False)
        with st.expander("Cluster validity distribution"):
            st.json(cluster_validity.get("status_distribution") or {}, expanded=False)
        with st.expander("Demand theme audit"):
            st.dataframe(demand_theme.get("themes") or [], use_container_width=True, hide_index=True)

    _render_pipeline_charts(run_dir, pipeline_report, noise_audit)
    st.markdown("---")

    # ---- IT-relevance gate section (v9 sprint) ----
    if it_audit:
        st.markdown("## IT-relevance gate")
        s = it_audit.get("summary", {}) or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Items in", s.get("n_items_in", 0))
        c2.metric("Items kept", s.get("n_items_kept", 0),
                  delta=f"−{s.get('n_items_in',0) - s.get('n_items_kept',0)}")
        c3.metric("Jobs IT / total",
                  f"{s.get('n_jobs_it', 0)} / {s.get('n_jobs_total', 0)}")
        c4.metric("Sentences IT / total",
                  f"{s.get('n_sentences_it', 0)} / {s.get('n_sentences_total', 0)}")

        dropped_jobs = it_audit.get("dropped_jobs", []) or []
        if dropped_jobs:
            with st.expander(f"Dropped jobs ({len(dropped_jobs)}) — non-IT postings"):
                for dj in dropped_jobs[:200]:
                    st.markdown(f"`{dj.get('job_id','?')}` — dropped **{dj.get('n_items_dropped',0)}** items")
                    excerpt = dj.get("description_excerpt", "")
                    if excerpt:
                        st.caption(excerpt)
                if len(dropped_jobs) > 200:
                    st.caption(f"... and {len(dropped_jobs) - 200} more")

        dropped_sents = it_audit.get("dropped_sentences", []) or []
        if dropped_sents:
            with st.expander(f"Dropped sentences ({len(dropped_sents)}) — boilerplate / non-IT within IT postings"):
                q = st.text_input("Search dropped sentences", "", key="it_dropped_search")
                ql = q.lower().strip()
                shown = [s_ for s_ in dropped_sents if not ql or ql in (s_.get("sentence_text","") or "").lower()]
                st.caption(f"Showing **{len(shown)}** of {len(dropped_sents)}.")
                for ds in shown[:200]:
                    st.markdown(f"- `{ds.get('sentence_id','?')}` (×{ds.get('n_items_dropped',0)}): _{ds.get('sentence_text','')}_")
                if len(shown) > 200:
                    st.caption(f"... and {len(shown) - 200} more (showing first 200)")

        st.markdown("---")

    # ---- Refinement moves section ----
    st.markdown("## Cluster refinement (LLM-as-arbiter)")
    ref = (pipeline_report.get("stages", {}).get("phase2_1_clustering") or {}).get("refinement")
    if not ref:
        st.info(
            "No refinement audit found for this run. Either refinement was "
            "disabled (default) or this is a legacy run from before v8."
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clusters processed", ref.get("n_clusters_processed", 0))
        c2.metric("Items moved OUT", ref.get("total_items_moved_out", 0))
        c3.metric("Items moved IN",  ref.get("total_items_moved_in", 0))
        c4.metric("LLM calls", ref.get("total_llm_calls", 0))
        c5, c6 = st.columns(2)
        c5.metric(
            "Mean cohesion before",
            f"{ref.get('cohesion_mean_before', 0):.3f}",
        )
        c6.metric(
            "Mean cohesion after",
            f"{ref.get('cohesion_mean_after', 0):.3f}",
            delta=f"{ref.get('cohesion_mean_after', 0) - ref.get('cohesion_mean_before', 0):+.3f}",
        )

        changed = [pc for pc in ref.get("per_cluster", []) if (pc.get("n_items_moved_out", 0) + pc.get("n_items_moved_in", 0)) > 0]
        if changed:
            st.markdown(f"### Clusters with moves ({len(changed)} of {ref.get('n_clusters_processed', 0)})")
            for pc in changed:
                with st.expander(
                    f"cluster_idx={pc['cluster_idx']} · out={pc['n_items_moved_out']} · in={pc['n_items_moved_in']} · "
                    f"cohesion {pc['initial_cohesion']:.3f} → {pc['final_cohesion']:.3f}"
                ):
                    for m in pc.get("move_decisions", []):
                        kind = m.get("kind", "?")
                        verdict = m.get("verdict", "?")
                        text = m.get("text", "")
                        reason = m.get("reason", "")
                        icon = "⬅" if verdict == "move_out" else "➡" if verdict == "join" else "·"
                        st.markdown(f"{icon} **{verdict}** `{text}` — _{reason}_")
        else:
            st.caption("No clusters were modified — every item stayed where HDBSCAN placed it.")

    st.markdown("---")

    # ---- Noise audit section ----
    st.markdown("## Items that did NOT reach a competency")
    if not noise_audit:
        st.info(
            "No noise_audit.json found for this run. v8.1+ runs include it; "
            "legacy runs do not."
        )
        return

    n_total = noise_audit.get("n_items_dropped_total", 0)
    by_stage: dict = noise_audit.get("by_stage", {}) or {}
    st.caption(
        f"**{n_total} items** dropped in total across all Phase 2.1 stages. "
        "Each item below has the canonical text + the stage at which it was "
        "dropped + the reason."
    )

    # Stage summary chart (text bars)
    if by_stage:
        max_n = max(by_stage.values()) or 1
        st.markdown("### Drops by stage")
        for stage, n in sorted(by_stage.items(), key=lambda x: -x[1]):
            frac = n / max_n
            st.markdown(f"`{stage}` — **{n}**")
            st.markdown(_bar(frac, "#888"), unsafe_allow_html=True)

    # Per-stage filter + searchable item list
    items = noise_audit.get("items", []) or []
    if not items:
        return
    st.markdown("### Browse dropped items")
    stages = sorted(set(it.get("stage", "") for it in items))
    chosen_stages = st.multiselect(
        "Filter by stage", options=stages, default=stages,
    )
    q = st.text_input("Search by item text", "")
    q_lower = q.lower().strip()

    filtered = [
        it for it in items
        if it.get("stage") in chosen_stages
        and (not q_lower or q_lower in (it.get("text", "") or "").lower())
    ]
    st.caption(f"Showing **{len(filtered)} of {len(items)}** items.")

    # Group by stage, show items in expanders to avoid one huge wall of text
    by_stage_items: dict = {}
    for it in filtered:
        by_stage_items.setdefault(it.get("stage", ""), []).append(it)
    for stage in sorted(by_stage_items.keys()):
        stage_items = by_stage_items[stage]
        with st.expander(f"`{stage}` — {len(stage_items)} items"):
            for it in stage_items[:200]:  # cap per stage to keep page responsive
                txt = it.get("text", "")
                reason = it.get("reason", "")
                njobs = it.get("n_unique_jobs", 0)
                extra = it.get("extra") or {}
                meta_bits = []
                if njobs:
                    meta_bits.append(f"in {njobs} jobs")
                if extra:
                    meta_bits.extend(f"{k}={v}" for k, v in list(extra.items())[:3])
                meta = " · ".join(meta_bits)
                st.markdown(f"- **{txt}** — _{reason}_" + (f" ({meta})" if meta else ""))
            if len(stage_items) > 200:
                st.caption(f"... and {len(stage_items) - 200} more (showing first 200)")


def _render_coverage_page(comps):
    """Curriculum upload + coverage report — Phase 2.6 sub-deliverable."""
    st.markdown("# Curriculum coverage analysis")
    st.caption(
        "Upload your existing curriculum (PDF or plain text). The system will "
        "match each competency against the curriculum content and show which "
        "ones are already covered, partially covered, or missing — with "
        "priority weighting by future-of-work demand."
    )

    upload = st.file_uploader(
        "Upload curriculum (PDF or .txt)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=False,
    )
    pasted = st.text_area(
        "...or paste curriculum text directly",
        height=200,
        placeholder="Paste the curriculum / syllabus / module list here. Each paragraph or numbered item is treated as one section.",
    )

    if not upload and not pasted.strip():
        st.info("Upload a curriculum or paste text to see the coverage analysis.")
        return

    from curriculum_coverage import (
        compute_coverage,
        parse_curriculum_pdf,
        parse_curriculum_text,
        translate_sections_to_english,
    )

    auto_translate = st.checkbox(
        "Auto-translate Indonesian sections to English (LLM, ~1 sec/section)",
        value=True,
        help="Detects each section's language and translates Indonesian ones to English "
             "before matching against the competency database. Skipped sections are kept as-is.",
    )

    with st.spinner("Parsing curriculum..."):
        if upload is not None:
            data_bytes = upload.read()
            if upload.name.lower().endswith(".pdf"):
                sections = parse_curriculum_pdf(data_bytes)
            else:
                sections = parse_curriculum_text(data_bytes.decode("utf-8", errors="replace"))
        else:
            sections = parse_curriculum_text(pasted)

    if not sections:
        st.error(
            "Couldn't extract any sections from this curriculum. Try plain text "
            "with numbered headings or paragraph breaks."
        )
        return

    if auto_translate:
        with st.spinner("Detecting language + translating Indonesian sections..."):
            sections, n_translated = translate_sections_to_english(sections)
        if n_translated:
            st.info(f"Translated {n_translated} Indonesian section(s) to English.")

    st.success(f"Parsed {len(sections)} curriculum sections.")
    with st.expander(f"Preview parsed sections ({len(sections)})"):
        for s in sections[:20]:
            st.markdown(f"**{s.title}** _(id: {s.section_id})_")
            st.caption(s.text[:300] + ("..." if len(s.text) > 300 else ""))
        if len(sections) > 20:
            st.caption(f"... and {len(sections) - 20} more sections")

    with st.spinner("Embedding and matching... (~10 s)"):
        annotations, report = compute_coverage(comps, sections)

    # --- Summary KPIs ---
    cols = st.columns(4)
    cols[0].metric("Well covered (≥ 0.65)", report.n_well_covered)
    cols[1].metric("Partially covered (0.45–0.65)", report.n_partially_covered)
    cols[2].metric("Missing (< 0.45)", report.n_missing)
    cols[3].metric("Weighted coverage", f"{report.weighted_mean_coverage:.2f}")

    # --- Top priority gaps (the "you should add" list) ---
    st.markdown("### Top priority-weighted gaps")
    st.caption(
        "Competencies that are missing or weakly covered AND have high future-of-work demand. "
        "Adding these has the highest impact on curriculum relevance."
    )
    annot_by_id = {a.competency_id: a for a in annotations}
    for a in report.top_gaps:
        c = next((c for c in comps if c.get("id") == a.competency_id), None)
        if c is None:
            continue
        label_color = {
            "missing": "#d73027",
            "partially_covered": "#fc8d59",
            "well_covered": "#1a9850",
        }.get(a.coverage_label, "#888")
        st.markdown(
            f"**{_statement_first_sentence(c, False, max_chars=120)}** — "
            f"<span style='color:{label_color}'>{a.coverage_label}</span> "
            f"(coverage {a.coverage_score:.2f}, future weight {c.get('future_weight',0):.2f}, "
            f"priority-weighted gap {a.priority_weighted_gap:.2f})",
            unsafe_allow_html=True,
        )
        if a.best_matching_section_title:
            st.caption(f"Best (weak) match: _{a.best_matching_section_title}_")

    # --- Full coverage table ---
    st.markdown("### Full coverage breakdown")
    rows = []
    for c in comps:
        a = annot_by_id.get(c.get("id"))
        if a is None:
            continue
        rows.append({
            "Competency": _statement_first_sentence(c, False, max_chars=120),
            "KKNI": c.get("kkni_level"),
            "Future weight": round(float(c.get("future_weight", 0.0)), 2),
            "Coverage": round(a.coverage_score, 2),
            "Label": a.coverage_label,
            "Best match": a.best_matching_section_title or "—",
            "Priority gap": round(a.priority_weighted_gap, 2),
        })
    rows.sort(key=lambda r: -r["Priority gap"])
    st.dataframe(rows, use_container_width=True, hide_index=True)


def main():
    st.set_page_config(
        page_title="Competency Recommendations (v2)",
        page_icon="📚",
        layout="wide",
    )

    run_dir = _resolve_run_dir()
    data = load_run(str(run_dir))
    comps = data["competencies"]
    brs = {br["id"]: br for br in data["batch_reasonings"]}
    clusters = {c["id"]: c for c in data["clusters"]}
    # job_id -> title for the role-shape panel. Empty dict when this run's
    # jobs_metadata isn't reachable (older e2e runs); the panel hides itself.
    job_titles = load_job_titles(str(run_dir))
    corpus_report = _load_corpus_report(run_dir)

    # Top-level page switcher
    page = st.sidebar.radio(
        "Page",
        options=["Browse competencies", "Market insights", "Curriculum coverage", "Pipeline audit"],
        label_visibility="visible",
    )
    if page == "Pipeline audit":
        _render_pipeline_audit_page(run_dir)
        return
    if page == "Market insights":
        _render_market_insights_page(comps, run_dir, corpus_report)
        return
    if page == "Curriculum coverage":
        _render_coverage_page(comps)
        return

    # --------- Sidebar ---------
    # v9 sprint: language toggle. Switches between English originals and
    # Indonesian translations (when present in the run, populated by
    # `--translate-output-to-id`). Translations live in `{field}_id` keys.
    lang = st.sidebar.radio(
        "Language / Bahasa",
        options=["English", "Bahasa Indonesia"],
        index=0,
        horizontal=True,
    )
    # v2 (2026-05-13b) renamed title → statement; older runs still have title_id.
    has_id_translations = any(
        c.get("statement_id") or c.get("title_id") for c in comps[:5]
    )
    if lang == "Bahasa Indonesia" and not has_id_translations:
        st.sidebar.warning(
            "This run wasn't built with `--translate-output-to-id`, so "
            "Indonesian translations aren't available. Showing English."
        )
        lang = "English"
    use_indonesian = (lang == "Bahasa Indonesia")
    has_sprint1 = _has_sprint1_fields(comps)
    corpus_ctx = _corpus_context(comps, corpus_report)

    st.sidebar.markdown("## Filters")
    st.sidebar.caption(f"Reading from: `{run_dir.name}`")
    st.sidebar.caption(f"{len(comps)} competencies, {len(brs)} LLM calls")
    st.sidebar.caption(f"Corpus: {_corpus_status_label(corpus_ctx['status'])}")

    # KKNI level filter
    all_levels = sorted({c.get("kkni_level") for c in comps if c.get("kkni_level")})
    chosen_levels = st.sidebar.multiselect(
        "KKNI level", options=all_levels, default=all_levels,
    )

    # v9 sprint: target curriculum stage. When the user picks a stage, every
    # competency gets a fit chip on its detail page (matches / above / below).
    # "Any" leaves the chips off — competency complexity stands alone.
    target_stage = st.sidebar.selectbox(
        "Target curriculum stage",
        options=["Any", "SMK", "D3", "D4 / S1", "S2"],
        index=0,
        help="When set, each competency gets a fit chip showing whether its "
             "cognitive complexity matches a curriculum at this stage. KKNI "
             "and education-stage demand remain independent labels — this is "
             "purely a UX hint.",
    )

    # v9.10: SKKNI occupation filter. The competencies' `occupation_matches`
    # contain top-K SKKNI occupations they map to; the filter narrows the
    # list to ones that map to any of the user's chosen occupations.
    all_occupations = sorted({
        m.get("occupation_en", "")
        for c in comps
        for m in (c.get("occupation_matches") or [])
        if m.get("occupation_en")
    })
    chosen_occupations = []
    if all_occupations:
        chosen_occupations = st.sidebar.multiselect(
            "Filter by SKKNI occupation",
            options=all_occupations,
            default=[],
            help="Show only competencies that map to one of these Indonesian "
                 "occupational standards (skema okupasi). When empty, all "
                 "competencies are shown.",
        )

    chosen_classes = []
    chosen_statuses = []
    official_core_only = False
    min_source_jobs = 0
    if has_sprint1:
        all_classes = sorted({_competency_class_label(c) for c in comps})
        chosen_classes = st.sidebar.multiselect(
            "Recommendation class",
            options=all_classes,
            default=all_classes,
            help="Sprint-1 classification separates core technical competencies "
                 "from process, employability/support, and out-of-domain signals.",
        )
        official_core_only = st.sidebar.checkbox(
            "Core technical candidates only",
            value=False,
            help="Show only competencies that pass the sprint-1 core technical "
                 "candidate gates.",
        )
        all_statuses = sorted({_recommendation_status_label(c) for c in comps})
        chosen_statuses = st.sidebar.multiselect(
            "Recommendation status",
            options=all_statuses,
            default=all_statuses,
            help="Official-ready requires Indonesian corpus validity, official mapping, expert review, and stability validation.",
        )
        max_source_jobs = max(
            (int(c.get("n_source_jobs") or len(c.get("source_job_ids") or [])) for c in comps),
            default=0,
        )
        min_source_jobs = st.sidebar.slider(
            "Min source jobs",
            0,
            max_source_jobs,
            0,
            1,
            help="True unique job-posting support, shown against the run denominator.",
        )

    # Future-weight filter
    min_fw = st.sidebar.slider("Min future weight", 0.0, 1.0, 0.0, 0.05)
    # Grounding-score filter. After the 2026-05-13b contract change grounding
    # is a transparency LABEL (not a gate), so the slider default is 0.0 —
    # users see everything in the top-N by demand, and can raise the floor
    # if they only want well-evidenced competencies.
    min_grounding = st.sidebar.slider(
        "Min grounding score (transparency filter)",
        0.0, 1.0, 0.0, 0.05,
        help="Grounding is shown as a badge (✅ well-grounded / ⚠️ partial / "
             "🚧 thin). Raise this slider to hide thin-evidence competencies. "
             "The pipeline itself does NOT gate on grounding — it drops "
             "fabrications via an anti-hallucination filter and ranks the "
             "rest by demand.",
    )

    # Search
    q = st.sidebar.text_input("Search statement / skills", "")

    # Sort. New runs default to the class-aware recommendation contract.
    # The original priority score remains available as a baseline/ablation
    # ranking for research comparison.
    sort_options = [
        "priority baseline (desc)",
        "demand (desc)",
        "future_weight (desc)",
        "grounding_score (desc)",
        "n source jobs (desc)",
        "statement (a-z)",
    ]
    if has_sprint1:
        sort_options.insert(0, "recommendation rank")
        sort_options.insert(2, "core candidate rank")
        sort_options.insert(4, "job prevalence (desc)")
    sort_by = st.sidebar.selectbox(
        "Sort by",
        options=sort_options,
        index=0,
    )

    # --------- Filter logic ---------
    def _match(c):
        if c.get("kkni_level") not in chosen_levels:
            return False
        if c.get("future_weight", 0.0) < min_fw:
            return False
        if c.get("grounding_score", c.get("grounding_score_preview", 0.0)) < min_grounding:
            return False
        if has_sprint1:
            if chosen_classes and _competency_class_label(c) not in chosen_classes:
                return False
            if chosen_statuses and _recommendation_status_label(c) not in chosen_statuses:
                return False
            if official_core_only and not c.get("official_core_eligible"):
                return False
            n_jobs = int(c.get("n_source_jobs") or len(c.get("source_job_ids") or []))
            if n_jobs < min_source_jobs:
                return False
        if chosen_occupations:
            match_set = {
                m.get("occupation_en", "") for m in (c.get("occupation_matches") or [])
            }
            if not match_set.intersection(chosen_occupations):
                return False
        if q:
            blob = " ".join([
                c.get("statement", ""),
                # Legacy back-compat for runs that pre-date the schema change
                c.get("title", ""), c.get("description", ""),
                " ".join(c.get("related_skills", [])), c.get("rationale", ""),
            ]).lower()
            if q.lower() not in blob:
                return False
        return True

    filtered = [c for c in comps if _match(c)]

    if sort_by == "recommendation rank":
        filtered.sort(
            key=lambda c: (
                int(c.get("class_sort_group") if c.get("class_sort_group") is not None else 99),
                int(c.get("class_rank") or 999999),
                -_job_prevalence_pct(c),
                -int(c.get("n_source_jobs") or len(c.get("source_job_ids") or [])),
                -float(c.get("grounding_score", c.get("grounding_score_preview", 0.0))),
                -_public_future_weight(c),
                -float(c.get("priority_score", 0.0)),
                _statement_first_sentence(c, False).lower(),
            )
        )
    elif sort_by == "priority baseline (desc)":
        # v8 sprint Phase 4. Falls back to grounding+future when priority_score
        # is missing (legacy runs).
        def _priority_key(c):
            ps = c.get("priority_score")
            if ps is not None and ps > 0:
                return -float(ps)
            # Legacy fallback
            return -(0.30 * float(c.get("grounding_score", 0.0))
                     + 0.30 * float(c.get("future_weight", 0.0)))
        filtered.sort(key=_priority_key)
    elif sort_by == "core candidate rank":
        filtered.sort(
            key=lambda c: (
                not bool(c.get("official_core_eligible")),
                int(c.get("official_core_rank") or 999999),
                -int(c.get("n_source_jobs") or len(c.get("source_job_ids") or [])),
                _statement_first_sentence(c, False).lower(),
            )
        )
    elif sort_by == "demand (desc)":
        filtered.sort(key=lambda c: -float(c.get("demand_score", 0.0)))
    elif sort_by == "job prevalence (desc)":
        filtered.sort(key=lambda c: -_job_prevalence_pct(c))
    elif sort_by == "future_weight (desc)":
        filtered.sort(key=lambda c: -float(c.get("future_weight", 0.0)))
    elif sort_by == "grounding_score (desc)":
        filtered.sort(key=lambda c: -float(c.get("grounding_score", c.get("grounding_score_preview", 0.0))))
    elif sort_by == "n source jobs (desc)":
        filtered.sort(key=lambda c: -len(c.get("source_job_ids", [])))
    else:
        # statement (a-z) — use the same short label the browse list shows
        filtered.sort(key=lambda c: _statement_first_sentence(c, False).lower())

    # --------- Header ---------
    st.markdown("# Competency Recommendations")
    st.caption(
        f"Cluster-driven curriculum competencies generated from real job-market data. "
        f"Showing **{len(filtered)} of {len(comps)}** competencies."
    )
    _render_corpus_notice(comps, corpus_report)

    # Two-column layout: left = list, right = detail
    list_col, detail_col = st.columns([0.4, 0.6])

    with list_col:
        st.markdown("### Browse")
        if not filtered:
            st.info("No competencies match the current filters.")
            return
        # Use radio for selection — single-select list. Show the first
        # sentence of each statement; a tiny chip line beneath shows the
        # KKNI level and a grounding tier badge (✅/⚠️/🚧).
        def _browse_label(i, c):
            first = _statement_first_sentence(c, use_indonesian, max_chars=70)
            gs = c.get("grounding_score", c.get("grounding_score_preview", 0.0))
            tier_icon, _label, _color = _grounding_tier(gs)
            if has_sprint1:
                class_label = _competency_class_label(c).replace(" / ", "/")
                status_label = _recommendation_status_label(c)
                jobs = int(c.get("n_source_jobs") or len(c.get("source_job_ids") or []))
                return (
                    f"{i+1}. {first}  "
                    f"({class_label}, {status_label}, {jobs} jobs, {tier_icon} grounding {gs:.2f})"
                )
            return (
                f"{i+1}. {first}  "
                f"(KKNI {c.get('kkni_level','?')}, {tier_icon} grounding {gs:.2f})"
            )
        labels = [_browse_label(i, c) for i, c in enumerate(filtered)]
        selected = st.radio(
            "Pick a competency",
            options=range(len(filtered)),
            format_func=lambda i: labels[i],
            label_visibility="collapsed",
        )

    with detail_col:
        if selected is not None:
            _render_detail(
                filtered[selected], brs, clusters,
                job_titles=job_titles, target_stage=target_stage,
                use_indonesian=use_indonesian,
            )


if __name__ == "__main__":
    main()
