"""
plot_competency_v2_run.py

Generate dashboard-friendly and research-grade static plots for one completed
competency-v2 pipeline run.

Example:
    .\\run.ps1 scripts\\plot_competency_v2_run.py ^
        --run-dir results\\competency_v2_pipeline_n1k_openrouter_gpt54

Outputs:
    results/figures/<run_tag>/dashboard/
    results/figures/<run_tag>/research/

If an evidence package exists at results/evidence_packages/<run_tag>/, the
figures are also copied into results/evidence_packages/<run_tag>/figures/.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _run_tag(run_dir: Path, report: dict[str, Any]) -> str:
    if report.get("tag"):
        return str(report["tag"])
    prefix = "competency_v2_pipeline_"
    return run_dir.name[len(prefix):] if run_dir.name.startswith(prefix) else run_dir.name


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _short(text: str, width: int = 56) -> str:
    return textwrap.shorten(" ".join(str(text or "").split()), width=width, placeholder="...")


def _ranked(competencies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if any(c.get("class_rank") for c in competencies):
        return sorted(
            competencies,
            key=lambda c: (
                int(c.get("class_sort_group") if c.get("class_sort_group") is not None else 99),
                int(c.get("class_rank") or 999999),
                -_as_float(c.get("job_prevalence")),
                -int(c.get("n_source_jobs") or len(c.get("source_job_ids") or [])),
                -_as_float(c.get("grounding_score")),
                -_as_float(c.get("future_weight")),
                -_as_float(c.get("priority_score")),
                str(c.get("id") or ""),
            ),
        )
    return sorted(
        competencies,
        key=lambda c: (
            -_as_float(c.get("priority_score")),
            -len(c.get("source_job_ids") or []),
            str(c.get("id") or ""),
        ),
    )


def _ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "figure.titlesize": 14,
        "savefig.bbox": "tight",
    })
    return plt


def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    import matplotlib.pyplot as plt

    plt.close(fig)


def _competency_rows(competencies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rank, c in enumerate(_ranked(competencies), 1):
        rows.append({
            "rank": rank,
            "id": c.get("id", ""),
            "statement": c.get("statement", ""),
            "statement_short": _short(c.get("statement", ""), 68),
            "priority": _as_float(c.get("priority_score")),
            "demand": _as_float(c.get("demand_score")),
            "future": _as_float(c.get("future_weight")),
            "grounding": _as_float(c.get("grounding_score")),
            "kkni": c.get("kkni_level"),
            "n_jobs": int(c.get("n_source_jobs") or len(c.get("source_job_ids") or [])),
            "job_prevalence_pct": _as_float(c.get("job_prevalence_pct")),
            "competency_class": c.get("competency_class") or "",
            "competency_class_label": c.get("competency_class_label") or "",
            "support_tier": c.get("support_tier") or "",
            "recommendation_status": c.get("recommendation_status") or "",
            "corpus_validity_status": c.get("corpus_validity_status") or "",
            "n_related_skills": len(c.get("related_skills") or []),
            "n_occupations": len(c.get("occupation_matches") or []),
        })
    return rows


def _clean_pngs(*dirs: Path) -> None:
    for d in dirs:
        if not d.exists():
            continue
        for p in d.glob("*.png"):
            p.unlink()


def _is_official_occupation_match(match: dict[str, Any]) -> bool:
    source = str(match.get("source_type") or "")
    return bool(match.get("is_official")) or source.lower().startswith("template-based")


def _occupation_status(c: dict[str, Any]) -> str:
    matches = c.get("occupation_matches") or []
    if not matches:
        return "Unmapped"
    if any(_is_official_occupation_match(m) for m in matches if isinstance(m, dict)):
        return "Official/template"
    return "Community-only"


_TECH_TERMS = {
    "api", "rest", "cloud", "azure", "google cloud", "devops", "ci/cd", "github",
    "git", "infrastructure as code", "iac", "machine learning", "ai/ml",
    "artificial intelligence", "database", "sql", "frontend", "front-end",
    "backend", "back-end", "mobile", "android", "ios", "swift", "software",
    "application development", "computer vision", "image processing",
    "distributed", "hpc", "system architecture", "security", "privacy",
    "data protection", "service cloud", "salesforce", "servicenow",
}

_PROCESS_TERMS = {
    "agile", "scrum", "sprint", "project management", "technical leadership",
    "leadership", "lead a team", "technical leader", "architectural direction",
}

_SUPPORT_TERMS = {
    "attention to detail", "detail-oriented", "problem-solving", "analytical",
    "collaboration", "cross-functional", "teamwork", "mentor", "coaching",
    "time management", "priorities", "fast-paced", "customer success",
    "working with customers",
}

_OUT_OF_DOMAIN_TERMS = {
    "finance", "budgeting", "accounting", "payroll", "internal audit",
    "electrical engineering", "soldering",
}


def _recommendation_type(c: dict[str, Any]) -> str:
    stored_label = c.get("competency_class_label")
    if stored_label:
        return str(stored_label)
    stored = c.get("competency_class")
    if stored == "core_technical":
        return "Core technical"
    if stored == "engineering_process":
        return "Engineering process / leadership"
    if stored == "employability_support":
        return "Employability / support"
    if stored == "out_of_domain_or_needs_review":
        return "Needs review / out-of-domain"

    text = " ".join([
        str(c.get("statement") or ""),
        " ".join(str(s) for s in (c.get("related_skills") or [])),
    ]).lower()
    if any(term in text for term in _OUT_OF_DOMAIN_TERMS):
        return "Needs review / out-of-domain"
    if any(term in text for term in _PROCESS_TERMS):
        return "Engineering process / leadership"
    if any(term in text for term in _TECH_TERMS):
        return "Core technical"
    if any(term in text for term in _SUPPORT_TERMS):
        return "Employability / support"
    if c.get("occupation_matches"):
        return "Core technical"
    return "Needs review / out-of-domain"


def plot_priority_top(
    competencies: list[dict[str, Any]],
    out_path: Path,
    top_n: int = 30,
) -> None:
    plt = _ensure_matplotlib()
    rows = _competency_rows(competencies)[:top_n]
    if not rows:
        return
    labels = [f"{r['rank']}. {r['statement_short']}" for r in rows][::-1]
    values = [r["priority"] for r in rows][::-1]

    fig_h = max(6.0, 0.34 * len(rows))
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.barh(labels, values, color="#4477aa")
    ax.set_xlabel("Priority score")
    ax.set_title(f"Top {len(rows)} competency recommendations by priority")
    ax.set_xlim(0, max(values) * 1.12 if values else 1)
    for y, v in enumerate(values):
        ax.text(v + 0.006, y, f"{v:.3f}", va="center", fontsize=8)
    _save(fig, out_path)


def plot_score_components_top(competencies: list[dict[str, Any]], out_path: Path, top_n: int = 15) -> None:
    plt = _ensure_matplotlib()
    rows = _competency_rows(competencies)[:top_n]
    if not rows:
        return
    labels = [f"{r['rank']}. {r['statement_short']}" for r in rows][::-1]
    demand = [0.40 * r["demand"] for r in rows][::-1]
    grounding = [0.30 * r["grounding"] for r in rows][::-1]
    future = [0.30 * r["future"] for r in rows][::-1]
    ypos = list(range(len(rows)))

    fig_h = max(5.5, 0.42 * len(rows))
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.barh(ypos, demand, color="#4477aa", label="0.40 demand")
    ax.barh(ypos, grounding, left=demand, color="#228833", label="0.30 grounding")
    left2 = [d + g for d, g in zip(demand, grounding)]
    ax.barh(ypos, future, left=left2, color="#cc9944", label="0.30 future")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Priority score contribution")
    ax.set_title(f"Priority score components for top {len(rows)}")
    ax.legend(loc="lower right")
    _save(fig, out_path)


def plot_job_support_top(competencies: list[dict[str, Any]], out_path: Path, top_n: int = 30) -> None:
    plt = _ensure_matplotlib()
    rows = _competency_rows(competencies)[:top_n]
    if not rows:
        return
    labels = [f"{r['rank']}. {r['statement_short']}" for r in rows][::-1]
    values = [r["n_jobs"] for r in rows][::-1]
    colors = ["#cc6677" if v < 10 else "#4477aa" for v in values]
    fig_h = max(6.0, 0.34 * len(rows))
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.barh(labels, values, color=colors)
    ax.axvline(10, color="#aa0000", linestyle="--", linewidth=1.0, label="review threshold: 10 jobs")
    ax.set_xlabel("Unique source jobs")
    ax.set_title(f"Evidence support for top {len(rows)} recommendations")
    ax.legend()
    for y, v in enumerate(values):
        ax.text(v + 0.25, y, str(v), va="center", fontsize=8)
    _save(fig, out_path)


def plot_recommendation_type_mix(competencies: list[dict[str, Any]], out_path: Path) -> None:
    plt = _ensure_matplotlib()
    counts = Counter(_recommendation_type(c) for c in _ranked(competencies))
    if not counts:
        return
    order = [
        "Core technical",
        "Engineering process / leadership",
        "Employability / support",
        "Needs review / out-of-domain",
    ]
    labels = [k for k in order if counts.get(k)]
    values = [counts[k] for k in labels]
    colors = ["#4477aa", "#66aa55", "#cc9944", "#cc6677"][:len(labels)]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Competencies")
    title_suffix = "" if any(c.get("competency_class") for c in competencies) else " (heuristic)"
    ax.set_title(f"Recommendation type mix{title_suffix}")
    ax.tick_params(axis="x", rotation=25)
    for i, v in enumerate(values):
        ax.text(i, v + 0.1, str(v), ha="center")
    _save(fig, out_path)


def plot_recommendation_status_mix(competencies: list[dict[str, Any]], out_path: Path) -> None:
    plt = _ensure_matplotlib()
    labels_by_status = {
        "exploratory": "Exploratory",
        "review_candidate": "Review candidate",
        "official_ready": "Official ready",
    }
    counts = Counter(c.get("recommendation_status") or "unknown" for c in _ranked(competencies))
    if not counts:
        return
    order = ["official_ready", "review_candidate", "exploratory", "unknown"]
    labels = [k for k in order if counts.get(k)]
    labels.extend(sorted(k for k in counts if k not in labels))
    values = [counts[k] for k in labels]
    display_labels = [labels_by_status.get(k, k) for k in labels]
    colors = ["#228833", "#4477aa", "#cc9944", "#999999"][:len(labels)]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(display_labels, values, color=colors)
    ax.set_ylabel("Competencies")
    ax.set_title("Recommendation status")
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(values):
        ax.text(i, v + 0.1, str(v), ha="center")
    _save(fig, out_path)


def plot_education_by_kkni_heatmap(competencies: list[dict[str, Any]], out_path: Path) -> None:
    plt = _ensure_matplotlib()
    stages = ["SMK", "SMA", "D1", "D2", "D3", "D4", "S1", "Profesi", "S2", "S3"]
    kkni_levels = sorted({int(c.get("kkni_level")) for c in competencies if c.get("kkni_level") is not None})
    if not kkni_levels:
        return
    matrix = [[0.0 for _ in stages] for _ in kkni_levels]
    level_index = {level: i for i, level in enumerate(kkni_levels)}
    for c in competencies:
        level = c.get("kkni_level")
        if level is None:
            continue
        edu = c.get("education_levels_demanded") or {}
        if not isinstance(edu, dict):
            continue
        for stage, value in edu.items():
            if stage in stages:
                matrix[level_index[int(level)]][stages.index(stage)] += _as_float(value)
    if not any(any(v > 0 for v in row) for row in matrix):
        return
    fig, ax = plt.subplots(figsize=(9, 4.6))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, rotation=35, ha="right")
    ax.set_yticks(range(len(kkni_levels)))
    ax.set_yticklabels([f"KKNI {lvl}" for lvl in kkni_levels])
    ax.set_title("Education-stage demand by assigned KKNI level")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value > 0:
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Summed demand share")
    _save(fig, out_path)


def plot_occupation_official_status(competencies: list[dict[str, Any]], out_path: Path) -> None:
    plt = _ensure_matplotlib()
    counts = Counter(_occupation_status(c) for c in competencies)
    if not counts:
        return
    order = ["Official/template", "Community-only", "Unmapped"]
    labels = [k for k in order if counts.get(k)]
    values = [counts[k] for k in labels]
    colors = ["#228833", "#cc9944", "#dddddd"][:len(labels)]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Competencies")
    ax.set_title("Occupation mapping status")
    for i, v in enumerate(values):
        ax.text(i, v + 0.1, str(v), ha="center")
    _save(fig, out_path)


def plot_demand_future_scatter(competencies: list[dict[str, Any]], out_path: Path) -> None:
    plt = _ensure_matplotlib()
    rows = _competency_rows(competencies)
    if not rows:
        return

    x = [r["demand"] for r in rows]
    y = [r["future"] for r in rows]
    sizes = [max(40, r["n_jobs"] * 9) for r in rows]
    colors = [r["grounding"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(x, y, s=sizes, c=colors, cmap="viridis", alpha=0.78, edgecolor="#333333", linewidth=0.4)
    ax.set_xlabel("Demand score")
    ax.set_ylabel("Future relevance")
    ax.set_title("Current demand vs future relevance")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.22)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Grounding score")
    top = sorted(rows, key=lambda r: -r["priority"])[:8]
    for r in top:
        ax.annotate(str(r["rank"]), (r["demand"], r["future"]), fontsize=8, ha="center", va="center", color="white")
    _save(fig, out_path)


def plot_kkni_distribution(competencies: list[dict[str, Any]], out_path: Path) -> None:
    plt = _ensure_matplotlib()
    counts = Counter(c.get("kkni_level") for c in competencies if c.get("kkni_level") is not None)
    if not counts:
        return
    xs = sorted(counts)
    ys = [counts[x] for x in xs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(x) for x in xs], ys, color="#66aa55")
    ax.set_xlabel("KKNI level")
    ax.set_ylabel("Final competencies")
    ax.set_title("Competency complexity distribution")
    for i, y in enumerate(ys):
        ax.text(i, y + 0.1, str(y), ha="center")
    _save(fig, out_path)


def _education_totals(competencies: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for c in competencies:
        edu = c.get("education_levels_demanded") or {}
        if not isinstance(edu, dict):
            continue
        for stage, frac in edu.items():
            totals[str(stage)] += _as_float(frac)
    return dict(totals)


def plot_education_distribution(competencies: list[dict[str, Any]], out_path: Path) -> None:
    plt = _ensure_matplotlib()
    totals = _education_totals(competencies)
    if not totals:
        return
    preferred = ["SMK", "SMA", "D1", "D2", "D3", "D4", "S1", "Profesi", "S2", "S3"]
    keys = sorted(totals, key=lambda k: preferred.index(k) if k in preferred else 99)
    values = [totals[k] for k in keys]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(keys, values, color="#aa7744")
    ax.set_ylabel("Summed demand share across competencies")
    ax.set_title("Education-stage demand distribution")
    ax.tick_params(axis="x", rotation=35)
    _save(fig, out_path)


def plot_occupation_coverage(competencies: list[dict[str, Any]], out_path: Path) -> None:
    plt = _ensure_matplotlib()
    occ_counts: Counter = Counter()
    mapped = 0
    for c in competencies:
        matches = c.get("occupation_matches") or []
        if matches:
            mapped += 1
        for m in matches:
            occ = m.get("occupation_en") or m.get("occupation_id") or ""
            if occ:
                occ_counts[str(occ)] += 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.7]})
    axes[0].pie(
        [mapped, max(0, len(competencies) - mapped)],
        labels=["Mapped", "Unmapped"],
        autopct="%1.0f%%",
        colors=["#4477aa", "#dddddd"],
        startangle=90,
    )
    axes[0].set_title("SKKNI occupation coverage")

    top = occ_counts.most_common(10)
    if top:
        labels = [_short(k, 34) for k, _ in top][::-1]
        vals = [v for _, v in top][::-1]
        axes[1].barh(labels, vals, color="#7755aa")
        axes[1].set_xlabel("Competency matches")
        axes[1].set_title("Top mapped occupations")
    else:
        axes[1].text(0.5, 0.5, "No occupation matches", ha="center", va="center")
        axes[1].axis("off")
    _save(fig, out_path)


def _pipeline_funnel_values(report: dict[str, Any]) -> list[tuple[str, float]]:
    stages = report.get("stages", {}) if isinstance(report, dict) else {}
    p1 = stages.get("phase1_load", {})
    p21 = stages.get("phase2_1_clustering", {})
    p22 = stages.get("phase2_2_generation", {})
    p26 = stages.get("phase2_6_evidence_repair", {})
    p4 = stages.get("phase4_priority", {})

    vals = [
        ("Extracted items", p1.get("n_items")),
        ("Canonical items", p21.get("n_items_canonical")),
        ("Frequency-filtered items", p21.get("n_items_after_frequency_filter")),
        ("Clusters", (p21.get("n_clusters_hdbscan", 0) or 0) + (p21.get("n_clusters_recovered", 0) or 0) + (p21.get("n_clusters_split", 0) or 0)),
        ("Generated competencies", p22.get("n_competencies")),
    ]
    if p26:
        vals.append(("Evidence-repaired", p26.get("n_repaired")))
    vals.append(("Final recommendations", report.get("final_n_competencies") or p4.get("n_after_cap")))
    return [(label, float(value)) for label, value in vals if value is not None]


def plot_pipeline_funnel(report: dict[str, Any], out_path: Path) -> None:
    plt = _ensure_matplotlib()
    vals = _pipeline_funnel_values(report)
    if not vals:
        return
    labels = [v[0] for v in vals][::-1]
    values = [v[1] for v in vals][::-1]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(labels, values, color="#4477aa")
    ax.set_xscale("log")
    ax.set_xlabel("Count (log scale)")
    ax.set_title("Pipeline funnel")
    for y, v in enumerate(values):
        ax.text(v * 1.06, y, f"{int(v):,}", va="center")
    _save(fig, out_path)


def plot_grounding_before_after(report: dict[str, Any], competencies: list[dict[str, Any]], out_path: Path) -> None:
    plt = _ensure_matplotlib()
    p26 = (report.get("stages", {}) or {}).get("phase2_6_evidence_repair", {})
    final_scores = [_as_float(c.get("grounding_score")) for c in competencies]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    before = p26.get("grounding_mean_before")
    after = p26.get("grounding_mean_after")
    if before is not None and after is not None:
        axes[0].bar(["Before repair", "After repair"], [float(before), float(after)], color=["#cc6677", "#228833"])
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel("Mean grounding score")
        axes[0].set_title("Evidence repair effect")
    else:
        axes[0].text(0.5, 0.5, "No evidence-repair summary", ha="center", va="center")
        axes[0].axis("off")

    if final_scores:
        axes[1].hist(final_scores, bins=[0, 0.4, 0.6, 0.8, 0.9, 1.01], color="#4477aa", edgecolor="white")
        axes[1].axvline(0.8, color="#cc3311", linestyle="--", linewidth=1.2, label="0.80 gate")
        axes[1].set_xlabel("Final grounding score")
        axes[1].set_ylabel("Competencies")
        axes[1].set_title("Final grounding distribution")
        axes[1].legend()
    else:
        axes[1].axis("off")
    _save(fig, out_path)


def plot_cluster_cohesion(clusters: list[dict[str, Any]], out_path: Path) -> None:
    plt = _ensure_matplotlib()
    scores = [_as_float(c.get("cohesion_score")) for c in clusters if c.get("cohesion_score") is not None]
    if not scores:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=14, color="#66aa55", edgecolor="white")
    ax.axvline(0.55, color="#cc3311", linestyle="--", linewidth=1.2, label="threshold 0.55")
    ax.set_xlabel("Cluster cohesion score")
    ax.set_ylabel("Clusters")
    ax.set_title("Cluster cohesion distribution")
    ax.legend()
    _save(fig, out_path)


def plot_drop_reasons(failed: list[dict[str, Any]], noise_audit: dict[str, Any], out_path: Path) -> None:
    plt = _ensure_matplotlib()
    counts: Counter = Counter()
    for item in failed:
        reason = item.get("dropped_reason")
        if reason:
            counts[str(reason)] += 1
    for stage, n in (noise_audit.get("by_stage", {}) or {}).items():
        counts[f"phase2_1:{stage}"] += int(n)
    if not counts:
        return
    top = counts.most_common(12)
    labels = [_short(k, 36) for k, _ in top][::-1]
    values = [v for _, v in top][::-1]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(labels, values, color="#888888")
    ax.set_xlabel("Dropped / excluded count")
    ax.set_title("Top drop reasons across the pipeline")
    for y, v in enumerate(values):
        ax.text(v * 1.01, y, f"{v:,}", va="center", fontsize=8)
    _save(fig, out_path)


def plot_support_distributions(competencies: list[dict[str, Any]], out_path: Path) -> None:
    plt = _ensure_matplotlib()
    rows = _competency_rows(competencies)
    if not rows:
        return
    jobs = [r["n_jobs"] for r in rows]
    skills = [r["n_related_skills"] if "n_related_skills" in r else 0 for r in rows]
    sentences = [len(c.get("source_sentence_ids") or []) for c in _ranked(competencies)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, vals, title, xlabel, color in [
        (axes[0], jobs, "Source job support", "Unique jobs", "#4477aa"),
        (axes[1], skills, "Related skill support", "Related skills", "#66aa55"),
        (axes[2], sentences, "Sentence support", "Source sentences", "#aa7744"),
    ]:
        ax.hist(vals, bins=min(10, max(3, len(set(vals)))), color=color, edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Competencies")
    _save(fig, out_path)


def plot_cluster_size_vs_cohesion(clusters: list[dict[str, Any]], out_path: Path) -> None:
    plt = _ensure_matplotlib()
    rows = [
        (
            int(c.get("n_items") or 0),
            _as_float(c.get("cohesion_score")),
            int(c.get("n_unique_jobs") or 0),
            str(c.get("method") or "unknown"),
        )
        for c in clusters
        if c.get("cohesion_score") is not None
    ]
    if not rows:
        return
    method_colors = {"hdbscan": "#4477aa", "agglomerative_split": "#cc9944", "agglomerative_recovery": "#66aa55"}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for method in sorted({r[3] for r in rows}):
        subset = [r for r in rows if r[3] == method]
        ax.scatter(
            [r[0] for r in subset],
            [r[1] for r in subset],
            s=[max(30, r[2] * 1.5) for r in subset],
            alpha=0.72,
            color=method_colors.get(method, "#888888"),
            edgecolor="#333333",
            linewidth=0.35,
            label=method,
        )
    ax.axhline(0.55, color="#cc3311", linestyle="--", linewidth=1.0, label="threshold 0.55")
    ax.set_xlabel("Cluster item count")
    ax.set_ylabel("Cohesion score")
    ax.set_title("Cluster size vs cohesion")
    ax.legend()
    ax.grid(True, alpha=0.2)
    _save(fig, out_path)


def plot_cluster_method_distribution(clusters: list[dict[str, Any]], out_path: Path) -> None:
    plt = _ensure_matplotlib()
    counts = Counter(str(c.get("method") or "unknown") for c in clusters)
    if not counts:
        return
    labels, values = zip(*counts.most_common())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, values, color="#4477aa")
    ax.set_ylabel("Clusters")
    ax.set_title("Cluster construction method distribution")
    ax.tick_params(axis="x", rotation=25)
    for i, v in enumerate(values):
        ax.text(i, v + 0.3, str(v), ha="center")
    _save(fig, out_path)


def plot_competency_failure_stages(failed: list[dict[str, Any]], out_path: Path) -> None:
    plt = _ensure_matplotlib()
    counts = Counter(str(f.get("dropped_reason") or "unknown") for f in failed)
    if not counts:
        return
    order = ["anti_hallucination_filter", "evidence_repair_gate", "text_quality", "priority_rank_cap"]
    labels = [k for k in order if counts.get(k)] + [k for k in counts if k not in order]
    values = [counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(labels, values, color="#cc6677")
    ax.set_ylabel("Competencies")
    ax.set_title("Generated competency failure/exclusion stages")
    ax.tick_params(axis="x", rotation=25)
    for i, v in enumerate(values):
        ax.text(i, v + 0.5, str(v), ha="center")
    _save(fig, out_path)


def plot_kkni_source_confidence(competencies: list[dict[str, Any]], out_path: Path) -> None:
    plt = _ensure_matplotlib()
    source_counts = Counter(str(c.get("kkni_level_source") or "missing") for c in competencies)
    sims = [_as_float(c.get("kkni_match_similarity"), math.nan) for c in competencies]
    sims = [v for v in sims if not math.isnan(v) and v > 0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1, 1.2]})
    labels, values = zip(*source_counts.most_common()) if source_counts else ([], [])
    axes[0].bar(labels, values, color="#4477aa")
    axes[0].set_title("KKNI label source")
    axes[0].set_ylabel("Competencies")
    axes[0].tick_params(axis="x", rotation=25)
    if sims:
        axes[1].hist(sims, bins=8, color="#66aa55", edgecolor="white")
        axes[1].axvline(0.45, color="#cc3311", linestyle="--", linewidth=1.0, label="override threshold")
        axes[1].legend()
        axes[1].set_xlabel("KKNI SBERT similarity")
        axes[1].set_ylabel("Competencies")
        axes[1].set_title("SBERT-labelled KKNI confidence")
    else:
        axes[1].text(0.5, 0.5, "No SBERT KKNI similarities", ha="center", va="center")
        axes[1].axis("off")
    _save(fig, out_path)


def plot_demand_concentration(competencies: list[dict[str, Any]], out_path: Path) -> None:
    plt = _ensure_matplotlib()
    rows = _competency_rows(competencies)
    if not rows:
        return
    values = sorted([r["n_jobs"] for r in rows], reverse=True)
    total = sum(values) or 1
    cumulative = []
    running = 0
    for v in values:
        running += v
        cumulative.append(running / total)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = list(range(1, len(values) + 1))
    ax1.bar(x, values, color="#4477aa", alpha=0.78)
    ax1.set_xlabel("Competencies sorted by source-job support")
    ax1.set_ylabel("Unique source jobs")
    ax2 = ax1.twinx()
    ax2.plot(x, cumulative, color="#cc3311", marker="o", linewidth=1.4)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Cumulative share of source-job support")
    ax1.set_title("Demand concentration across final recommendations")
    _save(fig, out_path)


def copy_to_evidence_package(tag: str, source_root: Path) -> Path | None:
    evidence_dir = PROJECT_ROOT / "results" / "evidence_packages" / tag
    if not evidence_dir.exists():
        return None
    target = evidence_dir / "figures"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source_root, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="results/competency_v2_pipeline_n1k_openrouter_gpt54")
    parser.add_argument("--output-root", default="results/figures")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--no-copy-to-evidence-package", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    report = _load_json(run_dir / "pipeline_report.json", {})
    competencies = _load_json(run_dir / "competencies.json", [])
    failed = _load_json(run_dir / "competencies_failed.json", [])
    clusters = _load_json(run_dir / "clusters.json", [])
    noise_audit = _load_json(run_dir / "noise_audit.json", {})

    tag = _run_tag(run_dir, report)
    out_root = Path(args.output_root)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root
    fig_root = out_root / tag
    dashboard_dir = fig_root / "dashboard"
    research_dir = fig_root / "research"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    research_dir.mkdir(parents=True, exist_ok=True)
    _clean_pngs(dashboard_dir, research_dir)

    plot_priority_top(competencies, dashboard_dir / "priority_top30.png", top_n=args.top_n)
    plot_demand_future_scatter(competencies, dashboard_dir / "demand_future_scatter.png")
    plot_kkni_distribution(competencies, dashboard_dir / "kkni_distribution.png")
    plot_education_distribution(competencies, dashboard_dir / "education_distribution.png")
    plot_occupation_coverage(competencies, dashboard_dir / "occupation_coverage.png")
    plot_score_components_top(competencies, dashboard_dir / "score_components_top15.png")
    plot_job_support_top(competencies, dashboard_dir / "job_support_top30.png", top_n=args.top_n)
    plot_recommendation_type_mix(competencies, dashboard_dir / "recommendation_type_mix.png")
    plot_recommendation_status_mix(competencies, dashboard_dir / "recommendation_status_mix.png")
    plot_education_by_kkni_heatmap(competencies, dashboard_dir / "education_by_kkni_heatmap.png")
    plot_occupation_official_status(competencies, dashboard_dir / "occupation_official_status.png")

    plot_pipeline_funnel(report, research_dir / "pipeline_funnel.png")
    plot_grounding_before_after(report, competencies, research_dir / "grounding_before_after.png")
    plot_cluster_cohesion(clusters, research_dir / "cluster_cohesion_hist.png")
    plot_drop_reasons(failed, noise_audit, research_dir / "drop_reasons.png")
    plot_support_distributions(competencies, research_dir / "support_distributions.png")
    plot_cluster_size_vs_cohesion(clusters, research_dir / "cluster_size_vs_cohesion.png")
    plot_cluster_method_distribution(clusters, research_dir / "cluster_method_distribution.png")
    plot_competency_failure_stages(failed, research_dir / "competency_failure_stages.png")
    plot_kkni_source_confidence(competencies, research_dir / "kkni_source_confidence.png")
    plot_demand_concentration(competencies, research_dir / "demand_concentration.png")

    manifest = {
        "run_tag": tag,
        "dashboard_figures": sorted(p.name for p in dashboard_dir.glob("*.png")),
        "research_figures": sorted(p.name for p in research_dir.glob("*.png")),
    }
    (fig_root / "figures_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    copied = None
    if not args.no_copy_to_evidence_package:
        copied = copy_to_evidence_package(tag, fig_root)

    print(f"[OK] Wrote figures to: {fig_root}")
    print(f"     Dashboard figures: {dashboard_dir}")
    print(f"     Research figures:  {research_dir}")
    if copied:
        print(f"     Copied to evidence package: {copied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
