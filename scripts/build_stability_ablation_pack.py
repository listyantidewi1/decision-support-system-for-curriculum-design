"""
Build a stability and ablation package for a completed competency-v2 run.

Sprint 6 asks whether recommendations are robust enough for stakeholder trust.
This script evaluates that from already-generated artifacts:

* ranking ablations inside one run;
* top-K overlap across comparable completed runs;
* model/provider comparison summaries when older live model runs exist;
* runtime/token/cost-adjacent reporting.

It does not run new LLM calls. Controlled seed/model experiments can still be
run separately; this pack records whether such evidence exists and whether it
passes the stability threshold.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOKEN_RE = re.compile(r"[a-zA-Z0-9+#./-]+")


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        out = float(value)
        if math.isnan(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[float]) -> float | None:
    vals = list(values)
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def _compact(text: Any, max_chars: int = 180) -> str:
    raw = " ".join(str(text or "").split())
    if len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 3].rstrip() + "..."


def _run_tag(run_dir: Path, pipeline_report: dict[str, Any]) -> str:
    if pipeline_report.get("tag"):
        return str(pipeline_report["tag"])
    prefix = "competency_v2_pipeline_"
    return run_dir.name[len(prefix):] if run_dir.name.startswith(prefix) else run_dir.name


def competency_text(comp: dict[str, Any]) -> str:
    statement = str(comp.get("statement") or "").strip()
    if statement:
        return statement
    return " ".join(
        str(comp.get(key) or "").strip()
        for key in ["title", "description", "rationale"]
        if str(comp.get(key) or "").strip()
    )


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def token_set(text: Any) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(str(text or "").lower()) if len(t) > 1}


def token_jaccard(a: Any, b: Any) -> float:
    sa, sb = token_set(a), token_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def fuzzy_similarity(a: Any, b: Any) -> float:
    # Token Jaccard is intentionally used instead of edit distance here because
    # the matrices compare many long competency statements. It is fast,
    # deterministic, and good enough to detect topical overlap for ablation.
    return round(token_jaccard(a, b), 4)


def set_jaccard(a: Iterable[Any], b: Iterable[Any]) -> float:
    sa = {x for x in a if x not in (None, "")}
    sb = {x for x in b if x not in (None, "")}
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return round(len(sa & sb) / len(sa | sb), 4)


def symmetric_best_similarity(a_texts: list[str], b_texts: list[str]) -> float:
    if not a_texts or not b_texts:
        return 0.0
    a_to_b = [max(fuzzy_similarity(a, b) for b in b_texts) for a in a_texts]
    b_to_a = [max(fuzzy_similarity(b, a) for a in a_texts) for b in b_texts]
    return round((sum(a_to_b) + sum(b_to_a)) / (len(a_to_b) + len(b_to_a)), 4)


def spearman_on_union(base_ids: list[str], alt_ids: list[str], k: int) -> float:
    universe = list(dict.fromkeys(base_ids[:k] + alt_ids[:k]))
    if len(universe) < 2:
        return 1.0
    missing_rank = k + 1
    base_rank = {cid: i + 1 for i, cid in enumerate(base_ids[:k])}
    alt_rank = {cid: i + 1 for i, cid in enumerate(alt_ids[:k])}
    d2 = 0.0
    for cid in universe:
        d = base_rank.get(cid, missing_rank) - alt_rank.get(cid, missing_rank)
        d2 += d * d
    n = len(universe)
    return round(1 - (6 * d2) / (n * (n * n - 1)), 4)


def load_competencies(run_dir: Path) -> list[dict[str, Any]]:
    for name in ["competencies_sprint1_enriched.json", "competencies.json"]:
        path = run_dir / name
        data = _load_json(path, [])
        if isinstance(data, list) and data:
            return data
    return []


def rank_competencies(competencies: list[dict[str, Any]], strategy: str) -> list[dict[str, Any]]:
    def n_jobs(c: dict[str, Any]) -> int:
        return int(c.get("n_source_jobs") or len(_as_list(c.get("source_job_ids"))))

    def demand(c: dict[str, Any]) -> float:
        return _as_float(c.get("demand_score"), _as_float(c.get("job_prevalence")))

    def grounding(c: dict[str, Any]) -> float:
        return _as_float(c.get("grounding_score"))

    def future(c: dict[str, Any]) -> float:
        return _as_float(c.get("future_weight"))

    if strategy == "contract_class_aware":
        return sorted(
            competencies,
            key=lambda c: (
                int(c.get("class_sort_group") if c.get("class_sort_group") is not None else 99),
                int(c.get("class_rank") or 999999),
                -_as_float(c.get("job_prevalence")),
                -n_jobs(c),
                -grounding(c),
                -future(c),
                str(c.get("id", "")),
            ),
        )
    formulas = {
        "legacy_priority": lambda c: _as_float(c.get("priority_score")),
        "demand_only": lambda c: demand(c),
        "grounding_only": lambda c: grounding(c),
        "future_only": lambda c: future(c),
        "equal_demand_grounding_future": lambda c: (demand(c) + grounding(c) + future(c)) / 3,
        "no_future_demand_grounding": lambda c: 0.55 * demand(c) + 0.45 * grounding(c),
        "no_grounding_demand_future": lambda c: 0.60 * demand(c) + 0.40 * future(c),
        "demand_first_tuple": lambda c: demand(c),
    }
    if strategy not in formulas:
        raise ValueError(f"Unknown ranking strategy: {strategy}")
    if strategy == "demand_first_tuple":
        return sorted(competencies, key=lambda c: (-demand(c), -n_jobs(c), -grounding(c), -future(c), str(c.get("id", ""))))
    return sorted(competencies, key=lambda c: (-formulas[strategy](c), -n_jobs(c), str(c.get("id", ""))))


def build_ranking_ablation(competencies: list[dict[str, Any]], top_k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    strategies = [
        "contract_class_aware",
        "legacy_priority",
        "demand_first_tuple",
        "demand_only",
        "grounding_only",
        "future_only",
        "equal_demand_grounding_future",
        "no_future_demand_grounding",
        "no_grounding_demand_future",
    ]
    ranked_by_strategy = {name: rank_competencies(competencies, name) for name in strategies}
    baseline_ids = [str(c.get("id", "")) for c in ranked_by_strategy["contract_class_aware"][:top_k]]
    baseline_texts = [competency_text(c) for c in ranked_by_strategy["contract_class_aware"][:top_k]]

    rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    for strategy, ranked in ranked_by_strategy.items():
        ids = [str(c.get("id", "")) for c in ranked[:top_k]]
        texts = [competency_text(c) for c in ranked[:top_k]]
        overlap_ids = set(baseline_ids) & set(ids)
        rows.append({
            "strategy": strategy,
            "top_k": top_k,
            "topk_id_jaccard_vs_contract": set_jaccard(baseline_ids, ids),
            "topk_text_similarity_vs_contract": symmetric_best_similarity(baseline_texts, texts),
            "spearman_union_vs_contract": spearman_on_union(baseline_ids, ids, top_k),
            "n_overlap_with_contract": len(overlap_ids),
            "top_ids": " | ".join(ids),
        })
        for rank, comp in enumerate(ranked[:top_k], start=1):
            member_rows.append({
                "strategy": strategy,
                "rank": rank,
                "competency_id": comp.get("id", ""),
                "competency_class": comp.get("competency_class", ""),
                "statement": _compact(competency_text(comp), 260),
                "job_prevalence_pct": comp.get("job_prevalence_pct", ""),
                "demand_score": comp.get("demand_score", ""),
                "grounding_score": comp.get("grounding_score", ""),
                "future_weight": comp.get("future_weight", ""),
                "priority_score": comp.get("priority_score", ""),
            })
    return rows, member_rows


def run_stats(run_dir: Path) -> dict[str, Any] | None:
    comps = load_competencies(run_dir)
    if not comps:
        return None
    report = _load_json(run_dir / "pipeline_report.json", {})
    brs = _load_json(run_dir / "batch_reasonings.json", [])
    if not isinstance(brs, list):
        brs = []
    stages = report.get("stages", {}) if isinstance(report, dict) else {}
    sprint = stages.get("scientific_sprint1", {}) if isinstance(stages, dict) else {}
    phase2_2 = stages.get("phase2_2_generation", {}) if isinstance(stages, dict) else {}
    phase2_6 = stages.get("phase2_6_evidence_repair", {}) if isinstance(stages, dict) else {}
    prompt_tokens = sum(int(br.get("prompt_tokens") or 0) for br in brs if isinstance(br, dict))
    completion_tokens = sum(int(br.get("completion_tokens") or 0) for br in brs if isinstance(br, dict))
    latency = sum(_as_float(br.get("latency_seconds")) for br in brs if isinstance(br, dict))
    model = report.get("model") or (brs[0].get("model") if brs and isinstance(brs[0], dict) else "")
    provider = brs[0].get("provider") if brs and isinstance(brs[0], dict) else ""
    ranked = rank_competencies(comps, "contract_class_aware")
    top_texts = [competency_text(c) for c in ranked]
    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "run_tag": _run_tag(run_dir, report if isinstance(report, dict) else {}),
        "model": model,
        "provider": provider,
        "phase1_input": str(report.get("phase1_input", "")),
        "jobs_metadata": str(report.get("jobs_metadata", "")),
        "generation_contract": str(phase2_2.get("contract", "")),
        "evidence_repair_version": str(phase2_6.get("version", "")),
        "scientific_contract_schema": str(sprint.get("schema_version", "")),
        "total_runtime_seconds": _as_float(report.get("total_runtime_seconds")),
        "llm_latency_seconds": round(latency, 4),
        "n_llm_calls": len(brs),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "n_competencies": len(comps),
        "mean_grounding": _mean(_as_float(c.get("grounding_score")) for c in comps),
        "mean_job_prevalence_pct": _mean(_as_float(c.get("job_prevalence_pct")) for c in comps),
        "recommendation_status_distribution": dict(Counter(str(c.get("recommendation_status") or "unknown") for c in comps)),
        "competency_class_distribution": dict(Counter(str(c.get("competency_class") or "unknown") for c in comps)),
        "corpus_validity_status": sprint.get("corpus_validity_status", ""),
        "top_ids": [str(c.get("id", "")) for c in ranked],
        "top_texts": top_texts,
    }


def discover_run_dirs(patterns: list[str], base_run: Path) -> list[Path]:
    found: dict[str, Path] = {str(base_run.resolve()).lower(): base_run}
    for pattern in patterns:
        glob_path = pattern
        root = PROJECT_ROOT
        matches = sorted(root.glob(glob_path)) if not Path(pattern).is_absolute() else sorted(Path().glob(pattern))
        for path in matches:
            if path.is_dir() and (path / "competencies.json").exists():
                found[str(path.resolve()).lower()] = path
    return list(found.values())


def build_run_comparison(run_dirs: list[Path], base_run: Path, top_k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stats = [s for d in run_dirs if (s := run_stats(d)) is not None]
    if not stats:
        return [], []
    base_stat = next((s for s in stats if Path(s["run_dir"]).resolve() == base_run.resolve()), stats[0])
    for s in stats:
        s["same_phase1_as_base"] = bool(s.get("phase1_input") and s.get("phase1_input") == base_stat.get("phase1_input"))
        s["same_jobs_metadata_as_base"] = bool(s.get("jobs_metadata") and s.get("jobs_metadata") == base_stat.get("jobs_metadata"))
        s["same_model_as_base"] = bool(s.get("model") and s.get("model") == base_stat.get("model"))
        s["same_generation_contract_as_base"] = bool(
            s.get("generation_contract") and s.get("generation_contract") == base_stat.get("generation_contract")
        )
        s["same_evidence_repair_version_as_base"] = bool(
            s.get("evidence_repair_version") and s.get("evidence_repair_version") == base_stat.get("evidence_repair_version")
        )
        s["same_scientific_contract_as_base"] = bool(
            s.get("scientific_contract_schema") and s.get("scientific_contract_schema") == base_stat.get("scientific_contract_schema")
        )
        s["topk_id_jaccard_vs_base"] = set_jaccard(base_stat["top_ids"][:top_k], s["top_ids"][:top_k])
        s["topk_text_similarity_vs_base"] = symmetric_best_similarity(base_stat["top_texts"][:top_k], s["top_texts"][:top_k])
    matrix_rows: list[dict[str, Any]] = []
    for i, a in enumerate(stats):
        for b in stats[i + 1:]:
            matrix_rows.append({
                "run_a": a["run_name"],
                "run_b": b["run_name"],
                "same_phase1": bool(a.get("phase1_input") and a.get("phase1_input") == b.get("phase1_input")),
                "same_jobs_metadata": bool(a.get("jobs_metadata") and a.get("jobs_metadata") == b.get("jobs_metadata")),
                "same_model": bool(a.get("model") and a.get("model") == b.get("model")),
                "same_generation_contract": bool(
                    a.get("generation_contract") and a.get("generation_contract") == b.get("generation_contract")
                ),
                "same_evidence_repair_version": bool(
                    a.get("evidence_repair_version") and a.get("evidence_repair_version") == b.get("evidence_repair_version")
                ),
                "same_scientific_contract": bool(
                    a.get("scientific_contract_schema") and a.get("scientific_contract_schema") == b.get("scientific_contract_schema")
                ),
                "top_k": top_k,
                "topk_id_jaccard": set_jaccard(a["top_ids"][:top_k], b["top_ids"][:top_k]),
                "topk_text_similarity": symmetric_best_similarity(a["top_texts"][:top_k], b["top_texts"][:top_k]),
            })
    return stats, matrix_rows


def build_model_provider_comparison(patterns: list[str], top_k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dirs: list[Path] = []
    for pattern in patterns:
        dirs.extend(path for path in sorted(PROJECT_ROOT.glob(pattern)) if path.is_dir() and (path / "competencies.json").exists())
    unique = list({str(d.resolve()).lower(): d for d in dirs}.values())
    stats = [s for d in unique if (s := run_stats(d)) is not None]
    matrix_rows: list[dict[str, Any]] = []
    for i, a in enumerate(stats):
        for b in stats[i + 1:]:
            matrix_rows.append({
                "run_a": a["run_name"],
                "run_b": b["run_name"],
                "model_a": a["model"],
                "model_b": b["model"],
                "provider_a": a["provider"],
                "provider_b": b["provider"],
                "top_k": top_k,
                "topk_id_jaccard": set_jaccard(a["top_ids"][:top_k], b["top_ids"][:top_k]),
                "topk_text_similarity": symmetric_best_similarity(a["top_texts"][:top_k], b["top_texts"][:top_k]),
            })
    return stats, matrix_rows


def status_from_comparison(
    run_stats_rows: list[dict[str, Any]],
    matrix_rows: list[dict[str, Any]],
    min_runs: int,
    threshold: float,
) -> dict[str, Any]:
    controlled = [
        r for r in matrix_rows
        if r.get("same_phase1")
        and r.get("same_jobs_metadata")
        and r.get("same_model")
        and r.get("same_generation_contract")
        and r.get("same_evidence_repair_version")
        and r.get("same_scientific_contract")
    ]
    if len(run_stats_rows) < min_runs or not controlled:
        return {
            "status": "insufficient_controlled_runs",
            "n_runs": len(run_stats_rows),
            "n_controlled_pairs": len(controlled),
            "mean_topk_text_similarity": None,
            "threshold": threshold,
        }
    vals = [_as_float(r.get("topk_text_similarity")) for r in controlled]
    mean_val = _mean(vals)
    return {
        "status": "pass" if mean_val is not None and mean_val >= threshold else "fail",
        "n_runs": len(run_stats_rows),
        "n_controlled_pairs": len(controlled),
        "mean_topk_text_similarity": mean_val,
        "threshold": threshold,
    }


def ranking_robustness_status(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    ablations = [r for r in rows if r["strategy"] != "contract_class_aware"]
    vals = [_as_float(r.get("topk_id_jaccard_vs_contract")) for r in ablations]
    min_val = min(vals) if vals else None
    mean_val = _mean(vals)
    return {
        "status": "pass" if min_val is not None and min_val >= threshold else "fail",
        "min_topk_id_jaccard_vs_contract": round(min_val, 4) if min_val is not None else None,
        "mean_topk_id_jaccard_vs_contract": mean_val,
        "threshold": threshold,
        "n_ablations": len(ablations),
    }


def write_readme(path: Path, report: dict[str, Any]) -> None:
    text = f"""# Stability And Ablation Pack: {report.get('run_tag')}

Generated: {report.get('created_at')}

Source run:

```text
{report.get('base_run_dir')}
```

## Purpose

This package tests whether the recommendation ranking is robust to scoring
choices and whether completed runs show top-K stability.

## Files

| File | Purpose |
|---|---|
| `stability_ablation_report.json` | Machine-readable summary and gate status |
| `ranking_ablation.csv` | Top-K overlap between the contract ranking and alternative scoring rules |
| `ranking_ablation_topk_members.csv` | Top-K members under each ranking strategy |
| `run_comparison.csv` | Per-run metrics, runtime, token counts, corpus compatibility, and overlap with base run |
| `topk_overlap_matrix.csv` | Pairwise run overlap and fuzzy text similarity |
| `model_provider_comparison.csv` | Optional older live-model comparison summary |
| `model_provider_overlap_matrix.csv` | Pairwise model/provider top-K overlap |

## Interpretation

`official_stability_gate_status` is strict. It passes only when ranking
ablation is robust and enough controlled comparable runs exist. Controlled
runs must share Phase-1 input, jobs metadata, model, generation contract,
evidence-repair version, and recommendation contract. If the report says
`insufficient_controlled_runs`, the right next step is to run controlled
seed/provider experiments, not to relax the gate.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="results/competency_v2_pipeline_n1k_openrouter_gpt54")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--comparison-glob", action="append", default=["results/competency_v2_pipeline_n1k*"])
    parser.add_argument("--model-glob", action="append", default=["results/competency_v2_live*"])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--stability-threshold", type=float, default=0.80)
    parser.add_argument("--ranking-ablation-threshold", type=float, default=0.70)
    parser.add_argument("--min-controlled-runs", type=int, default=3)
    args = parser.parse_args()

    run_dir = _resolve(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    pipeline_report = _load_json(run_dir / "pipeline_report.json", {})
    tag = _run_tag(run_dir, pipeline_report if isinstance(pipeline_report, dict) else {})
    out_dir = _resolve(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "stability_ablation" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    competencies = load_competencies(run_dir)
    if not competencies:
        raise ValueError(f"No competencies found in {run_dir}")

    ranking_rows, member_rows = build_ranking_ablation(competencies, args.top_k)
    _write_csv(out_dir / "ranking_ablation.csv", ranking_rows, [
        "strategy", "top_k", "topk_id_jaccard_vs_contract",
        "topk_text_similarity_vs_contract", "spearman_union_vs_contract",
        "n_overlap_with_contract", "top_ids",
    ])
    _write_csv(out_dir / "ranking_ablation_topk_members.csv", member_rows, [
        "strategy", "rank", "competency_id", "competency_class", "statement",
        "job_prevalence_pct", "demand_score", "grounding_score",
        "future_weight", "priority_score",
    ])

    comparison_dirs = discover_run_dirs(args.comparison_glob, run_dir)
    run_rows, matrix_rows = build_run_comparison(comparison_dirs, run_dir, args.top_k)
    run_fieldnames = [
        "run_name", "run_tag", "model", "provider", "phase1_input",
        "jobs_metadata", "same_phase1_as_base", "same_jobs_metadata_as_base",
        "same_model_as_base", "same_generation_contract_as_base",
        "same_evidence_repair_version_as_base", "same_scientific_contract_as_base",
        "generation_contract", "evidence_repair_version", "scientific_contract_schema",
        "corpus_validity_status", "n_competencies", "mean_grounding",
        "mean_job_prevalence_pct", "total_runtime_seconds", "llm_latency_seconds",
        "n_llm_calls", "prompt_tokens", "completion_tokens", "total_tokens",
        "topk_id_jaccard_vs_base", "topk_text_similarity_vs_base",
        "run_dir",
    ]
    _write_csv(out_dir / "run_comparison.csv", run_rows, run_fieldnames)
    _write_csv(out_dir / "topk_overlap_matrix.csv", matrix_rows, [
        "run_a", "run_b", "same_phase1", "same_jobs_metadata",
        "same_model", "same_generation_contract", "same_evidence_repair_version",
        "same_scientific_contract",
        "top_k", "topk_id_jaccard", "topk_text_similarity",
    ])

    model_rows, model_matrix_rows = build_model_provider_comparison(args.model_glob, args.top_k)
    _write_csv(out_dir / "model_provider_comparison.csv", model_rows, run_fieldnames)
    _write_csv(out_dir / "model_provider_overlap_matrix.csv", model_matrix_rows, [
        "run_a", "run_b", "model_a", "model_b", "provider_a", "provider_b",
        "top_k", "topk_id_jaccard", "topk_text_similarity",
    ])

    ranking_status = ranking_robustness_status(ranking_rows, args.ranking_ablation_threshold)
    completed_run_status = status_from_comparison(
        run_rows,
        matrix_rows,
        min_runs=args.min_controlled_runs,
        threshold=args.stability_threshold,
    )
    official_status = (
        "pass"
        if ranking_status["status"] == "pass" and completed_run_status["status"] == "pass"
        else completed_run_status["status"]
        if completed_run_status["status"] != "pass"
        else "fail"
    )
    report = {
        "schema_version": "stability_ablation_pack_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_tag": tag,
        "base_run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "top_k": args.top_k,
        "official_stability_gate_status": official_status,
        "ranking_ablation_status": ranking_status,
        "completed_run_stability_status": completed_run_status,
        "n_discovered_comparison_runs": len(run_rows),
        "n_model_provider_runs": len(model_rows),
        "files": {
            "ranking_ablation": str(out_dir / "ranking_ablation.csv"),
            "ranking_ablation_topk_members": str(out_dir / "ranking_ablation_topk_members.csv"),
            "run_comparison": str(out_dir / "run_comparison.csv"),
            "topk_overlap_matrix": str(out_dir / "topk_overlap_matrix.csv"),
            "model_provider_comparison": str(out_dir / "model_provider_comparison.csv"),
            "model_provider_overlap_matrix": str(out_dir / "model_provider_overlap_matrix.csv"),
        },
        "interpretation": [
            "Ranking ablation tests sensitivity to scoring choices within one run.",
            "Completed-run stability only counts as official evidence when comparable runs share Phase-1 input, jobs metadata, model, generation contract, evidence-repair version, and recommendation contract.",
            "Model/provider comparison from older live runs is diagnostic, not official stability evidence unless run conditions are controlled.",
        ],
    }
    _write_json(out_dir / "stability_ablation_report.json", report)
    write_readme(out_dir / "README.md", report)

    print(f"[OK] Stability/ablation pack written to: {out_dir}")
    print(f"     ranking ablation: {ranking_status['status']} (min Jaccard={ranking_status['min_topk_id_jaccard_vs_contract']})")
    print(f"     completed-run stability: {completed_run_status['status']} (pairs={completed_run_status['n_controlled_pairs']})")
    print(f"     official stability gate: {official_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
