from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_scientific_validation_pack import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    build_official_readiness_report,
)
from scripts.build_stability_ablation_pack import (  # noqa: E402
    build_ranking_ablation,
    ranking_robustness_status,
    set_jaccard,
    status_from_comparison,
    symmetric_best_similarity,
)


def _comp(idx: int, *, cls_group: int = 0, demand: float = 0.5, grounding: float = 0.8, future: float = 0.7) -> dict:
    return {
        "id": f"c{idx}",
        "statement": f"Build service capability number {idx}.",
        "class_sort_group": cls_group,
        "class_rank": idx,
        "job_prevalence": demand,
        "demand_score": demand,
        "n_source_jobs": int(demand * 100),
        "grounding_score": grounding,
        "future_weight": future,
        "priority_score": 0.4 * demand + 0.3 * grounding + 0.3 * future,
    }


def test_set_and_text_similarity_are_deterministic() -> None:
    assert set_jaccard(["a", "b"], ["b", "c"]) == 0.3333
    assert symmetric_best_similarity(
        ["Design cloud deployment pipeline"],
        ["Configure cloud deployment pipeline"],
    ) > 0.3


def test_ranking_ablation_reports_overlap_against_contract() -> None:
    comps = [
        _comp(1, demand=0.9, grounding=0.9, future=0.4),
        _comp(2, demand=0.8, grounding=0.8, future=0.9),
        _comp(3, cls_group=2, demand=1.0, grounding=1.0, future=1.0),
    ]

    rows, members = build_ranking_ablation(comps, top_k=2)
    by_strategy = {r["strategy"]: r for r in rows}

    assert by_strategy["contract_class_aware"]["topk_id_jaccard_vs_contract"] == 1.0
    assert by_strategy["future_only"]["topk_id_jaccard_vs_contract"] < 1.0
    assert len([m for m in members if m["strategy"] == "contract_class_aware"]) == 2


def test_completed_run_stability_requires_strict_control_fields() -> None:
    run_rows = [{"run_name": "a"}, {"run_name": "b"}, {"run_name": "c"}]
    loose_matrix = [{
        "same_phase1": True,
        "same_jobs_metadata": True,
        "same_model": False,
        "same_generation_contract": True,
        "same_evidence_repair_version": True,
        "same_scientific_contract": True,
        "topk_text_similarity": 0.95,
    }]
    strict_matrix = [{
        **loose_matrix[0],
        "same_model": True,
    }]

    loose = status_from_comparison(run_rows, loose_matrix, min_runs=3, threshold=0.8)
    strict = status_from_comparison(run_rows, strict_matrix, min_runs=3, threshold=0.8)

    assert loose["status"] == "insufficient_controlled_runs"
    assert strict["status"] == "pass"


def test_official_readiness_uses_stability_ablation_status() -> None:
    metrics = {
        "competency_review": {"status": "pass"},
        "recommendation_review": {"status": "pass"},
        "sentence_relevance_gold": {"status": "pass"},
        "extraction_quality_gold": {"status": "pass"},
        "dacum_cbc_review": {"status": "pass"},
    }
    pipeline_report = {"stages": {"corpus_lineage_audit": {"corpus_validity_status": "valid_for_indonesian_claim"}}}
    readiness = build_official_readiness_report(
        pipeline_report,
        metrics,
        DEFAULT_THRESHOLDS,
        {"official_stability_gate_status": "insufficient_controlled_runs"},
    )

    assert readiness["gate_statuses"]["stability_validation"] == "insufficient_controlled_runs"
    assert readiness["official_ready"] is False
    assert "stability_validation" in readiness["blockers"]


def test_ranking_robustness_status_fails_low_overlap() -> None:
    rows = [
        {"strategy": "contract_class_aware", "topk_id_jaccard_vs_contract": 1.0},
        {"strategy": "future_only", "topk_id_jaccard_vs_contract": 0.4},
    ]

    status = ranking_robustness_status(rows, threshold=0.7)

    assert status["status"] == "fail"
    assert status["min_topk_id_jaccard_vs_contract"] == 0.4
