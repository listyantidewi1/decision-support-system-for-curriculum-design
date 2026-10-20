from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from occupation_mapper import _competency_embedding_text  # noqa: E402
from pipeline_scientific_audits import (  # noqa: E402
    CLUSTER_SPLIT,
    THEME_NOT_DETECTED,
    audit_cluster_validity,
    build_demand_theme_audit,
    classify_extraction_span,
    normalize_extraction_text,
)
from scripts.apply_scientific_sprint1 import (  # noqa: E402
    LLM_FUTURE_SOURCE,
    public_future_weight,
)


def test_skill_llm_skill_spans_are_not_automatically_hard_technical() -> None:
    category, flags = classify_extraction_span("communication skills", source_field="skill")
    assert category == "support_soft_skill"
    assert "support_soft_skill" in flags

    category, flags = classify_extraction_span("Bachelor's degree", source_field="skill")
    assert category == "credential"

    category, flags = classify_extraction_span("professional drive", source_field="skill")
    assert category == "personality_trait"


def test_punctuation_variants_normalize_before_clustering_audit() -> None:
    assert normalize_extraction_text("Python)") == "Python"
    assert normalize_extraction_text("Python.") == "Python"
    assert normalize_extraction_text("RESTful APIs.") == "REST APIs"
    assert normalize_extraction_text("Infrastructure as Code (IaC") == "Infrastructure as Code"


def test_specific_technical_spans_stay_hard_or_process() -> None:
    assert classify_extraction_span("Python)", source_field="skill")[0] == "hard_technical"
    assert classify_extraction_span("RESTful APIs.", source_field="skill")[0] == "hard_technical"
    assert classify_extraction_span("automated testing", source_field="skill")[0] == "technical_process"


def test_mixed_cluster_is_flagged_split_required(tmp_path: Path) -> None:
    clusters = [
        {
            "id": "cluster_h0045",
            "summary_label": "Power Apps Salesforce IoT user research",
            "top_terms": ["power apps", "salesforce", "iot", "user research", "aws"],
            "items": [
                "Power Apps",
                "Salesforce CRM",
                "IoT sensors",
                "user research",
                "AWS Lambda",
                "SharePoint workflow",
            ] * 7,
            "n_items": 42,
            "n_unique_jobs": 18,
            "cohesion_score": 0.58,
        }
    ]
    report = audit_cluster_validity(clusters, tmp_path)
    status = report["cluster_validity_by_id"]["cluster_h0045"]["cluster_validity_status"]
    assert status == CLUSTER_SPLIT
    assert (tmp_path / "cluster_split_candidates.csv").exists()


def test_occupation_embedding_uses_v2_statement_before_legacy_fields() -> None:
    text = _competency_embedding_text({
        "statement": "Design REST APIs for backend service integration.",
        "title": "",
        "description": "",
        "related_skills": ["REST API", "backend"],
    })
    assert "Design REST APIs" in text
    assert "Related skills" in text


def test_llm_future_weight_is_not_public_rank_evidence_by_default() -> None:
    comp = {
        "future_weight": 0.95,
        "future_weight_source": LLM_FUTURE_SOURCE,
    }
    assert public_future_weight(comp) == 0.0

    comp["future_weight_source"] = "deterministic_domain_mapping"
    assert public_future_weight(comp) == 0.95


def test_absent_theme_is_not_recommended_from_this_corpus(tmp_path: Path) -> None:
    report = build_demand_theme_audit(
        output_dir=tmp_path,
        competencies=[],
        failed_competencies=[],
        clusters=[],
        denominator=100,
    )
    python_row = next(row for row in report["themes"] if row["theme_id"] == "python")
    assert python_row["theme_evidence_status"] == THEME_NOT_DETECTED
    assert python_row["theme_recommendation_status"] == "not_recommended_from_this_corpus"
    assert (tmp_path / "not_recommended_themes.csv").exists()

