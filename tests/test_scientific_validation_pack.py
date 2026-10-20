from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_scientific_validation_pack import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    _binary_metrics,
    _write_csv_template,
    build_sentence_relevance_template,
    compute_competency_review_metrics,
    compute_extraction_metrics,
)


def _low_thresholds() -> dict[str, float]:
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update({
        "min_reviewed_competencies": 2,
        "min_extraction_gold_rows": 2,
    })
    return thresholds


def test_binary_metrics_counts_precision_recall_and_accuracy() -> None:
    rows = [
        {"pred": "yes", "gold": "yes"},
        {"pred": "yes", "gold": "no"},
        {"pred": "no", "gold": "yes"},
        {"pred": "no", "gold": "no"},
        {"pred": "", "gold": "yes"},
    ]

    metrics = _binary_metrics(rows, "pred", "gold")

    assert metrics["n_reviewed"] == 4
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fn"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["accuracy"] == 0.5


def test_review_metrics_pass_when_thresholds_are_met() -> None:
    rows = [
        {
            "competency_id": "c1",
            "validity_1_5": "5",
            "relevance_yes_partial_no": "yes",
            "grounding_clarity_1_5": "4",
            "rationale_clarity_1_5": "4",
            "actionability_1_5": "5",
            "hallucination_concern_yes_no": "no",
        },
        {
            "competency_id": "c2",
            "validity_1_5": "4",
            "relevance_yes_partial_no": "partial",
            "grounding_clarity_1_5": "4",
            "rationale_clarity_1_5": "4",
            "actionability_1_5": "4",
            "hallucination_concern_yes_no": "no",
        },
    ]

    metrics = compute_competency_review_metrics(rows, _low_thresholds())

    assert metrics["n_reviewed"] == 2
    assert metrics["validity_mean"] == 4.5
    assert metrics["relevance_yes_or_partial_rate"] == 1.0
    assert metrics["hallucination_concern_rate"] == 0.0
    assert metrics["status"] == "pass"


def test_extraction_metrics_compute_precision_type_accuracy_and_recall() -> None:
    rows = [
        {
            "gold_valid_extraction_yes_no": "yes",
            "gold_correct_type_yes_no": "yes",
            "gold_missing_spans_in_sentence": "1",
        },
        {
            "gold_valid_extraction_yes_no": "yes",
            "gold_correct_type_yes_no": "yes",
            "gold_missing_spans_in_sentence": "0",
        },
    ]

    metrics = compute_extraction_metrics(rows, _low_thresholds())

    assert metrics["n_reviewed"] == 2
    assert metrics["precision"] == 1.0
    assert metrics["type_accuracy"] == 1.0
    assert metrics["recall_with_missing_span_counts"] == 0.6667
    assert metrics["status"] == "pass"


def test_sentence_template_includes_evidence_and_contrast_rows(tmp_path: Path) -> None:
    phase1 = tmp_path / "phase1"
    phase1.mkdir()
    (phase1 / "job_sentences_mapping.csv").write_text(
        "sentence_id,job_id,sentence_text\n"
        "j1_0001,j1,Build APIs with Python.\n"
        "j2_0001,j2,Manage warehouse inventory.\n",
        encoding="utf-8",
    )
    competencies = [{
        "id": "comp_api",
        "statement": "Build reliable APIs.",
        "source_sentences": ["Build APIs with Python."],
        "source_sentence_ids": ["j1_0001"],
        "source_job_ids": ["j1"],
        "sentence_relevance_score": 0.95,
    }]

    rows = build_sentence_relevance_template(
        competencies,
        phase1_dir=phase1,
        per_competency_evidence=1,
        per_competency_contrast=1,
    )

    assert [r["sample_type"] for r in rows] == ["system_evidence", "contrast_sentence"]
    assert rows[0]["job_id"] == "j1"
    assert rows[0]["system_predicted_relevant_yes_no"] == "yes"
    assert rows[1]["job_id"] == "j2"
    assert rows[1]["system_predicted_relevant_yes_no"] == "no"


def test_template_writer_preserves_existing_labels_by_default(tmp_path: Path) -> None:
    path = tmp_path / "template.csv"
    fields = ["sample_id", "gold_label"]

    assert _write_csv_template(path, [{"sample_id": "s1", "gold_label": ""}], fields, overwrite=True) == "written"
    path.write_text("sample_id,gold_label\ns1,yes\n", encoding="utf-8")

    status = _write_csv_template(path, [{"sample_id": "s1", "gold_label": ""}], fields, overwrite=False)

    assert status == "preserved"
    assert "s1,yes" in path.read_text(encoding="utf-8")
