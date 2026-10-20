from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from extraction_typing import (  # noqa: E402
    apply_extraction_type_gate_to_items,
    phase1_decision_for_span,
)
from pipeline import KnowledgeItem, SkillItem, SkillType  # noqa: E402
from preprocess_jobs_pipeline import apply_min_sentence_coverage  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_phase1_decisions_demote_support_and_exclude_unanchored_knowledge() -> None:
    support = phase1_decision_for_span("communication skills", source_field="skill")
    assert support.phase1_target == "advanced_skills"
    assert support.skill_type == "Soft"
    assert support.hard_generation_eligible is False

    python = phase1_decision_for_span("Python)", source_field="skill")
    assert python.phase1_target == "advanced_skills"
    assert python.skill_type == "Hard"
    assert python.normalized_span == "Python"

    knowledge = phase1_decision_for_span("business operations", source_field="knowledge")
    assert knowledge.post_extraction_type == "knowledge_domain"
    assert knowledge.phase1_target == "excluded"


def test_converter_writes_typed_phase1_rows_and_exclusion_audits(tmp_path: Path) -> None:
    input_path = tmp_path / "skill_llm.jsonl"
    output_dir = tmp_path / "phase1"
    rows = [
        {
            "job_id": "j1",
            "sentence_id": "j1_0001",
            "sentence_text": "Use Python, cloud infrastructure, and communication skills.",
            "SKILL": [
                {"skill_span": "Python)"},
                {"skill_span": "communication skills"},
                {"skill_span": "Bachelor's degree"},
            ],
            "KNOWLEDGE": [
                {"skill_span": "cloud infrastructure"},
                {"skill_span": "business operations"},
            ],
        }
    ]
    input_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "convert_skill_llm_jsonl_to_phase1_csv.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    skills = _read_csv(output_dir / "advanced_skills.csv")
    knowledge = _read_csv(output_dir / "advanced_knowledge.csv")
    excluded = _read_csv(output_dir / "excluded_extractions.csv")
    decisions = _read_csv(output_dir / "extraction_typing_decisions.csv")

    assert [(r["skill"], r["type"], r["post_extraction_type"]) for r in skills] == [
        ("Python", "Hard", "hard_technical"),
        ("communication skills", "Soft", "support_soft_skill"),
    ]
    assert [(r["knowledge"], r["post_extraction_type"]) for r in knowledge] == [
        ("cloud infrastructure", "hard_technical")
    ]
    assert {r["post_extraction_type"] for r in excluded} == {"credential", "knowledge_domain"}
    assert len(decisions) == 5
    assert (output_dir / "extraction_typing_audit.json").exists()


def test_runtime_gate_protects_old_phase1_items_without_typing_columns() -> None:
    items = [
        SkillItem("Python", SkillType.HARD, 0.9, "Very High", "LLM", sentence_id="j1_0001"),
        SkillItem("communication skills", SkillType.HARD, 0.9, "Very High", "LLM", sentence_id="j1_0002"),
        SkillItem("Bachelor's degree", SkillType.HARD, 0.9, "Very High", "LLM", sentence_id="j2_0001"),
        KnowledgeItem("business operations", 0.9, "Very High", "LLM", sentence_id="j3_0001"),
    ]

    kept, report, rows = apply_extraction_type_gate_to_items(items)

    assert [getattr(i, "text", "") for i in kept] == ["Python", "communication skills"]
    assert kept[0].type == SkillType.HARD
    assert kept[1].type == SkillType.SOFT
    assert report["n_items_dropped"] == 2
    assert report["n_items_demoted_to_soft"] == 1
    assert report["n_unsafe_hard_generation_items_after"] == 0
    assert {r["gate_action"] for r in rows} >= {"kept", "demoted_to_soft", "dropped"}


def test_sentence_rescue_prevents_zero_kept_jobs() -> None:
    df = pd.DataFrame(
        [
            {
                "job_id": "j1",
                "job_date": "2026-01-01",
                "sentence_id": "j1_0001",
                "sentence_text": "Use Python to build and maintain data pipeline automation.",
                "is_relevant": False,
            },
            {
                "job_id": "j1",
                "job_date": "2026-01-01",
                "sentence_id": "j1_0002",
                "sentence_text": "About us",
                "is_relevant": False,
            },
            {
                "job_id": "j2",
                "job_date": "2026-01-01",
                "sentence_id": "j2_0001",
                "sentence_text": "Design REST APIs for backend services.",
                "is_relevant": True,
            },
        ]
    )

    repaired = apply_min_sentence_coverage(df)
    per_job_kept = repaired.groupby("job_id")["is_relevant_final"].sum().to_dict()

    assert per_job_kept == {"j1": 1, "j2": 1}
    rescued = repaired[repaired["rescued_by_min_job_coverage"]]
    assert rescued.iloc[0]["sentence_id"] == "j1_0001"
    assert repaired["relevance_decision"].tolist().count("rescued_by_min_job_coverage") == 1

