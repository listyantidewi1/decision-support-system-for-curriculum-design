import sys
from pathlib import Path

# Python puts the SCRIPT's directory (tests/) on sys.path, not the cwd, so the
# project modules below are unimportable when this file is run standalone --
# which is the only way it CAN be run here, since neither venv has pytest.
# Every other suite in tests/ already does this; this one was missed.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from competency_evidence_repair import EvidenceRepairConfig, repair_competencies
from competency_repair import (
    RepairStageConfig,
    dedupe_curriculum_competencies,
    select_recovery_candidates,
)
from competency_v2_schema import CompetencyV2
from scripts.apply_scientific_sprint1 import classify_competency


def _comp(
    cid: str,
    statement: str,
    skills: list[str],
    jobs: int = 5,
    sentences: list[str] | None = None,
) -> CompetencyV2:
    return CompetencyV2(
        id=cid,
        statement=statement,
        related_skills=skills,
        rationale="This is a test rationale that is long enough for schema-style checks.",
        batch_reasoning_id="br_test",
        contributing_item_ids=["i1", "i2"],
        source_job_ids=[f"j{i}" for i in range(jobs)],
        source_sentence_ids=[f"s{i}" for i in range(len(sentences or []))],
        source_sentences=sentences or [
            "Use Python to build data processing scripts for production workflows.",
            "Apply Python libraries to automate data preparation and reporting tasks.",
        ],
        grounding_score=1.0,
        skill_grounding_score=1.0,
        sentence_relevance_score=1.0,
        future_weight=0.7,
    )


def test_repair_classifier_keeps_generic_support_out_of_core() -> None:
    examples = [
        (
            "Apply a strong work ethic and flexible professional conduct.",
            ["Work ethic", "work flexibly"],
            "employability_support",
        ),
        (
            "Collaborate across functions to analyse information and present findings.",
            ["work cross-functionally", "Working cross-functionally"],
            "employability_support",
        ),
        (
            "Apply problem-solving methods to diagnose routine issues.",
            ["problem-solving skills", "Problem-solving"],
            "employability_support",
        ),
        (
            "Coach and mentor engineers so they improve delivery quality.",
            ["mentoring", "providing guidance and mentorship"],
            "engineering_process",
        ),
    ]
    for statement, skills, expected in examples:
        cls, _reason = classify_competency({"statement": statement, "related_skills": skills})
        assert cls == expected


def test_repair_classifier_accepts_specific_technical_anchors() -> None:
    examples = [
        ("Apply SQL to query relational databases.", ["SQL", "databases"]),
        ("Use Python to build data processing scripts.", ["Python"]),
        ("Design REST APIs for backend service integration.", ["REST API", "APIs"]),
        ("Implement infrastructure as code with Terraform.", ["Terraform", "Infrastructure as Code"]),
        ("Build automated testing workflows for software releases.", ["automated testing", "CI/CD"]),
    ]
    for statement, skills in examples:
        cls, _reason = classify_competency({"statement": statement, "related_skills": skills})
        assert cls == "core_technical"


def test_repair_classifier_marks_cad_and_clinical_safeguarding_out_of_domain() -> None:
    cad_cls, _ = classify_competency({
        "statement": "Create physical product designs using CAD and drafting methods.",
        "related_skills": ["CAD", "drafting", "prototyping"],
    })
    clinical_cls, _ = classify_competency({
        "statement": "Apply safeguarding requirements for clinical governance and mental capacity decisions.",
        "related_skills": ["safeguarding", "clinical governance"],
    })
    assert cad_cls == "out_of_domain_or_needs_review"
    assert clinical_cls == "out_of_domain_or_needs_review"


def test_evidence_repair_allows_high_support_single_specific_python_skill() -> None:
    comp = _comp(
        "python",
        "Apply Python to build and maintain software or data-processing solutions.",
        ["Python", "Python)", "Python."],
        jobs=18,
        sentences=[
            "Use Python to build data processing scripts for production workflows.",
            "Apply Python libraries to automate data preparation and reporting tasks.",
            "Maintain Python code used for analytics and operational reporting.",
        ],
    )
    repaired, dropped, _report = repair_competencies(
        [comp],
        EvidenceRepairConfig(
            disable_sbert=True,
            apply_final_evaluator_gate=False,
            min_supported_skills=2,
            min_supporting_sentences=2,
        ),
    )
    assert not dropped
    assert repaired[0].related_skills == ["Python"]
    assert len(repaired[0].source_sentences) >= 2


def test_evidence_repair_still_drops_single_generic_communication_skill() -> None:
    comp = _comp(
        "communication",
        "Communicate clearly with internal and external stakeholders.",
        ["communication skills", "communication skills."],
        jobs=25,
        sentences=[
            "Strong communication skills are required for this role.",
            "Communicate effectively with stakeholders and team members.",
        ],
    )
    repaired, dropped, _report = repair_competencies(
        [comp],
        EvidenceRepairConfig(
            disable_sbert=True,
            apply_final_evaluator_gate=False,
            min_supported_skills=2,
            min_supporting_sentences=2,
        ),
    )
    assert not repaired
    assert dropped


def test_curriculum_dedup_merges_frontend_and_fullstack_same_jobs() -> None:
    jobs = 5
    fullstack = _comp(
        "fullstack",
        "Design and integrate end-to-end software solutions across front-end, back-end, data, and infrastructure components.",
        ["front-end", "back-end", "frontend"],
        jobs=jobs,
    )
    frontend = _comp(
        "frontend",
        "Own and implement the frontend layer for browser-based and web user experiences.",
        ["frontend", "front-end"],
        jobs=jobs,
    )
    survivors, merged, groups = dedupe_curriculum_competencies([fullstack, frontend])
    assert len(survivors) == 1
    assert len(merged) == 1
    assert groups
    assert survivors[0].merged_from


def test_curriculum_dedup_keeps_cloud_platform_and_iac_separate() -> None:
    cloud = _comp(
        "cloud",
        "Analyse cloud platform concepts to select services across AWS, Azure, and Google Cloud.",
        ["cloud platforms", "AWS", "Azure", "Google Cloud"],
        jobs=24,
    )
    iac = _comp(
        "iac",
        "Implement infrastructure as code to provision and manage cloud infrastructure.",
        ["Terraform", "Infrastructure as Code"],
        jobs=9,
    )
    iac.source_job_ids = [f"x{i}" for i in range(9)]
    survivors, merged, _groups = dedupe_curriculum_competencies([cloud, iac])
    assert len(survivors) == 2
    assert not merged


def test_high_support_failed_technical_candidate_is_selected_for_recovery(tmp_path: Path) -> None:
    failed = _comp(
        "aws_fail",
        "Design and implement cloud migration and deployment solutions on AWS.",
        ["AWS", "Terraform", "deployment"],
        jobs=46,
    )
    cfg = RepairStageConfig(
        denominator=982,
        output_dir=tmp_path,
        enable_targeted_recovery=False,
    )
    selected = select_recovery_candidates([failed], cfg)
    assert [c.id for c in selected] == ["aws_fail"]
