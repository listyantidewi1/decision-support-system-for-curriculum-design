from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_corpus_lineage import (  # noqa: E402
    STATUS_INVALID,
    STATUS_REVIEW,
    STATUS_VALID_ENGINEERING,
    audit_corpus,
    profile_prepared_sentences,
)
from scripts.build_dacum_cbc_package import (  # noqa: E402
    _tokens,
    build_rows,
    infer_duty_area,
    infer_spektrum,
    mapping_status,
    skkni_unit_candidates,
)
from scripts.apply_scientific_sprint1 import (  # noqa: E402
    RECOMMENDATION_EXPLORATORY,
    RECOMMENDATION_OFFICIAL,
    RECOMMENDATION_REVIEW,
    STATUS_REVIEW,
    STATUS_VALID_ENGINEERING as CONTRACT_VALID_ENGINEERING,
    STATUS_VALID_INDONESIAN,
    classify_competency,
    enrich_competencies,
)
from scripts.prepare_indonesian_corpus import (  # noqa: E402
    build_sentence_rows,
    clean_text,
    split_sentences,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _minimal_comp(**updates) -> dict:
    comp = {
        "id": "comp_cloud_api",
        "statement": "Develop cloud APIs for software systems.",
        "related_skills": ["cloud", "API", "software development"],
        "source_job_ids": [f"j{i}" for i in range(1, 11)],
        "grounding_score": 0.91,
        "future_weight": 0.8,
        "occupation_matches": [{"source_type": "template-based", "is_official": True}],
    }
    comp.update(updates)
    return comp


def test_corpus_audit_validates_matching_engineering_lineage(tmp_path: Path) -> None:
    raw = tmp_path / "jobs_sample_n1k_seed42.csv"
    raw.write_text(
        "job_id,title,description,country,site,date_posted\n"
        "j1,Backend Developer,Build APIs and cloud systems.,US,indeed,2026-01-01\n"
        "j2,Cloud Engineer,Operate reliable software infrastructure.,US,indeed,2026-01-02\n",
        encoding="utf-8",
    )
    prepared = tmp_path / "data_prepared_n1k"
    prepared.mkdir()
    metadata = prepared / "jobs_metadata.csv"
    metadata.write_text(
        "job_id,title,description,min_education_kkni\n"
        "j1,Backend Developer,Build APIs and cloud systems.,6\n"
        "j2,Cloud Engineer,Operate reliable software infrastructure.,6\n",
        encoding="utf-8",
    )
    phase1 = tmp_path / "phase1_n1k"
    phase1.mkdir()
    (phase1 / "advanced_skills.csv").write_text(
        "job_id,item_id,text\nj1,s1,API development\nj2,s2,cloud infrastructure\n",
        encoding="utf-8",
    )
    (phase1 / "advanced_knowledge.csv").write_text("job_id,item_id,text\n", encoding="utf-8")
    (phase1 / "job_sentences_mapping.csv").write_text("job_id,sentence_id\nj1,j1_1\nj2,j2_1\n", encoding="utf-8")
    skill_in = tmp_path / "skill_llm_input_n1k.jsonl"
    skill_out = tmp_path / "skill_llm_extractions.jsonl"
    _write_jsonl(skill_in, [{"sentence_id": "j1_1"}, {"sentence_id": "j2_1"}])
    _write_jsonl(skill_out, [{"sentence_id": "j1_1"}, {"sentence_id": "j2_1"}])

    report, coverage_rows, distribution_rows, quality_rows = audit_corpus(
        dataset_id="unit_n1k",
        raw_csv=raw,
        prepared_dir=prepared,
        phase1_dir=phase1,
        skill_llm_input=skill_in,
        skill_llm_output=skill_out,
        jobs_metadata=metadata,
    )

    assert report["corpus_validity_status"] == STATUS_VALID_ENGINEERING
    assert report["job_id_overlap_raw_to_metadata"]["overlap_count"] == 2
    assert report["job_id_overlap_metadata_to_phase1"]["right_coverage"] == 1.0
    assert coverage_rows
    assert distribution_rows
    assert quality_rows


def test_corpus_audit_blocks_mismatched_lineage(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "job_id,title,description,country\nj1,Developer,Build APIs.,US\n",
        encoding="utf-8",
    )
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    metadata = prepared / "jobs_metadata.csv"
    metadata.write_text("job_id,title,description\nj1,Developer,Build APIs.\nj2,Analyst,Analyze systems.\n", encoding="utf-8")
    phase1 = tmp_path / "phase1"
    phase1.mkdir()
    (phase1 / "advanced_skills.csv").write_text("job_id,item_id,text\nj3,s1,cloud\n", encoding="utf-8")

    report, *_ = audit_corpus(
        dataset_id="unit_bad",
        raw_csv=raw,
        prepared_dir=prepared,
        phase1_dir=phase1,
        skill_llm_input=None,
        skill_llm_output=None,
        jobs_metadata=metadata,
    )

    assert report["corpus_validity_status"] == STATUS_INVALID
    assert "raw_to_metadata_overlap_below_0.95" in report["corpus_validity_reasons"]


def test_recommendation_status_is_capped_by_corpus_validity() -> None:
    us_context = {
        "corpus_validity_status": CONTRACT_VALID_ENGINEERING,
        "corpus_country": "US",
        "corpus_language": "en",
        "corpus_dataset_id": "unit_us",
    }
    enriched_us = enrich_competencies(
        [_minimal_comp()],
        denominator=10,
        min_official_jobs=2,
        min_grounding=0.8,
        strong_min_jobs=5,
        moderate_min_jobs=2,
        corpus_context=us_context,
    )
    assert enriched_us[0]["official_core_eligible"] is True
    assert enriched_us[0]["recommendation_status"] == RECOMMENDATION_REVIEW
    assert "corpus_valid_for_engineering_test_only" in enriched_us[0]["official_readiness_blockers"]

    unknown_context = {"corpus_validity_status": STATUS_REVIEW}
    enriched_unknown = enrich_competencies(
        [_minimal_comp()],
        denominator=10,
        min_official_jobs=2,
        min_grounding=0.8,
        strong_min_jobs=5,
        moderate_min_jobs=2,
        corpus_context=unknown_context,
    )
    assert enriched_unknown[0]["recommendation_status"] == RECOMMENDATION_EXPLORATORY

    id_ready = _minimal_comp(expert_review_passed=True, stability_passed=True)
    enriched_id = enrich_competencies(
        [id_ready],
        denominator=10,
        min_official_jobs=2,
        min_grounding=0.8,
        strong_min_jobs=5,
        moderate_min_jobs=2,
        corpus_context={"corpus_validity_status": STATUS_VALID_INDONESIAN, "corpus_country": "ID", "corpus_language": "id"},
    )
    assert enriched_id[0]["recommendation_status"] == RECOMMENDATION_OFFICIAL


def test_ai_token_matching_does_not_classify_maintain_as_ai() -> None:
    cls, reason = classify_competency({
        "statement": "Maintain reliable operational procedures.",
        "related_skills": [],
        "occupation_matches": [],
    })

    assert cls != "core_technical"
    assert "ai" not in reason.lower()


def test_prepare_indonesian_sentences_preserves_original_and_translation_slots() -> None:
    import pandas as pd

    df = pd.DataFrame([{
        "job_id": "id-job-1",
        "job_date": "2026-01-01",
        "description": "- Pendidikan minimal D3 Teknik Informatika\n- Menguasai Java dan SQL",
        "query_family": "software_developer",
        "query_term": "programmer",
        "site": "indeed",
        "_description_language": "id",
    }])

    rows = build_sentence_rows(
        df,
        raw_path=PROJECT_ROOT / "DATA" / "preprocessing" / "dummy.csv",
        translation_map={"id-job-1_0002": "Master Java and SQL."},
        min_len=5,
        max_len=200,
        language_detector="heuristic",
    )

    assert list(rows["sentence_id"]) == ["id-job-1_0001", "id-job-1_0002"]
    assert rows.loc[0, "sentence_text_original"] == "Pendidikan minimal D3 Teknik Informatika"
    assert rows.loc[0, "sentence_text_en"] == ""
    assert rows.loc[0, "translation_status"] == "original_not_translated"
    assert rows.loc[1, "sentence_text"] == "Master Java and SQL."
    assert rows.loc[1, "translation_status"] == "translated_external"


def test_prepared_sentence_profile_reports_bilingual_coverage(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "job_sentences_for_pipeline.csv").write_text(
        "job_id,sentence_id,sentence_text,sentence_text_original,sentence_text_en,source_language,normalized_language,translation_status\n"
        "j1,j1_0001,Original ID,Original ID,,id,id,original_not_translated\n"
        "j1,j1_0002,Translated EN,Original ID 2,Translated EN,id,en,translated_external\n",
        encoding="utf-8",
    )

    profile = profile_prepared_sentences(prepared)

    assert profile["prepared_sentence_count"] == 2
    assert profile["prepared_original_text_coverage"] == 1.0
    assert profile["prepared_pipeline_text_coverage"] == 1.0
    assert profile["prepared_english_text_coverage"] == 0.5
    assert profile["prepared_translation_status_distribution"]["original_not_translated"] == 1


def test_indonesian_validity_needs_manual_review_without_english_coverage(tmp_path: Path) -> None:
    raw = tmp_path / "raw_id.csv"
    raw.write_text(
        "job_id,title,description,location\n"
        "j1,Programmer,Pendidikan minimal D3 Teknik Informatika,Jakarta, Indonesia\n",
        encoding="utf-8",
    )
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "jobs_metadata.csv").write_text(
        "job_id,title,description,min_education_kkni\n"
        "j1,Programmer,Pendidikan minimal D3 Teknik Informatika,5\n",
        encoding="utf-8",
    )
    (prepared / "job_sentences_for_pipeline.csv").write_text(
        "job_id,sentence_id,sentence_text,sentence_text_original,sentence_text_en,source_language,normalized_language,translation_status\n"
        "j1,j1_0001,Pendidikan minimal D3 Teknik Informatika,Pendidikan minimal D3 Teknik Informatika,,id,id,original_not_translated\n",
        encoding="utf-8",
    )
    phase1 = tmp_path / "phase1"
    phase1.mkdir()
    (phase1 / "advanced_skills.csv").write_text("job_id,item_id,text\nj1,s1,Java\n", encoding="utf-8")

    report, *_ = audit_corpus(
        dataset_id="id_needs_translation",
        raw_csv=raw,
        prepared_dir=prepared,
        phase1_dir=phase1,
        skill_llm_input=None,
        skill_llm_output=None,
        jobs_metadata=prepared / "jobs_metadata.csv",
    )

    assert report["corpus_validity_status"] == STATUS_REVIEW
    assert "prepared_english_text_coverage_below_0.95" in report["corpus_validity_reasons"]


def test_dacum_cloud_and_devops_map_to_infrastructure_not_support() -> None:
    comp = {
        "statement": "Configure cloud platforms and CI/CD pipelines for reliable application deployment.",
        "related_skills": ["cloud infrastructure", "DevOps", "container orchestration"],
        "occupation_matches": [{"occupation_en": "Cloud Engineer"}],
        "competency_class": "core_technical",
    }
    spektrum = {
        "4.2.1": {
            "spektrum_code": "4.2.1",
            "bidang": "Teknologi Informasi",
            "program": "Teknik Komputer dan Jaringan",
            "konsentrasi": "Teknik Komputer dan Jaringan",
        },
        "4.1.1": {
            "spektrum_code": "4.1.1",
            "bidang": "Teknologi Informasi",
            "program": "Pengembangan Perangkat Lunak dan Gim",
            "konsentrasi": "Rekayasa Perangkat Lunak",
        },
    }

    assert infer_duty_area(comp)[0] == "Cloud and infrastructure operations"
    assert infer_spektrum(comp, spektrum)[0]["spektrum_code"] == "4.2.1"


def test_dacum_preserves_official_anchor_when_top_match_is_community() -> None:
    comp = {
        "id": "comp_mixed_occ",
        "statement": "Develop embedded software and troubleshoot computer systems.",
        "recommendation_status": RECOMMENDATION_REVIEW,
        "corpus_validity_status": CONTRACT_VALID_ENGINEERING,
        "competency_class": "core_technical",
        "source_job_ids": ["j1", "j2", "j3"],
        "occupation_matches": [
            {
                "occupation_en": "Embedded Systems Engineer",
                "occupation_id": "Teknisi Sistem Tertanam",
                "source_type": "community_non_official",
                "is_official": False,
            },
            {
                "occupation_en": "Computer Technician",
                "occupation_id": "Teknisi Komputer",
                "source_type": "template-based",
                "is_official": True,
            },
        ],
    }
    spektrum = {
        "4.2.1": {
            "spektrum_code": "4.2.1",
            "bidang": "Teknologi Informasi",
            "program": "Teknik Komputer dan Jaringan",
            "konsentrasi": "Teknik Komputer dan Jaringan",
        },
        "4.1.1": {
            "spektrum_code": "4.1.1",
            "bidang": "Teknologi Informasi",
            "program": "Pengembangan Perangkat Lunak dan Gim",
            "konsentrasi": "Rekayasa Perangkat Lunak",
        },
    }

    status, blockers = mapping_status(comp)
    chain_rows, *_ = build_rows([comp], spektrum, {}, [])
    row = chain_rows[0]

    assert status == "official_or_template"
    assert blockers == ["community_mapping_present_as_secondary"]
    assert row["top_occupation_en"] == "Embedded Systems Engineer"
    assert row["top_official_occupation_en"] == "Computer Technician"
    assert row["occupation_mapping_status"] == "official_or_template"
    assert row["curriculum_use_status"] == "curriculum_review_candidate"


def test_dacum_skkni_matching_ignores_stopword_only_overlap() -> None:
    comp = {
        "statement": "Use and maintain reliable data APIs.",
        "related_skills": ["data API"],
        "occupation_matches": [],
    }
    units = [
        {"skkni_domain": "x", "unit_code": "U1", "unit_title": "Use the and of"},
        {"skkni_domain": "x", "unit_code": "U2", "unit_title": "Maintain data API services"},
    ]
    candidates = skkni_unit_candidates(comp, units, top_n=5)

    assert {"use", "and", "the", "of"}.isdisjoint(_tokens("Use the and of"))
    assert [c["unit_code"] for c in candidates] == ["U2"]
