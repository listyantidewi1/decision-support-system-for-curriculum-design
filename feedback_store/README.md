# Feedback Store

Canonical storage for human-in-the-loop review feedback. Used by the pipeline for adaptation (Bloom overrides, human-verified filtering, competency few-shot examples).

## Schema

| File | Description |
|------|--------------|
| `human_verified_skills.csv` | Skills with human_valid, human_bloom; used to filter pipeline output |
| `human_verified_knowledge.csv` | Knowledge with human_valid; used to filter pipeline output |
| `competency_assessments.json` | `{ competency_id: { quality, relevant, notes } }` |
| `bloom_corrections.json` | `{ skill_text: correct_bloom }` for Bloom override |
| `metadata.json` | run_id, review_date, reviewer_id (optional) |

## RAG / Fine-Tuning Extension

The schema is designed to support future RAG and fine-tuning:

- **RAG**: `human_verified_skills` + high-quality competencies (quality >= 4, relevant == yes) can be embedded and indexed. At generation time, retrieve similar items to inject into the prompt.
- **Fine-tuning**: Build dataset `(skills, future_context) -> competency JSON` from human-approved competencies for LoRA or full fine-tuning.

## Usage

1. Run export scripts to create `expert_review_*.csv` in OUTPUT_DIR
2. Use the review web app or edit CSVs manually
3. Run `import_feedback.py` to merge feedback into this store (or web app writes directly)
4. Run pipeline with `--use_human_verified` and competency gen with `--human_verified_only`
