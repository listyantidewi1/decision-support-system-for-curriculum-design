# Gold Set Labeling Protocol

## Purpose
Create ground-truth labels for evaluating extraction quality, scoring
calibration, and future-domain mapping accuracy. Supports multi-reviewer
labeling with majority vote and inter-rater reliability (Cohen's Kappa).

## Labeling Options

| Method | Use Case |
|--------|----------|
| **Gold Labeling UI** | Multi-reviewer, auto-save, overlap set for IRR |
| **Manual CSV edit** | Single reviewer, direct edit of gold_*.csv |

## Gold Labeling UI (Recommended for Multi-Reviewer)

```bat
uvicorn gold_labeling_ui.app:app --reload
```

Open `http://127.0.0.1:8000/?labeler_id=alice` (use different labeler_id per reviewer).

Labels are stored in `DATA/labels/gold_labels/` per (gold_id, labeler_id).

**After all reviewers finish:**
```bat
python merge_gold_labels.py
```
Then run `evaluate_extraction.py` and `evaluate_future_mapping.py` (they use `*_merged.csv` automatically).

## Files to Label

| File | Columns to Fill | Items |
|------|----------------|-------|
| gold_skills.csv | is_correct, type_label, bloom_label | ~150 |
| gold_knowledge.csv | is_correct | ~100 |
| gold_future_domain.csv | true_domain_id | ~100 |

## Instructions

### gold_skills.csv
For each row, read the `skill` text and the `job_id` context.

- **is_correct**: Was this skill correctly extracted from the job text?
  - `yes` — the text is a real skill/competence mentioned in the posting
  - `no` — garbage, fragment, not a skill, or hallucinated
- **type_label**: What type of skill is this?
  - `Hard` — technical, domain-specific
  - `Soft` — interpersonal, transferable
  - `Both` — hybrid (e.g., "technical communication")
  - Leave blank if unsure
- **bloom_label**: Bloom's taxonomy level (for hard skills only):
  - `Remember`, `Understand`, `Apply`, `Analyze`, `Evaluate`, `Create`
  - Leave blank if unsure or soft skill
- **labeler_id**: Your reviewer ID (e.g., "dewi", "alice")
- **notes**: Optional free text

### gold_knowledge.csv
For each row, read the `knowledge` text and the `job_id` context.

- **is_correct**: Was this knowledge item correctly extracted?
  - `yes` — real knowledge/concept from the posting
  - `no` — garbage, fragment, or hallucinated
- **labeler_id**: Your reviewer ID
- **notes**: Optional

### gold_future_domain.csv
For each row, read the `item_text` and the `pipeline_domain` assigned by the
pipeline.

- **true_domain_id**: Which future domain best fits this item?
  - Use domain IDs from `future_domains.csv` (or `future_domains_dummy.csv`)
  - Write `none` if the item does not map to any domain
  - Write `unclear` if you cannot decide
- **labeler_id**: Your reviewer ID
- **notes**: Optional

## Overlap Items (IRR)
The **first 20 items** in each tab must be labeled by **all** reviewers to compute
inter-rater reliability (Cohen's Kappa). The Gold Labeling UI marks these with
an "Overlap (IRR)" badge. The export script uses the same random seed so overlap
is deterministic.

## Majority Vote
When multiple reviewers label the same item:
- **is_correct**: yes if count(yes) > count(no); else no (conservative tie-break)
- **type_label**, **bloom_label**, **true_domain_id**: mode (most frequent)

## Quality Checks
- Label all items (do not skip)
- If in doubt, mark `no` for is_correct (conservative)
- For manual CSV: save as UTF-8 (do not change column order)
