---
name: "Plan Review (Superseded)"
overview: Review findings have been merged into improve_pipeline_results_consolidated.plan.md
todos: []
isProject: false
---

# Plan Review: Issues to Address and Clarifications

**Superseded by:** [improve_pipeline_results_consolidated.plan.md](improve_pipeline_results_consolidated.plan.md) — use the consolidated plan for implementation.

---

## Summary of Findings (now integrated into consolidated plan)

---

## Critical Fixes

### 1. Top-3 Index and Few-Domain Handling

**Current plan:** `np.argsort(sim_mat, axis=1)[:, -2]` and `[:, -3]` for 2nd/3rd best.

**Finding:** The argsort approach is correct (`[:, -1]` = max, `[:, -2]` = 2nd max, `[:, -3]` = 3rd max). However, [future_weight_mapping.py](future_weight_mapping.py) must handle:

- `sim_mat.shape[1] >= 2`: only compute `top2_domain_id` (top3 is undefined).
- `sim_mat.shape[1] >= 3`: compute both `top2_domain_id` and `top3_domain_id`.

**Action:** When `sim_mat.shape[1] < 3`, set `top3_domain_id` (and optionally `top3_similarity`) to empty string or NaN. When `sim_mat.shape[1] < 2`, set both top2 and top3 accordingly.

### 2. Report Path for Calibration Weights

**Current plan:** "Use feedback_dir (or output_dir) to locate parameter_validation_report.json."

**Finding:** [validate_parameters.py](validate_parameters.py) writes `parameter_validation_report.json` to `output_dir` (line 353: `report_path = out_dir / args.output`). It writes `calibrated_threshold.json` to `feedback_dir`.

**Action:** In verify_skills.py, look for `parameter_validation_report.json` in **output_dir only** (the same `--output_dir` passed to verify_skills). Do not use feedback_dir for the validation report.

### 3. Column Mapping for Calibration Weights

**Current plan:** Compute weights from Brier for each signal.

**Finding:** [validate_parameters.py](validate_parameters.py) validates: `confidence_score`, `semantic_density`, `context_agreement`. [verify_skills.py](verify_skills.py) uses: `confidence_score`, `model_agreement_score` (or falls back to conf), `semantic_density`, and `_freq` (frequency). There is **no** `model_agreement_score` in the validation report. `advanced_skills.csv` has `context_agreement` per skill, not `model_agreement_score`.

**Action:**

- Map validation Brier scores: `confidence_score` → W_CONFIDENCE, `semantic_density` → W_DENSITY, `context_agreement` → W_AGREEMENT (as proxy for agreement; verify_skills prefers `model_agreement_score` when present but will use context_agreement if that is what exists in advanced_skills).
- Keep W_FREQUENCY fixed (no Brier in report); include it in normalization so the four weights still sum to 1.

### 4. Bloom Availability in Skill Sources

**Current plan:** Load Bloom from verified_skills.csv, advanced_skills_human_filtered.csv, or advanced_skills.csv.

**Finding:**

- [pipeline.py](pipeline.py) writes `advanced_skills.csv` with a `bloom` column (line 1773).
- [verify_skills.py](verify_skills.py) passes through all columns from advanced_skills.csv except `_freq`, so `verified_skills.csv` will have `bloom` when the input has it.
- [generate_competencies.py](generate_competencies.py) loads skills from these CSVs but currently only uses the `skill` column (lines 568–604). It does not read `bloom`.

**Action:** When loading skills in generate_competencies, also load the `bloom` column. For skills with multiple rows (e.g. from verified_skills), use the mode or the first non-N/A value as the Bloom level per skill. Handle missing `bloom` gracefully: if absent, omit Bloom from the prompt and mention "Bloom alignment optional when not available."

---

## Clarifications (Non-Blocking)

### 5. Phase 1 vs Phase 2 – When Does parameter_validation_report Exist?

**Finding:** Phase 1 runs `verify_skills` but does **not** run `validate_parameters`. Phase 2 runs `validate_parameters` then `verify_skills`. Therefore `parameter_validation_report.json` will not exist in Phase 1 runs.

**Clarification:** The plan already specifies "Fallback to current fixed weights if the report is missing." Ensure the implementation explicitly checks for the report path and falls back cleanly.

### 6. Gold File Domain ID Inconsistency

**Finding:** [DATA/labels/gold_future_domain.csv](DATA/labels/gold_future_domain.csv) contains `true_domain_id: "3-Dec"` (line 16), which likely should be `DEC03`. The evaluator compares lowercase strings, so `"3-dec"` will not match `"dec03"`.

**Recommendation:** Consider normalizing gold domain IDs or fixing this label. This is a data-quality issue, not a plan bug.

### 7. Optional Section 1.3 (Exclude Low-Margin)

**Status:** Section 1.3 is marked optional and is not in the todos. No change needed unless you want to add it as a future phase.

---

## Implementation Checklist Updates

Add these implementation notes to the plan:

1. **future_weight_mapping.py**: Guard top2/top3 with `sim_mat.shape[1] >= 2` and `>= 3`; use empty string or NaN when unavailable.
2. **verify_skills.py**: Load `parameter_validation_report.json` from `output_dir`; map `context_agreement` Brier to agreement weight; keep `W_FREQUENCY` fixed.
3. **generate_competencies.py**: Load `bloom` from the skill CSV; build `skill_bloom_map`; handle missing `bloom`; use mode/first non-N/A when multiple rows per skill.

---

## Validation After Implementation

1. Run with **< 3 domains** in future_domains.csv and confirm top3 columns are handled without error.
2. Run **Phase 1 only** and confirm verify_skills uses fixed weights (no crash when report is missing).
3. Run **Phase 2** and confirm calibration-aware weights are applied when the report exists.
4. Run competency generation with and without a `bloom` column and confirm both paths work.

