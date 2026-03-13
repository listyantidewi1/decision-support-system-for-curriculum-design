# Documentation Index

Central index for all project documentation.

## Core Documents

| Document | Description |
|----------|-------------|
| [../README.md](../README.md) | Project overview, quick start, repository structure |
| [../PIPELINE.md](../PIPELINE.md) | Pipeline architecture, phases, data flow, steps, extraction design (Direction A), troubleshooting |
| [../CALCULATIONS.md](../CALCULATIONS.md) | Formulas: priority score, future weight, ranking, voting, FDR |
| [../SCIENTIFIC_METHODOLOGY.md](../SCIENTIFIC_METHODOLOGY.md) | **Full scientific documentation** with worked examples |
| [../RESEARCH_QUESTIONS.md](../RESEARCH_QUESTIONS.md) | RQ1–RQ5, metrics, gold set design, power analysis, limitations |

## Scientific Methodology (Summary)

The **[SCIENTIFIC_METHODOLOGY.md](../SCIENTIFIC_METHODOLOGY.md)** document covers:

1. **Binomial test** — H₀: precision = 0.5 vs H₁: precision > 0.5
2. **Effect sizes** — Odds ratio, risk difference, Cohen's h
3. **Multi-comparison** — Bonferroni, Benjamini-Hochberg, two-proportion z-test
4. **Power analysis** — Required n for gold set (n ≥ 37 for power 0.80)
5. **Inter-rater reliability** — Cohen's Kappa (2 raters), Fleiss' Kappa (3+)
6. **Normalization** — Deterministic grouping for skills/knowledge/competencies
7. **Priority score & future weight** — Formulas with numerical examples
8. **FDR trend detection** — Benjamini-Hochberg for emerging/declining skills
9. **Weight sensitivity** — Jaccard vs baseline top-20
10. **Recall** — Limitation in gold-set paradigm
11. **Calibrated verification** — AUC-ROC, Brier, Youden
12. **P@N & NDCG** — Recommendation evaluation

## Scientific Plot Interpretations

| Document | Description |
|----------|-------------|
| [../docs/FIGURE_INTERPRETATIONS.md](../docs/FIGURE_INTERPRETATIONS.md) | How to read and interpret each scientific plot |

## Labeling & Review

| Document | Description |
|----------|-------------|
| [EXPERT_REVIEW_RUBRIC.md](EXPERT_REVIEW_RUBRIC.md) | Review criteria: Valid/Invalid, Quality 1–5, Relevance |
| [../DATA/labels/LABELING_PROTOCOL.md](../DATA/labels/LABELING_PROTOCOL.md) | Gold labeling instructions, majority vote, IRR overlap |
| [../gold_labeling_ui/README.md](../gold_labeling_ui/README.md) | Gold labeling web UI |
| [../dashboard/README.md](../dashboard/README.md) | Dashboard (admin, school) |
| [../review_ui/README.md](../review_ui/README.md) | Internal review UI |
| [../job_scraping/README.md](../job_scraping/README.md) | Job scrapers, default data source, pipeline integration |

## Extraction Design (Direction A)

- **BERT per sentence** (128-token limit); **LLM on full job** (full context)
- **BERT knowledge** passed to LLM as anti-hallucination context; **knowledge output LLM-only**
- See [PIPELINE.md](../PIPELINE.md) §2b and [SCIENTIFIC_METHODOLOGY.md](../SCIENTIFIC_METHODOLOGY.md) §16a

## Scripts (Key Scientific Logic)

| Script | Scientific / Logic |
|--------|--------------------|
| `run_with_job_scraping.py` | One-step: preprocess (english_jobs.csv) + pipeline |
| `evaluate_extraction.py` | Binomial test, effect sizes, Wilson CI, pairwise z-test, BH |
| `import_feedback.py` | Majority vote, Cohen's Kappa, Fleiss' Kappa |
| `future_weight_mapping.py` | Future weight formula, normalization, grouping |
| `generate_competencies.py` | Domain-based batching, anti-hallucination (prompt + related_skills filter) |
| `domain_batching.py` | Domain grouping, on-the-fly domain assignment, batch merge by domain similarity |
| `recommendations.py` | Priority score, weight sensitivity |
| `skill_time_trend_analysis.py` | FDR (Benjamini-Hochberg), trend labels |
| `validate_parameters.py` | AUC-ROC, Brier, calibration, Youden |
| `scripts/power_analysis_gold_set.py` | Power for one-proportion binomial |
