# `legacy/` — pre-approval code, kept for provenance, not for use

Everything here predates the proposal approval of **2026-10-04**, is not named
in `docs/IMPLEMENTATION_PLAN.md`, and is reached by nothing live.

It is **kept rather than deleted** so the research provenance survives: a
doctoral audit trail is an asset, and `git log` on these paths still shows how
the work developed. But:

- **This is not a specification.** Do not read anything here as a statement of
  what the system should do. `docs/IMPLEMENTATION_PLAN.md` is the only scope
  authority.
- **Live code must never import from here.** `tests/test_legacy_boundary.py`
  enforces that and fails if it happens.
- **These files are not expected to run.** They were retired in place, not
  ported: `run.bat` / `run_phase_2.bat` still contain repo-root-relative paths
  and call bare `python` (which has had no ML packages since the 2026-05-12
  venv migration, see `CLAUDE.md` §2), and `train_jobbert_research.py` imports
  `multitask_model`, which deliberately **stayed at the repo root**.

If you need something in here, port it deliberately under the plan. Do not
import it, and do not move it back without a reason recorded in the commit.

## What is here and why it was retired

| File | Why |
|---|---|
| `aggregate_results.py` | v1 result aggregator. Its only outbound references to `plot_generator.py` / `plot_scientific_analysis.py` are string literals in a `subprocess.run([...], check=False)` behind a `--plot` flag — broken since the venv migration, and silently so. |
| `skill_normalizer.py` | Superseded. `normalize_skills` / `apply_normalization` appear nowhere else in the repo, at symbol level. |
| `compute_education_demand.py` | Superseded by `competency_education_aggregator.aggregate_education_demand`, which `scripts/run_full_v2_pipeline.py:776` imports. Its only callers were the two `.bat` files, now here too. |
| `export_recommendations_for_review.py` | v1 review export. Only callers were the two `.bat` files. |
| `train_jobbert_research.py` | Root copy, 8,128 B @ 2025-11-30, superseded by the fuller archive copy at `baseline_versions/jobbert_crf/train_jobbert_research.py` (14,371 B @ 2025-12-01) which has its own `config.py`, `data_utils.py` and `multitask_model.py`. JobBERT was also ruled insufficient as the production extractor. |
| `scripts/weight_sensitivity_extraction.py` | v1 sensitivity analysis. Only caller was `run_phase_2.bat`. |
| `scripts/trend_score_sensitivity.py` | v1 trend sensitivity. Referenced only by pre-approval docs. |
| `scripts/power_analysis_gold_set.py` | Superseded by `docs/experiments/PREREG_gold_standard_size.md`, which registers the gold-standard size from a measured learning curve. |
| `scripts/smoke_role_normalizer.py` | Duplicate: defines `_ROLE_BUCKETS` locally instead of importing `role_normalization`. The live smoke test is `scripts/smoke_role_normalization.py`, which does import it. |
| `skill_trend_holdout_validation.py` | v1 check of trend detection on held-out months (old RQ3). Reads the pre-approval `results/` CSVs; nothing calls it; the researcher confirmed on 2026-11-19 that she does not run it. Its IDEA is worth porting on purpose into `competency/temporal.py` once there are enough time windows: predict direction on early windows, check it on later ones. |
| `plot_aggregated.bat` | v1 plotting launcher. Calls bare `python` and `aggregate_results.py` (already here). Moved 2026-11-19; it had been missed when its two siblings were retired. |
| `run.bat`, `run_phase_2.bat` | v1 orchestration entry points. Retired together with the scripts they call, so they do not survive as dangling references that a future session reads as proof those scripts are live. |

## What was deliberately NOT quarantined

Two categories were left in place on purpose, and the reasoning matters more
than the list:

**1. The `pipeline_orchestrator.py` cluster — live by coupling.**
`dashboard/app.py:18` imports `pipeline_orchestrator`, and `dashboard/` is
plan-named **ADAPT**. The orchestrator subprocess-launches roughly twenty
pre-approval root scripts (`generate_competencies.py`, `recommendations.py`,
`verify_skills.py`, `plot_generator.py`, `plot_scientific_analysis.py`,
`pipeline.py`, `import_feedback.py`, `apply_feedback.py`,
`validate_parameters.py`, `merge_gold_labels.py`, `export_for_review.py`,
`export_competencies_for_review.py`, `evaluate_*.py`, `log_run_metadata.py`,
`curriculum_loader.py`, `domain_batching.py`, …). Every one of them *looks*
like textbook legacy and is load-bearing today. The plan's ADAPT of
`dashboard/` is precisely the work of rewriting that stage table onto
`scripts/run_full_v2_pipeline.py`. Quarantine them **after** that lands, in the
same commit — not before, which would break a plan-named module with nothing
built to replace it.

**2. `multitask_model.py` — the silent-failure trap.** It is imported by
`train_jobbert_research.py:24` (now here) *and* `pipeline.py:43` (live by
coupling). `pipeline.py:41-50` wraps that import together with `import config`
in a single `try/except ImportError` that falls back to a `DummyConfig` with
empty label maps. Moving it would not raise — the pipeline would run and emit
plausible-looking garbage behind one `Warning:` line. It stays at the root
until `pipeline.py` is retired.

Also left in place: entry-point scripts with no importers
(`scripts/smoke_*.py`, `plot_rq_figures.py`,
`docs/experiments/learning_curve/*.py`). Having no importer is normal for a
script a human runs, and is not evidence of death.

---

Quarantined 2026-11-17. Verdicts came from a seven-agent survey of the repo
against the implementation plan, each LEGACY call then attacked by an
adversarial critic whose job was to refute it; the critic overturned three of
them and reclassified about twenty files as live-by-coupling. Every file moved
here was additionally confirmed by a mechanical check to have no Python
importer anywhere outside `.venv*`, and to be absent from the implementation
plan.
