# CLAUDE.md — read this before changing anything

Doctoral research codebase: analysing online job postings (OJP) to inform
Indonesian SMK software-engineering (RPL) curriculum design.

This file exists because the repo is **layered sediment**. Work from several
different assistants accumulated across 2026, much of it before the research
proposal was approved. Roughly 74% of the Python predates approval. Old code
here is not a maintenance obligation and **it is not a specification**. Every
session that skips this file re-derives the same archaeology and risks treating
superseded design as current requirement.

---

## 1. The two rules that matter most

### Rule 1 — `docs/IMPLEMENTATION_PLAN.md` is the only specification

The proposal was **approved 2026-10-04**; the build phase began **2026-10-20**.
`docs/IMPLEMENTATION_PLAN.md` is the living plan and the sole in-repo scope
authority. It marks each module **ADAPT / REPLACE / NEW**.

Where a document and the plan disagree, **the plan wins**. Where the plan is
silent, the thing is **out of scope**.

Everything in this list predates approval and describes the **superseded v1
pipeline**. It is history, not requirements:

| Stale (do not treat as current) | mtime |
|---|---|
| `README.md`, `PIPELINE.md` | 2026-05 |
| `SCIENTIFIC_METHODOLOGY.md`, `CALCULATIONS.md`, `RESEARCH_QUESTIONS.md` | 2026-05 |
| `docs/proposal/archive/KAJIAN_AKADEMIK.md`, `docs/proposal/archive/DOCUMENTATION_INDEX.md` | 2026-05 |
| `docs/proposal/archive/PROMPT_*.md` (incl. `PROMPT_REVISI_*`) | 2026-06 |

`docs/proposal/archive/` is where the researcher moved superseded documents on
2026-11-18: 19 pre-approval `.md` files plus old instrument drafts (v1.0/v2.0
`.docx`, gitignored). Nothing live reads it. Six files were deliberately left
OUT of the archive because something live depends on their path:
`docs/IMPLEMENTATION_PLAN.md`, `docs/PHASE_2_1_DESIGN.md`,
`docs/PHASE_2_2_DESIGN.md`, `docs/EXPERT_REVIEW_RUBRIC.md`,
`docs/EXTRACTOR_DECISION.md` and
`docs/proposal/PROMPT_INSTRUMEN_VALIDASI_AHLI_26062026.md` (the plan names it).
The validated instruments are the v3.0 set in
`docs/proposal/PDF/proposal_coworking/instruments/`; the annotation guideline is
`B_Paket_Design_Korpus_dan_Anotasi_v3.0.docx` (code: `AG-RPL-LIMA-v3.0`).

**The tell that proves they are stale:** they instruct you to use
`kkni_labeler.py`, which **does not exist**. The real module is
`competency_kkni_labeller.py`. A doc that names a non-existent file is not
describing this codebase.

> Known defect in the plan itself: line 3 cites
> `docs/proposal/Proposal_Disertasi_04102026_4D_final.pdf` as its source of
> truth. **That file does not exist.** The approved document is
> `docs/proposal/PDF/proposal_coworking/Proposal_Disertasi_04102026_4D_final.docx`.
> Until that pointer is fixed, the plan's own text is the de-facto top authority.

Subordinate scope docs the plan itself spawned (all post-approval, all
trustworthy): `docs/corpus/SOURCE_AUDIT.md`,
`docs/corpus/ACQUISITION_CADENCE.md`, `docs/experiments/split_spec_v1.md`,
`docs/experiments/PREREG_gold_standard_size.md`,
`docs/experiments/PREREG_test_scoring_rule.md`.

### Rule 2 — never date a file from `git log`

Commit dates in this repo **lie**. A single commit (message: *"feat: add
evidence-first core modules, corpus-prep scripts, and tests"*, dated
2026-10-20) bulk-tracked a large pile of files written months earlier. The
commit date is when a file was *tracked*, not when it was *written*.

```bash
ls -l --time-style=long-iso <file>     # mtime = the authorship signal
```

**mtime before 2026-10-04 → pre-approval.**

Also: **every commit SHA changed on 2026-11-17** when history was rewritten to
purge the scraped corpus (§6). Any SHA recorded before that date is dead. Find
commits by message.

### The corollary that prevents the worst mistake

**A pre-approval mtime does NOT mean legacy.** Of the 17 asset rows in the
plan's own inventory, **16 have pre-approval mtimes and are LIVE**, because the
plan names them for ADAPT. Age alone is never sufficient grounds to retire
something here. The test is:

- **LIVE** — the plan names it (any disposition, including ADAPT), *or*
  something the plan names imports it, *or* it is post-approval work, *or* a
  human still runs it as an entry point.
- **LEGACY** — pre-approval mtime **and** the plan never names it **and**
  nothing live reaches it.
- **UNSURE** — say so. A wrong LEGACY call breaks the pipeline; say UNSURE
  instead of guessing.

"Nothing imports it" is **not** evidence of death: entry-point scripts
(`scripts/smoke_*.py`, `plot_rq_figures.py`,
`docs/experiments/learning_curve/learning_curve.py`) legitimately have no
importers.

---

## 2. Which Python runs what

Two virtualenvs, **both inside the working tree**:

| Env | Use | Invoke |
|---|---|---|
| `.venv-ml/` | all ML / pipeline / gold-standard code, **including the Streamlit annotator UI** `gold_standard/app.py` (CPU torch, scipy, pandas, streamlit) | `.\run.ps1` wrapper, or `.venv-ml\Scripts\python.exe` |
| `.venv/` | the annotation collector only (fastapi, uvicorn) | `.venv\Scripts\python.exe` |

The two halves of the annotation stack need **different** venvs: the collector
(`annotation_server.api`, uvicorn) runs under `.venv`, which has fastapi and no
streamlit; the UI (`gold_standard/app.py`) runs under `.venv-ml`, which has
streamlit and no fastapi. Running either under the other's venv fails with
`No module named …`.

System Python on `C:` has **no** ML packages (since the 2026-05-12 migration).

Two consequences that have each cost real time:

1. **Running a suite under the wrong venv looks like a code failure.**
   `tests/test_annotation_server.py` scores 7/23 under `.venv-ml` (no fastapi)
   and 23/23 under `.venv`. A sweep that runs every suite under one interpreter
   is the wrong instrument for this repo.
2. **Both venvs pollute recursive searches.** A bare `grep -r` or
   `find .` returns thousands of `site-packages` hits and may time out before
   reaching project code. Always exclude them:
   ```bash
   grep -r --exclude-dir=.venv --exclude-dir=.venv-ml --exclude-dir=.git <pat> .
   ```
   Prefer `git ls-files` or the Grep tool, which respect this automatically.

---

## 3. Tests

**There is no pytest in this project** — no `pytest.ini`, `pyproject.toml`,
`setup.cfg`, `tox.ini` or `conftest.py`, and neither venv has pytest installed.
Tests are plain scripts with a `_run_standalone()` runner at the bottom:

```bash
.venv-ml\Scripts\python.exe tests\test_split.py
```

Current, passing, and the part of the repo you can trust:

| Suite | Tests | Venv |
|---|---|---|
| `test_split` | 44 | `.venv-ml` |
| `test_split_assert` | 42 | `.venv-ml` |
| `test_split_manifest` | 31 | `.venv-ml` |
| `test_split_groups` | 28 | `.venv-ml` |
| `test_split_seal` | 27 | `.venv-ml` |
| `test_split_logic` | 23 | `.venv-ml` |
| `test_split_keys` | 13 | `.venv-ml` |
| `test_iaa` | 49 | `.venv-ml` |
| `test_iaa_provenance` | 8 | `.venv-ml` |
| `test_metrics` | 25 | `.venv-ml` |
| `test_labeling` | 48 | `.venv-ml` |
| `test_labeling_run` | 18 | `.venv-ml` |
| `test_labeling_conformance` | 28 | `.venv-ml` |
| `test_gold_standard_pipeline` | 11 | `.venv-ml` |
| `test_annotation_server` | 23 | **`.venv`** |

**Five suites silently do not run at all** — they are pytest-style with no
standalone runner, so they print nothing and exit 0 having executed zero tests:
`test_corpus_lineage_and_contract`, `test_evidence_first_repairs`,
`test_scientific_validation_pack`, `test_stability_ablation_pack`,
`test_upstream_evidence_repair`. This is **deliberate** (2026-11-16): every
module they test is marked ADAPT in the plan, so making them green would pin
pre-approval contracts the plan requires to change. Rewrite them *with* their
module's adaptation, not before it. `test_repair_sprint` is also pytest-style
(7 of 8 run standalone; one needs `tmp_path`).

---

## 4. Traps — each of these has already bitten someone

**`pipeline_orchestrator.py` is the v1 keystone. Do not move anything it
calls.** `dashboard/app.py:18` imports it, and `dashboard/` is plan-named
ADAPT — so the orchestrator is live *by coupling*. It subprocess-launches ~20
pre-approval root scripts (`generate_competencies.py`, `recommendations.py`,
`verify_skills.py`, `plot_generator.py`, `pipeline.py`, `import_feedback.py`,
`validate_parameters.py`, `export_*.py`, `evaluate_*.py`, …). Those files look
like textbook legacy and are **load-bearing today**. The plan's ADAPT of
`dashboard/` *is* rewriting that stage table onto
`scripts/run_full_v2_pipeline.py`. Do that first; quarantine the orphaned
stages in the same commit. Quarantining first breaks a plan-named module with
nothing built to replace it.

**`multitask_model.py` fails silently, not loudly.** It is imported by both
`train_jobbert_research.py:24` (legacy) and `pipeline.py:43` (live by
coupling). `pipeline.py:41-50` wraps that import **and** `import config` in one
`try/except ImportError` with a `DummyConfig` fallback. Move or rename
`multitask_model.py` and the pipeline does not crash — it runs on empty label
maps and emits plausible-looking garbage behind a single `Warning:` line.
Prefer a loud failure to this.

**`scripts/` has no `__init__.py`.** Only these are imported as `scripts.*`:
`apply_scientific_sprint1`, `audit_corpus_lineage`, `build_dacum_cbc_package`,
`build_scientific_validation_pack`, `build_stability_ablation_pack`,
`prepare_indonesian_corpus`, `test_clustering_on_real_data`.

**`run.bat` / `run_phase_2.bat` are v1 and already broken.** They call bare
`python`, which has had no ML packages since the venv migration. They are the
only remaining callers of several legacy scripts, so they must be retired in
the same commit as those scripts or they become dangling references a future
session reads as proof of liveness.

**The Skill-LLM bundle is tracked at a doubled path, against the repo's own
written intent.** The Kaggle output was unpacked one level too deep, so
`baseline_versions/skill_llm/skill_llm/…` escapes the `.gitignore` rules
anchored at `baseline_versions/skill_llm/…`. It holds the **only** copy of the
trained LoRA adapter, while `baseline_versions/skill_llm/config.py:27` expects
it one level up at a path that does not exist. `INTEGRATION.md:157` says this
content is "kept local, not committed". Fix by moving the adapter to the path
`config.py` expects and extending `.gitignore` — not by deleting it.

**The annotation stack auto-starts from Windows Task Scheduler, which is not in
git.** `run_annotation_stack.ps1` (repo root) is LIVE and supervises the pilot
collector + annotator UI; it is registered as two per-user logon tasks,
`\GoldStandard\collector-8000` and `\GoldStandard\annotator-ui-8501`. Nothing
imports it and no committed file invokes it, so it reads like an orphan — it is
not. Confirm with `Get-ScheduledTask -TaskPath '\GoldStandard\'`, and read
`annotation_server/README.md` §"Surviving a reboot" before changing it. Its
mutable state lives **outside the tree** at `D:\Projects\_skill-extraction-ops\`
(logs, backups, and the synthetic dry-run DB); the real pilot DB is
`DATA/gold_standard/collector/pilot_100.sqlite`. `-Mode pilot|dryrun` is a
contamination boundary enforced by using different files, not by convention.

**`gold_standard/app.py` silently falls back to writing local JSON.** `app.py`
needs *both* `GS_API_URL` and a `?token=` in the URL to use the collector
(`api_enabled()`). Until 2026-11-17 a token-less visit rendered a fully working
annotator UI that wrote to `DATA/gold_standard/annotations/<split>_<code>.json`
— which is exactly where `gold_standard/iaa.py` reads from, so the work merged
into the reliability evidence unannounced. That contamination has happened once
already (`annotation_server/auth.py:7-8`). The app now refuses when a collector
is configured but no token is present; offline/local mode (no `GS_API_URL`) is
unchanged. Never set `GS_API_TOKEN` on the service: it makes every annotator
authenticate as one identity and UPSERT over each other, with no error anywhere.

**Synthetic IAA numbers reached the real evidence directory once, and the route
was structural.** `gold_standard/iaa.py` took `--ann-dir` for *input* while
hardcoding its *output* directory, so every dry run wrote into
`DATA/gold_standard/iaa/`; and it never checked that an annotated sentence id
belonged to the split's sample. Four reports from a four-sentence fixture
(ids `s1`,`s3`,`s4`) therefore sat there labelled `split: pilot_100`, annotators
AV/AI/AD, Fleiss κ 0.657 "substantial" — indistinguishable from a real pilot
result. They are now in `DATA/gold_standard/_synthetic/` with a
`SYNTHETIC_DO_NOT_CITE.md` explaining the measurements that identify them.
`iaa.py` gained `--out-dir`, a fail-closed id-membership invariant, a refusal to
read quarantined input without redirecting output, and a `provenance` block in
every report. `tests/test_iaa_provenance.py` enforces it, including a test that
fails if any report in the evidence directory has unverified ids — so a fifth
fake set cannot sit there quietly. **Never hand-place a file in
`DATA/gold_standard/iaa/` or `DATA/gold_standard/annotations/`.**

**`plot_aggregated.bat` is broken and is not the pattern.** Still tracked at the
repo root (mtime 2026-03-05), it calls bare `python` — no ML packages since the
2026-05-12 venv migration — and `aggregate_results.py`, which now lives in
`legacy/`. It was missed when its two siblings were quarantined. Its presence
next to `run.ps1` misreads as evidence that bare-`python` root launchers are
current. They are not: `run.ps1` is the model.

**No CI exists.** No `.github/`, no Makefile. Nothing is "run by CI", so that
can never be the reason a file is live.

---

## 5. The live map

Authoritative detail is in `docs/IMPLEMENTATION_PLAN.md` §0. Orientation only:

| Layer | Modules | Status |
|---|---|---|
| 0 — acquisition | `acquisition/` (adapters, batch store, cadence), `job_scraping/` | NEW / ADAPT |
| 1 — corpus governance | `preprocess_jobs_pipeline.py`, `scripts/audit_corpus_lineage.py`, `governance/` (`rules.py`, `ladder.py`), `it_relevance_filter.py` | ADAPT / NEW |
| 2 — gold standard | **`gold_standard/`** (pool, sampling, splits, IAA, seal), `annotation_server/`, `gold_labeling_ui/` | **current work** |
| 2b — silver labels | `labeling/` (verifier, alignment, consensus, guideline-only prompt pack, OpenRouter runner with a hard spending cap, a SYNTHETIC guideline-conformance set and a SkillSpan probe for measuring prompt changes; output goes to the gitignored `DATA/labeling/`, never under `DATA/gold_standard/`), `gold_standard/metrics.py` (the one shared evaluation engine) | NEW, **current work** |
| 3 — extraction | `baseline_versions/` (E0–E6 benchmarks), `extraction_typing.py`, `llm_client_router.py` | ADAPT / NEW |
| 4–5 — competency | `clustering/`, `competency_generator_v2.py`, `competency_evidence_repair.py`, `pipeline_scientific_audits.py` | REUSE / ADAPT |
| 6 — labels & refs | `competency_kkni_labeller.py`, `occupation_mapper.py`, `kkni.py`, `DATA/skema_okupasi_*`, `DATA/spektrum_keahlian/` | ADAPT |
| 7 — trends | `skill_time_trend_analysis.py`, `future_weight_mapping.py`, `future_domains.csv` | ADAPT |
| 8 — delivery | `scripts/build_*_package.py`, `dashboard_v2/app.py`, `dashboard/`, `expert_review.py` | ADAPT |
| chassis | `scripts/run_full_v2_pipeline.py` | ADAPT |

`gold_standard/` is the most current and best-tested part of the repo
(268 passing tests, a group-aware splitter with 17 fail-closed invariants, and
two pre-registrations). Treat it as the reference for how new work should look.

---

## 6. Data policy — the corpus is local only

The scraped job-posting corpus was **purged from all git history on 2026-11-17**
and force-pushed. The repo is **public**. Never commit:

- `DATA/preprocessing/`, `DATA/skillspan_ours/`, `DATA/samples/jobs_sample.csv`
- `results/`, `results_run1|3|4/`, `results_aggregated/`
- `skill_llm_*.jsonl`, `selected_skill_samples_*.html`
- `DATA/labels/gold_{skills,knowledge}*.csv`,
  `feedback_store/human_verified_skills.csv` — these embed raw Indeed job ids
  (`in-<16hex>` → `indeed.com/viewjob?jk=<id>`) plus lifted posting phrases, so
  they are **re-identifiable** even though the annotations are our own work.

`.gitignore` enforces all of the above; `git check-ignore` was verified for
every path. Pre-rewrite history is preserved outside the repo at
`D:\Projects\_skill-extraction-backup-2026-11-17\pre-purge-full.bundle`
(226 MB, all 133 commits, `git bundle verify` passed) — **do not delete it.**

Still open: past public exposure is not undone (the data was reachable for ~6
months; forks and caches keep it). A de-identified label release (hashed
`job_id`, no verbatim spans) is publishable later as its own decision.
`baseline_versions/**/datasets/*.jsonl` is SkillSpan-derived (Zhang et al.) and
was **not** purged — redistributing a reformatted copy of a licensed research
dataset is an undecided licence question.

### Collection constraints (standing, not negotiable)

- API keys and tokens never go in chat or git (`api_keys/` is gitignored).
- No anti-bot or detection evasion; respect `robots.txt`.
- A new external source needs a ToS/ethics audit plus promotor sign-off,
  recorded as a Kn decision in `docs/corpus/SOURCE_AUDIT.md`.
- Jobstreet sessions: ≥24 h cooldown, ≤25 terms / ~1,500–2,000 details per
  session, circuit breaker on 403/429.

---

## 7. `legacy/` — the quarantine

`legacy/` holds pre-approval code that nothing live reaches. It is kept, not
deleted, so provenance survives; it is **not** a source of requirements and
must never be imported by live code.

`tests/test_legacy_boundary.py` enforces this and fails if live code imports
from `legacy/`. If you need something in there, the answer is to port it
deliberately under the plan — not to import it.

---

## 8. Conventions

- Commit messages: conventional prefixes (`feat(scope):`, `fix:`, `chore:`,
  `docs:`). Say *why*, and state what was verified rather than asserting it
  works.
- Pre-registrations live in `docs/experiments/PREREG_*.md`. Amend the
  **evidence** section, never the registered decision, and date the
  registration before results exist.
- Adversarial review findings go in `docs/experiments/*_findings.md` with a
  resolution log. Findings that stay open are listed as open.
- Report measured numbers, not expectations. If a suite fails, say so with the
  output.
