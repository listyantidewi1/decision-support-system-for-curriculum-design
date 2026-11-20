# CLAUDE.md — read this before changing anything

> **Picking this work up?** After this file, read `docs/HANDOFF.md` (where every
> part stands, what is decided, what was in progress) and
> `docs/experiments/EXPERIMENT_LOG.md` (every run, with its numbers).

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
`docs/experiments/PREREG_test_scoring_rule.md`,
`docs/experiments/PREREG_pilot_gate.md`,
`docs/experiments/PREREG_d3_conflict_routing.md` (where a D1/D2 conflict goes:
the risk rule the researcher chose on 2026-11-19; a test keeps it identical to
`fivetag/conflict_routing.py`).

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
| `test_after_annotation` | 20 | `.venv-ml` |
| `test_gold_to_models` | 10 | `.venv-ml` |
| `test_fivetag` | 73 | `.venv-ml` |
| `test_competency` | 60 | `.venv-ml` |
| `test_zoning` | 6 | `.venv-ml` |
| `test_formulate` | 13 | `.venv-ml` |
| `test_skkni` | 27 | `.venv-ml` |
| `test_config_registry` | 5 | `.venv-ml` |
| `test_dashboard_v3` | 9 | `.venv-ml` |
| `test_acquisition_monthly` | 11 | `.venv-ml` |
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

**`gold_standard.sample` re-draws the pilot every time it runs.** Each run draws
`pilot_100` and `practice_10` again before anything else, including before
`--main`. The draw is seeded, so it is identical while the sentence pool is
unchanged — but the pool is rebuilt after corpus rounds, and the same command
would then have replaced the pilot sentences while the experts' annotations
still referred to the old ones. Since 2026-11-19 it refuses to overwrite a
sample whose sentences differ unless the split is named with `--redraw`. Never
redraw a sample that people have annotated.

**The pilot go/no-go rule is registered and must not change once pilot data
exist.** `docs/experiments/PREREG_pilot_gate.md` and `gold_standard/gate.py`
carry the same thresholds (a test keeps them identical). The researcher
approved it on 2026-11-19, before any annotation. Rehearsing it on synthetic
data is fine; editing the thresholds after a real pilot annotation is not.

**Nothing `fivetag/` has printed so far is a result.** No gold standard exists
yet (2026-11-19). The models were exercised on three things, none of which is
evidence: template-generated sentences (`fivetag.synthetic`), the guideline
conformance set, and SkillSpan sentences annotated under the guideline. The E0
lexicon was written with the last two in view, so E0's score on them is
optimistic by construction. The E1 numbers come from a tiny randomly-initialised
model (`fivetag.tiny`) that exists to prove the training loop runs. The router's
thresholds (`RouterConfig`) are starting values and must be fixed on the dev
split before the test split is opened. `labeling.versus_gold` and
`gold_standard.materialise` both refuse sealed test sentences; keep it that way.

**A `competency.pipeline` run is an engineering run until the extractor behind it
has been measured.** The first one (2026-11-19, E0 evidence, 581 postings) gave
445 statements; they are not findings and the output says so. Three things to
know before changing it: the clusterer's own "phrase seen in ≥ 2 postings"
filter is switched off on purpose (five-tag action phrases are long and mostly
unique; it discarded two thirds of the evidence), rarity is judged per group
instead; the KKNI labeller assigns almost nothing on this path (17 of 445)
because it was built to correct an LLM's suggestion; and the status "siap
ditelaah formal" is unreachable without three facts set by a person
(`contract.StudyFacts`). Lexicon `v2` (v1 + O*NET-derived technology names) is
what this pipeline uses; `fivetag`'s default is still `v1`.

**A competency is a verb plus a knowledge, and its statement names no tool.**
The researcher's decision of 2026-11-20 ("Perform programming", not "Perform
programming using Python, PHP and JavaScript"). `competency/structure.py` reads
the verbs and the shared object out of the group's hard-skill phrases;
`deterministic_statement` joins them and leaves the tools in the list beside
the competency. Do not put "using <tools>" back as the default (a test holds
it), and do not try to find the knowledge through the KNOWLEDGE tag: under 1%
of hard-skill phrases contain a KNOWLEDGE-tagged term, because the extractor
tags a governed noun phrase as part of the hard skill (EXP-028). The cognitive
level (C1–C6) is read from the verbs the postings use
(`structure.cognitive_profile`); a default verb such as "perform" is the
method's wording and must never set the level (her correction of 2026-11-20;
a test holds it). A group is a FAMILY: `structure.split_by_level` makes one
competency per cognitive level in it, each with its own statement and demand
("apply web programming" and "develop a website" are two competencies; her
words). Do not go back to one level per group. The verb-to-level table is the
file `competency/verbs/bloom_verbs_v2.tsv` with `SOURCES.md`; it is unreviewed,
and a changed row is a new version. The generator's paragraph blends levels, so
it is stored as the family's description, never as a competency statement.
`--reuse-v2-from <run>` takes over descriptions already paid for instead of
calling the model again.

**The competency chain drops sentences that are not requirements, and that
filter is unvalidated.** `governance/zoning.py` labels sentences from section
headers (weak labels), trains a small classifier on them and removes company,
benefit and legal sentences before anything is counted. It was built because
about 10 of 24 sampled "valid" families were things like "support you and your
family" or "apply early" (EXP-030). Its cut-off came from the developer reading
64 sentences. Do not quote its agreement figure as accuracy; when the gold
standard exists, measure how many expert spans sit in dropped sentences.
`--no-zone-filter` is the ablation. Related: a phrase that names its work with
a noun made from a verb ("web application development", "pengujian") is read
through `competency/verbs/deverbal_nouns_v1.tsv` and takes that verb's level;
"perform" in front of it is wording (the researcher's correction). And with
`--all`, families whose centres are at cosine >= 0.90 are merged, because the
clusterer splits one piece of work by its wording.

**SKKNI titles show how a statement is FORMULATED; the competencies come from
the market.** The promotors defined a competency on 2026-11-20 as a measurable
combination of knowledge, skills, abilities and behaviours, and called "develop
web application" a duty until it states how well, which standards and what
outcome. The researcher then corrected the developer: SKKNI unit titles are
examples of how to formulate a statement, not a format to translate the
market's competencies into; SKKNI is part of the problem (the programming
standard is from 2016). `competency/formulate.py` writes one SKKNI-style line
per competency from its own evidence (`competency/qualifiers.py`), uses SKKNI
wording only where a competency IS an SKKNI unit on the same level, and refuses
tools, numbers and standards the postings do not give. Do not bring back the
whole-unit writer of EXP-031 (elements, KUK): that was the misreading. A
"same as SKKNI" call keeps SKKNI wording only when a second model of another
family (`CONFIRM_MODEL`, Qwen) agrees: alone, the writer (GPT) was wrong in 4
of its 10 "same" calls (EXP-032). Do not drop that step to save a call; a
missed "same" costs nothing, a wrong one replaces the market's wording. The
reference judges of EXP-032 were Claude subagents, not experts. An
OpenRouter cost read from the account lags: read it again later.

**Competencies are built from SKILL_HARD only, and KKNI never decides anything.**
Both are the researcher's decisions of 2026-11-19. Tools and technology are
attached to a competency by sharing a sentence; soft skills, work behaviours
and KNOWLEDGE-tagged terms by sharing postings, each with a share and a lift
(that last list is the developer's reading). Do not put
KNOWLEDGE or SKILL_SOFT back into `GroupingConfig.competency_tags` as the
default, and do not let the KKNI level into `contract.decide` or
`delta.priority_order`: tests hold both. The good statement writer is the
pre-approval generator, reached through `competency/v2_generator.py` (paid,
capped); the short rule statements are the free fallback. On the whole corpus
the grouping runs in two stages (`core_min_postings`, `attach_threshold`); a
flat attach threshold and an audit that counted attached phrases both went
wrong once (experiment log, EXP-027).

**If you are new here, read `docs/HANDOFF.md` first.** It says where every part
stands, what is decided, what is open and what was in progress when the last
session ended. `docs/experiments/EXPERIMENT_LOG.md` has every run with its
numbers. Keep both current: they are what lets the next person, or the next
assistant, continue without the conversation that produced the code.

**The dashboards in `dashboard_v3/` must stay unable to collect, pay or touch
the gold standard.** A test reads their source for it. Publishing a profile is
a draft unless a person ticks "sah untuk dirilis"; never make that the default.
`dashboard/` and `dashboard_v2/` are the pre-approval dashboards: `dashboard/`
still imports `pipeline_orchestrator` (see the first trap of this section).

**`python -m acquisition.monthly run --go` really collects.** Without `--go` it
is a preview and touches no network. It adds a schedule on top of
`acquisition.run_batch` and no fetching code of its own (a test checks this). Do
not run it with `--go` as a side effect of something else, and do not raise the
limits in `configs/monthly_roster.json` (25 queries per session and 24 h rest
for detail-fetch platforms): they are the standing constraints of §6.

**Bare-`python` root launchers are not the pattern; `run.ps1` is.** The last one,
`plot_aggregated.bat`, was moved to `legacy/` on 2026-11-19 with its two
siblings: it called bare `python` (no ML packages since the 2026-05-12 venv
migration) and `aggregate_results.py`, which was already in `legacy/`.

**A clean-up audit on 2026-11-19 found almost nothing else to move, and that is
the correct result.** Of 250 tracked code files, 10 were pre-approval, unnamed
in the plan and unreachable from live code; after reading them only
`plot_aggregated.bat` was certainly dead. The roughly twenty old-looking scripts
at the repo root are still launched by `pipeline_orchestrator.py`, which
`dashboard/app.py` imports (first trap of this section). They can go only when
the researcher agrees that `dashboard/` and `dashboard_v2/` are retired in
favour of `dashboard_v3/`. **She decided on 2026-11-19 to keep the old
dashboards** ("might still be useful"), so the old orchestrator and everything
it launches stay LIVE. Do not quarantine them.
Of the two entry-point scripts that were unsure: `ingest_future_domains.py`
stays, because it is the generator of `future_domains.csv` (plan-named, and read
by `competency/temporal.py`) and holds its source citations;
`skill_trend_holdout_validation.py` went to `legacy/` after the researcher
confirmed she does not run it (it reads the pre-approval `results/` files).

**No CI exists.** No `.github/`, no Makefile. Nothing is "run by CI", so that
can never be the reason a file is live.

---

## 5. The live map

Authoritative detail is in `docs/IMPLEMENTATION_PLAN.md` §0. Orientation only:

| Layer | Modules | Status |
|---|---|---|
| 0 — acquisition | `acquisition/` (adapters, batch store, cadence; `monthly` = the monthly round as one command, driven by `configs/monthly_roster.json`), `job_scraping/` | NEW / ADAPT |
| 1 — corpus governance | `preprocess_jobs_pipeline.py`, `scripts/audit_corpus_lineage.py`, `governance/` (`rules.py`, `ladder.py`, `zoning.py` = duty/requirement sentence filter, `config_registry.py`), `it_relevance_filter.py` | ADAPT / NEW |
| 2 — gold standard | **`gold_standard/`** (pool, sampling, splits, IAA, seal; and the post-annotation kit: `after_annotation` = one command, `gate`, `calibration`, `adjudicate`, and `materialise` = seal the test labels and export train/dev for the models — see `docs/RUNBOOK_after_annotation.md`), `annotation_server/`, `gold_labeling_ui/` | **current work** |
| 2b — silver labels | `labeling/` (verifier, alignment, consensus, guideline-only prompt pack, OpenRouter runner with a hard spending cap, a SYNTHETIC guideline-conformance set and a SkillSpan probe for measuring prompt changes; output goes to the gitignored `DATA/labeling/`, never under `DATA/gold_standard/`), `gold_standard/metrics.py` (the one shared evaluation engine) | NEW, **current work** |
| 3 — extraction | **`fivetag/`** (every condition of proposal §3.3.1–3.3.3 behind one interface: E0 rules + versioned lexicon, E1/E2 encoder with layered BIO and the DAPT script, E3 GLiNER adapter, E4/E5 LLM with retrieval, E6 router, D1–D3 classifiers, `benchmark`; see `fivetag/README.md`), `baseline_versions/` (pre-approval benchmark material), `extraction_typing.py` (v1 classifier, still feeds the v1 competency stages), `llm_client_router.py` | NEW, **current work** / ADAPT |
| 4–5 — competency | **`competency/`** (five-tag evidence table → groups with the six validity statuses → grounded statement → KKNI / okupasi → status contract; one command, `python -m competency.pipeline`; plus `temporal` (windows, trend, volatility, persistence), `coverage` (four statuses against SKKNI or an uploaded curriculum), `delta` (five recommendation kinds), `endtoend` (P1–P3 and the two ablations), `structure` (the verbs and the knowledge of a competency), `termmap` (cluster map of knowledge terms and technologies); see `competency/README.md`). It REUSES `clustering/`, `pipeline_scientific_audits.py`, `competency_kkni_labeller.py`, `occupation_mapper.py`. `competency_generator_v2.py` and `competency_evidence_repair.py` are the pre-approval chain, still driven by `scripts/run_full_v2_pipeline.py` for the old dashboard | NEW, **current work** / REUSE |
| 6 — labels & refs | `competency_kkni_labeller.py`, `occupation_mapper.py`, `kkni.py`, `DATA/skema_okupasi_*`, `DATA/spektrum_keahlian/` | ADAPT |
| 7 — trends | `skill_time_trend_analysis.py`, `future_weight_mapping.py`, `future_domains.csv` | ADAPT |
| 8 — delivery | **`dashboard_v3/`** (`public.py` reads a published profile, shows the cluster map (`termmap_view.py`) and audits an uploaded curriculum; `admin.py` shows runs, works the two human review queues and publishes profiles; both read-only towards collection, paid models and the gold standard), `competency/profile.py` (versioned profile), `competency/figures.py` (every chart, from the output files). Pre-approval and still wired to the old chain: `scripts/build_*_package.py`, `dashboard_v2/app.py`, `dashboard/`, `expert_review.py` | NEW, **current work** / ADAPT |
| chassis | `scripts/run_full_v2_pipeline.py` | ADAPT |

`gold_standard/` is the most current and best-tested part of the repo
(268 passing tests, a group-aware splitter with 17 fail-closed invariants, and
two pre-registrations). Treat it as the reference for how new work should look.

---

## 6. Data policy — the corpus is local only

The scraped job-posting corpus was **purged from all git history on 2026-11-17**
and force-pushed. The repo is **public**. Never commit:

- `DATA/preprocessing/`, `DATA/skillspan_ours/`, `DATA/samples/jobs_sample.csv`
- `DATA/fivetag/` (train/dev rows exported from the gold standard, LLM benchmark
  runs, checkpoints), `DATA/labeling/` (ensemble runs), `DATA/competency/`
  (evidence packages) and `DATA/taxonomies/` (raw O*NET / ESCO downloads)
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

- `TODO.md` at the repo root is the researcher's running to-do list for the whole
  study (local, gitignored). When a task is finished, added or changed, update
  that file: tick the box, move finished items to "Done", and change the "Last
  updated" date. Keep its wording plain; the reader is the researcher, not a
  developer.
- Commit messages: conventional prefixes (`feat(scope):`, `fix:`, `chore:`,
  `docs:`). Say *why*, and state what was verified rather than asserting it
  works.
- Pre-registrations live in `docs/experiments/PREREG_*.md`. Amend the
  **evidence** section, never the registered decision, and date the
  registration before results exist.
- Adversarial review findings go in `docs/experiments/*_findings.md` with a
  resolution log. Findings that stay open are listed as open.
- **`docs/CONFIG_REGISTRY.md` is generated, never edited.** It lists every
  version, prompt hash, dictionary hash and threshold with its status
  (registered / decided / by design / starting value). After changing any of
  them run `python -m governance.config_registry`; `tests/test_config_registry.py`
  fails until you do.
- **Every experiment gets an entry in `docs/experiments/EXPERIMENT_LOG.md` on
  the day it is run** (researcher's instruction, 2026-11-19: she writes the
  dissertation from it). An experiment is anything that produces a number
  someone might quote: a model run, a prompt comparison, a rehearsal, a trial
  that cost credit. The entry gives the question, data, settings and versions,
  the numbers copied from the output files, the cost, the file paths, the
  status (DEVELOPMENT / REHEARSAL / ENGINEERING / CONFIRMATORY) and the limits.
  Never renumber or delete an entry. A run that is not in the log should be
  treated as not having happened.
- Report measured numbers, not expectations. If a suite fails, say so with the
  output.
