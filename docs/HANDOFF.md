# Handover — where the project stands and how to continue

Written for whoever picks this work up next: a person, or an AI assistant with
no memory of the earlier sessions. Read this, then `CLAUDE.md` (the rules and
traps of the repository), then `TODO.md` if it is present (the researcher's own
task list; it is local and not in git).

Last updated: **2026-11-19**. Update the date and the "state" tables whenever a
piece of work lands.

## 1. What this is

Doctoral research by Listyanti Dewi Astuti (Pendidikan Kejuruan, Universitas
Negeri Malang): online job postings are analysed with NLP and LLMs to inform the
software-engineering (RPL) curriculum of Indonesian vocational schools (SMK).

- The approved proposal: `docs/proposal/PDF/proposal_coworking/Proposal_Disertasi_04102026_4D_final.docx` (approved 2026-10-04).
- The only in-repo specification: `docs/IMPLEMENTATION_PLAN.md`.
- The annotation guideline: `docs/proposal/PDF/proposal_coworking/instruments/B_Paket_Design_Korpus_dan_Anotasi_v3.0.docx` (code AG-RPL-LIMA-v3.0). Five tags: SH hard skill, SS soft skill, KN knowledge, TT tool/technology, WB work behaviour.
- The researcher is not a software developer. Write to her in plain language,
  lead with the outcome, and say what is measured and what is not.

## 2. The one thing everything waits for

**The experts' gold standard does not exist yet.** Three experts must annotate
100 pilot sentences, then (after a go/no-go decision registered in advance)
1,200 sentences. Until then no number in this repository is a finding of the
study. Every run so far is DEVELOPMENT, REHEARSAL or ENGINEERING; see
`docs/experiments/EXPERIMENT_LOG.md`.

What the researcher is doing now (as of 2026-11-19): practice run of the
annotation page; recruiting the three annotators; opening the Fly.io and
Streamlit Community Cloud accounts so the page can go online.

## 3. State of each part

| Part | Where | State |
|---|---|---|
| Corpus collection | `acquisition/` | Running monthly. One command: `python -m acquisition.monthly status` / `run` (preview) / `run --go` (collects). Next Indonesian round due 2026-12-06. **Never run `--go` as a side effect.** |
| Corpus governance, ladder | `governance/`, `preprocess_jobs_pipeline.py` | In use. Indonesia: 5,718 eligible postings against a floor of 10,000. |
| Sentence pool, samples | `gold_standard/pool.py`, `sample.py` | Pool 662,248 sentences. `pilot_100` and `practice_10` drawn. `main_1200` NOT drawn (wait for the December round). |
| Annotation page + collector | `gold_standard/app.py`, `annotation_server/` | Works locally; runs from Windows Task Scheduler (`run_annotation_stack.ps1`). Not yet online. Hosting files are ready (`fly.toml`, `annotation_server/Dockerfile`). |
| After annotation | `gold_standard/after_annotation.py`, `gate.py`, `calibration.py`, `adjudicate.py`, `materialise.py` | Built and rehearsed. Runbook: `docs/RUNBOOK_after_annotation.md`. |
| Splits and sealed test | `gold_standard/split.py`, `seal.py` | Built, heavily tested. Not run on real data (no main sample yet). |
| Multi-LLM labelling | `labeling/` | Built. Prompt v0 is the default. Ensemble already run on `pilot_100` and stored blind. `labeling.versus_gold` compares it with the gold when that exists. |
| Extraction models E0–E6, D1–D3 | `fivetag/` | All built; see `fivetag/README.md`. E1 training rehearsed on Kaggle. Router thresholds untuned. |
| D1/D2 conflicts | `fivetag/conflict_routing.py` | Risk rule and acceptance rule registered (`docs/experiments/PREREG_d3_conflict_routing.md`). Adjudicator model not registered; Qwen 3.8 Flash is provisional. |
| Competency layers | `competency/` | Built; one engineering run. See `competency/README.md`. |
| Time analysis | `competency/temporal.py` | Built; run once on the whole pool (experiment log, EXP-023). Thresholds and weight formula are starting values. |
| Coverage audit, delta recommendations | `competency/coverage.py`, `delta.py` | Built and tested. Coverage against SKKNI run once (EXP-024). Delta needs an uploaded curriculum; none has been used yet. |
| Versioned profile | `competency/profile.py` | Built. One DRAFT exists locally: `DATA/competency/profiles/drafts/2026-0-draf1/`. No release exists. |
| End-to-end pipelines P1–P3 | `competency/endtoend.py` | Built and tested with a fake model; rehearsed once on 12 postings (EXP-025). Not a comparison yet: sample too small, extractors unmeasured. |
| Figures | `competency/figures.py` | Built. Charts of the development experiments are committed under `docs/experiments/figures/`; charts of runs stay with the run under `DATA/`. |
| Dashboards | `dashboard_v3/public.py`, `dashboard_v3/admin.py` | Built, tested headless. Not deployed. `dashboard/` and `dashboard_v2/` are the pre-approval versions, still wired to the old chain. |
| Dictionaries | `fivetag/lexicons/v1`, `v2` (+O*NET), `v3` (+ESCO) | `fivetag` defaults to v1; the competency chain uses v2; v3 is built but not used by default (EXP-021). |

## 4. How to run and check things

Two Python environments, both inside the repository (details in `CLAUDE.md` §2):

```powershell
.venv-ml\Scripts\python.exe tests\test_fivetag.py        # everything except the collector
.venv\Scripts\python.exe    tests\test_annotation_server.py
```

There is no pytest. Each test file is run directly and prints `N/N passed`.
The full list with expected counts is in `CLAUDE.md` §3. Run the suites of any
part you touch before and after touching it.

Commands that cost nothing and show the system working:

```powershell
.venv-ml\Scripts\python.exe -m fivetag.evaluate dev
.venv-ml\Scripts\python.exe -m fivetag.benchmark synthetic
.venv-ml\Scripts\python.exe -m competency.pipeline e0 --n-sentences 20000 --tag check
.venv-ml\Scripts\python.exe -m acquisition.monthly status
```

Facts about this computer that are not in the code:

- Model downloads go to `D:\hf_cache` (`HF_HOME`), installer cache to `D:\temp`. Drive C is nearly full; keep large files on D.
- Operational state of the annotation services (logs, backups) lives outside the repository in `D:\Projects\_skill-extraction-ops\`.
- A full backup of the git history from before the corpus purge is at `D:\Projects\_skill-extraction-backup-2026-11-17\`. Do not delete it.
- GPU training happens on Kaggle (free T4). Steps: `fivetag/README.md`, "Rehearsal before the gold exists".
- API keys are in `api_keys/` (gitignored). Never print them, never commit them. LLM calls go through OpenRouter only.

## 5. Rules that must not be broken

Full text in `CLAUDE.md`; the ones with consequences that cannot be undone:

1. The scraped corpus is **local only**. The repository is public. Nothing under `DATA/` is committed; no corpus sentence or posting identifier goes into code, tests or documents. Scan before every commit.
2. Annotators stay **blind** to model output. Nothing from any model is written under `DATA/gold_standard/`.
3. The sealed test split is opened **once**, for the final configuration. No tuning, comparison or example selection on it.
4. Registered rules (`docs/experiments/PREREG_*.md`) are not edited after data exist; only their evidence section is amended.
5. Collection: no detection evasion, respect robots.txt, 24 h rest and at most 25 queries per session on detail-fetch platforms, a new posting source needs an audit and the promotor's sign-off recorded in `docs/corpus/SOURCE_AUDIT.md`.
6. No paid model call without the researcher's go-ahead and a hard spending cap.
7. Commit and push only when the researcher says so.
8. Every experiment is logged in `docs/experiments/EXPERIMENT_LOG.md` the day it runs.

## 6. Work in progress (session of 2026-11-19, evening)

The researcher asked for the following, in this order. Tick each line here as it
lands, with the file that holds it.

- [x] ESCO skills fetched through the official ESCO API and added as lexicon v3 — `fivetag/build_lexicon_v3.py`, EXP-021
- [x] Repair: KKNI step — `competency/mapping.py`; every statement now gets a level, but only 1 of 445 is confident (EXP-024). **Still weak: needs another method or expert placement.**
- [x] Repair: the LLM statement writer may name at most three technologies — `competency/statement.py`. Not re-run on real groups since the change.
- [x] Repair: the verifier's two lost spans explained (repeated text without an occurrence number; about 1% of spans) — EXP-022. Verifier unchanged; whether to change it is the researcher's decision.
- [x] Time analysis — `competency/temporal.py`, EXP-023
- [x] Gap audit, four coverage statuses — `competency/coverage.py`, EXP-024
- [x] Delta recommendations, five kinds — `competency/delta.py` (tested; not yet run on a real curriculum)
- [x] Versioned profile with change log — `competency/profile.py`
- [x] End-to-end pipelines P1, P2, P3 and the two ablations — `competency/endtoend.py`, EXP-025
- [x] Admin dashboard and public dashboard — `dashboard_v3/`
- [x] Figures (asked for during the session) — `competency/figures.py`

What comes next, in the order I would take it:

1. Put the annotation page online once the researcher has the Fly.io and
   Streamlit accounts (steps in `annotation_server/README.md`, "Deploy").
2. A better KKNI step (see the repair above).
3. Re-run the LLM statement trial with the three-technology limit and log it.
4. Try the delta recommendations on a real SMK RPL curriculum document.
5. A larger P1–P3 rehearsal when credit allows; then the temporal-stability and
   repeated-run measurements the rehearsal could not make.
6. Everything that waits for the gold standard: tune the router on dev, choose
   the prompt and the adjudicator, train E1/E2 on Kaggle, the E0–E6 table.

If this list is partly ticked and the session ended, continue from the first
unticked line. Read `competency/README.md` and `fivetag/README.md` first; they
say what each layer assumes.

## 7. Decisions already taken (do not reopen without the researcher)

| Decision | Date | Where recorded |
|---|---|---|
| Pilot go/no-go thresholds | 2026-11-19 | `docs/experiments/PREREG_pilot_gate.md` |
| Overlapping spans allowed when the tags differ (guideline 6.9) | 2026-11-18 | `docs/IMPLEMENTATION_PLAN.md` |
| Two-of-three LLM agreement not accepted automatically until checked on gold | 2026-11-18 | `labeling/consensus.py` |
| A verb governing a list of nouns: the nouns are KNOWLEDGE; activity nouns ("unit testing") are SKILL_HARD | 2026-11-18 | `labeling/conformance.py` |
| Span boundary conventions C1–C6 | 2026-11-18/19 | `labeling/skillspan_guideline.py` |
| Prompt v0 stays the default (v2 not shown better) | 2026-11-19 | experiment log, EXP-006 |
| One tagging layer per tag in the encoder | 2026-11-19 | `fivetag/README.md` |
| ESCO and O*NET lists join the dictionary | 2026-11-19 | `fivetag/README.md` |
| Risk rule for D1/D2 conflicts and its acceptance thresholds (promotor agreed) | 2026-11-19 | `docs/experiments/PREREG_d3_conflict_routing.md` |
| Kitalulus and Kemnaker are in the monthly round (promotor agreed) | 2026-11-19 | `docs/corpus/SOURCE_AUDIT.md` K8, K10 |
| Training runs on Kaggle | 2026-11-19 | `fivetag/README.md` |
| Every experiment is logged | 2026-11-19 | `CLAUDE.md` §8 |

## 8. Decisions still open (the researcher's)

- The role of each tag in the competency layer, postings versus employers as the count of demand, the two contract thresholds, the verb table (`competency/README.md`).
- Which model is the registered adjudicator (provisional: Qwen 3.8 Flash; confirm on pilot gold).
- How the main annotation of 1,200 is paid for.
- Budget for labelling the whole corpus with the LLM ensemble.
- Whether the SkillSpan-derived files may stay in the public repository.
- Ethics position on storing annotators' work on a server in Singapore.
