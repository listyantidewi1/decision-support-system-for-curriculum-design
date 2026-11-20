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
| Competency layers | `competency/` | Built. **Competencies are built from hard skills only; tools, knowledge, soft skills and work behaviours are attached to each one** (researcher's decision, 2026-11-19). Two statement writers: rules (free) and the earlier LLM generator through `competency/v2_generator.py` (EXP-026). Whole-corpus runs: EXP-027 to EXP-030. Current: `e0_pool_all_v8` (EXP-033: fragments held back, families with the same statement merged), with all 599 review candidates formulated SKKNI-style in `formulated.json` (version 5, EXP-034: a fitting standard or quality is written; competencies that are not work or mix works are marked, 265 of 599; checked by independent judges on a fresh sample before the full run); draft profile `2026-0-draf7`. See `competency/README.md`. |
| Time analysis | `competency/temporal.py` | Built; run once on the whole pool (experiment log, EXP-023). Thresholds and weight formula are starting values. |
| Coverage audit, delta recommendations | `competency/coverage.py`, `delta.py` | Built and tested. Coverage against SKKNI run once (EXP-024). Delta needs an uploaded curriculum; none has been used yet. |
| Versioned profile | `competency/profile.py` | Built. One DRAFT exists locally: `DATA/competency/profiles/drafts/2026-0-draf1/`. No release exists. |
| End-to-end pipelines P1–P3 | `competency/endtoend.py` | Built and tested with a fake model; rehearsed once on 12 postings (EXP-025). Not a comparison yet: sample too small, extractors unmeasured. |
| Figures | `competency/figures.py` | Built. Charts of the development experiments are committed under `docs/experiments/figures/`; charts of runs stay with the run under `DATA/`. |
| Dashboards | `dashboard_v3/public.py`, `dashboard_v3/admin.py`, `dashboard_v3/termmap_view.py` (the cluster map tab) | Built, tested headless. Not deployed. `dashboard/` and `dashboard_v2/` are the pre-approval versions, still wired to the old chain. |
| Configuration registry | `governance/config_registry.py` → `docs/CONFIG_REGISTRY.md` | Generated from the code; a test fails if it is out of date. 26 settings are still "starting value": they are the list of things to agree with the promotor or the experts. |
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

Added after that list, same night and after midnight (20 November):

- [x] Configuration registry — `governance/config_registry.py`, `docs/CONFIG_REGISTRY.md`
- [x] Clean-up audit; two files moved to `legacy/` — `CLAUDE.md` §4
- [x] Bridge to the earlier LLM generator — `competency/v2_generator.py`, EXP-026
- [x] Competencies from hard skills only, with tools, knowledge, soft skills and work behaviours attached — `competency/grouping.py`, EXP-027
- [x] KKNI made information only, held by a test — `competency/contract.py`
- [x] Two-stage grouping for the whole corpus, and rank stability between windows — `competency/grouping.py`, `competency/temporal.py`, EXP-027

- [x] A competency as verb + knowledge, statement without tool names — `competency/structure.py`, `competency/statement.py`, EXP-028
- [x] Cluster map of knowledge terms and technologies, as a dashboard tab and a still figure — `competency/termmap.py`, `dashboard_v3/termmap_view.py`, `competency/figures.py`
- [x] The earlier generator run on the 30 most frequent hard-skill competencies of the whole corpus, cost measured from the account — EXP-028

The account balance can be read (numbers only) through OpenRouter's `/credits`
endpoint with the key the code already loads; that is how EXP-028 was costed.
The old generator itself still reports no usage.

- [x] One competency per cognitive level of a family; level filter and relatives in the dashboard — `competency/structure.py`, `dashboard_v3/public.py`, EXP-029
- [x] Verb table as a sourced file, 203 verbs — `competency/verbs/`, EXP-029
- [x] Generator descriptions for more families; they describe the family, not one level — EXP-029

- [x] Clustering check: families are tight but fragmented, and about 10 of 24 sampled "valid" families were not competencies — EXP-030
- [x] Sentence filter (duty and requirement sentences only) — `governance/zoning.py`, `tests/test_zoning.py`, EXP-030
- [x] Merge of near-duplicate families — `competency/grouping.py` `merge_similar`, EXP-030
- [x] Verbs read from nouns; light verbs — `competency/statement.py`, `competency/structure.py` `Reader`, EXP-030
- [x] Intervals, pooled level mix, rank ranges, family-level rank stability — `competency/smoothing.py`, EXP-030
- [x] SKKNI documents read in full (all 29 in `DATA/master_skkni`, 2015–2025) — `competency/skkni.py`
- [x] ~~Competency units in the full SKKNI form (EXP-031)~~ — withdrawn the same day: the researcher meant the formulation of statements, not whole units
- [x] Statements formulated like SKKNI titles from the market evidence, with SKKNI wording where a competency is an SKKNI unit on the same level; shown in the public dashboard — `competency/formulate.py`, EXP-032
- [x] Fragments held back; families with the same statement on the same level merged (only through statements that name specific work) — `competency/statement.py` `is_fragment`, `competency/structure.py` `merge_same_statement`, EXP-033
- [x] Formulation checked by independent judges (3 blind SKKNI-match judges, 2 statement reviewers), revised to version 4.3 (a second model must confirm a "same as SKKNI"), and checked again on 40 held-out competencies — EXP-032
- [x] All 599 "kandidat tinjauan" competencies of run `e0_pool_all_v8` formulated with version 5 (558 accepted; $2.48) — EXP-034
- [x] Decided by the researcher on 2026-11-21 and built as version 5: competencies that are not work or mix works are MARKED (not left out); a fitting standard or quality is written — EXP-034
- [ ] Open from EXP-034: (1) the writer now often attaches a quality to the wrong noun (9 of 34 checked) — a next version should fix it and be checked on a fresh sample; (2) 44% of the review candidates are marked not work or mixed: improve the clustering and the sentence filter upstream; (3) the marks and the "same as SKKNI" calls are checked against model judges only; (4) "same" is found rarely (1 of 7 on the check sample)
- [ ] Repair the two defects EXP-030 found: fragments (spans cut before their object) and the same statement in two families
- [ ] A cognitive-level classifier that reads the whole phrase in context, checked against phrases labelled by experts. Waits for the pilot and for expert labels; nothing is built.
- [ ] Education required by the posting (SMK, D3, S1) per competency. Not built.
- [ ] Reweighting time windows to a common mix of platforms and countries. Not built.

Known faults of the attached lists, not yet repaired (EXP-027, "What is visibly
weak"): lift is not comparable across languages; spelling variants of one soft
skill are separate lines; a rule statement can be a fragment of one posting.

What comes next, in the order I would take it:

1. Put the annotation page online once the researcher has the Fly.io and
   Streamlit accounts (steps in `annotation_server/README.md`, "Deploy").
2. Run the earlier generator on the hard-skill-only groups of the whole corpus
   (EXP-026 used the older grouping) and have the researcher read the result.
   It costs credit: needs her go-ahead and `--max-usd`.
3. A better KKNI step. The generator's own proposal is not it (level 5 for all
   13 statements in EXP-026). KKNI is information only, so this blocks nothing.
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
| Competencies come from SKILL_HARD. Each competency lists the tools and technology mostly used for it, and the soft skills and work behaviours expected with it | 2026-11-19 | `competency/__init__.py`, `competency/README.md` |
| A competency is a verb (placed on the taxonomy) plus a knowledge. The statement names no tool: "Perform programming", not "Perform programming using Python, PHP and JavaScript" | 2026-11-20 | `competency/structure.py`, `competency/statement.py` (tests hold both) |
| The cognitive level of a competency is read from the verbs the postings use (C1–C6), never from a default verb the method supplied | 2026-11-20 | `competency/structure.py` (a test holds it) |
| A group is a family; each cognitive level in it is a competency of its own, with its own statement and its own number of postings ("apply web programming" and "develop a website" are two competencies). No single level stands for a family | 2026-11-20 | `competency/structure.py` `split_by_level` |
| The verb-to-level table is a written, sourced file, to be reviewed by the researcher | 2026-11-20 | `competency/verbs/bloom_verbs_v2.tsv`, `SOURCES.md` |
| A phrase that names its work with a noun made from a verb takes that verb and its level: "Perform web application development" is "Develop web applications", C6 | 2026-11-20 | `competency/statement.py` `read_phrase`, `competency/verbs/deverbal_nouns_v1.tsv` |
| The promotors' definition of a competency: a measurable combination of knowledge, skills, abilities and behaviours; a task title ("develop web application") is a duty until it states how well, which standards and what outcome | 2026-11-20 | `competency/formulate.py`, `competency/qualifiers.py` |
| The researcher's correction: SKKNI unit titles are examples of how a competency statement is FORMULATED. The competencies come from the market (SKKNI is part of the problem; the programming standard is from 2016). Where a competency matches an SKKNI unit, the SKKNI wording may be used. NOT: translating competencies into whole SKKNI units (EXP-031 did that; withdrawn) | 2026-11-20 | `competency/formulate.py`, EXP-032 |
| Build the four improvements the developer recommended: read verbs from nouns, intervals and pooled shares, a filter for sentences that are not requirements, (later, after the pilot) a level classifier that reads context | 2026-11-20 | this file §6; EXP-030 |
| KKNI is additional information for the reader; it never enters the status or the order | 2026-11-19 | `competency/contract.py` (a test holds it) |
| The earlier LLM generator (`competency_generator_v2`) is the statement writer to build on, not the short rule statements | 2026-11-19 | `competency/v2_generator.py`, EXP-026 |
| Kitalulus and Kemnaker are in the monthly round (promotor agreed) | 2026-11-19 | `docs/corpus/SOURCE_AUDIT.md` K8, K10 |
| Training runs on Kaggle | 2026-11-19 | `fivetag/README.md` |
| Every experiment is logged | 2026-11-19 | `CLAUDE.md` §8 |
| The pre-approval dashboards (`dashboard/`, `dashboard_v2/`) are kept; the old orchestrator and its scripts therefore stay live | 2026-11-19 | `CLAUDE.md` §4 |

## 8. Decisions still open (the researcher's)

- The verb table (`competency/verbs/bloom_verbs_v2.tsv`, `verbs-v2-sourced-unreviewed`): the researcher has not reviewed it. The Indonesian rows are the developer's translations; a list her institution uses should replace them.
- Which cognitive levels suit which kind of institution (SMK, D3, S1). The dashboard only offers a filter; nothing maps a level to an institution.
- Whether the formulated statements of EXP-032 are what the promotors mean; who supplies measurable criteria the postings do not give.
- Whether SKKNI wording should also be used when the SKKNI title's verb is on another level than the postings ask (the developer's rule says no and shows both levels).
- **Whether the sentence filter (`governance/zoning.py`) and the later level classifier fit the approved plan or need the promotor's agreement.** The developer raised this before building; the researcher said to go ahead. The plan names a sentence relevance filter (`it_relevance_filter.py`, ADAPT); the zone filter is a cheap, weakly supervised relative of it. Not validated against people.
- Postings versus employers as the count of demand, the two contract thresholds, the verb table (`competency/README.md`).
- Whether a generator statement may be judged by the generator's own evidence filter (built that way; `docs/CONFIG_REGISTRY.md`).
- Which model is the registered adjudicator (provisional: Qwen 3.8 Flash; confirm on pilot gold).
- How the main annotation of 1,200 is paid for.
- Budget for labelling the whole corpus with the LLM ensemble.
- Whether the SkillSpan-derived files may stay in the public repository.
- Ethics position on storing annotators' work on a server in Singapore.
