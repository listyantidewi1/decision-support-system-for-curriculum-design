# Implementation Plan — Kerangka Kerja Analisis Kebutuhan Kompetensi RPL (Proposal 2026-10-04, Model 4D)

Status: living plan. Source of truth = `docs/proposal/Proposal_Disertasi_04102026_4D_final.pdf` (approved).
Timeline anchor: M1 ≈ October 2026 → M12 ≈ September 2027 (Tabel 3.15).

---

## 0. Asset inventory — what the repo already gives us

Legend: **REUSE** (works, minor config) · **ADAPT** (solid base, needs rework) · **NEW** (build from scratch).

| Proposal component (subbab) | Existing asset | Status |
|---|---|---|
| OJP acquisition (3.4.1) | `job_scraping/` — JobSpy scrapers (Indeed/LinkedIn/ZipRecruiter/Google), IDN + 7-country English | ADAPT — see Workstream A |
| Corpus governance, lineage, denominators (layer 1; 3.6.1) | `preprocess_jobs_pipeline.py` (clean, lang-detect, dedup, sentence split, relevance + rescue), `scripts/audit_corpus_lineage.py` | ADAPT (add country strata, coverage gate, N_raw→N_eligible ladder, near-dup, provenance table) |
| Domain filter Tabel 3.9 | `it_relevance_filter.py`, `pipeline_it_filter.py` | ADAPT (function-based RPL rules, okupasi-berbatas rules) |
| Five-tag gold standard (3.4.2) | `gold_labeling_ui/` (labeling UI), `DATA/skillspan_ours/` (BIO corpus infra), `expert_review.py` | ADAPT (span UI for 5 tags, stratified sampler, IAA module, adjudication log, group-aware split) |
| Multi-LLM ensemble labeling (3.4.3) | `llm_client_router.py` (OpenRouter multi-provider), schema validation in `competency_generator_v2.py` | NEW package, reusing router + verifier ideas |
| Benchmark E0–E6 (3.3.3) | `api_zero_shot/` eval harness (E4 base), Skill-LLM Kaggle training workflow (E1/E2 base), `evaluate_extraction.py` | ADAPT + NEW (E3 GLiNER, E5 retrieval+verifier, E6 router) |
| Deterministic typing D1–D3 (3.3.1) | `extraction_typing.py` ≈ D1 (lexicon–regex–priority) | ADAPT D1; NEW D2 (TACD), D3 (consensus) |
| Clustering → competency statement → grounding → dedup (layers 4–5) | `clustering/`, `competency_generator_v2.py`, `competency_evidence_repair.py`, `competency_repair.py`, `pipeline_scientific_audits.py` | REUSE/ADAPT (five-tag evidence, status renames) |
| KKNI / okupasi / official-reference audit (3.3.5) | `competency_kkni_labeller.py`, `occupation_mapper.py`, `kkni.py`, `DATA/master_skkni`, `DATA/skema_okupasi_*`, `DATA/spektrum_keahlian` | ADAPT (+4 coverage statuses: tercakup jelas/sebagian/belum/kandidat pembaruan) |
| Persistence, volatility, trend (3.3.4) | `future_weight_mapping.py` + `future_domains.csv` (WEF prior), BH-FDR trend code from Phase-1 (`skill_time_trends`) | ADAPT prior; NEW rolling windows, volatility, persistence weight, abstraction laddering (ESCO-anchored) |
| Recommendation contract + delta (3.3.6) | `scripts/apply_scientific_sprint1.py` (status/eligibility), `curriculum_coverage.py` | ADAPT (status names → eksploratif/kandidat tinjauan/siap ditelaah formal; NEW delta generator) |
| Versioned profiles + packages (3.3.7) | `scripts/build_evidence_package.py`, `build_dacum_cbc_package.py`, `build_scientific_validation_pack.py` | ADAPT into release builder with version log |
| Dashboards (Gambar 3.3) | `dashboard_v2/app.py` (Streamlit public), `dashboard/` (FastAPI admin) | ADAPT |
| HITL gates H1–H6 (Tabel 3.3) | `expert_review.py`, `feedback_store/`, gold UI | ADAPT into per-gate queues + artefact logs |
| E2E driver, stability/ablation | `scripts/run_full_v2_pipeline.py`, `scripts/build_stability_ablation_pack.py` | ADAPT as chassis for P1–P3 + ablations |
| Expert instruments (3.5) | `docs/proposal/PROMPT_INSTRUMEN_VALIDASI_AHLI_26062026.md` (15 lampiran, five-tag) | ADAPT (Borg&Gall→4D, 24 users, Aiken's V per Tabel 3.1) |
| Translator (bukti lintas bahasa as derived attribute) | `translator.py` | REUSE |

**Repo debt (do first):** critical modules are still untracked/uncommitted since June (`extraction_typing.py`, `competency_repair.py`, `pipeline_scientific_audits.py`, `scripts/apply_scientific_sprint1.py`, WIP in `llm_client_router.py`, `preprocess_jobs_pipeline.py`, `dashboard_v2/app.py`, results cleanup). A fresh checkout ImportErrors. Commit before anything else.

---

## 1. Empirical findings about the existing data (audited 2026-10-20)

- `english_jobs.csv`: **113,281 jobs** — US 42k, UK 21.6k, CA 19k, AU 12k, **SG 10.6k**, IE 5.4k, NZ 2.3k. 99.9% dated, but **84% posted 2026Q1** (scrape month), 10% 2025Q4, long thin tail + garbage dates (1983/1990 → need date-validation rule). Platform: effectively Indeed-only (113k indeed, 190 google).
- `indonesian_jobs_raw.csv`: **2,176 jobs**, Indeed-only, scraped 2026-03. SMK-filtered subset: 287.
- Implication 1 — **retrospective depth is an illusion**: boards only serve what is live (+ a short tail). A 36-month corpus cannot be scraped backwards; it must be **accumulated forward** via repeated batches (each batch = one time snapshot with `first_seen`), plus any licensed/archival source we can legally add. The proposal already anticipates this: temporal analysis is conditional, falls back to exploratory (1.4, 3.3.4).
- Implication 2 — **Indonesia is the binding constraint** (2.2k vs target 10k–20k): Indeed-only and SMK-skewed query terms. Fix: more platforms (Glints, Kalibrr, Jobstreet/SEEK, Karier.mu), broaden terms to the full RPL function family (proposal does NOT restrict to SMK-level postings), monthly batches.
- Implication 3 — OECD ≥5k and ASEAN ≥5k targets are **reachable**: OECD already exceeded in one batch; ASEAN has SG 10.6k, add MY/PH/TH/VN via JobStreet/SEEK + Indeed where available.
- `date_posted` garbage + "12-month" parameter behaving as "currently live" must be documented in the acquisition method notes (Tabel 3.10 `acquisition_method/version`).

---

## 2. Workstream A — Corpus acquisition & governance (M1–M5; 3.4.1, Tabel 3.9/3.10, Gambar 3.7) — **CURRENT PRIORITY**

**A1. Source & legality audit** → `docs/corpus/SOURCE_AUDIT.md`
Per source: access method (API / licensed dataset / public page), ToS position, rate limits, metadata completeness vs Tabel 3.10, decision (use / use-with-notes / drop). JobSpy scrapes Indeed/LinkedIn/Google HTML & private endpoints — LinkedIn ToS explicitly prohibits scraping (recommend **drop LinkedIn**); Indeed/Google public listing pages are defensible for research but must be documented honestly. Evaluate adding: Adzuna API (OECD, free research tier), USAJOBS API, Arbeitnow API, Jobstreet/SEEK endpoints (ID/MY/PH/SG), Glints & Kalibrr public endpoints, Karier.mu. Decision reviewed with promotor (gate for 3.7 ethics section).

**A2. Acquisition harness** → new package `acquisition/`
- Adapters: JobSpy wrapper (keep) + per-platform adapters (Glints, Kalibrr, Jobstreet, APIs).
- Output = **raw immutable layer**: append-only batches `DATA/corpus_raw/batch_<id>/`, one parquet/csv + manifest (`batch_id`, `acquisition_timestamp`, method+version, query terms, filters, platform). **No cleaning at this layer** (current scrapers clean before saving — fix: raw stays raw; cleaning produces derived versions linked by id).
- Full Tabel 3.10 schema: `source_record_id`/hashed URL, posting_date + first_seen/last_seen, country/region, language (detected downstream), `employer_hash`, `description_raw`, education/experience requirement fields, salary when present.
- Broadened IDN term list (full RPL family per 1.4, not SMK-only); keep `query_term/family` provenance.

**A3. Rolling batch schedule** (fills the temporal dimension)
- Ingest existing March-2026 files as **batch-001** (provenance documented post-hoc).
- **batch-002 now (Oct 2026)**: Indonesia all platforms + ASEAN + OECD refresh.
- Then monthly (Indonesia) / bi-monthly (ASEAN, OECD) batches through M12 → ≥7 snapshots spanning ~19 months (+ batch-001) for rolling-window analysis; 36-month claims stay conditional/exploratory as the proposal allows.
- `first_seen`/`last_seen` maintained across batches (same `source_record_id` reappearing = still open, not new demand).

**A4. Governance & audit module** (extends `preprocess_jobs_pipeline.py` + `audit_corpus_lineage.py`)
- Ladder with explicit denominators: N_raw → N_valid_metadata → N_unique (exact + **near-dup** via MinHash/SimHash — new) → N_domain (Tabel 3.9 function-based RPL rules incl. okupasi-berbatas) → N_eligible; duplicates kept in provenance table, never deleted.
- Country/language validation, date-sanity rule (e.g., reject posting_date < 2021 or > acquisition date), per-country **coverage gate report**, representativeness tables (platform × country × quarter) per Gambar 3.7.

**A5. Candidate sentences** — reuse sentence segmentation + relevance screening (+ rescue) → `N_candidate` denominator for Workstream C.

**Deliverables / gate:** source audit doc; harness v1; batch-001 ingested + batch-002 acquired; audit report with all denominators; coverage-gate verdict per stratum. (Feeds Gerbang 1–2.)

---

## 3. Workstream B — Gold standard & instruments (M3–M5; 3.4.2, 3.5)

1. Annotation guideline v1 (five-tag BIO/BIOES + relevance layer, boundary rules, tool-tech vs knowledge rules, adjudication protocol) — seed from Tabel 2.3 + existing instrument prompt content.
2. Stratified sampler (country/region, year, platform, okupasi group, language, sentence length) → pilot 100 → 3-expert calibration → guideline freeze → main 1,200 (3 × 1,200 independent, blind to LLM output).
3. Tooling: assess `gold_labeling_ui` for span annotation; if too far, use Label Studio (free, local) with five-tag config — decision in week 1 of B.
4. IAA module: Fleiss' Kappa / Krippendorff's Alpha (categories), exact span agreement + pairwise span-F1 (boundaries); adjudication log; **gold freeze with version id**; group-aware 60/20/20 split + temporal holdout; oversampling only in train/dev.
5. Instruments: regenerate Lampiran 1–15 from `PROMPT_INSTRUMEN_VALIDASI_AHLI_26062026.md` updated to the final proposal (4D stages, PP wording, 24-user operational test, Aiken's V validation of instruments themselves).

---

## 4. Workstream C — Multi-LLM ensemble labeling (M3–M5; 3.4.3, Gambar 3.8)

New `labeling/` package: shared prompt pack (guideline + few-shot from gold train/dev only) → ≥3 LLMs from different families via `llm_client_router` (e.g., gpt-5.4-mini + a Claude small + a Qwen/Llama; avoid deepseek-v3.2 per 2026-05 token bug) → deterministic output verification (JSON validity, span-exists-in-source with offsets, allowed labels, no overlap; INVALID/UNSUPPORTED log) → span alignment (exact, then token-overlap/IoU threshold tuned on dev) → consensus engine (unanimous / majority / CONFLICT / ABSTAIN) → silver store with consensus ratio + disagreement score + model versions. Held-out gold test opened only after the whole config is frozen. Report automatic-acceptance precision, conflict rate, abstention rate, cost/time per 1,000 sentences.

---

## 5. Workstream D — Benchmarks E0–E6, D1–D3, P1–P3 (M5–M9; 3.3.2–3.3.3)

- **E0** rules/dictionary (from `extraction_typing` lexicons + ESCO/O*NET lists). **E1** multilingual encoder + BIO (XLM-R / IndoBERT; Kaggle GPU workflow from Skill-LLM era). **E2** +DAPT on unlabeled corpus (condition, not assumption). **E3** GLiNER zero/few-shot. **E4** prompted LLM (adapt `api_zero_shot` harness). **E5** +retrieval + schema + deterministic verifier (assemble from C). **E6** full router framework (escalation triggers a–f of 3.3.1).
- **D1** = refactored `extraction_typing.py` (versioned lexicon/regex/priority tables); **D2** TACD (taxonomy anchor + cue-word decision table); **D3** consensus with CONFLICT escalation.
- Metrics engine (shared): strict/partial span-F1, macro-F1 per tag, confusion matrix, Wilson CIs, paired bootstrap + McNemar, grounding metrics, cost/time. One module, reused by B (IAA), C (silver QC), D, and E2E.
- **P1/P2/P3 (+ablations)** end-to-end on identical corpus via parameterized `run_full_v2_pipeline` chassis; decision by a-priori weighted matrix on dev; test + temporal holdout opened once.

---

## 6. Workstream E — Downstream framework (M6–M9; layers 4–5)

Clustering→statement→grounding→dedup: reuse v2 chain, switch evidence model to typed five-tag units. KKNI/okupasi mapping + **official-reference gap audit** (4 statuses). Temporal modules: rolling 6-month windows, share/slope/volatility, persistence weight (corpus trend × WEF prior via `future_domains.csv`), **abstraction laddering** over an ESCO-anchored internal taxonomy (needs ESCO v1.2.1 download). Contract status rename + **delta recommendations** (menambah/memperdalam/meninjau ulang jenjang/mempertahankan/mengurangi penekanan — the last capped at eksploratif without expert review). **Versioned profile builder** (Profil Kompetensi RPL <year>-<n>, changelog, korpus summary, denominators).

## 7. Workstream F — Dashboards & HITL (M7–M10)

Admin dashboard (FastAPI base): run pipeline, audit views, H1–H6 review queues with artefact logs (Tabel 3.3). Public dashboard (`dashboard_v2` rework to Gambar 3.3): filters, KKNI distribution, statement list + evidence drill-down, contract status, curriculum upload → coverage + delta report. SUS + task observation protocol for 24 users (instruments from B).

## 8. Workstream G — Validation & dissemination (M8–M12)

≥3-expert validation rounds (frozen model outputs vs post-HITL reported separately), Delphi 7–9 for siap-ditelaah-formal candidates, robustness (temporal/country holdout, ablations via stability pack), laporan kelulusan per gerbang, profil v1 release, limited dissemination forum, policy brief, artikel.

---

## 9. Key risks & open decisions

1. **Legality/ToS of scraping** (A1) — LinkedIn out; Indeed/Google documented; prefer APIs where available. Needs promotor sign-off. (3.7)
2. **36-month retrospective impossible** → forward accumulation design (A3); temporal claims conditional, exploratory fallback is already in the proposal.
3. **Indonesian volume risk** (2.2k now vs 10k–20k) → multi-platform + broadened terms + monthly batches; report honestly if target missed (proposal allows: perpanjang periode + laporkan).
4. **LLM budget**: ensemble = 3 × N_candidate calls; estimate after A5 (cheap models ≈ tens of USD per 100k sentences; verify before full run).
5. **GPU for E1/E2**: Kaggle workflow (existing) or small paid GPU; decide at D kickoff.
6. **Annotator recruitment** (3 experts, long commitment: 1,300 sentences each) — start recruiting during A.

## 10. Immediate next actions (this week)

1. ✅ 2026-10-20 — Git housekeeping: 4 commits (`8ff2db2b`…`fc5cde3a`), tag `v2-engineering-baseline`, tree clean.
2. ✅ 2026-10-20 — JobSpy smoke test (lib upgraded to 1.4.0): **Indeed works in all three strata** (IDN 13 / MY 20 / US 20 rows, full descriptions, fresh dates); **Google broken** ("no job data returned") → parked; LinkedIn untested pending A1.
3. ✅ 2026-10-20 — `docs/corpus/SOURCE_AUDIT.md` v1.0 draft: 11-source matrix, legal–ethical principles, decisions K1–K4 → **awaiting promotor sign-off**.
4. ✅ 2026-10-20 — `acquisition/` harness v1 (schema, immutable batch store + registry, jobspy adapter, run_batch CLI); batch-001 ingested from legacy CSVs (provenance reconstructed post-hoc, deviations in manifest).
5. ✅ 2026-10-20 — **batch-002** launched: qp-v2-2026-10 plan, 247 queries (60 broadened IDN terms + 17 core × 5 ASEAN + 17 core × 6 OECD), Indeed-only, hours_old=∞ (all live postings).
6. ☐ Encode Tabel 3.9 inclusion/exclusion as versioned config consumed by the governance module.
7. ☐ Jobstreet/Glints/Kalibrr adapter prototypes + Adzuna API key application (per SOURCE_AUDIT K3/K4, after promotor sign-off).
