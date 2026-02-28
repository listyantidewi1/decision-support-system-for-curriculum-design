# 🎓 Future-Aware Hybrid Skill Extraction Pipeline

### **A Curriculum-Intelligence System for Vocational Education (IT / Software / Game Development)**

This repository contains a full **research AI pipeline** designed to:

* Extract **skills** and **knowledge** from job postings
* Fuse outputs from **JobBERT + LLMs**
* Map results to **educational taxonomies**
* Evaluate **curriculum coverage**
* Integrate **future-of-job forecasts** (WEF / McKinsey style)
* Generate **competency statements** for curriculum development
* Support **expert review** via a web UI with multi-reviewer feedback

The system is modular, reproducible, and supports multi-run **experimental aggregation**.

---

# 📖 Documentation

| Document | Purpose |
|----------|---------|
| **README.md** (this file) | Overview, quick start, high-level workflow |
| **[PIPELINE.md](PIPELINE.md)** | Detailed pipeline documentation: phases, data flow, file dependencies, troubleshooting |
| **[CALCULATIONS.md](CALCULATIONS.md)** | Scientific formulas: ranking, voting, weighting, priority scores, FDR, evaluation metrics |
| **[RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md)** | Research questions (RQ1–RQ5), evaluation metrics, gold set design, ablation study |

---

# 🔍 Repository Structure

```
skill-extraction/
│
├── pipeline.py                      # Main hybrid extraction pipeline
├── config.py                        # Global configuration
├── plot_generator.py                # Visual analytics
├── verify_skills.py                 # Skill verification (calibrated or percentile)
├── generate_competencies.py         # Future-aware competency generator (LLM)
├── recommendations.py               # Ranked curriculum recommendations + ablation
├── enrich_with_dates.py             # Attach job_date → extraction outputs
├── skill_time_trend_analysis.py     # FDR-controlled time-series trends + stability
├── future_weight_mapping.py         # Map skills/knowledge → future domains (with margin)
├── ingest_future_domains.py         # Normalize WEF/O*NET/McKinsey → future_domains.csv
│
├── export_for_review.py             # Human-in-the-loop review tables
├── export_competencies_for_review.py # Competency review export
├── export_gold_set.py               # Stratified gold set for labeling
├── import_feedback.py               # Merge feedback_store → feedback artifacts
├── apply_feedback.py                # Apply Bloom/type corrections
│
├── evaluate_extraction.py           # P/R/F1 per extractor (BERT/GPT/Hybrid)
├── validate_parameters.py           # AUC, Brier, calibration, cross-validated AUC
├── evaluate_competency_generation.py # Competency quality metrics
├── evaluate_future_mapping.py       # Domain mapping accuracy vs gold labels
├── log_run_metadata.py              # Record run metadata for reproducibility
├── aggregate_results.py             # Aggregate runs + cross-run summary
├── preprocess_jobs_pipeline.py      # Raw jobs → jobs_sentences.csv, jobs_metadata.csv
│
├── review_ui/                       # Web UI for internal/development review
│   ├── app.py                       # FastAPI backend
│   ├── static/app.js                # Frontend logic
│   └── templates/index.html
│
├── gold_labeling_ui/                # Web UI for gold-set labeling (multi-reviewer)
│   ├── app.py                       # FastAPI backend
│   ├── static/app.js, style.css     # Frontend
│   └── templates/index.html         # Skills, Knowledge, Domain tabs
│
├── merge_gold_labels.py             # Majority-vote merge of multi-reviewer labels
│
├── dashboard/                       # Admin + school dashboard (production)
│   ├── app.py                       # FastAPI app, school review, results
│   ├── db.py                        # SQLite (schools, departments, users, runs)
│   ├── templates/                   # Jinja2 (admin, school)
│   └── static/style.css
│
├── feedback_store/                  # Per-reviewer feedback (default run)
│   ├── skill_feedback.csv
│   ├── knowledge_feedback.csv
│   └── competency_feedback.csv
│
├── DATA/
│   ├── labels/                      # Gold set for evaluation
│   │   ├── gold_skills.csv
│   │   ├── gold_knowledge.csv
│   │   ├── gold_future_domain.csv
│   │   └── gold_labels/             # Multi-reviewer labels (from gold_labeling_ui)
│   ├── samples/                     # Sample CSVs for dashboard simulation
│   │   ├── jobs_sample.csv
│   │   ├── curriculum_sample.csv
│   │   └── README.md
│   └── preprocessing/data_prepared/
│       ├── jobs_sentences.csv       # Pipeline input
│       └── jobs_metadata.csv        # job_id → job_date
│
├── results/                         # Output of a single run
├── results_aggregated/              # Aggregated results across runs
│
├── RESEARCH_QUESTIONS.md            # RQs, metrics, ablation design
├── CALCULATIONS.md                  # Ranking, voting, weighting formulas
├── PIPELINE.md                      # Detailed pipeline documentation
├── run.bat                          # Phase 1: Full pipeline (15 steps)
├── run_phase_2.bat                  # Phase 2: Post-review pipeline (13 steps)
└── scripts/create_sample_csvs.py   # Generate DATA/samples/*.csv for dashboard
```

---

# ⚡ Quick Start

```bat
REM 0. (One-time) Generate real future domains from WEF/O*NET/McKinsey
python ingest_future_domains.py

REM 1. Phase 1 — Full pipeline (14 steps: extraction → trends → recommendations → gold set)
run.bat

REM 2. (Optional) Label gold set: uvicorn gold_labeling_ui.app:app --reload
REM    Then python merge_gold_labels.py — see DATA/labels/LABELING_PROTOCOL.md

REM 3. Expert review — start the web UI (optional but recommended)
uvicorn review_ui.app:app --reload
REM Open http://127.0.0.1:8000/?reviewer_id=alice

REM 4. Phase 2 — Post-review (13 steps: calibration → re-generation → plots → evaluation)
run_phase_2.bat

REM 5. Label top-20 recommendations (results/recommendations.csv → expert_priority column)
python recommendations.py --evaluate

REM 6. (Optional) Dashboard simulation: upload DATA/samples/*.csv, run department pipeline
REM 7. (Optional) Multi-run: rename results → results_run1, repeat, then aggregate
python aggregate_results.py --run_dirs results_run1 results_run2 --output_dir results_aggregated
```

**Larger data:** Edit `run.bat` or run `python pipeline.py --sample_size 5000` (or `0` for no limit).

See [PIPELINE.md](PIPELINE.md) for detailed steps, data flow, and troubleshooting.
See [CALCULATIONS.md](CALCULATIONS.md) for ranking, voting, weighting, and evaluation formulas.
See [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md) for evaluation framework and metrics.

---

# 🖥 Dashboard (Admin + School)

This repo includes a **dashboard app** at `dashboard/` for production use:

- **Admin**: Schools, departments, users, runs, inter-rater reliability
- **School users**: Department-scoped uploads, runs, results, insights, **in-dashboard review**
- **Multi-reviewer**: Each school can have multiple users; feedback stored per `reviewer_id` (email)
- **Crowd-sourced aggregation**: Schools with same vocational field can aggregate results
- **Ranking modes**: `model_only` (pipeline scores only) or `human_adjusted` (incorporates expert feedback)

Run:

```bat
uvicorn dashboard.app:app --reload
```

Open `http://127.0.0.1:8000/dashboard/login` — default admin: `admin@local` / `admin123`.

See **[dashboard/README.md](dashboard/README.md)** for full documentation.

---

# 🔬 Review UI (Internal / Development)

For **internal or development** reviews (no auth, single output dir):

```bat
uvicorn review_ui.app:app --reload
```

Open `http://127.0.0.1:8000/?reviewer_id=alice` — feedback goes to `feedback_store/` (project root).

| Aspect | review_ui | Dashboard |
|--------|-----------|-----------|
| Purpose | Internal / dev | Production (schools) |
| Auth | None (URL param) | Login required |
| Reviewer ID | `?reviewer_id=` | Logged-in user email |
| Data | Default `results/`, `feedback_store/` | Per-department `data/schools/.../` |

**Notes:**
- Dashboard runs are isolated under `data/schools/{school_id}/departments/{department_id}/`
- Existing `run.bat` and `run_phase_2.bat` workflows remain unchanged

---

# 🚀 System Overview

This project introduces a **multi-layer curriculum intelligence pipeline** combining NLP, educational taxonomy, labour-market analysis, and future-of-work forecasting.

## **Main Stages**

### **1. Data Acquisition & Cleaning**

* Scraped job postings (IT / Software / Game Development)
* Cleaning markdown noise (** \ // etc.)
* Sentence splitting
* Every sentence carries **job_id + job_date**

### **2. Hybrid Extraction (JobBERT + GPT)**

* **JobBERT + CRF** for BIO-tagged skill/knowledge spans
* **GPT-based extractor** for structured JSON
* **Semantic agreement** using SBERT embeddings
* **Fusion Engine** merges both with:

  * confidence tiers
  * semantic density
  * hard vs soft skill classification

### **3. Taxonomy Layer**

* Hard vs soft skills
* Bloom’s taxonomy for **hard skills only**
* Semantic density scoring

### **4. Curriculum Mapping**

* Compare skills/knowledge with **SMK Software & Game Dev curriculum**
* Component mapping via SBERT
* Compute:

  * coverage percentage
  * HOT (Analyze-Evaluate-Create) distributions
  * component-level heatmaps

### **5. Future-of-Work Integration**

* Normalizes real forecast sources (WEF, O*NET, McKinsey) via `ingest_future_domains.py`
* Maps skills/knowledge to domains using SBERT cosine similarity
* Computes:

  ```
  future_weight = similarity(skill, domain) × trend_score
  ```
* Includes **mapping uncertainty** (top1-top2 similarity margin)
* Identifies future-critical skills, declining skills, and curriculum gaps

### **6. Time Trend Analysis**

* FDR-controlled (Benjamini-Hochberg) emerging/declining skill detection
* Outputs q-values (not just raw p-values) to control false discovery rate
* Stability analysis across multiple seeds and min_jobs thresholds

### **7. Competency Generator (LLM)**

* Uses verified skills + future context + empirical trend signals
* Produces competency IDs, titles, descriptions, related skills, future relevance notes

### **8. Curriculum Recommendations**

* Ranked skill gap priorities combining: **demand, empirical trend, future_weight** (coverage is for insights only, not prioritization)
* Schools use the system to design better curriculum; existing curriculum may be outdated
* Evidence traces per recommendation (job_ids, trend stats, domain info)
* Ablation study (remove one signal at a time; optional `with_coverage` variant)
* Expert evaluation: Precision@20, NDCG@20

### **9. Export for Review & Human-in-the-Loop**

Creates sampled CSVs for expert validation (500 skills, 200 knowledge, 100 competencies).

**Review workflow (single or multi-reviewer):**
1. Phase 1 (`run.bat`) exports review tables and gold-set labels automatically
2. Start the review web app: `uvicorn review_ui.app:app --reload`
3. **Multi-reviewer:** Each reviewer opens `http://localhost:8000/?reviewer_id=alice`
4. Review in browser; feedback auto-saves to `feedback_store/`
5. Phase 2 (`run_phase_2.bat`) imports feedback, calibrates scoring, re-generates, and evaluates

### **10. Scientific Evaluation**

* **Gold set labeling** (`DATA/labels/`) for ground-truth extraction quality
* **Extraction evaluation**: P/R/F1 per source (BERT/GPT/Hybrid) with Wilson CIs
* **Calibrated verification**: AUC-ROC, Brier score, calibration curve, cross-validated AUC
* **Domain mapping validation**: Top-1 accuracy vs expert labels
* **Reproducibility**: Run metadata with dataset hash, model versions, seeds

---

# 🧪 Experimental Workflow

The system supports **multiple independent runs** for robust evaluation.

### **1. Run an experiment (e.g., sample size = 1000)**

```bat
run.bat
```

For **larger data**: `python pipeline.py --sample_size 5000` (or `--sample_size 0` for no limit).

After the run completes, rename the results folder:

```
results → results_run1
```

Repeat:

```
results_run2
results_run3
...
```

### **2. Aggregate runs**

```bash
python aggregate_results.py --run_dirs results_run1 results_run2 results_run3 --output_dir results_aggregated
```

### **3. Generate final plots, competencies, and review tables**

Set `OUTPUT_DIR = "results_aggregated"` in `config.py`
Then run:

```bash
python plot_generator.py
python future_weight_mapping.py
python verify_skills.py
python generate_competencies.py
python export_for_review.py
python skill_time_trend_analysis.py --only_hard
```

---

# 📊 Visualizations & Analytics

The system generates:

### **Hybrid model comparison**

* JobBERT vs GPT vs Hybrid
* Skill/Knowledge counts
* Confidence score distributions

### **Bloom taxonomy distribution**

* For hard skills only
* Across JobBERT, GPT, Hybrid

### **Curriculum heatmap**

* Curriculum components (Y-axis)
* Bloom levels (X-axis)

### **Top-N clusters**

* Hard skills
* Knowledge items
* Soft skills
* Skills demanded but **not covered** by curriculum (insight)
* Skills "future-critical" but underrepresented

### **Time trend analysis**

* Emerging vs declining skills
* Based on `job_date`
* Monthly trend slopes

### **Future-of-work analytics**

* future_weight histogram
* Top future-weighted skills and knowledge
* Emerging skills coverage (covered vs not covered; insight only)

### **Coverage (insight only)**

* Coverage distribution across jobs
* Coverage improvement: Hybrid vs base models
* Coverage is not used for prioritization; schools use recommendations to design better curriculum.

---

# 🧠 Competency Generation

`generate_competencies.py` produces:

* JSON competency framework
* 10–25 competencies per batch
* Each includes:

  * id
  * title
  * description
  * **related skills**
  * **future relevance statement**

This output can be directly used in a **curriculum redesign document** or **expert workshop**.

---

# 📝 Pipeline Diagrams

All diagrams can be generated using the provided prompts in `/docs/prompts/`
(Or directly pasted into an AI image generator.)

### Includes:

* Full pipeline architecture
* Checkpoint diagrams:

  * preprocessing
  * hybrid extraction
  * taxonomy mapping
  * future-of-jobs layer
  * competency generation & review

---

# 💼 Future Work

* Train a **domain-specific SBERT** model for improved skill-domain matching
* Add **semantic search** over extracted competencies
* Testing with >10,000 job postings
* Incorporate additional national/regional forecast sources

---

# 🙌 Citation & Acknowledgment

If you use this pipeline or insights from this project, please cite:

```
[Astuti, Listyanti Dewi]. Future-aware hybrid skill extraction for curriculum intelligence. (2025)
```

---

# 📬 Contact

For questions or collaboration:

* **Author:** Listyanti Dewi Astuti
* **Affiliation:** SMK Negeri 12 Malang / Universitas Negeri Malang
* **Email:** [your email here]

---
