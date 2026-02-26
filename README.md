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

---

# 🔍 Repository Structure

```
skill-extraction/
│
├── pipeline.py                      # Main hybrid extraction pipeline
├── config.py                        # Global configuration
├── plot_generator.py                # Visual analytics
├── verify_skills.py                 # Skill verification tiers
├── generate_competencies.py         # Future-aware competency generator (LLM)
├── export_for_review.py             # Human-in-the-loop review tables
├── export_competencies_for_review.py # Competency review export
├── enrich_with_dates.py             # Attach job_date → extraction outputs
├── skill_time_trend_analysis.py     # Time-series trend analysis
├── future_weight_mapping.py         # Maps knowledge → future domains
├── import_feedback.py               # Merge feedback_store → feedback artifacts
├── apply_feedback.py                # Apply Bloom/type corrections
├── aggregate_results.py             # Aggregates multiple experiment runs
├── preprocess_jobs_pipeline.py      # Raw jobs → jobs_sentences.csv, jobs_metadata.csv
│
├── review_ui/                       # Web UI for expert review
│   ├── app.py                       # FastAPI backend
│   ├── static/app.js                # Frontend logic
│   └── templates/index.html
│
├── feedback_store/                  # Per-reviewer feedback (auto-saved)
│   ├── skill_feedback.csv
│   ├── knowledge_feedback.csv
│   └── competency_feedback.csv
│
├── DATA/preprocessing/data_prepared/
│   ├── jobs_sentences.csv           # Pipeline input
│   └── jobs_metadata.csv            # job_id → job_date
│
├── results/                         # Output of a single run
├── results_run1/                    # Snapshot of run 1
├── results_aggregated/              # Aggregated results across runs
│
├── run.bat                          # Phase 1: Full pipeline
└── run_phase_2.bat                  # Phase 2: Post-review pipeline
```

---

# ⚡ Quick Start

1. **Phase 1** — Run full pipeline: `run.bat`
2. **Review** — Start UI: `uvicorn review_ui.app:app --reload`, open `http://127.0.0.1:8000/?reviewer_id=alice`
3. **Phase 2** — After review: `run_phase_2.bat`

See [PIPELINE.md](PIPELINE.md) for detailed steps, data flow, and troubleshooting.

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

* Reads WEF/McKinsey-style domains from `future_domains_dummy.csv`
* Computes:

  ```
  future_weight = similarity(skill, domain) × trend_score
  ```
* Identifies:

  * future-critical skills
  * declining skills
  * curriculum gaps for future-ready design

### **6. Competency Generator (LLM)**

* Uses verified skills + future context
* Produces:

  * competency IDs
  * titles
  * descriptions
  * related skills
  * future relevance notes

### **7. Export for Review & Human-in-the-Loop**

Creates sampled CSVs for expert validation (500 skills, 200 knowledge, 100 competencies):

* `expert_review_jobs.csv`
* `expert_review_skills.csv` (with human_valid, human_bloom, human_notes columns)
* `expert_review_knowledge.csv` (with future weight + human_valid, human_notes)
* `expert_review_competencies.csv` (with human_quality, human_relevant, human_notes)

**Review workflow (single or multi-reviewer):**
1. Run `export_for_review.py` and `export_competencies_for_review.py`
2. Start the review web app: `uvicorn review_ui.app:app --reload`
3. **Multi-reviewer:** Each reviewer opens the app with a unique ID in the URL, e.g. `http://localhost:8000/?reviewer_id=alice` or `?reviewer_id=r1`. All reviewers see the same set; feedback is stored per reviewer.
4. Review in browser; feedback auto-saves to `feedback_store/`
5. Run `import_feedback.py` to merge feedback (uses majority vote when multiple reviewers)
6. Run `apply_feedback.py` to apply Bloom overrides and filtering
7. Run `generate_competencies.py --comprehensive` (or `--human_verified_only` for conservative mode)
8. Run `evaluate_competency_generation.py` to assess competency quality

---

# 🧪 Experimental Workflow

The system supports **multiple independent runs** for robust evaluation.

### **1. Run an experiment (e.g., sample size = 1000)**

```bat
run.bat
```

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
* Skills demanded but **not covered** by curriculum
* Skills "future-critical" but underrepresented

### **Time trend analysis**

* Emerging vs declining skills
* Based on `job_date`
* Monthly trend slopes

### **Future-of-work analytics**

* future_weight histogram
* top future-critical knowledge
* declining domains

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

* Incorporate **real** WEF/McKinsey datasets instead of dummy files
* Train a **domain-specific SBERT** model for improved matching
* Add **semantic search** over extracted competencies
* Testing with >10,000 job postings

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
