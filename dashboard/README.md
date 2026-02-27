# Dashboard — Admin & School

FastAPI dashboard for production use: admin management, school workflows, and in-dashboard expert review.

---

## Quick Start

**Before dashboard simulation:** Run the pipeline from project root:

```bat
run.bat              REM Phase 1: 14 steps
REM [Optional] Expert review via: uvicorn review_ui.app:app --reload
run_phase_2.bat      REM Phase 2: 13 steps (after review)
```

Then start the dashboard:

```bat
uvicorn dashboard.app:app --reload
```

Open `http://127.0.0.1:8000/dashboard/login`

- **Admin**: `admin@local` / `admin123` (seeded on first run)
- **School**: Create via Admin → Users (role=school, select school)

---

## Features

### Admin

| Page | Purpose |
|------|---------|
| Schools | Create schools |
| Departments | Create departments (name, vocational field) |
| Users | Create users (admin or school; school users need school_id) |
| Runs | All pipeline runs; inter-rater reliability snapshot |
| IRR | Cohen's Kappa, agreement %, shared items (when multi-reviewer) |

### School

| Page | Purpose |
|------|---------|
| Upload | Job postings (CSV), curriculum (CSV/JSON) per department. Sample files: `DATA/samples/jobs_sample.csv`, `DATA/samples/curriculum_sample.csv` |
| Runs | Trigger pipeline run; view status |
| Results | Skills, knowledge, competencies (ranked); aggregation toggle |
| Review | In-dashboard review (skills, knowledge, competencies) with sub-tabs |
| Insights | Plots with descriptions; click to enlarge |

---

## Multi-Reviewer & Reviewer Identity

- **Multiple users per school**: Admin creates multiple users with same `school_id`
- **Reviewer ID**: Logged-in user's **email** (unique in DB; no collision across schools)
- **Feedback storage**: Per `(item_id, reviewer_id)` in department `feedback_store/`
- **Merge**: Majority vote when aggregating; see [CALCULATIONS.md](../CALCULATIONS.md)

---

## Ranking Modes (Results Page)

| Mode | Description |
|------|-------------|
| **model_only** | Pipeline scores only; human feedback shown but not used for ordering |
| **human_adjusted** | Pipeline scores + expert verification boost/penalty |

Formulas: [CALCULATIONS.md](../CALCULATIONS.md) §4–6.

---

## Cross-School Aggregation

When "Aggregate with same vocational field across schools" is checked:

- Results combine data from all departments with the same `vocational_field`
- Contributor metadata (school, department, runs, uploads) is shown for transparency
- Review statuses and future weights are merged (any verified wins; max future_weight)

---

## Paths

| Scope | Path |
|-------|------|
| Department data | `data/schools/{school_id}/departments/{department_id}/` |
| Uploads | `uploads/` |
| Preprocessing | `preprocessing/` |
| Results | `results/` |
| Feedback | `feedback_store/` |
| Fallback | Project `results/` when department has no runs |

---

## Dependencies

```
fastapi
uvicorn
jinja2
python-multipart
pandas
```

Install: `pip install -r dashboard/requirements.txt`

---

## Non-Invasive Design

- Dashboard does **not** modify `DATA/`, `results/`, or `feedback_store/` at project root
- Pipeline runs via `pipeline_orchestrator.py` with path overrides
- `run.bat` and `run_phase_2.bat` remain unchanged
