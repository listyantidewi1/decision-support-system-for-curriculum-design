# Acquisition Cadence — Forward Accumulation Runbook

Status: living runbook. Owner: Listyanti Dewi Astuti. Last updated: **2026-10-30**.
Governs Workstream A3 (rolling batch schedule) from `docs/IMPLEMENTATION_PLAN.md`.

> **Why this document exists.** Job boards serve only *live* postings plus a short
> tail. A 36-month corpus therefore **cannot be scraped backwards** — it must be
> **accumulated forward**, one snapshot at a time, with `first_seen`/`last_seen`
> tracking each posting across batches. Every month we skip is a month of the
> temporal window that is permanently unrecoverable. This is the one acquisition
> task with a hard wall-clock dependency: it cannot be "caught up" later.

---

## 1. Source roster — live status (2026-10-30)

| Source | Reachable? | Tabel 3.9 pass | Role in cadence | Notes |
|---|---|---|---|---|
| **Indeed** (JobSpy) | ✅ yes | ~32% (ID) | Volume backbone, all strata | Robust; low ban risk; `fetch_description=true` for eligibility |
| **Glints** (Next.js state) | ✅ yes | **66%** (ID) | High-yield ID tech roles | Detail-fetch → rate-limit exposure; obey session guardrail |
| **Kalibrr** (public API) | ✅ yes | ~7% (ID) | Low-yield but legal fresh supply | General board; most results off-domain (correctly filtered) |
| **Jobstreet / SEEK** (v5 API) | ⛔ **BLOCKED** | 62% (ID), best metadata | *Would be* the top ID source | **Cloudflare managed-challenge ban** since 2026-10-29 (403 "Just a moment…"). Affects ID **and** MY/SG/PH (same infra/IP). Re-probe before each cadence; do **not** hammer. |
| **Adzuna** (official API) | ✅ yes | — | **Benchmark only** (`DATA/corpus_benchmark/`) | 500-char description truncation; **no Indonesia**. Never enters the extraction corpus. |
| LinkedIn / Glassdoor | historical only | — | Not re-acquired | K1/K5: historical rows kept with provenance; not part of forward cadence |

**Before every cadence run, re-probe Jobstreet** (single search GET, no detail fetch):
`.venv-ml\Scripts\python.exe -m acquisition.probe_jobstreet` (add `--all` for MY/SG/PH)
→ exit 0 / HTTP 200 means Jobstreet is back; queue batch-002f (the 32 ID terms interrupted in batch-002e) + MY/SG/PH.

---

## 2. Operational guardrails (do not exceed)

- **Detail-fetch platforms (Jobstreet, Glints):** ≤ **25 terms** *or* ≤ **1,500 detail fetches** per session; ≥ **24 h cooldown** between heavy sessions on the same platform. (Empirical: SEEK WAF banned after ~33 heavy terms / ~2,000 detail fetches in one session.)
- **Circuit breaker** (`run_batch.py`): aborts on 3 consecutive 403/429 zero-row queries and writes what it has. If it fires, stop that platform for ≥24 h; the missing terms become a supplementary batch.
- **Indeed (JobSpy):** more tolerant; still pace (`sleep_seconds` ≥ 4) and keep `results_wanted` sane for niche terms.
- **One platform per batch config** (the runner dispatches by the single `sites` entry). Run platforms sequentially, not interleaved, so a ban on one is isolated and legible in its own log.
- **Raw is immutable:** `write_batch` refuses to overwrite an existing `batch_id`. Never edit a written batch; add a new one.

---

## 3. Schedule

| Stratum | Cadence | Platforms (reachable) | Rationale |
|---|---|---|---|
| **Indonesia** | **Monthly** | Indeed + Glints (+ Kalibrr quarterly; + Jobstreet when unblocked) | Binding constraint (4,486 / 10k floor); needs steady forward gain |
| **ASEAN** (MY/SG/PH/TH/VN) | Bi-monthly | Indeed (+ Jobstreet MY/SG/PH when unblocked) | Target already met (12,568); maintain temporal series |
| **OECD** | Bi-monthly | Indeed (+ Adzuna → benchmark store only) | Target already met (59,069); maintain temporal series |

Target spine (§3.4): Indonesia 10k–20k unique · ASEAN ≥5k · OECD ≥5k · per-country coverage gate (provisional 1,000 unique eligible).

---

## 4. Monthly procedure

Batch-id convention for the forward series: **`batch-<YYYYMM>-<platform>-<stratum>`**
(e.g., `batch-202611-indeed-id`). Legacy `batch-00X` names are kept as-is; the store
treats `batch_id` as an opaque directory name, so the scheme is cosmetic.

1. **Re-probe Jobstreet** (§1). Record the result in the batch `notes`.
2. **Copy the latest config** for each (platform × stratum), bump `batch_id` to this
   month, set `hours_old` to cover the interval (monthly ⇒ **720**; bi-monthly ⇒ 1440),
   keep `fetch_description=true` for Indeed. Keep the query-plan version
   (`qp-v2` core + `qp-v2.1` expansion) unless terms change.
3. **Dry-run** each: `python -m acquisition.run_batch --config <cfg> --dry-run`.
4. **Run sequentially**, one platform at a time, each logging to a file:
   `python -m acquisition.run_batch --config <cfg>  *> <scratchpad>\<batch_id>.log`
   Watch for the circuit breaker; stop a platform for 24 h if it fires.
5. **Recompute the ladder** over all batches (Gambar 3.7):
   `python -m governance.ladder` → writes JSON/MD + `record_flags` to `results/corpus_audits/`.
6. **Verify the platform × stratum breakdown**, not just the headline N
   (lesson: a mixed `date_posted` format once silently NaT'd an entire source to 0
   eligible). Confirm each new batch contributed where expected.
7. **Update** `project_pipeline_v2_current_state` memory + the proposal-track memory
   with the new strata totals, and note any source-status change in `SOURCE_AUDIT.md`.
8. **Commit** the new config(s), the audit outputs, and doc/memory updates.

`first_seen`/`last_seen` are maintained automatically by `batch_store.seen_spans()`:
a `source_record_id` reappearing in a later batch = *still open*, not new demand.

---

## 5. What "done" looks like for a cadence month

- New batch(es) on disk under `DATA/corpus_raw/`, immutable, with full manifests.
- Fresh ladder in `results/corpus_audits/` with the per-stratum coverage-gate verdict.
- Indonesia eligible total moved up (or, if flat, a documented reason — e.g., snapshot
  saturation, meaning the gain must come from term expansion or the next month).
- Memory + `SOURCE_AUDIT.md` reflect reality; `git` tree clean.
