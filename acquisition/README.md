# acquisition/ — raw-immutable OJP corpus harness (Workstream A)

Implements proposal §3.4.1 / Tabel 3.10: every batch is an append-only snapshot under
`DATA/corpus_raw/<batch_id>/` holding `records.parquet` (source columns exactly as fetched —
**no cleaning at this layer** — plus provenance columns) and `manifest.json` (method, versions,
query plan, counts, deviations). `registry.parquet` (derived, rebuildable) tracks every
(record, batch) sighting; `batch_store.seen_spans()` computes `first_seen`/`last_seen`.

Platform decisions live in `docs/corpus/SOURCE_AUDIT.md` (A1). Current channels: Indeed via
python-jobspy; Google parked (broken), LinkedIn dropped pending sign-off.

## Usage

```powershell
# one-time: ingest the legacy March-2026 CSVs as batch-001
.venv-ml\Scripts\python.exe -m acquisition.ingest_legacy

# run a batch from a versioned query plan
.venv-ml\Scripts\python.exe -m acquisition.run_batch --config acquisition/configs/batch-002.json
```

Rolling schedule (plan A3): monthly Indonesia, bi-monthly ASEAN/OECD. Copy the latest config to
`batch-00N.json`, bump `batch_id` (and `hours_old` to 720 for monthly refreshes), and run.
