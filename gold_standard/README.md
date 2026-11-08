# `gold_standard/` — Workstream B data acuan (gold standard) builder

Researcher-side ("catatan peneliti") preparation and delivery of the five-tag
sentence annotation sample for tahap **Design** (Paket B), faithful to annotation
guideline **AG-RPL-LIMA-v3.0** (`docs/proposal/PDF/proposal_coworking/instruments/`).

The guideline (B-1) and forms (B-2) themselves are validated by three experts with
Aiken's V (sheet V-B) **before** real annotation; this package only prepares the
data the annotators work on — which the guideline marks as researcher work, not an
instrument — and delivers it in two interchangeable formats.

## Pipeline

```
record_flags_*.parquet        (governance ladder: eligible postings + language)
        │  build_pool
        ▼
DATA/gold_standard/sentence_pool.parquet   (~550k sentences; HTML/markdown stripped,
        │  sample                           ±1 context, relevance + ID-signal scores)
        ▼
DATA/gold_standard/pilot_100.{parquet,csv} + _manifest.json
        │  export_xlsx            │  app.py
        ▼                         ▼
worksheets/pilot_100_<AV|AI|AD|MASTER>.xlsx   Streamlit span annotator
```

Run order (production venv — `.\run.ps1` or `.venv-ml\Scripts\python.exe`):

```powershell
python -m gold_standard.build_pool          # eligible corpus -> sentence pool
python -m gold_standard.sample               # -> pilot_100 (add --main for 1,200)
python -m gold_standard.export_xlsx          # -> Form A-1 workbooks (one per annotator)
streamlit run gold_standard/app.py           # interactive annotator (?annotator=AV)
```

## Sampling design (purposive — see `*_manifest.json`)

- **English + Indonesian only** (`ANNOTATOR_LANGUAGES`): the three expert annotators
  read only English and Indonesian (incl. ID–EN code-mixing, rule 7.13), so the
  human-annotated gold standard is restricted to these. Non-EN/ID postings stay in
  the raw corpus and are handled downstream via the translation layer + the B-6
  LLM-labeling audit — not by direct human annotation. (This also drops rare
  cross-border mis-stratified postings, e.g. a Vietnamese listing tagged Indonesia.)
- **Indonesian + bilingual boost**: Indonesia 45% / ASEAN 20% / OECD 35% of the
  sample (vs ~6% ID in the corpus), with a per-stratum minimum share of Indonesian /
  mixed-language sentences so rules 7.13 / 7.14 are exercised.
- A deliberate **15% low-relevance minority** so NON_RELEVANT / UNCERTAIN are real
  options during calibration.
- At most **2 sentences per posting**; exact-duplicate sentence text removed.
- All randomness seeded (`SEED` in `__init__.py`); the main 1,200 is drawn disjoint
  from the pilot.

## Two delivery formats (compare, then pick one)

| | Form A-1 workbook (`export_xlsx`) | Streamlit app (`app.py`) |
|---|---|---|
| Span offsets | annotator types span text + `occurrence_no`; researcher derives offsets in a post-pass | selected from tokens → **exact** start/end, captured live |
| Setup | none; opens in Excel | `streamlit run`, one locked URL per annotator |
| Dropdowns | locked (label / confidence / flag / relevance / has_5tag), whole-column so inserted rows keep them | native widgets, guideline-enforced (no span on NON_RELEVANT, required note, etc.) |
| Output | one `.xlsx` per annotator (SENTENCE + 14 SPAN rows each, insert more as needed) | per-annotator JSON autosave + Form A-1 CSV export |

Both emit the **same canonical Form A-1 columns** (`FORM_A1_COLUMNS` in `app.py` /
`HEADERS` in `export_xlsx.py`), but they are **not** byte-identical: the **app is
authoritative for exact character offsets** (captured at selection time), while the
**xlsx offsets are derived in a researcher post-pass** from `span_text` + `occurrence_no`
(needed because a span text can repeat within a sentence). The IAA step therefore runs a
small per-format normalizer (xlsx → derive offsets + validate completeness; app → as-is)
before strict-span comparison. Prefer the app when exact offsets matter.

Outputs live under `DATA/gold_standard/` and are **gitignored** (reproducible from
this code + the raw corpus); only the code here is tracked.
