# `annotation_server/` — hosted collector for the pilot annotation

Lets the three expert annotators (AV/AI/AD) work from a link, and lets you
download everything from one place. Two pieces:

```
Streamlit annotator (gold_standard/app.py)   ──PUT /annotations/{sid}──▶   this collector
   one private link per annotator                                          SQLite file
                                                                                │
                                              GET /admin/export  ◀─────────── you
                                              (ZIP that gold_standard.iaa reads directly)
```

## Why a collector at all

Every free host (Streamlit Community Cloud, Hugging Face Spaces free tier) has an
**ephemeral filesystem**. The annotator app autosaves to local JSON, so on those
hosts a container restart silently destroys the annotations — and you would not
find out until you went looking for them. The collector keeps the data in one
SQLite file you control, which *is* the download.

It also closes two problems found during the build:

- **Identity.** The local app took its identity from `?annotator=AV`, which is
  guessable: anyone with the URL could open any annotator workspace, and a
  curious click was indistinguishable from expert annotation. Here the token maps
  server-side to one `(annotator, split)` and the client never chooses its own
  identity. No endpoint exposes one annotator work to another, so independence
  (guideline Bagian 10.1) is enforced rather than requested.
- **Trial contamination.** A real instance occurred: an interface trial left 8
  judged sentences and 10 spans in `pilot_100_AV.json`, which the app would have
  handed to the real annotator AV as their own work. Each annotator therefore gets
  **two** links — a `practice_10` one for warm-up and a `pilot_100` one for the
  real thing. Practice writes cannot land in the pilot split.

## Recommended hosting

| | Recommendation |
|---|---|
| **Annotator UI** | **Streamlit Community Cloud** — free, deploys from this repo, gives a stable public URL. Set `GS_API_URL` and nothing else; the UI keeps no state. |
| **Collector** | Any host with a **persistent disk**: Fly.io volume, Render disk, Railway volume, or a university VM. Mount it and point `GS_DB_PATH` at it. |
| **Alternative** | If you would rather not run a second service, run the collector on your own machine behind a Cloudflare Tunnel. Simplest, but your machine must be reachable while the annotators work — over a multi-day pilot that is the weak point, which is why a hosted disk is the recommendation. |

The one requirement that is not negotiable: **the collector database must sit on
storage that survives a restart.** If it does not, use the local-file mode
instead and collect the three JSON files by hand.

## Deploy

```bash
# 1. mint tokens (writes a gitignored file, prints the links to send)
.venv-ml\Scripts\python.exe -m annotation_server.make_tokens \
    --base-url https://YOUR-STREAMLIT-URL --out api_keys/annotation_tokens.json

# 2. run the collector (locally, or in the container)
pip install -r annotation_server/requirements.txt
GS_TOKENS_FILE=api_keys/annotation_tokens.json \
GS_DB_PATH=/data/annotations.sqlite \
uvicorn annotation_server.api:app --host 0.0.0.0 --port 8000

# 3. point the Streamlit app at it
GS_API_URL=https://YOUR-COLLECTOR-URL streamlit run gold_standard/app.py
```

`docker build -f annotation_server/Dockerfile .` builds the collector image.

**Secrets.** Tokens are never committed. Supply them as `GS_TOKENS` (a JSON
object) or `GS_TOKENS_FILE` (a path under the gitignored `api_keys/`). Send each
annotator only their own two links; keep the admin link to yourself.

## Collecting the results

```bash
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
     "https://YOUR-COLLECTOR-URL/admin/export?split=pilot_100" -o annotations.zip
unzip annotations.zip -d DATA/gold_standard/annotations/
.venv-ml\Scripts\python.exe -m gold_standard.iaa --split pilot_100
```

The ZIP is laid out as `{split}_{annotator}.json`, exactly the filenames
`gold_standard.iaa --ann-dir` expects, so the IAA step runs unchanged. It also
contains `events.csv` (an append-only trail of when each judgement was saved,
which is what lets you *demonstrate* independent annotation rather than assert
it) and `progress.json`.

`GET /admin/progress` shows per-annotator counts and first/last save time, so you
can see at a glance who has started and who has not.

## Endpoints

| Method | Path | Who | Purpose |
|---|---|---|---|
| GET | `/health` | open | liveness |
| GET | `/me` | annotator | resolve token to (annotator, split) |
| GET | `/annotations` | annotator | own saved work, for resume |
| PUT | `/annotations/{sentence_id}` | annotator | upsert one sentence (idempotent) |
| GET | `/admin/progress` | admin | per-annotator counts |
| GET | `/admin/export` | admin | ZIP (IAA-ready) |
| GET | `/admin/export.json` | admin | same payload as JSON |

Writes are per-sentence upserts, so a refresh or a second browser tab cannot
clobber earlier work, and a retried save is harmless.

Tests: `.venv\Scripts\python.exe tests\test_annotation_server.py` (14 tests;
needs the web venv, which has fastapi/httpx).
