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

## Hosting (decided)

| | Where | Notes |
|---|---|---|
| **Annotator UI** | **Streamlit Community Cloud** | Free, deploys from this GitHub repo, stable public URL. Set `GS_API_URL` and nothing else. |
| **Collector** | **Fly.io**, Singapore (`sin`), 1 GB volume | `fly.toml` at the repo root is ready. ~$2-5/month. Nearest Fly region to Indonesia. |

**The UI holds no research data.** `DATA/gold_standard/` is gitignored, so a UI
deployed from GitHub has no sentences on disk — it fetches them from
`GET /sentences` under the annotator's own token. Two things follow, both
wanted: no corpus text is published to GitHub, and the pilot sentences are not
readable from the public UI URL without a token. The sample CSVs are baked into
the *collector* image instead (~50 KB; never the 128 MB sentence pool).

**One machine only.** SQLite on one volume means `fly scale count 2` would give
each machine its own volume and split the three annotators across two databases
with no error. `fly.toml` pins `min_machines_running = 1` and
`auto_stop_machines = false`.

The one requirement that is not negotiable: **the collector database must sit on
storage that survives a restart.** If it does not, use the local-file mode
instead and collect the three JSON files by hand.

## Deploy

Prerequisites: a Fly.io account with `flyctl` installed, and this repo pushed to
GitHub (Streamlit Cloud deploys from GitHub, so unpushed commits are not
deployed).

```bash
# 1. COLLECTOR, on Fly.io. Creates the app, the persistent volume, then deploys.
fly launch --no-deploy --copy-config --name rpl-annotation-collector
fly volumes create gs_data --region sin --size 1
fly deploy                      # builds annotation_server/Dockerfile
fly status                      # note the hostname, e.g. rpl-annotation-collector.fly.dev
curl https://rpl-annotation-collector.fly.dev/health      # -> {"ok":true}

# 2. UI, on Streamlit Community Cloud (share.streamlit.io):
#    repo = this repo, branch = main, main file = gold_standard/app.py
#    then under Settings -> Secrets / environment:
#      GS_API_URL = "https://rpl-annotation-collector.fly.dev"
#    Root requirements.txt already pins exactly what the UI needs.

# 3. TOKENS. Mint them AFTER the UI URL exists - the URL is baked into each link.
.venv\Scripts\python.exe -m annotation_server.make_tokens --base-url https://YOUR-STREAMLIT-URL --out api_keys/annotation_tokens.json

# 4. Give the collector the tokens, as a secret (never in the image or git).
fly secrets set GS_TOKENS="$(cat api_keys/annotation_tokens.json)"
```

Step 3 prints six links: a `practice_10` and a `pilot_100` link for each of
AV / AI / AD. Send each annotator **only their own two**, and keep the admin
token to yourself.

To re-deploy after re-drawing a sample (`python -m gold_standard.sample`), run
`fly deploy` again so the new CSVs are baked in. Annotations are unaffected:
they live on the volume, not in the image.

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
| GET | `/sentences` | annotator | the caller's own split's sentences to annotate |
| GET | `/annotations` | annotator | own saved work, for resume |
| PUT | `/annotations/{sentence_id}` | annotator | upsert one sentence (idempotent) |
| GET | `/admin/progress` | admin | per-annotator counts |
| GET | `/admin/export` | admin | ZIP (IAA-ready) |
| GET | `/admin/export.json` | admin | same payload as JSON |

Writes are per-sentence upserts, so a refresh or a second browser tab cannot
clobber earlier work, and a retried save is harmless.

Tests: `.venv\Scripts\python.exe tests\test_annotation_server.py` (14 tests;
needs the web venv, which has fastapi/httpx).
