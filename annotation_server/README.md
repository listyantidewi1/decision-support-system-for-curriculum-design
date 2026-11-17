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

## Local dry run (do this before every deploy)

Runs the whole chain on one machine so a problem surfaces here instead of with
three experts waiting. Verified working 2026-11-17.

**The two halves need DIFFERENT venvs.** The collector needs `.venv` (fastapi,
no streamlit); the UI needs `.venv-ml` (streamlit, no fastapi). Running either
under the other's venv fails with `No module named ...`.

```bash
# 1. tokens, once. Writes to a gitignored file; NEVER paste tokens into chat.
#    (make_tokens refuses to overwrite an existing file -- move it aside first.)
.venv/Scripts/python.exe -m annotation_server.make_tokens     --base-url http://localhost:8501 --split pilot_100     --practice-split practice_10 --out api_keys/dryrun_tokens.json     > api_keys/DRYRUN_LINKS.txt

# 2. collector, on :8000. GS_DB_PATH somewhere disposable, NOT in DATA/.
GS_DB_PATH=/tmp/annotations_dryrun.sqlite GS_TOKENS_FILE=api_keys/dryrun_tokens.json GS_SAMPLES_DIR=DATA/gold_standard   .venv/Scripts/python.exe -m uvicorn annotation_server.api:app       --host 127.0.0.1 --port 8000 --log-level warning

# 3. UI, on :8501, in a second terminal
GS_API_URL=http://127.0.0.1:8000   .venv-ml/Scripts/python.exe -m streamlit run gold_standard/app.py       --server.port 8501 --server.headless true
```

Then open a link from `api_keys/DRYRUN_LINKS.txt` (start with a *practice* one).
Check: `curl http://127.0.0.1:8000/health` -> `{"ok":true}`.

### What the dry run must prove

Not just "it loads". Four things, in order:

1. `/sentences` is **token-scoped** -- a pilot token sees 100 rows, a practice
   token sees 10, and neither sees the other's.
2. **Sentence ids survive the round trip.** Every real id looks like
   `u-1e3f720f8c7fc360#007`. The `#` must be percent-encoded in the PUT path
   (`gold_standard/app.py` does this via `quote(sid, safe="")`). Unencoded, the
   server receives only the posting id, the two sentences drawn from one posting
   overwrite each other, and every saved id loses its suffix -- i.e. **every
   save is corrupted**, silently, and you find out at IAA time.
3. The admin zip export yields `<split>_{AV,AI,AD}.json` + `events.csv` +
   `progress.json`, which is exactly what `gold_standard.iaa --ann-dir` reads.
4. `gold_standard.iaa` runs on it and produces a token-level kappa. If kappa comes
   back `n/a` with "no sentence text was available", the usual cause is an
   out-of-schema `relevance` value: it must be one of
   `RELEVANT / NON_RELEVANT / UNCERTAIN`. `YA / TIDAK` is the `has_5tag`
   vocabulary, not relevance -- mixing them up silently empties the token level.

### Afterwards

Delete the dry-run tokens before minting live ones.

You no longer have to remember to move the dry-run IAA output: since 2026-11-17
`gold_standard.iaa` **refuses** to write into `DATA/gold_standard/iaa/` when the
annotated sentence ids are not in the split's sample, or when the input sits
under a `_synthetic/` directory. For a rehearsal, point it somewhere else:

```bash
.venv-ml\Scripts\python.exe -m gold_standard.iaa --split practice_10 \
    --ann-dir <unzipped> --out-dir <somewhere disposable>
```

Every report now records, under `provenance`, which directory it read and
whether its ids were verified against the sample. The four synthetic reports
that prompted this are in `DATA/gold_standard/_synthetic/iaa/`.

## Surviving a reboot

`run_annotation_stack.ps1` (repo root) supervises both halves and registers them
to start again at logon. Run everything from `d:\Projects\skill-extraction`.

```powershell
.\run_annotation_stack.ps1 status                  # what is actually running
.\run_annotation_stack.ps1 start   -Mode dryrun    # or -Mode pilot
.\run_annotation_stack.ps1 stop
.\run_annotation_stack.ps1 install -Mode dryrun    # auto-start at logon
.\run_annotation_stack.ps1 uninstall
.\run_annotation_stack.ps1 backup  -Mode pilot     # daily during the pilot
```

It supersedes the two manual commands in the dry-run section above; those still
work, but they leave the database in a disposable directory and nothing restarts
them.

**`-Mode` is the contamination boundary, and it is a different file, not a
convention:**

| Mode | Database | Tokens |
|---|---|---|
| `pilot` | `DATA\gold_standard\collector\pilot_100.sqlite` | `api_keys\annotation_tokens.json` |
| `dryrun` | `D:\Projects\_skill-extraction-ops\dryrun\annotations_dryrun.sqlite` | `api_keys\dryrun_tokens.json` |

The dry-run database sits **outside the repository** so synthetic rows cannot
reach the tree that feeds the reliability evidence, which is what line 73 above
requires. `pilot` mode refuses to start against a tokens file whose name
contains `dryrun`. A bare `restart` will not cross the boundary: it reads the
mode the running stack recorded and refuses if you name a different one.

Logs and backups also live outside the tree, at
`D:\Projects\_skill-extraction-ops\`. `.gitignore` covers `logs/*.log` but not
`logs/*.log.1`, so an in-repo log would become a tracked file on a public repo
at the first rotation. `--log-level warning` on uvicorn is load-bearing for the
same reason: at `info` the access log writes every annotator's `?token=…`.

### Switching from the dry run to the real pilot

```powershell
.\run_annotation_stack.ps1 stop
.\run_annotation_stack.ps1 uninstall
# delete the dry-run tokens, then mint live ones (make_tokens will not overwrite)
Remove-Item api_keys\dryrun_tokens.json, api_keys\DRYRUN_LINKS.txt
.venv\Scripts\python.exe -m annotation_server.make_tokens --base-url <UI-URL> `
    --split pilot_100 --practice-split practice_10 `
    --out api_keys\annotation_tokens.json > api_keys\LINKS.txt
.\run_annotation_stack.ps1 start   -Mode pilot -Init     # -Init creates the DB
.\run_annotation_stack.ps1 install -Mode pilot
.\run_annotation_stack.ps1 status  -Mode pilot
```

`-Init` is required exactly once. Without it a missing database is **fatal**,
because `api.py:50` + `store.py:51` would otherwise create a valid empty one at
a mistyped path and the annotators would simply see no saved work. `start -Init`
prints the empty database's sha256 — keep it; it is the evidence the pilot
started from an empty file.

### What the reboot setup must prove

In order, and none of them is `/health`:

1. **The tasks exist and are healthy.** `status` shows both as `Running` with
   `result=267009`. That constant means "still running"; a `0` means the action
   already exited, i.e. the supervisor died.
2. **The right database is in use.** `status` prints the path each supervisor
   actually resolved, read from the runtime-state file it wrote at startup —
   a running process's environment cannot be read from outside, so without that
   file this would be a guess.
3. **Authentication works.** `/admin/progress` returns 200 with the expected
   annotators. A 401 here beside a 200 from `/health` means the token table is
   empty (`auth.py:39-42`) and all three annotators are locked out.
4. **Rows advance.** The `db rows` line and the file's `last written` time move
   as annotators work. `/health` returns a hardcoded `{"ok":true}` without
   touching the database, the tokens or the samples, so it proves only that a
   process is answering.
5. **No work bypassed the collector.** `status` goes red if any `*.json` sits
   directly in `DATA/gold_standard/annotations/`. Those are judgements written
   by the UI's local-file fallback, and `gold_standard/iaa.py` reads that exact
   directory. The stack now refuses to start while any are present.

### Verified 2026-11-17, and what was not

Measured: the tasks register without elevation and the XML carries the intended
settings (`Delay PT20S`/`PT50S`, `ExecutionTimeLimit PT0S`,
`DisallowStartIfOnBatteries false`, `StopIfGoingOnBatteries false`,
`RestartOnFailure 3 × PT1M`, `MultipleInstancesPolicy IgnoreNew`); starting the
stack **through the scheduled tasks** binds both ports and serves the durable
database; `/admin/progress` returns all three annotators and 30 rows; a write
through the API reaches the durable file (idempotent re-save held `annotations`
at 30 and took `events` 60 → 61, with the `#` in the sentence id intact);
`backup` produced a snapshot with matching counts; all 297 tests still pass.

**An actual reboot was not performed.** The logon trigger is registered and
verified only by starting the tasks by hand. Confirm it after the next real
restart with `status`.

Five things this does **not** cover, all measured on this machine:

1. **A reboot nobody signs in after.** There is no autologon, and Windows Update
   active hours are 09:00→01:00, so an unattended restart can land between
   01:00 and 09:00 and leave the stack down until someone signs in. Sign in and
   run `status` before each annotation day. (True no-logon survival needs an
   `-AtStartup` task registered from an elevated shell, running as SYSTEM.)
2. **Battery.** The active power scheme never sleeps on mains but sleeps after
   180 s on battery, and this machine's only sleep state disconnects the
   network — so an unplugged laptop goes unreachable while the processes stay
   perfectly healthy. Keep it on mains.
3. **LAN reachability of `:8501`.** The firewall rule that matches today is
   scoped to `C:\Python313\python.exe` on the *Public* profile. Rebuilding a
   venv, or joining a network Windows classifies as Private, breaks it — and a
   hidden scheduled task has no desktop on which to show the firewall prompt.
   Test from a second device, not from the host.
4. **A hung-but-listening process.** Nothing here detects one; `status` would
   show everything green. The in-script supervisor restarts a child that
   *exits*, and gives up after five fast exits so a broken config fails visibly
   instead of flapping.
5. **Disk failure.** `D:` and `C:` are partitions of one physical NVMe. This
   survives a reboot, not a dead disk, and the database sits in a gitignored
   directory — which is exactly what `git clean -xdf` removes. Run `backup`
   daily during the pilot.

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
