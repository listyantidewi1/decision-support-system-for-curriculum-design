# Runbook — what to do when the annotators have finished

Written for the researcher working alone. Every command is typed in a terminal
opened in the project folder (in VS Code: **Terminal → New Terminal**).

The same steps serve the pilot (`pilot_100`) and, later, the main round
(`main_1200`). Replace the split name where it appears.

All of this was rehearsed end to end on 2026-11-19 with three language models
standing in for the annotators. That rehearsal is not a result; it only shows
the steps work. Its files are in `DATA/gold_standard/_synthetic/rehearsal/` if
you want to see what each step produces.

---

## Before anything: is the go/no-go rule approved?

It is: you approved `docs/experiments/PREREG_pilot_gate.md` on 2026-11-19 and it
is registered. If your promotor wants different thresholds, change them **before
the first expert saves an annotation**, in that document and in
`gold_standard/gate.py` together (a test keeps the two in step). After the first
real annotation the rule is not to be touched.

## Step 1 — Download the annotators' work

Check that all three have finished. Replace the address with your collector's:

```powershell
curl.exe -H "Authorization: Bearer YOUR-ADMIN-TOKEN" "https://YOUR-COLLECTOR/admin/progress?split=pilot_100"
```

Each annotator should show 100 sentences. Then download and unpack:

```powershell
curl.exe -H "Authorization: Bearer YOUR-ADMIN-TOKEN" "https://YOUR-COLLECTOR/admin/export?split=pilot_100" -o annotations.zip
Expand-Archive annotations.zip -DestinationPath DATA\gold_standard\annotations\
```

The admin token is the entry marked `"admin": true` in
`api_keys\annotation_tokens.json`. Never paste it into a chat or an email.

You should now have three files: `pilot_100_AV.json`, `pilot_100_AI.json`,
`pilot_100_AD.json`. Keep `annotations.zip` somewhere safe as the original.

## Step 2 — One command

```powershell
.venv-ml\Scripts\python.exe -m gold_standard.after_annotation --split pilot_100
```

It takes a minute or two and writes four things into `DATA\gold_standard\iaa\`:

| File | What it is |
|---|---|
| `iaa_pilot_100_….md` | the agreement report, in full |
| `gate_pilot_100_….json` | the go/no-go verdict (also printed on screen) |
| `calibration_pilot_100_….html` | the page for your session with the annotators |
| `adjudication_pilot_100_…_WORKSHEET.xlsx` | the spreadsheet for adjudication |

If the command stops with a message starting "refusing", read it: it is
protecting you from mixing test data into the real results. The message says
what to change.

## Step 3 — Read the verdict

| Verdict | What it means | What you do |
|---|---|---|
| **GO** | agreement is fine on all three measures | calibration session, freeze the guideline, start the main 1,200 |
| **RECALIBRATE** | at least one measure is borderline | calibration session, revise the guideline wording, give the annotators 30 fresh sentences, run this check again on those |
| **STOP** | at least one measure is low | do not start the main 1,200; talk to your promotor |
| **NOT READY** | someone has not finished | wait, then start again from step 1 |

With 100 sentences every figure is uncertain by about ±0.15. The pilot tells you
where the guideline is read differently. It is not the reliability figure for
the dissertation; that comes from the main 1,200.

## Step 4 — The calibration session

Open the `calibration_….html` file in a browser (double-click it). It has:

1. the agreement figures and the verdict;
2. suggested points to discuss, worked out from the data;
3. which tags were confused with which, and how boundaries differed;
4. a column per annotator: how many spans, which tags, what they marked alone;
5. every disputed sentence, with each annotator's spans side by side.

Bring these points as well; they came up while testing the language models and
your annotators will meet them too:

- the six boundary conventions C1–C6 (listed in `labeling/skillspan_guideline.py`
  and in `labeling/README.md`);
- your ruling that nouns governed by one verb are KNOWLEDGE, while activity
  nouns such as "unit testing" stay SKILL_HARD;
- whether a slash separates list items ("Angular/React.js");
- guideline example 2.3, whose span is not one continuous piece of text.

Whatever you decide, write it into the guideline and give it a new version
number before the main round.

## Step 5 — Adjudication

Open the `…_WORKSHEET.xlsx` file. It has two sheets to fill in and a "read me"
sheet that explains the columns.

- **sentences**: one row where relevance or the five-tag answer was not
  unanimous. Write the decision in the two FINAL columns.
- **spans**: one row per disputed span. In DECISION write `GOLD`, `AMBIGUOUS` or
  `EXCLUDE`. For `GOLD`, copy the exact span into "final text" and write its tag.
  To accept the suggestion shown in the row, write `GOLD` and leave the text and
  tag empty.

Spans that all three annotators marked identically are already gold and are not
listed.

Save the file, then:

```powershell
.venv-ml\Scripts\python.exe -m gold_standard.adjudicate finalise --split pilot_100 --worksheet "DATA\gold_standard\iaa\adjudication_pilot_100_…_WORKSHEET.xlsx" --version 1.0
```

If anything is unresolved or inconsistent it stops, writes nothing, and lists
every row that needs attention (for example a final text that is not in the
sentence, or a sentence marked NON_RELEVANT that has a gold span). Fix those
rows and run it again.

When it succeeds, the gold standard is in
`DATA\gold_standard\gold\pilot_100\v1.0\`, with a `MANIFEST.json` recording how
many spans came from full agreement, from adjudication and from accepted
suggestions. A version is written once; to change something, use a new version
number.

If you want every blank row to take the two-out-of-three answer automatically,
add `--accept-suggestions`. That is a method choice: it is recorded in the
manifest and you would report it in the dissertation.

## Step 6 — Compare the LLMs with the experts

The three LLMs already labelled the same 100 pilot sentences (stored where the
annotators cannot see them). Now that a gold version exists, one command scores
them against it. It costs nothing: it re-reads the stored answers.

```powershell
.venv-ml\Scripts\python.exe -m labeling.versus_gold DATA\labeling\runs\pilot_100_f342a899 --gold-dir DATA\gold_standard\gold\pilot_100\v1.0
```

It prints one line per model and two for the ensemble, and saves the full
result next to the run. The line to read first is **automatic acceptance**: of
the spans all three models agreed on, how many are in the experts' gold. That
number decides whether the ensemble may label the wider corpus without a human
checking each span.

It refuses to run on sentences of a sealed test part: the test set is never
used to compare or tune the LLMs.

## Step 7 — After a GO: the main 1,200

1. Draw the main sample (this never changes the pilot sample; it refuses if it
   would):
   ```powershell
   .venv-ml\Scripts\python.exe -m gold_standard.sample --main
   ```
2. Redeploy the collector so it serves the new sentences, and create links for
   `main_1200` (see `annotation_server/README.md`).
3. When the annotators finish, repeat steps 1–5 with `main_1200` in place of
   `pilot_100`. The agreement figures from that run are the ones you report.

## Step 8 — Split, seal the test part, hand train and dev to the models

The split is made from the sentences' metadata only, so it can be made as soon
as the 1,200 are drawn, before any annotation (Claude does this with you; it
runs `gold_standard.split` for the three regimes).

When the gold of the 1,200 is final (step 5 for `main_1200`), run once per
split:

```powershell
.venv-ml\Scripts\python.exe -m gold_standard.materialise --split-id grouped_random_v1 --gold-dir DATA\gold_standard\gold\main_1200\v1.0
```

It does two things. It writes the test labels into the sealed folder, from which
they are read only by the final scoring, once. And it writes `train.jsonl` and
`dev.jsonl` under `DATA\fivetag\gold\`, which is what the extraction models
learn from. The test sentences are never written there; the command checks this
and stops if it would happen.

If the gold and the split were made from different samples it stops and writes
nothing.

---

## If something goes wrong

| What you see | Likely cause |
|---|---|
| "Need >=2 annotator files" | the zip was unpacked somewhere else, or the files have different names |
| "annotated sentence ids are not in the … sample" | the annotation files belong to another split, or the sample file was replaced |
| "this worksheet was made from different annotation files" | the annotation files changed after the worksheet was made; make a fresh worksheet |
| "already exists" at finalise | that version is already written; choose a new `--version` |
| "REFUSED: … sentences are in a sealed test part" | the comparison in step 6 was pointed at test sentences; use the pilot or the dev part |
| "REFUSED: … gold sentences are not in split …" | the gold version and the split belong to different samples |
| "refusing to overwrite pilot_100.parquet" | the sentence pool was rebuilt; do not redraw a sample people have annotated |
