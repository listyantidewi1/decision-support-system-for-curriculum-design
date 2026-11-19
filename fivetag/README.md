# fivetag — the extraction and typing models

This folder holds every model condition the approved proposal compares
(§3.3.1–3.3.3), written against the five tags of the annotation guideline
(SH, SS, KN, TT, WB).

**Status, 19 November 2026: the code is built and tested. There are no results.**
The experts' gold standard does not exist yet, so nothing here has been trained
or scored on real data. Every number printed so far comes from stand-in data and
only shows that the code runs.

## What each piece is

| Code | Condition in the proposal | File | State |
|---|---|---|---|
| E0 | Rules and dictionary | `e0_rules.py`, `lexicons/v1/` | works |
| E1 | Multilingual encoder + BIO | `encoder.py` | training loop works; real training needs a GPU and the gold |
| E2 | E1 after domain-adaptive pretraining | `dapt.py`, then `encoder.py` | same |
| E3 | GLiNER open-type extraction | `gliner_adapter.py` | works without training on the real model (run 19 November); fine-tuning needs the gold |
| E4 | Prompted LLM, JSON schema, no retrieval | `llm.py` | works; can be scored from runs already paid for |
| E5 | LLM + retrieval + deterministic verifier | `llm.py` | works with a fake model; needs gold train sentences to retrieve from |
| E6 | NLP first, LLM only when a trigger fires | `router.py` | works; thresholds not tuned |
| D1 | Lexicon–Regex–Priority classifier | `typing_d.py` | works |
| D2 | Taxonomy-anchored context classifier | `typing_d.py` | works |
| D3 | Consensus of D1 and D2, else CONFLICT | `typing_d.py` | works |

All extractors return the same thing (a list of spans with character positions),
so all of them are scored by the one shared engine in `gold_standard/metrics.py`
and can be compared pair by pair.

Supporting files: `base.py` (the span type), `lexicon.py` (word lists with a
version and a fingerprint), `evaluate.py` (scoring), `benchmark.py` (one table
for all conditions), `synthetic.py` (invented test sentences), `tiny.py` (a toy
model for testing the training loop).

## Commands (all free, none uses the network)

```powershell
.venv-ml\Scripts\python.exe -m fivetag.evaluate dev --errors 10   # E0 on the development sentences
.venv-ml\Scripts\python.exe -m fivetag.typing_d                   # D1, D2, D3 on the same spans
.venv-ml\Scripts\python.exe -m fivetag.benchmark synthetic        # E0, toy E1, E6 on invented sentences
.venv-ml\Scripts\python.exe tests\test_fivetag.py                 # 62 tests
```

## What was measured on 19 November (development checks, not results)

Expected spans found, on the SkillSpan sentences you reviewed (74 and 120 spans):

| | batch 1 | batch 2 |
|---|---|---|
| E0 rules | 41 | 106 |
| E4 GPT-5.4 mini (stored run, prompt v0) | 47 | 103 |
| E4 Claude Haiku 5.5 | 51 | 113 |
| E4 Qwen 3.8 Flash | 57 | 108 |
| E6 router, E0 → Haiku | 50 | 115 |
| E3 GLiNER, no training, threshold 0.5 | 9 | 56 |
| E3 GLiNER, no training, threshold 0.3 | 13 | 73 |

How to read this:

- E0 looks strong on batch 2 because that batch is mostly technology names and
  the dictionary was written with these very sentences in view. On unseen
  postings it will be weaker. It finds every listed technology and misses most
  soft-skill phrases, which is what a dictionary baseline is expected to do.
- The router sent about two thirds of these sentences to the LLM (28/42 and
  36/58). That is too many to save money. Its thresholds are starting values
  and have to be set on the dev split.
- The LLM outputs were valid JSON almost always (98–100%) and about 1% of
  returned spans were not in the sentence.
- GLiNER without training is, in effect, a technology tagger: it found 54 of
  the 76 technology names in batch 2 (62 at the lower threshold) and almost
  none of the action phrases, soft skills or work behaviours. It was built to
  find named things, and the five-tag guideline asks for whole phrases with
  their verb. Whether fine-tuning on the gold changes that is what the real E3
  condition will measure. It ran at about 5 sentences per second on the CPU.

D1, D2, D3 on the same reference spans: they are right on 96–99% of the spans
they decide, abstain on 6–30% (phrases not in any list), and D3 flags 1–3% as
CONFLICT.

## Design choices

Confirmed by the researcher on 19 November 2026: choices 1 and 2 as built, and
choice 4 answered "yes, add the ESCO and O*NET lists". Choice 3 is still open.
All four are still to be mentioned to the promotor.

1. **One tagging layer per tag (E1/E2).** The guideline lets a technology sit
   inside an action phrase ("Configure Kubernetes clusters", rule 6.9). Ordinary
   BIO tagging cannot represent that, so the encoder has five small taggers on
   one shared body, one per tag. It is still "encoder + BIO" as the proposal
   says, but it is a design decision worth one sentence in chapter 3.
2. **How E5 finds its examples.** It picks the most similar annotated training
   sentences by character n-gram TF-IDF: simple, repeatable, no extra model. An
   embedding-based search is the alternative.
3. **Where a D3 conflict goes.** The proposal says to the LLM or to a human
   "according to its risk level". The risk rule is not defined anywhere yet, so
   the code reports the conflict and does not choose. **Open.**
4. **What counts as the taxonomy for D2.** Today it is the word lists in
   `lexicons/v1/`. The plan also names ESCO and O*NET lists. **Decided: add
   them.** Not done yet: the lists are not on this computer. They will go into a
   new lexicon version (`v2`) as extra anchor sources for D2 and extra
   dictionary entries for E0, so `v1` results stay reproducible.

Also yours: the word lists themselves. They are plain text files in
`lexicons/v1/`, one entry per line; you can read and correct them. Once real
results exist, a change to the lists should go into a new folder (`v2`) so old
results stay reproducible.

## Rehearsal before the gold exists: E1 on SkillSpan, on Kaggle

SkillSpan is public, expert-annotated job-posting text that is already in this
repository. Training E1 on it is a dress rehearsal: real text, the real
XLM-RoBERTa encoder, a real GPU. It is **not a study result**: SkillSpan is
English only and has two labels (skill, knowledge) defined by its own
guideline, so it says nothing about five-tag accuracy on Indonesian postings.

On kaggle.com:

1. Create → New Notebook.
2. In the right-hand panel, under *Session options*: **Accelerator: GPU T4**,
   **Internet: On** (Kaggle asks you to verify a phone number once for this).
3. Paste this into the first cell and run it:

   ```
   !git clone --depth 1 https://github.com/listyantidewi1/decision-support-system-for-curriculum-design.git repo
   %cd repo
   !python -m fivetag.skillspan_rehearsal --model xlm-roberta-base --epochs 6 --out /kaggle/working/e1_skillspan
   ```

4. When it finishes it prints about ten lines starting with `REHEARSAL`. Copy
   them, or download `skillspan_rehearsal_report.json` from the notebook's
   *Output* folder.

What to compare it with: the SkillSpan paper (Zhang et al., 2022, Table 5)
reports a combined span-F1 of about 57–60 for BERT-sized models on its own
files. The copy in this repository is reformatted, so a score in the same region
means the pipeline is sound; a much lower one means something is wrong and must
be found before the real experiment.

Checked locally on 19 November: with the real XLM-RoBERTa tokenizer, 2,161 of
the 2,162 gold spans in the SkillSpan dev file survive the conversion to tags and
back unchanged.

## When the gold standard exists

1. `python -m gold_standard.materialise …` writes `train.jsonl` and `dev.jsonl`
   (runbook step 8). The test part stays sealed.
2. Train E1 on a GPU (Kaggle):
   `python -m fivetag.encoder train --train train.jsonl --dev dev.jsonl --out runs/e1`
3. For E2, first adapt the encoder to posting language, then train the same way:
   `python -m fivetag.dapt --sentences corpus_sentences.txt --exclude test_sentences.txt --out runs/dapt`
   then `python -m fivetag.encoder train --model runs/dapt --condition E2 …`
4. Install GLiNER and run E3; run E4 and E5 with a spending cap; set the
   router's thresholds on dev; then one table with the paired tests
   (`fivetag.benchmark.compare`).
5. Choose the configuration on dev. Open the test set once.

## Rules this code keeps

- The sealed test split is never an input: the encoder, the retrieval pool and
  the DAPT script refuse or remove test sentences.
- Nothing is written under `DATA/gold_standard/`; model output goes to
  `DATA/fivetag/`, which stays on this computer (gitignored).
- No paid model call happens unless a command is given a spending cap; scoring
  and testing never call a model.
- Synthetic sentences are marked as synthetic in every row and every printout.
