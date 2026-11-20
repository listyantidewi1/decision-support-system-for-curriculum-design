# Experiment log

Every experiment run in this study is recorded here, at the time it is run:
what was asked, on which data, with which settings, what came out, what it
cost, where the output files are, and what it does **not** show.

Owner: Listyanti Dewi Astuti. Started 2026-11-19. Entries before that date were
reconstructed on 2026-11-19 from the stored result files; every number in them
was re-read from those files, not copied from notes.

## How to read the status of an entry

| Status | Meaning | May it be reported as a finding? |
|---|---|---|
| **DEVELOPMENT** | Run on development data to build or choose something (a prompt, a rule, a model). | No. It may be reported as *how the instrument was developed*. |
| **REHEARSAL** | Run on stand-in data to prove a procedure works before the real data exist. | No. |
| **ENGINEERING** | Run on real corpus text with a component whose accuracy is not yet measured. | No. |
| **CONFIRMATORY** | Run on the gold standard or the sealed test set under a rule registered beforehand. | Yes. |

**As of 2026-11-19 there is no CONFIRMATORY entry.** The gold standard does not
exist yet. Nothing below is a result of the study.

## Rules for adding an entry

1. Add the entry on the day the experiment is run, before interpreting it.
2. Give it the next number. Never renumber, never delete; a superseded entry is
   marked superseded and kept.
3. Copy numbers from the output files, and name the files.
4. State the cost, the versions (prompt, lexicon, rules, model) and the seed.
5. Write the limits before the conclusion.

## Settings shared by the LLM experiments

- Access: OpenRouter only. Models and list prices used for the spending guard
  (USD per million tokens, input / output): `openai/gpt-5.4-mini` 0.75 / 4.50;
  `anthropic/claude-haiku-5.5` 0.10 / 0.50; `qwen/qwen3.8-flash` 0.15 / 0.47.
- Temperature 0; JSON output requested; hidden reasoning switched off
  (`reasoning: {"effort": "none"}`) from EXP-002b onwards.
- Prompt versions (`labeling/prompts.py`; each has a frozen SHA-256):
  `v0-guideline-only` (the condensed guideline B v3.0, no examples),
  `v1-sh-action-and-noun-lists`, `v2-boundary-conventions`.
- Every model answer passes the deterministic verifier (`labeling/verify.py`)
  before it is counted, unless the entry says "as returned".
- "Charged" is the larger of the provider's reported cost and the list-price
  estimate, so it is an upper estimate of what was spent.

## Development data sets

None of these is the gold standard, and none comes from the study's corpus
except where stated.

| Name | What it is | Size | Reference made by |
|---|---|---|---|
| Conformance set v0 | Invented sentences, one or more per guideline rule (`labeling/conformance.py`) | 109 items, 135 expected spans | Written by the AI assistant from the guideline; reviewed and approved as a whole by the researcher, 2026-11-18 |
| SkillSpan-G batch 1 | Real sentences from the public SkillSpan dev file, annotated under guideline B v3.0 with conventions C1–C6 | 42 items, 74 expected spans | Drafted by the AI assistant; reviewed and approved by the researcher, 2026-11-18 |
| SkillSpan-G batch 2 | Same, different sentences | 58 items, 120 expected spans | Same; approved 2026-11-19 |
| SkillSpan probe | 100 seeded SkillSpan dev sentences with SkillSpan's own two labels (seed 20261118) | 100 sentences, 160 skill spans | SkillSpan's annotators (Zhang et al., 2022) |
| SkillSpan (full) | The released SkillSpan files | 4,800 / 3,174 / 3,569 sentences | SkillSpan's annotators |
| Synthetic rows | Template-generated sentences (`fivetag/synthetic.py`) | any size | By construction |
| `pilot_100` | 100 sentences of the study's corpus, drawn for the pilot annotation | 100 | **No reference yet**: the experts have not annotated |
| Pool sample | Postings sampled from the study's sentence pool (seed 7) | about 20,000 sentences, 581 postings | None |

The three reference sets written with the AI assistant are **partial**: they
list only the spans the guideline settles. A model's extra span is therefore not
necessarily wrong, so precision is not defined on them. What is reported is
"expected spans found" and "items passed" (an item passes when every expected
span is found and no forbidden span is produced).

---

## EXP-001 — Learning curve for the size of the gold standard

- **Date / status:** 2026-11-16 · DEVELOPMENT
- **Question:** how many annotated sentences are worth paying for?
- **Record:** `docs/experiments/PREREG_gold_standard_size.md` (the registration
  and its evidence) and `docs/experiments/learning_curve/` (script and outputs).
  The figures are in those files and are not repeated here.

## EXP-002 — First run of the three-model ensemble (smoke test)

- **Date / status:** 2026-11-18 · DEVELOPMENT
- **Question:** does the ensemble run end to end, and do the models return usable output?
- **Data:** `practice_10` (10 corpus sentences, the annotators' warm-up split).
- **Method:** prompt v0, three models, verifier, alignment, consensus.

| Run | Hidden reasoning | Candidates | Unanimous | Majority | Charged |
|---|---|---|---|---|---|
| a. `practice_10_d03786b2` | provider default | 46 | 7 | 10 | $0.0280 |
| b. `practice_10_4a57334b` | off | 51 | 15 | 11 | $0.0209 |

- **Finding for the method:** with hidden reasoning left on, one model spent its
  token allowance on reasoning and returned empty answers for some sentences.
  Hidden reasoning is off in every later run, and an empty answer counts as an
  invalid output.
- **Limits:** ten sentences; no reference.
- **Files:** `DATA/labeling/runs/practice_10_*`.

## EXP-003 — Do the models keep the verb of a skill phrase? (SkillSpan probe)

- **Date / status:** 2026-11-18 · DEVELOPMENT
- **Question:** the researcher observed that the models cut the operational verb
  off hard-skill spans. How often?
- **Data:** SkillSpan probe (100 sentences, 160 human skill spans).
- **Measure:** of the human skill spans a model overlapped with a skill-type
  span, the share whose head (first word) it kept.

| Prompt | GPT-5.4 mini | Claude Haiku 5.5 | Qwen 3.8 Flash | Charged |
|---|---|---|---|---|
| v0 | 131/143 = 91.6% | 98/111 = 88.3% | 121/130 = 93.1% | $0.1954 |
| v1 | 122/141 = 86.5% | 102/116 = 87.9% | 110/119 = 92.4% | $0.2259 |

- **Reading:** the head is kept about nine times in ten under v0; v1, written to
  improve this, did not improve it.
- **Limits:** SkillSpan's skill label is not the study's SKILL_HARD.
- **Files:** `DATA/labeling/runs/skillspan_probe_*/skillspan_score.json`.

## EXP-004 — Do the models follow each guideline rule? (conformance set)

- **Date / status:** 2026-11-18 · DEVELOPMENT
- **Data:** conformance set v0 (109 items, 135 expected spans).

| Prompt | GPT items / spans | Haiku items / spans | Qwen items / spans | Unanimous candidates | Charged |
|---|---|---|---|---|---|
| v0 | 106 / 130 | 103 / 125 | 108 / 134 | 125/140 | $0.2009 |
| v1 | 107 / 134 | 106 / 130 | 109 / 135 | 130/142 | $0.2332 |
| v2 | 106 / 134 | 105 / 130 | 108 / 134 | 130/144 | $0.2793 |

- **Limits:** invented sentences, most of them the guideline's own examples; the
  set is close to its ceiling and cannot separate the prompts.
- **Files:** `DATA/labeling/runs/conformance_v0_*/conformance_score.json`.

## EXP-005 — Prompt versions on real sentences, batch 1

- **Date / status:** 2026-11-18 · DEVELOPMENT. **Batch 1 was used to write
  prompts v1 and v2**, so it cannot judge them.
- **Data:** SkillSpan-G batch 1 (42 items, 74 expected spans).

| Prompt | GPT found / items | Haiku found / items | Qwen found / items | Charged |
|---|---|---|---|---|
| v0 | 47 / 19 | 51 / 22 | 57 / 27 | $0.0817 |
| v1 | 50 / 21 | 43 / 20 | 54 / 24 | $0.0945 |
| v2 | 55 / 25 | 55 / 28 | 57 / 28 | $0.1126 |

- **Files:** `DATA/labeling/runs/skillspan_guideline_{f342a899,6c256264,079f7d73}`.

## EXP-006 — Prompt versions on unseen sentences, batch 2

- **Date / status:** 2026-11-18 (sentences approved 2026-11-19) · DEVELOPMENT
- **Question:** is v2 better than v0 on sentences that were not used to write it?
- **Data:** SkillSpan-G batch 2 (58 items, 120 expected spans).

| Prompt | GPT found / items | Haiku found / items | Qwen found / items | Charged |
|---|---|---|---|---|
| v0 | 101 / 41 | 111 / 50 | 106 / 46 | $0.1127 |
| v1 | 105 / 43 | 109 / 48 | 107 / 46 | $0.1305 |
| v2 | 98 / 40 | 113 / 52 | 112 / 50 | $0.1547 |

Paired comparison v0 against v2, exact McNemar test on the expected spans
(found under one prompt only):

| Model | Batch 2: only v0 / only v2 / p | Both batches (194 spans): only v0 / only v2 / p |
|---|---|---|
| GPT-5.4 mini | 12 / 9 / 0.664 | 16 / 21 / 0.511 |
| Claude Haiku 5.5 | 3 / 5 / 0.727 | 12 / 18 / 0.362 |
| Qwen 3.8 Flash | 0 / 6 / 0.031 | 4 / 10 / 0.180 |

- **Conclusion:** v2 is not shown to be better than v0. **Decision: v0 stays
  the default prompt.** No prompt is final until it is chosen on the gold dev
  split.
- **Limits:** 58 items; three comparisons, uncorrected for multiplicity; the
  reference is partial.
- **Files:** `DATA/labeling/runs/skillspan_guideline_b2_*`.

## EXP-007 — The ensemble on the pilot sentences (stored, blind)

- **Date / status:** 2026-11-19 · REHEARSAL of the labelling run; its output is
  the LLM side of the comparison the pilot exists to make.
- **Data:** `pilot_100`. **No reference exists yet.**
- **Method:** prompt v0, three models. Run `pilot_100_f342a899`.
- **Result:** 341 candidate spans; 135 unanimous (39.6%), 89 majority (26.1%),
  12 boundary disagreements. Charged $0.2029.
- **Blindness:** the output is under `DATA/labeling/`, never under
  `DATA/gold_standard/`; the annotators do not see it.
- **What it will be used for:** `labeling.versus_gold` once the experts' gold
  exists (runbook step 6).

## EXP-008 — Rehearsal of the post-annotation procedure

- **Date / status:** 2026-11-19 · REHEARSAL, synthetic annotators
- **Method:** the three LLMs stood in for the three annotators on `pilot_100`
  (`labeling/rehearsal.py`); then the agreement report, the registered go/no-go
  gate, the calibration page, the adjudication worksheet and the gold
  finalisation were run in order.
- **Result:** every step ran; the gate printed GO. The agreement figures of
  this rehearsal describe three LLMs, not experts, and are not reported.
- **Files:** `DATA/gold_standard/_synthetic/rehearsal/pilot_100/` (quarantined;
  marked SYNTHETIC_DO_NOT_CITE).

## EXP-009 — Rehearsal of "LLMs against the experts' gold"

- **Date / status:** 2026-11-19 · REHEARSAL, synthetic gold
- **Method:** `labeling.versus_gold` on run EXP-007 against the rehearsal gold of
  EXP-008 (100 sentences, 224 gold spans).
- **Result:** the command runs and labels every line SYNTHETIC. Strict F1:
  GPT 0.778, Haiku 0.822, Qwen 0.849; unanimous-only ensemble precision 1.000,
  recall 0.603.
- **Limit that voids the numbers:** the rehearsal gold was made from the same
  three LLMs, so the comparison is circular. It shows the command works and
  nothing else.

## EXP-010 — The rule baseline E0 on the development sets

- **Date / status:** 2026-11-19 · DEVELOPMENT
- **Method:** `fivetag.evaluate dev`; rules `e0-rules-v1`, lexicon v1
  (SHA-256 `dadafcd01247…`). No model, no cost.

| Set | Expected spans found | Items passed | By tag (found/expected) |
|---|---|---|---|
| Conformance | 128/135 | 106/109 | SH 37/37, SS 9/9, KN 18/25, TT 51/51, WB 13/13 |
| SkillSpan-G batch 1 | 41/74 | 16/42 | SH 13/20, SS 8/30, KN 3/3, TT 13/13, WB 4/8 |
| SkillSpan-G batch 2 | 106/120 | 49/58 | SH 16/24, SS 2/6, KN 8/8, TT 76/76, WB 4/6 |

- **Limit that matters:** the lexicon and rules were written with these very
  sets in view. The figures are optimistic and say nothing about unseen postings.

## EXP-011 — The deterministic classifiers D1, D2, D3 on reference spans

- **Date / status:** 2026-11-19 · DEVELOPMENT
- **Method:** each reference span is given with its boundaries; the model must
  return its tag. `typing-v1`, lexicon v1.

| Set (spans) | Model | Macro-F1 | Correct of decided | Abstain | Conflict |
|---|---|---|---|---|---|
| Conformance (135) | D1 | 0.951 | 98.4% | 6.7% | – |
| | D2 | 0.953 | 98.4% | 5.9% | – |
| | D3 | 0.951 | 98.4% | 5.9% | 0.7% |
| Batch 1 (74) | D1 | 0.843 | 96.2% | 29.7% | – |
| | D2 | 0.855 | 96.2% | 29.7% | – |
| | D3 | 0.837 | 96.1% | 28.4% | 2.7% |
| Batch 2 (120) | D1 | 0.881 | 99.1% | 9.2% | – |
| | D2 | 0.882 | 99.1% | 8.3% | – |
| | D3 | 0.881 | 99.1% | 8.3% | 0.8% |

- **Limits:** same as EXP-010 (the word lists were written with these sets in view).

## EXP-012 — LLM output as returned, after the verifier, and through the router

- **Date / status:** 2026-11-19 · DEVELOPMENT. Free: re-reads the answers of EXP-005/006 (prompt v0).
- **Conditions:** E4 = spans as the model returned them; "+verifier" = the same
  answers after the deterministic verifier; E6 = E0 first, the model's verified
  answer only for sentences where a router trigger fires (`router-v1`, untuned thresholds).

Expected spans found (batch 1 of 74 / batch 2 of 120):

| Model | E4 as returned | + verifier | E6 router | Sentences escalated |
|---|---|---|---|---|
| GPT-5.4 mini | 47 / 103 | 47 / 101 | 44 / 109 | 28 of 42 / 36 of 58 |
| Claude Haiku 5.5 | 51 / 113 | 51 / 111 | 50 / 115 | same |
| Qwen 3.8 Flash | 57 / 108 | 57 / 106 | 50 / 110 | same |
| E0 alone | 41 / 106 | | | |

- Output structure: valid JSON in 98–100% of answers; 0–1.2% of returned spans
  were not in the sentence.
- **Open observation:** on batch 2 the verifier removed two correct spans for
  each model. Not yet investigated.
- **Limits:** the router's thresholds are starting values; it escalated about
  two thirds of the sentences.

## EXP-013 — Wiring check on synthetic sentences

- **Date / status:** 2026-11-19 · REHEARSAL
- **Method:** `fivetag.benchmark synthetic`: 600 synthetic training sentences,
  200 for scoring; E0, a tiny randomly initialised encoder as E1, and E6 with an
  oracle in place of the LLM.
- **Result (last run of the day):** strict F1 0.789 (E0), 0.926 (toy E1), 0.912 (E6 with oracle). The
  toy encoder's dev F1 rose from 0 to 0.885 in eight epochs in a smoke run.
- **What it shows:** the training loop learns and the scoring is wired
  correctly. The numbers have no other meaning.

## EXP-014 — GLiNER without training (E3, zero-shot)

- **Date / status:** 2026-11-19 · DEVELOPMENT
- **Method:** `urchade/gliner_multi-v2.1`, `gliner` 0.2.30, label descriptions
  `gliner-labels-v1`, CPU, about 5 sentences per second.

| Threshold | Conformance (135) | Batch 1 (74) | Batch 2 (120) | Technology spans, batch 2 |
|---|---|---|---|---|
| 0.5 | 52 | 9 | 56 | 54/76 |
| 0.3 | 63 | 13 | 73 | 62/76 |

- **Reading:** untrained GLiNER behaves as a technology tagger. It finds almost
  no action phrases (batch 2: 1/24 at 0.5, 8/24 at 0.3) or soft skills.
- **Limits:** one set of label descriptions; no fine-tuning.
- **Files:** `DATA/fivetag/rehearsal/e3_dev_{0.5,0.3}.json`.

## EXP-015 — Training rehearsal of E1 on SkillSpan (Kaggle)

- **Date / status:** 2026-11-19 · REHEARSAL
- **Question:** does the E1 code, on a real encoder and GPU, reach the scores
  published for SkillSpan?
- **Data:** SkillSpan (full), its two labels mapped onto two of the five tag
  layers (SKILL → SH, KNOWLEDGE → KN). "Test" here is SkillSpan's own test file.
- **Method:** `fivetag.skillspan_rehearsal`; layered BIO; batch 16; body
  learning rate 3e-5, head 1e-3; seed 42; best epoch chosen on dev; Tesla T4.

| Run | Encoder | Epochs | Dev strict F1 (95% CI) | Test strict F1 (95% CI) | Test partial F1 | Test skill / knowledge | Minutes |
|---|---|---|---|---|---|---|---|
| a | xlm-roberta-base | 6 | 0.551 (0.525–0.577) | 0.557 (0.533–0.583) | 0.687 | 0.483 / 0.622 | 11.9 |
| b | bert-base-cased | 6 | 0.499 (0.472–0.526) | 0.510 (0.486–0.535) | 0.651 | 0.421 / 0.589 | 10.6 |
| c | xlm-roberta-base | 20 | 0.582 (0.556–0.606) | 0.591 (0.566–0.616) | 0.708 | 0.533 / 0.642 | 40.1 |

Published (Zhang et al., 2022, mean of five seeds, trained on 5,866 sentences
including a source that was not released): combined 59.35 (BERT-base, dev) and
59.73 (best model, test); skill 56.64 and knowledge 63.88 on test.

Dev strict F1 by epoch in run c: 0.251, 0.415, 0.513, 0.539, 0.534, 0.546,
0.544, 0.562, 0.564, 0.561, 0.560, 0.574, 0.579, 0.563, 0.574, 0.582, 0.580,
0.581, 0.578, 0.575.

Checks made to explain the gap after run a:
- not the multilingual encoder (run b is lower, not higher);
- not the data copy: against the original release (`jjzha/skillspan`), same
  sentence counts and span lists differing in 5 of 11,543 sentences;
- the number of epochs: run c closes the gap. The paper trained 20 epochs with a
  CRF layer.
- With the real XLM-RoBERTa tokenizer, 2,161 of 2,162 dev spans survive the
  conversion to tags and back.

- **Decision:** the default number of epochs was raised from 10 to 20 and early
  stopping (`patience`) was added.
- **Limits:** one seed per run; English; two labels; not the study's task.

## EXP-016 — Effect of dictionary version 2 (O*NET)

- **Date / status:** 2026-11-19 · DEVELOPMENT
- **Method:** lexicon v2 = v1 + 165 technology names derived from O*NET 30.2
  "Technology Skills" (SOC 15-12xx, hot or in-demand; rules in
  `fivetag/build_lexicon_v2.py`). E0 run with each version.
- **Result:** no change on the three development sets (same spans found, same
  items passed). On 20,000 pool sentences (seed 7): technology spans 5,353 (v1)
  and 5,528 (v2), +175 (3.3%); sentences with a technology 2,568 and 2,641.
- **Limits:** the new hits were read by the developer only (they look like
  technologies: "Windows", "EC2", "Power Automate"); no precision estimate.

## EXP-017 — First run of the competency chain

- **Date / status:** 2026-11-19 · ENGINEERING
- **Data:** pool sample, seed 7: 20,021 sentences from 581 postings.
- **Method:** `competency.pipeline e0`: evidence from E0 (lexicon v2);
  grouping `grouping-v1` (multilingual MiniLM embeddings, HDBSCAN, cohesion
  floor 0.55, no per-phrase frequency filter); deterministic statements;
  KKNI labeller and okupasi mapper; contract `contract-v1` (at least 5
  postings, grounding at least 0.80: starting values). No LLM. About 4 minutes.
- **Result:** 14,451 evidence units (SH 5,138; SS 1,874; KN 1,007; TT 5,495;
  WB 937). 7,447 of 8,019 groupable units were grouped into 528 groups: 445
  valid, 27 to split, 25 low frequency, 24 support process, 4 out of domain, 3
  mixed domain. 445 statements: 271 kandidat tinjauan, 174 eksploratif (121 for
  fewer than 5 postings, 81 with no okupasi match). 82 statements use a default
  verb. KKNI level assigned to 17 of 445. 433 statements in English, 12 in
  Indonesian.
- **An earlier attempt the same day,** with the clusterer's own "phrase seen in
  at least 2 postings" filter, kept 360 of 5,111 distinct phrases and grouped
  2,626 of 8,019 units into 31 groups. The filter was switched off for five-tag
  phrases and rarity is judged per group.
- **Limits:** the evidence is from the rule extractor, whose accuracy is
  unknown; statements inherit its mistakes; counts are from 581 postings.
- **Files:** `DATA/competency/runs/e0_pool20k/`.

## EXP-018 — Conflict routing rehearsal and adjudicator trial

- **Date / status:** 2026-11-19 · REHEARSAL. The rule was registered first
  (`PREREG_d3_conflict_routing.md`, `d3-routing-v1`).
- **Data:** the evidence of EXP-017.
- **Result:** 130 conflicts in 14,451 units (0.9%): 31 disagreements, 99 where
  one classifier abstained. Risk: 26 high, 104 others (all inside the audit
  sample of 150). Trial adjudicator `anthropic/claude-haiku-5.5`: 130 usable
  answers of 130; sided with D2 in 99, with D1 in 22, with neither in 9.
  Charged $0.0040.
- **Limits:** no human reviewed the audit sample, so the adjudicator's accuracy
  is unknown. Haiku was a trial, not a registered choice.

## EXP-019 — First trial of the LLM statement writer

- **Date / status:** 2026-11-19 · DEVELOPMENT
- **Data:** the 40 valid groups of EXP-017 with the most postings.
- **Method:** `openai/gpt-5.4-mini`; the answer must start with an operational
  verb of the table, name only technologies of the group, and reach grounding
  0.80, otherwise the rule statement is used.
- **Result:** 30 accepted, 10 rejected (6 for technologies not in the evidence,
  3 for grounding, 1 for the verb). Charged $0.0261.
- **Limits:** quality was judged by reading only; several accepted statements
  list every technology of the group.
- **Files:** `DATA/competency/runs/e0_pool20k/llm_statements_trial.json`.

## EXP-020 — Which model tags a span best? (choosing the adjudicator)

- **Date / status:** 2026-11-19 · DEVELOPMENT
- **Question:** which of the three models should adjudicate D1/D2 conflicts?
- **Data:** every reference span of the three development sets: 329 spans (135
  from the conformance set, 194 from SkillSpan-G batches 1 and 2).
- **Method:** `fivetag.conflict_routing.evaluate_adjudicator`. Each model is
  shown the phrase in its sentence and asked for one tag (SH, SS, KN, TT, WB or
  NONE); correct = the reference tag. Asked twice: on all spans with no
  candidates shown, and on the spans where D1 and D2 differ with the two
  candidates shown, as in real use. Temperature 0, hidden reasoning off,
  at most 40 output tokens. Adjudication prompt as in `conflict_routing.SYSTEM`
  on this date.

| Model | All 329 spans (95% CI) | SkillSpan-G spans only (194) | Unusable answers | Charged |
|---|---|---|---|---|
| GPT-5.4 mini | 299 = 90.9% (87%–94%) | 172 = 88.7% | 0 | $0.0512 |
| Claude Haiku 5.5 | 307 = 93.3% (90%–96%) | 180 = 92.8% | 0 | $0.0085 |
| Qwen 3.8 Flash | 315 = 95.7% (93%–97%) | 185 = 95.4% | 3 | $0.0080 |

Correct by reference tag (KN of 36, SH of 81, SS of 45, TT of 140, WB of 27):
GPT 32, 75, 30, 137, 25; Haiku 26, 77, 40, 140, 24; Qwen 33, 75, 44, 137, 26.

Paired comparison on the same 329 spans (exact McNemar; right for one model
only): GPT 9 / Qwen 25, p = 0.009; GPT 10 / Haiku 18, p = 0.185; Haiku 6 /
Qwen 14, p = 0.115. Three comparisons, uncorrected.

- **Reading:** Qwen 3.8 Flash is the most accurate here and clearly better than
  GPT-5.4 mini. Its lead over Claude Haiku 5.5 is not statistically reliable.
  The two cheapest models are also the two best.
- **Limits, which are serious:**
  - Only **4** of the 329 spans are real D1/D2 conflicts, and all three models
    got all 4 right. The comparison therefore measures general span tagging,
    not the conflict cases the adjudicator will actually see.
  - 135 of the spans are invented sentences; the references were drafted by an
    AI assistant and approved by the researcher, not annotated by the experts.
  - One prompt, one run per model.
- **Decision:** none registered. Qwen 3.8 Flash is the provisional choice, to be
  confirmed on real conflicts with expert tags once the pilot gold exists.
- **Files:** `DATA/fivetag/adjudicator_eval/` (`result.json`, raw answers).

## EXP-021 — Dictionary version 3: the ICT branch of ESCO

- **Date / status:** 2026-11-19 · DEVELOPMENT
- **Source:** ESCO (European Commission), read through its public API
  (`ec.europa.eu/esco/api`), no account. Branch ISCED-F 061, "information and
  communication technologies": 295 concepts (148 under software and applications
  development and analysis, 104 under database and network design and
  administration, 41 under computer use, 2 in a residual group). About 300
  requests with a pause between them; cached locally.
- **Method:** `fivetag/build_lexicon_v3.py` sorts each concept by a mechanical
  rule (does the name contain a proper-name token?) into technology names or
  knowledge terms. E0 was then run with v2 and with v3.
- **Result:** 232 technology entries and 150 knowledge terms added. On the three
  development sets nothing changes (same spans found, same items passed). On
  20,000 pool sentences (seed 7), technology plus knowledge spans go from 6,536
  to 6,703 (+167, 2.6%).
- **A first build was wrong and was corrected the same day:** it added 295
  technology entries and raised the pool count to 7,110, mostly through two bad
  matches: "Computer Science" (205 hits; in postings it is almost always a
  degree field) and "ML" (113 hits; in ESCO it is a programming language, in
  postings it means machine learning). On the development sets that build made
  three more items fail. Those names and a few like them were excluded, and
  phrases ending in a generic head word ("... software", "... systems") were
  moved from technology to knowledge.
- **Limits:** the sorting rule and the exclusions were chosen by the developer
  after reading the hits; no expert has reviewed `ESCO_DERIVATION.tsv`. No
  precision estimate. v3 is not the default anywhere.

## EXP-022 — Why the verifier removed two correct spans (follow-up to EXP-012)

- **Date / status:** 2026-11-19 · DEVELOPMENT. Free: re-reads stored answers.
- **Finding:** both lost spans, for all three models, were technology names that
  occur more than once in their sentence. The model returned the text without
  saying which occurrence, and the verifier rejects that as
  `UNSUPPORTED_AMBIGUOUS_TEXT` instead of guessing.
- **How often:** on the 100 pilot sentences (EXP-007): GPT 3 rejected against
  298 kept, Haiku 2 against 221, Qwen 4 against 240 — about 1% of spans.
- **Decision:** none; the verifier is unchanged. It is working as designed, at a
  known cost of about 1% of spans. Whether a repeated phrase should instead be
  accepted at every occurrence is a choice about the silver labels and is the
  researcher's.

## EXP-023 — Time analysis over the whole sentence pool

- **Date / status:** 2026-11-19 · ENGINEERING
- **Data:** every posting of the pool with a readable date: 18,667 postings.
- **Method:** `competency.temporal` (`temporal-v1`): E0 with lexicon v2 over all
  sentences; half-year calendar windows; a window needs at least 200 postings;
  an item needs at least 20 postings. Windows kept: 2025-01..06 (354 postings),
  2025-07..12 (4,336), 2026-01..06 (6,911), 2026-07..12 (6,919). Five earlier
  windows (1 to 139 postings) were left out.

| Tag | Items analysed | Volatility low / medium / high | Top-50 turnover between windows | Trend detected |
|---|---|---|---|---|
| Technology | 281 | 119 / 98 / 64 | 0.21, 0.18, 0.31 | none |
| Knowledge | 141 | 21 / 34 / 86 | 0.39, 0.21, 0.31 | none |
| Soft skill | 90 | 12 / 19 / 59 | 0.31, 0.31, 0.46 | none |
| Work behaviour | 39 | 12 / 13 / 14 | 0, 0, 0 (fewer than 50 items) | none |

- **Reading:** the machinery works on real dates. No trend passes the test, and
  none could: with four windows a slope has two degrees of freedom. "Not
  detected" here means "cannot tell".
- **Limits, which are serious:** the first window has 354 postings against
  about 7,000 in the last two, so its shares are noisy; the mix of platforms and
  countries differs between windows, so a change in share may be a change in
  what was collected; the counts come from the rule extractor. The persistence
  weight uses an unvalidated formula.
- **Abstraction laddering** found nothing to move in the first run, because the
  ESCO parent table (EXP-021) was built after it started. The run was repeated
  the same evening as `pool_all_20261119b`, with identical counts in the table
  above. Of the 281 technologies only 36 have a node in the ESCO ICT branch; 4
  of those are highly volatile and were moved up to their parent group (COBOL,
  Drupal, Groovy, MATLAB), 32 stay specific. The other 245 have no ESCO anchor,
  so the laddering cannot act on them: the ESCO ICT branch covers too little of
  what postings name.
- **Files:** `DATA/competency/temporal/pool_all_20261119/` and
  `pool_all_20261119b/` (JSON, CSV, figures).

## EXP-024 — Competency chain with time analysis, coverage and the repaired KKNI step

- **Date / status:** 2026-11-19 · ENGINEERING. Supersedes the output of EXP-017
  (same sample, same evidence; the statements and statuses are unchanged).
- **Data and method:** as EXP-017, plus: `kkni-nearest-descriptor-v1`,
  `coverage-v1` against the 156 SKKNI unit titles, `temporal-v1` inside the
  run's own 581 postings. 158 seconds.
- **KKNI:** every one of the 445 statements now has a proposed level (level 3:
  121, level 4: 186, level 5: 104, level 6: 11, level 7: 23). **Only 1 is
  marked confident** (similarity at or above 0.45). The step no longer stays
  silent, but its proposals are weak: short statements are not close to any
  level descriptor. This needs another method or expert placement.
- **Coverage by SKKNI (proposed, not validated):** tercakup jelas 27, tercakup
  sebagian 259, belum tercakup 86, masukan pembaruan 73.
- **Time analysis inside the run:** exploratory, as it should be: only two
  windows reach 200 postings (220 and 206).
- **D1/D2 conflicts:** unchanged from EXP-018 (130; 26 high risk).
- **A draft profile** was published from this run:
  `DATA/competency/profiles/drafts/2026-0-draf1/`. It is marked DRAFT.
- **Limits:** as EXP-017.

## EXP-025 — Rehearsal of the end-to-end comparison P1, P2, P3 and two ablations

- **Date / status:** 2026-11-19 · REHEARSAL
- **Question:** does the comparison harness run all five pipelines on the same
  postings and produce the table the proposal asks for?
- **Data:** 12 postings drawn from the pool with seed 11:
  418 sentences. The same sentences for every pipeline.
- **Method:** `competency.endtoend`. P1: E0 (lexicon v2), D1/D2 conflicts left
  out, rule statements. P2: `qwen/qwen3.8-flash` on
  every sentence with three retrieved examples and the verifier, statements by
  `openai/gpt-5.4-mini`, checked. P3: E0 first, the same LLM only for
  sentences where a router trigger fires (`router-v1`, untuned), conflicts
  adjudicated by the LLM, statements as P2. Two ablations of P3: no
  deterministic verifier (LLM spans as returned, statements unchecked), and no
  persistence module (order by postings). Contract with at least 3 postings.
  **The retrieval pool is a stand-in:** the reviewed development sentences, not
  a gold training split.

| Pipeline | Evidence | Groups | Statements | Kandidat tinjauan | Sentences sent to the LLM | LLM statements rejected by the checks | Would have been rejected (kept) | Below the grounding threshold | D1/D2 conflicts |
|---|---|---|---|---|---|---|---|---|---|
| P1 | 343 | 10 | 6 | 6 | 0 | 0 | 0 | 0 | 4 |
| P2 | 656 | 32 | 21 | 19 | 418 | 10 | 0 | 0 | – |
| P3 | 566 | 25 | 15 | 13 | 305 | 9 | 0 | 0 | 35 |
| P3-tanpa-verifier | 569 | 24 | 15 | 11 | 305 | 0 | 6 | 2 | 36 |
| P3-tanpa-persistensi | 566 | 24 | 14 | 11 | 305 | 7 | 0 | 0 | 35 |

- **Overlap of the top-20 lists** (competencies matched by the phrases they are
  made of): P1 and P2 0.04; P1 and P3 0.00;
  P2 and P3 0.35.
- **Cost:** the first run charged $0.1542, almost all
  of it P2's extraction of every sentence; the P3 variants reused those answers.
  A second run, made the same evening after a counting slip in the report was
  fixed, reused the cached extraction and charged $0.0303.
  The table above is from the second run.
- **What the rehearsal shows about the harness:** it runs; the ablation without
  the verifier lets through statements the checks stop, which is what that
  ablation is for; the router sent 73% of the sentences
  to the LLM, so with untuned thresholds P3 is not yet cheaper than P2 by much.
- **Repeated calls are not identical.** The two runs asked the adjudicator and
  the statement writer again (those answers are not cached between runs) at
  temperature 0. Differences between the first and the second run:
  P3-tanpa-verifier: evidence 570 then 569, statements 15 then 15; P3-tanpa-persistensi: evidence 565 then 566, statements 15 then 14.
  Within the first run, the three P3 variants, which ask the adjudicator the
  same questions, also differed by one adjudicated span. So a real measurement
  of stability between runs is needed, as the proposal requires.
- **Limits:** 12 postings; groups are tiny and most are one posting's wording;
  extractors unmeasured; retrieval pool a stand-in; no expert judgement; no
  temporal stability. The numbers compare nothing yet.
- **Files:** `DATA/competency/runs/endtoend_rehearsal_20261119/`
  (`comparison.json`, `comparison_first_run.json`, one folder per pipeline,
  figures).

## EXP-026 — The earlier LLM competency generator on five-tag groups (first trial)

- **Date / status:** 2026-11-19 · ENGINEERING
- **Why:** the researcher pointed out that the recommendation system built
  before approval (`competency_generator_v2`, shown in `dashboard_v2`) wrote
  much better competencies than the short rule statements of EXP-017 and
  EXP-024. The plan says to reuse it. A bridge was written
  (`competency/v2_generator.py`, `v2-generator-bridge-v1`) and tried once.
- **Question:** does the earlier generator work when it is fed groups built from
  five-tag evidence, and how do its statements compare with the rule statements
  for the same groups?
- **Data:** the 581-posting sample of EXP-017/EXP-024 (20,000 sentences, seed 7).
  Evidence from E0, lexicon v2. **Grouping was still the earlier design: groups
  built from hard-skill AND knowledge phrases.** The decision that competencies
  come from hard skills only was taken later the same day, so this trial does
  not use it.
- **Settings:** the 15 valid technical groups with the most postings; model
  `openrouter:gpt-5.4-mini`, temperature 0.3, at most 2,000 output tokens, at
  most 2 retries, one competency per group (`max_competencies_per_cluster = 1`),
  generator version `v2.0.0`, its evidence filter switched on. Soft-skill
  reference list: the most frequent soft-skill phrases of the run.
- **Numbers** (from `report.json` and `competencies.json`):

| | |
|---|---|
| Groups sent | 15 |
| Statements returned | 13 |
| Dropped by the generator's own evidence filter | 2 |
| Length of a generator statement | 51 words on average (40 to 59) |
| Length of the rule statement for the same 13 groups | 4.3 words on average (2 to 8) |
| Lexical grounding (share of statement words found in the group's phrases) | 0.03 to 0.61; none reaches the 0.80 of the status contract |
| The generator's own grounding badge | 0.00 to 0.94; "passed" for 8, "not passed" for 5 |
| KKNI level proposed by the model | level 5 for all 13 |
| KKNI level of the nearest descriptor for the same 13 | level 4 for 6, level 3 for 3, level 5 for 3, level 7 for 1; none confident |
| Soft skills chosen by the model per statement | 4 to 6 |
| Status given by the contract as it stood | eksploratif for all 13 (blocked by "grounding<0.80"; one also by "okupasi_unmapped") |
| Run time | 263 seconds for the whole chain |

- **Two examples, same group:**
  rule: "Perform debugging using Git, JavaScript and AWS";
  generator: "Troubleshoot and resolve software defects by systematically
  debugging application code and runtime behavior. Apply testing, code review,
  and root cause analysis to isolate failures, verify fixes, and prevent
  recurrence. Work from observed symptoms to implement effective solutions for
  routine issues and complex system problems."
  rule: "Explain relational databases"; generator: "Design and optimize
  application databases and database schemas for reliable storage, retrieval,
  and data integrity. Apply relational database concepts, SQL, and standard data
  modeling methods ..."
- **The two it dropped:** a cloud-infrastructure statement, because it named
  "IAM" and no source sentence does; a software-practice statement, because it
  named "SDLC" and the abbreviation is not in that group's 34 sentences (the
  full phrase is). The filter is literal by design.
- **What was learned:**
  1. The earlier generator works on five-tag groups without change, and its
     statements read as competencies; the rule statements do not.
  2. The status contract's grounding check (share of statement words found in
     the evidence, at least 0.80) was made for short rule statements. It rejects
     every generator statement, because the generator is asked to abstract. The
     earlier system had already met this: its own code says the grounding score
     is "label-only, not a gate" and that the gate is the evidence filter. The
     contract was changed after this run to do the same for generator statements
     (`grounding_check = generator_evidence_filter`; a starting value, see the
     configuration registry). This run's files still show the old outcome.
  3. The model's KKNI proposal does not discriminate: level 5 thirteen times.
     It cannot replace the KKNI step. KKNI is in any case information for the
     reader only (the researcher, 2026-11-19) and never enters the status.
  4. Six of the 13 groups were made mostly of KNOWLEDGE phrases ("algorithms",
     "data structures", "relational databases"). For them the generator had to
     invent the action, and for "algorithms" it took it from very few postings
     ("object detection, classification, sensing, perception, and robotic
     processing"). This is an argument for the decision taken afterwards:
     build competencies from hard skills, attach knowledge to them.
  5. One group ("professional development", 40 postings) is employer-benefit
     text that E0 tagged as a skill; the generator duly wrote a competency about
     designing career pathways for employees. A statement is only as good as
     the group behind it.
- **Cost:** not metered. The earlier generator does not report token use, so
  the exact charge is unknown. The cap check before the run priced the worst
  case (every retry, full output allowance) at **$0.5783**; that is the upper
  bound. A plain estimate from the prompt sizes with no retries is about $0.07.
- **Limits:** 15 groups, one model, one run, no repeat (the generator is not
  deterministic at temperature 0.3); nobody but the developer has read the
  statements; the groups are of the earlier design; evidence is from the rule
  extractor whose accuracy is unmeasured.
- **Files:** `DATA/competency/runs/e0_pool20k_v2gen/` (generator statements) and
  `DATA/competency/runs/e0_pool20k/` (rule statements for the same groups).

## EXP-027 — The competency chain on the whole pool: hard-skill competencies with their tools, knowledge, soft skills and work behaviours

- **Date / status:** 2026-11-19 to 2026-11-20 (two runs, the second finished after midnight) · ENGINEERING
- **Question:** does the chain run on every posting; what does a competency
  look like when it is built from hard skills only and described by the other
  four tags; and how stable is the ranking of competencies from one time window
  to the next?
- **Data:** the whole sentence pool: 662,248 sentences from
  18,667 postings (8,186 employer/template
  groups). Evidence from E0, lexicon v2: 470,280 units
  (SH 166,182, SS 61,881, KN 31,565, TT 179,969, WB 30,683).
- **Command:** `python -m competency.pipeline e0 --all --tag e0_pool_all --min-postings 5`.
  No paid call. 13 minutes on the laptop's CPU.

### Two runs, and why the first is not the result

**First attempt (earlier design, flawed grouping).** Groups were built from
hard-skill, knowledge and soft-skill phrases (259,628 units). To
make the whole pool fit, grouping was split in two stages: phrases seen in at
least 3 postings were clustered (139,715 units), and
each rarer phrase was attached to the nearest cluster if its cosine to the
cluster's centre reached a flat 0.75. Result: 912 groups, of which
**697 "perlu dipecah"** and only 99 valid;
99 statements. Two faults produced this, both mine:
  1. the flat threshold attached 103,005 of 109,425 rare phrases
     (94%): a loose cluster accepted almost anything;
  2. the validity audit counted the attached phrases as members, so every large
     group looked too big and too loose and was marked "to be split".
  Its report is kept as `report_first_attempt.json`. Its rank stability (93
  items; rank correlation 0.34, 0.69, 0.26,
  mean 0.43) is the stability of a bad list and
  should not be quoted.

**Second run (the result of this entry).** Three changes:
  1. the researcher's decision of 2026-11-19: competencies are built from
     SKILL_HARD only (`competency_tags = ("SH",)`);
  2. a rare phrase is attached only when its cosine reaches
     max(0.75, √cohesion of that cluster), so a tight cluster asks for more;
  3. the audit judges a group by its stage-1 members only.

### Numbers of the second run (from `report.json`, `competencies.json`)

| | |
|---|---|
| Hard-skill evidence units | 166,182 |
| in stage 1 (phrase seen in at least 3 postings) | 67,811 |
| rarer phrases offered to stage 2 | 89,866 phrases, 98,371 units |
| of these attached | 33,244 phrases (37%), 37,722 units |
| Units in a group / not in any group | 105,428 / 60,754 (37% left out) |
| Groups | 684: sahih 667, penampung_proses_penunjang 9, perlu_dipecah 6, bercampur_domain 1, di_luar_bidang 1 |
| Statements (rule statements) | 667: kandidat_tinjauan 477, eksploratif 190 |
| Held back by | okupasi_unmapped 189, postings<5 3 |
| Statements that use a default verb | 88 |
| KKNI proposals marked confident | 2 of 667 (information only; never part of the status) |
| Coverage by SKKNI unit titles (proposed, not validated) | tercakup_jelas 36, tercakup_sebagian 320, belum_tercakup 136, masukan_pembaruan 175 |

**What is attached to a competency** (at most 8 lines are listed per kind; an
item must occur in at least 2 of the competency's postings):

| Attached | How | Competencies with at least one | Median lift of the listed lines | Lines with lift above 1.5 |
|---|---|---|---|---|
| Tools and technology | same sentence | 431 of 667 | — | — |
| Knowledge | same postings | 660 of 667 | 2.24 | 72% of 4,799 |
| Knowledge | same sentence | 215 of 667 | — | — |
| Soft skills | same postings | 667 of 667 | 1.48 | 49% of 5,139 |
| Work behaviours | same postings | 666 of 667 | 1.41 | 45% of 5,040 |

Share = part of the competency's postings that also ask for the item. Lift =
that share divided by the item's share among all 18,667 postings.

- **Three competencies as the run wrote them** (percent = share, × = lift):
  - **Perform backend development using Node.js, Java and Python** — 579 postings, 437 employers/templates
    - tools named with it: Node.js (96), Java (91), Python (86), Go (35), .NET (34), Spring Boot (32)
    - knowledge: cloud platforms 18% (×2.35), relational databases 14% (×3.4), microservices architecture 11% (×4.85), system design 10% (×2.25)
    - soft skills: communication 50% (×1.08), problem-solving 39% (×1.58), collaboration 27% (×1.06), leadership 15% (×0.76)
    - work behaviours: responsible 21% (×0.89), passionate 19% (×1.18), reliable 17% (×1.57), flexible 12% (×0.66)
  - **Perform automated testing using Git, Playwright and Selenium** — 1,126 postings, 791 employers/templates
    - tools named with it: Git (87), Playwright (57), Selenium (51), Python (47), Cypress (45), Docker (36)
    - knowledge: cloud platforms 12% (×1.52), design patterns 8% (×1.75), software development lifecycle 8% (×2.4), programming languages 7% (×1.05)
    - soft skills: communication 53% (×1.15), collaboration 36% (×1.41), problem-solving 32% (×1.28), leadership 23% (×1.21)
    - work behaviours: responsible 26% (×1.13), flexible 20% (×1.12), reliable 19% (×1.74), passionate 17% (×1.06)
  - **Design relational databases using SQL, PostgreSQL and SQL Server** — 578 postings, 442 employers/templates
    - tools named with it: SQL (130), PostgreSQL (49), SQL Server (43), MySQL (40), Oracle (31), MongoDB (21)
    - knowledge: cloud platforms 13% (×1.67), relational databases 9% (×2.23), cloud infrastructure 7% (×1.52), design patterns 6% (×1.31)
    - soft skills: communication 46% (×0.99), problem-solving 31% (×1.26), collaboration 23% (×0.89), leadership 11% (×0.59)
    - work behaviours: responsible 28% (×1.21), flexible 16% (×0.86), reliable 14% (×1.3), passionate 13% (×0.77)

### Rank stability across time windows

Proposal: "kesamaan peringkat dan keanggotaan daftar kompetensi teratas ...
pada jendela waktu yang berbeda". For the 574 competencies with
enough postings, the share of postings in each half-year window was ranked and
consecutive windows compared. Windows used (postings):
2025-01..2025-06 (354), 2025-07..2025-12 (4,336), 2026-01..2026-06 (6,911), 2026-07..2026-12 (6,919).

| Windows compared | Spearman rank correlation | Top-20 overlap | Top-50 overlap | Top-100 overlap |
|---|---|---|---|---|
| 2025-01..2025-06 to 2025-07..2025-12 | 0.75 | 0.60 | 0.67 | 0.61 |
| 2025-07..2025-12 to 2026-01..2026-06 | 0.91 | 0.54 | 0.69 | 0.79 |
| 2026-01..2026-06 to 2026-07..2026-12 | 0.76 | 0.43 | 0.49 | 0.55 |

Mean rank correlation 0.81. Trend test: {'tidak_terdeteksi': 574}
(no trend detected for any competency over four windows). Volatility: {'tinggi': 256, 'sedang': 176, 'rendah': 142}.

- **What the stability figure measures and what it does not:** the competencies
  are formed once, from all postings, and then ranked inside each window. It is
  the stability of the RANKING of a fixed list. It does not show whether the
  same competencies would be formed from one window's postings alone; that
  needs the chain run separately per window and has not been done.

### What is visibly weak

1. **Soft skills and work behaviours barely differ between competencies.**
   "communication" is asked for in about half of the postings of nearly every
   competency, with a lift close to 1: it is asked for everywhere. The lift is
   what shows the specific ones (for example "attention to detail" and
   "reliable" with testing). Read the lift, not the share.
2. **Lift is distorted by language.** About 4% of the pool is Indonesian. An
   Indonesian soft skill ("komunikasi") is rare among all postings and common
   among the postings of an Indonesian-language competency, so its lift is 10
   to 20. That is the language, not the competency. The lift must be computed
   inside one language, or Indonesian and English forms of one skill merged,
   before these numbers are compared across languages. Not done.
3. **Spelling variants are separate lines** ("problem-solving" and "problem
   solving"), which splits their share.
4. **Some rule statements are a peculiar phrase from a few postings**: several
   of the 25 most frequent statements carry an employer's name or a fragment of
   a sentence. The rule statement takes the group's most frequent verb and the
   most frequent object of that verb; in a broad group where nearly every
   object occurs once or twice, the winner is an accident.
   The earlier LLM generator (EXP-026) writes from the whole group and would not
   have this fault; it has not been run on these groups.
5. **Same-posting attachment is weak evidence.** It shows that employers who
   want the competency also want the soft skill, not that the skill is used in
   that work.
6. 37% of hard-skill units are in no group (phrases too rare and too
   far from every cluster). Whether they are noise or lost competencies has not
   been examined.
7. An English default verb is put in front of an Indonesian label for some
   groups ("Perform pemrograman ...").

- **Limits:** evidence from the rule extractor, accuracy unmeasured; the first
  window has far fewer postings than the others and the mix of platforms and
  countries differs between windows, so part of any instability is collection,
  not demand; `core_min_postings`, `attach_threshold`, the 8-line limit and the
  2-posting minimum are starting values chosen without tuning; cluster centres
  and cohesion are those of stage 1 and are not recomputed after attaching;
  attaching KNOWLEDGE is the developer's reading and awaits the researcher.
- **Files:** `DATA/competency/runs/e0_pool_all/` (evidence table, groups,
  competencies, `report.json`, `report_first_attempt.json`, figures).

## EXP-028 — A competency as verb + knowledge, the cognitive level read from the postings, the cluster map, and the earlier generator on the 30 most frequent competencies

- **Date / status:** 2026-11-20 · ENGINEERING
- **Why:** three instructions of the researcher on this day. (1) "A competency
  is actually verb + knowledge ... 'develop machine learning algorithm':
  'develop' is the verb (according to taxonomy) and 'machine learning
  algorithm' is a knowledge"; tools and technology support it. (2) The
  statement must not name tools: "Perform programming", not "Perform
  pemrograman using Python, PHP and JavaScript"; tools, work behaviours and
  soft skills are listed for the competency. (3) After seeing the dashboard:
  the cognitive level shown did not match what the postings ask; a group whose
  postings say "membangun"/"develop" (C6) was shown as "apply" (C3). She also
  asked for a view of the clusters of knowledge and of tools and technology,
  and approved running the earlier LLM generator.
- **Data:** the whole sentence pool, 662,248 sentences from
  18,667 postings; evidence from E0, lexicon v2
  (470,280 units). Same grouping as EXP-027 (684 groups, sahih 667, penampung_proses_penunjang 9, perlu_dipecah 6, bercampur_domain 1, di_luar_bidang 1).
- **Runs:** `e0_pool_all_v2gen` (the chain with the generator on the 30 most
  frequent valid groups; paid) and `e0_pool_all_v3` (the same chain after the
  cognitive-level correction; the generator statements of the first run were
  taken over with `--reuse-v2-from`, 29 reused, 0 not, no new call;
  11 minutes). The numbers below are from `e0_pool_all_v3` unless said otherwise.

### 1. Verb and knowledge read from the phrases (`verb-knowledge-v1`)

| | |
|---|---|
| Hard-skill units in groups | 105,428 |
| of these opening with a verb of the table | 72,640 (69%) |
| of these containing a term that is tagged KNOWLEDGE anywhere in the corpus (vocabulary of 934 terms seen in at least 3 postings) | **662 (0.6%)** |
| Groups with at least one shared knowledge object | 651 of 684 |
| Groups whose first knowledge object is also a KNOWLEDGE-tagged term | 3 |

The third row is the reason the knowledge of a competency is read from the
object of the verb and not from the KNOWLEDGE tag: the extractor tags a term
KNOWLEDGE when it stands alone, and when a verb governs it the whole phrase is
a hard skill. Requiring the tag would have left almost every competency without
a knowledge. Method (no part-of-speech tagger): the object is cut at the first
preposition; its head word is taken (last word in English, first in
Indonesian); the most shared head is lengthened a word at a time while more
than half of its postings keep the longer form.

Rule statements (638; median length 2 words; none names a tool):
562 are a verb of the postings plus the shared knowledge object,
75 are a default verb plus the group's most frequent noun phrase,
1 use the literal object. For the first kind, the share of the
group's postings behind the object: median 0.60; at least 0.20 for
482; below 0.05 for 3.

Status: kandidat_tinjauan 456, eksploratif 211; held back by okupasi_unmapped 210, postings<5 3.
(EXP-027, with tool names in the statements: 477 and 190. The okupasi step
compares the statement text, and a two-word statement matches less often.)

### 2. Cognitive level: before and after the correction

Before: the level shown was the level of the statement's verb, including a
DEFAULT verb the method had supplied. After: the level is read from the verbs
the postings use; a posting counts once for each level; the level is the one
most postings ask for (C1 remember … C6 create); a default verb sets nothing.

| | |
|---|---|
| Competencies with a level read from the postings | 635 of 667 |
| without (their postings use almost no verb) | 32 |
| by level | C6 276, C3 262, C5 63, C4 34, none 32 |
| Competencies present in both runs | 667 |
| whose shown level CHANGED | **84** (13% of those with a readable level) |
| of these, shown before with the level of a default verb | 40 |
| most frequent changes (before → after) | apply → create: 43; create → apply: 23; evaluate → apply: 7; apply → evaluate: 5; apply → analyze: 3 |
| Rule statements whose verb is still on another level than the postings ask | 46 (a default verb, or no verb of that level is used with the object) |

| Statement as shown before (EXP-027) | Level shown before | Rule statement now | Level read from the postings | Verbs of the postings |
|---|---|---|---|---|
| Perform pemrograman using Python, PHP and JavaScript | apply | Melakukan pemrograman | C6 create (65% of 266) | menulis, membangun, melakukan, membuat |
| Design relational databases using SQL, PostgreSQL and SQL Server | create | (generator statement) | C3 apply (56% of 512) | design, optimize, maintain, develop |
| Troubleshoot issues using SQL, Azure and Datadog | analyze | Troubleshoot issues | C4 analyze (100% of 345) | troubleshoot, melakukan |
| Perform web development using JavaScript, CSS and HTML | apply | Perform web development | none readable |  |
| Melakukan pengujian menggunakan Git, Android dan JIRA | apply | Melakukan pengujian | C5 evaluate (50% of 142) | test, melakukan, analyse, mengembangkan |
| Design REST APIs using Postman, Python and Java | create | Develop RESTful APIs | C6 create (71% of 331) | develop, integrate, build, design |

The first row is the researcher's own example.

### 3. The cluster map of knowledge and technology (`termmap-v1`)

500 terms (the 250 most frequent KNOWLEDGE terms and the 250 most frequent
technology names, each in at least 5 postings) in 20 clusters. Method:
positive pointwise mutual information of sharing a posting, truncated SVD to
32 dimensions, average-linkage clustering on cosine distance; t-SNE (seed
20261120) for the drawing only.

| Cluster | Terms | Knowledge | Technology | Named after |
|---|---|---|---|---|
| 1 | 125 | 55 | 70 | JavaScript, Git, React |
| 2 | 87 | 31 | 56 | AWS, Azure, Docker |
| 3 | 55 | 23 | 32 | Python, SQL, Power BI |
| 4 | 30 | 8 | 22 | Java, Jira, Oracle |
| 5 | 18 | 1 | 17 | C#, .NET, SQL Server |
| 6 | 19 | 16 | 3 | C++, programming languages, algorithms |
| 7 | 21 | 8 | 13 | Android, iOS, Kotlin |
| 8 | 22 | 15 | 7 | Salesforce, SAP, ServiceNow |
| 9 | 14 | 7 | 7 | Postman, Selenium, JMeter |
| 10 | 18 | 9 | 9 | Figma, WordPress, Google Analytics |

The other 10 clusters hold 91 terms, almost all rare KNOWLEDGE terms.
The clusters are of co-occurrence, not of meaning; their number
(20 = the square root of the number of terms, capped at 20) is a setting.
Shown as the tab "Peta pengetahuan & teknologi" of the public dashboard and as
`figures/term_map.png`.

### 4. The earlier generator on the 30 most frequent valid groups

Model `openrouter:gpt-5.4-mini`, generator `v2.0.0`, one competency per group, its evidence
filter on.

| | |
|---|---|
| Groups sent | 30 |
| Statements returned | 29 |
| Dropped by its evidence filter | 1 (it named "APIs" where no source sentence of the group does literally) |
| Length | 50 words on average (36 to 59) |
| Statements that name a tool of the group | 4 of 29 (reported, not removed: the old generator's prompt is not edited) |
| KKNI level proposed by the model | level 5: 28, level 6: 1 |
| Status under the contract (generator statements are judged by the generator's evidence filter) | kandidat_tinjauan 29 |
| Worst case priced before the run | $1.2504 (cap $1.80) |
| **Charged, read from the account** | **$0.1627** (usage $88.6437 → $88.8063) |

- **Reading of the 29 by the developer only:** most describe software work and
  read as competencies. 3 are clearly not software work: they are about
  employees' professional development, career development and an inclusive
  workplace. Those groups are
  employer-benefit and company-culture sentences that the rule extractor tagged
  as hard skills; the generator wrote a fluent statement for each. A fluent
  statement is no evidence that the group behind it is a competency.
- The model's KKNI proposal again does not discriminate (level 5 for 28 of 29).
- **What the cost measurement adds:** OpenRouter's `/credits` endpoint, read
  with the key the code already loads, gives usage before and after. The run
  cost 13% of its priced worst case. Credit on the account after the run:
  $95.00 bought, $88.81 used, **$6.19 left**. EXP-026 could not be
  costed this way afterwards; its true cost stays unknown (at most $0.5783).

### Limits

Evidence from the rule extractor, accuracy unmeasured, and visibly wrong for
some of the most frequent groups (benefit and culture sentences; spans cut
short, which give objects such as "clean"). The verb table that places each
verb on a level is the developer's and unreviewed (`verbs-v1-unreviewed`):
every level in section 2 follows from it. "The level most postings ask for" is
one of two possible rules; the highest level asked for by a fifth of the
verb-using postings is recorded beside it. The head-of-object rule is word
order, not grammar. One model, one run, no repeat for the generator; nobody but
the developer has read its statements.

- **Files:** `DATA/competency/runs/e0_pool_all_v3/` (final), `e0_pool_all_v2gen/`
  (the paid run), `e0_pool_all/` (EXP-027, the "before").

---

## Spending on paid model calls

| Entries | Charged |
|---|---|
| EXP-002 to EXP-007 (ensemble runs, 18–19 November) | $2.0732 |
| EXP-018 (adjudicator trial) | $0.0040 |
| EXP-019 (statement writer trial) | $0.0261 |
| EXP-020 (adjudicator comparison) | $0.0678 |
| EXP-025 (end-to-end rehearsal, two runs) | $0.1845 |
| EXP-026 (earlier generator, 15 groups) | not metered: at most $0.5783, estimated $0.07 |
| EXP-028 (earlier generator, 30 groups; read from the account) | $0.1627 |
| **Total recorded (metered runs)** | **$2.5183** |
| **Upper bound including EXP-026** | **$3.0966** |

"Charged" is an upper estimate (see the shared settings). The provider's own
balance is the authority.
