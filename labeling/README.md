# `labeling/` — multi-LLM ensemble labelling (Workstream C)

Implements proposal §3.4.3 (Gambar 3.8). At least three LLMs label the same
sentence independently; a deterministic verifier checks every output against the
source; surviving spans are aligned across models; a consensus rule decides what
is accepted automatically and what abstains.

**Status, 2026-11-18:** verifier, alignment, consensus, a guideline-only prompt
pack and the OpenRouter runner are built and smoke-tested on the 10 practice
sentences. The silver store, gold few-shot examples and the frozen
configuration are not.

| File | Does | Proposal clause |
|---|---|---|
| `schema.py` | output contract, closed label vocabulary | "format keluaran terstruktur yang sama" |
| `verify.py` | JSON, schema, span-in-source, offsets, labels; INVALID / UNSUPPORTED log | "pemeriksaan deterministik" |
| `align.py` | exact, then IoU alignment across models | "penyelarasan span dilakukan secara bertingkat" |
| `consensus.py` | six-way vote, UNANIMOUS / MAJORITY / CONFLICT, acceptance and abstention | "tiga tingkat konsensus" |
| `prompts.py` | the shared prompt pack, condensed from guideline B v3.0; version `v0-guideline-only`, no gold examples | "pedoman anotasi, definisi lima kategori ... yang sama" |
| `run.py` | calls every model through OpenRouter, keeps raw responses, enforces a spending cap, then verifies and votes | — |

```bash
.venv-ml\Scripts\python.exe tests\test_labeling.py       # 48 tests, no API calls
.venv-ml\Scripts\python.exe tests\test_labeling_run.py   # 18 tests, fake client, no API calls
.venv-ml\Scripts\python.exe tests\test_metrics.py        # 25 tests

# a real run: spends API credit, and refuses to start if the worst case exceeds the cap
.venv-ml\Scripts\python.exe -m labeling.run --split practice_10 --max-usd 0.50 --reasoning none
```

Runs are written to `DATA/labeling/runs/<run_id>/` (gitignored: they embed
corpus sentences). The runner refuses any output directory under
`DATA/gold_standard/`, so the annotators stay blind to LLM output.

## First smoke test, 2026-11-18 (10 practice sentences, 3 models)

Models, all via OpenRouter: `openai/gpt-5.4-mini`, `anthropic/claude-haiku-5.5`,
`qwen/qwen3.8-flash`. This is a working choice for the smoke test, not the
frozen ensemble.

| | provider-default reasoning | `--reasoning none` |
|---|---|---|
| responses with usable JSON | 26 / 30 | 30 / 30 |
| spans proposed, all found in the source | 81 | 105 |
| unsupported or invalid spans | 0 | 0 |
| candidates | 46 | 51 |
| unanimous (accepted) | 7 | 15 |
| majority (abstaining until validated on dev) | 10 | 11 |
| cost charged by the budget guard | $0.028 | $0.021 |

With provider-default reasoning, `qwen/qwen3.8-flash` spent its whole 1,500-token
allowance on hidden reasoning for 4 of 10 sentences and returned no answer.
Those four are counted as invalid outputs. With reasoning off all 30 responses
were valid, cheaper and faster. Whether the frozen configuration uses reasoning
is a decision for the development gold; `--reasoning none` is the working
setting.

These numbers describe 10 sentences with no gold to score against. They show
the pipeline runs end to end; they say nothing about accuracy.

## Measuring a prompt change before the gold exists

Agreement between models is not correctness: in the smoke test all three models
agreed that "wireframes" is SKILL_HARD. Two instruments give a number now. Both
are development aids, and neither is evidence about accuracy on the corpus.

| | `conformance.py` | `skillspan_probe.py` |
|---|---|---|
| Sentences | 109 synthetic, one guideline rule each | 100 real, from the SkillSpan-derived dev data |
| Reference | expected answer follows from the rule | spans marked by SkillSpan's human annotators |
| Tags | all five | two (model output is collapsed: SH/SS/WB -> SKILL, KN/TT -> KNOWLEDGE) |
| Shows | per-rule pass rate, e.g. rule 6.2 "keep the verb" | whether a model keeps the head (usually the verb) of a human-marked skill |
| Does not show | accuracy on real postings | five-tag correctness; SkillSpan follows a different guideline |
| Valid comparison | the same model under prompt v0 vs v1 | the same model under prompt v0 vs v1 |

Neither may be used to rank the models against each other, and neither lives
under `DATA/gold_standard/`. The conformance answers were reviewed and approved
as a whole by the researcher on 2026-11-18. Its `variant` items are checked by a
test never to appear in a prompt, so they measure the rule and not recall of an
example.

A third instrument, `skillspan_guideline.py`, is what the researcher asked for
when she suggested SkillSpan: REAL SkillSpan sentences whose spans and tags
follow guideline B v3.0 instead of SkillSpan's two labels. The annotations are
drafted by an LLM and reviewed by the researcher. Only spans the guideline
settles clearly are asserted. Batch 1 (42 software-related sentences, 74 spans)
and its six boundary conventions (C1-C6 in the module) were approved by her on
2026-11-18. Batch 2 (58 sentences, 120 spans) was drawn after prompt v2 was
frozen, so it is the test of v2 on unseen text; she reviewed and approved it on
2026-11-19, before seeing any model result on those sentences. The annotation file holds ids and span
texts only and stays in the gitignored `DATA/labeling/skillspan_guideline/`,
because redistributing SkillSpan-derived text is an open licence question.

### What the instruments showed, 2026-11-19

Batch 1 of the guideline-annotated SkillSpan sentences, expected spans found
out of 74 (strict: same boundaries and same tag):

| Model | v0 | v1 | v2 | boundary errors v0 -> v2 | of which too long |
|---|---|---|---|---|---|
| gpt-5.4-mini | 47 | 50 | 55 | 24 -> 17 | 19 -> 10 |
| claude-haiku-5.5 | 51 | 43 | 55 | 12 -> 5 | 8 -> 2 |
| qwen3.8-flash | 57 | 54 | 57 | 14 -> 13 | 11 -> 9 |

Tag errors were rare throughout (0-2 per model). The dominant error under v0
was a span that was too long (38 of 50 boundary differences across the three
models), which is what v2's conventions address. **This is a result on the
batch v2 was written with in view.**

Batch 2, which prompt v2 never saw, expected spans found out of 120:

| Model | v0 | v1 | v2 | v0 -> v2, same spans: gained / lost | exact McNemar p |
|---|---|---|---|---|---|
| gpt-5.4-mini | 101 | 105 | 98 | 9 / 12 | 0.66 |
| claude-haiku-5.5 | 111 | 109 | 113 | 5 / 3 | 0.73 |
| qwen3.8-flash | 106 | 107 | 112 | 6 / 0 | 0.03 |

**The batch-1 gain only partly carried over.** On unseen text v2 helped one
model clearly (qwen), one slightly (haiku) and not the third (gpt), which lost
spans by over-merging: "technologies such as Babel Jest Vue" as one span,
"Implementing and operating cloud services including implementing
instrumentation" as one span. v2's rules against splitting appear to push that
model too far the other way. Over both batches (194 spans) each model found 5 or
6 more spans under v2 than under v0, and for none of them is that
distinguishable from chance (p = 0.51, 0.36, 0.18). So v2 is not shown to be
better than v0 overall; it trades one kind of boundary error for another, and
the trade depends on the model.

Two limits on reading any of this. A set of 194 spans cannot resolve
differences of this size, so further prompt tuning against these two batches
would be fitting noise, and no unseen batch is left. And the reference follows
conventions C1-C6, which are NOT in guideline B v3.0: the three expert
annotators have not been given them, so a prompt tuned to them may drift from
what the human gold will contain. They belong on the pilot calibration agenda.

The conformance set is near its ceiling (94-99% of items under v0), so it
serves as a regression check and not as a measure of progress. The SkillSpan
probe showed that the models keep the head of a human-marked skill span about
nine times in ten under v0, so a general tendency to drop the verb was not
confirmed.

Prompt versions: `v2-boundary-conventions` adds the accepted conventions C1-C6
as rules (leave out lead-in nouns, evaluative words, a trailing generic
"skills", trailing context and a generic word after a technology name;
coordinated verbs sharing an object are one span). All three versions are
frozen by test. `v0-guideline-only` is frozen (finished runs are keyed on its
hash). `v1-sh-action-and-noun-lists` adds one section on SKILL_HARD: keep the
governing verb, activity nouns such as "automated testing" are still SH, thing
nouns are not, and — researcher ruling of 2026-11-18 — when one verb governs a
list of nouns, each noun is tagged KNOWLEDGE.

## How a sentence is decided

```python
from labeling.verify import verify_output
from labeling.consensus import ConsensusConfig, decide_sentence

cfg = ConsensusConfig(models=("model-a", "model-b", "model-c"))
outs = [verify_output(m, sid, sentence, raw_response[m], cfg.verify_policy)
        for m in cfg.models]
result = decide_sentence(sid, sentence, outs, cfg)
result.accepted            # spans that may enter the silver store
```

Each candidate span gets one status:

| Status | Rule | Disposition |
|---|---|---|
| `UNANIMOUS` | all n models, same tag, identical boundaries | accepted |
| `MAJORITY` | at least ⌈2n/3⌉ of the n models agree on tag and boundaries | **abstains** until `accept_majority` is switched on |
| `BOUNDARY_DISAGREEMENT` | ⌈2n/3⌉ agree on the tag, but no single pair of boundaries reaches ⌈2n/3⌉ | abstains |
| `NONE_MAJORITY` | ⌈2n/3⌉ of the models found no entity here (a minority proposed one) | abstains |
| `CONFLICT` | none of the above: label split, or too few valid votes | abstains |

With three models ⌈2n/3⌉ = 2.

## Decisions that are fixed by the approved proposal

- **The denominator is every configured model.** A model whose output failed
  verification does not make the remaining two "unanimous"; the best they can
  reach is `MAJORITY`.
- **`accept_majority` defaults to off.** The proposal allows majority labels to
  be accepted "hanya apabila aturan mayoritas tersebut sebelumnya menunjukkan
  tingkat ketepatan yang memadai pada development gold standard". Until that
  precision has been measured, only unanimous spans are accepted.
- **Nothing is forced.** Conflicts, boundary disagreements and single-model
  spans keep their evidence and abstain.
- **No text is created.** An accepted span is always `sentence[start:end]`.

## Decisions the proposal leaves open — made here, to be revisited

These are mine, not the proposal's. Each is recorded in the `config` block of
every result so that it can be frozen, reported and changed deliberately.

1. **IoU threshold 0.5 over characters is provisional.** It equals the rule
   behind the relaxed human-agreement figure in `gold_standard/iaa.py`, so
   "aligned" means for models what "partially agree" means for experts. The
   proposal requires the threshold to be set on the development gold, which
   does not exist yet. Token IoU is available (`iou_unit="token"`).
2. **Complete linkage.** Every pair of spans in a candidate must clear the
   threshold. Single linkage would chain A~B and B~C into "three models agree"
   even when A and C barely overlap.
3. **One span per model per candidate.** Two spans from one model are two
   claims.
4. **A model with a rejected span casts no vote where it is silent.** Its
   silence is not counted as NONE, because we discarded something it claimed.
   This can only lower agreement: a property test shows it never changes which
   spans are accepted.
5. **`consensus_ratio` and `disagreement_score` formulas.** The proposal names
   both measures without defining them. Here: the ratio is the share of all n
   models behind the most-voted decision; the score is the Shannon entropy of
   the votes over seven outcomes (five tags, NONE, no-vote), normalised to
   [0, 1]. This is not claimed to be the formula of Du et al. (2023).
6. **Boundary trimming.** Edge whitespace and punctuation are trimmed with the
   same function the annotator UI applies to a human selection, so a model is
   not charged a strict-match error for a trailing comma the UI would remove.
7. **No silent repair.** Stripping a markdown fence and re-anchoring on an
   offset mismatch both exist, both are off by default, and both are recorded
   on the output when used. A strict JSON-validity rate counts a fenced
   response as invalid.

## Researcher decisions, 2026-11-18

Three points were put to the researcher and decided. They are recorded here
because each one changes what the ensemble may accept.

**1. Overlapping spans follow the guideline.** The implementation plan
originally listed "no overlap" among the verifier's checks. The validated
guideline (B v3.0, rule 6.9) allows overlapping spans "apabila fungsinya
berbeda", with the example "Configure Kubernetes clusters" as SKILL_HARD
containing "Kubernetes" as TOOL_TECHNOLOGY. If experts may annotate that pattern
and models may not emit it, the ensemble can never match the gold on those
sentences. Decision: **overlap is allowed when the function differs.** The
verifier keeps overlapping spans with different tags and flags them
`OVERLAP_POTENTIAL`; overlapping spans with the same tag are rejected
(`overlap="allow_cross_label"`, the default). The plan's wording was amended the
same day to match.

**2. Majority acceptance stays off.** Only unanimous spans are accepted until
the precision of the two-thirds rule has been measured on the development gold.
`accept_majority` is switched on by that evidence, not before it.

**3. The provisional settings are accepted as provisional.** The IoU threshold
(0.5 over characters) and the formulas for `consensus_ratio` and
`disagreement_score` stand as working values. The threshold is to be set on the
development gold, as the proposal requires, before the configuration is frozen.

**4. One verb governing a list of nouns: the nouns are KNOWLEDGE** (ruling of
2026-11-18, on a practice sentence in which a single generic verb governs a
list of design stages and artefacts; the sentence itself is corpus text and is
not reproduced here). The validated guideline
does not settle this case: rule 6.2 says keep the verb, rule 6.5 says annotate
list items separately. The ruling is applied in prompt v1 and in three
conformance items. Two things remain open and should go to the pilot
calibration, because the annotators will meet them too: the guideline's own
example 7.13 tags "unit testing" as SKILL_HARD under a shared verb, so activity
nouns and thing nouns need telling apart; and coordinated verbs sharing one
object ("Creating, compiling, and debugging programs") are not covered. This
ruling is not in guideline B v3.0, so the three annotators do not know it yet.

## What still needs the human gold standard

- few-shot examples for the prompt pack (train/dev only)
- the IoU threshold (dev)
- whether `accept_majority` may be switched on (dev precision)
- the automatic-acceptance precision itself — `accepted_spans_by_sid()` feeds
  `gold_standard.metrics`, so it is scored with the same matcher as the gold

The held-out test is opened once, after this configuration is frozen.
