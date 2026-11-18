# `labeling/` — multi-LLM ensemble labelling (Workstream C)

Implements proposal §3.4.3 (Gambar 3.8). At least three LLMs label the same
sentence independently; a deterministic verifier checks every output against the
source; surviving spans are aligned across models; a consensus rule decides what
is accepted automatically and what abstains.

**Status, 2026-11-18: the deterministic half only.** No model is called by
anything in this directory yet.

| File | Does | Proposal clause |
|---|---|---|
| `schema.py` | output contract, closed label vocabulary | "format keluaran terstruktur yang sama" |
| `verify.py` | JSON, schema, span-in-source, offsets, labels; INVALID / UNSUPPORTED log | "pemeriksaan deterministik" |
| `align.py` | exact, then IoU alignment across models | "penyelarasan span dilakukan secara bertingkat" |
| `consensus.py` | six-way vote, UNANIMOUS / MAJORITY / CONFLICT, acceptance and abstention | "tiga tingkat konsensus" |

Not built yet: the prompt pack, the model runner (OpenRouter, via
`llm_client_router.py`), and the silver store. They depend on this half.

```bash
.venv-ml\Scripts\python.exe tests\test_labeling.py     # 48 tests, no API calls
.venv-ml\Scripts\python.exe tests\test_metrics.py      # 25 tests
```

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

## What still needs the human gold standard

- few-shot examples for the prompt pack (train/dev only)
- the IoU threshold (dev)
- whether `accept_majority` may be switched on (dev precision)
- the automatic-acceptance precision itself — `accepted_spans_by_sid()` feeds
  `gold_standard.metrics`, so it is scored with the same matcher as the gold

The held-out test is opened once, after this configuration is frozen.
