# Pre-registration: gold-standard size and the conditions for enlarging it

**Registered:** 2026-11-16, before any pilot annotation results exist.
**Amended:** 2026-11-16 (same day, still before any annotation result), to
record the completed learning curve: a third seed at 1,440 / 2,880 and the
4,800-sentence point, which the first run had omitted. The amendment changes
only Section 2 (evidence). **The registered design in Section 1 and the
enlargement trigger in Section 4 are unchanged**, and the completed curve
supports the 1,440 figure more strongly than the partial one did.
**Status at registration:** pilot_100 and practice_10 drawn; no annotator has
submitted work; main_1200 NOT yet drawn; instrument validation in progress
(2 of 3 validators returned).

This document exists so that the gold-standard sample size is fixed in advance
of seeing any annotation result. Everything below is committed now; nothing here
may be revised after pilot outcomes are known.

---

## 1. Registered primary design (unchanged)

As stated in the proposal (Subbab 3.4.2, verbatim: *"Ketiga ahli menganotasi
seluruh 1.200 kalimat yang sama secara independen sehingga tersedia 3.600 set
keputusan anotasi sebelum adjudikasi"*):

| Quantity | Value |
|---|---|
| Pilot sentences | 100 |
| Main gold-standard sentences | 1,200 |
| Annotators per sentence | 3 (all three annotate every sentence) |
| Total human annotation passes | 3,900 |
| Split | 720 train / 240 dev / 240 test |

The split is produced by the group-aware splitter under
`docs/experiments/split_spec_v1.md`: no employer and no posting template may
straddle any train/test boundary, with a temporal hold-out and a cross-country
external test as separate registered regimes. The test set is sealed; model and
configuration selection happens on dev only.

## 2. Evidence considered at registration

A learning curve was measured in-house (2026-11-16) because no paper in the
project's literature corpus reports one for skill-span extraction. SkillSpan
(Zhang et al., NAACL 2022) as the closest public analogue; BIO over 2 types;
distilbert-base-uncased; 8 epochs at every size; exact span match scored with
`_prf` from `gold_standard.iaa`; evaluated on a seeded 1,500-sentence DEV
subsample; SkillSpan's test split left sealed. Raw per-seed results:
`docs/experiments/learning_curve/`.

| train sentences | seeds | F1 (mean) | sd | seed spread | gain | above noise? |
|---|---|---|---|---|---|---|
| 180 | 2 | 0.275 | 0.040 | 0.057 | — | — |
| 360 | 2 | 0.354 | 0.014 | 0.020 | +0.079 | yes |
| 720 | 2 | 0.424 | 0.008 | 0.012 | +0.070 | yes |
| 1,440 | 3 | 0.468 | 0.004 | 0.008 | +0.044 | yes |
| 2,880 | 3 | 0.478 | 0.019 | 0.034 | +0.011 | NO |
| 4,800 | 3 | 0.536 | 0.002 | 0.003 | +0.058 | yes |

Marginal value of annotation, which is the quantity that matters when the cost
is linear in sentences:

| step | F1 gained | sentences added | F1 per 100 sentences |
|---|---|---|---|
| 180 → 360 | +0.079 | 180 | **+0.044** |
| 360 → 720 | +0.070 | 360 | **+0.019** |
| 720 → 1,440 | +0.044 | 720 | **+0.0061** |
| 1,440 → 4,800 | +0.069 | 3,360 | **+0.0020** |

Reading recorded at registration: accuracy continues to improve with training
data through 4,800 sentences (the full SkillSpan train split) with no observed
saturation, so there is NO plateau to point at. The flat step at 2,880 is
attributed to seed noise, not saturation — its seed spread (0.034) exceeds its
gain (0.011), and the 4,800 point above it is both higher and far tighter
(spread 0.003 across three seeds).

The decision-relevant quantity is therefore not where the curve flattens — it
does not — but where the MARGINAL value of a human-annotated sentence falls.
That knee is at 1,440: each additional 100 sentences buys +0.0061 F1 up to
1,440 and +0.0020 F1 beyond it, a threefold drop. Since annotator cost is
linear in sentences while the return past 1,440 is not, 1,440 is the
economically defensible ceiling for HUMAN annotation, and the training scale the
curve still rewards is supplied by the proposal's multi-LLM ensemble
(Workstream C) at near-zero marginal cost. The human gold standard's registered
function remains reliability evidence, sealed evaluation, and validation of the
LLM labels.

A secondary finding, recorded because it transfers to the five-tag scheme: the
longer and more abstract SKILL type trails the concrete KNOWLEDGE type at every
budget, and the gap closes monotonically with data (0.181 at 180 sentences →
0.087 at 4,800). Tags that behave like KNOWLEDGE (TT: concrete, closed
vocabulary) should be learnable from less data than tags that behave like SKILL
(SS, WB: phrasal, contextual).

Two derived precision calculations, also fixed at registration:

*Inter-annotator reliability.* Scaling 1/sqrt(n) from a measured bootstrap CI
width of 0.30 at n=100, the project's own claimability threshold
(`gold_standard/iaa.py: CI_WIDTH_CLAIMABLE = 0.20`) is met at n≈225. At
n=1,200 the expected CI width is ~0.087. Reliability is over-served by the
registered design and is NOT a reason to enlarge.

*Evaluation precision.* CI width on micro-F1 at F1≈0.50, as a function of test
sentences and span density:

| test sentences | @1.1 spans/sent | @2.0 | @3.0 |
|---|---|---|---|
| 240 | 0.121 | 0.089 | 0.073 |
| 480 | 0.085 | 0.063 | 0.052 |

Span density in this project's corpus is UNKNOWN at registration. SkillSpan's
is 1.08 spans/sentence. The registered 240-sentence test set is therefore the
design's weakest point, and its adequacy depends on a quantity the pilot will
measure.

## 3. Registered analysis decision (not conditional)

Comparisons among extraction conditions E0–E6, classifiers D1–D3 and pipelines
P1–P3 are reported as **paired differences on the identical sealed test items,
with a paired bootstrap confidence interval on the difference**. Overlap of two
marginal confidence intervals is NOT used as a test of difference. This is
registered now because it is the correct procedure irrespective of sample size,
and because it is what makes a 240-sentence test set usable.

## 4. Registered enlargement rule (conditional, single trigger)

The pilot's first computed output, before any agreement coefficient, is the
**measured span density** of the adjudicated pilot: total five-tag spans divided
by 100 sentences.

**IF** measured pilot span density < 1.5 spans/sentence,
**THEN** the gold standard is enlarged exactly once, to:

| Quantity | Value |
|---|---|
| Main gold-standard sentences | 2,400 |
| Split | 1,440 train / 480 dev / 480 test |
| Annotators per sentence | 3 (unchanged) |
| Total human annotation passes | 7,500 |

**ELSE** the registered 1,200-sentence design stands unchanged.

Justification fixed in advance: 1,440 is the largest train size whose gain over
720 (+0.044) exceeds the observed seed spread at every intermediate point, and
480 test sentences brings the evaluation CI width to ~0.085 even at the low
density that would have triggered the rule.

**No second enlargement is registered.** A further doubling to 2,880 showed a
gain inside seed noise, so it is not pre-authorised and must not be adopted on
the basis of the results of this study.

The enlarged sample, if triggered, is drawn with the same seeded stratified
procedure (`gold_standard/sample.py`), disjoint from pilot and practice
postings, and re-split by the same group-aware splitter. The already-annotated
1,200 are retained; the additional 1,200 are added, not redrawn.

## 5. What would invalidate this registration

- Any change to these numbers made after pilot agreement results are known.
- Drawing main_1200 (or main_2400) from a sentence pool whose
  `pool_sha1_16` differs from the one recorded in the sample manifest, without
  recording a new gold-standard version.
- Opening the sealed test split for anything other than final evaluation of a
  configuration already fixed on dev.
- Reporting a configuration comparison as significant on the basis of
  non-overlapping marginal intervals rather than a paired test.

## 6. Carried limitations

The learning curve is English, two tags, a single untuned encoder, and scores
exact span match on a dev subsample. It constrains the SHAPE of the
data-versus-accuracy relationship, not the absolute F1 attainable on Indonesian
five-tag data; absolute levels (0.42–0.54) sit below published SkillSpan
(0.577–0.648) by design and are not claimed as attainable targets. The
reliability projection assumes 1/sqrt(n) scaling from one measured point. The
density figures in section 2 are SkillSpan's, not this corpus's.
