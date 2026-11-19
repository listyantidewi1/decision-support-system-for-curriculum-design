# Registration — where a D1/D2 conflict goes (risk rule)

Status: **REGISTERED 2026-11-19**, chosen by the researcher (Listyanti Dewi
Astuti). Not yet discussed with the promotor. Registered before any real
conflict has been adjudicated: no gold standard exists and the LLM adjudicator
has not been run.

Code: `fivetag/conflict_routing.py`, version **`d3-routing-v1`**. A test
(`tests/test_fivetag.py`) fails if the constants in the code and in this
document differ.

## What the proposal requires

§3.3.1: when the two deterministic classifiers D1 (lexicon–regex–priority) and
D2 (taxonomy-anchored context decision) give different classes, the case is
labelled CONFLICT and "diteruskan ke mekanisme adjudication/LLM atau human
review sesuai tingkat risikonya", so that a conflict is not hidden by an
automatic choice that cannot be explained.

The proposal does not say what the risk levels are. This document does.

## The rule

Every conflict is first decided by the LLM adjudicator. Then:

| Risk | Which conflicts | Who decides |
|---|---|---|
| **high** | The span is evidence under a competency whose status is `kandidat_tinjauan` or `siap_ditelaah_formal` | A human. The LLM's answer is shown as a suggestion. |
| **audit** | A random sample of **150** of all other conflicts, drawn with seed **20261119** | A human. Used only to measure the LLM adjudicator. |
| **low** | Everything else | The LLM's decision stands. |

A conflict also goes to a human, whatever its risk, when the LLM's answer
cannot be used (not valid JSON, not an allowed label) or names a label that
neither classifier proposed.

"One classifier decided, the other abstained" counts as a conflict and is
reported separately from a true disagreement.

## Why this rule

- Human attention goes where a wrong tag could reach a curriculum
  recommendation: under the competencies about to be put in front of experts.
- The part left to the LLM is not taken on faith. The audit sample gives its
  accuracy with a confidence interval, and that figure is reported.
- Reading every conflict by hand is not feasible on the full corpus, and
  leaving every conflict to the LLM would put an unchecked model at the point
  the proposal says must stay explainable.

## Why 150

With 150 audited conflicts, an adjudicator that is right 90% of the time is
estimated to within about ±5 percentage points (Wilson 95% interval roughly
84%–94%). That is enough to tell "good enough to trust" from "not". If a run
has fewer than 150 non-high conflicts, all of them are audited.

150 is a **starting value**. It may be changed before the first real
adjudication. After real conflicts have been adjudicated it is changed only as
a new version of the rule, with the old results kept.

## What is reported

For every run: the number of conflicts and the conflict rate, split by kind
(disagreement, one abstained) and by risk; how many went to a human; how many
LLM answers were unusable; and, once the audit sample has been reviewed, the
LLM adjudicator's accuracy with its 95% interval. The accuracy is computed on
the audit sample only, because the high-risk conflicts are not a random sample.

## Not decided here

- Which model is the adjudicator, and its prompt. To be chosen on development
  data and registered before it is used on real conflicts.
- Who the human reviewer is.
- The acceptable accuracy for the adjudicator. To be agreed with the promotor
  before the audit result is seen.

## Addition, registered 2026-11-19 — when the adjudicator may be trusted

Version **`d3-acceptance-v1`**. Proposed by the researcher and agreed by the
promotor on 2026-11-19, before any audit sample has been reviewed. It settles
the third item of "Not decided here" above; the routing rule itself is
unchanged.

The adjudicator is judged on the audit sample only, by the **lower bound of the
Wilson 95% interval** of its accuracy (the share of audited conflicts where its
label equals the human reviewer's). The observed share alone is never the
criterion: with 150 items it carries a margin of about five points.

| Lower bound of the 95% interval | With 150 audited | Verdict |
|---|---|---|
| at least **0.80** | **130** or more correct | **Accept.** The LLM's decisions on low-risk conflicts stand. |
| at least **0.667**, below 0.80 | **112** to 129 correct | **Borderline.** A second audit sample of 150 is reviewed and this rule is applied once more to the 300 pooled. |
| below 0.667 | fewer than 112 correct | **Reject.** The LLM adjudicator is not used. Low-risk conflicts stay CONFLICT and are left out of the evidence. |

In the borderline case another model may be tried instead of a second sample.
That is a new adjudicator: it needs its own audit sample and its own verdict.

Reported with the verdict: the audit accuracy with its interval, and, beside it,
how often two experts give the same tag to the same span in the gold-standard
annotation. The adjudicator is not expected to exceed the experts' agreement
with each other, and the two figures are to be read together.

Anchors for 0.80 and 0.667: the levels Krippendorff gives for drawing
conclusions and for tentative conclusions in content analysis. They are
borrowed as conventions; his levels refer to an agreement coefficient, not to a
plain accuracy. The citation is to be verified by the researcher before it is
used in the dissertation.

## Evidence (amend this section only)

2026-11-19 — rehearsal on the engineering run `e0_pool20k` (rule-extractor
evidence, 581 postings, 14,451 evidence units; not a result).

- 130 conflicts, 0.9% of the evidence: 31 true disagreements, 99 where one
  classifier abstained. Most frequent: D1 abstains / D2 says technology (69),
  D1 hard skill / D2 knowledge (28), D1 technology / D2 abstains (17).
- Risk: 26 high (under a competency with status `kandidat_tinjauan`), 104
  others. Because there are fewer than 150 others, all 104 fall in the audit
  sample, so in a run of this size every conflict goes to a human. On the full
  corpus the audit sample stays at 150 while the low-risk remainder grows.
- A trial adjudicator (`anthropic/claude-haiku-5.5`, not a registered choice)
  was asked about all 130 after the rule was registered: 130 usable answers, 0
  unusable; it sided with D2 in 99, with D1 in 22, and named a label neither
  classifier proposed in 9 (these go to a human by the rule). Cost: $0.004.
- No human has reviewed the audit sample, so the adjudicator's accuracy is not
  known.

2026-11-19 — the researcher reports that the promotor agreed to this rule. The
accuracy the adjudicator must reach is still to be fixed, before the audit
result is seen.

2026-11-19 — candidate adjudicators compared on 329 reviewed development spans
(experiment log, EXP-020): Qwen 3.8 Flash 95.7%, Claude Haiku 5.5 93.3%,
GPT-5.4 mini 90.9%. Only 4 of those spans are real D1/D2 conflicts, so this is
a comparison of span tagging in general. No adjudicator is registered yet.

2026-11-19 — the acceptance rule above (`d3-acceptance-v1`) was registered on
the same day, after the promotor agreed to it and before any audit sample was
reviewed. In the borderline case the researcher's proposal said "a second
sample, or another model"; the registration makes that precise as written
above.
