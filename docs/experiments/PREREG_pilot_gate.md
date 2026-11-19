# Pre-registration — how pilot agreement is reported, and the go/no-go rule

**Status: REGISTERED 2026-11-19.** Proposed and approved by the researcher on the
same day, before any annotator was recruited and before any pilot annotation
existed. Not yet discussed with the promotor (see the approval table).

From this date the rule in section 4 is fixed. It may be rehearsed on synthetic
data as often as wanted, and the researcher asked for exactly that ("we'll see
if it's good after testing"). What it may NOT do is change after real pilot
annotations exist: if the thresholds are to be reconsidered, that has to happen
before the first expert saves a sentence, and be recorded here.

**Status at registration:** `pilot_100` and `practice_10` drawn; no annotator has
been recruited or has submitted work; `main_1200` not drawn; the three
validators have approved the instruments without revision (2026-11-18).

---

## 1. Why this is written down in advance

The pilot decides whether the study goes on to its most expensive step: three
experts annotating 1,200 sentences each. If the bar for "good enough" is chosen
after the pilot numbers are known, it is no longer a bar. So the measures, the
thresholds and what each outcome triggers are fixed here first.

## 2. What is reported for the pilot (always, whatever the result)

For the two sentence-level judgements, relevance and `has_5tag`:

- observed agreement;
- Fleiss' kappa and Krippendorff's alpha, each with its 95% bootstrap interval;
- **PABAK** (prevalence- and bias-adjusted kappa), and the share of each
  category across the three annotators.

Kappa is never reported alone. The pilot sample is deliberately skewed (15% of
it is borderline material so that NON_RELEVANT is a real option), and kappa
falls as one category dominates even when the annotators are equally accurate.
PABAK and the category shares are what make a low kappa interpretable.

For spans:

- exact-match and overlap (IoU ≥ 0.5) span F1 between each pair of annotators,
  with a bootstrap interval, overall and per tag;
- token-level Fleiss' kappa, the figure comparable to SkillSpan's published
  0.70–0.75.

Every figure is reported with the number of sentences it rests on.

## 3. What the pilot is NOT used to claim

With 100 sentences the 95% interval on these figures is about 0.3 wide: a kappa
of 0.73 has an interval running from roughly 0.58 to 0.85, which spans three of
the conventional bands. **The pilot therefore supports no reliability claim.**
It is calibration: it shows where the guideline is read differently. The
reliability figure the dissertation reports comes from the main 1,200.

## 4. The go/no-go rule

Three measures are read as point estimates:

| Measure | Why this one |
|---|---|
| relevance — PABAK | sentence-level, robust to the skewed sample |
| `has_5tag` — PABAK | sentence-level, robust to the skewed sample |
| spans — token-level Fleiss' kappa | span-level, with a chance baseline |

Each falls in a band:

| Band | Value |
|---|---|
| ok | 0.60 or higher |
| borderline | 0.40 up to 0.60 |
| low | below 0.40 |

The verdict:

| Verdict | Condition | What happens next |
|---|---|---|
| NOT READY | fewer than 95% of the sentences were judged by all three annotators | finish the annotation; read no agreement figure yet |
| GO | all three measures are ok | calibration session, freeze the guideline, start the main 1,200 |
| RECALIBRATE | no measure is low, at least one is borderline or undefined | calibration session; revise the guideline wording where disagreements cluster; the annotators do a fresh set of 30 sentences; run the check again on those |
| STOP | any measure is low | do not start the main 1,200; rework the guideline or the annotator training with the promotor |

Exact and overlap span F1 are reported beside the verdict but are not part of
the rule: there is no agreed benchmark for inter-annotator span F1 under a
five-tag scheme, so a threshold on it would be invented.

The thresholds 0.60 and 0.40 follow the boundaries of the conventional Landis
and Koch bands (0.41–0.60 moderate, 0.61–0.80 substantial). The researcher
accepted them as proposed, noting that she has no prior basis for the exact
values; that is the honest position, and it is why the pilot supports no
reliability claim (section 3) and why a RECALIBRATE verdict is treated as a
normal outcome, not a failure.

## 5. What may and may not change later

- The evidence sections of this document may be amended. The rule in section 4
  may not be changed once the first pilot annotation exists.
- A RECALIBRATE verdict is not a failure and is reported as what happened: the
  guideline was revised and re-checked on fresh sentences.
- If the rule is ever overridden, the override, the reason and who decided are
  recorded here, dated.

## 6. How it is enforced

`python -m gold_standard.gate <agreement report>` applies exactly the thresholds
above and writes its verdict next to the report. The thresholds in the code and
in this document are kept identical by a test
(`tests/test_after_annotation.py`). The check has been rehearsed end to end on
synthetic stand-in annotators (three language models), which is recorded as a
rehearsal and not as evidence.

---

## Approval

| | |
|---|---|
| Approved by the researcher | 2026-11-19 |
| Discussed with the promotor | not yet; to be raised before the pilot starts |
| Changes to the proposed thresholds | none |
