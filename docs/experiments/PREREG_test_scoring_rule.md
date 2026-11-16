# Pre-registration: the test-set scoring rule, and how AMBIGUOUS/EXCLUDE adjudications are scored

**Registered:** 2026-11-16, before any adjudication exists and before the sealed
test has ever been opened.

**Status at registration (verified, not asserted):**
`DATA/gold_standard/splits/` contains no split directory, `test_open_log.jsonl`
does not exist, `main_1200` has never been drawn, and no annotator has submitted
work. No number reported anywhere in this project depends on the choice
registered below. That is the only condition under which this choice can be made
honestly, which is why it is being made now rather than after the adjudication.

This document fixes the *scoring* rule. The *sample size* is registered
separately in `PREREG_gold_standard_size.md`; nothing here revises it.

---

## 1. The problem being pre-committed on

The scoring population is every sentence in `test_blind.parquet` — including
sentences that carry no gold span, so that a span predicted on a non-skill
sentence costs precision instead of vanishing. That rule is correct and is not
revisited here.

But it created a second question that the rule alone does not answer. After
adjudication, a span may be dropped as **AMBIGUOUS** or **EXCLUDE**: the three
annotators could not be brought to an agreed decision about it. Such a span is
not gold. If the sentence has no other adjudicated span, it then enters the
scoring population with an *empty* gold set — indistinguishable from a sentence
the annotators positively judged to contain no skill.

The consequence is not hypothetical. A model that predicts exactly the span the
human adjudicators marked undecided is charged a false positive it had no way to
avoid. Published precision is then deflated in proportion to the adjudication's
own ambiguity rate — a property of the annotation process, not of the model.
Measured on the registered repro (10 sealed sentences; 2 ACCEPT spans, 1
AMBIGUOUS, 1 EXCLUDE; a model that reproduces all four adjudicated spans): strict
precision **0.50**, whereas the model agreed with every decision the humans
actually made.

The deflation is unknowable in advance, because the ambiguity rate of an
adjudication that has not happened yet cannot be estimated. This is therefore
exactly the kind of choice that must be fixed before the data exist.

## 2. Registered decision

**Primary metric: strict micro span-F1** (`result["micro"]`), unchanged, exactly
as `MATCH_RULE_TEXT` defines it: exact span + type match, micro-averaged over
*all* sealed test sentences, with a predicted challenge span counting as a false
positive. This is the conservative reading and the one already written into the
module, so the headline number is not moved by this registration.

**Secondary metric: challenge-neutral micro span-F1**
(`result["micro_challenge_neutral"]`). A prediction whose
`(sentence_id, span_start, span_end, tag)` matches a row of
`labels/test_challenge_set.parquet` exactly is removed from `n_pred`: not
credited as a true positive (it is not gold), not charged as a false positive
(the gold standard does not assert it is wrong — it asserts the humans could not
agree). The exemption is deliberately the narrowest defensible one:

- a span that merely **overlaps** a challenge span is still a false positive;
- the same offsets with a **different tag** are still a false positive;
- the exemption never adds to `tp`, so it cannot raise recall.

**Both are reported together, always.** `challenge_cost_precision_delta` and
`challenge_cost_f1_delta` publish the gap. Citing either metric alone is a
reporting error, because the gap is the quantity that says how much of the
measured precision shortfall belongs to the annotation rather than the model.

## 3. What is forbidden by this registration

1. The primary metric may **not** be switched to the challenge-neutral one after
   seeing which is more favourable. The primary is `micro`, fixed here, before
   any adjudication.
2. The exemption may **not** be widened after results are known — not to
   overlapping spans, not to label-insensitive matches, not to "spans near a
   challenge span". Any such change invalidates this registration and must be
   reported as a post-hoc analysis.
3. A challenge span may **not** be backfilled into the strict gold. A span the
   adjudicators could not decide is not evidence that the span is correct.
4. The challenge set may **not** be edited after `materialise_labels` writes it.
   It is digested in `SEAL.json` (`test_challenge_set_sha256`) and a mismatch
   voids the score, so an edit that would buy precision is refused rather than
   silently honoured.

## 4. Enforcement (so this is a mechanism, not a promise)

- `gold_standard.seal.materialise_labels` **writes**
  `labels/test_challenge_set.parquet` and records its sha256. Previously the
  dropped rows were counted and discarded while `SEAL.json` claimed they were
  "kept for qualitative analysis"; that claim was false and is now true.
- `score_on_test` **refuses to score at all** when `SEAL.json` records dropped
  adjudications but the challenge file is absent. Scoring without it is possible
  in principle but would make the deflation invisible and unquantifiable, which
  is the defect this registration exists to prevent.
- A span appearing in both the strict gold and the challenge set is refused: it
  cannot be both credited and exempt.
- The empty-gold sealed sentences are partitioned and published as
  `n_non_skill_sentences` (annotators found no skill) and
  `n_challenge_only_sentences` (annotators could not decide); the partition is
  asserted to be exact at materialisation time.
- Both F1s and `n_pred_spans_matching_challenge_spans` are written into the
  single-open receipt in `test_open_log.jsonl`, so the pair cannot be separated
  after the fact.

Regression tests: `tests/test_split_seal.py` (27 tests), of which 11 cover this
rule, including the negative controls that the exemption is not a general
amnesty, that an inexact overlap and a wrong tag are still charged, and
`test_old_conflated_rule_is_the_negative_control`, which fails if the challenge
set stops being read.

## 5. Expected reporting shape

> On the sealed test (n = 240 sentences), exact span+type micro-F1 was X.XXX
> (95% CI [·, ·]). Of the 240 sentences, N carried no gold span: N₁ judged to
> contain no skill and N₂ whose only adjudicated spans were dropped as
> AMBIGUOUS/EXCLUDE. Treating the latter's spans as neither correct nor
> incorrect gives a challenge-neutral micro-F1 of Y.YYY; the gap of (Y − X)
> bounds the share of the measured error attributable to items the annotators
> could not resolve.

## 6. Carried limitations

- The exemption rests on exact-offset matching, so a correct prediction that
  tokenises a challenge span slightly differently is still charged. This is
  conservative and intentional, but it means the challenge-neutral number is a
  *lower* bound on the ambiguity cost, not a point estimate of it.
- `MATCH_RULE_TEXT` claims comparability with SkillSpan / NNOSE / Skill-LLM /
  SAP-IE while scoring label-sensitively, where the SkillSpan-comparable mode in
  `gold_standard.iaa` is label-insensitive. That overclaim is recorded in
  `split_reattack_findings.md` and is **not** fixed by this registration; the
  cross-paper comparison needs its own label-insensitive variant before any
  number is cited against published work.
- Nothing here constrains the *size* of the challenge set. If the adjudication
  turns out to drop a large share of spans, the strict precision becomes hard to
  interpret no matter which metric is cited, and that is a finding about the
  annotation instrument to report as such — not a reason to change the rule
  registered above.
