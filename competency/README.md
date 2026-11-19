# competency — from five-tag evidence to competency statements

This folder turns extracted spans into competencies, in the order the proposal
gives (§3.3.1): an evidence table, groups with a validity status, one
assessable statement per valid group, KKNI and okupasi, and a status with its
reasons.

**Status, 19 November 2026: built, tested, and run once on real corpus text as
an engineering run. Nothing it has printed is a finding.** The evidence in that
run came from the rule extractor (E0), whose accuracy has not been measured
against the gold standard.

## How the five tags are used

| Tag | Role |
|---|---|
| SKILL_HARD, KNOWLEDGE | What competencies are built from (the technical stream). |
| SKILL_SOFT | Grouped separately, as supporting competencies. |
| TOOL_TECHNOLOGY | Never a competency by itself. Attached to the groups whose evidence it shares a **sentence** with, and named in the statement as context. |
| WORK_BEHAVIOR | Reported as a profile of expected dispositions, not turned into competencies. |

This follows the proposal's sentence that a technology name is "bukti teknologi
yang masih harus dikaitkan dengan tindakan kerja". It is a design decision and
it is yours to confirm.

## One command

```powershell
.venv-ml\Scripts\python.exe -m competency.pipeline e0 --n-sentences 20000 --tag e0_pool20k
```

This is the P1 shape (rules only, no LLM, free). It takes about four minutes
and writes to `DATA\competency\runs\<tag>\` (local only; the files quote corpus
sentences). Open `competencies.csv` in Excel to read the result.

## What the first run showed (engineering run, 581 postings)

- 14,451 evidence units: 5,138 hard-skill, 1,007 knowledge, 1,874 soft-skill,
  5,495 technology, 937 work-behaviour.
- 528 groups: 445 valid (sahih), 27 to be split, 25 low-frequency, 24 support
  process, 4 out of domain, 3 mixed domain.
- 445 statements: 271 "kandidat tinjauan", 174 "eksploratif" (121 because fewer
  than 5 postings, 81 because no okupasi matched).
- No statement can reach "siap ditelaah formal" by itself: that needs three
  facts only a person can set (corpus valid for the Indonesian claim, expert
  review done, stability checked).

What is visibly weak, and why:

- **The statements are stiff.** "Perform programming using Python, Java and
  C++" is built only from words the postings used. That is what the no-LLM
  pipeline can do; the LLM path exists to write better ones, and its answers
  are checked against the evidence.
- **The LLM statement writer was tried on the 40 most frequent groups**
  (`openai/gpt-5.4-mini`, $0.026). 30 answers passed the checks; 10 were
  rejected and replaced by the rule statement: 6 named technologies that are
  not in the group's evidence, 3 were not grounded enough, 1 did not start with
  an allowed verb. The accepted ones are often better ("Develop software
  products" where the rule statement carried an employer's name) but several
  simply list every technology of the group; the prompt needs a limit. The
  comparison is in `llm_statements_trial.json` in the run folder.
- **Some statements are noise** ("Design reusable", an employer's name inside a
  statement). They come from the rule extractor's mistakes, which pass straight
  through. A better extractor gives better competencies; this layer does not
  repair the layer before it.
- **KKNI was assigned to only 17 of 445.** The existing KKNI labeller was built
  to correct a level suggested by an LLM and stays silent unless it is
  confident; with no suggestion to correct, it mostly says nothing. It needs
  rework for this path.
- **Almost everything is in English** (433 of 445): only about 4% of the pool's
  sentences are Indonesian.

## Decisions that are yours

1. The roles of the five tags (table above).
2. **What counts as demand: postings, or employers.** Each competency reports
   both the number of postings and the number of employer/template groups. One
   employer repeating an advertisement inflates the first, not the second.
3. The two contract thresholds (at least 5 postings, grounding at least 0.80):
   starting values carried over from the pre-approval pipeline.
4. The verb table that assigns Krathwohl levels (`statement.py`). I wrote it;
   no expert has reviewed it. The version name says so.

## Not built yet

The gap audit against official references, the education-level distribution,
the rolling time windows (persistence and volatility), the delta
recommendations, and the versioned profile.

## Reused, not rewritten

The clusterer (`clustering/`), the cluster validity audit
(`pipeline_scientific_audits.py`), the KKNI labeller and the okupasi mapper.
The old end-to-end driver (`scripts/run_full_v2_pipeline.py`) and the old
statement generator are untouched and still serve the old dashboard.
