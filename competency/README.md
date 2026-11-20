# competency — from five-tag evidence to competency statements

This folder turns extracted spans into competencies, in the order the proposal
gives (§3.3.1): an evidence table, groups with a validity status, one
assessable statement per valid group, KKNI and okupasi, and a status with its
reasons.

**Status, 20 November 2026: built, tested, and run on the whole corpus as an
engineering run. Nothing it has printed is a finding.** The evidence in that
run came from the rule extractor (E0), whose accuracy has not been measured
against the gold standard.

## What a competency is

**Decided by the researcher on 2026-11-20: a competency is a verb plus a
knowledge.** "Develop machine learning algorithms": *develop* is the verb,
placed on the taxonomy by the verb table; *machine learning algorithms* is the
knowledge it acts on. The statement is those two and nothing else. It names no
tool: "Perform programming", not "Perform programming using Python, PHP and
JavaScript". For each competency the most demanded tools and technologies, soft
skills and work behaviours are listed beside it.

How the two parts are read from the postings (`structure.py`):

- **Verbs.** Every hard-skill phrase of the group that opens with a verb of the
  table gives its verb or verbs ("design, develop and test web apps" gives
  three). Each verb is listed with its taxonomy level and its postings.
- **Knowledge.** Postings rarely write the same object twice ("scalable REST
  APIs for internal services", "secure REST APIs", "REST APIs"). So the object
  is cut at the first preposition, its head word is taken (the last word in
  English, the first in Indonesian), and the most shared head is lengthened one
  word at a time while more than half of its postings keep the longer form:
  "APIs" becomes "REST APIs". A group lists several such knowledge objects,
  each with its postings and the verbs used with it.
- **The statement** is the most shared knowledge object with the verb most used
  with it. A group made mostly of noun phrases ("programming", "unit testing")
  has no verb of its own and gets the default "Perform" / "Melakukan".

**The cognitive level** (the researcher's correction, 2026-11-20). The level
shown for a competency is the level the POSTINGS ask for, read from their own
verbs, with its Bloom code: C1 remember, C2 understand, C3 apply, C4 analyze,
C5 evaluate, C6 create. It is never the level of a verb the method supplied.
Before the correction a group named by the noun "pemrograman" was shown as
"apply" because its statement had the default verb "perform", while its
postings wrote "membangun" and "menulis" (create).

- Each posting counts once for every level its verbs sit on.
- The level shown is the one most postings ask for (a tie goes to the higher).
  Beside it: the whole distribution, and the highest level that at least a
  fifth of the verb-using postings ask for.
- The statement's verb is chosen on that level when the postings offer one
  ("Build websites", not "Maintain websites", when most postings say build).
- A group whose postings use almost no verb has NO level; the page says so.
- Which verb sits on which level is the verb table in `statement.py`. It is the
  developer's (`verbs-v1-unreviewed`): it must be reviewed by the researcher,
  because every level on the page follows from it.

Why the knowledge is read from the phrase and not from the KNOWLEDGE tag: the
extractor tags a term KNOWLEDGE when it stands alone ("knowledge of design
patterns"); when a verb governs it, the whole phrase is a hard skill. On the
whole pool under 1% of hard-skill phrases contain a term that is tagged
KNOWLEDGE anywhere (EXP-028).

There is no part-of-speech tagger in this step. "Head" is a rule about word
order and a list of prepositions, so a span the extractor cut short ("Write
clean") gives a wrong object. Each statement carries the share of the group's
postings that support its object (`object_support` in the CSV), so a weak one
can be seen.

## The map of knowledge and technology

`termmap.py` draws every frequent knowledge term and technology as a point.
Two points are close when the same postings ask for both; points are grouped
into clusters ("JavaScript, Git, React"; "AWS, Azure, Docker"; "Python, SQL,
Power BI"). It is written to `term_map.json` by every run, travels with a
published profile, and is the tab **"Peta pengetahuan & teknologi"** of the
public dashboard: filter by kind and by number of postings, put one cluster in
focus, see its terms and the competencies that use them. A still picture for a
document is drawn by `python -m competency.figures run <folder>`
(`figures/term_map.png`).

The clusters are of co-occurrence, not of meaning, and their number is a
setting. A language is also a cluster: Indonesian terms are asked for in
Indonesian postings.

```powershell
.venv-ml\Scripts\python.exe -m competency.termmap DATA\competency\runs\e0_pool_all
```

## How the five tags are used

**Decided by the researcher on 2026-11-19: competencies come from hard skills.
Everything else describes a competency.**

| Tag | Role | How it is attached |
|---|---|---|
| SKILL_HARD | What a competency is built from: a group of hard-skill phrases that mean the same work. | — |
| TOOL_TECHNOLOGY | The tools and technologies mostly used for the competency. | Named in the same **sentence** as one of its phrases. Counted in postings. |
| SKILL_SOFT | The soft skills expected with the competency. | Asked for in the same **postings**. |
| WORK_BEHAVIOR | The work behaviours expected with the competency. | Asked for in the same **postings**. |
| KNOWLEDGE | Other knowledge asked for with the competency. The knowledge the competency is ABOUT is the object of its verb (section above), not this list. | Same postings (`knowledge`); the few named in the same sentence are kept beside it (`knowledge_context`). This list is my reading. |

For soft skills, work behaviours and knowledge every line carries two numbers:

- **share**: of the postings that ask for this competency, the part that also
  ask for the item ("teamwork: 41%");
- **lift**: that share divided by the item's share among all postings. Above 1
  means it is asked for more often with this competency than in general. A soft
  skill with a high share and a lift near 1 is asked for everywhere and says
  nothing particular about this competency.

Why the sentence for technology and the posting for the others: a tool is
normally named with the work it is used for ("develop REST APIs using Spring
Boot"), while soft skills and behaviours sit in their own lines of a posting.
Sharing a posting is weaker evidence than sharing a sentence: it shows that
employers who want the competency also want the soft skill, not that the soft
skill is used in that work.

This is also the shape of an SKKNI unit: the unit of competency, then the
knowledge, the work attitude and the equipment it needs.

The earlier design of the same day (competencies from hard-skill AND knowledge
phrases, soft skills as competencies of their own) is still reachable for an
ablation: `GroupingConfig(competency_tags=("SH", "KN", "SS"))`. The runs
EXP-017, EXP-024, EXP-025 and EXP-026 used it.

**KKNI is information only** (the researcher, 2026-11-19). The proposed level
tells the reader where a competency sits. It is not part of the status and does
not order the list; a test holds this. The one place a level is compared is the
recommendation "meninjau ulang jenjang", which needs a person to enter the
curriculum's level and fires only when the proposed level is marked confident.

## Two ways to write the statement

| | What it gives | Cost |
|---|---|---|
| Rules (default) | "Build data pipelines": the verb and the knowledge the group's phrases share. Short, and every word is from the postings. No tool is named. | free |
| The earlier generator (`--statements v2 --v2-max-groups 20 --max-usd 0.80`) | Two to four sentences that abstract from the postings, with a rationale, soft skills and a proposed KKNI level. It is the generator of the pre-approval system (`competency_generator_v2`), fed with five-tag groups through `v2_generator.py`. It drops a statement that names a product or method no source sentence names. | paid; capped before the first call |

A generator statement is judged by the generator's own evidence filter, not by
the share of its words found in the evidence (it is asked to abstract, so that
share is always low). The first trial is EXP-026.

## One command

```powershell
.venv-ml\Scripts\python.exe -m competency.pipeline e0 --n-sentences 20000 --tag e0_pool20k
```

This is the P1 shape (rules only, no LLM, free). It takes about four minutes
and writes to `DATA\competency\runs\<tag>\` (local only; the files quote corpus
sentences). Open `competencies.csv` in Excel to read the result.

The whole corpus (free, about 13 minutes):

```powershell
.venv-ml\Scripts\python.exe -m competency.pipeline e0 --all --tag e0_pool_all --min-postings 5
.venv-ml\Scripts\python.exe -m competency.figures run DATA\competency\runs\e0_pool_all
```

## What the whole-corpus run showed (engineering run, 18,667 postings; EXP-027)

This is the run in the current design: competencies from hard skills only.

- 470,280 evidence units: 166,182 hard-skill, 179,969 technology, 61,881
  soft-skill, 31,565 knowledge, 30,683 work-behaviour.
- 684 groups of hard-skill phrases, 667 of them valid. 667 statements: 477
  "kandidat tinjauan", 190 "eksploratif" (189 because no okupasi matched).
- Every competency has soft skills attached, 666 have work behaviours, 660
  knowledge, 431 at least one tool.
- Ranking between consecutive half-year windows: rank correlation 0.75, 0.91,
  0.76. The first attempt of the same night, with the earlier design and a
  faulty grouping, gave 0.34, 0.69, 0.26 and must not be quoted.

What is visibly weak:

- **"communication" is on almost every competency** (about half of its
  postings, lift near 1). Soft skills and behaviours that are asked for
  everywhere do not characterise a competency; the lift column is there to
  separate them from the specific ones.
- **Lift across languages is not comparable.** Indonesian postings are about 4%
  of the pool, so an Indonesian soft skill gets a lift of 10 to 20 on an
  Indonesian-language competency. That is the language, not the competency.
- **Spelling variants are separate lines** ("problem-solving", "problem solving").
- **Some rule statements are an accident**: a fragment of one posting, or an
  employer's name. The earlier generator would not do this; it has not been run
  on these groups yet.
- 37% of hard-skill units are in no group.

## What the first run showed (engineering run, 581 postings; earlier design)

These numbers are from the design that built competencies from hard-skill AND
knowledge phrases and grouped soft skills separately.

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
- **KKNI was assigned to only 17 of 445** in the first run, because the existing
  labeller was built to correct a level suggested by an LLM and stays silent
  unless it is confident. Repaired the same evening: every statement now gets
  the nearest level with its similarity and a "confident" flag. A level that is
  not confident is a starting point for an expert, not a placement.
- **Almost everything is in English** (433 of 445): only about 4% of the pool's
  sentences are Indonesian.

## Decisions that are yours

1. Whether a competency may carry several verbs. One group often holds
   "design", "build" and "maintain" for the same knowledge; the statement uses
   the most frequent one and lists the others. They sit on different levels of
   the taxonomy, so this matters for the KKNI level and for assessment.
2. **What counts as demand: postings, or employers.** Each competency reports
   both the number of postings and the number of employer/template groups. One
   employer repeating an advertisement inflates the first, not the second.
3. The two contract thresholds (at least 5 postings, grounding at least 0.80):
   starting values carried over from the pre-approval pipeline.
4. The verb table that assigns Krathwohl levels (`statement.py`). I wrote it;
   no expert has reviewed it. The version name says so.

## The other modules in this folder (added 19 November, evening)

| Module | What it does | Proposal |
|---|---|---|
| `temporal.py` | Share per half-year window, trend with a correction for many tests, volatility, a persistence weight, and moving a volatile item up to its ESCO parent node. `python -m competency.temporal --tag <name> --tags TT KN SS WB` runs it on the whole pool. | §3.3.4 |
| `coverage.py` | Proposes one of the four coverage statuses against the SKKNI unit titles or an uploaded curriculum, and lists the nearest reference units. Always marked "not validated by an expert". | §3.3.5 |
| `delta.py` | The five kinds of recommendation on a curriculum document. "Mengurangi penekanan" is capped at eksploratif. The persistence weight orders recommendations inside a status and never changes a status. | §3.3.6 |
| `profile.py` | Publishes "Profil Kompetensi RPL <versi>" with a change log (naik, turun, masuk, keluar, with reasons). A version is written once. A profile from an engineering run goes to `drafts/` and says so. | §3.3.7 |
| `endtoend.py` | P1, P2, P3 and the two ablations on the same postings, with one comparison table. `python -m competency.endtoend --tag <name> --n-postings 12 --max-usd 1.20`. | §3.3.3 |
| `figures.py` | Every chart, drawn from the output files. `python -m competency.figures run <folder>` (also `temporal`, `endtoend`, `experiments`). | |

The dashboards that show all this are in `dashboard_v3/`.

Thresholds that are STARTING VALUES and need an expert or the promotor before
any release: the time-analysis thresholds and weight formula
(`TemporalConfig`), the coverage thresholds 0.65 and 0.45 (`CoverageConfig`),
the contract thresholds, and the verb table.

## Not built yet

The education-level distribution (it needs the education field of the postings,
which the sentence pool does not carry).

## Reused, not rewritten

The clusterer (`clustering/`), the cluster validity audit
(`pipeline_scientific_audits.py`), the KKNI labeller and the okupasi mapper.
The old statement generator (`competency_generator_v2`) is reused through
`v2_generator.py`; it is not edited. The old end-to-end driver
(`scripts/run_full_v2_pipeline.py`) is untouched and still serves the old
dashboard.
