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

**One competency per cognitive level** (the researcher, 2026-11-20). "Apply
web programming" and "develop a website" are two competencies on two levels,
even when the postings' phrases for them fall into one group of meaning. So a
group is a FAMILY, and `structure.split_by_level` cuts it by the level of the
verbs its phrases open with. Nobody should read one level into a whole family,
and institutions of different levels use the result.

- The levels are the revised Bloom taxonomy: C1 remember, C2 understand, C3
  apply, C4 analyze, C5 evaluate, C6 create.
- Each level of a family that at least 2 postings ask for is a competency of
  its own: its own statement (a verb of that level + the knowledge the phrases
  of that level share), its own number of postings, its own tools, soft skills
  and work behaviours, its own status. Every competency lists its relatives:
  the other levels of the same family with their postings.
- A phrase with verbs of two levels ("design and maintain databases") counts
  for both. The postings of the levels therefore add up to more than the
  family's.
- Phrases that open with no verb ("programming", "unit testing") form a
  competency with NO level: the postings name the work and say nothing about
  the level. It is shown as "tidak dinyatakan lowongan", with the default verb
  "Perform" / "Melakukan" in its statement and a note that this verb sets
  nothing.
- A level is never taken from a verb the method supplied. (Before this
  correction a group named by the noun "pemrograman" was shown as "apply"
  because of its default verb "perform", while its postings wrote "membangun"
  and "menulis".)
- The public dashboard has a filter by level, so an institution keeps the
  levels it teaches to. Which levels suit which kind of institution is NOT
  decided anywhere in the code.

**A noun made from a verb carries the verb** (the researcher's correction,
2026-11-20). "Perform web application development" looks like C3 because of
"perform", but the work is developing: the competency is "Develop web
applications", on C6. `statement.read_phrase` reads such phrases through
`verbs/deverbal_nouns_v1.tsv` ("development" = develop, "testing" = test,
"pengujian" = menguji), also behind a verb that only says "do it" ("perform
code reviews" = review code).

- The level is that of the noun's verb, and the page says it was read from a
  noun.
- The object needs care in English: the words before the noun are its object
  in "web application development" and only describe it in "automated
  testing". So the corpus is asked. *Confirmed*: postings write the verb with
  those very words ("develop web applications"). *Completed*: the words open an
  object that at least 5 postings write with that verb ("mobile development"
  becomes "Develop mobile applications"). *Unconfirmed*: neither; the statement
  keeps the noun phrase ("Perform unit testing") with the level of "to test",
  and says that "perform" is wording, not a level.
- A noun phrase that names no work ("programming languages") still has no level.

**The verb table.** Which verb sits on which level is
`verbs/bloom_verbs_v2.tsv`: 203 verbs, English and Indonesian, each with the
cognitive process it is read as, the other levels where published lists place
it, and the basis of the row. `verbs/SOURCES.md` gives the sources (Anderson &
Krathwohl 2001; Krathwohl 2002; Stanny 2016 on why verb lists disagree) and
says plainly what is not sourced: software-work verbs such as "deploy" or
"debug" are the developer's reading, and the Indonesian rows are the
developer's translations. The table is **unreviewed**; every level on the page
follows from it.

**The generator's paragraph describes the family.** The earlier LLM generator
writes one text per group and blends levels on purpose ("Design ... Apply ...
Evaluate ..."). It is therefore kept as the description of the family, shown
under each of its competencies, and is not a competency statement of one level.

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

## How a competency statement is formulated

**The promotors' definition (2026-11-20):** a competency is a measurable
combination of knowledge, skills, abilities and behaviours needed to perform a
task or role; "develop web application" is a duty, not yet a competency, until
it says how well it is done, which skills or standards apply and what outcome
it achieves. Their examples were SKKNI unit titles ("Menggunakan struktur
data", "Menulis kode dengan prinsip sesuai guidelines dan best practices").

**The researcher's correction (same day):** those SKKNI titles are examples of
HOW a statement is formulated, not a format to translate the market's
competencies into. SKKNI is part of the problem the study addresses (the
programming standard dates from 2016); the competencies come from current
postings. Where a competency IS an SKKNI unit, the SKKNI wording may be used.

So every competency gets one statement in Indonesian, shaped like an SKKNI
title, with an English gloss (`formulate.py`):

    <operational verb on the competency's level> <the specific object of work>
        [standard or method: "sesuai prinsip REST"]  [quality: "yang aman"]  [outcome: "untuk ..."]

- The bracketed parts come from the competency's own postings
  (`qualifiers.py`): quality words (each with its ISO/IEC 25010
  characteristic), named practices and standards, purpose clauses. A part
  enters only when at least 5% of the competency's postings (and at least 3)
  ask for it. No tool or product name enters a statement.
- The nearest SKKNI units are found among all SKKNI documents in
  `DATA/master_skkni` (`skkni.py`; they range from 2015 to 2025). A model
  formulates the statement and says whether one of them is the same
  competency. A checker refuses a verb off the level, a tool, a number, or a
  standard or quality the postings did not give.
- The statement uses the verb the postings use for the work (or
  "Melakukan"/"Melaksanakan" + its work-noun: "Melakukan debugging"), never a
  stock verb that merely sits on the same level. Qualities describe an
  artefact (an application, code, an API), not requirements or performance.
- **SKKNI wording** is used when the competency is the SAME as an SKKNI unit
  (same work, same object, same scope; judged strictly) AND the SKKNI title's
  head verb is on the competency's level AND a second model of another family
  (`CONFIRM_MODEL`, Qwen), shown the competency and that one unit and nothing
  the writer decided, also says "same". The title is used exactly as written,
  optionally with one standard or outcome; the market's statement is kept
  beside it. Otherwise the market's statement is used; when only the levels
  differ, both levels are shown.
- Slips in the writer's SKKNI fields (a code beside "tidak", a "sebagian"
  without a code) are asked about once, then read cautiously with a note;
  they do not cost the statement.
- **Checked by independent judges (EXP-032).** Three blind judges of the SKKNI
  relation agree with each other almost perfectly (Fleiss' kappa 0.91 and
  0.87). On the 43 competencies the prompts were developed on, version 4.3
  used SKKNI wording 4 times, all right (before the second model: 11 wrong in
  38 uses). On 40 competencies drawn at random from the others: 1 SKKNI
  wording, wrong (a mixed family behind a clean statement); 12 of 38
  statements have a major defect for at least one of two reviewers, most of
  them because the "competency" is not work (a work behaviour, a purpose
  clause, employer branding) or is a mixed family. The formulation turns such
  input into a fluent line; it does not yet refuse or flag it.
- A competency with a general object ("applications", "solutions") is flagged
  "terlalu umum".

```powershell
.venv-ml\Scripts\python.exe -m competency.skkni parse-all
.venv-ml\Scripts\python.exe -m competency.formulate --run DATA\competency\runs\<run> --sample 40 --max-usd 0.40
# every candidate, leaving out those already formulated (costs credit: about $0.0023 each)
.venv-ml\Scripts\python.exe -m competency.formulate --run DATA\competency\runs\<run> --all --skip-existing --max-usd 2.00
```

An earlier reading of the promotors' message (same day) wrote each competency
out as a whole SKKNI unit, with elements and performance criteria (EXP-031).
That was not what was meant, and the unit writer was removed.

## Two clean-ups of the competency list (EXP-033)

- **Fragments are held back.** A statement whose phrases were cut before
  their object ("Write clean", "Design scalable", "Perform Design, develop")
  names no work; `statement.is_fragment` finds it (no plain noun after the
  verb; only verb forms, or a trailing quality word or participle) and the
  contract gives it the blocker `statement_is_a_fragment`. The formulation
  never sends a fragment to the model.
- **Families that give the same statement on the same level are merged**
  (`structure.merge_same_statement`): "develop web applications" (verb) and
  "web application development" (noun) were two families with one statement.
  Only a statement that names specific work links families: a first run that
  let generic statements link ("Develop solutions", "Troubleshoot issues")
  chained unrelated work into catch-all families of thousands of postings and
  was withdrawn.

## Before anything is counted: only duty and requirement sentences

A posting also describes the company, lists benefits and tells the reader how
to apply. The rule extractor tags any phrase that opens with an action verb, so
"support you and your family" and "apply early" became hard skills and then
"valid" competencies: in a random sample of 24 valid families read on
2026-11-20, about 10 were of that kind. `governance/zoning.py` removes such
sentences first.

- Section headers in the postings ("Responsibilities", "Qualifications", "What
  we offer", "Tentang kami") label the sentences under them. These labels are
  weak: nobody checked them.
- A classifier learns from them and judges the sentences no header covers. It
  drops a sentence only when the probability that it is a duty or requirement
  is under 0.5.
- **It is not validated against people.** The cut-off was set from the
  developer's reading of 64 sentences. The gold standard will show how many
  expert-tagged spans sit in sentences this filter drops; that is the number
  to report. `--no-zone-filter` switches it off, to measure what it changes.

## Families that are the same work are merged

The clusterer separates phrases by wording, so "REST API development",
"integrating REST APIs" and "design and implement RESTful APIs" came out as
three families (119 pairs of families with centres at cosine 0.90 or more). With
`--all`, families whose centres are that close are merged
(`grouping.merge_similar`, complete linkage: every pair in a merged family is
that close). The threshold is a starting value.

## How sure a count is

`smoothing.py` adds to every competency: the 95% interval of its share of
postings; the range of ranks it takes when the postings are resampled 200
times (two competencies whose ranges overlap are not shown to differ); and its
share among the levels of its family, estimated together with all families so
that a level seen in three postings of a small family is not read as a firm
percentage. Rank stability over time is reported for families as well as for
competencies.

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

1. The verb table (`verbs/bloom_verbs_v2.tsv`): review it row by row, or name
   the list your institution uses for the Indonesian verbs.
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
