# The verb table: what it rests on, and what it does not

`bloom_verbs_v2.tsv` says, for each verb, which cognitive process of the
revised Bloom taxonomy it is read as. Every cognitive level the pipeline shows
(C1 remember … C6 create) follows from this file.

**Status: unreviewed.** The developer wrote it on 2026-11-20 after the
researcher asked for a written, detailed table. No expert has checked it. The
version name says so (`verbs-v2-sourced-unreviewed`). The researcher's review
is the next step; a change to a row is a new version.

## The framework

Anderson, L. W., & Krathwohl, D. R. (Eds.). (2001). *A taxonomy for learning,
teaching, and assessing: A revision of Bloom's taxonomy of educational
objectives.* New York: Longman.

Krathwohl, D. R. (2002). A revision of Bloom's taxonomy: An overview. *Theory
Into Practice, 41*(4), 212–218. https://doi.org/10.1207/s15430421tip4104_2
(Citation, pages and DOI checked online on 2026-11-20. The article states that
the cognitive process dimension has six categories and 19 specific processes.)

The 19 processes, as used in the `process` column:

| Level | Processes |
|---|---|
| C1 remember | recognizing, recalling |
| C2 understand | interpreting, exemplifying, classifying, summarizing, inferring, comparing, explaining |
| C3 apply | executing, implementing |
| C4 analyze | differentiating, organizing, attributing |
| C5 evaluate | checking, critiquing |
| C6 create | generating, planning, producing |

**To verify before citing:** the 19 names above, and the "alternative names"
that rows with basis `AK` rely on (for example *testing* and *monitoring* under
checking; *designing* under planning; *constructing* under producing; *using*
under implementing; *identifying* under recognizing), were written from the
developer's memory of the book's table of the cognitive process dimension. The
online check confirmed the count of 19 and the two processes of "remember"
only. Check the rest against the book or Table 3 of Krathwohl (2002).

## Why a verb alone does not fix a level

Stanny, C. J. (2016). Reevaluating Bloom's taxonomy: What measurable verbs can
and cannot say about student learning. *Education Sciences, 6*(4), 37.
https://www.mdpi.com/2227-7102/6/4/37 (open access; the DOI is 10.3390/educsci6040037 by the journal's pattern, not checked; the article itself was checked online on
2026-11-20: she analysed 30 published lists of "measurable verbs" and found 176
verbs that at least four lists place on a level; many verbs are placed on
several levels by different lists.)

This is why the table has an `also` column, and why the pipeline never reports
one level for a whole group of phrases. "Develop" is on C6 in most lists and on
C3 in some; "test" is C5 in the framework (checking) and C3 in everyday use.
The level of a competency is therefore reported together with the verbs it was
read from and the number of postings behind it.

## The shape "verb + knowledge, with dispositions beside it"

ACM/IEEE-CS *Computing Curricula 2020* (CC2020) describes a competency as
knowledge, skills and dispositions applied in a task context; the later CS2023
report maps skill levels to the revised Bloom taxonomy (found online on
2026-11-20 through secondary descriptions; the reports themselves were not read
for this file — read them before citing). This is the same shape the researcher
chose on 2026-11-20: a competency is a verb (the skill, with its level) plus a
knowledge, and the soft skills and work behaviours are listed beside it.

## What is NOT sourced

- **Software-work verbs** (`basis = DEV`): deploy, configure, debug,
  troubleshoot, refactor, architect, migrate and the like are not in the
  framework. Their level is the developer's reading of what the work demands.
- **Indonesian verbs** (`basis = ID`): each is the developer's translation of
  an English row. No official Indonesian list of "kata kerja operasional" for
  the revised taxonomy was found online on 2026-11-20; the lists that circulate
  in teaching materials differ from one another, as Stanny found for English.
  A list the researcher's institution uses should replace these rows.
- **Two rows changed from version 1 and worth a look:**
  "monitor" moved from analyze to evaluate (the framework lists monitoring
  under checking); "define" moved from apply to create, because in a job
  posting "define the requirements / the architecture" is planning, although
  verb lists place "define" on C1 (recalling a definition).

## Version 2.1 (2026-11-20, same day)

Five rows were added with basis `NOM`: program, code, script, model,
memprogram. They are verbs met only through their noun ("programming" =
program). Their base form is also a common noun ("code reviews"), so they are
never read as the verb that opens a phrase. No other row changed.

`deverbal_nouns_v1.tsv` lists the nouns made from a verb and their verb
("development" = develop, "pengujian" = menguji). It is the developer's list,
unreviewed. English morphology of this kind is regular (-ment, -ing, -ion);
Indonesian peN-…-an and per-…-an nouns likewise. The list is closed on purpose:
a noun is read through a verb only if a row says so.

## Rules the code follows

- A default verb ("perform", "melakukan", "explain", …) supplied by the method
  when the postings give no verb never sets a level.
- Interpersonal verbs (`basis = SOFT`) are in the table only so that they are
  recognised as verbs; they open soft-skill phrases, which do not form
  competencies.
- Every action verb the rule extractor can open a hard-skill phrase with must
  have a row here; a test fails otherwise.
