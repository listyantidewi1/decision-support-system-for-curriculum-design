"""The shared prompt pack — every model in the ensemble receives exactly this.

Proposal §3.4.3: "Setiap LLM menerima pedoman anotasi, definisi lima kategori,
contoh positif dan negatif, aturan batas span, serta format keluaran terstruktur
yang sama."

The text below is a condensed English rendering of the validated annotation
guideline (Paket B v3.0, code AG-RPL-LIMA-v3.0): its general principles (§2),
the five tag definitions with their exclusions (§4), the decision order (§5),
the boundary rules (§6) and the special cases (§7). The section numbers are kept
in the prompt so a disputed model decision can be traced to the rule it was
given.

**This is prompt version v0 — guideline only.** Every example in it is one of
the guideline's own illustrations. It contains NO sentence from the corpus and
NO gold annotation, because none exists yet. The proposal's few-shot examples
must come from the gold train/dev split only; adding them makes a new prompt
version with a new hash, and the held-out test stays closed until that version
is frozen.

Models are asked for the span TEXT (plus an occurrence index when the text
repeats), never for character offsets: language models miscount characters, and
`labeling.verify` derives the offsets deterministically from the source instead.
"""
from __future__ import annotations

import hashlib
import json

from gold_standard import GUIDELINE_CODE

PROMPT_VERSION = "v0-guideline-only"

SYSTEM_PROMPT = """\
You are annotating one sentence from a software-engineering job posting. Mark \
every span that belongs to one of five tags, following the annotation guideline \
below exactly. The sentence may be in English, Indonesian, or a mix.

## General principles (guideline §2)
- Label only what is written explicitly in the sentence. Do not add things a \
person in that job "usually" needs. For "Develop backend services using Python." \
mark "Develop backend services" and "Python"; do not add Docker or Git.
- The source text is the evidence. Copy each span character-for-character from \
the sentence. Never translate, correct, reorder or paraphrase.
- Not every requirement belongs to the five tags. Education, degrees, years of \
experience, age, location, contract type, salary, citizenship, administrative \
certificates and language requirements (e.g. "fluent in English") get NO tag.
- Job titles (e.g. "Backend Developer", "QA Engineer") get NO tag.
- If nothing in the sentence belongs to the five tags, return an empty list.

## The five tags (guideline §4)
SH = SKILL_HARD — a technical work action that can be observed or assessed. \
Helper question: can someone be asked to DO this in their job? Examples: \
"Develop REST APIs", "write SQL queries", "automated testing", "Design database \
schemas", "build CI/CD pipelines". Not for a bare technology name ("Python"), \
and not for a concept without an action word ("database normalization").

SS = SKILL_SOFT — an interpersonal or self-management ability that can be \
trained and shown in a task: communicating, collaborating, leading, \
coordinating, presenting, time management, problem solving as a general \
ability. Example: "Communicate technical issues to non-technical stakeholders". \
Not for personal traits (those are WB).

KN = KNOWLEDGE — a concept, principle, theory, architecture, method or domain \
that must be UNDERSTOOD. Examples: "relational database concepts", \
"object-oriented programming principles", "software architecture principles". \
Not for a technology name: in "Knowledge of Python", "Python" is TT.

TT = TOOL_TECHNOLOGY — the name of a specific programming language, tool, \
framework, library, platform, database system, cloud service or other specific \
technology. Examples: "Python", "React", "Docker", "PostgreSQL", "AWS". Not for a \
generic technology group with no product name ("cloud platforms", "frontend \
frameworks", "NoSQL databases").

WB = WORK_BEHAVIOR — an attitude, disposition, habit or trait expected of the \
person. Examples: "detail-oriented", "proactive", "self-motivated", "willing to \
learn". Not for abilities exercised with other people ("collaborate", \
"communicate" are SS). Rule of thumb: SS is a verb or trainable ability, WB is \
an adjective or trait.

## When the tag is unclear, ask in this order and stop at the first yes (§5)
1. Is it the name of a specific technology? -> TT
2. Does it describe a technical action the person must be able to do? -> SH
3. Does it describe a concept or principle to be understood? -> KN
4. Does it describe communicating, collaborating, leading, coordinating or \
self-management? -> SS
5. Does it describe a trait or a way of behaving? -> WB

Commonly confused:
- "Understanding object-oriented programming principles." -> \
"object-oriented programming principles" is KN (understanding is emphasised).
- "Apply object-oriented programming principles when designing applications." \
-> the whole phrase is SH (an action is emphasised).
- "Collaborate with product and QA teams." -> SS. "Proactive and \
detail-oriented." -> "Proactive" WB and "detail-oriented" WB, separately.
- "Present technical designs to stakeholders." -> SS. "Facilitate Scrum \
ceremonies." -> SH.

## Span boundaries (guideline §6) — scoring uses strict matching
6.1 Take the shortest span that keeps the meaning. "Must have experience in \
automated testing." -> "automated testing", not the whole sentence.
6.2 For a work action keep the verb. "Design database schemas." -> "Design \
database schemas", not "database schemas".
6.3 Leave out generic lead-in words: must, should, required to, able to, \
experience in, proficiency in, familiar with, strong knowledge of, memiliki \
kemampuan, berpengalaman dalam, menguasai.
6.4 Keep words that change the meaning: "automated testing" must not shrink to \
"testing"; "relational database concepts" must not shrink to "database".
6.5 List items joined by commas, "and"/"dan", "or"/"atau" are separate spans. \
"Python, Java, and Go." -> three TT spans.
6.6 A full name next to its acronym is ONE span: "Amazon Web Services (AWS)".
6.7 Do not include punctuation outside the entity name.
6.8 Separate the action from the technology used. "Build web applications using \
React." -> "Build web applications" SH and "React" TT.
6.9 Overlapping spans are allowed ONLY when their function differs. "Configure \
Kubernetes clusters." -> the whole phrase SH, and "Kubernetes" inside it TT. \
Never give two overlapping spans the same tag.
6.10 A version number attached to a technology name is part of the span: "Java \
17", "Python 3.11".

## Special cases (guideline §7)
7.1 A sentence that only lists technology names gets only TT spans. Do not \
invent an SH such as "programming".
7.2 Methodology names (Agile, Scrum, Kanban) are not automatically TT. \
"Understanding Agile principles" -> "Agile principles" KN. "Facilitate Scrum \
ceremonies" -> whole phrase SH.
7.4 "write automated tests using Selenium" -> "write automated tests" SH and \
"Selenium" TT.
7.9 A generic technology group in a context of understanding is KN \
("familiarity with NoSQL databases" -> "NoSQL databases" KN); in a context of \
action it is part of the SH span ("deploy to cloud platforms" -> whole phrase \
SH).
7.11 A negated mention is still tagged: "No prior experience with Kubernetes is \
required." -> "Kubernetes" TT.
7.12 An optional requirement ("preferred", "nice to have") is tagged the same.
7.13 Mixed Indonesian/English: "Mampu melakukan debugging dan unit testing \
menggunakan Jest." -> "melakukan debugging" SH, "unit testing" SH, "Jest" TT.
7.15 "problem solving", "analytical thinking", "critical thinking" stated as \
abilities are SS. "curious", "resilient", "ownership mindset" are WB.

## Output format
Return ONE JSON object and nothing else — no explanation, no markdown fence:

{"spans": [{"text": "<exact span copied from the sentence>", "label": "<SH|SS|KN|TT|WB>"}]}

- "text" must be an exact, case-sensitive substring of the TARGET sentence.
- If the same text occurs more than once in the target sentence, add \
"occurrence": <1-based index of the occurrence you mean>.
- The context sentences are there only to help you understand the target. \
Never take a span from them.
- If the target sentence contains no span of the five tags, return {"spans": []}.
"""


PROMPT_V1 = "v1-sh-action-and-noun-lists"
DEFAULT_PROMPT_VERSION = PROMPT_VERSION

# v1 = v0 plus one section. It was written after the first smoke test
# (2026-11-18), where the three models tagged bare nouns such as "wireframes" as
# SKILL_HARD and split verb-led skills inconsistently. Every sentence in it is
# drawn from the guideline or newly made up; none is a corpus sentence and none
# is an item of the conformance set (labeling/conformance.py), so that set still
# measures the rule and not recall of an example.
V1_ADDENDUM = (
    "## Clarifications on SKILL_HARD (prompt v1)\n"
    "- SH names an ACTION. When the sentence has the verb that carries the action, "
    "the verb is part of the span (rule 6.2): \"Design database schemas\", never "
    "\"database schemas\"; \"mengembangkan REST API\", never \"REST API\".\n"
    "- Before returning an SH span, check whether the verb that governs it is in the "
    "sentence. If it is and you left it out, extend the span to include it. Generic "
    "lead-in words are still left out (rule 6.3): \"Able to write SQL queries\" -> "
    "\"write SQL queries\".\n"
    "- An activity noun that itself names a technical action is SH even without a "
    "verb: \"automated testing\", \"unit testing\" (rules 7.4, 7.13).\n"
    "- A noun that names a thing, an artefact, a stage or a topic is NOT SH: "
    "\"mockups\", \"style guides\", \"production code\".\n"
    "- When ONE verb governs a LIST of such nouns, tag each noun item separately as "
    "KN and do not tag the verb (researcher ruling, 2026-11-18): \"You will work "
    "across information architecture, user journeys, and style guides.\" -> "
    "\"information architecture\" KN, \"user journeys\" KN, \"style guides\" KN. A "
    "specific technology name in such a list is still TT.\n\n"
)

PROMPT_V2 = "v2-boundary-conventions"

# v2 = v1 plus one section on span boundaries. On 42 real SkillSpan sentences
# annotated under the guideline (2026-11-19), tag errors were rare (0-2 per
# model out of 74 spans) and boundary errors dominated: of 50 boundary
# differences under v0, 38 were model spans that were TOO LONG ("good
# communication skills" for "communication", "AWS services" for "AWS"), 6 too
# short, 6 split. The rules below are conventions C1-C6 of
# labeling/skillspan_guideline.py, which the researcher accepted on 2026-11-18.
# They were written BEFORE the second batch of SkillSpan sentences was drawn, so
# that batch is a test of the rules on text this prompt has never seen. No
# example here is a sentence from the conformance set or from either batch.
V2_ADDENDUM = (
    "## Span boundary conventions (prompt v2)\n"
    "Most boundary mistakes are spans that are TOO LONG. Apply these conventions, "
    "which the researcher confirmed:\n"
    "- Leave out a lead-in noun or phrase, not only a lead-in verb (rule 6.3): "
    "\"Experience\", \"Proven track record of\", \"Ability to\", \"Interest in\", "
    "\"Responsible for\". \"Experience building data pipelines\" -> \"building data "
    "pipelines\".\n"
    "- Leave out an evaluative word in front: \"good\", \"strong\", \"excellent\", "
    "\"solid\", \"proven\".\n"
    "- Leave out a trailing generic \"skills\", \"abilities\" or \"capabilities\" when "
    "what remains is a complete noun phrase: \"excellent negotiation skills\" -> "
    "\"negotiation\". Keep it when the rest cannot stand alone: \"people skills\".\n"
    "- Leave out trailing context that says for whom, where or why (guideline "
    "example 4.1): \"Build mobile apps for our retail customers\" -> \"Build mobile "
    "apps\".\n"
    "- Leave out a generic word after a technology name: \"the Flutter framework\" "
    "-> \"Flutter\", \"Salesforce platform\" -> \"Salesforce\".\n"
    "- Coordinated verbs that share one object are ONE span, because a bare verb "
    "carries no meaning on its own: \"Design and implement data pipelines\" is one "
    "SH span, not \"Design\" plus \"implement data pipelines\".\n"
    "- Do not cut one ability into pieces: \"partner closely with the design "
    "group\" is one SS span, not \"partner closely\" plus \"design group\".\n"
    "- Working with or in a team, collaborating, coordinating people, coaching, "
    "mentoring and giving feedback are SS (guideline section 5).\n"
    "- A generic business outcome is not one of the five tags and gets no span: "
    "\"satisfy client expectations\", \"add value to the business\".\n"
    "- Where a list has lost its commas, each item is still a separate span (rule "
    "6.5): \"Java Kotlin and Swift\" -> three TT spans.\n\n"
)

PROMPT_VERSIONS = (PROMPT_VERSION, PROMPT_V1, PROMPT_V2)


def system_prompt(version: str = DEFAULT_PROMPT_VERSION) -> str:
    """The instructions for one prompt version."""
    if version == PROMPT_VERSION:
        return SYSTEM_PROMPT
    if version in (PROMPT_V1, PROMPT_V2):
        marker = "## Output format"
        if SYSTEM_PROMPT.count(marker) != 1:
            raise RuntimeError("the v0 prompt must have exactly one Output format section")
        extra = V1_ADDENDUM if version == PROMPT_V1 else V1_ADDENDUM + V2_ADDENDUM
        return SYSTEM_PROMPT.replace(marker, extra + marker)
    raise ValueError(f"unknown prompt version {version!r}; known: {PROMPT_VERSIONS}")


def build_messages(sentence: str, previous: str = "", following: str = "",
                   version: str = DEFAULT_PROMPT_VERSION) -> list[dict]:
    """Chat messages for one sentence. The annotators see the same context."""
    user = {
        "task": "Annotate the TARGET sentence only.",
        "previous_sentence_context": previous or "",
        "target_sentence": sentence,
        "next_sentence_context": following or "",
    }
    return [{"role": "system", "content": system_prompt(version)},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=1)}]


def prompt_sha256(version: str = DEFAULT_PROMPT_VERSION) -> str:
    """Identity of the prompt pack: any change to the instructions changes it."""
    blob = json.dumps({"version": version, "guideline": GUIDELINE_CODE,
                       "system": system_prompt(version),
                       "user_template": build_messages("{sentence}", "{previous}",
                                                       "{following}",
                                                       version)[1]["content"]},
                      ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def prompt_manifest(version: str = DEFAULT_PROMPT_VERSION) -> dict:
    return {"prompt_version": version, "guideline_code": GUIDELINE_CODE,
            "prompt_sha256": prompt_sha256(version), "n_gold_examples": 0,
            "system_prompt_chars": len(system_prompt(version))}
