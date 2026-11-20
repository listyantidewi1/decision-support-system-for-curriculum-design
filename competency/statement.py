"""One assessable competency statement per valid group, checked against its evidence.

Proposal §3.3.1: "sistem merumuskan satu pernyataan kompetensi yang diawali
kata kerja operasional (Krathwohl, 2002) dan dapat dinilai", and a statement
without enough support in its evidence is rejected or re-anchored.

Two ways to write the statement, for the two kinds of pipeline:

  deterministic  (P1: NLP and rules, no LLM). The verb is the one the postings
                 themselves use most in the group; the object is taken from the
                 most frequent phrase with that verb; the technologies are the
                 ones that share sentences with the group. Every word comes
                 from the evidence, so it is grounded by construction. It is
                 also stiff: it reads like the postings, not like a curriculum.

  llm            (P2 / P3). A model is given the group's evidence package and
                 asked for a statement in a fixed JSON shape. Its answer is
                 NOT trusted: the verb must be an operational verb of the table,
                 every technology it names must be in the group's own list, and
                 the grounding score must reach the threshold. If any check
                 fails, the deterministic statement is used and the failure is
                 recorded.

The verb table (VERBS) assigns each operational verb a level of Krathwohl's
cognitive process dimension. It is a first version written by the developer and
has NOT been reviewed by an expert; the table carries a version so a reviewed
one can replace it without confusion.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .grouping import Group

VERB_TABLE_VERSION = "verbs-v2-sourced-unreviewed"
VERB_TABLE = Path(__file__).resolve().parent / "verbs" / "bloom_verbs_v2.tsv"
# A statement that lists every tool of a group is a tool list, not a
# competency. First trial (EXP-019): several accepted statements named 7 or 8.
MAX_TECHNOLOGIES_IN_STATEMENT = 3
GROUNDING_VERSION = "grounding-lexical-v1"
LEVELS = ("remember", "understand", "apply", "analyze", "evaluate", "create")


def load_verb_table(path: Path = VERB_TABLE) -> dict[str, dict]:
    """verb -> {lang, level, process, also, basis}. The file and its sources are
    in `competency/verbs/`; read SOURCES.md before changing a row."""
    out: dict[str, dict] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        cells = line.split("\t")
        if len(cells) != 6:
            raise ValueError(f"{path.name} line {n}: 6 tab-separated cells expected, got {len(cells)}")
        verb, lang, level, process, also, basis = (c.strip() for c in cells)
        also_levels = [a for a in also.split(",") if a]
        if (level not in LEVELS or lang not in ("en", "id") or verb in out
                or any(a not in LEVELS or a == level for a in also_levels)):
            raise ValueError(f"{path.name} line {n}: bad or repeated row for {verb!r}")
        out[verb] = {"lang": lang, "level": level, "process": process, "also": also_levels,
                     "basis": basis}
    return out


def verb_table_sha256(path: Path = VERB_TABLE) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


VERB_INFO: dict[str, dict] = load_verb_table()
# base form -> cognitive process category (Anderson & Krathwohl)
VERBS: dict[str, str] = {v: row["level"] for v, row in VERB_INFO.items()}
INDONESIAN_VERBS = {v for v, row in VERB_INFO.items() if row["lang"] == "id"}
_STOP = {"a", "an", "the", "and", "or", "of", "to", "for", "in", "on", "with", "using", "by",
         "at", "as", "our", "your", "their", "is", "are", "dan", "atau", "yang", "untuk",
         "dengan", "di", "ke", "dari", "pada", "dalam", "menggunakan", "serta", "new"}
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#.\-]*")


def _forms() -> dict[str, str]:
    """Every inflected form in the action lexicon -> its base verb."""
    from fivetag.lexicon import load
    lx, out = load(), {}
    for section in ("verbs", "soft-verbs"):
        for line in lx.lines("actions", section):
            for form in line:
                out[form] = line[0]
    for v, row in VERB_INFO.items():
        out.setdefault(v, v)
        if row["lang"] == "en":          # verbs the extractor's lexicon does not inflect
            stem = v[:-1] if v.endswith("e") else v
            for form in (v + ("es" if v.endswith(("s", "x", "ch", "sh")) else "s"),
                         stem + "ing", stem + "ed", v + "d" if v.endswith("e") else v + "ed"):
                out.setdefault(form, v)
    return out


_FORMS: dict[str, str] | None = None


def base_verb(word: str) -> str | None:
    global _FORMS
    if _FORMS is None:
        _FORMS = _forms()
    base = _FORMS.get(word.lower())
    return base if base in VERBS else None


@dataclass
class Statement:
    text: str
    verb: str
    object: str
    technologies: list[str]
    krathwohl: str
    language: str                       # en | id
    method: str                         # deterministic-v1 | llm
    verb_source: str                    # evidence | default
    grounding: float = 0.0
    ungrounded: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    object_source: str = "phrase"       # phrase | knowledge (the object the phrases share)
    krathwohl_source: str = "statement verb"   # postings | statement verb | none

    def to_dict(self) -> dict:
        return asdict(self)


def _words(text: str) -> list[str]:
    return [w.rstrip(".").lower() for w in _WORD.findall(text)]


def grounding(statement_text: str, group: Group, verb: str = "", verb_source: str = "") -> tuple[float, list[str]]:
    """Share of the statement's content words that occur in the group's own
    evidence (member phrases, technologies, knowledge context). A default verb
    ("Explain", "Demonstrate") is supplied by the method, not by the postings,
    and is left out of the count rather than counted as grounded."""
    vocab: set[str] = set()
    for m in group.members:
        vocab.update(_words(m["text"]))
    for t in (group.technologies + group.knowledge_context
              + list(getattr(group, "knowledge_objects", []))):
        vocab.update(_words(t["text"]))
    vocab |= {base_verb(w) or w for w in list(vocab)}
    words = [w for w in _words(statement_text) if w not in _STOP]
    if verb_source == "default" and words and words[0] == verb.lower():
        words = words[1:]
    if not words:
        return 0.0, []
    missing = [w for w in words if w not in vocab and (base_verb(w) or w) not in vocab]
    return 1.0 - len(missing) / len(words), missing


def _join(names: list[str], language: str) -> str:
    if len(names) == 1:
        return names[0]
    conj = " dan " if language == "id" else " and "
    return ", ".join(names[:-1]) + conj + names[-1]


_JOINERS = ("and", "or", "dan", "atau", "&", "/")
_ID_MARKS = {"dan", "yang", "untuk", "dengan", "dalam", "atau", "serta", "pada", "memiliki",
             "mampu", "minimal", "pengalaman", "kami", "anda", "sebagai", "lulusan", "bidang",
             "menguasai", "kemampuan", "bekerja", "di", "dari", "akan", "tim"}
_EN_MARKS = {"and", "the", "of", "with", "to", "in", "for", "experience", "our", "you", "we",
             "is", "are", "as", "or", "will", "team", "work", "on", "a", "an"}


def split_verb_object(text: str) -> tuple[list[str], str]:
    """'design, develop and test web apps' -> (['design', 'develop', 'test'], 'web apps').
    A phrase that does not open with a verb of the table, or has nothing after
    the verbs, comes back as ([], the phrase)."""
    words = text.split()
    whole = text.strip(" ,.")
    first = base_verb(words[0].strip(",.")) if words else None
    if not first or len(words) < 2:
        return [], whole
    verbs, k = [first], 1
    while k < len(words):
        w = words[k].strip(",.")
        b = base_verb(w)
        if b:
            if b not in verbs:
                verbs.append(b)
        elif w.lower() not in _JOINERS:
            break
        k += 1
    obj = " ".join(words[k:]).strip(" ,.")
    return (verbs, obj) if obj else ([], whole)


def _mostly_indonesian(texts) -> bool | None:
    """True/False by a vote of function words over the sentences; None when
    there is nothing to vote on."""
    idn = eng = 0
    for t in texts:
        for w in _words(t):
            idn += w in _ID_MARKS
            eng += w in _EN_MARKS
    return None if idn + eng == 0 else idn > eng


def deterministic_statement(group: Group, max_technologies: int = 0) -> Statement:
    """The P1 statement: built only from what the postings wrote.

    The statement is a VERB and its OBJECT and nothing else (the researcher,
    2026-11-20: "Perform programming", not "Perform programming using Python,
    PHP and JavaScript"). The tools and technology of a competency are listed
    beside it, with the soft skills and work behaviours; they are not part of
    what the competency is. `max_technologies` above 0 puts tool names back,
    for an ablation only.

    The object is the knowledge the group's phrases share
    (`group.knowledge_objects`, set by `competency.structure`): a competency is
    a verb plus a knowledge. The verb is the one most used with that object.
    Without that list (a group never passed through `structure.annotate`) the
    most frequent literal object of the most frequent verb is used."""
    verbs: Counter = Counter()
    objects: dict[str, Counter] = {}
    verb_led_evidence = 0
    wanted_level = (getattr(group, "cognitive", None) or {}).get("level")
    for m in group.members:
        # only an action or soft-skill phrase can open with a verb: the
        # knowledge term "design patterns" is not the verb "design"
        found, obj = split_verb_object(m["text"]) if m["tag"] in ("SH", "SS") else ([], "")
        if found:
            # in a part cut for one level, the phrase speaks through its verb on
            # that level ("design and maintain X" is "maintain X" on C3)
            lead = ([v for v in found if VERBS.get(v) == wanted_level] or found)[0]
            verbs[lead] += m["n_postings"]
            objects.setdefault(lead, Counter())[obj] += m["n_postings"]
            verb_led_evidence += m["n_evidence"]
    notes: list[str] = []
    object_source = "phrase"
    # The statement must speak for the group. Verb-led phrases set it only when
    # they are at least half of the group's evidence; otherwise the group is
    # mostly nouns ("programming", "code reviews") and its most frequent phrase
    # is used with a default verb chosen by that phrase's tag.
    if verbs and verb_led_evidence * 2 >= group.n_evidence:
        verb = sorted(verbs, key=lambda v: (-verbs[v], v))[0]
        obj = sorted(objects[verb], key=lambda o: (-objects[verb][o], o))[0]
        source = "evidence"
        known = [k for k in getattr(group, "knowledge_objects", []) if k.get("verbs")]
        if known:
            # of the verbs used with that object, the one on the level the
            # postings ask for: the verb of a statement signals its level
            wanted = (getattr(group, "cognitive", None) or {}).get("level")
            on_level = [v for v in known[0]["verbs"] if VERBS.get(v) == wanted]
            obj, verb, object_source = known[0]["text"], (on_level or known[0]["verbs"])[0], \
                "knowledge"
            notes.append(f"the object is shared by {known[0]['n_postings']} of the group's "
                         f"{group.n_postings} postings ({known[0]['share']:.0%})")
    else:
        top = group.members[0]
        obj = top["text"]
        id_words = {"dan", "yang", "untuk", "dengan", "struktur", "konsep", "prinsip", "kerja"}
        indonesian = any(w in id_words for w in _words(obj)) and not any(
            w in ("and", "of", "the") for w in _words(obj))
        # a one-word label ("pemrograman") says nothing about its language:
        # ask the sentences it was taken from
        vote = _mostly_indonesian(s.get("sentence_text", "") for s in group.samples)
        if vote is not None and not indonesian and not any(
                w in ("and", "of", "the") for w in _words(obj)):
            indonesian = vote
        default = {"SH": ("perform", "melakukan"), "KN": ("explain", "menjelaskan"),
                   "SS": ("demonstrate", "menunjukkan")}[top["tag"]]
        verb = default[1] if indonesian else default[0]
        source = "default"
        notes.append("the group is mostly noun phrases: the verb is the method's default "
                     f"for a {top['tag']} phrase")
    language = "id" if verb in INDONESIAN_VERBS else "en"
    techs = [t["text"] for t in group.technologies[:max_technologies]
             if t["text"].lower() not in obj.lower()]
    text = f"{verb.capitalize()} {obj}"
    # "using X" belongs to an action. A knowledge statement ("Explain
    # algorithms") keeps its technologies in the evidence package only.
    if VERBS[verb] == "understand":
        techs = []
    if techs and group.stream == "technical":
        text += (" menggunakan " if language == "id" else " using ") + _join(techs, language)
    # The level is what the postings' verbs ask for. A default verb is the
    # method's wording, so its level is never reported as the level demanded.
    read = (getattr(group, "cognitive", None) or {}).get("level")
    if read:
        level, level_source = read, "postings"
        if VERBS[verb] != read:
            notes.append(f"the statement's verb '{verb}' is on the level '{VERBS[verb]}'; the "
                         f"postings' verbs ask for '{read}'")
    elif source == "default":
        level, level_source = "not stated", "none"
        notes.append("no cognitive level: the postings of this group use no verb")
    else:
        level, level_source = VERBS[verb], "statement verb"
    st = Statement(text, verb, obj, techs, level, language, "deterministic-v1", source,
                   notes=notes, object_source=object_source, krathwohl_source=level_source)
    st.grounding, st.ungrounded = grounding(text, group, verb, source)
    return st


# ------------------------------------------------------------------- LLM path
SYSTEM = (
    "You write one competency statement for a vocational software-engineering curriculum "
    "from evidence extracted from job postings. Rules: (1) one sentence that starts with ONE "
    "operational verb from the allowed list; (2) it must be assessable: an action on an "
    "object, optionally with a context; (3) use ONLY what the evidence shows: do not add "
    "technologies, tools or topics that are not listed; (4) write in the language named in "
    "the request; (5) name at most three technologies, the most frequent ones, or none. "
    "Answer with JSON only: "
    '{"statement": "...", "verb": "...", "object": "...", "technologies": ["..."]}')


def build_llm_messages(group: Group, language: str) -> list[dict]:
    allowed = sorted(v for v in VERBS if (v in INDONESIAN_VERBS) == (language == "id"))
    payload = {
        "language": "Indonesian" if language == "id" else "English",
        "allowed_verbs": allowed,
        "phrases_from_postings": [{"text": m["text"], "postings": m["n_postings"]}
                                  for m in group.members[:25]],
        "technologies_named_with_them": [t["text"] for t in group.technologies],
        "related_knowledge": [k["text"] for k in group.knowledge_context],
    }
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def check_llm_statement(raw: str | None, group: Group, language: str,
                        min_grounding: float) -> tuple[Statement | None, list[str]]:
    """(accepted statement or None, reasons for rejection)."""
    reasons: list[str] = []
    try:
        obj = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip()))
    except (ValueError, TypeError):
        return None, ["INVALID: not JSON"]
    if not isinstance(obj, dict) or not isinstance(obj.get("statement"), str) \
            or not obj["statement"].strip():
        return None, ["INVALID: no statement"]
    text = " ".join(obj["statement"].split())
    first = _words(text)[:1]
    verb = base_verb(first[0]) if first else None
    if verb is None or first[0] != verb:
        reasons.append("INVALID: does not start with an allowed operational verb")
    elif (verb in INDONESIAN_VERBS) != (language == "id"):
        reasons.append("INVALID: verb is not in the requested language")
    allowed_tech = {t["text"].lower() for t in group.technologies}
    techs = [t for t in obj.get("technologies") or [] if isinstance(t, str)]
    foreign = [t for t in techs if t.lower() not in allowed_tech]
    if foreign:
        reasons.append(f"UNSUPPORTED: technologies not in the evidence: {foreign}")
    named = [t["text"] for t in group.technologies
             if re.search(r"(?<![A-Za-z0-9])" + re.escape(t["text"]) + r"(?![A-Za-z0-9])", text,
                          re.I)]
    if max(len(techs), len(named)) > MAX_TECHNOLOGIES_IN_STATEMENT:
        reasons.append(f"INVALID: more than {MAX_TECHNOLOGIES_IN_STATEMENT} technologies named")
    score, missing = grounding(text, group)
    if score < min_grounding:
        reasons.append(f"UNSUPPORTED: grounding {score:.2f} < {min_grounding:.2f}; "
                       f"not in the evidence: {missing[:8]}")
    if reasons:
        return None, reasons
    return Statement(text, verb, str(obj.get("object") or ""), techs, VERBS[verb], language,
                     "llm", "evidence", score, missing), []


def unchecked_llm_statement(group: Group, call) -> Statement:
    """ABLATION ONLY ("P3 without the deterministic verifier"): the model's
    statement is used as it comes. Its grounding is still MEASURED, so the
    ablation can report how many unsupported statements the checks would have
    stopped, but nothing is rejected."""
    fallback = deterministic_statement(group)
    raw = call(build_llm_messages(group, fallback.language))
    try:
        obj = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip()))
        text = " ".join(str(obj["statement"]).split())
    except (ValueError, TypeError, KeyError):
        fallback.notes.append("LLM answer unusable even without checks: rule statement used")
        return fallback
    first = _words(text)[:1]
    verb = (base_verb(first[0]) if first else None) or fallback.verb
    st = Statement(text, verb, str(obj.get("object") or ""),
                   [t for t in obj.get("technologies") or [] if isinstance(t, str)],
                   VERBS.get(verb, "apply"), fallback.language, "llm-unchecked", "evidence")
    st.grounding, st.ungrounded = grounding(text, group)
    _, reasons = check_llm_statement(raw, group, fallback.language, 0.8)
    if reasons:
        st.notes.append("would have been rejected: " + "; ".join(reasons))
    return st


def llm_statement(group: Group, call, min_grounding: float = 0.8) -> Statement:
    """`call(messages) -> raw text`. Falls back to the deterministic statement,
    with the reasons recorded, when the model's answer does not pass."""
    fallback = deterministic_statement(group)
    raw = call(build_llm_messages(group, fallback.language))
    st, reasons = check_llm_statement(raw, group, fallback.language, min_grounding)
    if st is not None:
        return st
    fallback.notes.append("LLM statement rejected: " + "; ".join(reasons))
    return fallback
