"""The inside of a competency: an operational verb and the knowledge it acts on.

The researcher's definition (2026-11-20): a competency is a VERB plus a
KNOWLEDGE. "Develop machine learning algorithms": "develop" is the verb, placed
on the taxonomy by the verb table; "machine learning algorithms" is the
knowledge; the tools and technology that support it (Python, ...) are listed
beside the competency, not inside its statement.

A hard-skill phrase already has this shape, so this module takes it apart:

  verbs               the operational verbs the postings use for the group's
                      work, each with its taxonomy level and its postings. A
                      coordinated group ("design, develop and test web apps")
                      gives each of its verbs.
  knowledge objects   what those verbs act on: the object the group's phrases
                      SHARE. Postings rarely write the same object twice
                      ("scalable REST APIs for internal services", "secure
                      REST APIs", "REST APIs"), so the exact object is not
                      counted; its head is. For every phrase the noun phrase
                      after the verb is cut at the first preposition, and the
                      head word is taken (the last word in English, the first
                      in Indonesian). The most shared head is then lengthened
                      one word at a time for as long as more than half of its
                      postings keep the longer form: "APIs" -> "REST APIs".
                      The step is repeated on the remaining phrases, so a group
                      lists several knowledge objects, each with its postings
                      and the verbs used with it.

Why not the KNOWLEDGE tag itself. The extractor tags a term KNOWLEDGE when it
stands alone ("knowledge of design patterns"); when a verb governs it the whole
phrase is a hard skill. Measured on the whole pool (EXP-028): under 1% of
hard-skill phrases contain a term that is tagged KNOWLEDGE anywhere in the
corpus. So the object of the verb is read from the phrase, and whether the same
term is also tagged KNOWLEDGE elsewhere is reported as a flag
(`tagged_knowledge`), not required.

What this is not: there is no part-of-speech tagger here. "Head" is a rule
about word order and a list of prepositions; an adjective at the end of a
truncated span ("Write clean") is taken for a head. Such objects are extraction
errors passing through, and their support is reported so they can be seen.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

import dataclasses

from .evidence import Evidence, canonical_key
from .grouping import Describer, Group, GroupingConfig
from .statement import INDONESIAN_VERBS, VERBS, split_verb_object

STRUCTURE_VERSION = "verb-knowledge-v1"
_BREAK_EN = {"for", "with", "using", "in", "on", "to", "of", "across", "that", "which", "by",
             "from", "through", "within", "into", "at", "as", "via", "including", "such",
             "while", "where", "when", "based", "per", "like", "so", "under", "over", "before",
             "after", "during", "against", "towards", "toward", "between", "among", "without"}
_BREAK_ID = {"untuk", "dengan", "yang", "di", "pada", "dalam", "serta", "menggunakan", "sesuai",
             "agar", "dan", "atau", "ke", "dari", "secara", "sebagai", "guna", "terhadap",
             "oleh", "seperti", "termasuk", "melalui", "berdasarkan", "antara", "berbasis"}
_LEAD = {"the", "a", "an", "our", "your", "their", "its", "all", "any", "some", "various",
         "multiple", "both", "these", "those", "this", "that", "other", "new", "and", "or", "&"}
_GLUE = {"and", "or", "&", "/"}
# Krathwohl's cognitive processes in Bloom's order, with the usual codes
LEVELS = ("remember", "understand", "apply", "analyze", "evaluate", "create")
BLOOM_CODE = {level: f"C{i}" for i, level in enumerate(LEVELS, 1)}


@dataclass(frozen=True)
class StructureConfig:
    min_vocabulary_postings: int = 3      # a KNOWLEDGE term must be seen this often to count
    max_object_words: int = 4
    keep_longer_form_at: float = 0.5      # lengthen the head while MORE than this share agrees
    max_items: int = 8                    # verbs / knowledge objects listed per competency
    min_item_postings: int = 2
    # "highest level asked for": the highest level that at least this share of
    # the verb-using postings ask for (one posting must not set it)
    highest_level_floor: float = 0.20
    # a cognitive level becomes a competency of its own when at least this many
    # postings of the group use a verb on it
    min_variant_postings: int = 2
    version: str = STRUCTURE_VERSION


def object_chunk(obj: str, indonesian: bool) -> list[str]:
    """The noun phrase right after the verb: words up to the first preposition,
    without leading determiners and without punctuation."""
    stop = _BREAK_ID if indonesian else _BREAK_EN
    out: list[str] = []
    for raw in obj.split():
        w = raw.strip(" ,.;:()[]\"'")
        if not w:
            continue
        if w.lower() in stop:
            break
        out.append(w)
    while out and out[0].lower() in _LEAD:
        out.pop(0)
    while out and out[-1].lower() in _GLUE:
        out.pop()
    return out


def _gram(chunk: list[str], n: int, indonesian: bool) -> tuple[str, ...] | None:
    """The n words at the head side of the chunk, lower-cased."""
    if len(chunk) < n:
        return None
    part = chunk[:n] if indonesian else chunk[-n:]
    edge = part[-1] if indonesian else part[0]
    if edge.lower() in _GLUE:
        return None
    return tuple(w.lower() for w in part)


def shared_objects(rows: Sequence[tuple[str, list[str], list[str], bool]], n_group_postings: int,
                   cfg: StructureConfig, vocabulary: dict[str, dict] | None = None) -> list[dict]:
    """`rows` are (posting_id, verbs, chunk, indonesian) of the verb-led phrases
    of one group. Returns its knowledge objects, the most shared first."""
    vocabulary = vocabulary or {}
    remaining = [r for r in rows if r[2]]
    out: list[dict] = []
    while remaining and len(out) < cfg.max_items:
        heads: dict[tuple, set] = defaultdict(set)
        for posting, _, chunk, idn in remaining:
            heads[(idn,) + _gram(chunk, 1, idn)].add(posting)
        head, head_postings = min(heads.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        if len(head_postings) < cfg.min_item_postings:
            break
        current = [r for r in remaining if (r[3],) + _gram(r[2], 1, r[3]) == head]
        n = 1
        while n < cfg.max_object_words:
            longer: dict[tuple, set] = defaultdict(set)
            for posting, _, chunk, idn in current:
                g = _gram(chunk, n + 1, idn)
                if g:
                    longer[g].add(posting)
            if not longer:
                break
            best, best_postings = min(longer.items(), key=lambda kv: (-len(kv[1]), kv[0]))
            if (len(best_postings) <= cfg.keep_longer_form_at * len({r[0] for r in current})
                    or len(best_postings) < cfg.min_item_postings):
                break
            current = [r for r in current if _gram(r[2], n + 1, r[3]) == best]
            n += 1
        forms: Counter = Counter()
        verb_postings: dict[str, set] = defaultdict(set)
        postings = set()
        for posting, verbs, chunk, idn in current:
            forms[" ".join(chunk[:n] if idn else chunk[-n:])] += 1
            postings.add(posting)
            for v in verbs:
                verb_postings[v].add(posting)
        text = max(sorted(forms), key=lambda s: forms[s])
        out.append({
            "text": text, "n_postings": len(postings),
            "share": round(len(postings) / max(1, n_group_postings), 4),
            "verbs": [v for v, _ in sorted(verb_postings.items(),
                                           key=lambda kv: (-len(kv[1]), kv[0]))[:3]],
            "tagged_knowledge": canonical_key(text, "KN") in vocabulary})
        remaining = [r for r in remaining if (r[3],) + _gram(r[2], 1, r[3]) != head]
    return out


def cognitive_profile(verb_postings: dict[str, set], n_group_postings: int,
                      cfg: StructureConfig) -> dict:
    """The cognitive level the POSTINGS ask for, read from their own verbs.

    The researcher's correction (2026-11-20): a level must not come from a verb
    the method supplied. A group of the noun "pemrograman" was shown as "apply"
    because the statement was given the default verb "perform", while its
    postings wrote "membangun" and "menulis" (create).

    Each verb is placed on the taxonomy by the verb table. A posting counts once
    for a level however many verbs of that level it uses, and for every level
    it uses. `level` is the level most postings ask for (a tie goes to the
    higher one). `highest` is the highest level asked for by at least
    `highest_level_floor` of the postings that use a verb. When fewer than
    `min_item_postings` postings use a verb, no level is given."""
    by_level: dict[str, set] = defaultdict(set)
    for verb, ps in verb_postings.items():
        if verb in VERBS:
            by_level[VERBS[verb]] |= ps
    with_verb = set().union(*by_level.values()) if by_level else set()
    n = len(with_verb)
    out = {"level": None, "code": None, "share": None, "highest": None,
           "n_postings_with_a_verb": n,
           "share_of_group_postings": round(n / max(1, n_group_postings), 4), "levels": []}
    if n < cfg.min_item_postings:
        out["basis"] = "too few of the group's postings use a verb: no level can be read"
        return out
    rows = [{"level": lv, "code": BLOOM_CODE[lv], "n_postings": len(by_level[lv]),
             "share": round(len(by_level[lv]) / n, 4)}
            for lv in reversed(LEVELS) if by_level.get(lv)]
    top = max(rows, key=lambda r: (r["n_postings"], LEVELS.index(r["level"])))
    high = next((r for r in rows if r["share"] >= cfg.highest_level_floor), top)
    out.update(level=top["level"], code=top["code"], share=top["share"],
               highest={"level": high["level"], "code": high["code"], "share": high["share"]},
               levels=rows, basis="verbs written in the postings")
    return out


def split_by_level(groups: Sequence[Group], evidence: Sequence[Evidence], n_postings: int,
                   grouping_cfg: GroupingConfig | None = None,
                   posting_group: dict[str, str] | None = None,
                   cfg: StructureConfig | None = None) -> tuple[list[Group], dict]:
    """One competency per cognitive level of each group.

    The researcher (2026-11-20): "apply web programming" and "develop a website"
    are two different competencies on two different cognitive levels; where a
    group holds several verbs, make related competencies, one per level, and
    say how many postings ask for each. Nobody should be misled by one level
    standing for a whole group, and institutions of different levels use the
    result.

    A group of phrases that mean the same work (a FAMILY) is cut by the level
    of the verbs its phrases open with:

      * a phrase belongs to every level its verbs sit on ("design and maintain
        databases" is in the create part and in the apply part);
      * each part is described exactly like a group (its own phrases, postings,
        tools, soft skills, work behaviours) and gets its own verbs and its
        own shared knowledge object;
      * phrases that open with no verb ("programming", "unit testing") form a
        part without a level: the postings name the work and say nothing of
        the level;
      * a level that fewer than `min_variant_postings` postings ask for makes
        no competency; its units are counted in the report.

    The parts keep the family's validity status (it was judged on the meaning
    of the whole group). Because a phrase can be in two parts, the parts'
    postings add up to more than the family's."""
    cfg, gcfg = cfg or StructureConfig(), grouping_cfg or GroupingConfig()
    describer = Describer(evidence, n_postings, gcfg, posting_group)
    vocabulary = knowledge_vocabulary(evidence, cfg)
    by_id = {e.evidence_id: e for e in evidence}
    parsed: dict[str, tuple[list[str], list[str], bool]] = {}
    out: list[Group] = []
    too_small = 0
    for g in groups:
        by_level: dict[str | None, list[Evidence]] = defaultdict(list)
        for eid in g.evidence_ids:
            e = by_id[eid]
            if e.text not in parsed:
                verbs, obj = split_verb_object(e.text) if e.tag == "SH" else ([], "")
                idn = bool(verbs) and verbs[0] in INDONESIAN_VERBS
                parsed[e.text] = (verbs, object_chunk(obj, idn) if verbs else [], idn)
            verbs = parsed[e.text][0]
            if not verbs:
                by_level[None].append(e)
            for level in {VERBS[v] for v in verbs}:
                by_level[level].append(e)
        order = [lv for lv in reversed(LEVELS) if lv in by_level] + (
            [None] if None in by_level else [])
        family_postings = len({by_id[i].posting_id for i in g.evidence_ids})
        with_verb = len({e.posting_id for lv in order if lv for e in by_level[lv]})
        for level in order:
            part = by_level[level]
            postings = {e.posting_id for e in part}
            if level is not None and len(postings) < cfg.min_variant_postings:
                too_small += len(part)
                continue
            verb_postings: dict[str, set] = defaultdict(set)
            rows = []
            for e in part:
                verbs, chunk, idn = parsed[e.text]
                on_level = [v for v in verbs if VERBS[v] == level]
                for v in on_level:
                    verb_postings[v].add(e.posting_id)
                if on_level and chunk:
                    rows.append((e.posting_id, on_level, chunk, idn))
            n = len(postings)
            code = BLOOM_CODE.get(level)
            v = dataclasses.replace(
                g, group_id=f"{g.group_id}@{code or 'none'}", family_id=g.group_id, level=level,
                **describer.describe(part))
            v.verbs = sorted(
                ({"verb": w, "krathwohl": level, "n_postings": len(ps), "share": round(len(ps) / n, 4)}
                 for w, ps in verb_postings.items()),
                key=lambda r: (-r["n_postings"], r["verb"]))[:cfg.max_items]
            v.knowledge_objects = shared_objects(rows, n, cfg, vocabulary)
            v.cognitive = {
                "level": level, "code": code, "n_postings": n,
                "share_of_family_postings": round(n / max(1, family_postings), 4),
                "n_family_postings": family_postings,
                "n_family_postings_with_a_verb": with_verb,
                "basis": ("verbs written in the postings" if level else
                          "the postings name this work without a verb: no level can be read")}
            out.append(v)
    out.sort(key=lambda v: (-v.n_postings, v.group_id))
    report = {"n_families": len(groups), "n_competencies_by_level": len(out),
              "by_level": dict(Counter(v.cognitive["code"] or "none" for v in out)),
              "n_families_with_more_than_one_level": sum(
                  1 for n in Counter(v.family_id for v in out if v.level).values() if n > 1),
              "n_units_on_a_level_too_small_to_stand": too_small,
              "min_variant_postings": cfg.min_variant_postings}
    return out, report


def knowledge_vocabulary(evidence: Sequence[Evidence], cfg: StructureConfig) -> dict[str, dict]:
    """KNOWLEDGE key -> {"text": most frequent form, "n_postings": corpus-wide}."""
    postings: dict[str, set] = defaultdict(set)
    forms: dict[str, Counter] = defaultdict(Counter)
    for e in evidence:
        if e.tag == "KN":
            postings[e.key].add(e.posting_id)
            forms[e.key][e.text] += 1
    return {k: {"text": max(sorted(forms[k]), key=lambda s: forms[k][s]), "n_postings": len(p)}
            for k, p in postings.items() if len(p) >= cfg.min_vocabulary_postings}


def knowledge_in(obj: str, vocabulary: dict[str, dict], max_words: int = 6) -> str | None:
    """The key of the longest KNOWLEDGE-tagged term that is the object or sits
    inside it as whole words. Used only to measure how often that happens."""
    words = [w.strip(" ,.;:()") for w in obj.split()]
    words = [w for w in words if w]
    for n in range(min(max_words, len(words)), 0, -1):
        best, best_n = None, 0
        for i in range(len(words) - n + 1):
            key = canonical_key(" ".join(words[i:i + n]), "KN")
            row = vocabulary.get(key)
            if row and row["n_postings"] > best_n:
                best, best_n = key, row["n_postings"]
        if best:
            return best
    return None


def annotate(groups: Sequence[Group], evidence: Sequence[Evidence],
             cfg: StructureConfig | None = None) -> dict:
    """Fills `verbs` and `knowledge_objects` of every group; returns a report."""
    cfg = cfg or StructureConfig()
    vocabulary = knowledge_vocabulary(evidence, cfg)
    by_id = {e.evidence_id: e for e in evidence}
    parsed: dict[str, tuple[list[str], list[str], bool, bool]] = {}
    n_units = n_with_verb = n_with_tagged = 0
    for g in groups:
        verb_postings: dict[str, set] = defaultdict(set)
        rows, postings = [], set()
        for eid in g.evidence_ids:
            e = by_id[eid]
            postings.add(e.posting_id)
            if e.tag != "SH":
                continue
            if e.text not in parsed:
                verbs, obj = split_verb_object(e.text)
                idn = bool(verbs) and verbs[0] in INDONESIAN_VERBS
                parsed[e.text] = (verbs, object_chunk(obj, idn) if verbs else [], idn,
                                  knowledge_in(obj, vocabulary) is not None)
            verbs, chunk, idn, tagged = parsed[e.text]
            n_units += 1
            n_with_verb += bool(verbs)
            n_with_tagged += tagged
            for v in verbs:
                verb_postings[v].add(e.posting_id)
            if verbs and chunk:
                rows.append((e.posting_id, verbs, chunk, idn))
        n = max(1, len(postings))
        g.verbs = sorted(
            ({"verb": v, "krathwohl": VERBS.get(v, "not in the verb table"),
              "n_postings": len(p), "share": round(len(p) / n, 4)}
             for v, p in verb_postings.items() if len(p) >= cfg.min_item_postings),
            key=lambda r: (-r["n_postings"], r["verb"]))[:cfg.max_items]
        g.knowledge_objects = shared_objects(rows, n, cfg, vocabulary)
        g.cognitive = cognitive_profile(verb_postings, len(postings), cfg)
    with_obj = [g for g in groups if g.knowledge_objects]
    return {"structure_version": cfg.version, "n_groups": len(groups),
            "n_hard_skill_units_in_groups": n_units,
            "n_units_opening_with_a_verb": n_with_verb,
            "n_units_containing_a_term_tagged_knowledge": n_with_tagged,
            "n_knowledge_terms_in_vocabulary": len(vocabulary),
            "min_vocabulary_postings": cfg.min_vocabulary_postings,
            "n_groups_with_a_knowledge_object": len(with_obj),
            "n_groups_whose_first_object_is_tagged_knowledge": sum(
                1 for g in with_obj if g.knowledge_objects[0]["tagged_knowledge"]),
            "keep_longer_form_at": cfg.keep_longer_form_at,
            "n_groups_with_a_cognitive_level": sum(1 for g in groups if g.cognitive.get("level")),
            "cognitive_level_distribution": dict(Counter(
                g.cognitive["code"] for g in groups if g.cognitive.get("level"))),
            "highest_level_floor": cfg.highest_level_floor}
