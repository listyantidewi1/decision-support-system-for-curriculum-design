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
from .statement import INDONESIAN_VERBS, VERBS, read_phrase

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
    # "mobile development": the object is completed to the most frequent phrase
    # with that verb ("develop mobile applications") only when at least this
    # many postings write that phrase
    min_completion_postings: int = 5
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


class Reader:
    """Reads every hard-skill phrase of a corpus once: its verbs and its object.

    A phrase that opens with a verb states its object. A phrase that names its
    work with a noun made from a verb ("web application development", "unit
    testing") gives the verb, but in English not always the object: the words
    before the noun are its object in "web application development" and only
    describe it in "automated testing". The phrase alone cannot tell, so the
    corpus is asked:

      confirmed   some posting writes the verb with those very words as its
                  object ("develop web applications"): they are the object;
      completed   the words open an object that postings do write with that
                  verb, in at least `min_completion_postings` postings ("mobile"
                  -> "develop mobile applications"): the most frequent such
                  object is taken, and marked as completed;
      unconfirmed neither: the phrase counts for the level of its verb and
                  gives no object. Its statement keeps the noun phrase.

    Indonesian nouns ("pengembangan aplikasi web") and "development of X" give
    their object in place and need no such check."""

    def __init__(self, evidence: Sequence[Evidence], cfg: StructureConfig):
        self.cfg = cfg
        self.reading = {}
        seen: dict[tuple, set] = defaultdict(set)
        forms: dict[tuple, Counter] = defaultdict(Counter)
        for e in evidence:
            if e.tag != "SH":
                continue
            r = self.reading.get(e.text)
            if r is None:
                r = self.reading[e.text] = read_phrase(e.text)
            if r.obj:                                   # the object is stated in the phrase
                chunk = object_chunk(r.obj, r.indonesian)
                if chunk:
                    for v in r.verbs:
                        key = (v, self._norm(chunk))
                        seen[key].add(e.posting_id)
                        forms[key][tuple(chunk)] += 1
        self._seen, self._forms = seen, forms
        self._by_start: dict[tuple, list[tuple]] = defaultdict(list)
        for key in seen:
            self._by_start[(key[0], key[1][:1])].append(key)
        self._rows: dict[str, tuple[tuple[str, ...], list[str], bool, str]] = {}

    @staticmethod
    def _norm(chunk: Sequence[str]) -> tuple[str, ...]:
        words = [w.lower() for w in chunk]
        if words and len(words[-1]) > 3 and words[-1].endswith("s") and not words[-1].endswith("ss"):
            words[-1] = words[-1][:-1]                  # "applications" and "application" agree
        return tuple(words)

    def row(self, text: str) -> tuple[tuple[str, ...], list[str], bool, str]:
        """(verbs, object words, Indonesian?, how the object was read)."""
        if text in self._rows:
            return self._rows[text]
        r = self.reading.get(text) or read_phrase(text)
        cfg = self.cfg
        if not r.verbs:
            out = ((), [], False, "none")
        elif r.obj:
            out = (r.verbs, object_chunk(r.obj, r.indonesian), r.indonesian, "stated")
        elif r.how == "verb" or not r.modifier:
            out = (r.verbs, [], r.indonesian, "no object")
        else:
            words = object_chunk(r.modifier, False)
            norm = self._norm(words)
            verb = r.verbs[0]
            exact = (verb, norm)
            out = (r.verbs, [], False, "unconfirmed")
            if words and len(self._seen.get(exact, ())) >= cfg.min_item_postings:
                out = (r.verbs, list(max(sorted(self._forms[exact]),
                                         key=lambda f: self._forms[exact][f])), False, "confirmed")
            elif words:
                longer = [k for k in self._by_start.get((verb, norm[:1]), ())
                          if len(k[1]) == len(norm) + 1
                          and tuple(w.lower() for w in k[1][:len(norm)]) == tuple(
                              w.lower() for w in words)]
                if longer:
                    best = max(longer, key=lambda k: (len(self._seen[k]), k))
                    if len(self._seen[best]) >= cfg.min_completion_postings:
                        out = (r.verbs, list(max(sorted(self._forms[best]),
                                                 key=lambda f: self._forms[best][f])),
                               False, "completed")
        self._rows[text] = out
        return out

    def how(self, text: str) -> str:
        r = self.reading.get(text) or read_phrase(text)
        return r.how


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
    reader = Reader(evidence, cfg)
    by_id = {e.evidence_id: e for e in evidence}
    out: list[Group] = []
    too_small = 0
    read_as: Counter = Counter()
    for g in groups:
        by_level: dict[str | None, list[Evidence]] = defaultdict(list)
        for eid in g.evidence_ids:
            e = by_id[eid]
            verbs = reader.row(e.text)[0] if e.tag == "SH" else ()
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
            kinds: Counter = Counter()                  # how the units of this part were read
            nominal: dict[str, set] = defaultdict(set)  # noun phrases that gave no object
            nominal_form: dict[str, Counter] = defaultdict(Counter)
            nominal_idn: dict[str, bool] = {}
            for e in part:
                if level is None:
                    kinds["no verb"] += 1
                    continue
                verbs, chunk, idn, obj_how = reader.row(e.text)
                on_level = [v for v in verbs if VERBS[v] == level]
                how = reader.how(e.text)
                kinds["verb" if how == "verb" else "noun made from a verb"] += 1
                read_as[(how, obj_how)] += 1
                for v in on_level:
                    verb_postings[v].add(e.posting_id)
                if on_level and chunk:
                    rows.append((e.posting_id, on_level, chunk, idn))
                elif on_level and how != "verb":
                    r = reader.reading[e.text]
                    nominal[e.key].add(e.posting_id)
                    nominal_form[e.key][r.nominal or e.text] += 1
                    nominal_idn[e.key] = r.indonesian
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
                "read_from": dict(kinds),
                "basis": ("the verbs of the postings, and nouns made from a verb" if level else
                          "the postings name this work with a noun that carries no verb: "
                          "no level can be read")}
            if nominal:
                top = max(nominal, key=lambda k: (len(nominal[k]), k))
                forms = nominal_form[top]
                v.cognitive["nominal_label"] = {
                    "text": max(sorted(forms), key=lambda f: forms[f]),
                    "n_postings": len(nominal[top]), "indonesian": nominal_idn[top]}
            out.append(v)
    out.sort(key=lambda v: (-v.n_postings, v.group_id))
    report = {"n_families": len(groups), "n_competencies_by_level": len(out),
              "by_level": dict(Counter(v.cognitive["code"] or "none" for v in out)),
              "n_families_with_more_than_one_level": sum(
                  1 for n in Counter(v.family_id for v in out if v.level).values() if n > 1),
              "n_units_on_a_level_too_small_to_stand": too_small,
              "min_variant_postings": cfg.min_variant_postings,
              "units_with_a_level_by_how_they_were_read": {
                  f"{how}, object {obj}": n for (how, obj), n in sorted(read_as.items())},
              "min_completion_postings": cfg.min_completion_postings}
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


# Nouns too general to show that two families are the same work: "Develop
# solutions" or "Troubleshoot issues" is said of very different work. The first
# run of the merge linked unrelated families through such statements into
# catch-all families of thousands of postings (EXP-033).
GENERIC_NOUNS = {
    "application", "applications", "solution", "solutions", "system", "systems", "software",
    "feature", "features", "product", "products", "service", "services", "project", "projects",
    "process", "processes", "tool", "tools", "platform", "platforms", "issue", "issues",
    "problem", "problems", "task", "tasks", "work", "development", "operations", "operation",
    "initiatives", "results", "value", "things", "impact", "quality", "experience", "experiences",
    "support", "environment", "environments", "technology", "technologies", "functionality",
    "aplikasi", "solusi", "sistem", "fitur", "produk", "layanan", "proyek", "proses", "masalah",
    "pekerjaan", "pengembangan", "teknologi"}


def names_specific_work(statement: str) -> bool:
    """True when the statement's object says WHAT work: not empty, and not only
    general nouns ("Develop web applications" yes; "Develop solutions" no)."""
    from .statement import LIGHT_VERBS, base_verb
    words = [w.strip(",.;:()").lower() for w in statement.split()]
    words = [w for w in words if w]
    if not words:
        return False
    rest = words[1:]
    while rest and (rest[0] in LIGHT_VERBS or (words[0] in LIGHT_VERBS and base_verb(rest[0]))):
        rest = rest[1:]
    content = [w for w in rest if w not in {"the", "a", "an", "and", "or", "dan", "atau", "of", "for"}]
    return bool(content) and not all(w in GENERIC_NOUNS for w in content)


def merge_same_statement(families: Sequence[Group], parts: Sequence[Group],
                         evidence: Sequence[Evidence], n_postings: int,
                         grouping_cfg: GroupingConfig | None = None,
                         posting_group: dict[str, str] | None = None) -> tuple[list[Group], dict]:
    """Families whose parts get the SAME statement on the SAME level are one family.

    The clusterer separates one piece of work by its wording: "develop web
    applications" (verb) and "web application development" (noun) fell into two
    families whose centres were too far apart to merge, and both gave "Develop
    web applications" on C6 (EXP-030). Here the parts' rule statements are
    compared: two families that share one (level, statement) are merged, and so
    on transitively. A part without a level, or whose statement is a fragment
    or names no specific work ("Develop solutions"), does not link families.

    The merged family keeps the id, label, validity status and cohesion of its
    largest member; its evidence is the union; `merged_from` names the others."""
    from .statement import deterministic_statement, is_fragment
    parent = {g.group_id: g.group_id for g in families}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    first: dict[tuple[str, str], str] = {}
    links = []
    for p in parts:
        if not p.level or p.family_id not in parent:
            continue
        text = deterministic_statement(p).text
        if is_fragment(text) or not names_specific_work(text):
            continue
        key = (p.level, text.lower())
        other = first.setdefault(key, p.family_id)
        if other != p.family_id and find(other) != find(p.family_id):
            parent[find(p.family_id)] = find(other)
            links.append({"statement": text, "level": p.level, "families": [other, p.family_id]})
    sets: dict[str, list[Group]] = defaultdict(list)
    for g in families:
        sets[find(g.group_id)].append(g)
    if all(len(v) == 1 for v in sets.values()):
        return list(families), {"n_families_before": len(families), "n_merged_sets": 0,
                                "n_families_merged_away": 0, "links": []}
    describer = Describer(evidence, n_postings, grouping_cfg or GroupingConfig(), posting_group)
    by_id = {e.evidence_id: e for e in evidence}
    out: list[Group] = []
    for members in sets.values():
        if len(members) == 1:
            out.append(members[0])
            continue
        members.sort(key=lambda g: (-g.n_postings, g.group_id))
        base = members[0]
        ids = sorted({i for g in members for i in g.evidence_ids})
        merged = dataclasses.replace(
            base, merged_from=list(base.merged_from) + [g.group_id for g in members[1:]],
            **describer.describe([by_id[i] for i in ids]))
        out.append(merged)
    out.sort(key=lambda g: (-g.n_postings, g.group_id))
    return out, {"n_families_before": len(families),
                 "n_merged_sets": sum(1 for v in sets.values() if len(v) > 1),
                 "n_families_merged_away": sum(len(v) - 1 for v in sets.values()),
                 "links": links[:200]}


def annotate(groups: Sequence[Group], evidence: Sequence[Evidence],
             cfg: StructureConfig | None = None) -> dict:
    """Fills `verbs` and `knowledge_objects` of every group; returns a report."""
    cfg = cfg or StructureConfig()
    vocabulary = knowledge_vocabulary(evidence, cfg)
    reader = Reader(evidence, cfg)
    by_id = {e.evidence_id: e for e in evidence}
    tagged: dict[str, bool] = {}
    n_units = n_with_verb = n_with_tagged = n_from_noun = 0
    for g in groups:
        verb_postings: dict[str, set] = defaultdict(set)
        rows, postings = [], set()
        for eid in g.evidence_ids:
            e = by_id[eid]
            postings.add(e.posting_id)
            if e.tag != "SH":
                continue
            verbs, chunk, idn, _ = reader.row(e.text)
            if e.text not in tagged:
                tagged[e.text] = knowledge_in(e.text, vocabulary) is not None
            n_units += 1
            n_with_verb += bool(verbs)
            n_from_noun += bool(verbs) and reader.how(e.text) != "verb"
            n_with_tagged += tagged[e.text]
            for v in verbs:
                verb_postings[v].add(e.posting_id)
            if verbs and chunk:
                rows.append((e.posting_id, list(verbs), chunk, idn))
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
            "n_units_with_a_verb": n_with_verb,
            "n_units_whose_verb_is_read_from_a_noun": n_from_noun,
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
