"""Group the evidence by meaning and give every group a validity status.

The grouping itself is the existing clusterer (`clustering.cluster_skills`:
sentence embeddings, HDBSCAN, a cohesion floor). This module puts five-tag
evidence in front of it and reads its clusters back as groups of evidence:

  * SKILL_HARD and KNOWLEDGE go into the technical stream, SKILL_SOFT into the
    soft stream;
  * TOOL_TECHNOLOGY is attached afterwards, by counting which technologies
    share a SENTENCE with a group's evidence. A technology counted this way is
    one the postings name in the same breath as the action — not merely
    somewhere in the same advertisement;
  * the status comes from the existing validity audit and is reported under
    the proposal's six names.

`clusterer` can be replaced by any function with the same signature; the tests
use a simple one so they need no embedding model.
"""
from __future__ import annotations

import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Sequence

from . import GROUP_STATUS
from .evidence import Evidence

MULTILINGUAL_EMBEDDER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass(frozen=True)
class GroupingConfig:
    # The corpus mixes Indonesian and English, so the multilingual model, not
    # the clusterer's English default.
    embedder_model: str = MULTILINGUAL_EMBEDDER
    # 1, not the clusterer's own default of 2. That default was made for short
    # skill terms. Five-tag action phrases are long and mostly unique: in the
    # first run 5,111 distinct phrases shrank to 360 under "seen in >= 2
    # postings", and two thirds of the evidence never reached the clusterer.
    # Rarity is judged per GROUP instead (the audit's low-frequency status and
    # the contract's min_postings).
    min_postings_per_phrase: int = 1
    # Which tags a competency is BUILT from. The researcher's decision of
    # 2026-11-19: competencies come from hard skills. Everything else describes
    # a competency: the tools used for it, the knowledge behind it, the soft
    # skills and work behaviours expected of the person who has it.
    competency_tags: tuple[str, ...] = ("SH",)
    max_profile_items: int = 8            # soft skills / work behaviours listed per competency
    min_profile_postings: int = 2
    # Two-stage grouping, for a corpus too large to cluster every phrase.
    # Stage 1 clusters only the phrases seen in at least `core_min_postings`
    # postings: they define the themes. Stage 2 attaches each rarer phrase to
    # the nearest theme when it is close enough, and otherwise leaves it out.
    # 1 = one stage, every phrase clustered (fine up to a few thousand postings).
    core_min_postings: int = 1
    # floor of the cosine to a group's centre; the working threshold per group
    # is max(this, sqrt(cohesion of the group)). A starting value.
    attach_threshold: float = 0.75
    cohesion_threshold: float = 0.55
    max_technologies: int = 8
    min_technology_postings: int = 2      # a technology must co-occur in this many postings
    max_samples: int = 5
    version: str = "grouping-v1"


@dataclass
class Group:
    group_id: str
    stream: str                           # technical | soft
    label: str
    top_terms: list[str]
    members: list[dict]                   # {text, tag, n_evidence, n_postings}
    evidence_ids: list[str]
    n_evidence: int
    n_postings: int
    share_of_postings: float
    # Postings that share an employer or a template are one "posting group"
    # (the leakage groups of the gold-standard splitter). One employer posting
    # the same advertisement 36 times is 36 postings and 1 group; a demand
    # count that ignores this measures how often an employer advertises.
    n_posting_groups: int
    cohesion: float
    tag_counts: dict[str, int]
    # named in the SAME SENTENCE as the group's phrases
    technologies: list[dict] = field(default_factory=list)      # {text, n_postings}
    knowledge_context: list[dict] = field(default_factory=list)
    # asked for in the SAME POSTINGS: {text, n_postings, share, lift}. `share` is
    # the part of this group's postings that also ask for it; `lift` is that
    # share divided by its share over all postings (above 1 = asked for more
    # often with this competency than in general).
    soft_skills: list[dict] = field(default_factory=list)
    work_behaviours: list[dict] = field(default_factory=list)
    knowledge: list[dict] = field(default_factory=list)
    # set by competency.structure: the verbs the postings use for this work and
    # the knowledge those verbs act on (read from inside the phrases)
    verbs: list[dict] = field(default_factory=list)
    knowledge_objects: list[dict] = field(default_factory=list)
    cognitive: dict = field(default_factory=dict)       # the level the postings' verbs ask for
    # set by competency.structure.split_by_level: a group is split into one part
    # per cognitive level; each part is a competency of its own
    family_id: str = ""                   # the group this part was split from
    level: str | None = None              # the cognitive level of this part
    status: str = ""
    status_code: str = ""
    status_flags: list[str] = field(default_factory=list)
    samples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class _Skill:
    """What the clusterer reads from a skill item (it is duck-typed)."""

    def __init__(self, e: Evidence, kind: str):
        self.text, self.sentence_text, self.source = e.text, e.sentence_text, e.source
        self.sentence_id = e.sentence_id.replace("#", "_")     # "<posting>_<n>": its job-id rule
        self.type = SimpleNamespace(value=kind)                # "hard" | "soft"
        self.evidence_id = e.evidence_id
        self.key, self.posting_id = e.key, e.posting_id
        self.stream = "soft" if kind == "soft" else "technical"


class _Knowledge:
    """A knowledge item is recognised by the ABSENCE of a `type` attribute."""

    def __init__(self, e: Evidence):
        self.text, self.sentence_text, self.source = e.text, e.sentence_text, e.source
        self.sentence_id = e.sentence_id.replace("#", "_")
        self.evidence_id = e.evidence_id
        self.key, self.posting_id, self.stream = e.key, e.posting_id, "technical"


def _default_clusterer(items, cfg: GroupingConfig):
    from clustering import ClusteringConfig, cluster_skills
    ccfg = ClusteringConfig(embedder_model=cfg.embedder_model,
                            min_global_frequency=cfg.min_postings_per_phrase,
                            cohesion_threshold=cfg.cohesion_threshold)
    clusters, report = cluster_skills(items, ccfg)
    return clusters, {"n_items_input": report.n_items_input,
                      "n_items_canonical": report.n_items_canonical,
                      "n_items_after_frequency_filter": report.n_items_after_frequency_filter,
                      "n_items_ungrouped": report.n_items_ungrouped,
                      "n_clusters_dropped_low_cohesion": report.n_clusters_dropped_low_cohesion,
                      "runtime_seconds": report.runtime_seconds}


def _default_embed(texts: list[str], cfg: GroupingConfig):
    """Embeddings of phrases, made the way the clusterer makes them (same
    model, same prefix, unit length), so a phrase can be compared with the
    centre of a cluster."""
    from clustering import ClusteringConfig
    from clustering.skill_clusterer import _get_embedder
    prefix = ClusteringConfig().template_prefix
    return _get_embedder(cfg.embedder_model).encode(
        [prefix + t for t in texts], normalize_embeddings=True, convert_to_numpy=True,
        batch_size=256, show_progress_bar=False)


def attach_tail(clusters, tail_items, cfg: GroupingConfig, embed=None) -> dict:
    """Stage 2: add each rare phrase to the nearest cluster of its own stream
    when the cosine to that cluster's centre reaches `attach_threshold`.
    Clusters are changed in place (their `items` grow); their centre and
    cohesion are those of stage 1 and are not recomputed."""
    import numpy as np
    by_key: dict[tuple[str, str], list] = defaultdict(list)
    for it in tail_items:
        by_key[(it.stream, it.key)].append(it)
    report = {"n_tail_phrases": len(by_key), "n_tail_evidence": len(tail_items),
              "n_phrases_attached": 0, "n_evidence_attached": 0,
              "attach_threshold": cfg.attach_threshold}
    for stream in ("technical", "soft"):
        cs = [c for c in clusters
              if ("soft" if getattr(c, "stream", "") == "soft_skill" else "technical") == stream
              and getattr(c, "centroid_embedding", None) is not None]
        keys = [k for k in by_key if k[0] == stream]
        if not cs or not keys:
            continue
        centres = np.array([np.asarray(c.centroid_embedding, dtype=np.float32) for c in cs])
        norms = np.linalg.norm(centres, axis=1, keepdims=True)
        centres = centres / np.where(norms > 0, norms, 1.0)
        vecs = (embed or (lambda t: _default_embed(t, cfg)))([by_key[k][0].text for k in keys])
        sims = np.asarray(vecs, dtype=np.float32) @ centres.T
        # A phrase joins a group only if it is at least as close to the centre
        # as the group's own members typically are. For unit vectors the mean
        # cosine of members to their centre is about sqrt(cohesion), cohesion
        # being the mean pairwise cosine. `attach_threshold` is the floor.
        # (First whole-pool attempt, EXP-026: a flat 0.75 attached 94% of all
        # rare phrases, because the shared embedding prefix inflates cosines.)
        need = np.array([max(cfg.attach_threshold,
                             float(max(0.0, getattr(c, "cohesion_score", 0.0))) ** 0.5)
                         for c in cs], dtype=np.float32)
        sims = np.where(sims >= need[None, :], sims, -1.0)
        best = sims.argmax(axis=1)
        for row, k in enumerate(keys):
            if float(sims[row, best[row]]) >= 0.0:
                for it in by_key[k]:
                    it.attached = True               # a variant added in stage 2
                cs[int(best[row])].items.extend(by_key[k])
                report["n_phrases_attached"] += 1
                report["n_evidence_attached"] += len(by_key[k])
    return report


def _audit(groups: list[Group]) -> None:
    from pipeline_scientific_audits import audit_cluster_validity
    # The audit judges a group as the clusterer formed it: its size and cohesion
    # thresholds were set for that. Phrases attached in stage 2 are variants of
    # the group's themes, so they do not count towards "too large to be one
    # group". (First whole-pool run, EXP-026: counting them flagged 697 of 912
    # groups as needing a split.) Their postings still count as demand.
    dicts = []
    for g in groups:
        core = [m for m in g.members if not m.get("attached")] or g.members
        dicts.append({"id": g.group_id, "summary_label": g.label, "top_terms": g.top_terms,
                      "items": [m["text"] for m in core], "cohesion_score": g.cohesion,
                      "n_items": len(core), "n_unique_jobs": g.n_postings})
    with tempfile.TemporaryDirectory() as tmp:
        report = audit_cluster_validity(dicts, Path(tmp))
    by_id = report["cluster_validity_by_id"]
    for g in groups:
        row = by_id.get(g.group_id, {})
        g.status_code = row.get("cluster_validity_status", "")
        g.status = GROUP_STATUS.get(g.status_code, g.status_code)
        g.status_flags = list(row.get("cluster_validity_flags", []))


class Describer:
    """What a set of evidence units says about itself: its phrases, its postings,
    the tools named in its sentences, and what its postings also ask for.

    Built once from all the evidence; `describe` is then called for a whole
    group and again for each part of it (one part per cognitive level, see
    `competency.structure.split_by_level`), so a part is described by exactly
    the same rules as the group it came from."""

    def __init__(self, evidence: Sequence[Evidence], n_postings: int, cfg: GroupingConfig,
                 posting_group: dict[str, str] | None = None):
        self.cfg, self.n_postings, self.posting_group = cfg, n_postings, posting_group or {}
        self.tech_by_sentence: dict[str, list[Evidence]] = defaultdict(list)
        self.kn_by_sentence: dict[str, list[Evidence]] = defaultdict(list)
        # Knowledge is profiled both ways: `knowledge_context` (same sentence, rare
        # but direct) and `knowledge` (same postings: knowledge is usually asked for
        # on a line of its own, like soft skills).
        self.profile_postings: dict[str, dict[str, set]] = {t: defaultdict(set)
                                                            for t in ("SS", "WB", "KN")}
        self.profile_forms: dict[str, dict[str, Counter]] = {t: defaultdict(Counter)
                                                             for t in ("SS", "WB", "KN")}
        for e in evidence:
            if e.tag == "TT":
                self.tech_by_sentence[e.sentence_id].append(e)
            elif e.tag == "KN":
                self.kn_by_sentence[e.sentence_id].append(e)
            if e.tag in self.profile_postings and e.tag not in cfg.competency_tags:
                self.profile_postings[e.tag][e.key].add(e.posting_id)
                self.profile_forms[e.tag][e.key][e.text] += 1

    def _posting_profile(self, tag: str, postings: set) -> list[dict]:
        """What the postings of a group also ask for, with how typical that is."""
        cfg, rows = self.cfg, []
        for key, ps in self.profile_postings[tag].items():
            both = len(ps & postings)
            if both < cfg.min_profile_postings:
                continue
            share = both / len(postings)
            base = len(ps) / self.n_postings if self.n_postings else 0.0
            forms = self.profile_forms[tag][key]
            rows.append({"text": max(sorted(forms), key=lambda s: forms[s]), "n_postings": both,
                         "share": round(share, 4),
                         "lift": round(share / base, 2) if base > 0 else None})
        rows.sort(key=lambda r: (-r["n_postings"], r["text"]))
        return rows[:cfg.max_profile_items]

    def _context(self, table, sentence_ids: set, ids: set, exclude_members: bool) -> list[dict]:
        cfg, seen = self.cfg, {}
        for sid in sentence_ids:
            for t in table.get(sid, ()):
                if exclude_members and t.evidence_id in ids:
                    continue
                row = seen.setdefault(t.key, {"_postings": set(), "_forms": Counter()})
                row["_postings"].add(t.posting_id)
                row["_forms"][t.text] += 1
        rows = [{"text": max(sorted(r["_forms"]), key=lambda s: r["_forms"][s]),
                 "n_postings": len(r["_postings"])} for r in seen.values()
                if len(r["_postings"]) >= cfg.min_technology_postings]
        return sorted(rows, key=lambda r: (-r["n_postings"], r["text"]))[:cfg.max_technologies]

    def describe(self, ev: Sequence[Evidence], attached_ids: set | frozenset = frozenset()) -> dict:
        """The fields of a `Group` that are counted from its evidence."""
        cfg = self.cfg
        ids = {e.evidence_id for e in ev}
        postings = {e.posting_id for e in ev}
        sentence_ids = {e.sentence_id for e in ev}
        members: dict[tuple[str, str], dict] = {}
        for e in ev:
            m = members.setdefault((e.key, e.tag), {"text": e.text, "tag": e.tag, "n_evidence": 0,
                                                    "_postings": set(), "_forms": Counter(),
                                                    "_attached": True})
            m["_attached"] = m["_attached"] and e.evidence_id in attached_ids
            m["n_evidence"] += 1
            m["_postings"].add(e.posting_id)
            m["_forms"][e.text] += 1
        member_rows = []
        for m in members.values():
            member_rows.append({"text": max(sorted(m["_forms"]), key=lambda s: m["_forms"][s]),
                                "tag": m["tag"], "n_evidence": m["n_evidence"],
                                "n_postings": len(m["_postings"]),
                                "attached": bool(m["_attached"])})
        member_rows.sort(key=lambda r: (r["attached"], -r["n_postings"], -r["n_evidence"],
                                        r["text"]))
        samples, used = [], set()
        for e in sorted(ev, key=lambda e: (e.posting_id, e.sentence_id, e.start)):
            if e.posting_id in used:
                continue
            used.add(e.posting_id)
            samples.append({"evidence_id": e.evidence_id, "text": e.text, "tag": e.tag,
                            "sentence_text": e.sentence_text})
            if len(samples) >= cfg.max_samples:
                break
        return dict(
            members=member_rows, evidence_ids=sorted(ids), n_evidence=len(ev),
            n_postings=len(postings),
            share_of_postings=len(postings) / self.n_postings if self.n_postings else 0.0,
            n_posting_groups=len({self.posting_group.get(p, p) for p in postings}),
            tag_counts=dict(Counter(e.tag for e in ev)),
            technologies=self._context(self.tech_by_sentence, sentence_ids, ids, False),
            knowledge_context=self._context(self.kn_by_sentence, sentence_ids, ids, True),
            soft_skills=self._posting_profile("SS", postings),
            work_behaviours=self._posting_profile("WB", postings),
            knowledge=self._posting_profile("KN", postings), samples=samples)


def group_evidence(evidence: Sequence[Evidence], n_postings: int,
                   cfg: GroupingConfig | None = None,
                   clusterer: Callable | None = None,
                   posting_group: dict[str, str] | None = None,
                   embed: Callable | None = None,
                   keep_clusters: dict | None = None) -> tuple[list[Group], dict]:
    """(groups, report). `n_postings` is the denominator for every share.
    `posting_group` maps a posting to its employer/template group; without it
    every posting counts as its own group."""
    cfg = cfg or GroupingConfig()
    posting_group = posting_group or {}
    by_id = {e.evidence_id: e for e in evidence}
    items = []
    for e in evidence:
        if e.tag not in cfg.competency_tags:
            continue
        if e.tag == "SH":
            items.append(_Skill(e, "hard"))
        elif e.tag == "SS":
            items.append(_Skill(e, "soft"))
        elif e.tag == "KN":
            items.append(_Knowledge(e))
    core, tail, attach_report = items, [], None
    if cfg.core_min_postings > 1:
        seen: dict[tuple[str, str], set] = defaultdict(set)
        for it in items:
            seen[(it.stream, it.key)].add(it.posting_id)
        core = [it for it in items if len(seen[(it.stream, it.key)]) >= cfg.core_min_postings]
        tail = [it for it in items if len(seen[(it.stream, it.key)]) < cfg.core_min_postings]
    clusters, cluster_report = (clusterer or _default_clusterer)(core, cfg)
    if tail and clusters:
        attach_report = attach_tail(clusters, tail, cfg, embed)

    describer = Describer(evidence, n_postings, cfg, posting_group)
    groups: list[Group] = []
    for c in clusters:
        ev = [by_id[i.evidence_id] for i in c.items]
        if not ev:
            continue
        if keep_clusters is not None:            # for the v2 generator bridge
            keep_clusters[str(c.id)] = c
        attached_ids = {i.evidence_id for i in c.items if getattr(i, "attached", False)}
        stream = "soft" if getattr(c, "stream", "") == "soft_skill" else "technical"
        groups.append(Group(
            group_id=str(c.id), stream=stream, label=str(c.summary_label),
            top_terms=list(c.top_terms), cohesion=float(c.cohesion_score),
            **describer.describe(ev, attached_ids)))
    groups.sort(key=lambda g: (-g.n_postings, g.group_id))
    if groups:
        _audit(groups)
    grouped = {i for g in groups for i in g.evidence_ids}
    eligible = [e for e in evidence if e.tag in cfg.competency_tags]
    report = {"grouping_version": cfg.version, "embedder_model": cfg.embedder_model,
              "n_evidence_groupable": len(eligible),
              "n_evidence_grouped": len(grouped),
              "n_evidence_left_ungrouped": len(eligible) - len(grouped),
              "n_groups": len(groups),
              "status_distribution": dict(Counter(g.status for g in groups)),
              "two_stage": {"core_min_postings": cfg.core_min_postings,
                            "n_evidence_in_stage_1": len(core), **(attach_report or {})}
              if cfg.core_min_postings > 1 else None,
              "clusterer": cluster_report}
    return groups, report
