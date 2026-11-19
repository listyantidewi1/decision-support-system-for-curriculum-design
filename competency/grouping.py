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
    technologies: list[dict] = field(default_factory=list)      # {text, n_postings}
    knowledge_context: list[dict] = field(default_factory=list)
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


class _Knowledge:
    """A knowledge item is recognised by the ABSENCE of a `type` attribute."""

    def __init__(self, e: Evidence):
        self.text, self.sentence_text, self.source = e.text, e.sentence_text, e.source
        self.sentence_id = e.sentence_id.replace("#", "_")
        self.evidence_id = e.evidence_id


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


def _audit(groups: list[Group]) -> None:
    from pipeline_scientific_audits import audit_cluster_validity
    dicts = [{"id": g.group_id, "summary_label": g.label, "top_terms": g.top_terms,
              "items": [m["text"] for m in g.members], "cohesion_score": g.cohesion,
              "n_items": len(g.members), "n_unique_jobs": g.n_postings} for g in groups]
    with tempfile.TemporaryDirectory() as tmp:
        report = audit_cluster_validity(dicts, Path(tmp))
    by_id = report["cluster_validity_by_id"]
    for g in groups:
        row = by_id.get(g.group_id, {})
        g.status_code = row.get("cluster_validity_status", "")
        g.status = GROUP_STATUS.get(g.status_code, g.status_code)
        g.status_flags = list(row.get("cluster_validity_flags", []))


def group_evidence(evidence: Sequence[Evidence], n_postings: int,
                   cfg: GroupingConfig | None = None,
                   clusterer: Callable | None = None,
                   posting_group: dict[str, str] | None = None) -> tuple[list[Group], dict]:
    """(groups, report). `n_postings` is the denominator for every share.
    `posting_group` maps a posting to its employer/template group; without it
    every posting counts as its own group."""
    cfg = cfg or GroupingConfig()
    posting_group = posting_group or {}
    by_id = {e.evidence_id: e for e in evidence}
    items = []
    for e in evidence:
        if e.tag == "SH":
            items.append(_Skill(e, "hard"))
        elif e.tag == "SS":
            items.append(_Skill(e, "soft"))
        elif e.tag == "KN":
            items.append(_Knowledge(e))
    clusters, cluster_report = (clusterer or _default_clusterer)(items, cfg)

    tech_by_sentence: dict[str, list[Evidence]] = defaultdict(list)
    kn_by_sentence: dict[str, list[Evidence]] = defaultdict(list)
    for e in evidence:
        if e.tag == "TT":
            tech_by_sentence[e.sentence_id].append(e)
        elif e.tag == "KN":
            kn_by_sentence[e.sentence_id].append(e)

    groups: list[Group] = []
    for c in clusters:
        ev = [by_id[i.evidence_id] for i in c.items]
        if not ev:
            continue
        ids = {e.evidence_id for e in ev}
        postings = {e.posting_id for e in ev}
        sentence_ids = {e.sentence_id for e in ev}
        members: dict[tuple[str, str], dict] = {}
        for e in ev:
            m = members.setdefault((e.key, e.tag), {"text": e.text, "tag": e.tag, "n_evidence": 0,
                                                    "_postings": set(), "_forms": Counter()})
            m["n_evidence"] += 1
            m["_postings"].add(e.posting_id)
            m["_forms"][e.text] += 1
        member_rows = []
        for m in members.values():
            member_rows.append({"text": max(sorted(m["_forms"]), key=lambda s: m["_forms"][s]),
                                "tag": m["tag"], "n_evidence": m["n_evidence"],
                                "n_postings": len(m["_postings"])})
        member_rows.sort(key=lambda r: (-r["n_postings"], -r["n_evidence"], r["text"]))

        def context(table, exclude_members: bool):
            seen: dict[str, dict] = {}
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

        stream = "soft" if getattr(c, "stream", "") == "soft_skill" else "technical"
        samples, used = [], set()
        for e in sorted(ev, key=lambda e: (e.posting_id, e.sentence_id, e.start)):
            if e.posting_id in used:
                continue
            used.add(e.posting_id)
            samples.append({"evidence_id": e.evidence_id, "text": e.text, "tag": e.tag,
                            "sentence_text": e.sentence_text})
            if len(samples) >= cfg.max_samples:
                break
        groups.append(Group(
            group_id=str(c.id), stream=stream, label=str(c.summary_label),
            top_terms=list(c.top_terms), members=member_rows,
            evidence_ids=sorted(ids), n_evidence=len(ev), n_postings=len(postings),
            share_of_postings=len(postings) / n_postings if n_postings else 0.0,
            n_posting_groups=len({posting_group.get(p, p) for p in postings}),
            cohesion=float(c.cohesion_score), tag_counts=dict(Counter(e.tag for e in ev)),
            technologies=context(tech_by_sentence, False),
            knowledge_context=context(kn_by_sentence, True), samples=samples))
    groups.sort(key=lambda g: (-g.n_postings, g.group_id))
    if groups:
        _audit(groups)
    grouped = {i for g in groups for i in g.evidence_ids}
    eligible = [e for e in evidence if e.tag in ("SH", "SS", "KN")]
    report = {"grouping_version": cfg.version, "embedder_model": cfg.embedder_model,
              "n_evidence_groupable": len(eligible),
              "n_evidence_grouped": len(grouped),
              "n_evidence_left_ungrouped": len(eligible) - len(grouped),
              "n_groups": len(groups),
              "status_distribution": dict(Counter(g.status for g in groups)),
              "clusterer": cluster_report}
    return groups, report
