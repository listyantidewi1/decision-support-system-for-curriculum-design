"""A map of knowledge terms and of tools and technology, grouped by what is
asked for together.

What the reader sees: every frequent KNOWLEDGE term and every frequent
TOOL_TECHNOLOGY name is one point. Two points are close when the same postings
ask for both ("React", "TypeScript" and "Next.js" end up side by side; so do
"relational databases" and "SQL"). Points are grouped into clusters, and each
cluster is named after its three most frequent terms.

How it is computed, in the order of the code:

  1. For every term, the set of postings that ask for it. The most frequent
     `max_terms_per_tag` terms of each tag with at least `min_postings`
     postings are kept.
  2. For every pair of terms, how much more often they share a posting than
     their frequencies alone would give (positive pointwise mutual information).
  3. That table is reduced to `svd_dims` dimensions (truncated SVD). Each term
     is now a vector that describes the company it keeps.
  4. Clusters: average-linkage agglomerative clustering on the cosine distance
     between those vectors, cut at `n_clusters` (default: the square root of
     the number of terms, at most 20).
  5. The picture: t-SNE of the same vectors into two dimensions, with a fixed
     seed.

What it is not. The clusters are of CO-OCCURRENCE, not of meaning: two terms
are together because employers ask for them together. A language is therefore
also a cluster (Indonesian terms are asked for in Indonesian postings). The
positions in the picture are a drawing aid: only "near" and "far" mean
something, the axes mean nothing, and another seed gives another drawing of the
same clusters. The number of clusters is a setting, not a discovery.

Reads only term names and posting counts. Writes `term_map.json`: no sentence,
no posting identifier.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

TERMMAP_VERSION = "termmap-v1"
TAG_NAME = {"KN": "knowledge", "TT": "technology"}


@dataclass(frozen=True)
class TermMapConfig:
    tags: tuple[str, ...] = ("KN", "TT")
    min_postings: int = 5
    max_terms_per_tag: int = 250
    svd_dims: int = 32
    n_clusters: int | None = None         # None: round(sqrt(number of terms)), at most 20
    perplexity: float = 30.0
    seed: int = 20261120
    version: str = TERMMAP_VERSION


def build(rows: Iterable[tuple[str, str, str, str]], n_postings: int,
          cfg: TermMapConfig | None = None) -> dict:
    """`rows` are (posting_id, tag, key, text) of evidence units."""
    cfg = cfg or TermMapConfig()
    postings: dict[tuple[str, str], set] = defaultdict(set)
    forms: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for posting_id, tag, key, text in rows:
        if tag in cfg.tags:
            postings[(tag, key)].add(posting_id)
            forms[(tag, key)][text] += 1
    chosen: list[tuple[str, str]] = []
    for tag in cfg.tags:
        of_tag = sorted((k for k in postings if k[0] == tag and len(postings[k]) >= cfg.min_postings),
                        key=lambda k: (-len(postings[k]), k[1]))
        chosen.extend(of_tag[:cfg.max_terms_per_tag])
    n = len(chosen)
    terms = [{"term": max(sorted(forms[k]), key=lambda s: forms[k][s]), "tag": k[0],
              "kind": TAG_NAME.get(k[0], k[0]), "n_postings": len(postings[k]),
              "share": round(len(postings[k]) / n_postings, 4) if n_postings else 0.0,
              "x": float(i), "y": 0.0, "cluster": 1} for i, k in enumerate(chosen)]
    out = {"version": cfg.version, "settings": asdict(cfg), "n_postings": n_postings,
           "n_terms": n, "method": "PPMI of posting co-occurrence -> truncated SVD -> "
                                   "average-linkage clustering (cosine); t-SNE for the drawing",
           "notice": "Clusters of co-occurrence, not of meaning. Positions are a drawing aid.",
           "terms": terms, "clusters": []}
    if n >= 6:
        vectors = _vectors([postings[k] for k in chosen], n_postings, cfg)
        labels = _cluster(vectors, cfg)
        xy = _layout(vectors, cfg)
        for t, lab, (x, y) in zip(terms, labels, xy):
            t["cluster"], t["x"], t["y"] = int(lab), round(float(x), 3), round(float(y), 3)
    # number the clusters by weight: cluster 1 holds the most postings
    weight: Counter = Counter()
    for t in terms:
        weight[t["cluster"]] += t["n_postings"]
    order = {old: new for new, (old, _) in enumerate(
        sorted(weight.items(), key=lambda kv: (-kv[1], kv[0])), 1)}
    for t in terms:
        t["cluster"] = order[t["cluster"]]
    for cid in sorted(set(order.values())):
        members = sorted((t for t in terms if t["cluster"] == cid),
                         key=lambda t: (-t["n_postings"], t["term"]))
        out["clusters"].append({
            "cluster": cid, "label": ", ".join(t["term"] for t in members[:3]),
            "n_terms": len(members),
            "n_knowledge": sum(1 for t in members if t["tag"] == "KN"),
            "n_technology": sum(1 for t in members if t["tag"] == "TT"),
            "top_terms": [t["term"] for t in members[:12]]})
    return out


def _vectors(term_postings: list[set], n_postings: int, cfg: TermMapConfig):
    import numpy as np
    from scipy import sparse
    from sklearn.decomposition import TruncatedSVD
    index: dict[str, int] = {}
    r, c = [], []
    for i, ps in enumerate(term_postings):
        for p in ps:
            r.append(i)
            c.append(index.setdefault(p, len(index)))
    a = sparse.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)),
                          shape=(len(term_postings), len(index)))
    co = (a @ a.T).toarray().astype(np.float64)
    freq = co.diagonal().copy()
    total = float(max(n_postings, len(index), 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(co * total / np.outer(freq, freq))
    pmi[~np.isfinite(pmi)] = 0.0
    pmi = np.maximum(pmi, 0.0)
    np.fill_diagonal(pmi, 0.0)
    k = max(2, min(cfg.svd_dims, len(term_postings) - 1))
    vec = TruncatedSVD(n_components=k, random_state=cfg.seed).fit_transform(pmi)
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    # a term that shares a posting with no other kept term has a zero vector:
    # give it a direction of its own so that cosine distance is defined
    lonely = (norm[:, 0] < 1e-12)
    vec = vec / np.where(norm < 1e-12, 1.0, norm)
    if lonely.any():
        vec[lonely, :] = 0.0
        vec[lonely, 0] = 1.0
    return vec


def _cluster(vectors, cfg: TermMapConfig):
    from sklearn.cluster import AgglomerativeClustering
    n = len(vectors)
    k = cfg.n_clusters or max(2, min(20, round(math.sqrt(n))))
    k = max(2, min(k, n - 1))
    return AgglomerativeClustering(n_clusters=k, metric="cosine",
                                   linkage="average").fit_predict(vectors)


def _layout(vectors, cfg: TermMapConfig):
    from sklearn.manifold import TSNE
    n = len(vectors)
    perplexity = max(2.0, min(cfg.perplexity, (n - 1) / 3.0))
    return TSNE(n_components=2, metric="cosine", init="random", perplexity=perplexity,
                random_state=cfg.seed).fit_transform(vectors)


def from_evidence(evidence, n_postings: int, cfg: TermMapConfig | None = None) -> dict:
    return build(((e.posting_id, e.tag, e.key, e.text) for e in evidence), n_postings, cfg)


def write(run_dir: Path, cfg: TermMapConfig | None = None) -> dict:
    """Builds `term_map.json` for a finished run from its evidence table."""
    import pandas as pd
    run_dir = Path(run_dir)
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    df = pd.read_parquet(run_dir / "evidence.parquet", columns=["posting_id", "tag", "key", "text"])
    out = build(df.itertuples(index=False, name=None), report["denominators"]["n_postings"], cfg)
    out["source_run"] = run_dir.name
    out["evidence_source"] = report.get("evidence_source", "")
    (run_dir / "term_map.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                           encoding="utf-8")
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Cluster map of knowledge terms and technologies of a run.")
    ap.add_argument("run_dir")
    ap.add_argument("--min-postings", type=int, default=TermMapConfig.min_postings)
    ap.add_argument("--max-terms-per-tag", type=int, default=TermMapConfig.max_terms_per_tag)
    ap.add_argument("--n-clusters", type=int, default=None)
    args = ap.parse_args(argv)
    out = write(Path(args.run_dir), TermMapConfig(min_postings=args.min_postings,
                                                  max_terms_per_tag=args.max_terms_per_tag,
                                                  n_clusters=args.n_clusters))
    print(f"{out['n_terms']} terms in {len(out['clusters'])} clusters -> "
          f"{Path(args.run_dir) / 'term_map.json'}")
    for c in out["clusters"]:
        print(f"  {c['cluster']:2d}  {c['n_terms']:3d} terms "
              f"({c['n_knowledge']} knowledge, {c['n_technology']} technology)  {c['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
