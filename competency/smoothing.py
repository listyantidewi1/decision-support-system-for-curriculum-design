"""How sure the counts are: intervals, pooled shares, and ranks with a margin.

Three small statistics, each answering one way a reader could be misled by a
bare number.

  demand interval   "asked for by 4.2% of postings" is an estimate from a sample
                    of postings. `demand` adds the 95% Wilson interval.

  pooled level mix  A family's cognitive levels are counted from its own
                    postings. A level seen in 3 postings of a small family is a
                    weak basis for "12% of this family is C5". `pool_level_mix`
                    estimates each family's level shares together with all the
                    others (a Dirichlet-multinomial model, empirical Bayes):
                    every family is pulled towards the mix of all families,
                    strongly when it has few postings and hardly at all when it
                    has many. How strong the pull is (`strength`, in "postings'
                    worth of prior evidence") is estimated from the data by
                    maximum likelihood, not chosen.

  rank interval     "the 12th most demanded" is also an estimate. `bootstrap_
                    ranks` resamples the postings 200 times and reports the
                    range that holds 90% of the ranks an item received. Two
                    items whose ranges overlap are not shown to differ.

None of this repairs wrong evidence: an interval around a count of phrases that
are not requirements is still an interval around the wrong thing.
"""
from __future__ import annotations

from typing import Mapping, Sequence

SMOOTHING_VERSION = "smoothing-v1"
BOOTSTRAP_ROUNDS = 200
BOOTSTRAP_SEED = 20261120


def demand(n_postings: int, n_all: int) -> dict:
    from gold_standard.metrics import wilson_ci
    lo, hi = wilson_ci(n_postings, n_all) if n_all else (0.0, 0.0)
    return {"n_postings": n_postings, "share": round(n_postings / n_all, 6) if n_all else 0.0,
            "ci95": [round(lo, 6), round(hi, 6)]}


def pool_level_mix(counts: Mapping[str, Mapping[str, int]]) -> dict:
    """`counts[family][level]` = postings of that family on that level.

    Returns {"strength", "mix", "shares": {family: {level: {"raw", "pooled"}}}}."""
    import numpy as np
    from scipy.special import gammaln
    levels = sorted({lv for row in counts.values() for lv in row})
    families = [f for f in sorted(counts) if sum(counts[f].values()) > 0]
    if not families or not levels:
        return {"strength": None, "mix": {}, "shares": {}}
    k = np.array([[counts[f].get(lv, 0) for lv in levels] for f in families], dtype=float)
    n = k.sum(1)
    pi = k.sum(0) / k.sum()
    best_m, best_ll = None, -np.inf
    if len(families) >= 5 and len(levels) >= 2:
        for m in np.logspace(-1, 3, 81):
            a = m * pi
            use = a > 0
            ll = float((gammaln(m) - gammaln(n + m)).sum()
                       + (gammaln(k[:, use] + a[use]) - gammaln(a[use])).sum())
            if ll > best_ll:
                best_m, best_ll = float(m), ll
    m = best_m if best_m is not None else 0.0
    pooled = (k + m * pi) / (n[:, None] + m)
    return {"strength": round(m, 3) if best_m is not None else None,
            "mix": {lv: round(float(p), 4) for lv, p in zip(levels, pi)},
            "shares": {f: {lv: {"raw": round(float(k[i, j] / n[i]), 4),
                                "pooled": round(float(pooled[i, j]), 4)}
                           for j, lv in enumerate(levels) if k[i, j] > 0}
                       for i, f in enumerate(families)}}


def bootstrap_ranks(item_postings: Mapping[str, set], postings: Sequence[str],
                    rounds: int = BOOTSTRAP_ROUNDS, seed: int = BOOTSTRAP_SEED) -> dict[str, dict]:
    """item -> {"rank", "rank_interval90": [best, worst]} by number of postings.

    `postings` are all postings of the run (the ones resampled), not only those
    that mention an item."""
    import numpy as np
    from scipy import sparse
    items = sorted(item_postings)
    if not items or not postings:
        return {}
    col = {p: j for j, p in enumerate(sorted(set(postings)))}
    r, c = [], []
    for i, it in enumerate(items):
        for p in item_postings[it]:
            if p in col:
                r.append(i)
                c.append(col[p])
    a = sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(items), len(col)))
    counts = np.asarray(a.sum(1)).ravel()

    def ranks(v):
        order = np.argsort(-v, kind="stable")
        out = np.empty(len(v), dtype=int)
        out[order] = np.arange(1, len(v) + 1)
        return out
    point = ranks(counts)
    rng = np.random.default_rng(seed)
    n = len(col)
    draws = np.empty((rounds, len(items)), dtype=int)
    for b in range(rounds):
        w = rng.multinomial(n, np.full(n, 1.0 / n))
        draws[b] = ranks(a @ w)
    lo = np.percentile(draws, 5, axis=0)
    hi = np.percentile(draws, 95, axis=0)
    return {it: {"rank": int(point[i]), "rank_interval90": [int(round(lo[i])), int(round(hi[i]))]}
            for i, it in enumerate(items)}
