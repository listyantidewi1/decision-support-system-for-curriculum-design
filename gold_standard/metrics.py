"""Shared evaluation engine — one definition of every number the study reports.

`docs/IMPLEMENTATION_PLAN.md` §5 asks for a single metrics module reused by the
gold standard (IAA), the multi-LLM ensemble (silver QC), the E0–E6 benchmarks and
the end-to-end runs. This is that module.

**The matching kernels are NOT reimplemented here.** `prf`, `greedy_match`,
`percentile_ci` and `Span` are the very objects defined in `gold_standard.iaa`,
re-exported. The inter-annotator ceiling and the sealed-test score are both
computed with them; a model score is read against that ceiling, so a second
span-F1 anywhere in the repo would make the two non-comparable.
`tests/test_metrics.py` asserts the identity (`metrics.prf is iaa.prf`), so a
well-meant local copy fails the suite.

What this module adds on top of those kernels is what the approved proposal
names for the analysis (§3.6, "Setiap proporsi dilengkapi selang kepercayaan
Wilson ... selang kepercayaan bootstrap dan uji berpasangan McNemar"):

  * strict / partial span-F1 against a reference, micro and per tag, macro-F1
  * a confusion matrix and a span-level error taxonomy
  * Wilson score intervals for every proportion
  * a paired bootstrap on the F1 difference between two systems
  * an exact McNemar test on paired recall

Conventions, all inherited from `iaa` and all load-bearing:

  * P/R/F1 are **NaN, not 1.0**, when neither side produced a span.
  * The resampling unit is the **sentence** (or a cluster of sentences when
    `cluster_of` is given), never the span.
  * Matching is maximum-cardinality one-to-one.

"Reference" and "hypothesis" are used instead of "gold" and "prediction" because
the same functions score a model against gold, an ensemble against gold, and one
annotator against another.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import TAGS
from .iaa import (BOOTSTRAP_ITERS, BOOTSTRAP_SEED, Span, greedy_match,
                  percentile_ci, prf)

__all__ = [
    "Span", "prf", "greedy_match", "percentile_ci",
    "MATCH_MODES", "match_pairs", "contribs", "micro", "per_tag",
    "bootstrap_f1_ci", "paired_bootstrap_f1", "wilson_ci", "mcnemar_exact",
    "mcnemar_recall_strict", "confusion_matrix", "error_taxonomy",
]

TAG_CODES: list[str] = list(TAGS)

# The two rules the proposal reports ("strict/partial span-F1"), plus the two
# label-blind variants `iaa.span_level_iaa` already uses. `partial` is iaa's
# relaxed rule exactly: same label and character IoU >= 0.5.
MATCH_MODES: dict[str, dict] = {
    "strict": {"label_sensitive": True, "exact": True},
    "partial": {"label_sensitive": True, "exact": False},
    "boundary_only": {"label_sensitive": False, "exact": True},
    "partial_any_label": {"label_sensitive": False, "exact": False},
}

SpansBySid = Mapping[str, Sequence[Span]]


def _mode(mode: str) -> dict:
    if mode not in MATCH_MODES:
        raise ValueError(f"unknown match mode {mode!r}; expected one of {sorted(MATCH_MODES)}")
    return MATCH_MODES[mode]


# ------------------------------------------------------------------- matching
def match_pairs(a: Sequence[Span], b: Sequence[Span],
                eligible: Callable[[Span, Span], float | None]) -> list[tuple[int, int]]:
    """Index pairs of a maximum-cardinality one-to-one matching.

    `greedy_match` returns only the COUNT; the confusion matrix and the error
    taxonomy need to know WHICH spans were paired. This uses the same
    construction (unit cost per eligible edge, an infinitesimal score term to
    order ties) with a caller-supplied eligibility rule. For the two rules
    `greedy_match` supports, the cardinality is identical — pinned by a
    randomised test — so this is the same matcher, not a second opinion.
    """
    if not a or not b:
        return []
    eps = 1e-6
    cost = np.zeros((len(a), len(b)))
    ok = np.zeros((len(a), len(b)), dtype=bool)
    for i, sa in enumerate(a):
        for j, sb in enumerate(b):
            e = eligible(sa, sb)
            if e is not None:
                cost[i, j] = -(1.0 + eps * float(e))
                ok[i, j] = True
    ri, cj = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(ri, cj) if ok[i, j]]


def _same_bounds(sa: Span, sb: Span) -> bool:
    return sa.start == sb.start and sa.end == sb.end


# -------------------------------------------------------------------- scoring
def contribs(ref_by_sid: SpansBySid, hyp_by_sid: SpansBySid, sids: Iterable[str],
             mode: str = "strict", tag: str | None = None
             ) -> dict[str, tuple[int, int, int]]:
    """Per-sentence `(tp, n_ref, n_hyp)` for EVERY id in `sids`.

    `sids` is the scoring population and is taken literally: a sentence missing
    from either mapping counts as having no spans on that side. Passing only the
    sentences that happen to carry reference spans would silently delete every
    false positive on a reference-empty sentence — the defect recorded as F02 in
    `docs/experiments/split_review_findings.md`.
    """
    m = _mode(mode)
    out: dict[str, tuple[int, int, int]] = {}
    for sid in sids:
        r = list(ref_by_sid.get(sid, ()))
        h = list(hyp_by_sid.get(sid, ()))
        if tag is not None:
            r = [s for s in r if s.label == tag]
            h = [s for s in h if s.label == tag]
        out[sid] = (greedy_match(r, h, **m), len(r), len(h))
    return out


def micro(c: Mapping[str, tuple[int, int, int]]) -> dict:
    """Micro P/R/F1 over a `contribs` table (precision is w.r.t. the hypothesis)."""
    pr = prf(sum(v[0] for v in c.values()), sum(v[1] for v in c.values()),
             sum(v[2] for v in c.values()))
    return {"tp": pr["tp"], "n_ref": pr["n_a"], "n_hyp": pr["n_b"],
            "precision": pr["precision"], "recall": pr["recall"], "f1": pr["f1"],
            "n_sentences": len(c)}


def per_tag(ref_by_sid: SpansBySid, hyp_by_sid: SpansBySid, sids: Iterable[str],
            mode: str = "strict", tags: Sequence[str] | None = None) -> dict:
    """Per-tag P/R/F1 and their macro average.

    Macro-F1 is the mean over tags whose F1 is DEFINED. A tag that neither side
    used has F1 = NaN (nothing to agree on) and is left out of the mean rather
    than counted as 0 or as 1; `n_tags_defined` and `tags_undefined` say how
    many tags the average actually rests on, so a macro-F1 over two tags cannot
    pass for one over five.
    """
    sids = list(sids)
    tags = list(tags) if tags is not None else TAG_CODES
    by_tag = {t: micro(contribs(ref_by_sid, hyp_by_sid, sids, mode, tag=t)) for t in tags}
    defined = [by_tag[t]["f1"] for t in tags if by_tag[t]["f1"] == by_tag[t]["f1"]]
    return {
        "by_tag": by_tag,
        "macro_f1": float(np.mean(defined)) if defined else float("nan"),
        "n_tags_defined": len(defined),
        "tags_undefined": [t for t in tags if by_tag[t]["f1"] != by_tag[t]["f1"]],
        "mode": mode,
    }


# ------------------------------------------------------------- uncertainty
def _resample_units(sids: Sequence[str], cluster_of: Mapping[str, str] | None):
    """Units to resample: sentences, or clusters of sentences when given."""
    if cluster_of is None:
        return [[s] for s in sids]
    groups: dict[str, list[str]] = {}
    for s in sids:
        if s not in cluster_of:
            raise KeyError(f"sentence {s!r} has no entry in cluster_of; refusing to "
                           f"mix clustered and unclustered resampling")
        groups.setdefault(str(cluster_of[s]), []).append(s)
    return [groups[k] for k in sorted(groups)]


def bootstrap_f1_ci(c: Mapping[str, tuple[int, int, int]],
                    iters: int = BOOTSTRAP_ITERS, seed: int = BOOTSTRAP_SEED,
                    cluster_of: Mapping[str, str] | None = None
                    ) -> tuple[float, float, int]:
    """95% percentile CI for micro-F1, resampling sentences (or clusters).

    Sentences from one posting are not independent. Pass `cluster_of`
    (sentence id -> posting or leakage-group id) to resample whole clusters; the
    interval is then honest about that dependence instead of too narrow.
    """
    units = _resample_units(sorted(c), cluster_of)
    if len(units) < 2:
        return float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    n = len(units)
    vals = []
    for _ in range(iters):
        tp = nr = nh = 0
        for i in rng.integers(0, n, n):
            for sid in units[i]:
                t, r, h = c[sid]
                tp += t
                nr += r
                nh += h
        vals.append(prf(tp, nr, nh)["f1"])
    return percentile_ci(vals)


def paired_bootstrap_f1(c_a: Mapping[str, tuple[int, int, int]],
                        c_b: Mapping[str, tuple[int, int, int]],
                        iters: int = BOOTSTRAP_ITERS, seed: int = BOOTSTRAP_SEED,
                        cluster_of: Mapping[str, str] | None = None) -> dict:
    """Paired bootstrap on `F1(A) - F1(B)` over the SAME sentences.

    Both systems are scored on each resample of one shared draw, which is what
    makes the comparison paired: sentence difficulty cancels instead of adding
    noise. The two tables must cover exactly the same sentence ids — a
    comparison over different populations is refused, not approximated.

    `p_two_sided` is the bootstrap sign test, 2·min(P(Δ*≤0), P(Δ*≥0)) with
    add-one smoothing. It answers "could the sign of the difference be noise?";
    the CI is the primary statement.
    """
    if set(c_a) != set(c_b):
        only_a, only_b = sorted(set(c_a) - set(c_b)), sorted(set(c_b) - set(c_a))
        raise ValueError(
            f"paired comparison needs identical sentence sets; {len(only_a)} id(s) "
            f"only in A (e.g. {only_a[:3]}), {len(only_b)} only in B (e.g. {only_b[:3]})")
    f1_a, f1_b = micro(c_a)["f1"], micro(c_b)["f1"]
    units = _resample_units(sorted(c_a), cluster_of)
    out = {"f1_a": f1_a, "f1_b": f1_b, "delta": f1_a - f1_b,
           "n_units": len(units), "unit": "cluster" if cluster_of else "sentence",
           "iters": iters, "seed": seed}
    if len(units) < 2:
        out.update(ci95_low=float("nan"), ci95_high=float("nan"),
                   p_two_sided=float("nan"), n_valid_resamples=0)
        return out
    rng = np.random.default_rng(seed)
    n = len(units)
    deltas = []
    for _ in range(iters):
        a = [0, 0, 0]
        b = [0, 0, 0]
        for i in rng.integers(0, n, n):
            for sid in units[i]:
                for k in range(3):
                    a[k] += c_a[sid][k]
                    b[k] += c_b[sid][k]
        d = prf(*a)["f1"] - prf(*b)["f1"]
        if d == d:                                   # drop NaN resamples
            deltas.append(d)
    lo, hi, n_valid = percentile_ci(deltas)
    if n_valid:
        arr = np.asarray(deltas)
        p = 2.0 * min((np.sum(arr <= 0) + 1) / (n_valid + 1),
                      (np.sum(arr >= 0) + 1) / (n_valid + 1))
        p = float(min(1.0, p))
    else:
        p = float("nan")
    out.update(ci95_low=lo, ci95_high=hi, p_two_sided=p, n_valid_resamples=n_valid)
    return out


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n.

    Preferred over the normal approximation because it stays inside [0, 1] and
    behaves at k = 0 and k = n, which is where acceptance-precision and
    abstention rates on a small dev set will sit. n = 0 has no interval: (NaN, NaN).
    """
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if n == 0:
        return float("nan"), float("nan")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    from scipy.stats import norm
    z = float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    # At the edges the bound is EXACTLY the edge (algebraically centre == half at
    # k = 0), but floating point leaves ~5e-17, which would put the interval a
    # hair away from the observed proportion it must contain.
    if k == 0:
        lo = 0.0
    if k == n:
        hi = 1.0
    return lo, hi


def mcnemar_exact(b: int, c: int) -> dict:
    """Exact (binomial) McNemar test on the two discordant counts.

    b = items A got right and B got wrong; c = the reverse. Under H0 the
    discordant items split 50/50. The exact form is used because the discordant
    count on a 240-sentence test set can be small, where the chi-square
    approximation is unreliable. With no discordant items there is nothing to
    test and p = 1.0.
    """
    if b < 0 or c < 0:
        raise ValueError(f"discordant counts must be >= 0, got b={b}, c={c}")
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "p_value": 1.0}
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / (2 ** n)
    return {"b": b, "c": c, "n_discordant": n, "p_value": float(min(1.0, 2.0 * tail))}


def mcnemar_recall_strict(ref_by_sid: SpansBySid, hyp_a_by_sid: SpansBySid,
                          hyp_b_by_sid: SpansBySid, sids: Iterable[str]) -> dict:
    """McNemar on paired RECALL: which reference spans each system recovered.

    The paired item is the reference span, and "recovered" is the strict rule
    (identical boundaries and label), under which a hit is simply membership —
    no matching ambiguity. This tests recall only: a system that wins by
    emitting fewer false positives is invisible here, which is why the F1
    comparison is `paired_bootstrap_f1` and this test is its complement.
    """
    b = c = both = neither = 0
    for sid in sids:
        key_a = {(s.start, s.end, s.label) for s in hyp_a_by_sid.get(sid, ())}
        key_b = {(s.start, s.end, s.label) for s in hyp_b_by_sid.get(sid, ())}
        for s in {(r.start, r.end, r.label) for r in ref_by_sid.get(sid, ())}:
            in_a, in_b = s in key_a, s in key_b
            if in_a and in_b:
                both += 1
            elif in_a:
                b += 1
            elif in_b:
                c += 1
            else:
                neither += 1
    out = mcnemar_exact(b, c)
    out.update(both=both, neither=neither, n_reference_spans=b + c + both + neither,
               unit="reference span", rule="strict")
    return out


# -------------------------------------------------------- error analysis
MISSED = "MISSED"
SPURIOUS = "SPURIOUS"


def confusion_matrix(ref_by_sid: SpansBySid, hyp_by_sid: SpansBySid,
                     sids: Iterable[str], tags: Sequence[str] | None = None) -> dict:
    """Tag confusion over spans with IDENTICAL boundaries.

    `matrix[ref_tag][hyp_tag]` counts boundary-exact pairs; a reference span with
    no boundary-exact partner lands in `matrix[ref_tag]["MISSED"]`, and an
    unpaired hypothesis span in `matrix["SPURIOUS"][hyp_tag]`. Boundary errors
    therefore show up as one MISSED plus one SPURIOUS — `error_taxonomy`
    separates them; this table isolates the pure category question.
    """
    tags = list(tags) if tags is not None else TAG_CODES
    matrix = {r: {h: 0 for h in tags + [MISSED]} for r in tags}
    matrix[SPURIOUS] = {h: 0 for h in tags}
    off_schema = 0

    def _bounds_then_label(sa: Span, sb: Span):
        if not _same_bounds(sa, sb):
            return None
        return 1.0 if sa.label == sb.label else 0.5   # prefer same-label pairings

    for sid in sids:
        r = list(ref_by_sid.get(sid, ()))
        h = list(hyp_by_sid.get(sid, ()))
        pairs = match_pairs(r, h, _bounds_then_label)
        used_r, used_h = {i for i, _ in pairs}, {j for _, j in pairs}
        for i, j in pairs:
            if r[i].label in matrix and h[j].label in matrix[r[i].label]:
                matrix[r[i].label][h[j].label] += 1
            else:
                off_schema += 1
        for i, s in enumerate(r):
            if i not in used_r:
                if s.label in matrix:
                    matrix[s.label][MISSED] += 1
                else:
                    off_schema += 1
        for j, s in enumerate(h):
            if j not in used_h:
                if s.label in matrix[SPURIOUS]:
                    matrix[SPURIOUS][s.label] += 1
                else:
                    off_schema += 1
    return {"tags": tags, "matrix": matrix, "n_off_schema_spans": off_schema}


ERROR_CLASSES = ["correct", "wrong_category", "boundary_error",
                 "boundary_and_category", "missed", "over_extraction"]


def error_taxonomy(ref_by_sid: SpansBySid, hyp_by_sid: SpansBySid,
                   sids: Iterable[str]) -> dict:
    """Assign every reference and hypothesis span to exactly one error class.

    The proposal's taxonomy (§3.6) includes boundary error, missed entity, wrong
    category and over-extraction; those four are decidable from spans alone and
    are computed here. Its other classes — unsupported extraction, meaningless
    fragment, out-of-domain term — need the verifier log or a human judgement
    and are deliberately NOT guessed at.

    Spans are consumed in four passes, each a maximum one-to-one matching over
    what the previous pass left:

      1. identical boundaries, same label        -> correct
      2. identical boundaries, different label   -> wrong_category
      3. overlapping, same label                 -> boundary_error
      4. overlapping, different label            -> boundary_and_category

    Whatever reference spans remain are `missed`; remaining hypothesis spans are
    `over_extraction`. Counts are per pair for the four matched classes, so
    `correct + wrong_category + boundary_error + boundary_and_category + missed`
    equals the number of reference spans.
    """
    counts = {k: 0 for k in ERROR_CLASSES}
    n_ref = n_hyp = 0
    passes = [
        ("correct", lambda a, b: 1.0 if _same_bounds(a, b) and a.label == b.label else None),
        ("wrong_category", lambda a, b: 1.0 if _same_bounds(a, b) else None),
        ("boundary_error",
         lambda a, b: (a.iou(b) if a.label == b.label and a.iou(b) > 0 else None)),
        ("boundary_and_category", lambda a, b: (a.iou(b) if a.iou(b) > 0 else None)),
    ]
    for sid in sids:
        r = list(ref_by_sid.get(sid, ()))
        h = list(hyp_by_sid.get(sid, ()))
        n_ref += len(r)
        n_hyp += len(h)
        for name, rule in passes:
            pairs = match_pairs(r, h, rule)
            counts[name] += len(pairs)
            drop_r, drop_h = {i for i, _ in pairs}, {j for _, j in pairs}
            r = [s for i, s in enumerate(r) if i not in drop_r]
            h = [s for j, s in enumerate(h) if j not in drop_h]
        counts["missed"] += len(r)
        counts["over_extraction"] += len(h)
    return {"counts": counts, "n_reference_spans": n_ref, "n_hypothesis_spans": n_hyp,
            "not_computed": ["unsupported_extraction", "meaningless_fragment",
                             "out_of_domain_term"]}
