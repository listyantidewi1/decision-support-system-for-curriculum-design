"""Inter-annotator agreement (IAA) + adjudication for the gold standard (Paket B,
step B-5), faithful to guideline AG-RPL-LIMA-v3.0.

Three expert annotators (AV / AI / AD) judge each sampled sentence independently
(Bagian 10.1). This module reads their work — in any of the three delivery formats
the package emits — and reports agreement at three levels, then stages disagreements
for adjudication.

  1. Sentence level (nominal categories, 3 raters):
       * relevance  {RELEVANT, NON_RELEVANT, UNCERTAIN}
       * has_5tag   {YA, TIDAK}
     -> Fleiss' kappa (complete cases) + Krippendorff's alpha (handles the sentences
        an annotator left unjudged) + observed agreement + pairwise Cohen's kappa.

  2. Span level (SH/SS/KN/TT/WB with char offsets):
       * span-F1, pairwise and micro-averaged, under three match rules —
         exact (identical offsets+label), overlap (IoU>=0.5 + same label), and
         boundary-only (IoU>=0.5, label ignored). The exact->overlap gap is boundary
         disagreement; the overlap->boundary gap is label confusion.
       * per-tag span-F1 (which of the five tags are reliable).

  3. Token level (chance-corrected span agreement): each sentence is tokenized and
     every token gets the label of its covering span (or 'O'); tokens then become
     units with one label per annotator -> Fleiss / Krippendorff over {O,SH,SS,KN,TT,WB}.
     This is the chance-corrected counterpart to span-F1 (which has no chance baseline).

Adjudication: a provisional gold by majority (sentence) / unanimity (span), plus an
adjudication queue of every item that was not unanimous, laid out for the lead to
resolve. Nothing here finalizes gold — it prepares the adjudicator's worksheet.

Run:  .venv-ml\\Scripts\\python.exe -m gold_standard.iaa [--split pilot_100]
          [--format json|csv|xlsx] [--ann-dir DIR]
Reads   DATA/gold_standard/annotations/<split>_<AV|AI|AD>.json   (default; app format)
        or worksheets/<split>_<code>.xlsx  or a dir of Form A-1 CSVs.
Writes  DATA/gold_standard/iaa/iaa_<split>_<stamp>.{json,md}
        DATA/gold_standard/iaa/adjudication_<split>_<stamp>.csv
        DATA/gold_standard/iaa/provisional_gold_<split>_<stamp>.json   (all gitignored)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from . import ANNOTATORS, HAS_5TAG, RELEVANCE, TAGS

REPO_ROOT = Path(__file__).resolve().parents[1]
GS_DIR = REPO_ROOT / "DATA" / "gold_standard"
ANN_DIR = GS_DIR / "annotations"
WS_DIR = GS_DIR / "worksheets"
OUT_DIR = GS_DIR / "iaa"

TAG_SET = list(TAGS)                       # ["SH","SS","KN","TT","WB"]
TOKEN_LABELS = ["O"] + TAG_SET
_TOKEN_RE = re.compile(r"\S+")             # same tokenizer the annotator UI uses

# Landis & Koch (1977) bands for reporting kappa/alpha interpretability.
_LK_BANDS = [(0.0, "poor / none"), (0.2, "slight"), (0.4, "fair"),
             (0.6, "moderate"), (0.8, "substantial"), (1.01, "almost perfect")]


def interpret(k: float) -> str:
    if k != k:                             # NaN
        return "undefined"
    if k < 0:
        return "worse than chance"
    for hi, label in _LK_BANDS:
        if k < hi:
            return label
    return "almost perfect"


# --------------------------------------------------------------------------- data
@dataclass
class Span:
    start: int
    end: int
    label: str

    def iou(self, other: "Span") -> float:
        lo, hi = max(self.start, other.start), min(self.end, other.end)
        inter = max(0, hi - lo)
        union = (self.end - self.start) + (other.end - other.start) - inter
        return inter / union if union > 0 else 0.0


@dataclass
class SentAnn:
    relevance: str = ""
    has_5tag: str = ""
    spans: list[Span] = field(default_factory=list)


# annotator_code -> { sentence_id -> SentAnn }
AnnSet = dict[str, dict[str, SentAnn]]


def _spans_from_records(recs) -> list[Span]:
    out = []
    for s in recs:
        try:
            st, en = int(s["start_char"]), int(s["end_char"])
        except (TypeError, ValueError, KeyError):
            continue
        lab = (s.get("label") or "").strip()
        if lab in TAGS and en > st:
            out.append(Span(st, en, lab))
    return out


def load_json(path: Path) -> dict[str, SentAnn]:
    """App autosave format: {sentence_id: {relevance, has_5tag, spans:[...]}}."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, SentAnn] = {}
    for sid, rec in raw.items():
        if sid == "_meta" or not isinstance(rec, dict):
            continue
        out[sid] = SentAnn(
            relevance=(rec.get("relevance") or "").strip(),
            has_5tag=(rec.get("has_5tag") or "").strip(),
            spans=_spans_from_records(rec.get("spans", [])),
        )
    return out


def load_form_a1_csv(path: Path) -> dict[str, SentAnn]:
    """Form A-1 CSV (one row per span; a span-less sentence is one empty-span row).

    Emits ONLY judged sentences (relevance set or at least one span), so the sid
    set matches the app's JSON loader and span-F1 is identical across formats —
    the CSV/xlsx export writes a filler row for every sampled sentence, including
    ones an annotator never judged, and those must not be counted as agreement."""
    df = pd.read_csv(path, dtype=str).fillna("")
    out: dict[str, SentAnn] = {}
    for sid, g in df.groupby("sentence_id"):
        first = g.iloc[0]
        recs = [r for _, r in g.iterrows()
                if str(r.get("label", "")).strip() in TAGS
                and str(r.get("start_char", "")).strip() != ""]
        spans = _spans_from_records(recs)
        relevance = str(first.get("relevance", "")).strip()
        if not relevance and not spans:
            continue                                 # unjudged filler row — skip
        out[str(sid)] = SentAnn(
            relevance=relevance,
            has_5tag=str(first.get("has_5tag", "")).strip(),
            spans=spans,
        )
    return out


def _nth_occurrence(text: str, needle: str, n: int) -> tuple[int, int] | None:
    """Char offsets of the n-th (1-based) occurrence of `needle` in `text`."""
    if not needle:
        return None
    start = -1
    for _ in range(max(n, 1)):
        start = text.find(needle, start + 1)
        if start == -1:
            return None
    return start, start + len(needle)


def _flex_find(text: str, needle: str, occ: int, ignore_case: bool) -> tuple[int, int] | None:
    """Offsets of the occ-th (1-based) match of `needle` in `text`, tolerating
    differences in internal whitespace runs. Offsets index the ORIGINAL text."""
    if not needle:
        return None
    pat = r"\s+".join(re.escape(tok) for tok in needle.split())
    flags = re.IGNORECASE if ignore_case else 0
    matches = list(re.finditer(pat, text, flags))
    if len(matches) >= occ:
        m = matches[occ - 1]
        return m.start(), m.end()
    return None


def _resolve_span(text: str, txt: str, occ: int, start_char, end_char, lab: str):
    """Locate a typed span, returning (Span|None, resolved?). Tries, in order:
    exact nth-occurrence, whitespace-flexible, case-insensitive flexible, then the
    researcher-supplied start/end cells, before giving up (so a case/whitespace
    typo does not silently erase a span)."""
    off = _nth_occurrence(text, txt, occ)
    if not off:
        off = _flex_find(text, txt, occ, ignore_case=False)
    if not off:
        off = _flex_find(text, txt, occ, ignore_case=True)
    if off:
        return Span(off[0], off[1], lab), True
    try:                                             # researcher-supplied offsets
        s, e = int(float(start_char)), int(float(end_char))
        if 0 <= s < e <= len(text):
            return Span(s, e, lab), True
    except (TypeError, ValueError):
        pass
    return None, False


def _anns_from_anotasi_df(df: pd.DataFrame) -> tuple[dict[str, SentAnn], list[tuple[str, str]]]:
    """Parse a Form A-1 'Anotasi' sheet in ROW ORDER into judged-sentence anns.
    Rows are grouped into blocks by their SENTENCE marker so annotator-inserted
    overflow SPAN rows (blank row_type/sentence_id, as Excel 'Insert Row' leaves
    them) still attach to the current sentence. Returns (anns, unresolved) where
    unresolved lists (sentence_id, span_text) that could not be located."""
    df = df.fillna("")
    out: dict[str, SentAnn] = {}
    unresolved: list[tuple[str, str]] = []
    cur = None                                       # current block dict

    def flush(block):
        if block is None:
            return
        if not block["relevance"] and not block["spans"]:
            return                                   # unjudged — skip (JSON parity)
        out[block["sid"]] = SentAnn(block["relevance"], block["has_5tag"], block["spans"])

    for _, r in df.iterrows():
        rtype = str(r.get("row_type", "")).strip().upper()
        sid = str(r.get("sentence_id", "")).strip()
        stext = str(r.get("sentence_text", ""))
        is_sentence = rtype == "SENTENCE" or (rtype == "" and stext.strip() != "")
        if is_sentence:
            flush(cur)
            cur = {"sid": sid, "text": stext, "spans": [],
                   "relevance": str(r.get("relevance", "")).strip(),
                   "has_5tag": str(r.get("has_5tag", "")).strip()}
            continue
        lab = str(r.get("label", "")).strip()
        txt = str(r.get("span_text", "")).strip()
        is_span = rtype == "SPAN" or (rtype == "" and (txt or lab))
        if not is_span or cur is None or lab not in TAGS or not txt:
            continue
        try:
            occ = int(float(r.get("occurrence_no") or 1))
        except ValueError:
            occ = 1
        span, ok = _resolve_span(cur["text"], txt, occ, r.get("start_char"),
                                 r.get("end_char"), lab)
        if ok:
            cur["spans"].append(span)
        else:
            unresolved.append((cur["sid"], txt))
    flush(cur)
    return out, unresolved


def load_xlsx(path: Path) -> dict[str, SentAnn]:
    """Form A-1 workbook: derive span offsets from span_text (+ occurrence_no),
    taking sentence_text from the SENTENCE row. Captures annotator-inserted
    overflow SPAN rows, and prints a completeness warning for any span whose text
    could not be located (the README's promised 'validate completeness' step)."""
    df = pd.read_excel(path, sheet_name="Anotasi", dtype=str)
    anns, unresolved = _anns_from_anotasi_df(df)
    if unresolved:
        shown = ", ".join(f"{sid}:«{txt}»" for sid, txt in unresolved[:8])
        print(f"WARNING: load_xlsx {Path(path).name}: {len(unresolved)} span(s) "
              f"not locatable in sentence_text and dropped — {shown}"
              + (" …" if len(unresolved) > 8 else ""), flush=True)
    return anns


def load_annotator_set(split: str, fmt: str = "json",
                       ann_dir: Path | None = None) -> AnnSet:
    """Discover and load the AV/AI/AD files for `split` in the requested format."""
    loaders = {"json": (ann_dir or ANN_DIR, "{split}_{code}.json", load_json),
               "csv": (ann_dir or ANN_DIR, "{split}_{code}_form_a1.csv", load_form_a1_csv),
               "xlsx": (ann_dir or WS_DIR, "{split}_{code}.xlsx", load_xlsx)}
    base, pattern, loader = loaders[fmt]
    annset: AnnSet = {}
    for code in ANNOTATORS:
        p = base / pattern.format(split=split, code=code)
        if p.exists():
            annset[code] = loader(p)
    return annset


# ------------------------------------------------------------------ metric kernels
def fleiss_kappa(table: np.ndarray) -> float:
    """Fleiss' kappa. `table` is N_items x K_categories of rater counts; every row
    must sum to the same n (raters per item). Returns NaN if undefined."""
    table = np.asarray(table, dtype=float)
    if table.size == 0:
        return float("nan")
    n_i = table.sum(axis=1)
    n = n_i[0]
    if n < 2 or not np.allclose(n_i, n):
        return float("nan")
    N = table.shape[0]
    P_i = (np.square(table).sum(axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()
    p_j = table.sum(axis=0) / (N * n)
    P_e = np.square(p_j).sum()
    if np.isclose(P_e, 1.0):
        return 1.0 if np.isclose(P_bar, 1.0) else float("nan")
    return float((P_bar - P_e) / (1 - P_e))


def krippendorff_alpha(units: list[list[str]]) -> float:
    """Krippendorff's alpha for nominal data. `units` is a list of per-unit lists of
    observed category values (missing values simply omitted). Units with < 2 values
    are ignored. Returns NaN if undefined."""
    cats = sorted({v for u in units for v in u})
    if not cats:
        return float("nan")
    idx = {c: i for i, c in enumerate(cats)}
    K = len(cats)
    coincidence = np.zeros((K, K), dtype=float)
    for u in units:
        m = len(u)
        if m < 2:
            continue
        cnt = Counter(u)
        for c in cnt:
            for c2 in cnt:
                if c == c2:
                    pairs = cnt[c] * (cnt[c] - 1)
                else:
                    pairs = cnt[c] * cnt[c2]
                coincidence[idx[c], idx[c2]] += pairs / (m - 1)
    n_c = coincidence.sum(axis=1)
    n = n_c.sum()
    if n < 2:
        return float("nan")
    off = coincidence.sum() - np.trace(coincidence)          # sum_{c!=c'} o_cc'
    exp = float(np.sum(np.outer(n_c, n_c)) - np.sum(n_c * n_c))  # sum_{c!=c'} n_c n_c'
    if exp == 0:
        return 1.0 if off == 0 else float("nan")
    return float(1 - (n - 1) * off / exp)


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Pairwise Cohen's kappa over paired labels (same length, no missing)."""
    from sklearn.metrics import cohen_kappa_score
    if not a or len(set(a)) == len(set(b)) == 1 and set(a) == set(b):
        return 1.0 if a == b else float("nan")
    try:
        return float(cohen_kappa_score(a, b))
    except Exception:
        return float("nan")


# ----------------------------------------------------------------- sentence level
def _categorical_iaa(annset: AnnSet, field_name: str, categories: list[str]) -> dict:
    coders = list(annset)
    all_sids = sorted({s for d in annset.values() for s in d})

    def val(code, sid):
        a = annset[code].get(sid)
        v = getattr(a, field_name) if a else ""
        return v if v in categories else ""

    # Krippendorff over every sentence with >= 2 judgements.
    units = []
    for sid in all_sids:
        vals = [val(c, sid) for c in coders]
        vals = [v for v in vals if v]
        if len(vals) >= 2:
            units.append(vals)
    alpha = krippendorff_alpha(units)

    # Fleiss over complete cases (all coders judged the sentence).
    complete = [sid for sid in all_sids if all(val(c, sid) for c in coders)]
    table = np.zeros((len(complete), len(categories)), dtype=float)
    for r, sid in enumerate(complete):
        for c in coders:
            table[r, categories.index(val(c, sid))] += 1
    kappa = fleiss_kappa(table) if complete else float("nan")

    observed = float(np.mean([
        (table[r] * (table[r] - 1)).sum() / (len(coders) * (len(coders) - 1))
        for r in range(len(complete))
    ])) if complete else float("nan")

    pairwise = {}
    for a, b in combinations(coders, 2):
        paired = [(val(a, s), val(b, s)) for s in all_sids if val(a, s) and val(b, s)]
        if paired:
            xa, xb = zip(*paired)
            pairwise[f"{a}-{b}"] = {
                "n": len(paired),
                "percent_agreement": float(np.mean([x == y for x, y in paired])),
                "cohen_kappa": cohen_kappa(list(xa), list(xb)),
            }

    dist = {c: Counter() for c in coders}
    for c in coders:
        for sid in all_sids:
            v = val(c, sid)
            if v:
                dist[c][v] += 1

    return {
        "field": field_name,
        "n_sentences_any": len(units),
        "n_complete_cases": len(complete),
        "fleiss_kappa": kappa,
        "fleiss_interpretation": interpret(kappa),
        "krippendorff_alpha": alpha,
        "observed_agreement": observed,
        "pairwise": pairwise,
        "distribution": {c: dict(dist[c]) for c in coders},
    }


# --------------------------------------------------------------------- span level
def _eligible(sa: Span, sb: Span, label_sensitive: bool, exact: bool):
    """IoU score (1.0 for exact) if a->b is a legal match, else None."""
    if label_sensitive and sa.label != sb.label:
        return None
    if exact:
        return 1.0 if (sa.start == sb.start and sa.end == sb.end) else None
    iou = sa.iou(sb)
    return iou if iou >= 0.5 else None


def _greedy_match(a: list[Span], b: list[Span], label_sensitive: bool,
                  exact: bool) -> int:
    """Maximum-cardinality one-to-one match count between two span lists.

    TP is the size of a maximum bipartite matching over the eligible edges
    (exact offsets, or IoU>=0.5, and same label when label_sensitive) — NOT a
    highest-score-first greedy, which under-counts when an annotator has
    overlapping/nested spans. Solved with linear_sum_assignment on a cost matrix
    where each eligible edge costs -(1+IoU): the leading -1 maximises the number
    of eligible pairs chosen, and the +IoU only breaks ties toward tighter
    overlaps; ineligible pairs cost 0 so they are never preferred to an edge."""
    if not a or not b:
        return 0
    cost = np.zeros((len(a), len(b)))
    eligible = np.zeros((len(a), len(b)), dtype=bool)
    for i, sa in enumerate(a):
        for j, sb in enumerate(b):
            e = _eligible(sa, sb, label_sensitive, exact)
            if e is not None:
                cost[i, j] = -(1.0 + e)
                eligible[i, j] = True
    ri, cj = linear_sum_assignment(cost)
    return int(sum(1 for i, j in zip(ri, cj) if eligible[i, j]))


def _prf(tp: int, n_a: int, n_b: int) -> dict:
    """P/R/F1 (F1 symmetric, so reference/hypothesis order is irrelevant). When
    neither side produced a span there is nothing to agree on, so P/R/F1 are
    undefined (NaN -> rendered 'n/a') rather than a misleading 1.0."""
    if n_a == 0 and n_b == 0:
        nan = float("nan")
        return {"tp": 0, "n_a": 0, "n_b": 0, "precision": nan, "recall": nan, "f1": nan}
    prec = tp / n_b if n_b else 0.0
    rec = tp / n_a if n_a else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"tp": tp, "n_a": n_a, "n_b": n_b, "precision": prec, "recall": rec, "f1": f1}


def _span_pair(a_by_sid, b_by_sid, label_sensitive, exact) -> dict:
    """Micro-averaged P/R/F1 over all sentences both annotators share."""
    sids = set(a_by_sid) & set(b_by_sid)
    tp = na = nb = 0
    for sid in sids:
        a, b = a_by_sid[sid], b_by_sid[sid]
        tp += _greedy_match(a, b, label_sensitive, exact)
        na += len(a); nb += len(b)
    return _prf(tp, na, nb)


def _spans_by_sid(annset: AnnSet, code: str, only_label: str | None = None):
    out = {}
    for sid, a in annset[code].items():
        spans = a.spans if only_label is None else [s for s in a.spans if s.label == only_label]
        out[sid] = spans
    return out


def span_level_iaa(annset: AnnSet) -> dict:
    coders = list(annset)
    modes = {"exact": dict(label_sensitive=True, exact=True),
             "overlap_iou50": dict(label_sensitive=True, exact=False),
             "boundary_only_iou50": dict(label_sensitive=False, exact=False)}
    result = {"n_spans_per_annotator":
              {c: sum(len(a.spans) for a in annset[c].values()) for c in coders},
              "modes": {}}
    for mode, opts in modes.items():
        pairwise = {}
        agg_tp = agg_na = agg_nb = 0
        for a, b in combinations(coders, 2):
            pr = _span_pair(_spans_by_sid(annset, a), _spans_by_sid(annset, b), **opts)
            pairwise[f"{a}-{b}"] = pr
            agg_tp += pr["tp"]; agg_na += pr["n_a"]; agg_nb += pr["n_b"]
        micro = _prf(agg_tp, agg_na, agg_nb)
        macro_f1 = float(np.mean([pw["f1"] for pw in pairwise.values()])) if pairwise else float("nan")
        result["modes"][mode] = {"pairwise": pairwise, "micro": micro, "macro_f1": macro_f1}

    # per-tag overlap-F1 (micro across pairs)
    per_tag = {}
    for tag in TAG_SET:
        agg_tp = agg_na = agg_nb = 0
        for a, b in combinations(coders, 2):
            pr = _span_pair(_spans_by_sid(annset, a, tag), _spans_by_sid(annset, b, tag),
                            label_sensitive=True, exact=False)
            agg_tp += pr["tp"]; agg_na += pr["n_a"]; agg_nb += pr["n_b"]
        per_tag[tag] = _prf(agg_tp, agg_na, agg_nb)
    result["per_tag_overlap_f1"] = per_tag
    return result


# --------------------------------------------------------------------- token level
def _token_labels(text: str, spans: list[Span]) -> list[str]:
    """Label per token by maximum character overlap with a span (matching the
    annotator UI's any-overlap token coverage), else 'O'. This keeps a token that
    a span covers only partially — e.g. the 'C' in the token 'C,' when the stored
    span is [0,1) — which a midpoint rule would silently drop. Ties are broken
    toward the smallest (most specific) span, then the earliest start, preserving
    smallest-span-wins for overlapping OP spans."""
    toks = [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text or "")]
    labels = []
    for a, b in toks:
        best_key, best_label = None, "O"
        for s in spans:
            overlap = min(b, s.end) - max(a, s.start)
            if overlap <= 0:
                continue
            key = (overlap, -(s.end - s.start), -s.start)   # max overlap, then smallest, earliest
            if best_key is None or key > best_key:
                best_key, best_label = key, s.label
        labels.append(best_label)
    return labels


def token_level_iaa(annset: AnnSet, text_by_sid: dict[str, str]) -> dict:
    coders = list(annset)
    # Only sentences every coder judged (relevance set) AND we have text for.
    sids = [s for s in text_by_sid
            if all(annset[c].get(s) and annset[c][s].relevance for c in coders)]
    units, table_rows = [], []
    n_tokens = 0
    for sid in sids:
        per_coder = {c: _token_labels(text_by_sid[sid], annset[c][sid].spans) for c in coders}
        length = len(next(iter(per_coder.values())))
        if length == 0 or any(len(v) != length for v in per_coder.values()):
            continue
        for t in range(length):
            vals = [per_coder[c][t] for c in coders]
            units.append(vals)
            row = [0] * len(TOKEN_LABELS)
            for v in vals:
                row[TOKEN_LABELS.index(v)] += 1
            table_rows.append(row)
            n_tokens += 1
    table = np.asarray(table_rows, dtype=float) if table_rows else np.zeros((0, len(TOKEN_LABELS)))
    kappa = fleiss_kappa(table) if n_tokens else float("nan")
    alpha = krippendorff_alpha(units) if units else float("nan")
    # agreement excluding all-'O' tokens (the easy majority) — the honest view.
    nz_units = [u for u in units if any(v != "O" for v in u)]
    alpha_nz = krippendorff_alpha(nz_units) if nz_units else float("nan")
    return {
        "n_sentences": len(sids),
        "n_tokens": n_tokens,
        "fleiss_kappa": kappa,
        "fleiss_interpretation": interpret(kappa),
        "krippendorff_alpha": alpha,
        "krippendorff_alpha_excl_all_O": alpha_nz,
        "note": "tokens labeled by covering span; 'O' = no tag. alpha_excl_all_O drops "
                "tokens every annotator left untagged (removes the trivial-O inflation).",
    }


# --------------------------------------------------------------------- adjudication
def _cluster_spans(annset: AnnSet, sid: str):
    """Group overlapping spans (any coder) into clusters for reconciliation.
    Single-link by IoU>=0.5: clustering only GROUPS candidates that might refer to
    the same entity; whether a cluster is genuine agreement is decided separately
    (build_adjudication requires exact boundary+label identity), so a transitively
    linked cluster is never mistaken for agreement."""
    items = [(c, s) for c in annset for s in (annset[c].get(sid).spans
             if annset[c].get(sid) else [])]
    clusters: list[list[tuple[str, Span]]] = []
    for c, s in sorted(items, key=lambda cs: (cs[1].start, cs[1].end)):
        placed = False
        for cl in clusters:
            if any(s.iou(t) >= 0.5 for _cc, t in cl):
                cl.append((c, s)); placed = True; break
        if not placed:
            clusters.append([(c, s)])
    return clusters


def build_adjudication(annset: AnnSet, text_by_sid: dict[str, str]) -> dict:
    """Provisional gold + adjudication queue, faithful to AG-RPL-LIMA-v3.0:
      * FULL_AGREEMENT ('unanimous') requires ALL coders to agree on relevance,
        has_5tag, AND — for every span cluster — the SAME label at IDENTICAL char
        boundaries. Only then is the item auto-accepted and kept out of the queue.
      * Anything less is queued. A same-label cluster with differing offsets is a
        BOUND(ary) dispute; differing labels a LABEL dispute; a span not marked by
        all coders a PRESENCE dispute — a no gold span is ever synthesized wider
        than what every coder marked.
      * Provisional gold relevance/has_5tag follow a 2-of-3 majority, and the
        guideline's hard rule NON_RELEVANT => has_5tag=TIDAK (Bagian 3.2) is
        enforced over any raw vote."""
    coders = list(annset)
    all_sids = sorted({s for d in annset.values() for s in d} | set(text_by_sid))
    queue, gold = [], {}
    for sid in all_sids:
        anns = {c: annset[c].get(sid) for c in coders}
        judged = {c: a for c, a in anns.items() if a and a.relevance}
        if not judged:
            continue
        rel_counts = Counter(a.relevance for a in judged.values())
        rel_top, rel_n = rel_counts.most_common(1)[0]
        rel_unanimous = len(rel_counts) == 1 and len(judged) == len(coders)
        rel_majority = rel_n >= 2

        tag_counts = Counter(a.has_5tag for a in judged.values() if a.has_5tag)
        tag_unanimous = len(tag_counts) == 1 and len(judged) == len(coders)
        tag_top = tag_counts.most_common(1)[0] if tag_counts else ("", 0)

        # span clusters -> exact-agreement gold spans vs typed disputes
        span_disputes, gold_spans = [], []
        for cl in _cluster_spans(annset, sid):
            labels = Counter(s.label for _c, s in cl)
            offsets = {(s.start, s.end) for _c, s in cl}
            coders_in = {c for c, _s in cl}
            full = (len(coders_in) == len(coders) and len(labels) == 1
                    and len(offsets) == 1)
            if full:
                (st, en), = offsets
                gold_spans.append({"start": st, "end": en,
                                   "label": next(iter(labels))})
            else:
                dtype = ("LABEL" if len(labels) > 1
                         else "PRESENCE" if len(coders_in) < len(coders)
                         else "BOUND")
                span_disputes.append({
                    "type": dtype,
                    "coders": sorted(coders_in),
                    "labels": dict(labels),
                    "spans": [{"coder": c, "start": s.start, "end": s.end,
                               "label": s.label} for c, s in cl],
                })

        fully_unanimous = rel_unanimous and tag_unanimous and not span_disputes

        # provisional gold (NON_RELEVANT forces has_5tag=TIDAK per Bagian 3.2)
        gold_rel = rel_top if rel_majority else ""
        if gold_rel == "NON_RELEVANT":
            gold_tag = "TIDAK"
        elif tag_top[1] >= 2:
            gold_tag = tag_top[0]
        else:
            gold_tag = ""
        resolution = ("unanimous" if fully_unanimous
                      else "majority" if (rel_majority and not span_disputes
                                          and tag_top[1] >= 2) else "DISPUTE")
        gold[sid] = {"relevance": gold_rel, "has_5tag": gold_tag,
                     "spans": gold_spans, "resolution": resolution}

        if fully_unanimous:
            continue
        queue.append({
            "sentence_id": sid,
            "sentence_text": text_by_sid.get(sid, ""),
            "relevance_by_coder": {c: (a.relevance if a else "") for c, a in anns.items()},
            "relevance_agreement": "unanimous" if rel_unanimous
                                   else ("majority" if rel_majority else "none"),
            "has_5tag_by_coder": {c: (a.has_5tag if a else "") for c, a in anns.items()},
            "has_5tag_agreement": "unanimous" if tag_unanimous
                                  else ("majority" if tag_top[1] >= 2 else "none"),
            "n_span_disputes": len(span_disputes),
            "span_disputes": span_disputes,
        })
    return {"queue": queue, "provisional_gold": gold,
            "n_sentences": len(gold), "n_disputed": len(queue)}


# --------------------------------------------------------------------------- orchestrate
def _text_by_sid(split: str) -> dict[str, str]:
    p = GS_DIR / f"{split}.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p, columns=["sentence_id", "sentence_text"])
    return dict(zip(df["sentence_id"].astype(str), df["sentence_text"].astype(str)))


def compute_iaa(annset: AnnSet, text_by_sid: dict[str, str]) -> dict:
    return {
        "annotators": list(annset),
        "sentence_level": {
            "relevance": _categorical_iaa(annset, "relevance", RELEVANCE),
            "has_5tag": _categorical_iaa(annset, "has_5tag", HAS_5TAG),
        },
        "span_level": span_level_iaa(annset),
        "token_level": (token_level_iaa(annset, text_by_sid) if text_by_sid
                        else {"skipped": "no sentence text (sample parquet) available"}),
    }


def _fmt(x) -> str:
    return "n/a" if (x is None or (isinstance(x, float) and x != x)) else f"{x:.3f}"


def to_markdown(report: dict, adj: dict, split: str, stamp: str) -> str:
    L = [f"# IAA — {split} ({', '.join(report['annotators'])}) — {stamp}", "",
         "Guideline AG-RPL-LIMA-v3.0 · Fleiss κ (complete cases) + Krippendorff α "
         "(all ≥2-rater items). Bands: Landis & Koch 1977.", "",
         "## 1. Sentence level", "",
         "| Field | complete | Fleiss κ | (band) | Krippendorff α | observed |",
         "|---|--:|--:|---|--:|--:|"]
    for f in ("relevance", "has_5tag"):
        s = report["sentence_level"][f]
        L.append(f"| {f} | {s['n_complete_cases']} | {_fmt(s['fleiss_kappa'])} | "
                 f"{s['fleiss_interpretation']} | {_fmt(s['krippendorff_alpha'])} | "
                 f"{_fmt(s['observed_agreement'])} |")
    L += ["", "Pairwise (Cohen κ / % agree):"]
    for f in ("relevance", "has_5tag"):
        for pair, pw in report["sentence_level"][f]["pairwise"].items():
            L.append(f"- {f} {pair}: κ={_fmt(pw['cohen_kappa'])}, "
                     f"{pw['percent_agreement']*100:.0f}% (n={pw['n']})")

    sp = report["span_level"]
    L += ["", "## 2. Span level (F1)", "",
          f"Spans per annotator: {sp['n_spans_per_annotator']}", "",
          "| Match rule | micro P | micro R | micro F1 | macro F1 |", "|---|--:|--:|--:|--:|"]
    for mode, m in sp["modes"].items():
        mi = m["micro"]
        L.append(f"| {mode} | {_fmt(mi['precision'])} | {_fmt(mi['recall'])} | "
                 f"{_fmt(mi['f1'])} | {_fmt(m['macro_f1'])} |")
    L += ["", "exact→overlap gap = boundary disagreement; overlap→boundary_only gap = "
          "label confusion.", "", "Per-tag overlap-F1:", ""]
    for tag, pr in sp["per_tag_overlap_f1"].items():
        L.append(f"- {tag} ({TAGS[tag]}): F1={_fmt(pr['f1'])} "
                 f"(tp={pr['tp']}, refs={pr['n_a']}, hyps={pr['n_b']})")

    tl = report["token_level"]
    L += ["", "## 3. Token level (chance-corrected span agreement)", ""]
    if "skipped" in tl:
        L.append(f"_skipped: {tl['skipped']}_")
    else:
        L += [f"- tokens: {tl['n_tokens']} over {tl['n_sentences']} sentences",
              f"- Fleiss κ: {_fmt(tl['fleiss_kappa'])} ({tl['fleiss_interpretation']})",
              f"- Krippendorff α: {_fmt(tl['krippendorff_alpha'])}",
              f"- Krippendorff α (excl. all-O tokens): {_fmt(tl['krippendorff_alpha_excl_all_O'])}"]
    L += ["", "## 4. Adjudication", "",
          f"- sentences resolved: {adj['n_sentences']}",
          f"- queued for adjudication (not unanimous): {adj['n_disputed']}",
          "", "See the adjudication CSV + provisional-gold JSON beside this report."]
    return "\n".join(L)


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="pilot_100")
    ap.add_argument("--format", choices=["json", "csv", "xlsx"], default="json")
    ap.add_argument("--ann-dir", default=None)
    args = ap.parse_args()

    ann_dir = Path(args.ann_dir) if args.ann_dir else None
    annset = load_annotator_set(args.split, args.format, ann_dir)
    if len(annset) < 2:
        raise SystemExit(
            f"Need >=2 annotator files for {args.split} ({args.format}); found "
            f"{sorted(annset)}. Annotators AV/AI/AD must produce their files first "
            f"(app: DATA/gold_standard/annotations/{args.split}_<code>.json).")
    if len(annset) < len(ANNOTATORS):
        print(f"WARNING: only {sorted(annset)} present; IAA computed on this subset.",
              flush=True)

    text_by_sid = _text_by_sid(args.split)
    report = compute_iaa(annset, text_by_sid)
    adj = build_adjudication(annset, text_by_sid)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_full = {"created_utc": stamp, "split": args.split, "format": args.format,
                   "guideline": "AG-RPL-LIMA-v3.0", **report,
                   "adjudication_summary": {"n_sentences": adj["n_sentences"],
                                            "n_disputed": adj["n_disputed"]}}
    (OUT_DIR / f"iaa_{args.split}_{stamp}.json").write_text(
        json.dumps(report_full, indent=2, ensure_ascii=False), encoding="utf-8")
    md = to_markdown(report, adj, args.split, stamp)
    (OUT_DIR / f"iaa_{args.split}_{stamp}.md").write_text(md, encoding="utf-8")

    # adjudication worksheet (flat CSV) + provisional gold
    qrows = []
    for q in adj["queue"]:
        qrows.append({
            "sentence_id": q["sentence_id"],
            "relevance_agreement": q["relevance_agreement"],
            **{f"rel_{c}": q["relevance_by_coder"].get(c, "") for c in ANNOTATORS},
            "n_span_disputes": q["n_span_disputes"],
            "sentence_text": q["sentence_text"],
            "span_disputes": json.dumps(q["span_disputes"], ensure_ascii=False),
        })
    pd.DataFrame(qrows).to_csv(
        OUT_DIR / f"adjudication_{args.split}_{stamp}.csv", index=False)
    (OUT_DIR / f"provisional_gold_{args.split}_{stamp}.json").write_text(
        json.dumps(adj["provisional_gold"], indent=2, ensure_ascii=False), encoding="utf-8")

    print(md, flush=True)
    print(f"\n-> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
