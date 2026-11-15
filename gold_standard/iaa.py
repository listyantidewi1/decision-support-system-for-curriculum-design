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
# Deterministic last-resort tie-break for token labelling: alphabetically first
# tag wins (negated so the max() comparison in _token_labels picks it), so the
# result never depends on the order an annotator entered two identical spans.
_NEG_TAG_ORDER = {t: -i for i, t in enumerate(sorted(TAG_SET))}

# Per-file load problems (unresolvable / incomplete / duplicated rows), keyed by
# file name. Surfaced in the JSON + markdown report so a span that could not be
# read is visible to the researcher instead of looking like an annotator who
# simply marked nothing.
LOAD_PROBLEMS: dict[str, list[dict]] = {}

# Landis & Koch (1977) bands, INCLUSIVE upper bounds: <=.20 slight, <=.40 fair,
# <=.60 moderate, <=.80 substantial, else almost perfect. (A strict `<` test put
# every exact boundary value — .20/.40/.60/.80 — one band too high.)
_LK_BANDS = [(0.20, "slight"), (0.40, "fair"), (0.60, "moderate"),
             (0.80, "substantial")]


def interpret(k: float) -> str:
    if k != k:                             # NaN
        return "undefined"
    if k < 0:
        return "worse than chance"
    for hi, label in _LK_BANDS:
        if k <= hi:
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
        has_5tag = str(first.get("has_5tag", "")).strip()
        # "Judged" includes a has_5tag-only answer: ignoring it dropped sentences
        # the JSON loader keeps, which is the very parity F6 set out to achieve.
        if not relevance and not spans and not has_5tag:
            continue                                 # unjudged filler row — skip
        out[str(sid)] = SentAnn(
            relevance=relevance,
            has_5tag=has_5tag,
            spans=spans,
        )
    return out


# Mirrors trim_span() in gold_standard/app.py, which cannot be imported here
# because that module imports streamlit. Keep the two in sync: a mismatch makes
# the same human judgement score differently via the app and via the workbook.
_TRIM_LEADING = " \t\r\n,;:*•·"
_TRIM_TRAILING = " \t\r\n,;:.*•·"


def _trim_offsets(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start] in _TRIM_LEADING:
        start += 1
    while end > start and text[end - 1] in _TRIM_TRAILING:
        end -= 1
    return start, end


def _norm_ws(s: str) -> str:
    return " ".join((s or "").split()).casefold()


def _find_candidates(text: str, needle: str) -> list[tuple[int, int]]:
    """Every match of `needle` in `text`, in position order.

    ONE permissive match set — whitespace-run- and case-insensitive — rather than
    a cascade of per-tier searches. The cascade counted `occurrence_no` against a
    different set at each tier, so occurrence 1 could silently bind to a later
    location than the one the annotator counted.

    Word-boundary lookarounds (applied only where the needle itself starts/ends
    with a word character, so '.NET' still matches) stop a short needle binding
    inside a longer word — matching 'API' inside 'APIs' produced wrong offsets
    that were still reported as successfully resolved."""
    toks = (needle or "").split()
    if not toks:
        return []
    core = r"\s+".join(re.escape(t) for t in toks)
    lead = r"(?<!\w)" if re.match(r"\w", needle) else ""
    trail = r"(?!\w)" if re.search(r"\w\Z", needle) else ""
    return [(m.start(), m.end())
            for m in re.finditer(lead + core + trail, text, re.IGNORECASE)]


def _resolve_span(text: str, txt: str, occ: int, start_char, end_char, lab: str):
    """Locate a typed span. Returns (Span|None, ok, reason).

    Offsets are trimmed with the UI's punctuation rule so a workbook span and the
    same span captured in the app agree exactly. The researcher-supplied
    start/end cells are a last resort and are ACCEPTED ONLY IF the text they
    point at actually equals the typed span_text — a bounds check alone let a
    stale or mistyped offset pair fabricate a span over unrelated characters."""
    cands = _find_candidates(text, txt)
    if cands:
        if occ < 1:
            occ = 1
        if occ <= len(cands):
            s, e = cands[occ - 1]
            s, e = _trim_offsets(text, s, e)
            if e > s:
                return Span(s, e, lab), True, "matched"
        else:
            return None, False, (f"occurrence_no={occ} but only {len(cands)} "
                                 f"occurrence(s) of this text exist")
    try:
        s, e = int(float(start_char)), int(float(end_char))
    except (TypeError, ValueError):
        pass
    else:
        if 0 <= s < e <= len(text) and _norm_ws(text[s:e]) == _norm_ws(txt):
            s, e = _trim_offsets(text, s, e)
            if e > s:
                return Span(s, e, lab), True, "researcher_offsets"
        return None, False, "start/end cells do not contain the typed span_text"
    return None, False, "text not found in sentence"


def _anns_from_anotasi_df(df: pd.DataFrame) -> tuple[dict[str, SentAnn], list[dict]]:
    """Parse a Form A-1 'Anotasi' sheet in ROW ORDER into judged-sentence anns.

    Rows are grouped into blocks by their SENTENCE marker so annotator-inserted
    overflow SPAN rows (blank row_type/sentence_id, as Excel 'Insert Row' leaves
    them) still attach to the current sentence.

    A row only opens a new sentence block if it is explicitly marked SENTENCE, or
    carries BOTH a sentence_id and sentence_text. Treating "any row with text in
    the sentence_text column" as a new sentence meant one stray autofilled or
    pasted cell opened a bogus block with an empty id, which then stole every
    following SPAN row from the real sentence.

    Returns (anns, problems); each problem is a dict with sentence_id, span_text
    and reason, so a dropped span is reported rather than silently lost."""
    df = df.fillna("")
    out: dict[str, SentAnn] = {}
    problems: list[dict] = []
    blocks: list[dict] = []
    cur = None

    def start_block(sid, stext, rel, tag, row_no):
        nonlocal cur
        cur = {"sid": sid, "text": stext, "spans": [], "relevance": rel,
               "has_5tag": tag, "row": row_no}
        blocks.append(cur)

    for row_no, (_, r) in enumerate(df.iterrows(), start=2):
        rtype = str(r.get("row_type", "")).strip().upper()
        sid = str(r.get("sentence_id", "")).strip()
        stext = str(r.get("sentence_text", ""))
        lab = str(r.get("label", "")).strip()
        txt = str(r.get("span_text", "")).strip()

        if rtype == "SENTENCE" or (rtype == "" and sid and stext.strip()):
            start_block(sid, stext, str(r.get("relevance", "")).strip(),
                        str(r.get("has_5tag", "")).strip(), row_no)
            continue
        if rtype != "SPAN" and not (txt or lab):
            continue                                  # blank spacer row
        if cur is None:
            problems.append({"sentence_id": "", "span_text": txt,
                             "reason": f"row {row_no}: span row before any SENTENCE row"})
            continue
        # A half-filled span row is a real annotator mistake, not a blank: report it.
        if not txt or lab not in TAGS:
            if txt or lab:
                problems.append({
                    "sentence_id": cur["sid"], "span_text": txt,
                    "reason": f"row {row_no}: incomplete span row "
                              f"(span_text={txt!r}, label={lab!r})"})
            continue
        try:
            occ = int(float(r.get("occurrence_no") or 1))
        except (TypeError, ValueError):
            occ = 1
        span, ok, reason = _resolve_span(cur["text"], txt, occ, r.get("start_char"),
                                         r.get("end_char"), lab)
        if ok:
            cur["spans"].append(span)
        else:
            problems.append({"sentence_id": cur["sid"], "span_text": txt,
                             "reason": f"row {row_no}: {reason}"})

    # Merge blocks that share a sentence_id instead of letting the later one
    # overwrite the earlier (which silently discarded the earlier block's spans).
    for b in blocks:
        if not b["sid"]:
            if b["spans"] or b["relevance"]:
                problems.append({"sentence_id": "", "span_text": "",
                                 "reason": f"row {b['row']}: sentence block with no "
                                           "sentence_id — dropped"})
            continue
        if not b["relevance"] and not b["spans"] and not b["has_5tag"]:
            continue                                  # unjudged — skip (JSON parity)
        prev = out.get(b["sid"])
        if prev is None:
            out[b["sid"]] = SentAnn(b["relevance"], b["has_5tag"], list(b["spans"]))
        else:
            merged = list(prev.spans)
            for s in b["spans"]:
                if s not in merged:
                    merged.append(s)
            out[b["sid"]] = SentAnn(prev.relevance or b["relevance"],
                                    prev.has_5tag or b["has_5tag"], merged)
            problems.append({"sentence_id": b["sid"], "span_text": "",
                             "reason": f"row {b['row']}: duplicate SENTENCE block for "
                                       "this sentence_id — blocks merged"})
    return out, problems


def load_xlsx(path: Path) -> dict[str, SentAnn]:
    """Form A-1 workbook: derive span offsets from span_text (+ occurrence_no),
    taking sentence_text from the SENTENCE row. Captures annotator-inserted
    overflow SPAN rows, and reports every span it could not resolve."""
    df = pd.read_excel(path, sheet_name="Anotasi", dtype=str)
    anns, problems = _anns_from_anotasi_df(df)
    LOAD_PROBLEMS[Path(path).name] = problems
    if problems:
        shown = "; ".join(f"{p['sentence_id']}:«{p['span_text']}» {p['reason']}"
                          for p in problems[:8])
        print(f"WARNING: load_xlsx {Path(path).name}: {len(problems)} span row(s) "
              f"not usable — {shown}" + (" …" if len(problems) > 8 else ""), flush=True)
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
        # Zero marginal variance (every rating in one category): kappa is 0/0,
        # i.e. UNDEFINED, not 1.0. Returning 1.0 here would publish a perfect
        # chance-corrected agreement for data that carries no information about
        # whether the raters can discriminate. Report observed agreement instead.
        return float("nan")
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
        # Single observed category -> expected disagreement 0 -> alpha undefined.
        return float("nan")
    return float(1 - (n - 1) * off / exp)


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Pairwise Cohen's kappa over paired labels. NaN where kappa is undefined.

    Deliberately does NOT short-circuit constant vectors to 1.0: with zero
    marginal variance kappa is undefined, and scikit-learn's own NaN is the
    correct answer. Overriding it to 1.0 reports perfect chance-corrected
    agreement for data that cannot demonstrate any."""
    import warnings

    from sklearn.metrics import cohen_kappa_score
    if not a or not b or len(a) != len(b):
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
    where each eligible edge costs -(1 + EPS*IoU) and ineligible pairs cost 0.

    The IoU term MUST be infinitesimal relative to the unit cardinality term. An
    earlier version used -(1+IoU), where IoU in [0.5,1] is the same order of
    magnitude as the count: four tight pairs (-2 each = -8) then beat five loose
    pairs (-1.5 each = -7.5), so the optimiser traded away a true positive and
    span agreement was under-reported. With EPS*IoU <= EPS*4096 << 1, cardinality
    strictly dominates and IoU only orders ties."""
    if not a or not b:
        return 0
    EPS = 1e-6
    cost = np.zeros((len(a), len(b)))
    eligible = np.zeros((len(a), len(b)), dtype=bool)
    for i, sa in enumerate(a):
        for j, sb in enumerate(b):
            e = _eligible(sa, sb, label_sensitive, exact)
            if e is not None:
                cost[i, j] = -(1.0 + EPS * e)
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
        # nanmean, not mean: a pair where BOTH annotators produced no span has an
        # undefined F1, and np.mean would propagate that NaN and erase the valid
        # F1s of the other pairs. n_pairs_defined makes the omission visible
        # instead of silently averaging over fewer pairs than the reader assumes.
        f1s = np.array([pw["f1"] for pw in pairwise.values()], dtype=float)
        defined = f1s[~np.isnan(f1s)]
        macro_f1 = float(defined.mean()) if defined.size else float("nan")
        result["modes"][mode] = {
            "pairwise": pairwise, "micro": micro, "macro_f1": macro_f1,
            "n_pairs_defined": int(defined.size), "n_pairs_total": int(f1s.size),
        }

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
    span is [0,1) — which a midpoint rule would silently drop.

    Assignment rule, stated precisely: the span with the greatest character
    overlap on that token wins; ties go to the NARROWER span, then the earlier
    start, then the alphabetically first tag. Note this is NOT a general
    "smallest span wins" rule — for nested OP spans the narrower span only wins
    where the overlap is equal, which is the nested case that matters (the inner
    span lies wholly inside the token); where a wider span covers strictly more
    of the token, it wins. The final tag tie-break keeps the result independent
    of the order the annotator happened to enter two spans at identical
    offsets."""
    toks = [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text or "")]
    labels = []
    for a, b in toks:
        best_key, best_label = None, "O"
        for s in spans:
            overlap = min(b, s.end) - max(a, s.start)
            if overlap <= 0:
                continue
            # maximise overlap, then narrower span, then earlier start, then tag
            key = (overlap, -(s.end - s.start), -s.start,
                   _NEG_TAG_ORDER.get(s.label, 0))
            if best_key is None or key > best_key:
                best_key, best_label = key, s.label
        labels.append(best_label)
    return labels


def token_level_iaa(annset: AnnSet, text_by_sid: dict[str, str]) -> dict:
    """Chance-corrected span agreement at token granularity.

    Inclusion requires an IN-SCHEMA relevance from every coder (not merely a
    non-empty cell), so the token level and the sentence level describe the same
    set of sentences; `_categorical_iaa` discards out-of-schema values the same
    way. A sentence carrying spans but no relevance is reported under
    `n_sentences_excluded_no_relevance` rather than silently vanishing while its
    spans still count toward span-level F1."""
    coders = list(annset)
    have_text = [s for s in text_by_sid]
    sids, excluded_no_rel = [], 0
    for s in have_text:
        recs = [annset[c].get(s) for c in coders]
        if all(r is not None and r.relevance in RELEVANCE for r in recs):
            sids.append(s)
        elif any(r is not None and r.spans for r in recs):
            excluded_no_rel += 1

    units, table_rows = [], []
    n_tokens, n_sentences_used = 0, 0
    for sid in sids:
        per_coder = {c: _token_labels(text_by_sid[sid], annset[c][sid].spans) for c in coders}
        length = len(next(iter(per_coder.values())))
        if length == 0 or any(len(v) != length for v in per_coder.values()):
            continue
        n_sentences_used += 1
        for t in range(length):
            vals = [per_coder[c][t] for c in coders]
            units.append(vals)
            row = [0] * len(TOKEN_LABELS)
            for v in vals:
                row[TOKEN_LABELS.index(v)] += 1
            table_rows.append(row)
            n_tokens += 1

    table = np.asarray(table_rows, dtype=float) if table_rows else np.zeros((0, len(TOKEN_LABELS)))
    # A corpus where NO annotator recorded any span makes every token 'O': the
    # marginal has zero variance, so kappa/alpha are undefined. fleiss_kappa and
    # krippendorff_alpha now return NaN there, but guard explicitly so the
    # degenerate case is reported as such rather than inferred from a NaN.
    # Distinguish "nobody tagged anything" from "no sentence text was available":
    # both yield zero usable tokens, but only the first is a statement about the
    # annotators.
    any_tag = (n_tokens > 0) and any(v != "O" for u in units for v in u)
    no_tokens_at_all = n_tokens == 0
    kappa = fleiss_kappa(table) if n_tokens else float("nan")
    alpha = krippendorff_alpha(units) if units else float("nan")
    out = {
        "n_sentences_used": n_sentences_used,
        "n_sentences_excluded_no_relevance": excluded_no_rel,
        "n_tokens": n_tokens,
        "no_spans_anywhere": (not any_tag) and not no_tokens_at_all,
        "no_tokens_available": no_tokens_at_all,
        "fleiss_kappa": kappa,
        "fleiss_interpretation": interpret(kappa),
        "krippendorff_alpha": alpha,
        "note": "Tokens are labelled by the covering span ('O' = untagged). Because "
                "untagged tokens dominate, these coefficients are chance-corrected "
                "against a marginal in which 'O' is the overwhelming category; read "
                "them together with the per-tag span-F1, not instead of it.",
    }
    if no_tokens_at_all:
        out["note"] += (" No sentence text was available for the annotated sentence "
                        "ids, so no tokens could be scored — this says nothing about "
                        "the annotators.")
    elif not any_tag:
        out["note"] += (" NO annotator recorded any span, so the token marginal has "
                        "zero variance and both coefficients are undefined.")
    return out


# --------------------------------------------------------------------- adjudication
def _cluster_leftovers(leftovers: dict[str, list[Span]]):
    """Group disputed spans for readable reconciliation.

    Clustering is PRESENTATION ONLY — gold spans are decided by exact
    (start,end,label) unanimity in build_adjudication, never by cluster shape, so
    a transitively-linked or nested cluster can no longer destroy or fabricate a
    gold span. Single-link at IoU>=0.5 just keeps related offsets on one row of
    the adjudication worksheet."""
    items = [(c, s) for c, spans in leftovers.items() for s in spans]
    clusters: list[list[tuple[str, Span]]] = []
    for c, s in sorted(items, key=lambda cs: (cs[1].start, cs[1].end, cs[0])):
        placed = False
        for cl in clusters:
            if any(s.iou(t) >= 0.5 for _cc, t in cl):
                cl.append((c, s)); placed = True; break
        if not placed:
            clusters.append([(c, s)])
    return clusters


def build_adjudication(annset: AnnSet, text_by_sid: dict[str, str]) -> dict:
    """Provisional gold + adjudication queue, faithful to AG-RPL-LIMA-v3.0.

    FULL_AGREEMENT ('unanimous', auto-accepted and kept out of the queue) requires
    ALL of: every coder gave an in-schema relevance and they match; every coder
    ANSWERED has_5tag and they match; every span is endorsed by every coder at
    identical (start, end, label); and no guideline conflict was detected.

    Gold spans are the exact (start,end,label) groups all coders endorsed. Spans
    outside such a group are queued as typed disputes — LABEL (coders disagree on
    the tag), BOUND (same tag, different boundaries), PRESENCE (not everyone
    marked it) — and no gold span is ever synthesised from boundaries nobody wrote.

    Provisional relevance/has_5tag follow a 2-of-3 majority. Guideline Bagian 3.2
    (NON_RELEVANT => has_5tag=TIDAK and no spans) is enforced, but any override it
    performs is RECORDED in `guideline_conflicts` and forces the item into the
    queue, so an illegal combination the annotators agreed on is surfaced to them
    rather than silently corrected."""
    coders = list(annset)
    all_sids = sorted({s for d in annset.values() for s in d} | set(text_by_sid))
    queue, gold = [], {}
    for sid in all_sids:
        anns = {c: annset[c].get(sid) for c in coders}
        # A record counts as present if it carries an in-schema relevance OR any
        # span. Keying solely on relevance dropped sentences where every coder
        # marked spans but left the relevance cell blank — they vanished from both
        # gold and the queue while their spans still drove span-level F1.
        present = {c: a for c, a in anns.items() if a and (a.relevance or a.spans)}
        if not present:
            continue
        # Validate against the schema, exactly as _categorical_iaa does, so an
        # out-of-schema string cannot masquerade as a judgement or win a majority.
        rel_votes = {c: a.relevance for c, a in present.items() if a.relevance in RELEVANCE}
        tag_votes = {c: a.has_5tag for c, a in present.items() if a.has_5tag in HAS_5TAG}
        off_schema = sorted(
            [f"{c}:relevance={a.relevance!r}" for c, a in present.items()
             if a.relevance and a.relevance not in RELEVANCE]
            + [f"{c}:has_5tag={a.has_5tag!r}" for c, a in present.items()
               if a.has_5tag and a.has_5tag not in HAS_5TAG])

        rel_counts = Counter(rel_votes.values())
        rel_top, rel_n = rel_counts.most_common(1)[0] if rel_counts else ("", 0)
        rel_unanimous = len(rel_counts) == 1 and len(rel_votes) == len(coders)
        rel_majority = rel_n >= 2

        tag_counts = Counter(tag_votes.values())
        tag_top, tag_n = tag_counts.most_common(1)[0] if tag_counts else ("", 0)
        # Unanimity requires every coder to have ANSWERED has_5tag. Counting
        # `present` here instead of the has_5tag respondents let a blank answer
        # pass as agreement, so an item only 2-of-3 answered was auto-accepted.
        tag_unanimous = len(tag_counts) == 1 and len(tag_votes) == len(coders)
        tag_majority = tag_n >= 2

        # --- spans ---------------------------------------------------------
        # Gold spans come from EXACT (start,end,label) groups endorsed by every
        # coder, computed independently of clustering. Deriving them from whole
        # overlap-clusters destroyed genuinely unanimous spans whenever one coder
        # added a nested/overlapping extra span (the guideline's own nested SH/TT
        # + OP pattern), because the cluster then held more than one offset.
        exact_groups: dict[tuple[int, int, str], set[str]] = {}
        for c, a in present.items():
            for s in a.spans:
                exact_groups.setdefault((s.start, s.end, s.label), set()).add(c)
        unanimous_keys = {k for k, cs in exact_groups.items() if len(cs) == len(coders)}
        gold_spans = [{"start": st, "end": en, "label": lab}
                      for (st, en, lab) in sorted(unanimous_keys)]

        # Everything not in a unanimous exact group is reconciled as a dispute,
        # grouped by overlap purely for readability.
        leftovers = {c: [s for s in a.spans
                         if (s.start, s.end, s.label) not in unanimous_keys]
                     for c, a in present.items()}
        span_disputes = []
        for cl in _cluster_leftovers(leftovers):
            labels = Counter(s.label for _c, s in cl)
            offsets = {(s.start, s.end) for _c, s in cl}
            coders_in = {c for c, _s in cl}
            dtype = ("LABEL" if len(labels) > 1
                     else "PRESENCE" if len(coders_in) < len(coders)
                     else "BOUND" if len(offsets) > 1 else "PRESENCE")
            span_disputes.append({
                "type": dtype,
                "coders": sorted(coders_in),
                "labels": dict(labels),
                "spans": [{"coder": c, "start": s.start, "end": s.end,
                           "label": s.label} for c, s in cl],
            })

        # --- provisional gold ----------------------------------------------
        gold_rel = rel_top if rel_majority else ""
        gold_tag = tag_top if tag_majority else ""
        # Guideline Bagian 3.2: NON_RELEVANT implies has_5tag=TIDAK and no spans.
        # Record the override rather than applying it silently — if the coders
        # actually agreed on NON_RELEVANT+YA, or left spans on a NON_RELEVANT
        # sentence, that is a guideline breach they must be told about.
        guideline_conflicts = []
        if gold_rel == "NON_RELEVANT":
            if gold_tag and gold_tag != "TIDAK":
                guideline_conflicts.append(
                    f"relevance=NON_RELEVANT but has_5tag vote was {gold_tag} "
                    "(Bagian 3.2 forces TIDAK)")
            gold_tag = "TIDAK"
            if gold_spans or span_disputes:
                guideline_conflicts.append(
                    "relevance=NON_RELEVANT but spans were marked "
                    "(Bagian 3.2: a non-relevant sentence carries no tag)")
                gold_spans = []
        if off_schema:
            guideline_conflicts.append("out-of-schema value(s): " + "; ".join(off_schema))

        fully_unanimous = (rel_unanimous and tag_unanimous and not span_disputes
                           and not guideline_conflicts)
        resolution = ("unanimous" if fully_unanimous
                      else "majority" if (rel_majority and tag_majority
                                          and not span_disputes
                                          and not guideline_conflicts) else "DISPUTE")
        gold[sid] = {"relevance": gold_rel, "has_5tag": gold_tag,
                     "spans": gold_spans, "resolution": resolution,
                     "guideline_conflicts": guideline_conflicts}

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
                                  else ("majority" if tag_majority else "none"),
            "n_span_disputes": len(span_disputes),
            "span_disputes": span_disputes,
            "guideline_conflicts": guideline_conflicts,
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
          f"Spans recorded per annotator (all saved sentences): "
          f"{sp['n_spans_per_annotator']}", "",
          "Only F1 is reported: this is a symmetric agreement comparison, so "
          "precision and recall merely depend on which annotator is arbitrarily "
          "called the reference and are not interpretable here. Matched-pair counts "
          "below are summed over the three annotator pairs and restricted to "
          "sentences each pair both judged, so they do not equal the per-annotator "
          "totals above.", "",
          "| Match rule | micro F1 | macro F1 | pairs used | matched / spans A / spans B |",
          "|---|--:|--:|--:|--:|"]
    for mode, m in sp["modes"].items():
        mi = m["micro"]
        pairs = f"{m.get('n_pairs_defined', '?')}/{m.get('n_pairs_total', '?')}"
        L.append(f"| {mode} | {_fmt(mi['f1'])} | {_fmt(m['macro_f1'])} | {pairs} | "
                 f"{mi['tp']} / {mi['n_a']} / {mi['n_b']} |")
    L += ["", "exact→overlap gap = boundary disagreement; overlap→boundary_only gap = "
          "label confusion.", "", "Per-tag overlap-F1:", ""]
    for tag, pr in sp["per_tag_overlap_f1"].items():
        L.append(f"- {tag} ({TAGS[tag]}): F1={_fmt(pr['f1'])} "
                 f"(matched={pr['tp']}, spans={pr['n_a']}/{pr['n_b']})")

    tl = report["token_level"]
    L += ["", "## 3. Token level (chance-corrected span agreement)", ""]
    if "skipped" in tl:
        L.append(f"_skipped: {tl['skipped']}_")
    else:
        L += [f"- tokens: {tl['n_tokens']} over {tl['n_sentences_used']} sentences",
              f"- Fleiss κ: {_fmt(tl['fleiss_kappa'])} ({tl['fleiss_interpretation']})",
              f"- Krippendorff α: {_fmt(tl['krippendorff_alpha'])}"]
        if tl.get("n_sentences_excluded_no_relevance"):
            L.append(f"- EXCLUDED: {tl['n_sentences_excluded_no_relevance']} sentence(s) "
                     "carry spans but no in-schema relevance from every coder "
                     "(their spans still count at span level — reconcile before reporting)")
        if tl.get("no_tokens_available"):
            L.append("- **NOT SCORED**: no sentence text was available for the "
                     "annotated sentence ids (is the sample parquet present and do "
                     "the ids match?) — this is not a statement about the annotators.")
        elif tl.get("no_spans_anywhere"):
            L.append("- **UNDEFINED**: no annotator recorded any span, so the token "
                     "marginal has zero variance and both coefficients are undefined.")
        L.append(f"- _{tl['note']}_")
    L += ["", "## 4. Adjudication", "",
          f"- sentences resolved: {adj['n_sentences']}",
          f"- queued for adjudication (not unanimous): {adj['n_disputed']}"]
    n_conf = sum(1 for g in adj["provisional_gold"].values() if g.get("guideline_conflicts"))
    if n_conf:
        L.append(f"- **{n_conf} sentence(s) carry a guideline conflict** "
                 "(e.g. NON_RELEVANT combined with has_5tag=YA or with spans) — see "
                 "the `guideline_conflicts` column")
    L += ["", "See the adjudication CSV + provisional-gold JSON beside this report."]

    total_problems = sum(len(v) for v in LOAD_PROBLEMS.values())
    if total_problems:
        L += ["", "## 5. Load problems (spans that could NOT be read)", "",
              f"**{total_problems} span row(s) were unusable and are absent from every "
              "statistic above.** Resolve these before reporting any figure — a span "
              "lost here is indistinguishable from an annotator who marked nothing.", ""]
        for fname, probs in LOAD_PROBLEMS.items():
            for p in probs[:20]:
                L.append(f"- `{fname}` {p['sentence_id']}: «{p['span_text']}» — {p['reason']}")
            if len(probs) > 20:
                L.append(f"- `{fname}`: … and {len(probs) - 20} more")
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
                                            "n_disputed": adj["n_disputed"]},
                   # the full queue, not just counts: a has_5tag-only or
                   # guideline-conflict dispute is otherwise recorded nowhere
                   "adjudication_queue": adj["queue"],
                   "load_problems": LOAD_PROBLEMS}
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
            # has_5tag columns: without them a sentence queued ONLY for a has_5tag
            # dispute reached the adjudicator with no visible reason for being there
            "has_5tag_agreement": q.get("has_5tag_agreement", ""),
            **{f"tag_{c}": q.get("has_5tag_by_coder", {}).get(c, "") for c in ANNOTATORS},
            "guideline_conflicts": "; ".join(q.get("guideline_conflicts", [])),
            "n_span_disputes": q["n_span_disputes"],
            "span_dispute_types": ",".join(
                sorted({d["type"] for d in q["span_disputes"]})),
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
