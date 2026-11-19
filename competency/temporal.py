"""Time analysis: share per window, trend, volatility, persistence (proposal §3.3.4).

For any set of items (technologies, knowledge terms, competency groups) and the
postings that mention each, this computes, over rolling time windows:

  share        postings in the window that mention the item / postings in the window
  trend        slope of the share across windows, with a p-value, corrected for
               the number of items tested (Benjamini-Hochberg), and the
               Durbin-Watson statistic of the residuals
  volatility   how much the share varies (coefficient of variation) and how often
               the item enters or leaves the top-N list; plus, for the list as a
               whole, the turnover of the top-N between consecutive windows
  persistence  one weight per item from three ingredients the proposal names:
               current prevalence, the corpus trend, and the external prior of
               the Future of Jobs Report (`future_domains.csv`)
  abstraction  a highly volatile item is recommended at its parent node in the
               ESCO-anchored taxonomy, with the specific item kept as an example

What these numbers may and may not do (proposal, same section): the persistence
weight and the volatility class change only the ORDER of recommendations that
already share a status, and the level of abstraction at which one is phrased.
They never raise a status.

Adequacy. The analysis needs dated postings and enough of them per window. A
window with fewer postings than `min_postings_per_window` is left out and
reported. With fewer than `min_windows_for_trend` windows the result is marked
EXPLORATORY, and with three or four windows a trend test has almost no power:
"not detected" then means "cannot tell", not "stable".

A share is a share of the postings COLLECTED for a window. The mix of platforms
and countries differs between windows, so a change in share can be a change in
what was collected. Use `only_stratum` to hold the stratum fixed, and read
trends across strata with that in mind.

Every threshold and the weight formula in `TemporalConfig` is a STARTING VALUE.
The proposal requires them to be fixed in development, validated by experts and
written into the configuration registry.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "DATA" / "competency" / "temporal"
GS_DIR = REPO_ROOT / "DATA" / "gold_standard"
FUTURE_DOMAINS = REPO_ROOT / "future_domains.csv"

NOTICE = ("ENGINEERING RUN: counts come from an extractor whose accuracy has not been measured, "
          "and the windows are few. Not a finding.")


@dataclass(frozen=True)
class TemporalConfig:
    window_months: int = 6
    step_months: int = 6                  # smaller than window_months = overlapping windows
    min_postings_per_window: int = 200
    min_windows_for_trend: int = 3
    min_item_postings: int = 20           # rarer items are not analysed
    top_n: int = 50
    fdr_alpha: float = 0.05
    volatility_cv_medium: float = 0.25
    volatility_cv_high: float = 0.50
    weight_prevalence: float = 0.50
    weight_trend: float = 0.25
    weight_prior: float = 0.25
    thresholds_status: str = "starting values, not yet validated by experts"
    version: str = "temporal-v1"


def month_index(date_text) -> int | None:
    """'2026-03-14...' -> 2026*12 + 2. None when the date cannot be read."""
    s = str(date_text or "")
    if len(s) < 7 or not (s[:4].isdigit() and s[5:7].isdigit() and s[4] == "-"):
        return None
    y, m = int(s[:4]), int(s[5:7])
    return y * 12 + (m - 1) if 1 <= m <= 12 and 1990 <= y <= 2100 else None


def _label(m: int) -> str:
    return f"{m // 12:04d}-{m % 12 + 1:02d}"


def make_windows(posting_month: Mapping[str, int], cfg: TemporalConfig) -> list[dict]:
    """Calendar-aligned windows covering every dated posting."""
    months = list(posting_month.values())
    if not months:
        return []
    start = (min(months) // cfg.step_months) * cfg.step_months
    out, k = [], 0
    while start + k * cfg.step_months <= max(months):
        a = start + k * cfg.step_months
        b = a + cfg.window_months
        postings = {p for p, m in posting_month.items() if a <= m < b}
        out.append({"start": _label(a), "end": _label(b - 1), "_a": a, "_b": b,
                    "n_postings": len(postings), "_postings": postings,
                    "included": len(postings) >= cfg.min_postings_per_window})
        k += 1
    return out


def _linregress(y: Sequence[float]) -> tuple[float, float | None, float | None]:
    """(slope per window, two-sided p-value, Durbin-Watson). p is None with
    fewer than 3 points; DW is None with fewer than 4."""
    n = len(y)
    if n < 2:
        return 0.0, None, None
    x = list(range(n))
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sxx
    resid = [b - (my + slope * (a - mx)) for a, b in zip(x, y)]
    p = None
    if n >= 3:
        sse = sum(r * r for r in resid)
        if sse <= 1e-18:
            p = 0.0 if abs(slope) > 0 else 1.0
        else:
            from scipy import stats
            t = slope / math.sqrt(sse / (n - 2) / sxx)
            p = float(2 * stats.t.sf(abs(t), n - 2))
    dw = None
    if n >= 4 and sum(r * r for r in resid) > 1e-18:
        dw = sum((resid[i] - resid[i - 1]) ** 2 for i in range(1, n)) / sum(r * r for r in resid)
    return slope, p, dw


def benjamini_hochberg(pvalues: Sequence[float | None]) -> list[float | None]:
    """q-values; None stays None."""
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    m = len(idx)
    out: list[float | None] = [None] * len(pvalues)
    prev = 1.0
    for rank, i in zip(range(m, 0, -1), sorted(idx, key=lambda i: pvalues[i], reverse=True)):
        prev = min(prev, pvalues[i] * m / rank)
        out[i] = prev
    return out


def _percentile_rank(values: Sequence[float]) -> list[float]:
    """0..1; ties share the mean rank."""
    n = len(values)
    if n <= 1:
        return [0.5] * n
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        r = (i + j) / 2 / (n - 1)
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def load_prior(path: Path = FUTURE_DOMAINS) -> dict[str, tuple[float, str]]:
    """term (lower case) -> (prior 0..1, domain). The prior is the domain's
    trend score divided by the largest score in the file."""
    if not Path(path).is_file():
        return {}
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    scores = []
    for r in rows:
        try:
            scores.append(float(r.get("trend_score") or 0))
        except ValueError:
            scores.append(0.0)
    top = max(scores) if scores and max(scores) > 0 else 1.0
    out: dict[str, tuple[float, str]] = {}
    for r, s in zip(rows, scores):
        for term in (r.get("example_terms") or "").split(","):
            t = term.strip().lower()
            if t and (t not in out or s / top > out[t][0]):
                out[t] = (s / top, r.get("future_domain") or "")
    return out


def analyse(items: Mapping[str, set], posting_month: Mapping[str, int],
            cfg: TemporalConfig | None = None, *, labels: Mapping[str, str] | None = None,
            prior: Mapping[str, tuple[float, str]] | None = None,
            parent: Mapping[str, tuple[str, str]] | None = None) -> dict:
    """`items[item_id]` = the postings that mention it. `labels` gives a display
    text; `prior` and `parent` are looked up by the lower-cased label."""
    cfg, labels, prior, parent = cfg or TemporalConfig(), labels or {}, prior or {}, parent or {}
    windows = make_windows(posting_month, cfg)
    used = [w for w in windows if w["included"]]
    enough = len(used) >= cfg.min_windows_for_trend
    rows = []
    for item, postings in items.items():
        dated = {p for p in postings if p in posting_month}
        if len(dated) < cfg.min_item_postings:
            continue
        shares = [len(dated & w["_postings"]) / w["n_postings"] for w in used]
        slope, p, dw = _linregress(shares) if len(used) >= 2 else (0.0, None, None)
        mean = sum(shares) / len(shares) if shares else 0.0
        sd = math.sqrt(sum((s - mean) ** 2 for s in shares) / len(shares)) if shares else 0.0
        rows.append({"item": item, "label": labels.get(item, item), "n_postings": len(dated),
                     "shares": [round(s, 5) for s in shares],
                     "current_share": shares[-1] if shares else 0.0,
                     "slope_per_window": slope,
                     "relative_slope": slope / mean if mean > 0 else 0.0,
                     "p_value": p if enough else None, "durbin_watson": dw,
                     "cv": sd / mean if mean > 0 else 0.0})
    for r, q in zip(rows, benjamini_hochberg([r["p_value"] for r in rows])):
        r["q_value"] = q
        if q is None:
            r["trend"] = "tidak_cukup_data"
        elif q < cfg.fdr_alpha:
            r["trend"] = "menaik" if r["slope_per_window"] > 0 else "menyurut"
        else:
            r["trend"] = "tidak_terdeteksi"
        r["direction"] = ("naik" if r["relative_slope"] > 0.05 else
                          "turun" if r["relative_slope"] < -0.05 else "datar")
    # top-N membership per window
    tops = []
    for k in range(len(used)):
        ranked = sorted(rows, key=lambda r: (-r["shares"][k], r["item"]))[:cfg.top_n]
        tops.append({r["item"] for r in ranked})
    for r in rows:
        member = [r["item"] in t for t in tops]
        r["top_n_changes"] = sum(1 for a, b in zip(member, member[1:]) if a != b)
        r["volatility"] = ("tinggi" if r["cv"] >= cfg.volatility_cv_high else
                           "sedang" if r["cv"] >= cfg.volatility_cv_medium else "rendah")
    prev_rank = _percentile_rank([r["current_share"] for r in rows])
    trend_rank = _percentile_rank([r["relative_slope"] for r in rows])
    for r, a, b in zip(rows, prev_rank, trend_rank):
        key = r["label"].lower()
        hit = prior.get(key)
        r["prior"] = hit[0] if hit else 0.5
        r["prior_domain"] = hit[1] if hit else ""
        r["prior_matched"] = bool(hit)
        r["persistence_weight"] = (cfg.weight_prevalence * a + cfg.weight_trend * b
                                   + cfg.weight_prior * r["prior"])
        node = parent.get(key)
        if r["volatility"] == "tinggi" and node:
            r["recommend_at"] = {"level": "parent", "node": node[1], "code": node[0],
                                 "example": r["label"]}
        else:
            r["recommend_at"] = {"level": "specific", "node": r["label"],
                                 "anchored": bool(node)}
    rows.sort(key=lambda r: (-r["persistence_weight"], r["item"]))
    turnover = [{"from": used[k]["start"], "to": used[k + 1]["start"],
                 "top_n_turnover": 1 - len(tops[k] & tops[k + 1]) / max(1, len(tops[k] | tops[k + 1]))}
                for k in range(len(tops) - 1)]
    return {
        "notice": NOTICE, "config": asdict(cfg),
        "adequacy": "memadai" if enough else "eksploratif",
        "adequacy_reason": (f"{len(used)} windows with at least {cfg.min_postings_per_window} "
                            f"postings; {cfg.min_windows_for_trend} are needed for a trend"),
        "trend_test_note": ("with three or four windows a trend test has almost no power: "
                            "'tidak_terdeteksi' means 'cannot tell', not 'stable'"
                            if len(used) < 6 else ""),
        "windows": [{k: v for k, v in w.items() if not k.startswith("_")} for w in windows],
        "n_dated_postings": len(posting_month), "n_items_analysed": len(rows),
        "n_items_too_rare": len(items) - len(rows),
        "trend_counts": _count(rows, "trend"), "volatility_counts": _count(rows, "volatility"),
        "n_prior_matched": sum(r["prior_matched"] for r in rows),
        "n_recommended_at_parent": sum(r["recommend_at"]["level"] == "parent" for r in rows),
        "top_n_turnover": turnover, "items": rows,
    }


def _count(rows, key) -> dict:
    out: dict[str, int] = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return out


def aggregate(member_weights: Sequence[tuple[float, float]]) -> dict:
    """A competency's weight from its member skills: (weight, prevalence) pairs.
    The proposal leaves the rule open ("nilai maksimum atau rerata berbobot
    prevalensi"); both are reported and the choice is recorded by the caller."""
    if not member_weights:
        return {"max": None, "prevalence_weighted_mean": None}
    total = sum(p for _, p in member_weights)
    return {"max": max(w for w, _ in member_weights),
            "prevalence_weighted_mean": (sum(w * p for w, p in member_weights) / total
                                         if total > 0 else None)}


def esco_parents(lexicon_version: str = "v3") -> dict[str, tuple[str, str]]:
    from fivetag.lexicon import LEXICON_ROOT
    path = LEXICON_ROOT / lexicon_version / "ESCO_ICT.tsv"
    out: dict[str, tuple[str, str]] = {}
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                out[r["name"].lower()] = (r["parent_code"], r["parent_title"])
    return out


def write(result: dict, out_dir: Path, name: str = "temporal") -> Path:
    out_dir = Path(out_dir).resolve()
    if GS_DIR.resolve() in out_dir.parents:
        raise ValueError("temporal output must not be written under DATA/gold_standard/")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False),
                                          encoding="utf-8")
    labels = [f"{w['start']}..{w['end']}" for w in result["windows"] if w["included"]]
    with (out_dir / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item", "n_postings", *[f"share {x}" for x in labels], "direction",
                    "trend_test", "q_value", "volatility", "cv", "persistence_weight",
                    "prior_domain", "recommend_at", "example"])
        for r in result["items"]:
            rec = r["recommend_at"]
            w.writerow([r["label"], r["n_postings"], *[f"{s:.4f}" for s in r["shares"]],
                        r["direction"], r["trend"],
                        "" if r["q_value"] is None else f"{r['q_value']:.3f}", r["volatility"],
                        f"{r['cv']:.2f}", f"{r['persistence_weight']:.3f}", r["prior_domain"],
                        rec["node"], rec.get("example", "")])
    return out_dir / f"{name}.json"


def posting_months(gs_dir: Path = GS_DIR, only_stratum: str | None = None) -> dict[str, int]:
    import pandas as pd
    df = pd.read_parquet(gs_dir / "sentence_pool.parquet",
                         columns=["posting_id", "date_posted", "stratum"]).drop_duplicates(
        "posting_id")
    if only_stratum:
        df = df[df["stratum"] == only_stratum]
    out = {}
    for pid, d in zip(df["posting_id"], df["date_posted"]):
        m = month_index(d)
        if m is not None:
            out[str(pid)] = m
    return out


def run_pool(tag: str, tags: Sequence[str] = ("TT",), only_stratum: str | None = None,
             lexicon: str = "v2", cfg: TemporalConfig | None = None, log=print) -> dict:
    """Skill-level time analysis over the WHOLE sentence pool with the rule
    extractor. No grouping is needed at this level, so every dated posting is
    used."""
    import pandas as pd
    from fivetag.e0_rules import E0Rules
    from .evidence import canonical_key
    months = posting_months(only_stratum=only_stratum)
    df = pd.read_parquet(GS_DIR / "sentence_pool.parquet",
                         columns=["posting_id", "sentence_text"])
    df = df[df["posting_id"].isin(set(months))]
    log(f"{len(df)} sentences from {len(months)} dated postings"
        + (f" (stratum {only_stratum})" if only_stratum else ""))
    extractor = E0Rules(version=lexicon)
    wanted = set(tags)
    all_items: dict[str, dict[str, set]] = {t: {} for t in tags}
    all_surface: dict[str, dict[str, dict[str, int]]] = {t: {} for t in tags}
    for pid, text in zip(df["posting_id"], df["sentence_text"]):      # one pass for every tag
        for p in extractor.extract(text):
            if p.label in wanted:
                key = canonical_key(p.text, p.label)
                if key:
                    all_items[p.label].setdefault(key, set()).add(str(pid))
                    forms = all_surface[p.label].setdefault(key, {})
                    forms[p.text] = forms.get(p.text, 0) + 1
    out = {}
    for tag_code in tags:
        items, surface = all_items[tag_code], all_surface[tag_code]
        labels = {k: max(sorted(v), key=lambda s: v[s]) for k, v in surface.items()}
        result = analyse(items, months, cfg, labels=labels, prior=load_prior(),
                         parent=esco_parents())
        result["evidence_source"] = (f"E0 rules, lexicon {lexicon}, whole pool"
                                     + (f", stratum {only_stratum}" if only_stratum else ""))
        result["tag"] = tag_code
        write(result, OUT_ROOT / tag, f"temporal_{tag_code}")
        out[tag_code] = result
        log(f"{tag_code}: {result['n_items_analysed']} items analysed, adequacy "
            f"{result['adequacy']}, trend {result['trend_counts']}, volatility "
            f"{result['volatility_counts']}")
    return out


def main() -> None:
    import argparse
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="name of the output folder")
    ap.add_argument("--tags", nargs="+", default=["TT"], help="which of TT KN SS WB SH")
    ap.add_argument("--stratum", default=None, help="indonesia | asean | oecd")
    ap.add_argument("--lexicon", default="v2")
    args = ap.parse_args()
    results = run_pool(args.tag, args.tags, args.stratum, args.lexicon)
    for tag_code, r in results.items():
        print(f"\n{r['notice']}\n{tag_code}: windows " + ", ".join(
            f"{w['start']}..{w['end']} n={w['n_postings']}{'' if w['included'] else ' (left out)'}"
            for w in r["windows"]))
        for row in r["items"][:12]:
            print(f"  {row['label'][:28]:<28} shares {row['shares']}  {row['direction']:<5} "
                  f"vol {row['volatility']:<6} weight {row['persistence_weight']:.2f}")
    print(f"\n-> {OUT_ROOT / args.tag}")


if __name__ == "__main__":
    main()
