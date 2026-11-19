"""Score any extractor — on a full reference (gold) or on a partial one.

Two modes, because two kinds of reference exist:

  * FULL reference — every taggable span of every sentence is annotated (the
    gold standard, once it exists; `fivetag.synthetic` until then). Precision,
    recall and F1 are all meaningful. Uses `gold_standard.metrics` and nothing
    else, so the numbers are the same kind as the inter-annotator ceiling.

  * PARTIAL reference — only the spans the guideline settles are listed
    (`labeling.conformance`, `labeling.skillspan_guideline`). A prediction that
    is not listed is not necessarily wrong, so precision is undefined. What is
    reported is the share of expected spans found and the item pass rate — the
    same score the LLMs were given, which makes E0 directly comparable to them.

CLI (free, no network):

    python -m fivetag.evaluate dev              # E0 on every development set
    python -m fivetag.evaluate dev --errors 15  # ... and list what it got wrong
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Mapping, Sequence

from gold_standard import metrics

from . import TAG_CODES
from .base import Extractor, to_spans


def run_extractor(extractor: Extractor, sentences: Mapping[str, str]) -> tuple[dict, dict]:
    """(predictions by sentence id, timing). Deterministic order."""
    t0 = time.perf_counter()
    preds = {sid: extractor.extract(text) for sid, text in sentences.items()}
    dt = time.perf_counter() - t0
    n = len(sentences)
    return preds, {"n_sentences": n, "seconds": dt,
                   "sentences_per_second": (n / dt) if dt > 0 else float("inf")}


def _json_safe(x):
    if isinstance(x, float) and x != x:
        return None
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


def score_full(ref_by_sid: Mapping[str, Sequence], hyp_by_sid: Mapping[str, Sequence],
               sids: Sequence[str] | None = None, *, bootstrap: bool = True) -> dict:
    """Tabel 3.7 "akurasi ekstraksi" and "kualitas kategori" against a full reference.

    `sids` is the scoring population; it defaults to every reference sentence
    id, INCLUDING those with no reference span (false positives there count).
    """
    sids = list(sids if sids is not None else ref_by_sid)
    ref = {s: to_spans(ref_by_sid.get(s, ())) for s in sids}
    hyp = {s: to_spans(hyp_by_sid.get(s, ())) for s in sids}
    out = {"n_sentences": len(sids), "n_ref": sum(len(v) for v in ref.values()),
           "n_hyp": sum(len(v) for v in hyp.values())}
    for mode in ("strict", "partial"):
        c = metrics.contribs(ref, hyp, sids, mode)
        m = metrics.micro(c)
        if bootstrap and sids:
            m["f1_ci95"] = metrics.bootstrap_f1_ci(c)
        m["per_tag"] = metrics.per_tag(ref, hyp, sids, mode)
        out[mode] = m
    out["confusion_matrix"] = metrics.confusion_matrix(ref, hyp, sids)
    out["error_taxonomy"] = metrics.error_taxonomy(ref, hyp, sids)
    return out


def compare_full(ref_by_sid, hyp_a, hyp_b, sids: Sequence[str] | None = None,
                 mode: str = "strict") -> dict:
    """Paired comparison of two systems on the same sentences (bootstrap + McNemar)."""
    sids = list(sids if sids is not None else ref_by_sid)
    ref = {s: to_spans(ref_by_sid.get(s, ())) for s in sids}
    a = {s: to_spans(hyp_a.get(s, ())) for s in sids}
    b = {s: to_spans(hyp_b.get(s, ())) for s in sids}
    return {"mode": mode,
            "paired_bootstrap_f1": metrics.paired_bootstrap_f1(
                metrics.contribs(ref, a, sids, mode), metrics.contribs(ref, b, sids, mode)),
            "mcnemar_recall_strict": metrics.mcnemar_recall_strict(ref, a, b, sids)}


def score_partial(items, hyp_by_sid: Mapping[str, Sequence]) -> dict:
    """Against conformance-style items (expected / forbidden / closed)."""
    from labeling.conformance import score
    s = score({it.item_id: list(hyp_by_sid.get(it.item_id, ())) for it in items}, items)
    by_tag: dict[str, list[int]] = {t: [0, 0] for t in TAG_CODES}
    for it in items:
        got = {(p.start, p.end, p.label) for p in hyp_by_sid.get(it.item_id, ())}
        for e in it.expected:
            by_tag[e[2]][1] += 1
            by_tag[e[2]][0] += e in got
    s["expected_found_by_tag"] = {t: {"found": k, "n": n} for t, (k, n) in by_tag.items() if n}
    s.pop("conformance_version", None)
    s.pop("review_status", None)
    return s


def development_sets() -> dict[str, list]:
    """Every partial-reference set available locally. The SkillSpan batches live
    in the gitignored DATA/labeling/ and are simply absent on a fresh clone."""
    from labeling.conformance import ITEMS
    sets = {"conformance (synthetic, 109 items)": list(ITEMS)}
    try:
        from labeling.skillspan_guideline import batch_path, load_items
        for n in (1, 2):
            if batch_path(n).is_file():
                items, _ = load_items(batch_path(n))
                sets[f"SkillSpan under the guideline, batch {n} ({len(items)} items)"] = items
    except Exception as exc:                                   # noqa: BLE001
        sets[f"(SkillSpan batches unavailable: {exc})"] = []
    return sets


def evaluate_dev(extractor: Extractor) -> dict:
    out = {"extractor": extractor.manifest(), "sets": {}}
    for name, items in development_sets().items():
        if not items:
            continue
        preds, timing = run_extractor(extractor, {it.item_id: it.sentence for it in items})
        s = score_partial(items, preds)
        s["timing"] = timing
        out["sets"][name] = s
    return out


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["dev"])
    ap.add_argument("--errors", type=int, default=0, help="list this many failed items per set")
    ap.add_argument("--json", help="also write the full result to this file")
    args = ap.parse_args()

    from .e0_rules import E0Rules
    result = evaluate_dev(E0Rules())
    m = result["extractor"]
    print(f"{m['extractor']}  rules {m['rules_version']}  lexicon {m['lexicon_version']} "
          f"({m['lexicon_sha256'][:12]})")
    for name, s in result["sets"].items():
        print(f"\n{name}")
        print(f"  expected spans found : {s['expected_spans_found']}/{s['expected_spans_total']}")
        print(f"  items passed         : {s['passed']}/{s['n_items']}")
        print("  by tag               : " + "  ".join(
            f"{t} {v['found']}/{v['n']}" for t, v in s["expected_found_by_tag"].items()))
        print(f"  speed                : {s['timing']['sentences_per_second']:.0f} sentences/s")
        shown = 0
        for r in s["items"]:
            if shown >= args.errors:
                break
            if not r["pass"]:
                shown += 1
                print(f"    {r['item_id']}: missing {r['missing']}  forbidden {r['forbidden_hit']}"
                      + (f"  extra {r['extra']}" if r["extra"] else ""))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(_json_safe(result), fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
