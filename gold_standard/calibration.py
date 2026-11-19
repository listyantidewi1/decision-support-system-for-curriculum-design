"""Calibration report — the working document for the session with the annotators.

    .venv-ml\\Scripts\\python.exe -m gold_standard.calibration <iaa_report.json>

The agreement report (`gold_standard.iaa`) says HOW MUCH the annotators agreed.
This turns the same data into WHERE and HOW they disagreed, in a page a
researcher can put on a screen in front of three experts: which tags get
confused with which, what kind of boundary differences recur, who marks much
more or less than the others, and every disputed sentence with each annotator's
spans side by side.

It reads the annotation files named in the report's own `provenance.ann_dir` and
writes its page NEXT TO the report. It therefore inherits the report's location:
`gold_standard.iaa` already refuses to put a report built from synthetic or
unverified input into the evidence directory, so this page cannot end up there
either. A report built from such input is bannered as a rehearsal.
"""
from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from . import ANNOTATORS, FLAGS, TAGS
from . import gate as gate_mod

TAG_COLOURS = {"SH": "#2563eb", "SS": "#9333ea", "KN": "#0d9488",
               "TT": "#ea580c", "WB": "#be185d"}
CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#ddd;--card:#f7f7f8;--ok:#15803d;
--warn:#b45309;--bad:#b91c1c}
@media (prefers-color-scheme:dark){:root{--bg:#16181c;--fg:#e8e8e8;--muted:#9aa0a6;
--line:#333;--card:#1f2228;--ok:#4ade80;--warn:#fbbf24;--bad:#f87171}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,Segoe UI,sans-serif;
max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:22px}h2{font-size:18px;margin-top:28px}h3{font-size:15px;margin:0 0 6px}
.small{color:var(--muted);font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:14px;margin:14px 0}
.banner{border:2px solid var(--bad);color:var(--bad);border-radius:8px;padding:10px 14px;
font-weight:600;margin:12px 0}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{border-top:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}
.chip{display:inline-block;border-radius:4px;padding:1px 6px;margin:2px 4px 2px 0;
color:#fff;font-size:13px}.tag{font-weight:700;font-size:12px;margin-left:4px}
.none{color:var(--muted)}.ok{color:var(--ok);font-weight:600}
.warn{color:var(--warn);font-weight:600}.bad{color:var(--bad);font-weight:600}
.sent{font-size:16px;margin:4px 0 10px}.scroll{overflow-x:auto}
"""


def _e(x) -> str:
    return html.escape(str(x))


def _f(x, nd: int = 2) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if v != v else f"{v:.{nd}f}"


def _chip(text: str, label: str) -> str:
    return (f'<span class="chip" style="background:{TAG_COLOURS.get(label, "#555")}">'
            f'{_e(text)}<span class="tag">{_e(label)}</span></span>')


def bound_pattern(spans: list[dict]) -> str:
    """How the boundaries of one BOUND dispute differ, in words."""
    starts = {s["start"] for s in spans}
    ends = {s["end"] for s in spans}
    if len(starts) == 1 and len(ends) > 1:
        return "same start, different end"
    if len(ends) == 1 and len(starts) > 1:
        return "same end, different start"
    lo, hi = min(starts), max(ends)
    if any(s["start"] == lo and s["end"] == hi for s in spans):
        return "one span inside another"
    return "shifted or split"


def analyse(report: dict, raw_by_code: dict[str, dict]) -> dict:
    """The counts behind the page, as plain data (also what the tests check)."""
    coders = list(report.get("annotators", []))
    queue = report.get("adjudication_queue", [])
    dispute_types = Counter()
    tag_confusion = Counter()
    bound_patterns = Counter()
    alone = defaultdict(Counter)            # coder -> tag -> spans only they marked
    missed = Counter()                      # coder -> spans the other two both marked
    for q in queue:
        for d in q.get("span_disputes", []):
            dispute_types[d["type"]] += 1
            spans = d.get("spans", [])
            if d["type"] == "LABEL":
                tag_confusion[" / ".join(sorted(d.get("labels", {})))] += 1
            elif d["type"] == "BOUND":
                bound_patterns[bound_pattern(spans)] += 1
            elif d["type"] == "PRESENCE":
                who = sorted({s["coder"] for s in spans})
                if len(who) == 1:
                    for s in spans:
                        alone[who[0]][s["label"]] += 1
                elif len(who) == len(coders) - 1:
                    for c in coders:
                        if c not in who:
                            missed[c] += 1

    per_annotator = {}
    for c in coders:
        recs = {k: v for k, v in (raw_by_code.get(c) or {}).items()
                if k != "_meta" and isinstance(v, dict)}
        spans = [s for r in recs.values() for s in r.get("spans", [])]
        conf = []
        for s in spans:
            try:
                conf.append(float(s.get("confidence")))
            except (TypeError, ValueError):
                pass
        flags = Counter()
        for r in recs.values():
            for holder in [r] + list(r.get("spans", [])):
                for f in str(holder.get("flag") or "").replace(";", ",").split(","):
                    f = f.strip()
                    if f:
                        flags[f] += 1
        per_annotator[c] = {
            "n_sentences": len(recs),
            "relevance": dict(Counter(r.get("relevance") or "(blank)" for r in recs.values())),
            "has_5tag": dict(Counter(r.get("has_5tag") or "(blank)" for r in recs.values())),
            "n_spans": len(spans),
            "spans_by_tag": dict(Counter(s.get("label") for s in spans)),
            "mean_confidence": (sum(conf) / len(conf)) if conf else float("nan"),
            "flags": dict(flags),
            "marked_alone": dict(alone.get(c, {})),
            "missed_what_others_marked": missed.get(c, 0),
        }

    n_spans = [per_annotator[c]["n_spans"] for c in coders] or [0]
    agenda = []
    for pair, n in tag_confusion.most_common(3):
        agenda.append(f"Tags {pair} were given to the same span {n} time(s). Which rule "
                      f"of the guideline decides between them?")
    if bound_patterns:
        pat, n = bound_patterns.most_common(1)[0]
        agenda.append(f"The commonest boundary difference is \"{pat}\" ({n} of "
                      f"{sum(bound_patterns.values())}). Agree where such spans start and end.")
    if max(n_spans) > 1.5 * max(1, min(n_spans)):
        hi = max(coders, key=lambda c: per_annotator[c]["n_spans"])
        lo = min(coders, key=lambda c: per_annotator[c]["n_spans"])
        agenda.append(f"{hi} marked {per_annotator[hi]['n_spans']} spans and {lo} marked "
                      f"{per_annotator[lo]['n_spans']}. Discuss what counts as an entity.")
    n_rel = sum(1 for q in queue if q.get("relevance_agreement") != "unanimous")
    if n_rel:
        agenda.append(f"Relevance was not unanimous on {n_rel} sentence(s). Review a few "
                      f"together against guideline section 3.1.")
    n_conf = sum(1 for q in queue if q.get("guideline_conflicts"))
    if n_conf:
        agenda.append(f"{n_conf} sentence(s) break a guideline rule as annotated (for "
                      f"example NON_RELEVANT with a span). Clarify the rule.")
    return {"coders": coders, "n_disputed_sentences": len(queue),
            "dispute_types": dict(dispute_types), "tag_confusion": dict(tag_confusion),
            "bound_patterns": dict(bound_patterns), "per_annotator": per_annotator,
            "agenda": agenda}


def build(report: dict, raw_by_code: dict[str, dict], text_by_sid: dict[str, str],
          n_expected: int | None = None) -> str:
    a = analyse(report, raw_by_code)
    coders = a["coders"]
    sl = report.get("sentence_level", {})
    span = report.get("span_level", {}).get("modes", {})
    tok = report.get("token_level", {})
    prov = report.get("provenance") or {}
    synthetic = bool(prov.get("input_quarantined")) or not prov.get(
        "ids_verified_against_sample", True)
    g = gate_mod.evaluate(report, n_expected)

    p = [f"<!doctype html><meta charset='utf-8'><title>Calibration report</title>"
         f"<style>{CSS}</style>",
         f"<h1>Calibration report: {_e(report.get('split'))}</h1>",
         f"<p class='small'>Annotators: {_e(', '.join(coders))} · guideline "
         f"{_e(report.get('guideline'))} · agreement report {_e(report.get('created_utc'))}</p>"]
    if synthetic:
        p.append("<div class='banner'>REHEARSAL ON SYNTHETIC OR UNVERIFIED DATA. No expert "
                 "produced these annotations. Nothing on this page is evidence about the "
                 "study.</div>")

    # ---- 1. agreement at a glance
    p.append("<h2>1. How much did they agree?</h2>")
    p.append("<div class='scroll'><table><tr><th>What</th><th>Observed agreement</th>"
             "<th>Kappa</th><th>PABAK</th><th>Sentences</th></tr>")
    for name, key in (("Relevance of the sentence", "relevance"),
                      ("Does it contain one of the five tags", "has_5tag")):
        d = sl.get(key, {})
        p.append(f"<tr><td>{name}</td><td>{_f(d.get('observed_agreement'))}</td>"
                 f"<td>{_f(d.get('fleiss_kappa'))}</td><td>{_f(d.get('pabak'))}</td>"
                 f"<td>{_e(d.get('n_complete_cases', 'n/a'))}</td></tr>")
    p.append("</table></div>")
    ex = (span.get("exact", {}).get("micro") or {})
    ov = (span.get("overlap_iou50", {}).get("micro") or {})
    p.append("<div class='scroll'><table><tr><th>Spans</th><th>Value</th>"
             "<th>What it means</th></tr>"
             f"<tr><td>Exact match F1</td><td>{_f(ex.get('f1'))}</td>"
             "<td>same start, same end, same tag</td></tr>"
             f"<tr><td>Overlap F1</td><td>{_f(ov.get('f1'))}</td>"
             "<td>same tag, spans overlap by at least half</td></tr>"
             f"<tr><td>Token-level kappa</td><td>{_f(tok.get('fleiss_kappa'))}</td>"
             "<td>comparable to SkillSpan's published 0.70-0.75</td></tr></table></div>")
    p.append("<p class='small'>Kappa drops when one answer dominates, even if the "
             "annotators are equally careful, so read it together with PABAK. With about "
             "100 sentences every figure here is uncertain by roughly ±0.15: the pilot "
             "shows where the guideline is read differently, it does not measure "
             "reliability.</p>")

    # ---- gate
    cls = {"GO": "ok", "RECALIBRATE": "warn", "STOP": "bad", "NOT READY": "bad"}[g["verdict"]]
    p.append(f"<div class='card'><h3>Go/no-go check: <span class='{cls}'>"
             f"{_e(g['verdict'])}</span></h3><p>{_e(g['meaning'])}</p>"
             f"<p class='small'>Rule {_e(g['rule_version'])} — {_e(g['rule_status'])}. "
             + " · ".join(f"{_e(c['measure'])}: {_f(c['value'])} ({_e(c['band'])})"
                          for c in g["checks"]) + "</p>"
             + "".join(f"<p class='bad'>{_e(x)}</p>" for x in g["problems"]) + "</div>")

    # ---- 2. agenda
    p.append("<h2>2. Suggested points for the session</h2>")
    if a["agenda"]:
        p.append("<ol>" + "".join(f"<li>{_e(x)}</li>" for x in a["agenda"]) + "</ol>")
    else:
        p.append("<p class='none'>No recurring pattern stood out.</p>")

    # ---- 3. kinds of disagreement
    p.append("<h2>3. What kind of disagreements</h2>")
    names = {"LABEL": "Same span, different tag", "BOUND": "Same tag, different boundaries",
             "PRESENCE": "Not everyone marked it"}
    p.append("<div class='scroll'><table><tr><th>Kind</th><th>Count</th></tr>"
             + "".join(f"<tr><td>{_e(names.get(k, k))}</td><td>{v}</td></tr>"
                       for k, v in sorted(a["dispute_types"].items(), key=lambda kv: -kv[1]))
             + "</table></div>")
    if a["tag_confusion"]:
        p.append("<h3>Which tags were confused</h3><div class='scroll'><table><tr>"
                 "<th>Tags given to the same span</th><th>Count</th></tr>"
                 + "".join(f"<tr><td>{_e(k)}</td><td>{v}</td></tr>" for k, v in
                           sorted(a["tag_confusion"].items(), key=lambda kv: -kv[1]))
                 + "</table></div>")
    if a["bound_patterns"]:
        p.append("<h3>How boundaries differed</h3><div class='scroll'><table><tr>"
                 "<th>Pattern</th><th>Count</th></tr>"
                 + "".join(f"<tr><td>{_e(k)}</td><td>{v}</td></tr>" for k, v in
                           sorted(a["bound_patterns"].items(), key=lambda kv: -kv[1]))
                 + "</table></div>")

    # ---- 4. per annotator
    p.append("<h2>4. Each annotator</h2><div class='scroll'><table><tr><th></th>"
             + "".join(f"<th>{_e(c)} — {_e(ANNOTATORS.get(c, ''))}</th>" for c in coders)
             + "</tr>")

    def row(label, fn):
        p.append(f"<tr><td>{_e(label)}</td>"
                 + "".join(f"<td>{fn(a['per_annotator'][c])}</td>" for c in coders) + "</tr>")
    row("Sentences judged", lambda d: d["n_sentences"])
    row("Relevance answers", lambda d: _e(", ".join(f"{k} {v}" for k, v in
                                                     sorted(d["relevance"].items()))))
    row("Five-tag answers", lambda d: _e(", ".join(f"{k} {v}" for k, v in
                                                   sorted(d["has_5tag"].items()))))
    row("Spans marked", lambda d: d["n_spans"])
    row("Spans by tag", lambda d: "".join(_chip(str(d["spans_by_tag"].get(t, 0)), t)
                                          for t in TAGS))
    row("Marked alone (nobody else did)", lambda d: sum(d["marked_alone"].values()))
    row("Missed what both others marked", lambda d: d["missed_what_others_marked"])
    row("Mean confidence (1-4)", lambda d: _f(d["mean_confidence"], 1))
    row("Flags used", lambda d: _e(", ".join(f"{FLAGS.get(k, k)} {v}" for k, v in
                                             sorted(d["flags"].items())) or "none"))
    p.append("</table></div>")

    # ---- 5. sentences to discuss
    queue = sorted(report.get("adjudication_queue", []),
                   key=lambda q: (-(q.get("n_span_disputes") or 0),
                                  q.get("relevance_agreement") == "unanimous",
                                  q.get("sentence_id", "")))
    p.append(f"<h2>5. Sentences to discuss ({len(queue)})</h2>"
             "<p class='small'>Most disputed first. Each annotator's own spans are shown "
             "side by side.</p>")
    for n, q in enumerate(queue, 1):
        sid = q["sentence_id"]
        text = text_by_sid.get(sid) or q.get("sentence_text") or ""
        p.append(f"<div class='card'><h3>{n}. <span class='small'>{_e(sid)}</span></h3>"
                 f"<div class='sent'>{_e(text)}</div><div class='scroll'><table><tr>"
                 "<th>Annotator</th><th>Relevant?</th><th>Five-tag?</th><th>Spans</th></tr>")
        for c in coders:
            rec = (raw_by_code.get(c) or {}).get(sid) or {}
            chips = "".join(_chip(s.get("span_text") or text[int(s["start_char"]):
                                                             int(s["end_char"])],
                                  s.get("label", "?")) for s in rec.get("spans", []))
            p.append(f"<tr><td>{_e(c)}</td><td>{_e(rec.get('relevance') or '—')}</td>"
                     f"<td>{_e(rec.get('has_5tag') or '—')}</td>"
                     f"<td>{chips or '<span class=none>nothing</span>'}</td></tr>")
        p.append("</table></div>")
        notes = []
        for d in q.get("span_disputes", []):
            parts = ", ".join(f"{_e(s['coder'])}: “{_e(text[s['start']:s['end']])}” "
                              f"[{_e(s['label'])}]" for s in d.get("spans", []))
            notes.append(f"<li>{_e(names.get(d['type'], d['type']))} — {parts}</li>")
        for gc in q.get("guideline_conflicts", []):
            notes.append(f"<li class='bad'>Guideline conflict: {_e(gc)}</li>")
        if notes:
            p.append("<ul class='small'>" + "".join(notes) + "</ul>")
        p.append("</div>")
    return "\n".join(p)


def load_raw(ann_dir: Path, split: str, coders: list[str]) -> dict[str, dict]:
    out = {}
    for c in coders:
        f = Path(ann_dir) / f"{split}_{c}.json"
        if f.exists():
            out[c] = json.loads(f.read_text(encoding="utf-8"))
    return out


def write(report_path: Path) -> Path:
    from .iaa import _text_by_sid
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    split = report.get("split", "")
    ann_dir = Path((report.get("provenance") or {}).get("ann_dir", ""))
    if not ann_dir.is_dir():
        raise SystemExit(f"the annotation directory named in the report does not exist: "
                         f"{ann_dir}")
    text = _text_by_sid(split)
    raw = load_raw(ann_dir, split, list(report.get("annotators", [])))
    out = report_path.with_name(report_path.name.replace("iaa_", "calibration_", 1)
                                ).with_suffix(".html")
    out.write_text(build(report, raw, text, len(text) or None), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", help="the iaa_<split>_<stamp>.json file")
    args = ap.parse_args()
    print(f"-> {write(Path(args.report)).resolve()}")


if __name__ == "__main__":
    main()
