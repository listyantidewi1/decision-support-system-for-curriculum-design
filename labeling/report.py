"""Human-readable view of one ensemble run.

    .venv-ml\\Scripts\\python.exe -m labeling.report DATA\\labeling\\runs\\<run_id>

Writes `report.html` next to the run's data: for every sentence, what each model
marked and how the vote went. It makes no API call — it re-reads `raw.jsonl` and
re-runs the (free, deterministic) verifier and consensus.

The page shows corpus sentences and LLM output, so it stays in the gitignored
run directory. Do NOT show it to the three annotators before their own
annotation of those sentences is finished (guideline §2.7).
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from gold_standard import TAGS

from .consensus import ABSTAIN, ConsensusConfig, decide_sentence
from .verify import verify_output

TAG_COLOURS = {"SH": "#2563eb", "SS": "#9333ea", "KN": "#0d9488",
               "TT": "#ea580c", "WB": "#be185d"}
STATUS_NOTE = {
    "UNANIMOUS": "all models agree",
    "MAJORITY": "two-thirds agree",
    "BOUNDARY_DISAGREEMENT": "tag agreed, boundaries not",
    "NONE_MAJORITY": "only a minority marked it",
    "CONFLICT": "no agreement",
}
CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#ddd;--card:#f7f7f8;--ok:#15803d;--no:#b45309}
@media (prefers-color-scheme:dark){:root{--bg:#16181c;--fg:#e8e8e8;--muted:#9aa0a6;
--line:#333;--card:#1f2228;--ok:#4ade80;--no:#fbbf24}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,Segoe UI,sans-serif;
max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:22px}h2{font-size:15px;margin:0 0 6px}
.meta,.small{color:var(--muted);font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:14px;margin:16px 0}
.sent{font-size:16px;margin:4px 0 12px}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{border-top:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}
.chip{display:inline-block;border-radius:4px;padding:1px 6px;margin:2px 4px 2px 0;
color:#fff;font-size:13px}
.tag{font-weight:700;font-size:12px;opacity:.9;margin-left:4px}
.none{color:var(--muted)}.acc{color:var(--ok);font-weight:600}.abs{color:var(--no)}
.legend .chip{margin-right:8px}.scroll{overflow-x:auto}
"""


def _chip(text: str, label: str) -> str:
    colour = TAG_COLOURS.get(label, "#555")
    return (f'<span class="chip" style="background:{colour}">{html.escape(text)}'
            f'<span class="tag">{html.escape(label)}</span></span>')


def _vote(v: str) -> str:
    if v in TAG_COLOURS:
        return f'<span class="chip" style="background:{TAG_COLOURS[v]}">{v}</span>'
    return f'<span class="none">{"—" if v == "NONE" else "no vote"}</span>'


def build_report(run_dir: Path, text_by_sid: dict[str, str],
                 reference: dict[str, list[tuple[int, int, str]]] | None = None,
                 reference_label: str = "Reference") -> str:
    """`reference` maps a sentence id to its expected (start, end, tag) spans.
    When given, each sentence shows the reference and how many of its spans each
    model matched exactly (same boundaries, same tag)."""
    run_dir = Path(run_dir)
    reference = reference or {}
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    models = summary["run_config"]["models"]
    sha = summary.get("request_sha256", "")
    cfg = ConsensusConfig(models=tuple(models),
                          accept_majority=summary["run_config"].get("accept_majority", False))
    raw: dict[tuple[str, str], dict] = {}
    for line in (run_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("request_sha256", "") == sha:
                raw[(r["model"], r["sentence_id"])] = r
    sids = [json.loads(line)["sentence_id"] for line in
            (run_dir / "consensus.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    short = {m: m.split("/")[-1] for m in models}
    cost = summary["cost"]
    cons = summary["consensus"]
    parts = [f"<!doctype html><meta charset='utf-8'><title>Ensemble run</title><style>{CSS}</style>",
             "<h1>Multi-LLM annotation results</h1>",
             f"<p class='meta'>Run <b>{html.escape(run_dir.name)}</b> · "
             f"{summary['n_sentences']} sentences · prompt "
             f"{html.escape(summary['prompt']['prompt_version'])} · "
             f"cost ${cost['charged_usd_total']:.4f}</p>",
             "<p class='legend'>" + "".join(_chip(TAGS[t], t) for t in TAGS) + "</p>",
             "<p class='small'>Accepted = unanimous. Two-thirds majorities abstain until "
             "the majority rule has been validated on the development gold. "
             f"Candidates: {cons['n_candidates']}, accepted: {cons['acceptance_rate']['k']}. "
             "Do not show this page to the annotators before they finish.</p>"]

    for n, sid in enumerate(sids, 1):
        sentence = text_by_sid.get(sid)
        if sentence is None:
            parts.append(f"<div class='card'><h2>{n}. {html.escape(sid)}</h2>"
                         "<p class='abs'>sentence text not found in the sample</p></div>")
            continue
        outs = []
        for m in models:
            rec = raw.get((m, sid))
            if rec and rec.get("ok"):
                outs.append(verify_output(m, sid, sentence, rec.get("raw") or "",
                                          cfg.verify_policy))
        result = decide_sentence(sid, sentence, outs, cfg)
        by_model = {o.model: o for o in outs}

        parts.append(f"<div class='card'><h2>{n}. <span class='small'>{html.escape(sid)}"
                     f"</span></h2><div class='sent'>{html.escape(sentence)}</div>")
        parts.append("<div class='scroll'><table><tr><th>Model</th><th>What it marked</th></tr>")
        ref = reference.get(sid)
        if ref is not None:
            ref_cell = ("".join(_chip(sentence[a:b], t) for a, b, t in ref)
                        or "<span class='none'>nothing should be marked</span>")
            parts.append(f"<tr><td><b>{html.escape(reference_label)}</b></td>"
                         f"<td>{ref_cell}</td></tr>")
        for m in models:
            o = by_model.get(m)
            if o is None:
                cell = "<span class='abs'>no response</span>"
            elif o.status != "VALID":
                cell = "<span class='abs'>invalid response</span>"
            elif not o.spans:
                cell = "<span class='none'>nothing</span>"
            else:
                cell = "".join(_chip(s.text, s.label) for s in o.spans)
            if o is not None and o.rejections:
                cell += (f" <span class='small'>({len(o.rejections)} rejected: "
                         f"{html.escape(', '.join(sorted({r.code for r in o.rejections})))})</span>")
            if ref is not None and o is not None and o.status == "VALID":
                got = {(x.start, x.end, x.label) for x in o.spans}
                hit = sum(1 for r in ref if tuple(r) in got)
                cls = "acc" if hit == len(ref) else "abs"
                cell += (f" <span class='{cls}'>&nbsp;matched {hit} of {len(ref)}</span>"
                         if ref else "")
            parts.append(f"<tr><td>{html.escape(short[m])}</td><td>{cell}</td></tr>")
        parts.append("</table></div>")

        if result.candidates:
            head = "".join(f"<th>{html.escape(short[m])}</th>" for m in models)
            parts.append("<div class='scroll'><table><tr><th>Span</th>" + head
                         + "<th>Result</th><th>Decision</th></tr>")
            for c in result.candidates:
                span = c.text if c.text else " / ".join(sorted({v["text"] for v in c.variants}))
                votes = "".join(f"<td>{_vote(c.votes[m])}</td>" for m in models)
                decision = ("<span class='acc'>accepted</span>" if c.disposition != ABSTAIN
                            else "<span class='abs'>abstain</span>")
                parts.append(f"<tr><td>{html.escape(span)}</td>{votes}"
                             f"<td>{html.escape(STATUS_NOTE.get(c.status, c.status))}</td>"
                             f"<td>{decision}</td></tr>")
            parts.append("</table></div>")
        else:
            parts.append(f"<p class='small'>No model marked anything "
                         f"({html.escape(str(result.empty_consensus))}).</p>")
        parts.append("</div>")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    args = ap.parse_args()
    from .run import load_split
    text: dict[str, str] = {}
    for split in ("practice_10", "pilot_100"):
        try:
            text.update({s["sentence_id"]: s["sentence_text"] for s in load_split(split)})
        except FileNotFoundError:
            pass
    # Runs on SkillSpan sentences: take the text from the SkillSpan-derived file
    # and show the guideline reference next to the models' answers.
    reference: dict[str, list[tuple[int, int, str]]] = {}
    label = "Reference"
    try:
        from . import skillspan_guideline as ssg
        if ssg.SOURCE.exists():
            text.update(ssg._source_sentences())
            for batch in (1, 2):
                path = ssg.batch_path(batch)
                if path.exists():
                    items, meta = ssg.load_items(path)
                    for it in items:
                        reference[it.item_id] = list(it.expected)
            label = "Reference (guideline)"
    except Exception as exc:  # noqa: BLE001 - the plain report still works
        print(f"(no SkillSpan reference loaded: {type(exc).__name__}: {exc})")
    out = Path(args.run_dir) / "report.html"
    out.write_text(build_report(Path(args.run_dir), text, reference, label),
                   encoding="utf-8")
    print(f"-> {out.resolve()}")


if __name__ == "__main__":
    main()
