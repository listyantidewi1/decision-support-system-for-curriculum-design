"""Figures: one function per chart, each reading the same files the pipeline writes.

    python -m competency.figures run       DATA/competency/runs/<tag>
    python -m competency.figures temporal  DATA/competency/temporal/<tag>
    python -m competency.figures endtoend  DATA/competency/runs/endtoend_<tag>
    python -m competency.figures experiments          # development experiments

The charts are drawn from the output files, not from anything in memory, so the
same command redraws them when real data replace the stand-in data.

Every figure prints, in its bottom-left corner, the status of the data behind
it (engineering run, rehearsal, development). A chart copied out of this
repository still says what it is.

Design rules (they are deliberate; keep them when adding a chart):
  * the chart type follows the job of the data: amounts are bars, change over
    time is a line, one colour for one measure;
  * the five tags always have the same five colours, whatever is plotted;
  * one y-axis per chart: two measures get two panels, never two scales;
  * values are written at the marks, and a legend is present whenever there is
    more than one series;
  * text is dark ink, never the colour of a series.

Colours are the validated default palette of the charting guide used for this
project (colour-blind-safe in this fixed order).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_FIGURES = REPO_ROOT / "docs" / "experiments" / "figures"

SURFACE, INK, INK2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9",
                                         "#c3c2b7")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ONE = SERIES[0]                                    # single-measure charts
TAG_COLOR = dict(zip(("SH", "SS", "KN", "TT", "WB"), SERIES))
TAG_NAME = {"SH": "Hard skill", "SS": "Soft skill", "KN": "Knowledge",
            "TT": "Tool / technology", "WB": "Work behaviour"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE, "xtick.color": MUTED, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "legend.frameon": False, "savefig.facecolor": SURFACE})
    return plt


def _titles(ax, title: str, subtitle: str = "") -> None:
    """Bold title; the subtitle, when there is one, in regular secondary ink."""
    ax.set_title(title, pad=22 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.025, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK2,
                ha="left", va="bottom")


def _finish(fig, path: Path, note: str, legend_rows: int = 0) -> Path:
    """`legend_rows`: rows of a legend drawn under the plot, so the status note
    is placed below it instead of on top of it."""
    fig.text(0.01, 0.005 if not legend_rows else -0.05 - 0.055 * legend_rows, note, fontsize=7,
             color=MUTED, ha="left", va="bottom")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def hbar(labels, values, title, path, note, *, colors=None, value_fmt="{:,.0f}", xlabel="",
         subtitle=""):
    """Horizontal bars, largest on top, each value written at the end of its bar."""
    plt = _plt()
    n = len(labels)
    fig, ax = plt.subplots(figsize=(7.2, max(1.6, 0.34 * n + 1.1)))
    y = list(range(n))[::-1]
    ax.barh(y, values, height=0.62, color=colors or ONE, edgecolor=SURFACE, linewidth=1)
    top = max(values) if values and max(values) > 0 else 1
    for yi, v in zip(y, values):
        ax.text(v + top * 0.012, yi, value_fmt.format(v), va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, top * 1.14)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel(xlabel)
    if "%" in value_fmt:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    _titles(ax, title, subtitle)
    return _finish(fig, path, note)


def lines(x_labels, series: dict, title, path, note, *, ylabel="", pct=True, subtitle=""):
    """One line per series (at most eight), each labelled at its right end."""
    plt = _plt()
    names = list(series)[:8]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    xs = list(range(len(x_labels)))
    for name, color in zip(names, SERIES):
        ys = series[name]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=name)
    ends = sorted(((series[n][-1], n) for n in names), reverse=True)
    span = max(max(v) for v in series.values()) or 1
    last_y = None
    for v, name in ends:                              # nudge labels that would collide
        ypos = v if last_y is None or last_y - v > span * 0.045 else last_y - span * 0.045
        ax.text(xs[-1] + 0.08, ypos, name, fontsize=8, color=INK, va="center")
        last_y = ypos
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels)
    ax.set_xlim(-0.2, xs[-1] + 0.2 + 0.28 * max(1, len(xs) - 1))
    ax.set_ylim(bottom=0)
    if pct:
        digits = 0 if span >= 0.10 else 1 if span >= 0.01 else 2     # small shares need decimals
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.{digits}%}"))
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_ylabel(ylabel)
    _titles(ax, title, subtitle)
    if len(names) > 1:
        ax.legend(loc="upper left", fontsize=7.5, ncol=min(4, len(names)),
                  bbox_to_anchor=(0, -0.14))
    return _finish(fig, path, note,
                   legend_rows=0 if len(names) < 2 else -(-len(names) // min(4, len(names))))


def grouped_bars(groups, series: dict, title, path, note, *, ylabel="", ymax=None,
                 value_fmt="{:.0f}", subtitle="", reference: tuple | None = None):
    """A few series side by side for each group, values on the bars, legend."""
    plt = _plt()
    names = list(series)[:8]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    width = 0.8 / len(names)
    for k, (name, color) in enumerate(zip(names, SERIES)):
        xs = [i - 0.4 + width * (k + 0.5) for i in range(len(groups))]
        ax.bar(xs, series[name], width=width * 0.9, color=color, edgecolor=SURFACE,
               linewidth=1, label=name)
        for x, v in zip(xs, series[name]):
            ax.text(x, v, value_fmt.format(v), ha="center", va="bottom", fontsize=7.5, color=INK)
    if reference:
        ax.axhline(reference[0], color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        # at the left end of the line, clear of the value labels of the tallest bars
        ax.text(-0.48, reference[0], reference[1], fontsize=7.5, color=INK2, va="bottom",
                ha="left")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    if ymax:
        ax.set_ylim(0, ymax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_ylabel(ylabel)
    _titles(ax, title, subtitle)
    ax.legend(loc="upper left", fontsize=8, ncol=len(names), bbox_to_anchor=(0, -0.1))
    return _finish(fig, path, note, legend_rows=1)


def interval_dots(labels, values, intervals, title, path, note, *, xlabel="", xlim=None,
                  subtitle=""):
    """A point with its confidence interval per row: for accuracies and scores."""
    plt = _plt()
    n = len(labels)
    fig, ax = plt.subplots(figsize=(7.2, max(1.8, 0.5 * n + 1.2)))
    y = list(range(n))[::-1]
    for yi, v, (lo, hi) in zip(y, values, intervals):
        ax.plot([lo, hi], [yi, yi], color=ONE, linewidth=2, solid_capstyle="round")
        ax.plot([v], [yi], marker="o", markersize=8, color=ONE, markeredgecolor=SURFACE,
                markeredgewidth=1.5)
        ax.text(hi, yi + 0.22, f"{v:.1%}  ({lo:.0%}–{hi:.0%})", fontsize=8, color=INK, ha="right")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, n - 0.3)
    if xlim:
        ax.set_xlim(*xlim)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel(xlabel)
    _titles(ax, title, subtitle)
    return _finish(fig, path, note)


# ------------------------------------------------------------------ run figures
def figures_for_run(run_dir: Path) -> list[Path]:
    run_dir = Path(run_dir)
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    comps = json.loads((run_dir / "competencies.json").read_text(encoding="utf-8"))
    out, den = run_dir / "figures", report["denominators"]
    note = (f"{report['notice'].split(':')[0]} · {report['evidence_source']} · "
            f"{den['n_postings']} postings · not a finding")
    made = []
    tags = [t for t in TAG_COLOR if report["n_evidence_by_tag"].get(t)]
    made.append(hbar([TAG_NAME[t] for t in tags], [report["n_evidence_by_tag"][t] for t in tags],
                     "Evidence units by tag", out / "evidence_by_tag.png", note,
                     colors=[TAG_COLOR[t] for t in tags], xlabel="extracted spans"))
    st = report["grouping"]["status_distribution"]
    order = sorted(st, key=lambda k: -st[k])
    made.append(hbar([k.replace("_", " ") for k in order], [st[k] for k in order],
                     "Evidence groups by validity status", out / "group_status.png", note,
                     xlabel="groups"))
    cs = report["status_distribution"]
    order = [k for k in ("siap_ditelaah_formal", "kandidat_tinjauan", "eksploratif") if k in cs]
    made.append(hbar([k.replace("_", " ") for k in order], [cs[k] for k in order],
                     "Competency statements by recommendation status",
                     out / "contract_status.png", note, xlabel="statements"))
    blockers: dict[str, int] = {}
    for c in comps:
        for b in c["contract"]["blockers"]:
            blockers[b] = blockers.get(b, 0) + 1
    if blockers:
        order = sorted(blockers, key=lambda k: -blockers[k])
        made.append(hbar(order, [blockers[k] for k in order],
                         "Why statements stay exploratory", out / "blockers.png", note,
                         xlabel="statements held back by this reason"))
    top = comps[:15]
    if top:
        made.append(hbar([c["statement"]["text"][:60] for c in top],
                         [c["share_of_postings"] for c in top],
                         "Most frequent competency statements", out / "top_competencies.png", note,
                         value_fmt="{:.1%}", xlabel="share of postings"))
    import csv
    path = run_dir / "technology_profile.csv"
    if path.is_file():
        with path.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))[:15]
        if rows:
            made.append(hbar([r["text"] for r in rows], [float(r["share_of_postings"]) for r in rows],
                             "Most frequent technologies", out / "top_technologies.png", note,
                             value_fmt="{:.1%}", xlabel="share of postings",
                             colors=TAG_COLOR["TT"]))
    kk: dict[str, int] = {}
    for c in comps:
        k = c.get("kkni") or {}
        if k.get("level"):
            key = f"KKNI {k['level']}" + ("" if k.get("confident") else " (low confidence)")
            kk[key] = kk.get(key, 0) + 1
    if kk:
        order = sorted(kk)
        made.append(hbar(order, [kk[k] for k in order], "Proposed KKNI level of the statements",
                         out / "kkni_levels.png", note, xlabel="statements"))
    for ref, name in (("SKKNI", "coverage_skkni.png"),):
        cov: dict[str, int] = {}
        for c in comps:
            s = ((c.get("coverage") or {}).get(ref) or {}).get("proposed")
            if s:
                cov[s] = cov.get(s, 0) + 1
        if cov:
            order = [k for k in ("tercakup_jelas", "tercakup_sebagian", "masukan_pembaruan",
                                 "belum_tercakup") if k in cov]
            made.append(hbar([k.replace("_", " ") for k in order], [cov[k] for k in order],
                             f"Proposed coverage by {ref} (not yet validated by experts)",
                             out / name, note, xlabel="statements"))
    return made


def figures_for_temporal(folder: Path, top: int = 6) -> list[Path]:
    folder, made = Path(folder), []
    for path in sorted(folder.glob("temporal_*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        tag = r.get("tag", path.stem.split("_")[-1])
        labels = [f"{w['start']}\n{w['end']}" for w in r["windows"] if w["included"]]
        note = (f"ENGINEERING RUN · {r.get('evidence_source', '')} · adequacy: {r['adequacy']} · "
                f"not a finding")
        rows = sorted(r["items"], key=lambda x: -x["current_share"])[:top]
        if rows and len(labels) >= 2:
            made.append(lines(labels, {x["label"]: x["shares"] for x in rows},
                              f"Share of postings over time: most frequent {TAG_NAME.get(tag, tag).lower()} items",
                              folder / f"share_over_time_{tag}.png", note,
                              ylabel="share of postings in the window",
                              subtitle="half-year windows; a share depends on what was collected in each window"))
        movers = sorted(r["items"], key=lambda x: -abs(x["relative_slope"]))[:top]
        if movers and len(labels) >= 2:
            made.append(lines(labels, {x["label"]: x["shares"] for x in movers},
                              f"Largest relative changes: {TAG_NAME.get(tag, tag).lower()}",
                              folder / f"largest_changes_{tag}.png", note,
                              ylabel="share of postings in the window",
                              subtitle="descriptive only: with this few windows no trend can be tested"))
        vol = r["volatility_counts"]
        order = [k for k in ("rendah", "sedang", "tinggi") if k in vol]
        made.append(hbar([f"volatility {k}" for k in order], [vol[k] for k in order],
                         f"Volatility classes: {TAG_NAME.get(tag, tag).lower()}",
                         folder / f"volatility_{tag}.png", note, xlabel="items"))
    return made


def figures_for_endtoend(folder: Path) -> list[Path]:
    folder = Path(folder)
    r = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
    note = f"REHEARSAL · {r['n_postings']} postings, {r['n_sentences']} sentences · not a finding"
    names = [x["pipeline"] for x in r["rows"]]
    made = [
        hbar(names, [x["n_evidence"] for x in r["rows"]], "Evidence units found by each pipeline",
             folder / "figures" / "pipelines_evidence.png", note, xlabel="evidence units"),
        hbar(names, [x["n_competencies"] for x in r["rows"]],
             "Competency statements produced by each pipeline",
             folder / "figures" / "pipelines_competencies.png", note, xlabel="statements"),
        hbar(names, [x["extraction"].get("n_sentences_sent_to_llm", 0) for x in r["rows"]],
             "Sentences sent to the LLM by each pipeline",
             folder / "figures" / "pipelines_llm_sentences.png", note, xlabel="sentences"),
        hbar(names, [x["n_statements_below_grounding"]
                     + x["n_statements_that_would_have_been_rejected"] for x in r["rows"]],
             "Statements with unsupported content left in the output",
             folder / "figures" / "pipelines_unsupported.png", note, xlabel="statements"),
    ]
    return made


def figures_for_experiments(out: Path = EXPERIMENT_FIGURES) -> list[Path]:
    """Charts of the development experiments. Each reads stored result files and
    skips itself when they are not on this computer."""
    made = []
    runs = REPO_ROOT / "DATA" / "labeling" / "runs"
    models = ["openai/gpt-5.4-mini", "anthropic/claude-haiku-5.5", "qwen/qwen3.8-flash"]
    short = {"openai/gpt-5.4-mini": "GPT-5.4 mini", "anthropic/claude-haiku-5.5": "Claude Haiku 5.5",
             "qwen/qwen3.8-flash": "Qwen 3.8 Flash"}
    for batch, total, ids, exp in ((1, 74, ("f342a899", "6c256264", "079f7d73"), "EXP-005"),
                                   (2, 120, ("b2_f342a899", "b2_6c256264", "b2_079f7d73"), "EXP-006")):
        series, ok = {}, True
        for version, run_id in zip(("prompt v0", "prompt v1", "prompt v2"), ids):
            p = runs / f"skillspan_guideline_{run_id}" / "skillspan_guideline_score.json"
            if not p.is_file():
                ok = False
                break
            sc = json.loads(p.read_text(encoding="utf-8"))["models"]
            series[version] = [sc[m]["expected_spans_found"] for m in models]
        if ok:
            made.append(grouped_bars(
                [short[m] for m in models], series,
                f"Expected spans found, by model and prompt version (batch {batch})",
                out / f"{exp.lower()}_prompt_versions_batch{batch}.png",
                f"DEVELOPMENT · {exp} · SkillSpan sentences annotated under the guideline · not a finding",
                ylabel=f"expected spans found (of {total})", ymax=total * 1.12,
                reference=(total, f"all {total} expected spans")))
    p = REPO_ROOT / "DATA" / "fivetag" / "adjudicator_eval" / "result.json"
    if p.is_file():
        from gold_standard.metrics import wilson_ci
        res = json.loads(p.read_text(encoding="utf-8"))["results"]
        labels, vals, ivs = [], [], []
        for m in sorted(res, key=lambda m: -sum(res[m]["all"].values())):
            k, n = sum(res[m]["all"].values()), len(res[m]["all"])
            labels.append(short.get(m, m))
            vals.append(k / n)
            ivs.append(wilson_ci(k, n))
        made.append(interval_dots(labels, vals, ivs,
                                  "How often each model gives the reference tag (329 spans)",
                                  out / "exp-020_adjudicator_accuracy.png",
                                  "DEVELOPMENT · EXP-020 · only 4 of the 329 spans are real D1/D2 conflicts · not a finding",
                                  xlabel="share correct, with 95% interval", xlim=(0.8, 1.0)))
    p = REPO_ROOT / "docs" / "experiments" / "data" / "exp015_kaggle_runs.json"
    if p.is_file():
        d = json.loads(p.read_text(encoding="utf-8"))
        c = next(r for r in d["runs"] if r.get("dev_strict_f1_by_epoch"))
        made.append(lines([str(i) for i in range(1, len(c["dev_strict_f1_by_epoch"]) + 1)],
                          {"dev strict F1": c["dev_strict_f1_by_epoch"]},
                          "E1 training rehearsal on SkillSpan: dev score after each epoch",
                          out / "exp-015_dev_f1_by_epoch.png",
                          "REHEARSAL · EXP-015 · XLM-RoBERTa base on SkillSpan, Kaggle T4 · not a study result",
                          ylabel="strict span-F1 on dev", pct=False,
                          subtitle="6 epochs stopped while it was still learning; the best epoch was the 16th"))
        made.append(grouped_bars(
            [f"{r['model']}\n{r['epochs']} epochs" for r in d["runs"]],
            {"dev": [r["dev"]["f1"] for r in d["runs"]], "SkillSpan test": [r["test"]["f1"] for r in d["runs"]]},
            "E1 training rehearsal on SkillSpan: strict span-F1 of the three runs",
            out / "exp-015_runs.png",
            "REHEARSAL · EXP-015 · one seed per run · not a study result",
            ylabel="strict span-F1", ymax=0.72, value_fmt="{:.3f}",
            reference=(d["published"]["test_combined_best"], "published, best model on test")))
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["run", "temporal", "endtoend", "experiments"])
    ap.add_argument("folder", nargs="?")
    args = ap.parse_args()
    fn = {"run": figures_for_run, "temporal": figures_for_temporal,
          "endtoend": figures_for_endtoend}.get(args.what)
    made = figures_for_experiments() if fn is None else fn(Path(args.folder))
    for p in made:
        print(p)
    print(f"{len(made)} figures")


if __name__ == "__main__":
    main()
