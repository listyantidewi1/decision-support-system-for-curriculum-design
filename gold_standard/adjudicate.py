"""Adjudication: from three annotators' work to one gold standard.

    # 1. make the worksheet the researcher fills in during the adjudication sessions
    .venv-ml\\Scripts\\python.exe -m gold_standard.adjudicate worksheet --split main_1200

    # 2. turn the filled worksheet into the versioned gold standard
    .venv-ml\\Scripts\\python.exe -m gold_standard.adjudicate finalise --split main_1200 ^
        --worksheet <file.xlsx> --version 1.0

`gold_standard.iaa` already separates what all three annotators agreed on from
what is disputed. This module is the missing step after it: it lays the disputes
out as a spreadsheet (guideline Formulir B), and afterwards reads the decisions
back into the table that `gold_standard.seal` consumes.

What is and is not decided automatically:

  * A span all three annotators marked identically (same start, end and tag) is
    gold. Nobody has to confirm it.
  * EVERYTHING ELSE needs a recorded decision. A disputed span becomes gold only
    if the worksheet says so. Nothing is filled in by majority unless the run is
    started with --accept-suggestions, and then that fact is written into the
    manifest so the method can be reported truthfully.
  * A dispute can also be ruled AMBIGUOUS or EXCLUDE. Its annotator variants are
    then kept, marked as such, so that the sealed-test scorer can avoid charging
    a model for reproducing a span the adjudicators themselves could not settle.

Fail-closed checks at finalise — each stops the run and lists every offender:
an unresolved row; a final text that is not in the sentence (or is there twice
without a start position); an unknown tag; a NON_RELEVANT sentence carrying a
gold span (guideline 3.2); two gold spans with the same tag that overlap
(guideline 6.9); a worksheet made from annotation files that have since changed.

A gold version is written once. The output directory must not exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import ANNOTATORS, GUIDELINE_CODE, HAS_5TAG, RELEVANCE, TAGS
from .iaa import AnnSet, build_adjudication

REPO_ROOT = Path(__file__).resolve().parents[1]
GS_DIR = REPO_ROOT / "DATA" / "gold_standard"

GOLD, AMBIGUOUS, EXCLUDE = "GOLD", "AMBIGUOUS", "EXCLUDE"
DECISIONS = (GOLD, AMBIGUOUS, EXCLUDE)
REASONS = ("REL", "BOUND", "LABEL", "MISS", "EXTRA", "SCHEMA", "LANG", "OVERLAP")
KIND_NAMES = {"LABEL": "same span, different tag", "BOUND": "same tag, different boundaries",
              "PRESENCE": "not everyone marked it", "ADDED": "added by the adjudicators"}

SENT_HEAD = ["sentence_id", "sentence", "relevance AV", "relevance AI", "relevance AD",
             "suggested relevance", "FINAL relevance", "five-tag AV", "five-tag AI",
             "five-tag AD", "suggested five-tag", "FINAL five-tag", "guideline conflict",
             "note"]
SPAN_HEAD = ["dispute_id", "sentence_id", "sentence", "kind", "AV", "AI", "AD",
             "suggested text", "suggested tag", "DECISION", "final text", "final tag",
             "final start", "reason", "note"]


class AdjudicationError(Exception):
    """The worksheet cannot be turned into a gold standard as it stands."""


# ------------------------------------------------------------------ helpers
def fingerprint(annset: AnnSet) -> str:
    """Identity of an annotation set, so a worksheet cannot be applied to
    annotations that changed after it was made."""
    rows = []
    for code in sorted(annset):
        for sid in sorted(annset[code]):
            a = annset[code][sid]
            rows.append([code, sid, a.relevance, a.has_5tag,
                         sorted([s.start, s.end, s.label] for s in a.spans)])
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest()


def unanimous_spans(annset: AnnSet, sid: str) -> list[tuple[int, int, str]]:
    """Spans every annotator marked identically. Computed from the annotations
    directly, because the provisional gold empties this list when the majority
    relevance is NON_RELEVANT and the adjudicators may overrule that."""
    coders = list(annset)
    groups: dict[tuple[int, int, str], set[str]] = {}
    for c in coders:
        a = annset[c].get(sid)
        for s in (a.spans if a else []):
            groups.setdefault((s.start, s.end, s.label), set()).add(c)
    return sorted(k for k, who in groups.items() if len(who) == len(coders))


def _suggest(dispute: dict) -> tuple[int, int, str] | None:
    """The variant at least two annotators wrote identically, if there is one."""
    counts = Counter((s["start"], s["end"], s["label"]) for s in dispute["spans"])
    by_coder: dict[tuple[int, int, str], set[str]] = {}
    for s in dispute["spans"]:
        by_coder.setdefault((s["start"], s["end"], s["label"]), set()).add(s["coder"])
    best = [k for k, who in by_coder.items() if len(who) >= 2]
    return best[0] if len(best) == 1 and counts else None


def _majority(votes: dict[str, str], allowed) -> str:
    c = Counter(v for v in votes.values() if v in allowed)
    if not c:
        return ""
    top, n = c.most_common(1)[0]
    return top if n >= 2 else ""


def disputes_of(annset: AnnSet, text_by_sid: dict[str, str]) -> tuple[list[dict], dict]:
    """(dispute rows, provisional gold) with stable dispute ids."""
    adj = build_adjudication(annset, text_by_sid)
    adj = {**adj, "gold": adj["provisional_gold"]}
    rows = []
    for q in adj["queue"]:
        for k, d in enumerate(q.get("span_disputes", []), 1):
            rows.append({"dispute_id": f"{q['sentence_id']}::{k}",
                         "sentence_id": q["sentence_id"], **d})
    return rows, adj


# ---------------------------------------------------------------- worksheet
def make_worksheet(annset: AnnSet, text_by_sid: dict[str, str], path: Path,
                   split: str) -> dict:
    from openpyxl import Workbook
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
    from openpyxl.styles import Alignment, Font

    def clean(v):
        return ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v

    coders = list(ANNOTATORS)
    disputes, adj = disputes_of(annset, text_by_sid)
    wb = Workbook()

    ws = wb.active
    ws.title = "sentences"
    ws.append(SENT_HEAD)
    n_sent = 0
    for q in adj["queue"]:
        rel, tag = q.get("relevance_by_coder", {}), q.get("has_5tag_by_coder", {})
        if (q.get("relevance_agreement") == "unanimous"
                and q.get("has_5tag_agreement") == "unanimous"
                and not q.get("guideline_conflicts")):
            continue
        n_sent += 1
        ws.append([clean(v) for v in [
            q["sentence_id"], text_by_sid.get(q["sentence_id"], q.get("sentence_text", "")),
            *[rel.get(c, "") for c in coders], _majority(rel, RELEVANCE), "",
            *[tag.get(c, "") for c in coders], _majority(tag, HAS_5TAG), "",
            "; ".join(q.get("guideline_conflicts", [])), ""]])

    sp = wb.create_sheet("spans")
    sp.append(SPAN_HEAD)
    for d in disputes:
        text = text_by_sid.get(d["sentence_id"], "")
        cells = {c: " ; ".join(f"{text[s['start']:s['end']]} [{s['label']}]"
                               for s in d["spans"] if s["coder"] == c) for c in coders}
        sug = _suggest(d)
        sp.append([clean(v) for v in [
            d["dispute_id"], d["sentence_id"], text, KIND_NAMES.get(d["type"], d["type"]),
            *[cells[c] for c in coders],
            text[sug[0]:sug[1]] if sug else "", sug[2] if sug else "",
            "", "", "", "", "", ""]])

    info = wb.create_sheet("read me")
    fp = fingerprint(annset)
    for line in [
        f"Adjudication worksheet - {split} - guideline {GUIDELINE_CODE}",
        f"annotation fingerprint: {fp}",
        "",
        "Sheet 'sentences': one row for every sentence where relevance or the five-tag",
        "answer was not unanimous. Write the decision in the two FINAL columns.",
        f"  relevance: {' / '.join(RELEVANCE)}      five-tag: {' / '.join(HAS_5TAG)}",
        "",
        "Sheet 'spans': one row for every disputed span. Fill in:",
        f"  DECISION    {' / '.join(DECISIONS)}",
        "  final text  the exact span, copied from the sentence (only for GOLD)",
        f"  final tag   {' / '.join(TAGS)} (only for GOLD)",
        "  final start only if the final text occurs more than once in the sentence:",
        "              the position of its first character, counting from 0",
        f"  reason      {' / '.join(REASONS)} (Formulir B)",
        "",
        "To accept the suggestion in a row, write GOLD and leave final text and tag empty.",
        "AMBIGUOUS = the adjudicators could not settle it. EXCLUDE = it should not be",
        "annotated at all. Either way no gold span is created for that row.",
        "If one dispute should become two gold spans, add a row below with the same",
        "sentence_id, kind 'added by the adjudicators', DECISION GOLD, and its text and tag.",
        "",
        "Spans that all three annotators marked identically are already gold and are",
        "not listed here.",
    ]:
        info.append([line])
    info.column_dimensions["A"].width = 100

    for sheet, widths in ((ws, (24, 60, 14, 14, 14, 14, 16, 10, 10, 10, 12, 14, 30, 30)),
                          (sp, (28, 24, 60, 24, 28, 28, 28, 26, 9, 12, 28, 9, 9, 10, 30))):
        for c in sheet[1]:
            c.font = Font(bold=True)
        for i, w in enumerate(widths):
            sheet.column_dimensions[chr(ord("A") + i)].width = w
        for row in sheet.iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
        sheet.freeze_panes = "A2"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return {"path": str(path), "n_sentence_rows": n_sent, "n_span_rows": len(disputes),
            "fingerprint": fp}


def _read_sheet(wb, name: str, head: list[str]) -> list[dict]:
    if name not in wb.sheetnames:
        raise AdjudicationError(f"the worksheet has no '{name}' sheet")
    rows = list(wb[name].iter_rows(values_only=True))
    if not rows or [str(c or "") for c in rows[0]][:len(head)] != head:
        raise AdjudicationError(f"sheet '{name}' does not have the expected columns; "
                                f"do not rename or reorder them")
    out = []
    for n, r in enumerate(rows[1:], 2):
        d = {h: ("" if v is None else str(v).strip()) for h, v in zip(head, r)}
        if any(d.values()):
            d["_row"] = n
            out.append(d)
    return out


def _locate(sentence: str, text: str, start: str, envelope: tuple[int, int] | None):
    """(start, end) of `text` in `sentence`, or a problem string."""
    if not text:
        return "the final text is empty"
    if start != "":
        try:
            i = int(float(start))
        except ValueError:
            return f"final start {start!r} is not a number"
        if sentence[i:i + len(text)] != text:
            return f"the sentence does not contain {text!r} at position {i}"
        return i, i + len(text)
    occ, i = [], sentence.find(text)
    while i != -1:
        occ.append(i)
        i = sentence.find(text, i + 1)
    if not occ:
        return f"{text!r} is not in the sentence (it must be copied exactly)"
    if len(occ) > 1 and envelope:
        near = [o for o in occ if o < envelope[1] and o + len(text) > envelope[0]]
        if len(near) == 1:
            occ = near
    if len(occ) > 1:
        return f"{text!r} occurs {len(occ)} times; give its position in 'final start'"
    return occ[0], occ[0] + len(text)


# ------------------------------------------------------------------ finalise
def finalise(annset: AnnSet, text_by_sid: dict[str, str], worksheet: Path, out_dir: Path,
             version: str, split: str, accept_suggestions: bool = False,
             provenance: dict | None = None) -> dict:
    from openpyxl import load_workbook
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise AdjudicationError(f"{out_dir} already exists. A gold version is written "
                                f"once; choose a new --version.")
    wb = load_workbook(worksheet, read_only=True, data_only=True)
    try:
        info = " ".join(str(r[0] or "") for r in wb["read me"].iter_rows(values_only=True)) \
            if "read me" in wb.sheetnames else ""
        sent_list = _read_sheet(wb, "sentences", SENT_HEAD)
        span_rows = _read_sheet(wb, "spans", SPAN_HEAD)
    finally:
        wb.close()          # a read-only workbook keeps the file locked until closed
    fp = fingerprint(annset)
    if f"annotation fingerprint: {fp}" not in info:
        raise AdjudicationError(
            "this worksheet was made from different annotation files than the ones "
            "given now (the annotations changed, or it belongs to another split). "
            "Make a fresh worksheet and carry the decisions over.")

    sent_rows = {r["sentence_id"]: r for r in sent_list}
    disputes, adj = disputes_of(annset, text_by_sid)
    dispute_by_id = {d["dispute_id"]: d for d in disputes}
    problems: list[str] = []
    n_suggestions = 0

    # ---- sentence level
    sentences = {}
    for sid, g in adj["gold"].items():
        rel, tag, source = g["relevance"], g["has_5tag"], "agreed"
        row = sent_rows.get(sid)
        if row is not None:
            source = "adjudicated"
            for field, final_col, sug_col, allowed in (
                    ("rel", "FINAL relevance", "suggested relevance", RELEVANCE),
                    ("tag", "FINAL five-tag", "suggested five-tag", HAS_5TAG)):
                value = row[final_col].upper()
                if not value and accept_suggestions and row[sug_col]:
                    value, source = row[sug_col], "suggestion accepted"
                    n_suggestions += 1
                if value not in allowed:
                    problems.append(f"sentences row {row['_row']} ({sid}): {final_col} is "
                                    f"{row[final_col]!r}; expected one of {list(allowed)}")
                    value = ""
                if field == "rel":
                    rel = value
                else:
                    tag = value
        elif not rel or not tag:
            problems.append(f"{sid}: no agreed relevance/five-tag and no worksheet row")
        sentences[sid] = {"sentence_id": sid, "relevance": rel, "has_5tag": tag,
                          "source": source}

    # ---- spans
    spans: list[dict] = []
    for sid in adj["gold"]:
        for (a, b, t) in unanimous_spans(annset, sid):
            spans.append({"sentence_id": sid, "span_start": a, "span_end": b, "tag": t,
                          "verdict": GOLD, "source": "unanimous", "dispute_id": "",
                          "reason": ""})
    seen_disputes = set()
    for r in span_rows:
        where = f"spans row {r['_row']} ({r['dispute_id'] or r['sentence_id']})"
        sid = r["sentence_id"]
        if sid not in text_by_sid:
            problems.append(f"{where}: unknown sentence_id")
            continue
        d = dispute_by_id.get(r["dispute_id"])
        added = d is None
        if added and r["dispute_id"]:
            problems.append(f"{where}: dispute_id is not one of this worksheet's disputes; "
                            f"leave it empty for an added row")
            continue
        if not added:
            if r["dispute_id"] in seen_disputes:
                problems.append(f"{where}: this dispute appears twice; use an added row "
                                f"(empty dispute_id) for a second span")
                continue
            seen_disputes.add(r["dispute_id"])
        decision = r["DECISION"].upper()
        sug = None if added else _suggest(d)
        if not decision and accept_suggestions and sug:
            decision = GOLD
        if decision not in DECISIONS:
            problems.append(f"{where}: DECISION is {r['DECISION']!r}; expected one of "
                            f"{list(DECISIONS)}")
            continue
        reason = r["reason"].upper()
        if reason and reason not in REASONS:
            problems.append(f"{where}: reason {r['reason']!r} is not one of {list(REASONS)}")
        if decision == GOLD:
            text, tag, source = r["final text"], r["final tag"].upper(), "adjudicated"
            if not text and not tag and sug:
                a, b, tag = sug
                source = "suggestion accepted"
                n_suggestions += 1
            else:
                env = None if added else (min(s["start"] for s in d["spans"]),
                                          max(s["end"] for s in d["spans"]))
                got = _locate(text_by_sid[sid], text, r["final start"], env)
                if isinstance(got, str):
                    problems.append(f"{where}: {got}")
                    continue
                a, b = got
            if tag not in TAGS:
                problems.append(f"{where}: final tag {r['final tag']!r} is not one of "
                                f"{list(TAGS)}")
                continue
            spans.append({"sentence_id": sid, "span_start": a, "span_end": b, "tag": tag,
                          "verdict": GOLD, "source": source,
                          "dispute_id": r["dispute_id"], "reason": reason})
        else:
            if added:
                problems.append(f"{where}: an added row must have DECISION GOLD")
                continue
            for (a, b, t) in sorted({(s["start"], s["end"], s["label"]) for s in d["spans"]}):
                spans.append({"sentence_id": sid, "span_start": a, "span_end": b, "tag": t,
                              "verdict": decision, "source": "adjudicated",
                              "dispute_id": r["dispute_id"], "reason": reason})
    for did in sorted(set(dispute_by_id) - seen_disputes):
        problems.append(f"dispute {did} has no row in the worksheet")

    # ---- consistency of the result
    gold_spans = [s for s in spans if s["verdict"] == GOLD]
    keys = Counter((s["sentence_id"], s["span_start"], s["span_end"], s["tag"])
                   for s in gold_spans)
    for k, n in keys.items():
        if n > 1:
            problems.append(f"{k[0]}: the gold span [{k[1]}, {k[2]}) {k[3]} was created "
                            f"{n} times")
    by_sid: dict[str, list[dict]] = {}
    for s in gold_spans:
        by_sid.setdefault(s["sentence_id"], []).append(s)
    for sid, group in by_sid.items():
        if sentences.get(sid, {}).get("relevance") == "NON_RELEVANT":
            problems.append(f"{sid}: the sentence is NON_RELEVANT but has a gold span "
                            f"(guideline 3.2). Change the relevance or the span decision.")
        if sentences.get(sid, {}).get("has_5tag") == "TIDAK":
            problems.append(f"{sid}: five-tag is TIDAK but the sentence has a gold span")
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                x, y = group[i], group[j]
                if (x["tag"] == y["tag"]
                        and max(x["span_start"], y["span_start"])
                        < min(x["span_end"], y["span_end"])
                        and (x["span_start"], x["span_end"]) != (y["span_start"], y["span_end"])):
                    problems.append(f"{sid}: two gold {x['tag']} spans overlap "
                                    f"(guideline 6.9 allows overlap only between "
                                    f"different tags)")
    for sid, s in sentences.items():
        if s["has_5tag"] == "YA" and sid not in by_sid:
            dropped = [x for x in spans if x["sentence_id"] == sid and x["verdict"] != GOLD]
            if not dropped:
                problems.append(f"{sid}: five-tag is YA but the sentence has no gold span")

    if problems:
        raise AdjudicationError(
            f"{len(problems)} problem(s); nothing was written:\n  " + "\n  ".join(problems))

    # ---- write, once
    out_dir.mkdir(parents=True)
    span_df = pd.DataFrame(spans, columns=["sentence_id", "span_start", "span_end", "tag",
                                           "verdict", "source", "dispute_id", "reason"])
    span_df = span_df.drop_duplicates(
        subset=["sentence_id", "span_start", "span_end", "tag", "verdict"]).sort_values(
        ["sentence_id", "span_start", "span_end", "tag"], kind="stable").reset_index(drop=True)
    sent_df = pd.DataFrame(sorted(sentences.values(), key=lambda d: d["sentence_id"]))
    span_df.to_parquet(out_dir / "gold_spans.parquet", index=False)
    sent_df.to_parquet(out_dir / "gold_sentences.parquet", index=False)

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    manifest = {
        "gold_version": version, "split": split, "guideline": GUIDELINE_CODE,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "annotators": sorted(annset),
        "annotation_fingerprint": fp,
        "worksheet": {"file": str(worksheet), "sha256": sha(Path(worksheet))},
        "accept_suggestions": accept_suggestions,
        "n_decisions_taken_from_suggestions": n_suggestions,
        "n_sentences": len(sent_df),
        "n_sentences_adjudicated": int((sent_df["source"] != "agreed").sum()),
        "n_spans_by_verdict": {k: int(v) for k, v in
                               span_df["verdict"].value_counts().items()},
        "n_gold_spans_by_source": {k: int(v) for k, v in span_df[
            span_df["verdict"] == GOLD]["source"].value_counts().items()},
        "n_gold_spans_by_tag": {k: int(v) for k, v in span_df[
            span_df["verdict"] == GOLD]["tag"].value_counts().items()},
        "n_disputes": len(disputes),
        "files": {"gold_spans.parquet": sha(out_dir / "gold_spans.parquet"),
                  "gold_sentences.parquet": sha(out_dir / "gold_sentences.parquet")},
        "provenance": provenance or {},
    }
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


# ----------------------------------------------------------------------- CLI
def _load(split: str, ann_dir: Path | None):
    from .iaa import ANN_DIR, _quarantine_marker, _text_by_sid, load_annotator_set
    text = _text_by_sid(split)
    if not text:
        raise SystemExit(f"the sample for {split} is missing ({GS_DIR / (split + '.parquet')})")
    resolved = (Path(ann_dir) if ann_dir else ANN_DIR).resolve()
    annset = load_annotator_set(split, "json", Path(ann_dir) if ann_dir else None, text)
    if len(annset) < len(ANNOTATORS):
        raise SystemExit(f"need all {len(ANNOTATORS)} annotators' files for {split}; found "
                         f"{sorted(annset)} in {resolved}")
    foreign = sorted({s for d in annset.values() for s in d} - set(text))
    if foreign:
        raise SystemExit(f"{len(foreign)} annotated sentence id(s) are not in the {split} "
                         f"sample, e.g. {foreign[:3]}")
    return annset, text, resolved, _quarantine_marker(resolved)


def _refuse_evidence_tree(quarantined, out: Path) -> None:
    out = Path(out).resolve()
    if quarantined and "_synthetic" not in out.parts and GS_DIR.resolve() in out.parents:
        raise SystemExit(
            f"refusing: the annotations are quarantined as synthetic ({quarantined}), so "
            f"nothing made from them may be written into {GS_DIR} outside a '_synthetic' "
            f"folder. Give an output path somewhere else.")


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worksheet")
    w.add_argument("--split", required=True)
    w.add_argument("--ann-dir", default=None)
    w.add_argument("--out", default=None)
    f = sub.add_parser("finalise")
    f.add_argument("--split", required=True)
    f.add_argument("--ann-dir", default=None)
    f.add_argument("--worksheet", required=True)
    f.add_argument("--version", required=True)
    f.add_argument("--out-dir", default=None)
    f.add_argument("--accept-suggestions", action="store_true",
                   help="where a row is left blank and two annotators agreed exactly, "
                        "take their answer. Recorded in the manifest.")
    args = ap.parse_args()
    annset, text, resolved, quarantined = _load(args.split, args.ann_dir)

    if args.cmd == "worksheet":
        out = Path(args.out) if args.out else GS_DIR / "adjudication" / \
            f"adjudication_{args.split}.xlsx"
        _refuse_evidence_tree(quarantined, out)
        if out.exists():
            raise SystemExit(f"{out} already exists; it may hold decisions. Move it aside "
                             f"or give another --out.")
        r = make_worksheet(annset, text, out, args.split)
        print(f"{r['n_sentence_rows']} sentence rows and {r['n_span_rows']} span rows -> "
              f"{r['path']}")
        return

    out_dir = Path(args.out_dir) if args.out_dir else GS_DIR / "gold" / args.split / \
        f"v{args.version}"
    _refuse_evidence_tree(quarantined, out_dir)
    try:
        m = finalise(annset, text, Path(args.worksheet), out_dir, args.version, args.split,
                     args.accept_suggestions,
                     {"ann_dir": str(resolved), "input_quarantined": quarantined})
    except AdjudicationError as exc:
        raise SystemExit(f"NOT FINALISED: {exc}")
    print(f"gold standard {args.split} v{args.version}: {m['n_sentences']} sentences, "
          f"spans by verdict {m['n_spans_by_verdict']}")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
