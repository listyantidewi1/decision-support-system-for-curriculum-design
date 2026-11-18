"""Real SkillSpan sentences, annotated under THIS study's guideline.

    .venv-ml\\Scripts\\python.exe -m labeling.skillspan_guideline export-review
    .venv-ml\\Scripts\\python.exe -m labeling.skillspan_guideline run --max-usd 1.00 --prompt-version v0-guideline-only

The conformance set (`labeling.conformance`) tests rules on made-up sentences.
The researcher asked for the same thing on real text: sentences taken from the
SkillSpan-derived data, with spans and tags that follow Paket B v3.0 rather than
SkillSpan's own two-label scheme.

How the annotations came about, because it limits what they can be used for:

  * They were DRAFTED by an LLM applying the guideline, then put to the
    researcher for review. Until she has reviewed a batch its status is draft.
    An LLM-drafted reference can favour the model family that drafted it, and a
    reviewer who starts from a draft tends to accept it; so, like the
    conformance set, this is for comparing ONE model under two prompts — never
    for ranking models, and never as evidence about the corpus.
  * It is not gold and not the pilot: one reviewer, not three independent
    experts, and the sentences are SkillSpan's, not the study's corpus.
  * Only software-related sentences are used. A large part of the source is from
    other occupations (laboratory, logistics, fundraising), which the study's
    domain filter would never pass to an annotator.
  * The source text has its commas stripped ("scalability efficiency and
    maintainability") and is English only.
  * Only spans the guideline settles clearly are asserted. A sentence may hold
    other taggable material that is deliberately left unscored, so — except for
    items marked closed — an extra span from a model is not an error.

The annotation file holds sentence ids and span texts, never whole sentences;
the sentence is looked up in the local SkillSpan-derived file at load time. Both
that file's redistribution status and this one's are an open licence question
(CLAUDE.md section 6), so the annotation file lives in the gitignored
DATA/labeling/ and only this code is tracked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gold_standard import TAGS

from .conformance import ANY_TAG, Item, export_review, score_run
from .skillspan_probe import SOURCE

REPO_ROOT = Path(__file__).resolve().parents[1]
SET_DIR = REPO_ROOT / "DATA" / "labeling" / "skillspan_guideline"
ANNOTATIONS = SET_DIR / "annotations.json"          # batch 1


def batch_path(batch: int) -> Path:
    """Annotation file of one batch.

    Batches are kept apart because they play different roles. Batch 1 was in
    view when prompt v2 was written, so a gain on it could be the prompt fitting
    those sentences. Batch 2 was drawn AFTER prompt v2 was frozen (2026-11-19):
    it is the check that a gain carries over to text the prompt never saw.
    """
    return ANNOTATIONS if batch == 1 else SET_DIR / f"annotations_batch{batch}.json"

# Boundary conventions the draft applies where the guideline is silent. Each is
# a question for the researcher, repeated in the review workbook.
CONVENTIONS = [
    ("C1", "Coordinated verbs sharing one object are ONE span: 'Write and maintain "
           "technical documentation'. Splitting would leave a bare verb, which rule 6.1 "
           "(shortest span that keeps the meaning) does not allow."),
    ("C2", "A trailing generic 'skills' is left out when what remains is a complete noun "
           "phrase ('communication skills' -> 'communication'), as in the guideline's own "
           "'problem solving' (7.15). It is kept when needed ('interpersonal skills')."),
    ("C3", "The source has no commas. Where a list is evident its items are separate spans "
           "(rule 6.5): 'leadership team building and management skills' -> three spans."),
    ("C4", "Trailing context is dropped, as in guideline example 4.1 ('Develop REST APIs for "
           "internal services' -> 'Develop REST APIs'): 'implement new features in frontend "
           "projects' -> 'implement new features'."),
    ("C5", "Working with or in a team, collaborating, coaching, mentoring are SKILL_SOFT, "
           "following 'Collaborate with product and QA teams' (section 5)."),
    ("C6", "Generic business outcomes ('meet customer needs', 'make a real impact') are not "
           "one of the five tags and are left unmarked (rule 2.6)."),
]


def _source_sentences() -> dict[str, str]:
    out = {}
    with SOURCE.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if line.strip():
                msgs = json.loads(line).get("messages") or []
                if len(msgs) >= 2:
                    out[f"ss-dev-{i:05d}"] = msgs[1].get("content") or ""
    return out


def _locate(sentence: str, entry: dict) -> tuple[int, int]:
    text = entry["text"]
    n = sentence.count(text)
    if n == 1:
        i = sentence.index(text)
    elif n > 1 and isinstance(entry.get("occurrence"), int) and 1 <= entry["occurrence"] <= n:
        i = -1
        for _ in range(entry["occurrence"]):
            i = sentence.index(text, i + 1)
    else:
        raise ValueError(f"{text!r} occurs {n} times in {sentence!r} and no usable "
                         f"'occurrence' was given")
    return i, i + len(text)


def load_items(path: Path = ANNOTATIONS) -> tuple[list[Item], dict]:
    """(items, metadata). Raises if an annotation does not fit its sentence."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    sentences = _source_sentences()
    items = []
    for row in data["items"]:
        sid = row["sentence_id"]
        if sid not in sentences:
            raise KeyError(f"{sid} is not in {SOURCE}")
        sentence = sentences[sid]
        exp = []
        for e in row.get("expected", []):
            if e["tag"] not in TAGS:
                raise ValueError(f"{sid}: unknown tag {e['tag']!r}")
            exp.append((*_locate(sentence, e), e["tag"]))
        forb = []
        for f in row.get("forbidden", []):
            tag = f.get("tag") or ANY_TAG
            if tag != ANY_TAG and tag not in TAGS:
                raise ValueError(f"{sid}: unknown tag {tag!r}")
            forb.append((*_locate(sentence, f), tag))
        closed = bool(row.get("closed"))
        if not exp and not closed:
            raise ValueError(f"{sid}: an item with no expected span must be closed")
        items.append(Item(sid, row.get("conventions", ""), "en", "skillspan-sentence",
                          sentence, tuple(exp), tuple(forb), closed, row.get("note", "")))
    if len({it.item_id for it in items}) != len(items):
        raise ValueError("duplicate sentence ids in the annotation file")
    meta = {k: v for k, v in data.items() if k != "items"}
    return items, meta


def review_readme(items: list[Item], meta: dict) -> list[str]:
    lines = [
        f"SkillSpan sentences annotated under guideline B v3.0 — {len(items)} items.",
        f"Status: {meta.get('review_status', 'DRAFT')}",
        "",
        "These are REAL sentences from the SkillSpan dataset, not from your corpus. The",
        "spans and tags were drafted by an AI applying your guideline. Please check each",
        "one and mark column J with Y or N; corrections go in column K.",
        "",
        "Only spans the guideline settles clearly are listed. A sentence may contain other",
        "taggable text that is deliberately left out and will not be scored.",
        "The 'rule' column lists which of the conventions below the item relies on.",
        "Items whose note starts with CHECK are the ones I am least sure of.",
        "",
        "Conventions applied where the guideline is silent — please confirm or overrule:",
    ]
    lines += [f"  {code}. {text}" for code, text in CONVENTIONS]
    lines += ["", "Left out on purpose: sentences from non-software occupations, and",
              "sentences too garbled by the source's tokenisation to annotate reliably."]
    return lines


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ep = sub.add_parser("export-review")
    ep.add_argument("--batch", type=int, default=1)
    rp = sub.add_parser("run")
    rp.add_argument("--batch", type=int, default=1)
    rp.add_argument("--max-usd", type=float, required=True)
    rp.add_argument("--prompt-version", required=True)
    rp.add_argument("--reasoning", default="none")
    rp.add_argument("--max-tokens", type=int, default=400)
    rp.add_argument("--retries", type=int, default=1)
    sp = sub.add_parser("score")
    sp.add_argument("run_dir")
    sp.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    items, meta = load_items(batch_path(args.batch))
    tag = "" if args.batch == 1 else f"_b{args.batch}"
    if args.cmd == "export-review":
        name = ("skillspan_guideline_REVIEW.xlsx" if args.batch == 1
                else f"skillspan_guideline_batch{args.batch}_REVIEW.xlsx")
        out = export_review(SET_DIR / name, items, review_readme(items, meta))
        print(f"{len(items)} items -> {out}")
        return
    if args.cmd == "run":
        from .run import DEFAULT_MODELS, RUNS_DIR, BudgetError, RunConfig, run
        reasoning = None if args.reasoning == "default" else {"effort": args.reasoning}
        cfg = RunConfig(models=tuple(DEFAULT_MODELS), prices=DEFAULT_MODELS,
                        max_usd=args.max_usd, reasoning=reasoning,
                        max_tokens=args.max_tokens, retries=args.retries,
                        prompt_version=args.prompt_version)
        run_dir = RUNS_DIR / f"skillspan_guideline{tag}_{cfg.request_sha256()[:8]}"
        sentences = [{"sentence_id": it.item_id, "sentence_text": it.sentence,
                      "previous_sentence": "", "next_sentence": ""} for it in items]
        try:
            summary = run(sentences, cfg, run_dir)
        except BudgetError as exc:
            raise SystemExit(f"BUDGET: {exc}")
        print(f"spent ${summary['cost']['charged_usd_total']:.4f}")
    else:
        run_dir = Path(args.run_dir)
    result = score_run(run_dir, items)
    for s in result["models"].values():          # those two describe the conformance set
        s.pop("conformance_version", None)
        s.pop("review_status", None)
    result["reference"] = meta
    (run_dir / "skillspan_guideline_score.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"prompt {result['prompt_version']}   reference: "
          f"{meta.get('review_status', 'DRAFT')}")
    for model, s in result["models"].items():
        print(f"  {model:30} {s['passed']}/{s['n_items']} items pass, "
              f"{s['expected_spans_found']}/{s['expected_spans_total']} expected spans found")


if __name__ == "__main__":
    main()
