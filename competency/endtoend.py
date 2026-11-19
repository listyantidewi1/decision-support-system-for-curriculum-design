"""The three end-to-end pipelines and their ablations (proposal §3.3.3, second level).

    P1  NLP and deterministic rules, no LLM
    P2  the full LLM path, with schema constraint, retrieval and the verifier
    P3  the hybrid proposed by the framework: rules first, the LLM only where an
        escalation trigger fires, D1/D2 conflicts adjudicated, statements
        written by the LLM and checked
    P3 without the deterministic verifier        (ablation)
    P3 without the persistence & volatility module (ablation)

All of them run on the SAME postings, and each ends in the same competency
chain (`competency.pipeline.run`), so what differs between two rows of the
comparison is exactly what their definitions say differs.

What is measured for each (the end-to-end metrics the proposal names, as far as
they can be measured before the experts exist): how much evidence and how many
competencies come out, with which statuses; grounding of the statements and the
number below the threshold ("klaim tanpa dukungan"); what the checks rejected;
how much the top of the list overlaps between pipelines; cost and time per
1,000 sentences.

What is NOT measured here, and why:
  * appropriateness of classification and level according to experts — needs
    the experts;
  * temporal stability of the ranking — needs a run per time window on far more
    postings than a paid rehearsal can afford;
  * stability between repeated runs — answers are cached and the temperature is
    0, so a repeat returns the same text; a real test needs fresh calls.

Until the gold standard exists this is a REHEARSAL of the comparison: it proves
the harness runs and shows what the table will look like. The retrieval pool of
P2 and P3 is a stand-in (the reviewed development sentences), because the gold
training split it will really use does not exist yet.

    python -m competency.endtoend --n-postings 40 --max-usd 1.00 --tag rehearsal_1
    python -m competency.endtoend --only P1 --n-postings 400 --tag p1_only     (free)
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from . import VALID
from .contract import ContractConfig
from .evidence import build_evidence
from .pipeline import GS_DIR, RUNS_DIR, run

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTICE = ("REHEARSAL of the end-to-end comparison on a small sample, with a stand-in retrieval "
          "pool and unmeasured extractors. It shows the harness works. Not a finding.")
EXTRACTION_MODEL = "qwen/qwen3.8-flash"
STATEMENT_MODEL = "openai/gpt-5.4-mini"


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    extraction: str            # rules | llm | router
    statements: str            # deterministic | llm
    verifier: bool = True      # deterministic verifier on LLM spans and statements
    persistence: bool = True   # persistence weight orders the recommendations
    description: str = ""


SPECS: dict[str, PipelineSpec] = {s.name: s for s in (
    PipelineSpec("P1", "rules", "deterministic",
                 description="NLP and deterministic rules, no LLM"),
    PipelineSpec("P2", "llm", "llm", description="full LLM path with retrieval and verifier"),
    PipelineSpec("P3", "router", "llm", description="hybrid: rules first, LLM on escalation"),
    PipelineSpec("P3-tanpa-verifier", "router", "llm", verifier=False,
                 description="ablation: P3 without the deterministic verifier"),
    PipelineSpec("P3-tanpa-persistensi", "router", "llm", persistence=False,
                 description="ablation: P3 without the persistence & volatility module"),
)}


def sample_postings(n_postings: int, seed: int = 11, gs_dir: Path = GS_DIR) -> dict[str, dict]:
    """All pool sentences of `n_postings` postings drawn with a seed."""
    import pandas as pd
    df = pd.read_parquet(gs_dir / "sentence_pool.parquet",
                         columns=["sentence_id", "posting_id", "sentence_text"])
    ids = df["posting_id"].drop_duplicates().sample(n=n_postings, random_state=seed)
    sub = df[df["posting_id"].isin(set(ids))]
    return {r.sentence_id: {"text": r.sentence_text, "posting_id": r.posting_id}
            for r in sub.itertuples(index=False)}


def standin_retrieval_pool() -> list[dict]:
    """The reviewed development sentences as demonstration pool. A STAND-IN for
    the gold training split."""
    from fivetag.evaluate import development_sets
    rows = []
    for items in development_sets().values():
        for it in items:
            rows.append({"sentence_id": it.item_id, "text": it.sentence,
                         "spans": [{"start": s, "end": e, "label": t} for s, e, t in it.expected]})
    return rows


class LlmExtraction:
    """LLM spans for sentences, asked once and cached; verified and as-returned
    readings of the same answers."""

    def __init__(self, sentences: Mapping[str, dict], out_dir: Path, max_usd: float,
                 model: str = EXTRACTION_MODEL, client_factory=None, k: int = 3, log=print):
        self.sentences, self.out_dir, self.max_usd = sentences, Path(out_dir), max_usd
        self.model, self.client_factory, self.k, self.log = model, client_factory, k, log
        self.spent, self.n_calls = 0.0, 0
        self._verified: dict[str, list] = {}
        self._raw: dict[str, list] = {}
        self._retriever = None

    def ensure(self, sentence_ids) -> None:
        from fivetag.llm import Retriever, parse_unverified, run_llm
        from labeling.run import DEFAULT_MODELS
        need = {s: self.sentences[s]["text"] for s in sentence_ids if s not in self._verified}
        if not need:
            return
        if self._retriever is None:
            self._retriever = Retriever(standin_retrieval_pool())
        out = run_llm(need, condition="E5", model=self.model, prices=DEFAULT_MODELS[self.model],
                      max_usd=max(1e-9, self.max_usd - self.spent), out_dir=self.out_dir,
                      retriever=self._retriever, k=self.k, max_tokens=300,
                      reasoning={"effort": "none"}, retries=0,
                      client_factory=self.client_factory, log=self.log)
        self.spent += out["info"]["charged_usd"]
        self.n_calls += out["info"]["n_calls_made"]
        self._verified.update(out["preds"])
        raw = {}
        for line in (self.out_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("ok"):
                    raw[r["sentence_id"]] = r.get("raw")
        for sid, text in need.items():
            self._raw[sid] = parse_unverified(raw.get(sid), text)[0]

    def preds(self, sid: str, verified: bool = True) -> list:
        return list((self._verified if verified else self._raw).get(sid, []))


def extract(spec: PipelineSpec, sentences: Mapping[str, dict], llm: LlmExtraction | None,
            lexicon: str = "v2") -> tuple[dict, dict]:
    """(predictions by sentence id, what happened)."""
    from fivetag.e0_rules import E0Rules
    e0 = E0Rules(version=lexicon)
    info: dict = {"extraction": spec.extraction}
    if spec.extraction == "rules":
        return {s: e0.extract(r["text"]) for s, r in sentences.items()}, info
    if llm is None:
        raise ValueError(f"{spec.name} needs an LLM")
    if spec.extraction == "llm":
        llm.ensure(list(sentences))
        info["n_sentences_sent_to_llm"] = len(sentences)
        return {s: llm.preds(s, spec.verifier) for s in sentences}, info
    from fivetag.router import Router, summarise
    router = Router(e0, None)
    first = {s: router.route(r["text"]) for s, r in sentences.items()}
    escalated = [s for s, r in first.items() if r.escalated]
    llm.ensure(escalated)
    preds = {s: (llm.preds(s, spec.verifier) if s in set(escalated) else first[s].preds)
             for s in sentences}
    summary = summarise(list(first.values()))
    info.update(n_sentences_sent_to_llm=len(escalated),
                escalation_rate=summary["escalation_rate"], by_trigger=summary["by_trigger"])
    return preds, info


def resolve_conflicts(evidence: list, spec: PipelineSpec, adjudicate=None) -> tuple[list, dict]:
    """D1/D2 conflicts. Without an LLM (P1) a conflict stays unresolved and its
    span is left out. With one (P3) the adjudicator's label is used; "NONE"
    removes the span. P2 is the pure LLM path and does not use D1/D2."""
    from dataclasses import replace
    from fivetag import conflict_routing as CR
    if spec.extraction == "llm":
        return evidence, {"used": False}
    conflicts = CR.find_conflicts(evidence)
    by_id = {c.evidence_id: c for c in conflicts}
    info = {"used": True, "n_conflicts": len(conflicts),
            "by_kind": CR.summarise(conflicts)["by_kind"]}
    if adjudicate is None:
        info["resolution"] = "left out (no LLM in this pipeline)"
        return [e for e in evidence if e.evidence_id not in by_id], info
    CR.adjudicate_with_llm(conflicts, adjudicate)
    out, changed, removed = [], 0, 0
    for e in evidence:
        c = by_id.get(e.evidence_id)
        if c is None or c.llm_status != "ok":
            out.append(e)
        elif c.llm_label == CR.NONE:
            removed += 1
        elif c.llm_label != e.tag:
            changed += 1
            out.append(replace(e, tag=c.llm_label,
                               evidence_id=f"{e.sentence_id}:{e.start}-{e.end}:{c.llm_label}"))
        else:
            out.append(e)
    info.update(resolution="LLM adjudicator", n_tag_changed=changed, n_removed=removed,
                n_unusable=sum(c.llm_status == "unusable" for c in conflicts))
    seen, unique = set(), []
    for e in out:
        if e.evidence_id not in seen:
            seen.add(e.evidence_id)
            unique.append(e)
    return unique, info


def metrics(report: dict, competencies: list[dict], min_grounding: float) -> dict:
    g = [c["statement"]["grounding"] for c in competencies]
    return {"n_evidence": report["n_evidence"], "n_groups": report["grouping"]["n_groups"],
            "n_valid_groups": report["grouping"]["status_distribution"].get(VALID, 0),
            "n_competencies": len(competencies), "by_status": report["status_distribution"],
            "mean_grounding": sum(g) / len(g) if g else None,
            "n_statements_below_grounding": sum(1 for x in g if x < min_grounding),
            "n_llm_statements_rejected": report["n_llm_statements_rejected"],
            "n_statements_that_would_have_been_rejected": sum(
                1 for c in competencies
                if any("would have been rejected" in n for n in c["statement"]["notes"]))}


def top_overlap(a: list[dict], b: list[dict], k: int = 20) -> float | None:
    """Jaccard overlap of the two top-k lists, matching competencies by the
    phrases they are made of."""
    from .profile import match
    a, b = a[:k], b[:k]
    if not a or not b:
        return None
    m = len(match(a, b))
    return m / (len(a) + len(b) - m)


def run_all(sentences: Mapping[str, dict], out_root: Path, specs=None, *, max_usd: float = 1.0,
            client_factory=None, clusterer=None, mapping: bool = True, embedder=None,
            min_postings: int = 3, posting_month="auto", log=print) -> dict:
    from fivetag.llm import capped_caller
    from labeling.run import DEFAULT_MODELS
    from .temporal import posting_months
    specs = [SPECS[s] if isinstance(s, str) else s for s in (specs or SPECS)]
    out_root = Path(out_root)
    needs_llm = any(s.extraction != "rules" or s.statements == "llm" for s in specs)
    llm = adjudicate = write = None
    if needs_llm:
        llm = LlmExtraction(sentences, out_root / "_llm_extraction", max_usd * 0.75,
                            client_factory=client_factory, log=log)
        adjudicate = capped_caller(EXTRACTION_MODEL, DEFAULT_MODELS[EXTRACTION_MODEL],
                                   max_usd * 0.05, max_tokens=40, reasoning={"effort": "none"},
                                   client_factory=client_factory,
                                   log_path=out_root / "_llm_adjudicator.jsonl")
        cache: dict[str, str | None] = {}
        raw_write = capped_caller(STATEMENT_MODEL, DEFAULT_MODELS[STATEMENT_MODEL],
                                  max_usd * 0.20, max_tokens=200, reasoning={"effort": "none"},
                                  client_factory=client_factory,
                                  log_path=out_root / "_llm_statements.jsonl")

        def write(messages):
            key = json.dumps(messages, sort_keys=True, ensure_ascii=False)
            if key not in cache:
                cache[key] = raw_write(messages)
            return cache[key]
    months = posting_month
    if posting_month == "auto":
        try:
            months = posting_months()
        except Exception:                                      # noqa: BLE001
            months = None
    cfg = ContractConfig(min_postings=min_postings)
    n_sent = len(sentences)
    rows, lists = [], {}
    for spec in specs:
        t0 = time.perf_counter()
        before = (llm.spent if llm else 0.0, adjudicate.spent() if adjudicate else 0.0,
                  raw_write.spent() if needs_llm else 0.0)
        preds, ext = extract(spec, sentences, llm)
        evidence = build_evidence(preds, sentences, source=spec.name)
        evidence, conf = resolve_conflicts(
            evidence, spec, adjudicate if spec.extraction == "router" else None)
        report = run(evidence, sentences, out_root / spec.name, evidence_source=spec.description,
                     contract_cfg=cfg, clusterer=clusterer, mapping=mapping, embedder=embedder,
                     llm_call=write if spec.statements == "llm" else None,
                     check_statements=spec.verifier, use_persistence=spec.persistence,
                     posting_month=months, pipeline_name=spec.name, log=lambda *_: None)
        comps = json.loads((Path(report["out_dir"]) / "competencies.json").read_text("utf-8"))
        lists[spec.name] = comps
        cost = ((llm.spent - before[0]) if llm else 0.0) + (
            (adjudicate.spent() - before[1]) if adjudicate else 0.0) + (
            (raw_write.spent() - before[2]) if needs_llm else 0.0)
        seconds = time.perf_counter() - t0
        row = {"pipeline": spec.name, "spec": asdict(spec), "extraction": ext, "conflicts": conf,
               **metrics(report, comps, cfg.min_grounding),
               "new_cost_usd": cost, "seconds": seconds,
               "seconds_per_1000_sentences": 1000 * seconds / n_sent if n_sent else None}
        rows.append(row)
        log(f"{spec.name}: {row['n_evidence']} evidence, {row['n_competencies']} competencies, "
            f"${cost:.4f}, {seconds:.0f}s")
    names = [r["pipeline"] for r in rows]
    overlap = {f"{a} vs {b}": top_overlap(lists[a], lists[b])
               for i, a in enumerate(names) for b in names[i + 1:]}
    result = {"notice": NOTICE, "n_sentences": n_sent,
              "n_postings": len({r["posting_id"] for r in sentences.values()}),
              "models": {"extraction_and_adjudication": EXTRACTION_MODEL,
                         "statements": STATEMENT_MODEL} if needs_llm else {},
              "retrieval_pool": "stand-in: reviewed development sentences" if needs_llm else "",
              "max_usd": max_usd,
              "total_charged_usd": (llm.spent + adjudicate.spent() + raw_write.spent())
              if needs_llm else 0.0,
              "note_on_cost": ("new_cost_usd is what a pipeline ADDED: answers already paid for "
                               "by an earlier pipeline in this run are reused at no cost, so the "
                               "order of the pipelines matters for this column"),
              "rows": rows, "top20_overlap": overlap,
              "not_measured": ["expert appropriateness", "temporal stability of the ranking",
                               "stability between repeated runs"]}
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "comparison.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return result


def to_text(result: dict) -> str:
    lines = [result["notice"], "",
             f"{result['n_postings']} postings, {result['n_sentences']} sentences; charged "
             f"${result['total_charged_usd']:.4f} of a cap of ${result['max_usd']:.2f}", ""]
    head = (f"{'pipeline':<22}{'evid.':>7}{'groups':>8}{'comp.':>7}{'kand.':>7}{'ground.':>9}"
            f"{'<thr':>6}{'rej.':>6}{'to LLM':>8}{'$ new':>9}{'s/1000':>8}")
    lines += [head, "-" * len(head)]
    for r in result["rows"]:
        g = f"{r['mean_grounding']:.2f}" if r["mean_grounding"] is not None else "-"
        lines.append(
            f"{r['pipeline']:<22}{r['n_evidence']:>7}{r['n_groups']:>8}{r['n_competencies']:>7}"
            f"{r['by_status'].get('kandidat_tinjauan', 0):>7}{g:>9}"
            f"{r['n_statements_below_grounding']:>6}"
            f"{r['n_llm_statements_rejected'] + r['n_statements_that_would_have_been_rejected']:>6}"
            f"{r['extraction'].get('n_sentences_sent_to_llm', 0):>8}{r['new_cost_usd']:>9.4f}"
            f"{r['seconds_per_1000_sentences']:>8.1f}")
    lines += ["", "top-20 overlap between pipelines: " + ", ".join(
        f"{k} {v:.2f}" if v is not None else f"{k} -" for k, v in result["top20_overlap"].items())]
    return "\n".join(lines)


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-postings", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max-usd", type=float, default=1.0)
    ap.add_argument("--only", nargs="+", choices=list(SPECS), help="run only these pipelines")
    ap.add_argument("--no-mapping", action="store_true")
    args = ap.parse_args()
    sentences = sample_postings(args.n_postings, args.seed)
    result = run_all(sentences, RUNS_DIR / f"endtoend_{args.tag}", args.only, max_usd=args.max_usd,
                     mapping=not args.no_mapping)
    print()
    print(to_text(result))
    print(f"\n-> {RUNS_DIR / f'endtoend_{args.tag}'}")


if __name__ == "__main__":
    main()
