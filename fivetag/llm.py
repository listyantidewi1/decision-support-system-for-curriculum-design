"""E4 and E5 — the two LLM extraction conditions.

  E4  prompted LLM + JSON schema, no retrieval. Its output is taken AS RETURNED:
      a span is used if its text can be found in the sentence, and nothing else
      is checked. This is the condition that measures "kemampuan dasar dan
      risiko unsupported output", so what the model got wrong is counted, not
      cleaned away.

  E5  LLM + evidence retrieval + JSON schema + deterministic verifier. Before
      the sentence, the model is shown the k most similar sentences from the
      TRAINING split with their gold spans; its answer then goes through
      `labeling.verify`, and only verified spans are kept.

Both reuse the prompt pack and the OpenRouter caller of `labeling/`, so E4 is the
same request the silver-label ensemble makes — which means E4 can be scored
from an ensemble run already paid for (`from_stored_run`), at no cost.

Spending: `run_llm` has the same hard cap as `labeling.run`. It refuses to start
when the worst case exceeds `max_usd`. Nothing in this module makes a paid call
on import or as a side effect of scoring.

Leakage: the retrieval pool must be the training split. `Retriever` refuses a
pool row marked `"split": "test"`, and never returns the query sentence itself.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Mapping, Sequence

from labeling.prompts import DEFAULT_PROMPT_VERSION, build_messages
from labeling.schema import normalise_label
from labeling.verify import verify_output

from .base import Pred, dedupe, make_pred

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "DATA" / "fivetag" / "runs"
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


# -------------------------------------------------------------- E4: as returned
def parse_unverified(raw: str | None, sentence: str, source: str = "E4") -> tuple[list[Pred], dict]:
    """(spans usable as returned, what was wrong with the output).

    The stats are the "Struktur LLM" and "Grounding" rows of Tabel 3.7: JSON
    validity, schema conformance, missing fields, unsupported spans.
    """
    st = {"json_valid": False, "schema_ok": False, "n_returned": 0, "n_located": 0,
          "n_unsupported": 0, "n_bad_label": 0, "n_missing_field": 0}
    if not raw or not raw.strip():
        return [], st
    try:
        obj = json.loads(_FENCE.sub("", raw.strip()))
    except (ValueError, TypeError):
        return [], st
    st["json_valid"] = True
    spans = obj.get("spans") if isinstance(obj, dict) else None
    if not isinstance(spans, list):
        return [], st
    preds = []
    st["n_returned"] = len(spans)
    for sp in spans:
        if not isinstance(sp, dict) or not isinstance(sp.get("text"), str) or not sp["text"] \
                or "label" not in sp:
            st["n_missing_field"] += 1
            continue
        label = normalise_label(sp["label"])
        if label is None:
            st["n_bad_label"] += 1
            continue
        text, a, b = sp["text"], sp.get("start"), sp.get("end")
        if isinstance(a, int) and isinstance(b, int) and sentence[a:b] == text:
            start = a
        else:
            n = sentence.count(text)
            if n == 0:
                st["n_unsupported"] += 1
                continue
            occ = sp.get("occurrence")
            start = -1
            for _ in range(occ if isinstance(occ, int) and 1 <= occ <= n else 1):
                start = sentence.index(text, start + 1)
        st["n_located"] += 1
        preds.append(make_pred(sentence, start, start + len(text), label, source=source,
                               rule="as-returned"))
    st["schema_ok"] = st["n_missing_field"] == 0 and st["n_bad_label"] == 0
    return dedupe(preds), st


def parse_verified(raw: str | None, sentence: str, sentence_id: str = "", model: str = "",
                   source: str = "E5") -> tuple[list[Pred], dict]:
    """Spans that passed the deterministic verifier, and its verdict."""
    vo = verify_output(model, sentence_id, sentence, raw or "")
    preds = [make_pred(sentence, s.start, s.end, s.label, source=source, rule="verified")
             for s in vo.spans]
    cats: dict[str, int] = {}
    for r in vo.rejections:
        cats[r.category] = cats.get(r.category, 0) + 1
    return dedupe(preds), {"status": vo.status, "n_verified": len(vo.spans), "rejections": cats}


def summarise_structure(stats: Sequence[dict]) -> dict:
    """Rates over a set of `parse_unverified` stats."""
    n = len(stats)
    ret = sum(s["n_returned"] for s in stats)
    rate = lambda k, d: (k / d) if d else float("nan")     # noqa: E731
    return {"n_outputs": n,
            "json_validity": rate(sum(s["json_valid"] for s in stats), n),
            "schema_conformance": rate(sum(s["schema_ok"] for s in stats), n),
            "n_spans_returned": ret,
            "unsupported_span_rate": rate(sum(s["n_unsupported"] for s in stats), ret),
            "missing_field_rate": rate(sum(s["n_missing_field"] for s in stats), ret),
            "bad_label_rate": rate(sum(s["n_bad_label"] for s in stats), ret)}


def from_stored_run(run_dir: Path, model: str, sentences: Mapping[str, str],
                    verified: bool = False) -> tuple[dict[str, list[Pred]], dict]:
    """Predictions of one model from a finished `labeling.run` — free.

    `verified=False` is E4 (as returned). `verified=True` is the same answers
    after the verifier: E4 plus the verifier, without retrieval — the ablation
    between E4 and E5. Sentences the run has no answer for are returned empty
    and counted under `n_missing`.
    """
    run_dir = Path(run_dir)
    sha = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["request_sha256"]
    raw: dict[str, dict] = {}
    for line in (run_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("model") == model and r.get("request_sha256") == sha:
                raw[r["sentence_id"]] = r
    preds, stats, missing = {}, [], 0
    for sid, text in sentences.items():
        rec = raw.get(sid)
        if not rec or not rec.get("ok"):
            preds[sid], missing = [], missing + 1
            continue
        if verified:
            preds[sid], _ = parse_verified(rec.get("raw"), text, sid, model, source="E4+verifier")
        else:
            preds[sid], st = parse_unverified(rec.get("raw"), text)
            stats.append(st)
    info = {"run_dir": str(run_dir), "model": model, "request_sha256": sha,
            "verified": verified, "n_missing": missing}
    if not verified:
        info["structure"] = summarise_structure(stats)
    return preds, info


# ---------------------------------------------------------------- E5: retrieval
class Retriever:
    """Nearest annotated sentences by character n-gram TF-IDF. Deterministic:
    no model download, no randomness, ties broken by pool order."""

    def __init__(self, pool: Sequence[dict], ngram: tuple[int, int] = (2, 4)):
        from sklearn.feature_extraction.text import TfidfVectorizer
        bad = [r.get("sentence_id") for r in pool if r.get("split") == "test"]
        if bad:
            raise ValueError(f"retrieval pool contains sealed test rows: {bad[:3]}")
        if not pool:
            raise ValueError("empty retrieval pool")
        self.pool = list(pool)
        self.ngram = ngram
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, lowercase=True)
        self._matrix = self._vec.fit_transform([r["text"] for r in self.pool])

    def top_k(self, sentence: str, k: int = 4, exclude_id: str | None = None) -> list[dict]:
        import numpy as np
        sims = (self._matrix @ self._vec.transform([sentence]).T).toarray().ravel()
        order = np.argsort(-sims, kind="stable")
        out = []
        for i in order:
            r = self.pool[int(i)]
            if r.get("sentence_id") == exclude_id or r["text"] == sentence:
                continue
            out.append(r)
            if len(out) >= k:
                break
        return out

    def manifest(self) -> dict:
        h = hashlib.sha256()
        for r in self.pool:
            h.update(json.dumps([r.get("sentence_id"), r["text"], r.get("spans")],
                                sort_keys=True, ensure_ascii=False).encode("utf-8"))
        return {"retriever": "tfidf-char_wb", "ngram": list(self.ngram),
                "pool_size": len(self.pool), "pool_sha256": h.hexdigest()}


def _demo_answer(row: dict) -> str:
    text, spans, seen = row["text"], [], {}
    for s in sorted(row.get("spans", ()), key=lambda s: (s["start"], s["end"])):
        t = text[s["start"]:s["end"]]
        entry = {"text": t, "label": s["label"]}
        if text.count(t) > 1:
            entry["occurrence"] = text.count(t, 0, s["start"]) + 1
        seen[t] = True
        spans.append(entry)
    return json.dumps({"spans": spans}, ensure_ascii=False)


def build_e5_messages(sentence: str, demos: Sequence[dict],
                      prompt_version: str = DEFAULT_PROMPT_VERSION) -> list[dict]:
    """System prompt, then each retrieved example as a question/answer pair,
    then the sentence. The examples are evidence of how the experts tagged
    similar text; the instructions are unchanged from E4."""
    base = build_messages(sentence, "", "", prompt_version)
    msgs = [base[0]]
    for d in demos:
        msgs.append(build_messages(d["text"], "", "", prompt_version)[1])
        msgs.append({"role": "assistant", "content": _demo_answer(d)})
    msgs.append(base[1])
    return msgs


# ------------------------------------------------------------------- live runs
def run_llm(sentences: Mapping[str, str], *, condition: str, model: str, prices: dict,
            max_usd: float, out_dir: Path, prompt_version: str = DEFAULT_PROMPT_VERSION,
            retriever: Retriever | None = None, k: int = 4, max_tokens: int = 600,
            reasoning: dict | None = None, retries: int = 1,
            client_factory: Callable | None = None, log=print) -> dict:
    """Ask one model about every sentence, under a hard spending cap.

    Returns {"preds", "info"}. Answers are cached in `out_dir/raw.jsonl` by the
    exact request, so a second call costs nothing.
    """
    from labeling.run import (BudgetError, RunConfig, _call_cost, _default_client_factory,
                              _tokens_upper_bound, call_model)
    if condition not in ("E4", "E5"):
        raise ValueError("condition must be E4 or E5")
    if condition == "E5" and retriever is None:
        raise ValueError("E5 needs a retriever built on the training split")
    out_dir = Path(out_dir).resolve()
    if (REPO_ROOT / "DATA" / "gold_standard").resolve() in out_dir.parents:
        raise ValueError("model output must not be written under DATA/gold_standard/: "
                         "annotators stay blind to it")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = RunConfig(models=(model,), prices={model: prices}, max_usd=max_usd,
                    max_tokens=max_tokens, retries=retries, reasoning=reasoning,
                    prompt_version=prompt_version)
    raw_path = out_dir / "raw.jsonl"
    cache: dict[tuple[str, str], dict] = {}
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                cache[(r["sentence_id"], r["request_sha256"])] = r

    plan = []
    for sid, text in sentences.items():
        if condition == "E5":
            messages = build_e5_messages(text, retriever.top_k(text, k, exclude_id=sid),
                                         prompt_version)
        else:
            messages = build_messages(text, "", "", prompt_version)
        sha = hashlib.sha256(json.dumps(
            [cfg.request_sha256(), condition, model, messages],
            sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        plan.append((sid, text, messages, sha))

    p = cfg.prices[model]

    def worst(messages) -> float:
        return ((_tokens_upper_bound(messages) * p["usd_per_m_in"]
                 + max_tokens * p["usd_per_m_out"]) / 1e6) * (1 + retries)

    todo = [x for x in plan if (x[0], x[3]) not in cache or not cache[(x[0], x[3])].get("ok")]
    need = sum(worst(m) for _, _, m, _ in todo)
    if need > max_usd:
        raise BudgetError(f"worst case ${need:.4f} for {len(todo)} calls exceeds the cap "
                          f"${max_usd:.2f}")
    log(f"{condition} {model}: {len(todo)} calls to make, {len(plan) - len(todo)} cached, "
        f"worst case ${need:.4f}")
    spent = 0.0
    if todo:
        client = (client_factory or _default_client_factory)(model)
        with raw_path.open("a", encoding="utf-8", newline="\n") as fh:
            for sid, _text, messages, sha in todo:
                if spent + worst(messages) > max_usd:
                    raise BudgetError(f"stopped at ${spent:.4f}: the next call could pass "
                                      f"the cap ${max_usd:.2f}")
                rec = call_model(client, model, messages, cfg)
                rec.update(model=model, sentence_id=sid, request_sha256=sha, condition=condition)
                spent += _call_cost(rec, model, cfg)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                cache[(sid, sha)] = rec

    preds, stats, failed = {}, [], 0
    for sid, text, _messages, sha in plan:
        rec = cache.get((sid, sha))
        if not rec or not rec.get("ok"):
            preds[sid], failed = [], failed + 1
            continue
        if condition == "E4":
            preds[sid], st = parse_unverified(rec.get("raw"), text)
            stats.append(st)
        else:
            preds[sid], st = parse_verified(rec.get("raw"), text, sid, model)
            stats.append(st)
    info = {"condition": condition, "model": model, "prompt_version": prompt_version,
            "n_sentences": len(plan), "n_calls_made": len(todo), "n_failed": failed,
            "charged_usd": spent, "max_usd": max_usd,
            "retrieval": retriever.manifest() | {"k": k} if condition == "E5" else None}
    if condition == "E4":
        info["structure"] = summarise_structure(stats)
    else:
        info["verifier"] = {"n_invalid_outputs": sum(s["status"] != "VALID" for s in stats),
                            "n_verified_spans": sum(s["n_verified"] for s in stats)}
    (out_dir / f"summary_{condition}.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"preds": preds, "info": info}


class StoredPredictions:
    """Predictions computed earlier, behind the `Extractor` interface. Lets the
    router use LLM answers that were already paid for, by sentence text."""

    def __init__(self, name: str, sentences: Mapping[str, str],
                 preds: Mapping[str, Sequence[Pred]], info: dict | None = None):
        self.name = name
        self._by_text = {sentences[sid]: list(p) for sid, p in preds.items() if sid in sentences}
        self.info = info or {}

    def has(self, sentence: str) -> bool:
        return sentence in self._by_text

    def extract(self, sentence: str) -> list[Pred]:
        if sentence not in self._by_text:
            raise KeyError("no stored prediction for this sentence")
        return list(self._by_text[sentence])

    def manifest(self) -> dict:
        return {"extractor": self.name, "stored": True, **self.info}
