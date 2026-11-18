"""Ensemble runner — calls the models, keeps every raw response, never overspends.

    .venv-ml\\Scripts\\python.exe -m labeling.run --split practice_10 --max-usd 0.50

Every model is reached through OpenRouter (`llm_client_router`, forced with the
`openrouter:` prefix — researcher instruction 2026-11-18: no ensemble model goes
through Jatevo).

Three properties matter more than anything else here:

  * **A hard budget.** Before the first call the worst case is computed —
    every remaining call at its full prompt size and its full `max_tokens` — and
    the run REFUSES to start if that exceeds `--max-usd`. The cap therefore
    holds even if a provider reports no cost. A second guard stops mid-run if
    the measured spend reaches the cap anyway.
  * **Raw responses are kept verbatim** in an append-only `raw.jsonl`, keyed by
    (model, sentence, prompt hash). Re-running the same configuration makes no
    call it has already made, so verification and consensus can be redone for
    free and a crash never pays twice.
  * **Blindness.** The annotators must not see LLM output before their own work
    is finished (guideline §2.7; plan: "blind to LLM output"). Runs are written
    under DATA/labeling/, and the runner refuses any output directory inside
    DATA/gold_standard/, which is what the annotation UI and the IAA read.

A failed call is recorded and left as a MISSING output; the consensus then counts
that model as casting no vote. It is never silently dropped from the ensemble.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from . import LABELING_VERSION
from .consensus import ConsensusConfig, decide_sentence, summarise
from .prompts import (DEFAULT_PROMPT_VERSION, PROMPT_VERSIONS, build_messages,
                      prompt_manifest, prompt_sha256)
from .verify import summarise_verification, verify_output

REPO_ROOT = Path(__file__).resolve().parents[1]
GS_DIR = REPO_ROOT / "DATA" / "gold_standard"
RUNS_DIR = REPO_ROOT / "DATA" / "labeling" / "runs"

# Chosen 2026-11-18 for the first smoke test: three different model families,
# as the proposal requires. NOT the frozen ensemble — that is fixed on the
# development gold. Prices are USD per million tokens as listed by OpenRouter on
# that date; they are an upper-bound input to the budget guard, and the measured
# cost comes from each response.
DEFAULT_MODELS: dict[str, dict] = {
    "openai/gpt-5.4-mini": {"usd_per_m_in": 0.75, "usd_per_m_out": 4.50},
    "anthropic/claude-haiku-5.5": {"usd_per_m_in": 0.10, "usd_per_m_out": 0.50},
    "qwen/qwen3.8-flash": {"usd_per_m_in": 0.15, "usd_per_m_out": 0.47},
}


class BudgetError(RuntimeError):
    """The run would exceed, or has reached, its spending cap."""


@dataclass(frozen=True)
class RunConfig:
    models: tuple[str, ...]
    prices: dict                       # model -> {usd_per_m_in, usd_per_m_out}
    max_usd: float
    max_tokens: int = 1500
    temperature: float | None = 0.0
    retries: int = 2
    accept_majority: bool = False
    # OpenRouter's unified `reasoning` object, sent to every model alike, e.g.
    # {"effort": "none"}. None = each provider's default. Hidden reasoning is
    # billed as output and counts against max_tokens: in the first smoke test
    # (2026-11-18) one model spent its whole allowance on it for 4 of 10
    # sentences and returned no answer at all.
    reasoning: dict | None = None
    prompt_version: str = DEFAULT_PROMPT_VERSION

    def __post_init__(self):
        object.__setattr__(self, "models", tuple(self.models))
        missing = [m for m in self.models if m not in self.prices]
        if missing:
            raise ValueError(f"no price given for {missing}; the budget guard needs "
                             f"an upper bound for every model")
        if self.max_usd <= 0:
            raise ValueError("max_usd must be positive")

    def consensus_config(self) -> ConsensusConfig:
        return ConsensusConfig(models=self.models, accept_majority=self.accept_majority)

    def request_sha256(self) -> str:
        """Identity of a request: the prompt AND every parameter that can change
        the answer. It is the cache key, so changing max_tokens or the reasoning
        setting re-asks the models instead of silently reusing old answers."""
        blob = json.dumps({"prompt": prompt_sha256(self.prompt_version),
                           "max_tokens": self.max_tokens,
                           "temperature": self.temperature, "reasoning": self.reasoning,
                           "response_format": "json_object"}, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _tokens_upper_bound(messages: Sequence[dict]) -> int:
    """Deliberately pessimistic prompt-token estimate: one token per 2.5 chars
    (real tokenisers average ~4 for English, ~3 for Indonesian), plus overhead."""
    chars = sum(len(m.get("content") or "") for m in messages)
    return int(chars / 2.5) + 50


def worst_case_usd(n_calls_by_model: dict[str, int], prompt_tokens: int,
                   cfg: RunConfig) -> float:
    total = 0.0
    for model, n in n_calls_by_model.items():
        p = cfg.prices[model]
        per_call = (prompt_tokens * p["usd_per_m_in"]
                    + cfg.max_tokens * p["usd_per_m_out"]) / 1e6
        total += n * per_call * (1 + cfg.retries)       # a retry pays again
    return total


def _as_dict(resp) -> dict:
    if isinstance(resp, dict):
        return resp
    dump = getattr(resp, "model_dump", None)
    return dump() if callable(dump) else {}


def _default_client_factory(model: str):
    from llm_client_router import get_client_for_model
    client, provider = get_client_for_model(f"openrouter:{model}")
    if provider != "openrouter":
        raise RuntimeError(f"expected OpenRouter for {model}, got {provider}")
    return client


def call_model(client, model: str, messages: list[dict], cfg: RunConfig) -> dict:
    """One model, one sentence. Returns a record; never raises on an API error."""
    kwargs = {"model": model, "messages": messages, "max_tokens": cfg.max_tokens,
              "response_format": {"type": "json_object"},
              "extra_body": {"usage": {"include": True}}}
    if cfg.reasoning is not None:
        kwargs["extra_body"]["reasoning"] = dict(cfg.reasoning)
    if cfg.temperature is not None:
        kwargs["temperature"] = cfg.temperature
    sent_temperature = cfg.temperature
    last_error, attempts = None, 0
    t0 = time.time()
    attempt = 0
    while attempt <= cfg.retries:
        attempts += 1
        try:
            resp = _as_dict(client.chat.completions.create(**kwargs))
            choice = (resp.get("choices") or [{}])[0]
            usage = resp.get("usage") or {}
            return {
                "ok": True,
                "raw": (choice.get("message") or {}).get("content"),
                "finish_reason": choice.get("finish_reason"),
                "served_model": resp.get("model"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": ((usage.get("completion_tokens_details") or {})
                                     .get("reasoning_tokens")),
                "reported_cost_usd": usage.get("cost"),
                "temperature_sent": sent_temperature,
                "attempts": attempts,
                "latency_s": round(time.time() - t0, 3),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 — every failure is recorded
            last_error = f"{type(exc).__name__}: {exc}"[:500]
            # some reasoning models reject a non-default temperature; drop it
            # once, visibly, rather than fail the whole model. A rejected
            # request is not billed, so this does not use up a retry.
            if "temperature" in kwargs and "temperature" in str(exc).lower():
                kwargs.pop("temperature")
                sent_temperature = None
                continue
            if attempt < cfg.retries:
                time.sleep(1.5 * (attempt + 1))
            attempt += 1
    return {"ok": False, "raw": None, "finish_reason": None, "served_model": None,
            "prompt_tokens": None, "completion_tokens": None, "reasoning_tokens": None,
            "reported_cost_usd": None, "temperature_sent": sent_temperature,
            "attempts": attempts, "latency_s": round(time.time() - t0, 3),
            "error": last_error}


def _call_cost(rec: dict, model: str, cfg: RunConfig) -> float:
    """Spend to charge against the cap: the larger of what the provider reports
    and what the token counts imply at list price, so an absent or optimistic
    report cannot loosen the guard."""
    p = cfg.prices[model]
    est = ((rec.get("prompt_tokens") or 0) * p["usd_per_m_in"]
           + (rec.get("completion_tokens") or 0) * p["usd_per_m_out"]) / 1e6
    return max(float(rec.get("reported_cost_usd") or 0.0), est)


def _load_raw(path: Path) -> dict[tuple[str, str, str], dict]:
    out: dict[tuple[str, str, str], dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[(r["model"], r["sentence_id"], r.get("request_sha256", ""))] = r
    return out


def run(sentences: Sequence[dict], cfg: RunConfig, out_dir: Path,
        client_factory: Callable[[str], object] = _default_client_factory,
        log: Callable[[str], None] = print) -> dict:
    """Label `sentences` with every configured model, then verify and vote.

    Each sentence is a dict with `sentence_id` and `sentence_text`, and
    optionally `previous_sentence` / `next_sentence`.
    """
    out_dir = Path(out_dir).resolve()
    if GS_DIR.resolve() in [out_dir, *out_dir.parents]:
        raise ValueError(
            f"refusing to write ensemble output under {GS_DIR}: the annotation UI "
            f"and the IAA read that tree, and annotators must stay blind to LLM "
            f"output. Use a directory under {RUNS_DIR}.")
    ids = [s["sentence_id"] for s in sentences]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sentence ids in the input")

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.jsonl"
    sha = cfg.request_sha256()
    have = _load_raw(raw_path)

    todo = [(m, s) for s in sentences for m in cfg.models
            if not have.get((m, s["sentence_id"], sha), {}).get("ok")]
    already_spent = sum(r.get("charged_usd") or 0.0 for r in have.values())

    # ---- hard pre-flight: the cap must hold in the worst case
    prompt_ub = max((_tokens_upper_bound(build_messages(
        s["sentence_text"], s.get("previous_sentence", ""), s.get("next_sentence", ""),
        cfg.prompt_version)) for s in sentences), default=0)
    n_by_model = {m: sum(1 for mm, _ in todo if mm == m) for m in cfg.models}
    worst = worst_case_usd(n_by_model, prompt_ub, cfg)
    log(f"calls to make: {len(todo)} ({len(have)} cached)   "
        f"worst case ${worst:.4f}   cap ${cfg.max_usd:.2f}   "
        f"already spent in this run dir ${already_spent:.4f}")
    if already_spent + worst > cfg.max_usd:
        raise BudgetError(
            f"worst case ${already_spent + worst:.4f} exceeds the cap "
            f"${cfg.max_usd:.2f}; raise --max-usd, lower max_tokens, or label "
            f"fewer sentences. No call was made.")

    clients: dict[str, object] = {}
    spent = already_spent
    with raw_path.open("a", encoding="utf-8") as fh:
        for model, s in todo:
            if spent >= cfg.max_usd:
                raise BudgetError(f"measured spend ${spent:.4f} reached the cap "
                                  f"${cfg.max_usd:.2f}; stopped before the next call.")
            if model not in clients:
                clients[model] = client_factory(model)
            rec = call_model(clients[model], model, build_messages(
                s["sentence_text"], s.get("previous_sentence", ""),
                s.get("next_sentence", ""), cfg.prompt_version), cfg)
            rec.update(model=model, sentence_id=s["sentence_id"],
                       prompt_sha256=prompt_sha256(cfg.prompt_version),
                       prompt_version=cfg.prompt_version, request_sha256=sha,
                       created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            rec["charged_usd"] = _call_cost(rec, model, cfg)
            spent += rec["charged_usd"]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            have[(model, s["sentence_id"], sha)] = rec

    # ---- verify + vote (free: reads only what was stored)
    ccfg = cfg.consensus_config()
    verified, results = [], []
    for s in sentences:
        outs = []
        for m in cfg.models:
            rec = have.get((m, s["sentence_id"], sha))
            # The model answered (and was paid) but may have returned no content
            # — e.g. hidden reasoning used up max_tokens. That is an INVALID
            # output and must sit in the JSON-validity denominator; only a call
            # that failed outright is a missing output.
            if rec and rec.get("ok"):
                outs.append(verify_output(m, s["sentence_id"], s["sentence_text"],
                                          rec.get("raw") or "", ccfg.verify_policy))
        verified.extend(outs)
        results.append(decide_sentence(s["sentence_id"], s["sentence_text"], outs, ccfg))

    with (out_dir / "consensus.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    used = [have[(m, s["sentence_id"], sha)] for s in sentences for m in cfg.models
            if (m, s["sentence_id"], sha) in have]
    per_model = {}
    for m in cfg.models:
        recs = [r for r in used if r["model"] == m]
        per_model[m] = {
            "n_calls": len(recs),
            "n_failed": sum(1 for r in recs if not r.get("ok")),
            "n_truncated": sum(1 for r in recs if r.get("finish_reason") == "length"),
            "n_empty_content": sum(1 for r in recs if r.get("ok") and not r.get("raw")),
            "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in recs),
            "completion_tokens": sum(r.get("completion_tokens") or 0 for r in recs),
            "reasoning_tokens": sum(r.get("reasoning_tokens") or 0 for r in recs),
            "charged_usd": round(sum(r.get("charged_usd") or 0.0 for r in recs), 6),
            "mean_latency_s": round(sum(r.get("latency_s") or 0 for r in recs)
                                    / len(recs), 3) if recs else None,
            "served_models": sorted({str(r.get("served_model")) for r in recs
                                     if r.get("served_model")}),
            "temperature_dropped": any(r.get("temperature_sent") is None
                                       and cfg.temperature is not None for r in recs),
        }
    total = sum(v["charged_usd"] for v in per_model.values())
    n = len(sentences)
    summary = {
        "labeling_version": LABELING_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_sentences": n,
        "prompt": prompt_manifest(cfg.prompt_version),
        "request_sha256": sha,
        "run_config": {**asdict(cfg), "models": list(cfg.models)},
        "consensus_config": ccfg.to_dict(),
        "cost": {"charged_usd_total": round(total, 6),
                 "usd_per_1000_sentences": round(1000 * total / n, 4) if n else None,
                 "per_model": per_model},
        "verification": summarise_verification(verified),
        "consensus": summarise(results),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def load_split(split: str) -> list[dict]:
    import pandas as pd
    df = pd.read_parquet(GS_DIR / f"{split}.parquet")
    cols = ["sentence_id", "sentence_text", "previous_sentence", "next_sentence"]
    df = df[[c for c in cols if c in df.columns]].fillna("")
    return df.astype(str).to_dict("records")


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["practice_10", "pilot_100"], default="practice_10")
    ap.add_argument("--max-usd", type=float, required=True,
                    help="hard spending cap for this run, in USD")
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--reasoning", default="default",
                    help="'default' (each provider's own), or an effort sent to every "
                         "model: none | minimal | low | medium | high")
    ap.add_argument("--prompt-version", choices=PROMPT_VERSIONS,
                    default=DEFAULT_PROMPT_VERSION)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    reasoning = None if args.reasoning == "default" else {"effort": args.reasoning}
    cfg = RunConfig(models=tuple(DEFAULT_MODELS), prices=DEFAULT_MODELS,
                    max_usd=args.max_usd, max_tokens=args.max_tokens, reasoning=reasoning,
                    prompt_version=args.prompt_version)
    run_id = args.run_id or f"{args.split}_{cfg.request_sha256()[:8]}"
    out_dir = RUNS_DIR / run_id
    try:
        summary = run(load_split(args.split), cfg, out_dir)
    except BudgetError as exc:
        raise SystemExit(f"BUDGET: {exc}")
    c = summary["cost"]
    print(f"\nspent ${c['charged_usd_total']:.4f}  "
          f"(${c['usd_per_1000_sentences']:.2f} per 1,000 sentences)")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
