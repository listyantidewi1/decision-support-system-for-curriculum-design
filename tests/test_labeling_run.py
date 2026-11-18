"""Tests for labeling.prompts and labeling.run — with a FAKE client.

Standalone runner (the repo has no pytest):

    .venv-ml\\Scripts\\python.exe tests\\test_labeling_run.py

No network, no API key, no spend: every "model" here is a local function. What
is under test is the machinery around the calls — the budget cap, the response
cache, the blindness guard and the hand-off to verify/consensus.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_standard import GUIDELINE_CODE, TAGS                             # noqa: E402
from labeling import prompts, run as runner                                # noqa: E402
from labeling.run import BudgetError, RunConfig                            # noqa: E402

MODELS = ("fam-a/model", "fam-b/model", "fam-c/model")
PRICES = {m: {"usd_per_m_in": 1.0, "usd_per_m_out": 2.0} for m in MODELS}
SENTS = [
    {"sentence_id": "u-aaaaaaaaaaaaaaaa#001",
     "sentence_text": "Build web applications using React.",
     "previous_sentence": "", "next_sentence": ""},
    {"sentence_id": "u-aaaaaaaaaaaaaaaa#002",
     "sentence_text": "Bachelor degree required.",
     "previous_sentence": "Build web applications using React.", "next_sentence": ""},
]
GOOD = {"spans": [{"text": "Build web applications", "label": "SH"},
                  {"text": "React", "label": "TT"}]}


def _raises(exc, fn, *a, **k) -> bool:
    try:
        fn(*a, **k)
    except exc:
        return True
    return False


class FakeClient:
    """Stands in for the OpenAI SDK client. `answer(model, target)` -> content."""

    def __init__(self, model: str, answer, calls: list, cost=0.0001,
                 fail_times: int = 0, reject_temperature: bool = False):
        self.model, self._answer, self._calls = model, answer, calls
        self._cost, self._fail, self._reject_t = cost, fail_times, reject_temperature
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self._calls.append(kw)
        if self._reject_t and "temperature" in kw:
            raise ValueError("Unsupported parameter: temperature")
        if self._fail > 0:
            self._fail -= 1
            raise ConnectionError("boom")
        target = json.loads(kw["messages"][1]["content"])["target_sentence"]
        return {"model": kw["model"] + "-2026",
                "choices": [{"message": {"content": self._answer(self.model, target)},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                          "cost": self._cost}}


def _factory(answer, calls, **kw):
    return lambda model: FakeClient(model, answer, calls, **kw)


def _all_agree(model, target):
    return json.dumps(GOOD if "React" in target else {"spans": []})


def _cfg(**kw) -> RunConfig:
    base = dict(models=MODELS, prices=PRICES, max_usd=5.0, max_tokens=200, retries=0)
    base.update(kw)
    return RunConfig(**base)


# -------------------------------------------------------------------- prompts
def test_prompt_states_every_tag_and_the_output_contract():
    p = prompts.SYSTEM_PROMPT
    for code, full in TAGS.items():
        assert code in p and full in p, code
    assert '{"spans": []}' in p and '"occurrence"' in p
    assert "Never translate" in p
    # the guideline's own nested example, so models may emit what experts may
    assert "Configure Kubernetes clusters" in p


def test_prompt_is_guideline_only_and_carries_no_corpus_text():
    m = prompts.prompt_manifest()
    assert m["n_gold_examples"] == 0 and m["prompt_version"].startswith("v0")
    assert m["guideline_code"] == GUIDELINE_CODE
    # no sentence id of either real shape leaked into the instructions
    import re
    assert not re.search(r"\b(u|in)-[0-9a-f]{16}", prompts.SYSTEM_PROMPT)


def test_prompt_hash_is_stable_and_sensitive():
    a, b = prompts.prompt_sha256(), prompts.prompt_sha256()
    assert a == b and len(a) == 64
    original = prompts.SYSTEM_PROMPT
    try:
        prompts.SYSTEM_PROMPT = original + " "
        assert prompts.prompt_sha256() != a
    finally:
        prompts.SYSTEM_PROMPT = original
    assert prompts.prompt_sha256() == a


def test_messages_carry_target_and_context_separately():
    msgs = prompts.build_messages("Target here.", "Before.", "After.")
    assert [m["role"] for m in msgs] == ["system", "user"]
    user = json.loads(msgs[1]["content"])
    assert user["target_sentence"] == "Target here."
    assert user["previous_sentence_context"] == "Before."
    assert user["next_sentence_context"] == "After."


# --------------------------------------------------------------------- runner
def test_run_end_to_end_with_fake_models():
    calls: list = []
    with tempfile.TemporaryDirectory() as td:
        summary = runner.run(SENTS, _cfg(), Path(td) / "r", _factory(_all_agree, calls),
                             log=lambda s: None)
        assert len(calls) == 6                              # 2 sentences x 3 models
        assert all(c["extra_body"] == {"usage": {"include": True}} for c in calls)
        assert all(c["response_format"] == {"type": "json_object"} for c in calls)
        rows = [json.loads(x) for x in
                (Path(td) / "r" / "consensus.jsonl").read_text("utf-8").splitlines()]
    assert summary["n_sentences"] == 2
    assert summary["consensus"]["status_rates"]["UNANIMOUS"]["k"] == 2
    assert summary["consensus"]["sentences_with_no_candidate"] == {"UNANIMOUS_NONE": 1}
    assert [c["text"] for c in rows[0]["candidates"]] == ["Build web applications", "React"]
    assert abs(summary["cost"]["charged_usd_total"] - 6 * 0.00014) < 1e-9
    assert summary["prompt"]["prompt_sha256"] == prompts.prompt_sha256()
    for m in MODELS:
        assert summary["verification"][m]["json_valid"]["rate"] == 1.0
        assert summary["cost"]["per_model"][m]["served_models"] == [m + "-2026"]


def test_charged_cost_is_the_larger_of_reported_and_list_price():
    """A provider that reports $0 cannot loosen the budget guard."""
    cfg = _cfg()
    rec = {"prompt_tokens": 100, "completion_tokens": 20, "reported_cost_usd": 0.0}
    # list price: (100*1 + 20*2)/1e6 = 0.00014
    assert abs(runner._call_cost(rec, MODELS[0], cfg) - 0.00014) < 1e-12
    rec["reported_cost_usd"] = 0.5
    assert runner._call_cost(rec, MODELS[0], cfg) == 0.5


def test_rerun_makes_no_call_it_already_made():
    calls: list = []
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "r"
        runner.run(SENTS, _cfg(), out, _factory(_all_agree, calls), log=lambda s: None)
        n_first = len(calls)
        again = runner.run(SENTS, _cfg(), out, _factory(_all_agree, calls),
                           log=lambda s: None)
        assert len(calls) == n_first == 6                  # nothing paid twice
        assert again["consensus"]["status_rates"]["UNANIMOUS"]["k"] == 2
        assert len((out / "raw.jsonl").read_text("utf-8").splitlines()) == 6


def test_preflight_refuses_before_any_call_when_worst_case_exceeds_cap():
    calls: list = []
    with tempfile.TemporaryDirectory() as td:
        tight = _cfg(max_usd=0.000001)
        assert _raises(BudgetError, runner.run, SENTS, tight, Path(td) / "r",
                       _factory(_all_agree, calls), lambda s: None)
        assert calls == []                                  # no call was made
        assert not (Path(td) / "r" / "consensus.jsonl").exists()


def test_worst_case_counts_full_max_tokens_and_retries():
    cfg = _cfg(max_tokens=1000, retries=2)
    # per call: (500*1 + 1000*2)/1e6 = 0.0025; 4 calls; x3 for retries = 0.03
    got = runner.worst_case_usd({MODELS[0]: 4}, 500, cfg)
    assert abs(got - 0.03) < 1e-12


def test_midrun_guard_stops_when_measured_spend_reaches_the_cap():
    """Reported cost far above list price: the second guard must trip."""
    calls: list = []
    with tempfile.TemporaryDirectory() as td:
        cfg = _cfg(max_usd=0.05)                            # passes the pre-flight
        assert _raises(BudgetError, runner.run, SENTS, cfg, Path(td) / "r",
                       _factory(_all_agree, calls, cost=0.03), lambda s: None)
        assert len(calls) == 2                              # 0.03 + 0.03 >= 0.05 -> stop
        assert len((Path(td) / "r" / "raw.jsonl").read_text("utf-8").splitlines()) == 2


def test_failed_model_is_recorded_and_casts_no_vote():
    calls: list = []

    def factory(model):
        return FakeClient(model, _all_agree, calls,
                          fail_times=99 if model == MODELS[2] else 0)
    with tempfile.TemporaryDirectory() as td:
        s = runner.run(SENTS[:1], _cfg(), Path(td) / "r", factory, log=lambda x: None)
        row = json.loads((Path(td) / "r" / "consensus.jsonl").read_text("utf-8"))
    assert s["cost"]["per_model"][MODELS[2]]["n_failed"] == 1
    assert row["model_status"][MODELS[2]]["rejection_codes"] == ["MISSING_OUTPUT"]
    assert {c["status"] for c in row["candidates"]} == {"MAJORITY"}   # never UNANIMOUS
    assert all(c["votes"][MODELS[2]] == "NO_VOTE" for c in row["candidates"])


def test_failed_call_is_retried_on_the_next_run_but_successes_are_not():
    calls: list = []
    state = {"fail": 99}

    def factory(model):
        return FakeClient(model, _all_agree, calls,
                          fail_times=state["fail"] if model == MODELS[2] else 0)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "r"
        runner.run(SENTS[:1], _cfg(), out, factory, log=lambda x: None)
        assert len(calls) == 3
        state["fail"] = 0
        s = runner.run(SENTS[:1], _cfg(), out, factory, log=lambda x: None)
        assert len(calls) == 4                              # only the failed one again
        assert s["consensus"]["status_rates"]["UNANIMOUS"]["k"] == 2


def test_temperature_is_dropped_visibly_for_a_model_that_rejects_it():
    calls: list = []
    with tempfile.TemporaryDirectory() as td:
        s = runner.run(SENTS[:1], _cfg(), Path(td) / "r",
                       _factory(_all_agree, calls, reject_temperature=True),
                       log=lambda x: None)
    assert all(v["temperature_dropped"] for v in s["cost"]["per_model"].values())
    assert s["consensus"]["status_rates"]["UNANIMOUS"]["k"] == 2


def test_answer_with_no_content_is_an_invalid_output_not_a_missing_one():
    """Found in the first real run: hidden reasoning used up max_tokens and the
    model returned nothing. That response was paid for and must count against
    the model's JSON-validity rate instead of vanishing from the denominator."""
    calls: list = []

    def answer(model, target):
        return None if model == MODELS[2] else _all_agree(model, target)
    with tempfile.TemporaryDirectory() as td:
        s = runner.run(SENTS[:1], _cfg(), Path(td) / "r", _factory(answer, calls),
                       log=lambda x: None)
        row = json.loads((Path(td) / "r" / "consensus.jsonl").read_text("utf-8"))
    v = s["verification"][MODELS[2]]
    assert (v["json_valid"]["k"], v["json_valid"]["n"]) == (0, 1)
    assert s["cost"]["per_model"][MODELS[2]]["n_empty_content"] == 1
    assert row["model_status"][MODELS[2]]["rejection_codes"] == ["INVALID_JSON"]
    assert all(c["votes"][MODELS[2]] == "NO_VOTE" for c in row["candidates"])


def test_changing_a_request_parameter_asks_again_instead_of_reusing_answers():
    calls: list = []
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "r"
        runner.run(SENTS[:1], _cfg(), out, _factory(_all_agree, calls), log=lambda x: None)
        assert len(calls) == 3
        runner.run(SENTS[:1], _cfg(reasoning={"effort": "none"}), out,
                   _factory(_all_agree, calls), log=lambda x: None)
        assert len(calls) == 6                              # different request -> re-asked
        assert all(c["extra_body"]["reasoning"] == {"effort": "none"} for c in calls[3:])
        runner.run(SENTS[:1], _cfg(max_tokens=999), out, _factory(_all_agree, calls),
                   log=lambda x: None)
        assert len(calls) == 9
        runner.run(SENTS[:1], _cfg(), out, _factory(_all_agree, calls), log=lambda x: None)
        assert len(calls) == 9                              # the first config is still cached
    assert _cfg().request_sha256() != _cfg(reasoning={"effort": "none"}).request_sha256()
    assert _cfg().request_sha256() == _cfg().request_sha256()


def test_runner_refuses_to_write_where_annotators_could_see_it():
    """Blindness: nothing from the ensemble may land in DATA/gold_standard/."""
    calls: list = []
    for bad in (runner.GS_DIR / "annotations", runner.GS_DIR / "iaa" / "x", runner.GS_DIR):
        assert _raises(ValueError, runner.run, SENTS, _cfg(), bad,
                       _factory(_all_agree, calls), lambda s: None)
    assert calls == []
    assert runner.GS_DIR.resolve() not in [runner.RUNS_DIR.resolve(),
                                           *runner.RUNS_DIR.resolve().parents]


def test_config_needs_a_price_for_every_model_and_a_positive_cap():
    assert _raises(ValueError, RunConfig, models=MODELS, prices={}, max_usd=1.0)
    assert _raises(ValueError, RunConfig, models=MODELS, prices=PRICES, max_usd=0)
    assert _raises(ValueError, runner.run, SENTS + SENTS[:1], _cfg(), Path("x"),
                   _factory(_all_agree, []), lambda s: None)       # duplicate ids


def test_default_ensemble_is_three_families_all_via_openrouter():
    fams = {m.split("/")[0] for m in runner.DEFAULT_MODELS}
    assert len(runner.DEFAULT_MODELS) == 3 and len(fams) == 3
    src = (PROJECT_ROOT / "labeling" / "run.py").read_text(encoding="utf-8")
    assert 'get_client_for_model(f"openrouter:{model}")' in src
    assert "jatevo:" not in src


def _run_standalone() -> int:
    """Plain-python runner (the repo has no pytest). Pytest-compatible too."""
    fns = {n: f for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)}
    passed, failed = 0, []
    for name, fn in fns.items():
        try:
            fn()
            passed += 1
            print(f"  PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed"
          + (f", {len(failed)} FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
