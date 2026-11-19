"""Tests for the two bridges from the finished gold to the models:

  gold_standard.materialise   seal the test labels, export train/dev for training
  labeling.versus_gold        score the LLM ensemble against the gold

Standalone runner (the repo has no pytest):

    .venv-ml\\Scripts\\python.exe tests\\test_gold_to_models.py

Everything is built in temporary folders from invented sentences. Nothing here
reads or writes the real DATA/ tree.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_standard import materialise as mat                               # noqa: E402
from labeling import versus_gold as vg                                     # noqa: E402

SPLIT_ID = "grouped_random_v1"
TEXT = {
    "tr1": "Build web applications using React.",
    "tr2": "Our office is located in Jakarta.",
    "dv1": "Experience with Docker and good teamwork.",
    "te1": "Knowledge of design patterns.",
    "te2": "Familiar with Kubernetes.",
}
PARTS = {"train": ["tr1", "tr2"], "dev": ["dv1"], "test_blind": ["te1", "te2"]}


def _span(sid, text, tag, verdict="GOLD"):
    a = TEXT[sid].index(text)
    return {"sentence_id": sid, "span_start": a, "span_end": a + len(text), "tag": tag,
            "verdict": verdict, "source": "t", "dispute_id": "", "reason": ""}


GOLD_ROWS = [
    _span("tr1", "Build web applications", "SH"), _span("tr1", "React", "TT"),
    _span("dv1", "Docker", "TT"), _span("dv1", "teamwork", "SS"),
    _span("dv1", "good teamwork", "WB", "AMBIGUOUS"),
    _span("te1", "design patterns", "KN"), _span("te2", "Kubernetes", "TT"),
]


def _env(gold_rows=GOLD_ROWS, parts=PARTS):
    root = Path(tempfile.mkdtemp(prefix="g2m_"))
    splits, gs = root / "splits", root / "gs"
    d = splits / SPLIT_ID
    d.mkdir(parents=True)
    gs.mkdir()
    for name, sids in parts.items():
        pd.DataFrame({"sentence_id": sids, "sentence_text": [TEXT[s] for s in sids]}).to_parquet(
            d / f"{name}.parquet", index=False)
    test_ids = parts["test_blind"]
    (d / "SEAL.json").write_text(json.dumps({
        "split_id": SPLIT_ID, "regime_id": "grouped_random", "assignment_sha256": "a" * 64,
        "test_sentence_id_list_sha256": hashlib.sha256(
            "\n".join(sorted(test_ids)).encode("utf-8")).hexdigest(),
        "labels_materialised": False, "opened_utc": None}, indent=2), encoding="utf-8")
    gold = root / "gold" / "v1"
    gold.mkdir(parents=True)
    pd.DataFrame(gold_rows).to_parquet(gold / "gold_spans.parquet", index=False)
    return root, splits, gs, gold


def _refused(fn, exc):
    try:
        fn()
    except exc:
        return True
    return False


# ------------------------------------------------------------------ materialise
def test_materialise_seals_test_and_exports_only_train_and_dev():
    root, splits, _gs, gold = _env()
    m = mat.materialise(SPLIT_ID, gold, splits_dir=splits, export_root=root / "export")
    out = Path(m["out_dir"])
    assert sorted(p.name for p in out.iterdir()) == ["MANIFEST.json", "dev.jsonl", "train.jsonl"]
    assert m["n_sentences"] == {"train": 2, "dev": 1, "test_sealed": 2}
    assert m["test_exported"] is False
    exported = (out / "train.jsonl").read_text(encoding="utf-8") \
        + (out / "dev.jsonl").read_text(encoding="utf-8")
    assert "design patterns" not in exported and "Kubernetes" not in exported
    labels = pd.read_parquet(splits / SPLIT_ID / "labels" / "test_labels.parquet")
    assert sorted(labels["sentence_id"]) == ["te1", "te2"]
    seal = json.loads((splits / SPLIT_ID / "SEAL.json").read_text(encoding="utf-8"))
    assert seal["labels_materialised"] is True


def test_exported_rows_are_what_the_encoder_reads():
    from fivetag.encoder import check_rows
    from fivetag.synthetic import read_jsonl
    root, splits, _gs, gold = _env()
    m = mat.materialise(SPLIT_ID, gold, splits_dir=splits, export_root=root / "export")
    train = read_jsonl(Path(m["out_dir"]) / "train.jsonl")
    dev = read_jsonl(Path(m["out_dir"]) / "dev.jsonl")
    check_rows(train)
    check_rows(dev)
    by = {r["sentence_id"]: r for r in train + dev}
    assert [(r["text"][s["start"]:s["end"]], s["label"]) for r in [by["tr1"]]
            for s in r["spans"]] == [("Build web applications", "SH"), ("React", "TT")]
    assert by["tr2"]["spans"] == [] and by["tr2"]["split"] == "train"
    # an AMBIGUOUS span is not a label, and the sentence says so instead of vanishing
    assert [s["label"] for s in by["dv1"]["spans"]] == ["TT", "SS"]
    assert by["dv1"]["n_challenge_spans"] == 1
    assert m["n_sentences_with_challenge_spans"] == {"train": 0, "dev": 1}


def test_materialise_refuses_gold_from_another_sample_and_writes_nothing():
    foreign = GOLD_ROWS + [{**GOLD_ROWS[0], "sentence_id": "zz9"}]
    root, splits, _gs, gold = _env(gold_rows=foreign)
    assert _refused(lambda: mat.materialise(SPLIT_ID, gold, splits_dir=splits,
                                            export_root=root / "export"), mat.MaterialiseError)
    assert not (root / "export").exists()
    assert not (splits / SPLIT_ID / "labels").exists()


def test_materialise_refuses_overlapping_parts_and_a_span_that_does_not_fit():
    parts = {**PARTS, "dev": ["dv1", "te1"]}
    root, splits, _gs, gold = _env(parts=parts)
    assert _refused(lambda: mat.materialise(SPLIT_ID, gold, splits_dir=splits,
                                            export_root=root / "export"), mat.MaterialiseError)
    bad = [dict(r) for r in GOLD_ROWS]
    bad[0]["span_end"] = 999
    root, splits, _gs, gold = _env(gold_rows=bad)
    assert _refused(lambda: mat.materialise(SPLIT_ID, gold, splits_dir=splits,
                                            export_root=root / "export"), mat.MaterialiseError)
    assert not (splits / SPLIT_ID / "labels").exists()


def test_materialise_never_exports_under_the_gold_standard_tree():
    root, splits, _gs, gold = _env()
    target = mat.REPO_ROOT / "DATA" / "gold_standard" / "_export_test_should_not_exist"
    assert _refused(lambda: mat.materialise(SPLIT_ID, gold, splits_dir=splits,
                                            export_root=target), mat.MaterialiseError)
    assert not target.exists()
    assert not (splits / SPLIT_ID / "labels").exists()      # refused BEFORE sealing


# ------------------------------------------------------------------ versus gold
MODELS = ["fam-a/m", "fam-b/m", "fam-c/m"]
PILOT = ["tr1", "tr2", "dv1"]


def _vg_env(answers: dict, gold_rows=None, sample_ids=PILOT, splits_test=()):
    root = Path(tempfile.mkdtemp(prefix="vg_"))
    gs, gold, run = root / "gs", root / "gold" / "v1", root / "run"
    for d in (gs, gold, run):
        d.mkdir(parents=True)
    pd.DataFrame({"sentence_id": sample_ids,
                  "sentence_text": [TEXT[s] for s in sample_ids]}).to_parquet(
        gs / "pilot_100.parquet", index=False)
    rows = gold_rows if gold_rows is not None else [r for r in GOLD_ROWS
                                                    if r["sentence_id"] in sample_ids]
    pd.DataFrame(rows).to_parquet(gold / "gold_spans.parquet", index=False)
    pd.DataFrame({"sentence_id": sample_ids}).to_parquet(gold / "gold_sentences.parquet",
                                                         index=False)
    (gold / "MANIFEST.json").write_text(json.dumps({"split": "pilot_100"}), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps({
        "request_sha256": "sha", "prompt": {"prompt_version": "v-test"},
        "run_config": {"models": MODELS}}), encoding="utf-8")
    with (run / "raw.jsonl").open("w", encoding="utf-8") as fh:
        for (model, sid), spans in answers.items():
            fh.write(json.dumps({"model": model, "sentence_id": sid, "request_sha256": "sha",
                                 "ok": True, "raw": json.dumps({"spans": spans})}) + "\n")
    splits = root / "splits"
    if splits_test:
        (splits / "x").mkdir(parents=True)
        pd.DataFrame({"sentence_id": list(splits_test)}).to_parquet(
            splits / "x" / "test_blind.parquet", index=False)
    return root, gs, gold, run, splits


RIGHT = {"tr1": [{"text": "Build web applications", "label": "SH"},
                 {"text": "React", "label": "TT"}],
         "tr2": [],
         "dv1": [{"text": "Docker", "label": "TT"}, {"text": "teamwork", "label": "SS"}]}


def _answers(per_model: dict) -> dict:
    return {(m, sid): spans for m, table in per_model.items() for sid, spans in table.items()}


def test_versus_gold_scores_each_model_and_the_ensemble():
    wrong = {**RIGHT, "tr2": [{"text": "Jakarta", "label": "TT"}],          # a false positive
             "dv1": [{"text": "Docker", "label": "TT"}]}                    # one miss
    _root, gs, gold, run, splits = _vg_env(_answers({MODELS[0]: RIGHT, MODELS[1]: RIGHT,
                                                     MODELS[2]: wrong}))
    r = vg.compare(run, gold, splits_dir=splits, gs_dir=gs)
    assert r["n_sentences"] == 3 and r["gold"]["n_gold_spans"] == 4
    assert r["gold"]["n_challenge_spans_not_scored"] == 1
    assert r["models"][MODELS[0]]["strict"]["f1"] == 1.0
    c = r["models"][MODELS[2]]["strict"]
    assert (c["tp"], c["n_ref"], c["n_hyp"]) == (3, 4, 4)
    # unanimous acceptance keeps only what all three said: 3 spans, all gold
    u = r["ensemble"]["unanimous_only"]
    assert u["automatic_acceptance_precision"]["k"] == 3 == u["automatic_acceptance_precision"]["n"]
    assert u["strict"]["recall"] == 0.75
    assert r["ensemble"]["with_majority"]["strict"]["recall"] == 1.0
    assert len(r["pairs"]) == 3
    text = vg.to_text(r)
    assert "automatic acceptance" in text and "SYNTHETIC" not in text


def test_a_missing_answer_is_scored_as_empty_not_skipped():
    partial = {k: v for k, v in RIGHT.items() if k != "dv1"}
    _root, gs, gold, run, splits = _vg_env(_answers({MODELS[0]: RIGHT, MODELS[1]: RIGHT,
                                                     MODELS[2]: partial}))
    r = vg.compare(run, gold, splits_dir=splits, gs_dir=gs)
    assert r["run"]["n_sentences_without_answer"][MODELS[2]] == 1
    assert r["models"][MODELS[2]]["strict"]["recall"] == 0.5
    assert r["models"][MODELS[2]]["strict"]["n_sentences"] == 3


def test_versus_gold_refuses_sealed_test_sentences():
    _root, gs, gold, run, splits = _vg_env(_answers({m: RIGHT for m in MODELS}),
                                           splits_test=("dv1",))
    assert _refused(lambda: vg.compare(run, gold, splits_dir=splits, gs_dir=gs),
                    vg.VersusGoldError)


def test_versus_gold_refuses_a_run_over_other_sentences_and_a_gold_that_does_not_fit():
    other = {(m, "zz"): [] for m in MODELS}
    _root, gs, gold, run, splits = _vg_env(other)
    assert _refused(lambda: vg.compare(run, gold, splits_dir=splits, gs_dir=gs),
                    vg.VersusGoldError)
    bad = [dict(r) for r in GOLD_ROWS if r["sentence_id"] in PILOT]
    bad[0]["span_end"] = 999
    _root, gs, gold, run, splits = _vg_env(_answers({m: RIGHT for m in MODELS}), gold_rows=bad)
    assert _refused(lambda: vg.compare(run, gold, splits_dir=splits, gs_dir=gs),
                    vg.VersusGoldError)


def test_a_rehearsal_gold_is_labelled_synthetic_everywhere():
    root, gs, gold, run, splits = _vg_env(_answers({m: RIGHT for m in MODELS}))
    quarantined = root / "_synthetic" / "gold" / "v1"
    quarantined.parent.mkdir(parents=True)
    gold.rename(quarantined)
    r = vg.compare(run, quarantined, splits_dir=splits, gs_dir=gs)
    assert r["gold"]["synthetic"] is True
    text = vg.to_text(r)
    assert text.startswith("SYNTHETIC GOLD") and text.count("[SYNTHETIC]") >= 5


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
