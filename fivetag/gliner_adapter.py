"""E3 — GLiNER open-type extraction: "kemampuan open-label extraction tanpa LLM besar".

GLiNER takes label DESCRIPTIONS in plain words at prediction time, so it can be
run with no training (zero-shot) and, later, after fine-tuning on the gold
training split. The descriptions below are the experiment's only free
parameter in the zero-shot setting; they are versioned for that reason.

Status (2026-11-19): run against the real model. `gliner` 0.2.30 is installed
in `.venv-ml` and the multilingual checkpoint (`urchade/gliner_multi-v2.1`,
about 1.2 GB) is in the Hugging Face cache on drive D. Three things learned on
the way, all handled or recorded here:

  * `GLiNER.from_pretrained` returns an object of a specific subclass; the
    prediction method lives on that object, not on the `GLiNER` class.
  * Loading by name downloads EVERY file of the model repository, including a
    second copy of the weights. On a slow connection that stalled for hours.
    So a complete local copy is used when there is one (`_local_snapshot`).
  * Its tokenizer needs `protobuf` >= 7.36.1; with an older one the error
    message asks for `tiktoken`, which is misleading.

With `flat_ner=False` the model returns overlapping spans of the SAME label
("good teamwork" and "teamwork", both soft skill). The guideline allows one span
per tag at a place (rule 6.4), so within a tag the higher-scored span is kept.
Spans of different tags may still overlap (rule 6.9).

The unit tests use a stand-in object so they need no download; `extract` raises
a clear error if the real model ever returns entities of another shape.
"""
from __future__ import annotations

from pathlib import Path

from .base import Pred, dedupe, make_pred

LABELS_VERSION = "gliner-labels-v1"
DEFAULT_MODEL = "urchade/gliner_multi-v2.1"

# tag -> the description GLiNER is asked for
LABEL_PROMPTS: dict[str, str] = {
    "SH": "technical task or hard skill",
    "SS": "soft skill",
    "KN": "knowledge area or concept",
    "TT": "tool or technology",
    "WB": "work attitude or personal trait",
}


def _local_snapshot(model_name: str) -> str | None:
    """Folder of an already downloaded copy that has its weights, or None."""
    try:
        from huggingface_hub import snapshot_download
        folder = Path(snapshot_download(model_name, local_files_only=True))
    except Exception:                                          # noqa: BLE001
        return None
    has_weights = any((folder / f).is_file() for f in ("model.safetensors", "pytorch_model.bin"))
    return str(folder) if has_weights and (folder / "gliner_config.json").is_file() else None


class GlinerExtractor:
    name = "E3"

    def __init__(self, model=None, model_name: str = DEFAULT_MODEL, threshold: float = 0.5,
                 label_prompts: dict[str, str] | None = None):
        self.model_name, self.threshold = model_name, threshold
        self.label_prompts = dict(label_prompts or LABEL_PROMPTS)
        self._tag_of = {v: k for k, v in self.label_prompts.items()}
        if model is None:
            try:
                from gliner import GLiNER
            except ImportError as exc:
                raise ImportError("E3 needs the 'gliner' package: "
                                  r".venv-ml\Scripts\pip install gliner") from exc
            model = GLiNER.from_pretrained(_local_snapshot(model_name) or model_name)
        self.model = model

    def extract(self, sentence: str) -> list[Pred]:
        if not sentence or not sentence.strip():
            return []
        ents = self.model.predict_entities(sentence, list(self.label_prompts.values()),
                                           threshold=self.threshold, flat_ner=False)
        out = []
        for e in ents:
            try:
                start, end, label = int(e["start"]), int(e["end"]), self._tag_of[e["label"]]
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"unexpected GLiNER entity {e!r}; check the installed "
                                   f"gliner version against this adapter") from exc
            if 0 <= start < end <= len(sentence):
                out.append(make_pred(sentence, start, end, label,
                                     score=float(e.get("score", 1.0)), source=self.name,
                                     rule="gliner"))
        kept: list[Pred] = []
        for p in sorted(out, key=lambda p: (-p.score, p.start, p.end, p.label)):
            if not any(k.label == p.label and k.start < p.end and p.start < k.end for k in kept):
                kept.append(p)
        return dedupe(kept)

    def manifest(self) -> dict:
        return {"extractor": self.name, "model_name": self.model_name,
                "threshold": self.threshold, "labels_version": LABELS_VERSION,
                "label_prompts": self.label_prompts}
