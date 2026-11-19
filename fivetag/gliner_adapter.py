"""E3 — GLiNER open-type extraction: "kemampuan open-label extraction tanpa LLM besar".

GLiNER takes label DESCRIPTIONS in plain words at prediction time, so it can be
run with no training (zero-shot) and, later, after fine-tuning on the gold
training split. The descriptions below are the experiment's only free
parameter in the zero-shot setting; they are versioned for that reason.

Status (2026-11-19): the `gliner` package is NOT installed in `.venv-ml`, and
its multilingual checkpoint is a download of several hundred MB. This adapter
is therefore tested against a stand-in object with the same `predict_entities`
method, not against the real model. Before the first real run:

    .venv-ml\\Scripts\\pip install gliner
    python -c "from fivetag.gliner_adapter import GlinerExtractor; \\
               print(GlinerExtractor().extract('Experience with Docker.'))"

and confirm that the entity dictionaries it returns carry `start`, `end`,
`label` and `score` — `extract` raises a clear error if they do not.
"""
from __future__ import annotations

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
            model = GLiNER.from_pretrained(model_name)
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
        return dedupe(out)

    def manifest(self) -> dict:
        return {"extractor": self.name, "model_name": self.model_name,
                "threshold": self.threshold, "labels_version": LABELS_VERSION,
                "label_prompts": self.label_prompts}
