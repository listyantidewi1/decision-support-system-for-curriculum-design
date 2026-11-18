"""Deterministic verifier — nothing a model says is used until it passes here.

Proposal §3.4.3: "Pemeriksaan mencakup validitas struktur JSON, keberadaan span
pada teks sumber, kesesuaian posisi karakter, penggunaan label yang
diperbolehkan, serta pemeriksaan terhadap span yang tidak didukung oleh teks.
Prediksi yang gagal ... tidak langsung digunakan dalam proses voting dan dicatat
sebagai INVALID atau UNSUPPORTED."

There is no model call, no fuzzy matching and no randomness in this file: the
same response and sentence always yield the same verdict. Two things are
deliberately NOT done:

  * No silent repair. A span whose text is not in the sentence is UNSUPPORTED,
    full stop — it is not "corrected" to the nearest similar string, because
    the final span must stay "dapat ditelusuri secara langsung pada teks sumber".
    The two repairs that exist (strip a markdown fence, re-anchor on an offset
    mismatch) are opt-in, off by default, and recorded on the output.
  * No guessing between occurrences. If a model gives only text and the text
    appears twice, it must say which occurrence; otherwise the span is rejected.

Boundary whitespace and edge punctuation are trimmed with the SAME rule the
annotator UI applies to a human selection (`gold_standard.iaa._trim_offsets`),
so a model is not charged a strict-match boundary error for a trailing comma
that the UI would have removed from an expert's span.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass

from gold_standard.iaa import _trim_offsets

from .schema import (INVALID, NOTE, UNSUPPORTED, VALID, Rejection,
                     VerifiedOutput, VerifiedSpan, normalise_label)

OVERLAP_POLICIES = ("allow_cross_label", "forbid_all", "allow_all")

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n?\s*```\s*$", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class VerifyPolicy:
    """Every switch that changes a verdict, so it can be frozen and reported.

    overlap
        `allow_cross_label` (default) follows the validated guideline, rule 6.9:
        overlapping spans are legitimate "apabila fungsinya berbeda" — e.g.
        "Configure Kubernetes clusters" as SH with "Kubernetes" inside it as TT.
        Such spans are kept and flagged OVERLAP_POTENTIAL; two overlapping spans
        with the SAME label are rejected. `forbid_all` is the literal reading of
        the implementation plan ("no overlap") and rejects every overlap.
    reanchor_on_offset_mismatch
        Off: offsets that do not point at the claimed text are UNSUPPORTED.
        On: if the claimed text occurs exactly once, anchor there instead and
        record the repair.
    strip_code_fence
        Off: a response wrapped in ```json ... ``` is INVALID JSON, which is
        what a strict JSON-validity rate should count. On: strip one fence and
        record the repair.
    """
    overlap: str = "allow_cross_label"
    reanchor_on_offset_mismatch: bool = False
    strip_code_fence: bool = False

    def __post_init__(self):
        if self.overlap not in OVERLAP_POLICIES:
            raise ValueError(f"overlap must be one of {OVERLAP_POLICIES}, got {self.overlap!r}")

    def to_dict(self) -> dict:
        return asdict(self)


def _short(obj, limit: int = 200) -> str:
    try:
        s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        s = repr(obj)
    return s if len(s) <= limit else s[:limit] + "…"


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _occurrences(text: str, needle: str) -> list[int]:
    out, i = [], text.find(needle)
    while i != -1:
        out.append(i)
        i = text.find(needle, i + 1)
    return out


def _first_present(item: dict, *keys):
    for k in keys:
        if k in item:
            return True, item[k]
    return False, None


def _anchor_by_text(sentence: str, claimed, item: dict):
    """(start, end, anchor) or a Rejection, locating `claimed` in the sentence."""
    if not isinstance(claimed, str) or not claimed:
        return Rejection("INVALID_ITEM", INVALID, "span",
                         "span has neither usable offsets nor a non-empty text", _short(item))
    occ = _occurrences(sentence, claimed)
    if not occ:
        return Rejection("UNSUPPORTED_TEXT_NOT_FOUND", UNSUPPORTED, "span",
                         "the claimed text does not occur in the source sentence",
                         _short(item))
    if len(occ) == 1:
        return occ[0], occ[0] + len(claimed), "text_unique"
    _, k = _first_present(item, "occurrence")
    if _is_int(k) and 1 <= k <= len(occ):
        return occ[k - 1], occ[k - 1] + len(claimed), "text_occurrence"
    return Rejection("UNSUPPORTED_AMBIGUOUS_TEXT", UNSUPPORTED, "span",
                     f"the claimed text occurs {len(occ)} times and no valid "
                     f"1-based 'occurrence' was given", _short(item))


def _verify_item(sentence: str, item, policy: VerifyPolicy, repairs: list[str]):
    """One raw span -> VerifiedSpan, or a Rejection."""
    if not isinstance(item, dict):
        return Rejection("INVALID_ITEM", INVALID, "span", "span is not a JSON object",
                         _short(item))
    label = normalise_label(item.get("label"))
    if label is None:
        return Rejection("INVALID_LABEL", INVALID, "span",
                         f"label {item.get('label')!r} is not one of the five tags",
                         _short(item))
    has_text, claimed = _first_present(item, "text", "span_text")
    if has_text and claimed is not None and not isinstance(claimed, str):
        return Rejection("INVALID_ITEM", INVALID, "span", "text is not a string",
                         _short(item))
    has_s, start = _first_present(item, "start", "start_char")
    has_e, end = _first_present(item, "end", "end_char")

    if has_s or has_e:
        if not (has_s and has_e and _is_int(start) and _is_int(end)):
            return Rejection("INVALID_OFFSETS", INVALID, "span",
                             "start and end must both be integers", _short(item))
        if not (0 <= start < end <= len(sentence)):
            return Rejection("INVALID_OFFSETS", INVALID, "span",
                             f"offsets [{start}, {end}) fall outside the sentence "
                             f"(length {len(sentence)}) or are empty", _short(item))
        if isinstance(claimed, str) and sentence[start:end] != claimed:
            if not policy.reanchor_on_offset_mismatch:
                return Rejection(
                    "UNSUPPORTED_OFFSET_MISMATCH", UNSUPPORTED, "span",
                    f"the text at [{start}, {end}) is {sentence[start:end]!r}, not the "
                    f"claimed {claimed!r}", _short(item))
            got = _anchor_by_text(sentence, claimed, item)
            if isinstance(got, Rejection):
                return got
            start, end, _ = got
            anchor = "reanchored_by_text"
            repairs.append("reanchored_by_text")
        else:
            anchor = "offsets_and_text" if isinstance(claimed, str) else "offsets_only"
    else:
        got = _anchor_by_text(sentence, claimed, item)
        if isinstance(got, Rejection):
            return got
        start, end, anchor = got

    flags = []
    ts, te = _trim_offsets(sentence, start, end)
    if (ts, te) != (start, end):
        flags.append("TRIMMED")
    if ts >= te:
        return Rejection("INVALID_EMPTY_SPAN", INVALID, "span",
                         "nothing is left of the span after trimming edge "
                         "whitespace and punctuation", _short(item))
    return VerifiedSpan(ts, te, label, sentence[ts:te], anchor, tuple(flags))


def _resolve_within_output(spans: list[VerifiedSpan], policy: VerifyPolicy
                           ) -> tuple[list[VerifiedSpan], list[Rejection]]:
    """Duplicates, contradictory labels and overlaps inside ONE model's answer."""
    rejections: list[Rejection] = []

    # exact duplicates: keep one, log it; harmless, so it does not block NONE
    seen, unique = set(), []
    for s in spans:
        key = (s.start, s.end, s.label)
        if key in seen:
            rejections.append(Rejection(
                "DUPLICATE_SPAN", NOTE, "span", "exact duplicate of an earlier span",
                _short(s.to_dict()), blocks_none=False))
        else:
            seen.add(key)
            unique.append(s)

    # one stretch of text given two different tags: the model contradicts
    # itself, and choosing either tag for it would be our decision, not its.
    by_bounds: dict[tuple[int, int], list[VerifiedSpan]] = {}
    for s in unique:
        by_bounds.setdefault(s.bounds, []).append(s)
    kept = []
    for group in by_bounds.values():
        if len(group) > 1:
            for s in group:
                rejections.append(Rejection(
                    "INVALID_CONFLICTING_LABELS", INVALID, "span",
                    "the same span was given more than one label: "
                    + ", ".join(sorted(g.label for g in group)), _short(s.to_dict())))
        else:
            kept.append(group[0])

    if policy.overlap == "allow_all":
        doomed, flagged = set(), _overlapping_indices(kept, lambda a, b: True)
    elif policy.overlap == "forbid_all":
        doomed, flagged = _overlapping_indices(kept, lambda a, b: True), set()
    else:  # allow_cross_label — guideline rule 6.9
        doomed = _overlapping_indices(kept, lambda a, b: a.label == b.label)
        flagged = _overlapping_indices(kept, lambda a, b: a.label != b.label) - doomed

    out = []
    for i, s in enumerate(kept):
        if i in doomed:
            code = ("INVALID_OVERLAP" if policy.overlap == "forbid_all"
                    else "INVALID_SAME_LABEL_OVERLAP")
            rejections.append(Rejection(
                code, INVALID, "span",
                "overlaps another span from the same model"
                + ("" if policy.overlap == "forbid_all" else " that has the same label"),
                _short(s.to_dict())))
        elif i in flagged and "OVERLAP_POTENTIAL" not in s.flags:
            out.append(VerifiedSpan(s.start, s.end, s.label, s.text, s.anchor,
                                    s.flags + ("OVERLAP_POTENTIAL",)))
        else:
            out.append(s)
    out.sort(key=lambda s: (s.start, s.end, s.label))
    return out, rejections


def _overlapping_indices(spans: list[VerifiedSpan], when) -> set[int]:
    hit: set[int] = set()
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            a, b = spans[i], spans[j]
            if max(a.start, b.start) < min(a.end, b.end) and when(a, b):
                hit.update((i, j))
    return hit


def verify_output(model: str, sentence_id: str, sentence: str, raw,
                  policy: VerifyPolicy | None = None) -> VerifiedOutput:
    """Verify one model's response for one sentence.

    `raw` is the response text, or an already-parsed object when the API
    returned structured output. Never raises on bad model output: every failure
    becomes a logged Rejection, because a crash would lose the evidence.
    """
    policy = policy or VerifyPolicy()
    out = VerifiedOutput(model=model, sentence_id=sentence_id, status=VALID)

    data = raw
    if isinstance(raw, (str, bytes)):
        text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        if policy.strip_code_fence:
            m = _FENCE_RE.match(text)
            if m:
                text = m.group(1)
                out.repairs.append("stripped_code_fence")
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, RecursionError) as exc:
            out.status = INVALID
            out.rejections.append(Rejection(
                "INVALID_JSON", INVALID, "output", f"not valid JSON: {exc}", _short(text)))
            return out

    if not isinstance(data, dict) or not isinstance(data.get("spans"), list):
        out.status = INVALID
        out.rejections.append(Rejection(
            "INVALID_SCHEMA", INVALID, "output",
            'expected a JSON object with a "spans" list', _short(data)))
        return out

    candidates: list[VerifiedSpan] = []
    for item in data["spans"]:
        got = _verify_item(sentence, item, policy, out.repairs)
        if isinstance(got, Rejection):
            out.rejections.append(got)
        else:
            candidates.append(got)

    out.spans, more = _resolve_within_output(candidates, policy)
    out.rejections.extend(more)
    return out


def summarise_verification(outputs: list[VerifiedOutput]) -> dict:
    """Per-model verification rates, each with a Wilson interval.

    These are the LLM-specific measures the proposal lists (§3.6): JSON validity,
    schema conformance and unsupported-output rate. The denominators are stated
    because they differ: validity is per response, unsupported is per span the
    model proposed.
    """
    from gold_standard.metrics import wilson_ci

    by_model: dict[str, list[VerifiedOutput]] = {}
    for o in outputs:
        by_model.setdefault(o.model, []).append(o)

    def rate(k: int, n: int) -> dict:
        lo, hi = wilson_ci(k, n)
        return {"k": k, "n": n, "rate": (k / n) if n else float("nan"),
                "wilson95": [lo, hi]}

    report = {}
    for model in sorted(by_model):
        outs = by_model[model]
        codes = Counter(r.code for o in outs for r in o.rejections)
        n_json_bad = sum(1 for o in outs if any(r.code == "INVALID_JSON" for r in o.rejections))
        n_schema_bad = sum(1 for o in outs
                           if any(r.code == "INVALID_SCHEMA" for r in o.rejections))
        span_rej = [r for o in outs for r in o.rejections
                    if r.level == "span" and r.category != NOTE]
        n_kept = sum(len(o.spans) for o in outs)
        n_proposed = n_kept + len(span_rej)
        report[model] = {
            "n_responses": len(outs),
            "json_valid": rate(len(outs) - n_json_bad, len(outs)),
            "schema_conformant": rate(len(outs) - n_json_bad - n_schema_bad, len(outs)),
            "n_spans_proposed": n_proposed,
            "n_spans_kept": n_kept,
            "span_unsupported": rate(sum(1 for r in span_rej if r.category == UNSUPPORTED),
                                     n_proposed),
            "span_invalid": rate(sum(1 for r in span_rej if r.category == INVALID),
                                 n_proposed),
            "n_responses_repaired": sum(1 for o in outs if o.repairs),
            "rejection_codes": dict(sorted(codes.items())),
        }
    return report
