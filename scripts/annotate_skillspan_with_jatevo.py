"""Annotate prepared job sentences into a SkillSpan-style token dataset.

Input:  sentence candidates from scripts/prepare_skillspan_training_corpus.py
Output: JSONL records with tokens plus BIO tags:
  - tags_hard_skill
  - tags_soft_skill
  - tags_knowledge

The script also writes an audit JSONL with raw model responses and alignment
decisions. Use small batches first; full-corpus annotation is a paid LLM job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llm_client_router import get_client_for_model, strip_provider_prefix  # noqa: E402


LABELS = ("HARD_SKILL", "SOFT_SKILL", "KNOWLEDGE")
TAG_FIELD = {
    "HARD_SKILL": "tags_hard_skill",
    "SOFT_SKILL": "tags_soft_skill",
    "KNOWLEDGE": "tags_knowledge",
}
TOKEN_RE = re.compile(
    r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:[.+#/_-][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)*|[^\w\s]",
    re.UNICODE,
)


SYSTEM_PROMPT = """You create token-level NER annotations for job-posting competency extraction.

Return JSON only. Annotate only spans that are explicitly demanded by the sentence.

Labels:
- HARD_SKILL: an assessable ability, technical/tool proficiency, work method, or job-specific practical skill. Examples: Python, SQL, REST APIs, automated testing, code review, menguasai Java, mendokumentasikan kode.
- SOFT_SKILL: employability/interpersonal/cognitive work behavior. Examples: communication, collaboration, leadership, initiative, attention to detail, mampu berkomunikasi dengan baik, teliti.
- KNOWLEDGE: conceptual/domain knowledge or theory without direct action phrasing. Examples: microservices architecture, distributed systems, security practices, Computer Science, konsep reactive programming.

Span boundary rules:
- Use token_start and token_end indexes from the provided token list. token_end is inclusive.
- Keep spans concise. Prefer the smallest meaningful span.
- Tag the competency head, not the surrounding requirement phrase.
- Do not include verbs or requirement wrappers in a span: use, using, experience with, knowledge of, understanding of, proficient in, able to, ability to, responsible for, menguasai, memahami, memiliki, mampu, terbiasa, dapat, wajib.
- For "menguasai bahasa pemograman Java", tag "bahasa pemograman Java" or "Java", not "menguasai".
- For "memahami dan menguasai PHP", tag "PHP" as HARD_SKILL; do not tag "memahami" by itself.
- For "Knowledge in C#, ASP.NET, MS SQL", tag each technology as HARD_SKILL if the sentence requires practical use, or KNOWLEDGE only if the sentence clearly asks conceptual understanding.
- Do not tag company names, locations, benefits, salaries, years of experience, job type, or application instructions.
- Do not tag generic words alone: tools, systems, process, experience, degree, certification.
- Degrees, years, and certifications are not competencies. If a field is explicitly required, annotate only the field/domain, not the credential phrase.
- A technology can be HARD_SKILL when the sentence requires using/mastering it; it can be KNOWLEDGE when the sentence asks for understanding/knowledge of a concept.
- Indonesian text must keep exact Indonesian spans; labels remain in English.
"""


def _path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text or ""))


def _bio_tags(n: int) -> list[str]:
    return ["O"] * n


def _normal_token(token: str) -> str:
    return re.sub(r"\s+", " ", str(token or "").strip().lower())


def find_token_span(tokens: list[str], span_text: str) -> tuple[int, int] | None:
    span_tokens = tokenize(span_text)
    if not span_tokens:
        return None
    wanted = [_normal_token(t) for t in span_tokens]
    hay = [_normal_token(t) for t in tokens]
    width = len(wanted)
    for i in range(0, len(hay) - width + 1):
        if hay[i : i + width] == wanted:
            return i, i + width - 1
    compact_wanted = "".join(wanted)
    for i in range(len(hay)):
        acc = ""
        for j in range(i, min(len(hay), i + width + 3)):
            acc += hay[j]
            if acc == compact_wanted:
                return i, j
            if len(acc) > len(compact_wanted) + 4:
                break
    return None


def parse_llm_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip())
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def prompt_for_batch(batch: list[dict[str, Any]]) -> str:
    packed = []
    for rec in batch:
        tokens = rec["_tokens"]
        packed.append({
            "sentence_id": rec["sentence_id"],
            "language": rec.get("language", ""),
            "title": rec.get("title", ""),
            "sentence_text": rec.get("sentence_text", ""),
            "tokens": [{"i": i, "t": tok} for i, tok in enumerate(tokens)],
        })
    return (
        "Annotate these records. Return exactly this JSON shape:\n"
        '{"records":[{"sentence_id":"...","spans":[{"label":"HARD_SKILL|SOFT_SKILL|KNOWLEDGE","token_start":0,"token_end":0,"text":"exact token text","confidence":"high|medium|low"}]}]}\n\n'
        "Records:\n"
        + json.dumps(packed, ensure_ascii=False, indent=2)
    )


def call_model(
    client: Any,
    model: str,
    batch: list[dict[str, Any]],
    temperature: float,
    *,
    max_retries: int,
    retry_backoff_seconds: float,
) -> tuple[dict[str, Any] | None, str]:
    last_error = ""
    for attempt in range(max(0, max_retries) + 1):
        try:
            response = client.chat.completions.create(
                model=strip_provider_prefix(model),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_for_batch(batch)},
                ],
                temperature=temperature,
                max_tokens=2500,
            )
            content = response.choices[0].message.content or ""
            return parse_llm_json(content), content
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                wait = retry_backoff_seconds * (attempt + 1)
                print(f"[WARN] model call failed on attempt {attempt + 1}: {last_error}; retrying in {wait:.1f}s")
                time.sleep(wait)
    raise RuntimeError(f"model call failed after {max_retries + 1} attempts: {last_error}")


def normalize_span(raw: dict[str, Any], tokens: list[str]) -> tuple[dict[str, Any] | None, str]:
    label = str(raw.get("label") or "").strip().upper()
    if label not in LABELS:
        return None, "invalid_label"
    start = raw.get("token_start")
    end = raw.get("token_end")
    try:
        start_i = int(start)
        end_i = int(end)
    except (TypeError, ValueError):
        start_i = end_i = -1
    if 0 <= start_i <= end_i < len(tokens):
        text = " ".join(tokens[start_i : end_i + 1])
        return {
            "label": label,
            "token_start": start_i,
            "token_end": end_i,
            "text": text,
            "confidence": str(raw.get("confidence") or "").strip().lower() or "unspecified",
            "alignment": "token_index",
        }, "ok"
    text = str(raw.get("text") or "").strip()
    found = find_token_span(tokens, text)
    if found:
        start_i, end_i = found
        return {
            "label": label,
            "token_start": start_i,
            "token_end": end_i,
            "text": " ".join(tokens[start_i : end_i + 1]),
            "confidence": str(raw.get("confidence") or "").strip().lower() or "unspecified",
            "alignment": "text_fallback",
        }, "ok"
    return None, "alignment_failed"


def apply_spans(tokens: list[str], spans: list[dict[str, Any]]) -> dict[str, list[str]]:
    tags = {field: _bio_tags(len(tokens)) for field in TAG_FIELD.values()}
    for span in sorted(spans, key=lambda s: (s["label"], -(s["token_end"] - s["token_start"]))):
        field = TAG_FIELD[span["label"]]
        start = int(span["token_start"])
        end = int(span["token_end"])
        if any(tags[field][i] != "O" for i in range(start, end + 1)):
            continue
        tags[field][start] = "B"
        for i in range(start + 1, end + 1):
            tags[field][i] = "I"
    tags["tags_skill"] = merge_skill_tags(tags["tags_hard_skill"], tags["tags_soft_skill"])
    return tags


def merge_skill_tags(hard: list[str], soft: list[str]) -> list[str]:
    merged = ["O"] * len(hard)
    occupied = [h != "O" or s != "O" for h, s in zip(hard, soft)]
    for i, is_occ in enumerate(occupied):
        if not is_occ:
            continue
        if i == 0 or not occupied[i - 1]:
            merged[i] = "B"
        else:
            merged[i] = "I"
    return merged


def make_output_record(rec: dict[str, Any], spans: list[dict[str, Any]], parse_status: str) -> dict[str, Any]:
    tokens = rec["_tokens"]
    tags = apply_spans(tokens, spans)
    out = {
        "idx": rec.get("idx"),
        "sentence_id": rec.get("sentence_id"),
        "job_id": rec.get("job_id"),
        "tokens": tokens,
        "tags_skill": tags["tags_skill"],
        "tags_hard_skill": tags["tags_hard_skill"],
        "tags_soft_skill": tags["tags_soft_skill"],
        "tags_knowledge": tags["tags_knowledge"],
        "source": rec.get("dataset_id") or rec.get("source") or "",
        "language": rec.get("language") or "",
        "sentence_text": rec.get("sentence_text") or "",
        "title": rec.get("title") or "",
        "spans": spans,
        "annotation_status": parse_status,
    }
    for field in ("tags_skill", "tags_hard_skill", "tags_soft_skill", "tags_knowledge"):
        if len(out[field]) != len(tokens):
            raise ValueError(f"{field} length mismatch for {rec.get('sentence_id')}")
    return out


def existing_sentence_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for row in load_jsonl(path):
        sid = str(row.get("sentence_id") or "")
        if sid:
            ids.add(sid)
    return ids


def annotate_batch(
    batch: list[dict[str, Any]],
    *,
    client: Any | None,
    model: str,
    temperature: float,
    dry_run: bool,
    max_retries: int,
    retry_backoff_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if dry_run:
        payload = {"records": [{"sentence_id": rec["sentence_id"], "spans": []} for rec in batch]}
        raw_response = json.dumps(payload)
    else:
        payload, raw_response = call_model(
            client,
            model,
            batch,
            temperature,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
    audit = {
        "batch_id": hashlib.sha1("|".join(r["sentence_id"] for r in batch).encode("utf-8")).hexdigest()[:12],
        "sentence_ids": [r["sentence_id"] for r in batch],
        "raw_response": raw_response,
        "parse_ok": isinstance(payload, dict),
        "alignment_failures": [],
    }
    by_id = {}
    if isinstance(payload, dict):
        for row in payload.get("records") or []:
            sid = str(row.get("sentence_id") or "")
            if sid:
                by_id[sid] = row
    outputs: list[dict[str, Any]] = []
    for rec in batch:
        model_row = by_id.get(rec["sentence_id"])
        spans: list[dict[str, Any]] = []
        parse_status = "ok" if isinstance(payload, dict) and model_row is not None else "missing_model_record"
        for raw_span in (model_row or {}).get("spans") or []:
            span, status = normalize_span(raw_span, rec["_tokens"])
            if span:
                spans.append(span)
            else:
                audit["alignment_failures"].append({
                    "sentence_id": rec["sentence_id"],
                    "status": status,
                    "span": raw_span,
                })
        outputs.append(make_output_record(rec, spans, parse_status))
    return outputs, audit


def build_report(output_rows: list[dict[str, Any]], audits: list[dict[str, Any]], *, args: argparse.Namespace) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    token_label_counts: Counter[str] = Counter()
    for row in output_rows:
        for span in row.get("spans") or []:
            label_counts[span["label"]] += 1
        for label, field in TAG_FIELD.items():
            token_label_counts[label] += sum(1 for tag in row.get(field, []) if tag != "O")
    return {
        "schema_version": "2026-06-23.skillspan_ours.annotation.v1",
        "input": args.input_jsonl,
        "output": args.output_jsonl,
        "model": args.model,
        "batch_size": args.batch_size,
        "records_written_this_run": len(output_rows),
        "span_counts": dict(label_counts),
        "token_label_counts": dict(token_label_counts),
        "batches": len(audits),
        "parse_fail_batches": sum(1 for a in audits if not a.get("parse_ok")),
        "alignment_failures": sum(len(a.get("alignment_failures") or []) for a in audits),
        "dry_run": args.dry_run,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--audit-jsonl", default="")
    parser.add_argument("--report-json", default="")
    parser.add_argument("--model", default="openrouter:gpt-5.4-mini")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=8.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write all-O records without calling the model.")
    args = parser.parse_args()

    input_path = _path(args.input_jsonl)
    output_path = _path(args.output_jsonl)
    audit_path = _path(args.audit_jsonl) if args.audit_jsonl else output_path.with_suffix(".audit.jsonl")
    report_path = _path(args.report_json) if args.report_json else output_path.with_suffix(".report.json")
    rows = load_jsonl(input_path)
    if args.max_records and args.max_records > 0:
        rows = rows[: args.max_records]
    if args.resume:
        done = existing_sentence_ids(output_path)
        rows = [r for r in rows if str(r.get("sentence_id") or "") not in done]
        print(f"[INFO] resume enabled: {len(done)} existing sentence_ids skipped")
    for row in rows:
        row["_tokens"] = tokenize(row.get("sentence_text") or "")
    rows = [r for r in rows if r["_tokens"]]

    client = None
    provider = "dry_run"
    if not args.dry_run:
        client, provider = get_client_for_model(args.model, request_timeout=120)
    print(f"[INFO] annotating {len(rows)} records via provider={provider}, model={args.model}")

    all_outputs: list[dict[str, Any]] = []
    all_audits: list[dict[str, Any]] = []
    for start in range(0, len(rows), max(1, args.batch_size)):
        batch = rows[start : start + max(1, args.batch_size)]
        outputs, audit = annotate_batch(
            batch,
            client=client,
            model=args.model,
            temperature=args.temperature,
            dry_run=args.dry_run,
            max_retries=args.max_retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
        )
        append_jsonl(output_path, outputs)
        append_jsonl(audit_path, [audit])
        all_outputs.extend(outputs)
        all_audits.append(audit)
        print(f"[INFO] wrote batch {len(all_audits)} ({len(outputs)} records)")
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    report = build_report(all_outputs, all_audits, args=args)
    report["provider"] = provider
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] output: {output_path}")
    print(f"[INFO] audit: {audit_path}")
    print(f"[INFO] report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
