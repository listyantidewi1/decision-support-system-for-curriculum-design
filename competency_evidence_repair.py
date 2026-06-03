"""
Evidence linker / repair pass for competency-v2 outputs.

The repair pass is intentionally conservative:

1. Normalize noisy related-skill variants such as "Python)" and "Python.".
2. Drop generic or soft-skill-like related skills that should not anchor a
   hard-skill competency.
3. Keep only source sentences that support both the competency statement and
   at least one related skill.
4. Drop competencies that no longer have enough supported skills or evidence.

When sentence-transformers is available, the pass uses SBERT cosine similarity.
When it is not available, it falls back to lexical matching and marks the audit
accordingly.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from competency_v2_schema import CompetencyV2


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
REPAIR_VERSION = "v2.6.0"


@dataclass
class EvidenceRepairConfig:
    """Configuration for evidence repair."""

    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    disable_sbert: bool = False
    statement_threshold: float = 0.45
    skill_threshold: float = 0.45
    weak_statement_threshold: float = 0.38
    min_supported_skills: int = 2
    min_supporting_sentences: int = 2
    max_supporting_sentences: int = 8
    drop_generic_skills: bool = True
    apply_final_evaluator_gate: bool = True
    final_grounding_threshold: float = 0.80


@dataclass
class EvidenceRepairReport:
    n_input: int
    n_repaired: int
    n_dropped: int
    n_sbert_used: int
    n_lexical_fallback: int
    n_skills_removed: int
    n_sentences_removed: int
    grounding_mean_before: float
    grounding_mean_after: float
    grounding_min_before: float
    grounding_min_after: float
    config: EvidenceRepairConfig

    def to_dict(self) -> dict:
        return {
            "version": REPAIR_VERSION,
            "n_input": self.n_input,
            "n_repaired": self.n_repaired,
            "n_dropped": self.n_dropped,
            "pass_rate": round(self.n_repaired / max(1, self.n_input), 4),
            "n_sbert_used": self.n_sbert_used,
            "n_lexical_fallback": self.n_lexical_fallback,
            "n_skills_removed": self.n_skills_removed,
            "n_sentences_removed": self.n_sentences_removed,
            "grounding_mean_before": round(float(self.grounding_mean_before), 4),
            "grounding_mean_after": round(float(self.grounding_mean_after), 4),
            "grounding_min_before": round(float(self.grounding_min_before), 4),
            "grounding_min_after": round(float(self.grounding_min_after), 4),
            "config": dict(self.config.__dict__),
        }


_TRAILING_PUNCT_RE = re.compile(r"^[\s\"'`([{<]+|[\s\"'`)\]}>.,;:!?]+$")
_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")


_GENERIC_SKILL_KEYS = {
    "build",
    "develop",
    "skilled",
    "hands on",
    "hands-on",
    "hands -on",
    "hands on experience",
    "hands-on experience",
    "hands -on experience",
    "communication skills",
    "communication abilities",
    "communication and interpersonal skills",
    "interpersonal skills",
    "communicator",
    "communicate effectively",
    "problem solving",
    "analytical thinking",
    "attention to detail",
}


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return statistics.mean(vals) if vals else 0.0


def _min(values: Iterable[float]) -> float:
    vals = list(values)
    return min(vals) if vals else 0.0


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_key(text: str) -> str:
    t = str(text or "").strip().lower()
    t = re.sub(r"\s*-\s*", "-", t)
    t = t.strip(" \t\r\n\"'`([{<.,;:!?)]}>")
    t = re.sub(r"[\s_/|,;:()\[\]{}]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _clean_skill_surface(text: str) -> str:
    t = str(text or "").strip()
    t = re.sub(r"\s*-\s*", "-", t)
    t = _TRAILING_PUNCT_RE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _sentence_key(text: str) -> str:
    t = str(text or "").lower()
    t = re.sub(r"\s*([+#./-])\s*", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _lexical_match(skill_key: str, sentence: str) -> bool:
    if not skill_key or not sentence:
        return False
    sent = _sentence_key(sentence)
    compact_skill = skill_key.replace(" ", "")
    compact_sent = sent.replace(" ", "")
    return skill_key in sent or compact_skill in compact_sent


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(_sentence_key(text)) if len(t) >= 2}


def _lexical_similarity(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _dedupe_skills(skills: Sequence[str], drop_generic: bool = True) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    removed: list[str] = []
    seen: set[str] = set()
    for skill in skills or []:
        cleaned = _clean_skill_surface(str(skill))
        key = _normalize_key(cleaned)
        if not key:
            continue
        if key in seen:
            removed.append(str(skill))
            continue
        if drop_generic and key in _GENERIC_SKILL_KEYS:
            removed.append(str(skill))
            continue
        seen.add(key)
        kept.append(cleaned)
    return kept, removed


def _load_embedder(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception as exc:  # pragma: no cover - depends on local ML env
        logger.warning("SBERT unavailable for evidence repair: %s", exc)
        return None


def _semantic_matrices(
    statement: str,
    skills: list[str],
    sentences: list[str],
    embedder,
) -> tuple[list[float], list[list[float]]]:
    if embedder is None or not skills or not sentences:
        return [], []
    texts = [statement] + skills + sentences
    embs = embedder.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    statement_vec = embs[0]
    skill_vecs = embs[1 : 1 + len(skills)]
    sent_vecs = embs[1 + len(skills) :]
    statement_scores = list(sent_vecs @ statement_vec)
    skill_scores = skill_vecs @ sent_vecs.T
    return [float(x) for x in statement_scores], [
        [float(v) for v in row] for row in skill_scores
    ]


def _lexical_matrices(
    statement: str,
    skills: list[str],
    sentences: list[str],
) -> tuple[list[float], list[list[float]]]:
    statement_scores = [_lexical_similarity(statement, s) for s in sentences]
    skill_scores = []
    for skill in skills:
        skey = _normalize_key(skill)
        row = []
        for sent in sentences:
            row.append(1.0 if _lexical_match(skey, sent) else _lexical_similarity(skill, sent))
        skill_scores.append(row)
    return statement_scores, skill_scores


def _rehydrate_competencies(dicts: Sequence[dict[str, Any]]) -> list[CompetencyV2]:
    field_names = set(CompetencyV2.__dataclass_fields__.keys())
    out = []
    for d in dicts:
        if "statement" not in d:
            title = str(d.get("title") or "").strip()
            desc = str(d.get("description") or "").strip()
            d = dict(d, statement=f"{title}. {desc}".strip(". "))
        if "statement_evidence_concerns" not in d and "title_evidence_concerns" in d:
            d = dict(d, statement_evidence_concerns=d["title_evidence_concerns"])
        kwargs = {k: v for k, v in d.items() if k in field_names}
        out.append(CompetencyV2(**kwargs))
    return out


def _repair_one(
    comp: CompetencyV2,
    config: EvidenceRepairConfig,
    embedder,
) -> tuple[Optional[CompetencyV2], dict[str, Any]]:
    before_skills = list(comp.related_skills or [])
    before_sentences = list(comp.source_sentences or [])
    before_grounding = float(comp.grounding_score or 0.0)

    skills, removed_skills = _dedupe_skills(before_skills, config.drop_generic_skills)
    sentences = [str(s).strip() for s in before_sentences if str(s).strip()]

    audit: dict[str, Any] = {
        "version": REPAIR_VERSION,
        "method": "sbert" if embedder is not None and not config.disable_sbert else "lexical",
        "before": {
            "n_related_skills": len(before_skills),
            "n_source_sentences": len(before_sentences),
            "grounding_score": round(before_grounding, 4),
        },
        "removed_related_skills": removed_skills,
        "removed_source_sentences": [],
        "kept_sentence_scores": [],
        "drop_reason": "",
    }

    if len(skills) < config.min_supported_skills or len(sentences) < config.min_supporting_sentences:
        audit["drop_reason"] = "insufficient_input_after_skill_normalization"
        comp.evidence_repair_audit = audit
        return None, audit

    if embedder is not None and not config.disable_sbert:
        statement_scores, skill_scores = _semantic_matrices(comp.statement, skills, sentences, embedder)
    else:
        statement_scores, skill_scores = _lexical_matrices(comp.statement, skills, sentences)

    if not statement_scores or not skill_scores:
        audit["drop_reason"] = "no_scores_available"
        comp.evidence_repair_audit = audit
        return None, audit

    skill_keys = [_normalize_key(s) for s in skills]
    good_statement_idxs = {
        i for i, score in enumerate(statement_scores)
        if score >= config.statement_threshold
    }

    supported_skills: list[str] = []
    support_sentence_idxs: set[int] = set()
    unsupported_skills: list[str] = []
    weak_supported_skills: list[str] = []

    for si, skill in enumerate(skills):
        candidates: list[tuple[float, int, bool, bool]] = []
        for sent_idx, sent in enumerate(sentences):
            lexical_hit = _lexical_match(skill_keys[si], sent)
            skill_hit = skill_scores[si][sent_idx] >= config.skill_threshold or lexical_hit
            statement_hit = statement_scores[sent_idx] >= config.statement_threshold
            weak_statement_hit = statement_scores[sent_idx] >= config.weak_statement_threshold
            if skill_hit and (statement_hit or weak_statement_hit):
                score = (
                    statement_scores[sent_idx]
                    + 0.50 * skill_scores[si][sent_idx]
                    + (0.10 if lexical_hit else 0.0)
                )
                candidates.append((score, sent_idx, statement_hit, lexical_hit))
        candidates.sort(reverse=True)
        if not candidates:
            unsupported_skills.append(skill)
            continue
        best_score, best_idx, strong_statement, _lexical = candidates[0]
        supported_skills.append(skill)
        support_sentence_idxs.add(best_idx)
        if not strong_statement:
            weak_supported_skills.append(skill)

    if len(supported_skills) < config.min_supported_skills:
        audit["drop_reason"] = "fewer_than_min_supported_skills"
        audit["unsupported_related_skills"] = unsupported_skills
        comp.evidence_repair_audit = audit
        return None, audit

    # Add the best statement-supporting sentences until the UI cap. This keeps
    # high-relevance evidence visible without reintroducing broad cluster noise.
    ranked_statement_idxs = sorted(
        good_statement_idxs,
        key=lambda idx: statement_scores[idx],
        reverse=True,
    )
    keep_idxs: list[int] = []
    for idx in list(support_sentence_idxs) + ranked_statement_idxs:
        if idx not in keep_idxs:
            keep_idxs.append(idx)
        if len(keep_idxs) >= config.max_supporting_sentences:
            break

    if len(keep_idxs) < config.min_supporting_sentences:
        audit["drop_reason"] = "fewer_than_min_supporting_sentences"
        audit["unsupported_related_skills"] = unsupported_skills
        comp.evidence_repair_audit = audit
        return None, audit

    kept_statement_good = sum(1 for idx in keep_idxs if statement_scores[idx] >= config.statement_threshold)
    kept_skill_good = 0
    for skill_idx, _skill in enumerate(supported_skills):
        original_skill_idx = skills.index(_skill)
        if any(
            skill_scores[original_skill_idx][sent_idx] >= config.skill_threshold
            or _lexical_match(_normalize_key(_skill), sentences[sent_idx])
            for sent_idx in keep_idxs
        ):
            kept_skill_good += 1

    skill_grounding = kept_skill_good / max(1, len(supported_skills))
    sentence_relevance = kept_statement_good / max(1, len(keep_idxs))
    grounding = min(skill_grounding, sentence_relevance)

    kept_sentences = [sentences[idx] for idx in keep_idxs]
    removed_sentence_texts = [
        s for i, s in enumerate(sentences)
        if i not in set(keep_idxs)
    ]

    comp.related_skills = supported_skills
    comp.source_sentences = kept_sentences
    comp.grounding_score = round(float(grounding), 4)
    comp.skill_grounding_score = round(float(skill_grounding), 4)
    comp.sentence_relevance_score = round(float(sentence_relevance), 4)
    comp.grounding_passed = grounding >= config.statement_threshold
    comp.grounding_method = f"evidence_repair_{audit['method']}"
    comp.grounding_reasoning = (
        f"Evidence repair kept {len(kept_sentences)}/{len(before_sentences)} sentences "
        f"and {len(supported_skills)}/{len(before_skills)} related skills. "
        f"skill_grounding={skill_grounding:.3f}; sentence_relevance={sentence_relevance:.3f}; "
        f"canonical={grounding:.3f}."
    )
    comp.evaluator_version = REPAIR_VERSION

    audit.update({
        "after": {
            "n_related_skills": len(comp.related_skills),
            "n_source_sentences": len(comp.source_sentences),
            "grounding_score": round(float(comp.grounding_score), 4),
            "skill_grounding_score": round(float(comp.skill_grounding_score), 4),
            "sentence_relevance_score": round(float(comp.sentence_relevance_score), 4),
        },
        "unsupported_related_skills": unsupported_skills,
        "weak_statement_supported_skills": weak_supported_skills,
        "removed_source_sentences": removed_sentence_texts[:20],
        "kept_sentence_scores": [
            {
                "sentence": sentences[idx],
                "statement_score": round(float(statement_scores[idx]), 4),
                "max_skill_score": round(float(max(row[idx] for row in skill_scores)), 4),
            }
            for idx in keep_idxs
        ],
    })
    comp.evidence_repair_audit = audit
    return comp, audit


def repair_competencies(
    competencies: Sequence[CompetencyV2],
    config: Optional[EvidenceRepairConfig] = None,
) -> tuple[list[CompetencyV2], list[CompetencyV2], EvidenceRepairReport]:
    if config is None:
        config = EvidenceRepairConfig()
    embedder = None if config.disable_sbert else _load_embedder(config.sbert_model)
    repaired: list[CompetencyV2] = []
    dropped: list[CompetencyV2] = []
    audits: list[dict[str, Any]] = []
    before_scores = [float(c.grounding_score or 0.0) for c in competencies]

    for comp in competencies:
        fixed, audit = _repair_one(comp, config, embedder)
        audits.append(audit)
        if fixed is None:
            comp.evidence_repair_audit = audit
            dropped.append(comp)
        else:
            repaired.append(fixed)

    if config.apply_final_evaluator_gate and repaired:
        try:
            from competency_evaluator import EvaluatorConfig, evaluate_competencies

            eval_config = EvaluatorConfig(
                grounding_threshold=config.final_grounding_threshold,
                sbert_threshold=config.skill_threshold,
                disable_sbert=config.disable_sbert,
            )
            passed, failed, _eval_report = evaluate_competencies(repaired, config=eval_config)
            for comp in failed:
                comp.evidence_repair_audit = dict(comp.evidence_repair_audit or {})
                comp.evidence_repair_audit["drop_reason"] = "final_evaluator_grounding_gate"
                comp.evidence_repair_audit["final_grounding_threshold"] = config.final_grounding_threshold
            repaired = passed
            dropped.extend(failed)
        except Exception as exc:  # pragma: no cover - depends on local ML env
            logger.warning("final evaluator gate skipped after evidence repair: %s", exc)

    after_scores = [float(c.grounding_score or 0.0) for c in repaired]
    report = EvidenceRepairReport(
        n_input=len(competencies),
        n_repaired=len(repaired),
        n_dropped=len(dropped),
        n_sbert_used=sum(1 for a in audits if a.get("method") == "sbert"),
        n_lexical_fallback=sum(1 for a in audits if a.get("method") == "lexical"),
        n_skills_removed=sum(
            len(a.get("removed_related_skills", [])) + len(a.get("unsupported_related_skills", []))
            for a in audits
        ),
        n_sentences_removed=sum(len(a.get("removed_source_sentences", [])) for a in audits),
        grounding_mean_before=_mean(before_scores),
        grounding_mean_after=_mean(after_scores),
        grounding_min_before=_min(before_scores),
        grounding_min_after=_min(after_scores),
        config=config,
    )
    return repaired, dropped, report


def _load_competencies_json(path: Path) -> list[CompetencyV2]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a list")
    return _rehydrate_competencies(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--statement-threshold", type=float, default=0.45)
    parser.add_argument("--skill-threshold", type=float, default=0.45)
    parser.add_argument("--weak-statement-threshold", type=float, default=0.38)
    parser.add_argument("--min-supported-skills", type=int, default=2)
    parser.add_argument("--min-supporting-sentences", type=int, default=2)
    parser.add_argument("--max-supporting-sentences", type=int, default=8)
    parser.add_argument("--final-grounding-threshold", type=float, default=0.80)
    parser.add_argument("--skip-final-evaluator-gate", action="store_true")
    parser.add_argument("--no-sbert", action="store_true")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.is_absolute():
        in_dir = PROJECT_ROOT / in_dir
    if not in_dir.exists():
        raise FileNotFoundError(in_dir)

    out_dir = Path(args.output_dir) if args.output_dir else in_dir.with_name(in_dir.name + "_evidence_repaired")
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in [
        "competencies_evaluated.json",
        "passing.json",
        "failing.json",
        "evaluator_report.json",
    ]:
        stale_path = out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    comps = _load_competencies_json(in_dir / "competencies.json")
    cfg = EvidenceRepairConfig(
        disable_sbert=args.no_sbert,
        statement_threshold=args.statement_threshold,
        skill_threshold=args.skill_threshold,
        weak_statement_threshold=args.weak_statement_threshold,
        min_supported_skills=args.min_supported_skills,
        min_supporting_sentences=args.min_supporting_sentences,
        max_supporting_sentences=args.max_supporting_sentences,
        final_grounding_threshold=args.final_grounding_threshold,
        apply_final_evaluator_gate=not args.skip_final_evaluator_gate,
    )
    repaired, dropped, report = repair_competencies(comps, cfg)

    # Copy run context files first, then overwrite competencies/report.
    for name in [
        "clusters.json",
        "batch_reasonings.json",
        "noise_audit.json",
        "competencies_failed.json",
        "pipeline_report.json",
    ]:
        src = in_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    failed_existing = []
    failed_path = in_dir / "competencies_failed.json"
    if failed_path.exists():
        loaded = json.loads(failed_path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            failed_existing = loaded
    dropped_dicts = []
    for c in dropped:
        d = c.to_dict()
        d["dropped_reason"] = "evidence_repair_gate"
        d["dropped_detail"] = c.evidence_repair_audit
        dropped_dicts.append(d)

    (out_dir / "competencies.json").write_text(
        json.dumps([c.to_dict() for c in repaired], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "competencies_failed.json").write_text(
        json.dumps(failed_existing + dropped_dicts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "evidence_repair_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report_path = out_dir / "pipeline_report.json"
    if report_path.exists():
        pipe = json.loads(report_path.read_text(encoding="utf-8"))
        old_tag = pipe.get("tag")
        repaired_tag = out_dir.name
        prefix = "competency_v2_pipeline_"
        if repaired_tag.startswith(prefix):
            repaired_tag = repaired_tag[len(prefix):]
        pipe["source_tag_before_evidence_repair"] = old_tag
        pipe["tag"] = repaired_tag
        pipe.setdefault("stages", {})["phase2_6_evidence_repair"] = report.to_dict()
        pipe["final_n_competencies"] = len(repaired)
        report_path.write_text(json.dumps(pipe, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] evidence repair: {len(repaired)} kept, {len(dropped)} dropped")
    print(f"     mean grounding: {report.grounding_mean_before:.4f} -> {report.grounding_mean_after:.4f}")
    print(f"     min grounding:  {report.grounding_min_before:.4f} -> {report.grounding_min_after:.4f}")
    print(f"     output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
