"""Bridge to the competency generator that already existed (`competency_generator_v2`).

The plan's instruction for this layer is "reuse v2 chain, switch evidence model
to typed five-tag units". The first version of `competency/` wrote its own
short statements instead ("Perform programming using Python, Java and C++").
The researcher pointed out on 2026-11-19 that the earlier system's output was
much better, and it is: statements of two to four sentences that abstract from
the postings, a rationale, the soft skills that go with the competency, a KKNI
level proposed from cognitive complexity, provenance, an evidence audit and a
filter against content the sources do not support.

So this module does what the plan said. It hands the generator the SAME
clusters the grouping step made, built from five-tag evidence:

  * the hard-skill groups are the clusters it writes from (competencies come
    from SKILL_HARD: the researcher's decision of 2026-11-19);
  * the most frequent SKILL_SOFT phrases are its soft-skill reference list. What
    it picks from that list is the model's judgement (`v2.soft_skills_required`);
    the soft skills COUNTED in the competency's own postings are a separate,
    measured list (`evidence.soft_skills`);
  * TOOL_TECHNOLOGY reaches it through the source sentences it is shown, and
    its own filter only lets a product name through if a source names it.

One competency per group (`max_competencies_per_cluster = 1`), because the
proposal says "satu pernyataan kompetensi" per valid group and the status
contract is per group.

Status. A statement returned here passed the generator's evidence filter;
that filter, not the lexical grounding share, is what the status contract uses
for these statements (see `competency.contract`).

Spending. The old generator has no cap and does not report what a call cost, so
only the worst case is known exactly. The old generator has no cap. This bridge adds one: before any call
it builds every prompt, prices the worst case (every retry, the full output
allowance) and refuses to start if that exceeds `max_usd`.
"""
from __future__ import annotations

from typing import Sequence

from .grouping import Group
from .statement import INDONESIAN_VERBS, VERBS, Statement, base_verb, grounding

BRIDGE_VERSION = "v2-generator-bridge-v1"
PRICE_PER_M = {"in": 0.75, "out": 4.50}        # gpt-5.4-mini list price, USD per million tokens


class GeneratorBudgetError(RuntimeError):
    pass


def worst_case_usd(prompts: Sequence[str], system: str, max_tokens: int, retries: int) -> float:
    total = 0.0
    for p in prompts:
        tokens_in = int((len(p) + len(system)) / 2.5) + 50
        total += (tokens_in * PRICE_PER_M["in"] + max_tokens * PRICE_PER_M["out"]) / 1e6
    return total * (1 + retries)


def generate(groups: Sequence[Group], clusters_by_id: dict, *, max_groups: int, max_usd: float,
             soft_skills: Sequence[str] = (), model: str | None = None, log=print) -> dict:
    """Statements for the `max_groups` valid technical groups with the most
    postings. Returns {"statements": {group_id: (Statement, extras)}, "info"}."""
    import competency_generator_v2 as G
    from competency_v2_schema import GeneratorConfig
    chosen = [g for g in groups if g.status == "sahih" and g.stream == "technical"
              and g.group_id in clusters_by_id][:max_groups]
    cfg = GeneratorConfig(max_competencies_per_cluster=1)
    if model:
        cfg.model = model
    clusters = [clusters_by_id[g.group_id] for g in chosen]
    soft = list(soft_skills)[:30]
    prompts = [G.build_cluster_prompt(c, cfg, top_soft_skills=soft) for c in clusters]
    worst = worst_case_usd(prompts, G.SYSTEM_PROMPT, cfg.max_tokens, cfg.max_retries)
    if worst > max_usd:
        raise GeneratorBudgetError(
            f"worst case ${worst:.2f} for {len(clusters)} groups exceeds the cap ${max_usd:.2f}; "
            f"lower max_groups or raise the cap")
    log(f"v2 generator: {len(clusters)} groups, model {cfg.model}, worst case ${worst:.2f}")
    passing, reasonings, dropped = G.generate_competencies_v2(clusters, cfg, top_soft_skills=soft)
    by_group = {g.group_id: g for g in chosen}
    out: dict = {}
    absorbed = 0
    for comp in passing:
        g = by_group.get(str(comp.cluster_id))
        if g is None or g.group_id in out:
            continue
        absorbed += len(getattr(comp, "merged_from", []) or [])
        extras = _extras(comp)
        # The researcher (2026-11-20): a statement is a verb and its object; tools
        # are listed beside it. The old generator may still name one. It is not
        # edited here, so the names are reported, not removed.
        low = " " + " ".join(str(comp.statement).lower().split()) + " "
        extras["tools_named_in_statement"] = [
            t["text"] for t in g.technologies
            if f" {t['text'].lower()} " in low.replace(",", " ").replace(".", " ")]
        out[g.group_id] = (_to_statement(comp, g), extras)
    info = {"bridge_version": BRIDGE_VERSION, "model": cfg.model, "n_groups_sent": len(clusters),
            "n_statements": len(out), "n_dropped_by_its_filter": len(dropped),
            "n_absorbed_by_its_dedup": absorbed, "worst_case_usd": worst,
            "n_statements_naming_a_tool": sum(
                1 for _, x in out.values() if x["tools_named_in_statement"]),
            "generator_version": getattr(passing[0], "generator_version", "") if passing else "",
            "dropped": [{"group_id": str(c.cluster_id), "statement": c.statement,
                         "concerns": list(getattr(c, "statement_evidence_concerns", []) or [])}
                        for c in dropped]}
    return {"statements": out, "info": info}


def _to_statement(comp, group: Group) -> Statement:
    text = " ".join(str(comp.statement).split())
    words = text.split()
    first = words[0].strip(",.").lower() if words else ""
    verb = base_verb(first) or first
    score, missing = grounding(text, group)
    return Statement(
        text=text, verb=verb, object="", technologies=[],
        krathwohl=VERBS.get(verb, "not in the verb table"),
        language="id" if verb in INDONESIAN_VERBS else "en", method="v2-generator",
        verb_source="evidence", grounding=score, ungrounded=missing[:20],
        notes=["grounding here is the lexical measure against the group's own phrases; a "
               "statement that abstracts, as this generator is asked to, scores lower on it. "
               "The generator's own evidence audit is in `v2`."])


def _extras(comp) -> dict:
    keep = ("rationale", "related_skills", "soft_skills_required", "soft_skills_description",
            "kkni_level", "kkni_level_source", "grounding_score", "grounding_passed",
            "grounding_method", "statement_evidence_concerns", "source_job_ids", "model",
            "generator_version", "merged_from")
    return {k: getattr(comp, k, None) for k in keep}
