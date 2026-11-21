"""The configuration registry: every version, prompt, dictionary and threshold in one place.

The approved proposal requires it (§3.3.3): "Seluruh konfigurasi, versi model,
prompt, kamus, dan ambang didokumentasikan dalam registri konfigurasi agar
setiap hasil dapat direproduksi dan setiap klaim keterbaikan dapat diaudit."

The registry is GENERATED from the code, never typed by hand, so it cannot
drift from what actually runs:

    python -m governance.config_registry          # rewrite docs/CONFIG_REGISTRY.md
    python -m governance.config_registry --check  # fail if the document is out of date

`tests/test_config_registry.py` runs the check, so changing a threshold without
regenerating the registry fails the suite.

Every entry carries a status, and the status is the point:

  registered      fixed in a registration document before data existed; changing
                  it is a new version of that rule
  decided         a decision of the researcher, recorded with its date
  by design       part of a definition (a version name, a seed, a hash)
  starting value  NOT yet agreed. It must be fixed, with the promotor or the
                  experts, before any confirmatory result or released profile
                  depends on it
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "CONFIG_REGISTRY.md"
JSON_OUT = REPO_ROOT / "docs" / "config_registry.json"
REGISTERED, DECIDED, DESIGN, STARTING = "registered", "decided", "by design", "starting value"


def _row(area, name, value, status, where, note=""):
    return {"area": area, "name": name, "value": value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True), "status": status, "where": where, "note": note}


def collect() -> list[dict]:
    rows: list[dict] = []
    add = lambda *a, **k: rows.append(_row(*a, **k))                      # noqa: E731

    # ---- gold standard
    import gold_standard as GS
    from gold_standard import gate, iaa
    area = "Gold standard"
    add(area, "annotation guideline", GS.GUIDELINE_CODE, DECIDED, "gold_standard/__init__.py",
        "validated by three validators, 2026-11-18")
    add(area, "sample sizes (pilot / main / practice)", [GS.PILOT_N, GS.MAIN_N, GS.PRACTICE_N],
        REGISTERED, "docs/experiments/PREREG_gold_standard_size.md")
    add(area, "sampling seed", GS.SEED, DESIGN, "gold_standard/__init__.py")
    add(area, "pilot gate version", gate.RULE_VERSION, REGISTERED,
        "docs/experiments/PREREG_pilot_gate.md")
    add(area, "pilot gate: go at / stop below", [gate.GO_AT, gate.STOP_BELOW], REGISTERED,
        "docs/experiments/PREREG_pilot_gate.md")
    add(area, "pilot gate: minimum complete share", gate.MIN_COMPLETE_SHARE, REGISTERED,
        "docs/experiments/PREREG_pilot_gate.md")
    add(area, "train / dev / test split", "720 / 240 / 240, group-aware", REGISTERED,
        "docs/experiments/split_spec_v1.md")
    add(area, "test scoring rule", "strict span-F1, sealed denominator", REGISTERED,
        "docs/experiments/PREREG_test_scoring_rule.md")
    add(area, "bootstrap iterations / seed", [iaa.BOOTSTRAP_ITERS, iaa.BOOTSTRAP_SEED], DESIGN,
        "gold_standard/iaa.py")

    # ---- labelling
    import labeling
    from labeling import prompts
    from labeling.consensus import ConsensusConfig
    from labeling.run import DEFAULT_MODELS
    from labeling.verify import VerifyPolicy
    area = "LLM labelling"
    add(area, "labelling version", labeling.LABELING_VERSION, DESIGN, "labeling/__init__.py")
    add(area, "default prompt", prompts.DEFAULT_PROMPT_VERSION, DECIDED, "labeling/prompts.py",
        "v0 kept after EXP-006; the final prompt is chosen on the gold dev split")
    for v in prompts.PROMPT_VERSIONS:
        add(area, f"prompt {v} (SHA-256)", prompts.prompt_sha256(v), DESIGN, "labeling/prompts.py")
    add(area, "ensemble models and list prices (USD per million tokens)", DEFAULT_MODELS,
        STARTING, "labeling/run.py", "final models are registered before the test is opened")
    c = {f.name: f.default for f in dataclasses.fields(ConsensusConfig)
         if f.default is not dataclasses.MISSING}
    add(area, "span alignment threshold (IoU, unit)", [c["iou_threshold"], c["iou_unit"]], DESIGN,
        "labeling/consensus.py")
    add(area, "majority rule", "ceil(2n/3) of the configured models", DESIGN,
        "labeling/consensus.py")
    add(area, "accept two-of-three automatically", c["accept_majority"], DECIDED,
        "labeling/consensus.py", "off until checked on gold (2026-11-18)")
    add(area, "verifier policy",
        {f.name: f.default for f in dataclasses.fields(VerifyPolicy)}, DECIDED,
        "labeling/verify.py", "cross-label overlap allowed per guideline 6.9")

    # ---- extraction and typing
    import fivetag
    from fivetag import conflict_routing as CR
    from fivetag import e0_rules, gliner_adapter, lexicon, router, typing_d
    from fivetag.encoder import EncoderConfig
    area = "Extraction and typing"
    add(area, "fivetag version", fivetag.FIVETAG_VERSION, DESIGN, "fivetag/__init__.py")
    for v in ("v1", "v2", "v3"):
        if (lexicon.LEXICON_ROOT / v).is_dir():
            add(area, f"lexicon {v} (SHA-256)", lexicon.lexicon_sha256(lexicon.LEXICON_ROOT / v),
                DESIGN, f"fivetag/lexicons/{v}/")
    add(area, "default lexicon of fivetag", lexicon.DEFAULT_VERSION, DECIDED, "fivetag/lexicon.py",
        "v1 kept as default so earlier numbers stay reproducible")
    add(area, "E0 rules version", e0_rules.RULES_VERSION, DESIGN, "fivetag/e0_rules.py")
    add(area, "E0 rules v2: quality words a listed object may open with",
        f"{len(e0_rules.QUALITY_MODIFIERS)} words", STARTING, "fivetag/e0_rules.py",
        "the developer's list, its most frequent words taken from the words that ended the cut "
        "phrases of run e0_pool_all_v8 (EXP-035)")
    add(area, "E0 rule scores (dictionary / activity / head / verb / suffix)",
        [e0_rules.SCORE_DICT, e0_rules.SCORE_ACTIVITY, e0_rules.SCORE_HEAD, e0_rules.SCORE_VERB,
         e0_rules.SCORE_SUFFIX], STARTING, "fivetag/e0_rules.py",
        "fixed confidences, not probabilities; the router reads them")
    e = {f.name: f.default for f in dataclasses.fields(EncoderConfig)}
    add(area, "E1/E2 encoder defaults", e, STARTING, "fivetag/encoder.py",
        "20 epochs after EXP-015; the epoch count for the real gold is chosen on dev")
    add(area, "E3 GLiNER model / threshold / labels",
        [gliner_adapter.DEFAULT_MODEL, 0.5, gliner_adapter.LABELS_VERSION], STARTING,
        "fivetag/gliner_adapter.py")
    add(area, "typing version", typing_d.TYPING_VERSION, DESIGN, "fivetag/typing_d.py")
    add(area, "D2 fuzzy threshold / minimum length / cue window",
        [typing_d.FUZZY_THRESHOLD, typing_d.FUZZY_MIN_LENGTH, typing_d.CUE_WINDOW], STARTING,
        "fivetag/typing_d.py", "the proposal calls for a frozen threshold: freeze it on dev")
    r = {f.name: f.default for f in dataclasses.fields(router.RouterConfig)}
    add(area, "E6 router thresholds", {k: r[k] for k in (
        "uncertain_low", "uncertain_high", "disagreement_iou", "complex_min_conjunctions",
        "merge")}, STARTING, "fivetag/router.py", "to be tuned on the dev split")
    add(area, "E6 router version", router.ROUTER_VERSION, DESIGN, "fivetag/router.py")
    add(area, "D1/D2 conflict routing version", CR.ROUTING_VERSION, REGISTERED,
        "docs/experiments/PREREG_d3_conflict_routing.md")
    add(area, "conflict routing: high-risk statuses", list(CR.HIGH_RISK_STATUSES), REGISTERED,
        "docs/experiments/PREREG_d3_conflict_routing.md")
    add(area, "conflict routing: audit sample size / seed", [CR.AUDIT_SAMPLE_N, CR.AUDIT_SEED],
        REGISTERED, "docs/experiments/PREREG_d3_conflict_routing.md")
    add(area, "adjudicator acceptance: accept at / borderline at (lower bound)",
        [CR.ACCEPT_LOWER_BOUND, CR.BORDERLINE_LOWER_BOUND], REGISTERED,
        "docs/experiments/PREREG_d3_conflict_routing.md", "agreed by the promotor, 2026-11-19")
    add(area, "adjudicator model", "not registered (provisional: qwen/qwen3.8-flash)", STARTING,
        "docs/experiments/EXPERIMENT_LOG.md", "EXP-020; confirm on pilot gold")

    # ---- competency
    import competency
    from competency import coverage, delta, endtoend, mapping, profile, statement, temporal
    from competency.contract import ContractConfig
    from competency.grouping import GroupingConfig
    area = "Competency layers"
    add(area, "competency version", competency.COMPETENCY_VERSION, DESIGN, "competency/__init__.py")
    add(area, "roles of the five tags",
        "competencies are built from SH only; TT attached by shared sentence; SS and WB "
        "attached by shared postings, with share and lift",
        DECIDED, "competency/__init__.py", "the researcher, 2026-11-19")
    add(area, "what a competency is",
        "an operational verb (placed on the taxonomy by the verb table) plus the knowledge it "
        "acts on; the statement names no tool", DECIDED, "competency/structure.py",
        "the researcher, 2026-11-20")
    from competency import structure, termmap
    s = {f.name: f.default for f in dataclasses.fields(structure.StructureConfig)}
    add(area, "how the knowledge of a competency is read from its phrases",
        {k: v for k, v in s.items() if k != "version"}, STARTING, "competency/structure.py",
        "the shared head of the objects; a word-order rule, no part-of-speech tagger")
    add(area, "KNOWLEDGE-tagged terms asked for in the same postings",
        "listed beside a competency as other knowledge, with share and lift", STARTING,
        "competency/__init__.py", "the developer's reading; shown, never part of the statement")
    add(area, "cognitive level of a competency",
        "read from the verbs the postings use (C1-C6); never from a default verb supplied by "
        "the method; phrases without a verb form a competency with no level",
        DECIDED, "competency/structure.py",
        "the researcher's correction, 2026-11-20; a test holds it")
    m = {f.name: f.default for f in dataclasses.fields(termmap.TermMapConfig)}
    add(area, "cluster map of knowledge and technology",
        {k: v for k, v in m.items() if k != "version"}, STARTING, "competency/termmap.py",
        "the number of clusters is a setting, not a discovery")
    add(area, "cluster map version", termmap.TERMMAP_VERSION, DESIGN, "competency/termmap.py")
    add(area, "role of the KKNI level",
        "information for the reader only; never part of the status or the order",
        DECIDED, "competency/contract.py", "the researcher, 2026-11-19; a test holds it")
    add(area, "grounding check for generator statements",
        "the generator's own evidence filter, not the lexical share", STARTING,
        "competency/contract.py", "the developer's choice; see EXP-026")
    g = {f.name: f.default for f in dataclasses.fields(GroupingConfig)}
    add(area, "grouping settings", g, STARTING, "competency/grouping.py",
        "cohesion floor carried over from the pre-approval clusterer")
    add(area, "verb table", statement.VERB_TABLE_VERSION, STARTING,
        "competency/verbs/bloom_verbs_v2.tsv",
        f"{len(statement.VERB_INFO)} verbs with their sources in competency/verbs/SOURCES.md; "
        "no expert review yet")
    add(area, "verb table hash (sha256)", statement.verb_table_sha256(), DESIGN,
        "competency/verbs/bloom_verbs_v2.tsv")
    add(area, "nouns made from a verb", statement.NOMINALS_VERSION, STARTING,
        "competency/verbs/deverbal_nouns_v1.tsv",
        f"{len(statement.NOMINALS)} nouns ('development' = develop); the developer's list, "
        "unreviewed. A phrase that names its work with one of them takes the verb's level "
        "(the researcher's correction, 2026-11-20)")
    add(area, "verbs that only say 'do it'", sorted(statement.LIGHT_VERBS), STARTING,
        "competency/statement.py", "in 'perform code reviews' the work is the noun's verb")
    from competency import smoothing
    from governance import zoning
    z = {f.name: f.default for f in dataclasses.fields(zoning.ZoningConfig)}
    add(area, "sentence filter: duty and requirement sentences only",
        {k: v for k, v in z.items() if k != "version"}, STARTING, "governance/zoning.py",
        "weak labels from section headers plus a classifier; not validated against people; "
        "the cut-off 0.5 was set from the developer's reading of 64 sentences (EXP-030)")
    add(area, "sentence filter version", zoning.ZONING_VERSION, DESIGN, "governance/zoning.py")
    add(area, "intervals, pooled level mix, rank ranges", smoothing.SMOOTHING_VERSION, DESIGN,
        "competency/smoothing.py",
        f"{smoothing.BOOTSTRAP_ROUNDS} bootstrap rounds, seed {smoothing.BOOTSTRAP_SEED}; the "
        "strength of the pooling is estimated from the data")
    from competency import formulate, qualifiers, skkni
    add(area, "SKKNI parser", skkni.PARSER_VERSION, DESIGN, "competency/skkni.py",
        "every SKKNI PDF in DATA/master_skkni, whatever its year")
    add(area, "how a competency statement is formulated",
        "like an SKKNI unit title: an operational verb on the competency's level, the specific "
        "object of work, and the standard, quality and outcome its postings ask for; no tool "
        "names. The competencies come from the postings, not from SKKNI",
        DECIDED, "competency/formulate.py",
        "the promotors' definition and the researcher's correction, 2026-11-20")
    add(area, "SKKNI wording in a statement",
        "used when a competency IS an SKKNI unit; only when the SKKNI title's verb is on the "
        "competency's level (otherwise both levels are shown)", STARTING,
        "competency/formulate.py",
        "the researcher: 'you might use the SKKNI wordings'; the level condition is the developer's")
    add(area, "a 'same as SKKNI' decision needs a second model",
        "the writer's 'sama' stands only when a second model, shown the competency and that one "
        "unit and nothing the writer decided, also says 'sama'; otherwise it becomes 'sebagian' "
        "and the market's statement is kept", STARTING, "competency/formulate.py",
        "the developer's rule: the writer alone was wrong in 4 of its 10 'sama' calls on the "
        "development sample (EXP-032)")
    add(area, "second model for the 'same as SKKNI' decision", formulate.CONFIRM_MODEL, STARTING,
        "competency/formulate.py", "another family than the writer (GPT) and than the reference "
        "judges (Claude); one definition of 'sama' is shared by both prompts")
    add(area, "a statement names the standard or quality its postings ask for",
        "a qualifier that fits the object is written (the most frequent, at most two); when "
        "none fits, the writer must say why", DECIDED, "competency/formulate.py",
        "the researcher, 2026-11-21 (the writer had left one out in 8 of 19 held-out cases, EXP-032)")
    add(area, "competencies that are not work, or mix different works, are marked",
        "marked, not left out; a mark changes no status", DECIDED, "competency/formulate.py",
        "the researcher, 2026-11-21")
    add(area, "what counts as 'not work' and as a 'mixed family'",
        "a focused second model's judgement (MARK_SYSTEM: work behaviour, purpose clause, "
        "employer text, benefit, call to apply, self-development, non-software task, cut phrase; "
        "a family whose most frequent object covers only a small part of its postings)", STARTING,
        "competency/formulate.py", "the developer's wording of the reviewers' categories (EXP-032)")
    add(area, "model that marks", formulate.MARK_MODEL, STARTING, "competency/formulate.py",
        "a focused call, shown only the evidence; in a dry run the writer, asked to mark as a "
        "sixth task, missed a case its own prompt named (EXP-034)")
    add(area, "'kalimat berulang' mark (one employer's text)",
        {"distinct sentences per posting at most": formulate.FormulateConfig.repeated_sentence_ratio,
         "with at least postings": formulate.FormulateConfig.repeated_sentence_min_postings},
        STARTING, "competency/formulate.py",
        "on e0_pool_all_v8 the 10 review candidates at or below 0.3 were all employer text (EXP-034)")
    fc = {f.name: f.default for f in dataclasses.fields(formulate.FormulateConfig)}
    add(area, "when a standard, quality or outcome enters a statement",
        {k: v for k, v in fc.items() if k != "version"}, STARTING, "competency/formulate.py")
    add(area, "statement formulation version", formulate.FORMULATE_VERSION, DESIGN,
        "competency/formulate.py")
    add(area, "statement writer model", endtoend.STATEMENT_MODEL, STARTING,
        "competency/formulate.py", "checked by the module's own checker; EXP-032")
    add(area, "quality words and their ISO/IEC 25010 characteristic",
        f"{len(qualifiers.QUALITY_TERMS)} words", STARTING, "competency/qualifiers.py",
        "the developer's mapping; check the standard before citing")
    add(area, "named practices and standards read from postings",
        [n for n, _ in qualifiers.NAMED_STANDARDS], STARTING, "competency/qualifiers.py")
    from competency import structure as _structure
    add(area, "families that give the same statement on the same level are merged",
        f"only through a statement that names specific work; {len(_structure.GENERIC_NOUNS)} "
        "general nouns ('solutions', 'applications', 'issues') link nothing", STARTING,
        "competency/structure.py", "the developer's rule (EXP-033)")
    add(area, "a statement cut before its object is held back",
        "'statement_is_a_fragment' blocks the status kandidat tinjauan and is not formulated",
        STARTING, "competency/statement.py", "the developer's rule (EXP-033)")
    add(area, "one competency per cognitive level of a group",
        "a group (family) is split by the level of its verbs; each level is a competency with "
        "its own statement and its own number of postings", DECIDED, "competency/structure.py",
        "the researcher, 2026-11-20")
    add(area, "grounding measure", statement.GROUNDING_VERSION, DESIGN, "competency/statement.py")
    add(area, "technologies an LLM statement may name",
        statement.MAX_TECHNOLOGIES_IN_STATEMENT, STARTING, "competency/statement.py",
        "the short LLM writer of EXP-019 only; out of step with the decision that a "
        "statement names no tool, and superseded by the earlier generator")
    k = {f.name: f.default for f in dataclasses.fields(ContractConfig)}
    add(area, "contract: minimum postings / minimum grounding",
        [k["min_postings"], k["min_grounding"]], STARTING, "competency/contract.py",
        "to be agreed with the promotor before any release")
    add(area, "KKNI step", mapping.KKNI_STEP_VERSION, STARTING, "competency/mapping.py",
        "confident on 1 of 445 statements (EXP-024); needs another method")
    v = {f.name: f.default for f in dataclasses.fields(coverage.CoverageConfig)}
    add(area, "coverage thresholds (partial / covered)", [v["partial_at"], v["covered_at"]],
        STARTING, "competency/coverage.py", "not calibrated against expert judgement")
    t = {f.name: f.default for f in dataclasses.fields(temporal.TemporalConfig)}
    add(area, "time analysis settings", {x: t[x] for x in t if x not in (
        "thresholds_status", "version")}, STARTING, "competency/temporal.py",
        "the proposal requires expert validation of these")
    add(area, "time analysis version", temporal.TemporalConfig.version, DESIGN,
        "competency/temporal.py")
    add(area, "delta rule version", delta.DELTA_VERSION, DESIGN, "competency/delta.py",
        "'mengurangi penekanan' capped at eksploratif by the proposal")
    add(area, "profile: same-competency overlap (Jaccard)", profile.MATCH_JACCARD, STARTING,
        "competency/profile.py")
    add(area, "end-to-end models (extraction / statements)",
        [endtoend.EXTRACTION_MODEL, endtoend.STATEMENT_MODEL], STARTING, "competency/endtoend.py",
        "rehearsal choices, not registered")

    # ---- acquisition
    roster = json.loads((REPO_ROOT / "acquisition" / "configs" / "monthly_roster.json").read_text(
        encoding="utf-8"))
    area = "Corpus collection"
    add(area, "monthly roster version", roster["roster_version"], DESIGN,
        "acquisition/configs/monthly_roster.json")
    add(area, "limits for detail-fetch platforms", roster["limits"], DECIDED,
        "docs/corpus/ACQUISITION_CADENCE.md", "standing constraints; not to be raised")
    add(area, "sources and cadence (days)",
        {s["key"]: s["cadence_days"] for s in roster["sources"] if s.get("enabled", True)},
        DECIDED, "docs/corpus/SOURCE_AUDIT.md", "each source has a recorded Kn decision")
    return rows


def to_markdown(rows: list[dict]) -> str:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    out = ["# Configuration registry", "",
           "Generated by `python -m governance.config_registry` from the code. Do not edit by "
           "hand: change the code or the registration, then regenerate. A test fails when this "
           "document and the code disagree.", "",
           "| Status | Meaning | Entries |", "|---|---|---|",
           f"| registered | fixed in a registration document before data existed | {counts.get(REGISTERED, 0)} |",
           f"| decided | a recorded decision of the researcher | {counts.get(DECIDED, 0)} |",
           f"| by design | part of a definition: a version, a seed, a hash | {counts.get(DESIGN, 0)} |",
           f"| **starting value** | **not yet agreed; must be fixed before any confirmatory result or released profile depends on it** | **{counts.get(STARTING, 0)}** |",
           ""]
    area = None
    for r in rows:
        if r["area"] != area:
            area = r["area"]
            out += ["", f"## {area}", "", "| Setting | Value | Status | Where | Note |",
                    "|---|---|---|---|---|"]
        value = r["value"].replace("|", "\\|")
        status = f"**{r['status']}**" if r["status"] == STARTING else r["status"]
        out.append(f"| {r['name']} | `{value}` | {status} | `{r['where']}` | {r['note']} |")
    out += ["", "## Still to be fixed", "",
            "Every entry marked **starting value**, in one list:", ""]
    out += [f"- {r['area']}: {r['name']}" for r in rows if r["status"] == STARTING]
    return "\n".join(out) + "\n"


def write() -> list[dict]:
    rows = collect()
    DOC.write_text(to_markdown(rows), encoding="utf-8", newline="\n")
    JSON_OUT.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                        newline="\n")
    return rows


def check() -> bool:
    """True when the committed registry is what the code generates now."""
    if not DOC.is_file():
        return False
    return DOC.read_text(encoding="utf-8").replace("\r\n", "\n") == to_markdown(collect())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        ok = check()
        print("registry is up to date" if ok else
              "registry is OUT OF DATE: run python -m governance.config_registry")
        raise SystemExit(0 if ok else 1)
    rows = write()
    n = sum(1 for r in rows if r["status"] == STARTING)
    print(f"{len(rows)} entries, {n} still starting values -> {DOC}")


if __name__ == "__main__":
    main()
