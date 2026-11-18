# Scientific Evaluation Protocol

This document is the evaluation contract for the current v2 research pipeline.
It defines what the system is allowed to claim, what evidence must be produced,
which baselines must be compared, and which thresholds must be met before a
pipeline output can be treated as research-grade.

The system is a competency recommendation system. It takes job postings as
labor-market evidence, extracts skills and knowledge, groups related evidence,
generates competency recommendations, labels those competencies with contextual
metadata, and ranks them using demand, trend, and future-work signals.

This protocol should be treated as the stable scientific control document for
experiments. Implementation files may change; the evaluation commitments below
should not change without recording a protocol revision.

---

## 1. Core Research Claim

The defensible claim is:

> The system can generate traceable, evidence-based competency recommendations
> from job postings and future-work insight sources, with measurable grounding,
> coherence, stability, and expert usefulness for curriculum design.

The system does not claim to produce the one correct curriculum. It produces
ranked recommendations that curriculum stakeholders can inspect, accept,
reject, or adapt.

---

## 2. Non-Negotiable Design Principles

1. **No black box recommendations**
   - Every competency must be traceable to source job postings, source
     sentences, extracted skills/knowledge, cluster membership, and future-work
     signals.

2. **Bloom taxonomy is removed from the pipeline**
   - The system must not classify Bloom levels.
   - Bloom-level interpretation is left to curriculum stakeholders.

3. **KKNI and education level are informational labels**
   - KKNI level is post-hoc metadata.
   - Education level is derived from job-posting education requirements when
     available.
   - Neither KKNI nor education labels may directly determine the priority
     ranking unless a later protocol explicitly adds that as a tested variant.

4. **Hard-skill competency first, soft-skill description second**
   - Generated competencies focus on hard skills and knowledge.
   - Soft skills may be described as supporting requirements for each
     competency, but they should not become separate hard-skill competencies
     unless the evidence supports that.

5. **Evidence beats fluency**
   - A fluent competency with weak evidence is not acceptable.
   - A competency must pass grounding and traceability gates before it is
     considered valid for recommendation.

---

## 3. Current Pipeline Under Evaluation

The current v2 pipeline should be evaluated as the following stages:

```text
Raw job postings
      |
      v
Job cleaning and sentence preparation
      |
      v
Sentence relevance filtering
      |
      v
Skill and knowledge extraction
      |
      v
Normalization and evidence consolidation
      |
      v
Skill/knowledge clustering
      |
      v
Competency generation
      |
      v
Grounding, text-quality, and hallucination checks
      |
      v
KKNI, education-level, occupation, and future-domain labels
      |
      v
Priority ranking
      |
      v
Public recommendation and curriculum coverage interface
```

### Current Important Components

| Component | Current role |
|---|---|
| `sentence_relevance_filter.py`, `it_relevance_filter.py` | Keep requirement-relevant text and reduce noise |
| `baseline_versions/skill_llm/` | Layer-1 sentence-level extractor evidence and training/evaluation artifacts |
| `pipeline.py` / v2 extraction exports | Production extraction and source evidence |
| `clustering/skill_clusterer.py` | Groups related extracted skills/knowledge into coherent clusters |
| `competency_generator_v2.py` | Generates cluster-driven competencies with rationale and evidence references |
| `competency_evaluator.py` | Computes grounding/coherence/coverage and flags weak competencies |
| `competency_kkni_labeller.py` | Adds informational KKNI labels |
| `competency_education_aggregator.py` | Adds education-level labels from job evidence |
| `occupation_mapper.py` | Adds occupational context labels |
| `future_weight_mapping.py` | Maps extracted items to future-work domains |
| `scripts/run_full_v2_pipeline.py` | Current full v2 pipeline runner |
| `dashboard_v2/app.py` | Public-facing recommendation and coverage interface |

---

## 4. Units Of Analysis

The evaluation uses several units. They must not be mixed.

| Unit | Meaning | Example metric |
|---|---|---|
| Sentence | One job-posting sentence | relevance precision/recall |
| Span | Extracted skill/knowledge text span | strict span F1 |
| Extracted item | Normalized skill/knowledge with provenance | item correctness |
| Cluster | Group of related extracted items | cohesion score, expert coherence |
| Competency | Generated curriculum recommendation | grounding, validity, usefulness |
| Recommendation list | Ranked competencies | Precision@20, NDCG@20, stability |

If a report says "accuracy", it must specify which unit is being evaluated.

---

## 5. Research Questions And Gates

### RQ1. Extraction Quality

**Question:** Does the extraction layer identify job-relevant skills and
knowledge with sufficient quality to support downstream competency generation?

#### Automatic Evaluation

Use SkillSpan or another exhaustively annotated extraction benchmark.

| Metric | Target / decision rule |
|---|---|
| Total strict span F1 | >= 0.55 for production Layer 1 |
| Skill strict span F1 | Report |
| Knowledge strict span F1 | Report |
| Verb-short rate | <= 0.244 |
| Parse failure rate | <= 0.02 |

#### Production Gold-Set Evaluation

Use a manually labeled sample from the actual job-posting corpus.

| Metric | Target / decision rule |
|---|---|
| Skill item precision | >= 0.70 |
| Knowledge item precision | >= 0.70 |
| Wilson 95% CI lower bound | Report |
| Inter-rater reliability | Cohen/Fleiss kappa >= 0.41 minimum; >= 0.60 target |

#### Required Baselines

| Baseline | Purpose |
|---|---|
| Skill-LLM / Layer 1 only | Determine sentence-level extraction quality |
| Full-posting LLM only | Determine whether context helps |
| Hybrid/fused output | Determine whether production fusion improves precision/coverage |
| Older JobBERT path, if retained | Historical comparison, not necessarily production |

---

### RQ2. Sentence Relevance Filtering

**Question:** Does the system remove irrelevant job-posting text without
discarding important skill/knowledge evidence?

#### Gold-Set Design

Sample sentences from job postings and label each as:

- `relevant`: contains job requirement, skill, knowledge, task, tool, method,
  responsibility, qualification, or education requirement.
- `irrelevant`: company overview, benefit, salary, logistics, legal boilerplate,
  application instruction, or unrelated promotional text.

#### Metrics

| Metric | Target / decision rule |
|---|---|
| Relevant-class precision | >= 0.90 |
| Relevant-class recall | >= 0.85 |
| Retention rate | Report |
| False-negative audit | Mandatory review of discarded requirement sentences |

#### Failure Rule

If recall for relevant sentences is below 0.85, the filter must become more
conservative. In uncertain cases, the pipeline should keep the sentence rather
than discard it.

---

### RQ3. Cluster Coherence

**Question:** Are extracted skills and knowledge grouped into coherent clusters
before competency generation?

#### Automatic Metrics

| Metric | Target / decision rule |
|---|---|
| Mean intra-cluster SBERT cosine | >= 0.50 minimum; higher is better |
| Minimum cluster cohesion | Report and audit |
| Oversized cluster count | Report |
| Noise/drop count by stage | Report |
| Singleton or ungrouped items | Report |

#### Expert Review

Reviewers see cluster items without the generated competency and rate:

| Label | Meaning |
|---|---|
| 1 | Unrelated items; should not be one cluster |
| 2 | Weak relation; substantial noise |
| 3 | Acceptable relation with some noise |
| 4 | Strongly coherent |
| 5 | Very coherent and curriculum-ready |

Target: mean expert cluster coherence >= 3.5.

#### Required Baselines

| Baseline | Purpose |
|---|---|
| Old domain-batched grouping | Test whether v2 clustering improves coherence |
| Demand-only grouping or frequency-only top list | Show why clustering is needed |
| HDBSCAN vs agglomerative, if both available | Select method by empirical cohesion and expert rating |

---

### RQ4. Competency Generation Quality

**Question:** Are generated competencies grounded, coherent, useful, and not
hallucinated?

#### Automatic Metrics

| Metric | Definition | Target / decision rule |
|---|---|---|
| Grounding score | Fraction of related skills traceable to source evidence | >= 0.80 per competency |
| Mean grounding score | Mean over final competency set | >= 0.85 target |
| Coherence score | Mean pairwise similarity among related skills | Report; compare to baseline |
| Evidence count | Unique jobs/sentences supporting the competency | Report |
| Hallucinated skill count | Related skills not found in contributing evidence | 0 for accepted competencies |
| Text quality gate | Clear title/description, no malformed output | Must pass |

#### Expert Review Dimensions

Each competency should be reviewed by domain/curriculum experts using:

| Dimension | Scale | Target |
|---|---|---|
| Validity | 1-5 | mean >= 3.5 |
| Relevance for curriculum design | yes/partial/no | P@20 target below |
| Grounding clarity | 1-5 | mean >= 3.5 |
| Rationale clarity | 1-5 | mean >= 3.5 |
| Actionability | 1-5 | mean >= 3.5 |

#### Competency Rejection Rules

Reject or flag a competency if any of these are true:

1. Grounding score < 0.80.
2. It contains related skills not present in the evidence chain.
3. It combines visibly unrelated skills.
4. It is too broad to be actionable.
5. It is merely a job title, tool name, or topic name rather than a competency.
6. It cannot show source job postings or source sentences.

---

### RQ5. Recommendation Ranking Quality

**Question:** Do the top-ranked competencies match expert priorities for
curriculum design?

#### Default Ranking Formula

The v2 system may use competency-level signals. The default current scoring
logic must be documented in each run report. If the formula changes, record
the exact formula and version.

Known current direction:

```text
priority = 0.40 * demand + 0.30 * grounding_or_trend + 0.30 * future_weight
```

Older skill-level formula:

```text
priority = 0.40 * demand + 0.30 * trend + 0.30 * future_weight
```

The exact active formula in `pipeline_report.json` is the authoritative record
for each experiment.

#### Metrics

| Metric | Target / decision rule |
|---|---|
| Precision@20 | >= 0.60 |
| NDCG@20 | >= 0.60 |
| Expert top-20 relevance | >= 0.80 yes/partial combined |
| Rank explanation completeness | 100% of top-20 has score breakdown |

#### Required Ablations

| Ablation | Purpose |
|---|---|
| Demand only | Establish a simple labor-market baseline |
| No future signal | Test contribution of future-work insight |
| No trend signal, if trend is active | Test contribution of empirical time trend |
| No grounding/quality component, if active | Test whether quality-aware ranking matters |
| Old competency generator vs v2 generator | Test value of cluster-driven generation |

---

### RQ6. Production-Volume Stability

**Question:** At what corpus size do the recommendations stabilize?

#### Experimental Design

Run the full v2 pipeline at:

```text
N = 200, 500, 1000, 2500, 5000, 10000
```

Use at least 3 random seeds for each N when compute budget permits.

#### Metrics

| Metric | Target / decision rule |
|---|---|
| Top-20 Jaccard across seeds at same N | >= 0.80 at declared stable N |
| Top-20 Jaccard between N and previous N | >= 0.85 at declared stable N |
| Mean grounding score by N | Report |
| Mean expert validity by N, if reviewed | Report |
| Runtime and cost by N | Report |

#### Stability Claim Rule

Do not claim production stability until both same-N seed stability and
between-N stability pass.

---

### RQ7. Future-Work Signal Validity

**Question:** Does future-domain mapping align with expert judgment and improve
the usefulness of recommendations?

#### Metrics

| Metric | Target / decision rule |
|---|---|
| Future-domain Top-1 accuracy | >= 0.60 |
| Future-domain Top-3 accuracy | >= 0.80 |
| Mapping margin | Report |
| Ranking delta vs no-future ablation | Report |
| Sensitivity to trend-score perturbation | Top-20 Jaccard >= 0.70 target |

#### Important Limitation

Future-domain scores are expert-informed ordinal signals, not ground-truth
measurements. The research should report sensitivity analysis rather than
pretend the exact numbers are objectively calibrated.

---

## 6. Expert Review Protocol

### Reviewer Types

The study should aim for at least two reviewer groups:

1. **Curriculum/domain experts**
   - Evaluate whether competencies are useful and curriculum-appropriate.

2. **Technical/software experts**
   - Evaluate whether skills/knowledge and competency groupings are technically
     coherent.

If only one group is available, report that limitation explicitly.

### Review Sets

| Review set | Recommended size |
|---|---|
| Sentence relevance | 200 sentences |
| Extraction correctness | 150 skills + 100 knowledge items |
| Cluster coherence | 50 clusters |
| Competency validity | 75 competencies |
| Top recommendation list | Top 20 or top 30 competencies |

### Inter-Rater Reliability

Use overlap items for at least 20-30% of the review set.

| Setup | Metric |
|---|---|
| 2 reviewers | Cohen's kappa |
| 3+ reviewers | Fleiss' kappa or Krippendorff's alpha |
| Ordinal 1-5 ratings | Weighted kappa or ICC, if available |

Minimum acceptable agreement: kappa >= 0.41. Target: kappa >= 0.60.

---

## 7. Provenance Audit

Every accepted competency must provide this chain:

```text
Competency
  -> related skills/knowledge
  -> contributing item IDs
  -> source sentence IDs
  -> source sentence text
  -> source job IDs
  -> demand/future/trend signals used in scoring
```

### Provenance Completeness Metrics

| Metric | Target |
|---|---|
| Competencies with at least one source job | 100% |
| Competencies with at least one source sentence | 100% |
| Related skills traceable to evidence | >= 80% per competency |
| Top-20 with score breakdown | 100% |
| Top-20 with visible rationale | 100% |

### Manual Provenance Spot Check

For the final candidate run, manually inspect at least 20 competencies:

1. Open competency.
2. Read related skills.
3. Read source sentences.
4. Confirm sentences actually support the competency.
5. Mark any weak chain as `provenance_concern = true`.

Report the concern rate.

---

## 8. Dataset And Run Versioning

Every experiment must save a `pipeline_report.json` or equivalent run metadata
with:

| Field | Required |
|---|---|
| Run tag | yes |
| Input corpus path and sample size | yes |
| Random seed | yes |
| Extraction model/version | yes |
| LLM provider/model | yes |
| Prompt version/hash | yes |
| Relevance filter version | yes |
| Clustering config | yes |
| Competency generator version | yes |
| KKNI/education labeler version | yes |
| Ranking formula | yes |
| Runtime and cost estimate | recommended |

Generated experiment folders should not be committed to Git by default.
Instead, commit code, protocols, and small summary tables. Large results should
be archived separately with a run tag and checksum if they are used in the
dissertation.

---

## 9. Reproducibility Workflow

Recommended experiment workflow:

```text
1. Choose run tag
2. Freeze code commit hash
3. Freeze input dataset/sample
4. Run extraction/relevance/clustering/generation
5. Save pipeline_report.json
6. Run automatic evaluation
7. Export human-review packets
8. Import reviewer labels
9. Generate final evaluation report
10. Record pass/fail decision against this protocol
```

No experiment should be treated as final if the code commit hash is unknown.

---

## 10. Minimum Evidence Package For Dissertation Use

A run can be cited as a main dissertation result only if it has:

1. A clean Git commit hash.
2. A run metadata file.
3. An extraction quality report.
4. A relevance-filter report.
5. A clustering quality report.
6. A competency evaluation report.
7. A recommendation ranking report.
8. A provenance completeness report.
9. At least one expert-review dataset.
10. Inter-rater reliability, if more than one reviewer participated.

---

## 11. Current Candidate Baseline Note

As of the current repository snapshot, the latest visible local v2 experiment is:

```text
results/competency_v2_pipeline_n1k_v10d_with_okupasi/
```

Its `pipeline_report.json` reports:

| Field | Value |
|---|---|
| Input extracted items | 11,842 |
| Items after frequency/noise filtering | 564 |
| Generated competencies before final cap | 108 |
| Final recommendation cap | 30 |
| Mean cluster cohesion | 0.7992 |
| Median cluster cohesion | 0.8502 |
| Final formula recorded | 0.40 * demand + 0.30 * grounding + 0.30 * future_weight |

This run may be treated as a candidate engineering baseline, not yet as a final
research result. It still needs the full evaluation package described above.

---

## 12. Pass/Fail Decision Table

| Layer | Pass condition | If fail |
|---|---|---|
| Sentence relevance | precision >= 0.90 and recall >= 0.85 | Make filter more conservative; review false negatives |
| Extraction | benchmark gate passes and production precision >= 0.70 | Improve extractor/prompt/fusion or narrow claim |
| Clustering | mean automatic cohesion acceptable and expert coherence >= 3.5 | Tune clustering, split broad clusters, improve noise removal |
| Competency generation | grounding >= 0.80 per accepted competency and expert validity >= 3.5 | Reject weak competencies; revise generator prompt/schema |
| Recommendation ranking | P@20 >= 0.60 and NDCG@20 >= 0.60 | Revisit ranking formula and ablation results |
| Stability | top-20 Jaccard passes at declared N | Increase sample size or narrow production claim |
| Traceability | 100% of top recommendations trace to evidence | Do not publish recommendations until fixed |

---

## 13. What To Do Next

Immediate next actions:

1. Freeze a clean code checkpoint.
2. Decide which v2 run is the current candidate baseline.
3. Generate review packets for:
   - sentence relevance,
   - extraction correctness,
   - cluster coherence,
   - competency validity,
   - top recommendation usefulness.
4. Run automatic evaluation reports.
5. Use expert review results to decide whether the v2 system is ready for a
   larger N stability experiment.

The next research milestone is not "more generated competencies." It is:

> A traceable candidate run that passes automatic gates and is ready for expert
> validation.

