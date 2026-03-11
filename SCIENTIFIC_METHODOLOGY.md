# Scientific Methodology

This document provides a **detailed, scientifically rigorous** description of all statistical methods, formulas, and decision logic used in the Future-Aware Hybrid Skill Extraction pipeline. It includes worked examples for every major procedure.

**Related documents:**
- [CALCULATIONS.md](CALCULATIONS.md) — Formulas and summary tables
- [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md) — Research questions, gold set design, limitations
- [PIPELINE.md](PIPELINE.md) — Pipeline flow, data dependencies

---

## Table of Contents

1. [Binomial Test for Extraction Precision](#1-binomial-test-for-extraction-precision)
2. [Effect Sizes for Proportions](#2-effect-sizes-for-proportions)
3. [Multi-Comparison Correction](#3-multi-comparison-correction)
4. [Power Analysis for Gold Set Size](#4-power-analysis-for-gold-set-size)
5. [Inter-Rater Reliability (Cohen's Kappa and Fleiss' Kappa)](#5-inter-rater-reliability-cohens-kappa-and-fleiss-kappa)
6. [Normalization and Grouping](#6-normalization-and-grouping)
7. [Priority Score and Future Weight](#7-priority-score-and-future-weight)
8. [FDR-Controlled Trend Detection](#8-fdr-controlled-trend-detection)
9. [Weight Sensitivity Analysis](#9-weight-sensitivity-analysis)
10. [Recall in the Gold-Set Paradigm](#10-recall-in-the-gold-set-paradigm)
11. [Calibrated Verification Metrics](#11-calibrated-verification-metrics)
12. [Future-Domain Mapping Evaluation](#12-future-domain-mapping-evaluation)
13. [Domain-Based Batching for Competency Generation](#13-domain-based-batching-for-competency-generation)
14. [Recommendation Evaluation (P@N, NDCG)](#14-recommendation-evaluation-pn-ndcg)

---

## 1. Binomial Test for Extraction Precision

### Purpose

To test whether an extractor's precision is better than chance. We use a one-sided binomial test against the null hypothesis that precision = 0.5 (random guessing).

### Hypothesis

- **H₀**: precision = 0.5
- **H₁**: precision > 0.5

### Formula

Let X ~ Binomial(n, p₀) where n = number of labeled items, p₀ = 0.5. We reject H₀ if the observed count of correct extractions is in the critical region:

```
P(X ≥ k | p₀ = 0.5) ≤ α
```

We choose the smallest k such that P(X ≥ k | p₀) ≤ α = 0.05. The p-value is:

```
p-value = P(X ≥ observed_correct | p₀ = 0.5)
```

### Implementation

Uses `scipy.stats.binomtest(n_correct, n, p=0.5, alternative="greater")`.

### Worked Example

**Scenario:** Hybrid extractor has 120 correct out of 150 labeled skills.

- n = 150, n_correct = 120
- precision = 120/150 = 0.80
- H₀: p = 0.5
- p-value = P(X ≥ 120 | n=150, p=0.5)

For large n, the binomial distribution is approximately normal. In practice:

- Expected under H₀: 75 correct
- Observed: 120 correct
- The p-value is extremely small (≈ 0), so we **reject H₀** and conclude precision is significantly better than chance.

**Result:** `p_value_vs_chance ≈ 0`, `significant_at_005 = True`

### Wilson Score Confidence Interval

For a proportion p with n observations, the Wilson score interval gives better coverage than the normal approximation, especially for small n or extreme p:

```
centre = p + z²/(2n)
margin = z × √(p(1-p)/n + z²/(4n²))
denom = 1 + z²/n
lower = (centre - margin) / denom
upper = (centre + margin) / denom
```

With z = 1.96 (95% CI).

**Example:** precision p = 0.72, n = 100  
centre = 0.72 + 3.84/200 = 0.739, margin ≈ 0.088, denom ≈ 1.038  
CI ≈ **[0.628, 0.798]**

---

## 2. Effect Sizes for Proportions

### Odds Ratio (vs Chance)

When comparing precision p against chance p₀ = 0.5:

```
odds(p) = p / (1 - p)
odds(p₀) = p₀ / (1 - p₀) = 1   (when p₀ = 0.5)

odds_ratio = odds(p) / odds(p₀) = p / (1 - p)
```

**Interpretation:**
- OR = 1: no effect vs chance
- OR > 1: better than chance (e.g., OR = 4 → 4× more likely to be correct than incorrect)
- OR < 1: worse than chance

### Risk Difference

```
risk_difference = p - p₀
```

**Interpretation:** Absolute increase in correct proportion over chance. For p₀ = 0.5, risk_difference = p - 0.5.

### Cohen's h (Comparing Two Proportions)

For comparing two proportions p₁ and p₂:

```
h = 2 × (arcsin(√p₁) - arcsin(√p₂))
```

**Effect size interpretation:**
- |h| ≈ 0.2: small
- |h| ≈ 0.5: medium
- |h| ≈ 0.8: large

### Worked Examples

**Example 1 — Odds ratio and risk difference**

- precision p = 0.80, p₀ = 0.5
- odds_ratio = 0.80 / 0.20 = **4.0**
- risk_difference = 0.80 - 0.5 = **0.30**

**Example 2 — Cohen's h (BERT vs LLM)**

- BERT precision p₁ = 0.65, LLM precision p₂ = 0.50
- h = 2 × (arcsin(√0.65) - arcsin(√0.50))
- arcsin(√0.65) ≈ 0.947, arcsin(√0.50) ≈ 0.785
- h ≈ 2 × 0.162 ≈ **0.32** (small-to-medium effect)

---

## 3. Multi-Comparison Correction

When testing multiple sources (BERT, LLM, Hybrid) or performing pairwise comparisons, we correct for multiple testing to control the family-wise error rate (FWER) or false discovery rate (FDR).

### Bonferroni Correction (Per-Source Tests)

For k independent tests, multiply each p-value by k, or equivalently use α/k instead of α:

```
p_adjusted = min(1, p_raw × k)
```

**Example:** 3 sources, raw p-values [0.02, 0.08, 0.001]. Bonferroni:
- 0.02 × 3 = 0.06 (not significant at 0.05)
- 0.08 × 3 = 0.24
- 0.001 × 3 = 0.003 (still significant)

### Benjamini-Hochberg (Pairwise Comparisons)

For pairwise two-proportion z-tests between sources, we apply Benjamini-Hochberg FDR:

1. Sort p-values ascending: p_(1) ≤ p_(2) ≤ … ≤ p_(m)
2. Find largest k such that p_(k) ≤ (k/m) × α
3. Reject hypotheses 1..k; q-value for each: q_i = p_i × m / rank_i

**Example:** 3 pairwise comparisons, p-values [0.03, 0.06, 0.12], α = 0.05

| Rank | p-value | BH threshold (k/3 × 0.05) | Reject? |
|------|---------|---------------------------|---------|
| 1 | 0.03 | 0.0167 | No (0.03 > 0.0167) |
| 2 | 0.06 | 0.0333 | No |
| 3 | 0.12 | 0.05 | No |

No rejections after BH. If p-values were [0.01, 0.02, 0.10]:
- 0.01 ≤ 0.0167 → reject first
- 0.02 ≤ 0.0333 → reject second
- 0.10 > 0.05 → do not reject third

### Two-Proportion Z-Test

For comparing BERT (n₁, x₁ correct) vs LLM (n₂, x₂ correct):

```
p₁ = x₁/n₁,  p₂ = x₂/n₂
p_pool = (x₁ + x₂) / (n₁ + n₂)
SE = √[p_pool(1 - p_pool) × (1/n₁ + 1/n₂)]
z = (p₁ - p₂) / SE
p-value = 2 × (1 - Φ(|z|))   [two-tailed]
```

---

## 4. Power Analysis for Gold Set Size

### Purpose

Determine the minimum number of labeled items (n) needed to detect a meaningful precision improvement over chance with adequate statistical power.

### Assumptions

| Parameter | Default | Meaning |
|-----------|---------|---------|
| p₀ | 0.5 | Null (chance) |
| p₁ | 0.7 | Alternative (RQ1 target) |
| α | 0.05 | Type I error |
| Power | 0.80 | 1 - β (Type II error) |

### Method

1. Find critical value k: smallest integer such that P(X ≥ k | p₀, n) ≤ α
2. Power = P(X ≥ k | p₁, n)

We search for smallest n such that power ≥ 0.80.

### Worked Example

- p₀ = 0.5, p₁ = 0.7, α = 0.05, target power = 0.80
- For n = 37: critical k ≈ 24; P(X ≥ 24 | p₁=0.7, n=37) ≈ 0.80
- **Result:** n ≥ 37 required for power 0.80

**Current gold set:**
- Skills: n = 150 → power ≈ 1.00
- Knowledge: n = 100 → power ≈ 0.99

### Running the Script

```bash
python scripts/power_analysis_gold_set.py --p0 0.5 --p1 0.7
```

---

## 5. Inter-Rater Reliability (Cohen's Kappa and Fleiss' Kappa)

### When to Use Which

| Raters | Metric |
|--------|--------|
| 2 | Cohen's Kappa |
| 3+ | Fleiss' Kappa |

### Cohen's Kappa (2 Raters)

```
κ = (P_o - P_e) / (1 - P_e)
```

- **P_o** = observed agreement = fraction of items where both raters agree
- **P_e** = expected agreement by chance = Σ (freq₁(j) × freq₂(j)) over categories j

**Interpretation:**
- κ < 0: worse than chance
- κ = 0: chance agreement
- κ ∈ (0, 0.20]: slight
- κ ∈ (0.20, 0.40]: fair
- κ ∈ (0.40, 0.60]: moderate
- κ ∈ (0.60, 0.80]: substantial
- κ > 0.80: almost perfect

**Example:**

| Item | Rater A | Rater B |
|------|---------|---------|
| 1 | valid | valid |
| 2 | valid | invalid |
| 3 | invalid | invalid |
| 4 | valid | valid |
| 5 | invalid | invalid |

- P_o = 4/5 = 0.80
- P_e: freq_A(valid)=3/5, freq_A(invalid)=2/5; freq_B(valid)=2/5, freq_B(invalid)=3/5
- P_e = (3/5)(2/5) + (2/5)(3/5) = 12/25 + 6/25 = 0.72
- κ = (0.80 - 0.72) / (1 - 0.72) = 0.08 / 0.28 ≈ **0.29** (fair agreement)

### Fleiss' Kappa (3+ Raters)

For each subject i and category j, let n_ij = number of raters who assigned subject i to category j. N = number of raters, n = number of subjects.

```
P_i = (1 / (N×(N-1))) × Σ_j [n_ij × (n_ij - 1)]
P̄ = (1/n) × Σ_i P_i

p_j = proportion of all assignments to category j
P_e = Σ_j p_j²

κ = (P̄ - P_e) / (1 - P_e)
```

**Example:** 3 raters, 5 items, 2 categories (valid, invalid)

| Item | valid | invalid |
|------|-------|---------|
| 1 | 3 | 0 |
| 2 | 2 | 1 |
| 3 | 0 | 3 |
| 4 | 3 | 0 |
| 5 | 1 | 2 |

- For item 1: P₁ = (1/6) × [3×2 + 0] = 1
- For item 2: P₂ = (1/6) × [2×1 + 1×0] = 0.33
- … similarly for others
- P̄ = mean of P_i
- p_valid, p_invalid from column sums
- P_e = p_valid² + p_invalid²
- κ = (P̄ - P_e) / (1 - P_e)

---

## 6. Normalization and Grouping

### Normalization Rule

To merge equivalent items (e.g., "AI agents" vs "AI Agents", "AI/ML" vs "AI ML"):

```
normalize(text) = lowercase(strip(text))
                ; replace [\s_/|,;:.()\[\]{}]+ with single space
                ; collapse repeated spaces; strip
```

**Implementation (regex):**
```python
t = str(text).strip().lower()
t = re.sub(r"[\s_/|,;:.()\[\]{}]+", " ", t)
t = re.sub(r"\s+", " ", t).strip()
```

### Examples

| Raw Input | Normalized Key |
|-----------|----------------|
| "AI agents" | "ai agents" |
| "AI Agents" | "ai agents" |
| "AI/ML" | "ai ml" |
| "AI ML" | "ai ml" |
| "Python Programming" | "python programming" |
| "  Data Analysis  " | "data analysis" |

### Grouping Logic

| Item Type | Group Key | Canonical Form | Aggregation |
|-----------|-----------|----------------|-------------|
| Skills | normalize(skill) | Most frequent raw | demand_freq = sum; type/bloom = mode |
| Knowledge | normalize(knowledge) | Most frequent raw | freq = sum; mean_confidence = mean |
| Competencies | normalize(title) | First in group | occurrence_count = size of group |

### Worked Example

**Input (skills from multiple jobs):**
- "Python", "python", "Python " → all map to key "python"
- Raw counts: Python=45, python=30, "Python "=5
- **Canonical form:** "Python" (most frequent)
- **Aggregated demand_freq:** 45 + 30 + 5 = 80

---

## 7. Priority Score and Future Weight

### Priority Score (recommendations.py)

```
priority_score = 0.40 × demand_norm + 0.30 × trend_norm + 0.30 × future_norm
```

**Signal definitions:**

1. **demand_norm** = demand_freq / max(demand_freq) ∈ [0, 1]
2. **trend_norm** ∈ [-1, 1]:
   - Emerging: min(1.0, |slope| / 0.5)
   - Declining: max(-1.0, -|slope| / 0.5)
   - Stable: 0
3. **future_norm** = future_weight / max(|future_weight|) ∈ [0, 1]

**Coverage** is not used by default (curriculum is for insights only).

### Worked Example

Skill "Machine Learning":
- demand_freq = 80, max_demand = 100 → demand_norm = 0.80
- slope = 0.15 (emerging), |slope|/0.5 = 0.30 → trend_norm = 0.30
- future_weight = 0.7, max_fw = 0.9 → future_norm = 0.78

```
priority_score = 0.40×0.80 + 0.30×0.30 + 0.30×0.78
               = 0.32 + 0.09 + 0.234 = 0.644
```

### Future Weight (future_weight_mapping.py)

```
future_weight = similarity(item, best_domain) × trend_score
```

- **similarity:** cosine similarity between SBERT embedding of item and domain (name + example terms)
- **trend_score:** from domain metadata (e.g., Strong_Growth ≈ 1.0, Decline ≈ -0.5)

**Mapping margin (uncertainty):**
```
mapping_margin = top1_similarity - top2_similarity
```

- Large margin → confident mapping
- Small margin → ambiguous (top 2 domains very close)

**Example:**
- Skill "Deep Learning" vs domains: AI & Machine Learning (sim=0.85), Cloud (sim=0.72)
- best_domain = AI & Machine Learning, trend_score = 1.0
- future_weight = 0.85 × 1.0 = **0.85**
- mapping_margin = 0.85 - 0.72 = **0.13**

---

## 8. FDR-Controlled Trend Detection

### Purpose

Identify skills that are significantly emerging or declining over time while controlling the false discovery rate across many tests.

### Method

1. For each skill: linear regression of monthly frequency vs time
2. Obtain p-value for slope (H₀: slope = 0)
3. Apply **Benjamini-Hochberg** to get q-values
4. Label: Emerging if slope > 0 and q < 0.05; Declining if slope < 0 and q < 0.05; else Stable

### Benjamini-Hochberg (Skill Trends)

```
Sort p-values: p_(1) ≤ p_(2) ≤ … ≤ p_(m)
For i = m-1 down to 1:
  rank = i
  q_(i) = min(p_(i) × m / rank, q_(i+1))
```

### Worked Example

5 skills, p-values = [0.001, 0.02, 0.03, 0.15, 0.40], slopes = [+0.2, -0.15, +0.1, +0.05, -0.02]

| Rank | p | q = p×5/rank |
|------|---|--------------|
| 1 | 0.001 | 0.005 |
| 2 | 0.02 | 0.05 |
| 3 | 0.03 | 0.05 |
| 4 | 0.15 | 0.1875 |
| 5 | 0.40 | 0.40 |

Skills 1, 2, 3 have q ≤ 0.05:
- Skill 1: slope > 0 → **Emerging**
- Skill 2: slope < 0 → **Declining**
- Skill 3: slope > 0 → **Emerging**
- Skills 4, 5: **Stable**

---

## 9. Weight Sensitivity Analysis

### Purpose

Assess how robust the top-20 recommendations are to changes in the priority score weights.

### Method

1. **Baseline:** (w_demand=0.4, w_trend=0.3, w_future=0.3)
2. **Sweep** alternative weight configurations
3. For each config: compute top-20, then **Jaccard similarity** vs baseline

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

### Configurations Tested

| Config | w_demand | w_trend | w_future |
|--------|----------|---------|----------|
| d0.3_t0.35_f0.35 | 0.30 | 0.35 | 0.35 |
| d0.4_t0.3_f0.3 | 0.40 | 0.30 | 0.30 (baseline) |
| d0.5_t0.25_f0.25 | 0.50 | 0.25 | 0.25 |
| d0.3_t0.2_f0.5 | 0.30 | 0.20 | 0.50 |
| d0.4_t0.2_f0.4 | 0.40 | 0.20 | 0.40 |
| d0.5_t0.2_f0.3 | 0.50 | 0.20 | 0.30 |

### Worked Example

- Baseline top-20: {A, B, C, …, T}
- Config d0.5_t0.25_f0.25 top-20: {A, B, D, E, …, U}
- Intersection: 16 skills
- Union: 24 skills
- **Jaccard = 16/24 ≈ 0.67**

High Jaccard → robust to weight changes.

### Running

```bash
python recommendations.py --sensitivity
```

Output: `weight_sensitivity_report.json`

---

## 10. Recall in the Gold-Set Paradigm

### Limitation

In gold-set evaluation:
- **Precision** = tp / n (correct among labeled)
- **Recall** requires knowing the total number of true positives in the **full population**
- Gold set = stratified **sample** of extractions, not an enumeration of all true items

### Consequence

We cannot compute true recall. We report **recall_estimate** = precision when the gold set is a sample of extractions (not a sample of all job postings). Interpret as "recall on the labeled sample" rather than population recall.

### When Recall Is Meaningful

- If gold set were a random sample of **all jobs** with exhaustive enumeration of true skills per job, then recall = tp / (tp + fn) would be estimable.
- Our design: gold set = sample of **extractions** to label. So we measure extraction correctness (precision-like) per item, not coverage of a known universe.

---

## 11. Calibrated Verification Metrics

### AUC-ROC

Area under the Receiver Operating Characteristic curve. Measures discrimination: ability to rank correct extractions higher than incorrect.

- AUC = 0.5: random
- AUC = 1.0: perfect

### Brier Score

Mean squared error of predicted probability vs observed outcome:

```
Brier = (1/n) × Σ (predicted_i - observed_i)²
```

- 0 = perfect calibration
- 0.25 = random (for binary)

### Calibration Curve

10 bins of predicted probability. For each bin: mean predicted prob vs observed frequency. Ideal: points on diagonal.

### Youden Index

Maximize (sensitivity + specificity - 1) to choose optimal threshold.

---

## 12. Future-Domain Mapping Evaluation

### Top-1 Accuracy

For each item with expert-labeled `true_domain_id`, compare to pipeline `best_domain_id`:

```
top1_accuracy = count(true_domain_id == best_domain_id) / n_evaluable
```

Items with `true_domain_id` in {none, unclear} are excluded from evaluation.

### Mapping Margin

`mapping_margin = top1_similarity - top2_similarity` indicates mapping confidence. Higher margin → more confident assignment.

---

## 13. Domain-Based Batching for Competency Generation

### Purpose

To group skills by thematic domain before sending them to the LLM, so each competency-generation batch contains related skills (e.g., AI/ML together, cloud together). This reduces forced groupings of unrelated skills and improves scientific validity.

### Confidence Thresholds

Skills with domain assignments from `future_skill_weights.csv` are classified by confidence:

- **High confidence:** similarity ≥ 0.45 and mapping_margin ≥ 0.05 → use `best_future_domain`
- **Low confidence:** below either threshold → assign to "Uncertain" batch

Low-confidence skills are not forced into a specific domain to avoid false thematic grouping.

### Normalized-Key Lookup

To reduce coverage gap from case/punctuation variants (e.g., "AI/ML" vs "AI ML"), `load_skill_future_weights` builds a dual index:

1. Exact key: `str(skill).strip()`
2. Normalized key: `normalize_for_grouping(skill)` (lowercase, collapse punctuation)

Lookup order: exact first, then normalized.

### On-the-Fly Fallback

Skills not in `future_skill_weights.csv` get domain assignment via embedding similarity to `future_domains.csv` (when available). If no future_domains or embedding fails, they go to "Unmapped".

### Batch Merge (Strongly Similar Domains)

When two domain batches have domains with cosine similarity ≥ 0.7 (of their embeddings: domain_text + example_terms), they are merged. This avoids tiny batches (e.g., 5 skills) while preserving thematic coherence.

---

## 14. Recommendation Evaluation (P@N, NDCG)

Used when experts label top-N curriculum recommendations in `recommendations.csv` (expert_priority column).

### Precision@N

```
P@N = (count(yes) + 0.5 × count(partial)) / N
```

Expert labels: yes / partial / no for top-N recommendations.

**Example:** N=20, 12 yes, 4 partial, 4 no  
P@20 = (12 + 2) / 20 = **0.70**

### NDCG@N

```
rel_i = 1 (yes), 0.5 (partial), 0 (no)
DCG = Σ (rel_i / log₂(i+2))
IDCG = DCG of ideal ranking (all yes first)
NDCG = DCG / IDCG
```

**Example (N=5):** rel = [1, 0.5, 1, 0, 1]  
DCG = 1/log₂(2) + 0.5/log₂(3) + 1/log₂(4) + 0 + 1/log₂(6) ≈ 1 + 0.315 + 0.5 + 0 + 0.387 ≈ 2.20  
IDCG (ideal [1,1,1,0.5,0]) ≈ 2.89  
NDCG ≈ 2.20/2.89 ≈ **0.76**

---

## Summary

| Component | Key Formula / Method |
|-----------|----------------------|
| Binomial test | H₀: p=0.5, one-sided, scipy.binomtest |
| Odds ratio | p/(1-p) when p₀=0.5 |
| Multi-comparison | Bonferroni (per-source), BH (pairwise) |
| Power analysis | n≥37 for p₁=0.7, α=0.05, power=0.80 |
| Cohen's Kappa | (P_o - P_e)/(1 - P_e), 2 raters |
| Fleiss' Kappa | (P̄ - P_e)/(1 - P_e), 3+ raters |
| Normalization | lowercase, collapse punctuation/spaces |
| Priority score | 0.4×demand + 0.3×trend + 0.3×future |
| Future weight | similarity × trend_score |
| FDR trends | Benjamini-Hochberg, q<0.05 |
| Weight sensitivity | Jaccard vs baseline top-20 |
| Recall | Not computable; report recall_estimate |
| P@N | (yes + 0.5×partial)/N |
| NDCG | DCG/IDCG with rel ∈ {0, 0.5, 1} |
| Top-1 domain | count(agree)/n_evaluable |
| Domain batching | similarity≥0.45, margin≥0.05 for high conf; merge when cos_sim≥0.7 |

---

---

## Scientific Visualization (plot_scientific_analysis.py)

The pipeline generates **scientific plots** that visualize the quantitative analysis:

| Plot | Description |
|------|-------------|
| `scientific_extraction_precision_skills/knowledge.png` | Precision by source with Wilson 95% CI; chance line at 0.5 |
| `scientific_odds_ratio_skills/knowledge.png` | Odds ratio vs chance (1.0) by source |
| `scientific_binomial_test_skills/knowledge.png` | Binomial distribution under H₀ with observed count |
| `scientific_trend_volcano.png` | FDR volcano: slope vs -log₁₀(q); Emerging/Declining/Stable |
| `scientific_trend_regression_samples.png` | Sample regression lines for top emerging/declining skills |
| `scientific_calibration_curve.png` | Reliability diagram (predicted vs observed) |
| `scientific_power_curve.png` | Power vs n for one-sided binomial test |
| `scientific_future_mapping.png` | Top-1 accuracy and mapping margin distribution |
| `scientific_weight_sensitivity.png` | Jaccard vs baseline for weight configurations |

Run: `python plot_scientific_analysis.py` (invoked by run.bat step 17).

**Interpretations:** See [docs/FIGURE_INTERPRETATIONS.md](docs/FIGURE_INTERPRETATIONS.md) for detailed explanations of each figure.

---

*Last updated: Domain-based batching (2025), full scientific methodology with worked examples.*
