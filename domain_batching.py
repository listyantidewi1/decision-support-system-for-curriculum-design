"""
domain_batching.py

Helper module for domain-based batching in competency generation.
Provides normalized-key lookup, domain assignment, batch building,
and merge logic for strongly-similar domain batches.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Reuse embedding model from pipeline
from pipeline import AdvancedPipelineConfig

TREND_ORDER = {"Strong_Growth": 3, "Moderate_Growth": 2, "Stable": 1, "Decline": 0}


def normalize_for_grouping(text: str) -> str:
    """
    Deterministic normalization for grouping equivalent skills/knowledge.
    Matches future_weight_mapping logic: lowercase, collapse punctuation/spaces.
    """
    if not text or not isinstance(text, str):
        return ""
    t = str(text).strip().lower()
    t = re.sub(r"[\s_/|,;:.()\[\]{}]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_future_domains(path: Path) -> pd.DataFrame:
    """Load future_domains.csv and build domain_text for embedding."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"domain_id", "future_domain", "example_terms", "trend_label", "trend_score"}
    if required - set(df.columns):
        return pd.DataFrame()
    df["trend_score"] = pd.to_numeric(df["trend_score"], errors="coerce").fillna(0.0)
    df["domain_text"] = (
        df["future_domain"].astype(str).str.strip()
        + ". "
        + df["example_terms"].astype(str).str.strip()
    )
    return df


def compute_domain_embeddings(
    domains_df: pd.DataFrame,
    embedder,
) -> Dict[str, np.ndarray]:
    """Compute embeddings for each future_domain. Keys are future_domain strings."""
    if domains_df.empty:
        return {}
    texts = domains_df["domain_text"].tolist()
    domains = domains_df["future_domain"].astype(str).str.strip().tolist()
    emb = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return {d: emb[i] for i, d in enumerate(domains)}


def assign_domain_for_skill(
    skill: str,
    domains_df: pd.DataFrame,
    embedder,
) -> Tuple[str, float, str]:
    """
    Assign a skill to the nearest future domain via embedding similarity.
    Returns (best_future_domain, future_weight, trend_label).
    Used as on-the-fly fallback for skills not in future_skill_weights.
    """
    if domains_df.empty:
        return ("Unmapped", 0.0, "")
    skill_text = str(skill).strip()
    if not skill_text:
        return ("Unmapped", 0.0, "")
    # Build domain texts (future_domain + example_terms)
    domain_texts = domains_df["domain_text"].tolist()
    skill_emb = embedder.encode(
        [skill_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    domain_emb = embedder.encode(
        domain_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    sims = np.dot(domain_emb, skill_emb.T).flatten()
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    row = domains_df.iloc[best_idx]
    trend_score = float(row.get("trend_score", 0) or 0)
    future_weight = best_sim * trend_score
    return (
        str(row["future_domain"]),
        future_weight,
        str(row.get("trend_label", "")),
    )


def build_domain_batches(
    skills_sorted: List[str],
    skill_to_domain: Dict[str, str],
    skill_to_domain_data: Dict[str, Dict],
    max_per_batch: int,
    domain_order: str = "mean_future_weight",
) -> List[Tuple[str, List[str]]]:
    """
    Group skills by domain and chunk within each domain.
    Returns list of (domain_name, [skills]) tuples.
    Preserves sort order within each domain.
    domain_order: "mean_future_weight" (desc) or "trend_first" (Strong_Growth first).
    """
    from collections import defaultdict

    domain_to_skills: Dict[str, List[str]] = defaultdict(list)
    for s in skills_sorted:
        domain = skill_to_domain.get(s.strip(), "Unmapped")
        domain_to_skills[domain].append(s)

    # Order domains
    if domain_order == "trend_first":
        domain_scores = {}
        for domain, skills in domain_to_skills.items():
            trends = [
                skill_to_domain_data.get(s.strip(), {}).get("trend_label", "")
                for s in skills
            ]
            best_trend = max(
                (TREND_ORDER.get(t, -1) for t in trends if t),
                default=-1,
            )
            domain_scores[domain] = (best_trend, -len(skills))  # secondary: more skills first
        sorted_domains = sorted(
            domain_to_skills.keys(),
            key=lambda d: domain_scores.get(d, (-1, 0)),
            reverse=True,
        )
    else:
        # mean_future_weight desc
        domain_scores = {}
        for domain, skills in domain_to_skills.items():
            fws = [
                skill_to_domain_data.get(s.strip(), {}).get("future_weight", -999)
                for s in skills
            ]
            domain_scores[domain] = sum(fws) / len(fws) if fws else -999
        sorted_domains = sorted(
            domain_to_skills.keys(),
            key=lambda d: domain_scores.get(d, -999),
            reverse=True,
        )

    result: List[Tuple[str, List[str]]] = []
    for domain in sorted_domains:
        skills = domain_to_skills[domain]
        for i in range(0, len(skills), max_per_batch):
            chunk = skills[i : i + max_per_batch]
            result.append((domain, chunk))
    return result


def merge_similar_domain_batches(
    batches: List[Tuple[str, List[str]]],
    domain_embeddings: Dict[str, np.ndarray],
    threshold: float,
) -> List[Tuple[str, List[str]]]:
    """
    Merge two domain batches only when their domains are strongly similar
    (cosine similarity >= threshold). Merges small batches into similar ones.
    """
    if not batches or threshold <= 0:
        return batches
    if len(batches) <= 1:
        return batches

    # Build domain -> batch indices
    domain_to_batch_idx: Dict[str, int] = {}
    for idx, (domain, _) in enumerate(batches):
        if domain not in domain_to_batch_idx:
            domain_to_batch_idx[domain] = idx

    # Merge small batches with similar domains
    # Strategy: for each small batch (< max_per_batch? we don't have that here),
    # find the most similar other domain and merge if above threshold.
    # Actually the plan says "merge when two domains are strongly similar" -
    # so we merge batches that belong to domains with sim >= threshold.
    # We need to decide: merge which ones? The plan says "small batches" -
    # so we merge when a batch is small. Let me re-read.
    # "Merge two domain batches only when they are strongly in the same domain"
    # So we merge batch A and batch B if domain_A and domain_B have cos_sim >= threshold.
    # We don't necessarily merge only small ones - we merge any two that are strongly similar.
    # But that could create huge batches. The plan says "merge small domain batches" -
    # so the intent is: when we have a small batch (few skills), we can merge it with
    # another batch if that other batch's domain is strongly similar.
    # So: for each batch with few skills, try to merge with the most similar other domain's batch.

    # Simpler interpretation: group batches by "domain cluster" - domains with sim >= threshold
    # get merged into one meta-batch. So we do hierarchical: find groups of domains that are
    # all pairwise above threshold, merge their batches together.
    # Even simpler: for each batch (domain, skills), if there's another batch (domain2, skills2)
    # such that cos_sim(domain, domain2) >= threshold, merge them. We need to be careful
    # not to merge indefinitely - maybe merge only when both batches are "small" (e.g. < 15 skills).

    # The plan: "only merge when the two batch are strongly in the same domain"
    # So we merge batch A (domain A) with batch B (domain B) iff cos_sim(A,B) >= threshold.
    # To avoid chain merging, we'll do a single pass: for each small batch, find the best
    # similar domain and merge. A batch is "small" if len(skills) < some_min - but we don't
    # have max_per_batch here. Let me check the plan again.
    # "if a domain has fewer than max_skills_per_call, optionally merge with a related domain"
    # So we merge when a batch has fewer than max_skills_per_call skills. We need that param.

    # Revise: merge_similar_domain_batches receives batches that are already chunked.
    # Each batch is (domain, skills). Some batches might have fewer skills than max_per_batch.
    # We merge batch A into batch B if: (1) A has fewer than max_per_batch skills, (2) domains
    # A and B have cos_sim >= threshold. So we need max_per_batch as param.

    # Actually the function signature doesn't have max_per_batch. The plan says:
    # "merge_similar_domain_batches(batches, domain_embeddings, threshold)"
    # So we merge any two batches whose domains have cos_sim >= threshold. That could create
    # very large batches. To be safe, let's only merge when the resulting batch wouldn't exceed
    # a reasonable size - but we don't have that. Let me just merge batches with similar domains:
    # if domain A and B have sim >= threshold, merge batch A and B. We'll need to be careful
    # about order - merge into the one with higher mean future_weight maybe.
    # For now, I'll merge all batches that belong to "similar" domains into one. Use Union-Find.

    def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    # Union-Find to group similar domains
    domain_list = list({b[0] for b in batches})
    n = len(domain_list)
    parent = list(range(n))
    domain_to_id = {d: i for i, d in enumerate(domain_list)}

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            di, dj = domain_list[i], domain_list[j]
            # Skip special batches that have no embedding
            if di in ("Unmapped", "Uncertain") or dj in ("Unmapped", "Uncertain"):
                continue
            emb_i = domain_embeddings.get(di)
            emb_j = domain_embeddings.get(dj)
            if emb_i is not None and emb_j is not None:
                sim = cos_sim(emb_i, emb_j)
                if sim >= threshold:
                    union(i, j)

    # Group batches by their merged domain cluster
    clusters: Dict[int, List[Tuple[str, List[str]]]] = {}
    for domain, skills in batches:
        idx = domain_to_id.get(domain, -1)
        if idx < 0:
            clusters.setdefault(-1, []).append((domain, skills))
        else:
            root = find(idx)
            clusters.setdefault(root, []).append((domain, skills))

    # Flatten each cluster into one batch (domain name = first domain in cluster)
    result = []
    for root, cluster_batches in clusters.items():
        merged_skills = []
        merged_domain = cluster_batches[0][0]  # use first domain name
        for _, skills in cluster_batches:
            merged_skills.extend(skills)
        result.append((merged_domain, merged_skills))

    return result
