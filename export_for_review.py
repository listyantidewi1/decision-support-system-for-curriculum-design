"""
export_for_review.py

Prepares CSV files for human-in-the-loop expert review using existing
pipeline outputs. Uses stratified sampling to reduce expert workload.

Inputs (from config.OUTPUT_DIR):
    - comprehensive_analysis.csv
    - verified_skills.csv  (if present) OR advanced_skills.csv
    - advanced_knowledge.csv (optional)
    - future_skill_weights_dummy.csv (optional)

Outputs:
    - expert_review_jobs.csv
    - expert_review_skills.csv (with review columns: review_id, human_valid, human_bloom, human_notes)
    - expert_review_knowledge.csv (with review columns: review_id, human_valid, human_notes)
"""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

import config  # uses config.OUTPUT_DIR


def load_comprehensive_analysis(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "comprehensive_analysis.csv"
    if not path.exists():
        raise FileNotFoundError(f"comprehensive_analysis.csv not found at {path}")
    return pd.read_csv(path)


def load_best_skills_table(output_dir: Path) -> pd.DataFrame:
    """
    Prefer verified_skills.csv if it exists, otherwise advanced_skills.csv.
    """
    verified_path = output_dir / "verified_skills.csv"
    adv_path = output_dir / "advanced_skills.csv"

    if verified_path.exists():
        df = pd.read_csv(verified_path)
        df["source_file"] = "verified_skills.csv"
        return df
    elif adv_path.exists():
        df = pd.read_csv(adv_path)
        df["source_file"] = "advanced_skills.csv"
        return df
    else:
        raise FileNotFoundError(
            f"Neither verified_skills.csv nor advanced_skills.csv "
            f"found in {output_dir}"
        )


def build_job_review_table(comp: pd.DataFrame) -> pd.DataFrame:
    """
    Selects the most relevant columns for expert review at the job level.
    """
    cols = [
        "job_id",
        "raw_text",
        "final_skills",
        "final_skills_count",
        "final_knowledge",
        "final_knowledge_count",
        "final_hard_skills",
        "final_soft_skills",
        "final_bloom_distribution",
        "coverage_percentage",
        "skill_coverage_pct",
        "knowledge_coverage_pct",
        "missing_components_count",
        "model_agreement_score",
        "final_avg_confidence",
        "avg_semantic_density",
        "avg_context_agreement",
    ]

    existing = [c for c in cols if c in comp.columns]
    return comp[existing].copy()


def _stratified_sample_skills(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Proportional stratified sampling by verification_level, type, bloom."""
    strata_cols = [c for c in ["verification_level", "type", "bloom"] if c in df.columns]
    if not strata_cols or len(df) <= n:
        return df.sample(n=min(n, len(df)), random_state=42)
    try:
        total = len(df)
        parts = []
        for _, grp in df.groupby(strata_cols, dropna=False):
            alloc = max(1, round(n * len(grp) / total))
            parts.append(grp.sample(n=min(alloc, len(grp)), random_state=42))
        result = pd.concat(parts).reset_index(drop=True)
        if len(result) > n:
            result = result.sample(n=n, random_state=42)
        return result
    except Exception:
        return df.sample(n=min(n, len(df)), random_state=42)


def _make_review_id(job_id: str, skill: str) -> str:
    """Generate unique review_id for traceability."""
    raw = f"{job_id}|{skill}"
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"skill_{h}"


def build_skill_review_table(skills: pd.DataFrame, max_skills: int = 500) -> pd.DataFrame:
    """
    Normalise skill table for expert review with stratified sampling.
    Adds review columns: review_id, human_valid, human_bloom, human_notes.
    """
    cols = [
        "job_id",
        "skill",
        "type",
        "bloom",
        "confidence_score",
        "confidence_tier",
        "verification_level",
        "is_verified",
        "semantic_density",
        "context_agreement",
        "source",
        "source_file",
    ]

    existing = [c for c in cols if c in skills.columns]
    df = skills[existing].copy()

    # Add review columns
    df["review_id"] = df.apply(
        lambda r: _make_review_id(str(r.get("job_id", "")), str(r.get("skill", ""))),
        axis=1,
    )
    df["human_valid"] = ""
    df["human_type"] = ""
    df["human_bloom"] = ""
    df["human_notes"] = ""
    df["reviewer_id"] = ""

    # Sort for nicer reading
    sort_spec = [
        ("is_verified", False),
        ("confidence_score", False),
        ("job_id", True),
    ]
    sort_cols = [c for c, _ in sort_spec if c in df.columns]
    sort_asc = [a for c, a in sort_spec if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=sort_asc)

    # Prioritize hybrid (BERT+GPT) results, then fill from other sources
    if len(df) > max_skills and "source" in df.columns:
        hybrid = df[df["source"] == "BERT+GPT"]
        non_hybrid = df[df["source"] != "BERT+GPT"]
        if len(hybrid) >= max_skills:
            df = _stratified_sample_skills(hybrid, max_skills)
        else:
            remaining = max_skills - len(hybrid)
            sampled_rest = _stratified_sample_skills(non_hybrid, remaining)
            df = pd.concat([hybrid, sampled_rest]).reset_index(drop=True)
    elif len(df) > max_skills:
        df = _stratified_sample_skills(df, max_skills)

    return df


def _make_knowledge_review_id(knowledge: str) -> str:
    """Generate unique review_id for knowledge items."""
    h = hashlib.md5(knowledge.encode("utf-8")).hexdigest()[:12]
    return f"know_{h}"


def build_knowledge_review_table(knowledge_df: pd.DataFrame,
                                 future_weights: pd.DataFrame,
                                 max_knowledge: int = 200) -> pd.DataFrame:
    """
    Build a knowledge-level review table by joining advanced_knowledge
    with future_weight info. Adds review columns: review_id, human_valid, human_notes.
    """
    # Ensure 'knowledge' exists in both
    if "knowledge" not in knowledge_df.columns:
        raise ValueError("knowledge_df must contain 'knowledge' column.")
    if "knowledge" not in future_weights.columns:
        raise ValueError("future_weights must contain 'knowledge' column.")

    # Basic cleaning
    knowledge_df = knowledge_df.copy()
    knowledge_df["knowledge"] = knowledge_df["knowledge"].astype(str).str.strip()

    future_weights = future_weights.copy()
    future_weights["knowledge"] = future_weights["knowledge"].astype(str).str.strip()

    # Aggregate knowledge_df for nicer view (optional)
    agg = (
        knowledge_df.groupby("knowledge")
        .agg(
            freq=("job_id", "nunique") if "job_id" in knowledge_df.columns else ("knowledge", "count"),
            mean_confidence=("confidence_score", "mean")
            if "confidence_score" in knowledge_df.columns
            else ("knowledge", "count"),
        )
        .reset_index()
    )

    # Join with future weights
    merged = agg.merge(
        future_weights[
            [
                "knowledge",
                "best_future_domain",
                "trend_label",
                "trend_score",
                "similarity",
                "future_weight",
            ]
        ],
        on="knowledge",
        how="left",
    )

    # Sort: highest future_weight first (future critical)
    merged = merged.sort_values(
        ["future_weight", "freq"],
        ascending=[False, False],
    )

    # Add review columns
    merged["review_id"] = merged["knowledge"].apply(_make_knowledge_review_id)
    merged["human_valid"] = ""
    merged["human_notes"] = ""
    merged["reviewer_id"] = ""

    # Proportional stratified sampling by trend_label (if available)
    if len(merged) > max_knowledge:
        if "trend_label" in merged.columns:
            total = len(merged)
            parts = []
            for _, grp in merged.groupby("trend_label", dropna=False):
                alloc = max(1, round(max_knowledge * len(grp) / total))
                parts.append(grp.sample(n=min(alloc, len(grp)), random_state=42))
            merged = pd.concat(parts).reset_index(drop=True)
            if len(merged) > max_knowledge:
                merged = merged.sample(n=max_knowledge, random_state=42)
        else:
            merged = merged.sample(n=max_knowledge, random_state=42)

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Export job-level and skill-level tables for expert review."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(config.OUTPUT_DIR),
        help="Directory containing pipeline outputs (default: config.OUTPUT_DIR)",
    )
    parser.add_argument(
        "--jobs_csv",
        type=str,
        default="expert_review_jobs.csv",
        help="Output CSV for job-level review (default: expert_review_jobs.csv)",
    )
    parser.add_argument(
        "--skills_csv",
        type=str,
        default="expert_review_skills.csv",
        help="Output CSV for skill-level review (default: expert_review_skills.csv)",
    )
    parser.add_argument(
        "--max_skills",
        type=int,
        default=500,
        help="Max skills to export for review (stratified sampling, default: 500)",
    )
    parser.add_argument(
        "--max_knowledge",
        type=int,
        default=200,
        help="Max knowledge items to export for review (default: 200)",
    )

    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    print(f"[INFO] Loading comprehensive_analysis.csv from {out_dir}")
    comp = load_comprehensive_analysis(out_dir)

    print("[INFO] Building job-level review table...")
    jobs_df = build_job_review_table(comp)
    jobs_path = out_dir / args.jobs_csv
    jobs_df.to_csv(jobs_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved job-level review table to {jobs_path}")

    print("[INFO] Loading best available skills table (verified or advanced)...")
    skills_raw = load_best_skills_table(out_dir)

    print(f"[INFO] Building skill-level review table (max_skills={args.max_skills})...")
    skills_df = build_skill_review_table(skills_raw, max_skills=args.max_skills)
    skills_path = out_dir / args.skills_csv
    skills_df.to_csv(skills_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved skill-level review table to {skills_path}")
    
        # --- Knowledge-level review with future weights ---
    try:
        knowledge_path1 = out_dir / "advanced_knowledge_with_dates.csv"
        knowledge_path2 = out_dir / "advanced_knowledge.csv"

        if knowledge_path1.exists():
            knowledge_df = pd.read_csv(knowledge_path1)
            print(f"[INFO] Using {knowledge_path1} for knowledge-level review.")
        elif knowledge_path2.exists():
            knowledge_df = pd.read_csv(knowledge_path2)
            print(f"[INFO] Using {knowledge_path2} for knowledge-level review.")
        else:
            print("[WARN] No advanced_knowledge CSV found; "
                  "skipping expert_review_knowledge export.")
            knowledge_df = None

        fw_path = out_dir / "future_skill_weights_dummy.csv"
        if knowledge_df is not None and fw_path.exists():
            future_weights = pd.read_csv(fw_path)
            print(f"[INFO] Building knowledge-level review table (max_knowledge={args.max_knowledge})...")
            knowledge_review_df = build_knowledge_review_table(
                knowledge_df,
                future_weights,
                max_knowledge=args.max_knowledge,
            )
            knowledge_out_path = out_dir / "expert_review_knowledge.csv"
            knowledge_review_df.to_csv(
                knowledge_out_path, index=False, encoding="utf-8-sig"
            )
            print(f"[INFO] Saved knowledge-level review table to {knowledge_out_path}")
        elif knowledge_df is not None:
            print("[WARN] future_skill_weights_dummy.csv not found; "
                  "cannot build expert_review_knowledge.csv.")
    except Exception as e:
        print(f"[WARN] Failed to build knowledge-level review table: {e}")



if __name__ == "__main__":
    main()
