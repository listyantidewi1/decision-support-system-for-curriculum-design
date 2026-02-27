"""
export_competencies_for_review.py

Exports a sampled subset of competency_proposals.json for human-in-the-loop review.
Uses stratified sampling by batch_id to ensure diversity.

Outputs:
    - expert_review_competencies.csv (with human_quality, human_relevant, human_notes columns)
"""

import argparse
import json
from pathlib import Path

import pandas as pd

import config


def load_competencies(output_dir: Path, input_file: str = "competency_proposals.json") -> list:
    """Load competencies from JSON."""
    path = output_dir / input_file
    if not path.exists():
        raise FileNotFoundError(f"Competency file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("competencies", [])


def _load_exclude_competency_ids(exclude_path: Path) -> set:
    """Load competency_ids from existing review file to exclude."""
    if not exclude_path or not exclude_path.exists():
        return set()
    df = pd.read_csv(exclude_path)
    if "competency_id" not in df.columns:
        return set()
    return set(str(r["competency_id"]).strip() for _, r in df.iterrows())


def build_competency_review_table(
    competencies: list,
    max_competencies: int = 100,
    exclude_ids: set | None = None,
) -> pd.DataFrame:
    """
    Build competency review table with stratified sampling by batch_id.
    Adds review columns: human_quality, human_relevant, human_notes.
    """
    exclude_ids = exclude_ids or set()
    rows = []
    for idx, c in enumerate(competencies):
        comp_id = c.get("id", f"C{idx+1}")
        batch_id = c.get("batch_id", 0)
        cid = f"{comp_id}_b{batch_id}" if batch_id else comp_id
        if cid in exclude_ids:
            continue
        rows.append({
            "competency_id": f"{comp_id}_b{batch_id}" if batch_id else comp_id,
            "batch_id": batch_id,
            "title": c.get("title", ""),
            "description": c.get("description", ""),
            "related_skills": json.dumps(c.get("related_skills", []), ensure_ascii=False) if isinstance(c.get("related_skills"), list) else str(c.get("related_skills", "")),
            "future_relevance": c.get("future_relevance", ""),
            "all_skills_human_verified": c.get("all_skills_human_verified", ""),
        })
    df = pd.DataFrame(rows)

    # Add review columns
    df["human_quality"] = ""  # 1-5 or poor/fair/good/excellent
    df["human_relevant"] = ""  # yes / no / partial
    df["human_notes"] = ""
    df["reviewer_id"] = ""

    # Stratified sampling by batch_id
    if len(df) > max_competencies:
        n_batches = df["batch_id"].nunique()
        per_batch = max(1, max_competencies // n_batches)
        sampled = (
            df.groupby("batch_id", group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), per_batch), random_state=42))
            .reset_index(drop=True)
        )
        if len(sampled) > max_competencies:
            sampled = sampled.sample(n=max_competencies, random_state=42)
        df = sampled

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Export competencies for expert review (sampled subset)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(config.OUTPUT_DIR),
        help="Directory containing competency_proposals.json (default: config.OUTPUT_DIR)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="competency_proposals.json",
        help="Input competency JSON file (default: competency_proposals.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="expert_review_competencies.csv",
        help="Output CSV for competency review (default: expert_review_competencies.csv)",
    )
    parser.add_argument(
        "--max_competencies",
        type=int,
        default=100,
        help="Max competencies to export for review (default: 100)",
    )
    parser.add_argument(
        "--exclude_from_competencies",
        type=str,
        default=None,
        help="Path to existing expert_review_competencies.csv - exclude items already in it",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new samples to existing file instead of overwriting",
    )

    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    print(f"[INFO] Loading competencies from {out_dir / args.input}")
    competencies = load_competencies(out_dir, args.input)
    print(f"[INFO] Total competencies: {len(competencies)}")

    exclude_path = Path(args.exclude_from_competencies) if args.exclude_from_competencies else None
    if args.append and not exclude_path:
        exclude_path = out_dir / args.output
    exclude_ids = _load_exclude_competency_ids(exclude_path) if exclude_path else set()
    if exclude_ids:
        print(f"[INFO] Excluding {len(exclude_ids)} already-reviewed competencies")

    print(f"[INFO] Building competency review table (max={args.max_competencies})...")
    df = build_competency_review_table(
        competencies,
        max_competencies=args.max_competencies,
        exclude_ids=exclude_ids,
    )

    output_path = out_dir / args.output
    if not df.empty:
        if args.append and output_path.exists():
            existing = pd.read_csv(output_path)
            df = pd.concat([existing, df]).drop_duplicates(subset=["competency_id"], keep="first")
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] Saved to {output_path} ({len(df)} competencies)")
    else:
        print("[INFO] No new competencies to add (all already reviewed or empty)")


if __name__ == "__main__":
    main()
