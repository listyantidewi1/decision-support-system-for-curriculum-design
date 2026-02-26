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


def build_competency_review_table(competencies: list, max_competencies: int = 100) -> pd.DataFrame:
    """
    Build competency review table with stratified sampling by batch_id.
    Adds review columns: human_quality, human_relevant, human_notes.
    """
    rows = []
    for idx, c in enumerate(competencies):
        comp_id = c.get("id", f"C{idx+1}")
        batch_id = c.get("batch_id", 0)
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

    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    print(f"[INFO] Loading competencies from {out_dir / args.input}")
    competencies = load_competencies(out_dir, args.input)
    print(f"[INFO] Total competencies: {len(competencies)}")

    print(f"[INFO] Building competency review table (max={args.max_competencies})...")
    df = build_competency_review_table(competencies, max_competencies=args.max_competencies)

    output_path = out_dir / args.output
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved to {output_path} ({len(df)} competencies)")


if __name__ == "__main__":
    main()
