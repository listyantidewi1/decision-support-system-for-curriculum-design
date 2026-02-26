"""
ingest_future_domains.py

Normalizes future-of-work sources (WEF, O*NET, ESCO, McKinsey, national
strategy docs) into a single future_domains.csv compatible with
future_weight_mapping.py.

Each source adapter reads its specific format and outputs rows in the
standard schema:
    domain_id, future_domain, example_terms, trend_label, trend_score,
    source, horizon_year

Usage:
    python ingest_future_domains.py --sources wef onet custom --output future_domains.csv

The script can also merge a custom CSV that already follows the schema.
"""

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

import config

REQUIRED_COLS = {
    "domain_id", "future_domain", "example_terms",
    "trend_label", "trend_score",
}
OPTIONAL_COLS = {"source", "horizon_year"}

TREND_LABELS = {"Strong_Growth", "Moderate_Growth", "Stable", "Decline"}


# ---------------------------------------------------------------------------
# Built-in domain definitions derived from published reports
# ---------------------------------------------------------------------------

WEF_DOMAINS = [
    {"domain_id": "WEF01", "future_domain": "AI & Machine Learning",
     "example_terms": "machine learning, deep learning, generative AI, large language models, neural networks, MLOps, computer vision, NLP",
     "trend_label": "Strong_Growth", "trend_score": 1.0, "source": "WEF_FutureOfJobs_2023", "horizon_year": 2030},
    {"domain_id": "WEF02", "future_domain": "Big Data & Analytics",
     "example_terms": "data analytics, data engineering, business intelligence, data visualization, statistical modeling, data pipelines, data warehousing",
     "trend_label": "Strong_Growth", "trend_score": 0.9, "source": "WEF_FutureOfJobs_2023", "horizon_year": 2030},
    {"domain_id": "WEF03", "future_domain": "Cybersecurity",
     "example_terms": "cybersecurity, information security, penetration testing, threat intelligence, zero trust, SOC, incident response, encryption",
     "trend_label": "Strong_Growth", "trend_score": 0.9, "source": "WEF_FutureOfJobs_2023", "horizon_year": 2030},
    {"domain_id": "WEF04", "future_domain": "Cloud Computing & Infrastructure",
     "example_terms": "cloud computing, kubernetes, docker, AWS, Azure, GCP, terraform, infrastructure as code, CI/CD, DevOps",
     "trend_label": "Strong_Growth", "trend_score": 0.85, "source": "WEF_FutureOfJobs_2023", "horizon_year": 2030},
    {"domain_id": "WEF05", "future_domain": "Digital Platforms & E-commerce",
     "example_terms": "e-commerce, digital marketing, platform engineering, marketplace, fintech, digital payments, SaaS",
     "trend_label": "Moderate_Growth", "trend_score": 0.7, "source": "WEF_FutureOfJobs_2023", "horizon_year": 2030},
    {"domain_id": "WEF06", "future_domain": "Environmental & Sustainability Technology",
     "example_terms": "green technology, sustainable computing, energy efficiency, smart grids, environmental monitoring, carbon footprint",
     "trend_label": "Moderate_Growth", "trend_score": 0.6, "source": "WEF_FutureOfJobs_2023", "horizon_year": 2035},
    {"domain_id": "WEF07", "future_domain": "Human-AI Collaboration",
     "example_terms": "prompt engineering, AI-assisted coding, copilot, human-in-the-loop, explainable AI, AI ethics, responsible AI",
     "trend_label": "Strong_Growth", "trend_score": 0.95, "source": "WEF_FutureOfJobs_2023", "horizon_year": 2030},
]

ONET_DOMAINS = [
    {"domain_id": "ONET01", "future_domain": "Software Development & Engineering",
     "example_terms": "software engineering, full-stack development, API design, microservices, agile, scrum, version control, testing, debugging",
     "trend_label": "Moderate_Growth", "trend_score": 0.65, "source": "ONET_BrightOutlook_2024", "horizon_year": 2033},
    {"domain_id": "ONET02", "future_domain": "Database & Systems Administration",
     "example_terms": "database administration, SQL, PostgreSQL, MongoDB, system administration, Linux, networking, backup, monitoring",
     "trend_label": "Moderate_Growth", "trend_score": 0.5, "source": "ONET_BrightOutlook_2024", "horizon_year": 2033},
    {"domain_id": "ONET03", "future_domain": "UX/UI & Digital Design",
     "example_terms": "user experience, user interface, Figma, wireframing, prototyping, accessibility, responsive design, user research",
     "trend_label": "Moderate_Growth", "trend_score": 0.55, "source": "ONET_BrightOutlook_2024", "horizon_year": 2033},
    {"domain_id": "ONET04", "future_domain": "Game Development & Interactive Media",
     "example_terms": "game development, Unity, Unreal Engine, 3D modeling, game design, VR, AR, interactive media, shader programming",
     "trend_label": "Moderate_Growth", "trend_score": 0.6, "source": "ONET_BrightOutlook_2024", "horizon_year": 2033},
]

MCKINSEY_DOMAINS = [
    {"domain_id": "MCK01", "future_domain": "Process Automation & RPA",
     "example_terms": "robotic process automation, workflow automation, scripting, low-code, no-code, business process management",
     "trend_label": "Moderate_Growth", "trend_score": 0.5, "source": "McKinsey_FutureOfWork_2023", "horizon_year": 2030},
    {"domain_id": "MCK02", "future_domain": "Advanced Communication & Collaboration",
     "example_terms": "technical writing, cross-functional collaboration, remote work tools, stakeholder management, presentation skills",
     "trend_label": "Moderate_Growth", "trend_score": 0.4, "source": "McKinsey_FutureOfWork_2023", "horizon_year": 2030},
]

DECLINE_DOMAINS = [
    {"domain_id": "DEC01", "future_domain": "Routine Data Entry & Clerical",
     "example_terms": "data entry, routine clerical work, simple repetitive administration, manual bookkeeping",
     "trend_label": "Decline", "trend_score": -0.6, "source": "WEF_FutureOfJobs_2023", "horizon_year": 2030},
    {"domain_id": "DEC02", "future_domain": "Legacy Web Development",
     "example_terms": "static websites, basic HTML only, table-based layouts, Flash development",
     "trend_label": "Decline", "trend_score": -0.4, "source": "Expert_consensus", "horizon_year": 2028},
    {"domain_id": "DEC03", "future_domain": "Routine Maintenance Coding",
     "example_terms": "boilerplate code updates, simple feature maintenance, repetitive bug fixing without design thinking",
     "trend_label": "Decline", "trend_score": -0.3, "source": "McKinsey_FutureOfWork_2023", "horizon_year": 2030},
]

BUILTIN_SOURCES = {
    "wef": WEF_DOMAINS,
    "onet": ONET_DOMAINS,
    "mckinsey": MCKINSEY_DOMAINS,
    "decline": DECLINE_DOMAINS,
}


def load_custom_csv(path: Path) -> List[dict]:
    """Load a user-provided CSV that already follows the schema."""
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Custom CSV missing required columns: {missing}")
    return df.to_dict(orient="records")


def normalize_trend_label(label: str) -> str:
    label = str(label).strip()
    mapping = {
        "strong_growth": "Strong_Growth",
        "moderate_growth": "Moderate_Growth",
        "stable": "Stable",
        "decline": "Decline",
        "declining": "Decline",
    }
    return mapping.get(label.lower(), label)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest and normalize future-of-work domain sources."
    )
    parser.add_argument(
        "--sources", nargs="+", default=["wef", "onet", "mckinsey", "decline"],
        help="Built-in sources to include: wef, onet, mckinsey, decline (default: all)",
    )
    parser.add_argument(
        "--custom", type=str, default=None,
        help="Path to a custom future_domains CSV to merge in",
    )
    parser.add_argument(
        "--output", type=str, default="future_domains.csv",
        help="Output CSV file name (default: future_domains.csv in project root)",
    )

    args = parser.parse_args()

    all_rows: List[dict] = []

    for src in args.sources:
        src_lower = src.lower()
        if src_lower in BUILTIN_SOURCES:
            rows = BUILTIN_SOURCES[src_lower]
            all_rows.extend(rows)
            print(f"[INFO] Added {len(rows)} domains from built-in source: {src}")
        else:
            print(f"[WARN] Unknown built-in source '{src}'; skipping. "
                  f"Available: {list(BUILTIN_SOURCES.keys())}")

    if args.custom:
        custom_path = Path(args.custom)
        if custom_path.exists():
            custom_rows = load_custom_csv(custom_path)
            all_rows.extend(custom_rows)
            print(f"[INFO] Added {len(custom_rows)} domains from custom CSV: {custom_path}")
        else:
            print(f"[WARN] Custom CSV not found: {custom_path}")

    if not all_rows:
        print("[ERROR] No domains loaded. Check --sources or --custom.")
        return

    df = pd.DataFrame(all_rows)

    for col in OPTIONAL_COLS:
        if col not in df.columns:
            df[col] = ""

    df["trend_label"] = df["trend_label"].apply(normalize_trend_label)
    df["trend_score"] = pd.to_numeric(df["trend_score"], errors="coerce").fillna(0.0)

    out_path = Path(config.PROJECT_ROOT) / args.output
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved {len(df)} domains to {out_path}")
    print(f"[INFO] Sources: {df['source'].unique().tolist()}")


if __name__ == "__main__":
    main()
