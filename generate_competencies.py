"""
generate_competencies_llm.py

Takes high/medium-verified skills from verified_skills.csv and asks an LLM
(through OpenRouter) to propose competency statements / curriculum components.

Output:
    competency_proposals.json  (list of JSON objects with competencies)
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from openai import OpenAI

import config  # uses config.OUTPUT_DIR

COMPETENCY_REQUIRED_FIELDS = {"id", "title", "description", "related_skills", "future_relevance"}
MAX_RETRIES = 2
RETRY_BACKOFF_BASE = 2  # seconds


def load_skill_time_trends(
    output_dir: Path,
    trends_file: str = "skill_time_trends.csv",
) -> Dict[str, Dict]:
    """
    Load skill -> {trend_label, slope, p_value} from skill_time_trends.csv
    (empirical emerging/declining from job posting frequency over time).
    Returns empty dict if file missing.
    """
    path = output_dir / trends_file
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "skill" not in df.columns or "trend_label" not in df.columns:
        return {}
    df = df.drop_duplicates(subset=["skill"], keep="first")
    return {
        str(row["skill"]).strip(): {
            "trend_label": str(row["trend_label"]),
            "slope": float(row.get("slope", 0) or 0),
            "p_value": float(row.get("p_value", 1) or 1),
        }
        for _, row in df.iterrows()
    }


def load_skill_future_weights(
    output_dir: Path,
    weights_file: str = "future_skill_weights.csv",
) -> Dict[str, Dict]:
    """
    Load skill -> {best_future_domain, trend_label, future_weight} from
    future_skill_weights.csv. Returns empty dict if file missing.
    """
    path = output_dir / weights_file
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    required = {"skill", "best_future_domain", "trend_label", "future_weight"}
    if required - set(df.columns):
        return {}
    # Keep first occurrence per skill (file is sorted by future_weight desc)
    df = df.drop_duplicates(subset=["skill"], keep="first")
    return {
        str(row["skill"]).strip(): {
            "best_future_domain": str(row["best_future_domain"]),
            "trend_label": str(row["trend_label"]),
            "future_weight": float(row["future_weight"]),
        }
        for _, row in df.iterrows()
    }


def build_future_context(output_dir: Path,
                         future_file: str = "future_skill_weights_dummy.csv",
                         top_k_domains: int = 5,
                         top_k_knowledge: int = 15) -> str:
    """
    Build a short text summary of future-critical domains and knowledge items
    from future_skill_weights_dummy.csv to guide competency generation.
    """
    fw_path = output_dir / future_file
    if not fw_path.exists():
        print(f"[WARN] Future weights file not found at {fw_path}. "
              f"Continuing without future context.")
        return ""

    df = pd.read_csv(fw_path)

    required_cols = {"knowledge", "best_future_domain",
                     "trend_label", "future_weight"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[WARN] Future weights file missing columns {missing}. "
              f"Continuing without future context.")
        return ""

    # Aggregate by domain
    domain_stats = (
        df.groupby("best_future_domain")["future_weight"]
        .mean()
        .reset_index()
        .sort_values("future_weight", ascending=False)
        .head(top_k_domains)
    )

    # Top knowledge items overall
    top_kw = (
        df.sort_values("future_weight", ascending=False)
        .head(top_k_knowledge)[["knowledge", "best_future_domain", "future_weight"]]
    )

    # Build human-readable context text
    lines = []
    lines.append("Future-critical domains (average future_weight):")
    for _, row in domain_stats.iterrows():
        lines.append(
            f"- {row['best_future_domain']} "
            f"(avg future_weight={row['future_weight']:.2f})"
        )

    lines.append("")
    lines.append("Example high future-weight knowledge items:")
    for _, row in top_kw.iterrows():
        lines.append(
            f"- {row['knowledge']} "
            f"(domain={row['best_future_domain']}, "
            f"future_weight={row['future_weight']:.2f})"
        )

    return "\n".join(lines)


# ----------------------------------------------------------------------
# Helper: load OpenRouter / OpenAI client the same way as in pipeline
# ----------------------------------------------------------------------

def load_openrouter_client() -> OpenAI:
    base_url = "https://openrouter.ai/api/v1"

    # 1) env var, if available
    api_key = os.getenv("OPENROUTER_API_KEY")

    # 2) fallback: api_keys/OpenRouter.txt
    if not api_key:
        key_path = Path("api_keys") / "OpenRouter.txt"
        try:
            with open(key_path, "r", encoding="utf-8") as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            raise RuntimeError(
                f"OpenRouter API key not found. Set OPENROUTER_API_KEY "
                f"or create {key_path}"
            )

    if not api_key:
        raise RuntimeError("OpenRouter API key is empty.")

    return OpenAI(api_key=api_key, base_url=base_url)


# ----------------------------------------------------------------------
# Few-shot examples from human assessments
# ----------------------------------------------------------------------

def load_few_shot_examples(
    output_dir: Path,
    feedback_dir: Path,
    proposals_file: str = "competency_proposals.json",
    assessments_file: str = "competency_assessments.json",
    min_quality: int = 4,
    max_examples: int = 3,
) -> List[Dict]:
    """
    Load high-quality competencies (human_quality >= min_quality, human_relevant=yes)
    as few-shot examples for prompt tuning.
    """
    proposals_path = output_dir / proposals_file
    assessments_path = feedback_dir / assessments_file
    if not proposals_path.exists() or not assessments_path.exists():
        return []

    proposals = json.loads(proposals_path.read_text(encoding="utf-8"))
    comps = proposals.get("competencies", [])
    assessments = json.loads(assessments_path.read_text(encoding="utf-8"))

    examples = []
    for c in comps:
        cid = c.get("id", "")
        bid = c.get("batch_id", 0)
        key = f"{cid}_b{bid}" if bid else cid
        a = assessments.get(key, {})
        q = str(a.get("quality", "")).strip()
        r = str(a.get("relevant", "")).lower().strip()
        q_num = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5}.get(q, 0)
        if q_num >= min_quality and r == "yes":
            examples.append(c)
            if len(examples) >= max_examples:
                break
    return examples


# ----------------------------------------------------------------------
# Prompt template
# ----------------------------------------------------------------------

def build_prompt(
    skills: List[str],
    future_context: str = "",
    few_shot_examples: Optional[List[Dict]] = None,
    skill_future_weights: Optional[Dict[str, Dict]] = None,
    skill_time_trends: Optional[Dict[str, Dict]] = None,
) -> str:
    # Annotate skills with future_weight and/or empirical trend when available
    def format_skill(s: str) -> str:
        s_stripped = s.strip()
        parts = []
        if skill_future_weights and s_stripped in skill_future_weights:
            w = skill_future_weights[s_stripped]
            fw = w.get("future_weight", 0)
            domain = w.get("best_future_domain", "")
            trend = w.get("trend_label", "")
            parts.append(f"future_weight={fw:.2f}, domain={domain}, forecast_trend={trend}")
        if skill_time_trends and s_stripped in skill_time_trends:
            t = skill_time_trends[s_stripped]
            emp = t.get("trend_label", "")
            if emp and emp != "Stable":
                parts.append(f"empirical_trend={emp}")
        if parts:
            return f"- {s} ({'; '.join(parts)})"
        return f"- {s}"

    bullet_list = "\n".join(format_skill(s) for s in skills)

    future_block = ""
    if future_context.strip():
        future_block = f"""
Additional context about future-critical domains and technologies:

{future_context}
"""

    few_shot_block = ""
    if few_shot_examples:
        ex_str = json.dumps(
            [{"id": e.get("id"), "title": e.get("title"), "description": e.get("description"),
              "related_skills": e.get("related_skills", []), "future_relevance": e.get("future_relevance", "")}
             for e in few_shot_examples],
            indent=2,
            ensure_ascii=False,
        )
        few_shot_block = f"""
Here are examples of HIGH-QUALITY competencies (use similar style and structure):

{ex_str}

"""

    future_weighting_block = ""
    if skill_future_weights or skill_time_trends:
        future_weighting_block = """
8. FUTURE WEIGHTING (important): Skills may be annotated with:
   - future_weight, domain, forecast_trend: from domain forecasts (Strong_Growth, Moderate_Growth, Decline).
   - empirical_trend: from actual job posting frequency over time (Emerging, Declining).
   - PRIORITIZE competencies that cover skills with HIGH future_weight and/or empirical_trend=Emerging.
   - Give emerging skills more prominence in competency titles and descriptions.
   - DEPRIORITIZE competencies built mainly around skills with NEGATIVE future_weight or empirical_trend=Declining.
   - Do not create competencies that focus primarily on declining skills.

"""

    return f"""
You are an expert in competency-based education and vocational curriculum design.

You are given a list of VERIFIED job skills (mostly hard skills) extracted from
real job postings in the Software & Game Development domain.

Your task is to group related skills and generate COMPETENCY STATEMENTS that are
suitable for use in an upper-secondary / vocational curriculum (e.g., Indonesian SMK).

Please follow these rules:

1. Output MUST be valid JSON only (no explanation text).
2. The root should be a JSON object with key "competencies" whose value is a list.
3. Each competency is an object with:
   - "id": a short identifier (e.g., "C1", "C2", ...).
   - "title": a concise competency title.
   - "description": 2–3 sentences competency statement that can be used in a curriculum.
   - "related_skills": list of skill phrases from the input that this competency covers.
   - "future_relevance": a short note (1–2 sentences) on why this competency
     matters for the future of work (based on the context if available).
4. Try to produce between 10 and 25 competencies for this batch.
5. Prefer higher-level, integrative competencies, not trivial one-skill items.
6. Give slightly higher priority and more detail to skills and themes that appear
   in future-critical domains (AI, data, cloud, security, human–AI collaboration),
   if such context is provided.
{future_weighting_block}
{few_shot_block}
{future_block}

Here are the verified skills:

{bullet_list}
"""


# ----------------------------------------------------------------------
# LLM call
# ----------------------------------------------------------------------

def _validate_competency(c: dict) -> bool:
    """Check that a competency dict has all required fields."""
    return (
        isinstance(c, dict)
        and all(k in c for k in COMPETENCY_REQUIRED_FIELDS)
        and isinstance(c.get("related_skills"), list)
    )


def _parse_llm_json(content: str) -> Optional[Dict]:
    """Parse LLM response JSON with fallback strategies."""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = re.sub(r"\n```$", "", content)
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    last_brace = max(content.rfind("}"), content.rfind("]"))
    if last_brace != -1:
        try:
            return json.loads(content[: last_brace + 1])
        except json.JSONDecodeError:
            pass

    return None


def call_llm_for_competencies(client: OpenAI,
                              skills: List[str],
                              model_name: str,
                              future_context: str = "",
                              few_shot_examples: Optional[List[Dict]] = None,
                              skill_future_weights: Optional[Dict[str, Dict]] = None,
                              skill_time_trends: Optional[Dict[str, Dict]] = None) -> Dict:
    prompt = build_prompt(
        skills,
        future_context=future_context,
        few_shot_examples=few_shot_examples,
        skill_future_weights=skill_future_weights,
        skill_time_trends=skill_time_trends,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise curriculum designer. "
                "Always respond with VALID JSON only. "
                "Do NOT include any markdown fences or commentary."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    last_error = None
    for attempt in range(1 + MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=4000,
            )
            content = resp.choices[0].message.content or ""
            data = _parse_llm_json(content)

            if data is not None:
                comps = data.get("competencies", [])
                valid = [c for c in comps if _validate_competency(c)]
                invalid_count = len(comps) - len(valid)
                if invalid_count > 0:
                    print(f"[WARN] Dropped {invalid_count} competencies "
                          f"with missing fields (attempt {attempt + 1})")
                data["competencies"] = valid
                return data

            last_error = "JSON parse failed"
        except Exception as e:
            last_error = str(e)

        if attempt < MAX_RETRIES:
            wait = RETRY_BACKOFF_BASE ** (attempt + 1)
            print(f"[WARN] Attempt {attempt + 1} failed ({last_error}), "
                  f"retrying in {wait}s...")
            time.sleep(wait)

    raw = content if 'content' in locals() else f"Error: {last_error}"
    with open("last_competency_raw_response.txt", "w", encoding="utf-8") as f:
        f.write(raw)

    print(f"[WARN] All {1 + MAX_RETRIES} attempts failed. "
          f"Raw content saved to last_competency_raw_response.txt.")
    return {"competencies": []}



# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate competency proposals from verified skills using an LLM."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(config.OUTPUT_DIR),
        help="Directory containing verified_skills.csv (default: config.OUTPUT_DIR)",
    )
    parser.add_argument(
        "--verified_file",
        type=str,
        default="verified_skills.csv",
        help="Input verified skills file (default: verified_skills.csv)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="competency_proposals.json",
        help="Output JSON file (default: competency_proposals.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek/deepseek-chat",
        help="Model name on OpenRouter (default: deepseek/deepseek-chat)",
    )
    parser.add_argument(
        "--max_skills_per_call",
        type=int,
        default=30,
        help="Number of skills per LLM call (default: 30)",
    )
    parser.add_argument(
        "--human_verified_only",
        action="store_true",
        help="Use only human-verified skills from feedback_store/human_verified_skills.csv",
    )
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Use all skills (with Bloom/type corrections) from advanced_skills_human_filtered.csv. "
             "Tags each competency with all_skills_human_verified. Some competencies may be from unverified skills.",
    )

    args = parser.parse_args()
    if args.human_verified_only and args.comprehensive:
        raise ValueError("Cannot use both --human_verified_only and --comprehensive. Choose one.")

    out_dir = Path(args.output_dir)
    output_path = out_dir / args.output_json
    feedback_dir = Path(config.PROJECT_ROOT) / "feedback_store"

    # Build future-of-work context from future_skill_weights_dummy.csv (if available)
    future_context = build_future_context(out_dir)
    if future_context:
        print("[INFO] Future context loaded and will be injected into LLM prompt.")
    else:
        print("[INFO] No future context available; generating competencies from skills only.")

    # Load human-verified skill set (for tagging in comprehensive mode)
    human_verified_skills: set = set()
    human_path = feedback_dir / "human_verified_skills.csv"
    if human_path.exists():
        hv_df = pd.read_csv(human_path)
        if "skill" in hv_df.columns:
            human_verified_skills = {
                str(s).strip().lower() for s in hv_df["skill"].dropna().unique()
            }
            print(f"[INFO] Loaded {len(human_verified_skills)} human-verified skills for tagging")

    if args.human_verified_only:
        human_path = feedback_dir / "human_verified_skills.csv"
        if not human_path.exists():
            raise FileNotFoundError(
                f"human_verified_skills.csv not found at {human_path}. "
                "Run export_for_review, review in web app, then import_feedback first."
            )
        print(f"[INFO] Using human-verified skills from {human_path}")
        df = pd.read_csv(human_path)
        if "skill" not in df.columns:
            raise ValueError("human_verified_skills.csv must contain 'skill' column")
        grp = df.groupby("skill").size().reset_index()
        grp.columns = ["skill", "freq"]
        skills_sorted = grp.sort_values("freq", ascending=False)["skill"].tolist()
    elif args.comprehensive:
        comp_path = out_dir / "advanced_skills_human_filtered.csv"
        if not comp_path.exists():
            comp_path = out_dir / "advanced_skills.csv"
        if not comp_path.exists():
            raise FileNotFoundError(
                f"Neither advanced_skills_human_filtered.csv nor advanced_skills.csv found in {out_dir}. "
                "Run apply_feedback first (without --filter_only) to apply Bloom/type corrections."
            )
        print(f"[INFO] Comprehensive mode: using all skills from {comp_path.name}")
        df = pd.read_csv(comp_path)
        if "skill" not in df.columns:
            raise ValueError(f"{comp_path.name} must contain 'skill' column")
        grp = df.groupby("skill").size().reset_index()
        grp.columns = ["skill", "freq"]
        skills_sorted = grp.sort_values("freq", ascending=False)["skill"].tolist()
    else:
        verified_path = out_dir / args.verified_file
        if not verified_path.exists():
            raise FileNotFoundError(f"verified_skills.csv not found: {verified_path}")
        print(f"[INFO] Reading verified skills from {verified_path}")
        df = pd.read_csv(verified_path)
        if "is_verified" not in df.columns:
            raise ValueError("verified_skills.csv must contain 'is_verified' column")
        df_v = df[df["is_verified"] == True].copy()
        if df_v.empty:
            raise RuntimeError("No verified skills found (is_verified == True).")
        grp = df_v.groupby("skill").size().reset_index()
        grp.columns = ["skill", "freq"]
        skills_sorted = grp.sort_values("freq", ascending=False)["skill"].tolist()

    print(f"[INFO] Total unique skills for competency generation: {len(skills_sorted)}")

    # Load skill future weights (for annotation and reordering)
    skill_future_weights = load_skill_future_weights(out_dir)
    if skill_future_weights:
        print(f"[INFO] Loaded future weights for {len(skill_future_weights)} skills")

    # Load empirical time trends (emerging/declining from job posting frequency)
    skill_time_trends = load_skill_time_trends(out_dir)
    if skill_time_trends:
        print(f"[INFO] Loaded empirical trends for {len(skill_time_trends)} skills")

    # Reorder skills: future_weight first, then empirical Emerging > Stable > Declining
    def sort_key(s: str) -> tuple:
        s_stripped = s.strip()
        fw = -999.0
        if skill_future_weights and s_stripped in skill_future_weights:
            fw = skill_future_weights[s_stripped].get("future_weight", -999.0)
        emp_rank = 0  # 2=Emerging, 1=Stable, 0=Declining or unknown
        if skill_time_trends and s_stripped in skill_time_trends:
            lbl = skill_time_trends[s_stripped].get("trend_label", "")
            emp_rank = {"Emerging": 2, "Stable": 1, "Declining": 0}.get(lbl, 0)
        return (fw, emp_rank)

    skills_sorted = sorted(skills_sorted, key=sort_key, reverse=True)
    if skill_future_weights or skill_time_trends:
        print("[INFO] Skills reordered by future_weight and empirical trend (emerging first)")

    # Load few-shot examples from human assessments (if available)
    few_shot = load_few_shot_examples(out_dir, feedback_dir, max_examples=3)
    if few_shot:
        print(f"[INFO] Using {len(few_shot)} few-shot examples from human assessments")

    client = load_openrouter_client()

    all_competencies = []
    batch_id = 1

    # Chunk skills for multiple LLM calls
    for i in range(0, len(skills_sorted), args.max_skills_per_call):
        chunk = skills_sorted[i : i + args.max_skills_per_call]
        print(f"[INFO] Calling LLM for batch {batch_id} "
              f"({len(chunk)} skills)...")

        data = call_llm_for_competencies(
            client,
            chunk,
            args.model,
            future_context=future_context,
            few_shot_examples=few_shot or [],
            skill_future_weights=skill_future_weights or None,
            skill_time_trends=skill_time_trends or None,
        )

        # Expecting {"competencies": [ ... ]}
        comps = data.get("competencies", [])
        # Annotate with batch id
        for c in comps:
            c.setdefault("batch_id", batch_id)
        all_competencies.extend(comps)

        batch_id += 1

    # Tag competencies with human verification status (comprehensive mode)
    if args.comprehensive and human_verified_skills:
        for c in all_competencies:
            related = c.get("related_skills") or []
            if not related:
                c["all_skills_human_verified"] = False
            else:
                related_lower = [str(s).strip().lower() for s in related]
                c["all_skills_human_verified"] = all(
                    r in human_verified_skills for r in related_lower
                )
    elif args.comprehensive:
        for c in all_competencies:
            c["all_skills_human_verified"] = False  # no human feedback loaded

    print(f"[INFO] Total competencies generated: {len(all_competencies)}")

    # Build output with metadata for comprehensive mode
    output_obj = {"competencies": all_competencies}
    if args.comprehensive:
        verified_count = sum(1 for c in all_competencies if c.get("all_skills_human_verified"))
        output_obj["generation_mode"] = "comprehensive"
        output_obj["note"] = (
            "Some competencies in this file are generated from skills not yet human-verified. "
            "Check the 'all_skills_human_verified' field per competency. "
            f"{verified_count}/{len(all_competencies)} competencies have all related skills human-verified."
        )

    output_path.write_text(
        json.dumps(output_obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[INFO] Saved competency proposals to {output_path}")


if __name__ == "__main__":
    main()
