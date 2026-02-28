"""
power_analysis_gold_set.py

Power analysis for gold set size: what precision difference can we detect
with given n, alpha, and power?

Assumptions (one-proportion binomial test):
    H0: precision = 0.5 (chance)
    H1: precision = p1 (e.g., 0.7)
    alpha = 0.05
    power = 0.80

Output: required n and/or power for current gold set sizes (150 skills, 100 knowledge).
"""

import argparse
import sys
from pathlib import Path

from scipy import stats

# Allow import from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def power_for_n(n: int, p0: float = 0.5, p1: float = 0.7, alpha: float = 0.05) -> float:
    """
    Power of one-sided binomial test when true proportion is p1.
    Critical region: reject H0 if X >= k where k = smallest int s.t. P(X >= k | p0) <= alpha.
    Power = P(X >= k | p1).
    """
    if n <= 0:
        return 0.0
    dist0 = stats.binom(n, p0)
    # Find k: P(X >= k) <= alpha under H0
    for k in range(0, n + 1):
        if 1 - dist0.cdf(k - 1) <= alpha:
            break
    else:
        k = n + 1  # never reject
    dist1 = stats.binom(n, p1)
    return 1 - dist1.cdf(k - 1)


def n_for_power(
    power: float = 0.80,
    p0: float = 0.5,
    p1: float = 0.7,
    alpha: float = 0.05,
    n_max: int = 500,
) -> int:
    """Smallest n such that power >= target."""
    for n in range(1, n_max + 1):
        if power_for_n(n, p0, p1, alpha) >= power:
            return n
    return n_max


def main():
    parser = argparse.ArgumentParser(
        description="Power analysis for gold set size (one-proportion binomial test)."
    )
    parser.add_argument(
        "--p0",
        type=float,
        default=0.5,
        help="Null hypothesis: precision = p0 (default: 0.5)",
    )
    parser.add_argument(
        "--p1",
        type=float,
        default=0.7,
        help="Alternative: precision = p1 (default: 0.7, target from RQ1)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Type I error (default: 0.05)",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=0.80,
        help="Target power (default: 0.80)",
    )
    parser.add_argument(
        "--n_skills",
        type=int,
        default=150,
        help="Current gold set size for skills (default: 150)",
    )
    parser.add_argument(
        "--n_knowledge",
        type=int,
        default=100,
        help="Current gold set size for knowledge (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Write results to RESEARCH_QUESTIONS.md Power analysis subsection (path relative to PROJECT_ROOT)",
    )
    args = parser.parse_args()

    n_req = n_for_power(power=args.power, p0=args.p0, p1=args.p1, alpha=args.alpha)
    pw_skills = power_for_n(args.n_skills, args.p0, args.p1, args.alpha)
    pw_knowledge = power_for_n(args.n_knowledge, args.p0, args.p1, args.alpha)

    print(f"Power analysis (H0: p={args.p0}, H1: p={args.p1}, alpha={args.alpha})")
    print(f"  Required n for power {args.power}: {n_req}")
    print(f"  Current n_skills={args.n_skills} -> power={pw_skills:.3f}")
    print(f"  Current n_knowledge={args.n_knowledge} -> power={pw_knowledge:.3f}")

    if args.output:
        out_path = Path(config.PROJECT_ROOT) / args.output
        if out_path.exists():
            content = out_path.read_text(encoding="utf-8")
            section = f"""
### Power Analysis

- Assumptions: H0 precision={args.p0} (chance), H1 precision={args.p1}, alpha={args.alpha}, target power={args.power}
- Required n: {n_req} items to achieve power >= {args.power}
- Current gold set: skills n={args.n_skills} yields power={pw_skills:.3f}; knowledge n={args.n_knowledge} yields power={pw_knowledge:.3f}
- Interpretation: With n=150 skills we have adequate power to detect precision > {args.p0} when true precision is {args.p1} or higher.
"""
            # Insert before "## Ablation" or similar if present
            if "## Ablation" in content and "### Power Analysis" not in content:
                content = content.replace(
                    "## Ablation Study Design",
                    "## Power Analysis (Gold Set)\n" + section.strip() + "\n\n---\n\n## Ablation Study Design",
                    1,
                )
                out_path.write_text(content, encoding="utf-8")
                print(f"[INFO] Appended power analysis to {out_path}")
            else:
                print(f"[WARN] Could not insert power analysis; add manually to {out_path}")


if __name__ == "__main__":
    main()
