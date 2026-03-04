"""
plot_scientific_analysis.py

Generates visualizations for scientific/quantitative analysis to add rigor and
explain how calculations perform. Complements plot_generator.py with
statistical, regression, and evaluation-focused plots.

Scientific methods visualized (see SCIENTIFIC_METHODOLOGY.md):
    - Extraction precision with Wilson CI (binomial paradigm)
    - Odds ratio vs chance (effect size)
    - Skill time trends: FDR volcano, sample regression lines
    - Calibration curve (reliability diagram)
    - Power curve for gold set size
    - Future-domain mapping accuracy and margin
    - Weight sensitivity (Jaccard vs baseline)

Inputs (from config.OUTPUT_DIR):
    - extraction_evaluation_report.json
    - skill_time_trends.csv
    - advanced_skills_with_dates.csv (for regression lines)
    - parameter_validation_report.json
    - future_mapping_evaluation_report.json
    - weight_sensitivity_report.json

Outputs:
    - results/figures/scientific_*.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

import config

FIG_DIR_NAME = "figures"


def _unwrap_aggregated_json(data) -> dict:
    """If data is aggregated format (list of per-run dicts), return first run for plotting."""
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        return data[0]
    if isinstance(data, dict):
        return data
    return {}


def ensure_fig_dir(output_dir: Path) -> Path:
    fig_dir = output_dir / FIG_DIR_NAME
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def plot_extraction_precision_with_ci(extraction_report: dict, fig_dir: Path) -> None:
    """Bar chart: precision by source with Wilson 95% CI error bars."""
    if not extraction_report:
        return
    for metric_type in ["skills", "knowledge"]:
        data = extraction_report.get(metric_type, {}).get("by_source", [])
        by_source = [r for r in data if r.get("source") != "all"]
        if not by_source:
            continue
        sources = [r["source"] for r in by_source]
        precisions = [r["precision"] for r in by_source]
        ci = [r.get("precision_ci_95", [r["precision"], r["precision"]]) for r in by_source]
        sig = [r.get("significant_at_005", False) for r in by_source]
        err_lo = [p - ci[i][0] for i, p in enumerate(precisions)]
        err_hi = [ci[i][1] - p for i, p in enumerate(precisions)]
        colors = ["#2ecc71" if s else "#95a5a6" for s in sig]
        x = np.arange(len(sources))
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(x, precisions, color=colors, edgecolor="black", linewidth=0.5)
        ax.errorbar(x, precisions, yerr=[err_lo, err_hi], fmt="none", color="black", capsize=4)
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.7, label="Chance (0.5)")
        ax.set_xticks(x)
        ax.set_xticklabels(sources, rotation=20, ha="right")
        ax.set_ylabel("Precision")
        ax.set_title(f"Extraction Precision by Source ({metric_type.capitalize()})\nWilson 95% CI; green = sig. vs chance")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.5, len(sources) - 0.5)
        plt.tight_layout()
        plt.savefig(fig_dir / f"scientific_extraction_precision_{metric_type}.png", dpi=300)
        plt.close()
        print(f"[INFO] Saved scientific_extraction_precision_{metric_type}.png")


def plot_odds_ratio_forest(extraction_report: dict, fig_dir: Path) -> None:
    """Forest-style plot: odds ratio vs chance (1.0) by source."""
    if not extraction_report:
        return
    for metric_type in ["skills", "knowledge"]:
        data = extraction_report.get(metric_type, {}).get("by_source", [])
        by_source = [r for r in data if r.get("source") != "all"]
        if not by_source:
            continue
        sources = [r["source"] for r in by_source]
        ors = [r.get("odds_ratio", 1.0) for r in by_source]
        ors_safe = [min(max(o, 0.1), 100) for o in ors]
        colors = ["#3498db" if o >= 1 else "#e74c3c" for o in ors]
        x = np.arange(len(sources))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(x, ors_safe, color=colors, edgecolor="black", linewidth=0.5)
        ax.axvline(1.0, color="black", linestyle="--", linewidth=2, label="Chance (OR=1)")
        ax.set_yticks(x)
        ax.set_yticklabels(sources)
        ax.set_xlabel("Odds Ratio vs Chance (p0=0.5)")
        ax.set_title(f"Effect Size: Odds Ratio by Source ({metric_type.capitalize()})")
        ax.legend()
        if max(ors_safe) > 5 or min(ors_safe) < 0.5:
            ax.set_xscale("log")
        plt.tight_layout()
        plt.savefig(fig_dir / f"scientific_odds_ratio_{metric_type}.png", dpi=300)
        plt.close()
        print(f"[INFO] Saved scientific_odds_ratio_{metric_type}.png")


def plot_binomial_test_example(extraction_report: dict, fig_dir: Path) -> None:
    """Illustrate binomial test: observed count vs H0 distribution for 'all' or first source with n>20."""
    if not extraction_report:
        return
    for metric_type in ["skills", "knowledge"]:
        data = extraction_report.get(metric_type, {}).get("by_source", [])
        row = next((r for r in data if r.get("source") == "all"), data[0] if data else None)
        if not row or row.get("n", 0) < 10:
            continue
        n = row["n"]
        k = row["n_correct"]
        p0 = 0.5
        x_vals = np.arange(0, n + 1)
        pmf = stats.binom.pmf(x_vals, n, p0)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x_vals, pmf, color="#ecf0f1", edgecolor="#7f8c8d", linewidth=0.5, label=f"H0: p={p0}")
        ax.axvline(k, color="#e74c3c", linewidth=3, label=f"Observed = {k}")
        ax.axvline(n * p0, color="#3498db", linestyle="--", linewidth=1.5, label=f"Expected under H0 = {int(n*p0)}")
        ax.set_xlabel("Number of correct extractions")
        ax.set_ylabel("Probability (H₀: Bin(n, 0.5))")
        ax.set_title(f"Binomial Test: {metric_type.capitalize()} (n={n}, observed={k})\nH0: precision=0.5 vs H1: precision>0.5")
        ax.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"scientific_binomial_test_{metric_type}.png", dpi=300)
        plt.close()
        print(f"[INFO] Saved scientific_binomial_test_{metric_type}.png")


def plot_trend_volcano(trends_df: pd.DataFrame, fig_dir: Path) -> None:
    """Volcano-style: slope vs -log10(q_value), colored by trend_label (Emerging/Declining/Stable)."""
    if trends_df.empty or "slope" not in trends_df.columns or "q_value" not in trends_df.columns:
        return
    df = trends_df.copy()
    df["neg_log10_q"] = -np.log10(df["q_value"].replace(0, 1e-10).values)
    df["trend_label"] = df.get("trend_label", "Stable")
    colors = {"Emerging": "#27ae60", "Declining": "#e74c3c", "Stable": "#95a5a6"}
    df["color"] = df["trend_label"].map(colors).fillna("#95a5a6")
    fig, ax = plt.subplots(figsize=(10, 7))
    for label in ["Stable", "Declining", "Emerging"]:
        subset = df[df["trend_label"] == label]
        if subset.empty:
            continue
        ax.scatter(subset["slope"], subset["neg_log10_q"], c=colors[label], label=label, alpha=0.7, s=30)
    ax.axhline(-np.log10(0.05), color="black", linestyle="--", alpha=0.7, label="q=0.05")
    ax.axvline(0, color="gray", linestyle="-", alpha=0.5)
    ax.set_xlabel("Slope (freq/month)")
    ax.set_ylabel("-log10(q-value)")
    ax.set_title("FDR Volcano: Skill Time Trends\nGreen=Emerging, Red=Declining, Gray=Stable")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "scientific_trend_volcano.png", dpi=300)
    plt.close()
    print("[INFO] Saved scientific_trend_volcano.png")


# Stop words and low-quality terms to exclude from regression sample plots
_REGRESSION_STOP_WORDS = {
    "and", "or", "the", "a", "an", "on", "to", "of", "in", "at", "by", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may", "might", "must",
    "can", "it", "its", "this", "that", "these", "those", "as", "if", "than",
    "but", "so", "just", "only", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "not", "nor", "too", "very", "also", "now", "here",
    "there", "when", "where", "why", "how", "all", "any", "same", "own",
    "detail", "details",  # too ambiguous (attention to detail vs. project details)
    "building", "driven", "motivated", "oriented", "based",  # generic suffixes
}


def _is_quality_skill_for_regression(skill: str) -> bool:
    """Filter out stop words and low-quality terms for regression sample plots."""
    if not isinstance(skill, str) or not skill.strip():
        return False
    t = skill.strip().lower()
    words = t.split()
    if len(words) < 2:
        # Single-word skills: exclude if it's a stop word
        return t not in _REGRESSION_STOP_WORDS and len(t) >= 4
    if any(w.lower() in _REGRESSION_STOP_WORDS for w in words):
        return False
    if len(t) < 4:
        return False
    return True


def plot_sample_regression_lines(skills_dates_df: pd.DataFrame, trends_df: pd.DataFrame, fig_dir: Path, n_skills: int = 3) -> None:
    """Plot actual data + fitted regression for top emerging and top declining skills.
    Filters out stop words and low-quality terms (e.g. 'and', 'detail') for scientific rigor."""
    if skills_dates_df.empty or trends_df.empty:
        return
    if "job_date" not in skills_dates_df.columns or "skill" not in skills_dates_df.columns:
        return
    skills_dates_df = skills_dates_df.copy()
    skills_dates_df["job_date"] = pd.to_datetime(skills_dates_df["job_date"], errors="coerce")
    skills_dates_df["year_month"] = skills_dates_df["job_date"].dt.to_period("M").astype(str)
    months = sorted(skills_dates_df["year_month"].unique())
    month_idx = {m: i for i, m in enumerate(months)}
    skills_dates_df["month_idx"] = skills_dates_df["year_month"].map(month_idx)
    freq_df = skills_dates_df.groupby(["skill", "year_month", "month_idx"])["job_id"].nunique().reset_index()
    freq_df.columns = ["skill", "year_month", "month_idx", "freq"]
    # Filter to high-quality skills only
    trends_df = trends_df[trends_df["skill"].apply(_is_quality_skill_for_regression)]
    if trends_df.empty:
        return
    emerging = trends_df[trends_df["trend_label"] == "Emerging"].nlargest(n_skills, "slope")
    declining = trends_df[trends_df["trend_label"] == "Declining"].nsmallest(n_skills, "slope")
    to_plot = []
    for _, r in emerging.iterrows():
        to_plot.append((r["skill"], r["slope"], "Emerging", "#27ae60"))
    for _, r in declining.iterrows():
        to_plot.append((r["skill"], r["slope"], "Declining", "#e74c3c"))
    if not to_plot:
        # Fallback: use top positive and negative slopes (Stable) for illustration
        pos = trends_df[trends_df["slope"] > 0].nlargest(n_skills, "slope")
        neg = trends_df[trends_df["slope"] < 0].nsmallest(n_skills, "slope")
        for _, r in pos.iterrows():
            to_plot.append((r["skill"], r["slope"], "Positive slope", "#27ae60"))
        for _, r in neg.iterrows():
            to_plot.append((r["skill"], r["slope"], "Negative slope", "#e74c3c"))
    if not to_plot:
        return
    n_plots = len(to_plot)
    cols = 2
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = axes.flatten()
    for idx, (skill, slope, label, color) in enumerate(to_plot):
        ax = axes[idx] if idx < len(axes) else axes[-1]
        sub = freq_df[freq_df["skill"] == skill]
        if sub.empty:
            continue
        x = sub["month_idx"].values.astype(float)
        y = sub["freq"].values.astype(float)
        ax.scatter(x, y, color=color, s=50, alpha=0.8, label="Observed")
        if len(x) >= 2:
            res = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 50)
            y_line = res.intercept + res.slope * x_line
            ax.plot(x_line, y_line, color="black", linestyle="--", linewidth=2, label=f"Fit: slope={res.slope:.3f}")
        ax.set_xlabel("Month index")
        ax.set_ylabel("Job count")
        ax.set_title(f"{skill[:40]}... ({label})" if len(skill) > 40 else f"{skill} ({label})")
        ax.legend()
    for idx in range(len(to_plot), len(axes)):
        axes[idx].set_visible(False)
    plt.suptitle("Sample Regression: Skill Frequency vs Time (FDR-significant)", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "scientific_trend_regression_samples.png", dpi=300)
    plt.close()
    print("[INFO] Saved scientific_trend_regression_samples.png")


def plot_calibration_curve(validation_report: dict, fig_dir: Path) -> None:
    """Reliability diagram: predicted vs observed frequency per bin."""
    if not validation_report or validation_report.get("status") != "ok":
        return
    validations = validation_report.get("validations", [])
    with_cal = [v for v in validations if v.get("calibration", {}).get("bin_centers")]
    if not with_cal:
        return
    n_axes = min(3, len(with_cal))
    fig, axes = plt.subplots(1, n_axes, figsize=(5 * n_axes, 5))
    if n_axes == 1:
        axes = [axes]
    for idx, v in enumerate(with_cal[:3]):
        cal = v["calibration"]
        ax = axes[idx] if idx < len(axes) else axes[-1]
        pred = cal["predicted_mean"]
        obs = cal["observed_freq"]
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.plot(pred, obs, "o-", color="#3498db", linewidth=2, markersize=8, label="Model")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(f"{v.get('column', 'Score')} (Brier={v.get('brier_score', 'N/A')})")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    plt.suptitle("Calibration Curves (Reliability Diagram)", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "scientific_calibration_curve.png", dpi=300)
    plt.close()
    print("[INFO] Saved scientific_calibration_curve.png")


def plot_power_curve(fig_dir: Path, p0: float = 0.5, p1: float = 0.7, alpha: float = 0.05, n_max: int = 150) -> None:
    """Power vs n for one-sided binomial test."""
    from scipy.stats import binom
    n_vals = np.arange(5, n_max + 1, 2)
    powers = []
    for n in n_vals:
        dist0 = binom(n, p0)
        k_crit = 0
        for k in range(0, n + 1):
            if 1 - dist0.cdf(k - 1) <= alpha:
                k_crit = k
                break
        dist1 = binom(n, p1)
        power = 1 - dist1.cdf(k_crit - 1)
        powers.append(power)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_vals, powers, color="#9b59b6", linewidth=2)
    ax.axhline(0.80, color="red", linestyle="--", alpha=0.8, label="Target power 0.80")
    ax.axvline(37, color="green", linestyle=":", alpha=0.8, label="n=37 (min for 0.80)")
    ax.fill_between(n_vals, 0, powers, alpha=0.2, color="#9b59b6")
    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("Power")
    ax.set_title(f"Power Curve: H0 p={p0} vs H1 p={p1}, alpha={alpha}\nOne-sided binomial test")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.set_xlim(5, n_max)
    plt.tight_layout()
    plt.savefig(fig_dir / "scientific_power_curve.png", dpi=300)
    plt.close()
    print("[INFO] Saved scientific_power_curve.png")


def plot_future_mapping(mapping_report: dict, fig_dir: Path) -> None:
    """Top-1 accuracy bar and mapping margin distribution (if per-item data exists)."""
    if not mapping_report or mapping_report.get("status") != "ok":
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    top1 = mapping_report.get("top1_accuracy", 0)
    axes[0].bar(["Top-1 accuracy"], [top1], color="#3498db", edgecolor="black")
    axes[0].axhline(0.6, color="red", linestyle="--", alpha=0.7, label="Target 0.60")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Future-Domain Mapping: Top-1 vs Expert")
    axes[0].legend()
    per_item = mapping_report.get("per_item", [])
    if per_item:
        margins = [r.get("margin", 0) for r in per_item if isinstance(r.get("margin"), (int, float))]
        if margins:
            axes[1].hist(margins, bins=20, color="#9b59b6", edgecolor="black", alpha=0.8)
            axes[1].axvline(np.mean(margins), color="red", linestyle="--", linewidth=2, label=f"Mean={np.mean(margins):.3f}")
            axes[1].set_xlabel("Mapping margin (top1_sim - top2_sim)")
            axes[1].set_ylabel("Count")
            axes[1].set_title("Mapping Confidence (Margin)")
            axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No per-item margin data", ha="center", va="center", transform=axes[1].transAxes)
    plt.suptitle("Future-Domain Mapping Evaluation", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "scientific_future_mapping.png", dpi=300)
    plt.close()
    print("[INFO] Saved scientific_future_mapping.png")


def plot_weight_sensitivity(sensitivity_report: dict, fig_dir: Path) -> None:
    """Jaccard vs baseline for each weight configuration."""
    if not sensitivity_report or "configs" not in sensitivity_report:
        return
    configs = sensitivity_report["configs"]
    keys = list(configs.keys())
    jaccards = [configs[k].get("jaccard_vs_baseline", 0) for k in keys]
    colors = ["#2ecc71" if j >= 0.6 else "#f39c12" if j >= 0.4 else "#e74c3c" for j in jaccards]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(keys))
    ax.bar(x, jaccards, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0.6, color="green", linestyle="--", alpha=0.7, label="Robust (J≥0.6)")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=30, ha="right")
    ax.set_ylabel("Jaccard vs baseline")
    ax.set_title("Weight Sensitivity: Top-20 Overlap with Baseline (d0.4_t0.3_f0.3)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(fig_dir / "scientific_weight_sensitivity.png", dpi=300)
    plt.close()
    print("[INFO] Saved scientific_weight_sensitivity.png")


def main():
    parser = argparse.ArgumentParser(
        description="Generate scientific/quantitative analysis plots for pipeline evaluation."
    )
    parser.add_argument("--output_dir", type=str, default=str(config.OUTPUT_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    fig_dir = ensure_fig_dir(output_dir)

    extraction_path = output_dir / "extraction_evaluation_report.json"
    extraction_report = {}
    if extraction_path.exists():
        try:
            raw = json.loads(extraction_path.read_text(encoding="utf-8"))
            extraction_report = _unwrap_aggregated_json(raw)
        except Exception as e:
            print(f"[WARN] Could not load extraction report: {e}")

    trends_path = output_dir / "skill_time_trends.csv"
    trends_df = pd.DataFrame()
    if trends_path.exists():
        try:
            trends_df = pd.read_csv(trends_path)
        except Exception as e:
            print(f"[WARN] Could not load skill trends: {e}")

    skills_dates_path = output_dir / "advanced_skills_with_dates.csv"
    skills_dates_df = pd.DataFrame()
    if skills_dates_path.exists():
        try:
            skills_dates_df = pd.read_csv(skills_dates_path)
        except Exception as e:
            print(f"[WARN] Could not load skills with dates: {e}")

    validation_path = output_dir / "parameter_validation_report.json"
    validation_report = {}
    if validation_path.exists():
        try:
            raw = json.loads(validation_path.read_text(encoding="utf-8"))
            validation_report = _unwrap_aggregated_json(raw)
        except Exception as e:
            print(f"[WARN] Could not load validation report: {e}")

    mapping_path = output_dir / "future_mapping_evaluation_report.json"
    mapping_report = {}
    if mapping_path.exists():
        try:
            raw = json.loads(mapping_path.read_text(encoding="utf-8"))
            mapping_report = _unwrap_aggregated_json(raw)
        except Exception as e:
            print(f"[WARN] Could not load mapping report: {e}")

    sensitivity_path = output_dir / "weight_sensitivity_report.json"
    sensitivity_report = {}
    if sensitivity_path.exists():
        try:
            raw = json.loads(sensitivity_path.read_text(encoding="utf-8"))
            sensitivity_report = _unwrap_aggregated_json(raw)
        except Exception as e:
            print(f"[WARN] Could not load weight sensitivity report: {e}")

    print("[INFO] Generating scientific analysis plots...")

    plot_extraction_precision_with_ci(extraction_report, fig_dir)
    plot_odds_ratio_forest(extraction_report, fig_dir)
    plot_binomial_test_example(extraction_report, fig_dir)
    plot_trend_volcano(trends_df, fig_dir)
    plot_sample_regression_lines(skills_dates_df, trends_df, fig_dir)
    plot_calibration_curve(validation_report, fig_dir)
    plot_power_curve(fig_dir)
    plot_future_mapping(mapping_report, fig_dir)
    plot_weight_sensitivity(sensitivity_report, fig_dir)

    print("[INFO] Scientific analysis plots done.")


if __name__ == "__main__":
    main()
