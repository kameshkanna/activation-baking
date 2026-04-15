"""
Plotting utilities for the ICML 2026 paper:
"Norm-Calibrated Activation Baking: Behavioural Adapters via Weight-Space Symmetry Alignment"

All figures saved as both .pdf (vector, for paper) and .png (300 DPI, for slides).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "text.usetex": False,
})

logger = logging.getLogger(__name__)

MODEL_COLORS = {
    "llama": "#1f77b4",
    "qwen": "#ff7f0e",
    "gemma": "#2ca02c",
    "mistral": "#d62728",
}

MODEL_LABELS = {
    "llama": "Llama-3.1-8B",
    "qwen": "Qwen2.5-7B",
    "gemma": "Gemma-2-9B",
    "mistral": "Mistral-7B",
}

BEHAVIOR_LABELS = {
    "sycophancy_suppression": "Sycophancy\nSuppression",
    "refusal_calibration": "Refusal\nCalibration",
    "verbosity_control": "Verbosity\nControl",
    "formality": "Formality",
    "uncertainty_expression": "Uncertainty\nExpression",
}

METHOD_LABELS = {
    "none": "No Steering",
    "raw_addition": "Raw Addition",
    "pca_uncalibrated": "PCA (no K)",
    "pca_k_calibrated": "PCA + K-Cal (ours)",
}

METHOD_COLORS = {
    "none": "#999999",
    "raw_addition": "#aec7e8",
    "pca_uncalibrated": "#ffbb78",
    "pca_k_calibrated": "#2ca02c",
}


class PaperFigures:
    """Generates all figures for the ICML 2026 workshop paper."""

    def __init__(
        self,
        results_dir: str = "results",
        output_dir: str = "results/plots",
    ) -> None:
        """
        Initialise figure generator.

        Args:
            results_dir: Root directory containing experiment outputs.
            output_dir: Directory to write .pdf and .png files.
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig: plt.Figure, name: str) -> None:
        """Save figure as both PDF and PNG."""
        for ext in ("pdf", "png"):
            path = self.output_dir / f"{name}.{ext}"
            fig.savefig(path)
            logger.info("Saved %s", path)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 1: Norm trajectories + K-values across all models
    # ------------------------------------------------------------------

    def plot_norm_trajectories(self) -> None:
        """
        Line plot of mean activation norm vs normalised layer depth for all models.
        Dual y-axis: left = mean_norm, right = K_value.
        Saves: norm_trajectories_all_models.pdf/png
        """
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax2 = ax1.twinx()

        for model_name in MODEL_COLORS:
            csv_path = self.results_dir / "norm_profiles" / f"{model_name}.csv"
            if not csv_path.exists():
                logger.warning("Missing norm profile: %s", csv_path)
                continue
            df = pd.read_csv(csv_path)
            depth = np.linspace(0, 1, len(df))
            color = MODEL_COLORS[model_name]
            label = MODEL_LABELS[model_name]
            ax1.plot(depth, df["mean_norm"], color=color, linewidth=2, label=label)
            ax2.plot(depth, df["k_value"], color=color, linewidth=1.2, linestyle="--", alpha=0.6)

        ax1.set_xlabel("Normalised Layer Depth")
        ax1.set_ylabel("Mean Activation Norm (L2)")
        ax2.set_ylabel("K-value (μ / √d)", color="#555555")
        ax1.legend(loc="upper left", framealpha=0.9)
        ax1.set_title("Activation Norm and K-Value Trajectories Across Architectures")

        # Dashed-line legend entry for K
        from matplotlib.lines import Line2D
        extra = Line2D([0], [0], linestyle="--", color="#555555", linewidth=1.2, label="K-value (right axis)")
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles + [extra], labels + ["K-value (right axis)"], loc="upper left", framealpha=0.9)

        fig.tight_layout()
        self._save(fig, "norm_trajectories_all_models")

    # ------------------------------------------------------------------
    # Figure 2: Permutation invariance
    # ------------------------------------------------------------------

    def plot_permutation_invariance(self) -> None:
        """
        Box plot of subspace cosine similarity across permutations × layers for each model.
        Shows that PCA directions are invariant to neuron permutations.
        Saves: permutation_invariance.pdf/png
        """
        records: List[Dict] = []
        for model_name in MODEL_COLORS:
            behavior_dir = self.results_dir / "permutation_invariance" / model_name
            if not behavior_dir.exists():
                continue
            for behavior_csv in behavior_dir.glob("*/invariance_scores.csv"):
                behavior = behavior_csv.parent.name
                df = pd.read_csv(behavior_csv)
                for _, row in df.iterrows():
                    records.append({
                        "model": MODEL_LABELS[model_name],
                        "behavior": BEHAVIOR_LABELS.get(behavior, behavior),
                        "cosine_sim": row["subspace_cosine_sim"],
                    })

        if not records:
            logger.warning("No permutation invariance data found.")
            return

        data = pd.DataFrame(records)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.boxplot(
            data=data,
            x="model",
            y="cosine_sim",
            hue="behavior",
            ax=ax,
            palette="Set2",
            width=0.6,
            linewidth=1.2,
        )
        # Random baseline: expected cosine sim of random unit vectors in n_components-dim subspace
        n_components = 5
        random_baseline = 1.0 / np.sqrt(n_components)
        ax.axhline(random_baseline, color="red", linestyle="--", linewidth=1.5, label=f"Random baseline ({random_baseline:.2f})")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Model")
        ax.set_ylabel("Subspace Cosine Similarity")
        ax.set_title("Permutation Invariance of PCA Behavioral Directions")
        ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
        fig.tight_layout()
        self._save(fig, "permutation_invariance")

    # ------------------------------------------------------------------
    # Figure 3: Weight-space alignment heatmap
    # ------------------------------------------------------------------

    def plot_weight_space_alignment(self) -> None:
        """
        Heatmap: layer × behavior, colour = mean_max_alignment for each model.
        Side by side: PCA alignment vs random baseline.
        Saves: weight_space_alignment.pdf/png
        """
        models = [m for m in MODEL_COLORS if (self.results_dir / "weight_alignment" / m).exists()]
        if not models:
            logger.warning("No weight alignment data found.")
            return

        n_models = len(models)
        fig, axes = plt.subplots(2, n_models, figsize=(4 * n_models, 7))
        if n_models == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        for col, model_name in enumerate(models):
            alignment_data: Dict[str, pd.DataFrame] = {}
            model_dir = self.results_dir / "weight_alignment" / model_name
            for behavior_dir in model_dir.iterdir():
                csv_path = behavior_dir / "alignment_per_layer.csv"
                if csv_path.exists():
                    alignment_data[behavior_dir.name] = pd.read_csv(csv_path)

            if not alignment_data:
                continue

            behaviors = sorted(alignment_data.keys())
            first_df = next(iter(alignment_data.values()))
            n_layers = len(first_df)
            layer_step = max(1, n_layers // 10)

            pca_matrix = np.zeros((n_layers, len(behaviors)))
            rand_matrix = np.zeros((n_layers, len(behaviors)))

            for j, b in enumerate(behaviors):
                df = alignment_data[b]
                pca_matrix[:, j] = df["mean_max_alignment"].values
                rand_matrix[:, j] = df["random_baseline_alignment"].values

            behavior_labels = [BEHAVIOR_LABELS.get(b, b) for b in behaviors]
            layer_labels = [str(i) if i % layer_step == 0 else "" for i in range(n_layers)]

            for row, (mat, title) in enumerate([(pca_matrix, "PCA Directions"), (rand_matrix, "Random Baseline")]):
                ax = axes[row][col]
                sns.heatmap(
                    mat,
                    ax=ax,
                    xticklabels=behavior_labels,
                    yticklabels=layer_labels,
                    cmap="YlOrRd",
                    vmin=0.0,
                    vmax=0.6,
                    cbar=(col == n_models - 1),
                    linewidths=0.3,
                )
                ax.set_title(f"{MODEL_LABELS[model_name]}\n{title}", fontsize=10)
                ax.set_xlabel("Behavior" if row == 1 else "")
                ax.set_ylabel("Layer Index" if col == 0 else "")

        fig.suptitle("Alignment of PCA Directions with Weight-Matrix Singular Vectors", fontsize=13, y=1.02)
        fig.tight_layout()
        self._save(fig, "weight_space_alignment")

    # ------------------------------------------------------------------
    # Figure 4: Efficacy comparison bar chart
    # ------------------------------------------------------------------

    def plot_efficacy_comparison(self) -> None:
        """
        Grouped bar chart: accuracy per method across behaviors.
        Saves: efficacy_comparison.pdf/png
        """
        records: List[Dict] = []
        for model_name in MODEL_COLORS:
            model_dir = self.results_dir / "efficacy" / model_name
            if not model_dir.exists():
                continue
            for behavior_dir in model_dir.iterdir():
                csv_path = behavior_dir / "comparison.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    for _, row in df.iterrows():
                        records.append({
                            "model": MODEL_LABELS[model_name],
                            "behavior": BEHAVIOR_LABELS.get(behavior_dir.name, behavior_dir.name),
                            "method": METHOD_LABELS.get(row["method"], row["method"]),
                            "accuracy": row["accuracy"],
                        })

        if not records:
            logger.warning("No efficacy data found.")
            return

        data = pd.DataFrame(records)
        # Average across models
        grouped = data.groupby(["behavior", "method"])["accuracy"].mean().reset_index()

        behaviors = list(BEHAVIOR_LABELS.values())
        methods = list(METHOD_LABELS.values())
        x = np.arange(len(behaviors))
        width = 0.18
        offsets = np.linspace(-(len(methods) - 1) / 2, (len(methods) - 1) / 2, len(methods)) * width

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, method_label in enumerate(methods):
            method_data = grouped[grouped["method"] == method_label]
            vals = [
                method_data[method_data["behavior"] == b]["accuracy"].values[0]
                if len(method_data[method_data["behavior"] == b]) > 0
                else 0.0
                for b in behaviors
            ]
            method_key = [k for k, v in METHOD_LABELS.items() if v == method_label][0]
            ax.bar(
                x + offsets[i],
                vals,
                width,
                label=method_label,
                color=METHOD_COLORS[method_key],
                edgecolor="black",
                linewidth=0.6,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(behaviors, fontsize=10)
        ax.set_ylabel("Activation Direction Accuracy")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.6, label="Random (0.50)")
        ax.set_title("Baking Efficacy: Method Comparison Across Behaviors (avg. over 4 models)")
        ax.legend(loc="lower right", framealpha=0.9)
        fig.tight_layout()
        self._save(fig, "efficacy_comparison")

    # ------------------------------------------------------------------
    # Figure 5: Cross-architecture CKA heatmap
    # ------------------------------------------------------------------

    def plot_cross_arch_cka(self) -> None:
        """
        4×4 CKA similarity heatmap between architectures, per behavior (averaged).
        Saves: cross_arch_cka.pdf/png
        """
        model_names = list(MODEL_COLORS.keys())
        model_labels = [MODEL_LABELS[m] for m in model_names]

        behaviors = list(BEHAVIOR_LABELS.keys())
        n_behaviors = len(behaviors)

        fig, axes = plt.subplots(1, min(n_behaviors, 2), figsize=(6 * min(n_behaviors, 2), 5))
        if n_behaviors == 1:
            axes = [axes]
        else:
            axes = list(axes)

        for idx, behavior in enumerate(behaviors[:2]):
            csv_path = self.results_dir / "cross_arch" / behavior / "cka_matrix.csv"
            if not csv_path.exists():
                logger.warning("Missing CKA matrix: %s", csv_path)
                mat = np.eye(len(model_names))
            else:
                df = pd.read_csv(csv_path, index_col=0)
                mat = df.reindex(index=model_names, columns=model_names, fill_value=np.nan).values

            ax = axes[idx]
            mask = np.isnan(mat)
            sns.heatmap(
                mat,
                ax=ax,
                xticklabels=model_labels,
                yticklabels=model_labels,
                cmap="Blues",
                vmin=0.0,
                vmax=1.0,
                annot=True,
                fmt=".2f",
                mask=mask,
                linewidths=0.5,
                cbar=True,
            )
            ax.set_title(f"CKA Similarity — {BEHAVIOR_LABELS.get(behavior, behavior)}")

        fig.suptitle("Cross-Architecture Behavioral Direction Similarity (CKA)", fontsize=13)
        fig.tight_layout()
        self._save(fig, "cross_arch_cka")

    # ------------------------------------------------------------------
    # Figure 6: K-spectral correlation scatter
    # ------------------------------------------------------------------

    def plot_k_spectral_correlation(self) -> None:
        """
        Scatter: K-value vs spectral_norm per layer, all models overlaid.
        Includes regression line and Pearson r annotation.
        Saves: k_spectral_correlation.pdf/png
        """
        from scipy import stats

        fig, ax = plt.subplots(figsize=(6, 5))
        all_k: List[float] = []
        all_s: List[float] = []

        for model_name in MODEL_COLORS:
            csv_path = self.results_dir / "k_calibration" / f"{model_name}_k_vs_spectral.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path).dropna(subset=["k_value", "spectral_norm_down"])
            k_vals = df["k_value"].values
            s_vals = df["spectral_norm_down"].values
            ax.scatter(
                k_vals, s_vals,
                color=MODEL_COLORS[model_name],
                label=MODEL_LABELS[model_name],
                alpha=0.6,
                s=30,
                edgecolors="none",
            )
            all_k.extend(k_vals.tolist())
            all_s.extend(s_vals.tolist())

        if len(all_k) > 2:
            slope, intercept, r_value, p_value, _ = stats.linregress(all_k, all_s)
            x_line = np.linspace(min(all_k), max(all_k), 200)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, "k--", linewidth=1.5, label=f"Regression (r={r_value:.3f})")
            ax.annotate(
                f"Pearson r = {r_value:.3f}\np = {p_value:.2e}",
                xy=(0.97, 0.05),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
            )

        ax.set_xlabel("K-value (μ / √d)")
        ax.set_ylabel("Spectral Norm of W_down")
        ax.set_title("K-Calibration vs. Weight Spectral Norm (per layer, all models)")
        ax.legend(loc="upper left", framealpha=0.9)
        fig.tight_layout()
        self._save(fig, "k_spectral_correlation")

    # ------------------------------------------------------------------
    # Generate all figures
    # ------------------------------------------------------------------

    def generate_all(self) -> None:
        """Generate all paper figures sequentially."""
        logger.info("Generating all paper figures → %s", self.output_dir)
        self.plot_norm_trajectories()
        self.plot_permutation_invariance()
        self.plot_weight_space_alignment()
        self.plot_efficacy_comparison()
        self.plot_cross_arch_cka()
        self.plot_k_spectral_correlation()
        logger.info("All figures generated.")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/plots")
    parser.add_argument("--figure", default="all", help="Specific figure name or 'all'")
    args = parser.parse_args()

    figs = PaperFigures(results_dir=args.results_dir, output_dir=args.output_dir)
    if args.figure == "all":
        figs.generate_all()
    else:
        method = getattr(figs, f"plot_{args.figure}", None)
        if method is None:
            raise ValueError(f"Unknown figure: {args.figure}")
        method()
