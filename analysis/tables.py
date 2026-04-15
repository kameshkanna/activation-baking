"""
LaTeX table generation for the ICML 2026 paper:
"Norm-Calibrated Activation Baking: Behavioural Adapters via Weight-Space Symmetry Alignment"
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_NAMES = ["llama", "qwen", "gemma", "mistral"]
MODEL_LABELS = {
    "llama": "Llama-3.1-8B",
    "qwen": "Qwen2.5-7B",
    "gemma": "Gemma-2-9B",
    "mistral": "Mistral-7B",
}
BEHAVIOR_LABELS = {
    "sycophancy_suppression": "Sycophancy Supp.",
    "refusal_calibration": "Refusal Cal.",
    "verbosity_control": "Verbosity Ctrl.",
    "formality": "Formality",
    "uncertainty_expression": "Uncertainty Expr.",
}
METHOD_LABELS = {
    "none": "No Steering",
    "raw_addition": "Raw Addition",
    "pca_uncalibrated": r"PCA (no $K$)",
    "pca_k_calibrated": r"\textbf{PCA + $K$-Cal (ours)}",
}


def _bold_max_in_row(values: List[float], fmt: str = "{:.3f}") -> List[str]:
    """Format row values, bolding the maximum."""
    max_val = max(values)
    return [
        r"\textbf{" + fmt.format(v) + "}" if abs(v - max_val) < 1e-9 else fmt.format(v)
        for v in values
    ]


class PaperTables:
    """Generates LaTeX tables for the ICML 2026 workshop paper."""

    def __init__(
        self,
        results_dir: str = "results",
        output_dir: str = "results/plots",
    ) -> None:
        """
        Initialise table generator.

        Args:
            results_dir: Root directory containing experiment outputs.
            output_dir: Directory to write .tex table files.
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _write(self, content: str, filename: str) -> None:
        path = self.output_dir / filename
        path.write_text(content, encoding="utf-8")
        logger.info("Saved table: %s", path)

    # ------------------------------------------------------------------
    # Table 1: Main efficacy results
    # ------------------------------------------------------------------

    def table_efficacy_main(self) -> None:
        """
        Main results table: rows=behaviors, columns=methods,
        cells=mean accuracy ± std across models.
        Best value per row is bolded.
        Saves: table_efficacy.tex
        """
        methods = list(METHOD_LABELS.keys())
        behaviors = list(BEHAVIOR_LABELS.keys())

        # Aggregate across models
        agg: Dict[str, Dict[str, List[float]]] = {b: {m: [] for m in methods} for b in behaviors}

        for model_name in MODEL_NAMES:
            for behavior in behaviors:
                csv_path = self.results_dir / "efficacy" / model_name / behavior / "comparison.csv"
                if not csv_path.exists():
                    continue
                df = pd.read_csv(csv_path)
                for method in methods:
                    row = df[df["method"] == method]
                    if not row.empty:
                        agg[behavior][method].append(float(row["accuracy"].values[0]))

        col_header = " & ".join(METHOD_LABELS.values())
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{Baking efficacy: activation direction accuracy (mean $\pm$ std over 4 models). "
            r"Best per row in \textbf{bold}.}",
            r"\label{tab:efficacy}",
            r"\begin{tabular}{l" + "c" * len(methods) + "}",
            r"\toprule",
            r"\textbf{Behavior} & " + col_header + r" \\",
            r"\midrule",
        ]

        for behavior in behaviors:
            means = []
            stds = []
            for method in methods:
                vals = agg[behavior][method]
                means.append(np.mean(vals) if vals else 0.0)
                stds.append(np.std(vals) if len(vals) > 1 else 0.0)

            formatted = []
            max_mean = max(means)
            for mean, std in zip(means, stds):
                cell = f"{mean:.3f}" + (f" $\\pm$ {std:.3f}" if std > 0 else "")
                if abs(mean - max_mean) < 1e-9:
                    cell = r"\textbf{" + cell + "}"
                formatted.append(cell)

            row_label = BEHAVIOR_LABELS[behavior]
            lines.append(row_label + " & " + " & ".join(formatted) + r" \\")

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        self._write("\n".join(lines), "table_efficacy.tex")

    # ------------------------------------------------------------------
    # Table 2: Permutation invariance
    # ------------------------------------------------------------------

    def table_permutation_invariance(self) -> None:
        """
        Rows=models, columns=behaviors, cells=mean cosine sim ± std.
        Saves: table_perm_inv.tex
        """
        behaviors = list(BEHAVIOR_LABELS.keys())
        col_header = " & ".join(BEHAVIOR_LABELS.values())
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{Permutation invariance: mean subspace cosine similarity of PCA directions "
            r"before and after random neuron permutation (5 permutations $\times$ all layers). "
            r"Higher is better; random baseline $\approx 0.45$.}",
            r"\label{tab:perm_inv}",
            r"\begin{tabular}{l" + "c" * len(behaviors) + "}",
            r"\toprule",
            r"\textbf{Model} & " + col_header + r" \\",
            r"\midrule",
        ]

        for model_name in MODEL_NAMES:
            row_vals = []
            for behavior in behaviors:
                csv_path = (
                    self.results_dir / "permutation_invariance" / model_name / behavior / "invariance_scores.csv"
                )
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    mean = df["subspace_cosine_sim"].mean()
                    std = df["subspace_cosine_sim"].std()
                    row_vals.append(f"{mean:.3f} $\\pm$ {std:.3f}")
                else:
                    row_vals.append("—")

            lines.append(MODEL_LABELS[model_name] + " & " + " & ".join(row_vals) + r" \\")

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        self._write("\n".join(lines), "table_perm_inv.tex")

    # ------------------------------------------------------------------
    # Table 3: K-spectral correlation
    # ------------------------------------------------------------------

    def table_k_spectral_correlation(self) -> None:
        """
        Rows=models, columns={Pearson r, Spearman r, p-value, mean K/spectral ratio}.
        Saves: table_k_spectral.tex
        """
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{Correlation between K-calibration values and weight spectral norms "
            r"(MLP down projection) across all layers.}",
            r"\label{tab:k_spectral}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"\textbf{Model} & \textbf{Pearson $r$} & \textbf{Spearman $\rho$} "
            r"& \textbf{$p$-value} & \textbf{Mean $K/\sigma_1$} \\",
            r"\midrule",
        ]

        for model_name in MODEL_NAMES:
            json_path = self.results_dir / "k_calibration" / f"{model_name}_correlation.json"
            if json_path.exists():
                with open(json_path) as f:
                    stats = json.load(f)
                pearson = f"{stats.get('pearson_r', 0):.3f}"
                spearman = f"{stats.get('spearman_r', 0):.3f}"
                pval = f"{stats.get('p_value', 1):.2e}"
                ratio = f"{stats.get('mean_ratio', 0):.3f}"
            else:
                pearson = spearman = pval = ratio = "—"
            lines.append(
                f"{MODEL_LABELS[model_name]} & {pearson} & {spearman} & {pval} & {ratio} \\\\"
            )

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        self._write("\n".join(lines), "table_k_spectral.tex")

    # ------------------------------------------------------------------
    # Generate all tables
    # ------------------------------------------------------------------

    def generate_all(self) -> None:
        """Generate all paper tables."""
        logger.info("Generating all paper tables → %s", self.output_dir)
        self.table_efficacy_main()
        self.table_permutation_invariance()
        self.table_k_spectral_correlation()
        logger.info("All tables generated.")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Generate paper tables")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/plots")
    args = parser.parse_args()
    PaperTables(results_dir=args.results_dir, output_dir=args.output_dir).generate_all()
