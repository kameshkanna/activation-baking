"""
experiments/06_weight_space_alignment.py

Weight-Space Alignment Experiment — empirical proof that PCA behavioral
directions lie in the same subspace as the top singular vectors of transformer
weight matrices.

For each layer in the target model the script:
  1. Loads the MLP down_proj weight matrix W_d ∈ R^{hidden × intermediate}.
  2. Computes the top-k right singular vectors via truncated SVD.
  3. Measures absolute cosine similarity between every PCA component and every
     singular vector → alignment[i, j] = |dot(c_i, v_j)|.
  4. Repeats steps 2-3 with random unit vectors as a null baseline.
  5. Records the mean-max alignment (per component, maxed over singular vectors,
     then averaged over components) alongside the random baseline.

Outputs:
  results/weight_alignment/{model}/{behavior}/alignment_per_layer.csv
    Columns: layer_idx, layer_name, mean_max_alignment,
             random_baseline_alignment, alignment_ratio

  results/weight_alignment/{model}/{behavior}/alignment_matrix.pt
    Full float32 tensor of shape [n_layers, n_components, top_k]
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from activation_baking.model_utils import ModelInfo, detect_model_info

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("06_weight_space_alignment")


# ---------------------------------------------------------------------------
# SVD helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def top_k_right_singular_vectors(
    weight: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Compute the top-k right singular vectors of a weight matrix.

    Uses ``torch.linalg.svd`` with ``full_matrices=False`` (economy SVD) so the
    decomposition is W = U @ diag(S) @ Vh where Vh is already ``(min_rank, d)``.
    We return the first ``k`` rows of Vh (each row is a right singular vector in
    R^{hidden}).

    Args:
        weight: 2-D weight matrix of shape ``[out_features, in_features]``.
            For MLP down_proj this is ``[hidden, intermediate]``.
        k: Number of top singular vectors to retain.

    Returns:
        Float32 tensor of shape ``[k, in_features]`` — the top-k right singular
        vectors (already L2-normalised because Vh is unitary).

    Raises:
        ValueError: If ``k`` exceeds ``min(weight.shape)``.
    """
    if weight.dim() != 2:
        raise ValueError(
            f"Expected a 2-D weight matrix, got shape {tuple(weight.shape)}."
        )
    max_rank = min(weight.shape)
    if k > max_rank:
        raise ValueError(
            f"k={k} exceeds matrix rank {max_rank} for shape {tuple(weight.shape)}."
        )

    # Cast to float32 for numerical stability; SVD on fp16/bf16 can diverge.
    W = weight.float()
    # Economy SVD: Vh has shape [min_rank, in_features]
    _, _, Vh = torch.linalg.svd(W, full_matrices=False)
    return Vh[:k]  # [k, in_features]


@torch.no_grad()
def random_unit_vectors(
    k: int,
    dim: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample k random unit vectors in R^{dim}.

    Args:
        k: Number of vectors.
        dim: Dimensionality.
        device: Target device.
        generator: Optional torch.Generator for reproducibility.

    Returns:
        Float32 tensor of shape ``[k, dim]``.
    """
    raw = torch.randn(k, dim, generator=generator, device=device)
    return F.normalize(raw, dim=-1)


@torch.no_grad()
def compute_alignment_matrix(
    directions: torch.Tensor,
    reference_vectors: torch.Tensor,
) -> torch.Tensor:
    """Compute absolute cosine similarity between directions and reference vectors.

    Because PCA components and singular vectors are sign-ambiguous, we take the
    absolute value of the dot product (both tensors are unit-normalised).

    Args:
        directions: Float32 tensor of shape ``[n_components, hidden]``.
        reference_vectors: Float32 tensor of shape ``[k, hidden]``.

    Returns:
        Float32 tensor of shape ``[n_components, k]`` where entry ``[i, j]`` is
        ``|dot(c_i, v_j)|``.
    """
    # Both tensors are unit vectors so dot product == cosine similarity.
    directions_n = F.normalize(directions.float(), dim=-1)       # [n, H]
    ref_n = F.normalize(reference_vectors.float(), dim=-1)        # [k, H]
    sim = directions_n @ ref_n.T                                   # [n, k]
    return sim.abs()


@torch.no_grad()
def mean_max_alignment(alignment: torch.Tensor) -> float:
    """Return the mean-max alignment score.

    For each direction (row), take the max over reference vectors (columns), then
    average over all directions.

    Args:
        alignment: ``[n_components, k]`` absolute cosine similarity matrix.

    Returns:
        Scalar float in ``[0, 1]``.
    """
    return alignment.max(dim=-1).values.mean().item()


# ---------------------------------------------------------------------------
# Module navigation
# ---------------------------------------------------------------------------

def _get_weight_matrix(
    model: AutoModelForCausalLM,
    layer_idx: int,
    model_info: ModelInfo,
) -> Tuple[torch.Tensor, str]:
    """Fetch the MLP down_proj weight matrix for a specific layer.

    Args:
        model: Loaded HuggingFace causal LM.
        layer_idx: 0-based layer index.
        model_info: Populated ModelInfo.

    Returns:
        Tuple of (weight_tensor, module_name_string).

    Raises:
        AttributeError: If the module path cannot be resolved.
    """
    proj_name = model_info.mlp_down_proj_names[layer_idx]
    parts = proj_name.split(".")
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]  # type: ignore[index]
        else:
            module = getattr(module, part)

    weight: torch.Tensor = module.weight.detach().cpu()  # type: ignore[attr-defined]
    return weight, proj_name


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

def run_weight_alignment_experiment(
    model_id: str,
    behavior: str,
    device: torch.device,
    output_dir: Path,
    results_dir: Path,
    top_k: int,
    seed: int,
) -> None:
    """Run weight-space alignment analysis for one model × behavior pair.

    Args:
        model_id: HuggingFace model identifier.
        behavior: Behavior name corresponding to saved PCA directions.
        device: Torch device for model loading.
        output_dir: Root directory for output artefacts.
        results_dir: Root results directory where PCA directions are stored.
        top_k: Number of top singular vectors to compare against.
        seed: Random seed for the random baseline.

    Raises:
        FileNotFoundError: If the PCA directions file is absent.
    """
    torch.manual_seed(seed)
    model_slug = model_id.replace("/", "__")
    save_dir = output_dir / model_slug / behavior
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load PCA directions
    # ------------------------------------------------------------------
    directions_path = (
        results_dir / "pca_directions" / model_slug / behavior / "directions.pt"
    )
    if not directions_path.exists():
        raise FileNotFoundError(
            f"PCA directions not found at {directions_path}. "
            "Run experiment 04 (contrastive extraction) first."
        )

    directions_payload = torch.load(directions_path, map_location="cpu")
    # Expected format: dict[layer_idx -> Tensor[n_components, hidden]]
    # OR a single Tensor [n_layers, n_components, hidden].
    if isinstance(directions_payload, dict):
        layer_indices_with_dirs = sorted(directions_payload.keys())
        directions_by_layer: dict = directions_payload
    elif isinstance(directions_payload, torch.Tensor) and directions_payload.dim() == 3:
        directions_by_layer = {
            i: directions_payload[i] for i in range(directions_payload.shape[0])
        }
        layer_indices_with_dirs = list(range(directions_payload.shape[0]))
    else:
        raise ValueError(
            f"Unexpected format for directions.pt: {type(directions_payload)}. "
            "Expected dict[int -> Tensor] or Tensor[n_layers, n_components, hidden]."
        )

    n_components: int = next(iter(directions_by_layer.values())).shape[0]
    hidden_size: int = next(iter(directions_by_layer.values())).shape[1]
    logger.info(
        "Loaded PCA directions: %d layers, %d components, hidden=%d",
        len(layer_indices_with_dirs),
        n_components,
        hidden_size,
    )

    # ------------------------------------------------------------------
    # Load model (weights only; no gradient tracking needed)
    # ------------------------------------------------------------------
    logger.info("Loading model '%s' for weight matrix extraction", model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",  # keep on CPU to save VRAM — SVD is CPU-bound
        low_cpu_mem_usage=True,
    )
    model.eval()
    model_info = detect_model_info(model, model_id)
    logger.info(
        "Model loaded: %d layers, hidden=%d", model_info.num_layers, model_info.hidden_size
    )

    # ------------------------------------------------------------------
    # Per-layer alignment
    # ------------------------------------------------------------------
    generator = torch.Generator()
    generator.manual_seed(seed)

    all_alignment_rows: List[dict] = []
    # Full alignment tensor: [n_layers, n_components, top_k]
    # We only store layers that have saved directions.
    n_valid_layers = len(layer_indices_with_dirs)
    full_alignment = torch.zeros(n_valid_layers, n_components, top_k)

    for tensor_idx, layer_idx in tqdm(
        enumerate(layer_indices_with_dirs),
        total=n_valid_layers,
        desc="Layers",
        dynamic_ncols=True,
    ):
        if layer_idx >= model_info.num_layers:
            logger.warning(
                "Layer index %d exceeds model depth %d — skipping.",
                layer_idx,
                model_info.num_layers,
            )
            continue

        pca_dirs: torch.Tensor = directions_by_layer[layer_idx]  # [n_components, H]
        layer_name = model_info.mlp_down_proj_names[layer_idx]

        try:
            W, _ = _get_weight_matrix(model, layer_idx, model_info)
        except (AttributeError, IndexError) as exc:
            logger.warning("Could not load weight for layer %d: %s", layer_idx, exc)
            continue

        # Adjust top_k to actual matrix rank
        actual_k = min(top_k, min(W.shape))

        try:
            singular_vecs = top_k_right_singular_vectors(W, actual_k)  # [k, H_in]
        except (torch.linalg.LinAlgError, ValueError) as exc:
            logger.warning("SVD failed for layer %d: %s", layer_idx, exc)
            del W
            gc.collect()
            continue

        # Singular vectors may have a different dim than PCA (if W_in != hidden).
        # The MLP down_proj has W: [hidden, intermediate] so right singular vectors
        # live in R^{intermediate}.  We need hidden-dim vectors.  For alignment we
        # use left singular vectors (U rows), which live in R^{hidden}.
        # Recompute using left singular vectors when dimensions mismatch.
        if singular_vecs.shape[-1] != hidden_size:
            logger.debug(
                "Dimension mismatch: singular_vecs.dim=%d vs hidden=%d — "
                "using left singular vectors instead.",
                singular_vecs.shape[-1],
                hidden_size,
            )
            W_f = W.float()
            U, _, _ = torch.linalg.svd(W_f, full_matrices=False)
            singular_vecs = U[:, :actual_k].T  # [k, hidden]

        alignment = compute_alignment_matrix(pca_dirs, singular_vecs)  # [n_comp, k]

        # Pad to top_k if actual_k < top_k
        if actual_k < top_k:
            pad = torch.zeros(n_components, top_k - actual_k)
            alignment = torch.cat([alignment, pad], dim=-1)

        full_alignment[tensor_idx] = alignment

        mma = mean_max_alignment(alignment)

        # Random baseline
        rand_vecs = random_unit_vectors(actual_k, hidden_size, device=torch.device("cpu"), generator=generator)
        rand_alignment = compute_alignment_matrix(pca_dirs, rand_vecs)
        rand_mma = mean_max_alignment(rand_alignment)

        alignment_ratio = mma / (rand_mma + 1e-9)

        all_alignment_rows.append(
            {
                "layer_idx": layer_idx,
                "layer_name": layer_name,
                "mean_max_alignment": mma,
                "random_baseline_alignment": rand_mma,
                "alignment_ratio": alignment_ratio,
            }
        )

        del W, singular_vecs, alignment, rand_vecs, rand_alignment
        gc.collect()

    # ------------------------------------------------------------------
    # Persist results
    # ------------------------------------------------------------------
    alignment_csv_path = save_dir / "alignment_per_layer.csv"
    alignment_pt_path = save_dir / "alignment_matrix.pt"

    df = pd.DataFrame(all_alignment_rows)
    df.to_csv(alignment_csv_path, index=False)
    torch.save(full_alignment, alignment_pt_path)

    logger.info("Saved per-layer alignment table → %s", alignment_csv_path)
    logger.info("Saved full alignment tensor    → %s", alignment_pt_path)

    if not df.empty:
        logger.info(
            "Summary — mean_max_alignment: %.4f | random_baseline: %.4f | ratio: %.2f",
            df["mean_max_alignment"].mean(),
            df["random_baseline_alignment"].mean(),
            df["alignment_ratio"].mean(),
        )

    # Clean up model from memory
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 06: Weight-Space Alignment — show PCA directions align "
            "with top singular vectors of MLP weight matrices."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model identifier.",
    )
    parser.add_argument(
        "--behavior",
        type=str,
        required=True,
        help="Behavior name (must have saved PCA directions under results/pca_directions/).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/weight_alignment"),
        help="Root directory for output artefacts.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root results directory (contains pca_directions/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the null-baseline generator.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top singular vectors to compare against.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for experiment 06."""
    args = _parse_args()
    device = torch.device(args.device)

    run_weight_alignment_experiment(
        model_id=args.model,
        behavior=args.behavior,
        device=device,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        top_k=args.top_k,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
