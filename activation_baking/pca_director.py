"""
PCADirector: Extracts and applies PCA-based behavioral directions from contrastive activation diffs.

Implements principal-angle subspace similarity for permutation invariance experiments,
and provides calibrated activation steering via the fitted PCA components.
"""

import gc
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


@dataclass
class BehavioralDirections:
    """
    Stores PCA-derived behavioral directions for a single model layer.

    Attributes:
        layer_name: Fully qualified module path (e.g. 'model.layers.16').
        components: Unit-norm principal directions, shape [n_components, hidden_size].
        explained_variance_ratio: Fraction of variance each component explains,
            shape [n_components].
        mean_diff: Mean of all contrastive diffs at this layer, shape [hidden_size].
        n_pairs_fit: Number of contrastive pairs used to fit the PCA.
        k_value: Calibrated magnitude scalar; populated after KCalibrator runs.
    """

    layer_name: str
    components: torch.Tensor
    explained_variance_ratio: np.ndarray
    mean_diff: torch.Tensor
    n_pairs_fit: int
    k_value: Optional[float] = None


class PCADirector:
    """
    Fits PCA-based behavioral directions from contrastive activation differences and
    applies them as targeted steering interventions during model inference.

    Methods
    -------
    fit : Extract principal directions per layer from contrastive diffs.
    set_k_values : Attach calibrated K scalars to fitted directions (in-place).
    apply_steering : Add a calibrated PCA steering vector to a residual-stream tensor.
    compute_permutation_invariance : Quantify subspace stability across neuron permutations.
    save / load : Serialise and deserialise fitted directions to/from disk.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        activation_diffs: Dict[str, torch.Tensor],
        n_components: int = 5,
    ) -> Dict[str, BehavioralDirections]:
        """
        Fit PCA to contrastive activation differences for each layer.

        For each layer the diffs are mean-centred and then decomposed via
        sklearn's SVD-backed PCA.  The resulting components are L2-normalised
        before being stored so that `apply_steering` arithmetic is clean.

        Parameters
        ----------
        activation_diffs:
            Mapping from layer name to a float tensor of shape
            [n_pairs, hidden_size] containing (positive - negative)
            residual-stream differences.
        n_components:
            Number of principal components to retain per layer.

        Returns
        -------
        Dict[str, BehavioralDirections]
            Fitted directions keyed by layer name.

        Raises
        ------
        TypeError
            If `activation_diffs` is not a dict or values are not torch.Tensor.
        ValueError
            If any diff tensor has fewer than `n_components` rows, or does not
            have exactly 2 dimensions.
        """
        if not isinstance(activation_diffs, dict):
            raise TypeError(
                f"activation_diffs must be a dict, got {type(activation_diffs)}"
            )
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")

        directions: Dict[str, BehavioralDirections] = {}

        for layer_name, diffs in activation_diffs.items():
            if not isinstance(diffs, torch.Tensor):
                raise TypeError(
                    f"Layer '{layer_name}': expected torch.Tensor, got {type(diffs)}"
                )
            if diffs.ndim != 2:
                raise ValueError(
                    f"Layer '{layer_name}': diffs must be 2-D [n_pairs, hidden_size], "
                    f"got shape {tuple(diffs.shape)}"
                )
            n_pairs, hidden_size = diffs.shape
            effective_components = min(n_components, n_pairs)
            if effective_components < n_components:
                self._logger.warning(
                    "Layer '%s': requested %d components but only %d pairs available; "
                    "clamping to %d.",
                    layer_name,
                    n_components,
                    n_pairs,
                    effective_components,
                )

            # Move to CPU + float32 for sklearn
            diffs_cpu: np.ndarray = diffs.detach().float().cpu().numpy()

            mean_diff_np: np.ndarray = diffs_cpu.mean(axis=0)
            centred: np.ndarray = diffs_cpu - mean_diff_np

            pca = PCA(n_components=effective_components, svd_solver="full")
            pca.fit(centred)

            # components_ is already unit-norm in sklearn, but we re-normalise
            # defensively to guard against any future sklearn changes.
            raw_components: np.ndarray = pca.components_  # [k, hidden_size]
            norms: np.ndarray = np.linalg.norm(raw_components, axis=1, keepdims=True)
            unit_components: np.ndarray = raw_components / np.clip(norms, 1e-12, None)

            directions[layer_name] = BehavioralDirections(
                layer_name=layer_name,
                components=torch.from_numpy(unit_components).float(),
                explained_variance_ratio=pca.explained_variance_ratio_.copy(),
                mean_diff=torch.from_numpy(mean_diff_np).float(),
                n_pairs_fit=n_pairs,
            )

            self._logger.debug(
                "Layer '%s': fitted %d components (EVR sum=%.4f) from %d pairs.",
                layer_name,
                effective_components,
                pca.explained_variance_ratio_.sum(),
                n_pairs,
            )

        gc.collect()
        self._logger.info("PCADirector.fit complete: %d layers.", len(directions))
        return directions

    # ------------------------------------------------------------------
    # K-value attachment
    # ------------------------------------------------------------------

    def set_k_values(
        self,
        directions: Dict[str, "BehavioralDirections"],
        k_values: Dict[str, float],
    ) -> None:
        """
        Attach calibrated K scalars to fitted BehavioralDirections in-place.

        Parameters
        ----------
        directions:
            Output of :py:meth:`fit`.
        k_values:
            Mapping from layer name to calibrated K scalar.  Keys not present
            in `directions` are silently ignored.

        Raises
        ------
        TypeError
            If any value in `k_values` is not a real number.
        """
        for layer_name, k in k_values.items():
            if not isinstance(k, (int, float)):
                raise TypeError(
                    f"k_values['{layer_name}'] must be numeric, got {type(k)}"
                )
            if layer_name in directions:
                directions[layer_name].k_value = float(k)
            else:
                self._logger.warning(
                    "set_k_values: layer '%s' not found in directions; skipping.",
                    layer_name,
                )

    # ------------------------------------------------------------------
    # Steering
    # ------------------------------------------------------------------

    def apply_steering(
        self,
        activations: torch.Tensor,
        directions: "BehavioralDirections",
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        Add a calibrated PCA steering vector to residual-stream activations.

        Intervention formula::

            steered = act + alpha * k * Σ_i [ (mean_diff · c_i) * c_i ]

        where ``c_i`` are unit principal components and
        ``mean_diff · c_i`` aligns the perturbation with the contrastive
        direction of the training pairs.

        Parameters
        ----------
        activations:
            Residual-stream tensor of shape ``[batch, hidden_size]`` or
            ``[hidden_size]``.  The device of this tensor determines where
            all intermediate computation runs.
        directions:
            Fitted :class:`BehavioralDirections` for the target layer.
        alpha:
            Global scaling factor applied on top of the calibrated K value.

        Returns
        -------
        torch.Tensor
            Steered activations; same shape and device as input.

        Raises
        ------
        TypeError
            If `activations` is not a torch.Tensor.
        ValueError
            If hidden-size dimension of activations does not match components.
        RuntimeError
            If `directions.k_value` has not been set (is None).
        """
        if not isinstance(activations, torch.Tensor):
            raise TypeError(
                f"activations must be a torch.Tensor, got {type(activations)}"
            )
        if directions.k_value is None:
            raise RuntimeError(
                f"Layer '{directions.layer_name}': k_value is None. "
                "Call set_k_values() before apply_steering()."
            )

        squeeze = activations.ndim == 1
        act = activations.unsqueeze(0) if squeeze else activations  # [batch, hidden]

        hidden_size = act.shape[-1]
        expected_hidden = directions.components.shape[-1]
        if hidden_size != expected_hidden:
            raise ValueError(
                f"activations hidden_size={hidden_size} does not match "
                f"components hidden_size={expected_hidden} for layer "
                f"'{directions.layer_name}'."
            )

        device = act.device
        dtype = act.dtype

        # Move steering tensors to the same device/dtype as activations (lazy move,
        # no persistent state mutation — keeps Baker's hook thread-safe).
        components = directions.components.to(device=device, dtype=dtype)  # [k, h]
        mean_diff = directions.mean_diff.to(device=device, dtype=dtype)    # [h]

        # Projection weights: scalar alignment of mean_diff with each component.
        # Shape: [k]  (purely vectorised, no Python loop)
        projection_weights = torch.mv(components, mean_diff)  # [k]

        # Weighted sum of components → steering vector [hidden_size]
        steering_vector = torch.mv(
            components.T, projection_weights
        )  # [h] = components^T @ weights

        # Scale and broadcast
        scaled_vector = (alpha * directions.k_value * steering_vector).unsqueeze(0)  # [1, h]
        steered = act + scaled_vector

        if squeeze:
            steered = steered.squeeze(0)
        return steered

    # ------------------------------------------------------------------
    # Permutation invariance
    # ------------------------------------------------------------------

    def compute_permutation_invariance(
        self,
        dirs_original: Dict[str, "BehavioralDirections"],
        dirs_permuted: Dict[str, "BehavioralDirections"],
    ) -> Dict[str, float]:
        """
        Compute subspace similarity between two sets of PCA directions via principal angles.

        The similarity metric is the mean cosine of the principal angles between
        the two subspaces at each layer.  A value of 1.0 indicates identical
        subspaces; 0.0 indicates orthogonal subspaces.

        Algorithm::

            M = A.components @ B.components.T   # [k_a, k_b]
            singular_values = svd(M).S           # principal cosines (clamped to [0,1])
            score = mean(singular_values)

        Parameters
        ----------
        dirs_original:
            Directions fitted on the original (un-permuted) model.
        dirs_permuted:
            Directions fitted on the neuron-permuted model.

        Returns
        -------
        Dict[str, float]
            Mean cosine of principal angles per layer (0.0 – 1.0).

        Raises
        ------
        ValueError
            If a layer present in `dirs_original` is missing from `dirs_permuted`.
        """
        scores: Dict[str, float] = {}
        shared_layers = set(dirs_original.keys()) & set(dirs_permuted.keys())
        missing = set(dirs_original.keys()) - set(dirs_permuted.keys())
        if missing:
            self._logger.warning(
                "compute_permutation_invariance: %d layers in dirs_original are "
                "absent from dirs_permuted and will be skipped: %s",
                len(missing),
                missing,
            )

        for layer_name in shared_layers:
            comp_a = dirs_original[layer_name].components.float()   # [k_a, h]
            comp_b = dirs_permuted[layer_name].components.float()   # [k_b, h]

            # Cross-Gram matrix
            cross_gram: torch.Tensor = comp_a @ comp_b.T  # [k_a, k_b]

            # Singular values are the cosines of the principal angles
            singular_values = torch.linalg.svdvals(cross_gram)  # [min(k_a, k_b)]
            singular_values = singular_values.clamp(0.0, 1.0)

            scores[layer_name] = singular_values.mean().item()

        gc.collect()
        return scores

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(
        self,
        directions: Dict[str, "BehavioralDirections"],
        path: str,
    ) -> None:
        """
        Serialise fitted directions to disk as a pickle file.

        Parameters
        ----------
        directions:
            Output of :py:meth:`fit` (with or without k_values attached).
        path:
            Destination file path.  Parent directories are created automatically.

        Raises
        ------
        TypeError
            If `directions` is not a dict.
        """
        if not isinstance(directions, dict):
            raise TypeError(
                f"directions must be a dict, got {type(directions)}"
            )
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Move all tensors to CPU before pickling for portability
        portable: Dict[str, BehavioralDirections] = {}
        for layer_name, bd in directions.items():
            portable[layer_name] = BehavioralDirections(
                layer_name=bd.layer_name,
                components=bd.components.cpu(),
                explained_variance_ratio=bd.explained_variance_ratio.copy(),
                mean_diff=bd.mean_diff.cpu(),
                n_pairs_fit=bd.n_pairs_fit,
                k_value=bd.k_value,
            )

        with out_path.open("wb") as fh:
            pickle.dump(portable, fh, protocol=pickle.HIGHEST_PROTOCOL)

        self._logger.info("Directions saved to '%s' (%d layers).", out_path, len(portable))

    @classmethod
    def load(cls, path: str) -> Dict[str, "BehavioralDirections"]:
        """
        Deserialise fitted directions from disk.

        Parameters
        ----------
        path:
            Path to the pickle file written by :py:meth:`save`.

        Returns
        -------
        Dict[str, BehavioralDirections]

        Raises
        ------
        FileNotFoundError
            If the file does not exist at `path`.
        """
        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"Directions file not found: '{in_path}'")

        with in_path.open("rb") as fh:
            directions: Dict[str, BehavioralDirections] = pickle.load(fh)

        logger.info(
            "Directions loaded from '%s' (%d layers).", in_path, len(directions)
        )
        return directions
