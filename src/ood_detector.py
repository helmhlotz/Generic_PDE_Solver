"""Out-of-distribution (OOD) detection for PDE queries.

A ``PDEFeaturizer`` converts a (ParsedPDE, bc_specs) pair into a fixed-length
feature vector.  An ``OODDetector`` loads a *manifest* (built during training)
and checks at inference time whether a new query falls within the training
distribution.

Public API
----------
    from ood_detector import PDEFeaturizer, OODDetector

    # At inference time:
    is_ood, reason = detector.check(parsed_pde, bc_specs)
    if is_ood:
        ...  # fall back to FD

    # At training time (called by trainer.py):
    features = [PDEFeaturizer.featurize(p, bc) for p, bc in training_set]
    OODDetector.build_manifest(np.stack(features), "pretrained_models/fno_manifest.npz")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Feature definition
# ---------------------------------------------------------------------------

# 7 PDE coefficients + 2 RHS stats + 4 walls × (type_enc, val_l2, alpha, beta)
FEATURE_DIM = 25

_WALL_ORDER = ("left", "right", "bottom", "top")

_BC_TYPE_ENC = {"dirichlet": 0.0, "neumann": 1.0, "robin": 2.0}

_FEATURE_NAMES: list[str] = (
    ["g", "a", "b", "c", "d", "e", "f", "rhs_l2", "rhs_max"]
    + [f"{wall}_{k}" for wall in _WALL_ORDER for k in ("type", "val_l2", "alpha", "beta")]
)

assert len(_FEATURE_NAMES) == FEATURE_DIM, "FEATURE_DIM mismatch"


# ---------------------------------------------------------------------------
# Featurizer
# ---------------------------------------------------------------------------

class PDEFeaturizer:
    """Convert a ``(ParsedPDE, bc_specs)`` pair to a fixed-length numpy vector."""

    @staticmethod
    def featurize(
        parsed_pde,
        bc_specs: dict,
        eval_points: int = 16,
    ) -> np.ndarray:
        """Return a ``(FEATURE_DIM,)`` float32 array.

        Parameters
        ----------
        parsed_pde  : ``ParsedPDE`` from ``pde_parser.parse_pde``
        bc_specs    : ``dict[str, ParsedBC]`` from ``pde_parser.parse_bc``
        eval_points : resolution used to numerically summarise RHS / BC values
        """
        feat = np.zeros(FEATURE_DIM, dtype=np.float32)
        idx = 0

        # --- 7 PDE coefficients ---
        for attr in ("g", "a", "b", "c", "d", "e", "f"):
            feat[idx] = float(getattr(parsed_pde, attr, 0.0))
            idx += 1

        # --- RHS statistics (L2, max abs) ---
        s = np.linspace(0.0, 1.0, eval_points)
        xs, ys = np.meshgrid(s, s, indexing="ij")
        try:
            rhs_fn = parsed_pde.rhs_fn()
            rhs_vals = np.asarray(rhs_fn(xs, ys), dtype=np.float32)
            feat[idx]     = float(np.sqrt(np.mean(rhs_vals ** 2)))   # RMS ≈ L2/N
            feat[idx + 1] = float(np.max(np.abs(rhs_vals)))
        except Exception:
            pass
        idx += 2

        # --- Per-wall BC features ---
        s1d = np.linspace(0.0, 1.0, eval_points)
        # For each wall: (x_values, y_values) as Tensors along the wall
        # tangent varies; normal coord is fixed at the wall boundary.
        coords_1d = torch.tensor(s1d, dtype=torch.float32)
        zeros_1d  = torch.zeros(eval_points, dtype=torch.float32)
        ones_1d   = torch.ones(eval_points, dtype=torch.float32)
        _WALL_XY: dict[str, tuple] = {
            "left":   (zeros_1d, coords_1d),
            "right":  (ones_1d,  coords_1d),
            "bottom": (coords_1d, zeros_1d),
            "top":    (coords_1d, ones_1d),
        }
        for wall in _WALL_ORDER:
            spec = bc_specs.get(wall)
            if spec is None:
                idx += 4
                continue

            # BC type encoding
            feat[idx] = _BC_TYPE_ENC.get(getattr(spec, "bc_type", "dirichlet"), 0.0)
            idx += 1

            # Value function L2 along wall
            try:
                val_fn = spec.value_fn()
                xx_w, yy_w = _WALL_XY[wall]
                vals = val_fn(xx_w, yy_w).numpy().astype(np.float32)
                feat[idx] = float(np.sqrt(np.mean(vals ** 2)))
            except Exception:
                pass
            idx += 1

            # Robin coefficients (0 for non-Robin walls)
            feat[idx]     = float(getattr(spec, "alpha", None) or 0.0)
            feat[idx + 1] = float(getattr(spec, "beta",  None) or 0.0)
            idx += 2

        assert idx == FEATURE_DIM
        return feat


# ---------------------------------------------------------------------------
# OOD detector
# ---------------------------------------------------------------------------

class OODDetector:
    """Check whether a new PDE query is within the training distribution.

    The detector applies three checks in order:
    1. Hard structural checks (time-dependent, hyperbolic, degenerate operator)
    2. Bounding-box check: normalised feature outside [-0.25, 1.25]
    3. KNN check: nearest normalised distance to training features > threshold

    Parameters
    ----------
    manifest_path : str | Path
        Path to the ``.npz`` manifest produced by :meth:`build_manifest`.
    """

    def __init__(self, manifest_path: str | Path) -> None:
        data = np.load(str(manifest_path))
        self._features_norm: np.ndarray = data["features_norm"].astype(np.float32)
        self._feat_min: np.ndarray      = data["feat_min"].astype(np.float32)
        self._feat_range: np.ndarray    = data["feat_range"].astype(np.float32)
        self._threshold: float          = float(data["threshold"])

    # ------------------------------------------------------------------
    def check(self, parsed_pde, bc_specs: dict) -> tuple[bool, str]:
        """Return ``(is_ood, reason)`` for the given PDE + BCs.

        ``is_ood == False`` means the query is within the supported distribution.
        ``reason`` is an empty string when ``is_ood == False``, otherwise a
        human-readable description of why the query is out-of-distribution.
        """
        # --- Hard structural rules ---
        if getattr(parsed_pde, "is_time_dependent", False):
            return True, "time-dependent PDE is not supported by the FNO"

        pde_class = getattr(parsed_pde, "pde_class", "elliptic")
        if pde_class not in ("elliptic", "parabolic"):
            return True, f"{pde_class} PDE is outside the supported class (elliptic / parabolic)"

        a = float(getattr(parsed_pde, "a", 0.0))
        b = float(getattr(parsed_pde, "b", 0.0))
        if abs(a) < 1e-8 and abs(b) < 1e-8:
            return True, "degenerate operator (a ≈ 0 and b ≈ 0) is not supported"

        # --- Feature-based checks ---
        feat = PDEFeaturizer.featurize(parsed_pde, bc_specs)
        feat_norm = self._normalize(feat)

        # Bounding box: allow small extra-polation margin
        if np.any(feat_norm < -0.25) or np.any(feat_norm > 1.25):
            bad = int(np.argmax((feat_norm < -0.25) | (feat_norm > 1.25)))
            val = float(feat_norm[bad])
            name = _FEATURE_NAMES[bad] if bad < len(_FEATURE_NAMES) else str(bad)
            return True, (
                f"feature '{name}' (normalised value {val:.3g}) is outside the "
                "training bounding box [-0.25, 1.25]"
            )

        # KNN: minimum L2 distance in normalised feature space
        min_dist = self._min_dist(feat_norm)
        if min_dist > self._threshold:
            return True, (
                f"nearest-neighbour distance {min_dist:.4f} exceeds the "
                f"training threshold {self._threshold:.4f} — "
                "PDE may be too different from training distribution"
            )

        return False, ""

    # ------------------------------------------------------------------
    def _normalize(self, feat: np.ndarray) -> np.ndarray:
        """Apply the training-set normalisation to a single feature vector."""
        return (feat - self._feat_min) / (self._feat_range + 1e-8)

    def _min_dist(self, feat_norm: np.ndarray) -> float:
        """Return the L2 distance to the closest training-set sample."""
        diffs = self._features_norm - feat_norm[np.newaxis, :]  # (N, D)
        dists = np.sqrt((diffs ** 2).sum(axis=1))               # (N,)
        return float(dists.min())

    # ------------------------------------------------------------------
    # Called at training time once to persist the manifest
    # ------------------------------------------------------------------
    @classmethod
    def build_manifest(
        cls,
        features: np.ndarray,
        manifest_path: str | Path,
        ood_percentile: float = 95.0,
    ) -> None:
        """Compute normalisation stats and KNN threshold; save as ``.npz``.

        Parameters
        ----------
        features      : (N, D) array of training-set feature vectors
        manifest_path : output path for the ``.npz`` file
        ood_percentile: leave-one-out KNN distances above this percentile are
                        treated as out-of-distribution at inference time
        """
        features = np.asarray(features, dtype=np.float32)
        n = len(features)

        feat_min   = features.min(axis=0)
        feat_range = features.max(axis=0) - features.min(axis=0)
        feat_range[feat_range < 1e-8] = 1.0   # avoid divide-by-zero for constant dims

        features_norm = (features - feat_min) / feat_range

        # Leave-one-out nearest-neighbour distances (O(N²D) but fast in numpy)
        nn_dists = np.zeros(n, dtype=np.float32)
        for i in range(n):
            diffs = features_norm - features_norm[i : i + 1]  # (N, D)
            dists = np.sqrt((diffs ** 2).sum(axis=1))          # (N,)
            dists[i] = np.inf                                   # exclude self
            nn_dists[i] = float(dists.min())

        threshold = float(np.percentile(nn_dists, ood_percentile))

        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(manifest_path),
            features_norm=features_norm,
            feat_min=feat_min,
            feat_range=feat_range,
            threshold=threshold,
        )
        print(
            f"OOD manifest saved to {manifest_path} "
            f"({n} samples, threshold={threshold:.4f} @ {ood_percentile}th pct)"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def is_available(manifest_path: str | Path) -> bool:
        """Return True if the manifest file exists and can be loaded."""
        p = Path(manifest_path)
        if not p.exists():
            return False
        try:
            np.load(str(p))
            return True
        except Exception:
            return False


