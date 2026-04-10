"""PDE parameter space definition and Latin Hypercube Sampling utilities.

This module defines the distribution of PDE problems used for training the
conditional FNO.  All randomness is routed through explicit np.random.Generator
objects so that experiments are fully reproducible.

Public API
----------
    from pde_space import PDESpaceConfig, LHSSampler, BCGenerator, SourceGenerator

    sampler = LHSSampler(PDESpaceConfig())
    problems = sampler.generate(n_samples=2000, seed=42)
    # Each element: {"pde_str": ..., "bc_dict": ...}
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PDESpaceConfig:
    """Hyper-parameters for the distribution of training PDE problems."""

    # Elliptic operator coefficients for u_xx and u_yy
    a_range: tuple[float, float] = (0.5, 2.0)
    b_range: tuple[float, float] = (0.5, 2.0)

    # Helmholtz decay term: f*u added to LHS; kept ≤ 0 to preserve ellipticity
    helm_f_range: tuple[float, float] = (-1.0, 0.0)
    helm_f_prob: float = 0.20          # probability of including the f*u term

    # Amplitude ranges for source term and BC values
    src_amplitude_range: tuple[float, float] = (-3.0, 3.0)
    bc_amplitude_range: tuple[float, float] = (-2.0, 2.0)

    # Fraction of walls assigned Neumann or Robin BCs; remainder are Dirichlet
    neumann_prob: float = 0.15
    robin_prob: float = 0.10

    # Spatial wavenumbers used in BC and source expressions
    k_values: list[int] = field(default_factory=lambda: [1, 2, 3])


# ---------------------------------------------------------------------------
# Latin Hypercube Sampling
# ---------------------------------------------------------------------------

def lhs(n_samples: int, n_dims: int, seed: int = 42) -> np.ndarray:
    """Return an (n_samples, n_dims) Latin Hypercube Sample in [0, 1]^n_dims.

    Each column contains exactly one sample per stratum [k/n, (k+1)/n) with a
    uniform random jitter within the stratum.
    """
    rng = np.random.default_rng(seed)
    result = np.empty((n_samples, n_dims))
    for d in range(n_dims):
        perm = rng.permutation(n_samples)
        result[:, d] = (perm + rng.uniform(size=n_samples)) / n_samples
    return result


# ---------------------------------------------------------------------------
# BC generator
# ---------------------------------------------------------------------------

class BCGenerator:
    """Sample random boundary-condition configurations.

    All randomness is taken from an explicit ``np.random.Generator`` so that
    the full training run can be reproduced from a single seed.
    """

    # Tangent coordinate for each wall
    _WALL_TANGENT: dict[str, str] = {
        "left": "y", "right": "y",
        "top": "x",  "bottom": "x",
    }

    @staticmethod
    def _random_expr(
        tangent: str,
        amplitude: float,
        k_values: list[int],
        rng: np.random.Generator,
    ) -> str:
        """Return a symbolic expression string for a BC value along one wall."""
        r = float(rng.uniform())
        a = round(float(amplitude), 3)
        if r < 0.30:
            return "0"
        elif r < 0.55:
            k = int(rng.choice(k_values))
            return f"{a}*sin({k}*pi*{tangent})"
        elif r < 0.75:
            k = int(rng.choice(k_values))
            return f"{a}*cos({k}*pi*{tangent})"
        elif r < 0.90:
            return f"{a}"   # constant
        else:
            c = round(float(rng.uniform(-1.0, 1.0)), 3)
            return f"{a}*{tangent}**2 + {c}*{tangent}"

    @classmethod
    def sample(
        cls,
        amplitude_scale: float = 1.0,
        rng: np.random.Generator | None = None,
        k_values: list[int] | None = None,
        neumann_prob: float = 0.15,
        robin_prob: float = 0.10,
    ) -> dict[str, dict]:
        """Return a ``bc_dict`` suitable for ``pde_parser.parse_bc``."""
        if rng is None:
            rng = np.random.default_rng()
        if k_values is None:
            k_values = [1, 2, 3]

        bc_dict: dict[str, dict] = {}
        for wall in ["left", "right", "top", "bottom"]:
            tangent = cls._WALL_TANGENT[wall]
            val_expr = cls._random_expr(tangent, amplitude_scale, k_values, rng)
            r = float(rng.uniform())
            if r < neumann_prob:
                bc_dict[wall] = {"type": "neumann", "value": val_expr}
            elif r < neumann_prob + robin_prob:
                alpha = round(float(rng.uniform(0.5, 2.0)), 2)
                beta  = round(float(rng.uniform(0.1, 1.0)), 2)
                bc_dict[wall] = {
                    "type": "robin",
                    "value": val_expr,
                    "alpha": str(alpha),
                    "beta":  str(beta),
                }
            else:
                bc_dict[wall] = {"type": "dirichlet", "value": val_expr}
        return bc_dict


# ---------------------------------------------------------------------------
# Source term generator
# ---------------------------------------------------------------------------

class SourceGenerator:
    """Sample random PDE right-hand-side (source) expressions."""

    _TEMPLATES = [
        "0",
        "{a}*sin({k1}*pi*x)*sin({k2}*pi*y)",
        "{a}*cos({k1}*pi*x)*cos({k2}*pi*y)",
        "{a}*sin({k1}*pi*x)",
        "{a}*cos({k2}*pi*y)",
        "{a}*x*(1-x)*y*(1-y)",
        "{a}",
        "{a}*exp(-{b}*((x-0.5)**2 + (y-0.5)**2))",
    ]

    @classmethod
    def sample(
        cls,
        amplitude: float = 1.0,
        rng: np.random.Generator | None = None,
        k_values: list[int] | None = None,
    ) -> str:
        """Return a symbolic string for f(x,y)."""
        if rng is None:
            rng = np.random.default_rng()
        if k_values is None:
            k_values = [1, 2, 3]

        template = cls._TEMPLATES[int(rng.integers(len(cls._TEMPLATES)))]
        a  = round(float(amplitude), 3)
        b  = round(float(rng.uniform(2.0, 8.0)), 2)
        k1 = int(rng.choice(k_values))
        k2 = int(rng.choice(k_values))
        return template.format(a=a, b=b, k1=k1, k2=k2)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lerp(t: float, lo: float, hi: float) -> float:
    """Linearly interpolate t ∈ [0,1] to [lo, hi]."""
    return lo + t * (hi - lo)


def _build_pde_str(a: float, b: float, helm_f: float, src_str: str) -> str:
    """Build a string like ``'1.2*u_xx + 0.8*u_yy - 0.3*u = 0.5*sin(pi*x)'``.

    Parameters
    ----------
    a, b     : coefficients for u_xx and u_yy
    helm_f   : coefficient for the u term (≤ 0 for ellipticity); 0 omits it
    src_str  : right-hand-side expression string
    """
    tol = 1e-8
    terms: list[str] = []

    # u_xx term
    if abs(a - 1.0) < tol:
        terms.append("u_xx")
    else:
        terms.append(f"{a:.4g}*u_xx")

    # u_yy term
    if abs(b - 1.0) < tol:
        terms.append("u_yy")
    else:
        terms.append(f"{b:.4g}*u_yy")

    # Optional Helmholtz or decay term
    if abs(helm_f) > tol:
        terms.append(f"{helm_f:.4g}*u")

    lhs = " + ".join(terms).replace("+ -", "- ")
    return f"{lhs} = {src_str}"


# ---------------------------------------------------------------------------
# LHS sampler — combines the pieces above into a dataset generator
# ---------------------------------------------------------------------------

# Number of LHS dimensions: (a, b, src_amplitude, bc_amplitude, helm_f)
_LHS_DIM = 5


class LHSSampler:
    """Generate a diverse set of 2-D PDE training problems via LHS.

    The 5-D LHS space covers:
      dim 0 → coefficient a for u_xx  (PDESpaceConfig.a_range)
      dim 1 → coefficient b for u_yy  (PDESpaceConfig.b_range)
      dim 2 → source amplitude        (PDESpaceConfig.src_amplitude_range)
      dim 3 → BC amplitude            (PDESpaceConfig.bc_amplitude_range)
      dim 4 → Helmholtz f coefficient (PDESpaceConfig.helm_f_range, gated by helm_f_prob)
    """

    def __init__(self, config: PDESpaceConfig | None = None) -> None:
        self.config = config or PDESpaceConfig()

    def generate(self, n_samples: int = 2000, seed: int = 42) -> list[dict]:
        """Return a list of ``{"pde_str": ..., "bc_dict": ...}`` dicts.

        Parameters
        ----------
        n_samples : int   — number of distinct problems to generate
        seed      : int   — master seed; all sub-RNGs are derived from it
        """
        cfg = self.config
        pts = lhs(n_samples, _LHS_DIM, seed=seed)
        # Separate RNG for discrete choices (BC type, template selection)
        rng = np.random.default_rng(seed + 1)

        problems: list[dict] = []
        for i in range(n_samples):
            a       = _lerp(pts[i, 0], *cfg.a_range)
            b       = _lerp(pts[i, 1], *cfg.b_range)
            src_amp = _lerp(pts[i, 2], *cfg.src_amplitude_range)
            bc_amp  = _lerp(pts[i, 3], *cfg.bc_amplitude_range)
            helm_f_raw = _lerp(pts[i, 4], *cfg.helm_f_range)

            # Gate the Helmholtz term stochastically
            helm_f = helm_f_raw if float(rng.uniform()) < cfg.helm_f_prob else 0.0

            src_str = SourceGenerator.sample(
                amplitude=src_amp,
                rng=rng,
                k_values=cfg.k_values,
            )
            bc_dict = BCGenerator.sample(
                amplitude_scale=bc_amp,
                rng=rng,
                k_values=cfg.k_values,
                neumann_prob=cfg.neumann_prob,
                robin_prob=cfg.robin_prob,
            )
            pde_str = _build_pde_str(a, b, helm_f, src_str)
            problems.append({"pde_str": pde_str, "bc_dict": bc_dict})

        return problems
