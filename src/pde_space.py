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
    """Hyper-parameters for the distribution of training PDE problems.

    Elliptic safety notes
    ---------------------
    * a, b > 0 always (u_xx / u_yy diffusion coefficients).
    * helm_f ≤ 0 keeps the zero-order term non-positive, preserving coercivity.
    * |d|, |e| are kept small relative to min(a, b) so that the mesh Péclet
      number  Pe = |d|·h / (2a)  stays < 1 on a 64×64 grid (h ~ 0.016),
      i.e. |d| < 2a/h ~ 64·a.  The default range (−0.5, 0.5) is very
      conservative and can be widened after validation.
    * |c| must satisfy  c² < 4ab  for ellipticity of the cross-derivative term
      u_xy.  With a,b ≥ 0.5 the bound is 4·0.5·0.5 = 1.0, so c_range is kept
      within (−0.8, 0.8) by default.  c_prob = 0.0 disables it until validated.
    """

    # ------------------------------------------------------------------ #
    # Second-order diffusion coefficients (must be > 0)                   #
    # ------------------------------------------------------------------ #
    a_range: tuple[float, float] = (0.5, 2.0)   # u_xx
    b_range: tuple[float, float] = (0.5, 2.0)   # u_yy

    # ------------------------------------------------------------------ #
    # First-order advection / drift coefficients                          #
    # Conservative initial range; widen after confirming solver stability  #
    # ------------------------------------------------------------------ #
    d_range: tuple[float, float] = (-0.5, 0.5)  # u_x — cross-flow drift
    d_prob:  float = 0.25                         # probability of including d*u_x

    e_range: tuple[float, float] = (-0.5, 0.5)  # u_y — axial drift
    e_prob:  float = 0.25                         # probability of including e*u_y

    # ------------------------------------------------------------------ #
    # Optional cross-derivative u_xy (disabled by default until validated)#
    # Keep |c| < sqrt(4ab) ~ 1.0 for ellipticity                         #
    # ------------------------------------------------------------------ #
    c_range: tuple[float, float] = (-0.8, 0.8)
    c_prob:  float = 0.00                         # set > 0 to enable

    # ------------------------------------------------------------------ #
    # Helmholtz / reaction term: f*u, kept ≤ 0 to preserve ellipticity   #
    # ------------------------------------------------------------------ #
    helm_f_range: tuple[float, float] = (-1.0, 0.0)
    helm_f_prob: float = 0.20

    # ------------------------------------------------------------------ #
    # Amplitude ranges for source term and BC values                      #
    # BC amplitude is non-negative because BCGenerator currently applies   #
    # abs(amplitude_scale) for robustness with Robin sampling.            #
    # ------------------------------------------------------------------ #
    src_amplitude_range: tuple[float, float] = (-3.0, 3.0)
    bc_amplitude_range: tuple[float, float] = (0.0, 2.0)

    # ------------------------------------------------------------------ #
    # BC type probabilities                                               #
    # ------------------------------------------------------------------ #
    neumann_prob: float = 0.15
    robin_prob: float = 0.10

    # ------------------------------------------------------------------ #
    # Spatial wavenumbers used in BC and source expressions               #
    # ------------------------------------------------------------------ #
    k_values: list[int] = field(default_factory=lambda: [1, 2, 3])

    # ------------------------------------------------------------------ #
    # Thermal-domain source templates                                     #
    # ------------------------------------------------------------------ #
    thermal_source_prob: float = 0.40  # fraction of samples using a thermal template

    @classmethod
    def thermal_v2(cls) -> "PDESpaceConfig":
        """Return a thermal-focused v2 preset with conservative FD-safe defaults.

        This profile is intended for steady thermal demos and early buoyancy-like
        scalar advection-diffusion experiments where we want:
        - stronger thermal-source coverage
        - non-zero first-order transport terms, but with stable magnitudes
        - no mixed-derivative term until explicitly validated
        """
        return cls(
            a_range=(0.6, 2.0),
            b_range=(0.6, 2.0),
            d_range=(-0.2, 0.2),
            d_prob=0.15,
            e_range=(-0.2, 0.2),
            e_prob=0.15,
            c_range=(-0.6, 0.6),
            c_prob=0.0,
            helm_f_range=(-0.8, 0.0),
            helm_f_prob=0.15,
            src_amplitude_range=(-2.5, 2.5),
            bc_amplitude_range=(0.0, 1.5),
            neumann_prob=0.20,
            robin_prob=0.15,
            k_values=[1, 2, 3],
            thermal_source_prob=0.70,
        )


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

        # Robin BCs require α > 0 for well-posedness. A negative amplitude on
        # the value expression can combine with positive α to create an
        # ill-conditioned system. Always use the absolute value so that the
        # sign of the BC value is embedded in the randomly chosen expression
        # template (e.g. sin/cos oscillate naturally) rather than the scale.
        safe_amplitude = abs(amplitude_scale)

        bc_dict: dict[str, dict] = {}
        for wall in ["left", "right", "top", "bottom"]:
            tangent = cls._WALL_TANGENT[wall]
            val_expr = cls._random_expr(tangent, safe_amplitude, k_values, rng)
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

    # General-purpose templates
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

    # Thermal-domain templates
    # ---------------------------------------------------------------
    # 1. Localized heater — Gaussian hot spot at a random (x0, y0)
    # 2. Stratified heating — source varies only with depth (y)
    # 3. Uniform volumetric source — constant heat generation rate
    # ---------------------------------------------------------------
    _THERMAL_TEMPLATES = [
        # Localized heater: centred at ({x0}, {y0}) with width controlled by {b}
        "{a}*exp(-{b}*((x-{x0})**2 + (y-{y0})**2))",
        # Off-centre heater (lower-left quadrant)
        "{a}*exp(-{b}*((x-{x0l})**2 + (y-{y0l})**2))",
        # Two-spot heater (symmetric sources)
        "{a}*(exp(-{b}*((x-0.25)**2+(y-0.5)**2))+exp(-{b}*((x-0.75)**2+(y-0.5)**2)))",
        # Stratified: sinusoidal variation in y only (layer-by-layer heating)
        "{a}*sin({k2}*pi*y)",
        # Stratified: linear gradient — hotter at the bottom
        "{a}*(1 - y)",
        # Stratified: cosine ramp
        "{a}*cos({k2}*pi*y)",
        # Uniform volumetric source (constant internal heat generation)
        "{a}",
    ]

    @classmethod
    def sample(
        cls,
        amplitude: float = 1.0,
        rng: np.random.Generator | None = None,
        k_values: list[int] | None = None,
        thermal_prob: float = 0.0,
    ) -> str:
        """Return a symbolic string for f(x,y).

        Parameters
        ----------
        amplitude     : scale factor applied to the source amplitude
        rng           : reproducible random generator
        k_values      : candidate spatial wavenumbers
        thermal_prob  : probability of drawing from the thermal-domain templates
                        instead of the generic set
        """
        if rng is None:
            rng = np.random.default_rng()
        if k_values is None:
            k_values = [1, 2, 3]

        a  = round(float(amplitude), 3)
        b  = round(float(rng.uniform(2.0, 8.0)), 2)
        k1 = int(rng.choice(k_values))
        k2 = int(rng.choice(k_values))

        if thermal_prob > 0.0 and float(rng.uniform()) < thermal_prob:
            template = cls._THERMAL_TEMPLATES[
                int(rng.integers(len(cls._THERMAL_TEMPLATES)))
            ]
            # Random heater centres: main spot near domain centre, off-centre in
            # lower-left quadrant
            x0  = round(float(rng.uniform(0.3, 0.7)), 2)
            y0  = round(float(rng.uniform(0.3, 0.7)), 2)
            x0l = round(float(rng.uniform(0.1, 0.4)), 2)
            y0l = round(float(rng.uniform(0.1, 0.4)), 2)
            return template.format(a=a, b=b, k1=k1, k2=k2,
                                   x0=x0, y0=y0, x0l=x0l, y0l=y0l)

        template = cls._TEMPLATES[int(rng.integers(len(cls._TEMPLATES)))]
        return template.format(a=a, b=b, k1=k1, k2=k2)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lerp(t: float, lo: float, hi: float) -> float:
    """Linearly interpolate t ∈ [0,1] to [lo, hi]."""
    return lo + t * (hi - lo)


def _build_pde_str(
    a: float,
    b: float,
    helm_f: float,
    src_str: str,
    d: float = 0.0,
    e: float = 0.0,
    c: float = 0.0,
) -> str:
    """Build a PDE string from operator coefficients and a source expression.

    Emits terms in the order: a*u_xx + b*u_yy [+ c*u_xy] [+ d*u_x] [+ e*u_y]
    [+ helm_f*u] = src_str.  Terms with coefficient 0 are omitted.

    Parameters
    ----------
    a, b     : diffusion coefficients for u_xx and u_yy (must be > 0)
    helm_f   : zero-order reaction coefficient (≤ 0 keeps the operator elliptic)
    src_str  : right-hand-side source expression string
    d        : first-order coefficient for u_x (advection in x); 0 omits it
    e        : first-order coefficient for u_y (advection in y); 0 omits it
    c        : cross-derivative coefficient for u_xy; 0 omits it
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

    # Optional cross-derivative u_xy
    if abs(c) > tol:
        terms.append(f"{c:.4g}*u_xy")

    # Optional first-order terms
    if abs(d) > tol:
        terms.append(f"{d:.4g}*u_x")
    if abs(e) > tol:
        terms.append(f"{e:.4g}*u_y")

    # Optional Helmholtz / reaction term
    if abs(helm_f) > tol:
        terms.append(f"{helm_f:.4g}*u")

    lhs = " + ".join(terms).replace("+ -", "- ")
    return f"{lhs} = {src_str}"


# ---------------------------------------------------------------------------
# LHS sampler — combines the pieces above into a dataset generator
# ---------------------------------------------------------------------------

# Number of LHS dimensions:
#   0 → a          (u_xx diffusion coefficient)
#   1 → b          (u_yy diffusion coefficient)
#   2 → src_amp    (source amplitude)
#   3 → bc_amp     (BC amplitude)
#   4 → helm_f     (Helmholtz/reaction coefficient, gated by helm_f_prob)
#   5 → d_raw      (u_x drift coefficient, gated by d_prob)
#   6 → e_raw      (u_y drift coefficient, gated by e_prob)
_LHS_DIM = 7


class LHSSampler:
    """Generate a diverse set of 2-D PDE training problems via LHS.

    The 7-D LHS space covers:
      dim 0 → coefficient a for u_xx  (PDESpaceConfig.a_range)
      dim 1 → coefficient b for u_yy  (PDESpaceConfig.b_range)
      dim 2 → source amplitude        (PDESpaceConfig.src_amplitude_range)
      dim 3 → BC amplitude            (PDESpaceConfig.bc_amplitude_range)
      dim 4 → Helmholtz f coefficient (PDESpaceConfig.helm_f_range, gated by helm_f_prob)
      dim 5 → d coefficient for u_x   (PDESpaceConfig.d_range, gated by d_prob)
      dim 6 → e coefficient for u_y   (PDESpaceConfig.e_range, gated by e_prob)

    The optional cross-derivative coefficient c (u_xy) is sampled from rng
    directly (not via LHS) and is controlled by ``PDESpaceConfig.c_prob``.
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
        # Separate RNG for discrete choices (BC type, template selection, gates)
        rng = np.random.default_rng(seed + 1)

        problems: list[dict] = []
        for i in range(n_samples):
            a       = _lerp(pts[i, 0], *cfg.a_range)
            b       = _lerp(pts[i, 1], *cfg.b_range)
            src_amp = _lerp(pts[i, 2], *cfg.src_amplitude_range)
            bc_amp  = _lerp(pts[i, 3], *cfg.bc_amplitude_range)
            helm_f_raw = _lerp(pts[i, 4], *cfg.helm_f_range)
            d_raw      = _lerp(pts[i, 5], *cfg.d_range)
            e_raw      = _lerp(pts[i, 6], *cfg.e_range)

            # Gate the Helmholtz term stochastically
            helm_f = helm_f_raw if float(rng.uniform()) < cfg.helm_f_prob else 0.0

            # Gate first-order drift terms; scale check keeps local Péclet < 1
            # on a 64×64 grid (h ≈ 0.016): |d|·h/(2a) < 1  =>  |d| < 125·a.
            # The conservative d_range already satisfies this; the gate simply
            # controls sparsity in the training distribution.
            d = d_raw if float(rng.uniform()) < cfg.d_prob else 0.0
            e = e_raw if float(rng.uniform()) < cfg.e_prob else 0.0

            # Optional cross-derivative (sampled from rng, not LHS)
            if cfg.c_prob > 0.0 and float(rng.uniform()) < cfg.c_prob:
                c = round(float(rng.uniform(*cfg.c_range)), 4)
            else:
                c = 0.0

            src_str = SourceGenerator.sample(
                amplitude=src_amp,
                rng=rng,
                k_values=cfg.k_values,
                thermal_prob=cfg.thermal_source_prob,
            )
            bc_dict = BCGenerator.sample(
                amplitude_scale=bc_amp,
                rng=rng,
                k_values=cfg.k_values,
                neumann_prob=cfg.neumann_prob,
                robin_prob=cfg.robin_prob,
            )
            pde_str = _build_pde_str(a, b, helm_f, src_str, d=d, e=e, c=c)
            problems.append({"pde_str": pde_str, "bc_dict": bc_dict})

        return problems
