"""PDE and boundary-condition parser.

Supported PDE language (sympy-compatible strings)
--------------------------------------------------
Symbols:  x, y, t, u, u_x, u_y, u_xx, u_yy, u_xy, u_t
          plus standard math:  pi, sin, cos, exp, sqrt, abs

Example PDE strings (steady-state)
    "u_xx + u_yy = 0"                       Laplace
    "u_xx + u_yy = sin(pi*x)*cos(pi*y)"     Poisson with sinusoidal source
    "u_xx + u_yy + 4*u = 0"                 Helmholtz k²=4

Example PDE strings (time-dependent)
    "u_t + u_xx + u_yy = 0"                 Heat/diffusion equation
    "u_t - u_xx = sin(pi*x)*exp(-t)"        Heat with source
    "u_tt - u_xx = 0"                        Wave equation (if u_tt supported)

Boundary condition dict (all four walls must be specified)
    {
        "left":   {"type": "dirichlet", "value": "0"},
        "right":  {"type": "dirichlet", "value": "0"},
        "bottom": {"type": "dirichlet", "value": "0"},
        "top":    {"type": "dirichlet", "value": "sin(pi*x)"},
    }
    type options:
        "dirichlet"   ->  u = value
        "neumann"     ->  du/dn = value   (outward normal derivative)
        "robin"       ->  alpha*u + beta*du/dn = value
                          requires extra keys "alpha" and "beta"
"""

from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np
import sympy as sp
import torch

# ---------------------------------------------------------------------------
# Sympy symbols used in user expressions
# ---------------------------------------------------------------------------
_x, _y, _t = sp.symbols("x y t")
_u = sp.Symbol("u")
_u_x = sp.Symbol("u_x")
_u_y = sp.Symbol("u_y")
_u_xx = sp.Symbol("u_xx")
_u_yy = sp.Symbol("u_yy")
_u_xy = sp.Symbol("u_xy")
_u_t = sp.Symbol("u_t")

_ALLOWED_SYMS: dict[str, Any] = {
    "x": _x,
    "y": _y,
    "t": _t,
    "u": _u,
    "u_x": _u_x,
    "u_y": _u_y,
    "u_xx": _u_xx,
    "u_yy": _u_yy,
    "u_xy": _u_xy,
    "u_t": _u_t,
    "pi": sp.pi,
    "sin": sp.sin,
    "cos": sp.cos,
    "exp": sp.exp,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "log": sp.log,
    "tanh": sp.tanh,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
}

# ---------------------------------------------------------------------------
# Parsed representations
# ---------------------------------------------------------------------------

class ParsedPDE:
    """Coefficients of  g*u_t + a*u_xx + b*u_yy + c*u_xy + d*u_x + e*u_y + f*u = rhs(x,y,t)."""

    def __init__(
        self,
        g: float,
        a: float,
        b: float,
        c: float,
        d: float,
        e: float,
        f: float,
        rhs_expr: sp.Expr,
        is_poisson_like: bool,
        pde_class: str,
        discriminant: float | None,
        raw_lhs: str,
        raw_rhs: str,
    ):
        self.g = float(g)          # u_t  coefficient (0 for steady-state PDEs)
        self.a = float(a)          # u_xx coefficient
        self.b = float(b)          # u_yy coefficient
        self.c = float(c)          # u_xy coefficient
        self.d = float(d)          # u_x  coefficient
        self.e = float(e)          # u_y  coefficient
        self.f = float(f)          # u    coefficient
        self.rhs_expr = rhs_expr   # sympy Expr for source f(x,y,t)
        self.is_poisson_like = is_poisson_like
        self.has_time_derivative = (g != 0.0)
        self.has_time_variable = (_t in rhs_expr.free_symbols)
        self.is_time_dependent = self.has_time_derivative or self.has_time_variable
        self.pde_class = pde_class
        self.discriminant = discriminant
        self.raw_lhs = raw_lhs
        self.raw_rhs = raw_rhs

    def rhs_fn(self) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]:
        """Return a function  f(xx, yy, tt=None) -> Tensor  (same shape as xx)."""
        if self.rhs_expr == 0:
            return lambda xx, yy, tt=None: torch.zeros_like(xx)
        fn = sp.lambdify((_x, _y, _t), self.rhs_expr, modules=["numpy"])
        def wrapper(xx: torch.Tensor, yy: torch.Tensor, tt: torch.Tensor | None = None) -> torch.Tensor:
            if tt is None:
                tt = torch.zeros_like(xx)
            arr = fn(xx.detach().cpu().numpy(), yy.detach().cpu().numpy(), tt.detach().cpu().numpy())
            return torch.tensor(arr, dtype=xx.dtype, device=xx.device)
        return wrapper

    def __repr__(self) -> str:
        return (
            f"ParsedPDE(g={self.g}, a={self.a}, b={self.b}, c={self.c}, "
            f"d={self.d}, e={self.e}, f={self.f}, "
            f"rhs={self.rhs_expr}, poisson_like={self.is_poisson_like}, "
            f"time_derivative={self.has_time_derivative}, "
            f"time_variable={self.has_time_variable}, "
            f"time_dependent={self.is_time_dependent}, "
            f"class={self.pde_class!r}, discriminant={self.discriminant})"
        )


def classify_pde(g: float, a: float, b: float, c: float) -> tuple[str, float | None]:
    """Classify PDE by type and return its discriminant.

    For second-order PDEs in two variables with operator
        a*u_xx + c*u_xy + b*u_yy
    the discriminant is
        D = c^2 - 4ab.

    Classification:
    - elliptic: D < 0
    - parabolic: D = 0
    - hyperbolic: D > 0

    For first-order-in-time plus second-order-in-space PDEs, classify as parabolic.
    """
    has_second_order_spatial = any(coeff != 0.0 for coeff in (a, b, c))
    if g != 0.0 and has_second_order_spatial:
        return "parabolic", 0.0
    if not has_second_order_spatial:
        return "other", None

    discriminant = c**2 - 4.0 * a * b
    if np.isclose(discriminant, 0.0):
        return "parabolic", 0.0
    if discriminant < 0.0:
        return "elliptic", discriminant
    return "hyperbolic", discriminant


class ParsedBC:
    """Boundary condition on one wall."""

    def __init__(
        self,
        wall: str,
        bc_type: str,
        value_expr: sp.Expr,
        alpha: float,
        beta: float,
    ):
        self.wall = wall          # "left"|"right"|"top"|"bottom"
        self.bc_type = bc_type    # "dirichlet"|"neumann"|"robin"
        self.value_expr = value_expr
        self.alpha = float(alpha)
        self.beta = float(beta)

    def value_fn(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Return value_fn(xx, yy) -> Tensor."""
        if self.value_expr == 0:
            return lambda xx, yy: torch.zeros_like(xx)
        fn = sp.lambdify((_x, _y), self.value_expr, modules=["numpy"])
        def wrapper(xx: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
            arr = fn(xx.cpu().numpy(), yy.cpu().numpy())
            if np.isscalar(arr):
                return torch.full_like(xx, float(arr))
            return torch.tensor(arr, dtype=xx.dtype, device=xx.device)
        return wrapper

    def __repr__(self) -> str:
        return (
            f"ParsedBC(wall={self.wall!r}, type={self.bc_type!r}, "
            f"alpha={self.alpha}, beta={self.beta}, value={self.value_expr})"
        )


class ParsedIC:
    """Initial condition u(x, y, 0) = f(x, y)."""

    def __init__(self, ic_expr: sp.Expr):
        self.ic_expr = ic_expr  # sympy expression f(x, y)

    def ic_fn(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Return ic_fn(xx, yy) -> Tensor."""
        if self.ic_expr == 0:
            return lambda xx, yy: torch.zeros_like(xx)
        fn = sp.lambdify((_x, _y), self.ic_expr, modules=["numpy"])
        def wrapper(xx: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
            arr = fn(xx.cpu().numpy(), yy.cpu().numpy())
            if np.isscalar(arr):
                return torch.full_like(xx, float(arr))
            return torch.tensor(arr, dtype=xx.dtype, device=xx.device)
        return wrapper

    def __repr__(self) -> str:
        return f"ParsedIC(ic={self.ic_expr})"


# ---------------------------------------------------------------------------
# PDE parser
# ---------------------------------------------------------------------------

def parse_pde(pde_str: str) -> ParsedPDE:
    """Parse a PDE string into a ParsedPDE.

    Parameters
    ----------
    pde_str : str
        String of the form  ``"<LHS> = <RHS>"`` where LHS contains derivative
        symbols and RHS is a function of x, y, and optionally t.
        
        Supports time-dependent PDEs:
            "u_t + u_xx + u_yy = sin(pi*x)"   (heat/diffusion)
            "u_xx + u_yy = 0"                   (steady-state Laplace)

    Returns
    -------
    ParsedPDE
    """
    pde_str = pde_str.strip()

    if "=" not in pde_str:
        raise ValueError(f"PDE string must contain '=': {pde_str!r}")

    eq_idx = pde_str.index("=")
    lhs_str = pde_str[:eq_idx].strip()
    rhs_str = pde_str[eq_idx + 1:].strip()

    lhs_expr = _parse_expr(lhs_str)
    rhs_expr = _parse_expr(rhs_str)

    # Move everything except u_t/u_xx/u_yy/u_xy/u_x/u_y/u contributions to RHS
    # i.e. collect LHS coefficients of each derivative symbol
    coeff_of = {sym: lhs_expr.coeff(sym) for sym in
                [_u_t, _u_xx, _u_yy, _u_xy, _u_x, _u_y, _u]}

    g = float(coeff_of[_u_t]) if coeff_of[_u_t] is not None else 0.0
    a = float(coeff_of[_u_xx]) if coeff_of[_u_xx] is not None else 0.0
    b = float(coeff_of[_u_yy]) if coeff_of[_u_yy] is not None else 0.0
    c = float(coeff_of[_u_xy]) if coeff_of[_u_xy] is not None else 0.0
    d = float(coeff_of[_u_x]) if coeff_of[_u_x] is not None else 0.0
    e = float(coeff_of[_u_y]) if coeff_of[_u_y] is not None else 0.0
    f = float(coeff_of[_u]) if coeff_of[_u] is not None else 0.0

    # Validate: LHS should contain only these derivative syms (no x/y/t dependence)
    reconstructed_lhs = (
        g * _u_t + a * _u_xx + b * _u_yy + c * _u_xy + d * _u_x + e * _u_y + f * _u
    )
    residual_lhs = sp.simplify(lhs_expr - reconstructed_lhs)
    if residual_lhs != 0:
        # Accept it but warn — extra terms moved to negative RHS
        rhs_expr = rhs_expr + residual_lhs

    # Poisson-like: second-order steady (g=0), no mixed or first-order terms, constant coeffs
    poisson_like = (
        g == 0.0
        and (a != 0 or b != 0)
        and c == 0
        and d == 0
        and e == 0
        and _is_constant(rhs_expr)  # rhs may depend on x,y (that's fine), just no u
    )

    pde_class, discriminant = classify_pde(g, a, b, c)

    return ParsedPDE(
        g=g, a=a, b=b, c=c, d=d, e=e, f=f,
        rhs_expr=rhs_expr,
        is_poisson_like=poisson_like,
        pde_class=pde_class,
        discriminant=discriminant,
        raw_lhs=lhs_str,
        raw_rhs=rhs_str,
    )


def _is_constant(expr: sp.Expr) -> bool:
    """Return True if expr does not depend on u or its derivatives (but may depend on t)."""
    u_syms = {_u, _u_x, _u_y, _u_xx, _u_yy, _u_xy, _u_t}
    return not (expr.free_symbols & u_syms)


def _parse_expr(s: str) -> sp.Expr:
    """Parse a string expression using the allowed symbols."""
    try:
        return sp.sympify(s, locals=_ALLOWED_SYMS)
    except Exception as exc:
        raise ValueError(f"Cannot parse expression {s!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# BC parser
# ---------------------------------------------------------------------------

_VALID_WALLS = {"left", "right", "top", "bottom"}
_VALID_TYPES = {"dirichlet", "neumann", "robin"}


def parse_bc(bc_dict: dict[str, dict]) -> dict[str, ParsedBC]:
    """Parse a boundary-condition dict.

    Parameters
    ----------
    bc_dict : dict
        Keys are wall names (left/right/top/bottom).  Values are dicts with:
            type:   "dirichlet" | "neumann" | "robin"
            value:  expression string  (required)
            alpha:  float string       (robin only, default "1")
            beta:   float string       (robin only, default "0")

    Returns
    -------
    dict mapping wall name -> ParsedBC
    """
    result: dict[str, ParsedBC] = {}

    for wall, spec in bc_dict.items():
        wall = wall.lower().strip()
        if wall not in _VALID_WALLS:
            raise ValueError(f"Unknown wall {wall!r}. Must be one of {_VALID_WALLS}.")

        bc_type = spec.get("type", "dirichlet").lower().strip()
        if bc_type not in _VALID_TYPES:
            raise ValueError(
                f"Unknown BC type {bc_type!r} on wall {wall!r}. "
                f"Must be one of {_VALID_TYPES}."
            )

        value_str = str(spec.get("value", "0"))
        value_expr = _parse_expr(value_str)

        if bc_type == "dirichlet":
            alpha, beta = 1.0, 0.0
        elif bc_type == "neumann":
            alpha, beta = 0.0, 1.0
        else:  # robin
            alpha = float(spec.get("alpha", 1.0))
            beta = float(spec.get("beta", 0.0))

        result[wall] = ParsedBC(
            wall=wall,
            bc_type=bc_type,
            value_expr=value_expr,
            alpha=alpha,
            beta=beta,
        )

    missing = _VALID_WALLS - set(result.keys())
    if missing:
        # Default missing walls to homogeneous Dirichlet
        for wall in missing:
            result[wall] = ParsedBC(
                wall=wall,
                bc_type="dirichlet",
                value_expr=sp.Integer(0),
                alpha=1.0,
                beta=0.0,
            )

    return result


# ---------------------------------------------------------------------------
# IC parser
# ---------------------------------------------------------------------------

def parse_ic(ic_str: str) -> ParsedIC:
    """Parse an initial condition string.

    Parameters
    ----------
    ic_str : str
        Expression string for initial condition, e.g., "sin(pi*x)*cos(pi*y)"
        or "0" for homogeneous. Can use x, y, pi, sin, cos, exp, sqrt, etc.

    Returns
    -------
    ParsedIC
    """
    ic_str = ic_str.strip()
    if not ic_str:
        ic_str = "0"
    
    ic_expr = _parse_expr(ic_str)
    return ParsedIC(ic_expr)


# ---------------------------------------------------------------------------
# Finite-difference residual builder
# ---------------------------------------------------------------------------

def build_fd_residual(
    parsed: ParsedPDE,
) -> Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]:
    """Return a function  residual(u: Tensor[n,n], tt=None) -> Tensor[n-2, n-2].

    The residual is the spatial part of the PDE evaluated at interior points:
        g*u_t + a*u_xx + b*u_yy + c*u_xy + d*u_x + e*u_y + f*u = rhs
    
    For time-dependent PDEs (g != 0), u_t is returned as a constraint that should
    be satisfied by the time-stepping scheme (e.g., forward Euler).
    """
    g, a, b, c, d, e, f_coef = parsed.g, parsed.a, parsed.b, parsed.c, parsed.d, parsed.e, parsed.f
    rhs_fn = parsed.rhs_fn()

    def residual(u: torch.Tensor, tt: torch.Tensor | None = None) -> torch.Tensor:
        n = u.shape[0]
        h = 1.0 / (n - 1)
        x_1d = torch.linspace(0, 1, n, device=u.device, dtype=u.dtype)
        xx, yy = torch.meshgrid(x_1d, x_1d, indexing="ij")
        xx_int = xx[1:-1, 1:-1]
        yy_int = yy[1:-1, 1:-1]
        
        if tt is None:
            tt_int = torch.zeros_like(xx_int)
        else:
            tt_int = tt[1:-1, 1:-1]

        result = torch.zeros_like(u[1:-1, 1:-1])

        if a != 0.0:
            d2u_dx2 = (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / h**2
            result = result + a * d2u_dx2
        if b != 0.0:
            d2u_dy2 = (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / h**2
            result = result + b * d2u_dy2
        if c != 0.0:
            d2u_dxy = (u[2:, 2:] - u[2:, :-2] - u[:-2, 2:] + u[:-2, :-2]) / (4.0 * h**2)
            result = result + c * d2u_dxy
        if d != 0.0:
            du_dx = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * h)
            result = result + d * du_dx
        if e != 0.0:
            du_dy = (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * h)
            result = result + e * du_dy
        if f_coef != 0.0:
            result = result + f_coef * u[1:-1, 1:-1]

        rhs = rhs_fn(xx_int, yy_int, tt_int)
        return result - rhs

    return residual


# ---------------------------------------------------------------------------
# Convenience: check if PDE is Poisson-like (fast-path routing)
# ---------------------------------------------------------------------------

def is_poisson_like(parsed: ParsedPDE) -> bool:
    return parsed.is_poisson_like
