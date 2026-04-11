from __future__ import annotations

import numpy as np
import torch

from pde_parser import ParsedBC, ParsedIC, ParsedPDE, build_fd_residual


def compute_sample_metrics(
    u_pred: np.ndarray,
    u_fd: np.ndarray,
) -> tuple[float, float, float]:
    """Return ``(rel_l2, rmse, max_err)`` for a prediction vs FD reference pair.

    Both arrays are cast to float64 before comparison.  The relative L2 error
    is stabilised against zero-norm references with a small epsilon.

    Parameters
    ----------
    u_pred : predicted solution array (any shape).
    u_fd   : finite-difference reference array (same shape as u_pred).
    """
    u_pred = np.asarray(u_pred, dtype=np.float64)
    u_fd   = np.asarray(u_fd,   dtype=np.float64)
    diff   = u_pred - u_fd
    fd_norm = float(np.linalg.norm(u_fd))
    rel_l2  = float(np.linalg.norm(diff) / (fd_norm + 1e-12))
    rmse    = float(np.sqrt(np.mean(diff ** 2)))
    max_err = float(np.max(np.abs(diff)))
    return rel_l2, rmse, max_err


def solve_fd_jacobi(
    pde_obj: "GeneralPDE",
    n_points: int,
    device: torch.device,
    max_iterations: int = 5000,
    tolerance: float = 1e-5,
    print_every: int | None = None,
    sanitize_on_divergence: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, list[float]]:
    """Shared Jacobi FD solver used by both training and inference."""
    x_1d = torch.linspace(0, 1, n_points, device=device)
    y_1d = torch.linspace(0, 1, n_points, device=device)
    xx, yy = torch.meshgrid(x_1d, y_1d, indexing="ij")

    u = torch.zeros((n_points, n_points), device=device, dtype=torch.float32)
    u = pde_obj.apply_boundary_conditions(u, x_1d, y_1d)
    residual_fn = build_fd_residual(pde_obj.parsed_pde)

    parsed = pde_obj.parsed_pde
    h = 1.0 / (n_points - 1)
    diag = torch.zeros((n_points - 2, n_points - 2), device=device, dtype=torch.float32)
    if parsed.a != 0.0:
        diag = diag + parsed.a * (-2.0 / h**2)
    if parsed.b != 0.0:
        diag = diag + parsed.b * (-2.0 / h**2)
    if parsed.f != 0.0:
        diag = diag + parsed.f

    # Account for the implicit dependence of Robin/Neumann boundary values on
    # the adjacent interior points when forming the Jacobi diagonal.
    for wall, spec in pde_obj.bc_specs.items():
        alpha_w, beta_w = spec.alpha, spec.beta
        if beta_w == 0.0:
            continue
        denom_w = alpha_w + 3.0 * beta_w / (2.0 * h)
        if abs(denom_w) < 1e-14:
            continue
        coupling = 4.0 * beta_w / (h * denom_w)
        if wall == "left":
            diag[0, :] = diag[0, :] + parsed.a * coupling / h**2
        elif wall == "right":
            diag[-1, :] = diag[-1, :] + parsed.a * coupling / h**2
        elif wall == "bottom":
            diag[:, 0] = diag[:, 0] + parsed.b * coupling / h**2
        else:
            diag[:, -1] = diag[:, -1] + parsed.b * coupling / h**2

    if (diag.abs() < 1e-14).any():
        raise ValueError(
            "Degenerate FD stencil: the diagonal coefficient is zero at one or more "
            "interior points (a ≈ 0, b ≈ 0, and f ≈ 0). Jacobi relaxation cannot proceed."
        )

    history: list[float] = []
    for it in range(max_iterations):
        u_old = u.clone()
        residual = residual_fn(u)
        u[1:-1, 1:-1] = u[1:-1, 1:-1] - residual / diag
        u = pde_obj.apply_boundary_conditions(u, x_1d, y_1d)

        if torch.isnan(u).any() or torch.isinf(u).any():
            if sanitize_on_divergence:
                u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
                u = pde_obj.apply_boundary_conditions(u, x_1d, y_1d)
                return u.detach(), xx, yy, history
            return None, xx, yy, history

        change = torch.max(torch.abs(u - u_old)).item()
        rms_residual = torch.sqrt(torch.mean(residual**2)).item()
        history.append(rms_residual)

        if print_every is not None and (it % print_every == 0):
            print(f"  Iter {it}: residual={rms_residual:.3e}, change={change:.3e}")

        if change < tolerance:
            break

    return u.detach(), xx, yy, history


class GeneralPDE:
    """Compute physics, BC, and IC losses for an arbitrary parsed PDE.

    Parameters
    ----------
    parsed_pde : ParsedPDE
    bc_specs   : dict[str, ParsedBC]
    ic_specs   : ParsedIC, optional
        Initial condition; only used when the PDE is time-dependent.
    """

    def __init__(
        self,
        parsed_pde: ParsedPDE,
        bc_specs: dict[str, ParsedBC],
        ic_specs: ParsedIC | None = None,
    ):
        self.parsed_pde = parsed_pde
        self.bc_specs = bc_specs
        self.ic_specs = ic_specs
        self._residual_fn = build_fd_residual(parsed_pde)

    def compute_pde_loss(
        self,
        u: torch.Tensor,
        tt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mean squared PDE residual at interior grid points.

        Parameters
        ----------
        u  : Tensor of shape (n, n)
        tt : Tensor of shape (n, n), optional
            Time values at each grid point; required for time-dependent PDEs.
        """
        res = self._residual_fn(u, tt)
        return torch.mean(res ** 2)

    def compute_bc_loss(
        self, u: torch.Tensor, x_1d: torch.Tensor, y_1d: torch.Tensor
    ) -> torch.Tensor:
        """Enforce  α·u + β·∂u/∂n = value  on each wall.

        Normal derivatives are approximated with 2nd-order one-sided stencils:
            forward  (left/bottom):  du/dn ≈ (-3u[0] + 4u[1] - u[2]) / (2h)
            backward (right/top):    du/dn ≈  (3u[-1] - 4u[-2] + u[-3]) / (2h)
        """
        h = 1.0 / (u.shape[0] - 1)
        loss = torch.tensor(0.0, device=u.device)

        for wall, spec in self.bc_specs.items():
            val_fn = spec.value_fn()
            alpha = spec.alpha
            beta = spec.beta

            if wall == "left":
                u_w = u[0, :]
                xx_w = torch.zeros_like(y_1d)
                yy_w = y_1d
                # Outward normal is -x; 2nd-order one-sided: -du/dx
                du_dn = -(-3 * u[0, :] + 4 * u[1, :] - u[2, :]) / (2 * h)
            elif wall == "right":
                u_w = u[-1, :]
                xx_w = torch.ones_like(y_1d)
                yy_w = y_1d
                # Outward normal is +x; 2nd-order one-sided: +du/dx
                du_dn = (3 * u[-1, :] - 4 * u[-2, :] + u[-3, :]) / (2 * h)
            elif wall == "bottom":
                u_w = u[:, 0]
                xx_w = x_1d
                yy_w = torch.zeros_like(x_1d)
                # Outward normal is -y; 2nd-order one-sided: -du/dy
                du_dn = -(-3 * u[:, 0] + 4 * u[:, 1] - u[:, 2]) / (2 * h)
            else:  # top
                u_w = u[:, -1]
                xx_w = x_1d
                yy_w = torch.ones_like(x_1d)
                # Outward normal is +y; 2nd-order one-sided: +du/dy
                du_dn = (3 * u[:, -1] - 4 * u[:, -2] + u[:, -3]) / (2 * h)

            target = val_fn(xx_w, yy_w)
            lhs = alpha * u_w + beta * du_dn
            loss = loss + torch.mean((lhs - target) ** 2)

        return loss

    def compute_ic_loss(
        self,
        u_0: torch.Tensor,
        xx: torch.Tensor,
        yy: torch.Tensor,
    ) -> torch.Tensor:
        """Mean squared residual between u(x,y,0) and the parsed initial condition.

        Returns 0 if no initial condition was provided.
        """
        if self.ic_specs is None:
            return torch.tensor(0.0, device=u_0.device)
        ic_fn = self.ic_specs.ic_fn()
        target = ic_fn(xx, yy)
        return torch.mean((u_0 - target) ** 2)

    def apply_boundary_conditions(
        self,
        u: torch.Tensor,
        x_1d: torch.Tensor,
        y_1d: torch.Tensor,
    ) -> torch.Tensor:
        """Hard-enforce BCs by writing boundary values directly into u.

        All three BC types are handled via the unified 2nd-order one-sided ghost-node
        formula that matches the stencil used in ``compute_bc_loss``:

            α·u[wall] + β·du/dn = g

        Solving for u[wall] using a 2nd-order one-sided approximation of du/dn:

            left/bottom  (outward normal in −direction):
                du/dn ≈ −(−3u[0] + 4u[1] − u[2]) / (2h)
                u[0] = (g + β·(4u[1] − u[2])/(2h)) / (α + 3β/(2h))

            right/top  (outward normal in +direction):
                du/dn ≈  (3u[−1] − 4u[−2] + u[−3]) / (2h)
                u[−1] = (g + β·(4u[−2] − u[−3])/(2h)) / (α + 3β/(2h))

        Special cases recovered exactly:
          - Dirichlet (α=1, β=0):  u[wall] = g
          - Neumann   (α=0, β=1):  u[wall] = g·(2h/3) + (4u[n1] − u[n2])/3
          - Robin     (general):   weighted combination

        Using the same-order stencil as compute_bc_loss ensures BC loss is
        numerically zero when BCs are exactly satisfied, preventing spurious
        loss contributions during training and evaluation.
        """
        u = u.clone()
        h = 1.0 / (u.shape[0] - 1)

        for wall, spec in self.bc_specs.items():
            alpha, beta = spec.alpha, spec.beta
            # 2nd-order coefficient: α + 3β/(2h)
            denom = alpha + 3.0 * beta / (2.0 * h)
            if abs(denom) < 1e-14:
                # Degenerate BC coefficients — skip to avoid division by zero
                continue

            val_fn = spec.value_fn()

            if wall == "left":
                target = val_fn(torch.zeros_like(y_1d), y_1d)
                # du/dn = -du/dx at left; 2nd-order: -(−3u[0]+4u[1]−u[2])/(2h)
                u[0, :] = (target + (beta / (2.0 * h)) * (4.0 * u[1, :] - u[2, :])) / denom
            elif wall == "right":
                target = val_fn(torch.ones_like(y_1d), y_1d)
                # du/dn = +du/dx at right; 2nd-order: (3u[-1]−4u[-2]+u[-3])/(2h)
                u[-1, :] = (target + (beta / (2.0 * h)) * (4.0 * u[-2, :] - u[-3, :])) / denom
            elif wall == "bottom":
                target = val_fn(x_1d, torch.zeros_like(x_1d))
                # du/dn = -du/dy at bottom; 2nd-order: -(−3u[:,0]+4u[:,1]−u[:,2])/(2h)
                u[:, 0] = (target + (beta / (2.0 * h)) * (4.0 * u[:, 1] - u[:, 2])) / denom
            else:  # top
                target = val_fn(x_1d, torch.ones_like(x_1d))
                # du/dn = +du/dy at top; 2nd-order: (3u[:,-1]−4u[:,-2]+u[:,-3])/(2h)
                u[:, -1] = (target + (beta / (2.0 * h)) * (4.0 * u[:, -2] - u[:, -3])) / denom

        return u
