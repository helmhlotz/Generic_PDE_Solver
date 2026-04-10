from __future__ import annotations

import torch

from pde_parser import ParsedBC, ParsedIC, ParsedPDE, build_fd_residual


def _first_derivatives(field):
    dx = 1.0 / (field.shape[0] - 1)
    dy = 1.0 / (field.shape[1] - 1)
    dfield_dx = (field[2:, 1:-1] - field[:-2, 1:-1]) / (2.0 * dx)
    dfield_dy = (field[1:-1, 2:] - field[1:-1, :-2]) / (2.0 * dy)
    return dfield_dx, dfield_dy


def _second_derivatives(field):
    dx = 1.0 / (field.shape[0] - 1)
    dy = 1.0 / (field.shape[1] - 1)
    d2_dx2 = (field[2:, 1:-1] - 2.0 * field[1:-1, 1:-1] + field[:-2, 1:-1]) / dx**2
    d2_dy2 = (field[1:-1, 2:] - 2.0 * field[1:-1, 1:-1] + field[1:-1, :-2]) / dy**2
    return d2_dx2, d2_dy2


class LaplacePDE:
    """Laplace equation on [0,1]^2.

    Boundary: u=0 on left/right/bottom, u=sin(πx) on top.
    Exact solution: sin(πx)·sinh(πy)/sinh(π).
    """

    def compute_pde_loss(self, u):
        d2u_dx2, d2u_dy2 = _second_derivatives(u)
        return torch.mean((d2u_dx2 + d2u_dy2) ** 2)

    def compute_bc_loss(self, u, x_1d):
        target_top = torch.sin(torch.pi * x_1d)
        return (
            torch.mean(u[0, :] ** 2)
            + torch.mean(u[-1, :] ** 2)
            + torch.mean(u[:, 0] ** 2)
            + torch.mean((u[:, -1] - target_top) ** 2)
        )

    def exact_solution(self, xx, yy):
        pi = torch.tensor(torch.pi, device=xx.device, dtype=xx.dtype)
        return torch.sin(pi * xx) * torch.sinh(pi * yy) / torch.sinh(pi)


class NavierStokesPDE:
    """Incompressible Navier-Stokes for a lid-driven cavity.

    Boundary: u=lid_velocity on top wall, no-slip on all other walls.
    """

    def __init__(self, viscosity: float, density: float, lid_velocity: float):
        self.viscosity = viscosity
        self.density = density
        self.lid_velocity = lid_velocity

    def compute_pde_loss(self, fields):
        u_vel, v_vel, pressure = fields["u"], fields["v"], fields["p"]

        du_dx, du_dy = _first_derivatives(u_vel)
        dv_dx, dv_dy = _first_derivatives(v_vel)
        dp_dx, dp_dy = _first_derivatives(pressure)
        d2u_dx2, d2u_dy2 = _second_derivatives(u_vel)
        d2v_dx2, d2v_dy2 = _second_derivatives(v_vel)

        u_c = u_vel[1:-1, 1:-1]
        v_c = v_vel[1:-1, 1:-1]

        continuity = du_dx + dv_dy
        momentum_x = (
            u_c * du_dx + v_c * du_dy
            + dp_dx / self.density
            - self.viscosity * (d2u_dx2 + d2u_dy2)
        )
        momentum_y = (
            u_c * dv_dx + v_c * dv_dy
            + dp_dy / self.density
            - self.viscosity * (d2v_dx2 + d2v_dy2)
        )

        return (
            torch.mean(continuity ** 2)
            + torch.mean(momentum_x ** 2)
            + torch.mean(momentum_y ** 2)
        )

    def compute_bc_loss(self, fields, x_1d):
        u_vel, v_vel, pressure = fields["u"], fields["v"], fields["p"]
        lid_target = torch.full_like(x_1d, self.lid_velocity)

        velocity_bc = (
            torch.mean(u_vel[0, :] ** 2)
            + torch.mean(u_vel[-1, :] ** 2)
            + torch.mean(u_vel[:, 0] ** 2)
            + torch.mean((u_vel[:, -1] - lid_target) ** 2)
            + torch.mean(v_vel[0, :] ** 2)
            + torch.mean(v_vel[-1, :] ** 2)
            + torch.mean(v_vel[:, 0] ** 2)
            + torch.mean(v_vel[:, -1] ** 2)
        )
        pressure_anchor = torch.mean(pressure[1:-1, 1:-1]) ** 2
        return velocity_bc + pressure_anchor


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
