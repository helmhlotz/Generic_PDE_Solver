"""Conditional neural PDE solvers and operator-model utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from models.checkpoints import load_model_weights
from models.conditional_inputs import ConditionalGrid2D
from models.fno_model import FNO2DModel
from pde_parser import ParsedBC, ParsedIC, ParsedPDE
from physics.pde_helpers import GeneralPDE


class NeuralPDEModel(ABC):
    """Common interface for trainable neural PDE solvers."""

    model: nn.Module
    history: dict[str, list[float]]

    @abstractmethod
    def train(self, n_epochs: int = 3000, print_every: int = 500) -> None:
        """Train model parameters for the current PDE problem."""

    @abstractmethod
    def evaluate(self, n_eval: int = 100, t: float | None = None) -> dict[str, Any]:
        """Evaluate the learned solution field on an evaluation grid."""

    def save_weights(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)


class ConditionalFNO2D(NeuralPDEModel):
    """Conditional FNO solver for a parsed PDE and boundary specification."""

    def __init__(
        self,
        parsed_pde: ParsedPDE,
        bc_specs: dict[str, ParsedBC],
        ic_specs: ParsedIC | None = None,
        n_points: int = 50,
        t_value: float = 0.0,
        lr: float = 1e-3,
        lambda_physics: float = 1.0,
        lambda_bc: float = 10.0,
        lambda_ic: float = 10.0,
        width: int = 32,
        n_modes: tuple = (12, 12),
        n_layers: int = 4,
        pretrained_path: str | None = None,
        seed: int = 42,
        device: str | None = None,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.n_points = n_points
        self.t_value = float(t_value)
        self.lr = lr
        self.lambda_physics = lambda_physics
        self.lambda_bc = lambda_bc
        self.lambda_ic = lambda_ic
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.pde_obj = GeneralPDE(parsed_pde, bc_specs, ic_specs)
        source_fn = parsed_pde.rhs_fn()

        self.model = FNO2DModel(
            in_channels=7,
            out_channels=1,
            width=width,
            n_modes=n_modes,
            n_layers=n_layers,
        ).to(self.device)

        if pretrained_path is not None:
            if Path(pretrained_path).exists():
                load_model_weights(self.model, pretrained_path, self.device)
                print(f"Loaded pretrained FNO model from {pretrained_path}")
            else:
                print(f"Pretrained FNO path {pretrained_path!r} not found; training from scratch.")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        grid = ConditionalGrid2D(n_points, bc_specs, source_fn, self.device, t_value=self.t_value)
        self.input_grid = grid.input_grid
        self.x_1d = grid.x_1d
        self.y_1d = grid.y_1d
        self.xx = grid.xx
        self.yy = grid.yy
        self.tt = grid.tt

        self.history: dict[str, list[float]] = {"physics": [], "bc": [], "ic": [], "total": []}

    def _predict(self, input_grid: torch.Tensor) -> torch.Tensor:
        return self.model(input_grid).squeeze(0).squeeze(-1)

    def train(self, n_epochs: int = 3000, print_every: int = 500) -> None:
        self.model.train()
        for epoch in range(1, n_epochs + 1):
            self.optimizer.zero_grad()

            u = self._predict(self.input_grid)
            physics_loss = self.pde_obj.compute_pde_loss(u, self.tt)
            bc_loss = self.pde_obj.compute_bc_loss(u, self.x_1d, self.y_1d)
            ic_loss = torch.tensor(0.0, device=self.device)
            if self.pde_obj.parsed_pde.is_time_dependent and abs(self.t_value) < 1e-12:
                ic_loss = self.pde_obj.compute_ic_loss(u, self.xx, self.yy)

            loss = (
                self.lambda_physics * physics_loss
                + self.lambda_bc * bc_loss
                + self.lambda_ic * ic_loss
            )
            loss.backward()
            self.optimizer.step()

            self.history["physics"].append(physics_loss.item())
            self.history["bc"].append(bc_loss.item())
            self.history["ic"].append(ic_loss.item())
            self.history["total"].append(loss.item())

            if epoch % print_every == 0 or epoch == 1:
                print(
                    f"epoch {epoch:5d} | total {loss.item():.4e} | "
                    f"physics {physics_loss.item():.4e} | bc {bc_loss.item():.4e} | "
                    f"ic {ic_loss.item():.4e}"
                )

    def evaluate(self, n_eval: int = 100, t: float | None = None) -> dict[str, Any]:
        self.model.eval()
        source_fn = self.pde_obj.parsed_pde.rhs_fn()
        t_eval = self.t_value if t is None else float(t)
        grid = ConditionalGrid2D(
            n_eval, self.pde_obj.bc_specs, source_fn, self.device, t_value=t_eval
        )
        with torch.no_grad():
            u = self._predict(grid.input_grid)
        return {
            "u": u.cpu(),
            "xx": grid.xx.cpu(),
            "yy": grid.yy.cpu(),
            "residual": float(self.pde_obj.compute_pde_loss(u, grid.tt).item()),
        }

    def save_weights(self, path: str) -> None:
        super().save_weights(path)
        print(f"Saved weights -> {path}")

    def export_torchscript(self, path: str) -> None:
        self.model.eval()
        dummy = self.input_grid.cpu()
        with torch.no_grad():
            traced = torch.jit.trace(self.model.cpu(), dummy)
        traced.save(path)
        print(f"Exported TorchScript -> {path}")


class _PointwiseConditionalPINNNet(nn.Module):
    """Pointwise MLP that maps the 7-channel conditional encoding to u(x,y)."""

    def __init__(self, in_channels: int = 7, hidden: int = 64, n_layers: int = 4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_channels, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, input_grid: torch.Tensor) -> torch.Tensor:
        batch, nx, ny, channels = input_grid.shape
        flat = input_grid.reshape(batch * nx * ny, channels)
        out = self.net(flat)
        return out.reshape(batch, nx, ny, 1)


class DeepONet2DModel(nn.Module):
    """Fixed-resolution DeepONet consuming the shared 7-channel conditional grid."""

    def __init__(
        self,
        n_points: int,
        in_channels: int = 7,
        branch_hidden: int = 128,
        branch_layers: int = 3,
        trunk_hidden: int = 128,
        trunk_layers: int = 3,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        if n_points < 4:
            raise ValueError(f"DeepONet2DModel requires n_points >= 4; got {n_points}.")
        if in_channels < 1:
            raise ValueError(f"DeepONet2DModel requires in_channels >= 1; got {in_channels}.")
        if branch_layers < 1 or trunk_layers < 1:
            raise ValueError("DeepONet2DModel branch_layers and trunk_layers must be >= 1.")
        if branch_hidden < 1 or trunk_hidden < 1 or latent_dim < 1:
            raise ValueError("DeepONet2DModel widths and latent_dim must be >= 1.")

        self.n_points = int(n_points)
        self.in_channels = int(in_channels)
        self.latent_dim = int(latent_dim)
        self.register_buffer("coord_grid", self._make_coord_grid(self.n_points), persistent=False)

        self.branch_net = self._make_mlp(
            in_dim=self.n_points * self.n_points * self.in_channels,
            hidden_dim=branch_hidden,
            n_layers=branch_layers,
            out_dim=self.latent_dim,
        )
        self.trunk_net = self._make_mlp(
            in_dim=2,
            hidden_dim=trunk_hidden,
            n_layers=trunk_layers,
            out_dim=self.latent_dim,
        )
        self.output_bias = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _make_mlp(
        *,
        in_dim: int,
        hidden_dim: int,
        n_layers: int,
        out_dim: int,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        current_dim = in_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.Tanh()])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, out_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _make_coord_grid(n_points: int) -> torch.Tensor:
        x_1d = torch.linspace(0.0, 1.0, n_points)
        y_1d = torch.linspace(0.0, 1.0, n_points)
        xx, yy = torch.meshgrid(x_1d, y_1d, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

    def forward(self, input_grid: torch.Tensor) -> torch.Tensor:
        if input_grid.ndim != 4:
            raise ValueError(
                f"DeepONet2DModel expects input_grid rank 4 (batch, n, n, c); got {input_grid.shape}."
            )
        batch, nx, ny, channels = input_grid.shape
        if channels != self.in_channels:
            raise ValueError(
                f"DeepONet2DModel expects {self.in_channels} channels; got {channels}."
            )
        if nx != self.n_points or ny != self.n_points:
            raise ValueError(
                "DeepONet2DModel resolution mismatch: "
                f"expected {self.n_points}x{self.n_points}, got {nx}x{ny}."
            )

        branch_input = input_grid.reshape(batch, nx * ny * channels)
        branch_latent = self.branch_net(branch_input)
        trunk_input = self.coord_grid.to(device=input_grid.device, dtype=input_grid.dtype)
        trunk_latent = self.trunk_net(trunk_input)

        out = torch.einsum("bp,mp->bm", branch_latent, trunk_latent) + self.output_bias
        return out.reshape(batch, self.n_points, self.n_points, 1)


class ConditionalPINNOperator2D(NeuralPDEModel):
    """Shared-weights conditional PINN trained across many PDE problems."""

    def __init__(
        self,
        parsed_pde: ParsedPDE,
        bc_specs: dict[str, ParsedBC],
        ic_specs: ParsedIC | None = None,
        n_points: int = 50,
        t_value: float = 0.0,
        lr: float = 1e-3,
        lambda_physics: float = 1.0,
        lambda_bc: float = 10.0,
        lambda_ic: float = 10.0,
        hidden: int = 64,
        n_layers: int = 4,
        pretrained_path: str | None = None,
        seed: int = 42,
        device: str | None = None,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.n_points = n_points
        self.t_value = float(t_value)
        self.lr = lr
        self.lambda_physics = lambda_physics
        self.lambda_bc = lambda_bc
        self.lambda_ic = lambda_ic
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.pde_obj = GeneralPDE(parsed_pde, bc_specs, ic_specs)
        source_fn = parsed_pde.rhs_fn()
        self.model = _PointwiseConditionalPINNNet(
            in_channels=7,
            hidden=hidden,
            n_layers=n_layers,
        ).to(self.device)

        if pretrained_path is not None:
            load_model_weights(self.model, pretrained_path, self.device)
            print(f"Loaded shared conditional PINN model from {pretrained_path}")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        grid = ConditionalGrid2D(n_points, bc_specs, source_fn, self.device, t_value=self.t_value)
        self.input_grid = grid.input_grid
        self.x_1d = grid.x_1d
        self.y_1d = grid.y_1d
        self.xx = grid.xx
        self.yy = grid.yy
        self.tt = grid.tt

        self.history: dict[str, list[float]] = {"physics": [], "bc": [], "ic": [], "total": []}

    def _predict(self, input_grid: torch.Tensor) -> torch.Tensor:
        return self.model(input_grid).squeeze(0).squeeze(-1)

    def train(
        self,
        n_epochs: int = 1000,
        print_every: int = 500,
        n_lbfgs_steps: int = 0,
        early_stop_tol: float = 1e-6,
    ) -> None:
        self.model.train()

        if n_epochs > 0:
            for epoch in range(1, n_epochs + 1):
                self.optimizer.zero_grad()
                u = self._predict(self.input_grid)
                physics_loss = self.pde_obj.compute_pde_loss(u, self.tt)
                bc_loss = self.pde_obj.compute_bc_loss(u, self.x_1d, self.y_1d)
                ic_loss = torch.tensor(0.0, device=self.device)
                if self.pde_obj.parsed_pde.is_time_dependent and abs(self.t_value) < 1e-12:
                    ic_loss = self.pde_obj.compute_ic_loss(u, self.xx, self.yy)

                loss = (
                    self.lambda_physics * physics_loss
                    + self.lambda_bc * bc_loss
                    + self.lambda_ic * ic_loss
                )
                loss.backward()
                self.optimizer.step()

                self.history["physics"].append(physics_loss.item())
                self.history["bc"].append(bc_loss.item())
                self.history["ic"].append(ic_loss.item())
                self.history["total"].append(loss.item())

                if epoch % print_every == 0 or epoch == 1:
                    print(
                        f"[SharedPINN] epoch {epoch:5d} | total {loss.item():.4e} | "
                        f"physics {physics_loss.item():.4e} | bc {bc_loss.item():.4e}"
                    )

                if loss.item() < early_stop_tol:
                    print(f"  Early stop at epoch {epoch}: loss {loss.item():.3e}")
                    return

        if n_lbfgs_steps > 0:
            lbfgs = torch.optim.LBFGS(
                self.model.parameters(),
                lr=1.0,
                max_iter=20,
                history_size=50,
                line_search_fn="strong_wolfe",
            )

            for step in range(1, n_lbfgs_steps + 1):
                def closure():
                    lbfgs.zero_grad()
                    u = self._predict(self.input_grid)
                    physics_loss = self.pde_obj.compute_pde_loss(u, self.tt)
                    bc_loss = self.pde_obj.compute_bc_loss(u, self.x_1d, self.y_1d)
                    ic_loss = torch.tensor(0.0, device=self.device)
                    if self.pde_obj.parsed_pde.is_time_dependent and abs(self.t_value) < 1e-12:
                        ic_loss = self.pde_obj.compute_ic_loss(u, self.xx, self.yy)
                    loss = (
                        self.lambda_physics * physics_loss
                        + self.lambda_bc * bc_loss
                        + self.lambda_ic * ic_loss
                    )
                    loss.backward()
                    return loss

                loss = lbfgs.step(closure)
                loss_val = float(loss.item())
                self.history["total"].append(loss_val)
                if step % print_every == 0 or step == 1:
                    print(f"[SharedPINN/LBFGS] step {step:5d} | total {loss_val:.4e}")
                if loss_val < early_stop_tol:
                    print(f"  Early stop at L-BFGS step {step}: loss {loss_val:.3e}")
                    return

    def evaluate(self, n_eval: int = 100, t: float | None = None) -> dict[str, Any]:
        self.model.eval()
        source_fn = self.pde_obj.parsed_pde.rhs_fn()
        t_eval = self.t_value if t is None else float(t)
        grid = ConditionalGrid2D(
            n_eval, self.pde_obj.bc_specs, source_fn, self.device, t_value=t_eval
        )
        with torch.no_grad():
            u = self._predict(grid.input_grid)
        return {
            "u": u.cpu(),
            "xx": grid.xx.cpu(),
            "yy": grid.yy.cpu(),
            "residual": float(self.pde_obj.compute_pde_loss(u, grid.tt).item()),
        }


class CollocationPINN2D(NeuralPDEModel):
    """Collocation PINN that solves a single PDE problem by optimization."""

    def __init__(
        self,
        parsed_pde: ParsedPDE,
        bc_specs: dict[str, ParsedBC],
        ic_specs: ParsedIC | None = None,
        n_points: int = 50,
        t_end: float = 1.0,
        n_time: int = 20,
        lr: float = 1e-3,
        lambda_physics: float = 1.0,
        lambda_bc: float = 10.0,
        lambda_ic: float = 10.0,
        hidden: int = 64,
        n_layers: int = 4,
        pretrained_path: str | None = None,
        seed: int = 42,
        device: str | None = None,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.parsed_pde = parsed_pde
        self.bc_specs = bc_specs
        self.ic_specs = ic_specs
        self.n_points = n_points
        self.t_end = t_end
        self.n_time = n_time
        self.lr = lr
        self.lambda_physics = lambda_physics
        self.lambda_bc = lambda_bc
        self.lambda_ic = lambda_ic
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        in_dim = 3 if parsed_pde.is_time_dependent else 2
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.model = nn.Sequential(*layers).to(self.device)

        if pretrained_path is not None:
            load_model_weights(self.model, pretrained_path, self.device, skip_if_torchscript=True)
            print(f"Loaded pretrained PINN model from {pretrained_path}")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.pde_obj = GeneralPDE(parsed_pde, bc_specs, ic_specs)
        self.history: dict[str, list[float]] = {
            "physics": [], "bc": [], "ic": [], "total": []
        }

    def _build_collocation(self) -> torch.Tensor:
        x_1d = torch.linspace(0, 1, self.n_points, device=self.device)
        y_1d = torch.linspace(0, 1, self.n_points, device=self.device)
        xx, yy = torch.meshgrid(x_1d, y_1d, indexing="ij")
        if self.parsed_pde.is_time_dependent:
            t1d = torch.linspace(0, self.t_end, self.n_time, device=self.device)
            xf = xx.reshape(-1).repeat(self.n_time)
            yf = yy.reshape(-1).repeat(self.n_time)
            tf = t1d.repeat_interleave(self.n_points**2)
            xyt = torch.stack([xf, yf, tf], dim=1).requires_grad_(True)
        else:
            xyt = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).requires_grad_(True)
        return xyt

    def _predict(self, xyt: torch.Tensor) -> torch.Tensor:
        return self.model(xyt).squeeze(-1)

    def _compute_pde_loss_autograd(self, xyt: torch.Tensor) -> torch.Tensor:
        u = self._predict(xyt)

        g = self.parsed_pde.g
        a = self.parsed_pde.a
        b = self.parsed_pde.b
        c = self.parsed_pde.c
        d = self.parsed_pde.d
        e = self.parsed_pde.e
        f_coef = self.parsed_pde.f

        ones = torch.ones_like(u)

        def grad1(out, inp):
            return torch.autograd.grad(
                out, inp, grad_outputs=ones, create_graph=True, retain_graph=True
            )[0]

        def grad2_diag(first_deriv, inp, col):
            return torch.autograd.grad(
                first_deriv[:, col], inp, grad_outputs=ones, create_graph=True, retain_graph=True
            )[0][:, col]

        du = grad1(u, xyt)
        du_dx = du[:, 0]

        residual = torch.zeros_like(u)

        if a != 0.0:
            d2u_dx2 = grad2_diag(du, xyt, 0)
            residual = residual + a * d2u_dx2
        if b != 0.0:
            d2u_dy2 = grad2_diag(du, xyt, 1)
            residual = residual + b * d2u_dy2
        if c != 0.0:
            d2u_dxy = torch.autograd.grad(
                du_dx, xyt, grad_outputs=ones, create_graph=True, retain_graph=True
            )[0][:, 1]
            residual = residual + c * d2u_dxy
        if d != 0.0:
            residual = residual + d * du_dx
        if e != 0.0:
            residual = residual + e * du[:, 1]
        if f_coef != 0.0:
            residual = residual + f_coef * u
        if g != 0.0 and xyt.shape[1] == 3:
            residual = residual + g * du[:, 2]

        x_col = xyt[:, 0]
        y_col = xyt[:, 1]
        t_col = xyt[:, 2] if xyt.shape[1] == 3 else torch.zeros_like(x_col)
        rhs_fn = self.parsed_pde.rhs_fn()
        rhs = rhs_fn(x_col, y_col, t_col)

        return torch.mean((residual - rhs) ** 2)

    def _compute_bc_loss(self) -> torch.Tensor:
        x_1d = torch.linspace(0, 1, self.n_points, device=self.device)
        y_1d = torch.linspace(0, 1, self.n_points, device=self.device)
        t_bc = torch.zeros(self.n_points, device=self.device)

        loss = torch.tensor(0.0, device=self.device)

        wall_inputs = {
            "left": (torch.zeros_like(y_1d), y_1d, t_bc),
            "right": (torch.ones_like(y_1d), y_1d, t_bc),
            "bottom": (x_1d, torch.zeros_like(x_1d), t_bc),
            "top": (x_1d, torch.ones_like(x_1d), t_bc),
        }

        for wall, spec in self.bc_specs.items():
            xw, yw, tw = wall_inputs[wall]
            if self.parsed_pde.is_time_dependent:
                inp = torch.stack([xw, yw, tw], dim=1).requires_grad_(True)
            else:
                inp = torch.stack([xw, yw], dim=1).requires_grad_(True)

            u_w = self._predict(inp)
            val_fn = spec.value_fn()
            target = val_fn(xw, yw)

            if spec.bc_type == "dirichlet":
                loss = loss + torch.mean((u_w - target) ** 2)
            else:
                du = torch.autograd.grad(
                    u_w, inp, grad_outputs=torch.ones_like(u_w), create_graph=True
                )[0]
                if wall in ("left", "right"):
                    sign = 1.0 if wall == "right" else -1.0
                    du_dn = sign * du[:, 0]
                else:
                    sign = 1.0 if wall == "top" else -1.0
                    du_dn = sign * du[:, 1]
                lhs = spec.alpha * u_w + spec.beta * du_dn
                loss = loss + torch.mean((lhs - target) ** 2)

        return loss

    def _compute_ic_loss(self) -> torch.Tensor:
        if self.ic_specs is None or not self.parsed_pde.is_time_dependent:
            return torch.tensor(0.0, device=self.device)

        x_1d = torch.linspace(0, 1, self.n_points, device=self.device)
        y_1d = torch.linspace(0, 1, self.n_points, device=self.device)
        xx, yy = torch.meshgrid(x_1d, y_1d, indexing="ij")
        xf, yf = xx.reshape(-1), yy.reshape(-1)
        tf = torch.zeros_like(xf)

        inp = torch.stack([xf, yf, tf], dim=1)
        u_0 = self._predict(inp)

        ic_fn = self.ic_specs.ic_fn()
        target = ic_fn(xf, yf)
        return torch.mean((u_0 - target) ** 2)

    def train(
        self,
        n_epochs: int = 1000,
        print_every: int = 500,
        n_lbfgs_steps: int = 200,
        early_stop_tol: float = 1e-6,
    ) -> None:
        self.model.train()

        if n_epochs > 0:
            xyt = self._build_collocation()
            for epoch in range(1, n_epochs + 1):
                self.optimizer.zero_grad()

                physics_loss = self._compute_pde_loss_autograd(xyt)
                bc_loss = self._compute_bc_loss()
                ic_loss = self._compute_ic_loss()

                loss = (
                    self.lambda_physics * physics_loss
                    + self.lambda_bc * bc_loss
                    + self.lambda_ic * ic_loss
                )
                loss.backward()
                self.optimizer.step()

                self.history["physics"].append(physics_loss.item())
                self.history["bc"].append(bc_loss.item())
                self.history["ic"].append(ic_loss.item())
                self.history["total"].append(loss.item())

                if epoch % print_every == 0 or epoch == 1:
                    print(
                        f"[Adam]   epoch {epoch:5d} | total {loss.item():.4e} | "
                        f"physics {physics_loss.item():.4e} | bc {bc_loss.item():.4e}"
                    )

                if loss.item() < early_stop_tol:
                    print(f"  Early stop at Adam epoch {epoch}: loss {loss.item():.3e}")
                    return

        if n_lbfgs_steps > 0:
            lbfgs = torch.optim.LBFGS(
                self.model.parameters(),
                lr=1.0,
                max_iter=20,
                history_size=50,
                tolerance_grad=1e-9,
                tolerance_change=1e-11,
                line_search_fn="strong_wolfe",
            )
            _last: dict = {"phy": 0.0, "bc": 0.0, "ic": 0.0, "total": 0.0}

            for step in range(1, n_lbfgs_steps + 1):
                def closure():
                    lbfgs.zero_grad()
                    xyt_c = self._build_collocation()
                    phy = self._compute_pde_loss_autograd(xyt_c)
                    bc = self._compute_bc_loss()
                    ic = self._compute_ic_loss()
                    l = self.lambda_physics * phy + self.lambda_bc * bc + self.lambda_ic * ic
                    l.backward()
                    _last["phy"] = phy.item()
                    _last["bc"] = bc.item()
                    _last["ic"] = ic.item()
                    _last["total"] = l.item()
                    return l

                lbfgs.step(closure)
                self.history["physics"].append(_last["phy"])
                self.history["bc"].append(_last["bc"])
                self.history["ic"].append(_last["ic"])
                self.history["total"].append(_last["total"])

                if step % print_every == 0 or step == 1:
                    print(
                        f"[L-BFGS] step  {step:5d} | total {_last['total']:.4e} | "
                        f"physics {_last['phy']:.4e} | bc {_last['bc']:.4e}"
                    )

                if _last["total"] < early_stop_tol:
                    print(f"  Early stop at L-BFGS step {step}: loss {_last['total']:.3e}")
                    break

    def evaluate(self, n_eval: int = 100, t: float | None = None) -> dict[str, Any]:
        self.model.eval()
        x_1d = torch.linspace(0, 1, n_eval, device=self.device)
        y_1d = torch.linspace(0, 1, n_eval, device=self.device)
        xx, yy = torch.meshgrid(x_1d, y_1d, indexing="ij")
        xf, yf = xx.reshape(-1), yy.reshape(-1)

        with torch.no_grad():
            if self.parsed_pde.is_time_dependent:
                t = 0.0 if t is None else float(t)
                tf = torch.full_like(xf, t)
                inp = torch.stack([xf, yf, tf], dim=1)
            else:
                inp = torch.stack([xf, yf], dim=1)
            u = self.model(inp).squeeze(-1).reshape(n_eval, n_eval)

        return {
            "u": u.cpu(),
            "xx": xx.cpu(),
            "yy": yy.cpu(),
        }

    def save_weights(self, path: str) -> None:
        super().save_weights(path)
        print(f"Saved PINN weights -> {path}")


# Backward-compatible public name: "ConditionalPINN2D" now refers only to the
# shared operator-style PINN. The collocation solver remains available under an
# explicit name so runtime code cannot confuse the two roles.
ConditionalPINN2D = ConditionalPINNOperator2D


__all__ = [
    "CollocationPINN2D",
    "ConditionalFNO2D",
    "DeepONet2DModel",
    "ConditionalPINN2D",
    "ConditionalPINNOperator2D",
    "NeuralPDEModel",
    "_PointwiseConditionalPINNNet",
]
