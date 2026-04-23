"""Generalized inference engine for 2-D scalar PDEs.

Public API
----------
    from inference_engine import InferenceEngine, SolverOption, FNOConfig, GridConfig

    option = SolverOption(
        solver_type="fno",
        fno=FNOConfig(
            model_path="pretrained_models/fno.pt",
            manifest_path="pretrained_models/manifest.npz",
        ),
    )
    engine = InferenceEngine(solver_option=option)

    result = engine.solve(
        pde_str = "u_xx + u_yy = sin(pi*x)*cos(pi*y)",
        bc_dict = {
            "left":   {"type": "dirichlet", "value": "0"},
            "right":  {"type": "dirichlet", "value": "0"},
            "bottom": {"type": "dirichlet", "value": "0"},
            "top":    {"type": "dirichlet", "value": "sin(pi*x)"},
        },
    )
    # result.is_ood == True  =>  query is out-of-distribution; FD fallback was used

Inference paths
---------------
FNO path  (FNO, query within training distribution)
    Single forward pass through the conditional FNO.

OOD fallback  (FNO, query outside training distribution)
    Detected via a KNN check against the training-set manifest.
    Routes to FD and sets ``result.is_ood = True`` with a reason string.

FD path  (solver_type="fd" or no learned artifact available)
    Finite difference iterative solver using Jacobi relaxation.

DeepONet path  (solver_type="deeponet")
    Single forward pass through the fixed-resolution DeepONet operator.

Offline training
----------------
    Use ``src/trainer.py`` to build ``fno.pt`` / ``deeponet.pt`` and ``manifest.npz``:
        python src/trainer.py fno --samples 2000 --epochs 30
        python src/trainer.py train --solver deeponet --train-dataset pretrained_models/train_data
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).parent))

from models.checkpoints import load_model_weights, read_checkpoint_arch, read_checkpoint_n_points
from models.conditional_inputs import CONDITIONAL_INPUT_CHANNELS, ConditionalGrid2D
from models.conditional_solvers import (
    ConditionalFNO2D,
    DeepONet2DModel,
)
from models.fno_model import FNO2DModel
from ood_detector import OODDetector
from pde_parser import ParsedPDE, parse_bc, parse_pde
from physics.pde_helpers import GeneralPDE, solve_fd_jacobi
from typing import Any, Literal


@dataclass
class GridConfig:
    n_points: int = 64

    def __post_init__(self) -> None:
        if self.n_points < 4:
            raise ValueError(
                f"GridConfig.n_points must be >= 4 (one-sided BC stencil needs "
                f"at least 3 interior points); got {self.n_points}."
            )


@dataclass
class FDConfig:
    max_iterations: int = 5000
    tolerance: float = 1e-5
    print_every: int = 500


@dataclass
class FNOConfig:
    model_path: str | None = None
    width: int = 32
    n_modes: tuple[int, int] = (12, 12)
    n_layers: int = 4
    lambda_physics: float = 1.0
    lambda_bc: float = 10.0
    print_every: int = 500
    # Path to the OOD manifest built by trainer.py; None disables OOD checking.
    manifest_path: str | None = "pretrained_models/manifest.npz"


@dataclass
class DeepONetConfig:
    model_path: str | None = None
    manifest_path: str | None = "pretrained_models/manifest.npz"
    branch_hidden: int = 128
    branch_layers: int = 3
    trunk_hidden: int = 128
    trunk_layers: int = 3
    latent_dim: int = 128
    epochs: int = 20
    lr: float = 1e-3
    lambda_bc: float = 0.0
    print_every: int = 500


@dataclass
class SolverOption:
    """Top-level solver configuration used by InferenceEngine.

    Defaults route to FD with conservative settings. UI can selectively
    override fields for FNO/DeepONet/FD runs.
    """

    solver_type: Literal["fd", "fno", "deeponet"] = "fd"
    grid: GridConfig = field(default_factory=GridConfig)
    fd: FDConfig = field(default_factory=FDConfig)
    fno: FNOConfig = field(default_factory=FNOConfig)
    deeponet: DeepONetConfig = field(default_factory=DeepONetConfig)
    device: str | None = None


@dataclass(frozen=True)
class _SolveRequest:
    parsed_pde: ParsedPDE
    bc_specs: dict[str, Any]
    pde_obj: GeneralPDE
    source_fn: Any
    n_points: int
    lam_bc: float
    fno_epochs: int
    fno_lr: float
    deeponet_epochs: int
    deeponet_lr: float
    print_every: int


@dataclass(frozen=True)
class _SolvePlan:
    route: Literal[
        "fd",
        "fd_missing_artifact",
        "fd_resolution_mismatch",
        "fd_ood_fallback",
        "fno_offline",
        "fno_torchscript",
        "deeponet_offline",
    ]
    route_reason: str
    ood_reason: str = ""

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

class SolveResult:
    """Container for the solution returned by InferenceEngine.solve()."""

    def __init__(
        self,
        u: np.ndarray,
        xx: np.ndarray,
        yy: np.ndarray,
        residual: float,
        bc_error: float,
        method: str,
        history: list[float] | None = None,
        is_ood: bool = False,
        ood_reason: str = "",
        route_reason: str = "",
    ):
        self.u = u                    # (n, n) numpy array — solution field
        self.xx = xx                  # (n, n) x-coordinates
        self.yy = yy                  # (n, n) y-coordinates
        self.residual = residual      # mean squared PDE residual
        self.bc_error = bc_error      # mean squared BC error
        self.method = method          # "fno" | "fd" | "deeponet" | "torchscript"
        self.history = history or []  # loss per epoch (fd)
        self.is_ood = is_ood          # True when OOD detected and FD fallback used
        self.ood_reason = ood_reason  # human-readable OOD explanation
        self.route_reason = route_reason  # human-readable routing decision

    def to_dict(self) -> dict[str, Any]:
        return {
            "u": self.u,
            "xx": self.xx,
            "yy": self.yy,
            "residual": self.residual,
            "bc_error": self.bc_error,
            "method": self.method,
            "history": self.history,
            "is_ood": self.is_ood,
            "ood_reason": self.ood_reason,
            "route_reason": self.route_reason,
        }


# ---------------------------------------------------------------------------
# Solver abstractions
# ---------------------------------------------------------------------------

class _AbstractSolver(ABC):
    """Common abstraction for all inference back-end solvers."""

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")

    @abstractmethod
    def solve(self, *args, **kwargs):
        """Execute solver-specific inference/training and return outputs."""


class _FNOSolver(_AbstractSolver):
    """FNO solver handling both fast forward and fine-tune modes."""

    def __init__(
        self,
        model: FNO2DModel,
        width: int,
        n_modes: tuple,
        n_layers: int,
        device: torch.device = None,
    ):
        super().__init__(device)
        self.model = model
        self.width = width
        self.n_modes = n_modes
        self.n_layers = n_layers

    def solve(
        self,
        parsed_pde: ParsedPDE,
        bc_specs: dict,
        pde_obj: GeneralPDE,
        source_fn,
        n_points: int,
    ) -> SolveResult:
        return self._inference(bc_specs, pde_obj, source_fn, n_points)

    def _inference(
        self,
        bc_specs: dict,
        pde_obj: GeneralPDE,
        source_fn,
        n_points: int,
    ) -> SolveResult:
        grid = ConditionalGrid2D(n_points, pde_obj.parsed_pde, bc_specs, self.device)
        self.model.eval()
        with torch.no_grad():
            u = self.model(grid.input_grid).squeeze(0).squeeze(-1)

        # FNO output is a 2-D spatial snapshot; computing a PDE residual that
        # includes a time-derivative term (g·u_t) against tt=None is meaningless.
        # Guard here catches any path that bypasses the top-level solve() check.
        if pde_obj.parsed_pde.is_time_dependent:
            residual = float("nan")
            bc_err = float("nan")
        else:
            residual = float(pde_obj.compute_pde_loss(u).item())
            bc_err = float(pde_obj.compute_bc_loss(u, grid.x_1d, grid.y_1d).item())
        return SolveResult(
            u=u.cpu().numpy(),
            xx=grid.xx.cpu().numpy(),
            yy=grid.yy.cpu().numpy(),
            residual=residual,
            bc_error=bc_err,
            method="fno",
        )


class _FDSolver(_AbstractSolver):
    """Finite difference solver using iterative Jacobi relaxation."""

    def __init__(self, device: torch.device = None):
        super().__init__(device)

    def solve(
        self,
        pde_obj: GeneralPDE,
        n_points: int,
        max_iterations: int = 5000,
        tolerance: float = 1e-5,
        print_every: int = 500,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
        """Solve PDE using Jacobi iteration with boundary conditions.

        Returns (u, xx, yy, history)
        """
        print(f"FD solver: {n_points}×{n_points} grid, max {max_iterations} iterations")
        u, xx, yy, history = solve_fd_jacobi(
            pde_obj=pde_obj,
            n_points=n_points,
            device=self.device,
            max_iterations=max_iterations,
            tolerance=tolerance,
            print_every=print_every,
            sanitize_on_divergence=True,
        )
        if u is None:
            raise RuntimeError("FD solver failed to produce a solution.")
        if history and history[-1] < tolerance:
            print(f"  Converged after {len(history) - 1} iterations")

        return (
            u.cpu().numpy(),
            xx.cpu().numpy(),
            yy.cpu().numpy(),
            history,
        )


# ---------------------------------------------------------------------------
# Helper: detect file format
# ---------------------------------------------------------------------------

def _is_torchscript(path: Path) -> bool:
    """Return True if *path* is a TorchScript ZIP archive (not a pickle state-dict).
    
    TorchScript archives contain 'constants.pkl' or 'code/' directory.
    Regular state dicts don't have these files.
    """
    if not zipfile.is_zipfile(path):
        return False
    
    try:
        with zipfile.ZipFile(path, 'r') as z:
            names = z.namelist()
            # TorchScript archives have code or constants
            has_torchscript_structure = (
                any('constants.pkl' in name for name in names) or
                any(name.startswith('code/') for name in names)
            )
            return has_torchscript_structure
    except Exception:
        return False


def _is_deeponet_checkpoint(path: str | None, device: torch.device) -> bool:
    """Return True if *path* is a DeepONet metadata-envelope checkpoint."""
    if path is None:
        return False
    return read_checkpoint_arch(path, device) == "deeponet"


# ---------------------------------------------------------------------------
# Main inference engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """Load a pre-trained conditional FNO and solve arbitrary 2-D PDEs.

    Parameters
    ----------
    solver_option : SolverOption
        Full solver configuration.  Use the :class:`SolverOption`,
        :class:`FNOConfig`, :class:`DeepONetConfig`, and :class:`FDConfig`
        dataclasses to build the configuration.
    """

    def __init__(
        self,
        solver_option: SolverOption,
    ):
        self.options = solver_option
        self.device = torch.device(
            self.options.device
            if self.options.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.width = self.options.fno.width
        self.n_modes = self.options.fno.n_modes
        self.n_layers = self.options.fno.n_layers
        self._model: FNO2DModel | None = None
        self._jit_model = None  # set when a TorchScript archive is loaded
        self._ood_detector: OODDetector | None = None
        self._deeponet_model: DeepONet2DModel | None = None
        self._has_fno_artifact = False
        self._has_deeponet_artifact = False
        self._deeponet_n_points: int | None = None

        if self.options.solver_type == "fno":
            self._model = FNO2DModel(
                in_channels=CONDITIONAL_INPUT_CHANNELS,
                out_channels=1,
                width=self.options.fno.width,
                n_modes=self.options.fno.n_modes,
                n_layers=self.options.fno.n_layers,
            ).to(self.device)
            if self.options.fno.model_path is not None:
                p = Path(self.options.fno.model_path)
                if p.exists():
                    if _is_torchscript(p):
                        self._jit_model = torch.jit.load(str(p), map_location=self.device)
                        self._jit_model.eval()
                        self._has_fno_artifact = True
                        print(f"Loaded TorchScript FNO from {p}")
                    else:
                        load_model_weights(self._model, str(p), self.device)
                        self._model.eval()
                        self._has_fno_artifact = True
                        # torch.compile + FFT kernels can fail at runtime on some
                        # backends/versions (stride/layout assertions in _fft_c2c).
                        # Keep eager mode as the default for reliability; allow
                        # opt-in compile via env var for benchmarking.
                        enable_compile = os.getenv("FNO_ENABLE_TORCH_COMPILE", "0") == "1"
                        if enable_compile and hasattr(torch, "compile"):
                            self._model = torch.compile(self._model, mode="reduce-overhead")
                            print("Enabled torch.compile for FNO (FNO_ENABLE_TORCH_COMPILE=1)")
                        print(f"Loaded conditional FNO from {p}")
                else:
                    print(f"FNO model path {p} not found; solve() will fall back to FD.")
            manifest = self.options.fno.manifest_path
            if manifest is not None and OODDetector.is_available(manifest):
                self._ood_detector = OODDetector(manifest)
                print(f"OOD detector loaded from {manifest}")
        elif self.options.solver_type == "deeponet":
            if self.options.deeponet.model_path is not None:
                p = Path(self.options.deeponet.model_path)
                if p.exists():
                    self._deeponet_n_points = read_checkpoint_n_points(str(p), self.device)
                    if self._deeponet_n_points is None:
                        self._deeponet_n_points = self.options.grid.n_points
                    self._deeponet_model = DeepONet2DModel(
                        n_points=self._deeponet_n_points,
                        in_channels=CONDITIONAL_INPUT_CHANNELS,
                        branch_hidden=self.options.deeponet.branch_hidden,
                        branch_layers=self.options.deeponet.branch_layers,
                        trunk_hidden=self.options.deeponet.trunk_hidden,
                        trunk_layers=self.options.deeponet.trunk_layers,
                        latent_dim=self.options.deeponet.latent_dim,
                    ).to(self.device)
                    load_model_weights(
                        self._deeponet_model,
                        self.options.deeponet.model_path,
                        self.device,
                        skip_if_torchscript=True,
                    )
                    self._deeponet_model.eval()
                    self._has_deeponet_artifact = True
                    print(f"Loaded DeepONet model from {self.options.deeponet.model_path}")
                else:
                    print(f"DeepONet model path {p} not found; solve() will fall back to FD.")
            manifest = self.options.deeponet.manifest_path
            if manifest is not None and OODDetector.is_available(manifest):
                self._ood_detector = OODDetector(manifest)
                print(f"OOD detector loaded from {manifest}")

    # ------------------------------------------------------------------
    def solve(
        self,
        pde_str: str,
        bc_dict: dict[str, dict],
        ic_str: str | None = None,
        n_points: int | None = None,
        lambda_bc: float | None = None,
        fno_epochs: int | None = None,
        fno_lr: float | None = None,
        deeponet_epochs: int | None = None,
        deeponet_lr: float | None = None,
        print_every: int | None = None,
    ) -> SolveResult:
        """Parse PDE + BCs and return the solution field.

        Parameters
        ----------
        pde_str         : PDE string, e.g. ``"u_xx + u_yy = sin(pi*x)"``.
        bc_dict         : per-wall BCs dict.
        ic_str          : initial condition for time-dependent PDEs (optional).
        n_points        : grid resolution (n × n); defaults to GridConfig.n_points.
        lambda_bc       : BC-loss weight for FNO/DeepONet.
        fno_epochs      : epochs for online FNO training (overrides FNO-online default).
        fno_lr          : learning rate for online FNO training.
        deeponet_epochs : retained for API symmetry; DeepONet v1 is offline only.
        deeponet_lr     : retained for API symmetry; DeepONet v1 is offline only.
        print_every     : log interval for iterative solvers.

        Returns
        -------
        SolveResult
            ``result.is_ood == True`` when an FNO OOD detection triggered FD fallback.
        """
        request = self._build_request(
            pde_str=pde_str,
            bc_dict=bc_dict,
            ic_str=ic_str,
            n_points=n_points,
            lambda_bc=lambda_bc,
            fno_epochs=fno_epochs,
            fno_lr=fno_lr,
            deeponet_epochs=deeponet_epochs,
            deeponet_lr=deeponet_lr,
            print_every=print_every,
        )
        plan = self._plan_solve(request)
        return self._execute_plan(request, plan)

    def _build_request(
        self,
        *,
        pde_str: str,
        bc_dict: dict[str, dict],
        ic_str: str | None,
        n_points: int | None,
        lambda_bc: float | None,
        fno_epochs: int | None,
        fno_lr: float | None,
        deeponet_epochs: int | None,
        deeponet_lr: float | None,
        print_every: int | None,
    ) -> _SolveRequest:
        parsed_pde = parse_pde(pde_str)
        bc_specs = parse_bc(bc_dict)

        if parsed_pde.is_time_dependent:
            raise NotImplementedError(
                "Time-dependent PDEs are not supported by the current inference paths. "
                "The active FNO/DeepONet/FD solvers only handle steady-state problems."
            )

        ic_specs = None
        if ic_str and parsed_pde.is_time_dependent:
            from pde_parser import parse_ic

            ic_specs = parse_ic(ic_str)

        pde_obj = GeneralPDE(parsed_pde, bc_specs, ic_specs)
        source_fn = parsed_pde.rhs_fn()

        n_points_eff = int(n_points if n_points is not None else self.options.grid.n_points)
        fno_cfg = self.options.fno
        deeponet_cfg = self.options.deeponet
        fd_cfg = self.options.fd
        solver_type = self.options.solver_type

        lam_bc_eff = float(
            lambda_bc if lambda_bc is not None else (
                fno_cfg.lambda_bc if solver_type == "fno" else deeponet_cfg.lambda_bc
            )
        )
        fno_epochs_eff = int(
            fno_epochs if fno_epochs is not None else (
                deeponet_epochs if deeponet_epochs is not None else deeponet_cfg.epochs
            )
        )
        fno_lr_eff = float(
            fno_lr if fno_lr is not None else (
                deeponet_lr if deeponet_lr is not None else deeponet_cfg.lr
            )
        )
        deeponet_epochs_eff = int(
            deeponet_epochs if deeponet_epochs is not None else deeponet_cfg.epochs
        )
        deeponet_lr_eff = float(deeponet_lr if deeponet_lr is not None else deeponet_cfg.lr)
        print_every_eff = int(
            print_every if print_every is not None else (
                fno_cfg.print_every if solver_type == "fno" else
                deeponet_cfg.print_every if solver_type == "deeponet" else
                fd_cfg.print_every
            )
        )

        return _SolveRequest(
            parsed_pde=parsed_pde,
            bc_specs=bc_specs,
            pde_obj=pde_obj,
            source_fn=source_fn,
            n_points=n_points_eff,
            lam_bc=lam_bc_eff,
            fno_epochs=fno_epochs_eff,
            fno_lr=fno_lr_eff,
            deeponet_epochs=deeponet_epochs_eff,
            deeponet_lr=deeponet_lr_eff,
            print_every=print_every_eff,
        )

    def _plan_solve(self, request: _SolveRequest) -> _SolvePlan:
        solver_type = self.options.solver_type
        if solver_type == "fd":
            return _SolvePlan("fd", "solver_type=fd")

        if solver_type == "fno":
            if not self._has_fno_artifact:
                return _SolvePlan(
                    "fd_missing_artifact",
                    "FNO artifact unavailable; using FD fallback.",
                )
            is_ood, ood_reason = self._check_ood(request=request)
            if is_ood:
                return _SolvePlan(
                    "fd_ood_fallback",
                    f"OOD detected ({ood_reason}); using FD fallback.",
                    ood_reason=ood_reason,
                )
            if self._jit_model is not None:
                return _SolvePlan("fno_torchscript", "Using TorchScript FNO artifact.")
            if self._model is not None:
                return _SolvePlan("fno_offline", "Using loaded FNO operator artifact.")
            return _SolvePlan(
                "fd_missing_artifact",
                "FNO model state unavailable after load; using FD fallback.",
            )

        if solver_type == "deeponet":
            if not self._has_deeponet_artifact or self._deeponet_model is None:
                return _SolvePlan(
                    "fd_missing_artifact",
                    "DeepONet artifact unavailable; using FD fallback.",
                )
            if self._deeponet_n_points is not None and request.n_points != self._deeponet_n_points:
                return _SolvePlan(
                    "fd_resolution_mismatch",
                    "DeepONet resolution mismatch; using FD fallback.",
                )
            is_ood, ood_reason = self._check_ood(request=request)
            if is_ood:
                return _SolvePlan(
                    "fd_ood_fallback",
                    f"OOD detected ({ood_reason}); using FD fallback.",
                    ood_reason=ood_reason,
                )
            return _SolvePlan("deeponet_offline", "Using loaded DeepONet operator artifact.")

        raise RuntimeError(f"Unsupported solver type {solver_type!r}")

    def _execute_plan(self, request: _SolveRequest, plan: _SolvePlan) -> SolveResult:
        print(plan.route_reason)
        if plan.route == "fd":
            result = self._fd_path(
                request.pde_obj,
                request.n_points,
                request.print_every,
                max_iterations=self.options.fd.max_iterations,
                tolerance=self.options.fd.tolerance,
            )
            result.route_reason = plan.route_reason
            return result

        if plan.route == "fd_missing_artifact":
            return self._fallback_to_fd(
                pde_obj=request.pde_obj,
                n_points=request.n_points,
                print_every=request.print_every,
                method="fd_missing_artifact",
                route_reason=plan.route_reason,
            )

        if plan.route == "fd_resolution_mismatch":
            return self._fallback_to_fd(
                pde_obj=request.pde_obj,
                n_points=request.n_points,
                print_every=request.print_every,
                method="fd_resolution_mismatch",
                route_reason=plan.route_reason,
            )

        if plan.route == "fd_ood_fallback":
            return self._fallback_to_fd(
                pde_obj=request.pde_obj,
                n_points=request.n_points,
                print_every=request.print_every,
                method="fd_fallback",
                route_reason=plan.route_reason,
                is_ood=True,
                ood_reason=plan.ood_reason,
            )

        if plan.route == "fno_torchscript":
            result = self._jit_path(
                request.parsed_pde,
                request.bc_specs,
                request.pde_obj,
                request.source_fn,
                request.n_points,
            )
            result.route_reason = plan.route_reason
            return result

        if plan.route == "fno_offline":
            result = self._fno_path(
                pde_obj=request.pde_obj,
                n_points=request.n_points,
            )
            result.route_reason = plan.route_reason
            return result

        if plan.route == "deeponet_offline":
            result = self._deeponet_path(
                pde_obj=request.pde_obj,
                n_points=request.n_points,
            )
            result.route_reason = plan.route_reason
            return result

        raise RuntimeError(f"Unhandled solve route {plan.route!r}")

    # ------------------------------------------------------------------
    def _jit_path(
        self,
        parsed_pde: ParsedPDE,
        bc_specs: dict,
        pde_obj: GeneralPDE,
        source_fn,
        n_points: int,
    ) -> SolveResult:
        """Run a pre-traced TorchScript model using the shared conditional input."""
        grid = ConditionalGrid2D(n_points, parsed_pde, bc_specs, self.device)
        try:
            with torch.no_grad():
                out = self._jit_model(grid.input_grid)
                u = out.squeeze(0).squeeze(-1)  # (n, n)
        except RuntimeError as exc:
            if any(k in str(exc).lower() for k in ("size", "shape", "dimension")):
                print(
                    "TorchScript model resolution mismatch. "
                    "Falling back to FD solver..."
                )
                # Clear JIT model and use FD fallback
                self._jit_model = None
                self._has_fno_artifact = False
                return self._fallback_to_fd(
                    pde_obj=pde_obj,
                    n_points=n_points,
                    print_every=500,
                    method="fd_runtime_fallback",
                    route_reason="TorchScript model resolution mismatch; using FD fallback.",
                )
            raise

        residual = float(pde_obj.compute_pde_loss(u).item())
        bc_err   = float(pde_obj.compute_bc_loss(u, grid.x_1d, grid.y_1d).item())
        return SolveResult(
            u=u.cpu().numpy(),
            xx=grid.xx.cpu().numpy(),
            yy=grid.yy.cpu().numpy(),
            residual=residual,
            bc_error=bc_err,
            method="torchscript",
        )

    def _check_ood(
        self,
        *,
        request: _SolveRequest,
    ) -> tuple[bool, str]:
        """Return ``(is_ood, reason)`` for the current solve request."""
        if self._ood_detector is None:
            return False, ""
        return self._ood_detector.check(request.parsed_pde, request.bc_specs)

    def _fallback_to_fd(
        self,
        *,
        pde_obj: GeneralPDE,
        n_points: int,
        print_every: int,
        method: str,
        route_reason: str,
        is_ood: bool = False,
        ood_reason: str = "",
    ) -> SolveResult:
        result = self._fd_path(
            pde_obj,
            n_points,
            print_every,
            max_iterations=self.options.fd.max_iterations,
            tolerance=self.options.fd.tolerance,
        )
        result.method = method
        result.route_reason = route_reason
        result.is_ood = is_ood
        result.ood_reason = ood_reason
        return result

    def _offline_grid_path(
        self,
        *,
        model: nn.Module,
        pde_obj: GeneralPDE,
        n_points: int,
        method: str,
    ) -> SolveResult:
        """Run a preloaded operator model on the shared conditional grid."""
        grid = ConditionalGrid2D(n_points, pde_obj.parsed_pde, pde_obj.bc_specs, self.device)
        model.eval()
        with torch.no_grad():
            u = model(grid.input_grid).squeeze(0).squeeze(-1)
        residual = float(pde_obj.compute_pde_loss(u).item())
        bc_err = float(pde_obj.compute_bc_loss(u, grid.x_1d, grid.y_1d).item())
        return SolveResult(
            u=u.cpu().numpy(),
            xx=grid.xx.cpu().numpy(),
            yy=grid.yy.cpu().numpy(),
            residual=residual,
            bc_error=bc_err,
            method=method,
        )

    def _online_eval_to_result(
        self,
        *,
        pde_obj: GeneralPDE,
        eval_out: dict[str, torch.Tensor],
        n_points: int,
        method: str,
        history: list[float],
    ) -> SolveResult:
        """Convert online model evaluate() output into a SolveResult."""
        u = eval_out["u"].to(self.device)
        xx = eval_out["xx"].to(self.device)
        yy = eval_out["yy"].to(self.device)
        x_1d = torch.linspace(0, 1, n_points, device=self.device)
        y_1d = torch.linspace(0, 1, n_points, device=self.device)
        residual = float(pde_obj.compute_pde_loss(u).item())
        bc_err = float(pde_obj.compute_bc_loss(u, x_1d, y_1d).item())
        return SolveResult(
            u=u.cpu().numpy(),
            xx=xx.cpu().numpy(),
            yy=yy.cpu().numpy(),
            residual=residual,
            bc_error=bc_err,
            method=method,
            history=history,
        )

    # ------------------------------------------------------------------
    def _fno_path(
        self,
        pde_obj: GeneralPDE,
        n_points: int,
    ) -> SolveResult:
        if self._model is None:
            raise RuntimeError("FNO model is not loaded for offline inference.")
        return self._offline_grid_path(
            model=self._model,
            pde_obj=pde_obj,
            n_points=n_points,
            method="fno",
        )

    # ------------------------------------------------------------------
    def _fno_online_path(
        self,
        parsed_pde: ParsedPDE,
        bc_specs: dict,
        pde_obj: GeneralPDE,
        source_fn,
        n_points: int,
        lam_phys: float = 1.0,
        lam_bc: float = 10.0,
        n_epochs: int = 1000,
        lr: float = 1e-3,
        print_every: int = 500,
        pretrained_path: str | None = None,
    ) -> SolveResult:
        """Per-problem online FNO training, with optional warm-start from pretrained weights."""
        model = ConditionalFNO2D(
            parsed_pde=parsed_pde,
            bc_specs=bc_specs,
            ic_specs=pde_obj.ic_specs,
            n_points=n_points,
            lr=lr,
            lambda_physics=lam_phys,
            lambda_bc=lam_bc,
            width=self.width,
            n_modes=self.n_modes,
            n_layers=self.n_layers,
            pretrained_path=pretrained_path,
            device=str(self.device),
        )
        model.train(n_epochs=n_epochs, print_every=print_every)
        eval_out = model.evaluate(n_eval=n_points, t=0.0)
        history = model.history.get("total", [])
        return self._online_eval_to_result(
            pde_obj=pde_obj,
            eval_out=eval_out,
            n_points=n_points,
            method="fno",
            history=history,
        )

    # ------------------------------------------------------------------
    def _fd_path(
        self,
        pde_obj: GeneralPDE,
        n_points: int,
        print_every: int,
        max_iterations: int = 5000,
        tolerance: float = 1e-5,
    ) -> SolveResult:
        """Solve using finite difference iterative method."""
        fd_solver = _FDSolver(device=self.device)
        u, xx, yy, history = fd_solver.solve(
            pde_obj,
            n_points,
            max_iterations=max_iterations,
            tolerance=tolerance,
            print_every=print_every,
        )

        u_tensor = torch.tensor(u, dtype=torch.float32, device=self.device)
        residual = float(pde_obj.compute_pde_loss(u_tensor).item())
        x_1d = torch.linspace(0, 1, n_points, device=self.device)
        y_1d = torch.linspace(0, 1, n_points, device=self.device)
        bc_err = float(pde_obj.compute_bc_loss(u_tensor, x_1d, y_1d).item())

        return SolveResult(
            u=u,
            xx=xx,
            yy=yy,
            residual=residual,
            bc_error=bc_err,
            method="fd",
            history=history,
        )

    # ------------------------------------------------------------------
    def _deeponet_path(
        self,
        pde_obj: GeneralPDE,
        n_points: int,
    ) -> SolveResult:
        if self._deeponet_model is None:
            raise RuntimeError("DeepONet model is not loaded for offline inference.")
        return self._offline_grid_path(
            model=self._deeponet_model,
            pde_obj=pde_obj,
            n_points=n_points,
            method="deeponet",
        )
