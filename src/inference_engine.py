"""Generalized inference engine for 2-D scalar PDEs.

Public API
----------
    from inference_engine import InferenceEngine, SolverOption, FNOConfig, GridConfig

    option = SolverOption(
        solver_type="fno",
        fno=FNOConfig(
            model_path="pretrained_models/fno.pt",
            manifest_path="pretrained_models/fno_manifest.npz",
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

FD path  (solver_type="fd" or no FNO model available)
    Finite difference iterative solver using Jacobi relaxation.

PINN path  (solver_type="pinn")
    Trains a SharedConditionalPINN2D (optional warm-start from a state-dict).

Offline training
----------------
    Use ``src/trainer.py`` to build ``fno.pt`` and ``fno_manifest.npz``:
        python src/trainer.py fno --samples 2000 --epochs 30
        python src/trainer.py pinn --samples 200 --steps-per-problem 3 --n-epochs 20
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

from models.checkpoints import load_model_weights, read_checkpoint_arch
from models.conditional_inputs import ConditionalGrid2D
from models.conditional_solvers import ConditionalFNO2D, SharedConditionalPINN2D, _PointwiseConditionalPINNNet
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
    manifest_path: str | None = None


@dataclass
class PINNConfig:
    model_path: str | None = None
    manifest_path: str | None = None
    hidden: int = 64
    n_layers: int = 4
    epochs: int = 1000          # Adam warm-up steps (L-BFGS handles final convergence)
    n_lbfgs_steps: int = 200   # L-BFGS refinement steps after Adam warm-up
    early_stop_tol: float = 1e-6  # stop early when total loss drops below this
    lr: float = 1e-3
    lambda_physics: float = 1.0
    lambda_bc: float = 10.0
    print_every: int = 500


@dataclass
class SolverOption:
    """Top-level solver configuration used by InferenceEngine.

    Defaults route to FD with conservative settings. UI can selectively
    override fields for FNO/PINN/FD runs.
    """

    solver_type: Literal["fd", "fno", "pinn"] = "fd"
    grid: GridConfig = field(default_factory=GridConfig)
    fd: FDConfig = field(default_factory=FDConfig)
    fno: FNOConfig = field(default_factory=FNOConfig)
    pinn: PINNConfig = field(default_factory=PINNConfig)
    device: str | None = None

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
    ):
        self.u = u                    # (n, n) numpy array — solution field
        self.xx = xx                  # (n, n) x-coordinates
        self.yy = yy                  # (n, n) y-coordinates
        self.residual = residual      # mean squared PDE residual
        self.bc_error = bc_error      # mean squared BC error
        self.method = method          # "fno" | "fd" | "pinn" | "torchscript"
        self.history = history or []  # loss per epoch (pinn / fd)
        self.is_ood = is_ood          # True when OOD detected and FD fallback used
        self.ood_reason = ood_reason  # human-readable OOD explanation

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
        grid = ConditionalGrid2D(n_points, bc_specs, source_fn, self.device)
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


def _is_shared_pinn_checkpoint(path: str | None, device: torch.device) -> bool:
    """Return True if *path* is a shared-PINN metadata-envelope checkpoint."""
    if path is None:
        return False
    return read_checkpoint_arch(path, device) == "shared_pinn"


# ---------------------------------------------------------------------------
# Main inference engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """Load a pre-trained conditional FNO and solve arbitrary 2-D PDEs.

    Parameters
    ----------
    solver_option : SolverOption
        Full solver configuration.  Use the :class:`SolverOption`,
        :class:`FNOConfig`, :class:`PINNConfig`, and :class:`FDConfig`
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
        self._pinn_model_path = self.options.pinn.model_path
        self._pinn_hidden = self.options.pinn.hidden
        self._pinn_layers = self.options.pinn.n_layers

        self._model: FNO2DModel | None = None
        self._jit_model = None  # set when a TorchScript archive is loaded
        self._ood_detector: OODDetector | None = None
        self._pinn_net: _PointwiseConditionalPINNNet | None = None

        if self.options.solver_type == "fno":
            self._model = FNO2DModel(
                in_channels=7,
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
                        print(f"Loaded TorchScript FNO from {p}")
                    else:
                        load_model_weights(self._model, str(p), self.device)
                        self._model.eval()
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
                    print(f"FNO model path {p} not found; solve() will fall back to online training.")
            manifest = self.options.fno.manifest_path
            if manifest is not None and OODDetector.is_available(manifest):
                self._ood_detector = OODDetector(manifest)
                print(f"OOD detector loaded from {manifest}")
        elif self.options.solver_type == "pinn":
            self._pinn_net = _PointwiseConditionalPINNNet(
                in_channels=7,
                hidden=self.options.pinn.hidden,
                n_layers=self.options.pinn.n_layers,
            ).to(self.device)
            if self.options.pinn.model_path is not None:
                p = Path(self.options.pinn.model_path)
                if p.exists():
                    load_model_weights(
                        self._pinn_net,
                        self.options.pinn.model_path,
                        self.device,
                        skip_if_torchscript=True,
                    )
                    self._pinn_net.eval()
                    print(f"Loaded PINN model from {self.options.pinn.model_path}")
                else:
                    print(f"PINN model path {p} not found; solve() will fall back to online training.")
            manifest = self.options.pinn.manifest_path
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
        lambda_physics: float | None = None,
        lambda_bc: float | None = None,
        fno_epochs: int | None = None,
        fno_lr: float | None = None,
        pinn_epochs: int | None = None,
        pinn_lr: float | None = None,
        print_every: int | None = None,
    ) -> SolveResult:
        """Parse PDE + BCs and return the solution field.

        Parameters
        ----------
        pde_str         : PDE string, e.g. ``"u_xx + u_yy = sin(pi*x)"``.
        bc_dict         : per-wall BCs dict.
        ic_str          : initial condition for time-dependent PDEs (optional).
        n_points        : grid resolution (n × n); defaults to GridConfig.n_points.
        lambda_physics  : physics-loss weight for PINN.
        lambda_bc       : BC-loss weight for PINN.
        fno_epochs      : epochs for online FNO training (overrides FNO-online default).
        fno_lr          : learning rate for online FNO training.
        pinn_epochs     : Adam epochs for the PINN path (overrides PINNConfig).
        pinn_lr         : learning rate for the PINN path (overrides PINNConfig).
        print_every     : log interval for iterative solvers.

        Returns
        -------
        SolveResult
            ``result.is_ood == True`` when an FNO OOD detection triggered FD fallback.
        """
        # --- 1. Parse ---
        parsed_pde = parse_pde(pde_str)
        bc_specs   = parse_bc(bc_dict)

        if parsed_pde.is_time_dependent:
            raise NotImplementedError(
                "Time-dependent PDEs are not supported by the current inference paths. "
                "The active FNO/PINN/FD solvers only handle steady-state problems."
            )

        ic_specs = None
        if ic_str and parsed_pde.is_time_dependent:
            from pde_parser import parse_ic
            ic_specs = parse_ic(ic_str)

        pde_obj   = GeneralPDE(parsed_pde, bc_specs, ic_specs)
        source_fn = parsed_pde.rhs_fn()

        # --- 2. Resolve effective runtime options ---
        n_points_eff   = int(n_points if n_points is not None else self.options.grid.n_points)
        fno_cfg        = self.options.fno
        pinn_cfg       = self.options.pinn
        fd_cfg         = self.options.fd
        solver_type    = self.options.solver_type

        lam_phys_eff = float(
            lambda_physics if lambda_physics is not None else (
                fno_cfg.lambda_physics if solver_type == "fno" else pinn_cfg.lambda_physics
            )
        )
        lam_bc_eff = float(
            lambda_bc if lambda_bc is not None else (
                fno_cfg.lambda_bc if solver_type == "fno" else pinn_cfg.lambda_bc
            )
        )
        # Keep backward compatibility: when fno_* are omitted, fall back to
        # legacy pinn_* overrides for the FNO online path.
        fno_epochs_eff  = int(
            fno_epochs if fno_epochs is not None else (
                pinn_epochs if pinn_epochs is not None else pinn_cfg.epochs
            )
        )
        fno_lr_eff      = float(
            fno_lr if fno_lr is not None else (
                pinn_lr if pinn_lr is not None else pinn_cfg.lr
            )
        )
        pinn_epochs_eff = int(pinn_epochs if pinn_epochs is not None else pinn_cfg.epochs)
        pinn_lr_eff     = float(pinn_lr if pinn_lr is not None else pinn_cfg.lr)
        print_every_eff = int(
            print_every if print_every is not None else (
                fno_cfg.print_every  if solver_type == "fno"  else
                pinn_cfg.print_every if solver_type == "pinn" else
                fd_cfg.print_every
            )
        )

        # --- 3. Route by configured solver type ---
        if solver_type == "fno":
            # File existence check: if no model file, fall back to online FNO training
            if fno_cfg.model_path is None or not Path(fno_cfg.model_path).exists():
                print("FNO model file not found; using online FNO training...")
                return self._fno_online_path(
                    parsed_pde, bc_specs, pde_obj, source_fn, n_points_eff,
                    lam_phys=lam_phys_eff, lam_bc=lam_bc_eff,
                    n_epochs=fno_epochs_eff, lr=fno_lr_eff,
                    print_every=print_every_eff,
                    pretrained_path=fno_cfg.model_path,
                )
            # OOD gate: check before running the FNO forward pass
            if self._ood_detector is not None:
                is_ood, ood_reason = self._ood_detector.check(parsed_pde, bc_specs)
                if is_ood:
                    print(f"OOD detected ({ood_reason}); falling back to FD solver.")
                    fd_result = self._fd_path(
                        pde_obj, n_points_eff, print_every_eff,
                        max_iterations=fd_cfg.max_iterations,
                        tolerance=fd_cfg.tolerance,
                    )
                    fd_result.is_ood    = True
                    fd_result.ood_reason = ood_reason
                    fd_result.method    = "fd_fallback"
                    return fd_result

            if self._jit_model is not None:
                return self._jit_path(parsed_pde, bc_specs, pde_obj, source_fn, n_points_eff)
            return self._fast_path(parsed_pde, bc_specs, pde_obj, source_fn, n_points_eff)

        if solver_type == "pinn":
            # File existence check: if no model file, use online PINN training
            if pinn_cfg.model_path is None or not Path(pinn_cfg.model_path).exists():
                print("PINN model file not found; using online PINN training...")
                return self._pinn_online_path(
                    pde_obj=pde_obj,
                    n_points=n_points_eff,
                    n_epochs=pinn_epochs_eff,
                    lr=pinn_lr_eff,
                    lam_phys=lam_phys_eff,
                    lam_bc=lam_bc_eff,
                    print_every=print_every_eff,
                    n_lbfgs_steps=pinn_cfg.n_lbfgs_steps,
                    early_stop_tol=pinn_cfg.early_stop_tol,
                )
            # OOD gate: check before offline PINN inference
            if self._ood_detector is not None:
                is_ood, ood_reason = self._ood_detector.check(parsed_pde, bc_specs)
                if is_ood:
                    print(f"OOD detected ({ood_reason}); falling back to FD solver.")
                    fd_result = self._fd_path(
                        pde_obj, n_points_eff, print_every_eff,
                        max_iterations=fd_cfg.max_iterations,
                        tolerance=fd_cfg.tolerance,
                    )
                    fd_result.is_ood    = True
                    fd_result.ood_reason = ood_reason
                    fd_result.method    = "fd_fallback"
                    return fd_result
            print("Using PINN solver (offline)...")
            return self._pinn_path(
                pde_obj=pde_obj,
                n_points=n_points_eff,
                n_epochs=pinn_epochs_eff,
                lr=pinn_lr_eff,
                lam_phys=lam_phys_eff,
                lam_bc=lam_bc_eff,
                print_every=print_every_eff,
                n_lbfgs_steps=pinn_cfg.n_lbfgs_steps,
                early_stop_tol=pinn_cfg.early_stop_tol,
            )

        print("Using finite difference solver...")
        return self._fd_path(
            pde_obj,
            n_points_eff,
            print_every_eff,
            max_iterations=fd_cfg.max_iterations,
            tolerance=fd_cfg.tolerance,
        )

    # ------------------------------------------------------------------
    def _jit_path(
        self,
        parsed_pde: ParsedPDE,
        bc_specs: dict,
        pde_obj: GeneralPDE,
        source_fn,
        n_points: int,
    ) -> SolveResult:
        """Run a pre-traced TorchScript model using conditional 7-channel input."""
        grid = ConditionalGrid2D(n_points, bc_specs, source_fn, self.device)
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
                return self._fd_path(
                    pde_obj,
                    n_points,
                    print_every=500,
                    max_iterations=self.options.fd.max_iterations,
                    tolerance=self.options.fd.tolerance,
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

    # ------------------------------------------------------------------
    def _fast_path(
        self,
        parsed_pde: ParsedPDE,
        bc_specs: dict,
        pde_obj: GeneralPDE,
        source_fn,
        n_points: int,
    ) -> SolveResult:
        solver = _FNOSolver(
            model=self._model,
            width=self.width,
            n_modes=self.n_modes,
            n_layers=self.n_layers,
            device=self.device,
        )
        return solver.solve(
            parsed_pde=parsed_pde,
            bc_specs=bc_specs,
            pde_obj=pde_obj,
            source_fn=source_fn,
            n_points=n_points,
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
        u       = eval_out["u"].to(self.device)
        xx      = eval_out["xx"].to(self.device)
        yy      = eval_out["yy"].to(self.device)
        history = model.history.get("total", [])
        x_1d = torch.linspace(0, 1, n_points, device=self.device)
        y_1d = torch.linspace(0, 1, n_points, device=self.device)
        residual = float(pde_obj.compute_pde_loss(u).item())
        bc_err   = float(pde_obj.compute_bc_loss(u, x_1d, y_1d).item())
        return SolveResult(
            u=u.cpu().numpy(),
            xx=xx.cpu().numpy(),
            yy=yy.cpu().numpy(),
            residual=residual,
            bc_error=bc_err,
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
    def _pinn_path(
        self,
        pde_obj: GeneralPDE,
        n_points: int,
        n_epochs: int,
        lr: float,
        lam_phys: float,
        lam_bc: float,
        print_every: int,
        n_lbfgs_steps: int = 200,
        early_stop_tol: float = 1e-6,
    ) -> SolveResult:
        # Offline fast path: single forward pass through pre-loaded PINN weights
        if self._pinn_net is not None:
            source_fn = pde_obj.parsed_pde.rhs_fn()
            grid = ConditionalGrid2D(n_points, pde_obj.bc_specs, source_fn, self.device)
            self._pinn_net.eval()
            with torch.no_grad():
                u = self._pinn_net(grid.input_grid).squeeze(0).squeeze(-1)
            residual = float(pde_obj.compute_pde_loss(u).item())
            bc_err   = float(pde_obj.compute_bc_loss(u, grid.x_1d, grid.y_1d).item())
            return SolveResult(
                u=u.cpu().numpy(),
                xx=grid.xx.cpu().numpy(),
                yy=grid.yy.cpu().numpy(),
                residual=residual,
                bc_error=bc_err,
                method="pinn",
            )
        # _pinn_net is None (load failed or was suppressed); fall through to online training.
        print("PINN net not loaded; falling back to online PINN training...")
        return self._pinn_online_path(
            pde_obj=pde_obj,
            n_points=n_points,
            n_epochs=n_epochs,
            lr=lr,
            lam_phys=lam_phys,
            lam_bc=lam_bc,
            print_every=print_every,
            n_lbfgs_steps=n_lbfgs_steps,
            early_stop_tol=early_stop_tol,
        )

    # ------------------------------------------------------------------
    def _pinn_online_path(
        self,
        pde_obj: GeneralPDE,
        n_points: int,
        n_epochs: int,
        lr: float,
        lam_phys: float,
        lam_bc: float,
        print_every: int,
        n_lbfgs_steps: int = 200,
        early_stop_tol: float = 1e-6,
    ) -> SolveResult:
        """Per-problem online PINN training when no pre-trained weights are available."""
        model = SharedConditionalPINN2D(
            parsed_pde=pde_obj.parsed_pde,
            bc_specs=pde_obj.bc_specs,
            ic_specs=pde_obj.ic_specs,
            n_points=n_points,
            lr=lr,
            lambda_physics=lam_phys,
            lambda_bc=lam_bc,
            hidden=self._pinn_hidden,
            n_layers=self._pinn_layers,
            pretrained_path=None,
            device=str(self.device),
        )
        model.train(
            n_epochs=n_epochs,
            print_every=print_every,
            n_lbfgs_steps=n_lbfgs_steps,
            early_stop_tol=early_stop_tol,
        )
        eval_out = model.evaluate(n_eval=n_points, t=0.0)
        u       = eval_out["u"].to(self.device)
        xx      = eval_out["xx"].to(self.device)
        yy      = eval_out["yy"].to(self.device)
        history = model.history.get("total", [])

        x_1d = torch.linspace(0, 1, n_points, device=self.device)
        y_1d = torch.linspace(0, 1, n_points, device=self.device)
        residual = float(pde_obj.compute_pde_loss(u).item())
        bc_err   = float(pde_obj.compute_bc_loss(u, x_1d, y_1d).item())

        return SolveResult(
            u=u.cpu().numpy(),
            xx=xx.cpu().numpy(),
            yy=yy.cpu().numpy(),
            residual=residual,
            bc_error=bc_err,
            method="pinn",
            history=history,
        )
