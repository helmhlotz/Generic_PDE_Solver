"""Offline training pipeline for conditional FNO/PINN solvers.

For FNO, this module now supports an explicit two-stage workflow:
1) data generation via Latin Hypercube Sampling + finite-difference solves
2) hybrid training from generated datasets (supervised + physics + BC loss)

Usage (CLI)
-----------
    # Generate FNO training data with LHS + FD solver
    python trainer.py fno-generate --samples 5000 --dataset-path pretrained_models/fno_train_data.npz

    # Train FNO on generated data using hybrid loss
    python trainer.py fno-train --train-dataset pretrained_models/fno_train_data.npz \
                                --val-dataset pretrained_models/fno_val_data.npz \
                                --epochs 30

    # Train PINN: 5000-problem pool, 3 steps/problem, 20 epochs
    python trainer.py pinn --samples 5000 --steps-per-problem 3 --n-epochs 20

    # Backward-compatible one-shot FNO pipeline (generate + train)
    python trainer.py fno --samples 5000 --epochs 30

    # Evaluate trained FNO against FD ground truth on a held-out test set
    python trainer.py fno-test --test-dataset pretrained_models/fno_val_data.npz \
                               --fno-path pretrained_models/fno.pt

    # Evaluate trained PINN against FD ground truth on a held-out test set
    python trainer.py pinn-test --test-dataset pretrained_models/fno_val_data.npz \
                                --pinn-path pretrained_models/pinn.pt

Public API
----------
    from trainer import SharedFDDataGenerator, HybridFNOTrainer, PINNTrainer
    from pde_space import PDESpaceConfig

    generator = SharedFDDataGenerator()
    generator.generate(config=PDESpaceConfig(), n_samples=5000, save_path="pretrained_models/fno_train_data.npz")

    trainer = HybridFNOTrainer(n_points=32)
    trainer.train(train_dataset_path="pretrained_models/fno_train_data.npz")
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure src/ is in path when invoked directly
sys.path.insert(0, str(Path(__file__).parent))

from models.conditional_inputs import ConditionalGrid2D
from models.conditional_solvers import _PointwiseConditionalPINNNet
from models.fno_model import FNO2DModel
from pde_space import PDESpaceConfig, LHSSampler
from ood_detector import PDEFeaturizer, OODDetector
from pde_parser import build_fd_residual, parse_bc, parse_pde
from physics.pde_helpers import GeneralPDE


# ---------------------------------------------------------------------------
# Chunk size for fault-tolerant generation
# ---------------------------------------------------------------------------

_CHUNK_SIZE = 200


# ---------------------------------------------------------------------------
# Top-level picklable helpers for ProcessPoolExecutor
# ---------------------------------------------------------------------------

def _solve_fd_standalone(
    pde_obj: GeneralPDE,
    n_points: int,
    device: torch.device,
    max_iterations: int,
    tolerance: float,
) -> torch.Tensor | None:
    """Standalone FD solver (not a method) so it can run in worker processes."""
    x_1d = torch.linspace(0, 1, n_points, device=device)
    y_1d = torch.linspace(0, 1, n_points, device=device)
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
    if (diag.abs() < 1e-14).any():
        return None

    for _ in range(max_iterations):
        u_old = u.clone()
        residual = residual_fn(u)
        u[1:-1, 1:-1] = u[1:-1, 1:-1] - residual / diag
        u = pde_obj.apply_boundary_conditions(u, x_1d, y_1d)
        if torch.isnan(u).any() or torch.isinf(u).any():
            return None
        if torch.max(torch.abs(u - u_old)).item() < tolerance:
            break

    return u.detach()


def _solve_one(args: tuple) -> dict | None:
    """Top-level picklable worker for ProcessPoolExecutor.

    Always uses CPU so CUDA contexts are never shared across processes.
    All arguments must be plain picklable types (no torch.device, no lambdas).
    """
    prob, n_points, max_iters, tol = args
    try:
        parsed_pde = parse_pde(prob["pde_str"])
        bc_specs = parse_bc(prob["bc_dict"])
        source_fn = parsed_pde.rhs_fn()
        device = torch.device("cpu")
        grid = ConditionalGrid2D(n_points, bc_specs, source_fn, device)
        pde_obj = GeneralPDE(parsed_pde, bc_specs)
        u_fd = _solve_fd_standalone(pde_obj, n_points, device, max_iters, tol)
        if u_fd is None:
            return None
        feat = PDEFeaturizer.featurize(parsed_pde, bc_specs)
        return {
            "input": grid.input_grid.squeeze(0).cpu().numpy().astype(np.float32),
            "target": u_fd.cpu().numpy().astype(np.float32),
            "pde_str": prob["pde_str"],
            "bc_json": json.dumps(prob["bc_dict"], sort_keys=True),
            "feat": feat,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# FNO data generation + training
# ---------------------------------------------------------------------------

class SharedFDDataGenerator:
    """Generate shared FD-supervised datasets for both FNO and PINN."""

    GRID_POINTS = 64

    def __init__(
        self,
        n_points: int = GRID_POINTS,
        device: str | None = None,
    ) -> None:
        self.n_points = n_points
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def _solve_fd(
        self,
        pde_obj: GeneralPDE,
        max_iterations: int,
        tolerance: float,
    ) -> torch.Tensor | None:
        """Solve one PDE with Jacobi FD; return u (n,n) or None if failed."""
        n_points = self.n_points
        x_1d = torch.linspace(0, 1, n_points, device=self.device)
        y_1d = torch.linspace(0, 1, n_points, device=self.device)
        u = torch.zeros((n_points, n_points), device=self.device, dtype=torch.float32)
        u = pde_obj.apply_boundary_conditions(u, x_1d, y_1d)
        residual_fn = build_fd_residual(pde_obj.parsed_pde)

        parsed = pde_obj.parsed_pde
        h = 1.0 / (n_points - 1)
        diag = torch.zeros((n_points - 2, n_points - 2), device=self.device, dtype=torch.float32)
        if parsed.a != 0.0:
            diag = diag + parsed.a * (-2.0 / h**2)
        if parsed.b != 0.0:
            diag = diag + parsed.b * (-2.0 / h**2)
        if parsed.f != 0.0:
            diag = diag + parsed.f
        if (diag.abs() < 1e-14).any():
            return None

        for _ in range(max_iterations):
            u_old = u.clone()
            residual = residual_fn(u)
            u[1:-1, 1:-1] = u[1:-1, 1:-1] - residual / diag
            u = pde_obj.apply_boundary_conditions(u, x_1d, y_1d)

            if torch.isnan(u).any() or torch.isinf(u).any():
                return None
            if torch.max(torch.abs(u - u_old)).item() < tolerance:
                break

        return u.detach()

    def generate(
        self,
        config: PDESpaceConfig | None = None,
        n_samples: int = 5000,
        seed: int = 42,
        save_path: str = "pretrained_models/fno_train_data.npz",
        manifest_path: str | None = "pretrained_models/fno_manifest.npz",
        max_iterations: int = 5000,
        tolerance: float = 1e-5,
        print_every: int = 200,
        n_workers: int | None = None,
        chunk_size: int = _CHUNK_SIZE,
    ) -> int:
        """Generate and save dataset with fields input_grid and fd_target.

        Uses ProcessPoolExecutor across CPU cores with chunk-based save/resume
        for fault tolerance.  Each chunk is written to ``<save_path>_chunk_NNNNNN.npz``
        so an interrupted run can be resumed without re-computing finished chunks.
        """
        if config is None:
            config = PDESpaceConfig()

        sampler = LHSSampler(config)
        problems = sampler.generate(n_samples=n_samples, seed=seed)

        n_workers = n_workers or multiprocessing.cpu_count()

        inputs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        pde_strs: list[str] = []
        bc_json: list[str] = []
        feats: list[np.ndarray] = []

        total_done = 0
        for chunk_start in range(0, len(problems), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(problems))
            chunk_path = str(save_path).replace(".npz", f"_chunk_{chunk_start:06d}.npz")

            # Resume: load an already-computed chunk instead of re-solving.
            if Path(chunk_path).exists():
                print(f"  Loading existing chunk {chunk_start}–{chunk_end} ...")
                c = np.load(chunk_path, allow_pickle=True)
                inputs.extend(c["inputs"])
                targets.extend(c["targets"])
                pde_strs.extend(c["pde_strs"].tolist())
                bc_json.extend(c["bc_dict_json"].tolist())
                feats.extend(c["feats"])
                total_done += len(c["inputs"])
                continue

            chunk_problems = problems[chunk_start:chunk_end]
            args_list = [
                (p, self.n_points, max_iterations, tolerance)
                for p in chunk_problems
            ]

            chunk_results: list[dict] = []
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = {ex.submit(_solve_one, a): i for i, a in enumerate(args_list)}
                for fut in as_completed(futures):
                    r = fut.result()
                    if r is not None:
                        chunk_results.append(r)

            if chunk_results:
                c_inputs = np.stack([r["input"] for r in chunk_results])
                c_targets = np.stack([r["target"] for r in chunk_results])
                c_pde_strs = np.array([r["pde_str"] for r in chunk_results], dtype=object)
                c_bc_json = np.array([r["bc_json"] for r in chunk_results], dtype=object)
                c_feats = np.stack([r["feat"] for r in chunk_results])

                Path(chunk_path).parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    chunk_path,
                    inputs=c_inputs,
                    targets=c_targets,
                    pde_strs=c_pde_strs,
                    bc_dict_json=c_bc_json,
                    feats=c_feats,
                )

                inputs.extend(c_inputs)
                targets.extend(c_targets)
                pde_strs.extend(c_pde_strs.tolist())
                bc_json.extend(c_bc_json.tolist())
                feats.extend(c_feats)
                total_done += len(chunk_results)

            print(f"  generated {total_done}/{n_samples} samples")

        if not inputs:
            raise RuntimeError("No valid FD samples generated; dataset is empty.")

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path,
            inputs=np.stack(inputs),
            targets=np.stack(targets),
            pde_strs=np.array(pde_strs, dtype=object),
            bc_dict_json=np.array(bc_json, dtype=object),
            n_points=np.array([self.n_points], dtype=np.int32),
        )
        print(f"Saved dataset: {save_path} ({len(inputs)} samples)")

        if manifest_path is not None and feats:
            OODDetector.build_manifest(np.stack(feats), manifest_path=manifest_path)
            print(f"Saved OOD manifest: {manifest_path}")

        return len(inputs)


# Backward-compatible alias
FNODataGenerator = SharedFDDataGenerator


class HybridFNOTrainer:
    """Train FNO on pre-generated datasets with hybrid (data+physics+BC) loss."""

    def __init__(
        self,
        n_points: int = 32,
        width: int = 32,
        n_modes: tuple[int, int] = (12, 12),
        n_layers: int = 4,
        lr: float = 5e-4,
        lam_data: float = 1.0,
        lam_phys: float = 1.0,
        lam_bc: float = 10.0,
        device: str | None = None,
    ) -> None:
        self.n_points  = n_points
        self.lam_data  = lam_data
        self.lam_phys  = lam_phys
        self.lam_bc    = lam_bc
        self.device    = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = FNO2DModel(
            in_channels=7,
            out_channels=1,
            width=width,
            n_modes=n_modes,
            n_layers=n_layers,
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler: optim.lr_scheduler.LRScheduler | None = None

    def _load_examples(self, dataset_path: str) -> list[dict[str, Any]]:
        """Load dataset file and reconstruct training examples."""
        npz = np.load(dataset_path, allow_pickle=True)
        inputs = npz["inputs"]
        targets = npz["targets"]
        pde_strs = npz["pde_strs"]
        bc_json = npz["bc_dict_json"]

        examples: list[dict[str, Any]] = []
        for i in range(len(inputs)):
            n_points = int(inputs[i].shape[0])
            parsed_pde = parse_pde(str(pde_strs[i]))
            bc_dict = json.loads(str(bc_json[i]))
            bc_specs = parse_bc(bc_dict)
            pde_obj = GeneralPDE(parsed_pde, bc_specs)
            x_1d = torch.linspace(0, 1, n_points, device=self.device)
            y_1d = torch.linspace(0, 1, n_points, device=self.device)
            examples.append(
                {
                    "input": torch.tensor(inputs[i], dtype=torch.float32, device=self.device),
                    "target": torch.tensor(targets[i], dtype=torch.float32, device=self.device),
                    "pde_obj": pde_obj,
                    "x_1d": x_1d,
                    "y_1d": y_1d,
                }
            )
        return examples

    def _eval_examples(self, examples: list[dict[str, Any]]) -> float:
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for ex in examples:
                u_pred = self.model(ex["input"].unsqueeze(0)).squeeze(0).squeeze(-1)
                data_loss = torch.mean((u_pred - ex["target"]) ** 2)
                phys_loss = ex["pde_obj"].compute_pde_loss(u_pred)
                bc_loss = ex["pde_obj"].compute_bc_loss(u_pred, ex["x_1d"], ex["y_1d"])
                total += (
                    self.lam_data * data_loss
                    + self.lam_phys * phys_loss
                    + self.lam_bc * bc_loss
                ).item()
        self.model.train()
        return total / max(len(examples), 1)

    def train(
        self,
        train_dataset_path: str,
        val_dataset_path: str | None = None,
        n_epochs: int = 30,
        save_path: str = "pretrained_models/fno.pt",
        print_every: int = 200,
        eval_every: int = 1,
        seed: int = 42,
    ) -> None:
        """Train FNO from generated dataset using hybrid supervision + physics."""
        rng = np.random.default_rng(seed)
        train_examples = self._load_examples(train_dataset_path)
        val_examples = self._load_examples(val_dataset_path) if val_dataset_path else []

        total_steps = len(train_examples) * n_epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps, 1), eta_min=1e-5
        )
        print(
            f"Training hybrid FNO on {len(train_examples)} examples "
            f"for {n_epochs} epochs (T_max={total_steps})..."
        )

        best_val_loss: float = float("inf")
        best_state: dict[str, Any] | None = None
        step = 0

        for epoch in range(1, n_epochs + 1):
            order = rng.permutation(len(train_examples))
            running = 0.0
            for idx in order:
                ex = train_examples[int(idx)]
                self.optimizer.zero_grad()
                u_pred = self.model(ex["input"].unsqueeze(0)).squeeze(0).squeeze(-1)
                data_loss = torch.mean((u_pred - ex["target"]) ** 2)
                phys_loss = ex["pde_obj"].compute_pde_loss(u_pred)
                bc_loss = ex["pde_obj"].compute_bc_loss(u_pred, ex["x_1d"], ex["y_1d"])
                loss = (
                    self.lam_data * data_loss
                    + self.lam_phys * phys_loss
                    + self.lam_bc * bc_loss
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                step += 1
                running += loss.item()
                if step % print_every == 0:
                    print(
                        f"  step {step:>7d} | data={data_loss.item():.3e} "
                        f"phys={phys_loss.item():.3e} bc={bc_loss.item():.3e} "
                        f"total={loss.item():.3e}"
                    )

            avg_train = running / max(len(train_examples), 1)
            if epoch % eval_every == 0:
                val_loss = self._eval_examples(val_examples) if val_examples else avg_train
                improved = val_loss < best_val_loss
                if improved:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                marker = " *" if improved else ""
                print(
                    f"Epoch {epoch:3d}/{n_epochs} | train={avg_train:.4e} "
                    f"val={val_loss:.4e} best={best_val_loss:.4e}{marker}"
                )

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Saved final hybrid FNO weights → {save_path}")

        if best_state is not None:
            best_path = str(save_path).replace(".pt", "_best.pt")
            torch.save(best_state, best_path)
            print(f"Saved best  hybrid FNO weights → {best_path}")
    # ------------------------------------------------------------------
    def test(
        self,
        test_dataset_path: str,
        print_every: int = 10,
    ) -> dict[str, float]:
        """Evaluate the trained FNO against FD ground truth on a held-out test set.

        Loads the dataset, runs the model in eval mode, and computes per-sample
        error metrics comparing FNO predictions to the stored FD targets.

        Parameters
        ----------
        test_dataset_path : Path to .npz test dataset (same format as training data).
        print_every       : Print per-sample stats every N samples.

        Returns
        -------
        summary : dict with aggregate metrics (rel_l2, rmse, max_err, bc_error,
                  pde_residual; mean/std/p50/p90/max where applicable).
        """
        examples = self._load_examples(test_dataset_path)
        if not examples:
            raise RuntimeError(f"No examples loaded from {test_dataset_path!r}")

        print(f"Testing FNO on {len(examples)} samples from {test_dataset_path!r} ...")

        rel_l2s: list[float] = []
        max_errs: list[float] = []
        rmses: list[float] = []
        bc_errs: list[float] = []
        pde_ress: list[float] = []

        self.model.eval()
        with torch.no_grad():
            for i, ex in enumerate(examples):
                u_pred = self.model(ex["input"].unsqueeze(0)).squeeze(0).squeeze(-1)
                u_fd   = ex["target"]

                diff    = u_pred - u_fd
                fd_norm = torch.norm(u_fd) + 1e-12
                rel_l2  = float(torch.norm(diff) / fd_norm)
                max_err = float(torch.max(torch.abs(diff)))
                rmse    = float(torch.sqrt(torch.mean(diff ** 2)))
                bc_err  = float(ex["pde_obj"].compute_bc_loss(u_pred, ex["x_1d"], ex["y_1d"]))
                pde_res = float(ex["pde_obj"].compute_pde_loss(u_pred))

                rel_l2s.append(rel_l2)
                max_errs.append(max_err)
                rmses.append(rmse)
                bc_errs.append(bc_err)
                pde_ress.append(pde_res)

                if (i + 1) % print_every == 0:
                    print(
                        f"  [{i + 1:>5d}/{len(examples)}] "
                        f"rel_l2={rel_l2:.3e}  max={max_err:.3e}  rmse={rmse:.3e}"
                    )
        self.model.train()

        arr = np.array(rel_l2s)
        summary = {
            "n_samples":     len(examples),
            "rel_l2_mean":   float(np.mean(arr)),
            "rel_l2_std":    float(np.std(arr)),
            "rel_l2_p50":    float(np.median(arr)),
            "rel_l2_p90":    float(np.percentile(arr, 90)),
            "rel_l2_max":    float(np.max(arr)),
            "rmse_mean":     float(np.mean(rmses)),
            "max_err_mean":  float(np.mean(max_errs)),
            "max_err_max":   float(np.max(max_errs)),
            "bc_error_mean": float(np.mean(bc_errs)),
            "pde_res_mean":  float(np.mean(pde_ress)),
        }
        _print_test_summary(summary, "FNO")
        return summary

# ---------------------------------------------------------------------------
# PINN trainer
# ---------------------------------------------------------------------------

class PINNTrainer:
    """Train a shared-weights conditional PINN across LHS-sampled problems.

    This mirrors ``HybridFNOTrainer`` conceptually: a single PINN checkpoint is
    updated over many PDE instances so it can learn reusable patterns across
    the sampled PDE family instead of solving each problem from scratch.

    Parameters
    ----------
    n_points  : grid resolution
    hidden    : hidden units per MLP layer
    n_layers  : number of MLP layers
    lr        : Adam learning rate
    lam_phys  : physics-loss weight
    lam_bc    : BC-loss weight
    device    : torch device string
    """

    def __init__(
        self,
        n_points: int = 32,
        hidden: int = 64,
        n_layers: int = 4,
        lr: float = 1e-3,
        lam_phys: float = 1.0,
        lam_bc: float = 10.0,
        device: str | None = None,
    ) -> None:
        self.n_points  = n_points
        self.hidden    = hidden
        self.n_layers  = n_layers
        self.lr        = lr
        self.lam_phys  = lam_phys
        self.lam_bc    = lam_bc
        self.device    = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.net = _PointwiseConditionalPINNNet(
            in_channels=7,
            hidden=self.hidden,
            n_layers=self.n_layers,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    # ------------------------------------------------------------------
    def _build_problem_data(
        self, problems: list[dict]
    ) -> list[tuple[ConditionalGrid2D, GeneralPDE]]:
        """Parse and build (grid, pde_obj) pairs; silently skip unparseable entries."""
        data: list[tuple[ConditionalGrid2D, GeneralPDE]] = []
        for prob in problems:
            try:
                parsed_pde = parse_pde(prob["pde_str"])
                bc_specs = parse_bc(prob["bc_dict"])
                source_fn = parsed_pde.rhs_fn()
                grid     = ConditionalGrid2D(self.n_points, bc_specs, source_fn, self.device)
                pde_obj  = GeneralPDE(parsed_pde, bc_specs)
                data.append((grid, pde_obj))
            except Exception:
                continue
        return data

    def _load_problems_from_dataset(self, dataset_path: str) -> list[dict]:
        """Load (pde_str, bc_dict) pairs from generated dataset."""
        npz = np.load(dataset_path, allow_pickle=True)
        pde_strs = npz["pde_strs"]
        bc_json = npz["bc_dict_json"]
        problems: list[dict] = []
        for i in range(len(pde_strs)):
            try:
                problems.append(
                    {
                        "pde_str": str(pde_strs[i]),
                        "bc_dict": json.loads(str(bc_json[i])),
                    }
                )
            except Exception:
                continue
        return problems

    def _eval_val_data(
        self, val_data: list[tuple[ConditionalGrid2D, GeneralPDE]]
    ) -> float:
        """Return average total loss over pre-built val pairs under ``net.eval()``."""
        self.net.eval()
        total = 0.0
        with torch.no_grad():
            for grid, pde_obj in val_data:
                u    = self.net(grid.input_grid).squeeze(0).squeeze(-1)
                phys = pde_obj.compute_pde_loss(u)
                bc   = pde_obj.compute_bc_loss(u, grid.x_1d, grid.y_1d)
                total += (self.lam_phys * phys + self.lam_bc * bc).item()
        self.net.train()
        return total / len(val_data) if val_data else float("nan")

    # ------------------------------------------------------------------
    def train(
        self,
        config: PDESpaceConfig | None = None,
        n_samples: int = 5000,
        steps_per_problem: int = 3,
        n_epochs: int = 20,
        save_path: str = "pretrained_models/pinn.pt",
        seed: int = 42,
        print_every: int = 500,
        n_val: int = 32,
        eval_every: int = 1000,
        train_dataset_path: str | None = None,
        val_dataset_path: str | None = None,
    ) -> None:
        """Train a shared conditional PINN via a multi-task loop.

        Rather than fully converging on each problem in sequence, this method
        cycles through the entire problem pool ``n_epochs`` times and takes
        only ``steps_per_problem`` gradient steps per problem per visit.
        This biases the network toward learning reusable solution-operator
        patterns instead of memorising individual solutions.

        Parameters
        ----------
        config            : PDE parameter space configuration
        n_samples         : training problem pool size
        steps_per_problem : gradient steps per problem per epoch visit (1–5)
        n_epochs          : full passes over the training pool
        save_path         : path for the final (and best) checkpoint
        seed              : master random seed
        print_every       : log every N total gradient steps
        n_val             : held-out validation pool size
        eval_every        : evaluate val set every N total gradient steps
        """
        if config is None:
            config = PDESpaceConfig()

        rng = np.random.default_rng(seed)
        if train_dataset_path is not None:
            print(f"Loading PINN training problems from dataset: {train_dataset_path}")
            train_problems = self._load_problems_from_dataset(train_dataset_path)
            if val_dataset_path is not None:
                print(f"Loading PINN validation problems from dataset: {val_dataset_path}")
                val_problems = self._load_problems_from_dataset(val_dataset_path)
            else:
                # Auto 80/20 split: last 20% as validation (deterministic, no rng needed)
                n_val_auto = max(1, len(train_problems) // 5)
                val_problems = train_problems[-n_val_auto:]
                train_problems = train_problems[:-n_val_auto]
                print(
                    f"Auto 80/20 split: {len(train_problems)} train / "
                    f"{len(val_problems)} val problems"
                )
        else:
            sampler = LHSSampler(config)
            print(f"Generating {n_samples} training problems (seed={seed})...")
            train_problems = sampler.generate(n_samples=n_samples, seed=seed)

            val_seed = seed + 99999
            print(f"Generating {n_val} validation problems (seed={val_seed})...")
            val_problems = sampler.generate(n_samples=n_val, seed=val_seed)

        print("Pre-building training grids...")
        train_data = self._build_problem_data(train_problems)
        print("Pre-building validation grids...")
        val_data   = self._build_problem_data(val_problems)
        print(f"  {len(train_data)} train / {len(val_data)} val problems ready.")

        best_val_loss: float = float("inf")
        best_state: dict[str, Any] | None = None
        total_steps = 0

        self.net.train()
        for epoch in range(n_epochs):
            order = rng.permutation(len(train_data))
            for idx in order:
                grid, pde_obj = train_data[int(idx)]
                for _ in range(steps_per_problem):
                    self.optimizer.zero_grad()
                    u    = self.net(grid.input_grid).squeeze(0).squeeze(-1)
                    phys = pde_obj.compute_pde_loss(u)
                    bc   = pde_obj.compute_bc_loss(u, grid.x_1d, grid.y_1d)
                    loss = self.lam_phys * phys + self.lam_bc * bc
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    total_steps += 1

                    if total_steps % print_every == 0:
                        print(f"  step {total_steps:>7d} | "
                              f"phys={phys.item():.3e}  bc={bc.item():.3e}  "
                              f"total={loss.item():.3e}")

                    if total_steps % eval_every == 0 and val_data:
                        val_loss = self._eval_val_data(val_data)
                        improved = not np.isnan(val_loss) and val_loss < best_val_loss
                        if improved:
                            best_val_loss = val_loss
                            best_state = {
                                k: v.cpu().clone()
                                for k, v in self.net.state_dict().items()
                            }
                        marker = " *" if improved else ""
                        print(f"  step {total_steps:>7d} | "
                              f"val={val_loss:.3e}  best={best_val_loss:.3e}{marker}")
                        self.net.train()

            print(f"Epoch {epoch + 1}/{n_epochs} done  (total_steps={total_steps})")

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"arch": "shared_pinn", "state_dict": self.net.state_dict()}, save_path)
        print(f"Saved final shared PINN weights → {save_path}")

        if best_state is not None:
            best_path = str(save_path).replace(".pt", "_best.pt")
            torch.save({"arch": "shared_pinn", "state_dict": best_state}, best_path)
            print(f"Saved best  shared PINN weights → {best_path}")
        else:
            print("No improvement recorded; only the final checkpoint was saved.")

    # ------------------------------------------------------------------
    def _load_examples(self, dataset_path: str) -> list[dict[str, Any]]:
        """Load dataset and rebuild (grid, pde_obj, fd_target) at self.n_points.

        FD targets are bilinear-downsampled to self.n_points when the stored
        resolution differs (datasets are always generated at 64\u00d764).
        """
        npz      = np.load(dataset_path, allow_pickle=True)
        pde_strs = npz["pde_strs"]
        bc_json  = npz["bc_dict_json"]
        targets  = npz["targets"]   # (N, 64, 64)

        examples: list[dict[str, Any]] = []
        for i in range(len(pde_strs)):
            try:
                parsed_pde = parse_pde(str(pde_strs[i]))
                bc_dict    = json.loads(str(bc_json[i]))
                bc_specs   = parse_bc(bc_dict)
                source_fn  = parsed_pde.rhs_fn()
                grid       = ConditionalGrid2D(self.n_points, bc_specs, source_fn, self.device)
                pde_obj    = GeneralPDE(parsed_pde, bc_specs)
                x_1d       = torch.linspace(0, 1, self.n_points, device=self.device)
                y_1d       = torch.linspace(0, 1, self.n_points, device=self.device)
                t = torch.tensor(targets[i], dtype=torch.float32, device=self.device)
                if t.shape[0] != self.n_points:
                    t = torch.nn.functional.interpolate(
                        t.unsqueeze(0).unsqueeze(0),
                        size=(self.n_points, self.n_points),
                        mode="bilinear",
                        align_corners=True,
                    ).squeeze(0).squeeze(0)
                examples.append({
                    "input":   grid.input_grid.squeeze(0),
                    "target":  t,
                    "pde_obj": pde_obj,
                    "x_1d":    x_1d,
                    "y_1d":    y_1d,
                })
            except Exception:
                continue
        return examples

    # ------------------------------------------------------------------
    def test(
        self,
        test_dataset_path: str,
        print_every: int = 10,
    ) -> dict[str, float]:
        """Evaluate the trained shared PINN against FD ground truth on a held-out test set.

        Runs the shared net in eval mode on each sample and reports error metrics
        comparing predictions to the stored FD targets (downsampled to self.n_points).

        Parameters
        ----------
        test_dataset_path : Path to .npz test dataset.
        print_every       : Print per-sample stats every N samples.

        Returns
        -------
        summary : dict with aggregate metrics.
        """
        examples = self._load_examples(test_dataset_path)
        if not examples:
            raise RuntimeError(f"No examples loaded from {test_dataset_path!r}")

        print(f"Testing PINN on {len(examples)} samples from {test_dataset_path!r} ...")

        rel_l2s: list[float] = []
        max_errs: list[float] = []
        rmses: list[float] = []
        bc_errs: list[float] = []
        pde_ress: list[float] = []

        self.net.eval()
        with torch.no_grad():
            for i, ex in enumerate(examples):
                u_pred = self.net(ex["input"].unsqueeze(0)).squeeze(0).squeeze(-1)
                u_fd   = ex["target"]

                diff    = u_pred - u_fd
                fd_norm = torch.norm(u_fd) + 1e-12
                rel_l2  = float(torch.norm(diff) / fd_norm)
                max_err = float(torch.max(torch.abs(diff)))
                rmse    = float(torch.sqrt(torch.mean(diff ** 2)))
                bc_err  = float(ex["pde_obj"].compute_bc_loss(u_pred, ex["x_1d"], ex["y_1d"]))
                pde_res = float(ex["pde_obj"].compute_pde_loss(u_pred))

                rel_l2s.append(rel_l2)
                max_errs.append(max_err)
                rmses.append(rmse)
                bc_errs.append(bc_err)
                pde_ress.append(pde_res)

                if (i + 1) % print_every == 0:
                    print(
                        f"  [{i + 1:>5d}/{len(examples)}] "
                        f"rel_l2={rel_l2:.3e}  max={max_err:.3e}  rmse={rmse:.3e}"
                    )
        self.net.train()

        arr = np.array(rel_l2s)
        summary = {
            "n_samples":     len(examples),
            "rel_l2_mean":   float(np.mean(arr)),
            "rel_l2_std":    float(np.std(arr)),
            "rel_l2_p50":    float(np.median(arr)),
            "rel_l2_p90":    float(np.percentile(arr, 90)),
            "rel_l2_max":    float(np.max(arr)),
            "rmse_mean":     float(np.mean(rmses)),
            "max_err_mean":  float(np.mean(max_errs)),
            "max_err_max":   float(np.max(max_errs)),
            "bc_error_mean": float(np.mean(bc_errs)),
            "pde_res_mean":  float(np.mean(pde_ress)),
        }
        _print_test_summary(summary, "PINN")
        return summary


# ---------------------------------------------------------------------------
# Test / evaluation helpers
# ---------------------------------------------------------------------------

def _print_test_summary(summary: dict[str, float], label: str) -> None:
    """Print a formatted test-results table comparing solver to FD ground truth."""
    pad = 50
    print()
    print("=" * pad)
    print(f" Test results  [{label} vs FD ground truth]")
    print("=" * pad)
    print(f"  Samples : {summary['n_samples']}")
    print()
    print(f"  Relative L2 error:")
    print(f"    mean : {summary['rel_l2_mean']:.4e}")
    print(f"    std  : {summary['rel_l2_std']:.4e}")
    print(f"    p50  : {summary['rel_l2_p50']:.4e}")
    print(f"    p90  : {summary['rel_l2_p90']:.4e}")
    print(f"    max  : {summary['rel_l2_max']:.4e}")
    print()
    print(f"  RMSE (mean)        : {summary['rmse_mean']:.4e}")
    print(f"  Max abs err (mean) : {summary['max_err_mean']:.4e}")
    print(f"  Max abs err (max)  : {summary['max_err_max']:.4e}")
    print()
    print(f"  BC error  (mean)   : {summary['bc_error_mean']:.4e}")
    print(f"  PDE resid (mean)   : {summary['pde_res_mean']:.4e}")
    print("=" * pad)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline trainer for FNO / PINN PDE solvers"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- fno generate sub-command ---
    fno_gen_sub = sub.add_parser("fno-generate", help="Generate FD-supervised dataset from LHS PDE samples")
    fno_gen_sub.add_argument("--samples",       type=int,   default=5000)
    fno_gen_sub.add_argument("--resolution",    type=int,   default=64,
                         help="FD grid resolution n×n (default: 64)")
    fno_gen_sub.add_argument("--seed",          type=int,   default=42)
    fno_gen_sub.add_argument("--dataset-path",  type=str,   default="pretrained_models/fno_train_data.npz")
    fno_gen_sub.add_argument("--n-val",         type=int,   default=64)
    fno_gen_sub.add_argument("--val-dataset-path", type=str, default="pretrained_models/fno_val_data.npz")
    fno_gen_sub.add_argument("--manifest-path", type=str,   default="pretrained_models/fno_manifest.npz")
    fno_gen_sub.add_argument("--max-iterations", type=int,  default=5000)
    fno_gen_sub.add_argument("--tolerance",      type=float, default=1e-5)
    fno_gen_sub.add_argument("--print-every",    type=int,   default=200)
    fno_gen_sub.add_argument("--n-workers",       type=int,   default=None,
                         help="Worker processes for parallel FD generation (default: cpu_count)")
    fno_gen_sub.add_argument("--device",         type=str,   default=None)

    # --- fno train sub-command ---
    fno_train_sub = sub.add_parser("fno-train", help="Train conditional FNO from generated dataset")
    fno_train_sub.add_argument("--train-dataset", type=str, required=True)
    fno_train_sub.add_argument("--val-dataset",   type=str, default=None)
    fno_train_sub.add_argument("--epochs",        type=int,   default=30)
    fno_train_sub.add_argument("--resolution",    type=int,   default=64)
    fno_train_sub.add_argument("--width",         type=int,   default=32)
    fno_train_sub.add_argument("--modes",         type=int,   default=12)
    fno_train_sub.add_argument("--layers",        type=int,   default=4)
    fno_train_sub.add_argument("--lr",            type=float, default=5e-4)
    fno_train_sub.add_argument("--lam-data",      type=float, default=1.0)
    fno_train_sub.add_argument("--lam-phys",      type=float, default=1.0)
    fno_train_sub.add_argument("--lam-bc",        type=float, default=10.0)
    fno_train_sub.add_argument("--fno-path",      type=str,   default="pretrained_models/fno.pt")
    fno_train_sub.add_argument("--seed",          type=int,   default=42)
    fno_train_sub.add_argument("--print-every",   type=int,   default=200)
    fno_train_sub.add_argument("--eval-every",    type=int,   default=1)
    fno_train_sub.add_argument("--device",        type=str,   default=None)

    # --- fno one-shot sub-command (backward-compatible) ---
    fno_sub = sub.add_parser("fno", help="One-shot FNO pipeline: generate FD data then hybrid-train")
    fno_sub.add_argument("--samples",       type=int,   default=5000)
    fno_sub.add_argument("--epochs",        type=int,   default=30,
                       help="Hybrid training epochs")
    fno_sub.add_argument("--resolution",    type=int,   default=64)
    fno_sub.add_argument("--width",         type=int,   default=32)
    fno_sub.add_argument("--modes",         type=int,   default=12)
    fno_sub.add_argument("--layers",        type=int,   default=4)
    fno_sub.add_argument("--lr",            type=float, default=5e-4)
    fno_sub.add_argument("--lam-data",      type=float, default=1.0)
    fno_sub.add_argument("--lam-phys",      type=float, default=1.0)
    fno_sub.add_argument("--lam-bc",        type=float, default=10.0)
    fno_sub.add_argument("--seed",          type=int,   default=42)
    fno_sub.add_argument("--dataset-path",  type=str,   default="pretrained_models/fno_train_data.npz")
    fno_sub.add_argument("--val-dataset-path", type=str, default="pretrained_models/fno_val_data.npz")
    fno_sub.add_argument("--max-iterations", type=int, default=5000)
    fno_sub.add_argument("--tolerance", type=float, default=1e-5)
    fno_sub.add_argument("--fno-path",      type=str,   default="pretrained_models/fno.pt")
    fno_sub.add_argument("--manifest-path", type=str,   default="pretrained_models/fno_manifest.npz")
    fno_sub.add_argument("--device",        type=str,   default=None)
    fno_sub.add_argument("--print-every",   type=int,   default=200)
    fno_sub.add_argument("--n-val",         type=int,   default=64,
                       help="Number of held-out validation problems (default: 64)")
    fno_sub.add_argument("--eval-every",    type=int,   default=1,
                       help="Evaluate val set every N epochs (default: 1)")
    fno_sub.add_argument("--n-workers",     type=int,   default=None,
                       help="Worker processes for parallel FD generation (default: cpu_count)")

    # --- pinn sub-command ---
    pinn_sub = sub.add_parser("pinn", help="Train the shared PINN solution operator")
    pinn_sub.add_argument("--samples",           type=int,   default=5000,
                        help="Training problem pool size (default: 5000)")
    pinn_sub.add_argument("--train-dataset",     type=str,   default=None,
                        help="Optional shared dataset generated by fno-generate; overrides --samples")
    pinn_sub.add_argument("--val-dataset",       type=str,   default=None,
                        help="Optional validation dataset for PINN")
    pinn_sub.add_argument("--steps-per-problem",  type=int,   default=3,
                        help="Gradient steps per problem per visit (default: 3)")
    pinn_sub.add_argument("--n-epochs",           type=int,   default=20,
                        help="Full passes over the problem pool (default: 20)")
    pinn_sub.add_argument("--resolution",         type=int,   default=32)
    pinn_sub.add_argument("--hidden",             type=int,   default=64)
    pinn_sub.add_argument("--layers",             type=int,   default=4)
    pinn_sub.add_argument("--lr",                 type=float, default=1e-3)
    pinn_sub.add_argument("--lam-phys",           type=float, default=1.0,
                        help="Physics-loss weight λ_phys (default: 1.0)")
    pinn_sub.add_argument("--lam-bc",             type=float, default=10.0,
                        help="BC-loss weight λ_bc (default: 10.0)")
    pinn_sub.add_argument("--seed",               type=int,   default=42)
    pinn_sub.add_argument("--pinn-path",          type=str,   default="pretrained_models/pinn.pt")
    pinn_sub.add_argument("--device",             type=str,   default=None)
    pinn_sub.add_argument("--print-every",        type=int,   default=500)
    pinn_sub.add_argument("--n-val",              type=int,   default=32,
                        help="Held-out validation pool size (default: 32)")
    pinn_sub.add_argument("--eval-every",         type=int,   default=1000,
                        help="Evaluate val set every N gradient steps (default: 1000)")

    # --- fno-test sub-command ---
    fno_test_sub = sub.add_parser(
        "fno-test",
        help="Evaluate trained FNO against FD ground truth (80/20 guardrail)",
    )
    fno_test_sub.add_argument("--test-dataset", type=str, required=True,
                            help="Path to held-out test .npz dataset")
    fno_test_sub.add_argument("--fno-path",     type=str, default="pretrained_models/fno.pt")
    fno_test_sub.add_argument("--resolution",   type=int, default=64)
    fno_test_sub.add_argument("--width",        type=int, default=32)
    fno_test_sub.add_argument("--modes",        type=int, default=12)
    fno_test_sub.add_argument("--layers",       type=int, default=4)
    fno_test_sub.add_argument("--print-every",  type=int, default=10)
    fno_test_sub.add_argument("--device",       type=str, default=None)

    # --- pinn-test sub-command ---
    pinn_test_sub = sub.add_parser(
        "pinn-test",
        help="Evaluate trained PINN against FD ground truth (80/20 guardrail)",
    )
    pinn_test_sub.add_argument("--test-dataset", type=str, required=True,
                             help="Path to held-out test .npz dataset")
    pinn_test_sub.add_argument("--pinn-path",    type=str, default="pretrained_models/pinn.pt")
    pinn_test_sub.add_argument("--resolution",   type=int, default=32)
    pinn_test_sub.add_argument("--hidden",       type=int, default=64)
    pinn_test_sub.add_argument("--layers",       type=int, default=4)
    pinn_test_sub.add_argument("--print-every",  type=int, default=10)
    pinn_test_sub.add_argument("--device",       type=str, default=None)

    return parser


def main() -> None:
    parser = _make_parser()
    args   = parser.parse_args()

    if args.command == "fno-generate": 
        generator = SharedFDDataGenerator(
            n_points=args.resolution,
            device=args.device,
        )
        generator.generate(
            n_samples=args.samples,
            seed=args.seed,
            save_path=args.dataset_path,
            manifest_path=args.manifest_path,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            print_every=args.print_every,
            n_workers=args.n_workers,
        )
        if args.n_val > 0:
            generator.generate(
                n_samples=args.n_val,
                seed=args.seed + 99999,
                save_path=args.val_dataset_path,
                manifest_path=None,
                max_iterations=args.max_iterations,
                tolerance=args.tolerance,
                print_every=max(20, args.print_every // 4),
                n_workers=args.n_workers,
            )

    elif args.command == "fno-train":
        trainer = HybridFNOTrainer(
            n_points=args.resolution,
            width=args.width,
            n_modes=(args.modes, args.modes),
            n_layers=args.layers,
            lr=args.lr,
            lam_data=args.lam_data,
            lam_phys=args.lam_phys,
            lam_bc=args.lam_bc,
            device=args.device,
        )
        trainer.train(
            train_dataset_path=args.train_dataset,
            val_dataset_path=args.val_dataset,
            n_epochs=args.epochs,
            save_path=args.fno_path,
            seed=args.seed,
            print_every=args.print_every,
            eval_every=args.eval_every,
        )

    elif args.command == "fno":
        generator = SharedFDDataGenerator(
            n_points=args.resolution,
            device=args.device,
        )
        generator.generate(
            n_samples=args.samples,
            seed=args.seed,
            save_path=args.dataset_path,
            manifest_path=args.manifest_path,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            print_every=args.print_every,
            n_workers=args.n_workers,
        )
        val_dataset_path = None
        if args.n_val > 0:
            val_dataset_path = args.val_dataset_path
            generator.generate(
                n_samples=args.n_val,
                seed=args.seed + 99999,
                save_path=val_dataset_path,
                manifest_path=None,
                max_iterations=args.max_iterations,
                tolerance=args.tolerance,
                print_every=max(20, args.print_every // 4),
                n_workers=args.n_workers,
            )

        trainer = HybridFNOTrainer(
            n_points=args.resolution,
            width=args.width,
            n_modes=(args.modes, args.modes),
            n_layers=args.layers,
            lr=args.lr,
            lam_data=args.lam_data,
            lam_phys=args.lam_phys,
            lam_bc=args.lam_bc,
            device=args.device,
        )
        trainer.train(
            train_dataset_path=args.dataset_path,
            val_dataset_path=val_dataset_path,
            n_epochs=args.epochs,
            save_path=args.fno_path,
            seed=args.seed,
            print_every=args.print_every,
            eval_every=args.eval_every,
        )
    elif args.command == "pinn":
        trainer = PINNTrainer(
            n_points=args.resolution,
            hidden=args.hidden,
            n_layers=args.layers,
            lr=args.lr,
            lam_phys=args.lam_phys,
            lam_bc=args.lam_bc,
            device=args.device,
        )
        trainer.train(
            n_samples=args.samples,
            steps_per_problem=args.steps_per_problem,
            n_epochs=args.n_epochs,
            save_path=args.pinn_path,
            seed=args.seed,
            print_every=args.print_every,
            n_val=args.n_val,
            eval_every=args.eval_every,
            train_dataset_path=args.train_dataset,
            val_dataset_path=args.val_dataset,
        )

    elif args.command == "fno-test":
        trainer = HybridFNOTrainer(
            n_points=args.resolution,
            width=args.width,
            n_modes=(args.modes, args.modes),
            n_layers=args.layers,
            device=args.device,
        )
        state = torch.load(args.fno_path, map_location=str(trainer.device))
        trainer.model.load_state_dict(state)
        trainer.test(
            test_dataset_path=args.test_dataset,
            print_every=args.print_every,
        )

    elif args.command == "pinn-test":
        trainer = PINNTrainer(
            n_points=args.resolution,
            hidden=args.hidden,
            n_layers=args.layers,
            device=args.device,
        )
        ckpt = torch.load(args.pinn_path, map_location=str(trainer.device))
        state_dict = (
            ckpt["state_dict"]
            if isinstance(ckpt, dict) and "state_dict" in ckpt
            else ckpt
        )
        trainer.net.load_state_dict(state_dict)
        trainer.test(
            test_dataset_path=args.test_dataset,
            print_every=args.print_every,
        )


if __name__ == "__main__":
    # On macOS, Python 3.8+ already defaults to 'spawn', but be explicit so
    # worker processes don't inherit CUDA state or inconsistent module state.
    multiprocessing.set_start_method("spawn", force=True)
    main()
