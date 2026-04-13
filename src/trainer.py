"""Offline training pipeline for conditional FNO / PINN solvers.

Public CLI
----------
    $ python -m src.trainer generate  …  # FD-supervised dataset generation
    $ python -m src.trainer train     …  # model training (fno|pinn)
    $ python -m src.trainer manifest  …  # OOD manifest creation
    $ python -m src.trainer test      …  # evaluation on a held-out set

Legacy entry-points (``fno-generate``, ``fno-train``, ``fno``, ``pinn``,
``fno-test``, ``pinn-test``) are kept as thin wrappers for back-compatibility
but emit a deprecation warning so users can migrate to the unified CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# Ensure src/ is in path when invoked directly
sys.path.insert(0, str(Path(__file__).parent))

from dataset import PDEOperatorDataset, RepeatDataset, collate_operator_batch
from models.conditional_inputs import ConditionalGrid2D
from models.conditional_solvers import _PointwiseConditionalPINNNet
from models.fno_model import FNO2DModel
from pde_space import PDESpaceConfig, LHSSampler
from ood_detector import PDEFeaturizer, OODDetector
from pde_parser import parse_bc, parse_pde
from physics.pde_helpers import GeneralPDE, compute_sample_metrics, solve_fd_jacobi
from training import train_operator


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    log.addHandler(_h)


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
    try:
        u, _, _, _ = solve_fd_jacobi(
            pde_obj=pde_obj,
            n_points=n_points,
            device=device,
            max_iterations=max_iterations,
            tolerance=tolerance,
            print_every=None,
            sanitize_on_divergence=False,
        )
        return u
    except ValueError:
        return None


def _solve_one_worker(args: tuple) -> dict | None:
    """Top-level picklable worker for ProcessPoolExecutor.

    Always uses CPU so CUDA contexts are never shared across processes.
    All arguments must be plain picklable types (no torch.device, no lambdas).
    """
    prob, n_points, device_str, max_iters, tol = args
    try:
        parsed_pde = parse_pde(prob["pde_str"])
        bc_specs = parse_bc(prob["bc_dict"])
        source_fn = parsed_pde.rhs_fn()
        device = torch.device(device_str)
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

    def generate(
        self,
        config: PDESpaceConfig | None = None,
        n_samples: int = 5000,
        seed: int = 42,
        save_path: str = "pretrained_models/fno_train_data.npz",
        max_iterations: int = 5000,
        tolerance: float = 1e-5,
        print_every: int = 200,
        n_workers: int | None = None,
        chunk_size: int = _CHUNK_SIZE,
        ood_percentile: float = 95.0,
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
                log.info("Loading existing chunk %d–%d …", chunk_start, chunk_end)
                c = np.load(chunk_path, allow_pickle=True)
                inputs.extend(c["inputs"])
                targets.extend(c["targets"])
                pde_strs.extend(c["pde_strs"].tolist())
                bc_json.extend((c["bc_json"] if "bc_json" in c else c["bc_dict_json"]).tolist())
                feats.extend(c["feats"])
                total_done += len(c["inputs"])
                continue

            chunk_problems = problems[chunk_start:chunk_end]
            args_list = [
                (p, self.n_points, str(self.device), max_iterations, tolerance)
                for p in chunk_problems
            ]

            chunk_results: list[dict] = []
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = {ex.submit(_solve_one_worker, a): i for i, a in enumerate(args_list)}
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
                    bc_json=c_bc_json,
                    feats=c_feats,
                )

                inputs.extend(c_inputs)
                targets.extend(c_targets)
                pde_strs.extend(c_pde_strs.tolist())
                bc_json.extend(c_bc_json.tolist())
                feats.extend(c_feats)
                total_done += len(chunk_results)

        log.info("Generated %d/%d samples …", total_done, n_samples)

        if not inputs:
            raise RuntimeError("No valid FD samples generated; dataset is empty.")

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path,
            inputs=np.stack(inputs),
            targets=np.stack(targets),
            pde_strs=np.array(pde_strs, dtype=object),
            bc_json=np.array(bc_json, dtype=object),
            feats=np.stack(feats),
            n_points=np.array([self.n_points], dtype=np.int32),
        )
        log.info("Saved dataset %s (%d samples)", save_path, len(inputs))

        return len(inputs)


def _load_dataset_features(dataset_path: str) -> np.ndarray:
    """Load PDE feature vectors stored in a generated dataset."""
    data = np.load(dataset_path, allow_pickle=True)
    if "feats" not in data:
        raise KeyError(
            f"Dataset {dataset_path!r} does not contain stored PDE features. "
            "Re-generate it with the current trainer before building an OOD manifest."
        )
    return np.asarray(data["feats"], dtype=np.float32)


def build_ood_manifest(
    train_dataset: str,
    out_path: str,
    percentile: float = 95.0,
    val_dataset: str | None = None,
) -> None:
    """Build an OOD manifest from generated dataset features.

    Feature vectors are computed from PDE coefficients and BC specs at generation
    time, so this manifest is valid for both FNO and PINN solvers.
    """
    train_feats = _load_dataset_features(train_dataset)
    val_feats = _load_dataset_features(val_dataset) if val_dataset is not None else None
    OODDetector.build_manifest(
        train_feats,
        manifest_path=out_path,
        ood_percentile=percentile,
        calibration_features=val_feats,
    )
    log.info("Saved OOD manifest to %s", out_path)


def _make_operator_loader(dataset: torch.utils.data.Dataset, *, shuffle: bool) -> DataLoader:
    """Create a standard operator DataLoader with the shared collate fn."""
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        collate_fn=collate_operator_batch,
    )


def _run_and_save_operator_training(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    loss_cfg: dict[str, float],
    epochs: int,
    print_every: int,
    eval_every: int,
    eval_mode: str,
    save_path: str,
    label: str = "checkpoint",
    state_wrap: Callable[[dict[str, torch.Tensor]], Any] | None = None,
) -> None:
    """Run the shared training loop and persist final/best checkpoints."""
    result = train_operator(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_cfg=loss_cfg,
        epochs=epochs,
        print_every=print_every,
        eval_every=eval_every,
        eval_mode=eval_mode,
        checkpoint_path=str(save_path).replace(".pt", ".ckpt"),
    )

    wrap = state_wrap or (lambda state: state)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(wrap(result["last_state"]), save_path)
    log.info("Saved final %s → %s", label, save_path)

    if result["best_state"] is not None:
        best_path = str(save_path).replace(".pt", "_best.pt")
        torch.save(wrap(result["best_state"]), best_path)
        log.info("Saved best  %s → %s", label, best_path)


def _split_train_val_dataset(
    dataset: torch.utils.data.Dataset,
    val_fraction: float,
    seed: int,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset | None]:
    """Deterministically split a dataset into train/val subsets."""
    if val_fraction <= 0.0 or len(dataset) < 2:
        return dataset, None

    n_val = max(1, int(round(len(dataset) * val_fraction)))
    n_val = min(n_val, len(dataset) - 1)
    if n_val <= 0:
        return dataset, None

    perm = np.random.default_rng(seed).permutation(len(dataset)).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


class _DatasetTrainerBase:
    """Shared abstraction for dataset-driven operator training workflows."""

    device: torch.device
    optimizer: torch.optim.Optimizer
    scheduler: optim.lr_scheduler.LRScheduler | None

    def _operator_module(self) -> nn.Module:
        raise NotImplementedError

    def _summary_label(self) -> str:
        raise NotImplementedError

    def _checkpoint_label(self) -> str:
        raise NotImplementedError

    def _loss_cfg(self) -> dict[str, float]:
        raise NotImplementedError

    def _wrap_state_dict(self, state: dict[str, torch.Tensor]) -> Any:
        return state

    def _build_loaders_from_datasets(
        self,
        *,
        train_dataset_path: str,
        val_dataset_path: str | None,
        n_points: int,
        use_targets: bool,
        rebuild_inputs: bool,
        shuffle_train: bool,
        steps_per_problem: int,
        auto_val_fraction: float,
        seed: int,
    ) -> tuple[DataLoader, DataLoader | None, int, int, bool]:
        """Build train/val loaders from generated datasets only."""
        train_ds_base = PDEOperatorDataset(
            npz_path=train_dataset_path,
            device=self.device,
            n_points=n_points,
            use_targets=use_targets,
            rebuild_inputs=rebuild_inputs,
        )

        auto_split = False
        if val_dataset_path is None:
            train_ds, val_ds = _split_train_val_dataset(
                train_ds_base,
                val_fraction=auto_val_fraction,
                seed=seed,
            )
            auto_split = val_ds is not None
        else:
            train_ds = train_ds_base
            val_ds = PDEOperatorDataset(
                npz_path=val_dataset_path,
                device=self.device,
                n_points=n_points,
                use_targets=use_targets,
                rebuild_inputs=rebuild_inputs,
            )

        train_len = len(train_ds)
        val_len = len(val_ds) if val_ds is not None else 0

        train_for_loader: torch.utils.data.Dataset = train_ds
        if steps_per_problem > 1:
            train_for_loader = RepeatDataset(train_ds, steps_per_problem)

        train_loader = _make_operator_loader(train_for_loader, shuffle=shuffle_train)
        val_loader = _make_operator_loader(val_ds, shuffle=False) if val_ds is not None else None
        return train_loader, val_loader, train_len, val_len, auto_split

    def _load_examples(
        self,
        dataset_path: str,
        *,
        use_targets: bool = True,
        rebuild_inputs: bool = False,
    ) -> list[dict[str, Any]]:
        dataset = PDEOperatorDataset(
            npz_path=dataset_path,
            device=self.device,
            n_points=self.n_points,
            use_targets=use_targets,
            rebuild_inputs=rebuild_inputs,
        )
        return dataset.examples

    def _train_from_dataset(
        self,
        *,
        train_dataset_path: str,
        val_dataset_path: str | None,
        n_epochs: int,
        save_path: str,
        print_every: int,
        eval_every: int,
        seed: int,
        train_log_label: str,
        steps_per_problem: int = 1,
        use_targets: bool = True,
        rebuild_inputs: bool = False,
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader, val_loader, n_train, n_val, auto_split = self._build_loaders_from_datasets(
            train_dataset_path=train_dataset_path,
            val_dataset_path=val_dataset_path,
            n_points=self.n_points,
            use_targets=use_targets,
            rebuild_inputs=rebuild_inputs,
            shuffle_train=True,
            steps_per_problem=steps_per_problem,
            auto_val_fraction=0.2,
            seed=seed,
        )

        total_steps = len(train_loader) * n_epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps, 1), eta_min=1e-5
        )
        if auto_split:
            log.info("Auto validation split: %d train / %d val examples", n_train, n_val)
        else:
            log.info("Loaded datasets: %d train / %d val examples", n_train, n_val)
        log.info(
            "Training %s on %d examples for %d epochs (T_max=%d) ...",
            train_log_label, n_train, n_epochs, total_steps,
        )

        _run_and_save_operator_training(
            model=self._operator_module(),
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            loss_cfg=self._loss_cfg(),
            epochs=n_epochs,
            print_every=print_every,
            eval_every=eval_every,
            eval_mode="epoch",
            save_path=save_path,
            label=self._checkpoint_label(),
            state_wrap=self._wrap_state_dict,
        )

    def _test_from_dataset(
        self,
        *,
        test_dataset_path: str,
        print_every: int,
        summary_label: str,
        rebuild_inputs: bool = False,
    ) -> dict[str, float]:
        examples = self._load_examples(
            test_dataset_path,
            use_targets=True,
            rebuild_inputs=rebuild_inputs,
        )
        if not examples:
            raise RuntimeError(f"No examples loaded from {test_dataset_path!r}")

        log.info(
            "Testing %s on %d samples from %r ...",
            summary_label,
            len(examples),
            test_dataset_path,
        )

        rel_l2s: list[float] = []
        max_errs: list[float] = []
        rmses: list[float] = []
        bc_errs: list[float] = []
        pde_ress: list[float] = []

        model = self._operator_module()
        model.eval()
        with torch.no_grad():
            for i, ex in enumerate(examples):
                u_pred = model(ex["input"].unsqueeze(0)).squeeze(0).squeeze(-1)
                u_fd = ex["target"]

                rel_l2, rmse, max_err = compute_sample_metrics(
                    u_pred.cpu().numpy(), u_fd.cpu().numpy()
                )
                bc_err = float(ex["pde_obj"].compute_bc_loss(u_pred, ex["x_1d"], ex["y_1d"]))
                pde_res = float(ex["pde_obj"].compute_pde_loss(u_pred))

                rel_l2s.append(rel_l2)
                max_errs.append(max_err)
                rmses.append(rmse)
                bc_errs.append(bc_err)
                pde_ress.append(pde_res)

                if (i + 1) % print_every == 0:
                    log.info(
                        "  [%5d/%d] rel_l2=%.3e  max=%.3e  rmse=%.3e",
                        i + 1, len(examples), rel_l2, max_err, rmse,
                    )
        model.train()

        arr = np.array(rel_l2s)
        summary = {
            "n_samples": len(examples),
            "rel_l2_mean": float(np.mean(arr)),
            "rel_l2_std": float(np.std(arr)),
            "rel_l2_p50": float(np.median(arr)),
            "rel_l2_p90": float(np.percentile(arr, 90)),
            "rel_l2_max": float(np.max(arr)),
            "rmse_mean": float(np.mean(rmses)),
            "max_err_mean": float(np.mean(max_errs)),
            "max_err_max": float(np.max(max_errs)),
            "bc_error_mean": float(np.mean(bc_errs)),
            "pde_res_mean": float(np.mean(pde_ress)),
        }
        _print_test_summary(summary, summary_label)
        return summary


class FNOTrainer(_DatasetTrainerBase):
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

    def _operator_module(self) -> nn.Module:
        return self.model

    def _summary_label(self) -> str:
        return "FNO"

    def _checkpoint_label(self) -> str:
        return "hybrid FNO weights"

    def _loss_cfg(self) -> dict[str, float]:
        return {
            "lam_data": self.lam_data,
            "lam_phys": self.lam_phys,
            "lam_bc": self.lam_bc,
        }

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
        self._train_from_dataset(
            train_dataset_path=train_dataset_path,
            val_dataset_path=val_dataset_path,
            n_epochs=n_epochs,
            save_path=save_path,
            print_every=print_every,
            eval_every=eval_every,
            seed=seed,
            train_log_label="hybrid FNO",
        )

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
        return self._test_from_dataset(
            test_dataset_path=test_dataset_path,
            print_every=print_every,
            summary_label=self._summary_label(),
        )

class PINNTrainer(_DatasetTrainerBase):
    """Train a shared-weights conditional PINN across LHS-sampled problems.

    This mirrors ``FNOTrainer`` conceptually: a single PINN checkpoint is
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
        lam_data: float = 1.0,
        lam_phys: float = 1.0,
        lam_bc: float = 10.0,
        device: str | None = None,
    ) -> None:
        self.n_points  = n_points
        self.hidden    = hidden
        self.n_layers  = n_layers
        self.lr        = lr
        self.lam_data  = lam_data
        self.lam_phys  = lam_phys
        self.lam_bc    = lam_bc
        self.device    = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = _PointwiseConditionalPINNNet(
            in_channels=7,
            hidden=self.hidden,
            n_layers=self.n_layers,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler: optim.lr_scheduler.LRScheduler | None = None

    def _operator_module(self) -> nn.Module:
        return self.model

    def _summary_label(self) -> str:
        return "PINN"

    def _checkpoint_label(self) -> str:
        return "shared PINN weights"

    def _loss_cfg(self) -> dict[str, float]:
        return {
            "lam_data": self.lam_data,
            "lam_phys": self.lam_phys,
            "lam_bc": self.lam_bc,
        }

    def _wrap_state_dict(self, state: dict[str, torch.Tensor]) -> Any:
        return {"arch": "shared_pinn", "state_dict": state}

    def train(
        self,
        train_dataset_path: str,
        val_dataset_path: str | None = None,
        steps_per_problem: int = 3,
        n_epochs: int = 20,
        save_path: str = "pretrained_models/pinn.pt",
        seed: int = 42,
        print_every: int = 500,
        eval_every: int = 1000,
    ) -> None:
        """Train shared PINN from generated datasets only."""
        if steps_per_problem != 1:
            log.warning(
                "steps_per_problem=%d is ignored for dataset-driven PINN training; using one pass per example.",
                steps_per_problem,
            )
        self._train_from_dataset(
            train_dataset_path=train_dataset_path,
            val_dataset_path=val_dataset_path,
            n_epochs=n_epochs,
            save_path=save_path,
            print_every=print_every,
            eval_every=eval_every,
            seed=seed,
            train_log_label="shared PINN",
        )

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
        return self._test_from_dataset(
            test_dataset_path=test_dataset_path,
            print_every=print_every,
            summary_label=self._summary_label(),
        )


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
# PDESpace smoke benchmark (coverage + FD solvability guardrail)
# ---------------------------------------------------------------------------

def _build_space_config(space_name: str) -> PDESpaceConfig:
    """Return a PDESpaceConfig preset by name."""
    key = space_name.strip().lower()
    if key in ("thermal-v2", "thermal_v2"):
        return PDESpaceConfig.thermal_v2()
    if key == "default":
        return PDESpaceConfig()
    raise ValueError(f"Unknown PDESpace preset {space_name!r}. Use 'default' or 'thermal-v2'.")


def _is_y_only_trig(rhs: str) -> bool:
    """Return True if rhs contains sin/cos(k*pi*y) and no x symbol."""
    rhs_clean = rhs.replace(" ", "")
    has_y_trig = bool(re.search(r"(sin|cos)\([^)]*pi\*y\)", rhs_clean))
    return has_y_trig and ("x" not in rhs_clean)


def _run_pdespace_smoke(
    config: PDESpaceConfig,
    n_samples: int,
    seed: int,
    n_fd_samples: int,
    fd_resolution: int,
    fd_max_iterations: int,
    fd_tolerance: float,
) -> dict[str, float]:
    """Compute lightweight PDESpace coverage + FD-solvability metrics."""
    sampler = LHSSampler(config)
    problems = sampler.generate(n_samples=n_samples, seed=seed)

    d_nonzero = 0
    e_nonzero = 0
    c_nonzero = 0
    f_nonzero = 0
    gaussian_rhs = 0
    stratified_rhs = 0
    neumann_walls = 0
    robin_walls = 0
    total_walls = 0

    parsed_pairs: list[tuple[Any, dict]] = []
    for prob in problems:
        parsed = parse_pde(prob["pde_str"])
        bc_specs = parse_bc(prob["bc_dict"])
        parsed_pairs.append((parsed, bc_specs))

        d_nonzero += int(abs(parsed.d) > 1e-12)
        e_nonzero += int(abs(parsed.e) > 1e-12)
        c_nonzero += int(abs(parsed.c) > 1e-12)
        f_nonzero += int(abs(parsed.f) > 1e-12)

        rhs = prob["pde_str"].split("=", 1)[1].strip()
        rhs_nospace = rhs.replace(" ", "")
        gaussian_rhs += int("exp(-" in rhs_nospace)
        stratified_rhs += int("(1-y)" in rhs_nospace or _is_y_only_trig(rhs))

        for wall in ("left", "right", "bottom", "top"):
            bc_t = prob["bc_dict"][wall]["type"]
            neumann_walls += int(bc_t == "neumann")
            robin_walls += int(bc_t == "robin")
            total_walls += 1

    # FD guardrail on a subset for speed
    n_fd = min(n_fd_samples, len(parsed_pairs))
    fd_ok = 0
    for parsed, bc_specs in parsed_pairs[:n_fd]:
        pde_obj = GeneralPDE(parsed, bc_specs)
        u, _, _, _ = solve_fd_jacobi(
            pde_obj=pde_obj,
            n_points=fd_resolution,
            device=torch.device("cpu"),
            max_iterations=fd_max_iterations,
            tolerance=fd_tolerance,
            print_every=None,
            sanitize_on_divergence=False,
        )
        fd_ok += int(u is not None)

    n = float(len(problems))
    return {
        "n_samples": float(len(problems)),
        "d_nonzero_frac": d_nonzero / n,
        "e_nonzero_frac": e_nonzero / n,
        "c_nonzero_frac": c_nonzero / n,
        "f_nonzero_frac": f_nonzero / n,
        "gaussian_rhs_frac": gaussian_rhs / n,
        "stratified_rhs_frac": stratified_rhs / n,
        "neumann_frac": neumann_walls / float(total_walls),
        "robin_frac": robin_walls / float(total_walls),
        "fd_ok_frac": (fd_ok / float(n_fd)) if n_fd > 0 else float("nan"),
    }


def _print_smoke_summary(space_name: str, metrics: dict[str, float]) -> None:
    """Print a compact PDESpace smoke report."""
    print()
    print("=" * 58)
    print(f" PDESpace smoke report [{space_name}]")
    print("=" * 58)
    print(f"  samples               : {int(metrics['n_samples'])}")
    print(f"  d non-zero frac       : {metrics['d_nonzero_frac']:.3f}")
    print(f"  e non-zero frac       : {metrics['e_nonzero_frac']:.3f}")
    print(f"  c non-zero frac       : {metrics['c_nonzero_frac']:.3f}")
    print(f"  f non-zero frac       : {metrics['f_nonzero_frac']:.3f}")
    print(f"  gaussian RHS frac     : {metrics['gaussian_rhs_frac']:.3f}")
    print(f"  stratified RHS frac   : {metrics['stratified_rhs_frac']:.3f}")
    print(f"  neumann wall frac     : {metrics['neumann_frac']:.3f}")
    print(f"  robin wall frac       : {metrics['robin_frac']:.3f}")
    print(f"  FD solvability frac   : {metrics['fd_ok_frac']:.3f}")
    print("=" * 58)


# ---------------------------------------------------------------------------
# CLI helpers – shared trainer-construction + run wrappers
# ---------------------------------------------------------------------------

def _alias_warning(old: str, new: str) -> None:
    log.warning("Command '%s' is deprecated; use the unified '%s' sub-command instead.", old, new)


def _run_fno_training(args: argparse.Namespace) -> None:
    """Instantiate FNOTrainer and run training.

    Works with both the unified ``train --solver fno`` args and the legacy
    ``fno-train`` args (different save-path key; tolerates None-defaulted args).
    """
    trainer = FNOTrainer(
        n_points=args.resolution or 64,
        width=args.width,
        n_modes=(args.modes, args.modes),
        n_layers=args.layers,
        lr=args.lr if args.lr is not None else 5e-4,
        lam_data=args.lam_data,
        lam_phys=args.lam_phys,
        lam_bc=args.lam_bc,
        device=args.device,
    )
    save_path = (
        getattr(args, "checkpoint", None)
        or getattr(args, "fno_path", None)
        or "pretrained_models/fno.pt"
    )
    trainer.train(
        train_dataset_path=args.train_dataset,
        val_dataset_path=getattr(args, "val_dataset", None),
        n_epochs=args.epochs if args.epochs is not None else 30,
        save_path=save_path,
        seed=args.seed,
        print_every=args.print_every if args.print_every is not None else 200,
        eval_every=args.eval_every if args.eval_every is not None else 1,
    )
    log.info("FNO training complete.")


def _run_pinn_training(args: argparse.Namespace) -> None:
    """Instantiate PINNTrainer and run training.

    Works with both the unified ``train --solver pinn`` args and the legacy
    ``pinn`` args (``n_epochs`` vs ``epochs``; different save-path key).
    """
    trainer = PINNTrainer(
        n_points=args.resolution or 32,
        hidden=args.hidden,
        n_layers=args.layers,
        lr=args.lr if args.lr is not None else 1e-3,
        lam_data=args.lam_data,
        lam_phys=args.lam_phys,
        lam_bc=args.lam_bc,
        device=args.device,
    )
    save_path = (
        getattr(args, "checkpoint", None)
        or getattr(args, "pinn_path", None)
        or "pretrained_models/pinn.pt"
    )
    n_epochs = getattr(args, "n_epochs", None) or (
        args.epochs if args.epochs is not None else 20
    )
    trainer.train(
        train_dataset_path=args.train_dataset,
        val_dataset_path=getattr(args, "val_dataset", None),
        steps_per_problem=args.steps_per_problem,
        n_epochs=n_epochs,
        save_path=save_path,
        seed=args.seed,
        print_every=args.print_every if args.print_every is not None else 500,
        eval_every=args.eval_every if args.eval_every is not None else 1,
    )
    log.info("PINN training complete.")

def _run_fno_testing(args: argparse.Namespace) -> None:
    """Instantiate FNOTrainer, load checkpoint, and run evaluation.

    Accepts both ``args.checkpoint`` (unified ``test``) and ``args.fno_path``
    (legacy ``fno-test``).
    """
    trainer = FNOTrainer(
        n_points=args.resolution or 64,
        width=args.width,
        n_modes=(args.modes, args.modes),
        n_layers=args.layers,
        device=args.device,
    )
    ckpt_path = (
        getattr(args, "checkpoint", None)
        or getattr(args, "fno_path", None)
        or "pretrained_models/fno.pt"
    )
    state = torch.load(ckpt_path, map_location=str(trainer.device))
    trainer.model.load_state_dict(state)
    trainer.test(test_dataset_path=args.test_dataset, print_every=args.print_every)


def _run_pinn_testing(args: argparse.Namespace) -> None:
    """Instantiate PINNTrainer, load checkpoint, and run evaluation.

    Accepts both ``args.checkpoint`` (unified ``test``) and ``args.pinn_path``
    (legacy ``pinn-test``).
    """
    trainer = PINNTrainer(
        n_points=args.resolution or 32,
        hidden=args.hidden,
        n_layers=args.layers,
        device=args.device,
    )
    ckpt_path = (
        getattr(args, "checkpoint", None)
        or getattr(args, "pinn_path", None)
        or "pretrained_models/pinn.pt"
    )
    ckpt = torch.load(ckpt_path, map_location=str(trainer.device))
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    trainer.model.load_state_dict(state_dict)
    trainer.test(test_dataset_path=args.test_dataset, print_every=args.print_every)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _run_generate_command(args: argparse.Namespace) -> None:
    """Unified dataset generation flow (train + optional val split)."""
    generator = SharedFDDataGenerator(
        n_points=args.resolution,
        device=args.device,
    )
    generator.generate(
        n_samples=args.samples,
        seed=args.seed,
        save_path=args.dataset_path,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        print_every=args.print_every,
        n_workers=args.n_workers,
        ood_percentile=args.ood_percentile,
    )
    if getattr(args, "manifest_path", None):
        log.warning(
            "--manifest-path on generate/fno-generate is deprecated; "
            "use the 'manifest' sub-command instead."
        )

    val_dataset_path: str | None = None
    if args.n_val > 0:
        val_dataset_path = args.val_dataset_path
        generator.generate(
            n_samples=args.n_val,
            seed=args.seed + 99999,
            save_path=val_dataset_path,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            print_every=max(20, args.print_every // 4),
            n_workers=args.n_workers,
            ood_percentile=args.ood_percentile,
        )


def _run_manifest_command(args: argparse.Namespace) -> None:
    """Build an OOD manifest from generated dataset features."""
    build_ood_manifest(
        train_dataset=args.train_dataset,
        out_path=args.out,
        percentile=args.percentile,
        val_dataset=args.val_dataset,
    )


def _run_train_command(args: argparse.Namespace) -> None:
    """Unified training flow for FNO and PINN."""
    if args.solver == "fno":
        if not args.train_dataset:
            raise SystemExit("train --solver fno requires --train-dataset")
        _run_fno_training(args)
        return

    if args.solver == "pinn":
        if not args.train_dataset:
            raise SystemExit("train --solver pinn requires --train-dataset")
        _run_pinn_training(args)
        return

    raise SystemExit(f"Unsupported solver {args.solver!r}. Use fno or pinn.")


def _run_test_command(args: argparse.Namespace) -> None:
    """Unified test flow for FNO and PINN."""
    if args.solver == "fno":
        _run_fno_testing(args)
        return

    if args.solver == "pinn":
        _run_pinn_testing(args)
        return

    raise SystemExit(f"Unsupported solver {args.solver!r}. Use fno or pinn.")


# ---------------------------------------------------------------------------
# Legacy command wrappers (thin adapters so main() stays clean)
# ---------------------------------------------------------------------------

def _run_fno_generate(args: argparse.Namespace) -> None:
    _alias_warning("fno-generate", "generate")
    _run_generate_command(args)
    if args.manifest_path:
        build_ood_manifest(
            train_dataset=args.dataset_path,
            out_path=args.manifest_path,
            percentile=args.ood_percentile,
            val_dataset=args.val_dataset_path if args.n_val > 0 else None,
        )


def _run_fno_train(args: argparse.Namespace) -> None:
    _alias_warning("fno-train", "train --solver fno")
    _run_fno_training(args)


def _run_fno_one_shot(args: argparse.Namespace) -> None:
    _alias_warning("fno", "generate + train --solver fno")
    generator = SharedFDDataGenerator(n_points=args.resolution, device=args.device)
    generator.generate(
        n_samples=args.samples,
        seed=args.seed,
        save_path=args.dataset_path,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        print_every=args.print_every,
        n_workers=args.n_workers,
        ood_percentile=args.ood_percentile,
    )
    val_dataset_path = None
    if args.n_val > 0:
        val_dataset_path = args.val_dataset_path
        generator.generate(
            n_samples=args.n_val,
            seed=args.seed + 99999,
            save_path=val_dataset_path,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            print_every=max(20, args.print_every // 4),
            n_workers=args.n_workers,
        )
    if args.manifest_path:
        build_ood_manifest(
            train_dataset=args.dataset_path,
            out_path=args.manifest_path,
            percentile=args.ood_percentile,
            val_dataset=val_dataset_path,
        )
    trainer = FNOTrainer(
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


def _run_pinn_one_shot(args: argparse.Namespace) -> None:
    _alias_warning("pinn", "train --solver pinn")
    _run_pinn_training(args)


def _run_fno_test(args: argparse.Namespace) -> None:
    _alias_warning("fno-test", "test --solver fno")
    _run_fno_testing(args)


def _run_pinn_test(args: argparse.Namespace) -> None:
    _alias_warning("pinn-test", "test --solver pinn")
    _run_pinn_testing(args)


def _run_pdespace_smoke_command(args: argparse.Namespace) -> None:
    cfg = _build_space_config(args.space)
    metrics = _run_pdespace_smoke(
        config=cfg,
        n_samples=args.samples,
        seed=args.seed,
        n_fd_samples=args.fd_samples,
        fd_resolution=args.fd_resolution,
        fd_max_iterations=args.fd_max_iterations,
        fd_tolerance=args.fd_tolerance,
    )
    _print_smoke_summary(args.space, metrics)
    checks = [
        ("d_nonzero_frac",    metrics["d_nonzero_frac"],    args.min_d_nonzero),
        ("e_nonzero_frac",    metrics["e_nonzero_frac"],    args.min_e_nonzero),
        ("gaussian_rhs_frac", metrics["gaussian_rhs_frac"], args.min_gaussian_rhs),
        ("stratified_rhs_frac", metrics["stratified_rhs_frac"], args.min_stratified_rhs),
        ("neumann+robin frac", metrics["neumann_frac"] + metrics["robin_frac"], args.min_neumann_robin),
        ("fd_ok_frac",        metrics["fd_ok_frac"],        args.min_fd_ok),
    ]
    failed = [(name, val, thr) for name, val, thr in checks if val < thr]
    if failed:
        print("Smoke check FAILED:")
        for name, val, thr in failed:
            print(f"  {name}: {val:.3f} < required {thr:.3f}")
        raise SystemExit(2)
    print("Smoke check PASSED.")


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline trainer for FNO / PINN PDE solvers"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- unified generate sub-command ---
    gen_sub = sub.add_parser("generate", help="Generate FD-supervised datasets")
    gen_sub.add_argument("--samples", type=int, default=5000)
    gen_sub.add_argument("--resolution", type=int, default=64,
                         help="FD grid resolution n×n (default: 64)")
    gen_sub.add_argument("--seed", type=int, default=42)
    gen_sub.add_argument("--dataset-path", type=str, default="pretrained_models/fno_train_data.npz")
    gen_sub.add_argument("--n-val", type=int, default=64)
    gen_sub.add_argument("--val-dataset-path", type=str, default="pretrained_models/fno_val_data.npz")
    gen_sub.add_argument("--max-iterations", type=int, default=5000)
    gen_sub.add_argument("--tolerance", type=float, default=1e-5)
    gen_sub.add_argument("--print-every", type=int, default=200)
    gen_sub.add_argument("--n-workers", type=int, default=None,
                         help="Worker processes for parallel FD generation (default: cpu_count)")
    gen_sub.add_argument("--ood-percentile", type=float, default=95.0,
                         help="KNN threshold percentile for OOD detection (default: 95.0)")
    gen_sub.add_argument("--device", type=str, default=None)

    # --- unified train sub-command ---
    train_sub = sub.add_parser("train", help="Train a solver (fno|pinn)")
    train_sub.add_argument("--solver", type=str, choices=["fno", "pinn"], required=True)
    train_sub.add_argument("--train-dataset", type=str, default=None)
    train_sub.add_argument("--val-dataset", type=str, default=None)
    train_sub.add_argument("--epochs", type=int, default=None)
    train_sub.add_argument("--steps-per-problem", type=int, default=3,
                           help="PINN only: gradient steps per problem")
    train_sub.add_argument("--resolution", type=int, default=None)
    train_sub.add_argument("--width", type=int, default=32,
                           help="FNO only")
    train_sub.add_argument("--modes", type=int, default=12,
                           help="FNO only")
    train_sub.add_argument("--hidden", type=int, default=64,
                           help="PINN only")
    train_sub.add_argument("--layers", type=int, default=4)
    train_sub.add_argument("--lr", type=float, default=None)
    train_sub.add_argument("--lam-data", type=float, default=1.0,
                           help="FNO only")
    train_sub.add_argument("--lam-phys", type=float, default=1.0)
    train_sub.add_argument("--lam-bc", type=float, default=10.0)
    train_sub.add_argument("--checkpoint", type=str, default=None)
    train_sub.add_argument("--seed", type=int, default=42)
    train_sub.add_argument("--print-every", type=int, default=None)
    train_sub.add_argument("--eval-every", type=int, default=None)
    train_sub.add_argument("--device", type=str, default=None)

    # --- unified test sub-command ---
    test_sub = sub.add_parser("test", help="Evaluate a solver checkpoint (fno|pinn)")
    test_sub.add_argument("--solver", type=str, choices=["fno", "pinn"], required=True)
    test_sub.add_argument("--test-dataset", type=str, required=True,
                          help="Path to held-out test .npz dataset")
    test_sub.add_argument("--checkpoint", type=str, default=None)
    test_sub.add_argument("--resolution", type=int, default=None)
    test_sub.add_argument("--width", type=int, default=32,
                          help="FNO only")
    test_sub.add_argument("--modes", type=int, default=12,
                          help="FNO only")
    test_sub.add_argument("--hidden", type=int, default=64,
                          help="PINN only")
    test_sub.add_argument("--layers", type=int, default=4)
    test_sub.add_argument("--print-every", type=int, default=10)
    test_sub.add_argument("--device", type=str, default=None)

    # --- unified manifest sub-command ---
    man_sub = sub.add_parser("manifest", help="Build OOD manifest from dataset features")
    man_sub.add_argument("--train-dataset", type=str, required=True)
    man_sub.add_argument("--val-dataset", type=str, default=None)
    man_sub.add_argument("--out", type=str, default="pretrained_models/fno_manifest.npz")
    man_sub.add_argument("--percentile", type=float, default=95.0)

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
    fno_gen_sub.add_argument("--ood-percentile",  type=float, default=95.0,
                         help="KNN threshold percentile for OOD detection (default: 95.0)")
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
    fno_sub.add_argument("--ood-percentile", type=float, default=95.0,
                       help="KNN threshold percentile for OOD detection (default: 95.0)")

    # --- pinn sub-command ---
    pinn_sub = sub.add_parser("pinn", help="Train the shared PINN solution operator")
    pinn_sub.add_argument("--train-dataset",     type=str,   required=True,
                        help="Shared dataset generated by generate/fno-generate")
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
    pinn_sub.add_argument("--lam-data",           type=float, default=1.0,
                        help="Data-loss weight lambda_data (default: 1.0)")
    pinn_sub.add_argument("--lam-phys",           type=float, default=1.0,
                        help="Physics-loss weight λ_phys (default: 1.0)")
    pinn_sub.add_argument("--lam-bc",             type=float, default=10.0,
                        help="BC-loss weight λ_bc (default: 10.0)")
    pinn_sub.add_argument("--seed",               type=int,   default=42)
    pinn_sub.add_argument("--pinn-path",          type=str,   default="pretrained_models/pinn.pt")
    pinn_sub.add_argument("--device",             type=str,   default=None)
    pinn_sub.add_argument("--print-every",        type=int,   default=500)
    pinn_sub.add_argument("--eval-every",         type=int,   default=1,
                        help="Evaluate val set every N epochs (default: 1)")

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

    # --- pdespace-smoke sub-command ---
    smoke_sub = sub.add_parser(
        "pdespace-smoke",
        help="Run PDESpace coverage + FD-solvability smoke checks (regression guardrail)",
    )
    smoke_sub.add_argument(
        "--space",
        type=str,
        default="thermal-v2",
        choices=["default", "thermal-v2"],
        help="PDESpace preset to evaluate",
    )
    smoke_sub.add_argument("--samples", type=int, default=240,
                           help="Number of sampled PDE problems")
    smoke_sub.add_argument("--seed", type=int, default=42)
    smoke_sub.add_argument("--fd-samples", type=int, default=48,
                           help="Subset size for FD solvability check")
    smoke_sub.add_argument("--fd-resolution", type=int, default=32)
    smoke_sub.add_argument("--fd-max-iterations", type=int, default=1000)
    smoke_sub.add_argument("--fd-tolerance", type=float, default=1e-5)

    # Coverage / stability thresholds
    smoke_sub.add_argument("--min-d-nonzero", type=float, default=0.08)
    smoke_sub.add_argument("--min-e-nonzero", type=float, default=0.08)
    smoke_sub.add_argument("--min-gaussian-rhs", type=float, default=0.20)
    smoke_sub.add_argument("--min-stratified-rhs", type=float, default=0.15)
    smoke_sub.add_argument("--min-neumann-robin", type=float, default=0.20)
    smoke_sub.add_argument("--min-fd-ok", type=float, default=0.25)

    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    # --- unified commands ---
    if args.command == "generate":
        _run_generate_command(args)
        return
    if args.command == "train":
        _run_train_command(args)
        return
    if args.command == "manifest":
        _run_manifest_command(args)
        return
    if args.command == "test":
        _run_test_command(args)
        return

    # --- legacy commands (kept for backward compatibility) ---
    if args.command == "fno-generate":
        _run_fno_generate(args)
        return
    if args.command == "fno-train":
        _run_fno_train(args)
        return
    if args.command == "fno":
        _run_fno_one_shot(args)
        return
    if args.command == "pinn":
        _run_pinn_one_shot(args)
        return
    if args.command == "fno-test":
        _run_fno_test(args)
        return
    if args.command == "pinn-test":
        _run_pinn_test(args)
        return
    if args.command == "pdespace-smoke":
        _run_pdespace_smoke_command(args)
        return

    raise RuntimeError(f"Unknown command {args.command!r}")


if __name__ == "__main__":
    # On macOS, Python 3.8+ already defaults to 'spawn', but be explicit so
    # worker processes don't inherit CUDA state or inconsistent module state.
    multiprocessing.set_start_method("spawn", force=True)
    main()
