from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from models.conditional_inputs import ConditionalGrid2D
from pde_parser import parse_bc, parse_pde
from physics.pde_helpers import GeneralPDE


class PDEOperatorDataset(Dataset):
    """Dataset of conditional PDE operator samples for FNO or PINN training."""

    def __init__(
        self,
        device: torch.device,
        npz_path: str | None = None,
        problems: list[dict[str, Any]] | None = None,
        n_points: int | None = None,
        use_targets: bool = True,
        rebuild_inputs: bool = False,
    ) -> None:
        if npz_path is None and problems is None:
            raise ValueError("Either npz_path or problems must be provided.")

        self.device = device
        self.use_targets = use_targets
        self.examples: list[dict[str, Any]] = []

        if npz_path is not None:
            self._load_from_npz(npz_path, n_points=n_points, rebuild_inputs=rebuild_inputs)
        else:
            self._load_from_problems(problems or [], n_points=n_points)

    def _load_from_npz(
        self,
        npz_path: str,
        n_points: int | None,
        rebuild_inputs: bool,
    ) -> None:
        data = np.load(npz_path, allow_pickle=True)
        inputs = data["inputs"]
        targets = data["targets"] if "targets" in data else None
        pde_strs = data["pde_strs"]
        bc_json = data["bc_json"] if "bc_json" in data else data["bc_dict_json"]

        for idx, (pde_str_raw, bc_json_raw) in enumerate(zip(pde_strs, bc_json)):
            parsed = parse_pde(str(pde_str_raw))
            bc_dict = json.loads(str(bc_json_raw))
            bc_specs = parse_bc(bc_dict)
            pde_obj = GeneralPDE(parsed, bc_specs)

            sample_n_points = int(n_points or inputs[idx].shape[0])
            if rebuild_inputs or inputs[idx].shape[0] != sample_n_points:
                source_fn = parsed.rhs_fn()
                grid = ConditionalGrid2D(sample_n_points, bc_specs, source_fn, self.device)
                input_tensor = grid.input_grid.squeeze(0)
                x_1d = grid.x_1d
                y_1d = grid.y_1d
            else:
                input_tensor = torch.tensor(inputs[idx], dtype=torch.float32, device=self.device)
                x_1d = torch.linspace(0, 1, input_tensor.shape[0], device=self.device)
                y_1d = torch.linspace(0, 1, input_tensor.shape[1], device=self.device)

            target_tensor: torch.Tensor | None = None
            if self.use_targets and targets is not None:
                target_tensor = torch.tensor(targets[idx], dtype=torch.float32, device=self.device)
                if target_tensor.shape[0] != sample_n_points:
                    target_tensor = torch.nn.functional.interpolate(
                        target_tensor.unsqueeze(0).unsqueeze(0),
                        size=(sample_n_points, sample_n_points),
                        mode="bilinear",
                        align_corners=True,
                    ).squeeze(0).squeeze(0)

            self.examples.append(
                {
                    "input": input_tensor,
                    "target": target_tensor,
                    "has_target": target_tensor is not None,
                    "pde_obj": pde_obj,
                    "x_1d": x_1d,
                    "y_1d": y_1d,
                }
            )

    def _load_from_problems(
        self,
        problems: list[dict[str, Any]],
        n_points: int | None,
    ) -> None:
        if n_points is None:
            raise ValueError("n_points is required when building a dataset from problems.")

        for prob in problems:
            parsed = parse_pde(prob["pde_str"])
            bc_specs = parse_bc(prob["bc_dict"])
            source_fn = parsed.rhs_fn()
            grid = ConditionalGrid2D(n_points, bc_specs, source_fn, self.device)
            pde_obj = GeneralPDE(parsed, bc_specs)
            self.examples.append(
                {
                    "input": grid.input_grid.squeeze(0),
                    "target": None,
                    "has_target": False,
                    "pde_obj": pde_obj,
                    "x_1d": grid.x_1d,
                    "y_1d": grid.y_1d,
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


class RepeatDataset(Dataset):
    """Virtual repeat wrapper to emulate multiple steps per problem."""

    def __init__(self, base: Dataset, repeats: int) -> None:
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        self.base = base
        self.repeats = repeats

    def __len__(self) -> int:
        return len(self.base) * self.repeats

    def __getitem__(self, idx: int) -> Any:
        return self.base[idx % len(self.base)]


def collate_operator_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Batch collation for PDE samples.

    The shared loss path currently supports one PDE object per batch, so this
    collate function intentionally keeps `pde_obj` scalar and requires
    `batch_size=1`.
    """
    if len(batch) != 1:
        raise ValueError("collate_operator_batch currently requires batch_size=1")

    sample = batch[0]
    target = sample["target"]
    return {
        "input": sample["input"].unsqueeze(0),
        "target": target.unsqueeze(0) if target is not None else None,
        "has_target": sample["has_target"],
        "pde_obj": sample["pde_obj"],
        "x_1d": sample["x_1d"],
        "y_1d": sample["y_1d"],
    }
