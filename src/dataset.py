from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from models.conditional_inputs import ConditionalGrid2D
from pde_parser import parse_bc, parse_pde
from physics.pde_helpers import GeneralPDE


CANONICAL_DATASET_SCHEMA_VERSION = 1


class DatasetArtifactError(ValueError):
    """Raised when a dataset artifact is missing required fields or is malformed."""


@dataclass(frozen=True)
class DatasetArtifact:
    """Canonical in-memory representation of a persisted dataset artifact."""

    schema_version: int
    inputs: np.ndarray
    targets: np.ndarray
    feats: np.ndarray
    pde_strs: np.ndarray
    bc_json: np.ndarray
    n_points: int
    num_samples: int


@dataclass(frozen=True)
class OperatorMaterializationConfig:
    """Controls how canonical dataset artifacts are materialized for training."""

    device: torch.device
    n_points: int | None = None
    use_targets: bool = True
    rebuild_inputs: bool = False


def _require_field(data: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    if name not in data:
        raise DatasetArtifactError(f"Dataset artifact is missing required field {name!r}.")
    return data[name]


def _extract_n_points(data: np.lib.npyio.NpzFile, inputs: np.ndarray) -> int:
    if "n_points" in data:
        raw = np.asarray(data["n_points"])
        if raw.size != 1:
            raise DatasetArtifactError("Dataset field 'n_points' must be a scalar or length-1 array.")
        return int(raw.reshape(-1)[0])
    return int(inputs.shape[1])


def _extract_schema_version(data: np.lib.npyio.NpzFile) -> int:
    if "schema_version" in data:
        raw = np.asarray(data["schema_version"])
        if raw.size != 1:
            raise DatasetArtifactError(
                "Dataset field 'schema_version' must be a scalar or length-1 array."
            )
        return int(raw.reshape(-1)[0])
    return 0


def load_dataset_artifact(path: str) -> DatasetArtifact:
    """Load and validate a dataset artifact, normalizing legacy field names."""
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Dataset artifact not found: {path}")

    data = np.load(str(artifact_path), allow_pickle=True)
    inputs = _require_field(data, "inputs")
    targets = _require_field(data, "targets")
    feats = _require_field(data, "feats")
    pde_strs = _require_field(data, "pde_strs")
    bc_json = data["bc_json"] if "bc_json" in data else _require_field(data, "bc_dict_json")
    schema_version = _extract_schema_version(data)
    n_points = _extract_n_points(data, inputs)

    if inputs.ndim != 4:
        raise DatasetArtifactError(f"'inputs' must have shape (N, n, n, C); got {inputs.shape}.")
    if targets.ndim != 3:
        raise DatasetArtifactError(f"'targets' must have shape (N, n, n); got {targets.shape}.")
    if feats.ndim != 2:
        raise DatasetArtifactError(f"'feats' must have shape (N, D); got {feats.shape}.")

    num_samples = int(inputs.shape[0])
    if targets.shape[0] != num_samples or feats.shape[0] != num_samples:
        raise DatasetArtifactError(
            "Dataset sample counts do not match across 'inputs', 'targets', and 'feats'."
        )
    if len(pde_strs) != num_samples or len(bc_json) != num_samples:
        raise DatasetArtifactError(
            "Dataset sample counts do not match across 'pde_strs' and 'bc_json'."
        )
    if inputs.shape[1] != inputs.shape[2]:
        raise DatasetArtifactError(f"'inputs' must be square in spatial dimensions; got {inputs.shape}.")
    if targets.shape[1] != targets.shape[2]:
        raise DatasetArtifactError(
            f"'targets' must be square in spatial dimensions; got {targets.shape}."
        )
    if inputs.shape[1] != n_points or targets.shape[1] != n_points:
        raise DatasetArtifactError(
            f"'n_points'={n_points} does not match persisted tensor resolution "
            f"inputs={inputs.shape[1]} / targets={targets.shape[1]}."
        )

    normalized_pde_strs = np.asarray([str(item) for item in pde_strs], dtype=object)
    normalized_bc_json = np.asarray([str(item) for item in bc_json], dtype=object)
    for idx, raw in enumerate(normalized_bc_json):
        try:
            json.loads(raw)
        except Exception as exc:
            raise DatasetArtifactError(
                f"Dataset field 'bc_json' contains invalid JSON at sample {idx}."
            ) from exc

    return DatasetArtifact(
        schema_version=schema_version,
        inputs=np.asarray(inputs, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        feats=np.asarray(feats, dtype=np.float32),
        pde_strs=normalized_pde_strs,
        bc_json=normalized_bc_json,
        n_points=n_points,
        num_samples=num_samples,
    )


def materialize_operator_examples(
    artifact: DatasetArtifact,
    *,
    config: OperatorMaterializationConfig,
) -> list[dict[str, Any]]:
    """Convert a canonical dataset artifact into operator-training examples."""
    examples: list[dict[str, Any]] = []
    for idx, (pde_str_raw, bc_json_raw) in enumerate(zip(artifact.pde_strs, artifact.bc_json)):
        parsed = parse_pde(str(pde_str_raw))
        bc_dict = json.loads(str(bc_json_raw))
        bc_specs = parse_bc(bc_dict)
        pde_obj = GeneralPDE(parsed, bc_specs)

        sample_n_points = int(config.n_points or artifact.inputs[idx].shape[0])
        if config.rebuild_inputs or artifact.inputs[idx].shape[0] != sample_n_points:
            source_fn = parsed.rhs_fn()
            grid = ConditionalGrid2D(sample_n_points, bc_specs, source_fn, config.device)
            input_tensor = grid.input_grid.squeeze(0)
            x_1d = grid.x_1d
            y_1d = grid.y_1d
        else:
            input_tensor = torch.tensor(
                artifact.inputs[idx],
                dtype=torch.float32,
                device=config.device,
            )
            x_1d = torch.linspace(0, 1, input_tensor.shape[0], device=config.device)
            y_1d = torch.linspace(0, 1, input_tensor.shape[1], device=config.device)

        target_tensor: torch.Tensor | None = None
        if config.use_targets:
            target_tensor = torch.tensor(
                artifact.targets[idx],
                dtype=torch.float32,
                device=config.device,
            )
            if target_tensor.shape[0] != sample_n_points:
                target_tensor = torch.nn.functional.interpolate(
                    target_tensor.unsqueeze(0).unsqueeze(0),
                    size=(sample_n_points, sample_n_points),
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(0).squeeze(0)

        examples.append(
            {
                "input": input_tensor,
                "target": target_tensor,
                "has_target": target_tensor is not None,
                "pde_obj": pde_obj,
                "x_1d": x_1d,
                "y_1d": y_1d,
            }
        )
    return examples


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
        artifact = load_dataset_artifact(npz_path)
        self.examples.extend(
            materialize_operator_examples(
                artifact,
                config=OperatorMaterializationConfig(
                    device=self.device,
                    n_points=n_points,
                    use_targets=self.use_targets,
                    rebuild_inputs=rebuild_inputs,
                ),
            )
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


class MaterializedOperatorDataset(Dataset):
    """Dataset wrapper around already-materialized operator examples."""

    def __init__(self, examples: list[dict[str, Any]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


def load_operator_dataset(
    path: str,
    *,
    config: OperatorMaterializationConfig,
) -> MaterializedOperatorDataset:
    """Load a canonical dataset artifact and materialize training examples."""
    artifact = load_dataset_artifact(path)
    examples = materialize_operator_examples(artifact, config=config)
    return MaterializedOperatorDataset(examples)


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

    PDE metadata remains per-sample so the shared loss path can evaluate
    physics and BC terms independently across a mini-batch.
    """
    has_targets = {sample["has_target"] for sample in batch}
    if len(has_targets) != 1:
        raise ValueError("All samples in a batch must agree on target availability")

    target = None
    if batch[0]["target"] is not None:
        target = torch.stack([sample["target"] for sample in batch], dim=0)

    return {
        "input": torch.stack([sample["input"] for sample in batch], dim=0),
        "target": target,
        "has_target": batch[0]["has_target"],
        "pde_obj": [sample["pde_obj"] for sample in batch],
        "x_1d": torch.stack([sample["x_1d"] for sample in batch], dim=0),
        "y_1d": torch.stack([sample["y_1d"] for sample in batch], dim=0),
    }


def collate_supervised_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Batch collation for pure supervised operator learning (data loss only)."""
    has_targets = {sample["has_target"] for sample in batch}
    if has_targets != {True}:
        raise ValueError("collate_supervised_batch requires targets for all samples")

    return {
        "input": torch.stack([sample["input"] for sample in batch], dim=0),
        "target": torch.stack([sample["target"] for sample in batch], dim=0),
        "has_target": True,
    }
