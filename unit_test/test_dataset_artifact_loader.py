import json

import numpy as np
import pytest
import torch

from dataset import (
    CANONICAL_DATASET_SCHEMA_VERSION,
    DatasetArtifactError,
    MaterializedOperatorDataset,
    OperatorMaterializationConfig,
    load_operator_dataset,
    load_dataset_artifact,
    materialize_operator_examples,
)


def _artifact_kwargs(*, bc_key: str = "bc_json") -> dict:
    return {
        "schema_version": np.array([CANONICAL_DATASET_SCHEMA_VERSION], dtype=np.int32),
        "inputs": np.zeros((2, 4, 4, 7), dtype=np.float32),
        "targets": np.ones((2, 4, 4), dtype=np.float32),
        "feats": np.zeros((2, 25), dtype=np.float32),
        "pde_strs": np.array(["u_xx + u_yy = 0", "u_xx + u_yy = 1"], dtype=object),
        bc_key: np.array(
            [
                json.dumps(
                    {
                        "left": {"type": "dirichlet", "value": "0"},
                        "right": {"type": "dirichlet", "value": "0"},
                        "bottom": {"type": "dirichlet", "value": "0"},
                        "top": {"type": "dirichlet", "value": "0"},
                    }
                ),
                json.dumps(
                    {
                        "left": {"type": "dirichlet", "value": "1"},
                        "right": {"type": "dirichlet", "value": "1"},
                        "bottom": {"type": "dirichlet", "value": "1"},
                        "top": {"type": "dirichlet", "value": "1"},
                    }
                ),
            ],
            dtype=object,
        ),
        "n_points": np.array([4], dtype=np.int32),
    }


def test_load_dataset_artifact_accepts_canonical_schema(tmp_path):
    path = tmp_path / "canonical.npz"
    np.savez_compressed(path, **_artifact_kwargs())

    artifact = load_dataset_artifact(str(path))

    assert artifact.schema_version == CANONICAL_DATASET_SCHEMA_VERSION
    assert artifact.num_samples == 2
    assert artifact.n_points == 4
    assert artifact.inputs.shape == (2, 4, 4, 7)
    assert artifact.targets.shape == (2, 4, 4)
    assert artifact.feats.shape == (2, 25)


def test_load_dataset_artifact_normalizes_legacy_bc_key(tmp_path):
    path = tmp_path / "legacy.npz"
    kwargs = _artifact_kwargs(bc_key="bc_dict_json")
    kwargs.pop("schema_version")
    np.savez_compressed(path, **kwargs)

    artifact = load_dataset_artifact(str(path))

    assert artifact.schema_version == 0
    assert artifact.bc_json.shape == (2,)
    assert all(isinstance(item, str) for item in artifact.bc_json.tolist())


def test_load_dataset_artifact_requires_feats(tmp_path):
    path = tmp_path / "missing_feats.npz"
    kwargs = _artifact_kwargs()
    kwargs.pop("feats")
    np.savez_compressed(path, **kwargs)

    with pytest.raises(DatasetArtifactError, match="feats"):
        load_dataset_artifact(str(path))


def test_materialize_operator_examples_from_canonical_artifact(tmp_path):
    path = tmp_path / "materialize.npz"
    np.savez_compressed(path, **_artifact_kwargs())

    artifact = load_dataset_artifact(str(path))
    examples = materialize_operator_examples(
        artifact,
        config=OperatorMaterializationConfig(device=torch.device("cpu")),
    )

    assert len(examples) == 2
    assert examples[0]["input"].shape == (4, 4, 7)
    assert examples[0]["target"].shape == (4, 4)
    assert examples[0]["has_target"] is True


def test_load_operator_dataset_builds_training_dataset_from_artifact(tmp_path):
    path = tmp_path / "operator_dataset.npz"
    np.savez_compressed(path, **_artifact_kwargs())

    dataset = load_operator_dataset(
        str(path),
        config=OperatorMaterializationConfig(device=torch.device("cpu")),
    )

    assert isinstance(dataset, MaterializedOperatorDataset)
    assert len(dataset) == 2
    sample = dataset[0]
    assert sample["input"].shape == (4, 4, 7)
    assert sample["target"].shape == (4, 4)
