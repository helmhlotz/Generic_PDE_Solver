import json

import numpy as np
import pytest

import evaluate
from dataset import CANONICAL_DATASET_SCHEMA_VERSION
from inference_engine import GridConfig, SolveResult, SolverOption


def _write_dataset(path, *, bc_key: str) -> np.ndarray:
    target = np.array(
        [
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0],
            ]
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        path,
        schema_version=np.array([CANONICAL_DATASET_SCHEMA_VERSION], dtype=np.int32),
        inputs=np.zeros((1, 4, 4, 13), dtype=np.float32),
        targets=target,
        feats=np.zeros((1, 25), dtype=np.float32),
        pde_strs=np.array(["u_xx + u_yy = 0"], dtype=object),
        n_points=np.array([4], dtype=np.int32),
        **{
            bc_key: np.array(
                [
                    json.dumps(
                        {
                            "left": {"type": "dirichlet", "value": "0"},
                            "right": {"type": "dirichlet", "value": "0"},
                            "bottom": {"type": "dirichlet", "value": "0"},
                            "top": {"type": "dirichlet", "value": "0"},
                        }
                    )
                ],
                dtype=object,
            )
        },
    )
    return target[0]


def test_evaluate_dataset_e2e_accepts_current_bc_json_schema(tmp_path, monkeypatch):
    dataset_path = tmp_path / "eval_data_current.npz"
    expected = _write_dataset(dataset_path, bc_key="bc_json")

    class _FakeEngine:
        def __init__(self, solver_option):
            self.solver_option = solver_option

        def solve(self, pde_str, bc_dict, print_every):
            return SolveResult(
                u=expected.copy(),
                xx=np.zeros_like(expected),
                yy=np.zeros_like(expected),
                residual=0.25,
                bc_error=0.5,
                method=self.solver_option.solver_type,
            )

    monkeypatch.setattr(evaluate, "InferenceEngine", _FakeEngine)

    metrics, summary = evaluate.evaluate_dataset(
        str(dataset_path),
        SolverOption(solver_type="deeponet", grid=GridConfig(n_points=4)),
        print_every=1,
    )

    assert len(metrics) == 1
    assert metrics[0].method == "deeponet"
    assert metrics[0].rel_l2 == 0.0
    assert metrics[0].rmse == 0.0
    assert summary["n_samples"] == 1
    assert summary["n_failed"] == 0
    assert summary["rel_l2_mean"] == 0.0
    assert summary["rmse_mean"] == 0.0
    assert summary["bc_error_mean"] == 0.5
    assert summary["pde_res_mean"] == 0.25


def test_evaluate_dataset_e2e_accepts_legacy_bc_dict_json_schema(tmp_path, monkeypatch):
    dataset_path = tmp_path / "eval_data_legacy.npz"
    expected = _write_dataset(dataset_path, bc_key="bc_dict_json")

    class _FakeEngine:
        def __init__(self, solver_option):
            self.solver_option = solver_option

        def solve(self, pde_str, bc_dict, print_every):
            return SolveResult(
                u=expected.copy(),
                xx=np.zeros_like(expected),
                yy=np.zeros_like(expected),
                residual=0.0,
                bc_error=0.0,
                method=self.solver_option.solver_type,
            )

    monkeypatch.setattr(evaluate, "InferenceEngine", _FakeEngine)

    metrics, summary = evaluate.evaluate_dataset(
        str(dataset_path),
        SolverOption(solver_type="fd", grid=GridConfig(n_points=4)),
        print_every=1,
    )

    assert len(metrics) == 1
    assert metrics[0].method == "fd"
    assert summary["n_samples"] == 1
    assert summary["n_failed"] == 0


def test_evaluate_dataset_fails_early_on_resolution_mismatch(tmp_path):
    dataset_path = tmp_path / "eval_data_current.npz"
    _write_dataset(dataset_path, bc_key="bc_json")

    with pytest.raises(ValueError, match="Re-run with --n-points 4"):
        evaluate.evaluate_dataset(
            str(dataset_path),
            SolverOption(solver_type="fd"),
            print_every=1,
        )
