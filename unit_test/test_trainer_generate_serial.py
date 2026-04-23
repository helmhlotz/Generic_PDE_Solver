import numpy as np
import pytest

import trainer


def test_shared_fd_data_generator_runs_serially_when_n_workers_is_one(tmp_path, monkeypatch):
    problems = [
        {"pde_str": "pde-0", "bc_dict": {"left": {"type": "dirichlet", "value": "0"}}},
        {"pde_str": "pde-1", "bc_dict": {"left": {"type": "dirichlet", "value": "1"}}},
    ]

    class _FakeSampler:
        def __init__(self, config):
            self.config = config

        def generate(self, n_samples, seed):
            assert n_samples == 2
            assert seed == 7
            return problems

    seen = []

    def _fake_solve_one_worker(args):
        problem = args[0]
        seen.append(problem["pde_str"])
        index = len(seen)
        return {
            "input": np.full((4, 4, 13), index, dtype=np.float32),
            "target": np.full((4, 4), index, dtype=np.float32),
            "pde_str": problem["pde_str"],
            "bc_json": problem["bc_dict"]["left"]["value"],
            "feat": np.full((3,), index, dtype=np.float32),
        }

    monkeypatch.setattr(trainer, "LHSSampler", _FakeSampler)
    monkeypatch.setattr(trainer, "_solve_one_worker", _fake_solve_one_worker)

    out_path = tmp_path / "train_data.npz"
    generator = trainer.SharedFDDataGenerator(n_points=4, device="cpu")

    count = generator.generate(
        n_samples=2,
        seed=7,
        save_path=str(out_path),
        n_workers=1,
        chunk_size=10,
    )

    assert count == 2
    assert seen == ["pde-0", "pde-1"]

    artifact = np.load(out_path, allow_pickle=True)
    assert artifact["inputs"].shape == (2, 4, 4, 13)
    assert artifact["targets"].shape == (2, 4, 4)
    assert artifact["pde_strs"].tolist() == ["pde-0", "pde-1"]


def test_shared_fd_data_generator_validates_and_warns_on_legacy_resume_chunk(
    tmp_path,
    monkeypatch,
):
    problems = [{"pde_str": "pde-0", "bc_dict": {"left": {"type": "dirichlet", "value": "0"}}}]

    class _FakeSampler:
        def __init__(self, config):
            self.config = config

        def generate(self, n_samples, seed):
            assert n_samples == 1
            return problems

    monkeypatch.setattr(trainer, "LHSSampler", _FakeSampler)

    chunk_path = tmp_path / "train_data_chunk_000000.npz"
    np.savez_compressed(
        chunk_path,
        inputs=np.zeros((1, 4, 4, 13), dtype=np.float32),
        targets=np.ones((1, 4, 4), dtype=np.float32),
        pde_strs=np.array(["pde-0"], dtype=object),
        bc_dict_json=np.array(['{"left": {"type": "dirichlet", "value": "0"}}'], dtype=object),
        feats=np.zeros((1, 3), dtype=np.float32),
    )

    out_path = tmp_path / "train_data.npz"
    generator = trainer.SharedFDDataGenerator(n_points=4, device="cpu")

    with pytest.warns(UserWarning, match="bc_dict_json"):
        count = generator.generate(
            n_samples=1,
            seed=7,
            save_path=str(out_path),
            n_workers=1,
            chunk_size=10,
        )

    assert count == 1
    artifact = np.load(out_path, allow_pickle=True)
    assert artifact["bc_json"].tolist() == ['{"left": {"type": "dirichlet", "value": "0"}}']


def test_unified_train_eval_every_defaults_to_one_epoch():
    parser = trainer._make_parser()

    args = parser.parse_args(["train", "--solver", "fno", "--train-dataset", "train.npz"])

    assert args.eval_every == 1
    assert args.auto_val_fraction == 0.2


@pytest.mark.parametrize("command", [["train"], ["fno-train"], ["fno"]])
def test_eval_every_help_uses_epoch_units(command, capsys):
    parser = trainer._make_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([*command, "--help"])

    captured = capsys.readouterr()
    assert "Evaluate validation set every N epochs" in captured.out


def test_unified_generate_warns_with_exact_manifest_command(tmp_path, monkeypatch, caplog):
    parser = trainer._make_parser()
    args = parser.parse_args(
        [
            "generate",
            "--dataset-path",
            str(tmp_path / "train_data.npz"),
            "--val-dataset-path",
            str(tmp_path / "val_data.npz"),
            "--n-val",
            "8",
        ]
    )

    generated = []

    class _FakeGenerator:
        def __init__(self, n_points, device):
            self.n_points = n_points
            self.device = device

        def generate(self, **kwargs):
            generated.append(kwargs["save_path"])
            return 8 if kwargs["save_path"] == args.val_dataset_path else 32

    monkeypatch.setattr(trainer, "SharedFDDataGenerator", _FakeGenerator)

    with caplog.at_level("WARNING"):
        trainer._run_generate_command(args)

    assert generated == [args.dataset_path, args.val_dataset_path]
    assert "--train-dataset " + args.dataset_path in caplog.text
    assert "--val-dataset " + args.val_dataset_path in caplog.text
    assert "--out " + str(tmp_path / "manifest.npz") in caplog.text


def test_unified_generate_builds_manifest_when_requested(tmp_path, monkeypatch):
    parser = trainer._make_parser()
    args = parser.parse_args(
        [
            "generate",
            "--dataset-path",
            str(tmp_path / "train_data.npz"),
            "--val-dataset-path",
            str(tmp_path / "val_data.npz"),
            "--n-val",
            "4",
            "--manifest-path",
            str(tmp_path / "manifest.npz"),
        ]
    )

    calls = []

    class _FakeGenerator:
        def __init__(self, n_points, device):
            self.n_points = n_points
            self.device = device

        def generate(self, **kwargs):
            return 4

    def _fake_build_ood_manifest(*, train_dataset, out_path, percentile, val_dataset):
        calls.append((train_dataset, out_path, percentile, val_dataset))

    monkeypatch.setattr(trainer, "SharedFDDataGenerator", _FakeGenerator)
    monkeypatch.setattr(trainer, "build_ood_manifest", _fake_build_ood_manifest)

    trainer._run_generate_command(args)

    assert calls == [
        (args.dataset_path, args.manifest_path, args.ood_percentile, args.val_dataset_path)
    ]


def test_fno_trainer_rejects_physics_or_bc_loss():
    with pytest.raises(ValueError, match="collate_supervised_batch"):
        trainer.FNOTrainer(lam_phys=1.0, device="cpu")

    with pytest.raises(ValueError, match="collate_supervised_batch"):
        trainer.FNOTrainer(lam_bc=1.0, device="cpu")


def test_checkpoint_path_derivation_is_suffix_safe():
    assert trainer._derived_checkpoint_path("models/fno") == trainer.Path("models/fno.ckpt")
    assert trainer._derived_best_path("models/fno") == trainer.Path("models/fno_best")
    assert trainer._derived_checkpoint_path("models/fno.pt") == trainer.Path("models/fno.ckpt")
    assert trainer._derived_best_path("models/fno.pt") == trainer.Path("models/fno_best.pt")
