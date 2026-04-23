import pytest
import torch
from torch.utils.data import DataLoader

import trainer
import training


class _BoomModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        raise RuntimeError("boom")


def test_evaluate_operator_restores_train_mode_on_exception():
    model = _BoomModel()
    loader = DataLoader(
        [
            {
                "input": torch.zeros((1, 1, 1)),
                "target": torch.zeros((1, 1)),
                "has_target": True,
            }
        ],
        batch_size=1,
        collate_fn=lambda batch: batch[0],
    )

    with pytest.raises(RuntimeError, match="boom"):
        training.evaluate_operator(model, loader, {"lam_data": 1.0, "lam_phys": 0.0, "lam_bc": 0.0})

    assert model.training is True


def test_test_from_dataset_restores_train_mode_on_exception(monkeypatch):
    trainer_obj = trainer.FNOTrainer(device="cpu")

    monkeypatch.setattr(
        trainer_obj,
        "_load_examples",
        lambda *args, **kwargs: [{"input": torch.zeros((4, 4, 13)), "target": torch.zeros((4, 4))}],
    )
    monkeypatch.setattr(trainer_obj.model, "forward", lambda x: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        trainer_obj.test("ignored.npz")

    assert trainer_obj.model.training is True
