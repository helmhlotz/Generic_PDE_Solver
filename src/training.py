from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _mean_physics_loss(
    u_pred: torch.Tensor,
    pde_obj: Any,
) -> torch.Tensor:
    if u_pred.ndim == 2:
        return pde_obj.compute_pde_loss(u_pred)
    if len(pde_obj) != u_pred.shape[0]:
        raise ValueError("pde_obj batch length must match prediction batch size")
    losses = [obj.compute_pde_loss(u_pred[idx]) for idx, obj in enumerate(pde_obj)]
    return torch.stack(losses).mean()


def _mean_bc_loss(
    u_pred: torch.Tensor,
    pde_obj: Any,
    x_1d: torch.Tensor,
    y_1d: torch.Tensor,
) -> torch.Tensor:
    if u_pred.ndim == 2:
        return pde_obj.compute_bc_loss(u_pred, x_1d, y_1d)
    if len(pde_obj) != u_pred.shape[0]:
        raise ValueError("pde_obj batch length must match prediction batch size")
    if x_1d.ndim != 2 or y_1d.ndim != 2:
        raise ValueError("Batched x_1d and y_1d tensors must have shape (batch, n_points)")
    losses = [
        obj.compute_bc_loss(u_pred[idx], x_1d[idx], y_1d[idx])
        for idx, obj in enumerate(pde_obj)
    ]
    return torch.stack(losses).mean()


def hybrid_loss(
    u_pred: torch.Tensor,
    target: torch.Tensor | None,
    pde_obj: Any | None,
    x_1d: torch.Tensor | None,
    y_1d: torch.Tensor | None,
    lam_data: float = 1.0,
    lam_phys: float = 1.0,
    lam_bc: float = 1.0,
    with_data: bool = True,
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=u_pred.device)
    if with_data:
        if target is None:
            raise ValueError("target is required when with_data=True")
        loss = loss + lam_data * torch.mean((u_pred - target) ** 2)
    if lam_phys != 0.0:
        if pde_obj is None:
            raise ValueError("pde_obj is required when lam_phys != 0")
        loss = loss + lam_phys * _mean_physics_loss(u_pred, pde_obj)
    if lam_bc != 0.0:
        if pde_obj is None or x_1d is None or y_1d is None:
            raise ValueError("pde_obj, x_1d, y_1d are required when lam_bc != 0")
        loss = loss + lam_bc * _mean_bc_loss(u_pred, pde_obj, x_1d, y_1d)
    return loss


def evaluate_operator(
    model: nn.Module,
    loader: DataLoader,
    loss_cfg: dict[str, float],
) -> float:
    model.eval()
    total = 0.0
    n = 0
    try:
        with torch.no_grad():
            for batch in loader:
                pred = model(batch["input"]).squeeze(-1)
                target = batch["target"]
                loss = hybrid_loss(
                    u_pred=pred,
                    target=target,
                    pde_obj=batch.get("pde_obj"),
                    x_1d=batch.get("x_1d"),
                    y_1d=batch.get("y_1d"),
                    with_data=batch["has_target"],
                    **loss_cfg,
                )
                total += loss.item()
                n += 1
    finally:
        model.train()
    return total / max(n, 1)


def train_operator(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    loss_cfg: dict[str, float],
    epochs: int,
    print_every: int,
    eval_every: int,
    eval_mode: str = "epoch",
    checkpoint_path: str | None = None,
    resume: str | None = None,
) -> dict[str, Any]:
    """Train an operator model with optional periodic validation.

    Parameters
    ----------
    eval_every : int
        Validation frequency. The units depend on ``eval_mode``:
        ``eval_mode="epoch"`` evaluates every N epochs, while
        ``eval_mode="step"`` evaluates every N gradient steps.
        Public trainer entrypoints use epoch mode.
    """
    start_epoch = 1
    best_val = float("inf")
    step = 0

    if resume:
        ckpt = torch.load(resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["opt_state"])
        if scheduler is not None and ckpt.get("sched_state") is not None:
            scheduler.load_state_dict(ckpt["sched_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt["best_val"])
        step = int(ckpt.get("step", 0))
        print(f"Resuming from epoch {start_epoch}")

    best_state: dict[str, torch.Tensor] | None = None
    last_state: dict[str, torch.Tensor] | None = None

    def maybe_eval(marker: int) -> None:
        nonlocal best_val, best_state
        if val_loader is None or marker % eval_every != 0:
            return
        val_loss = evaluate_operator(model, val_loader, loss_cfg)
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "step": step,
                        "model_state": model.state_dict(),
                        "opt_state": optimizer.state_dict(),
                        "sched_state": scheduler.state_dict() if scheduler is not None else None,
                        "best_val": best_val,
                    },
                    checkpoint_path,
                )
        marker_prefix = (
            f"Step {step:>7d}"
            if eval_mode == "step"
            else f"Epoch {epoch:3d}/{epochs}"
        )
        marker_suffix = " *" if improved else ""
        print(f"{marker_prefix} | val={val_loss:.4e} best={best_val:.4e}{marker_suffix}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch["input"]).squeeze(-1)
            target = batch["target"]
            loss = hybrid_loss(
                u_pred=pred,
                target=target,
                pde_obj=batch.get("pde_obj"),
                x_1d=batch.get("x_1d"),
                y_1d=batch.get("y_1d"),
                with_data=batch["has_target"],
                **loss_cfg,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            step += 1
            if step % print_every == 0:
                print(f"[{epoch}/{epochs}] step={step:06d} loss={loss.item():.3e}")
            if eval_mode == "step":
                # In step mode, eval_every is interpreted in gradient steps.
                maybe_eval(step)

        if eval_mode == "epoch":
            # Public trainer paths use epoch mode, so eval_every means epochs.
            maybe_eval(epoch)

    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return {
        "best_val": best_val,
        "best_state": best_state,
        "last_state": last_state,
        "step": step,
    }
