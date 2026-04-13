from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def hybrid_loss(
    u_pred: torch.Tensor,
    target: torch.Tensor | None,
    pde_obj: Any,
    x_1d: torch.Tensor,
    y_1d: torch.Tensor,
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
    loss = loss + lam_phys * pde_obj.compute_pde_loss(u_pred)
    loss = loss + lam_bc * pde_obj.compute_bc_loss(u_pred, x_1d, y_1d)
    return loss


def evaluate_operator(
    model: nn.Module,
    loader: DataLoader,
    loss_cfg: dict[str, float],
) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["input"]).squeeze(0).squeeze(-1)
            target = batch["target"].squeeze(0) if batch["target"] is not None else None
            loss = hybrid_loss(
                u_pred=pred,
                target=target,
                pde_obj=batch["pde_obj"],
                x_1d=batch["x_1d"],
                y_1d=batch["y_1d"],
                with_data=batch["has_target"],
                **loss_cfg,
            )
            total += loss.item()
            n += 1
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
            pred = model(batch["input"]).squeeze(0).squeeze(-1)
            target = batch["target"].squeeze(0) if batch["target"] is not None else None
            loss = hybrid_loss(
                u_pred=pred,
                target=target,
                pde_obj=batch["pde_obj"],
                x_1d=batch["x_1d"],
                y_1d=batch["y_1d"],
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
                maybe_eval(step)

        if eval_mode == "epoch":
            maybe_eval(epoch)

    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return {
        "best_val": best_val,
        "best_state": best_state,
        "last_state": last_state,
        "step": step,
    }
