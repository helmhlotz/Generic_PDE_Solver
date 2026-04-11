"""Conditional input encoders for PDE solver models."""

from __future__ import annotations

from typing import Callable

import torch

from pde_parser import ParsedBC


class ConditionalGrid2D:
    """Build the shared 7-channel input tensor for the conditional solvers.

    Channel layout:
        0: x coordinate
        1: y coordinate
        2: source term
        3: boundary-condition RHS/value field on the boundary
        4: repeated RHS field for derivative-involving BCs (beta != 0)
        5: alpha coefficient
        6: beta coefficient

    Channel 4 is intentionally not a numerically estimated normal-flux field.
    """

    def __init__(
        self,
        n_points: int,
        bc_specs: dict[str, ParsedBC],
        source_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None,
        device: torch.device,
        t_value: float = 0.0,
    ):
        self.n_points = n_points
        self.device = device

        x_1d = torch.linspace(0, 1, n_points, device=device)
        y_1d = torch.linspace(0, 1, n_points, device=device)
        xx, yy = torch.meshgrid(x_1d, y_1d, indexing="ij")

        self.x_1d = x_1d
        self.y_1d = y_1d
        self.xx = xx
        self.yy = yy
        self.tt = torch.full_like(xx, float(t_value))

        ch_x = xx
        ch_y = yy
        if source_fn:
            src_raw = source_fn(xx, yy, self.tt)
            if not isinstance(src_raw, torch.Tensor):
                src_raw = torch.tensor(float(src_raw), dtype=xx.dtype, device=xx.device)
            ch_src = src_raw.expand_as(xx) if src_raw.shape != xx.shape else src_raw
        else:
            ch_src = torch.zeros_like(xx)

        ch_val, ch_flux, ch_alpha, ch_beta = self._encode_bcs(bc_specs, xx, yy, n_points)

        self.input_grid = torch.stack(
            [ch_x, ch_y, ch_src, ch_val, ch_flux, ch_alpha, ch_beta], dim=-1
        ).unsqueeze(0)

    @staticmethod
    def _encode_bcs(
        bc_specs: dict[str, ParsedBC],
        xx: torch.Tensor,
        yy: torch.Tensor,
        n: int,
    ):
        ch_val = torch.zeros(n, n, device=xx.device)
        ch_flux = torch.zeros(n, n, device=xx.device)
        ch_alpha = torch.zeros(n, n, device=xx.device)
        ch_beta = torch.zeros(n, n, device=xx.device)

        wall_slices = {
            "left": (0, slice(None)),
            "right": (-1, slice(None)),
            "bottom": (slice(None), 0),
            "top": (slice(None), -1),
        }

        for wall, spec in bc_specs.items():
            xi, yi = wall_slices[wall]
            xx_w = xx[xi, yi]
            yy_w = yy[xi, yi]

            val_fn = spec.value_fn()
            v = val_fn(xx_w, yy_w)

            ch_val[xi, yi] = v
            ch_flux[xi, yi] = v if spec.beta != 0 else torch.zeros_like(v)
            ch_alpha[xi, yi] = spec.alpha
            ch_beta[xi, yi] = spec.beta

        return ch_val, ch_flux, ch_alpha, ch_beta


__all__ = ["ConditionalGrid2D"]
