"""Conditional input encoders for PDE solver models."""

from __future__ import annotations

import torch

from pde_parser import ParsedBC, ParsedPDE


CONDITIONAL_INPUT_CHANNELS = 13
BC_TYPE_INTERIOR = 0.0
BC_TYPE_DIRICHLET = 1.0
BC_TYPE_NEUMANN = 2.0
BC_TYPE_ROBIN = 3.0

_BC_TYPE_CODE = {
    "dirichlet": BC_TYPE_DIRICHLET,
    "neumann": BC_TYPE_NEUMANN,
    "robin": BC_TYPE_ROBIN,
}
_BC_TYPE_PRIORITY = {
    "dirichlet": 1,
    "neumann": 2,
    "robin": 3,
}
_WALL_PRIORITY = {
    "top": 1,
    "bottom": 2,
    "right": 3,
    "left": 4,
}


class ConditionalGrid2D:
    """Build the shared 13-channel input tensor for the conditional solvers.

    Channel layout:
        0: x coordinate
        1: y coordinate
        2: PDE coefficient a (u_xx)
        3: PDE coefficient b (u_yy)
        4: PDE coefficient d (u_x)
        5: PDE coefficient e (u_y)
        6: PDE coefficient f (u)
        7: PDE coefficient g (u_t)
        8: source / RHS field
        9: boundary-condition RHS/value field on the boundary
       10: boundary-condition type code
       11: alpha coefficient
       12: beta coefficient
    """

    def __init__(
        self,
        n_points: int,
        parsed_pde: ParsedPDE,
        bc_specs: dict[str, ParsedBC],
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

        source_fn = parsed_pde.rhs_fn()
        src_raw = source_fn(xx, yy, self.tt)
        if not isinstance(src_raw, torch.Tensor):
            src_raw = torch.tensor(float(src_raw), dtype=xx.dtype, device=xx.device)
        ch_src = src_raw.expand_as(xx) if src_raw.shape != xx.shape else src_raw

        ch_val, ch_type, ch_alpha, ch_beta = self._encode_bcs(bc_specs, xx, yy, n_points)

        self.input_grid = torch.stack(
            [
                xx,
                yy,
                torch.full_like(xx, parsed_pde.a),
                torch.full_like(xx, parsed_pde.b),
                torch.full_like(xx, parsed_pde.d),
                torch.full_like(xx, parsed_pde.e),
                torch.full_like(xx, parsed_pde.f),
                torch.full_like(xx, parsed_pde.g),
                ch_src,
                ch_val,
                ch_type,
                ch_alpha,
                ch_beta,
            ],
            dim=-1,
        ).unsqueeze(0)

    @staticmethod
    def _encode_bcs(
        bc_specs: dict[str, ParsedBC],
        xx: torch.Tensor,
        yy: torch.Tensor,
        n: int,
    ):
        ch_val = torch.zeros(n, n, device=xx.device)
        ch_type = torch.zeros(n, n, device=xx.device)
        ch_alpha = torch.zeros(n, n, device=xx.device)
        ch_beta = torch.zeros(n, n, device=xx.device)
        priority = torch.full((n, n), -1, device=xx.device, dtype=torch.int32)

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
            type_priority = _BC_TYPE_PRIORITY[spec.bc_type]
            wall_priority = _WALL_PRIORITY[wall]
            cell_priority = type_priority * 10 + wall_priority
            mask = priority[xi, yi] < cell_priority

            ch_val[xi, yi] = torch.where(mask, v, ch_val[xi, yi])
            ch_type[xi, yi] = torch.where(
                mask,
                torch.full_like(v, _BC_TYPE_CODE[spec.bc_type]),
                ch_type[xi, yi],
            )
            ch_alpha[xi, yi] = torch.where(mask, torch.full_like(v, spec.alpha), ch_alpha[xi, yi])
            ch_beta[xi, yi] = torch.where(mask, torch.full_like(v, spec.beta), ch_beta[xi, yi])
            priority[xi, yi] = torch.where(mask, torch.full_like(priority[xi, yi], cell_priority), priority[xi, yi])

        return ch_val, ch_type, ch_alpha, ch_beta


__all__ = [
    "BC_TYPE_DIRICHLET",
    "BC_TYPE_INTERIOR",
    "BC_TYPE_NEUMANN",
    "BC_TYPE_ROBIN",
    "CONDITIONAL_INPUT_CHANNELS",
    "ConditionalGrid2D",
]
