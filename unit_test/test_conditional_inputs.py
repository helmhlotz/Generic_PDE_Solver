import torch

from models.conditional_inputs import (
    BC_TYPE_DIRICHLET,
    BC_TYPE_INTERIOR,
    BC_TYPE_ROBIN,
    CONDITIONAL_INPUT_CHANNELS,
    ConditionalGrid2D,
)
from pde_parser import parse_bc, parse_pde


def test_conditional_grid_emits_13_channel_layout_and_coefficients():
    parsed = parse_pde("1.5*u_xx + 2.0*u_yy + 0.25*u_x + -0.5*u_y + 0.75*u = sin(pi*x)")
    bc_specs = parse_bc(
        {
            "left": {"type": "dirichlet", "value": "1"},
            "right": {"type": "dirichlet", "value": "2"},
            "bottom": {"type": "dirichlet", "value": "3"},
            "top": {"type": "dirichlet", "value": "4"},
        }
    )

    grid = ConditionalGrid2D(5, parsed, bc_specs, torch.device("cpu"))
    input_grid = grid.input_grid.squeeze(0)

    assert input_grid.shape == (5, 5, CONDITIONAL_INPUT_CHANNELS)
    assert torch.allclose(input_grid[..., 2], torch.full((5, 5), 1.5))
    assert torch.allclose(input_grid[..., 3], torch.full((5, 5), 2.0))
    assert torch.allclose(input_grid[..., 4], torch.full((5, 5), 0.25))
    assert torch.allclose(input_grid[..., 5], torch.full((5, 5), -0.5))
    assert torch.allclose(input_grid[..., 6], torch.full((5, 5), 0.75))
    assert torch.allclose(input_grid[..., 7], torch.zeros((5, 5)))
    assert torch.isclose(input_grid[2, 2, 8], torch.tensor(1.0))
    assert torch.isclose(input_grid[0, 2, 9], torch.tensor(1.0))
    assert torch.isclose(input_grid[2, 2, 10], torch.tensor(BC_TYPE_INTERIOR))
    assert torch.isclose(input_grid[0, 2, 10], torch.tensor(BC_TYPE_DIRICHLET))


def test_conditional_grid_corner_uses_bc_type_priority_then_wall_priority():
    parsed = parse_pde("u_xx + u_yy = 0")
    bc_specs = parse_bc(
        {
            "left": {"type": "dirichlet", "value": "1"},
            "top": {"type": "robin", "value": "5", "alpha": "2.0", "beta": "3.0"},
            "right": {"type": "dirichlet", "value": "0"},
            "bottom": {"type": "dirichlet", "value": "0"},
        }
    )

    grid = ConditionalGrid2D(5, parsed, bc_specs, torch.device("cpu"))
    input_grid = grid.input_grid.squeeze(0)

    assert torch.isclose(input_grid[0, 4, 9], torch.tensor(5.0))
    assert torch.isclose(input_grid[0, 4, 10], torch.tensor(BC_TYPE_ROBIN))
    assert torch.isclose(input_grid[0, 4, 11], torch.tensor(2.0))
    assert torch.isclose(input_grid[0, 4, 12], torch.tensor(3.0))

    same_type_bc = parse_bc(
        {
            "left": {"type": "dirichlet", "value": "7"},
            "top": {"type": "dirichlet", "value": "9"},
            "right": {"type": "dirichlet", "value": "0"},
            "bottom": {"type": "dirichlet", "value": "0"},
        }
    )
    same_type_grid = ConditionalGrid2D(5, parsed, same_type_bc, torch.device("cpu")).input_grid.squeeze(0)

    assert torch.isclose(same_type_grid[0, 4, 9], torch.tensor(7.0))
    assert torch.isclose(same_type_grid[0, 4, 10], torch.tensor(BC_TYPE_DIRICHLET))
