"""Streamlit web UI for generalized PINN/FNO inference.

Launch:
    cd /path/to/FNO
    streamlit run src/app.py
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import streamlit as st

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from pde_parser import parse_pde, parse_bc, parse_ic
from inference_engine import (
    InferenceEngine,
    SolverOption,
    GridConfig,
    FDConfig,
    FNOConfig,
    PINNConfig,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Generalized PDE Solver",
    page_icon="",
    layout="wide",
)

st.title("Generalized PDE Solver")
st.caption(
    "Type any scalar PDE and boundary conditions. "
    "The conditional FNO gives an instant prediction; "
    "run `python src/trainer.py fno` to update the model."
)
# ---------------------------------------------------------------------------

def _derive_manifest_path(model_path: str) -> str | None:
    """Return the OOD manifest path for a given FNO model path, or None if absent.

    Looks for ``<model_dir>/fno_manifest.npz`` first (the default name written
    by ``trainer.py fno-generate``), then tries ``<model_stem>_manifest.npz``.
    """
    p = Path(model_path)
    candidates = [
        p.parent / "fno_manifest.npz",
        p.with_name(p.stem + "_manifest.npz"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None

# ---------------------------------------------------------------------------
_DEFAULT_PDE  = "u_xx + u_yy = 0"
_DEFAULT_BC: dict[str, dict] = {
    "left":   {"type": "dirichlet", "value": "0"},
    "right":  {"type": "dirichlet", "value": "0"},
    "bottom": {"type": "dirichlet", "value": "0"},
    "top":    {"type": "dirichlet", "value": "sin(pi*x)"},
}
_BC_TYPES = ["dirichlet", "neumann", "robin"]
_WALLS = ["left", "right", "bottom", "top"]

# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    n_points = st.slider("Grid resolution (n×n)", min_value=16, max_value=512, value=50, step=1)

    st.divider()
    st.subheader("Solver")

    use_pretrained = st.toggle("Use pretrained model", value=True)
    solver_type = st.radio("Solver type", ["FNO", "PINN", "FD"], horizontal=True)

    if solver_type == "FD":
        fno_path = ""
        pinn_path = ""
        st.warning("FD solver selected: the app will skip FNO/PINN and use finite difference.")
    elif use_pretrained:
        if solver_type == "FNO":
            fno_path = st.text_input(
                "FNO weights path",
                value="pretrained_models/fno.pt",
                help="State-dict (.pt) or TorchScript archive for the conditional FNO.",
            )
            pinn_path = ""
        else:
            pinn_path = st.text_input(
                "PINN weights path",
                value="pretrained_models/pinn.pt",
                help="State-dict (.pt) of the pretrained MLP PINN. "
                     "Used as warm-start before further training.",
            )
            fno_path = ""
    else:
        fno_path = ""
        pinn_path = ""

    # Model-specific architecture configuration
    if solver_type == "FNO":
        with st.expander("FNO configuration", expanded=False):
            fno_width  = st.number_input("Channel width",         value=32, min_value=8,  step=8,  key="fno_width")
            fno_modes  = st.number_input("Fourier modes per dim", value=12, min_value=4,  step=2,  key="fno_modes")
            fno_layers = st.number_input("FNO blocks",            value=4,  min_value=1,  step=1,  key="fno_layers")
        pinn_hidden, pinn_layers_n = 64, 4
    elif solver_type == "PINN":
        with st.expander("PINN configuration", expanded=False):
            pinn_hidden   = st.number_input("Hidden units per layer", value=64, min_value=16, step=16, key="pinn_hidden")
            pinn_layers_n = st.number_input("Number of layers",       value=4,  min_value=2,  step=1,  key="pinn_layers")
        fno_width, fno_modes, fno_layers = 32, 12, 4
    else:
        fno_width, fno_modes, fno_layers = 32, 12, 4
        pinn_hidden, pinn_layers_n = 64, 4

    st.divider()
    st.subheader("Training")

    if solver_type == "FNO":
        pinn_epochs = 3000  # unused when FNO
    elif solver_type == "PINN":
        pinn_epochs = st.number_input(
            "Training epochs",
            min_value=100, max_value=10000, value=3000, step=100,
            help="Number of gradient steps for the MLP PINN.",
        )
    else:
        pinn_epochs = 3000

    st.divider()
    lam_phys = st.number_input("λ physics", value=1.0,  min_value=0.01)
    lam_bc   = st.number_input("λ BC",      value=10.0, min_value=0.01)

# ---------------------------------------------------------------------------
# Main panel — PDE input
# ---------------------------------------------------------------------------
st.subheader("PDE")
st.markdown(
    "**Symbols:** `x`, `y`, `u`, `u_x`, `u_y`, `u_xx`, `u_yy`, `u_xy`, `u_t`, "
    "`pi`, `sin`, `cos`, `exp`, `sqrt`"
)

pde_col, status_col = st.columns([3, 1])
with pde_col:
    pde_str = st.text_input("Equation (LHS = RHS)", value=_DEFAULT_PDE, key="pde_input", label_visibility="collapsed")
with status_col:
    pde_ok = False
    parsed_pde = None
    if pde_str.strip():
        try:
            parsed_pde = parse_pde(pde_str)
            pde_ok = True
            class_label = parsed_pde.pde_class.capitalize()
            if parsed_pde.discriminant is None:
                tag = class_label
            else:
                tag = f"{class_label} (D={parsed_pde.discriminant:.3g})"
            st.success(tag)
        except Exception as exc:
            st.error(f"Parse error: {exc}")

# ---------------------------------------------------------------------------
# Boundary conditions — one per wall
# ---------------------------------------------------------------------------
st.subheader("Boundary Conditions")

bc_dict: dict[str, dict] = {}
cols = st.columns(4)

for col, wall in zip(cols, _WALLS):
    with col:
        st.markdown(f"**{wall.capitalize()} wall**")
        bc_type = st.selectbox(
            "Type", _BC_TYPES,
            index=_BC_TYPES.index(_DEFAULT_BC[wall]["type"]),
            key=f"bc_type_{wall}",
        )
        val_expr = st.text_input(
            "Value",
            value=_DEFAULT_BC[wall].get("value", "0"),
            key=f"bc_val_{wall}",
        )
        spec: dict = {"type": bc_type, "value": val_expr}
        if bc_type == "robin":
            alpha_str = st.text_input("α", value="1.0", key=f"bc_alpha_{wall}")
            beta_str  = st.text_input("β", value="1.0", key=f"bc_beta_{wall}")
            spec["alpha"] = alpha_str
            spec["beta"]  = beta_str
        bc_dict[wall] = spec

# Validate BCs
bc_ok = False
bc_specs = None
try:
    bc_specs = parse_bc(bc_dict)
    bc_ok = True
except Exception as exc:
    st.warning(f"BC parse error: {exc}")

# Initial Condition (for time-dependent PDEs)
ic_str = ""
ic_ok = False
ic_specs = None
if pde_ok and parsed_pde.is_time_dependent:
    st.subheader("Initial Condition")
    if parsed_pde.has_time_derivative:
        ic_help = "Initial condition u(x,y,t=0). Use symbols: x, y, sin, cos, exp, sqrt, pi, etc."
    else:
        ic_help = "This PDE depends on time through t. Provide an initial state u(x,y,t=0)."
    ic_str = st.text_input(
        "u(x,y,0)",
        value="0",
        key="ic_input",
        label_visibility="collapsed",
        help=ic_help,
    )
    try:
        ic_specs = parse_ic(ic_str)
        ic_ok = True
    except Exception as exc:
        st.warning(f"IC parse error: {exc}")
else:
    ic_ok = True  # No IC needed for steady-state PDEs

# ---------------------------------------------------------------------------
# Solve button
# ---------------------------------------------------------------------------
st.divider()
solve_disabled = not (pde_ok and bc_ok and ic_ok)
if st.button("Solve", type="primary", disabled=solve_disabled, use_container_width=True):
    with st.spinner("Solving..."):
        try:
            if solver_type == "FNO":
                option = SolverOption(
                    solver_type="fno",
                    device="cpu",
                    grid=GridConfig(n_points=int(n_points)),
                    fno=FNOConfig(
                        model_path=fno_path if (use_pretrained and fno_path) else None,
                        manifest_path=_derive_manifest_path(fno_path) if (use_pretrained and fno_path) else None,
                        width=int(fno_width),
                        n_modes=(int(fno_modes), int(fno_modes)),
                        n_layers=int(fno_layers),
                        lambda_physics=float(lam_phys),
                        lambda_bc=float(lam_bc),
                    ),
                )
                with st.expander("Solver config", expanded=False):
                    st.write(f"solver type: {solver_type}  |  grid n_points: {n_points}")
                try:
                    engine = InferenceEngine(solver_option=option)
                except (FileNotFoundError, RuntimeError) as model_err:
                    st.error(f"**FNO Model Error:** {model_err}")
                    st.info("**Suggestions:**\n"
                            "1. Check the model file path is correct\n"
                            "2. Verify the file is a valid PyTorch model\n"
                            "3. Uncheck 'Use pretrained model' to train from scratch")
                    st.session_state.pop("result", None)
                    st.stop()
            elif solver_type == "PINN":
                option = SolverOption(
                    solver_type="pinn",
                    device="cpu",
                    grid=GridConfig(n_points=int(n_points)),
                    pinn=PINNConfig(
                        model_path=pinn_path if (use_pretrained and pinn_path) else None,
                        hidden=int(pinn_hidden),
                        n_layers=int(pinn_layers_n),
                        epochs=int(pinn_epochs),
                        lambda_physics=float(lam_phys),
                        lambda_bc=float(lam_bc),
                    ),
                )
                with st.expander("Solver config", expanded=False):
                    st.write(f"solver type: {solver_type}  |  grid n_points: {n_points}")
                try:
                    engine = InferenceEngine(solver_option=option)
                except (FileNotFoundError, RuntimeError) as model_err:
                    st.error(f"**PINN Model Error:** {model_err}")
                    st.info("**Suggestions:**\n"
                            "1. Check the model file path is correct\n"
                            "2. Verify the file is a valid PyTorch model\n"
                            "3. Uncheck 'Use pretrained model' to train from scratch")
                    st.session_state.pop("result", None)
                    st.stop()
            else:
                st.warning("Using FD solver only (FNO/PINN disabled by Solver type).")
                option = SolverOption(
                    solver_type="fd",
                    device="cpu",
                    grid=GridConfig(n_points=int(n_points)),
                    fd=FDConfig(print_every=500),
                )
                with st.expander("Solver config", expanded=False):
                    st.write(f"solver type: {solver_type}  |  grid n_points: {n_points}")
                engine = InferenceEngine(solver_option=option)

            _t0 = time.perf_counter()
            result = engine.solve(
                pde_str=pde_str,
                bc_dict=bc_dict,
                ic_str=ic_str if parsed_pde.is_time_dependent else None,
            )
            _elapsed = time.perf_counter() - _t0
            with st.expander("Debug info", expanded=False):
                st.write(f"pde: {pde_str}")
                st.write(f"bc: {bc_dict}")
                st.write(f"ic: {ic_str}")
            st.session_state["result"] = result
            st.session_state["solve_time"] = _elapsed
        except FileNotFoundError as fnf_err:
            st.error(f"**File Not Found:** {fnf_err}")
            st.session_state.pop("result", None)
        except RuntimeError as runtime_err:
            # Capture model loading and other runtime errors
            error_msg = str(runtime_err)
            if "PytorchStreamReader" in error_msg or "constants.pkl" in error_msg:
                st.error(f"**Model File Corrupted:** The model file appears to be corrupted or incomplete.\n\n{error_msg}")
                st.info("Try:\n1. Re-download the model file\n2. Check the file integrity\n3. Use a different model checkpoint")
            else:
                st.error(f"**Solver Error:** {error_msg}")
            st.session_state.pop("result", None)
        except Exception as exc:
            st.error(f"**Unexpected Error:** {exc}")
            st.session_state.pop("result", None)

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------
if "result" in st.session_state:
    result = st.session_state["result"]
    u  = result.u
    xx = result.xx
    yy = result.yy

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Method", result.method.upper())
    m2.metric("Physics residual", f"{result.residual:.3e}")
    m3.metric("BC error", f"{result.bc_error:.3e}")
    m4.metric("Solve time", f"{st.session_state.get('solve_time', 0.0):.2f}s")

    # OOD warning
    if result.is_ood:
        st.warning(
            f"**Out-of-distribution query** — FD fallback was used.\n\n"
            f"**Reason:** {result.ood_reason}\n\n"
            "To expand coverage, retrain the FNO with more diverse samples:\n"
            "```\npython src/trainer.py fno --samples 2000 --epochs 30\n```"
        )

    if result.method == "torchscript":
        st.info(
            "**TorchScript model** — BCs and grid resolution are fixed from training "
            "(the BC settings above were not applied to this run). "
            "Run `python src/trainer.py fno` to generate `fno.pt` for fully "
            "generalised inference.",
            # icon="i",
        )

    # Plots
    fig_cols = st.columns(2 if not result.history else 3)

    # Solution contour
    with fig_cols[0]:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        cf = ax.contourf(xx, yy, u, levels=30, cmap="jet")
        fig.colorbar(cf, ax=ax)
        ax.set_title("Solution u(x,y)")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        st.pyplot(fig)
        plt.close(fig)

    # 3-D surface
    with fig_cols[1]:
        fig = plt.figure(figsize=(4, 3.5))
        ax3 = fig.add_subplot(111, projection="3d")
        ax3.plot_surface(xx, yy, u, cmap="viridis", alpha=0.9)
        ax3.set_title("Surface u(x,y)")
        ax3.set_xlabel("x"); ax3.set_ylabel("y")
        st.pyplot(fig)
        plt.close(fig)

    # Loss history
    if result.history and len(fig_cols) > 2:
        with fig_cols[2]:
            fig, ax = plt.subplots(figsize=(4, 3.5))
            ax.semilogy(result.history, linewidth=1)
            ax.set_title("Loss history")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
            st.pyplot(fig)
            plt.close(fig)

    st.divider()

    # Downloads
    dl1, dl2 = st.columns(2)

    with dl1:
        # CSV download
        flat_data = np.column_stack([xx.ravel(), yy.ravel(), u.ravel()])
        csv_buf = io.StringIO()
        np.savetxt(csv_buf, flat_data, delimiter=",", header="x,y,u", comments="")
        st.download_button(
            "⬇ Download u(x,y) as CSV",
            data=csv_buf.getvalue(),
            file_name="solution.csv",
            mime="text/csv",
        )

    with dl2:
        # PNG download
        fig, ax = plt.subplots(figsize=(5, 4))
        cf = ax.contourf(xx, yy, u, levels=40, cmap="jet")
        fig.colorbar(cf, ax=ax)
        ax.set_title(f"Solution — {pde_str}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        png_buf = io.BytesIO()
        fig.savefig(png_buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        st.download_button(
            "⬇ Download plot as PNG",
            data=png_buf.getvalue(),
            file_name="solution.png",
            mime="image/png",
        )

# ---------------------------------------------------------------------------
# Example presets (sidebar bottom)
# ---------------------------------------------------------------------------
def apply_preset(preset: dict) -> None:
    """Apply preset values to session state before rerun."""
    st.session_state["pde_input"] = preset["pde"]
    for wall, spec in preset["bc"].items():
        st.session_state[f"bc_type_{wall}"] = spec["type"]
        st.session_state[f"bc_val_{wall}"]  = spec.get("value", "0")
        # Clear robin keys from previous presets
        st.session_state.pop(f"bc_alpha_{wall}", None)
        st.session_state.pop(f"bc_beta_{wall}", None)


with st.sidebar:
    st.divider()
    st.subheader("Example presets")

    presets = {
        "Laplace (Dirichlet)": {
            "pde": "u_xx + u_yy = 0",
            "bc": {
                "left":   {"type": "dirichlet", "value": "0"},
                "right":  {"type": "dirichlet", "value": "0"},
                "bottom": {"type": "dirichlet", "value": "0"},
                "top":    {"type": "dirichlet", "value": "sin(pi*x)"},
            },
        },
        "Poisson with source": {
            "pde": "u_xx + u_yy = -2*sin(pi*x)*sin(pi*y)",
            "bc": {
                "left":   {"type": "dirichlet", "value": "0"},
                "right":  {"type": "dirichlet", "value": "0"},
                "bottom": {"type": "dirichlet", "value": "0"},
                "top":    {"type": "dirichlet", "value": "0"},
            },
        },
        "Helmholtz k²=4": {
            "pde": "u_xx + u_yy + 4*u = 0",
            "bc": {
                "left":   {"type": "dirichlet", "value": "0"},
                "right":  {"type": "dirichlet", "value": "0"},
                "bottom": {"type": "dirichlet", "value": "0"},
                "top":    {"type": "dirichlet", "value": "sin(pi*x)"},
            },
        },
        "Neumann on left wall": {
            "pde": "u_xx + u_yy = 0",
            "bc": {
                "left":   {"type": "neumann",   "value": "0"},
                "right":  {"type": "dirichlet", "value": "1"},
                "bottom": {"type": "dirichlet", "value": "0"},
                "top":    {"type": "dirichlet", "value": "0"},
            },
        },
        "Maxwell TE (driven cavity)": {
            "pde": "u_xx + u_yy + 20*u = exp(-50*((x-0.5)**2 + (y-0.5)**2))",
            "bc": {
                "left":   {"type": "dirichlet", "value": "0"},
                "right":  {"type": "dirichlet", "value": "0"},
                "bottom": {"type": "dirichlet", "value": "0"},
                "top":    {"type": "dirichlet", "value": "0"},
            },
        },
    }

    for label, preset in presets.items():
        st.button(label, use_container_width=True, on_click=apply_preset, args=(preset,))
