# Full Codebase Exploration

---

## 1. High-Level Overview (Top-Down)

### Purpose and Goal

A generalized 2-D scalar PDE solver. The core idea is a **conditional** approach: a single model trained once on thousands of diverse PDEs can instantly solve any new PDE from that distribution in one forward pass, conditioned on the PDE operator and boundary conditions encoded as a 7-channel spatial field. Two fallback modes are provided when the primary solver can't handle the query.

---

### Architecture Layers

```
┌──────────────────────────────────────────────────────┐
│  app.py  ←  launcher.py                              │  UI / Entry
├──────────────────────────────────────────────────────┤
│  inference_engine.py                                 │  Inference orchestration
├──────────────────────────────────────────────────────┤
│  models/  (FNO + PINN + checkpoint + inputs)        │  Neural model stack
├──────────────────────────────────────────────────────┤
│  trainer.py             evaluate.py                  │  Offline pipeline
├──────────────────────────────────────────────────────┤
│  physics/pde_helpers.py                              │  PDE residuals + BCs
├──────────────────────────────────────────────────────┤
│  pde_parser.py    pde_space.py    ood_detector.py   │  PDE representation
└──────────────────────────────────────────────────────┘
```

---

### Module Dependency Graph

```
app.py
  └─ inference_engine
       ├─ pde_parser
       ├─ models/
       │    ├─ fno_model  ──→  fno_layers
       │    ├─ conditional_inputs  ──→  pde_parser
       │    ├─ conditional_solvers ──→  fno_model, physics/pde_helpers
       │    └─ checkpoints
       ├─ ood_detector  ──→  pde_parser (featurize)
       └─ physics/pde_helpers  ──→  pde_parser

trainer.py
  ├─ pde_space       (LHS problem sampling)
  ├─ pde_parser
  ├─ ood_detector   (manifest building)
  ├─ models/conditional_inputs
  ├─ models/fno_model
  ├─ models/conditional_solvers
  └─ physics/pde_helpers

evaluate.py
  └─ inference_engine   (reuses full solve pipeline)
```

`pde_parser`, `pde_space`, and `ood_detector` are leaf modules with no intra-project dependencies.

---

### Three Solver Backends

| Backend | When Used | Speed | Cost |
|---|---|---|---|
| **FNO offline** | `solver_type="fno"`, model file exists, in-distribution | ~ms (1 fwd pass) | pretrained |
| **FNO online** | `solver_type="fno"`, model file missing | ~minutes (per-problem training) | none |
| **PINN offline** | `solver_type="pinn"`, model file exists, in-distribution | ~ms (1 fwd pass) | pretrained |
| **PINN online** | `solver_type="pinn"`, model file missing | ~minutes (per-problem training) | none |
| **FD** | OOD fallback for FNO or PINN, or `solver_type="fd"` | ~seconds (Jacobi) | none |

Both FNO and PINN operate in **offline mode** (single forward pass) when a pretrained weight file is found on disk. Both fall back to **online per-problem training** when the file is absent. Both route to **FD** when a manifest is loaded and the query is out-of-distribution.

---

### Key Conditional Input: 7-Channel Grid

Both FNO and PINN receive the same `(1, n, n, 7)` tensor describing the problem:

| Channel | Content |
|---|---|
| 0 | x-coordinates |
| 1 | y-coordinates |
| 2 | source term $f(x,y)$ |
| 3 | BC value per wall |
| 4 | BC flux per wall |
| 5 | Robin $\alpha$ per wall |
| 6 | Robin $\beta$ per wall |

---

### Three-Stage OOD Detection (FNO and PINN)

Both FNO and PINN support optional OOD gating via a manifest file (`FNOConfig.manifest_path` / `PINNConfig.manifest_path`). When a manifest is loaded, `OODDetector` runs three checks in order before the offline forward pass:
1. **Hard rules** — time-dependent or hyperbolic PDEs are always OOD
2. **Bounding box** — 25-dim normalized feature vector outside `[-0.25, 1.25]`
3. **KNN distance** — min L2 distance to training manifold exceeds a trained threshold

Any fail → FD fallback, `result.is_ood = True`. OOD checks only apply when a model file exists (the online paths bypass OOD gating entirely).

---

### Offline Training Pipeline

```
pde_space (LHS) → N problem dicts
    ↓
SharedFDDataGenerator.generate()
    parse_pde + parse_bc
    ConditionalGrid2D  (7-ch input)
    Jacobi FD solve  (64×64 target)
    PDEFeaturizer  (25-dim OOD feature)
    → .npz dataset  +  manifest.npz
    ↓
HybridFNOTrainer.train()           (or PINNTrainer for PINN)
    hybrid loss = λ_data·MSE(pred, fd_target)
                + λ_phys·PDE_residual
                + λ_bc·BC_error
    cosine-annealing Adam  +  grad clip
    → fno.pt / pinn.pt  +  *_best.pt
```

---

## 2. Detailed Exploration (Bottom-Up)

---

### `pde_parser.py`

**In:** PDE string e.g. `"u_xx + u_yy = sin(pi*x)"`, BC dict, IC string  
**Out:** `ParsedPDE`, `ParsedBC`, `ParsedIC`, `build_fd_residual()` closure

`parse_pde()` uses sympy to extract scalar coefficients from the standard form:

$$g \cdot u_t + a \cdot u_{xx} + b \cdot u_{yy} + c \cdot u_{xy} + d \cdot u_x + e \cdot u_y + f \cdot u = \text{rhs}(x,y,t)$$

It classifies the PDE (elliptic $D < 0$, parabolic $D = 0$, hyperbolic $D > 0$) via $D = c^2 - 4ab$, and stores a sympy-lambdified `rhs_fn()` callable.

`ParsedBC` wraps one wall's type (`dirichlet`/`neumann`/`robin`), value expression, and Robin coefficients $(\alpha, \beta)$ — where $\alpha u + \beta \partial u / \partial n = \text{value}$.

`build_fd_residual()` returns a function `residual(u, tt) → Tensor` using 2nd-order central differences for the interior points.

---

### pde_space.py

**In:** `PDESpaceConfig` (parameter ranges), `n_samples`, `seed`  
**Out:** list of `{"pde_str": ..., "bc_dict": ...}` dicts

`LHSSampler` uses Latin Hypercube Sampling across the PDE parameter space. The space covers:
- $a, b \in [0.5, 2.0]$ (u_xx, u_yy coefficients)
- Optional Helmholtz term $f \in [-1, 0]$ (20% prob)
- Source terms from 8 templates (sinusoids, Gaussians, polynomials)
- BC types: Dirichlet (75%), Neumann (15%), Robin (10%)
- BC values: zero, sinusoid, cosine, constant, polynomial

LHS guarantees uniform coverage without clustering.

---

### ood_detector.py

**In (featurize):** `ParsedPDE`, `bc_specs`  
**Out:** `np.ndarray` shape `(25,)` — `[g, a, b, c, d, e, f, rhs_rms, rhs_max, (type_enc, val_l2, α, β) × 4 walls]`

**In (check):** `ParsedPDE`, `bc_specs` + loaded `manifest.npz`  
**Out:** `(bool, str)` — `(is_ood, reason)`

`build_manifest()` (called by trainer) normalizes and stores the full training feature matrix, plus a KNN-derived distance threshold (95th-percentile of leave-one-out nearest-neighbor distances).

---

### `physics/pde_helpers.py`

**In:** `ParsedPDE`, `bc_specs`, `u` tensor `(n, n)`  
**Out:** scalar loss tensors

`GeneralPDE.compute_pde_loss()` calls `build_fd_residual` under the hood and returns the mean-squared interior residual.

`compute_bc_loss()` enforces $\alpha u + \beta \partial u / \partial n = \text{value}$ on all four walls using 2nd-order one-sided stencils (forward for left/bottom, backward for right/top).

`compute_ic_loss()` applies only for time-dependent PDEs at $t=0$.

`LaplacePDE` and `NavierStokesPDE` are legacy specialized implementations (not used by the main pipeline).

---

### `models/conditional_inputs.py` — `ConditionalGrid2D`

**In:** `n_points`, `bc_specs`, `source_fn`, `device`, `t_value=0`  
**Out:** `self.input_grid` — shape `(1, n, n, 7)` float32 tensor

1. Builds meshgrid `xx, yy` on $[0,1]^2$
2. Evaluates `source_fn(xx, yy, tt)` → channel 2
3. For each of 4 walls, evaluates the BC value function along the wall edge and stamps it into channels 3–6
4. Stacks all 7 channels and adds a batch dimension

This is the **sole input constructor** shared by FNO, PINN offline, and PINN online.

---

### `models/fno_layers.py`

Core building blocks (all channel-first `(B, C, H, W)`):

| Component | Role |
|---|---|
| `SpectralConv` | N-D rFFT, keep first $k$ modes, complex weight multiply, irFFT back |
| `FNOBlock` | `SpectralConv(x) + skip(x)` → GELU → optional norm → `ChannelMLP(x) + mlp_skip(x)` |
| `ChannelMLP` | Two `Conv1d(kernel=1)` layers; spatial-agnostic |
| `SoftGating` | Learned per-channel scalar $w \cdot x$ |
| `DomainPadding` | Symmetric zero-pad before blocks, crop after (reduces boundary aliasing) |
| `GridEmbedding2D` | Appends `(x, y)` coordinate channels to input |

`SpectralConv` stores `weight_real` and `weight_imag` separately. Hermitian symmetry is enforced by zeroing imaginary parts of DC and Nyquist bins before `irfft`.

---

### `models/fno_model.py` — `FNO2DModel`

**In:** tensor `(B, n, n, 7)` channel-last  
**Out:** tensor `(B, n, n, 1)` — solution field $u$

```
input (B, n, n, 7)
  [optional GridEmbedding2D → (B, n, n, 9)]
  Linear lift: 7 → width
  permute → (B, width, n, n)
  [optional DomainPadding.pad]
  × n_layers FNOBlock
  [optional DomainPadding.unpad]
  permute → (B, n, n, width)
  GELU( Linear(width → 128) )
  Linear(128 → 1)
output (B, n, n, 1)
```

---

### `models/conditional_solvers.py` — `_PointwiseConditionalPINNNet`

**In:** `(B, n, n, 7)` channel-last grid tensor  
**Out:** `(B, n, n, 1)` — solution field $u$

```
input (B, n, n, 7)
  reshape → (B×n×n, 7)
  Linear(7 → hidden) + Tanh
  × (n_layers − 1) Linear(hidden → hidden) + Tanh
  Linear(hidden → 1)
  reshape → (B, n, n, 1)
output (B, n, n, 1)
```

This is a pointwise MLP: no spatial convolutions, no receptive field coupling. Each grid point is processed independently from its 7-channel condition vector. The same backbone is used for both offline inference and online training.

`ConditionalFNO2D` and `SharedConditionalPINN2D` are wrapper classes that hold the model, optimizer, and training loop — used only in the **online training fallback** path.

---

### `models/checkpoints.py`

**In:** `nn.Module`, path, device  
**Out:** in-place weight load

Detects checkpoint format by inspecting the ZIP structure:
- Plain state-dict → `model.load_state_dict(ckpt)`
- Metadata envelope `{"arch": ..., "state_dict": ...}` → unwrap
- TorchScript archive (contains `constants.pkl`) → `jit.load()` → extract state_dict

`skip_if_torchscript=True` (used for PINN loading) silently skips TorchScript archives rather than raising — the PINN online fallback will then train from scratch.

---

### `inference_engine.py` — `InferenceEngine`

**Configuration in:** `SolverOption` dataclass (one-time at construction)  
**Solve in:** `pde_str`, `bc_dict`, `ic_str=None`  
**Out:** `SolveResult(u, xx, yy, residual, bc_error, method, history, is_ood, ood_reason)`

#### `__init__` — model loading (offline, done once)

```
if solver_type == "fno":
    instantiate FNO2DModel
    if model_path → load_model_weights or jit.load → .eval()
    if manifest_path → OODDetector(manifest)     # optional OOD gate

elif solver_type == "pinn":
    instantiate _PointwiseConditionalPINNNet
    if model_path → load_model_weights → .eval() # self._pinn_net set
    if manifest_path → OODDetector(manifest)     # optional OOD gate
```

Model loading happens once at construction time for both FNO and PINN. `solve()` then uses the pre-loaded network for every query.

#### `solve()` — routing

Both FNO and PINN follow the identical three-stage decision tree:

```
parse_pde(pde_str) → ParsedPDE
parse_bc(bc_dict)  → bc_specs
GeneralPDE(parsed_pde, bc_specs)

solver_type == "fno":
    model_path missing or file not found?
        └─→ _fno_online_path         (ConditionalFNO2D, per-problem training)
    OODDetector.check()
        └─ OOD? → _fd_path, is_ood=True
    in-distribution:
        └─ JIT model?  → _jit_path
        └─ else        → _fast_path  (FNO2DModel, single fwd pass)

solver_type == "pinn":
    model_path missing or file not found?
        └─→ _pinn_online_path        (ConditionalPINN2D, per-problem training)
    OODDetector.check()
        └─ OOD? → _fd_path, is_ood=True
    in-distribution:
        └─→ _pinn_path               (_PointwiseConditionalPINNNet, single fwd pass)

solver_type == "fd":
    _fd_path: Jacobi relaxation
```

#### Online training paths

`_fno_online_path` — wraps `ConditionalFNO2D` (physics + BC loss, Adam, no FD target). Used when FNO weight file is absent.

`_pinn_online_path` — wraps `ConditionalPINN2D` / `SharedConditionalPINN2D` (Adam warm-up → L-BFGS, physics + BC + IC loss). Used when PINN weight file is absent.

Both online paths skip OOD checking and return `SolveResult` with a populated `history` list.

---

### `trainer.py`

**CLI subcommands and their pipelines:**

| Command | What it does |
|---|---|
| `fno-generate` | LHS sample → FD solve 64×64 → save `.npz` + `manifest.npz` |
| `fno-train` | Load `.npz` → `HybridFNOTrainer` → `fno.pt` + `fno_best.pt` |
| `fno` | `fno-generate` + `fno-train` in one shot |
| `pinn` | LHS sample → `PINNTrainer` → `pinn.pt` + `pinn_best.pt` |
| `fno-test` | `evaluate.py` pipeline on test `.npz` |
| `pinn-test` | Same for PINN |

`HybridFNOTrainer` loss per step:

$$\mathcal{L} = \lambda_\text{data} \cdot \|u_\text{pred} - u_\text{FD}\|^2 + \lambda_\text{phys} \cdot \text{PDE\_residual} + \lambda_\text{BC} \cdot \text{BC\_error}$$

Cosine-annealing LR + gradient norm clipping 1.0. Best checkpoint saved separately.

---

### evaluate.py

**In:** dataset `.npz`, `SolverOption`  
**Out:** `list[EvalMetrics]`, summary dict

Loops over all samples, calls `InferenceEngine.solve()` for each. The engine is constructed once and reused, so PINN weights are loaded once and the fast offline path is used for every sample.

$$\text{rel\_l2} = \frac{\|u_\text{pred} - u_\text{FD}\|_2}{\|u_\text{FD}\|_2} \qquad \text{rmse} = \sqrt{\frac{1}{N}\sum_i(u_\text{pred,i} - u_\text{FD,i})^2}$$

Also records `max_err`, `bc_error`, `pde_residual`, `method`, `is_ood` per sample, then aggregates mean/std/max.

---

### `app.py` / `launcher.py`

`app.py` is the Streamlit front-end. It:
1. Renders sidebar (grid res, solver type, model paths, architecture params, λ weights, presets)
2. Takes PDE string + BC table + optional IC as live input
3. Calls `parse_pde()` in real-time for classification feedback
4. On "Solve": builds `SolverOption` → `InferenceEngine(solver_option)` → `engine.solve()` → stores `SolveResult`
   - FNO / PINN with model file on disk: instant offline inference
   - FNO / PINN without model file: online per-problem training
   - Either, with manifest, OOD query: FD fallback with OOD warning banner
5. Renders: contour plot, 3D surface, loss history (online paths only), OOD warning, CSV/PNG downloads

`launcher.py` wraps `streamlit run src/app.py` as a single callable for PyInstaller packaging.