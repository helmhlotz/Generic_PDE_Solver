Created 6 todos

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src)

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/pyproject.toml#1-1), lines 1 to 50

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src/models)

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src/physics)

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src/trainer.py#1-1), lines 1 to 200

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src/models/fno_model.py)

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src/dataset.py)

Starting: *Read core model architecture* (2/6)

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src/trainer.py#200-200), lines 200 to 500

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src/training.py)

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src/models/conditional_inputs.py)

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src/models/conditional_solvers.py#1-1), lines 1 to 200

Starting: *Read FNO training logic* (3/6)

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src/trainer.py#500-500), lines 500 to 500

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src/models/conditional_solvers.py#200-200), lines 200 to 300

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src/physics/pde_helpers.py#1-1), lines 1 to 200

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src/trainer.py#500-500), lines 500 to 700

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src/pde_parser.py#1-1), lines 1 to 200

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src/models/fno_layers.py#1-1), lines 1 to 200

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src/trainer.py#700-700), lines 700 to 900

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src/physics/pde_helpers.py#200-200), lines 200 to 278

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src/trainer.py#900-900), lines 900 to 1100

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src/pde_space.py#1-1), lines 1 to 200

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src/trainer.py#1100-1100), lines 1100 to 1300

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src/pde_parser.py#100-100), lines 100 to 200

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src/trainer.py#1300-1300), lines 1300 to 1500

Searched for text `build_fd_residual` (`**/src/pde_parser.py`), 1 result

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src/pde_parser.py#420-420), lines 420 to 479

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src/models/fno_layers.py#200-200), lines 200 to 200

Starting: *Identify issues and propose fixes* (6/6)

Now let me also check the `evaluate.py` and `inference_engine.py` for completeness: 

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src/inference_engine.py#1-1), lines 1 to 100

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src/evaluate.py#1-1), lines 1 to 100

Read [](file:///Users/jie/Documents/VSCode/ML/FNO/src/inference_engine.py#100-100), lines 100 to 300

Completed: *Identify issues and propose fixes* (6/6)

After thoroughly reading the codebase, here is my analysis of the issues in both FNO and PINN training, along with proposed fixes.

---

## Issue Analysis

### 🔴 CRITICAL Issues

#### 1. FNO is trained with physics+BC loss that overwhelms data loss

**Location**: training.py → `hybrid_loss()`, trainer.py → `FNOTrainer._loss_cfg()`

FNO is an **operator-learning** model — it learns the mapping from conditional inputs to solutions purely from data supervision. The default loss weights are `lam_data=1.0, lam_phys=1.0, lam_bc=10.0`, meaning the BC loss dominates by 10×. This is problematic because:

- The FD targets already satisfy the PDE and BCs, so physics/BC losses are **redundant** when data loss is well-minimized
- The BC loss gradient can overwhelm the data loss, causing the model to focus on boundary satisfaction at the expense of interior accuracy
- Computing FD residuals on the FNO output adds gradient noise and computational overhead with no benefit
- The input channels 3–6 already encode BC information, so the model should learn BCs from the data

**Fix**: For FNO, use **pure data loss** (or very small physics/BC weights for mild regularization):
```python
# FNOTrainer._loss_cfg should return:
{"lam_data": 1.0, "lam_phys": 0.0, "lam_bc": 0.0}
# Or for mild regularization:
{"lam_data": 1.0, "lam_phys": 0.01, "lam_bc": 0.01}
```

---

#### 2. Batch size forced to 1 — devastating for FNO training efficiency

**Location**: dataset.py → `collate_operator_batch()`, trainer.py → `_make_operator_loader()`

The collate function explicitly requires `batch_size=1`:
```python
if len(batch) != 1:
    raise ValueError("collate_operator_batch currently requires batch_size=1")
```

This exists because `pde_obj` is kept as a scalar for physics/BC loss computation. With batch_size=1:
- **GPU utilization is extremely poor** (a single 64×64 sample is tiny)
- **Gradient estimates are very noisy** (single-sample stochastic gradient)
- **Training is unnecessarily slow**

**Fix**: When FNO uses data-only loss, `pde_obj` is unnecessary during training. Create a separate collate function for FNO that supports batch_size > 1:
```python
def collate_fno_batch(batch):
    inputs = torch.stack([s["input"] for s in batch])
    targets = torch.stack([s["target"] for s in batch]) if batch[0]["target"] is not None else None
    return {
        "input": inputs,
        "target": targets,
        "has_target": batch[0]["has_target"],
        # No pde_obj, x_1d, y_1d needed for data-only loss
    }
```
Then use `batch_size=16` or `32` for FNO training.

---

#### 3. PINN `steps_per_problem` is explicitly ignored

**Location**: trainer.py → `PINNTrainer.train()`

```python
if steps_per_problem != 1:
    log.warning(
        "steps_per_problem=%d is ignored for dataset-driven PINN training; "
        "using one pass per example.",
        steps_per_problem,
    )
```

PINNs need **many gradient steps on the same problem** to converge to a solution that satisfies the PDE and BCs. With only 1 step per problem per epoch, the PINN:
- Never converges on any individual problem
- Gets pulled in conflicting directions by different problems each step
- Essentially performs random walks in parameter space

The `RepeatDataset` wrapper (used in the base class) doesn't solve this — it just repeats the same sample in the epoch queue, but the optimizer state is shared across all problems, so repeated visits don't give the same convergence behavior as dedicated inner-loop steps.

**Fix**: Implement a proper inner loop for PINN training:
```python
for epoch in range(n_epochs):
    for problem in dataset:
        for _ in range(steps_per_problem):  # inner loop
            optimizer.zero_grad()
            u = model(problem["input"].unsqueeze(0)).squeeze(0).squeeze(-1)
            loss = physics_loss(u) + bc_loss(u)
            loss.backward()
            optimizer.step()
```

---

#### 4. PINN uses finite-difference PDE loss instead of automatic differentiation

**Location**: pde_helpers.py → `GeneralPDE.compute_pde_loss()`, pde_parser.py → `build_fd_residual()`

The PINN computes PDE residuals using **finite-difference stencils** on the output grid:
```python
d2u_dx2 = (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / h**2
```

This is **fundamentally wrong for a PINN**. The core innovation of PINNs is using **automatic differentiation** to compute exact derivatives $\frac{\partial u}{\partial x}$, $\frac{\partial^2 u}{\partial x^2}$, etc. by differentiating through the network. FD-based residuals:

- Introduce $O(h^2)$ truncation error that depends on grid resolution
- Defeat the purpose of using a neural network (which can represent smooth functions with exact derivatives)
- Make the PINN no better than a standard neural network with FD regularization

The pointwise MLP (`_PointwiseConditionalPINNNet`) processes each grid point independently, so autodiff can compute exact spatial derivatives by treating (x, y) as differentiable inputs.

**Fix**: Implement autodiff-based PDE loss for PINN. The forward pass needs to be restructured so that (x, y) coordinates are differentiable inputs:
```python
def compute_pde_loss_autodiff(self, model, x, y, t=None):
    x.requires_grad_(True)
    y.requires_grad_(True)
    u = model(x, y, ...)  # forward pass with x,y as inputs
    u_x = torch.autograd.grad(u, x, ...)
    u_xx = torch.autograd.grad(u_x, x, ...)
    # etc.
    residual = self.a * u_xx + self.b * u_yy + ... - rhs
    return torch.mean(residual ** 2)
```

---

### 🟠 IMPORTANT Issues

#### 5. No input/target normalization

**Location**: dataset.py → `PDEOperatorDataset`, conditional_inputs.py → `ConditionalGrid2D`

The 7 input channels have vastly different scales:
| Channel | Content | Typical Range |
|---------|---------|---------------|
| 0 | x | [0, 1] |
| 1 | y | [0, 1] |
| 2 | source | [-3, 3] |
| 3 | BC value | [0, 2] |
| 4 | BC flux | [0, 2] or 0 |
| 5 | alpha | 0 or 1 |
| 6 | beta | 0 or 1 |

The target solution can have arbitrary magnitude. Without normalization:
- The source channel (range 6×) dominates the feature space
- Spectral convolutions in FNO operate on features with mismatched scales
- Loss is dominated by large-magnitude solutions
- Training is unstable and convergence is slow

**Fix**: Add per-channel standardization (zero mean, unit variance) computed from the training dataset statistics, and target normalization (e.g., divide by `max|u_fd|` per sample or dataset-wide std).

---

#### 6. Validation loss uses hybrid loss instead of task-specific metric

**Location**: training.py → `evaluate_operator()`

The validation loss uses the same `hybrid_loss` as training. For FNO, the best checkpoint should be selected based on **prediction accuracy** (data loss), not a mix of data+physics+BC loss. A model with low BC loss but poor interior accuracy would be selected over a model with better overall prediction quality.

**Fix**: Use task-specific validation metrics:
- **FNO**: Pure MSE against FD targets
- **PINN**: Physics+BC loss (since PINN should satisfy PDE constraints)

---

#### 7. PINN architecture is too small for shared-weights conditional learning

**Location**: conditional_solvers.py → `_PointwiseConditionalPINNNet`

Default: `hidden=64, n_layers=4` → only ~17K parameters. Standard PINNs for **single** problems use 8 layers × 200 units (~160K params). A shared conditional PINN across many PDE problems needs even more capacity.

**Fix**: Increase defaults to `hidden=128, n_layers=6` or larger, and add Fourier feature embeddings or SIREN-style activations for better high-frequency learning.

---

#### 8. No L-BFGS fine-tuning for dataset-driven PINN training

**Location**: trainer.py → `PINNTrainer`

The single-problem `ConditionalPINN2D` supports L-BFGS refinement, but `PINNTrainer` uses only Adam. L-BFGS is critical for PINNs to achieve low residuals.

**Fix**: Add an optional L-BFGS refinement phase after Adam pre-training, at least for the final epochs or for fine-tuning on specific problems.

---

### 🟡 MINOR Issues

#### 9. `PDEOperatorDataset` stores `pde_obj` per sample — memory waste

Each `GeneralPDE` object holds parsed PDE data and a residual function. For 5000+ samples, this is significant. For FNO with data-only loss, `pde_obj` is never used during training.

**Fix**: Store only PDE coefficients and BC specs; reconstruct `pde_obj` on-demand (or skip entirely for FNO).

---

#### 10. No gradient accumulation

With batch_size=1, gradient estimates are noisy. Gradient accumulation (e.g., accumulate over 16 steps before updating) would simulate batch_size=16 without changing the data pipeline.

**Fix**: Add `gradient_accumulation_steps` parameter to `train_operator()`.

---

#### 11. FNO model `squeeze` operations are fragile

**Location**: training.py → `train_operator()`

```python
pred = model(batch["input"]).squeeze(0).squeeze(-1)
target = batch["target"].squeeze(0) if batch["target"] is not None else None
```

These `.squeeze(0)` calls assume batch_size=1 and will silently produce wrong shapes if batch_size > 1.

**Fix**: Use explicit dimension selection: `pred[:, :, :, 0]` instead of `.squeeze(-1)`.

---

## Summary Table

| # | Severity | Component | Issue | Impact |
|---|----------|-----------|-------|--------|
| 1 | 🔴 Critical | FNO Loss | Physics+BC loss dominates data loss | FNO learns BCs at expense of interior |
| 2 | 🔴 Critical | DataLoader | Batch size forced to 1 | Poor GPU use, noisy gradients, slow |
| 3 | 🔴 Critical | PINN Train | steps_per_problem ignored | PINN never converges per problem |
| 4 | 🔴 Critical | PINN Loss | FD residuals instead of autodiff | PINN loses its key advantage |
| 5 | 🟠 Important | Data | No input/target normalization | Unstable/slow training |
| 6 | 🟠 Important | Validation | Hybrid loss for model selection | Wrong best-model selection |
| 7 | 🟠 Important | PINN Arch | Too small (64×4) | Insufficient capacity |
| 8 | 🟠 Important | PINN Train | No L-BFGS refinement | Can't reach low residuals |
| 9 | 🟡 Minor | Dataset | Stores pde_obj per sample | Memory waste |
| 10 | 🟡 Minor | Training | No gradient accumulation | Noisy gradients at bs=1 |
| 11 | 🟡 Minor | Training | Fragile squeeze ops | Breaks if bs>1 |

