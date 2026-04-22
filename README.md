# PDE Solver — ML Orchestration & Architecture

A production-oriented system for solving 2-D scalar PDEs using neural operators.
The key design concern is **orchestration**: how offline training pipelines, runtime inference routing, OOD detection, and a live UI fit together as a coherent system — not the specifics of any single model.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        OFFLINE PIPELINE                         │
│                                                                 │
│  PDESpaceConfig + LHSSampler                                    │
│       │  samples N (pde_str, bc_dict) pairs via LHS             │
│       ▼                                                         │
│  SharedFDDataGenerator                                          │
│       │  parallel FD solves via ProcessPoolExecutor             │
│       │  chunk-and-save (fault-tolerant, resumable)             │
│       ▼                                                         │
│  train_data   val_data                          │
│  [inputs(N,64,64,7)   targets(N,64,64)   feats(N,25)]           │
│       │                                                         │
│       ├──► FNOTrainer ──► fno.pt + fno_best.pt                  │
│       │      hybrid loss: data + physics + BC                   │
│       │                                                         │
│       ├──► DeepONetTrainer ─► deeponet.pt + deeponet_best.pt   │
│       │      fixed-resolution operator learning (data + optional BC) │
│       │      mirrors FNO loop (cosine LR + epoch validation)    │
│       │                                                         │
│       │   Both trainers share _DatasetTrainerBase               │
│       │   (PDEOperatorDataset + DataLoader, auto 80/20 split)   │
│       │                                                         │
│       └──► OODDetector.build_manifest ──► manifest.npz          │
│              KNN threshold: 95th-percentile LOO distance        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       INFERENCE ENGINE                          │
│                                                                 │
│  InferenceEngine.solve(pde_str, bc_dict)                        │
│       │                                                         │
│       ▼                                                         │
│  pde_parser  ──► ParsedPDE + bc_specs + pde_obj                 │
│       │                                                         │
│       ▼                                                         │
│  Route by solver_type                                           │
│       │                                                         │
│       ├── "fno"                                                 │
│       │    │  model file missing?                               │
│       │    ├── Yes ──► _fno_online_path (ConditionalFNO2D)      │
│       │    └── No  ──► _maybe_ood_fd_fallback()                 │
│       │                   │ in-dist? ──► _fno_path (offline)   │
│       │                   │ OOD?     ──► _fd_path + is_ood=True │
│       │                                                         │
│       ├── "deeponet"                                            │
│       │    │  model file missing / resolution mismatch?         │
│       │    ├── Yes ──► _fd_path                                 │
│       │    └── No  ──► _maybe_ood_fd_fallback()                 │
│       │                   │ in-dist? ──► _deeponet_path         │
│       │                   │ OOD?     ──► _fd_path + is_ood=True │
│       │                                                         │
│       └── "fd"  ──► _fd_path (_FDSolver, Jacobi relaxation)    │
│                                                                 │
│  Returns SolveResult(u, xx, yy, residual, bc_error,             │
│                       method, history, is_ood, ood_reason)      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         STREAMLIT UI                            │
│                                                                 │
│  app.py ──► SolverOption (GridConfig, FNOConfig, DeepONetConfig,│
│              FDConfig)  ──► InferenceEngine ──► SolveResult     │
│  launcher.py: single-binary entry via streamlit CLI             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Repository Layout

```
src/
├── app.py                    Streamlit UI; builds SolverOption from widgets
├── launcher.py               PyInstaller-compatible entry point
├── trainer.py                CLI + API for data generation and training;
│                              _DatasetTrainerBase shared loader abstraction for FNO + DeepONet
├── inference_engine.py       Runtime routing engine; mirrored FNO/DeepONet solve skeleton
├── evaluate.py               Standalone FD-guardrail evaluation script
├── pde_parser.py             Sympy-based PDE and BC string parser
├── pde_space.py              PDESpaceConfig, LHSSampler, BCGenerator
├── ood_detector.py           PDEFeaturizer (25-d) and KNN OOD detector
│
├── models/
│   ├── fno_model.py          FNO2DModel (channel-last, 7-ch input)
│   ├── fno_layers.py         FNOBlock, SpectralConv2d, GridEmbedding2D
│   ├── conditional_inputs.py ConditionalGrid2D — builds 7-channel input tensor
│   ├── conditional_solvers.py ConditionalFNO2D, DeepONet2DModel
│   └── checkpoints.py        load_model_weights, read_checkpoint_arch
│
└── physics/
    └── pde_helpers.py        GeneralPDE: compute_pde_loss, compute_bc_loss,
                               apply_boundary_conditions; Jacobi stencil helpers

pretrained_models/
├── fno.pt                    Final FNO weights
├── fno_best.pt               Best-val-loss FNO checkpoint
├── manifest.npz              Shared OOD manifest (features_norm, threshold)
├── train_data        Training dataset
└── val_data          Validation dataset
```

---

## Key Orchestration Concepts

### 1. Offline data generation — parallel + fault-tolerant

`trainer.py generate` splits the LHS problem list into 200-sample chunks.
Each chunk is solved in parallel via `ProcessPoolExecutor` (one process per CPU core) and immediately written to `<dataset>_chunk_NNNNNN.npz`.
On re-run, existing chunks are loaded and skipped — the pipeline is safe to interrupt and resume.

```
python src/trainer.py generate \
    --samples 5000 \
    --dataset-path pretrained_models/train_data \
    --n-workers 8          # default: cpu_count()
```

### 2. Two-stage training

Data generation and model training are decoupled, enabling independent reuse of each stage.

Both `FNOTrainer` and `DeepONetTrainer` inherit `_DatasetTrainerBase`, which provides a single `_build_loaders_from_datasets()` method: it loads a `PDEOperatorDataset`, wraps it in a `DataLoader`, and optionally performs a deterministic 80/20 train/val split when `--val-dataset` is omitted. Their train/test scaffolding is mirrored through shared `_train_from_dataset()` and `_test_from_dataset()` paths, with solver-specific behavior isolated to small hook methods.

```bash
# Stage 1 — generate once, reuse many times
python src/trainer.py generate --samples 5000 \
    --dataset-path pretrained_models/train_data

# Stage 2a — train FNO (supervised + physics + BC hybrid loss)
python src/trainer.py train --solver fno \
    --train-dataset pretrained_models/train_data \
    --val-dataset   pretrained_models/val_data \
    --epochs 30

# Stage 2b — train fixed-resolution DeepONet on same dataset
python src/trainer.py train --solver deeponet \
    --train-dataset pretrained_models/train_data \
    --epochs 20 --lam-data 1.0

# Stage 3 — build OOD manifest as an explicit step
python src/trainer.py manifest \
    --train-dataset pretrained_models/train_data \
    --val-dataset pretrained_models/val_data \
    --out pretrained_models/manifest.npz

# Stage 4 — evaluate checkpoints
python src/trainer.py test --solver fno \
    --test-dataset pretrained_models/val_data \
    --checkpoint pretrained_models/fno.pt

# One-shot pipeline (generate + train in one command)
python src/trainer.py fno --samples 5000 --epochs 30

# Legacy aliases remain for backward compatibility but print a deprecation warning.
# fno-generate  →  generate
# fno-train     →  train --solver fno
# fno           →  generate + train --solver fno
# fno-test      →  test --solver fno
```

### 3. Inference routing

`InferenceEngine.solve()` is the single entry point regardless of backend. FNO and DeepONet follow the same routing skeleton: model-availability check → shared OOD gate → offline fast path (if available) or FD fallback.

| Condition | Route |
|---|---|
| `solver_type="fno"`, weights exist, query in-distribution | Offline forward pass (`_fno_path`) |
| `solver_type="fno"`, weights exist, OOD detected | Jacobi FD fallback; `result.is_ood=True` |
| `solver_type="fno"`, weights missing | Online FNO training, optional warm-start |
| `solver_type="deeponet"`, weights loaded, query in-distribution | Offline forward pass (`_deeponet_path`) |
| `solver_type="deeponet"`, weights loaded, OOD detected | Jacobi FD fallback; `result.is_ood=True` |
| `solver_type="deeponet"`, weights missing or resolution mismatch | Jacobi FD fallback |
| `solver_type="fd"` | Jacobi FD always |

### 4. OOD detection

`OODDetector` can gate both FNO and DeepONet queries when a manifest is configured. At training time, `PDEFeaturizer.featurize()` converts each `(ParsedPDE, bc_specs)` pair to a 25-dimensional vector encoding PDE coefficients, RHS statistics, and per-wall BC parameters. `OODDetector.build_manifest()` stores the normalised training features and the 95th-percentile leave-one-out KNN distance as the threshold.

At inference time, three checks run in order:
1. **Structural**: time-dependent or hyperbolic PDE → OOD immediately
2. **Bounding-box**: normalised feature outside `[−0.25, 1.25]` per dimension
3. **KNN distance**: nearest-neighbour distance in normalised feature space exceeds threshold

### 5. Unified input encoding

All three solver paths (FNO offline, FNO online, fixed-resolution DeepONet) consume the same 7-channel tensor produced by `ConditionalGrid2D`:

```
channel 0 : x coordinate
channel 1 : y coordinate
channel 2 : source term f(x,y)
channel 3 : BC value field
channel 4 : BC RHS field for derivative-involving BCs
channel 5 : BC alpha (Robin)
channel 6 : BC beta  (Robin)
shape: (1, n, n, 7)  — batch-first, channel-last
```

This unified encoding decouples the PDE specification from model architecture, allows checkpoints to be shared between online and offline paths, and makes the OOD feature vector directly interpretable.

Note: channel 4 is not a numerically estimated normal-flux field. In the current implementation it repeats the BC right-hand side on Neumann/Robin walls together with the `(alpha, beta)` coefficients, so the boundary operator is represented consistently with the pretrained checkpoints.

### 6. Checkpoint formats

`checkpoints.py` handles two transparently:

| Format | When written | How loaded |
|---|---|---|
| Plain state-dict `{layer: tensor}` | `torch.save(model.state_dict(), ...)` | Direct `load_state_dict` |
| Metadata envelope `{"arch": "deeponet", "n_points": 32, "state_dict": {...}}` | `DeepONetTrainer` saves | Unwrapped before `load_state_dict` |
| TorchScript archive (ZIP) | `ConditionalFNO2D.export_torchscript()` | `torch.jit.load`; `skip_if_torchscript=True` available for warm-start bypass |

`read_checkpoint_arch()` inspects the `arch` key without loading weights, allowing routing decisions without a full model instantiation.

---

## Running the UI

```bash
pip install -e .
streamlit run src/app.py
# or via the launcher entry point:
python src/launcher.py
```

The sidebar exposes solver selection, weight paths, architecture hyperparameters, and loss weights. Every submit builds a fresh `SolverOption` and calls `InferenceEngine.solve()` — the UI has no inference logic of its own.

---

## Evaluating Against FD Ground Truth

```bash
# Evaluate a trained FNO
python src/evaluate.py fno \
    --dataset pretrained_models/val_data \
    --model   pretrained_models/fno.pt

# Evaluate a trained DeepONet
python src/evaluate.py deeponet \
    --dataset pretrained_models/val_data \
    --model   pretrained_models/deeponet.pt

# Sanity check: FD against itself (should give ~zero error)
python src/evaluate.py fd \
    --dataset pretrained_models/val_data
```

Metrics: relative L2 error, RMSE, max absolute error, BC satisfaction, PDE residual — reported as mean / std / p50 / p90 / max.

---

## Dependencies

```
torch >= 2.2
numpy >= 1.26
sympy >= 1.12
streamlit >= 1.32
matplotlib >= 3.8
```

Python 3.10–3.12. Install with `pip install -e .` from the repo root.

---

## Roadmap

### Performance (partially done)
- [ ] **DeepONet inference feedback** — Surface fixed-resolution DeepONet artifact status and FD fallback reasons directly in the UI.
- [ ] **Pre-evaluate BC arrays before Jacobi loop** — `GeneralPDE.apply_boundary_conditions` calls sympy lambdas 5000 × 4 walls per solve. Pre-compute all four boundary value tensors once outside the loop and pass them in. ~30 lines across `pde_helpers.py` and `inference_engine.py`.
- [x] **Structured `logging`** — All progress output in `trainer.py` now uses `logging.getLogger(__name__)` (INFO/WARNING level) in place of `print()`. Formatted summary tables (`_print_test_summary`, `_print_smoke_summary`) remain as `print()` for human-readable terminal output.

### Scale
- [x] **`DataLoader`-based training** — Both `FNOTrainer` and `DeepONetTrainer` stream data through `PDEOperatorDataset` + `DataLoader` via `_DatasetTrainerBase._build_loaders_from_datasets()`. They share mirrored `_train_from_dataset()` and `_test_from_dataset()` control flow (cosine scheduler + epoch validation in both).
- [ ] **Pre-parse PDE/BC coefficients at generation time** — `_load_examples()` calls sympy `parse_pde()` per sample; at 50 ms × 50k samples = 40+ min before the first training step. Store the 6 parsed float coefficients (`a, b, c, d, e, f`) directly in the `.npz` at generation time in `_solve_one_worker()`, then reconstruct `ParsedPDE` from floats in `_load_examples()`.
- [ ] **HPC scheduler/worker** — Generation is already parallelisable with the current `ProcessPoolExecutor` setup. Training uses `DataLoader` but still runs single-process; DDP + mid-run checkpointing are needed before it can be distributed across nodes.

### Future
- [ ] **Fine-tune on more cases** — The offline training infrastructure is sound. Scaling to larger LHS samples is straightforward once sympy pre-parsing removes the `_load_examples()` load-time bottleneck for the test path.
- [ ] **N-D support** — `SpectralConv` and `SoftGating` in `fno_layers.py` are already fully N-D. Every layer from `ConditionalGrid2D` downward is 2D-hardcoded (`B, H, W, _` unpacking, 6 named 2D PDE coefficients, fixed `FEATURE_DIM=25` OOD manifest). Requires redesigning the input encoding, parser, and OOD featurizer simultaneously.
