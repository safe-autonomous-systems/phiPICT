# phiPICT: solver changes over PICT

An overview of the main changes to the linear solvers and the simulation core relative to PICT, and how they are implemented. Every feature can be switched off for comparison (see *Ablation* below).

## Fused Krylov solvers

`csrc/phipict/solvers/krylov/`

PICT's CG and BiCGStab call cuBLAS and cuSPARSE and read scalars back to the host several times per iteration. On small grids that overhead dominates.

- CG and BiCGStab are rewritten as a few fused kernels per iteration: SpMV, vector updates and dot products in one pass.
- The scalar recurrences run on the device, in the last block of each reduction, so a solve never synchronises with the host.
- The iteration loop is captured once into a CUDA graph with a conditional WHILE node, so a whole solve is one graph launch.
- Workspaces come from PyTorch's caching allocator, and everything runs on PyTorch's stream.
- Optional Jacobi preconditioning is applied inside the SpMV.

Result: 8–75× faster solves on small (2D) systems and 1.5–3× on large (3D) ones, with the same iteration counts as the legacy solvers. The legacy solvers remain available (`_C.SetKrylovSettings(enabled=False)`).

## Native AMG-preconditioned CG

`_C.AMGPCGSolve`, `src/phipict/solvers/amg.py`

The existing torch AMG-PCG, used for the electric potential, is replaced by a CUDA implementation:
- the V-cycle and the PCG loop run in one CUDA graph;
- the Jacobi smoother is fused with the residual;
- row-parallel SpMV kernels adapt to each level's row length;
- the V-cycle runs in fp32 for fp64 solves.

The potential solve becomes about 3–4× faster at the same iteration count.

## AMG for the pressure solve

The pressure matrix changes every step, so PICT solves it with plain CG, which needs 1000–2600 iterations on the 3D ducts.

- phiPICT keeps the AMG interpolation (P, R) of the grid frozen and recomputes only the coarse operators `R A P` on the GPU (Galerkin refresh), fully every few steps. This works because the pressure matrix keeps the sparsity pattern of the potential matrix.
- The interpolation is set up once per sparsity pattern with pyamg. The pressure and potential solves and every env reset on the same grid share it, and it can also be cached on disk (`PHIPICT_AMG_CACHE_DIR`).
- The pressure then needs 6–22 iterations per solve.
- It is on automatically for large systems (≥ 200k unknowns).
- Memory stays bounded: only P/R are cached, indices are int32 and shared, and the Galerkin products run in chunks.

## Smaller features

- **Pressure warm start:** each pressure solve starts from the previous result.
- **Intermediate pressure tolerance:** a looser tolerance for all but the last corrector.
- **Potential result reuse** (MHD): each potential solve starts from the previous potential.

## Batched environments

`Domain.setBatchSize(B)` runs B copies of the same environment in one simulation, so that small environments, which are limited by host overhead, share every kernel launch.

- **Data.**
  - State fields get a leading batch dimension.
  - Grid, sparsity patterns, the potential matrix and the AMG interpolation are shared.
  - Boundary and static data can be shared or given per environment.
- **Kernels.** The device-side domain description exists once per environment, and every kernel picks its environment by `blockIdx.y`.
- **Solves.**
  - Batched solves iterate until every environment has converged.
  - Relative tolerances are resolved per environment, exactly as in independent runs.
  - The pressure AMG refreshes the coarse operators of all environments with one sparse product per level.
- **Differentiable path.** Batching works there too: each environment has its own gradients, and shared parameters get the sum over the environments.
- **Adaptive substeps.** Each environment splits the time step by its own CFL condition (`adaptive_CFL_per_env`, on by default), so it takes the substeps it would take on its own.
  - The time step is stored in each environment's device domain, so every kernel reads its environment's step.
  - The batch runs as many substeps as the environment that needs the most. An environment that is already done runs the extra substeps too, and its state is restored afterwards. The work is the same as with shared substeps.
  - Hooks receive the `[B]` substep sizes as `time_step`.
  - The differentiable path keeps shared substeps.

Throughput on a small 2D MHD duct: 213 env steps/s unbatched, about 1900 at B = 64–128. Batched results match separate runs (`tests/test_batching.py`).

## Code structure

The monolithic CUDA kernel file and `domain_structs.cpp` are split by topic under `csrc/phipict/`: `piso/`, `mhd/`, `solvers/`, `domain/`, `grid/`, `math/` and `common/`. See `csrc/phipict/README.md`. Build with `MAX_JOBS=1 python setup.py build_ext --inplace`.

## Ablation

```bash
python runscripts/benchmark/ablate_solver_improvements.py env_id=<env id>
python runscripts/benchmark/eval_solver_ablation.py --env-id <env id>
```

- **Variants:** `runscripts/configs/solver_ablation/pict.yaml` (all features off) and `phipict.yaml` (all on).
- **Outputs:** `output/solver_ablation/<env_id>/`.
- **Protocol:** 25 timed env steps with seeded random actions per variant; reset not timed.

Results (A100-40GB, fp64, 5 PISO steps per env step; `output/solver_ablation/<env_id>/`):

| Environment | PICT | phiPICT | Speedup | Peak memory (PICT / phiPICT) |
|---|---|---|---|---|
| `HartmannSmall3D-downward-easy-v0` (5M cells) | 79.9 ± 2.0 s | 1.61 ± 0.05 s | 49.7× | 6.3 / 12.7 GiB |
| `CylinderJet3D-hard-v0` (2.5M cells) | 1.87 ± 0.12 s | 0.61 ± 0.01 s | 3.1× | 2.0 / 3.2 GiB |

- **Hartmann:** PICT's pressure CG needs ~1700 iterations per solve and its unpreconditioned potential solve ~9800 (cap 10,000); phiPICT needs 8.7 and 19.
- **Cylinder** (2.5M cells, no MHD): pressure AMG (on automatically at this size), warm start and intermediate tolerance bring the pressure solves from 73 to 0.55 iterations on average: most of them converge immediately.
- **Convergence:** all solves of both variants reach their tolerance.
- **Medium 3D duct** (10M cells, earlier version of the ablation, 5 steps): 140 s vs. 3.7 s (38×). There PICT's potential solve hits the iteration cap, so its time is a lower bound.

The validation cases (Hartmann, Hunt, Shercliff) reproduce the reference results within the solver tolerances.
