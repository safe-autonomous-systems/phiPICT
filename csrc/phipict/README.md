# phiPICT C++/CUDA sources

All paths below are relative to this directory, which is the include root (`#include "piso/pressure.h"`).

| Directory | Contents |
|---|---|
| `api.h` | Umbrella header of all simulation functions exposed to Python (included by `bindings.cpp`). |
| `bindings.cpp` | The pybind11 module `phipict._C`. |
| `common/` | Basic types (`custom_types.h`, `grid_definitions.h`, `transformations.h`), dtype/dimension dispatch, logging and the `optional` shim. |
| `domain/` | Host-side data structures: `domain_structs.h` (declarations of `CSRmatrix`, the boundaries, `Block`, `Domain`), their implementations (`csr_matrix.cpp`, `boundaries.cpp`, `block.cpp`, `domain.cpp`) and the device-side mirror `domain_structs_gpu.h` (`DomainGPU`, `BlockGPU`: the atlas the kernels read). |
| `piso/` | The PISO simulation kernels and their host wrappers, one file per stage. Each `.cu` has a `.h` with its public functions. |
| `piso/device/` | Device helpers shared by all simulation kernels, in dependency order (each header includes the previous one; `common.cuh` includes all). |
| `mhd/` | Inductionless MHD: electric potential matrix, Poisson RHS, field gradients, face-based current density (and their gradients). |
| `solvers/` | Linear solvers: `linear_solve.cu` (the `SolveLinear` entry point), the fused Krylov solvers and native AMG-PCG (`krylov/`), the legacy cuBLAS/cuSPARSE CG and BiCGStab (`legacy_*.cu`, `linear_solvers.h`). |
| `grid/` | Grid generation, resampling to uniform grids, vector transforms. |
| `math/` | Small dense matrix/vector ops, eigen decomposition, orthogonal bases. |
| `noise/` | Simplex noise (separate, optional extension). |
| `legacy/` | The single-grid API of the original PICT (not compiled). |

## Files in `piso/`

| File | Contents |
|---|---|
| `advection.cu` | Advection-diffusion (prediction step): matrix and RHS, and their gradients. |
| `pressure.cu` | Pressure correction: matrix, RHS and divergence, and their gradients. |
| `velocity_correction.cu` | Velocity correction with the pressure gradient, and its gradient. |
| `result_copy.cu` | Copies between the block fields and the flat solve vectors. |
| `analysis.cu` | Diagnostics: velocity divergence, pressure gradient, spatial velocity gradients. |
| `sgs.cu` | Sub-grid scale models (Smagorinsky, WALE). |
| `launch.cu` | Host helpers of the kernel launches (thread-block layout, atlas upload). |
| `non_ortho_flags.h` | Flags selecting the non-orthogonal correction terms. |

## Files in `piso/device/`

The headers are listed in include order.

| File | Contents |
|---|---|
| `launch.cuh` | Launch configuration, index arithmetic, `KERNEL_PER_CELL_LOOP`. |
| `topology.cuh` | Boundaries, block connectivity, neighbour cell resolution. |
| `velocity.cuh` | Velocity access, contravariant components, pressure gradients. |
| `discretization.cuh` | Face transforms, interpolation, determinants, Laplace coefficients, face fluxes, viscosity. |
| `nonortho_laplace.cuh` | Non-orthogonal Laplace components, pressure with boundaries, velocity sources. |
| `block_data.cuh` | Block and boundary data access, neighbour and corner values, data gradients. |
| `nonortho_rhs.cuh` | Non-orthogonal Laplace RHS contributions. |
| `csr_rows.cuh` | Helpers to assemble CSR matrix rows. |

## Batched environments

The atlas holds one `DomainGPU` per environment, and every simulation kernel reads `p_domain[blockIdx.y]`. Gradient pointers are per environment too. A gradient of batch size 1 belongs to shared data and accumulates the sum over environments, which is why gradient kernels must write shared data with `atomicAdd`. See `SOLVER_CHANGES.md` in the repository root.

## Conventions

- Device functions defined in a header must be templates or `inline`. Several `.cu` files include the headers, so a plain definition is linked more than once.
