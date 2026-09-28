// Copyright 2026 Jannis Becktepe
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * Fused, host-sync-free Krylov solvers (CG and BiCGStab).
 *
 * Differences to the legacy cuBLAS/cuSPARSE solvers in cg_solver_kernel.cu and
 * bicgstab_solver_kernel.cu:
 *  - all scalars (alpha, beta, rho, ...) live on the device; the host only polls a
 *    "all batches done" flag every `checkInterval` iterations, asynchronously
 *  - SpMV and the dot products of an iteration are fused into one kernel, the
 *    scalar recurrences run in the last block of that kernel (deterministic,
 *    fixed-order reduction), vector updates are fused into one kernel
 *  - CG uses the single-reduction Chronopoulos-Gear formulation (one reduction
 *    phase per iteration)
 *  - work vectors live in a cached workspace allocated through PyTorch's caching
 *    allocator; everything runs on PyTorch's current stream
 *  - the iteration loop is replayed from captured CUDA graphs
 *  - every kernel carries a batch dimension: vectors are [nBatches, n], the matrix
 *    values are either shared by all batches (valBatchStride=0, e.g. velocity
 *    components) or one matrix per batch with the same sparsity pattern
 *    (valBatchStride=nnz, e.g. batched environments). Convergence is tracked per batch.
 */

#pragma once

#ifndef _INCLUDE_KRYLOV
#define _INCLUDE_KRYLOV

#include <string>
#include <vector>

#include "common/custom_types.h"
#include "solvers/linear_solvers.h"

namespace krylov {

enum class Preconditioner : int8_t {
	NONE = 0,
	JACOBI = 1,
};

struct Options {
	index_t maxit = 1000;
	double tol = 1e-8;
	bool normalized = true;         // NORM2_NORMALIZED (true) or NORM2 (false)
	index_t residualResetSteps = 0; // recompute r=b-Ax every n iterations (0: never)
	bool returnBest = false;        // return the iterate with the lowest residual if not converged
	Preconditioner precond = Preconditioner::NONE;
	// With per-batch matrix values (valBatchStride>0): number of consecutive right-hand
	// sides that share one matrix, e.g. the velocity components of one environment.
	index_t rhsPerMatrix = 1;
	// Optional per-batch tolerances (size nBatches); empty: `tol` for all batches.
	std::vector<double> tols;
};

struct GlobalSettings {
	bool enabled = true;           // use this backend in SolveLinear (false: legacy solvers)
	index_t checkInterval = 8;     // iterations between host convergence polls
	bool useGraphs = true;         // replay iteration chunks from CUDA graphs
	Preconditioner cgPrecond = Preconditioner::NONE;
	Preconditioner bicgPrecond = Preconditioner::NONE;
};

GlobalSettings& settings();

/* Solve A x = b for nBatches right-hand sides, in place in x.
 * A in CSR (row offsets aRow[n+1], column indices aIndex[nnz], values aVal).
 * Right-hand side k uses the matrix values at aVal + (k / rhsPerMatrix) * valBatchStride;
 * valBatchStride=0: one matrix shared by all right-hand sides. */
template <typename scalar_t>
solverReturn_t cgSolve(const scalar_t *aVal, const index_t *aIndex, const index_t *aRow, const index_t n, const index_t nnz,
	const index_t valBatchStride, const scalar_t *b, scalar_t *x, const index_t nBatches, const Options &opt);

template <typename scalar_t>
solverReturn_t bicgstabSolve(const scalar_t *aVal, const index_t *aIndex, const index_t *aRow, const index_t n, const index_t nnz,
	const index_t valBatchStride, const scalar_t *b, scalar_t *x, const index_t nBatches, const Options &opt);

/* Explicit transpose of a CSR matrix (values, indices, row offsets) into cached buffers;
 * numMatrices>1: consecutive value arrays of matrices with the same pattern.
 * Used for adjoint solves of non-symmetric systems, where a transposed SpMV with CSR
 * would need atomics. Returned pointers stay valid until the next call. */
template <typename scalar_t>
void csrTranspose(const scalar_t *aVal, const index_t *aIndex, const index_t *aRow, const index_t n, const index_t nnz,
	const index_t numMatrices, const scalar_t **tVal, const index_t **tIndex, const index_t **tRow);

void releaseWorkspaces();

} // namespace krylov

#endif // _INCLUDE_KRYLOV
