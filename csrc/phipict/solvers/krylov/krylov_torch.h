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

/* Torch-facing entry points of the fused Krylov solvers (kept out of krylov.h so the
 * large simulation kernels do not depend on it). */

#pragma once

#include <vector>

#include <ATen/ATen.h>

#include "solvers/krylov/krylov.h"

namespace krylov {

/* AMG-preconditioned CG, fully on the GPU (one CUDA graph per solve).
 * The system matrix (aRow, aCol, aVal; int32 indices) is used for the Krylov iteration in the
 * dtype of rhs/x. `levels` holds, for every level but the coarsest,
 * (A row, A col, A val, inverse diagonal, R row, R col, R val, P row, P col, P val), with
 * int32 indices and values in the hierarchy dtype (the dtype of coarsePinv; float32 gives a
 * mixed-precision preconditioner for a float64 solve). coarsePinv is the dense
 * pseudo-inverse of the coarsest operator. Solves in place in x. `tolerances` optionally holds
 * one tolerance per right-hand side (batched environments), overriding `tol`. */
solverReturn_t amgPcgSolveTorch(const at::Tensor &aRow, const at::Tensor &aCol, const at::Tensor &aVal,
	const std::vector<std::vector<at::Tensor>> &levels, const at::Tensor &coarsePinv,
	const at::Tensor &rhs, at::Tensor &x, const index_t maxit, const double tol, const bool normalized, const bool returnBest,
	const bool projectConstant, const int preSweeps, const int postSweeps, const double omega,
	const std::vector<double> &tolerances = {});

} // namespace krylov
