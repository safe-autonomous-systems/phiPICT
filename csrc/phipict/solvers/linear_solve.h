// Original work Copyright 2025 Aleksandra Franz, Nils Thuerey
// Modified work Copyright 2026 Jannis Becktepe
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
//
//
// Declarations of solvers/linear_solve.cu.

#ifndef _PHIPICT_SOLVERS_LINEAR_SOLVE_H
#define _PHIPICT_SOLVERS_LINEAR_SOLVE_H

#include "domain/domain_structs.h"
#include "solvers/linear_solvers.h"

solverReturn_t SolveLinear(std::shared_ptr<CSRmatrix> A, torch::Tensor RHS, torch::Tensor x, torch::Tensor maxit, torch::Tensor tol, const ConvergenceCriterion conv,
	const bool useBiCG, const bool matrixRankDeficient, const index_t residualResetSteps, const bool transposeA, const bool printResidual, const bool returnBestResult,
	const bool withPreconditioner);

/** Compute the outer product of vectors a and b. Write to result to out_pattern.value using its sparsity pattern. */
void SparseOuterProduct(torch::Tensor &a, torch::Tensor &b, std::shared_ptr<CSRmatrix> out_pattern);

#endif //_PHIPICT_SOLVERS_LINEAR_SOLVE_H
