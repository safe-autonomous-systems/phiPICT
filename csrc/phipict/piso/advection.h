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
// Declarations of piso/advection.cu.

#ifndef _PHIPICT_PISO_ADVECTION_H
#define _PHIPICT_PISO_ADVECTION_H

#include "domain/domain_structs.h"
#include "solvers/linear_solvers.h"

/** Build the matrix for the advection-diffusion system (prediction step)
  * inputs:
  * - domain.viscosity
  * - block.velocity
  * - block.transform (inverse and determinant)
  * - boundary.velocity
  * - boundary.transform
  * outputs (writes to):
  * - domain.C: full sparse matrix (value, row, index)
  * - domain.A: diagonal of C
  */
void SetupAdvectionMatrixEulerImplicit(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
	const int8_t nonOrthoFlags, const bool forPassiveScalar, const index_t passiveScalarChannel);

/** Build the RHS of the prediction step for the passive scalar domain.scalarRHS. Adds non-orthogonal components using domain.scalarResult.
  * inputs:
  * - domain.viscosity
  * - domain.scalarResult (only if RHS nonOrthoFlags are set)
  * - block.passiveScalar
  * - block.transform
  * - boundary.passiveScalar
  * - boundary.velocity
  * - boundary.transform
  * outputs (writes to):
  * - domain.scalarRHS
  */
void SetupAdvectionScalarEulerImplicitRHS(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags);

/** Build the RHS of the prediction step for the velocity domain.velocityRHS. Adds non-orthogonal components using domain.velocityResult.
  * inputs:
  * - domain.viscosity
  * - domain.velocityResult (only if RHS nonOrthoFlags are set)
  * - block.velocity
  * - block.velocitySource
  * - block.pressure (if applyPressureGradient)
  * - block.transform
  * - boundary.velocity
  * - boundary.transform
  * outputs (writes to):
  * - domain.velocityRHS
  */
void SetupAdvectionVelocityEulerImplicitRHS(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags, const bool applyPressureGradient);

/* --- Backwards/Gradient kernels --- 
 * Behaves like an "inverse" of the forward operations/kernels, output_grad->input_grad.
 * Not all gradient paths are implemented, meaning that not all forward inputs are differentiable (e.g. transformations metrics, boundary values).
 */

/** Build all 3 above.*/
//void SetupAdvectionEulerImplicitCombined(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags, const bool applyPressureGradient);

#ifdef WITH_GRAD

/**
  * inputs:
  * - domain.C_grad
  * - domain.A_grad
  * - block.transform
  * - boundary.transform
  * outputs:
  * - domain.viscosity_grad
  * - block.velocity_grad
  * - boundary.velocity_grad
  * not differentiable (outputs not implemented):
  * - block.transform_grad
  * - boundary.transform_grad
  */
void SetupAdvectionMatrixEulerImplicit_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
	const int8_t nonOrthoFlags, const bool forPassiveScalar, const index_t passiveScalarChannel);

/**
  * inputs:
  * - domain.scalarRHS_grad
  * - domain.viscosity
  * - block.transform
  * - boundary.passiveScalar
  * - boundary.velocity
  * - boundary.transform
  * outputs:
  * - domain.viscosity_grad
  * - domain.scalarResult_grad (only if RHS nonOrthoFlags are set)
  * - block.passiveScalar_grad
  * - boundary.passiveScalar_grad
  * - boundary.velocity_grad
  * not differentiable:
  * - block.transform_grad
  * - boundary.transform_grad
  */
void SetupAdvectionScalarEulerImplicitRHS_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags);

/**
  * inputs:
  * - domain.velocityRHS_grad
  * - domain.viscosity
  * - block.transform
  * - boundary.velocity
  * - boundary.transform
  * outputs:
  * - domain.viscosity_grad
  * - domain.velocityResult_grad (only if RHS nonOrthoFlags are set)
  * - block.velocity_grad
  * - block.velocitySource_grad
  * - boundary.velocity_grad
  * not differentiable:
  * - block.pressure_grad (TODO?, only if applyPressureGradient)
  * - block.transform_grad
  * - boundary.transform_grad
  */
void SetupAdvectionVelocityEulerImplicitRHS_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags, const bool applyPressureGradient);

#endif //WITH_GRAD

#endif //_PHIPICT_PISO_ADVECTION_H
