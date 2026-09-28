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
// Declarations of piso/pressure.cu.

#ifndef _PHIPICT_PISO_PRESSURE_H
#define _PHIPICT_PISO_PRESSURE_H

#include "domain/domain_structs.h"
#include "solvers/linear_solvers.h"

/** Build the whole pressure correction system:
  * domain.P: sparse matrix for linear system
  * domain.pressureRHS: RHS as vector field, buffered for velocity correction
  * domain.pressureRHSdiv: divergence of pressureRHS, the RHS used in the linear system. includes non-orthogonal components added after the divergence.
  */
void SetupPressureCorrection(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm);

/** Build only the pressure matrix.
  * inputs:
  * - domain.A
  * - block.transform
  * outputs (writes to):
  * - domain.P: full sparse matrix (value, row, index)
  */
void SetupPressureMatrix(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags, const bool useFaceTransform);

/** Build domain.pressureRHS and domain.pressureRHSdiv 
  * inputs:
  * - domain.A
  * - domain.C
  * - domain.velocityResult
  * - domain.pressureResult (only if RHS nonOrthoFlags are set)
  * - block.velocity
  * - block.velocitySource
  * - block.transform
  * - boundary.velocity
  * - boundary.transform
  * outputs (writes to):
  * - domain.pressureRHS
  * - domain.pressureRHSdiv
  */
void SetupPressureRHS(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm);

/** Build only domain.pressureRHSdiv, still including the non-orthogonal components.
  * inputs:
  * - domain.A (only if RHS nonOrthoFlags are set)
  * - domain.pressureRHS
  * - domain.pressureResult (only if RHS nonOrthoFlags are set)
  * - block.transform
  * - boundary.velocity
  * - boundary.transform
  * outputs (writes to):
  * - domain.pressureRHSdiv
  */
void SetupPressureRHSdiv(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm);

#ifdef WITH_GRAD

/** Backwards pass for: non-orthogonal RHS components (TODO), pressureRHSdiv, pressureRHS, pressure matrix (not implemented)
  * inputs/outputs: identical to SetupPressureRHS_GRAD
  * not differentiable:
  * - (complete pressure matrix setup, only uses domain.A and transforms)
  */
void SetupPressureCorrection_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
	const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm);

void SetupPressureMatrix_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
	const int8_t nonOrthoFlags, const bool useFaceTransform);

/** Backwards pass for: non-orthogonal RHS components (TODO), pressureRHSdiv, pressureRHS
  * inputs:
  * - domain.C
  * - domain.A
  * - domain.viscosity
  * - domain.pressureRHS_grad
  * - domain.pressureRHSdiv_grad
  * - block.transform
  * - boundary.velocity
  * - boundary.transform
  * outputs:
  * - domain.pressureResult_grad (only if RHS nonOrthoFlags are set)
  * - domain.pressureRHS_grad
  * - domain.velocityResult_grad
  * - domain.viscosity_grad
  * - block.velocity_grad
  * - block.velocitySource_grad
  * not differentiable:
  * - domain.C_grad (TODO)
  * - domain.A_grad (TODO)
  */
void SetupPressureRHS_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
	const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm);

/** Backwards pass for: non-orthogonal RHS components (TODO), pressureRHSdiv
  * inputs:
  * - domain.pressureRHSdiv_grad
  * - block.transform
  * outputs:
  * - domain.pressureResult_grad (TODO, only if RHS nonOrthoFlags are set)
  * - domain.pressureRHS_grad
  * not differentiable:
  * - block.transform_grad
  */
void SetupPressureRHSdiv_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
	const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm);

#endif //WITH_GRAD

#endif //_PHIPICT_PISO_PRESSURE_H
