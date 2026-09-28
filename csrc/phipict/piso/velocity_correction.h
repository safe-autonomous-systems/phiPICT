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
// Declarations of piso/velocity_correction.cu.

#ifndef _PHIPICT_PISO_VELOCITY_CORRECTION_H
#define _PHIPICT_PISO_VELOCITY_CORRECTION_H

#include "domain/domain_structs.h"
#include "solvers/linear_solvers.h"

/** Make the velocity divergence free using the pressure gradient: velocityResult = pressureRHS - grad(pressure).
  * inputs:
  * - domain.A
  * - domain.pressureRHS (= predicted velocity)
  * - block.pressure
  * - block.transform
  * outputs (writes to):
  * - domain.velocityResult
  */
void CorrectVelocity(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const index_t version, const bool timeStepNorm);

#ifdef WITH_GRAD

/**
  * inputs:
  * - domain.A
  * - domain.velocityResult_grad
  * - block.transform
  * outputs:
  * - domain.pressureRHS_grad
  * - block.pressure_grad
  * not differentiable:
  * - domain.A_grad
  */
void CorrectVelocity_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const bool timeStepNorm);

#endif //WITH_GRAD

#endif //_PHIPICT_PISO_VELOCITY_CORRECTION_H
