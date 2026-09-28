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
// Declarations of grid/transform_vectors.cu.

#ifndef _PHIPICT_GRID_TRANSFORM_VECTORS_H
#define _PHIPICT_GRID_TRANSFORM_VECTORS_H

#include "domain/domain_structs.h"
#include "solvers/linear_solvers.h"

/** Compute dot(transform.T, vector), or dot(transform.T_inv, vector) if inverse.
  * Output to new tensor.
  */
torch::Tensor TransformVectors(const torch::Tensor &vectors, const torch::Tensor &transforms, const bool inverse);

#endif //_PHIPICT_GRID_TRANSFORM_VECTORS_H
