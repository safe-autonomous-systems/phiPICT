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
// Declarations of mhd/potential.cu.

#ifndef _PHIPICT_MHD_POTENTIAL_H
#define _PHIPICT_MHD_POTENTIAL_H

#include "domain/domain_structs.h"
#include "solvers/linear_solvers.h"

/** Build the Laplacian matrix into domain->Epot.
 *  For THIN_WALL potential BC face cells, the thin-wall BC
 *  dphi/dn = Cw * nabla^2_tau(phi) is incorporated directly as tangential
 *  surface Laplacian corrections — no augmented system is needed.
 *  Requires domain->Epot to be allocated via domain->CreateEpotMatrix(). */
void SetupEpotMatrix(std::shared_ptr<Domain> domain, const int8_t nonOrthoFlags, const bool useFaceTransform);

/** Compute the electric-potential Poisson RHS, div(u×B), for the inductionless MHD solve.
 *  vectorField: flat u×B field [totalSize * spatialDims], same layout as domain.pressureRHS.
 *  At a solid wall the normal flux is dropped (insulating, j_n=0); at an open boundary it is
 *  the cell-centre (u×B)_n (∂φ/∂n=0, current exits and closes virtually outside). Walls and
 *  open bounds are both BoundaryType::FIXED and are told apart by their velocity BC, see
 *  isInsulatingWallBound. Either way this matches the reconstruction in
 *  ComputeCurrentDensityFaceBased so that div(j)=0 holds discretely at boundary cells.
 *  Returns a new tensor of shape [totalSize]. */
torch::Tensor ComputeEpotRHS(std::shared_ptr<Domain> domain, const torch::Tensor &vectorField);

/** Compute the gradient of an arbitrary flat scalar field with Neumann (zero-gradient) BCs.
 *  scalarField: flat tensor of shape [totalSize], same layout as domain.pressureResult.
 *  Returns a new tensor of shape [totalSize * spatialDims]. */
torch::Tensor ComputeFieldGradient(std::shared_ptr<Domain> domain, const torch::Tensor &scalarField);

/** Compute FVM (Green-Gauss) gradient of an arbitrary flat scalar field.
 *  Uses face-based interpolation consistent with the Laplacian stencil.
 *  scalarField: flat tensor of shape [totalSize].
 *  Returns a new tensor of shape [totalSize * spatialDims]. */
torch::Tensor ComputeFieldGradientFVM(std::shared_ptr<Domain> domain, const torch::Tensor &scalarField);

/** Compute Lorentz force using face-based current density reconstruction.
 *  Guarantees discrete div(j)=0 by using the same stencil as the Poisson solve.
 *  epotField: flat [totalSize], uCrossBField: flat [totalSize * spatialDims or totalSize*3 for 2D].
 *  Returns current density tensor of shape [totalSize * 3]. */
torch::Tensor ComputeCurrentDensityFaceBased(std::shared_ptr<Domain> domain,
		const torch::Tensor &epotField, const torch::Tensor &uCrossBField);

#ifdef WITH_GRAD

torch::Tensor ComputeEpotRHSGrad(std::shared_ptr<Domain> domain,
		const torch::Tensor &gradDivergence);

std::pair<torch::Tensor, torch::Tensor> ComputeCurrentDensityFaceBasedGrad(
		std::shared_ptr<Domain> domain,
		const torch::Tensor &epotField,
		const torch::Tensor &uCrossBField,
		const torch::Tensor &gradJ);

#endif //WITH_GRAD

#endif //_PHIPICT_MHD_POTENTIAL_H
