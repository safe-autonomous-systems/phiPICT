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
// Stencil helpers used by several kernel files (upwind stencil availability,
// pressure gradient dispatch over the spatial dimension).

#pragma once

#include "piso/device/csr_rows.cuh"

/* Whether the 3-cell 1D stencil (UU, U, D) needed by the deferred-correction
 * scheme is available for face `bound` of cell `pos` as plain interior cells of
 * the same block. `outflow` says whether the outward face flux is positive, i.e.
 * whether the upwind cell is `pos` itself (then UU is the opposite neighbour) or
 * the neighbour (then UU sits two cells away).
 *
 * Where the stencil is incomplete (i.e. block boundaries, periodic and connected
 * faces) central differencing is kept for that face. Both PISO_build_matrix
 * and kPISO_build_advection_RHS call this with the same fluxes, so the implicit
 * and explicit parts always agree on which faces are deferred. */
template <typename scalar_t>
__device__ inline bool hasFarUpwindCell(const I4 &pos, const index_t bound, const bool outflow, const BlockGPU<scalar_t> &block){
	const index_t axis = axisFromBound(bound);
	const index_t faceSign = faceSignFromBound(bound);
	const index_t last = block.size.a[axis] - 1;
	const index_t iD = pos.a[axis] + faceSign;
	if(iD < 0 || iD > last){ return false; }
	const index_t iUU = outflow ? (pos.a[axis] - faceSign) : (pos.a[axis] + 2*faceSign);
	return iUU >= 0 && iUU <= last;
}

template<typename scalar_t>
__device__ void tempGetPressureGradientDimSwitch(const BlockGPU<scalar_t> &block, const I4 &pos, const DomainGPU<scalar_t> &domain, scalar_t *pressureGradOut){
	switch(domain.numDims){
	case 1:
	{
		const Vector<scalar_t, 1> pressureGrad = getPressureGradient<scalar_t, 1>(block, pos, domain);
		pressureGradOut[0] = pressureGrad.a[0];
		return;
	}
	case 2:
	{
		const Vector<scalar_t, 2> pressureGrad = getPressureGradient<scalar_t, 2>(block, pos, domain);
		pressureGradOut[0] = pressureGrad.a[0];
		pressureGradOut[1] = pressureGrad.a[1];
		return;
	}
	case 3:
	{
		const Vector<scalar_t, 3> pressureGrad = getPressureGradient<scalar_t, 3>(block, pos, domain);
		pressureGradOut[0] = pressureGrad.a[0];
		pressureGradOut[1] = pressureGrad.a[1];
		pressureGradOut[2] = pressureGrad.a[2];
		return;
	}
	default:
		return;
	}
}

template<typename scalar_t>
__device__ void tempGetPressureGradientFVMDimSwitch(const BlockGPU<scalar_t> &block, const I4 &pos, const DomainGPU<scalar_t> &domain,
		const index_t gradientInterpolation, scalar_t *pressureGradOut){
	switch(domain.numDims){
	case 1:
	{
		const Vector<scalar_t, 1> pressureGrad = getPressureGradientFVM<scalar_t, 1>(block, pos, domain, gradientInterpolation);
		pressureGradOut[0] = pressureGrad.a[0];
		return;
	}
	case 2:
	{
		const Vector<scalar_t, 2> pressureGrad = getPressureGradientFVM<scalar_t, 2>(block, pos, domain, gradientInterpolation);
		pressureGradOut[0] = pressureGrad.a[0];
		pressureGradOut[1] = pressureGrad.a[1];
		return;
	}
	case 3:
	{
		const Vector<scalar_t, 3> pressureGrad = getPressureGradientFVM<scalar_t, 3>(block, pos, domain, gradientInterpolation);
		pressureGradOut[0] = pressureGrad.a[0];
		pressureGradOut[1] = pressureGrad.a[1];
		pressureGradOut[2] = pressureGrad.a[2];
		return;
	}
	default:
		return;
	}
}

