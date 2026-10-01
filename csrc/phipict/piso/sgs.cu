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
// Sub-grid scale models (Smagorinsky, WALE).

#include "piso/device/common.cuh"



/* --- Sub-Grid Scale models --- */

template<typename scalar_t, int DIMS>
__global__
void k_SGSviscosityIncompressibleSmagorinsky(DomainGPU<scalar_t> *p_domain, const scalar_t coefficient, scalar_t **pp_blockViscosity_out, 
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){
	
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		
		scalar_t d = 0; // norm of strain-rate tensor
		{
			MatrixSquare<scalar_t, DIMS> velocityGrads = {0};
			I4 tempPos = pos;
			for(index_t i=0; i<DIMS; ++i){
				tempPos.w = i;
				velocityGrads.v[i] = getBlockDataGradient<scalar_t, DIMS>(tempPos, s_block, s_domain, GridDataType::VELOCITY);
			}

			for(index_t i=0; i<DIMS ; ++i){
				for(index_t j=i; j<DIMS; ++j){
					scalar_t s = static_cast<scalar_t>(0.5) * (velocityGrads.a[i][j] + velocityGrads.a[j][i]);
					s *= s;
					if(i!=j){
						s *= 2; // abusing symmetry with j=i start for inner loop
					}
					d += s;
				}
			}
			d = sqrt(2*d);
		}

		// NOTE: this filter width is the max cell edge length, not the usual (dx*dy*dz)^(1/3),
		// and 'coefficient' is therefore C_s^2 rather than C_s. Existing C_smag values are tuned
		// against this convention; changing it invalidates them. SGSviscosityIncompressibleWALE
		// below deliberately uses cellVolume^(1/DIMS) and the textbook coefficient instead.
		scalar_t delta = 0; // cell size, here max length along the grid axes
		if(s_block.hasTransform){
			// first transform matrix contains the cell size as column vectors
			MatrixSquare<scalar_t, DIMS> transformMetrics = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(s_block.transform)[flattenIndex(pos, s_block)].M;
			for(index_t i=0; i<DIMS ; ++i){
				// length/magnitude of a column vector
				scalar_t colMag = 0;
				for(index_t rowIdx=0; rowIdx<DIMS; ++rowIdx){
					colMag += transformMetrics.a[rowIdx][i] * transformMetrics.a[rowIdx][i];
				}
				//colMag = _sqrtT<scalar_t>(colMag) no need to compute sqrt for every dimension
				delta = max(delta, colMag); // TODO: rows or columns? columns
			}
			//delta = _sqrtT<scalar_t>(delta); is squared later anyway
		} else {
			delta = 1;
		}

		pp_blockViscosity_out[targetBlockIdx][blockIdx.y*s_block.stride.w + flatPos] = coefficient * delta * d; // * delta; [B, 1, cells] per block
	)

}

template<typename scalar_t, int DIMS>
__host__
void _SGSviscosityIncompressibleSmagorinsky(std::shared_ptr<Domain> domain, const torch::Tensor coefficient, std::vector<torch::Tensor> SGSviscosities){
	
	const size_t alignmentBytes = alignof(scalar_t*);
	const size_t atlasSizeBytes = sizeof(scalar_t*) * SGSviscosities.size();
	size_t allocSizeBytes = atlasSizeBytes + alignmentBytes;
	
	auto byteOptions = torch::TensorOptions().dtype(torch::kUInt8).layout(torch::kStrided).device(domain->getDevice().type(), domain->getDevice().index());
	auto byteOptionsCPU = torch::TensorOptions().dtype(torch::kUInt8).layout(torch::kStrided);
	
	torch::Tensor t_pointers_SGSviscosities_CPU = torch::zeros(allocSizeBytes, byteOptionsCPU);
	torch::Tensor t_pointers_SGSviscosities_GPU = torch::zeros(allocSizeBytes, byteOptions);
	
	void *p_host = reinterpret_cast<void*>(t_pointers_SGSviscosities_CPU.data_ptr<uint8_t>());
	void *p_device = reinterpret_cast<void*>(t_pointers_SGSviscosities_GPU.data_ptr<uint8_t>());
	
	TORCH_CHECK(std::align(alignmentBytes, atlasSizeBytes, p_host, allocSizeBytes), "Failed to align CPU block viscosity atlas.")
	TORCH_CHECK(std::align(alignmentBytes, atlasSizeBytes, p_device, allocSizeBytes), "Failed to align GPU block viscosity atlas.")
	
	// pointer to host memory containing device pointers
	scalar_t **pp_SGSviscosities_host = reinterpret_cast<scalar_t**>(p_host);
	for(size_t i=0; i<SGSviscosities.size(); ++i){
		pp_SGSviscosities_host[i] = SGSviscosities[i].data_ptr<scalar_t>();
	}
	
	CopyToGPU(p_device, p_host, atlasSizeBytes);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	// gradient
	BEGIN_SAMPLE;
	k_SGSviscosityIncompressibleSmagorinsky<scalar_t, DIMS><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), coefficient.data_ptr<scalar_t>()[0],
			reinterpret_cast<scalar_t**>(p_device),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("k_SGSviscosityIncompressibleSmagorinsky");
}

std::vector<torch::Tensor> SGSviscosityIncompressibleSmagorinsky(std::shared_ptr<Domain> domain, const torch::Tensor coefficient){

	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	
	CHECK_INPUT_HOST(coefficient);
	TORCH_CHECK(coefficient.dim()==1 && coefficient.size(0)==1, "coefficient tensor must have shape [1]");
	
	std::vector<torch::Tensor> SGSviscosities;
	for(const auto &block : domain->getBlocks()){
		SGSviscosities.push_back(torch::zeros_like(block->pressure)); //pressure has correct shape with a single channel and must always exist.
	}
	
	DISPATCH_FTYPES_DIMS(domain, "SGSviscosityIncompressibleSmagorinsky", 
		// TODO: allocate tensor for pointers (**scalar_t)
		// gather pointers from SGSviscosities and copy to GPU tensor
		// dispatch k_SGSviscosityIncompressibleSmagorinsky
		_SGSviscosityIncompressibleSmagorinsky<scalar_t, dim>(domain, coefficient, SGSviscosities);
	);
	
	return SGSviscosities;
}

/* --- WALE (Wall-Adapting Local Eddy-viscosity, Nicoud & Ducros 1999) --- */

template <typename scalar_t>
__device__ inline scalar_t _cbrtT(const scalar_t &x);
template <>
__device__ inline float _cbrtT<float>(const float &x){ return cbrtf(x); }
template <>
__device__ inline double _cbrtT<double>(const double &x){ return cbrt(x); }

/** Squared filter width Delta^2 = |cellVolume|^(2/DIMS). */
template <typename scalar_t, int DIMS>
__device__ inline scalar_t _filterWidthSqrFromDet(const scalar_t det){
	const scalar_t vol = det < static_cast<scalar_t>(0) ? -det : det;
	if(DIMS==1){
		return vol*vol;
	} else if(DIMS==2){
		return vol;
	} else { // DIMS==3
		const scalar_t d = _cbrtT<scalar_t>(vol);
		return d*d;
	}
}

/** WALE eddy viscosity.
 *  nu_t = (Cw*Delta)^2 * (Sd:Sd)^(3/2) / ( (S:S)^(5/2) + (Sd:Sd)^(5/4) )
 *  with g_ij = du_i/dx_j, g2 = g.g, S = sym(g), Sd = sym(g2) - tr(g2)/DIMS * I.
 *  NOTE: coefficientSqr is Cw^2; the host wrapper squares the textbook Cw it is given.
 *  NOTE: for DIMS<3 and divergence-free g, Cayley-Hamilton makes g2 isotropic, so Sd and hence
 *        nu_t vanish identically. WALE is only meaningful in 3D; the Python layer rejects ndims<3. */
template <typename scalar_t, int DIMS>
__global__
void k_SGSviscosityIncompressibleWALE(DomainGPU<scalar_t> *p_domain, const scalar_t coefficientSqr, scalar_t **pp_blockViscosity_out,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){

	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,

		const I4 pos = unflattenIndex(flatPos, s_block);

		// velocity gradient tensor: g.a[i][j] = du_i/dx_j
		MatrixSquare<scalar_t, DIMS> g = {0};
		{
			I4 tempPos = pos;
			for(index_t i=0; i<DIMS; ++i){
				tempPos.w = i;
				g.v[i] = getBlockDataGradient<scalar_t, DIMS>(tempPos, s_block, s_domain, GridDataType::VELOCITY);
			}
		}

		// g2 = g.g  ->  g2.a[i][j] = g_ik g_kj
		MatrixSquare<scalar_t, DIMS> g2 = {0};
		for(index_t i=0; i<DIMS; ++i){
			for(index_t j=0; j<DIMS; ++j){
				scalar_t s = 0;
				for(index_t k=0; k<DIMS; ++k){
					s += g.a[i][k] * g.a[k][j];
				}
				g2.a[i][j] = s;
			}
		}

		scalar_t g2Trace = 0;
		for(index_t i=0; i<DIMS; ++i){ g2Trace += g2.a[i][i]; }
		const scalar_t g2TraceOverDims = g2Trace / static_cast<scalar_t>(DIMS);

		scalar_t SS = 0;   // S_ij S_ij
		scalar_t SdSd = 0; // Sd_ij Sd_ij
		for(index_t i=0; i<DIMS; ++i){
			for(index_t j=i; j<DIMS; ++j){
				// abusing symmetry with j=i start for inner loop: off-diagonals count twice
				const scalar_t mult = (i==j) ? static_cast<scalar_t>(1) : static_cast<scalar_t>(2);

				const scalar_t S = static_cast<scalar_t>(0.5) * (g.a[i][j] + g.a[j][i]);
				SS += mult * S * S;

				scalar_t Sd = static_cast<scalar_t>(0.5) * (g2.a[i][j] + g2.a[j][i]);
				if(i==j){ Sd -= g2TraceOverDims; }
				SdSd += mult * Sd * Sd;
			}
		}

		// (Sd:Sd)^(3/2) / ( (S:S)^(5/2) + (Sd:Sd)^(5/4) ), all fractional powers via sqrt only.
		// The operator is homogeneous of degree 1 in g (nu_t ~ |g|*Delta^2), so it is unbounded
		// by design. The only degenerate case is 0/0, where both denominator terms underflow to
		// exactly 0 (e.g. zero/uniform flow); the den>0 guard returns 0 there, so no additive
		// epsilon, which would be dimensional and mis-scale the model, is needed.
		scalar_t opWALE = 0;
		{
			const scalar_t num  = SdSd * _sqrtT<scalar_t>(SdSd);
			const scalar_t den1 = SS * SS * _sqrtT<scalar_t>(SS);
			const scalar_t den2 = SdSd * _sqrtT<scalar_t>(_sqrtT<scalar_t>(SdSd));
			const scalar_t den  = den1 + den2;
			if(den > static_cast<scalar_t>(0)){
				opWALE = num / den;
			}
		}

		// Delta^2 = |cellVolume|^(2/DIMS); TransformGPU::det IS the cell volume.
		scalar_t deltaSqr = 1;
		if(s_block.hasTransform){
			const scalar_t det = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(s_block.transform)[flattenIndex(pos, s_block)].det;
			deltaSqr = _filterWidthSqrFromDet<scalar_t, DIMS>(det);
		}

		pp_blockViscosity_out[targetBlockIdx][blockIdx.y*s_block.stride.w + flatPos] = coefficientSqr * deltaSqr * opWALE; // [B, 1, cells] per block
	)

}

template<typename scalar_t, int DIMS>
__host__
void _SGSviscosityIncompressibleWALE(std::shared_ptr<Domain> domain, const torch::Tensor coefficient, std::vector<torch::Tensor> SGSviscosities){

	const size_t alignmentBytes = alignof(scalar_t*);
	const size_t atlasSizeBytes = sizeof(scalar_t*) * SGSviscosities.size();
	size_t allocSizeBytes = atlasSizeBytes + alignmentBytes;

	auto byteOptions = torch::TensorOptions().dtype(torch::kUInt8).layout(torch::kStrided).device(domain->getDevice().type(), domain->getDevice().index());
	auto byteOptionsCPU = torch::TensorOptions().dtype(torch::kUInt8).layout(torch::kStrided);

	torch::Tensor t_pointers_SGSviscosities_CPU = torch::zeros(allocSizeBytes, byteOptionsCPU);
	torch::Tensor t_pointers_SGSviscosities_GPU = torch::zeros(allocSizeBytes, byteOptions);

	void *p_host = reinterpret_cast<void*>(t_pointers_SGSviscosities_CPU.data_ptr<uint8_t>());
	void *p_device = reinterpret_cast<void*>(t_pointers_SGSviscosities_GPU.data_ptr<uint8_t>());

	TORCH_CHECK(std::align(alignmentBytes, atlasSizeBytes, p_host, allocSizeBytes), "Failed to align CPU block viscosity atlas.")
	TORCH_CHECK(std::align(alignmentBytes, atlasSizeBytes, p_device, allocSizeBytes), "Failed to align GPU block viscosity atlas.")

	// pointer to host memory containing device pointers
	scalar_t **pp_SGSviscosities_host = reinterpret_cast<scalar_t**>(p_host);
	for(size_t i=0; i<SGSviscosities.size(); ++i){
		pp_SGSviscosities_host[i] = SGSviscosities[i].data_ptr<scalar_t>();
	}

	CopyToGPU(p_device, p_host, atlasSizeBytes);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)

	// The binding takes the textbook Cw; square it here so the kernel receives Cw^2.
	const scalar_t Cw = coefficient.data_ptr<scalar_t>()[0];
	const scalar_t CwSqr = Cw * Cw;

	BEGIN_SAMPLE;
	k_SGSviscosityIncompressibleWALE<scalar_t, DIMS><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), CwSqr,
			reinterpret_cast<scalar_t**>(p_device),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("k_SGSviscosityIncompressibleWALE");
}

std::vector<torch::Tensor> SGSviscosityIncompressibleWALE(std::shared_ptr<Domain> domain, const torch::Tensor coefficient){


	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");

	CHECK_INPUT_HOST(coefficient);
	TORCH_CHECK(coefficient.dim()==1 && coefficient.size(0)==1, "coefficient tensor must have shape [1]");
	TORCH_CHECK(coefficient.item<double>()>=0, "WALE coefficient Cw must be non-negative.");

	std::vector<torch::Tensor> SGSviscosities;
	for(const auto &block : domain->getBlocks()){
		SGSviscosities.push_back(torch::zeros_like(block->pressure)); //pressure has correct shape with a single channel and must always exist.
	}

	DISPATCH_FTYPES_DIMS(domain, "SGSviscosityIncompressibleWALE",
		_SGSviscosityIncompressibleWALE<scalar_t, dim>(domain, coefficient, SGSviscosities);
	);

	return SGSviscosities;
}

