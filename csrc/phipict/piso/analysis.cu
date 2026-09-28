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
// Diagnostics: velocity divergence, pressure gradient, spatial velocity gradients.

#include "piso/device/common.cuh"


/* --- Utility --- */



template <typename scalar_t>
__global__ void k_computeVelocityDivergenceFromFlux(DomainGPU<scalar_t> *p_domain, scalar_t *divergence, //const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks, const index_t divergenceBatchStride){
	// batched environments: flat per-environment slices
	divergence += blockIdx.y * divergenceBatchStride;

	
	// divergence of colocated vector field
	// using central differences
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		
		scalar_t fluxes[6]; //DIMS*2
		computeFluxesNDLoop<scalar_t>(pos, fluxes, s_block, s_domain, nullptr);
			
		scalar_t div = 0;
		for(index_t dim=0;dim<s_domain.numDims;++dim){
			div += fluxes[dim*2+1] - fluxes[dim*2];
		}
		
		divergence[flatPos + s_block.globalOffset] = div;
	)
}

template <typename scalar_t>
void _ComputeVelocityDivergence(std::shared_ptr<Domain> domain, torch::Tensor &divergence){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	// divergence
	BEGIN_SAMPLE;
	k_computeVelocityDivergenceFromFlux<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), divergence.data_ptr<scalar_t>(), p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			divergence.numel()/domain->getBatchSize()
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Divergence Velocity");
	
}
torch::Tensor ComputeVelocityDivergence(std::shared_ptr<Domain> domain){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	
	torch::Tensor divergence = torch::zeros_like(domain->pressureResult);
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "ComputeVelocityDivergence", ([&] {
		_ComputeVelocityDivergence<scalar_t>(
			domain, divergence
		);
	}));
	
	return divergence;
}

// finite volume pressure gradient
template <typename scalar_t>
__global__ void k_computePressureGradient(DomainGPU<scalar_t> *p_domain, const bool useFVM, const index_t gradientInterpolation, scalar_t *gradient,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks, const index_t gradientBatchStride){
	// batched environments: flat per-environment slices
	gradient += blockIdx.y * gradientBatchStride;

	//vel.x = pressureRHS.x - inv(A)*gradX(pressure)
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		I4 pos = unflattenIndex(flatPos, s_block);
		
		scalar_t pressureGrad[3];
		if(useFVM){
			tempGetPressureGradientFVMDimSwitch<scalar_t>(s_block, pos, s_domain, gradientInterpolation, pressureGrad);
		}else{
			tempGetPressureGradientDimSwitch<scalar_t>(s_block, pos, s_domain, pressureGrad);
		}
		
		for(index_t dim=0;dim<s_domain.numDims;++dim){
			pos.w = dim;
			const index_t flatCompPosGlobal = flattenIndexGlobal(pos, s_block, s_domain);
			gradient[flatCompPosGlobal] = pressureGrad[dim];
		}
	)
}
template <typename scalar_t>
void _ComputePressureGradient(std::shared_ptr<Domain> domain, const bool useFVM, const index_t gradientInterpolation, torch::Tensor &gradient){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	// gradient
	BEGIN_SAMPLE;
	k_computePressureGradient<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), useFVM, gradientInterpolation, gradient.data_ptr<scalar_t>(),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			gradient.numel()/domain->getBatchSize()
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Pressure Gradient");
	
}
torch::Tensor ComputePressureGradient(std::shared_ptr<Domain> domain, const bool useFVM, const index_t gradientInterpolation){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	
	torch::Tensor gradient = torch::zeros_like(domain->velocityResult);
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "ComputePressureGradient", ([&] {
		_ComputePressureGradient<scalar_t>(
			domain, useFVM, gradientInterpolation, gradient
		);
	}));
	
	return gradient;
}

template<typename scalar_t, int DIMS>
__global__
void k_computeSpatialVelocityGradients(DomainGPU<scalar_t> *p_domain, scalar_t **pp_blockDimGradients_out, 
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){
	
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		// provides s_block, targetBlockIdx, flatPos
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		
		for(index_t dim=0; dim<DIMS; ++dim){
			// spatial gradients of the dim'th velocity component
			I4 tempPos = pos;
			tempPos.w = dim;
			Vector<scalar_t, DIMS> g = getBlockDataGradient<scalar_t, DIMS>(tempPos, s_block, s_domain, GridDataType::VELOCITY);
			
			for(index_t i=0; i<DIMS ; ++i){
				tempPos.w = i;
				// outputs [B, DIMS, cells] per block: environment blockIdx.y
				pp_blockDimGradients_out[targetBlockIdx*DIMS + dim][blockIdx.y*DIMS*s_block.stride.w + flattenIndex(tempPos, s_block)] = g.a[i];
			}
		}
	)
}
template <typename scalar_t>
void _ComputeSpatialVelocityGradients(std::shared_ptr<Domain> domain, std::vector<std::vector<torch::Tensor>> &gradients){
	
	const size_t numBlocks = domain->getNumBlocks();
	const size_t numDims = domain->getSpatialDims();
	const size_t numTensors = numBlocks * numDims;
	
	const size_t alignmentBytes = alignof(scalar_t*);
	const size_t atlasSizeBytes = sizeof(scalar_t*) * numTensors;
	size_t allocSizeBytes = atlasSizeBytes + alignmentBytes;
	
	auto byteOptions = torch::TensorOptions().dtype(torch::kUInt8).layout(torch::kStrided).device(domain->getDevice().type(), domain->getDevice().index());
	auto byteOptionsCPU = torch::TensorOptions().dtype(torch::kUInt8).layout(torch::kStrided);
	
	torch::Tensor t_pointers_gradients_CPU = torch::zeros(allocSizeBytes, byteOptionsCPU);
	torch::Tensor t_pointers_gradients_GPU = torch::zeros(allocSizeBytes, byteOptions);
	
	void *p_host = reinterpret_cast<void*>(t_pointers_gradients_CPU.data_ptr<uint8_t>());
	void *p_device = reinterpret_cast<void*>(t_pointers_gradients_GPU.data_ptr<uint8_t>());
	
	TORCH_CHECK(std::align(alignmentBytes, atlasSizeBytes, p_host, allocSizeBytes), "Failed to align CPU block viscosity atlas.")
	TORCH_CHECK(std::align(alignmentBytes, atlasSizeBytes, p_device, allocSizeBytes), "Failed to align GPU block viscosity atlas.")
	
	// pointer to host memory containing device pointers
	scalar_t **pp_gradients_host = reinterpret_cast<scalar_t**>(p_host);
	for(size_t i=0; i<numTensors; ++i){
		pp_gradients_host[i] = gradients[i/numDims][i%numDims].data_ptr<scalar_t>();
	}
	
	CopyToGPU(p_device, p_host, atlasSizeBytes);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	// gradient
	BEGIN_SAMPLE;
	SWITCH_DIMS(numDims,
		k_computeSpatialVelocityGradients<scalar_t, dim><<<blocks, threads>>>(
				reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), reinterpret_cast<scalar_t**>(p_device),
				p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
			);
	);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Spatial Velocity Gradient");
	
}
std::vector<std::vector<torch::Tensor>> ComputeSpatialVelocityGradients(std::shared_ptr<Domain> domain){

	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	
	std::vector<std::vector<torch::Tensor>> gradients;
	for(const auto &block : domain->getBlocks()){
		std::vector<torch::Tensor> blockGradients;
		for(index_t dim=0; dim < domain->getSpatialDims(); ++dim){
			blockGradients.push_back(torch::zeros_like(block->velocity));
		}
		gradients.push_back(blockGradients);
	}
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "ComputeSpatialVelocityGradients", ([&] {
		_ComputeSpatialVelocityGradients<scalar_t>(
			domain, gradients
		);
	}));
	
	return gradients;
}

