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
// Copies between the block fields and the flat solve vectors (and their gradients).

#include "piso/device/common.cuh"


/* --- Copy functions between the per-block grids and the corresponding global domain.*Result grids --- */

/* Copy `count` values for all B batched environments with one strided copy. The environment
 * slices of a flat solve vector are numel/B apart, those of a block field stride(0) apart. */
static inline int64_t envStride(const torch::Tensor &t, const index_t B){
	if(B<=1) return 0;
	return t.dim()==1 ? t.numel()/B : t.stride(0);
}
template <typename scalar_t>
static void copyBatched(scalar_t *dst, const int64_t dstEnvStride, const scalar_t *src, const int64_t srcEnvStride, const int64_t count, const index_t B){
	if(B<=1){
		CUDA_CHECK_RETURN(cudaMemcpy(dst, src, sizeof(scalar_t)*count, cudaMemcpyDeviceToDevice));
		return;
	}
	CUDA_CHECK_RETURN(cudaMemcpy2DAsync(dst, sizeof(scalar_t)*dstEnvStride, src, sizeof(scalar_t)*srcEnvStride, sizeof(scalar_t)*count, B,
		cudaMemcpyDeviceToDevice, 0));
}

template <typename scalar_t>
void _CopyScalarResultToBlocks(std::shared_ptr<Domain> domain){
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	
	for(auto block : domain->blocks){
		const index_t blockSizeFlat = block->getStrides().w;
		for(index_t channel=0; channel<domain->getPassiveScalarChannels(); ++channel){
			copyBatched<scalar_t>(
				block->getPassiveScalarDataPtr<scalar_t>() + blockSizeFlat*channel, envStride(block->passiveScalar.value(), domain->getBatchSize()),
				domain->scalarResult.data_ptr<scalar_t>() + block->globalOffset + domain->getTotalSize()*channel, envStride(domain->scalarResult, domain->getBatchSize()),
				blockSizeFlat, domain->getBatchSize()
			); //dst, src, count, environments
		}
	}
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("CopyScalarResultToBlocks");
	
}
void CopyScalarResultToBlocks(std::shared_ptr<Domain> domain){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	TORCH_CHECK(domain->hasPassiveScalar(), "Domain has no passive scalar set.")
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CopyScalarResultToBlocks", ([&] {
		_CopyScalarResultToBlocks<scalar_t>(domain);
	}));
}
template <typename scalar_t>
void _CopyScalarResultFromBlocks(std::shared_ptr<Domain> domain){
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	
	for(auto block : domain->blocks){
		const index_t blockSizeFlat = block->getStrides().w;
		for(index_t channel=0; channel<domain->getPassiveScalarChannels(); ++channel){
			copyBatched<scalar_t>(
				domain->scalarResult.data_ptr<scalar_t>() + block->globalOffset + domain->getTotalSize()*channel, envStride(domain->scalarResult, domain->getBatchSize()),
				block->getPassiveScalarDataPtr<scalar_t>() + blockSizeFlat*channel, envStride(block->passiveScalar.value(), domain->getBatchSize()),
				blockSizeFlat, domain->getBatchSize()
			); //dst, src, count, environments
		}
	}
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("CopyScalarResultFromBlocks");
	
}
void CopyScalarResultFromBlocks(std::shared_ptr<Domain> domain){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	TORCH_CHECK(domain->hasPassiveScalar(), "Domain has no passive scalar set.")
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CopyScalarResultFromBlocks", ([&] {
		_CopyScalarResultFromBlocks<scalar_t>(domain);
	}));
}
template <typename scalar_t>
void _CopyPressureResultToBlocks(std::shared_ptr<Domain> domain){
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	
	for(auto block : domain->blocks){
		copyBatched<scalar_t>(
			block->pressure.data_ptr<scalar_t>(), envStride(block->pressure, domain->getBatchSize()),
			domain->pressureResult.data_ptr<scalar_t>() + block->globalOffset, envStride(domain->pressureResult, domain->getBatchSize()),
			block->getStrides().w, domain->getBatchSize()
		); //dst, src, count, environments
	}
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("CopyPressureResultToBlocks");
	
}
void CopyPressureResultToBlocks(std::shared_ptr<Domain> domain){

	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");

	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CopyPressureResultToBlocks", ([&] {
		_CopyPressureResultToBlocks<scalar_t>(domain);
	}));
}
template <typename scalar_t>
void _CopyEpotResultToBlocks(std::shared_ptr<Domain> domain){

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;

	for(auto block : domain->blocks){
		TORCH_CHECK(block->hasEpot(), "Block does not have an epot tensor. Call CreateEpotOnBlocks() first.");
		copyBatched<scalar_t>(
			block->epot.value().data_ptr<scalar_t>(), envStride(block->epot.value(), domain->getBatchSize()),
			domain->epotResult.value().data_ptr<scalar_t>() + block->globalOffset, envStride(domain->epotResult.value(), domain->getBatchSize()),
			block->getStrides().w, domain->getBatchSize()
		); //dst, src, count, environments
	}

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("CopyEpotResultToBlocks");

}
void CopyEpotResultToBlocks(std::shared_ptr<Domain> domain){

	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	TORCH_CHECK(domain->hasEpot(), "Domain does not have epot fields. Call CreateEpotResult() first.");

	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CopyEpotResultToBlocks", ([&] {
		_CopyEpotResultToBlocks<scalar_t>(domain);
	}));
}
template <typename scalar_t>
void _CopyEpotResultFromBlocks(std::shared_ptr<Domain> domain){

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;

	for(auto block : domain->blocks){
		TORCH_CHECK(block->hasEpot(), "Block does not have an epot tensor. Call CreateEpotOnBlocks() first.");
		copyBatched<scalar_t>(
			domain->epotResult.value().data_ptr<scalar_t>() + block->globalOffset, envStride(domain->epotResult.value(), domain->getBatchSize()),
			block->epot.value().data_ptr<scalar_t>(), envStride(block->epot.value(), domain->getBatchSize()),
			block->getStrides().w, domain->getBatchSize()
		); //dst, src, count, environments
	}

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("CopyEpotResultFromBlocks");

}
void CopyEpotResultFromBlocks(std::shared_ptr<Domain> domain){

	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	TORCH_CHECK(domain->hasEpot(), "Domain does not have epot fields. Call CreateEpotResult() first.");

	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CopyEpotResultFromBlocks", ([&] {
		_CopyEpotResultFromBlocks<scalar_t>(domain);
	}));
}

template <typename scalar_t>
void _CopyPressureResultFromBlocks(std::shared_ptr<Domain> domain){
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	
	for(auto block : domain->blocks){
		copyBatched<scalar_t>(
			domain->pressureResult.data_ptr<scalar_t>() + block->globalOffset, envStride(domain->pressureResult, domain->getBatchSize()),
			block->pressure.data_ptr<scalar_t>(), envStride(block->pressure, domain->getBatchSize()),
			block->getStrides().w, domain->getBatchSize()
		); //dst, src, count, environments
	}
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("CopyPressureResultFromBlocks");
	
}
void CopyPressureResultFromBlocks(std::shared_ptr<Domain> domain){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CopyPressureResultFromBlocks", ([&] {
		_CopyPressureResultFromBlocks<scalar_t>(domain);
	}));
}


template <typename scalar_t>
void _CopyVelocityResultToBlocks(std::shared_ptr<Domain> domain){
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	
	for(auto block : domain->blocks){
		const index_t blockSizeFlat = block->getStrides().w;
		for(index_t dim=0; dim<domain->getSpatialDims(); ++dim){
			copyBatched<scalar_t>(
				block->velocity.data_ptr<scalar_t>() + blockSizeFlat*dim, envStride(block->velocity, domain->getBatchSize()),
				domain->velocityResult.data_ptr<scalar_t>() + block->globalOffset + domain->getTotalSize()*dim, envStride(domain->velocityResult, domain->getBatchSize()),
				blockSizeFlat, domain->getBatchSize()
			); //dst, src, count, environments
		}
	}
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("CopyVelocityResultToBlocks");
	
}
void CopyVelocityResultToBlocks(std::shared_ptr<Domain> domain){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CopyVelocityResultToBlocks", ([&] {
		_CopyVelocityResultToBlocks<scalar_t>(domain);
	}));
}
template <typename scalar_t>
void _CopyVelocityResultFromBlocks(std::shared_ptr<Domain> domain){
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	
	for(auto block : domain->blocks){
		const index_t blockSizeFlat = block->getStrides().w;
		for(index_t dim=0; dim<domain->getSpatialDims(); ++dim){
			copyBatched<scalar_t>(
				domain->velocityResult.data_ptr<scalar_t>() + block->globalOffset + domain->getTotalSize()*dim, envStride(domain->velocityResult, domain->getBatchSize()),
				block->velocity.data_ptr<scalar_t>() + blockSizeFlat*dim, envStride(block->velocity, domain->getBatchSize()),
				blockSizeFlat, domain->getBatchSize()
			); //dst, src, count, environments
		}
	}
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("CopyVelocityResultFromBlocks");
	
}
void CopyVelocityResultFromBlocks(std::shared_ptr<Domain> domain){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CopyVelocityResultFromBlocks", ([&] {
		_CopyVelocityResultFromBlocks<scalar_t>(domain);
	}));
}

#ifdef WITH_GRAD

template <typename scalar_t>
void _CopyScalarResultGradFromBlocks(std::shared_ptr<Domain> domain){
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	
	for(auto block : domain->blocks){
		const index_t blockSizeFlat = block->getStrides().w;
		for(index_t channel=0; channel<domain->getPassiveScalarChannels(); ++channel){
			copyBatched<scalar_t>(
				domain->scalarResult_grad.data_ptr<scalar_t>() + block->globalOffset + domain->getTotalSize()*channel, envStride(domain->scalarResult_grad, domain->getBatchSize()),
				block->passiveScalar_grad.data_ptr<scalar_t>() + blockSizeFlat*channel, envStride(block->passiveScalar_grad, domain->getBatchSize()),
				blockSizeFlat, domain->getBatchSize()
			); //dst, src, count, environments
		}
	}
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("CopyScalarResultGradFromBlocks");
	
}
void CopyScalarResultGradFromBlocks(std::shared_ptr<Domain> domain){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CopyScalarResultGradFromBlocks", ([&] {
		_CopyScalarResultGradFromBlocks<scalar_t>(domain);
	}));
}
template <typename scalar_t>
void _CopyScalarResultGradToBlocks(std::shared_ptr<Domain> domain){
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	
	for(auto block : domain->blocks){
		const index_t blockSizeFlat = block->getStrides().w;
		for(index_t channel=0; channel<domain->getPassiveScalarChannels(); ++channel){
			copyBatched<scalar_t>(
				block->passiveScalar_grad.data_ptr<scalar_t>() + blockSizeFlat*channel, envStride(block->passiveScalar_grad, domain->getBatchSize()),
				domain->scalarResult_grad.data_ptr<scalar_t>() + block->globalOffset + domain->getTotalSize()*channel, envStride(domain->scalarResult_grad, domain->getBatchSize()),
				blockSizeFlat, domain->getBatchSize()
			); //dst, src, count, environments
		}
	}
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("CopyScalarResultGradToBlocks");
	
}
void CopyScalarResultGradToBlocks(std::shared_ptr<Domain> domain){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CopyScalarResultGradToBlocks", ([&] {
		_CopyScalarResultGradToBlocks<scalar_t>(domain);
	}));
}
template <typename scalar_t>
void _CopyPressureResultGradFromBlocks(std::shared_ptr<Domain> domain){
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	
	for(auto block : domain->blocks){
		copyBatched<scalar_t>(
			domain->pressureResult_grad.data_ptr<scalar_t>() + block->globalOffset, envStride(domain->pressureResult_grad, domain->getBatchSize()),
			block->pressure_grad.data_ptr<scalar_t>(), envStride(block->pressure_grad, domain->getBatchSize()),
			block->getStrides().w, domain->getBatchSize()
		); //dst, src, count, environments
	}
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("CopyPressureResultGradFromBlocks");
	
}
void CopyPressureResultGradFromBlocks(std::shared_ptr<Domain> domain){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CopyPressureResultGradFromBlocks", ([&] {
		_CopyPressureResultGradFromBlocks<scalar_t>(domain);
	}));
}
template <typename scalar_t>
void _CopyVelocityResultGradFromBlocks(std::shared_ptr<Domain> domain){
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	
	for(auto block : domain->blocks){
		const index_t blockSizeFlat = block->getStrides().w;
		for(index_t dim=0; dim<domain->getSpatialDims(); ++dim){
			copyBatched<scalar_t>(
				domain->velocityResult_grad.data_ptr<scalar_t>() + block->globalOffset + domain->getTotalSize()*dim, envStride(domain->velocityResult_grad, domain->getBatchSize()),
				block->velocity_grad.data_ptr<scalar_t>() + blockSizeFlat*dim, envStride(block->velocity_grad, domain->getBatchSize()),
				blockSizeFlat, domain->getBatchSize()
			); //dst, src, count, environments
		}
	}
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("CopyVelocityResultGradFromBlocks");
	
}
void CopyVelocityResultGradFromBlocks(std::shared_ptr<Domain> domain){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CopyVelocityResultGradFromBlocks", ([&] {
		_CopyVelocityResultGradFromBlocks<scalar_t>(domain);
	}));
}
template <typename scalar_t>
void _CopyVelocityResultGradToBlocks(std::shared_ptr<Domain> domain){
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	
	for(auto block : domain->blocks){
		const index_t blockSizeFlat = block->getStrides().w;
		for(index_t dim=0; dim<domain->getSpatialDims(); ++dim){
			copyBatched<scalar_t>(
				block->velocity_grad.data_ptr<scalar_t>() + blockSizeFlat*dim, envStride(block->velocity_grad, domain->getBatchSize()),
				domain->velocityResult_grad.data_ptr<scalar_t>() + block->globalOffset + domain->getTotalSize()*dim, envStride(domain->velocityResult_grad, domain->getBatchSize()),
				blockSizeFlat, domain->getBatchSize()
			); //dst, src, count, environments
		}
	}
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("CopyVelocityResultGradToBlocks");
	
}
void CopyVelocityResultGradToBlocks(std::shared_ptr<Domain> domain){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CopyVelocityResultGradToBlocks", ([&] {
		_CopyVelocityResultGradToBlocks<scalar_t>(domain);
	}));
}

// ============================================================
// CopyEpotResultGradFromBlocks
// ============================================================

template <typename scalar_t>
void _CopyEpotResultGradFromBlocks(std::shared_ptr<Domain> domain){

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;

	for(auto block : domain->blocks){
		TORCH_CHECK(block->hasEpotGrad(), "Block does not have an epot_grad tensor. Call CreateEpotGradOnBlocks() first.");
		copyBatched<scalar_t>(
			domain->epotResult_grad.value().data_ptr<scalar_t>() + block->globalOffset, envStride(domain->epotResult_grad.value(), domain->getBatchSize()),
			block->epot_grad.value().data_ptr<scalar_t>(), envStride(block->epot_grad.value(), domain->getBatchSize()),
			block->getStrides().w, domain->getBatchSize()
		); //dst, src, count, environments
	}

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("CopyEpotResultGradFromBlocks");
}

void CopyEpotResultGradFromBlocks(std::shared_ptr<Domain> domain){

	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");

	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CopyEpotResultGradFromBlocks", ([&] {
		_CopyEpotResultGradFromBlocks<scalar_t>(domain);
	}));
}
#endif //WITH_GRAD

