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
// Host helpers of the kernel launches.

#include "piso/device/common.cuh"

void CopyToGPU(void *p_dst, const void *p_src, const size_t bytes){
	//copy to GPU
	CUDA_CHECK_RETURN(cudaMemcpy(p_dst, p_src, bytes, cudaMemcpyHostToDevice)); //dst, src, bytes, kind
}
void CopyToGPUStrided(void *p_dst, const size_t dstPitch, const void *p_src, const size_t bytes, const size_t count){
	CUDA_CHECK_RETURN(cudaMemcpy2D(p_dst, dstPitch, p_src, bytes, bytes, count, cudaMemcpyHostToDevice)); //dst, dpitch, src, spitch, width, height, kind
}
void CopyDomainToGPU(const DomainAtlasSet &domainAtlas){
	//copy atlas to GPU
	CUDA_CHECK_RETURN(cudaMemcpy(domainAtlas.p_device, domainAtlas.p_host, domainAtlas.sizeBytes, cudaMemcpyHostToDevice)); //dst, src, bytes, kind
}

__host__ void ComputeThreadBlocks(std::shared_ptr<Domain> domain, int32_t &threads, dim3 &blocks, std::vector<index_t> &blockIdxByThreadBlock, std::vector<index_t> &threadBlockOffsetInBlock) {
	index_t numThreadBlocks = 0;
	const index_t maxBlockSize = domain->getMaxBlockSize();
	const index_t threadBlockSize = maxBlockSize>=MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : ALIGN_UP(maxBlockSize, WARP_SIZE);
	index_t blockIdx = 0;
	blockIdxByThreadBlock.clear();
	threadBlockOffsetInBlock.clear();
	for(std::shared_ptr<const Block> block : domain->blocks){
		const index_t blockSize = block->getStrides().w;
		const index_t threadBlocksPerBlock = divCeil(blockSize, threadBlockSize);
		//py::print("block", blockIdx, ", size", blockSize, ", blocks", threadBlocksPerBlock);
		numThreadBlocks += threadBlocksPerBlock;
		for(index_t i=0; i<threadBlocksPerBlock; ++i){
			blockIdxByThreadBlock.push_back(blockIdx);
			threadBlockOffsetInBlock.push_back(i);
			//py::print("push", blockIdx, i);
		}
		++blockIdx;
	}

	threads = threadBlockSize;
	// batched environments: grid row blockIdx.y processes environment blockIdx.y
	blocks = dim3(numThreadBlocks, domain->getBatchSize());
}

__host__ torch::Tensor CopyBlockIndices(std::shared_ptr<Domain> domain,
		std::vector<index_t> &blockIdxByThreadBlock, std::vector<index_t> &threadBlockOffsetInBlock,
		index_t *&p_blockIdxByThreadBlock, index_t *&p_threadBlockOffsetInBlock){
	auto byteOptions = torch::TensorOptions().dtype(torch_kIndex).layout(torch::kStrided).device(domain->getDevice().type(), domain->getDevice().index());
	torch::Tensor allocTensor = torch::zeros(blockIdxByThreadBlock.size() + threadBlockOffsetInBlock.size(), byteOptions);
	p_blockIdxByThreadBlock = allocTensor.data_ptr<index_t>();
	p_threadBlockOffsetInBlock = p_blockIdxByThreadBlock + blockIdxByThreadBlock.size();
	CUDA_CHECK_RETURN(cudaMemcpy(p_blockIdxByThreadBlock, blockIdxByThreadBlock.data(), blockIdxByThreadBlock.size()*sizeof(index_t), cudaMemcpyHostToDevice)); //dst, src, bytes, kind
	CUDA_CHECK_RETURN(cudaMemcpy(p_threadBlockOffsetInBlock, threadBlockOffsetInBlock.data(), threadBlockOffsetInBlock.size()*sizeof(index_t), cudaMemcpyHostToDevice)); //dst, src, bytes, kind
	return allocTensor;
}
