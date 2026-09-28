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
// Launch configuration, index arithmetic and the per-cell kernel loop.

#pragma once

#include "api.h"
#include "solvers/linear_solvers.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "common/logging.h"


const size_t WARP_SIZE = 32;
const size_t MAX_BLOCK_SIZE = 512; //1024;


static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
  if (err == cudaSuccess) return;
  std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
            << err << ") at " << file << ":" << line << std::endl;
  exit(10);
}
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)

//const int WARP_SIZE = 32;

__device__ constexpr
int divCeil(const int a, const int b){
	return (a + b - 1)/b;
}

template<typename I, typename M>
__device__ constexpr
I posMod(const I a, const M m){
	I mod = a % m;
	if(mod<0){
		mod += m;
	}
	return mod;
}




__device__ inline index_t GetBlockIdxForThreadBlock(const index_t threadBlockIdx){
	//TODO
	//blockIdxByThreadBlock[threadBlockIdx];
	return 0;
}


#define SWITCH_DIMS_CASE(DIM, ...) \
	case DIM: { \
		const index_t dim = DIM; \
		__VA_ARGS__; \
		break; \
	}

#define SWITCH_DIMS_SWITCH(DIMS, ...) \
	switch(DIMS) { \
		__VA_ARGS__ \
		default: \
			TORCH_CHECK(false, "Unknown dimension."); \
	}

#define SWITCH_DIMS(DIM, ...) \
	SWITCH_DIMS_SWITCH(DIM, SWITCH_DIMS_CASE(1, __VA_ARGS__) SWITCH_DIMS_CASE(2, __VA_ARGS__) SWITCH_DIMS_CASE(3, __VA_ARGS__))

#define DISPATCH_FTYPES_DIMS(DOMAIN, NAME, ...) \
	AT_DISPATCH_FLOATING_TYPES(DOMAIN->getDtype(), NAME, ([&] { \
		SWITCH_DIMS(DOMAIN->getSpatialDims(), \
			__VA_ARGS__; \
		); \
	}));

template <typename scalar_t>
__device__ constexpr
S4<scalar_t> makeS4GPU(const scalar_t x = 0, const scalar_t y = 0, const scalar_t z = 0, const scalar_t w = 0){
	//return {{.x=x, .y=y, .z=z, .w=w}};
	return {{x, y, z, w}};
}
const auto makeI4GPU = makeS4GPU<int32_t>;
// const auto makeF4GPU = makeS4GPU<float>;
// const auto makeD4GPU = makeS4GPU<double>;

__device__ inline
index_t flattenIndex(const I4 &pos, const I4 &stride){
	return pos.x + stride.y*pos.y + stride.z*pos.z + stride.w*pos.w;
}

template<typename scalar_t>
__device__ inline
index_t flattenIndex(const I4 &pos, const BlockGPU<scalar_t> &block){
	//return pos.x + block.stride.y*pos.y + block.stride.z*pos.z + block.stride.w*pos.w;
	return flattenIndex(pos, block.stride);
}

template<typename scalar_t>
__device__ inline
index_t flattenIndex(const I4 &pos, const BlockGPU<scalar_t> *block){
	//return pos.x + block->stride.y*pos.y + block->stride.z*pos.z + block->stride.w*pos.w;
	return flattenIndex(pos, block->stride);
}

__device__ inline
I4 unflattenIndex(const index_t idx, const I4 &size, const I4 &stride){
	return makeI4GPU(idx%size.x, (idx/stride.y)%size.y, (idx/stride.z)%size.z, (idx/stride.w)%size.w);
} 

template<typename scalar_t>
__device__ inline
I4 unflattenIndex(const index_t idx, const BlockGPU<scalar_t> &block){
	//return makeI4GPU(idx%block.size.x, (idx/block.stride.y)%block.size.y, (idx/block.stride.z)%block.size.z, (idx/block.stride.w)%block.size.w);
	return unflattenIndex(idx, block.size, block.stride);
} 

template<typename scalar_t>
__device__ inline
index_t flattenIndexGlobal(const I4 &pos, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain){
	const index_t dim = pos.w;
	I4 tempPos = pos;
	tempPos.w = 0;
	return flattenIndex(tempPos, block.stride) + block.globalOffset + domain.numCells*dim;
}

template<typename scalar_t>
__device__ inline
index_t flattenIndexGlobal(const I4 &pos, const BlockGPU<scalar_t> *block, const DomainGPU<scalar_t> &domain){
	const index_t dim = pos.w;
	I4 tempPos = pos;
	tempPos.w = 0;
	return flattenIndex(tempPos, block->stride) + block->globalOffset + domain.numCells*dim;
}

/* Host helpers of the kernel launches (piso/launch.cu). */
void CopyToGPU(void *p_dst, const void *p_src, const size_t bytes);
void CopyDomainToGPU(const DomainAtlasSet &domainAtlas);
__host__ void ComputeThreadBlocks(std::shared_ptr<Domain> domain, int32_t &threads, dim3 &blocks, std::vector<index_t> &blockIdxByThreadBlock, std::vector<index_t> &threadBlockOffsetInBlock);
__host__ torch::Tensor CopyBlockIndices(std::shared_ptr<Domain> domain,
		std::vector<index_t> &blockIdxByThreadBlock, std::vector<index_t> &threadBlockOffsetInBlock,
		index_t *&p_blockIdxByThreadBlock, index_t *&p_threadBlockOffsetInBlock);

#define SETUP_KERNEL_PER_CELL(domain, idxName, offName) \
	int32_t threads; \
	dim3 blocks; \
	index_t *p_##idxName; \
	index_t *p_##offName; \
	std::vector<index_t> idxName; \
	std::vector<index_t> offName; \
	ComputeThreadBlocks(domain, threads, blocks, idxName, offName); \
	torch::Tensor t_blockIdxByThreadBlock = CopyBlockIndices(domain, idxName, offName, p_##idxName, p_##offName);

#define KERNEL_PER_CELL_LOOP(p_domain, p_idx, p_off, num, ...) \
	__shared__ DomainGPU<scalar_t> s_domain; \
	__shared__ BlockGPU<scalar_t> s_block; \
	if(threadIdx.x==0){ s_domain = p_domain[blockIdx.y]; /* batched environments: one domain per grid row */ } \
	__syncthreads(); \
	index_t repetitions = num/gridDim.x; \
	if(blockIdx.x<(num - gridDim.x*repetitions) ){ ++repetitions; } \
	index_t loadedBlockIdx = -1; \
	for(index_t r=0;r<repetitions;++r){ \
		const index_t currentThreadBlockIdx = gridDim.x*r + blockIdx.x; \
		const index_t targetBlockIdx = p_idx[currentThreadBlockIdx];  \
		const index_t threadBlockOffsetInBlock = p_off[currentThreadBlockIdx];  \
		if(threadIdx.x==0 && targetBlockIdx!=loadedBlockIdx){ \
			s_block = s_domain.blocks[targetBlockIdx]; \
			loadedBlockIdx = targetBlockIdx; \
		} \
		__syncthreads(); \
		const index_t flatPos = threadBlockOffsetInBlock*blockDim.x + threadIdx.x; \
		if(flatPos<s_block.stride.w){ \
			__VA_ARGS__ \
	}}
