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
// Transformation of vector fields with the grid transforms.

#include "piso/device/common.cuh"

/* move to grid_gen.cu?*/

struct GridInfo{
	I4 size;
	I4 stride;
};

__host__ inline GridInfo MakeGridInfo(const index_t sizeX=1, const index_t sizeY=1, const index_t sizeZ=1, const index_t channels=1){
	GridInfo grid;
	memset(&grid, 0, sizeof(GridInfo));
	grid.size.x = sizeX;
	grid.size.y = sizeY;
	grid.size.z = sizeZ;
	grid.size.w = channels;
	grid.stride.x = 1;
	grid.stride.y = sizeX;
	grid.stride.z = sizeX*sizeY;
	grid.stride.w = sizeX*sizeY*sizeZ;
	return grid;
}

template<typename scalar_t, int DIMS>
__global__ void k_TransformVectors(const scalar_t *p_vectors, const GridInfo gridInfo, TransformGPU<scalar_t,DIMS> *p_transforms, scalar_t *p_vectorsTransformed, const bool inverse,
		const bool batchedTransforms){
	// batch: blockIdx.y; the transforms are shared unless batchedTransforms
	const index_t batchOffset = blockIdx.y * gridInfo.stride.w * DIMS;
	p_vectors += batchOffset;
	p_vectorsTransformed += batchOffset;
	if(batchedTransforms) p_transforms += blockIdx.y * gridInfo.stride.w;
    for(index_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < gridInfo.stride.w; flatIdx += blockDim.x * gridDim.x){
        const I4 pos = unflattenIndex(flatIdx, gridInfo.size, gridInfo.stride);
        TransformGPU<scalar_t,DIMS> *p_T = p_transforms + flatIdx;

		MatrixSquare<scalar_t,DIMS> t = inverse ? p_T->Minv : p_T->M;

		Vector<scalar_t,DIMS> v;
		for(index_t dim=0; dim<DIMS; ++dim){
			I4 tempPos = pos;
			tempPos.w = dim;
			v.a[dim] = p_vectors[flattenIndex(tempPos, gridInfo.stride)];
		}

		v = matmul(t, v);

		for(index_t dim=0; dim<DIMS; ++dim){
			I4 tempPos = pos;
			tempPos.w = dim;
			p_vectorsTransformed[flattenIndex(tempPos, gridInfo.stride)] = v.a[dim];
		}
		
	}
}

#ifdef DISPATCH_FTYPES_DIMS
#undef DISPATCH_FTYPES_DIMS
#endif
#define DISPATCH_FTYPES_DIMS(FTYPE, DIMS, NAME, ...) \
	AT_DISPATCH_FLOATING_TYPES(FTYPE, NAME, ([&] { \
		SWITCH_DIMS(DIMS, \
			__VA_ARGS__; \
		); \
	}));

torch::Tensor TransformVectors(const torch::Tensor &vectors, const torch::Tensor &transforms, const bool inverse){
	// vectors: NCDHW
	// transforms: NDHWT
    CHECK_INPUT_CUDA(vectors);
	TORCH_CHECK(2<vectors.dim() && vectors.dim()<6, "vectors must have batch and channel dimension and be 1-3D.");
	TORCH_CHECK(vectors.size(0)>=1, "vectors batch dimension must be at least 1.");
	index_t dims = vectors.dim()-2;
	TORCH_CHECK(vectors.size(1)==dims, "vectors channel dimension must match spatial dimensionality.");
	
    CHECK_INPUT_CUDA(transforms);
	TORCH_CHECK(transforms.dim() == vectors.dim(), "dimensionality of transforms must match vectors.");
	TORCH_CHECK(transforms.size(0)==1 || transforms.size(0)==vectors.size(0), "transforms batch dimension must be 1 or match the vectors.");
	TORCH_CHECK(transforms.size(-1)==TransformNumValues(dims), "transforms channels must match spatial dimensions.");
	for(index_t dim=1; dim<dims+1; ++dim){
		TORCH_CHECK(transforms.size(dim) == vectors.size(dim+1), "spatial dimensions of transforms must match vectors.");
	}
	
	
	const GridInfo gridInfo = MakeGridInfo(vectors.size(-1), dims>1?vectors.size(-2):1, dims>2?vectors.size(-3):1, vectors.size(1));
	
    auto valueOptions = torch::TensorOptions().dtype(vectors.scalar_type()).layout(torch::kStrided).device(vectors.device().type(), vectors.device().index());
	
    torch::Tensor transformedVectors = torch::zeros_like(vectors);
    //const index_t totalSize = sizeX*sizeY; ==grid.strides.w
    
	DISPATCH_FTYPES_DIMS(vectors.scalar_type(), dims, "TransformVectors",
		int minGridSize = 0, blockSize = 0, gridSize = 0;
	    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_TransformVectors<scalar_t, dim>, 0, 0);
	    gridSize = (gridInfo.stride.w + blockSize - 1) / blockSize;
		k_TransformVectors<scalar_t, dim><<<dim3(gridSize, vectors.size(0)), blockSize>>>(
			vectors.contiguous().data_ptr<scalar_t>(), gridInfo, reinterpret_cast<TransformGPU<scalar_t,dim>*>(transforms.data_ptr<scalar_t>()), transformedVectors.data_ptr<scalar_t>(), inverse,
			transforms.size(0)>1
		);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	);
	
	return transformedVectors;
}

