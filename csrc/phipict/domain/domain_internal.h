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
// Tensor helpers shared by the domain sources.

#pragma once

#include "domain/domain_structs.h"
#include "api.h"
#include "grid/grid_gen.h"

#include <type_traits>
#include <memory>


/*
inline bool CompareDevice(const torch::Device &device, const torch::Device &otherDevice){
	return device.type()==otherDevice.type() && device.index()==otherDevice.index();
}*/

inline std::string BoundaryTypeToString(const BoundaryType bt){
	switch (bt)
	{
	case BoundaryType::DIRICHLET:
		return "DIRICHLET";
	case BoundaryType::DIRICHLET_VARYING:
		return "DIRICHLET_VARYING";
	case BoundaryType::NEUMANN:
		return "NEUMANN";
	case BoundaryType::CONNECTED_GRID:
		return "CONNECTED";
	case BoundaryType::PERIODIC:
		return "PERIODIC";
	case BoundaryType::FIXED:
		return "FIXED";
	default:
		return "UNDEFINED";
	}
}

inline bool IsTensorEmpty(const torch::Tensor &tensor){
	return tensor.dim()==1 && tensor.size(0)==0;
}
template <typename scalar_t>
scalar_t* getTensorDataPtr(const torch::Tensor &tensor){
	return IsTensorEmpty(tensor) ? nullptr : tensor.data_ptr<scalar_t>();
}
template <typename scalar_t>
scalar_t* getOptionalTensorDataPtr(const optional<torch::Tensor> &tensor){
	return tensor.has_value() ? tensor.value().data_ptr<scalar_t>() : nullptr;
}

/* to clone optional tensors */
inline optional<torch::Tensor> cloneOptionalTensor(const optional<torch::Tensor> &t){
	if(t) {
		return t.value().clone();
	} else {
		return nullopt;
	}
}

inline std::vector<int64_t> GetTensorShape(const torch::Tensor &tensor){
	std::vector<int64_t> shape(tensor.dim(),1);
	for(index_t dim=0;dim<tensor.dim();++dim){
		shape[dim] = tensor.size(dim);
	}
	return shape;
};
inline bool CheckTensor(const torch::Tensor &tensor, const index_t channels, const torch::Tensor &refTensor, const std::string &name){
	CHECK_INPUT_CUDA(tensor);
	const index_t numDims = refTensor.dim();
	TORCH_CHECK(tensor.dim()==numDims, "Dimensions of " + name + " must be " + std::to_string(numDims) + ".");
	TORCH_CHECK(tensor.size(0)==refTensor.size(0) || tensor.size(0)==1, "Batch size (dim 0) of " + name + " must be 1 or " + std::to_string(refTensor.size(0)) + ".");
	TORCH_CHECK(tensor.size(1)==channels, "Channels (dim 1) of " + name + " must be " + std::to_string(channels) + ".");
	for(int dim=2;dim<numDims;++dim){
		TORCH_CHECK(refTensor.size(dim)==tensor.size(dim), "Spatial dimension " + std::to_string(dim) + " fo " + name + " must match (" + std::to_string(refTensor.size(dim)) + ").");
	}
	TORCH_CHECK(tensor.dtype()==refTensor.dtype(), name + "has wrong dtype.");
	return true;
}

struct TensorInfo{
	index_t dims;
	index_t spatialDims;
	I4 size;
	index_t batchSize;
	index_t channels;
	torch::Dtype dtype;
	optional<torch::Device> device=nullopt;
};

inline TensorInfo getFieldInfo(torch::Tensor tensor, const bool isStaggered){
	CHECK_INPUT_CUDA(tensor);
	TensorInfo info = {.dtype=tensor.scalar_type(), .device=tensor.device()};
	
	TORCH_CHECK(3<=tensor.dim() && tensor.dim()<=5, "Fields must be 1D, 2D, or 3D in NCDHW layout.");
	info.dims = tensor.dim();
	info.spatialDims = info.dims-2;
	
	info.batchSize = tensor.size(0);
	info.channels = tensor.size(1);
	
	index_t dim=0;
	for(; dim<info.spatialDims; ++dim){
		info.size.a[dim] = tensor.size(info.dims - 1 - dim);
		if(isStaggered) { info.size.a[dim] -= 1; }
		TORCH_CHECK(info.size.a[dim]>2, "all spatial dimensions must be at least 3.");
	}
	for(; dim<3; ++dim){
		info.size.a[dim] = 1;
	}
	info.size.w = info.spatialDims;
	
	return info;
}

inline torch::Tensor CreateTensor(const index_t batches, const index_t channels, const I4 size, const index_t dims, const torch::TensorOptions options){
	std::vector<int64_t> shape;
	shape.push_back(batches);
	shape.push_back(channels);
	for(index_t i=dims-1; 0<=i; --i){
		shape.push_back(size.a[i]);
	}
	return torch::zeros(shape, options);
}
inline torch::Tensor CreateTensorFromRef(const index_t batches, const index_t channels, const torch::Tensor &refTensor){
	std::vector<int64_t> shape = GetTensorShape(refTensor);
	shape[0] = batches;
	shape[1] = channels;
	auto options = torch::TensorOptions().dtype(refTensor.scalar_type()).layout(torch::kStrided).device(refTensor.device().type(), refTensor.device().index());
	return torch::zeros(shape, options);
}

inline index_t AxisToIndex(const std::string &side){
	if(side.compare("x")==0)  return 0;
	else if(side.compare("y")==0) return 1;
	else if(side.compare("z")==0) return 2;
	return -1;
}

inline index_t BoundarySideToIndex(const std::string &side){
	if(side.compare("-x")==0)  return 0;
	else if(side.compare("+x")==0) return 1;
	else if(side.compare("-y")==0) return 2;
	else if(side.compare("+y")==0) return 3;
	else if(side.compare("-z")==0) return 4;
	else if(side.compare("+z")==0) return 5;
	return -1;
}

inline dim_t BoundaryIndexToDim(const dim_t index){
	//return static_cast<dim_t>((static_cast<uint32_t>(index)>>1) & 3);
	return index>>1;
}
inline dim_t BoundaryDimToIndex(const dim_t index){
	//return static_cast<dim_t>((static_cast<uint32_t>(index)<<1));
	return index<<1;
}
inline dim_t BoundaryIndexToDirection(const dim_t index){
	return index & 1;
}

inline std::string BoundaryIndexToString(const index_t index){
	switch (index)
	{
	case 0:
		return "-x";
	case 1:
		return "+x";
	case 2:
		return "-y";
	case 3:
		return "+y";
	case 4:
		return "-z";
	case 5:
		return "+z";
	default:
		return "?";
	}
}



