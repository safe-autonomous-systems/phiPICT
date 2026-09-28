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
// CSR matrix container.

#include "domain/domain_internal.h"

CSRmatrix::CSRmatrix(torch::Tensor &a_value, torch::Tensor &a_index, torch::Tensor &a_row) : value(a_value), index(a_index), row(a_row) {
	TORCH_CHECK(value.dim()==1, "value must be 1D.");
	TORCH_CHECK(index.dim()==1, "index must be 1D.");
	TORCH_CHECK(row.dim()==1, "row must be 1D.");
	// batched environments: value may hold several matrices with the same pattern
	TORCH_CHECK(index.size(0)>0 ? value.size(0)%index.size(0)==0 : value.size(0)==0,
		"value size must be a multiple of the index size (one or more matrices with the same pattern).");
	TORCH_CHECK(index.dtype()==torch_kIndex, "index must have integer type.");
	//TORCH_CHECK(index.scalar_type()==torch_kIndex, "index must have integer scalar_type."); //both scalar_type() and dtype() work here
	TORCH_CHECK(row.dtype()==torch_kIndex, "row must have integer type."); 
	TORCH_CHECK(index.device()==value.device(), "all tensors must be on the same device.");
	TORCH_CHECK(row.device()==value.device(), "all tensors must be on the same device.");
}

CSRmatrix::CSRmatrix(const int32_t numValues, const int32_t numRows, const torch::Dtype valueType, const torch::Device device){
	auto valueOptions = torch::TensorOptions().dtype(valueType).layout(torch::kStrided).device(device.type(), device.index());
	auto indexOptions = torch::TensorOptions().dtype(torch_kIndex).layout(torch::kStrided).device(device.type(), device.index());

	value = torch::zeros(numValues, valueOptions);
	index = torch::zeros(numValues, indexOptions);
	row = torch::zeros(numRows+1, indexOptions);
}

void CSRmatrix::Detach(){
	value = value.detach();
}

void CSRmatrix::CreateValue(){
	value = torch::zeros_like(value);
	isTensorChanged = true;
}

void CSRmatrix::setValue(torch::Tensor &tensor){
	TORCH_CHECK(tensor.dim()==1, "value must be 1D.");
	TORCH_CHECK(value.size(0)==tensor.size(0), "value must have the correct size.");
	TORCH_CHECK(value.dtype()==tensor.dtype(), "value must have the correct type.");
	TORCH_CHECK(value.device()==tensor.device(), "all tensors must be on the same device.");
	value = tensor;
	isTensorChanged = true;
}

std::shared_ptr<CSRmatrix> CSRmatrix::Copy() const {
	torch::Tensor v = value;
	torch::Tensor i = index;
	torch::Tensor r = row;
	return std::make_shared<CSRmatrix>(v, i, r);
}
std::shared_ptr<CSRmatrix> CSRmatrix::Clone() const {
	torch::Tensor t_value = value.clone();
	torch::Tensor t_index = index.clone();
	torch::Tensor t_row = row.clone();
	
	return std::make_shared<CSRmatrix>(t_value, t_index, t_row);
}
std::shared_ptr<CSRmatrix> CSRmatrix::WithZeroValue() const {
	torch::Tensor v = torch::zeros_like(value);
	torch::Tensor i = index;
	torch::Tensor r = row;
	return std::make_shared<CSRmatrix>(v, i, r);
}

std::shared_ptr<CSRmatrix> CSRmatrix::toType(const torch::Dtype type) const {
	torch::Tensor t_value = value.toType(type);
	torch::Tensor t_index = index.clone();
	torch::Tensor t_row = row.clone();
	
	return std::make_shared<CSRmatrix>(t_value, t_index, t_row);
}

std::string CSRmatrix::ToString() const {
	std::ostringstream repr;
	repr << "CSRmatrix( size=" << getSize() << ", rows=" << getRows() << " )";
	return repr.str();
	
}

