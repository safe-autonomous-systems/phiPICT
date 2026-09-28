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
// Helpers to assemble CSR matrix rows.

#pragma once

#include "piso/device/nonortho_rhs.cuh"


__device__ inline int findLowestColumnIndex(int *indices, int size){
	int value = INT_MAX;
	int index = -1;
	for(int i=0;i<size;++i){
		if(indices[i]>=0 && indices[i]<value){
			value = indices[i];
			index = i;
		}
	}
	return index;
}

template<typename T>
__device__ index_t ArrayIndexOf(const T *array, const T value, const index_t size){
	for(int i=0;i<size;++i){
		if(array[i]==value){
			return i;
		}
	}
	return -1;
}

// load row from CSR matrix into csrValues (size must be at least spatialDims*2+1)
// loaded values will be sorted to be in [diag,-x,+x,-y,+y,-z,+z] order
template<typename scalar_t>
__device__
void LoadCSRrowNeighborSorted(const I4 &pos, const CSRmatrixGPU<scalar_t> &csr, const DomainGPU<scalar_t> &domain, const BlockGPU<scalar_t> &block, 
			scalar_t *csrValues){ // to be loaded like: diag,-x,+x,-y,+y,-z,+z
	
	const index_t flatPosGlobal = flattenIndexGlobal(pos, block, domain);
	
	const index_t csrStart = csr.row[flatPosGlobal];
	const index_t csrEnd = csr.row[flatPosGlobal+1];
	const index_t rowSize = csrEnd - csrStart;
	
	// row is sorted by global cell index, not neighbor direction
	index_t csrIndices[7] = {0};
	index_t i = 0;
	for(;i<rowSize && i<7;++i){
		csrIndices[i] = csr.index[csrStart+i];
	}
	for(; i<7;++i){
		csrIndices[i] = -1;
	}
	
	// undo global index sorting
	{ // diagonal entry
		const index_t aIndex = ArrayIndexOf(csrIndices, flatPosGlobal, 7);
		if(aIndex>=0){
			csrValues[0] = csr.value[csrStart+aIndex];
		}
	}
	// neighbor entries
	for(index_t bound=0; bound<(domain.numDims*2); ++bound){
		const index_t dim = axisFromBound(bound);
		const index_t isUpper = boundIsUpper(bound);
		const index_t faceSign = faceSignFromBound(bound);
		
		const bool atBound = isAtBound(pos, bound, block);
		if(!atBound || !isEmptyBound(bound, block.boundaries)){
			//calculate index of neighbour
			I4 tempPos = pos;
			tempPos.w = dim;
			const BlockGPU<scalar_t> *p_block = &block;
			// resolve neighbor cell
			if(atBound && block.boundaries[bound].type==BoundaryType::CONNECTED_GRID){
				p_block = domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
				tempPos = computeConnectedPosWithChannel(tempPos, dim, &block.boundaries[bound].cb, domain);
			}else{
				if(atBound && block.boundaries[bound].type==BoundaryType::PERIODIC){
					tempPos.a[dim] = isUpper ? 0 : block.size.a[dim]-1;
				}else{
					tempPos.a[dim] = pos.a[dim] + faceSign;
				}
			}
			tempPos.w = 0; // indices are "scalar"
			const index_t nIndex = flattenIndexGlobal(tempPos, p_block, domain); //flattenIndex(tempPos, p_block) + p_block->globalOffset;
			const index_t aIndex = ArrayIndexOf(csrIndices, nIndex, 7);
			
			if(aIndex>=0){
				csrValues[bound + 1] = csr.value[csrStart+aIndex];
			}
		} // else invalid neighbor
	}
}

