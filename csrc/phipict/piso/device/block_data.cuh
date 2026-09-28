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
// Block/boundary data access, neighbour and corner values, data gradients.

#pragma once

#include "piso/device/nonortho_laplace.cuh"

/** Returns the start pointer of the requested data tensor. */
template<typename scalar_t>
__device__
scalar_t *getBlockDataPtr(const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const GridDataType type){
	if(type==GridDataType::IS_FIXED_BOUNDARY) { return nullptr; }
	
	const index_t dataIndex = gridDataTypeToIndex(type);
	const bool isGrad = isGradDataType(type);
	const bool isGlobal = isGlobalDataType(type);
	const bool isResult = isResultDataType(type);
	const bool isRHS = isRHSDataType(type);
	
	if(!isGrad){
		if(!isGlobal){ // block data
			return p_block->data[dataIndex];
		}
		if(isResult){
			return domain.results[dataIndex];
		}
		if(isRHS){
			return domain.RHS[dataIndex];
		}
	}
#ifdef WITH_GRAD
	else {
		if(!isGlobal){ // block data
			return p_block->grad[dataIndex];
		}
		if(isResult){
			return domain.results_grad[dataIndex];
		}
		if(isRHS){
			return domain.RHS_grad[dataIndex];
		}
	}
#endif //WITH_GRAD
	
	return nullptr;
}

/** Returns the pointer to the data of 'type' of cell 'pos' in a block. */
template<typename scalar_t>
__device__
scalar_t *getBlockCellDataPtr(I4 pos, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const GridDataType type){
	scalar_t* p_data = getBlockDataPtr<scalar_t>(p_block, domain, type);
	if(p_data==nullptr){ return nullptr;}
	
	//if(isScalarDataType(type)) { pos.w = 0; } // passive scalar can have multiple channels now
	const bool isGlobal = isGlobalDataType(type);
	const index_t flatPos = isGlobal ? flattenIndexGlobal(pos, p_block, domain) : flattenIndex(pos, p_block);
	
	return p_data + flatPos;
}


template<typename scalar_t>
__device__
scalar_t getBlockData(const I4 pos, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const GridDataType type){
	
	const scalar_t* p_data = getBlockCellDataPtr<scalar_t>(pos, p_block, domain, type);
	
	if(p_data){ 
		return *p_data;
	} else {
		return 0;
	}
}

template<typename scalar_t>
__device__
void writeBlockData(const scalar_t data, I4 pos, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const GridDataType type){
	scalar_t* p_data = getBlockCellDataPtr<scalar_t>(pos, p_block, domain, type);
	
	if(p_data){ *p_data = data; }
}

#ifdef WITH_GRAD
template<typename scalar_t>
__device__
void scatterBlockData(const scalar_t data, I4 pos, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const GridDataType type){
	scalar_t* p_data = getBlockCellDataPtr<scalar_t>(pos, p_block, domain, type);
	
	if(p_data){ atomicAdd(p_data, data); }
}
#endif //WITH_GRAD

template<typename scalar_t>
__device__
BoundaryConditionType getFixedBoundaryType(const I4 &pos, const index_t bound, const BlockGPU<scalar_t> *p_block, const GridDataType type){ //, const DomainGPU<scalar_t> &domain
	switch(p_block->boundaries[bound].type){
		case BoundaryType::VALUE:
		case BoundaryType::DIRICHLET_VARYING:
			return BoundaryConditionType::DIRICHLET;
		case BoundaryType::GRADIENT:
			return BoundaryConditionType::NEUMANN;
		case BoundaryType::FIXED:
		{
			const index_t typeIndex = gridDataTypeToIndex(type);
			const FixedBoundaryGPU<scalar_t> *p_bound = &(p_block->boundaries[bound].fb);
			const FixedBoundaryDataGPU<scalar_t> *p_data = &(p_bound->data[typeIndex]);
			if(p_data->isStaticType){
				return p_data->boundaryType;
			} else {
				return p_data->p_boundaryTypes[pos.w];
			}
			break;
		}
		default:
			return BoundaryConditionType::DIRICHLET;
	}

}
template<typename scalar_t>
__device__
scalar_t getFixedBoundaryData(const I4 &pos, const index_t bound, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const GridDataType type){
	
	if(type==GridDataType::IS_FIXED_BOUNDARY) { return 1; }
	if(isGradDataType(type)) { return 0; } // TODO
	
	const GridDataType baseType = gridDataTypeToBaseType(type);
	
	switch(p_block->boundaries[bound].type){
		case BoundaryType::VALUE:
		{
			const StaticDirichletBoundaryGPU<scalar_t> *p_bound = &(p_block->boundaries[bound].sdb);
			switch(baseType){
				case GridDataType::VELOCITY:
					return p_bound->velocity.a[pos.w];
				case GridDataType::PRESSURE:
					return p_block->pressure[flattenIndex(pos, p_block)]; // assumes 0 pressure gradient at boundary
				case GridDataType::PASSIVE_SCALAR:
					return p_bound->scalar;
				default:
					return 0;
			}
		}
		case BoundaryType::DIRICHLET_VARYING:
		{
			const index_t dim = axisFromBound(bound);
			I4 boundPos = pos;
			if(isScalarDataType(type)) { boundPos.w = 0; }
			boundPos.a[dim] = 0;
			
			const VaryingDirichletBoundaryGPU<scalar_t> *p_bound = &(p_block->boundaries[bound].vdb);
			const index_t flatBoundPos = flattenIndex(boundPos, p_bound->stride);
			switch(baseType){
				case GridDataType::VELOCITY:
					return p_bound->velocity[flatBoundPos];
				case GridDataType::PRESSURE:
					return p_block->pressure[flattenIndex(pos, p_block)]; // assumes 0 pressure gradient at boundary
				case GridDataType::PASSIVE_SCALAR:
					return p_bound->scalar[flatBoundPos];
				default:
					return 0;
			}
		}
		case BoundaryType::GRADIENT: //TODO: how to handle this case here?
			// not implemented
			return 0;
		case BoundaryType::FIXED:
		{
			const index_t typeIndex = gridDataTypeToIndex(type);
			const FixedBoundaryGPU<scalar_t> *p_bound = &(p_block->boundaries[bound].fb);
			const FixedBoundaryDataGPU<scalar_t> *p_data = &(p_bound->data[typeIndex]);
			if(p_data->isStatic){
				//return (isScalarDataType(type) ? p_data->data[0] : p_data->data[pos.w]);
				return p_data->data[pos.w];
			} else {
				const index_t dim = axisFromBound(bound);
				I4 boundPos = pos;
				//if(isScalarDataType(type)) { boundPos.w = 0; } // passive scalar can have multiple channels now
				boundPos.a[dim] = 0;
				const index_t flatBoundPos = flattenIndex(boundPos, p_bound->stride);
				
				return p_data->data[flatBoundPos];
			}
			break;
		}
		default:
			return 0;
	}
}

#ifdef WITH_GRAD
template<typename scalar_t>
__device__
void scatterFixedBoundaryData(const scalar_t data, const I4 &pos, const index_t bound, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const GridDataType type){
	
	if(type==GridDataType::IS_FIXED_BOUNDARY || !(p_block->boundaries[bound].type==BoundaryType::FIXED)) {
		return;
	}
	
	const index_t typeIndex = gridDataTypeToIndex(type);
	const FixedBoundaryGPU<scalar_t> *p_bound = &(p_block->boundaries[bound].fb);
	const FixedBoundaryDataGPU<scalar_t> *p_dataType = &(p_bound->data[typeIndex]);
	
	index_t offset = 0;
	if(p_dataType->isStatic){
		//offset = isScalarDataType(type) ? 0 : pos.w; // passive scalar can have multiple channels now
		offset = pos.w;
	} else {
		const index_t dim = axisFromBound(bound);
		I4 boundPos = pos;
		//if(isScalarDataType(type)) { boundPos.w = 0; }
		boundPos.a[dim] = 0;
		offset = flattenIndex(boundPos, p_bound->stride);
		
	}
	
	scalar_t *p_data = isGradDataType(type) ? p_dataType->grad : p_dataType->data;
	atomicAdd(p_data + offset, data);
}

#endif //WITH_GRAD

/** 
  * If the requested data type is static the spatial position is ignored.
  */
template<typename scalar_t>
__device__
scalar_t *getFixedBoundaryCellDataPtr(I4 pos, const index_t bound, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const GridDataType type){
	
	if(type==GridDataType::IS_FIXED_BOUNDARY) { return nullptr; }
	if(!(p_block->boundaries[bound].type==BoundaryType::FIXED)){ return nullptr; }
	
	/*if(isScalarDataType(type)){ // passive scalar can have multiple channels now
		pos.w = 0;
	}*/
	
	const bool isGrad = isGradDataType(type);
	const index_t typeIndex = gridDataTypeToIndex(type);
	const FixedBoundaryGPU<scalar_t> *p_bound = &(p_block->boundaries[bound].fb);
	const FixedBoundaryDataGPU<scalar_t> *p_dataType = &(p_bound->data[typeIndex]);
	
	index_t offset = pos.w;
	if(!(p_dataType->isStatic)){
		const index_t dim = axisFromBound(bound);
		pos.a[dim] = 0;
		offset = flattenIndex(pos, p_bound->stride);
	}
	
	if(!isGrad){
		return p_dataType->data + offset;
	}
#ifdef WITH_GRAD
	else {
		return p_dataType->grad + offset;
	}
#endif //WITH
	
	return nullptr;
}

template<typename scalar_t>
__device__ scalar_t getBlockDataNeighbor(const I4 &pos, const index_t dir, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain,
		const GridDataType type, const bool zeroBound){
	/* pos: original position to start from
	dir: in [0,dim*2] as [-x,+x,-y,..,+z]
	*/
	
	const index_t dim = axisFromBound(dir);
	const bool isUpper = boundIsUpper(dir);
	const index_t faceSign = faceSignFromBound(dir);
	
	const BlockGPU<scalar_t> *p_block = &block;
	I4 tempPos = pos;
	
	if(isUpper ? pos.a[dim]==(block.size.a[dim]-1) : pos.a[dim]==0){ //check if there is a boundary in the direction we want to move
		
		switch(block.boundaries[dir].type){
			case BoundaryType::VALUE:
			case BoundaryType::DIRICHLET_VARYING:
			case BoundaryType::FIXED:
			case BoundaryType::GRADIENT:
				return zeroBound ? 0 : getFixedBoundaryData(tempPos, dir, p_block, domain, type);
			case BoundaryType::CONNECTED_GRID:
			{
				//handle multi-block grids, load from correct cell of the connected grid
				const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + p_block->boundaries[dir].cb.connectedGridIndex;
				tempPos = computeConnectedPos<scalar_t>(tempPos, dim, &(p_block->boundaries[dir].cb), domain, 1);
				p_block = p_connectedBlock;
				break;
			}
			case BoundaryType::PERIODIC:
				// compute flux to cell on other side
				// special case of connection to another block
				tempPos.a[dim] = isUpper ? 0 : p_block->size.a[dim]-1;
				break;
			default:
				return 0;
		}
	} else {
		//same block, just update position
		tempPos.a[dim] += faceSign;
	}
	
	return getBlockData(tempPos, p_block, domain, type);
}

#ifdef WITH_GRAD
template<typename scalar_t>
__device__ void scatterBlockDataNeighbor(const scalar_t data, const I4 &pos, const index_t dir, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain,
		const GridDataType type, const bool zeroBound){
	/* pos: original position to start from
	dir: in [0,dim*2] as [-x,+x,-y,..,+z]
	*/
	
	const index_t dim = axisFromBound(dir);
	const bool isUpper = boundIsUpper(dir);
	const index_t faceSign = faceSignFromBound(dir);
	
	const BlockGPU<scalar_t> *p_block = &block;
	I4 tempPos = pos;
	
	if(isUpper ? pos.a[dim]==(block.size.a[dim]-1) : pos.a[dim]==0){ //check if there is a boundary in the direction we want to move
		
		switch(block.boundaries[dir].type){
			case BoundaryType::VALUE:
			case BoundaryType::DIRICHLET_VARYING:
			case BoundaryType::FIXED:
			case BoundaryType::GRADIENT:
				//if(!zeroBound){ scatterFixedBoundaryData(data, tempPos, dir, p_block, domain, type); }
				return; // TODO: boundary not differentiable
			case BoundaryType::CONNECTED_GRID:
			{
				//handle multi-block grids, load from correct cell of the connected grid
				const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + p_block->boundaries[dir].cb.connectedGridIndex;
				tempPos = computeConnectedPos<scalar_t>(tempPos, dim, &(p_block->boundaries[dir].cb), domain, 1);
				p_block = p_connectedBlock;
				break;
			}
			case BoundaryType::PERIODIC:
				// compute flux to cell on other side
				// special case of connection to another block
				tempPos.a[dim] = isUpper ? 0 : p_block->size.a[dim]-1;
				break;
			default:
				return;
		}
	} else {
		//same block, just update position
		tempPos.a[dim] += faceSign;
	}
	
	scatterBlockData(data, tempPos, p_block, domain, type);
}

#endif //WITH_GRAD

template<typename scalar_t>
__device__ scalar_t getBlockDataNeighborDiagonal(const I4 pos, const index_t dir1, const index_t dir2, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain,
		const GridDataType type, const bool zeroBound){
	/* pos: original position to start from
	dir: in [0,dim*2] as [-x,+x,-y,..,+z]
	*/

	const bool dir1Empty = isEmptyBound(dir1, block.boundaries);
	// if dir1 leads to a prescibed boundary check dir2 first. if both are prescribed dir2 will be used
	const index_t dirs[2] = {dir1Empty ? dir2 : dir1, dir1Empty ? dir1 : dir2};
	
	const BlockGPU<scalar_t> *p_block = &block;
	I4 tempPos = pos;
	for(index_t i=0; i<2; ++i){
		const index_t bound = dirs[i];
		const index_t dim = axisFromBound(bound);
		const bool isUpper = boundIsUpper(bound);
		const index_t faceSign = faceSignFromBound(bound);
		if(isUpper ? tempPos.a[dim]==(p_block->size.a[dim]-1) : tempPos.a[dim]==0){ //check if there is a boundary in the direction we want to move
			switch(p_block->boundaries[bound].type){
				case BoundaryType::VALUE:
				case BoundaryType::DIRICHLET_VARYING:
				case BoundaryType::FIXED:
				case BoundaryType::GRADIENT:
					return zeroBound ? 0 : getFixedBoundaryData(tempPos, bound, p_block, domain, type);
				case BoundaryType::CONNECTED_GRID:
				{
					//handle multi-block grids, load from correct cell of the connected grid
					const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + p_block->boundaries[bound].cb.connectedGridIndex;
					tempPos = computeConnectedPos<scalar_t>(tempPos, dim, &(p_block->boundaries[bound].cb), domain, 1);
					p_block = p_connectedBlock;
					break;
				}
				case BoundaryType::PERIODIC:
					// compute flux to cell on other side
					// special case of connection to another block
					tempPos.a[dim] = isUpper ? 0 : p_block->size.a[dim]-1;
					break;
				default:
					return 0;
			}
		} else {
			//same block, just update position
			tempPos.a[dim] += faceSign;
		}
	}
	
	return getBlockData(tempPos, p_block, domain, type);
}

#ifdef WITH_GRAD
template<typename scalar_t>
__device__ void scatterBlockDataNeighborDiagonal(const scalar_t data, const I4 pos, const index_t dir1, const index_t dir2,
		const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain,
		const GridDataType type, const bool zeroBound){
	/* pos: original position to start from
	dir: in [0,dim*2] as [-x,+x,-y,..,+z]
	*/

	const bool dir1Empty = isEmptyBound(dir1, block.boundaries);
	// if dir1 leads to a prescibed boundary check dir2 first. if both are prescribed dir2 will be used
	const index_t dirs[2] = {dir1Empty ? dir2 : dir1, dir1Empty ? dir1 : dir2};
	
	const BlockGPU<scalar_t> *p_block = &block;
	I4 tempPos = pos;
	for(index_t i=0; i<2; ++i){
		const index_t bound = dirs[i];
		const index_t dim = axisFromBound(bound);
		const bool isUpper = boundIsUpper(bound);
		const index_t faceSign = faceSignFromBound(bound);
		if(isUpper ? tempPos.a[dim]==(p_block->size.a[dim]-1) : tempPos.a[dim]==0){ //check if there is a boundary in the direction we want to move
			switch(p_block->boundaries[bound].type){
				case BoundaryType::VALUE:
				case BoundaryType::DIRICHLET_VARYING:
				case BoundaryType::FIXED:
				case BoundaryType::GRADIENT:
					//if(!zeroBound){ scatterFixedBoundaryData(data, tempPos, bound, p_block, domain, type);}
					return; // TODO: boundary not differentiable
				case BoundaryType::CONNECTED_GRID:
				{
					//handle multi-block grids, load from correct cell of the connected grid
					const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + p_block->boundaries[bound].cb.connectedGridIndex;
					tempPos = computeConnectedPos<scalar_t>(tempPos, dim, &(p_block->boundaries[bound].cb), domain, 1);
					p_block = p_connectedBlock;
					break;
				}
				case BoundaryType::PERIODIC:
					// compute flux to cell on other side
					// special case of connection to another block
					tempPos.a[dim] = isUpper ? 0 : p_block->size.a[dim]-1;
					break;
				default:
					return;
			}
		} else {
			//same block, just update position
			tempPos.a[dim] += faceSign;
		}
	}
	
	scatterBlockData(data, tempPos, p_block, domain, type);
}

#endif //WITH_GRAD

/** Helper for getCornerValue(). */
template<typename scalar_t>
struct CycleDirection{
	index_t dir1;
	index_t dir2;
	const BlockGPU<scalar_t> *p_block;
	I4 pos;
};
/** Return type of getCornerValue(). */
template<typename scalar_t>
struct CornerValue{
	scalar_t data;
	index_t numCells; // 0 if data is from boundary.
	BoundaryConditionType boundType; // the type of the boundary if the data comes from a boundary
};

/**
 * compute the corner value of a cell by interpolating the adjacent cells
 * can exclude cells, depth 0 for center cell, depth 1 for direct neighbors
 * returns: CornerValue<scalar_t>
 *   .data: interpolated corner value
 *   .numCells: cells used to compute the value. set to 0 to indicate that the value was taken from a fixed boundary.
 */
template <typename scalar_t>
__device__
CornerValue<scalar_t> getCornerValue(const I4 &pos, const index_t dir1, const index_t dir2,
		const bool includeDepth0, const bool includeDepth1, const index_t maxDepth,
		const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain, const GridDataType type) {
	
	scalar_t data = 0; // accumulated data from included cells
	index_t numCells = 0; // traversed cells, used as divisor
	BoundaryConditionType boundType = BoundaryConditionType::DIRICHLET;
	
	// depth0, the center cell, assumed to be valid
	if(includeDepth0){
		data += getBlockData(pos, &block, domain, type);
	}
	++numCells;
	
	// setup for traversal. go in both directions around the corner. stop if same cell is found, fixed boundary is found, or maxDepth is reached.
	CycleDirection<scalar_t> cycleDir[2] = {
		{.dir1=dir1, .dir2=dir2, .p_block=&block, .pos=pos},
		{.dir1=dir2, .dir2=dir1, .p_block=&block, .pos=pos}
	};
	
	for(index_t depth=1; depth<=maxDepth; ++depth){
		
		for(index_t d=0; d<2; ++d){
			const index_t axis = axisFromBound(cycleDir[d].dir1);
			const index_t faceSign = faceSignFromBound(cycleDir[d].dir1);
			//const index_t isUpper = boundIsUpper(cycleDir[d].dir1);
			const bool atBound = isAtBound(cycleDir[d].pos, cycleDir[d].dir1, cycleDir[d].p_block);
			
			if(atBound){
				switch(cycleDir[d].p_block->boundaries[cycleDir[d].dir1].type){
					case BoundaryType::VALUE:
						return {.data=getFixedBoundaryData(cycleDir[d].pos, cycleDir[d].dir1, cycleDir[d].p_block, domain, type),
								.numCells=0,
								.boundType=BoundaryConditionType::DIRICHLET};
					case BoundaryType::DIRICHLET_VARYING:
					case BoundaryType::FIXED:
					{
						// check if we can interpolate to the corner along the SAME boundary (no further checks for going to adjacent blocks, etc.)
						const bool atBound2 = isAtBound(cycleDir[d].pos, cycleDir[d].dir2, cycleDir[d].p_block);
						boundType = getFixedBoundaryType(cycleDir[d].pos, cycleDir[d].dir1, cycleDir[d].p_block, type);
						if(atBound2){
							// TODO: extrapolate from other direction?
							return {.data=getFixedBoundaryData(cycleDir[d].pos, cycleDir[d].dir1, cycleDir[d].p_block, domain, type),
									.numCells=0,
									.boundType=boundType};
						} else {
							I4 neighborPos = cycleDir[d].pos;
							neighborPos.a[axisFromBound(cycleDir[d].dir2)] += faceSignFromBound(cycleDir[d].dir2);
							return {.data=(getFixedBoundaryData(cycleDir[d].pos, cycleDir[d].dir1, cycleDir[d].p_block, domain, type)
										+ getFixedBoundaryData(neighborPos, cycleDir[d].dir1, cycleDir[d].p_block, domain, type)) * static_cast<scalar_t>(0.5),
									.numCells=0,
									.boundType=boundType};
						}
						break;
					}
					case BoundaryType::GRADIENT:
						// gradient is given, but this boundary is not (yet) supported.
						return {.data=0, .numCells=0, .boundType=BoundaryConditionType::NEUMANN};
					case BoundaryType::CONNECTED_GRID:
					{
						//handle multi-block grids, go correct cell of the connected grid
						const ConnectedBoundaryGPU<scalar_t> *p_cb = &(cycleDir[d].p_block->boundaries[cycleDir[d].dir1].cb);
						
						cycleDir[d].p_block = domain.blocks + p_cb->connectedGridIndex;
						
						cycleDir[d].pos.a[axis] += faceSign;
						cycleDir[d].pos = computeConnectedPos<scalar_t>(cycleDir[d].pos, axis, p_cb, domain, 1);
						
						// update directions, respecting any shuffling and inversion
						const index_t dir1 = cycleDir[d].dir1;
						cycleDir[d].dir1 = computeConnectedDir(cycleDir[d].dir2, axis, p_cb, domain);
						cycleDir[d].dir2 = invertBound(computeConnectedDir(dir1, axis, p_cb, domain));
						
						break;
					}
					case BoundaryType::PERIODIC:
					{
						const index_t isUpper = boundIsUpper(cycleDir[d].dir1);
						cycleDir[d].pos.a[axis] = isUpper ? 0 : cycleDir[d].p_block->size.a[axis]-1;
						
						const index_t dir1 = cycleDir[d].dir1;
						cycleDir[d].dir1 = cycleDir[d].dir2;
						cycleDir[d].dir2 = invertBound(dir1);
						break;
					}
					default:
						break;
				}
				
			} else {
				// just the neighbor cell in this block
				cycleDir[d].pos.a[axis] += faceSign;
				// update direction for next step to keep going around same corner
				const index_t dir1 = cycleDir[d].dir1;
				cycleDir[d].dir1 = cycleDir[d].dir2;
				cycleDir[d].dir2 = invertBound(dir1);
			}
			
			// check if same cell is found
			const index_t dOther = d^1; // (d+1)%2
			if(cycleDir[d].p_block==cycleDir[dOther].p_block && cycleDir[d].pos==cycleDir[dOther].pos){
				goto returnData; // break double loop
			}
			
			// add cell's data
			if(depth>1 || includeDepth1){
				data += getBlockData(cycleDir[d].pos, cycleDir[d].p_block, domain, type);
			}
			++numCells;
		}
		
	}
	
	returnData:
	return {.data=data / static_cast<scalar_t>(numCells), .numCells=numCells, .boundType=boundType} ;
}

#ifdef WITH_GRAD
/**
 * scatter the gradient of a corner value of a cell to the adjacent cells
 * can exclude cells, depth 0 for center cell, depth 1 for direct neighbors
 * input: CornerValue<scalar_t>
 *   .data: gradient value to scatter
 *   .numCells: cells used to compute the value, as returned by getCornerValue() with the same arguments.
 */
template <typename scalar_t>
__device__
void scatterCornerValue_GRAD(const CornerValue<scalar_t> cVal_grad, const I4 &pos, const index_t dir1, const index_t dir2,
		const bool includeDepth0, const bool includeDepth1, const index_t maxDepth,
		const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain, const GridDataType type) {
	
	// if cVal_grad.numCells==0 the forward value came from a boundary, so we must not scatter to normal cells here.
	// still have to go through the search to find that boundary.
	const bool isValueFromCells = cVal_grad.numCells>0;
	
	scalar_t data_grad = cVal_grad.data;
	if(isValueFromCells){ 
		data_grad /= static_cast<scalar_t>(cVal_grad.numCells);
	}
	
	// depth0, the center cell, assumed to be valid
	if(includeDepth0){
		//data += getBlockData(pos, &block, domain, type);
		scatterBlockData<scalar_t>(data_grad, pos, &block, domain, type);
	}
	
	// setup for traversal. go in both directions around the corner. stop if same cell is found, fixed boundary is found, or maxDepth is reached.
	CycleDirection<scalar_t> cycleDir[2] = {
		{.dir1=dir1, .dir2=dir2, .p_block=&block, .pos=pos},
		{.dir1=dir2, .dir2=dir1, .p_block=&block, .pos=pos}
	};
	
	for(index_t depth=1; depth<=maxDepth; ++depth){
		
		for(index_t d=0; d<2; ++d){
			const index_t axis = axisFromBound(cycleDir[d].dir1);
			const index_t faceSign = faceSignFromBound(cycleDir[d].dir1);
			//const index_t isUpper = boundIsUpper(cycleDir[d].dir1);
			const bool atBound = isAtBound(cycleDir[d].pos, cycleDir[d].dir1, cycleDir[d].p_block);
			
			if(atBound){
				switch(cycleDir[d].p_block->boundaries[cycleDir[d].dir1].type){
					case BoundaryType::FIXED:
					{
						// check if we can interpolate to the corner along the SAME boundary (no further checks for going to adjacent blocks, etc.)
						const bool atBound2 = isAtBound(cycleDir[d].pos, cycleDir[d].dir2, cycleDir[d].p_block);
						if(atBound2){
							// TODO: extrapolate from other direction?
							scatterFixedBoundaryData(data_grad, cycleDir[d].pos, cycleDir[d].dir1, cycleDir[d].p_block, domain, type);
						} else {
							I4 neighborPos = cycleDir[d].pos;
							neighborPos.a[axisFromBound(cycleDir[d].dir2)] += faceSignFromBound(cycleDir[d].dir2);
							data_grad *= static_cast<scalar_t>(0.5);
							scatterFixedBoundaryData(data_grad, cycleDir[d].pos, cycleDir[d].dir1, cycleDir[d].p_block, domain, type);
							scatterFixedBoundaryData(data_grad, neighborPos, cycleDir[d].dir1, cycleDir[d].p_block, domain, type);
						}
						return;
					}
					case BoundaryType::CONNECTED_GRID:
					{
						//handle multi-block grids, go correct cell of the connected grid
						const ConnectedBoundaryGPU<scalar_t> *p_cb = &(cycleDir[d].p_block->boundaries[cycleDir[d].dir1].cb);
						
						cycleDir[d].p_block = domain.blocks + p_cb->connectedGridIndex;
						
						cycleDir[d].pos.a[axis] += faceSign;
						cycleDir[d].pos = computeConnectedPos<scalar_t>(cycleDir[d].pos, axis, p_cb, domain, 1);
						
						// update directions, respecting any shuffling and inversion
						const index_t dir1 = cycleDir[d].dir1;
						cycleDir[d].dir1 = computeConnectedDir(cycleDir[d].dir2, axis, p_cb, domain);
						cycleDir[d].dir2 = invertBound(computeConnectedDir(dir1, axis, p_cb, domain));
						
						break;
					}
					case BoundaryType::PERIODIC:
					{
						const index_t isUpper = boundIsUpper(cycleDir[d].dir1);
						cycleDir[d].pos.a[axis] = isUpper ? 0 : cycleDir[d].p_block->size.a[axis]-1;
						
						const index_t dir1 = cycleDir[d].dir1;
						cycleDir[d].dir1 = cycleDir[d].dir2;
						cycleDir[d].dir2 = invertBound(dir1);
						break;
					}
					default:
						break;
				}
				
			} else {
				// just the neighbor cell in this block
				cycleDir[d].pos.a[axis] += faceSign;
				// update direction for next step to keep going around same corner
				const index_t dir1 = cycleDir[d].dir1;
				cycleDir[d].dir1 = cycleDir[d].dir2;
				cycleDir[d].dir2 = invertBound(dir1);
			}
			
			// check if same cell is found
			const index_t dOther = d^1; // (d+1)%2
			if(cycleDir[d].p_block==cycleDir[dOther].p_block && cycleDir[d].pos==cycleDir[dOther].pos){
				return; // break double loop
			}
			
			// scatter cell's data
			if(isValueFromCells && (depth>1 || includeDepth1)){
				//data += getBlockData(cycleDir[d].pos, cycleDir[d].p_block, domain, type);
				scatterBlockData(data_grad, cycleDir[d].pos, cycleDir[d].p_block, domain, type);
			}
		}
		
	}
}



#endif //WITH_GRAD

template <typename scalar_t, index_t DIMS>
__device__
Vector<scalar_t,DIMS> getBlockDataGradient(const I4 &pos, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain, const GridDataType type){
	Vector<scalar_t,DIMS> dataGrad = {.a={0}};
	for(index_t dim=0; dim<DIMS; ++dim){
		//getBlockDataNeighbor might return boundary values, which are defined on faces, not cell centers,
		// thus the finite difference using it directly would be wrong
		scalar_t distance = 2.0;
		for(index_t isUpper=0; isUpper<2; ++isUpper){
			index_t boundDir = (dim<<1) + isUpper;
			scalar_t value = 0;
			
			const CellInfo<scalar_t> cellInfo = resolveNeighborCell(pos, boundDir, &block, domain).cell; //don't need axis mapping, just the value and boundary information
			if(cellInfo.isBlock){
				value = getBlockData(cellInfo.pos, cellInfo.p_block, domain, type);
			} else {
				BoundaryConditionType boundType = 
					cellInfo.p_bound->data[gridDataTypeToIndex(type)].isStaticType ?
						cellInfo.p_bound->data[gridDataTypeToIndex(type)].boundaryType :
						cellInfo.p_bound->data[gridDataTypeToIndex(type)].p_boundaryTypes[pos.w];
						
				if(boundType==BoundaryConditionType::DIRICHLET){
					value = getFixedBoundaryData(pos, boundDir, &block, domain, type);
					distance -= 0.5;
				}else{ // NEUMANN, ignore boundary and use one-sided difference
					value = getBlockData(pos, &block, domain, type);
					distance -= 1.0;
				}
			}
			
			dataGrad.a[dim] += (isUpper*2 -1) * value;
		}
		dataGrad.a[dim] /= distance;
	}
	
	if(block.hasTransform){
		I4 tempPos = pos;
		tempPos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(block.transform) + flattenIndex(tempPos, block);
		//pressureGrad = matmul(T->Minv, pressureGrad);
		dataGrad = matmul(dataGrad, T->Minv);
	}
	
	return dataGrad;
}


