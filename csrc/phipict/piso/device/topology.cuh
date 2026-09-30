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
// Boundaries, block connectivity and neighbour cell resolution.

#pragma once

#include "piso/device/launch.cuh"


template<typename scalar_t>
 __host__ __device__ inline
 bool isEmptyBound(const index_t idx, const BoundaryGPU<scalar_t> *bounds){
	return
		bounds[idx].type==BoundaryType::DIRICHLET
		|| bounds[idx].type==BoundaryType::DIRICHLET_VARYING
		|| bounds[idx].type==BoundaryType::GRADIENT
		|| bounds[idx].type==BoundaryType::FIXED;
}

/** Electric potential BC of the face cell that `pos` touches on FIXED bound `idx`.
 *  A face has one PotentialBC unless FixedBoundary::setPotentialTypes gave it a per-cell
 *  mask, which is indexed at the face position (normal coordinate 0).
 *
 *  The face type is an explicit setting because it cannot be inferred: Block::CloseBoundary
 *  makes solid walls AND prescribed-velocity in/outflows alike as BoundaryType::FIXED with a
 *  DIRICHLET velocity BC, so open planes must be marked PotentialBC::OPEN. The default,
 *  INSULATING, is the all-walls case.
 *
 *  Only valid for a FIXED bound: the fb member of the union is garbage otherwise. */
template<typename scalar_t>
 __host__ __device__ inline
 PotentialBC potentialBCAt(const I4 &pos, const index_t idx, const BoundaryGPU<scalar_t> *bounds){
	const FixedBoundaryGPU<scalar_t> &fb = bounds[idx].fb;
	if(fb.potentialTypes == nullptr) return fb.potentialType;
	I4 facePos = pos;
	facePos.a[idx>>1] = 0;
	facePos.w = 0;
	return static_cast<PotentialBC>(fb.potentialTypes[flattenIndex(facePos, fb.potentialStride)]);
}

/** Value of the face cell of `pos` on FIXED bound `idx`, 0 if the face has no values:
 *  the prescribed φ where potentialBCAt is DIRICHLET, the prescribed current into the fluid
 *  through the cell where it is CURRENT. Meaningless elsewhere. */
template<typename scalar_t>
 __host__ __device__ inline
 scalar_t potentialValueAt(const I4 &pos, const index_t idx, const BoundaryGPU<scalar_t> *bounds){
	const FixedBoundaryGPU<scalar_t> &fb = bounds[idx].fb;
	if(fb.potentialValues == nullptr) return static_cast<scalar_t>(0);
	I4 facePos = pos;
	facePos.a[idx>>1] = 0;
	facePos.w = 0;
	return fb.potentialValues[flattenIndex(facePos, fb.potentialStride)];
}

#ifdef WITH_GRAD
/** Slot of the face cell of `pos` in the potential values gradient of FIXED bound `idx`,
 *  nullptr if the gradient is not tracked. */
template<typename scalar_t>
 __host__ __device__ inline
 scalar_t* potentialValueGradPtr(const I4 &pos, const index_t idx, const BoundaryGPU<scalar_t> *bounds){
	const FixedBoundaryGPU<scalar_t> &fb = bounds[idx].fb;
	if(fb.potentialValuesGrad == nullptr) return nullptr;
	I4 facePos = pos;
	facePos.a[idx>>1] = 0;
	facePos.w = 0;
	return fb.potentialValuesGrad + flattenIndex(facePos, fb.potentialStride);
}
#endif //WITH_GRAD

/** True if the face cell of `pos` on bound `idx` is φ Dirichlet (ghost cell φ = value). */
template<typename scalar_t>
 __host__ __device__ inline
 bool isEpotDirichletBound(const I4 &pos, const index_t idx, const BoundaryGPU<scalar_t> *bounds){
	return bounds[idx].type==BoundaryType::FIXED
		&& potentialBCAt(pos, idx, bounds)==PotentialBC::DIRICHLET;
}

/** True if the face cell of `pos` on bound `idx` is a solid wall (INSULATING or THIN_WALL)
 *  rather than an open plane or a φ=0 face. For the inductionless MHD solve such a wall is
 *  insulating: j_n = 0. A no-slip wall has u_wall = 0, hence (u×B)_n|_wall = 0 and
 *  dphi/dn = 0, which is what the homogeneous-Neumann Epot matrix already assumes, so
 *  dropping the boundary flux is exact there. A thin conducting wall also takes no normal
 *  current into the fluid-side face; its tangential wall current enters the matrix through
 *  the Robin correction only.
 *
 *  Moving walls (u_wall != 0) would need (u×B)_n evaluated at the wall rather than a plain
 *  drop; no MHD case currently uses one. */
template<typename scalar_t>
 __host__ __device__ inline
 bool isInsulatingWallBound(const I4 &pos, const index_t idx, const BoundaryGPU<scalar_t> *bounds){
	if(bounds[idx].type!=BoundaryType::FIXED) return false;
	const PotentialBC bc = potentialBCAt(pos, idx, bounds);
	return bc==PotentialBC::INSULATING || bc==PotentialBC::THIN_WALL;
}

/** True if the face cell of `pos` on bound `idx` has a prescribed current (CURRENT): the
 *  normal current into the fluid is the cell's potential value, both in the Poisson RHS and
 *  in the reconstructed current density. Like an insulating wall it is a Neumann face of
 *  the matrix, and with a zero value it is exactly that wall. */
template<typename scalar_t>
 __host__ __device__ inline
 bool isEpotCurrentBound(const I4 &pos, const index_t idx, const BoundaryGPU<scalar_t> *bounds){
	return bounds[idx].type==BoundaryType::FIXED
		&& potentialBCAt(pos, idx, bounds)==PotentialBC::CURRENT;
}

/** True if the face cell of `pos` on bound `idx` is a thin conducting wall. */
template<typename scalar_t>
 __host__ __device__ inline
 bool isThinWallBound(const I4 &pos, const index_t idx, const BoundaryGPU<scalar_t> *bounds){
	return bounds[idx].type==BoundaryType::FIXED
		&& potentialBCAt(pos, idx, bounds)==PotentialBC::THIN_WALL;
}

// boundary: [0,dims*2) = [-x,+x,-y,+y,-z,+z]
// lowest bit is the direction (0=lower, 1= upper), next 2 bits are the axis (00=x, 01=y, 10=z)
__host__ __device__ constexpr
index_t axisFromBound(const index_t bound){
	return bound>>1; // remove the direction bit
}
/** axis must be non-negative, isUpper 0 or 1. */
__host__ __device__ constexpr
index_t axisToBound(const index_t axis, const index_t isUpper){
	return (axis<<1) | isUpper;
}

__host__ __device__ constexpr
index_t boundIsUpper(const index_t bound){
	return bound&1; // check the direction bit
}

__host__ __device__ constexpr
index_t invertBound(const index_t bound){
	return bound^1; // flip the direction bit
}

__host__ __device__ constexpr
index_t faceSignFromBound(const index_t bound){
	return boundIsUpper(bound)*2 - 1;
}

/**
 * check if the cell of block "block" at position "pos" has a boundary (fixed or connected) in the direction "bound".
 */
template<typename scalar_t>
__device__ inline
bool isAtBound(const I4 &pos, const index_t bound, const BlockGPU<scalar_t> &block){
	const index_t axis = axisFromBound(bound);
	return boundIsUpper(bound) ? pos.a[axis]==(block.size.a[axis]-1) : pos.a[axis]==0;
}
template<typename scalar_t>
__device__ inline
bool isAtBound(const I4 &pos, const index_t bound, const BlockGPU<scalar_t> *p_block){
	const index_t axis = axisFromBound(bound);
	return boundIsUpper(bound) ? pos.a[axis]==(p_block->size.a[axis]-1) : pos.a[axis]==0;
}

/**
 * returns: positive a s.t. (otherAxis + a)%dims == axis
 */
__host__ __device__ constexpr
index_t getAxisRelativeToOther(const index_t axis, const index_t otherAxis, const index_t dims){
	return posMod(axis - otherAxis, dims);
}

struct RowMeta{
	int endOffset;
	int size;
};
template<typename scalar_t>
__device__ RowMeta getCSRMatrixRowEndOffsetFromBlockBoundaries3D(const int flatPos, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain){
	const I4 pos = unflattenIndex(flatPos, block);
		
	int rowSize = 2*domain.numDims+1;
	int rowEndOffset=(flatPos+1)*rowSize;
	
	//subtract for open/closed bounds
	// number of cells with boundary at ? before and including current cell:
	//X
	// -x
	if(isEmptyBound(0,block.boundaries)){
		rowEndOffset -= (flatPos/block.size.x)+1;
		if(pos.x==0){
			--rowSize;
		}
	}
	// +x
	if(isEmptyBound(1,block.boundaries)){
		rowEndOffset -= (flatPos + 1)/block.size.x;
		if(pos.x==block.size.x-1){
			--rowSize;
		}
	}
	//Y
	if(domain.numDims>1){
		// -y
		if(isEmptyBound(2,block.boundaries)){
			rowEndOffset -= block.size.x*pos.z //previous slices
				+ (pos.y==0 ? pos.x+1 : block.size.x); // current slice
			if(pos.y==0){
				--rowSize;
			}
		}
		// +y
		if(isEmptyBound(3,block.boundaries)){
			rowEndOffset -= block.size.x*pos.z //previous slices
				+ (pos.y==(block.size.y-1) ? pos.x+1 : 0); // current slice
			if(pos.y==block.size.y-1){
				--rowSize;
			}
		}
	}
	//Z
	if(domain.numDims>2){
		// -z
		if(isEmptyBound(4,block.boundaries)){
			rowEndOffset -= (pos.z==0 ? flatPos+1 : block.stride.z); // flatPos=pos.x+pos.y*size.y, stride.z=size.x*size.y
			if(pos.z==0){
				--rowSize;
			}
		}
		// +z
		if(isEmptyBound(5,block.boundaries)){
			rowEndOffset -= (pos.z==(block.size.z-1) ? flatPos+1 - (pos.z*block.stride.z): 0); // stride.z=size.x*size.y
			if(pos.z==block.size.z-1){
				--rowSize;
			}
		}
	}
	return {rowEndOffset, rowSize};
}

template<typename scalar_t>
__device__ inline
index_t computeConnectedDir(const index_t dir, const index_t boundaryDim, const ConnectedBoundaryGPU<scalar_t> *p_cb, const DomainGPU<scalar_t> &domain){
	
	const index_t dirAxis = axisFromBound(dir);
	const index_t dirRelativeToConnection = getAxisRelativeToOther(dirAxis, boundaryDim, domain.numDims);
	
	// p_cb->axes uses the same format as face indexing, but the isUpper-bit means that the connection is inverted.
	return p_cb->axes.a[dirRelativeToConnection] ^ boundIsUpper(dir);
}

template<typename scalar_t>
__device__ inline
I4 computeConnectedPos(const I4 pos, const index_t boundaryDim, const ConnectedBoundaryGPU<scalar_t> *p_cb, const DomainGPU<scalar_t> &domain, const index_t borderOffset=0){
	const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + p_cb->connectedGridIndex;
	I4 connectedPos = pos; //sets w
	
	index_t connectedAxis = p_cb->axes.a[0]>>1;
	//connectedPos.w = connectedAxis;
	connectedPos.a[connectedAxis] = (p_cb->axes.a[0] & 1) ? p_connectedBlock->size.a[connectedAxis] - 1 -borderOffset: borderOffset;
	if(domain.numDims>1){
		index_t axis = (boundaryDim+1)%domain.numDims;
		connectedAxis = p_cb->axes.a[1]>>1;
		connectedPos.a[connectedAxis] = (p_cb->axes.a[1] & 1) ? p_connectedBlock->size.a[connectedAxis]-1 - pos.a[axis] : pos.a[axis];
		if(domain.numDims>2){
			axis = (boundaryDim+2)%domain.numDims;
			connectedAxis = p_cb->axes.a[2]>>1;
			connectedPos.a[connectedAxis] = (p_cb->axes.a[2] & 1) ? p_connectedBlock->size.a[connectedAxis]-1 - pos.a[axis] : pos.a[axis];
		}
	}
	
	return connectedPos;
}


template<typename scalar_t>
__device__ inline I4 computeConnectedPosWithChannel(const I4 pos, const index_t boundaryDim, const ConnectedBoundaryGPU<scalar_t> *p_cb, const DomainGPU<scalar_t> &domain, const index_t borderOffset=0){
	// computeConnectedPos() does not change pos.w
	I4 connectedPos = computeConnectedPos(pos, boundaryDim, p_cb, domain, borderOffset);
	// the requested component is not necessarily the boundary axis here
	// boundaryAxis = bound>>1; connected to axes[0]
	// requestedAxis = pos.w; connected to axes[?]
	// pos.W == bA -> axes[0], pos.w == (bA+1)%dims -> axes[1], pos.w == (bA+2)%dims -> axes[2]
	const index_t connectionIndex = posMod(pos.w-boundaryDim, domain.numDims); //needs positive mod
	connectedPos.w = p_cb->axes.a[connectionIndex]>>1;
	return connectedPos;
}

template<typename scalar_t>
__device__ inline bool isGlobalIndexInBlock(const index_t idx, const BlockGPU<scalar_t> *p_block){
	const index_t blockGlobalIndexStart = p_block->globalOffset;
	const index_t blockGlobalIndexEnd = blockGlobalIndexStart + p_block->stride.w;
	return (blockGlobalIndexStart<=idx && idx<blockGlobalIndexEnd);
}
template<typename scalar_t>
__device__ inline index_t getBlockIndexFromGlobalIndex(const index_t idx, const DomainGPU<scalar_t> &domain){
	for(index_t blockIdx=0; blockIdx<domain.numBlocks; ++blockIdx){
		const BlockGPU<scalar_t> *p_block = domain.blocks + blockIdx;
		if(isGlobalIndexInBlock(idx, p_block)){
			return blockIdx;
		}
	}
	return -1;
}

template<typename scalar_t>
__device__ inline index_t getBoundaryIndexToBlock(const index_t otherBlockIdx, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain){
	for(index_t bound=0; bound<(domain.numDims*2); ++bound){
		if(block.boundaries[bound].type==BoundaryType::CONNECTED_GRID && block.boundaries[bound].cb.connectedGridIndex==otherBlockIdx){
			return bound;
		}
	}
	return -1;
}


template<typename scalar_t>
struct CellInfo{
	I4 pos;
	bool isBlock;
	union{
		//struct blockInfo{
			const BlockGPU<scalar_t> *p_block;
		//};
		//struct boundInfo{
			const FixedBoundaryGPU<scalar_t> *p_bound;
		//};
	};
};


template<typename scalar_t>
struct NeighborCellInfo{
	CellInfo<scalar_t> cell;
	Vector<index_t, 3> axisMapping;
};

template<typename scalar_t>
__device__ inline
void initDefaultAxisMapping(NeighborCellInfo<scalar_t> &info){
	info.axisMapping.a[0] = 0;
	info.axisMapping.a[1] = 2;
	info.axisMapping.a[2] = 4;
}

template<typename scalar_t>
__device__ inline
index_t flattenIndex(const CellInfo<scalar_t> &info){
	return info.isBlock ? flattenIndex(info.pos, info.p_block->stride) : flattenIndex(info.pos, info.p_bound->stride);
}
template<typename scalar_t>
__device__ inline
index_t flattenIndexGlobal(const CellInfo<scalar_t> &info, const DomainGPU<scalar_t> &domain){
	return info.isBlock ? flattenIndexGlobal(info.pos, info.p_block, domain) : 0;
}

template<typename scalar_t>
__device__ inline
NeighborCellInfo<scalar_t> resolveNeighborCell(const I4 &pos, const index_t dir, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain){
	
	NeighborCellInfo<scalar_t> info;
	initDefaultAxisMapping(info);
	const index_t axis = axisFromBound(dir);
	
	if(isAtBound(pos, dir, p_block)){
		switch(p_block->boundaries[dir].type){
			case BoundaryType::FIXED:
			{
				info.cell.isBlock = false;
				info.cell.pos = pos;
				info.cell.pos.a[axis] = 0;
				info.cell.p_bound = &(p_block->boundaries[dir].fb);
				break;
			}
			case BoundaryType::CONNECTED_GRID:
			{
				const ConnectedBoundaryGPU<scalar_t> *p_cb = &(p_block->boundaries[dir].cb);
				info.cell.isBlock = true;
				info.cell.p_block = domain.blocks + p_cb->connectedGridIndex;
				info.cell.pos = computeConnectedPos(pos, axis, p_cb, domain);
				for(index_t dim=0; dim<domain.numDims; ++dim){
					info.axisMapping.a[dim]=computeConnectedDir(info.axisMapping.a[dim], axis, p_cb, domain);
				}
				break;
			}
			case BoundaryType::PERIODIC:
			{
				info.cell.isBlock = true;
				info.cell.pos = pos;
				info.cell.pos.a[axis] = boundIsUpper(dir) ? 0 : p_block->size.a[axis]-1;
				info.cell.p_block = p_block;
				break;
			}
			default:
				break;
		}
	} else {
		info.cell.isBlock = true;
		info.cell.pos = pos;
		info.cell.pos.a[axis] += faceSignFromBound(dir);
		info.cell.p_block = p_block;
	}
	return info;
}
