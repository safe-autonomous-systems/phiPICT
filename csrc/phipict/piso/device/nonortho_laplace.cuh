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
// Non-orthogonal Laplace components, pressure with boundaries, velocity sources.

#pragma once

#include "piso/device/discretization.cuh"



// --- Laplace kernel coefficients for transformed grids. Used for pressure solve and diffusion. ---

/**
 * to resolve cyclic axis indices from interpolateNonOrthoLaplaceComponents
 */
__host__ __device__
constexpr index_t getAxisRelativeToOtherNonOrtho(const index_t axis, const index_t otherAxis, const index_t dims){
	return (dims - 1 - otherAxis + axis)%dims; 
}

template <typename scalar_t>
__device__
scalar_t inline getInterpolatedNonOrthoLaplaceComponent(const scalar_t *alphaInterp, const index_t face, const index_t otherAxis, const index_t dims){
	const index_t otherAxisRelativeToFace = getAxisRelativeToOtherNonOrtho(otherAxis, axisFromBound(face), dims);
	return alphaInterp[face*(dims-1) + otherAxisRelativeToFace]; // always =alphaInterp[face] for 2D
}

#ifdef WITH_GRAD
template <typename scalar_t>
__device__
void inline addInterpolatedNonOrthoLaplaceComponent_GRAD(scalar_t value, scalar_t *alphaInterp_grad, const index_t face, const index_t otherAxis, const index_t dims){
	const index_t otherAxisRelativeToFace = getAxisRelativeToOtherNonOrtho(otherAxis, axisFromBound(face), dims);
	alphaInterp_grad[face*(dims-1) + otherAxisRelativeToFace] += value; // always =alphaInterp[face] for 2D
}

#endif //WITH_GRAD

template <typename scalar_t, index_t DIMS>
__device__
void interpolateNonOrthoLaplaceComponents(const I4 &pos, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain, scalar_t *alphaInterpOut,
	const bool withViscosity, const bool withA, const bool useFaceTransform){
	// useFaceTransform is experimental
	// needed length for alphaInterpOut: 2*DIMS*(DIMS-1). 2D: 4, 3D: 12
	// 2D: -x01, +x01, -y10, +y10. (note: 01=10)
	// 3D: -x01, -x02, +x01, +x02, -y12, -y 10, +y12, +y10, -z20, -z21, +z20, +z21

	// get center coefficients
	const Vector<scalar_t, getLaplaceCoefficientsFullLength(DIMS)> alphasP = getLaplaceCoefficientsFull<scalar_t, DIMS>(pos, &block);
	const index_t flatPosGlobal = block.globalOffset + flattenIndex(pos, block);
	
	const scalar_t one = static_cast<scalar_t>(1.0);
	const scalar_t raP = (withA ? one /domain.Adiag[flatPosGlobal] : one) * (withViscosity ? getViscosityBlock<scalar_t>(pos, &block, domain, false, 0) : one);

	// loop neighbours to get interpolated coefficients
	for(index_t face=0; face<DIMS*2; ++face){
		const index_t axis = face>>1;
		// Using alias for copied code. TODO: unify.
		const index_t &dim = axis;
		const index_t &bound = face;
		const index_t faceSign = -1 + ((face&1)<<1);
		const bool isUpper = face&1;
		I4 posN = pos;
		posN.w = axis;
		Vector<scalar_t, DIMS> alphasN;
		index_t flatPosGlobalN = 0;
		if((0<pos.a[axis] && pos.a[axis]<(block.size.a[axis]-1)) || !isEmptyBound(bound, block.boundaries)){ // not at prescribed bound
			const BlockGPU<scalar_t> *p_block = &block;
			if(((pos.a[dim]==0 && !isUpper) || (pos.a[dim]==block.size.a[axis]-1 && isUpper)) && block.boundaries[bound].type==BoundaryType::CONNECTED_GRID){
				const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
				posN = computeConnectedPosWithChannel(posN, dim, &block.boundaries[bound].cb, domain);
				alphasN = getLaplaceCoefficientsSingleAxis<scalar_t, DIMS>(posN, axis, p_connectedBlock);
				posN.w = 0;
				flatPosGlobalN = flattenIndex(posN, p_connectedBlock) + p_connectedBlock->globalOffset;
				p_block = p_connectedBlock;
			}else {
				if(!isUpper && pos.a[dim]==0 && block.boundaries[bound].type==BoundaryType::PERIODIC){
					posN.a[dim] = block.size.a[dim]-1;
				}else if(isUpper && pos.a[dim]==block.size.a[dim]-1 && block.boundaries[bound].type==BoundaryType::PERIODIC){
					posN.a[dim] = 0;
				}else{
					posN.a[dim] = pos.a[dim] + faceSign;
				}
				alphasN = getLaplaceCoefficientsSingleAxis<scalar_t, DIMS>(posN, axis, &block);
				posN.w = 0;
				flatPosGlobalN = flattenIndex(posN, block) + block.globalOffset;
				//rowValues[bound+1] = -1 / advectionMatrixDiagonal[tempFlatPos] * invLaplace;
			}
			const scalar_t raN = (withA ? one /domain.Adiag[flatPosGlobalN] : one) * (withViscosity ? getViscosityBlock<scalar_t>(posN, p_block, domain, false, 0) : one);
			if(useFaceTransform){
				//overwrite alphasN with face transform
				alphasN = getLaplaceCoefficientsSingleFace<scalar_t, DIMS>(pos, face, &block); // <- ISSUE here? no
				//alphasN = getLaplaceCoefficientsSingleFace<scalar_t, DIMS>(posN, face, &block);
			}
			for(index_t i=1; i<DIMS; ++i){
				const index_t otherAxis = (axis + i)%DIMS;
				scalar_t alphaFace = 0;
				if(useFaceTransform){
					alphaFace = alphasN.a[otherAxis] * (raP + raN) * 0.5;
				}else{
					alphaFace = (alphasP.a[getLaplaceCoefficientCenterIndex(axis, otherAxis)]*raP + alphasN.a[otherAxis]*raN )*0.5;
				}
				alphaInterpOut[face*(DIMS-1) + (i-1)] = alphaFace;
			}
		} else {
			//TODO: get boundary transform metrics directly? 
			
			//getLaplaceCoefficientsNeighbourBoundaryVarying<scalar_t, DIMS>(, axis, bound)
			for(index_t i=1; i<DIMS; ++i){
				alphaInterpOut[face*(DIMS-1) + (i-1)] = 0;
			}
		}
	}
}

#ifdef WITH_GRAD

template <typename scalar_t, index_t DIMS>
__device__
void scatterNonOrthoLaplaceComponents_GRAD(scalar_t *alphaInterp_grad, const I4 &pos, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain,
		const bool withViscosity, const bool withA, const bool useFaceTransform){
	// useFaceTransform is experimental
	// needed length for alphaInterpOut: 2*DIMS*(DIMS-1). 2D: 4, 3D: 12
	// 2D: -x01, +x01, -y10, +y10. (note: 01=10)
	// 3D: -x01, -x02, +x01, +x02, -y12, -y 10, +y12, +y10, -z20, -z21, +z20, +z21

	// get center coefficients
	const Vector<scalar_t, getLaplaceCoefficientsFullLength(DIMS)> alphasP = getLaplaceCoefficientsFull<scalar_t, DIMS>(pos, &block);
	const index_t flatPosGlobal = block.globalOffset + flattenIndex(pos, block);
	
	const scalar_t one = static_cast<scalar_t>(1.0);
	//const scalar_t raP = (withA ? one /domain.Adiag[flatPosGlobal] : one) * (withViscosity ? getViscosityBlock<scalar_t>(pos, &block, domain, false, 0) : one);
	scalar_t raP_grad = 0;

	// loop neighbours to get interpolated coefficients
	for(index_t face=0; face<DIMS*2; ++face){
		const index_t axis = face>>1;
		// Using alias for copied code. TODO: unify.
		const index_t &dim = axis;
		const index_t &bound = face;
		const index_t faceSign = -1 + ((face&1)<<1);
		const bool isUpper = face&1;
		I4 posN = pos;
		posN.w = axis;
		Vector<scalar_t, DIMS> alphasN;
		index_t flatPosGlobalN = 0;
		if((0<pos.a[axis] && pos.a[axis]<(block.size.a[axis]-1)) || !isEmptyBound(bound, block.boundaries)){ // not at prescribed bound
			const BlockGPU<scalar_t> *p_block = &block;
			if(((pos.a[dim]==0 && !isUpper) || (pos.a[dim]==block.size.a[axis]-1 && isUpper)) && block.boundaries[bound].type==BoundaryType::CONNECTED_GRID){
				const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
				posN = computeConnectedPosWithChannel(posN, dim, &block.boundaries[bound].cb, domain);
				alphasN = getLaplaceCoefficientsSingleAxis<scalar_t, DIMS>(posN, axis, p_connectedBlock);
				posN.w = 0;
				flatPosGlobalN = flattenIndex(posN, p_connectedBlock) + p_connectedBlock->globalOffset;
				p_block = p_connectedBlock;
			}else {
				if(!isUpper && pos.a[dim]==0 && block.boundaries[bound].type==BoundaryType::PERIODIC){
					posN.a[dim] = block.size.a[dim]-1;
				}else if(isUpper && pos.a[dim]==block.size.a[dim]-1 && block.boundaries[bound].type==BoundaryType::PERIODIC){
					posN.a[dim] = 0;
				}else{
					posN.a[dim] = pos.a[dim] + faceSign;
				}
				alphasN = getLaplaceCoefficientsSingleAxis<scalar_t, DIMS>(posN, axis, &block);
				posN.w = 0;
				flatPosGlobalN = flattenIndex(posN, block) + block.globalOffset;
				//rowValues[bound+1] = -1 / advectionMatrixDiagonal[tempFlatPos] * invLaplace;
			}
			if(useFaceTransform){
				//overwrite alphasN with face transform
				alphasN = getLaplaceCoefficientsSingleFace<scalar_t, DIMS>(pos, face, &block); // <- ISSUE here? no
				//alphasN = getLaplaceCoefficientsSingleFace<scalar_t, DIMS>(posN, face, &block);
			}

			//const scalar_t raN = (withA ? one /domain.Adiag[flatPosGlobalN] : one) * (withViscosity ? getViscosityBlock<scalar_t>(posN, p_block, domain, false, 0) : one);
			scalar_t raN_grad = 0;

			for(index_t i=1; i<DIMS; ++i){
				const index_t otherAxis = (axis + i)%DIMS;
				const scalar_t alphaFace_grad = alphaInterp_grad[face*(DIMS-1) + (i-1)];
				if(useFaceTransform){
					//alphaFace = alphasN.a[otherAxis] * (raP + raN) * 0.5;
					raP_grad += alphasN.a[otherAxis] * 0.5 * alphaFace_grad;
					raN_grad += alphasN.a[otherAxis] * 0.5 * alphaFace_grad;
				}else{
					//alphaFace = (alphasP.a[getLaplaceCoefficientCenterIndex(axis, otherAxis)]*raP + alphasN.a[otherAxis]*raN )*0.5;
					raP_grad += alphasP.a[getLaplaceCoefficientCenterIndex(axis, otherAxis)] * 0.5 * alphaFace_grad;
					raN_grad += alphasN.a[otherAxis] * 0.5 * alphaFace_grad;

				}
				//alphaInterpOut[face*(DIMS-1) + (i-1)] = alphaFace;
			}
			
			const scalar_t raN = withA ? one /domain.Adiag[flatPosGlobalN] : one;
			
			if(withViscosity){
				scalar_t raN_gradTemp = raN_grad;
				if(withA){
					raN_gradTemp *= raN; 
				}
				scatterViscosityBlock_GRAD<scalar_t>(raN_gradTemp, posN, p_block, domain, false, 0);
			}
			if(withA){
				if(withViscosity){
					raN_grad *= getViscosityBlock<scalar_t>(posN, p_block, domain, false, 0);
				}
				atomicAdd(domain.Adiag_grad + flatPosGlobalN, -raN_grad*raN*raN);
			}
		}
	}
	
	const scalar_t raP = withA ? one /domain.Adiag[flatPosGlobal] : one;

	if(withViscosity){
		scalar_t raP_gradTemp = raP_grad;
		if(withA){
			raP_gradTemp *= raP; 
		}
		scatterViscosityBlock_GRAD<scalar_t>(raP_gradTemp, pos, &block, domain, false, 0);
	}
	if(withA){
		if(withViscosity){
			raP_grad *= getViscosityBlock<scalar_t>(pos, &block, domain, false, 0);
		}
		atomicAdd(domain.Adiag_grad + flatPosGlobal, -raP_grad*raP*raP);
	}
	
}

#endif //WITH_GRAD


template<typename scalar_t>
__device__ scalar_t getPressureNeighborDiagonal(const I4 pos, const index_t dir1, const index_t dir2, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain, const bool zeroBound){
	/* pos: original position to start from
	dir: in [0,dim*2] as [-x,+x,-y,..,+z]
	*/

	const bool dir1Empty = isEmptyBound(dir1, block.boundaries);
	// if dir1 leads to a prescibed boundary check dir2 first. if both are prescribed dir2 will be used
	const index_t dirs[2] = {dir1Empty ? dir2 : dir1, dir1Empty ? dir1 : dir2};
	//const index_t &d1 = dir1Empty ? dir2 : dir1;
	//const index_t &d2 = dir1Empty ? dir1 : dir2;
	
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
				case BoundaryType::GRADIENT: //TODO: how to handle this case here?
					// enforce 0 pressure gradient to avoid changing the prescribed value
					return zeroBound ? 0 : p_block->pressure[flattenIndex(tempPos, p_block)];
					break;
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
					break;
			}
		} else {
			//same block, just update position
			tempPos.a[dim] += faceSign;
		}
	}
	tempPos.w = 0;
	return p_block->pressure[flattenIndex(tempPos, p_block)];
}

/** DEPRECATED, use getBlockData or getBlockDataNeighbor. */
template<typename scalar_t>
__device__ scalar_t getPressureAtWithBounds(const I4 pos, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain, const bool zeroBound){
	// read value at location with support for 1-cell ghost layer
	// any position outside the domain will be treated as being on the ghost layer
	// only one coordinate may be outside the domain (corner ghost cells are not supported)
	
	const int flatPos = flattenIndex(pos, block);
	I4 tempPos = pos;
	tempPos.w = 0; //pressure is scalar
	
	for(int dim=0; dim<domain.numDims; ++dim)
	{
		int bound = dim*2;
		
		if(pos.a[dim]<0){// lower boundary
			switch(block.boundaries[bound].type){
				case BoundaryType::VALUE:
				case BoundaryType::DIRICHLET_VARYING:
				case BoundaryType::FIXED:
				case BoundaryType::GRADIENT: //TODO: how to handle this case here?
					// enforce 0 pressure gradient to avoid changing the prescribed value
					tempPos.a[dim] = 0;
					return zeroBound ? 0 : block.pressure[flattenIndex(tempPos, block)];
					// compute flux only from center cell?
					// velN = velC - grad*distance
					//return grid[flatPos] - domain.bounds[bound].prescribedValue; //* distance
					//break;
				case BoundaryType::CONNECTED_GRID:
				{
					//handle multi-block grids, load from correct cell of the connected grid
					const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
					const I4 otherPos = computeConnectedPos(tempPos, dim, &block.boundaries[bound].cb, domain, -(pos.a[dim]+1));
					// pressure is scalar, so pos.w==0 always
					return p_connectedBlock->pressure[flattenIndex(otherPos, p_connectedBlock)];
				}
				case BoundaryType::PERIODIC:
					// compute flux to cell on other side
					// special case of connection to another block
					tempPos.a[dim] += block.size.a[dim];
					return block.pressure[flattenIndex(tempPos, block)];
				default:
					return 0;
			}
		}
		
		bound = dim*2 + 1;
		
		if(pos.a[dim]>=block.size.a[dim]){// upper boundary
			switch(block.boundaries[bound].type){
				case BoundaryType::VALUE:
				case BoundaryType::DIRICHLET_VARYING:
				case BoundaryType::FIXED:
				case BoundaryType::GRADIENT: //TODO: how to handle this case here?
					// enforce 0 pressure gradient to avoid changing the prescribed value
					tempPos.a[dim] = block.size.a[dim] - 1;
					return zeroBound ? 0 : block.pressure[flattenIndex(tempPos, block)];
				/* case BoundaryType::GRADIENT:
					// compute flux only from center cell?
					// velP = celC + grad*distance
					return grid[flatPos] + domain.bounds[bound].prescribedValue; //* distance
					break; */
				case BoundaryType::CONNECTED_GRID:
				{
					//handle multi-block grids, load from correct cell of the connected grid
					const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
					const I4 otherPos = computeConnectedPos(tempPos, dim, &block.boundaries[bound].cb, domain, pos.a[dim]-block.size.a[dim]);
					return p_connectedBlock->pressure[flattenIndex(otherPos, p_connectedBlock)];
				}
				case BoundaryType::PERIODIC:
					// compute flux to cell on other side
					// special case of connection to another block
					tempPos.a[dim] -= block.size.a[dim];
					return block.pressure[flattenIndex(tempPos, block)];
				default:
					return 0;
			}
		}
	}
	return block.pressure[flatPos];
}

template<typename scalar_t>
__device__
scalar_t getBlockVelocitySource(const I4 &pos, const BlockGPU<scalar_t> *p_block){
	if(p_block->velocitySource!=nullptr){
		if(p_block->isVelocitySourceStatic){
			return p_block->velocitySource[pos.w];
		} else {
			return p_block->velocitySource[flattenIndex(pos, p_block)];
		}
	}
	return 0;
}
template<typename scalar_t>
__device__
void scatterBlockVelocitySource_GRAD(const scalar_t vel_grad, const I4 &pos, const BlockGPU<scalar_t> *p_block){
	if(p_block->velocitySource_grad!=nullptr){
		if(p_block->isVelocitySourceStatic){
			atomicAdd(p_block->velocitySource_grad + pos.w, vel_grad);
		} else {
			atomicAdd(p_block->velocitySource_grad + flattenIndex(pos, p_block), vel_grad);
		}
	}
}

