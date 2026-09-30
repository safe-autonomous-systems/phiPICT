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
// Face transforms, interpolation, determinants, Laplace coefficients, face fluxes, viscosity.

#pragma once

#include "piso/device/velocity.cuh"


template <typename scalar_t, int DIMS>
__device__ inline
const TransformGPU<scalar_t, DIMS>* getFaceTransformPtr(I4 facePos, const index_t face, const BlockGPU<scalar_t> *p_block){
	if(!p_block->hasFaceTransform){ return nullptr; }
	const index_t dim = axisFromBound(face);
	const index_t isUpper = boundIsUpper(face);
	//I4 facePos = pos;
	facePos.a[dim] += isUpper;
	facePos.w = dim;
	I4 faceSize = p_block->size;
	for(index_t i=0; i<DIMS; ++i) { faceSize.a[i] += 1; }
	I4 faceStride = {{.x=1, .y=faceSize.x, .z=faceSize.x*faceSize.y, .w=faceSize.x*faceSize.y*faceSize.z}};
	const index_t flatFacePos = flattenIndex(facePos, faceStride);
	const TransformGPU<scalar_t, DIMS>* p_T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_block->faceTransform ) + flatFacePos;
	return p_T;
}

template <typename scalar_t, int DIMS>
__device__ inline
const TransformGPU<scalar_t, DIMS>* getFaceTransformPtr(const I4 &facePos, const index_t face, const BlockGPU<scalar_t> &block){
	return getFaceTransformPtr<scalar_t, DIMS>(facePos, face, &block);
}

template <typename scalar_t, int DIMS>
__device__ inline
Vector<scalar_t,DIMS> lerpHalfpoint(const Vector<scalar_t,DIMS> &v1, const Vector<scalar_t,DIMS> &v2){
	return (v1 + v2)*static_cast<scalar_t>(0.5);
}

template <typename scalar_t, int DIMS>
__device__ inline
Vector<scalar_t,DIMS> slerp1Halfpoint(const Vector<scalar_t,DIMS> &v1, const Vector<scalar_t,DIMS> &v2, const scalar_t eps){
	// normalize vectors
	const scalar_t sqlen1 = dot(v1,v1);
	const scalar_t sqlen2 = dot(v2,v2);
	if(sqlen1<eps || sqlen2<eps){
		// vector to short to normalize, default to lerp
		return lerpHalfpoint(v1,v2);
	}
	const Vector<scalar_t,DIMS> n1 = v1*rsqrt(sqlen1);
	const Vector<scalar_t,DIMS> n2 = v2*rsqrt(sqlen1);
	
	// get angle between vectors
	const scalar_t d = dot(n1, n2);
	if(d<(eps-1) || (1-eps)<d){
		return lerpHalfpoint(v1,v2);
	}
	const scalar_t rad = acos(d);
	if(rad<eps){
		return lerpHalfpoint(v1,v2);
	}
	
	// slerp direction and magnitude
	const scalar_t w = sin(0.5*rad) / sin(rad);
	const Vector<scalar_t,DIMS> v = (v1 + v2)*w;
	
	return v;
}

template <typename scalar_t, int DIMS>
__device__ inline
Vector<scalar_t,DIMS> slerp2Halfpoint(const Vector<scalar_t,DIMS> &v1, const Vector<scalar_t,DIMS> &v2, const scalar_t eps){
	// normalize vectors
	const scalar_t sqlen1 = dot(v1,v1);
	const scalar_t sqlen2 = dot(v2,v2);
	if(sqlen1<eps || sqlen2<eps){
		// vector to short to normalize, default to lerp
		return lerpHalfpoint(v1,v2);
	}
	const Vector<scalar_t,DIMS> n1 = v1*rsqrt(sqlen1);
	const Vector<scalar_t,DIMS> n2 = v2*rsqrt(sqlen1);
	
	// get angle between vectors
	// halfpoint simplification
	Vector<scalar_t,DIMS> n = lerpHalfpoint(n1, n2);
	const scalar_t sqlen = dot(n,n);
	if(sqlen<eps){
		// vector to short to normalize, default to lerp
		return lerpHalfpoint(v1,v2);
	}
	n *= rsqrt(sqlen);
	
	// lerp magnitude
	const scalar_t m1 = sqrt(sqlen1);
	const scalar_t m2 = sqrt(sqlen2);
	const scalar_t m = (m1 + m2)*0.5;
	
	const Vector<scalar_t,DIMS> v = n*m;
	
	return v;
}

template <typename scalar_t, int DIMS>
__device__ inline
scalar_t getSlerpWeight(const Vector<scalar_t,DIMS> &v1, const Vector<scalar_t,DIMS> &v2, const scalar_t eps){
	
	// normalize vectors
	const scalar_t sqlen1 = dot(v1,v1);
	const scalar_t sqlen2 = dot(v2,v2);
	if(sqlen1<eps || sqlen2<eps){
		// vector to short to normalize, default to lerp
		return 0.5;
	}
	const Vector<scalar_t,DIMS> n1 = v1*rsqrt(sqlen1);
	const Vector<scalar_t,DIMS> n2 = v2*rsqrt(sqlen1);
	
	// get angle between vectors
	const scalar_t rad = acos(dot(n1, n2));
	if(rad<eps){
		return 0.5;
	}
	
	const scalar_t weight = sin(0.5*rad) / sin(rad);
	
	return weight;
}

template <typename scalar_t, int DIMS>
__device__ inline Vector<scalar_t,DIMS> getPressureGradientFVM(const BlockGPU<scalar_t> &block, I4 pos, const DomainGPU<scalar_t> &domain,
		const index_t gradientInterpolation){
	// use a finite volume approach (instead of finite differencing)
	// interpolate pressure at the faces, multiply with face area and face normal
	// face area * face normal = row of inverse transform
	// identical to finite difference gradient for orthogonal grids/transforms
	if(!block.hasTransform){ return getPressureGradient<scalar_t, DIMS>(block, pos, domain);}
	

	Vector<scalar_t,DIMS> pressureGrad = {.a={0}};

	pos.w = 0;
	const index_t flatIndexP = flattenIndex(pos, block);
	const TransformGPU<scalar_t, DIMS> *p_TP = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(block.transform) + flatIndexP;
	const scalar_t pP = block.pressure[flatIndexP];
	

	for(index_t face=0; face<(DIMS*2); ++face){
		const index_t dim = axisFromBound(face); //face>>1;
		const bool isUpper = boundIsUpper(face); // face&1;
		const index_t faceSign = faceSignFromBound(face); //(face&1) * 2 - 1; // -1 if lower, +1 if upper
		const bool atBound = isAtBound(pos, face, &block);
		
		const Vector<scalar_t, DIMS> fluxP = p_TP->Minv.v[dim] * (p_TP->det * pP);
		
		
		I4 tempPos = pos;
		const TransformGPU<scalar_t, DIMS> *p_T = nullptr;
		index_t axisNeighbor = dim; // needed for possible axis shuffling over CONNECTED_GRID boundaries.
		scalar_t transformSign = 1;
		scalar_t p = 0;
		if(atBound){ //((pos.a[dim]==0 && !isUpper) || (pos.a[dim]==(block.size.a[dim]-1)) && isUpper)){
			switch(block.boundaries[face].type){
				case BoundaryType::VALUE:
					// static boundary with transforms should never happen, but p_T = p_TP is a reasonable fallback.
					p_T = p_TP;
					tempPos.a[dim] = pos.a[dim] - faceSign;
					p = 2*pP - block.pressure[flattenIndex(tempPos, block)];
					break;
				case BoundaryType::DIRICHLET_VARYING:
				case BoundaryType::FIXED:
				{
					// boundary is closed, so pressure gradient is 0.
					tempPos.a[dim] = 0;
					if(block.boundaries[face].type==BoundaryType::DIRICHLET_VARYING){
						p_T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(block.boundaries[face].vdb.transform) + flattenIndex(tempPos, block.boundaries[face].vdb.stride);
					} else {
						p_T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(block.boundaries[face].fb.transform) + flattenIndex(tempPos, block.boundaries[face].fb.stride);
					}
					//p = pP;  //using center pressure (extrapolation with p-grad=0) leads to oscillations at the boundary.
					// instead, extrapolate using the pressure gradient at the opposite side. Minimum block resolution requirements mean that there can't be a boundary.
					tempPos.a[dim] = pos.a[dim] - faceSign;
					if(gradientInterpolation!=3){
						p = pP + (pP - block.pressure[flattenIndex(tempPos, block)]) *0.5;
						const Vector<scalar_t, DIMS> fluxN = p_T->Minv.v[dim] * (p_T->det * p);
						pressureGrad += fluxN * static_cast<scalar_t>(faceSign);// * boundaryWeights[dim]);
						
						continue;
					}else{
						p = pP + (pP - block.pressure[flattenIndex(tempPos, block)]);
					}
					break;
				}
				case BoundaryType::GRADIENT:
					// unsupported
					continue;
				case BoundaryType::CONNECTED_GRID:
				{
					const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + block.boundaries[face].cb.connectedGridIndex;
					tempPos.w = dim;
					I4 otherPos = computeConnectedPosWithChannel(tempPos, dim, &block.boundaries[face].cb, domain);
					axisNeighbor = otherPos.w; // for axis shuffling
					const bool otherIsUpper = boundIsUpper(block.boundaries[face].cb.axes.a[0]);
					if(otherIsUpper==isUpper) {
						transformSign = -1;
					}
					otherPos.w = 0;
					const index_t flatIndex = flattenIndex(otherPos, p_connectedBlock);
					p_T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_connectedBlock->transform) + flatIndex;
					p = p_connectedBlock->pressure[flatIndex];
					break;
				}
				case BoundaryType::PERIODIC:
				{
					// compute flux to cell on other side
					// special case of connection to another block
					tempPos.a[dim] = isUpper ? 0 : block.size.a[dim] - 1;
					const index_t flatIndex = flattenIndex(tempPos, block);
					p_T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(block.transform) + flatIndex;
					p = block.pressure[flatIndex];
					break;
				}
				default:
					continue;
			}
		} else {
			tempPos.a[dim] += faceSign;
			const index_t flatIndex = flattenIndex(tempPos, block);
			p_T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(block.transform) + flatIndex;
			p = block.pressure[flatIndex];
		}
		
		// version more like the flux computation. computes the fluxes of the individual cells first before interpolating to the face.
		const Vector<scalar_t, DIMS> fluxN = p_T->Minv.v[axisNeighbor] * (p_T->det * p * transformSign);
		switch(gradientInterpolation){
			case 0:
			{ // lerp
				pressureGrad += (fluxP + fluxN) * (static_cast<scalar_t>(0.5) * faceSign);
				break;
			}
			case 1:
			{ // slerp direction and magnitude
				pressureGrad += slerp1Halfpoint(fluxP, fluxN, static_cast<scalar_t>(1e-5)) * static_cast<scalar_t>(faceSign);
				break;
			}
			case 2:
			{ // slerp direction, lerp magnitude
				pressureGrad += slerp2Halfpoint(fluxP, fluxN, static_cast<scalar_t>(1e-5)) * static_cast<scalar_t>(faceSign);
				break;
			}
			case 3:
			{
				p_T = getFaceTransformPtr<scalar_t, DIMS>(pos, face, block);
				pressureGrad += p_T->Minv.v[dim] * (p_T->det * (p + pP) * static_cast<scalar_t>(0.5) * static_cast<scalar_t>(faceSign));
				break;
			}
			default:
				break;
		}
		//pressureGrad += (fluxP + fluxN) * (static_cast<scalar_t>(1e-5)0.5) * faceSign);// * boundaryWeights[dim]);
	}
	
	pressureGrad *= static_cast<scalar_t>(1.0) / (p_TP->det);// * detWeight);
	
	return pressureGrad;
}

template <typename scalar_t, int DIMS>
__device__ inline scalar_t getDeterminant(const BlockGPU<scalar_t> &block, I4 pos){
	if(block.hasTransform){
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(block.transform) + flattenIndex(pos, block);
		const scalar_t det = T->det;
		return det;
	} else {
		return 1;
	}
}

template <typename scalar_t>
__device__ inline scalar_t getDeterminantDimSwitch(const BlockGPU<scalar_t> &block, const I4 pos, const index_t nDims){
	switch(nDims){
	case 1:
		return getDeterminant<scalar_t, 1>(block, pos);
	case 2:
		return getDeterminant<scalar_t, 2>(block, pos);
	case 3:
		return getDeterminant<scalar_t, 3>(block, pos);
	default:
		return 1;
	}
}

template <typename scalar_t, int DIMS>
__device__ inline scalar_t getDeterminant(const BlockGPU<scalar_t> *p_block, I4 pos){
	if(p_block->hasTransform){
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_block->transform) + flattenIndex(pos, p_block);
		const scalar_t det = T->det;
		return det;
	} else {
		return 1;
	}
}

template <typename scalar_t>
__device__ inline scalar_t getDeterminantDimSwitch(const BlockGPU<scalar_t> *p_block, const I4 pos, const index_t nDims){
	switch(nDims){
	case 1:
		return getDeterminant<scalar_t, 1>(p_block, pos);
	case 2:
		return getDeterminant<scalar_t, 2>(p_block, pos);
	case 3:
		return getDeterminant<scalar_t, 3>(p_block, pos);
	default:
		return 1;
	}
}

template<typename scalar_t, int DIMS>
__device__ inline scalar_t getTransformMetricOrthogonal(I4 pos, const BlockGPU<scalar_t> *p_block){
	if(p_block->hasTransform){
		const index_t component = pos.w;
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_block->transform) + flattenIndex(pos, p_block);
		return T->Minv.a[component][component];
		
	}else{
		return 1;
	}
}

template <typename scalar_t>
__device__ inline scalar_t getTransformMetricOrthogonalDimSwitch(const I4 pos, const BlockGPU<scalar_t> *p_block, const index_t nDims){
	switch(nDims){
	case 1:
		return getTransformMetricOrthogonal<scalar_t, 1>(pos, p_block);
	case 2:
		return getTransformMetricOrthogonal<scalar_t, 2>(pos, p_block);
	case 3:
		return getTransformMetricOrthogonal<scalar_t, 3>(pos, p_block);
	default:
		return 1;
	}
}


template<typename scalar_t, int DIMS>
__device__ inline scalar_t getLaplaceCoefficientOrthogonal(I4 pos, const BlockGPU<scalar_t> *p_block){
	if(p_block->hasTransform){
		const index_t component = pos.w;
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_block->transform) + flattenIndex(pos, p_block);
		const scalar_t det = T->det;
		//load more globally as it is needed for all sides?
		//const MatrixSquare<scalar_t, 3> mInv = T->Minv;
		const Vector<scalar_t, DIMS> mInvRow = T->Minv.v[component];
		return det*dot(mInvRow, mInvRow);
		
	}else{
		return 1;
	}
}

template<typename scalar_t, int DIMS>
__device__ inline scalar_t getLaplaceCoefficientOrthogonalFace(I4 pos, const index_t face, const BlockGPU<scalar_t> *p_block){
	if(p_block->hasFaceTransform){
		const index_t component = axisFromBound(face);//pos.w;
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = getFaceTransformPtr<scalar_t, DIMS>(pos, face, p_block);
		const scalar_t det = T->det;
		//load more globally as it is needed for all sides?
		//const MatrixSquare<scalar_t, 3> mInv = T->Minv;
		const Vector<scalar_t, DIMS> mInvRow = T->Minv.v[component];
		return det*dot(mInvRow, mInvRow);
		
	}else{
		return 1;
	}
}

template<typename scalar_t, int DIMS>
__device__ inline scalar_t getLaplaceCoefficient(I4 pos, const index_t component1, const index_t component2, const BlockGPU<scalar_t> *p_block){
	if(p_block->hasTransform){
		//const index_t component = pos.w;
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_block->transform) + flattenIndex(pos, p_block);
		const scalar_t det = T->det;
		//load more globally as it is needed for all sides?
		//const MatrixSquare<scalar_t, 3> mInv = T->Minv;
		const Vector<scalar_t, DIMS> mInvRow1 = T->Minv.v[component1];
		const Vector<scalar_t, DIMS> mInvRow2 = T->Minv.v[component2];
		return det*dot(mInvRow1, mInvRow2);
		
	}else{
		return 1;
	}
}

__device__ constexpr index_t getLaplaceCoefficientCenterIndex(const index_t component1, const index_t component2){
	return component1 + component2 + ((component1>>1) | (component2>>1));
}

__device__ constexpr index_t getLaplaceCoefficientsFullLength(const index_t dims){
	return dims * (dims+1) / 2; // gauss sum
}

template<typename scalar_t, int DIMS>
__device__ inline Vector<scalar_t, getLaplaceCoefficientsFullLength(DIMS)> getLaplaceCoefficientsFull(I4 pos, const BlockGPU<scalar_t> *p_block){
	// 3D: all 6 coefficients
	if(p_block->hasTransform){
		//const index_t component = pos.w;
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_block->transform) + flattenIndex(pos, p_block);
		const scalar_t det = T->det;
		
		Vector<scalar_t, getLaplaceCoefficientsFullLength(DIMS)> coefficients;
		for(index_t c1=0; c1<DIMS; ++c1){
			for(index_t c2=0; c2<=c1; ++c2){
				coefficients.a[getLaplaceCoefficientCenterIndex(c1,c2)] = det*dot(T->Minv.v[c1], T->Minv.v[c2]);
			}
		}
		return coefficients;
		
	}else{
		Vector<scalar_t, DIMS * (DIMS+1) / 2> coefficients;
		for(index_t c1=0; c1<DIMS; ++c1){
			for(index_t c2=0; c2<=c1; ++c2){
				coefficients.a[getLaplaceCoefficientCenterIndex(c1,c2)] = c1==c2 ? 1 : 0;
			}
		}
		return coefficients;
	}
}

template<typename scalar_t, int DIMS>
__device__ inline Vector<scalar_t, DIMS> getLaplaceCoefficientsSingleAxis(I4 pos, const index_t axis, const BlockGPU<scalar_t> *p_block){
	// 3D: 3 coefficients: neighbour direction x all
	if(p_block->hasTransform){
		//const index_t component = pos.w;
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_block->transform) + flattenIndex(pos, p_block);
		const scalar_t det = T->det;
		
		Vector<scalar_t, DIMS> coefficients;
		for(index_t c1=0; c1<DIMS; ++c1){
			coefficients.a[c1] = det*dot(T->Minv.v[c1], T->Minv.v[axis]);
		}
		return coefficients;
		
	}else{
		Vector<scalar_t, DIMS> coefficients;
		for(index_t c1=0; c1<DIMS; ++c1){
			coefficients.a[c1] = c1==axis ? 1 : 0;
		}
		return coefficients;
	}
}

template<typename scalar_t, int DIMS>
__device__ inline Vector<scalar_t, DIMS> getLaplaceCoefficientsSingleFace(I4 pos, const index_t face, const BlockGPU<scalar_t> *p_block){
	// 3D: 3 coefficients: neighbour direction x all
	const index_t axis = axisFromBound(face);
	if(p_block->hasTransform){
		//const index_t component = pos.w;
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = getFaceTransformPtr<scalar_t, DIMS>(pos, face, p_block);
		const scalar_t det = T->det;
		
		Vector<scalar_t, DIMS> coefficients;
		for(index_t c1=0; c1<DIMS; ++c1){
			coefficients.a[c1] = det*dot(T->Minv.v[c1], T->Minv.v[axis]);
		}
		return coefficients;
		
	}else{
		Vector<scalar_t, DIMS> coefficients;
		for(index_t c1=0; c1<DIMS; ++c1){
			coefficients.a[c1] = c1==axis ? 1 : 0;
		}
		return coefficients;
	}
}

/** go from current position 'pos' in 'block' to the neighbor cell in direction/face 'dir'.
 * Compute the laplace coefficients that contain the axis of 'dir'.
 * If there is a prescribed boundary in direction 'dir', its transformation is returned.
 * If there is a connected block with shuffled axes, the connection is resolved s.t. the laplace coefficients are valid for 'block'.
 * If the connection (block or boundary) has no transformation, laplace coefficients based on the identity transformation are returned.
 * 
 */
template<typename scalar_t, int DIMS>
__device__ inline Vector<scalar_t, DIMS> getLaplaceCoefficientsSingleAxisNeighbor(I4 pos, const index_t dir, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain){
	
	const index_t axis = axisFromBound(dir);
	pos.w = 0;
	NeighborCellInfo<scalar_t> cellInfo;
	const TransformGPU<scalar_t, DIMS> *p_T = nullptr;
	
	if(isAtBound(pos, dir, block) && isEmptyBound(dir, block.boundaries) && block.boundaries[dir].type!=BoundaryType::FIXED){
		switch(block.boundaries[dir].type){
			case BoundaryType::DIRICHLET_VARYING:
			{
				const VaryingDirichletBoundaryGPU<scalar_t> *p_bound = &(block.boundaries[dir].vdb);
				if(p_bound->hasTransform){
					cellInfo.cell.isBlock = false;
					initDefaultAxisMapping(cellInfo);
					pos.a[axis] = 0;
					p_T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_bound->transform) + flattenIndex(pos, p_bound->stride);
				}
			}
			default:
				p_T = nullptr;
				break;
		}
	} else {
		cellInfo = resolveNeighborCell<scalar_t>(pos, dir, &block, domain);
		cellInfo.cell.pos.w = 0; // might have been changed by resolveNeighborCell
		if(cellInfo.cell.isBlock){
			if(cellInfo.cell.p_block->hasTransform){
				p_T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(cellInfo.cell.p_block->transform)
					+ flattenIndex(cellInfo.cell.pos, cellInfo.cell.p_block);
			}
		} else {
			if(cellInfo.cell.p_bound->hasTransform){
				p_T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(cellInfo.cell.p_bound->transform)
					+ flattenIndex(cellInfo.cell.pos, cellInfo.cell.p_bound->stride);
			}
		}
		
	}
	
	if(p_T!=nullptr){
		const scalar_t det = p_T->det;
		
		Vector<scalar_t, DIMS> coefficients;
		const index_t axisMapped = axisFromBound(cellInfo.axisMapping.a[axis]); // identity in most cases
		const Vector<scalar_t, DIMS> metricMapped = p_T->Minv.v[axisMapped];
		for(index_t c1=0; c1<DIMS; ++c1){
			const index_t c1Mapped = axisFromBound(cellInfo.axisMapping.a[c1]);
			coefficients.a[c1] = det*dot(p_T->Minv.v[c1Mapped], metricMapped);
		}
		return coefficients;
		
	}else{
		Vector<scalar_t, DIMS> coefficients;
		// TODO: shuffle for connected grid?
		for(index_t c1=0; c1<DIMS; ++c1){
			coefficients.a[c1] = c1==axis ? 1 : 0;
		}
		return coefficients;
	}
}
template<typename scalar_t, int DIMS>
__device__ inline Vector<scalar_t, DIMS> getLaplaceCoefficientsSingleAxisNeighbor(const NeighborCellInfo<scalar_t> &cellInfo, const index_t dir, const DomainGPU<scalar_t> &domain){
	
	const index_t axis = axisFromBound(dir);
	const TransformGPU<scalar_t, DIMS> *p_T = nullptr;
	
	I4 pos = cellInfo.cell.pos;
	pos.w = 0;
	if(cellInfo.cell.isBlock){
		if(cellInfo.cell.p_block->hasTransform){
			p_T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(cellInfo.cell.p_block->transform)
				+ flattenIndex(pos, cellInfo.cell.p_block);
		}
	} else {
		if(cellInfo.cell.p_bound->hasTransform){
			p_T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(cellInfo.cell.p_bound->transform)
				+ flattenIndex(pos, cellInfo.cell.p_bound->stride);
		}
	}
		
	
	
	Vector<scalar_t, DIMS> coefficients = {.a={0}};
	if(p_T!=nullptr){
		const scalar_t det = p_T->det;
		const index_t axisMapped = axisFromBound(cellInfo.axisMapping.a[axis]); // identity in most cases
		const Vector<scalar_t, DIMS> metricMapped = p_T->Minv.v[axisMapped];
		for(index_t c1=0; c1<DIMS; ++c1){
			const index_t c1Mapped = axisFromBound(cellInfo.axisMapping.a[c1]);
			coefficients.a[c1] = det*dot(p_T->Minv.v[c1Mapped], metricMapped);
		}
		
	}else{
		coefficients.a[axis] = 1;
	}
	return coefficients;
}

template<typename scalar_t, int DIMS, typename BOUNDARY_T>
__device__ inline scalar_t getLaplaceCoefficientOrthogonalBoundary(I4 pos, const BOUNDARY_T *p_bound){
	if(p_bound->hasTransform){
		const index_t component = pos.w;
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_bound->transform) + flattenIndex(pos, p_bound->stride);
		const scalar_t det = T->det;
		const Vector<scalar_t, DIMS> mInvRow = T->Minv.v[component];
		return det*dot(mInvRow, mInvRow);

	}else{
		return 1;
	}
}

// Returns the face area magnitude |S_f| = det_f * |Minv[dim]| for a boundary face.
// Used for Neumann BC diffusion flux: kappa * (dphi/dn) * |S_f|.
// Note: getLaplaceCoefficientOrthogonalBoundary returns det_f * |Minv[dim]|^2 = |S_f|^2 / det_f,
// while this returns |S_f| = sqrt(det_f * alpha).
template<typename scalar_t, int DIMS, typename BOUNDARY_T>
__device__ inline scalar_t getFaceAreaOrthogonalBoundary(I4 pos, const BOUNDARY_T *p_bound){
	if(p_bound->hasTransform){
		const index_t component = pos.w;
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_bound->transform) + flattenIndex(pos, p_bound->stride);
		const scalar_t det = T->det;
		const Vector<scalar_t, DIMS> mInvRow = T->Minv.v[component];
		return det * sqrt(dot(mInvRow, mInvRow));
	}else{
		return 1;
	}
}

template<typename scalar_t, int DIMS, typename BOUNDARY_T>
__device__ inline Vector<scalar_t, DIMS> getLaplaceCoefficientsNeighbourBoundary(I4 pos, const index_t axis, const BOUNDARY_T *p_bound){
	// 3D: 3 coefficients: neighbour direction x all
	if(p_bound->hasTransform){
		//const index_t component = pos.w;
		pos.w = 0;
		pos.a[axis] = 0; //assuming p_bound is on axis
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_bound->transform) + flattenIndex(pos, p_bound->stride);
		const scalar_t det = T->det;
		
		Vector<scalar_t, DIMS> coefficients;
		for(index_t c1=0; c1<DIMS; ++c1){
			coefficients.a[c1] = det*dot(T->Minv.v[c1], T->Minv.v[axis]);
		}
		return coefficients;
		
	}else{
		Vector<scalar_t, DIMS> coefficients;
		for(index_t c1=0; c1<DIMS; ++c1){
			coefficients.a[c1] = c1==axis ? 1 : 0;
		}
		return coefficients;
	}
}

template <typename scalar_t>
__device__ inline scalar_t getLaplaceCoefficientOrthogonalDimSwitch(const I4 pos, const BlockGPU<scalar_t> *p_block, const index_t nDims){
	switch(nDims){
	case 1:
		return getLaplaceCoefficientOrthogonal<scalar_t, 1>(pos, p_block);
	case 2:
		return getLaplaceCoefficientOrthogonal<scalar_t, 2>(pos, p_block);
	case 3:
		return getLaplaceCoefficientOrthogonal<scalar_t, 3>(pos, p_block);
	default:
		return 1;
	}
}

// ||M^{-T}[normalAxis]|| — norm of the normal row of Minv.
// Used for computing surface Laplace coefficients in the thin-wall Robin BC.
template<typename scalar_t, int DIMS>
__device__ inline scalar_t getMinvNorm(I4 pos, const index_t normalAxis, const BlockGPU<scalar_t> *p_block){
	if(!p_block->hasTransform) return static_cast<scalar_t>(1);
	pos.w = 0;
	const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<const TransformGPU<scalar_t, DIMS>*>(p_block->transform) + flattenIndex(pos, p_block);
	scalar_t norm_sq = 0;
	for(int d = 0; d < DIMS; ++d)
		norm_sq += T->Minv.v[normalAxis].a[d] * T->Minv.v[normalAxis].a[d];
	return sqrt(norm_sq);
}

template<typename scalar_t>
__device__ inline scalar_t getMinvNormDimSwitch(I4 pos, const index_t normalAxis, const BlockGPU<scalar_t> *p_block, const index_t nDims){
	switch(nDims){
	case 1: return getMinvNorm<scalar_t, 1>(pos, normalAxis, p_block);
	case 2: return getMinvNorm<scalar_t, 2>(pos, normalAxis, p_block);
	case 3: return getMinvNorm<scalar_t, 3>(pos, normalAxis, p_block);
	default: return static_cast<scalar_t>(1);
	}
}

template <typename scalar_t>
__device__ inline scalar_t getLaplaceCoefficientOrthogonalBoundaryVaryingDimSwitch(const I4 pos, const VaryingDirichletBoundaryGPU<scalar_t> *p_bound, const index_t nDims){
	switch(nDims){
	case 1:
		return getLaplaceCoefficientOrthogonalBoundary<scalar_t, 1, VaryingDirichletBoundaryGPU<scalar_t>>(pos, p_bound);
	case 2:
		return getLaplaceCoefficientOrthogonalBoundary<scalar_t, 2, VaryingDirichletBoundaryGPU<scalar_t>>(pos, p_bound);
	case 3:
		return getLaplaceCoefficientOrthogonalBoundary<scalar_t, 3, VaryingDirichletBoundaryGPU<scalar_t>>(pos, p_bound);
	default:
		return 1;
	}
}
template <typename scalar_t>
__device__ inline scalar_t getLaplaceCoefficientOrthogonalBoundaryFixedDimSwitch(const I4 pos, const FixedBoundaryGPU<scalar_t> *p_bound, const index_t nDims){
	switch(nDims){
	case 1:
		return getLaplaceCoefficientOrthogonalBoundary<scalar_t, 1, FixedBoundaryGPU<scalar_t>>(pos, p_bound);
	case 2:
		return getLaplaceCoefficientOrthogonalBoundary<scalar_t, 2, FixedBoundaryGPU<scalar_t>>(pos, p_bound);
	case 3:
		return getLaplaceCoefficientOrthogonalBoundary<scalar_t, 3, FixedBoundaryGPU<scalar_t>>(pos, p_bound);
	default:
		return 1;
	}
}

template <typename scalar_t>
__device__ inline scalar_t getFaceAreaOrthogonalBoundaryFixedDimSwitch(const I4 pos, const FixedBoundaryGPU<scalar_t> *p_bound, const index_t nDims){
	switch(nDims){
	case 1:
		return getFaceAreaOrthogonalBoundary<scalar_t, 1, FixedBoundaryGPU<scalar_t>>(pos, p_bound);
	case 2:
		return getFaceAreaOrthogonalBoundary<scalar_t, 2, FixedBoundaryGPU<scalar_t>>(pos, p_bound);
	case 3:
		return getFaceAreaOrthogonalBoundary<scalar_t, 3, FixedBoundaryGPU<scalar_t>>(pos, p_bound);
	default:
		return 1;
	}
}

template <typename scalar_t>
__device__ inline scalar_t getLaplaceCoefficientOrthogonalFaceDimSwitch(const I4 &pos, const index_t face, const BlockGPU<scalar_t> *p_block, const index_t nDims){
	switch(nDims){
	case 1:
		return getLaplaceCoefficientOrthogonalFace<scalar_t, 1>(pos, face, p_block);
	case 2:
		return getLaplaceCoefficientOrthogonalFace<scalar_t, 2>(pos, face, p_block);
	case 3:
		return getLaplaceCoefficientOrthogonalFace<scalar_t, 3>(pos, face, p_block);
	default:
		return 1;
	}
}


/**
 * Compute the advective fluxes for all faces of the cell at 'pos'. The fluxes are NOT multiplied by the face sign.
 */
template <typename scalar_t>
__device__ void computeFluxesNDLoop(const I4 pos, scalar_t* fluxes, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain, const scalar_t *velocityGlobal){

	for(index_t bound=0; bound<(domain.numDims*2); ++bound){
		const index_t dim = axisFromBound(bound);
		const index_t isUpper = boundIsUpper(bound);
		const bool atBound = isAtBound(pos, bound, &block);

		I4 tempPos = pos;
		tempPos.w = dim;

		const scalar_t velC = getContravariantComponentDimSwitch(tempPos, &block, domain, velocityGlobal);

		if(atBound){// lower boundary
			switch(block.boundaries[bound].type){
			case BoundaryType::DIRICHLET:
				// enforce flux
				fluxes[bound] = block.boundaries[bound].sdb.velocity.a[dim];
				break;
			case BoundaryType::DIRICHLET_VARYING:
				// enforce flux
			{
				tempPos.a[dim] = 0;
				fluxes[bound] = getContravariantComponentBoundaryVaryingDimSwitch(tempPos, &block.boundaries[bound].vdb, domain);
				break;
			}
			case BoundaryType::FIXED:
				// enforce flux
			{
				tempPos.a[dim] = 0;
				fluxes[bound] = getContravariantComponentBoundaryFixedDimSwitch(tempPos, &block.boundaries[bound].fb, domain);
				break;
			}
			case BoundaryType::GRADIENT:
				// compute flux only from center cell?
				// velN = velC - grad*distance*2 & flux = (velN+velC)*0.5 ->
				fluxes[bound] = 0; //velC + block.boundaries[bound].snb.boundaryGradient.a[dim] * 0.5; //* distance
				break;
			case BoundaryType::CONNECTED_GRID:
			{
				// handle multi-block grids, load from correct cell of the connected grid
				const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
				I4 otherPos = computeConnectedPos(tempPos, dim, &block.boundaries[bound].cb, domain);
				// tempPos.w is always dim, so connectedAxis==0. otherwise use computeConnectedPosWithChannel()
				otherPos.w = block.boundaries[bound].cb.axes.a[0]>>1;
				scalar_t velN = getContravariantComponentDimSwitch(otherPos, p_connectedBlock, domain, velocityGlobal);
				
				// if the connection goes upper to upper or lower to lower, the velocity has to be inverted
				// (if it goes upper->lower or lower->upper it should be fine)
				//const bool isUpper = boundIsUpper(bound);
				const bool otherIsUpper = boundIsUpper(block.boundaries[bound].cb.axes.a[0]);
				if(otherIsUpper==isUpper) {
					// connected to a boundary of the same side (lower/upper), need to invert its flux to be consistent
					velN = -velN;
				}
				fluxes[bound] = (velN + velC) * 0.5f;
				break;
			}
			case BoundaryType::PERIODIC:
			{
				// compute flux to cell on other side
				// special case of connection to another block
				tempPos.a[dim] = isUpper ? 0 : block.size.a[dim] - 1;
				const scalar_t velN = getContravariantComponentDimSwitch(tempPos, &block, domain, velocityGlobal);
				fluxes[bound] = (velN + velC) * 0.5f;
				break;
			}
			default:
				fluxes[bound] = 0;
				break;
			}
		}else{
			tempPos.a[dim] = pos.a[dim] + faceSignFromBound(bound);
			const scalar_t velN = getContravariantComponentDimSwitch(tempPos, &block, domain, velocityGlobal);
			fluxes[bound] = (velN + velC) * 0.5f;
		}
	}

}

/**
 * Compute the u×B face fluxes for the electric-potential Poisson RHS div(u×B).
 *
 * Identical to computeFluxesNDLoop for interior, CONNECTED and PERIODIC faces.
 *
 * At an *open* prescribed boundary (velocity NEUMANN, e.g. an outflow) it uses the
 * cell-centre normal flux velC instead of dropping it. This makes the RHS boundary flux
 * match j_b = (u×B)_n as set by computeCurrentDensityFaceBased, realising ∂φ/∂n=0 and
 * keeping div(j)=0 discretely at boundary cells (current is free to exit the domain,
 * closing virtually outside). Dropping the flux there would instead source spurious
 * Lorentz loops at the outflow whenever (u×B)_n ≠ 0.
 *
 * At a *solid wall* (velocity DIRICHLET) the flux is dropped, giving the insulating
 * j_n = 0 the physics requires: a wall is not a place current can leave through. Since
 * u_wall = 0 the true (u×B)_n|_wall vanishes anyway, so dropping is exact here and stays
 * consistent with both the homogeneous-Neumann Epot matrix and the reconstruction.
 * Both cases are BoundaryType::FIXED, so the velocity BC does the discriminating.
 */
template <typename scalar_t>
__device__ void computeEpotFluxesNDLoop(const I4 pos, scalar_t* fluxes, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain, const scalar_t *vectorField){

	for(index_t bound=0; bound<(domain.numDims*2); ++bound){
		const index_t dim = axisFromBound(bound);
		const index_t isUpper = boundIsUpper(bound);
		const bool atBound = isAtBound(pos, bound, &block);

		I4 tempPos = pos;
		tempPos.w = dim;

		const scalar_t velC = getContravariantComponentDimSwitch(tempPos, &block, domain, vectorField);

		if(atBound){
			switch(block.boundaries[bound].type){
			case BoundaryType::DIRICHLET:
			case BoundaryType::DIRICHLET_VARYING:
			case BoundaryType::GRADIENT:
			case BoundaryType::FIXED:
				// Solid wall: insulating, j_n = 0 -> drop the flux.
				// Open bound: ∂φ/∂n=0, so j_b = (u×B)_n evaluated at the cell centre
				// (zero-gradient extrapolation), matching computeCurrentDensityFaceBased.
				// Prescribed current I into the fluid: the flux along +axis is +I through a
				// lower face and -I through an upper one, again as in the reconstruction.
				if(isEpotCurrentBound(pos, bound, block.boundaries)){
					const scalar_t current = potentialValueAt(pos, bound, block.boundaries);
					fluxes[bound] = isUpper ? -current : current;
				} else {
					fluxes[bound] = isInsulatingWallBound(pos, bound, block.boundaries) ? 0 : velC;
				}
				break;
			case BoundaryType::CONNECTED_GRID:
			{
				const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
				I4 otherPos = computeConnectedPos(tempPos, dim, &block.boundaries[bound].cb, domain);
				otherPos.w = block.boundaries[bound].cb.axes.a[0]>>1;
				scalar_t velN = getContravariantComponentDimSwitch(otherPos, p_connectedBlock, domain, vectorField);
				const bool otherIsUpper = boundIsUpper(block.boundaries[bound].cb.axes.a[0]);
				if(otherIsUpper==isUpper) velN = -velN;
				fluxes[bound] = (velN + velC) * 0.5f;
				break;
			}
			case BoundaryType::PERIODIC:
			{
				tempPos.a[dim] = isUpper ? 0 : block.size.a[dim] - 1;
				const scalar_t velN = getContravariantComponentDimSwitch(tempPos, &block, domain, vectorField);
				fluxes[bound] = (velN + velC) * 0.5f;
				break;
			}
			default:
				fluxes[bound] = 0;
				break;
			}
		}else{
			tempPos.a[dim] = pos.a[dim] + faceSignFromBound(bound);
			const scalar_t velN = getContravariantComponentDimSwitch(tempPos, &block, domain, vectorField);
			fluxes[bound] = (velN + velC) * 0.5f;
		}
	}

}

#ifdef WITH_GRAD


template <typename scalar_t>
__device__ void ScatterFluxesGradNDLoop(const I4 pos, const scalar_t* fluxesGrad, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain, scalar_t *velocityGlobalGrad){
	
	
	for(index_t dim=0; dim<domain.numDims; ++dim) {
		I4 tempPos = pos;
		tempPos.w = dim;
		
		scalar_t velCGrad = 0;
		
		for(index_t isUpper=0; isUpper<2; ++isUpper){
		
			int bound = dim*2 + isUpper;
			const bool atBound = isAtBound(pos, bound, &block);
			//index_t faceSign = -1 + (bound&1)*2;
		
			if(atBound){// lower boundary
				switch(block.boundaries[bound].type){
				case BoundaryType::FIXED:
				{
					tempPos.a[dim] = 0;
					//fluxes[bound] = getContravariantComponentBoundaryFixedDimSwitch(tempPos, &block.boundaries[bound].fb, domain);
					scatterContravariantComponentBoundaryFixedDimSwitch(fluxesGrad[bound], tempPos, &block.boundaries[bound].fb, domain);
					break;
				}
				case BoundaryType::CONNECTED_GRID:
				{
					const bool otherIsUpper = boundIsUpper(block.boundaries[bound].cb.axes.a[0]);
					const scalar_t otherFactor = (otherIsUpper==isUpper) ? -0.5 : 0.5;
					
					const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
					I4 otherPos = computeConnectedPos(tempPos, dim, &block.boundaries[bound].cb, domain);
					otherPos.w = block.boundaries[bound].cb.axes.a[0]>>1;
					
					scatterContravariantComponentDimSwitch(otherFactor *fluxesGrad[bound], otherPos, p_connectedBlock, domain, velocityGlobalGrad);
					velCGrad += 0.5f*fluxesGrad[bound];
					break;
				}
				case BoundaryType::PERIODIC:
				{
					tempPos.a[dim] = isUpper ? 0 : block.size.a[dim] - 1;
					//block.velocity_grad[flattenIndex(tempPos, block)] = 0.5f*fluxesGrad[bound];
					//atomicAdd(block.velocity_grad + flattenIndex(tempPos, block), 0.5f*fluxesGrad[bound]);
					scatterContravariantComponentDimSwitch(0.5f*fluxesGrad[bound], tempPos, &block, domain, velocityGlobalGrad);
					velCGrad += 0.5f*fluxesGrad[bound];
					break;
				}
				default:
					break;
				}
			}else{
				tempPos.a[dim] = pos.a[dim] + faceSignFromBound(bound);
				//block.velocity_grad[flattenIndex(tempPos, block)] = 0.5f*fluxesGrad[bound];
				//atomicAdd(block.velocity_grad + flattenIndex(tempPos, block), 0.5f*fluxesGrad[bound]);
				scatterContravariantComponentDimSwitch(0.5f*fluxesGrad[bound], tempPos, &block, domain, velocityGlobalGrad);
				velCGrad += 0.5f*fluxesGrad[bound];
			}
		}
		
		tempPos.a[dim] = pos.a[dim];
		//block.velocity_grad[flattenIndex(tempPos, block)] = velCGrad;
		//atomicAdd(block.velocity_grad + flattenIndex(tempPos, block), velCGrad);
		scatterContravariantComponentDimSwitch(velCGrad, tempPos, &block, domain, velocityGlobalGrad);
	}
}

// Specialised scatter for ComputeEpotRHSGrad, matching computeEpotFluxesNDLoop.
// At an open prescribed boundary the forward flux is the cell-centre value velC=(u×B)_n,
// so the gradient flows back to the cell-centre vectorField with full weight (accumulated
// into velCGrad below). At a solid wall the forward flux is the constant 0, which carries
// no dependence on vectorField and therefore contributes no gradient.
template <typename scalar_t>
__device__ void ScatterFieldDivGradNDLoop(
		const I4 pos,
		const scalar_t* fluxesGrad,
		const BlockGPU<scalar_t> &block,
		const DomainGPU<scalar_t> &domain,
		scalar_t *fieldGrad)
{
	for(index_t dim = 0; dim < domain.numDims; ++dim) {
		I4 tempPos = pos;
		tempPos.w = dim;
		scalar_t velCGrad = 0;

		for(index_t isUpper = 0; isUpper < 2; ++isUpper) {
			int bound = dim*2 + isUpper;
			const bool atBound = isAtBound(pos, bound, &block);

			if(atBound) {
				switch(block.boundaries[bound].type) {
				case BoundaryType::FIXED:
				case BoundaryType::DIRICHLET:
				case BoundaryType::DIRICHLET_VARYING:
				case BoundaryType::GRADIENT:
					// Open bound: forward flux = velC (cell-centre (u×B)_n) -> full-weight
					// grad to P. Solid wall: forward flux is the constant 0 -> no grad.
					// Prescribed current: the flux is the value, independent of (u×B) ->
					// no grad here (the value's gradient is taken in k_computeEpotRHSGrad)
					if(!isInsulatingWallBound(pos, bound, block.boundaries)
							&& !isEpotCurrentBound(pos, bound, block.boundaries)){
						velCGrad += fluxesGrad[bound];
					}
					break;
				case BoundaryType::CONNECTED_GRID:
				{
					const bool otherIsUpper = boundIsUpper(block.boundaries[bound].cb.axes.a[0]);
					const scalar_t otherFactor = (otherIsUpper == isUpper) ? -0.5 : 0.5;
					const BlockGPU<scalar_t> *p_connectedBlock =
						domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
					I4 otherPos = computeConnectedPos(tempPos, dim,
						&block.boundaries[bound].cb, domain);
					otherPos.w = block.boundaries[bound].cb.axes.a[0] >> 1;
					scatterContravariantComponentDimSwitch(
						otherFactor * fluxesGrad[bound], otherPos,
						p_connectedBlock, domain, fieldGrad);
					velCGrad += 0.5f * fluxesGrad[bound];
					break;
				}
				case BoundaryType::PERIODIC:
				{
					tempPos.a[dim] = isUpper ? 0 : block.size.a[dim] - 1;
					scatterContravariantComponentDimSwitch(
						0.5f * fluxesGrad[bound], tempPos, &block, domain, fieldGrad);
					velCGrad += 0.5f * fluxesGrad[bound];
					break;
				}
				default:
					break;
				}
			} else {
				tempPos.a[dim] = pos.a[dim] + faceSignFromBound(bound);
				scatterContravariantComponentDimSwitch(
					0.5f * fluxesGrad[bound], tempPos, &block, domain, fieldGrad);
				velCGrad += 0.5f * fluxesGrad[bound];
			}
		}

		tempPos.a[dim] = pos.a[dim];
		scatterContravariantComponentDimSwitch(velCGrad, tempPos, &block, domain, fieldGrad);
	}
}

#endif //WITH_GRAD

template <typename scalar_t, index_t DIMS>
__device__
void computeFluxesWithFaceTransforms(const I4 pos, scalar_t* fluxes, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain, const scalar_t *velocityGlobal){
	// UNTESTED!
	for(index_t bound=0; bound<(domain.numDims*2); ++bound){
		const index_t dim = axisFromBound(bound);
		const index_t isUpper = boundIsUpper(bound);
		const bool atBound = isAtBound(pos, bound, &block);
		
		I4 tempPos = pos;
		tempPos.w = dim;
		
		//const scalar_t velC = getContravariantComponentDimSwitch(tempPos, &block, domain, velocityGlobal);

		bool isBoundVel = false;
		Vector<scalar_t,DIMS> velN = {.a={0}};
		
		if(atBound){// lower boundary
			switch(block.boundaries[bound].type){
			case BoundaryType::DIRICHLET:
				// enforce flux
				for(index_t compIdx=0; compIdx<DIMS; ++compIdx){
					velN.a[compIdx] = block.boundaries[bound].sdb.velocity.a[compIdx]; // sdb.velocity always has DIMS==3
				}
				break;
			case BoundaryType::DIRICHLET_VARYING:
				// enforce flux
			{
				tempPos.a[dim] = 0;
				//fluxes[bound] = getContravariantComponentBoundaryVaryingDimSwitch(tempPos, &block.boundaries[bound].vdb, domain);
				velN = getVelocityFromBoundaryVarying<scalar_t, DIMS>(tempPos, &block.boundaries[bound].vdb);
				break;
			}
			case BoundaryType::FIXED:
				// enforce flux
			{
				tempPos.a[dim] = 0;
				velN = getVelocityFromBoundaryFixed<scalar_t, DIMS>(tempPos, &block.boundaries[bound].fb);
				break;
			}
			case BoundaryType::GRADIENT:
				// compute flux only from center cell?
				// velN = velC - grad*distance*2 & flux = (velN+velC)*0.5 ->
				fluxes[bound] = 0; //velC + block.boundaries[bound].snb.boundaryGradient.a[dim] * 0.5; //* distance
				continue;
				break;
			case BoundaryType::CONNECTED_GRID:
			{
				// handle multi-block grids, load from correct cell of the connected grid
				const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
				I4 otherPos = computeConnectedPos(tempPos, dim, &block.boundaries[bound].cb, domain);
				//scalar_t velN = getContravariantComponentDimSwitch(otherPos, p_connectedBlock, domain, velocityGlobal);
				velN = getVelocityFromBlock<scalar_t, DIMS>(otherPos, p_connectedBlock, domain, velocityGlobal);
				break;
			}
			case BoundaryType::PERIODIC:
			{
				// compute flux to cell on other side
				// special case of connection to another block
				tempPos.a[dim] = isUpper ? 0 : block.size.a[dim] - 1;
				//const scalar_t velN = getContravariantComponentDimSwitch(tempPos, &block, domain, velocityGlobal);
				velN = getVelocityFromBlock<scalar_t, DIMS>(tempPos, &block, domain, velocityGlobal);
				break;
			}
			default:
				fluxes[bound] = 0;
				break;
			}
		}else{
			tempPos.a[dim] = pos.a[dim] + faceSignFromBound(bound);
			//const scalar_t velN = getContravariantComponentDimSwitch(tempPos, &block, domain, velocityGlobal);
			velN = getVelocityFromBlock<scalar_t, DIMS>(tempPos, &block, domain, velocityGlobal);
		}

		if(!isBoundVel){
			const Vector<scalar_t,DIMS> velC = getVelocityFromBlock<scalar_t, DIMS>(pos, &block, domain, velocityGlobal);
			velN = slerp2Halfpoint(velC, velN, static_cast<scalar_t>(1e-5));
		}

		const TransformGPU<scalar_t, DIMS> *p_T = getFaceTransformPtr<scalar_t, DIMS>(pos, bound, block);
		fluxes[bound] = p_T->det * dot(p_T->Minv.v[dim], velN);
	}
}


template <typename scalar_t>
__device__
scalar_t getViscosity(const DomainGPU<scalar_t> &domain, const bool forPassiveScalar, const index_t passiveScalarChannel){
	if(forPassiveScalar && (domain.scalarViscosity!=nullptr)){
		if(domain.scalarViscosityStatic){
			return domain.scalarViscosity[0];
		}else{
			return domain.scalarViscosity[passiveScalarChannel];
		}
	}else{
		return domain.viscosity;
	}
}
template <typename scalar_t>
__device__
scalar_t getViscosityBlock(const I4 &pos, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const bool forPassiveScalar, const index_t passiveScalarChannel){
	if(forPassiveScalar){
		// TODO: per-block passive scalar
		return getViscosity<scalar_t>(domain, true, passiveScalarChannel);
	}else {
		if(p_block->viscosity==nullptr){
			// Alternative: set p_block->viscosity to global domain viscosity on copy to gpu
			return getViscosity<scalar_t>(domain, false, 0);
		} else {
			if(p_block->isViscosityStatic){
				return p_block->viscosity[0];
			} else {
				I4 tempPos = pos;
				tempPos.w = 0; // velocity viscosity can't be per channel
				const index_t flatPos = flattenIndex(tempPos, p_block);
				return p_block->viscosity[flatPos];
			}
		}
	}
}
template <typename scalar_t>
__device__
scalar_t getViscosityFixedBoundary(const I4 &posBlock, const FixedBoundaryGPU<scalar_t> *p_fbound, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const bool forPassiveScalar, const index_t passiveScalarChannel){
	// TODO: per-boundary viscosity
	return getViscosityBlock<scalar_t>(posBlock, p_block, domain, forPassiveScalar, passiveScalarChannel);
}

#ifdef WITH_GRAD

template <typename scalar_t>
__device__
void scatterViscosity_GRAD(const scalar_t viscosity_grad, const DomainGPU<scalar_t> &domain, const bool forPassiveScalar, const index_t passiveScalarChannel){
	
	scalar_t *p_viscosity_grad = nullptr;
	if(forPassiveScalar && (domain.scalarViscosity_grad!=nullptr)){
		if(domain.scalarViscosityStatic){
			p_viscosity_grad = domain.scalarViscosity_grad;
		}else{
			p_viscosity_grad = domain.scalarViscosity_grad + passiveScalarChannel;
		}
	}else{
		p_viscosity_grad = domain.viscosity_grad;
	}
	
	// TODO: this is one address per channel, use better reduction scheme?
	if(p_viscosity_grad){ atomicAdd(p_viscosity_grad, viscosity_grad); }
}
template <typename scalar_t>
__device__
void scatterViscosityBlock_GRAD(const scalar_t viscosity_grad, const I4 &pos, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const bool forPassiveScalar, const index_t passiveScalarChannel){
	if(forPassiveScalar){
		// TODO: per-block passive scalar
		scatterViscosity_GRAD(viscosity_grad, domain, true, passiveScalarChannel);
	}else {
		if(p_block->viscosity==nullptr){ // forward value was taken from global, so we scatter to global grad
			// Alternative: set p_block->viscosity_grad to global domain viscosity on copy to gpu
			scatterViscosity_GRAD(viscosity_grad, domain, false, 0);
		} else {
			scalar_t *p_viscosity_grad = nullptr;
			if(p_block->isViscosityStatic){
				p_viscosity_grad = p_block->viscosity_grad;
			} else {
				I4 tempPos = pos;
				tempPos.w = 0;
				const index_t flatPos = flattenIndex(tempPos, p_block);
				p_viscosity_grad = p_block->viscosity_grad + flatPos;
			}
			if(p_viscosity_grad){ atomicAdd(p_viscosity_grad, viscosity_grad); }
		}
	}
}
template <typename scalar_t>
__device__
void scatterViscosityBoundary_GRAD(const scalar_t viscosity_grad, const I4 &posBlock, const FixedBoundaryGPU<scalar_t> *p_fbound, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const bool forPassiveScalar, const index_t passiveScalarChannel){
	// TODO: per-boundary viscosity
	scatterViscosityBlock_GRAD(viscosity_grad, posBlock, p_block, domain, forPassiveScalar, passiveScalarChannel);
}

#endif //WITH_GRAD
