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
// Velocity access, contravariant components and pressure gradients.

#pragma once

#include "piso/device/topology.cuh"



template <typename scalar_t, int DIMS>
__device__ scalar_t VelocityToContravariantComponent(const Vector<scalar_t,DIMS> &vel, const BlockGPU<scalar_t> *p_block, I4 pos){
	if(p_block->hasTransform){
		const index_t component = pos.w;
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_block->transform) + flattenIndex(pos, p_block);
		const scalar_t det = T->det;
		//load more globally as it is needed for all sides
		//const MatrixSquare<scalar_t, 3> mInv = T->Minv;
		const Vector<scalar_t, DIMS> mInvRow = T->Minv.v[component];
		//Minv[pos.w] is needed for 2 faces
		return det*dot(mInvRow, vel);
	} else {
		return vel.a[pos.w];
	}
}

template <typename scalar_t, int DIMS>
__device__ scalar_t VelocityToContravariantComponentBoundaryVarying(const Vector<scalar_t,DIMS> &vel, const VaryingDirichletBoundaryGPU<scalar_t> *p_bound, I4 pos){
	if(p_bound->hasTransform){
		const index_t component = pos.w;
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_bound->transform) + flattenIndex(pos, p_bound->stride);
		const scalar_t det = T->det;
		const Vector<scalar_t, DIMS> mInvRow = T->Minv.v[component];
		return det*dot(mInvRow, vel);
	} else {
		return vel.a[pos.w];
	}
}
template <typename scalar_t, int DIMS>
__device__ scalar_t VelocityToContravariantComponentBoundaryFixed(const Vector<scalar_t,DIMS> &vel, const FixedBoundaryGPU<scalar_t> *p_bound, I4 pos){
	if(p_bound->hasTransform){
		const index_t component = pos.w;
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_bound->transform) + flattenIndex(pos, p_bound->stride);
		const scalar_t det = T->det;
		const Vector<scalar_t, DIMS> mInvRow = T->Minv.v[component];
		return det*dot(mInvRow, vel);
	} else {
		return vel.a[pos.w];
	}
}


#ifdef WITH_GRAD
template <typename scalar_t, int DIMS>
__device__ Vector<scalar_t,DIMS> VelocityGradFromContravariantComponentGrad(const scalar_t &velGrad, const BlockGPU<scalar_t> *p_block, I4 pos){
	if(p_block->hasTransform){
		const index_t component = pos.w;
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_block->transform) + flattenIndex(pos, p_block);
		const scalar_t det = T->det;
		
		const Vector<scalar_t, DIMS> mInvRow = T->Minv.v[component];
		
		return mInvRow*(det*velGrad);
	} else {
		Vector<scalar_t,DIMS> grad = {.a={0}};
		grad.a[pos.w] = velGrad;
		return grad;
	}
}

template <typename scalar_t, int DIMS>
__device__ Vector<scalar_t,DIMS> VelocityGradFromContravariantComponentGradBoundaryFixed(const scalar_t &velGrad, const FixedBoundaryGPU<scalar_t> *p_bound, I4 pos){
	if(p_bound->hasTransform){
		const index_t component = pos.w;
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_bound->transform) + flattenIndex(pos, p_bound->stride);
		const scalar_t det = T->det;
		
		const Vector<scalar_t, DIMS> mInvRow = T->Minv.v[component];
		
		return mInvRow*(det*velGrad);
	} else {
		Vector<scalar_t,DIMS> grad = {.a={0}};
		grad.a[pos.w] = velGrad;
		return grad;
	}
}
#endif //WITH_GRAD


template <typename scalar_t, int DIMS>
__device__ Vector<scalar_t,DIMS> getVelocityFromBlock(I4 pos, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const scalar_t *velocityGlobal){
	Vector<scalar_t,DIMS> vel = {.a={0}};
	//pos.w = 0;
	//const index_t blockStride = p_block->stride.w;
	//const index_t blockOffset = p_block->globalOffset;
	//index_t flatPos = flattenIndex(pos, p_block->stride);
	for(index_t dim=0; dim<DIMS; ++dim){
		pos.w = dim;
		if(velocityGlobal==nullptr){
			vel.a[dim] = p_block->velocity[flattenIndex(pos, p_block)];
		}else{
			vel.a[dim] = velocityGlobal[flattenIndexGlobal(pos, p_block, domain)];
		}
		//flatPos += blockStride;
	}
	return vel;
}

template <typename scalar_t, int DIMS>
__device__ Vector<scalar_t,DIMS> getVelocityFromBoundaryVarying(I4 pos, const VaryingDirichletBoundaryGPU<scalar_t> *p_bound){
	Vector<scalar_t,DIMS> vel = {.a={0}};
	for(index_t dim=0; dim<DIMS; ++dim){
		pos.w = dim;
		vel.a[dim] = p_bound->velocity[flattenIndex(pos, p_bound->stride)];
	}
	return vel;
}
template <typename scalar_t, int DIMS>
__device__ Vector<scalar_t,DIMS> getVelocityFromBoundaryFixed(I4 pos, const FixedBoundaryGPU<scalar_t> *p_bound){
	const FixedBoundaryDataGPU<scalar_t> *p_data = &(p_bound->velocity);
	Vector<scalar_t,DIMS> vel = {.a={0}};
	if(p_data->boundaryType==BoundaryConditionType::DIRICHLET){
		if(p_data->isStatic){
			for(index_t dim=0; dim<DIMS; ++dim){
				pos.w = dim;
				vel.a[dim] = p_data->data[pos.w];
			}
		} else {
			for(index_t dim=0; dim<DIMS; ++dim){
				pos.w = dim;
				vel.a[dim] = p_data->data[flattenIndex(pos, p_bound->stride)];
			}
		}
	}
	// else TODO
	return vel;
}


#ifdef WITH_GRAD
template <typename scalar_t, int DIMS>
__device__ void scatterVelocityGradToGrid(const Vector<scalar_t,DIMS> &velGrad, I4 pos, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, scalar_t *velocityGlobalGrad){
	for(index_t dim=0; dim<DIMS; ++dim){
		pos.w = dim;
		if(velocityGlobalGrad==nullptr){
			atomicAdd(p_block->velocity_grad + flattenIndex(pos, p_block), velGrad.a[dim]);
		}else{
			atomicAdd(velocityGlobalGrad + flattenIndexGlobal(pos, p_block, domain), velGrad.a[dim]);
		}
	}
}

template <typename scalar_t, int DIMS>
__device__ void scatterVelocityGradToGridBoundaryFixed(const Vector<scalar_t,DIMS> &velGrad, I4 pos, const FixedBoundaryGPU<scalar_t> *p_bound){
	for(index_t dim=0; dim<DIMS; ++dim){
		const FixedBoundaryDataGPU<scalar_t> *p_data = &(p_bound->velocity);
		if(p_data->isStatic){
			atomicAdd(p_data->grad + dim, velGrad.a[dim]);
		} else {
			pos.w = dim;
			atomicAdd(p_data->grad + flattenIndex(pos, p_bound->stride), velGrad.a[dim]);
		}
	}
}
#endif //WITH_GRAD


template <typename scalar_t, int DIMS>
__device__ inline scalar_t getContravariantComponent(const I4 pos, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const scalar_t *velocityGlobal){
	if(p_block->hasTransform){
		Vector<scalar_t,DIMS> vel = getVelocityFromBlock<scalar_t, DIMS>(pos, p_block, domain, velocityGlobal);
		return VelocityToContravariantComponent<scalar_t, DIMS>(vel, p_block, pos);
	} else {
		if(velocityGlobal==nullptr){
			return p_block->velocity[flattenIndex(pos, p_block)];
		}else{
			return velocityGlobal[flattenIndexGlobal(pos, p_block, domain)];
		}
	}
}

template <typename scalar_t, int DIMS>
__device__ inline scalar_t getContravariantComponentBoundaryVarying(const I4 pos, const VaryingDirichletBoundaryGPU<scalar_t> *p_bound){
	if(p_bound->hasTransform){
		Vector<scalar_t,DIMS> vel = getVelocityFromBoundaryVarying<scalar_t, DIMS>(pos, p_bound);
		return VelocityToContravariantComponentBoundaryVarying<scalar_t, DIMS>(vel, p_bound, pos);
	} else {
		return p_bound->velocity[flattenIndex(pos, p_bound->stride)];
	}
}

template <typename scalar_t, int DIMS>
__device__ inline scalar_t getContravariantComponentBoundaryFixed(const I4 pos, const FixedBoundaryGPU<scalar_t> *p_bound){
	if(p_bound->hasTransform){
		Vector<scalar_t,DIMS> vel = getVelocityFromBoundaryFixed<scalar_t, DIMS>(pos, p_bound);
		return VelocityToContravariantComponentBoundaryFixed<scalar_t, DIMS>(vel, p_bound, pos);
	} else {
		//return getFixedBoundaryData();//p_bound->velocity[flattenIndex(pos, p_bound->stride)];
		const FixedBoundaryDataGPU<scalar_t> *p_data = &(p_bound->velocity);
		if(p_data->isStatic){
			return p_data->data[pos.w];
		} else {
			//const index_t dim = axisFromBound(bound);
			//I4 boundPos = pos;
			//if(isScalarDataType(type)) { boundPos.w = 0; }
			//boundPos.a[dim] = 0;
			const index_t flatBoundPos = flattenIndex(pos, p_bound->stride);
			return p_data->data[flatBoundPos];
		}
	}
}

#ifdef WITH_GRAD
template <typename scalar_t, int DIMS>
__device__ void scatterContravariantComponentGrad(const scalar_t &velGrad, const I4 pos, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, scalar_t *velocityGlobalGrad){
	if(p_block->hasTransform){
		Vector<scalar_t,DIMS> grad = VelocityGradFromContravariantComponentGrad<scalar_t, DIMS>(velGrad, p_block, pos);
		scatterVelocityGradToGrid<scalar_t, DIMS>(grad, pos, p_block, domain, velocityGlobalGrad);
	} else {
		if(velocityGlobalGrad==nullptr){
			atomicAdd(p_block->velocity_grad + flattenIndex(pos, p_block), velGrad);
		}else{
			atomicAdd(velocityGlobalGrad + flattenIndexGlobal(pos, p_block, domain), velGrad);
		}
	}
}

template <typename scalar_t, int DIMS>
__device__ void scatterContravariantComponentGradBoundaryFixed(const scalar_t &velGrad, const I4 pos, const FixedBoundaryGPU<scalar_t> *p_bound){
	if(p_bound->hasTransform){
		Vector<scalar_t,DIMS> grad = VelocityGradFromContravariantComponentGradBoundaryFixed<scalar_t, DIMS>(velGrad, p_bound, pos);
		scatterVelocityGradToGridBoundaryFixed<scalar_t, DIMS>(grad, pos, p_bound);
	} else {
		const FixedBoundaryDataGPU<scalar_t> *p_data = &(p_bound->velocity);
		if(p_data->isStatic){
			atomicAdd(p_data->grad + pos.w, velGrad);
		}else{
			const index_t flatBoundPos = flattenIndex(pos, p_bound->stride);
			atomicAdd(p_data->grad + flatBoundPos, velGrad);
		}
	}
}
#endif //WITH_GRAD


template <typename scalar_t>
__device__ inline scalar_t getContravariantComponentDimSwitch(const I4 pos, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const scalar_t *velocityGlobal){
	switch(domain.numDims){
	case 1:
		return getContravariantComponent<scalar_t, 1>(pos, p_block, domain, velocityGlobal);
	case 2:
		return getContravariantComponent<scalar_t, 2>(pos, p_block, domain, velocityGlobal);
	case 3:
		return getContravariantComponent<scalar_t, 3>(pos, p_block, domain, velocityGlobal);
	default:
		return 0;
	}
}

template <typename scalar_t>
__device__ inline scalar_t getContravariantComponentBoundaryVaryingDimSwitch(const I4 pos, const VaryingDirichletBoundaryGPU<scalar_t> *p_bound, const DomainGPU<scalar_t> &domain){
	switch(domain.numDims){
	case 1:
		return getContravariantComponentBoundaryVarying<scalar_t, 1>(pos, p_bound);
	case 2:
		return getContravariantComponentBoundaryVarying<scalar_t, 2>(pos, p_bound);
	case 3:
		return getContravariantComponentBoundaryVarying<scalar_t, 3>(pos, p_bound);
	default:
		return 0;
	}
}

template <typename scalar_t>
__device__ inline scalar_t getContravariantComponentBoundaryFixedDimSwitch(const I4 pos, const FixedBoundaryGPU<scalar_t> *p_bound, const DomainGPU<scalar_t> &domain){
	switch(domain.numDims){
	case 1:
		return getContravariantComponentBoundaryFixed<scalar_t, 1>(pos, p_bound);
	case 2:
		return getContravariantComponentBoundaryFixed<scalar_t, 2>(pos, p_bound);
	case 3:
		return getContravariantComponentBoundaryFixed<scalar_t, 3>(pos, p_bound);
	default:
		return 0;
	}
}

#ifdef WITH_GRAD
template <typename scalar_t>
__device__ void scatterContravariantComponentDimSwitch(const scalar_t &velGrad, const I4 pos, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, scalar_t *velocityGlobalGrad){
	switch(domain.numDims){
	case 1:
		scatterContravariantComponentGrad<scalar_t, 1>(velGrad, pos, p_block, domain, velocityGlobalGrad);
		break;
	case 2:
		scatterContravariantComponentGrad<scalar_t, 2>(velGrad, pos, p_block, domain, velocityGlobalGrad);
		break;
	case 3:
		scatterContravariantComponentGrad<scalar_t, 3>(velGrad, pos, p_block, domain, velocityGlobalGrad);
		break;
	default:
		break;
	}
}

template <typename scalar_t>
__device__ void scatterContravariantComponentBoundaryFixedDimSwitch(const scalar_t &velGrad, const I4 pos, const FixedBoundaryGPU<scalar_t> *p_bound, const DomainGPU<scalar_t> &domain){
	switch(domain.numDims){
	case 1:
		scatterContravariantComponentGradBoundaryFixed<scalar_t, 1>(velGrad, pos, p_bound);
		break;
	case 2:
		scatterContravariantComponentGradBoundaryFixed<scalar_t, 2>(velGrad, pos, p_bound);
		break;
	case 3:
		scatterContravariantComponentGradBoundaryFixed<scalar_t, 3>(velGrad, pos, p_bound);
		break;
	default:
		break;
	}
}
#endif //WITH_GRAD




template <typename scalar_t, int DIMS>
__device__ inline Vector<scalar_t,DIMS> getPressureGradient(const BlockGPU<scalar_t> &block, const I4 pos, const DomainGPU<scalar_t> &domain){
	Vector<scalar_t,DIMS> pressureGrad;
	for(index_t dim=0; dim<DIMS; ++dim){
		I4 tempPos = pos;
		tempPos.w = 0;
		//boundary handling, correct extrapolation distance norm
		scalar_t fac = 0.5;
		if(!(pos.a[dim]==0 && isEmptyBound(dim*2,block.boundaries))){
			tempPos.a[dim] = pos.a[dim]-1;
		}else{
			fac = 1;
		}
		const scalar_t valN = getPressureAtWithBounds(tempPos, block, domain, false);
		if(!(pos.a[dim]==block.size.a[dim]-1 && isEmptyBound(dim*2+1,block.boundaries))){
			tempPos.a[dim] = pos.a[dim]+1;
		}else{
			tempPos.a[dim] = pos.a[dim];
			fac = 1;
		}
		const scalar_t valP = getPressureAtWithBounds(tempPos, block, domain, false);
		pressureGrad.a[dim] = (valP - valN)*fac;
	}
	
	if(block.hasTransform){
		I4 tempPos = pos;
		tempPos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(block.transform) + flattenIndex(tempPos, block);
		//pressureGrad = matmul(T->Minv, pressureGrad);
		pressureGrad = matmul(pressureGrad, T->Minv); // = matmul(transpose(T->Minv), pressureGrad)
	}
	
	return pressureGrad;
}


#ifdef WITH_GRAD
template <typename scalar_t, int DIMS>
__device__ inline void scatterPressureGradientGrad(Vector<scalar_t,DIMS> pressureGradGrad, const BlockGPU<scalar_t> &block, const I4 pos, const DomainGPU<scalar_t> &domain){
	
	if(block.hasTransform){
		I4 tempPos = pos;
		tempPos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(block.transform) + flattenIndex(tempPos, block);
		pressureGradGrad = matmul(T->Minv, pressureGradGrad);
	}
	
	for(index_t dim=0; dim<DIMS; ++dim){
		I4 posP = pos;
		posP.w = 0;
		I4 posN = pos;
		posN.w = 0;
		//boundary handling, correct extrapolation distance norm
		scalar_t fac = 0.5;
		if(!(pos.a[dim]==0 && isEmptyBound(dim*2,block.boundaries))){
			posN.a[dim] = pos.a[dim]-1;
		}else{
			fac = 1;
		}
		if(!(pos.a[dim]==block.size.a[dim]-1 && isEmptyBound(dim*2+1,block.boundaries))){
			posP.a[dim] = pos.a[dim]+1;
		}else{
			fac = 1;
		}
		const scalar_t gradP = pressureGradGrad.a[dim] * fac;
		const scalar_t gradN = pressureGradGrad.a[dim] * -fac;
		scatterPressureGradToWithBounds(gradN, posN, block, domain);
		scatterPressureGradToWithBounds(gradP, posP, block, domain);
	}
}

#endif //WITH_GRAD
