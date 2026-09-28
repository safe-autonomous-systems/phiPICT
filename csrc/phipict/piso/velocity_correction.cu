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
// Velocity correction with the pressure gradient, and its gradient.

#include "piso/device/common.cuh"


/* --- Velocity correction --- */


// pressure gradient differenced over +-1
template <typename scalar_t>
__global__ void PISO_update_velocity(DomainGPU<scalar_t> *p_domain, const bool timeStepNorm,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){
	//vel.x = pressureRHS.x - inv(A)*gradX(pressure)
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		const scalar_t timeStep = s_domain.timeStep; // per batched environment
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		const index_t flatPosGlobal = s_block.globalOffset + flatPos;
		
		const scalar_t rDiag = 1/s_domain.Adiag[flatPosGlobal];
		
		scalar_t pressureGrad[3];
		tempGetPressureGradientDimSwitch<scalar_t>(s_block, pos, s_domain, pressureGrad);
		
		for(int dim=0;dim<s_domain.numDims;++dim){
			I4 tempPos = pos;
			tempPos.w = 0;
			
			scalar_t velUpdate = - rDiag * pressureGrad[dim]; // * timeStep;
			
			if(timeStepNorm){
				velUpdate *= timeStep;
			}
			
			
			tempPos.w = dim;
			tempPos.a[dim] = pos.a[dim];
			const int flatCompPosGlobal = flattenIndexGlobal(tempPos, s_block, s_domain);
			velUpdate += s_domain.pressureRHS[flatCompPosGlobal];
			s_domain.velocityResult[flatCompPosGlobal] = velUpdate;
		}
	)
}


// finite volume pressure gradient
template <typename scalar_t>
__global__ void PISO_update_velocity_PressureFVM(DomainGPU<scalar_t> *p_domain, const index_t gradientInterpolation, const bool timeStepNorm,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){
	//vel.x = pressureRHS.x - inv(A)*gradX(pressure)
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		const scalar_t timeStep = s_domain.timeStep; // per batched environment
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		const index_t flatPosGlobal = s_block.globalOffset + flatPos;
		
		//const scalar_t det = getDeterminantDimSwitch(s_block, pos, s_domain.numDims); // cell volume
		const scalar_t rDiag = 1/s_domain.Adiag[flatPosGlobal];
		
		scalar_t pressureGrad[3];
		tempGetPressureGradientFVMDimSwitch<scalar_t>(s_block, pos, s_domain, gradientInterpolation, pressureGrad);
		
		for(int dim=0;dim<s_domain.numDims;++dim){
			I4 tempPos = pos;
			tempPos.w = 0;
			scalar_t velUpdate = - rDiag * pressureGrad[dim]; // * timeStep;
			
			if(timeStepNorm){
				velUpdate *= timeStep;
			}
			
			tempPos.w = dim;
			tempPos.a[dim] = pos.a[dim];
			const int flatCompPosGlobal = flattenIndexGlobal(tempPos, s_block, s_domain);
			velUpdate += s_domain.pressureRHS[flatCompPosGlobal];
			s_domain.velocityResult[flatCompPosGlobal] = velUpdate;
		}
	)
}


template <typename scalar_t>
__global__ void PISO_update_velocity_v4_orthogonal(DomainGPU<scalar_t> *p_domain,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){
	//TODO: non-orthogonal transformations
	//vel.x = pressureRHS.x - inv(A)*gradX(pressure)
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		const scalar_t timeStep = s_domain.timeStep; // per batched environment
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		const index_t flatPosGlobal = s_block.globalOffset + flatPos;
		
		const scalar_t ra = 1/s_domain.Adiag[flatPosGlobal];
		
		//scalar_t pressureGrad[3];
		//tempGetPressureGradientDimSwitch<scalar_t>(s_block, pos, s_domain, pressureGrad);
		const scalar_t p = s_block.pressure[flattenIndex(pos, s_block)];
		const scalar_t det = getDeterminantDimSwitch(s_block, pos, s_domain.numDims);
		
		
		for(int dim=0;dim<s_domain.numDims;++dim){
			scalar_t velUpdate = 0;
			
			I4 tempPos = pos;
			tempPos.w = dim;
			const int flatCompPosGlobal = flattenIndexGlobal(tempPos, s_block, s_domain);
			const scalar_t hbyA = s_domain.pressureRHS[flatCompPosGlobal];
			const scalar_t t = getTransformMetricOrthogonalDimSwitch<scalar_t>(pos, &s_block, s_domain.numDims);
			const scalar_t cc = det*t; //covariant transform coefficient
			
			//lower boundary
			index_t bound = dim*2;
			if(pos.a[dim]!=0 || !isEmptyBound(bound, s_block.boundaries)){
				tempPos = pos;
				tempPos.w = dim;
				index_t tempFlatPosGlobal = 0;
				scalar_t pL = 0;
				scalar_t hbyAL = 0;
				scalar_t tL = 1; //transform metric
				scalar_t detL = 1;
				if(pos.a[dim]==0 && s_block.boundaries[bound].type==BoundaryType::CONNECTED_GRID){
					const BlockGPU<scalar_t> *p_connectedBlock = s_domain.blocks + s_block.boundaries[bound].cb.connectedGridIndex;
					tempPos = computeConnectedPosWithChannel(tempPos, dim, &s_block.boundaries[bound].cb, s_domain);
					hbyAL = s_domain.pressureRHS[flattenIndexGlobal(tempPos, p_connectedBlock, s_domain)];
					tL = getTransformMetricOrthogonalDimSwitch<scalar_t>(tempPos, p_connectedBlock, s_domain.numDims);
					tempPos.w = 0;
					detL = getDeterminantDimSwitch(p_connectedBlock, tempPos, s_domain.numDims);
					const index_t blockFlatIdx = flattenIndex(tempPos, p_connectedBlock);
					pL = p_connectedBlock->pressure[blockFlatIdx];
					tempFlatPosGlobal = blockFlatIdx + p_connectedBlock->globalOffset;
				}else {
					if(pos.a[dim]==0 && s_block.boundaries[bound].type==BoundaryType::PERIODIC){
						tempPos.a[dim] = s_block.size.a[dim]-1;
					}else{
						tempPos.a[dim] = pos.a[dim]-1;
					}
					hbyAL = s_domain.pressureRHS[flattenIndexGlobal(tempPos, s_block, s_domain)];
					tL = getTransformMetricOrthogonalDimSwitch<scalar_t>(tempPos, &s_block, s_domain.numDims);
					tempPos.w = 0;
					detL = getDeterminantDimSwitch(s_block, tempPos, s_domain.numDims);
					const index_t blockFlatIdx = flattenIndex(tempPos, s_block);
					pL = s_block.pressure[blockFlatIdx];
					tempFlatPosGlobal = blockFlatIdx + s_block.globalOffset;
				}
				const scalar_t raL = 1 /s_domain.Adiag[tempFlatPosGlobal];
				const scalar_t ccL = detL*tL;
				velUpdate += (cc*hbyA + ccL*hbyAL - (cc*t*ra + ccL*tL*raL)*(p - pL) * timeStep) * static_cast<scalar_t>(0.25);
			}
			//upper boundary
			++bound;
			if(pos.a[dim]!=s_block.size.a[dim]-1 || !isEmptyBound(bound, s_block.boundaries)){
				tempPos = pos;
				tempPos.w = dim;
				index_t tempFlatPosGlobal = 0;
				scalar_t pU = 0;
				scalar_t hbyAU = 0;
				scalar_t tU = 1; //transform metric
				scalar_t detU = 1;
				if(pos.a[dim]==s_block.size.a[dim]-1 && s_block.boundaries[bound].type==BoundaryType::CONNECTED_GRID){
					const BlockGPU<scalar_t> *p_connectedBlock = s_domain.blocks + s_block.boundaries[bound].cb.connectedGridIndex;
					tempPos = computeConnectedPosWithChannel(tempPos, dim, &s_block.boundaries[bound].cb, s_domain);
					hbyAU = s_domain.pressureRHS[flattenIndexGlobal(tempPos, p_connectedBlock, s_domain)];
					tU = getTransformMetricOrthogonalDimSwitch<scalar_t>(tempPos, p_connectedBlock, s_domain.numDims);
					tempPos.w = 0;
					detU = getDeterminantDimSwitch(p_connectedBlock, tempPos, s_domain.numDims);
					const index_t blockFlatIdx = flattenIndex(tempPos, p_connectedBlock);
					pU = p_connectedBlock->pressure[blockFlatIdx];
					tempFlatPosGlobal = blockFlatIdx + p_connectedBlock->globalOffset;
				}else {
					if(pos.a[dim]==s_block.size.a[dim]-1 && s_block.boundaries[bound].type==BoundaryType::PERIODIC){
						tempPos.a[dim] = 0;
					}else{
						tempPos.a[dim] = pos.a[dim]+1;
					}
					hbyAU = s_domain.pressureRHS[flattenIndexGlobal(tempPos, s_block, s_domain)];
					tU = getTransformMetricOrthogonalDimSwitch<scalar_t>(tempPos, &s_block, s_domain.numDims);
					tempPos.w = 0;
					detU = getDeterminantDimSwitch(s_block, tempPos, s_domain.numDims);
					const index_t blockFlatIdx = flattenIndex(tempPos, s_block);
					pU = s_block.pressure[blockFlatIdx];
					tempFlatPosGlobal = blockFlatIdx + s_block.globalOffset;
				}
				const scalar_t raU = 1 /s_domain.Adiag[tempFlatPosGlobal];
				const scalar_t ccU = detU*tU;
				velUpdate += (cc*hbyA + ccU*hbyAU - (cc*t*ra + ccU*tU*raU)*(pU - p) * timeStep) * static_cast<scalar_t>(0.25);
			}
			
			//velUpdate is a flux here, transform back with inverse transform
			//orthogonal, so can use inverse of t
			velUpdate /= det*t;
			s_domain.velocityResult[flatCompPosGlobal] = velUpdate;
		}
	)
}

template <typename scalar_t>
void _CorrectVelocity(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU, const index_t version, const bool timeStepNorm){ //, const torch::Tensor &timeStepCPU){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	BEGIN_SAMPLE;
	switch(version){
	case 1:
		PISO_update_velocity<scalar_t><<<blocks, threads>>>(
				reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepNorm,
				p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
			);
		break;
	// 2, 3: DO NOT USE, this is a smoothing kernel
	case 4:
		PISO_update_velocity_v4_orthogonal<scalar_t><<<blocks, threads>>>(
				reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
				p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
			);
		break;
	case 5:
		PISO_update_velocity_PressureFVM<scalar_t><<<blocks, threads>>>(
				reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), 0, timeStepNorm, // original linear interpolation of metrics/fluxes
				p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
			);
		break;
	case 6:
		PISO_update_velocity_PressureFVM<scalar_t><<<blocks, threads>>>(
				reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), 3, timeStepNorm, // use face metrics instead of interpolation
				p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
			);
		break;
	default:
		PISO_update_velocity<scalar_t><<<blocks, threads>>>(
				reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepNorm,
				p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
			);
		break;
	}
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("correct u");
	
}
void CorrectVelocity(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const index_t version, const bool timeStepNorm){ //, const torch::Tensor &timeStep){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	domain->setTimeStep(timeStep); // [1] or one per batched environment
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CorrectVelocity", ([&] {
		_CorrectVelocity<scalar_t>(
			domain, timeStep, version, timeStepNorm
		);
	}));
}

#ifdef WITH_GRAD
template<typename scalar_t>
__device__ void tempScatterPressureGradientGradDimSwitch(const scalar_t *pressureGradGradIn, const BlockGPU<scalar_t> &block, const I4 &pos, const DomainGPU<scalar_t> &domain){
	switch(domain.numDims){
	case 1:
	{
		const Vector<scalar_t, 1> pressureGradGrad = {.a={pressureGradGradIn[0]}};
		scatterPressureGradientGrad<scalar_t, 1>(pressureGradGrad, block, pos, domain);
		return;
	}
	case 2:
	{
		const Vector<scalar_t, 2> pressureGradGrad = {.a={pressureGradGradIn[0], pressureGradGradIn[1]}};
		scatterPressureGradientGrad<scalar_t, 2>(pressureGradGrad, block, pos, domain);
		return;
	}
	case 3:
	{
		const Vector<scalar_t, 3> pressureGradGrad = {.a={pressureGradGradIn[0], pressureGradGradIn[1], pressureGradGradIn[2]}};
		scatterPressureGradientGrad<scalar_t, 3>(pressureGradGrad, block, pos, domain);
		return;
	}
	default:
		return;
	}
}

template <typename scalar_t>
__global__ void PISO_update_velocity_GRAD(DomainGPU<scalar_t> *p_domain, const bool timeStepNorm,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){
	//vel.x = pressureRHS.x - inv(A)*gradX(pressure)
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		const scalar_t timeStep = s_domain.timeStep; // per batched environment
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		const index_t flatPosGlobal = s_block.globalOffset + flatPos;
		
		const scalar_t rDiag = 1/s_domain.Adiag[flatPosGlobal];
		scalar_t rDiag_grad = 0;
		
		scalar_t pressureGrad[3];
		tempGetPressureGradientDimSwitch<scalar_t>(s_block, pos, s_domain, pressureGrad);
		scalar_t pressureGradGrad[3] = {0};
		
		for(int dim=0;dim<s_domain.numDims;++dim){
			
			I4 tempPos = pos;
			tempPos.w = dim;
			int flatCompPosGlobal = flattenIndexGlobal(tempPos, s_block, s_domain);
			
			scalar_t velUpdateGrad = s_domain.velocityResult_grad[flatCompPosGlobal];
			
			// w.r.t. pressure rhs
			s_domain.pressureRHS_grad[flatCompPosGlobal] = velUpdateGrad;
			
			if(timeStepNorm){
				velUpdateGrad *= timeStep;
			}
			
			//FWD: scalar_t velUpdate = - rDiag * pressureGrad[dim];
			
			pressureGradGrad[dim] = - rDiag * velUpdateGrad;
			rDiag_grad -= pressureGrad[dim] * velUpdateGrad;
			
		}
		
		// scatter pressure grad grad:
		tempScatterPressureGradientGradDimSwitch(pressureGradGrad, s_block, pos, s_domain);
		s_domain.Adiag_grad[flatPosGlobal] = - rDiag_grad * rDiag * rDiag;
	)
}

template <typename scalar_t>
void _CorrectVelocity_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU, const bool timeStepNorm){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	BEGIN_SAMPLE;
	PISO_update_velocity_GRAD<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepNorm,
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("correct u grad");
	
}
void CorrectVelocity_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const bool timeStepNorm){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Gradient for Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	domain->setTimeStep(timeStep); // [1] or one per batched environment
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CorrectVelocity_GRAD", ([&] {
		_CorrectVelocity_GRAD<scalar_t>(
			domain, timeStep, timeStepNorm
		);
	}));
}

#endif //WITH_GRAD
