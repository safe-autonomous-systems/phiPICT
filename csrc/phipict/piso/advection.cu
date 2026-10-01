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
// Advection-diffusion (prediction step): matrix and RHS, and their gradients.

#include "piso/device/common.cuh"



/* --- Advection/Diffusion --- */


template <typename scalar_t>
__global__ void PISO_build_matrix(DomainGPU<scalar_t> *p_domain,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks,
		const int8_t nonOrthoFlags, const bool forPassiveScalar, const index_t passiveScalarChannel){
	
	__shared__ DomainGPU<scalar_t> s_domain;
	__shared__ BlockGPU<scalar_t> s_block; //info about current main block

	if(threadIdx.x==0){
		//TODO improve loading?
		s_domain = p_domain[blockIdx.y]; // batched environments: one domain per grid row
	}
	__syncthreads();
	const scalar_t timeStep = s_domain.timeStep; // per batched environment

	//first naive implementation, 1 thread per cell, no sharing
	// each thread/cell builds its row in the matrix
	index_t repetitions = numThreadBlocks/gridDim.x; // ceilDiv(numThreadBlocks, gridDim.x);
	if(blockIdx.x<(numThreadBlocks - gridDim.x*repetitions)){
		++repetitions;
	}
	index_t loadedBlockIdx = -1;
	
	
	for(index_t r=0;r<repetitions;++r){
		const index_t currentThreadBlockIdx = gridDim.x*r + blockIdx.x;
		/*if(currentThreadBlockIdx>=numThreadBlocks){
			return;
		}*/

		const index_t targetBlockIdx = p_blockIdxByThreadBlock[currentThreadBlockIdx];
		const index_t threadBlockOffsetInBlock = p_threadBlockOffsetInBlock[currentThreadBlockIdx];
		if(threadIdx.x==0){
			if(targetBlockIdx!=loadedBlockIdx){
				s_block = s_domain.blocks[targetBlockIdx];
				loadedBlockIdx = targetBlockIdx;
			}
		}
		__syncthreads();

		const index_t flatPos = threadBlockOffsetInBlock*blockDim.x + threadIdx.x;
		if(flatPos<s_block.stride.w){
			const I4 pos = unflattenIndex(flatPos, s_block);
			
			//get fluxes
			scalar_t fluxes[6];
			computeFluxesNDLoop<scalar_t>(pos, fluxes, s_block, s_domain, nullptr);
			
			//compute matrix
			
			//if(flatPos==0){
			//	csrMatrixRow[0] = 0;
			//}
			
			const RowMeta row = getCSRMatrixRowEndOffsetFromBlockBoundaries3D(flatPos, s_block, s_domain);
			int rowStartOffset = row.endOffset - row.size + s_block.csrOffset;
			
			s_domain.C.row[s_block.globalOffset + flatPos+1] = row.endOffset + s_block.csrOffset;
			//csrMatrixIndex[flatPos+1] = row.size;
			// csrMatrixIndex[flatPos] = pos.x;
			// csrMatrixIndex[flatPos + c_domain.stride.w + 1] = pos.y;
			// csrMatrixIndex[flatPos + c_domain.stride.w*2 + 2] = pos.z;
			
			
			// row entries have to be in ascending column order (for cublas)
			// for a default inner cell: -z,-y,-x,diag,+x,+y,+z
			// for boundary cell
			// - if the boundary is open or closed (no connection), the entry is simply removed
			// - if the boundary is periodic, the entry moves to the other end
			//   - grid size 1: the cell connects to itself
			//   - grid size 2: connects to same cell in both directions
			//   - cell at lower z border, domain is z-periodic: -y,-x,diag,+x,+y,+z,-z
			
			// alternative: compute flat indices, sort
			int indices[7]; // diag,-x,+x,-y,+y,-z,+z
			scalar_t rowValues[7] = {0};
			
			const scalar_t det = s_block.hasTransform ? getDeterminantDimSwitch(s_block, pos, s_domain.numDims) : 1;
			scalar_t diag = det/timeStep; // + static_cast<scalar_t>(s_domain.numDims*2)*s_domain.viscosity; //1/dt;
			indices[0] = flatPos + s_block.globalOffset;
			
			// viscosity
			//const scalar_t viscosity = getViscosity(s_domain, forPassiveScalar, passiveScalarChannel);
			const scalar_t viscosity_p = getViscosityBlock(pos, &s_block, s_domain, forPassiveScalar, passiveScalarChannel);
			//const scalar_t viscosityHalf = static_cast<scalar_t>(0.5)*viscosity; // *0.5 comes from interpolating alpha later
			scalar_t alpha_p[3] = {1,1,1};
			for(index_t dim=0;dim<s_domain.numDims;++dim){
				I4 tempPos = pos;
				tempPos.w = dim;
				alpha_p[dim] = getLaplaceCoefficientOrthogonalDimSwitch(tempPos, &s_block, s_domain.numDims);
			}
			
			// for every face: add advective fluxes, compute and add viscosity/diffusion.
			for(index_t bound=0; bound<(s_domain.numDims*2); ++bound){
				const index_t dim = axisFromBound(bound);
				const index_t isUpper = boundIsUpper(bound);
				const index_t faceSign = faceSignFromBound(bound);
				
				const bool atBound = isAtBound(pos, bound, s_block);
				if(!atBound || !isEmptyBound(bound, s_block.boundaries)){
					
					{ // advective fluxes
						const scalar_t faceFluxOut = faceSign * fluxes[bound];
						const bool deferred = s_domain.advectionScheme != AdvectionScheme::CENTRAL
							&& hasFarUpwindCell(pos, bound, faceFluxOut >= 0, s_block);
						if(deferred){
							// First-order upwind implicitly; this keeps the matrix an
							// M-matrix. The high-order part of the face value is added
							// as an explicit correction in kPISO_build_advection_RHS.
							if(faceFluxOut >= 0){
								diag += faceFluxOut;
							}else{
								rowValues[bound+1] += faceFluxOut;
							}
						}else{
							const scalar_t faceFlux = static_cast<scalar_t>(0.5) * faceFluxOut;
							diag += faceFlux;
							rowValues[bound+1] += faceFlux;
						}
					}
					
					{ // orthogonal viscosity
						//calculate index of neighbour
						I4 tempPos = pos;
						tempPos.w = dim;
						scalar_t alpha = 1;
						const BlockGPU<scalar_t> *p_block = &s_block;
						// resolve neighbor cell
						if(atBound && s_block.boundaries[bound].type==BoundaryType::CONNECTED_GRID){
							p_block = s_domain.blocks + s_block.boundaries[bound].cb.connectedGridIndex;
							tempPos = computeConnectedPosWithChannel(tempPos, dim, &s_block.boundaries[bound].cb, s_domain);
						}else{
							if(atBound && s_block.boundaries[bound].type==BoundaryType::PERIODIC){
								tempPos.a[dim] = isUpper ? 0 : s_block.size.a[dim]-1;
							}else{
								tempPos.a[dim] = pos.a[dim] + faceSign;
							}
						}
						// need correct component/channel for laplace coefficient
						alpha = getLaplaceCoefficientOrthogonalDimSwitch(tempPos, p_block, s_domain.numDims);
						tempPos.w = forPassiveScalar ?  passiveScalarChannel : 0;
						const scalar_t viscosity_n = getViscosityBlock(tempPos, p_block, s_domain, forPassiveScalar, passiveScalarChannel);
						tempPos.w = 0; // indices are "scalar"
						indices[bound+1] = flattenIndexGlobal(tempPos, p_block, s_domain); //flattenIndex(tempPos, p_block) + p_block->globalOffset;
						
						//const scalar_t viscosityCoeff = viscosityHalf*(alpha_p[dim] + alpha);
						const scalar_t viscosityCoeff = (alpha_p[dim]*viscosity_p + alpha*viscosity_n) * static_cast<scalar_t>(0.5);
						diag += viscosityCoeff;
						rowValues[bound+1] -= viscosityCoeff;
					}
					
					// laplace non-orthogonal coefficients from face tangential directions
					const bool includeNonOrthoNeighbors = nonOrthoFlags & NON_ORTHO_DIRECT_MATRIX;
					const bool includeNonOrthoDiag = nonOrthoFlags & NON_ORTHO_CENTER_MATRIX;
					if(s_domain.numDims>1 && (includeNonOrthoDiag || includeNonOrthoNeighbors)){
						scalar_t alphaInterp[12]; // size for 3D. contains viscosity
						if(s_domain.numDims==2) interpolateNonOrthoLaplaceComponents<scalar_t, 2>(pos, s_block, s_domain, alphaInterp, !forPassiveScalar, false, false); // currently always uses velocity viscosity
						else if(s_domain.numDims==3) interpolateNonOrthoLaplaceComponents<scalar_t, 3>(pos, s_block, s_domain, alphaInterp, !forPassiveScalar, false, false); // TODO: extend to use correct scalar viscosity
						
						I4 channelPos = pos;
						GridDataType dataType = GridDataType::VELOCITY;
						if(forPassiveScalar){
							channelPos.w = passiveScalarChannel;
							dataType = GridDataType::PASSIVE_SCALAR;
						}
					
						for(index_t i=1; i<s_domain.numDims; ++i){ // loop other axes
							const index_t tAxis = (dim + i)%s_domain.numDims;
							// *viscosity_p is a workaround for global scalar viscosity
							const scalar_t alpha = getInterpolatedNonOrthoLaplaceComponent(alphaInterp, bound, tAxis, s_domain.numDims) * (forPassiveScalar ? viscosity_p : static_cast<scalar_t>(1.0));
							
							if(alpha!=0){ // grid is non-orthogonal here
								for(index_t tIsUpper=0; tIsUpper<2; ++tIsUpper){ // loop corners
									const index_t tFace = axisToBound(tAxis, tIsUpper); //(tAxis<<1) + tIsUpper;
									const index_t tFaceSign = faceSignFromBound(tFace);
									const CornerValue<scalar_t> cVal = getCornerValue<scalar_t>(channelPos, bound, tFace,
										false, false, 2, s_block, s_domain, dataType); // computes interpolation divisor and checks for boundaries.
									const bool cornerAtBound = cVal.numCells<1;
									const bool boundIsGradient = cVal.boundType==BoundaryConditionType::NEUMANN;
									
									if(cornerAtBound){
										if(boundIsGradient){
											// simplified treatment: ignore boundary gradient and use one-sided difference from other side
											const scalar_t interpolationNorm = 0.25; // from other side, can't be anything else but 1/4
											const index_t tFaceOther = invertBound(tFace);
											const scalar_t viscosityCoeff = faceSign * tFaceSign * alpha * interpolationNorm;
											if(includeNonOrthoDiag){
												diag -= 3 * viscosityCoeff;
											}
											if(includeNonOrthoNeighbors){
												rowValues[bound+1] -= 3 * viscosityCoeff;
												rowValues[tFaceOther+1] += viscosityCoeff;
											}
											// if diagonals where to be included in the matrix: 
											// rowValues[diagonalOther] += viscosityCoeff;
										}
										// else: prescribed value, added on RHS, nothing to do here.
									} else {
										// normal corner with interpolated value
										const scalar_t interpolationNorm = 1.0 / static_cast<scalar_t>(cVal.numCells);
										const scalar_t viscosityCoeff = faceSign * tFaceSign * alpha * interpolationNorm;
										if(includeNonOrthoDiag){
											diag -= viscosityCoeff; // TODO: is this correct?
										}
										if(includeNonOrthoNeighbors){
											rowValues[bound+1] -= viscosityCoeff; // TODO: is this correct?
											rowValues[tFace+1] -= viscosityCoeff;
										}
										// if diagonals were to be included in the matrix: 
										// rowValues[diagonal] -= viscosityCoeff;
									}
								}
							}
						}
					}
				}else{ // face is a prescribed boundary
					//  TODO: missing non-orthogonal coefficients?
					index_t slip = 0;
					switch(s_block.boundaries[bound].type){
						case BoundaryType::DIRICHLET:
						{
							slip = s_block.boundaries[bound].sdb.slip;
							break;
						}
						case BoundaryType::DIRICHLET_VARYING:
						{
							slip = s_block.boundaries[bound].vdb.slip;
							break;
						}
						case BoundaryType::FIXED:
						{
							// TODO: type based on what the matrix is for.
							BoundaryConditionType boundaryType; // = s_block.boundaries[bound].fb.velocity.boundaryType;
							if(forPassiveScalar){
								I4 tempPos = pos;
								tempPos.w = passiveScalarChannel;
								boundaryType = getFixedBoundaryType(tempPos, bound, &s_block, GridDataType::PASSIVE_SCALAR);
							}else{
								boundaryType = s_block.boundaries[bound].fb.velocity.boundaryType;
							}
							slip = boundaryType==BoundaryConditionType::DIRICHLET ? 0 : 1;
							break;
						}
					}
					
					diag += (1 - slip) * 2 * viscosity_p * alpha_p[dim];
					indices[bound+1] = -1; //invalid/unused
				}
			}
			
			
			for(int dim=s_domain.numDims;dim<3;++dim){
				indices[dim*2+1] = -1; //invalid/unused
				indices[dim*2+2] = -1; //invalid/unused
			}

			
			//TODO
			rowValues[0] = diag;
			
			// sort, naive for now
			for(int i=0;i<row.size;++i){
				int colIndex = findLowestColumnIndex(indices, 7);
				rowValues[colIndex] /= det;
				
				if(colIndex<0){
					s_domain.C.index[rowStartOffset + i] = -1;
					s_domain.C.value[rowStartOffset + i] = -1.0f;
				}else{
					s_domain.C.index[rowStartOffset + i] = indices[colIndex];
					s_domain.C.value[rowStartOffset + i] = rowValues[colIndex];
				}
				
				indices[colIndex] = -1;
			}
			
			s_domain.Adiag[flatPos + s_block.globalOffset] = rowValues[0];
		}
	}
}

#ifdef WITH_GRAD

template <typename scalar_t>
__global__ void PISO_build_matrix_GRAD(DomainGPU<scalar_t> *p_domain,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks,
		const int8_t nonOrthoFlags, const bool forPassiveScalar, const index_t passiveScalarChannel){
	
	const scalar_t half = static_cast<scalar_t>(0.5);
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		const scalar_t timeStep = s_domain.timeStep; // per batched environment
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		const index_t flatPosGlobal = flattenIndexGlobal(pos, s_block, s_domain);
		
		const scalar_t det = s_block.hasTransform ? getDeterminantDimSwitch(s_block, pos, s_domain.numDims) : 1;
		const scalar_t rDet = 1/det;
		
		// load row from advection CSR matrix grad
		scalar_t csrValuesGrad[7] = {0}; // to be loaded like: diag,-x,+x,-y,+y,-z,+z
		LoadCSRrowNeighborSorted(pos, s_domain.C_grad, s_domain, s_block, csrValuesGrad);
		for(index_t i=0; i<(s_domain.numDims*2+1); ++i){
			csrValuesGrad[i] *= rDet;
		}
		
		// add gradient from separate diagonal
		//csrValuesGrad[0] += s_domain.Adiag_grad[flatPosGlobal];
		scalar_t diagGrad = s_domain.Adiag_grad[flatPosGlobal] * rDet + csrValuesGrad[0];
		
		
		scalar_t fluxesGrad[7] = {0};

		// Forward fluxes, needed only to reproduce the upwind branch taken by
		// PISO_build_matrix: which side of the face the flux was assigned to
		// depends on its sign. CENTRAL splits 50/50 and needs no branch, so skip
		// the recompute entirely in that case.
		const bool deferredAdvection = s_domain.advectionScheme != AdvectionScheme::CENTRAL;
		scalar_t advFluxes[6];
		if(deferredAdvection){
			computeFluxesNDLoop<scalar_t>(pos, advFluxes, s_block, s_domain, nullptr);
		}

		scalar_t viscosity_grad = 0;
		scalar_t alpha_p[3] = {1,1,1};
		for(index_t dim=0;dim<s_domain.numDims;++dim){
			I4 tempPos = pos;
			tempPos.w = dim;
			alpha_p[dim] = getLaplaceCoefficientOrthogonalDimSwitch(tempPos, &s_block, s_domain.numDims);
		}
		
		
		for(index_t bound=0; bound<(s_domain.numDims*2); ++bound){
			const index_t dim = axisFromBound(bound);
			const index_t isUpper = boundIsUpper(bound);
			const index_t faceSign = faceSignFromBound(bound);
			
			const bool atBound = isAtBound(pos, bound, s_block);
			if(!atBound || !isEmptyBound(bound, s_block.boundaries)){
				
				// flux grad and orthogonal diffusion grad
				{
					//calculate index of neighbour
					I4 tempPos = pos;
					tempPos.w = dim;
					scalar_t alpha = 1;
					const BlockGPU<scalar_t> *p_block = &s_block;
					// resolve neighbor cell
					if(atBound && s_block.boundaries[bound].type==BoundaryType::CONNECTED_GRID){
						p_block = s_domain.blocks + s_block.boundaries[bound].cb.connectedGridIndex;
						tempPos = computeConnectedPosWithChannel(tempPos, dim, &s_block.boundaries[bound].cb, s_domain);
					}else{
						if(atBound && s_block.boundaries[bound].type==BoundaryType::PERIODIC){
							tempPos.a[dim] = isUpper ? 0 : s_block.size.a[dim]-1;
						}else{
							tempPos.a[dim] = pos.a[dim] + faceSign;
						}
					}
					// need correct component/channel for laplace coefficient
					alpha = getLaplaceCoefficientOrthogonalDimSwitch(tempPos, p_block, s_domain.numDims);
					
					// Adjoint of the advective flux assignment in PISO_build_matrix.
					// CENTRAL puts half the flux on each side; LINEAR_UPWIND puts
					// all of it on the upwind side, so the adjoint picks up only that
					// side's gradient. The branch is frozen (sign(faceFluxOut) is treated
					// as constant), which is exact away from faceFluxOut == 0.
					if(deferredAdvection){
						const scalar_t faceFluxOut = faceSign * advFluxes[bound];
						if(hasFarUpwindCell(pos, bound, faceFluxOut >= 0, s_block)){
							fluxesGrad[bound] = faceSign * (faceFluxOut >= 0 ? diagGrad : csrValuesGrad[bound+1]);
						}else{
							fluxesGrad[bound] = faceSign * half * (diagGrad + csrValuesGrad[bound+1]);
						}
					}else{
						fluxesGrad[bound] = faceSign * half * (diagGrad + csrValuesGrad[bound+1]);
					}

					const scalar_t viscosityCoeff_grad = diagGrad - csrValuesGrad[bound+1];
					//viscosity_grad += half * (alpha_p[dim] + alpha) * viscosityCoeff_grad;
					viscosity_grad += half * alpha_p[dim] * viscosityCoeff_grad;
					const scalar_t viscosity_n_grad = half * alpha * viscosityCoeff_grad;
					tempPos.w = forPassiveScalar ? passiveScalarChannel : 0;
					scatterViscosityBlock_GRAD<scalar_t>(viscosity_n_grad, tempPos, p_block, s_domain, forPassiveScalar, passiveScalarChannel);
				
				}
				
				// non-ortho diffusion grad
				const bool includeNonOrthoNeighbors = nonOrthoFlags & NON_ORTHO_DIRECT_MATRIX;
				const bool includeNonOrthoDiag = nonOrthoFlags & NON_ORTHO_CENTER_MATRIX;
				if(s_domain.numDims>1 && (includeNonOrthoDiag || includeNonOrthoNeighbors)){
					scalar_t alphaInterp[12]; // size for 3D
					scalar_t alphaInterp_grad[12] = {0};
					if(s_domain.numDims==2) interpolateNonOrthoLaplaceComponents<scalar_t, 2>(pos, s_block, s_domain, alphaInterp, false, false, false);
					else if(s_domain.numDims==3) interpolateNonOrthoLaplaceComponents<scalar_t, 3>(pos, s_block, s_domain, alphaInterp, false, false, false);
					
					I4 channelPos = pos;
					GridDataType dataType = GridDataType::VELOCITY;
					if(forPassiveScalar){
						channelPos.w = passiveScalarChannel;
						dataType = GridDataType::PASSIVE_SCALAR;
					}
					
					for(index_t i=1; i<s_domain.numDims; ++i){ // loop other axes
						const index_t tAxis = (dim + i)%s_domain.numDims;
						const scalar_t alpha = getInterpolatedNonOrthoLaplaceComponent(alphaInterp, bound, tAxis, s_domain.numDims);
						
						if(alpha!=0){ // grid is non-orthogonal here
							scalar_t alphaVisc_grad = 0;
							for(index_t tIsUpper=0; tIsUpper<2; ++tIsUpper){ // loop corners
								const index_t tFace = axisToBound(tAxis, tIsUpper); //(tAxis<<1) + tIsUpper;
								const index_t tFaceSign = faceSignFromBound(tFace);
								const CornerValue<scalar_t> cVal = getCornerValue<scalar_t>(channelPos, bound, tFace,
									false, false, 2, s_block, s_domain, dataType); // computes interpolation divisor and checks for boundaries.
								const bool cornerAtBound = cVal.numCells<1;
								const bool boundIsGradient = cVal.boundType==BoundaryConditionType::NEUMANN;
								
								if(cornerAtBound){
									if(boundIsGradient){
										// simplified treatment: ignore boudnary gradient and use one-sided difference from other side
										const scalar_t interpolationNorm = 0.25; // from other side, can't be anything else but 1/4
										const index_t tFaceOther = invertBound(tFace);
										//const scalar_t viscosityCoeff = faceSign * tFaceSign * viscosity * alpha * interpolationNorm;
										scalar_t viscosityCoeff_grad = 0;
										if(includeNonOrthoDiag){
											//diag -= 3 * viscosityCoeff;
											viscosityCoeff_grad -= 3 * diagGrad;
										}
										if(includeNonOrthoNeighbors){
											//rowValues[bound+1] -= 3 * viscosityCoeff;
											viscosityCoeff_grad -= 3 * csrValuesGrad[bound+1];
											//rowValues[tFaceOther+1] += viscosityCoeff;
											viscosityCoeff_grad += csrValuesGrad[tFaceOther+1];
										}
										// if diagonals where to be included in the matrix: 
										// rowValues[diagonalOther] += viscosityCoeff;
										alphaVisc_grad += faceSign * tFaceSign * interpolationNorm * viscosityCoeff_grad;
									}
									// else: prescribed value, added on RHS, nothing to do here.
								} else {
									// normal corner with interpolated value
									const scalar_t interpolationNorm = 1.0 / static_cast<scalar_t>(cVal.numCells);
									//const scalar_t viscosityCoeff = faceSign * tFaceSign * viscosity * alpha * interpolationNorm;
									scalar_t viscosityCoeff_grad = 0;
									if(includeNonOrthoDiag){
										//diag -= viscosityCoeff;
										viscosityCoeff_grad -= diagGrad;
									}
									if(includeNonOrthoNeighbors){
										//rowValues[bound+1] -= viscosityCoeff;
										viscosityCoeff_grad -= csrValuesGrad[bound+1];
										//rowValues[tFace+1] -= viscosityCoeff;
										viscosityCoeff_grad -= csrValuesGrad[tFace+1];
									}
									// if diagonals where to be included in the matrix: 
									// rowValues[diagonal] -= viscosityCoeff;
									alphaVisc_grad += faceSign * tFaceSign * interpolationNorm * viscosityCoeff_grad;
								}
							}
							if(forPassiveScalar){
								viscosity_grad += alphaVisc_grad * alpha;
							} else {
								//viscosity_grad += alphaVisc_grad * alpha;
								addInterpolatedNonOrthoLaplaceComponent_GRAD(alphaVisc_grad, alphaInterp_grad, bound, tAxis, s_domain.numDims);
							}
						}
					}
					if(!forPassiveScalar){
						if(s_domain.numDims==2) scatterNonOrthoLaplaceComponents_GRAD<scalar_t, 2>(alphaInterp_grad, pos, s_block, s_domain, true, false, false);
						else if(s_domain.numDims==3) scatterNonOrthoLaplaceComponents_GRAD<scalar_t, 3>(alphaInterp_grad, pos, s_block, s_domain, true, false, false);
					}
				}
				
			} else { // face is prescribed boundary
				index_t slip = 0;
				switch(s_block.boundaries[bound].type){
					case BoundaryType::DIRICHLET:
					{
						slip = s_block.boundaries[bound].sdb.slip;
						break;
					}
					case BoundaryType::DIRICHLET_VARYING:
					{
						slip = s_block.boundaries[bound].vdb.slip;
						break;
					}
					case BoundaryType::FIXED:
					{
						BoundaryConditionType boundaryType; // = s_block.boundaries[bound].fb.velocity.boundaryType;
						if(forPassiveScalar){
							I4 tempPos = pos;
							tempPos.w = passiveScalarChannel;
							boundaryType = getFixedBoundaryType(tempPos, bound, &s_block, GridDataType::PASSIVE_SCALAR);
						}else{
							boundaryType = s_block.boundaries[bound].fb.velocity.boundaryType;
						}
						//const BoundaryConditionType boundaryType = s_block.boundaries[bound].fb.velocity.boundaryType;
						slip = boundaryType==BoundaryConditionType::DIRICHLET ? 0 : 1;
						break;
					}
				}
				//diag += (1 + 1 - slip*2) * viscosity * alpha_p[dim];
				viscosity_grad += (1 - slip)*2  * alpha_p[dim] * diagGrad;
				
			}
			
		}
		
		ScatterFluxesGradNDLoop<scalar_t>(pos, fluxesGrad, s_block, s_domain, nullptr);
		{
			I4 tempPos = pos;
			tempPos.w = forPassiveScalar ? passiveScalarChannel : 0;
			scatterViscosityBlock_GRAD<scalar_t>(viscosity_grad, tempPos, &s_block, s_domain, forPassiveScalar, passiveScalarChannel);
		}
	)
}

#endif //WITH_GRAD

template <typename scalar_t>
__global__ void kPISO_build_scalar_advection_RHS(DomainGPU<scalar_t> *p_domain,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks,
		const int8_t nonOrthoFlags){
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		const scalar_t timeStep = s_domain.timeStep; // per batched environment
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		//weird issue: memory access violation if flatPos is used directly.
		
		const scalar_t det = s_block.hasTransform ? getDeterminantDimSwitch(s_block, pos, s_domain.numDims) : 1;

		// The scalar matrix is built by the same PISO_build_matrix, so scalar
		// transport uses the same convective scheme as the momentum equation and
		// needs the matching deferred correction. Without it the scalar would be
		// left on plain first-order upwind.
		const bool deferredAdvection = s_domain.advectionScheme != AdvectionScheme::CENTRAL;
		scalar_t advFluxes[6];
		if(deferredAdvection){
			computeFluxesNDLoop<scalar_t>(pos, advFluxes, s_block, s_domain, nullptr);
		}

		for(index_t channel=0; channel<s_domain.passiveScalarChannels; ++channel){
			I4 tempPos = pos;
			tempPos.w = channel;

			//const scalar_t viscosity = getViscosity(s_domain, s_domain.scalarViscosity!=nullptr, channel);

			const int tempFlatPos = flattenIndex(tempPos, s_block);
			scalar_t tempRHS = det * s_block.scalarData[tempFlatPos]/timeStep;
			//TODO: pressure, external forces?
			//if(flatPos<27 && s_block.globalOffset==0)
				
			for(index_t bound=0; bound<(s_domain.numDims*2); ++bound){ 
				if(((bound&1)==0 && pos.a[bound>>1]==0) || ((bound&1)==1 && pos.a[bound>>1]==s_block.size.a[bound>>1]-1)){
					
					const scalar_t faceNormal = static_cast<scalar_t>((bound&1)*2 -1);
					//const scalar_t isTangentialDir = static_cast<scalar_t>((bound>>1)!=dim); // * 2 -1;
					switch(s_block.boundaries[bound].type){
					case BoundaryType::FIXED:
					{
						I4 boundPos = pos;
						boundPos.w = channel;
						boundPos.a[bound>>1] = 0;
						//const index_t flatTempPos = flattenIndex(boundPos, s_block.boundaries[bound].vdb.stride);
						const scalar_t scalar = getFixedBoundaryData(boundPos, bound, &s_block, s_domain, GridDataType::PASSIVE_SCALAR); // Dirichlet: face value; Neumann: dphi/dn (physical gradient, outward normal)
						const BoundaryConditionType scalarBoundaryType = getFixedBoundaryType(boundPos, bound, &s_block, GridDataType::PASSIVE_SCALAR);

						const FixedBoundaryGPU<scalar_t> *p_fb = &(s_block.boundaries[bound].fb);

						boundPos.w = bound>>1;
						const scalar_t alpha = getLaplaceCoefficientOrthogonalBoundaryFixedDimSwitch(boundPos, p_fb, s_domain.numDims);
						//const scalar_t flux = s_block.boundaries[bound].vdb.velocity[flattenIndex(boundPos, s_block.boundaries[bound].vdb.stride)] * faceNormal;
						const scalar_t flux = getContravariantComponentBoundaryFixedDimSwitch(boundPos, p_fb, s_domain) * faceNormal;

						// from viscosity
						const scalar_t viscosity = getViscosityFixedBoundary(tempPos, p_fb, &s_block, s_domain, true, channel);
						if(scalarBoundaryType==BoundaryConditionType::DIRICHLET){
							// from advection: scalar is the prescribed face value
							tempRHS -= scalar * flux;
							// from diffusion: ghost cell method
							tempRHS += scalar * viscosity * 2 * alpha;
						}else{ // NEUMANN: scalar is dphi/dn (physical gradient w.r.t. outward normal)
							// from advection: use interior cell value as face value (zero-order extrapolation)
							tempRHS -= s_block.scalarData[tempFlatPos] * flux;
							// from diffusion: flux = (dphi/dn) * kappa * |S_face|
							const scalar_t faceArea = getFaceAreaOrthogonalBoundaryFixedDimSwitch(boundPos, p_fb, s_domain.numDims);
							tempRHS += scalar * viscosity * faceArea;
						}
						
						break;
					}
					case BoundaryType::VALUE:
					{
						// from advection
						tempRHS -= s_block.boundaries[bound].sdb.scalar * s_block.boundaries[bound].sdb.velocity.a[bound>>1] * faceNormal;
						// from viscosity
						const scalar_t viscosity = getViscosity(s_domain, s_domain.scalarViscosity!=nullptr, channel);
						tempRHS += (1-s_block.boundaries[bound].vdb.slip) * viscosity * 2 * s_block.boundaries[bound].sdb.scalar; //* isTangentialDir
						break;
					}
					case BoundaryType::DIRICHLET_VARYING:
					{
						I4 tempPos = pos;
						tempPos.w = 0; // VaryingDirichletBoundary does not support channels
						tempPos.a[bound>>1] = 0;
						//const index_t flatTempPos = flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride);
						const scalar_t scalar = s_block.boundaries[bound].vdb.scalar[flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride)];
						tempPos.w = bound>>1;
						const scalar_t alpha = getLaplaceCoefficientOrthogonalBoundaryVaryingDimSwitch(tempPos, &s_block.boundaries[bound].vdb, s_domain.numDims);
						//const scalar_t flux = s_block.boundaries[bound].vdb.velocity[flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride)] * faceNormal;
						const scalar_t flux = getContravariantComponentBoundaryVaryingDimSwitch(tempPos, &s_block.boundaries[bound].vdb, s_domain) * faceNormal;
						
						// from advection
						tempRHS -= scalar * flux;
						// from viscosity
						const scalar_t viscosity = getViscosity(s_domain, s_domain.scalarViscosity!=nullptr, channel);
						tempRHS += scalar * (1-s_block.boundaries[bound].vdb.slip) * viscosity * 2 * alpha; //* isTangentialDir
						break;
					}
					default:
						break;
					}
				}
			}
			
			
			if(deferredAdvection){
				// Mirrors the velocity version in kPISO_build_advection_RHS: the matrix
				// holds first-order upwind, so put -F*(phi_f^HO - phi_f^UD) on the RHS
				// with phi_f^HO - phi_f^UD = 0.5*psi(r)*(phi_D - phi_U). Likewise lagged
				// to the start of the step.
				for(index_t bound=0; bound<(s_domain.numDims*2); ++bound){
					const index_t axis = axisFromBound(bound);
					const index_t faceSign = faceSignFromBound(bound);
					const scalar_t faceFluxOut = faceSign * advFluxes[bound];
					const bool outflow = faceFluxOut >= 0;
					// Faces the matrix kept central need no correction.
					if(!hasFarUpwindCell(pos, bound, outflow, s_block)){ continue; }

					I4 cellPos = pos;
					cellPos.w = channel;
					const scalar_t phiP = s_block.scalarData[flattenIndex(cellPos, s_block)];
					cellPos.a[axis] = pos.a[axis] + faceSign;
					const scalar_t phiN = s_block.scalarData[flattenIndex(cellPos, s_block)];
					cellPos.a[axis] = outflow ? (pos.a[axis] - faceSign) : (pos.a[axis] + 2*faceSign);
					const scalar_t phiUU = s_block.scalarData[flattenIndex(cellPos, s_block)];

					const scalar_t phiU = outflow ? phiP : phiN;
					const scalar_t phiD = outflow ? phiN : phiP;
					const scalar_t dD = phiD - phiU;
					const scalar_t absDD = dD < 0 ? -dD : dD;
					if(absDD < static_cast<scalar_t>(1e-20)){ continue; }

					// LINEAR_UPWIND: psi = r
					const scalar_t psi = (phiU - phiUU) / dD;
					tempRHS -= faceFluxOut * static_cast<scalar_t>(0.5) * psi * dD;
				}
			}

			// getNonOrthoLaplaceRHS assumes the laplace term to be positive/added on the LHS and returns negated coefficients for the RHS.
			// The diffusion term is negative on the LHS, so consequently the RHS has to be subtracted.
			//tempRHS -= getNonOrthoLaplaceRHSDimSwitch<scalar_t>(pos, s_block, s_domain, nonOrthoFlags, GridDataType::PASSIVE_SCALAR_RESULT, false) * s_domain.viscosity;
			const scalar_t viscosity = getViscosityBlock(tempPos, &s_block, s_domain, true, channel);
			tempRHS -= getNonOrthoLaplaceRHSDimSwitch_v2<scalar_t>(tempPos, s_block, s_domain, nonOrthoFlags, GridDataType::PASSIVE_SCALAR_RESULT, false, false, false) * viscosity;
			
			tempRHS /= det; // in the used formulation, the advection and diffusion parts have to be divided by the determinant. Other source terms are unaffected, so add them below.
			
			const index_t flatPosGlobal = flattenIndexGlobal(tempPos, s_block, s_domain);
			s_domain.scalarRHS[flatPosGlobal] = tempRHS;
		}
	)
}

#ifdef WITH_GRAD
template <typename scalar_t>
__global__ void kPISO_build_scalar_advection_RHS_GRAD(DomainGPU<scalar_t> *p_domain,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks,
		const int8_t nonOrthoFlags){
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		const scalar_t timeStep = s_domain.timeStep; // per batched environment
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		
		const scalar_t det = s_block.hasTransform ? getDeterminantDimSwitch(s_block, pos, s_domain.numDims) : 1;
		
		for(index_t channel=0; channel<s_domain.passiveScalarChannels; ++channel){
			I4 tempPos = pos;
			tempPos.w = channel;
			
			const scalar_t viscosity = getViscosity(s_domain, s_domain.scalarViscosity!=nullptr, channel);
			
			const index_t flatPosGlobal = flattenIndexGlobal(tempPos, s_block, s_domain);
		
			scalar_t RHSgrad = s_domain.scalarRHS_grad[flatPosGlobal];
			
			RHSgrad /= det;
			
			// non-ortho laplace gradients w.r.t. scalarResult
			scatterNonOrthoLaplaceRHSDimSwitch_v2_GRAD<scalar_t>(-RHSgrad * viscosity, tempPos, s_block, s_domain, nonOrthoFlags,
				GridDataType::PASSIVE_SCALAR_RESULT_GRAD, false, false, false);
			
			// non-ortho laplace gradients w.r.t. viscosity
			
			scalar_t viscosity_grad = -RHSgrad * getNonOrthoLaplaceRHSDimSwitch_v2<scalar_t>(tempPos, s_block, s_domain,
				nonOrthoFlags, GridDataType::PASSIVE_SCALAR_RESULT, false, false, false);
			
			
			// gradients from/for boundaries
			for(index_t bound=0; bound<(s_domain.numDims*2); ++bound){ 
				if(((bound&1)==0 && pos.a[bound>>1]==0) || ((bound&1)==1 && pos.a[bound>>1]==s_block.size.a[bound>>1]-1)){
					const scalar_t faceNormal = static_cast<scalar_t>((bound&1)*2 -1);
					if(!(s_block.boundaries[bound].type==BoundaryType::FIXED)){ continue; }
					
					// forward values
					
					I4 tempPos = pos;
					tempPos.w = channel;
					tempPos.a[bound>>1] = 0;
					//const index_t flatTempPos = flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride);
					const scalar_t scalar = getFixedBoundaryData(tempPos, bound, &s_block, s_domain, GridDataType::PASSIVE_SCALAR);
					const BoundaryConditionType scalarBoundaryType = getFixedBoundaryType(tempPos, bound, &s_block, GridDataType::PASSIVE_SCALAR);
					
					tempPos.w = bound>>1;
					const scalar_t alpha = getLaplaceCoefficientOrthogonalBoundaryFixedDimSwitch(tempPos, &s_block.boundaries[bound].fb, s_domain.numDims);
					//const scalar_t flux = s_block.boundaries[bound].vdb.velocity[flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride)] * faceNormal;
					const scalar_t flux = getContravariantComponentBoundaryFixedDimSwitch(tempPos, &s_block.boundaries[bound].fb, s_domain) * faceNormal;
					
					
					// gradients
					
					scalar_t flux_grad = 0;
					// from advection
					flux_grad -= scalar * RHSgrad;
					
					scalar_t scalar_grad = 0;
					// from advection
					scalar_grad -= flux * RHSgrad;
					// from viscosity
					if(scalarBoundaryType==BoundaryConditionType::DIRICHLET){
						scalar_grad += viscosity * 2 * alpha * RHSgrad;
						viscosity_grad += scalar * 2 * alpha * RHSgrad;
					} else {
						// TODO: The Neumann path needs updating to match the forward kernel fix:
						// - advection grad should scatter to scalarData_grad (interior value), not scalar_grad (boundary data)
						// - diffusion grad should use faceArea = getFaceAreaOrthogonalBoundaryFixedDimSwitch(...)
						scalar_grad += viscosity * RHSgrad;
						viscosity_grad += scalar * RHSgrad;
					}
					
					// scatter flux grad
					scatterContravariantComponentBoundaryFixedDimSwitch(flux_grad*faceNormal, tempPos, &s_block.boundaries[bound].fb, s_domain);
					//scatter scalar grad
					tempPos.w = channel;
					scatterFixedBoundaryData(scalar_grad, tempPos, bound, &s_block, s_domain, GridDataType::PASSIVE_SCALAR_GRAD);
					
				}
			}
			
			scatterViscosity_GRAD(viscosity_grad, s_domain, s_domain.scalarViscosity_grad!=nullptr, channel);
			
			
			RHSgrad *= det;
			
			const int tempFlatPos = flattenIndex(tempPos, s_block);
			s_block.scalarData_grad[tempFlatPos] = RHSgrad/timeStep;
			
		}
		
	)
}
#endif //WITH_GRAD

template <typename scalar_t, int DIMS>
__global__ void kPISO_build_advection_RHS(DomainGPU<scalar_t> *p_domain,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks,
		const int8_t nonOrthoFlags, const bool applyPressureGradient){
	
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		const scalar_t timeStep = s_domain.timeStep; // per batched environment
	{ //<- necessary for the macro to work with multiple-argument templates?
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		const scalar_t det = s_block.hasTransform ? getDeterminant<scalar_t, DIMS>(s_block, pos) : 1;

		// Face fluxes for the deferred high-order convection correction. Same
		// fluxes the matrix is built from, so both agree on the upwind direction.
		const bool deferredAdvection = s_domain.advectionScheme != AdvectionScheme::CENTRAL;
		scalar_t advFluxes[6];
		if(deferredAdvection){
			computeFluxesNDLoop<scalar_t>(pos, advFluxes, s_block, s_domain, nullptr);
		}

		for(index_t dim=0;dim<DIMS;++dim){
			I4 tempPos = pos;
			tempPos.w = dim;
			const index_t tempFlatPos = flattenIndex(tempPos, s_block);
			scalar_t tempRHS = det * s_block.velocity[tempFlatPos]/timeStep; // /c_domain.timeStep
			
			
			//Source terms: external forces, boundary velocity

			//tempPos.w = dim;
			//tempPos.a[dim] = pos.a[dim];
			for(index_t bound=0; bound<(s_domain.numDims*2); ++bound){ 
				if(((bound&1)==0 && pos.a[bound>>1]==0) || ((bound&1)==1 && pos.a[bound>>1]==s_block.size.a[bound>>1]-1)){
					const scalar_t faceNormal = static_cast<scalar_t>((bound&1)*2 -1);
					//const scalar_t isTangentialDir = static_cast<scalar_t>((bound>>1)!=dim); // * 2 -1;
					switch(s_block.boundaries[bound].type){
					case BoundaryType::FIXED:
					{
						I4 tempPos = pos;
						tempPos.w = dim;
						tempPos.a[bound>>1] = 0;
						//const index_t flatTempPos = flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride);
						const scalar_t vel = getFixedBoundaryData(tempPos, bound, &s_block, s_domain, GridDataType::VELOCITY);
						tempPos.w = bound>>1;
						const FixedBoundaryGPU<scalar_t> *p_fb = &(s_block.boundaries[bound].fb);
						const scalar_t alpha = getLaplaceCoefficientOrthogonalBoundaryFixedDimSwitch(tempPos, p_fb, s_domain.numDims);
						//const scalar_t flux = s_block.boundaries[bound].vdb.velocity[flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride)] * faceNormal;
						const scalar_t flux = getContravariantComponentBoundaryFixedDimSwitch(tempPos, p_fb, s_domain) * faceNormal;
						
						// from advection
						tempRHS -= vel * flux;
						// from viscosity
						const scalar_t boundViscosity = getViscosityFixedBoundary(pos, p_fb, &s_block, s_domain, false, 0);
						// TODO: simple slip from boundary type
						const scalar_t slip = s_block.boundaries[bound].fb.velocity.boundaryType==BoundaryConditionType::DIRICHLET ? 0 : 1;
						tempRHS += vel * (1-slip) * boundViscosity * 2 * alpha; //* isTangentialDir
						break;
					}
					case BoundaryType::VALUE:
					{
						// from advection
						tempRHS -= s_block.boundaries[bound].sdb.velocity.a[dim] * s_block.boundaries[bound].sdb.velocity.a[bound>>1] * faceNormal;
						// from viscosity
						const scalar_t viscosity = getViscosityBlock(pos, &s_block, s_domain, false, 0);
						tempRHS += (1-s_block.boundaries[bound].vdb.slip) * viscosity * 2 * s_block.boundaries[bound].sdb.velocity.a[dim]; //* isTangentialDir
						break;
					}
					case BoundaryType::DIRICHLET_VARYING:
					{
						I4 tempPos = pos;
						tempPos.w = dim;
						tempPos.a[bound>>1] = 0;
						//const index_t flatTempPos = flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride);
						const scalar_t vel = s_block.boundaries[bound].vdb.velocity[flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride)];
						tempPos.w = bound>>1;
						const scalar_t alpha = getLaplaceCoefficientOrthogonalBoundaryVaryingDimSwitch(tempPos, &s_block.boundaries[bound].vdb, s_domain.numDims);
						//const scalar_t flux = s_block.boundaries[bound].vdb.velocity[flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride)] * faceNormal;
						const scalar_t flux = getContravariantComponentBoundaryVaryingDimSwitch(tempPos, &s_block.boundaries[bound].vdb, s_domain) * faceNormal;
						
						// from advection
						tempRHS -= vel * flux;
						// from viscosity
						const scalar_t viscosity = getViscosityBlock(pos, &s_block, s_domain, false, 0);
						tempRHS += vel * (1-s_block.boundaries[bound].vdb.slip) * viscosity * 2 * alpha; //* isTangentialDir
						break;
					}
					default:
						break;
					}
				}
			}
			
			if(deferredAdvection){
				// Deferred correction. The matrix carries first-order upwind, so put
				// -F*(phi_f^HO - phi_f^UD) on the RHS, with phi_f^HO - phi_f^UD =
				// 0.5*psi(r)*(phi_D - phi_U). Same sign convention as the boundary
				// advection terms above (the LHS holds +F*phi_f).
				//
				// Evaluated from the block velocity, which is the start-of-step field:
				// the advection loops update velocityResult, not the block data, so the
				// correction stays lagged by one step even when the caller iterates
				// (advect_non_ortho_steps). That is the usual explicit treatment of a
				// high-order convection correction, and for a run driven to steady state
				// the lag vanishes exactly at convergence. Switch the reads to
				// s_domain.velocityResult if the correction should follow the inner
				// iterations instead.
				for(index_t bound=0; bound<(s_domain.numDims*2); ++bound){
					const index_t axis = axisFromBound(bound);
					const index_t faceSign = faceSignFromBound(bound);
					const scalar_t faceFluxOut = faceSign * advFluxes[bound];
					const bool outflow = faceFluxOut >= 0;
					// Faces the matrix kept central need no correction.
					if(!hasFarUpwindCell(pos, bound, outflow, s_block)){ continue; }

					I4 cellPos = pos;
					cellPos.w = dim;
					const scalar_t phiP = s_block.velocity[flattenIndex(cellPos, s_block)];
					cellPos.a[axis] = pos.a[axis] + faceSign;
					const scalar_t phiN = s_block.velocity[flattenIndex(cellPos, s_block)];
					cellPos.a[axis] = outflow ? (pos.a[axis] - faceSign) : (pos.a[axis] + 2*faceSign);
					const scalar_t phiUU = s_block.velocity[flattenIndex(cellPos, s_block)];

					const scalar_t phiU = outflow ? phiP : phiN;
					const scalar_t phiD = outflow ? phiN : phiP;
					const scalar_t dD = phiD - phiU;
					const scalar_t absDD = dD < 0 ? -dD : dD;
					// Flat face: phi_f = phi_U for any psi, so there is nothing to add.
					// Skipping also keeps r finite.
					if(absDD < static_cast<scalar_t>(1e-20)){ continue; }

					// LINEAR_UPWIND: psi = r
					const scalar_t psi = (phiU - phiUU) / dD;
					tempRHS -= faceFluxOut * static_cast<scalar_t>(0.5) * psi * dD;
				}
			}

			//tempRHS -= getNonOrthoLaplaceRHSDimSwitch<scalar_t>(pos, s_block, s_domain, nonOrthoFlags, GridDataType::VELOCITY_RESULT, false) * s_domain.viscosity;
			tempRHS -= getNonOrthoLaplaceRHSDimSwitch_v2<scalar_t>(tempPos, s_block, s_domain, nonOrthoFlags, GridDataType::VELOCITY_RESULT, true, false, false); //* viscosity;

			tempRHS /= det; // in the used formulation, the advection and diffusion parts have to be divided by the determinant. Other source terms are unaffected, so add them below.

			tempRHS += getBlockVelocitySource(tempPos, &s_block);
			
			if(applyPressureGradient){
				const Vector<scalar_t, DIMS> pressureGrad = getPressureGradient<scalar_t, DIMS>(s_block, pos, s_domain);
				//const Vector<scalar_t, DIMS> pressureGrad = getPressureGradientFVM<scalar_t, DIMS>(s_block, pos, s_domain, 0);
				tempRHS -= pressureGrad.a[dim] * timeStep;
			}
			
			const index_t flatPosGlobal = flattenIndexGlobal(tempPos, s_block, s_domain);
			s_domain.velocityRHS[flatPosGlobal] = tempRHS;
			//s_domain.velocityRHS[flatPos + s_block.globalOffset + s_domain.numCells*dim] = tempRHS;
		}
	})
}

#ifdef WITH_GRAD
template <typename scalar_t, int DIMS>
__global__ void kPISO_build_advection_RHS_GRAD(DomainGPU<scalar_t> *p_domain,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks,
		const int8_t nonOrthoFlags, const bool applyPressureGradient){
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		const scalar_t timeStep = s_domain.timeStep; // per batched environment
	 //<- necessary for the macro to work with multiple-argument templates...
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		
		const scalar_t det = s_block.hasTransform ? getDeterminant<scalar_t, DIMS>(s_block, pos) : 1;
		const scalar_t viscosity = getViscosityBlock(pos, &s_block, s_domain, false, 0);
		
		for(index_t dim=0;dim<DIMS;++dim){
			I4 tempPos = pos;
			tempPos.w = dim;
			const index_t tempFlatPos = flattenIndex(tempPos, s_block);
			
			scalar_t RHSgrad = s_domain.velocityRHS_grad[flatPos + s_block.globalOffset + s_domain.numCells*dim];

			scatterBlockVelocitySource_GRAD(RHSgrad, tempPos, &s_block);
			
			RHSgrad /= det;
			
			// non-ortho laplace gradients w.r.t. velocityResult
			scatterNonOrthoLaplaceRHSDimSwitch_v2_GRAD<scalar_t>(-RHSgrad, tempPos, s_block, s_domain, nonOrthoFlags,
				GridDataType::VELOCITY_RESULT_GRAD, true, false, false);
			
			// gradients from/for boundaries
			for(index_t bound=0; bound<(s_domain.numDims*2); ++bound){ 
				if(((bound&1)==0 && pos.a[bound>>1]==0) || ((bound&1)==1 && pos.a[bound>>1]==s_block.size.a[bound>>1]-1)){
					const scalar_t faceNormal = static_cast<scalar_t>((bound&1)*2 -1);
					if(!(s_block.boundaries[bound].type==BoundaryType::FIXED)){ continue; }
					
					// forward values
					
					I4 tempPos = pos;
					tempPos.w = dim;
					tempPos.a[bound>>1] = 0;
					//const index_t flatTempPos = flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride);
					const scalar_t vel = getFixedBoundaryData(tempPos, bound, &s_block, s_domain, GridDataType::VELOCITY);
					tempPos.w = bound>>1;
					const FixedBoundaryGPU<scalar_t> *p_fb = &(s_block.boundaries[bound].fb);
					const scalar_t alpha = getLaplaceCoefficientOrthogonalBoundaryFixedDimSwitch(tempPos, p_fb, s_domain.numDims);
					//const scalar_t flux = s_block.boundaries[bound].vdb.velocity[flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride)] * faceNormal;
					const scalar_t flux = getContravariantComponentBoundaryFixedDimSwitch(tempPos, p_fb, s_domain) * faceNormal;
					
					// from advection
					//tempRHS -= vel * flux;
					// from viscosity
					// TODO: simple slip from boundary type
					const scalar_t slip = s_block.boundaries[bound].fb.velocity.boundaryType==BoundaryConditionType::DIRICHLET ? 0 : 1;
					//tempRHS += vel * (1-slip) * s_domain.viscosity * 2 * alpha; //* isTangentialDir
					
					// gradients
					
					scalar_t flux_grad = 0;
					// from advection
					flux_grad -= vel * RHSgrad;
					
					scalar_t vel_grad = 0;
					// from advection
					vel_grad -= flux * RHSgrad;
					// from viscosity
					const scalar_t boundViscosity = getViscosityFixedBoundary<scalar_t>(pos, p_fb, &s_block, s_domain, false, 0);
					vel_grad += (1-slip) * boundViscosity * 2 * alpha * RHSgrad;
					
					const scalar_t boundViscosity_grad = vel * (1-slip) * 2 * alpha * RHSgrad;
					
					// scatter flux grad
					scatterContravariantComponentBoundaryFixedDimSwitch<scalar_t>(flux_grad*faceNormal, tempPos, &s_block.boundaries[bound].fb, s_domain);
					//scatter vel grad
					tempPos.w = dim;
					scatterFixedBoundaryData<scalar_t>(vel_grad, tempPos, bound, &s_block, s_domain, GridDataType::VELOCITY_GRAD);
					//scatter viscosity grad
					scatterViscosityBoundary_GRAD<scalar_t>(boundViscosity_grad, pos, p_fb, &s_block, s_domain, false, 0);
					
				}
			}
			
			//scatterViscosityBlock_GRAD<scalar_t>(viscosity_grad, pos, &s_block, s_domain, false, 0);
			
			
			RHSgrad *= det;
			
			s_block.velocity_grad[tempFlatPos] = RHSgrad / timeStep; // * det
		}
	)
}

/* Adjoint of the deferred high-order correction in kPISO_build_advection_RHS /
 * kPISO_build_scalar_advection_RHS. LINEAR_UPWIND only; CENTRAL builds no
 * correction, so this kernel is then simply not launched.
 *
 * For LINEAR_UPWIND psi = r = (phi_U - phi_UU)/dD, so the correction collapses to
 *     tempRHS -= F * 0.5 * psi * dD = F * (-0.5) * (phi_U - phi_UU)
 * which is *linear* in the field: dD cancels and there is no 1/dD in the adjoint.
 * The derivatives are therefore constants,
 *     d/d(phi_U) = -0.5*F,  d/d(phi_UU) = +0.5*F,  d/d(phi_D) = 0,
 * plus d/dF = -0.5*(phi_U - phi_UU) routed through ScatterFluxesGradNDLoop.
 *
 * This runs as a *separate* kernel launched after the main RHS_GRAD kernel, and
 * must stay that way: the correction couples three cells along each axis, so its
 * gradient has to be scattered with atomicAdd, whereas the main kernel *assigns*
 * s_block.velocity_grad (valid there only because each cell's block velocity
 * enters nothing but its own RHS). Merging the two would let the assignment race
 * with, and clobber, the scattered contributions.
 *
 * The branches (sign of the face flux, hasFarUpwindCell, and the flat-face skip)
 * are frozen, i.e. treated as constant w.r.t. the solution. That is exact except
 * on the measure-zero sets where they switch. */
template <typename scalar_t, int DIMS>
__global__ void kPISO_build_advection_RHS_deferred_GRAD(DomainGPU<scalar_t> *p_domain,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){

	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
	 //<- necessary for the macro to work with multiple-argument templates...

		// Every target of this kernel is velocity_grad: the three-cell stencil below
		// and the d/dF scatter both end up there (ScatterFluxesGradNDLoop and its
		// helpers atomicAdd into block.velocity_grad without a null check). It is
		// null whenever the velocity is not among the differentiated tensors, so
		// bail out rather than fault. The test is block-uniform (s_block is shared),
		// so skipping here cannot desynchronise the __syncthreads() in the loop.
		if(s_block.velocity_grad==nullptr){ continue; }

		const I4 pos = unflattenIndex(flatPos, s_block);
		const scalar_t det = s_block.hasTransform ? getDeterminant<scalar_t, DIMS>(s_block, pos) : 1;

		// Same fluxes the forward kernel used (block velocity, not velocityResult).
		scalar_t advFluxes[6];
		computeFluxesNDLoop<scalar_t>(pos, advFluxes, s_block, s_domain, nullptr);
		scalar_t fluxesGrad[7] = {0};

		for(index_t dim=0; dim<DIMS; ++dim){
			// The correction sits before the "tempRHS /= det" in the forward kernel.
			const scalar_t RHSgrad = s_domain.velocityRHS_grad[flatPos + s_block.globalOffset + s_domain.numCells*dim] / det;

			for(index_t bound=0; bound<(s_domain.numDims*2); ++bound){
				const index_t axis = axisFromBound(bound);
				const index_t faceSign = faceSignFromBound(bound);
				const scalar_t faceFluxOut = faceSign * advFluxes[bound];
				const bool outflow = faceFluxOut >= 0;
				if(!hasFarUpwindCell(pos, bound, outflow, s_block)){ continue; }

				I4 cellPos = pos;
				cellPos.w = dim;
				const index_t idxP = flattenIndex(cellPos, s_block);
				cellPos.a[axis] = pos.a[axis] + faceSign;
				const index_t idxN = flattenIndex(cellPos, s_block);
				cellPos.a[axis] = outflow ? (pos.a[axis] - faceSign) : (pos.a[axis] + 2*faceSign);
				const index_t idxUU = flattenIndex(cellPos, s_block);

				const scalar_t phiP = s_block.velocity[idxP];
				const scalar_t phiN = s_block.velocity[idxN];
				const scalar_t phiUU = s_block.velocity[idxUU];
				const scalar_t phiU = outflow ? phiP : phiN;
				const scalar_t phiD = outflow ? phiN : phiP;
				const scalar_t dD = phiD - phiU;
				const scalar_t absDD = dD < 0 ? -dD : dD;
				// Mirror the forward flat-face skip exactly: this is the adjoint of the
				// code as written, not of the idealized formula (for LINEAR_UPWIND the
				// term is 0.5*(phi_U - phi_UU) and does not actually vanish at dD == 0).
				if(absDD < static_cast<scalar_t>(1e-20)){ continue; }

				const index_t idxU = outflow ? idxP : idxN;
				const scalar_t half_RHSgrad = static_cast<scalar_t>(0.5) * RHSgrad;
				atomicAdd(s_block.velocity_grad + idxU,  -faceFluxOut * half_RHSgrad);
				atomicAdd(s_block.velocity_grad + idxUU,  faceFluxOut * half_RHSgrad);
				fluxesGrad[bound] -= faceSign * (phiU - phiUU) * half_RHSgrad;
			}
		}

		ScatterFluxesGradNDLoop<scalar_t>(pos, fluxesGrad, s_block, s_domain, nullptr);
	)
}

/* Passive-scalar counterpart of kPISO_build_advection_RHS_deferred_GRAD. */
template <typename scalar_t>
__global__ void kPISO_build_scalar_advection_RHS_deferred_GRAD(DomainGPU<scalar_t> *p_domain,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){

	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,

		const I4 pos = unflattenIndex(flatPos, s_block);
		const scalar_t det = s_block.hasTransform ? getDeterminantDimSwitch(s_block, pos, s_domain.numDims) : 1;

		scalar_t advFluxes[6];
		computeFluxesNDLoop<scalar_t>(pos, advFluxes, s_block, s_domain, nullptr);
		scalar_t fluxesGrad[7] = {0};

		// d/dF flows back into the velocity field, which has no gradient buffer when
		// only the passive scalar is differentiated. ScatterFluxesGradNDLoop would
		// then fault on a null block.velocity_grad.
		const bool scatterFluxGrad = s_block.velocity_grad!=nullptr;

		for(index_t channel=0; channel<s_domain.passiveScalarChannels; ++channel){
			I4 tempPos = pos;
			tempPos.w = channel;
			const scalar_t RHSgrad = s_domain.scalarRHS_grad[flattenIndexGlobal(tempPos, s_block, s_domain)] / det;

			for(index_t bound=0; bound<(s_domain.numDims*2); ++bound){
				const index_t axis = axisFromBound(bound);
				const index_t faceSign = faceSignFromBound(bound);
				const scalar_t faceFluxOut = faceSign * advFluxes[bound];
				const bool outflow = faceFluxOut >= 0;
				if(!hasFarUpwindCell(pos, bound, outflow, s_block)){ continue; }

				I4 cellPos = pos;
				cellPos.w = channel;
				const index_t idxP = flattenIndex(cellPos, s_block);
				cellPos.a[axis] = pos.a[axis] + faceSign;
				const index_t idxN = flattenIndex(cellPos, s_block);
				cellPos.a[axis] = outflow ? (pos.a[axis] - faceSign) : (pos.a[axis] + 2*faceSign);
				const index_t idxUU = flattenIndex(cellPos, s_block);

				const scalar_t phiP = s_block.scalarData[idxP];
				const scalar_t phiN = s_block.scalarData[idxN];
				const scalar_t phiUU = s_block.scalarData[idxUU];
				const scalar_t phiU = outflow ? phiP : phiN;
				const scalar_t phiD = outflow ? phiN : phiP;
				const scalar_t dD = phiD - phiU;
				const scalar_t absDD = dD < 0 ? -dD : dD;
				if(absDD < static_cast<scalar_t>(1e-20)){ continue; }

				const index_t idxU = outflow ? idxP : idxN;
				const scalar_t half_RHSgrad = static_cast<scalar_t>(0.5) * RHSgrad;
				atomicAdd(s_block.scalarData_grad + idxU,  -faceFluxOut * half_RHSgrad);
				atomicAdd(s_block.scalarData_grad + idxUU,  faceFluxOut * half_RHSgrad);
				if(scatterFluxGrad){ fluxesGrad[bound] -= faceSign * (phiU - phiUU) * half_RHSgrad; }
			}
		}

		if(scatterFluxGrad){
			ScatterFluxesGradNDLoop<scalar_t>(pos, fluxesGrad, s_block, s_domain, nullptr);
		}
	)
}

#endif //WITH_GRAD



template <typename scalar_t>
void _SetupAdvectionMatrixEulerImplicit(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU,
		const int8_t nonOrthoFlags, const bool forPassiveScalar, const index_t passiveScalarChannel){
	// setup domain
	int32_t threads;
	dim3 blocks;
	std::vector<index_t> blockIdxByThreadBlock;
	std::vector<index_t> threadBlockOffsetInBlock;
	ComputeThreadBlocks(domain, threads, blocks, blockIdxByThreadBlock, threadBlockOffsetInBlock);
	index_t *p_blockIdxByThreadBlock;
	index_t *p_threadBlockOffsetInBlock;
	torch::Tensor t_blockIdxByThreadBlock = CopyBlockIndices(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock); //keep the torch::Tensor to deallocate at the end

	// make A matrix
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	//LOG("Dispatch A matrix kernel");
	BEGIN_SAMPLE;
 	PISO_build_matrix<scalar_t><<<blocks, threads>>>(
		reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			nonOrthoFlags, forPassiveScalar, passiveScalarChannel
	);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("Build A-Matrix");
	
}

void SetupAdvectionMatrixEulerImplicit(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
		const int8_t nonOrthoFlags, const bool forPassiveScalar, const index_t passiveScalarChannel){
	
	if(forPassiveScalar){
		TORCH_CHECK(domain->hasPassiveScalar(), "domain does not have passive scalar.");
		TORCH_CHECK(0<=passiveScalarChannel && passiveScalarChannel<domain->getPassiveScalarChannels(), "Passive scalar channel index out of bounds.");
	}
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	domain->setTimeStep(timeStep); // [1] or one per batched environment
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "SetupAdvectionMatrixEulerImplicit", ([&] {
		_SetupAdvectionMatrixEulerImplicit<scalar_t>(
			domain, timeStep, nonOrthoFlags, forPassiveScalar, passiveScalarChannel
		);
	}));
}

#ifdef WITH_GRAD

template <typename scalar_t>
void _SetupAdvectionMatrixEulerImplicit_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU,
		const int8_t nonOrthoFlags, const bool forPassiveScalar, const index_t passiveScalarChannel){
	// setup domain
	int32_t threads;
	dim3 blocks;
	std::vector<index_t> blockIdxByThreadBlock;
	std::vector<index_t> threadBlockOffsetInBlock;
	ComputeThreadBlocks(domain, threads, blocks, blockIdxByThreadBlock, threadBlockOffsetInBlock);
	index_t *p_blockIdxByThreadBlock;
	index_t *p_threadBlockOffsetInBlock;
	torch::Tensor t_blockIdxByThreadBlock = CopyBlockIndices(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock); //keep the torch::Tensor to deallocate at the end

	// make A matrix
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	//LOG("Dispatch A matrix kernel");
	BEGIN_SAMPLE;
 	PISO_build_matrix_GRAD<scalar_t><<<blocks, threads>>>(
		reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
		p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
		nonOrthoFlags, forPassiveScalar, passiveScalarChannel
	);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("Build A-Matrix");
	
}

void SetupAdvectionMatrixEulerImplicit_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
		const int8_t nonOrthoFlags, const bool forPassiveScalar, const index_t passiveScalarChannel){
	
	if(forPassiveScalar){
		TORCH_CHECK(domain->hasPassiveScalar(), "domain does not have passive scalar.");
		TORCH_CHECK(0<=passiveScalarChannel && passiveScalarChannel<domain->getPassiveScalarChannels(), "Passive scalar channel index out of bounds.");
	}
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	domain->setTimeStep(timeStep); // [1] or one per batched environment
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "SetupAdvectionMatrixEulerImplicit", ([&] {
		_SetupAdvectionMatrixEulerImplicit_GRAD<scalar_t>(
			domain, timeStep, nonOrthoFlags, forPassiveScalar, passiveScalarChannel
		);
	}));
}

#endif //WITH_GRAD

template <typename scalar_t>
void _SetupAdvectionScalarEulerImplicitRHS(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU, const int8_t nonOrthoFlags){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
 	kPISO_build_scalar_advection_RHS<scalar_t><<<blocks, threads>>>(
		reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
		p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
		nonOrthoFlags
	);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("scalar Advection RHS");
	
}
void SetupAdvectionScalarEulerImplicitRHS(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags){
	
	TORCH_CHECK(domain->hasPassiveScalar(), "domain does not have passive scalar.");
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	domain->setTimeStep(timeStep); // [1] or one per batched environment
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "SetupAdvectionScalarEulerImplicitRHS", ([&] {
		_SetupAdvectionScalarEulerImplicitRHS<scalar_t>(
			domain, timeStep, nonOrthoFlags
		);
	}));
}

#ifdef WITH_GRAD
template <typename scalar_t>
void _SetupAdvectionScalarEulerImplicitRHS_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU, const int8_t nonOrthoFlags){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
 	kPISO_build_scalar_advection_RHS_GRAD<scalar_t><<<blocks, threads>>>(
		reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
		p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
		nonOrthoFlags
	);
	// See the velocity version: separate launch so the atomicAdd scatter cannot
	// race with the assignment to scalarData_grad above.
	if(domain->getAdvectionScheme()==AdvectionScheme::LINEAR_UPWIND){
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		kPISO_build_scalar_advection_RHS_deferred_GRAD<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
		);
	}
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("scalar Advection RHS");
	
}
void SetupAdvectionScalarEulerImplicitRHS_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags){
	
	TORCH_CHECK(domain->hasPassiveScalar(), "domain does not have passive scalar.");
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	domain->setTimeStep(timeStep); // [1] or one per batched environment
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "SetupAdvectionScalarEulerImplicitRHS", ([&] {
		_SetupAdvectionScalarEulerImplicitRHS_GRAD<scalar_t>(
			domain, timeStep, nonOrthoFlags
		);
	}));
}
#endif //WITH_GRAD

template <typename scalar_t, int DIMS>
void _SetupAdvectionVelocityEulerImplicitRHS(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU, const int8_t nonOrthoFlags, const bool applyPressureGradient){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
 	kPISO_build_advection_RHS<scalar_t, DIMS><<<blocks, threads>>>(
		reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
		p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
		nonOrthoFlags, applyPressureGradient
	);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("Velocity Advection RHS");
	
}
void SetupAdvectionVelocityEulerImplicitRHS(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags, const bool applyPressureGradient){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	domain->setTimeStep(timeStep); // [1] or one per batched environment
	
	DISPATCH_FTYPES_DIMS(domain, "SetupAdvectionVelocityEulerImplicitRHS",
		_SetupAdvectionVelocityEulerImplicitRHS<scalar_t, dim>(
			domain, timeStep, nonOrthoFlags, applyPressureGradient
		)
	)
}

#ifdef WITH_GRAD

template <typename scalar_t, int DIMS>
void _SetupAdvectionVelocityEulerImplicitRHS_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU,
		const int8_t nonOrthoFlags, const bool applyPressureGradient){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
 	kPISO_build_advection_RHS_GRAD<scalar_t, DIMS><<<blocks, threads>>>(
		reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
		p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
		nonOrthoFlags, applyPressureGradient
	);
	// Deferred high-order correction, as a second launch so its atomicAdd scatter
	// cannot race with the assignment to velocity_grad above. CENTRAL has no
	// correction term, so it needs no adjoint here.
	if(domain->getAdvectionScheme()==AdvectionScheme::LINEAR_UPWIND){
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		kPISO_build_advection_RHS_deferred_GRAD<scalar_t, DIMS><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
		);
	}
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Velocity Advection RHS grad");
	
}
void SetupAdvectionVelocityEulerImplicitRHS_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
		const int8_t nonOrthoFlags, const bool applyPressureGradient){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	domain->setTimeStep(timeStep); // [1] or one per batched environment
	DISPATCH_FTYPES_DIMS(domain, "SetupAdvectionVelocityEulerImplicitRHS_GRAD",
		_SetupAdvectionVelocityEulerImplicitRHS_GRAD<scalar_t, dim>(
			domain, timeStep, nonOrthoFlags, applyPressureGradient
		)
	)
}

#endif //WITH_GRAD

template <typename scalar_t, int DIMS>
void _SetupAdvectionEulerImplicitCombined(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU, const int8_t nonOrthoFlags, const bool applyPressureGradient){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	// make A matrix
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	LOG("Dispatch A matrix kernel");
	BEGIN_SAMPLE;
 	PISO_build_matrix<scalar_t><<<blocks, threads>>>(
		reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
		p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
		nonOrthoFlags, false, 0
	);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("Build A-Matrix");
	
	// make velocity RHS
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
 	kPISO_build_advection_RHS<scalar_t, DIMS><<<blocks, threads>>>(
		reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
		p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
		nonOrthoFlags, applyPressureGradient
	);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("Velocity Advection RHS");
	
	// make passive scalar RHS
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
 	kPISO_build_scalar_advection_RHS<scalar_t><<<blocks, threads>>>(
		reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
		p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
		nonOrthoFlags
	);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("scalar Advection RHS");
	
}
void SetupAdvectionEulerImplicitCombined(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags, const bool applyPressureGradient){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	domain->setTimeStep(timeStep); // [1] or one per batched environment
	
	DISPATCH_FTYPES_DIMS(domain, "SetupAdvectionEulerImplicitCombined",
		_SetupAdvectionEulerImplicitCombined<scalar_t, dim>(
			domain, timeStep, nonOrthoFlags, applyPressureGradient 
		)
	)
}

