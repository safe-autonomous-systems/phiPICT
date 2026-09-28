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
// Pressure correction: matrix, RHS and divergence, and their gradients.

#include "piso/device/common.cuh"



/* --- PRESSURE SOLVE --- */

template <typename scalar_t> //, index_t DIMS>
__global__ void PISO_build_pressure_matrix(DomainGPU<scalar_t> *p_domain, //const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks,
		const int8_t nonOrthoFlags, const bool useFaceTransform){
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		const index_t flatPosGlobal = s_block.globalOffset + flattenIndex(pos, s_block); //flatPos;
		
		const RowMeta row = getCSRMatrixRowEndOffsetFromBlockBoundaries3D(flatPos, s_block, s_domain);
		const index_t rowStartOffset = row.endOffset - row.size + s_block.csrOffset;
		
		s_domain.P.row[flatPosGlobal+1] = row.endOffset + s_block.csrOffset;
		//s_domain.P.row[s_block.globalOffset + threadIdx.x] = repetitions;
		
		// alternative: compute flat indices, sort
		index_t indices[7]; // diag,-x,+x,-y,+y,-z,+z
		scalar_t rowValues[7] = {0};
		
		// orthogonal transform coefficients
		scalar_t alphaP[3];
		for(index_t dim=0;dim<s_domain.numDims;++dim){
			I4 tempPos = pos;
			tempPos.w = dim;
			alphaP[dim] = getLaplaceCoefficientOrthogonalDimSwitch(tempPos, &s_block, s_domain.numDims);
		}
		const scalar_t raP = static_cast<scalar_t>(1.0) /s_domain.Adiag[flatPosGlobal];
		indices[0] = flatPosGlobal;
		
		for(index_t bound=0; bound<(s_domain.numDims*2); ++bound){
			const index_t dim = axisFromBound(bound);
			const index_t isUpper = boundIsUpper(bound);
			const index_t faceSign = faceSignFromBound(bound);
			const bool atBound = isAtBound(pos, bound, &s_block);
			const bool atPrescribedBound = atBound && isEmptyBound(bound, s_block.boundaries);
			
			if(!atPrescribedBound){
				// laplace difference to diffusion: sign, includes 1/A, does not include domain.viscosity
				{ // laplace orthogonal coefficients from face normal directions
					I4 tempPos = pos;
					tempPos.w = dim;
					const BlockGPU<scalar_t> *p_block = &s_block;
					
					if(atBound && s_block.boundaries[bound].type==BoundaryType::CONNECTED_GRID){
						p_block = s_domain.blocks + s_block.boundaries[bound].cb.connectedGridIndex;
						tempPos = computeConnectedPosWithChannel(tempPos, dim, &s_block.boundaries[bound].cb, s_domain);
					}else {
						if(atBound && s_block.boundaries[bound].type==BoundaryType::PERIODIC){
							tempPos.a[dim] = isUpper ? 0 : s_block.size.a[dim]-1;
						}else{
							tempPos.a[dim] = pos.a[dim] + faceSign;
						}
					}
					
					const scalar_t alphaN = getLaplaceCoefficientOrthogonalDimSwitch(tempPos, p_block, s_domain.numDims);
					
					tempPos.w = 0;
					const index_t tempFlatPosGlobal = flattenIndex(tempPos, p_block) + p_block->globalOffset;
					const scalar_t raN = static_cast<scalar_t>(1.0) / s_domain.Adiag[tempFlatPosGlobal];
					
					scalar_t coefficient = 0;
					if(useFaceTransform){
						tempPos = pos;
						p_block = &s_block;
						//const scalar_t alphaFace = getLaplaceCoefficientOrthogonalFaceDimSwitch(pos, bound, &s_block, s_domain.numDims); // <- ISSUE here?
						const scalar_t alphaFace = getLaplaceCoefficientOrthogonalFaceDimSwitch(tempPos, bound, p_block, s_domain.numDims);
						// TODO: (1/aP + 1/aN)/2 vs. 1/((aP + aN)/2) ? 
						const scalar_t raFace = static_cast<scalar_t>(0.5) * (raP + raN);
						coefficient = alphaFace * raFace;
					}else{
						coefficient = static_cast<scalar_t>(0.5) * (alphaP[dim]*raP + alphaN*raN);
					}
					
					rowValues[0] -= coefficient;
					rowValues[bound+1] += coefficient;
					indices[bound+1] = tempFlatPosGlobal;
				}
				
				// laplace non-orthogonal coefficients from face tangential directions
				const bool includeNonOrthoNeighbors = nonOrthoFlags & NON_ORTHO_DIRECT_MATRIX;
				const bool includeNonOrthoDiag = nonOrthoFlags & NON_ORTHO_CENTER_MATRIX;
				if(s_domain.numDims>1 && (includeNonOrthoDiag || includeNonOrthoNeighbors)){
					// non-orthogonal transform coefficients
					scalar_t alphaInterp[12]; // size for 3D
					if(s_domain.numDims==2) interpolateNonOrthoLaplaceComponents<scalar_t, 2>(pos, s_block, s_domain, alphaInterp, false, true, useFaceTransform);
					else if(s_domain.numDims==3) interpolateNonOrthoLaplaceComponents<scalar_t, 3>(pos, s_block, s_domain, alphaInterp, false, true, useFaceTransform);
					
					for(index_t i=1; i<s_domain.numDims; ++i){ // loop other axes
						const index_t tAxis = (dim + i)%s_domain.numDims;
						const scalar_t alpha = getInterpolatedNonOrthoLaplaceComponent(alphaInterp, bound, tAxis, s_domain.numDims);
						
						if(alpha!=0){ // grid is non-orthogonal here
							for(index_t tIsUpper=0; tIsUpper<2; ++tIsUpper){ // loop corners
								const index_t tFace = axisToBound(tAxis, tIsUpper); //(tAxis<<1) + tIsUpper;
								const index_t tFaceSign = faceSignFromBound(tFace);
								const CornerValue<scalar_t> cVal = getCornerValue<scalar_t>(pos, bound, tFace,
									false, false, 2, s_block, s_domain, GridDataType::IS_FIXED_BOUNDARY); // does not read data. computes interpolation divisor and checks for boundaries.
								const bool cornerAtBound = cVal.numCells<1;
								const bool boundIsGradient = true; //gridDataTypeToBaseType(type)==GridDataType::PRESSURE; // TODO: get from bound with FIXED boundary implementation
								
								if(cornerAtBound){
									if(boundIsGradient){
										// simplified treatment: ignore boudnary gradient and use one-sided difference from other side
										const scalar_t interpolationNorm = 0.25; // from other side, can't be anything else but 1/4
										const index_t tFaceOther = invertBound(tFace);
										const scalar_t coefficient = faceSign * tFaceSign * alpha * interpolationNorm;
										if(includeNonOrthoDiag){
											rowValues[0] += 3 * coefficient;
										}
										if(includeNonOrthoNeighbors){
											rowValues[bound+1] += 3 * coefficient;
											rowValues[tFaceOther+1] -= coefficient;
										}
										// if diagonals where to be included in the matrix: 
										// rowValues[diagonalOther] += coefficient;
									}
									// else: prescribed value, added on RHS, nothing to do here.
								} else {
									// normal corner with interpolated value
									const scalar_t interpolationNorm = 1.0 / static_cast<scalar_t>(cVal.numCells);
									const scalar_t coefficient = faceSign * tFaceSign * alpha * interpolationNorm;
									if(includeNonOrthoDiag){
										rowValues[0] += coefficient;
									}
									if(includeNonOrthoNeighbors){
										rowValues[bound+1] += coefficient;
										rowValues[tFace+1] += coefficient;
									}
									// if diagonals where to be included in the matrix: 
									// rowValues[diagonal] -= coefficient;
								}
							}
						}
					}
				}
			} else { // prescribedBound
				// TODO: some non-orthogonal handling?
				indices[bound+1] = -1; //invalid/unused
			}
		}
		
		
		for(int dim=s_domain.numDims;dim<3;++dim){
			indices[dim*2+1] = -1; //invalid/unused
			indices[dim*2+2] = -1; //invalid/unused
		}
		
		
		// sort, naive for now
		for(index_t i=0;i<row.size;++i){
			index_t colIndex = findLowestColumnIndex(indices, 7);
			
			/* if((rowStartOffset + i)>=127){
				
			}else  */if(colIndex<0){
				s_domain.P.index[rowStartOffset + i] = -1;
				s_domain.P.value[rowStartOffset + i] = -1.0f;
			}else{
				s_domain.P.index[rowStartOffset + i] = indices[colIndex];
				s_domain.P.value[rowStartOffset + i] = rowValues[colIndex];
			}
			
			indices[colIndex] = -1;
		}
	)
}

#ifdef WITH_GRAD

template <typename scalar_t> //, index_t DIMS>
__global__ void PISO_build_pressure_matrix_GRAD(DomainGPU<scalar_t> *p_domain, //const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks,
		const int8_t nonOrthoFlags, const bool useFaceTransform){
	
	const scalar_t half = static_cast<scalar_t>(0.5);
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		const index_t flatPosGlobal = flattenIndexGlobal(pos, s_block, s_domain);
		
		//const scalar_t det = s_block.hasTransform ? getDeterminantDimSwitch(s_block, pos, s_domain.numDims) : 1;
		//const scalar_t rDet = 1/det;
		
		// load row from pressure CSR matrix grad
		scalar_t csrValuesGrad[7] = {0}; // to be loaded like: diag,-x,+x,-y,+y,-z,+z
		LoadCSRrowNeighborSorted(pos, s_domain.P_grad, s_domain, s_block, csrValuesGrad);
		
		// orthogonal transform coefficients
		scalar_t alphaP[3];
		for(index_t dim=0;dim<s_domain.numDims;++dim){
			I4 tempPos = pos;
			tempPos.w = dim;
			alphaP[dim] = getLaplaceCoefficientOrthogonalDimSwitch(tempPos, &s_block, s_domain.numDims);
		}
		scalar_t raP_grad = 0;
		
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
					scalar_t alphaN = 1;
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
					alphaN = getLaplaceCoefficientOrthogonalDimSwitch(tempPos, p_block, s_domain.numDims);
					
					tempPos.w = 0;
					const index_t tempFlatPosGlobal = flattenIndex(tempPos, p_block) + p_block->globalOffset;
					const scalar_t raN = static_cast<scalar_t>(1.0) / s_domain.Adiag[tempFlatPosGlobal];
					
					const scalar_t coefficient_grad = csrValuesGrad[bound+1] - csrValuesGrad[0];
					raP_grad += half * alphaP[dim] * coefficient_grad;
					const scalar_t raN_grad = half * alphaN * coefficient_grad;
					
					tempPos.w = 0;
					atomicAdd(s_domain.Adiag_grad + tempFlatPosGlobal, -raN_grad*raN*raN);
				
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
					
					for(index_t i=1; i<s_domain.numDims; ++i){ // loop other axes
						const index_t tAxis = (dim + i)%s_domain.numDims;
						const scalar_t alpha = getInterpolatedNonOrthoLaplaceComponent(alphaInterp, bound, tAxis, s_domain.numDims);
						
						if(alpha!=0){ // grid is non-orthogonal here
							scalar_t alphaRa_grad = 0;
							for(index_t tIsUpper=0; tIsUpper<2; ++tIsUpper){ // loop corners
								const index_t tFace = axisToBound(tAxis, tIsUpper); //(tAxis<<1) + tIsUpper;
								const index_t tFaceSign = faceSignFromBound(tFace);
								const CornerValue<scalar_t> cVal = getCornerValue<scalar_t>(channelPos, bound, tFace,
									false, false, 2, s_block, s_domain, GridDataType::IS_FIXED_BOUNDARY); // computes interpolation divisor and checks for boundaries.
								const bool cornerAtBound = cVal.numCells<1;
								const bool boundIsGradient = true; //cVal.boundType==BoundaryConditionType::NEUMANN;
								
								if(cornerAtBound){
									if(boundIsGradient){
										// simplified treatment: ignore boudnary gradient and use one-sided difference from other side
										const scalar_t interpolationNorm = 0.25; // from other side, can't be anything else but 1/4
										const index_t tFaceOther = invertBound(tFace);
										
										scalar_t coefficient_grad = 0;
										if(includeNonOrthoDiag){
											coefficient_grad += 3 * csrValuesGrad[0];
										}
										if(includeNonOrthoNeighbors){
											coefficient_grad += 3 * csrValuesGrad[bound+1];
											coefficient_grad -= csrValuesGrad[tFaceOther+1];
										}
										
										alphaRa_grad += faceSign * tFaceSign * interpolationNorm * coefficient_grad;
									}
									// else: prescribed value, added on RHS, nothing to do here.
								} else {
									// normal corner with interpolated value
									const scalar_t interpolationNorm = 1.0 / static_cast<scalar_t>(cVal.numCells);
									//const scalar_t viscosityCoeff = faceSign * tFaceSign * viscosity * alpha * interpolationNorm;
									scalar_t coefficient_grad = 0;
									if(includeNonOrthoDiag){
										coefficient_grad += csrValuesGrad[0];
									}
									if(includeNonOrthoNeighbors){
										coefficient_grad += csrValuesGrad[bound+1];
										coefficient_grad += csrValuesGrad[tFace+1];
									}
									// if diagonals where to be included in the matrix: 
									// rowValues[diagonal] -= viscosityCoeff;
									alphaRa_grad += faceSign * tFaceSign * interpolationNorm * coefficient_grad;
								}
							}
							{
								//ra_grad += alphaRa_grad * alpha;
								addInterpolatedNonOrthoLaplaceComponent_GRAD(alphaRa_grad, alphaInterp_grad, bound, tAxis, s_domain.numDims);
							}
						}
					}
					
					if(s_domain.numDims==2) scatterNonOrthoLaplaceComponents_GRAD<scalar_t, 2>(alphaInterp_grad, pos, s_block, s_domain, false, true, false);
					else if(s_domain.numDims==3) scatterNonOrthoLaplaceComponents_GRAD<scalar_t, 3>(alphaInterp_grad, pos, s_block, s_domain, false, true, false);
					
				}
				
			}
			
		}
		const scalar_t raP = static_cast<scalar_t>(1.0) /s_domain.Adiag[flatPosGlobal];
		atomicAdd(s_domain.Adiag_grad + flatPosGlobal, -raP_grad*raP*raP);
	); //KERNEL_PER_CELL_LOOP
}

#endif //WITH_GRAD

#define PRESSURE_RHS_WITH_BOUNDARY_SOURCES

template <typename scalar_t>
__global__ void PISO_build_pressure_rhs(DomainGPU<scalar_t> *p_domain,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){
	
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		const scalar_t timeStep = s_domain.timeStep; // per batched environment
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		const index_t flatPosGlobal = s_block.globalOffset + flatPos;
		const scalar_t rDiag = 1 / s_domain.Adiag[flatPosGlobal];
		
		const scalar_t det = s_block.hasTransform ? getDeterminantDimSwitch(s_block, pos, s_domain.numDims) : 1;
		
		// load row from advection CSR matrix
		index_t csrIndices[7];
		scalar_t csrValues[7];
		const index_t csrStart = s_domain.C.row[flatPosGlobal];
		const index_t csrEnd = s_domain.C.row[flatPosGlobal+1];
		const index_t rowSize = csrEnd - csrStart;
		for(index_t i=0;i<rowSize && i<7;++i){
			csrIndices[i] = s_domain.C.index[csrStart+i];
			csrValues[i] = s_domain.C.value[csrStart+i];
		}

		// Face fluxes for the deferred high-order convection correction. Same
		// fluxes kPISO_build_advection_RHS used, so both agree on the upwind
		// direction and on which faces the matrix kept central.
		const bool deferredAdvection = s_domain.advectionScheme != AdvectionScheme::CENTRAL;
		scalar_t advFluxes[6];
		if(deferredAdvection){
			computeFluxesNDLoop<scalar_t>(pos, advFluxes, s_block, s_domain, nullptr);
		}

		for(index_t dim=0;dim<s_domain.numDims;++dim){ // loop velocity components
			I4 compPos = pos;
			compPos.w = dim;
			const index_t flatCompPos = flattenIndex(compPos, s_block);

			const scalar_t velOld = s_block.velocity[flatCompPos]/timeStep; //det *
			
			scalar_t H = 0;
			for(index_t i=0;i<rowSize;++i){
				index_t idx = csrIndices[i]; // these are global, but for scalars
				if(idx!=flatPosGlobal){ //omit diagonal entries
					const scalar_t vel = s_domain.velocityResult[idx + s_domain.numCells*dim];
					H += csrValues[i] * vel; //H'u*
				}
			}
			
			// source terms
			scalar_t S = 0;
			
			// - non-orthogonal transformation - AFTER divergence!
			
#ifdef PRESSURE_RHS_WITH_BOUNDARY_SOURCES
			// - boundary source terms. TODO: is this correct here? yes
			for(index_t bound=0; bound<(s_domain.numDims*2); ++bound){ 
				if(((bound&1)==0 && pos.a[bound>>1]==0) || ((bound&1)==1 && pos.a[bound>>1]==s_block.size.a[bound>>1]-1)){
					const scalar_t faceNormal = static_cast<scalar_t>((bound&1)*2 -1); // [0,1] -> [-1,1]
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
						S -= vel * flux;
						// from viscosity
						const scalar_t boundViscosity = getViscosityFixedBoundary(pos, p_fb, &s_block, s_domain, false, 0);
						// TODO: simple slip from boundary type
						const scalar_t slip = s_block.boundaries[bound].fb.velocity.boundaryType==BoundaryConditionType::DIRICHLET ? 0 : 1;
						S += vel * (1-slip) * boundViscosity * 2 * alpha; //* isTangentialDir
						break;
					}
					case BoundaryType::VALUE:
					{
						// from advection
						S -= s_block.boundaries[bound].sdb.velocity.a[dim] * s_block.boundaries[bound].sdb.velocity.a[bound>>1] * faceNormal;
						// from viscosity
						const scalar_t viscosity = getViscosityBlock(pos, &s_block, s_domain, false, 0);
						S += viscosity * 2 * s_block.boundaries[bound].sdb.velocity.a[dim]; //* isTangentialDir
						break;
					}
					case BoundaryType::DIRICHLET_VARYING:
					{
						I4 tempPos = pos;
						tempPos.w = dim;
						tempPos.a[bound>>1] = 0;
						//const index_t flatTempPos = flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride);
						const scalar_t vel = s_block.boundaries[bound].vdb.velocity[flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride)];
						//tempPos.w = 0;
						tempPos.w = bound>>1;
						const scalar_t alpha = getLaplaceCoefficientOrthogonalBoundaryVaryingDimSwitch(tempPos, &s_block.boundaries[bound].vdb, s_domain.numDims);
						//const scalar_t flux = s_block.boundaries[bound].vdb.velocity[flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride)]*faceNormal;
						const scalar_t flux = getContravariantComponentBoundaryVaryingDimSwitch(tempPos, &s_block.boundaries[bound].vdb, s_domain) * faceNormal;
						
						// from advection
						S -= vel * flux;
						// from viscosity
						const scalar_t viscosity = getViscosityBlock(pos, &s_block, s_domain, false, 0);
						S += vel * (1-s_block.boundaries[bound].vdb.slip) * viscosity * 2 * alpha; //* isTangentialDir
						break;
					}
					default:
						break;
					}
				}
			}
			
			// TODO: add non-ortho boundary values from advection?

			S /= det;
#endif
			S += getBlockVelocitySource(compPos, &s_block);

			if(deferredAdvection){
				// hbyA must reproduce the momentum predictor's source exactly. The
				// predictor put -F*(phi_f^HO - phi_f^UD) on its RHS
				// (kPISO_build_advection_RHS); repeat it here from the same
				// start-of-step block velocity.
				//
				// Without this term the identity a_P*u*_P + H = velOld + S - corr
				// makes rDiag*(velOld - H + S) equal u* + rDiag*corr, so the
				// corrector adds back precisely the correction the predictor
				// subtracted, and the scheme collapses to the first-order upwind
				// carried by the matrix.
				scalar_t deferredCorr = 0;
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
					// Flat face: phi_f = phi_U for any psi, and r stays finite.
					if(absDD < static_cast<scalar_t>(1e-20)){ continue; }

					// LINEAR_UPWIND: psi = r
					const scalar_t psi = (phiU - phiUU) / dD;
					deferredCorr += faceFluxOut * static_cast<scalar_t>(0.5) * psi * dD;
				}
				// The advection terms carry a 1/det, as in the predictor RHS.
				S -= deferredCorr / det;
			}

			s_domain.pressureRHS[flatPosGlobal + s_domain.numCells*dim] = rDiag *(velOld-H + S);
			
		}
	)
}

#ifdef WITH_GRAD
template <typename scalar_t>
__global__ void PISO_build_pressure_rhs_GRAD(DomainGPU<scalar_t> *p_domain,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){
	
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		const scalar_t timeStep = s_domain.timeStep; // per batched environment
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		const index_t flatPosGlobal = s_block.globalOffset + flatPos;
		const scalar_t rDiag = 1 / s_domain.Adiag[flatPosGlobal];
		scalar_t rDiag_grad = 0;
		
#ifdef PRESSURE_RHS_WITH_BOUNDARY_SOURCES
		const scalar_t det = s_block.hasTransform ? getDeterminantDimSwitch(s_block, pos, s_domain.numDims) : 1;
#endif
		
		// load row from advection CSR matrix
		index_t csrIndices[7];
		scalar_t csrValues[7];
		const index_t csrStart = s_domain.C.row[flatPosGlobal];
		const index_t csrEnd = s_domain.C.row[flatPosGlobal+1];
		const index_t rowSize = csrEnd - csrStart;
		for(index_t i=0;i<rowSize && i<7;++i){
			csrIndices[i] = s_domain.C.index[csrStart+i];
			csrValues[i] = s_domain.C.value[csrStart+i];
		}
		scalar_t csrValues_grad[7] = {0};
		
		for(index_t dim=0;dim<s_domain.numDims;++dim){
			I4 compPos = pos;
			compPos.w = dim;
			const index_t flatCompPos = flattenIndex(compPos, s_block);
			
			const scalar_t pGrad = s_domain.pressureRHS_grad[flatPosGlobal + s_domain.numCells*dim];
			
			rDiag_grad += s_domain.pressureRHS[flatPosGlobal + s_domain.numCells*dim] / rDiag * pGrad;
			
			s_block.velocity_grad[flatCompPos] = rDiag/timeStep * pGrad;
			
			//scalar_t H = 0;
			for(index_t i=0;i<rowSize;++i){
				index_t idx = csrIndices[i]; // these are global, but for scalars
				if(idx!=flatPosGlobal){ //omit diagonal entries
					// w.r.t. velocityResult
					atomicAdd(s_domain.velocityResult_grad + (idx + s_domain.numCells*dim), -1*csrValues[i]*rDiag*pGrad);
					// w.r.t. C.value
					const scalar_t vel = s_domain.velocityResult[idx + s_domain.numCells*dim];
					csrValues_grad[i] -= 1*vel*rDiag*pGrad;
				}
			}
			
			// source terms
			scalar_t S_grad = rDiag * pGrad;

			scatterBlockVelocitySource_GRAD(S_grad, compPos, &s_block);

#ifdef PRESSURE_RHS_WITH_BOUNDARY_SOURCES

			S_grad /= det;
			
			//scalar_t viscosity_grad = 0;

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
					flux_grad -= vel * S_grad;
					
					scalar_t vel_grad = 0;
					// from advection
					vel_grad -= flux * S_grad;
					// from viscosity
					const scalar_t boundViscosity = getViscosityFixedBoundary(pos, p_fb, &s_block, s_domain, false, 0);
					vel_grad += (1-slip) * boundViscosity * 2 * alpha * S_grad;
					
					const scalar_t boundViscosity_grad = vel * (1-slip) * 2 * alpha * S_grad;
					
					// scatter flux grad
					scatterContravariantComponentBoundaryFixedDimSwitch<scalar_t>(flux_grad*faceNormal, tempPos, &s_block.boundaries[bound].fb, s_domain);
					//scatter vel grad
					tempPos.w = dim;
					scatterFixedBoundaryData<scalar_t>(vel_grad, tempPos, bound, &s_block, s_domain, GridDataType::VELOCITY_GRAD);
					//scatter viscosity grad
					scatterViscosityBoundary_GRAD<scalar_t>(boundViscosity_grad, pos, p_fb, &s_block, s_domain, false, 0);
					
				}
			}
			//scatterViscosity_GRAD(viscosity_grad, s_domain, false, 0);
#endif //PRESSURE_RHS_WITH_BOUNDARY_SOURCES
		}
		
		//TODO: torch autograd logistics
		for(index_t i=0;i<rowSize && i<7;++i){
			s_domain.C_grad.value[csrStart+i] = csrValues_grad[i];
		}
		
		//s_domain.Adiag_grad[flatPosGlobal] = rDiag_grad * (- rDiag * rDiag);
		// needs to be added to not overwrite Adiag_grad from SetupPressureRHSdiv_GRAD. Does not need atomics here.
		s_domain.Adiag_grad[flatPosGlobal] += rDiag_grad * (- rDiag * rDiag);
		
	)
}

/* Adjoint of the deferred high-order correction in PISO_build_pressure_rhs.
 * LINEAR_UPWIND only; CENTRAL builds no correction.
 *
 * PISO_build_pressure_rhs repeats the predictor's deferred correction so that
 * hbyA matches the momentum source it was built from. The forward is
 *     deferredCorr += F * 0.5 * psi * dD;   S -= deferredCorr/det;
 *     pressureRHS   = rDiag * (velOld - H + S)
 * so d(pressureRHS)/d(term) = -rDiag/det. That is the predictor's -1/det times an
 * extra rDiag, which is the *only* difference from
 * kPISO_build_advection_RHS_deferred_GRAD: fold rDiag into RHSgrad and the three
 * scatters below are identical to the predictor's.
 *
 * For LINEAR_UPWIND psi = r = (phi_U - phi_UU)/dD, so the correction collapses to
 * F*0.5*(phi_U - phi_UU), which is linear in the field: dD cancels and there is no
 * 1/dD in the adjoint. The derivatives are constants,
 *     d/d(phi_U) = 0.5*F,  d/d(phi_UU) = -0.5*F,  d/d(phi_D) = 0,
 * plus d/dF = 0.5*(phi_U - phi_UU) routed through ScatterFluxesGradNDLoop.
 *
 * Like its predictor counterpart this must stay a *separate* launch after
 * PISO_build_pressure_rhs_GRAD: the correction couples three cells along each
 * axis so its gradient has to be scattered with atomicAdd, whereas that kernel
 * *assigns* s_block.velocity_grad. Merging the two would let the assignment race
 * with, and clobber, the scattered contributions.
 *
 * rDiag needs no gradient contribution here: PISO_build_pressure_rhs_GRAD derives
 * rDiag_grad from the stored forward pressureRHS, which already contains the
 * correction.
 *
 * The branches (sign of the face flux, hasFarUpwindCell, and the flat-face skip)
 * are frozen, i.e. treated as constant w.r.t. the solution. That is exact except
 * on the measure-zero sets where they switch. */
template <typename scalar_t>
__global__ void PISO_build_pressure_rhs_deferred_GRAD(DomainGPU<scalar_t> *p_domain,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){

	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,

		// Every target of this kernel is velocity_grad: the three-cell stencil below
		// and the d/dF scatter both end up there (ScatterFluxesGradNDLoop and its
		// helpers atomicAdd into block.velocity_grad without a null check). The test
		// is block-uniform (s_block is shared), so skipping here cannot desynchronise
		// the __syncthreads() in the loop.
		if(s_block.velocity_grad==nullptr){ continue; }

		const I4 pos = unflattenIndex(flatPos, s_block);
		const index_t flatPosGlobal = s_block.globalOffset + flatPos;
		const scalar_t rDiag = 1 / s_domain.Adiag[flatPosGlobal];
		const scalar_t det = s_block.hasTransform ? getDeterminantDimSwitch(s_block, pos, s_domain.numDims) : 1;

		// Same fluxes the forward kernel used (block velocity, not velocityResult).
		scalar_t advFluxes[6];
		computeFluxesNDLoop<scalar_t>(pos, advFluxes, s_block, s_domain, nullptr);
		scalar_t fluxesGrad[7] = {0};

		for(index_t dim=0; dim<s_domain.numDims; ++dim){
			// d(loss)/d(deferredCorr term), sign folded in so the scatters below match
			// kPISO_build_advection_RHS_deferred_GRAD line for line.
			const scalar_t RHSgrad = s_domain.pressureRHS_grad[flatPosGlobal + s_domain.numCells*dim] * rDiag / det;

			for(index_t bound=0; bound<(s_domain.numDims*2); ++bound){
				const index_t axis = axisFromBound(bound);
				const index_t faceSign = faceSignFromBound(bound);
				const scalar_t faceFluxOut = faceSign * advFluxes[bound];
				const bool outflow = faceFluxOut >= 0;
				// Faces the matrix kept central need no correction.
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

#endif //WITH_GRAD


template <typename scalar_t>
__global__ void k_computePressureRHSdivergenceFromFlux(DomainGPU<scalar_t> *p_domain,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks,
		const bool useFaceTransform, const bool timeStepNorm){
	
	// divergence of colocated vector field
	// using central differences
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		const scalar_t timeStep = s_domain.timeStep; // per batched environment
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		
		scalar_t fluxes[6]; //DIMS*2
		if(useFaceTransform){
			switch(s_domain.numDims){
				case 1:
					computeFluxesWithFaceTransforms<scalar_t, 1>(pos, fluxes, s_block, s_domain, s_domain.pressureRHS);
					break;
				case 2:
					computeFluxesWithFaceTransforms<scalar_t, 2>(pos, fluxes, s_block, s_domain, s_domain.pressureRHS);
					break;
				case 3:
					computeFluxesWithFaceTransforms<scalar_t, 3>(pos, fluxes, s_block, s_domain, s_domain.pressureRHS);
					break;
				default:
					break;
			}
		}else{
			computeFluxesNDLoop(pos, fluxes, s_block, s_domain, s_domain.pressureRHS);
		}
			
		scalar_t div = 0;
		for(index_t dim=0;dim<s_domain.numDims;++dim){
			div += fluxes[dim*2+1] - fluxes[dim*2];
		}
		
		
		//const scalar_t det = getDeterminantDimSwitch(s_block, pos, s_domain.numDims);
		
		if(timeStepNorm){
			div /= timeStep;
		}
		
		s_domain.pressureRHSdiv[flatPos + s_block.globalOffset] = div;
	)
}

#ifdef WITH_GRAD

template <typename scalar_t>
__global__ void k_computePressureRHSdivergenceFromFlux_GRAD(DomainGPU<scalar_t> *p_domain,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks,
		const bool timeStepNorm){
	
	// divergence of colocated vector field
	// using central differences
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		const scalar_t timeStep = s_domain.timeStep; // per batched environment
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		index_t flatPosGlobal = flattenIndexGlobal(pos, s_block, s_domain);
		
		scalar_t divGrad = s_domain.pressureRHSdiv_grad[flatPosGlobal];
		
		if(timeStepNorm){
			divGrad /= timeStep;
		}
		
		scalar_t fluxesGrad[6]; //DIMS*2
			
		for(index_t dim=0;dim<s_domain.numDims;++dim){
			fluxesGrad[dim*2+1] = divGrad;
			fluxesGrad[dim*2] = -divGrad;
		}
		
		ScatterFluxesGradNDLoop(pos, fluxesGrad, s_block, s_domain, s_domain.pressureRHS_grad);
		
	)
}
#endif //WITH_GRAD

template <typename scalar_t>
__global__ void k_pressureRHSaddNonOrthoComponents(DomainGPU<scalar_t> *p_domain, //const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks,
		const int8_t nonOrthoFlags, const bool useFaceTransform){
	
	// divergence of colocated vector field
	// using central differences
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		
		scalar_t S = s_domain.pressureRHSdiv[flatPos + s_block.globalOffset];
		
		// - non-orthogonal transformation
		if((nonOrthoFlags & NON_ORTHO_DIRECT_RHS) | (nonOrthoFlags & NON_ORTHO_DIAGONAL_RHS)){
			S += getNonOrthoLaplaceRHSDimSwitch_v2<scalar_t>(pos, s_block, s_domain, nonOrthoFlags, GridDataType::PRESSURE_RESULT, false, true, useFaceTransform);
		}
		
		
		s_domain.pressureRHSdiv[flatPos + s_block.globalOffset] = S;
	)
}
#ifdef WITH_GRAD
template <typename scalar_t>
__global__ void k_pressureRHSaddNonOrthoComponents_GRAD(DomainGPU<scalar_t> *p_domain, //const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks,
		const int8_t nonOrthoFlags, const bool useFaceTransform){
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		const I4 pos = unflattenIndex(flatPos, s_block);
		
		const scalar_t RHSdiv_grad = s_domain.pressureRHSdiv_grad[flatPos + s_block.globalOffset];
		
		if((nonOrthoFlags & NON_ORTHO_DIRECT_RHS) | (nonOrthoFlags & NON_ORTHO_DIAGONAL_RHS)){
			scatterNonOrthoLaplaceRHSDimSwitch_v2_GRAD<scalar_t>(RHSdiv_grad, pos, s_block, s_domain, nonOrthoFlags,
				GridDataType::PRESSURE_RESULT_GRAD, false, true, useFaceTransform);
		}
	)
}

#endif //WITH_GRAD

template <typename scalar_t>
void _SetupPressureCorrection(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU,
		const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	// make P matrix
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	PISO_build_pressure_matrix<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			nonOrthoFlags, useFaceTransform
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Build p-Matrix");
	
	// make P rhs
	BEGIN_SAMPLE;
	PISO_build_pressure_rhs<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Build p-RHS");
	
	// divergence P rhs
	BEGIN_SAMPLE;
	k_computePressureRHSdivergenceFromFlux<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			useFaceTransform, timeStepNorm
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Divergence p-RHS");
	
	if((nonOrthoFlags & NON_ORTHO_DIRECT_RHS) | (nonOrthoFlags & NON_ORTHO_DIAGONAL_RHS)){
		BEGIN_SAMPLE;
		k_pressureRHSaddNonOrthoComponents<scalar_t><<<blocks, threads>>>(
				reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), //timeStepCPU.data_ptr<scalar_t>()[0],
				p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
				nonOrthoFlags, useFaceTransform
			);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		END_SAMPLE("p-RHS add non ortho");
	}
	
}
void SetupPressureCorrection(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
		const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	domain->setTimeStep(timeStep); // [1] or one per batched environment
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "SetupPressureCorrection", ([&] {
		_SetupPressureCorrection<scalar_t>(
			domain, timeStep, nonOrthoFlags, useFaceTransform, timeStepNorm
		);
	}));
}



template <typename scalar_t>
void _SetupPressureMatrix(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU, const int8_t nonOrthoFlags, const bool useFaceTransform){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	// make P matrix
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	PISO_build_pressure_matrix<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			nonOrthoFlags, useFaceTransform
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Build p-Matrix");
	
}
void SetupPressureMatrix(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags, const bool useFaceTransform){

	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	domain->setTimeStep(timeStep); // [1] or one per batched environment

	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "SetupPressureMatrix", ([&] {
		_SetupPressureMatrix<scalar_t>(
			domain, timeStep, nonOrthoFlags, useFaceTransform
		);
	}));
}


template <typename scalar_t>
void _SetupPressureRHS(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU,
		const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	
	// make P rhs
	BEGIN_SAMPLE;
	PISO_build_pressure_rhs<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Build p-RHS");
	
	// divergence P rhs
	BEGIN_SAMPLE;
	k_computePressureRHSdivergenceFromFlux<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			useFaceTransform, timeStepNorm
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Divergence p-RHS");
	
	if((nonOrthoFlags & NON_ORTHO_DIRECT_RHS) | (nonOrthoFlags & NON_ORTHO_DIAGONAL_RHS)){
		BEGIN_SAMPLE;
		k_pressureRHSaddNonOrthoComponents<scalar_t><<<blocks, threads>>>(
				reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), //timeStepCPU.data_ptr<scalar_t>()[0],
				p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
				nonOrthoFlags, useFaceTransform
			);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		END_SAMPLE("p-RHS add non ortho");
	}
	
}
void SetupPressureRHS(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
		const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	domain->setTimeStep(timeStep); // [1] or one per batched environment
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "SetupPressureRHS", ([&] {
		_SetupPressureRHS<scalar_t>(
			domain, timeStep, nonOrthoFlags, useFaceTransform, timeStepNorm
		);
	}));
}

template <typename scalar_t>
void _SetupPressureRHSdiv(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU,
		const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	// divergence P rhs
	BEGIN_SAMPLE;
	k_computePressureRHSdivergenceFromFlux<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			useFaceTransform, timeStepNorm
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Divergence p-RHS");
	
	if((nonOrthoFlags & NON_ORTHO_DIRECT_RHS) | (nonOrthoFlags & NON_ORTHO_DIAGONAL_RHS)){
		BEGIN_SAMPLE;
		k_pressureRHSaddNonOrthoComponents<scalar_t><<<blocks, threads>>>(
				reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), //timeStepCPU.data_ptr<scalar_t>()[0],
				p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
				nonOrthoFlags, useFaceTransform
			);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		END_SAMPLE("p-RHS add non ortho");
	}
	
}
void SetupPressureRHSdiv(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
		const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	domain->setTimeStep(timeStep); // [1] or one per batched environment
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "SetupPressureRHSdiv", ([&] {
		_SetupPressureRHSdiv<scalar_t>(
			domain, timeStep, nonOrthoFlags, useFaceTransform, timeStepNorm
		);
	}));
}

#ifdef WITH_GRAD
template <typename scalar_t>
void _SetupPressureCorrection_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU,
		const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	if((nonOrthoFlags & NON_ORTHO_DIRECT_RHS) | (nonOrthoFlags & NON_ORTHO_DIAGONAL_RHS)){
		BEGIN_SAMPLE;
		k_pressureRHSaddNonOrthoComponents_GRAD<scalar_t><<<blocks, threads>>>(
				reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), //timeStepCPU.data_ptr<scalar_t>()[0],
				p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
				nonOrthoFlags, useFaceTransform
			);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		END_SAMPLE("p-RHS add non ortho");
	}
	
	// divergence P rhs
	BEGIN_SAMPLE;
	k_computePressureRHSdivergenceFromFlux_GRAD<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			timeStepNorm
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Divergence p-RHS grad");
	
	// make P rhs
	BEGIN_SAMPLE;
	PISO_build_pressure_rhs_GRAD<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Build p-RHS grad");
	
	// Deferred high-order correction, as a second launch so its atomicAdd scatter
	// cannot race with the assignment to velocity_grad above. CENTRAL has no
	// correction term, so it needs no adjoint here.
	if(domain->getAdvectionScheme()==AdvectionScheme::LINEAR_UPWIND){
		BEGIN_SAMPLE;
		PISO_build_pressure_rhs_deferred_GRAD<scalar_t><<<blocks, threads>>>(
				reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
				p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
			);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		END_SAMPLE("Build p-RHS deferred grad");
	}
	
	BEGIN_SAMPLE;
	PISO_build_pressure_matrix_GRAD<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			nonOrthoFlags, useFaceTransform
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Build p-Matrix grad");
	
}
void SetupPressureCorrection_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
		const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	domain->setTimeStep(timeStep); // [1] or one per batched environment
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "SetupPressureCorrection_GRAD", ([&] {
		_SetupPressureCorrection_GRAD<scalar_t>(
			domain, timeStep, nonOrthoFlags, useFaceTransform, timeStepNorm
		);
	}));
}

template <typename scalar_t>
void _SetupPressureMatrix_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU, const int8_t nonOrthoFlags, const bool useFaceTransform){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	PISO_build_pressure_matrix_GRAD<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			nonOrthoFlags, useFaceTransform
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Build p-Matrix grad");
	
}
void SetupPressureMatrix_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags, const bool useFaceTransform){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	domain->setTimeStep(timeStep); // [1] or one per batched environment
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "SetupPressureMatrix_GRAD", ([&] {
		_SetupPressureMatrix_GRAD<scalar_t>(
			domain, timeStep, nonOrthoFlags, useFaceTransform
		);
	}));
}


template <typename scalar_t>
void _SetupPressureRHS_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU,
		const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	// Non-ortho components
	if((nonOrthoFlags & NON_ORTHO_DIRECT_RHS) | (nonOrthoFlags & NON_ORTHO_DIAGONAL_RHS)){
		BEGIN_SAMPLE;
		k_pressureRHSaddNonOrthoComponents_GRAD<scalar_t><<<blocks, threads>>>(
				reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), //timeStepCPU.data_ptr<scalar_t>()[0],
				p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
				nonOrthoFlags, useFaceTransform
			);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		END_SAMPLE("p-RHS add non ortho");
	}
	
	// divergence P rhs
	BEGIN_SAMPLE;
	k_computePressureRHSdivergenceFromFlux_GRAD<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			timeStepNorm
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Divergence p-RHS grad");
	
	// make P rhs
	BEGIN_SAMPLE;
	PISO_build_pressure_rhs_GRAD<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Build p-RHS grad");
	
	// Deferred high-order correction, as a second launch so its atomicAdd scatter
	// cannot race with the assignment to velocity_grad above. CENTRAL has no
	// correction term, so it needs no adjoint here.
	if(domain->getAdvectionScheme()==AdvectionScheme::LINEAR_UPWIND){
		BEGIN_SAMPLE;
		PISO_build_pressure_rhs_deferred_GRAD<scalar_t><<<blocks, threads>>>(
				reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
				p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
			);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		END_SAMPLE("Build p-RHS deferred grad");
	}
	
}
void SetupPressureRHS_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
		const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	domain->setTimeStep(timeStep); // [1] or one per batched environment

	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "SetupPressureRHS_GRAD", ([&] {
		_SetupPressureRHS_GRAD<scalar_t>(
			domain, timeStep, nonOrthoFlags, useFaceTransform, timeStepNorm
		);
	}));
}


template <typename scalar_t>
void _SetupPressureRHSdiv_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU,
		const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	// Non-ortho components
	if((nonOrthoFlags & NON_ORTHO_DIRECT_RHS) | (nonOrthoFlags & NON_ORTHO_DIAGONAL_RHS)){
		BEGIN_SAMPLE;
		k_pressureRHSaddNonOrthoComponents_GRAD<scalar_t><<<blocks, threads>>>(
				reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), //timeStepCPU.data_ptr<scalar_t>()[0],
				p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
				nonOrthoFlags, useFaceTransform
			);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		END_SAMPLE("p-RHS add non ortho");
	}
	
	// divergence P rhs
	BEGIN_SAMPLE;
	k_computePressureRHSdivergenceFromFlux_GRAD<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			timeStepNorm
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Divergence p-RHS grad");
	
}
void SetupPressureRHSdiv_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
		const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	domain->setTimeStep(timeStep); // [1] or one per batched environment
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "SetupPressureRHSdiv_GRAD", ([&] {
		_SetupPressureRHSdiv_GRAD<scalar_t>(
			domain, timeStep, nonOrthoFlags, useFaceTransform, timeStepNorm
		);
	}));
}

#endif //WITH_GRAD
