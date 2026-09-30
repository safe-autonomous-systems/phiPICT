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
// Inductionless MHD: electric potential matrix, Poisson RHS, field gradients,
// face-based current density, and their gradients.

#include "piso/device/common.cuh"

/* ==================== MHD: Electric potential Laplacian matrix ==================== */

/** Standard Laplacian kernel for the electric potential Poisson equation.
 *  Identical to PISO_build_pressure_matrix but without 1/A weighting:
 *  raP = raN = 1.0 instead of 1/Adiag[...].
 *  Writes to domain.Epot instead of domain.P.
 */
template <typename scalar_t>
__global__ void PISO_build_epot_matrix(DomainGPU<scalar_t> *p_domain,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks,
		const int8_t nonOrthoFlags, const bool useFaceTransform){

	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,

		const I4 pos = unflattenIndex(flatPos, s_block);
		const index_t flatPosGlobal = s_block.globalOffset + flattenIndex(pos, s_block);

		const RowMeta row = getCSRMatrixRowEndOffsetFromBlockBoundaries3D(flatPos, s_block, s_domain);
		const index_t rowStartOffset = row.endOffset - row.size + s_block.csrOffset;

		s_domain.Epot.row[flatPosGlobal+1] = row.endOffset + s_block.csrOffset;

		index_t indices[7]; // diag,-x,+x,-y,+y,-z,+z
		scalar_t rowValues[7] = {0};

		// orthogonal Laplace coefficients (raP = 1 — no 1/A weighting)
		scalar_t alphaP[3];
		for(index_t dim=0;dim<s_domain.numDims;++dim){
			I4 tempPos = pos;
			tempPos.w = dim;
			alphaP[dim] = getLaplaceCoefficientOrthogonalDimSwitch(tempPos, &s_block, s_domain.numDims);
		}
		const scalar_t raP = static_cast<scalar_t>(1.0);
		indices[0] = flatPosGlobal;

		for(index_t bound=0; bound<(s_domain.numDims*2); ++bound){
			const index_t dim = axisFromBound(bound);
			const index_t isUpper = boundIsUpper(bound);
			const index_t faceSign = faceSignFromBound(bound);
			const bool atBound = isAtBound(pos, bound, &s_block);
			const bool atPrescribedBound = atBound && isEmptyBound(bound, s_block.boundaries);

			if(!atPrescribedBound){
				{
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
					const scalar_t raN = static_cast<scalar_t>(1.0); // no 1/A weighting

					scalar_t coefficient = 0;
					if(useFaceTransform){
						tempPos = pos;
						p_block = &s_block;
						const scalar_t alphaFace = getLaplaceCoefficientOrthogonalFaceDimSwitch(tempPos, bound, p_block, s_domain.numDims);
						const scalar_t raFace = static_cast<scalar_t>(0.5) * (raP + raN);
						coefficient = alphaFace * raFace;
					}else{
						coefficient = static_cast<scalar_t>(0.5) * (alphaP[dim]*raP + alphaN*raN);
					}

					rowValues[0] -= coefficient;
					rowValues[bound+1] += coefficient;
					indices[bound+1] = tempFlatPosGlobal;
				}

				const bool includeNonOrthoNeighbors = nonOrthoFlags & NON_ORTHO_DIRECT_MATRIX;
				const bool includeNonOrthoDiag = nonOrthoFlags & NON_ORTHO_CENTER_MATRIX;
				if(s_domain.numDims>1 && (includeNonOrthoDiag || includeNonOrthoNeighbors)){
					scalar_t alphaInterp[12];
					if(s_domain.numDims==2) interpolateNonOrthoLaplaceComponents<scalar_t, 2>(pos, s_block, s_domain, alphaInterp, false, true, useFaceTransform);
					else if(s_domain.numDims==3) interpolateNonOrthoLaplaceComponents<scalar_t, 3>(pos, s_block, s_domain, alphaInterp, false, true, useFaceTransform);

					for(index_t i=1; i<s_domain.numDims; ++i){
						const index_t tAxis = (dim + i)%s_domain.numDims;
						const scalar_t alpha = getInterpolatedNonOrthoLaplaceComponent(alphaInterp, bound, tAxis, s_domain.numDims);

						if(alpha!=0){
							for(index_t tIsUpper=0; tIsUpper<2; ++tIsUpper){
								const index_t tFace = axisToBound(tAxis, tIsUpper);
								const index_t tFaceSign = faceSignFromBound(tFace);
								const CornerValue<scalar_t> cVal = getCornerValue<scalar_t>(pos, bound, tFace,
									false, false, 2, s_block, s_domain, GridDataType::IS_FIXED_BOUNDARY);
								const bool cornerAtBound = cVal.numCells<1;
								const bool boundIsGradient = true;

								if(cornerAtBound){
									if(boundIsGradient){
										const scalar_t interpolationNorm = 0.25;
										const index_t tFaceOther = invertBound(tFace);
										const scalar_t coefficient = faceSign * tFaceSign * alpha * interpolationNorm;
										if(includeNonOrthoDiag){
											rowValues[0] += 3 * coefficient;
										}
										if(includeNonOrthoNeighbors){
											rowValues[bound+1] += 3 * coefficient;
											rowValues[tFaceOther+1] -= coefficient;
										}
									}
								} else {
									const scalar_t interpolationNorm = 1.0 / static_cast<scalar_t>(cVal.numCells);
									const scalar_t coefficient = faceSign * tFaceSign * alpha * interpolationNorm;
									if(includeNonOrthoDiag){
										rowValues[0] += coefficient;
									}
									if(includeNonOrthoNeighbors){
										rowValues[bound+1] += coefficient;
										rowValues[tFace+1] += coefficient;
									}
								}
							}
						}
					}
				}
			} else {
				indices[bound+1] = -1;
				// Dirichlet φ=0: treat outflow as ghost cell with φ=0.
				// Half-cell coefficient uses the same orthogonal alpha as the interior
				// face but with the boundary cell's own metric (no neighbor).
				// RHS contribution is -coef * 0 = 0, so only diagonal is modified.
				if (isEpotDirichletBound(pos, bound, s_block.boundaries)) {
					const scalar_t coef = alphaP[dim] * raP;
					rowValues[0] -= coef;
				}
			}
		}

		for(int dim=s_domain.numDims;dim<3;++dim){
			indices[dim*2+1] = -1;
			indices[dim*2+2] = -1;
		}

		// Thin-wall Robin correction: for each THIN_WALL boundary face cell of cell P,
		// add C_w * nabla^2_tau(phi)|_w to the tangential off-diagonal entries.
		// This implements the fluid-domain-only thin-wall approximation:
		//   dphi/dn|_w = C_w * nabla^2_tau(phi)|_w
		// approximated by using the adjacent fluid cell values for the tangential Laplacian.
		for(index_t twBound = 0; twBound < (s_domain.numDims*2); ++twBound){
			const index_t twDim = axisFromBound(twBound);
			if(!isAtBound(pos, twBound, &s_block)) continue;
			if(!isThinWallBound(pos, twBound, s_block.boundaries)) continue;

			const scalar_t Cw = s_block.boundaries[twBound].fb.potentialCw;
			const scalar_t Minv_n_P = getMinvNormDimSwitch(pos, twDim, &s_block, s_domain.numDims);

			for(index_t tDim = 0; tDim < s_domain.numDims; ++tDim){
				if(tDim == twDim) continue;
				for(index_t tSide = 0; tSide < 2; ++tSide){
					const index_t tBound = tDim * 2 + tSide;
					if(indices[tBound+1] < 0) continue; // no tangential neighbor (insulating edge)

					const index_t tFaceSign = faceSignFromBound(tBound);
					const bool atTBound = isAtBound(pos, tBound, &s_block);
					I4 tNbrPos = pos;
					if(atTBound){
						if(s_block.boundaries[tBound].type == BoundaryType::PERIODIC){
							tNbrPos.a[tDim] = (tSide == 1) ? 0 : s_block.size.a[tDim] - 1;
						} else {
							continue; // connected or other: skip for surface Laplacian
						}
					} else {
						tNbrPos.a[tDim] = pos.a[tDim] + tFaceSign;
					}
					// The wall current only flows along the conducting part of a mixed face:
					// no surface-Laplacian link to a neighbour whose face cell is not thin wall.
					if(!isThinWallBound(tNbrPos, twBound, s_block.boundaries)) continue;

					// Alpha at tangential neighbor in the tangential direction
					tNbrPos.w = tDim;
					const scalar_t alphaN_t = getLaplaceCoefficientOrthogonalDimSwitch(tNbrPos, &s_block, s_domain.numDims);

					// ||Minv[normalAxis]|| at tangential neighbor (for curved grids)
					tNbrPos.w = 0;
					const scalar_t Minv_n_N = getMinvNormDimSwitch(tNbrPos, twDim, &s_block, s_domain.numDims);

					// Surface Laplace coefficient (averaged, consistent with interior face treatment)
					const scalar_t surfCoeff = Cw * static_cast<scalar_t>(0.5) * (alphaP[tDim] * Minv_n_P + alphaN_t * Minv_n_N);

					rowValues[0]        -= surfCoeff; // diagonal
					rowValues[tBound+1] += surfCoeff; // tangential off-diagonal
				}
			}
		}

		for(index_t i=0;i<row.size;++i){
			index_t colIndex = findLowestColumnIndex(indices, 7);
			if(colIndex<0){
				s_domain.Epot.index[rowStartOffset + i] = -1;
				s_domain.Epot.value[rowStartOffset + i] = -1.0f;
			}else{
				s_domain.Epot.index[rowStartOffset + i] = indices[colIndex];
				s_domain.Epot.value[rowStartOffset + i] = rowValues[colIndex];
			}
			indices[colIndex] = -1;
		}
	)
}

template <typename scalar_t>
void _SetupEpotMatrix(std::shared_ptr<Domain> domain, const int8_t nonOrthoFlags, const bool useFaceTransform){

	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	PISO_build_epot_matrix<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			nonOrthoFlags, useFaceTransform
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Build epot-Matrix");
}

void SetupEpotMatrix(std::shared_ptr<Domain> domain, const int8_t nonOrthoFlags, const bool useFaceTransform){

	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	TORCH_CHECK(domain->Epot.has_value(), "domain.Epot is not allocated. Call domain.CreateEpotMatrix() first.");

	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "SetupEpotMatrix", ([&] {
		_SetupEpotMatrix<scalar_t>(domain, nonOrthoFlags, useFaceTransform);
	}));
}


/* ==================== MHD: Electric-potential Poisson RHS ==================== */

/** Compute the epot Poisson RHS div(u×B) using FV fluxes.
 *  vectorField layout: [totalSize * spatialDims] — same as domain.pressureRHS.
 *  At empty/prescribed boundaries the normal flux χ_b=(u×B)_n is included (∂φ/∂n=0),
 *  matching j_b in ComputeCurrentDensityFaceBased so div(j)=0 holds discretely there.
 *  Output: divergence[totalSize].
 */
template <typename scalar_t>
__global__ void k_computeEpotRHS(DomainGPU<scalar_t> *p_domain, const scalar_t *vectorField,
		scalar_t *divergence,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks,
		const bool useFaceTransform, const index_t vectorFieldBatchStride, const index_t divergenceBatchStride){
	// batched environments: flat per-environment slices
	vectorField += blockIdx.y * vectorFieldBatchStride;
	divergence += blockIdx.y * divergenceBatchStride;


	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,

		const I4 pos = unflattenIndex(flatPos, s_block);

		scalar_t fluxes[6]; // DIMS*2
		if(useFaceTransform){
			switch(s_domain.numDims){
				case 1: computeFluxesWithFaceTransforms<scalar_t, 1>(pos, fluxes, s_block, s_domain, vectorField); break;
				case 2: computeFluxesWithFaceTransforms<scalar_t, 2>(pos, fluxes, s_block, s_domain, vectorField); break;
				case 3: computeFluxesWithFaceTransforms<scalar_t, 3>(pos, fluxes, s_block, s_domain, vectorField); break;
				default: break;
			}
		}else{
			// ∂φ/∂n=0: include the boundary flux χ_b=(u×B)_n in the Poisson RHS.
			// computeEpotFluxesNDLoop sets the empty-boundary flux to the cell-centre
			// (u×B)_n, exactly matching j_b in ComputeCurrentDensityFaceBased, so
			// div(j)=0 holds discretely at boundary cells (current may exit the domain).
			computeEpotFluxesNDLoop(pos, fluxes, s_block, s_domain, vectorField);
		}

		scalar_t div = 0;
		for(index_t dim=0;dim<s_domain.numDims;++dim){
			div += fluxes[dim*2+1] - fluxes[dim*2];
		}

		// Prescribed φ = g at a Dirichlet face cell: the matrix has the ghost cell's
		// diagonal part (-coef*φ_P), its known part coef*g moves to the RHS. Same
		// coefficient as PISO_build_epot_matrix (alphaP*raP, raP=1).
		for(index_t bound=0; bound<(s_domain.numDims*2); ++bound){
			if(!isAtBound(pos, bound, &s_block) || !isEmptyBound(bound, s_block.boundaries)) continue;
			if(!isEpotDirichletBound(pos, bound, s_block.boundaries)) continue;
			const scalar_t g = potentialValueAt(pos, bound, s_block.boundaries);
			if(g == static_cast<scalar_t>(0)) continue;
			I4 tempPos = pos;
			tempPos.w = axisFromBound(bound);
			div -= getLaplaceCoefficientOrthogonalDimSwitch(tempPos, &s_block, s_domain.numDims) * g;
		}

		divergence[flatPos + s_block.globalOffset] = div;
	)
}

template <typename scalar_t>
void _ComputeEpotRHS(std::shared_ptr<Domain> domain, const torch::Tensor &vectorField,
		torch::Tensor &divergence, const bool useFaceTransform){

	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)

	BEGIN_SAMPLE;
	k_computeEpotRHS<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
			vectorField.data_ptr<scalar_t>(),
			divergence.data_ptr<scalar_t>(),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			useFaceTransform, vectorField.numel()/domain->getBatchSize(), divergence.numel()/domain->getBatchSize()
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Epot RHS");
}

torch::Tensor ComputeEpotRHS(std::shared_ptr<Domain> domain, const torch::Tensor &vectorField){

	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	TORCH_CHECK(vectorField.dim()==1, "vectorField must be 1D (flat).");
	TORCH_CHECK(vectorField.size(0)==(index_t)(domain->getTotalSize() * domain->getSpatialDims() * domain->getBatchSize()),
		"vectorField size must be totalSize * spatialDims (* batch size).");
	TORCH_CHECK(vectorField.device()==domain->getDevice(), "vectorField must be on the same device as the domain.");
	TORCH_CHECK(vectorField.scalar_type()==domain->getDtype(), "vectorField dtype must match domain dtype.");
	CHECK_CONTIGUOUS(vectorField);

	torch::Tensor divergence = torch::zeros_like(domain->pressureResult);

	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "ComputeEpotRHS", ([&] {
		_ComputeEpotRHS<scalar_t>(domain, vectorField, divergence, false);
	}));

	return divergence;
}

/* ==================== MHD: Generic scalar field gradient ==================== */

/** Read a value from a flat scalar field with Neumann (zero-gradient) boundary handling.
 *  scalarField: flat [totalSize], indexed as scalarField[flattenIndex(pos, block) + block.globalOffset].
 *  At boundaries, mirrors the nearest interior cell (giving zero normal gradient).
 */
template <typename scalar_t>
__device__ inline scalar_t getScalarFieldAtNeumann(const I4 pos, const BlockGPU<scalar_t> &block, const scalar_t *scalarField){
	I4 tempPos = pos;
	tempPos.w = 0;
	// Clamp out-of-bounds coordinates to the nearest valid cell (Neumann mirror).
	for(int dim=0; dim<4; ++dim){
		if(tempPos.a[dim] < 0) tempPos.a[dim] = 0;
		if(dim < 3 && tempPos.a[dim] >= block.size.a[dim]) tempPos.a[dim] = block.size.a[dim] - 1;
	}
	return scalarField[flattenIndex(tempPos, block) + block.globalOffset];
}

/** Compute gradient of an arbitrary flat scalar field.
 *  Uses central differences with Neumann (zero-gradient) BCs at domain boundaries.
 *  scalarField: flat [totalSize].
 *  gradient output: flat [totalSize * spatialDims], same layout as domain.velocityResult.
 */
template <typename scalar_t>
__global__ void k_computeFieldGradient(DomainGPU<scalar_t> *p_domain, const scalar_t *scalarField,
		scalar_t *gradient,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks, const index_t fieldBatchStride, const index_t gradientBatchStride){
	// batched environments: flat per-environment slices
	scalarField += blockIdx.y * fieldBatchStride;
	gradient += blockIdx.y * gradientBatchStride;


	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,

		I4 pos = unflattenIndex(flatPos, s_block);

		// Collect all index-space differences first (needed for full Minv matrix transform)
		scalar_t grad[3] = {0, 0, 0};

		for(index_t dim=0; dim<s_domain.numDims; ++dim){
			scalar_t fac = static_cast<scalar_t>(0.5);

			// lower neighbour
			I4 posN = pos;
			if(pos.a[dim] > 0){
				posN.a[dim] = pos.a[dim] - 1;
			}else{
				// at lower boundary: Neumann — mirror, one-sided difference
				fac = static_cast<scalar_t>(1.0);
			}
			const scalar_t valN = getScalarFieldAtNeumann<scalar_t>(posN, s_block, scalarField);

			// upper neighbour
			I4 posP = pos;
			if(pos.a[dim] < s_block.size.a[dim] - 1){
				posP.a[dim] = pos.a[dim] + 1;
			}else{
				// at upper boundary: Neumann — mirror, one-sided difference
				fac = static_cast<scalar_t>(1.0);
			}
			const scalar_t valP = getScalarFieldAtNeumann<scalar_t>(posP, s_block, scalarField);

			grad[dim] = (valP - valN) * fac;
		}

		// Apply coordinate transform (same as getPressureGradient)
		if(s_block.hasTransform){
			I4 tempPos = pos;
			tempPos.w = 0;
			switch(s_domain.numDims){
			case 2: {
				const TransformGPU<scalar_t, 2> *T = reinterpret_cast<TransformGPU<scalar_t, 2>*>(s_block.transform) + flattenIndex(tempPos, s_block);
				Vector<scalar_t, 2> g; g.a[0] = grad[0]; g.a[1] = grad[1];
				g = matmul(g, T->Minv);
				grad[0] = g.a[0]; grad[1] = g.a[1];
				break;
			}
			case 3: {
				const TransformGPU<scalar_t, 3> *T = reinterpret_cast<TransformGPU<scalar_t, 3>*>(s_block.transform) + flattenIndex(tempPos, s_block);
				Vector<scalar_t, 3> g; g.a[0] = grad[0]; g.a[1] = grad[1]; g.a[2] = grad[2];
				g = matmul(g, T->Minv);
				grad[0] = g.a[0]; grad[1] = g.a[1]; grad[2] = g.a[2];
				break;
			}
			default: break;
			}
		}

		// Write out transformed gradient components
		for(index_t dim=0; dim<s_domain.numDims; ++dim){
			I4 tempPos = pos;
			tempPos.w = dim;
			const index_t flatCompPosGlobal = flattenIndexGlobal(tempPos, s_block, s_domain);
			gradient[flatCompPosGlobal] = grad[dim];
		}
	)
}

template <typename scalar_t>
void _ComputeFieldGradient(std::shared_ptr<Domain> domain, const torch::Tensor &scalarField, torch::Tensor &gradient){

	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)

	BEGIN_SAMPLE;
	k_computeFieldGradient<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
			scalarField.data_ptr<scalar_t>(),
			gradient.data_ptr<scalar_t>(),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			scalarField.numel()/domain->getBatchSize(), gradient.numel()/domain->getBatchSize()
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Field Gradient");
}

torch::Tensor ComputeFieldGradient(std::shared_ptr<Domain> domain, const torch::Tensor &scalarField){

	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	TORCH_CHECK(scalarField.dim()==1, "scalarField must be 1D (flat).");
	TORCH_CHECK(scalarField.size(0)==(index_t)domain->getTotalSize() * domain->getBatchSize(),
		"scalarField size must equal domain totalSize (* batch size).");
	TORCH_CHECK(scalarField.device()==domain->getDevice(), "scalarField must be on the same device as the domain.");
	TORCH_CHECK(scalarField.scalar_type()==domain->getDtype(), "scalarField dtype must match domain dtype.");
	CHECK_CONTIGUOUS(scalarField);

	torch::Tensor gradient = torch::zeros_like(domain->velocityResult);

	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "ComputeFieldGradient", ([&] {
		_ComputeFieldGradient<scalar_t>(domain, scalarField, gradient);
	}));

	return gradient;
}

/** Compute FVM (Green-Gauss) gradient of an arbitrary flat scalar field.
 *  Uses the same face-based stencil as the Laplacian, ensuring consistency
 *  with the Poisson equation.  Modeled after getPressureGradientFVM.
 *  scalarField: flat [totalSize].
 *  gradient output: flat [totalSize * spatialDims].
 */
template <typename scalar_t, int DIMS>
__device__ inline Vector<scalar_t,DIMS> getScalarGradientFVM(
		const BlockGPU<scalar_t> &block, I4 pos, const DomainGPU<scalar_t> &domain,
		const scalar_t *scalarField){
	// For blocks without transforms, central differences are equivalent to FVM gradient.
	if(!block.hasTransform){
		Vector<scalar_t,DIMS> grad;
		for(index_t dim=0; dim<DIMS; ++dim){
			I4 posN = pos, posP = pos;
			posN.w = posP.w = 0;
			scalar_t fac = static_cast<scalar_t>(0.5);
			if(pos.a[dim] > 0){
				posN.a[dim] = pos.a[dim] - 1;
			}else{
				fac = static_cast<scalar_t>(1.0);
			}
			if(pos.a[dim] < block.size.a[dim] - 1){
				posP.a[dim] = pos.a[dim] + 1;
			}else{
				fac = static_cast<scalar_t>(1.0);
			}
			const scalar_t valN = scalarField[flattenIndex(posN, block) + block.globalOffset];
			const scalar_t valP = scalarField[flattenIndex(posP, block) + block.globalOffset];
			grad.a[dim] = (valP - valN) * fac;
		}
		return grad;
	}

	// FVM gradient using Green-Gauss theorem: grad(φ) = (1/V) * Σ_f φ_f · S_f
	Vector<scalar_t,DIMS> scalarGrad = {.a={0}};

	pos.w = 0;
	const index_t flatIndexP = flattenIndex(pos, block);
	const TransformGPU<scalar_t, DIMS> *p_TP = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(block.transform) + flatIndexP;
	const scalar_t phiP = scalarField[flatIndexP + block.globalOffset];

	for(index_t face=0; face<(DIMS*2); ++face){
		const index_t dim = axisFromBound(face);
		const bool isUpper = boundIsUpper(face);
		const index_t faceSign = faceSignFromBound(face);
		const bool atBound = isAtBound(pos, face, &block);

		const Vector<scalar_t, DIMS> fluxP = p_TP->Minv.v[dim] * (p_TP->det * phiP);

		I4 tempPos = pos;
		const TransformGPU<scalar_t, DIMS> *p_T = nullptr;
		index_t axisNeighbor = dim;
		scalar_t transformSign = 1;
		scalar_t phi = 0;

		if(atBound){
			switch(block.boundaries[face].type){
			case BoundaryType::DIRICHLET:
				// Static Dirichlet: Neumann-like extrapolation for the scalar field.
				p_T = p_TP;
				tempPos.a[dim] = pos.a[dim] - faceSign;
				phi = 2*phiP - scalarField[flattenIndex(tempPos, block) + block.globalOffset];
				break;
			case BoundaryType::DIRICHLET_VARYING:
			case BoundaryType::FIXED:
			{
				// Neumann BC (zero normal gradient): extrapolate.
				tempPos.a[dim] = 0;
				if(block.boundaries[face].type==BoundaryType::DIRICHLET_VARYING){
					p_T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(block.boundaries[face].vdb.transform) + flattenIndex(tempPos, block.boundaries[face].vdb.stride);
				} else {
					p_T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(block.boundaries[face].fb.transform) + flattenIndex(tempPos, block.boundaries[face].fb.stride);
				}
				tempPos.a[dim] = pos.a[dim] - faceSign;
				phi = phiP + (phiP - scalarField[flattenIndex(tempPos, block) + block.globalOffset]) * static_cast<scalar_t>(0.5);
				const Vector<scalar_t, DIMS> fluxN = p_T->Minv.v[dim] * (p_T->det * phi);
				scalarGrad += fluxN * static_cast<scalar_t>(faceSign);
				continue;
			}
			case BoundaryType::GRADIENT:
				continue;
			case BoundaryType::CONNECTED_GRID:
			{
				const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + block.boundaries[face].cb.connectedGridIndex;
				tempPos.w = dim;
				I4 otherPos = computeConnectedPosWithChannel(tempPos, dim, &block.boundaries[face].cb, domain);
				axisNeighbor = otherPos.w;
				const bool otherIsUpper = boundIsUpper(block.boundaries[face].cb.axes.a[0]);
				if(otherIsUpper==isUpper) {
					transformSign = -1;
				}
				otherPos.w = 0;
				const index_t flatIndex = flattenIndex(otherPos, p_connectedBlock);
				p_T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_connectedBlock->transform) + flatIndex;
				phi = scalarField[flatIndex + p_connectedBlock->globalOffset];
				break;
			}
			case BoundaryType::PERIODIC:
			{
				tempPos.a[dim] = isUpper ? 0 : block.size.a[dim] - 1;
				const index_t flatIndex = flattenIndex(tempPos, block);
				p_T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(block.transform) + flatIndex;
				phi = scalarField[flatIndex + block.globalOffset];
				break;
			}
			default:
				continue;
			}
		} else {
			tempPos.a[dim] += faceSign;
			const index_t flatIndex = flattenIndex(tempPos, block);
			p_T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(block.transform) + flatIndex;
			phi = scalarField[flatIndex + block.globalOffset];
		}

		// Interpolate cell-center fluxes to face (lerp)
		const Vector<scalar_t, DIMS> fluxN = p_T->Minv.v[axisNeighbor] * (p_T->det * phi * transformSign);
		scalarGrad += (fluxP + fluxN) * (static_cast<scalar_t>(0.5) * faceSign);
	}

	// Divide by cell volume (det) to get the gradient
	scalarGrad *= static_cast<scalar_t>(1.0) / (p_TP->det);

	return scalarGrad;
}

template <typename scalar_t>
__global__ void k_computeFieldGradientFVM(DomainGPU<scalar_t> *p_domain, const scalar_t *scalarField,
		scalar_t *gradient,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks, const index_t fieldBatchStride, const index_t gradientBatchStride){
	// batched environments: flat per-environment slices
	scalarField += blockIdx.y * fieldBatchStride;
	gradient += blockIdx.y * gradientBatchStride;


	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,

		I4 pos = unflattenIndex(flatPos, s_block);

		Vector<scalar_t, 3> grad = {.a={0}};
		switch(s_domain.numDims){
		case 2: {
			Vector<scalar_t, 2> g = getScalarGradientFVM<scalar_t, 2>(s_block, pos, s_domain, scalarField);
			grad.a[0] = g.a[0]; grad.a[1] = g.a[1];
			break;
		}
		case 3: {
			Vector<scalar_t, 3> g = getScalarGradientFVM<scalar_t, 3>(s_block, pos, s_domain, scalarField);
			grad = g;
			break;
		}
		default: break;
		}

		// Write out gradient components
		for(index_t dim=0; dim<s_domain.numDims; ++dim){
			I4 tempPos = pos;
			tempPos.w = dim;
			const index_t flatCompPosGlobal = flattenIndexGlobal(tempPos, s_block, s_domain);
			gradient[flatCompPosGlobal] = grad.a[dim];
		}
	)
}

template <typename scalar_t>
void _ComputeFieldGradientFVM(std::shared_ptr<Domain> domain, const torch::Tensor &scalarField, torch::Tensor &gradient){

	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)

	BEGIN_SAMPLE;
	k_computeFieldGradientFVM<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
			scalarField.data_ptr<scalar_t>(),
			gradient.data_ptr<scalar_t>(),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			scalarField.numel()/domain->getBatchSize(), gradient.numel()/domain->getBatchSize()
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Field Gradient FVM");
}

torch::Tensor ComputeFieldGradientFVM(std::shared_ptr<Domain> domain, const torch::Tensor &scalarField){

	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	TORCH_CHECK(scalarField.dim()==1, "scalarField must be 1D (flat).");
	TORCH_CHECK(scalarField.size(0)==(index_t)domain->getTotalSize() * domain->getBatchSize(),
		"scalarField size must equal domain totalSize (* batch size).");
	TORCH_CHECK(scalarField.device()==domain->getDevice(), "scalarField must be on the same device as the domain.");
	TORCH_CHECK(scalarField.scalar_type()==domain->getDtype(), "scalarField dtype must match domain dtype.");
	CHECK_CONTIGUOUS(scalarField);

	torch::Tensor gradient = torch::zeros_like(domain->velocityResult);

	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "ComputeFieldGradientFVM", ([&] {
		_ComputeFieldGradientFVM<scalar_t>(domain, scalarField, gradient);
	}));

	return gradient;
}

/* ==================== MHD: Face-based Lorentz force ==================== */

/** Compute the Lorentz force using face-based current density reconstruction.
 *
 *  The current density at each face is computed as:
 *      j_face = -alpha_face * (phi_N - phi_P) + chi_face
 *  where alpha_face is the same Laplace coefficient used in the epot matrix,
 *  and chi_face is the face flux of u×B (same as used for the divergence RHS).
 *  This guarantees that div(j) = 0 discretely, matching the Poisson solve.
 *
 *  Cell-center J is reconstructed from face averages and transformed from
 *  contravariant to physical coordinates via the adjugate of Minv.
 *  epotField:    flat [totalSize] — electric potential φ.
 *  uCrossBField: flat [totalSize * DIMS] — u×B (same field used for Poisson RHS).
 *  jOut:         flat [totalSize * 3] — output current density J (always 3 components).
 */
template <typename scalar_t, int DIMS>
__device__ void computeCurrentDensityFaceBased(
		const I4 pos,
		const BlockGPU<scalar_t> &block,
		const DomainGPU<scalar_t> &domain,
		const scalar_t *epotField,
		const scalar_t *uCrossBField,
		scalar_t *jOut)
{
	I4 pos0 = pos;
	pos0.w = 0;
	const index_t flatP = flattenIndex(pos0, block);
	const scalar_t phiP = epotField[flatP + block.globalOffset];

	// Laplace coefficients and contravariant u×B at cell center
	scalar_t alphaP[3] = {0};
	scalar_t cvelP[3] = {0};
	for(index_t d = 0; d < DIMS; ++d){
		I4 tp = pos;
		tp.w = d;
		alphaP[d] = getLaplaceCoefficientOrthogonal<scalar_t, DIMS>(tp, &block);
		cvelP[d] = getContravariantComponent<scalar_t, DIMS>(tp, &block, domain, uCrossBField);
	}

	// Compute face current fluxes
	scalar_t j_face[6] = {0}; // [lower_x, upper_x, lower_y, upper_y, lower_z, upper_z]

	for(index_t bound = 0; bound < DIMS * 2; ++bound){
		const index_t dim = axisFromBound(bound);
		const index_t isUpper = boundIsUpper(bound);
		const index_t faceSign = faceSignFromBound(bound);
		const bool atBound = isAtBound(pos, bound, &block);
		const bool atPrescribedBound = atBound && isEmptyBound(bound, block.boundaries);

		if(atPrescribedBound){
			// Solid wall: insulating, j_n = 0.
			// Open bound: ∂φ/∂n=0 → j_b = (u×B)_n, using the cell-centre value
			// (zero-gradient extrapolation).
			// φ Dirichlet (e.g. a symmetry plane across which φ is odd, where the
			// current must cross the face rather than be reflected, or an electrode):
			// the general face formula with the ghost value φ_N = g,
			// i.e. j_b = alpha*faceSign*(φ_P - g) + (u×B)_n.
			// The alpha is the matrix's Dirichlet coefficient (alphaP*raP, raP=1), so the
			// φ term cancels the diagonal the matrix adds for this face.
			// Prescribed current I into the fluid: j_b = -faceSign*I along +axis (+I through
			// a lower face, -I through an upper one), independent of φ and (u×B).
			// Each case matches the RHS flux in computeEpotFluxesNDLoop, keeping div(j)=0
			// discrete against the Poisson matrix.
			if(isInsulatingWallBound(pos, bound, block.boundaries)){
				j_face[bound] = 0;
			} else if(isEpotCurrentBound(pos, bound, block.boundaries)){
				j_face[bound] = -static_cast<scalar_t>(faceSign)
					* potentialValueAt(pos, bound, block.boundaries);
			} else if(isEpotDirichletBound(pos, bound, block.boundaries)){
				const scalar_t g = potentialValueAt(pos, bound, block.boundaries);
				j_face[bound] = alphaP[dim] * static_cast<scalar_t>(faceSign) * (phiP - g) + cvelP[dim];
			} else {
				j_face[bound] = cvelP[dim];
			}
			continue;
		}

		scalar_t phiN = 0;
		scalar_t alphaN = 0;
		scalar_t cvelN = 0;

		if(atBound){
			switch(block.boundaries[bound].type){
			case BoundaryType::CONNECTED_GRID:
			{
				const BlockGPU<scalar_t> *p_connBlock = domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
				I4 tempPos = pos;
				tempPos.w = dim;
				I4 otherPos = computeConnectedPosWithChannel(tempPos, dim, &block.boundaries[bound].cb, domain);
				const index_t connAxis = otherPos.w;

				// Phi at neighbor
				I4 otherPos0 = otherPos;
				otherPos0.w = 0;
				phiN = epotField[flattenIndex(otherPos0, p_connBlock) + p_connBlock->globalOffset];

				// Alpha at neighbor (use connected axis)
				alphaN = getLaplaceCoefficientOrthogonal<scalar_t, DIMS>(otherPos, p_connBlock);

				// u×B contravariant at neighbor
				I4 velPos = otherPos;
				velPos.w = connAxis;
				cvelN = getContravariantComponent<scalar_t, DIMS>(velPos, p_connBlock, domain, uCrossBField);
				const bool otherIsUpper = boundIsUpper(block.boundaries[bound].cb.axes.a[0]);
				if(otherIsUpper == isUpper) cvelN = -cvelN;
				break;
			}
			case BoundaryType::PERIODIC:
			{
				I4 tempPos = pos;
				tempPos.w = dim;
				tempPos.a[dim] = isUpper ? 0 : block.size.a[dim] - 1;

				I4 tempPos0 = tempPos;
				tempPos0.w = 0;
				phiN = epotField[flattenIndex(tempPos0, block) + block.globalOffset];
				alphaN = getLaplaceCoefficientOrthogonal<scalar_t, DIMS>(tempPos, &block);
				cvelN = getContravariantComponent<scalar_t, DIMS>(tempPos, &block, domain, uCrossBField);
				break;
			}
			default:
				j_face[bound] = 0;
				continue;
			}
		} else {
			I4 tempPos = pos;
			tempPos.w = dim;
			tempPos.a[dim] = pos.a[dim] + faceSign;

			I4 tempPos0 = tempPos;
			tempPos0.w = 0;
			phiN = epotField[flattenIndex(tempPos0, block) + block.globalOffset];
			alphaN = getLaplaceCoefficientOrthogonal<scalar_t, DIMS>(tempPos, &block);
			cvelN = getContravariantComponent<scalar_t, DIMS>(tempPos, &block, domain, uCrossBField);
		}

		// Face Laplace coefficient (same as epot matrix, raP=raN=1)
		const scalar_t coefficient = static_cast<scalar_t>(0.5) * (alphaP[dim] + alphaN);

		// Face flux of u×B (interpolated to face)
		const scalar_t chi_face = static_cast<scalar_t>(0.5) * (cvelP[dim] + cvelN);

		// Face current flux in +d direction: j_f = -alpha * faceSign * (phi_N - phi_P) + chi
		// The faceSign is required because phiN is always the neighbor in the faceSign direction,
		// so (phiN - phiP) gives -(phi_P - phi_M) for the lower face, which has the wrong sign
		// for the +d current. With faceSign: upper (+1) gives -alpha*(phi_N-phi_P), lower (-1)
		// gives +alpha*(phi_M-phi_P). This ensures div(j) = j_upper - j_lower = 0 when Poisson is satisfied.
		j_face[bound] = -coefficient * static_cast<scalar_t>(faceSign) * (phiN - phiP) + chi_face;
	}

	// Average upper and lower face currents to cell center (contravariant)
	scalar_t j_avg[3] = {0};
	for(index_t d = 0; d < DIMS; ++d){
		j_avg[d] = static_cast<scalar_t>(0.5) * (j_face[d*2] + j_face[d*2+1]);
	}

	// Reconstruct physical current density: J = adj(Minv) * j_avg
	// adj(Minv) = M / det(M), both already stored in the transform struct.
	scalar_t J[3] = {0, 0, 0};
	if(block.hasTransform){
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(block.transform) + flatP;
		const scalar_t inv_det = static_cast<scalar_t>(1) / T->det;
		if constexpr (DIMS == 2){
			J[0] = (T->M.v[0].a[0] * j_avg[0] + T->M.v[0].a[1] * j_avg[1]) * inv_det;
			J[1] = (T->M.v[1].a[0] * j_avg[0] + T->M.v[1].a[1] * j_avg[1]) * inv_det;
			// Out-of-plane current: J_z = (u×B)_z — no phi gradient in z for a 2D domain.
			// uCrossBField stores a 3rd component block at offset domain.numCells*2.
			J[2] = uCrossBField[flatP + block.globalOffset + domain.numCells * 2];
		}
		if constexpr (DIMS == 3){
			J[0] = (T->M.v[0].a[0] * j_avg[0] + T->M.v[0].a[1] * j_avg[1] + T->M.v[0].a[2] * j_avg[2]) * inv_det;
			J[1] = (T->M.v[1].a[0] * j_avg[0] + T->M.v[1].a[1] * j_avg[1] + T->M.v[1].a[2] * j_avg[2]) * inv_det;
			J[2] = (T->M.v[2].a[0] * j_avg[0] + T->M.v[2].a[1] * j_avg[1] + T->M.v[2].a[2] * j_avg[2]) * inv_det;
		}
	} else {
		for(index_t d = 0; d < DIMS; ++d) J[d] = j_avg[d];
		// Out-of-plane current for 2D: J_z = (u×B)_z stored at domain.numCells*2 offset.
		if constexpr (DIMS == 2) {
			J[2] = uCrossBField[flatP + block.globalOffset + domain.numCells * 2];
		}
	}

	// Write J to output (always 3 components, stored as [J0..., J1..., J2...])
	const index_t numCells = domain.numCells;
	const index_t globalFlat = flatP + block.globalOffset;
	jOut[globalFlat]                   = J[0];
	jOut[globalFlat + numCells]        = J[1];
	jOut[globalFlat + numCells * 2]    = J[2];
}

template <typename scalar_t>
__global__ void k_computeCurrentDensityFaceBased(DomainGPU<scalar_t> *p_domain,
		const scalar_t *epotField, const scalar_t *uCrossBField,
		scalar_t *jOut,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks, const index_t epotBatchStride, const index_t uCrossBBatchStride, const index_t jBatchStride){
	// batched environments: flat per-environment slices
	epotField += blockIdx.y * epotBatchStride;
	uCrossBField += blockIdx.y * uCrossBBatchStride;
	jOut += blockIdx.y * jBatchStride;


	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,

		I4 pos = unflattenIndex(flatPos, s_block);

		switch(s_domain.numDims){
		case 2:
			computeCurrentDensityFaceBased<scalar_t, 2>(pos, s_block, s_domain,
				epotField, uCrossBField, jOut);
			break;
		case 3:
			computeCurrentDensityFaceBased<scalar_t, 3>(pos, s_block, s_domain,
				epotField, uCrossBField, jOut);
			break;
		default: break;
		}
	)
}

template <typename scalar_t>
void _ComputeCurrentDensityFaceBased(std::shared_ptr<Domain> domain,
		const torch::Tensor &epotField, const torch::Tensor &uCrossBField,
		torch::Tensor &jOut){

	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)

	BEGIN_SAMPLE;
	k_computeCurrentDensityFaceBased<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
			epotField.data_ptr<scalar_t>(),
			uCrossBField.data_ptr<scalar_t>(),
			jOut.data_ptr<scalar_t>(),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			epotField.numel()/domain->getBatchSize(), uCrossBField.numel()/domain->getBatchSize(), jOut.numel()/domain->getBatchSize()
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Current Density Face-Based");
}

torch::Tensor ComputeCurrentDensityFaceBased(std::shared_ptr<Domain> domain,
		const torch::Tensor &epotField, const torch::Tensor &uCrossBField){

	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	TORCH_CHECK(epotField.dim()==1, "epotField must be 1D (flat).");
	TORCH_CHECK(epotField.size(0)==(index_t)domain->getTotalSize() * domain->getBatchSize(),
		"epotField size must equal domain totalSize (* batch size).");
	TORCH_CHECK(uCrossBField.dim()==1, "uCrossBField must be 1D (flat).");
	// For 2D domains, uCrossBField may have 3 components (including out-of-plane z) to support
	// cases where B lies in the 2D plane (e.g. Hartmann with e_b=[0,1,0]).
	TORCH_CHECK(uCrossBField.size(0)==(index_t)(domain->getTotalSize() * domain->getSpatialDims() * domain->getBatchSize())
		|| (domain->getSpatialDims()==2 && uCrossBField.size(0)==(index_t)(domain->getTotalSize() * 3 * domain->getBatchSize())),
		"uCrossBField size must be totalSize * spatialDims (or totalSize*3 for 2D domains).");
	TORCH_CHECK(epotField.device()==domain->getDevice(), "epotField must be on the same device as the domain.");
	TORCH_CHECK(uCrossBField.device()==domain->getDevice(), "uCrossBField must be on the same device as the domain.");
	TORCH_CHECK(epotField.scalar_type()==domain->getDtype(), "epotField dtype must match domain dtype.");
	TORCH_CHECK(uCrossBField.scalar_type()==domain->getDtype(), "uCrossBField dtype must match domain dtype.");
	CHECK_CONTIGUOUS(epotField);
	CHECK_CONTIGUOUS(uCrossBField);

	// Output: J always has 3 components (flat [totalSize * 3])
	torch::Tensor jOut = torch::zeros(
		{(index_t)domain->getTotalSize() * 3 * domain->getBatchSize()},
		torch::TensorOptions().dtype(domain->getDtype()).device(domain->getDevice())
	);

	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "ComputeCurrentDensityFaceBased", ([&] {
		_ComputeCurrentDensityFaceBased<scalar_t>(domain, epotField, uCrossBField, jOut);
	}));

	return jOut;
}

/* ==================== end MHD kernels ==================== */

#ifdef WITH_GRAD

// ============================================================
// ComputeEpotRHSGrad
// ============================================================

template <typename scalar_t>
__global__ void k_computeEpotRHSGrad(
		DomainGPU<scalar_t> *p_domain,
		const scalar_t *gradDivergence,
		scalar_t *gradVectorField,
		const index_t *p_blockIdxByThreadBlock,
		const index_t *p_threadBlockOffsetInBlock,
		const index_t numThreadBlocks, const index_t divBatchStride, const index_t vectorBatchStride){
	// batched environments: flat per-environment slices
	gradDivergence += blockIdx.y * divBatchStride;
	gradVectorField += blockIdx.y * vectorBatchStride;

	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,

		const I4 pos = unflattenIndex(flatPos, s_block);
		index_t flatPosGlobal = flattenIndexGlobal(pos, s_block, s_domain);

		scalar_t divGrad = gradDivergence[flatPosGlobal];

		scalar_t fluxesGrad[6];
		for(index_t dim = 0; dim < s_domain.numDims; ++dim){
			fluxesGrad[dim*2+1] =  divGrad;
			fluxesGrad[dim*2]   = -divGrad;
		}

		ScatterFieldDivGradNDLoop(pos, fluxesGrad, s_block, s_domain, gradVectorField);

		// Adjoint of the Dirichlet term div -= coef*g: each face cell belongs to this cell
		// only, and the gradient holds one slice per environment, so no atomics are needed.
		// A prescribed current I enters the divergence as -I through a lower (flux +I) and
		// an upper face (flux -I) alike, see computeEpotFluxesNDLoop.
		for(index_t bound=0; bound<(s_domain.numDims*2); ++bound){
			if(!isAtBound(pos, bound, &s_block) || !isEmptyBound(bound, s_block.boundaries)) continue;
			const bool isDirichlet = isEpotDirichletBound(pos, bound, s_block.boundaries);
			const bool isCurrent = isEpotCurrentBound(pos, bound, s_block.boundaries);
			if(!isDirichlet && !isCurrent) continue;
			scalar_t *gradG = potentialValueGradPtr(pos, bound, s_block.boundaries);
			if(gradG == nullptr) continue;
			if(isCurrent){
				*gradG -= divGrad;
				continue;
			}
			I4 tempPos = pos;
			tempPos.w = axisFromBound(bound);
			*gradG -= getLaplaceCoefficientOrthogonalDimSwitch(tempPos, &s_block, s_domain.numDims) * divGrad;
		}
	)
}

template <typename scalar_t>
void _ComputeEpotRHSGrad(std::shared_ptr<Domain> domain,
		const torch::Tensor &gradDivergence, torch::Tensor &gradVectorField){

	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)

	BEGIN_SAMPLE;
	k_computeEpotRHSGrad<scalar_t><<<blocks, threads>>>(
		reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
		gradDivergence.data_ptr<scalar_t>(),
		gradVectorField.data_ptr<scalar_t>(),
		p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
		gradDivergence.numel()/domain->getBatchSize(), gradVectorField.numel()/domain->getBatchSize()
	);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("ComputeEpotRHSGrad");
}

torch::Tensor ComputeEpotRHSGrad(std::shared_ptr<Domain> domain,
		const torch::Tensor &gradDivergence){

	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");

	const int64_t numDims = static_cast<int64_t>(domain->getSpatialDims());
	const int64_t batchSize = static_cast<int64_t>(domain->getBatchSize());
	TORCH_CHECK(gradDivergence.numel()==static_cast<int64_t>(domain->getTotalSize())*batchSize,
		"gradDivergence size must equal domain totalSize (* batch size).");
	torch::Tensor gradVectorField = torch::zeros(
		{batchSize * static_cast<int64_t>(domain->getTotalSize()) * numDims},
		gradDivergence.options()
	);
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "ComputeEpotRHSGrad", ([&] {
		_ComputeEpotRHSGrad<scalar_t>(domain, gradDivergence, gradVectorField);
	}));
	return gradVectorField;
}

// ============================================================
// ComputeCurrentDensityFaceBasedGrad
// ============================================================

template <typename scalar_t, int DIMS>
__device__ void computeCurrentDensityFaceBasedGrad(
		const I4 pos,
		const BlockGPU<scalar_t> &block,
		const DomainGPU<scalar_t> &domain,
		const scalar_t *epotField,
		const scalar_t *uCrossBField,
		const scalar_t *gradJ,
		scalar_t *gradEpot,
		scalar_t *gradUcb)
{
	I4 pos0 = pos;
	pos0.w = 0;
	const index_t flatP = flattenIndex(pos0, block);
	const index_t globalP = flatP + block.globalOffset;
	const index_t numCells = domain.numCells;

	// Load incoming gradient of J (physical, 3 components stored as [J0..., J1..., J2...])
	scalar_t gJ[3];
	gJ[0] = gradJ[globalP];
	gJ[1] = gradJ[globalP + numCells];
	gJ[2] = gradJ[globalP + numCells * 2];

	// Step 2A: grad_j_avg = M^T * gJ / det  (adjoint of J = M * j_avg / det)
	scalar_t grad_j_avg[3] = {0, 0, 0};
	if(block.hasTransform){
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(block.transform) + flatP;
		const scalar_t inv_det = static_cast<scalar_t>(1) / T->det;
		// M^T multiply: grad_j_avg[k] = sum_i M[i][k] * gJ[i] * inv_det
		for(index_t k = 0; k < DIMS; ++k){
			for(index_t i = 0; i < DIMS; ++i){
				grad_j_avg[k] += T->M.v[i].a[k] * gJ[i];
			}
			grad_j_avg[k] *= inv_det;
		}
	} else {
		for(index_t k = 0; k < DIMS; ++k) grad_j_avg[k] = gJ[k];
	}

	// Step 2B: grad_j_face[bound] = 0.5 * grad_j_avg[dim]  (adjoint of face averaging)
	scalar_t grad_j_face[6] = {0};
	for(index_t d = 0; d < DIMS; ++d){
		grad_j_face[d*2]   = static_cast<scalar_t>(0.5) * grad_j_avg[d];
		grad_j_face[d*2+1] = static_cast<scalar_t>(0.5) * grad_j_avg[d];
	}

	// Recompute needed forward quantities (alpha, cvel) for each face
	scalar_t alphaP[3] = {0};
	for(index_t d = 0; d < DIMS; ++d){
		I4 tp = pos;
		tp.w = d;
		alphaP[d] = getLaplaceCoefficientOrthogonal<scalar_t, DIMS>(tp, &block);
	}

	// Steps 2C+2D: loop over faces, scatter gradients to epot and ucb
	for(index_t bound = 0; bound < DIMS * 2; ++bound){
		const index_t dim = axisFromBound(bound);
		const index_t isUpper = boundIsUpper(bound);
		const index_t faceSign = faceSignFromBound(bound);
		const bool atBound = isAtBound(pos, bound, &block);
		const bool atPrescribedBound = atBound && isEmptyBound(bound, block.boundaries);

		if(atPrescribedBound){
			// Forward at a solid wall: j_face[bound] = 0, a constant with no dependence on
			// either phi or (u×B) -> nothing to scatter.
			if(isInsulatingWallBound(pos, bound, block.boundaries)) continue;
			// Forward at a prescribed current: j_face[bound] = -faceSign*I, independent of
			// phi and (u×B); only the value I gets a gradient (this cell's face cell only)
			if(isEpotCurrentBound(pos, bound, block.boundaries)){
				scalar_t *gradG = potentialValueGradPtr(pos, bound, block.boundaries);
				if(gradG != nullptr) *gradG -= static_cast<scalar_t>(faceSign) * grad_j_face[bound];
				continue;
			}
			// Forward at a φ=0 Dirichlet bound carries an extra alpha*faceSign*phi_P term
			// (ghost cell), which depends on this cell's own phi only.
			if(isEpotDirichletBound(pos, bound, block.boundaries)){
				const scalar_t gradPhiTerm = alphaP[dim] * static_cast<scalar_t>(faceSign) * grad_j_face[bound];
				atomicAdd(gradEpot + globalP, gradPhiTerm);
				// j_b = alpha*faceSign*(φ_P - g) + (u×B)_n: g enters with the opposite sign.
				// The face cell belongs to this cell only (see k_computeEpotRHSGrad).
				scalar_t *gradG = potentialValueGradPtr(pos, bound, block.boundaries);
				if(gradG != nullptr) *gradG -= gradPhiTerm;
			}
			// Forward at an open bound: j_face[bound] = cvelP[dim] (∂φ/∂n=0, no phi
			// dependence). Scatter the full-weight gradient to the cell-centre (u×B)_n.
			I4 velPosP = pos;
			velPosP.w = dim;
			scatterContravariantComponentGrad<scalar_t, DIMS>(
				grad_j_face[bound], velPosP, &block, domain, gradUcb);
			continue;
		}

		const scalar_t gf = grad_j_face[bound];

		// Determine neighbor and its alpha for coefficient reconstruction
		scalar_t alphaN = 0;

		if(atBound){
			switch(block.boundaries[bound].type){
			case BoundaryType::CONNECTED_GRID:
			{
				const BlockGPU<scalar_t> *p_connBlock = domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
				I4 tempPos = pos;
				tempPos.w = dim;
				I4 otherPos = computeConnectedPosWithChannel(tempPos, dim, &block.boundaries[bound].cb, domain);
				const index_t connAxis = otherPos.w;
				alphaN = getLaplaceCoefficientOrthogonal<scalar_t, DIMS>(otherPos, p_connBlock);

				// Step 2C: gradient w.r.t. phi
				const scalar_t coeff = static_cast<scalar_t>(0.5) * (alphaP[dim] + alphaN);
				const scalar_t weight = coeff * static_cast<scalar_t>(faceSign) * gf;
				atomicAdd(gradEpot + globalP, weight);
				I4 otherPos0 = otherPos;
				otherPos0.w = 0;
				atomicAdd(gradEpot + flattenIndex(otherPos0, p_connBlock) + p_connBlock->globalOffset, -weight);

				// Step 2D: gradient w.r.t. ucb (cvelP contribution)
				I4 velPosP = pos;
				velPosP.w = dim;
				scatterContravariantComponentGrad<scalar_t, DIMS>(
					static_cast<scalar_t>(0.5) * gf, velPosP, &block, domain, gradUcb);

				// Step 2D: gradient w.r.t. ucb (cvelN contribution)
				I4 velPosN = otherPos;
				velPosN.w = connAxis;
				const bool otherIsUpper = boundIsUpper(block.boundaries[bound].cb.axes.a[0]);
				const scalar_t sign = (otherIsUpper == isUpper) ? static_cast<scalar_t>(-1) : static_cast<scalar_t>(1);
				scatterContravariantComponentGrad<scalar_t, DIMS>(
					static_cast<scalar_t>(0.5) * gf * sign, velPosN, p_connBlock, domain, gradUcb);
				break;
			}
			case BoundaryType::PERIODIC:
			{
				I4 tempPos = pos;
				tempPos.w = dim;
				tempPos.a[dim] = isUpper ? 0 : block.size.a[dim] - 1;
				alphaN = getLaplaceCoefficientOrthogonal<scalar_t, DIMS>(tempPos, &block);

				const scalar_t coeff = static_cast<scalar_t>(0.5) * (alphaP[dim] + alphaN);
				const scalar_t weight = coeff * static_cast<scalar_t>(faceSign) * gf;
				atomicAdd(gradEpot + globalP, weight);
				I4 tempPos0 = tempPos;
				tempPos0.w = 0;
				atomicAdd(gradEpot + flattenIndex(tempPos0, block) + block.globalOffset, -weight);

				// cvelP
				I4 velPosP = pos;
				velPosP.w = dim;
				scatterContravariantComponentGrad<scalar_t, DIMS>(
					static_cast<scalar_t>(0.5) * gf, velPosP, &block, domain, gradUcb);
				// cvelN
				I4 velPosN = tempPos;
				scatterContravariantComponentGrad<scalar_t, DIMS>(
					static_cast<scalar_t>(0.5) * gf, velPosN, &block, domain, gradUcb);
				break;
			}
			default:
				continue;
			}
		} else {
			I4 tempPos = pos;
			tempPos.w = dim;
			tempPos.a[dim] = pos.a[dim] + faceSign;
			alphaN = getLaplaceCoefficientOrthogonal<scalar_t, DIMS>(tempPos, &block);

			const scalar_t coeff = static_cast<scalar_t>(0.5) * (alphaP[dim] + alphaN);
			const scalar_t weight = coeff * static_cast<scalar_t>(faceSign) * gf;
			atomicAdd(gradEpot + globalP, weight);

			I4 tempPos0 = tempPos;
			tempPos0.w = 0;
			atomicAdd(gradEpot + flattenIndex(tempPos0, block) + block.globalOffset, -weight);

			// cvelP
			I4 velPosP = pos;
			velPosP.w = dim;
			scatterContravariantComponentGrad<scalar_t, DIMS>(
				static_cast<scalar_t>(0.5) * gf, velPosP, &block, domain, gradUcb);
			// cvelN
			scatterContravariantComponentGrad<scalar_t, DIMS>(
				static_cast<scalar_t>(0.5) * gf, tempPos, &block, domain, gradUcb);
		}
	}

	// Special case for 2D out-of-plane component: J[2] = uCrossBField[globalP + numCells*2]
	if constexpr (DIMS == 2){
		atomicAdd(gradUcb + globalP + numCells * 2, gJ[2]);
	}
}

template <typename scalar_t>
__global__ void k_computeCurrentDensityFaceBasedGrad(
		DomainGPU<scalar_t> *p_domain,
		const scalar_t *epotField,
		const scalar_t *uCrossBField,
		const scalar_t *gradJ,
		scalar_t *gradEpot,
		scalar_t *gradUcb,
		const index_t *p_blockIdxByThreadBlock,
		const index_t *p_threadBlockOffsetInBlock,
		const index_t numThreadBlocks, const index_t epotBatchStride, const index_t uCrossBBatchStride, const index_t jBatchStride){
	// batched environments: flat per-environment slices
	epotField += blockIdx.y * epotBatchStride;
	gradEpot += blockIdx.y * epotBatchStride;
	uCrossBField += blockIdx.y * uCrossBBatchStride;
	gradUcb += blockIdx.y * uCrossBBatchStride;
	gradJ += blockIdx.y * jBatchStride;

	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,

		I4 pos = unflattenIndex(flatPos, s_block);
		switch(s_domain.numDims){
		case 2:
			computeCurrentDensityFaceBasedGrad<scalar_t, 2>(pos, s_block, s_domain,
				epotField, uCrossBField, gradJ, gradEpot, gradUcb);
			break;
		case 3:
			computeCurrentDensityFaceBasedGrad<scalar_t, 3>(pos, s_block, s_domain,
				epotField, uCrossBField, gradJ, gradEpot, gradUcb);
			break;
		default: break;
		}
	)
}

template <typename scalar_t>
void _ComputeCurrentDensityFaceBasedGrad(std::shared_ptr<Domain> domain,
		const torch::Tensor &epotField, const torch::Tensor &uCrossBField,
		const torch::Tensor &gradJ,
		torch::Tensor &gradEpot, torch::Tensor &gradUcb){

	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)

	BEGIN_SAMPLE;
	k_computeCurrentDensityFaceBasedGrad<scalar_t><<<blocks, threads>>>(
		reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
		epotField.data_ptr<scalar_t>(),
		uCrossBField.data_ptr<scalar_t>(),
		gradJ.data_ptr<scalar_t>(),
		gradEpot.data_ptr<scalar_t>(),
		gradUcb.data_ptr<scalar_t>(),
		p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
		epotField.numel()/domain->getBatchSize(), uCrossBField.numel()/domain->getBatchSize(), gradJ.numel()/domain->getBatchSize()
	);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("ComputeCurrentDensityFaceBasedGrad");
}

std::pair<torch::Tensor, torch::Tensor> ComputeCurrentDensityFaceBasedGrad(
		std::shared_ptr<Domain> domain,
		const torch::Tensor &epotField,
		const torch::Tensor &uCrossBField,
		const torch::Tensor &gradJ){

	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");

	const int64_t batchSize = static_cast<int64_t>(domain->getBatchSize());
	TORCH_CHECK(epotField.numel()==static_cast<int64_t>(domain->getTotalSize())*batchSize,
		"epotField size must equal domain totalSize (* batch size).");
	TORCH_CHECK(uCrossBField.numel()%batchSize==0 && gradJ.numel()%batchSize==0,
		"uCrossBField and gradJ must hold the batched environments.");
	torch::Tensor gradEpot = torch::zeros_like(epotField);
	torch::Tensor gradUcb  = torch::zeros_like(uCrossBField);
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "ComputeCurrentDensityFaceBasedGrad", ([&] {
		_ComputeCurrentDensityFaceBasedGrad<scalar_t>(domain, epotField, uCrossBField, gradJ, gradEpot, gradUcb);
	}));
	return {gradEpot, gradUcb};
}

#endif //WITH_GRAD

