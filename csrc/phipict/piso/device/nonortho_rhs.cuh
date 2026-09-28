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
// Non-orthogonal Laplace RHS contributions and their gradients.

#pragma once

#include "piso/device/block_data.cuh"

/**
 * Face-based
 * assumes the Laplace term is added/positive on the LHS
 */
template <typename scalar_t, index_t DIMS>
__device__
scalar_t getNonOrthoLaplaceRHS_v2(const I4 &pos, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain,
		const int8_t nonOrthoFlags, const GridDataType type, const bool withViscosity, const bool withA, const bool useFaceTransform){
	
	if(!((nonOrthoFlags & NON_ORTHO_DIRECT_RHS) || (nonOrthoFlags & NON_ORTHO_DIAGONAL_RHS))) { return 0;}
	
	scalar_t S = 0;
	
	const bool forPassiveScalar = gridDataTypeToBaseType(type)==GridDataType::PASSIVE_SCALAR;
	
	const Vector<scalar_t, getLaplaceCoefficientsFullLength(DIMS)> alphasP = getLaplaceCoefficientsFull<scalar_t, DIMS>(pos, &block);
	I4 posScalar = pos;
	posScalar.w = 0;
	const index_t flatPosScalarGlobal = flattenIndexGlobal(posScalar, block, domain); //block.globalOffset + flattenIndex(pos, block);
	const scalar_t one = static_cast<scalar_t>(1.0);
	const scalar_t raP = (withA ? one /domain.Adiag[flatPosScalarGlobal] : one) * (withViscosity ? getViscosityBlock<scalar_t>(pos, &block, domain, forPassiveScalar, pos.w) : one);
	

	for(index_t face=0; face<(domain.numDims*2); ++face){
		const index_t axis = axisFromBound(face);
		const bool isUpper = boundIsUpper(face);
		const index_t faceSign = faceSignFromBound(face);

		const bool atBound = isAtBound(pos, face, &block); //isUpper ? pos.a[axis]==(block.size.a[axis]-1) : pos.a[axis]==0; // face is at a boundary
		const bool prescribedBound = atBound && isEmptyBound(face, block.boundaries);

		if(prescribedBound){
			// pressure boundaries are fixed to grad=0
			//if(gridDataTypeToBaseType(type)==GridDataType::PRESSURE){ continue; }
			if(getFixedBoundaryType(pos, face, &block, type)==BoundaryConditionType::NEUMANN || gridDataTypeToBaseType(type)==GridDataType::PRESSURE){ continue;}
			
			Vector<scalar_t, DIMS> boundAlpha = {.a={0}};
			if(useFaceTransform){
				boundAlpha = getLaplaceCoefficientsSingleFace<scalar_t, DIMS>(pos, face, &block);
			}else if(block.boundaries[face].type==BoundaryType::DIRICHLET_VARYING){
				boundAlpha = getLaplaceCoefficientsNeighbourBoundary<scalar_t, DIMS, VaryingDirichletBoundaryGPU<scalar_t>>(pos, axis, &(block.boundaries[face].vdb));
			}else if(block.boundaries[face].type==BoundaryType::FIXED){
				boundAlpha = getLaplaceCoefficientsNeighbourBoundary<scalar_t, DIMS, FixedBoundaryGPU<scalar_t>>(pos, axis, &(block.boundaries[face].fb));
			}
			//Adiag does not exist at bounds, but is only needed for pressure which doesn't reach here.
			
			for(index_t dim=1; dim<DIMS; ++dim){
				const index_t tAxis = (axis+dim)%DIMS;
				// check both ends of the tangent axis
				const bool tLowerAtBound = pos.a[tAxis]==0;
				const bool tUpperAtBound = pos.a[tAxis]==(block.size.a[tAxis]-1);
				// due to minimum resolution the cell can't be at 2 opposing boundaries.
				
				switch(block.boundaries[face].type){
					case BoundaryType::VALUE:
						// gradient along boundary is 0, unless this cell is also at a connected border in tangent direction
						if(tUpperAtBound || tLowerAtBound){
							// TODO: check tangential boundaries?
						} else {
							// gradient is 0, noting to add.
						}
						break;
					case BoundaryType::DIRICHLET_VARYING:
					case BoundaryType::FIXED:
						// gradient along boundary may be non-zero.
					{
						// TODO: check possible connected boundaries at tangential faces?
						I4 lowerPos = pos;
						I4 upperPos = pos;
						scalar_t distanceFactor = 0.5;
						if(!tLowerAtBound) {
							lowerPos.a[tAxis] -= 1;
						}
						if(!tUpperAtBound) {
							upperPos.a[tAxis] += 1;
						}
						if(tLowerAtBound || tUpperAtBound) {
							distanceFactor = 1.0; // one-sided difference if one side is not available for central difference.
						}
						const scalar_t tDataGrad = distanceFactor * (getFixedBoundaryData(upperPos, face, &block, domain, type) - getFixedBoundaryData(lowerPos, face, &block, domain, type));
						const scalar_t boundViscosity = withViscosity ? getViscosityBlock<scalar_t>(pos, &block, domain, forPassiveScalar, pos.w) : one;
						S -= faceSign * boundAlpha.a[tAxis] * tDataGrad * boundViscosity *
							(block.boundaries[face].type==BoundaryType::DIRICHLET_VARYING ? (1 - block.boundaries[face].vdb.slip) : 1 ); // * domain.viscosity
						break;
					}
					case BoundaryType::GRADIENT:
						// gradient is given, but this boundary is not (yet) supported.
						break;
					default:
						break;
				}
			}
			continue; // next face
		} else {
			
			// face is not prescribed, neighbor cell exists (but may be over a block connection)
			
			const NeighborCellInfo<scalar_t> neighborInfo = resolveNeighborCell<scalar_t>(pos, face, &block, domain);
			Vector<scalar_t, DIMS> alphasN = {.a={0}};
			if(useFaceTransform){
				alphasN = getLaplaceCoefficientsSingleFace<scalar_t, DIMS>(pos, face, &block);
			}else{
				alphasN = getLaplaceCoefficientsSingleAxisNeighbor<scalar_t, DIMS>(neighborInfo, face, domain);
			}
			
			const index_t flatPosGlobalN = flattenIndexGlobal(neighborInfo.cell, domain);
			const scalar_t raN = (withA ? one /domain.Adiag[flatPosGlobalN] : one)
				* (withViscosity ? getViscosityBlock<scalar_t>(neighborInfo.cell.pos, neighborInfo.cell.p_block, domain, forPassiveScalar, pos.w) : one);
			
			
			for(index_t dim=1; dim<DIMS; ++dim){
				const index_t tAxis = (axis+dim)%DIMS;
				scalar_t faceAlpha = 0;
				if(useFaceTransform){
					faceAlpha = alphasN.a[tAxis]*(raP + raN)*0.5;
				}else{
					faceAlpha = (alphasP.a[getLaplaceCoefficientCenterIndex(axis, tAxis)]*raP + alphasN.a[tAxis]*raN )*0.5;
				}
				//const scalar_t faceAlpha = getInterpolatedNonOrthoLaplaceComponent(alphaInterp, face, tAxis, DIMS);
				scalar_t tDataGrad = 0;
				for(index_t tIsUpper=0; tIsUpper<2; ++tIsUpper){
					const index_t tFace = axisToBound(tAxis, tIsUpper);//(tAxis<<1) + tIsUpper;
					const index_t tFaceSign = faceSignFromBound(tFace);
					//const bool tAtBound = isAtBound(pos, tFace, &block);
					
					CornerValue<scalar_t> cVal = getCornerValue(pos, face, tFace,
						false, nonOrthoFlags & NON_ORTHO_DIRECT_RHS, 2,
						block, domain, type);
					const bool cornerAtBound = cVal.numCells==0;
					//const bool boundIsGradient = gridDataTypeToBaseType(type)==GridDataType::PRESSURE; // TODO: get from bound with FIXED boundary implementation
					const bool boundIsGradient = cVal.boundType==BoundaryConditionType::NEUMANN || gridDataTypeToBaseType(type)==GridDataType::PRESSURE;
					
					if(cornerAtBound && boundIsGradient){
						// the influence of this is ~0 if the grid is orthogonal at the boundary.
						// simple handling: ignore the gradient boundary condition and use one sided difference from other side (other side can't be at boundary).
						// other face adds corner interpolation by default, so this is calculated to turn that into the one-sided difference.
						//const scalar_t gradScale = 0.5;
						const index_t tOtherFace = invertBound(tFace);
						const index_t tOtherFaceSign = -tFaceSign;
						if(nonOrthoFlags & NON_ORTHO_DIRECT_RHS){
							tDataGrad += tFaceSign * getBlockDataNeighbor(pos, face, block, domain, type, false) * 0.75;
							tDataGrad += tOtherFaceSign * getBlockDataNeighbor(pos, tOtherFace, block, domain, type, false) * 0.25;
						}
						if(nonOrthoFlags & NON_ORTHO_DIAGONAL_RHS){
							tDataGrad += tOtherFaceSign * getBlockDataNeighborDiagonal(pos, face, tOtherFace, block, domain, type, false) * 0.25;
						}
					} else {
						tDataGrad += tFaceSign * cVal.data; // TODO *0.5?
					}
				}
				
				S -= faceSign * faceAlpha * tDataGrad; // * domain.viscosity;
			}
			
		}
	}
	
	return S;
}
template <typename scalar_t>
__device__
scalar_t getNonOrthoLaplaceRHSDimSwitch_v2(const I4 &pos, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain,
		const int8_t nonOrthoFlags, const GridDataType type, const bool withViscosity, const bool withA, const bool useFaceTransform){
	switch(domain.numDims){
	//case 1:
		//return 0; // 1D can't be non-orthogonal
	case 2:
		return getNonOrthoLaplaceRHS_v2<scalar_t, 2>(pos, block, domain, nonOrthoFlags, type, withViscosity, withA, useFaceTransform);
	case 3:
		return getNonOrthoLaplaceRHS_v2<scalar_t, 3>(pos, block, domain, nonOrthoFlags, type, withViscosity, withA, useFaceTransform);
	default:
		return 0;
	}
}

#ifdef WITH_GRAD

/**
 * Face-based
 * assumes the Laplace term is added/positive on the LHS
 */
template <typename scalar_t, index_t DIMS>
__device__
void scatterNonOrthoLaplaceRHS_v2_GRAD(const scalar_t S_grad, const I4 &pos, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain,
		const int8_t nonOrthoFlags, const GridDataType type, const bool withViscosity, const bool withA, const bool useFaceTransform){
	
	if(!((nonOrthoFlags & NON_ORTHO_DIRECT_RHS) || (nonOrthoFlags & NON_ORTHO_DIAGONAL_RHS))) { return;}
	
	const bool forPassiveScalar = gridDataTypeToBaseType(type)==GridDataType::PASSIVE_SCALAR;
	const GridDataType fwdType = gridDataTypeWithoutGrad(type);

	const Vector<scalar_t, getLaplaceCoefficientsFullLength(DIMS)> alphasP = getLaplaceCoefficientsFull<scalar_t, DIMS>(pos, &block);
	I4 posScalar = pos;
	posScalar.w = 0;
	const index_t flatPosScalarGlobal = flattenIndexGlobal(posScalar, block, domain); //block.globalOffset + flattenIndex(pos, block);
	const scalar_t one = static_cast<scalar_t>(1.0);
	const scalar_t raP = (withA ? one /domain.Adiag[flatPosScalarGlobal] : one);
	const scalar_t viscP = (withViscosity ? getViscosityBlock<scalar_t>(pos, &block, domain, forPassiveScalar, pos.w) : one);
	const scalar_t raviscP = raP * viscP;
	const bool withDiffableDataFactor = withViscosity || withA;
	scalar_t raP_grad = 0; //raviscP_grad

	

	for(index_t face=0; face<(domain.numDims*2); ++face){
		const index_t axis = axisFromBound(face);
		const bool isUpper = boundIsUpper(face);
		const index_t faceSign = faceSignFromBound(face);

		const bool atBound = isAtBound(pos, face, &block); //isUpper ? pos.a[axis]==(block.size.a[axis]-1) : pos.a[axis]==0; // face is at a boundary
		const bool prescribedBound = atBound && isEmptyBound(face, block.boundaries);
		
		if(prescribedBound){
			if(block.boundaries[face].type!=BoundaryType::FIXED // only FixedBoundary boundaries are differentiable
					|| getFixedBoundaryType(pos, face, &block, type)==BoundaryConditionType::NEUMANN
					|| gridDataTypeToBaseType(type)==GridDataType::PRESSURE){ // pressure boundaries are fixed to grad=0
				continue;
			}
			
			Vector<scalar_t, DIMS> boundAlpha = {.a={0}};
			if(useFaceTransform){
				boundAlpha = getLaplaceCoefficientsSingleFace<scalar_t, DIMS>(pos, face, &block);
			}else{
				boundAlpha = getLaplaceCoefficientsNeighbourBoundary<scalar_t, DIMS, FixedBoundaryGPU<scalar_t>>(pos, axis, &(block.boundaries[face].fb));
			}
			//Adiag does not exist at bounds, but is only needed for pressure which doesn't reach here.
			
			for(index_t dim=1; dim<DIMS; ++dim){
				const index_t tAxis = (axis+dim)%DIMS;
				// check both ends of the tangent axis
				const bool tLowerAtBound = pos.a[tAxis]==0;
				const bool tUpperAtBound = pos.a[tAxis]==(block.size.a[tAxis]-1);
				// due to minimum resolution the cell can't be at 2 opposing boundaries.
				
				I4 lowerPos = pos;
				I4 upperPos = pos;
				scalar_t distanceFactor = 0.5;
				if(!tLowerAtBound) {
					lowerPos.a[tAxis] -= 1;
				}
				if(!tUpperAtBound) {
					upperPos.a[tAxis] += 1;
				}
				if(tLowerAtBound || tUpperAtBound) {
					distanceFactor = 1.0; // one-sided difference if one side is not available for central difference.
				}
				
				const scalar_t tDataGrad_grad = - faceSign * boundAlpha.a[tAxis] * S_grad * distanceFactor *
					(block.boundaries[face].type==BoundaryType::DIRICHLET_VARYING ? (1 - block.boundaries[face].vdb.slip) : 1 );
				
				if(withViscosity){
					const scalar_t tDataGrad = (getFixedBoundaryData(upperPos, face, &block, domain, fwdType) - getFixedBoundaryData(lowerPos, face, &block, domain, fwdType));
					scatterViscosityBoundary_GRAD<scalar_t>(tDataGrad_grad * tDataGrad, pos, &(block.boundaries[face].fb), &block, domain, forPassiveScalar, pos.w);
				}
				
				const scalar_t boundViscosity = withViscosity ? getViscosityBlock<scalar_t>(pos, &block, domain, forPassiveScalar, pos.w) : one;
				scatterFixedBoundaryData(tDataGrad_grad * boundViscosity, upperPos, face, &block, domain, type);
				scatterFixedBoundaryData(-tDataGrad_grad * boundViscosity, lowerPos, face, &block, domain, type);
			}
			continue; // next face
		} else {
			// face is not prescribed, neighbor cell exists (but may be over a block connection)
			
			const NeighborCellInfo<scalar_t> neighborInfo = resolveNeighborCell<scalar_t>(pos, face, &block, domain);
			Vector<scalar_t, DIMS> alphasN = {.a={0}};
			if(useFaceTransform){
				alphasN = getLaplaceCoefficientsSingleFace<scalar_t, DIMS>(pos, face, &block);
			}else{
				alphasN = getLaplaceCoefficientsSingleAxisNeighbor<scalar_t, DIMS>(neighborInfo, face, domain);
			}
			
			const index_t flatPosGlobalN = flattenIndexGlobal(neighborInfo.cell, domain);
			const scalar_t raN = (withA ? one /domain.Adiag[flatPosGlobalN] : one);
			const scalar_t viscN = (withViscosity ? getViscosityBlock<scalar_t>(neighborInfo.cell.pos, neighborInfo.cell.p_block, domain, forPassiveScalar, pos.w) : one);
			const scalar_t raviscN = raN * viscN;
			scalar_t raN_grad = 0;
			
			for(index_t dim=1; dim<DIMS; ++dim){
				const index_t tAxis = (axis+dim)%DIMS;
				scalar_t faceAlpha = 0;
				if(useFaceTransform){
					faceAlpha = alphasN.a[tAxis]*(raviscP + raviscN)*0.5;
				}else{
					faceAlpha = (alphasP.a[getLaplaceCoefficientCenterIndex(axis, tAxis)]*raviscP + alphasN.a[tAxis]*raviscN )*0.5;
				}
				
				//if(abs(faceAlpha) < static_cast<scalar_t>(1e-5)) { continue; }
				
				// dS / d tDataGrad: S -= faceSign * faceAlpha * tDataGrad
				const scalar_t tDataGrad_grad = -S_grad * faceSign * faceAlpha;
				
				scalar_t tDataGrad = 0;
				for(index_t tIsUpper=0; tIsUpper<2; ++tIsUpper){
					const index_t tFace = axisToBound(tAxis, tIsUpper);//(tAxis<<1) + tIsUpper;
					const index_t tFaceSign = faceSignFromBound(tFace);
					//const bool tAtBound = isAtBound(pos, tFace, &block);
					
					CornerValue<scalar_t> cVal_grad = getCornerValue<scalar_t>(pos, face, tFace,
						false, nonOrthoFlags & NON_ORTHO_DIRECT_RHS, 2,
						block, domain, fwdType); //GridDataType::IS_FIXED_BOUNDARY);
					const bool cornerAtBound = cVal_grad.numCells==0;
					const bool boundIsGradient = cVal_grad.boundType==BoundaryConditionType::NEUMANN || gridDataTypeToBaseType(type)==GridDataType::PRESSURE; // TODO: get from bound with FIXED boundary implementation
					
					if(cornerAtBound && boundIsGradient){
						// the influence of this is ~0 if the grid is orthogonal at the boundary.
						// simple handling: ignore the gradient boundary condition and use one sided difference from other side (other side can't be at boundary).
						// other face adds corner interpolation by default, so this is calculated to turn that into the one-sided difference.
						//const scalar_t gradScale = 0.5;
						const index_t tOtherFace = invertBound(tFace);
						const index_t tOtherFaceSign = -tFaceSign;
						if(nonOrthoFlags & NON_ORTHO_DIRECT_RHS){
							if(withDiffableDataFactor) {tDataGrad += tFaceSign * getBlockDataNeighbor(pos, face, block, domain, fwdType, false) * 0.75;}
							scatterBlockDataNeighbor<scalar_t>(tDataGrad_grad * tFaceSign * static_cast<scalar_t>(0.75), pos, face, block, domain, type, false);
							if(withDiffableDataFactor) {tDataGrad += tOtherFaceSign * getBlockDataNeighbor(pos, tOtherFace, block, domain, fwdType, false) * 0.25;}
							scatterBlockDataNeighbor<scalar_t>(tDataGrad_grad * tOtherFaceSign * static_cast<scalar_t>(0.25), pos, tOtherFace, block, domain, type, false);
						}
						if(nonOrthoFlags & NON_ORTHO_DIAGONAL_RHS){
							if(withDiffableDataFactor) {tDataGrad += tOtherFaceSign * getBlockDataNeighborDiagonal(pos, face, tOtherFace, block, domain, fwdType, false) * 0.25;}
							scatterBlockDataNeighborDiagonal<scalar_t>(tDataGrad_grad * tOtherFaceSign * static_cast<scalar_t>(0.25),
								pos, face, tOtherFace, block, domain, type, false);
						}
					} else {
						// d tDataGrad / d cVal.data: tDataGrad += tFaceSign * cVal.data;
						if(withDiffableDataFactor) {tDataGrad += tFaceSign * cVal_grad.data;}
						//cVal_grad.data = static_cast<scalar_t>(cVal_grad.numCells);
						cVal_grad.data = tDataGrad_grad * tFaceSign;
						scatterCornerValue_GRAD<scalar_t>(cVal_grad, pos, face, tFace,
							false, nonOrthoFlags & NON_ORTHO_DIRECT_RHS, 2,
							block, domain, type);
					}
				}
				
				//S -= faceSign * faceAlpha * tDataGrad;
				if(withDiffableDataFactor){
					const scalar_t faceAlpha_grad = -S_grad * faceSign * tDataGrad;
					if(useFaceTransform){
						//faceAlpha = alphasN.a[tAxis]*(raP + raN)*0.5;
						raP_grad += alphasN.a[tAxis]*0.5 * faceAlpha_grad;
						raN_grad += alphasN.a[tAxis]*0.5 * faceAlpha_grad;
					}else{
						//faceAlpha = (alphasP.a[getLaplaceCoefficientCenterIndex(axis, tAxis)]*raP + alphasN.a[tAxis]*raN )*0.5;
						raP_grad += alphasP.a[getLaplaceCoefficientCenterIndex(axis, tAxis)]*0.5 * faceAlpha_grad;
						raN_grad += alphasN.a[tAxis]*0.5 * faceAlpha_grad;
					}
				}
				
			}
			if(withViscosity){
				scatterViscosityBlock_GRAD<scalar_t>(raN_grad * raN, neighborInfo.cell.pos, neighborInfo.cell.p_block, domain, forPassiveScalar, pos.w);
				//scatterViscosityBlock_GRAD<scalar_t>(1, neighborInfo.cell.pos, neighborInfo.cell.p_block, domain, forPassiveScalar, pos.w);
			}
			if(withA){
				atomicAdd(domain.Adiag_grad + flatPosGlobalN, raN_grad * viscN * (-raN * raN));
			}
		}
	}
	if(withViscosity){
		scatterViscosityBlock_GRAD<scalar_t>(raP_grad * raP, pos, &block, domain, forPassiveScalar, pos.w);
	}
	if(withA){
		atomicAdd(domain.Adiag_grad + flatPosScalarGlobal, raP_grad * viscP * (-raP * raP));
	}
}
template <typename scalar_t>
__device__
void scatterNonOrthoLaplaceRHSDimSwitch_v2_GRAD(const scalar_t data, const I4 &pos, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain,
		const int8_t nonOrthoFlags, const GridDataType type, const bool withViscosity, const bool withA, const bool useFaceTransform){
	switch(domain.numDims){
	//case 1:
		//break; // 1D can't be non-orthogonal
	case 2:
		scatterNonOrthoLaplaceRHS_v2_GRAD<scalar_t, 2>(data, pos, block, domain, nonOrthoFlags, type, withViscosity, withA, useFaceTransform);
		break;
	case 3:
		scatterNonOrthoLaplaceRHS_v2_GRAD<scalar_t, 3>(data, pos, block, domain, nonOrthoFlags, type, withViscosity, withA, useFaceTransform);
		break;
	default:
		break;
	}
}

template<typename scalar_t>
__device__ void scatterPressureGradToWithBounds(const scalar_t pressureGrad, const I4 pos, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain){
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
					atomicAdd(block.pressure_grad + flattenIndex(tempPos, block), pressureGrad);
					return;
				case BoundaryType::CONNECTED_GRID:
				{
					const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
					const I4 otherPos = computeConnectedPos(tempPos, dim, &block.boundaries[bound].cb, domain, -(pos.a[dim]+1));
					atomicAdd(p_connectedBlock->pressure_grad + flattenIndex(otherPos, p_connectedBlock), pressureGrad);
					return;
				}
				case BoundaryType::PERIODIC:
					tempPos.a[dim] += block.size.a[dim];
					atomicAdd(block.pressure_grad + flattenIndex(tempPos, block), pressureGrad);
					return;
				default:
					return;
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
					atomicAdd(block.pressure_grad + flattenIndex(tempPos, block), pressureGrad);
					return;
				case BoundaryType::CONNECTED_GRID:
				{
					const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
					const I4 otherPos = computeConnectedPos(tempPos, dim, &block.boundaries[bound].cb, domain, pos.a[dim]-block.size.a[dim]);
					atomicAdd(p_connectedBlock->pressure_grad + flattenIndex(otherPos, p_connectedBlock), pressureGrad);
					return;
				}
				case BoundaryType::PERIODIC:
					tempPos.a[dim] -= block.size.a[dim];
					atomicAdd(block.pressure_grad + flattenIndex(tempPos, block), pressureGrad);
					return;
				default:
					return;
			}
		}
	}
	atomicAdd(block.pressure_grad + flatPos, pressureGrad);
}

#endif //WITH_GRAD
