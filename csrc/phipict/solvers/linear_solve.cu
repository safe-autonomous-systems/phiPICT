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
// SolveLinear (dispatch to the fused Krylov or the legacy solvers) and SparseOuterProduct.

#include "piso/device/common.cuh"

/* --- linear solve --- */

template <typename scalar_t>
solverReturn_t _SolveLinear(std::shared_ptr<CSRmatrix> A, torch::Tensor RHS, torch::Tensor x, torch::Tensor maxit, torch::Tensor tol, const ConvergenceCriterion conv,
		const bool useBiCG, const bool matrixRankDeficient, const index_t residualResetSteps, const bool transposeA, const bool printResidual, const bool returnBestResult,
		const bool withPreconditioner){
	
	//CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	//BEGIN_SAMPLE;
	
	const index_t n = A->getRows();
	const index_t nnz = A->getNnz();
	const index_t nBatches = RHS.size(0)/n;
	// batched environments: one matrix per environment (same pattern), stored as
	// consecutive value arrays; right-hand side k uses matrix k / rhsPerMatrix
	const index_t numMatrices = A->getBatchSize();
	TORCH_CHECK(nBatches % numMatrices == 0, "Number of right-hand sides (" + std::to_string(nBatches)
		+ ") must be a multiple of the number of matrices (" + std::to_string(numMatrices) + ").");
	const index_t rhsPerMatrix = nBatches / numMatrices;
	const index_t valStride = numMatrices > 1 ? nnz : 0;
	const bool perBatchTol = tol.size(0) > 1;
	
	solverReturn_t ret;
	
	// Fused, sync-free solvers (krylov/krylov.cu). They cover everything the simulation
	// uses; the legacy cuBLAS/cuSPARSE solvers remain for ILU-preconditioned BiCGStab,
	// the ABS_* convergence criteria, the rank-deficient CG mode, and residual printing.
	const bool krylovSupported = krylov::settings().enabled && !printResidual && !matrixRankDeficient
		&& (conv==ConvergenceCriterion::NORM2 || conv==ConvergenceCriterion::NORM2_NORMALIZED)
		&& !(useBiCG && withPreconditioner);
	if(krylovSupported){
		krylov::Options opt;
		opt.maxit = maxit.data_ptr<index_t>()[0];
		opt.tol = tol.data_ptr<scalar_t>()[0];
		if(perBatchTol){
			opt.tols.resize(nBatches);
			for(index_t k=0; k<nBatches; ++k) opt.tols[k] = tol.data_ptr<scalar_t>()[k];
		}
		opt.normalized = conv==ConvergenceCriterion::NORM2_NORMALIZED;
		opt.residualResetSteps = residualResetSteps;
		opt.returnBest = returnBestResult;
		opt.rhsPerMatrix = rhsPerMatrix;
		const scalar_t *aVal = A->value.data_ptr<scalar_t>();
		const index_t *aIndex = A->index.data_ptr<index_t>();
		const index_t *aRow = A->row.data_ptr<index_t>();
		if(useBiCG){
			opt.returnBest = false; // not supported by the legacy BiCGStab either
			opt.precond = krylov::settings().bicgPrecond;
			if(transposeA){
				krylov::csrTranspose<scalar_t>(aVal, aIndex, aRow, n, nnz, numMatrices, &aVal, &aIndex, &aRow);
			}
			ret = krylov::bicgstabSolve<scalar_t>(aVal, aIndex, aRow, n, nnz, valStride, RHS.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), nBatches, opt);
		} else {
			// CG is only valid for symmetric matrices, for which A^T = A
			opt.precond = krylov::settings().cgPrecond;
			ret = krylov::cgSolve<scalar_t>(aVal, aIndex, aRow, n, nnz, valStride, RHS.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), nBatches, opt);
		}
		return ret;
	}
	
	if(numMatrices > 1 || perBatchTol){
		// The legacy solvers take one matrix and one tolerance: solve every right-hand
		// side on its own (sequentially), each with its matrix and tolerance.
		for(index_t k=0; k<nBatches; ++k){
			const index_t m = k / rhsPerMatrix;
			auto Ak = std::make_shared<CSRmatrix>(*A);
			Ak->value = A->value.narrow(0, m*nnz, nnz);
			torch::Tensor tolk = tol.narrow(0, perBatchTol ? k : 0, 1).clone();
			torch::Tensor rhsk = RHS.narrow(0, k*n, n);
			torch::Tensor xk = x.narrow(0, k*n, n);
			solverReturn_t retk = _SolveLinear<scalar_t>(Ak, rhsk, xk, maxit, tolk, conv, useBiCG, matrixRankDeficient, residualResetSteps,
				transposeA, printResidual, returnBestResult, withPreconditioner);
			ret.insert(ret.end(), retk.begin(), retk.end());
		}
		return ret;
	}
	
	if(useBiCG){
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		BEGIN_SAMPLE;
		ret = bicgstabSolveGPU<scalar_t>(A->value.data_ptr<scalar_t>(), A->index.data_ptr<index_t>(), A->row.data_ptr<index_t>(), n, nnz,
		RHS.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), nBatches,
		withPreconditioner,
		maxit.data_ptr<index_t>()[0], tol.data_ptr<scalar_t>()[0], conv, transposeA);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
		END_SAMPLE("SolveLinear_BiCGstab");
	}/*
	else if(matrixRankDeficient){
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		BEGIN_SAMPLE;
		ret = cgSolvePreconGPU<scalar_t>(A->value.data_ptr<scalar_t>(), A->index.data_ptr<index_t>(), A->row.data_ptr<index_t>(),
			n, nnz, RHS.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), nBatches, maxit.data_ptr<index_t>()[0], tol.data_ptr<scalar_t>()[0], conv,
			residualResetSteps, nullptr, transposeA, printResidual, returnBestResult);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
		END_SAMPLE("SolveLinear_CG");
	}*/
	else{
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		BEGIN_SAMPLE;
		ret = cgSolveGPU<scalar_t>(A->value.data_ptr<scalar_t>(), A->index.data_ptr<index_t>(), A->row.data_ptr<index_t>(),
			n, nnz, RHS.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), nBatches, maxit.data_ptr<index_t>()[0], tol.data_ptr<scalar_t>()[0], conv,
			matrixRankDeficient, residualResetSteps, nullptr, transposeA, printResidual, returnBestResult);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
		END_SAMPLE("SolveLinear_CG");
	}
	
	//CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	//END_SAMPLE("SolveLinear");
	return ret;
}
solverReturn_t SolveLinear(std::shared_ptr<CSRmatrix> A, torch::Tensor RHS, torch::Tensor x, torch::Tensor maxit, torch::Tensor tol, const ConvergenceCriterion conv,
		const bool useBiCG, const bool matrixRankDeficient, const index_t residualResetSteps, const bool transposeA, const bool printResidual, const bool returnBestResult,
		const bool withPreconditioner){
	
	const index_t n = A->getRows();
	
	CHECK_INPUT_CUDA(RHS);
	TORCH_CHECK(RHS.dim()==1, "RHS must be a vector.");
	TORCH_CHECK((RHS.size(0)%n)==0, "Size of RHS must be a multiple of the matrix size.");
	TORCH_CHECK(RHS.scalar_type()==A->getDtype(), "Data type of RHS must match A.");
	
	CHECK_INPUT_CUDA(x);
	TORCH_CHECK(x.dim()==1, "x must be a vector.");
	TORCH_CHECK(x.size(0)==RHS.size(0), "Size of x must match RHS.");
	TORCH_CHECK(x.scalar_type()==A->getDtype(), "Data type of x must match A.");
	
	CHECK_INPUT_HOST(maxit);
	TORCH_CHECK(maxit.dim()==1, "maxit must be 1D.");
	TORCH_CHECK(maxit.size(0)==1, "maxit must be a scalar.");
	TORCH_CHECK(maxit.dtype()==torch_kIndex, "Data type of maxit must be int32.");
	
	CHECK_INPUT_HOST(tol);
	TORCH_CHECK(tol.dim()==1, "tol must be 1D.");
	TORCH_CHECK(tol.size(0)==1 || tol.size(0)==RHS.size(0)/n, "tol must be a scalar or one value per right-hand side.");
	TORCH_CHECK(tol.scalar_type()==A->getDtype(), "Data type of tol must match A.");
	
	solverReturn_t ret;

	AT_DISPATCH_FLOATING_TYPES(A->getDtype(), "SolveLinear", ([&] {
		ret = _SolveLinear<scalar_t>(A, RHS, x, maxit, tol, conv, useBiCG, matrixRankDeficient, residualResetSteps, transposeA, printResidual, returnBestResult, withPreconditioner);
	}));

	return ret;
}

template <typename scalar_t>
void _SparseOuterProduct(torch::Tensor &a, torch::Tensor &b, std::shared_ptr<CSRmatrix> out_pattern){
	
	const index_t n = out_pattern->getRows();
	const index_t nnz = out_pattern->getSize();
	
	OuterProductToSparseMatrix(a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), out_pattern->value.data_ptr<scalar_t>(), out_pattern->index.data_ptr<index_t>(), out_pattern->row.data_ptr<index_t>(), n, nnz);
}
void SparseOuterProduct(torch::Tensor &a, torch::Tensor &b, std::shared_ptr<CSRmatrix> out_pattern){
	const index_t n = out_pattern->getRows();
	
	CHECK_INPUT_CUDA(a);
	TORCH_CHECK(a.dim()==1, "a must be a vector.");
	TORCH_CHECK(a.size(0)==n, "Size of a must match matrix size.");
	TORCH_CHECK(a.scalar_type()==out_pattern->getDtype(), "Data type of a must match C.");
	
	CHECK_INPUT_CUDA(b);
	TORCH_CHECK(b.dim()==1, "b must be a vector.");
	TORCH_CHECK(b.size(0)==n, "Size of b must match matrix size.");
	TORCH_CHECK(b.scalar_type()==out_pattern->getDtype(), "Data type of b must match C.");

	AT_DISPATCH_FLOATING_TYPES(out_pattern->getDtype(), "SparseOuterProduct", ([&] {
		_SparseOuterProduct<scalar_t>(a, b, out_pattern);
	}));
}

