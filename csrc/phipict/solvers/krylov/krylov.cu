// Copyright 2026 Jannis Becktepe
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

/*
 * Fused Krylov solvers, see krylov.h for the overview.
 *
 * Kernel structure per iteration (B batches, n rows, "gather" = SpMV input):
 *
 *  CG (Chronopoulos-Gear, preconditioner M = I or diag(A)^-1):
 *    cgSpmvDots : w = A (M r);  gamma=(r,Mr), delta=(w,Mr), rr=(r,r)   | last block: alpha, beta, convergence
 *    cgUpdate   : p = Mr + beta p;  s = w + beta s;  x += alpha p;  r -= alpha s
 *
 *  BiCGStab (van der Vorst, right preconditioned with M):
 *    bicgP      : p = r + beta (p - omega v)
 *    bicgV      : v = A (M p);  (r0,v)                                  | last block: alpha
 *    bicgT      : s = r - alpha v (on the fly); t = A (M s); (t,s),(t,t),(s,s) | last block: omega / half-step convergence
 *    bicgUpdate : x += alpha Mp + omega Ms;  r = s - omega t;  (r0,r),(r,r)  | last block: beta, convergence
 *
 * The "last block" pattern (threadfence + atomic counter) makes the reductions
 * deterministic: partial sums are written per block and summed in a fixed order.
 * The whole iteration loop is captured into a CUDA graph with a conditional WHILE
 * node (CUDA >= 12.4) whose condition is cleared on the device once every batch
 * has finished, so a solve is one graph launch plus one readback of the results.
 */

#include "solvers/krylov/krylov.h"
#include "solvers/krylov/krylov_torch.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <vector>


#if CUDART_VERSION >= 12040
#define KRYLOV_HAS_CONDITIONAL_GRAPHS 1
#else
#define KRYLOV_HAS_CONDITIONAL_GRAPHS 0
#endif

namespace krylov {

static void checkCuda(const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess) return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
	exit(10);
}
static void checkCuda(const char *file, unsigned line, const char *statement, cusparseStatus_t err) {
	if (err == CUSPARSE_STATUS_SUCCESS) return;
	std::cerr << statement << " returned " << cusparseGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
	exit(10);
}
#define KRYLOV_CHECK(value) checkCuda(__FILE__, __LINE__, #value, value)

GlobalSettings& settings() {
	static GlobalSettings s;
	return s;
}

/* --- device state --- */

enum DoneStatus : int {
	RUNNING = 0,
	CONVERGED = 1,
	NONFINITE = 2,
	MAXIT = 3,
	RISING = 4, // residual rising for too many iterations, best iterate is returned
};

struct BatchState {
	double tol;
	double rho;      // CG: gamma of the previous iteration; BiCG: (r0, r)
	double alpha;
	double beta;
	double omega;
	double crit;     // current convergence criterion (normalized or plain 2-norm of r)
	double best;     // lowest criterion so far (returnBest)
	double last;     // criterion of the previous iteration (returnBest)
	double halfCrit; // BiCG: criterion of s
	int it;          // number of completed iterations
	int maxit;
	int bestIt;
	int rising;
	int done;        // DoneStatus
	int restart;     // next iteration starts with beta=0
	int saveBest;    // the update kernel stores x as the best iterate before updating
	int halfDone;    // BiCG: converged after the first half-step
	int usedIt;      // iteration count reported to the caller
	int pad;
};

struct Control {
	unsigned int counter; // blocks finished in the current kernel (last-block pattern)
	int allDone;
	int loops;            // iterations executed by the graph loop (safety bound)
	int maxLoops;
};

constexpr int BLOCK = 256;
constexpr int NUM_WARPS = BLOCK / 32;
constexpr int RISING_CUTOFF = 100;

/* --- reduction helpers --- */

// Reduces vals[NV] over the block; the result is valid in thread 0.
template <int NV>
__device__ inline void blockReduce(double (&vals)[NV]) {
	__shared__ double smem[NV][NUM_WARPS];
	const int lane = threadIdx.x & 31;
	const int warp = threadIdx.x >> 5;
#pragma unroll
	for (int q = 0; q < NV; ++q) {
		double v = vals[q];
#pragma unroll
		for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
		if (lane == 0) smem[q][warp] = v;
	}
	__syncthreads();
	if (warp == 0) {
#pragma unroll
		for (int q = 0; q < NV; ++q) {
			double v = lane < NUM_WARPS ? smem[q][lane] : 0.0;
#pragma unroll
			for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
			vals[q] = v;
		}
	}
	__syncthreads();
}

// Write this block's partial sums for the batches of this group, then determine whether
// this is the last block of the grid to finish.
template <int NB, int NQ>
__device__ inline bool storePartialsAndCheckLast(double (&acc)[NB][NQ], const int b0, const int nb, double *partial, Control *ctrl) {
	double flat[NB * NQ];
#pragma unroll
	for (int k = 0; k < NB; ++k)
#pragma unroll
		for (int q = 0; q < NQ; ++q) flat[k * NQ + q] = acc[k][q];
	blockReduce<NB * NQ>(flat);

	__shared__ bool isLast;
	if (threadIdx.x == 0) {
		const int G = gridDim.x;
		for (int k = 0; k < nb; ++k)
			for (int q = 0; q < NQ; ++q)
				partial[((b0 + k) * G + blockIdx.x) * NQ + q] = flat[k * NQ + q];
		__threadfence();
		const unsigned int total = gridDim.x * gridDim.y;
		const unsigned int ticket = atomicAdd(&ctrl->counter, 1u);
		isLast = (ticket == total - 1);
	}
	__syncthreads();
	return isLast;
}

// In the last block: sum the partials of batch b in a fixed order. Result valid in thread 0.
template <int NQ>
__device__ inline void reduceBatch(const double *partial, const int b, const int G, double (&out)[NQ]) {
	__threadfence();
#pragma unroll
	for (int q = 0; q < NQ; ++q) out[q] = 0.0;
	for (int g = threadIdx.x; g < G; g += BLOCK) {
		const volatile double *pp = partial + (b * G + g) * NQ;
#pragma unroll
		for (int q = 0; q < NQ; ++q) out[q] += pp[q];
	}
	blockReduce<NQ>(out);
}

__device__ inline void finishLastBlock(BatchState *st, const int B, Control *ctrl) {
	if (threadIdx.x == 0) {
		int nDone = 0;
		for (int b = 0; b < B; ++b) nDone += st[b].done != RUNNING;
		ctrl->allDone = (nDone == B);
		ctrl->counter = 0;
	}
}

__device__ inline bool isFinite(const double v) { return isfinite(v); }

/* --- matrix access --- */

template <typename T>
struct CsrView {
	const T *val;
	const index_t *col;
	const index_t *row;
	index_t n;
	index_t valStride; // 0: shared by all batches
	int rpm;           // right-hand sides per matrix: batch b uses matrix b / rpm
};

/* Batches are processed in groups of up to NB right-hand sides per thread that share one
 * matrix. blockIdx.y enumerates (matrix, group within the matrix). With a shared matrix
 * (rpm = B) this is simply the groups of NB consecutive batches. */
struct BatchGroup {
	int b0; // first batch of the group
	int nb; // batches in the group
	int m;  // matrix index
};
template <int NB>
__device__ inline BatchGroup batchGroup(const int B, const int rpm) {
	const int gpm = (rpm + NB - 1) / NB; // groups per matrix
	BatchGroup g;
	g.m = blockIdx.y / gpm;
	const int j = blockIdx.y - g.m * gpm;
	g.b0 = g.m * rpm + j * NB;
	g.nb = max(0, min(min(NB, rpm - j * NB), B - g.b0));
	return g;
}
// number of block rows (gridDim.y) for B batches with rpm right-hand sides per matrix
static inline int batchGroupRows(const int B, const int rpm, const int NB) {
	const int numMatrices = (B + rpm - 1) / rpm;
	return numMatrices * ((rpm + NB - 1) / NB);
}

// Load a vector entry, optionally preconditioned with the inverse diagonal.
template <typename T, bool PRE>
__device__ inline T loadPre(const T *__restrict__ v, const T *__restrict__ dinv, const index_t i) {
	if (PRE) return v[i] * dinv[i];
	return v[i];
}

/* --- common kernels --- */

template <typename T>
__global__ void kInitState(BatchState *st, const int B, const double tol, const int maxit, Control *ctrl, const int maxLoops) {
	for (int b = threadIdx.x; b < B; b += blockDim.x) {
		BatchState s = {};
		s.tol = tol;
		s.maxit = maxit;
		s.best = INFINITY;
		s.restart = 1;
		s.bestIt = -1;
		st[b] = s;
	}
	if (threadIdx.x == 0) {
		ctrl->counter = 0;
		ctrl->allDone = 0;
		ctrl->loops = 0;
		ctrl->maxLoops = maxLoops;
	}
}

// Inverse diagonal for Jacobi preconditioning. Rows without (non-zero) diagonal get 1.
template <typename T>
__global__ void kInverseDiagonal(const CsrView<T> A, T *dinv, const int Bmat) {
	const index_t n = A.n;
	for (index_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n * Bmat; idx += gridDim.x * blockDim.x) {
		const index_t b = idx / n;
		const index_t i = idx - b * n;
		const T *val = A.val + b * A.valStride;
		T d = 0;
		for (index_t j = A.row[i]; j < A.row[i + 1]; ++j) {
			if (A.col[j] == i) { d = val[j]; break; }
		}
		dinv[idx] = d != T(0) ? T(1) / d : T(1);
	}
}

// r = b - A x for all running batches; reduces (r, r) and (r, D^-1 r) (with dinv) and sets
// rho = (r, M r) and restart for the Krylov recurrences. With copyR0, stores r in r0 (BiCGStab
// shadow residual). Initial call (resetEvery=0): also the initial convergence check.
// With resetEvery>0 it only acts on batches whose iteration count is a positive multiple of it
// (residual replacement inside the graph loop).
template <typename T, int NB>
__global__ void kResidual(const CsrView<T> A, const T *__restrict__ bvec, const T *__restrict__ x, T *__restrict__ r, T *__restrict__ r0,
		const T *__restrict__ dinv, BatchState *st, const int B, double *partial, Control *ctrl, const double normScale, const int resetEvery) {
	const index_t n = A.n;
	const BatchGroup grp = batchGroup<NB>(B, A.rpm);
	const int b0 = grp.b0;
	const int nb = grp.nb;
	auto isActive = [&](const BatchState &s) {
		return s.done == RUNNING && (resetEvery <= 0 || (s.it > 0 && (s.it % resetEvery) == 0));
	};
	bool active[NB];
	bool any = false;
#pragma unroll
	for (int k = 0; k < NB; ++k) { active[k] = k < nb && isActive(st[b0 + k]); any |= active[k]; }
	double acc[NB][2];
#pragma unroll
	for (int k = 0; k < NB; ++k) acc[k][0] = acc[k][1] = 0.0;

	if (any) {
		const T *val = A.val + grp.m * A.valStride;
		const T *dv = dinv ? dinv + (A.valStride ? grp.m * n : 0) : nullptr;
		for (index_t i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += gridDim.x * BLOCK) {
			T sum[NB];
#pragma unroll
			for (int k = 0; k < NB; ++k) sum[k] = 0;
			const index_t end = A.row[i + 1];
			for (index_t j = A.row[i]; j < end; ++j) {
				const index_t c = A.col[j];
				const T a = val[j];
#pragma unroll
				for (int k = 0; k < NB; ++k) if (active[k]) sum[k] += a * x[(b0 + k) * n + c];
			}
#pragma unroll
			for (int k = 0; k < NB; ++k) {
				if (active[k]) {
					const index_t o = (b0 + k) * n + i;
					const T ri = bvec[o] - sum[k];
					r[o] = ri;
					if (r0 != nullptr) r0[o] = ri;
					acc[k][0] += double(ri) * double(ri);
					if (dv) acc[k][1] += double(ri) * double(ri * dv[i]);
				}
			}
		}
	}

	if (storePartialsAndCheckLast<NB, 2>(acc, b0, nb, partial, ctrl)) {
		for (int b = 0; b < B; ++b) {
			if (!isActive(st[b])) continue; // uniform across the block
			double red[2];
			reduceBatch<2>(partial, b, gridDim.x, red);
			if (threadIdx.x == 0) {
				BatchState &s = st[b];
				s.rho = dinv ? red[1] : red[0];
				s.restart = 1;
				if (resetEvery <= 0) {
					const double crit = sqrt(red[0]) * normScale;
					s.crit = crit;
					if (!isFinite(crit)) { s.done = NONFINITE; s.usedIt = 0; }
					// The legacy BiCGStab stops on an already converged start (reports -1).
					// The legacy CG always takes at least one iteration before its first
					// check: stopping here would return the initial guess (x = 0 without a
					// warm start) whenever ||b||/sqrt(n) < tol, e.g. for the channel's small
					// pressure RHS, silently dropping the pressure correction. Only an exactly
					// zero residual, where the first CG step would divide 0 by 0, stops CG here.
					else if (r0 != nullptr ? crit < s.tol : crit == 0.0) {
						s.done = CONVERGED;
						s.usedIt = r0 != nullptr ? -1 : 0;
					}
				}
			}
			__syncthreads();
		}
		finishLastBlock(st, B, ctrl);
	}
}

// Copy the best iterate back into x for batches that did not converge.
template <typename T>
__global__ void kRestoreBest(T *x, const T *xbest, const BatchState *st, const index_t n, const int B) {
	for (index_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n * B; idx += gridDim.x * blockDim.x) {
		const BatchState &s = st[idx / n];
		if ((s.done == MAXIT || s.done == RISING) && s.bestIt >= 0 && s.best < s.crit) x[idx] = xbest[idx];
	}
}

#if KRYLOV_HAS_CONDITIONAL_GRAPHS
__device__ inline void setLoopCondition(cudaGraphConditionalHandle handle, Control *ctrl) {
	const int loops = ++ctrl->loops;
	const bool cont = !ctrl->allDone && loops < ctrl->maxLoops;
	cudaGraphSetConditional(handle, cont ? 1u : 0u);
}
#endif

struct LoopHandle {
#if KRYLOV_HAS_CONDITIONAL_GRAPHS
	cudaGraphConditionalHandle handle;
#else
	unsigned long long handle;
#endif
	bool active;
};

__device__ inline void maybeSetCondition(const LoopHandle loop, Control *ctrl) {
#if KRYLOV_HAS_CONDITIONAL_GRAPHS
	if (loop.active) setLoopCondition(loop.handle, ctrl);
#endif
}

/* --- CG kernels ---
 * Classic (preconditioned) CG with two fused kernels per iteration:
 *   kCgDirection: z = M r;  p' = z + beta p  (written to the other ping-pong buffer, and
 *                 evaluated on the fly at the gathered columns);  q = A p';  (p', q)
 *                 last block: alpha = gamma / (p', q)
 *   kCgUpdate:    x += alpha p';  r -= alpha q;  (r, r), (r, M r)
 *                 last block: convergence, beta = (r, M r) / gamma
 * q is always computed as A p' (no recurrence for A p), which keeps the method as
 * robust as the textbook formulation in single precision. With an explicit
 * preconditioner (AMG V-cycle, PM_VECTOR) the (r, M r) reduction runs in kCgDots after
 * the V-cycle instead.
 */

// Preconditioner modes of the CG kernels
enum PrecondMode : int {
	PM_NONE = 0,   // z = r
	PM_JACOBI = 1, // z = diag(A)^-1 r, computed on the fly
	PM_VECTOR = 2, // z given explicitly (e.g. an AMG V-cycle applied to r)
};

template <typename T, int PM>
__device__ inline T precondValue(const T *__restrict__ r, const T *__restrict__ z, const T *__restrict__ dv, const index_t o, const index_t i) {
	if (PM == PM_VECTOR) return z[o];
	if (PM == PM_JACOBI) return r[o] * dv[i];
	return r[o];
}

template <typename T, int NB, int PM>
__global__ void kCgDirection(const CsrView<T> A, const T *__restrict__ r, const T *__restrict__ z, const T *__restrict__ dinv,
		const T *__restrict__ pIn, T *__restrict__ pOut, T *__restrict__ q, BatchState *st, const int B, double *partial, Control *ctrl) {
	const index_t n = A.n;
	const BatchGroup grp = batchGroup<NB>(B, A.rpm);
	const int b0 = grp.b0;
	const int nb = grp.nb;
	bool active[NB], restart[NB];
	T beta[NB];
	bool any = false;
#pragma unroll
	for (int k = 0; k < NB; ++k) {
		active[k] = k < nb && st[b0 + k].done == RUNNING;
		restart[k] = active[k] && st[b0 + k].restart;
		beta[k] = active[k] ? T(st[b0 + k].beta) : T(0);
		any |= active[k];
	}
	double acc[NB][1];
#pragma unroll
	for (int k = 0; k < NB; ++k) acc[k][0] = 0.0;

	if (any) {
		const T *val = A.val + grp.m * A.valStride;
		const T *dv = PM == PM_JACOBI ? dinv + (A.valStride ? grp.m * n : 0) : nullptr;
		for (index_t i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += gridDim.x * BLOCK) {
			T sum[NB];
#pragma unroll
			for (int k = 0; k < NB; ++k) sum[k] = 0;
			const index_t end = A.row[i + 1];
			for (index_t j = A.row[i]; j < end; ++j) {
				const index_t c = A.col[j];
				const T a = val[j];
#pragma unroll
				for (int k = 0; k < NB; ++k) {
					if (active[k]) {
						const index_t o = (b0 + k) * n + c;
						const T zc = precondValue<T, PM>(r, z, dv, o, c);
						sum[k] += a * (restart[k] ? zc : zc + beta[k] * pIn[o]);
					}
				}
			}
#pragma unroll
			for (int k = 0; k < NB; ++k) {
				if (active[k]) {
					const index_t o = (b0 + k) * n + i;
					const T zi = precondValue<T, PM>(r, z, dv, o, i);
					const T pi = restart[k] ? zi : zi + beta[k] * pIn[o];
					pOut[o] = pi;
					q[o] = sum[k];
					acc[k][0] += double(pi) * double(sum[k]);
				}
			}
		}
	}

	if (storePartialsAndCheckLast<NB, 1>(acc, b0, nb, partial, ctrl)) {
		for (int b = 0; b < B; ++b) {
			if (st[b].done != RUNNING) continue; // uniform across the block
			double red[1];
			reduceBatch<1>(partial, b, gridDim.x, red);
			if (threadIdx.x == 0) {
				BatchState &s = st[b];
				const double alpha = s.rho / red[0]; // rho holds gamma = (r, M r)
				s.restart = 0;
				if (!isFinite(alpha)) { s.done = NONFINITE; s.usedIt = s.it; }
				else s.alpha = alpha;
			}
			__syncthreads();
		}
		finishLastBlock(st, B, ctrl);
	}
}

// x += alpha p; r -= alpha q; (r, r) and (r, M r). Last block: convergence and beta.
template <typename T, int NB, int PM>
__global__ void kCgUpdate(const index_t n, const int rpm, const index_t dinvStride, T *__restrict__ r, const T *__restrict__ p, const T *__restrict__ q,
		T *__restrict__ x, T *__restrict__ xbest, const T *__restrict__ dinv, BatchState *st, const int B, double *partial, Control *ctrl,
		const double normScale, const bool returnBest, const LoopHandle loop) {
	const BatchGroup grp = batchGroup<NB>(B, rpm);
	const int b0 = grp.b0;
	const int nb = grp.nb;
	double acc[NB][2];
#pragma unroll
	for (int k = 0; k < NB; ++k) acc[k][0] = acc[k][1] = 0.0;
#pragma unroll
	for (int k = 0; k < NB; ++k) {
		if (k >= nb) break;
		const BatchState &bs = st[b0 + k];
		if (bs.done != RUNNING) continue;
		const T alpha = T(bs.alpha);
		const bool saveBest = bs.saveBest && xbest != nullptr;
		const index_t off = (b0 + k) * n;
		const T *dv = PM == PM_JACOBI ? dinv + grp.m * dinvStride : nullptr;
		for (index_t i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += gridDim.x * BLOCK) {
			const index_t o = off + i;
			const T xi = x[o];
			if (saveBest) xbest[o] = xi;
			x[o] = xi + alpha * p[o];
			const T ri = r[o] - alpha * q[o];
			r[o] = ri;
			acc[k][0] += double(ri) * double(ri);
			if (PM == PM_JACOBI) acc[k][1] += double(ri) * double(ri * dv[i]);
		}
	}

	if (storePartialsAndCheckLast<NB, 2>(acc, b0, nb, partial, ctrl)) {
		for (int b = 0; b < B; ++b) {
			if (st[b].done != RUNNING) continue;
			double red[2];
			reduceBatch<2>(partial, b, gridDim.x, red);
			if (threadIdx.x == 0) {
				BatchState &s = st[b];
				const double crit = sqrt(red[0]) * normScale;
				++s.it;
				s.crit = crit;
				s.saveBest = 0;
				s.usedIt = s.it - 1;
				if (!isFinite(crit)) {
					s.done = NONFINITE;
				} else {
					if (returnBest) {
						if (crit < s.best) { s.best = crit; s.bestIt = s.it - 1; s.saveBest = 1; }
						if (s.it > 1 && crit >= s.last) ++s.rising; else s.rising = 0;
						s.last = crit;
					}
					if (crit < s.tol) s.done = CONVERGED;
					else if (s.it >= s.maxit) s.done = MAXIT;
					else if (returnBest && s.rising >= RISING_CUTOFF) s.done = RISING;
					else if (PM != PM_VECTOR) {
						const double gamma = PM == PM_JACOBI ? red[1] : red[0];
						s.beta = gamma / s.rho;
						s.rho = gamma;
						if (!isFinite(s.beta)) s.done = NONFINITE;
					}
				}
			}
			__syncthreads();
		}
		finishLastBlock(st, B, ctrl);
		if (threadIdx.x == 0) maybeSetCondition(loop, ctrl);
	}
}

// (r, z) for an explicit preconditioned vector z. Last block: gamma and beta.
template <typename T>
__global__ void kCgDots(const index_t n, const T *__restrict__ r, const T *__restrict__ z, BatchState *st, const int B, double *partial, Control *ctrl) {
	const int b = blockIdx.y;
	double acc[1][1] = {{0.0}};
	if (st[b].done == RUNNING) {
		for (index_t i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += gridDim.x * BLOCK) acc[0][0] += double(r[b * n + i]) * double(z[b * n + i]);
	}
	if (storePartialsAndCheckLast<1, 1>(acc, b, 1, partial, ctrl)) {
		for (int bb = 0; bb < B; ++bb) {
			if (st[bb].done != RUNNING) continue;
			double red[1];
			reduceBatch<1>(partial, bb, gridDim.x, red);
			if (threadIdx.x == 0) {
				BatchState &s = st[bb];
				if (!s.restart) s.beta = red[0] / s.rho;
				s.rho = red[0];
				if (!isFinite(s.rho) || !isFinite(s.beta)) { s.done = NONFINITE; s.usedIt = s.it; }
			}
			__syncthreads();
		}
		finishLastBlock(st, B, ctrl);
	}
}

/* --- BiCGStab kernels --- */

// p = r + beta (p - omega v), or p = r on restart
template <typename T>
__global__ void kBicgP(const index_t n, const T *__restrict__ r, T *__restrict__ p, const T *__restrict__ v, const BatchState *st, const int B) {
	const int b = blockIdx.y;
	const BatchState &bs = st[b];
	if (bs.done != RUNNING) return;
	const bool restart = bs.restart;
	const T beta = T(bs.beta), omega = T(bs.omega);
	const index_t off = b * n;
	for (index_t i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += gridDim.x * BLOCK) {
		const index_t o = off + i;
		p[o] = restart ? r[o] : r[o] + beta * (p[o] - omega * v[o]);
	}
}

// v = A (M p); (r0, v). Last block: alpha = rho / (r0, v)
template <typename T, int NB, bool PRE>
__global__ void kBicgV(const CsrView<T> A, const T *__restrict__ p, T *__restrict__ v, const T *__restrict__ r0, const T *__restrict__ dinv,
		BatchState *st, const int B, double *partial, Control *ctrl) {
	const index_t n = A.n;
	const BatchGroup grp = batchGroup<NB>(B, A.rpm);
	const int b0 = grp.b0;
	const int nb = grp.nb;
	bool active[NB];
	bool any = false;
#pragma unroll
	for (int k = 0; k < NB; ++k) { active[k] = k < nb && st[b0 + k].done == RUNNING; any |= active[k]; }
	double acc[NB][1];
#pragma unroll
	for (int k = 0; k < NB; ++k) acc[k][0] = 0.0;

	if (any) {
		const T *val = A.val + grp.m * A.valStride;
		const T *dv = PRE ? dinv + (A.valStride ? grp.m * n : 0) : nullptr;
		for (index_t i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += gridDim.x * BLOCK) {
			T sum[NB];
#pragma unroll
			for (int k = 0; k < NB; ++k) sum[k] = 0;
			const index_t end = A.row[i + 1];
			for (index_t j = A.row[i]; j < end; ++j) {
				const index_t c = A.col[j];
				const T a = val[j];
				const T dc = PRE ? dv[c] : T(1);
#pragma unroll
				for (int k = 0; k < NB; ++k) if (active[k]) sum[k] += a * (PRE ? p[(b0 + k) * n + c] * dc : p[(b0 + k) * n + c]);
			}
#pragma unroll
			for (int k = 0; k < NB; ++k) {
				if (active[k]) {
					const index_t o = (b0 + k) * n + i;
					v[o] = sum[k];
					acc[k][0] += double(r0[o]) * double(sum[k]);
				}
			}
		}
	}

	if (storePartialsAndCheckLast<NB, 1>(acc, b0, nb, partial, ctrl)) {
		for (int b = 0; b < B; ++b) {
			if (st[b].done != RUNNING) continue;
			double red[1];
			reduceBatch<1>(partial, b, gridDim.x, red);
			if (threadIdx.x == 0) {
				BatchState &s = st[b];
				const double alpha = s.rho / red[0];
				s.restart = 0;
				if (!isFinite(alpha)) { s.done = NONFINITE; s.usedIt = s.it; }
				else s.alpha = alpha;
			}
			__syncthreads();
		}
		finishLastBlock(st, B, ctrl);
	}
}

// t = A (M s) with s = r - alpha v computed on the fly; (t,s), (t,t), (s,s).
// Last block: omega, or convergence after the half step.
template <typename T, int NB, bool PRE>
__global__ void kBicgT(const CsrView<T> A, const T *__restrict__ r, const T *__restrict__ v, T *__restrict__ t, const T *__restrict__ dinv,
		BatchState *st, const int B, double *partial, Control *ctrl, const double normScale) {
	const index_t n = A.n;
	const BatchGroup grp = batchGroup<NB>(B, A.rpm);
	const int b0 = grp.b0;
	const int nb = grp.nb;
	bool active[NB];
	bool any = false;
	T alpha[NB];
#pragma unroll
	for (int k = 0; k < NB; ++k) {
		active[k] = k < nb && st[b0 + k].done == RUNNING;
		alpha[k] = active[k] ? T(st[b0 + k].alpha) : T(0);
		any |= active[k];
	}
	double acc[NB][3];
#pragma unroll
	for (int k = 0; k < NB; ++k) acc[k][0] = acc[k][1] = acc[k][2] = 0.0;

	if (any) {
		const T *val = A.val + grp.m * A.valStride;
		const T *dv = PRE ? dinv + (A.valStride ? grp.m * n : 0) : nullptr;
		for (index_t i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += gridDim.x * BLOCK) {
			T sum[NB];
#pragma unroll
			for (int k = 0; k < NB; ++k) sum[k] = 0;
			const index_t end = A.row[i + 1];
			for (index_t j = A.row[i]; j < end; ++j) {
				const index_t c = A.col[j];
				const T a = val[j];
				const T dc = PRE ? dv[c] : T(1);
#pragma unroll
				for (int k = 0; k < NB; ++k) {
					if (active[k]) {
						const index_t o = (b0 + k) * n + c;
						const T sc = r[o] - alpha[k] * v[o];
						sum[k] += a * (PRE ? sc * dc : sc);
					}
				}
			}
#pragma unroll
			for (int k = 0; k < NB; ++k) {
				if (active[k]) {
					const index_t o = (b0 + k) * n + i;
					const T si = r[o] - alpha[k] * v[o];
					t[o] = sum[k];
					acc[k][0] += double(sum[k]) * double(si);
					acc[k][1] += double(sum[k]) * double(sum[k]);
					acc[k][2] += double(si) * double(si);
				}
			}
		}
	}

	if (storePartialsAndCheckLast<NB, 3>(acc, b0, nb, partial, ctrl)) {
		for (int b = 0; b < B; ++b) {
			if (st[b].done != RUNNING) continue;
			double red[3];
			reduceBatch<3>(partial, b, gridDim.x, red);
			if (threadIdx.x == 0) {
				BatchState &s = st[b];
				const double critS = sqrt(red[2]) * normScale;
				s.halfCrit = critS;
				if (!isFinite(critS)) { s.done = NONFINITE; s.crit = critS; s.usedIt = s.it; }
				else if (critS < s.tol) { s.halfDone = 1; s.omega = 0.0; }
				else {
					const double omega = red[0] / red[1];
					if (!isFinite(omega)) { s.done = NONFINITE; s.crit = critS; s.usedIt = s.it; }
					else s.omega = omega;
				}
			}
			__syncthreads();
		}
		finishLastBlock(st, B, ctrl);
	}
}

// s = r - alpha v; x += alpha M p + omega M s; r = s - omega t; (r0, r), (r, r).
// Last block: convergence, beta.
template <typename T, int NB, bool PRE>
__global__ void kBicgUpdate(const index_t n, const int rpm, const index_t dinvStride, T *__restrict__ r, const T *__restrict__ r0, const T *__restrict__ p,
		const T *__restrict__ v, const T *__restrict__ t, T *__restrict__ x, const T *__restrict__ dinv,
		BatchState *st, const int B, double *partial, Control *ctrl, const double normScale, const LoopHandle loop) {
	const BatchGroup grp = batchGroup<NB>(B, rpm);
	const int b0 = grp.b0;
	const int nb = grp.nb;
	double acc[NB][2];
#pragma unroll
	for (int k = 0; k < NB; ++k) acc[k][0] = acc[k][1] = 0.0;
#pragma unroll
	for (int k = 0; k < NB; ++k) {
		if (k >= nb) break;
		const BatchState &bs = st[b0 + k];
		if (bs.done != RUNNING) continue;
		const T alpha = T(bs.alpha), omega = T(bs.omega);
		const bool half = bs.halfDone;
		const index_t off = (b0 + k) * n;
		const T *dv = PRE ? dinv + grp.m * dinvStride : nullptr;
		for (index_t i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += gridDim.x * BLOCK) {
			const index_t o = off + i;
			const T di = PRE ? dv[i] : T(1);
			const T si = r[o] - alpha * v[o];
			if (half) {
				x[o] += alpha * p[o] * di;
				r[o] = si;
			} else {
				x[o] += (alpha * p[o] + omega * si) * di;
				const T ri = si - omega * t[o];
				r[o] = ri;
				acc[k][0] += double(r0[o]) * double(ri);
				acc[k][1] += double(ri) * double(ri);
			}
		}
	}

	if (storePartialsAndCheckLast<NB, 2>(acc, b0, nb, partial, ctrl)) {
		for (int b = 0; b < B; ++b) {
			if (st[b].done != RUNNING) continue;
			double red[2];
			reduceBatch<2>(partial, b, gridDim.x, red);
			if (threadIdx.x == 0) {
				BatchState &s = st[b];
				if (s.halfDone) {
					s.done = CONVERGED;
					s.crit = s.halfCrit;
					s.usedIt = s.it;
				} else {
					const double crit = sqrt(red[1]) * normScale;
					s.crit = crit;
					if (!isFinite(crit)) { s.done = NONFINITE; s.usedIt = s.it; }
					else if (crit < s.tol) { s.done = CONVERGED; s.usedIt = s.it + 1; }
					else {
						++s.it;
						s.usedIt = s.it - 1;
						const double rhoNew = red[0];
						const double beta = (rhoNew / s.rho) * (s.alpha / s.omega);
						s.rho = rhoNew;
						s.beta = beta;
						if (!isFinite(beta)) s.done = NONFINITE;
						else if (s.it >= s.maxit) s.done = MAXIT;
					}
				}
			}
			__syncthreads();
		}
		finishLastBlock(st, B, ctrl);
		// allDone is only final here, in the last block of the last kernel of the iteration
		if (threadIdx.x == 0) maybeSetCondition(loop, ctrl);
	}
}

/* --- workspace --- */

struct GraphEntry {
	cudaGraphExec_t exec = nullptr;
	unsigned long long lastUse = 0;
};

// Control block at offset 0, BatchState array after it
constexpr int64_t STATE_OFFSET = 128;
static_assert(sizeof(Control) <= STATE_OFFSET, "Control does not fit");

struct Workspace {
	at::Tensor vecs;     // all work vectors, [numVecs, capacity]
	at::Tensor partials; // doubles
	at::Tensor stateBuf; // BatchState[B] + Control
	int64_t capacity = 0;  // entries per vector
	int64_t partialCap = 0;
	int64_t stateCap = 0;
	int numVecs = 0;
	int device = -1;
	BatchState *hostState = nullptr; // pinned
	int hostStateCap = 0;
	cudaStream_t captureStream = nullptr;
	std::map<std::vector<unsigned long long>, GraphEntry> graphs;
	unsigned long long useCounter = 0;

	~Workspace() {
		clearGraphs();
		if (hostState) cudaFreeHost(hostState);
		if (captureStream) cudaStreamDestroy(captureStream);
	}
	void clearGraphs() {
		for (auto &g : graphs) if (g.second.exec) cudaGraphExecDestroy(g.second.exec);
		graphs.clear();
	}
	// returns true if buffers were (re)allocated, i.e. cached graphs are invalid
	bool reserve(const int nv, const int64_t entries, const at::ScalarType type, const int64_t nPartials, const int B, const int dev) {
		bool changed = false;
		if (dev != device) {
			vecs = at::Tensor(); partials = at::Tensor(); stateBuf = at::Tensor();
			capacity = partialCap = stateCap = 0;
			device = dev;
			if (captureStream) { cudaStreamDestroy(captureStream); captureStream = nullptr; }
			changed = true;
		}
		if (nv > numVecs || entries > capacity || !vecs.defined() || vecs.scalar_type() != type) {
			const int64_t cap = std::max<int64_t>(entries, capacity);
			vecs = at::empty({std::max(nv, numVecs), cap}, at::TensorOptions().dtype(type).device(at::kCUDA, dev));
			numVecs = std::max(nv, numVecs);
			capacity = cap;
			changed = true;
		}
		if (nPartials > partialCap) {
			partials = at::empty({nPartials}, at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, dev));
			partialCap = nPartials;
			changed = true;
		}
		const int64_t stateBytes = int64_t(B) * sizeof(BatchState) + STATE_OFFSET;
		if (stateBytes > stateCap) {
			stateBuf = at::zeros({stateBytes}, at::TensorOptions().dtype(at::kByte).device(at::kCUDA, dev));
			stateCap = stateBytes;
			changed = true;
		}
		if (B > hostStateCap) {
			if (hostState) cudaFreeHost(hostState);
			KRYLOV_CHECK(cudaMallocHost(&hostState, sizeof(BatchState) * B));
			hostStateCap = B;
		}
		if (!captureStream) KRYLOV_CHECK(cudaStreamCreateWithFlags(&captureStream, cudaStreamNonBlocking));
		if (changed) clearGraphs();
		return changed;
	}
	// auxiliary buffer with a possibly different type (AMG hierarchy vectors)
	at::Tensor aux;
	bool reserveAux(const int64_t entries, const at::ScalarType type) {
		if (aux.defined() && aux.numel() >= entries && aux.scalar_type() == type) return false;
		aux = at::empty({std::max<int64_t>(entries, 1)}, at::TensorOptions().dtype(type).device(at::kCUDA, device));
		clearGraphs();
		return true;
	}
	template <typename T> T *vec(const int i) { return vecs.data_ptr<T>() + i * capacity; }
	double *partial() { return partials.data_ptr<double>(); }
	Control *control() { return reinterpret_cast<Control *>(stateBuf.data_ptr<uint8_t>()); }
	BatchState *state() { return reinterpret_cast<BatchState *>(stateBuf.data_ptr<uint8_t>() + STATE_OFFSET); }
};

static std::mutex g_mutex;
static std::map<std::pair<int, int>, std::unique_ptr<Workspace>> g_workspaces; // (solver kind, device)

static Workspace &getWorkspace(const int kind, const int device) {
	auto &ws = g_workspaces[{kind, device}];
	if (!ws) ws.reset(new Workspace());
	return *ws;
}

void releaseWorkspaces() {
	std::lock_guard<std::mutex> lock(g_mutex);
	g_workspaces.clear();
}

static int numSMs(const int device) {
	static std::map<int, int> cache;
	auto it = cache.find(device);
	if (it != cache.end()) return it->second;
	int v = 0;
	KRYLOV_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, device));
	cache[device] = v;
	return v;
}

// Grid width (blocks per batch group): enough blocks to saturate the GPU, grid-stride beyond that.
static int gridWidth(const index_t n, const int device, const int groups) {
	const int want = std::max(1, (n + BLOCK - 1) / BLOCK);
	const int maxBlocks = std::max(1, numSMs(device) * 8 / std::max(1, groups));
	return std::min(want, maxBlocks);
}

// per-batch tolerances: strided copy into BatchState::tol
static void setBatchTolerances(BatchState *st, const int B, const Options &opt, cudaStream_t stream) {
	if (opt.tols.empty()) return;
	TORCH_CHECK(int(opt.tols.size()) == B, "Number of tolerances (", opt.tols.size(), ") does not match the number of right-hand sides (", B, ").");
	KRYLOV_CHECK(cudaMemcpy2DAsync(&st[0].tol, sizeof(BatchState), opt.tols.data(), sizeof(double), sizeof(double), B, cudaMemcpyHostToDevice, stream));
}

// right-hand sides per matrix, and a check of the batch layout
static int rhsPerMatrix(const index_t valBatchStride, const index_t B, const Options &opt) {
	if (valBatchStride == 0) return B;
	const int rpm = std::max<int>(1, opt.rhsPerMatrix);
	TORCH_CHECK(B % rpm == 0, "Number of right-hand sides (", B, ") is not a multiple of the right-hand sides per matrix (", rpm, ").");
	return rpm;
}

static solverReturn_t collectResults(Workspace &ws, const int B, const bool returnBest, cudaStream_t stream) {
	KRYLOV_CHECK(cudaMemcpyAsync(ws.hostState, ws.state(), sizeof(BatchState) * B, cudaMemcpyDeviceToHost, stream));
	KRYLOV_CHECK(cudaStreamSynchronize(stream));
	solverReturn_t infos;
	infos.reserve(B);
	for (int b = 0; b < B; ++b) {
		const BatchState &s = ws.hostState[b];
		LinearSolverResultInfo info;
		info.finalResidual = s.crit;
		info.usedIterations = s.usedIt;
		info.converged = s.done == CONVERGED;
		info.isFiniteResidual = std::isfinite(s.crit);
		if (s.done == RUNNING) { // loop bound hit, should not happen
			info.converged = false;
		}
		if (returnBest && (s.done == MAXIT || s.done == RISING) && s.bestIt >= 0 && s.best < s.crit) {
			info.usedIterations = s.bestIt;
			info.finalResidual = s.best;
		}
		infos.push_back(info);
	}
	return infos;
}

/* Run `body` as the loop of a cached graph with a conditional WHILE node, or, without
 * graph support, by launching it repeatedly and polling ctrl->allDone asynchronously. */
template <typename LaunchBody>
static void runLoop(Workspace &ws, const std::vector<unsigned long long> &key, cudaStream_t stream, const int maxLoops, LaunchBody body) {
	const auto &cfg = settings();
#if KRYLOV_HAS_CONDITIONAL_GRAPHS
	if (cfg.useGraphs) {
		auto it = ws.graphs.find(key);
		if (it == ws.graphs.end()) {
			if (ws.graphs.size() >= 32) ws.clearGraphs();
			cudaGraph_t graph;
			KRYLOV_CHECK(cudaGraphCreate(&graph, 0));
			cudaGraphConditionalHandle handle;
			KRYLOV_CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault));
			cudaGraphNodeParams params = {};
			params.type = cudaGraphNodeTypeConditional;
			params.conditional.handle = handle;
			params.conditional.type = cudaGraphCondTypeWhile;
			params.conditional.size = 1;
			cudaGraphNode_t node;
#if CUDART_VERSION >= 13000
			// CUDA 13 added the edge data argument (the former cudaGraphAddNode_v2)
			KRYLOV_CHECK(cudaGraphAddNode(&node, graph, nullptr, nullptr, 0, &params));
#else
			KRYLOV_CHECK(cudaGraphAddNode(&node, graph, nullptr, 0, &params));
#endif
			cudaGraph_t bodyGraph = params.conditional.phGraph_out[0];
			KRYLOV_CHECK(cudaStreamBeginCaptureToGraph(ws.captureStream, bodyGraph, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal));
			body(ws.captureStream, LoopHandle{handle, true});
			KRYLOV_CHECK(cudaStreamEndCapture(ws.captureStream, &bodyGraph));
			GraphEntry entry;
			KRYLOV_CHECK(cudaGraphInstantiate(&entry.exec, graph, 0));
			KRYLOV_CHECK(cudaGraphDestroy(graph));
			it = ws.graphs.emplace(key, entry).first;
		}
		it->second.lastUse = ++ws.useCounter;
		KRYLOV_CHECK(cudaGraphLaunch(it->second.exec, stream));
		return;
	}
#endif
	// fallback: host loop with lagged asynchronous polling
	const int check = std::max<int>(1, cfg.checkInterval);
	int *flags = reinterpret_cast<int *>(ws.hostState); // reused as scratch, results are copied afterwards
	cudaEvent_t events[2];
	KRYLOV_CHECK(cudaEventCreateWithFlags(&events[0], cudaEventDisableTiming));
	KRYLOV_CHECK(cudaEventCreateWithFlags(&events[1], cudaEventDisableTiming));
	int chunk = 0;
	for (int loops = 0; loops < maxLoops; ) {
		for (int k = 0; k < check && loops < maxLoops; ++k, ++loops) body(stream, LoopHandle{0, false});
		KRYLOV_CHECK(cudaMemcpyAsync(flags + (chunk & 1), &ws.control()->allDone, sizeof(int), cudaMemcpyDeviceToHost, stream));
		KRYLOV_CHECK(cudaEventRecord(events[chunk & 1], stream));
		if (chunk > 0) {
			KRYLOV_CHECK(cudaEventSynchronize(events[(chunk - 1) & 1]));
			if (flags[(chunk - 1) & 1]) break;
		}
		++chunk;
	}
	KRYLOV_CHECK(cudaStreamSynchronize(stream));
	cudaEventDestroy(events[0]);
	cudaEventDestroy(events[1]);
}

#define KRYLOV_NB_DISPATCH(NB_VAL, ...)                         \
	switch (NB_VAL) {                                           \
		case 1: { constexpr int NB = 1; __VA_ARGS__; break; }   \
		case 2: { constexpr int NB = 2; __VA_ARGS__; break; }   \
		case 3: { constexpr int NB = 3; __VA_ARGS__; break; }   \
		default: { constexpr int NB = 4; __VA_ARGS__; break; }  \
	}

#define KRYLOV_PRE_DISPATCH(PRE_VAL, ...)                              \
	if (PRE_VAL) { constexpr bool PRE = true; __VA_ARGS__; }           \
	else { constexpr bool PRE = false; __VA_ARGS__; }

/* --- CG driver --- */

template <typename scalar_t>
solverReturn_t cgSolve(const scalar_t *aVal, const index_t *aIndex, const index_t *aRow, const index_t n, const index_t nnz,
		const index_t valBatchStride, const scalar_t *b, scalar_t *x, const index_t B, const Options &opt) {
	std::lock_guard<std::mutex> lock(g_mutex);
	int device;
	KRYLOV_CHECK(cudaGetDevice(&device));
	cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();

	// up to 4 right-hand sides that share a matrix are handled by one thread, which then
	// reads the matrix once for all of them
	const int rpm = rhsPerMatrix(valBatchStride, B, opt);
	const int NBv = std::min<int>(rpm, 4);
	const int groups = batchGroupRows(B, rpm, NBv);
	const int G = gridWidth(n, device, groups);
	const bool pre = opt.precond == Preconditioner::JACOBI;
	const bool reset = opt.residualResetSteps > 0;
	const int Bmat = valBatchStride == 0 ? 1 : B / rpm;

	enum { R = 0, Q, P0, P1, X, BV, XBEST, DINV, NUMV };
	Workspace &ws = getWorkspace(0, device);
	ws.reserve(NUMV, int64_t(n) * B, c10::CppTypeToScalarType<scalar_t>::value, int64_t(B) * G * 3, B, device);
	scalar_t *r = ws.vec<scalar_t>(R), *q = ws.vec<scalar_t>(Q), *p0 = ws.vec<scalar_t>(P0), *p1 = ws.vec<scalar_t>(P1);
	scalar_t *xw = ws.vec<scalar_t>(X), *bw = ws.vec<scalar_t>(BV);
	scalar_t *xbest = opt.returnBest ? ws.vec<scalar_t>(XBEST) : nullptr;
	scalar_t *dinv = pre ? ws.vec<scalar_t>(DINV) : nullptr;
	BatchState *st = ws.state();
	Control *ctrl = ws.control();
	double *partial = ws.partial();

	const CsrView<scalar_t> A{aVal, aIndex, aRow, n, valBatchStride, rpm};
	const double normScale = opt.normalized ? 1.0 / std::sqrt(double(n)) : 1.0;
	const dim3 grid(G, groups);
	// the loop body holds two iterations (ping-pong of the search direction)
	const int maxLoops = opt.maxit / 2 + 2;

	// the graph works on workspace copies of x and b, so it can be reused across calls
	KRYLOV_CHECK(cudaMemcpyAsync(xw, x, sizeof(scalar_t) * n * B, cudaMemcpyDeviceToDevice, stream));
	if (reset) KRYLOV_CHECK(cudaMemcpyAsync(bw, b, sizeof(scalar_t) * n * B, cudaMemcpyDeviceToDevice, stream));
	kInitState<scalar_t><<<1, 32, 0, stream>>>(st, B, opt.tol, opt.maxit, ctrl, maxLoops);
	setBatchTolerances(st, B, opt, stream);
	if (pre) kInverseDiagonal<scalar_t><<<std::min(4096, (n * Bmat + 255) / 256), 256, 0, stream>>>(A, dinv, Bmat);
	KRYLOV_NB_DISPATCH(NBv,
		kResidual<scalar_t, NB><<<grid, BLOCK, 0, stream>>>(A, b, xw, r, nullptr, dinv, st, B, partial, ctrl, normScale, 0);
	);
	KRYLOV_CHECK(cudaGetLastError());

	auto body = [&](cudaStream_t strm, const LoopHandle loop) {
		KRYLOV_NB_DISPATCH(NBv, KRYLOV_PRE_DISPATCH(pre,
			constexpr int PM = PRE ? PM_JACOBI : PM_NONE;
			for (int half = 0; half < 2; ++half) {
				scalar_t *pIn = half == 0 ? p0 : p1, *pOut = half == 0 ? p1 : p0;
				if (reset) kResidual<scalar_t, NB><<<grid, BLOCK, 0, strm>>>(A, bw, xw, r, nullptr, dinv, st, B, partial, ctrl, normScale, opt.residualResetSteps);
				kCgDirection<scalar_t, NB, PM><<<grid, BLOCK, 0, strm>>>(A, r, nullptr, dinv, pIn, pOut, q, st, B, partial, ctrl);
				kCgUpdate<scalar_t, NB, PM><<<grid, BLOCK, 0, strm>>>(n, rpm, valBatchStride ? n : 0, r, pOut, q, xw, xbest, dinv, st, B, partial, ctrl,
					normScale, opt.returnBest, half == 1 ? loop : LoopHandle{loop.handle, false});
			}
		));
	};
	const std::vector<unsigned long long> key = {0ull, (unsigned long long)rpm, sizeof(scalar_t), (unsigned long long)aVal, (unsigned long long)aIndex, (unsigned long long)aRow,
		(unsigned long long)n, (unsigned long long)valBatchStride, (unsigned long long)B, (unsigned long long)pre,
		(unsigned long long)opt.returnBest, (unsigned long long)opt.residualResetSteps, (unsigned long long)std::llround(normScale * 1e15)};
	runLoop(ws, key, stream, maxLoops, body);
	KRYLOV_CHECK(cudaGetLastError());

	if (opt.returnBest) kRestoreBest<scalar_t><<<std::min(4096, (n * B + 255) / 256), 256, 0, stream>>>(xw, xbest, st, n, B);
	KRYLOV_CHECK(cudaMemcpyAsync(x, xw, sizeof(scalar_t) * n * B, cudaMemcpyDeviceToDevice, stream));
	return collectResults(ws, B, opt.returnBest, stream);
}

/* --- BiCGStab driver --- */

template <typename scalar_t>
solverReturn_t bicgstabSolve(const scalar_t *aVal, const index_t *aIndex, const index_t *aRow, const index_t n, const index_t nnz,
		const index_t valBatchStride, const scalar_t *b, scalar_t *x, const index_t B, const Options &opt) {
	std::lock_guard<std::mutex> lock(g_mutex);
	int device;
	KRYLOV_CHECK(cudaGetDevice(&device));
	cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();

	const int rpm = rhsPerMatrix(valBatchStride, B, opt);
	const int NBv = std::min<int>(rpm, 4);
	const int groups = batchGroupRows(B, rpm, NBv);
	const int G = gridWidth(n, device, groups);
	const int Gp = gridWidth(n, device, B);
	const bool pre = opt.precond == Preconditioner::JACOBI;
	const int Bmat = valBatchStride == 0 ? 1 : B / rpm;

	enum { R = 0, R0, P, V, T, X, DINV, NUMV };
	Workspace &ws = getWorkspace(1, device);
	ws.reserve(NUMV, int64_t(n) * B, c10::CppTypeToScalarType<scalar_t>::value, int64_t(B) * std::max(G, Gp) * 3, B, device);
	scalar_t *r = ws.vec<scalar_t>(R), *r0 = ws.vec<scalar_t>(R0), *p = ws.vec<scalar_t>(P), *v = ws.vec<scalar_t>(V), *t = ws.vec<scalar_t>(T);
	scalar_t *xw = ws.vec<scalar_t>(X);
	scalar_t *dinv = pre ? ws.vec<scalar_t>(DINV) : nullptr;
	BatchState *st = ws.state();
	Control *ctrl = ws.control();
	double *partial = ws.partial();

	const CsrView<scalar_t> A{aVal, aIndex, aRow, n, valBatchStride, rpm};
	const double normScale = opt.normalized ? 1.0 / std::sqrt(double(n)) : 1.0;
	const dim3 grid(G, groups);
	const dim3 gridP(Gp, B);
	const int maxLoops = opt.maxit + 2;

	KRYLOV_CHECK(cudaMemcpyAsync(xw, x, sizeof(scalar_t) * n * B, cudaMemcpyDeviceToDevice, stream));
	kInitState<scalar_t><<<1, 32, 0, stream>>>(st, B, opt.tol, opt.maxit, ctrl, maxLoops);
	setBatchTolerances(st, B, opt, stream);
	if (pre) kInverseDiagonal<scalar_t><<<std::min(4096, (n * Bmat + 255) / 256), 256, 0, stream>>>(A, dinv, Bmat);
	KRYLOV_NB_DISPATCH(NBv,
		kResidual<scalar_t, NB><<<grid, BLOCK, 0, stream>>>(A, b, xw, r, r0, nullptr, st, B, partial, ctrl, normScale, 0);
	);
	KRYLOV_CHECK(cudaGetLastError());

	auto body = [&](cudaStream_t strm, const LoopHandle loop) {
		kBicgP<scalar_t><<<gridP, BLOCK, 0, strm>>>(n, r, p, v, st, B);
		KRYLOV_NB_DISPATCH(NBv, KRYLOV_PRE_DISPATCH(pre,
			kBicgV<scalar_t, NB, PRE><<<grid, BLOCK, 0, strm>>>(A, p, v, r0, dinv, st, B, partial, ctrl);
			kBicgT<scalar_t, NB, PRE><<<grid, BLOCK, 0, strm>>>(A, r, v, t, dinv, st, B, partial, ctrl, normScale);
			kBicgUpdate<scalar_t, NB, PRE><<<grid, BLOCK, 0, strm>>>(n, rpm, valBatchStride ? n : 0, r, r0, p, v, t, xw, dinv, st, B, partial, ctrl, normScale, loop);
		));
	};
	const std::vector<unsigned long long> key = {1ull, (unsigned long long)rpm, sizeof(scalar_t), (unsigned long long)aVal, (unsigned long long)aIndex, (unsigned long long)aRow,
		(unsigned long long)n, (unsigned long long)valBatchStride, (unsigned long long)B, (unsigned long long)pre,
		(unsigned long long)std::llround(normScale * 1e15)};
	runLoop(ws, key, stream, maxLoops, body);
	KRYLOV_CHECK(cudaGetLastError());

	KRYLOV_CHECK(cudaMemcpyAsync(x, xw, sizeof(scalar_t) * n * B, cudaMemcpyDeviceToDevice, stream));
	return collectResults(ws, B, false, stream);
}


template <typename H>
struct AMGLevel {
	CsrView<H> A;    // level operator, n_l x n_l
	const H *dinv;   // inverse diagonal of A (Jacobi smoother)
	CsrView<H> R;    // restriction, n_{l+1} x n_l
	CsrView<H> P;    // prolongation, n_l x n_{l+1}
	index_t nnzA, nnzR, nnzP;
};

struct AMGOptions {
	bool projectConstant; // singular system with constant nullspace: keep r and M r mean-free
	int preSweeps;
	int postSweeps;
	double omega;         // Jacobi damping
	bool batchedCoarse = false; // one coarse pseudo-inverse per environment
};

/* --- AMG V-cycle kernels ---
 * Level vectors are [B, n_l]; level matrices are shared by all batches or strided by
 * valStride (batched environments with the same sparsity pattern). The finest level reads
 * the CG residual (type T) directly, optionally shifted by its mean (singular systems). */

// load b_i of the current batch, converted to the hierarchy type, minus the optional shift
template <typename TB, typename H>
__device__ inline H loadRhs(const TB *__restrict__ b, const index_t o, const double *shift, const int batch) {
	return shift ? H(double(b[o]) - shift[batch]) : H(b[o]);
}

/* The V-cycle kernels process every row with a group of W consecutive lanes (CSR-vector):
 * the coarse AMG operators have 15-40 entries per row, for which one thread per row reads
 * memory uncoalesced and is latency bound. The row loop is warp-uniform (all lanes of a warp
 * iterate the same number of times), so the shuffle reduction is safe. W=1 is the scalar
 * (thread-per-row) kernel. */

// sum_j M_ij g(col_j) over row i by the W lanes of a group; result valid in lane 0
template <int W, typename H, typename Gather>
__device__ inline H groupRowSum(const index_t *__restrict__ row, const index_t *__restrict__ col, const H *__restrict__ val,
		const index_t i, const bool valid, const int lane, Gather gather) {
	H sum = 0;
	if (valid) {
		const index_t end = row[i + 1];
		for (index_t j = row[i] + lane; j < end; j += W) sum += val[j] * gather(col[j]);
	}
#pragma unroll
	for (int o = W / 2; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o, W);
	return sum;
}

// warp-uniform loop over the rows handled by this lane's group
#define GROUP_ROW_LOOP(W, n)                                                          \
	const int lane = threadIdx.x % (W);                                               \
	const index_t numGroups = index_t(gridDim.x) * (BLOCK / (W));                     \
	const index_t warpFirstGroup = (index_t(blockIdx.x) * BLOCK + (threadIdx.x & ~31)) / (W); \
	const int groupInWarp = (threadIdx.x & 31) / (W);                                 \
	for (index_t base = warpFirstGroup; base < (n); base += numGroups)

// First pre-smoothing sweep from x=0 fused with the residual:
// x = w D^-1 b;  res = b - A x
template <int W, typename TB, typename H>
__global__ void kPresmoothResidual(const CsrView<H> A, const H *__restrict__ dinv, const TB *__restrict__ b, const double *shift,
		H *__restrict__ x, H *__restrict__ res, const H omega, const BatchState *__restrict__ st) {
	if (st[blockIdx.y].done != RUNNING) return;
	const index_t n = A.n;
	const int batch = blockIdx.y;
	const H *val = A.val + batch * A.valStride;
	const H *dv = dinv + (A.valStride ? batch * n : 0);
	const index_t off = batch * n;
	GROUP_ROW_LOOP(W, n) {
		const index_t i = base + groupInWarp;
		const bool valid = i < n;
		const H sum = groupRowSum<W, H>(A.row, A.col, val, i, valid, lane,
			[&](const index_t c) { return dv[c] * loadRhs<TB, H>(b, off + c, shift, batch); });
		if (valid && lane == 0) {
			const H bi = loadRhs<TB, H>(b, off + i, shift, batch);
			x[off + i] = omega * dv[i] * bi;
			res[off + i] = bi - omega * sum;
		}
	}
}

// One damped Jacobi sweep: out = x + w D^-1 (b - A x)
template <int W, typename TB, typename H, typename TO>
__global__ void kJacobiSweep(const CsrView<H> A, const H *__restrict__ dinv, const TB *__restrict__ b, const double *shift,
		const H *__restrict__ x, TO *__restrict__ out, const H omega, const BatchState *__restrict__ st) {
	if (st[blockIdx.y].done != RUNNING) return;
	const index_t n = A.n;
	const int batch = blockIdx.y;
	const H *val = A.val + batch * A.valStride;
	const H *dv = dinv + (A.valStride ? batch * n : 0);
	const index_t off = batch * n;
	GROUP_ROW_LOOP(W, n) {
		const index_t i = base + groupInWarp;
		const bool valid = i < n;
		const H sum = groupRowSum<W, H>(A.row, A.col, val, i, valid, lane, [&](const index_t c) { return x[off + c]; });
		if (valid && lane == 0) {
			const H bi = loadRhs<TB, H>(b, off + i, shift, batch);
			out[off + i] = TO(x[off + i] + omega * dv[i] * (bi - sum));
		}
	}
}

// res = b - A x
template <int W, typename TB, typename H>
__global__ void kLevelResidual(const CsrView<H> A, const TB *__restrict__ b, const double *shift, const H *__restrict__ x, H *__restrict__ res, const BatchState *__restrict__ st) {
	if (st[blockIdx.y].done != RUNNING) return;
	const index_t n = A.n;
	const int batch = blockIdx.y;
	const H *val = A.val + batch * A.valStride;
	const index_t off = batch * n;
	GROUP_ROW_LOOP(W, n) {
		const index_t i = base + groupInWarp;
		const bool valid = i < n;
		const H sum = groupRowSum<W, H>(A.row, A.col, val, i, valid, lane, [&](const index_t c) { return x[off + c]; });
		if (valid && lane == 0) res[off + i] = loadRhs<TB, H>(b, off + i, shift, batch) - sum;
	}
}

// out (+)= M in, for a rectangular M with M.n rows and nIn columns (restriction / prolongation)
template <int W, typename H>
__global__ void kSpmvRect(const CsrView<H> M, const H *__restrict__ in, const index_t nIn, H *__restrict__ out, const bool accumulate, const BatchState *__restrict__ st) {
	if (st[blockIdx.y].done != RUNNING) return;
	const index_t n = M.n;
	const int batch = blockIdx.y;
	const H *val = M.val + batch * M.valStride;
	const H *vin = in + batch * nIn;
	const index_t off = batch * n;
	GROUP_ROW_LOOP(W, n) {
		const index_t i = base + groupInWarp;
		const bool valid = i < n;
		const H sum = groupRowSum<W, H>(M.row, M.col, val, i, valid, lane, [&](const index_t c) { return vin[c]; });
		if (valid && lane == 0) out[off + i] = accumulate ? out[off + i] + sum : sum;
	}
}

// lanes per row for a CSR matrix with the given mean row length
static int rowGroupWidth(const double nnzPerRow) {
	const char *env = std::getenv("PHIPICT_AMG_ROW_WIDTH");
	if (env) {
		const int w = std::atoi(env);
		if (w == 1 || w == 2 || w == 4 || w == 8 || w == 16 || w == 32) return w;
	}
	if (nnzPerRow <= 3.0) return 1;
	if (nnzPerRow <= 6.0) return 2;
	if (nnzPerRow <= 12.0) return 4;
	if (nnzPerRow <= 24.0) return 8;
	if (nnzPerRow <= 48.0) return 16;
	return 32;
}

#define KRYLOV_W_DISPATCH(W_VAL, ...)                           \
	switch (W_VAL) {                                            \
		case 1: { constexpr int W = 1; __VA_ARGS__; break; }    \
		case 2: { constexpr int W = 2; __VA_ARGS__; break; }    \
		case 4: { constexpr int W = 4; __VA_ARGS__; break; }    \
		case 8: { constexpr int W = 8; __VA_ARGS__; break; }    \
		case 16: { constexpr int W = 16; __VA_ARGS__; break; }  \
		default: { constexpr int W = 32; __VA_ARGS__; break; }  \
	}

// coarsest level: x = pinv b (dense, nc x nc, row-major)
template <typename H>
__global__ void kCoarseSolve(const H *__restrict__ pinv, const index_t nc, const H *__restrict__ b, H *__restrict__ x, const BatchState *__restrict__ st,
		const bool batchedPinv) {
	if (st[blockIdx.y].done != RUNNING) return;
	const int batch = blockIdx.y;
	if (batchedPinv) pinv += int64_t(batch) * nc * nc; // one coarse operator per environment
	for (index_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nc; i += gridDim.x * blockDim.x) {
		double sum = 0;
		for (index_t j = 0; j < nc; ++j) sum += double(pinv[i * nc + j]) * double(b[batch * nc + j]);
		x[batch * nc + i] = H(sum);
	}
}

// mean of v per batch, written to means[b]
template <typename T>
__global__ void kMean(const T *__restrict__ v, const index_t n, const int B, double *means, double *partial, Control *ctrl, const BatchState *__restrict__ st) {
	const int batch = blockIdx.y;
	double acc[1][1] = {{0.0}};
	if (st[batch].done == RUNNING) for (index_t i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += gridDim.x * BLOCK) acc[0][0] += double(v[batch * n + i]);
	if (storePartialsAndCheckLast<1, 1>(acc, batch, 1, partial, ctrl)) {
		for (int b = 0; b < B; ++b) {
			double red[1];
			reduceBatch<1>(partial, b, gridDim.x, red);
			if (threadIdx.x == 0) means[b] = red[0] / double(n);
			__syncthreads();
		}
		if (threadIdx.x == 0) ctrl->counter = 0;
	}
}

template <typename T>
__global__ void kSubtractMean(T *__restrict__ v, const index_t n, const double *means, const BatchState *__restrict__ st) {
	if (st[blockIdx.y].done != RUNNING) return;
	const int batch = blockIdx.y;
	const T m = T(means[batch]);
	for (index_t i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += gridDim.x * BLOCK) v[batch * n + i] -= m;
}

/* --- AMG-preconditioned CG driver --- */

template <typename T, typename H>
solverReturn_t amgPcgSolve(const CsrView<T> Afine, const std::vector<AMGLevel<H>> &levels, const H *coarsePinv, const index_t nc,
		const T *b, T *x, const index_t B, const Options &opt, const AMGOptions &amgOpt) {
	std::lock_guard<std::mutex> lock(g_mutex);
	int device;
	KRYLOV_CHECK(cudaGetDevice(&device));
	cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
	const index_t n = Afine.n;
	const int L = levels.size(); // levels with smoother and transfer operators; the coarsest (dense) one is extra
	TORCH_CHECK(L >= 1, "AMG hierarchy needs at least one level above the coarsest");
	TORCH_CHECK(levels[0].A.n == n, "finest AMG level does not match the system matrix");
	const bool project = amgOpt.projectConstant;
	const int pre = std::max(1, amgOpt.preSweeps), post = std::max(1, amgOpt.postSweeps);
	const H omega = H(amgOpt.omega);

	const int G = gridWidth(n, device, B);
	enum { R = 0, Q, P0, P1, X, BV, XBEST, U, NUMV };
	Workspace &ws = getWorkspace(2, device);
	ws.reserve(NUMV, int64_t(n) * B, c10::CppTypeToScalarType<T>::value, int64_t(B) * G * 3 + 2 * B, B, device);
	// hierarchy vectors per level: x, res, y (ping-pong), b (coarse levels) + coarsest b and x
	std::vector<int64_t> offs(L + 1);
	int64_t total = 0;
	for (int l = 0; l < L; ++l) { offs[l] = total; total += int64_t(4) * levels[l].A.n * B; }
	offs[L] = total;
	total += int64_t(2) * nc * B;
	ws.reserveAux(total, c10::CppTypeToScalarType<H>::value);
	H *aux = ws.aux.data_ptr<H>();
	auto lx = [&](int l) { return aux + offs[l]; };
	auto lres = [&](int l) { return aux + offs[l] + int64_t(levels[l].A.n) * B; };
	auto ly = [&](int l) { return aux + offs[l] + int64_t(2) * levels[l].A.n * B; };
	auto lb = [&](int l) { return l < L ? aux + offs[l] + int64_t(3) * levels[l].A.n * B : aux + offs[L]; }; // b of level l (l>=1)
	H *cx = aux + offs[L] + int64_t(nc) * B;

	T *r = ws.vec<T>(R), *q = ws.vec<T>(Q), *p0 = ws.vec<T>(P0), *p1 = ws.vec<T>(P1);
	T *xw = ws.vec<T>(X), *bw = ws.vec<T>(BV), *u = ws.vec<T>(U);
	T *xbest = opt.returnBest ? ws.vec<T>(XBEST) : nullptr;
	BatchState *st = ws.state();
	Control *ctrl = ws.control();
	double *partial = ws.partial();
	double *means = partial + int64_t(B) * G * 3; // [2*B]: mean of r, mean of u
	double *meanR = means, *meanU = means + B;

	const double normScale = opt.normalized ? 1.0 / std::sqrt(double(n)) : 1.0;
	const dim3 grid(G, B);
	const int maxLoops = opt.maxit / 2 + 2; // two iterations per loop body

	KRYLOV_CHECK(cudaMemcpyAsync(xw, x, sizeof(T) * n * B, cudaMemcpyDeviceToDevice, stream));
	KRYLOV_CHECK(cudaMemcpyAsync(bw, b, sizeof(T) * n * B, cudaMemcpyDeviceToDevice, stream));
	kInitState<T><<<1, 32, 0, stream>>>(st, B, opt.tol, opt.maxit, ctrl, maxLoops);
	setBatchTolerances(st, B, opt, stream);
	if (project) {
		// singular (pure Neumann) system: make the RHS compatible
		kMean<T><<<grid, BLOCK, 0, stream>>>(bw, n, B, meanR, partial, ctrl, st);
		kSubtractMean<T><<<grid, BLOCK, 0, stream>>>(bw, n, meanR, st);
	}
	kResidual<T, 1><<<grid, BLOCK, 0, stream>>>(Afine, bw, xw, r, nullptr, nullptr, st, B, partial, ctrl, normScale, 0);
	KRYLOV_CHECK(cudaGetLastError());

	// u = M r: one V-cycle. `cur[l]` tracks which of the x/y buffers of level l holds the
	// current iterate; the other one is the target of the next (out-of-place) Jacobi sweep.
	auto vcycle = [&](cudaStream_t strm) {
		const double *shift = nullptr;
		if (project) {
			kMean<T><<<grid, BLOCK, 0, strm>>>(r, n, B, meanR, partial, ctrl, st);
			shift = meanR;
		}
		std::vector<H *> cur(L), other(L);
		// grid for a level kernel with W lanes per row
		auto lgrid = [&](const index_t rows, const int W) {
			const int want = std::max<int64_t>(1, (int64_t(rows) * W + BLOCK - 1) / BLOCK);
			return dim3(std::min(want, std::max(1, numSMs(device) * 8 / B)), B);
		};
		// downward: pre-smooth, residual, restrict
		for (int l = 0; l < L; ++l) {
			const AMGLevel<H> &lv = levels[l];
			const int wa = rowGroupWidth(double(lv.nnzA) / lv.A.n);
			const int wr = rowGroupWidth(double(lv.nnzR) / lv.R.n);
			const dim3 g = lgrid(lv.A.n, wa);
			cur[l] = lx(l);
			other[l] = ly(l);
			H *resl = lres(l);
			KRYLOV_W_DISPATCH(wa,
				if (l == 0) kPresmoothResidual<W, T, H><<<g, BLOCK, 0, strm>>>(lv.A, lv.dinv, r, shift, cur[l], resl, omega, st);
				else kPresmoothResidual<W, H, H><<<g, BLOCK, 0, strm>>>(lv.A, lv.dinv, lb(l), nullptr, cur[l], resl, omega, st);
				for (int k = 1; k < pre; ++k) {
					if (l == 0) kJacobiSweep<W, T, H, H><<<g, BLOCK, 0, strm>>>(lv.A, lv.dinv, r, shift, cur[l], other[l], omega, st);
					else kJacobiSweep<W, H, H, H><<<g, BLOCK, 0, strm>>>(lv.A, lv.dinv, lb(l), nullptr, cur[l], other[l], omega, st);
					std::swap(cur[l], other[l]);
				}
				if (pre > 1) {
					if (l == 0) kLevelResidual<W, T, H><<<g, BLOCK, 0, strm>>>(lv.A, r, shift, cur[l], resl, st);
					else kLevelResidual<W, H, H><<<g, BLOCK, 0, strm>>>(lv.A, lb(l), nullptr, cur[l], resl, st);
				}
			);
			KRYLOV_W_DISPATCH(wr,
				kSpmvRect<W, H><<<lgrid(lv.R.n, wr), BLOCK, 0, strm>>>(lv.R, resl, lv.A.n, lb(l + 1), false, st);
			);
		}
		// coarsest level
		kCoarseSolve<H><<<dim3((nc + 127) / 128, B), 128, 0, strm>>>(coarsePinv, nc, lb(L), cx, st, amgOpt.batchedCoarse);
		// upward: prolongate the correction, post-smooth
		for (int l = L - 1; l >= 0; --l) {
			const AMGLevel<H> &lv = levels[l];
			const int wa = rowGroupWidth(double(lv.nnzA) / lv.A.n);
			const int wp = rowGroupWidth(double(lv.nnzP) / lv.P.n);
			const dim3 g = lgrid(lv.A.n, wa);
			const H *xc = l + 1 < L ? cur[l + 1] : cx;
			const index_t ncoarse = l + 1 < L ? levels[l + 1].A.n : nc;
			KRYLOV_W_DISPATCH(wp,
				kSpmvRect<W, H><<<lgrid(lv.P.n, wp), BLOCK, 0, strm>>>(lv.P, xc, ncoarse, cur[l], true, st);
			);
			KRYLOV_W_DISPATCH(wa,
				for (int k = 0; k < post; ++k) {
					if (l == 0) {
						if (k == post - 1) kJacobiSweep<W, T, H, T><<<g, BLOCK, 0, strm>>>(lv.A, lv.dinv, r, shift, cur[l], u, omega, st);
						else kJacobiSweep<W, T, H, H><<<g, BLOCK, 0, strm>>>(lv.A, lv.dinv, r, shift, cur[l], other[l], omega, st);
					} else {
						kJacobiSweep<W, H, H, H><<<g, BLOCK, 0, strm>>>(lv.A, lv.dinv, lb(l), nullptr, cur[l], other[l], omega, st);
					}
					std::swap(cur[l], other[l]);
				}
			);
		}
		if (project) {
			kMean<T><<<grid, BLOCK, 0, strm>>>(u, n, B, meanU, partial, ctrl, st);
			kSubtractMean<T><<<grid, BLOCK, 0, strm>>>(u, n, meanU, st);
		}
	};

	auto body = [&](cudaStream_t strm, const LoopHandle loop) {
		for (int half = 0; half < 2; ++half) {
			T *pIn = half == 0 ? p0 : p1, *pOut = half == 0 ? p1 : p0;
			vcycle(strm);
			kCgDots<T><<<grid, BLOCK, 0, strm>>>(n, r, u, st, B, partial, ctrl);
			kCgDirection<T, 1, PM_VECTOR><<<grid, BLOCK, 0, strm>>>(Afine, r, u, nullptr, pIn, pOut, q, st, B, partial, ctrl);
			kCgUpdate<T, 1, PM_VECTOR><<<grid, BLOCK, 0, strm>>>(n, 1, 0, r, pOut, q, xw, xbest, nullptr, st, B, partial, ctrl, normScale,
				opt.returnBest, half == 1 ? loop : LoopHandle{loop.handle, false});
		}
	};
	const char *rowWidthEnv = std::getenv("PHIPICT_AMG_ROW_WIDTH");
	std::vector<unsigned long long> key = {2ull, (unsigned long long)amgOpt.batchedCoarse, (unsigned long long)Afine.valStride, sizeof(T), sizeof(H), (unsigned long long)(rowWidthEnv ? std::atoi(rowWidthEnv) : 0), (unsigned long long)Afine.val, (unsigned long long)Afine.col,
		(unsigned long long)Afine.row, (unsigned long long)n, (unsigned long long)B, (unsigned long long)project, (unsigned long long)pre,
		(unsigned long long)post, (unsigned long long)std::llround(amgOpt.omega * 1e12), (unsigned long long)opt.returnBest,
		(unsigned long long)coarsePinv, (unsigned long long)nc, (unsigned long long)std::llround(normScale * 1e15)};
	for (const auto &lv : levels) {
		for (const void *ptr : {(const void *)lv.A.val, (const void *)lv.A.col, (const void *)lv.A.row, (const void *)lv.dinv,
				(const void *)lv.R.val, (const void *)lv.R.col, (const void *)lv.R.row, (const void *)lv.P.val, (const void *)lv.P.col, (const void *)lv.P.row})
			key.push_back((unsigned long long)ptr);
		key.push_back((unsigned long long)lv.A.n);
		key.push_back((unsigned long long)lv.A.valStride);
	}
	runLoop(ws, key, stream, maxLoops, body);
	KRYLOV_CHECK(cudaGetLastError());

	if (opt.returnBest) kRestoreBest<T><<<std::min(4096, (n * B + 255) / 256), 256, 0, stream>>>(xw, xbest, st, n, B);
	KRYLOV_CHECK(cudaMemcpyAsync(x, xw, sizeof(T) * n * B, cudaMemcpyDeviceToDevice, stream));
	return collectResults(ws, B, opt.returnBest, stream);
}

/* --- torch entry point for the AMG solver --- */

template <typename H>
static CsrView<H> csrFromTensors(const at::Tensor &row, const at::Tensor &col, const at::Tensor &val, const index_t valStride) {
	TORCH_CHECK(row.is_cuda() && col.is_cuda() && val.is_cuda(), "AMG tensors must be on the GPU");
	TORCH_CHECK(row.scalar_type() == at::kInt && col.scalar_type() == at::kInt, "AMG index tensors must be int32");
	TORCH_CHECK(row.is_contiguous() && col.is_contiguous() && val.is_contiguous(), "AMG tensors must be contiguous");
	return CsrView<H>{val.data_ptr<H>(), col.data_ptr<index_t>(), row.data_ptr<index_t>(), index_t(row.numel() - 1), valStride, 1};
}
// values of one matrix per environment (batched environments, shared pattern), or one shared matrix
template <typename H>
static CsrView<H> csrFromTensorsBatched(const at::Tensor &row, const at::Tensor &col, const at::Tensor &val, const index_t B, const std::string &name) {
	const index_t nnz = col.numel();
	TORCH_CHECK(val.numel() == nnz || val.numel() == int64_t(nnz) * B, name + ": values must hold one matrix or one per right-hand side.");
	return csrFromTensors<H>(row, col, val, val.numel() == nnz ? 0 : nnz);
}

solverReturn_t amgPcgSolveTorch(const at::Tensor &aRow, const at::Tensor &aCol, const at::Tensor &aVal,
		const std::vector<std::vector<at::Tensor>> &levels, const at::Tensor &coarsePinv,
		const at::Tensor &rhs, at::Tensor &x, const index_t maxit, const double tol, const bool normalized, const bool returnBest,
		const bool projectConstant, const int preSweeps, const int postSweeps, const double omega,
		const std::vector<double> &tolerances) {
	const c10::cuda::CUDAGuard guard(rhs.device());
	TORCH_CHECK(rhs.is_contiguous() && x.is_contiguous() && rhs.numel() == x.numel(), "rhs and x must be contiguous and of equal size");
	TORCH_CHECK(rhs.scalar_type() == aVal.scalar_type() && x.scalar_type() == aVal.scalar_type(), "rhs, x and the matrix must have the same dtype");
	const index_t n = aRow.numel() - 1;
	TORCH_CHECK(rhs.numel() % n == 0, "rhs size must be a multiple of the matrix size");
	const index_t B = rhs.numel() / n;
	const auto htype = coarsePinv.scalar_type();
	for (const auto &lv : levels) TORCH_CHECK(lv.size() == 10, "each AMG level needs (A row, col, val, dinv, R row, col, val, P row, col, val)");
	Options opt;
	opt.maxit = maxit;
	opt.tol = tol;
	opt.normalized = normalized;
	opt.returnBest = returnBest;
	opt.tols = tolerances;
	AMGOptions amgOpt{projectConstant, preSweeps, postSweeps, omega};
	solverReturn_t ret;
	AT_DISPATCH_FLOATING_TYPES(aVal.scalar_type(), "amgPcgSolve", ([&] {
		using T = scalar_t;
		const CsrView<T> Afine = csrFromTensorsBatched<T>(aRow, aCol, aVal, B, "system matrix");
		auto run = [&](auto hTag) {
			using H = decltype(hTag);
			std::vector<AMGLevel<H>> lv;
			for (const auto &t : levels) {
				TORCH_CHECK(t[2].scalar_type() == htype && t[3].scalar_type() == htype && t[6].scalar_type() == htype && t[9].scalar_type() == htype,
					"all hierarchy values must have the dtype of the coarse pseudo-inverse");
				const CsrView<H> Al = csrFromTensorsBatched<H>(t[0], t[1], t[2], B, "level operator");
				TORCH_CHECK(t[3].numel() == (Al.valStride ? int64_t(Al.n) * B : Al.n), "inverse diagonal must match the level operator(s)");
				lv.push_back(AMGLevel<H>{Al, t[3].data_ptr<H>(),
					csrFromTensors<H>(t[4], t[5], t[6], 0), csrFromTensors<H>(t[7], t[8], t[9], 0),
					index_t(t[1].numel()), index_t(t[5].numel()), index_t(t[8].numel())});
			}
			// coarsest level: [nc, nc] shared or [B, nc, nc] per environment
			const index_t nc = index_t(coarsePinv.size(-1));
			amgOpt.batchedCoarse = coarsePinv.dim() == 3 && B > 1;
			TORCH_CHECK(coarsePinv.numel() == (amgOpt.batchedCoarse ? int64_t(nc) * nc * B : int64_t(nc) * nc), "coarse pseudo-inverse has the wrong size");
			ret = amgPcgSolve<T, H>(Afine, lv, coarsePinv.contiguous().data_ptr<H>(), nc, rhs.data_ptr<T>(), x.data_ptr<T>(), B, opt, amgOpt);
		};
		if (htype == at::kFloat) run(float{});
		else if (htype == at::kDouble) run(double{});
		else TORCH_CHECK(false, "AMG hierarchy dtype must be float32 or float64");
	}));
	return ret;
}

/* --- CSR transpose --- */

struct TransposeBuffers {
	at::Tensor val, index, row, work;
};

template <typename scalar_t>
void csrTranspose(const scalar_t *aVal, const index_t *aIndex, const index_t *aRow, const index_t n, const index_t nnz,
		const index_t numMatrices, const scalar_t **tVal, const index_t **tIndex, const index_t **tRow) {
	static std::map<int, TransposeBuffers> buffers;
	static std::map<int, cusparseHandle_t> handles;
	int device;
	KRYLOV_CHECK(cudaGetDevice(&device));
	cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
	auto &h = handles[device];
	if (!h) KRYLOV_CHECK(cusparseCreate(&h));
	KRYLOV_CHECK(cusparseSetStream(h, stream));
	auto &buf = buffers[device];
	const auto opts = at::TensorOptions().device(at::kCUDA, device);
	const auto type = c10::CppTypeToScalarType<scalar_t>::value;
	if (!buf.val.defined() || buf.val.numel() < int64_t(nnz) * numMatrices || buf.val.scalar_type() != type)
		buf.val = at::empty({int64_t(nnz) * numMatrices}, opts.dtype(type));
	if (!buf.index.defined() || buf.index.numel() < nnz) buf.index = at::empty({nnz}, opts.dtype(at::kInt));
	if (!buf.row.defined() || buf.row.numel() < n + 1) buf.row = at::empty({n + 1}, opts.dtype(at::kInt));
	const cudaDataType dt = std::is_same<scalar_t, double>::value ? CUDA_R_64F : CUDA_R_32F;
	size_t workSize = 0;
	KRYLOV_CHECK(cusparseCsr2cscEx2_bufferSize(h, n, n, nnz, aVal, aRow, aIndex, buf.val.data_ptr(), buf.row.data_ptr<index_t>(), buf.index.data_ptr<index_t>(),
		dt, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &workSize));
	if (!buf.work.defined() || size_t(buf.work.numel()) < workSize) buf.work = at::empty({int64_t(std::max<size_t>(workSize, 1))}, opts.dtype(at::kByte));
	// same pattern for all matrices: the index arrays are rewritten identically each time
	for (index_t m = 0; m < numMatrices; ++m) {
		KRYLOV_CHECK(cusparseCsr2cscEx2(h, n, n, nnz, aVal + int64_t(m) * nnz, aRow, aIndex, buf.val.data_ptr<scalar_t>() + int64_t(m) * nnz,
			buf.row.data_ptr<index_t>(), buf.index.data_ptr<index_t>(),
			dt, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, buf.work.data_ptr()));
	}
	*tVal = buf.val.data_ptr<scalar_t>();
	*tIndex = buf.index.data_ptr<index_t>();
	*tRow = buf.row.data_ptr<index_t>();
}

/* --- instantiations --- */

template solverReturn_t cgSolve<float>(const float*, const index_t*, const index_t*, const index_t, const index_t, const index_t, const float*, float*, const index_t, const Options&);
template solverReturn_t cgSolve<double>(const double*, const index_t*, const index_t*, const index_t, const index_t, const index_t, const double*, double*, const index_t, const Options&);
template solverReturn_t bicgstabSolve<float>(const float*, const index_t*, const index_t*, const index_t, const index_t, const index_t, const float*, float*, const index_t, const Options&);
template solverReturn_t bicgstabSolve<double>(const double*, const index_t*, const index_t*, const index_t, const index_t, const index_t, const double*, double*, const index_t, const Options&);
template void csrTranspose<float>(const float*, const index_t*, const index_t*, const index_t, const index_t, const index_t, const float**, const index_t**, const index_t**);
template void csrTranspose<double>(const double*, const index_t*, const index_t*, const index_t, const index_t, const index_t, const double**, const index_t**, const index_t**);

} // namespace krylov
