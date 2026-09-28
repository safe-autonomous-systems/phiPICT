"""Tests of the fused Krylov solvers (csrc/phipict/krylov) against the legacy ones."""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from phipict import _C  # noqa: E402


def _laplacian_2d(
    nx: int, ny: int, dtype, device, anisotropy: float = 1.0, shift: float = 0.0
):
    """Negative definite 5-point Laplacian with Dirichlet walls, as a sparse CSR tensor.

    ``shift`` adds a first-order (upwind-like) x-derivative, making the matrix
    non-symmetric (for the BiCGStab tests).
    """

    def tridiag(m: int, lo: float, mid: float, hi: float) -> torch.Tensor:
        return (
            torch.diag(torch.full((m - 1,), lo, dtype=dtype), -1)
            + torch.diag(torch.full((m,), mid, dtype=dtype))
            + torch.diag(torch.full((m - 1,), hi, dtype=dtype), 1)
        )

    tx = tridiag(nx, 1.0, -2.0 - shift, 1.0 + shift)
    ty = tridiag(ny, 1.0, -2.0, 1.0)
    A = torch.kron(torch.eye(ny, dtype=dtype), tx) + anisotropy * torch.kron(
        ty, torch.eye(nx, dtype=dtype)
    )
    A -= 1e-3 * torch.eye(nx * ny, dtype=dtype)
    return A.to(device).to_sparse_csr()


def _csr(A: torch.Tensor):
    return _C.CSRmatrix(
        A.values().contiguous(),
        A.col_indices().to(torch.int32).contiguous(),
        A.crow_indices().to(torch.int32).contiguous(),
    )


def _solve(
    A,
    b,
    x0,
    use_bicg=False,
    transpose=False,
    fused=True,
    tol=1e-10,
    maxit=5000,
    return_best=False,
    **settings,
):
    _C.SetKrylovSettings(enabled=fused, **settings)
    x = x0.clone()
    infos = _C.SolveLinear(
        _csr(A),
        b,
        x,
        torch.IntTensor([maxit]),
        torch.tensor([tol], dtype=b.dtype),
        _C.ConvergenceCriterion.NORM2_NORMALIZED,
        use_bicg,
        False,
        0,
        transpose,
        False,
        return_best,
        False,
    )
    _C.SetKrylovSettings()
    return x, infos


def _true_res(A, b, x, transpose=False):
    n = A.shape[0]
    M = A.to_dense().T if transpose else A.to_dense()
    nb = b.numel() // n
    return (
        torch.linalg.vector_norm(b.view(nb, n).T - M @ x.view(nb, n).T, dim=0)
        / math.sqrt(n)
    ).tolist()


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("graphs", [True, False])
@pytest.mark.parametrize("precond", ["none", "jacobi"])
def test_cg_matches_legacy(dtype, graphs, precond):
    dev = torch.device("cuda")
    A = _laplacian_2d(40, 30, dtype, dev, anisotropy=10.0)
    n = A.shape[0]
    torch.manual_seed(0)
    b = torch.randn(n, dtype=dtype, device=dev)
    x0 = torch.zeros_like(b)
    tol = 1e-9 if dtype == torch.float64 else 1e-4
    x_new, info_new = _solve(
        A, b, x0, tol=tol, useGraphs=graphs, cgPreconditioner=precond
    )
    x_old, info_old = _solve(A, b, x0, tol=tol, fused=False)
    assert info_new[0].converged
    assert _true_res(A, b, x_new)[0] < 2 * tol
    rel = (
        torch.linalg.vector_norm(x_new - x_old) / torch.linalg.vector_norm(x_old)
    ).item()
    assert rel < (1e-6 if dtype == torch.float64 else 1e-2)
    if precond == "none":
        # same Krylov method, the iteration counts agree up to rounding effects
        assert abs(info_new[0].usedIterations - info_old[0].usedIterations) <= max(
            3, 0.05 * info_old[0].usedIterations
        )


def test_cg_multi_rhs_and_warm_start():
    dev = torch.device("cuda")
    dtype = torch.float64
    A = _laplacian_2d(33, 17, dtype, dev)
    n = A.shape[0]
    torch.manual_seed(1)
    b = torch.randn(3 * n, dtype=dtype, device=dev)
    b[n : 2 * n] *= 1e-3  # very different scales per RHS: independent convergence
    x0 = torch.randn(3 * n, dtype=dtype, device=dev) * 0.1
    x, infos = _solve(A, b, x0, tol=1e-10)
    assert len(infos) == 3
    assert all(i.converged for i in infos)
    assert max(_true_res(A, b, x)) < 2e-10


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("n_rhs", [1, 3])
def test_bicgstab_matches_legacy(transpose, n_rhs):
    dev = torch.device("cuda")
    dtype = torch.float64
    A = _laplacian_2d(30, 25, dtype, dev, shift=2.0)  # non-symmetric
    n = A.shape[0]
    torch.manual_seed(2)
    b = torch.randn(n * n_rhs, dtype=dtype, device=dev)
    x0 = torch.zeros_like(b)
    x_new, info_new = _solve(A, b, x0, use_bicg=True, transpose=transpose, tol=1e-10)
    x_old, info_old = _solve(
        A, b, x0, use_bicg=True, transpose=transpose, tol=1e-10, fused=False
    )
    assert all(i.converged for i in info_new)
    assert max(_true_res(A, b, x_new, transpose)) < 2e-10
    assert max(_true_res(A, b, x_old, transpose)) < 2e-10


def test_zero_rhs_and_converged_start():
    dev = torch.device("cuda")
    A = _laplacian_2d(10, 10, torch.float64, dev)
    b = torch.randn(100, dtype=torch.float64, device=dev)
    x_exact, _ = _solve(A, b, torch.zeros_like(b), tol=1e-13)
    for bicg in (False, True):
        x, infos = _solve(A, b, x_exact, use_bicg=bicg, tol=1e-8)
        assert infos[0].converged
        assert torch.allclose(x, x_exact)


def test_return_best_on_maxit():
    dev = torch.device("cuda")
    A = _laplacian_2d(60, 60, torch.float64, dev)
    b = torch.randn(A.shape[0], dtype=torch.float64, device=dev)
    x, infos = _solve(A, b, torch.zeros_like(b), tol=1e-14, maxit=20, return_best=True)
    assert not infos[0].converged
    # the returned iterate is the best one seen and the reported residual is its own
    assert _true_res(A, b, x)[0] == pytest.approx(infos[0].finalResidual, rel=1e-6)


def test_nonfinite_is_reported():
    dev = torch.device("cuda")
    A = _laplacian_2d(10, 10, torch.float64, dev)
    b = torch.randn(A.shape[0], dtype=torch.float64, device=dev)
    b[3] = float("nan")
    for bicg in (False, True):
        _, infos = _solve(A, b, torch.zeros_like(b), use_bicg=bicg)
        assert not infos[0].isFiniteResidual
        assert not infos[0].converged


@pytest.mark.parametrize("native_dtype", [None, torch.float32])
def test_native_amg_matches_python(native_dtype):
    pytest.importorskip("pyamg")
    from phipict.solvers import amg

    if not amg.native_amg_available():
        pytest.skip("extension built without AMGPCGSolve")
    dev = torch.device("cuda")
    A = _laplacian_2d(64, 48, torch.float64, dev, anisotropy=50.0)
    n = A.shape[0]
    h = amg.build_amg_hierarchy(_csr(A), max_coarse=32)
    h.native_dtype = native_dtype
    torch.manual_seed(3)
    b = torch.randn(n, dtype=torch.float64, device=dev)

    amg.USE_NATIVE = False
    x_py = torch.zeros_like(b)
    info_py = amg.amg_pcg_solve(b, x_py, h, tol=1e-10, max_iter=200)
    amg.USE_NATIVE = True
    x_nat = torch.zeros_like(b)
    info_nat = amg.amg_pcg_solve(b, x_nat, h, tol=1e-10, max_iter=200)

    assert info_py[0].converged and info_nat[0].converged
    assert _true_res(A, b, x_nat)[0] < 2e-10
    # same preconditioner and Krylov method (up to the beta formula and precision)
    assert abs(info_nat[0].usedIterations - info_py[0].usedIterations) <= 3


def test_galerkin_refresh_is_exact_for_the_same_operator():
    pytest.importorskip("pyamg")
    from phipict.solvers import amg

    dev = torch.device("cuda")
    A = _laplacian_2d(32, 32, torch.float64, dev, anisotropy=5.0)
    h = amg.build_amg_hierarchy(_csr(A), max_coarse=16)
    g = amg.galerkin_refresh(h, _csr(A))
    for lh, lg in zip(h.levels, g.levels, strict=True):
        assert torch.allclose(lh.A.to_dense(), lg.A.to_dense(), rtol=1e-10, atol=1e-12)


def test_interpolation_disk_cache(tmp_path, monkeypatch):
    pytest.importorskip("pyamg")
    from phipict.solvers import amg

    dev = torch.device("cuda")
    A = _laplacian_2d(40, 40, torch.float64, dev, anisotropy=20.0)
    monkeypatch.setenv("PHIPICT_AMG_CACHE_DIR", str(tmp_path))
    amg.clear_interpolation_cache()
    h1 = amg.hierarchy_for(_csr(A), max_coarse=16)
    assert len(list(tmp_path.glob("amg_interp_*.pt"))) == 1
    amg.clear_interpolation_cache()
    h2 = amg.hierarchy_for(
        _csr(A), max_coarse=16
    )  # rebuilt from the stored prolongators
    amg.clear_interpolation_cache()
    assert len(h1.levels) == len(h2.levels)
    for l1, l2 in zip(h1.levels, h2.levels, strict=True):
        assert torch.allclose(l1.A.to_dense(), l2.A.to_dense(), rtol=1e-10, atol=1e-12)
    b = torch.randn(A.shape[0], dtype=torch.float64, device=dev)
    i1 = amg.amg_pcg_solve(b, torch.zeros_like(b), h1, tol=1e-10, max_iter=100)[0]
    i2 = amg.amg_pcg_solve(b, torch.zeros_like(b), h2, tol=1e-10, max_iter=100)[0]
    assert i1.converged and i2.converged and i1.usedIterations == i2.usedIterations
