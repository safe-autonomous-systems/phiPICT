# Copyright 2026 Jannis Becktepe
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the AMG-preconditioned potential solve."""

import pytest
import torch
from test_domain_io_mhd import (  # noqa: E402
    DTYPE,
    HARTMANN,
    REYNOLDS,
    _make_domain,
)

from phipict.solvers import stats as solver_stats  # noqa: E402
from phipict.solvers.amg import (  # noqa: E402
    amg_pcg_solve,
    build_amg_hierarchy,
    csr_to_scipy,
    galerkin_refresh,
    is_pyamg_available,
)

pytestmark = pytest.mark.skipif(
    not is_pyamg_available(), reason="AMG setup needs the pyamg package"
)


def _make_sim(use_amg: bool, potential_tol: float = 1e-8):
    from phipict.simulation.mhd import MHDSimulation

    domain = _make_domain(hartmann_Cw=0.0, heat_both_walls=False)
    sim = MHDSimulation(
        domain=domain,
        dt=1e-3,
        stuart_number=torch.tensor(HARTMANN**2 / REYNOLDS),
        e_b=torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE),
        substeps=1,
        corrector_steps=2,
        advection_tol=1e-5,
        pressure_tol=5e-6,
        potential_tol=potential_tol,
        non_orthogonal=False,
        potential_normalize=True,
        potential_use_BiCG=False,
        potential_solve_max_iter=2000,
        potential_use_preconditioner=use_amg,
    )
    sim.make_divergence_free(max_iter=1000)
    return domain, sim


def _epot_matrix(domain):
    return domain.Epot


# ---------------------------------------------------------------------------
# Matrix conversion
# ---------------------------------------------------------------------------


def test_conversion_preserves_the_pure_neumann_structure() -> None:
    """The epot operator has exactly zero row sums; conversion must keep that.

    It is the defining property of the singular pure-Neumann system and the
    cheapest possible check that the CSR was read correctly (column indices,
    row pointers and the sentinel filter all have to be right for it to hold).
    """
    domain, _ = _make_sim(use_amg=False)
    A = csr_to_scipy(_epot_matrix(domain))

    n = _epot_matrix(domain).getRows()
    assert A.shape == (n, n)
    row_sums = abs(A.sum(axis=1)).max()
    assert row_sums < 1e-12, f"max |row sum| = {row_sums:.3e}, expected ~0"


def test_conversion_matches_the_gpu_matvec() -> None:
    """The converted operator must reproduce what the solver actually applies."""
    domain, _ = _make_sim(use_amg=False)
    csr = _epot_matrix(domain)
    n = csr.getRows()

    A_host = csr_to_scipy(csr)
    A_dev = torch.sparse_csr_tensor(
        csr.row.to(torch.int64), csr.index.to(torch.int64), csr.value, size=(n, n)
    )

    x = torch.randn(n, dtype=csr.value.dtype, device=csr.value.device)
    ref = (A_dev @ x.unsqueeze(1)).squeeze(1).cpu().numpy()
    got = A_host @ x.cpu().numpy()
    assert abs(got - ref).max() < 1e-10


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


def test_hierarchy_coarsens_and_stays_cheap() -> None:
    domain, _ = _make_sim(use_amg=False)
    h = build_amg_hierarchy(_epot_matrix(domain), project_constant=True)

    assert len(h.levels) >= 2, "no coarsening happened"
    sizes = [lvl.A.shape[0] for lvl in h.levels]
    assert sizes == sorted(sizes, reverse=True), f"levels not shrinking: {sizes}"
    # Above ~2 the coarse operators are filling in and every V-cycle gets costly.
    assert h.operator_complexity < 3.0


def test_chunked_galerkin_products_match(monkeypatch) -> None:
    """Galerkin products in row chunks equal the one-shot products."""
    from phipict.solvers import amg

    domain, _ = _make_sim(use_amg=False)
    csr = _epot_matrix(domain)
    h = build_amg_hierarchy(csr, project_constant=True)
    whole = galerkin_refresh(h, csr)
    monkeypatch.setattr(amg, "GALERKIN_CHUNK_PRODUCTS", 997)
    chunked = galerkin_refresh(h, csr)
    for a, b in zip(whole.levels, chunked.levels, strict=True):
        assert torch.equal(a.A.crow_indices(), b.A.crow_indices())
        assert torch.equal(a.A.col_indices(), b.A.col_indices())
        assert torch.allclose(a.A.values(), b.A.values(), rtol=1e-12, atol=0)
    assert torch.allclose(whole.coarse_pinv, chunked.coarse_pinv, rtol=1e-8)


def test_interpolation_cache_keeps_no_operators() -> None:
    """The cache holds P and R only; hierarchies built on it share their native
    conversions."""
    from phipict.solvers import amg

    amg.clear_interpolation_cache()
    domain, _ = _make_sim(use_amg=False)
    csr = _epot_matrix(domain)
    first = amg.hierarchy_for(csr, project_constant=True)
    (cached,) = amg._INTERPOLATION_CACHE.values()
    assert all(lvl.A._nnz() == 0 for lvl in cached.levels)
    second = amg.hierarchy_for(csr, project_constant=True)
    l1, _ = first.native_levels(torch.float32)
    l2, _ = second.native_levels(torch.float32)
    for a, b in zip(l1, l2, strict=True):
        assert all(x.data_ptr() == y.data_ptr() for x, y in zip(a[4:], b[4:]))
        assert a[0].dtype == torch.int32
    amg.clear_interpolation_cache()


def test_vcycle_reduces_the_residual() -> None:
    """One V-cycle must be a contraction. Otherwise it is not a preconditioner."""
    domain, _ = _make_sim(use_amg=False)
    csr = _epot_matrix(domain)
    n = csr.getRows()
    h = build_amg_hierarchy(csr, project_constant=True)

    A = h.levels[0].A
    torch.manual_seed(0)
    b = torch.randn(n, dtype=h.dtype, device=h.device)
    b = b - b.mean()  # compatible RHS for the singular system

    x = h(b)
    residual_before = torch.linalg.vector_norm(b)
    residual_after = torch.linalg.vector_norm(b - (A @ x.unsqueeze(1)).squeeze(1))
    assert residual_after < residual_before


def test_vcycle_output_has_no_constant_mode_when_projecting() -> None:
    domain, _ = _make_sim(use_amg=False)
    csr = _epot_matrix(domain)
    h = build_amg_hierarchy(csr, project_constant=True)

    torch.manual_seed(0)
    r = torch.randn(csr.getRows(), dtype=h.dtype, device=h.device)
    z = h(r)
    assert abs(float(z.mean())) < 1e-12


# ---------------------------------------------------------------------------
# Preconditioned CG
# ---------------------------------------------------------------------------


def test_pcg_solves_the_singular_system() -> None:
    domain, _ = _make_sim(use_amg=False)
    csr = _epot_matrix(domain)
    n = csr.getRows()
    h = build_amg_hierarchy(csr, project_constant=True)
    A = h.levels[0].A

    torch.manual_seed(0)
    x_true = torch.randn(n, dtype=h.dtype, device=h.device)
    x_true = x_true - x_true.mean()
    b = (A @ x_true.unsqueeze(1)).squeeze(1)

    x = torch.zeros_like(b)
    infos = amg_pcg_solve(b, x, h, tol=1e-12, max_iter=500)

    assert len(infos) == 1
    assert infos[0].converged, f"did not converge: {infos[0]}"
    assert infos[0].isFiniteResidual
    # Only the range-space component is determined for a singular system.
    torch.testing.assert_close(x - x.mean(), x_true, rtol=1e-6, atol=1e-8)


def test_pcg_reports_non_convergence_rather_than_lying() -> None:
    domain, _ = _make_sim(use_amg=False)
    csr = _epot_matrix(domain)
    h = build_amg_hierarchy(csr, project_constant=True)

    torch.manual_seed(0)
    b = torch.randn(csr.getRows(), dtype=h.dtype, device=h.device)
    b = b - b.mean()
    x = torch.zeros_like(b)
    infos = amg_pcg_solve(b, x, h, tol=1e-30, max_iter=3)

    assert not infos[0].converged
    assert infos[0].usedIterations <= 3


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_amg_matches_plain_cg_but_in_far_fewer_iterations() -> None:
    """The whole point: same potential field, a fraction of the iterations."""
    domain_ref, sim_ref = _make_sim(use_amg=False)
    with solver_stats.SolverStatsRecorder(time_solves=False) as rec_ref:
        for _ in range(3):
            sim_ref.single_step()

    domain_amg, sim_amg = _make_sim(use_amg=True)
    with solver_stats.SolverStatsRecorder(time_solves=False) as rec_amg:
        for _ in range(3):
            sim_amg.single_step()

    iters_ref = rec_ref.summary()["solver/epot/iters_mean"]
    iters_amg = rec_amg.summary()["solver/epot/iters_mean"]
    assert iters_amg < iters_ref / 2, (
        f"AMG used {iters_amg:.1f} iterations vs {iters_ref:.1f} unpreconditioned; "
        "expected a large reduction"
    )
    assert rec_amg.summary()["solver/epot/n_not_converged"] == 0

    epot_ref = domain_ref.getBlock(0).epot
    epot_amg = domain_amg.getBlock(0).epot
    scale = float(epot_ref.abs().max())
    assert float((epot_amg - epot_ref).abs().max()) < 1e-5 * scale


def test_hierarchy_is_built_once_and_reused() -> None:
    """Setup must amortise: the epot matrix is constant, so rebuilding is waste."""
    _, sim = _make_sim(use_amg=True)
    sim.single_step()
    first = sim._epot_amg_hierarchy
    assert first is not None
    sim.single_step()
    assert sim._epot_amg_hierarchy is first
