"""Batched environments: B copies of one domain simulated together must reproduce B
separate simulations (same grid, different states and forcing)."""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

import phipict  # noqa: E402
from phipict.grid import shapes  # noqa: E402

DEV = torch.device("cuda")
DTYPE = torch.float64
TIGHT = {
    "advection_tol": phipict.SolverTolerance(rtol=1e-12, atol=1e-14),
    "pressure_tol": phipict.SolverTolerance(rtol=1e-12, atol=1e-14),
}


def make_channel(nx: int = 24, ny: int = 12) -> phipict.Domain:
    """Periodic 2D channel made of two blocks connected along x, walls at +-y."""
    viscosity = torch.tensor([0.05], dtype=DTYPE)
    domain = phipict.Domain(2, viscosity, name="channel", device=DEV, dtype=DTYPE)
    blocks = []
    for name in ("left", "right"):
        vel = torch.zeros(1, 2, ny, nx, dtype=DTYPE, device=DEV)
        blocks.append(domain.CreateBlock(velocity=vel, name=name))
    left, right = blocks
    left.ConnectBlock("+x", right, "-x", "-y")
    right.ConnectBlock("+x", left, "-x", "-y")
    for b in blocks:
        b.CloseBoundary("-y")
        b.CloseBoundary("+y")
    domain.PrepareSolve()
    return domain


def initial_state(domain: phipict.Domain, seed: int) -> list[torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [
        0.3 * torch.randn(b.velocity.shape[1:], generator=g, dtype=DTYPE).to(DEV)
        for b in domain.getBlocks()
    ]


def run_single(
    make, states, sources, steps, sim_cls=phipict.Simulation, sim_kwargs=None
):
    """One unbatched simulation per environment."""
    results = []
    for state, source in zip(states, sources, strict=True):
        domain = make()
        for block, v in zip(domain.getBlocks(), state, strict=True):
            block.velocity.copy_(v.unsqueeze(0))
            block.setVelocitySource(source.view(1, -1).clone())
        domain.UpdateDomainData()
        sim = sim_cls(domain=domain, **(sim_kwargs or {}))
        for _ in range(steps):
            sim.single_step()
        results.append([b.velocity.clone() for b in domain.getBlocks()])
    return results


def run_batched(
    make, states, sources, steps, sim_cls=phipict.Simulation, sim_kwargs=None
):
    """All environments in one batched simulation."""
    B = len(states)
    domain = make()
    domain.setBatchSize(B)
    domain.PrepareSolve()
    assert domain.getBatchSize() == B
    for i, block in enumerate(domain.getBlocks()):
        for b in range(B):
            block.velocity[b].copy_(states[b][i])
        block.setVelocitySource(
            torch.stack(sources).clone()
        )  # [B, dims]: per-env static forcing
    domain.UpdateDomainData()
    sim = sim_cls(domain=domain, **(sim_kwargs or {}))
    for _ in range(steps):
        sim.single_step()
    return [[blk.velocity[b].clone() for blk in domain.getBlocks()] for b in range(B)]


def assert_envs_match(batched, single, rtol):
    for b, (fields_b, fields_s) in enumerate(zip(batched, single, strict=True)):
        for fb, fs in zip(fields_b, fields_s, strict=True):
            err = float(
                torch.linalg.vector_norm(fb - fs.squeeze(0))
                / torch.linalg.vector_norm(fs)
            )
            assert err < rtol, f"environment {b}: relative difference {err:.3e}"


def test_batched_domain_shapes():
    domain = make_channel()
    n = domain.getTotalSize()
    nnz = domain.P.getNnz()
    domain.setBatchSize(4)
    domain.PrepareSolve()
    assert domain.getBatchSize() == 4
    assert domain.getBlocks()[0].velocity.shape[0] == 4
    assert domain.getBlocks()[0].pressure.shape[0] == 4
    assert domain.velocityRHS.numel() == 4 * 2 * n
    assert domain.pressureResult.numel() == 4 * n
    assert domain.P.getBatchSize() == 4 and domain.P.getNnz() == nnz
    assert domain.C.value.numel() == 4 * nnz


@pytest.mark.parametrize("B", [1, 3])
def test_batched_channel_matches_single(B):
    steps = 15
    states = [initial_state(make_channel(), seed) for seed in range(B)]
    sources = [
        torch.tensor([1.0 + 0.5 * b, 0.1 * b], dtype=DTYPE, device=DEV)
        for b in range(B)
    ]
    kwargs = dict(dt=0.02, substeps=1, non_orthogonal=False, **TIGHT)
    single = run_single(make_channel, states, sources, steps, sim_kwargs=kwargs)
    batched = run_batched(make_channel, states, sources, steps, sim_kwargs=kwargs)
    assert_envs_match(batched, single, rtol=1e-9)
    # the environments really differ
    if B > 1:
        assert float(torch.linalg.vector_norm(batched[0][0] - batched[1][0])) > 1e-3


def make_hartmann(nx: int = 40, ny: int = 20) -> phipict.Domain:
    y_weights = shapes.make_weights("simple", res=ny, grading=5, refinement="BOTH")
    grid = shapes.generate_grid_vertices_2D(
        [ny + 1, nx + 1],
        [(0.0, -1.0), (4.0, -1.0), (0.0, 1.0), (4.0, 1.0)],
        x_weights=y_weights,
        dtype=DTYPE,
    ).to(DEV)
    domain = phipict.Domain(
        2, torch.tensor([0.5], dtype=DTYPE), name="Hartmann", device=DEV, dtype=DTYPE
    )
    block = domain.CreateBlock(vertexCoordinates=grid, name="Block")
    block.CloseBoundary("-y")
    block.CloseBoundary("+y")
    block.MakePeriodic("x")
    domain.PrepareSolve()
    return domain


@pytest.mark.parametrize("use_amg", [False, True])
def test_batched_mhd_matches_single(use_amg):
    if use_amg:
        pytest.importorskip("pyamg")
    B, steps, ha = 3, 10, 5.0
    states = [initial_state(make_hartmann(), 10 + seed) for seed in range(B)]
    sources = [
        torch.tensor([20.0 * (1 + b), 0.0], dtype=DTYPE, device=DEV) for b in range(B)
    ]
    kwargs = dict(
        dt=0.01,
        substeps=1,
        non_orthogonal=False,
        stuart_number=torch.tensor(ha**2 / 2.0),
        e_b=torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE),
        potential_tol=phipict.SolverTolerance(rtol=1e-12, atol=1e-14),
        potential_use_preconditioner=use_amg,
        **TIGHT,
    )
    single = run_single(
        make_hartmann, states, sources, steps, phipict.MHDSimulation, kwargs
    )
    batched = run_batched(
        make_hartmann, states, sources, steps, phipict.MHDSimulation, kwargs
    )
    assert_envs_match(batched, single, rtol=1e-8)


def test_per_env_relative_tolerance():
    """A relative tolerance is resolved per environment: an environment with a tiny
    right-hand side is solved to its own relative accuracy, not the batch's."""
    from phipict import _C

    domain = make_channel()
    n = domain.getTotalSize()
    domain.setBatchSize(2)
    domain.PrepareSolve()
    sim = phipict.Simulation(domain=domain, dt=0.02, substeps=1, non_orthogonal=False)
    sim.single_step()  # assembles the pressure matrix of both environments
    rhs = torch.randn(2 * n, dtype=DTYPE, device=DEV)
    rhs[n:] *= 1e-6
    rhs = rhs - rhs.view(2, n).mean(dim=1, keepdim=True).repeat_interleave(n)
    x, _ = sim.linear_solve(
        domain.P, rhs, tol=phipict.SolverTolerance(rtol=1e-8), tag="test"
    )
    A = torch.sparse_csr_tensor(
        domain.P.row.long(),
        domain.P.index.long(),
        domain.P.value[: domain.P.getNnz()],
        (n, n),
    )
    A2 = torch.sparse_csr_tensor(
        domain.P.row.long(),
        domain.P.index.long(),
        domain.P.value[domain.P.getNnz() :],
        (n, n),
    )
    for b, Ab in enumerate((A, A2)):
        r = rhs[b * n : (b + 1) * n] - (
            Ab @ x[b * n : (b + 1) * n].unsqueeze(1)
        ).squeeze(1)
        rel = float(
            torch.linalg.vector_norm(r)
            / torch.linalg.vector_norm(rhs[b * n : (b + 1) * n])
        )
        assert rel < 1e-7, f"environment {b}: relative residual {rel:.2e}"
    assert _C.GetKrylovSettings()["enabled"]
    assert math.isfinite(float(x.abs().max()))


def _poisson_like(nx: int, ny: int, B: int, seed: int):
    """B negative definite 5-point operators, one pattern, random coefficients."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = nx * ny
    idx = torch.arange(n).reshape(ny, nx)
    pairs = torch.cat(
        [
            torch.stack([idx[:, :-1].reshape(-1), idx[:, 1:].reshape(-1)]),
            torch.stack([idx[:-1].reshape(-1), idx[1:].reshape(-1)]),
        ],
        dim=1,
    )
    mats = []
    for _ in range(B):
        w = 0.5 + torch.rand(pairs.shape[1], generator=g, dtype=DTYPE)
        rows = torch.cat([pairs[0], pairs[1], torch.arange(n)])
        cols = torch.cat([pairs[1], pairs[0], torch.arange(n)])
        diag = (
            torch.zeros(n, dtype=DTYPE)
            .index_add_(0, pairs[0], w)
            .index_add_(0, pairs[1], w)
            + 1e-2
        )
        vals = torch.cat([w, w, -diag])  # negative definite like the pressure matrix
        mats.append(
            torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (n, n))
            .coalesce()
            .to_sparse_csr()
        )
    return mats


@pytest.mark.parametrize("per_env", [False, True], ids=["map", "per_env"])
def test_galerkin_map_matches_rap(per_env, monkeypatch):
    pytest.importorskip("pyamg")
    from phipict import _C
    from phipict.solvers import amg

    if per_env:  # operators too large for a map: one Galerkin product per env
        monkeypatch.setattr(amg, "GALERKIN_MAP_PRODUCTS", 0)

    mats = [m.to(DEV) for m in _poisson_like(20, 16, 3, seed=1)]
    first = _C.CSRmatrix(
        mats[0].values(), mats[0].col_indices().int(), mats[0].crow_indices().int()
    )
    base = amg.build_amg_hierarchy(first, max_coarse=10)
    h = amg.BatchedAMGHierarchy.build(
        base, mats[0].crow_indices(), mats[0].col_indices(), 3
    )
    assert all((m.G is None) == per_env for m in h.maps)
    h.refresh(torch.cat([m.values() for m in mats]))
    for b, A in enumerate(mats):
        for level, lvl in enumerate(base.levels[: len(h.maps)]):
            A = (lvl.R @ (A @ lvl.P)).to_sparse_csr()
            m = h.maps[level]
            dense = torch.zeros(A.shape, dtype=DTYPE, device=DEV)
            rows = torch.repeat_interleave(
                torch.arange(A.shape[0], device=DEV), m.crow[1:] - m.crow[:-1]
            )
            dense[rows, m.col] = h.values[level + 1][b]
            assert torch.allclose(dense, A.to_dense(), rtol=1e-10, atol=1e-12)


def test_batched_amg_solve_per_environment():
    pytest.importorskip("pyamg")
    from phipict import _C
    from phipict.solvers import amg

    B = 4
    mats = [m.to(DEV) for m in _poisson_like(40, 30, B, seed=2)]
    n = mats[0].shape[0]
    batched = _C.CSRmatrix(
        torch.cat([m.values() for m in mats]),
        mats[0].col_indices().int(),
        mats[0].crow_indices().int(),
    )
    h = amg.batched_hierarchy_for(batched, B, reuse_interpolation=False, max_coarse=10)
    b = torch.randn(B * n, dtype=DTYPE, device=DEV)
    b[n : 2 * n] *= 1e-4  # a very different scale: per-environment tolerances
    tols = [
        1e-10 * float(torch.linalg.vector_norm(b[i * n : (i + 1) * n])) / math.sqrt(n)
        for i in range(B)
    ]
    x = torch.zeros_like(b)
    infos = amg.amg_pcg_solve(b, x, h, tol=tols, max_iter=200)
    assert len(infos) == B and all(i.converged for i in infos)
    for i, A in enumerate(mats):
        r = b[i * n : (i + 1) * n] - (A @ x[i * n : (i + 1) * n].unsqueeze(1)).squeeze(
            1
        )
        assert float(torch.linalg.vector_norm(r)) / math.sqrt(n) < 2 * tols[i]
        assert infos[i].usedIterations < 60  # AMG, not plain CG iteration counts


def test_batched_pressure_amg_matches_single():
    pytest.importorskip("pyamg")
    B, steps = 3, 10
    states = [initial_state(make_channel(), 20 + seed) for seed in range(B)]
    sources = [
        torch.tensor([1.0 + 0.5 * b, 0.1 * b], dtype=DTYPE, device=DEV)
        for b in range(B)
    ]
    kwargs = {
        "dt": 0.02,
        "substeps": 1,
        "non_orthogonal": False,
        "pressure_use_amg": True,
        **TIGHT,
    }
    single = run_single(make_channel, states, sources, steps, sim_kwargs=kwargs)
    batched = run_batched(make_channel, states, sources, steps, sim_kwargs=kwargs)
    assert_envs_match(batched, single, rtol=1e-8)


def make_channel_3d(nx: int = 12, ny: int = 8, nz: int = 6) -> phipict.Domain:
    """Periodic 3D channel (walls at +-y) for the SGS test."""
    domain = phipict.Domain(
        3, torch.tensor([0.01], dtype=DTYPE), name="channel3d", device=DEV, dtype=DTYPE
    )
    vel = torch.zeros(1, 3, nz, ny, nx, dtype=DTYPE, device=DEV)
    block = domain.CreateBlock(velocity=vel, name="block")
    block.CloseBoundary("-y")
    block.CloseBoundary("+y")
    block.MakePeriodic("x")
    block.MakePeriodic("z")
    domain.PrepareSolve()
    return domain


@pytest.mark.parametrize("model", ["smagorinsky", "wale"])
def test_batched_sgs_matches_single(model):
    from phipict.simulation.sgs import append_sgs_viscosity_prep_fn

    B, steps = 2, 5
    states = [initial_state(make_channel_3d(), 30 + seed) for seed in range(B)]
    sources = [
        torch.tensor([1.0 + b, 0.0, 0.2 * b], dtype=DTYPE, device=DEV) for b in range(B)
    ]

    def sim_cls(domain, **kwargs):
        prep_fn: dict = {}
        append_sgs_viscosity_prep_fn(prep_fn, model, 0.17, 3, DTYPE, DEV)
        return phipict.Simulation(domain=domain, prep_fn=prep_fn, **kwargs)

    kwargs = {"dt": 0.02, "substeps": 1, "non_orthogonal": False, **TIGHT}
    single = run_single(make_channel_3d, states, sources, steps, sim_cls, kwargs)
    batched = run_batched(make_channel_3d, states, sources, steps, sim_cls, kwargs)
    assert_envs_match(batched, single, rtol=1e-9)


def test_batched_spatial_velocity_gradients():
    from phipict import _C

    domain = make_channel_3d()
    domain.setBatchSize(2)
    domain.PrepareSolve()
    block = domain.getBlocks()[0]
    torch.manual_seed(4)
    block.velocity.copy_(torch.randn_like(block.velocity))
    domain.UpdateDomainData()
    grads = _C.ComputeSpatialVelocityGradients(domain)
    for b in range(2):
        single = make_channel_3d()
        single.getBlocks()[0].velocity.copy_(block.velocity[b : b + 1])
        single.UpdateDomainData()
        ref = _C.ComputeSpatialVelocityGradients(single)
        for d in range(3):
            assert torch.allclose(grads[0][d][b], ref[0][d][0], rtol=1e-12, atol=1e-12)


def _grad_run(make, states, sources, steps, sim_cls, sim_kwargs, weights, stuart):
    """One (batched if ``len(states) > 1``) differentiable simulation; returns the
    loss gradients w.r.t. the initial velocities, the velocity sources and (for MHD)
    the Stuart number. The loss weights every environment differently."""
    B = len(states)
    domain = make()
    if B > 1:
        domain.setBatchSize(B)
        domain.PrepareSolve()
    v0 = [
        torch.stack([states[b][i] for b in range(B)]).requires_grad_(True)
        for i in range(len(domain.getBlocks()))
    ]
    src = torch.stack(sources).clone().requires_grad_(True)
    for block, v in zip(domain.getBlocks(), v0, strict=True):
        block.setVelocity(v.clone())
        block.setVelocitySource(src.clone())
    domain.UpdateDomainData()
    kwargs = dict(sim_kwargs)
    if stuart is not None:
        kwargs["stuart_number"] = stuart
    sim = sim_cls(domain=domain, differentiable=True, **kwargs)
    for _ in range(steps):
        sim.single_step()
    w = torch.tensor(weights, dtype=DTYPE, device=DEV).view(B, 1, 1, 1)
    loss = sum((w * blk.velocity**2).sum() for blk in domain.getBlocks())
    inputs = [*v0, src] + ([stuart] if stuart is not None else [])
    grads = torch.autograd.grad(loss, inputs)
    domain.Detach()
    return [g.detach() for g in grads]


def _assert_batched_grads_match(make, sim_cls, sim_kwargs, B, steps, mhd):
    states = [initial_state(make(), 20 + seed) for seed in range(B)]
    sources = [
        torch.tensor([2.0 * (1 + b), 0.1 * b], dtype=DTYPE, device=DEV)
        for b in range(B)
    ]
    weights = [1.0 + b for b in range(B)]

    def stuart():
        return torch.tensor([12.5], dtype=DTYPE, requires_grad=True) if mhd else None

    batched = _grad_run(
        make, states, sources, steps, sim_cls, sim_kwargs, weights, stuart()
    )
    single = [
        _grad_run(
            make,
            [states[b]],
            [sources[b]],
            steps,
            sim_cls,
            sim_kwargs,
            [weights[b]],
            stuart(),
        )
        for b in range(B)
    ]
    n_fields = len(states[0])
    names = [f"velocity[{i}]" for i in range(n_fields)] + ["source"]
    for k, name in enumerate(names):
        expected = torch.cat([single[b][k] for b in range(B)])
        err = float(
            torch.linalg.vector_norm(batched[k] - expected)
            / torch.linalg.vector_norm(expected)
        )
        assert err < 1e-7, f"d loss / d {name}: relative difference {err:.3e}"
        # per-environment gradients really differ
        assert float(torch.linalg.vector_norm(expected[0] - expected[1])) > 0
    if mhd:
        # the Stuart number is shared: its gradient sums over the environments
        expected = sum(single[b][-1] for b in range(B))
        err = float((batched[-1] - expected).abs().max() / expected.abs().max())
        assert err < 1e-7, f"d loss / d stuart: relative difference {err:.3e}"


def test_batched_gradients_match_single():
    kwargs = dict(dt=0.02, substeps=1, non_orthogonal=False, **TIGHT)
    _assert_batched_grads_match(
        make_channel, phipict.Simulation, kwargs, B=3, steps=4, mhd=False
    )


def test_batched_mhd_gradients_match_single():
    kwargs = dict(
        dt=0.01,
        substeps=1,
        non_orthogonal=False,
        e_b=torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE),
        potential_tol=phipict.SolverTolerance(rtol=1e-12, atol=1e-14),
        potential_use_preconditioner=False,
        **TIGHT,
    )
    _assert_batched_grads_match(
        make_hartmann, phipict.MHDSimulation, kwargs, B=2, steps=3, mhd=True
    )


def make_duct(nx: int = 24, ny: int = 10) -> phipict.Domain:
    """2D duct with a fixed inflow at -x, a varying outflow at +x and walls at +-y."""
    y_weights = shapes.make_weights("simple", res=ny, grading=3, refinement="BOTH")
    grid = shapes.generate_grid_vertices_2D(
        [ny + 1, nx + 1],
        [(0.0, -1.0), (3.0, -1.0), (0.0, 1.0), (3.0, 1.0)],
        x_weights=y_weights,
        dtype=DTYPE,
    ).to(DEV)
    domain = phipict.Domain(
        2, torch.tensor([0.05], dtype=DTYPE), name="duct", device=DEV, dtype=DTYPE
    )
    block = domain.CreateBlock(vertexCoordinates=grid, name="duct")
    inflow = torch.zeros(1, 2, ny, 1, dtype=DTYPE, device=DEV)
    inflow[0, 0] = 1.0
    block.CloseBoundary("-x", inflow)
    block.CloseBoundary("+x", inflow.clone())
    block.getBoundary("+x").makeVelocityVarying()
    block.CloseBoundary("-y")
    block.CloseBoundary("+y")
    domain.PrepareSolve()
    return domain


def test_batched_flux_balance_and_stacking():
    """Per-environment inflows are balanced per environment by the shared outflow,
    and stacked domains keep the state of every environment."""
    from phipict.batching import stack_domains
    from phipict.core.piso_simulation import (
        balance_boundary_fluxes,
        get_fixed_boundary_fluxes,
    )

    B = 3
    domains = [make_duct() for _ in range(B)]
    for b, d in enumerate(domains):
        d.getBlocks()[0].velocity.fill_(0.1 * b)
    domain = stack_domains(domains)
    block = domain.getBlocks()[0]
    assert domain.getBatchSize() == B
    for b in range(B):
        assert torch.allclose(
            block.velocity[b], torch.full_like(block.velocity[b], 0.1 * b)
        )

    inflow_bound, outflow_bound = block.getBoundary("-x"), block.getBoundary("+x")
    scales = torch.tensor([0.5, 1.0, 2.0], dtype=DTYPE, device=DEV)
    inflow = inflow_bound.velocity.expand(B, *inflow_bound.velocity.shape[1:])
    inflow_bound.setVelocity((inflow * scales.view(B, 1, 1, 1)).contiguous())
    balance_boundary_fluxes(domain, [outflow_bound])

    fixed = [(0, inflow_bound)]
    free = [(1, outflow_bound)]
    net = get_fixed_boundary_fluxes(fixed) + get_fixed_boundary_fluxes(free)
    assert net.shape == (B,)
    assert float(net.abs().max()) < 1e-12
    # the outflow carries each environment's own inflow
    out_flux = get_fixed_boundary_fluxes(free)
    assert torch.allclose(out_flux / out_flux[1], scales / scales[1])


def _scaled_states(make, B, seed0=0):
    """Initial states whose speed grows with the environment index, so the
    environments need different numbers of adaptive substeps."""
    return [
        [(1.0 + 3.0 * b) * v for v in initial_state(make(), seed0 + b)]
        for b in range(B)
    ]


def test_max_velocity_per_env():
    domain = make_channel()
    domain.setBatchSize(3)
    domain.PrepareSolve()
    for b in range(3):
        for block in domain.getBlocks():
            block.velocity[b].fill_(0.5 * (b + 1))
        domain.getBlocks()[0].velocity[b, 1, 0, 0] = -2.0 * (b + 1)
    max_vel = domain.getMaxVelocityPerEnv(True, False)
    assert max_vel.shape == (3,)
    assert torch.allclose(max_vel.cpu(), torch.tensor([2.0, 4.0, 6.0], dtype=DTYPE))
    assert float(domain.getMaxVelocity(True, False)) == float(max_vel.max())
    with pytest.raises(RuntimeError):
        domain.setTimeStep(torch.ones(2, dtype=DTYPE))


@pytest.mark.parametrize("per_env", [True, False])
def test_batched_adaptive_matches_single(per_env):
    """With per-environment adaptive substeps, every environment of a batch takes
    the substeps it takes on its own; with shared substeps the slow ones do not."""
    B, steps = 3, 4
    states = _scaled_states(make_hartmann, B)
    sources = [torch.tensor([0.5, 0.0], dtype=DTYPE, device=DEV)] * B
    kwargs = dict(
        dt=0.2,
        substeps="ADAPTIVE",
        adaptive_CFL=0.8,
        non_orthogonal=False,
        **TIGHT,
    )
    single = run_single(make_hartmann, states, sources, steps, sim_kwargs=kwargs)
    batched = run_batched(
        make_hartmann,
        states,
        sources,
        steps,
        sim_kwargs=dict(kwargs, adaptive_CFL_per_env=per_env),
    )
    if per_env:
        assert_envs_match(batched, single, rtol=1e-9)
    else:
        # the fastest environment sets the substeps of all
        assert_envs_match(batched[-1:], single[-1:], rtol=1e-9)
        with pytest.raises(AssertionError):
            assert_envs_match(batched[:1], single[:1], rtol=1e-6)


def test_batched_adaptive_substep_counts():
    """The batch takes the substeps of its slowest-stepping environment, and the
    total time advances by one step."""
    B = 3
    states = _scaled_states(make_hartmann, B)
    kwargs = {"dt": 0.2, "substeps": "ADAPTIVE", "non_orthogonal": False}

    def load(domain, env_states):
        for i, block in enumerate(domain.getBlocks()):
            for b, state in enumerate(env_states):
                block.velocity[b].copy_(state[i])
        domain.UpdateDomainData()

    counts = []
    for state in states:
        domain = make_hartmann()
        load(domain, [state])
        sim = phipict.Simulation(domain=domain, **kwargs)
        sim.single_step()
        counts.append(sim.total_step)
    assert len(set(counts)) > 1, "environments must need different substep counts"

    domain = make_hartmann()
    domain.setBatchSize(B)
    domain.PrepareSolve()
    load(domain, states)
    sim = phipict.Simulation(domain=domain, **kwargs)
    sim.single_step()
    assert sim.total_step == max(counts)
    assert sim.total_time == pytest.approx(0.2)


def test_batched_adaptive_mhd_matches_single():
    B, steps, ha = 3, 3, 5.0
    states = _scaled_states(make_hartmann, B, seed0=10)
    sources = [torch.tensor([20.0, 0.0], dtype=DTYPE, device=DEV)] * B
    kwargs = dict(
        dt=0.05,
        substeps="ADAPTIVE",
        non_orthogonal=False,
        stuart_number=torch.tensor(ha**2 / 2.0),
        e_b=torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE),
        potential_tol=phipict.SolverTolerance(rtol=1e-12, atol=1e-14),
        **TIGHT,
    )
    single = run_single(
        make_hartmann, states, sources, steps, phipict.MHDSimulation, kwargs
    )
    batched = run_batched(
        make_hartmann, states, sources, steps, phipict.MHDSimulation, kwargs
    )
    assert_envs_match(batched, single, rtol=1e-8)


def _run_duct(scales, steps, batched):
    """Ducts with inflows scaled per environment and an advective outflow,
    run as one batch or one simulation per scale."""
    from phipict.core.hooks import Hook, Hooks
    from phipict.core.piso_simulation import update_advective_boundaries

    def run(domain, scale):
        block = domain.getBlocks()[0]
        inflow = block.getBoundary("-x")
        B = domain.getBatchSize()
        vel = inflow.velocity.expand(B, *inflow.velocity.shape[1:])
        inflow.setVelocity((vel * scale.view(-1, 1, 1, 1)).contiguous())
        outflow = block.getBoundary("+x")
        outflow.setVelocity((vel * scale.view(-1, 1, 1, 1)).contiguous())
        # the advective update needs a varying outflow scalar
        channels, ny = block.passiveScalar.shape[1], vel.shape[2]
        outflow.setPassiveScalar(block.passiveScalar.new_zeros(1, channels, ny, 1))
        block.velocity.copy_(
            block.velocity.new_tensor([1.0, 0.0]).view(1, 2, 1, 1)
            * scale.view(-1, 1, 1, 1)
        )
        domain.UpdateDomainData()
        char_vel = torch.cat(
            [scale.view(-1, 1), torch.zeros_like(scale).view(-1, 1)], 1
        )

        def outflow_fn(domain, time_step, **kwargs):
            update_advective_boundaries(domain, [outflow], char_vel, time_step.to(DEV))

        sim = phipict.Simulation(
            domain=domain,
            dt=0.3,
            substeps="ADAPTIVE",
            non_orthogonal=False,
            hooks=Hooks().append(Hook.PRE, outflow_fn),
            **TIGHT,
        )
        for _ in range(steps):
            sim.single_step()
        return [
            [block.velocity[b].clone(), outflow.velocity[b].clone()]
            for b in range(domain.getBatchSize())
        ]

    if batched:
        domain = make_duct()
        domain.setBatchSize(len(scales))
        domain.PrepareSolve()
        return run(domain, scales)
    return [run(make_duct(), scale.view(1))[0] for scale in scales]


def test_batched_adaptive_outflow_matches_single():
    """Per-environment substep sizes reach the advective outflow update, and the
    outflow of a finished environment is restored."""
    scales = torch.tensor([0.5, 1.0, 3.0], dtype=DTYPE, device=DEV)
    single = _run_duct(scales, steps=4, batched=False)
    batched = _run_duct(scales, steps=4, batched=True)
    assert_envs_match(batched, single, rtol=1e-9)
