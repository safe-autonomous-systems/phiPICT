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

"""Tests for ``save_domain``/``load_domain`` on the MHD duct domain."""

from typing import Any

import pytest
import torch
from conftest import import_fluidgym_on_phipict

from phipict import _C
from phipict.io.domain_io import (  # noqa: E402
    load_domain,
    save_domain,
)

CUDA_DEVICE = torch.device("cuda")

DTYPE = torch.float64

# Small enough to build and solve in a second, large enough that every boundary
# face has an interior neighbour and the profiles are not degenerate
RES_X, RES_Y, RES_Z = 6, 4, 4
LX, LY, LZ = 3.0, 7.0, 2.0

REYNOLDS = 5000.0
HARTMANN = 200.0
PRANDTL = 0.025

FACES = ["-x", "+x", "-y", "+y", "-z", "+z"]


def _make_domain(hartmann_Cw: float, heat_both_walls: bool) -> _C.Domain:
    """Build the tiny MHD duct domain with the env's settings."""
    viscosity = torch.tensor([1.0 * (LZ / 2) / REYNOLDS], dtype=DTYPE)
    grid = import_fluidgym_on_phipict("fluidgym.envs.mhd.grid")
    return grid.make_mhd_domain(
        ndims=3,
        Lx=LX,
        Ly=LY,
        Lz=LZ,
        viscosity=viscosity,
        prandtl_number=PRANDTL,
        T_in=0.0,
        heat_flux_walls=1.0,
        heat_both_walls=heat_both_walls,
        Cw=hartmann_Cw,
        res_x=RES_X,
        res_y=RES_Y,
        res_z=RES_Z,
        x_grading_type="chebyshev_identity",
        x_grading=0.0,
        y_grading_type="tanh",
        y_grading=3.3,
        z_grading_type="chebyshev_identity",
        z_grading=0.96,
        T_wall_transition_length=1.0,
        dtype=DTYPE,
        device=CUDA_DEVICE,
        init_with_noise=False,
    )


def _fill_fields(domain: _C.Domain, seed: int = 0) -> None:
    """Put distinguishable data in every saved cell field, ``epot`` included.

    ``epot`` is only materialized by the potential solve, so a freshly built
    domain has none -- but a checkpoint written mid-episode does, and
    ``MHDEnv._load_initial_domain`` indexes ``block.epot`` unconditionally.
    """
    block = domain.getBlock(0)
    gen = torch.Generator(device=CUDA_DEVICE).manual_seed(seed)

    def rand_like(t: torch.Tensor) -> torch.Tensor:
        return torch.rand(t.shape, generator=gen, dtype=t.dtype, device=t.device)

    block.setVelocity(rand_like(block.velocity).contiguous())
    block.setPressure(rand_like(block.pressure).contiguous())
    block.setPassiveScalar(rand_like(block.passiveScalar).contiguous())
    block.setEpot(rand_like(block.pressure).contiguous())


def _assert_tensors_equal(a: Any, b: Any, what: str) -> None:
    if a is None or b is None:
        assert (a is None) == (b is None), f"{what}: one side is None"
        return
    assert a.shape == b.shape, f"{what}: shape {a.shape} != {b.shape}"
    assert a.dtype == b.dtype, f"{what}: dtype {a.dtype} != {b.dtype}"
    assert torch.equal(a.cpu(), b.cpu()), f"{what}: values differ"


def _assert_boundaries_equal(block_ref: _C.Block, block_new: _C.Block) -> None:
    for face in FACES:
        bound_ref = block_ref.getBoundary(face)
        bound_new = block_new.getBoundary(face)

        assert bound_ref.type == bound_new.type, f"{face}: boundary type"

        # Velocity: DIRICHLET everywhere here, but +x is made *varying* for the
        # advective outflow update, which shows up as a full-face tensor.
        assert bound_ref.velocityType == bound_new.velocityType, f"{face}: velocityType"
        assert bound_ref.isVelocityStatic == bound_new.isVelocityStatic, (
            f"{face}: isVelocityStatic (varying inflow/outflow lost)"
        )
        _assert_tensors_equal(
            bound_ref.velocity, bound_new.velocity, f"{face}: velocity"
        )

        # Temperature: Dirichlet at the inlet/outlet, Neumann (heat flux) on the
        # Shercliff walls, Neumann (adiabatic) on the Hartmann walls.
        assert bound_ref.hasPassiveScalar() == bound_new.hasPassiveScalar(), (
            f"{face}: hasPassiveScalar"
        )
        assert list(bound_ref.passiveScalarTypes) == list(
            bound_new.passiveScalarTypes
        ), f"{face}: passiveScalarTypes"
        _assert_tensors_equal(
            bound_ref.passiveScalar, bound_new.passiveScalar, f"{face}: passiveScalar"
        )

        # Electric potential: thin-wall Robin BC on the walls, open (current
        # leaving the domain) on the streamwise faces.
        assert bound_ref.getPotentialBC() == bound_new.getPotentialBC(), (
            f"{face}: potential BC"
        )
        assert bound_ref.getPotentialCw() == bound_new.getPotentialCw(), (
            f"{face}: potential cw"
        )
        assert bound_ref.hasPotentialTypes() == bound_new.hasPotentialTypes(), (
            f"{face}: per-cell potential BC"
        )
        _assert_tensors_equal(
            bound_ref.potentialTypes, bound_new.potentialTypes, f"{face}: types"
        )
        _assert_tensors_equal(
            bound_ref.potentialValues, bound_new.potentialValues, f"{face}: values"
        )


@pytest.mark.parametrize("hartmann_Cw", [0.0, 0.05], ids=["insulating", "thin_wall"])
@pytest.mark.parametrize(
    "heat_both_walls", [False, True], ids=["one_wall", "both_walls"]
)
def test_mhd_domain_save_load_roundtrip(
    tmp_path, monkeypatch, hartmann_Cw: float, heat_both_walls: bool
) -> None:
    """A saved MHD duct domain reloads with identical fields and BCs."""
    monkeypatch.chdir(tmp_path)  # make_mhd_domain writes grid/profile pdfs

    domain = _make_domain(hartmann_Cw=hartmann_Cw, heat_both_walls=heat_both_walls)
    _fill_fields(domain)

    path = tmp_path / "domain"
    save_domain(domain, path)
    loaded = load_domain(path, dtype=DTYPE, device=CUDA_DEVICE)

    assert loaded.getSpatialDims() == domain.getSpatialDims()
    assert loaded.getPassiveScalarChannels() == domain.getPassiveScalarChannels()
    _assert_tensors_equal(domain.viscosity, loaded.viscosity, "domain viscosity")
    assert loaded.hasPassiveScalarViscosity() == domain.hasPassiveScalarViscosity()
    if domain.hasPassiveScalarViscosity():
        _assert_tensors_equal(
            domain.passiveScalarViscosity,
            loaded.passiveScalarViscosity,
            "scalar viscosity",
        )

    assert len(loaded.getBlocks()) == len(domain.getBlocks())
    block, block_loaded = domain.getBlock(0), loaded.getBlock(0)

    _assert_tensors_equal(
        block.vertexCoordinates, block_loaded.vertexCoordinates, "vertexCoordinates"
    )
    _assert_tensors_equal(block.velocity, block_loaded.velocity, "velocity")
    _assert_tensors_equal(block.pressure, block_loaded.pressure, "pressure")
    _assert_tensors_equal(
        block.passiveScalar, block_loaded.passiveScalar, "passiveScalar"
    )
    assert block_loaded.hasEpot(), "epot was not restored"
    _assert_tensors_equal(block.epot, block_loaded.epot, "epot")

    _assert_boundaries_equal(block, block_loaded)

    # The reloaded domain must be usable, not just structurally equal.
    loaded.PrepareSolve()


def test_mhd_domain_roundtrip_runs_a_step(tmp_path, monkeypatch) -> None:
    """A reloaded thin-wall domain still solves one PISO step with Lorentz force."""
    from phipict.simulation.mhd import MHDSimulation

    monkeypatch.chdir(tmp_path)

    domain = _make_domain(hartmann_Cw=0.05, heat_both_walls=False)

    path = tmp_path / "domain"
    save_domain(domain, path)
    loaded = load_domain(path, dtype=DTYPE, device=CUDA_DEVICE)
    loaded.PrepareSolve()

    sim = MHDSimulation(
        domain=loaded,
        dt=1e-3,
        stuart_number=torch.tensor(HARTMANN**2 / REYNOLDS),
        e_b=torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE),
        substeps=1,
        corrector_steps=2,
        advection_tol=1e-5,
        pressure_tol=5e-6,
        potential_tol=5e-6,
        non_orthogonal=False,
        potential_normalize=True,
        potential_use_BiCG=False,
        potential_solve_max_iter=1000,
    )
    sim.make_divergence_free(max_iter=1000)
    sim.single_step()

    block = loaded.getBlock(0)
    assert torch.isfinite(block.velocity).all()
    assert torch.isfinite(block.passiveScalar).all()
    assert block.hasEpot() and torch.isfinite(block.epot).all()
