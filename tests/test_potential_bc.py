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

"""Tests for the typed boundary-condition API and per-cell electric potential BCs."""

import json

import pytest
import torch

import phipict
from phipict import _C, Face, PotentialBC, bc
from phipict.core import piso_diff
from phipict.grid import shapes
from phipict.grid.helpers import get_cell_size
from phipict.io.domain_io import load_domain, save_domain
from phipict.simulation.mhd import _epot_matrix_is_anchored

DEVICE = torch.device("cuda")
DTYPE = torch.float64
NX, NY, NZ = 8, 6, 5
CW = 0.1

SPECS = {
    PotentialBC.INSULATING: bc.Potential.Insulating(),
    PotentialBC.OPEN: bc.Potential.Open(),
    PotentialBC.DIRICHLET: bc.Potential.Dirichlet(),
    PotentialBC.THIN_WALL: bc.Potential.ThinWall(cw=CW),
    PotentialBC.CURRENT: bc.Potential.Current(),
}


def _grid(nx: int = NX, ny: int = NY, nz: int = NZ) -> torch.Tensor:
    """A graded (curvilinear-transform) duct grid, vertices [1, 3, nz+1, ny+1, nx+1]."""
    y_weights = shapes.make_weights("simple", res=ny, grading=5, refinement="BOTH")
    grid = shapes.generate_grid_vertices_2D(
        [ny + 1, nx + 1],
        [(0.0, -1.0), (2.0, -1.0), (0.0, 1.0), (2.0, 1.0)],
        x_weights=y_weights,
        dtype=DTYPE,
    )
    grid = shapes.extrude_grid_z(grid, res_z=nz, start_z=-1.0, end_z=1.0)
    return grid.to(DEVICE).contiguous()


def _domain(
    set_bcs=None, nx: int = NX, ny: int = NY, nz: int = NZ, batch: int = 1
) -> _C.Domain:
    """Duct periodic in x, walls at ±y and ±z; ``set_bcs(block)`` sets the BCs."""
    domain = phipict.Domain(
        3, torch.tensor([0.1], dtype=DTYPE), name="D", device=DEVICE, dtype=DTYPE
    )
    block = domain.CreateBlock(vertexCoordinates=_grid(nx, ny, nz), name="B")
    for face in ("-y", "+y", "-z", "+z"):
        block.CloseBoundary(face)
    block.MakePeriodic("x")
    if set_bcs is not None:
        set_bcs(block)
    if batch > 1:
        domain.setBatchSize(batch)
    domain.PrepareSolve()
    domain.SetupEpotOnDomain(0, False)
    return domain


def _operators(domain: _C.Domain, seed: int = 0) -> dict[str, torch.Tensor]:
    """Epot matrix, Poisson RHS and face-based current density for random fields."""
    n = domain.getTotalSize() * domain.getBatchSize()
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    ucb = torch.randn(3 * n, dtype=DTYPE, device=DEVICE, generator=gen)
    phi = torch.randn(n, dtype=DTYPE, device=DEVICE, generator=gen)
    assert domain.Epot is not None
    return {
        "row": domain.Epot.row.clone(),
        "index": domain.Epot.index.clone(),
        "value": domain.Epot.value.clone(),
        "rhs": _C.ComputeEpotRHS(domain, ucb),
        "J": _C.ComputeCurrentDensityFaceBased(domain, phi, ucb).reshape(
            domain.getBatchSize(), 3, -1
        ),
    }


def _face_y_minus_mask() -> torch.Tensor:
    """Mask over the -y face cells [NZ, NX]: the lower half in x."""
    mask = torch.zeros(NZ, NX, dtype=torch.bool)
    mask[:, : NX // 2] = True
    return mask


def _flat(z: int, y: int, x: int) -> int:
    return (z * NY + y) * NX + x


@pytest.mark.parametrize("bc_type", list(SPECS))
def test_uniform_mask_matches_face_type(bc_type: PotentialBC) -> None:
    """A per-cell mask that is uniform gives exactly the operators of the face type."""
    cw = CW if bc_type == PotentialBC.THIN_WALL else None

    def uniform(block: _C.Block) -> None:
        bc.set_bc(block, Face.Y_MINUS, SPECS[bc_type])

    def masked(block: _C.Block) -> None:
        types = torch.full((1, 1, NZ, 1, NX), int(bc_type), dtype=torch.int8)
        block.getBoundary(Face.Y_MINUS).setPotentialTypes(types, cw=cw)

    ref, new = _operators(_domain(uniform)), _operators(_domain(masked))
    for name in ref:
        assert torch.equal(ref[name], new[name]), name


PAIRS = [
    (PotentialBC.DIRICHLET, PotentialBC.INSULATING),
    (PotentialBC.DIRICHLET, PotentialBC.OPEN),
    (PotentialBC.OPEN, PotentialBC.INSULATING),
    (PotentialBC.THIN_WALL, PotentialBC.INSULATING),
    (PotentialBC.THIN_WALL, PotentialBC.DIRICHLET),
    (PotentialBC.CURRENT, PotentialBC.INSULATING),
    (PotentialBC.CURRENT, PotentialBC.DIRICHLET),
    (PotentialBC.THIN_WALL, PotentialBC.CURRENT),
]


@pytest.mark.parametrize(
    ("inside", "outside"), PAIRS, ids=[f"{a.name}-{b.name}" for a, b in PAIRS]
)
def test_mixed_face_composes_uniform_faces(
    inside: PotentialBC, outside: PotentialBC
) -> None:
    """Each cell of a mixed face behaves exactly like its own type's uniform face.

    Matrix rows, the Poisson RHS and the current density of a cell depend only on
    the condition of its own face cell, except the thin-wall surface Laplacian,
    which links tangential neighbours: at the patch border it is cut, so the two
    border columns are checked for that separately.
    """
    mask = _face_y_minus_mask()

    def mixed(block: _C.Block) -> None:
        bc.set_bc(block, Face.Y_MINUS, SPECS[outside])
        bc.set_bc(block, Face.Y_MINUS, SPECS[inside], where=mask)

    def uniform(bc_type: PotentialBC):
        return lambda block: bc.set_bc(block, Face.Y_MINUS, SPECS[bc_type])

    new = _operators(_domain(mixed))
    ref_in = _operators(_domain(uniform(inside)))
    ref_out = _operators(_domain(uniform(outside)))
    assert torch.equal(new["row"], ref_in["row"])
    assert torch.equal(new["index"], ref_in["index"])

    thin_wall = PotentialBC.THIN_WALL in (inside, outside)
    # x is periodic, so the patch also borders on itself across the wrap
    border = {NX // 2 - 1, NX // 2, 0, NX - 1}
    row = new["row"].cpu()
    for z in range(NZ):
        for y in range(NY):
            for x in range(NX):
                cell = _flat(z, y, x)
                ref = ref_in if (y == 0 and mask[z, x]) else ref_out
                if y == 0 and thin_wall and x in border:
                    continue
                if y != 0:
                    # away from the face both references agree
                    ref = ref_in
                start, end = int(row[cell]), int(row[cell + 1])
                assert torch.equal(new["value"][start:end], ref["value"][start:end]), (
                    f"matrix row of cell {(z, y, x)}"
                )
                assert new["rhs"][cell] == ref["rhs"][cell], f"rhs of {(z, y, x)}"
                assert torch.equal(new["J"][..., cell], ref["J"][..., cell]), (
                    f"J of {(z, y, x)}"
                )

    if thin_wall:
        # the thin-wall correction is conservative: every row still sums like the
        # uniform face of its own type (zero unless the cell is Dirichlet)
        for z in range(NZ):
            for x in border:
                cell = _flat(z, 0, x)
                start, end = int(row[cell]), int(row[cell + 1])
                ref = ref_in if mask[z, x] else ref_out
                torch.testing.assert_close(
                    new["value"][start:end].sum(),
                    ref["value"][start:end].sum(),
                    rtol=0,
                    atol=1e-12,
                )


def test_mixed_dirichlet_face_anchors_and_runs() -> None:
    """Half φ=0, half insulating: the matrix is anchored and the MHD solve converges."""
    mask = _face_y_minus_mask()

    def mixed(block: _C.Block) -> None:
        bc.set_bc(block, Face.Y_MINUS, bc.Potential.Dirichlet(), where=mask)

    domain = _domain(mixed)
    assert _epot_matrix_is_anchored(domain)
    assert not _epot_matrix_is_anchored(_domain())

    block = domain.getBlock(0)
    torch.manual_seed(0)
    velocity = 0.1 * torch.randn_like(block.velocity)
    velocity[:, 0] += 1.0
    block.setVelocity(velocity.contiguous())
    domain.UpdateDomainData()
    sim = phipict.MHDSimulation(
        domain=domain,
        dt=5e-3,
        stuart_number=torch.tensor(20.0),
        e_b=torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE),
        substeps=1,
        non_orthogonal=False,
        potential_return_best_result=False,
        potential_tol=phipict.SolverTolerance(rtol=1e-10, atol=1e-14),
    )
    for _ in range(3):
        assert sim.single_step()
    assert torch.isfinite(block.velocity).all()
    assert block.epot is not None and torch.isfinite(block.epot).all()

    # φ is pinned by the Dirichlet cells, so it must not be mean-normalised away
    phi = domain.epotResult
    assert phi is not None and phi.abs().max() > 0
    assert phi.mean().abs() > 1e-8


def test_mixed_face_adjoint() -> None:
    """RHS and current density gradients match finite differences on a mixed face."""
    nx, ny, nz = 4, 4, 3
    mask = torch.zeros(nz, nx, dtype=torch.bool)
    mask[:, :2] = True

    def mixed(block: _C.Block) -> None:
        bc.set_bc(block, Face.Y_MINUS, bc.Potential.ThinWall(cw=CW))
        bc.set_bc(block, Face.Y_MINUS, bc.Potential.Dirichlet(), where=mask)
        bc.set_bc(block, Face.Z_PLUS, bc.Potential.Open())

    domain = _domain(mixed, nx=nx, ny=ny, nz=nz)
    n = domain.getTotalSize()
    gen = torch.Generator(device=DEVICE).manual_seed(3)
    ucb = torch.randn(3 * n, dtype=DTYPE, device=DEVICE, generator=gen)
    phi = torch.randn(n, dtype=DTYPE, device=DEVICE, generator=gen)
    ucb.requires_grad_(True)
    phi.requires_grad_(True)

    assert torch.autograd.gradcheck(
        lambda u: piso_diff.ComputeEpotRHS(domain, u),
        (ucb,),
        eps=1e-6,
        atol=1e-6,
        nondet_tol=1e-12,
    )
    assert torch.autograd.gradcheck(
        lambda p, u: piso_diff.ComputeCurrentDensityFaceBased(domain, p, u),
        (phi, ucb),
        eps=1e-6,
        atol=1e-6,
        nondet_tol=1e-12,
    )


def test_where_all_true_collapses_to_face_type() -> None:
    """A mask covering the whole face leaves no per-cell types behind."""

    def set_bcs(block: _C.Block) -> None:
        bc.set_bc(
            block, Face.Y_MINUS, bc.Potential.Dirichlet(), where=_face_y_minus_mask()
        )
        bc.set_bc(
            block, Face.Y_MINUS, bc.Potential.Dirichlet(), where=~_face_y_minus_mask()
        )

    domain = _domain(set_bcs)
    block = domain.getBlock(0)
    bound = block.getBoundary(Face.Y_MINUS)
    assert not bound.hasPotentialTypes()
    assert bound.getPotentialBC() == PotentialBC.DIRICHLET
    assert bc.get_bc(block, Face.Y_MINUS, bc.Potential) == bc.Potential.Dirichlet()


def test_get_bc() -> None:
    mask = _face_y_minus_mask()

    def set_bcs(block: _C.Block) -> None:
        bc.set_bc(block, Face.Y_PLUS, bc.Potential.ThinWall(cw=CW))
        bc.set_bc(block, Face.Y_MINUS, bc.Potential.Open(), where=mask)

    domain = _domain(set_bcs)
    block = domain.getBlock(0)
    assert bc.get_bc(block, Face.Y_PLUS, bc.Potential) == bc.Potential.ThinWall(cw=CW)
    assert bc.get_bc(block, Face.Z_MINUS, bc.Potential) == bc.Potential.Insulating()
    types = bc.get_bc(block, Face.Y_MINUS, bc.Potential)
    assert isinstance(types, torch.Tensor)
    expected = torch.where(mask, int(PotentialBC.OPEN), int(PotentialBC.INSULATING))
    assert torch.equal(types.cpu(), expected.to(torch.int8))
    velocity = bc.get_bc(block, Face.Y_MINUS, bc.Velocity)
    assert isinstance(velocity, bc.Velocity.Dirichlet)


def test_thin_wall_cw_conflict() -> None:
    mask = _face_y_minus_mask()

    def set_bcs(block: _C.Block) -> None:
        bc.set_bc(block, Face.Y_MINUS, bc.Potential.ThinWall(cw=CW))
        # overwriting only part of the thin wall with another cw is ambiguous
        bc.set_bc(block, Face.Y_MINUS, bc.Potential.ThinWall(cw=2 * CW), where=mask)

    with pytest.raises(ValueError, match="one cw"):
        _domain(set_bcs)

    def keep_cw(block: _C.Block) -> None:
        bc.set_bc(block, Face.Y_MINUS, bc.Potential.ThinWall(cw=CW))
        bc.set_bc(block, Face.Y_MINUS, bc.Potential.Dirichlet(), where=mask)

    domain = _domain(keep_cw)
    bound = domain.getBlock(0).getBoundary(Face.Y_MINUS)
    assert bound.hasPotentialTypes() and bound.getPotentialCw() == CW


def test_set_bc_errors() -> None:
    domain = _domain()
    block = domain.getBlock(0)
    with pytest.raises(ValueError, match="FIXED"):
        bc.set_bc(block, Face.X_MINUS, bc.Potential.Dirichlet())  # periodic
    with pytest.raises(ValueError, match="cells"):
        bc.set_bc(
            block,
            Face.Y_MINUS,
            bc.Potential.Dirichlet(),
            where=torch.ones(NX, NZ, dtype=torch.bool),
        )
    with pytest.raises(TypeError, match="bool"):
        bc.set_bc(
            block, Face.Y_MINUS, bc.Potential.Dirichlet(), where=torch.ones(NZ, NX)
        )
    with pytest.raises(NotImplementedError):
        bc.set_bc(
            block, Face.Y_MINUS, bc.Velocity.Dirichlet(), where=_face_y_minus_mask()
        )
    with pytest.raises(ValueError, match="cw > 0"):
        bc.Potential.ThinWall(cw=0.0)
    with pytest.raises(RuntimeError, match="THIN_WALL"):
        block.getBoundary(Face.Y_MINUS).setPotentialBC(PotentialBC.THIN_WALL)
    with pytest.raises(RuntimeError, match="PotentialBC members"):
        block.getBoundary(Face.Y_MINUS).setPotentialTypes(
            torch.full((1, 1, NZ, 1, NX), 7, dtype=torch.int8)
        )


def test_legacy_setters_map_to_potential_bc() -> None:
    domain = _domain()
    bound = domain.getBlock(0).getBoundary(Face.Y_MINUS)
    bound.setEpotCw(CW)
    assert bound.getPotentialBC() == PotentialBC.THIN_WALL and bound.getEpotCw() == CW
    bound.setEpotCw(0.0)
    assert bound.getPotentialBC() == PotentialBC.INSULATING and not bound.hasEpotCw()
    bound.setEpotInsulating(False)
    assert bound.getPotentialBC() == PotentialBC.OPEN and not bound.isEpotInsulating()
    bound.setEpotDirichlet(True)
    assert bound.getPotentialBC() == PotentialBC.DIRICHLET and bound.hasEpotDirichlet()
    bound.setEpotDirichlet(False)
    assert bound.getPotentialBC() == PotentialBC.INSULATING


def _mixed_domain() -> _C.Domain:
    def set_bcs(block: _C.Block) -> None:
        bc.set_bc(block, Face.Y_PLUS, bc.Potential.ThinWall(cw=CW))
        bc.set_bc(block, Face.Y_MINUS, bc.Potential.ThinWall(cw=CW))
        bc.set_bc(
            block, Face.Y_MINUS, bc.Potential.Dirichlet(), where=_face_y_minus_mask()
        )
        bc.set_bc(block, Face.Z_MINUS, bc.Potential.Open())

    return _domain(set_bcs)


def test_copy_and_clone_keep_potential_bc() -> None:
    domain = _mixed_domain()
    ref = _operators(domain)
    for copy in (domain.Copy(), domain.Clone()):
        copy.PrepareSolve()
        new = _operators(copy)
        for name in ref:
            assert torch.equal(ref[name], new[name]), name


def test_save_load_keeps_potential_bc(tmp_path) -> None:
    domain = _mixed_domain()
    ref = _operators(domain)
    path = tmp_path / "domain"
    save_domain(domain, path)
    loaded = load_domain(path, dtype=DTYPE, device=DEVICE)
    loaded.PrepareSolve()
    loaded.SetupEpotOnDomain(0, False)
    new = _operators(loaded)
    for name in ref:
        assert torch.equal(ref[name], new[name]), name


def test_load_legacy_potential_keys(tmp_path) -> None:
    """Domains saved with the old independent epot flags load with the same BCs."""

    def set_bcs(block: _C.Block) -> None:
        bc.set_bc(block, Face.Y_PLUS, bc.Potential.ThinWall(cw=CW))
        bc.set_bc(block, Face.Y_MINUS, bc.Potential.Dirichlet())
        bc.set_bc(block, Face.Z_MINUS, bc.Potential.Open())

    domain = _domain(set_bcs)
    ref = _operators(domain)
    path = tmp_path / "domain"
    save_domain(domain, path)

    json_path = path.with_name("domain.json")
    saved = json.loads(json_path.read_text())
    # files of that era had neither a format version nor a save id
    saved.pop("format_version")
    saved.pop("save_id")
    legacy = {"THIN_WALL": {"epotCw": CW}, "DIRICHLET": {"epotDirichlet": True}}
    legacy["OPEN"] = {"epotInsulating": False}
    for bound_dict in saved["blocks"][0]["boundaries"]:
        name = bound_dict.pop("potentialBC", None)
        bound_dict.pop("potentialCw", None)
        if name is not None:
            bound_dict.update(legacy[name])
    json_path.write_text(json.dumps(saved))

    loaded = load_domain(path, dtype=DTYPE, device=DEVICE)
    loaded.PrepareSolve()
    loaded.SetupEpotOnDomain(0, False)
    new = _operators(loaded)
    for name in ref:
        assert torch.equal(ref[name], new[name]), name


def _epot_matvec(domain: _C.Domain, x: torch.Tensor) -> torch.Tensor:
    assert domain.Epot is not None
    n = domain.getTotalSize()
    matrix = torch.sparse_csr_tensor(
        domain.Epot.row.long(),
        domain.Epot.index.long(),
        domain.Epot.value[: domain.Epot.getNnz()],
        (n, n),
    )
    return matrix @ x


@pytest.mark.filterwarnings("ignore:Sparse CSR tensor support is in beta")
def test_dirichlet_value_is_exact_for_constant_potential() -> None:
    """φ = c on a whole Dirichlet face, no u×B, insulating elsewhere: φ ≡ c, j ≡ 0."""
    c = 0.7

    def set_bcs(block: _C.Block) -> None:
        bc.set_bc(block, Face.Y_MINUS, bc.Potential.Dirichlet(value=c))

    domain = _domain(set_bcs)
    n = domain.getTotalSize()
    zero_ucb = torch.zeros(3 * n, dtype=DTYPE, device=DEVICE)
    phi = torch.full((n,), c, dtype=DTYPE, device=DEVICE)
    rhs = _C.ComputeEpotRHS(domain, zero_ucb)
    torch.testing.assert_close(_epot_matvec(domain, phi), rhs, rtol=0, atol=1e-12)
    assert rhs.abs().max() > 0
    J = _C.ComputeCurrentDensityFaceBased(domain, phi, zero_ucb)
    torch.testing.assert_close(J, torch.zeros_like(J), rtol=0, atol=1e-12)


def test_dirichlet_values_are_linear_and_local() -> None:
    """Values shift the RHS and J only at their own Dirichlet face cells, linearly."""
    mask = _face_y_minus_mask()
    gen = torch.Generator().manual_seed(5)
    values = torch.randn(NZ, NX, dtype=DTYPE, generator=gen)

    def structure(block: _C.Block) -> None:
        bc.set_bc(block, Face.Y_MINUS, bc.Potential.Dirichlet(), where=mask)

    def with_values(block: _C.Block) -> None:
        structure(block)
        bc.set_potential_values(block, Face.Y_MINUS, values)

    ref, new = _operators(_domain(structure)), _operators(_domain(with_values))
    assert torch.equal(ref["value"], new["value"]), "values must not touch the matrix"
    for z in range(NZ):
        for y in range(NY):
            for x in range(NX):
                cell = _flat(z, y, x)
                if y == 0 and mask[z, x]:
                    assert new["rhs"][cell] != ref["rhs"][cell]
                    assert not torch.equal(new["J"][..., cell], ref["J"][..., cell])
                else:
                    # insulating cells ignore their (stored) values
                    assert new["rhs"][cell] == ref["rhs"][cell]
                    assert torch.equal(new["J"][..., cell], ref["J"][..., cell])

    doubled = _operators(
        _domain(
            lambda block: (
                structure(block),
                bc.set_potential_values(block, Face.Y_MINUS, 2 * values),
            )
        )
    )
    torch.testing.assert_close(
        doubled["rhs"] - ref["rhs"],
        2 * (new["rhs"] - ref["rhs"]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_batched_dirichlet_values() -> None:
    """Per-environment values with a shared structure match single-environment runs."""
    mask = _face_y_minus_mask()
    gen = torch.Generator().manual_seed(6)
    values = torch.randn(2, NZ, NX, dtype=DTYPE, generator=gen)

    def structure(block: _C.Block) -> None:
        bc.set_bc(block, Face.Y_MINUS, bc.Potential.Dirichlet(), where=mask)
        bc.set_bc(block, Face.Y_PLUS, bc.Potential.ThinWall(cw=CW))

    batched = _domain(structure, batch=2)
    bc.set_potential_values(batched.getBlock(0), Face.Y_MINUS, values)
    assert batched.Epot is not None and batched.Epot.getBatchSize() == 1
    new = _operators(batched)

    n = batched.getTotalSize()
    gen_ref = torch.Generator(device=DEVICE).manual_seed(0)
    ucb = torch.randn(2, 3 * n, dtype=DTYPE, device=DEVICE, generator=gen_ref)
    phi = torch.randn(2, n, dtype=DTYPE, device=DEVICE, generator=gen_ref)
    for env in range(2):
        single = _domain(
            lambda block, env=env: (
                structure(block),
                bc.set_potential_values(block, Face.Y_MINUS, values[env]),
            )
        )
        rhs = _C.ComputeEpotRHS(single, ucb[env].contiguous())
        J = _C.ComputeCurrentDensityFaceBased(
            single, phi[env].contiguous(), ucb[env].contiguous()
        ).reshape(3, n)
        assert torch.equal(new["rhs"].reshape(2, n)[env], rhs)
        assert torch.equal(new["J"][env], J)


def test_set_potential_values_in_place_and_mhd_run() -> None:
    """Changing the actuator values between steps needs no rebuild and runs stably."""
    mask = _face_y_minus_mask()

    def set_bcs(block: _C.Block) -> None:
        bc.set_bc(block, Face.Y_MINUS, bc.Potential.Dirichlet(value=0.0), where=mask)

    domain = _domain(set_bcs)
    block = domain.getBlock(0)
    bound = block.getBoundary(Face.Y_MINUS)
    storage = bound.potentialValues
    assert storage is not None
    sim = phipict.MHDSimulation(
        domain=domain,
        dt=5e-3,
        stuart_number=torch.tensor(20.0),
        e_b=torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE),
        substeps=1,
        non_orthogonal=False,
        potential_return_best_result=False,
        potential_tol=phipict.SolverTolerance(rtol=1e-10, atol=1e-14),
    )
    epot_matrix = domain.Epot
    for step in range(3):
        bc.set_potential_values(block, Face.Y_MINUS, 0.1 * (step + 1), where=mask)
        assert bound.potentialValues.data_ptr() == storage.data_ptr()
        assert sim.single_step()
        assert domain.Epot is epot_matrix
    phi = block.epot
    assert phi is not None and torch.isfinite(phi).all()
    # the electrode potential drives the field: φ next to the electrodes is near 0.3
    near = phi[0, 0, :, 0, : NX // 2]
    assert (near - 0.3).abs().max() < 0.3


def test_save_load_keeps_dirichlet_values(tmp_path) -> None:
    mask = _face_y_minus_mask()
    values = torch.linspace(0, 1, NZ * NX, dtype=DTYPE).reshape(NZ, NX)

    def set_bcs(block: _C.Block) -> None:
        bc.set_bc(block, Face.Y_MINUS, bc.Potential.Dirichlet(value=values), where=mask)

    domain = _domain(set_bcs)
    ref = _operators(domain)
    path = tmp_path / "domain"
    save_domain(domain, path)
    loaded = load_domain(path, dtype=DTYPE, device=DEVICE)
    loaded.PrepareSolve()
    loaded.SetupEpotOnDomain(0, False)
    new = _operators(loaded)
    for name in ref:
        assert torch.equal(ref[name], new[name]), name


@pytest.mark.parametrize("shared", [False, True])
def test_dirichlet_values_adjoint(shared: bool) -> None:
    """RHS and current density gradients w.r.t. the values match finite differences.

    Per-environment values get one gradient slice each; values shared by the
    environments get the sum over them.
    """
    batch = 2
    mask = _face_y_minus_mask()

    def structure(block: _C.Block) -> None:
        bc.set_bc(block, Face.Y_MINUS, bc.Potential.Dirichlet(), where=mask)
        bc.set_bc(block, Face.Y_PLUS, bc.Potential.ThinWall(cw=CW))

    domain = _domain(structure, batch=batch)
    block = domain.getBlock(0)
    n = domain.getTotalSize() * batch
    gen = torch.Generator(device=DEVICE).manual_seed(7)
    ucb = torch.randn(3 * n, dtype=DTYPE, device=DEVICE, generator=gen)
    phi = torch.randn(n, dtype=DTYPE, device=DEVICE, generator=gen)
    shape = (NZ, NX) if shared else (batch, NZ, NX)
    values = torch.randn(*shape, dtype=DTYPE, device=DEVICE, generator=gen)
    values.requires_grad_(True)

    def rhs(g: torch.Tensor) -> torch.Tensor:
        bc.set_potential_values(block, Face.Y_MINUS, g)
        return piso_diff.ComputeEpotRHS(domain, ucb)

    def current(g: torch.Tensor) -> torch.Tensor:
        bc.set_potential_values(block, Face.Y_MINUS, g)
        return piso_diff.ComputeCurrentDensityFaceBased(domain, phi, ucb)

    for fn in (rhs, current):
        assert torch.autograd.gradcheck(
            fn, (values,), eps=1e-6, atol=1e-8, nondet_tol=1e-12
        )

    # the tracked values are a new per-environment tensor carrying the graph
    bound = block.getBoundary(Face.Y_MINUS)
    rhs(values)
    assert bound.potentialValues.requires_grad
    assert bound.potentialValues.shape[0] == batch
    assert not bound.hasPotentialValuesGrad(), (
        "the grad buffer must not outlive backward"
    )


def test_dirichlet_values_gradient_through_mhd_steps() -> None:
    """d loss / d electrode values through differentiable MHD steps matches FD."""
    batch, steps = 2, 2
    mask = _face_y_minus_mask()

    def structure(block: _C.Block) -> None:
        bc.set_bc(block, Face.Y_MINUS, bc.Potential.Dirichlet(), where=mask)

    gen = torch.Generator(device=DEVICE).manual_seed(8)
    v0 = 0.1 * torch.randn(
        batch, 3, NZ, NY, NX, dtype=DTYPE, device=DEVICE, generator=gen
    )
    w_vel = torch.rand(batch, 3, NZ, NY, NX, dtype=DTYPE, device=DEVICE, generator=gen)
    w_phi = torch.randn(batch, 1, NZ, NY, NX, dtype=DTYPE, device=DEVICE, generator=gen)
    values = torch.randn(batch, NZ, NX, dtype=DTYPE, device=DEVICE, generator=gen)
    direction = torch.randn(batch, NZ, NX, dtype=DTYPE, device=DEVICE, generator=gen)
    tight = phipict.SolverTolerance(rtol=1e-12, atol=1e-14)

    def loss(g: torch.Tensor) -> tuple[torch.Tensor, _C.Domain]:
        domain = _domain(structure, batch=batch)
        block = domain.getBlock(0)
        block.setVelocity(v0.clone())
        domain.UpdateDomainData()
        sim = phipict.MHDSimulation(
            domain=domain,
            dt=5e-3,
            stuart_number=torch.tensor(20.0),
            e_b=torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE),
            substeps=1,
            non_orthogonal=False,
            differentiable=True,
            advection_tol=tight,
            pressure_tol=tight,
            potential_tol=tight,
            potential_use_preconditioner=False,
            potential_return_best_result=False,
        )
        bc.set_potential_values(block, Face.Y_MINUS, g)
        for _ in range(steps):
            assert sim.single_step()
        assert block.epot is not None
        value = (w_vel * block.velocity**2).sum() + (w_phi * block.epot).sum()
        return value, domain

    g = values.clone().requires_grad_(True)
    value, domain = loss(g)
    (grad,) = torch.autograd.grad(value, g)
    domain.Detach()
    assert (
        not domain.getBlock(0).getBoundary(Face.Y_MINUS).potentialValues.requires_grad
    )

    # only the Dirichlet cells act on the flow
    assert torch.all(grad[:, ~mask.to(DEVICE)] == 0)
    assert float(grad[:, mask.to(DEVICE)].abs().min()) > 0
    # per-environment gradients really differ
    assert not torch.allclose(grad[0], grad[1])

    eps = 1e-4
    with torch.no_grad():
        plus, _ = loss(values + eps * direction)
        minus, _ = loss(values - eps * direction)
    fd = float((plus - minus) / (2 * eps))
    ad = float((grad * direction).sum())
    assert abs(ad - fd) <= 1e-6 * abs(fd), f"AD {ad:.10e} vs FD {fd:.10e}"


# ---------------------------------------------------------------------------
# Prescribed current (CURRENT)
# ---------------------------------------------------------------------------


def _electrode_masks() -> tuple[torch.Tensor, torch.Tensor]:
    """Two electrodes over the ±y face cells [NZ, NX], one per face."""
    lower = torch.zeros(NZ, NX, dtype=torch.bool)
    lower[1:3, 1:3] = True
    upper = torch.zeros(NZ, NX, dtype=torch.bool)
    upper[2:4, 4:7] = True
    return lower, upper


def _dense_epot(domain: _C.Domain) -> torch.Tensor:
    assert domain.Epot is not None
    n = domain.getTotalSize()
    return torch.sparse_csr_tensor(
        domain.Epot.row.long(),
        domain.Epot.index.long(),
        domain.Epot.value[: domain.Epot.getNnz()],
        (n, n),
    ).to_dense()


@pytest.mark.filterwarnings("ignore:Sparse CSR tensor support is in beta")
def test_zero_current_is_insulating_wall() -> None:
    """Current cells at zero current give exactly the insulating wall's solve."""
    lower, upper = _electrode_masks()

    def electrodes(block: _C.Block) -> None:
        bc.set_bc(block, Face.Y_MINUS, bc.Potential.Current(value=0.0), where=lower)
        bc.set_bc(block, Face.Y_PLUS, bc.Potential.Current(value=0.0), where=upper)

    plain, with_electrodes = _domain(None), _domain(electrodes)
    assert not _epot_matrix_is_anchored(with_electrodes)
    ref, new = _operators(plain), _operators(with_electrodes)
    for name in ref:
        assert torch.equal(ref[name], new[name]), name

    # a few MHD steps from the same state give the same flow
    gen = torch.Generator(device=DEVICE).manual_seed(11)
    v0 = 0.1 * torch.randn(1, 3, NZ, NY, NX, dtype=DTYPE, device=DEVICE, generator=gen)
    fields = []
    for domain in (plain, with_electrodes):
        block = domain.getBlock(0)
        block.setVelocity(v0.clone())
        domain.UpdateDomainData()
        sim = phipict.MHDSimulation(
            domain=domain,
            dt=5e-3,
            stuart_number=torch.tensor(20.0),
            e_b=torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE),
            substeps=1,
            non_orthogonal=False,
            potential_use_preconditioner=False,
            potential_tol=phipict.SolverTolerance(rtol=1e-10, atol=1e-14),
        )
        for _ in range(3):
            assert sim.single_step()
        assert block.epot is not None
        fields.append((block.velocity.clone(), block.epot.clone()))
    for ref_field, new_field in zip(*fields, strict=True):
        torch.testing.assert_close(new_field, ref_field, rtol=0, atol=1e-12)


@pytest.mark.filterwarnings("ignore:Sparse CSR tensor support is in beta")
def test_current_reproduces_dirichlet_currents() -> None:
    """Prescribing the currents of a Dirichlet solution reproduces that solution.

    Electrodes held at fixed potentials on both ±y faces, no u×B: the currents
    into the fluid through their cells, prescribed as Current cells instead, give
    the same potential (up to the free constant) and the same current density.
    This pins the sign convention on lower and upper faces alike.
    """
    lower, upper = _electrode_masks()
    potentials = {Face.Y_MINUS: 0.8, Face.Y_PLUS: -0.3}
    masks = {Face.Y_MINUS: lower, Face.Y_PLUS: upper}
    wall_y = {Face.Y_MINUS: 0, Face.Y_PLUS: NY - 1}

    def dirichlet(block: _C.Block) -> None:
        for face, value in potentials.items():
            bc.set_bc(
                block, face, bc.Potential.Dirichlet(value=value), where=masks[face]
            )

    domain_d = _domain(dirichlet)
    n = domain_d.getTotalSize()
    zero_ucb = torch.zeros(3 * n, dtype=DTYPE, device=DEVICE)
    rhs_d = _C.ComputeEpotRHS(domain_d, zero_ucb)
    phi_d = torch.linalg.solve(_dense_epot(domain_d), rhs_d)
    J_d = _C.ComputeCurrentDensityFaceBased(domain_d, phi_d, zero_ucb)

    # the RHS carries -coef*g at a Dirichlet cell: unit values give the coefficients
    block_d = domain_d.getBlock(0)
    for face in potentials:
        bc.set_potential_values(block_d, face, 1.0, where=masks[face])
    coef = -_C.ComputeEpotRHS(domain_d, zero_ucb)

    # the current into the fluid through every electrode cell, coef (g - phi_P)
    currents = {}
    for face, value in potentials.items():
        values = torch.zeros(NZ, NX, dtype=DTYPE, device=DEVICE)
        for z in range(NZ):
            for x in range(NX):
                if masks[face][z, x]:
                    cell = _flat(z, wall_y[face], x)
                    values[z, x] = coef[cell] * (value - phi_d[cell])
        currents[face] = values
    total = sum(float(values.sum()) for values in currents.values())
    assert abs(total) < 1e-10, "the insulated duct conserves the electrode current"
    assert float(currents[Face.Y_MINUS].sum()) > 0

    def prescribed(block: _C.Block) -> None:
        for face, values in currents.items():
            bc.set_bc(
                block, face, bc.Potential.Current(value=values), where=masks[face]
            )

    domain_c = _domain(prescribed)
    rhs_c = _C.ComputeEpotRHS(domain_c, zero_ucb)
    assert abs(float(rhs_c.sum())) < 1e-10
    phi_c = torch.linalg.pinv(_dense_epot(domain_c)) @ rhs_c
    offset = phi_c - phi_d
    torch.testing.assert_close(
        offset - offset.mean(), torch.zeros_like(offset), rtol=0, atol=1e-9
    )
    J_c = _C.ComputeCurrentDensityFaceBased(domain_c, phi_c, zero_ucb)
    torch.testing.assert_close(J_c, J_d, rtol=0, atol=1e-9)


def test_set_potential_current_spreads_by_area() -> None:
    """The total current is shared by the cells' areas, on a graded face."""
    mask = torch.zeros(NY, NX, dtype=torch.bool)
    mask[1:5, 2:6] = True

    def electrode(block: _C.Block) -> None:
        bc.set_bc(block, Face.Z_MINUS, bc.Potential.Current(), where=mask)

    domain = _domain(electrode)
    block = domain.getBlock(0)
    bc.set_potential_current(block, Face.Z_MINUS, 2.0, where=mask)
    values = block.getBoundary(Face.Z_MINUS).potentialValues
    assert values is not None
    values = values[0, 0, 0]  # [NY, NX]
    torch.testing.assert_close(
        values.sum(), torch.tensor(2.0, dtype=DTYPE, device=DEVICE)
    )
    assert torch.all(values[~mask.to(DEVICE)] == 0)

    # uniform density: the share of a cell is its area dx*dy, from the cell volume
    area = get_cell_size(block)[0, 0, 0] / (2.0 / NZ)  # [NY, NX], z is uniform
    density = values[mask.to(DEVICE)] / area[mask.to(DEVICE)]
    torch.testing.assert_close(density, density.mean().expand_as(density))
    # the y grading makes the shares differ, so this is not a plain even split
    assert float(values[mask.to(DEVICE)].std()) > 0

    # one total per environment
    batched = _domain(electrode, batch=2)
    bc.set_potential_current(
        batched.getBlock(0),
        Face.Z_MINUS,
        torch.tensor([1.0, -3.0], dtype=DTYPE),
        where=mask,
    )
    per_env = batched.getBlock(0).getBoundary(Face.Z_MINUS).potentialValues
    assert per_env is not None
    torch.testing.assert_close(
        per_env.sum(dim=(1, 2, 3, 4)),
        torch.tensor([1.0, -3.0], dtype=DTYPE, device=DEVICE),
    )


def test_save_load_keeps_current_values(tmp_path) -> None:
    lower, _ = _electrode_masks()
    values = torch.linspace(-1, 1, NZ * NX, dtype=DTYPE).reshape(NZ, NX)

    def set_bcs(block: _C.Block) -> None:
        bc.set_bc(block, Face.Y_MINUS, bc.Potential.Current(value=values), where=lower)

    domain = _domain(set_bcs)
    ref = _operators(domain)
    path = tmp_path / "domain"
    save_domain(domain, path)
    loaded = load_domain(path, dtype=DTYPE, device=DEVICE)
    loaded.PrepareSolve()
    loaded.SetupEpotOnDomain(0, False)
    new = _operators(loaded)
    for name in ref:
        assert torch.equal(ref[name], new[name]), name


@pytest.mark.parametrize("shared", [False, True])
def test_current_values_adjoint(shared: bool) -> None:
    """RHS and current density gradients w.r.t. prescribed currents match FD.

    Electrodes on both ±y faces (the +y one next to thin-wall cells), per
    environment or shared by the environments.
    """
    batch = 2
    lower, upper = _electrode_masks()

    def structure(block: _C.Block) -> None:
        bc.set_bc(block, Face.Y_MINUS, bc.Potential.Current(), where=lower)
        bc.set_bc(block, Face.Y_PLUS, bc.Potential.ThinWall(cw=CW))
        bc.set_bc(block, Face.Y_PLUS, bc.Potential.Current(), where=upper)

    domain = _domain(structure, batch=batch)
    block = domain.getBlock(0)
    n = domain.getTotalSize() * batch
    gen = torch.Generator(device=DEVICE).manual_seed(12)
    ucb = torch.randn(3 * n, dtype=DTYPE, device=DEVICE, generator=gen)
    phi = torch.randn(n, dtype=DTYPE, device=DEVICE, generator=gen)
    shape = (NZ, NX) if shared else (batch, NZ, NX)
    values_lower = torch.randn(*shape, dtype=DTYPE, device=DEVICE, generator=gen)
    values_upper = torch.randn(*shape, dtype=DTYPE, device=DEVICE, generator=gen)
    values_lower.requires_grad_(True)
    values_upper.requires_grad_(True)

    def set_values(g_lower: torch.Tensor, g_upper: torch.Tensor) -> None:
        bc.set_potential_values(block, Face.Y_MINUS, g_lower)
        bc.set_potential_values(block, Face.Y_PLUS, g_upper)

    def rhs(g_lower: torch.Tensor, g_upper: torch.Tensor) -> torch.Tensor:
        set_values(g_lower, g_upper)
        return piso_diff.ComputeEpotRHS(domain, ucb)

    def current(g_lower: torch.Tensor, g_upper: torch.Tensor) -> torch.Tensor:
        set_values(g_lower, g_upper)
        return piso_diff.ComputeCurrentDensityFaceBased(domain, phi, ucb)

    for fn in (rhs, current):
        assert torch.autograd.gradcheck(
            fn, (values_lower, values_upper), eps=1e-6, atol=1e-8, nondet_tol=1e-12
        )

    # the currents act at the electrode cells only
    (grad_lower,) = torch.autograd.grad(
        rhs(values_lower, values_upper).sum(), [values_lower]
    )
    grad_cells = grad_lower if shared else grad_lower[0]
    assert torch.all(grad_cells[~lower.to(DEVICE)] == 0)
    assert torch.all(grad_cells[lower.to(DEVICE)] != 0)


def test_current_gradient_through_mhd_steps() -> None:
    """d loss / d prescribed currents through differentiable MHD steps matches FD."""
    batch, steps = 2, 2
    lower, upper = _electrode_masks()

    def structure(block: _C.Block) -> None:
        bc.set_bc(block, Face.Y_MINUS, bc.Potential.Current(), where=lower)
        bc.set_bc(block, Face.Y_PLUS, bc.Potential.Current(), where=upper)

    gen = torch.Generator(device=DEVICE).manual_seed(13)
    v0 = 0.1 * torch.randn(
        batch, 3, NZ, NY, NX, dtype=DTYPE, device=DEVICE, generator=gen
    )
    w_vel = torch.rand(batch, 3, NZ, NY, NX, dtype=DTYPE, device=DEVICE, generator=gen)
    w_phi = torch.randn(batch, 1, NZ, NY, NX, dtype=DTYPE, device=DEVICE, generator=gen)

    def dipole(total: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # +I spread over the lower electrode, -I over the upper one: zero net
        # current, as the singular potential system requires
        lower_values = total[:, None, None] * lower.to(DEVICE) / lower.sum()
        upper_values = -total[:, None, None] * upper.to(DEVICE) / upper.sum()
        return lower_values, upper_values

    totals = torch.tensor([0.7, -1.3], dtype=DTYPE, device=DEVICE)
    direction = torch.tensor([1.0, 0.5], dtype=DTYPE, device=DEVICE)
    tight = phipict.SolverTolerance(rtol=1e-12, atol=1e-14)

    def loss(total: torch.Tensor) -> tuple[torch.Tensor, _C.Domain]:
        domain = _domain(structure, batch=batch)
        block = domain.getBlock(0)
        block.setVelocity(v0.clone())
        domain.UpdateDomainData()
        sim = phipict.MHDSimulation(
            domain=domain,
            dt=5e-3,
            stuart_number=torch.tensor(20.0),
            e_b=torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE),
            substeps=1,
            non_orthogonal=False,
            differentiable=True,
            advection_tol=tight,
            pressure_tol=tight,
            potential_tol=tight,
            potential_use_preconditioner=False,
            potential_return_best_result=False,
        )
        lower_values, upper_values = dipole(total)
        bc.set_potential_values(block, Face.Y_MINUS, lower_values)
        bc.set_potential_values(block, Face.Y_PLUS, upper_values)
        for _ in range(steps):
            assert sim.single_step()
        assert block.epot is not None
        epot = block.epot - block.epot.mean(dim=(1, 2, 3, 4), keepdim=True)
        value = (w_vel * block.velocity**2).sum() + (w_phi * epot).sum()
        return value, domain

    total = totals.clone().requires_grad_(True)
    value, domain = loss(total)
    (grad,) = torch.autograd.grad(value, total)
    domain.Detach()
    assert torch.all(grad != 0)

    eps = 1e-4
    with torch.no_grad():
        plus, _ = loss(totals + eps * direction)
        minus, _ = loss(totals - eps * direction)
    fd = float((plus - minus) / (2 * eps))
    ad = float((grad * direction).sum())
    assert abs(ad - fd) <= 1e-6 * abs(fd), f"AD {ad:.10e} vs FD {fd:.10e}"
