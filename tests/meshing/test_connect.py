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

"""Automatic block connections, periodicity and BlockMesh inference."""

import itertools
import math

import pytest
import torch

import phipict
import phipict.meshing as pm
from phipict import Face

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
TIGHT = {
    "advection_tol": phipict.SolverTolerance(rtol=1e-12, atol=1e-14),
    "pressure_tol": phipict.SolverTolerance(rtol=1e-12, atol=1e-14),
}


def _orientations(ndims: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Index reorderings (axis order, flipped axes) keeping blocks right-handed."""
    out = []
    for perm in itertools.permutations(range(ndims)):
        inversions = sum(
            1 for i, j in itertools.combinations(range(ndims), 2) if perm[i] > perm[j]
        )
        for flips in itertools.product((0, 1), repeat=ndims):
            if (inversions + sum(flips)) % 2 == 0:
                out.append((perm, tuple(a for a in range(ndims) if flips[a])))
    return out


def _reorient(
    block: pm.MeshBlock, perm: tuple[int, ...], flips: tuple[int, ...]
) -> pm.MeshBlock:
    out = block.permute(perm)
    for a in flips:
        out = out.flip(a)
    return out


def _channel(ndims: int, n: int, orientation=None) -> pm.Mesh:
    """Two blocks in x, periodic in x (and z), walls at +-y."""
    walls = pm.Patch("walls", pm.Wall())
    lo, hi = (0.0,) * ndims, (1.0,) * ndims
    cells = (n,) * ndims
    fp = pm.FacePatches(y_minus=walls, y_plus=walls)
    a = pm.box(
        lo,
        hi,
        cells,
        grading=(None, pm.Symmetric(2.0)) + (None,) * (ndims - 2),
        patches=fp,
        name="a",
    )
    b = pm.box(
        (1.0,) + lo[1:],
        (2.0,) + hi[1:],
        cells,
        grading=(pm.Simple(1.5), pm.Symmetric(2.0)) + (None,) * (ndims - 2),
        patches=fp,
        name="b",
    )
    if orientation is not None:
        b = _reorient(b, *orientation)
    mesh = pm.Mesh([a, b])
    mesh.make_periodic("x")
    if ndims == 3:
        mesh.make_periodic("z")
    return mesh


def _velocity(centres: torch.Tensor) -> torch.Tensor:
    """A smooth initial velocity from cell centres ``[d, ...]``."""
    x, y = centres[0], centres[1]
    wall = y * (1 - y)
    u = [
        wall * (1 + 0.3 * torch.sin(math.pi * x)),
        0.05 * torch.sin(math.pi * x) * wall,
    ]
    if centres.shape[0] == 3:
        z = centres[2]
        u[0] = u[0] * (1 + 0.2 * torch.cos(2 * math.pi * z))
        u.append(0.05 * torch.sin(2 * math.pi * z) * wall)
    return torch.stack(u)


def _centres(coords: torch.Tensor) -> torch.Tensor:
    c = coords
    for dim in range(1, c.dim()):
        c = 0.5 * (
            c.narrow(dim, 0, c.shape[dim] - 1) + c.narrow(dim, 1, c.shape[dim] - 1)
        )
    return c


def _run(mesh: pm.Mesh, steps: int = 3) -> list[torch.Tensor]:
    domain = mesh.get_domain(
        viscosity=0.05, dtype=torch.float64, device="cuda", passive_scalar_channels=0
    )
    for block, sb in zip(mesh, domain.getBlocks(), strict=True):
        sb.velocity.copy_(_velocity(_centres(block.coords))[None].cuda())
    domain.UpdateDomainData()
    sim = phipict.Simulation(domain=domain, dt=0.02, **TIGHT)
    sim.make_divergence_free()
    for _ in range(steps):
        sim.single_step()
    return [sb.velocity[0].cpu() for sb in domain.getBlocks()]


# ------------------------------------------------------------------ geometry


@pytest.mark.parametrize("ndims", [2, 3])
def test_every_orientation_is_connected(ndims):
    for orientation in _orientations(ndims):
        mesh = _channel(ndims, 4, orientation)
        assert not mesh.free_faces()
        inner = [c for c in mesh.connections if c.translation is None]
        assert len(inner) == 1
        c = inner[0]
        assert {mesh.blocks[c.block_a].name, mesh.blocks[c.block_b].name} == {"a", "b"}


def test_orientation_codes_2d():
    """The codes of a plain side-by-side pair are those of the hand-written tests."""
    mesh = pm.Mesh([pm.box((0, 0), (1, 1), (4, 4)), pm.box((1, 0), (2, 1), (4, 4))])
    (c,) = mesh.connections
    assert (c.face_a, c.face_b) == (Face.X_PLUS, Face.X_MINUS)
    assert c.axis_codes(2)[0] == 2  # "-y": y of b, same direction
    flipped = pm.Mesh(
        [
            pm.box((0, 0), (1, 1), (4, 4)),
            pm.box((1, 0), (2, 1), (4, 4)).flip("x").flip("y"),
        ]
    )
    (c,) = flipped.connections
    assert (c.face_a, c.face_b) == (Face.X_PLUS, Face.X_PLUS)
    assert c.axis_codes(2)[0] == 3  # "+y": y of b, inverted


def test_non_conforming_interface_is_reported():
    a = pm.box((0, 0), (1, 1), (4, 4), name="a")
    b = pm.box((1, 0), (2, 1), (4, 6), name="b")
    with pytest.raises(pm.NonConformingInterfaceError, match="4.*6|6.*4"):
        pm.Mesh([a, b])
    c = pm.box((1, 0), (2, 1), (4, 4), grading=(None, 3.0), name="c")
    with pytest.raises(pm.NonConformingInterfaceError, match="grading"):
        pm.Mesh([a, c])


def test_free_faces_and_check():
    mesh = pm.Mesh([pm.box((0, 0), (1, 1), (4, 4), name="a")])
    report = mesh.check()
    assert not report.ok
    assert "a:X_MINUS" in report.errors[0]
    with pytest.raises(ValueError, match="without patch"):
        mesh.get_domain(viscosity=0.1, device="cpu")


def test_patch_names_and_set_bc():
    inlet = pm.Patch("inlet")
    mesh = pm.Mesh(
        [pm.box((0, 0), (1, 1), (4, 4), patches=pm.FacePatches.uniform(inlet, 2))]
    )
    assert not mesh.check().ok  # no condition yet
    with pytest.raises(KeyError, match="Did you mean 'inlet'"):
        mesh.set_bc("inlte", pm.Wall())
    mesh.set_bc("inlet", pm.Wall())
    assert mesh.check().ok
    assert len(mesh.faces("inlet")) == 4


def test_extrude_keeps_connections_and_periodicity():
    walls = pm.Patch("walls", pm.Wall())
    fp = pm.FacePatches(y_minus=walls, y_plus=walls)
    mesh = pm.Mesh(
        [
            pm.box((0, 0), (1, 1), (4, 4), patches=fp),
            pm.box((1, 0), (2, 1), (4, 4), patches=fp),
        ]
    )
    mesh.make_periodic("x")
    mesh3 = mesh.extrude((0.0, 0.5), cells=3)
    assert mesh3.ndims == 3
    assert not mesh3.free_faces()
    assert sum(c.is_periodic_axis for c in mesh3.connections) == 2  # z of both blocks
    assert (
        sum(
            c.translation is not None and not c.is_periodic_axis
            for c in mesh3.connections
        )
        == 1
    )


# ------------------------------------------------------------------ BlockMesh


def test_blockmesh_infers_cells_and_grading():
    bm = pm.BlockMesh()
    bm.add(
        pm.Quad(
            [(0, 0), (1, 0), (0, 1), (1, 1)],
            cells=(8, 6),
            grading=(None, pm.Symmetric(4.0)),
            name="a",
        )
    )
    bm.add(pm.Quad([(1, 0), (3, 0), (1, 1), (3, 1)], cells=(12, None), name="b"))
    mesh = bm.build()
    a, b = mesh.blocks
    assert b.cells == (12, 6)
    assert torch.allclose(a.coords[1, :, -1], b.coords[1, :, 0])  # grading inherited
    assert len(mesh.connections) == 1


def test_blockmesh_reverses_inherited_grading():
    bm = pm.BlockMesh()
    bm.add(
        pm.Quad(
            [(0, 0), (1, 0), (0, 1), (1, 1)],
            cells=(4, 8),
            grading=(None, 5.0),
            name="a",
        )
    )
    # b's y axis runs downwards along the shared edge
    bm.add(pm.Quad([(2, 1), (1, 1), (2, 0), (1, 0)], cells=(4, None), name="b"))
    mesh = bm.build()
    assert len(mesh.connections) == 1
    a, b = mesh.blocks
    assert torch.allclose(a.coords[1, :, -1], b.coords[1, :, -1].flip(0))


def test_blockmesh_conflicts():
    bm = pm.BlockMesh()
    bm.add(pm.Quad([(0, 0), (1, 0), (0, 1), (1, 1)], cells=(4, 8), name="a"))
    bm.add(pm.Quad([(1, 0), (2, 0), (1, 1), (2, 1)], cells=(4, 6), name="b"))
    with pytest.raises(ValueError, match="Conflicting cell counts"):
        bm.build()
    bm = pm.BlockMesh()
    bm.add(pm.Quad([(0, 0), (1, 0), (0, 1), (1, 1)], cells=(4, None), name="a"))
    with pytest.raises(ValueError, match="Cannot infer"):
        bm.build()


def test_blockmesh_shared_curved_edge():
    bm = pm.BlockMesh()
    bm.add(pm.Quad([(0, 0), (1, 0), (0, 1), (1, 1)], cells=(6, 6), name="a"))
    bm.add(pm.Quad([(1, 0), (2, 0), (1, 1), (2, 1)], cells=(6, 6), name="b"))
    bm.add_edge((1, 0), (1, 1), pm.Arc(through=(1.2, 0.5)))
    mesh = bm.build()
    a, b = mesh.blocks
    assert float(a.coords[0, 3, -1]) == pytest.approx(1.2)
    assert torch.allclose(a.coords[:, :, -1], b.coords[:, :, 0])


# ------------------------------------------------------------------ solver


@needs_cuda
@pytest.mark.parametrize("ndims", [2, 3])
def test_solver_is_independent_of_block_orientation(ndims):
    """Every relative orientation of two connected blocks gives the same flow."""
    n = 6 if ndims == 2 else 4
    ref = _run(_channel(ndims, n))
    for orientation in _orientations(ndims):
        vel = _run(_channel(ndims, n, orientation))
        assert torch.allclose(vel[0], ref[0], atol=1e-9), orientation
        # b's velocity is stored in its own index order
        expected = _reorient(pm.MeshBlock(ref[1]), *orientation).coords
        assert torch.allclose(vel[1], expected, atol=1e-9), orientation
