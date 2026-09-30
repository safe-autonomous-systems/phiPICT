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

"""Reading and writing OpenFOAM blockMeshDict files."""

import math
from pathlib import Path

import pytest
import torch

import phipict
import phipict.meshing as pm
from phipict import Face
from phipict.meshing.formats import FoamParseError, load_blockmeshdict, parse_foam
from phipict.meshing.quality import cell_volumes

EXAMPLE = (
    Path(__file__).parents[2]
    / "examples"
    / "meshes"
    / "flow_past_cylinder"
    / "blockMeshDict"
)
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

HEADER = (
    "FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }\n"
)

CAVITY = (
    HEADER
    + """
convertToMeters 0.1;
vertices ((0 0 0) (1 0 0) (1 1 0) (0 1 0) (0 0 0.1) (1 0 0.1) (1 1 0.1) (0 1 0.1));
blocks (hex (0 1 2 3 4 5 6 7) (20 20 1) simpleGrading (1 1 1));
edges ();
boundary
(
    movingWall { type wall; faces ((3 7 6 2)); }
    fixedWalls { type wall; faces ((0 4 7 3) (2 6 5 1) (1 5 4 0)); }
    frontAndBack { type empty; faces ((0 3 2 1) (4 5 6 7)); }
);
"""
)

# quarter of an annulus around the z axis, 3D, with arcs and multi-grading
ANNULUS = (
    HEADER
    + """
r1 1; r2 2;
c45 #calc "0";
"""
)

ARC3D = (
    HEADER
    + """
ri 1; ro 2; h 0.5;
vertices
(
    ($ri 0 0) ($ro 0 0) (0 $ro 0) (0 $ri 0)
    ($ri 0 $h) ($ro 0 $h) (0 $ro $h) (0 $ri $h)
);
blocks
(
    hex (0 1 2 3 4 5 6 7) ring (6 12 3)
        simpleGrading (((0.5 0.5 3) (0.5 0.5 0.333333333333)) 1 1)
);
edges
(
    arc 1 2 origin (0 0 0)
    arc 0 3 (0.707106781187 0.707106781187 0)
    arc 5 6 (1.41421356237 1.41421356237 0.5)
    arc 4 7 origin (0 0 0.5)
);
boundary
(
    inner { type wall; faces ((0 4 7 3)); }
    outer { type patch; faces ((1 2 6 5)); }
    sides { type symmetryPlane; faces ((0 1 5 4) (3 7 6 2)); }
    ends { type patch; faces ((0 3 2 1) (4 5 6 7)); }
);
"""
)

CYCLIC = (
    HEADER
    + """
vertices ((0 0 0) (2 0 0) (2 1 0) (0 1 0) (0 0 1) (2 0 1) (2 1 1) (0 1 1)
          (4 0 0) (4 1 0) (4 0 1) (4 1 1));
blocks
(
    hex (0 1 2 3 4 5 6 7) (8 6 1) edgeGrading (1 1 1 1 4 4 4 4 1 1 1 1)
    hex (1 8 9 2 5 10 11 6) (8 6 1) simpleGrading (1 4 1)
);
boundary
(
    left { type cyclic; neighbourPatch right; faces ((0 4 7 3)); }
    right { type cyclic; neighbourPatch left; faces ((8 9 11 10)); }
    walls { type wall; faces ((0 1 5 4) (1 8 10 5) (3 7 6 2) (2 6 11 9)); }
    frontAndBack { type empty; faces ((0 3 2 1) (4 5 6 7) (1 2 9 8) (5 10 11 6)); }
);
mergePatchPairs ();
"""
)


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "blockMeshDict"
    p.write_text(text)
    return p


def test_parser_handles_comments_macros_and_nesting():
    d = parse_foam(
        HEADER + "/* a\n b */ a 1; // c\n b $a; s { x (1 (2 3)); } l (n { t w; });"
    )
    assert d["a"] == [1] and d["b"] == [1]
    assert d["s"]["x"] == [[1, [2, 3]]]
    assert d["l"] == [["n", {"t": ["w"]}]]


def test_cavity_is_2d(tmp_path):
    mesh = pm.read_blockmeshdict(write(tmp_path, CAVITY))
    assert mesh.ndims == 2 and len(mesh) == 1
    (b,) = mesh.blocks
    assert b.cells == (20, 20)
    assert torch.allclose(
        b.coords[:, -1, -1], torch.tensor([0.1, 0.1], dtype=torch.float64)
    )
    assert b.patches.y_plus.name == "movingWall"
    assert (
        b.patches.x_minus.name
        == b.patches.x_plus.name
        == b.patches.y_minus.name
        == "fixedWalls"
    )
    assert isinstance(b.patches.y_plus.bc, pm.Wall)
    assert mesh.check().ok


def test_directives_are_rejected(tmp_path):
    with pytest.raises(FoamParseError, match="#calc"):
        pm.read_blockmeshdict(write(tmp_path, ANNULUS))
    merge = CAVITY + "mergePatchPairs ((a b));"
    with pytest.raises(FoamParseError, match="mergePatchPairs"):
        pm.read_blockmeshdict(write(tmp_path, merge))


def test_arcs_multigrading_3d(tmp_path):
    mesh = pm.read_blockmeshdict(write(tmp_path, ARC3D))
    assert mesh.ndims == 3
    (b,) = mesh.blocks
    assert b.name == "ring" and b.cells == (6, 12, 3)
    r = b.coords[:2].norm(dim=0)  # [z, y, x]
    assert torch.allclose(r[:, :, 0], torch.ones_like(r[:, :, 0]))  # inner arc
    assert torch.allclose(r[:, :, -1], torch.full_like(r[:, :, -1], 2.0))  # outer arc
    # radial multi-grading ((0.5 0.5 3) (0.5 0.5 1/3)): symmetric, fine at both ends
    sizes = r[0, 0].diff()
    assert float(sizes[2] / sizes[0]) == pytest.approx(3.0, rel=1e-6)
    assert torch.allclose(sizes, sizes.flip(0))
    assert float(cell_volumes(b.coords).min()) > 0
    assert isinstance(b.patches.y_minus.bc, pm.FreeSlip)
    assert b.patches.x_plus.bc is None  # "patch": condition up to the user
    assert not mesh.check().ok
    mesh.set_bc("outer", pm.Outflow())
    mesh.set_bc("ends", pm.Wall())
    assert mesh.check().ok


def test_cyclic_and_edge_grading(tmp_path):
    mesh = pm.read_blockmeshdict(write(tmp_path, CYCLIC))
    assert mesh.ndims == 2 and len(mesh) == 2
    a, b = mesh.blocks
    assert torch.allclose(a.coords[1, :, 0], b.coords[1, :, 0])  # same y grading
    sizes = a.coords[1, :, 0].diff()
    assert float(sizes[-1] / sizes[0]) == pytest.approx(4.0)
    periodic = [c for c in mesh.connections if c.translation is not None]
    assert len(periodic) == 1 and periodic[0].translation == pytest.approx((4.0, 0.0))
    assert not mesh.free_faces()


def test_user_cylinder_example():
    mesh = pm.read_blockmeshdict(EXAMPLE)
    assert mesh.ndims == 2 and len(mesh) == 9 and mesh.n_cells == 280 * 280
    assert len(mesh.connections) == 12
    assert {p.name for p in mesh.patches} == {"inlet", "outlet", "side1", "side2"}
    b0 = mesh.blocks[0]
    assert b0.cells == (40, 80)
    dx, dy = b0.coords[0, 0].diff(), b0.coords[1, :, 0].diff()
    assert float(dx[-1] / dx[0]) == pytest.approx(0.05)
    assert float(dy[-1] / dy[0]) == pytest.approx(0.05)
    lo, hi = mesh.bounds()
    assert lo.tolist() == [-32, -64] and hi.tolist() == [96, 64]
    report = mesh.check()
    assert report.min_scaled_jacobian == pytest.approx(1.0)


@needs_cuda
def test_user_cylinder_example_runs():
    mesh = pm.read_blockmeshdict(EXAMPLE)
    mesh.set_bc("inlet", pm.Inflow((1.0, 0.0)))
    mesh.set_bc("outlet", pm.Outflow())
    mesh.set_bc("side1", pm.FreeSlip())
    mesh.set_bc("side2", pm.FreeSlip())
    domain = mesh.get_domain(
        viscosity=0.01, dtype=torch.float64, device="cuda", passive_scalar_channels=0
    )
    sim = phipict.Simulation(domain=domain, dt=0.05)
    for _ in range(2):
        sim.single_step()
    assert all(torch.isfinite(b.velocity).all() for b in domain.getBlocks())
    outlet = mesh.boundaries(domain, "outlet")
    assert len(outlet) == 3


@pytest.mark.parametrize(
    "text", [CAVITY, ARC3D, CYCLIC], ids=["cavity", "arc3d", "cyclic"]
)
def test_write_read_roundtrip(tmp_path, text):
    src = write(tmp_path, text)
    bm, _ = load_blockmeshdict(src)
    out = tmp_path / "written"
    pm.write_blockmeshdict(bm, out)
    a = pm.read_blockmeshdict(src)
    b = pm.read_blockmeshdict(out)
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert torch.allclose(x.coords, y.coords, atol=1e-9)
        for face in Face:
            if int(face) < 2 * x.ndims:
                px, py = x.patches[face], y.patches[face]
                assert (px is None) == (py is None)
                if px is not None:
                    assert px.name == py.name


def test_write_blockmesh_from_api(tmp_path):
    bm = pm.BlockMesh()
    walls = pm.Patch("walls", pm.Wall())
    bm.add(
        pm.Quad(
            [(0, 0), (1, 0), (0, 1), (1, 1)],
            cells=(8, 6),
            grading=(None, pm.Symmetric(4.0)),
            edges=pm.QuadEdges(y_minus=pm.Arc(through=(0.5, -0.1))),
            patches=pm.FacePatches(y_minus=walls),
        )
    )
    bm.add(
        pm.Quad(
            [(1, 0), (3, 0), (1, 1), (3, 1)],
            cells=(12, None),
            patches=pm.FacePatches(y_minus=walls),
        )
    )
    ref = bm.build()
    pm.write_blockmeshdict(bm, tmp_path / "dict")
    back = pm.read_blockmeshdict(tmp_path / "dict")
    for x, y in zip(ref, back, strict=True):
        assert torch.allclose(x.coords, y.coords, atol=1e-9)
    assert not math.isnan(float(back.blocks[0].coords.sum()))
    with pytest.raises(ValueError, match="no blockMesh equivalent"):
        bad = pm.BlockMesh()
        bad.add(
            pm.Quad(
                [(0, 0), (1, 0), (0, 1), (1, 1)], cells=(4, 4), grading=pm.Tanh(2.0)
            )
        )
        pm.write_blockmeshdict(bad, tmp_path / "bad")
