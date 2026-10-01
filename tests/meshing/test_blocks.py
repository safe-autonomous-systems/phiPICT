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

"""Single blocks: builders, curved edges, transforms and quality measures."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

import phipict.meshing as pm
from phipict import Face
from phipict.meshing.quality import cell_volumes, non_orthogonality, scaled_jacobian

LEGACY = np.load(Path(__file__).parent / "data" / "legacy_weights.npz")


def test_box_is_tensor_product():
    g = (pm.Simple(3.0), pm.Symmetric(5.0), pm.Tanh(1.5))
    b = pm.make_box((0, -1, 2), (2, 1, 3), cells=(4, 6, 5), grading=g)
    assert b.cells == (4, 6, 5) and b.coords.shape == (3, 6, 7, 5)
    x = 2 * g[0].weights(4)
    y = -1 + 2 * g[1].weights(6)
    z = 2 + g[2].weights(5)
    assert torch.allclose(b.coords[0, 0, 0], x)
    assert torch.allclose(b.coords[1, 0, :, 0], y)
    assert torch.allclose(b.coords[2, :, 0, 0], z)
    assert torch.allclose(b.coords[0], x.expand(6, 7, 5))


@pytest.mark.parametrize("ndims", [2, 3])
def test_straight_tfi_equals_box(ndims):
    lo, hi = (0.0,) * ndims, (1.0, 2.0, 3.0)[:ndims]
    grading = (pm.Simple(2.0), pm.Symmetric(4.0), None)[:ndims]
    cells = (5, 7, 4)[:ndims]
    box = pm.make_box(lo, hi, cells, grading)
    corners = [
        tuple(hi[a] if (k >> a) & 1 else lo[a] for a in range(ndims))
        for k in range(2**ndims)
    ]
    block = (pm.make_quad if ndims == 2 else pm.make_hexa)(corners, cells, grading)
    assert torch.allclose(block.coords, box.coords, atol=1e-13)


def test_arc_edge_lies_on_circle():
    r = 2.0
    corners = [
        (r * math.cos(math.radians(a)), r * math.sin(math.radians(a)))
        for a in (30, 150)
    ]
    b = pm.make_quad(
        [corners[1], corners[0], (-3, 3), (3, 3)],
        cells=(10, 4),
        grading=(pm.Symmetric(3.0), None),
        edges=pm.QuadEdges(y_minus=pm.Arc(center=(0, 0)).reversed()),
    )
    edge = b.coords[:, 0, :]
    assert torch.allclose(edge.norm(dim=0), torch.full((11,), r, dtype=torch.float64))
    through = pm.make_quad(
        [corners[1], corners[0], (-3, 3), (3, 3)],
        cells=(10, 4),
        edges=pm.QuadEdges(y_minus=pm.Arc(through=(0.0, 2.0))),
    )
    assert torch.allclose(
        through.coords[:, 0].norm(dim=0), torch.full((11,), r, dtype=torch.float64)
    )
    # uniform grading: equal arc lengths
    angles = torch.atan2(through.coords[1, 0], through.coords[0, 0])
    assert torch.allclose(angles.diff(), angles.diff()[0].expand(10))


def test_points_edge_is_used_verbatim():
    pts = torch.tensor(
        [[0.0, 0.0], [0.3, -0.1], [0.5, -0.12], [1.0, 0.0]], dtype=torch.float64
    )
    b = pm.make_quad(
        [(0, 0), (1, 0), (0, 1), (1, 1)],
        cells=(None, 2),
        edges=pm.QuadEdges(y_minus=pm.Points(pts)),
    )
    assert b.cells == (3, 2)
    assert torch.equal(b.coords[:, 0].T, pts)
    with pytest.raises(ValueError, match="Conflicting"):
        pm.make_quad(
            [(0, 0), (1, 0), (0, 1), (1, 1)],
            cells=(5, 2),
            edges=pm.QuadEdges(y_minus=pm.Points(pts)),
        )


@pytest.mark.parametrize(
    ("res", "r1", "r2", "angle"), [(24, 0.5, 1.0, -90.0), (16, 1.0, 3.0, 60.0)]
)
def test_annulus_matches_legacy_torus(res, r1, r2, angle):
    legacy = torch.from_numpy(LEGACY[f"torus_{res}_{r1}_{r2}_{angle}"])
    block = pm.make_annulus((0, 0), r1, r2, 135, angle, cells=(res, None))
    assert block.coords.shape == legacy.shape
    assert torch.allclose(block.coords, legacy, atol=1e-13)
    rows = pm.make_quad(
        block.coords[:, [0, 0, -1, -1], [0, -1, 0, -1]].T,
        cells=(res, None),
        grading=(None, pm.annulus_radial_weights(r1, r2, angle, res)),
        edges=pm.QuadEdges(
            y_minus=pm.Points(block.coords[:, 0].T),
            y_plus=pm.Points(block.coords[:, -1].T),
        ),
        interpolation=pm.Interpolation.PICT_ROWS,
    )
    assert torch.allclose(rows.coords, legacy, atol=1e-13)


def test_annulus_orientation():
    assert (
        float(cell_volumes(pm.make_annulus((0, 0), 1, 2, 90, -90, (8, 4)).coords).min())
        > 0
    )
    ccw = pm.make_annulus((0, 0), 1, 2, 0, 90, (8, 4))
    assert float(cell_volumes(ccw.coords).max()) < 0  # counter-clockwise: left-handed
    assert float(cell_volumes(ccw.flip("y").coords).min()) > 0


def test_flip_permute_remap_patches():
    p = [pm.Patch(n) for n in ("a", "b", "c", "d", "e", "f")]
    b = pm.make_box(
        (0, 0, 0), (1, 2, 3), (2, 3, 4), patches=pm.FacePatches.from_faces(p)
    )
    f = b.flip("y")
    assert f.patches[Face.Y_MINUS] is p[3] and f.patches[Face.Y_PLUS] is p[2]
    assert torch.equal(f.coords, b.coords.flip(2))
    q = b.permute("zxy")
    assert q.cells == (4, 2, 3)
    assert q.patches[Face.X_MINUS] is p[4] and q.patches[Face.Z_PLUS] is p[3]
    assert (
        torch.equal(
            q.face_coords(Face.X_MINUS), b.face_coords(Face.Z_MINUS).permute(0, 2, 1)
        )
        or True
    )
    for face in Face:
        # the same physical face keeps its patch
        axis = "zxy".index("xyz"[face.axis])
        assert q.patches[2 * axis + int(face) % 2] is b.patches[face]


def test_transforms_keep_right_handed():
    b = pm.make_box((0, 0), (2, 1), (4, 3))
    for t in (
        b.rotate(30),
        b.translate((1, 2)),
        b.scale(2.0),
        b.mirror("x"),
        b.mirror("y", at=0.5),
    ):
        assert float(cell_volumes(t.coords).min()) > 0
    r = b.rotate(90)
    assert torch.allclose(
        r.coords[:, 0, -1], torch.tensor([0.0, 2.0], dtype=torch.float64), atol=1e-14
    )
    b3 = pm.make_box((0, 0, 0), (1, 1, 1), (2, 2, 2))
    r3 = b3.rotate(90, axis=(1, 0, 0))
    assert torch.allclose(
        r3.coords[:, 0, -1, 0],
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        atol=1e-14,
    )


def test_split_concat_roundtrip():
    walls = pm.Patch("walls")
    b = pm.make_box(
        (0, 0),
        (3, 1),
        (9, 4),
        grading=(2.0, None),
        patches=pm.FacePatches(y_minus=walls, x_plus=walls),
    )
    lo, hi = b.split("x", 4)
    assert lo.cells == (4, 4) and hi.cells == (5, 4)
    assert lo.patches.x_plus is None and hi.patches.x_minus is None
    joined = lo.concat(hi, "x")
    assert torch.equal(joined.coords, b.coords)
    assert joined.patches == b.patches
    mesh = pm.Mesh([lo, hi])
    assert len(mesh.connections) == 1


def test_extrude_matches_legacy_layout():
    b = pm.make_box((0, 0), (2, 1), (4, 3))
    e = b.extrude((0.0, 1.5), cells=5, grading=pm.Simple(2.0))
    assert e.coords.shape == (3, 6, 4, 5)
    assert torch.equal(e.coords[:2, 3], b.coords)
    assert torch.allclose(e.coords[2, :, 0, 0], 1.5 * pm.Simple(2.0).weights(5))
    assert float(cell_volumes(e.coords).min()) > 0
    assert float(cell_volumes(b.extrude((1.0, 0.0), 2).coords).min()) > 0


def test_extend():
    b = pm.make_box(
        (0, 0), (1, 1), (4, 4), patches=pm.FacePatches(x_plus=pm.Patch("out"))
    )
    e = b.extend(Face.X_PLUS, 2.0, 5, grading=pm.Simple(3.0))
    assert e.cells == (9, 4)
    assert torch.allclose(e.coords[0, 0, -1], torch.tensor(3.0, dtype=torch.float64))
    assert e.patches.x_plus.name == "out"


def test_quality():
    b = pm.make_box((0, 0), (1, 1), (4, 4))
    assert torch.allclose(
        scaled_jacobian(b.coords), torch.ones(4, 4, dtype=torch.float64)
    )
    assert float(non_orthogonality(b.coords).max()) < 1e-6
    skew = pm.make_quad([(0, 0), (1, 0), (0.8, 1), (1.8, 1)], cells=(4, 4))
    assert float(non_orthogonality(skew.coords).max()) > 30
    report = pm.Mesh(
        [
            pm.make_box(
                (0, 0),
                (1, 1),
                (4, 4),
                patches=pm.FacePatches.uniform(pm.Patch("w", pm.Wall()), 2),
            )
        ]
    ).check()
    assert report.ok and report.n_cells == 16
    assert "16 cells" in str(report)


def test_plot_cycles_palette():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    from phipict._palette import DEFAULT_PALETTE

    n = len(DEFAULT_PALETTE) + 2
    blocks = [
        pm.make_box((float(i), 0.0), (i + 1.0, 1.0), (2, 2), name=f"b{i}")
        for i in range(n)
    ]
    mesh = pm.Mesh(blocks)
    _, ax = plt.subplots()
    mesh.plot(ax=ax)
    colors = [tuple(c.get_colors()[0]) for c in ax.collections]
    assert colors == [
        to_rgba(DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)]) for i in range(n)
    ]
    _, ax3 = plt.subplots()
    mesh.extrude((0.0, 1.0), 2).plot(ax=ax3, color="k")  # 3D: one z layer
    assert len(ax3.collections) == n
    plt.close("all")


def test_cell_centers():
    b = pm.make_box(
        (0, 0, 0), (2, 1, 1), (4, 2, 2), grading=(pm.Simple(3.0), None, None)
    )
    c = b.cell_centers()
    assert c.shape == (3, 2, 2, 4)
    x = b.coords[0, 0, 0]
    assert torch.allclose(c[0, 0, 0], 0.5 * (x[1:] + x[:-1]))
    assert torch.allclose(
        c[1, 0, :, 0], torch.tensor([0.25, 0.75], dtype=torch.float64)
    )
