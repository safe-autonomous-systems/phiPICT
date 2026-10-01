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

"""The fluidgym meshes, rebuilt with ``phipict.meshing``, match the legacy grids.

The references in ``tests/meshing/data`` were made by the fluidgym environment
builders with ``phipict.grid`` (see ``make_references.py`` there). Each case is
rebuilt here the way a user of the new API would write it: vertices must agree up
to the float32 rounding of the legacy grids, and the face types and block
connections (with their axis codes, on a GPU) must be the same.
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import phipict
import phipict.meshing as pm
from phipict import Face

DATA = Path(__file__).parent / "data"
BOTH, START, END = pm.Cluster.BOTH, pm.Cluster.START, pm.Cluster.END
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def parabola(lo: float, hi: float) -> pm.boundary.VelocityProfile:
    """A parabolic inflow profile in y between ``lo`` and ``hi``, like fluidgym's."""

    def profile(p: torch.Tensor) -> torch.Tensor:
        t = (p[:, 1] - lo) / (hi - lo)
        u = 4.0 * t * (1.0 - t)
        return torch.stack([u] + [torch.zeros_like(u)] * (p.shape[1] - 1), 1)

    return profile


def load(case: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    data = np.load(DATA / f"{case}.npz")
    return json.loads(str(data["meta"])), {
        k: data[k] for k in data.files if k != "meta"
    }


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def cylinder(ndims: int, p: dict[str, Any]) -> pm.Mesh:
    """Cylinder in a channel: an O-grid of four ring+quad blocks and a wake block."""
    r, ct, qx = p["cylinder_radius"], p["circle_thickness"], p["quad_thickness_x"]
    off, n = p["cylinder_offset_y"], p["circle_resolution_angular"]
    r2 = r + ct
    qy = qx + off
    rox, roy = r + ct + qx, r + ct + qy
    top, bot = roy + off, roy - off
    ri = math.sin(math.radians(45)) * r2
    x_max = p["domain_length"] - rox
    wake_grading = pm.Geometric(p["vortex_street_refinement_base"], BOTH)
    arc = pm.Arc(center=(0.0, 0.0))

    cyl = pm.Patch("cylinder", pm.Wall())
    walls = pm.Patch("walls", pm.Wall())
    inlet = pm.Patch("inlet", pm.Inflow(parabola(-bot, top)))
    outlet = pm.Patch("outlet", pm.Outflow())

    def ring(start: float) -> pm.MeshBlock:
        # clockwise quarter rings: x along the angle, y outwards
        return pm.make_annulus((0.0, 0.0), r, r2, start, -90.0, cells=(n, None))

    n_rad = ring(135).cells[1]
    nq = math.ceil(qy / ct * n_rad) - 1  # radial cells of the outer quads

    ring_top = ring(135)
    quad_top = pm.make_quad(
        [(-ri, ri), (ri, ri), (-rox, top), (rox, top)],
        cells=(n, nq),
        edges=pm.QuadEdges(y_minus=arc),
    )
    block_top = ring_top.concat(quad_top, "y")
    block_top.name = "BlockCylinderTop"

    ring_right = ring(45).permute("yx").flip("y")  # x outwards, y up
    quad_right = pm.make_quad(
        [(ri, -ri), (rox, -bot), (ri, ri), (rox, top)],
        cells=(nq, n),
        grading=(None, wake_grading),
        edges=pm.QuadEdges(x_minus=pm.Edge(arc, pm.Uniform())),
    )
    block_right = ring_right.concat(quad_right, "x")
    block_right.name = "BlockCylinderRight"

    ring_bot = ring(-45).flip("x").flip("y")  # x to the right, y up (inwards)
    quad_bot = pm.make_quad(
        [(-rox, -bot), (rox, -bot), (-ri, -ri), (ri, -ri)],
        cells=(n, nq),
        edges=pm.QuadEdges(y_plus=arc),
    )
    block_bot = quad_bot.concat(ring_bot, "y")
    block_bot.name = "BlockCylinderBottom"

    ring_left = ring(-135).permute("yx").flip("x")  # x inwards, y up
    quad_left = pm.make_quad(
        [(-rox, -bot), (-ri, -ri), (-rox, top), (-ri, ri)],
        cells=(nq, n),
        edges=pm.QuadEdges(x_plus=arc),
    )
    block_left = quad_left.concat(ring_left, "x")
    block_left.name = "BlockCylinderLeft"

    wake = pm.make_box(
        (rox, -bot),
        (x_max, top),
        cells=(int((nq + 1) / qy * 18), n),
        grading=(None, wake_grading),
        name="BlockVortexStreet",
    )

    block_left.patches = pm.FacePatches(x_minus=inlet, x_plus=cyl)
    block_top.patches = pm.FacePatches(y_minus=cyl, y_plus=walls)
    block_right.patches = pm.FacePatches(x_minus=cyl)
    block_bot.patches = pm.FacePatches(y_minus=walls, y_plus=cyl)
    wake.patches = pm.FacePatches(x_plus=outlet, y_minus=walls, y_plus=walls)

    mesh = pm.Mesh([block_left, block_top, block_right, block_bot, wake])
    if ndims == 3:
        mesh = mesh.extrude((-2.0, 2.0), cells=n, periodic=True)
    return mesh


def _run(points: torch.Tensor, a: int, b: int) -> torch.Tensor:
    """Points ``a`` to ``b`` (inclusive), in either direction."""
    step = 1 if b >= a else -1
    return points[torch.arange(a, b + step, step)]


def airfoil(
    ndims: int,
    p: dict[str, Any],
    arrays: dict[str, np.ndarray],
    split: dict[str, list[int]],
) -> pm.Mesh:
    """Airfoil in a channel: a C-type grid around the surface points plus wake."""
    H, L = p["H"], p["L"]
    h = H / 2
    offset_left, front_x = 1.5, 0.5
    normal_res = 96 // p["resolution_div"]
    fine_at_start = pm.Geometric(0.97, START)  # PICT: make_weights_exp(0.97, "START")
    fine_at_end = pm.Geometric(0.97, END)

    # environment logic: the surface points and their tail spacing (float32 as there)
    pts32 = torch.from_numpy(arrays["airfoil_points"])
    end = pts32[0]
    end_ext = end + torch.stack(
        [torch.linalg.vector_norm(pts32[1] - pts32[0]), torch.tensor(0.0)]
    )
    ext = torch.cat([end_ext[None], pts32, end_ext[None]])
    min_size = float(torch.linalg.vector_norm(ext[1:] - ext[:-1], dim=1).min())
    sizes, dist = [min_size], min_size
    while dist < h:
        sizes.append(sizes[-1] * p["tail_grow_mul"])
        dist += sizes[-1]
    tail = pm.Explicit([0.0] + (np.cumsum(sizes) / dist).tolist())

    pts = pts32.double()
    top_pts = _run(pts, *split["top"])
    front_pts = _run(pts, *split["front"])
    bot_pts = _run(pts, *split["bot"])
    t0, t1 = top_pts[0].tolist(), top_pts[-1].tolist()
    b0, b1 = bot_pts[0].tolist(), bot_pts[-1].tolist()

    wall = pm.Patch("walls", pm.Wall())
    foil = pm.Patch("airfoil", pm.Wall())
    inlet = pm.Patch("inlet", pm.Inflow(parabola(-h, h)))
    outlet = pm.Patch("outlet", pm.Outflow())

    left = pm.make_quad(
        [(-offset_left, -h), (-front_x, -h), (-offset_left, h), (-front_x, h)],
        cells=(int(0.75 * normal_res) - 1, len(front_pts) - 1),
        patches=pm.FacePatches(x_minus=inlet, y_minus=wall, y_plus=wall),
        name="LeftBlock",
    )
    front = pm.make_quad(
        [(-front_x, -h), b0, (-front_x, h), t0],
        cells=(normal_res - 1, None),
        grading=(fine_at_start, None),
        edges=pm.QuadEdges(x_plus=pm.Points(front_pts)),
        patches=pm.FacePatches(x_plus=foil),
        name="AirfoilFront",
    )
    top = pm.make_quad(
        [t0, t1, (-front_x, h), (t1[0], h)],
        cells=(None, normal_res - 1),
        grading=(None, fine_at_end),
        edges=pm.QuadEdges(y_minus=pm.Points(top_pts)),
        patches=pm.FacePatches(y_minus=foil, y_plus=wall),
        name="AirfoilTop",
    )
    bottom = pm.make_quad(
        [(-front_x, -h), (b1[0], -h), b0, b1],
        cells=(None, normal_res - 1),
        grading=(None, fine_at_start),
        edges=pm.QuadEdges(y_plus=pm.Points(bot_pts)),
        # the legacy builder closes -y implicitly with +y (periodic partner)
        patches=pm.FacePatches(y_minus=wall, y_plus=foil),
        name="AirfoilBot",
    )
    tail_upper = pm.make_quad(
        [t1, (L, t1[1]), (t1[0], h), (L, h)],
        cells=(None, normal_res - 1),
        grading=(tail, fine_at_end),
        patches=pm.FacePatches(x_plus=outlet, y_plus=wall),
        name="TailUpper",
    )
    tail_lower = pm.make_quad(
        [(b1[0], -h), (L, -h), b1, (L, b1[1])],
        cells=(None, normal_res - 1),
        grading=(tail, fine_at_start),
        patches=pm.FacePatches(x_plus=outlet, y_minus=wall),
        name="TailLower",
    )
    mesh = pm.Mesh([left, front, top, bottom, tail_upper, tail_lower])
    if ndims == 3:
        mesh = mesh.extrude((-h, h), cells=p["res_z"], periodic=True)
    return mesh


def rbc(ndims: int, p: dict[str, Any]) -> pm.Mesh:
    """Rayleigh-Benard cell: a box graded towards the plates, periodic in x (and z)."""
    plates = pm.Patch("plates", pm.Wall())
    box = pm.make_box(
        (0.0, -0.5),
        (p["L"], 0.5),
        cells=(p["x"], p["y"]),
        grading=(None, pm.Geometric(p["base"], BOTH)),
        patches=pm.FacePatches(y_minus=plates, y_plus=plates),
        name="RBCBlock",
    )
    mesh = pm.Mesh([box])
    mesh.make_periodic("x")
    if ndims == 3:
        mesh = mesh.extrude((0.0, p["L"]), cells=p["x"], periodic=True)
    return mesh


def build(case: str) -> tuple[pm.Mesh, dict[str, Any], dict[str, np.ndarray]]:
    meta, arrays = load(case)
    ndims = meta["ndims"]
    if case.startswith("cylinder"):
        mesh = cylinder(ndims, meta["params"])
    elif case.startswith("airfoil"):
        mesh = airfoil(ndims, meta["params"], arrays, meta["split"])
    else:
        mesh = rbc(ndims, meta["params"])
    return mesh, meta, arrays


CASES = ["cylinder2d", "cylinder3d", "airfoil2d", "airfoil3d", "rbc2d", "rbc3d"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_vertices_match_legacy(case):
    mesh, meta, arrays = build(case)
    assert [b.name for b in mesh] == [b["name"] for b in meta["blocks"]]
    for i, block in enumerate(mesh):
        ref = torch.from_numpy(arrays[f"coords_{i}"]).double()
        assert block.coords.shape == ref.shape, block.name
        # the legacy grids are float32
        tol = 4 * torch.finfo(torch.float32).eps * max(1.0, float(ref.abs().max()))
        err = float((block.coords - ref).abs().max())
        assert err <= tol, f"{block.name}: max deviation {err:.3g} > {tol:.3g}"


def _faces(mesh: pm.Mesh) -> list[list[tuple[Any, ...]]]:
    """Face kinds of a mesh: connected block, periodic, or fixed (static?)."""
    out: list[list[tuple[Any, ...]]] = [[()] * (2 * mesh.ndims) for _ in mesh]
    for c in mesh.connections:
        if c.is_periodic_axis:
            out[c.block_a][int(c.face_a)] = ("PERIODIC",)
            out[c.block_b][int(c.face_b)] = ("PERIODIC",)
        else:
            out[c.block_a][int(c.face_a)] = ("CONNECTED", c.block_b)
            out[c.block_b][int(c.face_b)] = ("CONNECTED", c.block_a)
    for i, b in enumerate(mesh):
        for face, patch in b.patches.items():
            out[i][int(face)] = ("FIXED", _static(patch.bc))
    return out


def _static(spec: pm.BoundarySpec | None) -> bool:
    """Whether a condition gives a face one velocity (not per cell)."""
    if isinstance(spec, pm.Outflow):
        return False
    velocity = getattr(spec, "velocity", None)
    return not callable(velocity) and (
        not isinstance(velocity, torch.Tensor) or velocity.dim() <= 2
    )


def _ref_faces(meta: dict[str, Any]) -> list[list[tuple[Any, ...]]]:
    out = []
    for b in meta["blocks"]:
        row: list[tuple[Any, ...]] = []
        for f in b["faces"]:
            if f["type"] == "CONNECTED":
                row.append(("CONNECTED", f["block"]))
            elif f["type"] == "FIXED":
                row.append(("FIXED", f["static"]))
            else:
                row.append((f["type"],))
        out.append(row)
    return out


@pytest.mark.parametrize("case", CASES)
def test_topology_matches_legacy(case):
    mesh, meta, _ = build(case)
    assert mesh.check().ok, mesh.check()
    assert _faces(mesh) == _ref_faces(meta)


@needs_cuda
@pytest.mark.parametrize("case", CASES)
def test_domain_matches_legacy(case):
    """The solver sees the same boundaries and connection axis codes."""
    mesh, meta, _ = build(case)
    domain = mesh.get_domain(viscosity=0.01, dtype=torch.float32, device="cuda")
    for b, ref in zip(domain.getBlocks(), meta["blocks"], strict=True):
        assert b.name == ref["name"]
        for f, rf in enumerate(ref["faces"]):
            bound: Any = b.getBoundary(f)
            assert bound.type.name == rf["type"], (b.name, Face(f).name)
            if rf["type"] == "CONNECTED":
                names = [ob.name for ob in domain.getBlocks()]
                assert names.index(bound.getConnectedBlock().name) == rf["block"]
                assert [int(a) for a in bound.axes] == rf["axes"], (
                    b.name,
                    Face(f).name,
                )
            elif rf["type"] == "FIXED":
                assert bound.velocityType.name == rf["velocityType"]
                assert bool(bound.isVelocityStatic) == rf["static"]


@needs_cuda
@pytest.mark.parametrize("case", ["cylinder2d", "airfoil2d", "rbc2d"])
def test_domain_runs(case):
    mesh, _, _ = build(case)
    domain = mesh.get_domain(
        viscosity=0.01, dtype=torch.float64, device="cuda", passive_scalar_channels=0
    )
    sim = phipict.Simulation(domain=domain, dt=0.01)
    for _ in range(2):
        sim.single_step()
    for b in domain.getBlocks():
        assert torch.isfinite(b.velocity).all()
        for f in range(2 * mesh.ndims):
            bound = b.getBoundary(f)
            if bound.type.name == "FIXED":
                assert torch.isfinite(bound.velocity).all(), (b.name, f)


_NO_GRID = """
import importlib.abc, importlib.util, sys

import phipict  # the solver core itself still uses phipict.grid.resample

class NoGrid(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "phipict.grid" or name.startswith("phipict.grid."):
            raise ImportError(name + " is blocked")
        return None

grid = [m for m in sys.modules if m == "phipict.grid" or m.startswith("phipict.grid.")]
for name in grid:
    del sys.modules[name]
sys.meta_path.insert(0, NoGrid())

spec = importlib.util.spec_from_file_location("refs", sys.argv[1])
refs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refs)
meshes = [refs.build(case)[0] for case in refs.CASES]
import tempfile
from phipict.io import export
with tempfile.TemporaryDirectory() as tmp:
    export.write_vtk(meshes[0], tmp + "/mesh")
    refs.pm.read_vtk(tmp + "/mesh.vtm")
print("ok")
"""


def test_does_not_need_legacy_grid():
    """Meshing and export work with ``phipict.grid`` unavailable.

    The solver core still imports ``phipict.grid.resample`` (for the image
    output), so ``phipict`` is imported first; afterwards every ``phipict.grid``
    module is removed and blocked, and all meshes are built and exported.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", _NO_GRID, __file__],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("ok")
