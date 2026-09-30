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

"""VTK export of meshes and domains, and reading the files back."""

import struct
import zlib
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
import torch

import phipict
import phipict.meshing as pm
from phipict import Hook, Hooks
from phipict.io import export

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _mesh(ndims: int) -> pm.Mesh:
    walls = pm.Patch("walls", pm.Wall())
    fp = pm.FacePatches(y_minus=walls, y_plus=walls)
    lo, hi = (0.0,) * ndims, (1.0,) * ndims
    a = pm.box(
        lo,
        hi,
        (4, 3, 2)[:ndims],
        grading=(None, pm.Symmetric(3.0), None)[:ndims],
        patches=fp,
        name="left",
    )
    b = pm.box(
        (1.0,) + lo[1:],
        (2.5,) + hi[1:],
        (5, 3, 2)[:ndims],
        grading=(None, pm.Symmetric(3.0), None)[:ndims],
        patches=fp,
        name="right",
    )
    mesh = pm.Mesh([a, b.rotate(0.0)])
    mesh.make_periodic("x")
    if ndims == 3:
        mesh.make_periodic("z")
    return mesh


def _read_vts(path: Path) -> dict[str, np.ndarray]:
    """A small independent reader for the appended raw format that is written."""
    raw = path.read_bytes()
    start = raw.index(b"<AppendedData")
    under = raw.index(b"_", raw.index(b">", start)) + 1
    root = ElementTree.fromstring(raw[:start] + b"</VTKFile>")
    compressed = root.get("compressor") is not None
    out = {}
    for el in root.iter("DataArray"):
        pos = under + int(el.get("offset"))
        if compressed:
            n, _, _, csize = struct.unpack("<QQQQ", raw[pos : pos + 32])
            data = zlib.decompress(raw[pos + 32 : pos + 32 + csize])
            assert n == 1
        else:
            (nbytes,) = struct.unpack("<Q", raw[pos : pos + 8])
            data = raw[pos + 8 : pos + 8 + nbytes]
        dtype = {"Float64": "<f8", "Float32": "<f4"}[el.get("type")]
        comps = int(el.get("NumberOfComponents"))
        out[el.get("Name") or "Points"] = np.frombuffer(data, dtype).reshape(-1, comps)
    piece = root.find("StructuredGrid/Piece")
    out["extent"] = np.array([int(v) for v in piece.get("Extent").split()])
    return out


@pytest.mark.parametrize("ndims", [2, 3])
@pytest.mark.parametrize("compress", [True, False])
def test_mesh_roundtrip(tmp_path, ndims, compress):
    mesh = _mesh(ndims)
    vtm = export.write_vtk(mesh, tmp_path / "mesh", compress=compress)
    assert vtm.name == "mesh.vtm" and vtm.exists()
    files = sorted((tmp_path / "mesh").glob("*.vts"))
    blocks = [f for f in files if not f.name.startswith("boundary")]
    assert len(blocks) == 2
    data = _read_vts(blocks[0])
    c = mesh.blocks[0].coords
    pts = data["Points"]
    if ndims == 2:
        assert np.all(pts[:, 2] == 0)
    np.testing.assert_array_equal(pts[:, :ndims], c.reshape(ndims, -1).T.numpy())
    assert data["extent"].tolist()[1::2][:ndims] == list(mesh.blocks[0].cells)
    assert data["cellVolume"].shape == (mesh.blocks[0].n_cells, 1)
    back = pm.read_vtk(vtm)
    assert [b.name for b in back] == ["left", "right"]
    for x, y in zip(mesh, back, strict=True):
        assert torch.equal(x.coords, y.coords)
        assert x.patches.y_minus is not None and y.patches.y_minus.name == "walls"
    assert len([c for c in back.connections if c.translation is None]) == 1


@needs_cuda
def test_domain_fields(tmp_path):
    mesh = _mesh(2)
    domain = mesh.get_domain(viscosity=0.1, dtype=torch.float32, device="cuda")
    block = domain.getBlocks()[1]
    vel = torch.arange(
        block.velocity.numel(), dtype=torch.float32, device="cuda"
    ).reshape(block.velocity.shape)
    block.setVelocity(vel)
    vtm = export.write_vtk(
        domain,
        tmp_path / "sol",
        fields=export.Fields.ALL | export.Fields.QUALITY,
        mesh=mesh,
    )
    data = _read_vts(tmp_path / "sol" / "1_right.vts")
    v = data["velocity"]
    assert v.shape == (15, 3) and np.all(v[:, 2] == 0)
    np.testing.assert_array_equal(v[:, :2], vel[0].reshape(2, -1).T.cpu().numpy())
    assert "pressure" in data and "passiveScalar" in data and "scaledJacobian" in data
    root = ElementTree.parse(vtm).getroot()
    names = [d.get("name") for d in root.iter("DataSet")]
    assert "walls:left:Y_MINUS" in names


@needs_cuda
def test_series_hook(tmp_path):
    mesh = _mesh(2)
    domain = mesh.get_domain(
        viscosity=0.1, dtype=torch.float64, device="cuda", passive_scalar_channels=0
    )
    series = export.VTKSeries(tmp_path / "run", mesh=mesh)
    sim = phipict.Simulation(
        domain=domain, dt=0.01, hooks=Hooks().append(Hook.POST, series.hook(every=2))
    )
    for _ in range(5):
        sim.single_step()
    assert len(series.entries) == 3
    pvd = ElementTree.parse(tmp_path / "run.pvd").getroot()
    sets = pvd.findall("Collection/DataSet")
    assert len(sets) == 3
    times = [float(s.get("timestep")) for s in sets]
    assert times == sorted(times) and times[0] < times[-1]
    assert all((tmp_path / s.get("file")).exists() for s in sets)


def test_pyvista_reads_files(tmp_path):
    pv = pytest.importorskip("pyvista")
    mesh = _mesh(3)
    vtm = export.write_vtk(mesh, tmp_path / "mesh")
    data = pv.read(str(vtm))
    grid = data["blocks"]["left"]
    assert grid.n_cells == mesh.blocks[0].n_cells
    np.testing.assert_allclose(
        grid.points, mesh.blocks[0].coords.reshape(3, -1).T.numpy()
    )
