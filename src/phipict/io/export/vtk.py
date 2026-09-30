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

"""VTK XML output of meshes and solutions, for ParaView and pyvista.

A mesh or domain is written as a multi-block file ``<path>.vtm`` that references
one structured grid (``.vts``) per block in the directory ``<path>/``::

    from phipict.io import export

    export.write_vtk(domain, "out/cylinder")            # fields of the domain
    export.write_vtk(mesh, "out/cylinder_mesh")         # mesh with cell quality
    export.write_vtk(domain, "out/cylinder", mesh=mesh) # with named patches

Cell fields are written as VTK cell data; 2D blocks become grids of zero
thickness in z. Boundary patches of a :class:`~phipict.meshing.Mesh` are written
as a second group of the multi-block file, one grid per patch face. The data are
little-endian binary, zlib-compressed by default. Only numpy and the standard
library are needed.
"""

from __future__ import annotations

import os
import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Flag, auto
from pathlib import Path
from typing import TYPE_CHECKING
from xml.sax.saxutils import quoteattr

import numpy as np
import torch

from phipict import _C

if TYPE_CHECKING:
    from phipict.meshing import Mesh

__all__ = ["Fields", "write_vtk", "write_vts"]


class Fields(Flag):
    """Cell fields to export."""

    VELOCITY = auto()
    PRESSURE = auto()
    PASSIVE_SCALAR = auto()
    VELOCITY_SOURCE = auto()
    EPOT = auto()
    VISCOSITY = auto()
    #: Cell volume and scaled Jacobian, computed from the vertices.
    QUALITY = auto()
    NONE = 0
    ALL = VELOCITY | PRESSURE | PASSIVE_SCALAR | VELOCITY_SOURCE | EPOT | VISCOSITY


_VTK_TYPES = {
    np.dtype(np.float32): "Float32",
    np.dtype(np.float64): "Float64",
    np.dtype(np.int32): "Int32",
    np.dtype(np.int64): "Int64",
    np.dtype(np.int8): "Int8",
    np.dtype(np.uint8): "UInt8",
}


@dataclass
class _Array:
    name: str
    data: np.ndarray  # [n, components]


def _encode(data: np.ndarray, compress: bool) -> bytes:
    raw = (
        np.ascontiguousarray(data)
        .astype(data.dtype.newbyteorder("<"), copy=False)
        .tobytes()
    )
    if not compress:
        return struct.pack("<Q", len(raw)) + raw
    comp = zlib.compress(raw, 6)
    # one block: [#blocks, block size, last block size, compressed sizes...]
    return struct.pack("<QQQQ", 1, len(raw), len(raw), len(comp)) + comp


def write_vts(
    path: str | os.PathLike[str],
    coords: torch.Tensor,
    cell_data: Sequence[_Array] = (),
    compress: bool = True,
) -> None:
    """Write one structured grid as a ``.vts`` file.

    Parameters
    ----------
    path : str or os.PathLike
        Output file.
    coords : torch.Tensor
        Vertices ``[dims, (nz + 1,) ny + 1, nx + 1]``.
    cell_data : sequence
        Named cell arrays of shape ``[cells, components]`` in x-fastest order.
    compress : bool, optional
        Whether to zlib-compress the data. Default is True.
    """
    d = coords.shape[0]
    c = coords.detach().cpu().double()
    if d == 2:
        c = torch.cat([c, torch.zeros_like(c[:1])])[:, None]
    nz1, ny1, nx1 = c.shape[1:]
    points = c.permute(1, 2, 3, 0).reshape(-1, 3).numpy()
    extent = f"0 {nx1 - 1} 0 {ny1 - 1} 0 {nz1 - 1}"
    blobs: list[bytes] = []
    offset = 0

    def array_xml(name: str | None, data: np.ndarray) -> str:
        nonlocal offset
        blob = _encode(data, compress)
        blobs.append(blob)
        attrs = f'type="{_VTK_TYPES[data.dtype]}"'
        if name is not None:
            attrs += f" Name={quoteattr(name)}"
        comps = data.shape[1] if data.ndim == 2 else 1
        xml = f'<DataArray {attrs} NumberOfComponents="{comps}" format="appended" offset="{offset}"/>'
        offset += len(blob)
        return xml

    cells_xml = "".join(array_xml(a.name, a.data) for a in cell_data)
    points_xml = array_xml(None, points)
    compressor = ' compressor="vtkZLibDataCompressor"' if compress else ""
    header = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="StructuredGrid" version="1.0" byte_order="LittleEndian" '
        f'header_type="UInt64"{compressor}>\n'
        f'<StructuredGrid WholeExtent="{extent}">\n'
        f'<Piece Extent="{extent}">\n'
        f"<CellData>{cells_xml}</CellData>\n"
        f"<Points>{points_xml}</Points>\n"
        "</Piece>\n</StructuredGrid>\n"
        '<AppendedData encoding="raw">\n_'
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        for blob in blobs:
            f.write(blob)
        f.write(b"\n</AppendedData>\n</VTKFile>\n")


def _cell_array(
    name: str, t: torch.Tensor, batch_index: int, ndims: int
) -> _Array | None:
    """A block field ``[B, C, (z,) y, x]`` as cell data ``[cells, C]``."""
    if t.dim() != ndims + 2:
        return None  # static values (e.g. a [1, dims] source) are not fields
    b = batch_index if t.shape[0] > 1 else 0
    data = t[b].detach().cpu()
    comps = data.shape[0]
    data = data.reshape(comps, -1).T
    if name in ("velocity", "velocitySource") and comps == 2:
        data = torch.cat([data, torch.zeros_like(data[:, :1])], 1)  # VTK vectors are 3D
    arr = data.numpy()
    if arr.dtype not in _VTK_TYPES:
        arr = arr.astype(np.float64)
    return _Array(name, arr)


def _block_fields(
    block: _C.Block, fields: Fields, batch_index: int, ndims: int
) -> list[_Array]:
    out: list[_Array | None] = []
    if Fields.VELOCITY in fields:
        out.append(_cell_array("velocity", block.velocity, batch_index, ndims))
    if Fields.PRESSURE in fields:
        out.append(_cell_array("pressure", block.pressure, batch_index, ndims))
    if Fields.PASSIVE_SCALAR in fields and block.hasPassiveScalar():
        assert block.passiveScalar is not None
        out.append(
            _cell_array("passiveScalar", block.passiveScalar, batch_index, ndims)
        )
    if Fields.VELOCITY_SOURCE in fields and block.hasVelocitySource():
        assert block.velocitySource is not None
        out.append(
            _cell_array("velocitySource", block.velocitySource, batch_index, ndims)
        )
    if Fields.EPOT in fields and block.hasEpot():
        assert block.epot is not None
        out.append(_cell_array("epot", block.epot, batch_index, ndims))
    if Fields.VISCOSITY in fields and block.hasViscosity():
        assert block.viscosity is not None
        out.append(_cell_array("viscosity", block.viscosity, batch_index, ndims))
    return [a for a in out if a is not None]


def _quality(coords: torch.Tensor) -> list[_Array]:
    from phipict.meshing.quality import cell_volumes, scaled_jacobian

    return [
        _Array("cellVolume", cell_volumes(coords).reshape(-1, 1).numpy()),
        _Array("scaledJacobian", scaled_jacobian(coords).reshape(-1, 1).numpy()),
    ]


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)


def write_vtk(
    target: _C.Domain | Mesh,
    path: str | os.PathLike[str],
    *,
    fields: Fields = Fields.ALL,
    mesh: Mesh | None = None,
    batch_index: int = 0,
    compress: bool = True,
) -> Path:
    """Write a domain or mesh as a VTK multi-block file.

    Parameters
    ----------
    target : phipict.Domain or phipict.meshing.Mesh
        What to write. A domain needs vertex coordinates on its blocks.
    path : str or os.PathLike
        Output path; ``.vtm`` is added if missing, the blocks go to a directory of
        the same name without the extension.
    fields : Fields, optional
        Cell fields to write (for a domain). Default is all present fields; add
        ``Fields.QUALITY`` for cell volume and scaled Jacobian. A mesh always gets
        the quality fields.
    mesh : Mesh or None, optional
        The mesh the domain was made from, to also write its boundary patches.
        Default is None.
    batch_index : int, optional
        Environment to write from a batched domain. Default is 0.
    compress : bool, optional
        Whether to zlib-compress the data. Default is True.

    Returns
    -------
    pathlib.Path
        Path of the ``.vtm`` file.
    """
    from phipict.meshing import Mesh

    vtm = Path(path)
    if vtm.suffix != ".vtm":
        vtm = vtm.with_name(vtm.name + ".vtm")
    folder = vtm.with_suffix("")
    folder.mkdir(parents=True, exist_ok=True)

    datasets: list[str] = []
    if isinstance(target, Mesh):
        mesh = target
        blocks: list[tuple[str, torch.Tensor, list[_Array]]] = [
            (b.name, b.coords, _quality(b.coords)) for b in target
        ]
    else:
        ndims = target.getSpatialDims()
        blocks = []
        for sb in target.getBlocks():
            if not sb.hasVertexCoordinates():
                raise ValueError(
                    f"Block '{sb.name}' has no vertex coordinates to export."
                )
            coords = sb.vertexCoordinates
            assert coords is not None
            coords = coords[0].detach().cpu().double()
            arrays = _block_fields(sb, fields, batch_index, ndims)
            if Fields.QUALITY in fields:
                arrays += _quality(coords)
            blocks.append((sb.name, coords, arrays))

    for i, (name, coords, arrays) in enumerate(blocks):
        file = folder / f"{i}_{_safe(name)}.vts"
        write_vts(file, coords, arrays, compress)
        rel = os.path.relpath(file, vtm.parent)
        datasets.append(
            f'<DataSet index="{i}" name={quoteattr(name)} file={quoteattr(rel)}/>'
        )

    boundary: list[str] = []
    if mesh is not None:
        from phipict.meshing.block import grid_dim

        k = 0
        for block in mesh:
            for face, patch in block.patches.items():
                dim = grid_dim(face.axis, block.ndims)
                idx = 0 if int(face) % 2 == 0 else block.coords.shape[dim] - 1
                slab = block.coords.narrow(dim, idx, 1)
                file = folder / f"boundary_{k}_{_safe(patch.name)}.vts"
                write_vts(file, slab, (), compress)
                rel = os.path.relpath(file, vtm.parent)
                label = f"{patch.name}:{block.name}:{face.name}"
                boundary.append(
                    f'<DataSet index="{k}" name={quoteattr(label)} file={quoteattr(rel)}/>'
                )
                k += 1

    body = '<Block index="0" name="blocks">' + "".join(datasets) + "</Block>"
    if boundary:
        body += '<Block index="1" name="boundary">' + "".join(boundary) + "</Block>"
    vtm.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="vtkMultiBlockDataSet" version="1.0" byte_order="LittleEndian">\n'
        f"<vtkMultiBlockDataSet>{body}</vtkMultiBlockDataSet>\n"
        "</VTKFile>\n"
    )
    return vtm
