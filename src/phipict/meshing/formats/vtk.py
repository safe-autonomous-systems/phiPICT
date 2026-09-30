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

"""Structured grids from VTK files.

:func:`read_vtk` reads XML structured grids (``.vts``) and multi-block files of
them (``.vtm``), such as those of :func:`phipict.io.export.write_vtk` or
ParaView, with ascii, base64 or raw appended data, optionally zlib-compressed.
Other VTK files (e.g. legacy ``.vtk``) are read with pyvista, if installed.
"""

from __future__ import annotations

import base64
import re
import struct
import zlib
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
import torch

from phipict.bc import Face

from ..block import MeshBlock
from ..boundary import Patch
from ..mesh import Mesh

__all__ = ["read_vtk"]

_NP_TYPES = {
    "Float32": np.float32,
    "Float64": np.float64,
    "Int8": np.int8,
    "UInt8": np.uint8,
    "Int16": np.int16,
    "UInt16": np.uint16,
    "Int32": np.int32,
    "UInt32": np.uint32,
    "Int64": np.int64,
    "UInt64": np.uint64,
}


class _VTSFile:
    def __init__(self, path: Path) -> None:
        raw = path.read_bytes()
        # the appended section is binary; parse the XML before it
        marker = raw.find(b"<AppendedData")
        self.appended = b""
        self.encoding = "raw"
        if marker >= 0:
            start = raw.index(b">", marker) + 1
            m = re.search(rb'encoding="(\w+)"', raw[marker:start])
            if m:
                self.encoding = m.group(1).decode()
            under = raw.index(b"_", start)
            end = raw.rfind(b"</AppendedData>")
            self.appended = raw[under + 1 : end]
            xml = raw[:marker] + b"</VTKFile>"
        else:
            xml = raw
        self.root = ElementTree.fromstring(xml)
        self.header = "UInt64" if self.root.get("header_type") == "UInt64" else "UInt32"
        self.compressed = self.root.get("compressor") is not None
        self.little = self.root.get("byte_order", "LittleEndian") == "LittleEndian"

    def _uint(self) -> tuple[str, int]:
        e = "<" if self.little else ">"
        return (e + "Q", 8) if self.header == "UInt64" else (e + "I", 4)

    def _decode_blocks(self, data: bytes, b64: bool) -> tuple[bytes, int]:
        """Payload of a (possibly compressed) binary array and its encoded length."""
        fmt, size = self._uint()
        if not self.compressed:
            if b64:
                head = base64.b64decode(data[: 4 * ((size + 2) // 3)])[:size]
                n = struct.unpack(fmt, head)[0]
                full = base64.b64decode(data[: 4 * ((size + n + 2) // 3)])
                return full[size : size + n], 0
            n = struct.unpack(fmt, data[:size])[0]
            return data[size : size + n], size + n
        if b64:
            first = base64.b64decode(data[: 4 * ((3 * size + 2) // 3)])
            nblocks = struct.unpack(fmt, first[:size])[0]
            hlen = (3 + nblocks) * size
            header = base64.b64decode(data[: 4 * ((hlen + 2) // 3)])[:hlen]
            sizes = struct.unpack(fmt[0] + fmt[1] * (3 + nblocks), header)
            body_start = 4 * ((hlen + 2) // 3)
            body = base64.b64decode(data[body_start:])
        else:
            nblocks = struct.unpack(fmt, data[:size])[0]
            hlen = (3 + nblocks) * size
            sizes = struct.unpack(fmt[0] + fmt[1] * (3 + nblocks), data[:hlen])
            body = data[hlen:]
        out = []
        pos = 0
        for c in sizes[3:]:
            out.append(zlib.decompress(body[pos : pos + c]))
            pos += c
        return b"".join(out), hlen + pos

    def array(self, el: ElementTree.Element) -> np.ndarray:
        dtype = np.dtype(_NP_TYPES[el.get("type", "Float32")])
        dtype = dtype.newbyteorder("<" if self.little else ">")
        comps = int(el.get("NumberOfComponents", "1"))
        fmt = el.get("format", "ascii")
        if fmt == "ascii":
            values = np.array((el.text or "").split(), dtype=dtype)
        elif fmt == "binary":
            payload, _ = self._decode_blocks((el.text or "").strip().encode(), b64=True)
            values = np.frombuffer(payload, dtype=dtype)
        elif fmt == "appended":
            data = self.appended[int(el.get("offset", "0")) :]
            if self.encoding == "base64":
                payload, _ = self._decode_blocks(data.strip(), b64=True)
            else:
                payload, _ = self._decode_blocks(data, b64=False)
            values = np.frombuffer(payload, dtype=dtype)
        else:
            raise ValueError(f"Unknown VTK data format {fmt!r}.")
        return values.reshape(-1, comps).astype(np.float64)

    def coords(self) -> torch.Tensor:
        grid = self.root.find("StructuredGrid")
        if grid is None:
            raise ValueError("Not a VTK structured grid file.")
        piece = grid.find("Piece")
        assert piece is not None
        ext = [
            int(v)
            for v in (piece.get("Extent") or grid.get("WholeExtent") or "").split()
        ]
        nx, ny, nz = ext[1] - ext[0] + 1, ext[3] - ext[2] + 1, ext[5] - ext[4] + 1
        pts_el = piece.find("Points/DataArray")
        assert pts_el is not None
        pts = self.array(pts_el)
        c = torch.from_numpy(pts.T.copy()).reshape(3, nz, ny, nx)
        return c


def _from_coords(c: torch.Tensor, name: str) -> torch.Tensor:
    """Solver-layout vertices from ``[3, nz, ny, nx]`` VTK points."""
    if c.shape[1] == 1:
        if float(c[2].abs().max()) > 0 and float((c[2] - c[2].mean()).abs().max()) > 0:
            raise ValueError(
                f"Grid '{name}' is a single layer that is not planar in z."
            )
        return c[:2, 0]
    return c


def _datasets(path: Path) -> list[tuple[str, Path, str]]:
    """``(group, file, name)`` of every dataset in a .vtm file."""
    root = ElementTree.parse(path).getroot()
    mb = root.find("vtkMultiBlockDataSet")
    if mb is None:
        raise ValueError(f"{path} is not a VTK multi-block file.")
    out = []

    def walk(el: ElementTree.Element, group: str) -> None:
        for child in el:
            if child.tag == "Block":
                walk(child, child.get("name", group))
            elif child.tag == "DataSet" and child.get("file"):
                out.append(
                    (group, path.parent / child.get("file", ""), child.get("name", ""))
                )

    walk(mb, "")
    return out


def _read_pyvista(path: Path) -> list[tuple[str, torch.Tensor]]:
    try:
        import pyvista as pv  # type: ignore[import-not-found]
    except ImportError as err:
        raise ValueError(
            f"Reading {path.suffix} files needs pyvista (pip install phipict[mesh])."
        ) from err
    data: Any = pv.read(str(path))
    items = (
        [(data.get_block_name(i) or f"block{i}", data[i]) for i in range(data.n_blocks)]
        if isinstance(data, pv.MultiBlock)
        else [(path.stem, data)]
    )
    out = []
    for name, grid in items:
        if not isinstance(grid, pv.StructuredGrid):
            raise ValueError(f"'{name}' is not a structured grid.")
        nx, ny, nz = grid.dimensions
        pts = torch.from_numpy(np.asarray(grid.points, dtype=np.float64).T.copy())
        out.append((name, pts.reshape(3, nz, ny, nx)))
    return out


def read_vtk(path: str | Path, rel_tol: float = 1e-4) -> Mesh:
    """Read structured grids as a mesh.

    Every grid becomes a block; coinciding faces are connected. Grids of a
    ``boundary`` group written by :func:`phipict.io.export.write_vtk` restore the
    patches (without conditions). Grids that are one vertex thick in z and flat
    are read as 2D.

    Parameters
    ----------
    path : str or pathlib.Path
        A ``.vtm`` or ``.vts`` file, or any file pyvista reads.
    rel_tol : float, optional
        Face matching tolerance, see :class:`~phipict.meshing.Mesh`. Default is
        1e-4.

    Returns
    -------
    Mesh
        The mesh.
    """
    path = Path(path)
    grids: list[tuple[str, torch.Tensor]] = []
    patch_faces: list[tuple[str, str, str]] = []
    if path.suffix == ".vtm":
        for group, file, name in _datasets(path):
            if group == "boundary":
                parts = name.split(":")
                if len(parts) == 3:
                    patch_faces.append((parts[0], parts[1], parts[2]))
                continue
            grids.append((name or file.stem, _VTSFile(file).coords()))
    elif path.suffix == ".vts":
        grids.append((path.stem, _VTSFile(path).coords()))
    else:
        grids = _read_pyvista(path)
    blocks = [MeshBlock(_from_coords(c, n), n) for n, c in grids]
    mesh = Mesh(blocks, auto_connect=False, rel_tol=rel_tol)
    patches: dict[str, Patch] = {}
    for patch_name, block_name, face_name in patch_faces:
        patch = patches.setdefault(patch_name, Patch(patch_name))
        mesh.assign_patch([(mesh.index(block_name), Face[face_name])], patch)
    mesh.auto_connect()
    return mesh
