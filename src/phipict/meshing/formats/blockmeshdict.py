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

"""OpenFOAM ``blockMeshDict`` files.

blockMesh describes a mesh exactly like a :class:`~phipict.meshing.BlockMesh`:
vertices, hex blocks with cells and grading, curved edges and boundary patches.
Reading one gives the same mesh blockMesh would make, as phiPICT blocks::

    mesh = pm.read_blockmeshdict("system/blockMeshDict")
    mesh.set_bc("inlet", pm.Inflow((1.0, 0.0)))
    mesh.set_bc("outlet", pm.Outflow())

Supported: ``convertToMeters``/``scale``, ``$macros``, ``hex`` blocks with
``simpleGrading`` or ``edgeGrading`` (with multi-grading), ``arc`` (point and
``origin`` forms), ``polyLine``, ``spline`` and ``BSpline`` edges, ``boundary``
(and the old ``patches``) with ``defaultPatch``, and ``cyclic`` patches, which
become periodic connections. A mesh whose blocks have one cell between two
``empty`` patches is read as a 2D mesh.

Patch types map to conditions: ``wall`` to :class:`~phipict.meshing.Wall`,
``symmetry``/``symmetryPlane`` to :class:`~phipict.meshing.FreeSlip`; other
patches (``patch``, ...) are named only, so their condition must be set.

Not supported: ``#calc`` and other directives, projected vertices, edges and
faces, ``mergePatchPairs`` with entries, and collapsed hexes.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from phipict.bc import Face

from ..blockmesh import BlockMesh, Hex, Quad
from ..boundary import BoundarySpec, FacePatches, FreeSlip, Patch, Wall
from ..curves import Arc, BSpline, EdgeShape, Line, Points, Polyline, Spline
from ..grading import (
    Cluster,
    Geometric,
    Grading,
    MultiGrading,
    Segment,
    Simple,
    Symmetric,
    Uniform,
    _Reversed,
    as_grading,
)
from ..mesh import Mesh
from ..quality import cell_volumes
from ..shapes import CornerEdge, Edge, Interpolation, QuadEdges
from .foam import FoamDict, FoamParseError, parse_foam

__all__ = ["load_blockmeshdict", "read_blockmeshdict", "write_blockmeshdict"]

_DTYPE = torch.float64

#: solver corner (i + 2 j + 4 k) -> blockMesh hex vertex
_SOLVER_TO_FOAM = [0, 1, 3, 2, 4, 5, 7, 6]
#: blockMesh edgeGrading order as (start, end) blockMesh vertices
_FOAM_EDGES = [
    (0, 1),
    (3, 2),
    (7, 6),
    (4, 5),
    (0, 3),
    (1, 2),
    (5, 6),
    (4, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]
#: blockMesh vertices of each solver face
_FACE_FOAM_VERTICES = {
    Face.X_MINUS: (0, 3, 7, 4),
    Face.X_PLUS: (1, 2, 6, 5),
    Face.Y_MINUS: (0, 1, 5, 4),
    Face.Y_PLUS: (3, 2, 6, 7),
    Face.Z_MINUS: (0, 1, 2, 3),
    Face.Z_PLUS: (4, 5, 6, 7),
}
_PATCH_BC: dict[str, type[Wall] | type[FreeSlip]] = {
    "wall": Wall,
    "symmetry": FreeSlip,
    "symmetryPlane": FreeSlip,
}


def _entry(d: FoamDict, key: str, default: Any = None) -> Any:
    if key not in d:
        if default is not None:
            return default
        raise FoamParseError(f"The blockMeshDict has no '{key}' entry.")
    return d[key]


def _single(values: list[Any], key: str) -> Any:
    if len(values) != 1:
        raise FoamParseError(f"'{key}' should have a single value, got {values!r}.")
    return values[0]


def _grading(value: Any) -> Grading:
    """A grading from a number or a multi-grading list."""
    if isinstance(value, int | float):
        return Simple(float(value))
    if isinstance(value, list) and all(
        isinstance(v, list) and len(v) == 3 for v in value
    ):
        return MultiGrading(
            [Segment(float(a), float(b), float(c)) for a, b, c in value]
        )
    raise FoamParseError(f"Unsupported grading {value!r}.")


def _point(value: Any, scale: float) -> torch.Tensor:
    if not isinstance(value, list) or len(value) != 3:
        raise FoamParseError(f"Expected a point (x y z), got {value!r}.")
    return torch.tensor([float(v) for v in value], dtype=_DTYPE) * scale


class _Hex:
    """A parsed hex block."""

    def __init__(
        self,
        vertices: list[int],
        cells: list[int],
        gradings: list[Grading],
        zone: str | None,
    ) -> None:
        self.vertices = vertices  # blockMesh order
        self.cells = cells
        self.gradings = gradings  # 12, blockMesh edge order
        self.zone = zone


def _parse_blocks(items: list[Any]) -> list[_Hex]:
    blocks = []
    i = 0
    while i < len(items):
        if items[i] != "hex":
            raise FoamParseError(f"Only 'hex' blocks are supported, got {items[i]!r}.")
        verts = items[i + 1]
        i += 2
        zone = None
        if isinstance(items[i], str):
            zone = items[i]
            i += 1
        cells = items[i]
        i += 1
        kind = items[i]
        grading = items[i + 1]
        i += 2
        if not (isinstance(verts, list) and len(verts) == 8):
            raise FoamParseError(f"A hex needs 8 vertices, got {verts!r}.")
        if len(set(verts)) != 8:
            raise FoamParseError(f"Collapsed hex {verts} is not supported.")
        if not (isinstance(cells, list) and len(cells) == 3):
            raise FoamParseError(f"A hex needs 3 cell counts, got {cells!r}.")
        if kind == "simpleGrading":
            if len(grading) != 3:
                raise FoamParseError(f"simpleGrading needs 3 values, got {grading!r}.")
            gs = [_grading(g) for g in grading]
            per_edge = [gs[0]] * 4 + [gs[1]] * 4 + [gs[2]] * 4
        elif kind == "edgeGrading":
            if len(grading) != 12:
                raise FoamParseError(f"edgeGrading needs 12 values, got {grading!r}.")
            per_edge = [_grading(g) for g in grading]
        else:
            raise FoamParseError(f"Unknown grading type {kind!r}.")
        blocks.append(
            _Hex([int(v) for v in verts], [int(c) for c in cells], per_edge, zone)
        )
    return blocks


def _parse_edges(
    items: list[Any], vertices: torch.Tensor, scale: float
) -> list[tuple[int, int, EdgeShape]]:
    edges: list[tuple[int, int, EdgeShape]] = []
    i = 0
    while i < len(items):
        kind = items[i]
        v0, v1 = int(items[i + 1]), int(items[i + 2])
        i += 3
        shape: EdgeShape
        if kind == "arc":
            if items[i] == "origin":
                i += 1
                if isinstance(items[i], int | float):
                    raise FoamParseError(
                        "arc origin with a radius factor is not supported."
                    )
                shape = Arc(center=_point(items[i], scale))
            else:
                shape = Arc(through=_point(items[i], scale))
            i += 1
        elif kind in ("polyLine", "spline", "simpleSpline", "polySpline", "BSpline"):
            pts = (
                torch.stack([_point(p, scale) for p in items[i]])
                if items[i]
                else torch.zeros(0, 3, dtype=_DTYPE)
            )
            i += 1
            cls = {"polyLine": Polyline, "BSpline": BSpline}.get(kind, Spline)
            shape = cls(pts)
        elif kind == "line":
            shape = Line()
        else:
            raise FoamParseError(f"Edge type {kind!r} is not supported.")
        edges.append((v0, v1, shape))
    return edges


class _PatchInfo:
    def __init__(
        self, name: str, kind: str, faces: list[list[int]], neighbour: str | None
    ) -> None:
        self.name = name
        self.kind = kind
        self.faces = faces
        self.neighbour = neighbour


def _parse_boundary(d: FoamDict) -> list[_PatchInfo]:
    patches = []
    if "boundary" in d:
        items = _single(d["boundary"], "boundary")
        for i in range(0, len(items), 2):
            name, spec = items[i], items[i + 1]
            if not isinstance(spec, dict):
                raise FoamParseError(f"Patch {name!r} needs a dictionary.")
            kind = _single(spec.get("type", ["patch"]), "type")
            faces = _single(spec.get("faces", [[]]), "faces")
            nb = spec.get("neighbourPatch")
            patches.append(
                _PatchInfo(
                    str(name),
                    str(kind),
                    faces,
                    None if nb is None else str(_single(nb, "neighbourPatch")),
                )
            )
    elif "patches" in d:
        items = _single(d["patches"], "patches")
        for i in range(0, len(items), 3):
            kind, name, faces = items[i], items[i + 1], items[i + 2]
            patches.append(_PatchInfo(str(name), str(kind), faces, None))
    return patches


def _read(path: str | Path) -> FoamDict:
    return parse_foam(Path(path).read_text())


def load_blockmeshdict(
    path: str | Path, ndims: int | None = None
) -> tuple[BlockMesh, dict[str, Any]]:
    """The blocks of a blockMeshDict as an unbuilt :class:`BlockMesh`.

    Parameters
    ----------
    path : str or pathlib.Path
        The blockMeshDict.
    ndims : int or None, optional
        2 or 3; None detects 2D meshes from their empty patches.

    Returns
    -------
    BlockMesh
        The blocks with their edges and patches.
    dict
        Information for :func:`read_blockmeshdict`: the cyclic patch pairs and the
        default patch.
    """
    d = _read(path)
    if "mergePatchPairs" in d and any(_single(d["mergePatchPairs"], "mergePatchPairs")):
        raise FoamParseError(
            "mergePatchPairs is not supported (non-conforming interfaces)."
        )
    scale = float(_single(d.get("convertToMeters", d.get("scale", [1.0])), "scale"))
    raw_vertices = _single(_entry(d, "vertices"), "vertices")
    for v in raw_vertices:
        if not isinstance(v, list):
            raise FoamParseError(
                f"Unsupported vertex {v!r} (named or projected vertices)."
            )
    vertices = torch.stack([_point(v, scale) for v in raw_vertices])
    hexes = _parse_blocks(_single(_entry(d, "blocks"), "blocks"))
    edges = _parse_edges(_single(d.get("edges", [[]]), "edges"), vertices, scale)
    patch_infos = _parse_boundary(d)

    # patch of every (block, face), by the vertex sets of the faces
    face_of: dict[frozenset[int], list[tuple[int, Face]]] = {}
    for b, h in enumerate(hexes):
        for face, fv in _FACE_FOAM_VERTICES.items():
            face_of.setdefault(frozenset(h.vertices[k] for k in fv), []).append(
                (b, face)
            )
    patch_faces: dict[tuple[int, Face], _PatchInfo] = {}
    for info in patch_infos:
        for f in info.faces:
            owners = face_of.get(frozenset(int(v) for v in f))
            if not owners:
                raise FoamParseError(
                    f"Face {f} of patch '{info.name}' is not a block face."
                )
            for owner in owners:
                patch_faces[owner] = info

    # 2D: one cell between two empty patches in every block
    empty_axis: list[int | None] = []
    for b in range(len(hexes)):
        axis = None
        for a in range(3):
            infos = [patch_faces.get((b, Face(2 * a + s))) for s in (0, 1)]
            if all(i is not None and i.kind == "empty" for i in infos):
                axis = a
        empty_axis.append(axis)
    if ndims is None:
        ndims = 2 if all(a is not None for a in empty_axis) else 3
    if ndims == 2:
        for b, (h, ea) in enumerate(zip(hexes, empty_axis, strict=True)):
            if ea is None:
                raise FoamParseError(
                    f"Block {b} has no pair of empty patches for a 2D mesh."
                )
            if h.cells[ea] != 1:
                raise FoamParseError(
                    f"Block {b} has {h.cells[ea]} cells between its empty patches."
                )

    patches: dict[str, Patch] = {}
    for info in patch_infos:
        if info.kind == "empty":
            continue
        if info.kind == "cyclic":
            patches[info.name] = Patch(info.name, kind="cyclic")
            continue
        bc_cls = _PATCH_BC.get(info.kind)
        bc: BoundarySpec | None = bc_cls() if bc_cls is not None else None
        patches[info.name] = Patch(info.name, bc, kind=info.kind)

    edge_shapes = {frozenset((v0, v1)): (v0, v1, s) for v0, v1, s in edges}
    bm = BlockMesh()
    normal: int | None = None
    for b, h in enumerate(hexes):
        solver_verts = [h.vertices[_SOLVER_TO_FOAM[c]] for c in range(8)]
        # per-edge gradings (and shapes), keyed by solver corners
        foam_to_solver = {f: s for s, f in enumerate(_SOLVER_TO_FOAM)}
        corner_edges: list[CornerEdge] = []
        for k, (f0, f1) in enumerate(_FOAM_EDGES):
            s0, s1 = foam_to_solver[f0], foam_to_solver[f1]
            g0, g1 = h.vertices[f0], h.vertices[f1]
            shape: EdgeShape = Line()
            known = edge_shapes.get(frozenset((g0, g1)))
            if known is not None:
                shape = known[2] if known[0] == g0 else known[2].reversed()
            corner_edges.append(CornerEdge(s0, s1, Edge(shape, h.gradings[k])))
        faces: list[Patch | None] = [None] * 6
        for face in Face:
            pinfo = patch_faces.get((b, face))
            if pinfo is not None and pinfo.name in patches:
                faces[int(face)] = patches[pinfo.name]
        name = h.zone or f"block{b}"
        if ndims == 3:
            bm.add(
                Hex(
                    [vertices[v] for v in solver_verts],
                    tuple(h.cells),  # type: ignore[arg-type]
                    None,
                    corner_edges,
                    FacePatches.from_faces(faces),
                    name,
                )
            )
            continue
        ax = empty_axis[b]
        assert ax is not None
        # cyclic permutation that moves the empty axis to local z
        perm = [(ax + 1) % 3, (ax + 2) % 3, ax]
        corners3 = torch.stack([vertices[v] for v in solver_verts])
        # corner (i, j) of the quad: local axes perm[0], perm[1] at 0 along perm[2]
        quad_corners = []
        for j in (0, 1):
            for i in (0, 1):
                idx = (i << perm[0]) | (j << perm[1])
                quad_corners.append(corners3[idx])
        # the global axis the empty direction points along
        span = corners3[1 << ax] - corners3[0]
        n_axis = int(span.abs().argmax())
        if normal is None:
            normal = n_axis
        elif normal != n_axis:
            raise FoamParseError("The empty patches of the blocks are not parallel.")
        keep = [(n_axis + 1) % 3, (n_axis + 2) % 3]
        quad2 = [p[keep] for p in quad_corners]
        # edges and gradings of the two in-plane axes at 0 along the empty axis
        by_corners = {
            (ce.start, ce.end): ce.edge if isinstance(ce.edge, Edge) else Edge(ce.edge)
            for ce in corner_edges
        }

        c = [(i << perm[0]) | (j << perm[1]) for j in (0, 1) for i in (0, 1)]
        qe = QuadEdges(
            y_minus=_quad_edge(by_corners, keep, c[0], c[1]),
            y_plus=_quad_edge(by_corners, keep, c[2], c[3]),
            x_minus=_quad_edge(by_corners, keep, c[0], c[2]),
            x_plus=_quad_edge(by_corners, keep, c[1], c[3]),
        )
        pf = [
            faces[2 * perm[0]],
            faces[2 * perm[0] + 1],
            faces[2 * perm[1]],
            faces[2 * perm[1] + 1],
        ]
        bm.add(
            Quad(
                quad2,
                (h.cells[perm[0]], h.cells[perm[1]]),
                None,
                qe,
                FacePatches.from_faces(pf),
                name,
            )
        )
    done: set[str] = set()
    for p in patch_infos:
        if p.kind != "cyclic" or p.name in done:
            continue
        if p.neighbour is None or p.neighbour not in patches:
            raise FoamParseError(f"Cyclic patch '{p.name}' has no neighbourPatch.")
        bm.add_periodic(patches[p.name], patches[p.neighbour])
        done |= {p.name, p.neighbour}
    return bm, {"default": d.get("defaultPatch")}


def _quad_edge(
    by_corners: dict[tuple[int, int], Edge], keep: list[int], c0: int, c1: int
) -> Edge:
    """The in-plane edge from hex corner ``c0`` to ``c1``, projected to 2D."""
    if (c0, c1) in by_corners:
        e = by_corners[(c0, c1)]
    else:
        e = by_corners[(c1, c0)]
        grading = e.grading.reversed() if isinstance(e.grading, Grading) else e.grading
        e = Edge(e.shape.reversed(), grading)
    return Edge(_project_shape(e.shape, keep), e.grading)


def _project_shape(shape: EdgeShape, keep: list[int]) -> EdgeShape:
    if isinstance(shape, Line):
        return shape
    if isinstance(shape, Arc):
        if shape.through is not None:
            return Arc(through=shape.through[keep])
        assert shape.center is not None
        return Arc(center=shape.center[keep])
    if isinstance(shape, Polyline):
        return type(shape)(shape.points[:, keep])
    raise FoamParseError(f"Cannot project the edge {shape!r} to 2D.")


def read_blockmeshdict(
    path: str | Path, ndims: int | None = None, rel_tol: float = 1e-4
) -> Mesh:
    """Read a blockMeshDict as a :class:`~phipict.meshing.Mesh`.

    Parameters
    ----------
    path : str or pathlib.Path
        The blockMeshDict.
    ndims : int or None, optional
        2 or 3; None reads meshes with one cell between empty patches as 2D.
    rel_tol : float, optional
        Face matching tolerance, see :class:`~phipict.meshing.Mesh`. Default is
        1e-4.

    Returns
    -------
    Mesh
        The mesh with its connections, periodic (cyclic) connections and patches.

    Raises
    ------
    FoamParseError
        If the file uses unsupported features.
    """
    bm, info = load_blockmeshdict(path, ndims)
    mesh = bm.build(rel_tol=rel_tol)
    # blockMesh keeps the orientation of 3D hexes; a 2D block whose empty axis
    # points against the plane normal comes out mirrored
    if mesh.ndims == 2 and any(float(cell_volumes(b.coords).mean()) < 0 for b in mesh):
        mesh = mesh.transform(
            lambda b: b.flip(0) if float(cell_volumes(b.coords).mean()) < 0 else b
        )
    default = info["default"]
    free = mesh.free_faces()
    if free:
        name = "defaultFaces"
        kind = "empty"
        if isinstance(default, dict):
            name = str(_single(default.get("name", [name]), "name"))
            kind = str(_single(default.get("type", [kind]), "type"))
        bc_cls = _PATCH_BC.get(kind)
        patch = Patch(name, bc_cls() if bc_cls is not None else None, kind=kind)
        mesh.assign_patch(free, patch)
    return mesh


def _fmt(x: float) -> str:
    return f"{x:.15g}"


def _fmt_point(p: Sequence[float] | torch.Tensor) -> str:
    return "(" + " ".join(_fmt(float(v)) for v in p) + ")"


def _fmt_grading(g: Grading, cells: int) -> str:
    if isinstance(g, Uniform):
        return "1"
    if isinstance(g, Simple):
        return _fmt(g.ratio)
    if isinstance(g, Symmetric):
        return f"((0.5 0.5 {_fmt(g.ratio)}) (0.5 0.5 {_fmt(1.0 / g.ratio)}))"
    if isinstance(g, MultiGrading):
        segs = " ".join(
            f"({_fmt(s.length)} {_fmt(s.cells)} {_fmt(s.ratio)})" for s in g.segments
        )
        return f"({segs})"
    if isinstance(g, Geometric) and g.cluster != Cluster.BOTH:
        total = g.ratio ** (cells - 1)
        return _fmt(total if g.cluster == Cluster.START else 1.0 / total)
    if isinstance(g, _Reversed):
        inner = g.grading
        if isinstance(inner, MultiGrading):
            mirrored = [
                Segment(s.length, s.cells, 1.0 / s.ratio)
                for s in reversed(inner.segments)
            ]
            return _fmt_grading(MultiGrading(mirrored), cells)
        rev = inner.reversed()
        if not isinstance(rev, _Reversed):
            return _fmt_grading(rev, cells)
    raise ValueError(f"The grading {g!r} has no blockMesh equivalent.")


def _fmt_edge(v0: int, v1: int, shape: EdgeShape, lift: Any) -> str | None:
    if isinstance(shape, Line):
        return None
    if isinstance(shape, Arc):
        if shape.through is not None:
            return f"    arc {v0} {v1} {_fmt_point(lift(shape.through))}"
        assert shape.center is not None
        return f"    arc {v0} {v1} origin {_fmt_point(lift(shape.center))}"
    if isinstance(shape, Points):
        pts = shape.points[1:-1]
        kind = "polyLine"
    elif isinstance(shape, Spline):
        pts, kind = shape.points, "spline"
    elif isinstance(shape, BSpline):
        pts, kind = shape.points, "BSpline"
    elif isinstance(shape, Polyline):
        pts, kind = shape.points, "polyLine"
    else:
        raise ValueError(f"The edge {shape!r} has no blockMesh equivalent.")
    inner = " ".join(_fmt_point(lift(p)) for p in pts)
    return f"    {kind} {v0} {v1} ({inner})"


def write_blockmeshdict(
    block_mesh: BlockMesh, path: str | Path, thickness: float = 1.0
) -> None:
    """Write the blocks of a :class:`~phipict.meshing.BlockMesh` as a blockMeshDict.

    Cells and gradings are written as blockMesh ``edgeGrading``, curved edges as
    ``arc``/``polyLine``/``spline``/``BSpline``, and patches with their
    condition's type (``wall``, ``symmetryPlane``) or the type they were read
    with. A 2D mesh is written with one cell of ``thickness`` in z between
    ``empty`` patches.

    Parameters
    ----------
    block_mesh : BlockMesh
        The blocks.
    path : str or pathlib.Path
        Output file.
    thickness : float, optional
        Extent in z of a 2D mesh. Default is 1.

    Raises
    ------
    ValueError
        If a grading or edge has no blockMesh equivalent (e.g. tanh), or a block
        uses PICT row interpolation.
    """
    resolved = block_mesh.resolve()
    ndims = len(resolved[0].cells)
    neighbours = {a.name: b.name for a, b in block_mesh.periodic}
    neighbours.update({b.name: a.name for a, b in block_mesh.periodic})
    n_verts = 1 + max(v for r in resolved for v in r.vertices)
    verts = torch.zeros(n_verts, ndims, dtype=_DTYPE)
    for r in resolved:
        verts[r.vertices] = r.corners
    if ndims == 2:
        all_verts = torch.cat(
            [
                torch.cat([verts, torch.zeros(n_verts, 1, dtype=_DTYPE)], 1),
                torch.cat(
                    [verts, torch.full((n_verts, 1), thickness, dtype=_DTYPE)], 1
                ),
            ]
        )

        def lift(p: torch.Tensor) -> torch.Tensor:
            return torch.cat([p.to(_DTYPE), torch.zeros(1, dtype=_DTYPE)])
    else:
        all_verts = verts

        def lift(p: torch.Tensor) -> torch.Tensor:
            return p.to(_DTYPE)

    block_lines: list[str] = []
    edge_lines: dict[frozenset[int], str] = {}
    patch_faces: dict[str, tuple[str, list[list[int]]]] = {}
    for r in resolved:
        if r.interpolation != Interpolation.TFI:
            raise ValueError(
                f"Block '{r.name}' uses {r.interpolation}, which blockMesh lacks."
            )
        if ndims == 2:
            solver = r.vertices + [v + n_verts for v in r.vertices]
            cells = r.cells + [1]
        else:
            solver = list(r.vertices)
            cells = list(r.cells)
        foam = [solver[_SOLVER_TO_FOAM.index(f)] for f in range(8)]
        gradings = []
        for f0, f1 in _FOAM_EDGES:
            s0, s1 = _SOLVER_TO_FOAM.index(f0), _SOLVER_TO_FOAM.index(f1)
            a = (s0 ^ s1).bit_length() - 1
            if a == 2 and ndims == 2:
                gradings.append("1")
                continue
            others = [b for b in range(ndims) if b != a]
            lo = min(s0, s1)
            e = sum(((lo >> b) & 1) << k for k, b in enumerate(others))
            edge = r.edges[a][e]
            g = as_grading(edge.grading)
            gradings.append(_fmt_grading(g, cells[a]))
            layers = (0, n_verts) if ndims == 2 else (0,)
            for off in layers:
                g0, g1 = solver[lo] + off, solver[lo | (1 << a)] + off
                key = frozenset((g0, g1))
                if key not in edge_lines:
                    line = _fmt_edge(g0, g1, edge.shape, lift)
                    if line is not None:
                        edge_lines[key] = line
                if ndims == 3 or off == n_verts:
                    break
        block_lines.append(
            f"    hex ({' '.join(map(str, foam))}) {r.name} "
            f"({' '.join(map(str, cells))}) "
            f"edgeGrading ({' '.join(gradings)})"
        )
        for face, patch in r.patches.items():
            fv = [foam[k] for k in _FACE_FOAM_VERTICES[face]]
            if patch.name in neighbours:
                kind = "cyclic"
            else:
                kind = {Wall: "wall", FreeSlip: "symmetryPlane"}.get(
                    type(patch.bc), patch.kind or "patch"
                )  # type: ignore[call-overload]
            patch_faces.setdefault(patch.name, (kind, []))[1].append(fv)
        if ndims == 2:
            for face in (Face.Z_MINUS, Face.Z_PLUS):
                fv = [foam[k] for k in _FACE_FOAM_VERTICES[face]]
                patch_faces.setdefault("frontAndBack", ("empty", []))[1].append(fv)
    out = [
        "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
        "    class       dictionary;\n"
        "    object      blockMeshDict;\n}\n",
        "// written by phipict.meshing\n",
        "convertToMeters 1;\n",
        "vertices\n(\n"
        + "\n".join(f"    {_fmt_point(v)}" for v in all_verts)
        + "\n);\n",
        "blocks\n(\n" + "\n".join(block_lines) + "\n);\n",
        "edges\n(\n" + "\n".join(edge_lines.values()) + "\n);\n",
        "boundary\n(",
    ]
    for name, (kind, faces) in patch_faces.items():
        fl = "\n".join(f"            ({' '.join(map(str, f))})" for f in faces)
        extra = (
            f"\n        neighbourPatch {neighbours[name]};" if kind == "cyclic" else ""
        )
        out.append(
            f"    {name}\n    {{\n        type {kind};{extra}\n"
            f"        faces\n        (\n{fl}\n        );\n    }}"
        )
    out.append(");\n\nmergePatchPairs\n(\n);\n")
    Path(path).write_text("\n".join(out))
