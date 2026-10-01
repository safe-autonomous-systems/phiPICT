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

"""blockMesh-like mesh building from block topology.

Blocks are described by their corners first and discretised together, so that
neighbours agree without bookkeeping::

    bm = pm.BlockMesh()
    bm.add(pm.Quad(corners_a, cells=(32, 16), grading=(None, pm.Symmetric(10))))
    bm.add(pm.Quad(corners_b, cells=(48, None)))   # 16 cells in y, from block a
    bm.add_edge((1, 0), (1, 1), pm.Arc(through=(1.1, 0.5)))   # shared curved edge
    mesh = bm.build()

Corners closer than the merge tolerance are the same vertex, and edges between
the same two vertices are the same edge. Along chains of shared edges:

* the number of cells must agree, and an unknown number (None) is inferred;
* an axis without a grading takes the grading of a neighbour's shared edge,
  mirrored if the edge runs the other way, else it is uniform;
* a curved edge defined by one block or by :meth:`BlockMesh.add_edge` applies to
  every block with that edge.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from .boundary import FacePatches, Patch
from .curves import EdgeShape, Line, Points
from .grading import Explicit, Grading, GradingLike, as_grading
from .mesh import Mesh
from .shapes import (
    CornerEdge,
    Edge,
    Interpolation,
    QuadEdges,
    build_block,
    hex_edges,
)

__all__ = ["BlockMesh", "Hex", "Quad", "ResolvedBlock"]

_DTYPE = torch.float64
PointLike = Sequence[float] | torch.Tensor


@dataclass(eq=False)
class Quad:
    """A 2D block of a :class:`BlockMesh`.

    Attributes
    ----------
    corners : sequence of point
        ``(-x-y, +x-y, -x+y, +x+y)``.
    cells : tuple of int or None
        Cells along x and y; None to infer from neighbours.
    grading : sequence or Grading or float or None
        Grading per axis (or one for both); None to infer from neighbours.
    edges : QuadEdges or None
        Curved edges or per-edge gradings.
    patches : FacePatches or None
        Patches of the faces.
    name : str
        Name of the block.
    interpolation : Interpolation
        Interpolation of the interior.
    """

    corners: Sequence[PointLike]
    cells: tuple[int | None, int | None] = (None, None)
    grading: Sequence[GradingLike] | GradingLike = None
    edges: QuadEdges | None = None
    patches: FacePatches | None = None
    name: str = "block"
    interpolation: Interpolation = Interpolation.TFI

    @property
    def ndims(self) -> int:
        """Number of spatial dimensions."""
        return 2

    def edge_list(self) -> list[list[Edge]]:
        """``[axis][e]`` edges."""
        return (self.edges or QuadEdges()).by_axis()


@dataclass(eq=False)
class Hex:
    """A 3D block of a :class:`BlockMesh`.

    Attributes
    ----------
    corners : sequence of point
        Eight corners in solver order (``i + 2 j + 4 k`` for ``(x, y, z) = (i, j,
        k)``).
    cells : tuple of int or None
        Cells along x, y and z; None to infer from neighbours.
    grading : sequence or Grading or float or None
        Grading per axis (or one for all); None to infer from neighbours.
    edges : sequence of CornerEdge
        Curved edges or per-edge gradings.
    patches : FacePatches or None
        Patches of the faces.
    name : str
        Name of the block.
    """

    corners: Sequence[PointLike]
    cells: tuple[int | None, int | None, int | None] = (None, None, None)
    grading: Sequence[GradingLike] | GradingLike = None
    edges: Sequence[CornerEdge] = field(default_factory=tuple)
    patches: FacePatches | None = None
    name: str = "block"

    @property
    def ndims(self) -> int:
        """Number of spatial dimensions."""
        return 3

    @property
    def interpolation(self) -> Interpolation:
        """Interpolation of the interior (always transfinite in 3D)."""
        return Interpolation.TFI

    def edge_list(self) -> list[list[Edge]]:
        """``[axis][e]`` edges."""
        return hex_edges(self.edges)


BlockSpec = Quad | Hex


@dataclass
class ResolvedBlock:
    """A block of a :class:`BlockMesh` with everything inferred.

    Attributes
    ----------
    corners : torch.Tensor
        Corner coordinates ``[2**dims, dims]`` in solver order.
    vertices : list of int
        Index of each corner among the merged vertices.
    cells : list of int
        Cells per axis.
    gradings : list of Grading
        Grading per axis.
    edges : list of list of Edge
        ``[axis][e]`` edges, each with its shape and final grading.
    patches : FacePatches
        Patches of the faces.
    name : str
        Name of the block.
    interpolation : Interpolation
        Interpolation of the interior.
    """

    corners: torch.Tensor
    vertices: list[int]
    cells: list[int]
    gradings: list[Grading]
    edges: list[list[Edge]]
    patches: FacePatches
    name: str
    interpolation: Interpolation


def _edge_corners(a: int, e: int, ndims: int) -> tuple[int, int]:
    """Local corner indices at the start and end of edge ``e`` of axis ``a``."""
    others = [b for b in range(ndims) if b != a]
    lo = sum(((e >> k) & 1) << b for k, b in enumerate(others))
    return lo, lo | (1 << a)


def _is_explicit_shape(edge: Edge) -> bool:
    return not isinstance(edge.shape, Line)


def _reverse(edge: Edge) -> Edge:
    return Edge(
        edge.shape.reversed(),
        None if edge.grading is None else as_grading(edge.grading).reversed(),
    )


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(self, x: tuple[int, int]) -> tuple[int, int]:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: tuple[int, int], b: tuple[int, int]) -> None:
        self.parent[self.find(a)] = self.find(b)


class BlockMesh:
    """Builds a :class:`Mesh` from block corners, edges, cells and gradings.

    Parameters
    ----------
    merge_tol : float, optional
        Corners closer than this, relative to the size of the mesh, are merged.
        Default is 1e-9.
    """

    def __init__(self, merge_tol: float = 1e-9) -> None:
        self.merge_tol = merge_tol
        self._specs: list[BlockSpec] = []
        self._edges: list[tuple[torch.Tensor, torch.Tensor, Edge]] = []
        self._periodic: list[tuple[Patch, Patch]] = []

    @property
    def blocks(self) -> tuple[BlockSpec, ...]:
        """The block specifications."""
        return tuple(self._specs)

    def add(self, spec: BlockSpec) -> int:
        """Add a block and return its index."""
        if self._specs and spec.ndims != self._specs[0].ndims:
            raise ValueError("All blocks of a BlockMesh need the same dimension.")
        self._specs.append(spec)
        return len(self._specs) - 1

    @property
    def periodic(self) -> tuple[tuple[Patch, Patch], ...]:
        """The periodic patch pairs."""
        return tuple(self._periodic)

    def add_periodic(self, patch_a: Patch, patch_b: Patch) -> None:
        """Make the faces of two patches periodic partners (OpenFOAM ``cyclic``).

        The faces of ``patch_b`` must coincide with those of ``patch_a`` after a
        translation; they are connected instead of getting a condition.
        """
        self._periodic.append((patch_a, patch_b))

    def add_edge(
        self, start: PointLike, end: PointLike, edge: Edge | EdgeShape
    ) -> None:
        """Give the edge between two corners a shape and/or grading.

        It applies to every block with that edge, in the right direction.
        """
        e = edge if isinstance(edge, Edge) else Edge(edge)
        self._edges.append(
            (
                torch.as_tensor(start, dtype=_DTYPE),
                torch.as_tensor(end, dtype=_DTYPE),
                e,
            )
        )

    # -----------------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------------

    def _merge(self) -> tuple[torch.Tensor, list[list[int]]]:
        pts = [
            torch.stack([torch.as_tensor(p, dtype=_DTYPE).flatten() for p in s.corners])
            for s in self._specs
        ]
        for s, p in zip(self._specs, pts, strict=True):
            if p.shape != (2**s.ndims, s.ndims):
                raise ValueError(
                    f"Block '{s.name}' needs {2**s.ndims} corners with {s.ndims} "
                    "coordinates."
                )
        allp = torch.cat(pts)
        scale = float((allp.amax(0) - allp.amin(0)).norm()) or 1.0
        tol = self.merge_tol * scale
        unique: list[torch.Tensor] = []
        ids: list[list[int]] = []
        for p in pts:
            row = []
            for q in p:
                for k, u in enumerate(unique):
                    if float((u - q).abs().max()) <= tol:
                        row.append(k)
                        break
                else:
                    unique.append(q)
                    row.append(len(unique) - 1)
            ids.append(row)
        return torch.stack(unique), ids

    def _vertex(self, verts: torch.Tensor, p: torch.Tensor) -> int:
        scale = float((verts.amax(0) - verts.amin(0)).norm()) or 1.0
        d = (verts - p.view(1, -1)).abs().amax(1)
        k = int(d.argmin())
        if float(d[k]) > self.merge_tol * scale:
            raise ValueError(f"Edge end {p.tolist()} is not a block corner.")
        return k

    def build(self, rel_tol: float = 1e-4) -> Mesh:
        """Discretise the blocks and connect them.

        Parameters
        ----------
        rel_tol : float, optional
            Face matching tolerance of the connections, see :class:`Mesh`.
            Default is 1e-4.

        Returns
        -------
        Mesh
            The mesh.

        Raises
        ------
        ValueError
            If cell counts conflict or cannot be inferred, or corners of blocks do
            not form valid edges.
        """
        periodic = {id(p) for pair in self._periodic for p in pair}
        blocks = []
        faces: dict[int, list[torch.Tensor]] = {}
        for r in self.resolve():
            block = build_block(
                r.corners, r.cells, r.gradings, r.edges, None, r.name, r.interpolation
            )
            kept: list[Patch | None] = []
            for f, patch in enumerate(r.patches.as_list()[: 2 * block.ndims]):
                if patch is not None and id(patch) in periodic:
                    faces.setdefault(id(patch), []).append(block.face_coords(f))
                    patch = None
                kept.append(patch)
            block.patches = FacePatches.from_faces(kept)
            blocks.append(block)
        mesh = Mesh(blocks, rel_tol=rel_tol)
        for a, b in self._periodic:
            if id(a) not in faces or id(b) not in faces:
                raise ValueError(
                    f"Periodic patches '{a.name}'/'{b.name}' have no faces."
                )
            ca = torch.cat([f.reshape(f.shape[0], -1) for f in faces[id(a)]], 1).mean(1)
            cb = torch.cat([f.reshape(f.shape[0], -1) for f in faces[id(b)]], 1).mean(1)
            mesh.make_periodic(translation=(cb - ca).tolist(), rel_tol=rel_tol)
        return mesh

    def resolve(self) -> list[ResolvedBlock]:
        """The blocks with merged corners, inferred cells, gradings and shared edges.

        Returns
        -------
        list of ResolvedBlock
            One per block, in order.
        """
        if not self._specs:
            raise ValueError("The BlockMesh has no blocks.")
        verts, ids = self._merge()
        n = self._specs[0].ndims
        n_edges = 2 ** (n - 1)

        # edges of every block, keyed by their (sorted) vertex pair
        edges = [s.edge_list() for s in self._specs]
        keys: list[list[list[tuple[int, int]]]] = []  # [block][axis][e] -> (v0, v1)
        for b, s in enumerate(self._specs):
            kb = []
            for a in range(n):
                ka = []
                for e in range(n_edges):
                    lo, hi = _edge_corners(a, e, n)
                    v0, v1 = ids[b][lo], ids[b][hi]
                    if v0 == v1:
                        raise ValueError(
                            f"Block '{s.name}' has a collapsed edge along {'xyz'[a]}."
                        )
                    ka.append((v0, v1))
                kb.append(ka)
            keys.append(kb)

        # shared edge shapes: from add_edge, then from the blocks
        shapes: dict[frozenset[int], tuple[int, Edge]] = {}
        for p0, p1, given in self._edges:
            w0, w1 = self._vertex(verts, p0), self._vertex(verts, p1)
            shapes[frozenset((w0, w1))] = (w0, given)
        for b in range(len(self._specs)):
            for a in range(n):
                for e in range(n_edges):
                    edge = edges[b][a][e]
                    v0, v1 = keys[b][a][e]
                    key = frozenset((v0, v1))
                    if _is_explicit_shape(edge) and key not in shapes:
                        shapes[key] = (v0, Edge(edge.shape))

        def oriented(key_edge: tuple[int, Edge], v0: int) -> Edge:
            start, edge = key_edge
            return edge if start == v0 else _reverse(edge)

        for b in range(len(self._specs)):
            for a in range(n):
                for e in range(n_edges):
                    edge = edges[b][a][e]
                    v0, v1 = keys[b][a][e]
                    key = frozenset((v0, v1))
                    if not _is_explicit_shape(edge) and key in shapes:
                        shared = oriented(shapes[key], v0)
                        edges[b][a][e] = Edge(shared.shape, edge.grading)

        # cell counts: union of block axes along shared edges
        uf = _UnionFind()
        by_key: dict[frozenset[int], list[tuple[int, int, int]]] = {}
        for b in range(len(self._specs)):
            for a in range(n):
                uf.find((b, a))
                for e in range(n_edges):
                    by_key.setdefault(frozenset(keys[b][a][e]), []).append((b, a, e))
        for users in by_key.values():
            for b, a, _ in users[1:]:
                uf.union((users[0][0], users[0][1]), (b, a))

        gradings: list[list[GradingLike]] = []
        for s in self._specs:
            g = s.grading
            if isinstance(g, Sequence) and not isinstance(g, str):
                if len(g) != n:
                    raise ValueError(f"Block '{s.name}' needs one grading per axis.")
                gradings.append(list(g))
            else:
                gradings.append([g] * n)

        fixed: dict[tuple[int, int], dict[int, str]] = {}
        for b, s in enumerate(self._specs):
            for a in range(n):
                root = uf.find((b, a))
                srcs = fixed.setdefault(root, {})
                c = s.cells[a]
                if c is not None:
                    srcs.setdefault(int(c), f"cells of block '{s.name}'")
                axis_grading = gradings[b][a]
                if isinstance(axis_grading, Explicit):
                    srcs.setdefault(axis_grading.cells, f"grading of block '{s.name}'")
                for e in range(n_edges):
                    edge = edges[b][a][e]
                    if isinstance(edge.shape, Points):
                        srcs.setdefault(
                            edge.shape.cells, f"edge points of block '{s.name}'"
                        )
                    elif isinstance(edge.grading, Explicit):
                        srcs.setdefault(
                            edge.grading.cells, f"edge grading of block '{s.name}'"
                        )
        counts: list[list[int]] = []
        for b, s in enumerate(self._specs):
            row = []
            for a in range(n):
                srcs = fixed[uf.find((b, a))]
                if len(srcs) > 1:
                    listed = "; ".join(
                        f"{c} from {src}" for c, src in sorted(srcs.items())
                    )
                    raise ValueError(
                        f"Conflicting cell counts along axis {'xyz'[a]} of block "
                        f"'{s.name}' and the blocks sharing its edges: {listed}."
                    )
                if not srcs:
                    raise ValueError(
                        f"Cannot infer the cells along axis {'xyz'[a]} of block "
                        f"'{s.name}': neither it nor a block sharing its edges "
                        "gives them."
                    )
                row.append(next(iter(srcs)))
            counts.append(row)

        # gradings: unspecified axes take a neighbour's grading along a shared edge
        def edge_grading(b: int, a: int, e: int) -> tuple[GradingLike, int] | None:
            g = edges[b][a][e].grading
            if g is None:
                g = gradings[b][a]
            if g is None:
                return None
            return g, keys[b][a][e][0]

        changed = True
        while changed:
            changed = False
            for b in range(len(self._specs)):
                for a in range(n):
                    if gradings[b][a] is not None:
                        continue
                    for e in range(n_edges):
                        v0, v1 = keys[b][a][e]
                        for ob, oa, oe in by_key[frozenset((v0, v1))]:
                            if ob == b:
                                continue
                            found = edge_grading(ob, oa, oe)
                            if found is None or isinstance(
                                edges[ob][oa][oe].shape, Points
                            ):
                                continue
                            g, start = found
                            gg = as_grading(g)
                            gradings[b][a] = gg if start == v0 else gg.reversed()
                            changed = True
                            break
                        if gradings[b][a] is not None:
                            break

        out = []
        for b, s in enumerate(self._specs):
            axis_gradings = [as_grading(g) for g in gradings[b]]
            full = [
                [
                    Edge(
                        e.shape,
                        as_grading(e.grading)
                        if e.grading is not None
                        else axis_gradings[a],
                    )
                    for e in edges[b][a]
                ]
                for a in range(n)
            ]
            out.append(
                ResolvedBlock(
                    verts[ids[b]],
                    ids[b],
                    counts[b],
                    axis_gradings,
                    full,
                    s.patches or FacePatches(),
                    s.name,
                    s.interpolation,
                )
            )
        return out
