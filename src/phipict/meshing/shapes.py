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

"""Single blocks from corners, edges, resolution and grading.

Each builder returns a discretised :class:`~phipict.meshing.MeshBlock`::

    pm.box((0, -1), (10, 1), cells=(128, 64), grading=(None, pm.Symmetric(20)))
    pm.quad(corners, cells=(32, 16), edges=pm.QuadEdges(y_minus=pm.Arc(center=(0, 0))))
    pm.annulus((0, 0), 0.5, 1.0, start_angle=135, angle=-90, cells=(24, None))

Corners are listed in the solver's order, x fastest: ``(-x-y, +x-y, -x+y, +x+y)``
in 2D, followed by the same four at ``+z`` in 3D. Each axis takes a number of
cells and a grading (a :class:`~phipict.meshing.Grading`, a number for
:class:`~phipict.meshing.Simple`, or None for uniform). An edge may override
the grading of its axis and have a curved shape; :class:`~phipict.meshing.Points`
edges fix their vertices, and with them the number of cells of their axis.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

import torch

from .block import MeshBlock
from .boundary import FacePatches
from .curves import EdgeShape, Line, Points
from .grading import Explicit, Grading, GradingLike, as_grading
from .tfi import tfi_2d, tfi_3d

__all__ = [
    "CornerEdge",
    "Edge",
    "EdgeLike",
    "Interpolation",
    "QuadEdges",
    "annulus",
    "annulus_radial_weights",
    "box",
    "hexa",
    "quad",
]

_DTYPE = torch.float64
PointLike = Sequence[float] | torch.Tensor


class Interpolation(StrEnum):
    """How the interior vertices of a 2D block follow from its edges."""

    #: Transfinite (Coons) interpolation, as in blockMesh. The default.
    TFI = "TFI"
    #: PICT's ``generate_grid_vertices_2D``: rows blended between the -y and +y
    #: edges and stretched per coordinate onto the -x and +x edges. Only for
    #: reproducing grids made with it.
    PICT_ROWS = "PICT_ROWS"


@dataclass(frozen=True)
class Edge:
    """An edge with its own shape and/or grading.

    Attributes
    ----------
    shape : EdgeShape
        Shape of the edge. Default is straight.
    grading : Grading or float or None
        Distribution of the edge's vertices, overriding the grading of its axis.
        Default is None (the axis grading).
    """

    shape: EdgeShape = Line()
    grading: GradingLike = None


EdgeLike: TypeAlias = Edge | EdgeShape | None


def _as_edge(edge: EdgeLike) -> Edge:
    if edge is None:
        return Edge()
    if isinstance(edge, Edge):
        return edge
    if isinstance(edge, EdgeShape):
        return Edge(edge)
    raise TypeError(f"Not an edge: {edge!r}.")


@dataclass(frozen=True)
class QuadEdges:
    """The four edges of a 2D block, named by the face they form.

    ``y_minus``/``y_plus`` run along x (from ``-x`` to ``+x``), ``x_minus``/
    ``x_plus`` along y (from ``-y`` to ``+y``); shapes and points are given in
    that direction.
    """

    x_minus: EdgeLike = None
    x_plus: EdgeLike = None
    y_minus: EdgeLike = None
    y_plus: EdgeLike = None

    def by_axis(self) -> list[list[Edge]]:
        """``[axis][e]`` in the order of :mod:`phipict.meshing.tfi`."""
        return [
            [_as_edge(self.y_minus), _as_edge(self.y_plus)],
            [_as_edge(self.x_minus), _as_edge(self.x_plus)],
        ]


@dataclass(frozen=True)
class CornerEdge:
    """An edge of a 3D block between two of its corners.

    Attributes
    ----------
    start, end : int
        Corner indices in the solver order (``i + 2 j + 4 k`` for the corner at
        ``(x, y, z) = (i, j, k)``); the corners must differ along one axis.
    edge : Edge or EdgeShape
        The edge, given from ``start`` to ``end``.
    """

    start: int
    end: int
    edge: Edge | EdgeShape


def _corners(corners: Sequence[PointLike] | torch.Tensor, n: int) -> torch.Tensor:
    c = torch.stack([torch.as_tensor(p, dtype=_DTYPE).flatten() for p in corners])
    if c.shape[0] != 2**n or c.shape[1] != n:
        raise ValueError(f"A {n}D block needs {2**n} corners with {n} coordinates.")
    return c


def _per_axis(value: Sequence[GradingLike] | GradingLike, n: int) -> list[GradingLike]:
    if isinstance(value, Sequence) and not isinstance(value, str):
        if len(value) != n:
            raise ValueError(f"Need one grading per axis ({n}), got {len(value)}.")
        return list(value)
    return [value] * n


def _curve_length(shape: EdgeShape, start: torch.Tensor, end: torch.Tensor) -> float:
    if isinstance(shape, Points):
        pts = shape.points
    elif isinstance(shape, Line):
        return float(torch.linalg.vector_norm(end - start))
    else:
        pts = shape.sample(start, end, torch.linspace(0.0, 1.0, 257, dtype=_DTYPE))
    return float(torch.linalg.vector_norm(pts[1:] - pts[:-1], dim=-1).sum())


def _edge_cells(edge: Edge) -> int | None:
    if isinstance(edge.shape, Points):
        return edge.shape.cells
    if isinstance(edge.grading, Explicit):
        return edge.grading.cells
    return None


def resolve_cells(
    cells: Sequence[int | None], gradings: list[GradingLike], edges: list[list[Edge]]
) -> list[int]:
    """Cells per axis, taking fixed counts of points and explicit gradings."""
    out = []
    for a, n in enumerate(cells):
        fixed = {c for e in edges[a] if (c := _edge_cells(e)) is not None}
        g = gradings[a]
        if isinstance(g, Explicit):
            fixed.add(g.cells)
        if n is not None:
            fixed.add(int(n))
        if len(fixed) > 1:
            raise ValueError(
                f"Conflicting cell counts {sorted(fixed)} along axis {'xyz'[a]}."
            )
        if not fixed:
            raise ValueError(f"No cell count given along axis {'xyz'[a]}.")
        out.append(fixed.pop())
    return out


def discretize_edges(
    corners: torch.Tensor,
    cells: list[int],
    gradings: list[GradingLike],
    edges: list[list[Edge]],
) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]]:
    """Points and parameter weights of every edge, ``[axis][e]``."""
    n = len(cells)
    points: list[list[torch.Tensor]] = []
    weights: list[list[torch.Tensor]] = []
    for a in range(n):
        others = [b for b in range(n) if b != a]
        pa, wa = [], []
        for e, edge in enumerate(edges[a]):
            lo = 0
            for k, b in enumerate(others):
                lo |= ((e >> k) & 1) << b
            start, end = corners[lo], corners[lo | (1 << a)]
            if isinstance(edge.shape, Points):
                pts = edge.shape.sample(start, end, torch.zeros(cells[a] + 1))
                w = edge.shape.arc_weights()
            else:
                g: Grading = as_grading(
                    edge.grading if edge.grading is not None else gradings[a]
                )
                w = g.weights(cells[a], _curve_length(edge.shape, start, end))
                pts = edge.shape.sample(start, end, w)
            pa.append(pts)
            wa.append(w)
        points.append(pa)
        weights.append(wa)
    return points, weights


def _pict_rows(
    points: list[list[torch.Tensor]], row_weights: torch.Tensor
) -> torch.Tensor:
    """PICT's ``interpolate_vertices_from_borders_2D``, vectorised."""
    (b, t), (lft, r) = points
    ny1, nx1 = len(lft), len(b)
    v = row_weights.view(ny1, 1, 1)
    rows = b[None] * (1 - v) + t[None] * v  # [ny+1, nx+1, 2]
    x_start, x_end = rows[:, :1], rows[:, -1:]
    x_size = x_end - x_start
    target = (r - lft)[:, None]
    frac = torch.linspace(0.0, 1.0, nx1, dtype=_DTYPE).view(1, nx1, 1)
    close = torch.isclose(x_size, torch.zeros_like(x_size)).any(-1, keepdim=True)
    additive = rows - x_start + (target - x_size) * frac + lft[:, None]
    scaled = (rows - x_start) * (target / torch.where(x_size == 0, 1.0, x_size)) + lft[
        :, None
    ]
    return torch.where(close, additive, scaled).permute(2, 0, 1)


def build_block(
    corners: torch.Tensor,
    cells: Sequence[int | None],
    grading: Sequence[GradingLike],
    edges: list[list[Edge]],
    patches: FacePatches | None = None,
    name: str = "block",
    interpolation: Interpolation = Interpolation.TFI,
) -> MeshBlock:
    """Discretise a block given by corners and per-edge descriptions."""
    n = corners.shape[1]
    gradings = list(grading)
    counts = resolve_cells(cells, gradings, edges)
    points, weights = discretize_edges(corners, counts, gradings, edges)
    if n == 2:
        if interpolation == Interpolation.PICT_ROWS:
            row_w = as_grading(gradings[1]).weights(counts[1])
            coords = _pict_rows(points, row_w)
        else:
            coords = tfi_2d(points, weights)
    else:
        if interpolation != Interpolation.TFI:
            raise ValueError("3D blocks only support TFI.")
        coords = tfi_3d(points, weights)
    _pin_edges(coords, points)
    return MeshBlock(coords, name, patches or FacePatches())


def _pin_edges(coords: torch.Tensor, points: list[list[torch.Tensor]]) -> None:
    """Write the edge vertices into the block exactly (no interpolation round-off)."""
    n = coords.shape[0]
    for a in range(n):
        others = [b for b in range(n) if b != a]
        for e, pts in enumerate(points[a]):
            idx: list[int | slice] = [slice(None)] * (n + 1)
            for k, b in enumerate(others):
                idx[n - b] = -1 if (e >> k) & 1 else 0
            coords[tuple(idx)] = pts.T.to(coords.dtype)


def box(
    lower: PointLike,
    upper: PointLike,
    cells: Sequence[int],
    grading: Sequence[GradingLike] | GradingLike = None,
    *,
    patches: FacePatches | None = None,
    name: str = "box",
) -> MeshBlock:
    """An axis-aligned 2D or 3D block.

    Parameters
    ----------
    lower, upper : sequence of float
        Opposite corners; the dimension follows from their length.
    cells : sequence of int
        Number of cells per axis.
    grading : sequence or Grading or float or None, optional
        Grading per axis, or one for all. Default is uniform.
    patches : FacePatches or None, optional
        Patches of the faces. Default is none.
    name : str, optional
        Name of the block. Default is ``"box"``.

    Returns
    -------
    MeshBlock
        The block.
    """
    lo = torch.as_tensor(lower, dtype=_DTYPE).flatten()
    hi = torch.as_tensor(upper, dtype=_DTYPE).flatten()
    n = len(lo)
    if len(hi) != n or len(cells) != n:
        raise ValueError("lower, upper and cells need one entry per axis.")
    grads = _per_axis(grading, n)
    axes = [
        lo[a]
        + (hi[a] - lo[a])
        * as_grading(grads[a]).weights(int(cells[a]), float(hi[a] - lo[a]))
        for a in range(n)
    ]
    mesh = torch.meshgrid(*reversed(axes), indexing="ij")  # (z,) y, x
    coords = torch.stack(list(reversed(mesh)))
    return MeshBlock(coords, name, patches or FacePatches())


def quad(
    corners: Sequence[PointLike] | torch.Tensor,
    cells: Sequence[int | None],
    grading: Sequence[GradingLike] | GradingLike = None,
    edges: QuadEdges | None = None,
    *,
    patches: FacePatches | None = None,
    name: str = "quad",
    interpolation: Interpolation = Interpolation.TFI,
) -> MeshBlock:
    """A 2D block from four corners and optional curved edges.

    Parameters
    ----------
    corners : sequence of point
        ``(-x-y, +x-y, -x+y, +x+y)``.
    cells : sequence of int or None
        Cells along x and y; None where a :class:`~phipict.meshing.Points` edge or
        an explicit grading fixes it.
    grading : sequence or Grading or float or None, optional
        Grading per axis, or one for both. Default is uniform.
    edges : QuadEdges or None, optional
        Shapes and gradings of individual edges. Default is straight.
    patches : FacePatches or None, optional
        Patches of the faces. Default is none.
    name : str, optional
        Name of the block. Default is ``"quad"``.
    interpolation : Interpolation, optional
        Interpolation of the interior. Default is transfinite.

    Returns
    -------
    MeshBlock
        The block.
    """
    c = _corners(corners, 2)
    e = (edges or QuadEdges()).by_axis()
    return build_block(c, cells, _per_axis(grading, 2), e, patches, name, interpolation)


def hexa(
    corners: Sequence[PointLike] | torch.Tensor,
    cells: Sequence[int | None],
    grading: Sequence[GradingLike] | GradingLike = None,
    edges: Sequence[CornerEdge] = (),
    *,
    patches: FacePatches | None = None,
    name: str = "hex",
) -> MeshBlock:
    """A 3D block from eight corners and optional curved edges.

    Parameters
    ----------
    corners : sequence of point
        Corners in solver order, index ``i + 2 j + 4 k`` for ``(x, y, z) = (i, j,
        k)``.
    cells : sequence of int or None
        Cells along x, y and z.
    grading : sequence or Grading or float or None, optional
        Grading per axis, or one for all. Default is uniform.
    edges : sequence of CornerEdge, optional
        Shapes and gradings of individual edges. Default is straight.
    patches : FacePatches or None, optional
        Patches of the faces. Default is none.
    name : str, optional
        Name of the block. Default is ``"hex"``.

    Returns
    -------
    MeshBlock
        The block.
    """
    c = _corners(corners, 3)
    return build_block(c, cells, _per_axis(grading, 3), hex_edges(edges), patches, name)


def hex_edges(edges: Sequence[CornerEdge]) -> list[list[Edge]]:
    """``[axis][e]`` edges of a 3D block from corner-indexed edges."""
    out = [[Edge() for _ in range(4)] for _ in range(3)]
    for ce in edges:
        diff = ce.start ^ ce.end
        if diff not in (1, 2, 4):
            raise ValueError(f"Corners {ce.start} and {ce.end} do not share an edge.")
        a = diff.bit_length() - 1
        lo = min(ce.start, ce.end)
        others = [b for b in range(3) if b != a]
        e = sum(((lo >> b) & 1) << k for k, b in enumerate(others))
        edge = _as_edge(ce.edge)
        if ce.start > ce.end:
            edge = Edge(
                edge.shape.reversed(),
                None if edge.grading is None else as_grading(edge.grading).reversed(),
            )
        out[a][e] = edge
    return out


def annulus_radial_weights(
    r_inner: float, r_outer: float, angle: float, angular_cells: int
) -> Explicit:
    """Radial vertex distribution of PICT's ``make_torus_2D``.

    The radial cells grow linearly with the radius so that they stay roughly
    square; their number follows from that.

    Parameters
    ----------
    r_inner, r_outer : float
        Radii.
    angle : float
        Angular extent in degrees.
    angular_cells : int
        Cells along the angle.

    Returns
    -------
    Explicit
        The radial grading, which also fixes the number of radial cells.
    """
    x = angular_cells + 1
    r = r_outer - r_inner
    width_scale = 2 * math.pi / x * (abs(angle) / 360)
    sizes = []
    d = r_inner
    while d < r_outer:
        width = d * width_scale
        sizes.append(width)
        d += width
    scale = (d - r_inner) / r
    total = 0.0
    weights = [0.0]
    for s in sizes:
        total += s / scale
        weights.append(total / r)
    weights[-1] = 1.0
    return Explicit(weights)


def annulus(
    center: PointLike,
    r_inner: float,
    r_outer: float,
    start_angle: float,
    angle: float,
    cells: tuple[int, int | None],
    radial_grading: GradingLike = None,
    *,
    patches: FacePatches | None = None,
    name: str = "annulus",
) -> MeshBlock:
    """A 2D ring segment; x runs along the angle, y outwards along the radius.

    Parameters
    ----------
    center : point
        Centre of the ring.
    r_inner, r_outer : float
        Radii.
    start_angle : float
        Angle of the ``-x`` face in degrees, 0 is the x axis.
    angle : float
        Angular extent in degrees, counter-clockwise if positive. A negative
        (clockwise) extent gives a right-handed block; for a positive one flip an
        axis afterwards.
    cells : tuple of int and int or None
        Cells along the angle and the radius. With None radially, the cells grow
        with the radius to stay roughly square (PICT's ``make_torus_2D``).
    radial_grading : Grading or float or None, optional
        Radial grading if the radial cells are given. Default is uniform.
    patches : FacePatches or None, optional
        Patches of the faces. Default is none.
    name : str, optional
        Name of the block. Default is ``"annulus"``.

    Returns
    -------
    MeshBlock
        The block.
    """
    if not 0 < r_inner < r_outer:
        raise ValueError("Radii must satisfy 0 < r_inner < r_outer.")
    n_ang, n_rad = cells
    c = torch.as_tensor(center, dtype=_DTYPE).flatten()
    if n_rad is None:
        radial: Grading = annulus_radial_weights(r_inner, r_outer, angle, n_ang)
    else:
        radial = as_grading(radial_grading)
    start = math.radians(start_angle % 360)
    step = math.radians(angle / n_ang)
    phi = torch.tensor([start + step * i for i in range(n_ang + 1)], dtype=_DTYPE)
    ring = torch.stack([torch.cos(phi), torch.sin(phi)], -1)
    inner, outer = c + r_inner * ring, c + r_outer * ring
    corners = torch.stack([inner[0], inner[-1], outer[0], outer[-1]])
    edges = QuadEdges(y_minus=Points(inner), y_plus=Points(outer))
    return build_block(
        corners, (n_ang, n_rad), [None, radial], edges.by_axis(), patches, name
    )
