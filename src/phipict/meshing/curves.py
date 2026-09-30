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

"""Shapes of block edges.

As in blockMesh, a shape describes an edge *between* two given end points, so the
end points always coincide with the block corners::

    pm.Arc(center=(0, 0))            # circular arc around the origin
    pm.Arc(through=(0.7, 0.7))       # circular arc through a point
    pm.Polyline([(1, 0.2), (2, 0.1)])   # straight segments via interior points
    pm.Spline([(1, 0.2), (2, 0.1)])     # Catmull-Rom spline via interior points
    pm.Parametric(lambda t: ...)        # any curve c(t), t in [0, 1]
    pm.Points(surface)                  # given vertex positions, used as they are

The vertices along an edge are placed by arc length, so a grading distributes
them along the curve as it would along a straight edge. :class:`Points` is the
exception: it fixes the vertices themselves, e.g. those of a measured or
precomputed surface, and with them the number of cells of the edge.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

__all__ = [
    "Arc",
    "BSpline",
    "EdgeShape",
    "Line",
    "Points",
    "Parametric",
    "Polyline",
    "Sampled",
    "Spline",
]

_DTYPE = torch.float64
# samples per edge used to invert the arc length of curved shapes
_ARC_SAMPLES = 4096

PointLike = Sequence[float] | torch.Tensor


def _point(p: PointLike) -> torch.Tensor:
    return torch.as_tensor(p, dtype=_DTYPE).flatten()


def _points(ps: Sequence[PointLike] | torch.Tensor) -> torch.Tensor:
    if isinstance(ps, torch.Tensor):
        pts = ps.to(_DTYPE)
    else:
        pts = torch.stack([_point(p) for p in ps]) if len(ps) else torch.zeros(0, 0)
    return pts.reshape(len(pts), -1).to(_DTYPE)


class EdgeShape(ABC):
    """Shape of an edge between two end points."""

    def sample(
        self, start: torch.Tensor, end: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Points along the edge at normalised arc lengths.

        Parameters
        ----------
        start, end : torch.Tensor
            End points of the edge, shape ``[dims]``.
        weights : torch.Tensor
            Normalised arc lengths from 0 to 1, shape ``[n]``.

        Returns
        -------
        torch.Tensor
            Points of shape ``[n, dims]``; the first and last are exactly
            ``start`` and ``end`` when the weights span [0, 1].
        """
        start, end = start.to(_DTYPE), end.to(_DTYPE)
        t = self._arc_parameter(start, end, weights.to(_DTYPE))
        pts = self.evaluate(start, end, t)
        # pin the end points against round-off
        pts[weights <= 0.0] = start
        pts[weights >= 1.0] = end
        return pts

    @abstractmethod
    def evaluate(
        self, start: torch.Tensor, end: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Points at curve parameters ``t`` in [0, 1] (not arc length)."""

    def _arc_parameter(
        self, start: torch.Tensor, end: torch.Tensor, s: torch.Tensor
    ) -> torch.Tensor:
        """Curve parameters at normalised arc lengths ``s``."""
        t_dense = torch.linspace(0.0, 1.0, _ARC_SAMPLES + 1, dtype=_DTYPE)
        pts = self.evaluate(start, end, t_dense)
        seg = torch.linalg.vector_norm(pts[1:] - pts[:-1], dim=-1)
        arc = torch.zeros(_ARC_SAMPLES + 1, dtype=_DTYPE)
        arc[1:] = torch.cumsum(seg, 0)
        if float(arc[-1]) <= 0.0:
            raise ValueError("Edge has zero length.")
        arc /= arc[-1].clone()
        return _interp(s, arc, t_dense)

    def reversed(self) -> EdgeShape:
        """The same edge traversed from its end to its start."""
        return self


def _interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Piecewise linear interpolation, like ``numpy.interp``."""
    x = x.clamp(float(xp[0]), float(xp[-1]))
    idx = torch.searchsorted(xp, x, right=True).clamp(1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    denom = torch.where(x1 > x0, x1 - x0, torch.ones_like(x1))
    return f0 + (f1 - f0) * (x - x0) / denom


@dataclass(frozen=True)
class Line(EdgeShape):
    """Straight edge, the default."""

    def evaluate(
        self, start: torch.Tensor, end: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """See :meth:`EdgeShape.evaluate`."""
        return start + t[:, None] * (end - start)

    def _arc_parameter(
        self, start: torch.Tensor, end: torch.Tensor, s: torch.Tensor
    ) -> torch.Tensor:
        return s


def _to3(p: torch.Tensor) -> torch.Tensor:
    if len(p) == 3:
        return p
    if len(p) == 2:
        return torch.cat([p, p.new_zeros(1)])
    raise ValueError(f"Points need 2 or 3 coordinates, got {len(p)}.")


@dataclass(frozen=True, eq=False)
class Arc(EdgeShape):
    """Circular arc, given by a point on it or by its centre.

    With ``through``, the arc passes through that point (blockMesh ``arc``). With
    ``center`` (blockMesh ``arc ... origin``), the arc runs around the centre the
    short way; if the end points have different distances to it, the radius
    changes linearly with the angle.

    Attributes
    ----------
    through : torch.Tensor or None
        A point on the arc between its end points.
    center : torch.Tensor or None
        Centre of the arc.
    """

    through: torch.Tensor | None = None
    center: torch.Tensor | None = None

    def __init__(
        self, *, through: PointLike | None = None, center: PointLike | None = None
    ) -> None:
        if (through is None) == (center is None):
            raise ValueError("An Arc needs exactly one of `through` and `center`.")
        object.__setattr__(
            self, "through", None if through is None else _point(through)
        )
        object.__setattr__(self, "center", None if center is None else _point(center))

    def _frame(
        self, start: torch.Tensor, end: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]:
        """Centre, in-plane basis, start/end radius and swept angle."""
        s, e = _to3(start), _to3(end)
        if self.through is not None:
            m = _to3(self.through)
            c = _circumcenter(s, m, e)
            e1 = (s - c) / torch.linalg.vector_norm(s - c)
            n = torch.linalg.cross(s - c, m - c)
            n = n / torch.linalg.vector_norm(n)
            e2 = torch.linalg.cross(n, e1)
            phi_m = _angle(m - c, e1, e2)
            theta = _angle(e - c, e1, e2)
            if theta < phi_m:  # the arc runs the other way round
                e2 = -e2
                theta = 2.0 * math.pi - theta
        else:
            assert self.center is not None
            c = _to3(self.center)
            e1 = (s - c) / torch.linalg.vector_norm(s - c)
            n = torch.linalg.cross(s - c, e - c)
            if float(torch.linalg.vector_norm(n)) < 1e-12 * float(
                torch.linalg.vector_norm(s - c) * torch.linalg.vector_norm(e - c)
            ):
                raise ValueError(
                    "Arc end points are opposite each other (or collinear with the "
                    "centre), so the arc is ambiguous; use Arc(through=...)."
                )
            n = n / torch.linalg.vector_norm(n)
            e2 = torch.linalg.cross(n, e1)
            theta = _angle(e - c, e1, e2)
        r0 = float(torch.linalg.vector_norm(s - c))
        r1 = float(torch.linalg.vector_norm(e - c))
        return c, e1, e2, r0, r1, theta

    def evaluate(
        self, start: torch.Tensor, end: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """See :meth:`EdgeShape.evaluate`."""
        c, e1, e2, r0, r1, theta = self._frame(start, end)
        phi = t * theta
        r = r0 + (r1 - r0) * t
        pts = c + r[:, None] * (
            torch.cos(phi)[:, None] * e1 + torch.sin(phi)[:, None] * e2
        )
        return pts[:, : len(start)]

    def _arc_parameter(
        self, start: torch.Tensor, end: torch.Tensor, s: torch.Tensor
    ) -> torch.Tensor:
        _, _, _, r0, r1, _ = self._frame(start, end)
        if abs(r1 - r0) <= 1e-12 * max(r0, r1):
            return s  # constant radius: arc length is proportional to the angle
        return super()._arc_parameter(start, end, s)


def _angle(v: torch.Tensor, e1: torch.Tensor, e2: torch.Tensor) -> float:
    a = math.atan2(float(v @ e2), float(v @ e1))
    return a if a >= 0.0 else a + 2.0 * math.pi


def _circumcenter(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    ab, ac = b - a, c - a
    n = torch.linalg.cross(ab, ac)
    nn = float(n @ n)
    if nn < 1e-24 * float(ab @ ab) * float(ac @ ac):
        raise ValueError("Arc points are collinear.")
    return a + (
        torch.linalg.cross(n, ab) * float(ac @ ac)
        + torch.linalg.cross(ac, n) * float(ab @ ab)
    ) / (2.0 * nn)


@dataclass(frozen=True, eq=False)
class Polyline(EdgeShape):
    """Straight segments through interior points (blockMesh ``polyLine``).

    Attributes
    ----------
    points : torch.Tensor
        Interior points ``[n, dims]``, without the end points.
    """

    points: torch.Tensor

    def __init__(self, points: Sequence[PointLike] | torch.Tensor) -> None:
        object.__setattr__(self, "points", _points(points))

    def _all(self, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        return torch.cat([start[None], self.points.to(_DTYPE), end[None]])

    def evaluate(
        self, start: torch.Tensor, end: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """See :meth:`EdgeShape.evaluate`."""
        pts = self._all(start, end)
        tp = torch.linspace(0.0, 1.0, len(pts), dtype=_DTYPE)
        return torch.stack(
            [_interp(t, tp, pts[:, d]) for d in range(pts.shape[1])], dim=-1
        )

    def _arc_parameter(
        self, start: torch.Tensor, end: torch.Tensor, s: torch.Tensor
    ) -> torch.Tensor:
        pts = self._all(start, end)
        seg = torch.linalg.vector_norm(pts[1:] - pts[:-1], dim=-1)
        arc = torch.zeros(len(pts), dtype=_DTYPE)
        arc[1:] = torch.cumsum(seg, 0)
        arc /= arc[-1].clone()
        tp = torch.linspace(0.0, 1.0, len(pts), dtype=_DTYPE)
        return _interp(s, arc, tp)

    def reversed(self) -> EdgeShape:
        """See :meth:`EdgeShape.reversed`."""
        return Polyline(self.points.flip(0))


@dataclass(frozen=True, eq=False)
class Sampled(Polyline):
    """A densely sampled curve, e.g. an airfoil surface, including its end points.

    The first and last points must coincide with the edge's end points (within a
    small tolerance) and are replaced by them.

    Attributes
    ----------
    points : torch.Tensor
        Points ``[n, dims]`` along the curve, end points included.
    """

    def __init__(self, points: Sequence[PointLike] | torch.Tensor) -> None:
        pts = _points(points)
        if len(pts) < 2:
            raise ValueError("A sampled curve needs at least its two end points.")
        object.__setattr__(self, "points", pts)

    def _all(self, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        pts = self.points.to(_DTYPE).clone()
        scale = float(torch.linalg.vector_norm(end - start)) or 1.0
        for i, p in ((0, start), (-1, end)):
            if float(torch.linalg.vector_norm(pts[i] - p)) > 1e-6 * scale:
                raise ValueError(
                    f"Sampled curve end {pts[i].tolist()} does not match the edge "
                    f"end {p.tolist()}."
                )
            pts[i] = p
        return pts

    def reversed(self) -> EdgeShape:
        """See :meth:`EdgeShape.reversed`."""
        return Sampled(self.points.flip(0))


def _catmull_rom(pts: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Uniform Catmull-Rom spline through ``pts`` at parameters ``t`` in [0, 1]."""
    n_seg = len(pts) - 1
    ext = torch.cat([2 * pts[:1] - pts[1:2], pts, 2 * pts[-1:] - pts[-2:-1]])
    x = t.clamp(0.0, 1.0) * n_seg
    i = x.floor().long().clamp(max=n_seg - 1)
    u = (x - i)[:, None]
    p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
    return 0.5 * (
        2 * p1
        + (-p0 + p2) * u
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u**2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * u**3
    )


@dataclass(frozen=True, eq=False, init=False)
class Spline(Polyline):
    """Catmull-Rom spline through interior points (blockMesh ``spline``).

    Attributes
    ----------
    points : torch.Tensor
        Interior points ``[n, dims]``, without the end points.
    """

    def evaluate(
        self, start: torch.Tensor, end: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """See :meth:`EdgeShape.evaluate`."""
        return _catmull_rom(self._all(start, end), t)

    def _arc_parameter(
        self, start: torch.Tensor, end: torch.Tensor, s: torch.Tensor
    ) -> torch.Tensor:
        return EdgeShape._arc_parameter(self, start, end, s)

    def reversed(self) -> EdgeShape:
        """See :meth:`EdgeShape.reversed`."""
        return Spline(self.points.flip(0))


@dataclass(frozen=True, eq=False, init=False)
class BSpline(Polyline):
    """Cubic B-spline with interior control points (blockMesh ``BSpline``).

    The curve passes through the end points but, unlike :class:`Spline`, only
    approaches the interior points.

    Attributes
    ----------
    points : torch.Tensor
        Interior control points ``[n, dims]``, without the end points.
    """

    def evaluate(
        self, start: torch.Tensor, end: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """See :meth:`EdgeShape.evaluate`."""
        ctrl = self._all(start, end)
        # clamped uniform cubic B-spline: triple end points
        ctrl = torch.cat([ctrl[:1], ctrl[:1], ctrl, ctrl[-1:], ctrl[-1:]])
        n_seg = len(ctrl) - 3
        x = t.clamp(0.0, 1.0) * n_seg
        i = x.floor().long().clamp(max=n_seg - 1)
        u = (x - i)[:, None]
        b0 = (1 - u) ** 3 / 6
        b1 = (3 * u**3 - 6 * u**2 + 4) / 6
        b2 = (-3 * u**3 + 3 * u**2 + 3 * u + 1) / 6
        b3 = u**3 / 6
        return b0 * ctrl[i] + b1 * ctrl[i + 1] + b2 * ctrl[i + 2] + b3 * ctrl[i + 3]

    def _arc_parameter(
        self, start: torch.Tensor, end: torch.Tensor, s: torch.Tensor
    ) -> torch.Tensor:
        return EdgeShape._arc_parameter(self, start, end, s)

    def reversed(self) -> EdgeShape:
        """See :meth:`EdgeShape.reversed`."""
        return BSpline(self.points.flip(0))


@dataclass(frozen=True, eq=False)
class Parametric(EdgeShape):
    """A curve given as a function of a parameter in [0, 1].

    Attributes
    ----------
    fn : Callable
        Maps parameters ``t`` of shape ``[n]`` to points ``[n, dims]``. ``fn(0)``
        and ``fn(1)`` must match the edge's end points (within a small tolerance).
    """

    fn: Callable[[torch.Tensor], torch.Tensor]

    def evaluate(
        self, start: torch.Tensor, end: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """See :meth:`EdgeShape.evaluate`."""
        pts = torch.as_tensor(self.fn(t), dtype=_DTYPE).reshape(len(t), -1)
        return pts

    def sample(
        self, start: torch.Tensor, end: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """See :meth:`EdgeShape.sample`."""
        ends = self.evaluate(start, end, torch.tensor([0.0, 1.0], dtype=_DTYPE))
        scale = float(torch.linalg.vector_norm(end - start)) or 1.0
        for p, q in ((ends[0], start), (ends[1], end)):
            if float(torch.linalg.vector_norm(p - q.to(_DTYPE))) > 1e-6 * scale:
                raise ValueError(
                    f"Parametric curve end {p.tolist()} does not match the edge end "
                    f"{q.tolist()}."
                )
        return super().sample(start, end, weights)

    def reversed(self) -> EdgeShape:
        """See :meth:`EdgeShape.reversed`."""
        fn = self.fn
        return Parametric(lambda t: fn(1.0 - t))


@dataclass(frozen=True, eq=False)
class Points(EdgeShape):
    """The vertices of an edge, given explicitly and used as they are.

    Unlike the other shapes, the points are not resampled: they *are* the edge's
    vertices, so they fix its number of cells (``len(points) - 1``) and its
    vertex distribution. A grading of the edge is ignored. Use :class:`Sampled`
    to resample a curve with a grading instead.

    Attributes
    ----------
    points : torch.Tensor
        Vertices ``[n, dims]`` from the start to the end of the edge. The end
        points must coincide with the edge's corners (within a small tolerance).
    """

    points: torch.Tensor

    def __init__(self, points: Sequence[PointLike] | torch.Tensor) -> None:
        pts = _points(points)
        if len(pts) < 2:
            raise ValueError("An edge needs at least its two end points.")
        object.__setattr__(self, "points", pts)

    @property
    def cells(self) -> int:
        """Number of cells of the edge."""
        return len(self.points) - 1

    def arc_weights(self) -> torch.Tensor:
        """Normalised arc lengths of the points."""
        seg = torch.linalg.vector_norm(self.points[1:] - self.points[:-1], dim=-1)
        w = torch.zeros(len(self.points), dtype=_DTYPE)
        w[1:] = torch.cumsum(seg, 0)
        w /= w[-1].clone()
        w[-1] = 1.0
        return w

    def check_ends(self, start: torch.Tensor, end: torch.Tensor) -> None:
        """Raise if the end points do not coincide with ``start``/``end``."""
        pts = self.points
        scale = float(torch.linalg.vector_norm(pts[-1] - pts[0])) or 1.0
        for p, q in ((pts[0], start), (pts[-1], end)):
            if float(torch.linalg.vector_norm(p - q.to(_DTYPE))) > 1e-6 * scale:
                raise ValueError(
                    f"Edge points end at {p.tolist()}, but the corner is {q.tolist()}."
                )

    def evaluate(
        self, start: torch.Tensor, end: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """See :meth:`EdgeShape.evaluate`."""
        return Polyline(self.points[1:-1]).evaluate(start, end, t)

    def sample(
        self, start: torch.Tensor, end: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """See :meth:`EdgeShape.sample`."""
        if len(weights) != len(self.points):
            raise ValueError(
                f"The edge has {self.cells} cells from its points, but "
                f"{len(weights) - 1} "
                "are needed."
            )
        self.check_ends(start, end)
        return self.points.clone()

    def reversed(self) -> EdgeShape:
        """See :meth:`EdgeShape.reversed`."""
        return Points(self.points.flip(0))
