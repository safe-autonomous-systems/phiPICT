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

"""Cell distributions along a block edge.

A grading maps a number of cells to the normalised vertex positions along an
edge, ``n + 1`` strictly increasing weights from 0 to 1::

    pm.Simple(4.0).weights(8)          # last cell 4x the first (OpenFOAM simpleGrading)
    pm.Symmetric(10.0).weights(32)     # fine at both ends, centre cells 10x larger
    pm.FirstCell(1e-3).weights(64, length=2.0)   # first cell height 1e-3

Wherever a grading is accepted, a bare number means :class:`Simple` with that
ratio (the blockMesh idiom) and None means "not given": :class:`Uniform`, or the
grading of a connected block where a :class:`~phipict.meshing.BlockMesh` can infer it.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

import torch

__all__ = [
    "ChebyshevBlend",
    "Cluster",
    "Cosine",
    "Explicit",
    "FirstCell",
    "Geometric",
    "Grading",
    "GradingLike",
    "MultiGrading",
    "Segment",
    "Simple",
    "Symmetric",
    "Tanh",
    "Uniform",
    "as_grading",
    "cells_for_size",
]

_DTYPE = torch.float64


class Cluster(StrEnum):
    """Where a grading puts its smallest cells."""

    START = "START"
    END = "END"
    BOTH = "BOTH"


class Grading(ABC):
    """Distribution of the vertices along an edge."""

    @abstractmethod
    def weights(self, cells: int, length: float = 1.0) -> torch.Tensor:
        """Normalised vertex positions along the edge.

        Parameters
        ----------
        cells : int
            Number of cells, at least 1.
        length : float, optional
            Length of the edge. Only gradings given in absolute sizes use it.
            Default is 1.

        Returns
        -------
        torch.Tensor
            ``cells + 1`` strictly increasing float64 weights from 0 to 1.
        """

    def reversed(self) -> Grading:
        """The same grading seen from the other end of the edge."""
        return _Reversed(self)


GradingLike: TypeAlias = Grading | float | int | None


def as_grading(grading: GradingLike) -> Grading:
    """Normalise a grading shorthand.

    Parameters
    ----------
    grading : Grading or float or None
        A grading, a :class:`Simple` expansion ratio, or None for :class:`Uniform`.

    Returns
    -------
    Grading
        The grading.

    Raises
    ------
    TypeError
        If ``grading`` is none of the above.
    """
    if grading is None:
        return Uniform()
    if isinstance(grading, Grading):
        return grading
    if isinstance(grading, int | float) and not isinstance(grading, bool):
        return Simple(float(grading))
    raise TypeError(f"Not a grading: {grading!r}.")


def cells_for_size(length: float, size: float) -> int:
    """Number of uniform cells that comes closest to a target cell size.

    Parameters
    ----------
    length : float
        Length of the edge.
    size : float
        Target cell size.

    Returns
    -------
    int
        Number of cells, at least 1.
    """
    if size <= 0:
        raise ValueError(f"Cell size must be positive, got {size}.")
    return max(1, round(abs(length) / size))


def _check_cells(cells: int) -> None:
    if cells < 1:
        raise ValueError(f"A grading needs at least one cell, got {cells}.")


def _from_sizes(sizes: torch.Tensor) -> torch.Tensor:
    weights = torch.zeros(len(sizes) + 1, dtype=_DTYPE)
    weights[1:] = torch.cumsum(sizes, 0)
    weights /= weights[-1].clone()
    weights[-1] = 1.0
    return weights


def _geometric_sizes(cells: int, ratio: float) -> torch.Tensor:
    """``cells`` sizes growing geometrically from first to last by ``ratio``."""
    if cells == 1 or abs(ratio - 1.0) < 1e-12:
        return torch.ones(cells, dtype=_DTYPE)
    q = ratio ** (1.0 / (cells - 1))
    return q ** torch.arange(cells, dtype=_DTYPE)


def _cluster(fine_at_start: torch.Tensor, cluster: Cluster) -> torch.Tensor:
    """Place the refinement of weights that are fine at the start.

    ``fine_at_start`` are weights for :attr:`Cluster.START`; END mirrors them. BOTH
    is handled by the callers since it needs a symmetric law.
    """
    if cluster == Cluster.START:
        return fine_at_start
    if cluster == Cluster.END:
        return _mirror(fine_at_start)
    raise ValueError(f"Unhandled cluster {cluster!r}.")


def _mirror(weights: torch.Tensor) -> torch.Tensor:
    return (1.0 - weights).flip(0)


@dataclass(frozen=True)
class _Reversed(Grading):
    grading: Grading

    def weights(self, cells: int, length: float = 1.0) -> torch.Tensor:
        """See :meth:`Grading.weights`."""
        return _mirror(self.grading.weights(cells, length))

    def reversed(self) -> Grading:
        """See :meth:`Grading.reversed`."""
        return self.grading


@dataclass(frozen=True)
class Uniform(Grading):
    """Equal cells."""

    def weights(self, cells: int, length: float = 1.0) -> torch.Tensor:
        """See :meth:`Grading.weights`."""
        _check_cells(cells)
        return torch.linspace(0.0, 1.0, cells + 1, dtype=_DTYPE)

    def reversed(self) -> Grading:
        """See :meth:`Grading.reversed`."""
        return self


@dataclass(frozen=True)
class Simple(Grading):
    """Geometric growth with OpenFOAM's ``simpleGrading`` expansion ratio.

    Attributes
    ----------
    ratio : float
        Size of the last cell over the size of the first. Above 1 refines the
        start, below 1 the end.
    """

    ratio: float

    def __post_init__(self) -> None:
        if not self.ratio > 0:
            raise ValueError(f"Expansion ratio must be positive, got {self.ratio}.")

    def weights(self, cells: int, length: float = 1.0) -> torch.Tensor:
        """See :meth:`Grading.weights`."""
        _check_cells(cells)
        return _from_sizes(_geometric_sizes(cells, self.ratio))

    def reversed(self) -> Grading:
        """See :meth:`Grading.reversed`."""
        return Simple(1.0 / self.ratio)


@dataclass(frozen=True)
class Geometric(Grading):
    """Cells that grow by a constant factor from one cell to the next.

    Sizes grow by ``ratio`` per cell moving away from ``cluster`` (for BOTH, away
    from both ends towards the centre), so the smallest cells are at ``cluster``
    for a ratio above 1. This is PICT's ``make_weights_exp``, including its split
    of the cells for BOTH: ``Geometric(1.05, Cluster.BOTH)`` equals
    ``make_weights_exp(n, 1.05, "BOTH")``.

    Attributes
    ----------
    ratio : float
        Size ratio of neighbouring cells.
    cluster : Cluster
        Where the growth starts. Default is the start.
    """

    ratio: float
    cluster: Cluster = Cluster.START

    def __post_init__(self) -> None:
        if not self.ratio > 0:
            raise ValueError(f"Growth ratio must be positive, got {self.ratio}.")

    def weights(self, cells: int, length: float = 1.0) -> torch.Tensor:
        """See :meth:`Grading.weights`."""
        _check_cells(cells)
        exponents = list(range(cells))
        if self.cluster == Cluster.END:
            exponents.reverse()
        elif self.cluster == Cluster.BOTH:
            exponents = exponents[: cells // 2] + exponents[::-1][cells // 2 :]
        sizes = torch.tensor([self.ratio**e for e in exponents], dtype=_DTYPE)
        return _from_sizes(sizes)

    def reversed(self) -> Grading:
        """See :meth:`Grading.reversed`."""
        if self.cluster == Cluster.BOTH:
            return super().reversed()
        return Geometric(self.ratio, _reverse_cluster(self.cluster))


@dataclass(frozen=True)
class Segment:
    """One segment of a :class:`MultiGrading`.

    Attributes
    ----------
    length : float
        Relative length of the segment.
    cells : float
        Relative number of cells in the segment.
    ratio : float
        Expansion ratio within the segment, as for :class:`Simple`.
    """

    length: float
    cells: float
    ratio: float


@dataclass(frozen=True)
class MultiGrading(Grading):
    """Piecewise geometric grading, OpenFOAM's multi-grading.

    ``MultiGrading([Segment(0.2, 0.3, 4), Segment(0.6, 0.4, 1), Segment(0.2, 0.3,
    0.25)])`` is blockMesh's ``((0.2 0.3 4) (0.6 0.4 1) (0.2 0.3 0.25))``. Relative
    lengths and cell counts are normalised, and the cell counts are rounded like
    blockMesh does, with the last segment taking the remainder.

    Attributes
    ----------
    segments : tuple of Segment
        The segments from start to end.
    """

    segments: tuple[Segment, ...]

    def __init__(self, segments: Sequence[Segment | Sequence[float]]) -> None:
        segs = tuple(s if isinstance(s, Segment) else Segment(*s) for s in segments)
        if not segs:
            raise ValueError("A multi-grading needs at least one segment.")
        for s in segs:
            if s.length <= 0 or s.cells <= 0 or s.ratio <= 0:
                raise ValueError(f"Segment values must be positive: {s}.")
        object.__setattr__(self, "segments", segs)

    def _cells(self, cells: int) -> list[int]:
        total = sum(s.cells for s in self.segments)
        counts = [round(cells * s.cells / total) for s in self.segments[:-1]]
        counts.append(cells - sum(counts))
        if min(counts) < 1:
            raise ValueError(
                f"{cells} cells are too few for {len(self.segments)} grading segments."
            )
        return counts

    def weights(self, cells: int, length: float = 1.0) -> torch.Tensor:
        """See :meth:`Grading.weights`."""
        _check_cells(cells)
        total = sum(s.length for s in self.segments)
        sizes = []
        for seg, n in zip(self.segments, self._cells(cells), strict=True):
            s = _geometric_sizes(n, seg.ratio)
            sizes.append(s / s.sum() * (seg.length / total))
        return _from_sizes(torch.cat(sizes))

    def reversed(self) -> Grading:
        """See :meth:`Grading.reversed`."""
        # mirror the weights: reversing the segments would round the cell counts
        # of the segments differently
        return _Reversed(self)


@dataclass(frozen=True)
class Symmetric(Grading):
    """Geometric grading that is fine at both ends, e.g. between two walls.

    Attributes
    ----------
    ratio : float
        Size of the centre cells over the size of the end cells. Below 1 refines
        the centre instead.
    """

    ratio: float

    def __post_init__(self) -> None:
        if not self.ratio > 0:
            raise ValueError(f"Expansion ratio must be positive, got {self.ratio}.")

    def weights(self, cells: int, length: float = 1.0) -> torch.Tensor:
        """See :meth:`Grading.weights`."""
        _check_cells(cells)
        half = cells // 2
        if half == 0:
            return Uniform().weights(cells)
        left = _geometric_sizes(half, self.ratio)
        right = _geometric_sizes(cells - half, self.ratio).flip(0)
        # both halves span half the edge, even for an odd cell count
        sizes = torch.cat([left / left.sum(), right / right.sum()])
        return _from_sizes(sizes)

    def reversed(self) -> Grading:
        """See :meth:`Grading.reversed`."""
        return self


@dataclass(frozen=True)
class Tanh(Grading):
    """Hyperbolic tangent stretching.

    Attributes
    ----------
    beta : float
        Stretching strength, 0 is uniform. Larger values cluster harder.
    cluster : Cluster
        Where the smallest cells are. Default is both ends.
    """

    beta: float
    cluster: Cluster = Cluster.BOTH

    def weights(self, cells: int, length: float = 1.0) -> torch.Tensor:
        """See :meth:`Grading.weights`."""
        _check_cells(cells)
        if abs(self.beta) < 1e-12:
            return Uniform().weights(cells)
        t = torch.linspace(0.0, 1.0, cells + 1, dtype=_DTYPE)
        b = self.beta
        if self.cluster == Cluster.BOTH:
            w = 0.5 * (1.0 + torch.tanh(b * (2.0 * t - 1.0)) / math.tanh(b))
        else:
            w = _cluster(1.0 + torch.tanh(b * (t - 1.0)) / math.tanh(b), self.cluster)
        return _clean(w)

    def reversed(self) -> Grading:
        """See :meth:`Grading.reversed`."""
        return Tanh(self.beta, _reverse_cluster(self.cluster))


@dataclass(frozen=True)
class Cosine(Grading):
    """Cosine (Chebyshev-like) spacing.

    Attributes
    ----------
    cluster : Cluster
        Where the smallest cells are. Default is both ends.
    """

    cluster: Cluster = Cluster.BOTH

    def weights(self, cells: int, length: float = 1.0) -> torch.Tensor:
        """See :meth:`Grading.weights`."""
        _check_cells(cells)
        t = torch.linspace(0.0, 1.0, cells + 1, dtype=_DTYPE)
        if self.cluster == Cluster.BOTH:
            w = 0.5 - 0.5 * torch.cos(math.pi * t)
        else:
            w = _cluster(1.0 - torch.cos(0.5 * math.pi * t), self.cluster)
        return _clean(w)

    def reversed(self) -> Grading:
        """See :meth:`Grading.reversed`."""
        return Cosine(_reverse_cluster(self.cluster))


@dataclass(frozen=True)
class ChebyshevBlend(Grading):
    """Blend of a sine (Chebyshev) mapping and uniform spacing.

    Attributes
    ----------
    gamma : float
        Share of the sine mapping in [0, 1]: 0 is uniform, 1 a pure sine mapping.
    cluster : Cluster
        Where the smallest cells are. Default is both ends.
    """

    gamma: float
    cluster: Cluster = Cluster.BOTH

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}.")

    def weights(self, cells: int, length: float = 1.0) -> torch.Tensor:
        """See :meth:`Grading.weights`."""
        _check_cells(cells)
        g = self.gamma
        if self.cluster == Cluster.BOTH:
            xi = torch.linspace(-1.0, 1.0, cells + 1, dtype=_DTYPE)
            w = 0.5 * (g * torch.sin(0.5 * math.pi * xi) + (1 - g) * xi) + 0.5
        else:
            xi = torch.linspace(-1.0, 0.0, cells + 1, dtype=_DTYPE)
            w = _cluster(
                g * torch.sin(0.5 * math.pi * xi) + (1 - g) * xi + 1.0, self.cluster
            )
        return _clean(w)

    def reversed(self) -> Grading:
        """See :meth:`Grading.reversed`."""
        return ChebyshevBlend(self.gamma, _reverse_cluster(self.cluster))


@dataclass(frozen=True)
class FirstCell(Grading):
    """Geometric grading with a prescribed size of the smallest cell.

    The expansion ratio is solved for, so the first cell has ``size`` for the
    edge length it is used on, e.g. to hit a target ``y+`` at a wall.

    Attributes
    ----------
    size : float
        Size of the smallest cell(s), in the units of the coordinates.
    cluster : Cluster
        Where the smallest cells are. Default is the start.
    """

    size: float
    cluster: Cluster = Cluster.START

    def __post_init__(self) -> None:
        if not self.size > 0:
            raise ValueError(f"First cell size must be positive, got {self.size}.")

    def weights(self, cells: int, length: float = 1.0) -> torch.Tensor:
        """See :meth:`Grading.weights`."""
        _check_cells(cells)
        if self.cluster == Cluster.BOTH:
            half = cells // 2
            if half == 0:
                return Uniform().weights(cells)
            q_left = _solve_growth(half, self.size / (0.5 * length))
            q_right = _solve_growth(cells - half, self.size / (0.5 * length))
            left = q_left ** torch.arange(half, dtype=_DTYPE)
            right = (q_right ** torch.arange(cells - half, dtype=_DTYPE)).flip(0)
            sizes = torch.cat([left / left.sum(), right / right.sum()])
            return _from_sizes(sizes)
        q = _solve_growth(cells, self.size / length)
        w = _from_sizes(q ** torch.arange(cells, dtype=_DTYPE))
        return _cluster(w, self.cluster)

    def reversed(self) -> Grading:
        """See :meth:`Grading.reversed`."""
        return FirstCell(self.size, _reverse_cluster(self.cluster))


def _solve_growth(cells: int, fraction: float) -> float:
    """Growth factor q > 0 with ``fraction * sum(q**i for i < cells) == 1``.

    The sum grows monotonically with q, from 1 at q = 0, so a bisection finds it.
    """
    if not fraction < 1.0:
        raise ValueError(
            "The first cell must be smaller than the edge it is on "
            f"(it would span {fraction:.3g} of it)."
        )
    if cells == 1 or abs(fraction * cells - 1.0) < 1e-12:
        return 1.0

    def total(q: float) -> float:
        if abs(q - 1.0) < 1e-14:
            return fraction * cells
        return fraction * (q**cells - 1.0) / (q - 1.0)

    lo, hi = 0.0, 2.0
    while total(hi) < 1.0:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True, eq=False)
class Explicit(Grading):
    """Given vertex weights.

    Attributes
    ----------
    values : torch.Tensor
        Strictly increasing weights from 0 to 1; their count fixes the number of
        cells.
    """

    values: torch.Tensor

    def __init__(self, values: Sequence[float] | torch.Tensor) -> None:
        w = torch.as_tensor(values, dtype=_DTYPE).flatten().clone()
        if len(w) < 2:
            raise ValueError("Explicit weights need at least two values.")
        if abs(float(w[0])) > 1e-12 or abs(float(w[-1]) - 1.0) > 1e-12:
            raise ValueError("Explicit weights must start at 0 and end at 1.")
        if not bool((w[1:] > w[:-1]).all()):
            raise ValueError("Explicit weights must be strictly increasing.")
        object.__setattr__(self, "values", w)

    @property
    def cells(self) -> int:
        """Number of cells the weights describe."""
        return len(self.values) - 1

    def weights(self, cells: int, length: float = 1.0) -> torch.Tensor:
        """See :meth:`Grading.weights`."""
        if cells != self.cells:
            raise ValueError(
                f"Explicit grading has {self.cells} cells, but {cells} are needed."
            )
        return self.values.clone()

    def reversed(self) -> Grading:
        """See :meth:`Grading.reversed`."""
        return Explicit(_mirror(self.values))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Explicit):
            return NotImplemented
        return torch.equal(self.values, other.values)

    def __hash__(self) -> int:
        return hash(tuple(self.values.tolist()))


def _reverse_cluster(cluster: Cluster) -> Cluster:
    return {
        Cluster.START: Cluster.END,
        Cluster.END: Cluster.START,
        Cluster.BOTH: Cluster.BOTH,
    }[Cluster(cluster)]


def _clean(weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(_DTYPE)
    weights[0] = 0.0
    weights[-1] = 1.0
    return weights
