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

"""A discretised structured block: vertex coordinates plus face patches.

The vertices are stored as the solver expects them, without the batch dimension:
``[dims, (nz + 1,) ny + 1, nx + 1]`` with the coordinate channels ordered x, y, z
and the grid dimensions z, y, x. Blocks are geometry only (float64 on the CPU);
:meth:`Mesh.get_domain <phipict.meshing.Mesh.get_domain>` casts them for the solver.

All operations return new blocks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from phipict.bc import Face

from .boundary import FacePatches, Patch
from .grading import GradingLike, as_grading

__all__ = ["MeshBlock", "grid_dim", "tangential_axes"]

_DTYPE = torch.float64
_AXES = "xyz"


def grid_dim(axis: int, ndims: int) -> int:
    """Dimension of the vertex tensor ``[dims, (z,) y, x]`` along a logical axis."""
    return ndims - axis


def tangential_axes(face: Face | int, ndims: int) -> tuple[int, ...]:
    """The in-face axes of a face in the solver's order ``(a + 1, a + 2) % dims``."""
    a = int(face) >> 1
    return tuple((a + k) % ndims for k in range(1, ndims))


def _parse_axis(axis: int | str) -> int:
    if isinstance(axis, str):
        if axis not in _AXES:
            raise ValueError(f"Unknown axis {axis!r}; use 'x', 'y' or 'z'.")
        return _AXES.index(axis)
    return int(axis)


@dataclass(eq=False)
class MeshBlock:
    """A structured block of a mesh.

    Attributes
    ----------
    coords : torch.Tensor
        Vertices ``[dims, (nz + 1,) ny + 1, nx + 1]``, float64 on the CPU.
    name : str
        Name of the block, also used for the solver block.
    patches : FacePatches
        Patch of each boundary face.
    """

    coords: torch.Tensor
    name: str = "block"
    patches: FacePatches = field(default_factory=FacePatches)

    def __post_init__(self) -> None:
        c = torch.as_tensor(self.coords)
        if c.dim() >= 3 and c.shape[0] == 1 and c.shape[1] == c.dim() - 2:
            c = c[0]  # solver layout with a batch dimension
        ndims = c.dim() - 1
        if ndims not in (2, 3) or c.shape[0] != ndims:
            raise ValueError(
                "Block vertices need the shape [dims, (nz+1,) ny+1, nx+1] with dims "
                f"2 or 3, got {list(c.shape)}."
            )
        if min(c.shape[1:]) < 2:
            raise ValueError(
                f"A block needs at least one cell per axis, got {list(c.shape)}."
            )
        self.coords = c.detach().to("cpu", _DTYPE).contiguous()
        for face, _ in self.patches.items():
            if int(face) >= 2 * ndims:
                raise ValueError(f"A {ndims}D block has no face {face.name}.")

    @classmethod
    def from_coords(
        cls,
        coords: torch.Tensor,
        name: str = "block",
        patches: FacePatches | None = None,
    ) -> MeshBlock:
        """A block from a vertex tensor, e.g. one of :mod:`phipict.grid.shapes`.

        Parameters
        ----------
        coords : torch.Tensor
            Vertices ``[(1,) dims, (nz + 1,) ny + 1, nx + 1]``.
        name : str, optional
            Name of the block. Default is ``"block"``.
        patches : FacePatches or None, optional
            Patches of the faces. Default is None (no patches).

        Returns
        -------
        MeshBlock
            The block.
        """
        return cls(coords, name, patches or FacePatches())

    # -----------------------------------------------------------------------
    # Shape
    # -----------------------------------------------------------------------

    @property
    def ndims(self) -> int:
        """Number of spatial dimensions."""
        return int(self.coords.shape[0])

    @property
    def cells(self) -> tuple[int, ...]:
        """Number of cells per axis, ``(nx, ny[, nz])``."""
        return tuple(int(n) - 1 for n in reversed(self.coords.shape[1:]))

    @property
    def n_cells(self) -> int:
        """Total number of cells."""
        return math.prod(self.cells)

    def cell_centers(self) -> torch.Tensor:
        """Cell centres, the mean of each cell's corner vertices.

        Returns
        -------
        torch.Tensor
            ``[dims, (nz,) ny, nx]``, the layout of the solver's cell fields
            without the batch dimension.
        """
        c = self.coords
        for dim in range(1, c.dim()):
            c = 0.5 * (
                c.narrow(dim, 0, c.shape[dim] - 1) + c.narrow(dim, 1, c.shape[dim] - 1)
            )
        return c

    def face_coords(self, face: Face | int) -> torch.Tensor:
        """Vertices of a face.

        Parameters
        ----------
        face : Face
            The face.

        Returns
        -------
        torch.Tensor
            ``[dims, n_t1 + 1(, n_t2 + 1)]``, indexed by the face's tangential axes
            in the solver's order (see :func:`tangential_axes`).
        """
        face = Face(face)
        d = self.ndims
        if face.axis >= d:
            raise ValueError(f"A {d}D block has no face {face.name}.")
        idx = 0 if int(face) % 2 == 0 else -1
        f = self.coords.select(grid_dim(face.axis, d), idx)
        # remaining grid dims are the other axes in descending axis order
        remaining = [a for a in reversed(range(d)) if a != face.axis]
        order = [remaining.index(a) + 1 for a in tangential_axes(face, d)]
        return f.permute(0, *order)

    def __repr__(self) -> str:
        cells = "x".join(str(n) for n in self.cells)
        patches = ", ".join(f"{f.name}={p.name}" for f, p in self.patches.items())
        return f"MeshBlock({self.name!r}, cells={cells}, patches=[{patches}])"

    # -----------------------------------------------------------------------
    # Transforms
    # -----------------------------------------------------------------------

    def _with(
        self, coords: torch.Tensor, patches: FacePatches | None = None
    ) -> MeshBlock:
        return MeshBlock(
            coords, self.name, self.patches if patches is None else patches
        )

    def translate(self, offset: Sequence[float]) -> MeshBlock:
        """Moved by ``offset``."""
        off = torch.as_tensor(offset, dtype=_DTYPE).view(-1, *[1] * self.ndims)
        return self._with(self.coords + off)

    def scale(
        self, factor: float | Sequence[float], center: Sequence[float] | None = None
    ) -> MeshBlock:
        """Scaled about ``center`` (default: the origin), per axis if a sequence.

        A negative factor mirrors the block; see :meth:`mirror` to keep it
        right-handed.
        """
        f = torch.as_tensor(factor, dtype=_DTYPE).expand(self.ndims)
        c = (
            torch.zeros(self.ndims, dtype=_DTYPE)
            if center is None
            else torch.as_tensor(center, dtype=_DTYPE)
        )
        shape = (-1, *[1] * self.ndims)
        return self._with((self.coords - c.view(shape)) * f.view(shape) + c.view(shape))

    def rotate(
        self,
        angle: float,
        axis: Sequence[float] | None = None,
        center: Sequence[float] | None = None,
        degrees: bool = True,
    ) -> MeshBlock:
        """Rotated about ``center`` (default: the origin).

        Parameters
        ----------
        angle : float
            Rotation angle, counter-clockwise.
        axis : sequence of float or None, optional
            Rotation axis of a 3D block. Default is z.
        center : sequence of float or None, optional
            Point the rotation is about. Default is the origin.
        degrees : bool, optional
            Whether ``angle`` is in degrees. Default is True.

        Returns
        -------
        MeshBlock
            The rotated block.
        """
        rot = rotation_matrix(angle, self.ndims, axis, degrees)
        d = self.ndims
        c = (
            torch.zeros(d, dtype=_DTYPE)
            if center is None
            else torch.as_tensor(center, dtype=_DTYPE)
        )
        pts = self.coords.reshape(d, -1) - c[:, None]
        out = (rot @ pts + c[:, None]).reshape(self.coords.shape)
        return self._with(out)

    def mirror(self, axis: int | str, at: float = 0.0) -> MeshBlock:
        """Mirrored at the plane ``axis = at``.

        The vertex order along the block's own x axis is reversed as well, so the
        block stays right-handed; the patches follow their faces.
        """
        a = _parse_axis(axis)
        coords = self.coords.clone()
        coords[a] = 2.0 * at - coords[a]
        return self._with(coords).flip(0)

    # -----------------------------------------------------------------------
    # Index operations
    # -----------------------------------------------------------------------

    def flip(self, axis: int | str) -> MeshBlock:
        """With the vertex order along a logical axis reversed.

        This mirrors the block's orientation (the vertices do not move); the
        patches of the two faces of the axis swap.
        """
        a = _parse_axis(axis)
        faces = self.patches.as_list()[: 2 * self.ndims]
        faces[2 * a], faces[2 * a + 1] = faces[2 * a + 1], faces[2 * a]
        return self._with(
            self.coords.flip(grid_dim(a, self.ndims)), FacePatches.from_faces(faces)
        )

    def permute(self, axes: Sequence[int | str]) -> MeshBlock:
        """With the logical axes reordered: new axis ``i`` is old axis ``axes[i]``.

        An odd permutation changes the handedness of the block; combine it with
        :meth:`flip` to keep it right-handed.
        """
        order = [_parse_axis(a) for a in axes]
        d = self.ndims
        if sorted(order) != list(range(d)):
            raise ValueError(f"{list(axes)} is not a permutation of the {d} axes.")
        dims = [0] + [grid_dim(order[a], d) for a in reversed(range(d))]
        old = self.patches.as_list()
        new = [old[2 * order[a] + s] for a in range(d) for s in range(2)]
        return self._with(self.coords.permute(*dims), FacePatches.from_faces(new))

    def split(self, axis: int | str, index: int) -> tuple[MeshBlock, MeshBlock]:
        """Two blocks cut at vertex ``index`` along an axis.

        The new faces at the cut have no patch; :class:`~phipict.meshing.Mesh`
        connects them again.
        """
        a = _parse_axis(axis)
        n = self.cells[a]
        if not 0 < index < n:
            raise ValueError(f"Split index must be in (0, {n}), got {index}.")
        dim = grid_dim(a, self.ndims)
        lo = self.coords.narrow(dim, 0, index + 1)
        hi = self.coords.narrow(dim, index, n + 1 - index)
        return (
            MeshBlock(lo, f"{self.name}_0", self.patches.set(2 * a + 1, None)),
            MeshBlock(hi, f"{self.name}_1", self.patches.set(2 * a, None)),
        )

    def concat(self, other: MeshBlock, axis: int | str, tol: float = 1e-9) -> MeshBlock:
        """One block of this and ``other``, joined along an axis.

        The ``+axis`` face of this block must coincide with the ``-axis`` face of
        ``other``. The faces along the joint must have the same patch or none.
        """
        a = _parse_axis(axis)
        d = self.ndims
        fa = self.face_coords(2 * a + 1)
        fb = other.face_coords(2 * a)
        if fa.shape != fb.shape:
            raise ValueError(
                f"Cannot join '{self.name}' and '{other.name}' along {_AXES[a]}: "
                f"the faces have {list(fa.shape[1:])} and {list(fb.shape[1:])} "
                "vertices."
            )
        scale = (
            float(
                (
                    fa.amax(dim=tuple(range(1, d))) - fa.amin(dim=tuple(range(1, d)))
                ).norm()
            )
            or 1.0
        )
        if float((fa - fb).abs().max()) > tol * scale:
            raise ValueError(
                f"Cannot join '{self.name}' and '{other.name}' along {_AXES[a]}: the "
                "faces do not coincide."
            )
        pa, pb = self.patches.as_list(), other.patches.as_list()
        merged: list[Patch | None] = []
        for f in range(2 * d):
            if f == 2 * a:
                merged.append(pa[f])
            elif f == 2 * a + 1:
                merged.append(pb[f])
            elif pa[f] is None or pb[f] is None or pa[f] is pb[f]:
                merged.append(pa[f] or pb[f])
            else:
                raise ValueError(
                    f"Cannot join '{self.name}' and '{other.name}': face "
                    f"{Face(f).name} has the patches '{pa[f].name}' and '{pb[f].name}'."  # type: ignore[union-attr]
                )
        dim = grid_dim(a, d)
        coords = torch.cat(
            [self.coords, other.coords.narrow(dim, 1, other.coords.shape[dim] - 1)], dim
        )
        return MeshBlock(coords, self.name, FacePatches.from_faces(merged))

    # -----------------------------------------------------------------------
    # 2D -> 3D
    # -----------------------------------------------------------------------

    def extrude(
        self,
        z_range: tuple[float, float],
        cells: int,
        grading: GradingLike = None,
        z_patches: tuple[Patch | None, Patch | None] = (None, None),
    ) -> MeshBlock:
        """A 3D block from a 2D block, extruded along z.

        Parameters
        ----------
        z_range : tuple of float
            Start and end of the block in z.
        cells : int
            Number of cells along z.
        grading : Grading or float or None, optional
            Distribution along z. Default is uniform.
        z_patches : tuple of Patch or None, optional
            Patches of the ``-z`` and ``+z`` faces. Default is none (periodic or
            connected).

        Returns
        -------
        MeshBlock
            The 3D block.
        """
        if self.ndims != 2:
            raise ValueError("Only 2D blocks can be extruded.")
        z0, z1 = z_range
        w = as_grading(grading).weights(cells, abs(z1 - z0))
        z = z0 + (z1 - z0) * w
        ny1, nx1 = self.coords.shape[1:]
        xy = self.coords[:, None].expand(2, cells + 1, ny1, nx1)
        zz = z.view(1, -1, 1, 1).expand(1, cells + 1, ny1, nx1)
        coords = torch.cat([xy, zz], 0)
        faces = self.patches.as_list()[:4] + list(z_patches)
        if z1 < z0:  # keep the block right-handed
            return MeshBlock(coords, self.name, FacePatches.from_faces(faces)).flip(2)
        return MeshBlock(coords, self.name, FacePatches.from_faces(faces))

    def extend(
        self, face: Face | int, length: float, cells: int, grading: GradingLike = None
    ) -> MeshBlock:
        """Grown beyond a face by extrapolating its outermost grid lines.

        Each face vertex moves on along the direction of the grid line ending in
        it, e.g. to add an outflow buffer.

        Parameters
        ----------
        face : Face
            The face to grow at; its patch moves to the new face.
        length : float
            Distance to extend by.
        cells : int
            Number of new cell layers.
        grading : Grading or float or None, optional
            Distribution of the new layers, outwards from the face. Default is
            uniform.

        Returns
        -------
        MeshBlock
            The larger block.
        """
        face = Face(face)
        d = self.ndims
        dim = grid_dim(face.axis, d)
        n = self.coords.shape[dim]
        upper = int(face) % 2 == 1
        edge = self.coords.narrow(dim, n - 1 if upper else 0, 1)
        inner = self.coords.narrow(dim, n - 2 if upper else 1, 1)
        direction = edge - inner
        direction = direction / torch.linalg.vector_norm(direction, dim=0, keepdim=True)
        w = as_grading(grading).weights(cells, length)[1:]
        shape = [1] * (d + 1)
        shape[dim] = cells
        layers = edge + direction * (length * w).view(shape)
        if upper:
            coords = torch.cat([self.coords, layers], dim)
        else:
            coords = torch.cat([layers.flip(dim), self.coords], dim)
        return self._with(coords)


def rotation_matrix(
    angle: float, ndims: int, axis: Sequence[float] | None = None, degrees: bool = True
) -> torch.Tensor:
    """Matrix of a counter-clockwise rotation about ``axis`` (z by default)."""
    a = math.radians(angle) if degrees else angle
    c, s = math.cos(a), math.sin(a)
    if ndims == 2:
        if axis is not None:
            raise ValueError("2D rotations have no axis.")
        return torch.tensor([[c, -s], [s, c]], dtype=_DTYPE)
    k = torch.tensor([0.0, 0.0, 1.0] if axis is None else axis, dtype=_DTYPE)
    k = k / torch.linalg.vector_norm(k)
    K = torch.tensor(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=_DTYPE
    )
    return torch.eye(3, dtype=_DTYPE) + s * K + (1 - c) * (K @ K)
