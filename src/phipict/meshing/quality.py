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

"""Cell quality measures of structured blocks.

All functions take block vertices ``[dims, (nz + 1,) ny + 1, nx + 1]`` and return
one value per cell, ``[(nz,) ny, nx]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = [
    "MeshReport",
    "aspect_ratio",
    "cell_volumes",
    "non_orthogonality",
    "scaled_jacobian",
]


def _corner(c: torch.Tensor, bits: tuple[int, ...]) -> torch.Tensor:
    """Vertex ``bits = (i, j[, k])`` of every cell."""
    d = c.shape[0]
    idx: list[slice] = [slice(None)]
    for a in reversed(range(d)):  # grid dims are (z,) y, x
        idx.append(slice(1, None) if bits[a] else slice(None, -1))
    return c[tuple(idx)]


def _edge_vectors(c: torch.Tensor, axis: int) -> list[torch.Tensor]:
    """The cell edge vectors along ``axis``, one per parallel edge."""
    d = c.shape[0]
    others = [a for a in range(d) if a != axis]
    out = []
    for rest in range(2 ** (d - 1)):
        lo = [0] * d
        for n, a in enumerate(others):
            lo[a] = (rest >> n) & 1
        hi = list(lo)
        hi[axis] = 1
        out.append(_corner(c, tuple(hi)) - _corner(c, tuple(lo)))
    return out


def cell_volumes(coords: torch.Tensor) -> torch.Tensor:
    """Signed cell volumes (areas in 2D), from the Jacobian at the cell centre.

    Negative values mark inverted (left-handed) cells.
    """
    d = coords.shape[0]
    jac = [torch.stack(_edge_vectors(coords, a)).mean(0) for a in range(d)]
    if d == 2:
        return jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0]
    return (jac[0] * torch.linalg.cross(jac[1], jac[2], dim=0)).sum(0)


def scaled_jacobian(coords: torch.Tensor) -> torch.Tensor:
    """Smallest normalised corner Jacobian of each cell, in [-1, 1].

    1 is a rectangular cell; values at or below 0 mark degenerate or inverted
    cells.
    """
    d = coords.shape[0]
    worst: torch.Tensor | None = None
    for bits in range(2**d):
        corner = tuple((bits >> a) & 1 for a in range(d))
        vecs = []
        for a in range(d):
            lo = list(corner)
            hi = list(corner)
            lo[a], hi[a] = 0, 1
            vecs.append(_corner(coords, tuple(hi)) - _corner(coords, tuple(lo)))
        if d == 2:
            det = vecs[0][0] * vecs[1][1] - vecs[0][1] * vecs[1][0]
        else:
            det = (vecs[0] * torch.linalg.cross(vecs[1], vecs[2], dim=0)).sum(0)
        norm = torch.stack([torch.linalg.vector_norm(v, dim=0) for v in vecs]).prod(0)
        sj = det / norm.clamp_min(1e-300)
        worst = sj if worst is None else torch.minimum(worst, sj)
    assert worst is not None
    return worst


def aspect_ratio(coords: torch.Tensor) -> torch.Tensor:
    """Longest over shortest mean edge length of each cell."""
    d = coords.shape[0]
    lengths = torch.stack(
        [
            torch.stack(
                [torch.linalg.vector_norm(v, dim=0) for v in _edge_vectors(coords, a)]
            ).mean(0)
            for a in range(d)
        ]
    )
    return lengths.amax(0) / lengths.amin(0).clamp_min(1e-300)


def non_orthogonality(coords: torch.Tensor) -> torch.Tensor:
    """Largest non-orthogonality angle of a cell's inner faces, in degrees.

    The angle between the face normal and the line joining the two cell centres,
    for the faces between cells of the block (block interfaces are not included).
    Cells without an inner face get 0.
    """
    d = coords.shape[0]
    centres = torch.stack(
        [_corner(coords, tuple((b >> a) & 1 for a in range(d))) for b in range(2**d)]
    ).mean(0)
    worst = torch.zeros(centres.shape[1:], dtype=coords.dtype)
    for axis in range(d):
        gd = d - axis  # grid dim of the axis in the vertex tensor
        cd = gd  # same index in the centre tensor [dims, ...]
        n = centres.shape[cd]
        if n < 2:
            continue
        link = centres.narrow(cd, 1, n - 1) - centres.narrow(cd, 0, n - 1)
        # inner faces: vertex slabs 1..n-1 along the axis
        slab = coords.narrow(gd, 1, n - 1)
        others = [a for a in range(d) if a != axis]
        if d == 2:
            (o,) = others
            od = d - o
            m = slab.shape[od]
            t = slab.narrow(od, 1, m - 1) - slab.narrow(od, 0, m - 1)
            normal = torch.stack([t[1], -t[0]])
        else:
            o1, o2 = others
            g1, g2 = d - o1, d - o2
            p = [
                slab.narrow(g1, i, slab.shape[g1] - 1).narrow(g2, j, slab.shape[g2] - 1)
                for i in (0, 1)
                for j in (0, 1)
            ]
            normal = torch.linalg.cross(p[3] - p[0], p[1] - p[2], dim=0)
        cos = (normal * link).sum(0).abs() / (
            torch.linalg.vector_norm(normal, dim=0)
            * torch.linalg.vector_norm(link, dim=0)
        ).clamp_min(1e-300)
        angle = torch.rad2deg(torch.arccos(cos.clamp(max=1.0)))
        lo = [slice(None)] * d
        hi = [slice(None)] * d
        lo[cd - 1] = slice(None, -1)
        hi[cd - 1] = slice(1, None)
        worst[tuple(lo)] = torch.maximum(worst[tuple(lo)], angle)
        worst[tuple(hi)] = torch.maximum(worst[tuple(hi)], angle)
    return worst


@dataclass
class MeshReport:
    """Quality summary of a mesh, from :meth:`Mesh.check <phipict.meshing.Mesh.check>`.

    Attributes
    ----------
    n_blocks, n_cells : int
        Size of the mesh.
    min_volume : float
        Smallest cell volume (area in 2D).
    min_scaled_jacobian : float
        Smallest normalised corner Jacobian, 1 for rectangular cells.
    max_aspect_ratio : float
        Largest cell aspect ratio.
    max_non_orthogonality : float
        Largest non-orthogonality within the blocks, in degrees.
    errors : list of str
        Problems that make the mesh unusable for a domain.
    warnings : list of str
        Problems that may hurt the solution.
    """

    n_blocks: int
    n_cells: int
    min_volume: float
    min_scaled_jacobian: float
    max_aspect_ratio: float
    max_non_orthogonality: float
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether there are no errors."""
        return not self.errors

    def raise_if_invalid(self) -> None:
        """Raise a ValueError listing the errors, if any."""
        if self.errors:
            raise ValueError("Invalid mesh:\n  " + "\n  ".join(self.errors))

    def __str__(self) -> str:
        lines = [
            f"{self.n_blocks} blocks, {self.n_cells} cells",
            f"min volume {self.min_volume:.4g}, min scaled Jacobian "
            f"{self.min_scaled_jacobian:.3f}",
            f"max aspect ratio {self.max_aspect_ratio:.3g}, max non-orthogonality "
            f"{self.max_non_orthogonality:.1f} deg",
        ]
        lines += [f"ERROR: {e}" for e in self.errors]
        lines += [f"warning: {w}" for w in self.warnings]
        return "\n".join(lines)
