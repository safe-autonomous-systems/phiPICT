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

"""Transfinite interpolation of block vertices from the block edges.

Edges of one logical axis are indexed by the corner bits of the other axes, in
axis order: the x edges of a 3D block are ``(y, z) = (0, 0), (1, 0), (0, 1), (1,
1)``, the y edges ``(x, z)`` and the z edges ``(x, y)``; in 2D the x edges are
``y = 0, 1`` and the y edges ``x = 0, 1``. Every edge runs in the positive
direction of its axis.

Opposite edges may carry different vertex distributions (blockMesh's
``edgeGrading``). As in blockMesh, the interpolation parameter of an interior
vertex then blends the edge weights of its axis with the parameters of the other
axes, which reproduces the edge vertices exactly and keeps straight grid lines
straight when all edges of an axis share one distribution.
"""

from __future__ import annotations

import torch

__all__ = ["tfi_2d", "tfi_3d"]


def tfi_2d(
    edges: list[list[torch.Tensor]], weights: list[list[torch.Tensor]]
) -> torch.Tensor:
    """Vertices of a 2D block from its four edges.

    Parameters
    ----------
    edges : list of list of torch.Tensor
        ``edges[axis][e]``: points ``[n_axis + 1, dims]`` of edge ``e`` of the axis.
    weights : list of list of torch.Tensor
        ``weights[axis][e]``: normalised arc lengths of the edge's points.

    Returns
    -------
    torch.Tensor
        Vertices ``[dims, ny + 1, nx + 1]``.
    """
    (b, t), (lft, r) = edges
    (wb, wt), (wl, wr) = weights
    ub, ut = wb[None, :], wt[None, :]  # [1, nx+1]
    vl, vr = wl[:, None], wr[:, None]  # [ny+1, 1]
    # u = ub (1 - v) + ut v and v = vl (1 - u) + vr u, solved in closed form
    u = (ub + (ut - ub) * vl) / (1.0 - (ut - ub) * (vr - vl))
    v = vl + (vr - vl) * u
    u, v = u[None], v[None]  # [1, ny+1, nx+1]
    B, T = b.T[:, None, :], t.T[:, None, :]  # [d, 1, nx+1]
    L, R = lft.T[:, :, None], r.T[:, :, None]  # [d, ny+1, 1]
    p00, p10 = b[0][:, None, None], b[-1][:, None, None]
    p01, p11 = t[0][:, None, None], t[-1][:, None, None]
    return (
        (1 - v) * B
        + v * T
        + (1 - u) * L
        + u * R
        - (
            (1 - u) * (1 - v) * p00
            + u * (1 - v) * p10
            + (1 - u) * v * p01
            + u * v * p11
        )
    )


def _bilinear(a: torch.Tensor, b: torch.Tensor) -> list[torch.Tensor]:
    """Blend factors of the 4 edges indexed by bits ``(a, b)``."""
    return [(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b]


def _mix(factors: list[torch.Tensor], values: torch.Tensor) -> torch.Tensor:
    """``sum(factors[e] * values[e])`` over the 4 edges of an axis."""
    return torch.stack([f * values[e] for e, f in enumerate(factors)]).sum(0)


def tfi_3d(
    edges: list[list[torch.Tensor]],
    weights: list[list[torch.Tensor]],
    iterations: int = 20,
) -> torch.Tensor:
    """Vertices of a 3D block from its twelve edges.

    Parameters
    ----------
    edges : list of list of torch.Tensor
        ``edges[axis][e]``: points ``[n_axis + 1, 3]`` of edge ``e`` of the axis,
        see the module docstring for the edge order.
    weights : list of list of torch.Tensor
        ``weights[axis][e]``: normalised arc lengths of the edge's points.
    iterations : int, optional
        Fixed-point iterations for the blended parameters. Default is 20.

    Returns
    -------
    torch.Tensor
        Vertices ``[3, nz + 1, ny + 1, nx + 1]``.
    """
    nx, ny, nz = (len(edges[a][0]) for a in range(3))
    wx = torch.stack(weights[0])[:, None, None, :]  # [4, 1, 1, nx+1]
    wy = torch.stack(weights[1])[:, None, :, None]  # [4, 1, ny+1, 1]
    wz = torch.stack(weights[2])[:, :, None, None]  # [4, nz+1, 1, 1]
    shape = (nz, ny, nx)
    u = wx.mean(0).expand(shape)
    v = wy.mean(0).expand(shape)
    w = wz.mean(0).expand(shape)
    uniform = all(
        torch.equal(ws[0], ws[e]) for ws in weights for e in range(1, len(ws))
    )
    for _ in range(0 if uniform else iterations):
        u, v, w = (
            _mix(_bilinear(v, w), wx),
            _mix(_bilinear(u, w), wy),
            _mix(_bilinear(u, v), wz),
        )

    ex = [e.T[:, None, None, :] for e in edges[0]]  # [3, 1, 1, nx+1]
    ey = [e.T[:, None, :, None] for e in edges[1]]
    ez = [e.T[:, :, None, None] for e in edges[2]]
    out = torch.zeros((3, *shape), dtype=edges[0][0].dtype)
    for f, e in zip(_bilinear(v, w), ex, strict=True):
        out += f * e
    for f, e in zip(_bilinear(u, w), ey, strict=True):
        out += f * e
    for f, e in zip(_bilinear(u, v), ez, strict=True):
        out += f * e
    # corners C[i, j, k] from the x edges (j, k)
    for k in range(2):
        for j in range(2):
            e = edges[0][j + 2 * k]
            for i, c in ((0, e[0]), (1, e[-1])):
                f = (u if i else 1 - u) * (v if j else 1 - v) * (w if k else 1 - w)
                out -= 2.0 * f * c[:, None, None, None]
    return out
