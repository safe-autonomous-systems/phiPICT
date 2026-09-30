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

"""Block connections found from the geometry.

Two block faces are connected when their vertex grids coincide, in any relative
orientation; periodic connections when they coincide after a translation. The
orientation is turned into the axis codes of ``Block.ConnectBlock``, so nobody has
to work those out by hand.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from phipict.bc import Face

from .block import MeshBlock, tangential_axes

__all__ = [
    "Connection",
    "NonConformingInterfaceError",
    "Orientation",
    "find_connections",
    "match_faces",
]


class NonConformingInterfaceError(ValueError):
    """Two block faces touch but their vertices do not coincide."""


@dataclass(frozen=True)
class Orientation:
    """How the in-face axes of one face map onto those of another.

    Attributes
    ----------
    perm : tuple of int
        ``perm[k]``: the tangential axis slot of face B that slot ``k`` of face A
        runs along (slots in the order of
        :func:`~phipict.meshing.block.tangential_axes`).
    flip : tuple of bool
        ``flip[k]``: whether slot ``k`` of A runs opposite to its partner in B.
    """

    perm: tuple[int, ...]
    flip: tuple[bool, ...]


@dataclass(frozen=True)
class Connection:
    """A connection of face ``face_a`` of block ``block_a`` to a face of ``block_b``.

    Attributes
    ----------
    block_a, block_b : int
        Indices of the blocks in the mesh (may be equal).
    face_a, face_b : Face
        The connected faces.
    orientation : Orientation
        Relative orientation of the faces.
    translation : tuple of float or None
        Offset from face A to face B of a periodic connection, None for faces that
        coincide.
    """

    block_a: int
    face_a: Face
    block_b: int
    face_b: Face
    orientation: Orientation
    translation: tuple[float, ...] | None = None

    def axis_codes(self, ndims: int) -> tuple[int, int]:
        """``axis1``/``axis2`` arguments of ``Block.ConnectBlock``.

        Each code is ``(axis of B) << 1 | inverted`` for the in-face axes of A in
        the solver's order; the second is -1 in 2D.
        """
        tb = tangential_axes(self.face_b, ndims)
        codes = [
            (tb[p] << 1) | int(f)
            for p, f in zip(self.orientation.perm, self.orientation.flip, strict=True)
        ]
        return codes[0], codes[1] if ndims == 3 else -1

    @property
    def is_periodic_axis(self) -> bool:
        """Whether this is the plain periodicity of one block along one axis."""
        return (
            self.block_a == self.block_b
            and self.face_a.axis == self.face_b.axis
            and self.face_a != self.face_b
            and all(p == k for k, p in enumerate(self.orientation.perm))
            and not any(self.orientation.flip)
        )


def _orientations(ndims: int) -> list[Orientation]:
    n = ndims - 1
    return [
        Orientation(tuple(perm), tuple(flip))
        for perm in itertools.permutations(range(n))
        for flip in itertools.product((False, True), repeat=n)
    ]


def _oriented(face_b: torch.Tensor, o: Orientation) -> torch.Tensor:
    """Face B's vertices indexed like face A's under an orientation."""
    out = face_b.permute(0, *[p + 1 for p in o.perm])
    dims = [k + 1 for k, f in enumerate(o.flip) if f]
    return out.flip(dims) if dims else out


def _spacing(face: torch.Tensor) -> float:
    """Smallest distance between neighbouring vertices of a face."""
    d = []
    for dim in range(1, face.dim()):
        diff = face.diff(dim=dim)
        d.append(float(torch.linalg.vector_norm(diff, dim=0).min()))
    return min(d)


def match_faces(
    face_a: torch.Tensor,
    face_b: torch.Tensor,
    tol: float,
    translation: torch.Tensor | None = None,
) -> Orientation | None:
    """The orientation under which two faces coincide, if any.

    Parameters
    ----------
    face_a, face_b : torch.Tensor
        Face vertices as returned by :meth:`MeshBlock.face_coords`.
    tol : float
        Absolute tolerance on the vertex positions.
    translation : torch.Tensor or None, optional
        Offset added to face A before comparing. Default is None.

    Returns
    -------
    Orientation or None
        The orientation, None if the faces do not coincide.
    """
    ndims = face_a.shape[0]
    a = (
        face_a
        if translation is None
        else face_a + translation.view(-1, *[1] * (ndims - 1))
    )
    for o in _orientations(ndims):
        b = _oriented(face_b, o)
        if b.shape == a.shape and float((a - b).abs().max()) <= tol:
            return o
    return None


def _corners(face: torch.Tensor) -> torch.Tensor:
    """Corner vertices ``[n, dims]`` of a face."""
    if face.dim() == 2:
        return torch.stack([face[:, 0], face[:, -1]])
    return torch.stack([face[:, 0, 0], face[:, -1, 0], face[:, 0, -1], face[:, -1, -1]])


def _corner_sets_match(ca: torch.Tensor, cb: torch.Tensor, tol: float) -> bool:
    dist = torch.cdist(ca, cb)
    return bool(
        (dist.min(dim=1).values <= tol).all() and (dist.min(dim=0).values <= tol).all()
    )


def find_connections(
    blocks: Sequence[MeshBlock],
    faces: Sequence[tuple[int, Face]],
    rel_tol: float = 1e-4,
    translation: Sequence[float] | None = None,
) -> list[Connection]:
    """Connections between free faces whose vertices coincide.

    Parameters
    ----------
    blocks : sequence of MeshBlock
        The blocks.
    faces : sequence of tuple of int and Face
        Candidate ``(block, face)`` pairs, usually all faces without a patch or
        connection.
    rel_tol : float, optional
        Tolerance relative to the smallest vertex spacing of a face. Default is
        1e-4.
    translation : sequence of float or None, optional
        Look for periodic partners at this offset instead. Default is None.

    Returns
    -------
    list of Connection
        The connections, each face used at most once.

    Raises
    ------
    NonConformingInterfaceError
        If two faces share their corners but not their vertices, e.g. because the
        blocks have different resolutions or gradings along the interface.
    """
    coords = [blocks[b].face_coords(f) for b, f in faces]
    tols = [rel_tol * _spacing(c) for c in coords]
    corners = [_corners(c) for c in coords]
    shift = (
        None
        if translation is None
        else torch.as_tensor(translation, dtype=torch.float64)
    )
    used: set[int] = set()
    found: list[Connection] = []
    for i in range(len(faces)):
        if i in used:
            continue
        ca = corners[i] if shift is None else corners[i] + shift
        for j in range(len(faces)):
            if j == i or j in used:
                continue
            if shift is None and j < i:
                continue
            tol = min(tols[i], tols[j])
            if not _corner_sets_match(ca, corners[j], tol):
                continue
            o = match_faces(coords[i], coords[j], tol, shift)
            if o is None:
                (ba, fa), (bb, fb) = faces[i], faces[j]
                na = [n - 1 for n in coords[i].shape[1:]]
                nb = [n - 1 for n in coords[j].shape[1:]]
                detail = (
                    f"{na} vs {nb} cells; match the resolutions or split a block"
                    if sorted(na) != sorted(nb)
                    else "same cell counts but different vertex positions; match the "
                    "gradings (or curved edges) along the interface"
                )
                raise NonConformingInterfaceError(
                    f"Face {fa.name} of block '{blocks[ba].name}' and face {fb.name} "
                    f"of block '{blocks[bb].name}' share their corners but are not "
                    f"conforming: {detail}."
                )
            (ba, fa), (bb, fb) = faces[i], faces[j]
            found.append(
                Connection(
                    ba,
                    fa,
                    bb,
                    fb,
                    o,
                    None if shift is None else tuple(float(x) for x in shift),
                )
            )
            used.update((i, j))
            break
    return found
