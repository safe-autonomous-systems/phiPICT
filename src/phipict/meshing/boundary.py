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

"""Named boundary patches and their conditions.

A :class:`Patch` names a set of block faces (``"inlet"``, ``"walls"``) and carries
the condition the faces get when the mesh becomes a domain::

    inlet = pm.Patch("inlet", pm.Inflow((1.0, 0.0)))
    walls = pm.Patch("walls", pm.Wall())
    block = pm.make_box((0, 0), (4, 1), cells=(64, 16),
                   patches=pm.FacePatches(x_minus=inlet, y_minus=walls, y_plus=walls))

Imported meshes carry the patch names of the file; their conditions are set with
:meth:`Mesh.set_bc <phipict.meshing.Mesh.set_bc>`. Conditions of further fields
(passive scalar, electric potential) are given as ``extra`` specs of
:mod:`phipict.bc`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, fields, replace
from typing import TypeAlias

import torch

from phipict import _C
from phipict import bc as _bc
from phipict.bc import Face

__all__ = [
    "BoundarySpec",
    "FacePatches",
    "FreeSlip",
    "Inflow",
    "Outflow",
    "Patch",
    "VelocityLike",
    "VelocityProfile",
    "Wall",
    "apply_boundary",
]

VelocityProfile: TypeAlias = Callable[[torch.Tensor], torch.Tensor]
VelocityLike: TypeAlias = torch.Tensor | Sequence[float] | VelocityProfile


@dataclass(frozen=True, eq=False)
class Wall:
    """No-slip wall, optionally moving (``CloseBoundary``).

    Attributes
    ----------
    velocity : torch.Tensor or sequence of float or Callable or None
        Wall velocity: static ``(u, v[, w])``, varying ``[1, dims, *face]``, or a
        profile, see :class:`Inflow`. None is a resting wall.
    extra : tuple of phipict.bc.Spec
        Further conditions, e.g. ``bc.Potential.ThinWall(cw=0.1)``.
    """

    velocity: VelocityLike | None = None
    extra: tuple[_bc.Spec, ...] = ()


@dataclass(frozen=True, eq=False)
class FreeSlip:
    """Free-slip wall or symmetry plane (``OpenBoundary``).

    Attributes
    ----------
    extra : tuple of phipict.bc.Spec
        Further conditions.
    """

    extra: tuple[_bc.Spec, ...] = ()


@dataclass(frozen=True, eq=False)
class Inflow:
    """Prescribed inflow velocity.

    Attributes
    ----------
    velocity : torch.Tensor or sequence of float or Callable
        Inflow velocity: static ``(u, v[, w])``, varying ``[1, dims, *face]``, or a
        profile ``fn(points) -> velocities`` that maps the face cell centres
        ``[n, dims]`` to velocities ``[n, dims]``, e.g.
        ``lambda p: torch.stack([1 - p[:, 1] ** 2, 0 * p[:, 1]], 1)``.
    extra : tuple of phipict.bc.Spec
        Further conditions.
    """

    velocity: VelocityLike
    extra: tuple[_bc.Spec, ...] = ()


@dataclass(frozen=True, eq=False)
class Outflow:
    """Outflow with a varying velocity.

    The face velocity must be updated during the simulation, e.g. with the
    callback of :meth:`Mesh.outflow_hook <phipict.meshing.Mesh.outflow_hook>`.
    :meth:`Mesh.get_domain <phipict.meshing.Mesh.get_domain>` scales the initial
    velocity so that the outflow balances the inflow.

    Attributes
    ----------
    velocity : torch.Tensor or sequence of float or Callable or None
        Initial velocity, as for :class:`Inflow`. None is the outward face
        normal, which the flux balance then scales.
    extra : tuple of phipict.bc.Spec
        Further conditions, e.g. ``bc.Potential.Open()``.
    """

    velocity: VelocityLike | None = None
    extra: tuple[_bc.Spec, ...] = ()


BoundarySpec: TypeAlias = Wall | FreeSlip | Inflow | Outflow


@dataclass(eq=False)
class Patch:
    """A named group of boundary faces with a condition.

    Patches are compared by identity; a mesh keeps one patch per name.

    Attributes
    ----------
    name : str
        Name of the patch.
    bc : BoundarySpec or None
        Condition of the faces. None until set, e.g. for imported patches.
    kind : str or None
        Type of the patch in the file it was read from (``"wall"``, ``"patch"``,
        ...), for information.
    """

    name: str
    bc: BoundarySpec | None = None
    kind: str | None = None

    def __repr__(self) -> str:
        bc = type(self.bc).__name__ if self.bc is not None else None
        return f"Patch({self.name!r}, bc={bc})"


_FACE_FIELDS = ("x_minus", "x_plus", "y_minus", "y_plus", "z_minus", "z_plus")


@dataclass(frozen=True)
class FacePatches:
    """The patch of each face of a block; None for faces without one.

    Faces without a patch are connected to other blocks, or made periodic.
    """

    x_minus: Patch | None = None
    x_plus: Patch | None = None
    y_minus: Patch | None = None
    y_plus: Patch | None = None
    z_minus: Patch | None = None
    z_plus: Patch | None = None

    def __getitem__(self, face: Face | int) -> Patch | None:
        patch: Patch | None = getattr(self, _FACE_FIELDS[int(face)])
        return patch

    def set(self, face: Face | int, patch: Patch | None) -> FacePatches:
        """A copy with the patch of one face replaced."""
        return replace(self, **{_FACE_FIELDS[int(face)]: patch})

    def items(self) -> Iterator[tuple[Face, Patch]]:
        """The faces that have a patch, with their patch."""
        for f in fields(self):
            patch = getattr(self, f.name)
            if patch is not None:
                yield Face(_FACE_FIELDS.index(f.name)), patch

    @classmethod
    def from_faces(cls, patches: Sequence[Patch | None]) -> FacePatches:
        """From one entry per face, in :class:`~phipict.Face` order."""
        return cls(**dict(zip(_FACE_FIELDS, patches, strict=False)))

    def as_list(self) -> list[Patch | None]:
        """One entry per face, in :class:`~phipict.Face` order."""
        return [getattr(self, name) for name in _FACE_FIELDS]

    @classmethod
    def uniform(cls, patch: Patch | None, ndims: int) -> FacePatches:
        """Every face of an ``ndims`` block with the same patch."""
        return cls.from_faces([patch] * (2 * ndims))


def _velocity_tensor(
    velocity: VelocityLike, block: _C.Block, face: Face
) -> torch.Tensor:
    ref = block.velocity
    dims = ref.shape[1]
    if callable(velocity):
        centres = _face_centres(block, face)  # [*face, dims]
        shape = centres.shape[:-1]
        vel = torch.as_tensor(velocity(centres.reshape(-1, dims)), dtype=ref.dtype)
        if vel.shape != (centres.shape[:-1].numel(), dims):
            raise ValueError(
                f"The velocity profile of face {face.name} of block '{block.name}' "
                f"must return [{shape.numel()}, {dims}], got {list(vel.shape)}."
            )
        # [1, dims, (z,) y, x] with size 1 along the face normal
        vel = vel.reshape(*shape, dims).movedim(-1, 0)[None]
        return vel.to(ref.device).contiguous()
    vel = torch.as_tensor(velocity, dtype=ref.dtype).to(ref.device)
    if vel.dim() == 1:
        if vel.shape[0] != dims:
            raise ValueError(
                f"Velocity of face {face.name} of block '{block.name}' needs {dims} "
                f"components, got {vel.shape[0]}."
            )
        vel = vel.view(1, dims)
    return vel.contiguous()


def _face_centres(block: _C.Block, face: Face) -> torch.Tensor:
    """Face cell centres ``[(z,) y, x, dims]``, size 1 along the face normal."""
    coords = block.vertexCoordinates
    if coords is None:
        raise ValueError(
            f"Block '{block.name}' has no vertex coordinates for a profile."
        )
    c = coords[0].detach().double().cpu()  # [dims, (z+1,) y+1, x+1]
    dims = c.shape[0]
    normal = dims - face.axis
    idx = 0 if int(face) % 2 == 0 else c.shape[normal] - 1
    f = c.narrow(normal, idx, 1)
    for d in range(1, dims + 1):
        if d != normal:
            f = 0.5 * (f.narrow(d, 0, f.shape[d] - 1) + f.narrow(d, 1, f.shape[d] - 1))
    return f.movedim(0, -1)


def _face_normals(block: _C.Block, face: Face) -> torch.Tensor:
    """Outward unit normals of the face cells ``[(z,) y, x, dims]``."""
    coords = block.vertexCoordinates
    if coords is None:
        raise ValueError(f"Block '{block.name}' has no vertex coordinates.")
    c = coords[0].detach().double().cpu()
    dims = c.shape[0]
    normal_dim = dims - face.axis
    last = c.shape[normal_dim] - 1
    upper = int(face) % 2 == 1
    f = c.narrow(normal_dim, last if upper else 0, 1)
    inner = c.narrow(normal_dim, last - 1 if upper else 1, 1)
    other = [d for d in range(1, dims + 1) if d != normal_dim]

    def cells(t: torch.Tensor) -> torch.Tensor:
        for d in other:
            t = 0.5 * (t.narrow(d, 0, t.shape[d] - 1) + t.narrow(d, 1, t.shape[d] - 1))
        return t

    if dims == 2:
        (d,) = other
        t = f.narrow(d, 1, f.shape[d] - 1) - f.narrow(d, 0, f.shape[d] - 1)
        n = torch.stack([t[1], -t[0]])
    else:
        d1, d2 = other
        p = [
            f.narrow(d1, i, f.shape[d1] - 1).narrow(d2, j, f.shape[d2] - 1)
            for i in (0, 1)
            for j in (0, 1)
        ]
        n = torch.linalg.cross(p[3] - p[0], p[1] - p[2], dim=0)
    n = n / torch.linalg.vector_norm(n, dim=0, keepdim=True)
    outward = cells(f) - cells(inner)
    sign = torch.sign((n * outward).sum(0, keepdim=True))
    return (n * sign).movedim(0, -1)


def apply_boundary(spec: BoundarySpec, block: _C.Block, face: Face) -> None:
    """Set up a face of a solver block with a condition.

    Parameters
    ----------
    spec : BoundarySpec
        The condition.
    block : phipict.Block
        The block.
    face : Face
        The face.

    Raises
    ------
    TypeError
        If ``spec`` is not a boundary spec.
    """
    face = Face(face)
    if isinstance(spec, Wall | Inflow):
        block.CloseBoundary(int(face))
        if spec.velocity is not None:
            vel = _velocity_tensor(spec.velocity, block, face)
            _bc.set_bc(block, face, _bc.Velocity.Dirichlet(vel))
    elif isinstance(spec, FreeSlip):
        block.OpenBoundary(int(face))
    elif isinstance(spec, Outflow):
        block.CloseBoundary(int(face))
        if spec.velocity is not None:
            vel = _velocity_tensor(spec.velocity, block, face)
        else:
            ref = block.velocity
            normals = _face_normals(block, face)  # [*face, dims]
            vel = normals.movedim(-1, 0)[None].to(ref.device, ref.dtype).contiguous()
        _bc.set_bc(block, face, _bc.Velocity.Dirichlet(vel))
        bound = block.getBoundary(int(face))
        assert isinstance(bound, _C.FixedBoundary)
        bound.makeVelocityVarying()
    else:
        raise TypeError(f"Not a boundary spec: {spec!r}.")
    for extra in spec.extra:
        _bc.set_bc(block, face, extra)
