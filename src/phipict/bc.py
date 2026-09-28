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

"""Boundary conditions of a block face, one typed spec per field.

A face is addressed with :class:`Face`, a condition with a spec whose class names
its field, so neither is a string::

    from phipict import Face, bc

    bc.set_bc(block, Face.Y_MINUS, bc.Potential.ThinWall(cw=0.1))
    # phi = 0 only on some face cells, the rest keeps its current condition
    bc.set_bc(block, Face.Y_MINUS, bc.Potential.Dirichlet(), where=mask)

``where`` is a boolean mask over the face's cells, in the block's own cell index
order with the face normal dropped: for a ``[NZ, NY, NX]`` block the ``±y`` faces
take ``[NZ, NX]`` (``[NZ, 1, NX]`` is accepted too). Without ``where`` the spec
applies to the whole face. Per-cell conditions are available for the electric
potential; velocity and passive scalar take one condition per face.

The face must be FIXED (made by ``Block.CloseBoundary`` or ``Block.OpenBoundary``),
since only those carry field conditions. Set conditions before the simulation is
created: :class:`~phipict.MHDSimulation` assembles the potential matrix once.

The *values* of Dirichlet potential cells are the exception: they enter only the
right-hand side, never the matrix, so they may change every step and differ between
batched environments. That makes wall electrodes usable as actuators next to fixed
insulating wall parts::

    bc.set_bc(block, Face.Y_MINUS, bc.Potential.Dirichlet(), where=electrodes)
    sim = phipict.MHDSimulation(domain=domain, ...)
    for step in range(n):
        # [NZ, NX] for all environments, or [B, NZ, NX] per environment
        bc.set_potential_values(block, Face.Y_MINUS, action)
        sim.single_step()
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TypeAlias

import torch

from phipict import _C

__all__ = [
    "Face",
    "Potential",
    "Scalar",
    "Spec",
    "Velocity",
    "get_bc",
    "set_bc",
    "set_potential_values",
]


class Face(IntEnum):
    """A face of a block; the value is the solver's boundary index.

    Being an ``int``, a member is accepted wherever a boundary index is, e.g.
    ``block.getBoundary(Face.X_MINUS)``.
    """

    X_MINUS = 0
    X_PLUS = 1
    Y_MINUS = 2
    Y_PLUS = 3
    Z_MINUS = 4
    Z_PLUS = 5

    @property
    def axis(self) -> int:
        """Spatial axis of the face normal (0 = x, 1 = y, 2 = z)."""
        return self.value >> 1


class Potential:
    """Electric potential conditions (inductionless MHD)."""

    @dataclass(frozen=True)
    class Insulating:
        """Solid insulating wall, ``j_n = 0``. The default of every FIXED face."""

        type = _C.PotentialBC.INSULATING

    @dataclass(frozen=True)
    class Open:
        """Open in/outflow plane, ``dphi/dn = 0`` with ``j_n = (u x B)_n``.

        Must be set explicitly on in/outflows: ``CloseBoundary`` makes them FIXED
        faces just like walls.
        """

        type = _C.PotentialBC.OPEN

    @dataclass(frozen=True, eq=False)
    class Dirichlet:
        """Prescribed ``phi``, e.g. grounded, an odd symmetry plane or an electrode.

        Anchors the potential.

        Attributes
        ----------
        value : float or torch.Tensor or None
            ``phi`` of the cells, as for :func:`set_potential_values`. None keeps the
            current values (0 if none were set).
        """

        value: float | torch.Tensor | None = None
        type = _C.PotentialBC.DIRICHLET

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, Potential.Dirichlet):
                return NotImplemented
            a, b = self.value, other.value
            if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
                return (
                    isinstance(a, torch.Tensor)
                    and isinstance(b, torch.Tensor)
                    and torch.equal(a, b)
                )
            return a == b

    @dataclass(frozen=True)
    class ThinWall:
        """Thin conducting wall, ``dphi/dn = cw * laplace_tau(phi)``.

        Attributes
        ----------
        cw : float
            Wall conductance ratio, > 0. One value per face.
        """

        cw: float
        type = _C.PotentialBC.THIN_WALL

        def __post_init__(self) -> None:
            if not self.cw > 0.0:
                raise ValueError(
                    f"ThinWall needs a wall conductance ratio cw > 0, got {self.cw}."
                )


class Velocity:
    """Velocity conditions."""

    @dataclass(frozen=True)
    class Dirichlet:
        """Prescribed velocity.

        Attributes
        ----------
        value : torch.Tensor or None
            Boundary velocity, static ``[1, dims]`` or varying ``[1, dims, *face]``.
            None keeps the current value (zero, i.e. no-slip, after
            ``CloseBoundary``).
        """

        value: torch.Tensor | None = None
        type = _C.BoundaryConditionType.DIRICHLET

    @dataclass(frozen=True)
    class Neumann:
        """Prescribed normal velocity gradient.

        Attributes
        ----------
        value : torch.Tensor or None
            Boundary gradient, same layout as :class:`Velocity.Dirichlet`. None keeps
            the current value.
        """

        value: torch.Tensor | None = None
        type = _C.BoundaryConditionType.NEUMANN


class Scalar:
    """Passive scalar conditions."""

    @dataclass(frozen=True)
    class Dirichlet:
        """Prescribed passive scalar.

        Attributes
        ----------
        value : torch.Tensor or None
            Boundary scalar of all channels. None keeps the current value.
        channel : int or None
            Channel whose condition type is set; None sets all channels.
        """

        value: torch.Tensor | None = None
        channel: int | None = None
        type = _C.BoundaryConditionType.DIRICHLET

    @dataclass(frozen=True)
    class Neumann:
        """Prescribed normal passive scalar gradient.

        Attributes
        ----------
        value : torch.Tensor or None
            Boundary gradient of all channels. None keeps the current value.
        channel : int or None
            Channel whose condition type is set; None sets all channels.
        """

        value: torch.Tensor | None = None
        channel: int | None = None
        type = _C.BoundaryConditionType.NEUMANN


PotentialSpec: TypeAlias = (
    Potential.Insulating | Potential.Open | Potential.Dirichlet | Potential.ThinWall
)
VelocitySpec: TypeAlias = Velocity.Dirichlet | Velocity.Neumann
ScalarSpec: TypeAlias = Scalar.Dirichlet | Scalar.Neumann
Spec: TypeAlias = PotentialSpec | VelocitySpec | ScalarSpec



def _fixed_boundary(block: _C.Block, face: Face) -> _C.FixedBoundary:
    bound = block.getBoundary(int(face))
    if bound.type != _C.BoundaryType.FIXED:
        raise ValueError(
            f"Face {face.name} of block '{block.name}' is {bound.type.name}, but field "
            "conditions need a FIXED face; use block.CloseBoundary or "
            "block.OpenBoundary first."
        )
    assert isinstance(bound, _C.FixedBoundary)
    return bound


def _face_shape(block: _C.Block, face: Face) -> list[int]:
    """The face's cell grid as ``[(D), H, W]`` with size 1 along the normal."""
    shape = list(block.velocity.shape[2:])  # [(z), y, x]
    shape[len(shape) - 1 - face.axis] = 1
    return shape


def _face_cells(block: _C.Block, face: Face) -> list[int]:
    """The face's cell grid with the normal dropped, the shape of ``where``."""
    shape = _face_shape(block, face)
    return [n for i, n in enumerate(shape) if i != len(shape) - 1 - face.axis]


def _face_mask(block: _C.Block, face: Face, where: torch.Tensor) -> torch.Tensor:
    """``where`` as a ``[1, 1, (D), H, W]`` bool tensor on the block's device."""
    shape = _face_shape(block, face)
    squeezed = _face_cells(block, face)
    if where.dtype != torch.bool:
        raise TypeError(f"where must be a bool tensor, got {where.dtype}.")
    if list(where.shape) not in (squeezed, shape):
        raise ValueError(
            f"where has shape {list(where.shape)}, but face {face.name} of block "
            f"'{block.name}' has {squeezed} cells (or {shape} with the normal kept)."
        )
    return where.reshape(1, 1, *shape).to(block.velocity.device)


def _update_domain(bound: _C.FixedBoundary) -> None:
    """Refresh the solver's pointers after a boundary tensor was replaced."""
    domain = bound.getParentDomain()
    if domain is not None and domain.IsInitialized():
        domain.UpdateDomainData()


def _write_potential_values(
    block: _C.Block,
    face: Face,
    bound: _C.FixedBoundary,
    value: float | torch.Tensor,
    mask: torch.Tensor | None,
) -> None:
    """Write ``value`` into the face's Dirichlet values where ``mask`` is set."""
    full = _face_shape(block, face)
    cells = _face_cells(block, face)
    dtype, device = block.velocity.dtype, block.velocity.device
    v = torch.as_tensor(value, dtype=dtype, device=device)
    if v.dim() == 0:
        v = v.reshape(1, 1, *([1] * len(full))).expand(1, 1, *full)
    elif list(v.shape) == cells:
        v = v.reshape(1, 1, *full)
    elif v.dim() == len(cells) + 1 and list(v.shape[1:]) == cells:
        v = v.reshape(v.shape[0], 1, *full)
    else:
        raise ValueError(
            f"Potential values have shape {list(v.shape)}, but face {face.name} of "
            f"block '{block.name}' takes a scalar, {cells} or [batch, *{cells}]."
        )
    batch = block.velocity.shape[0]
    if v.shape[0] not in (1, batch):
        raise ValueError(
            f"Potential values have batch size {v.shape[0]}, the domain has {batch}."
        )

    current = bound.potentialValues if bound.hasPotentialValues() else None
    if torch.is_grad_enabled() and (
        v.requires_grad or (current is not None and current.requires_grad)
    ):
        # Differentiable: a new tensor per call, so every step keeps its own graph
        # node, with one slice per environment for the per-env gradient
        base = (
            current
            if current is not None
            else torch.zeros(1, 1, *full, dtype=dtype, device=device)
        )
        target = v.expand(batch, 1, *full)
        if mask is not None:
            target = torch.where(mask, target, base.expand(batch, 1, *full))
        bound.setPotentialValues(target.contiguous())
        _update_domain(bound)
        return

    size = max(v.shape[0], current.shape[0] if current is not None else 1)
    if current is None:
        target = torch.zeros(size, 1, *full, dtype=dtype, device=device)
    elif current.shape[0] != size:
        target = current.expand(size, *current.shape[1:]).clone()
    else:
        target = current  # in place: the solver keeps its pointer
    with torch.no_grad():
        if mask is None:
            target.copy_(v.expand_as(target))
        else:
            target.copy_(torch.where(mask, v.expand_as(target), target))
    if target is not current:
        bound.setPotentialValues(target)
        _update_domain(bound)


def set_potential_values(
    block: _C.Block,
    face: Face,
    values: float | torch.Tensor,
    where: torch.Tensor | None = None,
) -> None:
    """Set the prescribed ``phi`` of a face's Dirichlet cells.

    Only the right-hand side depends on these values, so this is cheap enough to
    call every step: the matrix (and its AMG hierarchy) is not rebuilt, and the
    values are written in place whenever their shape allows it. Values that need
    a gradient (or replace ones that do) are installed as a new per-environment
    tensor instead, so the potential kernels differentiate w.r.t. them. Values at
    cells that are not Dirichlet are stored but have no effect.

    Parameters
    ----------
    block : phipict.Block
        Block whose face is set.
    face : Face
        The face; must be FIXED.
    values : float or torch.Tensor
        A scalar, a tensor over the face cells (shape of ``where``) shared by all
        environments, or ``[batch, *face cells]`` with one slice per environment.
    where : torch.Tensor or None
        Boolean mask over the face cells restricting the update. Default is None
        (the whole face).

    Raises
    ------
    ValueError
        If the face is not FIXED or a shape does not match.
    """
    face = Face(face)
    bound = _fixed_boundary(block, face)
    mask = None if where is None else _face_mask(block, face, where)
    _write_potential_values(block, face, bound, values, mask)


def _set_potential(
    block: _C.Block, face: Face, spec: PotentialSpec, where: torch.Tensor | None
) -> None:
    bound = _fixed_boundary(block, face)
    _set_potential_types(block, face, bound, spec, where)
    if isinstance(spec, Potential.Dirichlet) and spec.value is not None:
        mask = None if where is None else _face_mask(block, face, where)
        _write_potential_values(block, face, bound, spec.value, mask)


def _set_potential_types(
    block: _C.Block,
    face: Face,
    bound: _C.FixedBoundary,
    spec: PotentialSpec,
    where: torch.Tensor | None,
) -> None:
    cw = spec.cw if isinstance(spec, Potential.ThinWall) else None
    if where is None:
        bound.setPotentialBC(spec.type, cw=cw)
        return

    mask = _face_mask(block, face, where)
    if bound.hasPotentialTypes():
        assert bound.potentialTypes is not None
        types = bound.potentialTypes.clone()
    else:
        types = torch.full(
            mask.shape,
            int(bound.getPotentialBC()),
            dtype=torch.int8,
            device=mask.device,
        )

    thin_wall = int(_C.PotentialBC.THIN_WALL)
    kept_thin_wall = bool(((types == thin_wall) & ~mask).any())
    if kept_thin_wall and cw is not None and cw != bound.getPotentialCw():
        raise ValueError(
            f"Face {face.name} already has thin-wall cells with cw="
            f"{bound.getPotentialCw()}; a face takes one cw, got {cw}."
        )
    if cw is None and kept_thin_wall:
        cw = bound.getPotentialCw()

    types[mask] = int(spec.type)
    if bool((types == types.flatten()[0]).all()):
        # a uniform face needs no mask (and keeps the per-face fast path)
        bound.setPotentialBC(_C.PotentialBC(int(types.flatten()[0])), cw=cw)
    else:
        bound.setPotentialTypes(types, cw=cw)


def _set_velocity(block: _C.Block, face: Face, spec: VelocitySpec) -> None:
    bound = _fixed_boundary(block, face)
    bound.setVelocityType(spec.type)
    if spec.value is not None:
        bound.setVelocity(spec.value)


def _set_scalar(block: _C.Block, face: Face, spec: ScalarSpec) -> None:
    bound = _fixed_boundary(block, face)
    if not bound.hasPassiveScalar():
        raise ValueError(
            f"Face {face.name} of block '{block.name}' has no passive scalar."
        )
    if spec.channel is None:
        bound.setPassiveScalarType(spec.type)
    else:
        types = list(bound.passiveScalarTypes or [])
        if not 0 <= spec.channel < len(types):
            raise ValueError(
                f"Passive scalar channel {spec.channel} out of range [0, {len(types)})."
            )
        types[spec.channel] = spec.type
        bound.setPassiveScalarType(types)
    if spec.value is not None:
        bound.setPassiveScalar(spec.value)


def set_bc(
    block: _C.Block, face: Face, spec: Spec, where: torch.Tensor | None = None
) -> None:
    """Set a boundary condition on a face of a block.

    Parameters
    ----------
    block : phipict.Block
        Block whose face is set.
    face : Face
        The face; must be FIXED.
    spec : Spec
        The condition, e.g. ``Potential.Dirichlet()``. Its class selects the field.
    where : torch.Tensor or None
        Boolean mask over the face cells (see the module docstring). Only the
        masked cells take ``spec``, the others keep their condition. None sets the
        whole face. Default is None.

    Raises
    ------
    ValueError
        If the face is not FIXED, ``where`` does not match the face, or a thin-wall
        ``cw`` conflicts with the face's remaining thin-wall cells.
    NotImplementedError
        If ``where`` is given for a velocity or passive scalar condition.
    TypeError
        If ``spec`` is not a condition spec, or ``where`` is not boolean.
    """
    face = Face(face)
    if isinstance(spec, PotentialSpec):
        _set_potential(block, face, spec, where)
        return
    if where is not None:
        raise NotImplementedError(
            "Per-cell conditions (where=...) are only implemented for Potential."
        )
    if isinstance(spec, VelocitySpec):
        _set_velocity(block, face, spec)
    elif isinstance(spec, ScalarSpec):
        _set_scalar(block, face, spec)
    else:
        raise TypeError(f"Not a boundary condition spec: {spec!r}.")


def get_bc(
    block: _C.Block, face: Face, field: type[Potential] | type[Velocity] | type[Scalar]
) -> Spec | list[ScalarSpec] | torch.Tensor:
    """Get the boundary condition of one field on a face of a block.

    Parameters
    ----------
    block : phipict.Block
        Block whose face is read.
    face : Face
        The face; must be FIXED.
    field : type
        ``Potential``, ``Velocity`` or ``Scalar``.

    Returns
    -------
    Spec or list of Spec or torch.Tensor
        The face's condition. For ``Scalar`` one spec per channel. For a
        ``Potential`` that varies over the face, its per-cell ``PotentialBC``
        values as an int8 tensor over the face cells (normal dropped, like
        ``where``). A Dirichlet potential with values carries them as
        ``[batch or 1, *face cells]``.
    """
    face = Face(face)
    bound = _fixed_boundary(block, face)
    if field is Potential:
        if bound.hasPotentialTypes():
            assert bound.potentialTypes is not None
            return bound.potentialTypes.reshape(_face_cells(block, face)).clone()
        bc_type = bound.getPotentialBC()
        if bc_type == _C.PotentialBC.THIN_WALL:
            return Potential.ThinWall(cw=bound.getPotentialCw())
        if bc_type == _C.PotentialBC.DIRICHLET and bound.hasPotentialValues():
            assert bound.potentialValues is not None
            values = bound.potentialValues
            return Potential.Dirichlet(
                value=values.reshape(values.shape[0], *_face_cells(block, face)).clone()
            )
        if bc_type == _C.PotentialBC.DIRICHLET:
            return Potential.Dirichlet()
        if bc_type == _C.PotentialBC.OPEN:
            return Potential.Open()
        return Potential.Insulating()
    if field is Velocity:
        velocity_spec = (
            Velocity.Dirichlet
            if bound.velocityType == _C.BoundaryConditionType.DIRICHLET
            else Velocity.Neumann
        )
        return velocity_spec(value=bound.velocity)
    if field is Scalar:
        if not bound.hasPassiveScalar():
            raise ValueError(
                f"Face {face.name} of block '{block.name}' has no passive scalar."
            )
        return [
            (
                Scalar.Dirichlet
                if t == _C.BoundaryConditionType.DIRICHLET
                else Scalar.Neumann
            )(channel=c)
            for c, t in enumerate(bound.passiveScalarTypes or [])
        ]
    raise TypeError(f"Unknown field {field!r}; use Potential, Velocity or Scalar.")
