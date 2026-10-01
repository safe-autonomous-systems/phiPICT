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

"""A multi-block mesh and its conversion to a solver domain.

A :class:`Mesh` holds discretised blocks, the connections between them and the
patches of the boundary faces. Connections are found from the geometry::

    mesh = pm.Mesh([left, right])        # coinciding faces are connected
    mesh.make_periodic("x")              # faces that coincide after a shift in x
    mesh.set_bc("inlet", pm.Inflow((1.0, 0.0)))
    print(mesh.check())
    domain = mesh.get_domain(viscosity=1e-3, dtype=torch.float64, device="cuda")

Every face of the mesh must end up connected, periodic or in a patch with a
condition; :meth:`Mesh.get_domain` lists the faces and patches that are not.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import torch

from phipict import _C
from phipict.bc import Face

from .block import MeshBlock, tangential_axes
from .boundary import BoundarySpec, FacePatches, Outflow, Patch, apply_boundary
from .connect import Connection, Orientation, find_connections, match_faces
from .grading import GradingLike
from .quality import (
    MeshReport,
    aspect_ratio,
    cell_volumes,
    non_orthogonality,
    scaled_jacobian,
)

__all__ = ["Mesh"]

_AXES = "xyz"


class Mesh:
    """Structured multi-block mesh.

    Parameters
    ----------
    blocks : iterable of MeshBlock, optional
        Blocks of the mesh. Blocks with the same name are renamed.
    auto_connect : bool, optional
        Whether to connect the coinciding free faces of the blocks. Default is
        True.
    rel_tol : float, optional
        Tolerance of the face matching, relative to the smallest vertex spacing
        of a face. Default is 1e-4.
    """

    def __init__(
        self,
        blocks: Iterable[MeshBlock] = (),
        *,
        auto_connect: bool = True,
        rel_tol: float = 1e-4,
    ) -> None:
        self._blocks: list[MeshBlock] = []
        self._connections: list[Connection] = []
        self._patches: dict[str, Patch] = {}
        self.rel_tol = rel_tol
        for block in blocks:
            self.add(block)
        if auto_connect:
            self.auto_connect()

    # -----------------------------------------------------------------------
    # Contents
    # -----------------------------------------------------------------------

    @property
    def blocks(self) -> tuple[MeshBlock, ...]:
        """The blocks, in the order of the solver blocks."""
        return tuple(self._blocks)

    @property
    def connections(self) -> tuple[Connection, ...]:
        """The connections between block faces, periodic ones included."""
        return tuple(self._connections)

    @property
    def patches(self) -> tuple[Patch, ...]:
        """The patches, one per name."""
        return tuple(self._patches.values())

    @property
    def ndims(self) -> int:
        """Number of spatial dimensions."""
        if not self._blocks:
            raise ValueError("The mesh has no blocks.")
        return self._blocks[0].ndims

    @property
    def n_cells(self) -> int:
        """Total number of cells."""
        return sum(b.n_cells for b in self._blocks)

    def __len__(self) -> int:
        return len(self._blocks)

    def __iter__(self) -> Iterator[MeshBlock]:
        return iter(self._blocks)

    def __repr__(self) -> str:
        if not self._blocks:
            return "Mesh(empty)"
        patches = ", ".join(p.name for p in self.patches)
        return (
            f"Mesh({self.ndims}D, {len(self)} blocks, {self.n_cells} cells, "
            f"{len(self._connections)} connections, patches=[{patches}])"
        )

    def index(self, name: str) -> int:
        """Index of the block called ``name``."""
        for i, b in enumerate(self._blocks):
            if b.name == name:
                return i
        raise KeyError(_unknown("block", name, [b.name for b in self._blocks]))

    def block(self, name: str) -> MeshBlock:
        """The block called ``name``."""
        return self._blocks[self.index(name)]

    def patch(self, name: str) -> Patch:
        """The patch called ``name``."""
        if name not in self._patches:
            raise KeyError(_unknown("patch", name, list(self._patches)))
        return self._patches[name]

    # -----------------------------------------------------------------------
    # Building
    # -----------------------------------------------------------------------

    def add(self, block: MeshBlock) -> int:
        """Add a block (without connecting it) and return its index.

        The block's patches are unified with the mesh's patches of the same name.
        """
        if self._blocks and block.ndims != self.ndims:
            raise ValueError(
                f"Cannot add a {block.ndims}D block to a {self.ndims}D mesh."
            )
        faces = [
            self._register(p) if p is not None else None
            for p in block.patches.as_list()
        ]
        name = block.name
        names = {b.name for b in self._blocks}
        k = 1
        while name in names:
            name = f"{block.name}_{k}"
            k += 1
        self._blocks.append(
            MeshBlock(block.coords, name, FacePatches.from_faces(faces))
        )
        return len(self._blocks) - 1

    def _register(self, patch: Patch) -> Patch:
        known = self._patches.get(patch.name)
        if known is None:
            self._patches[patch.name] = patch
            return patch
        if known is patch:
            return patch
        if patch.bc is not None and known.bc is not None and patch.bc is not known.bc:
            raise ValueError(
                f"Two different patches are called '{patch.name}'; use one Patch "
                "object per name."
            )
        if known.bc is None:
            known.bc = patch.bc
        return known

    def set_bc(self, patch: str | Patch, spec: BoundarySpec) -> None:
        """Set the condition of a patch.

        Parameters
        ----------
        patch : str or Patch
            The patch or its name.
        spec : BoundarySpec
            The condition, e.g. ``pm.Inflow((1.0, 0.0))``.
        """
        name = patch.name if isinstance(patch, Patch) else patch
        self.patch(name).bc = spec

    def assign_patch(self, faces: Iterable[tuple[int, Face]], patch: Patch) -> None:
        """Put block faces into a patch.

        Parameters
        ----------
        faces : iterable of tuple of int and Face
            ``(block, face)`` pairs, e.g. from :meth:`free_faces`.
        patch : Patch
            The patch.
        """
        patch = self._register(patch)
        connected = {(c.block_a, c.face_a) for c in self._connections}
        connected |= {(c.block_b, c.face_b) for c in self._connections}
        for b, f in faces:
            face = Face(f)
            if (b, face) in connected:
                raise ValueError(
                    f"Face {face.name} of block '{self._blocks[b].name}' is connected."
                )
            block = self._blocks[b]
            block.patches = block.patches.set(face, patch)

    def free_faces(self) -> list[tuple[int, Face]]:
        """Faces without a patch or connection, as ``(block, face)``."""
        connected = {(c.block_a, c.face_a) for c in self._connections}
        connected |= {(c.block_b, c.face_b) for c in self._connections}
        return [
            (b, Face(f))
            for b, block in enumerate(self._blocks)
            for f in range(2 * block.ndims)
            if block.patches[f] is None and (b, Face(f)) not in connected
        ]

    def auto_connect(self, rel_tol: float | None = None) -> list[Connection]:
        """Connect the free faces whose vertices coincide.

        Parameters
        ----------
        rel_tol : float or None, optional
            Matching tolerance relative to the smallest vertex spacing of a face.
            Default is the mesh's ``rel_tol``.

        Returns
        -------
        list of Connection
            The new connections.

        Raises
        ------
        NonConformingInterfaceError
            If two faces touch at their corners but their vertices differ.
        """
        found = find_connections(
            self._blocks,
            self.free_faces(),
            self.rel_tol if rel_tol is None else rel_tol,
        )
        self._connections += found
        return found

    def make_periodic(
        self,
        axis: int | str | None = None,
        *,
        translation: Sequence[float] | None = None,
        rel_tol: float | None = None,
    ) -> list[Connection]:
        """Connect free faces that coincide after a shift.

        Parameters
        ----------
        axis : int or str or None, optional
            Make the mesh periodic along this axis, over its full extent.
        translation : sequence of float or None, optional
            Or: connect faces that coincide after this shift.
        rel_tol : float or None, optional
            Matching tolerance, see :meth:`auto_connect`.

        Returns
        -------
        list of Connection
            The new periodic connections.

        Raises
        ------
        ValueError
            If no faces match.
        """
        if (axis is None) == (translation is None):
            raise ValueError("Give exactly one of `axis` and `translation`.")
        if translation is None:
            a = _AXES.index(axis) if isinstance(axis, str) else int(axis)  # type: ignore[arg-type]
            lo, hi = self.bounds()
            shift = [0.0] * self.ndims
            shift[a] = float(hi[a] - lo[a])
            translation = shift
        found = find_connections(
            self._blocks,
            self.free_faces(),
            self.rel_tol if rel_tol is None else rel_tol,
            translation=translation,
        )
        if not found:
            raise ValueError(
                f"No free faces coincide after the shift {list(translation)}."
            )
        self._connections += found
        return found

    def connect(
        self,
        block_a: int | str,
        face_a: Face | int,
        block_b: int | str,
        face_b: Face | int,
        *,
        orientation: Orientation | None = None,
        translation: Sequence[float] | None = None,
    ) -> Connection:
        """Connect two faces explicitly.

        The orientation is found from the geometry unless given, so this is only
        needed for faces that do not coincide (e.g. periodic ones with a
        rotation).

        Parameters
        ----------
        block_a, block_b : int or str
            Blocks, by index or name.
        face_a, face_b : Face
            The faces.
        orientation : Orientation or None, optional
            Relative orientation of the faces. Default is found from the geometry.
        translation : sequence of float or None, optional
            Shift from face A to face B for periodic faces. Default is None.

        Returns
        -------
        Connection
            The connection.
        """
        a = self.index(block_a) if isinstance(block_a, str) else block_a
        b = self.index(block_b) if isinstance(block_b, str) else block_b
        fa, fb = Face(face_a), Face(face_b)
        if orientation is None:
            ca = self._blocks[a].face_coords(fa)
            cb = self._blocks[b].face_coords(fb)
            shift = (
                None
                if translation is None
                else torch.as_tensor(translation, dtype=torch.float64)
            )
            tol = self.rel_tol * _min_spacing([ca, cb])
            orientation = match_faces(ca, cb, tol, shift)
            if orientation is None:
                raise ValueError(
                    f"Face {fa.name} of block '{self._blocks[a].name}' and face "
                    f"{fb.name} of block '{self._blocks[b].name}' do not coincide; "
                    "pass `orientation` to connect them anyway."
                )
        conn = Connection(
            a,
            fa,
            b,
            fb,
            orientation,
            None if translation is None else tuple(float(x) for x in translation),
        )
        taken = {(c.block_a, c.face_a) for c in self._connections} | {
            (c.block_b, c.face_b) for c in self._connections
        }
        for key in ((a, fa), (b, fb)):
            if key in taken:
                raise ValueError(
                    f"Face {key[1].name} of block {key[0]} is already connected."
                )
        self._connections.append(conn)
        return conn

    # -----------------------------------------------------------------------
    # Geometry
    # -----------------------------------------------------------------------

    def bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Lower and upper corner of the axis-aligned bounding box."""
        pts = torch.cat([b.coords.reshape(b.ndims, -1) for b in self._blocks], 1)
        return pts.amin(1), pts.amax(1)

    def transform(self, fn: Any) -> Mesh:
        """A copy with every block mapped by ``fn(block) -> block``.

        The connections are found again, so ``fn`` may also reorder the vertices
        of a block, e.g. ``lambda b: b.rotate(30)``.
        """
        return self._rebuilt([fn(b) for b in self._blocks])

    def _rebuilt(self, blocks: list[MeshBlock], extra_dims: int = 0) -> Mesh:
        out = Mesh(auto_connect=False, rel_tol=self.rel_tol)
        for b in blocks:
            out.add(b)
        for c in self._connections:
            if c.translation is None:
                continue
            shift = list(c.translation) + [0.0] * extra_dims
            ca = out._blocks[c.block_a].face_coords(c.face_a)
            cb = out._blocks[c.block_b].face_coords(c.face_b)
            tol = out.rel_tol * _min_spacing([ca, cb])
            o = match_faces(ca, cb, tol, torch.tensor(shift, dtype=torch.float64))
            if o is None:
                raise ValueError(
                    f"The periodic faces {c.face_a.name} of block "
                    f"'{blocks[c.block_a].name}' and {c.face_b.name} of block "
                    f"'{blocks[c.block_b].name}' no longer match."
                )
            out._connections.append(
                Connection(c.block_a, c.face_a, c.block_b, c.face_b, o, tuple(shift))
            )
        out.auto_connect()
        return out

    def extrude(
        self,
        z_range: tuple[float, float],
        cells: int,
        grading: GradingLike = None,
        *,
        periodic: bool = True,
        z_patches: tuple[Patch | None, Patch | None] = (None, None),
    ) -> Mesh:
        """A 3D mesh from this 2D mesh, extruded along z.

        Parameters
        ----------
        z_range : tuple of float
            Start and end in z.
        cells : int
            Number of cells along z.
        grading : Grading or float or None, optional
            Distribution along z. Default is uniform.
        periodic : bool, optional
            Whether the mesh is periodic in z. Default is True.
        z_patches : tuple of Patch or None, optional
            Patches of the ``-z``/``+z`` faces if not periodic.

        Returns
        -------
        Mesh
            The 3D mesh, with the 2D connections and patches carried over.
        """
        if self.ndims != 2:
            raise ValueError("Only 2D meshes can be extruded.")
        if periodic and any(p is not None for p in z_patches):
            raise ValueError("A mesh periodic in z has no z patches.")
        if not periodic and any(p is None for p in z_patches):
            raise ValueError("Give both z patches of a mesh that is not periodic in z.")
        blocks = [b.extrude(z_range, cells, grading, z_patches) for b in self._blocks]
        out = self._rebuilt(blocks, extra_dims=1)
        if periodic:
            out.make_periodic("z")
        return out

    # -----------------------------------------------------------------------
    # Patches
    # -----------------------------------------------------------------------

    def faces(self, patch: str | Patch) -> list[tuple[int, Face]]:
        """The ``(block, face)`` pairs of a patch."""
        p = self.patch(patch.name if isinstance(patch, Patch) else patch)
        return [
            (i, f)
            for i, b in enumerate(self._blocks)
            for f, q in b.patches.items()
            if q is p
        ]

    def boundaries(self, domain: _C.Domain, patch: str | Patch) -> list[_C.Boundary]:
        """The solver boundaries of a patch in a domain made by :meth:`get_domain`.

        E.g. the outflow boundaries to update during the simulation.
        """
        blocks = domain.getBlocks()
        return [blocks[b].getBoundary(int(f)) for b, f in self.faces(patch)]

    def outflow_hook(
        self,
        domain: _C.Domain,
        velocity: float | Sequence[float] | torch.Tensor,
        tol: float | None = None,
    ) -> Any:
        """A callback that advects the :class:`~phipict.meshing.Outflow` faces.

        Register it before each step, ``Hooks().append(Hook.PRE, fn)``. It calls
        :func:`~phipict.core.piso_simulation.update_advective_boundaries` for the
        outflow boundaries of a domain made by :meth:`get_domain`.

        Parameters
        ----------
        domain : phipict.Domain
            The domain.
        velocity : float or sequence of float or torch.Tensor
            Velocity the outflow is advected with, e.g. the mean inflow velocity:
            a vector, or a number for the x component.
        tol : float or None, optional
            Tolerance of the flux balance. Default is the solver's.

        Returns
        -------
        Callable
            The callback.
        """
        from phipict.core.piso_simulation import update_advective_boundaries

        bounds = [
            b
            for p in self.patches
            if isinstance(p.bc, Outflow)
            for b in self.boundaries(domain, p)
        ]
        if not bounds:
            raise ValueError("The mesh has no Outflow patch.")
        dims = self.ndims
        if isinstance(velocity, int | float):
            velocity = [float(velocity)] + [0.0] * (dims - 1)
        vel = torch.as_tensor(velocity, dtype=domain.getDtype()).reshape(1, dims)
        vel = vel.to(domain.getDevice())

        def callback(domain: _C.Domain, time_step: Any, **_: Any) -> None:
            dt = (
                time_step.to(domain.getDevice())
                if isinstance(time_step, torch.Tensor)
                else time_step
            )
            update_advective_boundaries(domain, bounds, vel, dt, tol=tol)

        return callback

    # -----------------------------------------------------------------------
    # Checks
    # -----------------------------------------------------------------------

    def check(self) -> MeshReport:
        """Check the cells, faces and patches of the mesh.

        Returns
        -------
        MeshReport
            Quality measures; errors for inverted cells, free faces and patches
            without a condition.
        """
        vols, sj, ar, no = [], [], [], []
        errors: list[str] = []
        warnings: list[str] = []
        for b in self._blocks:
            v = cell_volumes(b.coords)
            s = scaled_jacobian(b.coords)
            vols.append(float(v.min()))
            sj.append(float(s.min()))
            ar.append(float(aspect_ratio(b.coords).max()))
            no.append(float(non_orthogonality(b.coords).max()))
            n_bad = int((s <= 0).sum())
            if n_bad:
                hint = (
                    " (left-handed block: flip one axis)" if bool((v < 0).all()) else ""
                )
                errors.append(
                    f"block '{b.name}' has {n_bad} inverted or degenerate cells{hint}"
                )
        free = self.free_faces()
        if free:
            listed = ", ".join(f"{self._blocks[b].name}:{f.name}" for b, f in free)
            errors.append(f"faces without patch or connection: {listed}")
        no_bc = [p.name for p in self.patches if p.bc is None and self.faces(p)]
        if no_bc:
            errors.append(
                f"patches without a condition: {', '.join(no_bc)} (use Mesh.set_bc)"
            )
        if max(no, default=0.0) > 70.0:
            warnings.append(f"high non-orthogonality {max(no):.1f} deg")
        if max(ar, default=1.0) > 1000.0:
            warnings.append(f"high aspect ratio {max(ar):.3g}")
        return MeshReport(
            n_blocks=len(self._blocks),
            n_cells=self.n_cells,
            min_volume=min(vols, default=0.0),
            min_scaled_jacobian=min(sj, default=1.0),
            max_aspect_ratio=max(ar, default=1.0),
            max_non_orthogonality=max(no, default=0.0),
            errors=errors,
            warnings=warnings,
        )

    # -----------------------------------------------------------------------
    # Solver
    # -----------------------------------------------------------------------

    def get_domain(
        self,
        viscosity: float | torch.Tensor,
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cuda",
        passive_scalar_channels: int = 1,
        scalar_viscosity: torch.Tensor | None = None,
        name: str = "Domain",
        prepare: bool = True,
        balance_outflow: bool = True,
    ) -> _C.Domain:
        """A solver domain with the blocks, connections and conditions of the mesh.

        Solver block ``i`` is block ``i`` of the mesh.

        Parameters
        ----------
        viscosity : float or torch.Tensor
            Kinematic viscosity.
        dtype : torch.dtype, optional
            Precision of the domain. Default is float32.
        device : str or torch.device, optional
            Device of the domain. Default is ``"cuda"``.
        passive_scalar_channels : int, optional
            Number of passive scalar channels. Default is 1.
        scalar_viscosity : torch.Tensor or None, optional
            Diffusivity of the passive scalar. Default is None.
        name : str, optional
            Name of the domain. Default is ``"Domain"``.
        prepare : bool, optional
            Whether to call ``domain.PrepareSolve()``. Default is True.
        balance_outflow : bool, optional
            Whether to scale the :class:`~phipict.meshing.Outflow` velocities so
            that the boundary fluxes balance, as the solver requires. Default is
            True.

        Returns
        -------
        phipict.Domain
            The domain.

        Raises
        ------
        ValueError
            If faces have neither patch nor connection, or patches have no
            condition.
        """
        report_errors = [e for e in self.check().errors if not e.startswith("block ")]
        if report_errors:
            raise ValueError("Cannot make a domain:\n  " + "\n  ".join(report_errors))
        device = torch.device(device)
        visc = torch.as_tensor(viscosity, dtype=dtype).reshape(-1)
        domain = _C.Domain(
            self.ndims,
            visc,
            name,
            dtype,
            device,
            passive_scalar_channels,
            scalar_viscosity,
        )
        solver_blocks = [
            domain.CreateBlock(
                vertexCoordinates=b.coords[None].to(device, dtype).contiguous(),
                name=b.name,
            )
            for b in self._blocks
        ]
        for c in self._connections:
            if c.is_periodic_axis:
                solver_blocks[c.block_a].MakePeriodic(c.face_a.axis)
            else:
                axis1, axis2 = c.axis_codes(self.ndims)
                # the binding takes unsigned codes; the second is unused in 2D
                solver_blocks[c.block_a].ConnectBlock(
                    int(c.face_a),
                    solver_blocks[c.block_b],
                    int(c.face_b),
                    axis1,
                    max(axis2, 0),
                )
        outflow = []
        for block, solver_block in zip(self._blocks, solver_blocks, strict=True):
            for face, patch in block.patches.items():
                assert patch.bc is not None
                apply_boundary(patch.bc, solver_block, face)
                if isinstance(patch.bc, Outflow):
                    outflow.append(solver_block.getBoundary(int(face)))
        if outflow and balance_outflow:
            from phipict.core.piso_simulation import balance_boundary_fluxes

            balance_boundary_fluxes(domain, outflow)
        if prepare:
            domain.PrepareSolve()
        return domain

    @classmethod
    def from_domain(cls, domain: _C.Domain) -> Mesh:
        """The mesh of a solver domain, e.g. one from :func:`~phipict.io.load_domain`.

        Connections and periodicity are read from the domain. Fixed faces get a
        patch ``"fixed"`` without a condition.
        """
        blocks = domain.getBlocks()
        mesh = cls(auto_connect=False)
        fixed = Patch("fixed", kind="fixed")
        d = domain.getSpatialDims()
        conns: list[Connection] = []
        for i, sb in enumerate(blocks):
            if not sb.hasVertexCoordinates():
                raise ValueError(f"Block '{sb.name}' has no vertex coordinates.")
            faces: list[Patch | None] = []
            for f in range(2 * d):
                bound: Any = sb.getBoundary(f)
                faces.append(fixed if bound.type == _C.BoundaryType.FIXED else None)
                if bound.type == _C.BoundaryType.CONNECTED:
                    j = next(
                        k
                        for k, ob in enumerate(blocks)
                        if ob is bound.getConnectedBlock()
                        or ob.name == bound.getConnectedBlock().name
                        and k != -1
                    )
                    axes = list(bound.axes)
                    fb = Face(axes[0])
                    if (j, int(fb)) < (i, f):
                        continue
                    tb = tangential_axes(fb, d)
                    perm = tuple(tb.index(code >> 1) for code in axes[1:d])
                    flip = tuple(bool(code & 1) for code in axes[1:d])
                    conns.append(Connection(i, Face(f), j, fb, Orientation(perm, flip)))
                elif bound.type == _C.BoundaryType.PERIODIC and f % 2 == 0:
                    conns.append(
                        Connection(
                            i,
                            Face(f),
                            i,
                            Face(f + 1),
                            Orientation(tuple(range(d - 1)), (False,) * (d - 1)),
                        )
                    )
            coords = sb.vertexCoordinates
            assert coords is not None
            mesh.add(MeshBlock(coords, sb.name, FacePatches.from_faces(faces)))
        mesh._connections = conns
        return mesh

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------

    def plot(
        self,
        ax: Any = None,
        color: str | Sequence[str] | None = None,
        linewidth: float = 0.5,
        z_index: int = 0,
    ) -> Any:
        """Plot the grid lines of the blocks with matplotlib (the ``plot`` extra).

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            Axes to draw into. Default is the current pyplot axes.
        color : str or sequence of str or None, optional
            One colour for all blocks, or colours cycled over the blocks. Default
            is the phiPICT palette, cycled if there are more blocks than colours.
        linewidth : float, optional
            Line width. Default is 0.5.
        z_index : int, optional
            Vertex layer in z that is plotted for a 3D mesh, which is assumed to
            be extruded along z. Default is 0.

        Returns
        -------
        matplotlib.axes.Axes
            The axes.
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        from phipict._palette import DEFAULT_PALETTE

        if ax is None:
            ax = plt.gca()
        if color is None:
            colors: Sequence[str] = DEFAULT_PALETTE
        elif isinstance(color, str):
            colors = [color]
        else:
            colors = color
        for i, b in enumerate(self._blocks):
            # 3D meshes are extruded along z: plot one layer of vertices
            xy = b.coords if b.ndims == 2 else b.coords[:2, z_index]
            c = xy.numpy()
            lines = [c[:, j, :].T for j in range(c.shape[1])]
            lines += [c[:, :, k].T for k in range(c.shape[2])]
            ax.add_collection(
                LineCollection(
                    lines, colors=colors[i % len(colors)], linewidths=linewidth
                )
            )
        ax.autoscale()
        ax.set_aspect("equal")
        return ax


def _min_spacing(faces: Sequence[torch.Tensor]) -> float:
    out = []
    for face in faces:
        for dim in range(1, face.dim()):
            out.append(float(torch.linalg.vector_norm(face.diff(dim=dim), dim=0).min()))
    return min(out)


def _unknown(kind: str, name: str, known: list[str]) -> str:
    close = difflib.get_close_matches(name, known, n=3)
    hint = f" Did you mean {', '.join(repr(c) for c in close)}?" if close else ""
    return f"No {kind} called {name!r}.{hint} Known: {', '.join(known) or 'none'}."
