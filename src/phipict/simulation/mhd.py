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

"""MHD simulation class extending the PISO solver with inductionless MHD.

Implements the inductionless MHD approximation with a static magnetic field.
The magnetic field is defined by the Hartmann number Ha, Reynolds number Re,
and the field direction e_b. The electric potential is solved via a Poisson
equation, and the Lorentz force is injected as a velocity source term.

Governing equations (non-dimensional):
    Current density:      j = -grad(phi) + u x e_b
    Electric potential:   laplacian(phi) = div(u x e_b)
    Lorentz force:        F_L = N * (j x e_b),  where N = Ha^2 / Re

Each FIXED face carries a :class:`phipict.PotentialBC` (INSULATING by default,
OPEN, DIRICHLET or THIN_WALL), optionally per face cell; see :mod:`phipict.bc`.
Thin conducting walls (THIN_WALL) use the following BC:
    dphi/dn|_w = Cw * nabla^2_tau(phi)|_w
implemented as a fluid-domain-only correction in the Poisson matrix
(tangential surface Laplacian terms added for wall-adjacent cells).
No augmented system with wall-face unknowns is required. The correction is
conservative (diagonal and tangential off-diagonal cancel), so it leaves the
matrix singular just like an insulating wall.
"""

import numbers
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import torch

from phipict import _C
from phipict.core import piso_diff
from phipict.core.hooks import Hook, Hooks
from phipict.core.piso_simulation import cpu_device
from phipict.logging import get_logger
from phipict.simulation.simulation import Simulation
from phipict.solvers.amg import AMGHierarchy, hierarchy_for
from phipict.solvers.tolerance import SolverTolerance

_LOG = get_logger("MHDsim")


def _epot_matrix_is_anchored(domain: _C.Domain) -> bool:
    """Return True if some boundary pins the electric potential (full-rank matrix).

    Only a Dirichlet φ=0 face cell (e.g. an outflow) anchors the system: it is the sole
    boundary treatment that leaves an unbalanced diagonal entry behind
    (``rowValues[0] -= coef`` with no matching off-diagonal). Every other face
    keeps the row sum at zero, so the constant vector stays in the nullspace and
    the matrix is singular (pure Neumann), as in the all-insulating-wall
    Shercliff duct.

    A conducting thin wall (THIN_WALL) does *not* anchor it, despite touching
    the diagonal: its Robin term is added conservatively (diagonal ``-=``,
    tangential off-diagonal ``+=``), so the row sum is unchanged. That matches
    the physics — ``dphi/dn|_w = Cw * nabla^2_tau(phi)|_w`` is invariant under
    ``phi -> phi + const``, so a thin-wall-only domain is as singular as an
    insulating one.

    A prescribed current (CURRENT) does not anchor it either: it is a Neumann face
    of the matrix, its current entering the right-hand side only. The system then
    stays singular, so the prescribed currents have to sum to zero (e.g. an
    electrode pair), or the mean projection of the right-hand side spreads the
    excess over the volume.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain whose boundaries are inspected.

    Returns
    -------
    bool
        True if any boundary cell imposes a Dirichlet condition on the potential.
    """
    for block in domain.getBlocks():
        for _idx, bound in block.getFixedBoundaries():
            if bound.hasPotentialDirichlet():
                return True
    return False


def _flat_to_block(
    block: _C.Block,
    flat: torch.Tensor,
    dims: int,
) -> torch.Tensor:
    """Extract a block's portion from a flat vector field.

    Parameters
    ----------
    block: _C.Block
        The block for which to extract the field portion.

    flat: torch.Tensor
        Flat vector field, shape [batch * dims * totalSize].

    dims: int
        Number of spatial dimensions (2 or 3).

    Returns
    -------
    torch.Tensor
        Block-shaped tensor [batch, dims, *spatial].
    """
    offset = block.globalOffset
    n_cells = block.getStrides().w
    spatial = block.velocity.shape[2:]
    batch = block.velocity.shape[0]

    # [batch][dims][totalSize] layout of the flat solve vectors
    per_env = flat.reshape(batch, dims, -1)
    return per_env[:, :, offset : offset + n_cells].reshape(batch, dims, *spatial)


def _subtract_env_mean(flat: torch.Tensor, batch: int) -> torch.Tensor:
    """Subtract the mean of every batched environment's slice of a flat vector.

    Parameters
    ----------
    flat : torch.Tensor
        Flat vector holding ``batch`` equal, consecutive environment slices.
    batch : int
        Number of environments.

    Returns
    -------
    torch.Tensor
        ``flat`` with each environment's mean removed, same shape.
    """
    per_env = flat.reshape(batch, -1)
    return (per_env - per_env.mean(dim=1, keepdim=True)).reshape(flat.shape)


def _lift_2d_to_3d(field_2d: torch.Tensor) -> torch.Tensor:
    """Lift a 2D block field [1, 2, *spatial] to 3D [1, 3, *spatial] with zeros.

    Parameters
    ----------
    field_2d : torch.Tensor
        Block field of shape ``[1, 2, *spatial]``.

    Returns
    -------
    torch.Tensor
        The field of shape ``[1, 3, *spatial]``, with a zero third component.
    """
    zeros = torch.zeros_like(field_2d[:, :1])
    return torch.cat([field_2d, zeros], dim=1)


def _validate_e_b(
    e_b: torch.Tensor,
    domain: _C.Domain,
    dims: int,
) -> tuple[int, ...] | None:
    """Check a magnetic field spec against the domain, see :func:`_prepare_e_b`.

    Parameters
    ----------
    e_b : torch.Tensor
        Magnetic field spec, of shape ``(3,)`` or ``(3, *spatial)``.
    domain : phipict._C.Domain
        Domain the field must be compatible with.
    dims : int
        Number of spatial dimensions (2 or 3).

    Raises
    ------
    ValueError
        If ``e_b`` is not a tensor, has an unsupported shape, is not
        broadcastable to the domain grid, or is spatially varying on a
        multi-block domain.

    Returns
    -------
    tuple[int, ...] | None
        The domain's spatial cell shape for a spatially varying field, None for
        a uniform one.
    """
    if not isinstance(e_b, torch.Tensor):
        raise ValueError("e_b must be a torch.Tensor.")

    if e_b.ndim == 1:
        if e_b.shape != (3,):
            raise ValueError(
                f"A uniform e_b must have shape (3,), got {tuple(e_b.shape)}."
            )
        return None

    if e_b.ndim != dims + 1 or e_b.shape[0] != 3:
        raise ValueError(
            f"A spatially varying e_b must have shape (3, *spatial) with {dims} "
            f"spatial dimensions, got {tuple(e_b.shape)}."
        )

    blocks = domain.getBlocks()
    if len(blocks) != 1:
        raise ValueError(
            "A spatially varying e_b is only supported on single-block domains, "
            f"got {len(blocks)} blocks."
        )
    spatial = tuple(blocks[0].velocity.shape[2:])

    for axis, (size, grid_size) in enumerate(zip(e_b.shape[1:], spatial, strict=True)):
        if size not in (1, grid_size):
            raise ValueError(
                f"e_b of shape {tuple(e_b.shape)} is not broadcastable to the "
                f"domain grid (3, {', '.join(str(s) for s in spatial)}): spatial "
                f"axis {axis} has size {size}, expected 1 or {grid_size}."
            )

    return spatial


def _prepare_e_b(
    e_b: torch.Tensor,
    domain: _C.Domain,
    dims: int,
) -> tuple[torch.Tensor, tuple[int, ...] | None]:
    """Bring a magnetic field spec into the layout the per-cell cross products use.

    Two forms are accepted:

    * a uniform field of shape ``(3,)`` (a direction; its magnitude multiplies
      the one folded into the Stuart number), and
    * a spatially varying field of shape ``(3, *spatial)``, where ``spatial``
      either matches the domain's cell grid (``(nz, ny, nx)`` in 3D,
      ``(ny, nx)`` in 2D) or is broadcastable to it, so e.g. ``(3, 1, 1, nx)``
      describes a field that only varies along the streamwise direction.

    The field always has 3 components, also in 2D, since the cross products are
    evaluated in 3D. A spatially varying field requires a single-block domain,
    as the block-to-grid mapping is otherwise ambiguous.

    Parameters
    ----------
    e_b : torch.Tensor
        Magnetic field spec, of shape ``(3,)`` or ``(3, *spatial)``.
    domain : phipict._C.Domain
        Domain the field is prepared for.
    dims : int
        Number of spatial dimensions (2 or 3).

    Returns
    -------
    tuple[torch.Tensor, tuple[int, ...] | None]
        The field, either ``(3,)`` for the uniform case or
        ``(*spatial, 3)`` (spatial dims kept broadcastable, i.e. size 1 where
        the input was 1) for a varying one, plus the domain's spatial cell
        shape. ``None`` in the uniform case, where it is not needed.
    """
    spatial = _validate_e_b(e_b, domain, dims)

    field = e_b.detach().cpu()
    if not field.is_floating_point():
        field = field.float()

    if spatial is None:
        return field, None

    # Move the component axis last so the cross products can broadcast against
    # per-cell vectors, and keep the singleton spatial axes unexpanded. A
    # (3, 1, 1, nx) field must not blow up to one vector per cell
    return field.movedim(0, -1).contiguous(), spatial


def _cross_with_e_b(
    v: torch.Tensor,
    e_b: torch.Tensor,
    spatial: tuple[int, ...] | None,
) -> torch.Tensor:
    """Cross a per-cell vector field ``v`` [total_size, 3] with the magnetic field.

    ``e_b`` is either uniform (shape ``(3,)``) or a spatially varying field in
    the ``(*spatial, 3)`` layout produced by :func:`_prepare_e_b`, in which case
    ``v`` is viewed on the cell grid so the two broadcast against each other.

    Parameters
    ----------
    v : torch.Tensor
        Per-cell vectors of shape ``[total_size, 3]``.
    e_b : torch.Tensor
        Magnetic field, of shape ``(3,)`` or ``(*spatial, 3)``.
    spatial : tuple of int or None
        The domain's spatial cell shape. Required for a spatially varying
        field, ignored for a uniform one.

    Returns
    -------
    torch.Tensor
        ``v x e_b``, of the same shape as ``v``.
    """
    if e_b.ndim == 1:
        return torch.linalg.cross(v, e_b.expand_as(v), dim=-1)

    assert spatial is not None
    return torch.linalg.cross(v.reshape(*spatial, 3), e_b, dim=-1).reshape(v.shape)


def _compute_u_cross_eb_flat(
    vel_flat: torch.Tensor,
    e_b: torch.Tensor,
    total_size: int,
    dims: int,
    spatial: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Compute u x e_b for a flat velocity field, returning a flat 3-component field.

    For 2D, lifts to 3D and crosses with e_b.

    Parameters
    ----------
    vel_flat : torch.Tensor
        Flat velocity field of shape ``[batch * dims * total_size]``.
    e_b : torch.Tensor
        Magnetic field in the layout produced by :func:`_prepare_e_b`.
    total_size : int
        Number of cells in the domain.
    dims : int
        Number of spatial dimensions (2 or 3).
    spatial : tuple of int or None, optional
        The domain's spatial cell shape, needed for a spatially varying field.
        Default is None.

    Returns
    -------
    torch.Tensor
        Flat 3-component field ``u x e_b`` of shape ``[batch * 3 * total_size]``.
    """
    eb = e_b.to(device=vel_flat.device, dtype=vel_flat.dtype)

    # [batch, dims, totalSize] -> per-cell vectors [batch, totalSize, 3]; 2D fields are
    # lifted to 3D with a zero z component
    v = vel_flat.reshape(-1, dims, total_size)
    if dims == 2:
        v = torch.cat([v, torch.zeros_like(v[:, :1])], dim=1)
    v = v.transpose(1, 2)
    cross_3d = (
        torch.stack([_cross_with_e_b(vb, eb, spatial) for vb in v])
        if eb.ndim > 1
        else _cross_with_e_b(v, eb, spatial)
    )
    # All 3 components: x, y (in-plane for the Poisson RHS) and z (out-of-plane current
    # source needed when B has a component in the 2D plane, e.g. the Hartmann case).
    # Layout [batch][3][totalSize]
    return cross_3d.transpose(1, 2).reshape(-1)


class MHDSimulation(Simulation):
    """MHD simulation using the inductionless approximation.

    Parameters
    ----------
    domain: _C.Domain
        The simulation domain (must be initialized via PrepareSolve).

    dt: float
        Time step size.

    stuart_number: torch.Tensor
        Stuart number N = Ha^2 / Re as a scalar tensor (learnable).

    e_b: torch.Tensor
        Magnetic field. Either a uniform direction of shape ``(3,)``, or a
        spatially varying field of shape ``(3, *spatial)``, i.e. ``(3, nz, ny, nx)``
        in 3D and ``(3, ny, nx)`` in 2D, whose spatial axes may be 1 to
        broadcast, e.g. ``(3, 1, 1, nx)`` for a field that only varies along the
        streamwise direction. It always carries 3 components, also in 2D. A spatially
        varying field requires a single-block domain.

    potential_tol: float, SolverTolerance or None
        Tolerance for the electric potential Poisson solve. A float is an
        absolute tolerance on ``||r||_2/sqrt(n)``; a
        :class:`~phipict.solvers.tolerance.SolverTolerance` specifies
        it relative to the RHS instead, which keeps its meaning across domain
        sizes. If None, falls back to the dtype default.

    potential_use_BiCG: bool
        Whether to use BiCGStab for the potential solve. Defaults to False (uses CG).

    potential_normalize: bool
        Whether to subtract the mean from the electric potential result
        (analogous to normalize_pressure_result). Defaults to True. Ignored when
        a Dirichlet φ boundary anchors the matrix, since φ is then unique.

    potential_non_ortho_flags: int
        Non-orthogonal correction flags for the epot matrix. Defaults to 0.

    potential_use_face_transform: bool
        Whether to use face transforms for the epot matrix. Defaults to False.

    potential_solve_max_iter: int
        Maximum iterations for the potential linear solver. Defaults to 5000.

    potential_solve_dtype: torch.dtype or None
        Data type to use for the electric potential linear solve. If set (e.g.
        ``torch.float64``), the Poisson matrix and RHS are up-cast to this dtype
        for the solve and the result is cast back to the domain's global dtype,
        so the rest of the simulation keeps running at the global precision.
        Unlike ``solver_double_fallback`` (which only retries in double after a
        single-precision solve fails), this solves at the requested precision
        straight away. If None (default), the solve uses the global dtype.

    exclude_potential_solve_gradients: bool
        Whether to exclude gradients from the electric potential Poisson solve
        (analogous to ``exclude_pressure_solve_gradients`` in the base
        Simulation). When True, the potential solve runs under ``torch.no_grad``
        so no gradients flow through it. Defaults to False.

    potential_reuse_result: bool
        Whether to warm start the potential solve from the previous step's phi
        instead of a zero initial guess. Defaults to True. Forced off in
        differentiable mode, where the solver backend ignores an initial guess.

    potential_use_preconditioner: bool
        Whether to solve the potential Poisson equation with AMG-preconditioned
        CG instead of unpreconditioned CG. Defaults to False.

        The unpreconditioned solve dominates the runtime of an MHD run, so this
        is the main lever on it. The hierarchy is built once (on the host, via
        pyamg) and reused for every subsequent solve, which is only sound because
        the epot matrix values are constant for a run (see
        :mod:`phipict.solvers.amg`). The first solve therefore pays a
        noticeable setup cost.

    potential_amg_options: dict or None
        Extra keyword arguments for :func:`phipict.solvers.amg.hierarchy_for`
        (and :func:`phipict.solvers.amg.build_amg_hierarchy`), e.g.
        ``{"method": "ruge_stuben", "strength": 0.5, "num_pre_smooth": 2}`` or
        ``{"reuse_interpolation": False}`` for a dedicated setup.

    potential_return_best_result: bool
        Whether a potential solve that fails to reach ``potential_tol`` should
        fall back to its best iterate (True, the default) or raise
        ``LinsolveError`` (False).

    **kwargs
        Additional keyword arguments passed to the parent Simulation class.
    """

    def __init__(
        self,
        domain: _C.Domain,
        dt: float,
        stuart_number: torch.Tensor,
        e_b: torch.Tensor,
        potential_tol: float | SolverTolerance | None = None,
        potential_use_BiCG: bool = False,
        potential_normalize: bool = True,
        potential_non_ortho_flags: int = 0,
        potential_use_face_transform: bool = False,
        potential_solve_max_iter: int = 5000,
        potential_solve_dtype: torch.dtype | None = None,
        exclude_potential_solve_gradients: bool = False,
        potential_reuse_result: bool = True,
        potential_return_best_result: bool = True,
        potential_use_preconditioner: bool = False,
        potential_amg_options: dict | None = None,
        **kwargs: Any,
    ) -> None:
        # a copy: the callbacks added below must not leak into the caller's hooks
        hooks, prep_fn = kwargs.pop("hooks", None), kwargs.pop("prep_fn", None)
        if hooks is not None and prep_fn is not None:
            raise ValueError("Pass hooks or its former name prep_fn, not both.")
        hooks = Hooks(hooks if hooks is not None else prep_fn)

        super().__init__(domain=domain, dt=dt, hooks=hooks, **kwargs)

        if not isinstance(stuart_number, torch.Tensor) or stuart_number.numel() != 1:
            raise ValueError("stuart_number must be a scalar tensor.")

        self._stuart_number = stuart_number
        self._e_b, self._e_b_spatial = _prepare_e_b(
            e_b, domain, domain.getSpatialDims()
        )
        self._potential_tol = potential_tol
        self._potential_use_BiCG = potential_use_BiCG
        self._potential_normalize = potential_normalize
        self._potential_non_ortho_flags = potential_non_ortho_flags
        self._potential_use_face_transform = potential_use_face_transform
        self._potential_solve_max_iter = potential_solve_max_iter
        self._potential_solve_dtype = potential_solve_dtype
        self._exclude_potential_solve_gradients = exclude_potential_solve_gradients
        self._potential_return_best_result = potential_return_best_result
        self._potential_use_preconditioner = potential_use_preconditioner
        self._potential_amg_options = dict(potential_amg_options or {})

        # Cached AMG hierarchy plus the matrix it was built from. The epot matrix
        # is built once and its values never change during a run, so one setup
        # serves every solve, but PrepareSolve() reallocates it and a
        # potential_solve_dtype cast produces a new object each step, so the
        # identity of the matrix actually handed to the solver is the cache key
        self._epot_amg_hierarchy: AMGHierarchy | None = None
        self._epot_amg_matrix: Any = None
        self._potential_reuse_result = potential_reuse_result

        dims = domain.getSpatialDims()
        self._dims = dims

        # Allocate epot fields, build the Laplacian matrix, and store flags so
        # that domain.Copy() + domain.PrepareSolve() auto-rebuilds epot on copies
        domain.SetupEpotOnDomain(
            self._potential_non_ortho_flags,
            self._potential_use_face_transform,
        )

        # Detect whether the Epot Poisson matrix is singular (pure Neumann). It is
        # rank deficient unless a Dirichlet φ=0 face (e.g. an outflow) anchors the
        # potential; insulating and thin-wall (THIN_WALL) faces both leave the row
        # sums at zero
        self._potential_rank_deficient = not _epot_matrix_is_anchored(domain)

        # e_b on the compute device (cached on first use)
        self._e_b_device: torch.Tensor | None = None

        # Set by set_magnetic_field() when the field changed since the last
        # potential solve; the next step re-solves before forming the force
        self._potential_stale = False

        # Init u x e_b
        total_size: int = domain.getTotalSize()
        vel: torch.Tensor = domain.velocityResult
        self._u_cross_eb_flat = _compute_u_cross_eb_flat(
            vel,
            self._e_b_for(vel),
            total_size,
            self._dims,
            self._e_b_spatial,
        )

        # Register the magnetic step to run after each complete PISO step
        self.hooks.append(Hook.POST, self._magnetic_step_callback)

        def add_lorentz_force(
            domain: _C.Domain, time_step: torch.Tensor, **kwargs: Any
        ) -> None:
            # A field changed since the last magnetic step would otherwise be
            # crossed with a current of the old field, which is not
            # divergence-free, so bring u x e_b and phi up to date first
            if self._potential_stale:
                self._magnetic_step_callback(domain, time_step)
                self._potential_stale = False

            # Face-based current density: discretely divergence-free, shape
            # [totalSize*3]
            epot_result = domain.epotResult
            assert epot_result is not None
            J_flat = piso_diff.ComputeCurrentDensityFaceBased(
                domain,
                epot_result,
                self._u_cross_eb_flat,
            )
            assert isinstance(J_flat, torch.Tensor)

            # F_L = N * (J x e_b), computed in Python for differentiability
            total_size = domain.getTotalSize()
            batch = domain.getBatchSize()
            N = self._stuart_number.to(
                device=domain.getDevice(), dtype=epot_result.dtype
            )
            eb = self._e_b_for(epot_result)
            J = J_flat.reshape(batch, 3, total_size).transpose(
                1, 2
            )  # [B, totalSize, 3]
            if eb.ndim > 1:
                F_L = N * torch.stack(
                    [_cross_with_e_b(Jb, eb, self._e_b_spatial) for Jb in J]
                )
            else:
                F_L = N * _cross_with_e_b(J, eb, self._e_b_spatial)  # [B, totalSize, 3]

            # Flatten as [B * dims * totalSize] to match velocityResult layout
            F_L_flat = F_L[..., : self._dims].transpose(1, 2).reshape(-1).contiguous()

            for block in domain.getBlocks():
                F_L_block = _flat_to_block(block, F_L_flat, self._dims)

                # Add to existing velocity source if present
                if (S := block.velocitySource) is not None:
                    if S.ndim == 2:
                        S = S.view(*S.shape[:2], *[1] * (F_L_block.ndim - 2))
                    F_L_block = F_L_block + S

                block.setVelocitySource(F_L_block)
            domain.UpdateDomainData()

        self.hooks.append(Hook.PRE_VELOCITY_SETUP, add_lorentz_force)

    def detach(self) -> None:
        """Detach the cross-step state kept outside the ``Domain``."""
        if self._u_cross_eb_flat is not None:
            self._u_cross_eb_flat = self._u_cross_eb_flat.detach()
        # A field set through set_magnetic_field() may carry the action's graph
        self._e_b = self._e_b.detach()
        self._e_b_device = None

    @property
    def magnetic_field(self) -> torch.Tensor:
        """The magnetic field in the internal layout, see :func:`_prepare_e_b`.

        Assigning restores a value previously read from here (the BPTT
        checkpoint carry) and, unlike :meth:`set_magnetic_field`, never
        triggers a potential re-solve.

        Returns
        -------
        torch.Tensor
            The field, of shape ``(3,)`` if uniform or ``(*spatial, 3)`` if
            spatially varying.
        """
        return self._e_b

    @magnetic_field.setter
    def magnetic_field(self, e_b: torch.Tensor) -> None:
        """Restore a magnetic field previously read from :attr:`magnetic_field`.

        Parameters
        ----------
        e_b : torch.Tensor
            Field in the internal layout, see :func:`_prepare_e_b`.
        """
        # The restored field comes with the phi and u x e_b it was solved with
        self._e_b = e_b
        self._e_b_device = None
        self._potential_stale = False

    def set_magnetic_field(self, e_b: torch.Tensor) -> None:
        """Replace the magnetic field between steps, keeping its autograd graph.

        Parameters
        ----------
        e_b: torch.Tensor
            The new magnetic field, ``(3,)`` or ``(3, *spatial)``.
        """
        domain = self.domain
        assert domain is not None

        spatial = _validate_e_b(e_b, domain, self._dims)
        if (spatial is None) != (self._e_b_spatial is None):
            raise ValueError(
                "set_magnetic_field cannot switch between a uniform and a "
                "spatially varying field; construct the simulation with the "
                "kind of field it will be driven with."
            )

        field = e_b if spatial is None else e_b.movedim(0, -1)
        field = field.to(device=domain.getDevice(), dtype=domain.getDtype())

        current = self._e_b.detach().to(device=field.device, dtype=field.dtype)
        if current.shape != field.shape or not torch.equal(current, field.detach()):
            self._potential_stale = True

        self._e_b = field
        self._e_b_device = None

    def _get_epot_amg_hierarchy(self, epot_mat: _C.CSRmatrix) -> AMGHierarchy:
        """Return the AMG hierarchy for ``epot_mat``, building it on first use.

        Parameters
        ----------
        epot_mat : phipict._C.CSRmatrix
            System matrix of the electric potential solve.

        Returns
        -------
        AMGHierarchy
            The cached hierarchy, rebuilt if ``epot_mat`` is a different object
            than the one it was built for.
        """
        if self._epot_amg_matrix is epot_mat and self._epot_amg_hierarchy is not None:
            return self._epot_amg_hierarchy

        if self._epot_amg_hierarchy is not None:
            _LOG.debug("Epot matrix changed; rebuilding the AMG hierarchy.")

        options = dict(self._potential_amg_options)
        options.setdefault("project_constant", self._potential_rank_deficient)

        _LOG.debug(
            "Setting up the AMG hierarchy for the electric potential solve "
            "(%d unknowns). The interpolation is shared with every operator of "
            "the same sparsity pattern (e.g. the pressure matrix), so a host-side "
            "setup happens at most once per grid.",
            epot_mat.getRows(),
        )
        self._epot_amg_hierarchy = hierarchy_for(epot_mat, **options)
        self._epot_amg_matrix = epot_mat
        return self._epot_amg_hierarchy

    @property
    def potential_tol(self) -> float | SolverTolerance | None:
        """Tolerance of the electric potential Poisson solve.

        Returns
        -------
        float or SolverTolerance or None
            The configured tolerance, or None to use the solver default.
        """
        return self._potential_tol

    @potential_tol.setter
    def potential_tol(self, potential_tol: float | SolverTolerance | None) -> None:
        """Set the tolerance of the electric potential Poisson solve.

        Parameters
        ----------
        potential_tol : float or SolverTolerance or None
            Positive absolute tolerance, a relative :class:`SolverTolerance`,
            or None to use the solver default.

        Raises
        ------
        TypeError
            If ``potential_tol`` is of none of the accepted types.
        ValueError
            If a numeric tolerance is not positive.
        """
        if potential_tol is not None and not isinstance(potential_tol, SolverTolerance):
            if not isinstance(potential_tol, numbers.Real):
                raise TypeError("potential_tol must be float, SolverTolerance or None.")
            if not potential_tol > 0:
                raise ValueError("potential_tol must be positive.")
        self._potential_tol = potential_tol

    @property
    def potential_solve_max_iter(self) -> int:
        """Iteration limit of the electric potential Poisson solve.

        Returns
        -------
        int
            The configured maximum number of solver iterations.
        """
        return self._potential_solve_max_iter

    @potential_solve_max_iter.setter
    def potential_solve_max_iter(self, max_iter: int) -> None:
        """Set the iteration limit of the electric potential Poisson solve.

        Parameters
        ----------
        max_iter : int
            Positive maximum number of solver iterations.

        Raises
        ------
        ValueError
            If ``max_iter`` is not a positive integer.
        """
        if not isinstance(max_iter, numbers.Integral) or max_iter < 1:
            raise ValueError("potential_solve_max_iter must be a positive integer.")
        self._potential_solve_max_iter = int(max_iter)

    def _e_b_for(self, ref: torch.Tensor) -> torch.Tensor:
        """Return e_b on ``ref``'s device and dtype, cached across steps.

        A spatially varying field is large enough that re-uploading it on every
        substep would be wasteful, so the converted copy is kept around and only
        rebuilt when the device or dtype changes.

        Parameters
        ----------
        ref : torch.Tensor
            Tensor whose device and dtype the field is matched to.

        Returns
        -------
        torch.Tensor
            The magnetic field on ``ref``'s device and in ``ref``'s dtype.
        """
        cached = self._e_b_device
        if cached is None or cached.device != ref.device or cached.dtype != ref.dtype:
            cached = self._e_b.to(device=ref.device, dtype=ref.dtype)
            self._e_b_device = cached
        return cached

    def _epot_initial_guess(
        self,
        domain: _C.Domain,
        epot_rhs: torch.Tensor,
    ) -> torch.Tensor | None:
        """Build an initial guess for the potential solve from the previous phi.

        Returns None whenever the previous result cannot be used safely, in
        which case the solver falls back to a zero initial guess.

        The guess is cloned because the solver writes its iterate into ``x``
        in place. When the matrix is singular it is also mean-centred to match
        the mean-centred RHS: the constant vector is then in the nullspace, so
        keeping the initial guess in the zero-mean subspace stops the constant
        mode from being carried across time steps. With an anchored (full-rank)
        matrix phi is uniquely determined and the previous result is reused as-is.

        Offered on the differentiable path too: guesses are detached, so this
        changes how fast the solve converges, not what it converges to.

        Parameters
        ----------
        domain : phipict._C.Domain
            Domain holding the previous potential result.
        epot_rhs : torch.Tensor
            Right-hand side of the current potential solve, used to check that
            the previous result is still compatible.

        Returns
        -------
        torch.Tensor or None
            A detached, cloned initial guess, or None to solve from zero.
        """
        if not self._potential_reuse_result:
            return None

        prev = domain.epotResult
        if prev is None:
            return None

        # Guard against a domain that changed shape underneath us (e.g. after a
        # reset or a resize) and against a poisoned previous solve
        if prev.numel() != epot_rhs.numel():
            return None
        if not torch.isfinite(prev).all():
            _LOG.warning(
                "Previous epot result is not finite, falling back to a zero "
                "initial guess for the potential solve."
            )
            return None

        x = (
            prev.detach()
            .reshape(epot_rhs.shape)
            .to(device=epot_rhs.device, dtype=epot_rhs.dtype)
        )
        # .to() is a no-op returning self when device+dtype already match, so
        # clone unconditionally to keep the domain's tensor un-mutated
        x = x.clone()
        if self._potential_rank_deficient:
            x = _subtract_env_mean(x, domain.getBatchSize())
        return x

    def _env_state_extras(self) -> list[Callable[[], torch.Tensor | None]]:
        """Getters of per-environment state outside the domain.

        Adds ``u x e_b`` of the last magnetic step, which the Lorentz force of the
        next substep uses.

        Returns
        -------
        list of Callable[[], torch.Tensor or None]
            Getters of flat tensors holding one slice per environment.
        """
        return [*super()._env_state_extras(), lambda: self._u_cross_eb_flat]

    def _magnetic_step_callback(
        self,
        domain: _C.Domain,
        time_step: torch.Tensor,
        max_iter: int | None = None,
        **kwargs: Any,
    ) -> bool:
        """Perform the magnetic step after each PISO substep.

        Steps:
        1. Compute u x e_b from the divergence-free velocity.
        2. Solve ∇²φ = ∇·(u x e_b) for the electric potential.
        3. Copy φ to blocks.

        ``max_iter`` overrides :attr:`potential_solve_max_iter` for this solve
        only.

        Parameters
        ----------
        domain : phipict._C.Domain
            Domain the substep was taken on.
        time_step : torch.Tensor
            Time step size of the substep.
        max_iter : int or None, optional
            Iteration limit for this solve, overriding
            :attr:`potential_solve_max_iter`. Default is None.
        **kwargs : Any
            Ignored; accepted so the callback matches the PISO callback
            signature.

        Returns
        -------
        bool
            Whether the potential solve converged.
        """
        total_size: int = domain.getTotalSize()
        vel: torch.Tensor = domain.velocityResult

        u_cross_eb = _compute_u_cross_eb_flat(
            vel,
            self._e_b_for(vel),
            total_size,
            self._dims,
            self._e_b_spatial,
        )
        self._u_cross_eb_flat = u_cross_eb  # store for add_lorentz_force

        batch = domain.getBatchSize()
        u_cross_eb_inplane = (
            self._u_cross_eb_flat.reshape(batch, 3, total_size)[:, : self._dims]
            .reshape(-1)
            .contiguous()
        )
        epot_rhs = piso_diff.ComputeEpotRHS(domain, u_cross_eb_inplane)
        assert isinstance(epot_rhs, torch.Tensor)

        domain.setEpotRHS(epot_rhs)
        domain.UpdateDomainData()

        # Optionally solve the Poisson system at a higher precision than the
        # global dtype: up-cast the matrix + RHS, solve, then cast back
        assert isinstance(domain.epotRHS, torch.Tensor)
        assert isinstance(domain.Epot, _C.CSRmatrix)

        global_dtype = domain.epotRHS.dtype
        epot_mat = domain.Epot
        epot_rhs = domain.epotRHS
        if (
            self._potential_solve_dtype is not None
            and self._potential_solve_dtype != global_dtype
        ):
            epot_mat = epot_mat.toType(self._potential_solve_dtype)
            epot_rhs = epot_rhs.to(self._potential_solve_dtype)

        # Project the RHS onto the compatible subspace of the singular system (per
        # batched environment)
        if self._potential_rank_deficient:
            epot_rhs = _subtract_env_mean(epot_rhs, batch)

        x = self._epot_initial_guess(domain, epot_rhs)

        with (
            torch.no_grad()
            if self._exclude_potential_solve_gradients
            else nullcontext()
        ):
            phi, solve_ok = self.linear_solve(
                epot_mat,
                epot_rhs,
                x=x,  # pyright: ignore[reportArgumentType]
                use_BiCG=self._potential_use_BiCG,
                tol=self._potential_tol,
                matrix_rank_deficient=self._potential_rank_deficient,
                # Same nullspace, same reason as the pressure solve: the forward
                # RHS is mean-projected above, the adjoint RHS is an incoming
                # gradient and has to be projected in backward
                adjoint_rank_deficient=self._potential_rank_deficient,
                return_best_result=self._potential_return_best_result,
                max_iter=(
                    self._potential_solve_max_iter if max_iter is None else max_iter
                ),
                amg_hierarchy=(
                    self._get_epot_amg_hierarchy(epot_mat)
                    if self._potential_use_preconditioner
                    else None
                ),
                tag="epot",
            )
        assert isinstance(phi, torch.Tensor)

        # Cast the result back to the global dtype before storing it
        if phi.dtype != global_dtype:
            phi = phi.to(global_dtype)

        # Only meaningful while the constant mode is free: with an anchored
        # matrix φ is uniquely determined and shifting it would break the
        # Dirichlet face it is anchored against
        if self._potential_normalize and self._potential_rank_deficient:
            phi = _subtract_env_mean(phi, batch)

        domain.setEpotResult(phi)
        domain.UpdateDomainData()

        piso_diff.CopyEpotResultToBlocks(domain)

        return bool(solve_ok)

    def make_current_divergence_free(self, max_iter: int = 1000) -> bool:
        """Solve the potential for the current velocity so that div(j) = 0.

        The MHD counterpart of
        :meth:`~phipict.core.piso_simulation.PisoSimulation.make_divergence_free`:
        where that one runs pressure corrections until the velocity is
        divergence free, this one recomputes ``u x e_b`` from the velocity
        currently stored in the blocks and solves the Poisson equation
        ``laplacian(phi) = div(u x e_b)`` once, which is exactly the condition
        ``div(j) = 0`` for ``j = -grad(phi) + u x e_b``. The velocity is left
        untouched.

        Unlike ``make_divergence_free`` there is no ``iterations`` argument:
        with the velocity held fixed the Poisson problem is linear and does not
        change between repeats, so a second solve would reproduce the first.

        Parameters
        ----------
        max_iter : int, optional
            Maximum iterations for the potential solve, overriding
            :attr:`potential_solve_max_iter` for this call. Default is 1000.

        Returns
        -------
        bool
            Whether the potential solve succeeded.
        """
        self._check_domain()
        domain = self.domain

        # Pull the (possibly user-set) block velocities into domain.velocityResult,
        # which is what the magnetic step reads
        backend = piso_diff if self.differentiable else _C
        backend.CopyVelocityResultFromBlocks(domain)
        domain.UpdateDomainData()

        # The magnetic step does not use the time step, so any valid value works
        time_step = torch.tensor([1], device=cpu_device, dtype=domain.A.dtype)
        solve_ok = self._magnetic_step_callback(domain, time_step, max_iter=max_iter)

        # phi now matches the current field and velocity
        self._potential_stale = False
        return solve_ok

    def get_current_density(
        self,
        block: _C.Block,
    ) -> torch.Tensor:
        """Compute the current density j = -grad(phi) + u x e_b for a block.

        Parameters
        ----------
        block : _C.Block
            The block for which to compute the current density.

        Returns
        -------
        torch.Tensor
            Current density field, shape [1, dims, *spatial] for 3D
            or [1, 3, *spatial] for 2D (lifted to 3 components).
        """
        domain = self.domain

        assert domain is not None
        assert isinstance(domain.epotResult, torch.Tensor)

        # grad(phi): flat vector field [dims * totalSize]
        grad_phi_flat = _C.ComputeFieldGradient(domain, domain.epotResult)

        # Extract block portion of grad(phi)
        grad_phi = _flat_to_block(block, grad_phi_flat, self._dims)

        # Extract block portion of u x e_b
        if self._u_cross_eb_flat is None:
            raise RuntimeError(
                "No u x e_b available. Run at least one simulation step first."
            )

        if self._dims == 2:
            # u_cross_eb_flat has 3 components for 2D; extract all 3 per block
            u_cross_eb = _flat_to_block(block, self._u_cross_eb_flat, 3)
            # Lift grad(phi) from 2 to 3 components for consistent subtraction
            grad_phi = _lift_2d_to_3d(grad_phi)
        else:
            u_cross_eb = _flat_to_block(block, self._u_cross_eb_flat, self._dims)

        return -grad_phi + u_cross_eb
