# Original work Copyright 2025 Aleksandra Franz, Nils Thuerey
# Modified work Copyright 2026 Jannis Becktepe
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
#
# Modifications:
# AMG solve path, pressure warm start, relative solver tolerances,
# rank-deficient pressure handling; removed StopHandler, VTK output and
# dead debug code.
# Moved into the phipict package, formatted, linted and typed.

"""PISO fluid simulation on multi-block domains."""

import numbers
import os
from pathlib import Path
import warnings
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from types import ModuleType
from typing import Any, Literal, cast

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from phipict import _C
from phipict.batching import EnvStateSnapshot
from phipict.core import piso_diff
from phipict.core.hooks import Hook, Hooks
from phipict.io.domain_io import save_domain
from phipict.io.output import (
    ResamplingShape,
    ntonp,
    plot_grids,
    save_domain_images,
)
from phipict.logging import get_logger
from phipict.solvers import stats as solver_stats
from phipict.solvers.amg import (
    AMGHierarchy,
    BatchedAMGHierarchy,
    amg_pcg_solve,
    batched_hierarchy_for,
    galerkin_refresh,
    hierarchy_for,
    is_pyamg_available,
    values_fingerprint,
)
from phipict.solvers.tolerance import (
    PRECISION_FLOOR_FACTOR,
    SolverTolerance,
    resolve_tolerance,
)
from phipict.utils.profiling import SAMPLE

cpu_device = torch.device("cpu")
_LOG = get_logger("PISOsim")


def subtract_env_mean(flat: torch.Tensor, batch: int) -> torch.Tensor:
    """Subtract the mean of every batched environment's slice of a flat vector.

    The flat solve vectors of a batched domain hold ``batch`` equal, consecutive
    environment slices; with ``batch == 1`` this is ``flat - flat.mean()``.

    Parameters
    ----------
    flat : torch.Tensor
        Flat vector.
    batch : int
        Number of environments.

    Returns
    -------
    torch.Tensor
        ``flat`` with each environment's mean removed.
    """
    if batch == 1:
        return flat - torch.mean(flat)
    per_env = flat.reshape(batch, -1)
    return (per_env - per_env.mean(dim=1, keepdim=True)).reshape(flat.shape)


def _split_channels(
    flat: torch.Tensor, batch: int, channels: int
) -> list[torch.Tensor]:
    """Split a flat ``[batch][channels][cells]`` vector into one flat vector per channel.

    Parameters
    ----------
    flat : torch.Tensor
        Flat vector of all environments and channels.
    batch : int
        Number of environments.
    channels : int
        Number of channels.

    Returns
    -------
    list of torch.Tensor
        ``channels`` flat ``[batch][cells]`` vectors.
    """
    per = flat.reshape(batch, channels, -1)
    return [per[:, c].reshape(-1).contiguous() for c in range(channels)]


def _join_channels(parts: Sequence[torch.Tensor], batch: int) -> torch.Tensor:
    """Inverse of :func:`_split_channels`.

    Parameters
    ----------
    parts : sequence of torch.Tensor
        One flat ``[batch][cells]`` vector per channel.
    batch : int
        Number of environments.

    Returns
    -------
    torch.Tensor
        Flat ``[batch][channels][cells]`` vector.
    """
    return torch.stack([p.reshape(batch, -1) for p in parts], dim=1).reshape(-1)


def tensor_as_np(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a detached NumPy array on the CPU.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor.

    Returns
    -------
    numpy.ndarray
        The tensor data.
    """
    return tensor.detach().cpu().numpy()


def get_max_time_step(
    domain: _C.Domain,
    time_step_target: float,
    CFL_cond: float = 0.8,
    with_transformations: bool = True,
) -> tuple[float, int]:
    """Split a time step into substeps that satisfy a CFL condition.

    Parameters
    ----------
    domain : phipict._C.Domain
        The simulation domain.
    time_step_target : float
        Time step to split.
    CFL_cond : float, optional
        Target CFL number. Default is 0.8.
    with_transformations : bool, optional
        Whether to measure the velocity in computational space. Default is True.

    Returns
    -------
    time_step : float
        Substep size.
    substeps : int
        Number of substeps.
    """
    max_vel = domain.getMaxVelocity(True, with_transformations).cpu().numpy()
    max_time_step = CFL_cond / max_vel
    if max_time_step >= time_step_target:
        ss = 1
        ts = time_step_target
    else:
        ss = int(np.ceil(time_step_target / max_time_step))
        ts = time_step_target / ss
    return ts, ss


def _cfl_substep(
    time_step_target: float, max_vel: Any, CFL_cond: float
) -> tuple[Any, int]:
    """Split the remaining time of a step into substeps that meet a CFL bound.

    Parameters
    ----------
    time_step_target : float
        Remaining time of the step.
    max_vel : numpy.ndarray or float
        Current maximum velocity (a NumPy scalar array keeps its dtype).
    CFL_cond : float
        CFL number the substeps must respect.

    Returns
    -------
    time_step : NumPy scalar array or float
        Size of the next substep.
    substeps : int
        Number of substeps the remaining time is split into.
    """
    max_time_step: Any  # NumPy scalar array or float
    if np.isclose(max_vel, 0):
        max_time_step = time_step_target
    else:
        max_time_step = CFL_cond / max_vel
    if max_time_step >= time_step_target:
        return time_step_target, 1
    substeps = int(np.ceil(time_step_target / max_time_step))
    return time_step_target / substeps, substeps


def getVelocityResultMaxMag(domain: _C.Domain) -> float:
    """Get the maximum velocity magnitude of ``domain.velocityResult``.

    Parameters
    ----------
    domain : phipict._C.Domain
        The simulation domain.

    Returns
    -------
    float
        Maximum velocity magnitude.
    """
    vel = domain.velocityResult
    # [batch][dims][cells]: the maximum over all batched environments
    vel = torch.reshape(vel, (-1, domain.getSpatialDims(), domain.getTotalSize()))
    vel_max = torch.max(torch.abs(vel)).cpu().numpy().tolist()
    if vel_max > 0:
        return torch.max(torch.linalg.vector_norm(vel, dim=1)).cpu().numpy().tolist()
    return 0


def getVelocityResultMaxVel(domain: _C.Domain) -> float:
    """Get the maximum absolute velocity component of ``domain.velocityResult``.

    Parameters
    ----------
    domain : phipict._C.Domain
        The simulation domain.

    Returns
    -------
    float
        Maximum absolute velocity component.
    """
    vel = domain.velocityResult
    vel_max = torch.max(torch.abs(vel)).cpu().numpy().tolist()
    return vel_max


def _get_fixed_boundary_fluxes_torch(
    bound: _C.FixedBoundary, bound_idx: int
) -> torch.Tensor:
    """Per-cell boundary-normal fluxes of a FixedBoundary, as torch ops.

    Mirrors ``FixedBoundary::GetFluxes`` (``det * dot(Minv[axis], vel)``, see
    ``VelocityToContravariantComponentBoundaryFixed``), but stays in the autograd
    graph, which the C++ call does not.

    Parameters
    ----------
    bound : phipict._C.FixedBoundary
        Boundary to compute the fluxes of.
    bound_idx : int
        Face index of the boundary; its axis is ``bound_idx // 2``.

    Returns
    -------
    torch.Tensor
        Per-cell boundary-normal fluxes of shape ``[1, 1, *boundary_spatial]``.
    """
    dims = bound.getSpatialDims()
    bound_axis = bound_idx // 2
    vel = bound.velocity
    if vel.dim() == 2:  # static NC, broadcast over the boundary
        sizes = bound.getSizes()  # x,y,z,w
        spatial = [sizes[dims - 1 - d] for d in range(dims)]  # z,y,x
        vel = vel.reshape([1, dims] + [1] * dims).expand(*([1, dims] + spatial))
    if not bound.hasTransform():
        return vel[:, bound_axis : bound_axis + 1]
    transform = bound.transform  # NDHWC
    assert transform is not None
    inv_row_start = dims * dims + bound_axis * dims
    t_inv_row = transform[..., inv_row_start : inv_row_start + dims]  # NDHWC
    det = transform[..., -1]  # NDHW
    vel = torch.movedim(vel, 1, -1)  # NCDHW -> NDHWC
    return (det * torch.sum(t_inv_row * vel, dim=-1)).unsqueeze(1)


def _per_env_sum(fluxes: torch.Tensor) -> torch.Tensor:
    """Sum per-cell fluxes ``[N, ...]`` to one total per batched environment.

    Returns shape ``[N]``. A single environment (``N == 1``) is reduced exactly as
    before batching existed, as one sum over all elements.
    """
    if fluxes.dim() == 0 or fluxes.size(0) == 1:
        return torch.sum(fluxes).reshape(1)
    return fluxes.reshape(fluxes.size(0), -1).sum(dim=1)


def get_fixed_boundary_flux(bound: _C.Boundary, bound_idx: int) -> torch.Tensor:
    """Compute the total boundary-normal flux through a fixed boundary.

    Parameters
    ----------
    bound : phipict._C.Boundary
        A :class:`phipict._C.FixedBoundary` with Dirichlet velocity, or with
        Neumann (free-slip) velocity, whose flux is zero.
    bound_idx : int
        Face index of the boundary.

    Returns
    -------
    torch.Tensor
        Total flux per batched environment, of shape ``[N]`` (``N`` the batch size
        of the boundary velocity: 1 for a boundary shared by all environments),
        differentiable if gradients are enabled and the boundary velocity requires
        them.

    Raises
    ------
    ValueError
        If the velocity is neither a Dirichlet nor a Neumann condition.
    """
    assert isinstance(bound, _C.FixedBoundary)
    if bound.velocityType == _C.BoundaryConditionType.NEUMANN:
        # free-slip (OpenBoundary): the normal velocity is zero, so is the flux
        domain = bound.getParentDomain()
        return torch.zeros([1], dtype=domain.getDtype(), device=domain.getDevice())
    if bound.velocityType != _C.BoundaryConditionType.DIRICHLET:
        raise ValueError("Only DIRICHLET boundaries are supportet")

    # `GetFluxes` runs in C++ and returns a tensor with no grad_fn, so a flux
    # balance built on it can never carry a derivative. The torch path is only
    # taken when a graph is actually wanted, which keeps every non-differentiable
    # forward on exactly the arithmetic it had before
    if torch.is_grad_enabled() and bound.velocity.requires_grad:
        return _per_env_sum(_get_fixed_boundary_fluxes_torch(bound, bound_idx))
    return _per_env_sum(bound.GetFluxes())


def get_fixed_boundary_fluxes(
    list_idx_bound: Sequence[tuple[int, _C.Boundary]],
) -> torch.Tensor:
    """Compute the net outward flux through a set of fixed boundaries.

    Parameters
    ----------
    list_idx_bound : Sequence of (int, phipict._C.Boundary)
        Face indices and boundaries.

    Returns
    -------
    torch.Tensor
        Net flux of shape ``[1]``, or ``[B]`` per environment if any of the
        boundaries holds per-environment velocities of batch size ``B``.

    Raises
    ------
    TypeError
        If a boundary is not a :class:`phipict._C.FixedBoundary`.
    """
    if len(list_idx_bound) == 0:
        return torch.zeros([1], dtype=torch.float32, device=torch.device("cuda"))

    domain = list_idx_bound[0][1].getParentDomain()
    boundary_flux = torch.zeros([1], dtype=domain.getDtype(), device=domain.getDevice())
    for boundIdx, bound in list_idx_bound:
        if isinstance(bound, _C.FixedBoundary):
            flux = get_fixed_boundary_flux(bound, boundIdx)
        else:
            raise TypeError("boundary type not supported.")
        # _LOG.info("fixed bound flux: %s", boundary_flux)

        # out of place: a per-environment flux [B] broadcasts the [1] accumulator
        if boundIdx % 2 == 0:
            boundary_flux = boundary_flux - flux
        else:
            boundary_flux = boundary_flux + flux
    return boundary_flux


def get_varying_boundary_flux(bound: _C.Boundary, bound_idx: int) -> torch.Tensor:
    """Compute the total boundary-normal flux through a varying Dirichlet boundary.

    Parameters
    ----------
    bound : phipict._C.Boundary
        A :class:`phipict._C.VaryingDirichletBoundary`.
    bound_idx : int
        Face index of the boundary.

    Returns
    -------
    torch.Tensor
        Total flux per batched environment, of shape ``[N]``.
    """
    assert isinstance(bound, _C.VaryingDirichletBoundary)
    dims = bound.getSpatialDims()
    bound_axis = bound_idx // 2
    if bound.hasTransform:
        shape = bound.getSizes()  # x,y,z
        transform = bound.transform  # NDHWC

        inv_row_start = dims * dims + bound_axis * dims
        inv_row_end = inv_row_start + dims
        t_inv_row = transform[..., inv_row_start:inv_row_end].view(
            -1, 1, dims
        )  # (NDHW)1C
        J = transform[..., -1].view(-1, 1, 1)  # (NDHW)11
        bound_vel = torch.moveaxis(bound.boundaryVelocity, 1, -1).view(
            -1, dims, 1
        )  # NCDHW -> NDHWC -> (NDHW)C1
        flux = J * torch.bmm(t_inv_row, bound_vel)

        return _per_env_sum(flux.reshape(bound.boundaryVelocity.size(0), -1))
    else:
        return _per_env_sum(bound.boundaryVelocity[:, bound_axis])


def restrict_inflow(bound_vel: torch.Tensor, bound_idx: int) -> torch.Tensor:
    """Clamp the boundary-normal velocity so there is no inflow.

    Only valid for orthogonal transforms. Note: this breaks the outflow.

    Parameters
    ----------
    bound_vel : torch.Tensor
        Boundary velocity in NCDHW layout.
    bound_idx : int
        Face index of the boundary.

    Returns
    -------
    torch.Tensor
        Clamped boundary velocity.
    """
    bound_axis = bound_idx >> 1
    bound_dir = bound_idx & 1
    # only for orthogonal transforms
    dims = bound_vel.dim() - 2
    channels = list(bound_vel.split(dims, dim=1))
    flux = channels[bound_axis]
    zero = torch.tensor(0, dtype=bound_vel.dtype, device=bound_vel.device)
    if bound_dir == 0:  # lower bound, inlfow is positive
        flux = torch.minimum(flux, zero)
    else:
        flux = torch.maximum(flux, zero)
    channels[bound_axis] = flux
    return torch.cat(channels, dim=1)


def get_advective_velocity(
    velms: torch.Tensor | Sequence[torch.Tensor],
    velm_idx: int,
    bound: Any,
    bound_idx: int,
) -> torch.Tensor:
    """Compute the boundary-normal advection velocity of an outflow boundary.

    Parameters
    ----------
    velms : torch.Tensor or Sequence of torch.Tensor
        Advection velocity, global or one per boundary: static ``[N, dims]`` or
        varying NCDHW, with ``N`` 1 (shared by all environments) or the batch
        size.
    velm_idx : int
        Index of the boundary in ``velms`` if it is a sequence.
    bound : phipict._C.FixedBoundary or phipict._C.VaryingDirichletBoundary
        The boundary.
    bound_idx : int
        Face index of the boundary.

    Returns
    -------
    torch.Tensor
        Contravariant boundary-normal velocity in N1DHW layout.
    """
    velm = velms if isinstance(velms, torch.Tensor) else velms[velm_idx]

    dims = bound.getSpatialDims()
    velm_static = velm.dim() == 2  # NC or NCDHW (1,dims,[[z,]y,]x)

    bound_axis = bound_idx >> 1
    bound_size = bound.getSizes()  # x,y,z,w
    bound_spatial_shape = [bound_size[dims - 1 - d] for d in range(dims)]  # z,y,x
    hasTransform = (
        bound.hasTransform()
        if isinstance(bound, _C.FixedBoundary)
        else bound.hasTransform
    )

    if not velm_static:
        assert all(velm.size(2 + d) == bound_spatial_shape[d] for d in range(dims)), (
            "varying velm size must match boundary"
        )

    if hasTransform:
        n_env = velm.size(0)
        if velm_static:  # broadcast to boundary shape
            velm = velm.reshape([n_env, dims] + [1] * dims).repeat(
                1, 1, *bound_spatial_shape
            )
        transform = bound.transform  # NDHWC
        inv_row_start = dims * dims + bound_axis * dims
        inv_row_end = inv_row_start + dims
        if n_env == transform.size(0):
            t_inv_row = transform[..., inv_row_start:inv_row_end].view(
                -1, 1, dims
            )  # (NDHW)1C
            velm = torch.moveaxis(velm, 1, -1).view(
                -1, dims, 1
            )  # NCDHW -> NDHWC -> (NDHW)C1
            advective_vel = torch.bmm(t_inv_row, velm)  # (NDHW) 1 1
        else:
            # per-environment velocities on a shared transform: broadcast N
            t_inv_row = transform[..., inv_row_start:inv_row_end]  # 1DHWC
            advective_vel = torch.sum(t_inv_row * torch.moveaxis(velm, 1, -1), dim=-1)
        advective_vel = advective_vel.reshape([n_env, 1] + bound_spatial_shape)  # NCDHW
    elif velm_static:
        # [N, 1, 1(, 1), 1]: broadcasts over the boundary, not the batch
        advective_vel = velm[:, bound_axis : bound_axis + 1].reshape(
            [velm.size(0), 1] + [1] * dims
        )
    else:
        advective_vel = velm[:, bound_axis : bound_axis + 1]  # N1DHW

    return advective_vel


def balance_boundary_fluxes(
    domain: _C.Domain,
    free_bounds: Sequence[_C.Boundary],
    tol: float | None = None,
    differentiable: bool = False,
) -> None:
    """Rescale the free boundaries so the net boundary flux is zero.

    With batched environments the net flux is balanced per environment: a free
    boundary shared by all environments becomes a per-environment boundary when
    the environments need different scales.

    Parameters
    ----------
    domain : phipict._C.Domain
        The simulation domain.
    free_bounds : Sequence of phipict._C.Boundary
        Boundaries whose velocity may be rescaled.
    tol : float or None, optional
        Solver tolerance defining the balance threshold. Default is None
        (dtype default).
    differentiable : bool, optional
        Keep the flux computation in the autograd graph, so that ``flux_scale``
        carries its dependence on the boundary velocities.

        Without it the adjoint keeps the net-flux component of the free-boundary
        gradient that the rescale projects out in the forward. Measured on the
        MHD duct against central finite differences over 10 PISO steps: the
        derivative of the wall Nusselt number w.r.t. the outflow boundary velocity
        came out +1.08 against a true -0.108 (wrong sign, 11x), and the outflow
        heat flux was off by 321%; differentiating the balance brings both to the
        accuracy of interior perturbations (14% and 3%). Default is False.
    """
    scale_all = True

    with torch.no_grad() if not differentiable else nullcontext():
        boundaries = []
        for block in domain.getBlocks():
            boundaries.extend(block.getFixedBoundaries())  # list((boundIdx, bound),)

        fixed_boundaries = [_ for _ in boundaries if _[1] not in free_bounds]
        fixed_boundary_flux = get_fixed_boundary_fluxes(fixed_boundaries)
        variable_boundaries = [_ for _ in boundaries if _[1] in free_bounds]
        variable_boundary_flux = get_fixed_boundary_fluxes(variable_boundaries)

    # compensation_additive = False
    # _LOG.info("bounds %d: fixed %d, var %d", len(boundaries), len(fixed_boundaries), len(variable_boundaries))
    # _LOG.info("Fluxes: fixed %s, var %s", fixed_boundary_flux, variable_boundary_flux)
    # _LOG.info("Boundary flux balance %s", domain.GetBoundaryFluxBalance().cpu().numpy())
    balance_tol = piso_diff._get_solver_tolerance(tol, dtype=domain.getDtype())
    assert balance_tol is not None
    if not torch.allclose(
        fixed_boundary_flux + variable_boundary_flux,
        torch.zeros_like(fixed_boundary_flux),
        atol=balance_tol * 0.01,
    ):
        flux_scale = -fixed_boundary_flux / variable_boundary_flux
        # d_flux = fixed_boundary_flux - variable_boundary_flux
        # _LOG.info("FluxScale: %s", flux_scale)

        for boundIdx, bound in variable_boundaries:
            bound_vel = bound.velocity
            # one scale per environment, broadcast over channels and space
            if flux_scale.numel() > 1:
                env_scale = flux_scale.reshape([-1] + [1] * (bound_vel.dim() - 1))
            else:
                env_scale = flux_scale
            # TODO: scale everyting or only the boundary normal component?
            if scale_all:
                bound.setVelocity(bound_vel * env_scale)
            else:
                # TODO: non-ortho transform
                vel_comps = list(torch.split(bound_vel, domain.getSpatialDims(), dim=1))
                vel_comps[boundIdx // 2] = vel_comps[boundIdx // 2] * env_scale
                bound.setVelocity(torch.cat(vel_comps, axis=1))  # type: ignore[call-overload]
    # _LOG.info("Boundary flux balance %s", domain.GetBoundaryFluxBalance().cpu().numpy())


def update_advective_boundaries(
    domain: _C.Domain,
    bounds: list[_C.Boundary],
    velms: torch.Tensor | list[torch.Tensor],
    dt: float | torch.Tensor,
    tol: float | None = None,
    differentiable: bool = False,
) -> None:
    """Advect the velocity and scalar of outflow boundaries.

    Afterwards, the boundary fluxes are balanced with
    :func:`balance_boundary_fluxes`. See also
    https://www.tfd.chalmers.se/~hani/kurser/OS_CFD_2022/LeandroLucchese/Report_Lucchese.pdf

    Parameters
    ----------
    domain : phipict._C.Domain
        The simulation domain.
    bounds : list of phipict._C.Boundary
        Boundaries that are updated.
    velms : torch.Tensor or list of torch.Tensor
        Velocity to advect the boundaries with, global or one per boundary.
    dt : float or torch.Tensor
        Time step, a scalar or one per batched environment (shape ``[B]``, as
        the hooks receive it with per-environment adaptive substeps).
    tol : float or None, optional
        See :func:`balance_boundary_fluxes`. Default is None.
    differentiable : bool, optional
        Whether to record the update in the autograd graph. Default is False.

    Raises
    ------
    ValueError
        If a boundary is not part of the domain or not varying.
    """
    with (
        torch.no_grad() if not differentiable else nullcontext(),
        SAMPLE("advect bounds"),
    ):
        assert isinstance(bounds, list) and len(bounds) > 0
        # batch size 1 (shared by all environments) or one entry per environment
        batch_sizes = (1, domain.getBatchSize())
        assert (
            isinstance(velms, torch.Tensor)
            and velms.dim() == 2
            and velms.size(0) in batch_sizes
            and velms.size(1) == domain.getSpatialDims()
        ) or (
            isinstance(velms, list)
            and len(velms) == len(bounds)
            and all(
                isinstance(velm, torch.Tensor)
                and (velm.dim() == 2 or velm.dim() == (domain.getSpatialDims() + 2))
                and velm.size(0) in batch_sizes
                and velm.size(1) == domain.getSpatialDims()
                for velm in velms
            )
        ), "velms tensors have wrong shape"

        # if any([block.hasTransform for block in domain.getBlocks()]):
        #    raise RuntimeError("update_advective_boundaries does not support transformations.")

        dims = domain.getSpatialDims()
        if isinstance(dt, torch.Tensor) and dt.numel() > 1:
            # one substep size per batched environment, broadcast over N1(D)HW
            dt = dt.view(-1, *([1] * (dims + 1)))
        boundaries: list[tuple[Any, int, Any]] = []
        for blockIdx in range(domain.getNumBlocks()):
            block = domain.getBlock(blockIdx)
            for boundIdx in range(dims * 2):
                bound = block.getBoundary(boundIdx)
                if isinstance(
                    bound,
                    (
                        _C.VaryingDirichletBoundary,
                        _C.StaticDirichletBoundary,
                    ),
                ):
                    boundaries.append((block, boundIdx, bound))
                    warnings.warn("DirichletBoundary is deprecated.", stacklevel=1)
                if isinstance(bound, _C.FixedBoundary):
                    boundaries.append((block, boundIdx, bound))

        # fixed_boundaries = [_ for _ in boundaries if _[2] not in bounds]
        variable_boundaries = [_ for _ in boundaries if _[2] in bounds]

        if len(bounds) != len(variable_boundaries):
            raise ValueError(
                "Bounds passed as advective outflow are not (all) part of the domain."
            )

        # _LOG.info("%d fixed, %d variable", len(fixed_boundaries), len(variable_boundaries))

        for block, boundIdx, bound in variable_boundaries:
            if isinstance(bound, (_C.VaryingDirichletBoundary, _C.FixedBoundary)):
                passive_scalar = (
                    block.passiveScalar if block.hasPassiveScalar() else None
                )
                if boundIdx == 0:
                    vel_slice = block.velocity[..., :1]
                    scal_slice = (
                        passive_scalar[..., :1] if passive_scalar is not None else None
                    )
                elif boundIdx == 1:
                    vel_slice = block.velocity[..., -1:]
                    scal_slice = (
                        passive_scalar[..., -1:] if passive_scalar is not None else None
                    )
                elif boundIdx == 2:
                    vel_slice = block.velocity[..., :1, :]
                    scal_slice = (
                        passive_scalar[..., :1, :]
                        if passive_scalar is not None
                        else None
                    )
                elif boundIdx == 3:
                    vel_slice = block.velocity[..., -1:, :]
                    scal_slice = (
                        passive_scalar[..., -1:, :]
                        if passive_scalar is not None
                        else None
                    )
                elif boundIdx == 4:
                    vel_slice = block.velocity[..., :1, :, :]
                    scal_slice = (
                        passive_scalar[..., :1, :, :]
                        if passive_scalar is not None
                        else None
                    )
                elif boundIdx == 5:
                    vel_slice = block.velocity[..., -1:, :, :]
                    scal_slice = (
                        passive_scalar[..., -1:, :, :]
                        if passive_scalar is not None
                        else None
                    )
                else:
                    raise RuntimeError

                if True:
                    alpha = (
                        dt
                        * 2
                        * get_advective_velocity(
                            velms, bounds.index(bound), bound, boundIdx
                        )
                    )  # N1DHW
                else:
                    vel_m = torch.abs(velms[bounds.index(bound)])
                    alpha = dt * 2 * vel_m  # dt * flow speed / distance(center, face)
                    hasTransform = (
                        bound.hasTransform()
                        if isinstance(bound, _C.FixedBoundary)
                        else bound.hasTransform
                    )
                    if hasTransform:
                        bound_axis = boundIdx >> 1
                        matrix_element = dims * dims + bound_axis * dims + bound_axis
                        alpha = (
                            alpha * bound.transform[..., matrix_element]
                        )  # inverse cell size in boundary-normal direction, assumes orthogonal transformation

                t = 1 - 1 / (1 + alpha)  # interpolation weight
                # _LOG.info("advective bound interpolation weight: %s", t)

                if isinstance(bound, _C.FixedBoundary):
                    if bound.isVelocityStatic:
                        raise ValueError(
                            "outflow boundaries must be varying. Use FixedBoundary.makeVelocityVarying() to make the velocity varying."
                        )
                    vel_bound = bound.velocity
                    if block.hasPassiveScalar() and bound.hasPassiveScalar():
                        if bound.isPassiveScalarStatic():
                            raise ValueError(
                                "outflow boundary passive scalar must be varying."
                            )
                        if any(
                            cond != _C.BoundaryConditionType.DIRICHLET
                            for cond in bound.passiveScalarTypes  # type: ignore[union-attr]
                        ):
                            raise ValueError(
                                "passive scalar outflow boundary condition must be Dirichlet."
                            )
                        scal_bound = bound.passiveScalar
                    else:
                        scal_bound = None
                else:
                    vel_bound = bound.boundaryVelocity
                    scal_bound = bound.boundaryScalar

                # _LOG.info("advective bound vel: %s", vel_bound)
                vel_bound_update = vel_bound - t * (vel_bound - vel_slice)  # type: ignore[operator]
                bound.setVelocity(vel_bound_update)

                if scal_bound is not None:
                    scal_bound_update = scal_bound - t * (scal_bound - scal_slice)  # type: ignore[operator]
                    # scal_bound.copy_(scal_bound - t*(scal_bound - scal_slice))
                    bound.setPassiveScalar(scal_bound_update)
            else:
                raise TypeError

    balance_boundary_fluxes(domain, bounds, tol=tol, differentiable=differentiable)


def update_advective_boundaries_static(
    domain: _C.Domain,
    bounds: Sequence[_C.Boundary],
    velms: Sequence[torch.Tensor],
    dt: float | torch.Tensor,
) -> None:
    """Advect the scalar of deprecated varying Dirichlet outflow boundaries.

    Parameters
    ----------
    domain : phipict._C.Domain
        The simulation domain.
    bounds : Sequence of phipict._C.Boundary
        Boundaries that are updated.
    velms : Sequence of torch.Tensor
        Advection velocity per boundary.
    dt : float or torch.Tensor
        Time step.
    """
    boundaries: list[tuple[int, Any]] = []
    for blockIdx in range(domain.getNumBlocks()):
        block = domain.getBlock(blockIdx)
        for boundIdx in range(domain.getSpatialDims() * 2):
            bound = block.getBoundary(boundIdx)
            if isinstance(
                bound,
                (_C.VaryingDirichletBoundary, _C.StaticDirichletBoundary),
            ):
                boundaries.append((boundIdx, bound))

    variable_boundaries = [_ for _ in boundaries if _[1] in bounds]

    for boundIdx, bound in variable_boundaries:
        if isinstance(bound, _C.VaryingDirichletBoundary):
            passive_scalar = block.passiveScalar
            assert passive_scalar is not None
            if boundIdx == 0:
                scal_slice = passive_scalar[..., :1]
            elif boundIdx == 1:
                scal_slice = passive_scalar[..., -1:]
            elif boundIdx == 2:
                scal_slice = passive_scalar[..., :1, :]
            elif boundIdx == 3:
                scal_slice = passive_scalar[..., -1:, :]
            elif boundIdx == 4:
                scal_slice = passive_scalar[..., :1, :, :]
            elif boundIdx == 5:
                scal_slice = passive_scalar[..., -1:, :, :]
            else:
                raise RuntimeError
            vel_m = torch.abs(velms[bounds.index(bound)])
            scal_bound = bound.boundaryScalar
            scal_bound.copy_(scal_bound - (dt * 2 * vel_m) * (scal_bound - scal_slice))
        else:
            raise TypeError


def append_prep_fn(prep_fn: Any, name: Hook | str, fn: Callable[..., Any]) -> Any:
    """Register a simulation callback after existing ones.

    The functional form of :meth:`~phipict.core.hooks.Hooks.append`, which is
    the preferred way; also works on a plain dict of callbacks by hook name.

    Parameters
    ----------
    prep_fn : Hooks or dict of str to Any
        Hooks, or callbacks by hook name (a callable or a list/tuple of them).
    name : Hook or str
        The hook, e.g. ``Hook.PRE``, or its name.
    fn : Callable
        Callback, called with keyword arguments only.

    Returns
    -------
    Hooks or dict of str to Any
        ``prep_fn``, modified in place.

    Raises
    ------
    ValueError
        If ``name`` is not a :class:`~phipict.core.hooks.Hook`.
    """
    if isinstance(prep_fn, Hooks):
        return prep_fn.append(name, fn)
    assert isinstance(prep_fn, dict)
    name = Hook.parse(name)
    if name not in prep_fn:
        prep_fn[name] = [fn]
    else:
        if isinstance(prep_fn[name], tuple):
            prep_fn[name] = list(prep_fn[name])

        if not isinstance(prep_fn[name], list):
            prep_fn[name] = [prep_fn[name], fn]
        else:
            prep_fn[name].append(fn)
    return prep_fn


def prepend_prep_fn(prep_fn: Any, name: Hook | str, fn: Callable[..., Any]) -> Any:
    """Register a simulation callback before existing ones.

    The functional form of :meth:`~phipict.core.hooks.Hooks.prepend`, which is
    the preferred way; also works on a plain dict of callbacks by hook name.

    Parameters
    ----------
    prep_fn : Hooks or dict of str to Any
        Hooks, or callbacks by hook name (a callable or a list/tuple of them).
    name : Hook or str
        The hook, e.g. ``Hook.PRE``, or its name.
    fn : Callable
        Callback, called with keyword arguments only.

    Returns
    -------
    Hooks or dict of str to Any
        ``prep_fn``, modified in place.

    Raises
    ------
    ValueError
        If ``name`` is not a :class:`~phipict.core.hooks.Hook`.
    """
    if isinstance(prep_fn, Hooks):
        return prep_fn.prepend(name, fn)
    assert isinstance(prep_fn, dict)
    name = Hook.parse(name)
    if name not in prep_fn:
        prep_fn[name] = [fn]
    else:
        if isinstance(prep_fn[name], tuple):
            prep_fn[name] = list(prep_fn[name])

        if not isinstance(prep_fn[name], list):
            prep_fn[name] = [fn, prep_fn[name]]
        else:
            prep_fn[name].insert(0, fn)
    return prep_fn


class Simulation:
    """PISO simulation of a multi-block domain.

    Parameters
    ----------
    domain : phipict._C.Domain or None, optional
        The initialized simulation domain. Default is None.
    time_step : float, optional
        Time step per substep. Default is 1.
    substeps : int or "ADAPTIVE", optional
        Substeps per iteration, or ``"ADAPTIVE"`` to derive them from
        ``adaptive_CFL``. Default is 1.
    corrector_steps : int, optional
        Number of PISO corrector steps. Default is 2.
    density_viscosity : float or None, optional
        Superseded by ``Domain.scalarViscosity``; must be None.
    adaptive_CFL : float, optional
        CFL number for adaptive substeps. Default is 0.8.
    adaptive_CFL_per_env : bool, optional
        With batched environments, derive the adaptive substeps of every
        environment from its own maximum velocity, so it takes the substeps it
        would take on its own. Otherwise all take the substeps of the fastest one.
        Not supported with ``differentiable``, which uses the shared substeps.
        Default is True.
    prep_fn : Hooks, dict or None, optional
        The former name of ``hooks``, kept for compatibility. Default is None.
    advection_use_BiCG : bool, optional
        Whether to solve the advection with BiCGStab. Default is True.
    pressure_use_BiCG : bool, optional
        Whether to solve the pressure with BiCGStab. Default is False.
    scipy_solve_advection : bool, optional
        Whether to solve the advection with SciPy. Default is False.
    scipy_solve_pressure : bool, optional
        Whether to solve the pressure with SciPy. Default is False.
    preconditionBiCG : bool, optional
        Whether BiCGStab uses a preconditioner. Default is False.
    BiCG_precondition_fallback : bool, optional
        Whether to retry a failed BiCGStab solve with a preconditioner.
        Default is True.
    advection_tol : float, SolverTolerance or None, optional
        Advection solver tolerance. None uses the dtype default.
    pressure_tol : float, SolverTolerance or None, optional
        Pressure solver tolerance. None uses the dtype default.
    pressure_tol_intermediate : float, SolverTolerance or None, optional
        Tolerance of all but the final pressure solve. None uses
        ``pressure_tol``.
    pressure_warm_start : bool, optional
        Whether to start pressure solves from the previous substep's result.
        Default is False.
    pressure_use_amg : bool or None, optional
        Whether to solve the pressure equation with AMG-preconditioned CG. The
        interpolation of the hierarchy is set up once per sparsity pattern (on
        the host, with pyamg) and cached; whenever the pressure matrix values
        change, only the coarse operators are recomputed on the GPU (Galerkin
        products with the frozen interpolation). Requires pyamg and a symmetric
        pressure matrix (``pressure_use_BiCG=False``). None (the default)
        enables it automatically when pyamg is installed and the pressure system
        has at least :attr:`PRESSURE_AMG_AUTO_MIN_ROWS` unknowns; below that,
        the (Jacobi-preconditioned) CG is faster than the V-cycle overhead.
    pressure_amg_options : dict or None, optional
        Keyword arguments of :func:`phipict.solvers.amg.hierarchy_for`, e.g.
        ``{"method": "ruge_stuben", "native_dtype": torch.float32}``, plus
        ``coarse_refresh_interval`` (default 4): recompute the coarse operators
        only on every n-th change of the pressure matrix; in between only the
        finest level follows the matrix. A full refresh is forced earlier when
        a solve needs more than twice the iterations of the first solve after
        the last full refresh. Default is None.
    convergence_tol : float or None, optional
        Stop when the maximum velocity change of a step is below this value.
        Default is None.
    solver_double_fallback : bool, optional
        Whether to retry failed single precision solves in double precision.
        Default is False.
    advect_non_ortho_steps : int, optional
        Non-orthogonal correction steps of the advection. Default is 1.
    pressure_non_ortho_steps : int, optional
        Non-orthogonal correction steps of the pressure. Default is 1.
    normalize_pressure_result : bool, optional
        Whether to subtract the mean pressure. Default is True.
    pressure_return_best_result : bool, optional
        Whether a non-converged pressure solve returns its best iterate.
        Default is False.
    advect_passive_scalar : bool, optional
        Whether to advect the passive scalar. Default is True.
    pressure_time_step_normalized : bool, optional
        Whether the pressure is normalized by the time step. Default is False.
    velocity_corrector : {"FD", "FVM_CENTER", "FVM_FACE"}, optional
        Pressure gradient discretization of the velocity correction.
        Default is ``"FD"``.
    non_orthogonal : bool, optional
        Whether to use non-orthogonal corrections. Default is True.
    differentiable : bool, optional
        Whether to use the differentiable backend :mod:`phipict.core.piso_diff`.
        Default is False.
    exclude_advection_solve_gradients : bool, optional
        Whether to run advection solves without gradients. Default is False.
    exclude_pressure_solve_gradients : bool, optional
        Whether to run pressure solves without gradients. Default is False.
    log_dir : str or None, optional
        Output directory. Default is None.
    log_interval : int, optional
        Iterations between log outputs, 0 disables them. Default is 0.
    log_images : bool, optional
        Whether to write images. Default is True.
    norm_vel : bool, optional
        Whether to normalize velocity images to the maximum magnitude.
        Default is False.
    block_layout : list or None, optional
        Block arrangement of images, see :func:`phipict.io.output.arrange_blocks`.
        Default is None.
    output_mode3D : str, optional
        Reduction of 3D images, see :func:`phipict.io.output.reduce_3D`.
        Default is ``"slice"``.
    log_fn : Callable or None, optional
        Called after every iteration with ``domain``, ``out_dir``, ``it``,
        ``out_it`` and ``total_step``. Default is None.
    output_resampling_coords : Sequence of torch.Tensor or None, optional
        Vertex coordinates for resampled images. None uses the domain's.
    output_resampling_shape : ResamplingShape or None, optional
        Shape of resampled images. None disables resampling.
    output_resampling_fill_max_steps : int, optional
        Hole-filling iterations of the resampling. Default is 0.
    save_domain_name : str or None, optional
        Name to save the domain under after :meth:`run`. Default is None.
    stop_fn : Callable or None, optional
        Returns True to stop the simulation. Default never stops.
    hooks : Hooks, dict or None, optional
        Callbacks run at the hooks of every PISO step, see
        :class:`~phipict.core.hooks.Hooks`. Default is None (no callbacks).
    """

    # these must match the definition in 'PISO_multiblock_cuda.h'
    NON_ORTHO_DIRECT_MATRIX = 1
    NON_ORTHO_DIRECT_RHS = 2  # less stable than NON_ORTHO_DIRECT_MATRIX
    NON_ORTHO_DIAGONAL_MATRIX = 4  # not implemented
    NON_ORTHO_DIAGONAL_RHS = 8
    NON_ORTHO_CENTER_MATRIX = 16

    # pressure systems at least this large use AMG when pressure_use_amg is None
    PRESSURE_AMG_AUTO_MIN_ROWS = 200_000
    # batched environments: total unknowns of all environments (the batched V-cycle
    # pays off later, see SOLVER_CHANGES.md, batched environments)
    PRESSURE_AMG_AUTO_MIN_ROWS_BATCHED = 400_000

    __NON_ORTHO_MODE = (
        NON_ORTHO_CENTER_MATRIX | NON_ORTHO_DIRECT_MATRIX | NON_ORTHO_DIAGONAL_RHS
    )  # Bit flags

    def __init__(
        self,
        domain: _C.Domain | None = None,
        *,
        time_step: float = 1.0,
        substeps: int | Literal["ADAPTIVE"] = 1,
        corrector_steps: int = 2,
        density_viscosity: float | None = None,
        adaptive_CFL: float = 0.8,
        adaptive_CFL_per_env: bool = True,
        prep_fn: Hooks | dict[str, Any] | None = None,
        advection_use_BiCG: bool = True,
        pressure_use_BiCG: bool = False,
        scipy_solve_advection: bool = False,
        scipy_solve_pressure: bool = False,
        preconditionBiCG: bool = False,
        BiCG_precondition_fallback: bool = True,
        advection_tol: float | SolverTolerance | None = None,
        pressure_tol: float | SolverTolerance | None = None,
        pressure_tol_intermediate: float | SolverTolerance | None = None,
        pressure_warm_start: bool = False,
        pressure_use_amg: bool | None = None,
        pressure_amg_options: dict[str, Any] | None = None,
        convergence_tol: float | None = None,
        solver_double_fallback: bool = False,
        advect_non_ortho_steps: int = 1,
        pressure_non_ortho_steps: int = 1,
        normalize_pressure_result: bool = True,
        pressure_return_best_result: bool = False,
        advect_passive_scalar: bool = True,
        pressure_time_step_normalized: bool = False,  # advect_velocity:bool=True,
        velocity_corrector: str = "FD",
        non_orthogonal: bool = True,
        differentiable: bool = False,
        exclude_advection_solve_gradients: bool = False,
        exclude_pressure_solve_gradients: bool = False,
        log_dir: str | None = None,
        log_interval: int = 0,
        log_images: bool = True,
        norm_vel: bool = False,
        block_layout: list | None = None,
        output_mode3D: str = "slice",
        log_fn: Callable[..., Any] | None = None,
        output_resampling_coords: Sequence[torch.Tensor] | None = None,
        output_resampling_shape: ResamplingShape | None = None,
        output_resampling_fill_max_steps: int = 0,
        save_domain_name: str | None = None,
        stop_fn: Callable[[], bool] | None = lambda: False,
        hooks: Hooks | dict[str, Any] | None = None,
    ) -> None:
        self.__LOG = get_logger("PISOsim")
        self.__differentiable = False

        self.domain = domain
        self.density_viscosity = density_viscosity
        if hooks is not None and prep_fn is not None:
            raise ValueError("Pass hooks or its former name prep_fn, not both.")
        self.hooks = hooks if hooks is not None else prep_fn

        self.non_orthogonal = non_orthogonal

        self.time_step = time_step
        self.substeps = substeps
        self.corrector_steps = corrector_steps
        self.convergence_tol = convergence_tol
        self.adaptive_CFL = adaptive_CFL
        self.adaptive_CFL_per_env = adaptive_CFL_per_env

        self.scipy_solve_advection = scipy_solve_advection
        self.advection_use_BiCG = advection_use_BiCG
        self.advect_non_ortho_steps = advect_non_ortho_steps
        self.advection_tol = advection_tol
        self.advect_passive_scalar = advect_passive_scalar

        self.scipy_solve_pressure = scipy_solve_pressure
        self.pressure_use_BiCG = pressure_use_BiCG
        self.pressure_non_ortho_steps = pressure_non_ortho_steps
        self.pressure_tol = pressure_tol
        self.pressure_tol_intermediate = pressure_tol_intermediate
        self.pressure_warm_start = pressure_warm_start
        self.pressure_use_amg = pressure_use_amg
        self._pressure_amg_options = dict(pressure_amg_options or {})
        # hierarchy of the current pressure matrix and the fingerprint of the
        # matrix values it was built for
        self._pressure_amg_hierarchy: AMGHierarchy | BatchedAMGHierarchy | None = None
        self._pressure_amg_fingerprint: tuple | None = None
        # refreshes since the last full (Galerkin) refresh, the largest iteration
        # count of the solves right after it, and whether the coarse levels are
        # considered stale (see _note_pressure_amg_iterations)
        self._pressure_amg_refreshes: int = 0
        self._pressure_amg_baseline_iters: int | None = None
        self._pressure_amg_stale: bool = False
        # Per-corrector cache of the last converged pressure, used as the initial
        # guess for the same corrector index at the next sub-step. Keyed on the
        # corrector index rather than being a single field on purpose: corrector 0
        # solves against div(u*) from the momentum predictor and is O(p), while
        # later correctors solve against an already-projected velocity and are
        # orders of magnitude smaller. One shared guess would hand each corrector
        # the other's magnitude, which is worse than starting from zero.
        self.__pressure_guess: dict[int, torch.Tensor] = {}
        self.normalize_pressure_result = normalize_pressure_result
        self.pressure_return_best_result = pressure_return_best_result
        self.pressure_time_step_normalized = pressure_time_step_normalized

        self.solver_double_fallback = solver_double_fallback
        self.linear_solve_max_iterations = 5000
        self.preconditionBiCG = preconditionBiCG
        self.BiCG_precondition_fallback = BiCG_precondition_fallback

        self._velocity_corrector_versions = {"FD": 1, "FVM_CENTER": 5, "FVM_FACE": 6}
        self.velocity_corrector = velocity_corrector

        self.log_dir = log_dir
        self.log_interval = log_interval
        self.log_images = log_images
        self.norm_vel = norm_vel
        self.block_layout = block_layout

        self.output_mode3D = output_mode3D
        self.log_fn = log_fn
        self.output_resampling_coords = output_resampling_coords
        self.output_resampling_shape = output_resampling_shape
        self.output_resampling_fill_max_steps = output_resampling_fill_max_steps
        self.save_domain_name = save_domain_name

        self.differentiable = differentiable

        self.exclude_advection_solve_gradients = exclude_advection_solve_gradients
        self.exclude_pressure_solve_gradients = exclude_pressure_solve_gradients
        # Cuts the pressure path out of the velocity-correction backward. Set from
        # the outside (like `linear_solve_max_iterations`), per simulation instance
        self.exclude_pressure_gradient_adjoint = False
        # self.exclude_all_linsolve_gradients = False

        self.print_adaptive_step_info = False

        self.stop_fn = stop_fn

        self.reset_step_counters()

    @property
    def domain(self) -> _C.Domain:
        """The simulation domain.

        Only None before a domain is assigned; methods that need it raise a
        ``ValueError`` in that case.

        Returns
        -------
        _C.Domain
            The simulation domain.
        """
        return cast(_C.Domain, self.__domain)

    @domain.setter
    def domain(self, domain: _C.Domain | None) -> None:
        """Set the simulation domain.

        Parameters
        ----------
        domain : _C.Domain or None
            The simulation domain.

        Raises
        ------
        TypeError
            domain must be a phipict.Domain object or None.
        RuntimeError
            domain must be initilized. Call domain.PrepareSolve() before assignment.
        """
        if domain is not None:
            if not isinstance(domain, _C.Domain):
                raise TypeError("domain must be a phipict.Domain object or None.")
            if not domain.IsInitialized():
                raise RuntimeError(
                    "domain must be initilized. Call domain.PrepareSolve() before assignment."
                )
        self.__domain = domain
        self.__check_differentiable()

    def _check_domain(self) -> None:
        """Raise if no domain has been assigned yet.

        Raises
        ------
        ValueError
            If no domain is set.
        """
        if self.__domain is None:
            raise ValueError("no domain set.")

    def __get_dtype(self) -> torch.dtype:
        self._check_domain()
        if self.domain.getNumBlocks() == 0:
            raise RuntimeError("Block required to infer dtype")
        return self.domain.getBlock(0).velocity.dtype

    def save_domain(self, name: str = "domain") -> None:
        """Save the domain to the log directory, logging failures.

        Parameters
        ----------
        name : str, optional
            File name without extension. Default is ``"domain"``.

        Raises
        ------
        RuntimeError
            If no log directory is set.
        """
        self._check_domain()
        if self.log_dir is None:
            raise RuntimeError("no log_dir set.")
        domain_path = Path(self.log_dir) / name
        try:
            save_domain(self.domain, domain_path)
        except BaseException:
            self.__LOG.exception("FAILED to save sim:")
        else:
            self.__LOG.info("sim saved as: %s", name)

    def reset_image_out_idx(self) -> None:
        """Reset the index of written images to 0."""
        self.img_out_idx = 0

    def save_domain_images(
        self, idx: int | None = None, max_mag: float = 1, vel_exr: bool = False
    ) -> None:
        """Write images of the domain fields to the log directory.

        Parameters
        ----------
        idx : int or None, optional
            Image index. If None, the internal counter is used and incremented.
            Default is None.
        max_mag : float, optional
            Velocity magnitude mapped to full brightness. Default is 1.
        vel_exr : bool, optional
            Whether to also write the velocity as EXR. Default is False.

        Raises
        ------
        RuntimeError
            If no log directory is set.
        """
        self._check_domain()
        if self.log_dir is None:
            raise RuntimeError("no log_dir set.")
        _idx = self.img_out_idx if idx is None else idx
        try:
            save_domain_images(
                self.domain,
                self.log_dir,
                _idx,
                layout=self.block_layout,
                norm_p=True,
                max_mag=max_mag,
                mode3D=self.output_mode3D,
                vel_exr=vel_exr,
                vertex_coord_list=self.output_resampling_coords,
                resampling_out_shape=self.output_resampling_shape,
                fill_max_steps=self.output_resampling_fill_max_steps,
            )
        except BaseException:
            self.__LOG.exception("FAILED to save images %s:", _idx)
        if idx is None:
            self.img_out_idx += 1

    @property
    def output_resampling_coords(self) -> Sequence[torch.Tensor] | None:
        """Vertex coordinates for resampled images, defaulting to the domain's.

        Returns
        -------
        Sequence of torch.Tensor or None
            Vertex coordinates for resampled images, defaulting to the domain's.
        """
        if self.__output_resampling_coords is not None:
            return self.__output_resampling_coords
        elif self.domain.hasVertexCoordinates():
            return self.domain.getVertexCoordinates()
        else:
            return None

    @output_resampling_coords.setter
    def output_resampling_coords(
        self, output_resampling_coords: Sequence[torch.Tensor] | None
    ) -> None:
        """Set the vertex coordinates for resampled images, or None for the domain's.

        Parameters
        ----------
        output_resampling_coords : Sequence of torch.Tensor or None
            Vertex coordinates for resampled images, defaulting to the domain's.
        """
        self.__output_resampling_coords = output_resampling_coords

    @property
    def density_viscosity(self) -> float | None:
        """Superseded by ``Domain.scalarViscosity``; always None.

        Returns
        -------
        float or None
            Superseded by ``Domain.scalarViscosity``; always None.
        """
        return self.__density_viscosity

    @density_viscosity.setter
    def density_viscosity(self, density_viscosity: float | None) -> None:
        """Reject any value; superseded by ``Domain.scalarViscosity``.

        Parameters
        ----------
        density_viscosity : float or None
            Superseded by ``Domain.scalarViscosity``; always None.

        Raises
        ------
        NotImplementedError
            Simulation.density_viscosity is superseded by Domain.scalarViscosity.
        TypeError
            density_viscosity must be float or None.
        ValueError
            density_viscosity must not be negative.
        """
        if density_viscosity is not None:
            raise NotImplementedError(
                "Simulation.density_viscosity is superseded by Domain.scalarViscosity."
            )
        if not (
            density_viscosity is None or isinstance(density_viscosity, numbers.Real)
        ):
            raise TypeError("density_viscosity must be float or None.")
        if density_viscosity is not None and density_viscosity < 0:
            raise ValueError("density_viscosity must not be negative.")
        self.__density_viscosity = density_viscosity

    def __check_differentiable(self) -> None:
        if self.differentiable:
            pass
            # if self.non_orthogonal: raise NotImplementedError("Differentiability is only implemented for orthogonal mode.")
            # if not self.pressure_time_step_normalized: raise NotImplementedError("Differentiable mode expects pressure to the time step normalized.")

    @property
    def differentiable(self) -> bool:
        """Whether the differentiable backend is used.

        Returns
        -------
        bool
            Whether the differentiable backend is used.
        """
        return self.__differentiable

    @differentiable.setter
    def differentiable(self, differentiable: bool) -> None:
        """Set whether the differentiable backend is used.

        Parameters
        ----------
        differentiable : bool
            Whether the differentiable backend is used.

        Raises
        ------
        TypeError
            differentiable must be bool.
        """
        if not isinstance(differentiable, bool):
            raise TypeError("differentiable must be bool.")
        self.__differentiable = differentiable
        self.__backend: ModuleType = piso_diff if differentiable else _C
        self.__check_differentiable()

    @property
    def _correct_velocity_kwargs(self) -> dict[str, bool]:
        """Extra kwargs for `CorrectVelocity`, only the differentiable backend takes.

        The non-differentiable backend is the raw C++ binding, which would reject
        the gradient-exclusion kwarg, so it is passed only in differentiable mode.

        Returns
        -------
        dict[str, bool]
            Extra kwargs for `CorrectVelocity`, only the differentiable backend takes.
        """
        if self.differentiable and self.exclude_pressure_gradient_adjoint:
            return {"exclude_pressure_grad": True}
        return {}

    @property
    def non_orthogonal(self) -> bool:
        """Whether non-orthogonal corrections are used.

        Returns
        -------
        bool
            Whether non-orthogonal corrections are used.
        """
        return self.__non_orthogonal

    @non_orthogonal.setter
    def non_orthogonal(self, non_orthogonal: bool) -> None:
        """Set whether non-orthogonal corrections are used.

        Parameters
        ----------
        non_orthogonal : bool
            Whether non-orthogonal corrections are used.

        Raises
        ------
        TypeError
            non_orthogonal must be bool.
        """
        if not isinstance(non_orthogonal, bool):
            raise TypeError("non_orthogonal must be bool.")
        self.__non_orthogonal = non_orthogonal
        self.__non_ortho_flags = Simulation.__NON_ORTHO_MODE if non_orthogonal else 0
        self.__check_differentiable()

    @property
    def time_step(self) -> float:
        """Time step per substep.

        Returns
        -------
        float
            Time step per substep.
        """
        return self.__time_step

    @time_step.setter
    def time_step(self, time_step: float) -> None:
        """Set the time step per substep.

        Parameters
        ----------
        time_step : float
            Time step per substep.

        Raises
        ------
        TypeError
            time_step must be float.
        """
        if not isinstance(time_step, numbers.Real):
            raise TypeError("time_step must be float.")
        self.__time_step: float = time_step

    def __get_time_step_torch(self) -> torch.Tensor:
        self._check_domain()
        return torch.tensor(
            [self.__time_step], device=cpu_device, dtype=self.__get_dtype()
        )

    @property
    def substeps(self) -> int:
        """Substeps per iteration, -1 for adaptive substeps.

        Returns
        -------
        int
            Substeps per iteration, -1 for adaptive substeps.
        """
        return self.__substeps

    @substeps.setter
    def substeps(self, substeps: int | str) -> None:
        """Set the number of substeps per iteration, -1 for adaptive substeps.

        Parameters
        ----------
        substeps : int or str
            Substeps per iteration, -1 for adaptive substeps.

        Raises
        ------
        TypeError
            substeps must be integral type or 'ADAPTIVE'.
        ValueError
            substeps must be positive.
        """
        if isinstance(substeps, str) and substeps.upper() == "ADAPTIVE":
            substeps = -1
        else:
            if not isinstance(substeps, numbers.Integral):
                raise TypeError("substeps must be integral type or 'ADAPTIVE'.")
            if not substeps > 0:
                raise ValueError("substeps must be positive.")
        self.__substeps = substeps

    @property
    def adaptive_CFL(self) -> float:
        """CFL number for adaptive substeps.

        Returns
        -------
        float
            CFL number for adaptive substeps.
        """
        return self.__adaptive_CFL

    @adaptive_CFL.setter
    def adaptive_CFL(self, adaptive_CFL: float) -> None:
        """Set the CFL number for adaptive substeps.

        Parameters
        ----------
        adaptive_CFL : float
            CFL number for adaptive substeps.

        Raises
        ------
        TypeError
            adaptive_CFL must be float.
        ValueError
            adaptive_CFL must be positive.
        """
        if not isinstance(adaptive_CFL, numbers.Real):
            raise TypeError("adaptive_CFL must be float.")
        if not adaptive_CFL > 0:
            raise ValueError("adaptive_CFL must be positive.")
        self.__adaptive_CFL: float = adaptive_CFL

    @property
    def corrector_steps(self) -> int:
        """Number of PISO corrector steps.

        Returns
        -------
        int
            Number of PISO corrector steps.
        """
        return self.__corrector_steps

    @corrector_steps.setter
    def corrector_steps(self, corrector_steps: int) -> None:
        """Set the number of PISO corrector steps.

        Parameters
        ----------
        corrector_steps : int
            Number of PISO corrector steps.

        Raises
        ------
        TypeError
            corrector_steps must be integral type.
        ValueError
            corrector_steps must not be negative.
        """
        if not isinstance(corrector_steps, numbers.Integral):
            raise TypeError("corrector_steps must be integral type.")
        if corrector_steps < 0:
            raise ValueError("corrector_steps must not be negative.")
        self.__corrector_steps: int = corrector_steps

    @property
    def convergence_tol(self) -> float | None:
        """Velocity change below which the simulation stops.

        Returns
        -------
        float or None
            Velocity change below which the simulation stops.
        """
        return self.__convergence_tol

    @convergence_tol.setter
    def convergence_tol(self, convergence_tol: float | None) -> None:
        """Set the velocity change below which the simulation stops.

        Parameters
        ----------
        convergence_tol : float or None
            Velocity change below which the simulation stops.

        Raises
        ------
        TypeError
            convergence_tol must be float or None.
        ValueError
            convergence_tol must be positive.
        """
        if convergence_tol is not None:
            if not isinstance(convergence_tol, numbers.Real):
                raise TypeError("convergence_tol must be float or None.")
            if not convergence_tol > 0:
                raise ValueError("convergence_tol must be positive.")
        self.__convergence_tol = convergence_tol

    # Advection

    @property
    def scipy_solve_advection(self) -> bool:
        """Whether the advection is solved with SciPy.

        Returns
        -------
        bool
            Whether the advection is solved with SciPy.
        """
        return self.__scipy_solve_advection

    @scipy_solve_advection.setter
    def scipy_solve_advection(self, scipy_solve_advection: bool) -> None:
        """Set whether the advection is solved with SciPy.

        Parameters
        ----------
        scipy_solve_advection : bool
            Whether the advection is solved with SciPy.

        Raises
        ------
        TypeError
            scipy_solve_advection must be bool.
        """
        if not isinstance(scipy_solve_advection, bool):
            raise TypeError("scipy_solve_advection must be bool.")
        self.__scipy_solve_advection = scipy_solve_advection

    @property
    def advection_use_BiCG(self) -> bool:
        """Whether the advection is solved with BiCGStab.

        Returns
        -------
        bool
            Whether the advection is solved with BiCGStab.
        """
        return self.__advection_use_BiCG

    @advection_use_BiCG.setter
    def advection_use_BiCG(self, advection_use_BiCG: bool) -> None:
        """Set whether the advection is solved with BiCGStab.

        Parameters
        ----------
        advection_use_BiCG : bool
            Whether the advection is solved with BiCGStab.

        Raises
        ------
        TypeError
            advection_use_BiCG must be bool.
        """
        if not isinstance(advection_use_BiCG, bool):
            raise TypeError("advection_use_BiCG must be bool.")
        self.__advection_use_BiCG = advection_use_BiCG

    @property
    def advect_non_ortho_steps(self) -> int:
        """Non-orthogonal correction steps of the advection.

        Returns
        -------
        int
            Non-orthogonal correction steps of the advection.
        """
        return self.__advect_non_ortho_steps

    @advect_non_ortho_steps.setter
    def advect_non_ortho_steps(self, advect_non_ortho_steps: int) -> None:
        """Set the number of non-orthogonal correction steps of the advection.

        Parameters
        ----------
        advect_non_ortho_steps : int
            Non-orthogonal correction steps of the advection.

        Raises
        ------
        TypeError
            advect_non_ortho_steps must be integral type.
        ValueError
            advect_non_ortho_steps must not be negative.
        """
        if not isinstance(advect_non_ortho_steps, numbers.Integral):
            raise TypeError("advect_non_ortho_steps must be integral type.")
        if advect_non_ortho_steps < 0:
            raise ValueError("advect_non_ortho_steps must not be negative.")
        self.__advect_non_ortho_steps: int = advect_non_ortho_steps

    @property
    def advection_tol(self) -> float | SolverTolerance | None:
        """Advection solver tolerance, the dtype default if unset.

        Returns
        -------
        float or SolverTolerance or None
            Advection solver tolerance, the dtype default if unset.
        """
        if self.__advection_tol is None:
            return piso_diff._get_solver_tolerance(dtype=self.__get_dtype())
        else:
            return self.__advection_tol

    @advection_tol.setter
    def advection_tol(self, advection_tol: float | SolverTolerance | None) -> None:
        """Set the advection solver tolerance, or None for the dtype default.

        Parameters
        ----------
        advection_tol : float or SolverTolerance or None
            Advection solver tolerance, the dtype default if unset.

        Raises
        ------
        TypeError
            advection_tol must be float, SolverTolerance or None.
        ValueError
            advection_tol must be positive.
        """
        if advection_tol is not None and not isinstance(advection_tol, SolverTolerance):
            if not isinstance(advection_tol, numbers.Real):
                raise TypeError("advection_tol must be float, SolverTolerance or None.")
            if not advection_tol > 0:
                raise ValueError("advection_tol must be positive.")
        self.__advection_tol = advection_tol

    # Pressure

    @property
    def scipy_solve_pressure(self) -> bool:
        """Whether the pressure is solved with SciPy.

        Returns
        -------
        bool
            Whether the pressure is solved with SciPy.
        """
        return self.__scipy_solve_pressure

    @scipy_solve_pressure.setter
    def scipy_solve_pressure(self, scipy_solve_pressure: bool) -> None:
        """Set whether the pressure is solved with SciPy.

        Parameters
        ----------
        scipy_solve_pressure : bool
            Whether the pressure is solved with SciPy.

        Raises
        ------
        TypeError
            scipy_solve_pressure must be bool.
        """
        if not isinstance(scipy_solve_pressure, bool):
            raise TypeError("scipy_solve_pressure must be bool.")
        self.__scipy_solve_pressure = scipy_solve_pressure

    @property
    def pressure_use_BiCG(self) -> bool:
        """Whether the pressure is solved with BiCGStab.

        Returns
        -------
        bool
            Whether the pressure is solved with BiCGStab.
        """
        return self.__pressure_use_BiCG

    @pressure_use_BiCG.setter
    def pressure_use_BiCG(self, pressure_use_BiCG: bool) -> None:
        """Set whether the pressure is solved with BiCGStab.

        Parameters
        ----------
        pressure_use_BiCG : bool
            Whether the pressure is solved with BiCGStab.

        Raises
        ------
        TypeError
            pressure_use_BiCG must be bool.
        """
        if not isinstance(pressure_use_BiCG, bool):
            raise TypeError("pressure_use_BiCG must be bool.")
        self.__pressure_use_BiCG = pressure_use_BiCG

    @property
    def pressure_non_ortho_steps(self) -> int:
        """Non-orthogonal correction steps of the pressure.

        Returns
        -------
        int
            Non-orthogonal correction steps of the pressure.
        """
        return self.__pressure_non_ortho_steps

    @pressure_non_ortho_steps.setter
    def pressure_non_ortho_steps(self, pressure_non_ortho_steps: int) -> None:
        """Set the number of non-orthogonal correction steps of the pressure.

        Parameters
        ----------
        pressure_non_ortho_steps : int
            Non-orthogonal correction steps of the pressure.

        Raises
        ------
        TypeError
            pressure_non_ortho_steps must be integral type.
        ValueError
            pressure_non_ortho_steps must not be negative.
        """
        if not isinstance(pressure_non_ortho_steps, numbers.Integral):
            raise TypeError("pressure_non_ortho_steps must be integral type.")
        if pressure_non_ortho_steps < 0:
            raise ValueError("pressure_non_ortho_steps must not be negative.")
        self.__pressure_non_ortho_steps: int = pressure_non_ortho_steps

    @property
    def pressure_tol(self) -> float | SolverTolerance | torch.Tensor:
        """Pressure solver tolerance, the dtype default (as tensor) if unset.

        Returns
        -------
        float or SolverTolerance or torch.Tensor
            Pressure solver tolerance, the dtype default (as tensor) if unset.
        """
        if self.__pressure_tol is None:
            return piso_diff._get_solver_tolerance_torch(dtype=self.__get_dtype())
        else:
            return self.__pressure_tol

    @pressure_tol.setter
    def pressure_tol(self, pressure_tol: float | SolverTolerance | None) -> None:
        """Set the pressure solver tolerance, or None for the dtype default.

        Parameters
        ----------
        pressure_tol : float or SolverTolerance or None
            Pressure solver tolerance, the dtype default (as tensor) if unset.

        Raises
        ------
        TypeError
            pressure_tol must be float, SolverTolerance or None.
        ValueError
            pressure_tol must be positive.
        """
        if pressure_tol is not None and not isinstance(pressure_tol, SolverTolerance):
            if not isinstance(pressure_tol, numbers.Real):
                raise TypeError("pressure_tol must be float, SolverTolerance or None.")
            if not pressure_tol > 0:
                raise ValueError("pressure_tol must be positive.")
        self.__pressure_tol = pressure_tol

    @property
    def pressure_tol_intermediate(self) -> float | SolverTolerance | None:
        """Tolerance for every pressure solve that is not the final corrector.

        ``None`` (the default) means every corrector uses ``pressure_tol``, i.e.
        the historical behaviour. Only the last corrector's pressure survives into
        the solution.

        Returns
        -------
        float or SolverTolerance or None
            Tolerance for every pressure solve that is not the final corrector.
        """
        return self.__pressure_tol_intermediate

    @pressure_tol_intermediate.setter
    def pressure_tol_intermediate(self, value: float | SolverTolerance | None) -> None:
        """Set the tolerance for every pressure solve that is not the final corrector.

        Parameters
        ----------
        value : float or SolverTolerance or None
            Tolerance for every pressure solve that is not the final corrector.

        Raises
        ------
        TypeError
            pressure_tol_intermediate must be float, SolverTolerance or None.
        ValueError
            pressure_tol_intermediate must be positive.
        """
        if value is not None and not isinstance(value, SolverTolerance):
            if not isinstance(value, numbers.Real):
                raise TypeError(
                    "pressure_tol_intermediate must be float, SolverTolerance or None."
                )
            if not value > 0:
                raise ValueError("pressure_tol_intermediate must be positive.")
        self.__pressure_tol_intermediate = value

    @property
    def pressure_warm_start(self) -> bool:
        """Seed each pressure solve with the previous sub-step's result.

        Off by default.

        Returns
        -------
        bool
            Seed each pressure solve with the previous sub-step's result.
        """
        return self.__pressure_warm_start

    @pressure_warm_start.setter
    def pressure_warm_start(self, value: bool) -> None:
        """Set whether each pressure solve is seeded with the previous sub-step's result.

        Parameters
        ----------
        value : bool
            Seed each pressure solve with the previous sub-step's result.

        Raises
        ------
        TypeError
            pressure_warm_start must be bool.
        """
        if not isinstance(value, bool):
            raise TypeError("pressure_warm_start must be bool.")
        self.__pressure_warm_start = value

    def _solve_is_differentiated(self) -> bool:
        """Whether a linear solve issued right now would record a graph.

        Returns
        -------
        bool
            True if the autograd backend is in use and gradients are enabled.
        """
        return self.differentiable and torch.is_grad_enabled()

    def _pressure_tol_for(
        self, cstep: int, pstep: int, corrector_steps: int
    ) -> float | SolverTolerance | torch.Tensor:
        """``pressure_tol`` on the final corrector, the intermediate one before.

        Parameters
        ----------
        cstep : int
            Index of the current corrector step.
        pstep : int
            Index of the current non-orthogonal pressure step.
        corrector_steps : int
            Total number of corrector steps of this PISO step.

        Returns
        -------
        float or SolverTolerance or torch.Tensor
            :attr:`pressure_tol` on the final solve, and
            :attr:`pressure_tol_intermediate` on every earlier one, or
            :attr:`pressure_tol` throughout if no intermediate tolerance is set.
        """
        if self.__pressure_tol_intermediate is None:
            return self.pressure_tol
        is_final = (cstep == corrector_steps - 1) and (
            pstep == self.pressure_non_ortho_steps - 1
        )
        return self.pressure_tol if is_final else self.__pressure_tol_intermediate

    @staticmethod
    def _as_guess(t: torch.Tensor | None) -> torch.Tensor | None:
        """A detached, finite initial iterate for a linear solve, or ``None``.

        Parameters
        ----------
        t : torch.Tensor or None
            Candidate initial iterate.

        Returns
        -------
        torch.Tensor or None
            ``t`` detached from the graph, or None if it is missing, empty or
            not finite throughout.
        """
        if t is None or not torch.is_tensor(t) or t.numel() == 0:
            return None
        if not torch.isfinite(t).all():
            return None
        return t.detach()

    def _pressure_initial_guess(
        self, cstep: int, reference: torch.Tensor
    ) -> torch.Tensor | None:
        """Cached guess for corrector ``cstep``, or ``None`` if unusable.

        Offered on the differentiable path too: ``linear_solve_GPU`` takes the
        guess as a detached constant, so it changes how fast the solve converges
        and not what it converges to, and it keeps the differentiated forward on
        the same trajectory as an evaluation run.

        Parameters
        ----------
        cstep : int
            Index of the corrector step whose cached guess is requested.
        reference : torch.Tensor
            Tensor whose shape, dtype and device the guess must match.

        Returns
        -------
        torch.Tensor or None
            The cached guess on ``reference``'s device and dtype, or None if
            warm starting is off or no compatible guess is cached.
        """
        if not self.__pressure_warm_start:
            return None
        guess = self.__pressure_guess.get(cstep, None)
        if guess is None or guess.shape != reference.shape:
            return None
        return guess.to(dtype=reference.dtype, device=reference.device)

    def _pressure_rank_deficient(self, P: _C.CSRmatrix | None) -> bool:
        """Check whether the pressure Poisson operator has the constant nullspace.

        Parameters
        ----------
        P : phipict._C.CSRmatrix or None
            Assembled pressure matrix, or None if it has not been built yet.

        Returns
        -------
        bool
            True if every row sum vanishes, i.e. the operator is pure Neumann.
            False if ``P`` is None.
        """
        if P is None:
            return False
        cached = getattr(self, "_pressure_rank_deficient_cache", None)
        if cached is not None and cached[0] is P:
            return cached[1]

        # batched environments share the pattern, and the nullspace is a property
        # of the boundary conditions: the first environment's values decide
        value = P.value.reshape(-1)[: P.getNnz()]
        row = P.row
        n = row.numel() - 1
        counts = (row[1:] - row[:-1]).to(torch.int64).to(value.device)
        segment = torch.repeat_interleave(torch.arange(n, device=value.device), counts)
        row_sum = torch.zeros(n, dtype=value.dtype, device=value.device).index_add_(
            0, segment, value
        )
        row_abs = torch.zeros(n, dtype=value.dtype, device=value.device).index_add_(
            0, segment, value.abs()
        )
        # Relative to each row's magnitude and the dtype's round-off, so a pure
        # Neumann matrix assembled in float32 is still recognised
        floor = PRECISION_FLOOR_FACTOR * torch.finfo(value.dtype).eps
        deficient = bool((row_sum.abs() <= floor * row_abs).all().item())
        self._pressure_rank_deficient_cache = (P, deficient)
        self.__LOG.debug(
            "Pressure matrix is %s; the adjoint solve %s project out the "
            "constant mode.",
            "singular (pure Neumann)" if deficient else "anchored (full rank)",
            "will" if deficient else "will not",
        )
        return deficient

    def _get_pressure_amg_hierarchy(
        self, P: _C.CSRmatrix
    ) -> AMGHierarchy | BatchedAMGHierarchy | None:
        """AMG hierarchy of the current pressure matrix, or None if AMG is off.

        The hierarchy is refreshed (Galerkin products with the cached
        interpolation) whenever the values of ``P`` changed, which is detected
        from a fingerprint of the values. Within a PISO step the correctors
        re-assemble identical matrices, so there is one refresh per step.

        Parameters
        ----------
        P : phipict._C.CSRmatrix
            Assembled pressure matrix.

        Returns
        -------
        AMGHierarchy or None
            Hierarchy whose finest level equals ``P``, or None if
            ``pressure_use_amg`` is off or not applicable.
        """
        if self.pressure_use_BiCG or self.scipy_solve_pressure:
            return None
        use_amg = self.pressure_use_amg
        batch = P.getBatchSize()
        if batch > 1:
            return self._get_batched_pressure_amg_hierarchy(P, use_amg)
        if use_amg is None:
            if not (
                P.getRows() >= self.PRESSURE_AMG_AUTO_MIN_ROWS and is_pyamg_available()
            ):
                return None
        elif not use_amg:
            return None
        elif not is_pyamg_available():
            raise ImportError(
                "pressure_use_amg needs the 'pyamg' package for the AMG setup. "
                "Install it with `pip install pyamg`."
            )
        fingerprint = (P.getRows(), P.getSize(), *values_fingerprint(P.value))
        if (
            self._pressure_amg_hierarchy is not None
            and self._pressure_amg_fingerprint == fingerprint
        ):
            return self._pressure_amg_hierarchy
        options = dict(self._pressure_amg_options)
        interval = int(options.pop("coarse_refresh_interval", 4))
        refreshes = self._pressure_amg_refreshes
        stale = self._pressure_amg_stale
        with SAMPLE("pressure AMG refresh"):
            previous = self._pressure_amg_hierarchy
            if (
                isinstance(previous, AMGHierarchy)
                and interval > 1
                and refreshes % interval != 0
                and not stale
                and previous.levels[0].A.shape[0] == P.getRows()
            ):
                # only the finest level follows the new operator; the coarse
                # operators of the last full refresh are kept
                self._pressure_amg_hierarchy = galerkin_refresh(
                    previous, P, coarse=False
                )
            else:
                # free the old operators first: the full refresh does not need them
                previous = self._pressure_amg_hierarchy = None
                self._pressure_amg_hierarchy = hierarchy_for(
                    P,
                    project_constant=self._pressure_rank_deficient(P),
                    **options,
                )
                self._pressure_amg_refreshes = 0
                self._pressure_amg_stale = False
                self._pressure_amg_baseline_iters = None
                refreshes = 0
        self._pressure_amg_refreshes = refreshes + 1
        self._pressure_amg_fingerprint = fingerprint
        return self._pressure_amg_hierarchy

    def _get_batched_pressure_amg_hierarchy(
        self, P: _C.CSRmatrix, use_amg: bool | None
    ) -> BatchedAMGHierarchy | None:
        """Batched counterpart of :meth:`_get_pressure_amg_hierarchy`.

        All environments share the interpolation; their coarse operators are
        refreshed together (one sparse-dense product per level). In auto mode
        (``use_amg is None``) the AMG is used when the pressure systems of all
        environments together have at least
        :attr:`PRESSURE_AMG_AUTO_MIN_ROWS_BATCHED` unknowns.

        Parameters
        ----------
        P : phipict._C.CSRmatrix
            Batched pressure matrix.
        use_amg : bool or None
            ``pressure_use_amg``.

        Returns
        -------
        BatchedAMGHierarchy or None
            The refreshed hierarchy, or None if AMG is not used.
        """
        batch = P.getBatchSize()
        if use_amg is None:
            use_amg = (
                P.getRows() * batch >= self.PRESSURE_AMG_AUTO_MIN_ROWS_BATCHED
                and is_pyamg_available()
            )
        if not use_amg:
            return None
        if not is_pyamg_available():
            raise ImportError(
                "pressure_use_amg needs the 'pyamg' package for the AMG setup. "
                "Install it with `pip install pyamg`."
            )
        fingerprint = (P.getRows(), P.getSize(), *values_fingerprint(P.value))
        previous = self._pressure_amg_hierarchy
        if (
            isinstance(previous, BatchedAMGHierarchy)
            and self._pressure_amg_fingerprint == fingerprint
        ):
            return previous
        options = dict(self._pressure_amg_options)
        interval = int(options.pop("coarse_refresh_interval", 4))
        coarse = (
            not isinstance(previous, BatchedAMGHierarchy)
            or interval <= 1
            or self._pressure_amg_refreshes % interval == 0
            or self._pressure_amg_stale
        )
        with SAMPLE("pressure AMG refresh"):
            hierarchy = batched_hierarchy_for(
                P,
                batch,
                project_constant=self._pressure_rank_deficient(P),
                previous=previous
                if isinstance(previous, BatchedAMGHierarchy)
                else None,
                coarse=coarse,
                **options,
            )
        if coarse:
            self._pressure_amg_refreshes = 0
            self._pressure_amg_stale = False
            self._pressure_amg_baseline_iters = None
        self._pressure_amg_refreshes += 1
        self._pressure_amg_hierarchy = hierarchy
        self._pressure_amg_fingerprint = fingerprint
        return hierarchy

    def _note_pressure_amg_iterations(self, infos: Sequence[Any]) -> None:
        """Track pressure AMG iteration counts to detect stale coarse operators.

        The solves after a full refresh (all correctors of that step) set the
        baseline as their largest iteration count. A later solve
        that needs more than twice the baseline (plus a small margin) forces a
        full Galerkin refresh at the next change of the pressure matrix.

        Parameters
        ----------
        infos : sequence of LinearSolverResultInfo
            Result infos of a pressure solve.
        """
        iters = max(int(i.usedIterations) for i in infos)
        baseline = self._pressure_amg_baseline_iters
        if baseline is None or self._pressure_amg_refreshes <= 1:
            self._pressure_amg_baseline_iters = (
                iters if baseline is None else max(baseline, iters)
            )
        elif iters > 2 * baseline + 5:
            self._pressure_amg_stale = True

    def _store_pressure_guess(self, cstep: int, result: torch.Tensor) -> None:
        """Cache a corrector's pressure result as the next step's warm start.

        A no-op unless :attr:`pressure_warm_start` is set.

        Parameters
        ----------
        cstep : int
            Index of the corrector step the result belongs to.
        result : torch.Tensor
            Pressure result to cache. Stored detached and cloned.
        """
        if self.__pressure_warm_start:
            self.__pressure_guess[cstep] = result.detach().clone()

    def clear_pressure_guess(self) -> None:
        """Drop the warm-start cache, e.g. after the domain or dt changed."""
        self.__pressure_guess.clear()

    @property
    def pressure_guess_state(self) -> torch.Tensor | None:
        """The warm-start cache as one stacked tensor, or None if empty.

        Cross-step solver state, so a BPTT checkpoint has to carry it; see
        ``FluidEnv._checkpoint_accessors``.

        Returns
        -------
        torch.Tensor or None
            The warm-start cache as one stacked tensor, or None if empty.
        """
        if not self.__pressure_guess:
            return None
        keys = sorted(self.__pressure_guess)
        return torch.stack([self.__pressure_guess[k] for k in keys])

    @pressure_guess_state.setter
    def pressure_guess_state(self, value: torch.Tensor | None) -> None:
        """Set the warm-start cache as one stacked tensor, or None if empty.

        Parameters
        ----------
        value : torch.Tensor or None
            The warm-start cache as one stacked tensor, or None if empty.
        """
        if value is None:
            self.__pressure_guess.clear()
            return
        self.__pressure_guess = {k: value[k] for k in range(value.size(0))}

    @property
    def normalize_pressure_result(self) -> bool:
        """Whether the mean pressure is subtracted.

        Returns
        -------
        bool
            Whether the mean pressure is subtracted.
        """
        return self.__normalize_pressure_result

    @normalize_pressure_result.setter
    def normalize_pressure_result(self, normalize_pressure_result: bool) -> None:
        """Set whether the mean pressure is subtracted.

        Parameters
        ----------
        normalize_pressure_result : bool
            Whether the mean pressure is subtracted.

        Raises
        ------
        TypeError
            normalize_pressure_result must be bool.
        """
        if not isinstance(normalize_pressure_result, bool):
            raise TypeError("normalize_pressure_result must be bool.")
        self.__normalize_pressure_result = normalize_pressure_result

    @property
    def pressure_return_best_result(self) -> bool:
        """Whether a non-converged pressure solve returns its best iterate.

        Returns
        -------
        bool
            Whether a non-converged pressure solve returns its best iterate.
        """
        return self.__pressure_return_best_result

    @pressure_return_best_result.setter
    def pressure_return_best_result(self, pressure_return_best_result: bool) -> None:
        """Set whether a non-converged pressure solve returns its best iterate.

        Parameters
        ----------
        pressure_return_best_result : bool
            Whether a non-converged pressure solve returns its best iterate.

        Raises
        ------
        TypeError
            pressure_return_best_result must be bool.
        """
        if not isinstance(pressure_return_best_result, bool):
            raise TypeError("pressure_return_best_result must be bool.")
        self.__pressure_return_best_result = pressure_return_best_result

    @property
    def velocity_corrector(self) -> str:
        """Pressure gradient discretization of the velocity correction.

        Returns
        -------
        str
            Pressure gradient discretization of the velocity correction.
        """
        return self.__velocity_corrector

    @velocity_corrector.setter
    def velocity_corrector(self, velocity_corrector: str) -> None:
        """Set the pressure gradient discretization of the velocity correction.

        Parameters
        ----------
        velocity_corrector : str
            Pressure gradient discretization of the velocity correction.

        Raises
        ------
        ValueError
            If the value is invalid.
        """
        if velocity_corrector not in self._velocity_corrector_versions:
            raise ValueError(
                "velocity_corrector must be one of: %s"
                % (list(self._velocity_corrector_versions.keys()),)
            )
        self.__velocity_corrector = velocity_corrector
        self._velocity_corrector_version = self._velocity_corrector_versions[
            velocity_corrector
        ]

    # Logging/Output

    @property
    def log_dir(self) -> str | None:
        """Output directory, created on assignment.

        Returns
        -------
        str or None
            Output directory, created on assignment.
        """
        return self.__log_dir

    @log_dir.setter
    def log_dir(self, log_dir: str | None) -> None:
        """Set the output directory, creating it on assignment.

        Parameters
        ----------
        log_dir : str or None
            Output directory, created on assignment.
        """
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.__log_dir: str | None = log_dir
        else:
            self.__log_dir = None

    @property
    def log_interval(self) -> int:
        """Iterations between log outputs, 0 if disabled.

        Returns
        -------
        int
            Iterations between log outputs, 0 if disabled.
        """
        return self.__log_interval

    @log_interval.setter
    def log_interval(self, log_interval: int | None) -> None:
        """Set the number of iterations between log outputs, 0 to disable.

        Parameters
        ----------
        log_interval : int or None
            Iterations between log outputs, 0 if disabled.

        Raises
        ------
        TypeError
            log_interval must be integral type or None.
        ValueError
            log_interval must not be negative.
        """
        if log_interval is None:
            log_interval = 0
        if not isinstance(log_interval, numbers.Integral):
            raise TypeError("log_interval must be integral type or None.")
        if log_interval < 0:
            raise ValueError("log_interval must not be negative.")
        self.__log_interval: int = log_interval

    @property
    def log_images(self) -> bool:
        """Whether images are written.

        Returns
        -------
        bool
            Whether images are written.
        """
        return self.__log_images

    @log_images.setter
    def log_images(self, log_images: bool) -> None:
        """Set whether images are written.

        Parameters
        ----------
        log_images : bool
            Whether images are written.

        Raises
        ------
        TypeError
            log_images must be bool.
        """
        if not isinstance(log_images, bool):
            raise TypeError("log_images must be bool.")
        self.__log_images = log_images

    @property
    def norm_vel(self) -> bool:
        """Whether velocity images are normalized.

        Returns
        -------
        bool
            Whether velocity images are normalized.
        """
        return self.__norm_vel

    @norm_vel.setter
    def norm_vel(self, norm_vel: bool) -> None:
        """Set whether velocity images are normalized.

        Parameters
        ----------
        norm_vel : bool
            Whether velocity images are normalized.

        Raises
        ------
        TypeError
            norm_vel must be bool.
        """
        if not isinstance(norm_vel, bool):
            raise TypeError("norm_vel must be bool.")
        self.__norm_vel = norm_vel

    @property
    def block_layout(self) -> list | None:
        """Block arrangement of images.

        Returns
        -------
        list or None
            Block arrangement of images.
        """
        return self.__block_layout

    @block_layout.setter
    def block_layout(self, block_layout: list | None) -> None:
        """Set the block arrangement of images.

        Parameters
        ----------
        block_layout : list or None
            Block arrangement of images.

        Raises
        ------
        TypeError
            block_layout must be list or None.
        """
        if not (block_layout is None or isinstance(block_layout, list)):
            raise TypeError("block_layout must be list or None.")
        self.__block_layout = block_layout

    @property
    def stop_fn(self) -> Callable[[], bool]:
        """Function returning True to stop the simulation.

        Returns
        -------
        Callable[[], bool]
            Function returning True to stop the simulation.
        """
        return self.__stop_fn

    @stop_fn.setter
    def stop_fn(self, stop_fn: Callable[[], bool] | None) -> None:
        """Set the function returning True to stop the simulation.

        Parameters
        ----------
        stop_fn : Callable[[], bool] or None
            Function returning True to stop the simulation.

        Raises
        ------
        TypeError
            stop_fn must be callable or None.
        """
        if stop_fn is None:
            stop_fn = lambda: False
        if not callable(stop_fn):
            raise TypeError("stop_fn must be callable or None.")
        self.__stop_fn = stop_fn

    def _check_stop(self) -> bool:
        """Ask :attr:`stop_fn` whether the simulation should stop.

        Returns
        -------
        bool
            True if a stop function is set and returns True.
        """
        return self.stop_fn is not None and self.stop_fn()

    @property
    def hooks(self) -> Hooks:
        """The callbacks run at the hooks of every PISO step.

        Returns
        -------
        Hooks
            The callbacks; register more with ``sim.hooks.append(hook, fn)``.
        """
        return self.__hooks

    @hooks.setter
    def hooks(self, hooks: Hooks | Mapping[Hook | str, Any] | None) -> None:
        """Set the callbacks run at the hooks of every PISO step.

        Parameters
        ----------
        hooks : Hooks, Mapping or None
            The callbacks: a :class:`~phipict.core.hooks.Hooks` (used as is), a dict
            of callbacks by hook (name), or None for no callbacks.

        Raises
        ------
        TypeError
            If ``hooks`` is neither.
        ValueError
            If a key is not a :class:`~phipict.core.hooks.Hook`.
        """
        if not (hooks is None or isinstance(hooks, (Hooks, Mapping))):
            raise TypeError("hooks must be Hooks, a dict of callbacks or None.")
        self.__hooks = Hooks.coerce(hooks)

    @property
    def prep_fn(self) -> Hooks:
        """The former name of :attr:`hooks`, kept for compatibility."""
        return self.__hooks

    @prep_fn.setter
    def prep_fn(self, prep_fn: Hooks | Mapping[Hook | str, Any] | None) -> None:
        self.hooks = prep_fn

    def _run_prep_fn(self, name: Hook, **kwargs: Any) -> None:
        """Run the callbacks registered for a hook, if any.

        Parameters
        ----------
        name : Hook
            The hook.
        **kwargs : Any
            Keyword arguments forwarded to each callback.
        """
        self.__hooks.run(name, **kwargs)

    def reset_total_step(self) -> None:
        """Reset the total step counter to 0."""
        self.total_step = 0

    def reset_total_time(self) -> None:
        """Reset the total simulated time to 0."""
        self.total_time: float = 0

    def end_step(self, time_step: float | torch.Tensor = 0) -> None:
        """Advance the step counter and the simulated time.

        Parameters
        ----------
        time_step : float or torch.Tensor, optional
            Simulated time of the step. Default is 0.
        """
        self.total_step += 1
        self.total_time += ntonp(time_step)[0]

    def reset_step_counters(self) -> None:
        """Reset the step, time and image counters."""
        self.reset_total_step()
        self.reset_total_time()
        self.reset_image_out_idx()

    def linear_solve_scipy(
        self, csrMat: _C.CSRmatrix, rhs: torch.Tensor, x: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Solve ``A x = b`` with SciPy's direct sparse solver on the host.

        Parameters
        ----------
        csrMat : phipict._C.CSRmatrix
            System matrix.
        rhs : torch.Tensor
            Right-hand side.
        x : torch.Tensor or None, optional
            Unused. Default is None.

        Returns
        -------
        torch.Tensor
            Solution on the GPU.
        """
        with SAMPLE("scipy linear solve"):
            A = csr_matrix(  # type: ignore[call-overload]
                (csrMat.value.cpu(), csrMat.index.cpu(), csrMat.row.cpu()),
                shape=[csrMat.getRows()] * 2,
            )
            with SAMPLE("linear solve"):
                x_np = spsolve(A, torch.flatten(rhs.cpu()))
            return torch.reshape(torch.tensor(x_np, dtype=rhs.dtype), rhs.size()).cuda()

    def linear_solve_GPU(
        self,
        csrMat: _C.CSRmatrix,
        rhs: torch.Tensor,
        x: torch.Tensor | None = None,
        use_BiCG: bool = False,
        tol: float | torch.Tensor | None = None,
        max_iter: int = 5000,
        matrix_rank_deficient: bool = False,
        residual_reset_step: int = 0,
        return_best_result: bool = False,
        double_fallback: bool = False,
        BiCG_with_preconditioner: bool = True,
        BiCG_precondition_fallback: bool = False,
        adjoint_rank_deficient: bool = False,
        tag: str = "unknown",
        amg_hierarchy: AMGHierarchy | BatchedAMGHierarchy | None = None,
        fwd_rtol: float | None = None,
    ) -> tuple[torch.Tensor, bool]:
        """Solve ``A x = b`` with the CUDA solvers.

        Uses the differentiable :func:`phipict.core.piso_diff.linear_solve_GPU`
        if the solve is differentiated, and an in-place solve otherwise.

        Parameters
        ----------
        csrMat : phipict._C.CSRmatrix
            System matrix.
        rhs : torch.Tensor
            Right-hand side.
        x : torch.Tensor or None, optional
            Initial guess. Default is None (zero).
        use_BiCG : bool, optional
            Whether to use BiCGStab instead of CG. Default is False.
        tol : float, torch.Tensor or None, optional
            Absolute tolerance. None uses the dtype default. Default is None.
        max_iter : int, optional
            Maximum number of iterations. Default is 5000.
        matrix_rank_deficient : bool, optional
            Rank deficiency flag of the CUDA solver. Default is False.
        residual_reset_step : int, optional
            Interval of residual recomputation. Default is 0.
        return_best_result : bool, optional
            Whether a non-converged solve returns its best iterate.
            Default is False.
        double_fallback : bool, optional
            Whether to retry a failed single precision solve in double
            precision. Default is False.
        BiCG_with_preconditioner : bool, optional
            Whether BiCGStab uses a preconditioner. Default is True.
        BiCG_precondition_fallback : bool, optional
            Whether to retry a failed BiCGStab solve with a preconditioner.
            Default is False.
        adjoint_rank_deficient : bool, optional
            Whether the adjoint solve projects out the constant nullspace.
            Default is False.
        tag : str, optional
            Label for solver statistics. Default is ``"unknown"``.
        amg_hierarchy : AMGHierarchy or None, optional
            AMG preconditioner for differentiated solves. Default is None.
        fwd_rtol : float or None, optional
            The ``rtol`` ``tol`` was resolved from, which the adjoint of a
            differentiated solve is held to by default; None if ``tol`` is
            absolute. Default is None.

        Returns
        -------
        x : torch.Tensor
            Solution.
        solve_ok : bool
            Always True; failures raise :class:`~phipict.core.piso_diff.LinsolveError`.
        """
        if not self._solve_is_differentiated():
            convergence_criterion = (
                _C.ConvergenceCriterion.NORM2_NORMALIZED
            )  # RMSE of residual vector
            transpose = False  # used for backprop
            print_residual = False  # Print final solver residual information. Use for debugging. Only for CG.
            # return_best_result = False # saves best intermediate result and returns it instead of final result after max_iter if the solve does not converge. Only for CG.
            maxit_torch = torch.IntTensor([max_iter])
            tol_torch = piso_diff._get_solver_tolerance_torch(
                tol, rhs.dtype
            )  # torch.ones([1], dtype = rhs.dtype)*tol
            with SAMPLE("GPU linear solve"):
                if x is None:
                    x = torch.zeros_like(rhs)
                elif self.differentiable:
                    # The kernel writes the iterate through `x`'s data pointer,
                    # which autograd's version counter cannot see. On a
                    # differentiable run the caller's buffer (e.g.
                    # `domain.velocityResult`) may already be saved for a
                    # backward, so solve into a private copy instead. Bit-identical
                    # on a non-differentiable run, where no copy is made at all.
                    x = x.detach().clone()

                piso_diff._linear_solve_wrapper(
                    csrMat,
                    rhs,
                    x,
                    maxit_torch,
                    tol_torch,
                    convergence_criterion,
                    use_BiCG,
                    matrix_rank_deficient,
                    residual_reset_step,
                    transpose,
                    print_residual,
                    return_best_result,
                    double_fallback=double_fallback,
                    debug_out=False,
                    BiCG_with_preconditioner=BiCG_with_preconditioner,
                    BiCG_precondition_fallback=BiCG_precondition_fallback,
                    tag=tag,
                )

                return x, True
        else:
            return (
                self.__backend.linear_solve_GPU(
                    csrMat,
                    rhs,
                    transpose=False,
                    use_BiCG=use_BiCG,
                    tol=tol,
                    max_iter=max_iter,
                    return_best_result=return_best_result,
                    double_fallback=double_fallback,
                    BiCG_with_preconditioner=BiCG_with_preconditioner,
                    BiCG_precondition_fallback=BiCG_precondition_fallback,
                    adjoint_rank_deficient=adjoint_rank_deficient,
                    tag=tag,
                    x0=self._as_guess(x),
                    amg_hierarchy=amg_hierarchy,
                    matrix_rank_deficient=matrix_rank_deficient,
                    residual_reset_step=residual_reset_step,
                    fwd_rtol=fwd_rtol,
                    env_batch=self.domain.getBatchSize()
                    if self.domain is not None
                    else 1,
                ),
                True,
            )

    def linear_solve_AMG(
        self,
        rhs: torch.Tensor,
        x: torch.Tensor | None,
        amg_hierarchy: AMGHierarchy | BatchedAMGHierarchy,
        tol: float | torch.Tensor,
        max_iter: int,
        return_best_result: bool = False,
        tag: str = "unknown",
    ) -> tuple[torch.Tensor, bool]:
        """Solve with AMG-preconditioned CG (host setup, on-device V-cycle).

        The operator is the finest level of ``amg_hierarchy``, not ``csrMat``:
        the hierarchy was built from that matrix, so this guarantees the Krylov
        iteration and the preconditioner cannot disagree about the operator.

        Emits the same telemetry and raises the same errors as the CUDA path --
        the result infos duck-type ``LinearSolverResultInfo``, so
        ``_check_solver_return_infos`` and ``solver_stats`` handle them unchanged.

        Parameters
        ----------
        rhs : torch.Tensor
            Right-hand side.
        x : torch.Tensor or None
            Initial guess, updated in place. None starts from zero.
        amg_hierarchy : AMGHierarchy
            Multigrid hierarchy of the system matrix.
        tol : float or torch.Tensor
            Absolute tolerance.
        max_iter : int
            Maximum number of iterations.
        return_best_result : bool, optional
            Whether a non-converged solve returns its best iterate.
            Default is False.
        tag : str, optional
            Label for solver statistics. Default is ``"unknown"``.

        Returns
        -------
        x : torch.Tensor
            Solution.
        solve_ok : bool
            Always True; failures raise :class:`~phipict.core.piso_diff.LinsolveError`.
        """
        # a tensor with several entries holds one tolerance per right-hand side
        # (batched environments)
        tol_arg: float | list[float]
        if isinstance(tol, torch.Tensor) and tol.numel() > 1:
            tol_arg = [float(t) for t in tol.detach().cpu().reshape(-1)]
        else:
            tol_arg = float(
                tol
                if not isinstance(tol, torch.Tensor)
                else tol.detach().cpu().reshape(-1)[0]
            )
        tol_value = max(tol_arg) if isinstance(tol_arg, list) else tol_arg
        if x is None:
            x = torch.zeros_like(rhs)

        probe = solver_stats.Probe(rhs, tag) if solver_stats.is_active() else None
        if probe is not None:
            probe.start()

        with SAMPLE("AMG linear solve"):
            solver_infos = amg_pcg_solve(
                rhs,
                x,
                amg_hierarchy,
                tol=tol_arg,
                max_iter=max_iter,
                return_best_result=return_best_result,
            )
        if amg_hierarchy is self._pressure_amg_hierarchy:
            self._note_pressure_amg_iterations(solver_infos)

        if probe is not None:
            probe.stop(
                solver_infos,
                tolerance=tol_value,
                max_iterations=int(max_iter),
                use_BiCG=False,
            )

        piso_diff._check_solver_return_infos(
            solver_infos,
            False,
            False,
            tol_value,
            max_iter,
            return_best_result,
            is_FWD=None,
            debug_out=False,
        )

        return x, True

    def linear_solve(
        self,
        csrMat: _C.CSRmatrix,
        rhs: torch.Tensor,
        x: torch.Tensor | None = None,
        use_BiCG: bool = False,
        tol: float | SolverTolerance | torch.Tensor | None = None,
        max_iter: int | None = None,
        matrix_rank_deficient: bool = False,
        residual_reset_step: int = 0,
        use_scipy: bool = False,
        return_best_result: bool = False,
        amg_hierarchy: AMGHierarchy | BatchedAMGHierarchy | None = None,
        adjoint_rank_deficient: bool = False,
        tag: str = "unknown",
    ) -> tuple[torch.Tensor, bool]:
        """Solve ``A x = b`` with the configured solver.

        Relative tolerances are resolved against ``rhs`` here. Non-differentiated
        solves with an AMG hierarchy use :meth:`linear_solve_AMG`, all others
        :meth:`linear_solve_GPU` (or SciPy if requested).

        Parameters
        ----------
        csrMat : phipict._C.CSRmatrix
            System matrix.
        rhs : torch.Tensor
            Right-hand side.
        x : torch.Tensor or None, optional
            Initial guess. Default is None (zero).
        use_BiCG : bool, optional
            Whether to use BiCGStab instead of CG. Default is False.
        tol : float, SolverTolerance, torch.Tensor or None, optional
            Tolerance. None uses the dtype default. Default is None.
        max_iter : int or None, optional
            Maximum number of iterations. None uses
            ``linear_solve_max_iterations``. Default is None.
        matrix_rank_deficient : bool, optional
            Rank deficiency flag of the CUDA solver. Default is False.
        residual_reset_step : int, optional
            Interval of residual recomputation. Default is 0.
        use_scipy : bool, optional
            Whether to use :meth:`linear_solve_scipy`. Default is False.
        return_best_result : bool, optional
            Whether a non-converged solve returns its best iterate.
            Default is False.
        amg_hierarchy : AMGHierarchy or None, optional
            AMG preconditioner for a symmetric matrix. Default is None.
        adjoint_rank_deficient : bool, optional
            Whether the adjoint solve projects out the constant nullspace.
            Default is False.
        tag : str, optional
            Label for solver statistics. Default is ``"unknown"``.

        Returns
        -------
        x : torch.Tensor
            Solution.
        solve_ok : bool
            Whether the solve succeeded.
        """
        if use_scipy:
            return self.linear_solve_scipy(csrMat, rhs, x), True
        else:
            if max_iter == None:  # noqa: E711
                max_iter = self.linear_solve_max_iterations

            # Single choke point for every solve in the codebase: a relative
            # SolverTolerance is turned into the absolute number the kernels
            # expect here, against this solve's own RHS. Floats pass through
            # untouched, so absolute tolerances keep their exact behaviour
            batch = self.domain.getBatchSize() if self.domain is not None else 1
            
            # batched environments share the tolerance settings, but a relative
            # tolerance is resolved against each environment's own RHS
            abs_tol: Any = resolve_tolerance(tol, rhs, batch_size=batch)
            if isinstance(abs_tol, list):
                # one tolerance per batched environment, repeated for the right-hand
                # sides of an environment (e.g. its velocity components)
                rows = rhs.numel() // csrMat.getRows()
                abs_tol = torch.tensor(abs_tol, dtype=torch.float64).repeat_interleave(
                    rows // batch
                )

            if amg_hierarchy is not None and not self._solve_is_differentiated():
                assert abs_tol is not None
                return self.linear_solve_AMG(
                    rhs,
                    x,
                    amg_hierarchy,
                    tol=abs_tol,
                    max_iter=max_iter,
                    return_best_result=return_best_result,
                    tag=tag,
                )
            return self.linear_solve_GPU(
                csrMat,
                rhs,
                x,
                use_BiCG,
                abs_tol,
                max_iter,
                matrix_rank_deficient,
                residual_reset_step,
                return_best_result,
                double_fallback=self.solver_double_fallback,
                BiCG_with_preconditioner=self.preconditionBiCG,
                BiCG_precondition_fallback=self.BiCG_precondition_fallback,
                adjoint_rank_deficient=adjoint_rank_deficient,
                tag=tag,
                amg_hierarchy=amg_hierarchy,
                # The adjoint is held to the same relative residual as the
                # forward, see `phipict.solvers.tolerance.set_adjoint_rtol`
                fwd_rtol=tol.rtol if isinstance(tol, SolverTolerance) else None,
            )

    def advect_static(
        self, iterations: int | torch.Tensor, time_step: torch.Tensor | None = None
    ) -> bool:
        """Advect only the passive scalar in the current velocity field.

        Parameters
        ----------
        iterations : int or torch.Tensor
            Number of steps.
        time_step : torch.Tensor or None, optional
            Time step. None uses :attr:`time_step`. Default is None.

        Returns
        -------
        bool
            Whether all solves succeeded.

        Raises
        ------
        ValueError
            If the domain has no passive scalar.
        """
        self._check_domain()
        solve_ok = True
        advect_non_ortho_reuse_result = True
        # Per-channel lists or joint tensors, depending on the branch.
        x: Any
        scalarResult: Any

        if isinstance(iterations, torch.Tensor):
            iterations = iterations.numpy()[0]

        if time_step is None:
            time_step = self.__get_time_step_torch()
        domain = self.domain
        non_ortho_flags = self.__non_ortho_flags

        if not domain.hasPassiveScalar():
            raise ValueError("Domain has no passive scalar to advect.")

        split_scalar_channels = not (
            domain.isPassiveScalarViscosityStatic()
            and domain.isAllFixedBoundariesPassiveScalarTypeStatic()
        )
        scalar_channels = domain.getPassiveScalarChannels()

        for step in range(iterations):
            with SAMPLE("advect static"):
                self._run_prep_fn(
                    Hook.PRE,
                    domain=domain,
                    local_step=step,
                    time_step=time_step,
                    total_step=self.total_step,
                    total_time=self.total_time,
                )

                domain.UpdateDomainData()

                if split_scalar_channels:
                    matrices = []
                    for channel in range(scalar_channels):
                        self.__backend.SetupAdvectionMatrix(
                            domain,
                            time_step,
                            non_ortho_flags,
                            forPassiveScalar=True,
                            passiveScalarChannel=channel,
                        )
                        matrices.append(domain.C.clone())
                else:
                    self.__backend.SetupAdvectionMatrix(
                        domain,
                        time_step,
                        non_ortho_flags,
                        forPassiveScalar=True,
                        passiveScalarChannel=0,
                    )
                self.__backend.CopyScalarResultFromBlocks(
                    domain
                )  # needed for non-ortho components on RHS

                for no_step in range(self.advect_non_ortho_steps):
                    self.__backend.SetupAdvectionScalar(
                        domain, time_step, non_ortho_flags
                    )  # creates RHS for all channels

                    self._run_prep_fn(
                        Hook.POST_SCALAR_SETUP,
                        domain=domain,
                        no_step=no_step,
                        local_step=step,
                        time_step=time_step,
                        total_step=self.total_step,
                        total_time=self.total_time,
                    )

                    with (
                        torch.no_grad()
                        if self.exclude_advection_solve_gradients
                        else nullcontext()
                    ):
                        if split_scalar_channels:
                            if no_step == 0 or not advect_non_ortho_reuse_result:
                                x = [None] * scalar_channels
                            else:
                                x = _split_channels(
                                    domain.scalarResult,
                                    domain.getBatchSize(),
                                    scalar_channels,
                                )
                            RHS = _split_channels(
                                domain.scalarRHS, domain.getBatchSize(), scalar_channels
                            )
                            scalarResult = []
                            for channel in range(scalar_channels):
                                sR, solve_ok = self.linear_solve(
                                    matrices[channel],
                                    RHS[channel],
                                    x=x[channel],
                                    use_BiCG=self.advection_use_BiCG,
                                    use_scipy=self.scipy_solve_advection,
                                    tol=self.advection_tol,
                                    tag="scalar",
                                )
                                scalarResult.append(sR)
                            del x
                            del RHS
                            scalarResult = _join_channels(
                                scalarResult, domain.getBatchSize()
                            )
                        else:
                            x = (
                                None
                                if (no_step == 0 or not advect_non_ortho_reuse_result)
                                else domain.scalarResult
                            )
                            scalarResult, solve_ok = self.linear_solve(
                                domain.C,
                                domain.scalarRHS,
                                x=x,
                                use_BiCG=self.advection_use_BiCG,
                                use_scipy=self.scipy_solve_advection,
                                tol=self.advection_tol,
                                tag="scalar",
                            )
                            del x

                    domain.setScalarResult(scalarResult)
                    domain.UpdateDomainData()

                    if not solve_ok or self._check_stop():
                        return solve_ok

                if split_scalar_channels:
                    del matrices

                self.__backend.CopyScalarResultToBlocks(
                    domain
                )  # set final result to blocks for next iteration/step

                if not solve_ok or self._check_stop():
                    return solve_ok

                self.end_step(time_step)

        return solve_ok

    def make_divergence_free(
        self, iterations: int | torch.Tensor = 1, max_iter: int = 1000
    ) -> bool:
        """Run pressure corrections to make the velocity divergence free.

        Parameters
        ----------
        iterations : int or torch.Tensor, optional
            Number of corrections. Default is 1.
        max_iter : int, optional
            Maximum iterations per pressure solve. Default is 1000.

        Returns
        -------
        bool
            Whether all solves succeeded.
        """
        self._check_domain()
        domain = self.domain
        own_pressure_rhs = domain.pressureRHS
        try:
            return self._make_divergence_free(iterations, max_iter)
        finally:
            # The projection corrects the velocity in place through a pressure RHS
            # that *is* velocityResult. A PISO step must not inherit that: it builds
            # the pressure RHS from neighbouring velocityResult entries while writing
            # it, which races and made results depend on thread scheduling (and
            # batched environments differ from each other)
            if domain.pressureRHS.data_ptr() == domain.velocityResult.data_ptr():
                if own_pressure_rhs.data_ptr() == domain.velocityResult.data_ptr():
                    own_pressure_rhs = torch.zeros_like(domain.velocityResult)
                domain.setPressureRHS(own_pressure_rhs)
                domain.UpdateDomainData()

    def _make_divergence_free(
        self, iterations: int | torch.Tensor, max_iter: int
    ) -> bool:
        """The projection of :meth:`make_divergence_free`, see there."""
        corrector_steps = 1
        # speeds up pressure_non_ortho_steps, no difference in result noticed.
        # On for differentiable runs too: guesses are detached, so this changes how
        # fast a solve converges, not what it converges to.
        pressure_reuse_result = True
        pressure_use_face_transform = False
        vcv = self._velocity_corrector_version

        if isinstance(iterations, torch.Tensor):
            iterations = iterations.numpy()[0]

        domain = self.domain
        non_ortho_flags = self.__non_ortho_flags
        # overwrite time step and A
        time_step = torch.tensor([1], device=cpu_device, dtype=domain.A.dtype)
        domain.setA(torch.ones_like(domain.A))

        for step in range(iterations):
            with SAMPLE("PISO step"):
                self._run_prep_fn(
                    Hook.PRE,
                    domain=domain,
                    local_step=step,
                    time_step=time_step,
                    total_step=self.total_step,
                    total_time=self.total_time,
                )

                domain.UpdateDomainData()
                self.__backend.CopyVelocityResultFromBlocks(domain)

                for cstep in range(corrector_steps):
                    with SAMPLE("corrector step"):
                        # update the existing velocity directly
                        domain.setPressureRHS(domain.velocityResult)
                        domain.UpdateDomainData()

                        self.__backend.SetupPressureMatrix(
                            domain,
                            time_step,
                            non_ortho_flags,
                            pressure_use_face_transform,
                        )

                        last_pressure_result = 0
                        for pstep in range(self.pressure_non_ortho_steps):
                            # build only div(rhs) + non-ortho from existing rhs vector field
                            self.__backend.SetupPressureRHSdiv(
                                domain,
                                time_step,
                                non_ortho_flags,
                                pressure_use_face_transform,
                                timeStepNorm=self.pressure_time_step_normalized,
                            )

                            with (
                                torch.no_grad()
                                if self.exclude_pressure_solve_gradients
                                else nullcontext()
                            ):
                                x = (
                                    None
                                    if (pstep == 0 or not pressure_reuse_result)
                                    else domain.pressureResult
                                )
                                pressureResult, solve_ok = self.linear_solve(
                                    domain.P,
                                    domain.pressureRHSdiv,
                                    x=x,  # matrix_rank_deficient=False, residual_reset_step=100,
                                    use_BiCG=self.pressure_use_BiCG,
                                    use_scipy=self.scipy_solve_pressure,
                                    tol=self.pressure_tol,
                                    return_best_result=self.pressure_return_best_result,
                                    max_iter=max_iter,
                                    adjoint_rank_deficient=self._pressure_rank_deficient(
                                        domain.P
                                    ),
                                    amg_hierarchy=self._get_pressure_amg_hierarchy(
                                        domain.P
                                    ),
                                    tag="pressure",
                                )
                                del x

                            if self.normalize_pressure_result:
                                pressureResult = subtract_env_mean(
                                    pressureResult, domain.getBatchSize()
                                )  # for numerical and backwards stability
                            domain.setPressureResult(pressureResult)
                            domain.UpdateDomainData()

                            if not solve_ok:
                                return solve_ok

                            if self._check_stop():
                                break

                        del last_pressure_result

                        self.__backend.CopyPressureResultToBlocks(domain)

                        self.__backend.CorrectVelocity(
                            domain,
                            time_step,
                            version=vcv,
                            timeStepNorm=self.pressure_time_step_normalized,
                            **self._correct_velocity_kwargs,
                        )

                self.__backend.CopyVelocityResultToBlocks(domain)

                if not solve_ok or self._check_stop():
                    return solve_ok

                self.end_step(time_step)

        return solve_ok

    def _PISO_split_step(
        self, iterations: int | torch.Tensor, time_step: torch.Tensor | None = None
    ) -> bool:
        """Advance the domain by a number of PISO substeps of fixed size.

        Runs the advection, pressure and correction stages of every substep,
        through the differentiable or the in-place backend as
        :attr:`differentiable` selects.

        Parameters
        ----------
        iterations : int or torch.Tensor
            Number of substeps to take. A tensor's first element is used.
        time_step : torch.Tensor or None, optional
            Substep size, a 1-element tensor. Defaults to :attr:`time_step`.

        Returns
        -------
        bool
            Whether every linear solve of every substep succeeded.
        """
        self._check_domain()
        solve_ok = True
        if time_step is None:
            time_step = self.__get_time_step_torch()
        # Per-channel lists or joint tensors, depending on the branch.
        x: Any
        scalarResult: Any
        advect_use_prev_result = True
        advect_non_ortho_reuse_result = True
        # speeds up pressure_non_ortho_steps, no difference in result noticed.
        # On for differentiable runs too: guesses are detached, so this changes how
        # fast a solve converges, not what it converges to.
        pressure_reuse_result = True
        pressure_use_face_transform = False
        # velocity corrector version. use different pressure gradients: 0 default (finite volume), 1 for finite differences, 4 for correcting fluxes (orthogonal), 5 for finite volume, 6 FVM with face transformations
        vcv = self._velocity_corrector_version
        pressure_dp = False

        if isinstance(iterations, torch.Tensor):
            iterations = iterations.numpy()[0]
        assert iterations > 0

        domain = self.domain
        non_ortho_flags = self.__non_ortho_flags

        for step in range(iterations):
            with SAMPLE("PISO step"):
                # _LOG.info("Substep %d", step)
                if self.convergence_tol is not None:
                    self.__backend.CopyVelocityResultFromBlocks(domain)
                    last_vel = domain.velocityResult.clone().detach()

                self._run_prep_fn(
                    Hook.PRE,
                    domain=domain,
                    local_step=step,
                    time_step=time_step,
                    total_step=self.total_step,
                    total_time=self.total_time,
                )
                domain.UpdateDomainData()

                advection_matrix_for_velocity = False
                if self.advect_passive_scalar and domain.hasPassiveScalar():
                    with SAMPLE("Advect scalar"):
                        split_scalar_channels = not (
                            domain.isPassiveScalarViscosityStatic()
                            and domain.isAllFixedBoundariesPassiveScalarTypeStatic()
                        )
                        scalar_channels = domain.getPassiveScalarChannels()

                        # if the scalar advection matrix can be reused for velocity advection
                        advection_matrix_for_velocity = not (
                            domain.hasPassiveScalarViscosity()
                            or domain.hasBlockViscosity()
                            or domain.hasPassiveScalarBlockViscosity()
                            or split_scalar_channels
                        )
                        # if self.density_viscosity is not None:
                        #    viscosity = domain.viscosity
                        #    domain.setViscosity(self.density_viscosity)
                        #    advection_matrix_for_velocity = False

                        # self.__backend.SetupAdvectionMatrix(domain, time_step, non_ortho_flags)
                        if split_scalar_channels:
                            matrices = []  # pre-compute all matrices to avoid re-computation during non-ortho steps
                            for channel in range(scalar_channels):
                                self.__backend.SetupAdvectionMatrix(
                                    domain,
                                    time_step,
                                    non_ortho_flags,
                                    forPassiveScalar=True,
                                    passiveScalarChannel=channel,
                                )
                                matrices.append(domain.C.clone())
                        else:
                            self.__backend.SetupAdvectionMatrix(
                                domain,
                                time_step,
                                non_ortho_flags,
                                forPassiveScalar=True,
                                passiveScalarChannel=0,
                            )

                        if (
                            not self.non_orthogonal
                        ):  # orthogonal version with gradient/backprop support
                            self.__backend.SetupAdvectionScalar(domain, time_step, 0)

                            self._run_prep_fn(
                                Hook.POST_SCALAR_SETUP,
                                domain=domain,
                                no_step=0,
                                local_step=step,
                                time_step=time_step,
                                total_step=self.total_step,
                                total_time=self.total_time,
                            )

                            with (
                                torch.no_grad()
                                if self.exclude_advection_solve_gradients
                                else nullcontext()
                            ):
                                if split_scalar_channels:
                                    RHS = _split_channels(
                                        domain.scalarRHS,
                                        domain.getBatchSize(),
                                        scalar_channels,
                                    )
                                    scalarResult = []
                                    for channel in range(scalar_channels):
                                        sR, solve_ok = self.linear_solve(
                                            matrices[channel],
                                            RHS[channel],
                                            x=None,
                                            use_BiCG=self.advection_use_BiCG,
                                            use_scipy=self.scipy_solve_advection,
                                            tol=self.advection_tol,
                                            tag="scalar",
                                        )
                                        scalarResult.append(sR)
                                        del sR
                                    del RHS
                                    scalarResult = _join_channels(
                                        scalarResult, domain.getBatchSize()
                                    )
                                else:
                                    scalarResult, solve_ok = self.linear_solve(
                                        domain.C,
                                        domain.scalarRHS,
                                        x=None,
                                        use_BiCG=self.advection_use_BiCG,
                                        use_scipy=self.scipy_solve_advection,
                                        tol=self.advection_tol,
                                        tag="scalar",
                                    )

                            domain.setScalarResult(scalarResult)
                            domain.UpdateDomainData()

                        else:
                            self.__backend.CopyScalarResultFromBlocks(
                                domain
                            )  # needed for non-ortho components on RHS

                            for no_step in range(self.advect_non_ortho_steps):
                                self.__backend.SetupAdvectionScalar(
                                    domain, time_step, non_ortho_flags
                                )

                                self._run_prep_fn(
                                    Hook.POST_SCALAR_SETUP,
                                    domain=domain,
                                    no_step=no_step,
                                    local_step=step,
                                    time_step=time_step,
                                    total_step=self.total_step,
                                    total_time=self.total_time,
                                )

                                with (
                                    torch.no_grad()
                                    if self.exclude_advection_solve_gradients
                                    else nullcontext()
                                ):
                                    if split_scalar_channels:
                                        if (
                                            no_step == 0
                                            or not advect_non_ortho_reuse_result
                                        ):
                                            x = [None] * scalar_channels
                                        else:
                                            x = _split_channels(
                                                domain.scalarResult,
                                                domain.getBatchSize(),
                                                scalar_channels,
                                            )
                                        RHS = _split_channels(
                                            domain.scalarRHS,
                                            domain.getBatchSize(),
                                            scalar_channels,
                                        )
                                        scalarResult = []
                                        for channel in range(scalar_channels):
                                            sR, solve_ok = self.linear_solve(
                                                matrices[channel],
                                                RHS[channel],
                                                x=x[channel],
                                                use_BiCG=self.advection_use_BiCG,
                                                use_scipy=self.scipy_solve_advection,
                                                tol=self.advection_tol,
                                                tag="scalar",
                                            )
                                            scalarResult.append(sR)
                                        del x
                                        del RHS
                                        scalarResult = _join_channels(
                                            scalarResult, domain.getBatchSize()
                                        )
                                    else:
                                        x = (
                                            None
                                            if (
                                                no_step == 0
                                                or not advect_non_ortho_reuse_result
                                            )
                                            else domain.scalarResult
                                        )
                                        scalarResult, solve_ok = self.linear_solve(
                                            domain.C,
                                            domain.scalarRHS,
                                            x=x,
                                            use_BiCG=self.advection_use_BiCG,
                                            use_scipy=self.scipy_solve_advection,
                                            tol=self.advection_tol,
                                            tag="scalar",
                                        )
                                    del x

                                domain.setScalarResult(scalarResult)
                                domain.UpdateDomainData()

                                if not solve_ok or self._check_stop():
                                    return solve_ok

                        if split_scalar_channels:
                            del matrices

                        self.__backend.CopyScalarResultToBlocks(domain)

                with SAMPLE("Advect velocity"):
                    # DON'T use pressure from previous corrector steps, otherwise it's applied twice
                    apply_pressure_gradient = False

                    self._run_prep_fn(
                        Hook.PRE_VELOCITY_SETUP,
                        domain=domain,
                        local_step=step,
                        time_step=time_step,
                        total_step=self.total_step,
                        total_time=self.total_time,
                    )

                    # if self.density_viscosity is not None:
                    #    domain.setViscosity(viscosity)

                    if not advection_matrix_for_velocity:
                        self.__backend.SetupAdvectionMatrix(
                            domain, time_step, non_ortho_flags
                        )

                    if (
                        not self.non_orthogonal
                    ):  # orthogonal version with gradient/backprop support
                        self.__backend.SetupAdvectionVelocity(
                            domain, time_step, 0, apply_pressure_gradient
                        )

                        self._run_prep_fn(
                            Hook.POST_VELOCITY_SETUP,
                            domain=domain,
                            no_step=0,
                            local_step=step,
                            time_step=time_step,
                            total_step=self.total_step,
                            total_time=self.total_time,
                        )

                        with (
                            torch.no_grad()
                            if self.exclude_advection_solve_gradients
                            else nullcontext()
                        ):
                            x = (
                                None
                                if (not advect_use_prev_result)
                                else domain.velocityResult
                            )
                            velocityResult, solve_ok = self.linear_solve(
                                domain.C,
                                domain.velocityRHS,
                                x=x,
                                use_BiCG=self.advection_use_BiCG,
                                use_scipy=self.scipy_solve_advection,
                                tol=self.advection_tol,
                                tag="velocity",
                            )
                            del x

                        domain.setVelocityResult(velocityResult)
                        domain.UpdateDomainData()

                    else:
                        self.__backend.CopyVelocityResultFromBlocks(
                            domain
                        )  # needed for non-ortho components on RHS

                        for no_step in range(self.advect_non_ortho_steps):
                            self.__backend.SetupAdvectionVelocity(
                                domain,
                                time_step,
                                non_ortho_flags,
                                apply_pressure_gradient,
                            )

                            self._run_prep_fn(
                                Hook.POST_VELOCITY_SETUP,
                                domain=domain,
                                no_step=no_step,
                                local_step=step,
                                time_step=time_step,
                                total_step=self.total_step,
                                total_time=self.total_time,
                            )

                            with (
                                torch.no_grad()
                                if self.exclude_advection_solve_gradients
                                else nullcontext()
                            ):
                                x = (
                                    None
                                    if (
                                        no_step == 0
                                        or not advect_non_ortho_reuse_result
                                    )
                                    else domain.velocityResult
                                )
                                velocityResult, solve_ok = self.linear_solve(
                                    domain.C,
                                    domain.velocityRHS,
                                    x=x,
                                    use_BiCG=self.advection_use_BiCG,
                                    use_scipy=self.scipy_solve_advection,
                                    tol=self.advection_tol,
                                    tag="velocity",
                                )
                                del x

                            domain.setVelocityResult(velocityResult)
                            domain.UpdateDomainData()

                            if not solve_ok or self._check_stop():
                                return solve_ok

                    # CopyVelocityResultToBlocks(domain) not yet, the original vel on blocks is still needed for pressure rhs

                    if not solve_ok:
                        return solve_ok

                self._run_prep_fn(
                    Hook.POST_PREDICTION,
                    domain=domain,
                    local_step=step,
                    time_step=time_step,
                    total_step=self.total_step,
                    total_time=self.total_time,
                )

                # can use stop handler in POST_PREDICTION to stop sim after advection.
                if self._check_stop():
                    break

                for cstep in range(self.corrector_steps):
                    with SAMPLE("corrector step"):
                        if (
                            not self.non_orthogonal
                        ):  # orthogonal version with gradient/backprop support
                            self.__backend.SetupPressureCorrection(
                                domain,
                                time_step,
                                0,
                                pressure_use_face_transform,
                                timeStepNorm=self.pressure_time_step_normalized,
                            )

                            self._run_prep_fn(
                                Hook.POST_PRESSURE_SETUP,
                                domain=domain,
                                local_step=step,
                                time_step=time_step,
                                total_step=self.total_step,
                                total_time=self.total_time,
                            )

                            with (
                                torch.no_grad()
                                if self.exclude_pressure_solve_gradients
                                else nullcontext()
                            ):
                                pressureResult, solve_ok = self.linear_solve(
                                    domain.P,
                                    domain.pressureRHSdiv,
                                    x=self._pressure_initial_guess(
                                        cstep, domain.pressureRHSdiv
                                    ),  # matrix_rank_deficient=False, residual_reset_step=100,
                                    use_BiCG=self.pressure_use_BiCG,
                                    use_scipy=self.scipy_solve_pressure,
                                    tol=self._pressure_tol_for(
                                        cstep, 0, self.corrector_steps
                                    ),
                                    return_best_result=self.pressure_return_best_result,
                                    adjoint_rank_deficient=self._pressure_rank_deficient(
                                        domain.P
                                    ),
                                    amg_hierarchy=self._get_pressure_amg_hierarchy(
                                        domain.P
                                    ),
                                    tag="pressure",
                                )

                            if not solve_ok:
                                return solve_ok

                            if self.normalize_pressure_result:
                                pressureResult = subtract_env_mean(
                                    pressureResult, domain.getBatchSize()
                                )  # for numerical and backwards stability
                            self._store_pressure_guess(cstep, pressureResult)
                            domain.setPressureResult(pressureResult)
                            domain.UpdateDomainData()

                            self._run_prep_fn(
                                Hook.POST_PRESSURE_RESULT,
                                domain=domain,
                                local_step=step,
                                time_step=time_step,
                                total_step=self.total_step,
                                total_time=self.total_time,
                            )

                        else:  # non-ortho version
                            self.__backend.SetupPressureMatrix(
                                domain,
                                time_step,
                                non_ortho_flags,
                                pressure_use_face_transform,
                            )

                            for pstep in range(self.pressure_non_ortho_steps):
                                # self.__backend.SetupPressureCorrection(domain, time_step, non_ortho_flags)
                                if pstep == 0:
                                    # build rhs (vector field) (and div(rhs) + non-ortho)
                                    self.__backend.SetupPressureRHS(
                                        domain,
                                        time_step,
                                        non_ortho_flags,
                                        pressure_use_face_transform,
                                        timeStepNorm=self.pressure_time_step_normalized,
                                    )
                                    # self._run_prep_fn("POST_PRESSURE_RHS", domain=domain, local_step=step, time_step=time_step, total_step=self.total_step, total_time=self.total_time)
                                else:
                                    # build only div(rhs) + non-ortho from existing rhs vector field
                                    self.__backend.SetupPressureRHSdiv(
                                        domain,
                                        time_step,
                                        non_ortho_flags,
                                        pressure_use_face_transform,
                                        timeStepNorm=self.pressure_time_step_normalized,
                                    )

                                self._run_prep_fn(
                                    Hook.POST_PRESSURE_SETUP,
                                    domain=domain,
                                    local_step=step,
                                    time_step=time_step,
                                    total_step=self.total_step,
                                    total_time=self.total_time,
                                )

                                with (
                                    torch.no_grad()
                                    if self.exclude_pressure_solve_gradients
                                    else nullcontext()
                                ):
                                    # _LOG.info("Start pressure solve #%d", cstep)
                                    # pstep > 0 reuses the previous non-ortho
                                    # iterate; pstep == 0 falls back to the
                                    # per-corrector cache from the last sub-step.
                                    x = (
                                        domain.pressureResult
                                        if (pstep > 0 and pressure_reuse_result)
                                        else self._pressure_initial_guess(
                                            cstep, domain.pressureRHSdiv
                                        )
                                    )
                                    p_tol = self._pressure_tol_for(
                                        cstep, pstep, self.corrector_steps
                                    )
                                    if pressure_dp:
                                        self.__LOG.debug(
                                            "Pressure solve is double precision."
                                        )
                                        P = domain.P.toType(torch.float64)
                                        pressureRHSdiv = domain.pressureRHSdiv.to(
                                            torch.float64
                                        )
                                        if x is not None:
                                            x = x.to(torch.float64)
                                        pressureResult, solve_ok = self.linear_solve(
                                            P,
                                            pressureRHSdiv,
                                            x=x,
                                            use_BiCG=self.pressure_use_BiCG,
                                            use_scipy=self.scipy_solve_pressure,
                                            tol=p_tol,
                                            return_best_result=self.pressure_return_best_result,
                                            tag="pressure",
                                        )  # , x=domain.pressureResult
                                        pressureResult = pressureResult.to(
                                            domain.pressureRHSdiv.dtype
                                        )
                                        del P
                                        del pressureRHSdiv
                                    else:
                                        pressureResult, solve_ok = self.linear_solve(
                                            domain.P,
                                            domain.pressureRHSdiv,
                                            x=x,
                                            matrix_rank_deficient=False,
                                            residual_reset_step=100,
                                            use_BiCG=self.pressure_use_BiCG,
                                            use_scipy=self.scipy_solve_pressure,
                                            tol=p_tol,
                                            return_best_result=self.pressure_return_best_result,
                                            adjoint_rank_deficient=self._pressure_rank_deficient(
                                                domain.P
                                            ),
                                            amg_hierarchy=self._get_pressure_amg_hierarchy(
                                                domain.P
                                            ),
                                            tag="pressure",
                                        )
                                    del x
                                # solve_ok = True #DEBUG

                                if self.normalize_pressure_result:
                                    pressureResult = subtract_env_mean(
                                        pressureResult, domain.getBatchSize()
                                    )  # for numerical and backwards stability
                                # Cache the last non-ortho iterate of this
                                # corrector as the guess for the same corrector at
                                # the next sub-step.
                                if pstep == self.pressure_non_ortho_steps - 1:
                                    self._store_pressure_guess(cstep, pressureResult)
                                domain.setPressureResult(pressureResult)
                                domain.UpdateDomainData()

                                self._run_prep_fn(
                                    Hook.POST_PRESSURE_RESULT,
                                    domain=domain,
                                    local_step=step,
                                    time_step=time_step,
                                    total_step=self.total_step,
                                    total_time=self.total_time,
                                )

                                if not solve_ok:
                                    return solve_ok

                                if self._check_stop():
                                    break

                        self._run_prep_fn(
                            Hook.POST_PRESSURE_NON_ORTHO,
                            domain=domain,
                            local_step=step,
                            time_step=time_step,
                            total_step=self.total_step,
                            total_time=self.total_time,
                        )

                        self.__backend.CopyPressureResultToBlocks(domain)

                        self.__backend.CorrectVelocity(
                            domain,
                            time_step,
                            version=vcv,
                            timeStepNorm=self.pressure_time_step_normalized,
                            **self._correct_velocity_kwargs,
                        )  # vcv

                        self._run_prep_fn(
                            Hook.POST_VELOCITY_CORRECTION,
                            domain=domain,
                            local_step=step,
                            time_step=time_step,
                            total_step=self.total_step,
                            total_time=self.total_time,
                        )

                        if self._check_stop():
                            break

                self.__backend.CopyVelocityResultToBlocks(domain)

                self._run_prep_fn(
                    Hook.POST,
                    domain=domain,
                    local_step=step,
                    time_step=time_step,
                    total_step=self.total_step,
                    total_time=self.total_time,
                )

                if self.convergence_tol is not None:
                    step_max_diff = (
                        torch.max(torch.abs(last_vel - domain.velocityResult))
                        .cpu()
                        .numpy()
                    )
                    if step_max_diff < self.convergence_tol:
                        self.__LOG.info(
                            "Simulation step max difference is under convergence tolerance."
                        )
                        solve_ok = False  # to stop sim

                if not solve_ok or self._check_stop():
                    break

                self.end_step(time_step)

        return solve_ok

    def _PISO_adaptive_step(
        self, CFL_cond: float | None = None, max_subsetps: int = 1000
    ) -> bool:
        """Advance the domain by :attr:`time_step`, splitting it to meet a CFL bound.

        The substep size is re-derived from the current maximum velocity before
        each substep, so a step that accelerates is split further as it goes.

        Parameters
        ----------
        CFL_cond : float or None, optional
            CFL number the substeps must respect. Defaults to
            :attr:`adaptive_CFL`.
        max_subsetps : int, optional
            Upper bound on the number of substeps, after which the step is
            abandoned. Default is 1000.

        Returns
        -------
        bool
            Whether the full time step was taken with every solve succeeding.
        """
        self._check_domain()
        time_step_target = self.time_step
        domain = self.domain
        CFL_cond = CFL_cond if CFL_cond is not None else self.adaptive_CFL
        if (
            domain.getBatchSize() > 1
            and self.adaptive_CFL_per_env
            and not self.differentiable
        ):
            return self._PISO_adaptive_step_per_env(CFL_cond, max_subsetps)
        substep = 0
        warned = False
        while time_step_target > 0 and not np.isclose(time_step_target, 0):
            with SAMPLE("adaptive step"):
                max_vel = domain.getMaxVelocity(True, True)
                max_vel_np = max_vel.detach().cpu().numpy()

                ts, substeps = _cfl_substep(time_step_target, max_vel_np, CFL_cond)

                time_step_target -= ts
                ts_torch = torch.tensor(
                    [ts], dtype=domain.getBlock(0).velocity.dtype, device=cpu_device
                )

                # _LOG.info("Adaptive step v2: maxVel %f, substep %d, timestep %f, remaining time %f", max_vel_np, substep, ts, time_step_target)

                if substeps > max_subsetps and not warned:
                    self.__LOG.warning(
                        "adaptive step (CFL=%.02f) results in more than %d substeps (%d).",
                        CFL_cond,
                        max_subsetps,
                        substeps,
                    )
                    warned = True
                elif substep == 0 and self.print_adaptive_step_info:  #
                    self.__LOG.debug(
                        "Adaptive step %d substeps: %d. From CFL = %.02f, max vel = %.03e, time step = %.03e.",
                        substep,
                        substeps,
                        CFL_cond,
                        max_vel_np,
                        time_step_target,
                    )

                sim_ok = self._PISO_split_step(iterations=1, time_step=ts_torch)
                substep += 1

                if not sim_ok or self._check_stop():
                    return False
        self.__LOG.debug(
            "Adaptive time step %.03e (CFL=%.02f) used %d substeps.",
            self.time_step,
            CFL_cond,
            substep,
        )
        return True

    def _PISO_adaptive_step_per_env(self, CFL_cond: float, max_subsetps: int) -> bool:
        """:meth:`_PISO_adaptive_step` with substeps of every batched environment.

        Every environment splits :attr:`time_step` by its own maximum velocity,
        exactly as it would on its own. The batch runs until the environment with
        the most substeps is done; an environment that is done already runs
        through the remaining substeps too (at its last substep size), and its
        state is restored afterwards (:class:`~phipict.batching.EnvStateSnapshot`).
        A substep costs about as much as with substeps shared by all environments.

        :attr:`total_time` is the time of the environment furthest ahead during
        the step and advances by :attr:`time_step` in total. Hooks receive the
        ``[B]`` substep sizes as ``time_step``.

        Parameters
        ----------
        CFL_cond : float
            CFL number the substeps must respect.
        max_subsetps : int
            Upper bound on the number of substeps, after which a warning is logged.

        Returns
        -------
        bool
            Whether the full time step was taken with every solve succeeding.
        """
        domain = self.domain
        B = domain.getBatchSize()
        dtype = domain.getBlock(0).velocity.dtype
        start_time = self.total_time
        remaining: list[Any] = [self.time_step] * B  # float or NumPy scalar array
        time_steps: list[Any] = [self.time_step] * B
        substeps_taken = [0] * B
        substep = 0
        warned = False
        while True:
            active = [r > 0 and not np.isclose(r, 0) for r in remaining]
            if not any(active):
                break
            with SAMPLE("adaptive step"):
                max_vel_np = (
                    domain.getMaxVelocityPerEnv(True, True).detach().cpu().numpy()
                )
                max_substeps = 0
                for b in range(B):
                    if not active[b]:
                        continue  # keeps its last substep size
                    time_steps[b], substeps = _cfl_substep(
                        remaining[b], max_vel_np[b], CFL_cond
                    )
                    remaining[b] -= time_steps[b]
                    substeps_taken[b] += 1
                    max_substeps = max(max_substeps, substeps)

                if max_substeps > max_subsetps and not warned:
                    self.__LOG.warning(
                        "adaptive step (CFL=%.02f) results in more than %d substeps (%d).",
                        CFL_cond,
                        max_subsetps,
                        max_substeps,
                    )
                    warned = True

                done = [b for b in range(B) if not active[b]]
                snapshot = (
                    EnvStateSnapshot(
                        domain, torch.tensor(done), extra=self._env_state_extras()
                    )
                    if done
                    else None
                )
                ts_torch = torch.tensor(
                    [float(ts) for ts in time_steps], dtype=dtype, device=cpu_device
                )
                sim_ok = self._PISO_split_step(iterations=1, time_step=ts_torch)
                if snapshot is not None:
                    snapshot.restore()
                substep += 1
                self.total_time = start_time + self.time_step - float(min(remaining))

                if not sim_ok or self._check_stop():
                    return False
        self.total_time = start_time + self.time_step
        self.__LOG.debug(
            "Adaptive time step %.03e (CFL=%.02f) used %d batched substeps, per environment %s.",
            self.time_step,
            CFL_cond,
            substep,
            substeps_taken,
        )
        return True

    def _env_state_extras(self) -> list[Callable[[], torch.Tensor | None]]:
        """Getters of per-environment state outside the domain.

        Restored with the domain state for batched environments that sit out a
        substep, see :meth:`_PISO_adaptive_step_per_env`.

        Returns
        -------
        list of Callable[[], torch.Tensor or None]
            Getters of flat tensors holding one slice per environment.
        """
        return []

    def _flux_balance_tol(self) -> float | torch.Tensor:
        """Absolute threshold for the boundary flux balance check in :meth:`run`.

        A relative ``pressure_tol`` has no right-hand side to refer to before the
        first solve, so its absolute floor is used, or the dtype default if it
        has none.

        Returns
        -------
        float or torch.Tensor
            :attr:`pressure_tol` itself if it is not relative, otherwise its
            absolute floor, or the dtype default if it has none.
        """
        tol = self.pressure_tol
        if not isinstance(tol, SolverTolerance):
            return tol
        if tol.atol is not None:
            return tol.atol
        default_tol = piso_diff._get_solver_tolerance(dtype=self.__get_dtype())
        assert default_tol is not None, f"No default tolerance for {self.__get_dtype()}"
        return default_tol

    def run(
        self, iterations: int, static: bool = False, log_domain: bool = True
    ) -> bool:
        """Run the simulation.

        Each iteration advances :attr:`substeps` PISO steps of
        :attr:`time_step`, or :attr:`time_step` in total with adaptive substeps.

        Parameters
        ----------
        iterations : int
            Number of iterations.
        static : bool, optional
            Whether to only advect the passive scalar. Default is False.
        log_domain : bool, optional
            Whether to log the domain description. Default is True.

        Returns
        -------
        bool
            Whether the simulation finished without failure.

        Raises
        ------
        ValueError
            If images should be written without a log directory, or
            :attr:`substeps` is invalid.
        """
        self._check_domain()

        domain = self.domain

        if self.log_interval > 0 and self.log_images and self.log_dir is None:
            raise ValueError("need to specify log/output directory")
        self.__LOG.info(
            "Starting sim with %d iterations, output in %s.",
            iterations,
            self.log_dir or "NONE",
        )
        if log_domain:
            self.__LOG.debug(str(domain))
            for blockIdx in range(domain.getNumBlocks()):
                self.__LOG.debug(str(domain.getBlock(blockIdx)))

        domain_orientation = domain.GetCoordinateOrientation()
        if domain_orientation == 0:
            self.__LOG.warning(
                "Domain coordinate systems have mixed orientation. This can lead to issues with the simulation."
            )
        domain_flux_balance = domain.GetBoundaryFluxBalance()
        if ntonp(torch.abs(domain_flux_balance)) > ntonp(self._flux_balance_tol()):
            self.__LOG.warning(
                "Domain boundary is not divergence free (flux balance: %.03e). This can prevent pressure solve convergence.",
                domain_flux_balance,
            )
            return False
        # self.__LOG.info("Domain handedness %d, boundary flux balance: %s", domain_orientation, domain_flux_balance.cpu().numpy())

        with SAMPLE("runSim"):
            sim_ok = True
            time_step_target = self.time_step
            substeps = self.substeps
            max_mag: Any  # NumPy array or float
            if self.norm_vel:  # or True:
                max_mag_temp = tensor_as_np(domain.getMaxVelocityMagnitude(False))
                max_mag = max_mag_temp * 1.05
            else:
                max_mag = 1
                max_mag_temp = max_mag
            CFL_cond = 0.8
            adaptive_step = False
            time_step: torch.Tensor | None = None

            if substeps > 0:
                pass  # just fixed substeps. 1 iteration with have physical time = time_step*substeps.
            elif substeps == -1:
                # compute max time step for each iteration/substep based on current velocity. 1 iteration with have physical time = time_step.
                adaptive_step = True
            elif substeps == -2:
                # compute max time step based on initial conditions, then keep it constant. 1 iteration with have physical time = time_step.
                time_step_value, substeps = get_max_time_step(
                    domain, time_step_target, CFL_cond, with_transformations=True
                )
                self.__LOG.debug(
                    "Setting time step to %.02e, substeps to %d based on initial conditions.",
                    time_step_value,
                    substeps,
                )
                time_step = torch.tensor(
                    [time_step_value],
                    dtype=domain.getBlock(0).velocity.dtype,
                    device=cpu_device,
                )
            else:
                raise ValueError("Invalid substeps")

            out_dir = self.log_dir

            vel_exr = False  # not static

            if self.log_images:
                if domain.getSpatialDims() < 3 and domain.hasVertexCoordinates():
                    plot_grids(
                        domain.getVertexCoordinates(),
                        path=out_dir,
                        linewidth=0.5,
                        type="pdf",
                    )
                self.save_domain_images(max_mag=max_mag, vel_exr=vel_exr)
            for it in range(1, iterations + 1):
                with SAMPLE("Iteration"):
                    # LOG.info("It: %d/%d", it, iterations)
                    log = self.log_interval > 0 and (it % self.log_interval) == 0

                    self.__LOG.debug(
                        "It: %d/%d, substeps:%s, timestep:%f",
                        it,
                        iterations,
                        "adaptive" if adaptive_step else substeps,
                        time_step_target,
                    )
                    with SAMPLE("simIt"):
                        try:
                            if static:
                                sim_ok = self.advect_static(
                                    iterations=substeps, time_step=time_step
                                )
                            elif adaptive_step:
                                sim_ok = self._PISO_adaptive_step()
                            else:
                                sim_ok = self._PISO_split_step(
                                    iterations=substeps, time_step=time_step
                                )
                            # if not sim_ok:
                            # break
                        except piso_diff.LinsolveError:
                            # self.__LOG.error("Simulation failed in major step %d (total step %d):\n%s", it, self.total_step, str(e))
                            self.__LOG.exception(
                                "Simulation failed in major step %d (total step %d):",
                                it,
                                self.total_step,
                            )
                            break

                    with SAMPLE("vel mag"):
                        max_mag_temp = tensor_as_np(
                            domain.getMaxVelocityMagnitude(False)
                        )
                        # _LOG.info("Max vel magnitude: %.03e ", max_mag_temp, max_mag_temp_old)
                        if log:
                            max_vel_temp = tensor_as_np(domain.getMaxVelocity(False))
                        # max_vel_transformed = domain.getMaxVelocity(False, True).cpu().numpy()
                        # _LOG.info("Max vel: %.03e, with bounds %.03e, transformed block 0 %.03e", max_vel_temp, domain.getMaxVelocity(True).cpu().numpy(), max_vel_transformed)
                        if np.isnan(max_mag_temp):
                            self.__LOG.warning("NaN encountered in velocity, stopping")
                            sim_ok = False
                            break
                        if self.norm_vel:  # or True:
                            max_mag = max_mag_temp * 1.05

                    if log:
                        with SAMPLE("vel div"):
                            vel_div = _C.ComputeVelocityDivergence(domain).detach()
                            vel_div_abs = torch.abs(vel_div)
                        with SAMPLE("p stats"):
                            p = domain.pressureResult.detach()
                            p_mean = torch.mean(p).cpu().numpy()
                            p_min = torch.min(p).cpu().numpy()
                            p_max = torch.max(p).cpu().numpy()
                            del p
                        with SAMPLE("output"):
                            self.__LOG.info(
                                "%d/%d Stats:\nVelocity: max=%.03e, max mag=%.03e\nPressure: mean=%.03e, min=%.03e, max=%.03e\nDivergence: mean=%.03e, min=%.03e, max=%.03e, total=%.03e",
                                it,
                                iterations,
                                max_vel_temp,
                                max_mag_temp,
                                p_mean,
                                p_min,
                                p_max,
                                torch.mean(vel_div_abs).cpu().numpy(),
                                torch.min(vel_div_abs).cpu().numpy(),
                                torch.max(vel_div_abs).cpu().numpy(),
                                torch.sum(vel_div_abs).cpu().numpy(),
                            )
                            if self.log_images:
                                self.save_domain_images(
                                    max_mag=max_mag, vel_exr=vel_exr
                                )
                    if self.log_fn is not None:
                        with SAMPLE("log fn"):
                            self.log_fn(
                                domain=domain,
                                out_dir=out_dir,
                                it=it,
                                out_it=self.img_out_idx,
                                total_step=self.total_step,
                            )

                    if self._check_stop() or (not sim_ok):
                        break

        if self.save_domain_name is not None:
            self.save_domain(self.save_domain_name)

        self.__LOG.info("sim finished after %d total steps", self.total_step)

        return sim_ok
