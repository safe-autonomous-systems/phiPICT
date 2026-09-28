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
# MHD electric potential functions, AMG-preconditioned and rank-deficient
# solves, relative/rescaled solver tolerances, solver telemetry.
# Moved into the phipict package, formatted, linted and typed.

"""Differentiable wrappers of the PISO solver kernels.

Each wrapper runs a CUDA kernel of :mod:`phipict._C` inside a
:class:`torch.autograd.Function`, so gradients flow through the multi-block
domain fields it reads and writes.
"""

import logging
import math
from collections.abc import Callable, Collection, Sequence
from typing import Any, overload

import torch

from phipict import _C
from phipict.io.output import StringWriter
from phipict.solvers import stats as solver_stats
from phipict.solvers.amg import AMGHierarchy, BatchedAMGHierarchy, amg_pcg_solve
from phipict.solvers.tolerance import (
    ADJOINT_RTOL_FALLBACK,
    PRECISION_FLOOR_FACTOR,
    get_adjoint_rtol,
)
from phipict.utils.profiling import SAMPLE

_LOG = logging.getLogger("phipict.Diff")
_LOG_DEBUG = False


_EXCLUDED_GRADIENTS: dict[str, Any] = {}


def exclude_gradient(
    function_name: str, grad_name: str, is_gradient_input: bool
) -> None:
    """Exclude a gradient of a differentiable function.

    Excluded gradients are replaced by zeros (inputs) or None (outputs).

    Parameters
    ----------
    function_name : str
        Name of the differentiable function, e.g. ``"SetupPressureRHS"``.
    grad_name : str
        Name of the gradient tensor, must end with ``_GRAD``.
    is_gradient_input : bool
        Whether the gradient is an incoming gradient of the backward pass
        (True) or a computed gradient returned by it (False).

    Raises
    ------
    ValueError
        If the function does not exist or ``grad_name`` is not a gradient.
    """
    global _EXCLUDED_GRADIENTS
    if function_name not in _EXCLUDED_GRADIENTS:
        raise ValueError("Function '%s' does not exist" % (function_name,))
    if not grad_name.endswith("_GRAD"):
        raise ValueError(
            "Only gradients can be excluded (name should end with '_GRAD')."
        )
    _EXCLUDED_GRADIENTS[function_name][0 if is_gradient_input else 1].add(grad_name)


def clear_excluded_gradients() -> None:
    """Remove all gradient exclusions set via :func:`exclude_gradient`."""
    global _EXCLUDED_GRADIENTS
    for function_name, (input_set, output_set) in _EXCLUDED_GRADIENTS.items():
        input_set.clear()
        output_set.clear()


def _set_gradient_input(
    var_grad: torch.Tensor,
    grad_setter: Callable[[torch.Tensor], Any],
    function_name: str,
    grad_name: str,
) -> None:
    """Hand a gradient to the domain, zeroing it if it has been excluded.

    Parameters
    ----------
    var_grad : torch.Tensor
        Gradient to pass on.
    grad_setter : Callable
        Domain setter the gradient is handed to.
    function_name : str
        Name the exclusions are registered under, see :func:`exclude_gradient`.
    grad_name : str
        Name of this gradient within ``function_name``.
    """
    if grad_name in _EXCLUDED_GRADIENTS[function_name][0]:
        grad_setter(torch.zeros_like(var_grad))
    else:
        grad_setter(var_grad)


def check_non_ortho_rhs(non_ortho_flags: int) -> bool:
    """Check whether non-orthogonal components are added to the right-hand side.

    Parameters
    ----------
    non_ortho_flags : int
        Bit flags as defined in ``PISO_multiblock_cuda.h``.

    Returns
    -------
    bool
        True if a ``*_RHS`` non-orthogonal flag is set.
    """
    # these must match the definition in 'PISO_multiblock_cuda.h'
    NON_ORTHO_DIRECT_MATRIX = 1
    NON_ORTHO_DIRECT_RHS = 2  # less stable than NON_ORTHO_DIRECT_MATRIX
    NON_ORTHO_DIAGONAL_MATRIX = 4  # not implemented
    NON_ORTHO_DIAGONAL_RHS = 8
    NON_ORTHO_CENTER_MATRIX = 16

    return (non_ortho_flags & NON_ORTHO_DIRECT_RHS) > 0 or (
        non_ortho_flags & NON_ORTHO_DIAGONAL_RHS
    ) > 0


def flatten_domain(
    domain: _C.Domain,
    tensor_filter: Collection[str] | None,
    clone: bool = False,
    empty: bool = False,
    exclusion_list: Collection[str] = [],
) -> tuple[dict[str, Any], list[torch.Tensor | None]]:
    """Collect the tensors of a domain into a flat list.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to read from.
    tensor_filter : Collection of str or None
        Names of the tensors to collect, e.g. ``"VELOCITY"`` or
        ``"BOUNDARY_VELOCITY_GRAD"``. None collects all tensors.
    clone : bool, optional
        Whether to clone the tensors. Default is False.
    empty : bool, optional
        Whether to collect None instead of the tensors. Default is False.
    exclusion_list : Collection of str, optional
        Names of tensors to replace by None. Default is empty.

    Returns
    -------
    domain_dict : dict of str to Any
        Nested description (domain, ``"BLOCKS"``, ``"BOUNDARIES"``) mapping
        tensor names to indices in ``tensor_list``.
    tensor_list : list of torch.Tensor or None
        The collected tensors; unset optional tensors are None.
    """
    tensor_list: list[torch.Tensor | None] = []
    tensor_idx = 0

    def add_tensor(
        dict: dict[str, Any],
        name: str,
        tensor: torch.Tensor | None,
        on_cpu: bool = False,
    ) -> None:
        if tensor_filter is None or name in tensor_filter:
            nonlocal tensor_idx
            dict[name] = tensor_idx
            if empty or name in exclusion_list:
                tensor = None
            elif (
                tensor is not None
            ):  # some fields are optional and return None if not set
                # if tensor is None: raise ValueError("Tensor '"+name+"' is None.")
                if clone:
                    tensor = tensor.clone()
                if on_cpu:
                    tensor = tensor.cpu()
            tensor_list.append(tensor)
            tensor_idx += 1

    domain_dict: dict[str, Any] = {}
    add_tensor(domain_dict, "C", domain.C.value)
    add_tensor(domain_dict, "C_GRAD", domain.CGrad.value)
    add_tensor(domain_dict, "A", domain.A)
    add_tensor(domain_dict, "A_GRAD", domain.AGrad)
    add_tensor(domain_dict, "P", domain.P.value)
    add_tensor(domain_dict, "P_GRAD", domain.PGrad.value)
    add_tensor(domain_dict, "VISCOSITY", domain.viscosity)
    add_tensor(domain_dict, "VISCOSITY_GRAD", domain.viscosityGrad, on_cpu=True)
    add_tensor(domain_dict, "PASSIVE_SCALAR_VISCOSITY", domain.passiveScalarViscosity)
    add_tensor(
        domain_dict, "PASSIVE_SCALAR_VISCOSITY_GRAD", domain.passiveScalarViscosityGrad
    )

    add_tensor(domain_dict, "PASSIVE_SCALAR_RESULT", domain.scalarResult)
    add_tensor(domain_dict, "PASSIVE_SCALAR_RESULT_GRAD", domain.scalarResultGrad)
    add_tensor(domain_dict, "VELOCITY_RESULT", domain.velocityResult)
    add_tensor(domain_dict, "VELOCITY_RESULT_GRAD", domain.velocityResultGrad)
    add_tensor(domain_dict, "PRESSURE_RESULT", domain.pressureResult)
    add_tensor(domain_dict, "PRESSURE_RESULT_GRAD", domain.pressureResultGrad)
    add_tensor(domain_dict, "EPOT_RESULT", domain.epotResult)
    add_tensor(domain_dict, "EPOT_RESULT_GRAD", domain.epotResultGrad)

    add_tensor(domain_dict, "PASSIVE_SCALAR_RHS", domain.scalarRHS)
    add_tensor(domain_dict, "PASSIVE_SCALAR_RHS_GRAD", domain.scalarRHSGrad)
    add_tensor(domain_dict, "VELOCITY_RHS", domain.velocityRHS)
    add_tensor(domain_dict, "VELOCITY_RHS_GRAD", domain.velocityRHSGrad)
    add_tensor(domain_dict, "PRESSURE_RHS", domain.pressureRHS)
    add_tensor(domain_dict, "PRESSURE_RHS_GRAD", domain.pressureRHSGrad)
    add_tensor(domain_dict, "PRESSURE_RHS_DIV", domain.pressureRHSdiv)
    add_tensor(domain_dict, "PRESSURE_RHS_DIV_GRAD", domain.pressureRHSdivGrad)

    domain_dict["BLOCKS"] = []
    for block in domain.getBlocks():
        block_dict: dict[str, Any] = {}

        add_tensor(block_dict, "VISCOSITY_BLOCK", block.viscosity)
        add_tensor(block_dict, "VISCOSITY_BLOCK_GRAD", block.viscosityGrad)
        add_tensor(block_dict, "VELOCITY", block.velocity)
        add_tensor(block_dict, "VELOCITY_GRAD", block.velocityGrad)
        add_tensor(block_dict, "VELOCITY_SOURCE", block.velocitySource)
        add_tensor(block_dict, "VELOCITY_SOURCE_GRAD", block.velocitySourceGrad)
        add_tensor(block_dict, "PASSIVE_SCALAR", block.passiveScalar)
        add_tensor(block_dict, "PASSIVE_SCALAR_GRAD", block.passiveScalarGrad)
        add_tensor(block_dict, "PRESSURE", block.pressure)
        add_tensor(block_dict, "PRESSURE_GRAD", block.pressureGrad)
        add_tensor(block_dict, "EPOT", block.epot)
        add_tensor(block_dict, "EPOT_GRAD", block.epotGrad)

        block_dict["BOUNDARIES"] = [None] * (domain.getSpatialDims() * 2)
        for boundary_idx in range(domain.getSpatialDims() * 2):
            bound = block.getBoundary(boundary_idx)
            if isinstance(bound, _C.FixedBoundary):  # bound.type==_C.FIXED:
                bound_dict: dict[str, Any] = {}

                add_tensor(bound_dict, "BOUNDARY_VELOCITY", bound.velocity)
                add_tensor(bound_dict, "BOUNDARY_VELOCITY_GRAD", bound.velocityGrad)
                add_tensor(bound_dict, "BOUNDARY_PASSIVE_SCALAR", bound.passiveScalar)
                add_tensor(
                    bound_dict, "BOUNDARY_PASSIVE_SCALAR_GRAD", bound.passiveScalarGrad
                )

                block_dict["BOUNDARIES"][boundary_idx] = bound_dict

        domain_dict["BLOCKS"].append(block_dict)

    return domain_dict, tensor_list


def set_domain_tensors_from_flat(
    domain: _C.Domain,
    domain_dict: dict[str, Any],
    tensor_list: Sequence[torch.Tensor | None],
    tensor_filter: list[str] | None = None,
    clone: bool = False,
) -> None:
    """Write tensors collected with :func:`flatten_domain` back to a domain.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to write to.
    domain_dict : dict of str to Any
        Description returned by :func:`flatten_domain`.
    tensor_list : Sequence of torch.Tensor or None
        Tensors indexed by ``domain_dict``. None entries clear optional
        tensors where supported and are skipped otherwise.
    tensor_filter : list of str or None, optional
        Names of the tensors to write. None writes all. Default is None.
    clone : bool, optional
        Whether to clone the tensors. Default is False.
    """

    def set_tensor(
        dict: dict[str, Any],
        name: str,
        setter: Callable[[torch.Tensor], Any],
        clear_fn: Callable[[], Any] | None = None,
    ) -> None:
        if (tensor_filter is None or name in tensor_filter) and (name in dict):
            # _LOG.debug("set '"+name+"' to domain")
            tensor = tensor_list[dict[name]]
            if tensor is not None:
                if clone:
                    tensor = tensor.clone()
                setter(tensor)
            elif clear_fn is not None:
                clear_fn()

    set_tensor(domain_dict, "C", domain.C.setValue)
    set_tensor(domain_dict, "C_GRAD", domain.CGrad.setValue)
    set_tensor(domain_dict, "A", domain.setA)
    set_tensor(domain_dict, "A_GRAD", domain.setAGrad)
    set_tensor(domain_dict, "P", domain.P.setValue)
    set_tensor(domain_dict, "P_GRAD", domain.PGrad.setValue)
    set_tensor(domain_dict, "VISCOSITY", domain.setViscosity)
    # set_tensor(domain_dict, "VISCOSITY_GRAD", domain.setViscosityGrad)
    set_tensor(
        domain_dict,
        "PASSIVE_SCALAR_VISCOSITY",
        domain.setScalarViscosity,
        domain.clearScalarViscosity,
    )

    set_tensor(domain_dict, "PASSIVE_SCALAR_RESULT", domain.setScalarResult)
    set_tensor(domain_dict, "PASSIVE_SCALAR_RESULT_GRAD", domain.setScalarResultGrad)
    set_tensor(domain_dict, "VELOCITY_RESULT", domain.setVelocityResult)
    set_tensor(domain_dict, "VELOCITY_RESULT_GRAD", domain.setVelocityResultGrad)
    set_tensor(domain_dict, "PRESSURE_RESULT", domain.setPressureResult)
    set_tensor(domain_dict, "PRESSURE_RESULT_GRAD", domain.setPressureResultGrad)
    set_tensor(domain_dict, "EPOT_RESULT", domain.setEpotResult)
    set_tensor(domain_dict, "EPOT_RESULT_GRAD", domain.setEpotResultGrad)

    set_tensor(domain_dict, "PASSIVE_SCALAR_RHS", domain.setScalarRHS)
    set_tensor(domain_dict, "PASSIVE_SCALAR_RHS_GRAD", domain.setScalarRHSGrad)
    set_tensor(domain_dict, "VELOCITY_RHS", domain.setVelocityRHS)
    set_tensor(domain_dict, "VELOCITY_RHS_GRAD", domain.setVelocityRHSGrad)
    set_tensor(domain_dict, "PRESSURE_RHS", domain.setPressureRHS)
    set_tensor(domain_dict, "PRESSURE_RHS_GRAD", domain.setPressureRHSGrad)
    set_tensor(domain_dict, "PRESSURE_RHS_DIV", domain.setPressureRHSdiv)
    set_tensor(domain_dict, "PRESSURE_RHS_DIV_GRAD", domain.setPressureRHSdivGrad)

    for block_idx, block in enumerate(domain.getBlocks()):
        block_dict = domain_dict["BLOCKS"][block_idx]

        set_tensor(
            block_dict, "VISCOSITY_BLOCK", block.setViscosity, block.clearViscosity
        )
        set_tensor(block_dict, "VISCOSITY_BLOCK_GRAD", block.setViscosityGrad)
        set_tensor(block_dict, "VELOCITY", block.setVelocity)
        set_tensor(block_dict, "VELOCITY_GRAD", block.setVelocityGrad)
        set_tensor(
            block_dict,
            "VELOCITY_SOURCE",
            block.setVelocitySource,
            block.clearVelocitySource,
        )
        set_tensor(block_dict, "VELOCITY_SOURCE_GRAD", block.setVelocitySourceGrad)
        set_tensor(block_dict, "PASSIVE_SCALAR", block.setPassiveScalar)
        set_tensor(block_dict, "PASSIVE_SCALAR_GRAD", block.setPassiveScalarGrad)
        set_tensor(block_dict, "PRESSURE", block.setPressure)
        set_tensor(block_dict, "PRESSURE_GRAD", block.setPressureGrad)
        set_tensor(block_dict, "EPOT", block.setEpot)
        set_tensor(block_dict, "EPOT_GRAD", block.setEpotGrad)

        for boundary_idx in range(domain.getSpatialDims() * 2):
            bound = block.getBoundary(boundary_idx)
            if isinstance(bound, _C.FixedBoundary):  # bound.type==_C.FIXED:
                bound_dict = block_dict["BOUNDARIES"][boundary_idx]

                set_tensor(bound_dict, "BOUNDARY_VELOCITY", bound.setVelocity)
                set_tensor(bound_dict, "BOUNDARY_VELOCITY_GRAD", bound.setVelocityGrad)
                set_tensor(
                    bound_dict, "BOUNDARY_PASSIVE_SCALAR", bound.setPassiveScalar
                )
                set_tensor(
                    bound_dict,
                    "BOUNDARY_PASSIVE_SCALAR_GRAD",
                    bound.setPassiveScalarGrad,
                )


@overload
def _get_solver_tolerance(tol: float, dtype: torch.dtype = ...) -> float: ...


@overload
def _get_solver_tolerance(
    tol: torch.Tensor, dtype: torch.dtype = ...
) -> torch.Tensor: ...


@overload
def _get_solver_tolerance(
    tol: None = None, dtype: torch.dtype = ...
) -> float | None: ...


def _get_solver_tolerance(
    tol: float | torch.Tensor | None = None, dtype: torch.dtype = torch.float32
) -> float | torch.Tensor | None:
    """Fill in the default solver tolerance for a dtype.

    Parameters
    ----------
    tol : float or torch.Tensor or None, optional
        Explicit tolerance, passed through unchanged. Default is None, which
        selects the dtype default.
    dtype : torch.dtype, optional
        Dtype the default is chosen for. Default is ``torch.float32``.

    Returns
    -------
    float or torch.Tensor or None
        ``tol`` if it was given, otherwise ``1e-5`` for ``float32`` and
        ``1e-8`` for ``float64``, or None for any other dtype.
    """
    if tol is None:
        if dtype == torch.float64:
            tol = 1e-8
        elif dtype == torch.float32:
            tol = 1e-5
    return tol


def _get_solver_tolerance_torch(
    tol: float | torch.Tensor | None = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Resolve a tolerance and wrap it in the 1-element tensor the kernels take.

    Parameters
    ----------
    tol : float or torch.Tensor or None, optional
        Explicit tolerance. Default is None, see :func:`_get_solver_tolerance`.
    dtype : torch.dtype, optional
        Dtype of the resulting tensor. Default is ``torch.float32``.

    Returns
    -------
    torch.Tensor
        1-element tensor holding the tolerance.
    """
    if isinstance(tol, torch.Tensor) and tol.numel() > 1:
        # one tolerance per right-hand side (batched environments)
        return tol.detach().to(device="cpu", dtype=dtype).reshape(-1).contiguous()
    tol = _get_solver_tolerance(tol, dtype)
    tol_torch = torch.tensor([tol], dtype=dtype)
    return tol_torch


def _tol_arg(tol: float | torch.Tensor) -> float | list[float]:
    """A tolerance for :func:`amg_pcg_solve`: a float, or one per right-hand side.

    Parameters
    ----------
    tol : float or torch.Tensor
        Tolerance as a number, a 1-element tensor or a tensor with one entry per
        right-hand side (batched environments).

    Returns
    -------
    float or list of float
        The tolerance(s).
    """
    if isinstance(tol, torch.Tensor) and tol.numel() > 1:
        return [float(t) for t in tol.detach().cpu().reshape(-1)]
    return _tol_value(tol)


def _tol_value(tol: float | torch.Tensor) -> float:
    """A tolerance as a Python float, whether it arrives as a tensor or a number.

    ``amg_pcg_solve`` takes a float where the CUDA kernels take a 1-element
    tensor; this is the same conversion ``Simulation.linear_solve_AMG`` does.

    Parameters
    ----------
    tol : float or torch.Tensor
        Tolerance, as a number or a tensor whose first element is used.

    Returns
    -------
    float
        The tolerance as a Python float.
    """
    if isinstance(tol, torch.Tensor):
        return float(tol.detach().cpu().reshape(-1)[0])
    return float(tol)


@overload
def _rescaled_bwd_tolerance(
    tol_torch: torch.Tensor,
    grad_x: torch.Tensor,
    n_rows: int | None = None,
    fwd_rtol: float | None = None,
    env_batch: int = 1,
) -> torch.Tensor: ...


@overload
def _rescaled_bwd_tolerance(
    tol_torch: float,
    grad_x: torch.Tensor,
    n_rows: int | None = None,
    fwd_rtol: float | None = None,
    env_batch: int = 1,
) -> float: ...


def _rescaled_bwd_tolerance(
    tol_torch: float | torch.Tensor,
    grad_x: torch.Tensor,
    n_rows: int | None = None,
    fwd_rtol: float | None = None,
    env_batch: int = 1,
) -> float | torch.Tensor:
    """The absolute tolerance holding the adjoint solve to a *relative* residual.

    The kernels take an absolute tolerance (ConvergenceCriterion.NORM2_NORMALIZED
    is ``||r|| / sqrt(n)``), and the number handed to a solve is resolved once,
    against the forward RHS. The adjoint system has the same operator but a
    completely different right-hand side, so reusing that number silently changes
    the requested tolerance.

    The returned tolerance is ``rtol * ||grad_x|| / sqrt(n)``. ``rtol`` is, in
    order: the override of :func:`~phipict.solvers.tolerance.set_adjoint_rtol`,
    the forward solve's own ``fwd_rtol``, or
    :data:`~phipict.solvers.tolerance.ADJOINT_RTOL_FALLBACK` for a forward solve
    with an absolute tolerance. It is strictly proportional to ``grad_x``, so
    scaling the loss scales every adjoint iterate and leaves the iteration count
    unchanged: the gradient is invariant to the loss scale.

    Parameters
    ----------
    tol_torch : float or torch.Tensor
        Tolerance the forward solve was run with; only its kind (float or tensor)
        is used.
    grad_x : torch.Tensor
        Right-hand side of the adjoint solve, i.e. the incoming gradient.
    n_rows : int or None, optional
        Number of matrix rows, used for the ``sqrt(n)`` of the convergence
        criterion. Defaults to ``grad_x.numel()``.
    fwd_rtol : float or None, optional
        The ``rtol`` of the forward solve's
        :class:`~phipict.solvers.tolerance.SolverTolerance`, or None if its
        tolerance is absolute. Default is None.
    env_batch : int, optional
        Number of batched environments whose adjoint right-hand sides
        ``grad_x`` holds. With more than one, every environment gets the
        tolerance of its own part of ``grad_x`` (a tensor with one tolerance per
        right-hand side). Default is 1.

    Returns
    -------
    float or torch.Tensor
        The rescaled tolerance, of the same kind as ``tol_torch``. ``tol_torch``
        is returned unchanged if the adjoint RHS norm is non-finite or zero.
    """
    if env_batch > 1:
        return _rescaled_bwd_tolerance_batched(
            tol_torch, grad_x, n_rows, fwd_rtol, env_batch
        )
    bwd = float(grad_x.detach().norm())
    if not math.isfinite(bwd) or bwd <= 0:
        return tol_torch

    # The criterion divides by sqrt(n)
    n = float(n_rows) if n_rows else float(grad_x.numel())
    rms = bwd / math.sqrt(max(n, 1.0))

    rtol = get_adjoint_rtol()
    if rtol is None:
        rtol = fwd_rtol if fwd_rtol is not None else ADJOINT_RTOL_FALLBACK

    # Strictly proportional to the adjoint RHS, and so to the loss: a Krylov solve
    # from the zero iterate is then homogeneous in its RHS, and the whole backward
    # invariant to the loss scale. Floored at the round-off level, below which the
    # solve can only stagnate
    rtol = max(rtol, PRECISION_FLOOR_FACTOR * torch.finfo(grad_x.dtype).eps)
    value = rtol * rms

    if isinstance(tol_torch, torch.Tensor):
        return torch.full_like(tol_torch, value)
    return value


def _rescaled_bwd_tolerance_batched(
    tol_torch: float | torch.Tensor,
    grad_x: torch.Tensor,
    n_rows: int | None,
    fwd_rtol: float | None,
    env_batch: int,
) -> torch.Tensor:
    """:func:`_rescaled_bwd_tolerance` for batched environments.

    Every environment is held to the relative residual of its own adjoint
    right-hand side, exactly as if it were solved alone; the tolerance of an
    environment is repeated for its right-hand sides.

    Returns
    -------
    torch.Tensor
        One tolerance per right-hand side (CPU, dtype of ``grad_x``).
    """
    n = int(n_rows) if n_rows else grad_x.numel() // env_batch
    per_env = grad_x.detach().reshape(env_batch, -1)
    rows_per_env = per_env.shape[1] // n
    rtol = get_adjoint_rtol()
    if rtol is None:
        rtol = fwd_rtol if fwd_rtol is not None else ADJOINT_RTOL_FALLBACK
    rtol = max(rtol, PRECISION_FLOOR_FACTOR * torch.finfo(grad_x.dtype).eps)
    norms = torch.linalg.vector_norm(per_env.to(torch.float64), dim=1).cpu()
    values = rtol * norms / math.sqrt(max(n, 1))
    # environments with a zero or non-finite adjoint RHS keep the forward tolerance
    fallback = (
        tol_torch.detach().cpu().to(torch.float64).reshape(-1)
        if isinstance(tol_torch, torch.Tensor)
        else torch.tensor([float(tol_torch)], dtype=torch.float64)
    )
    if fallback.numel() == env_batch * rows_per_env:
        fallback = fallback.reshape(env_batch, rows_per_env)[:, 0]
    else:
        fallback = fallback[:1].expand(env_batch)
    good = torch.isfinite(values) & (values > 0)
    values = torch.where(good, values, fallback)
    return values.repeat_interleave(rows_per_env).to(grad_x.dtype)


def _batched_matrix_grad(
    A: _C.CSRmatrix, grad_b: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """Gradient w.r.t. the values of batched matrices (one per environment).

    For ``A_m x_k = b_k`` with right-hand side k using matrix ``m = k // rpm``,
    ``dL/dA_m[i, j] = -sum_k grad_b_k[i] x_k[j]`` on the sparsity pattern:
    the per-environment counterpart of ``SparseOuterProduct``.

    Returns
    -------
    torch.Tensor
        Gradient w.r.t. ``A.value``, of shape ``[numMatrices * nnz]``.
    """
    n = A.getRows()
    nnz = A.getNnz()
    num_matrices = A.getBatchSize()
    row = A.row.to(torch.int64)
    rows = torch.repeat_interleave(
        torch.arange(n, device=row.device), row[1:] - row[:-1]
    )
    cols = A.index.to(torch.int64)
    gb = grad_b.reshape(num_matrices, -1, n)
    xx = x.reshape(num_matrices, -1, n)
    return -(gb[:, :, rows] * xx[:, :, cols]).sum(dim=1).reshape(num_matrices * nnz)


class LinsolveError(RuntimeError):
    """A linear solve did not converge or produced a non-finite residual."""


def _check_solver_return_infos(
    solver_infos: Sequence[Any],
    transpose: bool,
    use_BiCG: bool,
    tol: float,
    max_iter: int,
    return_best_result: bool,
    is_FWD: bool | None = None,
    debug_out: bool = False,
) -> None:
    """Check a solve's result infos and log or raise on failure.

    Parameters
    ----------
    solver_infos : Sequence of phipict._C.LinearSolverResultInfo
        One result info per right-hand side, as returned by ``_C.SolveLinear``.
    transpose : bool
        Whether the transposed system was solved. Used in the messages.
    use_BiCG : bool
        Whether BiCGStab was used instead of CG. Used in the messages.
    tol : float
        Tolerance the solve was run with. Used in the messages.
    max_iter : int
        Iteration limit the solve was run with. Used in the messages.
    return_best_result : bool
        Whether the solve returned its best iterate, in which case only a
        non-finite residual counts as a failure.
    is_FWD : bool or None, optional
        True for a forward solve, False for an adjoint solve, None if unknown.
        Used in the messages. Default is None.
    debug_out : bool, optional
        Whether to log the result infos of a successful solve. Default is False.

    Raises
    ------
    LinsolveError
        If a solve did not converge, or produced a non-finite residual.
    """
    # solver_infos: list(_C.LinearSolverResultInfo), as returned by _C.SolveLinear()

    fwd_str = "" if is_FWD is None else ("FWD, " if is_FWD else "BWD, ")

    if any(not solver_info.isFiniteResidual for solver_info in solver_infos):
        non_finite_batches = [
            i
            for i, solver_info in enumerate(solver_infos)
            if not solver_info.isFiniteResidual
        ]
        s = StringWriter()
        for i, solver_info in enumerate(solver_infos):
            s.write_line("\tRHS %d: %s", i, solver_info)
        raise LinsolveError(
            "Linear solve (%sT:%s, BiCG:%s, tol:%.03e) reported non-finite residual for RHS %s of %d. Solver infos:\n%s."
            % (
                fwd_str,
                transpose,
                use_BiCG,
                tol,
                non_finite_batches,
                len(solver_infos),
                s,
            )
        )
        s.reset()

    elif any(not solver_info.converged for solver_info in solver_infos):
        not_converged_batches = [
            i for i, solver_info in enumerate(solver_infos) if not solver_info.converged
        ]
        msg = (
            "Linear solve (%sT:%s, BiCG:%s, tol:%.03e) RHS %s of %d did not converge after %d iterations."
            % (
                fwd_str,
                transpose,
                use_BiCG,
                tol,
                not_converged_batches,
                len(solver_infos),
                max_iter,
            )
        )
        if return_best_result:
            _LOG.warning(msg)
            s = StringWriter()
            for i in not_converged_batches:
                s.write_line(
                    "\tRHS %d: best residual %.03e from iteration %d.",
                    i,
                    solver_infos[i].finalResidual,
                    solver_infos[i].usedIterations,
                )
            _LOG.warning("Using best results:\n%s", s)
            s.reset()
            if debug_out:
                converged_batches = [
                    i
                    for i, solver_info in enumerate(solver_infos)
                    if solver_info.converged
                ]
                for i in converged_batches:
                    s.write_line(
                        "\tRHS %d: residual %.03e from iteration %d.",
                        i,
                        solver_infos[i].finalResidual,
                        solver_infos[i].usedIterations,
                    )
                _LOG.debug("Converged results:\n%s", s)
                s.reset()
        else:
            s = StringWriter()
            for i, solver_info in enumerate(solver_infos):
                s.write_line("\tRHS %d: %s", i, solver_info)
            msg += " Solver infos:\n%s" % (s,)
            raise LinsolveError(msg)

    elif debug_out:
        s = StringWriter()
        for i, solver_info in enumerate(solver_infos):
            s.write_line(
                "\tRHS %d: residual %.03e from iteration %d.",
                i,
                solver_info.finalResidual,
                solver_info.usedIterations,
            )
        _LOG.debug(
            "Linear solve (%sT:%s, BiCG:%s, tol:%.03e) converged:\n%s",
            fwd_str,
            transpose,
            use_BiCG,
            tol,
            s,
        )
        s.reset()


def _linear_solve_wrapper(
    csrMat: _C.CSRmatrix,
    rhs: torch.Tensor,
    result: torch.Tensor,
    maxit_torch: torch.Tensor,
    tol_torch: torch.Tensor,
    convergence_criterion: _C.ConvergenceCriterion,
    use_BiCG: bool,
    matrix_rank_deficient: bool,
    residual_reset_step: int,
    transpose: bool,
    print_residual: bool,
    return_best_result: bool,
    is_FWD: bool | None = None,
    debug_out: bool = False,
    double_fallback: bool = False,
    BiCG_with_preconditioner: bool = True,
    BiCG_precondition_fallback: bool = False,
    tag: str = "unknown",
) -> None:
    """Run one linear solve through the CUDA kernels, writing into ``result``.

    A zero right-hand side is short-circuited: ``result`` is zeroed and no solve
    is run.

    Parameters
    ----------
    csrMat : phipict._C.CSRmatrix
        System matrix.
    rhs : torch.Tensor
        Right-hand side(s).
    result : torch.Tensor
        Initial iterate, overwritten with the solution.
    maxit_torch : torch.Tensor
        1-element tensor holding the iteration limit.
    tol_torch : torch.Tensor
        1-element tensor holding the tolerance.
    convergence_criterion : phipict._C.ConvergenceCriterion
        Norm the tolerance is applied to.
    use_BiCG : bool
        Use BiCGStab instead of CG.
    matrix_rank_deficient : bool
        Whether the operator is singular, e.g. a pure-Neumann Laplacian.
    residual_reset_step : int
        Iteration interval at which the residual is recomputed from scratch.
    transpose : bool
        Solve the transposed system.
    print_residual : bool
        Let the kernel print its residual history.
    return_best_result : bool
        Return the iterate with the smallest residual rather than the last one.
    is_FWD : bool or None, optional
        True for a forward solve, False for an adjoint solve, None if unknown.
        Default is None.
    debug_out : bool, optional
        Log the result infos of a successful solve. Default is False.
    double_fallback : bool, optional
        Retry a failed single-precision solve in double precision.
        Default is False.
    BiCG_with_preconditioner : bool, optional
        Use a preconditioner for BiCGStab. Default is True.
    BiCG_precondition_fallback : bool, optional
        Retry a failed BiCGStab solve with the preconditioner toggled.
        Default is False.
    tag : str, optional
        Name the solve is recorded under in :mod:`phipict.solvers.stats`.
        Default is ``"unknown"``.

    Raises
    ------
    LinsolveError
        If the solve fails and no fallback succeeds.
    """
    if not rhs.eq(0).all():
        # Opt-in telemetry (phipict.solvers.stats). Nothing is
        # measured, allocated or synchronized unless a recorder is active.
        probe = solver_stats.Probe(rhs, tag) if solver_stats.is_active() else None
        if probe is not None:
            probe.start()

        # with SAMPLE("_C.SolveLinear"):
        solver_infos = _C.SolveLinear(
            csrMat,
            rhs,
            result,
            maxit_torch,
            tol_torch,
            convergence_criterion,
            use_BiCG,
            matrix_rank_deficient,
            residual_reset_step,
            transpose,
            print_residual,
            return_best_result,
            BiCGwithPreconditioner=BiCG_with_preconditioner,
        )

        def not_solved(sinfos: Sequence[Any]) -> bool:
            """Whether a set of result infos counts as a failed solve.

            Parameters
            ----------
            sinfos : Sequence of phipict._C.LinearSolverResultInfo
                Result infos of one solve.

            Returns
            -------
            bool
                True if any solve did not converge, or, when the best iterate is
                returned, if any residual is non-finite.
            """
            return (
                not return_best_result and any(not sinfo.converged for sinfo in sinfos)
            ) or (
                return_best_result
                and any(not sinfo.isFiniteResidual for sinfo in sinfos)
            )

        if double_fallback and rhs.dtype == torch.float32 and not_solved(solver_infos):
            _LOG.warning("Single precision solve failed, trying double precision.")
            debug_out = True
            if debug_out:
                _LOG.debug(
                    "Single precision solver infos:\n%s", [str(_) for _ in solver_infos]
                )

            dp = torch.float64
            result_dp = torch.zeros_like(
                result, dtype=dp
            )  # do not start with a possibly corrupted result tensor
            solver_infos = _C.SolveLinear(
                csrMat.toType(dp),
                rhs.to(dp),
                result_dp,
                maxit_torch,
                tol_torch.to(dp),
                convergence_criterion,
                use_BiCG,
                matrix_rank_deficient,
                residual_reset_step,
                transpose,
                print_residual,
                return_best_result,
                BiCGwithPreconditioner=BiCG_with_preconditioner,
            )
            result = result_dp.to(result.dtype)
        # elif not double_fallback:
        #    raise RuntimeWarning("double_fallback is off!")

        if (
            BiCG_precondition_fallback
            and use_BiCG
            and not BiCG_with_preconditioner
            and not_solved(solver_infos)
        ):
            _LOG.warning("Not preconditioned BiCG solve failed, trying preconditioned.")
            debug_out = True
            if debug_out:
                _LOG.debug("Solver infos:\n%s", [str(_) for _ in solver_infos])

            result.zero_()  # may contain nan after failed solve

            solver_infos = _C.SolveLinear(
                csrMat,
                rhs,
                result,
                maxit_torch,
                tol_torch,
                convergence_criterion,
                use_BiCG,
                matrix_rank_deficient,
                residual_reset_step,
                transpose,
                print_residual,
                return_best_result,
                BiCGwithPreconditioner=True,
            )

        tol_value = tol_torch.detach().cpu().numpy()[0]
        maxit_value = maxit_torch.detach().cpu().numpy()[0]

        if probe is not None:
            probe.stop(
                solver_infos,
                tolerance=float(tol_value),
                max_iterations=int(maxit_value),
                use_BiCG=use_BiCG,
                transpose=transpose,
                is_fwd=is_FWD,
            )

        _check_solver_return_infos(
            solver_infos,
            transpose,
            use_BiCG,
            tol_value,
            maxit_value,
            return_best_result,
            is_FWD=is_FWD,
            debug_out=debug_out,
        )

    else:
        result.zero_()


def _project_out_constant(v: torch.Tensor, n_rows: int) -> torch.Tensor:
    """Remove the constant mode from each right-hand side in ``v``.

    For a singular operator whose nullspace is the constant vector (a pure
    Neumann Laplacian, i.e. every boundary carrying a flux condition and none
    prescribing a value) ``A y = g`` is solvable only for ``g`` orthogonal to
    ``null(A^T)``, and its solution is unique only up to ``null(A)``. Both spaces
    are ``span(1)`` for the symmetric case, so projecting the RHS makes the solve
    well posed and projecting the result selects the minimum-norm solution.

    Parameters
    ----------
    v : torch.Tensor
        Flat tensor holding one or more right-hand sides of ``n_rows`` each.
    n_rows : int
        Number of matrix rows, i.e. the length of one right-hand side.

    Returns
    -------
    torch.Tensor
        ``v`` with the per-RHS mean removed, or ``v`` unchanged if its size is
        not a positive multiple of ``n_rows``.
    """
    if n_rows <= 0 or v.numel() % n_rows != 0:
        return v
    shaped = v.reshape(-1, n_rows)
    return (shaped - shaped.mean(dim=1, keepdim=True)).reshape(v.shape)


def linear_solve_GPU(
    csrMat: _C.CSRmatrix,
    rhs: torch.Tensor,
    transpose: bool = False,
    use_BiCG: bool = False,
    tol: float | None = None,
    max_iter: int = 5000,
    return_best_result: bool = False,
    double_fallback: bool = False,
    BiCG_with_preconditioner: bool = True,
    BiCG_precondition_fallback: bool = False,
    adjoint_rank_deficient: bool = False,
    tag: str = "unknown",
    x0: torch.Tensor | None = None,
    amg_hierarchy: AMGHierarchy | BatchedAMGHierarchy | None = None,
    matrix_rank_deficient: bool = False,
    residual_reset_step: int = 0,
    fwd_rtol: float | None = None,
    env_batch: int = 1,
) -> torch.Tensor:
    """Solve ``A x = b`` differentiably w.r.t. the matrix values and ``b``.

    The backward pass solves the adjoint system with the transposed matrix.

    Parameters
    ----------
    csrMat : phipict._C.CSRmatrix
        System matrix.
    rhs : torch.Tensor
        Right-hand side(s), flat or batched along the first dimension.
    transpose : bool, optional
        Whether to solve with the transposed matrix. Default is False.
    use_BiCG : bool, optional
        Whether to use BiCGStab instead of CG. Default is False.
    tol : float or None, optional
        Absolute tolerance. None uses a dtype-dependent default. Default is None.
    max_iter : int, optional
        Maximum number of iterations. Default is 5000.
    return_best_result : bool, optional
        Whether to return the best iterate instead of raising if the solve does
        not converge. Default is False.
    double_fallback : bool, optional
        Whether to retry a failed single precision solve in double precision.
        Default is False.
    BiCG_with_preconditioner : bool, optional
        Whether BiCGStab uses a preconditioner. Default is True.
    BiCG_precondition_fallback : bool, optional
        Whether to retry a failed unpreconditioned BiCGStab solve with a
        preconditioner. Default is False.
    adjoint_rank_deficient : bool, optional
        Whether the matrix has the constant vector as nullspace; the adjoint
        solve is then projected onto its complement. Default is False.
    tag : str, optional
        Label for solver statistics. Default is ``"unknown"``.
    x0 : torch.Tensor or None, optional
        Initial guess of the forward solve. Default is None (zero).
    amg_hierarchy : AMGHierarchy or None, optional
        AMG preconditioner for a symmetric matrix; used instead of the CUDA
        solvers if given and ``use_BiCG`` is False. Default is None.
    matrix_rank_deficient : bool, optional
        Rank deficiency flag passed to the CUDA solver. Default is False.
    residual_reset_step : int, optional
        Interval of residual recomputation in the CUDA solver. Default is 0.
    fwd_rtol : float or None, optional
        The ``rtol`` ``tol`` was resolved from, which the adjoint solve is held to
        by default; None if ``tol`` is absolute. See
        :func:`~phipict.solvers.tolerance.set_adjoint_rtol`. Default is None.

    env_batch : int, optional
        Number of batched environments whose right-hand sides ``rhs`` holds;
        the adjoint solve resolves its relative tolerance per environment.
        Default is 1.

    Returns
    -------
    torch.Tensor
        Solution with the same shape as ``rhs``.

    Raises
    ------
    LinsolveError
        If a solve fails and ``return_best_result`` does not apply.
    """
    # The matrix is *not* cloned into this closure: a deep copy (value + index
    # + row) per solve is retained for the whole graph and, living in a Python
    # closure, is invisible to torch.autograd.graph hooks. Only the integer
    # sparsity pattern is captured -- it is never differentiated and is the same
    # object on every step, so it costs one pattern rather than one per solve.
    # The values travel through save_for_backward instead, which keeps a single
    # copy.
    A_index = csrMat.index
    A_row = csrMat.row

    def _make_matrix(values: torch.Tensor) -> _C.CSRmatrix:
        """Rebuild the operator from a value vector and the shared pattern."""
        return _C.CSRmatrix(values.detach(), A_index, A_row)

    convergence_criterion = _C.ConvergenceCriterion.NORM2_NORMALIZED
    dtype = rhs.dtype

    # Captured at function scope and detached, so it can never become an autograd
    # input. Exact: the map being differentiated is `b -> A^-1 b`, whose Jacobian
    # does not depend on the initial iterate.
    x0_const = None if x0 is None else x0.detach().reshape(rhs.shape).contiguous()

    # Safe here because forward and backward both run with grad disabled, so no
    # V-cycle is traced; `not use_BiCG` asserts the operator is symmetric, which is
    # what makes the same hierarchy valid for the adjoint solve.
    use_amg = amg_hierarchy is not None and not use_BiCG

    # `adjoint_rank_deficient` says the operator has the constant nullspace. It
    # deliberately does *not* touch the forward solve -- neither the kernel flag
    # above nor the RHS -- so forward trajectories stay bit-identical and this
    # change is confined to the gradient. The forward is well posed as it stands
    # because its RHS is compatible by construction (for the pressure Poisson,
    # `balance_boundary_fluxes` enforces zero net boundary flux). The adjoint RHS
    # is an incoming gradient with no such guarantee, which is what makes the
    # backward solve ill-posed on a duct where every boundary prescribes velocity.
    A_rows = csrMat.getRows()

    tol_torch = _get_solver_tolerance_torch(
        tol, dtype
    )  # torch.tensor([tol], dtype=dtype)
    maxit_torch = torch.IntTensor([max_iter])

    class LinearSolveFunction(torch.autograd.Function):
        """Autograd node of :func:`linear_solve_GPU`.

        Solves ``A x = b``; the backward pass solves the transposed system to
        obtain the gradients w.r.t. the matrix values and the right-hand side.
        """

        @staticmethod
        def forward(ctx: Any, A_val: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """Solve ``A x = b`` for ``x``.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            A_val : torch.Tensor
                Matrix values of the CSR system matrix.
            b : torch.Tensor
                Right-hand side(s).

            Returns
            -------
            torch.Tensor
                The solution, with the same shape as ``b``.
            """
            with SAMPLE("%sCG-FWD" % ("Bi" if use_BiCG else "",)):
                if _LOG_DEBUG:
                    _LOG.debug("linsolve %sCG forward", "Bi" if use_BiCG else "")

                # Cloned because the solvers write the iterate into `x` in place.
                if x0_const is None:
                    x = torch.zeros_like(b)
                else:
                    x = x0_const.to(dtype=b.dtype, device=b.device).clone()

                if use_amg:
                    # Probed like every other solve, or an AMG solve would be
                    # invisible to solver_stats and its cost unattributable.
                    probe = (
                        solver_stats.Probe(b, tag) if solver_stats.is_active() else None
                    )
                    if probe is not None:
                        probe.start()
                    assert amg_hierarchy is not None
                    with torch.no_grad():
                        solver_infos = amg_pcg_solve(
                            b,
                            x,
                            amg_hierarchy,
                            tol=_tol_arg(tol_torch),
                            max_iter=max_iter,
                            return_best_result=return_best_result,
                        )
                    if probe is not None:
                        probe.stop(
                            solver_infos,
                            tolerance=_tol_value(tol_torch),
                            max_iterations=max_iter,
                            use_BiCG=use_BiCG,
                            transpose=transpose,
                            is_fwd=True,
                        )
                    _check_solver_return_infos(
                        solver_infos,
                        transpose,
                        use_BiCG,
                        _tol_value(tol_torch),
                        max_iter,
                        return_best_result,
                        is_FWD=True,
                        debug_out=False,
                    )
                else:
                    # Local, so it is released when forward returns; only A_val is
                    # retained, via save_for_backward below.
                    A_fwd = _make_matrix(A_val)

                    _linear_solve_wrapper(
                        A_fwd,
                        b,
                        x,
                        maxit_torch,
                        tol_torch,
                        convergence_criterion,
                        use_BiCG,
                        matrix_rank_deficient,
                        residual_reset_step,
                        transpose,
                        False,
                        return_best_result,
                        is_FWD=True,
                        debug_out=False,
                        double_fallback=double_fallback,
                        BiCG_with_preconditioner=BiCG_with_preconditioner,
                        BiCG_precondition_fallback=BiCG_precondition_fallback,
                        tag=tag,
                    )

                    del A_fwd

                # ctx.save_for_backward(A_val, b, x)
                ctx.save_for_backward(A_val, x)

            return x  # , (0<=it and it<maxit)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(
            ctx: Any, grad_x: torch.Tensor
        ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
            """Solve the adjoint system and scatter its result onto ``A`` and ``b``.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            grad_x : torch.Tensor
                Gradient w.r.t. the solution ``x``.

            Returns
            -------
            tuple of (torch.Tensor or None)
                Gradients w.r.t. ``A_val`` and ``b``, each None if that input
                needs no gradient.
            """
            with SAMPLE("%sCG-BWD" % ("Bi" if use_BiCG else "",)):
                if _LOG_DEBUG:
                    _LOG.debug("linsolve %sCG backward", "Bi" if use_BiCG else "")

                grad_b = None
                grad_A_val = None
                if ctx.needs_input_grad[1] or ctx.needs_input_grad[0]:
                    with SAMPLE("RHSgrad"):
                        # A_val, b, x = ctx.saved_tensors
                        A_val, x = ctx.saved_tensors
                        A = _make_matrix(A_val)
                        grad_b = torch.zeros_like(grad_x)

                        # Make the adjoint solve well posed on a singular
                        # operator. Without this the incoming gradient carries a
                        # component along null(A^T) that no iterate can reduce,
                        # so CG cannot converge and returns a vector whose scale
                        # is set by the residual floor rather than by the
                        # gradient -- the mechanism behind the momentum adjoint
                        # gaining ~1e10 per step on the MHD duct.
                        if adjoint_rank_deficient:
                            grad_x = _project_out_constant(grad_x, A_rows)

                        # NORM2_NORMALIZED is an *absolute* RMS, and tol_torch
                        # was resolved against the forward RHS, so on its own it
                        # asks the adjoint for something unrelated to the adjoint:
                        # unreachable when ||grad_x|| >> ||b||, and already met at
                        # the zero iterate when ||grad_x|| << ||b||, which returns
                        # a zero adjoint and severs the chain. Rescaling holds it
                        # to a relative residual, the forward's rtol by default.
                        #
                        # After the projection above, so the norm is the one the
                        # solve actually starts from
                        bwd_tol = _rescaled_bwd_tolerance(
                            tol_torch,
                            grad_x,
                            n_rows=A_rows,
                            fwd_rtol=fwd_rtol,
                            env_batch=env_batch,
                        )

                        # if not grad_x.eq(0).all(): # will not converge if all 0, but grad should be 0 anyways in that case
                        # solver_info = _C.SolveLinear(A, grad_x, grad_b, maxit_torch, tol_torch, conv, use_BiCG, False, 0, not transpose, False, return_best_result)

                        # _check_solver_return_infos(solver_info, not transpose, use_BiCG, tol, max_iter, return_best_result, is_FWD=False, debug_out=False)
                        if use_amg:
                            # A^T == A, so the same hierarchy preconditions the
                            # adjoint solve; grad_b is already a zero iterate.
                            probe = (
                                solver_stats.Probe(grad_x, tag)
                                if solver_stats.is_active()
                                else None
                            )
                            if probe is not None:
                                probe.start()
                            assert amg_hierarchy is not None
                            with torch.no_grad():
                                solver_infos = amg_pcg_solve(
                                    grad_x,
                                    grad_b,
                                    amg_hierarchy,
                                    tol=_tol_arg(bwd_tol),
                                    max_iter=max_iter,
                                    return_best_result=return_best_result,
                                )
                            if probe is not None:
                                probe.stop(
                                    solver_infos,
                                    tolerance=_tol_value(bwd_tol),
                                    max_iterations=max_iter,
                                    use_BiCG=use_BiCG,
                                    transpose=not transpose,
                                    is_fwd=False,
                                )
                            _check_solver_return_infos(
                                solver_infos,
                                not transpose,
                                use_BiCG,
                                _tol_value(bwd_tol),
                                max_iter,
                                return_best_result,
                                is_FWD=False,
                                debug_out=False,
                            )
                        else:
                            _linear_solve_wrapper(
                                A,
                                grad_x,
                                grad_b,
                                maxit_torch,
                                bwd_tol,
                                convergence_criterion,
                                use_BiCG,
                                matrix_rank_deficient,
                                0,
                                not transpose,
                                False,
                                return_best_result,
                                is_FWD=False,
                                debug_out=False,
                                double_fallback=double_fallback,
                                BiCG_with_preconditioner=BiCG_with_preconditioner,
                                BiCG_precondition_fallback=BiCG_precondition_fallback,
                                tag=tag,
                            )

                    # Pick the minimum-norm adjoint: with a constant nullspace
                    # grad_b is determined only up to a constant, and letting
                    # that component ride along would feed an arbitrary offset
                    # into SparseOuterProduct below and into the next step.
                    if adjoint_rank_deficient:
                        grad_b = _project_out_constant(grad_b, A_rows)

                    # print("grad_b", grad_b)
                    if ctx.needs_input_grad[0]:  # gradient w.r.t. matrix
                        with SAMPLE("MatrixGrad"):
                            if A.getBatchSize() > 1:
                                # one matrix per environment
                                return _batched_matrix_grad(A, grad_b, x), grad_b
                            grad_A = (
                                A.WithZeroValue()
                            )  # creates new tensor for a.value initialized to 0
                            grad_A_val = grad_A.value
                            # if _LOG_DEBUG: _LOG.debug("A %s, db %s, x %s", grad_A, grad_b.size(), x.size())
                            if grad_A.getRows() == grad_b.size(0):  # no batched RHS
                                _C.SparseOuterProduct(grad_b, x, grad_A)
                            else:
                                grad_b_splits = torch.split(grad_b, grad_A.getRows())
                                x_splits = torch.split(x, grad_A.getRows())
                                for grad_b_split, x_split in zip(
                                    grad_b_splits, x_splits, strict=False
                                ):
                                    grad_A_split = grad_A.WithZeroValue()
                                    _C.SparseOuterProduct(
                                        grad_b_split, x_split, grad_A_split
                                    )
                                    # A_val
                                    grad_A_val += grad_A_split.value
                            grad_A_val *= -1

            return grad_A_val, grad_b

    return LinearSolveFunction.apply(csrMat.value, rhs)


_EXCLUDED_GRADIENTS["SetupAdvectionMatrix"] = (set(), set())


def SetupAdvectionMatrix(
    domain: _C.Domain,
    time_step: torch.Tensor,
    non_ortho_flags: int,
    forPassiveScalar: bool = False,
    passiveScalarChannel: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Set up the advection-diffusion matrix, differentiably.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to operate on.
    time_step : torch.Tensor
        Time step, a 1-element tensor.
    non_ortho_flags : int
        Non-orthogonal handling flags, see :func:`check_non_ortho_rhs`.
    forPassiveScalar : bool, optional
        Whether to use the passive scalar viscosity. Default is False.
    passiveScalarChannel : int, optional
        Passive scalar channel whose viscosity is used. Default is 0.

    Returns
    -------
    A : torch.Tensor
        Matrix diagonal, ``domain.A``.
    C_value : torch.Tensor
        Matrix values, ``domain.C.value``.

    Notes
    -----
    Tracked inputs: block.velocity, boundary.velocity, domain.viscosity and
    block.viscosity (or the passive scalar viscosity).
    """
    # input tensors that need to be tracked by pytorch
    tracked_tensor_filter = ["VELOCITY", "BOUNDARY_VELOCITY", "VISCOSITY"]
    if forPassiveScalar:
        tracked_tensor_filter.append("PASSIVE_SCALAR_VISCOSITY")
    else:
        tracked_tensor_filter.append("VISCOSITY_BLOCK")

    # the filter to get the gradient tensors corresponding to the tracked input tensors
    grad_tensor_filter = [s + "_GRAD" for s in tracked_tensor_filter]

    domain_dict, tracked_tensors = flatten_domain(domain, tracked_tensor_filter)

    # filter for any forward tensors that need to be saved for backwards
    backwards_tensor_filter: list[str] = []

    class SetupAdvectionMatrixFunction(torch.autograd.Function):
        """Autograd node of :func:`SetupAdvectionMatrix`.

        Writes ``domain.A`` and ``domain.C``.
        """

        @staticmethod
        def forward(
            ctx: Any, *tracked_tensors: torch.Tensor | None
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Run the SetupAdvectionMatrix kernel on the domain.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            *tracked_tensors : torch.Tensor or None
                Domain tensors autograd tracks for this op. The kernels read them
                from the domain itself; they are passed through so autograd records
                the dependency and routes the gradients back to them.

            Returns
            -------
            A : torch.Tensor
                Matrix diagonal, ``domain.A``.
            C_value : torch.Tensor
                Matrix values, ``domain.C.value``.
            """
            with SAMPLE("SetupAdvectionMatrix-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("SetupAdvectionMatrix forward")

                domain.CreateA()
                domain.C.CreateValue()
                domain.UpdateDomainData()
                _C.SetupAdvectionMatrix(
                    domain,
                    time_step,
                    non_ortho_flags,
                    forPassiveScalar=forPassiveScalar,
                    passiveScalarChannel=passiveScalarChannel,
                )

                saved_tensors = []
                saved_tensors.append(time_step)
                ctx.save_for_backward(*saved_tensors)

            return domain.A, domain.C.value  # .clone()

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(
            ctx: Any, A_grad: torch.Tensor, C_val_grad: torch.Tensor
        ) -> tuple[torch.Tensor | None, ...]:
            # if not non_ortho_flags==0: raise NotImplementedError("Only Orthogonal gradients are supported.")
            """Propagate the gradients of SetupAdvectionMatrix back to the domain tensors.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            A_grad : torch.Tensor
                Gradient w.r.t. the matrix diagonal ``domain.A``.
            C_val_grad : torch.Tensor
                Gradient w.r.t. the matrix values ``domain.C.value``.

            Returns
            -------
            tuple of (torch.Tensor or None)
                Gradients w.r.t. ``tracked_tensors``, in the same order. An entry
                is None where the corresponding input needs no gradient.
            """
            if any(ctx.needs_input_grad):
                with SAMPLE("SetupAdvectionMatrix-BWD"):
                    if _LOG_DEBUG:
                        _LOG.debug("SetupAdvectionMatrix backward")
                    time_step = ctx.saved_tensors[-1]
                    # domain.setAGrad(A_grad)
                    _set_gradient_input(
                        A_grad, domain.setAGrad, "SetupAdvectionMatrix", "A_GRAD"
                    )
                    # domain.CGrad.setValue(C_val_grad)
                    _set_gradient_input(
                        C_val_grad,
                        domain.CGrad.setValue,
                        "SetupAdvectionMatrix",
                        "C_GRAD",
                    )
                    domain.CreateViscosityGrad()  # also creates passive scalar and per-cell viscosity gradient tensors if neccessary
                    domain.CreateVelocityGradOnBlocks()
                    domain.CreateVelocityGradOnBoundaries()
                    domain.UpdateDomainData()
                    _C.SetupAdvectionMatrixGrad(
                        domain,
                        time_step,
                        non_ortho_flags,
                        forPassiveScalar=forPassiveScalar,
                        passiveScalarChannel=passiveScalarChannel,
                    )
                    # block_velocity_grad = [block.velocityGrad for block in domain.getBlocks()]
                    _, grad_tensors = flatten_domain(
                        domain,
                        grad_tensor_filter,
                        exclusion_list=_EXCLUDED_GRADIENTS["SetupAdvectionMatrix"][1],
                    )
            else:
                if _LOG_DEBUG:
                    _LOG.debug("SetupAdvectionMatrix backward empty")
                # block_velocity_grad = [None] * domain.getNumBlocks()
                _, grad_tensors = flatten_domain(domain, grad_tensor_filter, empty=True)

            # if _LOG_DEBUG: _LOG.debug("->velocity grad: %s", block_velocity_grad)

            return (*grad_tensors,)

    return SetupAdvectionMatrixFunction.apply(*tracked_tensors)


_EXCLUDED_GRADIENTS["SetupAdvectionScalar"] = (set(), set())


def SetupAdvectionScalar(
    domain: _C.Domain, time_step: torch.Tensor, non_ortho_flags: int
) -> torch.Tensor:
    """Set up the right-hand side of the passive scalar advection, differentiably.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to operate on.
    time_step : torch.Tensor
        Time step, a 1-element tensor.
    non_ortho_flags : int
        Non-orthogonal handling flags, see :func:`check_non_ortho_rhs`.

    Returns
    -------
    torch.Tensor
        The right-hand side, ``domain.scalarRHS``.

    Notes
    -----
    Inputs:

    - block.passiveScalar (not needed for backwards)
    - domain.scalarResult (if non-orthogonal)
    - block.transform (must be static, not differentiable)
    - domain.viscosity
    - boundary.passiveScalar
    - boundary.velocity
    - boundary.transform (must be static, not differentiable)
    """
    is_non_ortho = check_non_ortho_rhs(non_ortho_flags)
    # _LOG.debug("SetupAdvectionScalar is_non_ortho: %s",is_non_ortho)
    # additionally uses domain.scalarResult for non-orthogonal handling on the RHS

    tracked_tensor_filter = [
        "PASSIVE_SCALAR",
        "BOUNDARY_PASSIVE_SCALAR",
        "BOUNDARY_VELOCITY",
        "VISCOSITY",
        "PASSIVE_SCALAR_VISCOSITY",
    ]
    if is_non_ortho:
        tracked_tensor_filter.append("PASSIVE_SCALAR_RESULT")

    grad_tensor_filter = [s + "_GRAD" for s in tracked_tensor_filter]

    domain_dict, tracked_tensors = flatten_domain(domain, tracked_tensor_filter)

    backwards_tensor_filter = [
        "BOUNDARY_PASSIVE_SCALAR",
        "BOUNDARY_VELOCITY",
        "VISCOSITY",
        "PASSIVE_SCALAR_VISCOSITY",
    ]
    if is_non_ortho:
        backwards_tensor_filter.append("PASSIVE_SCALAR_RESULT")

    class SetupAdvectionScalarFunction(torch.autograd.Function):
        """Autograd node of :func:`SetupAdvectionScalar`.

        Writes ``domain.scalarRHS``.
        """

        @staticmethod
        def forward(ctx: Any, *tracked_tensors: torch.Tensor | None) -> torch.Tensor:
            """Run the SetupAdvectionScalar kernel on the domain.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            *tracked_tensors : torch.Tensor or None
                Domain tensors autograd tracks for this op. The kernels read them
                from the domain itself; they are passed through so autograd records
                the dependency and routes the gradients back to them.

            Returns
            -------
            torch.Tensor
                The right-hand side, ``domain.scalarRHS``.
            """
            with SAMPLE("SetupAdvectionScalar-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("scalarRHS forward")
                with torch.no_grad():
                    domain.CreateScalarRHS()
                    domain.UpdateDomainData()
                    _C.SetupAdvectionScalar(domain, time_step, non_ortho_flags)

                if backwards_tensor_filter:
                    domain_dict, saved_tensors = flatten_domain(
                        domain, backwards_tensor_filter
                    )
                    ctx.saved_tensors_domain_dict = domain_dict
                else:
                    ctx.saved_tensors_domain_dict = None
                    saved_tensors = []

                saved_tensors.append(time_step)
                ctx.save_for_backward(*saved_tensors)

            return domain.scalarRHS  # .clone()

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(
            ctx: Any, grad_rhs: torch.Tensor
        ) -> tuple[torch.Tensor | None, ...]:
            # if not non_ortho_flags==0: raise NotImplementedError("Only Orthogonal gradients are supported.")
            """Propagate the gradients of SetupAdvectionScalar back to the domain tensors.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            grad_rhs : torch.Tensor
                Gradient w.r.t. ``domain.scalarRHS``.

            Returns
            -------
            tuple of (torch.Tensor or None)
                Gradients w.r.t. ``tracked_tensors``, in the same order. An entry
                is None where the corresponding input needs no gradient.
            """
            with torch.no_grad():
                if any(ctx.needs_input_grad):
                    with SAMPLE("SetupAdvectionScalar-BWD"):
                        if _LOG_DEBUG:
                            _LOG.debug("scalarRHS backward")
                        time_step = ctx.saved_tensors[-1]
                        if ctx.saved_tensors_domain_dict is not None:
                            set_domain_tensors_from_flat(
                                domain, ctx.saved_tensors_domain_dict, ctx.saved_tensors
                            )

                        # domain.setScalarRHSGrad(grad_rhs)
                        _set_gradient_input(
                            grad_rhs,
                            domain.setScalarRHSGrad,
                            "SetupAdvectionScalar",
                            "PASSIVE_SCALAR_RHS_GRAD",
                        )
                        domain.CreateViscosityGrad()  # also creates passive scalar and per-cell viscosity gradient tensors if neccessary
                        # domain.CreatePassiveScalarViscosityGrad() # will clear the gradient field if dedicated passive scalar viscosity is not used
                        domain.CreatePassiveScalarGradOnBlocks()
                        domain.CreatePassiveScalarGradOnBoundaries()
                        domain.CreateVelocityGradOnBoundaries()
                        if is_non_ortho:
                            domain.CreateScalarResultGrad()
                        domain.UpdateDomainData()
                        _C.SetupAdvectionScalarGrad(domain, time_step, non_ortho_flags)
                        _, grad_tensors = flatten_domain(
                            domain,
                            grad_tensor_filter,
                            exclusion_list=_EXCLUDED_GRADIENTS["SetupAdvectionScalar"][
                                1
                            ],
                        )
                else:
                    if _LOG_DEBUG:
                        _LOG.debug("scalarRHS backward empty")
                    _, grad_tensors = flatten_domain(
                        domain, grad_tensor_filter, empty=True
                    )

            return (*grad_tensors,)

    rhs = SetupAdvectionScalarFunction.apply(*tracked_tensors)
    return rhs


_EXCLUDED_GRADIENTS["SetupAdvectionVelocity"] = (set(), set())


def SetupAdvectionVelocity(
    domain: _C.Domain,
    time_step: torch.Tensor,
    non_ortho_flags: int,
    apply_pressure_gradient: bool = False,
) -> torch.Tensor:
    """Set up the right-hand side of the velocity advection, differentiably.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to operate on.
    time_step : torch.Tensor
        Time step, a 1-element tensor.
    non_ortho_flags : int
        Non-orthogonal handling flags, see :func:`check_non_ortho_rhs`.
    apply_pressure_gradient : bool, optional
        Whether to include the pressure gradient. Default is False.

    Returns
    -------
    torch.Tensor
        The right-hand side, ``domain.velocityRHS``.

    Notes
    -----
    Inputs:

    - block.velocity (not needed for backwards)
    - block.velocitySource (not needed for backwards)
    - domain.velocityResult (if non-orthogonal)
    - block.transform (must be static, not differentiable)
    - domain.viscosity
    - block.viscosity
    - boundary.velocity
    - boundary.transform (must be static, not differentiable)
    """
    is_non_ortho = check_non_ortho_rhs(non_ortho_flags)

    tracked_tensor_filter = [
        "VELOCITY",
        "VELOCITY_SOURCE",
        "BOUNDARY_VELOCITY",
        "VISCOSITY",
        "VISCOSITY_BLOCK",
    ]
    if is_non_ortho:
        tracked_tensor_filter.append("VELOCITY_RESULT")

    grad_tensor_filter = [s + "_GRAD" for s in tracked_tensor_filter]

    domain_dict, tracked_tensors = flatten_domain(domain, tracked_tensor_filter)

    backwards_tensor_filter = ["BOUNDARY_VELOCITY", "VISCOSITY", "VISCOSITY_BLOCK"]
    backwards_tensor_filter.append(
        "VELOCITY_SOURCE"
    )  # DEBUG: value not needed but needs to be set to receive gradients
    if is_non_ortho:
        backwards_tensor_filter.append("VELOCITY_RESULT")

    class SetupAdvectionVelocityFunction(torch.autograd.Function):
        """Autograd node of :func:`SetupAdvectionVelocity`.

        Writes ``domain.velocityRHS``.
        """

        @staticmethod
        def forward(ctx: Any, *tracked_tensors: torch.Tensor | None) -> torch.Tensor:
            """Run the SetupAdvectionVelocity kernel on the domain.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            *tracked_tensors : torch.Tensor or None
                Domain tensors autograd tracks for this op. The kernels read them
                from the domain itself; they are passed through so autograd records
                the dependency and routes the gradients back to them.

            Returns
            -------
            torch.Tensor
                The right-hand side, ``domain.velocityRHS``.
            """
            with SAMPLE("SetupAdvectionVelocity-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("velocityRHS forward")

                domain.CreateVelocityRHS()
                domain.UpdateDomainData()
                _C.SetupAdvectionVelocity(
                    domain, time_step, non_ortho_flags, apply_pressure_gradient
                )

                if backwards_tensor_filter:
                    domain_dict, saved_tensors = flatten_domain(
                        domain, backwards_tensor_filter
                    )
                    ctx.saved_tensors_domain_dict = domain_dict
                else:
                    ctx.saved_tensors_domain_dict = None
                    saved_tensors = []

                saved_tensors.append(time_step)
                ctx.save_for_backward(*saved_tensors)

            return domain.velocityRHS  # .clone()

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(
            ctx: Any, grad_rhs: torch.Tensor
        ) -> tuple[torch.Tensor | None, ...]:
            # if not non_ortho_flags==0: raise NotImplementedError("Only Orthogonal gradients are supported.")
            """Propagate the gradients of SetupAdvectionVelocity back to the domain tensors.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            grad_rhs : torch.Tensor
                Gradient w.r.t. ``domain.velocityRHS``.

            Returns
            -------
            tuple of (torch.Tensor or None)
                Gradients w.r.t. ``tracked_tensors``, in the same order. An entry
                is None where the corresponding input needs no gradient.
            """
            if any(ctx.needs_input_grad):
                with SAMPLE("SetupAdvectionVelocity-BWD"):
                    if _LOG_DEBUG:
                        _LOG.debug("velocityRHS backward")
                    time_step = ctx.saved_tensors[-1]
                    if ctx.saved_tensors_domain_dict is not None:
                        set_domain_tensors_from_flat(
                            domain, ctx.saved_tensors_domain_dict, ctx.saved_tensors
                        )

                    # domain.setVelocityRHSGrad(grad_rhs)
                    _set_gradient_input(
                        grad_rhs,
                        domain.setVelocityRHSGrad,
                        "SetupAdvectionVelocity",
                        "VELOCITY_RHS_GRAD",
                    )
                    domain.CreateViscosityGrad()
                    domain.CreateVelocityGradOnBlocks()
                    domain.CreateVelocityGradOnBoundaries()
                    domain.CreateVelocitySourceGradOnBlocks()  # only create grad if velocity source exists, clears it otherwise
                    if is_non_ortho:
                        domain.CreateVelocityResultGrad()
                    domain.UpdateDomainData()

                    _C.SetupAdvectionVelocityGrad(
                        domain, time_step, non_ortho_flags, apply_pressure_gradient
                    )
                    # _LOG.debug("boundary[0].velocityGrad:\n%s", domain.getBlock(0).getBoundary(0).velocityGrad)
                    _, grad_tensors = flatten_domain(
                        domain,
                        grad_tensor_filter,
                        exclusion_list=_EXCLUDED_GRADIENTS["SetupAdvectionVelocity"][1],
                    )
            else:
                if _LOG_DEBUG:
                    _LOG.debug("velocityRHS backward empty")
                _, grad_tensors = flatten_domain(domain, grad_tensor_filter, empty=True)

            return (*grad_tensors,)

    return SetupAdvectionVelocityFunction.apply(*tracked_tensors)


def CopyScalarResultToBlocks(domain: _C.Domain) -> None:
    """Copy ``domain.scalarResult`` to ``block.passiveScalar``, differentiably.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to operate on.
    """

    class CopyScalarResultToBlocksFunction(torch.autograd.Function):
        """Autograd node of :func:`CopyScalarResultToBlocks`.

        Scatters ``domain.scalarResult`` onto ``block.passiveScalar``.
        """

        @staticmethod
        def forward(
            ctx: Any, scalar_result: torch.Tensor
        ) -> tuple[torch.Tensor | None, ...]:
            """Copy the scalar result onto the blocks.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            scalar_result : torch.Tensor
                Flat scalar result, ``domain.scalarResult``.

            Returns
            -------
            tuple of (torch.Tensor or None)
                The per-block passive scalar fields after the copy.
            """
            with SAMPLE("CopyScalarResultToBlocks-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("copy scalar forward")

                domain.CreatePassiveScalarOnBlocks()
                domain.UpdateDomainData()
                _C.CopyScalarResultToBlocks(domain)

                return (
                    *[block.passiveScalar for block in domain.getBlocks()],
                )  # .clone()

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(
            ctx: Any, *blocks_passive_scalar_grad: torch.Tensor
        ) -> torch.Tensor | None:
            """Gather the per-block gradients back into a flat gradient.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            *blocks_passive_scalar_grad : torch.Tensor
                Gradient w.r.t. each block's passive scalar field.

            Returns
            -------
            torch.Tensor or None
                Gradient w.r.t. ``scalar_result``, or None if it needs no gradient.
            """
            if ctx.needs_input_grad[0]:
                with SAMPLE("CopyScalarResultToBlocks-BWD"):
                    if _LOG_DEBUG:
                        _LOG.debug("copy scalar backward")
                    for block, s_grad in zip(
                        domain.getBlocks(), blocks_passive_scalar_grad, strict=False
                    ):
                        block.setPassiveScalarGrad(s_grad)
                    domain.CreateScalarResultGrad()
                    domain.UpdateDomainData()
                    _C.CopyScalarResultGradFromBlocks(domain)
                    scalarResultGrad = domain.scalarResultGrad
            else:
                if _LOG_DEBUG:
                    _LOG.debug("copy scalar backward empty")
                scalarResultGrad = None

            return scalarResultGrad

    CopyScalarResultToBlocksFunction.apply(domain.scalarResult)


def CopyScalarResultFromBlocks(domain: _C.Domain) -> None:
    """Copy ``block.passiveScalar`` to ``domain.scalarResult``.

    The backward pass is not implemented.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to operate on.
    """

    class CopyScalarResultFromBlocksFunction(torch.autograd.Function):
        """Autograd node of :func:`CopyScalarResultFromBlocks`.

        Writes ``domain.scalarResult``.
        """

        @staticmethod
        def forward(ctx: Any, *tracked_tensors: torch.Tensor | None) -> torch.Tensor:
            """Run the CopyScalarResultFromBlocks kernel on the domain.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            *tracked_tensors : torch.Tensor or None
                Domain tensors autograd tracks for this op. The kernels read them
                from the domain itself; they are passed through so autograd records
                the dependency and routes the gradients back to them.

            Returns
            -------
            torch.Tensor
                The gathered scalar result, ``domain.scalarResult``.
            """
            with SAMPLE("CopyScalarResultFromBlocks-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("copy scalar forward")

                domain.CreateScalarResult()
                domain.UpdateDomainData()
                _C.CopyScalarResultFromBlocks(domain)

                return domain.scalarResult

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(
            ctx: Any, scalar_result_grad: torch.Tensor
        ) -> tuple[torch.Tensor | None, ...]:
            """Propagate the gradients of CopyScalarResultFromBlocks back to the domain tensors.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            scalar_result_grad : torch.Tensor
                Gradient w.r.t. ``domain.scalarResult``.

            Returns
            -------
            tuple of (torch.Tensor or None)
                Gradients w.r.t. ``tracked_tensors``, in the same order. An entry
                is None where the corresponding input needs no gradient.
            """
            raise NotImplementedError("TODO: requires _C.CopyScalarResultGradToBlocks")

            if any(ctx.needs_input_grad):
                with SAMPLE("CopyScalarResultFromBlocks-BWD"):
                    if _LOG_DEBUG:
                        _LOG.debug("copy scalar backward")
                    domain.setScalarResultGrad(scalar_result_grad)
                    domain.CreatePassiveScalarGradOnBlocks()
                    domain.UpdateDomainData()
                    _C.CopyScalarResultGradToBlocks(domain)
                    block_scalar_grad: list[torch.Tensor | None] = [
                        block.passiveScalarGrad for block in domain.getBlocks()
                    ]
            else:
                if _LOG_DEBUG:
                    _LOG.debug("copy scalar backward empty")
                block_scalar_grad = [None] * domain.getNumBlocks()

            return (*block_scalar_grad,)

    block_scalar_data = [block.passiveScalar for block in domain.getBlocks()]
    CopyScalarResultFromBlocksFunction.apply(*block_scalar_data)


def CopyVelocityResultToBlocks(domain: _C.Domain) -> None:
    """Copy ``domain.velocityResult`` to ``block.velocity``, differentiably.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to operate on.
    """

    class CopyVelocityResultToBlocksFunction(torch.autograd.Function):
        """Autograd node of :func:`CopyVelocityResultToBlocks`.

        Scatters ``domain.velocityResult`` onto ``block.velocity``.
        """

        @staticmethod
        def forward(
            ctx: Any, velocity_result: torch.Tensor
        ) -> tuple[torch.Tensor, ...]:
            """Copy the velocity result onto the blocks.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            velocity_result : torch.Tensor
                Flat velocity result, ``domain.velocityResult``.

            Returns
            -------
            tuple of torch.Tensor
                The per-block velocity fields after the copy.
            """
            with SAMPLE("CopyVelocityResultToBlocks-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("copy velocity to forward")

                domain.CreateVelocityOnBlocks()
                domain.UpdateDomainData()
                _C.CopyVelocityResultToBlocks(domain)

                return (*[block.velocity for block in domain.getBlocks()],)  # .clone()

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(
            ctx: Any, *blocks_velocity_grad: torch.Tensor
        ) -> torch.Tensor | None:
            """Gather the per-block gradients back into a flat gradient.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            *blocks_velocity_grad : torch.Tensor
                Gradient w.r.t. each block's velocity field.

            Returns
            -------
            torch.Tensor or None
                Gradient w.r.t. ``velocity_result``, or None if it needs no gradient.
            """
            if ctx.needs_input_grad[0]:
                with SAMPLE("CopyVelocityResultToBlocks-BWD"):
                    if _LOG_DEBUG:
                        _LOG.debug("copy velocity to backward")
                    for block, v_grad in zip(
                        domain.getBlocks(), blocks_velocity_grad, strict=False
                    ):
                        v_grad = v_grad.contiguous()
                        block.setVelocityGrad(v_grad)
                    domain.CreateVelocityResultGrad()
                    domain.UpdateDomainData()
                    _C.CopyVelocityResultGradFromBlocks(domain)
                    velocityResultGrad = domain.velocityResultGrad
            else:
                if _LOG_DEBUG:
                    _LOG.debug("copy velocity to backward empty")
                velocityResultGrad = None

            return velocityResultGrad

    CopyVelocityResultToBlocksFunction.apply(domain.velocityResult)


def CopyVelocityResultFromBlocks(domain: _C.Domain) -> None:
    """Copy ``block.velocity`` to ``domain.velocityResult``, differentiably.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to operate on.
    """

    class CopyVelocityResultFromBlocksFunction(torch.autograd.Function):
        """Autograd node of :func:`CopyVelocityResultFromBlocks`.

        Writes ``domain.velocityResult``.
        """

        @staticmethod
        def forward(ctx: Any, *tracked_tensors: torch.Tensor | None) -> torch.Tensor:
            """Run the CopyVelocityResultFromBlocks kernel on the domain.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            *tracked_tensors : torch.Tensor or None
                Domain tensors autograd tracks for this op. The kernels read them
                from the domain itself; they are passed through so autograd records
                the dependency and routes the gradients back to them.

            Returns
            -------
            torch.Tensor
                The gathered velocity result, ``domain.velocityResult``.
            """
            with SAMPLE("CopyVelocityResultFromBlocks-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("copy velocity from forward")

                domain.CreateVelocityResult()
                domain.UpdateDomainData()
                _C.CopyVelocityResultFromBlocks(domain)

                return domain.velocityResult

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(
            ctx: Any, velocity_result_grad: torch.Tensor
        ) -> tuple[torch.Tensor | None, ...]:
            """Propagate the gradients of CopyVelocityResultFromBlocks back to the domain tensors.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            velocity_result_grad : torch.Tensor
                Gradient w.r.t. ``domain.velocityResult``.

            Returns
            -------
            tuple of (torch.Tensor or None)
                Gradients w.r.t. ``tracked_tensors``, in the same order. An entry
                is None where the corresponding input needs no gradient.
            """
            if any(ctx.needs_input_grad):
                with SAMPLE("CopyVelocityResultFromBlocks-BWD"):
                    if _LOG_DEBUG:
                        _LOG.debug("copy velocity from backward")
                    domain.setVelocityResultGrad(velocity_result_grad)
                    domain.CreateVelocityGradOnBlocks()
                    domain.UpdateDomainData()
                    _C.CopyVelocityResultGradToBlocks(domain)
                    block_velocity_grad: list[torch.Tensor | None] = [
                        block.velocityGrad for block in domain.getBlocks()
                    ]
            else:
                if _LOG_DEBUG:
                    _LOG.debug("copy velocity from backward empty")
                block_velocity_grad = [None] * domain.getNumBlocks()

            return (*block_velocity_grad,)

    block_velocity_data = [block.velocity for block in domain.getBlocks()]
    CopyVelocityResultFromBlocksFunction.apply(*block_velocity_data)


def CopyPressureResultToBlocks(domain: _C.Domain) -> None:
    """Copy ``domain.pressureResult`` to ``block.pressure``, differentiably.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to operate on.
    """

    class CopyPressureResultToBlocksFunction(torch.autograd.Function):
        """Autograd node of :func:`CopyPressureResultToBlocks`.

        Scatters ``domain.pressureResult`` onto ``block.pressure``.
        """

        @staticmethod
        def forward(
            ctx: Any, pressure_result: torch.Tensor
        ) -> tuple[torch.Tensor, ...]:
            """Copy the pressure result onto the blocks.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            pressure_result : torch.Tensor
                Flat pressure result, ``domain.pressureResult``.

            Returns
            -------
            tuple of torch.Tensor
                The per-block pressure fields after the copy.
            """
            with SAMPLE("CopyPressureResultToBlocks-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("copy pressure to forward")

                domain.CreatePressureOnBlocks()
                domain.UpdateDomainData()
                _C.CopyPressureResultToBlocks(domain)

                return (*[block.pressure for block in domain.getBlocks()],)  # .clone()

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(
            ctx: Any, *blocks_pressure_grad: torch.Tensor
        ) -> torch.Tensor | None:
            """Gather the per-block gradients back into a flat gradient.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            *blocks_pressure_grad : torch.Tensor
                Gradient w.r.t. each block's pressure field.

            Returns
            -------
            torch.Tensor or None
                Gradient w.r.t. ``pressure_result``, or None if it needs no gradient.
            """
            if ctx.needs_input_grad[0]:
                with SAMPLE("CopyPressureResultToBlocks-BWD"):
                    if _LOG_DEBUG:
                        _LOG.debug("copy pressure to backward")
                    for block, p_grad in zip(
                        domain.getBlocks(), blocks_pressure_grad, strict=False
                    ):
                        block.setPressureGrad(p_grad)
                    domain.CreatePressureResultGrad()
                    domain.UpdateDomainData()
                    _C.CopyPressureResultGradFromBlocks(domain)
                    pressureResultGrad = domain.pressureResultGrad
            else:
                if _LOG_DEBUG:
                    _LOG.debug("copy pressure to backward empty")
                pressureResultGrad = None

            return pressureResultGrad

    CopyPressureResultToBlocksFunction.apply(domain.pressureResult)


def CopyEpotResultToBlocks(domain: _C.Domain) -> None:
    """Copy ``domain.epotResult`` to ``block.epot``, differentiably.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to operate on.
    """

    class CopyEpotResultToBlocksFunction(torch.autograd.Function):
        """Autograd node of :func:`CopyEpotResultToBlocks`.

        Scatters ``domain.epotResult`` onto ``block.epot``.
        """

        @staticmethod
        def forward(
            ctx: Any, epot_result: torch.Tensor
        ) -> tuple[torch.Tensor | None, ...]:
            """Copy the electric potential result onto the blocks.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            epot_result : torch.Tensor
                Flat potential result, ``domain.epotResult``.

            Returns
            -------
            tuple of (torch.Tensor or None)
                The per-block electric potential fields after the copy.
            """
            with SAMPLE("CopyEpotResultToBlocks-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("copy epot to blocks forward")

                domain.CreateEpotOnBlocks()
                domain.UpdateDomainData()
                _C.CopyEpotResultToBlocks(domain)

                return (*[block.epot for block in domain.getBlocks()],)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx: Any, *blocks_epot_grad: torch.Tensor) -> torch.Tensor | None:
            """Gather the per-block gradients back into a flat gradient.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            *blocks_epot_grad : torch.Tensor
                Gradient w.r.t. each block's electric potential field.

            Returns
            -------
            torch.Tensor or None
                Gradient w.r.t. ``epot_result``, or None if it needs no gradient.
            """
            if ctx.needs_input_grad[0]:
                with SAMPLE("CopyEpotResultToBlocks-BWD"):
                    if _LOG_DEBUG:
                        _LOG.debug("copy epot to blocks backward")
                    for block, e_grad in zip(
                        domain.getBlocks(), blocks_epot_grad, strict=False
                    ):
                        block.setEpotGrad(e_grad.contiguous())
                    domain.CreateEpotResultGrad()
                    domain.UpdateDomainData()
                    _C.CopyEpotResultGradFromBlocks(domain)
                    return domain.epotResultGrad
            else:
                if _LOG_DEBUG:
                    _LOG.debug("copy epot to blocks backward empty")
                return None

    CopyEpotResultToBlocksFunction.apply(domain.epotResult)


def _potential_value_bounds(domain: _C.Domain) -> list[_C.FixedBoundary]:
    """FIXED faces whose prescribed potential values reach the potential kernels.

    The values enter :func:`ComputeEpotRHS` and :func:`ComputeCurrentDensityFaceBased`
    at the face's Dirichlet cells, linearly and never through the matrix, so they
    are tracked inputs of both.
    """
    bounds = []
    for block in domain.getBlocks():
        for boundary_idx in range(domain.getSpatialDims() * 2):
            bound = block.getBoundary(boundary_idx)
            if (
                isinstance(bound, _C.FixedBoundary)
                and bound.hasPotentialValues()
                and bound.hasPotentialDirichlet()
            ):
                bounds.append(bound)
    return bounds


def _potential_values_grad(
    domain: _C.Domain,
    bounds: Sequence[_C.FixedBoundary],
    values: Sequence[tuple[torch.Size, torch.dtype]],
    needs_grad: Sequence[bool],
    run_grad_kernel: Callable[[], Any],
) -> tuple[Any, list[torch.Tensor | None]]:
    """Run a potential grad kernel, also collecting d/d(potential values).

    Every face that needs a gradient gets a zero buffer with one slice per
    environment, which the kernel accumulates into; a value tensor shared by all
    environments (batch 1) gets the sum over them.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain the kernel runs on.
    bounds : Sequence of phipict._C.FixedBoundary
        Faces the values belong to, see :func:`_potential_value_bounds`.
    values : Sequence of tuple of torch.Size and torch.dtype
        Shape and dtype of the potential values the forward pass read, one per
        face.
    needs_grad : Sequence of bool
        Whether each face's values need a gradient.
    run_grad_kernel : Callable
        Runs the grad kernel and returns its result.

    Returns
    -------
    tuple
        The kernel's result, and the gradient of every face's values (None
        where not needed).
    """
    batch = domain.getBatchSize()
    grads: list[torch.Tensor | None] = [None] * len(bounds)
    for k, (bound, (shape, _)) in enumerate(zip(bounds, values, strict=True)):
        if needs_grad[k]:
            grad = torch.zeros(
                (batch, *shape[1:]),
                dtype=domain.getDtype(),
                device=domain.getDevice(),
            )
            bound.setPotentialValuesGrad(grad)
            grads[k] = grad
    if not any(needs_grad):
        return run_grad_kernel(), grads

    domain.UpdateDomainData()
    try:
        result = run_grad_kernel()
    finally:
        for bound, buffer in zip(bounds, grads, strict=True):
            if buffer is not None:
                bound.clearPotentialValuesGrad()
        domain.UpdateDomainData()

    for k, ((shape, dtype), buffer) in enumerate(zip(values, grads, strict=True)):
        if buffer is not None:
            if shape[0] != buffer.shape[0]:
                buffer = buffer.sum(dim=0, keepdim=True)
            grads[k] = buffer.to(dtype)
    return result, grads


_EXCLUDED_GRADIENTS["ComputeEpotRHS"] = (set(), set())


def ComputeEpotRHS(domain: _C.Domain, u_cross_eb_flat: torch.Tensor) -> torch.Tensor:
    """Compute the right-hand side of the electric potential Poisson equation.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to operate on.
    u_cross_eb_flat : torch.Tensor
        Flattened ``u x e_b`` field.

    Returns
    -------
    torch.Tensor
        Divergence of ``u x e_b`` plus the Dirichlet potential values' term,
        differentiable w.r.t. ``u_cross_eb_flat`` and the potential values of
        the domain's FIXED faces.
    """
    bounds = _potential_value_bounds(domain)

    class ComputeEpotRHSFunction(torch.autograd.Function):
        """Autograd node of :func:`ComputeEpotRHS`.

        Computes the divergence of ``u x e_b``, the source term of the electric
        potential Poisson equation, with the prescribed potential values of the
        Dirichlet face cells moved to it.
        """

        @staticmethod
        def forward(
            ctx: Any, vec_field: torch.Tensor, *values: torch.Tensor
        ) -> torch.Tensor:
            """Compute the divergence of a flat vector field.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            vec_field : torch.Tensor
                Flat cell-centred vector field, typically ``u x e_b``.
            *values : torch.Tensor
                The potential values of ``bounds``; the kernel reads them from
                the domain, they are inputs for the gradient only.

            Returns
            -------
            torch.Tensor
                The divergence of ``vec_field``.
            """
            with SAMPLE("ComputeEpotRHS-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("ComputeEpotRHS forward")
                ctx.values = [(v.shape, v.dtype) for v in values]
                return _C.ComputeEpotRHS(domain, vec_field.contiguous())

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(
            ctx: Any, grad_divergence: torch.Tensor
        ) -> tuple[torch.Tensor | None, ...]:
            """Apply the adjoint of the divergence operator.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            grad_divergence : torch.Tensor
                Gradient w.r.t. the divergence.

            Returns
            -------
            tuple of torch.Tensor or None
                Gradient w.r.t. ``vec_field`` and each of the potential values,
                None where not needed.
            """
            with SAMPLE("ComputeEpotRHS-BWD"):
                if _LOG_DEBUG:
                    _LOG.debug("ComputeEpotRHS backward")
                grad_divergence = grad_divergence.contiguous()
                grad_vec, grad_values = _potential_values_grad(
                    domain,
                    bounds,
                    ctx.values,
                    ctx.needs_input_grad[1:],
                    lambda: _C.ComputeEpotRHSGrad(domain, grad_divergence),
                )
                if not ctx.needs_input_grad[0]:
                    grad_vec = None
                return (grad_vec, *grad_values)

    values = [bound.potentialValues for bound in bounds]
    return ComputeEpotRHSFunction.apply(u_cross_eb_flat, *values)


_EXCLUDED_GRADIENTS["ComputeCurrentDensityFaceBased"] = (set(), set())


def ComputeCurrentDensityFaceBased(
    domain: _C.Domain, epot_result: torch.Tensor, u_cross_eb_flat: torch.Tensor
) -> torch.Tensor:
    """Compute the face-based current density ``J = -grad(phi) + u x e_b``.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to operate on.
    epot_result : torch.Tensor
        Electric potential.
    u_cross_eb_flat : torch.Tensor
        Flattened ``u x e_b`` field.

    Returns
    -------
    torch.Tensor
        Discretely divergence-free current density of shape
        ``[3 * totalSize]``, differentiable w.r.t. both inputs and the potential
        values of the domain's FIXED faces.
    """
    bounds = _potential_value_bounds(domain)

    class ComputeCurrentDensityFaceBasedFunction(torch.autograd.Function):
        """Autograd node of :func:`ComputeCurrentDensityFaceBased`.

        Computes ``J = -grad(phi) + u x e_b`` from face fluxes, so the result is
        discretely divergence-free.
        """

        @staticmethod
        def forward(
            ctx: Any,
            epot: torch.Tensor,
            u_cross_eb: torch.Tensor,
            *values: torch.Tensor,
        ) -> torch.Tensor:
            """Compute the face-based current density.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            epot : torch.Tensor
                Flat electric potential field.
            u_cross_eb : torch.Tensor
                Flat ``u x e_b`` field of shape ``[3 * totalSize]``.
            *values : torch.Tensor
                The potential values of ``bounds``; the kernel reads them from
                the domain, they are inputs for the gradient only.

            Returns
            -------
            torch.Tensor
                Current density of shape ``[3 * totalSize]``.
            """
            with SAMPLE("ComputeCurrentDensityFaceBased-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("ComputeCurrentDensityFaceBased forward")
                epot = epot.contiguous()
                u_cross_eb = u_cross_eb.contiguous()
                ctx.save_for_backward(epot, u_cross_eb)
                ctx.values = [(v.shape, v.dtype) for v in values]
                return _C.ComputeCurrentDensityFaceBased(domain, epot, u_cross_eb)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx: Any, grad_J: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
            """Apply the adjoint of the face-based current density operator.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            grad_J : torch.Tensor
                Gradient w.r.t. the current density.

            Returns
            -------
            tuple of torch.Tensor or None
                Gradients w.r.t. ``epot``, ``u_cross_eb`` and each of the
                potential values, None where not needed.
            """
            epot, u_cross_eb = ctx.saved_tensors
            grad_J = grad_J.contiguous()
            (grad_epot, grad_ucb), grad_values = _potential_values_grad(
                domain,
                bounds,
                ctx.values,
                ctx.needs_input_grad[2:],
                lambda: _C.ComputeCurrentDensityFaceBasedGrad(
                    domain, epot, u_cross_eb, grad_J
                ),
            )
            return (grad_epot, grad_ucb, *grad_values)

    values = [bound.potentialValues for bound in bounds]
    return ComputeCurrentDensityFaceBasedFunction.apply(
        epot_result, u_cross_eb_flat, *values
    )


_EXCLUDED_GRADIENTS["SetupPressureCorrection"] = (set(), set())


def SetupPressureCorrection(
    domain: _C.Domain,
    time_step: torch.Tensor,
    non_ortho_flags: int,
    use_face_transform: bool = False,
    timeStepNorm: bool = False,
) -> None:
    """Set up the pressure matrix and right-hand side, differentiably.

    Combination of :func:`SetupPressureMatrix` and :func:`SetupPressureRHS`
    (which includes :func:`SetupPressureRHSdiv`). Writes ``domain.P``,
    ``domain.pressureRHS`` and ``domain.pressureRHSdiv``.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to operate on.
    time_step : torch.Tensor
        Time step, a 1-element tensor.
    non_ortho_flags : int
        Non-orthogonal handling flags, see :func:`check_non_ortho_rhs`.
    use_face_transform : bool, optional
        Whether to use face transformations. Gradients do not support them.
        Default is False.
    timeStepNorm : bool, optional
        Whether the pressure is normalized by the time step. Default is False.
    """
    is_non_ortho = check_non_ortho_rhs(non_ortho_flags)

    tracked_tensor_filter = [
        "VELOCITY",
        "VELOCITY_SOURCE",
        "VELOCITY_RESULT",
        "BOUNDARY_VELOCITY",
        "VISCOSITY",
        "VISCOSITY_BLOCK",
        "A",
        "C",
    ]  # "C"
    if is_non_ortho:
        tracked_tensor_filter.append("PRESSURE_RESULT")

    grad_tensor_filter = [s + "_GRAD" for s in tracked_tensor_filter]

    domain_dict, tracked_tensors = flatten_domain(domain, tracked_tensor_filter)

    backwards_tensor_filter = [
        "BOUNDARY_VELOCITY",
        "VISCOSITY",
        "VISCOSITY_BLOCK",
        "A",
        "C",
        "VELOCITY_RESULT",
        "PRESSURE_RHS",
    ]
    backwards_tensor_filter.append(
        "VELOCITY_SOURCE"
    )  # DEBUG: value not needed but needs to be set to receive gradients
    if is_non_ortho:
        tracked_tensor_filter.append("PRESSURE_RESULT")

    class SetupPressureCorrectionFunction(torch.autograd.Function):
        """Autograd node of :func:`SetupPressureCorrection`.

        Writes ``domain.P``, ``domain.pressureRHS`` and ``domain.pressureRHSdiv``.
        """

        @staticmethod
        def forward(
            ctx: Any, *tracked_tensors: torch.Tensor | None
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Run the SetupPressureCorrection kernel on the domain.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            *tracked_tensors : torch.Tensor or None
                Domain tensors autograd tracks for this op. The kernels read them
                from the domain itself; they are passed through so autograd records
                the dependency and routes the gradients back to them.

            Returns
            -------
            P_value : torch.Tensor
                Pressure matrix values, ``domain.P.value``.
            pressureRHS : torch.Tensor
                Pressure right-hand side, ``domain.pressureRHS``.
            pressureRHSdiv : torch.Tensor
                Its divergence, ``domain.pressureRHSdiv``.
            """
            with SAMPLE("SetupPressureCorrection-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("setup pressure forward")

                domain.CreatePressureRHS()
                domain.CreatePressureRHSdiv()
                domain.P.CreateValue()
                domain.UpdateDomainData()
                _C.SetupPressureCorrection(
                    domain,
                    time_step,
                    non_ortho_flags,
                    use_face_transform,
                    timeStepNorm=timeStepNorm,
                )

                if backwards_tensor_filter:
                    domain_dict, saved_tensors = flatten_domain(
                        domain, backwards_tensor_filter
                    )
                    ctx.saved_tensors_domain_dict = domain_dict
                else:
                    ctx.saved_tensors_domain_dict = None
                    saved_tensors = []

                saved_tensors.append(time_step)
                ctx.save_for_backward(*saved_tensors)

            return domain.P.value, domain.pressureRHS, domain.pressureRHSdiv

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(
            ctx: Any,
            P_value_grad: torch.Tensor,
            pressureRHS_grad: torch.Tensor,
            pressureRHSdiv_grad: torch.Tensor,
        ) -> tuple[torch.Tensor | None, ...]:
            # if not non_ortho_flags==0: raise NotImplementedError("SetupPressureCorrection: Only Orthogonal gradients are supported.")
            """Propagate the gradients of SetupPressureCorrection back to the domain tensors.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            P_value_grad : torch.Tensor
                Gradient w.r.t. ``domain.P.value``.
            pressureRHS_grad : torch.Tensor
                Gradient w.r.t. ``domain.pressureRHS``.
            pressureRHSdiv_grad : torch.Tensor
                Gradient w.r.t. ``domain.pressureRHSdiv``.

            Returns
            -------
            tuple of (torch.Tensor or None)
                Gradients w.r.t. ``tracked_tensors``, in the same order. An entry
                is None where the corresponding input needs no gradient.
            """
            if use_face_transform:
                raise NotImplementedError(
                    "SetupPressureCorrection: Gradient does not support face transformations."
                )
            # if not timeStepNorm: raise NotImplementedError("SetupPressureCorrection: Gradient expect timeStepNorm.")

            # velocity_result_grad = None
            # block_velocity_grad = [None] * domain.getNumBlocks()

            if any(ctx.needs_input_grad):
                with SAMPLE("SetupPressureCorrection-BWD"):
                    if _LOG_DEBUG:
                        _LOG.debug("setup pressure backward")

                    time_step = ctx.saved_tensors[-1]
                    if ctx.saved_tensors_domain_dict is not None:
                        set_domain_tensors_from_flat(
                            domain, ctx.saved_tensors_domain_dict, ctx.saved_tensors
                        )

                    # domain.PGrad.setValue(P_value_grad)
                    _set_gradient_input(
                        P_value_grad,
                        domain.PGrad.setValue,
                        "SetupPressureCorrection",
                        "P_GRAD",
                    )
                    # domain.setPressureRHSGrad(pressureRHS_grad)
                    _set_gradient_input(
                        pressureRHS_grad,
                        domain.setPressureRHSGrad,
                        "SetupPressureCorrection",
                        "PRESSURE_RHS_GRAD",
                    )
                    # domain.setPressureRHSdivGrad(pressureRHSdiv_grad)
                    _set_gradient_input(
                        pressureRHSdiv_grad,
                        domain.setPressureRHSdivGrad,
                        "SetupPressureCorrection",
                        "PRESSURE_RHS_DIV_GRAD",
                    )
                    domain.CreateAGrad()
                    domain.CGrad.CreateValue()
                    domain.CreateViscosityGrad()
                    domain.CreateVelocityResultGrad()
                    domain.CreateVelocityGradOnBlocks()
                    domain.CreateVelocitySourceGradOnBlocks()  # only create grad if velocity source exists, clears it otherwise
                    domain.CreateVelocityGradOnBoundaries()
                    if is_non_ortho:
                        domain.CreatePressureResultGrad()
                    domain.UpdateDomainData()

                    _C.SetupPressureCorrectionGrad(
                        domain,
                        time_step,
                        non_ortho_flags,
                        use_face_transform,
                        timeStepNorm,
                    )
                    _, grad_tensors = flatten_domain(
                        domain,
                        grad_tensor_filter,
                        exclusion_list=_EXCLUDED_GRADIENTS["SetupPressureCorrection"][
                            1
                        ],
                    )
            else:
                if _LOG_DEBUG:
                    _LOG.debug("setup pressure backward empty")
                _, grad_tensors = flatten_domain(domain, grad_tensor_filter, empty=True)

            return (*grad_tensors,)

    # block_velocity_data = [block.velocity for block in domain.getBlocks()]
    # if is_non_ortho:
    # block_velocity_data.append(domain.pressureResult)
    SetupPressureCorrectionFunction.apply(*tracked_tensors)


_EXCLUDED_GRADIENTS["SetupPressureMatrix"] = (set(), set())


def SetupPressureMatrix(
    domain: _C.Domain,
    time_step: torch.Tensor,
    non_ortho_flags: int,
    use_face_transform: bool = False,
) -> None:
    """Set up the pressure matrix ``domain.P`` from ``domain.A``, differentiably.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to operate on.
    time_step : torch.Tensor
        Time step, a 1-element tensor.
    non_ortho_flags : int
        Non-orthogonal handling flags, see :func:`check_non_ortho_rhs`.
    use_face_transform : bool, optional
        Whether to use face transformations. Gradients do not support them.
        Default is False.
    """
    # is_non_ortho = check_non_ortho_rhs(non_ortho_flags)

    tracked_tensor_filter = ["A"]

    grad_tensor_filter = [s + "_GRAD" for s in tracked_tensor_filter]

    domain_dict, tracked_tensors = flatten_domain(domain, tracked_tensor_filter)

    backwards_tensor_filter = ["A"]

    class SetupPressureMatrixFunction(torch.autograd.Function):
        """Autograd node of :func:`SetupPressureMatrix`.

        Builds ``domain.P`` from the advection matrix diagonal ``domain.A``.
        """

        @staticmethod
        def forward(ctx: Any, A_diag: torch.Tensor) -> torch.Tensor:
            """Build the pressure matrix from the advection matrix diagonal.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            A_diag : torch.Tensor
                Advection matrix diagonal, ``domain.A``.

            Returns
            -------
            torch.Tensor
                Pressure matrix values, ``domain.P.value``.
            """
            with SAMPLE("SetupPressureMatrix-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("setup pressure matrix forward")

                domain.P.CreateValue()
                domain.UpdateDomainData()
                _C.SetupPressureMatrix(
                    domain, time_step, non_ortho_flags, use_face_transform
                )

                if backwards_tensor_filter:
                    domain_dict, saved_tensors = flatten_domain(
                        domain, backwards_tensor_filter
                    )
                    ctx.saved_tensors_domain_dict = domain_dict
                else:
                    ctx.saved_tensors_domain_dict = None
                    saved_tensors = []

                saved_tensors.append(time_step)
                ctx.save_for_backward(*saved_tensors)

            return domain.P.value

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(
            ctx: Any, P_value_grad: torch.Tensor
        ) -> tuple[torch.Tensor | None, ...]:
            """Propagate the pressure matrix gradient back to ``domain.A``.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            P_value_grad : torch.Tensor
                Gradient w.r.t. ``domain.P.value``.

            Returns
            -------
            tuple of (torch.Tensor or None)
                A one-element tuple holding the gradient w.r.t. ``A_diag``, or None
                if it needs no gradient.
            """
            if _LOG_DEBUG:
                _LOG.debug("setup pressure matrix backward (empty)")
            # if not non_ortho_flags==0: raise NotImplementedError("SetupPressureMatrix: Only Orthogonal gradients are supported.")
            if use_face_transform:
                raise NotImplementedError(
                    "SetupPressureMatrix: Gradient does not support face transformations."
                )

            A_diag_grad = None
            if any(ctx.needs_input_grad):
                with SAMPLE("SetupPressureMatrix-BWD"):
                    time_step = ctx.saved_tensors[-1]
                    if ctx.saved_tensors_domain_dict is not None:
                        set_domain_tensors_from_flat(
                            domain, ctx.saved_tensors_domain_dict, ctx.saved_tensors
                        )

                    # domain.PGrad.setValue(P_value_grad)
                    _set_gradient_input(
                        P_value_grad,
                        domain.PGrad.setValue,
                        "SetupPressureMatrix",
                        "P_GRAD",
                    )
                    domain.CreateAGrad()
                    domain.UpdateDomainData()

                    _C.SetupPressureMatrixGrad(
                        domain, time_step, non_ortho_flags, use_face_transform
                    )

                    # A_diag_grad = domain.AGrad
                    _, grad_tensors = flatten_domain(
                        domain,
                        grad_tensor_filter,
                        exclusion_list=_EXCLUDED_GRADIENTS["SetupPressureMatrix"][1],
                    )

            return (*grad_tensors,)

    SetupPressureMatrixFunction.apply(domain.A)


_EXCLUDED_GRADIENTS["SetupPressureRHS"] = (set(), set())


def SetupPressureRHS(
    domain: _C.Domain,
    time_step: torch.Tensor,
    non_ortho_flags: int,
    use_face_transform: bool = False,
    timeStepNorm: bool = False,
) -> None:
    """Set up the pressure right-hand side, differentiably.

    Includes :func:`SetupPressureRHSdiv`. Writes ``domain.pressureRHS`` and
    ``domain.pressureRHSdiv``.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to operate on.
    time_step : torch.Tensor
        Time step, a 1-element tensor.
    non_ortho_flags : int
        Non-orthogonal handling flags, see :func:`check_non_ortho_rhs`.
    use_face_transform : bool, optional
        Whether to use face transformations. Gradients do not support them.
        Default is False.
    timeStepNorm : bool, optional
        Whether the pressure is normalized by the time step. Default is False.

    Notes
    -----
    Inputs:

    - block.velocity (not needed for backwards) [u from before advection]
    - block.velocitySource (not needed for backwards)
    - block.velocityResult [u* output from advection]
    - domain.C
    - domain.A
    - block.transform (must be static, not differentiable)
    - domain.viscosity
    - block.viscosity
    - boundary.velocity
    - boundary.transform (must be static, not differentiable)
    """
    is_non_ortho = check_non_ortho_rhs(non_ortho_flags)

    tracked_tensor_filter = [
        "VELOCITY",
        "VELOCITY_SOURCE",
        "VELOCITY_RESULT",
        "BOUNDARY_VELOCITY",
        "VISCOSITY",
        "VISCOSITY_BLOCK",
        "C",
        "A",
    ]  #
    if is_non_ortho:
        tracked_tensor_filter.append("PRESSURE_RESULT")

    grad_tensor_filter = [s + "_GRAD" for s in tracked_tensor_filter]

    domain_dict, tracked_tensors = flatten_domain(domain, tracked_tensor_filter)

    backwards_tensor_filter = [
        "BOUNDARY_VELOCITY",
        "VISCOSITY",
        "VISCOSITY_BLOCK",
        "A",
        "C",
        "VELOCITY_RESULT",
        "PRESSURE_RHS",
    ]
    backwards_tensor_filter.append(
        "VELOCITY_SOURCE"
    )  # DEBUG: value not needed but needs to be set to receive gradients
    # if is_non_ortho:
    #    tracked_tensor_filter.append("PRESSURE_RESULT")

    class SetupPressureRHSFunction(torch.autograd.Function):
        """Autograd node of :func:`SetupPressureRHS`.

        Writes ``domain.pressureRHS`` and ``domain.pressureRHSdiv``.
        """

        @staticmethod
        def forward(
            ctx: Any, *tracked_tensors: torch.Tensor | None
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Run the SetupPressureRHS kernel on the domain.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            *tracked_tensors : torch.Tensor or None
                Domain tensors autograd tracks for this op. The kernels read them
                from the domain itself; they are passed through so autograd records
                the dependency and routes the gradients back to them.

            Returns
            -------
            pressureRHS : torch.Tensor
                Pressure right-hand side, ``domain.pressureRHS``.
            pressureRHSdiv : torch.Tensor
                Its divergence, ``domain.pressureRHSdiv``.
            """
            with SAMPLE("SetupPressureRHS-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("setup pressure RHS forward")

                domain.CreatePressureRHS()
                domain.CreatePressureRHSdiv()
                domain.UpdateDomainData()
                _C.SetupPressureRHS(
                    domain,
                    time_step,
                    non_ortho_flags,
                    use_face_transform,
                    timeStepNorm=timeStepNorm,
                )

                if backwards_tensor_filter:
                    domain_dict, saved_tensors = flatten_domain(
                        domain, backwards_tensor_filter
                    )
                    ctx.saved_tensors_domain_dict = domain_dict
                else:
                    ctx.saved_tensors_domain_dict = None
                    saved_tensors = []

                saved_tensors.append(time_step)
                ctx.save_for_backward(*saved_tensors)

            return domain.pressureRHS, domain.pressureRHSdiv

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(
            ctx: Any, pressureRHS_grad: torch.Tensor, pressureRHSdiv_grad: torch.Tensor
        ) -> tuple[torch.Tensor | None, ...]:
            # if not non_ortho_flags==0: raise NotImplementedError("SetupPressureRHS: Only Orthogonal gradients are supported.")
            """Propagate the gradients of SetupPressureRHS back to the domain tensors.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            pressureRHS_grad : torch.Tensor
                Gradient w.r.t. ``domain.pressureRHS``.
            pressureRHSdiv_grad : torch.Tensor
                Gradient w.r.t. ``domain.pressureRHSdiv``.

            Returns
            -------
            tuple of (torch.Tensor or None)
                Gradients w.r.t. ``tracked_tensors``, in the same order. An entry
                is None where the corresponding input needs no gradient.
            """
            if use_face_transform:
                raise NotImplementedError(
                    "SetupPressureRHS: Gradient does not support face transformations."
                )
            # if not timeStepNorm: raise NotImplementedError("SetupPressureRHS: Gradient expect timeStepNorm.")

            # velocity_result_grad = None
            # block_velocity_grad = [None] * domain.getNumBlocks()

            if any(ctx.needs_input_grad):
                with SAMPLE("SetupPressureRHS-BWD"):
                    if _LOG_DEBUG:
                        _LOG.debug("setup pressure RHS backward")

                    time_step = ctx.saved_tensors[-1]
                    if ctx.saved_tensors_domain_dict is not None:
                        set_domain_tensors_from_flat(
                            domain, ctx.saved_tensors_domain_dict, ctx.saved_tensors
                        )

                    # domain.setPressureRHSGrad(pressureRHS_grad)
                    _set_gradient_input(
                        pressureRHS_grad,
                        domain.setPressureRHSGrad,
                        "SetupPressureRHS",
                        "PRESSURE_RHS_GRAD",
                    )
                    # domain.setPressureRHSdivGrad(pressureRHSdiv_grad)
                    _set_gradient_input(
                        pressureRHSdiv_grad,
                        domain.setPressureRHSdivGrad,
                        "SetupPressureRHS",
                        "PRESSURE_RHS_DIV_GRAD",
                    )
                    domain.CreateAGrad()
                    domain.CGrad.CreateValue()
                    domain.CreateViscosityGrad()
                    domain.CreateVelocityResultGrad()
                    domain.CreateVelocityGradOnBlocks()
                    domain.CreateVelocitySourceGradOnBlocks()  # only create grad if velocity source exists, clears it otherwise
                    domain.CreateVelocityGradOnBoundaries()
                    if is_non_ortho:
                        domain.CreatePressureResultGrad()
                    domain.UpdateDomainData()

                    _C.SetupPressureRHSGrad(
                        domain,
                        time_step,
                        non_ortho_flags,
                        use_face_transform,
                        timeStepNorm,
                    )
                    _, grad_tensors = flatten_domain(
                        domain,
                        grad_tensor_filter,
                        exclusion_list=_EXCLUDED_GRADIENTS["SetupPressureRHS"][1],
                    )
            else:
                if _LOG_DEBUG:
                    _LOG.debug("setup pressure backward empty")
                _, grad_tensors = flatten_domain(domain, grad_tensor_filter, empty=True)

            return (*grad_tensors,)

    SetupPressureRHSFunction.apply(*tracked_tensors)


_EXCLUDED_GRADIENTS["SetupPressureRHSdiv"] = (set(), set())


def SetupPressureRHSdiv(
    domain: _C.Domain,
    time_step: torch.Tensor,
    non_ortho_flags: int,
    use_face_transform: bool = False,
    timeStepNorm: bool = False,
) -> None:
    """Compute the divergence of ``domain.pressureRHS``, differentiably.

    Adds the non-orthogonal right-hand side components if non-orthogonal
    handling is enabled. Writes ``domain.pressureRHSdiv``.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to operate on.
    time_step : torch.Tensor
        Time step, a 1-element tensor.
    non_ortho_flags : int
        Non-orthogonal handling flags, see :func:`check_non_ortho_rhs`.
    use_face_transform : bool, optional
        Whether to use face transformations. Gradients do not support them.
        Default is False.
    timeStepNorm : bool, optional
        Whether the pressure is normalized by the time step. Default is False.

    Notes
    -----
    Inputs:

    - domain.A (if non-orthogonal)
    - domain.pressureResult (if non-orthogonal)
    - domain.pressureRHS (not needed for backwards)
    - block.transform (must be static, not differentiable)
    - boundary.velocity
    - boundary.transform (must be static, not differentiable)
    """
    is_non_ortho = check_non_ortho_rhs(non_ortho_flags)

    tracked_tensor_filter = ["PRESSURE_RHS", "BOUNDARY_VELOCITY"]  # "A"
    if is_non_ortho:
        tracked_tensor_filter.append("PRESSURE_RESULT")
        tracked_tensor_filter.append("A")

    grad_tensor_filter = [s + "_GRAD" for s in tracked_tensor_filter]

    domain_dict, tracked_tensors = flatten_domain(domain, tracked_tensor_filter)

    backwards_tensor_filter: list[str] = []
    if is_non_ortho:
        backwards_tensor_filter.append("PRESSURE_RESULT")
        backwards_tensor_filter.append("A")

    class SetupPressureRHSdivFunction(torch.autograd.Function):
        """Autograd node of :func:`SetupPressureRHSdiv`.

        Writes ``domain.pressureRHSdiv``.
        """

        @staticmethod
        def forward(ctx: Any, *tracked_tensors: torch.Tensor | None) -> torch.Tensor:
            """Run the SetupPressureRHSdiv kernel on the domain.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            *tracked_tensors : torch.Tensor or None
                Domain tensors autograd tracks for this op. The kernels read them
                from the domain itself; they are passed through so autograd records
                the dependency and routes the gradients back to them.

            Returns
            -------
            torch.Tensor
                The divergence of the pressure right-hand side,
                ``domain.pressureRHSdiv``.
            """
            with SAMPLE("SetupPressureRHSdiv-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("setup pressure RHS div forward")

                domain.CreatePressureRHSdiv()
                domain.UpdateDomainData()
                _C.SetupPressureRHSdiv(
                    domain,
                    time_step,
                    non_ortho_flags,
                    use_face_transform,
                    timeStepNorm=timeStepNorm,
                )

                if backwards_tensor_filter:
                    domain_dict, saved_tensors = flatten_domain(
                        domain, backwards_tensor_filter
                    )
                    ctx.saved_tensors_domain_dict = domain_dict
                else:
                    ctx.saved_tensors_domain_dict = None
                    saved_tensors = []

                saved_tensors.append(time_step)
                ctx.save_for_backward(*saved_tensors)

            return domain.pressureRHSdiv

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(
            ctx: Any, pressureRHSdiv_grad: torch.Tensor
        ) -> tuple[torch.Tensor | None, ...]:
            # if not non_ortho_flags==0: raise NotImplementedError("SetupPressureRHSdiv: Only Orthogonal gradients are supported.")
            """Propagate the gradients of SetupPressureRHSdiv back to the domain tensors.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            pressureRHSdiv_grad : torch.Tensor
                Gradient w.r.t. ``domain.pressureRHSdiv``.

            Returns
            -------
            tuple of (torch.Tensor or None)
                Gradients w.r.t. ``tracked_tensors``, in the same order. An entry
                is None where the corresponding input needs no gradient.
            """
            if use_face_transform:
                raise NotImplementedError(
                    "SetupPressureRHSdiv: Gradient does not support face transformations."
                )
            # if not timeStepNorm: raise NotImplementedError("SetupPressureRHSdiv: Gradient expect timeStepNorm.")

            # pressureRHS_grad = None

            if any(ctx.needs_input_grad):
                with SAMPLE("SetupPressureRHSdiv-BWD"):
                    if _LOG_DEBUG:
                        _LOG.debug("setup pressure RHS div backward")

                    time_step = ctx.saved_tensors[-1]
                    if ctx.saved_tensors_domain_dict is not None:
                        set_domain_tensors_from_flat(
                            domain, ctx.saved_tensors_domain_dict, ctx.saved_tensors
                        )

                    # domain.setPressureRHSdivGrad(pressureRHSdiv_grad)
                    _set_gradient_input(
                        pressureRHSdiv_grad,
                        domain.setPressureRHSdivGrad,
                        "SetupPressureRHSdiv",
                        "PRESSURE_RHS_DIV_GRAD",
                    )
                    if is_non_ortho:
                        domain.CreateAGrad()
                    domain.CreatePressureRHSGrad()
                    domain.CreateVelocityGradOnBoundaries()
                    if is_non_ortho:
                        domain.CreatePressureResultGrad()
                    domain.UpdateDomainData()

                    _C.SetupPressureRHSdivGrad(
                        domain,
                        time_step,
                        non_ortho_flags,
                        use_face_transform,
                        timeStepNorm,
                    )
                    _, grad_tensors = flatten_domain(
                        domain,
                        grad_tensor_filter,
                        exclusion_list=_EXCLUDED_GRADIENTS["SetupPressureRHSdiv"][1],
                    )
            else:
                if _LOG_DEBUG:
                    _LOG.debug("setup pressure backward empty")
                _, grad_tensors = flatten_domain(domain, grad_tensor_filter, empty=True)

            return tuple(grad_tensors)

    SetupPressureRHSdivFunction.apply(*tracked_tensors)


_EXCLUDED_GRADIENTS["CorrectVelocity"] = (set(), set())


def CorrectVelocity(
    domain: _C.Domain,
    time_step: torch.Tensor,
    version: int,
    timeStepNorm: bool = False,
    exclude_pressure_grad: bool = False,
) -> None:
    """Apply the pressure correction to the velocity, differentiably.

    Writes ``domain.velocityResult``.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to operate on.
    time_step : torch.Tensor
        Time step, a 1-element tensor.
    version : int
        Pressure gradient discretization. Gradients are only implemented for
        the finite difference versions 0 and 1.
    timeStepNorm : bool, optional
        Whether the pressure is normalized by the time step. Default is False.
    exclude_pressure_grad : bool, optional
        Whether to drop ``PRESSURE_GRAD`` from this call's backward only,
        without touching the global exclusions. Default is False.

    Notes
    -----
    Inputs:

    - block.pressure
    - domain.A
    - domain.pressureRHS (not needed for backwards)
    - block.transform (must be static, not differentiable)
    - boundary.transform (must be static, not differentiable)
    """
    # is_non_ortho = check_non_ortho_rhs(non_ortho_flags)

    tracked_tensor_filter = ["A", "PRESSURE", "PRESSURE_RHS"]

    grad_tensor_filter = [s + "_GRAD" for s in tracked_tensor_filter]

    domain_dict, tracked_tensors = flatten_domain(domain, tracked_tensor_filter)

    backwards_tensor_filter = ["A", "PRESSURE"]

    class CorrectVelocityFunction(torch.autograd.Function):
        """Autograd node of :func:`CorrectVelocity`.

        Writes ``domain.velocityResult``.
        """

        @staticmethod
        def forward(ctx: Any, *tracked_tensors: torch.Tensor | None) -> torch.Tensor:
            """Run the CorrectVelocity kernel on the domain.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            *tracked_tensors : torch.Tensor or None
                Domain tensors autograd tracks for this op. The kernels read them
                from the domain itself; they are passed through so autograd records
                the dependency and routes the gradients back to them.

            Returns
            -------
            torch.Tensor
                The corrected, divergence-free velocity, ``domain.velocityResult``.
            """
            with SAMPLE("CorrectVelocity-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("correct velocity forward")

                domain.CreateVelocityResult()
                domain.UpdateDomainData()
                _C.CorrectVelocity(
                    domain, time_step, version, timeStepNorm=timeStepNorm
                )

                if backwards_tensor_filter:
                    domain_dict, saved_tensors = flatten_domain(
                        domain, backwards_tensor_filter
                    )
                    ctx.saved_tensors_domain_dict = domain_dict
                else:
                    ctx.saved_tensors_domain_dict = None
                    saved_tensors = []

                saved_tensors.append(time_step)
                ctx.save_for_backward(*saved_tensors)

            return domain.velocityResult

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(
            ctx: Any, velocity_result_grad: torch.Tensor
        ) -> tuple[torch.Tensor | None, ...]:
            # if not timeStepNorm: raise NotImplementedError("CorrectVelocity: Gradient expects timeStepNorm.")
            """Propagate the gradients of CorrectVelocity back to the domain tensors.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            velocity_result_grad : torch.Tensor
                Gradient w.r.t. ``domain.velocityResult``.

            Returns
            -------
            tuple of (torch.Tensor or None)
                Gradients w.r.t. ``tracked_tensors``, in the same order. An entry
                is None where the corresponding input needs no gradient.
            """
            if version not in [0, 1]:
                raise NotImplementedError(
                    "CorrectVelocity: Gradient only implemented for Finite Difference pressure gradients (version 0 or 1)."
                )

            if any(ctx.needs_input_grad):
                with SAMPLE("CorrectVelocity-BWD"):
                    if _LOG_DEBUG:
                        _LOG.debug("correct velocity backward")

                    time_step = ctx.saved_tensors[-1]
                    if ctx.saved_tensors_domain_dict is not None:
                        set_domain_tensors_from_flat(
                            domain, ctx.saved_tensors_domain_dict, ctx.saved_tensors
                        )

                    # domain.setVelocityResultGrad(velocity_result_grad)
                    _set_gradient_input(
                        velocity_result_grad,
                        domain.setVelocityResultGrad,
                        "CorrectVelocity",
                        "VELOCITY_RESULT_GRAD",
                    )
                    domain.CreateAGrad()
                    domain.CreatePressureGradOnBlocks()
                    domain.CreatePressureRHSGrad()
                    domain.UpdateDomainData()

                    _C.CorrectVelocityGrad(domain, time_step, timeStepNorm)
                    exclusion_list = _EXCLUDED_GRADIENTS["CorrectVelocity"][1]
                    if exclude_pressure_grad:
                        exclusion_list = exclusion_list | {"PRESSURE_GRAD"}
                    _, grad_tensors = flatten_domain(
                        domain,
                        grad_tensor_filter,
                        exclusion_list=exclusion_list,
                    )
            else:
                if _LOG_DEBUG:
                    _LOG.debug("correct velocity backward empty")
                _, grad_tensors = flatten_domain(domain, grad_tensor_filter, empty=True)

            return (*grad_tensors,)

    CorrectVelocityFunction.apply(*tracked_tensors)


def reset_domain(domain: _C.Domain) -> None:
    """Recreate the field tensors of a domain.

    Storing the output of a custom gradient method in the domain object seems to
    prevent memory from being freed. Resetting all tensors of the domain (e.g. at
    the end of an optimization loop) fixes this kind of memory leak.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to operate on.
    """
    domain.CreateVelocityOnBlocks()
    domain.CreatePressureOnBlocks()
    domain.CreatePassiveScalarOnBlocks()
    domain.PrepareSolve()


def _detached(tensor: torch.Tensor | None) -> torch.Tensor:
    """Detach a domain tensor that is known to be set.

    Parameters
    ----------
    tensor : torch.Tensor or None
        Domain tensor, asserted to be set.

    Returns
    -------
    torch.Tensor
        The tensor, detached from the autograd graph.
    """
    assert tensor is not None
    return tensor.detach()


def detach_domain_fwd(domain: _C.Domain) -> None:
    """Detach all forward tensors of a domain from the autograd graph.

    This seems to be sufficient to allow the memory to be reclaimed. Run it after
    each optimization step on the domain that was used for backprop, possibly
    followed by ``gc.collect()`` when debugging memory usage. Alternative:
    ``domain.DetachFwd()``.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to operate on.
    """
    for block in domain.getBlocks():
        block.setVelocity(_detached(block.velocity))
        block.setPressure(_detached(block.pressure))
        block.setPassiveScalar(_detached(block.passiveScalar))
    domain.setA(_detached(domain.A))
    domain.C.setValue(domain.C.value.detach())
    domain.P.setValue(domain.P.value.detach())
    domain.setScalarRHS(_detached(domain.scalarRHS))
    domain.setScalarResult(_detached(domain.scalarResult))
    domain.setVelocityRHS(_detached(domain.velocityRHS))
    domain.setVelocityResult(_detached(domain.velocityResult))
    domain.setPressureRHS(_detached(domain.pressureRHS))
    domain.setPressureRHSdiv(_detached(domain.pressureRHSdiv))
    domain.setPressureResult(_detached(domain.pressureResult))
    if domain.hasEpot():
        for block in domain.getBlocks():
            if block.hasEpot():
                block.setEpot(_detached(block.epot))
        domain.setEpotRHS(_detached(domain.epotRHS))
        domain.setEpotResult(_detached(domain.epotResult))


def is_tensor_empty(tensor: torch.Tensor) -> bool:
    """Check whether a tensor is an empty 1D placeholder.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to check.

    Returns
    -------
    bool
        True if the tensor has shape ``[0]``.
    """
    return tensor.dim() == 1 and tensor.size(0) == 0


def detach_domain_grad(domain: _C.Domain) -> None:
    """Detach all gradient tensors of a domain from the autograd graph.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to operate on.
    """
    for block in domain.getBlocks():
        if not is_tensor_empty(block.velocityGrad):
            block.setVelocityGrad(_detached(block.velocityGrad))
        if not is_tensor_empty(block.pressureGrad):
            block.setPressureGrad(_detached(block.pressureGrad))
        if not is_tensor_empty(block.passiveScalarGrad):
            block.setPassiveScalarGrad(_detached(block.passiveScalarGrad))
    if not is_tensor_empty(domain.AGrad):
        domain.setAGrad(_detached(domain.AGrad))
    domain.CGrad.setValue(domain.CGrad.value.detach())
    if not is_tensor_empty(domain.scalarRHSGrad):
        domain.setScalarRHSGrad(_detached(domain.scalarRHSGrad))
    if not is_tensor_empty(domain.scalarResultGrad):
        domain.setScalarResultGrad(_detached(domain.scalarResultGrad))
    if not is_tensor_empty(domain.velocityRHSGrad):
        domain.setVelocityRHSGrad(_detached(domain.velocityRHSGrad))
    if not is_tensor_empty(domain.velocityResultGrad):
        domain.setVelocityResultGrad(_detached(domain.velocityResultGrad))
    if not is_tensor_empty(domain.pressureRHSGrad):
        domain.setPressureRHSGrad(_detached(domain.pressureRHSGrad))
    if not is_tensor_empty(domain.pressureRHSdivGrad):
        domain.setPressureRHSdivGrad(_detached(domain.pressureRHSdivGrad))
    if not is_tensor_empty(domain.pressureResultGrad):
        domain.setPressureResultGrad(_detached(domain.pressureResultGrad))
    if domain.hasEpotResultGrad():
        for block in domain.getBlocks():
            epot_grad = block.epotGrad
            if (
                block.hasEpotGrad()
                and epot_grad is not None
                and not is_tensor_empty(epot_grad)
            ):
                block.setEpotGrad(_detached(block.epotGrad))
        domain.setEpotResultGrad(_detached(domain.epotResultGrad))


def detach_domain(domain: _C.Domain) -> None:
    """Detach all forward and gradient tensors of a domain.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to operate on.
    """
    detach_domain_fwd(domain)
    detach_domain_grad(domain)


def matmul(
    vectorMatrixA: torch.Tensor,
    vectorMatrixB: torch.Tensor,
    transposeA: bool = False,
    invertA: bool = False,
    transposeB: bool = False,
    invertB: bool = False,
    transposeOutput: bool = False,
    invertOutput: bool = False,
) -> torch.Tensor:
    """Multiply per-cell matrices/vectors stored in the channel dimension.

    Parameters
    ----------
    vectorMatrixA : torch.Tensor
        NCDHW tensor with vectors or flat row-major matrices in the C dimension.
    vectorMatrixB : torch.Tensor
        NCDHW tensor with vectors or flat row-major matrices in the C dimension.
    transposeA, transposeB : bool, optional
        Transpose the matrix after loading. Only affects matrices.
        Default is False.
    invertA, invertB : bool, optional
        Invert the matrix after loading. Only affects matrices; not
        differentiable. Default is False.
    transposeOutput : bool, optional
        Transpose the result before writing. Default is False.
    invertOutput : bool, optional
        Invert the result before writing; not differentiable. Default is False.

    Returns
    -------
    torch.Tensor
        NCDHW tensor holding a matrix if both inputs are matrices, a vector if
        exactly one is a matrix, and a scalar if both are vectors.
    """

    class matmulFunction(torch.autograd.Function):
        """Autograd node of :func:`matmul`.

        Multiplies per-cell matrices and vectors stored in the channel dimension.
        """

        @staticmethod
        def forward(
            ctx: Any, vectorMatrixA: torch.Tensor, vectorMatrixB: torch.Tensor
        ) -> torch.Tensor:
            """Multiply the two per-cell operands.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            vectorMatrixA : torch.Tensor
                NCDHW tensor holding a per-cell matrix or vector.
            vectorMatrixB : torch.Tensor
                NCDHW tensor holding a per-cell matrix or vector.

            Returns
            -------
            torch.Tensor
                NCDHW tensor holding the product: a matrix if both inputs are
                matrices, a vector if exactly one is, and a scalar otherwise.
            """
            with SAMPLE("matmul-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("matmul forward")

                ctx.transposeA = transposeA
                ctx.invertA = invertA
                ctx.transposeB = transposeB
                ctx.invertB = invertB
                ctx.transposeOutput = transposeOutput
                ctx.invertOutput = invertOutput

                ctx.save_for_backward(vectorMatrixA, vectorMatrixB)

                result = _C.matmul(
                    vectorMatrixA,
                    vectorMatrixB,
                    transposeA,
                    invertA,
                    transposeB,
                    invertB,
                    transposeOutput,
                    invertOutput,
                )

            return result

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(
            ctx: Any, result_grad: torch.Tensor
        ) -> tuple[torch.Tensor | None, ...]:
            """Propagate the product gradient back to both operands.

            Parameters
            ----------
            ctx : Any
                Autograd context.
            result_grad : torch.Tensor
                Gradient w.r.t. the product.

            Returns
            -------
            tuple of (torch.Tensor or None)
                Gradients w.r.t. ``vectorMatrixA`` and ``vectorMatrixB``, each None
                if that input needs no gradient.
            """
            if any(ctx.needs_input_grad):
                with SAMPLE("matmul-BWD"):
                    if _LOG_DEBUG:
                        _LOG.debug("matmul backward")

                    if ctx.invertOutput:
                        raise NotImplementedError(
                            "Can not compute gradients for inverted matrices (output)."
                        )
                    if ctx.needs_input_grad[0] and ctx.invertA:
                        raise NotImplementedError(
                            "Can not compute gradients for inverted matrices (input A)."
                        )
                    if ctx.needs_input_grad[1] and ctx.invertB:
                        raise NotImplementedError(
                            "Can not compute gradients for inverted matrices (input B)."
                        )

                    vectorMatrixA, vectorMatrixB = ctx.saved_tensors

                    # returns 0 gradient if the matrix is inverted
                    vectorMatrixA_grad, vectorMatrixB_grad = _C.matmulGrad(
                        vectorMatrixA,
                        vectorMatrixB,
                        result_grad,
                        ctx.transposeA,
                        ctx.invertA,
                        ctx.transposeB,
                        ctx.invertB,
                        ctx.transposeOutput,
                        ctx.invertOutput,
                    )

                    grad_tensors = (
                        (vectorMatrixA_grad if ctx.needs_input_grad[0] else None),
                        (vectorMatrixB_grad if ctx.needs_input_grad[1] else None),
                    )
            else:
                if _LOG_DEBUG:
                    _LOG.debug("matmul backward empty")
                grad_tensors = (None, None)

            return (*grad_tensors,)

    return matmulFunction.apply(vectorMatrixA, vectorMatrixB)
