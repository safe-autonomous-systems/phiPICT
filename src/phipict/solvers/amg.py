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

"""Algebraic-multigrid preconditioning for the Poisson-type solves."""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import numpy as np
import torch

__all__ = [
    "AMGHierarchy",
    "BatchedAMGHierarchy",
    "AMGSolveInfo",
    "amg_pcg_solve",
    "build_amg_hierarchy",
    "csr_to_scipy",
    "batched_hierarchy_for",
    "clear_interpolation_cache",
    "galerkin_refresh",
    "hierarchy_for",
    "is_pyamg_available",
    "values_fingerprint",
    "native_amg_available",
]

_LOG = logging.getLogger("phipict.AMG")

# Weighted-Jacobi damping factor
_JACOBI_OMEGA = 2.0 / 3.0

# Diagonal entries below this are treated as 1.0 rather than inverted
_MIN_ABS_DIAG = 1e-30


def is_pyamg_available() -> bool:
    """Return whether the optional ``pyamg`` dependency can be imported.

    Returns
    -------
    bool
        True if ``pyamg`` is installed and importable.
    """
    try:
        import pyamg  # noqa: F401
    except ImportError:
        return False
    return True


def _load_pyamg() -> ModuleType:
    """Import ``pyamg``, with an actionable error if it is missing.

    Returns
    -------
    ModuleType
        The imported ``pyamg`` module.

    Raises
    ------
    ImportError
        If ``pyamg`` is not installed.
    """
    try:
        import pyamg
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "AMG preconditioning needs the 'pyamg' package for the setup phase. "
            "Install it with `pip install pyamg`."
        ) from exc
    return pyamg


# ---------------------------------------------------------------------------
# Matrix conversion
# ---------------------------------------------------------------------------


def csr_to_scipy(csr_mat: Any) -> Any:
    """Convert a ``_C.CSRmatrix`` to a host-side scipy CSR matrix.

    Parameters
    ----------
    csr_mat : phipict._C.CSRmatrix
        Device-side matrix to convert. Sentinel entries with a negative column
        index are dropped and the row pointers recomputed.

    Returns
    -------
    scipy.sparse.csr_matrix
        Host-side ``float64`` copy of the matrix.
    """
    import scipy.sparse as sp

    n = csr_mat.getRows()
    row = csr_mat.row.detach().cpu()
    index = csr_mat.index.detach().cpu()
    value = csr_mat.value.detach().cpu()

    keep = index >= 0
    n_dropped = int((~keep).sum())
    if n_dropped:
        _LOG.warning(
            "Dropping %d sentinel entries (column index < 0) while converting the "
            "matrix for the AMG setup.",
            n_dropped,
        )
        # Recount the entries per row before compacting, so the row pointers stay
        # consistent with the filtered value/index arrays
        row_of_entry = torch.repeat_interleave(
            torch.arange(n, dtype=torch.long), row.diff().long()
        )
        counts = torch.bincount(row_of_entry[keep], minlength=n)
        row = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(counts, dim=0)])
        index = index[keep]
        value = value[keep]

    return sp.csr_matrix(
        (
            value.to(torch.float64).numpy(),
            index.to(torch.int64).numpy(),
            row.to(torch.int64).numpy(),
        ),
        shape=(n, n),
    )


def _scipy_to_torch_csr(
    mat: Any, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Lift a scipy sparse matrix onto the device as a ``torch.sparse_csr`` tensor.

    Parameters
    ----------
    mat : scipy.sparse.spmatrix
        Host-side sparse matrix.
    device : torch.device
        Device to place the result on.
    dtype : torch.dtype
        Data type of the resulting values.

    Returns
    -------
    torch.Tensor
        Sparse CSR tensor with sorted int32 indices, the index type of the CUDA
        solver, which then shares them instead of keeping a converted copy.
    """
    mat = mat.tocsr()
    mat.sort_indices()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Sparse CSR tensor support is in beta state"
        )
        return torch.sparse_csr_tensor(
            torch.as_tensor(mat.indptr, dtype=torch.int32, device=device),
            torch.as_tensor(mat.indices, dtype=torch.int32, device=device),
            torch.as_tensor(mat.data, dtype=dtype, device=device),
            size=mat.shape,
        )


def _spmv(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Sparse mat-vec that works for both 1-D vectors and (n, k) blocks.

    Parameters
    ----------
    A : torch.Tensor
        Sparse CSR matrix of shape ``[n, n]``.
    x : torch.Tensor
        Vector of shape ``[n]`` or block of shape ``[n, k]``.

    Returns
    -------
    torch.Tensor
        ``A @ x``, with the same shape as ``x``.
    """
    if x.dim() == 1:
        return (A @ x.unsqueeze(1)).squeeze(1)
    return A @ x


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


@dataclass
class _Level:
    """One level of an AMG hierarchy.

    Parameters
    ----------
    A : torch.Tensor
        Sparse CSR operator of shape ``[n, n]``.
    D_inv : torch.Tensor
        Dense inverse diagonal of ``A``, of shape ``[n]``, used by the Jacobi
        smoother.
    R : torch.Tensor or None, optional
        Sparse CSR restriction of shape ``[n_coarse, n]``. None on the coarsest
        level. Default is None.
    P : torch.Tensor or None, optional
        Sparse CSR prolongation of shape ``[n, n_coarse]``. None on the coarsest
        level. Default is None.
    """

    A: torch.Tensor  # sparse_csr, n x n
    D_inv: torch.Tensor  # dense, n
    R: torch.Tensor | None = None  # sparse_csr, n_coarse x n (None on coarsest)
    P: torch.Tensor | None = None  # sparse_csr, n x n_coarse (None on coarsest)
    # int32/dtype-converted copies of A and D_inv for the CUDA solver, per dtype
    # (see AMGHierarchy.native_levels); kept on the level so hierarchies that share
    # a level also share its conversion
    native: dict[Any, list[torch.Tensor]] = field(default_factory=dict)
    # the same for R and P, shared by every hierarchy built on this interpolation
    native_rp: dict[Any, list[torch.Tensor]] = field(default_factory=dict)


@dataclass
class AMGHierarchy:
    """A multigrid hierarchy, applied as a single V-cycle.

    Callable as ``z = hierarchy(r)``, i.e. it is a preconditioner in the form the
    Krylov solvers expect.
    """

    levels: list[_Level]
    coarse_pinv: torch.Tensor  # dense pseudo-inverse of the coarsest operator
    num_pre_smooth: int = 1
    num_post_smooth: int = 1
    project_constant: bool = False
    method: str = "ruge_stuben"
    setup_seconds: float = 0.0
    # dtype of the preconditioner in the native (CUDA) solver; None: the solve dtype.
    # float32 halves the memory traffic of the V-cycle for a float64 solve.
    native_dtype: torch.dtype | None = None
    _native_cache: dict[Any, Any] | None = None

    @property
    def device(self) -> torch.device:
        """Device the hierarchy's tensors live on.

        Returns
        -------
        torch.device
            Device of the finest level's tensors.
        """
        return self.levels[0].D_inv.device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the hierarchy's tensors.

        Returns
        -------
        torch.dtype
            Data type of the finest level's tensors.
        """
        return self.levels[0].D_inv.dtype

    @property
    def operator_complexity(self) -> float:
        """Total nnz across levels divided by the finest-level nnz.

        Returns
        -------
        float
            The operator complexity, or NaN if the finest level is empty.
        """
        nnz = [lvl.A._nnz() for lvl in self.levels]
        return sum(nnz) / nnz[0] if nnz[0] else math.nan

    def describe(self) -> str:
        """Return a human-readable summary of the level sizes and complexity.

        Returns
        -------
        str
            A multi-line report with one line per level.
        """
        rows = [
            f"  level {i}: n={lvl.A.shape[0]:>10d}  nnz={lvl.A._nnz():>11d}"
            for i, lvl in enumerate(self.levels)
        ]
        return (
            f"AMG hierarchy ({self.method}, {len(self.levels)} levels, "
            f"operator complexity {self.operator_complexity:.2f}, "
            f"setup {self.setup_seconds:.1f}s)\n" + "\n".join(rows)
        )

    # -----------------------------------------------------------------------
    # Native (CUDA) representation
    # -----------------------------------------------------------------------

    def native_levels(
        self, dtype: torch.dtype
    ) -> tuple[list[list[torch.Tensor]], torch.Tensor]:
        """Return the hierarchy in the layout of ``_C.AMGPCGSolve``.

        Every level but the coarsest becomes ``[A row, A col, A val, D_inv, R row,
        R col, R val, P row, P col, P val]`` with int32 indices and values in
        ``dtype``; the coarsest level is represented by its dense pseudo-inverse.
        The conversion is cached per dtype.

        Parameters
        ----------
        dtype : torch.dtype
            Value dtype of the preconditioner.

        Returns
        -------
        levels : list of list of torch.Tensor
            Per-level tensors.
        coarse_pinv : torch.Tensor
            Dense pseudo-inverse of the coarsest operator, in ``dtype``.
        """
        if self._native_cache is None:
            self._native_cache = {}
        cached = self._native_cache.get(dtype)
        if cached is not None:
            return cached

        def csr(m: torch.Tensor) -> list[torch.Tensor]:
            return [
                m.crow_indices().to(torch.int32).contiguous(),
                m.col_indices().to(torch.int32).contiguous(),
                m.values().to(dtype).contiguous(),
            ]

        levels = []
        for lvl in self.levels:
            if lvl.R is None or lvl.P is None:
                break
            if (
                getattr(lvl, "native", None) is None
            ):  # levels unpickled from older versions
                lvl.native = {}
            if getattr(lvl, "native_rp", None) is None:
                lvl.native_rp = {}
            t = lvl.native.get(dtype)
            if t is None:
                t = [*csr(lvl.A), lvl.D_inv.to(dtype).contiguous()]
                lvl.native[dtype] = t
            rp = lvl.native_rp.get(dtype)
            if rp is None:
                rp = [*csr(lvl.R), *csr(lvl.P)]
                lvl.native_rp[dtype] = rp
            levels.append(t + rp)
        result = (levels, self.coarse_pinv.to(dtype).contiguous())
        self._native_cache[dtype] = result
        return result

    # -----------------------------------------------------------------------
    # Application
    # -----------------------------------------------------------------------

    def _smooth(
        self, level: _Level, x: torch.Tensor, b: torch.Tensor, sweeps: int
    ) -> torch.Tensor:
        """Apply damped Jacobi sweeps on one level.

        Parameters
        ----------
        level : _Level
            Level holding the operator and its inverse diagonal.
        x : torch.Tensor
            Initial iterate.
        b : torch.Tensor
            Right-hand side on this level.
        sweeps : int
            Number of Jacobi sweeps.

        Returns
        -------
        torch.Tensor
            The smoothed iterate.
        """
        for _ in range(sweeps):
            x = x + _JACOBI_OMEGA * level.D_inv * (b - _spmv(level.A, x))
        return x

    def _v_cycle(self, b: torch.Tensor, level_idx: int) -> torch.Tensor:
        """Recursively apply one V-cycle starting at ``level_idx``.

        Parameters
        ----------
        b : torch.Tensor
            Right-hand side on level ``level_idx``.
        level_idx : int
            Index into :attr:`levels` to start from.

        Returns
        -------
        torch.Tensor
            Approximate solution on level ``level_idx``.
        """
        level = self.levels[level_idx]
        if level.R is None:  # coarsest
            return self.coarse_pinv @ b

        x = self._smooth(level, torch.zeros_like(b), b, self.num_pre_smooth)
        residual = b - _spmv(level.A, x)
        coarse_correction = self._v_cycle(_spmv(level.R, residual), level_idx + 1)
        x = x + _spmv(level.P, coarse_correction)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        return self._smooth(level, x, b, self.num_post_smooth)

    def __call__(self, r: torch.Tensor) -> torch.Tensor:
        """Apply one V-cycle to the residual ``r``.

        Parameters
        ----------
        r : torch.Tensor
            Residual on the finest level.

        Returns
        -------
        torch.Tensor
            The preconditioned residual, same shape as ``r``.
        """
        # Projecting only on entry and exit
        if self.project_constant:
            r = r - r.mean()
        z = self._v_cycle(r, 0)
        if self.project_constant:
            z = z - z.mean()
        return z


def build_amg_hierarchy(
    csr_mat: Any,
    *,
    method: str = "ruge_stuben",
    max_levels: int = 20,
    max_coarse: int = 64,
    strength: float = 0.25,
    num_pre_smooth: int = 1,
    num_post_smooth: int = 1,
    project_constant: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    **pyamg_kwargs: Any,
) -> AMGHierarchy:
    """Build a V-cycle hierarchy for ``csr_mat`` using pyamg on the host.

    Parameters
    ----------
    method: str
        ``"ruge_stuben"`` (default) or ``"smoothed_aggregation"``. Classical
        Ruge-Stuben is the default on the strength of measurements on the small
        control duct (6.7M unknowns, epot at rtol 3e-3), where it beat smoothed
        aggregation on every axis at identical Nu.

    csr_mat: phipict._C.CSRmatrix
        Device-side system matrix the hierarchy is built for.

    max_levels: int
        Maximum number of levels in the hierarchy. Defaults to 20.

    max_coarse: int
        Size below which coarsening stops. The coarsest operator is inverted
        densely, so keep it small.

    strength: float
        Strength-of-connection threshold ``theta`` handed to pyamg. Defaults
        to 0.25.

    num_pre_smooth: int
        Jacobi sweeps before coarse-grid correction. Defaults to 1.

    num_post_smooth: int
        Jacobi sweeps after coarse-grid correction. Defaults to 1.

    project_constant: bool
        Set for a singular (pure-Neumann) system, e.g. the insulating duct's
        potential equation. See the module docstring.

    device: torch.device or None
        Device to place the hierarchy on. Defaults to the device of ``csr_mat``.

    dtype: torch.dtype or None
        Data type of the hierarchy's tensors. Defaults to the dtype of
        ``csr_mat``.

    **pyamg_kwargs: Any
        Extra keyword arguments forwarded to the pyamg solver constructor.

    Returns
    -------
    AMGHierarchy
        The built hierarchy, callable as a preconditioner.

    Raises
    ------
    ValueError
        If ``method`` names no known coarsening scheme.
    ImportError
        If pyamg is not installed.

    Notes
    -----
    The setup runs on the CPU and needs the matrix on the host, so this is a
    genuinely expensive one-off: budget for it, and call it once per matrix.
    """
    import time

    pyamg = _load_pyamg()

    device = device or csr_mat.value.device
    dtype = dtype or csr_mat.value.dtype

    start = time.perf_counter()
    A_host = csr_to_scipy(csr_mat)

    if method == "smoothed_aggregation":
        ml = pyamg.smoothed_aggregation_solver(
            A_host,
            max_levels=max_levels,
            max_coarse=max_coarse,
            strength=("symmetric", {"theta": strength}),  # pyright: ignore[reportArgumentType]
            **pyamg_kwargs,
        )
    elif method == "ruge_stuben":
        ml = pyamg.ruge_stuben_solver(
            A_host,
            max_levels=max_levels,
            max_coarse=max_coarse,
            strength=("classical", {"theta": strength}),
            **pyamg_kwargs,
        )
    else:
        raise ValueError(
            f"Unknown AMG method {method!r}; expected 'smoothed_aggregation' or "
            "'ruge_stuben'."
        )

    levels: list[_Level] = []
    for i, lvl in enumerate(ml.levels):
        A_dev = _scipy_to_torch_csr(lvl.A, device, dtype)  # pyright: ignore[reportArgumentType]

        diag = torch.as_tensor(lvl.A.diagonal().copy(), dtype=dtype, device=device)
        # A zero diagonal would make the smoother produce inf/nan; leaving those
        # rows unsmoothed is the safe behaviour
        d_inv = torch.where(
            diag.abs() > _MIN_ABS_DIAG, 1.0 / diag, torch.ones_like(diag)
        )

        is_coarsest = i == len(ml.levels) - 1
        levels.append(
            _Level(
                A=A_dev,
                D_inv=d_inv,
                R=None if is_coarsest else _scipy_to_torch_csr(lvl.R, device, dtype),  # pyright: ignore[reportArgumentType]
                P=None if is_coarsest else _scipy_to_torch_csr(lvl.P, device, dtype),  # pyright: ignore[reportArgumentType]
            )
        )

    # Pseudo-inverse rather than an LU: the coarse operator of a singular system
    # is itself singular, and this is small enough (<= max_coarse) that a dense
    # pinv is free. It also keeps the module clear of any LAPACK/cuSOLVER path
    coarse_dense = np.asarray(ml.levels[-1].A.todense(), dtype=np.float64)
    coarse_pinv = torch.as_tensor(
        np.linalg.pinv(coarse_dense), dtype=dtype, device=device
    )

    hierarchy = AMGHierarchy(
        levels=levels,
        coarse_pinv=coarse_pinv,
        num_pre_smooth=num_pre_smooth,
        num_post_smooth=num_post_smooth,
        project_constant=project_constant,
        method=method,
        setup_seconds=time.perf_counter() - start,
    )

    _LOG.debug("%s", hierarchy.describe())

    return hierarchy


def _inverse_diagonal(A: torch.Tensor) -> torch.Tensor:
    """Dense inverse diagonal of a sparse CSR matrix, 1 where the diagonal vanishes.

    Parameters
    ----------
    A : torch.Tensor
        Square sparse CSR matrix.

    Returns
    -------
    torch.Tensor
        Inverse diagonal of shape ``[n]``.
    """
    n = A.shape[0]
    crow = A.crow_indices()
    rows = torch.repeat_interleave(
        torch.arange(n, device=crow.device), crow[1:] - crow[:-1]
    )
    cols = A.col_indices()
    on_diag = rows == cols
    diag = torch.zeros(n, dtype=A.dtype, device=A.device)
    diag.index_put_((rows[on_diag],), A.values()[on_diag], accumulate=True)
    return torch.where(diag.abs() > _MIN_ABS_DIAG, 1.0 / diag, torch.ones_like(diag))


def _csr_row_slice(M: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    """Rows ``start:stop`` of a sparse CSR matrix, sharing its storage."""
    crow = M.crow_indices()
    lo, hi = int(crow[start]), int(crow[stop])
    return torch.sparse_csr_tensor(
        crow[start : stop + 1] - crow[start],
        M.col_indices()[lo:hi],
        M.values()[lo:hi],
        size=(stop - start, M.shape[1]),
    )


def _csr_vstack(parts: list[torch.Tensor]) -> torch.Tensor:
    """Stack sparse CSR matrices with the same number of columns vertically."""
    if len(parts) == 1:
        return parts[0]
    crows, offset = [parts[0].crow_indices()[:1]], 0
    for m in parts:
        crows.append(m.crow_indices()[1:] + offset)
        offset += m._nnz()
    idx = torch.int64 if offset > torch.iinfo(torch.int32).max else torch.int32
    return torch.sparse_csr_tensor(
        torch.cat([c.to(idx) for c in crows]),
        torch.cat([m.col_indices().to(idx) for m in parts]),
        torch.cat([m.values() for m in parts]),
        size=(sum(m.shape[0] for m in parts), parts[0].shape[1]),
    )


# Upper bound on the intermediate products (sum over the entries X[i, k] of the
# non-zeros of row k of Y) per SpGEMM call of a Galerkin product. The SpGEMM
# workspace grows with them; splitting the rows of X bounds it (the finest level
# of a 20M-cell duct otherwise takes >10 GiB). ~12 bytes per product
GALERKIN_CHUNK_PRODUCTS = 20_000_000


def _spgemm_chunked(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """``X @ Y`` of sparse CSR matrices, in row blocks of ``X`` of bounded work."""
    x_crow = X.crow_indices().to(torch.int64)
    y_crow = Y.crow_indices().to(torch.int64)
    y_row_nnz = y_crow[1:] - y_crow[:-1]
    # products up to the end of every row of X
    per_entry = y_row_nnz[X.col_indices().to(torch.int64)]
    cum = torch.cat([per_entry.new_zeros(1), torch.cumsum(per_entry, 0)])
    del per_entry
    row_end = cum[x_crow[1:]]
    total = int(row_end[-1]) if row_end.numel() else 0
    if total <= GALERKIN_CHUNK_PRODUCTS:
        return (X @ Y).to_sparse_csr()
    chunks = -(-total // GALERKIN_CHUNK_PRODUCTS)
    targets = torch.arange(1, chunks, device=row_end.device) * (total / chunks)
    inner = torch.searchsorted(row_end.to(torch.float64), targets).tolist()
    n = X.shape[0]
    bounds = sorted({0, n, *(min(max(int(b), 1), n - 1) for b in inner)})
    return _csr_vstack(
        [
            (_csr_row_slice(X, a, b) @ Y).to_sparse_csr()
            for a, b in zip(bounds[:-1], bounds[1:], strict=True)
        ]
    )


def _galerkin_product(
    R: torch.Tensor, A: torch.Tensor, P: torch.Tensor
) -> torch.Tensor:
    """Coarse operator ``R A P`` (sparse CSR), with bounded SpGEMM workspaces."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Sparse CSR tensor support is in beta state"
        )
        return _spgemm_chunked(R, _spgemm_chunked(A, P))


def _fine_operator(csr_mat: Any) -> torch.Tensor:
    """``csr_mat`` as a sparse CSR tensor sharing its (int32) indices and values."""
    row, col, value = _csr_parts(csr_mat)
    n = row.numel() - 1
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Sparse CSR tensor support is in beta state"
        )
        return torch.sparse_csr_tensor(
            row.to(torch.int32), col.to(torch.int32), value.detach(), size=(n, n)
        )


def _interpolation_only(hierarchy: AMGHierarchy) -> AMGHierarchy:
    """The interpolation (P, R) of ``hierarchy`` without its operators.

    What the interpolation cache keeps: enough for :func:`galerkin_refresh`, but
    none of the operators, which belong to (and are freed with) the hierarchies
    built on it. The levels hold shape-only placeholders for ``A`` and ``D_inv``.
    """
    levels = []
    for lvl in hierarchy.levels:
        n = lvl.A.shape[0]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Sparse CSR tensor support is in beta state"
            )
            A = torch.sparse_csr_tensor(
                torch.zeros(n + 1, dtype=torch.int32, device=lvl.D_inv.device),
                torch.zeros(0, dtype=torch.int32, device=lvl.D_inv.device),
                torch.zeros(0, dtype=lvl.D_inv.dtype, device=lvl.D_inv.device),
                size=(n, n),
            )
        if getattr(lvl, "native_rp", None) is None:
            lvl.native_rp = {}
        levels.append(
            _Level(A=A, D_inv=lvl.D_inv[:0], R=lvl.R, P=lvl.P, native_rp=lvl.native_rp)
        )
    return AMGHierarchy(
        levels=levels,
        coarse_pinv=hierarchy.coarse_pinv[:0, :0],
        num_pre_smooth=hierarchy.num_pre_smooth,
        num_post_smooth=hierarchy.num_post_smooth,
        project_constant=hierarchy.project_constant,
        method=hierarchy.method,
        setup_seconds=hierarchy.setup_seconds,
        native_dtype=hierarchy.native_dtype,
    )


def galerkin_refresh(
    hierarchy: AMGHierarchy,
    csr_mat: Any,
    *,
    project_constant: bool | None = None,
    coarse: bool = True,
) -> AMGHierarchy:
    """Re-use the interpolation of ``hierarchy`` for a new fine-level operator.

    The prolongation and restriction operators are kept ("frozen"), and the
    coarse operators are recomputed as Galerkin products ``A_{l+1} = R_l A_l P_l``
    on the GPU. This is much cheaper than a new (host-side) AMG setup and stays
    an effective preconditioner as long as the new operator has the same
    sparsity pattern and a similar strength-of-connection structure, e.g. the
    pressure matrix of consecutive time steps, or the pressure matrix versus the
    electric-potential matrix of the same grid.

    Parameters
    ----------
    hierarchy : AMGHierarchy
        Hierarchy providing P and R.
    csr_mat : phipict._C.CSRmatrix or torch.Tensor
        New fine-level operator: a ``_C.CSRmatrix`` or a sparse CSR tensor with
        the same size as the finest level of ``hierarchy``.
    project_constant : bool or None, optional
        Overrides ``hierarchy.project_constant``. Default is None (keep).
    coarse : bool, optional
        Recompute the coarse operators (default). With False only the finest
        level is replaced and the coarse levels of ``hierarchy`` are re-used as
        they are: still a valid (slightly weaker) preconditioner when the
        operator changed little, at the cost of a diagonal extraction instead of
        the Galerkin products.

    Returns
    -------
    AMGHierarchy
        New hierarchy sharing P and R with ``hierarchy``.
    """
    import time

    start = time.perf_counter()
    fine = hierarchy.levels[0]
    A = csr_mat if isinstance(csr_mat, torch.Tensor) else _fine_operator(csr_mat)
    if A.shape != fine.A.shape:
        raise ValueError(
            f"Operator of shape {tuple(A.shape)} does not match the finest AMG level "
            f"{tuple(fine.A.shape)}."
        )
    A = A.to(dtype=fine.D_inv.dtype, device=fine.D_inv.device)

    levels: list[_Level] = []
    for i, lvl in enumerate(hierarchy.levels):
        if i > 0 and not coarse:
            levels.extend(hierarchy.levels[i:])
            break
        if getattr(lvl, "native_rp", None) is None:
            lvl.native_rp = {}
        levels.append(
            _Level(
                A=A,
                D_inv=_inverse_diagonal(A),
                R=lvl.R,
                P=lvl.P,
                native_rp=lvl.native_rp,
            )
        )
        if lvl.R is None or lvl.P is None:
            break
        A = _galerkin_product(lvl.R, A, lvl.P)

    if coarse or len(levels) == 1:
        coarse_pinv = torch.linalg.pinv(levels[-1].A.to_dense().to(torch.float64)).to(
            fine.D_inv.dtype
        )
    else:
        coarse_pinv = hierarchy.coarse_pinv
    return AMGHierarchy(
        levels=levels,
        coarse_pinv=coarse_pinv,
        num_pre_smooth=hierarchy.num_pre_smooth,
        num_post_smooth=hierarchy.num_post_smooth,
        project_constant=(
            hierarchy.project_constant if project_constant is None else project_constant
        ),
        method=hierarchy.method,
        setup_seconds=time.perf_counter() - start,
        native_dtype=hierarchy.native_dtype,
    )


# ---------------------------------------------------------------------------
# Shared interpolation (setup reuse across operators and simulations)
# ---------------------------------------------------------------------------

# Hierarchies whose P/R are re-used for every operator with the same sparsity
# pattern, keyed by `_sparsity_key`. The pressure and electric-potential matrices
# of a grid share their pattern, and so do the matrices of every episode of an
# environment, so one host-side pyamg setup serves all of them.
_INTERPOLATION_CACHE: dict[tuple, AMGHierarchy] = {}


def _csr_parts(csr_mat: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Row offsets, column indices and values of a ``_C.CSRmatrix`` or CSR tensor."""
    if isinstance(csr_mat, torch.Tensor):
        return csr_mat.crow_indices(), csr_mat.col_indices(), csr_mat.values()
    return csr_mat.row, csr_mat.index, csr_mat.value


def _sparsity_key(csr_mat: Any) -> tuple:
    """Cheap identifier of a sparsity pattern (size, nnz, weighted index sums)."""
    row, col, value = _csr_parts(csr_mat)
    n = row.numel() - 1
    col64 = col.to(torch.int64)
    weights = (
        torch.arange(col64.numel(), device=col64.device, dtype=torch.int64) % 7919 + 1
    )
    checks = torch.stack(
        [col64.sum(), (col64 * weights).sum(), row.to(torch.int64).sum()]
    )
    return (str(value.device), n, col.numel(), *checks.tolist())


def values_fingerprint(value: torch.Tensor) -> tuple[float, float]:
    """Fingerprint of matrix values, used to detect a changed operator.

    Parameters
    ----------
    value : torch.Tensor
        Matrix values.

    Returns
    -------
    tuple of float
        Sum and sum of squares of the values, in float64.
    """
    v = value.detach().reshape(-1).to(torch.float64)
    return tuple(torch.stack([v.sum(), (v * v).sum()]).tolist())  # type: ignore[return-value]


def _cache_file(
    key: tuple, project_constant: bool, build_kwargs: dict[str, Any]
) -> Any:
    """Path of the on-disk interpolation cache entry, or None if disabled."""
    import hashlib
    import os
    from pathlib import Path

    cache_dir = os.environ.get("PHIPICT_AMG_CACHE_DIR")
    if not cache_dir:
        return None
    ident = repr((key[1:], project_constant, sorted(build_kwargs.items())))
    digest = hashlib.sha1(ident.encode()).hexdigest()[:20]
    return Path(cache_dir) / f"amg_interp_{digest}.pt"


def _hierarchy_from_prolongators(
    csr_mat: Any, prolongators: list[torch.Tensor], project_constant: bool, method: str
) -> AMGHierarchy:
    """Rebuild a hierarchy from stored prolongators by Galerkin products."""
    import time

    start = time.perf_counter()
    _, _, value = _csr_parts(csr_mat)
    device, dtype = value.device, value.dtype
    levels: list[_Level] = []
    A = _fine_operator(csr_mat)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Sparse CSR tensor support is in beta state"
        )
        for P_host in prolongators:
            P = _int32_csr(P_host.to(device=device, dtype=dtype).to_sparse_csr())
            R = _int32_csr(P.t().to_sparse_csr())
            levels.append(_Level(A=A, D_inv=_inverse_diagonal(A), R=R, P=P))
            A = _galerkin_product(R, A, P)
    levels.append(_Level(A=A, D_inv=_inverse_diagonal(A)))
    coarse_pinv = torch.linalg.pinv(A.to_dense().to(torch.float64)).to(dtype)
    return AMGHierarchy(
        levels=levels,
        coarse_pinv=coarse_pinv,
        project_constant=project_constant,
        method=method,
        setup_seconds=time.perf_counter() - start,
    )


def _int32_csr(M: torch.Tensor) -> torch.Tensor:
    """``M`` (sparse CSR) with int32 indices."""
    if M.crow_indices().dtype == torch.int32:
        return M
    return torch.sparse_csr_tensor(
        M.crow_indices().to(torch.int32),
        M.col_indices().to(torch.int32),
        M.values(),
        size=M.shape,
    )


def clear_interpolation_cache() -> None:
    """Drop all cached AMG interpolation hierarchies (frees their GPU memory)."""
    _INTERPOLATION_CACHE.clear()


def hierarchy_for(
    csr_mat: Any,
    *,
    project_constant: bool = False,
    reuse_interpolation: bool = True,
    native_dtype: torch.dtype | None = torch.float32,
    **build_kwargs: Any,
) -> AMGHierarchy:
    """AMG hierarchy for ``csr_mat``, re-using a cached interpolation if possible.

    With ``reuse_interpolation``, the first operator of a sparsity pattern gets a
    full (host-side, pyamg) setup, which is cached; if the environment variable
    ``PHIPICT_AMG_CACHE_DIR`` is set, its prolongators are also stored there and
    later processes rebuild the hierarchy from them on the GPU in about a
    second instead of repeating the setup. Every further operator with
    the same pattern gets the cached P/R and a GPU Galerkin refresh of the coarse
    operators (see :func:`galerkin_refresh`), which takes milliseconds instead of
    seconds.

    Parameters
    ----------
    csr_mat : phipict._C.CSRmatrix
        Operator the hierarchy is for.
    project_constant : bool, optional
        Whether the operator is singular with the constant nullspace.
    reuse_interpolation : bool, optional
        Use and fill the interpolation cache. Default is True.
    native_dtype : torch.dtype or None, optional
        Preconditioner dtype of the CUDA solver, see :class:`AMGHierarchy`.
        Default is float32: the V-cycle is ~25% faster than in float64 and the
        Krylov iteration, which runs in the dtype of the system, is unaffected
        (identical iteration counts on the MHD duct). None uses the dtype of the
        solve.
    **build_kwargs : Any
        Forwarded to :func:`build_amg_hierarchy` for a new setup.

    Returns
    -------
    AMGHierarchy
        Hierarchy whose finest level is ``csr_mat``.
    """
    if not reuse_interpolation:
        h = build_amg_hierarchy(
            csr_mat, project_constant=project_constant, **build_kwargs
        )
        h.native_dtype = native_dtype
        return h
    key = _sparsity_key(csr_mat)
    base = _INTERPOLATION_CACHE.get(key)
    if base is None:
        cache_file = _cache_file(key, project_constant, build_kwargs)
        if cache_file is not None and cache_file.is_file():
            # pyamg setup of an earlier process: only the prolongators are stored,
            # the coarse operators are recomputed for this operator on the GPU
            prolongators = torch.load(cache_file, map_location="cpu", weights_only=True)
            base = _hierarchy_from_prolongators(
                csr_mat,
                prolongators,
                project_constant,
                str(build_kwargs.get("method", "ruge_stuben")),
            )
            _LOG.info(
                "AMG interpolation loaded from %s (%.1f s):\n%s",
                cache_file,
                base.setup_seconds,
                base.describe(),
            )
        else:
            base = build_amg_hierarchy(
                csr_mat, project_constant=project_constant, **build_kwargs
            )
            _LOG.info("AMG setup for a new sparsity pattern:\n%s", base.describe())
            if cache_file is not None:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_file.with_suffix(".tmp")
                torch.save(
                    [
                        lvl.P.to_sparse_coo().coalesce().cpu()
                        for lvl in base.levels
                        if lvl.P is not None
                    ],
                    tmp,
                )
                tmp.replace(cache_file)
        base.native_dtype = native_dtype
        # the cache keeps only P and R: the operators of this first hierarchy are
        # freed with it once its caller moves on to a refreshed one
        _INTERPOLATION_CACHE[key] = _interpolation_only(base)
        return base
    h = galerkin_refresh(base, csr_mat, project_constant=project_constant)
    h.native_dtype = native_dtype
    return h


# ---------------------------------------------------------------------------
# Batched environments
# ---------------------------------------------------------------------------


@dataclass
class _GalerkinMap:
    """Fixed linear map from the fine to the coarse operator values of one level.

    For fixed sparsity patterns of ``A``, ``R`` and ``P``, the values of the
    Galerkin operator ``R A P`` are a linear function of the values of ``A``:
    ``a_coarse = G @ a_fine`` with ``G[e, k] = sum R[c, i] P[j, d]`` over the
    entries ``k = (i, j)`` of ``A`` that contribute to the coarse entry
    ``e = (c, d)``. With ``G`` built once, the coarse operators of any number of
    operators with that pattern (batched environments) are one sparse-dense
    product.

    ``G`` has one entry per intermediate product of ``R A P``, which does not fit
    in memory for large operators. Beyond :data:`GALERKIN_MAP_PRODUCTS` it is
    None, and every environment's coarse operator is its own (chunked) Galerkin
    product instead, scattered into the coarse pattern.
    """

    G: torch.Tensor | None  # sparse CSR [nnz_coarse, nnz_fine]
    crow: torch.Tensor  # coarse pattern, int64
    col: torch.Tensor
    diag_pos: torch.Tensor  # position of the diagonal entry per coarse row


def _pattern_diag_pos(crow: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
    """Position of the diagonal entry of every row of a CSR pattern (-1 if absent)."""
    n = crow.numel() - 1
    rows = torch.repeat_interleave(
        torch.arange(n, device=crow.device), crow[1:] - crow[:-1]
    )
    pos = torch.full((n,), -1, dtype=torch.int64, device=crow.device)
    on_diag = rows == col
    pos[rows[on_diag]] = torch.nonzero(on_diag).reshape(-1)
    return pos


def _expand_rows(
    crow: torch.Tensor, rows: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """For every entry of ``rows``, enumerate the CSR positions of that row.

    Returns
    -------
    owner : torch.Tensor
        Index into ``rows`` of every expanded entry.
    pos : torch.Tensor
        CSR position of every expanded entry.
    """
    counts = (crow[1:] - crow[:-1])[rows]
    owner = torch.repeat_interleave(
        torch.arange(rows.numel(), device=rows.device), counts
    )
    starts = torch.cumsum(counts, 0) - counts
    within = torch.arange(owner.numel(), device=rows.device) - starts[owner]
    return owner, crow[rows][owner] + within


# Largest number of intermediate products of ``R A P`` for which a batched
# hierarchy precomputes the Galerkin map of a level. The map and its setup take
# ~50 bytes per product; beyond, the coarse operators are computed per env.
GALERKIN_MAP_PRODUCTS = 50_000_000


def _galerkin_map(
    crow: torch.Tensor,
    col: torch.Tensor,
    R: torch.Tensor,
    P: torch.Tensor,
    max_products: int | None = None,
) -> _GalerkinMap:
    """Build the :class:`_GalerkinMap` of ``R A P`` for the pattern ``(crow, col)``.

    With more than ``max_products`` (default :data:`GALERKIN_MAP_PRODUCTS`)
    intermediate products, only the coarse pattern is built (``G`` is None).
    """
    if max_products is None:
        max_products = GALERKIN_MAP_PRODUCTS
    device = crow.device
    n = crow.numel() - 1
    nnz = col.numel()
    rows = torch.repeat_interleave(torch.arange(n, device=device), crow[1:] - crow[:-1])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Sparse CSR tensor support is in beta state"
        )
        RT = R.t().to_sparse_csr()  # row i of R^T: the entries R[c, i]
    i64 = torch.int64
    rt_crow, rt_col, rt_val = (
        RT.crow_indices().to(i64),
        RT.col_indices().to(i64),
        RT.values(),
    )
    p_crow, p_col, p_val = P.crow_indices().to(i64), P.col_indices().to(i64), P.values()
    nc = P.shape[1]

    products = int(
        ((rt_crow[1:] - rt_crow[:-1])[rows] * (p_crow[1:] - p_crow[:-1])[col]).sum()
    )
    if products > max_products:
        del RT, rt_crow, rt_col, rt_val, rows
        return _galerkin_pattern(crow, col, R, P)

    # fine entry k=(i,j) -> entries (c, R[c,i]) of R^T row i ...
    k1, pos_r = _expand_rows(rt_crow, rows)
    c1, w1 = rt_col[pos_r], rt_val[pos_r]
    # ... -> entries (d, P[j,d]) of P row j
    k2, pos_p = _expand_rows(p_crow, col[k1])
    fine = k1[k2]
    coarse_key = c1[k2] * nc + p_col[pos_p]
    weight = w1[k2] * p_val[pos_p]

    keys, entry = torch.unique(coarse_key, sorted=True, return_inverse=True)
    c_rows = keys // nc
    c_col = keys % nc
    c_crow = torch.zeros(nc + 1, dtype=torch.int64, device=device)
    c_crow[1:] = torch.cumsum(torch.bincount(c_rows, minlength=nc), 0)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Sparse CSR tensor support is in beta state"
        )
        G = (
            torch.sparse_coo_tensor(
                torch.stack([entry, fine]), weight, (keys.numel(), nnz)
            )
            .coalesce()
            .to_sparse_csr()
        )
    return _GalerkinMap(
        G=G, crow=c_crow, col=c_col, diag_pos=_pattern_diag_pos(c_crow, c_col)
    )


def _galerkin_pattern(
    crow: torch.Tensor, col: torch.Tensor, R: torch.Tensor, P: torch.Tensor
) -> _GalerkinMap:
    """The coarse pattern of ``R A P`` for the pattern ``(crow, col)``, without map.

    The pattern is structural: that of the product of the patterns, whatever the
    values (SpGEMM keeps entries that cancel numerically).
    """
    ones = torch.ones(col.numel(), dtype=P.dtype, device=col.device)
    C = _galerkin_product(R, _csr(crow, col, ones), P)
    c_crow = C.crow_indices().to(torch.int64)
    c_col = C.col_indices().to(torch.int64)
    return _GalerkinMap(
        G=None, crow=c_crow, col=c_col, diag_pos=_pattern_diag_pos(c_crow, c_col)
    )


def _csr(crow: torch.Tensor, col: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Square sparse CSR matrix of the pattern ``(crow, col)`` with ``values``."""
    n = crow.numel() - 1
    idx = torch.int64 if col.numel() > torch.iinfo(torch.int32).max else torch.int32
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Sparse CSR tensor support is in beta state"
        )
        return torch.sparse_csr_tensor(crow.to(idx), col.to(idx), values, size=(n, n))


def _coarse_values_per_env(
    values: torch.Tensor,
    crow: torch.Tensor,
    col: torch.Tensor,
    R: torch.Tensor,
    P: torch.Tensor,
    coarse: _GalerkinMap,
) -> torch.Tensor:
    """Coarse operator values ``[B, nnz_c]`` of the B operators ``values [B, nnz]``.

    One Galerkin product per environment, scattered into the pattern of
    ``coarse`` (see :func:`_galerkin_pattern`).
    """
    nc = coarse.crow.numel() - 1
    c_rows = torch.repeat_interleave(
        torch.arange(nc, device=crow.device), coarse.crow[1:] - coarse.crow[:-1]
    )
    keys = c_rows * nc + coarse.col  # sorted: CSR rows, sorted columns
    out = torch.zeros(
        values.size(0), coarse.col.numel(), dtype=values.dtype, device=values.device
    )
    for b in range(values.size(0)):
        C = _galerkin_product(R, _csr(crow, col, values[b].to(P.dtype)), P)
        C_crow = C.crow_indices().to(torch.int64)
        rows_b = torch.repeat_interleave(
            torch.arange(nc, device=crow.device), C_crow[1:] - C_crow[:-1]
        )
        pos = torch.searchsorted(keys, rows_b * nc + C.col_indices().to(torch.int64))
        out[b, pos] = C.values().to(values.dtype)
    return out


@dataclass
class BatchedAMGHierarchy:
    """AMG hierarchies of B operators with one sparsity pattern (batched environments).

    All environments share the interpolation (``R``, ``P``) of ``base``; every
    environment has its own operators on every level, refreshed together by the
    fixed Galerkin maps (one sparse-dense product per level). Solved by the
    native AMG-PCG, one V-cycle per environment and iteration, until every
    environment has converged.
    """

    base: AMGHierarchy
    batch_size: int
    maps: list[_GalerkinMap]
    fine_crow: torch.Tensor  # int64 pattern of the finest level
    fine_col: torch.Tensor
    fine_diag_pos: torch.Tensor
    values: list[torch.Tensor]  # per level [B, nnz_l], hierarchy dtype of the refresh
    dinv: list[torch.Tensor]  # per level [B, n_l]
    coarse_pinv: torch.Tensor  # [B, nc, nc]
    project_constant: bool = False
    native_dtype: torch.dtype | None = None
    num_pre_smooth: int = 1
    num_post_smooth: int = 1
    setup_seconds: float = 0.0
    # persistent native buffers (stable pointers keep the CUDA graphs valid)
    _native: dict[Any, Any] = field(default_factory=dict)

    @staticmethod
    def build(
        base: AMGHierarchy, crow: torch.Tensor, col: torch.Tensor, batch_size: int
    ) -> BatchedAMGHierarchy:
        """Build the Galerkin maps of the pattern for the interpolation of ``base``."""
        import time

        start = time.perf_counter()
        crow, col = crow.to(torch.int64), col.to(torch.int64)
        maps: list[_GalerkinMap] = []
        c_crow, c_col = crow, col
        for lvl in base.levels:
            if lvl.R is None or lvl.P is None:
                break
            m = _galerkin_map(c_crow, c_col, lvl.R, lvl.P)
            maps.append(m)
            c_crow, c_col = m.crow, m.col
        dtype = base.levels[0].D_inv.dtype
        device = crow.device
        sizes = [crow.numel() - 1] + [m.crow.numel() - 1 for m in maps]
        nnzs = [col.numel()] + [m.col.numel() for m in maps]
        h = BatchedAMGHierarchy(
            base=base,
            batch_size=batch_size,
            maps=maps,
            fine_crow=crow,
            fine_col=col,
            fine_diag_pos=_pattern_diag_pos(crow, col),
            values=[
                torch.zeros(batch_size, nz, dtype=dtype, device=device) for nz in nnzs
            ],
            dinv=[torch.ones(batch_size, n, dtype=dtype, device=device) for n in sizes],
            coarse_pinv=torch.zeros(
                batch_size, sizes[-1], sizes[-1], dtype=dtype, device=device
            ),
            project_constant=base.project_constant,
            native_dtype=base.native_dtype,
            num_pre_smooth=base.num_pre_smooth,
            num_post_smooth=base.num_post_smooth,
        )
        h.setup_seconds = time.perf_counter() - start
        return h

    @staticmethod
    def _inverse_diag(values: torch.Tensor, diag_pos: torch.Tensor) -> torch.Tensor:
        diag = torch.where(
            diag_pos >= 0,
            values[:, diag_pos.clamp(min=0)],
            torch.zeros_like(values[:, :1]),
        )
        return torch.where(
            diag.abs() > _MIN_ABS_DIAG, 1.0 / diag, torch.ones_like(diag)
        )

    def refresh(self, fine_values: torch.Tensor, coarse: bool = True) -> None:
        """Set the finest operators (``[B * nnz]``) and recompute the hierarchy.

        Parameters
        ----------
        fine_values : torch.Tensor
            Values of the B finest operators, consecutive.
        coarse : bool, optional
            Recompute the coarse operators (default); False only updates the
            finest level, see :func:`galerkin_refresh`.
        """
        B = self.batch_size
        with torch.no_grad():
            self.values[0].copy_(fine_values.detach().reshape(B, -1))
            self.dinv[0].copy_(self._inverse_diag(self.values[0], self.fine_diag_pos))
            if not coarse:
                return
            patterns = [(self.fine_crow, self.fine_col)] + [
                (m.crow, m.col) for m in self.maps
            ]
            for level, m in enumerate(self.maps):
                if m.G is not None:
                    # [nnz_c, nnz_f] @ [nnz_f, B] -> values of all environments
                    self.values[level + 1].copy_((m.G @ self.values[level].T).T)
                else:
                    lvl = self.base.levels[level]
                    assert lvl.R is not None and lvl.P is not None
                    self.values[level + 1].copy_(
                        _coarse_values_per_env(
                            self.values[level], *patterns[level], lvl.R, lvl.P, m
                        )
                    )
                self.dinv[level + 1].copy_(
                    self._inverse_diag(self.values[level + 1], m.diag_pos)
                )
            last = self.maps[-1]
            nc = last.crow.numel() - 1
            rows = torch.repeat_interleave(
                torch.arange(nc, device=last.crow.device),
                last.crow[1:] - last.crow[:-1],
            )
            dense = torch.zeros(B, nc, nc, dtype=torch.float64, device=last.crow.device)
            dense[:, rows, last.col] = self.values[-1].to(torch.float64)
            self.coarse_pinv.copy_(torch.linalg.pinv(dense))

    def native_arguments(
        self, dtype: torch.dtype, pdtype: torch.dtype
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]], torch.Tensor]:
        """Arguments of ``_C.AMGPCGSolve`` in persistent buffers, updated in place."""
        key = (dtype, pdtype)
        buf = self._native.get(key)
        if buf is None:
            i32 = torch.int32
            fine = [
                self.fine_crow.to(i32),
                self.fine_col.to(i32),
                torch.empty(
                    self.values[0].numel(), dtype=dtype, device=self.fine_crow.device
                ),
            ]
            levels = []
            patterns = [(self.fine_crow, self.fine_col)] + [
                (m.crow, m.col) for m in self.maps
            ]
            for level, lvl in enumerate(self.base.levels[: len(self.maps)]):
                crow, col = patterns[level]
                assert lvl.R is not None and lvl.P is not None
                levels.append(
                    [
                        crow.to(i32),
                        col.to(i32),
                        torch.empty(
                            self.values[level].numel(), dtype=pdtype, device=crow.device
                        ),
                        torch.empty(
                            self.dinv[level].numel(), dtype=pdtype, device=crow.device
                        ),
                        lvl.R.crow_indices().to(i32),
                        lvl.R.col_indices().to(i32),
                        lvl.R.values().to(pdtype),
                        lvl.P.crow_indices().to(i32),
                        lvl.P.col_indices().to(i32),
                        lvl.P.values().to(pdtype),
                    ]
                )
            pinv = torch.empty_like(self.coarse_pinv, dtype=pdtype)
            buf = (fine, levels, pinv)
            self._native[key] = buf
        fine, levels, pinv = buf
        with torch.no_grad():
            fine[2].copy_(self.values[0].reshape(-1))
            for level, t in enumerate(levels):
                t[2].copy_(self.values[level].reshape(-1))
                t[3].copy_(self.dinv[level].reshape(-1))
            pinv.copy_(self.coarse_pinv)
        return fine, levels, pinv


def batched_hierarchy_for(
    csr_mat: Any,
    batch_size: int,
    *,
    project_constant: bool = False,
    previous: BatchedAMGHierarchy | None = None,
    coarse: bool = True,
    **kwargs: Any,
) -> BatchedAMGHierarchy:
    """Batched AMG hierarchy for the B operators of a batched ``_C.CSRmatrix``.

    The interpolation is the shared, cached one of the sparsity pattern (see
    :func:`hierarchy_for`), set up from the first environment's operator if the
    pattern is new. ``previous`` (same pattern and batch size) is refreshed in
    place instead of building new Galerkin maps.

    Parameters
    ----------
    csr_mat : phipict._C.CSRmatrix
        Matrix with ``batch_size`` value arrays and one pattern.
    batch_size : int
        Number of environments.
    project_constant : bool, optional
        Singular operators with the constant nullspace.
    previous : BatchedAMGHierarchy or None, optional
        Hierarchy to refresh.
    coarse : bool, optional
        Recompute the coarse operators, see :meth:`BatchedAMGHierarchy.refresh`.
    **kwargs : Any
        Forwarded to :func:`hierarchy_for` for the interpolation.

    Returns
    -------
    BatchedAMGHierarchy
        Refreshed hierarchy.
    """
    from phipict import _C

    nnz = csr_mat.getNnz()
    if (
        previous is None
        or previous.batch_size != batch_size
        or previous.fine_col.numel() != nnz
    ):
        first = _C.CSRmatrix(
            csr_mat.value[:nnz].contiguous(), csr_mat.index, csr_mat.row
        )
        base = hierarchy_for(first, project_constant=project_constant, **kwargs)
        previous = BatchedAMGHierarchy.build(
            base, csr_mat.row, csr_mat.index, batch_size
        )
        coarse = True
    previous.project_constant = project_constant
    previous.refresh(csr_mat.value, coarse=coarse)
    return previous


# ---------------------------------------------------------------------------
# Preconditioned CG
# ---------------------------------------------------------------------------


def native_amg_available() -> bool:
    """Return whether the compiled extension provides the CUDA AMG-PCG solver.

    Returns
    -------
    bool
        True if ``phipict._C.AMGPCGSolve`` exists.
    """
    try:
        from phipict import _C
    except ImportError:  # pragma: no cover - extension not built
        return False
    return hasattr(_C, "AMGPCGSolve")


# Set to False to force the pure-torch implementation (reference / ablation)
USE_NATIVE = True


@dataclass
class AMGSolveInfo:
    """Duck-type of ``_C.LinearSolverResultInfo``.

    The telemetry in ``phipict.solvers.stats`` and the error handling
    in ``piso_diff._check_solver_return_infos`` both read these four fields
    off whatever the solver returned, so the AMG path reports exactly like the
    CUDA ones and needs no special-casing downstream.
    """

    finalResidual: float
    usedIterations: int
    converged: bool
    isFiniteResidual: bool

    def __str__(self) -> str:
        return (
            f"AMGSolveInfo(finalResidual={self.finalResidual:.5g}, "
            f"usedIterations={self.usedIterations:d}, "
            f"converged={int(self.converged):d}, "
            f"isFiniteResidual={int(self.isFiniteResidual):d})"
        )


def _criterion_residual(r: torch.Tensor) -> torch.Tensor:
    """``||r||_2 / sqrt(n)`` -- the NORM2_NORMALIZED criterion the kernels use.

    Parameters
    ----------
    r : torch.Tensor
        Residual vector.

    Returns
    -------
    torch.Tensor
        Scalar tensor holding the normalized residual norm.
    """
    return torch.linalg.vector_norm(r) / math.sqrt(r.numel())


def amg_pcg_solve(
    rhs: torch.Tensor,
    x: torch.Tensor,
    hierarchy: AMGHierarchy | BatchedAMGHierarchy,
    *,
    tol: float | Sequence[float],
    max_iter: int,
    return_best_result: bool = False,
) -> list[AMGSolveInfo]:
    """Solve ``A x = rhs`` in place with AMG-preconditioned CG.

    ``A`` is the finest level of ``hierarchy``, so the operator the Krylov
    iteration uses is by construction the same one the hierarchy was built for.

    The stopping test is ``||r||_2/sqrt(n) < tol``, matching ``NORM2_NORMALIZED``
    in the CUDA kernels, so a tolerance keeps its meaning when switching solver.

    ``rhs`` may hold several right-hand sides concatenated (as the CUDA solver
    allows); they are solved one after another. ``x`` is written in place.

    Parameters
    ----------
    rhs : torch.Tensor
        Right-hand side(s), with a total size that is a multiple of the matrix
        size.
    x : torch.Tensor
        Initial iterate(s), same size as ``rhs``. Overwritten with the solution.
    hierarchy : AMGHierarchy
        Preconditioner; its finest level supplies the operator ``A``.
    tol : float or sequence of float
        Absolute tolerance on ``||r||_2 / sqrt(n)``, or one per right-hand side.
    max_iter : int
        Maximum CG iterations per right-hand side.
    return_best_result : bool, optional
        Write back the iterate with the smallest residual seen rather than the
        last one. Default is False.

    Returns
    -------
    list of AMGSolveInfo
        One result info per right-hand side.

    Raises
    ------
    ValueError
        If the size of ``rhs`` is not a multiple of the matrix size.
    """
    if isinstance(hierarchy, BatchedAMGHierarchy):
        return _native_pcg_batched(rhs, x, hierarchy, tol, max_iter, return_best_result)

    A = hierarchy.levels[0].A
    n = A.shape[0]

    if rhs.numel() % n:
        raise ValueError(
            f"RHS size {rhs.numel()} is not a multiple of the matrix size {n}."
        )
    n_rhs = rhs.numel() // n

    if USE_NATIVE and rhs.is_cuda and native_amg_available():
        return _native_pcg(rhs, x, hierarchy, tol, max_iter, return_best_result)

    rhs_flat = rhs.reshape(n_rhs, n)
    x_flat = x.reshape(n_rhs, n)

    tols = (
        [float(t) for t in tol] if isinstance(tol, Sequence) else [float(tol)] * n_rhs
    )
    if len(tols) != n_rhs:
        raise ValueError(f"{len(tols)} tolerances for {n_rhs} right-hand sides.")

    infos: list[AMGSolveInfo] = []
    for i in range(n_rhs):
        info = _pcg_one(
            A,
            hierarchy,
            rhs_flat[i],
            x_flat[i],
            tol=tols[i],
            max_iter=max_iter,
            return_best_result=return_best_result,
        )
        infos.append(info)
    return infos


def _native_pcg(
    rhs: torch.Tensor,
    x: torch.Tensor,
    hierarchy: AMGHierarchy,
    tol: float | Sequence[float],
    max_iter: int,
    return_best_result: bool,
) -> list[Any]:
    """Run :func:`amg_pcg_solve` with the CUDA implementation (``_C.AMGPCGSolve``).

    The Krylov iteration runs in the dtype of ``rhs`` with the operator of the
    finest level; the V-cycle runs in ``hierarchy.native_dtype`` (default: the
    dtype of ``rhs``). The whole solve is a single CUDA graph launch.

    Parameters
    ----------
    rhs, x, hierarchy, tol, max_iter, return_best_result
        See :func:`amg_pcg_solve`.

    Returns
    -------
    list of phipict._C.LinearSolverResultInfo
        One result info per right-hand side.
    """
    from phipict import _C

    dtype = rhs.dtype
    pdtype = hierarchy.native_dtype or dtype
    fine = hierarchy.levels[0].A
    key = ("fine", dtype)
    if hierarchy._native_cache is None:
        hierarchy._native_cache = {}
    fine_t = hierarchy._native_cache.get(key)
    if fine_t is None:
        fine_t = [
            fine.crow_indices().to(torch.int32).contiguous(),
            fine.col_indices().to(torch.int32).contiguous(),
            fine.values().to(dtype).contiguous(),
        ]
        hierarchy._native_cache[key] = fine_t
    levels, pinv = hierarchy.native_levels(pdtype)

    rhs_c = rhs.detach().reshape(-1).contiguous()
    in_place = x.is_contiguous() and x.dtype == dtype
    x_work = (
        x.detach().view(-1)
        if in_place
        else x.detach().reshape(-1).to(dtype).contiguous()
    )
    infos = _C.AMGPCGSolve(
        fine_t[0],
        fine_t[1],
        fine_t[2],
        levels,
        pinv,
        rhs_c,
        x_work,
        int(max_iter),
        float(max(tol)) if isinstance(tol, Sequence) else float(tol),
        True,  # NORM2_NORMALIZED, like the CUDA kernels
        bool(return_best_result),
        bool(hierarchy.project_constant),
        int(hierarchy.num_pre_smooth),
        int(hierarchy.num_post_smooth),
        float(_JACOBI_OMEGA),
        [float(t) for t in tol] if isinstance(tol, Sequence) else [],
    )
    if not all(info.isFiniteResidual for info in infos):
        # Match the CUDA solvers' contract: a non-finite result is zeroed
        for i, info in enumerate(infos):
            if not info.isFiniteResidual:
                x_work.view(len(infos), -1)[i].zero_()
    if not in_place:
        with torch.no_grad():
            x.copy_(x_work.view(x.shape))
    return list(infos)


def _native_pcg_batched(
    rhs: torch.Tensor,
    x: torch.Tensor,
    hierarchy: BatchedAMGHierarchy,
    tol: float | Sequence[float],
    max_iter: int,
    return_best_result: bool,
) -> list[Any]:
    """:func:`amg_pcg_solve` for batched environments (a hierarchy per environment)."""
    from phipict import _C

    if not (USE_NATIVE and rhs.is_cuda and native_amg_available()):
        raise NotImplementedError(
            "Batched AMG hierarchies need the native (CUDA) AMG solver."
        )
    dtype = rhs.dtype
    pdtype = hierarchy.native_dtype or dtype
    fine, levels, pinv = hierarchy.native_arguments(dtype, pdtype)
    rhs_c = rhs.detach().reshape(-1).contiguous()
    in_place = x.is_contiguous() and x.dtype == dtype
    x_work = (
        x.detach().view(-1)
        if in_place
        else x.detach().reshape(-1).to(dtype).contiguous()
    )
    infos = _C.AMGPCGSolve(
        fine[0],
        fine[1],
        fine[2],
        levels,
        pinv,
        rhs_c,
        x_work,
        int(max_iter),
        float(max(tol)) if isinstance(tol, Sequence) else float(tol),
        True,
        bool(return_best_result),
        bool(hierarchy.project_constant),
        int(hierarchy.num_pre_smooth),
        int(hierarchy.num_post_smooth),
        float(_JACOBI_OMEGA),
        [float(t) for t in tol] if isinstance(tol, Sequence) else [],
    )
    n = hierarchy.fine_crow.numel() - 1
    for i, info in enumerate(infos):
        if not info.isFiniteResidual:
            x_work.view(-1, n)[i].zero_()
    if not in_place:
        with torch.no_grad():
            x.copy_(x_work.view(x.shape))
    return list(infos)


def _pcg_one(
    A: torch.Tensor,
    M: AMGHierarchy,
    b: torch.Tensor,
    x_out: torch.Tensor,
    *,
    tol: float,
    max_iter: int,
    return_best_result: bool,
) -> AMGSolveInfo:
    """Solve a single right-hand side with preconditioned CG.

    Parameters
    ----------
    A : torch.Tensor
        Sparse CSR system matrix of shape ``[n, n]``.
    M : AMGHierarchy
        Preconditioner applied once per iteration.
    b : torch.Tensor
        Right-hand side of shape ``[n]``.
    x_out : torch.Tensor
        Initial iterate of shape ``[n]``, overwritten with the solution.
    tol : float
        Absolute tolerance on ``||r||_2 / sqrt(n)``.
    max_iter : int
        Maximum CG iterations.
    return_best_result : bool
        Write back the iterate with the smallest residual seen rather than the
        last one.

    Returns
    -------
    AMGSolveInfo
        Iteration count, final residual and convergence flags of the solve.
    """
    b = b.to(M.dtype)
    x = x_out.to(M.dtype).clone()

    if M.project_constant:
        b = b - b.mean()

    r = b - _spmv(A, x)
    if M.project_constant:
        r = r - r.mean()

    z = M(r)
    p = z.clone()
    rz = torch.dot(r, z)

    best_residual = float(_criterion_residual(r))
    best_x = x.clone() if return_best_result else None
    used_iterations = 0
    converged = best_residual < tol

    for iteration in range(max_iter):
        used_iterations = iteration
        residual = float(_criterion_residual(r))
        if not math.isfinite(residual):
            break
        if residual < best_residual:
            best_residual = residual
            if best_x is not None:
                best_x.copy_(x)
        if residual < tol:
            converged = True
            break

        Ap = _spmv(A, p)
        pAp = torch.dot(p, Ap)
        if float(pAp) == 0.0:
            break
        alpha = rz / pAp

        x = x + alpha * p
        r_new = r - alpha * Ap
        if M.project_constant:
            r_new = r_new - r_new.mean()
        z_new = M(r_new)

        # Polak-Ribiere beta: robust when the preconditioner is not exactly
        # constant between iterations, and it degenerates to Fletcher-Reeves
        # when it is
        beta = torch.dot(z_new, r_new - r) / rz
        p = z_new + beta * p
        r, z, rz = r_new, z_new, torch.dot(z_new, r_new)
    else:
        used_iterations = max_iter
        residual = float(_criterion_residual(r))
        if residual < best_residual:
            best_residual = residual
            if best_x is not None:
                best_x.copy_(x)
        converged = residual < tol

    final_residual = float(_criterion_residual(r))
    is_finite = math.isfinite(final_residual) and bool(torch.isfinite(x).all())

    if return_best_result and not converged and best_x is not None:
        x = best_x
        final_residual = best_residual

    if is_finite:
        x_out.copy_(x.to(x_out.dtype))
    else:
        # Match the CUDA solvers' contract: a non-finite result is zeroed rather
        # than propagated into the simulation state
        x_out.zero_()

    return AMGSolveInfo(
        finalResidual=final_residual,
        usedIterations=used_iterations,
        converged=bool(converged),
        isFiniteResidual=is_finite,
    )
