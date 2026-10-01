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

"""Scale-invariant tolerance specifications for the linear solvers."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any

import torch

__all__ = [
    "SolverTolerance",
    "get_adjoint_rtol",
    "parse_tolerance",
    "resolve_tolerance",
    "rhs_scale",
    "set_adjoint_rtol",
]


# Residuals cannot be driven below the round-off level of the accumulation, so a
# relative tolerance must be floored by this factor times machine epsilon to
# tell a stagnated solve apart from a failed one
PRECISION_FLOOR_FACTOR = 8.0

# Relative residual every adjoint solve is held to, overriding the per-solve rule;
# see `set_adjoint_rtol`
_ADJOINT_RTOL: float | None = None

# Adjoint rtol of a solve whose forward tolerance is absolute (no rtol). On
# CylinderJet2D, whose forward tolerances are all absolute, it gives a gradient
# within 9e-6 of a 1e-10 reference; deriving an rtol from the absolute forward
# tolerance instead was 5e-3 off
ADJOINT_RTOL_FALLBACK = 1e-4


def set_adjoint_rtol(rtol: float | None) -> None:
    """Set the relative residual every adjoint (backward) linear solve is held to.

    The adjoint solve stops on ``||r||_2 < rtol * ||g||_2``, with ``g`` the incoming
    gradient. The tolerance is strictly proportional to ``g``, so the backward pass
    is invariant to the scale of the loss. ``None`` (the default) holds each
    adjoint solve to the ``rtol`` of its forward solve's
    :class:`SolverTolerance`, and to :data:`ADJOINT_RTOL_FALLBACK` if the forward
    tolerance is absolute.

    Parameters
    ----------
    rtol : float or None
        Relative residual target of all adjoint solves, or None to match each
        forward solve.

    Raises
    ------
    ValueError
        If ``rtol`` is not positive.
    """
    global _ADJOINT_RTOL
    if rtol is not None:
        rtol = float(rtol)
        if not rtol > 0.0:
            raise ValueError(f"adjoint rtol must be positive, got {rtol}.")
    _ADJOINT_RTOL = rtol


def get_adjoint_rtol() -> float | None:
    """Return the relative residual of the adjoint solves, see `set_adjoint_rtol`.

    Returns
    -------
    float or None
        The adjoint relative residual target, or None if each adjoint solve
        matches its forward solve.
    """
    return _ADJOINT_RTOL


def rhs_scale(rhs: torch.Tensor) -> float:
    """Return ``||b||_2 / sqrt(n)``, the RHS in the solver's residual units.

    This is the same norm the ``NORM2_NORMALIZED`` criterion applies to the
    residual, so ``residual / rhs_scale`` is directly the relative residual.

    Parameters
    ----------
    rhs : torch.Tensor
        Right-hand side of the linear system.

    Returns
    -------
    float
        The RMS norm of ``rhs``, or 0.0 if ``rhs`` is empty.
    """
    n = rhs.numel()
    if n == 0:
        return 0.0
    # float64 accumulation: the norm of a large single-precision RHS is otherwise
    # itself inaccurate enough to matter at the tolerances used here
    norm = torch.linalg.vector_norm(rhs.detach().to(torch.float64))
    return float(norm.item()) / math.sqrt(n)


@dataclass(frozen=True)
class SolverTolerance:
    """A tolerance specified relative to the right-hand side.

    Parameters
    ----------
    rtol: float or None
        Relative residual target. The resolved absolute tolerance is
        ``rtol * ||b||_2 / sqrt(n)``, which makes the solver's stop test
        equivalent to ``||r||_2 < rtol * ||b||_2``.

    atol: float or None
        Absolute floor, in the same RMS units as the solver criterion. Prevents
        over-solving when the RHS is (near) zero, e.g. the electric potential
        equation at rest, where ``u x e_b`` is uniform and its divergence
        vanishes. At least one of ``rtol``/``atol`` must be given.

    Notes
    -----
    The resolved tolerance is additionally floored at
    ``PRECISION_FLOOR_FACTOR * eps * ||b||_2 / sqrt(n)``, below which the solver
    can only stagnate.
    """

    rtol: float | None = None
    atol: float | None = None

    def __post_init__(self) -> None:
        if self.rtol is None and self.atol is None:
            raise ValueError("SolverTolerance needs at least one of rtol, atol.")
        for name in ("rtol", "atol"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, numbers.Real):
                raise TypeError(f"SolverTolerance.{name} must be a float or None.")
            if not float(value) > 0:
                raise ValueError(f"SolverTolerance.{name} must be positive.")

    def resolve(self, rhs: torch.Tensor) -> float:
        """Return the absolute tolerance to hand to the solver for this RHS.

        Parameters
        ----------
        rhs : torch.Tensor
            Right-hand side the tolerance is resolved against.

        Returns
        -------
        float
            Absolute tolerance, floored at the dtype's precision limit and, if
            the RHS is (near) zero and no ``atol`` was given, at the dtype's
            default solver tolerance.
        """
        scale = rhs_scale(rhs)
        tol = 0.0
        if self.rtol is not None:
            tol = self.rtol * scale
        if self.atol is not None:
            tol = max(tol, self.atol)

        floor = PRECISION_FLOOR_FACTOR * torch.finfo(rhs.dtype).eps * scale
        tol = max(tol, floor)

        if not tol > 0.0:
            # ``scale`` is zero (or denormal) and no atol was given. The solve is
            # trivially satisfied by x=0; any positive number does, so use the
            # dtype default rather than handing the kernel a zero tolerance it
            # can never meet
            from phipict.core import piso_diff

            default_tol = piso_diff._get_solver_tolerance(dtype=rhs.dtype)
            assert default_tol is not None, f"No default tolerance for {rhs.dtype}"
            tol = default_tol
        return float(tol)

    def resolve_batched(self, rhs: torch.Tensor, batch_size: int) -> list[float]:
        """Resolve the tolerance separately for every batched environment.

        The environments' parts of ``rhs`` are its ``batch_size`` equal,
        consecutive slices. Every environment is held to the tolerance its own
        right-hand side gives, exactly as if it were solved alone; one global
        norm would couple the environments.

        Parameters
        ----------
        rhs : torch.Tensor
            Right-hand side of all environments.
        batch_size : int
            Number of environments.

        Returns
        -------
        list of float
            One absolute tolerance per environment.
        """
        slices = rhs.detach().reshape(batch_size, -1)
        n = slices.shape[1]
        norms = torch.linalg.vector_norm(slices.to(torch.float64), dim=1)
        scales = (norms / math.sqrt(n)).tolist() if n > 0 else [0.0] * batch_size
        return [self._resolve_scale(scale, rhs.dtype) for scale in scales]

    def _resolve_scale(self, scale: float, dtype: torch.dtype) -> float:
        """The absolute tolerance for a right-hand side of RMS norm ``scale``."""
        tol = 0.0
        if self.rtol is not None:
            tol = self.rtol * scale
        if self.atol is not None:
            tol = max(tol, self.atol)
        tol = max(tol, PRECISION_FLOOR_FACTOR * torch.finfo(dtype).eps * scale)
        if not tol > 0.0:
            from phipict.core import piso_diff

            default_tol = piso_diff._get_solver_tolerance(dtype=dtype)
            assert default_tol is not None, f"No default tolerance for {dtype}"
            tol = default_tol
        return float(tol)

    def __str__(self) -> str:
        parts = []
        if self.rtol is not None:
            parts.append(f"rtol={self.rtol:.03e}")
        if self.atol is not None:
            parts.append(f"atol={self.atol:.03e}")
        return f"SolverTolerance({', '.join(parts)})"


def parse_tolerance(spec: Any) -> float | SolverTolerance | None:
    """Convert a config value into a tolerance usable by the simulation.

    Accepts ``None``, a number (absolute tolerance, unchanged semantics), a
    :class:`SolverTolerance`, or a mapping with ``rtol``/``atol`` keys. The
    latter is what a Hydra/OmegaConf config node looks like, e.g.::

        tol:
          potential: {rtol: 1.0e-6, atol: 1.0e-12}
          pressure: 1.0e-5

    Parameters
    ----------
    spec : Any
        ``None``, a real number, a :class:`SolverTolerance`, or a mapping with
        ``rtol`` and/or ``atol`` keys.

    Returns
    -------
    float or SolverTolerance or None
        ``None`` and :class:`SolverTolerance` pass through, numbers become
        floats, and mappings become a :class:`SolverTolerance`.

    Raises
    ------
    ValueError
        If a mapping contains keys other than ``rtol``/``atol``.
    TypeError
        If ``spec`` is of none of the accepted forms.
    """
    if spec is None or isinstance(spec, SolverTolerance):
        return spec
    if isinstance(spec, numbers.Real) and not isinstance(spec, bool):
        return float(spec)

    # OmegaConf DictConfig is a Mapping, but only after resolution; go through
    # the generic mapping protocol so no hard dependency on omegaconf is needed
    if hasattr(spec, "keys"):
        keys = set(spec.keys())
        unknown = keys - {"rtol", "atol"}
        if unknown:
            raise ValueError(
                f"Unknown tolerance keys {sorted(unknown)}, "
                "expected 'rtol' and/or 'atol'."
            )
        rtol = spec.get("rtol", None)
        atol = spec.get("atol", None)
        return SolverTolerance(
            rtol=None if rtol is None else float(rtol),
            atol=None if atol is None else float(atol),
        )

    raise TypeError(
        f"Cannot interpret {spec!r} as a solver tolerance; expected None, a float, "
        "a SolverTolerance, or a mapping with rtol/atol."
    )


def resolve_tolerance(tol: Any, rhs: torch.Tensor, batch_size: int = 1) -> Any:
    """Resolve ``tol`` against ``rhs``, passing non-relative values through.

    This is the single choke point that turns a :class:`SolverTolerance` into the
    absolute number the CUDA kernels expect. Floats, tensors and ``None`` are
    returned unchanged so existing behaviour is bit-identical.

    Parameters
    ----------
    tol : Any
        A :class:`SolverTolerance`, or any value to pass through unchanged.
    rhs : torch.Tensor
        Right-hand side a :class:`SolverTolerance` is resolved against.
    batch_size : int, optional
        Number of batched environments whose right-hand sides ``rhs`` holds.
        With more than one, a :class:`SolverTolerance` is resolved per
        environment. Default is 1.

    Returns
    -------
    Any
        A float if ``tol`` is a :class:`SolverTolerance` (a list with one float
        per environment if ``batch_size > 1``), otherwise ``tol`` unchanged.
    """
    if isinstance(tol, SolverTolerance):
        if batch_size > 1:
            return tol.resolve_batched(rhs, batch_size)
        return tol.resolve(rhs)
    return tol
