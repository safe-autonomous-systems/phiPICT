"""Batched environments: B copies of one domain simulated together.

A batched domain holds B environments that share the grid, the block and boundary
layout and the sparsity patterns, but have their own state (``[B, ...]`` block
fields). See ``Domain.setBatchSize``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from phipict import _C

__all__ = ["EnvStateSnapshot", "copy_env_state", "stack_domains"]


def copy_env_state(dst: _C.Domain, env_idx: int, src: _C.Domain) -> None:
    """Copy the state of a single-environment domain into one environment of a batch.

    The block fields (velocity, pressure, passive scalar and electric potential)
    of ``src`` are written into slice ``env_idx`` of the fields of ``dst`` in
    place. Boundary data is part of the state too (e.g. an advective outflow):
    the velocity, passive scalar and Dirichlet potential values of every fixed
    boundary that differ from those of ``dst`` are copied, which makes that
    boundary per-environment (``[B, ...]``) if it was shared. Boundary data equal
    to that of ``dst`` stays shared.

    Parameters
    ----------
    dst : phipict._C.Domain
        Batched domain to write into.
    env_idx : int
        Environment of ``dst`` to overwrite.
    src : phipict._C.Domain
        Domain of batch size 1 with the same blocks as ``dst``.

    Raises
    ------
    ValueError
        If the domains do not match.
    """
    if src.getBatchSize() != 1:
        raise ValueError(f"src must have batch size 1, got {src.getBatchSize()}.")
    if not 0 <= env_idx < dst.getBatchSize():
        raise ValueError(
            f"env_idx {env_idx} out of range for batch size {dst.getBatchSize()}."
        )
    if src.getNumBlocks() != dst.getNumBlocks():
        raise ValueError("src and dst have a different number of blocks.")

    with torch.no_grad():
        for dst_block, src_block in zip(dst.getBlocks(), src.getBlocks(), strict=True):
            fields = [(dst_block.velocity, src_block.velocity)]
            fields.append((dst_block.pressure, src_block.pressure))
            if dst_block.hasPassiveScalar() and src_block.hasPassiveScalar():
                fields.append((dst_block.passiveScalar, src_block.passiveScalar))
            if dst_block.epot is not None and src_block.epot is not None:
                fields.append((dst_block.epot, src_block.epot))
            for dst_field, src_field in fields:
                if dst_field.shape[1:] != src_field.shape[1:]:
                    raise ValueError(
                        f"Block field shapes differ: {tuple(dst_field.shape[1:])} vs "
                        f"{tuple(src_field.shape[1:])}."
                    )
                dst_field[env_idx].copy_(src_field[0])

            n_bounds = 2 * dst.getSpatialDims()
            for bound_idx in range(n_bounds):
                dst_bound = dst_block.getBoundary(bound_idx)
                src_bound = src_block.getBoundary(bound_idx)
                if not isinstance(dst_bound, _C.FixedBoundary) or not isinstance(
                    src_bound, _C.FixedBoundary
                ):
                    continue
                _copy_boundary_data(
                    dst_bound.velocity,
                    src_bound.velocity,
                    env_idx,
                    dst.getBatchSize(),
                    dst_bound.setVelocity,
                )
                if dst_bound.hasPassiveScalar() and src_bound.hasPassiveScalar():
                    _copy_boundary_data(
                        dst_bound.passiveScalar,
                        src_bound.passiveScalar,
                        env_idx,
                        dst.getBatchSize(),
                        dst_bound.setPassiveScalar,
                    )
                # Dirichlet potential values (e.g. electrode actuators) are state; the
                # potential BC types are structure, shared by all environments
                if dst_bound.hasPotentialValues() or src_bound.hasPotentialValues():
                    src_values = src_bound.potentialValues
                    dst_values = dst_bound.potentialValues
                    if src_values is None:
                        assert dst_values is not None
                        src_values = torch.zeros_like(dst_values[:1])
                    if dst_values is None:
                        dst_values = torch.zeros_like(src_values)
                        dst_bound.setPotentialValues(dst_values)
                    _copy_boundary_data(
                        dst_values,
                        src_values,
                        env_idx,
                        dst.getBatchSize(),
                        dst_bound.setPotentialValues,
                    )
        dst.UpdateDomainData()


def _copy_boundary_data(
    dst: torch.Tensor,
    src: torch.Tensor,
    env_idx: int,
    batch_size: int,
    setter: Callable[[torch.Tensor], None],
) -> None:
    """Write boundary data ``src`` (batch size 1) into environment ``env_idx``.

    Shared data (batch size 1) that already equals ``src`` is left shared;
    otherwise it becomes per-environment first.
    """
    if dst.shape[1:] != src.shape[1:]:
        raise ValueError(
            f"Boundary data shapes differ: {tuple(dst.shape[1:])} vs "
            f"{tuple(src.shape[1:])}."
        )
    if dst.size(0) == 1:
        if torch.equal(dst, src):
            return
        dst = dst.expand(batch_size, *dst.shape[1:]).clone()
        dst[env_idx].copy_(src[0])
        setter(dst)
    else:
        dst[env_idx].copy_(src[0])


def stack_domains(domains: Sequence[_C.Domain]) -> _C.Domain:
    """Combine single-environment domains of one setup into a batched domain.

    The first domain becomes the batched domain (in place): its batch size is set
    to ``len(domains)`` and environment ``b`` gets the state of ``domains[b]``.
    Grid, boundaries and solver settings are those of the first domain.

    Parameters
    ----------
    domains : Sequence of phipict._C.Domain
        Domains of batch size 1 with identical blocks, e.g. different initial
        states of the same environment.

    Returns
    -------
    phipict._C.Domain
        The batched domain, prepared for solving.

    Raises
    ------
    ValueError
        If no domain is given or a domain is already batched.
    """
    if len(domains) == 0:
        raise ValueError("stack_domains needs at least one domain.")
    batched = domains[0]
    if batched.getBatchSize() != 1:
        raise ValueError(
            f"Domains must have batch size 1, got {batched.getBatchSize()}."
        )

    batched.setBatchSize(len(domains))
    batched.PrepareSolve()
    for env_idx, domain in enumerate(domains[1:], start=1):
        copy_env_state(batched, env_idx, domain)
    return batched


class EnvStateSnapshot:
    """State of some environments of a batched domain, to undo a step for them.

    Used by per-environment adaptive time stepping: an environment that has
    already covered its time step still runs through the remaining substeps of
    the batch, and its state is restored afterwards.

    The state is the one :func:`copy_env_state` copies (block fields and fixed
    boundary data), plus the velocity sources, the solution vectors of the domain
    (the next solve's initial guess, and the potential the Lorentz force uses)
    and ``extra`` flat tensors. Tensors are read through getters when restoring,
    since a step replaces rather than overwrites most of them.

    Parameters
    ----------
    domain : phipict._C.Domain
        Batched domain.
    env_idx : torch.Tensor
        Indices of the environments to restore, a 1D integer tensor.
    extra : Sequence of Callable[[], torch.Tensor or None], optional
        Getters of further flat per-environment state (``B`` consecutive slices),
        restored in place.
    """

    def __init__(
        self,
        domain: _C.Domain,
        env_idx: torch.Tensor,
        extra: Sequence[Callable[[], torch.Tensor | None]] = (),
    ) -> None:
        self._domain = domain
        self._batch = domain.getBatchSize()
        self._idx = env_idx.to(device=domain.getDevice(), dtype=torch.long)

        getters: list[Callable[[], torch.Tensor | None]] = []
        for block in domain.getBlocks():
            getters += [
                lambda block=block: block.velocity,
                lambda block=block: block.pressure,
                lambda block=block: block.passiveScalar,
                lambda block=block: block.epot,
            ]
        getters += [
            lambda: domain.velocityResult,
            lambda: domain.pressureResult,
            lambda: domain.scalarResult,
            lambda: domain.epotResult,
            *extra,
        ]
        # (getter, rows of the environments) of fields with a slice per environment
        self._fields: list[tuple[Callable[[], torch.Tensor | None], torch.Tensor]] = []
        # (getter, setter, data pointer, version, copy) of boundary data and velocity
        # sources, which may be shared
        self._bounds: list[
            tuple[
                Callable[[], torch.Tensor | None],
                Callable[[torch.Tensor], None],
                int,
                int,
                torch.Tensor | None,
            ]
        ] = []
        with torch.no_grad():
            for getter in getters:
                t = getter()
                if t is None or t.numel() == 0:
                    continue
                self._fields.append(
                    (getter, self._env_rows(t).index_select(0, self._idx))
                )

            dims = domain.getSpatialDims()
            for block in domain.getBlocks():
                # state too where a hook accumulates into it (the MHD Lorentz force)
                source = block.velocitySource
                self._bounds.append(
                    (
                        lambda block=block: block.velocitySource,
                        block.setVelocitySource,
                        0 if source is None else source.data_ptr(),
                        0 if source is None else source._version,
                        None if source is None else source.clone(),
                    )
                )
                for bound_idx in range(2 * dims):
                    bound = block.getBoundary(bound_idx)
                    if not isinstance(bound, _C.FixedBoundary):
                        continue
                    data = [
                        (lambda bound=bound: bound.velocity, bound.setVelocity),
                        (
                            lambda bound=bound: bound.potentialValues,
                            bound.setPotentialValues,
                        ),
                    ]
                    if bound.hasPassiveScalar():
                        data.append(
                            (
                                lambda bound=bound: bound.passiveScalar,
                                bound.setPassiveScalar,
                            )
                        )
                    for getter, setter in data:
                        t = getter()
                        if t is None:
                            continue
                        self._bounds.append(
                            (getter, setter, t.data_ptr(), t._version, t.clone())
                        )

    def _env_rows(self, t: torch.Tensor) -> torch.Tensor:
        """``t`` as ``[B, -1]``: block fields are ``[B, ...]``, flat vectors B parts."""
        if t.numel() % self._batch != 0:
            raise ValueError(
                f"Tensor of {t.numel()} elements has no slice per environment "
                f"(batch size {self._batch})."
            )
        return t.view(self._batch, -1)

    def restore(self) -> None:
        """Write the saved state back into the saved environments."""
        B = self._batch
        with torch.no_grad():
            for getter, rows in self._fields:
                t = getter()
                if t is None or t.numel() == 0:
                    continue
                self._env_rows(t).index_copy_(0, self._idx, rows)

            changed_bounds = False
            for getter, setter, ptr, version, saved in self._bounds:
                t = getter()
                if t is None or (t.data_ptr() == ptr and t._version == version):
                    continue  # untouched by the step
                if saved is None:
                    saved = torch.zeros_like(t[:1])  # none: no source
                if t.shape[1:] != saved.shape[1:]:
                    raise ValueError(
                        f"Boundary data changed shape during the step: "
                        f"{tuple(saved.shape)} to {tuple(t.shape)}."
                    )
                # restoring some environments makes shared data per-environment
                restored = t.expand(B, *t.shape[1:]).clone()
                restored[self._idx] = saved.expand(B, *saved.shape[1:])[self._idx]
                setter(restored)
                changed_bounds = True
        if changed_bounds:
            self._domain.UpdateDomainData()
