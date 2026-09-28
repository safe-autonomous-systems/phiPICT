"""Callbacks of a simulation at fixed points of a PISO step.

:class:`Hook` names the points, :class:`Hooks` holds the callbacks per point::

    hooks = Hooks().append(Hook.PRE, update_boundaries).append(
        Hook.PRE_VELOCITY_SETUP, add_forcing
    )
    sim = Simulation(domain=domain, hooks=hooks)

A :class:`Hook` member is a ``str`` equal to its name, and a plain dict of
callbacks by hook name (the former ``prep_fn``) is still accepted wherever a
:class:`Hooks` is.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from enum import StrEnum
from typing import Any

__all__ = ["Hook", "Hooks"]

HookFn = Callable[..., Any]


class Hook(StrEnum):
    """A point of a PISO step at which the registered callbacks are run.

    Listed in the order they run in one step. Every callback is called with the
    keyword arguments ``domain``, ``local_step``, ``time_step``, ``total_step`` and
    ``total_time``.
    """

    #: Start of the step, before anything is assembled (boundary updates, forcing).
    PRE = "PRE"
    #: After the passive-scalar advection system is assembled, before its solve.
    POST_SCALAR_SETUP = "POST_SCALAR_SETUP"
    #: Before the momentum system is assembled (e.g. velocity sources).
    PRE_VELOCITY_SETUP = "PRE_VELOCITY_SETUP"
    #: After the momentum system is assembled, before its solve.
    POST_VELOCITY_SETUP = "POST_VELOCITY_SETUP"
    #: After the momentum prediction, before the pressure correctors.
    POST_PREDICTION = "POST_PREDICTION"
    #: After a pressure system is assembled, before its solve.
    POST_PRESSURE_SETUP = "POST_PRESSURE_SETUP"
    #: After a pressure solve.
    POST_PRESSURE_RESULT = "POST_PRESSURE_RESULT"
    #: After the non-orthogonal pressure iterations of a corrector.
    POST_PRESSURE_NON_ORTHO = "POST_PRESSURE_NON_ORTHO"
    #: After the velocity correction of a corrector.
    POST_VELOCITY_CORRECTION = "POST_VELOCITY_CORRECTION"
    #: End of the step, after the velocity is written back to the blocks.
    POST = "POST"

    @classmethod
    def parse(cls, name: str | Hook) -> Hook:
        """The hook of ``name``, rejecting names no simulation runs.

        Parameters
        ----------
        name : str or Hook
            Hook or hook name.

        Returns
        -------
        Hook
            The hook.

        Raises
        ------
        ValueError
            If ``name`` is not a hook, e.g. a typo, whose callbacks would silently
            never run.
        """
        try:
            return cls(name)
        except ValueError:
            valid = ", ".join(hook.value for hook in cls)
            raise ValueError(f"Unknown hook {name!r}; valid hooks: {valid}.") from None


class Hooks:
    """The callbacks of a simulation, per :class:`Hook`, in the order they run.

    Lightweight and strict: only hooks a simulation runs are accepted, so a
    misspelled hook fails when it is registered instead of never running.

    Parameters
    ----------
    callbacks : Hooks, Mapping or None, optional
        Initial callbacks: a callable or a sequence of callables per hook (or hook
        name). Default is None (no callbacks).
    """

    def __init__(
        self, callbacks: Hooks | Mapping[Hook | str, Any] | None = None
    ) -> None:
        self._callbacks: dict[Hook, list[HookFn]] = {}
        if callbacks is None:
            return
        items = (
            callbacks._callbacks.items()
            if isinstance(callbacks, Hooks)
            else (callbacks.items())
        )
        for hook, fns in items:
            if fns is None:
                continue
            if callable(fns):
                fns = [fns]
            for fn in fns:
                if fn is not None:
                    self.append(hook, fn)

    @classmethod
    def coerce(cls, hooks: Hooks | Mapping[Hook | str, Any] | None) -> Hooks:
        """``hooks`` as :class:`Hooks`: returned as is if it is one, else converted.

        Parameters
        ----------
        hooks : Hooks, Mapping or None
            Hooks, a dict of callbacks by hook (name), or None for no callbacks.

        Returns
        -------
        Hooks
            The hooks.
        """
        return hooks if isinstance(hooks, Hooks) else cls(hooks)

    def append(self, hook: Hook | str, fn: HookFn) -> Hooks:
        """Register ``fn`` to run at ``hook`` after the callbacks already there.

        Parameters
        ----------
        hook : Hook or str
            The hook, or its name.
        fn : Callable
            Callback, called with keyword arguments only.

        Returns
        -------
        Hooks
            ``self``, so registrations can be chained.
        """
        self._callbacks.setdefault(Hook.parse(hook), []).append(fn)
        return self

    def prepend(self, hook: Hook | str, fn: HookFn) -> Hooks:
        """Register ``fn`` to run at ``hook`` before the callbacks already there.

        Parameters
        ----------
        hook : Hook or str
            The hook, or its name.
        fn : Callable
            Callback, called with keyword arguments only.

        Returns
        -------
        Hooks
            ``self``, so registrations can be chained.
        """
        self._callbacks.setdefault(Hook.parse(hook), []).insert(0, fn)
        return self

    def run(self, hook: Hook, **kwargs: Any) -> None:
        """Call the callbacks of ``hook`` in order, with ``kwargs``.

        Parameters
        ----------
        hook : Hook
            The hook.
        **kwargs : Any
            Keyword arguments for every callback.
        """
        for fn in self._callbacks.get(hook, ()):
            fn(**kwargs)

    def __getitem__(self, hook: Hook | str) -> tuple[HookFn, ...]:
        """The callbacks of ``hook``, in order (empty if there are none)."""
        return tuple(self._callbacks.get(Hook.parse(hook), ()))

    def __contains__(self, hook: object) -> bool:
        """Whether callbacks are registered at ``hook``."""
        return bool(self._callbacks.get(hook))  # type: ignore[call-overload]

    def __iter__(self) -> Iterator[Hook]:
        """The hooks with callbacks, in step order."""
        return (hook for hook in Hook if self._callbacks.get(hook))

    def __len__(self) -> int:
        """The number of registered callbacks."""
        return sum(len(fns) for fns in self._callbacks.values())

    def __repr__(self) -> str:
        entries = ", ".join(
            f"{hook.value}: [{', '.join(_name(fn) for fn in self[hook])}]"
            for hook in self
        )
        return f"Hooks({{{entries}}})"


def _name(fn: HookFn) -> str:
    return getattr(fn, "__qualname__", None) or repr(fn)
