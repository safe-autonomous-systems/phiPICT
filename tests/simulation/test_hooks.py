"""Simulation hooks: callbacks registered per Hook run at their points of a step."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

import phipict  # noqa: E402
from phipict import Hook, Hooks  # noqa: E402
from phipict.core.piso_simulation import append_prep_fn  # noqa: E402

DEV = torch.device("cuda")


def _channel() -> phipict.Domain:
    viscosity = torch.tensor([0.05], dtype=torch.float64)
    domain = phipict.Domain(
        2, viscosity, name="channel", device=DEV, dtype=torch.float64
    )
    vel = torch.zeros(1, 2, 8, 16, dtype=torch.float64, device=DEV)
    block = domain.CreateBlock(velocity=vel, name="channel")
    block.CloseBoundary("-y")
    block.CloseBoundary("+y")
    block.MakePeriodic("x")
    domain.PrepareSolve()
    return domain


def _recorder(calls: list[str], name: str):
    def fn(**kwargs):
        assert set(kwargs) == {
            "domain",
            "local_step",
            "time_step",
            "total_step",
            "total_time",
        }
        calls.append(name)

    return fn


def test_hooks_run_in_step_order():
    calls: list[str] = []
    hooks = (
        Hooks()
        .append(Hook.POST, _recorder(calls, "post"))
        .append(Hook.PRE, _recorder(calls, "pre"))
        .append(Hook.POST_VELOCITY_CORRECTION, _recorder(calls, "correction"))
        .prepend(Hook.PRE, _recorder(calls, "first"))
    )
    assert list(hooks) == [Hook.PRE, Hook.POST_VELOCITY_CORRECTION, Hook.POST]
    assert len(hooks) == 4

    sim = phipict.Simulation(domain=_channel(), dt=0.01, substeps=1, hooks=hooks)
    assert sim.hooks is hooks
    sim.single_step()
    # two pressure correctors
    assert calls == ["first", "pre", "correction", "correction", "post"]


def test_legacy_prep_fn_dicts_still_work():
    calls: list[str] = []
    prep_fn = {"PRE": _recorder(calls, "pre"), Hook.POST: [_recorder(calls, "post")]}
    append_prep_fn(prep_fn, "PRE", _recorder(calls, "pre2"))
    sim = phipict.Simulation(domain=_channel(), dt=0.01, substeps=1, prep_fn=prep_fn)
    append_prep_fn(sim.prep_fn, Hook.POST, _recorder(calls, "post2"))
    sim.single_step()
    assert calls == ["pre", "pre2", "post", "post2"]


def test_unknown_hooks_are_rejected():
    with pytest.raises(ValueError, match="Unknown hook"):
        Hooks().append("PRE_VELOCITY_STEUP", lambda **kw: None)
    with pytest.raises(ValueError, match="Unknown hook"):
        Hooks({"POST_PRESURE": lambda **kw: None})
    with pytest.raises(ValueError, match="Unknown hook"):
        append_prep_fn({}, "PREE", lambda **kw: None)
    with pytest.raises(ValueError, match="not both"):
        phipict.Simulation(domain=_channel(), dt=0.01, hooks=Hooks(), prep_fn={})


def test_mhd_simulation_does_not_modify_the_callers_hooks():
    hooks = Hooks().append(Hook.PRE, lambda **kw: None)
    domain = phipict.Domain(
        3,
        torch.tensor([0.05], dtype=torch.float64),
        name="d",
        device=DEV,
        dtype=torch.float64,
    )
    vel = torch.zeros(1, 3, 4, 4, 8, dtype=torch.float64, device=DEV)
    block = domain.CreateBlock(velocity=vel, name="b")
    for face in ("-y", "+y", "-z", "+z"):
        block.CloseBoundary(face)
    block.MakePeriodic("x")
    domain.PrepareSolve()
    sim = phipict.MHDSimulation(
        domain=domain,
        dt=0.01,
        stuart_number=torch.tensor(1.0),
        e_b=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        hooks=hooks,
    )
    assert len(hooks) == 1
    assert len(sim.hooks) > 1
