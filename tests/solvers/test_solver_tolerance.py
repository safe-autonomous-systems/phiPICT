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

"""Tests for the solver tolerance specification."""

import pytest
import torch
from test_domain_io_mhd import (  # noqa: E402
    DTYPE,
    HARTMANN,
    REYNOLDS,
    _make_domain,
)

from phipict.solvers import stats as solver_stats  # noqa: E402
from phipict.solvers.tolerance import (  # noqa: E402
    SolverTolerance,
    parse_tolerance,
    resolve_tolerance,
    rhs_scale,
)

# ---------------------------------------------------------------------------
# Pure-python parts (no GPU needed)
# ---------------------------------------------------------------------------


def test_rhs_scale_is_the_solver_criterion_norm() -> None:
    """``rhs_scale`` must be ||b||_2/sqrt(n), the norm NORM2_NORMALIZED uses."""
    b = torch.full((1000,), 3.0, dtype=torch.float64)
    assert rhs_scale(b) == pytest.approx(3.0)


def test_relative_tolerance_resolves_against_the_rhs() -> None:
    b = torch.full((1000,), 3.0, dtype=torch.float64)
    assert SolverTolerance(rtol=1e-6).resolve(b) == pytest.approx(3e-6)
    # The atol floor wins when it is the larger of the two.
    assert SolverTolerance(rtol=1e-12, atol=1e-9).resolve(b) == pytest.approx(1e-9)


def test_relative_tolerance_is_scale_invariant() -> None:
    """The same rtol on a RHS 100x larger must ask for a 100x larger residual.

    This is the property that lets a tolerance tuned on the small duct carry over
    to the large one, where the potential-equation RHS is bigger.
    """
    small = torch.full((1000,), 1.0, dtype=torch.float64)
    large = 100.0 * small
    tol = SolverTolerance(rtol=1e-6)
    assert tol.resolve(large) == pytest.approx(100.0 * tol.resolve(small))


def test_relative_tolerance_is_floored_at_round_off() -> None:
    b = torch.full((1000,), 3.0, dtype=torch.float64)
    resolved = SolverTolerance(rtol=1e-30).resolve(b)
    assert resolved > 0.0
    assert resolved < 1e-13  # ~ 8 * eps * 3


def test_zero_rhs_falls_back_to_the_dtype_default() -> None:
    """A zero RHS must not produce a zero (unreachable) tolerance."""
    assert (
        SolverTolerance(rtol=1e-6).resolve(torch.zeros(10, dtype=torch.float64)) > 0.0
    )


def test_parse_tolerance_accepts_configs_floats_and_none() -> None:
    assert parse_tolerance(None) is None
    assert parse_tolerance(1e-5) == 1e-5
    assert parse_tolerance({"rtol": 1e-6, "atol": 1e-12}) == SolverTolerance(
        rtol=1e-6, atol=1e-12
    )
    with pytest.raises(ValueError):
        parse_tolerance({"rtol": 1e-6, "nonsense": 1.0})
    with pytest.raises(ValueError):
        SolverTolerance()


def test_absolute_tolerances_pass_through_untouched() -> None:
    """Floats keep their exact meaning, so existing runs stay reproducible."""
    b = torch.ones(10, dtype=torch.float64)
    assert resolve_tolerance(1e-5, b) == 1e-5
    assert resolve_tolerance(None, b) is None


def test_recorder_is_inactive_by_default() -> None:
    assert not solver_stats.is_active()


def test_recorder_aggregates_and_reports() -> None:
    class Info:
        def __init__(self, iters, residual, converged=True, finite=True):
            self.usedIterations = iters
            self.finalResidual = residual
            self.converged = converged
            self.isFiniteResidual = finite

    b = torch.full((100,), 2.0, dtype=torch.float64)
    with solver_stats.SolverStatsRecorder(time_solves=False) as rec:
        assert solver_stats.is_active()
        probe = solver_stats.Probe(b, "epot")
        probe.start()
        probe.stop([Info(12, 1e-7)], tolerance=2e-6, max_iterations=1000)

        probe = solver_stats.Probe(b, "pressure")
        probe.start()
        probe.stop(
            [Info(1000, 5e-5, converged=False)], tolerance=2e-6, max_iterations=1000
        )

    assert not solver_stats.is_active()
    summary = rec.summary()
    assert summary["solver/epot/n_solves"] == 1
    assert summary["solver/epot/iters_mean"] == 12
    # rel_res = 1e-7 / 2.0
    assert summary["solver/epot/rel_res_max"] == pytest.approx(5e-8)
    assert summary["solver/pressure/n_not_converged"] == 1
    assert summary["solver/pressure/saturation_max"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# End to end on a tiny MHD domain
# ---------------------------------------------------------------------------


def _make_sim(domain, potential_tol, advection_tol=1e-5, pressure_tol=5e-6):
    from phipict.simulation.mhd import MHDSimulation

    return MHDSimulation(
        domain=domain,
        dt=1e-3,
        stuart_number=torch.tensor(HARTMANN**2 / REYNOLDS),
        e_b=torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE),
        substeps=1,
        corrector_steps=2,
        advection_tol=advection_tol,
        pressure_tol=pressure_tol,
        potential_tol=potential_tol,
        non_orthogonal=False,
        potential_normalize=True,
        potential_use_BiCG=False,
        potential_solve_max_iter=1000,
    )


@pytest.mark.parametrize("tol", [1e-6, SolverTolerance(rtol=1e-6, atol=1e-14)])
def test_a_step_runs_with_either_tolerance_form(tol) -> None:
    domain = _make_domain(hartmann_Cw=0.0, heat_both_walls=False)
    sim = _make_sim(domain, potential_tol=tol)
    sim.make_divergence_free(max_iter=1000)
    sim.single_step()

    block = domain.getBlock(0)
    assert torch.isfinite(block.velocity).all()
    assert block.hasEpot() and torch.isfinite(block.epot).all()


def test_every_solve_is_tagged_and_residuals_meet_the_tolerance() -> None:
    """Telemetry must name each solve and report a residual under its tolerance."""
    domain = _make_domain(hartmann_Cw=0.0, heat_both_walls=False)
    sim = _make_sim(domain, potential_tol=1e-6)
    sim.make_divergence_free(max_iter=1000)

    with solver_stats.SolverStatsRecorder(time_solves=False, keep_records=True) as rec:
        sim.single_step()

    tags = set(rec.tags)
    assert "epot" in tags, f"potential solve was not tagged, saw {tags}"
    assert "pressure" in tags, f"pressure solve was not tagged, saw {tags}"
    assert "unknown" not in tags, "some solve is still untagged"

    for record in rec.records:
        if record.converged:
            assert record.final_residual <= record.tolerance, (
                f"{record.tag}: reported converged at {record.final_residual:.3e} "
                f"but the tolerance was {record.tolerance:.3e}"
            )
        assert record.is_finite

    epot_tol = {r.tolerance for r in rec.records if r.tag == "epot"}
    assert epot_tol == {1e-6}


def test_relative_tolerance_reproduces_the_equivalent_absolute_solve() -> None:
    """rtol is a reparameterisation: same resolved number => same trajectory.

    Run one step with a relative potential tolerance, read back the absolute
    tolerance it resolved to, then re-run from a fresh domain with that number
    given directly. The two must agree to round-off.
    """
    domain = _make_domain(hartmann_Cw=0.0, heat_both_walls=False)
    sim = _make_sim(domain, potential_tol=SolverTolerance(rtol=1e-6, atol=1e-14))
    sim.make_divergence_free(max_iter=1000)
    with solver_stats.SolverStatsRecorder(time_solves=False, keep_records=True) as rec:
        sim.single_step()
    epot_records = [r for r in rec.records if r.tag == "epot"]
    assert epot_records, "no epot solve was recorded"
    resolved = epot_records[0].tolerance
    assert resolved == pytest.approx(1e-6 * epot_records[0].rhs_scale, rel=1e-9)
    relative_epot = domain.getBlock(0).epot.clone()

    domain2 = _make_domain(hartmann_Cw=0.0, heat_both_walls=False)
    sim2 = _make_sim(domain2, potential_tol=resolved)
    sim2.make_divergence_free(max_iter=1000)
    sim2.single_step()
    absolute_epot = domain2.getBlock(0).epot

    torch.testing.assert_close(relative_epot, absolute_epot, rtol=1e-10, atol=1e-12)
