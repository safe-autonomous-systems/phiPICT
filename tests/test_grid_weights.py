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

"""Tests for the grid grading schemes in ``phipict.grid.shapes``."""

import numpy as np
import pytest

from phipict.grid.shapes import make_weights

RES = 16
SCHEMES = [("simple", 4.0), ("chebyshev_identity", 0.8), ("tanh", 2.0)]


def _sizes(weights: list[float]) -> np.ndarray:
    return np.diff(np.asarray(weights))


@pytest.mark.parametrize(("grading_type", "grading"), SCHEMES)
@pytest.mark.parametrize("refinement", ["START", "END", "BOTH"])
def test_weights_span_the_unit_interval(grading_type, grading, refinement):
    weights = make_weights(grading_type, RES, grading, refinement)
    assert len(weights) == RES + 1
    assert weights[0] == pytest.approx(0.0, abs=1e-12)
    assert weights[-1] == pytest.approx(1.0, abs=1e-12)
    assert np.all(_sizes(weights) > 0)


@pytest.mark.parametrize(("grading_type", "grading"), SCHEMES)
def test_refinement_side(grading_type, grading):
    start = _sizes(make_weights(grading_type, RES, grading, "START"))
    end = _sizes(make_weights(grading_type, RES, grading, "END"))
    both = _sizes(make_weights(grading_type, RES, grading, "BOTH"))

    assert start[0] < start[-1]
    assert end[-1] < end[0]
    assert both[0] < both[RES // 2] and both[-1] < both[RES // 2]
    np.testing.assert_allclose(both, both[::-1], atol=1e-12)


@pytest.mark.parametrize("grading_type", ["simple", "tanh"])
def test_neutral_grading_is_uniform(grading_type):
    grading = 1.0 if grading_type == "simple" else 0.0
    weights = make_weights(grading_type, RES, grading, "START")
    np.testing.assert_allclose(weights, np.linspace(0.0, 1.0, RES + 1), atol=1e-12)


def test_unknown_options_raise():
    with pytest.raises(ValueError):
        make_weights("cubic", RES, 1.0, "START")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        make_weights("tanh", RES, 1.0, "MIDDLE")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        make_weights("chebyshev_identity", RES, 1.5, "START")
