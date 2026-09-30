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

"""Gradings of ``phipict.meshing``, checked against PICT's legacy weights.

The legacy weights are stored in ``tests/meshing/data/legacy_weights.npz`` (made by
``make_references.py``), so ``phipict.grid`` is not needed.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

import phipict.meshing as pm

C = pm.Cluster
LEGACY = np.load(Path(__file__).parent / "data" / "legacy_weights.npz")

ALL = [
    pm.Uniform(),
    pm.Simple(4.0),
    pm.Simple(0.2),
    pm.Symmetric(10.0),
    pm.Geometric(1.1, C.START),
    pm.Geometric(1.1, C.BOTH),
    pm.Tanh(2.0),
    pm.Tanh(1.5, C.END),
    pm.Cosine(),
    pm.Cosine(C.START),
    pm.ChebyshevBlend(0.8),
    pm.ChebyshevBlend(0.5, C.END),
    pm.FirstCell(0.01),
    pm.FirstCell(0.01, C.BOTH),
    pm.MultiGrading([(0.2, 0.3, 4.0), (0.6, 0.4, 1.0), (0.2, 0.3, 0.25)]),
]


@pytest.mark.parametrize("grading", ALL, ids=repr)
@pytest.mark.parametrize("cells", [1, 2, 5, 16, 33])
def test_weights_span_unit_interval(grading, cells):
    if isinstance(grading, pm.MultiGrading) and cells < 3:
        pytest.skip("too few cells for three segments")
    w = grading.weights(cells, length=1.0)
    assert w.dtype == torch.float64 and len(w) == cells + 1
    assert float(w[0]) == 0.0 and float(w[-1]) == 1.0
    assert bool((w.diff() > 0).all())


@pytest.mark.parametrize("grading", ALL, ids=repr)
def test_reversed_mirrors(grading):
    w = grading.weights(12)
    r = grading.reversed().weights(12)
    assert torch.allclose(r, (1 - w).flip(0), atol=1e-12)


def test_simple_is_openfoam_expansion_ratio():
    sizes = pm.Simple(4.0).weights(10).diff()
    assert float(sizes[-1] / sizes[0]) == pytest.approx(4.0)
    assert pm.as_grading(4.0) == pm.Simple(4.0)
    assert isinstance(pm.as_grading(None), pm.Uniform)


def test_symmetric_and_multigrading():
    sizes = pm.Symmetric(8.0).weights(20).diff()
    assert float(sizes[9] / sizes[0]) == pytest.approx(8.0)
    assert torch.allclose(sizes, sizes.flip(0))
    # blockMesh: ((0.5 0.5 8) (0.5 0.5 0.125)) is the same distribution
    multi = pm.MultiGrading([(0.5, 0.5, 8.0), (0.5, 0.5, 0.125)]).weights(20)
    assert torch.allclose(multi, pm.Symmetric(8.0).weights(20))
    # segment lengths and cell shares
    w = pm.MultiGrading([(0.2, 0.3, 1.0), (0.8, 0.7, 1.0)]).weights(10)
    assert float(w[3]) == pytest.approx(0.2)


def test_first_cell_size():
    w = pm.FirstCell(0.002).weights(40, length=2.0)
    assert float(w[1] * 2.0) == pytest.approx(0.002, rel=1e-9)
    both = pm.FirstCell(0.002, C.BOTH).weights(40, length=2.0)
    assert float(both[1] * 2.0) == pytest.approx(0.002, rel=1e-9)
    assert float((1 - both[-2]) * 2.0) == pytest.approx(0.002, rel=1e-9)
    with pytest.raises(ValueError):
        pm.FirstCell(3.0).weights(10, length=2.0)


def test_explicit_and_cells_for_size():
    g = pm.Explicit([0.0, 0.1, 0.5, 1.0])
    assert g.cells == 3
    with pytest.raises(ValueError):
        g.weights(4)
    with pytest.raises(ValueError):
        pm.Explicit([0.0, 0.6, 0.5, 1.0])
    assert pm.cells_for_size(2.0, 0.1) == 20


@pytest.mark.parametrize("res", [1, 2, 7, 16, 31])
@pytest.mark.parametrize("ref", ["START", "END", "BOTH"])
def test_matches_legacy(res, ref):
    """The same distributions as PICT's make_weights_* functions."""
    cluster = C(ref)
    for base in (0.95, 1.05, 1.2):
        legacy = LEGACY[f"exp_{res}_{ref}_{base}"]
        np.testing.assert_allclose(
            pm.Geometric(base, cluster).weights(res), legacy, atol=1e-14
        )
    for kind, g in (
        ("simple", 4.0),
        ("simple", 0.3),
        ("tanh", 2.0),
        ("chebyshev_identity", 0.7),
    ):
        key = f"{kind}_{res}_{ref}_{g}"
        if key not in LEGACY:
            continue  # the legacy simple grading fails for one-cell parts
        if kind == "simple":
            new = {
                C.START: pm.Simple(g),
                C.END: pm.Simple(1 / g),
                C.BOTH: pm.Symmetric(g),
            }[cluster]
        elif kind == "tanh":
            new = pm.Tanh(g, cluster)
        else:
            new = pm.ChebyshevBlend(g, cluster)
        np.testing.assert_allclose(
            new.weights(res), LEGACY[key], atol=1e-13, err_msg=key
        )
    if res > 1:
        np.testing.assert_allclose(
            pm.Cosine(cluster).weights(res), LEGACY[f"cos_{res}_{ref}"], atol=1e-14
        )
