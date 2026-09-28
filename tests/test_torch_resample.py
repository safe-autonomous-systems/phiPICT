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

"""Tests for the differentiable pure-torch multi-block resampling.

The compiled ``_C.SampleTransformedGridLocalToGlobalMulti`` kernel is a
raw pybind binding with no autograd wrapper, so it detaches every resampled
observation from the simulation graph.
``sample_multi_coords_to_uniform_grid_diff`` reimplements it in torch, and
``DiffMultiblockResampler`` adds cached geometry plus optional restriction to
the output cells an observation actually reads.

These tests cover
* the multi-block, curvilinear cylinder environment, and
* a synthetic two-block grid with a gap, which forces many
  hole-filling iterations and so exercises the restriction machinery's
  propagation back through the fill steps.
"""

import pytest
import torch
from conftest import import_fluidgym_on_phipict

from phipict import _C
from phipict.grid.resample import (
    DiffMultiblockResampler,
    get_uniform_transform,
    sample_multi_coords_to_uniform_grid,
    sample_multi_coords_to_uniform_grid_diff,
)
from phipict.grid.shapes import coords_to_center_coords

ENV_ID = "CylinderJet2D-easy-v0"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="The compiled resampling kernel requires CUDA.",
)


@pytest.fixture(scope="module")
def cylinder_env():
    fluidgym = import_fluidgym_on_phipict("fluidgym")
    env = fluidgym.make(ENV_ID, differentiable=True)
    env.reset(seed=42)
    yield env


def _kernel_resample(coords_list, data_list, out_shape, fill_max_steps):
    """Reference resampling via the compiled kernel, mirroring the obs pipeline."""
    dims = len(coords_list[0].size()) - 2
    out_shape = torch.tensor(list(out_shape), dtype=torch.int32)

    cell_coords = [coords_to_center_coords(v) for v in coords_list]
    all_vertices = torch.cat([v.view(dims, -1) for v in coords_list], dim=-1)
    mat = get_uniform_transform(
        "AABB_OUTER", all_vertices, out_shape, dims, dtype=data_list[0].dtype
    )
    out, _ = _C.SampleTransformedGridLocalToGlobalMulti(
        data_list, cell_coords, mat, out_shape, fillMaxSteps=fill_max_steps
    )
    return out


def _env_case(env):
    """(coords, data, out_shape) for the environment's own resampling setup."""
    coords = env._sim.output_resampling_coords
    data = [b.velocity.detach().clone() for b in env._domain.getBlocks()]
    return coords, data, list(env._sim.output_resampling_shape)


# ---------------------------------------------------------------------------
# Synthetic case: two blocks with a gap, resampled onto a much finer grid, so
# most of the output starts empty and the hole filling runs for many steps.
# ---------------------------------------------------------------------------

SYNTHETIC_OUT_SHAPE = [90, 40]  # (x, y)


def _make_block(x0, x1, y0, y1, nx, ny, warp, dtype, device):
    xs = torch.linspace(x0, x1, nx + 1, dtype=dtype)
    ys = torch.linspace(y0, y1, ny + 1, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gx = gx + warp * torch.sin(gy)
    return torch.stack([gx, gy]).unsqueeze(0).to(device)  # [1, 2, ny+1, nx+1]


@pytest.fixture(scope="module")
def sparse_case():
    dtype, device = torch.float64, "cuda"
    coords = [
        _make_block(0.0, 1.0, 0.0, 1.0, 6, 5, 0.05, dtype, device),
        _make_block(1.9, 3.0, 0.0, 1.0, 6, 5, -0.05, dtype, device),
    ]
    generator = torch.Generator(device=device).manual_seed(0)
    data = [
        torch.randn(1, 2, 5, 6, dtype=dtype, device=device, generator=generator),
        torch.randn(1, 2, 5, 6, dtype=dtype, device=device, generator=generator),
    ]
    return coords, data, SYNTHETIC_OUT_SHAPE


# ---------------------------------------------------------------------------
# Synthetic 3D case. Every other fixture here is 2D, and the compiled kernel's
# corner loop read `DIMS<<1` -- which is 4 in 2D (accidentally right) and 6 in 3D
# (two of the eight trilinear corners silently dropped). Nothing in this file
# could see that until there was a 3D fixture, and the defect reached production
# 3D observations. The grid below is deliberately *stretched*, like the MHD duct,
# so cells are clustered near the walls and the splat lands at non-trivial
# fractional positions in every axis.
# ---------------------------------------------------------------------------

SYNTHETIC_OUT_SHAPE_3D = [20, 14, 9]  # (x, y, z)


def _make_block_3d(nx, ny, nz, dtype, device):
    """One stretched curvilinear 3D block, [1, 3, nz+1, ny+1, nx+1]."""
    xs = torch.linspace(0.0, 2.0, nx + 1, dtype=dtype)
    # tanh clustering toward both walls, as in a duct
    t = torch.linspace(-1.0, 1.0, ny + 1, dtype=dtype)
    ys = torch.tanh(2.0 * t) / torch.tanh(torch.tensor(2.0, dtype=dtype))
    u = torch.linspace(-1.0, 1.0, nz + 1, dtype=dtype)
    zs = 0.5 * torch.tanh(1.5 * u) / torch.tanh(torch.tensor(1.5, dtype=dtype))
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
    gx = gx + 0.03 * torch.sin(3.0 * gy) * torch.cos(2.0 * gz)
    return torch.stack([gx, gy, gz]).unsqueeze(0).to(device)


@pytest.fixture(scope="module")
def case_3d():
    dtype, device = torch.float64, "cuda"
    nx, ny, nz = 24, 16, 10
    coords = [_make_block_3d(nx, ny, nz, dtype, device)]
    generator = torch.Generator(device=device).manual_seed(0)
    data = [
        torch.randn(1, 2, nz, ny, nx, dtype=dtype, device=device, generator=generator)
    ]
    return coords, data, SYNTHETIC_OUT_SHAPE_3D


def _splat_weights(coords_list, out_shape, dtype, n_corners):
    """Accumulated splat weight per output cell, using ``n_corners`` corners.

    A trilinear splat gives every source cell total weight 1, distributed over
    ``2**dims`` corners, so the accumulated weight sums to the number of source
    cells minus whatever falls outside the grid. Using fewer corners loses weight
    in a way that is independent of the data -- which is what makes this the
    sharpest possible probe of the corner loop.
    """
    dims = len(coords_list[0].size()) - 2
    device = coords_list[0].device
    out_shape_t = torch.tensor(list(out_shape), dtype=torch.int32)
    all_vertices = torch.cat([v.view(dims, -1) for v in coords_list], dim=-1)
    mat = get_uniform_transform("AABB_OUTER", all_vertices, out_shape_t, dims, dtype)
    inv = torch.inverse(mat[0].to(device=device, dtype=torch.float64))

    out_spatial = [int(out_shape[dims - 1 - d]) for d in range(dims)]
    axis_stride, stride = [0] * dims, 1
    for d in range(dims):
        axis_stride[d] = stride
        stride *= out_spatial[dims - 1 - d]

    wacc = torch.zeros((stride,), device=device, dtype=dtype)
    for coords in coords_list:
        centres = coords_to_center_coords(coords).reshape(dims, -1).to(torch.float64)
        ones = torch.ones((1, centres.size(1)), device=device, dtype=torch.float64)
        gi = (inv @ torch.cat([centres, ones], dim=0))[:dims]
        base = torch.floor(gi).to(torch.int64)
        frac = gi - base
        for corner in range(n_corners):
            idx = torch.zeros_like(base[0])
            weight = torch.ones_like(gi[0])
            valid = torch.ones_like(gi[0], dtype=torch.bool)
            for axis in range(dims):
                offset = (corner >> axis) & 1
                coord = base[axis] + offset
                weight = weight * (frac[axis] if offset else (1.0 - frac[axis]))
                valid = valid & (coord >= 0) & (coord < out_spatial[dims - 1 - axis])
                idx = idx + coord * axis_stride[axis]
            wacc.index_add_(0, idx[valid], weight[valid].to(dtype))
    return wacc


def _kernel_weights(coords_list, data_list, out_shape):
    """The compiled kernel's own accumulated weights (its second return value)."""
    dims = len(coords_list[0].size()) - 2
    out_shape_t = torch.tensor(list(out_shape), dtype=torch.int32)
    cell_coords = [coords_to_center_coords(v) for v in coords_list]
    all_vertices = torch.cat([v.view(dims, -1) for v in coords_list], dim=-1)
    mat = get_uniform_transform(
        "AABB_OUTER", all_vertices, out_shape_t, dims, dtype=data_list[0].dtype
    )
    _out, weights = _C.SampleTransformedGridLocalToGlobalMulti(
        data_list, cell_coords, mat, out_shape_t, fillMaxSteps=0
    )
    return weights.reshape(-1)


def test_kernel_uses_all_2n_corners_in_3d(case_3d):
    """The kernel must splat to all 8 corners in 3D, not 6.

    `resampling.cu` looped `idx < (DIMS<<1)`, i.e. DIMS*2 -- correct in 2D by
    coincidence (4 == 4), wrong in 3D (6 != 8), dropping corners 110 and 111.
    Weights are data-independent, so comparing them isolates the corner loop from
    everything else. Before the fix this failed with the kernel at exactly 0.75 of
    the correct total weight, the expected value of the two missing corners.
    """
    coords, data, out_shape = case_3d
    kernel = _kernel_weights(coords, data, out_shape)
    full = _splat_weights(coords, out_shape, data[0].dtype, 1 << 3)
    truncated = _splat_weights(coords, out_shape, data[0].dtype, 3 << 1)

    # The premise: 6 and 8 corners must actually differ on this fixture, or the
    # test proves nothing.
    assert not torch.allclose(full, truncated, atol=1e-8), (
        "fixture does not discriminate 6 vs 8 corners; make the grid less aligned"
    )
    torch.testing.assert_close(kernel, full, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("fill_max_steps", [0, 8])
def test_reference_matches_compiled_kernel_3d(case_3d, fill_max_steps):
    """The 2D equivalent of this test existed; the 3D gap is what hid the bug."""
    coords, data, out_shape = case_3d
    reference = _kernel_resample(coords, data, out_shape, fill_max_steps)
    result = sample_multi_coords_to_uniform_grid_diff(
        data, coords, out_shape, is_cell_coords=False, fill_max_steps=fill_max_steps
    )
    assert result.shape == reference.shape
    torch.testing.assert_close(result, reference, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("fill_max_steps", [0, 8])
def test_cached_matches_compiled_kernel_3d(case_3d, fill_max_steps):
    coords, data, out_shape = case_3d
    reference = _kernel_resample(coords, data, out_shape, fill_max_steps)
    resampler = DiffMultiblockResampler(
        coords, out_shape, data[0].dtype, fill_max_steps=fill_max_steps
    )
    torch.testing.assert_close(resampler(data), reference, atol=1e-10, rtol=1e-10)


def test_restricted_matches_kernel_3d(case_3d):
    """Per-axis (x, y, z) indices, the form `_get_sensor_locations` returns."""
    coords, data, out_shape = case_3d
    reference = _kernel_resample(coords, data, out_shape, 8)[0]
    generator = torch.Generator().manual_seed(3)
    picks = torch.stack(
        [torch.randint(0, out_shape[d], (24,), generator=generator) for d in range(3)]
    ).to(coords[0].device)
    resampler = DiffMultiblockResampler(
        coords, out_shape, data[0].dtype, fill_max_steps=8, out_indices=picks
    )
    got = resampler(data)[0]
    sx, sy, sz = (p.long() for p in picks)
    torch.testing.assert_close(got, reference[:, sz, sy, sx], atol=1e-10, rtol=1e-10)


def test_cylinder_is_multi_block(cylinder_env):
    # The whole point of the "Multi" kernel: several curvilinear blocks resampled
    # onto one uniform grid. Guard the assumption the other tests rely on.
    assert len(cylinder_env._domain.getBlocks()) > 1


def test_sparse_case_forces_many_fill_steps(sparse_case):
    # Guard the premise of the restriction tests below: without real holes the
    # backward propagation through the fill steps is never exercised.
    coords, _, out_shape = sparse_case
    resampler = DiffMultiblockResampler(
        coords, out_shape, torch.float64, fill_max_steps=32
    )
    assert len(resampler._fill_steps) > 5
    assert float(resampler._written.double().mean()) < 0.2


# ---------------------------------------------------------------------------
# The reference torch implementation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fill_max_steps", [0, 16])
def test_reference_matches_compiled_kernel(cylinder_env, fill_max_steps):
    coords, data, out_shape = _env_case(cylinder_env)

    reference = _kernel_resample(coords, data, out_shape, fill_max_steps)
    result = sample_multi_coords_to_uniform_grid_diff(
        data, coords, out_shape, is_cell_coords=False, fill_max_steps=fill_max_steps
    )

    assert result.shape == reference.shape
    # float32 splat + fill accumulation: agreement is at rounding level.
    torch.testing.assert_close(result, reference, atol=1e-3, rtol=1e-3)


def test_reference_matches_observation_pipeline(cylinder_env):
    # The velocity observation is exactly this resample at the pipeline's fill.
    env = cylinder_env
    coords, data, out_shape = _env_case(env)
    fill = env._sim.output_resampling_fill_max_steps

    reference = _kernel_resample(coords, data, out_shape, fill)
    result = sample_multi_coords_to_uniform_grid_diff(
        data, coords, out_shape, fill_max_steps=fill
    )
    torch.testing.assert_close(result, reference, atol=1e-3, rtol=1e-3)


def test_reference_is_differentiable(cylinder_env):
    # The compiled kernel detaches; the torch version must keep a graph so that
    # gradients of a resampled observation w.r.t. the velocity field exist.
    env = cylinder_env
    coords, data, out_shape = _env_case(env)
    leaves = [d.clone().requires_grad_(True) for d in data]

    out = sample_multi_coords_to_uniform_grid_diff(
        leaves,
        coords,
        out_shape,
        fill_max_steps=env._sim.output_resampling_fill_max_steps,
    )
    assert out.grad_fn is not None

    grads = torch.autograd.grad(out.sum(), leaves)
    # Every block feeds the uniform grid, so each must receive gradient.
    for g in grads:
        assert g is not None
        assert torch.count_nonzero(g) > 0


def test_compiled_kernel_detaches(cylinder_env):
    # The motivation for all of the above: pin the fact that the compiled path
    # breaks the graph, so a regression here is visible.
    coords, data, out_shape = _env_case(cylinder_env)
    leaves = [d.clone().requires_grad_(True) for d in data]

    out = sample_multi_coords_to_uniform_grid(leaves, coords, out_shape)
    assert out.grad_fn is None
    assert not out.requires_grad


# ---------------------------------------------------------------------------
# DiffMultiblockResampler: cached geometry (full grid)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fill_max_steps", [0, 16])
def test_cached_matches_compiled_kernel(cylinder_env, fill_max_steps):
    coords, data, out_shape = _env_case(cylinder_env)

    reference = _kernel_resample(coords, data, out_shape, fill_max_steps)
    resampler = DiffMultiblockResampler(
        coords, out_shape, data[0].dtype, fill_max_steps=fill_max_steps
    )
    result = resampler(data)

    assert result.shape == reference.shape
    torch.testing.assert_close(result, reference, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("fill_max_steps", [0, 8, 32])
def test_cached_matches_compiled_kernel_with_holes(sparse_case, fill_max_steps):
    coords, data, out_shape = sparse_case

    reference = _kernel_resample(coords, data, out_shape, fill_max_steps)
    resampler = DiffMultiblockResampler(
        coords, out_shape, data[0].dtype, fill_max_steps=fill_max_steps
    )
    torch.testing.assert_close(resampler(data), reference, atol=1e-10, rtol=1e-10)


def test_cached_is_reusable_across_calls(sparse_case):
    # The point of caching: the same operator, applied to different data and to
    # different channel counts, keeps giving the kernel's answer.
    coords, data, out_shape = sparse_case
    resampler = DiffMultiblockResampler(
        coords, out_shape, data[0].dtype, fill_max_steps=16
    )
    for scale in (1.0, -3.5, 0.25):
        scaled = [d * scale for d in data]
        reference = _kernel_resample(coords, scaled, out_shape, 16)
        torch.testing.assert_close(resampler(scaled), reference, atol=1e-10, rtol=1e-10)

    single = [d[:, :1] for d in data]
    reference = _kernel_resample(coords, single, out_shape, 16)
    torch.testing.assert_close(resampler(single), reference, atol=1e-10, rtol=1e-10)


def test_cached_gradient_matches_reference(cylinder_env):
    # Same operator, so the same gradient as the reference implementation.
    env = cylinder_env
    coords, data, out_shape = _env_case(env)
    fill = env._sim.output_resampling_fill_max_steps

    ref_leaves = [d.clone().requires_grad_(True) for d in data]
    cached_leaves = [d.clone().requires_grad_(True) for d in data]

    out_ref = sample_multi_coords_to_uniform_grid_diff(
        ref_leaves, coords, out_shape, fill_max_steps=fill
    )
    generator = torch.Generator(device=out_ref.device).manual_seed(0)
    weight = torch.randn(
        out_ref.shape, dtype=out_ref.dtype, device=out_ref.device, generator=generator
    )

    grad_ref = torch.autograd.grad((out_ref * weight).sum(), ref_leaves)

    resampler = DiffMultiblockResampler(
        coords, out_shape, data[0].dtype, fill_max_steps=fill
    )
    grad_cached = torch.autograd.grad(
        (resampler(cached_leaves) * weight).sum(), cached_leaves
    )

    for a, b in zip(grad_ref, grad_cached, strict=False):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# DiffMultiblockResampler: restriction to selected output cells
# ---------------------------------------------------------------------------


def _random_out_indices(out_shape, count, device, seed=0):
    """Random per-axis (x, y) output indices, in the layout the env sensors use."""
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.stack(
        [
            torch.randint(0, n, (count,), device=device, generator=generator)
            for n in out_shape
        ]
    ).to(torch.int32)


@pytest.mark.parametrize("fill_max_steps", [0, 16])
def test_restricted_matches_kernel(cylinder_env, fill_max_steps):
    coords, data, out_shape = _env_case(cylinder_env)
    indices = _random_out_indices(out_shape, 128, data[0].device)

    reference = _kernel_resample(coords, data, out_shape, fill_max_steps)
    expected = reference[0][:, indices[1].long(), indices[0].long()].unsqueeze(0)

    resampler = DiffMultiblockResampler(
        coords,
        out_shape,
        data[0].dtype,
        fill_max_steps=fill_max_steps,
        out_indices=indices,
    )
    result = resampler(data)

    assert result.shape == expected.shape
    torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("fill_max_steps", [0, 2, 8, 32])
def test_restricted_matches_kernel_with_holes(sparse_case, fill_max_steps):
    # With most of the grid empty, the selected cells are reached only after
    # several fill iterations, so this covers the backward propagation of the
    # row selection through the fill operators.
    coords, data, out_shape = sparse_case
    indices = _random_out_indices(out_shape, 64, data[0].device)

    reference = _kernel_resample(coords, data, out_shape, fill_max_steps)
    expected = reference[0][:, indices[1].long(), indices[0].long()].unsqueeze(0)

    resampler = DiffMultiblockResampler(
        coords,
        out_shape,
        data[0].dtype,
        fill_max_steps=fill_max_steps,
        out_indices=indices,
    )
    torch.testing.assert_close(resampler(data), expected, atol=1e-10, rtol=1e-10)


def test_restricted_accepts_flat_indices(sparse_case):
    # Flat indices into the flattened output grid must select the same cells as
    # the per-axis form.
    coords, data, out_shape = sparse_case
    indices = _random_out_indices(out_shape, 32, data[0].device)
    # out_spatial is out_shape reversed, so the flat index is y * nx + x.
    flat = indices[1].long() * out_shape[0] + indices[0].long()

    by_axis = DiffMultiblockResampler(
        coords, out_shape, data[0].dtype, fill_max_steps=8, out_indices=indices
    )
    by_flat = DiffMultiblockResampler(
        coords, out_shape, data[0].dtype, fill_max_steps=8, out_indices=flat
    )
    torch.testing.assert_close(by_axis(data), by_flat(data))


def test_restricted_gradient_matches_full(sparse_case):
    # Restricting must not change the gradient of the selected outputs.
    coords, data, out_shape = sparse_case
    indices = _random_out_indices(out_shape, 64, data[0].device)
    fill = 32

    full = DiffMultiblockResampler(
        coords, out_shape, data[0].dtype, fill_max_steps=fill
    )
    restricted = DiffMultiblockResampler(
        coords, out_shape, data[0].dtype, fill_max_steps=fill, out_indices=indices
    )

    full_leaves = [d.clone().requires_grad_(True) for d in data]
    restricted_leaves = [d.clone().requires_grad_(True) for d in data]

    generator = torch.Generator(device=data[0].device).manual_seed(1)
    weight = torch.randn(
        (1, data[0].size(1), indices.size(1)),
        dtype=data[0].dtype,
        device=data[0].device,
        generator=generator,
    )

    selected = full(full_leaves)[0][:, indices[1].long(), indices[0].long()]
    grad_full = torch.autograd.grad((selected.unsqueeze(0) * weight).sum(), full_leaves)
    grad_restricted = torch.autograd.grad(
        (restricted(restricted_leaves) * weight).sum(), restricted_leaves
    )

    for a, b in zip(grad_full, grad_restricted, strict=False):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)


def test_restricted_operator_is_small(cylinder_env):
    # The reason to restrict at all: for a handful of sensors the operator must
    # be orders of magnitude smaller than the full-grid one, and the cached fill
    # masks must be gone. Measured on the environment's real grid, where the
    # sensor count is genuinely small against the output resolution.
    coords, data, out_shape = _env_case(cylinder_env)
    fill = cylinder_env._sim.output_resampling_fill_max_steps
    indices = _random_out_indices(out_shape, 32, data[0].device)

    full = DiffMultiblockResampler(
        coords, out_shape, data[0].dtype, fill_max_steps=fill
    )
    restricted = DiffMultiblockResampler(
        coords, out_shape, data[0].dtype, fill_max_steps=fill, out_indices=indices
    )

    assert restricted._rows.numel() < full._rows.numel() / 100
    assert restricted._fill_steps == []


def test_restricted_gradient_matches_finite_difference(sparse_case):
    # Splat, weight normalisation and neighbour-mean fill are all linear in the
    # cell values, so the analytic gradient must match a finite-difference probe.
    coords, data, out_shape = sparse_case
    indices = _random_out_indices(out_shape, 16, data[0].device)
    resampler = DiffMultiblockResampler(
        coords, out_shape, data[0].dtype, fill_max_steps=16, out_indices=indices
    )

    generator = torch.Generator(device=data[0].device).manual_seed(2)
    weight = torch.randn(
        (1, data[0].size(1), indices.size(1)),
        dtype=data[0].dtype,
        device=data[0].device,
        generator=generator,
    )

    def readout(block0):
        return (resampler([block0] + data[1:]) * weight).sum()

    leaves = [d.clone().requires_grad_(True) for d in data]
    (analytic,) = torch.autograd.grad((resampler(leaves) * weight).sum(), leaves[0])

    flat = data[0].reshape(-1)
    probe = torch.linspace(0, flat.numel() - 1, steps=5).long()
    eps = 1e-4
    for p in probe:
        perturbed = flat.clone()
        perturbed[p] += eps
        plus = readout(perturbed.reshape(data[0].shape))
        perturbed[p] -= 2 * eps
        minus = readout(perturbed.reshape(data[0].shape))
        finite_difference = ((plus - minus) / (2 * eps)).item()
        torch.testing.assert_close(
            analytic.reshape(-1)[p].item(), finite_difference, atol=1e-6, rtol=1e-6
        )
