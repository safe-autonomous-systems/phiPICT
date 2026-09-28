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
# Elliptical distance scaling in rotate_grid; removed make_cosX_grid;
# grading schemes (make_weights_simple_grading/_chebyshev_identity/_tanh).
# Moved into the phipict package, formatted, linted and typed.

"""Generation and manipulation of structured grid coordinates."""

import math
from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

import numpy as np
import scipy.stats as stats
import torch
from scipy.spatial.transform import Rotation


def get_grid_coords(
    size: Sequence[int], dtype: torch.dtype = torch.int32
) -> torch.Tensor:
    """Build a grid of integer cell indices.

    Parameters
    ----------
    size : Sequence of int
        Grid size per axis.
    dtype : torch.dtype, optional
        Dtype of the coordinates. Default is ``torch.int32``.

    Returns
    -------
    torch.Tensor
        Index coordinates in CDHW layout.
    """
    axes = []
    for dim in size:
        axes.append(torch.range(0, dim - 1, dtype=dtype))

    grids = torch.meshgrid(*axes, indexing="xy")
    coords = torch.stack(grids)

    return coords  # CDHW


def get_grid_striped_x(
    size: Sequence[int], dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Build a grid with alternating 0/1 values along the first axis.

    Parameters
    ----------
    size : Sequence of int
        Grid size per axis.
    dtype : torch.dtype, optional
        Dtype of the grid. Default is ``torch.float32``.

    Returns
    -------
    torch.Tensor
        Striped grid.
    """
    coords = get_grid_coords(size, dtype)

    # coords_x = coords[0]
    # data = coords_x%2

    return coords[0] % 2


def make_matrix_rotation_2D(angle: float, degrees: bool = True) -> np.ndarray:
    """Build a 2D rotation matrix.

    Parameters
    ----------
    angle : float
        Counterclockwise rotation angle.
    degrees : bool, optional
        Whether ``angle`` is in degrees (True) or radians. Default is True.

    Returns
    -------
    numpy.ndarray
        Rotation matrix of shape ``[2, 2]``.
    """
    if degrees:
        angle = np.deg2rad(angle)
    return np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )


def make_matrix_rotation_3D(rotvec: np.ndarray, degrees: bool = True) -> np.ndarray:
    """Build a 3D rotation matrix from a rotation vector.

    Parameters
    ----------
    rotvec : numpy.ndarray
        Rotation vector (axis scaled by angle).
    degrees : bool, optional
        Whether the angle is in degrees (True) or radians. Default is True.

    Returns
    -------
    numpy.ndarray
        Rotation matrix of shape ``[3, 3]``.
    """
    return Rotation.from_rotvec(rotvec, degrees=degrees).as_matrix()


def make_rotation_distance_scaling_fn_sine(
    angle: float, r_start: float, r_end: float
) -> Callable[[float], float]:
    """Build a sine-shaped rotation weighting for :func:`rotate_grid`.

    Points closer than ``r_start`` or further than ``r_end`` from the center are
    not rotated; in between, the displacement follows a sine bump.

    Parameters
    ----------
    angle : float
        Rotation angle in degrees.
    r_start : float
        Inner radius, must be positive.
    r_end : float
        Outer radius.

    Returns
    -------
    Callable[[float], float]
        Function mapping a distance to a rotation scaling factor.

    Raises
    ------
    RuntimeError
        If ``r_start`` is not positive.
    """
    if not r_start > 0:
        raise RuntimeError("r_start must be positive.")
    r_half = (r_start + r_end) * 0.5
    rad = np.deg2rad(angle)

    def displacement(dist: float) -> float:
        return 1 - (np.cos((dist - r_start) / (r_end - r_start) * 2 * np.pi) + 1) * 0.5

    factor = rad * r_half / displacement(r_half)

    def distance_scaling(dist: float) -> float:
        if dist < r_start or r_end < dist:
            return 0
        else:
            return factor * displacement(dist) / (dist * rad)

    return distance_scaling


def make_rotation_distance_scaling_fn_sine_half(
    angle: float, r_end: float
) -> Callable[[float], float]:
    """Build a half-cosine rotation weighting for :func:`rotate_grid`.

    The weight is 1 at the center and decays to 0 at ``r_end``.

    Parameters
    ----------
    angle : float
        Unused.
    r_end : float
        Radius beyond which points are not rotated.

    Returns
    -------
    Callable[[float], float]
        Function mapping a distance to a rotation scaling factor.
    """

    def distance_scaling(dist: float) -> float:
        if r_end < dist:
            return 0
        else:
            return (np.cos(dist / r_end * np.pi) + 1) * 0.5

    return distance_scaling


def rotate_grid(
    grid: torch.Tensor,
    angle: float,
    axis: Sequence[float] | np.ndarray | None = None,
    center: str | Sequence[float] | torch.Tensor = "CENTER",
    distance_scaling: Callable[..., Any] | None = None,
    distance_axes: Sequence[float] | None = None,
) -> torch.Tensor:
    """Rotate grid coordinates.

    Parameters
    ----------
    grid : torch.Tensor
        Coordinates of shape NCHW or NCDHW with C = dims.
    angle : float
        Rotation angle in degrees.
    axis : Sequence of float, numpy.ndarray or None, optional
        Rotation axis, required in 3D. Default is None.
    center : str, Sequence of float or torch.Tensor, optional
        Rotation center: a coordinate, ``"ORIGIN"`` or ``"CENTER"`` (center of
        the bounding box). Default is ``"CENTER"``.
    distance_scaling : Callable or None, optional
        Function returning a factor for the rotation angle, depending on a
        point's distance to the center. Default is None (rigid rotation).
    distance_axes : Sequence of float or None, optional
        Per-axis radii ``[r_x, r_y(, r_z)]`` for an elliptical distance: each
        coordinate is divided by its radius before computing the norm, so the
        influence region stretches further along axes with larger radii. Only
        used with ``distance_scaling``. Default is None.

    Returns
    -------
    torch.Tensor
        Rotated coordinates with the same shape as ``grid``.

    Raises
    ------
    ValueError
        If the 3D rotation axis is too short to be normalized.
    """
    assert (
        isinstance(grid, torch.Tensor)
        and (grid.dim() in [4, 5])
        and grid.size(1) == (grid.dim() - 2)
    ), "grid must be a torch tensor with shape NCHW or NCDHW"
    dims = grid.dim() - 2
    axis_vec = np.zeros(3, dtype=np.float32)  # only used in 3D
    if dims == 3:
        assert isinstance(axis, (list, tuple)) and len(axis) == 3, (
            "3D rotation axis is required"
        )
        axis_vec = np.asarray(axis, dtype=np.float32)
        axis_norm = np.linalg.norm(axis_vec)
        if axis_norm < 1e-5:
            raise ValueError("rotation axis is too short for normalization")
        axis_vec /= axis_norm

    if distance_axes is not None:
        assert (
            isinstance(distance_axes, (list, tuple)) and len(distance_axes) == dims
        ), f"distance_axes must have {dims} elements, one per spatial dimension"

    grid = torch.moveaxis(grid, 1, -1)
    grid_size = grid.size()
    grid = torch.reshape(grid, (-1, dims))

    center_coords: Any
    if center == "ORIGIN":
        center_coords = [0] * dims
    elif center == "CENTER":
        lower, _ = grid.min(dim=0)
        upper, _ = grid.max(dim=0)
        center_coords = [(l + u) * 0.5 for l, u in zip(lower, upper, strict=False)]  # noqa: E741
    else:
        center_coords = center

    assert isinstance(center_coords, (list, tuple)) and len(center_coords) == dims
    center_tensor = torch.tensor(center_coords, device=grid.device, dtype=grid.dtype)
    grid = grid - center_tensor  # now centered on origin

    if distance_scaling is not None:
        if distance_axes is not None:
            axes_tensor = torch.tensor(
                distance_axes, device=grid.device, dtype=grid.dtype
            )
            distances = (
                torch.linalg.norm(grid / axes_tensor, dim=-1, keepdims=False)
                .cpu()
                .numpy()
            )
        else:
            distances = torch.linalg.norm(grid, dim=-1, keepdims=False).cpu().numpy()
        angles = [angle * distance_scaling(distance) for distance in distances]

        if dims == 2:
            matrices = [make_matrix_rotation_2D(a) for a in angles]
        else:
            matrices = [make_matrix_rotation_3D(a * axis_vec) for a in angles]

        rotation_matrices = torch.tensor(
            np.asarray(matrices), device=grid.device, dtype=grid.dtype
        )

    else:
        if dims == 2:
            rotation_matrix_np = make_matrix_rotation_2D(angle)
        else:
            rotation_matrix_np = make_matrix_rotation_3D(angle * axis_vec)
        rotation_matrix = torch.tensor(
            rotation_matrix_np, device=grid.device, dtype=grid.dtype
        )
        rotation_matrices = rotation_matrix.reshape((1, dims, dims)).repeat(
            grid.size(0), 1, 1
        )

    grid = torch.reshape(grid, (-1, dims, 1))
    grid = torch.bmm(rotation_matrices, grid)
    grid = torch.reshape(grid, (-1, dims))

    grid = grid + center_tensor
    grid = torch.reshape(grid, grid_size)
    grid = torch.moveaxis(grid, -1, 1)

    return grid


def get_grid_cube_centered(
    size: Sequence[int], cube_size: Sequence[int], dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Build a grid that is 1 inside a centered box and 0 elsewhere.

    Parameters
    ----------
    size : Sequence of int
        Grid size per axis.
    cube_size : Sequence of int
        Box size per axis, at most ``size``.
    dtype : torch.dtype, optional
        Dtype of the grid. Default is ``torch.float32``.

    Returns
    -------
    torch.Tensor
        Box indicator grid.
    """
    axes = []
    for dim, cs in zip(size, cube_size, strict=False):
        assert cs <= dim
        border1 = (dim - cs) // 2
        border2 = dim - cs - border1
        x = [0] * border1 + [1] * cs + [0] * border2
        axes.append(torch.tensor(x, dtype=dtype))

    grids = torch.meshgrid(*axes)
    data = torch.prod(torch.stack(grids), dim=0)

    return data


def get_grid_normal_dist(
    size: Sequence[int],
    mean: Sequence[float],
    var: Sequence[float],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a grid with a separable normal distribution over ``[-1, 1]``.

    Parameters
    ----------
    size : Sequence of int
        Grid size per axis.
    mean : Sequence of float
        Mean per axis.
    var : Sequence of float
        Variance per axis.
    dtype : torch.dtype, optional
        Dtype of the grid. Default is ``torch.float32``.

    Returns
    -------
    torch.Tensor
        Product of the per-axis probability densities.
    """
    # https://stackoverflow.com/questions/10138085/how-to-plot-normal-distribution
    axes = []
    for dim, m, v in zip(size, mean, var, strict=False):
        s = math.sqrt(v)
        # x = np.linspace(m-3*s, m+3*s, dim)
        # grid borders -1 to 1, so cell centers have some offset
        cell_coord = (dim / 2 - 0.5) / (dim / 2)
        x = np.linspace(-cell_coord, cell_coord, dim)
        axes.append(torch.tensor(stats.norm.pdf(x, m, s), dtype=dtype))

    grids = torch.meshgrid(*axes)
    data = torch.prod(torch.stack(grids), dim=0)

    return data


def ortho_transform_to_coords(transform: torch.Tensor, dims: int) -> torch.Tensor:
    """Reconstruct vertex coordinates from orthogonal per-cell transforms.

    Parameters
    ----------
    transform : torch.Tensor
        Per-cell transforms in NDHWC layout with ``C = 2 * dims**2 + 1``.
    dims : int
        Number of spatial dimensions.

    Returns
    -------
    torch.Tensor
        Vertex coordinates in NCDHW layout, starting at the origin.
    """
    assert transform.shape[-1] == 2 * dims * dims + 1
    coord_list = []
    for dim in range(dims):  # x,y,z
        scales = transform[..., (dims + 1) * dim]
        # padding argument goes in inverse dimension order
        coord = torch.nn.functional.pad(
            torch.cumsum(scales, dim=-(dim + 1)), [0, 0] * dim + [1, 0], value=0
        )
        coord = torch.nn.functional.pad(
            coord, [1, 0] * dim + [0, 0] + [1, 0] * (dims - 1 - dim), mode="replicate"
        )
        coord_list.append(coord)
    coords = torch.stack(coord_list, dim=1)
    return coords  # NCDHW


def coords_to_center_coords(coords: torch.Tensor) -> torch.Tensor:
    """Compute cell-center coordinates by averaging the vertices of each cell.

    Parameters
    ----------
    coords : torch.Tensor
        Vertex coordinates in NCHW or NCDHW layout.

    Returns
    -------
    torch.Tensor
        Cell-center coordinates, one cell smaller per spatial axis.

    Raises
    ------
    ValueError
        If the grid is not 2D or 3D.
    """
    dims = coords.shape[1]
    pool: torch.nn.Module
    if dims == 2:
        pool = torch.nn.AvgPool2d(2, stride=1)
    elif dims == 3:
        pool = torch.nn.AvgPool3d(2, stride=1)
    else:
        raise ValueError()
    return pool(coords)


def get_grid_normal_dist_from_ortho_transforms(
    transforms: torch.Tensor,
    mean: Sequence[float],
    var: Sequence[float],
    dtype: torch.dtype = torch.float32,
    normalize: bool = True,
) -> torch.Tensor:
    """Evaluate a separable normal distribution on an orthogonal grid.

    Parameters
    ----------
    transforms : torch.Tensor
        Per-cell orthogonal transforms, see :func:`ortho_transform_to_coords`.
    mean : Sequence of float
        Mean per axis.
    var : Sequence of float
        Variance per axis.
    dtype : torch.dtype, optional
        Dtype of the result. Default is ``torch.float32``.
    normalize : bool, optional
        Whether to map the grid extent to ``[-1, 1]`` first. Default is True.

    Returns
    -------
    torch.Tensor
        Product of the per-axis probability densities in DHW layout.
    """
    dims = len(transforms.shape) - 2
    vertex_coords = ortho_transform_to_coords(transforms, dims)
    coords = coords_to_center_coords(vertex_coords)
    size = [transforms.shape[-(i + 1)] for i in range(dims)]
    axes = []
    for dim, (res, m, v) in enumerate(zip(size, mean, var, strict=False)):
        s = math.sqrt(v)
        # x = np.linspace(m-3*s, m+3*s, dim)
        slicing = tuple(
            [0, dim]
            + [slice(None) if dim == (dims - 1 - d) else 0 for d in range(dims)]
        )
        # print(slicing)
        x = coords[slicing].cpu().numpy()
        # print(x)
        if normalize:
            slicing = tuple([0, dim] + [0] * dims)
            dim_min = vertex_coords[slicing].cpu().numpy()
            slicing = tuple([0, dim] + [-1] * dims)
            dim_max = vertex_coords[slicing].cpu().numpy()
            # print(dim_min, dim_max)
            # x = x / x[-1] * 2 -1 # -> [-1,1]
            x = (x - dim_min) / (dim_max - dim_min) * 2 - 1
            # print(x)
        axes.append(torch.tensor(stats.norm.pdf(x, m, s), dtype=dtype))

    axes = axes[::-1]

    grids = torch.meshgrid(*axes)  # C-DHW
    data = torch.prod(torch.stack(grids), dim=0)  # DHW

    return data


def interpolate_vertices_from_borders_2D(
    borders: Sequence[Any],
    x_weights: Sequence[float] | None = None,
    y_weights: Sequence[float] | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Interpolate inner grid vertices from the four border vertex lists.

    Parameters
    ----------
    borders : Sequence
        Vertex lists of the ``[-x, +x, -y, +y]`` borders, each a sequence of
        ``(x, y)`` coordinates.
    x_weights : Sequence of float or None, optional
        Interpolation weights along the x-borders (i.e. along y). Default is
        None (uniform).
    y_weights : Sequence of float or None, optional
        Interpolation weights along the y-borders (i.e. along x). Default is
        None (uniform).
    dtype : torch.dtype, optional
        Dtype of the grid. Default is ``torch.float32``.

    Returns
    -------
    torch.Tensor
        Vertex coordinates of shape ``[1, 2, res_y, res_x]``.
    """
    assert len(borders) == 4, "only 2D for now"
    dims = 2
    res = [len(borders[0]), len(borders[2])]  # y,x
    assert len(borders[1]) == res[0]
    assert len(borders[3]) == res[1]

    borders = [np.asarray(border) for border in borders]

    grid = torch.zeros((1, dims, res[0], res[1]), dtype=dtype)

    if x_weights is None:
        x_weights = [i / (res[0] - 1) for i in range(res[0])]
    else:
        assert len(x_weights) == res[0]

    if y_weights is None:
        y_weights = [i / (res[1] - 1) for i in range(res[1])]
    else:
        assert len(y_weights) == res[1]

    for y_idx in range(res[0]):
        y_weight_upper = x_weights[y_idx]
        y_weight_lower = 1 - x_weights[y_idx]
        x_start = borders[2][0] * y_weight_lower + borders[3][0] * y_weight_upper
        x_end = borders[2][-1] * y_weight_lower + borders[3][-1] * y_weight_upper
        x_size = x_end - x_start

        target_size = borders[1][y_idx] - borders[0][y_idx]
        size_diff = target_size - x_size

        for x_idx in range(res[1]):
            x_val = (
                borders[2][x_idx] * y_weight_lower + borders[3][x_idx] * y_weight_upper
            )
            x_frac = x_idx / (res[1] - 1)  # y_weights[y_idx]
            if np.any(np.isclose(x_size, 0)):
                x_val = x_val - x_start + size_diff * x_frac + borders[0][y_idx]
            else:
                size_fac = target_size / x_size
                x_val = (x_val - x_start) * size_fac + borders[0][y_idx]

            grid[0, 0, y_idx, x_idx] = x_val[0]
            grid[0, 1, y_idx, x_idx] = x_val[1]

    return grid


def _check_weights(weights: Sequence[float], res: int, name: str = "weights") -> None:
    """Validate a sequence of interpolation weights.

    Parameters
    ----------
    weights : Sequence of float
        Weights to check. Must be strictly increasing, lie in ``[0, 1]``, and
        start at 0 and end at 1.
    res : int
        Resolution the weights must match in length.
    name : str, optional
        Name used in error messages. Default is ``"weights"``.

    Raises
    ------
    TypeError
        If ``weights`` is not a list or tuple of numbers.
    ValueError
        If the length, range, endpoints or monotonicity requirements are violated.
    """
    if not (len(weights) == res):
        raise ValueError("Invalid %s: length must match resolution." % (name,))
    if not (
        isinstance(weights, (list, tuple))
        and all(isinstance(w, (int, float)) for w in weights)
    ):
        raise TypeError("Invalid %s: weights must be a list of float." % (name,))
    if not all((0 - 1e-5) <= w and w <= (1 + 1e-5) for w in weights):
        raise ValueError("Invalid %s: weights must be in [0,1]: %s" % (name, weights))
    if not np.isclose(weights[0], 0):
        raise ValueError(
            "Invalid %s: start weight must be 0, is %s." % (name, weights[0])
        )
    if not np.isclose(weights[-1], 1):
        raise ValueError(
            "Invalid %s: end weight must be 1, is %s." % (name, weights[-1])
        )
    if not all(weights[i] < weights[i + 1] for i in range(len(weights) - 1)):
        raise ValueError("Invalid %s: weights must be strictly increasing." % (name,))


def invert_weights(weights: Sequence[float]) -> list[float]:
    """Mirror interpolation weights so the refinement is at the other end.

    Parameters
    ----------
    weights : Sequence of float
        Strictly increasing weights from 0 to 1.

    Returns
    -------
    list of float
        Inverted weights.
    """
    _check_weights(weights, len(weights))

    sizes = [weights[i + 1] - weights[i] for i in range(len(weights) - 1)]
    inv_weights: list[float] = [0]
    size = 0.0
    # inverse cumulative sum of the sizes
    for i in range(len(sizes) - 1, -1, -1):
        size = size + sizes[i]
        inv_weights.append(size)

    return inv_weights


def make_weights_linear(res: int) -> list[float]:
    """Create uniform interpolation weights.

    Parameters
    ----------
    res : int
        Number of cells.

    Returns
    -------
    list of float
        ``res + 1`` weights from 0 to 1.
    """
    return [x / (res) for x in range(res + 1)]


def make_weights_exp(res: int, base: float, refinement: str) -> list[float]:
    """Create interpolation weights with exponentially growing cell sizes.

    Parameters
    ----------
    res : int
        Number of cells.
    base : float
        Size ratio between neighboring cells.
    refinement : str
        ``"START"``, ``"END"`` or ``"BOTH"``: where the smallest cells are.
        Other values behave like ``"START"``.

    Returns
    -------
    list of float
        ``res + 1`` weights from 0 to 1.
    """
    exponents = list(range(res))
    if refinement == "END":
        exponents.reverse()
    elif refinement == "BOTH":
        exponents = exponents[: res // 2] + list(reversed(exponents))[res // 2 :]

    sizes = [base**e for e in exponents]
    total_size = np.sum(sizes)
    weights: list[float] = [0]
    weights += [w / total_size for w in np.cumsum(sizes)]

    return weights


def make_weights_exp_global(
    res: int,
    global_scale: float,
    refinement: str,
    log_fn: Callable[..., Any] | None = None,
) -> list[float]:
    """Create exponential weights from the ratio of the largest to smallest cell.

    Parameters
    ----------
    res : int
        Number of cells.
    global_scale : float
        Size ratio between the largest and the smallest cell.
    refinement : str
        See :func:`make_weights_exp`.
    log_fn : Callable or None, optional
        Logging function called with a printf-style message. Default is None.

    Returns
    -------
    list of float
        ``res + 1`` weights from 0 to 1.
    """
    resolution = res // 2 if refinement == "BOTH" else res
    base = global_scale ** (1 / (resolution - 1))

    if log_fn is not None:
        log_fn("r %d exp weights global %.03e -> base %.03e", res, global_scale, base)

    return make_weights_exp(res, base, refinement)


def make_weights_cos(res: int, refinement: str) -> list[float]:
    """Create interpolation weights with cosine spacing.

    Parameters
    ----------
    res : int
        Number of cells.
    refinement : {"START", "END", "BOTH"}
        Where the smallest cells are.

    Returns
    -------
    list of float
        ``res + 1`` weights from 0 to 1.

    Raises
    ------
    ValueError
        If ``refinement`` is unknown.
    """
    c_start: float
    c_end: float
    n_mul: float
    n_add: float
    if refinement == "START":
        c_start = 0
        c_end = np.pi / 2
        n_mul = -1
        n_add = 1
    elif refinement == "END":
        c_start = np.pi / 2
        c_end = np.pi
        n_mul = -1
        n_add = 0
    elif refinement == "BOTH":
        c_start = 0
        c_end = np.pi
        n_mul = -0.5
        n_add = 0.5
    else:
        raise ValueError("Unkown refinement side.")

    def c_val_lerp(t: float) -> float:
        return c_start * (1 - t) + c_end * t

    weights = [np.cos(c_val_lerp(x / res)) * n_mul + n_add for x in range(res + 1)]
    return weights


# Grading schemes for wall-refined structured grids


def make_weights_simple_grading(
    res: int, grading: float, refinement: Literal["START", "END", "BOTH"]
) -> list[float]:
    """Create cell weights using OpenFOAM-style simpleGrading cell expansion ratio.

    Parameters
    ----------
    res : int
        Number of cells.
    grading : float
        Cell expansion ratio (last cell size / first cell size).
        > 1: refinement at START (or walls for BOTH)
        < 1: refinement at END (or center for BOTH)
    refinement : {"START", "END", "BOTH"}
        "START" : first cell is smallest
        "END"   : last cell is smallest
        "BOTH"  : symmetric, smallest cells at both ends (walls)

    Returns
    -------
    list of float
        ``res + 1`` weights from 0 to 1.

    Raises
    ------
    ValueError
        If ``refinement`` is unknown.
    """

    def geometric_sizes(n: int, r: float) -> list[float]:
        """Generate n cell sizes with geometric progression and expansion ratio r.

        Parameters
        ----------
        n : int
            Number of cells.
        r : float
            Expansion ratio (last cell size / first cell size). For example, r=2
            means the last cell is twice as large as the first cell.

        Returns
        -------
        list of float
            List of n cell sizes summing to 1.
        """
        if abs(r - 1.0) < 1e-10:
            return [1.0 / n] * n

        # For geometric series: sizes[i] = a * q^i
        # r = sizes[n-1] / sizes[0] = q^(n-1)  =>  q = r^(1/(n-1))

        q = r ** (1.0 / (n - 1))
        a = 1.0 / sum(q**i for i in range(n))

        return [a * q**i for i in range(n)]

    if refinement == "START":
        # grading > 1 means last/first > 1, so first cell is smallest
        sizes = geometric_sizes(res, grading)

    elif refinement == "END":
        # reverse: last cell is smallest
        sizes = geometric_sizes(res, grading)[::-1]

    elif refinement == "BOTH":
        # Symmetric: each half has refinement toward the walls (ends of full block)
        # First half: small -> large (grading > 1 means wall-refined)
        half = res // 2
        left_sizes = geometric_sizes(half, grading)  # small at left wall
        right_sizes = geometric_sizes(res - half, grading)[::-1]  # small at right wall
        sizes = left_sizes + right_sizes

    else:
        raise ValueError(
            f"Unknown refinement '{refinement}'. Use 'START', 'END', or 'BOTH'."
        )

    # Normalize (already sums to 1 per half, but re-normalize for safety)
    total = sum(sizes)
    sizes = [s / total for s in sizes]

    weights: list[float] = [0.0] + list(np.cumsum(sizes))
    return weights


def make_weights_chebyshev_identity(
    res: int, gamma: float, refinement: Literal["START", "END", "BOTH"]
) -> list[float]:
    """Create cell weights by blending the Chebyshev and identity transforms.

    Parameters
    ----------
    res : int
        Number of cells.
    gamma : float
        Weighting factor for the Chebyshev transformation (0 <= gamma <= 1).
        gamma = 0 corresponds to a uniform grid (identity).
        gamma = 1 corresponds to a pure Chebyshev (sine) spacing.
    refinement : {"START", "END", "BOTH"}
        "START" : first cell is smallest
        "END"   : last cell is smallest
        "BOTH"  : symmetric, smallest cells at both ends (walls)

    Returns
    -------
    list of float
        ``res + 1`` weights from 0 to 1.

    Raises
    ------
    ValueError
        If ``gamma`` is out of range or ``refinement`` is unknown.
    """
    if gamma < 0 or gamma > 1:
        raise ValueError("Gamma must be in the range [0, 1].")

    if refinement == "START":
        # Fine resolution at -1, mapping [-1, 0] to weights [0, 1]
        xi = np.linspace(-1, 0, res + 1)
        weights = (gamma * np.sin(np.pi / 2 * xi) + (1 - gamma) * xi) + 1.0

    elif refinement == "END":
        # Fine resolution at 1, mapping [0, 1] to weights [0, 1]
        xi = np.linspace(0, 1, res + 1)
        weights = gamma * np.sin(np.pi / 2 * xi) + (1 - gamma) * xi

    elif refinement == "BOTH":
        # Fine resolution at -1 and 1, mapping [-1, 1] to weights [0, 1]
        xi = np.linspace(-1, 1, res + 1)
        weights = 0.5 * (gamma * np.sin(np.pi / 2 * xi) + (1 - gamma) * xi) + 0.5

    else:
        raise ValueError(
            f"Unknown refinement '{refinement}'. Use 'START', 'END', or 'BOTH'."
        )

    # Force exact 0.0 and 1.0 at bounds to avoid floating point inaccuracies
    weights[0] = 0.0
    weights[-1] = 1.0

    return weights.tolist()


def make_weights_tanh(
    res: int, A: float, refinement: Literal["START", "END", "BOTH"]
) -> list[float]:
    """Create cell weights using the classical tanh transformation.

    Parameters
    ----------
    res : int
        Number of cells.
    A : float
        Stretching parameter (A > 0). Higher values cluster cells closer to the
        walls.
    refinement : {"START", "END", "BOTH"}
        "START" : first cell is smallest
        "END"   : last cell is smallest
        "BOTH"  : symmetric, smallest cells at both ends (walls)

    Returns
    -------
    list of float
        ``res + 1`` weights from 0 to 1.

    Raises
    ------
    ValueError
        If ``refinement`` is unknown.
    """
    # Fallback to uniform grading if stretching parameter is practically zero
    if abs(A) < 1e-10:
        return np.linspace(0.0, 1.0, res + 1).tolist()

    if refinement == "START":
        # Fine resolution at -1, mapping [-1, 0] to weights [0, 1]
        xi = np.linspace(-1, 0, res + 1)
        weights = (np.tanh(A * xi) / np.tanh(A)) + 1.0

    elif refinement == "END":
        # Fine resolution at 1, mapping [0, 1] to weights [0, 1]
        xi = np.linspace(0, 1, res + 1)
        weights = np.tanh(A * xi) / np.tanh(A)

    elif refinement == "BOTH":
        # Fine resolution at -1 and 1, mapping [-1, 1] to weights [0, 1]
        xi = np.linspace(-1, 1, res + 1)
        weights = 0.5 * (np.tanh(A * xi) / np.tanh(A)) + 0.5

    else:
        raise ValueError(
            f"Unknown refinement '{refinement}'. Use 'START', 'END', or 'BOTH'."
        )

    # Force exact 0.0 and 1.0 at bounds to avoid floating point inaccuracies
    weights[0] = 0.0
    weights[-1] = 1.0

    return weights.tolist()


def make_weights(
    grading_type: Literal["simple", "chebyshev_identity", "tanh"],
    res: int,
    grading: float,
    refinement: Literal["START", "END", "BOTH"],
) -> list[float]:
    """Create cell weights using the requested grading scheme.

    Parameters
    ----------
    grading_type : {"simple", "chebyshev_identity", "tanh"}
        Grading scheme, see :func:`make_weights_simple_grading`,
        :func:`make_weights_chebyshev_identity` and :func:`make_weights_tanh`.
    res : int
        Number of cells.
    grading : float
        Parameter of the grading scheme.
    refinement : {"START", "END", "BOTH"}
        Where the smallest cells are.

    Returns
    -------
    list of float
        ``res + 1`` weights from 0 to 1.

    Raises
    ------
    ValueError
        If ``grading_type`` is unknown.
    """
    if grading_type == "simple":
        return make_weights_simple_grading(res, grading, refinement)
    elif grading_type == "chebyshev_identity":
        return make_weights_chebyshev_identity(res, grading, refinement)
    elif grading_type == "tanh":
        return make_weights_tanh(res, grading, refinement)
    else:
        raise ValueError(
            f"Unknown grading type '{grading_type}'. Use 'simple', "
            "'chebyshev_identity', or 'tanh'."
        )


def generate_grid_vertices_2D(
    res: Sequence[int],
    corner_vertices: Sequence[Sequence[float]],
    border_vertices: Sequence[Any] | None = None,
    x_weights: Sequence[float] | None = None,
    y_weights: Sequence[float] | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate the vertices of a 2D block from its corners and borders.

    Parameters
    ----------
    res : Sequence of int
        Vertex resolution in ``[y, x]`` order.
    corner_vertices : Sequence of (x, y)
        Corners in ``[-x-y, +x-y, -x+y, +x+y]`` order.
    border_vertices : Sequence or None, optional
        Vertex lists of the ``[-x, +x, -y, +y]`` borders. Missing (None)
        borders are interpolated linearly from the corners. Default is None.
    x_weights : Sequence of float or None, optional
        Interpolation weights along the x-borders (i.e. along y). Default is
        None (uniform).
    y_weights : Sequence of float or None, optional
        Interpolation weights along the y-borders (i.e. along x). Default is
        None (uniform).
    dtype : torch.dtype, optional
        Dtype of the grid. Default is ``torch.float32``.

    Returns
    -------
    torch.Tensor
        Vertex coordinates of shape ``[1, 2, res_y, res_x]``.
    """
    border_to_corners = {0: (0, 2), 1: (1, 3), 2: (0, 1), 3: (2, 3)}
    assert isinstance(corner_vertices, (tuple, list))
    assert len(corner_vertices) == 4

    border_vertices = [None] * 4 if border_vertices is None else list(border_vertices)

    if x_weights is None:
        # weights for x-boundaries, so based on y coordinate
        x_weights = [i / (res[0] - 1) for i in range(res[0])]
    else:
        _check_weights(x_weights, res[0], "x_weights")

    if y_weights is None:
        y_weights = [i / (res[1] - 1) for i in range(res[1])]
    else:
        _check_weights(y_weights, res[1], "y_weights")

    for border_idx in range(len(border_vertices)):
        r = res[border_idx // 2]
        if border_vertices[border_idx] is None:
            lower_corner = corner_vertices[border_to_corners[border_idx][0]]
            upper_corner = corner_vertices[border_to_corners[border_idx][1]]
            weights = x_weights if border_idx < 2 else y_weights
            border_vertices[border_idx] = []
            for idx in range(r):
                weight_upper = weights[idx]
                weight_lower = 1 - weight_upper
                border_vertices[border_idx].append(
                    (
                        lower_corner[0] * weight_lower + upper_corner[0] * weight_upper,
                        lower_corner[1] * weight_lower + upper_corner[1] * weight_upper,
                    )
                )
        else:
            assert len(border_vertices[border_idx]) == r, "is %d, expected %d" % (
                len(border_vertices[border_idx]),
                r,
            )
            # TODO: check that corners match

    # print(border_vertices)

    return interpolate_vertices_from_borders_2D(
        border_vertices, x_weights=x_weights, y_weights=y_weights, dtype=dtype
    )


def extrapolate_boundary_layers(
    grid: torch.Tensor, boundaries: Sequence[tuple[str, float]] = []
) -> torch.Tensor:
    """Add linearly extrapolated vertex layers at the given boundaries.

    Parameters
    ----------
    grid : torch.Tensor
        2D vertex coordinates of shape NCHW.
    boundaries : Sequence of (str, float), optional
        Boundaries (``"-x"``, ``"+x"``, ``"-y"``, ``"+y"``) with their
        extrapolation scale, e.g. ``[("-x", 0.5)]``. Default is empty.

    Returns
    -------
    torch.Tensor
        Grid with the additional layers.
    """
    assert isinstance(grid, torch.Tensor) and grid.dim() == 4, (
        "grid must be a 2D torch tensor of shape NCHW"
    )
    assert grid.size(-1) > 1 and grid.size(-2) > 1, "grid must be at least 2x2"
    assert all(
        bound in ["-x", "+x", "-y", "+y"] and scale > 0 for bound, scale in boundaries
    )

    for bound, scale in boundaries:
        if bound == "-x":
            layer_1 = grid[..., :1]
            layer_2 = grid[..., 1:2]
        if bound == "+x":
            layer_1 = grid[..., -1:]
            layer_2 = grid[..., -2:-1]
        if bound == "-y":
            layer_1 = grid[..., :1, :]
            layer_2 = grid[..., 1:2, :]
        if bound == "+y":
            layer_1 = grid[..., -1:, :]
            layer_2 = grid[..., -2:-1, :]

        bound_layer = layer_1 + (layer_1 - layer_2) * scale

        if bound[0] == "-":
            parts = [bound_layer, grid]
        else:
            parts = [grid, bound_layer]
        if bound[1] == "x":
            dim = -1
        else:
            dim = -2

        grid = torch.cat(parts, dim=dim)

    return grid


def get_extrapolated_boundary_layer(
    grid: torch.Tensor, bound: str, scale: float
) -> torch.Tensor:
    """Get the boundary vertex layer together with a linearly extrapolated one.

    Parameters
    ----------
    grid : torch.Tensor
        2D vertex coordinates of shape NCHW.
    bound : {"-x", "+x", "-y", "+y"}
        Boundary.
    scale : float
        Extrapolation scale relative to the size of the boundary cells.

    Returns
    -------
    torch.Tensor
        The two layers, ordered along the boundary axis.
    """
    assert isinstance(grid, torch.Tensor) and grid.dim() == 4, (
        "grid must be a 2D torch tensor of shape NCHW"
    )
    assert grid.size(-1) > 1 and grid.size(-2) > 1, "grid must be at least 2x2"
    assert bound in ["-x", "+x", "-y", "+y"]

    if bound == "-x":
        layer_1 = grid[..., :1]
        layer_2 = grid[..., 1:2]
    if bound == "+x":
        layer_1 = grid[..., -1:]
        layer_2 = grid[..., -2:-1]
    if bound == "-y":
        layer_1 = grid[..., :1, :]
        layer_2 = grid[..., 1:2, :]
    if bound == "+y":
        layer_1 = grid[..., -1:, :]
        layer_2 = grid[..., -2:-1, :]

    bound_layer = layer_1 + (layer_1 - layer_2) * scale

    if bound[0] == "-":
        parts = [bound_layer, layer_1]
    else:
        parts = [layer_1, bound_layer]
    if bound[1] == "x":
        dim = -1
    else:
        dim = -2

    return torch.cat(parts, dim=dim)


def make_wall_refined_ortho_grid(
    res_x: int,
    res_y: int,
    corner_lower: Sequence[float] = (0, 0),
    corner_upper: Sequence[float] = (1, 1),
    wall_refinement: Sequence[str] = [],
    base: float | Sequence[float] = 1.05,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate an axis-aligned 2D grid with exponential refinement at walls.

    Parameters
    ----------
    res_x : int
        Number of cells in x.
    res_y : int
        Number of cells in y.
    corner_lower : Sequence of float, optional
        Lower corner ``(x, y)``. Default is ``(0, 0)``.
    corner_upper : Sequence of float, optional
        Upper corner ``(x, y)``. Default is ``(1, 1)``.
    wall_refinement : Sequence of str, optional
        Boundaries (``"-x"``, ``"+x"``, ``"-y"``, ``"+y"``) to refine towards.
        Default is empty.
    base : float or Sequence of float, optional
        Cell size ratio, see :func:`make_weights_exp`, globally or per axis.
        Default is 1.05.
    dtype : torch.dtype, optional
        Dtype of the grid. Default is ``torch.float32``.

    Returns
    -------
    torch.Tensor
        Vertex coordinates of shape ``[1, 2, res_y + 1, res_x + 1]``.
    """
    dims = 2
    assert isinstance(corner_lower, (list, tuple)) and len(corner_lower) == 2
    assert isinstance(corner_upper, (list, tuple)) and len(corner_upper) == 2
    corners = [
        tuple(corner_lower),
        (corner_upper[0], corner_lower[1]),
        (corner_lower[0], corner_upper[1]),
        tuple(corner_upper),
    ]
    # y_weights = [(i/(res_x))**exponent for i in range(res_x+1)]
    # y_weights_inv = [1 - (1 - i/(res_x))**exponent for i in range(res_x+1)]
    # x_weights = [(i/(res_y))**exponent for i in range(res_y+1)]
    # x_weights_inv = [1 - (1 - i/(res_y))**exponent for i in range(res_y+1)]

    if not isinstance(base, (list, tuple)):
        base = [cast(float, base)] * dims

    # transformation of block
    y_w = None
    if "-x" in wall_refinement:
        if "+x" in wall_refinement:
            y_w = make_weights_exp(
                res_x, base=base[0], refinement="BOTH"
            )  # y_weights[:res_x//2] + y_weights_inv[res_x//2:]
        else:
            y_w = make_weights_exp(res_x, base=base[0], refinement="START")  # y_weights
    elif "+x" in wall_refinement:
        y_w = make_weights_exp(res_x, base=base[0], refinement="END")  # y_weights_inv

    x_w = None
    if "-y" in wall_refinement:
        if "+y" in wall_refinement:
            x_w = make_weights_exp(
                res_y, base=base[1], refinement="BOTH"
            )  # x_weights[:res_x//2] + x_weights_inv[res_x//2:]
        else:
            x_w = make_weights_exp(res_y, base=base[1], refinement="START")  # x_weights
    elif "+y" in wall_refinement:
        x_w = make_weights_exp(res_y, base=base[1], refinement="END")  # x_weights_inv

    grid = generate_grid_vertices_2D(
        [res_y + 1, res_x + 1], corners, None, x_weights=x_w, y_weights=y_w, dtype=dtype
    )

    return grid


def extrude_grid_z(
    grid: torch.Tensor,
    res_z: int,
    start_z: float = 0.0,
    end_z: float = 1.0,
    weights_z: list | str | None = None,
    exp_base: float = 1.05,
) -> torch.Tensor:
    """Extrude a 2D grid along z.

    Parameters
    ----------
    grid : torch.Tensor
        2D vertex coordinates of shape ``[1, 2, H, W]``.
    res_z : int
        Number of cells in z.
    start_z : float, optional
        Lower z coordinate. Default is 0.
    end_z : float, optional
        Upper z coordinate. Default is 1.
    weights_z : list, str or None, optional
        Interpolation weights along z, or one of ``"LINEAR"`` (same as None),
        ``"EXP"``/``"EXP_BOTH"``, ``"EXP_START"``, ``"EXP_END"``.
        Default is None.
    exp_base : float, optional
        Cell size ratio for the exponential weights. Default is 1.05.

    Returns
    -------
    torch.Tensor
        3D vertex coordinates of shape ``[1, 3, res_z + 1, H, W]``.

    Raises
    ------
    ValueError
        If ``weights_z`` is unknown.
    """
    assert grid.dim() == 4 and grid.size(1) == 2
    res_x = grid.size(-1) - 1
    res_y = grid.size(-2) - 1

    if isinstance(weights_z, list):
        assert len(weights_z) == (res_z + 1)
    elif weights_z is None or weights_z == "LINEAR":
        weights_z = make_weights_linear(res_z)
    elif weights_z == "EXP" or weights_z == "EXP_BOTH":
        weights_z = make_weights_exp(res_z, base=exp_base, refinement="BOTH")
    elif weights_z == "EXP_START":
        weights_z = make_weights_exp(res_z, base=exp_base, refinement="START")
    elif weights_z == "EXP_END":
        weights_z = make_weights_exp(res_z, base=exp_base, refinement="EXP_END")
    else:
        raise ValueError("Unknown weights specification")

    def lerp(a: float, b: float, t: float) -> float:
        return a * (1 - t) + b * t

    coords_z = torch.tensor(
        [lerp(start_z, end_z, w) for w in weights_z],
        device=grid.device,
        dtype=grid.dtype,
    )

    coords_z = coords_z.reshape((1, 1, res_z + 1, 1, 1)).repeat(
        1, 1, 1, res_y + 1, res_x + 1
    )
    grid = grid.reshape((1, 2, 1, res_y + 1, res_x + 1)).repeat(1, 1, res_z + 1, 1, 1)
    grid = torch.cat([grid, coords_z], dim=1)

    return grid


def make_torus_2D(
    res: int,
    r1: float,
    r2: float,
    start_angle: float,
    angle: float,
    offset: str | Sequence[float] | np.ndarray | torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a 2D grid of a torus (annulus) segment.

    The x axis of the grid goes along the angle, the y axis along the radius.

    Parameters
    ----------
    res : int
        Number of cells along the angle; the radial resolution is chosen to give
        approximately square cells.
    r1 : float
        Inner radius.
    r2 : float
        Outer radius.
    start_angle : float
        Start angle in degrees, 0 is the x-axis.
    angle : float
        Angular extent in degrees, counterclockwise.
    offset : str, Sequence of float, numpy.ndarray, torch.Tensor or None, optional
        Translation ``(x, y)`` of the grid, or ``"CENTER"`` to center its
        bounding box on the origin. Default is None.
    dtype : torch.dtype, optional
        Dtype of the grid. Default is ``torch.float32``.

    Returns
    -------
    torch.Tensor
        Vertex coordinates in NCHW layout with C = (x, y).
    """
    assert res > 1
    assert r1 > 0
    assert r2 > r1
    start_angle = start_angle % 360
    x = res + 1
    deg_step = angle / (x - 1)
    rad_step = np.deg2rad(deg_step)
    start_rad = np.deg2rad(start_angle)
    end_rad = start_rad + np.deg2rad(angle)
    corners = [
        (np.cos(start_rad) * r1, np.sin(start_rad) * r1),
        (np.cos(end_rad) * r1, np.sin(end_rad) * r1),
        (np.cos(start_rad) * r2, np.sin(start_rad) * r2),
        (np.cos(end_rad) * r2, np.sin(end_rad) * r2),
    ]
    lower_border = [
        (np.cos(start_rad + rad_step * i) * r1, np.sin(start_rad + rad_step * i) * r1)
        for i in range(x)
    ]  # -y
    upper_border = [
        (np.cos(start_rad + rad_step * i) * r2, np.sin(start_rad + rad_step * i) * r2)
        for i in range(x)
    ]  # -y

    # roughly square cells, growing linearly with radius
    r = r2 - r1
    sizes = []
    d = r1
    y = 1
    width_scale = 2 * np.pi / x * (abs(angle) / 360)
    while d < r2:
        width = d * width_scale
        sizes.append(width)
        d += width
        y += 1
    scale = (d - r1) / r
    sizes = [w / scale for w in sizes]
    # interpolation weights in [0,1]
    x_weights: list[float] = [0]
    x_weights += [w / r for w in np.cumsum(sizes)]

    # print("square cells: (x=%d, y=%d),\ns=%s,\nw=%s"%(x,y,sizes,x_weights))

    # l_border = [() for i in range(res)]
    # print(lower_border)
    grid = generate_grid_vertices_2D(
        [y, x],
        corners,
        [None, None, lower_border, upper_border],
        x_weights=x_weights,
        dtype=dtype,
    )

    if offset == "CENTER":
        # via AABB
        vertex_corners = torch.tensor(corners, dtype=dtype)
        lower, _ = vertex_corners.min(dim=0)
        upper, _ = vertex_corners.max(dim=0)
        size = upper - lower
        center = lower + size * 0.5
        offset = -center
        print(offset)

    if isinstance(offset, (list, tuple, np.ndarray)):
        offset = torch.tensor(offset, dtype=dtype)

    if isinstance(offset, torch.Tensor):
        assert offset.dtype == dtype
        assert offset.dim() == 1
        assert offset.size(0) == 2
        offset = offset.view(1, 2, 1, 1)
        grid = grid + offset

    return grid
