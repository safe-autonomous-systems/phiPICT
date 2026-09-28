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
# matplotlib/imageio are imported on first use.
# Moved into the phipict package, formatted, linted and typed.

"""Text, plot and image output of grids and domain fields."""

import os
from collections.abc import Sequence
from typing import Any, TypeAlias, cast

import numpy as np
import torch

from phipict import _C
from phipict.grid.resample import (
    sample_multi_coords_to_uniform_grid,
)

# Block arrangement: "H"/"V" (or None for "H"), or rows of block indices (-1: empty).
Layout: TypeAlias = str | Sequence[Sequence[int]] | Sequence[int] | np.ndarray | None
# Padding color: a single value or one value per channel.
PadColor: TypeAlias = float | Sequence[float]
# Output shape of resampled images: size per axis or a single size for all axes.
ResamplingShape: TypeAlias = int | Sequence[int] | torch.Tensor


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a NumPy array.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor, may require gradients and live on any device.

    Returns
    -------
    numpy.ndarray
        Detached copy on the CPU.
    """
    return tensor.detach().cpu().numpy()


ttonp = tensor_to_numpy


def numerical_to_numpy(
    data: float | list | tuple | torch.Tensor | np.ndarray,
) -> np.ndarray:
    """Convert numerical data to a NumPy array.

    Parameters
    ----------
    data : float, list, tuple, torch.Tensor or numpy.ndarray
        Input data.

    Returns
    -------
    numpy.ndarray
        The data as array.

    Raises
    ------
    TypeError
        If the data type is not supported.
    """
    if isinstance(data, (int, float, list, tuple)):
        return np.asarray(data)
    if isinstance(data, torch.Tensor):
        return tensor_to_numpy(data)
    if isinstance(data, np.ndarray):
        return data
    raise TypeError


ntonp = numerical_to_numpy


class StringWriter:
    """Minimal writable text stream that collects its output in memory."""

    def __init__(self) -> None:
        """Create an empty writer."""
        self.__s: list[str] = []

    def write(self, s: str, *fmt: Any) -> None:
        """Write a string.

        Parameters
        ----------
        s : str
            String, used as printf-style format if ``fmt`` is given.
        *fmt : Any
            Format arguments.
        """
        if fmt:
            s = s % (*fmt,)
        self.__s.append(s)

    def write_line(self, s: str = "", *fmt: Any, newline: str = "\n") -> None:
        """Write a string followed by a newline.

        Parameters
        ----------
        s : str, optional
            String, used as printf-style format if ``fmt`` is given.
            Default is ``""``.
        *fmt : Any
            Format arguments.
        newline : str, optional
            Line terminator. Default is a newline.
        """
        if s:
            if fmt:
                s = s % (*fmt,)
            self.__s.append(s)
        self.__s.append(newline)

    def flush(self) -> None:
        """Do nothing; provided for stream compatibility."""
        pass

    def reset(self) -> None:
        """Discard all written text."""
        self.__s = []

    def get_string(self) -> str:
        """Get all written text.

        Returns
        -------
        str
            The concatenated text.
        """
        return "".join(self.__s)

    def __str__(self) -> str:
        return self.get_string()


def print_CSR_obj(csrMat: _C.CSRmatrix) -> str:
    """Format a CSR matrix as a dense text table.

    Parameters
    ----------
    csrMat : phipict._C.CSRmatrix
        Matrix to format.

    Returns
    -------
    str
        Dense representation, ``-`` for structural zeros.
    """
    return print_CSR(
        csrMat.row.cpu(), csrMat.index.cpu(), csrMat.value.cpu(), csrMat.getRows()
    )


def print_CSR(
    row: torch.Tensor, index: torch.Tensor, value: torch.Tensor, size: int
) -> str:
    """Format CSR matrix data as a dense text table.

    Parameters
    ----------
    row : torch.Tensor
        Row pointers of length ``size + 1``.
    index : torch.Tensor
        Column indices.
    value : torch.Tensor
        Nonzero values.
    size : int
        Number of rows (and columns).

    Returns
    -------
    str
        Dense representation, ``-`` for structural zeros.
    """
    sb = StringWriter()
    assert len(row) == size + 1

    for rowI in range(size):
        row_start = row[rowI]
        row_size = row[rowI + 1] - row_start
        col_step = 0
        for colI in range(size):
            if (
                (row_start + col_step) < len(index)
                and col_step < row_size
                and colI == index[row_start + col_step]
            ):
                sb.write("%5.2f" % value[row_start + col_step])
                col_step += 1
            else:
                sb.write("  -  ")
        sb.write_line()

    # print_fn("Matrix from CSR:\n%s", sb.get_string())
    return sb.get_string()


def print_grid(
    grid: torch.Tensor,
    fmt: str = "{:s}",
    string_buffer: StringWriter | None = None,
) -> str:
    """Format a 2D or 3D tensor as text.

    Parameters
    ----------
    grid : torch.Tensor
        Tensor with 2 or 3 dimensions; 3D tensors are printed slice by slice.
    fmt : str, optional
        Format string for each value. Default is ``"{:s}"``.
    string_buffer : StringWriter or None, optional
        Buffer to append to. Default is None (new buffer).

    Returns
    -------
    str
        All text in the buffer.

    Raises
    ------
    ValueError
        If the tensor has more than 3 dimensions.
    """
    grid = grid.cpu()
    if not string_buffer:
        string_buffer = StringWriter()
    if grid.dim() > 3:
        raise ValueError
    if grid.dim() == 3:
        for idx in range(grid.size()[0]):
            print_grid(grid[idx], fmt=fmt, string_buffer=string_buffer)
            if idx < (grid.size()[0] - 1):
                string_buffer.write_line("---")

    else:
        for row in range(grid.size()[0]):
            for col in range(grid.size()[1]):
                string_buffer.write(fmt.format(grid[row, col].numpy()))
                string_buffer.write(", ")
            string_buffer.write_line()

    return string_buffer.get_string()


def plot_grid(
    grid: torch.Tensor,
    color: str = "tab:blue",
    ax: Any = None,
    linewidth: float = 1.0,
) -> None:
    """Plot the cell edges of a 2D grid with matplotlib.

    Parameters
    ----------
    grid : torch.Tensor
        Vertex coordinates of shape ``[1, 2, H, W]``.
    color : str, optional
        Line color. Default is ``"tab:blue"``.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot into. Default is None (current pyplot axes).
    linewidth : float, optional
        Line width. Default is 1.
    """
    from matplotlib import pyplot as plt

    if ax is None:
        ax = plt
    grid = grid.cpu()
    shape = [grid.size(-2), grid.size(-1)]
    for y in range(shape[0]):
        # plot x edges
        vx, vy = torch.unbind(torch.reshape(grid[0, :, y, :], (2, -1)), dim=0)
        ax.plot(vx.numpy(), vy.numpy(), linestyle="-", color=color, linewidth=linewidth)
    for x in range(shape[1]):
        # plot x edges
        vx, vy = torch.unbind(torch.reshape(grid[0, :, :, x], (2, -1)), dim=0)
        ax.plot(vx.numpy(), vy.numpy(), linestyle="-", color=color, linewidth=linewidth)


def get_grid_AABB(grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the axis-aligned bounding box of a grid.

    Parameters
    ----------
    grid : torch.Tensor
        Coordinates in NCHW or NCDHW layout with C = dims.

    Returns
    -------
    tuple of torch.Tensor
        Minimum and maximum coordinate per axis.
    """
    assert isinstance(grid, torch.Tensor)

    grid_dim = grid.size(1)
    assert grid.dim() == (grid_dim + 2)

    grid_flat = torch.movedim(grid, 1, 0).reshape(grid_dim, -1)

    min_coords = torch.min(grid_flat, dim=1).values
    max_coords = torch.max(grid_flat, dim=1).values

    return min_coords, max_coords


def get_grids_AABB(
    grids: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the axis-aligned bounding box of multiple grids.

    Parameters
    ----------
    grids : Sequence of torch.Tensor
        Coordinates in NCHW or NCDHW layout with C = dims.

    Returns
    -------
    tuple of torch.Tensor
        Minimum and maximum coordinate per axis.
    """
    min_coords, max_coords = get_grid_AABB(grids[0])
    for grid in grids[1:]:
        min_new, max_new = get_grid_AABB(grid)
        min_coords = torch.minimum(min_coords, min_new)
        max_coords = torch.maximum(max_coords, max_new)
    return min_coords, max_coords


def save_plotted_grids(path: str, name: str = "grid", type: str = "svg") -> None:
    """Save and close the current pyplot figure.

    Parameters
    ----------
    path : str
        Output directory.
    name : str, optional
        File name without extension. Default is ``"grid"``.
    type : str, optional
        File extension / format. Default is ``"svg"``.
    """
    from matplotlib import pyplot as plt

    plt.gca().set_aspect("equal")
    plt.savefig(os.path.join(path, "%s.%s" % (name, type)))
    plt.close()
    plt.clf()


def plot_grids(
    grids: torch.Tensor | Sequence[torch.Tensor],
    color: str | list[str] = "tab:blue",
    path: str | None = None,
    name: str = "grid",
    type: str = "svg",
    linewidth: float = 1.0,
    fig_scale: float = 5,
) -> tuple[Any, Any] | None:
    """Plot the cell edges of one or more 2D grids.

    Parameters
    ----------
    grids : torch.Tensor or Sequence of torch.Tensor
        Vertex coordinates, each of shape ``[1, 2, H, W]``.
    color : str or list of str, optional
        One color for all grids or one per grid. Default is ``"tab:blue"``.
    path : str or None, optional
        Output directory. If None, the figure is returned instead of saved.
        Default is None.
    name : str, optional
        File name without extension. Default is ``"grid"``.
    type : str, optional
        File extension / format. Default is ``"svg"``.
    linewidth : float, optional
        Line width. Default is 1.
    fig_scale : float, optional
        Figure size of the shorter side in inches. Default is 5.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes) or None
        Figure and axes if ``path`` is None.

    Raises
    ------
    ValueError
        If a grid has an invalid shape or the number of colors does not match.
    """
    from matplotlib import pyplot as plt

    if not isinstance(grids, (list, tuple)):
        grids = [cast(torch.Tensor, grids)]
    if not isinstance(color, list):
        color = [color] * len(grids)

    if not all(
        isinstance(grid, torch.Tensor)
        and grid.dim() == 4
        and grid.size(0) == 1
        and grid.size(1) == 2
        for grid in grids
    ):
        raise ValueError(
            "grids must be a list of tensors with shape CNHW with N=1 and C=2"
        )

    if not len(color) == len(grids):
        raise ValueError("Need 1 color per grid or a global color")

    # set figure aspect ratio to fit grid(s)
    min_coords, max_coords = get_grids_AABB(grids)
    grids_size = max_coords.cpu().numpy() - min_coords.cpu().numpy()

    min_dim = np.argmin(grids_size)
    grids_size = grids_size / grids_size[min_dim] * fig_scale

    fig, ax = plt.subplots(1, 1, figsize=(grids_size[0], grids_size[1]))
    # fig.suptitle(name)

    for grid, c in zip(grids, color, strict=False):
        plot_grid(grid, color=c, ax=ax, linewidth=linewidth)

    ax.axis("equal")
    fig.tight_layout()

    if path is not None:
        os.makedirs(path, exist_ok=True)
        fig.savefig(os.path.join(path, "%s.%s" % (name, type)))
        plt.close(fig)
        return None
    else:
        return fig, ax


def vel_to_color(vel: torch.Tensor, mag: torch.Tensor) -> torch.Tensor:
    """Map 2D velocity directions to hue and magnitudes to brightness.

    Parameters
    ----------
    vel : torch.Tensor
        Velocity of shape ``[1, 2, H, W]``.
    mag : torch.Tensor
        Normalized magnitude of shape ``[1, H, W]``.

    Returns
    -------
    torch.Tensor
        RGB image of shape ``[H, W, 3]``.
    """
    h = torch.atan2(vel[0, 1], vel[0, 0]) + np.pi  # [0, 2pi]
    # hue/(np.pi/3) = hue/np.pi *3 -> [0,6]
    R = torch.clamp(torch.abs(h / np.pi * 3 - 3) - 1, 0, 1)
    G = torch.clamp(
        torch.abs((h + 4 / 3 * np.pi) % (2 * np.pi) / np.pi * 3 - 3) - 1, 0, 1
    )
    B = torch.clamp(
        torch.abs((h + 2 / 3 * np.pi) % (2 * np.pi) / np.pi * 3 - 3) - 1, 0, 1
    )
    color = torch.stack([R, G, B], dim=-1)
    return color * torch.unsqueeze(mag[0], -1)


def save_np_png_channels(data: torch.Tensor, path: str) -> None:
    """Save each channel of an image as a separate PNG.

    Parameters
    ----------
    data : torch.Tensor
        Image in HWC layout with values in ``[0, 1]``.
    path : str
        Output path containing a ``{channel}`` placeholder.
    """
    channels = torch.split(data, 1, dim=-1)
    for i, img in enumerate(channels):
        save_np_png(img.detach().cpu().numpy(), path.format(channel=i))


def save_np_png(data: np.ndarray, path: str) -> None:
    """Save an image as 8-bit PNG.

    Parameters
    ----------
    data : numpy.ndarray
        Image in HWC layout, values are clipped to ``[0, 1]``.
    path : str
        Output file.
    """
    import imageio

    data = np.clip(data, 0, 1)
    try:
        imageio.imwrite(path, (data * 255.0).astype(np.uint8), "png")  # type: ignore[call-overload]
    except ValueError as e:
        if (
            data.shape[-1] == 1
            and repr(e).find("Can't write images with one color channel.") > 0
        ):
            # newer pillow versions only work with 3-channel images
            data = np.repeat(data, 3, axis=-1)
            imageio.imwrite(path, (data * 255.0).astype(np.uint8), "png")  # type: ignore[call-overload]
        else:
            raise e


def save_np_exr(data: np.ndarray, path: str) -> None:
    """Save an image as EXR.

    Parameters
    ----------
    data : numpy.ndarray
        Image in HWC layout.
    path : str
        Output file.
    """
    import imageio

    imageio.imwrite(path, data, "exr")  # type: ignore[call-overload]


def save_np_img(data: np.ndarray, path: str, image_format: str) -> None:
    """Save an image as PNG or EXR.

    Parameters
    ----------
    data : numpy.ndarray
        Image in HWC layout.
    path : str
        Output file without extension.
    image_format : {"png", "exr"}
        Image format, case-insensitive.

    Raises
    ------
    OSError
        If the format is not supported.
    """
    if image_format.lower() == "png":
        save_np_png(data, path + ".png")
    elif image_format.lower() == "exr":
        save_np_exr(data, path + ".exr")
    else:
        raise OSError("Unsupported image format '%s'." % (image_format,))


def pad_to_size(
    data: torch.Tensor,
    size_x: int,
    size_y: int,
    padding: int = 0,
    pad_col: Sequence[float] | torch.Tensor = (0,),
) -> torch.Tensor:
    """Center an image on a canvas of the given size.

    Parameters
    ----------
    data : torch.Tensor
        Image in HWC layout.
    size_x : int
        Canvas width.
    size_y : int
        Canvas height.
    padding : int, optional
        Additional border on each side. Default is 0.
    pad_col : Sequence of float or torch.Tensor, optional
        Padding value per channel. Default is ``(0,)``.

    Returns
    -------
    torch.Tensor
        Padded image in HWC layout.
    """
    assert isinstance(data, torch.Tensor) and data.dim() == 3

    padded_x = size_x
    padded_y = size_y

    size_x = data.size(1)
    size_y = data.size(0)
    channels = data.size(-1)

    assert len(pad_col) == channels

    pad_left = (padded_x - size_x) // 2
    pad_right = padded_x - size_x - pad_left
    pad_top = (padded_y - size_y) // 2
    pad_bot = padded_y - size_y - pad_top
    paddings = [pad_left, pad_right, pad_top, pad_bot]
    paddings = [_ + padding for _ in paddings]

    channel_data = [
        torch.nn.functional.pad(data[..., i], paddings, value=float(pad_col[i]))
        for i in range(channels)
    ]
    data = torch.stack(channel_data, dim=-1)

    return data


def arrange_blocks(
    blocks: Sequence[torch.Tensor],
    layout: Layout = "H",
    padding: int = 2,
    pad_col: PadColor = 0,
) -> torch.Tensor:
    """Arrange block images on a single canvas.

    Parameters
    ----------
    blocks : Sequence of torch.Tensor
        Block images in HWC layout.
    layout : Layout, optional
        ``"H"`` (or None) for a row, ``"V"`` for a column, or rows of block
        indices with -1 for empty cells. Default is ``"H"``.
    padding : int, optional
        Border around each block. Default is 2.
    pad_col : PadColor, optional
        Padding color, globally or per channel. Default is 0.

    Returns
    -------
    torch.Tensor
        Combined image in HWC layout on the GPU.

    Raises
    ------
    ValueError
        If the layout is invalid.
    """
    rows_layout: Any
    if layout in ["H", "h", None]:
        rows_layout = [list(range(len(blocks)))]
    elif layout in ["V", "v"]:
        rows_layout = [[i] for i in range(len(blocks))]
    elif isinstance(layout, (np.ndarray, list)):
        rows_layout = np.asarray(layout, dtype=np.int32)
        if rows_layout.ndim == 1:
            rows_layout = np.expand_dims(rows_layout, 0)
        assert rows_layout.ndim == 2
    else:
        raise ValueError("invalid layout")

    channels = blocks[0].size(-1)
    if not isinstance(pad_col, (list, tuple)):
        pad_col = [cast(float, pad_col)] * channels
    pad_col_tensor = torch.FloatTensor(np.asarray(pad_col, dtype=np.float32)).cuda()

    row_heights = [0] * len(rows_layout)
    col_widths = [0] * len(rows_layout[0])
    for row_idx, row in enumerate(rows_layout):
        for col_idx, block_idx in enumerate(row):
            if block_idx >= 0:
                h = blocks[block_idx].size(0)
                w = blocks[block_idx].size(1)
                row_heights[row_idx] = max(row_heights[row_idx], h)
                col_widths[col_idx] = max(col_widths[col_idx], w)

    rows = []
    for row_idx, row in enumerate(rows_layout):
        row_data = []
        for col_idx, block_idx in enumerate(row):
            size_y = row_heights[row_idx]
            size_x = col_widths[col_idx]
            if block_idx == -1:
                row_data.append(
                    torch.ones(
                        (size_y + padding * 2, size_x + padding * 2, channels),
                        dtype=torch.float32,
                        device=torch.device("cuda"),
                    )
                    * pad_col_tensor
                )
            else:
                row_data.append(
                    pad_to_size(
                        blocks[block_idx], size_x, size_y, padding, pad_col_tensor
                    )
                )
        rows.append(torch.cat(row_data, dim=1))
    return torch.cat(rows, dim=0)


def reduce_3D(
    data: torch.Tensor, size: _C.Int4, axis3D: int = 0, mode3D: str = "slice"
) -> torch.Tensor:
    """Reduce 3D data to 2D.

    Parameters
    ----------
    data : torch.Tensor
        Data in DHW[C] layout.
    size : phipict._C.Int4
        Spatial size of the data, used to find the center slice.
    axis3D : int, optional
        Axis to reduce: 0 = z, 1 = y, 2 = x. Default is 0.
    mode3D : {"slice", "mean", "max"}, optional
        Take the center slice, the mean or the maximum along the axis.
        Default is ``"slice"``.

    Returns
    -------
    torch.Tensor
        Reduced data.

    Raises
    ------
    ValueError
        If ``axis3D`` or ``mode3D`` is invalid.
    """
    if mode3D == "slice":
        if axis3D == 0:
            data = data[size.z // 2]
        elif axis3D == 1:
            data = data[:, size.y // 2]
        elif axis3D == 2:
            data = data[:, :, size.x // 2]
        else:
            raise ValueError
    elif mode3D == "mean":
        data = torch.mean(data, dim=axis3D)
    elif mode3D == "max":
        data = torch.amax(data, dim=axis3D)
    else:
        raise ValueError
    return data


def _resample_block_data(
    data_list: Sequence[torch.Tensor],
    vertex_coord_list: Sequence[torch.Tensor],
    resampling_out_shape: ResamplingShape,
    ndims: int,
    fill_max_steps: int = 0,
) -> torch.Tensor:
    """Resample per-block data onto a single uniform grid.

    Parameters
    ----------
    data_list : Sequence of torch.Tensor
        Data per block, each of shape ``[1, C, *block_spatial]``.
    vertex_coord_list : Sequence of torch.Tensor
        Vertex coordinates per block, matching ``data_list``.
    resampling_out_shape : ResamplingShape
        Output resolution, as a single int applied to every axis, or one value
        per axis as a list, tuple or tensor.
    ndims : int
        Number of spatial dimensions (2 or 3).
    fill_max_steps : int, optional
        Hole-filling iterations for output cells not covered by any block.
        Default is 0.

    Returns
    -------
    torch.Tensor
        Resampled data of shape ``[1, C, *out_shape]``.

    Raises
    ------
    TypeError
        If ``resampling_out_shape`` is not an int, list, tuple or tensor.
    """
    if isinstance(resampling_out_shape, (list, tuple)):
        out_shape = torch.tensor(resampling_out_shape, dtype=torch.int32)
    elif isinstance(resampling_out_shape, torch.Tensor):
        out_shape = resampling_out_shape
    elif isinstance(resampling_out_shape, (int,)):
        out_shape = torch.tensor([resampling_out_shape] * ndims, dtype=torch.int32)
    else:
        raise TypeError("resampling_out_shape must be list, tensor, or int")

    data = sample_multi_coords_to_uniform_grid(
        data_list,
        vertex_coord_list,
        out_shape,
        is_cell_coords=False,
        fill_max_steps=fill_max_steps,
    )
    # if ndims==3:
    #    raise NotImplementedError("TODO: implement 3D reduction.")

    return data


def save_block_data_image(
    block_data: Sequence[torch.Tensor],
    domain: _C.Domain,
    path: str,
    name: str,
    id: int,
    min_val: float | torch.Tensor = 0,
    max_val: float | torch.Tensor = 1,
    normalize: bool = False,
    pad_col: PadColor = 1,
    layout: Layout = "H",
    axis3D: int = 0,
    mode3D: str = "slice",
    vertex_coord_list: Sequence[torch.Tensor] | None = None,
    resampling_out_shape: ResamplingShape | None = 10,
    fill_max_steps: int = 0,
    image_format: str = "PNG",
) -> None:
    """Save per-block data as image(s).

    Data with 1 or 3 channels is saved as a single image, otherwise each channel
    is saved separately.

    Parameters
    ----------
    block_data : Sequence of torch.Tensor
        Data per block, each of shape ``[1, C, *block_spatial]``.
    domain : phipict._C.Domain
        Domain to visualize.
    path : str
        Output directory.
    name : str
        File name prefix.
    id : int
        Index appended to the file name.
    min_val : float or torch.Tensor, optional
        Value mapped to 0. Default is 0.
    max_val : float or torch.Tensor, optional
        Value mapped to 1. Default is 1.
    normalize : bool, optional
        Whether to use the data range instead of ``min_val``/``max_val``.
        Default is False.
    pad_col : PadColor, optional
        Padding color, see :func:`arrange_blocks`. Default is 1.
    layout : Layout, optional
        Block arrangement, see :func:`arrange_blocks`. Default is ``"H"``.
    axis3D : int, optional
        Axis to reduce in 3D, see :func:`reduce_3D`. Default is 0.
    mode3D : str, optional
        Reduction mode in 3D, see :func:`reduce_3D`. Default is ``"slice"``.
    vertex_coord_list : Sequence of torch.Tensor or None, optional
        Per-block vertex coordinates. If given together with
        ``resampling_out_shape``, the blocks are resampled to a common uniform
        grid instead of being arranged side by side. Default is None.
    resampling_out_shape : ResamplingShape or None, optional
        Shape of the uniform grid. Default is 10.
    fill_max_steps : int, optional
        Number of hole-filling iterations of the resampling. Default is 0.
    image_format : str, optional
        ``"PNG"`` or ``"EXR"``. Default is ``"PNG"``.

    Raises
    ------
    TypeError
        If ``block_data`` is not a list of tensors.
    ValueError
        If ``block_data`` does not match the blocks of the domain.
    """
    if not (
        isinstance(block_data, (list, tuple))
        and all(isinstance(_, torch.Tensor) for _ in block_data)
    ):
        raise TypeError("block_data must be a list of tensors.")
    if not domain.getNumBlocks() == len(block_data):
        raise ValueError("block_data must match numbers of blocks on domain.")

    def cmp_data_block_size(data: torch.Tensor, block: _C.Block) -> bool:
        dims = block.getSpatialDims()
        if not data.dim() == (dims + 2):
            return False
        for dim in range(dims):
            if not data.size(dim + 2) == block.getDim(dim + 2):
                return False
        return True

    if not all(
        cmp_data_block_size(data, block)
        for data, block in zip(block_data, domain.getBlocks(), strict=False)
    ):
        raise ValueError("data dimensionality and spatial shape must match blocks.")

    channels = block_data[0].size(1)
    if not all(data.size(1) == channels for data in block_data):
        raise ValueError("inconsistent use of channels (dim 1).")

    num_blocks = domain.getNumBlocks()
    ndims = domain.getSpatialDims()

    if vertex_coord_list is None or resampling_out_shape is None:
        if normalize:
            min_val = torch.min(block_data[0])
            max_val = torch.max(block_data[0])
            for blockIdx in range(1, num_blocks):
                min_val = torch.minimum(min_val, torch.min(block_data[blockIdx]))
                max_val = torch.maximum(max_val, torch.max(block_data[blockIdx]))

        padding = 1

        block_data_out = []
        for blockIdx in range(0, num_blocks):
            block = domain.getBlock(blockIdx)
            size = block.getSizes()
            # data = block.passiveScalar[0,0]
            data = block_data[blockIdx][0]
            if ndims == 2:
                data = torch.permute(data, (1, 2, 0))  # CHW -> HWC
            elif ndims == 3:
                data = torch.permute(data, (1, 2, 3, 0))
                data = reduce_3D(data, size, axis3D, mode3D)
            data = (data - min_val) / (max_val - min_val)
            # block_data.append(torch.stack([data]*3, axis=-1))
            block_data_out.append(data)

        image = arrange_blocks(
            block_data_out, layout=layout, padding=padding, pad_col=pad_col
        )

    else:
        data = _resample_block_data(
            block_data,
            vertex_coord_list,
            resampling_out_shape,
            ndims,
            fill_max_steps=fill_max_steps,
        )
        data = data[0]  # NC[D]HW -> C[D]HW
        if ndims == 2:
            data = torch.permute(data, (1, 2, 0))  # CHW -> HWC
        elif ndims == 3:
            size = _C.Int4(x=data.size(-1), y=data.size(-2), z=data.size(-3))
            data = torch.permute(data, (1, 2, 3, 0))  # CDHW -> DHWC
            data = reduce_3D(data, size, axis3D, mode3D)

        if normalize:
            min_val = torch.min(data)
            max_val = torch.max(data)
        data = (data - min_val) / (max_val - min_val)

        # block_data = torch.stack([data]*3, axis=-1)
        image = data

    if channels not in [1, 3]:
        imgs = torch.split(image, 1, dim=-1)
        for i, img in enumerate(imgs):
            save_np_img(
                img.detach().cpu().numpy(),
                os.path.join(
                    path,
                    "%s_c%d_%04d"
                    % (
                        name,
                        i,
                        id,
                    ),
                ),
                image_format,
            )
    else:
        save_np_img(
            image.cpu().detach().numpy(),
            os.path.join(
                path,
                "%s_%04d"
                % (
                    name,
                    id,
                ),
            ),
            image_format,
        )


def _passive_scalar(block: _C.Block) -> torch.Tensor:
    """Get the passive scalar of a block that is known to have one.

    Parameters
    ----------
    block : phipict._C.Block
        Block to read the passive scalar from.

    Returns
    -------
    torch.Tensor
        The block's passive scalar field.
    """
    assert block.passiveScalar is not None
    return block.passiveScalar


def save_scalar_image(
    domain: _C.Domain,
    path: str,
    name: str,
    id: int,
    min_val: float | torch.Tensor = 0,
    max_val: float | torch.Tensor = 1,
    normalize: bool = False,
    pad_col: PadColor = 1,
    layout: Layout = "H",
    axis3D: int = 0,
    mode3D: str = "slice",
    vertex_coord_list: Sequence[torch.Tensor] | None = None,
    resampling_out_shape: ResamplingShape | None = 10,
    fill_max_steps: int = 0,
) -> None:
    """Save the passive scalar of a domain as PNG image(s).

    Does nothing if the domain has no passive scalar.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to visualize.
    path : str
        Output directory.
    name : str
        File name prefix.
    id : int
        Index appended to the file name.
    min_val : float or torch.Tensor, optional
        Value mapped to 0. Default is 0.
    max_val : float or torch.Tensor, optional
        Value mapped to 1. Default is 1.
    normalize : bool, optional
        Whether to use the data range instead of ``min_val``/``max_val``.
        Default is False.
    pad_col : PadColor, optional
        Padding color, see :func:`arrange_blocks`. Default is 1.
    layout : Layout, optional
        Block arrangement, see :func:`arrange_blocks`. Default is ``"H"``.
    axis3D : int, optional
        Axis to reduce in 3D, see :func:`reduce_3D`. Default is 0.
    mode3D : str, optional
        Reduction mode in 3D, see :func:`reduce_3D`. Default is ``"slice"``.
    vertex_coord_list : Sequence of torch.Tensor or None, optional
        Per-block vertex coordinates. If given together with
        ``resampling_out_shape``, the blocks are resampled to a common uniform
        grid instead of being arranged side by side. Default is None.
    resampling_out_shape : ResamplingShape or None, optional
        Shape of the uniform grid. Default is 10.
    fill_max_steps : int, optional
        Number of hole-filling iterations of the resampling. Default is 0.
    """
    # axis: 0=z, 1=y, 2=x
    assert domain.getNumBlocks() > 0
    ndims = domain.getSpatialDims()
    channels = domain.getPassiveScalarChannels()

    if not domain.hasPassiveScalar():
        return

    if vertex_coord_list is None or resampling_out_shape is None:
        if normalize:
            block = domain.getBlock(0)
            min_val = torch.min(_passive_scalar(block))
            max_val = torch.max(_passive_scalar(block))
            for blockIdx in range(1, domain.getNumBlocks()):
                block = domain.getBlock(blockIdx)
                min_val = torch.minimum(min_val, torch.min(_passive_scalar(block)))
                max_val = torch.maximum(max_val, torch.max(_passive_scalar(block)))

        padding = 1

        block_images = []
        for blockIdx in range(0, domain.getNumBlocks()):
            block = domain.getBlock(blockIdx)
            size = block.getSizes()
            # data = _passive_scalar(block)[0,0]
            data = _passive_scalar(block)[0]
            if ndims == 2:
                data = torch.permute(data, (1, 2, 0))  # CHW -> HWC
            elif ndims == 3:
                data = torch.permute(data, (1, 2, 3, 0))
                data = reduce_3D(data, size, axis3D, mode3D)
            data = (data - min_val) / (max_val - min_val)
            # block_images.append(torch.stack([data]*3, axis=-1))
            block_images.append(data)

        image = arrange_blocks(
            block_images, layout=layout, padding=padding, pad_col=pad_col
        )

    else:
        data_list = [_passive_scalar(block) for block in domain.getBlocks()]

        data = _resample_block_data(
            data_list,
            vertex_coord_list,
            resampling_out_shape,
            ndims,
            fill_max_steps=fill_max_steps,
        )
        data = data[0]  # NC[D]HW -> C[D]HW
        if ndims == 2:
            data = torch.permute(data, (1, 2, 0))  # CHW -> HWC
        elif ndims == 3:
            size = _C.Int4(x=data.size(-1), y=data.size(-2), z=data.size(-3))
            data = torch.permute(data, (1, 2, 3, 0))  # CDHW -> DHWC
            data = reduce_3D(data, size, axis3D, mode3D)

        if normalize:
            min_val = torch.min(data)
            max_val = torch.max(data)
        data = (data - min_val) / (max_val - min_val)

        # block_data = torch.stack([data]*3, axis=-1)
        image = data

    if channels not in [1, 3]:
        imgs = torch.split(image, 1, dim=-1)
        for i, img in enumerate(imgs):
            save_np_png(
                img.detach().cpu().numpy(),
                os.path.join(
                    path,
                    "%s_c%d_%04d.png"
                    % (
                        name,
                        i,
                        id,
                    ),
                ),
            )
    else:
        save_np_png(
            image.cpu().detach().numpy(),
            os.path.join(
                path,
                "%s_%04d.png"
                % (
                    name,
                    id,
                ),
            ),
        )


def save_pressure_image(
    domain: _C.Domain,
    path: str,
    name: str,
    id: int,
    min_val: float | torch.Tensor = 0,
    max_val: float | torch.Tensor = 1,
    normalize: bool = False,
    pad_col: PadColor = (0, 0, 0.5),
    layout: Layout = "H",
    axis3D: int = 0,
    mode3D: str = "slice",
    vertex_coord_list: Sequence[torch.Tensor] | None = None,
    resampling_out_shape: ResamplingShape | None = 10,
    fill_max_steps: int = 0,
) -> None:
    """Save the pressure of a domain as grayscale PNG image.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to visualize.
    path : str
        Output directory.
    name : str
        File name prefix.
    id : int
        Index appended to the file name.
    min_val : float or torch.Tensor, optional
        Value mapped to 0. Default is 0.
    max_val : float or torch.Tensor, optional
        Value mapped to 1. Default is 1.
    normalize : bool, optional
        Whether to use the data range instead of ``min_val``/``max_val``.
        Default is False.
    pad_col : PadColor, optional
        Padding color, see :func:`arrange_blocks`. Default is ``(0, 0, 0.5)``.
    layout : Layout, optional
        Block arrangement, see :func:`arrange_blocks`. Default is ``"H"``.
    axis3D : int, optional
        Axis to reduce in 3D, see :func:`reduce_3D`. Default is 0.
    mode3D : str, optional
        Reduction mode in 3D, see :func:`reduce_3D`. Default is ``"slice"``.
    vertex_coord_list : Sequence of torch.Tensor or None, optional
        Per-block vertex coordinates. If given together with
        ``resampling_out_shape``, the blocks are resampled to a common uniform
        grid instead of being arranged side by side. Default is None.
    resampling_out_shape : ResamplingShape or None, optional
        Shape of the uniform grid. Default is 10.
    fill_max_steps : int, optional
        Number of hole-filling iterations of the resampling. Default is 0.
    """
    assert domain.getNumBlocks() > 0
    # assert domain.getSpatialDims()==2
    ndims = domain.getSpatialDims()

    if vertex_coord_list is None or resampling_out_shape is None:
        if normalize:
            block = domain.getBlock(0)
            min_val = torch.min(block.pressure)
            max_val = torch.max(block.pressure)
            for blockIdx in range(1, domain.getNumBlocks()):
                block = domain.getBlock(blockIdx)
                min_val = torch.minimum(min_val, torch.min(block.pressure))
                max_val = torch.maximum(max_val, torch.max(block.pressure))

        padding = 1

        block_images = []
        for blockIdx in range(0, domain.getNumBlocks()):
            block = domain.getBlock(blockIdx)
            size = block.getSizes()
            data = block.pressure[0, 0]
            if ndims == 3:
                data = reduce_3D(data, size, axis3D, mode3D)
            data = (data - min_val) / (max_val - min_val)
            block_images.append(torch.stack([data] * 3, dim=-1))

        image = arrange_blocks(
            block_images, layout=layout, padding=padding, pad_col=pad_col
        )

    else:
        data_list = [block.pressure for block in domain.getBlocks()]

        data = _resample_block_data(
            data_list,
            vertex_coord_list,
            resampling_out_shape,
            ndims,
            fill_max_steps=fill_max_steps,
        )
        data = data[0, 0]
        if ndims == 3:
            size = _C.Int4(x=data.size(-1), y=data.size(-2), z=data.size(-3))
            data = reduce_3D(data, size, axis3D, mode3D)

        if normalize:
            min_val = torch.min(data)
            max_val = torch.max(data)
        data = (data - min_val) / (max_val - min_val)

        image = torch.stack([data] * 3, dim=-1)

    save_np_png(
        image.cpu().detach().numpy(),
        os.path.join(
            path,
            "%s_%04d.png"
            % (
                name,
                id,
            ),
        ),
    )


def save_velocity_image(
    domain: _C.Domain,
    path: str,
    name: str,
    id: int,
    max_mag: float | torch.Tensor = 1,
    normalize: bool = False,
    pad_col: PadColor = (1, 1, 1),
    layout: Layout = "H",
    axis3D: int = 0,
    mode3D: str = "slice",
    vertex_coord_list: Sequence[torch.Tensor] | None = None,
    resampling_out_shape: ResamplingShape | None = 10,
    fill_max_steps: int = 0,
) -> None:
    """Save the velocity of a domain as PNG image.

    In 2D, direction is encoded as hue and magnitude as brightness; in 3D, the
    absolute velocity components are used as RGB.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to visualize.
    path : str
        Output directory.
    name : str
        File name prefix.
    id : int
        Index appended to the file name.
    max_mag : float or torch.Tensor, optional
        Magnitude mapped to full brightness. Default is 1.
    normalize : bool, optional
        Whether to use the maximum magnitude instead of ``max_mag``.
        Default is False.
    pad_col : PadColor, optional
        Padding color, see :func:`arrange_blocks`. Default is ``(1, 1, 1)``.
    layout : Layout, optional
        Block arrangement, see :func:`arrange_blocks`. Default is ``"H"``.
    axis3D : int, optional
        Axis to reduce in 3D, see :func:`reduce_3D`. Default is 0.
    mode3D : str, optional
        Reduction mode in 3D, see :func:`reduce_3D`. Default is ``"slice"``.
    vertex_coord_list : Sequence of torch.Tensor or None, optional
        Per-block vertex coordinates. If given together with
        ``resampling_out_shape``, the blocks are resampled to a common uniform
        grid instead of being arranged side by side. Default is None.
    resampling_out_shape : ResamplingShape or None, optional
        Shape of the uniform grid. Default is 10.
    fill_max_steps : int, optional
        Number of hole-filling iterations of the resampling. Default is 0.
    """
    assert domain.getNumBlocks() > 0
    ndims = domain.getSpatialDims()

    if vertex_coord_list is None or resampling_out_shape is None:
        blocks = [
            domain.getBlock(blockIdx) for blockIdx in range(0, domain.getNumBlocks())
        ]
        mags = [torch.linalg.vector_norm(block.velocity, dim=1) for block in blocks]

        if normalize:
            max_mag = torch.max(mags[0])
            for mag in mags:
                max_mag = torch.maximum(max_mag, torch.max(mag))

        # vel /= max_mag
        mags = [mag / max_mag for mag in mags]

        padding = 1

        block_images = []
        for block, mag in zip(blocks, mags, strict=False):
            size = block.getSizes()
            if ndims == 2:
                data = vel_to_color(block.velocity, mag)
            else:
                data = torch.abs(block.velocity[0])
                data = torch.permute(data, (1, 2, 3, 0))
                data = reduce_3D(data, size, axis3D, mode3D)
                # data = torch.permute(block.velocity[0,:,size.z//2,:,:], (1,2,0)) / max_mag
                data = data / max_mag
            block_images.append(data)

        image = arrange_blocks(
            block_images, layout=layout, padding=padding, pad_col=pad_col
        )

    else:
        data_list = [block.velocity for block in domain.getBlocks()]

        data = _resample_block_data(
            data_list,
            vertex_coord_list,
            resampling_out_shape,
            ndims,
            fill_max_steps=fill_max_steps,
        )
        mag = torch.linalg.vector_norm(data, dim=1)

        # print(data.size(), mag.size())

        if normalize:
            max_mag = torch.max(mag)
        mag = mag / max_mag

        if ndims == 2:
            data = vel_to_color(data, mag)
        else:
            data = torch.abs(data[0])
            size = _C.Int4(x=data.size(-1), y=data.size(-2), z=data.size(-3))
            data = torch.permute(data, (1, 2, 3, 0))  # DHWC
            data = reduce_3D(data, size, axis3D, mode3D)
            data = data / max_mag

        image = data

        # print(block_data.size())

    save_np_png(
        image.cpu().detach().numpy(),
        os.path.join(
            path,
            "%s_%04d.png"
            % (
                name,
                id,
            ),
        ),
    )


def save_velocity_source_image(
    domain: _C.Domain,
    path: str,
    name: str,
    id: int,
    max_mag: float | torch.Tensor = 1,
    normalize: bool = False,
    pad_col: PadColor = (1, 1, 1),
    layout: Layout = "H",
    axis3D: int = 0,
    mode3D: str = "slice",
    vertex_coord_list: Sequence[torch.Tensor] | None = None,
    resampling_out_shape: ResamplingShape | None = 10,
    fill_max_steps: int = 0,
) -> None:
    """Save the dynamic velocity sources of a domain as PNG image.

    Does nothing if no block has a dynamic velocity source. See
    :func:`save_velocity_image` for the color encoding.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to visualize.
    path : str
        Output directory.
    name : str
        File name prefix.
    id : int
        Index appended to the file name.
    max_mag : float or torch.Tensor, optional
        Magnitude mapped to full brightness. Default is 1.
    normalize : bool, optional
        Whether to use the maximum magnitude instead of ``max_mag``.
        Default is False.
    pad_col : PadColor, optional
        Padding color, see :func:`arrange_blocks`. Default is ``(1, 1, 1)``.
    layout : Layout, optional
        Block arrangement, see :func:`arrange_blocks`. Default is ``"H"``.
    axis3D : int, optional
        Axis to reduce in 3D, see :func:`reduce_3D`. Default is 0.
    mode3D : str, optional
        Reduction mode in 3D, see :func:`reduce_3D`. Default is ``"slice"``.
    vertex_coord_list : Sequence of torch.Tensor or None, optional
        Per-block vertex coordinates. If given together with
        ``resampling_out_shape``, the blocks are resampled to a common uniform
        grid instead of being arranged side by side. Default is None.
    resampling_out_shape : ResamplingShape or None, optional
        Shape of the uniform grid. Default is 10.
    fill_max_steps : int, optional
        Number of hole-filling iterations of the resampling. Default is 0.
    """
    assert domain.getNumBlocks() > 0
    ndims = domain.getSpatialDims()

    if not any(
        block.hasVelocitySource() and not block.isVelocitySourceStatic
        for block in domain.getBlocks()
    ):
        return

    blocks = domain.getBlocks()
    vel_sources = [
        (
            cast(torch.Tensor, block.velocitySource)
            if (block.hasVelocitySource() and not block.isVelocitySourceStatic)
            else torch.zeros_like(block.velocity)
        )
        for block in blocks
    ]

    if vertex_coord_list is None or resampling_out_shape is None:
        # blocks = [domain.getBlock(blockIdx) for blockIdx in range(0, domain.getNumBlocks())]
        mags = [
            torch.linalg.vector_norm(vel_source, dim=1) for vel_source in vel_sources
        ]

        if normalize:
            max_mag = torch.max(mags[0])
            for mag in mags:
                max_mag = torch.maximum(max_mag, torch.max(mag))

        # vel /= max_mag
        mags = [mag / max_mag for mag in mags]

        padding = 1

        block_images = []
        for block, vel_source, mag in zip(blocks, vel_sources, mags, strict=False):
            size = block.getSizes()
            if ndims == 2:
                data = vel_to_color(vel_source, mag)
            else:
                data = torch.abs(vel_source[0])
                data = torch.permute(data, (1, 2, 3, 0))
                data = reduce_3D(data, size, axis3D, mode3D)
                # data = torch.permute(vel_source[0,:,size.z//2,:,:], (1,2,0)) / max_mag
                data = data / max_mag
            block_images.append(data)

        image = arrange_blocks(
            block_images, layout=layout, padding=padding, pad_col=pad_col
        )

    else:
        data_list = (
            vel_sources  # [block.velocitySource for block in domain.getBlocks()]
        )

        data = _resample_block_data(
            data_list,
            vertex_coord_list,
            resampling_out_shape,
            ndims,
            fill_max_steps=fill_max_steps,
        )
        mag = torch.linalg.vector_norm(data, dim=1)

        # print(data.size(), mag.size())

        if normalize:
            max_mag = torch.max(mag)
        mag = mag / max_mag

        if ndims == 2:
            data = vel_to_color(data, mag)
        else:
            data = torch.abs(data[0])
            size = _C.Int4(x=data.size(-1), y=data.size(-2), z=data.size(-3))
            data = torch.permute(data, (1, 2, 3, 0))  # DHWC
            data = reduce_3D(data, size, axis3D, mode3D)
            data = data / max_mag

        image = data

        # print(block_data.size())

    save_np_png(
        image.cpu().detach().numpy(),
        os.path.join(
            path,
            "%s_%04d.png"
            % (
                name,
                id,
            ),
        ),
    )


def save_velocity_exr(
    domain: _C.Domain,
    path: str,
    name: str,
    id: int,
    scale: float | torch.Tensor = 1,
    normalize: bool = False,
    pad_col: PadColor = (1, 1, 1),
    layout: Layout = "H",
    axis3D: int = 0,
    mode3D: str = "slice",
    vertex_coord_list: Sequence[torch.Tensor] | None = None,
    resampling_out_shape: ResamplingShape | None = 10,
    fill_max_steps: int = 0,
) -> None:
    """Save the raw velocity of a domain as EXR image.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to visualize.
    path : str
        Output directory.
    name : str
        File name prefix.
    id : int
        Index appended to the file name.
    scale : float or torch.Tensor, optional
        Factor applied to the velocity. Default is 1.
    normalize : bool, optional
        Whether to scale by the inverse maximum magnitude instead.
        Default is False.
    pad_col : PadColor, optional
        Padding color, see :func:`arrange_blocks`. Default is ``(1, 1, 1)``.
    layout : Layout, optional
        Block arrangement, see :func:`arrange_blocks`. Default is ``"H"``.
    axis3D : int, optional
        Axis to reduce in 3D, see :func:`reduce_3D`. Default is 0.
    mode3D : str, optional
        Reduction mode in 3D, see :func:`reduce_3D`. Default is ``"slice"``.
    vertex_coord_list : Sequence of torch.Tensor or None, optional
        Per-block vertex coordinates. Unused. Default is None.
    resampling_out_shape : ResamplingShape or None, optional
        Unused. Default is 10.
    fill_max_steps : int, optional
        Unused. Default is 0.
    """
    assert domain.getNumBlocks() > 0
    ndims = domain.getSpatialDims()

    blocks = [domain.getBlock(blockIdx) for blockIdx in range(0, domain.getNumBlocks())]

    if normalize:
        mags = [torch.linalg.vector_norm(block.velocity, dim=1) for block in blocks]
        max_mag = torch.max(mags[0])
        for mag in mags:
            max_mag = torch.maximum(max_mag, torch.max(mag))
        scale = 1 / max_mag

    padding = 2

    block_images = []
    for block in blocks:
        size = block.getSizes()
        if ndims == 2:
            data = torch.permute(block.velocity[0], (1, 2, 0))
            data = torch.nn.functional.pad(data, (0, 1, 0, 0, 0, 0), value=0)
        else:
            data = torch.permute(block.velocity[0], (1, 2, 3, 0))
            data = reduce_3D(data, size, axis3D, mode3D)
            # data = torch.permute(block.velocity[0,:,size.z//2,:,:], (1,2,0)) / max_mag
        data = data * scale
        block_images.append(data)

    image = arrange_blocks(
        block_images, layout=layout, padding=padding, pad_col=pad_col
    )

    save_np_exr(
        image.cpu().detach().numpy(),
        os.path.join(
            path,
            "%s_%04d.exr"
            % (
                name,
                id,
            ),
        ),
    )


def save_transform_exr(
    domain: _C.Domain,
    path: str,
    name: str,
    id: int,
    scale: float = 1,
    pad_col: PadColor = (1, 1, 1),
    layout: Layout = "H",
    axis3D: int = 0,
    mode3D: str = "slice",
    vertex_coord_list: Sequence[torch.Tensor] | None = None,
    resampling_out_shape: ResamplingShape | None = 10,
) -> None:
    """Save the block transformations of a 2D domain as EXR images.

    Writes one image per quantity: transform rows, inverse transform rows,
    determinant, and the face normals with their differences.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to visualize.
    path : str
        Output directory.
    name : str
        File name prefix.
    id : int
        Index appended to the file name.
    scale : float, optional
        Factor applied to the data. Default is 1.
    pad_col : PadColor, optional
        Padding color, see :func:`arrange_blocks`. Default is ``(1, 1, 1)``.
    layout : Layout, optional
        Block arrangement, see :func:`arrange_blocks`. Default is ``"H"``.
    axis3D : int, optional
        Axis to reduce in 3D, see :func:`reduce_3D`. Default is 0.
    mode3D : str, optional
        Reduction mode in 3D, see :func:`reduce_3D`. Default is ``"slice"``.
    vertex_coord_list : Sequence of torch.Tensor or None, optional
        Per-block vertex coordinates. Unused. Default is None.
    resampling_out_shape : ResamplingShape or None, optional
        Unused. Default is 10.

    Raises
    ------
    NotImplementedError
        For 3D domains.
    """
    assert domain.getNumBlocks() > 0
    ndims = domain.getSpatialDims()
    assert ndims in [2, 3]

    blocks = [domain.getBlock(blockIdx) for blockIdx in range(0, domain.getNumBlocks())]

    padding = 2

    if ndims == 2:
        keys = ["Tx", "Ty", "Tix", "Tiy", "det"] + [
            "Nx",
            "Ny",
            "Nxd",
            "Nyd",
            "Nd",
        ]
    else:
        keys = ["Tx", "Ty", "Tz", "Tix", "Tiy", "Tiz", "det"]
    block_data: dict[str, list[torch.Tensor]] = {_: [] for _ in keys}
    for block in blocks:
        size = block.getSizes()
        if block.hasTransform():
            if ndims == 2:
                t = block.transform
                assert t is not None
                block_data["Tx"].append(
                    scale
                    * torch.nn.functional.pad(
                        t[0, ..., :ndims], (0, 1, 0, 0, 0, 0), value=0
                    )
                )
                block_data["Ty"].append(
                    scale
                    * torch.nn.functional.pad(
                        t[0, ..., ndims : ndims * 2], (0, 1, 0, 0, 0, 0), value=0
                    )
                )
                block_data["Tix"].append(
                    scale
                    * torch.nn.functional.pad(
                        t[0, ..., ndims * 2 : ndims * 3], (0, 1, 0, 0, 0, 0), value=0
                    )
                )
                block_data["Tiy"].append(
                    scale
                    * torch.nn.functional.pad(
                        t[0, ..., ndims * 3 : ndims * 4], (0, 1, 0, 0, 0, 0), value=0
                    )
                )
                det = t[0, ..., ndims * 4 :]
                block_data["det"].append(
                    scale * torch.nn.functional.pad(det, (0, 2, 0, 0, 0, 0), value=0)
                )

                block_data["Nx"].append(block_data["Tix"][-1] * det)
                block_data["Ny"].append(block_data["Tiy"][-1] * det)

                block_data["Nxd"].append(
                    block_data["Nx"][-1][:, 2:, :] - block_data["Nx"][-1][:, :-2, :]
                )
                block_data["Nyd"].append(
                    block_data["Ny"][-1][2:, :, :] - block_data["Ny"][-1][:-2, :, :]
                )

                block_data["Nd"].append(
                    block_data["Nxd"][-1][1:-1, :, :]
                    + block_data["Nyd"][-1][:, 1:-1, :]
                )

            else:
                raise NotImplementedError("3D transformations.")
                data = torch.permute(block.velocity[0], (1, 2, 3, 0))
                data = reduce_3D(data, size, axis3D, mode3D)
                # data = torch.permute(block.velocity[0,:,size.z//2,:,:], (1,2,0)) / max_mag

        else:
            raise NotImplementedError("Block has no transformation set.")
        # data = data * scale
        # block_data.append(data)

    for key, data in block_data.items():
        image = arrange_blocks(data, layout=layout, padding=padding, pad_col=pad_col)

        save_np_exr(
            image.cpu().detach().numpy(),
            os.path.join(
                path,
                "%s_%s_%04d.exr"
                % (
                    name,
                    key,
                    id,
                ),
            ),
        )


def save_domain_images(
    domain: _C.Domain,
    out_dir: str,
    it: int,
    layout: Layout = "H",
    norm_p: bool = True,
    max_mag: float = 1,
    mode3D: str | Sequence[str] = "mean",
    vel_exr: bool = False,
    vertex_coord_list: Sequence[torch.Tensor] | None = None,
    resampling_out_shape: ResamplingShape | None = 10,
    fill_max_steps: int = 0,
) -> None:
    """Save images of the scalar, pressure and velocity fields of a domain.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to visualize.
    out_dir : str
        Output directory.
    it : int
        Index appended to the file names.
    layout : Layout, optional
        Block arrangement, see :func:`arrange_blocks`. Default is ``"H"``.
    norm_p : bool, optional
        Whether to normalize the pressure to its range. Default is True.
    max_mag : float, optional
        Velocity magnitude mapped to full brightness. Default is 1.
    mode3D : str or Sequence of str, optional
        Reduction mode(s) in 3D, see :func:`reduce_3D`. In 3D, images are
        written for every mode and axis. Default is ``"mean"``.
    vel_exr : bool, optional
        Whether to also save the velocity as EXR. Default is False.
    vertex_coord_list : Sequence of torch.Tensor or None, optional
        Per-block vertex coordinates. If given together with
        ``resampling_out_shape``, additional resampled images are written.
        Default is None.
    resampling_out_shape : ResamplingShape or None, optional
        Shape of the uniform grid. Default is 10.
    fill_max_steps : int, optional
        Number of hole-filling iterations of the resampling. Default is 0.
    """
    ndims = domain.getSpatialDims()
    if ndims == 2:
        mode2D = cast(str, mode3D)  # unused for 2D data
        save_scalar_image(domain, out_dir, "d", it, layout=layout, mode3D=mode2D)
        save_pressure_image(domain, out_dir, "p", it, normalize=norm_p, layout=layout)
        save_velocity_image(domain, out_dir, "v", it, max_mag=max_mag, layout=layout)
        save_velocity_source_image(
            domain, out_dir, "vs", it, max_mag=max_mag, layout=layout
        )
        if vertex_coord_list is not None and resampling_out_shape is not None:
            save_scalar_image(
                domain,
                out_dir,
                "d-r",
                it,
                layout=layout,
                mode3D=mode2D,
                vertex_coord_list=vertex_coord_list,
                resampling_out_shape=resampling_out_shape,
                fill_max_steps=fill_max_steps,
            )
            save_pressure_image(
                domain,
                out_dir,
                "p-r",
                it,
                normalize=norm_p,
                layout=layout,
                mode3D=mode2D,
                vertex_coord_list=vertex_coord_list,
                resampling_out_shape=resampling_out_shape,
                fill_max_steps=fill_max_steps,
            )
            save_velocity_image(
                domain,
                out_dir,
                "v-r",
                it,
                max_mag=max_mag,
                layout=layout,
                mode3D=mode2D,
                vertex_coord_list=vertex_coord_list,
                resampling_out_shape=resampling_out_shape,
                fill_max_steps=fill_max_steps,
            )
            save_velocity_source_image(
                domain,
                out_dir,
                "vs-r",
                it,
                max_mag=max_mag,
                layout=layout,
                mode3D=mode2D,
                vertex_coord_list=vertex_coord_list,
                resampling_out_shape=resampling_out_shape,
                fill_max_steps=fill_max_steps,
            )
            if vel_exr:
                save_velocity_exr(
                    domain,
                    out_dir,
                    "vx-r",
                    it,
                    layout=layout,
                    mode3D=mode2D,
                    vertex_coord_list=vertex_coord_list,
                    resampling_out_shape=resampling_out_shape,
                    fill_max_steps=fill_max_steps,
                )
    elif ndims == 3:
        modes = mode3D if isinstance(mode3D, (list, tuple)) else [cast(str, mode3D)]
        for m3D in modes:
            for dim, name in ((0, "z"), (1, "y"), (2, "x")):
                save_scalar_image(
                    domain,
                    out_dir,
                    "d-%s-%s" % (name, m3D),
                    it,
                    layout=layout,
                    axis3D=dim,
                    mode3D=m3D,
                )
                save_pressure_image(
                    domain,
                    out_dir,
                    "p-%s-%s" % (name, m3D),
                    it,
                    normalize=norm_p,
                    layout=layout,
                    axis3D=dim,
                    mode3D=m3D,
                )
                save_velocity_image(
                    domain,
                    out_dir,
                    "v-%s-%s" % (name, m3D),
                    it,
                    max_mag=max_mag,
                    layout=layout,
                    axis3D=dim,
                    mode3D=m3D,
                )
                if vertex_coord_list is not None and resampling_out_shape is not None:
                    save_scalar_image(
                        domain,
                        out_dir,
                        "d-r-%s-%s" % (name, m3D),
                        it,
                        layout=layout,
                        axis3D=dim,
                        mode3D=m3D,
                        vertex_coord_list=vertex_coord_list,
                        resampling_out_shape=resampling_out_shape,
                        fill_max_steps=fill_max_steps,
                    )
                    save_pressure_image(
                        domain,
                        out_dir,
                        "p-r-%s-%s" % (name, m3D),
                        it,
                        normalize=norm_p,
                        layout=layout,
                        axis3D=dim,
                        mode3D=m3D,
                        vertex_coord_list=vertex_coord_list,
                        resampling_out_shape=resampling_out_shape,
                        fill_max_steps=fill_max_steps,
                    )
                    save_velocity_image(
                        domain,
                        out_dir,
                        "v-r-%s-%s" % (name, m3D),
                        it,
                        max_mag=max_mag,
                        layout=layout,
                        axis3D=dim,
                        mode3D=m3D,
                        vertex_coord_list=vertex_coord_list,
                        resampling_out_shape=resampling_out_shape,
                        fill_max_steps=fill_max_steps,
                    )
                    if vel_exr:
                        save_velocity_exr(
                            domain,
                            out_dir,
                            "vx-r-%s-%s" % (name, m3D),
                            it,
                            layout=layout,
                            mode3D=m3D,
                            vertex_coord_list=vertex_coord_list,
                            resampling_out_shape=resampling_out_shape,
                            fill_max_steps=fill_max_steps,
                        )
