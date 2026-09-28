"""Evaluate the duct validation run: velocity field and streamwise profile figures.

Usage:
    python runscripts/validation/eval_duct.py
"""

import logging
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.interpolate import RegularGridInterpolator

from fluidgym import DEFAULT_PALETTE
from fluidgym.simulation.helpers import get_cell_centers
from phipict.io.domain_io import load_domain as load_domain_from_file

logger = logging.getLogger("validation")

FULL_WIDTH = 5.5
HALF_WIDTH = 0.49 * FULL_WIDTH
PANE_COLOR = "#F2F4F7"

mpl.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",

        "pdf.fonttype": 42,  # TrueType in PDF
        "ps.fonttype": 42,  # TrueType in PS/EPS

        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 10,

        # Clean aesthetics
        "figure.autolayout": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

sns.set_style(
    style="darkgrid",
    rc={
        "axes.facecolor": PANE_COLOR,
    },
)
sns.set_palette(DEFAULT_PALETTE)

PLOTS_DIR = Path("./paper/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path("./output")

# Corners of the zoomed-in region, in (x, y) domain coordinates. Only the
# bounding box is used, so the order of the four points does not matter
ZOOM_CORNERS: list[tuple[float, float]] = [
    (0.20, 0.80),
    (1.20, 0.80),
    (1.20, 1.00),
    (0.20, 1.00),
]

# Size of the inset, as a fraction of the width of the main axes. The height
# follows from the aspect of the zoom region so the inset is not distorted
ZOOM_WIDTH = 0.55
# Vertical gap between the main axes and the inset, in axes fractions
ZOOM_GAP = 0.18

# (y, z) location of the line plot. The default sits inside the side layer at
# the +y Shercliff wall, on the z = 0 mid-plane
LINE_Y = 0.96
LINE_Z = 0.0

# Cells per unit length used to resample the graded grid onto a uniform one
RENDER_RESOLUTION_X = 128
RENDER_RESOLUTION_Y = 1024

CMAP = "rainbow"


def resample(
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    data: np.ndarray,
    render_shape: tuple[int, int],
) -> np.ndarray:
    if data.shape != (len(x_centers), len(y_centers)):
        raise ValueError(
            f"Data shape {data.shape} does not match x_centers {len(x_centers)} "
            f"and y_centers {len(y_centers)}"
        )

    fn = RegularGridInterpolator((x_centers, y_centers), data, method="linear")

    ux = np.linspace(x_centers.min(), x_centers.max(), render_shape[0])
    uy = np.linspace(y_centers.min(), y_centers.max(), render_shape[1])

    pts = np.meshgrid(ux, uy, indexing="ij")
    pts_flat = np.array([p.flatten() for p in pts]).T

    return fn(pts_flat).reshape(render_shape)


def grid_size(grid_str: str) -> int:
    """Total number of cells of a grid given as e.g. '300x70x100_full'."""
    total = 1
    for extent in grid_str.split("_")[0].split("x"):
        total *= int(extent)
    return total


def load_run() -> tuple[DictConfig, Path]:
    """Config and directory of the duct run with the largest grid."""
    base_path = OUTPUT_DIR / "validation" / "duct"

    candidates: list[tuple[DictConfig, Path]] = []
    for ha_dir in sorted(base_path.iterdir()):
        if not ha_dir.is_dir():
            continue

        for grid_dir in sorted(ha_dir.iterdir()):
            if not grid_dir.is_dir():
                continue

            if not (grid_dir / "domain.json").exists():
                logger.warning(f"Domain not found in {grid_dir}, skipping.")
                continue

            config_path = grid_dir / "config.yaml"
            if not config_path.exists():
                logger.warning(f"Config not found in {grid_dir}, skipping.")
                continue

            candidates.append((OmegaConf.load(config_path), grid_dir))

    if not candidates:
        raise FileNotFoundError(f"No duct runs with a saved domain found in {base_path}.")

    return max(candidates, key=lambda item: grid_size(item[1].name))


def get_fields(cfg: DictConfig, run_dir: Path) -> dict[str, np.ndarray]:
    """Mid-z x-y slice of the streamwise velocity, plus the raw cell centers."""
    dtype = torch.float64 if cfg.precision == "double" else torch.float32
    domain = load_domain_from_file(
        run_dir / "domain", dtype=dtype, device=torch.device("cuda")
    )
    domain.PrepareSolve()
    block = domain.getBlocks()[0]

    centers = get_cell_centers(block.vertexCoordinates)  # [3, nz, ny, nx]
    x_centers = centers[0, 0, 0, :].cpu().numpy()  # [nx]
    y_centers = centers[1, 0, :, 0].cpu().numpy()  # [ny]
    z_centers = centers[2, :, 0, 0].cpu().numpy()  # [nz]

    u_x = block.velocity[0, 0].cpu().numpy()  # [nz, ny, nx]

    mid_z = int(np.argmin(np.abs(z_centers - LINE_Z)))
    u_x_slice = u_x[mid_z]  # [ny, nx]

    render_shape = (
        int((x_centers.max() - x_centers.min()) * RENDER_RESOLUTION_X),
        int((y_centers.max() - y_centers.min()) * RENDER_RESOLUTION_Y),
    )
    u_x_uniform = resample(
        x_centers=x_centers,
        y_centers=y_centers,
        data=u_x_slice.T,  # [nx, ny]
        render_shape=render_shape,
    ).T  # [res_y, res_x]

    iy = int(np.argmin(np.abs(y_centers - LINE_Y)))

    return {
        "u_x": u_x_uniform,
        "extent": np.array(
            [x_centers.min(), x_centers.max(), y_centers.min(), y_centers.max()]
        ),
        "x_centers": x_centers,
        "u_x_line": u_x_slice[iy],  # [nx]
        "line_y": np.array(y_centers[iy]),
        "line_z": np.array(z_centers[mid_z]),
    }


def plot_velocity(cfg: DictConfig, fields: dict[str, np.ndarray]) -> None:
    """Streamwise velocity field with a zoom inset on one of the jets."""
    x_min, x_max, y_min, y_max = (float(v) for v in fields["extent"])
    extent = (x_min, x_max, y_min, y_max)

    zoom_x = [corner[0] for corner in ZOOM_CORNERS]
    zoom_y = [corner[1] for corner in ZOOM_CORNERS]
    zoom_extent = (min(zoom_x), max(zoom_x), min(zoom_y), max(zoom_y))

    fig_height = FULL_WIDTH * (y_max - y_min) / (x_max - x_min)
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, fig_height))

    im = ax.imshow(
        fields["u_x"],
        extent=extent,
        origin="lower",
        cmap=sns.color_palette(CMAP, as_cmap=True),
    )
    ax.set_aspect(1.0)
    ax.grid(False)
    ax.set_xlabel(r"$x / H$")
    ax.set_ylabel(r"$y / H$")

    # The inset keeps the aspect of the main axes: a region of the field covers
    # a fraction of the axes box proportional to its data extent, so matching
    # those fractions leaves the zoomed field undistorted
    frac_x = (zoom_extent[1] - zoom_extent[0]) / (x_max - x_min)
    frac_y = (zoom_extent[3] - zoom_extent[2]) / (y_max - y_min)
    zoom_height = ZOOM_WIDTH * frac_y / frac_x

    axins = ax.inset_axes(
        [0.5 - ZOOM_WIDTH / 2, -ZOOM_GAP - zoom_height, ZOOM_WIDTH, zoom_height],
        xlim=(zoom_extent[0], zoom_extent[1]),
        ylim=(zoom_extent[2], zoom_extent[3]),
        xticks=[],
        yticks=[],
    )
    axins.imshow(
        fields["u_x"],
        extent=extent,
        origin="lower",
        cmap=sns.color_palette(CMAP, as_cmap=True),
        vmin=im.get_clim()[0],
        vmax=im.get_clim()[1],
    )
    axins.set_aspect("auto")
    axins.grid(False)
    for spine in axins.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")

    ax.indicate_inset_zoom(axins, edgecolor="black", alpha=1.0, linewidth=1.0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label(r"$u_x / U$")

    ax.set_title("Jet Detachments in a Duct Flow with " + r"$\mathrm{Ha} = " + f"{float(cfg.hartmann_number):.0f}$", fontsize=10)

    out_path = PLOTS_DIR / "duct_velocity.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_u_x_over_x(fields: dict[str, np.ndarray]) -> None:
    """Streamwise velocity along the duct at a fixed (y, z)."""
    y = float(fields["line_y"])
    z = float(fields["line_z"])

    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.6, 3.0))
    ax.plot(
        fields["x_centers"],
        fields["u_x_line"],
        color=sns.color_palette()[0],
        linewidth=1.2,
    )
    ax.set_xlabel(r"$x / H$")
    ax.set_ylabel(r"$u_x / U$")
    ax.set_title(rf"$y / H = {y:.3f}$, $z / H = {z:.3f}$", fontsize=10)
    ax.set_xlim(float(fields["x_centers"].min()), float(fields["x_centers"].max()))
    sns.despine(ax=ax)
    fig.tight_layout()

    out_path = PLOTS_DIR / "duct_u_x_over_x.pdf"
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    cfg, run_dir = load_run()
    logger.info(f"Evaluating duct run in {run_dir}.")

    fields = get_fields(cfg, run_dir)
    plot_velocity(cfg, fields)
    plot_u_x_over_x(fields)


if __name__ == "__main__":
    main()