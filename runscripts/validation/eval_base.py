"""Evaluate the Hartmann, Shercliff and Hunt validation runs: profiles or errors.

Usage:
    python runscripts/validation/eval_base.py --case hartmann --mode plot
    python runscripts/validation/eval_base.py --case hunt --mode error
"""

import argparse
import logging
import re
import zipfile
from collections import OrderedDict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.interpolate import RegularGridInterpolator

from fluidgym import DEFAULT_PALETTE
from fluidgym.simulation.helpers import get_cell_centers, get_cell_size
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

        # "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.dpi": 500,
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
TABLES_DIR = Path("./paper/tables")
TABLES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path("./output")

DOMAIN_H = 1.0
DOMAIN_U = 1.0

BOUNDARY_THICKNESS = {
    20.0: (0.5, 0.2),
    100.0: (0.4, 0.07),
    300.0: (0.2, 0.01),
    1000.0: (0.08, 0.005),
}


Y_LIMIT_MIN = {
    20.0: -1.0,
    100.0: 0.8,
    1000.0: 0.99,
}

Z_LIMIT_MIN = {
    "shercliff": {
        20.0: -1.0,
        100.0: 0.5,
        1000.0: 0.8,
    },
    "hunt": {
        20.0: -1.0,
        100.0: 0.0,
        1000.0: 0.7,
    },
}

def get_hartmann_profile(Ha: float, y: np.ndarray) -> np.ndarray:
    if Ha == 0.0:
        # Poiseuille profile
        return 1.5 * DOMAIN_U * (1 - (y / DOMAIN_H) ** 2)

    # Hartmann profile
    return (
        DOMAIN_U
        * (np.cosh(Ha) - np.cosh(Ha * y / DOMAIN_H))
        / (np.cosh(Ha) - np.sinh(Ha) / Ha)
    )


def get_shercliff_profile(
    pressure_gradient: float,
    y: np.ndarray,
    z: np.ndarray,
    cfg: DictConfig,
    d_B: float = 0.0,
) -> np.ndarray:
    viscosity = (cfg.domain.U * cfg.domain.H) / cfg.reynolds_number
    a = cfg.domain.H
    aspect_ratio = 1.0
    n_steps = 1000

    YY, ZZ = np.meshgrid(y / a, z / a, indexing="ij")  # [NY, NZ], normalized to [-1, 1]

    u = np.zeros_like(YY)

    for k in range(n_steps):
        alpha_k = (k + 0.5) * np.pi / aspect_ratio
        N_k = np.sqrt(cfg.hartmann_number**2 + 4 * alpha_k**2)
        r1_k = 0.5 * (cfg.hartmann_number + N_k)
        r2_k = 0.5 * (-cfg.hartmann_number + N_k)

        # -------------------------------------------------------------------
        # V2 (Eq. 5.10)
        # -------------------------------------------------------------------
        V_2_k = (
            (d_B * r2_k + (1 - np.exp(-2 * r2_k)) / (1 + np.exp(-2 * r2_k)))
            * (np.exp(-r1_k * (1 - YY)) + np.exp(-r1_k * (1 + YY)))
            / 2
        ) / (
            (1 + np.exp(-2 * r1_k)) / 2 * d_B * N_k
            + (1 + np.exp(-2 * (r1_k + r2_k))) / (1 + np.exp(-2 * r2_k))
        )
        # -------------------------------------------------------------------
        # V3 (Eq. 5.11)
        # -------------------------------------------------------------------
        V_3_k = (
            (d_B * r1_k + (1 - np.exp(-2 * r1_k)) / (1 + np.exp(-2 * r1_k)))
            * (np.exp(-r2_k * (1 - YY)) + np.exp(-r2_k * (1 + YY)))
            / 2
        ) / (
            (1 + np.exp(-2 * r2_k)) / 2 * d_B * N_k
            + (1 + np.exp(-2 * (r1_k + r2_k))) / (1 + np.exp(-2 * r1_k))
        )

        # -------------------------------------------------------------------
        # V (Eq. 5.9)
        # -------------------------------------------------------------------
        V_k = (
            (2 * (-1) ** k * np.cos(alpha_k * ZZ))
            / (aspect_ratio * alpha_k**3)
            * (1 - V_2_k - V_3_k)
        )

        u += V_k

    # -----------------------------------------------------------------------
    # v_x (Eq. 5.8) (here x instead of z)
    # -----------------------------------------------------------------------
    u = viscosity**-1 * -pressure_gradient * a**2 * u

    return u


def resample(x_centers: np.ndarray, y_centers: np.ndarray, data: np.ndarray, render_shape: tuple[int, int]) -> np.ndarray:
    if data.shape != (len(x_centers), len(y_centers)):
        raise ValueError(f"Data shape {data.shape} does not match x_centers {len(x_centers)} and y_centers {len(y_centers)}")

    fn = RegularGridInterpolator((x_centers, y_centers), data, method="linear")

    # Create a uniform meshgrid for the target resolution
    x_half = min(abs(x_centers.min()), abs(x_centers.max()))
    y_half = min(abs(y_centers.min()), abs(y_centers.max()))
    uy = np.linspace(-x_half, x_half, render_shape[0])
    uz = np.linspace(-y_half, y_half, render_shape[1])

    # Generate the 3D grid points for evaluation
    pts = np.meshgrid(uy, uz, indexing="ij")
    pts_flat = np.array([p.flatten() for p in pts]).T

    # Resample
    uniform_data = fn(pts_flat).reshape(render_shape)
    return uniform_data


def load_domain_paths(case: str) -> tuple[
    dict[str, DictConfig],
    dict[str, str],
]:
    base_path = OUTPUT_DIR / "validation" / case

    configs = {}
    domain_paths = {}

    for ha_dir in base_path.iterdir():
        if not ha_dir.is_dir():
            continue

        for grid_dir in ha_dir.iterdir():
            if not grid_dir.is_dir():
                continue

            if not (grid_dir / "domain.json").exists():
                logger.warning(f"Domain not found for {ha_dir.name} ({grid_dir.name}), skipping.")
                continue

            config_path = grid_dir / "config.yaml"
            if not config_path.exists():
                logger.warning(f"Config not found for {ha_dir.name} ({grid_dir.name}), skipping.")
                continue

            cfg = OmegaConf.load(config_path)
            key = f"{ha_dir.name} ({grid_dir.name})"
            configs[key] = cfg
            domain_paths[key] = grid_dir / "domain"

    def sort_key(key_str: str) -> tuple[float, int]:
        parts = re.findall(r"([0-9.]+) \((.+)\)", key_str)
        if parts:
            num_val, resolution = parts[0]
            total_res = int(np.prod([int(x) for x in resolution.split("x")]))
            return (float(num_val), total_res)
        return (0.0, 0)

    sorted_keys = sorted(configs.keys(), key=sort_key)
    sorted_configs = OrderedDict((key, configs[key]) for key in sorted_keys)
    sorted_domain_paths = OrderedDict((key, domain_paths[key]) for key in sorted_keys)

    return sorted_configs, sorted_domain_paths


def get_numerical_profile(domain_path: Path, cfg: DictConfig) -> dict[str, np.ndarray]:
    dtype = torch.float64 if cfg.precision == "double" else torch.float32
    domain = load_domain_from_file(domain_path, dtype=dtype, device=torch.device("cuda"))
    domain.PrepareSolve()
    block = domain.getBlocks()[0]

    cell_centers = get_cell_centers(block.vertexCoordinates)

    u_x_profile: np.ndarray = block.velocity[0, 0, ...].cpu().numpy()
    u_x_profile_mean = u_x_profile.mean(axis=-1)

    if cfg.type == "hartmann":
        numerical_y = cell_centers[1, :, 0].cpu().numpy()  # [NY]
        return {
            "y": numerical_y,
            "u_x": u_x_profile_mean,  # [NY]
        }
    else:  # shercliff / hunt
        numerical_y = cell_centers[1, 0, :, 0].cpu().numpy()  # [NY]
        numerical_z = cell_centers[2, :, 0, 0].cpu().numpy()  # [NZ]
        cell_sizes = get_cell_size(block).cpu().numpy()
        cell_sizes = cell_sizes[0, 0, ..., 0].T  # [NY, NZ]
        return {
            "y": numerical_y,
            "z": numerical_z,
            "u_x": u_x_profile_mean.T,  # [NY, NZ]
            "cell_sizes": cell_sizes,  # [NY, NZ]
        }


def plot_hartmann(configs: dict[str, DictConfig], domain_paths: dict[str, str], num_points: int = 100) -> None:
    # Keep only the lowest resolution for each Hartmann number
    all_Ha: list[float] = []
    filtered_keys = []
    for key in list(configs.keys()):
        Ha = float(configs[key].hartmann_number)
        if Ha not in all_Ha:
            filtered_keys.append(key)
            all_Ha.append(Ha)

    fig, ax = plt.subplots(1, 1, figsize=(HALF_WIDTH, HALF_WIDTH))
    analytical_y = np.linspace(-DOMAIN_H, DOMAIN_H, num_points)

    for i, case_name in enumerate(filtered_keys):
        cfg = configs[case_name]
        profile = get_numerical_profile(domain_paths[case_name], cfg)
        numerical_y = profile["y"]
        u_x_profile_mean = profile["u_x"]

        Ha = float(cfg.hartmann_number)
        analytical_u_x = get_hartmann_profile(Ha, analytical_y)

        sns.lineplot(x=analytical_y, y=analytical_u_x, label=f"Ha={Ha}", ax=ax)
        sns.scatterplot(
            x=numerical_y,
            y=u_x_profile_mean,
            label="Numerical" if i == len(filtered_keys) - 1 else None,
            ax=ax,
            marker="x",
            color="black",
        )
    ax.legend().remove()
    ax.set_title("Analytical vs Numerical Velocity Profiles")
    ax.set_xlabel(r"$y / H$")
    ax.set_ylabel(r"$\langle u_x \rangle_y / U$")

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3, fancybox=False, shadow=False, frameon=False)
    fig.subplots_adjust(top=0.9, bottom=0.23, left=0.13, right=0.97)
    plt.savefig(PLOTS_DIR / "hartmann_profile.pdf", format="pdf", bbox_inches="tight")


def plot_shercliff(case: str, configs: dict[str, DictConfig], domain_paths: dict[str, str]) -> None:
    # Keep only the highest resolution for each Hartmann number
    case_keys = list(configs.keys())
    filtered_keys = []
    for key, next_key in zip(case_keys, case_keys[1:] + [None]):
        Ha = float(configs[key].hartmann_number)
        if next_key is None or float(configs[next_key].hartmann_number) != Ha:
            filtered_keys.append(key)

    n_cases = len(filtered_keys)
    fig, axs = plt.subplots(3, 2, figsize=(FULL_WIDTH, FULL_WIDTH), squeeze=False)

    ha_values = []
    for ax, case_name in zip(axs, filtered_keys):
        cfg = configs[case_name]
        profile = get_numerical_profile(domain_paths[case_name], cfg)
        numerical_y = profile["y"]
        numerical_z = profile["z"]
        numerical_u_x = profile["u_x"]
        cell_sizes = profile["cell_sizes"]

        Ha = float(cfg.hartmann_number)
        ha_values.append(Ha)

        ny, nz = 1000, 500
        if Ha >= 1000.0:
            ny = 4000

        y_min = Y_LIMIT_MIN[Ha]
        z_min = Z_LIMIT_MIN[cfg.type][Ha]

        analytical_y = np.linspace(-cfg.domain.H, cfg.domain.H, ny)
        analytical_z = np.linspace(-cfg.domain.H, cfg.domain.H, nz)

        analytical_u_x = get_shercliff_profile(
            pressure_gradient=cfg.pressure_gradient,
            y=analytical_y,
            z=analytical_z,
            cfg=cfg,
            d_B=cfg.get("hartmann_Cw", 0.0),
        )

        analytical_u_x_mean_y = analytical_u_x.mean(axis=1)  # [NY]
        analytical_u_x_mean_z = analytical_u_x.mean(axis=0)  # [NZ]

        numerical_u_x_mean_y = (numerical_u_x * cell_sizes).mean(axis=1) / cell_sizes.mean(axis=1)
        numerical_u_x_mean_z = (numerical_u_x * cell_sizes).mean(axis=0) / cell_sizes.mean(axis=0)

        sns.lineplot(x=analytical_y, y=analytical_u_x_mean_y, ax=ax[0], color=sns.color_palette()[0], label="Analytical")
        sns.scatterplot(x=numerical_y, y=numerical_u_x_mean_y, ax=ax[0], marker="x", color="black", label="Numerical")
        ax[0].legend().remove()
        ax[0].set_xlabel(r"$y / H$")
        ax[0].set_ylabel(r"$\langle u_x \rangle_y / U$")
        ax[0].set_xlim((y_min, 1.0))

        sns.lineplot(x=analytical_z, y=analytical_u_x_mean_z, ax=ax[1], color=sns.color_palette()[0], label="Analytical")
        sns.scatterplot(x=numerical_z, y=numerical_u_x_mean_z, ax=ax[1], marker="x", color="black", label="Numerical")
        ax[1].legend().remove()
        ax[1].set_xlabel(r"$z / H$")
        ax[1].set_ylabel(r"$\langle u_x \rangle_y / U$")
        ax[1].set_xlim((z_min, 1.0))

    # Add row labels with Hartmann numbers
    for i, ha in enumerate(ha_values):
        y_pos = 1 - (i + 0.5) / n_cases
        fig.text(0.01, y_pos, f"Ha = {ha:.0f}", ha="left", va="center", fontsize=10, transform=fig.transFigure, rotation=90)

    fig.suptitle(f"Analytical vs. Numerical Solution for {case.capitalize()} Case")

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.035), ncol=2, fancybox=False, shadow=False, frameon=False)
    fig.subplots_adjust(top=0.94, bottom=0.1, left=0.15, right=0.95, hspace=0.3, wspace=0.25)
    plt.savefig(PLOTS_DIR / f"{case}_profile.pdf", format="pdf")

def add_x_direction(
        ax: plt.Axes,
        center: tuple[float, float] = (-0.26, 0.18),
        radius: float = 0.028,
        x_into_page: bool = True,
    ) -> None:
    # Streamwise x-axis direction indicator (⊗ into page / ⊙ out of page)
    # Drawn outside the plot area on the left, above the B-field arrow
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    cx, cy = center
    circle = Circle(
        (cx, cy), radius, transform=ax.transAxes,
        fill=False, edgecolor="black", linewidth=1, clip_on=False,
    )
    ax.add_patch(circle)
    if x_into_page:
        # Cross (×) inside circle
        d = radius * 0.6
        for dx, dy in [(d, d), (-d, d)]:
            ax.add_line(Line2D(
                [cx - dx, cx + dx], [cy - dy, cy + dy],
                transform=ax.transAxes, color="black", lw=1, clip_on=False,
            ))
    else:
        # Dot inside circle
        dot = Circle(
            (cx, cy), radius * 0.25, transform=ax.transAxes,
            color="black", clip_on=False,
        )
        ax.add_patch(dot)
    ax.text(
        cx, cy - radius - 0.03, r"$x$",
        transform=ax.transAxes, fontsize=8, color="black",
        ha="center", va="top", clip_on=False,
    )

def add_B_direction(
    ax: plt.Axes,
    center: tuple[float, float] = (-0.18, 0.10),
    b_direction: tuple[float, float] = (1.0, 0.0),
    arrow_len: float = 0.15,
) -> None:
    cx, cy = center
    by, bz = b_direction
    norm = (by**2 + bz**2) ** 0.5
    by, bz = by / norm, bz / norm
    ax.annotate(
        r"$\mathbf{B}$",
        xy=(cx + bz * arrow_len, cy + by * arrow_len),
        xytext=(cx, cy),
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=8,
        color="black",
        ha="center",
        va="center",
        arrowprops=dict(
            arrowstyle="-|>",
            color="black",
            lw=1,
            mutation_scale=10,
        ),
    )

def plot_shercliff_velocity(
        case: str,
        configs: dict[str, DictConfig],
        domain_paths: dict[str, str],
        render_shape: tuple[int, int] = (513, 513),
        cmap: str = "RdBu_r"
    ) -> None:
    # Keep only the highest resolution for each Hartmann number
    case_keys = list(configs.keys())
    filtered_keys = []
    for key, next_key in zip(case_keys, case_keys[1:] + [None]):
        Ha = float(configs[key].hartmann_number)
        if next_key is None or float(configs[next_key].hartmann_number) != Ha:
            filtered_keys.append(key)

    n_cases = len(filtered_keys)
    fig, axs = plt.subplots(1, n_cases, figsize=(FULL_WIDTH, 2.2))

    for ax, case_name in zip(axs, filtered_keys):
        cfg = configs[case_name]
        profile = get_numerical_profile(domain_paths[case_name], cfg)

        Ha = float(cfg.hartmann_number)

        resampled_data = resample(
            x_centers=profile["z"],       # [NZ]
            y_centers=profile["y"],       # [NY]
            data=profile["u_x"].T,        # [NZ, NY]
            render_shape=render_shape,
        )
        resampled_data = resampled_data.T
        resampled_data = (resampled_data - resampled_data.min()) / (resampled_data.max() - resampled_data.min() + 1e-10)
        img = sns.color_palette(cmap, as_cmap=True)(resampled_data, bytes=True)[:, :, :3]

        ax.imshow(img, extent=(-1.0, 1.0, -1.0, 1.0), origin="lower")
        ax.set_aspect("equal")
        ax.set_frame_on(False)
        ax.set_title(f"Ha = {float(Ha):.0f}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r"$z / H$")
        ax.set_ylabel(r"$y / H$")

    add_x_direction(axs[0], center=(-0.2, 0.0), x_into_page=True)
    add_B_direction(axs[0], center=(-0.1, -0.085), b_direction=(1.0, 0.0))

    plt.suptitle(case.capitalize() + ": " + r"Velocity Magnitude $\langle u_x \rangle_y / U$")
    fig.subplots_adjust(top=0.81, bottom=0.08, left=0.0, right=0.95, hspace=0.3, wspace=0.2)
    plt.savefig(PLOTS_DIR / f"{case}_velocity.pdf", format="pdf", bbox_inches="tight")

def plot_shercliff_velocity_3d(
        case: str,
        configs: dict[str, DictConfig],
        domain_paths: dict[str, str],
        render_shape: tuple[int, int] = (512, 512), # Lower res often looks better in 3D
        cmap: str = "rainbow"
    ) -> None:

    # Keep only the highest resolution for each Hartmann number
    case_keys = list(configs.keys())
    filtered_keys = []
    for key, next_key in zip(case_keys, case_keys[1:] + [None]):
        Ha = float(configs[key].hartmann_number)
        if next_key is None or float(configs[next_key].hartmann_number) != Ha:
            filtered_keys.append(key)

    n_cases = len(filtered_keys)

    # Initialize figure with 3D projection
    fig = plt.figure(figsize=(FULL_WIDTH, 2.2))


    for i, case_name in enumerate(filtered_keys):
        # Add 3D subplot manually
        ax = fig.add_subplot(1, n_cases, i + 1, projection="3d")

        cfg = configs[case_name]
        profile = get_numerical_profile(domain_paths[case_name], cfg)
        Ha = float(cfg.hartmann_number)

        Z_raw = profile["z"]  #
        Y_raw = profile["y"]
        U_raw = profile["u_x"]  # [NY, NZ]

        # The cases are driven by different pressure gradients, so their velocity
        # magnitudes live on different scales. Rescale each profile by its own bulk
        # (cell-size weighted) mean velocity so every surface has <u_x> = 1
        cell_sizes = profile["cell_sizes"]  # [NY, NZ]
        u_bulk = np.sum(U_raw * cell_sizes) / np.sum(cell_sizes)
        U_raw = U_raw / u_bulk

        # The outermost cell centers sit half a cell away from the walls, so the
        # surface would stop short of them. Append the walls explicitly, where the
        # no-slip condition gives u = 0
        Y_raw = np.concatenate(([-DOMAIN_H], Y_raw, [DOMAIN_H]))
        Z_raw = np.concatenate(([-DOMAIN_H], Z_raw, [DOMAIN_H]))
        U_raw = np.pad(U_raw, ((1, 1), (1, 1)), mode="constant", constant_values=0.0)

        # 2. Create the meshgrid from the actual (possibly non-uniform) coordinates
        # This ensures the surface 'bunches up' points where the resolution is high
        Z, Y = np.meshgrid(Z_raw, Y_raw, indexing="ij")
        U = U_raw.T

        # Plot the surface
        surf = ax.plot_surface(
            Z, Y, U,
            cmap=cmap,
            linewidth=0,
            antialiased=False,
            rcount=render_shape[0], ccount=render_shape[1]
        )
        # Vector PDF draws every quad separately; without matching edges the
        # background bleeds through the seams as gray hairlines
        surf.set_edgecolor("face")
        surf.set_rasterized(True)

        # Matching the reference image style
        ax.set_title(f"Ha = {float(Ha):.0f}")
        ax.set_xlabel(r"$z / H$")
        ax.set_ylabel(r"$y / H$")
        ax.set_zlabel(r"$u_x / U$", rotation=180, labelpad=0)

        # Set limits and view angle
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.view_init(elev=30, azim=-135)

        # On 3D axes `axes.facecolor` fills the whole subplot rectangle instead of
        # the cube faces, so we clear it and paint the three panes individually
        ax.patch.set_alpha(0.0)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color(PANE_COLOR)
            axis.pane.set_edgecolor("white")

    plt.suptitle(case.capitalize() + ": " + r"3D Velocity Profile", y=0.95)

    fig.subplots_adjust(top=0.78, bottom=0.18, left=0.11, right=0.95, wspace=0.5)

    plt.savefig(PLOTS_DIR / f"{case}_velocity_3d.pdf", format="pdf", bbox_inches=None)

def compute_hartmann_error(
        configs: dict[str, DictConfig],
        domain_paths: dict[str, str],
    ) -> None:
    result_rows = []

    for case_name, cfg in configs.items():
        profile = get_numerical_profile(domain_paths[case_name], cfg)
        numerical_y = profile["y"]
        u_x_profile_mean = profile["u_x"]

        Ha = float(cfg.hartmann_number)
        analytical_u_x = get_hartmann_profile(Ha, numerical_y)

        error = np.mean(np.abs(u_x_profile_mean - analytical_u_x)) / np.max(np.abs(analytical_u_x))

        grid_str = case_name.split("(")[-1].rstrip(")")
        nx, ny = map(int, grid_str.split("x"))

        result_rows += [{
            "Ha": Ha,
            "N_x": nx,
            "N_y": ny,
            "E_y": cfg.domain.grading_y,
            "Error (%)": error * 100,
        }]

    result_df = pd.DataFrame(result_rows)
    result_df = result_df.sort_values(by=["Ha", "N_x", "N_y", "E_y"])
    result_df.to_latex(TABLES_DIR / "hartmann_error_table.tex", index=False, float_format="%.2f")
    print(result_df)


def compute_shercliff_error(
        case: str,
        configs: dict[str, DictConfig],
        domain_paths: dict[str, str],
    ) -> None:
    result_rows = []

    for case_name, cfg in configs.items():
        try:
            profile = get_numerical_profile(domain_paths[case_name], cfg)
        except (zipfile.BadZipFile, EOFError):
            continue

        numerical_y = profile["y"]
        numerical_z = profile["z"]
        numerical_u_x_2d = profile["u_x"]
        cell_sizes = profile["cell_sizes"]

        Ha = float(cfg.hartmann_number)
        analytical_u_x_2d = get_shercliff_profile(
            pressure_gradient=cfg.pressure_gradient,
            y=numerical_y,
            z=numerical_z,
            cfg=cfg,
            d_B=cfg.get("hartmann_Cw", 0.0),
        )

        # 1) Velocity error — normalize by max analytical velocity
        error_u = np.mean(
            np.abs(numerical_u_x_2d - analytical_u_x_2d)
        ) / np.max(np.abs(analytical_u_x_2d)) * 100

        # 2) Weighted RMSE using 2D cell sizes — normalize by RMS of analytical solution
        rms_analytical = np.sqrt(
            np.sum(cell_sizes * analytical_u_x_2d**2) / np.sum(cell_sizes)
        )
        error_rmse = np.sqrt(
            np.sum(cell_sizes * (numerical_u_x_2d - analytical_u_x_2d) ** 2)
            / np.sum(cell_sizes)
        ) / rms_analytical * 100

        grid_str = case_name.split("(")[-1].rstrip(")")
        nx, ny, nz = map(int, grid_str.split("x"))

        result_rows += [{
            "Ha": Ha,
            "Nx": nx,
            "Ny": ny,
            "Nz": nz,
            "E_y": cfg.domain.grading_y,
            "E_z": cfg.domain.grading_z,
            "error_u (%)": error_u,
            "error_rmse (%)": error_rmse,
        }]

    result_df = pd.DataFrame(result_rows)
    result_df["Ha"] = result_df["Ha"].astype(float)
    result_df = result_df.sort_values(by=["Ha", "Nx", "Ny", "Nz", "E_y", "E_z"])
    result_df.to_latex(TABLES_DIR / f"{case}_error_table.tex", index=False, float_format="%.2f")
    print(result_df)


def main():
    parser = argparse.ArgumentParser(description="Evaluate simulation results.")
    parser.add_argument(
        "--case",
        type=str,
        required=True,
        choices=["hartmann", "shercliff", "hunt"],
        help="Name of the validation case to evaluate (e.g., 'hartmann', 'shercliff').",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["plot", "error"],
        help="Whether to plot or compute RMSE.",
    )
    args = parser.parse_args()

    configs, domain_paths = load_domain_paths(args.case)
    if args.mode == "plot":
        if args.case == "hartmann":
            plot_hartmann(configs, domain_paths)
        elif args.case == "shercliff" or args.case == "hunt":
            plot_shercliff(args.case, configs, domain_paths)
            # plot_shercliff_velocity(args.case, configs, domain_paths)
            plot_shercliff_velocity_3d(args.case, configs, domain_paths)
    else:
        if args.case == "shercliff" or args.case == "hunt":
            compute_shercliff_error(args.case, configs, domain_paths)
        else:
            compute_hartmann_error(configs, domain_paths)


if __name__ == "__main__":
    main()
