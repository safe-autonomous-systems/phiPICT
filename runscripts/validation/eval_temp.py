"""Evaluate the temperature validation runs: field plots or Nusselt-number errors.

Usage:
    python runscripts/validation/eval_temp.py --case temp1 --mode plot
    python runscripts/validation/eval_temp.py --case temp2 --mode error
"""

import argparse
import logging
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.interpolate import RegularGridInterpolator

from fluidgym import DEFAULT_PALETTE
from fluidgym.simulation.helpers import get_cell_centers
from phipict.io.domain_io import load_domain as load_domain_from_file

logger = logging.getLogger("validation")

mpl.rcParams.update(
    {
        "text.usetex": True,  # or False if you're okay with mathtext
        "pdf.fonttype": 42,  # TrueType in PDF
        "ps.fonttype": 42,  # TrueType in PS/EPS
    }
)

sns.set_style(
    style="whitegrid",
    rc={
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.bottom": True,
        "ytick.left": True,
    },
)
sns.set_palette(DEFAULT_PALETTE)

PLOTS_DIR = Path("./paper/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR = Path("./paper/tables")
TABLES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path("./output")

PLOT_NAME = "nusselt_full.pdf"
MAX_POINTS = 20000

FIGWIDTH = 10.0
MAX_FIGHEIGHT = 9.0
# Cells per unit length used to resample the graded grid onto a uniform one.
RENDER_RESOLUTION = 128
# Arrow spacing on the page, in inches. The arrows keep a uniform physical
# spacing, so a tall domain gets proportionally more of them along its long axis.
ARROW_SPACING_INCH = 0.1
# Longest arrow, in multiples of the arrow spacing.
ARROW_LENGTH = 1.5
# Temperature range of the shared color bar.
T_RANGE = (-0.5, 0.5)

NO_MHD = "no_mhd"

# Reference Nusselt numbers.
REFERENCE_NUSSELT: dict[str, dict[float | str, float]] = {
    "temp1": {
        0.0: 3.80,
        162.0: 5.15,
        325.0: 4.44,
        487.0: 3.44,
        650.0: 3.15,
        796.0: 3.01,
    },
    "temp2": {
        NO_MHD: 3.331,
        0.0: 3.245,
        0.01: 3.046,
        0.1: 2.277,
        1.0: 1.527,
        50.0: 1.390,
    },
}


def reference_key(case: str, hartmann: float, cw: float) -> float | str:
    """Key into REFERENCE_NUSSELT for a run with the given Ha / Cw."""
    if case == "temp2":
        return NO_MHD if hartmann == 0.0 else cw
    return hartmann


def get_reference(case: str, hartmann: float, cw: float) -> float | None:
    """Reference Nusselt number for a run, or None if there is none."""
    return REFERENCE_NUSSELT.get(case, {}).get(reference_key(case, hartmann, cw))


def panel_sort_key(case: str, cfg: DictConfig) -> tuple[float, float]:
    """Ordering of the field panels: by Ha (temp1), by Cw with no-MHD first (temp2)."""
    hartmann = float(cfg.hartmann_number)
    cw = float(cfg.get("hartmann_Cw", 0.0))
    if case == "temp2":
        # The Ha = 0 run has no MHD at all and comes before every Cw.
        return (0.0, 0.0) if hartmann == 0.0 else (1.0, cw)
    return (hartmann, cw)


def panel_label(case: str, cfg: DictConfig) -> str:
    """Title of a field panel."""
    hartmann = float(cfg.hartmann_number)
    cw = float(cfg.get("hartmann_Cw", 0.0))
    if case == "temp2":
        if hartmann == 0.0:
            return "No MHD"
        return rf"$C_w = {cw:g}$"
    return rf"$\mathrm{{Ha}} = {hartmann:.0f}$"


def load_runs(case: str) -> dict[str, tuple[DictConfig, Path]]:
    """Collect config and directory for every run of the given case."""
    base_path = OUTPUT_DIR / "validation" / case

    runs: dict[str, tuple[DictConfig, Path]] = {}

    for param_dir in sorted(base_path.iterdir()):
        if not param_dir.is_dir():
            continue

        for grid_dir in sorted(param_dir.iterdir()):
            if not grid_dir.is_dir():
                continue

            if not (grid_dir / "logs.csv").exists():
                logger.warning(f"Logs not found for {param_dir.name} ({grid_dir.name}), skipping.")
                continue

            config_path = grid_dir / "config.yaml"
            if not config_path.exists():
                logger.warning(f"Config not found for {param_dir.name} ({grid_dir.name}), skipping.")
                continue

            key = f"{param_dir.name} ({grid_dir.name})"
            runs[key] = (OmegaConf.load(config_path), grid_dir)

    def sort_key(key_str: str) -> tuple[float, float, int]:
        parts = re.findall(r"([0-9.]+)_([0-9.]+) \((.+)\)", key_str)
        if parts:
            ha, cw, resolution = parts[0]
            return (float(ha), float(cw), grid_size(resolution))
        return (0.0, 0.0, 0)

    return {key: runs[key] for key in sorted(runs, key=sort_key)}


def grid_size(grid_str: str) -> int:
    """Total number of cells of a grid given as e.g. '64x480x64'."""
    total = 1
    for extent in grid_str.split("x"):
        total *= int(extent)
    return total


def largest_grid_runs(runs: dict[str, tuple[DictConfig, Path]]) -> dict[str, tuple[DictConfig, Path]]:
    """Keep only the run with the largest grid for each parameter directory."""
    best: dict[str, str] = {}
    for key, (_, grid_dir) in runs.items():
        param = grid_dir.parent.name
        if param not in best or grid_size(grid_dir.name) > grid_size(runs[best[param]][1].name):
            best[param] = key
    return {key: runs[key] for key in runs if key in best.values()}


def read_time_step(cfg: DictConfig) -> tuple[float, str]:
    """Simulation time step from the run config, if available."""
    dt = cfg.get("sim", {}).get("dt")
    if dt is not None:
        return float(dt), "Time"
    return 1.0, "Step"


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


def get_temperature_slice(cfg: DictConfig, run_dir: Path) -> dict[str, np.ndarray] | None:
    """Mid-x z-y slice of temperature and in-plane velocity, on a uniform grid.

    Mirrors the slicing of ``plot_domain`` in ``validation_temp.py``: the
    horizontal axis is z (hot wall at -z, cold wall at +z), the vertical axis
    is y (the buoyancy direction).
    """
    domain_path = run_dir / "final_domain"
    if not domain_path.with_suffix(".json").exists():
        logger.warning(f"Final domain not found in {run_dir}, skipping.")
        return None

    dtype = torch.float64 if cfg.precision == "double" else torch.float32
    domain = load_domain_from_file(domain_path, dtype=dtype, device=torch.device("cuda"))
    domain.PrepareSolve()
    block = domain.getBlocks()[0]

    centers = get_cell_centers(block.vertexCoordinates)  # [3, nz, ny, nx]
    z_centers = centers[2, :, 0, 0].cpu().numpy()  # [nz]
    y_centers = centers[1, 0, :, 0].cpu().numpy()  # [ny]

    T = block.passiveScalar[0, 0]  # [nz, ny, nx]
    velocity = block.velocity[0]  # [3, nz, ny, nx]
    mid_x = T.shape[-1] // 2

    render_shape = (
        int(cfg.domain.L * RENDER_RESOLUTION),
        int(cfg.domain.H * RENDER_RESOLUTION),
    )

    def to_uniform(field: torch.Tensor) -> np.ndarray:
        return resample(
            x_centers=z_centers,
            y_centers=y_centers,
            data=field[:, :, mid_x].cpu().numpy(),
            render_shape=render_shape,
        )

    return {
        "T": to_uniform(T),  # [res_z, res_y]
        "u_z": to_uniform(velocity[2]),  # [res_z, res_y]
        "u_y": to_uniform(velocity[1]),  # [res_z, res_y]
    }


def plot_temperature_fields(case: str, runs: dict[str, tuple[DictConfig, Path]]) -> None:
    """All temperature fields of a case side by side, with velocity quivers."""
    ordered = sorted(runs.items(), key=lambda item: panel_sort_key(case, item[1][0]))

    panels = []
    for case_name, (cfg, run_dir) in ordered:
        fields = get_temperature_slice(cfg, run_dir)
        if fields is None:
            continue
        panels.append((case_name, cfg, fields))

    if not panels:
        print(f"No final domains found for case '{case}'.")
        return

    n_panels = len(panels)
    cfg_0 = panels[0][1]
    aspect = float(cfg_0.domain.H) / float(cfg_0.domain.L)

    panel_width = FIGWIDTH / n_panels
    fig_height = panel_width * aspect
    fig_width = FIGWIDTH
    if fig_height > MAX_FIGHEIGHT:
        scale = MAX_FIGHEIGHT / fig_height
        fig_width *= scale
        fig_height = MAX_FIGHEIGHT

    fig, axs = plt.subplots(1, n_panels, figsize=(fig_width, fig_height), squeeze=False)
    axs = axs[0]

    # Arrows are spaced by a fixed distance on the page, so narrow panels do not
    # end up with a handful of them. The reference speed is shared across the
    # panels so that arrow lengths are comparable; the 99th percentile keeps a
    # single fast spot from shrinking every other arrow to a dot.
    ref_speed = max(
        float(np.percentile(np.hypot(fields["u_z"], fields["u_y"]), 99))
        for _, _, fields in panels
    )
    panel_width_inch = fig_width / n_panels
    n_arrows_z = max(4, int(panel_width_inch / ARROW_SPACING_INCH))
    arrow_spacing = float(cfg_0.domain.L) / n_arrows_z

    for ax, (case_name, cfg, fields) in zip(axs, panels):
        half_L = float(cfg.domain.L) / 2
        half_H = float(cfg.domain.H) / 2
        extent = (-half_L, half_L, -half_H, half_H)

        img = ax.imshow(
            fields["T"].T,  # [res_y, res_z], rows = y
            extent=extent,
            origin="lower",
            cmap="rainbow",
            vmin=T_RANGE[0],
            vmax=T_RANGE[1],
        )

        res_z, _ = fields["u_z"].shape
        stride = max(1, int(round(res_z / n_arrows_z)))
        qu = fields["u_z"][::stride, ::stride]
        qv = fields["u_y"][::stride, ::stride]
        qz = np.linspace(-half_L, half_L, qu.shape[0])
        qy = np.linspace(-half_H, half_H, qu.shape[1])
        QZ, QY = np.meshgrid(qz, qy, indexing="ij")
        ax.quiver(
            QZ,
            QY,
            qu,
            qv,
            color="black",
            alpha=1.0,
            angles="xy",
            scale_units="xy",
            scale=max(ref_speed, 1e-10) / (ARROW_LENGTH * arrow_spacing),
            width=0.006,
            pivot="mid",
        )

        ax.set_title(panel_label(case, cfg), fontsize=9)
        ax.set_aspect("equal")
        ax.set_frame_on(False)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        logger.info(f"Rendered panel for {case_name}.")

    cbar = fig.colorbar(
        img,
        ax=axs.tolist(),
        shrink=0.6,
        pad=0.02,
        ticks=np.linspace(T_RANGE[0], T_RANGE[1], 5),
    )
    cbar.set_label(r"$T$")

    out_path = PLOTS_DIR / f"{case}_temperature.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_nusselt(case: str, case_name: str, cfg: DictConfig, run_dir: Path) -> None:
    """Plot the concatenated warmup + collection Nusselt curve for one run."""
    frames = [
        pd.read_csv(run_dir / name)
        for name in ("logs_warmup.csv", "logs.csv")
        if (run_dir / name).exists()
    ]
    full_df = pd.concat(frames, ignore_index=True).sort_values("step")
    if full_df.empty:
        logger.warning(f"Empty logs for {case_name}, skipping plot.")
        return

    time_step, time_label = read_time_step(cfg)

    stride = max(1, len(full_df) // MAX_POINTS)
    plot_df = full_df.iloc[::stride]

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(
        plot_df["step"].to_numpy() * time_step,
        plot_df["nusselt"].to_numpy(),
        color="tab:blue",
        linewidth=1.0,
    )

    logs_df = pd.read_csv(run_dir / "logs.csv")
    if not logs_df.empty:
        ax.axvline(
            float(logs_df["step"].iloc[0]) * time_step,
            color="gray",
            linestyle="--",
            linewidth=0.8,
        )
        nu_mean = logs_df["nusselt"].mean()
        ax.axhline(
            nu_mean,
            color="tab:red",
            linestyle=":",
            linewidth=1.0,
            label=f"mean (collection phase) = {nu_mean:.4f}",
        )

    nu_ref = get_reference(case, float(cfg.hartmann_number), float(cfg.get("hartmann_Cw", 0.0)))
    if nu_ref is not None:
        ax.axhline(
            nu_ref,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=f"reference = {nu_ref:.4f}",
        )

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=8)

    ax.set_xlabel(time_label)
    ax.set_ylabel("Nu")
    ax.set_title(case_name, fontsize=9)
    sns.despine(ax=ax)
    fig.tight_layout()
    out_path = run_dir / PLOT_NAME
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"wrote {out_path}")


def compute_temp_error(case: str, runs: dict[str, tuple[DictConfig, Path]]) -> pd.DataFrame:
    """Average Nu over the collection phase and print/write the table."""
    result_rows = []

    for case_name, (cfg, run_dir) in runs.items():
        logs_df = pd.read_csv(run_dir / "logs.csv")
        if logs_df.empty:
            logger.warning(f"Empty logs for {case_name}, skipping.")
            continue

        hartmann = float(cfg.hartmann_number)
        cw = float(cfg.get("hartmann_Cw", 0.0))

        row = {"Ha": hartmann}
        if case == "temp2":
            row["Cw"] = cw
        nu = logs_df["nusselt"].mean()
        nu_ref = get_reference(case, hartmann, cw)
        row["Nu"] = nu
        row["Nu (ref)"] = nu_ref
        row["Error (%)"] = None if nu_ref is None else abs(nu - nu_ref) / nu_ref * 100

        result_rows += [row]

    result_df = pd.DataFrame(result_rows)
    sort_cols = ["Ha", "Cw"] if case == "temp2" else ["Ha"]
    result_df = result_df.sort_values(by=sort_cols)
    result_df.to_latex(TABLES_DIR / f"{case}_table.tex", index=False, float_format="%.3f")
    print(result_df.to_string(index=False, float_format="%.3f"))
    return result_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate temperature validation results.")
    parser.add_argument(
        "--case",
        type=str,
        required=True,
        choices=["temp1", "temp2"],
        help="Name of the validation case to evaluate.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["plot", "error"],
        help="Whether to plot the temperature fields or compute the Nusselt error.",
    )
    args = parser.parse_args()

    runs = load_runs(args.case)
    if not runs:
        print(f"No runs found for case '{args.case}'.")
        return

    if args.mode == "plot":
        plot_temperature_fields(args.case, largest_grid_runs(runs))
    else:
        for case_name, (cfg, run_dir) in runs.items():
            plot_nusselt(args.case, case_name, cfg, run_dir)
        compute_temp_error(args.case, largest_grid_runs(runs))


if __name__ == "__main__":
    main()