"""Run the buoyant MHD validation cases and log the Nusselt number.

Usage:
    python runscripts/validation/validation_temp.py case=temp1_162
"""

import json
from pathlib import Path
import logging
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.interpolate import RegularGridInterpolator
from phipict import Face, Hook, bc

import fluidgym.simulation.pict.data.shapes as shapes
from fluidgym import DEFAULT_PALETTE
from fluidgym.envs.util.grid_gen import make_weights
from fluidgym.simulation.extensions import PISOtorch  # type: ignore
from fluidgym.simulation.helpers import get_cell_centers, get_cell_size
from fluidgym.simulation.mhd_simulation import MHDSimulation
from phipict.io.domain_io import load_domain, save_domain
from fluidgym.simulation.pict.util.output import plot_grid
from fluidgym.simulation.sgs import append_sgs_viscosity_prep_fn
from fluidgym.simulation.simulation import Simulation
from fluidgym.simulation.solver_tolerance import SolverTolerance

OmegaConf.register_new_resolver("eval", lambda x: eval(x))


logger = logging.getLogger("validation")

sns.set_style("whitegrid")
sns.set_palette(DEFAULT_PALETTE)


FIGWIDTH = 10.0

BOUNDARY_THICKNESS = {
    20.0: (0.5, 0.2),
    100.0: (0.4, 0.07),
    300.0: (0.2, 0.01),
    1000.0: (0.08, 0.005),
}

META_FILENAME = "simulation.json"

def save_metadata(data: dict[str, Any]) -> None:
    with open(META_FILENAME, "w") as f:
        json.dump(data, f, indent=4)

def load_metadata() -> dict[str, Any]:
    with open(META_FILENAME) as f:
        data = json.load(f)
    return dict(data)


def make_domain(
    cfg: DictConfig, dtype: torch.dtype, device: torch.device
) -> tuple[PISOtorch.Domain, dict[str, Any]]:
    reynolds_number = np.sqrt(cfg.grashof_number)

    x_weights = make_weights(
        grading_type=cfg.domain.grading_x_type,
        res=cfg.domain.nx,
        grading=cfg.domain.grading_x,
        refinement="BOTH",
    )

    y_weights = make_weights(
        grading_type=cfg.domain.grading_y_type,
        res=cfg.domain.ny,
        grading=cfg.domain.grading_y,
        refinement="BOTH",
    )

    # See Tagawa et al. (2002). L is the characteristic length
    half_width = cfg.domain.L / 2
    half_height = cfg.domain.H / 2

    # We generate a L x H grid in x/y
    grid = shapes.generate_grid_vertices_2D(
        [cfg.domain.ny + 1, cfg.domain.nx + 1],
        [
            (-half_width, -half_height),
            (half_width, -half_height),
            (-half_width, half_height),
            (half_width, half_height),
        ],
        None,
        x_weights=y_weights,
        y_weights=x_weights,
        dtype=dtype,
    )

    # Plot the x/y grid before extrusion (Hartmann walls at ±x, B field in x)
    fig, ax = plt.subplots(figsize=(2.0, 2.0 * cfg.domain.H / cfg.domain.L))
    plot_grid(grid, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.savefig("grid_x-y.pdf", format="pdf")
    plt.close(fig)

    z_weights = make_weights(
        grading_type=cfg.domain.grading_z_type,
        res=cfg.domain.nz,
        grading=cfg.domain.grading_z,
        refinement="BOTH",
    )

    grid = shapes.extrude_grid_z(
        grid,
        res_z=cfg.domain.nz,
        start_z=-half_width,
        end_z=half_width,
        weights_z=z_weights,
    )

    # Plot x-z slice (Shercliff walls at ±x, z is the extrusion/spanwise direction)
    fig, ax = plt.subplots(figsize=(6, 4))
    grid_x_z = torch.stack([grid[:, 0, :, 0, :], grid[:, 2, :, 0, :]], dim=1)
    plot_grid(grid_x_z, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect("equal")
    plt.savefig("grid_x-z.pdf", format="pdf")
    plt.close(fig)

    grid = grid.to(device).contiguous()
    kinematic_viscosity = torch.tensor(
        [(cfg.domain.U * cfg.domain.L) / reynolds_number],
        dtype=dtype,
        device=device,
    )

    domain = PISOtorch.Domain(
        3,
        kinematic_viscosity.cpu(),
        name="Domain",
        device=device,
        dtype=dtype,
    )

    block = domain.CreateBlock(vertexCoordinates=grid, name="Block")

    # We close all walls
    block.CloseBoundary("-x")
    block.CloseBoundary("+x")
    block.CloseBoundary("-y")
    block.CloseBoundary("+y")
    block.CloseBoundary("-z")
    block.CloseBoundary("+z")

    # Finite wall conductivity, same Cw on all six walls. Cw = 0 leaves the
    # default insulating (Neumann) condition
    hartmann_Cw = float(cfg.get("hartmann_Cw", 0.0))
    if hartmann_Cw > 0.0:
        for face in Face:
            bc.set_bc(block, face, bc.Potential.ThinWall(cw=hartmann_Cw))

    # Temperature field. alpha = nu/Pr, so it carries the same L as the viscosity
    # above. Identical for the L = 1 configs, but Pr would be off by L otherwise
    thermal_diffusivity = torch.tensor(
        [(cfg.domain.U * cfg.domain.L) / (reynolds_number * cfg.prandtl_number)],
        dtype=dtype,
        device=device,
    )
    domain.setScalarViscosity(thermal_diffusivity)

    block.getBoundary("-z").setPassiveScalar(
        torch.tensor([[cfg.domain.T_hot]], dtype=dtype, device=device)
    )
    block.getBoundary("+z").setPassiveScalar(
        torch.tensor([[cfg.domain.T_cold]], dtype=dtype, device=device)
    )
    block.getBoundary("-x").setPassiveScalarType(PISOtorch.BoundaryConditionType.NEUMANN)
    block.getBoundary("+x").setPassiveScalarType(PISOtorch.BoundaryConditionType.NEUMANN)
    block.getBoundary("-y").setPassiveScalarType(PISOtorch.BoundaryConditionType.NEUMANN)
    block.getBoundary("+y").setPassiveScalarType(PISOtorch.BoundaryConditionType.NEUMANN)

    buoyancy_factor = cfg.grashof_number / reynolds_number**2
    vel_src_velX_pad = torch.zeros_like(block.passiveScalar)

    T_ref = (cfg.domain.T_hot + cfg.domain.T_cold) / 2

    def buoyancy_fn_3d(domain, time_step, **kwargs):
        T = domain.getBlocks()[0].passiveScalar
        source = torch.cat(
            [vel_src_velX_pad, (T - T_ref) * buoyancy_factor, vel_src_velX_pad], dim=1
        )
        block.setVelocitySource(source)
        domain.UpdateDomainData()

    prep_fn: dict[str, Any] = {
        Hook.PRE_VELOCITY_SETUP: [buoyancy_fn_3d],
    }

    # Sets per-cell block viscosity to nu_SGS + nu before the momentum matrix is
    # assembled. Affects momentum only; the temperature keeps its molecular
    # diffusivity, so the Prandtl number is unchanged.
    append_sgs_viscosity_prep_fn(
        prep_fn,
        model=cfg.get("sgs_model", "none"),
        coefficient=cfg.get("sgs_coefficient", 0.325),
        ndims=3,
        dtype=dtype,
        cuda_device=device,
        cpu_device=torch.device("cpu"),
    )

    if cfg.get("init_with_grad", False):
        from fluidgym.simulation.extensions import (
            SimplexNoiseVariations,  # type: ignore[import-untyped,import-not-found]
        )

        curl_noise = SimplexNoiseVariations.GenerateSimplexNoiseVariation(
            (cfg.domain.nx, cfg.domain.ny, cfg.domain.nz),
            device,
            [2 / s for s in block.velocity.shape[2:]],
            [0] * 3,
            SimplexNoiseVariations.NoiseVariation.SIMPLEX,
        )
        curl_noise *= 0.01
        block.setVelocity(block.velocity + curl_noise)

        # Temperature gradient
        cell_centers = get_cell_centers(block.vertexCoordinates)  # [3, NZ, NY, NX]
        z_centers = cell_centers[2, :, 0, 0]  # [NZ]
        z_min, z_max = z_centers[0], z_centers[-1]
        T_grad = cfg.domain.T_hot + (cfg.domain.T_cold - cfg.domain.T_hot) * (z_centers - z_min) / (z_max - z_min)
        T_grad = T_grad[None, None, :, None, None].expand(
            1, 1, cfg.domain.nz, cfg.domain.ny, cfg.domain.nx
        ).contiguous()

        # Add noise
        temp_noise = torch.randn_like(T_grad) * 0.01 * (cfg.domain.T_hot - cfg.domain.T_cold)
        T_grad += temp_noise

        block.setPassiveScalar(T_grad)

    domain.PrepareSolve()

    return domain, prep_fn

def set_advection_scheme(cfg: DictConfig, domain: PISOtorch.Domain) -> None:
    """Select the convective scheme for the momentum equation."""
    scheme = cfg.get("advection_scheme", "central")
    try:
        domain.setAdvectionScheme(getattr(PISOtorch.AdvectionScheme, scheme.upper()))
    except AttributeError as e:
        raise ValueError(
            f"Unknown advection_scheme '{scheme}'. Expected one of: "
            "central, linear_upwind."
        ) from e


def solver_tolerance(cfg: DictConfig, name: str) -> float | SolverTolerance | None:
    """Resolve ``sim.<name>_tol`` into the form the simulation expects.

    ``absolute`` (the default) hands the value straight to the solver, which is
    the historical behaviour and stays bit-identical. ``relative`` wraps it as a
    :class:`SolverTolerance`, turning the stop test into
    ``||r||_2 < rtol * ||b||_2``.
    """
    value = cfg.sim.get(f"{name}_tol", None)
    if value is None:
        return None

    mode = cfg.sim.get(f"{name}_tol_mode", None) or cfg.sim.get("tol_mode", "absolute")
    mode = str(mode).lower()
    if mode == "absolute":
        return float(value)
    if mode != "relative":
        raise ValueError(
            f"sim.{name}_tol_mode must be 'absolute' or 'relative', got {mode!r}."
        )

    atol = cfg.sim.get("tol_atol", None)
    return SolverTolerance(
        rtol=float(value), atol=None if atol is None else float(atol)
    )


def log_tolerances(cfg: DictConfig, names: tuple[str, ...]) -> None:
    for name in names:
        logger.info("%s_tol: %s", name, solver_tolerance(cfg, name))


def make_simulation(
    cfg: DictConfig, domain: PISOtorch.Domain, prep_fn: dict[str, Any]
) -> Simulation:
    if cfg.hartmann_number > 0.0:
        raise ValueError("Hartmann number must be = 0 for standard simulation.")

    set_advection_scheme(cfg, domain)
    log_tolerances(cfg, ("advection", "pressure", "pressure_intermediate"))

    sim = Simulation(
        domain=domain,
        dt=cfg.sim.dt,
        substeps="ADAPTIVE",
        corrector_steps=cfg.sim.corrector_steps,
        advection_tol=solver_tolerance(cfg, "advection"),
        pressure_tol=solver_tolerance(cfg, "pressure"),
        pressure_tol_intermediate=solver_tolerance(cfg, "pressure_intermediate"),
        pressure_warm_start=bool(cfg.sim.get("pressure_warm_start", False)),
        non_orthogonal=False,
        prep_fn=prep_fn,
    )
    sim.make_divergence_free(max_iter=10000)
    return sim

def make_mhd_simulation(
    cfg: DictConfig, domain: PISOtorch.Domain, prep_fn: dict[str, Any]
) -> MHDSimulation:
    if cfg.hartmann_number == 0.0:
        raise ValueError("Hartmann number must be > 0 for MHD simulation.")

    set_advection_scheme(cfg, domain)
    log_tolerances(cfg, ("advection", "pressure", "pressure_intermediate", "potential"))

    e_b = torch.tensor([1.0, 0.0, 0.0], dtype=domain.getDtype())

    reynolds_number = np.sqrt(cfg.grashof_number)
    sim = MHDSimulation(
        domain=domain,
        dt=cfg.sim.dt,
        stuart_number=torch.tensor(cfg.hartmann_number**2 / reynolds_number),
        e_b=e_b,
        substeps="ADAPTIVE",
        corrector_steps=cfg.sim.corrector_steps,
        advection_tol=solver_tolerance(cfg, "advection"),
        pressure_tol=solver_tolerance(cfg, "pressure"),
        pressure_tol_intermediate=solver_tolerance(cfg, "pressure_intermediate"),
        pressure_warm_start=bool(cfg.sim.get("pressure_warm_start", False)),
        potential_tol=solver_tolerance(cfg, "potential"),
        non_orthogonal=False,
        potential_normalize=cfg.sim.potential_normalize,
        potential_use_BiCG=cfg.sim.potential_use_BiCG,
        potential_solve_max_iter=cfg.sim.potential_solve_max_iter,
        potential_use_preconditioner=bool(
            cfg.sim.get("potential_use_preconditioner", False)
        ),
        potential_amg_options=(
            OmegaConf.to_container(cfg.sim.potential_amg_options, resolve=True)
            if cfg.sim.get("potential_amg_options", None)
            else None
        ),
        prep_fn=prep_fn,
        adaptive_CFL=cfg.sim.adaptive_cfl,
    )
    sim.solver_double_fallback = True
    sim.linear_solve_max_iterations = cfg.sim.pressure_solve_max_iter
    sim.make_divergence_free(max_iter=10000)
    return sim

def resample(x_centers: np.ndarray, y_centers: np.ndarray, data: np.ndarray, render_shape: tuple[int, int]) -> np.ndarray:
    if data.shape != (len(x_centers), len(y_centers)):
        raise ValueError(f"Data shape {data.shape} does not match x_centers {len(x_centers)} and y_centers {len(y_centers)}")

    fn = RegularGridInterpolator((x_centers, y_centers), data, method="linear")

    # Create a uniform meshgrid for the target resolution
    ux = np.linspace(x_centers.min(), x_centers.max(), render_shape[0])
    uy = np.linspace(y_centers.min(), y_centers.max(), render_shape[1])

    # Generate the 3D grid points for evaluation
    pts = np.meshgrid(ux, uy, indexing="ij")
    pts_flat = np.array([p.flatten() for p in pts]).T

    # Resample
    uniform_data = fn(pts_flat).reshape(render_shape)
    return uniform_data

def compute_nusselt(cfg: DictConfig, domain: PISOtorch.Domain) -> torch.Tensor:
    """Compute the volume-averaged Nusselt number.

    Nu = 1 + sqrt(Gr) * Pr * <u_z * T>_vol

    u_z is the velocity component between the hot (-z) and cold (+z) walls.
    """
    block = domain.getBlocks()[0]
    T = block.passiveScalar[0, 0]   # [nz, ny, nx]
    u_z = block.velocity[0, 2, ...]   # [nz, ny, nx], z-component (between hot/cold walls)

    cell_size = get_cell_size(block).squeeze()  # [nz, ny, nx]

    rayleigh_number = cfg.grashof_number * cfg.prandtl_number
    dT_dz = (cfg.domain.T_cold - cfg.domain.T_hot) / cfg.domain.L

    inner_term = np.sqrt(rayleigh_number * cfg.prandtl_number) * u_z * T - dT_dz
    vol_mean_Nu = (inner_term * cell_size).sum() / cell_size.sum()
    return vol_mean_Nu


def plot_nusselt_curve(
    cfg: DictConfig,
    logs_warmup_df: pd.DataFrame,
    logs_df: pd.DataFrame,
    time_step: float,
) -> None:
    """Plot the Nusselt number over the full run (warmup + logging phase)."""
    full_df = pd.concat([logs_warmup_df, logs_df], ignore_index=True)
    if full_df.empty:
        return

    time = full_df["step"].to_numpy() * time_step
    nu = full_df["nusselt"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(time, nu, color="tab:blue", linewidth=1.0)

    if not logs_df.empty:
        warmup_end = int(cfg.warmup_steps) * time_step
        ax.axvline(warmup_end, color="gray", linestyle="--", linewidth=0.8)
        nu_mean = logs_df["nusselt"].mean()
        ax.axhline(
            nu_mean,
            color="tab:red",
            linestyle=":",
            linewidth=1.0,
            label=f"mean (logging phase) = {nu_mean:.4f}",
        )
        ax.legend(frameon=False, fontsize=8)

    ax.set_xlabel("Time")
    ax.set_ylabel("Nu")
    sns.despine(ax=ax)
    fig.tight_layout()
    plt.savefig("nusselt.pdf", format="pdf")
    plt.close(fig)


def compute_kinetic_energy(domain: PISOtorch.Domain) -> torch.Tensor:
    """Compute the volume-averaged kinetic energy: Ek = <0.5 * |u|^2>_vol."""
    block = domain.getBlocks()[0]
    u = block.velocity[0]  # [3, nz, ny, nx]

    cell_size = get_cell_size(block).squeeze()  # [nz, ny, nx]

    ke = (u ** 2).sum(dim=0)  # [nz, ny, nx]
    return (ke * cell_size).sum() / cell_size.sum()


def plot_domain(cfg: DictConfig, domain: PISOtorch.Domain, step: int):
    block = domain.getBlocks()[0]
    centers = get_cell_centers(block.vertexCoordinates) # shape: [3, nz, ny, nx], coords (dim 0) are in order (x, y, z)

    z_centers = centers[2, :, 0, 0].cpu().numpy() # shape [nx]
    y_centers = centers[1, 0, :, 0].cpu().numpy() # shape [nz]
    res_z = int(cfg.domain.L * 128)
    res_y = int(cfg.domain.H * 128)

    T = block.passiveScalar[0, ...] # shape [1, nz, ny, nx]
    velocity = block.velocity[0, ...] # shape [3, nz, ny, nx], components (dim 0) are in order (x, y, z)

    # Temperature
    T_slice_zy = T[0, :, :, T.shape[-1] // 2].cpu().numpy() # shape [1, nz, ny]
    T_uniform_zy = resample(
        x_centers=z_centers,
        y_centers=y_centers,
        data=T_slice_zy,
        render_shape=(res_z*2, res_y*2)
    )
    vel_z_slice_zy = velocity[2, :, :, velocity.shape[-1] // 2].cpu().numpy() # shape [nz, ny]
    vel_y_slice_zy = velocity[1, :, :, velocity.shape[-1] // 2].cpu().numpy() # shape [nz, ny]
    vel_z_uniform = resample(
        x_centers=z_centers,
        y_centers=y_centers,
        data=vel_z_slice_zy,
        render_shape=(res_z*2, res_y*2)
    )
    vel_y_uniform = resample(
        x_centers=z_centers,
        y_centers=y_centers,
        data=vel_y_slice_zy,
        render_shape=(res_z*2, res_y*2)
    )

    half_L = cfg.domain.L / 2
    half_H = cfg.domain.H / 2
    aspect = cfg.domain.H / cfg.domain.L
    fig, ax = plt.subplots(1, 1, figsize=(max(1.0, 7.0 / aspect), min(7.0, 7.0 * aspect)))
    T_uniform_zy = T_uniform_zy.T  # now shape [res_y, res_z], rows=y, cols=z
    T_plot_zy = (T_uniform_zy - T_uniform_zy.min()) / (T_uniform_zy.max() - T_uniform_zy.min() + 1e-10)
    img = sns.color_palette("rainbow", as_cmap=True)(T_plot_zy, bytes=True)[:, :, :3]
    ax.imshow(img, extent=(-half_L, half_L, -half_H, half_H), origin="lower")

    quiver_stride = 16
    qu = vel_z_uniform[::quiver_stride, ::quiver_stride]
    qv = vel_y_uniform[::quiver_stride, ::quiver_stride]
    half_L = cfg.domain.L / 2
    half_H = cfg.domain.H / 2
    qz = np.linspace(-half_L, half_L, qu.shape[0])
    qy = np.linspace(-half_H, half_H, qu.shape[1])
    QX, QY = np.meshgrid(qz, qy, indexing="ij")
    ax.quiver(QX, QY, qu, qv, color="black", alpha=1.0, scale=20.0)

    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    plt.savefig(f"temperature_{step:07d}.png", format="png", dpi=500)
    plt.close(fig)

    # Velocity magnitude
    vel_mag = torch.norm(velocity, dim=0).cpu().numpy() # shape [nz, ny, nx]
    vel_slice = vel_mag[:, :, vel_mag.shape[-1] // 2] # shape [ny, nx]
    vel_uniform = resample(
        x_centers=z_centers,
        y_centers=y_centers,
        data=vel_slice,
        render_shape=(res_z*16, res_y*16)
    )
    fig, ax = plt.subplots(1, 1, figsize=(FIGWIDTH * 0.5, FIGWIDTH * 0.5 * aspect))
    vel_uniform = vel_uniform.T
    vel_uniform = np.flip(vel_uniform, axis=0)
    vel_plot = vel_uniform / (vel_uniform.max() + 1e-10)
    img = sns.color_palette("rainbow", as_cmap=True)(vel_plot, bytes=True)[:, :, :3]
    ax.imshow(img, extent=(0.0, cfg.domain.L, 0.0, cfg.domain.H), origin="lower")
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    plt.savefig(f"velocity_{step:07d}.png", format="png")
    plt.close(fig)

    # DEBUG: the same slice on the raw cell grid, one pixel per cell, so that
    # grid-scale oscillations are not hidden by the resampling above.
    fig, ax = plt.subplots(1, 1, figsize=(FIGWIDTH * 0.5, FIGWIDTH * 0.5 * aspect))
    vel_raw = np.flip(vel_slice.T, axis=0)
    vel_raw_plot = vel_raw / (vel_raw.max() + 1e-10)
    img = sns.color_palette("rainbow", as_cmap=True)(vel_raw_plot, bytes=True)[:, :, :3]
    ax.imshow(img, extent=(0.0, cfg.domain.L, 0.0, cfg.domain.H), origin="lower", interpolation="nearest")
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    plt.savefig(f"velocity_raw_{step:07d}.png", format="png")
    plt.close(fig)

    # Potential
    if cfg.hartmann_number == 0.0:
        return

    epot = block.epot[0, ...] # shape [1, nz, ny, nx]
    epot_slice = epot[0, :, :, epot.shape[-1] // 2].cpu().numpy() # shape [nz, ny]
    epot_uniform = resample(
        x_centers=z_centers,
        y_centers=y_centers,
        data=epot_slice,
        render_shape=(res_z*4, res_y*4)
    )
    fig, ax = plt.subplots(1, 1, figsize=(FIGWIDTH * 0.5, FIGWIDTH * 0.5 * aspect))
    epot_uniform = epot_uniform.T
    epot_uniform = np.flip(epot_uniform, axis=0)
    epot_plot = (epot_uniform - epot_uniform.min()) / (epot_uniform.max() - epot_uniform.min() + 1e-10)
    img = sns.color_palette("RdBu_r", as_cmap=True)(epot_plot, bytes=True)[:, :, :3]
    ax.imshow(img, extent=(0.0, cfg.domain.L, 0.0, cfg.domain.H), origin="lower")
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    plt.savefig(f"potential_{step:07d}.png", format="png")
    plt.close(fig)


def plot_temperature_isosurface(cfg: DictConfig, domain: PISOtorch.Domain, step: int):
    from matplotlib.colors import BoundaryNorm
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    thresholds = [-0.42, -0.21, 0.0, 0.21, 0.42]
    T_min, T_max = cfg.domain.T_cold, cfg.domain.T_hot
    boundaries = [T_min] + thresholds + [T_max]
    cmap = plt.get_cmap("rainbow")

    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    # Center of each interval
    isovalues = [(boundaries[i] + boundaries[i+1]) / 2 for i in range(len(boundaries) - 1)]

    block = domain.getBlocks()[0]
    centers = get_cell_centers(block.vertexCoordinates)  # [3, nz, ny, nx]

    x_centers = centers[0, 0, 0, :].cpu().numpy()
    y_centers = centers[1, 0, :, 0].cpu().numpy()
    z_centers = centers[2, :, 0, 0].cpu().numpy()

    T = block.passiveScalar[0, 0].cpu().numpy()  # [nz, ny, nx]

    fig = plt.figure(figsize=(3.0, min(9.0, 3.0 * max(1.0, cfg.domain.H / cfg.domain.L))))
    ax = fig.add_subplot(111, projection="3d")

    for level in isovalues:
        if level <= T.min() or level >= T.max():
            continue
        verts, faces, _, _ = marching_cubes(T, level=level)
        # Scale verts from index space to physical space
        verts_phys = np.stack([
            np.interp(verts[:, 2], np.arange(len(x_centers)), x_centers),  # matplotlib x
            np.interp(verts[:, 0], np.arange(len(z_centers)), z_centers),  # matplotlib y (physical z)
            np.interp(verts[:, 1], np.arange(len(y_centers)), y_centers),  # matplotlib z = vertical (physical y)
        ], axis=1)
        color = cmap((level - T_min) / (T_max - T_min))
        mesh = Poly3DCollection(verts_phys[faces], alpha=1.0, facecolor=color, edgecolor="none")
        ax.add_collection3d(mesh)

    half_width = cfg.domain.L / 2
    ax.set_xlim(-half_width, half_width)
    ax.set_ylim(-half_width, half_width)
    ax.set_zlim(-cfg.domain.H / 2, cfg.domain.H / 2)
    ax.set_box_aspect([
        cfg.domain.L,
        cfg.domain.L,
        cfg.domain.H,
    ])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # 1. Remove background grid and panes
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")

    # 2. Hide ticks but keep labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.xaxis.labelpad = -15
    ax.yaxis.labelpad = -15

    # 3. Add the bounding box edges
    x_min, x_max = x_centers.min(), x_centers.max()
    z_min, z_max = z_centers.min(), z_centers.max()
    y_min, y_max = y_centers.min(), y_centers.max()

    # Define the 12 edges of a box
    # Format: [(x0, x1), (z0, z1), (y0, y1)]
    edges = [
        # Bottom face
        [[x_min, x_max], [z_min, z_min], [y_min, y_min]],
        [[x_min, x_max], [z_max, z_max], [y_min, y_min]],
        [[x_min, x_min], [z_min, z_max], [y_min, y_min]],
        [[x_max, x_max], [z_min, z_max], [y_min, y_min]],
        # Top face
        [[x_min, x_max], [z_min, z_min], [y_max, y_max]],
        [[x_min, x_max], [z_max, z_max], [y_max, y_max]],
        [[x_min, x_min], [z_min, z_max], [y_max, y_max]],
        [[x_max, x_max], [z_min, z_max], [y_max, y_max]],
        # Vertical connectors
        [[x_min, x_min], [z_min, z_min], [y_min, y_max]],
        [[x_max, x_max], [z_min, z_min], [y_min, y_max]],
        [[x_min, x_min], [z_max, z_max], [y_min, y_max]],
        [[x_max, x_max], [z_max, z_max], [y_min, y_max]],
    ]

    for edge in edges:
        ax.plot(edge[0], edge[1], edge[2], color="black", linewidth=1, zorder=10)

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.view_init(elev=22.5, azim=30)

    for collection in ax.collections:
        collection.set_clip_on(False)

    for line in ax.lines:
        line.set_clip_on(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, label="T", shrink=0.3, ticks=boundaries)

    fig.subplots_adjust(left=-0.2, right=0.9, top=1, bottom=0.2, wspace=0)

    plt.savefig(f"temperature_isosurface_{step:07d}.png", format="png", bbox_inches="tight")
    plt.close(fig)


@hydra.main(version_base="1.3", config_path="../configs", config_name="validation_temp")
def main(cfg: DictConfig):
    try:
        run_validation(cfg)
    except Exception as e:
        logger.exception("Job crashed with an exception")
        logger.error(e)
        raise


def run_validation(cfg: DictConfig):
    dtype = torch.float64 if cfg.precision == "double" else torch.float32
    device = torch.device("cuda")

    # -----------------------------------------------------------------------
    # Domain Setup
    # -----------------------------------------------------------------------
    domain, prep_fn = make_domain(cfg, dtype=dtype, device=device)
    start_step = 0
    if cfg.get("load_domain", False):
        loaded_domain = load_domain(Path("domain"), device=device, dtype=dtype)
        for block, loaded_block in zip(domain.getBlocks(), loaded_domain.getBlocks()):
            block.setVelocity(loaded_block.velocity)
            block.setPressure(loaded_block.pressure)
            block.setPassiveScalar(loaded_block.passiveScalar)

            if cfg.hartmann_number > 0.0:
                block.setEpot(loaded_block.epot)

        domain.PrepareSolve()
        metadata = load_metadata()
        start_step = metadata["step"] + 1
        logger.info(f"Resuming from saved domain at step {metadata['step']}.")

    # Log number of cells resolved within Hartmann and Shercliff boundary layers
    # B field is in x → Hartmann walls at ±x, Shercliff walls at ±z
    # Theoretical thicknesses: δ_Ha ~ L·Ha⁻¹, δ_Sh ~ L·Ha⁻¹/²
    if cfg.hartmann_number > 0.0:
        _block = domain.getBlocks()[0]
        _centers = get_cell_centers(_block.vertexCoordinates)
        _x_centers = _centers[0, 0, 0, :].cpu().numpy()
        _z_centers = _centers[2, :, 0, 0].cpu().numpy()
        _ha_thickness = cfg.domain.L / cfg.hartmann_number
        _sh_thickness = cfg.domain.L / cfg.hartmann_number ** 0.5
        _n_hartmann = int(np.sum(
            (_x_centers > cfg.domain.L / 2 - _ha_thickness) | (_x_centers < -cfg.domain.L / 2 + _ha_thickness)
        ))
        _n_shercliff = int(np.sum(
            (_z_centers > cfg.domain.L / 2 - _sh_thickness) | (_z_centers < -cfg.domain.L / 2 + _sh_thickness)
        ))
        logger.info(
            f"Boundary layer cells — "
            f"Hartmann ({_ha_thickness:.4f}): {_n_hartmann} cell(s) | "
            f"Shercliff ({_sh_thickness:.4f}): {_n_shercliff} cell(s)"
        )
    else:
        # Ha = 0 has no magnetic boundary layers at all, so the counts above are
        # undefined rather than zero. Logged anyway so the line never silently
        # goes missing between an MHD and a hydrodynamic run.
        logger.info(
            "Boundary layer cells — Ha = 0, no Hartmann/Shercliff layers "
            "(hydrodynamic case)"
        )

    # -----------------------------------------------------------------------
    # Simulation Setup
    # -----------------------------------------------------------------------
    if cfg.hartmann_number > 0.0:
        sim = make_mhd_simulation(cfg, domain, prep_fn=prep_fn)
    else:
        sim = make_simulation(cfg, domain, prep_fn=prep_fn)

    log_interval = int(cfg.log_interval / sim.time_step)

    # -----------------------------------------------------------------------
    # Warmup
    # -----------------------------------------------------------------------
    logs_warmup = []
    for step in range(start_step, int(cfg.warmup_steps)):
        ok = sim.single_step()
        time_passed = step * sim.time_step

        if not ok:
            logger.error(f"Solver failed to converge at step {step}.")

        nu = compute_nusselt(cfg, domain)
        ek = compute_kinetic_energy(domain)
        logs_warmup.append({"step": step, "nusselt": nu.item(), "kinetic_energy": ek.item()})

        if step % log_interval == 0:
            logger.info(f"T = {time_passed:.2f}: Nu = {nu.item():.4f}, Ek = {ek.item():.6f}")
            plot_domain(cfg, domain, step)
            plot_temperature_isosurface(cfg, domain, step)
            save_domain(domain, Path("domain"))
            save_metadata({"step": step})

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    logs = []
    logger.info("Warmup complete. Starting main logging phase.")
    for step in range(int(cfg.warmup_steps), int(cfg.warmup_steps) + int(cfg.log_steps)):
        ok = sim.single_step()
        time_passed = step * sim.time_step

        if not ok:
            logger.error(f"Solver failed to converge at step {step}.")

        nu = compute_nusselt(cfg, domain)
        ek = compute_kinetic_energy(domain)
        logs.append({"step": step, "nusselt": nu.item(), "kinetic_energy": ek.item()})

        if step % log_interval == 0:
            logger.info(f"T = {time_passed:.2f}: Nu = {nu.item():.4f}, Ek = {ek.item():.6f}")
            plot_domain(cfg, domain, step)
    logger.info("Logging phase complete. Generating final visualizations.")
    plot_temperature_isosurface(cfg, domain, step)
    plot_domain(cfg, domain, step)

    # save final domain
    save_domain(domain, Path("final_domain"))

    logs_warmup_df = pd.DataFrame(logs_warmup)
    logs_warmup_df.to_csv("logs_warmup.csv", index=False)

    logs_df = pd.DataFrame(logs)
    logs_df.to_csv("logs.csv", index=False)

    nu_mean = logs_df["nusselt"].mean()
    ek_mean = logs_df["kinetic_energy"].mean()
    logger.info(f"Average Nusselt number over last {len(logs_df)} steps: {nu_mean:.4f}")
    logger.info(f"Average kinetic energy over last {len(logs_df)} steps: {ek_mean:.6f}")

    if cfg.type == "temp1":
        plot_nusselt_curve(cfg, logs_warmup_df, logs_df, float(sim.time_step))

if __name__ == "__main__":
    main()
