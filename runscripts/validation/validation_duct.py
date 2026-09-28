"""Run the turbulent MHD duct validation case.

Usage:
    python runscripts/validation/validation_duct.py case=duct_2000
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
from fluidgym.simulation.solver_tolerance import SolverTolerance

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

def get_shercliff_profile(
    pressure_gradient: float,
    y: np.ndarray,
    z: np.ndarray,
    cfg: DictConfig,
    d_B: float = 0.0,
) -> np.ndarray:
    viscosity = (cfg.domain.U * cfg.domain.L) / cfg.reynolds_number
    a = cfg.domain.L
    aspect_ratio = 1.0
    n_steps = 1000

    YY, ZZ = np.meshgrid(y, z, indexing="ij")  # [NY, NZ]

    u = np.zeros_like(YY)

    for k in range(n_steps):
        alpha_k = (k + 0.5) * np.pi / aspect_ratio
        N_k = np.sqrt(cfg.hartmann_number**2 + 4 * alpha_k**2)
        r1_k = 0.5 * (cfg.hartmann_number + N_k)
        r2_k = 0.5 * (-cfg.hartmann_number + N_k)

        # ----------------------------------------------------------
        # V2 (Eq. 5.10)
        # ----------------------------------------------------------
        V_2_k = (
            (d_B * r2_k + (1 - np.exp(-2 * r2_k)) / (1 + np.exp(-2 * r2_k)))
            * (np.exp(-r1_k * (1 - YY)) + np.exp(-r1_k * (1 + YY)))
            / 2
        ) / (
            (1 + np.exp(-2 * r1_k)) / 2 * d_B * N_k
            + (1 + np.exp(-2 * (r1_k + r2_k))) / (1 + np.exp(-2 * r2_k))
        )
        # ----------------------------------------------------------
        # V3 (Eq. 5.11)
        # ----------------------------------------------------------
        V_3_k = (
            (d_B * r1_k + (1 - np.exp(-2 * r1_k)) / (1 + np.exp(-2 * r1_k)))
            * (np.exp(-r2_k * (1 - YY)) + np.exp(-r2_k * (1 + YY)))
            / 2
        ) / (
            (1 + np.exp(-2 * r2_k)) / 2 * d_B * N_k
            + (1 + np.exp(-2 * (r1_k + r2_k))) / (1 + np.exp(-2 * r1_k))
        )

        # ----------------------------------------------------------
        # V (Eq. 5.9)
        # ----------------------------------------------------------
        V_k = (
            (2 * (-1) ** k * np.cos(alpha_k * ZZ))
            / (aspect_ratio * alpha_k**3)
            * (1 - V_2_k - V_3_k)
        )

        u += V_k

    # ----------------------------------------------------------
    # v_x (Eq. 5.8) (here x instead of z)
    # ----------------------------------------------------------
    u = viscosity**-1 * -pressure_gradient * a**2 * u

    return u

def make_domain(
    cfg: DictConfig, dtype: torch.dtype, device: torch.device
) -> PISOtorch.Domain:
    if cfg.domain.duct not in ["full", "quarter"]:
        raise ValueError(f"Invalid duct type {cfg.domain.duct}. Must be 'full' or 'quarter'.")

    full_duct = cfg.domain.duct == "full"
    if full_duct:
        wall_refinement = "BOTH"
        start_y = -cfg.domain.L
        start_z = -cfg.domain.L
        ny = cfg.domain.ny * 2
        nz = cfg.domain.nz * 2
    else:
        wall_refinement = "END"
        start_y = 0.0
        start_z = 0.0
        ny = cfg.domain.ny
        nz = cfg.domain.nz

    y_weights = make_weights(
        grading_type=cfg.domain.grading_y_type,
        res=ny,
        grading=cfg.domain.grading_y,
        refinement=wall_refinement,
    )

    grid = shapes.generate_grid_vertices_2D(
        [ny + 1, cfg.domain.nx + 1],
        [
            (0.0, start_y),
            (cfg.domain.length * cfg.domain.L, start_y),
            (0.0, cfg.domain.L),
            (cfg.domain.length * cfg.domain.L, cfg.domain.L),
        ],
        None,
        x_weights=y_weights,
        y_weights=None,
        dtype=dtype,
    )

    # Plot x-y slice
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_grid(grid, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.savefig("grid_x-y.pdf", format="pdf")
    plt.close(fig)

    # Extrude in z direction
    z_weights = make_weights(
        grading_type=cfg.domain.grading_z_type,
        res=nz,
        grading=cfg.domain.grading_z,
        refinement=wall_refinement,
    )

    grid = shapes.extrude_grid_z(
        grid,
        res_z=nz,
        start_z=start_z,
        end_z=cfg.domain.L,
        weights_z=z_weights,
    )

    # Plot y-z slice
    fig, ax = plt.subplots(figsize=(10, 6))
    grid_y_z = torch.stack([grid[:, 1, :, :, 0], grid[:, 2, :, :, 0]], dim=1)
    plot_grid(grid_y_z, ax=ax)
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_aspect("equal")
    plt.savefig("grid_y-z.pdf", format="pdf")
    plt.close(fig)

    grid = grid.to(device).contiguous()
    kinematic_viscosity = torch.tensor(
        [(cfg.domain.U * cfg.domain.L) / cfg.reynolds_number],
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

    block.CloseBoundary("+y")
    block.CloseBoundary("+z")

    if full_duct:
        block.CloseBoundary("-y")
        block.CloseBoundary("-z")
    else:
        # Symmetry planes: u_n = 0, du_t/dn = 0 (free-slip). CloseBoundary("+y"/"+z")
        # above turned the default periodic pair into Dirichlet-0 walls, so these must
        # be re-opened afterwards
        block.OpenBoundary("-y")
        block.OpenBoundary("-z")

        # With B in z, phi is odd in y and even in z: the core current crosses y=0 at
        # its maximum, so that plane needs phi=0, while j_z=0 (the default insulating
        # treatment) is already the correct condition at z=0
        bc.set_bc(block, Face.Y_MINUS, bc.Potential.Dirichlet())

    # Conducting walls
    if full_duct:
        bc.set_bc(block, Face.Z_MINUS, bc.Potential.ThinWall(cw=cfg.hartmann_Cw))
    bc.set_bc(block, Face.Z_PLUS, bc.Potential.ThinWall(cw=cfg.hartmann_Cw))

    if cfg.get("init_with_analytical_profile", False):
        cell_centers = get_cell_centers(block.vertexCoordinates)  # [3, NY, NZ, NX]
        y_centers = cell_centers[1, 0, :, 0]  # [NY]
        z_centers = cell_centers[2, :, 0, 0]  # [NY]

        # For the quarter duct, we need to construct the full duct first and then
        # take the relevant half of the profile, since the Shercliff profile is defined
        # for the full duct
        if not full_duct:
            y_centers = torch.cat([-y_centers.flip(0), y_centers], dim=0)  # [2*NY]
            z_centers = torch.cat([-z_centers.flip(0), z_centers], dim=0)  # [2*NZ]

        profile = get_shercliff_profile(
            pressure_gradient=cfg.pressure_gradient,
            y=z_centers.cpu().numpy(),
            z=y_centers.cpu().numpy(),
            cfg=cfg,
            d_B=cfg.hartmann_Cw,
        )

        if cfg.dynamic_forcing:
            # We have a target velocity of U
            cell_sizes = get_cell_size(block)[0, 0, ..., 0].cpu().numpy() # [NY, NZ]
            if not full_duct:
                # Mirror along both axes to reconstruct the full duct cell sizes
                cell_sizes = np.concatenate([np.flip(cell_sizes, axis=0), cell_sizes], axis=0)  # [2*NY, NZ]
                cell_sizes = np.concatenate([np.flip(cell_sizes, axis=1), cell_sizes], axis=1)  # [2*NY, 2*NZ]

            profile_mean = (np.sum(profile * cell_sizes) / np.sum(cell_sizes))
            profile = profile * (cfg.domain.U / profile_mean)

        # Now, we take the upper corner of the duct
        if not full_duct:
            profile = profile[nz:, ny:]

        velocity = torch.zeros((1, 3, nz, ny, cfg.domain.nx), dtype=dtype, device=device)
        velocity[0, 0, :, :, :] = torch.from_numpy(profile).unsqueeze(-1).to(device)
        block.setVelocity(velocity)
    else:
        velocity = torch.zeros((1, 3, nz, ny, cfg.domain.nx), dtype=dtype, device=device)
        block.setVelocity(velocity)

    if cfg.get("init_with_noise", False):
        from fluidgym.simulation.extensions import (
            SimplexNoiseVariations,  # type: ignore[import-untyped,import-not-found]
        )

        curl_noise = SimplexNoiseVariations.GenerateSimplexNoiseVariation(
            (cfg.domain.nx, ny, nz),
            device,
            [2 / s for s in block.velocity.shape[2:]],
            [0] * 3,
            SimplexNoiseVariations.NoiseVariation.CURL,
        )
        if cfg.get("init_with_analytical_profile", False):
            curl_noise *= 0.5
        block.setVelocity(block.velocity + curl_noise)

    domain.PrepareSolve()

    return domain


def build_tolerances(cfg: DictConfig) -> dict[str, Any]:
    """Turn the ``tol:`` config block into solver kwargs.

    ``absolute`` reproduces the historical behaviour exactly (a plain float goes
    straight to the solver). ``relative`` wraps each value as a
    :class:`~fluidgym.simulation.solver_tolerance.SolverTolerance`; the stop test
    then becomes ``||r|| < rtol*||b||`` and is independent of the duct length.
    """
    tol = cfg.get("tol", None)
    if tol is None:
        return {}

    default_mode = str(tol.get("mode", "absolute")).lower()
    atol = tol.get("atol", None)

    def mode_for(name: str) -> str:
        # Per-solve override, so one solve can be switched to a relative
        # tolerance without changing the others. The solves have RHS norms that
        # differ by orders of magnitude, so a single global mode makes any
        # one-solve-at-a-time experiment impossible to interpret
        mode = tol.get(f"{name}_mode", None) or default_mode
        mode = str(mode).lower()
        if mode not in ("absolute", "relative"):
            raise ValueError(
                f"tol.{name}_mode must be 'absolute' or 'relative', got {mode!r}."
            )
        return mode

    def spec(name: str):
        value = tol.get(name, None)
        if value is None:
            return None
        if mode_for(name) == "absolute":
            return float(value)
        return SolverTolerance(
            rtol=float(value), atol=None if atol is None else float(atol)
        )

    return {
        "advection_tol": spec("advection"),
        "pressure_tol": spec("pressure"),
        # Only the final corrector's pressure survives into the solution, so the
        # intermediate ones can be solved loosely. Absent -> pressure_tol
        # everywhere, i.e. unchanged behaviour
        "pressure_tol_intermediate": spec("pressure_intermediate"),
        "potential_tol": spec("potential"),
    }


def make_mhd_simulation(
    cfg: DictConfig, domain: PISOtorch.Domain, prep_fn: dict[str, Any]
) -> MHDSimulation:
    if cfg.hartmann_number == 0.0:
        raise ValueError("Hartmann number must be > 0 for MHD simulation.")

    e_b = torch.tensor([0.0, 0.0, 1.0], dtype=domain.getDtype())

    sim = MHDSimulation(
        domain=domain,
        dt=cfg.sim.dt,
        stuart_number=torch.tensor(cfg.hartmann_number**2 / cfg.reynolds_number),
        e_b=e_b,
        substeps="ADAPTIVE",
        corrector_steps=cfg.sim.corrector_steps,
        non_orthogonal=False,
        potential_normalize=cfg.sim.potential_normalize,
        potential_use_BiCG=cfg.sim.potential_use_BiCG,
        potential_solve_max_iter=cfg.sim.potential_solve_max_iter,
        potential_return_best_result=bool(
            cfg.get("potential_return_best_result", True)
        ),
        potential_reuse_result=bool(cfg.get("potential_reuse_result", True)),
        potential_use_preconditioner=bool(
            cfg.get("potential_use_preconditioner", False)
        ),
        potential_amg_options=(
            OmegaConf.to_container(cfg.potential_amg_options, resolve=True)
            if cfg.get("potential_amg_options", None)
            else None
        ),
        pressure_warm_start=bool(cfg.get("pressure_warm_start", False)),
        prep_fn=prep_fn,
        adaptive_CFL=cfg.sim.adaptive_cfl,
        **build_tolerances(cfg),
    )
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

def get_mean_x_velocity(
    block: PISOtorch.Block
) -> torch.Tensor:
    u_x: torch.Tensor = block.velocity[0, 0, ...]
    cell_sizes = get_cell_size(block)[0, ...]

    u_x_mean = (u_x * cell_sizes).sum() / cell_sizes.sum()

    return u_x_mean

def compute_q_criterion(domain: PISOtorch.Domain) -> torch.Tensor:
    domain.UpdateDomainData()
    gradients: torch.Tensor = PISOtorch.ComputeSpatialVelocityGradients(domain)
    d_dx, d_dy, d_dz = gradients[0]

    du_dx = d_dx[0, 0, ...]
    du_dy = d_dy[0, 0, ...]
    du_dz = d_dz[0, 0, ...]

    dv_dy = d_dy[0, 1, ...]
    dv_dx = d_dx[0, 1, ...]
    dv_dz = d_dz[0, 1, ...]

    dw_dx = d_dx[0, 2, ...]
    dw_dy = d_dy[0, 2, ...]
    dw_dz = d_dz[0, 2, ...]

    grad_u = torch.stack(
        [
            torch.stack([du_dx, du_dy, du_dz], dim=0),
            torch.stack([dv_dx, dv_dy, dv_dz], dim=0),
            torch.stack([dw_dx, dw_dy, dw_dz], dim=0),
        ],
        dim=0,
    )

    # Compute the symmetric and antisymmetric parts
    S = 0.5 * (grad_u + grad_u.transpose(1, 0))
    Omega = 0.5 * (grad_u - grad_u.transpose(1, 0))

    # Compute the Frobenius norms
    S_norm_sq = torch.sum(S**2, dim=(0, 1))
    Omega_norm_sq = torch.sum(Omega**2, dim=(0, 1))

    # Compute Q
    Q = 0.5 * (Omega_norm_sq - S_norm_sq)

    return Q

def compute_velocity_fluctuations(
    block: PISOtorch.Block, cfg: DictConfig, pressure_gradient: float | None = None
) -> dict[str, float]:
    """Compute volume-averaged kinetic energy of velocity fluctuations."""
    cell_centers = get_cell_centers(block.vertexCoordinates)  # [3, NZ, NY, NX]
    y_centers = cell_centers[1, 0, :, 0]  # [NY]
    z_centers = cell_centers[2, :, 0, 0]  # [NZ]

    # For the quarter duct the Shercliff formula is defined on the full duct,
    # so we mirror the coordinates, compute the full profile, then crop
    full_duct = cfg.domain.duct == "full"
    if not full_duct:
        ny = y_centers.shape[0]
        nz = z_centers.shape[0]
        y_full = torch.cat([-y_centers.flip(0), y_centers], dim=0)
        z_full = torch.cat([-z_centers.flip(0), z_centers], dim=0)
        profile_full = get_shercliff_profile(
            pressure_gradient=pressure_gradient or cfg.pressure_gradient,
            y=z_full.cpu().numpy(),
            z=y_full.cpu().numpy(),
            cfg=cfg,
            d_B=cfg.get("hartmann_Cw", 0.0),
        )  # [2*NY, 2*NZ]
        u_ref = torch.from_numpy(profile_full[nz:, ny:]).to(
            device=block.velocity.device, dtype=block.velocity.dtype
        )  # [NZ, NY]
    else:
        u_ref = torch.from_numpy(
            get_shercliff_profile(
                pressure_gradient=pressure_gradient or cfg.pressure_gradient,
                y=z_centers.cpu().numpy(),
                z=y_centers.cpu().numpy(),
                cfg=cfg,
                d_B=cfg.get("hartmann_Cw", 0.0),
            )
        ).to(device=block.velocity.device, dtype=block.velocity.dtype)  # [NZ, NY]

    velocity = block.velocity[0]  # [3, NZ, NY, NX]
    u_x = velocity[0]  # [NZ, NY, NX]
    u_y = velocity[1]  # [NZ, NY, NX]
    u_z = velocity[2]  # [NZ, NY, NX]

    # Fluctuations: numerical - analytical reference
    # u_ref is [NZ, NY]; broadcast over NX
    du_x = u_x - u_ref.unsqueeze(-1)  # [NZ, NY, NX]
    du_y = u_y  # analytical u_y = 0
    du_z = u_z  # analytical u_z = 0

    # Cell volumes (weights for volume averaging)
    cell_sizes = get_cell_size(block)[0]  # [3, NZ, NY, NX]
    total_vol = cell_sizes.sum()

    def vol_avg(field: torch.Tensor) -> float:
        return ((field * cell_sizes).sum() / total_vol).item()

    e_streamwise = vol_avg(du_x**2)
    e_perp = vol_avg(du_y**2)
    e_para = vol_avg(du_z**2)
    e_transverse = e_perp + e_para

    return {
        "streamwise": e_streamwise,
        "transverse": e_transverse,
        "perp": e_perp,
        "para": e_para,
    }

def plot_domain(cfg: DictConfig, domain: PISOtorch.Domain, step: int, y_scale: float = 2.0):
    y_extent = 1.0 if cfg.domain.duct == "full" else 0.5
    y_res = 512 if cfg.domain.duct == "full" else 256

    # Scale y for better visibility
    y_extent *= y_scale
    y_res = int(y_res * y_scale)

    block = domain.getBlocks()[0]
    centers = get_cell_centers(block.vertexCoordinates) # shape: [3, nz, ny, nx], coords (dim 0) are in order (x, y, z)

    x_centers = centers[0, 0, 0, :].cpu().numpy() # shape [nx]
    y_centers = centers[1, 0, :, 0].cpu().numpy() # shape [nz]
    res_x = int(cfg.domain.length * cfg.domain.L * 64)
    res_y = int(cfg.domain.L * y_res)

    velocity = block.velocity[0, ...] # shape [3, nz, ny, nx], components (dim 0) are in order (x, y, z)
    epot = block.epot[0, ...] # shape [1, nz, ny, nx]

    # Velocity resampling
    u_x = velocity[0, ...].cpu().numpy() # shape [nz, ny, nx]
    u_y = velocity[1, ...].cpu().numpy() # shape [nz, ny, nx]
    u_magn = np.linalg.norm(velocity.cpu().numpy(), axis=0) # shape [nz, ny, nx]
    vel_x_slice = u_x[u_x.shape[0] // 2, :, :].T # shape [nx, ny]
    vel_y_slice = u_y[u_y.shape[0] // 2, :, :].T # shape [nx, ny]
    vel_magn_slice = u_magn[u_magn.shape[0] // 2, :, :].T # shape [nx, ny]
    vel_x_uniform = resample(
        x_centers=x_centers,
        y_centers=y_centers,
        data=vel_x_slice,
        render_shape=(res_x, res_y)
    )
    vel_y_uniform = resample(
        x_centers=x_centers,
        y_centers=y_centers,
        data=vel_y_slice,
        render_shape=(res_x, res_y)
    )
    vel_magn_uniform = resample(
        x_centers=x_centers,
        y_centers=y_centers,
        data=vel_magn_slice,
        render_shape=(res_x, res_y)
    )

    # x-velocity
    fig, ax = plt.subplots(1, 1, figsize=(FIGWIDTH * 0.5 * cfg.domain.length, FIGWIDTH * 0.5))
    vel_x_uniform = vel_x_uniform.T
    im = ax.imshow(vel_x_uniform, extent=(0.0, cfg.domain.length, 0.0, y_extent), origin="lower",
                   cmap=sns.color_palette("rainbow", as_cmap=True))
    plt.colorbar(im, ax=ax, label=r"$u_x$")
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    plt.savefig(f"velocity_{step:07d}.png", format="png")
    plt.close(fig)

    # Velocity magnitude
    fig, ax = plt.subplots(1, 1, figsize=(FIGWIDTH * 0.5 * cfg.domain.length, FIGWIDTH * 0.5))
    vel_magn_uniform = vel_magn_uniform.T
    vel_y_uniform = vel_y_uniform.T
    # vel_x_uniform was already transformed in the x-velocity plot block
    im = ax.imshow(vel_magn_uniform, extent=(0.0, cfg.domain.length, 0.0, y_extent), origin="lower",
                   cmap=sns.color_palette("viridis", as_cmap=True))
    plt.colorbar(im, ax=ax, label=r"$|u|$")
    quiver_stride = res_y // 32
    qu = vel_x_uniform[::quiver_stride, ::quiver_stride]
    qv = vel_y_uniform[::quiver_stride, ::quiver_stride]
    qx = np.linspace(0.0, cfg.domain.length, qu.shape[1])
    qy = np.linspace(0.0, y_extent, qu.shape[0])
    QX, QY = np.meshgrid(qx, qy)
    ax.quiver(QX, QY, qu, qv, color="white", alpha=0.6, scale=50.0)
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    plt.savefig(f"velocity_magnitude_{step:07d}.png", format="png")
    plt.close(fig)

    # Potential
    epot_slice = epot[0, epot.shape[0] // 2, :, :].T.cpu().numpy() # shape [nx, ny]
    epot_uniform = resample(
        x_centers=x_centers,
        y_centers=y_centers,
        data=epot_slice,
        render_shape=(res_x, res_y)
    )
    fig, ax = plt.subplots(1, 1, figsize=(FIGWIDTH * 0.5 * cfg.domain.length, FIGWIDTH * 0.5))
    epot_uniform = epot_uniform.T
    epot_uniform = np.flip(epot_uniform, axis=0)
    epot_uniform = np.flip(epot_uniform, axis=1)
    im = ax.imshow(epot_uniform, extent=(0.0, cfg.domain.length, 0.0, y_extent), origin="lower",
                   cmap=sns.color_palette("RdBu_r", as_cmap=True))
    plt.colorbar(im, ax=ax, label="Electric potential")
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    plt.savefig(f"potential_{step:07d}.png", format="png")
    plt.close(fig)

def plot_profiles(cfg: DictConfig, domain: PISOtorch.Domain, step: int, pressure_gradient: float | None = None):
    grids: list[torch.Tensor] = domain.getVertexCoordinates()
    block = domain.getBlocks()[0]
    cell_sizes = get_cell_size(block).cpu().numpy()
    cell_centers = get_cell_centers(grids[0]).cpu().numpy()
    u_x_profile: np.ndarray = block.velocity[0, 0, ...].cpu().numpy()

    numerical_y = cell_centers[1, 0, :, 0]
    numerical_z = cell_centers[2, :, 0, 0]

    numerical_ux_profile = u_x_profile.mean(axis=-1)
    analytical_y = np.linspace(-cfg.domain.L, cfg.domain.L, 1000)
    analytical_z = np.linspace(-cfg.domain.L, cfg.domain.L, 100)

    analytical_ux_profile = get_shercliff_profile(
        pressure_gradient=pressure_gradient or cfg.pressure_gradient,
        y=analytical_z,
        z=analytical_y,
        cfg=cfg,
        d_B=cfg.get("hartmann_Cw", 0.0),
    )

    cell_sizes = cell_sizes[0, 0, ..., 0].T  # [NY, NZ]

    analytical_u_x_slice_y = analytical_ux_profile[:, analytical_ux_profile.shape[1] // 2]
    analytical_u_x_slice_z = analytical_ux_profile[analytical_ux_profile.shape[0] // 2, :]

    if cfg.domain.duct == "quarter":
        numerical_u_x_slice_y = numerical_ux_profile[:, 0]
        numerical_u_x_slice_z = numerical_ux_profile[0, :]
    else:
        numerical_u_x_slice_y = numerical_ux_profile[:, numerical_ux_profile.shape[1] // 2]
        numerical_u_x_slice_z = numerical_ux_profile[numerical_ux_profile.shape[0] // 2, :]

    palette = sns.color_palette()

    # ----------------------------------------------------------
    # Velocity Profiles
    # ----------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    sns.lineplot(
        x=analytical_z,
        y=analytical_u_x_slice_y,
        ax=ax[0],
        color=palette[0],
        label="Analytical",
    )
    sns.scatterplot(
        x=numerical_z,
        y=numerical_u_x_slice_y,
        ax=ax[0],
        marker="x",
        color="black",
        label="Numerical",
    )
    ax[0].legend().remove()
    ax[0].set_title(f"Velocity Profile (Ha={cfg.hartmann_number})")
    ax[0].set_xlabel(r"$y / H$")
    ax[0].set_ylabel(r"$\bar u_x / U$")

    sns.lineplot(
        x=analytical_y,
        y=analytical_u_x_slice_z,
        ax=ax[1],
        color=palette[0],
        label="Analytical",
    )
    sns.scatterplot(
        x=numerical_y,
        y=numerical_u_x_slice_z,
        ax=ax[1],
        marker="x",
        color="black",
        label="Numerical",
    )
    ax[1].legend().remove()
    ax[1].set_title(f"Velocity Profile (Ha={cfg.hartmann_number})")
    ax[1].set_xlabel(r"$z / H$")
    ax[1].set_ylabel(r"$\bar u_x / U$")

    if cfg.domain.duct == "quarter":
        ax[0].set_xlim(0.0, cfg.domain.L)
        ax[1].set_xlim(0.0, cfg.domain.L)

    handles, labels = ax[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=4,
        fancybox=False,
        shadow=False,
        frameon=False,
    )
    fig.subplots_adjust(
        top=0.88,
        bottom=0.2,
        left=0.1,
        right=0.98,
    )
    plt.savefig(f"profiles_{step:07d}.png", format="png")
    plt.close(fig)

def plot_u_x_over_x(domain: PISOtorch.Domain, step: int, y: float = 0.96, z: float = 0.0):
    block = domain.getBlocks()[0]
    centers = get_cell_centers(block.vertexCoordinates)  # [3, NZ, NY, NX]
    x_centers = centers[0, 0, 0, :].cpu().numpy()  # [NX]
    y_centers = centers[1, 0, :, 0].cpu().numpy()  # [NY]
    z_centers = centers[2, :, 0, 0].cpu().numpy()  # [NZ]

    iy = int(np.argmin(np.abs(y_centers - y)))
    iz = int(np.argmin(np.abs(z_centers - z)))

    u_x = block.velocity[0, 0, iz, iy, :].cpu().numpy()  # [NX]

    fig, ax = plt.subplots(figsize=(FIGWIDTH, 4))
    ax.plot(x_centers, u_x, marker="s", label=f"y={y_centers[iy]:.3f}, z={z_centers[iz]:.3f}")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u_x$")
    ax.set_title(f"Streamwise velocity at y={y_centers[iy]:.3f}, z={z_centers[iz]:.3f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"u_x_over_x_{step:07d}.png", format="png")
    plt.close(fig)


@hydra.main(version_base="1.3", config_path="../configs", config_name="validation_duct")
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

    domain = make_domain(cfg, dtype=dtype, device=device)
    start_step = 0
    logs = []

    # ----------------------------------------------------------
    # Domain Setup
    # ----------------------------------------------------------
    if cfg.get("load_domain", False):
        loaded_domain = load_domain(Path("domain"), device=device, dtype=dtype)
        for block, loaded_block in zip(domain.getBlocks(), loaded_domain.getBlocks()):
            block.setVelocity(loaded_block.velocity)
            block.setPressure(loaded_block.pressure)
            block.setEpot(loaded_block.epot)
        domain.PrepareSolve()

        metadata = load_metadata()
        start_step = metadata["step"] + 1
        logs = list(pd.read_csv("logs.csv").to_dict(orient="records"))
        logger.info(f"Resuming from saved domain at step {metadata['step']}.")

    ndims = domain.getSpatialDims()

    # Log number of cells resolved within Hartmann and Shercliff boundary layers
    # B field is in z → Hartmann walls at ±z, Shercliff walls at ±y
    # Theoretical thicknesses: δ_Ha ~ Ha⁻¹, δ_Sh ~ Ha⁻¹/². Counted at the +y/+z
    # wall only, so the number reflects the near-wall resolution and does not
    # double when the domain covers both walls (full vs quarter duct)
    _block = domain.getBlocks()[0]
    _centers = get_cell_centers(_block.vertexCoordinates)
    _y_centers = _centers[1, 0, :, 0].cpu().numpy()
    _z_centers = _centers[2, :, 0, 0].cpu().numpy()
    _ha_thickness = cfg.domain.L / cfg.hartmann_number
    _sh_thickness = cfg.domain.L / cfg.hartmann_number ** 0.5
    _n_hartmann = int(np.sum(_z_centers > cfg.domain.L - _ha_thickness))
    _n_shercliff = int(np.sum(_y_centers > cfg.domain.L - _sh_thickness))
    logger.info(
        f"Boundary layer cells — "
        f"Hartmann (δ={_ha_thickness:.4f}): {_n_hartmann} cell(s) | "
        f"Shercliff (δ={_sh_thickness:.4f}): {_n_shercliff} cell(s)"
    )

    if cfg.dynamic_forcing and cfg.get("init_with_analytical_profile", True):
        logger.info("Computing initial dynamic forcing to maintain mean velocity.")

        # We init the forcing s.t. we have a mean velocity of 1
        profile = get_shercliff_profile(
            pressure_gradient=-1.0,
            y=np.linspace(-cfg.domain.L, cfg.domain.L, 100),
            z=np.linspace(-cfg.domain.L, cfg.domain.L, 100),
            cfg=cfg,
            d_B=cfg.get("hartmann_Cw", 0.0),
        )
        mean_velocity = profile.mean()
        current_forcing = [cfg.domain.U / mean_velocity]
        forcing_active = [True]
        logger.info(f"Initial forcing set to {current_forcing[0]:.3f} to achieve mean velocity of {cfg.domain.U}.")
    else:
        current_forcing = [-cfg.pressure_gradient]
        if cfg.get("forcing_start_step", 0) > 0:
            forcing_active = [False]
        else:
            forcing_active = [True]

    def apply_forcing(domain, time_step, **kwargs):
        u_x_mean = get_mean_x_velocity(domain.getBlocks()[0])
        if cfg.dynamic_forcing and forcing_active[0]:
            current_forcing[0] += cfg.forcing_factor * (cfg.domain.U - u_x_mean).item()

        S = torch.tensor(
            [[current_forcing[0]] + [0.0] * (ndims - 1)],
            dtype=domain.getDtype(),
            device=domain.getDevice(),
        )
        domain.getBlocks()[0].setVelocitySource(S)
        domain.UpdateDomainData()

    # ----------------------------------------------------------
    # Simulation Setup
    # ----------------------------------------------------------
    prep_fn = {Hook.PRE_VELOCITY_SETUP: apply_forcing}
    sim = make_mhd_simulation(cfg, domain, prep_fn=prep_fn)

    # ----------------------------------------------------------
    # Validation
    # ----------------------------------------------------------
    logs = []
    for step in range(start_step, cfg.max_steps):
        if cfg.get("forcing_start_step", 0) and step >= cfg.forcing_start_step and not forcing_active[0]:
            forcing_active[0] = True
            logger.info(f"Activating forcing at step {step}.")

        ok = sim.single_step()

        if not ok:
            logger.error(f"Solver failed to converge at step {step}.")

        if step % cfg.log_interval == 0:
            plot_domain(cfg, domain, step)
            plot_profiles(cfg, domain, step, pressure_gradient=-current_forcing[0])
            plot_u_x_over_x(domain, step)
            block = domain.getBlocks()[0]
            u_x_mean = get_mean_x_velocity(block)
            fluct = compute_velocity_fluctuations(block, cfg, pressure_gradient=-current_forcing[0])
            logs += [{
                "step": step,
                "mean_x_velocity": u_x_mean.item(),
                "forcing": current_forcing[0],
                "fluct_streamwise": fluct["streamwise"],
                "fluct_transverse": fluct["transverse"],
                "fluct_perp": fluct["perp"],
                "fluct_para": fluct["para"],
            }]
            logger.info(
                f"Step {step}: Mean x-velocity = {u_x_mean.item():.6f} | "
                f"Forcing = {current_forcing[0]:.3f} | "
                f"<u'x²> = {fluct['streamwise']:.6e} | "
                f"<u'⊥²+u'‖²> = {fluct['transverse']:.6e} | "
                f"<u'⊥²> = {fluct['perp']:.6e} | "
                f"<u'‖²> = {fluct['para']:.6e}"
            )
            pd.DataFrame(logs).to_csv("logs.csv", index=False)
            save_domain(domain, Path("domain"))
            save_metadata({"step": step})

    # ----------------------------------------------------------
    # Save and Plot Results
    # ----------------------------------------------------------
    # TODO

if __name__ == "__main__":
    main()
