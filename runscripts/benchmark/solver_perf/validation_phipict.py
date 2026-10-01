"""Physics validation (Hartmann / Hunt / Shercliff) against the phipict API.

Port of ``runscripts/validation/validation.py`` (which still imports the removed
``fluidgym.simulation`` modules) for the solver performance work: identical case
setup and evaluation, plus a ``+solver_stack`` switch:

- ``legacy``: legacy cuBLAS/cuSPARSE solvers, pure-torch AMG with its own setup,
- ``new``: fused Krylov backend, native AMG (fp32 V-cycle), pressure AMG in auto mode,
- ``new_amg``: as ``new`` with AMG forced on for the pressure solve.

Usage (outputs under output/solver_perf/validation/):
    python runscripts/benchmark/solver_perf/validation_phipict.py case=hartmann_10 \
        +solver_stack=new hydra.run.dir=output/solver_perf/validation/new/hartmann/10.0
"""

import json
from pathlib import Path
import logging
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from omegaconf import DictConfig, OmegaConf

import phipict.grid.resample as resampling
import phipict.grid.shapes as shapes
from fluidgym import DEFAULT_PALETTE
from phipict.grid.shapes import make_weights
from phipict import _C as PISOtorch
from phipict.grid.helpers import get_cell_centers, get_cell_size
from phipict.simulation.mhd import MHDSimulation
from phipict.io.domain_io import load_domain, save_domain
from phipict.io.output import plot_grid
from phipict.simulation.sgs import append_sgs_viscosity_prep_fn
from phipict.simulation.simulation import Simulation
from phipict.solvers.tolerance import SolverTolerance
from phipict import Face, Hook, Hooks, bc

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


def get_mean_x_velocity(
    block: PISOtorch.Block, cell_sizes: torch.Tensor
) -> torch.Tensor:
    u_x: torch.Tensor = block.velocity[0, 0, ...]
    cell_sizes = cell_sizes.squeeze()
    u_x_mean = (u_x * cell_sizes).sum() / cell_sizes.sum()
    return u_x_mean


def get_hartmann_profile(y: np.ndarray, cfg: DictConfig) -> np.ndarray:
    Ha = cfg.hartmann_number

    if Ha == 0:
        # Poiseuille profile
        return 1.5 * cfg.domain.U * (1 - (y / cfg.domain.H) ** 2)

    # Hartmann profile
    return (
        cfg.domain.U
        * (np.cosh(Ha) - np.cosh(Ha * y / cfg.domain.H))
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

    YY, ZZ = np.meshgrid(y, z, indexing="ij")  # [NY, NZ]

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


def make_domain(
    cfg: DictConfig, dtype: torch.dtype, device: torch.device
) -> PISOtorch.Domain:
    ndims = 2 if cfg.type == "hartmann" else 3

    y_weights = make_weights(
        grading_type=cfg.domain.grading_y_type,
        res=cfg.domain.ny,
        grading=cfg.domain.grading_y,
        refinement="BOTH",
    )

    grid = shapes.generate_grid_vertices_2D(
        [cfg.domain.ny + 1, cfg.domain.nx + 1],
        [
            (0.0, -cfg.domain.H),
            (cfg.domain.L, -cfg.domain.H),
            (0.0, cfg.domain.H),
            (cfg.domain.L, cfg.domain.H),
        ],
        None,
        x_weights=y_weights,
        y_weights=None,
        dtype=dtype,
    )
    fig, ax = plt.subplots(figsize=(6, 4))


    if cfg.get("rotate_grid_deg", 0.0) > 0.0:
        distance_scaling = shapes.make_rotation_distance_scaling_fn_sine_half(
            cfg.rotate_grid_deg,
            1.0
        )
        grid = shapes.rotate_grid(
            grid,
            angle=cfg.rotate_grid_deg,
            distance_scaling=distance_scaling,
            distance_axes=[5.0, 1.0]
        )

    if cfg.type == "shercliff" or cfg.type == "hunt":
        z_weights = make_weights(
            grading_type=cfg.domain.grading_z_type,
            res=cfg.domain.nz,
            grading=cfg.domain.grading_z,
            refinement="BOTH",
        )
        grid = shapes.extrude_grid_z(
            grid,
            res_z=cfg.domain.nz,
            start_z=-cfg.domain.H,
            end_z=cfg.domain.H,
            weights_z=z_weights,
        )

        # We extract a y-z slice
        _plot_grid = torch.stack([grid[:, 1, :, :, 0], grid[:, 2, :, :, 0]], dim=1)
    else:
        _plot_grid = grid

    plot_grid(_plot_grid, ax=ax)
    plt.savefig("grid.pdf", format="pdf")
    plt.close(fig)

    grid = grid.to(device).contiguous()

    kinematic_viscosity = torch.tensor(
        [(cfg.domain.U * cfg.domain.H) / cfg.reynolds_number],
        dtype=dtype,
        device=device,
    )

    domain = PISOtorch.Domain(
        ndims,
        kinematic_viscosity.cpu(),
        name="Domain",
        device=device,
        dtype=dtype,
    )

    block = domain.CreateBlock(vertexCoordinates=grid, name="Block")

    block.CloseBoundary("-y")
    block.CloseBoundary("+y")

    if cfg.type == "hunt":
        # Conducting walls
        bc.set_bc(block, Face.Y_MINUS, bc.Potential.ThinWall(cw=cfg.hartmann_Cw))
        bc.set_bc(block, Face.Y_PLUS, bc.Potential.ThinWall(cw=cfg.hartmann_Cw))

    if ndims == 3:
        block.CloseBoundary("-z")
        block.CloseBoundary("+z")

    block.MakePeriodic("x")

    domain.PrepareSolve()

    return domain


def set_advection_scheme(cfg: DictConfig, domain: PISOtorch.Domain) -> None:
    """Select the convective scheme for the momentum equation.

    ``central`` (the default) is the original discretization. It has exactly zero
    dissipation at the 2*dx mode, so grid-scale oscillations shed by a steep shear
    layer have no sink and persist under refinement. ``linear_upwind`` damps that
    mode while leaving resolved scales alone.
    """
    scheme = cfg.get("advection_scheme", "central")
    try:
        domain.setAdvectionScheme(getattr(PISOtorch.AdvectionScheme, scheme.upper()))
    except AttributeError as e:
        raise ValueError(
            f"Unknown advection_scheme '{scheme}'. Expected one of: "
            "central, linear_upwind."
        ) from e


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


def make_simulation(
    cfg: DictConfig, domain: PISOtorch.Domain, hooks: Hooks
) -> Simulation:
    if cfg.hartmann_number > 0.0:
        raise ValueError("Hartmann number must be = 0 for standard simulation.")

    set_advection_scheme(cfg, domain)

    tolerances = build_tolerances(cfg)
    tolerances.pop("potential_tol", None)

    sim = Simulation(
        domain=domain,
        dt=cfg.sim.dt,
        substeps="ADAPTIVE",
        corrector_steps=cfg.sim.corrector_steps,
        non_orthogonal=False,
        hooks=hooks,
        pressure_warm_start=bool(cfg.get("pressure_warm_start", False)),
        **stack_sim_kwargs(cfg),
        **tolerances,
    )
    sim.make_divergence_free()
    return sim


def plot_hartmann_case(
    hartmann_number: float,
    analytical_x: np.ndarray,
    analytical_y: np.ndarray,
    numerical_x: np.ndarray,
    numerical_y: np.ndarray,
) -> None:
    palette = sns.color_palette()

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(
        x=analytical_x, y=analytical_y, label="Analytical", ax=ax, color=palette[0]
    )
    sns.scatterplot(
        x=numerical_x,
        y=numerical_y,
        label="Numerical",
        ax=ax,
        marker="x",
        color="black",
    )
    ax.legend().remove()
    ax.set_title(f"Velocity Profile (Ha={hartmann_number})")
    ax.set_xlabel(r"$y / H$")
    ax.set_ylabel(r"$\bar u_x / U$")

    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=2,
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
    plt.savefig("hartmann_profile.pdf", format="pdf")


def plot_shercliff_case(
    hartmann_number: float,
    analytical_y: np.ndarray,
    analytical_z: np.ndarray,
    analytical_ux_profile: np.ndarray,  # [Y, Z]
    numerical_y: np.ndarray,
    numerical_z: np.ndarray,
    numerical_ux_profile: np.ndarray,  # [NY, NZ]
    cell_sizes: np.ndarray,
) -> None:
    cell_sizes = cell_sizes[0, 0, ..., 0].T  # [NY, NZ]

    analytical_u_x_mean_y = analytical_ux_profile.mean(axis=1)  # [NY]
    analytical_u_x_mean_z = analytical_ux_profile.mean(axis=0)  # [NZ]

    numerical_u_x_mean_y = (numerical_ux_profile * cell_sizes).mean(
        axis=1
    ) / cell_sizes.mean(axis=1)
    numerical_u_x_mean_z = (numerical_ux_profile * cell_sizes).mean(
        axis=0
    ) / cell_sizes.mean(axis=0)

    palette = sns.color_palette()

    # -----------------------------------------------------------------------
    # Velocity Profiles
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    sns.lineplot(
        x=analytical_y,
        y=analytical_u_x_mean_y,
        ax=ax[0],
        color=palette[0],
        label="Analytical",
    )
    sns.scatterplot(
        x=numerical_y,
        y=numerical_u_x_mean_y,
        ax=ax[0],
        marker="x",
        color="black",
        label="Numerical",
    )
    ax[0].legend().remove()
    ax[0].set_title(f"Velocity Profile (Ha={hartmann_number})")
    ax[0].set_xlabel(r"$y / H$")
    ax[0].set_ylabel(r"$\bar u_x / U$")

    sns.lineplot(
        x=analytical_z,
        y=analytical_u_x_mean_z,
        ax=ax[1],
        color=palette[0],
        label="Analytical",
    )
    sns.scatterplot(
        x=numerical_z,
        y=numerical_u_x_mean_z,
        ax=ax[1],
        marker="x",
        color="black",
        label="Numerical",
    )
    ax[1].legend().remove()
    ax[1].set_title(f"Velocity Profile (Ha={hartmann_number})")
    ax[1].set_xlabel(r"$z / H$")
    ax[1].set_ylabel(r"$\bar u_x / U$")

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
    plt.savefig("shercliff_profile.pdf", format="pdf")

    # -----------------------------------------------------------------------
    # Boundary Layers
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    d_s, d_Ha = BOUNDARY_THICKNESS.get(hartmann_number, (0.1, 0.01))

    analytical_y_idx = analytical_y >= 1 - d_Ha
    analytical_y = analytical_y[analytical_y_idx]
    analytical_u_x_mean_y = analytical_u_x_mean_y[analytical_y_idx]

    analytical_z_idx = analytical_z >= 1 - d_s
    analytical_z = analytical_z[analytical_z_idx]
    analytical_u_x_mean_z = analytical_u_x_mean_z[analytical_z_idx]

    numerical_y_idx = numerical_y >= 1 - d_Ha
    numerical_y = numerical_y[numerical_y_idx]
    numerical_u_x_mean_y = numerical_u_x_mean_y[numerical_y_idx]

    numerical_z_idx = numerical_z >= 1 - d_s
    numerical_z = numerical_z[numerical_z_idx]
    numerical_u_x_mean_z = numerical_u_x_mean_z[numerical_z_idx]

    sns.lineplot(
        x=analytical_y,
        y=analytical_u_x_mean_y,
        ax=ax[0],
        color=palette[0],
        label="Analytical",
    )
    sns.scatterplot(
        x=numerical_y,
        y=numerical_u_x_mean_y,
        ax=ax[0],
        marker="x",
        color="black",
        label="Numerical",
    )
    ax[0].legend().remove()
    ax[0].set_title(f"Velocity Profile (Ha={hartmann_number})")
    ax[0].set_xlabel(r"$y / H$")
    ax[0].set_ylabel(r"$\bar u_x / U$")

    sns.lineplot(
        x=analytical_z,
        y=analytical_u_x_mean_z,
        ax=ax[1],
        color=palette[0],
        label="Analytical",
    )
    sns.scatterplot(
        x=numerical_z,
        y=numerical_u_x_mean_z,
        ax=ax[1],
        marker="x",
        color="black",
        label="Numerical",
    )
    ax[1].legend().remove()
    ax[1].set_title(f"Velocity Profile (Ha={hartmann_number})")
    ax[1].set_xlabel(r"$z / H$")
    ax[1].set_ylabel(r"$\bar u_x / U$")

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
    plt.savefig("hartmann_layers.pdf", format="pdf")


def make_mhd_simulation(
    cfg: DictConfig, domain: PISOtorch.Domain, hooks: Hooks
) -> MHDSimulation:
    if cfg.hartmann_number == 0.0:
        raise ValueError("Hartmann number must be > 0 for MHD simulation.")

    set_advection_scheme(cfg, domain)

    e_b = torch.tensor([0.0, 1.0, 0.0], dtype=domain.getDtype())

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
            else ({"reuse_interpolation": False, "native_dtype": None} if str(cfg.get("solver_stack", "new")) == "legacy" else None)
        ),
        pressure_warm_start=bool(cfg.get("pressure_warm_start", False)),
        hooks=hooks,
        adaptive_CFL=cfg.sim.adaptive_cfl,
        **stack_sim_kwargs(cfg),
        **build_tolerances(cfg),
    )
    sim.linear_solve_max_iterations = cfg.sim.pressure_solve_max_iter
    sim.make_divergence_free()
    return sim

@hydra.main(version_base="1.3", config_path="../../configs", config_name="validation")
def main(cfg: DictConfig):
    try:
        run_validation(cfg)
    except Exception as e:
        logger.exception("Job crashed with an exception")
        logger.error(e)
        raise


def configure_solver_stack(cfg: DictConfig) -> None:
    from phipict.solvers import amg

    stack = str(cfg.get("solver_stack", "new"))
    logger.info("Solver stack: %s", stack)
    if stack == "legacy":
        PISOtorch.SetKrylovSettings(enabled=False)
        amg.USE_NATIVE = False
    elif stack in ("new", "new_amg"):
        PISOtorch.SetKrylovSettings(enabled=True)
        amg.USE_NATIVE = True
    else:
        raise ValueError(f"unknown solver_stack {stack!r}")


def stack_sim_kwargs(cfg: DictConfig) -> dict[str, Any]:
    stack = str(cfg.get("solver_stack", "new"))
    if stack == "legacy":
        return {"pressure_use_amg": False}
    if stack == "new_amg":
        return {"pressure_use_amg": True}
    return {}


def run_validation(cfg: DictConfig):
    configure_solver_stack(cfg)
    dtype = torch.float64 if cfg.precision == "double" else torch.float32
    device = torch.device("cuda")

    # -----------------------------------------------------------------------
    # Domain Setup
    # -----------------------------------------------------------------------
    if cfg.get("load_domain", False):
        domain = load_domain(Path("domain"), device=device, dtype=dtype)
        domain.PrepareSolve()
        metadata = load_metadata()
        start_step = metadata["step"] + 1
        logger.info(f"Resuming from saved domain at step {metadata['step']}.")
    else:
        domain = make_domain(cfg, dtype=dtype, device=device)
        start_step = 0
    ndims = domain.getSpatialDims()
    block = domain.getBlocks()[0]
    cell_sizes = get_cell_size(block)

    # Log number of cells resolved within Hartmann and Shercliff boundary layers
    # B field is in y → Hartmann walls at ±y, Shercliff walls at ±z
    # Theoretical thicknesses: δ_Ha ~ a·Ha⁻¹, δ_Sh ~ a·Ha⁻¹/²
    if cfg.hartmann_number > 0.0 and ndims == 3:
        _centers = get_cell_centers(block.vertexCoordinates)
        _y_centers = _centers[1, 0, :, 0].cpu().numpy()
        _z_centers = _centers[2, :, 0, 0].cpu().numpy()
        _ha_thickness = cfg.domain.H / cfg.hartmann_number
        _sh_thickness = cfg.domain.H / cfg.hartmann_number ** 0.5
        _n_hartmann = int(np.sum(
            (_y_centers > cfg.domain.H - _ha_thickness) | (_y_centers < -cfg.domain.H + _ha_thickness)
        ))
        _n_shercliff = int(np.sum(
            (_z_centers > cfg.domain.H - _sh_thickness) | (_z_centers < -cfg.domain.H + _sh_thickness)
        ))
        logger.info(
            f"Boundary layer cells — "
            f"Hartmann ({_ha_thickness:.4f}): {_n_hartmann} cell(s) | "
            f"Shercliff ({_sh_thickness:.4f}): {_n_shercliff} cell(s)"
        )

    # We keep a mutable list for the current forcing
    current_forcing = [-cfg.pressure_gradient]

    def apply_forcing(domain, time_step, **kwargs):
        u_x_mean = get_mean_x_velocity(domain.getBlocks()[0], cell_sizes)
        if cfg.dynamic_forcing:
            current_forcing[0] += cfg.forcing_factor * (cfg.domain.U - u_x_mean).item()

        S = torch.tensor(
            [[current_forcing[0]] + [0.0] * (ndims - 1)],
            dtype=domain.getDtype(),
            device=domain.getDevice(),
        )
        domain.getBlocks()[0].setVelocitySource(S)
        domain.UpdateDomainData()

    # -----------------------------------------------------------------------
    # Simulation Setup
    # -----------------------------------------------------------------------
    hooks = Hooks().append(Hook.PRE_VELOCITY_SETUP, apply_forcing)

    sgs_model = cfg.get("sgs_model", "none")
    append_sgs_viscosity_prep_fn(
        hooks,
        model=sgs_model,
        coefficient=cfg.get("sgs_coefficient", 0.325),
        ndims=ndims,
        dtype=dtype,
        cuda_device=device,
        cpu_device=torch.device("cpu"),
    )

    if cfg.hartmann_number == 0.0:
        sim = make_simulation(cfg, domain, hooks=hooks)
    else:
        sim = make_mhd_simulation(cfg, domain, hooks=hooks)

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------
    last_velocity = block.velocity.clone()
    for step in range(start_step, cfg.max_steps):
        ok = sim.single_step()

        if not ok:
            logger.error(f"Solver failed to converge at step {step}.")

        mean_x_vel = get_mean_x_velocity(block, cell_sizes)

        if step % cfg.log_interval == 0:
            logger.info(f"Step {step}, mean x-velocity: {mean_x_vel:.6f}, forcing: {current_forcing[0]:.6f}")
            save_domain(domain, Path("domain"))
            save_metadata({"step": step})

        if (
            step > cfg.min_steps
            and torch.abs(last_velocity - block.velocity).mean().item() < cfg.convergence_tol
        ):
            logger.info(
                f"Convergence achieved at step {step}. "
                f"Mean x-velocity: {mean_x_vel.item():.6f}"
            )
            break

    # -----------------------------------------------------------------------
    # Save and Plot Results
    # -----------------------------------------------------------------------
    grids: list[torch.Tensor] = domain.getVertexCoordinates()
    cell_centers = get_cell_centers(grids[0]).cpu().numpy()
    u_x_profile: np.ndarray = block.velocity[0, 0, ...].cpu().numpy()

    if cfg.type == "hartmann":
        analytical_x = np.linspace(-cfg.domain.H, cfg.domain.H, 100)
        analytical_y = get_hartmann_profile(analytical_x, cfg)

        if cfg.get("rotate_grid_deg", 0.0) > 0.0:
            res_x, res_y = cfg.domain.nx * 5, cfg.domain.ny * 2
            vel_resampled = resampling.sample_multi_coords_to_uniform_grid(
                data_list=[block.velocity for block in domain.getBlocks()],
                coords_list=[block.vertexCoordinates for block in domain.getBlocks()],
                out_shape=[res_x, res_y],
                fill_max_steps=8
            )
            y_centers = np.linspace(-cfg.domain.H, cfg.domain.H, res_y)
            u_x_profile: np.ndarray = vel_resampled[0, 0, ...].cpu().numpy()
        else:
            y_centers = cell_centers[1, :, 0]  # [NY]

        u_x_profile_mean = u_x_profile.mean(axis=-1)  # [NY]

        plot_hartmann_case(
            hartmann_number=cfg.hartmann_number,
            analytical_x=analytical_x,
            analytical_y=analytical_y,
            numerical_x=y_centers,
            numerical_y=u_x_profile_mean,
        )
    else:
        y_centers = cell_centers[1, 0, :, 0]  # [NY]
        z_centers = cell_centers[2, :, 0, 0]  # [NY]

        u_x_profile_mean = u_x_profile.mean(axis=-1)  # [NZ, NY]
        u_x_profile_mean = u_x_profile_mean.T  # [NY, NZ]

        analytical_y = np.linspace(-cfg.domain.H, cfg.domain.H, 1000)
        analytical_z = np.linspace(-cfg.domain.H, cfg.domain.H, 100)

        analytical_profile = get_shercliff_profile(
            pressure_gradient=cfg.pressure_gradient,
            y=analytical_y,
            z=analytical_z,
            cfg=cfg,
            d_B=cfg.get("hartmann_Cw", 0.0),
        )

        plot_shercliff_case(
            hartmann_number=cfg.hartmann_number,
            analytical_y=analytical_y,
            analytical_z=analytical_z,
            analytical_ux_profile=analytical_profile,
            numerical_y=y_centers,
            numerical_z=z_centers,
            numerical_ux_profile=u_x_profile_mean,
            cell_sizes=cell_sizes.cpu().numpy(),
        )


if __name__ == "__main__":
    main()
