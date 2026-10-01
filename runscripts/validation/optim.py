"""Learn the Stuart number by gradient descent to match a reference velocity field.

Usage:
    python runscripts/validation/optim.py domain_type=hartmann
"""

import gc
from collections.abc import Callable

import torch
from phipict import Face, Hook, bc

import fluidgym.simulation.pict.data.shapes as shapes
from fluidgym.envs.util.grid_gen import make_weights_simple_grading
from fluidgym.envs.util.profiles import get_inflow_profile
from fluidgym.simulation.extensions import PISOtorch  # type: ignore
from fluidgym.simulation.mhd_simulation import MHDSimulation

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")

import logging
import time

import hydra
import pandas as pd
from omegaconf import DictConfig

DTYPE = torch.float64
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MHD gradient tests
# ---------------------------------------------------------------------------

def make_static_forcing_fn(forcing: float) -> Callable:
    def fn(domain, time_step, **kwargs):
        ndims = domain.getSpatialDims()

        S = torch.tensor(
            [[forcing] + [0.0] * (ndims - 1)],
            dtype=domain.getDtype(),
            device=domain.getDevice(),
        )
        domain.getBlocks()[0].setVelocitySource(S)
        domain.UpdateDomainData()

    return fn


def make_domain_hartmann():
    """2D Hartmann domain matching hartmann_5.yaml exactly.

    Ha=5, Re=2, ny=50, nx=100, grading_y=10, L=10, H=1, dt=5e-3.
    """
    H, L = 1.0, 10.0
    ny, nx = 50, 100
    Re, Ha = 2.0, 5.0
    dt = 5e-3

    viscosity = torch.tensor([(1.0 * H) / Re], dtype=DTYPE, device=cpu_device)
    stuart_number = torch.tensor([Ha**2 / Re], dtype=DTYPE, device=cpu_device)

    y_weights = make_weights_simple_grading(ny, 10.0, "BOTH")
    grid = shapes.generate_grid_vertices_2D(
        [ny + 1, nx + 1],
        [(0.0, -H), (L, -H), (0.0, H), (L, H)],
        None,
        x_weights=y_weights,
        y_weights=None,
        dtype=DTYPE,
    )
    grid = grid.to(cuda_device).contiguous()
    domain = PISOtorch.Domain(2, viscosity, name="Hartmann", device=cuda_device, dtype=DTYPE)
    block = domain.CreateBlock(vertexCoordinates=grid, name="Block")
    block.CloseBoundary("-y")
    block.CloseBoundary("+y")
    block.MakePeriodic("x")

    e_b = torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE)

    velocity_profile = get_inflow_profile(h=H, res_y=ny, n_dims=2, dtype=DTYPE, device=cuda_device)
    velocity_profile = velocity_profile.expand(-1, -1, -1, nx).contiguous()
    block.setVelocity(velocity_profile)

    static_forcing = make_static_forcing_fn(forcing=1.0)
    prep_fn = {Hook.PRE_VELOCITY_SETUP: static_forcing}

    domain.PrepareSolve()

    return domain, e_b, False, prep_fn, stuart_number, dt


def make_domain_shercliff():
    """3D Shercliff domain matching shercliff_20.yaml exactly.

    Ha=20, Re=9, ny=40, nx=20, nz=40, grading_y=200, grading_z=50, L=2, H=1, dt=5e-3.
    """
    H, L = 1.0, 2.0
    ny, nx, nz = 40, 20, 40
    Re, Ha = 9.0, 20.0
    dt = 5e-3

    viscosity = torch.tensor([(1.0 * H) / Re], dtype=DTYPE, device=cpu_device)
    stuart_number = torch.tensor([Ha**2 / Re], dtype=DTYPE, device=cpu_device)

    y_weights = make_weights_simple_grading(ny, 200.0, "BOTH")
    z_weights = make_weights_simple_grading(nz, 50.0, "BOTH")
    grid = shapes.generate_grid_vertices_2D(
        [ny + 1, nx + 1],
        [(0.0, -H), (L, -H), (0.0, H), (L, H)],
        None,
        x_weights=y_weights,
        y_weights=None,
        dtype=DTYPE,
    )
    grid = shapes.extrude_grid_z(grid, res_z=nz, start_z=-H, end_z=H, weights_z=z_weights)
    grid = grid.to(cuda_device).contiguous()
    domain = PISOtorch.Domain(3, viscosity, name="Shercliff", device=cuda_device, dtype=DTYPE)
    block = domain.CreateBlock(vertexCoordinates=grid, name="Block")
    block.CloseBoundary("-y")
    block.CloseBoundary("+y")
    block.CloseBoundary("-z")
    block.CloseBoundary("+z")
    block.MakePeriodic("x")

    e_b = torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE)

    static_forcing = make_static_forcing_fn(forcing=50.0)
    prep_fn = {Hook.PRE_VELOCITY_SETUP: static_forcing}

    domain.PrepareSolve()

    return domain, e_b, False, prep_fn, stuart_number, dt


def make_domain_hunt():
    """3D Hunt domain matching hunt_20.yaml exactly.

    Ha=20, Re=9, ny=40, nx=20, nz=40, grading_y=200, grading_z=50, L=2, H=1,
    hartmann_Cw=0.1, dt=5e-3.
    """
    H, L = 1.0, 2.0
    ny, nx, nz = 40, 20, 40
    Re, Ha = 9.0, 20.0
    dt = 5e-3
    hartmann_Cw = 0.1

    viscosity = torch.tensor([(1.0 * H) / Re], dtype=DTYPE, device=cpu_device)
    stuart_number = torch.tensor([Ha**2 / Re], dtype=DTYPE, device=cpu_device)

    y_weights = make_weights_simple_grading(ny, 200.0, "BOTH")
    z_weights = make_weights_simple_grading(nz, 50.0, "BOTH")
    grid = shapes.generate_grid_vertices_2D(
        [ny + 1, nx + 1],
        [(0.0, -H), (L, -H), (0.0, H), (L, H)],
        None,
        x_weights=y_weights,
        y_weights=None,
        dtype=DTYPE,
    )
    grid = shapes.extrude_grid_z(grid, res_z=nz, start_z=-H, end_z=H, weights_z=z_weights)
    grid = grid.to(cuda_device).contiguous()
    domain = PISOtorch.Domain(3, viscosity, name="Hunt", device=cuda_device, dtype=DTYPE)
    block = domain.CreateBlock(vertexCoordinates=grid, name="Block")
    block.CloseBoundary("-y")
    block.CloseBoundary("+y")
    bc.set_bc(block, Face.Y_MINUS, bc.Potential.ThinWall(cw=hartmann_Cw))
    bc.set_bc(block, Face.Y_PLUS, bc.Potential.ThinWall(cw=hartmann_Cw))
    block.CloseBoundary("-z")
    block.CloseBoundary("+z")
    block.MakePeriodic("x")

    e_b = torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE)

    static_forcing = make_static_forcing_fn(forcing=50.0)
    prep_fn = {Hook.PRE_VELOCITY_SETUP: static_forcing}

    domain.PrepareSolve()

    return domain, e_b, False, prep_fn, stuart_number, dt


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def MSE(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a - b)**2)

def SSE(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sum((a - b)**2)

def loss_fn(
        domain: PISOtorch.Domain,
        target_domain: PISOtorch.Domain,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        loss_vel: bool = False,
        loss_temp: bool = False,
        loss_epot: bool = False,
    ) -> torch.Tensor:
    loss = torch.tensor(0.0, device=domain.getDevice())
    for block_idx in range(domain.getNumBlocks()):
        if loss_vel:
            loss = loss + loss_fn(domain.getBlock(block_idx).velocity, target_domain.getBlock(block_idx).velocity)
        if loss_temp:
            loss = loss + loss_fn(domain.getBlock(block_idx).passiveScalar, target_domain.getBlock(block_idx).passiveScalar)
    return loss

def _make_mhd_sim(
    domain: PISOtorch.Domain,
    e_b: torch.Tensor,
    is_non_ortho: bool,
    prep_fn: dict,
    stuart_number: torch.Tensor,
    dt: float,
    corr: int,
    substeps: int,
    differentiable: bool,
) -> MHDSimulation:
    return MHDSimulation(
        domain,
        dt=dt,
        stuart_number=stuart_number,
        e_b=e_b,
        corrector_steps=corr,
        substeps=substeps,
        pressure_time_step_normalized=True,
        differentiable=differentiable,
        non_orthogonal=is_non_ortho,
        advect_non_ortho_steps=2 if is_non_ortho else 1,
        pressure_non_ortho_steps=4 if is_non_ortho else 1,
        prep_fn=prep_fn,
        advection_tol=1e-5,
        pressure_tol=1e-5,
        potential_tol=1e-5,
        potential_use_preconditioner=True
    )


def test_optim_stuart_number(
    domain_fn: Callable,
    init_stuart_number: float,
    it: int,
    opt_it: int,
    lr: float = 0.1,
    corr: int = 2,
    substeps: int = 1,
    static: bool = False,
    loss_tol: float = 1e-5,
) -> pd.DataFrame:
    """Learn the Stuart number via SGD to match a target velocity field.

    The target is produced by running a non-differentiable forward sim with
    the true Stuart number from the domain function. The optimization then
    starts from ``init_stuart_number`` and differentiates through ``it``
    PISO steps per SGD iteration.
    """
    # -----------------------------------------------------------------------
    # Target: run non-differentiable sim with the true Stuart number
    # -----------------------------------------------------------------------
    target_domain, e_b, is_non_ortho, prep_fn, true_stuart_number, dt = domain_fn()

    sim_target = _make_mhd_sim(
        target_domain,
        e_b,
        is_non_ortho,
        prep_fn,
        stuart_number=true_stuart_number,
        dt=dt,
        corr=corr, substeps=substeps,
        differentiable=False,
    )
    sim_target.run(iterations=it, static=static)

    # -----------------------------------------------------------------------
    # Optimization: learn N starting from init_stuart_number
    # -----------------------------------------------------------------------

    stats = []
    start_time = time.perf_counter()

    N_learn = torch.tensor([init_stuart_number], dtype=DTYPE, requires_grad=True)
    optimizer = torch.optim.SGD([N_learn], lr=lr)

    last_loss = float("inf")
    loss_falling = True
    for step in range(opt_it):
        optimizer.zero_grad()

        # Build the differentiable sim once; sim._stuart_number holds a reference
        # to N_learn, so gradients flow through the Lorentz force each forward pass
        domain, e_b, is_non_ortho, prep_fn, _, dt = domain_fn()
        sim = _make_mhd_sim(
            domain,
            e_b,
            is_non_ortho,
            prep_fn,
            stuart_number=N_learn,
            dt=dt,
            corr=corr,
            substeps=substeps,
            differentiable=True,
        )
        sim.run(iterations=it, static=static)

        loss = SSE(
            domain.getBlock(0).velocity,
            target_domain.getBlock(0).velocity.detach(),
        )

        logger.info(
            "step %d, N=%.4f (true=%.4f), loss=%.6e",
            step, N_learn.item(), true_stuart_number.item(), loss.item(),
        )

        if loss.isnan():
            raise RuntimeError("loss is NaN.")
        if loss.item() >= last_loss and not loss.item() < 1e-5:
            loss_falling = False
        last_loss = loss.item()

        loss.backward()
        loss_val = loss.item()
        optimizer.step()

        domain.Detach()
        del loss, domain, sim
        gc.collect()
        torch.cuda.empty_cache()


        stats += [{
            "step": step,
            "time": time.perf_counter() - start_time,
            "N_learn": N_learn.item(),
            "loss": loss_val,
        }]

        if loss_val < loss_tol:
            logger.info(f"Converged at step {step} with loss {loss_val:.6e}.")
            break

    if not loss_falling:
        raise RuntimeError("Optimization test failed: loss not falling or converged.")

    df = pd.DataFrame(stats)
    df["target_N"] = true_stuart_number.item()
    return df


@hydra.main(version_base="1.3", config_path="../configs", config_name="validation_optim")
def main(cfg: DictConfig):
    try:
        run_optim(cfg)
    except Exception as e:
        logger.exception("Job crashed with an exception")
        logger.error(e)
        raise

def run_optim(cfg):
    if cfg.domain_type == "hartmann":
        domain_fn = make_domain_hartmann
    elif cfg.domain_type == "shercliff":
        domain_fn = make_domain_shercliff
    elif cfg.domain_type == "hunt":
        domain_fn = make_domain_hunt
    else:
        raise ValueError(f"Unknown domain type: {cfg.domain_type}")

    logs = test_optim_stuart_number(
        domain_fn=domain_fn,
        init_stuart_number=1.0,
        it=10,
        opt_it=5000,
        lr=0.1,
        corr=2,
        substeps=1,
        static=False,
        loss_tol=cfg.loss_tol,
    )
    logs.to_csv("optim_log.csv", index=False)


if __name__ == "__main__":
    main()
