"""3D MHD duct flow: Shercliff (insulating walls) and Hunt (conducting walls).

A square duct, periodic in x, with walls at y = +-1 (Hartmann walls, normal to
the field) and z = +-1 (side walls). With ``--cw 0`` all walls are insulating
(Shercliff flow). With ``--cw > 0`` the Hartmann walls are thin conducting walls
with wall conductance ratio ``Cw`` (Hunt flow).

The result is compared to the analytical solution of Hunt (1965) at the duct
centerline.

Usage:
    python examples/duct.py --cw 0      # Shercliff
    python examples/duct.py --cw 0.1    # Hunt
"""

import argparse

import numpy as np
import torch

import phipict
from phipict import Face, Hook, Hooks, bc
from phipict.grid import shapes
from phipict.grid.helpers import get_cell_centers


def make_domain(
    re: float, cw: float, dtype: torch.dtype, device: torch.device
) -> phipict.Domain:
    nx, ny, nz, length = 50, 40, 40, 2.0
    # Strong refinement towards the thin Hartmann layers (y), milder towards the
    # side layers (z)
    y_weights = shapes.make_weights("simple", res=ny, grading=200, refinement="BOTH")
    z_weights = shapes.make_weights("simple", res=nz, grading=50, refinement="BOTH")
    grid = shapes.generate_grid_vertices_2D(
        [ny + 1, nx + 1],
        [(0.0, -1.0), (length, -1.0), (0.0, 1.0), (length, 1.0)],
        x_weights=y_weights,
        dtype=dtype,
    )
    grid = shapes.extrude_grid_z(
        grid, res_z=nz, start_z=-1.0, end_z=1.0, weights_z=z_weights
    )
    grid = grid.to(device).contiguous()

    viscosity = torch.tensor([1.0 / re], dtype=dtype)
    domain = phipict.Domain(3, viscosity, name="Duct", device=device, dtype=dtype)
    block = domain.CreateBlock(vertexCoordinates=grid, name="Block")
    for side in ("-y", "+y", "-z", "+z"):
        block.CloseBoundary(side)
    if cw > 0.0:
        # Thin conducting Hartmann walls: dphi/dn = Cw * laplace_tau(phi)
        for face in (Face.Y_MINUS, Face.Y_PLUS):
            bc.set_bc(block, face, bc.Potential.ThinWall(cw=cw))
    block.MakePeriodic("x")
    domain.PrepareSolve()
    return domain


def hunt_profile(
    y: np.ndarray, z: np.ndarray, ha: float, re: float, gradient: float, cw: float
) -> np.ndarray:
    """Hunt (1965) solution, Eqs. 5.8-5.11, for a square duct (Cw=0: Shercliff)."""
    Y, Z = np.meshgrid(y, z, indexing="ij")
    u = np.zeros_like(Y)
    for k in range(500):
        a = (k + 0.5) * np.pi
        n = np.sqrt(ha**2 + 4 * a**2)
        r1, r2 = 0.5 * (ha + n), 0.5 * (-ha + n)
        e = lambda r, s: np.exp(-r * (1 - s)) + np.exp(-r * (1 + s))  # noqa: E731
        t1 = (1 - np.exp(-2 * r1)) / (1 + np.exp(-2 * r1))
        t2 = (1 - np.exp(-2 * r2)) / (1 + np.exp(-2 * r2))
        den = 1 + np.exp(-2 * (r1 + r2))
        v2 = (
            (cw * r2 + t2)
            * e(r1, Y)
            / 2
            / ((1 + np.exp(-2 * r1)) / 2 * cw * n + den / (1 + np.exp(-2 * r2)))
        )
        v3 = (
            (cw * r1 + t1)
            * e(r2, Y)
            / 2
            / ((1 + np.exp(-2 * r2)) / 2 * cw * n + den / (1 + np.exp(-2 * r1)))
        )
        u += 2 * (-1) ** k * np.cos(a * Z) / a**3 * (1 - v2 - v3)
    return re * gradient * u


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ha", type=float, default=20.0)
    parser.add_argument("--re", type=float, default=9.0)
    parser.add_argument("--cw", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    dtype, device = torch.float64, torch.device("cuda")
    domain = make_domain(args.re, args.cw, dtype=dtype, device=device)
    block = domain.getBlocks()[0]
    gradient = 1.0

    def apply_forcing(domain: phipict.Domain, **kwargs: object) -> None:
        source = torch.tensor([[gradient, 0.0, 0.0]], dtype=dtype, device=device)
        domain.getBlocks()[0].setVelocitySource(source)
        domain.UpdateDomainData()

    sim = phipict.MHDSimulation(
        domain=domain,
        dt=5e-3,
        stuart_number=torch.tensor(args.ha**2 / args.re),
        e_b=torch.tensor([0.0, 1.0, 0.0], dtype=dtype),
        substeps="ADAPTIVE",
        non_orthogonal=False,
        hooks=Hooks().append(Hook.PRE_VELOCITY_SETUP, apply_forcing),
        pressure_tol=phipict.SolverTolerance(rtol=1e-6, atol=1e-14),
        potential_tol=phipict.SolverTolerance(rtol=1e-8, atol=1e-14),
        potential_use_preconditioner=True,  # AMG-preconditioned CG, needs pyamg
    )
    sim.make_divergence_free()

    for _ in range(args.steps):
        sim.single_step()

    # velocity has shape [1, 3, nz, ny, nx]
    centers = get_cell_centers(block.vertexCoordinates)
    y, z = centers[1, 0, :, 0].cpu().numpy(), centers[2, :, 0, 0].cpu().numpy()
    u = block.velocity[0, 0].mean(dim=-1).T.cpu().numpy()  # [ny, nz]
    exact = hunt_profile(y, z, args.ha, args.re, gradient, args.cw)
    err = np.abs(u - exact).max() / np.abs(exact).max()
    print(
        f"Ha={args.ha}, Cw={args.cw}: u_max={u.max():.4f} "
        f"(exact {exact.max():.4f}), max. relative error={err:.2e}"
    )


if __name__ == "__main__":
    main()
