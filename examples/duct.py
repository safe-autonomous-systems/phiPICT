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
import phipict.meshing as pm
from phipict import Hook, Hooks, bc


def make_mesh(cw: float) -> pm.Mesh:
    """Square duct, periodic in x, with walls at y = +-1 and z = +-1."""
    nx, ny, nz, length = 50, 40, 40, 2.0
    # Thin conducting Hartmann walls: dphi/dn = Cw * laplace_tau(phi)
    thin_wall = (bc.Potential.ThinWall(cw=cw),) if cw > 0.0 else ()
    hartmann_walls = pm.Patch("hartmann_walls", pm.Wall(extra=thin_wall))
    side_walls = pm.Patch("side_walls", pm.Wall())  # insulating
    block = pm.make_box(
        (0.0, -1.0, -1.0),
        (length, 1.0, 1.0),
        cells=(nx, ny, nz),
        # Strong refinement towards the thin Hartmann layers (y), milder towards
        # the side layers (z)
        grading=(None, pm.Symmetric(200.0), pm.Symmetric(50.0)),
        patches=pm.FacePatches(
            y_minus=hartmann_walls,
            y_plus=hartmann_walls,
            z_minus=side_walls,
            z_plus=side_walls,
        ),
        name="Block",
    )
    mesh = pm.Mesh([block])
    mesh.make_periodic("x")
    return mesh


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
    mesh = make_mesh(args.cw)
    domain = mesh.get_domain(
        viscosity=1.0 / args.re, dtype=dtype, device=device, name="Duct"
    )
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
    centers = mesh.blocks[0].cell_centers()
    y, z = centers[1, 0, :, 0].numpy(), centers[2, :, 0, 0].numpy()
    u = block.velocity[0, 0].mean(dim=-1).T.cpu().numpy()  # [ny, nz]
    exact = hunt_profile(y, z, args.ha, args.re, gradient, args.cw)
    err = np.abs(u - exact).max() / np.abs(exact).max()
    print(
        f"Ha={args.ha}, Cw={args.cw}: u_max={u.max():.4f} "
        f"(exact {exact.max():.4f}), max. relative error={err:.2e}"
    )


if __name__ == "__main__":
    main()
