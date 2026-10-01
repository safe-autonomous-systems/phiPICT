"""2D Hartmann flow: pressure-driven channel flow in a transverse magnetic field.

The channel is periodic in x and bounded by insulating walls at y = +-1. The
field points along y. With a constant pressure gradient G the fully developed
profile is ``u(y) = G/N * (1 - cosh(Ha*y) / cosh(Ha))`` with ``N = Ha^2 / Re``.

Usage:
    python examples/hartmann.py --ha 10
"""

import argparse
import math

import torch

import phipict
import phipict.meshing as pm
from phipict import Hook, Hooks


def make_mesh(nx: int, ny: int, length: float) -> pm.Mesh:
    """Channel periodic in x between insulating no-slip walls at y = +-1."""
    walls = pm.Patch("walls", pm.Wall())  # insulating by default
    block = pm.make_box(
        (0.0, -1.0),
        (length, 1.0),
        cells=(nx, ny),
        # refined towards both walls to resolve the Hartmann layers
        grading=(None, pm.Symmetric(10.0)),
        patches=pm.FacePatches(y_minus=walls, y_plus=walls),
        name="Block",
    )
    mesh = pm.Mesh([block])
    mesh.make_periodic("x")
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ha", type=float, default=10.0)
    parser.add_argument("--re", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    dtype, device = torch.float64, torch.device("cuda")
    mesh = make_mesh(nx=100, ny=50, length=10.0)
    domain = mesh.get_domain(
        viscosity=1.0 / args.re, dtype=dtype, device=device, name="Hartmann"
    )
    block = domain.getBlocks()[0]

    stuart = args.ha**2 / args.re
    gradient = stuart  # gives a centerline velocity of ~1

    # The Lorentz force is added on top of the velocity source, so the source
    # has to be reset before every step
    def apply_forcing(domain: phipict.Domain, **kwargs: object) -> None:
        source = torch.tensor([[gradient, 0.0]], dtype=dtype, device=device)
        domain.getBlocks()[0].setVelocitySource(source)
        domain.UpdateDomainData()

    sim = phipict.MHDSimulation(
        domain=domain,
        dt=5e-3,
        stuart_number=torch.tensor(stuart),
        e_b=torch.tensor([0.0, 1.0, 0.0], dtype=dtype),
        substeps="ADAPTIVE",
        non_orthogonal=False,
        hooks=Hooks().append(Hook.PRE_VELOCITY_SETUP, apply_forcing),
        pressure_tol=phipict.SolverTolerance(rtol=1e-6, atol=1e-14),
        potential_tol=phipict.SolverTolerance(rtol=1e-8, atol=1e-14),
    )
    sim.make_divergence_free()

    for _ in range(args.steps):
        sim.single_step()

    y = mesh.blocks[0].cell_centers()[1, :, 0].to(device)
    u = block.velocity[0, 0].mean(dim=-1)
    exact = gradient / stuart * (1 - torch.cosh(args.ha * y) / math.cosh(args.ha))
    err = (u - exact).abs().max() / exact.abs().max()
    print(f"Ha={args.ha}: u_max={u.max():.4f}, max. relative error={err:.2e}")


if __name__ == "__main__":
    main()
