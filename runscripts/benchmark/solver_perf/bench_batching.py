"""Throughput of batched environments: B copies of a small MHD duct in one simulation.

The case mirrors the small 2D fluidgym MHD env at the phipict level (480x64 graded cells,
periodic in x, insulating walls, passive temperature, uniform field along y, the env's
solver tolerances and warm starts). Each environment gets its own forcing and initial
perturbation. For every batch size B the time per PISO step of the batched simulation is
measured, and compared with running the B environments one after another (B=1 timing
times B).

Usage:
    python runscripts/benchmark/solver_perf/bench_batching.py --batch-sizes 1,2,4,8,16,32,64,128
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import phipict
from phipict.grid import shapes

DEV = torch.device("cuda")


def make_duct(nx: int, ny: int, dtype: torch.dtype) -> phipict.Domain:
    y_weights = shapes.make_weights("simple", res=ny, grading=10, refinement="BOTH")
    grid = shapes.generate_grid_vertices_2D(
        [ny + 1, nx + 1], [(0.0, -1.0), (18.75, -1.0), (0.0, 1.0), (18.75, 1.0)], x_weights=y_weights, dtype=dtype
    ).to(DEV)
    domain = phipict.Domain(
        2, torch.tensor([1.0 / 5000.0], dtype=dtype), name="duct", device=DEV, dtype=dtype, passiveScalarChannels=1,
        scalarViscosity=torch.tensor([1.0 / (5000.0 * 0.025)], dtype=dtype, device=DEV),
    )
    block = domain.CreateBlock(vertexCoordinates=grid, name="duct")
    t_lo = torch.ones(1, 1, 1, nx, dtype=dtype, device=DEV)
    block.CloseBoundary("-y", passiveScalar=t_lo)
    block.CloseBoundary("+y", passiveScalar=torch.zeros_like(t_lo))
    block.MakePeriodic("x")
    domain.PrepareSolve()
    return domain


def make_sim(domain: phipict.Domain, dt: float, pressure_amg: bool | None = False) -> phipict.MHDSimulation:
    return phipict.MHDSimulation(
        domain=domain,
        dt=dt,
        substeps=1,
        non_orthogonal=False,
        stuart_number=torch.tensor(200.0**2 / 5000.0),
        e_b=torch.tensor([0.0, 1.0, 0.0], dtype=domain.getDtype()),
        advection_tol=phipict.SolverTolerance(atol=1e-6),
        pressure_tol=phipict.SolverTolerance(rtol=1e-2, atol=1e-14),
        pressure_tol_intermediate=phipict.SolverTolerance(rtol=1e-1, atol=1e-14),
        potential_tol=phipict.SolverTolerance(rtol=1e-4, atol=1e-14),
        pressure_warm_start=True,
        potential_reuse_result=True,
        potential_use_preconditioner=False,  # 2D: no potential AMG in the env either
        pressure_use_amg=pressure_amg,
    )


def init_envs(domain: phipict.Domain, B: int, seed: int) -> None:
    """Parabolic profile plus a different random perturbation and forcing per env."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    block = domain.getBlocks()[0]
    _, _, ny, nx = block.velocity.shape
    y = torch.linspace(-1, 1, ny, dtype=block.velocity.dtype)
    for b in range(B):
        vel = torch.zeros(2, ny, nx, dtype=block.velocity.dtype)
        vel[0] = (1 - y**2).view(ny, 1)
        vel += 0.05 * torch.randn(vel.shape, generator=g, dtype=vel.dtype)
        block.velocity[b].copy_(vel.to(DEV))
    forcing = torch.tensor([[8.0 * (1 + 0.1 * b), 0.0] for b in range(B)], dtype=block.velocity.dtype)
    block.setVelocitySource(forcing.to(DEV))
    domain.UpdateDomainData()


def time_steps(sim: phipict.MHDSimulation, steps: int, warmup: int) -> float:
    for _ in range(warmup):
        sim.single_step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        sim.single_step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--nx", type=int, default=480)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dt", type=float, default=5e-3)
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--pressure-amg", choices=["off", "on", "auto"], default="off")
    parser.add_argument("--out", default="output/solver_perf/bench_batching.jsonl")
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)

    rows = []
    t_single = None
    for B in [int(b) for b in args.batch_sizes.split(",")]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        domain = make_duct(args.nx, args.ny, dtype)
        if B > 1:
            domain.setBatchSize(B)
            domain.PrepareSolve()
        init_envs(domain, B, seed=B)
        sim = make_sim(domain, args.dt, {"off": False, "on": True, "auto": None}[args.pressure_amg])
        t = time_steps(sim, args.steps, args.warmup)
        if B == 1:
            t_single = t
        row = dict(B=B, pressure_amg=args.pressure_amg, nx=args.nx, ny=args.ny, dtype=args.dtype, s_per_piso_step=t, env_steps_per_s=B / t,
                   speedup_vs_sequential=(t_single * B / t) if t_single else None,
                   peak_mem_gib=torch.cuda.max_memory_allocated() / 2**30,
                   finite=bool(torch.isfinite(domain.getBlocks()[0].velocity).all()))
        rows.append(row)
        print(json.dumps(row), flush=True)
        del sim, domain
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
