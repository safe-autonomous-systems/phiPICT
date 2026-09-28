"""Benchmark AMG-preconditioned CG on captured systems.

Compares the pure-torch AMG-PCG with the native CUDA one (fp64 and fp32
preconditioner) on:
  - the epot systems, with the hierarchy the simulation built (pickled by
    ``capture_systems.py``),
  - the pressure systems, with the epot interpolation frozen and the coarse
    operators refreshed by Galerkin products (``galerkin_refresh``), and
    optionally with a dedicated pyamg setup on the pressure matrix.

Usage:
    python runscripts/benchmark/solver_perf/bench_amg.py \
        output/solver_perf/systems/HartmannSmall3D-downward-easy-v0
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from phipict.solvers import amg


def true_residual(A: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(b - (A @ x.unsqueeze(1)).squeeze(1)) / math.sqrt(b.numel()))


def run(h, A, b, x0, tol, maxit, variant, repeats):
    if variant == "python":
        amg.USE_NATIVE = False
        h.native_dtype = None
    else:
        amg.USE_NATIVE = True
        h.native_dtype = torch.float32 if variant == "native_fp32" else None
    times = []
    for rep in range(repeats + 1):
        x = x0.clone()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        infos = amg.amg_pcg_solve(b, x, h, tol=tol, max_iter=maxit)
        torch.cuda.synchronize()
        if rep > 0:
            times.append(time.perf_counter() - t0)
    amg.USE_NATIVE = True
    t = sorted(times)[len(times) // 2]
    return t, infos[0], true_residual(A, b, x), x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("--variants", default="python,native,native_fp32")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pressure-own-setup", action="store_true", help="also time a dedicated pyamg setup on the pressure matrix")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    d = Path(args.dir)
    dev = torch.device("cuda")
    base = torch.load(d / "epot_amg_hierarchy.pt", weights_only=False)
    print(base.describe())
    rows = []

    def load(f):
        data = torch.load(f, weights_only=False)
        n = data["row"].numel() - 1
        A = torch.sparse_csr_tensor(data["row"].long().to(dev), data["index"].long().to(dev), data["value"].to(dev), (n, n))
        return data, A

    for f in sorted(d.glob("epot_amg_[0-9][0-9][0-9].pt")) + sorted(d.glob("pressure_[0-9][0-9][0-9].pt")):
        data, A = load(f)
        meta = data["meta"]
        b, x0 = data["rhs"].to(dev), data["x0"].to(dev)
        tol = meta["args"]["tol"]
        hierarchies = {}
        if f.name.startswith("epot"):
            hierarchies["own"] = base
        else:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            h = amg.galerkin_refresh(base, A)
            torch.cuda.synchronize()
            h.setup_seconds = time.perf_counter() - t0
            hierarchies["galerkin(epot P/R)"] = h
            if args.pressure_own_setup:
                import scipy.sparse as sp  # noqa: F401
                from phipict import _C

                csr = _C.CSRmatrix(data["value"].to(dev), data["index"].to(dev), data["row"].to(dev))
                hierarchies["own pyamg"] = amg.build_amg_hierarchy(csr, project_constant=base.project_constant)
        for hname, h in hierarchies.items():
            for variant in args.variants.split(","):
                t, info, res, _ = run(h, A, b, x0, tol, 500, variant, args.repeats)
                row = dict(system=f.stem, hierarchy=hname, setup_s=h.setup_seconds, variant=variant, time_ms=t * 1e3,
                           iters=info.usedIterations, ms_per_iter=t * 1e3 / max(1, info.usedIterations + 1),
                           converged=bool(info.converged), true_res=res, tol=tol, legacy_iters=meta["iters"],
                           legacy_ms=meta["seconds"] * 1e3)
                rows.append(row)
                print(f"{f.stem:<16s} {hname:<20s} setup={h.setup_seconds:6.2f}s {variant:<12s} t={t*1e3:9.2f} ms it={info.usedIterations:4d} "
                      f"ms/it={row['ms_per_iter']:7.2f} conv={info.converged} res={res:.2e} tol={tol:.2e} "
                      f"(in-sim before: {meta['kind']} {meta['iters']} it, {meta['seconds']*1e3:.0f} ms)")
        del A, hierarchies
        torch.cuda.empty_cache()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
