"""Time the native AMG-PCG for different lanes-per-row settings of the V-cycle kernels."""
import os
import sys
import time

import torch

from phipict.solvers import amg

D = sys.argv[1]
base = torch.load(D + "/epot_amg_hierarchy.pt", weights_only=False)
cases = []
for name in ["epot_amg_001", "pressure_002"]:
    d = torch.load(f"{D}/{name}.pt", weights_only=False)
    n = d["row"].numel() - 1
    A = torch.sparse_csr_tensor(d["row"].long().cuda(), d["index"].long().cuda(), d["value"].cuda(), (n, n))
    h = base if name.startswith("epot") else amg.galerkin_refresh(base, A)
    cases.append((name, h, d["rhs"].cuda(), d["x0"].cuda(), d["meta"]["args"]["tol"]))
for width in ["auto", "1", "2", "4", "8", "16", "32"]:
    if width == "auto":
        os.environ.pop("PHIPICT_AMG_ROW_WIDTH", None)
    else:
        os.environ["PHIPICT_AMG_ROW_WIDTH"] = width
    for name, h, b, x0, tol in cases:
        for pdt in (None, torch.float32):
            h.native_dtype = pdt
            ts = []
            for rep in range(4):
                x = x0.clone()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                info = amg.amg_pcg_solve(b, x, h, tol=tol, max_iter=500)[0]
                torch.cuda.synchronize()
                if rep:
                    ts.append(time.perf_counter() - t0)
            t = sorted(ts)[1]
            print(f"W={width:>4s} {name:<14s} {'fp32' if pdt else 'fp64'} t={t*1e3:8.2f} ms it={info.usedIterations} ms/it={t*1e3/(info.usedIterations+1):6.2f} conv={info.converged}", flush=True)
