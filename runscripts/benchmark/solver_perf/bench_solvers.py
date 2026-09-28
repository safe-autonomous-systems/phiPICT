"""Replay captured linear systems through the solver backends and compare them.

For every system written by ``capture_systems.py`` this runs ``_C.SolveLinear`` with
each requested backend configuration and reports wall time, iterations, time per
iteration, the true residual ``||b - A x||_2 / sqrt(n)`` of the returned solution
and its deviation from the legacy solution.

Usage:
    python runscripts/benchmark/solver_perf/bench_solvers.py \
        output/solver_perf/systems/HartmannSmall3D-downward-easy-v0 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from phipict import _C

# name -> kwargs of _C.SetKrylovSettings (None: legacy solvers)
CONFIGS: dict[str, dict | None] = {
    "legacy": None,
    "fused": dict(enabled=True, useGraphs=True, cgPreconditioner="none", bicgPreconditioner="none"),
    "fused_nograph": dict(enabled=True, useGraphs=False, cgPreconditioner="none", bicgPreconditioner="none"),
    "fused_jacobi": dict(enabled=True, useGraphs=True, cgPreconditioner="jacobi", bicgPreconditioner="jacobi"),
}


def apply_config(cfg: dict | None) -> None:
    if cfg is None:
        _C.SetKrylovSettings(enabled=False)
    else:
        _C.SetKrylovSettings(**cfg)


def true_residual(A: torch.Tensor, b: torch.Tensor, x: torch.Tensor, n: int) -> list[float]:
    nb = b.numel() // n
    res = b.view(nb, n).T - A @ x.view(nb, n).T
    return (torch.linalg.vector_norm(res, dim=0) / math.sqrt(n)).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--configs", default="legacy,fused,fused_nograph,fused_jacobi")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tags", default=None, help="comma separated subset of solve tags")
    parser.add_argument("--max-files", type=int, default=4, help="systems per tag")
    parser.add_argument("--out", default=None, help="write results as JSON lines")
    args = parser.parse_args()

    configs = args.configs.split(",")
    rows = []
    for d in args.dirs:
        files = sorted(Path(d).glob("*_[0-9][0-9][0-9].pt"))
        for f in files:
            tag, idx = f.stem.rsplit("_", 1)
            if args.tags and tag not in args.tags.split(","):
                continue
            if int(idx) >= args.max_files:
                continue
            data = torch.load(f, weights_only=False)
            meta = data["meta"]
            if meta["kind"] == "amg_pcg":
                continue  # replayed by bench_amg.py
            a = meta["args"]
            dev = torch.device("cuda")
            row, index, value = data["row"].to(dev), data["index"].to(dev), data["value"].to(dev)
            rhs, x0 = data["rhs"].to(dev), data["x0"].to(dev)
            n = row.numel() - 1
            mat = _C.CSRmatrix(value, index, row)
            A = torch.sparse_csr_tensor(row, index, value, (n, n))
            if a["transposeA"]:
                A = A.to_sparse_coo().t().to_sparse_csr()
            maxit = torch.IntTensor([a["maxit"]])
            tol = torch.tensor([a["tol"]], dtype=value.dtype)
            ref_x = None
            for cname in configs:
                apply_config(CONFIGS[cname])
                times = []
                for rep in range(args.repeats + 1):
                    x = x0.clone()
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    infos = _C.SolveLinear(mat, rhs, x, maxit, tol, _C.ConvergenceCriterion(a["conv"]), a["useBiCG"],
                                           a["rankDef"], a["resetSteps"], a["transposeA"], False, a["returnBest"],
                                           a["BiCGprecon"])
                    torch.cuda.synchronize()
                    if rep > 0:  # first run is warm-up (workspace allocation, graph capture)
                        times.append(time.perf_counter() - t0)
                t = sorted(times)[len(times) // 2]
                iters = [i.usedIterations for i in infos]
                res = true_residual(A, rhs, x, n)
                if cname == "legacy" or ref_x is None:
                    ref_x = x.clone()
                dev_rel = float(torch.linalg.vector_norm(x - ref_x) / max(torch.linalg.vector_norm(ref_x), 1e-300))
                row_out = dict(system=f"{Path(d).name}/{f.stem}", kind=meta["kind"], n=n, n_rhs=meta["n_rhs"],
                               config=cname, time_ms=t * 1e3, iters=iters,
                               us_per_iter=t * 1e6 / max(1, max(iters) + 1),
                               converged=[i.converged for i in infos], true_res=res, tol=a["tol"],
                               rel_diff_to_first=dev_rel)
                rows.append(row_out)
                print(f"{row_out['system']:<55s} {cname:<14s} n={n:<8d} t={t*1e3:9.3f} ms iters={iters} "
                      f"us/it={row_out['us_per_iter']:8.1f} conv={row_out['converged']} "
                      f"res={['%.2e' % r for r in res]} tol={a['tol']:.2e} diff={dev_rel:.1e}")
            del mat, A
    apply_config(CONFIGS["fused"])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
