"""Capture the linear systems a real environment solves, for offline solver benchmarks.

Runs a few PISO steps of a fluidgym environment and records every linear solve
(matrix, right-hand side, initial guess and solver arguments) that goes through
``_C.SolveLinear`` or the AMG path. The recorded systems are what
``bench_solvers.py`` replays, so solver backends can be compared on exactly the
matrices the simulation produces, without re-running the simulation.

Usage:
    python runscripts/benchmark/solver_perf/capture_systems.py \
        --env HartmannSmall3D-downward-easy-v0 --steps 3 --out output/solver_perf/systems
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

import fluidgym
from phipict import _C
from phipict.core import piso_diff, piso_simulation

_CURRENT_TAG = {"tag": "unknown"}


class _CProxy:
    """Stands in for ``phipict._C`` inside ``piso_diff`` and records the solves."""

    def __init__(self, recorder: "_Recorder") -> None:
        self._recorder = recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(_C, name)

    def SolveLinear(self, A, rhs, x, maxit, tol, conv, useBiCG, rankDef, resetSteps, transposeA,
                    printResidual, returnBest, BiCGwithPreconditioner=True):  # noqa: N802,N803
        BiCGprecon = BiCGwithPreconditioner
        """Record the system, then run the real solve and record its result."""
        entry = self._recorder.begin(
            kind="bicgstab" if useBiCG else "cg",
            A=(A.row, A.index, A.value),
            rhs=rhs,
            x0=x,
            args=dict(
                maxit=int(maxit[0]), tol=float(tol[0]), conv=int(conv), useBiCG=bool(useBiCG),
                rankDef=bool(rankDef), resetSteps=int(resetSteps), transposeA=bool(transposeA),
                returnBest=bool(returnBest), BiCGprecon=bool(BiCGprecon),
            ),
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        infos = _C.SolveLinear(A, rhs, x, maxit, tol, conv, useBiCG, rankDef, resetSteps, transposeA,
                               printResidual, returnBest, BiCGprecon)
        torch.cuda.synchronize()
        self._recorder.end(entry, infos, time.perf_counter() - t0)
        return infos


class _Recorder:
    """Keeps the first ``per_tag`` systems of every solve tag and writes them to disk."""

    def __init__(self, out: Path, per_tag: int) -> None:
        self.out = out
        self.per_tag = per_tag
        self.counts: dict[str, int] = {}
        self.log: list[dict[str, Any]] = []

    def begin(self, kind: str, A, rhs, x0, args) -> dict[str, Any]:
        tag = _CURRENT_TAG["tag"]
        idx = self.counts.get(tag, 0)
        self.counts[tag] = idx + 1
        entry: dict[str, Any] = {"tag": tag, "idx": idx, "kind": kind, "args": args,
                                 "n": int(A[0].numel() - 1), "nnz": int(A[2].numel()),
                                 "n_rhs": int(rhs.numel() // (A[0].numel() - 1)),
                                 "dtype": str(rhs.dtype)}
        if idx < self.per_tag:
            entry["save"] = {
                "row": A[0].detach().clone(), "index": A[1].detach().clone(),
                "value": A[2].detach().clone(), "rhs": rhs.detach().clone(),
                "x0": x0.detach().clone(),
            }
        return entry

    def end(self, entry: dict[str, Any], infos, seconds: float) -> None:
        entry["iters"] = [int(i.usedIterations) for i in infos]
        entry["converged"] = [bool(i.converged) for i in infos]
        entry["residual"] = [float(i.finalResidual) for i in infos]
        entry["seconds"] = seconds
        save = entry.pop("save", None)
        if save is not None:
            name = f"{entry['tag']}_{entry['idx']:03d}.pt"
            torch.save({**{k: (v.cpu() if torch.is_tensor(v) else v) for k, v in save.items()},
                        "meta": {k: v for k, v in entry.items()}}, self.out / name)
            entry["file"] = name
        self.log.append(entry)
        print(f"  {entry['tag']:>10s} #{entry['idx']:<3d} {entry['kind']:>8s} n={entry['n']:>9d} "
              f"rhs={entry['n_rhs']} iters={entry['iters']} t={seconds*1e3:9.2f} ms")


def _patch_tags() -> None:
    """Make ``_linear_solve_wrapper`` publish its tag before it calls the kernel."""
    orig = piso_diff._linear_solve_wrapper

    def wrapper(*args, tag: str = "unknown", **kwargs):
        _CURRENT_TAG["tag"] = tag
        return orig(*args, tag=tag, **kwargs)

    piso_diff._linear_solve_wrapper = wrapper


def _patch_amg(recorder: _Recorder) -> None:
    """Record the AMG-PCG solves (epot), whose matrix is the hierarchy's finest level."""
    orig = piso_simulation.amg_pcg_solve

    def amg_wrapper(rhs, x, hierarchy, *, tol, max_iter, return_best_result=False):
        A = hierarchy.levels[0].A
        idx = recorder.counts.get("epot_amg", 0)
        recorder.counts["epot_amg"] = idx + 1
        entry: dict[str, Any] = {"tag": "epot_amg", "idx": idx, "kind": "amg_pcg",
                                 "args": dict(maxit=max_iter, tol=float(tol), returnBest=return_best_result,
                                              project_constant=hierarchy.project_constant),
                                 "n": int(A.shape[0]), "nnz": int(A._nnz()), "n_rhs": 1,
                                 "dtype": str(rhs.dtype)}
        if idx < recorder.per_tag:
            entry["save"] = {"row": A.crow_indices().to(torch.int32).clone(),
                             "index": A.col_indices().to(torch.int32).clone(),
                             "value": A.values().clone(), "rhs": rhs.detach().clone(),
                             "x0": x.detach().clone()}
            if idx == 0:
                torch.save(hierarchy, recorder.out / "epot_amg_hierarchy.pt")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        infos = orig(rhs, x, hierarchy, tol=tol, max_iter=max_iter, return_best_result=return_best_result)
        torch.cuda.synchronize()
        recorder.end(entry, infos, time.perf_counter() - t0)
        return infos

    piso_simulation.amg_pcg_solve = amg_wrapper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="HartmannSmall3D-downward-easy-v0")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--per-tag", type=int, default=4)
    parser.add_argument("--step-length", type=float, default=None,
                        help="Physical time per env step; default is the env's own.")
    parser.add_argument("--out", default="output/solver_perf/systems")
    parser.add_argument("--local-data", default="/cephfs/users/becktepe/git_projects/fluidgym_mhd/local_data")
    args = parser.parse_args()

    out = Path(args.out) / args.env
    out.mkdir(parents=True, exist_ok=True)
    fluidgym.config.update("local_data_path", args.local_data)

    recorder = _Recorder(out, args.per_tag)
    _patch_tags()
    _patch_amg(recorder)
    piso_diff._C = _CProxy(recorder)

    env = fluidgym.make(args.env, load_initial_domain=True, load_domain_statistics=False,
                        randomize_initial_state=False, differentiable=False,
                        **({} if args.step_length is None else {"step_length": args.step_length}))
    env.seed(42)
    env.reset()
    env._sim.substeps = 1
    action = env._zero_action
    for step in range(args.steps):
        print(f"step {step}")
        with torch.no_grad():
            env.step(action)

    torch.save(recorder.log, out / "log.pt")
    print(f"peak memory {torch.cuda.max_memory_allocated()/2**30:.2f} GiB, wrote {out}")


if __name__ == "__main__":
    main()
