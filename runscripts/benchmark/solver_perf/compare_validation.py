"""Compare validation runs of the solver stacks with each other and with the reference runs.

Reads (never writes) ``output_bkp/validation/<type>/<Ha>/<res>/domain.npz`` as the reference
and ``output/solver_perf/validation/<stack>/<type>/<Ha>/domain.npz`` for every stack.
Reports the relative L2 difference of the final velocity fields, and for the Hartmann case
also the error of the x-averaged profile against the analytical solution.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REF = Path("output_bkp/validation")
NEW = Path("output/solver_perf/validation")


def velocity(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return np.load(path)["1"].astype(np.float64)


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def _hartmann_error(u: np.ndarray, ha: float) -> float:
    from phipict.grid import shapes

    ny = u.shape[-2]
    # vertex positions in [0, 1] (same grading as the validation case), mapped to [-1, 1]
    edges = -1.0 + 2.0 * np.asarray(shapes.make_weights("simple", res=ny, grading=10, refinement="BOTH"), dtype=np.float64)
    y = 0.5 * (edges[1:] + edges[:-1])
    dy = np.diff(edges)
    prof = u[0, 0].mean(axis=-1)  # [y]
    exact = 1.0 - np.cosh(ha * y) / np.cosh(ha)
    exact *= (prof * dy).sum() / (exact * dy).sum()
    return float(np.abs(prof - exact).max() / np.abs(exact).max())


def main() -> None:
    stacks = sys.argv[1:] or ["legacy", "new", "new_amg"]
    for case_dir in sorted(REF.glob("*/*/*")):
        typ, ha, res = case_dir.parts[-3:]
        try:
            float(ha)
        except ValueError:
            continue
        ref = velocity(case_dir / "domain.npz")
        rows = []
        for stack in stacks:
            cand = sorted((NEW / stack / typ).glob(f"{float(ha):g}*")) + sorted((NEW / stack / typ).glob(f"{int(float(ha))}"))
            if not cand:
                continue
            u = velocity(cand[0] / "domain.npz")
            if u is None:
                continue
            row = f"  {stack:<8s} rel.L2 vs reference: {rel_l2(u, ref):.3e}"
            if typ == "hartmann":
                row += f"  profile error vs analytic: {_hartmann_error(u, float(ha)):.3e} (reference run: {_hartmann_error(ref, float(ha)):.3e})"
            rows.append((stack, u, row))
        if not rows:
            continue
        print(f"{typ} Ha={ha} {res}")
        for _, _, row in rows:
            print(row)
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                print(f"  {rows[i][0]} vs {rows[j][0]}: rel.L2 {rel_l2(rows[i][1], rows[j][1]):.3e}")


if __name__ == "__main__":
    main()
