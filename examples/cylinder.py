# Copyright 2026 Jannis Becktepe
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flow past a cylinder in a channel, meshed with ``phipict.meshing``.

An O-grid of four blocks around the cylinder sits in the middle of a 3 x 3
arrangement of channel blocks. Only the corners, a few cell counts and the
gradings are given; the other cell counts, the shared arcs and all twelve block
connections follow from the shared corners. The solution is written as a VTK
time series for ParaView.
"""

import argparse
import math

import torch

import phipict
import phipict.meshing as pm
from phipict import Hook, Hooks
from phipict.io import export


def make_mesh(n: int = 16, radius: float = 0.5, half: float = 1.5) -> pm.Mesh:
    """Cylinder of ``radius`` inside a square O-grid of half-width ``half``."""
    x0, x1, h = -4.0, 16.0, 2.0  # channel: x in [x0, x1], y in [-h, h]
    arc = pm.Arc(center=(0.0, 0.0))

    def on_circle(deg: float) -> tuple[float, float]:
        return radius * math.cos(math.radians(deg)), radius * math.sin(
            math.radians(deg)
        )

    cylinder = pm.Patch("cylinder", pm.Wall())
    walls = pm.Patch("walls", pm.FreeSlip())
    inlet = pm.Patch("inlet", pm.Inflow(lambda p: parabola(p, h)))
    outlet = pm.Patch("outlet", pm.Outflow())

    radial = pm.Simple(4.0)  # cells grow 4x from the cylinder to the square
    bm = pm.BlockMesh()
    # O-grid: every block runs outwards along one axis, right-handed. The angular
    # cells are given per block (their edges only meet at corners), the radial
    # ones propagate along the shared diagonal edges.
    bm.add(
        pm.Quad(  # right: x outwards, y upwards
            [on_circle(-45), (half, -half), on_circle(45), (half, half)],
            cells=(n, n),
            grading=(radial, None),
            edges=pm.QuadEdges(x_minus=arc),
            patches=pm.FacePatches(x_minus=cylinder),
            name="ring_right",
        )
    )
    bm.add(
        pm.Quad(  # top: x to the right, y outwards
            [on_circle(135), on_circle(45), (-half, half), (half, half)],
            cells=(n, None),  # radial cells: shared with ring_right
            grading=(None, radial),
            edges=pm.QuadEdges(y_minus=arc),
            patches=pm.FacePatches(y_minus=cylinder),
            name="ring_top",
        )
    )
    bm.add(
        pm.Quad(  # left: x inwards, y upwards
            [(-half, -half), on_circle(-135), (-half, half), on_circle(135)],
            cells=(None, n),
            grading=(radial.reversed(), None),
            edges=pm.QuadEdges(x_plus=arc),
            patches=pm.FacePatches(x_plus=cylinder),
            name="ring_left",
        )
    )
    bm.add(
        pm.Quad(  # bottom: x to the right, y inwards
            [(-half, -half), (half, -half), on_circle(-135), on_circle(-45)],
            cells=(n, None),
            grading=(None, radial.reversed()),
            edges=pm.QuadEdges(y_plus=arc),
            patches=pm.FacePatches(y_plus=cylinder),
            name="ring_bottom",
        )
    )
    # channel blocks around the O-grid; unknown cell counts come from neighbours
    xs, ys = [x0, -half, half, x1], [-h, -half, half, h]
    cells_x = [12, None, 64]
    cells_y = [8, None, 8]
    grading_x = [pm.Simple(0.5), None, pm.Simple(4.0)]
    for i in range(3):
        for j in range(3):
            if i == j == 1:
                continue  # the O-grid
            faces = {}
            if i == 0:
                faces["x_minus"] = inlet
            if i == 2:
                faces["x_plus"] = outlet
            if j == 0:
                faces["y_minus"] = walls
            if j == 2:
                faces["y_plus"] = walls
            bm.add(
                pm.Quad(
                    [
                        (xs[i], ys[j]),
                        (xs[i + 1], ys[j]),
                        (xs[i], ys[j + 1]),
                        (xs[i + 1], ys[j + 1]),
                    ],
                    cells=(cells_x[i], cells_y[j]),
                    grading=(grading_x[i], pm.Symmetric(2.0) if j != 1 else None),
                    patches=pm.FacePatches(**faces),
                    name=f"channel_{i}{j}",
                )
            )
    return bm.build()


def parabola(points: torch.Tensor, h: float) -> torch.Tensor:
    """Parabolic inflow with mean velocity 1 at the face cell centres."""
    y = points[:, 1]
    u = 1.5 * (1.0 - (y / h) ** 2)
    return torch.stack([u] + [torch.zeros_like(u)] * (points.shape[1] - 1), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--re", type=float, default=100.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--three-d", action="store_true", help="extrude, periodic in z")
    parser.add_argument("--out", default="out/cylinder", help="VTK output path")
    args = parser.parse_args()

    mesh = make_mesh()
    if args.three_d:
        mesh = mesh.extrude((-1.0, 1.0), cells=8, periodic=True)
    print(mesh)
    print(mesh.check())

    dtype, device = torch.float32, torch.device("cuda")
    domain = mesh.get_domain(
        viscosity=1.0 / args.re, dtype=dtype, device=device, passive_scalar_channels=0
    )
    series = export.VTKSeries(args.out, mesh=mesh)
    hooks = (
        Hooks()
        .append(Hook.PRE, mesh.outflow_hook(domain, velocity=1.0))
        .append(Hook.POST, series.hook(every=max(1, args.steps // 20)))
    )
    # AMG-preconditioned pressure solves: plain CG struggles on the skewed O-grid
    sim = phipict.Simulation(
        domain=domain, dt=0.05, substeps="ADAPTIVE", hooks=hooks, pressure_use_amg=True
    )
    sim.make_divergence_free()
    for _ in range(args.steps):
        sim.single_step()
    u = torch.cat([b.velocity[0, 0].flatten() for b in domain.getBlocks()])
    print(f"max u = {float(u.max()):.3f}; wrote {series.path}")


if __name__ == "__main__":
    main()
