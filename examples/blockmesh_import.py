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

"""Run a case meshed with an OpenFOAM blockMeshDict.

The dictionary in ``examples/meshes/flow_past_cylinder`` is a 2D channel of 3 x 3
blocks (one cell between ``empty`` front and back patches) from the sdfibm
immersed-boundary examples. Its patches are all of type ``wall``; here the inlet
and outlet get flow conditions and the sides become free-slip.
"""

import argparse
from pathlib import Path

import torch

import phipict
import phipict.meshing as pm
from phipict import Hook, Hooks
from phipict.io import export

MESH = Path(__file__).parent / "meshes" / "flow_past_cylinder" / "blockMeshDict"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict", type=Path, default=MESH, help="blockMeshDict to read")
    parser.add_argument("--re", type=float, default=200.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--out", default="out/blockmesh", help="VTK output path")
    args = parser.parse_args()

    mesh = pm.read_blockmeshdict(args.dict)
    print(mesh)
    for patch in mesh.patches:
        print(f"  patch {patch.name!r} ({patch.kind}): {len(mesh.faces(patch))} faces")

    mesh.set_bc("inlet", pm.Inflow((1.0, 0.0)))
    mesh.set_bc("outlet", pm.Outflow())
    mesh.set_bc("side1", pm.FreeSlip())
    mesh.set_bc("side2", pm.FreeSlip())
    report = mesh.check()
    print(report)
    report.raise_if_invalid()

    domain = mesh.get_domain(
        viscosity=1.0 / args.re,
        dtype=torch.float64,  # float32 pressure solves stall on this large domain
        device=torch.device("cuda"),
        passive_scalar_channels=0,
    )
    series = export.VTKSeries(args.out, mesh=mesh)
    hooks = (
        Hooks()
        .append(Hook.PRE, mesh.outflow_hook(domain, velocity=1.0))
        .append(Hook.POST, series.hook(every=max(1, args.steps // 10)))
    )
    sim = phipict.Simulation(
        domain=domain, dt=0.5, substeps="ADAPTIVE", hooks=hooks, pressure_use_amg=True
    )
    sim.make_divergence_free()
    for _ in range(args.steps):
        sim.single_step()
    print(f"wrote {series.path}")


if __name__ == "__main__":
    main()
