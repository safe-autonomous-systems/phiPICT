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

"""Structured multi-block meshes: building, importing and converting to a domain.

Three ways to make a :class:`Mesh`, from low to high level::

    import phipict.meshing as pm

    # 1. blocks from vertex tensors, e.g. existing grids
    mesh = pm.Mesh([pm.MeshBlock.from_coords(coords)])

    # 2. blocks from corners, resolution and grading per axis
    walls = pm.Patch("walls", pm.Wall())
    block = pm.box((0, -1), (10, 1), cells=(128, 64), grading=(None, pm.Symmetric(20)),
                   patches=pm.FacePatches(y_minus=walls, y_plus=walls))
    mesh = pm.Mesh([block])
    mesh.make_periodic("x")

    #    or blockMesh-like, with shared corners and inferred resolutions
    bm = pm.BlockMesh()
    bm.add(pm.Quad(...))
    mesh = bm.build()

    # 3. from a file
    mesh = pm.read_blockmeshdict("system/blockMeshDict")

Blocks are connected wherever their faces coincide, so face indices and axis
codes never have to be written by hand. Every other face belongs to a
:class:`Patch` with a boundary condition, and :meth:`Mesh.get_domain` creates the
solver domain.
"""

from .block import MeshBlock
from .blockmesh import BlockMesh, Hex, Quad
from .boundary import (
    BoundarySpec,
    FacePatches,
    FreeSlip,
    Inflow,
    Outflow,
    Patch,
    Wall,
)
from .connect import Connection, NonConformingInterfaceError, Orientation
from .curves import (
    Arc,
    BSpline,
    EdgeShape,
    Line,
    Parametric,
    Points,
    Polyline,
    Sampled,
    Spline,
)
from .formats import read_blockmeshdict, read_vtk, write_blockmeshdict
from .grading import (
    ChebyshevBlend,
    Cluster,
    Cosine,
    Explicit,
    FirstCell,
    Geometric,
    Grading,
    MultiGrading,
    Segment,
    Simple,
    Symmetric,
    Tanh,
    Uniform,
    as_grading,
    cells_for_size,
)
from .mesh import Mesh
from .quality import MeshReport
from .shapes import (
    CornerEdge,
    Edge,
    Interpolation,
    QuadEdges,
    annulus,
    annulus_radial_weights,
    box,
    hexa,
    quad,
)

__all__ = [
    "Arc",
    "BSpline",
    "BlockMesh",
    "BoundarySpec",
    "ChebyshevBlend",
    "Cluster",
    "Connection",
    "CornerEdge",
    "Cosine",
    "Edge",
    "EdgeShape",
    "Explicit",
    "FacePatches",
    "FirstCell",
    "FreeSlip",
    "Geometric",
    "Grading",
    "Hex",
    "Inflow",
    "Interpolation",
    "Line",
    "Mesh",
    "MeshBlock",
    "MeshReport",
    "MultiGrading",
    "NonConformingInterfaceError",
    "Orientation",
    "Outflow",
    "Parametric",
    "Patch",
    "Points",
    "Polyline",
    "Quad",
    "QuadEdges",
    "Sampled",
    "Segment",
    "Simple",
    "Spline",
    "Symmetric",
    "Tanh",
    "Uniform",
    "Wall",
    "annulus",
    "annulus_radial_weights",
    "as_grading",
    "box",
    "cells_for_size",
    "hexa",
    "quad",
    "read_blockmeshdict",
    "read_vtk",
    "write_blockmeshdict",
]
