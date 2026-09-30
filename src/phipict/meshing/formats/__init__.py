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

"""Mesh file formats: OpenFOAM blockMeshDict and VTK structured grids."""

from .blockmeshdict import load_blockmeshdict, read_blockmeshdict, write_blockmeshdict
from .foam import FoamParseError, parse_foam
from .vtk import read_vtk

__all__ = [
    "FoamParseError",
    "load_blockmeshdict",
    "parse_foam",
    "read_blockmeshdict",
    "read_vtk",
    "write_blockmeshdict",
]
