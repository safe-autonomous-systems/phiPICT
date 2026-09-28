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

"""PhiPICT: a differentiable multi-block PISO fluid solver based on PICT."""

# torch must be imported before the compiled extension so that its shared
# libraries are loaded and resolved instead of whatever LD_LIBRARY_PATH points to.
import torch  # noqa: F401, I001

from phipict import _C, bc
from phipict._C import (
    AdvectionScheme,
    Block,
    Boundary,
    BoundaryConditionType,
    BoundaryType,
    ConnectedBoundary,
    ConvergenceCriterion,
    CSRmatrix,
    Domain,
    FixedBoundary,
    LinearSolverResultInfo,
    PeriodicBoundary,
    PotentialBC,
)
from phipict.bc import Face, get_bc, set_bc
from phipict.core.hooks import Hook, Hooks
from phipict.logging import get_logger, set_verbosity
from phipict.simulation.mhd import MHDSimulation
from phipict.simulation.simulation import Simulation
from phipict.solvers.tolerance import SolverTolerance

__all__ = [
    "AdvectionScheme",
    "Block",
    "Boundary",
    "BoundaryConditionType",
    "BoundaryType",
    "CSRmatrix",
    "ConnectedBoundary",
    "ConvergenceCriterion",
    "Domain",
    "Face",
    "FixedBoundary",
    "Hook",
    "Hooks",
    "LinearSolverResultInfo",
    "MHDSimulation",
    "PeriodicBoundary",
    "PotentialBC",
    "Simulation",
    "SolverTolerance",
    "_C",
    "bc",
    "get_bc",
    "get_logger",
    "set_bc",
    "set_verbosity",
]
