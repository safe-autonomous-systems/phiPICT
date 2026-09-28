// Original work Copyright 2025 Aleksandra Franz, Nils Thuerey
// Modified work Copyright 2026 Jannis Becktepe
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//
// Device-side helpers shared by the simulation kernels (PISO and MHD).

#pragma once

#include "api.h"

#include "solvers/linear_solvers.h"
#include "solvers/krylov/krylov.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <limits>

#define LOGGING
#ifdef LOGGING
//#define PROFILING
#endif
#include "common/logging.h"

#include "piso/device/launch.cuh"
#include "piso/device/topology.cuh"
#include "piso/device/velocity.cuh"
#include "piso/device/discretization.cuh"
#include "piso/device/nonortho_laplace.cuh"
#include "piso/device/block_data.cuh"
#include "piso/device/nonortho_rhs.cuh"
#include "piso/device/csr_rows.cuh"
#include "piso/device/shared_stencils.cuh"
