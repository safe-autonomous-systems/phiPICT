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
// All simulation functions exposed to Python.

#ifndef _PHIPICT_API_H
#define _PHIPICT_API_H

#include "domain/domain_structs.h"
#include "solvers/linear_solvers.h"
#include "piso/non_ortho_flags.h"

#include "piso/advection.h"
#include "piso/pressure.h"
#include "piso/velocity_correction.h"
#include "piso/analysis.h"
#include "mhd/potential.h"
#include "piso/result_copy.h"
#include "piso/sgs.h"
#include "solvers/linear_solve.h"
#include "grid/transform_vectors.h"

#endif //_PHIPICT_API_H
