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
// Declarations of piso/result_copy.cu.

#ifndef _PHIPICT_PISO_RESULT_COPY_H
#define _PHIPICT_PISO_RESULT_COPY_H

#include "domain/domain_structs.h"
#include "solvers/linear_solvers.h"

/** Copy from domain.scalarResult to block.scalarData. */
void CopyScalarResultToBlocks(std::shared_ptr<Domain> domain);

/** Copy from block.scalarData to domain.scalarResult. */
void CopyScalarResultFromBlocks(std::shared_ptr<Domain> domain);

/** Copy from domain.pressureResult to block.pressure. */
void CopyPressureResultToBlocks(std::shared_ptr<Domain> domain);

/** Copy from domain.epotResult to block.epot. */
void CopyEpotResultToBlocks(std::shared_ptr<Domain> domain);

/** Copy from block.epot to domain.epotResult. */
void CopyEpotResultFromBlocks(std::shared_ptr<Domain> domain);

/** Copy from block.pressure to domain.pressureResult. */
void CopyPressureResultFromBlocks(std::shared_ptr<Domain> domain);

/** Copy from domain.velocityResult to block.velocity. */
void CopyVelocityResultToBlocks(std::shared_ptr<Domain> domain);

/** Copy from block.velocity to domain.velocityResult. */
void CopyVelocityResultFromBlocks(std::shared_ptr<Domain> domain);

#ifdef WITH_GRAD

/** Copy from block.passiveScalar_grad to domain.scalarResult_grad. */
void CopyScalarResultGradFromBlocks(std::shared_ptr<Domain> domain);

/** Copy from domain.scalarResult_grad to block.passiveScalar_grad. */
void CopyScalarResultGradToBlocks(std::shared_ptr<Domain> domain);
void CopyPressureResultGradFromBlocks(std::shared_ptr<Domain> domain);
void CopyVelocityResultGradFromBlocks(std::shared_ptr<Domain> domain);
void CopyVelocityResultGradToBlocks(std::shared_ptr<Domain> domain);
void CopyEpotResultGradFromBlocks(std::shared_ptr<Domain> domain);

#endif //WITH_GRAD

#endif //_PHIPICT_PISO_RESULT_COPY_H
