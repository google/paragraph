/* Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef PARAGRAPH_TRANSLATION_UTILS_H_
#define PARAGRAPH_TRANSLATION_UTILS_H_

#include <vector>

#include "paragraph/graph/graph.h"

namespace paragraph {

// Helper functions for conversion bettwen consecutive processor id and
// processor coordinates on a Mesh or Torus
std::vector<uint64_t> ConsecutiveProcessorIdToMeshCoordinates(
    int64_t processor_id,
    const std::vector<uint64_t>& dimension_sizes,
    uint64_t concentration);

uint64_t MeshCoordinatesToConsecutiveProcessorId(
    const std::vector<uint64_t>& coordinates,
    const std::vector<uint64_t>& dimension_sizes,
    uint64_t concentration);

// 2D swizzling that produces Hamiltonian cycle for all the verteces of 2D
// Mesh or Torus. It is used to map these topologies to logical ring topology.
CommunicationGroup Swizzling2dMeshToRing(
    const std::vector<uint64_t>& dimension_sizes,
    uint64_t concentration);

}  // namespace paragraph

#endif  // PARAGRAPH_TRANSLATION_UTILS_H_
