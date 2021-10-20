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
// processor coordinates on a Grid, such as Mesh or Torus
std::vector<uint64_t> ConsecutiveProcessorIdToGridCoordinates(
    int64_t processor_id,
    const std::vector<uint64_t>& dimension_sizes,
    uint64_t concentration);

uint64_t GridCoordinatesToConsecutiveProcessorId(
    const std::vector<uint64_t>& coordinates,
    const std::vector<uint64_t>& dimension_sizes,
    uint64_t concentration);

CommunicationGroup CommunicationGroupLocalProjection(
    int64_t processor_id,
    const CommunicationGroup& comm_group,
    const std::vector<uint64_t>& dimension_sizes,
    uint64_t concentration);

CommunicationGroup CommunicationGroupProjectionOnGrid(
    int64_t processor_id,
    const CommunicationGroup& comm_group,
    size_t dimension,
    bool include_concentrators,
    const std::vector<uint64_t>& dimension_sizes,
    uint64_t concentration);

// 2D swizzling algorithm that produces Hamiltonian cycle through  all the
// vertices of 2D Grid, such as Mesh or Torus. It is used to map these
// topologies onto logical ring topology. This is not the optimal algorithm as
// it does not use all the available link on 2D grid, but it takes into
// consideration phisical proximity to maintain communication only between
// physical neighbors on the grid.
// The algorithm demands 1st dimension (X) of the grid to be even. For M x N
// grid, it starts in the grid point with coordinate (0, 0), goes down to the
// grid point with coordinate (0, N), switches to the next column, changes
// direction, and goes from the point with coordinate (1, N) to the point with
// with coordinate (1, 1), then switches again and goes down, and so on.
// It finishes the cycle by going from the point with coordinate (M, 0) to the
// point with coordinates (1, 0) and makes the cycle.
// You can see the result of this algorithm for 6 x 3 grid below:
// *<--*<--*<--*---*<--*
// |                   ^
// v                   |
// *   *-->*   *-->*   *
// |   ^   |   ^   |   ^
// v   |   v   |   v   |
// *-->*   *-->*   *-->*
CommunicationGroup Swizzling2dGridToRing(
    const std::vector<uint64_t>& dimension_sizes,
    uint64_t concentration);

}  // namespace paragraph

#endif  // PARAGRAPH_TRANSLATION_UTILS_H_
