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
#include "paragraph/translation/utils.h"

#include "gtest/gtest.h"
#include "paragraph/graph/graph.h"
#include "paragraph/shim/test_macros.h"

// Tests conversion between consecutive processor id and 2d grid coordinates
TEST(TranslationUtils, ConsecutiveProcessorIdToGridCoordinates) {
  uint64_t concentration = 2;
  std::vector<uint64_t> dimension_sizes = {4, 3};
  std::vector<uint64_t> target_coord_1 = {1, 0, 0};
  EXPECT_EQ(paragraph::ConsecutiveProcessorIdToGridCoordinates(
      1, dimension_sizes, concentration),
            target_coord_1);

  std::vector<uint64_t> target_coord_2 = {0, 1, 1};
  EXPECT_EQ(paragraph::ConsecutiveProcessorIdToGridCoordinates(
      10, dimension_sizes, concentration),
            target_coord_2);

  std::vector<uint64_t> target_coord_3 = {0, 3, 2};
  EXPECT_EQ(paragraph::ConsecutiveProcessorIdToGridCoordinates(
      22, dimension_sizes, concentration),
            target_coord_3);
}

// Tests conversion between 2d grid coordinates and consecutive processor id
TEST(TranslationUtils, GridCoordinatesToConsecutiveProcessorId) {
  uint64_t concentration = 2;
  std::vector<uint64_t> dimension_sizes = {4, 3};
  EXPECT_EQ(paragraph::GridCoordinatesToConsecutiveProcessorId(
      {1, 0, 0}, dimension_sizes, concentration),
            1);
  EXPECT_EQ(paragraph::GridCoordinatesToConsecutiveProcessorId(
      {0, 1, 1}, dimension_sizes, concentration),
            10);
  EXPECT_EQ(paragraph::GridCoordinatesToConsecutiveProcessorId(
      {0, 3, 2}, dimension_sizes, concentration),
            22);
}

// Tests Communication group intersection with local processors
TEST(TranslationUtils, CommunicationGroupLocalProjection) {
  uint64_t concentration = 2;
  std::vector<uint64_t> dimension_sizes = {4, 3};
  paragraph::CommunicationGroup comm_group = {0, 1, 2, 3, 4, 5, 11, 12};
  paragraph::CommunicationGroup test_group = {2, 3};
  EXPECT_EQ(paragraph::CommunicationGroupLocalProjection(
      3, comm_group, dimension_sizes, concentration),
            test_group);
  paragraph::CommunicationGroup test_group_2;
  EXPECT_EQ(paragraph::CommunicationGroupLocalProjection(
      3, {0, 1}, dimension_sizes, concentration),
            test_group_2);
}

// Tests Communication group intersection with processors in particular
// dimensions
TEST(TranslationUtils, CommunicationGroupProjectionOnGrid) {
  uint64_t concentration = 2;
  std::vector<uint64_t> dimension_sizes = {2, 3};
  paragraph::CommunicationGroup comm_group = {0, 1, 2, 3, 4, 5, 10, 11};
  paragraph::CommunicationGroup test_group = {2, 3, 10, 11};
  EXPECT_EQ(paragraph::CommunicationGroupProjectionOnGrid(
      3, comm_group, 1, true, dimension_sizes, concentration),
            test_group);
  paragraph::CommunicationGroup test_group_2 = {1, 3};
  EXPECT_EQ(paragraph::CommunicationGroupProjectionOnGrid(
      1, comm_group, 0, false, dimension_sizes, concentration),
            test_group_2);
}

// Tests 2d swizzling to map 2D grid on a logical ring
TEST(TranslationUtils, Swizzling2dGridToRing) {
  uint64_t concentration = 2;
  std::vector<uint64_t> dimension_sizes = {4, 3};

  paragraph::CommunicationGroup target_group_1 = {0, 1, 8, 9, 16, 17,
                                                  18, 19, 10, 11,
                                                  12, 13, 20, 21,
                                                  22, 23, 14, 15,
                                                  6, 7, 4, 5, 2, 3};
  EXPECT_EQ(paragraph::Swizzling2dGridToRing(dimension_sizes, concentration),
            target_group_1);

  paragraph::CommunicationGroup target_group_2 = {0, 2, 3, 1};
  EXPECT_EQ(paragraph::Swizzling2dGridToRing({2, 2}, 1), target_group_2);
}
