/* Copyright 2020 Google LLC
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
#ifndef PARAGRAPH_SHIM_TEST_MACROS_H_
#define PARAGRAPH_SHIM_TEST_MACROS_H_

#include <utility>

#include "gtest/gtest.h"
#include "paragraph/shim/macros.h"

// Macros for testing the results of functions that returns absl::Status.
#define EXPECT_OK(statement) \
  EXPECT_EQ(absl::OkStatus(), (statement))
#define ASSERT_OK(statement) \
  ASSERT_EQ(absl::OkStatus(), (statement))

#define ASSERT_OK_AND_ASSIGN(lhs, rexpr)        \
  ASSERT_OK_AND_ASSIGN_IMPL(CONCAT_MACRO(       \
      _status_or, __COUNTER__), lhs, rexpr)

#define ASSERT_OK_AND_ASSIGN_IMPL(statusor, lhs, rexpr)  \
  auto statusor = (rexpr);                               \
  ASSERT_TRUE(statusor.status().ok()) <<                 \
      statusor.status();                                 \
  lhs = std::move(statusor.value())

#endif  // PARAGRAPH_SHIM_TEST_MACROS_H_
