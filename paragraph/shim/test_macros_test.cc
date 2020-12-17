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
#include "paragraph/shim/test_macros.h"

#include <memory>

#include "absl/memory/memory.h"
#include "gtest/gtest.h"
#include "paragraph/shim/statusor.h"

namespace shim {
namespace {

StatusOr<std::unique_ptr<int>> ReturnUniquePtr() {
  // Uses implicit constructor from T&&
  return absl::make_unique<int>(42);
}

TEST(ShimMacros, ExpectOk) {
  // Checks that EXPECT_OK(Status::Ok) doesn't break
  StatusOr<std::unique_ptr<int>> thing(ReturnUniquePtr());
  EXPECT_OK(thing.status());
}

TEST(ShimMacros, AssertOk) {
  // Checks that ASSERT_OK(Status::Ok) doesn't break
  StatusOr<std::unique_ptr<int>> thing(ReturnUniquePtr());
  ASSERT_OK(thing.status());
}

}  // namespace
}  // namespace shim
