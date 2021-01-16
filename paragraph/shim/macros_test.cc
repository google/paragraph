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
#include "paragraph/shim/macros.h"

#include <memory>

#include "absl/memory/memory.h"
#include "gtest/gtest.h"

namespace shim {
namespace {

class NoDefaultConstructor {
 public:
  explicit NoDefaultConstructor(int foo);
};

static_assert(!std::is_default_constructible<NoDefaultConstructor>(),
              "Should not be default-constructible.");

absl::Status check_return_error(absl::Status x) {
  RETURN_IF_ERROR(x);
  return absl::OkStatus();
}

StatusOr<std::unique_ptr<int>> ReturnUniquePtr() {
  // Uses implicit constructor from T&&
  return absl::make_unique<int>(42);
}

StatusOr<int> check_return_42(absl::Status in_status) {
  StatusOr<int> thing;
  StatusOr<int> thing2;
  if (in_status.ok()) {
    thing = StatusOr<int>(42);
  } else {
    thing = StatusOr<int>(in_status);
  }
  ASSIGN_OR_RETURN(thing2, thing);
  return thing2;
}

StatusOr<int> double_check_return(absl::Status in_status) {
  StatusOr<int> thing;
  StatusOr<int> thing2;
  StatusOr<int> thing3;
  if (in_status.ok()) {
    thing = StatusOr<int>(42);
  } else {
    thing = StatusOr<int>(in_status);
  }
  ASSIGN_OR_RETURN(thing2, thing);
  ASSIGN_OR_RETURN(thing3, thing2);
  return thing3;
}
TEST(ShimMacros, Check) {
  // Checks that CHECK(true) doesn't break
  CHECK(true) << "I can work with streams!";
  CHECK_EQ(1, 1);
  CHECK_NE(1, 2);
  CHECK_LE(1, 1);
  CHECK_LE(1, 2);
  CHECK_LT(1, 2);
  CHECK_GE(2, 2);
  CHECK_GE(2, 1);
  CHECK_GT(2, 1);

  // Checks that CHECK(false) breaks
  EXPECT_DEATH(CHECK(false) << "Uses stream.", "Uses stream.");
  EXPECT_DEATH(CHECK_EQ(1, 2), "");
  EXPECT_DEATH(CHECK_NE(1, 1), "");
  EXPECT_DEATH(CHECK_LE(2, 1), "");
  EXPECT_DEATH(CHECK_LT(1, 1), "");
  EXPECT_DEATH(CHECK_LT(2, 1), "");
  EXPECT_DEATH(CHECK_GE(1, 2), "");
  EXPECT_DEATH(CHECK_GT(1, 1), "");
  EXPECT_DEATH(CHECK_GT(1, 2), "");
}

TEST(ShimMacros, CheckOk) {
  // Checks that CHECK_OK(Status::Ok) doesn't break
  StatusOr<std::unique_ptr<int>> thing(ReturnUniquePtr());
  CHECK_OK(thing.status());
  // Checks that CHECK_OK(!Status::Ok) breaks
  StatusOr<NoDefaultConstructor> statusor(
      absl::Status(absl::StatusCode::kCancelled, "Cancelled"));
  EXPECT_DEATH(CHECK_OK(statusor.status()), "Cancelled");
}

TEST(ShimMacros, ReturnIfError) {
  // Checks RETURN_IF_ERROR doesn't break with OkStatus
  absl::Status statusok = absl::OkStatus();
  EXPECT_TRUE(check_return_error(statusok).ok());

  // CHECKS RETURN_IF_BREAK if status is not OkSTatus
  absl::Status badstatus(absl::StatusCode::kCancelled, "");
  EXPECT_EQ(check_return_error(badstatus), badstatus);
}

absl::Status fail_false() {
  RETURN_IF_FALSE(false, absl::InvalidArgumentError) <<
                  "error: got false";
  return absl::OkStatus();
}

absl::Status fail_true() {
  RETURN_IF_TRUE(true, absl::InternalError) << "error: got true";
  absl::Status status = absl::OkStatus();
  return status;
}

// Test RETURN_IF_FALSE and friends that they return
// absl::InternalError if condition fails
TEST(ShimMacros, ReturnIfCondition) {
  EXPECT_EQ(fail_false(),
            absl::InvalidArgumentError("error: got false"));
  EXPECT_EQ(fail_true(), absl::InternalError("error: got true"));
}

TEST(ShimMacros, AssignOrReturn) {
  // Checks ASSIGN_OR_RETURN doesn't break with OkStatus
  absl::Status statusok = absl::OkStatus();
  StatusOr<int> thing = check_return_42(statusok);
  EXPECT_TRUE(thing.ok());
  EXPECT_EQ(thing.value(), 42);
  StatusOr<int> thing2 = double_check_return(statusok);
  EXPECT_TRUE(thing2.ok());
  EXPECT_EQ(thing2.value(), 42);

  // CHECKS ASSIGN_OR_RETURN if status is not OkSTatus
  absl::Status badstatus(absl::StatusCode::kCancelled, "");
  StatusOr<int> thing3 = check_return_42(badstatus);
  EXPECT_EQ(check_return_42(badstatus).status(), badstatus);
}

TEST(ShimMacros, CheckOkAndAssign) {
  // Checks CHECK_OK_AND_ASSIGN doesn't break with OkStatus
  absl::Status statusok = absl::OkStatus();
  CHECK_OK_AND_ASSIGN(int thing, check_return_42(statusok));
  CHECK_OK_AND_ASSIGN(int thing2, check_return_42(statusok));
  thing++;
  thing2++;
}

}  // namespace
}  // namespace shim
