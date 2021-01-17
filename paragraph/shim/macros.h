/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *     ==============================================================================*/
/* Modified by Google */

#ifndef PARAGRAPH_SHIM_MACROS_H_
#define PARAGRAPH_SHIM_MACROS_H_

#include <iostream>
#include <sstream>
#include <string>
#include <utility>

#include "paragraph/shim/statusor.h"

#ifdef __has_builtin
#define HAS_BUILTIN(x) __has_builtin(x)
#else
#define HAS_BUILTIN(x) 0
#endif

// Compilers can be told that a certain branch is not likely to be taken
// (for instance, a CHECK failure), and use that information in static
// analysis. Giving it this information can help it optimize for the
// common case in the absence of better information (ie.
// -fprofile-arcs).
#if HAS_BUILTIN(__builtin_expect) || (defined(__GNUC__) && __GNUC__ >= 3)
#define PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#else
#define PREDICT_FALSE(x) (x)
#define PREDICT_TRUE(x) (x)
#endif

#define CONCAT_IMPL(x, y) x##y
#define CONCAT_MACRO(x, y) CONCAT_IMPL(x, y)

struct StreamFatal {
  std::ostream& stream_;
  explicit StreamFatal(std::ostream& os = std::cerr) : stream_(os) {}

  template <typename T>
  StreamFatal& operator<<(const T& val) {
    stream_ << val;
    return *this;
  }

  ~StreamFatal() {
    stream_ << std::endl;
    abort();
  }
};

struct StreamToStatus {
  std::ostringstream stream_;
  absl::Status (*error_status_)(absl::string_view);
  explicit StreamToStatus(absl::Status (*error_status)(absl::string_view))
      : error_status_(error_status) {}

  template <typename T>
  StreamToStatus& operator<<(const T& val) {
    stream_ << val;
    return *this;
  }

  operator absl::Status() const& {
    return error_status_(stream_.str());
  }
  operator absl::Status() && {
    return error_status_(stream_.str());
  }

  template <typename U>
  operator shim::StatusOr<U>() const& {
    return error_status_(stream_.str());
  }
  template <typename U>
  operator shim::StatusOr<U>() && {
    return error_status_(stream_.str());
  }
};

// NOLINTNEXTLINE
#define CHECK(val) if (val) {} else (                                   \
    StreamFatal() << "CHECK(" << #val                                   \
    << ") Failed in file: " << __FILE__ << ", at line: "                \
    << __LINE__ << ". ")

#define CHECK_EQ(val1, val2) CHECK(val1 == val2)
#define CHECK_NE(val1, val2) CHECK(val1 != val2)
#define CHECK_LE(val1, val2) CHECK(val1 <= val2)
#define CHECK_LT(val1, val2) CHECK(val1 < val2)
#define CHECK_GE(val1, val2) CHECK(val1 >= val2)
#define CHECK_GT(val1, val2) CHECK(val1 > val2)

#define CHECK_OK(val)                           \
  do {                                          \
    if (PREDICT_FALSE(!val.ok())) {             \
      fprintf(stderr,                           \
              "CHECK failed with Status %s\n",  \
              val.ToString().c_str());          \
      CHECK(val.ok());                          \
    }                                           \
  } while (0)

// For propagating errors when calling a function.
#define RETURN_IF_ERROR(...)                    \
  do {                                          \
    absl::Status _status = (__VA_ARGS__);       \
    if (PREDICT_FALSE(!_status.ok())) {         \
      return _status;                           \
    }                                           \
  } while (0)

#define RETURN_IF_FALSE(val, err_status)        \
  if (PREDICT_FALSE(!(val)))                    \
    return StreamToStatus(err_status)

#define RETURN_IF_TRUE(val, err_status)         \
  if (PREDICT_FALSE(val))                       \
    return StreamToStatus(err_status)

#define ASSIGN_OR_RETURN(lhs, rexpr)            \
  ASSIGN_OR_RETURN_IMPL(CONCAT_MACRO(           \
      _status_or, __COUNTER__), lhs, rexpr)

#define ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr) \
  auto statusor = (rexpr);                          \
  if (PREDICT_FALSE(!statusor.ok())) {              \
    return statusor.status();                       \
  }                                                 \
  lhs = std::move(statusor.value())

#define CHECK_OK_AND_ASSIGN(lhs, rexpr)                 \
    CHECK_OK_AND_ASSIGN_IMPL(CONCAT_MACRO(              \
        _status_or, __COUNTER__), lhs, rexpr)

#define CHECK_OK_AND_ASSIGN_IMPL(statusor, lhs, rexpr)   \
  auto statusor = (rexpr);                               \
  CHECK(statusor.status().ok()) <<                       \
      statusor.status();                                 \
  lhs = std::move(statusor.value())

#endif  // PARAGRAPH_SHIM_MACROS_H_
