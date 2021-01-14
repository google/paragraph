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
#include "paragraph/bridging/hlo/hlo_converter.h"

#include "gtest/gtest.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

TEST(HloConverter, SmallGraphCost) {
  xla::XlaBuilder builder = xla::XlaBuilder("matrix_multiply");
  auto lhs = Parameter(&builder, 0, xla::ShapeUtil::MakeShape(
      xla::F32, {10, 5}), "lhs");
  auto rhs = Parameter(&builder, 1, xla::ShapeUtil::MakeShape(
      xla::F32, {5, 30}), "rhs");
  Dot(lhs, rhs);
  auto computation_status = builder.Build();
  TF_CHECK_OK(computation_status.status());
  auto computation = computation_status.ConsumeValueOrDie();
  auto config = xla::HloModule::CreateModuleConfigFromProto(
      computation.proto(), xla::DebugOptions()).ConsumeValueOrDie();
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          xla::HloModule::CreateFromProto(
      computation.proto(), config));

  const ComputeCostAnalysis::Properties perf_prop = {
    { ComputeCostAnalysis::kFlopsKey, 1.0 },
    { ComputeCostAnalysis::kTranscendentalsKey, 1.0 },
    { ComputeCostAnalysis::kBytesAccessedKey, 1.0 }
  };
  TF_ASSERT_OK_AND_ASSIGN(auto graph,
                          HloConverter(hlo_module.get(), perf_prop));
}
