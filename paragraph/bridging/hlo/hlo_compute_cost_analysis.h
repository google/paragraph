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
#ifndef PARAGRAPH_BRIDGING_HLO_HLO_COMPUTE_COST_ANALYSIS_H_
#define PARAGRAPH_BRIDGING_HLO_HLO_COMPUTE_COST_ANALYSIS_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

int64_t ShapeSize(const xla::Shape& shape);

class ComputeCostAnalysis : public xla::HloCostAnalysis {
 public:
  static constexpr const char kOperandBytesAccessedKey[] =
      "operand_bytes_accessed";
  static constexpr const char kOutputBytesAccessedKey[] =
      "output_bytes_accessed";
  explicit ComputeCostAnalysis(const ShapeSizeFunction& shape_size);

  ComputeCostAnalysis(const ShapeSizeFunction& shape_size,
                      const Properties& per_second_rates);

  // All-Reduce originally considers bytes from operands, not bytes coming from
  // the network, which also come through the memory. We need to double operand
  // bytes to get cost estimate right.
  xla::Status HandleAllReduce(const xla::HloInstruction* crs) override;

  xla::Status Postprocess(const xla::HloInstruction* hlo) override;

  // Need to update new properties after HloCostAnalysis is finished to access
  // data from nested computations. Additional metrics can't be placed in
  // HloCostAnalysis::hlo_properties_ as it's protected, nor can they be
  // propagated by overloading HloCostAnalysis::ProcessSubcomputation.
  xla::Status UpdateInstructionProperties();

  float GetPropertyForHlo(const xla::HloInstruction& hlo,
                          const std::string& key,
                          float default_value = 0.0) const;

 protected:
  // As in HloCostAnalysis
  typedef std::unordered_map<const xla::HloInstruction*, Properties>
      HloToProperties;
  HloToProperties instruction_properties_;

  std::unique_ptr<xla::HloCostAnalysis> CreateNestedCostAnalysis(
    const ShapeSizeFunction& shape_size,
    const Properties& per_second_rates) override;
};

#endif  // PARAGRAPH_BRIDGING_HLO_HLO_COMPUTE_COST_ANALYSIS_H_

