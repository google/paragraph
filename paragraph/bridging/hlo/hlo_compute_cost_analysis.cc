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
#include "paragraph/bridging/hlo/hlo_compute_cost_analysis.h"

#include <algorithm>
#include <memory>
#include <string>

// Cost analysis of HloModule
int64_t ShapeSize(const xla::Shape& shape) {
  constexpr int64_t kPointerSize = 8;
  return xla::ShapeUtil::ByteSizeOf(shape, kPointerSize);
}

ComputeCostAnalysis::ComputeCostAnalysis(const ShapeSizeFunction& shape_size)
    : xla::HloCostAnalysis(shape_size) {
    // Dummy default values - 1 TeraFlops and 100 GB/s
    // Can't use constexpr because properties is a map (not a literal type),
    // Can't use const because Properties are defined inside the class
    set_flops_per_second(1. * 1e12);
    set_transcendentals_per_second(1. * 1e12);
    set_bytes_per_second(100.0 * 1024 * 1024 * 1024);
}

ComputeCostAnalysis::ComputeCostAnalysis(
    const ShapeSizeFunction& shape_size,
    const Properties& per_second_rates)
    : xla::HloCostAnalysis(shape_size, per_second_rates) {}

xla::Status ComputeCostAnalysis::HandleAllReduce(
    const xla::HloInstruction* crs) {
  TF_CHECK_OK(xla::HloCostAnalysis::HandleAllReduce(crs));
  // Sometimes cost analysis underreport bytes written. In that case we need to
  // take it from BytesRead
  int64_t bytes_written = 0;
  for (const xla::ShapeUtil::IndexedShape& indexed_shape :
       xla::ShapeUtil::GetLeafShapes(crs->shape())) {
    if (current_properties_.find(GetOutputBytesAccessedKey(
            indexed_shape.index)) != current_properties_.end()) {
      bytes_written += current_properties_[GetOutputBytesAccessedKey(
          indexed_shape.index)];
    }
  }
  if (bytes_written == 0) {
    current_properties_[kBytesAccessedKey] +=
        current_properties_[kBytesAccessedKey];
  }
  // AllReduce has only one regestered operand, data from the network (that's
  // written to memory though RDMA) should be taken into account, too.
  for (int64_t operund_num = 0;
       operund_num < crs->operand_count();
       ++operund_num) {
    for (const xla::ShapeUtil::IndexedShape& indexed_shape :
        xla::ShapeUtil::GetLeafShapes(crs->operand(operund_num)->shape())) {
      current_properties_[kBytesAccessedKey] +=
          current_properties_[GetOperandBytesAccessedKey(
              operund_num, indexed_shape.index).c_str()];
      SetOperandBytesAccessed(
          operund_num, indexed_shape.index,
          2 * current_properties_[GetOperandBytesAccessedKey(
              operund_num, indexed_shape.index).c_str()]);
      }
    }
  return xla::Status::OK();
}

xla::Status ComputeCostAnalysis::Postprocess(
    const xla::HloInstruction* hlo) {
  TF_CHECK_OK(xla::HloCostAnalysis::Postprocess(hlo));
  return xla::Status::OK();
}

xla::Status ComputeCostAnalysis::UpdateInstructionProperties() {
  for (auto& map_it : hlo_properties_) {
    const xla::HloInstruction* hlo = map_it.first;
    Properties hlo_property = {
      { kFlopsKey, std::max(flop_count(*hlo), 0LL) },
      { kTranscendentalsKey, std::max(transcendental_count(*hlo), 0LL) },
      { kBytesAccessedKey, std::max(bytes_accessed(*hlo), 0LL) },
      { kOperandBytesAccessedKey, std::max(GetBytesRead(*hlo), 0LL) },
      { kOutputBytesAccessedKey, std::max(GetBytesWritten(*hlo), 0LL) },
      { kOptimalSecondsKey, std::max(static_cast<double>(optimal_seconds(*hlo)),
                                     0.0) }
    };
    // HLO cost analysis tends to miss bytes written by AllReduce
    if (hlo->opcode() == xla::HloOpcode::kAllReduce) {
      if (hlo_property[kOutputBytesAccessedKey] == 0) {
        hlo_property[kOutputBytesAccessedKey] = GetBytesRead(*hlo) / 2;
      }
    }
    TF_RET_CHECK(instruction_properties_.emplace(hlo, hlo_property).second);
  }
  return xla::Status::OK();
}

float ComputeCostAnalysis::GetPropertyForHlo(const xla::HloInstruction& hlo,
                                             const std::string& key,
                                             float default_value) const {
  auto it = instruction_properties_.find(&hlo);
  if (it == instruction_properties_.end()) {
    return 0.0f;
  } else {
    return GetProperty(key, it->second);
  }
}

std::unique_ptr<xla::HloCostAnalysis>
  ComputeCostAnalysis::CreateNestedCostAnalysis(
    const ComputeCostAnalysis::ShapeSizeFunction& shape_size,
    const ComputeCostAnalysis::Properties& per_second_rates) {
  std::unique_ptr<xla::HloCostAnalysis> cost_analysis =
      absl::make_unique<ComputeCostAnalysis>(shape_size, per_second_rates);
  return cost_analysis;
}
