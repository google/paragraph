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
#include "paragraph/translation/reducescatter/ring_over_2d_grid_reducescatter_translator.h"

#include <memory>
#include <string>
#include <utility>
#include <unordered_set>
#include <vector>

#include "factory/ObjectFactory.h"
#include "paragraph/translation/utils.h"

namespace paragraph {

RingOver2dGridReduceScatterTranslator::RingOver2dGridReduceScatterTranslator(
    nlohmann::json config) {
  CHECK_NE(config.find("dimension_widths"), config.end()) <<
      "2D Grid should have field 'dimension_widths' as an array of size 2.";
  CHECK(config["dimension_widths"].is_array()) <<
      "2D Grid config field 'dimension_widths' should be an array.";
  CHECK_EQ(config["dimension_widths"].size(), 2) <<
      "2D Grid config field 'dimension_widths' should should have size 2.";
  for (size_t i = 0; i < config["dimension_widths"].size(); i++) {
    uint64_t width = config["dimension_widths"][i].get<uint64_t>();
    CHECK_GT(width, 1) << "Grid width should be more than 1.";
    dimension_sizes_.push_back(width);
  }
  // Extract concentration (number of processors per mesh node) from config
  // By default equals 1
  concentration_ = 1;
  if (config.find("concentration") != config.end()) {
    concentration_ = config["concentration"].get<uint64_t>();
  }

  // Create json config for internal BiDirectional Ring reduce-scatter
  nlohmann::json implicit_config = R"(
    { "algorithm": "bidir-ring" }
  )"_json;
  // If we have a barrier in 2D Grid, we need to instanciate a barrier before
  // reduce-scatter in each dimension
  if (config.find("barrier") != config.end()) {
    implicit_config["barrier"] = config["barrier"];
  }
  auto maybe_reducescatter_translator = ReduceScatterTranslator::Create(
      implicit_config);
  CHECK_OK(maybe_reducescatter_translator.status());
  reducescatter_translator_ = std::move(maybe_reducescatter_translator.value());
}

shim::StatusOr<std::unique_ptr<Subroutine>>
    RingOver2dGridReduceScatterTranslator::GetSubroutine(
        Subroutine* reduction_subroutine,
        const std::string& name_prefix,
        Instruction* calling_instruction,
        int64_t processor_id,
        int64_t processor_index,
        const CommunicationGroup& comm_group,
        double comm_size) const {
  auto graph = calling_instruction->GetGraph();
  auto reducescatter_subroutine = absl::make_unique<Subroutine>(
      absl::StrCat(name_prefix, "_ring-2d-grid"), graph);
  auto reducescatter_sub_ptr = reducescatter_subroutine.get();
  RETURN_IF_FALSE(comm_group.at(processor_index) == processor_id,
                  absl::InvalidArgumentError) <<
      "Processor index points to the wrong Processor ID.";
  std::vector<uint64_t> processor_coordinates;
  std::unordered_set<int64_t> comm_world(comm_group.begin(), comm_group.end());
  CommunicationGroup full_ring = Swizzling2dGridToRing(dimension_sizes_,
                                                       concentration_);
  CommunicationGroup ring_comm_group;
  // We form new comm group from given comm group in the swizzled ring order
  for (int64_t id : full_ring) {
    if (comm_world.find(id) != comm_world.end()) {
      ring_comm_group.push_back(id);
    }
  }
  ASSIGN_OR_RETURN(auto reducescatter, Instruction::Create(
      Opcode::kReduceScatter,
      absl::StrCat(name_prefix,
                   "_ring-2d-grid"),
      reducescatter_sub_ptr,
      /*is_root = */ true));
  reducescatter->AppendCommunicationGroup(ring_comm_group);
  reducescatter->SetBytesOut(comm_size);
  ASSIGN_OR_RETURN(auto reduction_subroutine_clone,
                   reduction_subroutine->Clone("", /*reset_ids*/ false));
  reducescatter->AppendInnerSubroutine(std::move(
      reduction_subroutine_clone));
  RETURN_IF_ERROR(reducescatter_translator_->Translate(reducescatter));
  return reducescatter_subroutine;
}

registerWithObjectFactory(
    "ring-2d-grid",
    ReduceScatterTranslator,
    RingOver2dGridReduceScatterTranslator,
    nlohmann::json);

}  // namespace paragraph
