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
#include "paragraph/translation/allreduce/ring_over_2d_grid_allreduce_translator.h"

#include <memory>
#include <string>
#include <utility>

#include "factory/ObjectFactory.h"
#include "paragraph/translation/utils.h"

namespace paragraph {

RingOver2dGridAllReduceTranslator::RingOver2dGridAllReduceTranslator(
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

  // Create json config for internal reduce-scatter and all-gather
  nlohmann::json implicit_config = R"(
    { "algorithm": "bidir-ring" }
  )"_json;
  // If we have a barrier in 2D Grid, we need to instanciate a barrier before
  // all-gather in each dimension
  if (config.find("barrier") != config.end()) {
    implicit_config["barrier"] = config["barrier"];
  }

  auto maybe_allreduce_translator = AllReduceTranslator::Create(
      implicit_config);
  CHECK_OK(maybe_allreduce_translator.status());
  allreduce_translator_ = std::move(maybe_allreduce_translator.value());
}

shim::StatusOr<std::unique_ptr<Subroutine>>
    RingOver2dGridAllReduceTranslator::GetSubroutine(
        Subroutine* reduction_subroutine,
        const std::string& name_prefix,
        Instruction* calling_instruction,
        int64_t processor_id,
        int64_t processor_index,
        const CommunicationGroup& comm_group,
        double comm_size) const {
  auto graph = calling_instruction->GetGraph();
  auto allreduce_subroutine = absl::make_unique<Subroutine>(
      absl::StrCat(name_prefix, "_mesh-2d"), graph);
  auto allreduce_sub_ptr = allreduce_subroutine.get();
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
  ASSIGN_OR_RETURN(auto allreduce, Instruction::Create(
      Opcode::kAllReduce,
      absl::StrCat(name_prefix,
                   "_ring-2d-grid"),
      allreduce_sub_ptr,
      /*is_root = */ true));
  allreduce->AppendCommunicationGroup(ring_comm_group);
  allreduce->SetBytesOut(comm_size);
  ASSIGN_OR_RETURN(auto reduction_subroutine_clone,
                   reduction_subroutine->Clone("", /*reset_ids*/ false));
  allreduce->AppendInnerSubroutine(std::move(
      reduction_subroutine_clone));
  RETURN_IF_ERROR(allreduce_translator_->Translate(allreduce));
  return allreduce_subroutine;
}

registerWithObjectFactory(
    "ring-2d-grid",
    AllReduceTranslator,
    RingOver2dGridAllReduceTranslator,
    nlohmann::json);

}  // namespace paragraph
