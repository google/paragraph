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
#include "paragraph/translation/allgather/torus_2d_allgather_translator.h"

#include <memory>
#include <string>
#include <utility>
#include <unordered_set>
#include <vector>

#include "factory/ObjectFactory.h"
#include "paragraph/translation/utils.h"

namespace paragraph {

Torus2dAllGatherTranslator::Torus2dAllGatherTranslator(
    nlohmann::json config) {
  CHECK_NE(config.find("dimension_widths"), config.end()) <<
      "2D Torus should have field 'dimension_widths' as an array of size 2.";
  CHECK(config["dimension_widths"].is_array()) <<
      "2D Torus config field 'dimension_widths' should be an array.";
  CHECK_EQ(config["dimension_widths"].size(), 2) <<
      "2D Torus config field 'dimension_widths' should should have size 2.";
  for (size_t i = 0; i < config["dimension_widths"].size(); i++) {
    uint64_t width = config["dimension_widths"][i].get<uint64_t>();
    CHECK_GT(width, 1) << "Torus width should be more than 1.";
    dimension_sizes_.push_back(width);
  }
  // Extract concentration (number of processors per torus node) from config
  // By default equals 1
  concentration_ = 1;
  if (config.find("concentration") != config.end()) {
    concentration_ = config["concentration"].get<uint64_t>();
  }

  // Create json config for internal 1D Torus all-gather
  nlohmann::json implicit_config = R"(
    { "algorithm": "bidir-ring" }
  )"_json;
  // If we have a barrier in 2D Torus, we need to instanciate a barrier before
  // all-gather in each dimension
  if (config.find("barrier") != config.end()) {
    implicit_config["barrier"] = config["barrier"];
  }
  auto maybe_allgather_translator = AllGatherTranslator::Create(
      implicit_config);
  CHECK_OK(maybe_allgather_translator.status());
  allgather_translator_ = std::move(maybe_allgather_translator.value());
}

shim::StatusOr<std::unique_ptr<Subroutine>>
    Torus2dAllGatherTranslator::GetSubroutine(
        const std::string& name_prefix,
        Instruction* calling_instruction,
        int64_t processor_id,
        int64_t processor_index,
        const CommunicationGroup& comm_group,
        double comm_size) const {
  auto graph = calling_instruction->GetGraph();
  auto allgather_subroutine = absl::make_unique<Subroutine>(
      absl::StrCat(name_prefix, "_torus-2d"), graph);
  auto allgather_sub_ptr = allgather_subroutine.get();
  RETURN_IF_FALSE(comm_group.at(processor_index) == processor_id,
                  absl::InvalidArgumentError) <<
      "Processor index points to the wrong Processor ID.";
  Instruction* previous_instruction = nullptr;
  std::vector<uint64_t> processor_coordinates;
  std::unordered_set<int64_t> whole_world(comm_group.begin(), comm_group.end());
  // Check if we have non-trivial concentration first
  if (concentration_ > 1) {
    processor_coordinates = ConsecutiveProcessorIdToGridCoordinates(
        processor_id, dimension_sizes_, concentration_);
    CommunicationGroup comm_group_conc;
    for (uint64_t i = 0; i < concentration_; i++) {
      processor_coordinates.at(0) = i;
      uint64_t new_processor_id = GridCoordinatesToConsecutiveProcessorId(
          processor_coordinates, dimension_sizes_, concentration_);
      if (whole_world.find(new_processor_id) != whole_world.end()) {
        comm_group_conc.push_back(new_processor_id);
      }
    }
    if (comm_group_conc.size() > 1) {
      ASSIGN_OR_RETURN(auto allgather_conc, Instruction::Create(
          Opcode::kAllGather,
          absl::StrCat(name_prefix,
                       "_dim-conc"),
          allgather_sub_ptr));
      allgather_conc->AppendCommunicationGroup(comm_group_conc);
      allgather_conc->SetBytesOut(comm_size * concentration_ /
                                      comm_group.size());
      RETURN_IF_ERROR(allgather_translator_->Translate(allgather_conc));
      previous_instruction = allgather_conc;
    }
  }
  // Now do the same for every dimension of the torus
  for (size_t dim = 0; dim < dimension_sizes_.size(); dim++) {
    processor_coordinates = ConsecutiveProcessorIdToGridCoordinates(
        processor_id, dimension_sizes_, concentration_);
    CommunicationGroup comm_group_torus;
    uint64_t dim_width = dimension_sizes_.at(dim);
    for (uint64_t i = 0; i < dim_width; i++) {
      processor_coordinates.at(dim + 1) = i;
      uint64_t new_processor_id = GridCoordinatesToConsecutiveProcessorId(
          processor_coordinates, dimension_sizes_, concentration_);
      if (whole_world.find(new_processor_id) != whole_world.end()) {
        comm_group_torus.push_back(new_processor_id);
      }
    }
    // If we don't have any communication in original comm_group within the
    // current dimension, just leave it
    if (comm_group_torus.size() > 1) {
      ASSIGN_OR_RETURN(auto allgather_torus, Instruction::Create(
          Opcode::kAllGather,
          absl::StrCat(name_prefix, "_dim-", dim),
          allgather_sub_ptr));
      allgather_torus->AppendCommunicationGroup(comm_group_torus);
      allgather_torus->SetBytesOut(comm_size * dim_width /
                                      comm_group.size());
      if (previous_instruction != nullptr) {
        allgather_torus->AddOperand(previous_instruction);
      }
      RETURN_IF_ERROR(allgather_translator_->Translate(allgather_torus));
      previous_instruction = allgather_torus;
    }
  }
  // Set root instruction for allgather subroutine
  RETURN_IF_ERROR(allgather_subroutine->SetRootInstruction(
      previous_instruction));
  return allgather_subroutine;
}

registerWithObjectFactory(
    "torus-2d",
    AllGatherTranslator,
    Torus2dAllGatherTranslator,
    nlohmann::json);

}  // namespace paragraph
