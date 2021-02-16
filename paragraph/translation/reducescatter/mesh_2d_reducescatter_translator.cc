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
#include "paragraph/translation/reducescatter/mesh_2d_reducescatter_translator.h"

#include <memory>
#include <string>
#include <utility>
#include <unordered_set>
#include <vector>

#include "factory/ObjectFactory.h"
#include "paragraph/translation/utils.h"

namespace paragraph {

Mesh2dReduceScatterTranslator::Mesh2dReduceScatterTranslator(
    nlohmann::json config) {
  CHECK_NE(config.find("dimension_widths"), config.end()) <<
      "2D Mesh should have field 'dimension_widths' as an array of size 2.";
  CHECK(config["dimension_widths"].is_array()) <<
      "2D Mesh config field 'dimension_widths' should be an array.";
  CHECK_EQ(config["dimension_widths"].size(), 2) <<
      "2D Mesh config field 'dimension_widths' should should have size 2.";
  for (size_t i = 0; i < config["dimension_widths"].size(); i++) {
    uint64_t width = config["dimension_widths"][i].get<uint64_t>();
    CHECK_GT(width, 1) << "Mesh width should be more than 1.";
    dimension_sizes_.push_back(width);
  }
  // Extract concentration (number of processors per mesh node) from config
  // By default equals 1
  concentration_ = 1;
  if (config.find("concentration") != config.end()) {
    concentration_ = config["concentration"].get<uint64_t>();
  }

  // Create json config for internal 1D Mesh reduce-scatter
  nlohmann::json implicit_config = R"(
    { "algorithm": "mesh-1d" }
  )"_json;
  // If we have a barrier in 2D Mesh, we need to instantiate a barrier before
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
    Mesh2dReduceScatterTranslator::GetSubroutine(
        Subroutine* reduction_subroutine,
        const std::string& name_prefix,
        Instruction* calling_instruction,
        int64_t processor_id,
        int64_t processor_index,
        const CommunicationGroup& comm_group,
        double comm_size) const {
  auto graph = calling_instruction->GetGraph();
  auto reducescatter_subroutine = absl::make_unique<Subroutine>(
      absl::StrCat(name_prefix, "_mesh-2d"), graph);
  auto reducescatter_sub_ptr = reducescatter_subroutine.get();
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
      ASSIGN_OR_RETURN(auto reducescatter_conc, Instruction::Create(
          Opcode::kReduceScatter,
          absl::StrCat(name_prefix,
                       "_dim-conc"),
          reducescatter_sub_ptr));
      reducescatter_conc->AppendCommunicationGroup(comm_group_conc);
      reducescatter_conc->SetBytesOut(comm_size * concentration_ /
                                      comm_group.size());
      ASSIGN_OR_RETURN(auto reduction_subroutine_conc,
                       reduction_subroutine->Clone("", /*reset_ids*/ false));
      reduction_subroutine_conc->ScalePerformance(1.0 * concentration_
                                                  / comm_group.size());
      reducescatter_conc->AppendInnerSubroutine(std::move(
          reduction_subroutine_conc));
      RETURN_IF_ERROR(reducescatter_translator_->Translate(reducescatter_conc));
      previous_instruction = reducescatter_conc;
    }
  }
  // Now do the same for every dimension of the mesh
  for (size_t dim = 0; dim < dimension_sizes_.size(); dim++) {
    processor_coordinates = ConsecutiveProcessorIdToGridCoordinates(
        processor_id, dimension_sizes_, concentration_);
    CommunicationGroup comm_group_mesh;
    uint64_t dim_width = dimension_sizes_.at(dim);
    for (uint64_t i = 0; i < dim_width; i++) {
      processor_coordinates.at(dim + 1) = i;
      uint64_t new_processor_id = GridCoordinatesToConsecutiveProcessorId(
          processor_coordinates, dimension_sizes_, concentration_);
      if (whole_world.find(new_processor_id) != whole_world.end()) {
        comm_group_mesh.push_back(new_processor_id);
      }
    }
    // If we don't have any communication in original comm_group within the
    // current dimension, just leave it
    if (comm_group_mesh.size() > 1) {
      ASSIGN_OR_RETURN(auto reducescatter_mesh, Instruction::Create(
          Opcode::kReduceScatter,
          absl::StrCat(name_prefix, "_dim-", dim),
          reducescatter_sub_ptr));
      reducescatter_mesh->AppendCommunicationGroup(comm_group_mesh);
      reducescatter_mesh->SetBytesOut(comm_size * dim_width /
                                      comm_group.size());
      ASSIGN_OR_RETURN(auto reduction_subroutine_mesh,
                       reduction_subroutine->Clone("", /*reset_ids*/ false));
      reduction_subroutine_mesh->ScalePerformance(1.0 * dim_width
                                                  / comm_group.size());
      reducescatter_mesh->AppendInnerSubroutine(std::move(
          reduction_subroutine_mesh));
      if (previous_instruction != nullptr) {
        reducescatter_mesh->AddOperand(previous_instruction);
      }
      RETURN_IF_ERROR(reducescatter_translator_->Translate(reducescatter_mesh));
      previous_instruction = reducescatter_mesh;
    }
  }
  // Set root instruction for reducescatter subroutine
  RETURN_IF_ERROR(reducescatter_subroutine->SetRootInstruction(
      previous_instruction));
  return reducescatter_subroutine;
}

registerWithObjectFactory(
    "mesh-2d",
    ReduceScatterTranslator,
    Mesh2dReduceScatterTranslator,
    nlohmann::json);

}  // namespace paragraph
