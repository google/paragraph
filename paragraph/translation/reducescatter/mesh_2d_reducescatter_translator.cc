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
  // conentrated ports
  integrated_local_exchange_ = false;
  if (config.find("integrated_local_exchange") != config.end()) {
    integrated_local_exchange_ =
        config["integrated_local_exchange"].get<bool>();
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
  CommunicationGroup local_comm_group = CommunicationGroupLocalProjection(
      processor_id, comm_group, dimension_sizes_, concentration_);
  std::vector<double> stage_comm_sizes;
  // We prepare communication sizes for each stage and each dimension as
  // dimension and/or communication groups could be uneven
  for (size_t dim = 0; dim < dimension_sizes_.size(); dim++) {
    stage_comm_sizes.push_back(comm_size / comm_group.size());
    if (integrated_local_exchange_) {
      stage_comm_sizes.at(dim) *= local_comm_group.size();
    }
  }
  // We have as many stages as dimensions in the Torus
  for (size_t stage = 0; stage < dimension_sizes_.size(); stage++) {
    // We run AllGather in parallel for every dimension of Torus
    std::vector<Instruction*> parallel_reducescatter;
    for (size_t dim = 0; dim < dimension_sizes_.size(); dim++) {
      auto new_comm_group = CommunicationGroupProjectionOnGrid(
          processor_id, comm_group, dim, integrated_local_exchange_,
          dimension_sizes_, concentration_);
      // Every new stage we should increase communication size
      // On the first stage we only exchange data laying in the 1st dimension
      // On the second stage we exchange data from both 1st and 2nd dimensions
      stage_comm_sizes.at(dim) *= new_comm_group.size();
      if (integrated_local_exchange_) {
        stage_comm_sizes.at(dim) /= local_comm_group.size();
      }
      // If we don't have any communication in original comm_group within the
      // current dimension, just leave it
      if (new_comm_group.size() > 1) {
        ASSIGN_OR_RETURN(auto reducescatter_stage, Instruction::Create(
            Opcode::kReduceScatter,
            absl::StrCat(name_prefix, "_stage-", stage, "_dim-", dim),
            reducescatter_sub_ptr));
        reducescatter_stage->AppendCommunicationGroup(new_comm_group);
        reducescatter_stage->SetBytesOut(stage_comm_sizes.at(dim));
        if (previous_instruction != nullptr) {
          reducescatter_stage->AddOperand(previous_instruction);
        }
        ASSIGN_OR_RETURN(auto reduction_subroutine_stage,
                         reduction_subroutine->Clone("", /*reset_ids*/ false));
        reduction_subroutine_stage->ScalePerformance(
            1.0 * stage_comm_sizes.at(dim) / comm_size);
        reducescatter_stage->AppendInnerSubroutine(std::move(
            reduction_subroutine_stage));
        RETURN_IF_ERROR(reducescatter_translator_->Translate(
            reducescatter_stage));
        parallel_reducescatter.push_back(reducescatter_stage);
      }
    }
    ASSIGN_OR_RETURN(auto reducescatter_root, Instruction::Create(
        Opcode::kNull,
        absl::StrCat(name_prefix, "_stage-", stage, "_root"),
        reducescatter_sub_ptr));
    previous_instruction = reducescatter_root;
    for (auto& instr : parallel_reducescatter) {
      reducescatter_root->AddOperand(instr);
    }
  }
  // Check if we have non-trivial concentration and need to perform
  // explicit local exchange step
  if ((concentration_ > 1) && !integrated_local_exchange_) {
    if (local_comm_group.size() > 1) {
      ASSIGN_OR_RETURN(auto reducescatter_conc, Instruction::Create(
          Opcode::kReduceScatter,
          absl::StrCat(name_prefix, "_conc"),
          reducescatter_sub_ptr));
      reducescatter_conc->AppendCommunicationGroup(local_comm_group);
      reducescatter_conc->SetBytesOut(comm_size);
      if (previous_instruction != nullptr) {
        reducescatter_conc->AddOperand(previous_instruction);
      }
      ASSIGN_OR_RETURN(auto reduction_subroutine_conc,
                       reduction_subroutine->Clone("", /*reset_ids*/ false));
      reducescatter_conc->AppendInnerSubroutine(std::move(
          reduction_subroutine_conc));
      RETURN_IF_ERROR(reducescatter_translator_->Translate(
          reducescatter_conc));
      previous_instruction = reducescatter_conc;
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
