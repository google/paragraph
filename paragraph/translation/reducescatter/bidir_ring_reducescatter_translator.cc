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
#include "paragraph/translation/reducescatter/bidir_ring_reducescatter_translator.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "factory/ObjectFactory.h"
#include "paragraph/translation/barrier/barrier_translator.h"

namespace paragraph {

BiDirRingReduceScatterTranslator::BiDirRingReduceScatterTranslator(
    nlohmann::json config) {
  barrier_translator_ = nullptr;
  if (config.find("barrier") != config.end()) {
    auto maybe_translator = BarrierTranslator::Create(config["barrier"]);
    CHECK_OK(maybe_translator.status());
    barrier_translator_ = std::move(maybe_translator.value());
  }
  // Create json config for internal unidir_ring reduce-scatter
  nlohmann::json implicit_config = R"(
    {
      "reduce-scatter": {
        "algorithm": "unidir-ring"
      }
    }
  )"_json;
  auto maybe_reducescatter_translator = ReduceScatterTranslator::Create(
                       implicit_config["reduce-scatter"]);
  CHECK_OK(maybe_reducescatter_translator.status());
  reducescatter_translator_ = std::move(maybe_reducescatter_translator.value());
}

shim::StatusOr<std::unique_ptr<Subroutine>>
    BiDirRingReduceScatterTranslator::GetSubroutine(
        Subroutine* reduction_subroutine,
        const std::string& name_prefix,
        Instruction* calling_instruction,
        int64_t processor_id,
        int64_t processor_index,
        const CommunicationGroup& comm_group,
        double comm_size) const {
  auto graph = calling_instruction->GetGraph();
  auto reducescatter_subroutine = absl::make_unique<Subroutine>(
      absl::StrCat(name_prefix, "_bidir-ring"), graph);
  auto reducescatter_sub_ptr = reducescatter_subroutine.get();
  RETURN_IF_FALSE(comm_group.at(processor_index) == processor_id,
                  absl::InvalidArgumentError) <<
      "Processor index points to the wrong Processor ID.";
  Instruction* previous_instruction = nullptr;
  // Check if there is barrier in config, and if so then instantiate it and
  // translate using specific translator from config
  if (barrier_translator_ != nullptr) {
    ASSIGN_OR_RETURN(auto barrier, Instruction::Create(
        Opcode::kBarrier,
        absl::StrCat(name_prefix,
                     "_bidir-ring_barrier"),
        reducescatter_sub_ptr));
    previous_instruction = barrier;
    barrier->AppendCommunicationGroup(comm_group);
    RETURN_IF_ERROR(barrier_translator_->Translate(barrier));
  }
  // Create CW ReduceScatter instructions
  ASSIGN_OR_RETURN(auto reducescatter_cw, Instruction::Create(
      Opcode::kReduceScatter,
      absl::StrCat(name_prefix,
                   "_bidir-ring_cw"),
      reducescatter_sub_ptr));
  reducescatter_cw->SetBytesOut(comm_size / 2);
  reducescatter_cw->AppendCommunicationGroup(comm_group);
  ASSIGN_OR_RETURN(auto new_reduction_subroutine_cw,
                   reduction_subroutine->Clone(
                       "",
                       /*reset_ids*/ false));
  new_reduction_subroutine_cw->ScalePerformance(0.5);
  reducescatter_cw->AppendInnerSubroutine(std::move(
      new_reduction_subroutine_cw));
  // If there is a barrier, we need to add it as control dependency to the
  // first communication after it
  if (previous_instruction != nullptr) {
    reducescatter_cw->AddOperand(previous_instruction);
  }
  RETURN_IF_ERROR(reducescatter_translator_->Translate(reducescatter_cw));
  // Create CCW ReduceScatter instructions
  ASSIGN_OR_RETURN(auto reducescatter_ccw, Instruction::Create(
      Opcode::kReduceScatter,
      absl::StrCat(name_prefix,
                   "_bidir-ring_ccw"),
      reducescatter_sub_ptr));
  CommunicationGroup comm_group_reversed = comm_group;
  std::reverse(comm_group_reversed.begin(), comm_group_reversed.end());
  reducescatter_ccw->SetBytesOut(comm_size / 2);
  reducescatter_ccw->AppendCommunicationGroup(comm_group_reversed);
  ASSIGN_OR_RETURN(auto new_reduction_subroutine_ccw,
                   reduction_subroutine->Clone(
                       "",
                       /*reset_ids*/ false));
  new_reduction_subroutine_ccw->ScalePerformance(0.5);
  reducescatter_ccw->AppendInnerSubroutine(std::move(
      new_reduction_subroutine_ccw));
  // If there is a barrier, we need to add it as control dependency to the
  // first communication after it
  if (previous_instruction != nullptr) {
    reducescatter_ccw->AddOperand(previous_instruction);
  }
  RETURN_IF_ERROR(reducescatter_translator_->Translate(reducescatter_ccw));
  // Create root instruction with dependincies on cw and ccw reducescatter
  ASSIGN_OR_RETURN(auto root, Instruction::Create(
      Opcode::kNull,
      absl::StrCat(name_prefix,
                   "_bidir-ring_root_",
                   processor_id),
      reducescatter_sub_ptr,
      /*is_root*/ true));
  root->AddOperand(reducescatter_cw);
  root->AddOperand(reducescatter_ccw);
  return reducescatter_subroutine;
}

registerWithObjectFactory(
    "bidir-ring",
    ReduceScatterTranslator,
    BiDirRingReduceScatterTranslator,
    nlohmann::json);

}  // namespace paragraph
