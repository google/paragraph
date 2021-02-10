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
#include "paragraph/translation/allgather/bidir_ring_allgather_translator.h"

#include <memory>
#include <string>
#include <utility>

#include "factory/ObjectFactory.h"
#include "paragraph/translation/barrier/barrier_translator.h"

namespace paragraph {

BiDirRingAllGatherTranslator::BiDirRingAllGatherTranslator(
    nlohmann::json config) {
  barrier_translator_ = nullptr;
  if (config.find("barrier") != config.end()) {
    auto maybe_translator = BarrierTranslator::Create(config["barrier"]);
    CHECK_OK(maybe_translator.status());
    barrier_translator_ = std::move(maybe_translator.value());
  }
  // Create json config for internal unidir_ring all-gather
  nlohmann::json implicit_config = R"(
    { "algorithm": "unidir-ring" }
  )"_json;
  auto maybe_allgather_translator = AllGatherTranslator::Create(
                       implicit_config);
  CHECK_OK(maybe_allgather_translator.status());
  allgather_translator_ = std::move(maybe_allgather_translator.value());
}

shim::StatusOr<std::unique_ptr<Subroutine>>
    BiDirRingAllGatherTranslator::GetSubroutine(
        const std::string& name_prefix,
        Instruction* calling_instruction,
        int64_t processor_id,
        int64_t processor_index,
        const CommunicationGroup& comm_group,
        double comm_size) const {
  auto graph = calling_instruction->GetGraph();
  auto allgather_subroutine = absl::make_unique<Subroutine>(
      absl::StrCat(name_prefix, "_bidir-ring"), graph);
  auto allgather_sub_ptr = allgather_subroutine.get();
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
        allgather_sub_ptr));
    previous_instruction = barrier;
    barrier->AppendCommunicationGroup(comm_group);
    RETURN_IF_ERROR(barrier_translator_->Translate(barrier));
  }
  // Create CW AllGather instructions
  ASSIGN_OR_RETURN(auto allgather_cw, Instruction::Create(
      Opcode::kAllGather,
      absl::StrCat(name_prefix,
                   "_bidir-ring_cw"),
      allgather_sub_ptr));
  allgather_cw->SetBytesOut(comm_size / 2);
  allgather_cw->AppendCommunicationGroup(comm_group);
  // If there is a barrier, we need to add it as control dependency to the
  // first communication after it
  if (previous_instruction != nullptr) {
    allgather_cw->AddOperand(previous_instruction);
  }
  RETURN_IF_ERROR(allgather_translator_->Translate(allgather_cw));
  // Create CCW AllGather instructions
  ASSIGN_OR_RETURN(auto allgather_ccw, Instruction::Create(
      Opcode::kAllGather,
      absl::StrCat(name_prefix,
                   "_bidir-ring_ccw"),
      allgather_sub_ptr));
  CommunicationGroup comm_group_reversed = comm_group;
  std::reverse(comm_group_reversed.begin(), comm_group_reversed.end());
  allgather_ccw->SetBytesOut(comm_size / 2);
  allgather_ccw->AppendCommunicationGroup(comm_group_reversed);
  // If there is a barrier, we need to add it as control dependency to the
  // first communication after it
  if (previous_instruction != nullptr) {
    allgather_ccw->AddOperand(previous_instruction);
  }
  RETURN_IF_ERROR(allgather_translator_->Translate(allgather_ccw));
  // Create root instruction with dependincies on cw and ccw allgather
  ASSIGN_OR_RETURN(auto root, Instruction::Create(
      Opcode::kNull,
      absl::StrCat(name_prefix,
                   "_bidir-ring_root_",
                   processor_id),
      allgather_sub_ptr,
      /*is_root*/ true));
  root->AddOperand(allgather_cw);
  root->AddOperand(allgather_ccw);
  return allgather_subroutine;
}

registerWithObjectFactory(
    "bidir-ring",
    AllGatherTranslator,
    BiDirRingAllGatherTranslator,
    nlohmann::json);

}  // namespace paragraph
