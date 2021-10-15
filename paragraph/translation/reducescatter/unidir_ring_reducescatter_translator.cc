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
#include "paragraph/translation/reducescatter/unidir_ring_reducescatter_translator.h"

#include <memory>
#include <string>
#include <utility>

#include "factory/ObjectFactory.h"
#include "paragraph/translation/barrier/barrier_translator.h"

namespace paragraph {

UniDirRingReduceScatterTranslator::UniDirRingReduceScatterTranslator(
    nlohmann::json config) {
  barrier_translator_ = nullptr;
  if (config.find("barrier") != config.end()) {
    auto maybe_translator = BarrierTranslator::Create(config["barrier"]);
    CHECK_OK(maybe_translator.status());
    barrier_translator_ = std::move(maybe_translator.value());
  }
}

shim::StatusOr<std::unique_ptr<Subroutine>>
    UniDirRingReduceScatterTranslator::GetSubroutine(
        Subroutine* reduction_subroutine,
        const std::string& name_prefix,
        Instruction* calling_instruction,
        int64_t processor_id,
        int64_t processor_index,
        const CommunicationGroup& comm_group,
        double comm_size) const {
  auto graph = calling_instruction->GetGraph();
  auto reducescatter_subroutine = absl::make_unique<Subroutine>(
      absl::StrCat(name_prefix, "_unidir-ring"), graph);
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
                     "_unidir-ring_barrier"),
        reducescatter_sub_ptr));
    previous_instruction = barrier;
    barrier->AppendCommunicationGroup(comm_group);
    RETURN_IF_ERROR(barrier_translator_->Translate(barrier));
  }

  // Create CommunicationGroup for sendrecv instruction
  int64_t cw_peer_index = (comm_group.size() + processor_index + 1)
      % comm_group.size();
  int64_t ccw_peer_index = (comm_group.size() + processor_index - 1)
      % comm_group.size();
  CommunicationGroup sendrecv_comm_group = {
    comm_group.at(ccw_peer_index),
    comm_group.at(cw_peer_index)
  };
  // Send/Recv data 1/N data to/from neighbors N-1 times
  for (size_t i = 1; i < comm_group.size(); ++i) {
    ASSIGN_OR_RETURN(auto sendrecv, Instruction::Create(
        Opcode::kSendRecv,
        absl::StrCat(name_prefix,
                     "_unidir-ring_sendrecv_",
                     i),
        reducescatter_sub_ptr));
    sendrecv->AppendCommunicationGroup(sendrecv_comm_group);
    sendrecv->SetBytesIn(comm_size / comm_group.size());
    sendrecv->SetBytesOut(comm_size / comm_group.size());
    // We need to use previous communication as an operand to the next one`
    if (previous_instruction != nullptr) {
      sendrecv->AddOperand(previous_instruction);
    }
    // Add reduction part of the operation
    ASSIGN_OR_RETURN(auto reduction, Instruction::Create(
        Opcode::kCall,
        absl::StrCat(name_prefix,
                     "_unidir-ring_reduction_",
                     i),
        reducescatter_sub_ptr));
    ASSIGN_OR_RETURN(auto new_reduction_subroutine,
                     reduction_subroutine->Clone(absl::StrCat("_phase_", i),
                                                 /*reset_ids*/ true));
    new_reduction_subroutine->ScalePerformance(1.0/comm_group.size());
    reduction->AppendInnerSubroutine(std::move(new_reduction_subroutine));
    reduction->AddOperand(sendrecv);
    previous_instruction = reduction;
  }
  // Set root instruction for reducescatter subroutine
  RETURN_IF_ERROR(reducescatter_subroutine->SetRootInstruction(
      previous_instruction));
  return reducescatter_subroutine;
}

registerWithObjectFactory(
    "unidir-ring",
    ReduceScatterTranslator,
    UniDirRingReduceScatterTranslator,
    nlohmann::json);

}  // namespace paragraph
