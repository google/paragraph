/* Copyright 2021 Nic McDonald
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
#include "paragraph/translation/allreduce/dissemination_allreduce_translator.h"

#include <cmath>
#include <memory>
#include <string>
#include <utility>

#include "factory/ObjectFactory.h"

namespace paragraph {

DisseminationAllReduceTranslator::DisseminationAllReduceTranslator(
    nlohmann::json config) {}

shim::StatusOr<std::unique_ptr<Subroutine>>
DisseminationAllReduceTranslator::GetSubroutine(
    Subroutine* reduction_subroutine,
    const std::string& name_prefix,
    Instruction* calling_instruction,
    int64_t processor_id,
    int64_t processor_index,
    const CommunicationGroup& comm_group,
    double comm_size) const {
  auto graph = calling_instruction->GetGraph();
  auto allreduce_subroutine = absl::make_unique<Subroutine>(
      absl::StrCat(name_prefix, "_dissemination"), graph);
  auto allreduce_sub_ptr = allreduce_subroutine.get();
  RETURN_IF_FALSE(comm_group.at(processor_index) == processor_id,
                  absl::InvalidArgumentError) <<
      "Processor index points to the wrong Processor ID.";
  Instruction* previous_instruction = nullptr;

  // Dissemination makes ceil(log2(N)) iterations
  // Each iteration does a SendRecv then performs the reduction
  int64_t iterations =
      static_cast<int64_t>(std::ceil(std::log2(comm_group.size())));
  for (int64_t iteration = 0; iteration < iterations; iteration++) {
    // Determines the peers for this iteration
    int64_t offset = static_cast<int64_t>(std::pow(2, iteration));
    int64_t send_peer_index = (processor_index + offset) % comm_group.size();
    int64_t recv_peer_index;
    if (offset > processor_index) {
      recv_peer_index = processor_index + comm_group.size() - offset;
    } else {
      recv_peer_index = processor_index - offset;
    }
    int64_t send_peer = comm_group.at(send_peer_index);
    int64_t recv_peer = comm_group.at(recv_peer_index);

    // Creates the SendRecv instruction
    ASSIGN_OR_RETURN(auto sendrecv, Instruction::Create(
        Opcode::kSendRecv,
        absl::StrCat(name_prefix,
                     "_dissemination_sendrecv_",
                     iteration),
        allreduce_sub_ptr));
    sendrecv->AppendCommunicationGroup({send_peer, recv_peer});
    sendrecv->SetBytesIn(comm_size);
    sendrecv->SetBytesOut(comm_size);
    if (previous_instruction != nullptr) {
      sendrecv->AddOperand(previous_instruction);
    }

    // Creates the reduction instruction
    ASSIGN_OR_RETURN(auto reduction, Instruction::Create(
        Opcode::kCall,
        absl::StrCat(name_prefix,
                     "_dissemination_reduction_",
                     iteration),
        allreduce_sub_ptr));
    ASSIGN_OR_RETURN(auto new_reduction_subroutine,
                     reduction_subroutine->Clone(
                         absl::StrCat("_iteration_", iteration),
                         /*reset_ids*/ true));
    reduction->AppendInnerSubroutine(std::move(new_reduction_subroutine));
    reduction->AddOperand(sendrecv);
    previous_instruction = reduction;
  }

  // Set root instruction for allreduce subroutine
  RETURN_IF_ERROR(allreduce_subroutine->SetRootInstruction(
      previous_instruction));
  return allreduce_subroutine;
}

registerWithObjectFactory(
    "dissemination",
    AllReduceTranslator,
    DisseminationAllReduceTranslator,
    nlohmann::json);

}  // namespace paragraph
