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
#include <memory>
#include <string>
#include <vector>

#include "paragraph/translation/barrier/centralized_barrier_translator.h"
#include "factory/ObjectFactory.h"

namespace paragraph {

CentralizedBarrierTranslator::CentralizedBarrierTranslator(
    nlohmann::json config) {}

shim::StatusOr<std::unique_ptr<Subroutine>>
    CentralizedBarrierTranslator::GetSubroutine(
        const std::string& name_prefix,
        Instruction* calling_instruction,
        int64_t processor_id,
        int64_t processor_index,
        const CommunicationGroup& comm_group) const {
  auto graph = calling_instruction->GetGraph();
  auto barrier_subroutine = absl::make_unique<Subroutine>(
      absl::StrCat(name_prefix, "_centralized"), graph);
  auto barrier_sub_ptr = barrier_subroutine.get();
  RETURN_IF_FALSE(comm_group.at(processor_index) == processor_id,
                  absl::InvalidArgumentError) <<
      "Processor index points to the wrong Processor ID.";
  // Translation of barrier for non-coordinator processor
  if (processor_index != 0) {
    CommunicationGroup manager_comm_group;
    manager_comm_group.push_back(comm_group.at(0));

    // First send barrier flag to the manager
    ASSIGN_OR_RETURN(auto send, Instruction::Create(
        Opcode::kSend,
        absl::StrCat(name_prefix, "_centralized_send_to_", 0),
        barrier_sub_ptr));
    send->AppendCommunicationGroup(manager_comm_group);

    // Then receive barrier flag from the manager
    ASSIGN_OR_RETURN(auto recv, Instruction::Create(
        Opcode::kRecv,
        absl::StrCat(name_prefix, "_centralized_recv_from_", 0),
        barrier_sub_ptr,
        /*is_root*/ true));
    recv->AppendCommunicationGroup(manager_comm_group);

    // Create control dependency for recv on send and
    // add the instructions to the barrier subroutine
    recv->AddOperand(send);
  } else {
    // Translation for barrier coordinator
    // Create recv instructions from all the other processors in comm_group and
    // add them to the vector to create dependencies later
    std::vector<Instruction*> dependencies_from_recv;
    for (size_t peer_index = 1;
         peer_index < comm_group.size();
         ++peer_index) {
      CommunicationGroup peer_comm_group;
      peer_comm_group.push_back(comm_group.at(peer_index));
      ASSIGN_OR_RETURN(auto recv, Instruction::Create(
        Opcode::kRecv,
        absl::StrCat(name_prefix,
                     "_centralized_coordinator_recv_from_",
                     comm_group.at(peer_index)),
        barrier_sub_ptr));
      recv->AppendCommunicationGroup(peer_comm_group);
      dependencies_from_recv.push_back(recv);
    }
    // Create send instructions for all the other processors in comm_group and
    // add all recv instructions from dependencies vector as a control
    // dependency
    std::vector<Instruction*> dependencies_from_send;
    for (size_t peer_index = 1;
         peer_index < comm_group.size();
         ++peer_index) {
      CommunicationGroup peer_comm_group;
      peer_comm_group.push_back(comm_group.at(peer_index));
      ASSIGN_OR_RETURN(auto send, Instruction::Create(
        Opcode::kSend,
        absl::StrCat(name_prefix,
                     "_centralized_coordinator_send_to_",
                     comm_group.at(peer_index)),
        barrier_sub_ptr));
      send->AppendCommunicationGroup(peer_comm_group);
      for (auto& instr : dependencies_from_recv) {
        send->AddOperand(instr);
      }
      dependencies_from_send.push_back(send);
    }
    // Create root instruction with dependincies on send instructions
    ASSIGN_OR_RETURN(auto root, Instruction::Create(
        Opcode::kNull,
        absl::StrCat(name_prefix,
                     "_centralized_root_",
                     processor_id),
        barrier_sub_ptr,
        /*is_root*/ true));
    for (auto& instr : dependencies_from_send) {
      root->AddOperand(instr);
    }
  }
  return barrier_subroutine;
}

registerWithObjectFactory(
    "centralized",
    BarrierTranslator,
    CentralizedBarrierTranslator,
    nlohmann::json);

}  // namespace paragraph
