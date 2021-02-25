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
#include "paragraph/translation/allgather/mesh_1d_allgather_translator.h"

#include <memory>
#include <string>
#include <utility>

#include "factory/ObjectFactory.h"
#include "paragraph/translation/barrier/barrier_translator.h"

namespace paragraph {

Mesh1dAllGatherTranslator::Mesh1dAllGatherTranslator(
    nlohmann::json config) {
  barrier_translator_ = nullptr;
  if (config.find("barrier") != config.end()) {
    auto maybe_translator = BarrierTranslator::Create(config["barrier"]);
    CHECK_OK(maybe_translator.status());
    barrier_translator_ = std::move(maybe_translator.value());
  }
}

shim::StatusOr<std::unique_ptr<Subroutine>>
    Mesh1dAllGatherTranslator::GetSubroutine(
        const std::string& name_prefix,
        Instruction* calling_instruction,
        int64_t processor_id,
        int64_t processor_index,
        const CommunicationGroup& comm_group,
        double comm_size) const {
  auto graph = calling_instruction->GetGraph();
  auto allgather_subroutine = absl::make_unique<Subroutine>(
      absl::StrCat(name_prefix, "_mesh-1d"), graph);
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
                     "_mesh-1d_barrier"),
        allgather_sub_ptr));
    previous_instruction = barrier;
    barrier->AppendCommunicationGroup(comm_group);
    RETURN_IF_ERROR(barrier_translator_->Translate(barrier));
  }

  // Create CommunicationGroup for sendrecv instruction
  int64_t cw_peer_index = (comm_group.size() + processor_index + 1)
      % comm_group.size();
  int64_t ccw_peer_index = (comm_group.size() + processor_index - 1)
      % comm_group.size();
  CommunicationGroup cw_comm_group = {
    comm_group.at(cw_peer_index),
    comm_group.at(cw_peer_index)
  };
  CommunicationGroup ccw_comm_group = {
    comm_group.at(ccw_peer_index),
    comm_group.at(ccw_peer_index)
  };
  // Send/Recv data 1/N data to/from neighbors N-1 times
  for (size_t i = 0; i < comm_group.size() - 1; ++i) {
    // Check if the node sends/receives message CW and CCW
    bool cw_send_flag = WillSendCw(processor_index, i, comm_group.size());
    bool cw_recv_flag = WillSendCcw(processor_index + 1, i, comm_group.size());
    bool ccw_send_flag = WillSendCcw(processor_index, i, comm_group.size());
    bool ccw_recv_flag = WillSendCw(processor_index - 1, i, comm_group.size());
    Instruction* ccw_sendrecv = nullptr;
    Instruction* cw_sendrecv = nullptr;
    if (ccw_send_flag || ccw_recv_flag) {
      // Create CCW communication instruction
      if (ccw_send_flag && !ccw_recv_flag) {
        // If only sends CCW but doesn't receive from there
        ASSIGN_OR_RETURN(ccw_sendrecv, Instruction::Create(
            Opcode::kSend,
            absl::StrCat(name_prefix,
                         "_mesh-1d_ccw_send_",
                         i),
            allgather_sub_ptr));
        ccw_sendrecv->AppendCommunicationGroup(
            {comm_group.at(ccw_peer_index)});
        ccw_sendrecv->SetBytesOut(comm_size / comm_group.size());
      } else if (!ccw_send_flag && ccw_recv_flag) {
        // If only receives from CCW but doesn't send there
        ASSIGN_OR_RETURN(ccw_sendrecv, Instruction::Create(
            Opcode::kRecv,
            absl::StrCat(name_prefix,
                         "_mesh-1d_ccw_recv_",
                         i),
            allgather_sub_ptr));
        ccw_sendrecv->AppendCommunicationGroup(
            {comm_group.at(ccw_peer_index)});
        ccw_sendrecv->SetBytesIn(comm_size / comm_group.size());
      } else {
        // If both sends and receives to/from CCW
        ASSIGN_OR_RETURN(ccw_sendrecv, Instruction::Create(
            Opcode::kSendRecv,
            absl::StrCat(name_prefix,
                         "_mesh-1d_ccw_sendrecv_",
                         i),
            allgather_sub_ptr));
        ccw_sendrecv->AppendCommunicationGroup(ccw_comm_group);
        ccw_sendrecv->SetBytesIn(comm_size / comm_group.size());
        ccw_sendrecv->SetBytesOut(comm_size / comm_group.size());
      }
      // We need to use previous communication as an operand to the next one
      if (previous_instruction != nullptr) {
        ccw_sendrecv->AddOperand(previous_instruction);
      }
    }
    if (cw_send_flag || cw_recv_flag) {
      // Create CW communication instruction
      if (cw_send_flag && !cw_recv_flag) {
        // If only sends CW but doesn't receive from there
        ASSIGN_OR_RETURN(cw_sendrecv, Instruction::Create(
            Opcode::kSend,
            absl::StrCat(name_prefix,
                         "_mesh-1d_cw_send_",
                         i),
            allgather_sub_ptr));
        cw_sendrecv->AppendCommunicationGroup(
            {comm_group.at(cw_peer_index)});
        cw_sendrecv->SetBytesOut(comm_size / comm_group.size());
      } else if (!cw_send_flag && cw_recv_flag) {
        // If only receives from CW but doesn't send there
        ASSIGN_OR_RETURN(cw_sendrecv, Instruction::Create(
            Opcode::kRecv,
            absl::StrCat(name_prefix,
                         "_mesh-1d_cw_recv_",
                         i),
            allgather_sub_ptr));
        cw_sendrecv->AppendCommunicationGroup(
            {comm_group.at(cw_peer_index)});
        cw_sendrecv->SetBytesIn(comm_size / comm_group.size());
      } else {
        // If both sends and receives to/from CW
        ASSIGN_OR_RETURN(cw_sendrecv, Instruction::Create(
            Opcode::kSendRecv,
            absl::StrCat(name_prefix,
                         "_mesh-1d_cw_sendrecv_",
                         i),
            allgather_sub_ptr));
        cw_sendrecv->AppendCommunicationGroup(cw_comm_group);
        cw_sendrecv->SetBytesIn(comm_size / comm_group.size());
        cw_sendrecv->SetBytesOut(comm_size / comm_group.size());
      }
      // We need to use previous communication as an operand to the next one
      if (previous_instruction != nullptr) {
        cw_sendrecv->AddOperand(previous_instruction);
      }
    }
    // Figure out which instruction should be the route of the subroutine
    if ((ccw_send_flag || ccw_recv_flag) && (cw_send_flag || cw_recv_flag)) {
      // Create root instruction with dependincies on send instructions
      ASSIGN_OR_RETURN(auto root, Instruction::Create(
          Opcode::kNull,
          absl::StrCat(name_prefix,
                       "_mesh-1d_root_",
                       i),
        allgather_sub_ptr,
        /*is_root*/ true));
      root->AddOperand(cw_sendrecv);
      root->AddOperand(ccw_sendrecv);
      previous_instruction = root;
    } else if ((ccw_send_flag || ccw_recv_flag) &&
               !(cw_send_flag || cw_recv_flag)) {
      previous_instruction = ccw_sendrecv;
    } else if (!(ccw_send_flag || ccw_recv_flag) &&
               (cw_send_flag || cw_recv_flag)) {
      previous_instruction = cw_sendrecv;
    } else {
      break;
    }
  }
  // Set root instruction for allgather subroutine
  RETURN_IF_ERROR(allgather_subroutine->SetRootInstruction(
      previous_instruction));
  return allgather_subroutine;
}

bool Mesh1dAllGatherTranslator::WillSendCw(int64_t processor_index,
                                           int64_t iteration,
                                           int64_t communication_size) {
    return ((processor_index - iteration) >= 0) &&
        (processor_index + 1 < communication_size);
}

bool Mesh1dAllGatherTranslator::WillSendCcw(int64_t processor_index,
                                            int64_t iteration,
                                            int64_t communication_size) {
    return ((processor_index + iteration + 1) <= communication_size) &&
        (processor_index > 0);
}

registerWithObjectFactory(
    "mesh-1d",
    AllGatherTranslator,
    Mesh1dAllGatherTranslator,
    nlohmann::json);

}  // namespace paragraph
