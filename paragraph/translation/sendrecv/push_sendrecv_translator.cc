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
#include "paragraph/translation/sendrecv/push_sendrecv_translator.h"

#include <memory>
#include <string>

#include "factory/ObjectFactory.h"

namespace paragraph {

PushSendRecvTranslator::PushSendRecvTranslator(nlohmann::json config) {}

shim::StatusOr<std::unique_ptr<Subroutine>>
    PushSendRecvTranslator::GetSubroutine(
        const std::string& name_prefix,
        Instruction* calling_instruction,
        int64_t processor_id,
        const CommunicationGroup& comm_group,
        double comm_size_send,
        double comm_size_recv) const {
  auto graph = calling_instruction->GetGraph();
  auto sendrecv_subroutine = absl::make_unique<Subroutine>(
      absl::StrCat(name_prefix, "_push"), graph);
  auto sendrecv_sub_ptr = sendrecv_subroutine.get();
  RETURN_IF_FALSE(comm_group.size() == 2,
                  absl::InvalidArgumentError) << "Individualized SendRecv" <<
      "instruction should have exactly 2 peers in CommunicationGroup.";

  CommunicationGroup recv_comm_group = {comm_group.at(0)};
  ASSIGN_OR_RETURN(auto recvstart, Instruction::Create(
      Opcode::kRecvStart,
      absl::StrCat(name_prefix,
                   "_recvstart"),
      sendrecv_sub_ptr));
  recvstart->AppendCommunicationGroup(recv_comm_group);

  CommunicationGroup send_comm_group = {comm_group.at(1)};
  ASSIGN_OR_RETURN(auto sendstart, Instruction::Create(
      Opcode::kSendStart,
      absl::StrCat(name_prefix,
                   "_sendstart"),
      sendrecv_sub_ptr));
  sendstart->AppendCommunicationGroup(send_comm_group);
  sendstart->SetBytesIn(comm_size_send);
  sendstart->AddOperand(recvstart);

  ASSIGN_OR_RETURN(auto senddone, Instruction::Create(
      Opcode::kSendDone,
      absl::StrCat(name_prefix,
                   "_senddone"),
      sendrecv_sub_ptr));
  senddone->AppendCommunicationGroup(send_comm_group);
  senddone->AddOperand(sendstart);
  senddone->BondWith(sendstart);

  ASSIGN_OR_RETURN(auto recvdone, Instruction::Create(
      Opcode::kRecvDone,
      absl::StrCat(name_prefix,
                   "_recvdone"),
      sendrecv_sub_ptr));
  recvdone->AppendCommunicationGroup(recv_comm_group);
  recvdone->SetBytesOut(comm_size_recv);
  recvdone->AddOperand(recvstart);
  recvdone->AddOperand(sendstart);
  recvdone->BondWith(recvstart);

  ASSIGN_OR_RETURN(auto root, Instruction::Create(
      Opcode::kNull,
      absl::StrCat(name_prefix,
                   "_root"),
      sendrecv_sub_ptr, true));
  root->AddOperand(recvdone);
  root->AddOperand(senddone);
  return sendrecv_subroutine;
}

registerWithObjectFactory(
    "push",
    SendRecvTranslator,
    PushSendRecvTranslator,
    nlohmann::json);

}  // namespace paragraph
