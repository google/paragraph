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
#include "paragraph/translation/recv/push_recv_translator.h"

#include <memory>
#include <string>

#include "factory/ObjectFactory.h"

namespace paragraph {

PushRecvTranslator::PushRecvTranslator(nlohmann::json config) {}

shim::StatusOr<std::unique_ptr<Subroutine>>
    PushRecvTranslator::GetSubroutine(
        const std::string& name_prefix,
        Instruction* calling_instruction,
        int64_t processor_id,
        const CommunicationGroup& comm_group,
        double comm_size) const {
  auto graph = calling_instruction->GetGraph();
  auto recv_subroutine = absl::make_unique<Subroutine>(
      absl::StrCat(name_prefix, "_push"), graph);
  auto recv_sub_ptr = recv_subroutine.get();
  ASSIGN_OR_RETURN(auto recvstart, Instruction::Create(
      Opcode::kRecvStart,
      absl::StrCat(name_prefix,
                   "_recvstart"),
      recv_sub_ptr));
  recvstart->AppendCommunicationGroup(comm_group);

  ASSIGN_OR_RETURN(auto recvdone, Instruction::Create(
      Opcode::kRecvDone,
      absl::StrCat(name_prefix,
                   "_recvdone"),
      recv_sub_ptr, true));
  recvdone->AppendCommunicationGroup(comm_group);
  recvdone->SetBytesOut(comm_size);
  recvdone->AddOperand(recvstart);
  recvdone->BondWith(recvstart);
  return recv_subroutine;
}

registerWithObjectFactory(
    "push",
    RecvTranslator,
    PushRecvTranslator,
    nlohmann::json);

}  // namespace paragraph
