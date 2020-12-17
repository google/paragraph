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
#include "paragraph/translation/send/push_send_translator.h"

#include <memory>
#include <string>

#include "factory/ObjectFactory.h"

namespace paragraph {

PushSendTranslator::PushSendTranslator(nlohmann::json config) {}

shim::StatusOr<std::unique_ptr<Subroutine>>
    PushSendTranslator::GetSubroutine(
        const std::string& name_prefix,
        Instruction* calling_instruction,
        int64_t processor_id,
        const CommunicationGroup& comm_group,
        double comm_size) const {
  auto graph = calling_instruction->GetGraph();
  auto send_subroutine = absl::make_unique<Subroutine>(
      absl::StrCat(name_prefix, "_push"), graph);
  auto send_sub_ptr = send_subroutine.get();
  ASSIGN_OR_RETURN(auto sendstart, Instruction::Create(
      Opcode::kSendStart,
      absl::StrCat(name_prefix,
                   "_sendstart"),
      send_sub_ptr));
  sendstart->AppendCommunicationGroup(comm_group);
  sendstart->SetBytesIn(comm_size);

  ASSIGN_OR_RETURN(auto senddone, Instruction::Create(
      Opcode::kSendDone,
      absl::StrCat(name_prefix,
                   "_senddone"),
      send_sub_ptr, true));
  senddone->AppendCommunicationGroup(comm_group);
  senddone->AddOperand(sendstart);
  senddone->BondWith(sendstart);
  return send_subroutine;
}

registerWithObjectFactory(
    "push",
    SendTranslator,
    PushSendTranslator,
    nlohmann::json);

}  // namespace paragraph
