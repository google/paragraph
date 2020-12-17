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
#include <utility>

#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "gtest/gtest.h"
#include "paragraph/shim/test_macros.h"
#include "paragraph/translation/translation_map.h"

// Tests expanding recv instruction for a simple push-based protocol
TEST(RecvTranslator, Push) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 2);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  sub_ptr->SetId(3);
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "first_instruction", sub_ptr));
  instr_1->SetOps(4);

  ASSERT_OK_AND_ASSIGN(auto recv, paragraph::Instruction::Create(
      paragraph::Opcode::kRecv, "recv", sub_ptr));
  paragraph::CommunicationGroup recv_group = {0};
  recv->AppendCommunicationGroup(recv_group);
  recv->SetBytesOut(16);

  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "last_instruction", sub_ptr, true));
  instr_3->SetOps(4);

  nlohmann::json config = R"(
    {
      "recv": {
        "algorithm": "push"
      }
    }
  )"_json;
  ASSERT_OK_AND_ASSIGN(auto translators, paragraph::CreateTranslators(
      paragraph::TranslatorType::kProtocol, config));
  EXPECT_OK(translators["recv"]->Translate(recv));

  paragraph::InstructionProto recv_proto;
  std::string recv_str =
      R"proto(
name: "recv"
opcode: "recv"
instruction_id: 2
bytes_out: 16
communication_groups {
  group_ids: 0
}
inner_subroutines {
  name: "recv_push"
  subroutine_root_id: 5
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "recv_recvstart"
    opcode: "recv-start"
    instruction_id: 4
    bonded_instruction_id: 5
    communication_groups {
      group_ids: 0
    }
  }
  instructions {
    name: "recv_recvdone"
    opcode: "recv-done"
    instruction_id: 5
    bonded_instruction_id: 4
    bytes_out: 16
    communication_groups {
      group_ids: 0
    }
    operand_ids: 4
  }
}
      )proto";
  google::protobuf::TextFormat::ParseFromString(recv_str, &recv_proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      recv->ToProto().value(), recv_proto));
}
