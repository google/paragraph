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
#include <utility>

#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "gtest/gtest.h"
#include "paragraph/shim/test_macros.h"
#include "paragraph/translation/translation_map.h"

// Tests expanding sendrecv instruction for a simple push-based protocol
TEST(SendRecvTranslator, Push) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 2);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  sub_ptr->SetId(3);
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "first_instruction", sub_ptr));
  instr_1->SetOps(4);

  ASSERT_OK_AND_ASSIGN(auto sendrecv, paragraph::Instruction::Create(
      paragraph::Opcode::kSendRecv, "sendrecv", sub_ptr));
  paragraph::CommunicationGroup sendrecv_group = {0, 1};
  sendrecv->AppendCommunicationGroup(sendrecv_group);
  sendrecv->SetBytesIn(16);
  sendrecv->SetBytesOut(32);

  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "last_instruction", sub_ptr, true));
  instr_3->SetOps(4);

  nlohmann::json config = R"(
    {
      "sendrecv": {
        "algorithm": "push"
      }
    }
  )"_json;
  ASSERT_OK_AND_ASSIGN(auto translators, paragraph::CreateTranslators(
      paragraph::TranslatorType::kProtocol, config));
  EXPECT_OK(translators["sendrecv"]->Translate(sendrecv));

  paragraph::InstructionProto sendrecv_proto;
  std::string sendrecv_str =
      R"proto(
name: "sendrecv"
opcode: "sendrecv"
instruction_id: 2
bytes_in: 16
bytes_out: 32
communication_groups {
  group_ids: 0
  group_ids: 1
}
inner_subroutines {
  name: "sendrecv_push"
  subroutine_root_id: 8
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "sendrecv_recvstart"
    opcode: "recv-start"
    instruction_id: 4
    bonded_instruction_id: 7
    communication_groups {
      group_ids: 0
    }
  }
  instructions {
    name: "sendrecv_sendstart"
    opcode: "send-start"
    instruction_id: 5
    bonded_instruction_id: 6
    bytes_in: 16
    communication_groups {
      group_ids: 1
    }
    operand_ids: 4
  }
  instructions {
    name: "sendrecv_senddone"
    opcode: "send-done"
    instruction_id: 6
    bonded_instruction_id: 5
    communication_groups {
      group_ids: 1
    }
    operand_ids: 5
  }
  instructions {
    name: "sendrecv_recvdone"
    opcode: "recv-done"
    instruction_id: 7
    bonded_instruction_id: 4
    bytes_out: 32
    communication_groups {
      group_ids: 0
    }
    operand_ids: 4
    operand_ids: 5
  }
  instructions {
    name: "sendrecv_root"
    opcode: "null"
    instruction_id: 8
    operand_ids: 7
    operand_ids: 6
  }
}
      )proto";
  google::protobuf::TextFormat::ParseFromString(sendrecv_str, &sendrecv_proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      sendrecv->ToProto().value(), sendrecv_proto));
}
