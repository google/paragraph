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
#include "paragraph/translation/barrier/centralized_barrier_translator.h"

#include <memory>
#include <string>
#include <utility>

#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "gtest/gtest.h"
#include "paragraph/shim/test_macros.h"
#include "paragraph/translation/translation_map.h"

// Tests expanding centralized barrier for non-coordinator processor
TEST(CentralizedBarrier, NonCoordinator) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 2);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "first_instruction", sub_ptr));
  instr_1->SetOps(4);

  ASSERT_OK_AND_ASSIGN(auto barrier_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kBarrier, "barrier", sub_ptr));
  paragraph::CommunicationGroup barrier_group = {0, 1, 2};
  barrier_instr->AppendCommunicationGroup(barrier_group);

  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "last_instruction", sub_ptr, true));
  instr_3->SetOps(4);

  nlohmann::json config = R"(
    {
      "barrier": {
        "algorithm": "centralized"
      }
    }
  )"_json;

  ASSERT_OK_AND_ASSIGN(auto translators, paragraph::CreateTranslators(
      paragraph::TranslatorType::kCollective, config));
  EXPECT_OK(translators["barrier"]->Translate(barrier_instr));

  paragraph::InstructionProto barrier_proto;
  std::string barrier_str =
      R"proto(
name: "barrier"
opcode: "barrier"
instruction_id: 2
communication_groups {
  group_ids: 0
  group_ids: 1
  group_ids: 2
}
inner_subroutines {
  name: "barrier_centralized"
  subroutine_root_id: 5
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "barrier_centralized_send_to_0"
    opcode: "send"
    instruction_id: 4
    communication_groups {
      group_ids: 0
    }
  }
  instructions {
    name: "barrier_centralized_recv_from_0"
    opcode: "recv"
    instruction_id: 5
    communication_groups {
      group_ids: 0
    }
    operand_ids: 4
  }
}
      )proto";
  google::protobuf::TextFormat::ParseFromString(barrier_str, &barrier_proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      barrier_instr->ToProto().value(), barrier_proto));
}

// Tests expanding centralized barrier for coordinator processor
TEST(CentralizedBarrier, Coordinator) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 0);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  sub_ptr->SetId(3);
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "first_instruction", sub_ptr));
  instr_1->SetOps(4);

  ASSERT_OK_AND_ASSIGN(auto barrier_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kBarrier, "barrier", sub_ptr));
  paragraph::CommunicationGroup barrier_group = {0, 1, 2};
  barrier_instr->AppendCommunicationGroup(barrier_group);

  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "last_instruction", sub_ptr, true));
  instr_3->SetOps(4);

  nlohmann::json config = R"(
    {
      "barrier": {
        "algorithm": "centralized"
      }
    }
  )"_json;

  ASSERT_OK_AND_ASSIGN(auto translators, paragraph::CreateTranslators(
      paragraph::TranslatorType::kCollective, config));
  EXPECT_OK(translators["barrier"]->Translate(barrier_instr));

  paragraph::InstructionProto barrier_proto;
  std::string barrier_str =
      R"proto(
name: "barrier"
opcode: "barrier"
instruction_id: 2
communication_groups {
  group_ids: 0
  group_ids: 1
  group_ids: 2
}
inner_subroutines {
  name: "barrier_centralized"
  subroutine_root_id: 8
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "barrier_centralized_coordinator_recv_from_1"
    opcode: "recv"
    instruction_id: 4
    communication_groups {
      group_ids: 1
    }
  }
  instructions {
    name: "barrier_centralized_coordinator_recv_from_2"
    opcode: "recv"
    instruction_id: 5
    communication_groups {
      group_ids: 2
    }
  }
  instructions {
    name: "barrier_centralized_coordinator_send_to_1"
    opcode: "send"
    instruction_id: 6
    operand_ids: 4
    operand_ids: 5
    communication_groups {
      group_ids: 1
    }
  }
  instructions {
    name: "barrier_centralized_coordinator_send_to_2"
    opcode: "send"
    instruction_id: 7
    operand_ids: 4
    operand_ids: 5
    communication_groups {
      group_ids: 2
    }
  }
  instructions {
    name: "barrier_centralized_root_0"
    opcode: "null"
    instruction_id: 8
    operand_ids: 6
    operand_ids: 7
  }
}
      )proto";
  google::protobuf::TextFormat::ParseFromString(barrier_str, &barrier_proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      barrier_instr->ToProto().value(), barrier_proto));
}
