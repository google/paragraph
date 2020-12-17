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
#include "paragraph/translation/allgather/unidir_ring_allgather_translator.h"

#include <memory>
#include <string>
#include <utility>

#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "gtest/gtest.h"
#include "paragraph/shim/test_macros.h"
#include "paragraph/translation/translation_map.h"

// Tests expanding uni-directional ring all-gather
TEST(UniDirRingAllGather, NoBarrier) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 2);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  sub_ptr->SetId(3);
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "first_instruction", sub_ptr));
  instr_1->SetOps(4);

  ASSERT_OK_AND_ASSIGN(auto allgather, paragraph::Instruction::Create(
      paragraph::Opcode::kAllGather, "all-gather", sub_ptr));
  allgather->SetBytesOut(48);
  paragraph::CommunicationGroup allgather_group = {0, 1, 2};
  allgather->AppendCommunicationGroup(allgather_group);

  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "last_instruction", sub_ptr, true));
  instr_3->SetOps(4);

  nlohmann::json config = R"(
    {
      "all-gather": {
        "algorithm": "unidir-ring"
      }
    }
  )"_json;

  ASSERT_OK_AND_ASSIGN(auto translators, paragraph::CreateTranslators(
      paragraph::TranslatorType::kCollective, config));
  EXPECT_OK(translators["all-gather"]->Translate(allgather));

  paragraph::InstructionProto allgather_proto;
  std::string allgather_str =
      R"proto(
name: "all-gather"
opcode: "all-gather"
instruction_id: 2
bytes_out: 48
communication_groups {
  group_ids: 0
  group_ids: 1
  group_ids: 2
}
inner_subroutines {
  name: "all-gather_unidir-ring"
  subroutine_root_id: 5
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "all-gather_unidir-ring_sendrecv_1"
    opcode: "sendrecv"
    instruction_id: 4
    bytes_in: 16
    bytes_out: 16
    communication_groups {
      group_ids: 1
      group_ids: 0
    }
  }
  instructions {
    name: "all-gather_unidir-ring_sendrecv_2"
    opcode: "sendrecv"
    instruction_id: 5
    bytes_in: 16
    bytes_out: 16
    communication_groups {
      group_ids: 1
      group_ids: 0
    }
    operand_ids: 4
  }
}
      )proto";
  google::protobuf::TextFormat::ParseFromString(allgather_str,
                                                &allgather_proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      allgather->ToProto().value(), allgather_proto));
}

// Tests expanding uni-directional ring all-gather with barrier
TEST(UniDirRingAllGather, WithBarrier) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 2);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  sub_ptr->SetId(3);
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "first_instruction", sub_ptr));
  instr_1->SetOps(4);

  ASSERT_OK_AND_ASSIGN(auto allgather, paragraph::Instruction::Create(
      paragraph::Opcode::kAllGather, "all-gather", sub_ptr));
  allgather->SetBytesOut(48);
  paragraph::CommunicationGroup allgather_group = {0, 1, 2};
  allgather->AppendCommunicationGroup(allgather_group);

  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "last_instruction", sub_ptr, true));
  instr_3->SetOps(4);

  nlohmann::json config = R"(
    {
      "all-gather": {
        "algorithm": "unidir-ring",
        "barrier": {
          "algorithm": "centralized"
        }
      }
    }
  )"_json;

  ASSERT_OK_AND_ASSIGN(auto translators, paragraph::CreateTranslators(
      paragraph::TranslatorType::kCollective, config));
  EXPECT_OK(translators["all-gather"]->Translate(allgather));

  paragraph::InstructionProto allgather_proto;
  std::string allgather_str =
      R"proto(
name: "all-gather"
opcode: "all-gather"
instruction_id: 2
bytes_out: 48
communication_groups {
  group_ids: 0
  group_ids: 1
  group_ids: 2
}
inner_subroutines {
  name: "all-gather_unidir-ring"
  subroutine_root_id: 8
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "all-gather_unidir-ring_barrier"
    opcode: "barrier"
    instruction_id: 4
    communication_groups {
      group_ids: 0
      group_ids: 1
      group_ids: 2
    }
    inner_subroutines {
      name: "all-gather_unidir-ring_barrier_centralized"
      subroutine_root_id: 6
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-gather_unidir-ring_barrier_centralized_send_to_0"
        opcode: "send"
        instruction_id: 5
        communication_groups {
          group_ids: 0
        }
      }
      instructions {
        name: "all-gather_unidir-ring_barrier_centralized_recv_from_0"
        opcode: "recv"
        instruction_id: 6
        communication_groups {
          group_ids: 0
        }
        operand_ids: 5
      }
    }
  }
  instructions {
    name: "all-gather_unidir-ring_sendrecv_1"
    opcode: "sendrecv"
    instruction_id: 7
    bytes_in: 16
    bytes_out: 16
    communication_groups {
      group_ids: 1
      group_ids: 0
    }
    operand_ids: 4
  }
  instructions {
    name: "all-gather_unidir-ring_sendrecv_2"
    opcode: "sendrecv"
    instruction_id: 8
    bytes_in: 16
    bytes_out: 16
    communication_groups {
      group_ids: 1
      group_ids: 0
    }
    operand_ids: 7
  }
}
      )proto";
  google::protobuf::TextFormat::ParseFromString(allgather_str,
                                                &allgather_proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      allgather->ToProto().value(), allgather_proto));
}

// Tests expanding uni-directional ring all-gather
TEST(UniDirRingAllGather, InconsecutiveProcessors) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 2);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  sub_ptr->SetId(3);
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "first_instruction", sub_ptr));
  instr_1->SetOps(4);

  ASSERT_OK_AND_ASSIGN(auto allgather, paragraph::Instruction::Create(
      paragraph::Opcode::kAllGather, "all-gather", sub_ptr));
  allgather->SetBytesOut(48);
  paragraph::CommunicationGroup allgather_group = {0, 2, 4};
  allgather->AppendCommunicationGroup(allgather_group);

  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "last_instruction", sub_ptr, true));
  instr_3->SetOps(4);

  nlohmann::json config = R"(
    {
      "all-gather": {
        "algorithm": "unidir-ring"
      }
    }
  )"_json;

  ASSERT_OK_AND_ASSIGN(auto translators, paragraph::CreateTranslators(
      paragraph::TranslatorType::kCollective, config));
  EXPECT_OK(translators["all-gather"]->Translate(allgather));

  paragraph::InstructionProto allgather_proto;
  std::string allgather_str =
      R"proto(
name: "all-gather"
opcode: "all-gather"
instruction_id: 2
bytes_out: 48
communication_groups {
  group_ids: 0
  group_ids: 2
  group_ids: 4
}
inner_subroutines {
  name: "all-gather_unidir-ring"
  subroutine_root_id: 5
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "all-gather_unidir-ring_sendrecv_1"
    opcode: "sendrecv"
    instruction_id: 4
    bytes_in: 16
    bytes_out: 16
    communication_groups {
      group_ids: 0
      group_ids: 4
    }
  }
  instructions {
    name: "all-gather_unidir-ring_sendrecv_2"
    opcode: "sendrecv"
    instruction_id: 5
    bytes_in: 16
    bytes_out: 16
    communication_groups {
      group_ids: 0
      group_ids: 4
    }
    operand_ids: 4
  }
}
      )proto";
  google::protobuf::TextFormat::ParseFromString(allgather_str,
                                                &allgather_proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      allgather->ToProto().value(), allgather_proto));
}
