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

#include <memory>
#include <string>
#include <utility>

#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "gtest/gtest.h"
#include "paragraph/shim/test_macros.h"
#include "paragraph/translation/translation_map.h"

// Tests expanding dissemination all-reduce
TEST(DisseminationAllReduce, All) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 2);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  sub_ptr->SetId(3);
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "first_instruction", sub_ptr));
  instr_1->SetOps(4);

  ASSERT_OK_AND_ASSIGN(auto allreduce, paragraph::Instruction::Create(
      paragraph::Opcode::kAllReduce, "all-reduce", sub_ptr));
  allreduce->SetBytesOut(48);
  paragraph::CommunicationGroup allreduce_group = {0, 1, 2};
  allreduce->AppendCommunicationGroup(allreduce_group);

  auto reduction_sub = absl::make_unique<paragraph::Subroutine>(
      "reduction_subroutine", graph.get());
  auto reduction_ptr = reduction_sub.get();
  ASSERT_OK_AND_ASSIGN(auto op1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "op1", reduction_ptr));
  op1->SetBytesOut(48);
  ASSERT_OK_AND_ASSIGN(auto op2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "op2", reduction_ptr));
  op2->SetBytesOut(48);
  ASSERT_OK_AND_ASSIGN(auto sum_op, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "sum", reduction_ptr, true));
  sum_op->SetOps(96);
  sum_op->AddOperand(op1);
  sum_op->AddOperand(op2);
  allreduce->AppendInnerSubroutine(std::move(reduction_sub));

  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "last_instruction", sub_ptr, true));
  instr_3->SetOps(4);

  nlohmann::json config = R"(
    {
      "all-reduce": {
        "algorithm": "dissemination"
      }
    }
  )"_json;

  ASSERT_OK_AND_ASSIGN(auto translators, paragraph::CreateTranslators(
      paragraph::TranslatorType::kCollective, config));
  EXPECT_OK(translators["all-reduce"]->Translate(allreduce));

  paragraph::InstructionProto allreduce_proto;
  std::string allreduce_str =
      R"proto(
name: "all-reduce"
opcode: "all-reduce"
instruction_id: 2
bytes_out: 48
communication_groups {
  group_ids: 0
  group_ids: 1
  group_ids: 2
}
inner_subroutines {
  name: "all-reduce_dissemination"
  subroutine_root_id: 13
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "all-reduce_dissemination_sendrecv_0"
    opcode: "sendrecv"
    instruction_id: 7
    bytes_in: 48
    bytes_out: 48
    communication_groups {
      group_ids: 0
      group_ids: 1
    }
  }
  instructions {
    name: "all-reduce_dissemination_reduction_0"
    opcode: "call"
    instruction_id: 8
    operand_ids: 7
    inner_subroutines {
      name: "reduction_subroutine_iteration_0"
      subroutine_root_id: 11
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "op1_iteration_0"
        opcode: "delay"
        instruction_id: 9
        bytes_out: 48
      }
      instructions {
        name: "op2_iteration_0"
        opcode: "delay"
        instruction_id: 10
        bytes_out: 48
      }
      instructions {
        name: "sum_iteration_0"
        opcode: "delay"
        instruction_id: 11
        ops: 96
        operand_ids: 9
        operand_ids: 10
      }
    }
  }
  instructions {
    name: "all-reduce_dissemination_sendrecv_1"
    opcode: "sendrecv"
    instruction_id: 12
    bytes_in: 48
    bytes_out: 48
    communication_groups {
      group_ids: 1
      group_ids: 0
    }
    operand_ids: 8
  }
  instructions {
    name: "all-reduce_dissemination_reduction_1"
    opcode: "call"
    instruction_id: 13
    operand_ids: 12
    inner_subroutines {
      name: "reduction_subroutine_iteration_1"
      subroutine_root_id: 16
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "op1_iteration_1"
        opcode: "delay"
        instruction_id: 14
        bytes_out: 48
      }
      instructions {
        name: "op2_iteration_1"
        opcode: "delay"
        instruction_id: 15
        bytes_out: 48
      }
      instructions {
        name: "sum_iteration_1"
        opcode: "delay"
        instruction_id: 16
        ops: 96
        operand_ids: 14
        operand_ids: 15
      }
    }
  }
}
      )proto";
  google::protobuf::TextFormat::ParseFromString(allreduce_str,
                                                &allreduce_proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      allreduce->ToProto().value(), allreduce_proto));
}
