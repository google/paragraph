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
#include "paragraph/translation/allreduce/unidir_ring_allreduce_translator.h"

#include <memory>
#include <string>
#include <utility>

#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "gtest/gtest.h"
#include "paragraph/shim/test_macros.h"
#include "paragraph/translation/translation_map.h"

// Tests expanding uni-directional ring all-reduce
TEST(UniDirRingAllReduce, NoBarrier) {
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
        "algorithm": "unidir-ring"
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
  name: "all-reduce_unidir-ring"
  subroutine_root_id: 18
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "all-reduce_unidir-ring_reduce-scatter"
    opcode: "reduce-scatter"
    instruction_id: 7
    bytes_out: 48
    communication_groups {
      group_ids: 0
      group_ids: 1
      group_ids: 2
    }
    inner_subroutines {
      name: "all-reduce_unidir-ring_reduce-scatter_unidir-ring"
      subroutine_root_id: 14
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-reduce_unidir-ring_reduce-scatter_unidir-ring_sendrecv_1"
        opcode: "sendrecv"
        instruction_id: 8
        bytes_in: 16
        bytes_out: 16
        communication_groups {
          group_ids: 1
          group_ids: 0
        }
      }
      instructions {
        name: "all-reduce_unidir-ring_reduce-scatter_unidir-ring_reduction_1"
        opcode: "call"
        instruction_id: 9
        operand_ids: 8
        inner_subroutines {
          name: "reduction_subroutine_phase_1"
          subroutine_root_id: 12
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "op1_phase_1"
            opcode: "delay"
            instruction_id: 10
            bytes_out: 16
          }
          instructions {
            name: "op2_phase_1"
            opcode: "delay"
            instruction_id: 11
            bytes_out: 16
          }
          instructions {
            name: "sum_phase_1"
            opcode: "delay"
            instruction_id: 12
            ops: 32
            operand_ids: 10
            operand_ids: 11
          }
        }
      }
      instructions {
        name: "all-reduce_unidir-ring_reduce-scatter_unidir-ring_sendrecv_2"
        opcode: "sendrecv"
        instruction_id: 13
        bytes_in: 16
        bytes_out: 16
        communication_groups {
          group_ids: 1
          group_ids: 0
        }
        operand_ids: 9
      }
      instructions {
        name: "all-reduce_unidir-ring_reduce-scatter_unidir-ring_reduction_2"
        opcode: "call"
        instruction_id: 14
        operand_ids: 13
        inner_subroutines {
          name: "reduction_subroutine_phase_2"
          subroutine_root_id: 17
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "op1_phase_2"
            opcode: "delay"
            instruction_id: 15
            bytes_out: 16
          }
          instructions {
            name: "op2_phase_2"
            opcode: "delay"
            instruction_id: 16
            bytes_out: 16
          }
          instructions {
            name: "sum_phase_2"
            opcode: "delay"
            instruction_id: 17
            ops: 32
            operand_ids: 15
            operand_ids: 16
          }
        }
      }
    }
  }
  instructions {
    name: "all-reduce_unidir-ring_all-gather"
    opcode: "all-gather"
    instruction_id: 18
    bytes_out: 48
    communication_groups {
      group_ids: 0
      group_ids: 1
      group_ids: 2
    }
    operand_ids: 7
    inner_subroutines {
      name: "all-reduce_unidir-ring_all-gather_unidir-ring"
      subroutine_root_id: 20
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-reduce_unidir-ring_all-gather_unidir-ring_sendrecv_1"
        opcode: "sendrecv"
        instruction_id: 19
        bytes_in: 16
        bytes_out: 16
        communication_groups {
          group_ids: 1
          group_ids: 0
        }
      }
      instructions {
        name: "all-reduce_unidir-ring_all-gather_unidir-ring_sendrecv_2"
        opcode: "sendrecv"
        instruction_id: 20
        bytes_in: 16
        bytes_out: 16
        communication_groups {
          group_ids: 1
          group_ids: 0
        }
        operand_ids: 19
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

// Tests expanding uni-directional ring all-reduce with barrier
TEST(UniDirRingAllReduce, WithBarrier) {
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
        "algorithm": "unidir-ring",
        "barrier": {
          "algorithm": "centralized"
        }
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
  name: "all-reduce_unidir-ring"
  subroutine_root_id: 21
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "all-reduce_unidir-ring_barrier"
    opcode: "barrier"
    instruction_id: 7
    communication_groups {
      group_ids: 0
      group_ids: 1
      group_ids: 2
    }
    inner_subroutines {
      name: "all-reduce_unidir-ring_barrier_centralized"
      subroutine_root_id: 9
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-reduce_unidir-ring_barrier_centralized_send_to_0"
        opcode: "send"
        instruction_id: 8
        communication_groups {
          group_ids: 0
        }
      }
      instructions {
        name: "all-reduce_unidir-ring_barrier_centralized_recv_from_0"
        opcode: "recv"
        instruction_id: 9
        communication_groups {
          group_ids: 0
        }
        operand_ids: 8
      }
    }
  }
  instructions {
    name: "all-reduce_unidir-ring_reduce-scatter"
    opcode: "reduce-scatter"
    instruction_id: 10
    bytes_out: 48
    communication_groups {
      group_ids: 0
      group_ids: 1
      group_ids: 2
    }
    operand_ids: 7
    inner_subroutines {
      name: "all-reduce_unidir-ring_reduce-scatter_unidir-ring"
      subroutine_root_id: 17
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-reduce_unidir-ring_reduce-scatter_unidir-ring_sendrecv_1"
        opcode: "sendrecv"
        instruction_id: 11
        bytes_in: 16
        bytes_out: 16
        communication_groups {
          group_ids: 1
          group_ids: 0
        }
      }
      instructions {
        name: "all-reduce_unidir-ring_reduce-scatter_unidir-ring_reduction_1"
        opcode: "call"
        instruction_id: 12
        operand_ids: 11
        inner_subroutines {
          name: "reduction_subroutine_phase_1"
          subroutine_root_id: 15
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "op1_phase_1"
            opcode: "delay"
            instruction_id: 13
            bytes_out: 16
          }
          instructions {
            name: "op2_phase_1"
            opcode: "delay"
            instruction_id: 14
            bytes_out: 16
          }
          instructions {
            name: "sum_phase_1"
            opcode: "delay"
            instruction_id: 15
            ops: 32
            operand_ids: 13
            operand_ids: 14
          }
        }
      }
      instructions {
        name: "all-reduce_unidir-ring_reduce-scatter_unidir-ring_sendrecv_2"
        opcode: "sendrecv"
        instruction_id: 16
        bytes_in: 16
        bytes_out: 16
        communication_groups {
          group_ids: 1
          group_ids: 0
        }
        operand_ids: 12
      }
      instructions {
        name: "all-reduce_unidir-ring_reduce-scatter_unidir-ring_reduction_2"
        opcode: "call"
        instruction_id: 17
        operand_ids: 16
        inner_subroutines {
          name: "reduction_subroutine_phase_2"
          subroutine_root_id: 20
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "op1_phase_2"
            opcode: "delay"
            instruction_id: 18
            bytes_out: 16
          }
          instructions {
            name: "op2_phase_2"
            opcode: "delay"
            instruction_id: 19
            bytes_out: 16
          }
          instructions {
            name: "sum_phase_2"
            opcode: "delay"
            instruction_id: 20
            ops: 32
            operand_ids: 18
            operand_ids: 19
          }
        }
      }
    }
  }
  instructions {
    name: "all-reduce_unidir-ring_all-gather"
    opcode: "all-gather"
    instruction_id: 21
    bytes_out: 48
    communication_groups {
      group_ids: 0
      group_ids: 1
      group_ids: 2
    }
    operand_ids: 10
    inner_subroutines {
      name: "all-reduce_unidir-ring_all-gather_unidir-ring"
      subroutine_root_id: 23
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-reduce_unidir-ring_all-gather_unidir-ring_sendrecv_1"
        opcode: "sendrecv"
        instruction_id: 22
        bytes_in: 16
        bytes_out: 16
        communication_groups {
          group_ids: 1
          group_ids: 0
        }
      }
      instructions {
        name: "all-reduce_unidir-ring_all-gather_unidir-ring_sendrecv_2"
        opcode: "sendrecv"
        instruction_id: 23
        bytes_in: 16
        bytes_out: 16
        communication_groups {
          group_ids: 1
          group_ids: 0
        }
        operand_ids: 22
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
