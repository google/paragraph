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
#include "paragraph/translation/reducescatter/mesh_2d_ring_reducescatter_translator.h"

#include <memory>
#include <string>
#include <utility>

#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "gtest/gtest.h"
#include "paragraph/shim/test_macros.h"
#include "paragraph/translation/translation_map.h"

// Tests expanding 2D-Mesh reduce-scatter
TEST(Mesh2dRingReduceScatter, NoBarrier) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "first_instruction", sub_ptr));
  instr_1->SetOps(4);

  ASSERT_OK_AND_ASSIGN(auto reducescatter,
                       paragraph::Instruction::Create(
      paragraph::Opcode::kReduceScatter, "reduce-scatter", sub_ptr));
  reducescatter->SetBytesOut(80);
  paragraph::CommunicationGroup reducescatter_group = {0, 1, 2, 3, 4, 5, 6, 7};
  reducescatter->AppendCommunicationGroup(reducescatter_group);

  auto reduction_sub = absl::make_unique<paragraph::Subroutine>(
      "reduction_subroutine", graph.get());
  auto reduction_ptr = reduction_sub.get();
  ASSERT_OK_AND_ASSIGN(auto op1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "op1", reduction_ptr));
  op1->SetBytesOut(80);
  ASSERT_OK_AND_ASSIGN(auto op2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "op2", reduction_ptr));
  op2->SetBytesOut(80);
  ASSERT_OK_AND_ASSIGN(auto sum_op, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "sum", reduction_ptr, true));
  sum_op->SetOps(160);
  sum_op->AddOperand(op1);
  sum_op->AddOperand(op2);
  reducescatter->AppendInnerSubroutine(std::move(reduction_sub));

  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "last_instruction", sub_ptr, true));
  instr_3->SetOps(4);

  nlohmann::json config = R"(
    {
      "reduce-scatter": {
        "algorithm": "mesh-2d-ring",
        "concentration": 2,
        "dimension_widths": [2, 2]
      }
    }
  )"_json;

  ASSERT_OK_AND_ASSIGN(auto translators, paragraph::CreateTranslators(
     paragraph::TranslatorType::kCollective, config));
  EXPECT_OK(translators["reduce-scatter"]->Translate(reducescatter));

  paragraph::InstructionProto reducescatter_proto;
  std::string reducescatter_str =
      R"proto(
name: "reduce-scatter"
opcode: "reduce-scatter"
instruction_id: 2
bytes_out: 80
communication_groups {
  group_ids: 0
  group_ids: 1
  group_ids: 2
  group_ids: 3
  group_ids: 4
  group_ids: 5
  group_ids: 6
  group_ids: 7
}
inner_subroutines {
  name: "reduce-scatter_mesh-2d-ring"
  subroutine_root_id: 7
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "reduce-scatter_mesh-2d-ring"
    opcode: "reduce-scatter"
    instruction_id: 7
    bytes_out: 80
    communication_groups {
      group_ids: 0
      group_ids: 1
      group_ids: 4
      group_ids: 5
      group_ids: 6
      group_ids: 7
      group_ids: 2
      group_ids: 3
    }
    inner_subroutines {
      name: "reduce-scatter_mesh-2d-ring_bidir-ring"
      subroutine_root_id: 80
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw"
        opcode: "reduce-scatter"
        instruction_id: 8
        bytes_out: 40
        communication_groups {
          group_ids: 0
          group_ids: 1
          group_ids: 4
          group_ids: 5
          group_ids: 6
          group_ids: 7
          group_ids: 2
          group_ids: 3
        }
        inner_subroutines {
          name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 40
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 9
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw_unidir-ring_reduction_1"
            opcode: "call"
            instruction_id: 10
            operand_ids: 9
            inner_subroutines {
              name: "reduction_subroutine_phase_1"
              subroutine_root_id: 13
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_1"
                opcode: "delay"
                instruction_id: 11
                bytes_out: 5
              }
              instructions {
                name: "op2_phase_1"
                opcode: "delay"
                instruction_id: 12
                bytes_out: 5
              }
              instructions {
                name: "sum_phase_1"
                opcode: "delay"
                instruction_id: 13
                ops: 10
                operand_ids: 11
                operand_ids: 12
              }
            }
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 14
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
            operand_ids: 10
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw_unidir-ring_reduction_2"
            opcode: "call"
            instruction_id: 15
            operand_ids: 14
            inner_subroutines {
              name: "reduction_subroutine_phase_2"
              subroutine_root_id: 18
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_2"
                opcode: "delay"
                instruction_id: 16
                bytes_out: 5
              }
              instructions {
                name: "op2_phase_2"
                opcode: "delay"
                instruction_id: 17
                bytes_out: 5
              }
              instructions {
                name: "sum_phase_2"
                opcode: "delay"
                instruction_id: 18
                ops: 10
                operand_ids: 16
                operand_ids: 17
              }
            }
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_3"
            opcode: "sendrecv"
            instruction_id: 19
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
            operand_ids: 15
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw_unidir-ring_reduction_3"
            opcode: "call"
            instruction_id: 20
            operand_ids: 19
            inner_subroutines {
              name: "reduction_subroutine_phase_3"
              subroutine_root_id: 23
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_3"
                opcode: "delay"
                instruction_id: 21
                bytes_out: 5
              }
              instructions {
                name: "op2_phase_3"
                opcode: "delay"
                instruction_id: 22
                bytes_out: 5
              }
              instructions {
                name: "sum_phase_3"
                opcode: "delay"
                instruction_id: 23
                ops: 10
                operand_ids: 21
                operand_ids: 22
              }
            }
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_4"
            opcode: "sendrecv"
            instruction_id: 24
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
            operand_ids: 20
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw_unidir-ring_reduction_4"
            opcode: "call"
            instruction_id: 25
            operand_ids: 24
            inner_subroutines {
              name: "reduction_subroutine_phase_4"
              subroutine_root_id: 28
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_4"
                opcode: "delay"
                instruction_id: 26
                bytes_out: 5
              }
              instructions {
                name: "op2_phase_4"
                opcode: "delay"
                instruction_id: 27
                bytes_out: 5
              }
              instructions {
                name: "sum_phase_4"
                opcode: "delay"
                instruction_id: 28
                ops: 10
                operand_ids: 26
                operand_ids: 27
              }
            }
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_5"
            opcode: "sendrecv"
            instruction_id: 29
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
            operand_ids: 25
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw_unidir-ring_reduction_5"
            opcode: "call"
            instruction_id: 30
            operand_ids: 29
            inner_subroutines {
              name: "reduction_subroutine_phase_5"
              subroutine_root_id: 33
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_5"
                opcode: "delay"
                instruction_id: 31
                bytes_out: 5
              }
              instructions {
                name: "op2_phase_5"
                opcode: "delay"
                instruction_id: 32
                bytes_out: 5
              }
              instructions {
                name: "sum_phase_5"
                opcode: "delay"
                instruction_id: 33
                ops: 10
                operand_ids: 31
                operand_ids: 32
              }
            }
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_6"
            opcode: "sendrecv"
            instruction_id: 34
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
            operand_ids: 30
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw_unidir-ring_reduction_6"
            opcode: "call"
            instruction_id: 35
            operand_ids: 34
            inner_subroutines {
              name: "reduction_subroutine_phase_6"
              subroutine_root_id: 38
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_6"
                opcode: "delay"
                instruction_id: 36
                bytes_out: 5
              }
              instructions {
                name: "op2_phase_6"
                opcode: "delay"
                instruction_id: 37
                bytes_out: 5
              }
              instructions {
                name: "sum_phase_6"
                opcode: "delay"
                instruction_id: 38
                ops: 10
                operand_ids: 36
                operand_ids: 37
              }
            }
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_7"
            opcode: "sendrecv"
            instruction_id: 39
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
            operand_ids: 35
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw_unidir-ring_reduction_7"
            opcode: "call"
            instruction_id: 40
            operand_ids: 39
            inner_subroutines {
              name: "reduction_subroutine_phase_7"
              subroutine_root_id: 43
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_7"
                opcode: "delay"
                instruction_id: 41
                bytes_out: 5
              }
              instructions {
                name: "op2_phase_7"
                opcode: "delay"
                instruction_id: 42
                bytes_out: 5
              }
              instructions {
                name: "sum_phase_7"
                opcode: "delay"
                instruction_id: 43
                ops: 10
                operand_ids: 41
                operand_ids: 42
              }
            }
          }
        }
      }
      instructions {
        name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw"
        opcode: "reduce-scatter"
        instruction_id: 44
        bytes_out: 40
        communication_groups {
          group_ids: 3
          group_ids: 2
          group_ids: 7
          group_ids: 6
          group_ids: 5
          group_ids: 4
          group_ids: 1
          group_ids: 0
        }
        inner_subroutines {
          name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 76
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 45
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw_unidir-ring_reduction_1"
            opcode: "call"
            instruction_id: 46
            operand_ids: 45
            inner_subroutines {
              name: "reduction_subroutine_phase_1"
              subroutine_root_id: 49
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_1"
                opcode: "delay"
                instruction_id: 47
                bytes_out: 5
              }
              instructions {
                name: "op2_phase_1"
                opcode: "delay"
                instruction_id: 48
                bytes_out: 5
              }
              instructions {
                name: "sum_phase_1"
                opcode: "delay"
                instruction_id: 49
                ops: 10
                operand_ids: 47
                operand_ids: 48
              }
            }
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 50
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
            operand_ids: 46
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw_unidir-ring_reduction_2"
            opcode: "call"
            instruction_id: 51
            operand_ids: 50
            inner_subroutines {
              name: "reduction_subroutine_phase_2"
              subroutine_root_id: 54
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_2"
                opcode: "delay"
                instruction_id: 52
                bytes_out: 5
              }
              instructions {
                name: "op2_phase_2"
                opcode: "delay"
                instruction_id: 53
                bytes_out: 5
              }
              instructions {
                name: "sum_phase_2"
                opcode: "delay"
                instruction_id: 54
                ops: 10
                operand_ids: 52
                operand_ids: 53
              }
            }
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_3"
            opcode: "sendrecv"
            instruction_id: 55
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
            operand_ids: 51
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw_unidir-ring_reduction_3"
            opcode: "call"
            instruction_id: 56
            operand_ids: 55
            inner_subroutines {
              name: "reduction_subroutine_phase_3"
              subroutine_root_id: 59
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_3"
                opcode: "delay"
                instruction_id: 57
                bytes_out: 5
              }
              instructions {
                name: "op2_phase_3"
                opcode: "delay"
                instruction_id: 58
                bytes_out: 5
              }
              instructions {
                name: "sum_phase_3"
                opcode: "delay"
                instruction_id: 59
                ops: 10
                operand_ids: 57
                operand_ids: 58
              }
            }
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_4"
            opcode: "sendrecv"
            instruction_id: 60
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
            operand_ids: 56
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw_unidir-ring_reduction_4"
            opcode: "call"
            instruction_id: 61
            operand_ids: 60
            inner_subroutines {
              name: "reduction_subroutine_phase_4"
              subroutine_root_id: 64
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_4"
                opcode: "delay"
                instruction_id: 62
                bytes_out: 5
              }
              instructions {
                name: "op2_phase_4"
                opcode: "delay"
                instruction_id: 63
                bytes_out: 5
              }
              instructions {
                name: "sum_phase_4"
                opcode: "delay"
                instruction_id: 64
                ops: 10
                operand_ids: 62
                operand_ids: 63
              }
            }
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_5"
            opcode: "sendrecv"
            instruction_id: 65
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
            operand_ids: 61
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw_unidir-ring_reduction_5"
            opcode: "call"
            instruction_id: 66
            operand_ids: 65
            inner_subroutines {
              name: "reduction_subroutine_phase_5"
              subroutine_root_id: 69
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_5"
                opcode: "delay"
                instruction_id: 67
                bytes_out: 5
              }
              instructions {
                name: "op2_phase_5"
                opcode: "delay"
                instruction_id: 68
                bytes_out: 5
              }
              instructions {
                name: "sum_phase_5"
                opcode: "delay"
                instruction_id: 69
                ops: 10
                operand_ids: 67
                operand_ids: 68
              }
            }
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_6"
            opcode: "sendrecv"
            instruction_id: 70
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
            operand_ids: 66
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw_unidir-ring_reduction_6"
            opcode: "call"
            instruction_id: 71
            operand_ids: 70
            inner_subroutines {
              name: "reduction_subroutine_phase_6"
              subroutine_root_id: 74
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_6"
                opcode: "delay"
                instruction_id: 72
                bytes_out: 5
              }
              instructions {
                name: "op2_phase_6"
                opcode: "delay"
                instruction_id: 73
                bytes_out: 5
              }
              instructions {
                name: "sum_phase_6"
                opcode: "delay"
                instruction_id: 74
                ops: 10
                operand_ids: 72
                operand_ids: 73
              }
            }
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_7"
            opcode: "sendrecv"
            instruction_id: 75
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
            operand_ids: 71
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw_unidir-ring_reduction_7"
            opcode: "call"
            instruction_id: 76
            operand_ids: 75
            inner_subroutines {
              name: "reduction_subroutine_phase_7"
              subroutine_root_id: 79
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_7"
                opcode: "delay"
                instruction_id: 77
                bytes_out: 5
              }
              instructions {
                name: "op2_phase_7"
                opcode: "delay"
                instruction_id: 78
                bytes_out: 5
              }
              instructions {
                name: "sum_phase_7"
                opcode: "delay"
                instruction_id: 79
                ops: 10
                operand_ids: 77
                operand_ids: 78
              }
            }
          }
        }
      }
      instructions {
        name: "reduce-scatter_mesh-2d-ring_bidir-ring_root_1"
        opcode: "null"
        instruction_id: 80
        operand_ids: 8
        operand_ids: 44
      }
    }
  }
}
      )proto";
  google::protobuf::TextFormat::ParseFromString(reducescatter_str,
                                                &reducescatter_proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      reducescatter->ToProto().value(), reducescatter_proto));
}

// Tests expanding 1D-Mesh reduce-scatter with barrier
TEST(Mesh2dRingReduceScatter, WithBarrier) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 2);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  sub_ptr->SetId(3);
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "first_instruction", sub_ptr));
  instr_1->SetOps(4);

  ASSERT_OK_AND_ASSIGN(auto reducescatter,
                       paragraph::Instruction::Create(
      paragraph::Opcode::kReduceScatter, "reduce-scatter", sub_ptr));
  reducescatter->SetBytesOut(80);
  paragraph::CommunicationGroup reducescatter_group = {0, 1, 2, 3};
  reducescatter->AppendCommunicationGroup(reducescatter_group);

  auto reduction_sub = absl::make_unique<paragraph::Subroutine>(
      "reduction_subroutine", graph.get());
  auto reduction_ptr = reduction_sub.get();
  ASSERT_OK_AND_ASSIGN(auto op1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "op1", reduction_ptr));
  op1->SetBytesOut(80);
  ASSERT_OK_AND_ASSIGN(auto op2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "op2", reduction_ptr));
  op2->SetBytesOut(80);
  ASSERT_OK_AND_ASSIGN(auto sum_op, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "sum", reduction_ptr, true));
  sum_op->SetOps(160);
  sum_op->AddOperand(op1);
  sum_op->AddOperand(op2);
  reducescatter->AppendInnerSubroutine(std::move(reduction_sub));

  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "last_instruction", sub_ptr, true));
  instr_3->SetOps(4);

  nlohmann::json config = R"(
    {
      "reduce-scatter": {
        "algorithm": "mesh-2d-ring",
        "concentration": 1,
        "dimension_widths": [2, 2],
        "barrier": {
          "algorithm": "centralized"
        }
      }
    }
  )"_json;

  ASSERT_OK_AND_ASSIGN(auto translators, paragraph::CreateTranslators(
     paragraph::TranslatorType::kCollective, config));
  EXPECT_OK(translators["reduce-scatter"]->Translate(reducescatter));

  paragraph::InstructionProto reducescatter_proto;
  std::string reducescatter_str =
      R"proto(
name: "reduce-scatter"
opcode: "reduce-scatter"
instruction_id: 2
bytes_out: 80
communication_groups {
  group_ids: 0
  group_ids: 1
  group_ids: 2
  group_ids: 3
}
inner_subroutines {
  name: "reduce-scatter_mesh-2d-ring"
  subroutine_root_id: 7
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "reduce-scatter_mesh-2d-ring"
    opcode: "reduce-scatter"
    instruction_id: 7
    bytes_out: 80
    communication_groups {
      group_ids: 0
      group_ids: 2
      group_ids: 3
      group_ids: 1
    }
    inner_subroutines {
      name: "reduce-scatter_mesh-2d-ring_bidir-ring"
      subroutine_root_id: 43
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "reduce-scatter_mesh-2d-ring_bidir-ring_barrier"
        opcode: "barrier"
        instruction_id: 8
        communication_groups {
          group_ids: 0
          group_ids: 2
          group_ids: 3
          group_ids: 1
        }
        inner_subroutines {
          name: "reduce-scatter_mesh-2d-ring_bidir-ring_barrier_centralized"
          subroutine_root_id: 10
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_barrier_centralized_send_to_0"
            opcode: "send"
            instruction_id: 9
            communication_groups {
              group_ids: 0
            }
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_barrier_centralized_recv_from_0"
            opcode: "recv"
            instruction_id: 10
            communication_groups {
              group_ids: 0
            }
            operand_ids: 9
          }
        }
      }
      instructions {
        name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw"
        opcode: "reduce-scatter"
        instruction_id: 11
        bytes_out: 40
        communication_groups {
          group_ids: 0
          group_ids: 2
          group_ids: 3
          group_ids: 1
        }
        operand_ids: 8
        inner_subroutines {
          name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 23
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 12
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 0
              group_ids: 3
            }
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw_unidir-ring_reduction_1"
            opcode: "call"
            instruction_id: 13
            operand_ids: 12
            inner_subroutines {
              name: "reduction_subroutine_phase_1"
              subroutine_root_id: 16
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_1"
                opcode: "delay"
                instruction_id: 14
                bytes_out: 10
              }
              instructions {
                name: "op2_phase_1"
                opcode: "delay"
                instruction_id: 15
                bytes_out: 10
              }
              instructions {
                name: "sum_phase_1"
                opcode: "delay"
                instruction_id: 16
                ops: 20
                operand_ids: 14
                operand_ids: 15
              }
            }
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 17
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 0
              group_ids: 3
            }
            operand_ids: 13
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw_unidir-ring_reduction_2"
            opcode: "call"
            instruction_id: 18
            operand_ids: 17
            inner_subroutines {
              name: "reduction_subroutine_phase_2"
              subroutine_root_id: 21
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_2"
                opcode: "delay"
                instruction_id: 19
                bytes_out: 10
              }
              instructions {
                name: "op2_phase_2"
                opcode: "delay"
                instruction_id: 20
                bytes_out: 10
              }
              instructions {
                name: "sum_phase_2"
                opcode: "delay"
                instruction_id: 21
                ops: 20
                operand_ids: 19
                operand_ids: 20
              }
            }
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_3"
            opcode: "sendrecv"
            instruction_id: 22
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 0
              group_ids: 3
            }
            operand_ids: 18
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw_unidir-ring_reduction_3"
            opcode: "call"
            instruction_id: 23
            operand_ids: 22
            inner_subroutines {
              name: "reduction_subroutine_phase_3"
              subroutine_root_id: 26
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_3"
                opcode: "delay"
                instruction_id: 24
                bytes_out: 10
              }
              instructions {
                name: "op2_phase_3"
                opcode: "delay"
                instruction_id: 25
                bytes_out: 10
              }
              instructions {
                name: "sum_phase_3"
                opcode: "delay"
                instruction_id: 26
                ops: 20
                operand_ids: 24
                operand_ids: 25
              }
            }
          }
        }
      }
      instructions {
        name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw"
        opcode: "reduce-scatter"
        instruction_id: 27
        bytes_out: 40
        communication_groups {
          group_ids: 1
          group_ids: 3
          group_ids: 2
          group_ids: 0
        }
        operand_ids: 8
        inner_subroutines {
          name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 39
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 28
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 3
              group_ids: 0
            }
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw_unidir-ring_reduction_1"
            opcode: "call"
            instruction_id: 29
            operand_ids: 28
            inner_subroutines {
              name: "reduction_subroutine_phase_1"
              subroutine_root_id: 32
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_1"
                opcode: "delay"
                instruction_id: 30
                bytes_out: 10
              }
              instructions {
                name: "op2_phase_1"
                opcode: "delay"
                instruction_id: 31
                bytes_out: 10
              }
              instructions {
                name: "sum_phase_1"
                opcode: "delay"
                instruction_id: 32
                ops: 20
                operand_ids: 30
                operand_ids: 31
              }
            }
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 33
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 3
              group_ids: 0
            }
            operand_ids: 29
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw_unidir-ring_reduction_2"
            opcode: "call"
            instruction_id: 34
            operand_ids: 33
            inner_subroutines {
              name: "reduction_subroutine_phase_2"
              subroutine_root_id: 37
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_2"
                opcode: "delay"
                instruction_id: 35
                bytes_out: 10
              }
              instructions {
                name: "op2_phase_2"
                opcode: "delay"
                instruction_id: 36
                bytes_out: 10
              }
              instructions {
                name: "sum_phase_2"
                opcode: "delay"
                instruction_id: 37
                ops: 20
                operand_ids: 35
                operand_ids: 36
              }
            }
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_3"
            opcode: "sendrecv"
            instruction_id: 38
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 3
              group_ids: 0
            }
            operand_ids: 34
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw_unidir-ring_reduction_3"
            opcode: "call"
            instruction_id: 39
            operand_ids: 38
            inner_subroutines {
              name: "reduction_subroutine_phase_3"
              subroutine_root_id: 42
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_3"
                opcode: "delay"
                instruction_id: 40
                bytes_out: 10
              }
              instructions {
                name: "op2_phase_3"
                opcode: "delay"
                instruction_id: 41
                bytes_out: 10
              }
              instructions {
                name: "sum_phase_3"
                opcode: "delay"
                instruction_id: 42
                ops: 20
                operand_ids: 40
                operand_ids: 41
              }
            }
          }
        }
      }
      instructions {
        name: "reduce-scatter_mesh-2d-ring_bidir-ring_root_2"
        opcode: "null"
        instruction_id: 43
        operand_ids: 11
        operand_ids: 27
      }
    }
  }
}
      )proto";
  google::protobuf::TextFormat::ParseFromString(reducescatter_str,
                                                &reducescatter_proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      reducescatter->ToProto().value(), reducescatter_proto));
}

// Tests expanding 1D-Mesh reduce-scatter
TEST(Mesh2dRingReduceScatter, InconsecutiveProcessors) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 3);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "first_instruction", sub_ptr));
  instr_1->SetOps(4);

  ASSERT_OK_AND_ASSIGN(auto reducescatter,
                       paragraph::Instruction::Create(
      paragraph::Opcode::kReduceScatter, "reduce-scatter", sub_ptr));
  reducescatter->SetBytesOut(48);
  paragraph::CommunicationGroup reducescatter_group = {0, 3, 4};
  reducescatter->AppendCommunicationGroup(reducescatter_group);

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
  reducescatter->AppendInnerSubroutine(std::move(reduction_sub));

  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "last_instruction", sub_ptr, true));
  instr_3->SetOps(4);

  nlohmann::json config = R"(
    {
      "reduce-scatter": {
        "algorithm": "mesh-2d-ring",
        "dimension_widths": [2, 3]
      }
    }
  )"_json;

  ASSERT_OK_AND_ASSIGN(auto translators, paragraph::CreateTranslators(
     paragraph::TranslatorType::kCollective, config));
  EXPECT_OK(translators["reduce-scatter"]->Translate(reducescatter));

  paragraph::InstructionProto reducescatter_proto;
  std::string reducescatter_str =
      R"proto(
name: "reduce-scatter"
opcode: "reduce-scatter"
instruction_id: 2
bytes_out: 48
communication_groups {
  group_ids: 0
  group_ids: 3
  group_ids: 4
}
inner_subroutines {
  name: "reduce-scatter_mesh-2d-ring"
  subroutine_root_id: 7
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "reduce-scatter_mesh-2d-ring"
    opcode: "reduce-scatter"
    instruction_id: 7
    bytes_out: 48
    communication_groups {
      group_ids: 0
      group_ids: 4
      group_ids: 3
    }
    inner_subroutines {
      name: "reduce-scatter_mesh-2d-ring_bidir-ring"
      subroutine_root_id: 30
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw"
        opcode: "reduce-scatter"
        instruction_id: 8
        bytes_out: 24
        communication_groups {
          group_ids: 0
          group_ids: 4
          group_ids: 3
        }
        inner_subroutines {
          name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 15
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 9
            bytes_in: 8
            bytes_out: 8
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw_unidir-ring_reduction_1"
            opcode: "call"
            instruction_id: 10
            operand_ids: 9
            inner_subroutines {
              name: "reduction_subroutine_phase_1"
              subroutine_root_id: 13
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_1"
                opcode: "delay"
                instruction_id: 11
                bytes_out: 8
              }
              instructions {
                name: "op2_phase_1"
                opcode: "delay"
                instruction_id: 12
                bytes_out: 8
              }
              instructions {
                name: "sum_phase_1"
                opcode: "delay"
                instruction_id: 13
                ops: 16
                operand_ids: 11
                operand_ids: 12
              }
            }
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 14
            bytes_in: 8
            bytes_out: 8
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
            operand_ids: 10
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_cw_unidir-ring_reduction_2"
            opcode: "call"
            instruction_id: 15
            operand_ids: 14
            inner_subroutines {
              name: "reduction_subroutine_phase_2"
              subroutine_root_id: 18
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_2"
                opcode: "delay"
                instruction_id: 16
                bytes_out: 8
              }
              instructions {
                name: "op2_phase_2"
                opcode: "delay"
                instruction_id: 17
                bytes_out: 8
              }
              instructions {
                name: "sum_phase_2"
                opcode: "delay"
                instruction_id: 18
                ops: 16
                operand_ids: 16
                operand_ids: 17
              }
            }
          }
        }
      }
      instructions {
        name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw"
        opcode: "reduce-scatter"
        instruction_id: 19
        bytes_out: 24
        communication_groups {
          group_ids: 3
          group_ids: 4
          group_ids: 0
        }
        inner_subroutines {
          name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 26
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 20
            bytes_in: 8
            bytes_out: 8
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw_unidir-ring_reduction_1"
            opcode: "call"
            instruction_id: 21
            operand_ids: 20
            inner_subroutines {
              name: "reduction_subroutine_phase_1"
              subroutine_root_id: 24
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_1"
                opcode: "delay"
                instruction_id: 22
                bytes_out: 8
              }
              instructions {
                name: "op2_phase_1"
                opcode: "delay"
                instruction_id: 23
                bytes_out: 8
              }
              instructions {
                name: "sum_phase_1"
                opcode: "delay"
                instruction_id: 24
                ops: 16
                operand_ids: 22
                operand_ids: 23
              }
            }
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 25
            bytes_in: 8
            bytes_out: 8
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
            operand_ids: 21
          }
          instructions {
            name: "reduce-scatter_mesh-2d-ring_bidir-ring_ccw_unidir-ring_reduction_2"
            opcode: "call"
            instruction_id: 26
            operand_ids: 25
            inner_subroutines {
              name: "reduction_subroutine_phase_2"
              subroutine_root_id: 29
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_2"
                opcode: "delay"
                instruction_id: 27
                bytes_out: 8
              }
              instructions {
                name: "op2_phase_2"
                opcode: "delay"
                instruction_id: 28
                bytes_out: 8
              }
              instructions {
                name: "sum_phase_2"
                opcode: "delay"
                instruction_id: 29
                ops: 16
                operand_ids: 27
                operand_ids: 28
              }
            }
          }
        }
      }
      instructions {
        name: "reduce-scatter_mesh-2d-ring_bidir-ring_root_3"
        opcode: "null"
        instruction_id: 30
        operand_ids: 8
        operand_ids: 19
      }
    }
  }
}
      )proto";
  google::protobuf::TextFormat::ParseFromString(reducescatter_str,
                                                &reducescatter_proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      reducescatter->ToProto().value(), reducescatter_proto));
}
