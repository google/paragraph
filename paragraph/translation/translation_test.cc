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
#include "paragraph/translation/translation.h"

#include <memory>
#include <string>
#include <utility>

#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "paragraph/shim/test_macros.h"
#include "paragraph/translation/translation_map.h"

// Tests expanding send instruction for a simple push-based protocol
TEST(Translation, AllReduceGraph) {
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

  ASSERT_OK_AND_ASSIGN(auto barrier, paragraph::Instruction::Create(
      paragraph::Opcode::kBarrier, "barrier", sub_ptr));
  barrier->AppendCommunicationGroup(allreduce_group);
  barrier->AddOperand(allreduce);

  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "last_instruction", sub_ptr, true));
  instr_3->SetOps(4);

  nlohmann::json translation_config = R"(
    {
      "collective": {
        "all-reduce": {
          "algorithm": "unidir-ring"
        }
      },
      "protocol": {
        "sendrecv": {
          "algorithm": "push"
        }
      }
    }
  )"_json;

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<paragraph::Graph>> translated_graphs,
      paragraph::IndividualizeAndTranslate(graph.get(), translation_config));

  for (uint64_t index = 0; index < translated_graphs.size(); index++) {
    EXPECT_EQ(index, translated_graphs.at(index)->GetProcessorId());
  }

  paragraph::GraphProto graph_0_proto;
  std::string graph_0_str =
      R"proto(
name: "test_graph_0"
entry_subroutine {
  name: "test_subroutine"
  subroutine_root_id: 7
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "first_instruction"
    opcode: "delay"
    instruction_id: 1
    ops: 4
  }
  instructions {
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
      subroutine_root_id: 19
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-reduce_unidir-ring_reduce-scatter"
        opcode: "reduce-scatter"
        instruction_id: 8
        bytes_out: 48
        communication_groups {
          group_ids: 0
          group_ids: 1
          group_ids: 2
        }
        inner_subroutines {
          name: "all-reduce_unidir-ring_reduce-scatter_unidir-ring"
          subroutine_root_id: 15
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_unidir-ring_reduce-scatter_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 9
            bytes_in: 16
            bytes_out: 16
            communication_groups {
              group_ids: 2
              group_ids: 1
            }
            inner_subroutines {
              name: "all-reduce_unidir-ring_reduce-scatter_unidir-ring_sendrecv_1_push"
              subroutine_root_id: 26
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_unidir-ring_reduce-scatter_unidir-ring_sendrecv_1_recvstart"
                opcode: "recv-start"
                instruction_id: 22
                bonded_instruction_id: 25
                communication_groups {
                  group_ids: 2
                }
              }
              instructions {
                name: "all-reduce_unidir-ring_reduce-scatter_unidir-ring_sendrecv_1_sendstart"
                opcode: "send-start"
                instruction_id: 23
                bonded_instruction_id: 24
                bytes_in: 16
                communication_groups {
                  group_ids: 1
                }
                operand_ids: 22
              }
              instructions {
                name: "all-reduce_unidir-ring_reduce-scatter_unidir-ring_sendrecv_1_senddone"
                opcode: "send-done"
                instruction_id: 24
                bonded_instruction_id: 23
                communication_groups {
                  group_ids: 1
                }
                operand_ids: 23
              }
              instructions {
                name: "all-reduce_unidir-ring_reduce-scatter_unidir-ring_sendrecv_1_recvdone"
                opcode: "recv-done"
                instruction_id: 25
                bonded_instruction_id: 22
                bytes_out: 16
                communication_groups {
                  group_ids: 2
                }
                operand_ids: 22
                operand_ids: 23
              }
              instructions {
                name: "all-reduce_unidir-ring_reduce-scatter_unidir-ring_sendrecv_1_root"
                opcode: "null"
                instruction_id: 26
                operand_ids: 25
                operand_ids: 24
              }
            }
          }
          instructions {
            name: "all-reduce_unidir-ring_reduce-scatter_unidir-ring_reduction_1"
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
                bytes_out: 16
              }
              instructions {
                name: "op2_phase_1"
                opcode: "delay"
                instruction_id: 12
                bytes_out: 16
              }
              instructions {
                name: "sum_phase_1"
                opcode: "delay"
                instruction_id: 13
                ops: 32
                operand_ids: 11
                operand_ids: 12
              }
            }
          }
          instructions {
            name: "all-reduce_unidir-ring_reduce-scatter_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 14
            bytes_in: 16
            bytes_out: 16
            communication_groups {
              group_ids: 2
              group_ids: 1
            }
            operand_ids: 10
            inner_subroutines {
              name: "all-reduce_unidir-ring_reduce-scatter_unidir-ring_sendrecv_2_push"
              subroutine_root_id: 31
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_unidir-ring_reduce-scatter_unidir-ring_sendrecv_2_recvstart"
                opcode: "recv-start"
                instruction_id: 27
                bonded_instruction_id: 30
                communication_groups {
                  group_ids: 2
                }
              }
              instructions {
                name: "all-reduce_unidir-ring_reduce-scatter_unidir-ring_sendrecv_2_sendstart"
                opcode: "send-start"
                instruction_id: 28
                bonded_instruction_id: 29
                bytes_in: 16
                communication_groups {
                  group_ids: 1
                }
                operand_ids: 27
              }
              instructions {
                name: "all-reduce_unidir-ring_reduce-scatter_unidir-ring_sendrecv_2_senddone"
                opcode: "send-done"
                instruction_id: 29
                bonded_instruction_id: 28
                communication_groups {
                  group_ids: 1
                }
                operand_ids: 28
              }
              instructions {
                name: "all-reduce_unidir-ring_reduce-scatter_unidir-ring_sendrecv_2_recvdone"
                opcode: "recv-done"
                instruction_id: 30
                bonded_instruction_id: 27
                bytes_out: 16
                communication_groups {
                  group_ids: 2
                }
                operand_ids: 27
                operand_ids: 28
              }
              instructions {
                name: "all-reduce_unidir-ring_reduce-scatter_unidir-ring_sendrecv_2_root"
                opcode: "null"
                instruction_id: 31
                operand_ids: 30
                operand_ids: 29
              }
            }
          }
          instructions {
            name: "all-reduce_unidir-ring_reduce-scatter_unidir-ring_reduction_2"
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
                bytes_out: 16
              }
              instructions {
                name: "op2_phase_2"
                opcode: "delay"
                instruction_id: 17
                bytes_out: 16
              }
              instructions {
                name: "sum_phase_2"
                opcode: "delay"
                instruction_id: 18
                ops: 32
                operand_ids: 16
                operand_ids: 17
              }
            }
          }
        }
      }
      instructions {
        name: "all-reduce_unidir-ring_all-gather"
        opcode: "all-gather"
        instruction_id: 19
        bytes_out: 48
        communication_groups {
          group_ids: 0
          group_ids: 1
          group_ids: 2
        }
        operand_ids: 8
        inner_subroutines {
          name: "all-reduce_unidir-ring_all-gather_unidir-ring"
          subroutine_root_id: 21
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_unidir-ring_all-gather_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 20
            bytes_in: 16
            bytes_out: 16
            communication_groups {
              group_ids: 2
              group_ids: 1
            }
            inner_subroutines {
              name: "all-reduce_unidir-ring_all-gather_unidir-ring_sendrecv_1_push"
              subroutine_root_id: 36
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_unidir-ring_all-gather_unidir-ring_sendrecv_1_recvstart"
                opcode: "recv-start"
                instruction_id: 32
                bonded_instruction_id: 35
                communication_groups {
                  group_ids: 2
                }
              }
              instructions {
                name: "all-reduce_unidir-ring_all-gather_unidir-ring_sendrecv_1_sendstart"
                opcode: "send-start"
                instruction_id: 33
                bonded_instruction_id: 34
                bytes_in: 16
                communication_groups {
                  group_ids: 1
                }
                operand_ids: 32
              }
              instructions {
                name: "all-reduce_unidir-ring_all-gather_unidir-ring_sendrecv_1_senddone"
                opcode: "send-done"
                instruction_id: 34
                bonded_instruction_id: 33
                communication_groups {
                  group_ids: 1
                }
                operand_ids: 33
              }
              instructions {
                name: "all-reduce_unidir-ring_all-gather_unidir-ring_sendrecv_1_recvdone"
                opcode: "recv-done"
                instruction_id: 35
                bonded_instruction_id: 32
                bytes_out: 16
                communication_groups {
                  group_ids: 2
                }
                operand_ids: 32
                operand_ids: 33
              }
              instructions {
                name: "all-reduce_unidir-ring_all-gather_unidir-ring_sendrecv_1_root"
                opcode: "null"
                instruction_id: 36
                operand_ids: 35
                operand_ids: 34
              }
            }
          }
          instructions {
            name: "all-reduce_unidir-ring_all-gather_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 21
            bytes_in: 16
            bytes_out: 16
            communication_groups {
              group_ids: 2
              group_ids: 1
            }
            operand_ids: 20
            inner_subroutines {
              name: "all-reduce_unidir-ring_all-gather_unidir-ring_sendrecv_2_push"
              subroutine_root_id: 41
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_unidir-ring_all-gather_unidir-ring_sendrecv_2_recvstart"
                opcode: "recv-start"
                instruction_id: 37
                bonded_instruction_id: 40
                communication_groups {
                  group_ids: 2
                }
              }
              instructions {
                name: "all-reduce_unidir-ring_all-gather_unidir-ring_sendrecv_2_sendstart"
                opcode: "send-start"
                instruction_id: 38
                bonded_instruction_id: 39
                bytes_in: 16
                communication_groups {
                  group_ids: 1
                }
                operand_ids: 37
              }
              instructions {
                name: "all-reduce_unidir-ring_all-gather_unidir-ring_sendrecv_2_senddone"
                opcode: "send-done"
                instruction_id: 39
                bonded_instruction_id: 38
                communication_groups {
                  group_ids: 1
                }
                operand_ids: 38
              }
              instructions {
                name: "all-reduce_unidir-ring_all-gather_unidir-ring_sendrecv_2_recvdone"
                opcode: "recv-done"
                instruction_id: 40
                bonded_instruction_id: 37
                bytes_out: 16
                communication_groups {
                  group_ids: 2
                }
                operand_ids: 37
                operand_ids: 38
              }
              instructions {
                name: "all-reduce_unidir-ring_all-gather_unidir-ring_sendrecv_2_root"
                opcode: "null"
                instruction_id: 41
                operand_ids: 40
                operand_ids: 39
              }
            }
          }
        }
      }
    }
  }
  instructions {
    name: "barrier"
    opcode: "barrier"
    instruction_id: 6
    communication_groups {
      group_ids: 0
      group_ids: 1
      group_ids: 2
    }
    operand_ids: 2
  }
  instructions {
    name: "last_instruction"
    opcode: "delay"
    instruction_id: 7
    ops: 4
  }
}
      )proto";
  google::protobuf::TextFormat::ParseFromString(graph_0_str, &graph_0_proto);
  for (auto& g : translated_graphs) {
    if (g->GetName() == "test_graph_0") {
      EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
          g->ToProto().value(), graph_0_proto));
      // There are 41 Ids assign by the graph, of which 3 belong to initial
      // reduction subroutine
      EXPECT_EQ(translated_graphs.at(0)->InstructionCount(), 38);
    }
  }
}
