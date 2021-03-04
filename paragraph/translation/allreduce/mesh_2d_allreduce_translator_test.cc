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
#include "paragraph/translation/allreduce/mesh_2d_allreduce_translator.h"

#include <memory>
#include <string>
#include <utility>

#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "gtest/gtest.h"
#include "paragraph/shim/test_macros.h"
#include "paragraph/translation/translation_map.h"

paragraph::InstructionProto Mesh2dAllReduce_no_barrier_test_proto() {
  paragraph::InstructionProto proto;
  std::string test_str =
      R"proto(
name: "all-reduce"
opcode: "all-reduce"
instruction_id: 2
bytes_out: 48
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
  name: "all-reduce_mesh-2d"
  subroutine_root_id: 40
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "all-reduce_mesh-2d_reduce-scatter"
    opcode: "reduce-scatter"
    instruction_id: 7
    bytes_out: 48
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
      name: "all-reduce_mesh-2d_reduce-scatter_mesh-2d"
      subroutine_root_id: 34
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-reduce_mesh-2d_reduce-scatter_stage-0_dim-0"
        opcode: "reduce-scatter"
        instruction_id: 8
        bytes_out: 12
        communication_groups {
          group_ids: 0
          group_ids: 2
        }
        inner_subroutines {
          name: "all-reduce_mesh-2d_reduce-scatter_stage-0_dim-0_mesh-1d"
          subroutine_root_id: 10
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_mesh-2d_reduce-scatter_stage-0_dim-0_mesh-1d_ccw_sendrecv_0"
            opcode: "sendrecv"
            instruction_id: 9
            bytes_in: 6
            bytes_out: 6
            communication_groups {
              group_ids: 0
              group_ids: 0
            }
          }
          instructions {
            name: "all-reduce_mesh-2d_reduce-scatter_stage-0_dim-0_mesh-1d_ccw_reduction_0"
            opcode: "call"
            instruction_id: 10
            operand_ids: 9
            inner_subroutines {
              name: "reduction_subroutine_ccw_phase_0"
              subroutine_root_id: 13
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_ccw_phase_0"
                opcode: "delay"
                instruction_id: 11
                bytes_out: 6
              }
              instructions {
                name: "op2_ccw_phase_0"
                opcode: "delay"
                instruction_id: 12
                bytes_out: 6
              }
              instructions {
                name: "sum_ccw_phase_0"
                opcode: "delay"
                instruction_id: 13
                ops: 12
                operand_ids: 11
                operand_ids: 12
              }
            }
          }
        }
      }
      instructions {
        name: "all-reduce_mesh-2d_reduce-scatter_stage-0_dim-1"
        opcode: "reduce-scatter"
        instruction_id: 14
        bytes_out: 12
        communication_groups {
          group_ids: 2
          group_ids: 6
        }
        inner_subroutines {
          name: "all-reduce_mesh-2d_reduce-scatter_stage-0_dim-1_mesh-1d"
          subroutine_root_id: 16
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_mesh-2d_reduce-scatter_stage-0_dim-1_mesh-1d_cw_sendrecv_0"
            opcode: "sendrecv"
            instruction_id: 15
            bytes_in: 6
            bytes_out: 6
            communication_groups {
              group_ids: 6
              group_ids: 6
            }
          }
          instructions {
            name: "all-reduce_mesh-2d_reduce-scatter_stage-0_dim-1_mesh-1d_cw_reduction_0"
            opcode: "call"
            instruction_id: 16
            operand_ids: 15
            inner_subroutines {
              name: "reduction_subroutine_cw_phase_0"
              subroutine_root_id: 19
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_cw_phase_0"
                opcode: "delay"
                instruction_id: 17
                bytes_out: 6
              }
              instructions {
                name: "op2_cw_phase_0"
                opcode: "delay"
                instruction_id: 18
                bytes_out: 6
              }
              instructions {
                name: "sum_cw_phase_0"
                opcode: "delay"
                instruction_id: 19
                ops: 12
                operand_ids: 17
                operand_ids: 18
              }
            }
          }
        }
      }
      instructions {
        name: "all-reduce_mesh-2d_reduce-scatter_stage-0_root"
        opcode: "null"
        instruction_id: 20
        operand_ids: 8
        operand_ids: 14
      }
      instructions {
        name: "all-reduce_mesh-2d_reduce-scatter_stage-1_dim-0"
        opcode: "reduce-scatter"
        instruction_id: 21
        bytes_out: 24
        communication_groups {
          group_ids: 0
          group_ids: 2
        }
        operand_ids: 20
        inner_subroutines {
          name: "all-reduce_mesh-2d_reduce-scatter_stage-1_dim-0_mesh-1d"
          subroutine_root_id: 23
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_mesh-2d_reduce-scatter_stage-1_dim-0_mesh-1d_ccw_sendrecv_0"
            opcode: "sendrecv"
            instruction_id: 22
            bytes_in: 12
            bytes_out: 12
            communication_groups {
              group_ids: 0
              group_ids: 0
            }
          }
          instructions {
            name: "all-reduce_mesh-2d_reduce-scatter_stage-1_dim-0_mesh-1d_ccw_reduction_0"
            opcode: "call"
            instruction_id: 23
            operand_ids: 22
            inner_subroutines {
              name: "reduction_subroutine_ccw_phase_0"
              subroutine_root_id: 26
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_ccw_phase_0"
                opcode: "delay"
                instruction_id: 24
                bytes_out: 12
              }
              instructions {
                name: "op2_ccw_phase_0"
                opcode: "delay"
                instruction_id: 25
                bytes_out: 12
              }
              instructions {
                name: "sum_ccw_phase_0"
                opcode: "delay"
                instruction_id: 26
                ops: 24
                operand_ids: 24
                operand_ids: 25
              }
            }
          }
        }
      }
      instructions {
        name: "all-reduce_mesh-2d_reduce-scatter_stage-1_dim-1"
        opcode: "reduce-scatter"
        instruction_id: 27
        bytes_out: 24
        communication_groups {
          group_ids: 2
          group_ids: 6
        }
        operand_ids: 20
        inner_subroutines {
          name: "all-reduce_mesh-2d_reduce-scatter_stage-1_dim-1_mesh-1d"
          subroutine_root_id: 29
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_mesh-2d_reduce-scatter_stage-1_dim-1_mesh-1d_cw_sendrecv_0"
            opcode: "sendrecv"
            instruction_id: 28
            bytes_in: 12
            bytes_out: 12
            communication_groups {
              group_ids: 6
              group_ids: 6
            }
          }
          instructions {
            name: "all-reduce_mesh-2d_reduce-scatter_stage-1_dim-1_mesh-1d_cw_reduction_0"
            opcode: "call"
            instruction_id: 29
            operand_ids: 28
            inner_subroutines {
              name: "reduction_subroutine_cw_phase_0"
              subroutine_root_id: 32
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_cw_phase_0"
                opcode: "delay"
                instruction_id: 30
                bytes_out: 12
              }
              instructions {
                name: "op2_cw_phase_0"
                opcode: "delay"
                instruction_id: 31
                bytes_out: 12
              }
              instructions {
                name: "sum_cw_phase_0"
                opcode: "delay"
                instruction_id: 32
                ops: 24
                operand_ids: 30
                operand_ids: 31
              }
            }
          }
        }
      }
      instructions {
        name: "all-reduce_mesh-2d_reduce-scatter_stage-1_root"
        opcode: "null"
        instruction_id: 33
        operand_ids: 21
        operand_ids: 27
      }
      instructions {
        name: "all-reduce_mesh-2d_reduce-scatter_conc"
        opcode: "reduce-scatter"
        instruction_id: 34
        bytes_out: 48
        communication_groups {
          group_ids: 2
          group_ids: 3
        }
        operand_ids: 33
        inner_subroutines {
          name: "all-reduce_mesh-2d_reduce-scatter_conc_mesh-1d"
          subroutine_root_id: 36
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_mesh-2d_reduce-scatter_conc_mesh-1d_cw_sendrecv_0"
            opcode: "sendrecv"
            instruction_id: 35
            bytes_in: 24
            bytes_out: 24
            communication_groups {
              group_ids: 3
              group_ids: 3
            }
          }
          instructions {
            name: "all-reduce_mesh-2d_reduce-scatter_conc_mesh-1d_cw_reduction_0"
            opcode: "call"
            instruction_id: 36
            operand_ids: 35
            inner_subroutines {
              name: "reduction_subroutine_cw_phase_0"
              subroutine_root_id: 39
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_cw_phase_0"
                opcode: "delay"
                instruction_id: 37
                bytes_out: 24
              }
              instructions {
                name: "op2_cw_phase_0"
                opcode: "delay"
                instruction_id: 38
                bytes_out: 24
              }
              instructions {
                name: "sum_cw_phase_0"
                opcode: "delay"
                instruction_id: 39
                ops: 48
                operand_ids: 37
                operand_ids: 38
              }
            }
          }
        }
      }
    }
  }
  instructions {
    name: "all-reduce_mesh-2d_all-gather"
    opcode: "all-gather"
    instruction_id: 40
    bytes_out: 48
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
    operand_ids: 7
    inner_subroutines {
      name: "all-reduce_mesh-2d_all-gather_mesh-2d"
      subroutine_root_id: 51
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-reduce_mesh-2d_all-gather_stage-0_dim-0"
        opcode: "all-gather"
        instruction_id: 41
        bytes_out: 12
        communication_groups {
          group_ids: 0
          group_ids: 2
        }
        inner_subroutines {
          name: "all-reduce_mesh-2d_all-gather_stage-0_dim-0_mesh-1d"
          subroutine_root_id: 42
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_mesh-2d_all-gather_stage-0_dim-0_mesh-1d_ccw_sendrecv_0"
            opcode: "sendrecv"
            instruction_id: 42
            bytes_in: 6
            bytes_out: 6
            communication_groups {
              group_ids: 0
              group_ids: 0
            }
          }
        }
      }
      instructions {
        name: "all-reduce_mesh-2d_all-gather_stage-0_dim-1"
        opcode: "all-gather"
        instruction_id: 43
        bytes_out: 12
        communication_groups {
          group_ids: 2
          group_ids: 6
        }
        inner_subroutines {
          name: "all-reduce_mesh-2d_all-gather_stage-0_dim-1_mesh-1d"
          subroutine_root_id: 44
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_mesh-2d_all-gather_stage-0_dim-1_mesh-1d_cw_sendrecv_0"
            opcode: "sendrecv"
            instruction_id: 44
            bytes_in: 6
            bytes_out: 6
            communication_groups {
              group_ids: 6
              group_ids: 6
            }
          }
        }
      }
      instructions {
        name: "all-reduce_mesh-2d_all-gather_stage-0_root"
        opcode: "null"
        instruction_id: 45
        operand_ids: 41
        operand_ids: 43
      }
      instructions {
        name: "all-reduce_mesh-2d_all-gather_stage-1_dim-0"
        opcode: "all-gather"
        instruction_id: 46
        bytes_out: 24
        communication_groups {
          group_ids: 0
          group_ids: 2
        }
        operand_ids: 45
        inner_subroutines {
          name: "all-reduce_mesh-2d_all-gather_stage-1_dim-0_mesh-1d"
          subroutine_root_id: 47
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_mesh-2d_all-gather_stage-1_dim-0_mesh-1d_ccw_sendrecv_0"
            opcode: "sendrecv"
            instruction_id: 47
            bytes_in: 12
            bytes_out: 12
            communication_groups {
              group_ids: 0
              group_ids: 0
            }
          }
        }
      }
      instructions {
        name: "all-reduce_mesh-2d_all-gather_stage-1_dim-1"
        opcode: "all-gather"
        instruction_id: 48
        bytes_out: 24
        communication_groups {
          group_ids: 2
          group_ids: 6
        }
        operand_ids: 45
        inner_subroutines {
          name: "all-reduce_mesh-2d_all-gather_stage-1_dim-1_mesh-1d"
          subroutine_root_id: 49
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_mesh-2d_all-gather_stage-1_dim-1_mesh-1d_cw_sendrecv_0"
            opcode: "sendrecv"
            instruction_id: 49
            bytes_in: 12
            bytes_out: 12
            communication_groups {
              group_ids: 6
              group_ids: 6
            }
          }
        }
      }
      instructions {
        name: "all-reduce_mesh-2d_all-gather_stage-1_root"
        opcode: "null"
        instruction_id: 50
        operand_ids: 46
        operand_ids: 48
      }
      instructions {
        name: "all-reduce_mesh-2d_all-gather_conc"
        opcode: "all-gather"
        instruction_id: 51
        bytes_out: 48
        communication_groups {
          group_ids: 2
          group_ids: 3
        }
        operand_ids: 50
        inner_subroutines {
          name: "all-reduce_mesh-2d_all-gather_conc_mesh-1d"
          subroutine_root_id: 52
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_mesh-2d_all-gather_conc_mesh-1d_cw_sendrecv_0"
            opcode: "sendrecv"
            instruction_id: 52
            bytes_in: 24
            bytes_out: 24
            communication_groups {
              group_ids: 3
              group_ids: 3
            }
          }
        }
      }
    }
  }
}
      )proto";
  google::protobuf::TextFormat::ParseFromString(test_str,
                                                &proto);
  return proto;
}  // NOLINT

// Tests expanding 2D-Mesh all-reduce
TEST(Mesh2dAllReduce, NoBarrier) {
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
  paragraph::CommunicationGroup allreduce_group = {0, 1, 2, 3, 4, 5, 6, 7};
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
        "algorithm": "mesh-2d",
        "concentration": 2,
        "dimension_widths": [2, 2]
      }
    }
  )"_json;

  ASSERT_OK_AND_ASSIGN(auto translators, paragraph::CreateTranslators(
      paragraph::TranslatorType::kCollective, config));
  EXPECT_OK(translators["all-reduce"]->Translate(allreduce));

  paragraph::InstructionProto allreduce_proto =
      Mesh2dAllReduce_no_barrier_test_proto();
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      allreduce->ToProto().value(), allreduce_proto));
}

paragraph::InstructionProto Mesh2dAllReduce_with_barrier_test_proto() {
  paragraph::InstructionProto proto;
  std::string test_str =
      R"proto(
name: "all-reduce"
opcode: "all-reduce"
instruction_id: 2
bytes_out: 48
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
  name: "all-reduce_mesh-2d"
  subroutine_root_id: 58
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "all-reduce_mesh-2d_reduce-scatter"
    opcode: "reduce-scatter"
    instruction_id: 7
    bytes_out: 48
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
      name: "all-reduce_mesh-2d_reduce-scatter_mesh-2d"
      subroutine_root_id: 48
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-reduce_mesh-2d_reduce-scatter_stage-0_dim-0"
        opcode: "reduce-scatter"
        instruction_id: 8
        bytes_out: 12
        communication_groups {
          group_ids: 0
          group_ids: 2
        }
        inner_subroutines {
          name: "all-reduce_mesh-2d_reduce-scatter_stage-0_dim-0_mesh-1d"
          subroutine_root_id: 13
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_mesh-2d_reduce-scatter_stage-0_dim-0_unidir-ring_barrier"
            opcode: "barrier"
            instruction_id: 9
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
            inner_subroutines {
              name: "all-reduce_mesh-2d_reduce-scatter_stage-0_dim-0_unidir-ring_barrier_centralized"
              subroutine_root_id: 11
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_mesh-2d_reduce-scatter_stage-0_dim-0_unidir-ring_barrier_centralized_send_to_0"
                opcode: "send"
                instruction_id: 10
                communication_groups {
                  group_ids: 0
                }
              }
              instructions {
                name: "all-reduce_mesh-2d_reduce-scatter_stage-0_dim-0_unidir-ring_barrier_centralized_recv_from_0"
                opcode: "recv"
                instruction_id: 11
                communication_groups {
                  group_ids: 0
                }
                operand_ids: 10
              }
            }
          }
          instructions {
            name: "all-reduce_mesh-2d_reduce-scatter_stage-0_dim-0_mesh-1d_ccw_sendrecv_0"
            opcode: "sendrecv"
            instruction_id: 12
            bytes_in: 6
            bytes_out: 6
            communication_groups {
              group_ids: 0
              group_ids: 0
            }
            operand_ids: 9
          }
          instructions {
            name: "all-reduce_mesh-2d_reduce-scatter_stage-0_dim-0_mesh-1d_ccw_reduction_0"
            opcode: "call"
            instruction_id: 13
            operand_ids: 12
            inner_subroutines {
              name: "reduction_subroutine_ccw_phase_0"
              subroutine_root_id: 16
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_ccw_phase_0"
                opcode: "delay"
                instruction_id: 14
                bytes_out: 6
              }
              instructions {
                name: "op2_ccw_phase_0"
                opcode: "delay"
                instruction_id: 15
                bytes_out: 6
              }
              instructions {
                name: "sum_ccw_phase_0"
                opcode: "delay"
                instruction_id: 16
                ops: 12
                operand_ids: 14
                operand_ids: 15
              }
            }
          }
        }
      }
      instructions {
        name: "all-reduce_mesh-2d_reduce-scatter_stage-0_dim-1"
        opcode: "reduce-scatter"
        instruction_id: 17
        bytes_out: 12
        communication_groups {
          group_ids: 2
          group_ids: 6
        }
        inner_subroutines {
          name: "all-reduce_mesh-2d_reduce-scatter_stage-0_dim-1_mesh-1d"
          subroutine_root_id: 23
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_mesh-2d_reduce-scatter_stage-0_dim-1_unidir-ring_barrier"
            opcode: "barrier"
            instruction_id: 18
            communication_groups {
              group_ids: 2
              group_ids: 6
            }
            inner_subroutines {
              name: "all-reduce_mesh-2d_reduce-scatter_stage-0_dim-1_unidir-ring_barrier_centralized"
              subroutine_root_id: 21
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_mesh-2d_reduce-scatter_stage-0_dim-1_unidir-ring_barrier_centralized_coordinator_recv_from_6"
                opcode: "recv"
                instruction_id: 19
                communication_groups {
                  group_ids: 6
                }
              }
              instructions {
                name: "all-reduce_mesh-2d_reduce-scatter_stage-0_dim-1_unidir-ring_barrier_centralized_coordinator_send_to_6"
                opcode: "send"
                instruction_id: 20
                communication_groups {
                  group_ids: 6
                }
                operand_ids: 19
              }
              instructions {
                name: "all-reduce_mesh-2d_reduce-scatter_stage-0_dim-1_unidir-ring_barrier_centralized_root_2"
                opcode: "null"
                instruction_id: 21
                operand_ids: 20
              }
            }
          }
          instructions {
            name: "all-reduce_mesh-2d_reduce-scatter_stage-0_dim-1_mesh-1d_cw_sendrecv_0"
            opcode: "sendrecv"
            instruction_id: 22
            bytes_in: 6
            bytes_out: 6
            communication_groups {
              group_ids: 6
              group_ids: 6
            }
            operand_ids: 18
          }
          instructions {
            name: "all-reduce_mesh-2d_reduce-scatter_stage-0_dim-1_mesh-1d_cw_reduction_0"
            opcode: "call"
            instruction_id: 23
            operand_ids: 22
            inner_subroutines {
              name: "reduction_subroutine_cw_phase_0"
              subroutine_root_id: 26
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_cw_phase_0"
                opcode: "delay"
                instruction_id: 24
                bytes_out: 6
              }
              instructions {
                name: "op2_cw_phase_0"
                opcode: "delay"
                instruction_id: 25
                bytes_out: 6
              }
              instructions {
                name: "sum_cw_phase_0"
                opcode: "delay"
                instruction_id: 26
                ops: 12
                operand_ids: 24
                operand_ids: 25
              }
            }
          }
        }
      }
      instructions {
        name: "all-reduce_mesh-2d_reduce-scatter_stage-0_root"
        opcode: "null"
        instruction_id: 27
        operand_ids: 8
        operand_ids: 17
      }
      instructions {
        name: "all-reduce_mesh-2d_reduce-scatter_stage-1_dim-0"
        opcode: "reduce-scatter"
        instruction_id: 28
        bytes_out: 24
        communication_groups {
          group_ids: 0
          group_ids: 2
        }
        operand_ids: 27
        inner_subroutines {
          name: "all-reduce_mesh-2d_reduce-scatter_stage-1_dim-0_mesh-1d"
          subroutine_root_id: 33
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_mesh-2d_reduce-scatter_stage-1_dim-0_unidir-ring_barrier"
            opcode: "barrier"
            instruction_id: 29
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
            inner_subroutines {
              name: "all-reduce_mesh-2d_reduce-scatter_stage-1_dim-0_unidir-ring_barrier_centralized"
              subroutine_root_id: 31
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_mesh-2d_reduce-scatter_stage-1_dim-0_unidir-ring_barrier_centralized_send_to_0"
                opcode: "send"
                instruction_id: 30
                communication_groups {
                  group_ids: 0
                }
              }
              instructions {
                name: "all-reduce_mesh-2d_reduce-scatter_stage-1_dim-0_unidir-ring_barrier_centralized_recv_from_0"
                opcode: "recv"
                instruction_id: 31
                communication_groups {
                  group_ids: 0
                }
                operand_ids: 30
              }
            }
          }
          instructions {
            name: "all-reduce_mesh-2d_reduce-scatter_stage-1_dim-0_mesh-1d_ccw_sendrecv_0"
            opcode: "sendrecv"
            instruction_id: 32
            bytes_in: 12
            bytes_out: 12
            communication_groups {
              group_ids: 0
              group_ids: 0
            }
            operand_ids: 29
          }
          instructions {
            name: "all-reduce_mesh-2d_reduce-scatter_stage-1_dim-0_mesh-1d_ccw_reduction_0"
            opcode: "call"
            instruction_id: 33
            operand_ids: 32
            inner_subroutines {
              name: "reduction_subroutine_ccw_phase_0"
              subroutine_root_id: 36
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_ccw_phase_0"
                opcode: "delay"
                instruction_id: 34
                bytes_out: 12
              }
              instructions {
                name: "op2_ccw_phase_0"
                opcode: "delay"
                instruction_id: 35
                bytes_out: 12
              }
              instructions {
                name: "sum_ccw_phase_0"
                opcode: "delay"
                instruction_id: 36
                ops: 24
                operand_ids: 34
                operand_ids: 35
              }
            }
          }
        }
      }
      instructions {
        name: "all-reduce_mesh-2d_reduce-scatter_stage-1_dim-1"
        opcode: "reduce-scatter"
        instruction_id: 37
        bytes_out: 24
        communication_groups {
          group_ids: 2
          group_ids: 6
        }
        operand_ids: 27
        inner_subroutines {
          name: "all-reduce_mesh-2d_reduce-scatter_stage-1_dim-1_mesh-1d"
          subroutine_root_id: 43
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_mesh-2d_reduce-scatter_stage-1_dim-1_unidir-ring_barrier"
            opcode: "barrier"
            instruction_id: 38
            communication_groups {
              group_ids: 2
              group_ids: 6
            }
            inner_subroutines {
              name: "all-reduce_mesh-2d_reduce-scatter_stage-1_dim-1_unidir-ring_barrier_centralized"
              subroutine_root_id: 41
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_mesh-2d_reduce-scatter_stage-1_dim-1_unidir-ring_barrier_centralized_coordinator_recv_from_6"
                opcode: "recv"
                instruction_id: 39
                communication_groups {
                  group_ids: 6
                }
              }
              instructions {
                name: "all-reduce_mesh-2d_reduce-scatter_stage-1_dim-1_unidir-ring_barrier_centralized_coordinator_send_to_6"
                opcode: "send"
                instruction_id: 40
                communication_groups {
                  group_ids: 6
                }
                operand_ids: 39
              }
              instructions {
                name: "all-reduce_mesh-2d_reduce-scatter_stage-1_dim-1_unidir-ring_barrier_centralized_root_2"
                opcode: "null"
                instruction_id: 41
                operand_ids: 40
              }
            }
          }
          instructions {
            name: "all-reduce_mesh-2d_reduce-scatter_stage-1_dim-1_mesh-1d_cw_sendrecv_0"
            opcode: "sendrecv"
            instruction_id: 42
            bytes_in: 12
            bytes_out: 12
            communication_groups {
              group_ids: 6
              group_ids: 6
            }
            operand_ids: 38
          }
          instructions {
            name: "all-reduce_mesh-2d_reduce-scatter_stage-1_dim-1_mesh-1d_cw_reduction_0"
            opcode: "call"
            instruction_id: 43
            operand_ids: 42
            inner_subroutines {
              name: "reduction_subroutine_cw_phase_0"
              subroutine_root_id: 46
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_cw_phase_0"
                opcode: "delay"
                instruction_id: 44
                bytes_out: 12
              }
              instructions {
                name: "op2_cw_phase_0"
                opcode: "delay"
                instruction_id: 45
                bytes_out: 12
              }
              instructions {
                name: "sum_cw_phase_0"
                opcode: "delay"
                instruction_id: 46
                ops: 24
                operand_ids: 44
                operand_ids: 45
              }
            }
          }
        }
      }
      instructions {
        name: "all-reduce_mesh-2d_reduce-scatter_stage-1_root"
        opcode: "null"
        instruction_id: 47
        operand_ids: 28
        operand_ids: 37
      }
      instructions {
        name: "all-reduce_mesh-2d_reduce-scatter_conc"
        opcode: "reduce-scatter"
        instruction_id: 48
        bytes_out: 48
        communication_groups {
          group_ids: 2
          group_ids: 3
        }
        operand_ids: 47
        inner_subroutines {
          name: "all-reduce_mesh-2d_reduce-scatter_conc_mesh-1d"
          subroutine_root_id: 54
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_mesh-2d_reduce-scatter_conc_unidir-ring_barrier"
            opcode: "barrier"
            instruction_id: 49
            communication_groups {
              group_ids: 2
              group_ids: 3
            }
            inner_subroutines {
              name: "all-reduce_mesh-2d_reduce-scatter_conc_unidir-ring_barrier_centralized"
              subroutine_root_id: 52
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_mesh-2d_reduce-scatter_conc_unidir-ring_barrier_centralized_coordinator_recv_from_3"
                opcode: "recv"
                instruction_id: 50
                communication_groups {
                  group_ids: 3
                }
              }
              instructions {
                name: "all-reduce_mesh-2d_reduce-scatter_conc_unidir-ring_barrier_centralized_coordinator_send_to_3"
                opcode: "send"
                instruction_id: 51
                communication_groups {
                  group_ids: 3
                }
                operand_ids: 50
              }
              instructions {
                name: "all-reduce_mesh-2d_reduce-scatter_conc_unidir-ring_barrier_centralized_root_2"
                opcode: "null"
                instruction_id: 52
                operand_ids: 51
              }
            }
          }
          instructions {
            name: "all-reduce_mesh-2d_reduce-scatter_conc_mesh-1d_cw_sendrecv_0"
            opcode: "sendrecv"
            instruction_id: 53
            bytes_in: 24
            bytes_out: 24
            communication_groups {
              group_ids: 3
              group_ids: 3
            }
            operand_ids: 49
          }
          instructions {
            name: "all-reduce_mesh-2d_reduce-scatter_conc_mesh-1d_cw_reduction_0"
            opcode: "call"
            instruction_id: 54
            operand_ids: 53
            inner_subroutines {
              name: "reduction_subroutine_cw_phase_0"
              subroutine_root_id: 57
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_cw_phase_0"
                opcode: "delay"
                instruction_id: 55
                bytes_out: 24
              }
              instructions {
                name: "op2_cw_phase_0"
                opcode: "delay"
                instruction_id: 56
                bytes_out: 24
              }
              instructions {
                name: "sum_cw_phase_0"
                opcode: "delay"
                instruction_id: 57
                ops: 48
                operand_ids: 55
                operand_ids: 56
              }
            }
          }
        }
      }
    }
  }
  instructions {
    name: "all-reduce_mesh-2d_all-gather"
    opcode: "all-gather"
    instruction_id: 58
    bytes_out: 48
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
    operand_ids: 7
    inner_subroutines {
      name: "all-reduce_mesh-2d_all-gather_mesh-2d"
      subroutine_root_id: 83
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-reduce_mesh-2d_all-gather_stage-0_dim-0"
        opcode: "all-gather"
        instruction_id: 59
        bytes_out: 12
        communication_groups {
          group_ids: 0
          group_ids: 2
        }
        inner_subroutines {
          name: "all-reduce_mesh-2d_all-gather_stage-0_dim-0_mesh-1d"
          subroutine_root_id: 63
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_mesh-2d_all-gather_stage-0_dim-0_mesh-1d_barrier"
            opcode: "barrier"
            instruction_id: 60
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
            inner_subroutines {
              name: "all-reduce_mesh-2d_all-gather_stage-0_dim-0_mesh-1d_barrier_centralized"
              subroutine_root_id: 62
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_mesh-2d_all-gather_stage-0_dim-0_mesh-1d_barrier_centralized_send_to_0"
                opcode: "send"
                instruction_id: 61
                communication_groups {
                  group_ids: 0
                }
              }
              instructions {
                name: "all-reduce_mesh-2d_all-gather_stage-0_dim-0_mesh-1d_barrier_centralized_recv_from_0"
                opcode: "recv"
                instruction_id: 62
                communication_groups {
                  group_ids: 0
                }
                operand_ids: 61
              }
            }
          }
          instructions {
            name: "all-reduce_mesh-2d_all-gather_stage-0_dim-0_mesh-1d_ccw_sendrecv_0"
            opcode: "sendrecv"
            instruction_id: 63
            bytes_in: 6
            bytes_out: 6
            communication_groups {
              group_ids: 0
              group_ids: 0
            }
            operand_ids: 60
          }
        }
      }
      instructions {
        name: "all-reduce_mesh-2d_all-gather_stage-0_dim-1"
        opcode: "all-gather"
        instruction_id: 64
        bytes_out: 12
        communication_groups {
          group_ids: 2
          group_ids: 6
        }
        inner_subroutines {
          name: "all-reduce_mesh-2d_all-gather_stage-0_dim-1_mesh-1d"
          subroutine_root_id: 69
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_mesh-2d_all-gather_stage-0_dim-1_mesh-1d_barrier"
            opcode: "barrier"
            instruction_id: 65
            communication_groups {
              group_ids: 2
              group_ids: 6
            }
            inner_subroutines {
              name: "all-reduce_mesh-2d_all-gather_stage-0_dim-1_mesh-1d_barrier_centralized"
              subroutine_root_id: 68
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_mesh-2d_all-gather_stage-0_dim-1_mesh-1d_barrier_centralized_coordinator_recv_from_6"
                opcode: "recv"
                instruction_id: 66
                communication_groups {
                  group_ids: 6
                }
              }
              instructions {
                name: "all-reduce_mesh-2d_all-gather_stage-0_dim-1_mesh-1d_barrier_centralized_coordinator_send_to_6"
                opcode: "send"
                instruction_id: 67
                communication_groups {
                  group_ids: 6
                }
                operand_ids: 66
              }
              instructions {
                name: "all-reduce_mesh-2d_all-gather_stage-0_dim-1_mesh-1d_barrier_centralized_root_2"
                opcode: "null"
                instruction_id: 68
                operand_ids: 67
              }
            }
          }
          instructions {
            name: "all-reduce_mesh-2d_all-gather_stage-0_dim-1_mesh-1d_cw_sendrecv_0"
            opcode: "sendrecv"
            instruction_id: 69
            bytes_in: 6
            bytes_out: 6
            communication_groups {
              group_ids: 6
              group_ids: 6
            }
            operand_ids: 65
          }
        }
      }
      instructions {
        name: "all-reduce_mesh-2d_all-gather_stage-0_root"
        opcode: "null"
        instruction_id: 70
        operand_ids: 59
        operand_ids: 64
      }
      instructions {
        name: "all-reduce_mesh-2d_all-gather_stage-1_dim-0"
        opcode: "all-gather"
        instruction_id: 71
        bytes_out: 24
        communication_groups {
          group_ids: 0
          group_ids: 2
        }
        operand_ids: 70
        inner_subroutines {
          name: "all-reduce_mesh-2d_all-gather_stage-1_dim-0_mesh-1d"
          subroutine_root_id: 75
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_mesh-2d_all-gather_stage-1_dim-0_mesh-1d_barrier"
            opcode: "barrier"
            instruction_id: 72
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
            inner_subroutines {
              name: "all-reduce_mesh-2d_all-gather_stage-1_dim-0_mesh-1d_barrier_centralized"
              subroutine_root_id: 74
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_mesh-2d_all-gather_stage-1_dim-0_mesh-1d_barrier_centralized_send_to_0"
                opcode: "send"
                instruction_id: 73
                communication_groups {
                  group_ids: 0
                }
              }
              instructions {
                name: "all-reduce_mesh-2d_all-gather_stage-1_dim-0_mesh-1d_barrier_centralized_recv_from_0"
                opcode: "recv"
                instruction_id: 74
                communication_groups {
                  group_ids: 0
                }
                operand_ids: 73
              }
            }
          }
          instructions {
            name: "all-reduce_mesh-2d_all-gather_stage-1_dim-0_mesh-1d_ccw_sendrecv_0"
            opcode: "sendrecv"
            instruction_id: 75
            bytes_in: 12
            bytes_out: 12
            communication_groups {
              group_ids: 0
              group_ids: 0
            }
            operand_ids: 72
          }
        }
      }
      instructions {
        name: "all-reduce_mesh-2d_all-gather_stage-1_dim-1"
        opcode: "all-gather"
        instruction_id: 76
        bytes_out: 24
        communication_groups {
          group_ids: 2
          group_ids: 6
        }
        operand_ids: 70
        inner_subroutines {
          name: "all-reduce_mesh-2d_all-gather_stage-1_dim-1_mesh-1d"
          subroutine_root_id: 81
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_mesh-2d_all-gather_stage-1_dim-1_mesh-1d_barrier"
            opcode: "barrier"
            instruction_id: 77
            communication_groups {
              group_ids: 2
              group_ids: 6
            }
            inner_subroutines {
              name: "all-reduce_mesh-2d_all-gather_stage-1_dim-1_mesh-1d_barrier_centralized"
              subroutine_root_id: 80
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_mesh-2d_all-gather_stage-1_dim-1_mesh-1d_barrier_centralized_coordinator_recv_from_6"
                opcode: "recv"
                instruction_id: 78
                communication_groups {
                  group_ids: 6
                }
              }
              instructions {
                name: "all-reduce_mesh-2d_all-gather_stage-1_dim-1_mesh-1d_barrier_centralized_coordinator_send_to_6"
                opcode: "send"
                instruction_id: 79
                communication_groups {
                  group_ids: 6
                }
                operand_ids: 78
              }
              instructions {
                name: "all-reduce_mesh-2d_all-gather_stage-1_dim-1_mesh-1d_barrier_centralized_root_2"
                opcode: "null"
                instruction_id: 80
                operand_ids: 79
              }
            }
          }
          instructions {
            name: "all-reduce_mesh-2d_all-gather_stage-1_dim-1_mesh-1d_cw_sendrecv_0"
            opcode: "sendrecv"
            instruction_id: 81
            bytes_in: 12
            bytes_out: 12
            communication_groups {
              group_ids: 6
              group_ids: 6
            }
            operand_ids: 77
          }
        }
      }
      instructions {
        name: "all-reduce_mesh-2d_all-gather_stage-1_root"
        opcode: "null"
        instruction_id: 82
        operand_ids: 71
        operand_ids: 76
      }
      instructions {
        name: "all-reduce_mesh-2d_all-gather_conc"
        opcode: "all-gather"
        instruction_id: 83
        bytes_out: 48
        communication_groups {
          group_ids: 2
          group_ids: 3
        }
        operand_ids: 82
        inner_subroutines {
          name: "all-reduce_mesh-2d_all-gather_conc_mesh-1d"
          subroutine_root_id: 88
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_mesh-2d_all-gather_conc_mesh-1d_barrier"
            opcode: "barrier"
            instruction_id: 84
            communication_groups {
              group_ids: 2
              group_ids: 3
            }
            inner_subroutines {
              name: "all-reduce_mesh-2d_all-gather_conc_mesh-1d_barrier_centralized"
              subroutine_root_id: 87
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_mesh-2d_all-gather_conc_mesh-1d_barrier_centralized_coordinator_recv_from_3"
                opcode: "recv"
                instruction_id: 85
                communication_groups {
                  group_ids: 3
                }
              }
              instructions {
                name: "all-reduce_mesh-2d_all-gather_conc_mesh-1d_barrier_centralized_coordinator_send_to_3"
                opcode: "send"
                instruction_id: 86
                communication_groups {
                  group_ids: 3
                }
                operand_ids: 85
              }
              instructions {
                name: "all-reduce_mesh-2d_all-gather_conc_mesh-1d_barrier_centralized_root_2"
                opcode: "null"
                instruction_id: 87
                operand_ids: 86
              }
            }
          }
          instructions {
            name: "all-reduce_mesh-2d_all-gather_conc_mesh-1d_cw_sendrecv_0"
            opcode: "sendrecv"
            instruction_id: 88
            bytes_in: 24
            bytes_out: 24
            communication_groups {
              group_ids: 3
              group_ids: 3
            }
            operand_ids: 84
          }
        }
      }
    }
  }
}
      )proto";
  google::protobuf::TextFormat::ParseFromString(test_str,
                                                &proto);
  return proto;
}  // NOLINT

// Tests expanding 2D-Mesh all-reduce with barrier
TEST(Mesh2dAllReduce, WithBarrier) {
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
  paragraph::CommunicationGroup allreduce_group = {0, 1, 2, 3, 4, 5, 6, 7};
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
        "algorithm": "mesh-2d",
        "concentration": 2,
        "dimension_widths": [2, 2],
        "barrier": {
          "algorithm": "centralized"
        }
      }
    }
  )"_json;

  ASSERT_OK_AND_ASSIGN(auto translators, paragraph::CreateTranslators(
      paragraph::TranslatorType::kCollective, config));
  EXPECT_OK(translators["all-reduce"]->Translate(allreduce));

  paragraph::InstructionProto allreduce_proto =
      Mesh2dAllReduce_with_barrier_test_proto();
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      allreduce->ToProto().value(), allreduce_proto));
}
