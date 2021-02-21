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
#include "paragraph/translation/allgather/mesh_2d_ring_allgather_translator.h"

#include <memory>
#include <string>
#include <utility>

#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "gtest/gtest.h"
#include "paragraph/shim/test_macros.h"
#include "paragraph/translation/translation_map.h"

// Tests expanding 2D-Mesh all-gather
TEST(Mesh2dRingAllGather, NoBarrier) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "first_instruction", sub_ptr));
  instr_1->SetOps(4);

  ASSERT_OK_AND_ASSIGN(auto allgather,
                       paragraph::Instruction::Create(
      paragraph::Opcode::kAllGather, "all-gather", sub_ptr));
  allgather->SetBytesOut(80);
  paragraph::CommunicationGroup allgather_group = {0, 1, 2, 3, 4, 5, 6, 7};
  allgather->AppendCommunicationGroup(allgather_group);

  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "last_instruction", sub_ptr, true));
  instr_3->SetOps(4);

  nlohmann::json config = R"(
    {
      "all-gather": {
        "algorithm": "mesh-2d-ring",
        "concentration": 2,
        "dimension_widths": [2, 2]
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
  name: "all-gather_mesh-2d-ring"
  subroutine_root_id: 4
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "all-gather_mesh-2d-ring"
    opcode: "all-gather"
    instruction_id: 4
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
      name: "all-gather_mesh-2d-ring_bidir-ring"
      subroutine_root_id: 21
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-gather_mesh-2d-ring_bidir-ring_cw"
        opcode: "all-gather"
        instruction_id: 5
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
          name: "all-gather_mesh-2d-ring_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 12
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 6
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
          }
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 7
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
            operand_ids: 6
          }
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_3"
            opcode: "sendrecv"
            instruction_id: 8
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
            operand_ids: 7
          }
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_4"
            opcode: "sendrecv"
            instruction_id: 9
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
            operand_ids: 8
          }
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_5"
            opcode: "sendrecv"
            instruction_id: 10
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
            operand_ids: 9
          }
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_6"
            opcode: "sendrecv"
            instruction_id: 11
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
            operand_ids: 10
          }
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_7"
            opcode: "sendrecv"
            instruction_id: 12
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
            operand_ids: 11
          }
        }
      }
      instructions {
        name: "all-gather_mesh-2d-ring_bidir-ring_ccw"
        opcode: "all-gather"
        instruction_id: 13
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
          name: "all-gather_mesh-2d-ring_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 20
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 14
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
          }
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 15
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
            operand_ids: 14
          }
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_3"
            opcode: "sendrecv"
            instruction_id: 16
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
            operand_ids: 15
          }
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_4"
            opcode: "sendrecv"
            instruction_id: 17
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
            operand_ids: 16
          }
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_5"
            opcode: "sendrecv"
            instruction_id: 18
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
            operand_ids: 17
          }
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_6"
            opcode: "sendrecv"
            instruction_id: 19
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
            operand_ids: 18
          }
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_7"
            opcode: "sendrecv"
            instruction_id: 20
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
            operand_ids: 19
          }
        }
      }
      instructions {
        name: "all-gather_mesh-2d-ring_bidir-ring_root_1"
        opcode: "null"
        instruction_id: 21
        operand_ids: 5
        operand_ids: 13
      }
    }
  }
}
      )proto";
  google::protobuf::TextFormat::ParseFromString(allgather_str,
                                                &allgather_proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      allgather->ToProto().value(), allgather_proto));
}

// Tests expanding 1D-Mesh all-gather with barrier
TEST(Mesh2dRingAllGather, WithBarrier) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 2);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  sub_ptr->SetId(3);
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "first_instruction", sub_ptr));
  instr_1->SetOps(4);

  ASSERT_OK_AND_ASSIGN(auto allgather,
                       paragraph::Instruction::Create(
      paragraph::Opcode::kAllGather, "all-gather", sub_ptr));
  allgather->SetBytesOut(80);
  paragraph::CommunicationGroup allgather_group = {0, 1, 2, 3, 4, 5, 6, 7};
  allgather->AppendCommunicationGroup(allgather_group);

  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "last_instruction", sub_ptr, true));
  instr_3->SetOps(4);

  nlohmann::json config = R"(
    {
      "all-gather": {
        "algorithm": "mesh-2d-ring",
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
  EXPECT_OK(translators["all-gather"]->Translate(allgather));

  paragraph::InstructionProto allgather_proto;
  std::string allgather_str =
      R"proto(
name: "all-gather"
opcode: "all-gather"
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
  name: "all-gather_mesh-2d-ring"
  subroutine_root_id: 4
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "all-gather_mesh-2d-ring"
    opcode: "all-gather"
    instruction_id: 4
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
      name: "all-gather_mesh-2d-ring_bidir-ring"
      subroutine_root_id: 24
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-gather_mesh-2d-ring_bidir-ring_barrier"
        opcode: "barrier"
        instruction_id: 5
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
          name: "all-gather_mesh-2d-ring_bidir-ring_barrier_centralized"
          subroutine_root_id: 7
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_barrier_centralized_send_to_0"
            opcode: "send"
            instruction_id: 6
            communication_groups {
              group_ids: 0
            }
          }
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_barrier_centralized_recv_from_0"
            opcode: "recv"
            instruction_id: 7
            communication_groups {
              group_ids: 0
            }
            operand_ids: 6
          }
        }
      }
      instructions {
        name: "all-gather_mesh-2d-ring_bidir-ring_cw"
        opcode: "all-gather"
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
        operand_ids: 5
        inner_subroutines {
          name: "all-gather_mesh-2d-ring_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 15
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 9
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 7
              group_ids: 3
            }
          }
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 10
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 7
              group_ids: 3
            }
            operand_ids: 9
          }
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_3"
            opcode: "sendrecv"
            instruction_id: 11
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 7
              group_ids: 3
            }
            operand_ids: 10
          }
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_4"
            opcode: "sendrecv"
            instruction_id: 12
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 7
              group_ids: 3
            }
            operand_ids: 11
          }
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_5"
            opcode: "sendrecv"
            instruction_id: 13
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 7
              group_ids: 3
            }
            operand_ids: 12
          }
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_6"
            opcode: "sendrecv"
            instruction_id: 14
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 7
              group_ids: 3
            }
            operand_ids: 13
          }
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_7"
            opcode: "sendrecv"
            instruction_id: 15
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 7
              group_ids: 3
            }
            operand_ids: 14
          }
        }
      }
      instructions {
        name: "all-gather_mesh-2d-ring_bidir-ring_ccw"
        opcode: "all-gather"
        instruction_id: 16
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
        operand_ids: 5
        inner_subroutines {
          name: "all-gather_mesh-2d-ring_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 23
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 17
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 3
              group_ids: 7
            }
          }
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 18
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 3
              group_ids: 7
            }
            operand_ids: 17
          }
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_3"
            opcode: "sendrecv"
            instruction_id: 19
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 3
              group_ids: 7
            }
            operand_ids: 18
          }
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_4"
            opcode: "sendrecv"
            instruction_id: 20
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 3
              group_ids: 7
            }
            operand_ids: 19
          }
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_5"
            opcode: "sendrecv"
            instruction_id: 21
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 3
              group_ids: 7
            }
            operand_ids: 20
          }
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_6"
            opcode: "sendrecv"
            instruction_id: 22
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 3
              group_ids: 7
            }
            operand_ids: 21
          }
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_7"
            opcode: "sendrecv"
            instruction_id: 23
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 3
              group_ids: 7
            }
            operand_ids: 22
          }
        }
      }
      instructions {
        name: "all-gather_mesh-2d-ring_bidir-ring_root_2"
        opcode: "null"
        instruction_id: 24
        operand_ids: 8
        operand_ids: 16
      }
    }
  }
}
      )proto";
  google::protobuf::TextFormat::ParseFromString(allgather_str,
                                                &allgather_proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      allgather->ToProto().value(), allgather_proto));
}

// Tests expanding 1D-Mesh all-gather
TEST(Mesh2dRingAllGather, InconsecutiveProcessors) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 3);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "first_instruction", sub_ptr));
  instr_1->SetOps(4);

  ASSERT_OK_AND_ASSIGN(auto allgather,
                       paragraph::Instruction::Create(
      paragraph::Opcode::kAllGather, "all-gather", sub_ptr));
  allgather->SetBytesOut(48);
  paragraph::CommunicationGroup allgather_group = {0, 3, 4};
  allgather->AppendCommunicationGroup(allgather_group);

  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "last_instruction", sub_ptr, true));
  instr_3->SetOps(4);

  nlohmann::json config = R"(
    {
      "all-gather": {
        "algorithm": "mesh-2d-ring",
        "dimension_widths": [2, 3]
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
  group_ids: 3
  group_ids: 4
}
inner_subroutines {
  name: "all-gather_mesh-2d-ring"
  subroutine_root_id: 4
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "all-gather_mesh-2d-ring"
    opcode: "all-gather"
    instruction_id: 4
    bytes_out: 48
    communication_groups {
      group_ids: 0
      group_ids: 4
      group_ids: 3
    }
    inner_subroutines {
      name: "all-gather_mesh-2d-ring_bidir-ring"
      subroutine_root_id: 11
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-gather_mesh-2d-ring_bidir-ring_cw"
        opcode: "all-gather"
        instruction_id: 5
        bytes_out: 24
        communication_groups {
          group_ids: 0
          group_ids: 4
          group_ids: 3
        }
        inner_subroutines {
          name: "all-gather_mesh-2d-ring_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 7
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 6
            bytes_in: 8
            bytes_out: 8
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
          }
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_cw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 7
            bytes_in: 8
            bytes_out: 8
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
            operand_ids: 6
          }
        }
      }
      instructions {
        name: "all-gather_mesh-2d-ring_bidir-ring_ccw"
        opcode: "all-gather"
        instruction_id: 8
        bytes_out: 24
        communication_groups {
          group_ids: 3
          group_ids: 4
          group_ids: 0
        }
        inner_subroutines {
          name: "all-gather_mesh-2d-ring_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 10
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 9
            bytes_in: 8
            bytes_out: 8
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
          }
          instructions {
            name: "all-gather_mesh-2d-ring_bidir-ring_ccw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 10
            bytes_in: 8
            bytes_out: 8
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
            operand_ids: 9
          }
        }
      }
      instructions {
        name: "all-gather_mesh-2d-ring_bidir-ring_root_3"
        opcode: "null"
        instruction_id: 11
        operand_ids: 5
        operand_ids: 8
      }
    }
  }
}
      )proto";
  google::protobuf::TextFormat::ParseFromString(allgather_str,
                                                &allgather_proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      allgather->ToProto().value(), allgather_proto));
}
