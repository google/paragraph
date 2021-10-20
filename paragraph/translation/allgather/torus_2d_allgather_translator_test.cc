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
#include "paragraph/translation/allgather/torus_2d_allgather_translator.h"

#include <memory>
#include <string>
#include <utility>

#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "gtest/gtest.h"
#include "paragraph/shim/test_macros.h"
#include "paragraph/translation/translation_map.h"

paragraph::InstructionProto no_barrier_test_proto() {
  paragraph::InstructionProto proto;
  std::string test_str =
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
  name: "all-gather_torus-2d"
  subroutine_root_id: 45
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "all-gather_stage-0_dim-0"
    opcode: "all-gather"
    instruction_id: 4
    bytes_out: 40
    communication_groups {
      group_ids: 0
      group_ids: 1
      group_ids: 2
      group_ids: 3
    }
    inner_subroutines {
      name: "all-gather_stage-0_dim-0_bidir-ring"
      subroutine_root_id: 13
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-gather_stage-0_dim-0_bidir-ring_cw"
        opcode: "all-gather"
        instruction_id: 5
        bytes_out: 20
        communication_groups {
          group_ids: 0
          group_ids: 1
          group_ids: 2
          group_ids: 3
        }
        inner_subroutines {
          name: "all-gather_stage-0_dim-0_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 8
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_stage-0_dim-0_bidir-ring_cw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 6
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
          }
          instructions {
            name: "all-gather_stage-0_dim-0_bidir-ring_cw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 7
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
            operand_ids: 6
          }
          instructions {
            name: "all-gather_stage-0_dim-0_bidir-ring_cw_unidir-ring_sendrecv_3"
            opcode: "sendrecv"
            instruction_id: 8
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
            operand_ids: 7
          }
        }
      }
      instructions {
        name: "all-gather_stage-0_dim-0_bidir-ring_ccw"
        opcode: "all-gather"
        instruction_id: 9
        bytes_out: 20
        communication_groups {
          group_ids: 3
          group_ids: 2
          group_ids: 1
          group_ids: 0
        }
        inner_subroutines {
          name: "all-gather_stage-0_dim-0_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 12
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_stage-0_dim-0_bidir-ring_ccw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 10
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 2
              group_ids: 0
            }
          }
          instructions {
            name: "all-gather_stage-0_dim-0_bidir-ring_ccw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 11
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 2
              group_ids: 0
            }
            operand_ids: 10
          }
          instructions {
            name: "all-gather_stage-0_dim-0_bidir-ring_ccw_unidir-ring_sendrecv_3"
            opcode: "sendrecv"
            instruction_id: 12
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 2
              group_ids: 0
            }
            operand_ids: 11
          }
        }
      }
      instructions {
        name: "all-gather_stage-0_dim-0_bidir-ring_root_1"
        opcode: "null"
        instruction_id: 13
        operand_ids: 5
        operand_ids: 9
      }
    }
  }
  instructions {
    name: "all-gather_stage-0_dim-1"
    opcode: "all-gather"
    instruction_id: 14
    bytes_out: 40
    communication_groups {
      group_ids: 0
      group_ids: 1
      group_ids: 4
      group_ids: 5
    }
    inner_subroutines {
      name: "all-gather_stage-0_dim-1_bidir-ring"
      subroutine_root_id: 23
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-gather_stage-0_dim-1_bidir-ring_cw"
        opcode: "all-gather"
        instruction_id: 15
        bytes_out: 20
        communication_groups {
          group_ids: 0
          group_ids: 1
          group_ids: 4
          group_ids: 5
        }
        inner_subroutines {
          name: "all-gather_stage-0_dim-1_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 18
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_stage-0_dim-1_bidir-ring_cw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 16
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
          }
          instructions {
            name: "all-gather_stage-0_dim-1_bidir-ring_cw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 17
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
            operand_ids: 16
          }
          instructions {
            name: "all-gather_stage-0_dim-1_bidir-ring_cw_unidir-ring_sendrecv_3"
            opcode: "sendrecv"
            instruction_id: 18
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
            operand_ids: 17
          }
        }
      }
      instructions {
        name: "all-gather_stage-0_dim-1_bidir-ring_ccw"
        opcode: "all-gather"
        instruction_id: 19
        bytes_out: 20
        communication_groups {
          group_ids: 5
          group_ids: 4
          group_ids: 1
          group_ids: 0
        }
        inner_subroutines {
          name: "all-gather_stage-0_dim-1_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 22
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_stage-0_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 20
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
          }
          instructions {
            name: "all-gather_stage-0_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 21
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
            operand_ids: 20
          }
          instructions {
            name: "all-gather_stage-0_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_3"
            opcode: "sendrecv"
            instruction_id: 22
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
            operand_ids: 21
          }
        }
      }
      instructions {
        name: "all-gather_stage-0_dim-1_bidir-ring_root_1"
        opcode: "null"
        instruction_id: 23
        operand_ids: 15
        operand_ids: 19
      }
    }
  }
  instructions {
    name: "all-gather_stage-0_root"
    opcode: "null"
    instruction_id: 24
    operand_ids: 4
    operand_ids: 14
  }
  instructions {
    name: "all-gather_stage-1_dim-0"
    opcode: "all-gather"
    instruction_id: 25
    bytes_out: 80
    communication_groups {
      group_ids: 0
      group_ids: 1
      group_ids: 2
      group_ids: 3
    }
    operand_ids: 24
    inner_subroutines {
      name: "all-gather_stage-1_dim-0_bidir-ring"
      subroutine_root_id: 34
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-gather_stage-1_dim-0_bidir-ring_cw"
        opcode: "all-gather"
        instruction_id: 26
        bytes_out: 40
        communication_groups {
          group_ids: 0
          group_ids: 1
          group_ids: 2
          group_ids: 3
        }
        inner_subroutines {
          name: "all-gather_stage-1_dim-0_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 29
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_stage-1_dim-0_bidir-ring_cw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 27
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
          }
          instructions {
            name: "all-gather_stage-1_dim-0_bidir-ring_cw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 28
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
            operand_ids: 27
          }
          instructions {
            name: "all-gather_stage-1_dim-0_bidir-ring_cw_unidir-ring_sendrecv_3"
            opcode: "sendrecv"
            instruction_id: 29
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
            operand_ids: 28
          }
        }
      }
      instructions {
        name: "all-gather_stage-1_dim-0_bidir-ring_ccw"
        opcode: "all-gather"
        instruction_id: 30
        bytes_out: 40
        communication_groups {
          group_ids: 3
          group_ids: 2
          group_ids: 1
          group_ids: 0
        }
        inner_subroutines {
          name: "all-gather_stage-1_dim-0_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 33
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_stage-1_dim-0_bidir-ring_ccw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 31
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 2
              group_ids: 0
            }
          }
          instructions {
            name: "all-gather_stage-1_dim-0_bidir-ring_ccw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 32
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 2
              group_ids: 0
            }
            operand_ids: 31
          }
          instructions {
            name: "all-gather_stage-1_dim-0_bidir-ring_ccw_unidir-ring_sendrecv_3"
            opcode: "sendrecv"
            instruction_id: 33
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 2
              group_ids: 0
            }
            operand_ids: 32
          }
        }
      }
      instructions {
        name: "all-gather_stage-1_dim-0_bidir-ring_root_1"
        opcode: "null"
        instruction_id: 34
        operand_ids: 26
        operand_ids: 30
      }
    }
  }
  instructions {
    name: "all-gather_stage-1_dim-1"
    opcode: "all-gather"
    instruction_id: 35
    bytes_out: 80
    communication_groups {
      group_ids: 0
      group_ids: 1
      group_ids: 4
      group_ids: 5
    }
    operand_ids: 24
    inner_subroutines {
      name: "all-gather_stage-1_dim-1_bidir-ring"
      subroutine_root_id: 44
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-gather_stage-1_dim-1_bidir-ring_cw"
        opcode: "all-gather"
        instruction_id: 36
        bytes_out: 40
        communication_groups {
          group_ids: 0
          group_ids: 1
          group_ids: 4
          group_ids: 5
        }
        inner_subroutines {
          name: "all-gather_stage-1_dim-1_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 39
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_stage-1_dim-1_bidir-ring_cw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 37
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
          }
          instructions {
            name: "all-gather_stage-1_dim-1_bidir-ring_cw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 38
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
            operand_ids: 37
          }
          instructions {
            name: "all-gather_stage-1_dim-1_bidir-ring_cw_unidir-ring_sendrecv_3"
            opcode: "sendrecv"
            instruction_id: 39
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
            operand_ids: 38
          }
        }
      }
      instructions {
        name: "all-gather_stage-1_dim-1_bidir-ring_ccw"
        opcode: "all-gather"
        instruction_id: 40
        bytes_out: 40
        communication_groups {
          group_ids: 5
          group_ids: 4
          group_ids: 1
          group_ids: 0
        }
        inner_subroutines {
          name: "all-gather_stage-1_dim-1_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 43
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_stage-1_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 41
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
          }
          instructions {
            name: "all-gather_stage-1_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 42
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
            operand_ids: 41
          }
          instructions {
            name: "all-gather_stage-1_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_3"
            opcode: "sendrecv"
            instruction_id: 43
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
            operand_ids: 42
          }
        }
      }
      instructions {
        name: "all-gather_stage-1_dim-1_bidir-ring_root_1"
        opcode: "null"
        instruction_id: 44
        operand_ids: 36
        operand_ids: 40
      }
    }
  }
  instructions {
    name: "all-gather_stage-1_root"
    opcode: "null"
    instruction_id: 45
    operand_ids: 25
    operand_ids: 35
  }
}
    )proto";
  google::protobuf::TextFormat::ParseFromString(test_str,
                                                &proto);
  return proto;
}  // NOLINT

// Tests expanding 2D-Torus all-gather
TEST(Torus2dAllGather, NoBarrier) {
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
        "algorithm": "torus-2d",
        "concentration": 2,
        "dimension_widths": [2, 2],
        "integrated_local_exchange": true
      }
    }
  )"_json;

  ASSERT_OK_AND_ASSIGN(auto translators, paragraph::CreateTranslators(
     paragraph::TranslatorType::kCollective, config));
  EXPECT_OK(translators["all-gather"]->Translate(allgather));

  paragraph::InstructionProto allgather_proto = no_barrier_test_proto();
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      allgather->ToProto().value(), allgather_proto));
}

paragraph::InstructionProto with_barrier_test_proto() {
  paragraph::InstructionProto proto;
  std::string test_str =
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
  name: "all-gather_torus-2d"
  subroutine_root_id: 44
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "all-gather_stage-0_dim-0"
    opcode: "all-gather"
    instruction_id: 4
    bytes_out: 20
    communication_groups {
      group_ids: 0
      group_ids: 2
    }
    inner_subroutines {
      name: "all-gather_stage-0_dim-0_bidir-ring"
      subroutine_root_id: 12
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-gather_stage-0_dim-0_bidir-ring_barrier"
        opcode: "barrier"
        instruction_id: 5
        communication_groups {
          group_ids: 0
          group_ids: 2
        }
        inner_subroutines {
          name: "all-gather_stage-0_dim-0_bidir-ring_barrier_centralized"
          subroutine_root_id: 7
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_stage-0_dim-0_bidir-ring_barrier_centralized_send_to_0"
            opcode: "send"
            instruction_id: 6
            communication_groups {
              group_ids: 0
            }
          }
          instructions {
            name: "all-gather_stage-0_dim-0_bidir-ring_barrier_centralized_recv_from_0"
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
        name: "all-gather_stage-0_dim-0_bidir-ring_cw"
        opcode: "all-gather"
        instruction_id: 8
        bytes_out: 10
        communication_groups {
          group_ids: 0
          group_ids: 2
        }
        operand_ids: 5
        inner_subroutines {
          name: "all-gather_stage-0_dim-0_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 9
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_stage-0_dim-0_bidir-ring_cw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 9
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 0
            }
          }
        }
      }
      instructions {
        name: "all-gather_stage-0_dim-0_bidir-ring_ccw"
        opcode: "all-gather"
        instruction_id: 10
        bytes_out: 10
        communication_groups {
          group_ids: 2
          group_ids: 0
        }
        operand_ids: 5
        inner_subroutines {
          name: "all-gather_stage-0_dim-0_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 11
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_stage-0_dim-0_bidir-ring_ccw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 11
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 0
            }
          }
        }
      }
      instructions {
        name: "all-gather_stage-0_dim-0_bidir-ring_root_2"
        opcode: "null"
        instruction_id: 12
        operand_ids: 8
        operand_ids: 10
      }
    }
  }
  instructions {
    name: "all-gather_stage-0_dim-1"
    opcode: "all-gather"
    instruction_id: 13
    bytes_out: 20
    communication_groups {
      group_ids: 2
      group_ids: 6
    }
    inner_subroutines {
      name: "all-gather_stage-0_dim-1_bidir-ring"
      subroutine_root_id: 22
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-gather_stage-0_dim-1_bidir-ring_barrier"
        opcode: "barrier"
        instruction_id: 14
        communication_groups {
          group_ids: 2
          group_ids: 6
        }
        inner_subroutines {
          name: "all-gather_stage-0_dim-1_bidir-ring_barrier_centralized"
          subroutine_root_id: 17
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_stage-0_dim-1_bidir-ring_barrier_centralized_coordinator_recv_from_6"
            opcode: "recv"
            instruction_id: 15
            communication_groups {
              group_ids: 6
            }
          }
          instructions {
            name: "all-gather_stage-0_dim-1_bidir-ring_barrier_centralized_coordinator_send_to_6"
            opcode: "send"
            instruction_id: 16
            communication_groups {
              group_ids: 6
            }
            operand_ids: 15
          }
          instructions {
            name: "all-gather_stage-0_dim-1_bidir-ring_barrier_centralized_root_2"
            opcode: "null"
            instruction_id: 17
            operand_ids: 16
          }
        }
      }
      instructions {
        name: "all-gather_stage-0_dim-1_bidir-ring_cw"
        opcode: "all-gather"
        instruction_id: 18
        bytes_out: 10
        communication_groups {
          group_ids: 2
          group_ids: 6
        }
        operand_ids: 14
        inner_subroutines {
          name: "all-gather_stage-0_dim-1_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 19
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_stage-0_dim-1_bidir-ring_cw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 19
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 6
              group_ids: 6
            }
          }
        }
      }
      instructions {
        name: "all-gather_stage-0_dim-1_bidir-ring_ccw"
        opcode: "all-gather"
        instruction_id: 20
        bytes_out: 10
        communication_groups {
          group_ids: 6
          group_ids: 2
        }
        operand_ids: 14
        inner_subroutines {
          name: "all-gather_stage-0_dim-1_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 21
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_stage-0_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 21
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 6
              group_ids: 6
            }
          }
        }
      }
      instructions {
        name: "all-gather_stage-0_dim-1_bidir-ring_root_2"
        opcode: "null"
        instruction_id: 22
        operand_ids: 18
        operand_ids: 20
      }
    }
  }
  instructions {
    name: "all-gather_stage-0_root"
    opcode: "null"
    instruction_id: 23
    operand_ids: 4
    operand_ids: 13
  }
  instructions {
    name: "all-gather_stage-1_dim-0"
    opcode: "all-gather"
    instruction_id: 24
    bytes_out: 40
    communication_groups {
      group_ids: 0
      group_ids: 2
    }
    operand_ids: 23
    inner_subroutines {
      name: "all-gather_stage-1_dim-0_bidir-ring"
      subroutine_root_id: 32
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-gather_stage-1_dim-0_bidir-ring_barrier"
        opcode: "barrier"
        instruction_id: 25
        communication_groups {
          group_ids: 0
          group_ids: 2
        }
        inner_subroutines {
          name: "all-gather_stage-1_dim-0_bidir-ring_barrier_centralized"
          subroutine_root_id: 27
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_stage-1_dim-0_bidir-ring_barrier_centralized_send_to_0"
            opcode: "send"
            instruction_id: 26
            communication_groups {
              group_ids: 0
            }
          }
          instructions {
            name: "all-gather_stage-1_dim-0_bidir-ring_barrier_centralized_recv_from_0"
            opcode: "recv"
            instruction_id: 27
            communication_groups {
              group_ids: 0
            }
            operand_ids: 26
          }
        }
      }
      instructions {
        name: "all-gather_stage-1_dim-0_bidir-ring_cw"
        opcode: "all-gather"
        instruction_id: 28
        bytes_out: 20
        communication_groups {
          group_ids: 0
          group_ids: 2
        }
        operand_ids: 25
        inner_subroutines {
          name: "all-gather_stage-1_dim-0_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 29
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_stage-1_dim-0_bidir-ring_cw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 29
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 0
              group_ids: 0
            }
          }
        }
      }
      instructions {
        name: "all-gather_stage-1_dim-0_bidir-ring_ccw"
        opcode: "all-gather"
        instruction_id: 30
        bytes_out: 20
        communication_groups {
          group_ids: 2
          group_ids: 0
        }
        operand_ids: 25
        inner_subroutines {
          name: "all-gather_stage-1_dim-0_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 31
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_stage-1_dim-0_bidir-ring_ccw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 31
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 0
              group_ids: 0
            }
          }
        }
      }
      instructions {
        name: "all-gather_stage-1_dim-0_bidir-ring_root_2"
        opcode: "null"
        instruction_id: 32
        operand_ids: 28
        operand_ids: 30
      }
    }
  }
  instructions {
    name: "all-gather_stage-1_dim-1"
    opcode: "all-gather"
    instruction_id: 33
    bytes_out: 40
    communication_groups {
      group_ids: 2
      group_ids: 6
    }
    operand_ids: 23
    inner_subroutines {
      name: "all-gather_stage-1_dim-1_bidir-ring"
      subroutine_root_id: 42
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-gather_stage-1_dim-1_bidir-ring_barrier"
        opcode: "barrier"
        instruction_id: 34
        communication_groups {
          group_ids: 2
          group_ids: 6
        }
        inner_subroutines {
          name: "all-gather_stage-1_dim-1_bidir-ring_barrier_centralized"
          subroutine_root_id: 37
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_stage-1_dim-1_bidir-ring_barrier_centralized_coordinator_recv_from_6"
            opcode: "recv"
            instruction_id: 35
            communication_groups {
              group_ids: 6
            }
          }
          instructions {
            name: "all-gather_stage-1_dim-1_bidir-ring_barrier_centralized_coordinator_send_to_6"
            opcode: "send"
            instruction_id: 36
            communication_groups {
              group_ids: 6
            }
            operand_ids: 35
          }
          instructions {
            name: "all-gather_stage-1_dim-1_bidir-ring_barrier_centralized_root_2"
            opcode: "null"
            instruction_id: 37
            operand_ids: 36
          }
        }
      }
      instructions {
        name: "all-gather_stage-1_dim-1_bidir-ring_cw"
        opcode: "all-gather"
        instruction_id: 38
        bytes_out: 20
        communication_groups {
          group_ids: 2
          group_ids: 6
        }
        operand_ids: 34
        inner_subroutines {
          name: "all-gather_stage-1_dim-1_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 39
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_stage-1_dim-1_bidir-ring_cw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 39
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 6
              group_ids: 6
            }
          }
        }
      }
      instructions {
        name: "all-gather_stage-1_dim-1_bidir-ring_ccw"
        opcode: "all-gather"
        instruction_id: 40
        bytes_out: 20
        communication_groups {
          group_ids: 6
          group_ids: 2
        }
        operand_ids: 34
        inner_subroutines {
          name: "all-gather_stage-1_dim-1_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 41
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_stage-1_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 41
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 6
              group_ids: 6
            }
          }
        }
      }
      instructions {
        name: "all-gather_stage-1_dim-1_bidir-ring_root_2"
        opcode: "null"
        instruction_id: 42
        operand_ids: 38
        operand_ids: 40
      }
    }
  }
  instructions {
    name: "all-gather_stage-1_root"
    opcode: "null"
    instruction_id: 43
    operand_ids: 24
    operand_ids: 33
  }
  instructions {
    name: "all-gather_conc"
    opcode: "all-gather"
    instruction_id: 44
    bytes_out: 80
    communication_groups {
      group_ids: 2
      group_ids: 3
    }
    operand_ids: 43
    inner_subroutines {
      name: "all-gather_conc_bidir-ring"
      subroutine_root_id: 53
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-gather_conc_bidir-ring_barrier"
        opcode: "barrier"
        instruction_id: 45
        communication_groups {
          group_ids: 2
          group_ids: 3
        }
        inner_subroutines {
          name: "all-gather_conc_bidir-ring_barrier_centralized"
          subroutine_root_id: 48
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_conc_bidir-ring_barrier_centralized_coordinator_recv_from_3"
            opcode: "recv"
            instruction_id: 46
            communication_groups {
              group_ids: 3
            }
          }
          instructions {
            name: "all-gather_conc_bidir-ring_barrier_centralized_coordinator_send_to_3"
            opcode: "send"
            instruction_id: 47
            communication_groups {
              group_ids: 3
            }
            operand_ids: 46
          }
          instructions {
            name: "all-gather_conc_bidir-ring_barrier_centralized_root_2"
            opcode: "null"
            instruction_id: 48
            operand_ids: 47
          }
        }
      }
      instructions {
        name: "all-gather_conc_bidir-ring_cw"
        opcode: "all-gather"
        instruction_id: 49
        bytes_out: 40
        communication_groups {
          group_ids: 2
          group_ids: 3
        }
        operand_ids: 45
        inner_subroutines {
          name: "all-gather_conc_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 50
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_conc_bidir-ring_cw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 50
            bytes_in: 20
            bytes_out: 20
            communication_groups {
              group_ids: 3
              group_ids: 3
            }
          }
        }
      }
      instructions {
        name: "all-gather_conc_bidir-ring_ccw"
        opcode: "all-gather"
        instruction_id: 51
        bytes_out: 40
        communication_groups {
          group_ids: 3
          group_ids: 2
        }
        operand_ids: 45
        inner_subroutines {
          name: "all-gather_conc_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 52
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_conc_bidir-ring_ccw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 52
            bytes_in: 20
            bytes_out: 20
            communication_groups {
              group_ids: 3
              group_ids: 3
            }
          }
        }
      }
      instructions {
        name: "all-gather_conc_bidir-ring_root_2"
        opcode: "null"
        instruction_id: 53
        operand_ids: 49
        operand_ids: 51
      }
    }
  }
}
    )proto";
  google::protobuf::TextFormat::ParseFromString(test_str,
                                                &proto);
  return proto;
}  // NOLINT

// Tests expanding 1D-Torus all-gather with barrier
TEST(Torus2dAllGather, WithBarrier) {
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
        "algorithm": "torus-2d",
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

  paragraph::InstructionProto allgather_proto = with_barrier_test_proto();
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      allgather->ToProto().value(), allgather_proto));
}

paragraph::InstructionProto inconsecutive_proc_test_proto() {
  paragraph::InstructionProto proto;
  std::string test_str =
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
  name: "all-gather_torus-2d"
  subroutine_root_id: 21
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "all-gather_stage-0_dim-1"
    opcode: "all-gather"
    instruction_id: 4
    bytes_out: 48
    communication_groups {
      group_ids: 0
      group_ids: 2
      group_ids: 4
    }
    inner_subroutines {
      name: "all-gather_stage-0_dim-1_bidir-ring"
      subroutine_root_id: 11
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-gather_stage-0_dim-1_bidir-ring_cw"
        opcode: "all-gather"
        instruction_id: 5
        bytes_out: 24
        communication_groups {
          group_ids: 0
          group_ids: 2
          group_ids: 4
        }
        inner_subroutines {
          name: "all-gather_stage-0_dim-1_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 7
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_stage-0_dim-1_bidir-ring_cw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 6
            bytes_in: 8
            bytes_out: 8
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
          }
          instructions {
            name: "all-gather_stage-0_dim-1_bidir-ring_cw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 7
            bytes_in: 8
            bytes_out: 8
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
            operand_ids: 6
          }
        }
      }
      instructions {
        name: "all-gather_stage-0_dim-1_bidir-ring_ccw"
        opcode: "all-gather"
        instruction_id: 8
        bytes_out: 24
        communication_groups {
          group_ids: 4
          group_ids: 2
          group_ids: 0
        }
        inner_subroutines {
          name: "all-gather_stage-0_dim-1_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 10
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_stage-0_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_1"
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
            name: "all-gather_stage-0_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 10
            bytes_in: 8
            bytes_out: 8
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
            operand_ids: 9
          }
        }
      }
      instructions {
        name: "all-gather_stage-0_dim-1_bidir-ring_root_2"
        opcode: "null"
        instruction_id: 11
        operand_ids: 5
        operand_ids: 8
      }
    }
  }
  instructions {
    name: "all-gather_stage-0_root"
    opcode: "null"
    instruction_id: 12
    operand_ids: 4
  }
  instructions {
    name: "all-gather_stage-1_dim-1"
    opcode: "all-gather"
    instruction_id: 13
    bytes_out: 144
    communication_groups {
      group_ids: 0
      group_ids: 2
      group_ids: 4
    }
    operand_ids: 12
    inner_subroutines {
      name: "all-gather_stage-1_dim-1_bidir-ring"
      subroutine_root_id: 20
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-gather_stage-1_dim-1_bidir-ring_cw"
        opcode: "all-gather"
        instruction_id: 14
        bytes_out: 72
        communication_groups {
          group_ids: 0
          group_ids: 2
          group_ids: 4
        }
        inner_subroutines {
          name: "all-gather_stage-1_dim-1_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 16
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_stage-1_dim-1_bidir-ring_cw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 15
            bytes_in: 24
            bytes_out: 24
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
          }
          instructions {
            name: "all-gather_stage-1_dim-1_bidir-ring_cw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 16
            bytes_in: 24
            bytes_out: 24
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
            operand_ids: 15
          }
        }
      }
      instructions {
        name: "all-gather_stage-1_dim-1_bidir-ring_ccw"
        opcode: "all-gather"
        instruction_id: 17
        bytes_out: 72
        communication_groups {
          group_ids: 4
          group_ids: 2
          group_ids: 0
        }
        inner_subroutines {
          name: "all-gather_stage-1_dim-1_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 19
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-gather_stage-1_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 18
            bytes_in: 24
            bytes_out: 24
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
          }
          instructions {
            name: "all-gather_stage-1_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 19
            bytes_in: 24
            bytes_out: 24
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
            operand_ids: 18
          }
        }
      }
      instructions {
        name: "all-gather_stage-1_dim-1_bidir-ring_root_2"
        opcode: "null"
        instruction_id: 20
        operand_ids: 14
        operand_ids: 17
      }
    }
  }
  instructions {
    name: "all-gather_stage-1_root"
    opcode: "null"
    instruction_id: 21
    operand_ids: 13
  }
}
    )proto";
  google::protobuf::TextFormat::ParseFromString(test_str,
                                                &proto);
  return proto;
}  // NOLINT

// Tests expanding 1D-Torus all-gather
TEST(Torus2dAllGather, InconsecutiveProcessors) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 2);
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
  paragraph::CommunicationGroup allgather_group = {0, 2, 4};
  allgather->AppendCommunicationGroup(allgather_group);

  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "last_instruction", sub_ptr, true));
  instr_3->SetOps(4);

  nlohmann::json config = R"(
    {
      "all-gather": {
        "algorithm": "torus-2d",
        "dimension_widths": [2, 3]
      }
    }
  )"_json;

  ASSERT_OK_AND_ASSIGN(auto translators, paragraph::CreateTranslators(
     paragraph::TranslatorType::kCollective, config));
  EXPECT_OK(translators["all-gather"]->Translate(allgather));

  paragraph::InstructionProto allgather_proto =
      inconsecutive_proc_test_proto();
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      allgather->ToProto().value(), allgather_proto));
}
