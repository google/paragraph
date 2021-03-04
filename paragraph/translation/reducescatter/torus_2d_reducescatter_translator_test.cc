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
#include "paragraph/translation/reducescatter/torus_2d_reducescatter_translator.h"

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
  name: "reduce-scatter_torus-2d"
  subroutine_root_id: 144
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "reduce-scatter_stage-0_dim-0"
    opcode: "reduce-scatter"
    instruction_id: 7
    bytes_out: 40
    communication_groups {
      group_ids: 0
      group_ids: 1
      group_ids: 2
      group_ids: 3
    }
    inner_subroutines {
      name: "reduce-scatter_stage-0_dim-0_bidir-ring"
      subroutine_root_id: 40
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "reduce-scatter_stage-0_dim-0_bidir-ring_cw"
        opcode: "reduce-scatter"
        instruction_id: 8
        bytes_out: 20
        communication_groups {
          group_ids: 0
          group_ids: 1
          group_ids: 2
          group_ids: 3
        }
        inner_subroutines {
          name: "reduce-scatter_stage-0_dim-0_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 20
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_stage-0_dim-0_bidir-ring_cw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 9
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-0_bidir-ring_cw_unidir-ring_reduction_1"
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
            name: "reduce-scatter_stage-0_dim-0_bidir-ring_cw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 14
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
            operand_ids: 10
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-0_bidir-ring_cw_unidir-ring_reduction_2"
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
            name: "reduce-scatter_stage-0_dim-0_bidir-ring_cw_unidir-ring_sendrecv_3"
            opcode: "sendrecv"
            instruction_id: 19
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
            operand_ids: 15
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-0_bidir-ring_cw_unidir-ring_reduction_3"
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
        }
      }
      instructions {
        name: "reduce-scatter_stage-0_dim-0_bidir-ring_ccw"
        opcode: "reduce-scatter"
        instruction_id: 24
        bytes_out: 20
        communication_groups {
          group_ids: 3
          group_ids: 2
          group_ids: 1
          group_ids: 0
        }
        inner_subroutines {
          name: "reduce-scatter_stage-0_dim-0_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 36
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_stage-0_dim-0_bidir-ring_ccw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 25
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 2
              group_ids: 0
            }
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-0_bidir-ring_ccw_unidir-ring_reduction_1"
            opcode: "call"
            instruction_id: 26
            operand_ids: 25
            inner_subroutines {
              name: "reduction_subroutine_phase_1"
              subroutine_root_id: 29
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_1"
                opcode: "delay"
                instruction_id: 27
                bytes_out: 5
              }
              instructions {
                name: "op2_phase_1"
                opcode: "delay"
                instruction_id: 28
                bytes_out: 5
              }
              instructions {
                name: "sum_phase_1"
                opcode: "delay"
                instruction_id: 29
                ops: 10
                operand_ids: 27
                operand_ids: 28
              }
            }
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-0_bidir-ring_ccw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 30
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 2
              group_ids: 0
            }
            operand_ids: 26
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-0_bidir-ring_ccw_unidir-ring_reduction_2"
            opcode: "call"
            instruction_id: 31
            operand_ids: 30
            inner_subroutines {
              name: "reduction_subroutine_phase_2"
              subroutine_root_id: 34
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_2"
                opcode: "delay"
                instruction_id: 32
                bytes_out: 5
              }
              instructions {
                name: "op2_phase_2"
                opcode: "delay"
                instruction_id: 33
                bytes_out: 5
              }
              instructions {
                name: "sum_phase_2"
                opcode: "delay"
                instruction_id: 34
                ops: 10
                operand_ids: 32
                operand_ids: 33
              }
            }
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-0_bidir-ring_ccw_unidir-ring_sendrecv_3"
            opcode: "sendrecv"
            instruction_id: 35
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 2
              group_ids: 0
            }
            operand_ids: 31
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-0_bidir-ring_ccw_unidir-ring_reduction_3"
            opcode: "call"
            instruction_id: 36
            operand_ids: 35
            inner_subroutines {
              name: "reduction_subroutine_phase_3"
              subroutine_root_id: 39
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_3"
                opcode: "delay"
                instruction_id: 37
                bytes_out: 5
              }
              instructions {
                name: "op2_phase_3"
                opcode: "delay"
                instruction_id: 38
                bytes_out: 5
              }
              instructions {
                name: "sum_phase_3"
                opcode: "delay"
                instruction_id: 39
                ops: 10
                operand_ids: 37
                operand_ids: 38
              }
            }
          }
        }
      }
      instructions {
        name: "reduce-scatter_stage-0_dim-0_bidir-ring_root_1"
        opcode: "null"
        instruction_id: 40
        operand_ids: 8
        operand_ids: 24
      }
    }
  }
  instructions {
    name: "reduce-scatter_stage-0_dim-1"
    opcode: "reduce-scatter"
    instruction_id: 41
    bytes_out: 40
    communication_groups {
      group_ids: 0
      group_ids: 1
      group_ids: 4
      group_ids: 5
    }
    inner_subroutines {
      name: "reduce-scatter_stage-0_dim-1_bidir-ring"
      subroutine_root_id: 74
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "reduce-scatter_stage-0_dim-1_bidir-ring_cw"
        opcode: "reduce-scatter"
        instruction_id: 42
        bytes_out: 20
        communication_groups {
          group_ids: 0
          group_ids: 1
          group_ids: 4
          group_ids: 5
        }
        inner_subroutines {
          name: "reduce-scatter_stage-0_dim-1_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 54
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_stage-0_dim-1_bidir-ring_cw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 43
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-1_bidir-ring_cw_unidir-ring_reduction_1"
            opcode: "call"
            instruction_id: 44
            operand_ids: 43
            inner_subroutines {
              name: "reduction_subroutine_phase_1"
              subroutine_root_id: 47
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_1"
                opcode: "delay"
                instruction_id: 45
                bytes_out: 5
              }
              instructions {
                name: "op2_phase_1"
                opcode: "delay"
                instruction_id: 46
                bytes_out: 5
              }
              instructions {
                name: "sum_phase_1"
                opcode: "delay"
                instruction_id: 47
                ops: 10
                operand_ids: 45
                operand_ids: 46
              }
            }
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-1_bidir-ring_cw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 48
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
            operand_ids: 44
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-1_bidir-ring_cw_unidir-ring_reduction_2"
            opcode: "call"
            instruction_id: 49
            operand_ids: 48
            inner_subroutines {
              name: "reduction_subroutine_phase_2"
              subroutine_root_id: 52
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_2"
                opcode: "delay"
                instruction_id: 50
                bytes_out: 5
              }
              instructions {
                name: "op2_phase_2"
                opcode: "delay"
                instruction_id: 51
                bytes_out: 5
              }
              instructions {
                name: "sum_phase_2"
                opcode: "delay"
                instruction_id: 52
                ops: 10
                operand_ids: 50
                operand_ids: 51
              }
            }
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-1_bidir-ring_cw_unidir-ring_sendrecv_3"
            opcode: "sendrecv"
            instruction_id: 53
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
            operand_ids: 49
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-1_bidir-ring_cw_unidir-ring_reduction_3"
            opcode: "call"
            instruction_id: 54
            operand_ids: 53
            inner_subroutines {
              name: "reduction_subroutine_phase_3"
              subroutine_root_id: 57
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_3"
                opcode: "delay"
                instruction_id: 55
                bytes_out: 5
              }
              instructions {
                name: "op2_phase_3"
                opcode: "delay"
                instruction_id: 56
                bytes_out: 5
              }
              instructions {
                name: "sum_phase_3"
                opcode: "delay"
                instruction_id: 57
                ops: 10
                operand_ids: 55
                operand_ids: 56
              }
            }
          }
        }
      }
      instructions {
        name: "reduce-scatter_stage-0_dim-1_bidir-ring_ccw"
        opcode: "reduce-scatter"
        instruction_id: 58
        bytes_out: 20
        communication_groups {
          group_ids: 5
          group_ids: 4
          group_ids: 1
          group_ids: 0
        }
        inner_subroutines {
          name: "reduce-scatter_stage-0_dim-1_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 70
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_stage-0_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 59
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-1_bidir-ring_ccw_unidir-ring_reduction_1"
            opcode: "call"
            instruction_id: 60
            operand_ids: 59
            inner_subroutines {
              name: "reduction_subroutine_phase_1"
              subroutine_root_id: 63
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_1"
                opcode: "delay"
                instruction_id: 61
                bytes_out: 5
              }
              instructions {
                name: "op2_phase_1"
                opcode: "delay"
                instruction_id: 62
                bytes_out: 5
              }
              instructions {
                name: "sum_phase_1"
                opcode: "delay"
                instruction_id: 63
                ops: 10
                operand_ids: 61
                operand_ids: 62
              }
            }
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 64
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
            operand_ids: 60
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-1_bidir-ring_ccw_unidir-ring_reduction_2"
            opcode: "call"
            instruction_id: 65
            operand_ids: 64
            inner_subroutines {
              name: "reduction_subroutine_phase_2"
              subroutine_root_id: 68
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_2"
                opcode: "delay"
                instruction_id: 66
                bytes_out: 5
              }
              instructions {
                name: "op2_phase_2"
                opcode: "delay"
                instruction_id: 67
                bytes_out: 5
              }
              instructions {
                name: "sum_phase_2"
                opcode: "delay"
                instruction_id: 68
                ops: 10
                operand_ids: 66
                operand_ids: 67
              }
            }
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_3"
            opcode: "sendrecv"
            instruction_id: 69
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
            operand_ids: 65
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-1_bidir-ring_ccw_unidir-ring_reduction_3"
            opcode: "call"
            instruction_id: 70
            operand_ids: 69
            inner_subroutines {
              name: "reduction_subroutine_phase_3"
              subroutine_root_id: 73
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_3"
                opcode: "delay"
                instruction_id: 71
                bytes_out: 5
              }
              instructions {
                name: "op2_phase_3"
                opcode: "delay"
                instruction_id: 72
                bytes_out: 5
              }
              instructions {
                name: "sum_phase_3"
                opcode: "delay"
                instruction_id: 73
                ops: 10
                operand_ids: 71
                operand_ids: 72
              }
            }
          }
        }
      }
      instructions {
        name: "reduce-scatter_stage-0_dim-1_bidir-ring_root_1"
        opcode: "null"
        instruction_id: 74
        operand_ids: 42
        operand_ids: 58
      }
    }
  }
  instructions {
    name: "reduce-scatter_stage-0_root"
    opcode: "null"
    instruction_id: 75
    operand_ids: 7
    operand_ids: 41
  }
  instructions {
    name: "reduce-scatter_stage-1_dim-0"
    opcode: "reduce-scatter"
    instruction_id: 76
    bytes_out: 80
    communication_groups {
      group_ids: 0
      group_ids: 1
      group_ids: 2
      group_ids: 3
    }
    operand_ids: 75
    inner_subroutines {
      name: "reduce-scatter_stage-1_dim-0_bidir-ring"
      subroutine_root_id: 109
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "reduce-scatter_stage-1_dim-0_bidir-ring_cw"
        opcode: "reduce-scatter"
        instruction_id: 77
        bytes_out: 40
        communication_groups {
          group_ids: 0
          group_ids: 1
          group_ids: 2
          group_ids: 3
        }
        inner_subroutines {
          name: "reduce-scatter_stage-1_dim-0_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 89
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_stage-1_dim-0_bidir-ring_cw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 78
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-0_bidir-ring_cw_unidir-ring_reduction_1"
            opcode: "call"
            instruction_id: 79
            operand_ids: 78
            inner_subroutines {
              name: "reduction_subroutine_phase_1"
              subroutine_root_id: 82
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_1"
                opcode: "delay"
                instruction_id: 80
                bytes_out: 10
              }
              instructions {
                name: "op2_phase_1"
                opcode: "delay"
                instruction_id: 81
                bytes_out: 10
              }
              instructions {
                name: "sum_phase_1"
                opcode: "delay"
                instruction_id: 82
                ops: 20
                operand_ids: 80
                operand_ids: 81
              }
            }
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-0_bidir-ring_cw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 83
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
            operand_ids: 79
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-0_bidir-ring_cw_unidir-ring_reduction_2"
            opcode: "call"
            instruction_id: 84
            operand_ids: 83
            inner_subroutines {
              name: "reduction_subroutine_phase_2"
              subroutine_root_id: 87
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_2"
                opcode: "delay"
                instruction_id: 85
                bytes_out: 10
              }
              instructions {
                name: "op2_phase_2"
                opcode: "delay"
                instruction_id: 86
                bytes_out: 10
              }
              instructions {
                name: "sum_phase_2"
                opcode: "delay"
                instruction_id: 87
                ops: 20
                operand_ids: 85
                operand_ids: 86
              }
            }
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-0_bidir-ring_cw_unidir-ring_sendrecv_3"
            opcode: "sendrecv"
            instruction_id: 88
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
            operand_ids: 84
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-0_bidir-ring_cw_unidir-ring_reduction_3"
            opcode: "call"
            instruction_id: 89
            operand_ids: 88
            inner_subroutines {
              name: "reduction_subroutine_phase_3"
              subroutine_root_id: 92
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_3"
                opcode: "delay"
                instruction_id: 90
                bytes_out: 10
              }
              instructions {
                name: "op2_phase_3"
                opcode: "delay"
                instruction_id: 91
                bytes_out: 10
              }
              instructions {
                name: "sum_phase_3"
                opcode: "delay"
                instruction_id: 92
                ops: 20
                operand_ids: 90
                operand_ids: 91
              }
            }
          }
        }
      }
      instructions {
        name: "reduce-scatter_stage-1_dim-0_bidir-ring_ccw"
        opcode: "reduce-scatter"
        instruction_id: 93
        bytes_out: 40
        communication_groups {
          group_ids: 3
          group_ids: 2
          group_ids: 1
          group_ids: 0
        }
        inner_subroutines {
          name: "reduce-scatter_stage-1_dim-0_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 105
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_stage-1_dim-0_bidir-ring_ccw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 94
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 2
              group_ids: 0
            }
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-0_bidir-ring_ccw_unidir-ring_reduction_1"
            opcode: "call"
            instruction_id: 95
            operand_ids: 94
            inner_subroutines {
              name: "reduction_subroutine_phase_1"
              subroutine_root_id: 98
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_1"
                opcode: "delay"
                instruction_id: 96
                bytes_out: 10
              }
              instructions {
                name: "op2_phase_1"
                opcode: "delay"
                instruction_id: 97
                bytes_out: 10
              }
              instructions {
                name: "sum_phase_1"
                opcode: "delay"
                instruction_id: 98
                ops: 20
                operand_ids: 96
                operand_ids: 97
              }
            }
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-0_bidir-ring_ccw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 99
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 2
              group_ids: 0
            }
            operand_ids: 95
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-0_bidir-ring_ccw_unidir-ring_reduction_2"
            opcode: "call"
            instruction_id: 100
            operand_ids: 99
            inner_subroutines {
              name: "reduction_subroutine_phase_2"
              subroutine_root_id: 103
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_2"
                opcode: "delay"
                instruction_id: 101
                bytes_out: 10
              }
              instructions {
                name: "op2_phase_2"
                opcode: "delay"
                instruction_id: 102
                bytes_out: 10
              }
              instructions {
                name: "sum_phase_2"
                opcode: "delay"
                instruction_id: 103
                ops: 20
                operand_ids: 101
                operand_ids: 102
              }
            }
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-0_bidir-ring_ccw_unidir-ring_sendrecv_3"
            opcode: "sendrecv"
            instruction_id: 104
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 2
              group_ids: 0
            }
            operand_ids: 100
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-0_bidir-ring_ccw_unidir-ring_reduction_3"
            opcode: "call"
            instruction_id: 105
            operand_ids: 104
            inner_subroutines {
              name: "reduction_subroutine_phase_3"
              subroutine_root_id: 108
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_3"
                opcode: "delay"
                instruction_id: 106
                bytes_out: 10
              }
              instructions {
                name: "op2_phase_3"
                opcode: "delay"
                instruction_id: 107
                bytes_out: 10
              }
              instructions {
                name: "sum_phase_3"
                opcode: "delay"
                instruction_id: 108
                ops: 20
                operand_ids: 106
                operand_ids: 107
              }
            }
          }
        }
      }
      instructions {
        name: "reduce-scatter_stage-1_dim-0_bidir-ring_root_1"
        opcode: "null"
        instruction_id: 109
        operand_ids: 77
        operand_ids: 93
      }
    }
  }
  instructions {
    name: "reduce-scatter_stage-1_dim-1"
    opcode: "reduce-scatter"
    instruction_id: 110
    bytes_out: 80
    communication_groups {
      group_ids: 0
      group_ids: 1
      group_ids: 4
      group_ids: 5
    }
    operand_ids: 75
    inner_subroutines {
      name: "reduce-scatter_stage-1_dim-1_bidir-ring"
      subroutine_root_id: 143
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "reduce-scatter_stage-1_dim-1_bidir-ring_cw"
        opcode: "reduce-scatter"
        instruction_id: 111
        bytes_out: 40
        communication_groups {
          group_ids: 0
          group_ids: 1
          group_ids: 4
          group_ids: 5
        }
        inner_subroutines {
          name: "reduce-scatter_stage-1_dim-1_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 123
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_stage-1_dim-1_bidir-ring_cw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 112
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-1_bidir-ring_cw_unidir-ring_reduction_1"
            opcode: "call"
            instruction_id: 113
            operand_ids: 112
            inner_subroutines {
              name: "reduction_subroutine_phase_1"
              subroutine_root_id: 116
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_1"
                opcode: "delay"
                instruction_id: 114
                bytes_out: 10
              }
              instructions {
                name: "op2_phase_1"
                opcode: "delay"
                instruction_id: 115
                bytes_out: 10
              }
              instructions {
                name: "sum_phase_1"
                opcode: "delay"
                instruction_id: 116
                ops: 20
                operand_ids: 114
                operand_ids: 115
              }
            }
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-1_bidir-ring_cw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 117
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
            operand_ids: 113
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-1_bidir-ring_cw_unidir-ring_reduction_2"
            opcode: "call"
            instruction_id: 118
            operand_ids: 117
            inner_subroutines {
              name: "reduction_subroutine_phase_2"
              subroutine_root_id: 121
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_2"
                opcode: "delay"
                instruction_id: 119
                bytes_out: 10
              }
              instructions {
                name: "op2_phase_2"
                opcode: "delay"
                instruction_id: 120
                bytes_out: 10
              }
              instructions {
                name: "sum_phase_2"
                opcode: "delay"
                instruction_id: 121
                ops: 20
                operand_ids: 119
                operand_ids: 120
              }
            }
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-1_bidir-ring_cw_unidir-ring_sendrecv_3"
            opcode: "sendrecv"
            instruction_id: 122
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
            operand_ids: 118
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-1_bidir-ring_cw_unidir-ring_reduction_3"
            opcode: "call"
            instruction_id: 123
            operand_ids: 122
            inner_subroutines {
              name: "reduction_subroutine_phase_3"
              subroutine_root_id: 126
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_3"
                opcode: "delay"
                instruction_id: 124
                bytes_out: 10
              }
              instructions {
                name: "op2_phase_3"
                opcode: "delay"
                instruction_id: 125
                bytes_out: 10
              }
              instructions {
                name: "sum_phase_3"
                opcode: "delay"
                instruction_id: 126
                ops: 20
                operand_ids: 124
                operand_ids: 125
              }
            }
          }
        }
      }
      instructions {
        name: "reduce-scatter_stage-1_dim-1_bidir-ring_ccw"
        opcode: "reduce-scatter"
        instruction_id: 127
        bytes_out: 40
        communication_groups {
          group_ids: 5
          group_ids: 4
          group_ids: 1
          group_ids: 0
        }
        inner_subroutines {
          name: "reduce-scatter_stage-1_dim-1_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 139
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_stage-1_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 128
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-1_bidir-ring_ccw_unidir-ring_reduction_1"
            opcode: "call"
            instruction_id: 129
            operand_ids: 128
            inner_subroutines {
              name: "reduction_subroutine_phase_1"
              subroutine_root_id: 132
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_1"
                opcode: "delay"
                instruction_id: 130
                bytes_out: 10
              }
              instructions {
                name: "op2_phase_1"
                opcode: "delay"
                instruction_id: 131
                bytes_out: 10
              }
              instructions {
                name: "sum_phase_1"
                opcode: "delay"
                instruction_id: 132
                ops: 20
                operand_ids: 130
                operand_ids: 131
              }
            }
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 133
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
            operand_ids: 129
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-1_bidir-ring_ccw_unidir-ring_reduction_2"
            opcode: "call"
            instruction_id: 134
            operand_ids: 133
            inner_subroutines {
              name: "reduction_subroutine_phase_2"
              subroutine_root_id: 137
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_2"
                opcode: "delay"
                instruction_id: 135
                bytes_out: 10
              }
              instructions {
                name: "op2_phase_2"
                opcode: "delay"
                instruction_id: 136
                bytes_out: 10
              }
              instructions {
                name: "sum_phase_2"
                opcode: "delay"
                instruction_id: 137
                ops: 20
                operand_ids: 135
                operand_ids: 136
              }
            }
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_3"
            opcode: "sendrecv"
            instruction_id: 138
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
            operand_ids: 134
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-1_bidir-ring_ccw_unidir-ring_reduction_3"
            opcode: "call"
            instruction_id: 139
            operand_ids: 138
            inner_subroutines {
              name: "reduction_subroutine_phase_3"
              subroutine_root_id: 142
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_3"
                opcode: "delay"
                instruction_id: 140
                bytes_out: 10
              }
              instructions {
                name: "op2_phase_3"
                opcode: "delay"
                instruction_id: 141
                bytes_out: 10
              }
              instructions {
                name: "sum_phase_3"
                opcode: "delay"
                instruction_id: 142
                ops: 20
                operand_ids: 140
                operand_ids: 141
              }
            }
          }
        }
      }
      instructions {
        name: "reduce-scatter_stage-1_dim-1_bidir-ring_root_1"
        opcode: "null"
        instruction_id: 143
        operand_ids: 111
        operand_ids: 127
      }
    }
  }
  instructions {
    name: "reduce-scatter_stage-1_root"
    opcode: "null"
    instruction_id: 144
    operand_ids: 76
    operand_ids: 110
  }
}
    )proto";
  google::protobuf::TextFormat::ParseFromString(test_str,
                                                &proto);
  return proto;
}  // NOLINT

// Tests expanding 2D-Torus reduce-scatter
TEST(Torus2dReduceScatter, NoBarrier) {
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
        "algorithm": "torus-2d",
        "concentration": 2,
        "dimension_widths": [2, 2],
        "integrated_local_exchange": true
      }
    }
  )"_json;

  ASSERT_OK_AND_ASSIGN(auto translators, paragraph::CreateTranslators(
     paragraph::TranslatorType::kCollective, config));
  EXPECT_OK(translators["reduce-scatter"]->Translate(reducescatter));

  paragraph::InstructionProto reducescatter_proto = no_barrier_test_proto();
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      reducescatter->ToProto().value(), reducescatter_proto));
}

paragraph::InstructionProto with_barrier_test_proto() {
  paragraph::InstructionProto proto;
  std::string test_str =
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
  name: "reduce-scatter_torus-2d"
  subroutine_root_id: 79
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "reduce-scatter_stage-0_dim-0"
    opcode: "reduce-scatter"
    instruction_id: 7
    bytes_out: 20
    communication_groups {
      group_ids: 0
      group_ids: 2
    }
    inner_subroutines {
      name: "reduce-scatter_stage-0_dim-0_bidir-ring"
      subroutine_root_id: 23
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "reduce-scatter_stage-0_dim-0_bidir-ring_barrier"
        opcode: "barrier"
        instruction_id: 8
        communication_groups {
          group_ids: 0
          group_ids: 2
        }
        inner_subroutines {
          name: "reduce-scatter_stage-0_dim-0_bidir-ring_barrier_centralized"
          subroutine_root_id: 10
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_stage-0_dim-0_bidir-ring_barrier_centralized_send_to_0"
            opcode: "send"
            instruction_id: 9
            communication_groups {
              group_ids: 0
            }
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-0_bidir-ring_barrier_centralized_recv_from_0"
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
        name: "reduce-scatter_stage-0_dim-0_bidir-ring_cw"
        opcode: "reduce-scatter"
        instruction_id: 11
        bytes_out: 10
        communication_groups {
          group_ids: 0
          group_ids: 2
        }
        operand_ids: 8
        inner_subroutines {
          name: "reduce-scatter_stage-0_dim-0_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 13
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_stage-0_dim-0_bidir-ring_cw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 12
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 0
            }
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-0_bidir-ring_cw_unidir-ring_reduction_1"
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
                bytes_out: 5
              }
              instructions {
                name: "op2_phase_1"
                opcode: "delay"
                instruction_id: 15
                bytes_out: 5
              }
              instructions {
                name: "sum_phase_1"
                opcode: "delay"
                instruction_id: 16
                ops: 10
                operand_ids: 14
                operand_ids: 15
              }
            }
          }
        }
      }
      instructions {
        name: "reduce-scatter_stage-0_dim-0_bidir-ring_ccw"
        opcode: "reduce-scatter"
        instruction_id: 17
        bytes_out: 10
        communication_groups {
          group_ids: 2
          group_ids: 0
        }
        operand_ids: 8
        inner_subroutines {
          name: "reduce-scatter_stage-0_dim-0_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 19
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_stage-0_dim-0_bidir-ring_ccw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 18
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 0
              group_ids: 0
            }
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-0_bidir-ring_ccw_unidir-ring_reduction_1"
            opcode: "call"
            instruction_id: 19
            operand_ids: 18
            inner_subroutines {
              name: "reduction_subroutine_phase_1"
              subroutine_root_id: 22
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_1"
                opcode: "delay"
                instruction_id: 20
                bytes_out: 5
              }
              instructions {
                name: "op2_phase_1"
                opcode: "delay"
                instruction_id: 21
                bytes_out: 5
              }
              instructions {
                name: "sum_phase_1"
                opcode: "delay"
                instruction_id: 22
                ops: 10
                operand_ids: 20
                operand_ids: 21
              }
            }
          }
        }
      }
      instructions {
        name: "reduce-scatter_stage-0_dim-0_bidir-ring_root_2"
        opcode: "null"
        instruction_id: 23
        operand_ids: 11
        operand_ids: 17
      }
    }
  }
  instructions {
    name: "reduce-scatter_stage-0_dim-1"
    opcode: "reduce-scatter"
    instruction_id: 24
    bytes_out: 20
    communication_groups {
      group_ids: 2
      group_ids: 6
    }
    inner_subroutines {
      name: "reduce-scatter_stage-0_dim-1_bidir-ring"
      subroutine_root_id: 41
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "reduce-scatter_stage-0_dim-1_bidir-ring_barrier"
        opcode: "barrier"
        instruction_id: 25
        communication_groups {
          group_ids: 2
          group_ids: 6
        }
        inner_subroutines {
          name: "reduce-scatter_stage-0_dim-1_bidir-ring_barrier_centralized"
          subroutine_root_id: 28
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_stage-0_dim-1_bidir-ring_barrier_centralized_coordinator_recv_from_6"
            opcode: "recv"
            instruction_id: 26
            communication_groups {
              group_ids: 6
            }
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-1_bidir-ring_barrier_centralized_coordinator_send_to_6"
            opcode: "send"
            instruction_id: 27
            communication_groups {
              group_ids: 6
            }
            operand_ids: 26
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-1_bidir-ring_barrier_centralized_root_2"
            opcode: "null"
            instruction_id: 28
            operand_ids: 27
          }
        }
      }
      instructions {
        name: "reduce-scatter_stage-0_dim-1_bidir-ring_cw"
        opcode: "reduce-scatter"
        instruction_id: 29
        bytes_out: 10
        communication_groups {
          group_ids: 2
          group_ids: 6
        }
        operand_ids: 25
        inner_subroutines {
          name: "reduce-scatter_stage-0_dim-1_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 31
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_stage-0_dim-1_bidir-ring_cw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 30
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 6
              group_ids: 6
            }
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-1_bidir-ring_cw_unidir-ring_reduction_1"
            opcode: "call"
            instruction_id: 31
            operand_ids: 30
            inner_subroutines {
              name: "reduction_subroutine_phase_1"
              subroutine_root_id: 34
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_1"
                opcode: "delay"
                instruction_id: 32
                bytes_out: 5
              }
              instructions {
                name: "op2_phase_1"
                opcode: "delay"
                instruction_id: 33
                bytes_out: 5
              }
              instructions {
                name: "sum_phase_1"
                opcode: "delay"
                instruction_id: 34
                ops: 10
                operand_ids: 32
                operand_ids: 33
              }
            }
          }
        }
      }
      instructions {
        name: "reduce-scatter_stage-0_dim-1_bidir-ring_ccw"
        opcode: "reduce-scatter"
        instruction_id: 35
        bytes_out: 10
        communication_groups {
          group_ids: 6
          group_ids: 2
        }
        operand_ids: 25
        inner_subroutines {
          name: "reduce-scatter_stage-0_dim-1_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 37
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_stage-0_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 36
            bytes_in: 5
            bytes_out: 5
            communication_groups {
              group_ids: 6
              group_ids: 6
            }
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-1_bidir-ring_ccw_unidir-ring_reduction_1"
            opcode: "call"
            instruction_id: 37
            operand_ids: 36
            inner_subroutines {
              name: "reduction_subroutine_phase_1"
              subroutine_root_id: 40
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_1"
                opcode: "delay"
                instruction_id: 38
                bytes_out: 5
              }
              instructions {
                name: "op2_phase_1"
                opcode: "delay"
                instruction_id: 39
                bytes_out: 5
              }
              instructions {
                name: "sum_phase_1"
                opcode: "delay"
                instruction_id: 40
                ops: 10
                operand_ids: 38
                operand_ids: 39
              }
            }
          }
        }
      }
      instructions {
        name: "reduce-scatter_stage-0_dim-1_bidir-ring_root_2"
        opcode: "null"
        instruction_id: 41
        operand_ids: 29
        operand_ids: 35
      }
    }
  }
  instructions {
    name: "reduce-scatter_stage-0_root"
    opcode: "null"
    instruction_id: 42
    operand_ids: 7
    operand_ids: 24
  }
  instructions {
    name: "reduce-scatter_stage-1_dim-0"
    opcode: "reduce-scatter"
    instruction_id: 43
    bytes_out: 40
    communication_groups {
      group_ids: 0
      group_ids: 2
    }
    operand_ids: 42
    inner_subroutines {
      name: "reduce-scatter_stage-1_dim-0_bidir-ring"
      subroutine_root_id: 59
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "reduce-scatter_stage-1_dim-0_bidir-ring_barrier"
        opcode: "barrier"
        instruction_id: 44
        communication_groups {
          group_ids: 0
          group_ids: 2
        }
        inner_subroutines {
          name: "reduce-scatter_stage-1_dim-0_bidir-ring_barrier_centralized"
          subroutine_root_id: 46
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_stage-1_dim-0_bidir-ring_barrier_centralized_send_to_0"
            opcode: "send"
            instruction_id: 45
            communication_groups {
              group_ids: 0
            }
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-0_bidir-ring_barrier_centralized_recv_from_0"
            opcode: "recv"
            instruction_id: 46
            communication_groups {
              group_ids: 0
            }
            operand_ids: 45
          }
        }
      }
      instructions {
        name: "reduce-scatter_stage-1_dim-0_bidir-ring_cw"
        opcode: "reduce-scatter"
        instruction_id: 47
        bytes_out: 20
        communication_groups {
          group_ids: 0
          group_ids: 2
        }
        operand_ids: 44
        inner_subroutines {
          name: "reduce-scatter_stage-1_dim-0_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 49
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_stage-1_dim-0_bidir-ring_cw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 48
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 0
              group_ids: 0
            }
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-0_bidir-ring_cw_unidir-ring_reduction_1"
            opcode: "call"
            instruction_id: 49
            operand_ids: 48
            inner_subroutines {
              name: "reduction_subroutine_phase_1"
              subroutine_root_id: 52
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_1"
                opcode: "delay"
                instruction_id: 50
                bytes_out: 10
              }
              instructions {
                name: "op2_phase_1"
                opcode: "delay"
                instruction_id: 51
                bytes_out: 10
              }
              instructions {
                name: "sum_phase_1"
                opcode: "delay"
                instruction_id: 52
                ops: 20
                operand_ids: 50
                operand_ids: 51
              }
            }
          }
        }
      }
      instructions {
        name: "reduce-scatter_stage-1_dim-0_bidir-ring_ccw"
        opcode: "reduce-scatter"
        instruction_id: 53
        bytes_out: 20
        communication_groups {
          group_ids: 2
          group_ids: 0
        }
        operand_ids: 44
        inner_subroutines {
          name: "reduce-scatter_stage-1_dim-0_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 55
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_stage-1_dim-0_bidir-ring_ccw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 54
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 0
              group_ids: 0
            }
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-0_bidir-ring_ccw_unidir-ring_reduction_1"
            opcode: "call"
            instruction_id: 55
            operand_ids: 54
            inner_subroutines {
              name: "reduction_subroutine_phase_1"
              subroutine_root_id: 58
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_1"
                opcode: "delay"
                instruction_id: 56
                bytes_out: 10
              }
              instructions {
                name: "op2_phase_1"
                opcode: "delay"
                instruction_id: 57
                bytes_out: 10
              }
              instructions {
                name: "sum_phase_1"
                opcode: "delay"
                instruction_id: 58
                ops: 20
                operand_ids: 56
                operand_ids: 57
              }
            }
          }
        }
      }
      instructions {
        name: "reduce-scatter_stage-1_dim-0_bidir-ring_root_2"
        opcode: "null"
        instruction_id: 59
        operand_ids: 47
        operand_ids: 53
      }
    }
  }
  instructions {
    name: "reduce-scatter_stage-1_dim-1"
    opcode: "reduce-scatter"
    instruction_id: 60
    bytes_out: 40
    communication_groups {
      group_ids: 2
      group_ids: 6
    }
    operand_ids: 42
    inner_subroutines {
      name: "reduce-scatter_stage-1_dim-1_bidir-ring"
      subroutine_root_id: 77
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "reduce-scatter_stage-1_dim-1_bidir-ring_barrier"
        opcode: "barrier"
        instruction_id: 61
        communication_groups {
          group_ids: 2
          group_ids: 6
        }
        inner_subroutines {
          name: "reduce-scatter_stage-1_dim-1_bidir-ring_barrier_centralized"
          subroutine_root_id: 64
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_stage-1_dim-1_bidir-ring_barrier_centralized_coordinator_recv_from_6"
            opcode: "recv"
            instruction_id: 62
            communication_groups {
              group_ids: 6
            }
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-1_bidir-ring_barrier_centralized_coordinator_send_to_6"
            opcode: "send"
            instruction_id: 63
            communication_groups {
              group_ids: 6
            }
            operand_ids: 62
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-1_bidir-ring_barrier_centralized_root_2"
            opcode: "null"
            instruction_id: 64
            operand_ids: 63
          }
        }
      }
      instructions {
        name: "reduce-scatter_stage-1_dim-1_bidir-ring_cw"
        opcode: "reduce-scatter"
        instruction_id: 65
        bytes_out: 20
        communication_groups {
          group_ids: 2
          group_ids: 6
        }
        operand_ids: 61
        inner_subroutines {
          name: "reduce-scatter_stage-1_dim-1_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 67
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_stage-1_dim-1_bidir-ring_cw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 66
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 6
              group_ids: 6
            }
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-1_bidir-ring_cw_unidir-ring_reduction_1"
            opcode: "call"
            instruction_id: 67
            operand_ids: 66
            inner_subroutines {
              name: "reduction_subroutine_phase_1"
              subroutine_root_id: 70
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_1"
                opcode: "delay"
                instruction_id: 68
                bytes_out: 10
              }
              instructions {
                name: "op2_phase_1"
                opcode: "delay"
                instruction_id: 69
                bytes_out: 10
              }
              instructions {
                name: "sum_phase_1"
                opcode: "delay"
                instruction_id: 70
                ops: 20
                operand_ids: 68
                operand_ids: 69
              }
            }
          }
        }
      }
      instructions {
        name: "reduce-scatter_stage-1_dim-1_bidir-ring_ccw"
        opcode: "reduce-scatter"
        instruction_id: 71
        bytes_out: 20
        communication_groups {
          group_ids: 6
          group_ids: 2
        }
        operand_ids: 61
        inner_subroutines {
          name: "reduce-scatter_stage-1_dim-1_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 73
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_stage-1_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 72
            bytes_in: 10
            bytes_out: 10
            communication_groups {
              group_ids: 6
              group_ids: 6
            }
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-1_bidir-ring_ccw_unidir-ring_reduction_1"
            opcode: "call"
            instruction_id: 73
            operand_ids: 72
            inner_subroutines {
              name: "reduction_subroutine_phase_1"
              subroutine_root_id: 76
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_1"
                opcode: "delay"
                instruction_id: 74
                bytes_out: 10
              }
              instructions {
                name: "op2_phase_1"
                opcode: "delay"
                instruction_id: 75
                bytes_out: 10
              }
              instructions {
                name: "sum_phase_1"
                opcode: "delay"
                instruction_id: 76
                ops: 20
                operand_ids: 74
                operand_ids: 75
              }
            }
          }
        }
      }
      instructions {
        name: "reduce-scatter_stage-1_dim-1_bidir-ring_root_2"
        opcode: "null"
        instruction_id: 77
        operand_ids: 65
        operand_ids: 71
      }
    }
  }
  instructions {
    name: "reduce-scatter_stage-1_root"
    opcode: "null"
    instruction_id: 78
    operand_ids: 43
    operand_ids: 60
  }
  instructions {
    name: "reduce-scatter_conc"
    opcode: "reduce-scatter"
    instruction_id: 79
    bytes_out: 80
    communication_groups {
      group_ids: 2
      group_ids: 3
    }
    operand_ids: 78
    inner_subroutines {
      name: "reduce-scatter_conc_bidir-ring"
      subroutine_root_id: 96
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "reduce-scatter_conc_bidir-ring_barrier"
        opcode: "barrier"
        instruction_id: 80
        communication_groups {
          group_ids: 2
          group_ids: 3
        }
        inner_subroutines {
          name: "reduce-scatter_conc_bidir-ring_barrier_centralized"
          subroutine_root_id: 83
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_conc_bidir-ring_barrier_centralized_coordinator_recv_from_3"
            opcode: "recv"
            instruction_id: 81
            communication_groups {
              group_ids: 3
            }
          }
          instructions {
            name: "reduce-scatter_conc_bidir-ring_barrier_centralized_coordinator_send_to_3"
            opcode: "send"
            instruction_id: 82
            communication_groups {
              group_ids: 3
            }
            operand_ids: 81
          }
          instructions {
            name: "reduce-scatter_conc_bidir-ring_barrier_centralized_root_2"
            opcode: "null"
            instruction_id: 83
            operand_ids: 82
          }
        }
      }
      instructions {
        name: "reduce-scatter_conc_bidir-ring_cw"
        opcode: "reduce-scatter"
        instruction_id: 84
        bytes_out: 40
        communication_groups {
          group_ids: 2
          group_ids: 3
        }
        operand_ids: 80
        inner_subroutines {
          name: "reduce-scatter_conc_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 86
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_conc_bidir-ring_cw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 85
            bytes_in: 20
            bytes_out: 20
            communication_groups {
              group_ids: 3
              group_ids: 3
            }
          }
          instructions {
            name: "reduce-scatter_conc_bidir-ring_cw_unidir-ring_reduction_1"
            opcode: "call"
            instruction_id: 86
            operand_ids: 85
            inner_subroutines {
              name: "reduction_subroutine_phase_1"
              subroutine_root_id: 89
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_1"
                opcode: "delay"
                instruction_id: 87
                bytes_out: 20
              }
              instructions {
                name: "op2_phase_1"
                opcode: "delay"
                instruction_id: 88
                bytes_out: 20
              }
              instructions {
                name: "sum_phase_1"
                opcode: "delay"
                instruction_id: 89
                ops: 40
                operand_ids: 87
                operand_ids: 88
              }
            }
          }
        }
      }
      instructions {
        name: "reduce-scatter_conc_bidir-ring_ccw"
        opcode: "reduce-scatter"
        instruction_id: 90
        bytes_out: 40
        communication_groups {
          group_ids: 3
          group_ids: 2
        }
        operand_ids: 80
        inner_subroutines {
          name: "reduce-scatter_conc_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 92
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_conc_bidir-ring_ccw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 91
            bytes_in: 20
            bytes_out: 20
            communication_groups {
              group_ids: 3
              group_ids: 3
            }
          }
          instructions {
            name: "reduce-scatter_conc_bidir-ring_ccw_unidir-ring_reduction_1"
            opcode: "call"
            instruction_id: 92
            operand_ids: 91
            inner_subroutines {
              name: "reduction_subroutine_phase_1"
              subroutine_root_id: 95
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_1"
                opcode: "delay"
                instruction_id: 93
                bytes_out: 20
              }
              instructions {
                name: "op2_phase_1"
                opcode: "delay"
                instruction_id: 94
                bytes_out: 20
              }
              instructions {
                name: "sum_phase_1"
                opcode: "delay"
                instruction_id: 95
                ops: 40
                operand_ids: 93
                operand_ids: 94
              }
            }
          }
        }
      }
      instructions {
        name: "reduce-scatter_conc_bidir-ring_root_2"
        opcode: "null"
        instruction_id: 96
        operand_ids: 84
        operand_ids: 90
      }
    }
  }
}
    )proto";
  google::protobuf::TextFormat::ParseFromString(test_str,
                                                &proto);
  return proto;
}  // NOLINT

// Tests expanding 1D-Torus reduce-scatter with barrier
TEST(Torus2dReduceScatter, WithBarrier) {
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
  EXPECT_OK(translators["reduce-scatter"]->Translate(reducescatter));

  paragraph::InstructionProto reducescatter_proto = with_barrier_test_proto();
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      reducescatter->ToProto().value(), reducescatter_proto));
}

paragraph::InstructionProto inconsecutive_proc_test_proto() {
  paragraph::InstructionProto proto;
  std::string test_str =
      R"proto(
name: "reduce-scatter"
opcode: "reduce-scatter"
instruction_id: 2
bytes_out: 48
communication_groups {
  group_ids: 0
  group_ids: 2
  group_ids: 4
}
inner_subroutines {
  name: "reduce-scatter_torus-2d"
  subroutine_root_id: 56
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "reduce-scatter_stage-0_dim-1"
    opcode: "reduce-scatter"
    instruction_id: 7
    bytes_out: 48
    communication_groups {
      group_ids: 0
      group_ids: 2
      group_ids: 4
    }
    inner_subroutines {
      name: "reduce-scatter_stage-0_dim-1_bidir-ring"
      subroutine_root_id: 30
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "reduce-scatter_stage-0_dim-1_bidir-ring_cw"
        opcode: "reduce-scatter"
        instruction_id: 8
        bytes_out: 24
        communication_groups {
          group_ids: 0
          group_ids: 2
          group_ids: 4
        }
        inner_subroutines {
          name: "reduce-scatter_stage-0_dim-1_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 15
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_stage-0_dim-1_bidir-ring_cw_unidir-ring_sendrecv_1"
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
            name: "reduce-scatter_stage-0_dim-1_bidir-ring_cw_unidir-ring_reduction_1"
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
            name: "reduce-scatter_stage-0_dim-1_bidir-ring_cw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 14
            bytes_in: 8
            bytes_out: 8
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
            operand_ids: 10
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-1_bidir-ring_cw_unidir-ring_reduction_2"
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
        name: "reduce-scatter_stage-0_dim-1_bidir-ring_ccw"
        opcode: "reduce-scatter"
        instruction_id: 19
        bytes_out: 24
        communication_groups {
          group_ids: 4
          group_ids: 2
          group_ids: 0
        }
        inner_subroutines {
          name: "reduce-scatter_stage-0_dim-1_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 26
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_stage-0_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 20
            bytes_in: 8
            bytes_out: 8
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-1_bidir-ring_ccw_unidir-ring_reduction_1"
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
            name: "reduce-scatter_stage-0_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 25
            bytes_in: 8
            bytes_out: 8
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
            operand_ids: 21
          }
          instructions {
            name: "reduce-scatter_stage-0_dim-1_bidir-ring_ccw_unidir-ring_reduction_2"
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
        name: "reduce-scatter_stage-0_dim-1_bidir-ring_root_2"
        opcode: "null"
        instruction_id: 30
        operand_ids: 8
        operand_ids: 19
      }
    }
  }
  instructions {
    name: "reduce-scatter_stage-0_root"
    opcode: "null"
    instruction_id: 31
    operand_ids: 7
  }
  instructions {
    name: "reduce-scatter_stage-1_dim-1"
    opcode: "reduce-scatter"
    instruction_id: 32
    bytes_out: 144
    communication_groups {
      group_ids: 0
      group_ids: 2
      group_ids: 4
    }
    operand_ids: 31
    inner_subroutines {
      name: "reduce-scatter_stage-1_dim-1_bidir-ring"
      subroutine_root_id: 55
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "reduce-scatter_stage-1_dim-1_bidir-ring_cw"
        opcode: "reduce-scatter"
        instruction_id: 33
        bytes_out: 72
        communication_groups {
          group_ids: 0
          group_ids: 2
          group_ids: 4
        }
        inner_subroutines {
          name: "reduce-scatter_stage-1_dim-1_bidir-ring_cw_unidir-ring"
          subroutine_root_id: 40
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_stage-1_dim-1_bidir-ring_cw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 34
            bytes_in: 24
            bytes_out: 24
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-1_bidir-ring_cw_unidir-ring_reduction_1"
            opcode: "call"
            instruction_id: 35
            operand_ids: 34
            inner_subroutines {
              name: "reduction_subroutine_phase_1"
              subroutine_root_id: 38
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_1"
                opcode: "delay"
                instruction_id: 36
                bytes_out: 24
              }
              instructions {
                name: "op2_phase_1"
                opcode: "delay"
                instruction_id: 37
                bytes_out: 24
              }
              instructions {
                name: "sum_phase_1"
                opcode: "delay"
                instruction_id: 38
                ops: 48
                operand_ids: 36
                operand_ids: 37
              }
            }
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-1_bidir-ring_cw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 39
            bytes_in: 24
            bytes_out: 24
            communication_groups {
              group_ids: 0
              group_ids: 4
            }
            operand_ids: 35
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-1_bidir-ring_cw_unidir-ring_reduction_2"
            opcode: "call"
            instruction_id: 40
            operand_ids: 39
            inner_subroutines {
              name: "reduction_subroutine_phase_2"
              subroutine_root_id: 43
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "op1_phase_2"
                opcode: "delay"
                instruction_id: 41
                bytes_out: 24
              }
              instructions {
                name: "op2_phase_2"
                opcode: "delay"
                instruction_id: 42
                bytes_out: 24
              }
              instructions {
                name: "sum_phase_2"
                opcode: "delay"
                instruction_id: 43
                ops: 48
                operand_ids: 41
                operand_ids: 42
              }
            }
          }
        }
      }
      instructions {
        name: "reduce-scatter_stage-1_dim-1_bidir-ring_ccw"
        opcode: "reduce-scatter"
        instruction_id: 44
        bytes_out: 72
        communication_groups {
          group_ids: 4
          group_ids: 2
          group_ids: 0
        }
        inner_subroutines {
          name: "reduce-scatter_stage-1_dim-1_bidir-ring_ccw_unidir-ring"
          subroutine_root_id: 51
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "reduce-scatter_stage-1_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_1"
            opcode: "sendrecv"
            instruction_id: 45
            bytes_in: 24
            bytes_out: 24
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-1_bidir-ring_ccw_unidir-ring_reduction_1"
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
                bytes_out: 24
              }
              instructions {
                name: "op2_phase_1"
                opcode: "delay"
                instruction_id: 48
                bytes_out: 24
              }
              instructions {
                name: "sum_phase_1"
                opcode: "delay"
                instruction_id: 49
                ops: 48
                operand_ids: 47
                operand_ids: 48
              }
            }
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_2"
            opcode: "sendrecv"
            instruction_id: 50
            bytes_in: 24
            bytes_out: 24
            communication_groups {
              group_ids: 4
              group_ids: 0
            }
            operand_ids: 46
          }
          instructions {
            name: "reduce-scatter_stage-1_dim-1_bidir-ring_ccw_unidir-ring_reduction_2"
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
                bytes_out: 24
              }
              instructions {
                name: "op2_phase_2"
                opcode: "delay"
                instruction_id: 53
                bytes_out: 24
              }
              instructions {
                name: "sum_phase_2"
                opcode: "delay"
                instruction_id: 54
                ops: 48
                operand_ids: 52
                operand_ids: 53
              }
            }
          }
        }
      }
      instructions {
        name: "reduce-scatter_stage-1_dim-1_bidir-ring_root_2"
        opcode: "null"
        instruction_id: 55
        operand_ids: 33
        operand_ids: 44
      }
    }
  }
  instructions {
    name: "reduce-scatter_stage-1_root"
    opcode: "null"
    instruction_id: 56
    operand_ids: 32
  }
}
    )proto";
  google::protobuf::TextFormat::ParseFromString(test_str,
                                                &proto);
  return proto;
}  // NOLINT

// Tests expanding 1D-Torus reduce-scatter
TEST(Torus2dReduceScatter, InconsecutiveProcessors) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 2);
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
  paragraph::CommunicationGroup reducescatter_group = {0, 2, 4};
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
        "algorithm": "torus-2d",
        "dimension_widths": [2, 3]
      }
    }
  )"_json;

  ASSERT_OK_AND_ASSIGN(auto translators, paragraph::CreateTranslators(
     paragraph::TranslatorType::kCollective, config));
  EXPECT_OK(translators["reduce-scatter"]->Translate(reducescatter));

  paragraph::InstructionProto reducescatter_proto =
      inconsecutive_proc_test_proto();
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      reducescatter->ToProto().value(), reducescatter_proto));
}
