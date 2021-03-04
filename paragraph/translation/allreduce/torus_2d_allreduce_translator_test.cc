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
#include "paragraph/translation/allreduce/torus_2d_allreduce_translator.h"

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
  name: "all-reduce_torus-2d"
  subroutine_root_id: 80
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "all-reduce_torus-2d_reduce-scatter"
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
      name: "all-reduce_torus-2d_reduce-scatter_torus-2d"
      subroutine_root_id: 66
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-0"
        opcode: "reduce-scatter"
        instruction_id: 8
        bytes_out: 12
        communication_groups {
          group_ids: 0
          group_ids: 2
        }
        inner_subroutines {
          name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-0_bidir-ring"
          subroutine_root_id: 21
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-0_bidir-ring_cw"
            opcode: "reduce-scatter"
            instruction_id: 9
            bytes_out: 6
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-0_bidir-ring_cw_unidir-ring"
              subroutine_root_id: 11
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-0_bidir-ring_cw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 10
                bytes_in: 3
                bytes_out: 3
                communication_groups {
                  group_ids: 0
                  group_ids: 0
                }
              }
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-0_bidir-ring_cw_unidir-ring_reduction_1"
                opcode: "call"
                instruction_id: 11
                operand_ids: 10
                inner_subroutines {
                  name: "reduction_subroutine_phase_1"
                  subroutine_root_id: 14
                  execution_probability: 1
                  execution_count: 1
                  instructions {
                    name: "op1_phase_1"
                    opcode: "delay"
                    instruction_id: 12
                    bytes_out: 3
                  }
                  instructions {
                    name: "op2_phase_1"
                    opcode: "delay"
                    instruction_id: 13
                    bytes_out: 3
                  }
                  instructions {
                    name: "sum_phase_1"
                    opcode: "delay"
                    instruction_id: 14
                    ops: 6
                    operand_ids: 12
                    operand_ids: 13
                  }
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-0_bidir-ring_ccw"
            opcode: "reduce-scatter"
            instruction_id: 15
            bytes_out: 6
            communication_groups {
              group_ids: 2
              group_ids: 0
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-0_bidir-ring_ccw_unidir-ring"
              subroutine_root_id: 17
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-0_bidir-ring_ccw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 16
                bytes_in: 3
                bytes_out: 3
                communication_groups {
                  group_ids: 0
                  group_ids: 0
                }
              }
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-0_bidir-ring_ccw_unidir-ring_reduction_1"
                opcode: "call"
                instruction_id: 17
                operand_ids: 16
                inner_subroutines {
                  name: "reduction_subroutine_phase_1"
                  subroutine_root_id: 20
                  execution_probability: 1
                  execution_count: 1
                  instructions {
                    name: "op1_phase_1"
                    opcode: "delay"
                    instruction_id: 18
                    bytes_out: 3
                  }
                  instructions {
                    name: "op2_phase_1"
                    opcode: "delay"
                    instruction_id: 19
                    bytes_out: 3
                  }
                  instructions {
                    name: "sum_phase_1"
                    opcode: "delay"
                    instruction_id: 20
                    ops: 6
                    operand_ids: 18
                    operand_ids: 19
                  }
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-0_bidir-ring_root_2"
            opcode: "null"
            instruction_id: 21
            operand_ids: 9
            operand_ids: 15
          }
        }
      }
      instructions {
        name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-1"
        opcode: "reduce-scatter"
        instruction_id: 22
        bytes_out: 12
        communication_groups {
          group_ids: 2
          group_ids: 6
        }
        inner_subroutines {
          name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-1_bidir-ring"
          subroutine_root_id: 35
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-1_bidir-ring_cw"
            opcode: "reduce-scatter"
            instruction_id: 23
            bytes_out: 6
            communication_groups {
              group_ids: 2
              group_ids: 6
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-1_bidir-ring_cw_unidir-ring"
              subroutine_root_id: 25
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-1_bidir-ring_cw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 24
                bytes_in: 3
                bytes_out: 3
                communication_groups {
                  group_ids: 6
                  group_ids: 6
                }
              }
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-1_bidir-ring_cw_unidir-ring_reduction_1"
                opcode: "call"
                instruction_id: 25
                operand_ids: 24
                inner_subroutines {
                  name: "reduction_subroutine_phase_1"
                  subroutine_root_id: 28
                  execution_probability: 1
                  execution_count: 1
                  instructions {
                    name: "op1_phase_1"
                    opcode: "delay"
                    instruction_id: 26
                    bytes_out: 3
                  }
                  instructions {
                    name: "op2_phase_1"
                    opcode: "delay"
                    instruction_id: 27
                    bytes_out: 3
                  }
                  instructions {
                    name: "sum_phase_1"
                    opcode: "delay"
                    instruction_id: 28
                    ops: 6
                    operand_ids: 26
                    operand_ids: 27
                  }
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-1_bidir-ring_ccw"
            opcode: "reduce-scatter"
            instruction_id: 29
            bytes_out: 6
            communication_groups {
              group_ids: 6
              group_ids: 2
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-1_bidir-ring_ccw_unidir-ring"
              subroutine_root_id: 31
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 30
                bytes_in: 3
                bytes_out: 3
                communication_groups {
                  group_ids: 6
                  group_ids: 6
                }
              }
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-1_bidir-ring_ccw_unidir-ring_reduction_1"
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
                    bytes_out: 3
                  }
                  instructions {
                    name: "op2_phase_1"
                    opcode: "delay"
                    instruction_id: 33
                    bytes_out: 3
                  }
                  instructions {
                    name: "sum_phase_1"
                    opcode: "delay"
                    instruction_id: 34
                    ops: 6
                    operand_ids: 32
                    operand_ids: 33
                  }
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-1_bidir-ring_root_2"
            opcode: "null"
            instruction_id: 35
            operand_ids: 23
            operand_ids: 29
          }
        }
      }
      instructions {
        name: "all-reduce_torus-2d_reduce-scatter_stage-0_root"
        opcode: "null"
        instruction_id: 36
        operand_ids: 8
        operand_ids: 22
      }
      instructions {
        name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-0"
        opcode: "reduce-scatter"
        instruction_id: 37
        bytes_out: 24
        communication_groups {
          group_ids: 0
          group_ids: 2
        }
        operand_ids: 36
        inner_subroutines {
          name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-0_bidir-ring"
          subroutine_root_id: 50
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-0_bidir-ring_cw"
            opcode: "reduce-scatter"
            instruction_id: 38
            bytes_out: 12
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-0_bidir-ring_cw_unidir-ring"
              subroutine_root_id: 40
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-0_bidir-ring_cw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 39
                bytes_in: 6
                bytes_out: 6
                communication_groups {
                  group_ids: 0
                  group_ids: 0
                }
              }
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-0_bidir-ring_cw_unidir-ring_reduction_1"
                opcode: "call"
                instruction_id: 40
                operand_ids: 39
                inner_subroutines {
                  name: "reduction_subroutine_phase_1"
                  subroutine_root_id: 43
                  execution_probability: 1
                  execution_count: 1
                  instructions {
                    name: "op1_phase_1"
                    opcode: "delay"
                    instruction_id: 41
                    bytes_out: 6
                  }
                  instructions {
                    name: "op2_phase_1"
                    opcode: "delay"
                    instruction_id: 42
                    bytes_out: 6
                  }
                  instructions {
                    name: "sum_phase_1"
                    opcode: "delay"
                    instruction_id: 43
                    ops: 12
                    operand_ids: 41
                    operand_ids: 42
                  }
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-0_bidir-ring_ccw"
            opcode: "reduce-scatter"
            instruction_id: 44
            bytes_out: 12
            communication_groups {
              group_ids: 2
              group_ids: 0
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-0_bidir-ring_ccw_unidir-ring"
              subroutine_root_id: 46
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-0_bidir-ring_ccw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 45
                bytes_in: 6
                bytes_out: 6
                communication_groups {
                  group_ids: 0
                  group_ids: 0
                }
              }
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-0_bidir-ring_ccw_unidir-ring_reduction_1"
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
                    bytes_out: 6
                  }
                  instructions {
                    name: "op2_phase_1"
                    opcode: "delay"
                    instruction_id: 48
                    bytes_out: 6
                  }
                  instructions {
                    name: "sum_phase_1"
                    opcode: "delay"
                    instruction_id: 49
                    ops: 12
                    operand_ids: 47
                    operand_ids: 48
                  }
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-0_bidir-ring_root_2"
            opcode: "null"
            instruction_id: 50
            operand_ids: 38
            operand_ids: 44
          }
        }
      }
      instructions {
        name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-1"
        opcode: "reduce-scatter"
        instruction_id: 51
        bytes_out: 24
        communication_groups {
          group_ids: 2
          group_ids: 6
        }
        operand_ids: 36
        inner_subroutines {
          name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-1_bidir-ring"
          subroutine_root_id: 64
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-1_bidir-ring_cw"
            opcode: "reduce-scatter"
            instruction_id: 52
            bytes_out: 12
            communication_groups {
              group_ids: 2
              group_ids: 6
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-1_bidir-ring_cw_unidir-ring"
              subroutine_root_id: 54
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-1_bidir-ring_cw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 53
                bytes_in: 6
                bytes_out: 6
                communication_groups {
                  group_ids: 6
                  group_ids: 6
                }
              }
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-1_bidir-ring_cw_unidir-ring_reduction_1"
                opcode: "call"
                instruction_id: 54
                operand_ids: 53
                inner_subroutines {
                  name: "reduction_subroutine_phase_1"
                  subroutine_root_id: 57
                  execution_probability: 1
                  execution_count: 1
                  instructions {
                    name: "op1_phase_1"
                    opcode: "delay"
                    instruction_id: 55
                    bytes_out: 6
                  }
                  instructions {
                    name: "op2_phase_1"
                    opcode: "delay"
                    instruction_id: 56
                    bytes_out: 6
                  }
                  instructions {
                    name: "sum_phase_1"
                    opcode: "delay"
                    instruction_id: 57
                    ops: 12
                    operand_ids: 55
                    operand_ids: 56
                  }
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-1_bidir-ring_ccw"
            opcode: "reduce-scatter"
            instruction_id: 58
            bytes_out: 12
            communication_groups {
              group_ids: 6
              group_ids: 2
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-1_bidir-ring_ccw_unidir-ring"
              subroutine_root_id: 60
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 59
                bytes_in: 6
                bytes_out: 6
                communication_groups {
                  group_ids: 6
                  group_ids: 6
                }
              }
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-1_bidir-ring_ccw_unidir-ring_reduction_1"
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
                    bytes_out: 6
                  }
                  instructions {
                    name: "op2_phase_1"
                    opcode: "delay"
                    instruction_id: 62
                    bytes_out: 6
                  }
                  instructions {
                    name: "sum_phase_1"
                    opcode: "delay"
                    instruction_id: 63
                    ops: 12
                    operand_ids: 61
                    operand_ids: 62
                  }
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-1_bidir-ring_root_2"
            opcode: "null"
            instruction_id: 64
            operand_ids: 52
            operand_ids: 58
          }
        }
      }
      instructions {
        name: "all-reduce_torus-2d_reduce-scatter_stage-1_root"
        opcode: "null"
        instruction_id: 65
        operand_ids: 37
        operand_ids: 51
      }
      instructions {
        name: "all-reduce_torus-2d_reduce-scatter_conc"
        opcode: "reduce-scatter"
        instruction_id: 66
        bytes_out: 48
        communication_groups {
          group_ids: 2
          group_ids: 3
        }
        operand_ids: 65
        inner_subroutines {
          name: "all-reduce_torus-2d_reduce-scatter_conc_bidir-ring"
          subroutine_root_id: 79
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_conc_bidir-ring_cw"
            opcode: "reduce-scatter"
            instruction_id: 67
            bytes_out: 24
            communication_groups {
              group_ids: 2
              group_ids: 3
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_reduce-scatter_conc_bidir-ring_cw_unidir-ring"
              subroutine_root_id: 69
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_conc_bidir-ring_cw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 68
                bytes_in: 12
                bytes_out: 12
                communication_groups {
                  group_ids: 3
                  group_ids: 3
                }
              }
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_conc_bidir-ring_cw_unidir-ring_reduction_1"
                opcode: "call"
                instruction_id: 69
                operand_ids: 68
                inner_subroutines {
                  name: "reduction_subroutine_phase_1"
                  subroutine_root_id: 72
                  execution_probability: 1
                  execution_count: 1
                  instructions {
                    name: "op1_phase_1"
                    opcode: "delay"
                    instruction_id: 70
                    bytes_out: 12
                  }
                  instructions {
                    name: "op2_phase_1"
                    opcode: "delay"
                    instruction_id: 71
                    bytes_out: 12
                  }
                  instructions {
                    name: "sum_phase_1"
                    opcode: "delay"
                    instruction_id: 72
                    ops: 24
                    operand_ids: 70
                    operand_ids: 71
                  }
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_conc_bidir-ring_ccw"
            opcode: "reduce-scatter"
            instruction_id: 73
            bytes_out: 24
            communication_groups {
              group_ids: 3
              group_ids: 2
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_reduce-scatter_conc_bidir-ring_ccw_unidir-ring"
              subroutine_root_id: 75
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_conc_bidir-ring_ccw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 74
                bytes_in: 12
                bytes_out: 12
                communication_groups {
                  group_ids: 3
                  group_ids: 3
                }
              }
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_conc_bidir-ring_ccw_unidir-ring_reduction_1"
                opcode: "call"
                instruction_id: 75
                operand_ids: 74
                inner_subroutines {
                  name: "reduction_subroutine_phase_1"
                  subroutine_root_id: 78
                  execution_probability: 1
                  execution_count: 1
                  instructions {
                    name: "op1_phase_1"
                    opcode: "delay"
                    instruction_id: 76
                    bytes_out: 12
                  }
                  instructions {
                    name: "op2_phase_1"
                    opcode: "delay"
                    instruction_id: 77
                    bytes_out: 12
                  }
                  instructions {
                    name: "sum_phase_1"
                    opcode: "delay"
                    instruction_id: 78
                    ops: 24
                    operand_ids: 76
                    operand_ids: 77
                  }
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_conc_bidir-ring_root_2"
            opcode: "null"
            instruction_id: 79
            operand_ids: 67
            operand_ids: 73
          }
        }
      }
    }
  }
  instructions {
    name: "all-reduce_torus-2d_all-gather"
    opcode: "all-gather"
    instruction_id: 80
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
      name: "all-reduce_torus-2d_all-gather_torus-2d"
      subroutine_root_id: 107
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-reduce_torus-2d_all-gather_stage-0_dim-0"
        opcode: "all-gather"
        instruction_id: 81
        bytes_out: 12
        communication_groups {
          group_ids: 0
          group_ids: 2
        }
        inner_subroutines {
          name: "all-reduce_torus-2d_all-gather_stage-0_dim-0_bidir-ring"
          subroutine_root_id: 86
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_torus-2d_all-gather_stage-0_dim-0_bidir-ring_cw"
            opcode: "all-gather"
            instruction_id: 82
            bytes_out: 6
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_all-gather_stage-0_dim-0_bidir-ring_cw_unidir-ring"
              subroutine_root_id: 83
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_all-gather_stage-0_dim-0_bidir-ring_cw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 83
                bytes_in: 3
                bytes_out: 3
                communication_groups {
                  group_ids: 0
                  group_ids: 0
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_all-gather_stage-0_dim-0_bidir-ring_ccw"
            opcode: "all-gather"
            instruction_id: 84
            bytes_out: 6
            communication_groups {
              group_ids: 2
              group_ids: 0
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_all-gather_stage-0_dim-0_bidir-ring_ccw_unidir-ring"
              subroutine_root_id: 85
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_all-gather_stage-0_dim-0_bidir-ring_ccw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 85
                bytes_in: 3
                bytes_out: 3
                communication_groups {
                  group_ids: 0
                  group_ids: 0
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_all-gather_stage-0_dim-0_bidir-ring_root_2"
            opcode: "null"
            instruction_id: 86
            operand_ids: 82
            operand_ids: 84
          }
        }
      }
      instructions {
        name: "all-reduce_torus-2d_all-gather_stage-0_dim-1"
        opcode: "all-gather"
        instruction_id: 87
        bytes_out: 12
        communication_groups {
          group_ids: 2
          group_ids: 6
        }
        inner_subroutines {
          name: "all-reduce_torus-2d_all-gather_stage-0_dim-1_bidir-ring"
          subroutine_root_id: 92
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_torus-2d_all-gather_stage-0_dim-1_bidir-ring_cw"
            opcode: "all-gather"
            instruction_id: 88
            bytes_out: 6
            communication_groups {
              group_ids: 2
              group_ids: 6
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_all-gather_stage-0_dim-1_bidir-ring_cw_unidir-ring"
              subroutine_root_id: 89
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_all-gather_stage-0_dim-1_bidir-ring_cw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 89
                bytes_in: 3
                bytes_out: 3
                communication_groups {
                  group_ids: 6
                  group_ids: 6
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_all-gather_stage-0_dim-1_bidir-ring_ccw"
            opcode: "all-gather"
            instruction_id: 90
            bytes_out: 6
            communication_groups {
              group_ids: 6
              group_ids: 2
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_all-gather_stage-0_dim-1_bidir-ring_ccw_unidir-ring"
              subroutine_root_id: 91
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_all-gather_stage-0_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 91
                bytes_in: 3
                bytes_out: 3
                communication_groups {
                  group_ids: 6
                  group_ids: 6
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_all-gather_stage-0_dim-1_bidir-ring_root_2"
            opcode: "null"
            instruction_id: 92
            operand_ids: 88
            operand_ids: 90
          }
        }
      }
      instructions {
        name: "all-reduce_torus-2d_all-gather_stage-0_root"
        opcode: "null"
        instruction_id: 93
        operand_ids: 81
        operand_ids: 87
      }
      instructions {
        name: "all-reduce_torus-2d_all-gather_stage-1_dim-0"
        opcode: "all-gather"
        instruction_id: 94
        bytes_out: 24
        communication_groups {
          group_ids: 0
          group_ids: 2
        }
        operand_ids: 93
        inner_subroutines {
          name: "all-reduce_torus-2d_all-gather_stage-1_dim-0_bidir-ring"
          subroutine_root_id: 99
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_torus-2d_all-gather_stage-1_dim-0_bidir-ring_cw"
            opcode: "all-gather"
            instruction_id: 95
            bytes_out: 12
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_all-gather_stage-1_dim-0_bidir-ring_cw_unidir-ring"
              subroutine_root_id: 96
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_all-gather_stage-1_dim-0_bidir-ring_cw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 96
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
            name: "all-reduce_torus-2d_all-gather_stage-1_dim-0_bidir-ring_ccw"
            opcode: "all-gather"
            instruction_id: 97
            bytes_out: 12
            communication_groups {
              group_ids: 2
              group_ids: 0
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_all-gather_stage-1_dim-0_bidir-ring_ccw_unidir-ring"
              subroutine_root_id: 98
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_all-gather_stage-1_dim-0_bidir-ring_ccw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 98
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
            name: "all-reduce_torus-2d_all-gather_stage-1_dim-0_bidir-ring_root_2"
            opcode: "null"
            instruction_id: 99
            operand_ids: 95
            operand_ids: 97
          }
        }
      }
      instructions {
        name: "all-reduce_torus-2d_all-gather_stage-1_dim-1"
        opcode: "all-gather"
        instruction_id: 100
        bytes_out: 24
        communication_groups {
          group_ids: 2
          group_ids: 6
        }
        operand_ids: 93
        inner_subroutines {
          name: "all-reduce_torus-2d_all-gather_stage-1_dim-1_bidir-ring"
          subroutine_root_id: 105
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_torus-2d_all-gather_stage-1_dim-1_bidir-ring_cw"
            opcode: "all-gather"
            instruction_id: 101
            bytes_out: 12
            communication_groups {
              group_ids: 2
              group_ids: 6
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_all-gather_stage-1_dim-1_bidir-ring_cw_unidir-ring"
              subroutine_root_id: 102
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_all-gather_stage-1_dim-1_bidir-ring_cw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 102
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
            name: "all-reduce_torus-2d_all-gather_stage-1_dim-1_bidir-ring_ccw"
            opcode: "all-gather"
            instruction_id: 103
            bytes_out: 12
            communication_groups {
              group_ids: 6
              group_ids: 2
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_all-gather_stage-1_dim-1_bidir-ring_ccw_unidir-ring"
              subroutine_root_id: 104
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_all-gather_stage-1_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 104
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
            name: "all-reduce_torus-2d_all-gather_stage-1_dim-1_bidir-ring_root_2"
            opcode: "null"
            instruction_id: 105
            operand_ids: 101
            operand_ids: 103
          }
        }
      }
      instructions {
        name: "all-reduce_torus-2d_all-gather_stage-1_root"
        opcode: "null"
        instruction_id: 106
        operand_ids: 94
        operand_ids: 100
      }
      instructions {
        name: "all-reduce_torus-2d_all-gather_conc"
        opcode: "all-gather"
        instruction_id: 107
        bytes_out: 48
        communication_groups {
          group_ids: 2
          group_ids: 3
        }
        operand_ids: 106
        inner_subroutines {
          name: "all-reduce_torus-2d_all-gather_conc_bidir-ring"
          subroutine_root_id: 112
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_torus-2d_all-gather_conc_bidir-ring_cw"
            opcode: "all-gather"
            instruction_id: 108
            bytes_out: 24
            communication_groups {
              group_ids: 2
              group_ids: 3
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_all-gather_conc_bidir-ring_cw_unidir-ring"
              subroutine_root_id: 109
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_all-gather_conc_bidir-ring_cw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 109
                bytes_in: 12
                bytes_out: 12
                communication_groups {
                  group_ids: 3
                  group_ids: 3
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_all-gather_conc_bidir-ring_ccw"
            opcode: "all-gather"
            instruction_id: 110
            bytes_out: 24
            communication_groups {
              group_ids: 3
              group_ids: 2
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_all-gather_conc_bidir-ring_ccw_unidir-ring"
              subroutine_root_id: 111
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_all-gather_conc_bidir-ring_ccw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 111
                bytes_in: 12
                bytes_out: 12
                communication_groups {
                  group_ids: 3
                  group_ids: 3
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_all-gather_conc_bidir-ring_root_2"
            opcode: "null"
            instruction_id: 112
            operand_ids: 108
            operand_ids: 110
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

// Tests expanding 2D-Torus all-reduce
TEST(Torus2dAllReduce, NoBarrier) {
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
        "algorithm": "torus-2d",
        "concentration": 2,
        "dimension_widths": [2, 2]
      }
    }
  )"_json;

  ASSERT_OK_AND_ASSIGN(auto translators, paragraph::CreateTranslators(
      paragraph::TranslatorType::kCollective, config));
  EXPECT_OK(translators["all-reduce"]->Translate(allreduce));

  paragraph::InstructionProto allreduce_proto = no_barrier_test_proto();
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      allreduce->ToProto().value(), allreduce_proto));
}

paragraph::InstructionProto with_barrier_test_proto() {
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
  name: "all-reduce_torus-2d"
  subroutine_root_id: 80
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "all-reduce_torus-2d_reduce-scatter"
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
      name: "all-reduce_torus-2d_reduce-scatter_torus-2d"
      subroutine_root_id: 79
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-0"
        opcode: "reduce-scatter"
        instruction_id: 8
        bytes_out: 12
        communication_groups {
          group_ids: 2
          group_ids: 3
        }
        inner_subroutines {
          name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-0_bidir-ring"
          subroutine_root_id: 25
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-0_bidir-ring_barrier"
            opcode: "barrier"
            instruction_id: 9
            communication_groups {
              group_ids: 2
              group_ids: 3
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-0_bidir-ring_barrier_centralized"
              subroutine_root_id: 12
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-0_bidir-ring_barrier_centralized_coordinator_recv_from_3"
                opcode: "recv"
                instruction_id: 10
                communication_groups {
                  group_ids: 3
                }
              }
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-0_bidir-ring_barrier_centralized_coordinator_send_to_3"
                opcode: "send"
                instruction_id: 11
                communication_groups {
                  group_ids: 3
                }
                operand_ids: 10
              }
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-0_bidir-ring_barrier_centralized_root_2"
                opcode: "null"
                instruction_id: 12
                operand_ids: 11
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-0_bidir-ring_cw"
            opcode: "reduce-scatter"
            instruction_id: 13
            bytes_out: 6
            communication_groups {
              group_ids: 2
              group_ids: 3
            }
            operand_ids: 9
            inner_subroutines {
              name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-0_bidir-ring_cw_unidir-ring"
              subroutine_root_id: 15
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-0_bidir-ring_cw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 14
                bytes_in: 3
                bytes_out: 3
                communication_groups {
                  group_ids: 3
                  group_ids: 3
                }
              }
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-0_bidir-ring_cw_unidir-ring_reduction_1"
                opcode: "call"
                instruction_id: 15
                operand_ids: 14
                inner_subroutines {
                  name: "reduction_subroutine_phase_1"
                  subroutine_root_id: 18
                  execution_probability: 1
                  execution_count: 1
                  instructions {
                    name: "op1_phase_1"
                    opcode: "delay"
                    instruction_id: 16
                    bytes_out: 3
                  }
                  instructions {
                    name: "op2_phase_1"
                    opcode: "delay"
                    instruction_id: 17
                    bytes_out: 3
                  }
                  instructions {
                    name: "sum_phase_1"
                    opcode: "delay"
                    instruction_id: 18
                    ops: 6
                    operand_ids: 16
                    operand_ids: 17
                  }
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-0_bidir-ring_ccw"
            opcode: "reduce-scatter"
            instruction_id: 19
            bytes_out: 6
            communication_groups {
              group_ids: 3
              group_ids: 2
            }
            operand_ids: 9
            inner_subroutines {
              name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-0_bidir-ring_ccw_unidir-ring"
              subroutine_root_id: 21
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-0_bidir-ring_ccw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 20
                bytes_in: 3
                bytes_out: 3
                communication_groups {
                  group_ids: 3
                  group_ids: 3
                }
              }
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-0_bidir-ring_ccw_unidir-ring_reduction_1"
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
                    bytes_out: 3
                  }
                  instructions {
                    name: "op2_phase_1"
                    opcode: "delay"
                    instruction_id: 23
                    bytes_out: 3
                  }
                  instructions {
                    name: "sum_phase_1"
                    opcode: "delay"
                    instruction_id: 24
                    ops: 6
                    operand_ids: 22
                    operand_ids: 23
                  }
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-0_bidir-ring_root_2"
            opcode: "null"
            instruction_id: 25
            operand_ids: 13
            operand_ids: 19
          }
        }
      }
      instructions {
        name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-1"
        opcode: "reduce-scatter"
        instruction_id: 26
        bytes_out: 12
        communication_groups {
          group_ids: 0
          group_ids: 2
        }
        inner_subroutines {
          name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-1_bidir-ring"
          subroutine_root_id: 42
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-1_bidir-ring_barrier"
            opcode: "barrier"
            instruction_id: 27
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-1_bidir-ring_barrier_centralized"
              subroutine_root_id: 29
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-1_bidir-ring_barrier_centralized_send_to_0"
                opcode: "send"
                instruction_id: 28
                communication_groups {
                  group_ids: 0
                }
              }
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-1_bidir-ring_barrier_centralized_recv_from_0"
                opcode: "recv"
                instruction_id: 29
                communication_groups {
                  group_ids: 0
                }
                operand_ids: 28
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-1_bidir-ring_cw"
            opcode: "reduce-scatter"
            instruction_id: 30
            bytes_out: 6
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
            operand_ids: 27
            inner_subroutines {
              name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-1_bidir-ring_cw_unidir-ring"
              subroutine_root_id: 32
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-1_bidir-ring_cw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 31
                bytes_in: 3
                bytes_out: 3
                communication_groups {
                  group_ids: 0
                  group_ids: 0
                }
              }
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-1_bidir-ring_cw_unidir-ring_reduction_1"
                opcode: "call"
                instruction_id: 32
                operand_ids: 31
                inner_subroutines {
                  name: "reduction_subroutine_phase_1"
                  subroutine_root_id: 35
                  execution_probability: 1
                  execution_count: 1
                  instructions {
                    name: "op1_phase_1"
                    opcode: "delay"
                    instruction_id: 33
                    bytes_out: 3
                  }
                  instructions {
                    name: "op2_phase_1"
                    opcode: "delay"
                    instruction_id: 34
                    bytes_out: 3
                  }
                  instructions {
                    name: "sum_phase_1"
                    opcode: "delay"
                    instruction_id: 35
                    ops: 6
                    operand_ids: 33
                    operand_ids: 34
                  }
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-1_bidir-ring_ccw"
            opcode: "reduce-scatter"
            instruction_id: 36
            bytes_out: 6
            communication_groups {
              group_ids: 2
              group_ids: 0
            }
            operand_ids: 27
            inner_subroutines {
              name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-1_bidir-ring_ccw_unidir-ring"
              subroutine_root_id: 38
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 37
                bytes_in: 3
                bytes_out: 3
                communication_groups {
                  group_ids: 0
                  group_ids: 0
                }
              }
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-1_bidir-ring_ccw_unidir-ring_reduction_1"
                opcode: "call"
                instruction_id: 38
                operand_ids: 37
                inner_subroutines {
                  name: "reduction_subroutine_phase_1"
                  subroutine_root_id: 41
                  execution_probability: 1
                  execution_count: 1
                  instructions {
                    name: "op1_phase_1"
                    opcode: "delay"
                    instruction_id: 39
                    bytes_out: 3
                  }
                  instructions {
                    name: "op2_phase_1"
                    opcode: "delay"
                    instruction_id: 40
                    bytes_out: 3
                  }
                  instructions {
                    name: "sum_phase_1"
                    opcode: "delay"
                    instruction_id: 41
                    ops: 6
                    operand_ids: 39
                    operand_ids: 40
                  }
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-0_dim-1_bidir-ring_root_2"
            opcode: "null"
            instruction_id: 42
            operand_ids: 30
            operand_ids: 36
          }
        }
      }
      instructions {
        name: "all-reduce_torus-2d_reduce-scatter_stage-0_root"
        opcode: "null"
        instruction_id: 43
        operand_ids: 8
        operand_ids: 26
      }
      instructions {
        name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-0"
        opcode: "reduce-scatter"
        instruction_id: 44
        bytes_out: 24
        communication_groups {
          group_ids: 2
          group_ids: 3
        }
        operand_ids: 43
        inner_subroutines {
          name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-0_bidir-ring"
          subroutine_root_id: 61
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-0_bidir-ring_barrier"
            opcode: "barrier"
            instruction_id: 45
            communication_groups {
              group_ids: 2
              group_ids: 3
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-0_bidir-ring_barrier_centralized"
              subroutine_root_id: 48
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-0_bidir-ring_barrier_centralized_coordinator_recv_from_3"
                opcode: "recv"
                instruction_id: 46
                communication_groups {
                  group_ids: 3
                }
              }
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-0_bidir-ring_barrier_centralized_coordinator_send_to_3"
                opcode: "send"
                instruction_id: 47
                communication_groups {
                  group_ids: 3
                }
                operand_ids: 46
              }
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-0_bidir-ring_barrier_centralized_root_2"
                opcode: "null"
                instruction_id: 48
                operand_ids: 47
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-0_bidir-ring_cw"
            opcode: "reduce-scatter"
            instruction_id: 49
            bytes_out: 12
            communication_groups {
              group_ids: 2
              group_ids: 3
            }
            operand_ids: 45
            inner_subroutines {
              name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-0_bidir-ring_cw_unidir-ring"
              subroutine_root_id: 51
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-0_bidir-ring_cw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 50
                bytes_in: 6
                bytes_out: 6
                communication_groups {
                  group_ids: 3
                  group_ids: 3
                }
              }
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-0_bidir-ring_cw_unidir-ring_reduction_1"
                opcode: "call"
                instruction_id: 51
                operand_ids: 50
                inner_subroutines {
                  name: "reduction_subroutine_phase_1"
                  subroutine_root_id: 54
                  execution_probability: 1
                  execution_count: 1
                  instructions {
                    name: "op1_phase_1"
                    opcode: "delay"
                    instruction_id: 52
                    bytes_out: 6
                  }
                  instructions {
                    name: "op2_phase_1"
                    opcode: "delay"
                    instruction_id: 53
                    bytes_out: 6
                  }
                  instructions {
                    name: "sum_phase_1"
                    opcode: "delay"
                    instruction_id: 54
                    ops: 12
                    operand_ids: 52
                    operand_ids: 53
                  }
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-0_bidir-ring_ccw"
            opcode: "reduce-scatter"
            instruction_id: 55
            bytes_out: 12
            communication_groups {
              group_ids: 3
              group_ids: 2
            }
            operand_ids: 45
            inner_subroutines {
              name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-0_bidir-ring_ccw_unidir-ring"
              subroutine_root_id: 57
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-0_bidir-ring_ccw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 56
                bytes_in: 6
                bytes_out: 6
                communication_groups {
                  group_ids: 3
                  group_ids: 3
                }
              }
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-0_bidir-ring_ccw_unidir-ring_reduction_1"
                opcode: "call"
                instruction_id: 57
                operand_ids: 56
                inner_subroutines {
                  name: "reduction_subroutine_phase_1"
                  subroutine_root_id: 60
                  execution_probability: 1
                  execution_count: 1
                  instructions {
                    name: "op1_phase_1"
                    opcode: "delay"
                    instruction_id: 58
                    bytes_out: 6
                  }
                  instructions {
                    name: "op2_phase_1"
                    opcode: "delay"
                    instruction_id: 59
                    bytes_out: 6
                  }
                  instructions {
                    name: "sum_phase_1"
                    opcode: "delay"
                    instruction_id: 60
                    ops: 12
                    operand_ids: 58
                    operand_ids: 59
                  }
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-0_bidir-ring_root_2"
            opcode: "null"
            instruction_id: 61
            operand_ids: 49
            operand_ids: 55
          }
        }
      }
      instructions {
        name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-1"
        opcode: "reduce-scatter"
        instruction_id: 62
        bytes_out: 24
        communication_groups {
          group_ids: 0
          group_ids: 2
        }
        operand_ids: 43
        inner_subroutines {
          name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-1_bidir-ring"
          subroutine_root_id: 78
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-1_bidir-ring_barrier"
            opcode: "barrier"
            instruction_id: 63
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-1_bidir-ring_barrier_centralized"
              subroutine_root_id: 65
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-1_bidir-ring_barrier_centralized_send_to_0"
                opcode: "send"
                instruction_id: 64
                communication_groups {
                  group_ids: 0
                }
              }
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-1_bidir-ring_barrier_centralized_recv_from_0"
                opcode: "recv"
                instruction_id: 65
                communication_groups {
                  group_ids: 0
                }
                operand_ids: 64
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-1_bidir-ring_cw"
            opcode: "reduce-scatter"
            instruction_id: 66
            bytes_out: 12
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
            operand_ids: 63
            inner_subroutines {
              name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-1_bidir-ring_cw_unidir-ring"
              subroutine_root_id: 68
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-1_bidir-ring_cw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 67
                bytes_in: 6
                bytes_out: 6
                communication_groups {
                  group_ids: 0
                  group_ids: 0
                }
              }
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-1_bidir-ring_cw_unidir-ring_reduction_1"
                opcode: "call"
                instruction_id: 68
                operand_ids: 67
                inner_subroutines {
                  name: "reduction_subroutine_phase_1"
                  subroutine_root_id: 71
                  execution_probability: 1
                  execution_count: 1
                  instructions {
                    name: "op1_phase_1"
                    opcode: "delay"
                    instruction_id: 69
                    bytes_out: 6
                  }
                  instructions {
                    name: "op2_phase_1"
                    opcode: "delay"
                    instruction_id: 70
                    bytes_out: 6
                  }
                  instructions {
                    name: "sum_phase_1"
                    opcode: "delay"
                    instruction_id: 71
                    ops: 12
                    operand_ids: 69
                    operand_ids: 70
                  }
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-1_bidir-ring_ccw"
            opcode: "reduce-scatter"
            instruction_id: 72
            bytes_out: 12
            communication_groups {
              group_ids: 2
              group_ids: 0
            }
            operand_ids: 63
            inner_subroutines {
              name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-1_bidir-ring_ccw_unidir-ring"
              subroutine_root_id: 74
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 73
                bytes_in: 6
                bytes_out: 6
                communication_groups {
                  group_ids: 0
                  group_ids: 0
                }
              }
              instructions {
                name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-1_bidir-ring_ccw_unidir-ring_reduction_1"
                opcode: "call"
                instruction_id: 74
                operand_ids: 73
                inner_subroutines {
                  name: "reduction_subroutine_phase_1"
                  subroutine_root_id: 77
                  execution_probability: 1
                  execution_count: 1
                  instructions {
                    name: "op1_phase_1"
                    opcode: "delay"
                    instruction_id: 75
                    bytes_out: 6
                  }
                  instructions {
                    name: "op2_phase_1"
                    opcode: "delay"
                    instruction_id: 76
                    bytes_out: 6
                  }
                  instructions {
                    name: "sum_phase_1"
                    opcode: "delay"
                    instruction_id: 77
                    ops: 12
                    operand_ids: 75
                    operand_ids: 76
                  }
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_reduce-scatter_stage-1_dim-1_bidir-ring_root_2"
            opcode: "null"
            instruction_id: 78
            operand_ids: 66
            operand_ids: 72
          }
        }
      }
      instructions {
        name: "all-reduce_torus-2d_reduce-scatter_stage-1_root"
        opcode: "null"
        instruction_id: 79
        operand_ids: 44
        operand_ids: 62
      }
    }
  }
  instructions {
    name: "all-reduce_torus-2d_all-gather"
    opcode: "all-gather"
    instruction_id: 80
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
      name: "all-reduce_torus-2d_all-gather_torus-2d"
      subroutine_root_id: 120
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "all-reduce_torus-2d_all-gather_stage-0_dim-0"
        opcode: "all-gather"
        instruction_id: 81
        bytes_out: 12
        communication_groups {
          group_ids: 2
          group_ids: 3
        }
        inner_subroutines {
          name: "all-reduce_torus-2d_all-gather_stage-0_dim-0_bidir-ring"
          subroutine_root_id: 90
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_torus-2d_all-gather_stage-0_dim-0_bidir-ring_barrier"
            opcode: "barrier"
            instruction_id: 82
            communication_groups {
              group_ids: 2
              group_ids: 3
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_all-gather_stage-0_dim-0_bidir-ring_barrier_centralized"
              subroutine_root_id: 85
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_all-gather_stage-0_dim-0_bidir-ring_barrier_centralized_coordinator_recv_from_3"
                opcode: "recv"
                instruction_id: 83
                communication_groups {
                  group_ids: 3
                }
              }
              instructions {
                name: "all-reduce_torus-2d_all-gather_stage-0_dim-0_bidir-ring_barrier_centralized_coordinator_send_to_3"
                opcode: "send"
                instruction_id: 84
                communication_groups {
                  group_ids: 3
                }
                operand_ids: 83
              }
              instructions {
                name: "all-reduce_torus-2d_all-gather_stage-0_dim-0_bidir-ring_barrier_centralized_root_2"
                opcode: "null"
                instruction_id: 85
                operand_ids: 84
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_all-gather_stage-0_dim-0_bidir-ring_cw"
            opcode: "all-gather"
            instruction_id: 86
            bytes_out: 6
            communication_groups {
              group_ids: 2
              group_ids: 3
            }
            operand_ids: 82
            inner_subroutines {
              name: "all-reduce_torus-2d_all-gather_stage-0_dim-0_bidir-ring_cw_unidir-ring"
              subroutine_root_id: 87
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_all-gather_stage-0_dim-0_bidir-ring_cw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 87
                bytes_in: 3
                bytes_out: 3
                communication_groups {
                  group_ids: 3
                  group_ids: 3
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_all-gather_stage-0_dim-0_bidir-ring_ccw"
            opcode: "all-gather"
            instruction_id: 88
            bytes_out: 6
            communication_groups {
              group_ids: 3
              group_ids: 2
            }
            operand_ids: 82
            inner_subroutines {
              name: "all-reduce_torus-2d_all-gather_stage-0_dim-0_bidir-ring_ccw_unidir-ring"
              subroutine_root_id: 89
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_all-gather_stage-0_dim-0_bidir-ring_ccw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 89
                bytes_in: 3
                bytes_out: 3
                communication_groups {
                  group_ids: 3
                  group_ids: 3
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_all-gather_stage-0_dim-0_bidir-ring_root_2"
            opcode: "null"
            instruction_id: 90
            operand_ids: 86
            operand_ids: 88
          }
        }
      }
      instructions {
        name: "all-reduce_torus-2d_all-gather_stage-0_dim-1"
        opcode: "all-gather"
        instruction_id: 91
        bytes_out: 12
        communication_groups {
          group_ids: 0
          group_ids: 2
        }
        inner_subroutines {
          name: "all-reduce_torus-2d_all-gather_stage-0_dim-1_bidir-ring"
          subroutine_root_id: 99
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_torus-2d_all-gather_stage-0_dim-1_bidir-ring_barrier"
            opcode: "barrier"
            instruction_id: 92
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_all-gather_stage-0_dim-1_bidir-ring_barrier_centralized"
              subroutine_root_id: 94
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_all-gather_stage-0_dim-1_bidir-ring_barrier_centralized_send_to_0"
                opcode: "send"
                instruction_id: 93
                communication_groups {
                  group_ids: 0
                }
              }
              instructions {
                name: "all-reduce_torus-2d_all-gather_stage-0_dim-1_bidir-ring_barrier_centralized_recv_from_0"
                opcode: "recv"
                instruction_id: 94
                communication_groups {
                  group_ids: 0
                }
                operand_ids: 93
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_all-gather_stage-0_dim-1_bidir-ring_cw"
            opcode: "all-gather"
            instruction_id: 95
            bytes_out: 6
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
            operand_ids: 92
            inner_subroutines {
              name: "all-reduce_torus-2d_all-gather_stage-0_dim-1_bidir-ring_cw_unidir-ring"
              subroutine_root_id: 96
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_all-gather_stage-0_dim-1_bidir-ring_cw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 96
                bytes_in: 3
                bytes_out: 3
                communication_groups {
                  group_ids: 0
                  group_ids: 0
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_all-gather_stage-0_dim-1_bidir-ring_ccw"
            opcode: "all-gather"
            instruction_id: 97
            bytes_out: 6
            communication_groups {
              group_ids: 2
              group_ids: 0
            }
            operand_ids: 92
            inner_subroutines {
              name: "all-reduce_torus-2d_all-gather_stage-0_dim-1_bidir-ring_ccw_unidir-ring"
              subroutine_root_id: 98
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_all-gather_stage-0_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 98
                bytes_in: 3
                bytes_out: 3
                communication_groups {
                  group_ids: 0
                  group_ids: 0
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_all-gather_stage-0_dim-1_bidir-ring_root_2"
            opcode: "null"
            instruction_id: 99
            operand_ids: 95
            operand_ids: 97
          }
        }
      }
      instructions {
        name: "all-reduce_torus-2d_all-gather_stage-0_root"
        opcode: "null"
        instruction_id: 100
        operand_ids: 81
        operand_ids: 91
      }
      instructions {
        name: "all-reduce_torus-2d_all-gather_stage-1_dim-0"
        opcode: "all-gather"
        instruction_id: 101
        bytes_out: 24
        communication_groups {
          group_ids: 2
          group_ids: 3
        }
        operand_ids: 100
        inner_subroutines {
          name: "all-reduce_torus-2d_all-gather_stage-1_dim-0_bidir-ring"
          subroutine_root_id: 110
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_torus-2d_all-gather_stage-1_dim-0_bidir-ring_barrier"
            opcode: "barrier"
            instruction_id: 102
            communication_groups {
              group_ids: 2
              group_ids: 3
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_all-gather_stage-1_dim-0_bidir-ring_barrier_centralized"
              subroutine_root_id: 105
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_all-gather_stage-1_dim-0_bidir-ring_barrier_centralized_coordinator_recv_from_3"
                opcode: "recv"
                instruction_id: 103
                communication_groups {
                  group_ids: 3
                }
              }
              instructions {
                name: "all-reduce_torus-2d_all-gather_stage-1_dim-0_bidir-ring_barrier_centralized_coordinator_send_to_3"
                opcode: "send"
                instruction_id: 104
                communication_groups {
                  group_ids: 3
                }
                operand_ids: 103
              }
              instructions {
                name: "all-reduce_torus-2d_all-gather_stage-1_dim-0_bidir-ring_barrier_centralized_root_2"
                opcode: "null"
                instruction_id: 105
                operand_ids: 104
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_all-gather_stage-1_dim-0_bidir-ring_cw"
            opcode: "all-gather"
            instruction_id: 106
            bytes_out: 12
            communication_groups {
              group_ids: 2
              group_ids: 3
            }
            operand_ids: 102
            inner_subroutines {
              name: "all-reduce_torus-2d_all-gather_stage-1_dim-0_bidir-ring_cw_unidir-ring"
              subroutine_root_id: 107
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_all-gather_stage-1_dim-0_bidir-ring_cw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 107
                bytes_in: 6
                bytes_out: 6
                communication_groups {
                  group_ids: 3
                  group_ids: 3
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_all-gather_stage-1_dim-0_bidir-ring_ccw"
            opcode: "all-gather"
            instruction_id: 108
            bytes_out: 12
            communication_groups {
              group_ids: 3
              group_ids: 2
            }
            operand_ids: 102
            inner_subroutines {
              name: "all-reduce_torus-2d_all-gather_stage-1_dim-0_bidir-ring_ccw_unidir-ring"
              subroutine_root_id: 109
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_all-gather_stage-1_dim-0_bidir-ring_ccw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 109
                bytes_in: 6
                bytes_out: 6
                communication_groups {
                  group_ids: 3
                  group_ids: 3
                }
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_all-gather_stage-1_dim-0_bidir-ring_root_2"
            opcode: "null"
            instruction_id: 110
            operand_ids: 106
            operand_ids: 108
          }
        }
      }
      instructions {
        name: "all-reduce_torus-2d_all-gather_stage-1_dim-1"
        opcode: "all-gather"
        instruction_id: 111
        bytes_out: 24
        communication_groups {
          group_ids: 0
          group_ids: 2
        }
        operand_ids: 100
        inner_subroutines {
          name: "all-reduce_torus-2d_all-gather_stage-1_dim-1_bidir-ring"
          subroutine_root_id: 119
          execution_probability: 1
          execution_count: 1
          instructions {
            name: "all-reduce_torus-2d_all-gather_stage-1_dim-1_bidir-ring_barrier"
            opcode: "barrier"
            instruction_id: 112
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
            inner_subroutines {
              name: "all-reduce_torus-2d_all-gather_stage-1_dim-1_bidir-ring_barrier_centralized"
              subroutine_root_id: 114
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_all-gather_stage-1_dim-1_bidir-ring_barrier_centralized_send_to_0"
                opcode: "send"
                instruction_id: 113
                communication_groups {
                  group_ids: 0
                }
              }
              instructions {
                name: "all-reduce_torus-2d_all-gather_stage-1_dim-1_bidir-ring_barrier_centralized_recv_from_0"
                opcode: "recv"
                instruction_id: 114
                communication_groups {
                  group_ids: 0
                }
                operand_ids: 113
              }
            }
          }
          instructions {
            name: "all-reduce_torus-2d_all-gather_stage-1_dim-1_bidir-ring_cw"
            opcode: "all-gather"
            instruction_id: 115
            bytes_out: 12
            communication_groups {
              group_ids: 0
              group_ids: 2
            }
            operand_ids: 112
            inner_subroutines {
              name: "all-reduce_torus-2d_all-gather_stage-1_dim-1_bidir-ring_cw_unidir-ring"
              subroutine_root_id: 116
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_all-gather_stage-1_dim-1_bidir-ring_cw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 116
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
            name: "all-reduce_torus-2d_all-gather_stage-1_dim-1_bidir-ring_ccw"
            opcode: "all-gather"
            instruction_id: 117
            bytes_out: 12
            communication_groups {
              group_ids: 2
              group_ids: 0
            }
            operand_ids: 112
            inner_subroutines {
              name: "all-reduce_torus-2d_all-gather_stage-1_dim-1_bidir-ring_ccw_unidir-ring"
              subroutine_root_id: 118
              execution_probability: 1
              execution_count: 1
              instructions {
                name: "all-reduce_torus-2d_all-gather_stage-1_dim-1_bidir-ring_ccw_unidir-ring_sendrecv_1"
                opcode: "sendrecv"
                instruction_id: 118
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
            name: "all-reduce_torus-2d_all-gather_stage-1_dim-1_bidir-ring_root_2"
            opcode: "null"
            instruction_id: 119
            operand_ids: 115
            operand_ids: 117
          }
        }
      }
      instructions {
        name: "all-reduce_torus-2d_all-gather_stage-1_root"
        opcode: "null"
        instruction_id: 120
        operand_ids: 101
        operand_ids: 111
      }
    }
  }
}
    )proto";
  google::protobuf::TextFormat::ParseFromString(test_str,
                                                &proto);
  return proto;
}  // NOLINT

// Tests expanding 2D-Torus all-reduce with barrier
TEST(Torus2dAllReduce, WithBarrier) {
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
        "algorithm": "torus-2d",
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

  paragraph::InstructionProto allreduce_proto = with_barrier_test_proto();
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      allreduce->ToProto().value(), allreduce_proto));
}
