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
#ifndef PARAGRAPH_GRAPH_OPCODE_H_
#define PARAGRAPH_GRAPH_OPCODE_H_

#include <string>

#include "absl/container/flat_hash_set.h"
#include "paragraph/shim/macros.h"
#include "paragraph/shim/statusor.h"

namespace paragraph {

// Opcode list for instruction
#define OPCODE_LIST(V)                                                 \
  V(kAllGather, "all-gather")                                          \
  V(kAllReduce, "all-reduce")                                          \
  V(kAllToAll, "all-to-all")                                           \
  V(kBarrier, "barrier")                                               \
  V(kBroadcast, "broadcast")                                           \
  V(kCall, "call")                                                     \
  V(kConditional, "conditional")                                       \
  V(kDelay, "delay")                                                   \
  V(kGather, "gather")                                                 \
  V(kInfeed, "infeed")                                                 \
  V(kNull, "null")                                                     \
  V(kOutfeed, "outfeed")                                               \
  V(kRecv, "recv")                                                     \
  V(kRecvDone, "recv-done")                                            \
  V(kRecvStart, "recv-start")                                          \
  V(kReduce, "reduce")                                                 \
  V(kReduceScatter, "reduce-scatter")                                  \
  V(kScatter, "scatter")                                               \
  V(kSend, "send")                                                     \
  V(kSendRecv, "sendrecv")                                             \
  V(kSendDone, "send-done")                                            \
  V(kSendStart, "send-start")                                          \
  V(kWhile, "while")

enum class Opcode {
#define DECLARE_ENUM(enum_name, opcode_name) enum_name,
  OPCODE_LIST(DECLARE_ENUM)
#undef DECLARE_ENUM
};

std::string OpcodeToString(Opcode opcode);
shim::StatusOr<Opcode> StringToOpcode(const std::string& opcode_name);

bool OpcodeIsProtocolLevelCommunication(Opcode opcode);
bool OpcodeIsCollectiveCommunication(Opcode opcode);
bool OpcodeIsIndividualCommunication(Opcode opcode);
bool OpcodeIsControlFlow(Opcode opcode);
bool OpcodeIsGeneralPurpose(Opcode opcode);

}  // namespace paragraph

#endif  // PARAGRAPH_GRAPH_OPCODE_H_
