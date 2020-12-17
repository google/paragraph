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
#include "paragraph/graph/opcode.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"

namespace paragraph {

std::string OpcodeToString(Opcode opcode) {
  switch (opcode) {
#define CASE_OPCODE_STRING(enum_name, opcode_name)                     \
  case Opcode::enum_name:                                              \
    return opcode_name;
    OPCODE_LIST(CASE_OPCODE_STRING)
#undef CASE_OPCODE_STRING
  default:
    return "UNKNOWN";
  }
}

shim::StatusOr<Opcode> StringToOpcode(const std::string& opcode_name) {
  static auto* opcode_map = new absl::flat_hash_map<std::string, Opcode>({
#define STRING_TO_OPCODE_ENTRY(enum_name, opcode_name) \
  {opcode_name, Opcode::enum_name},
      OPCODE_LIST(STRING_TO_OPCODE_ENTRY)
#undef STRING_TO_OPCODE_ENTRY
  });
  auto it = opcode_map->find(opcode_name);
  if (it == opcode_map->end()) {
    return absl::InvalidArgumentError(
        "Unknown opcode " + opcode_name + " in StringToOpcode conversion");
  }
  return it->second;
}

bool OpcodeIsProtocolLevelCommunication(Opcode opcode) {
  static auto* protocol_map = new absl::flat_hash_set<Opcode>({
    Opcode::kSendStart,
    Opcode::kSendDone,
    Opcode::kRecvStart,
    Opcode::kRecvDone,
  });
  return protocol_map->find(opcode) != protocol_map->end();
}

bool OpcodeIsCollectiveCommunication(Opcode opcode) {
  static auto* collective_map = new absl::flat_hash_set<Opcode>({
    Opcode::kAllGather,
    Opcode::kAllReduce,
    Opcode::kAllToAll,
    Opcode::kBarrier,
    Opcode::kBroadcast,
    Opcode::kGather,
    Opcode::kReduce,
    Opcode::kReduceScatter,
    Opcode::kScatter,
  });
  return collective_map->find(opcode) != collective_map->end();
}

bool OpcodeIsIndividualCommunication(Opcode opcode) {
  static auto* individual_map = new absl::flat_hash_set<Opcode>({
    Opcode::kSend,
    Opcode::kRecv,
    Opcode::kSendRecv,
  });
  return individual_map->find(opcode) != individual_map->end();
}

bool OpcodeIsControlFlow(Opcode opcode) {
  static auto* control_flow_map = new absl::flat_hash_set<Opcode>({
    Opcode::kCall,
    Opcode::kConditional,
    Opcode::kWhile,
  });
  return control_flow_map->find(opcode) != control_flow_map->end();
}

bool OpcodeIsGeneralPurpose(Opcode opcode) {
  static auto* general_map = new absl::flat_hash_set<Opcode>({
    Opcode::kNull,
    Opcode::kDelay,
    Opcode::kInfeed,
    Opcode::kOutfeed,
  });
  return general_map->find(opcode) != general_map->end();
}

}  // namespace paragraph
