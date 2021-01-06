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
#include "paragraph/bridging/hlo/hlo_converter.h"

#include <memory>
#include <string>
#include <vector>

#include "paragraph/bridging/hlo/hlo_compute_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/util.h"

xla::StatusOr<paragraph::GraphProto> HloModuleToGraphProto(
    const xla::HloModule* module,
    const ComputeCostAnalysis* cost_analysis,
    bool reset_ids);
xla::StatusOr<paragraph::SubroutineProto> HloComputationToSubroutineProto(
    const xla::HloComputation* computation,
    const ComputeCostAnalysis* cost_analysis,
    int64_t new_id);

const std::string HloToParagraphOpcodeString(
    xla::HloOpcode hlo_opcode) {
  if (hlo_opcode == xla::HloOpcode::kWhile) {
    return "while";
  } else if (hlo_opcode == xla::HloOpcode::kCall) {
    return "call";
  } else if (hlo_opcode == xla::HloOpcode::kConditional) {
    return "conditional";
  } else if (hlo_opcode == xla::HloOpcode::kAllGather) {
    return "all-gather";
  } else if (hlo_opcode == xla::HloOpcode::kAllReduce) {
    return "all-reduce";
  } else if (hlo_opcode == xla::HloOpcode::kAllToAll) {
    return "all-to-all";
  } else if (hlo_opcode == xla::HloOpcode::kSend) {
    return "send-start";
  } else if (hlo_opcode == xla::HloOpcode::kSendDone) {
    return "send-done";
  } else if (hlo_opcode == xla::HloOpcode::kRecv) {
    return "recv-start";
  } else if (hlo_opcode == xla::HloOpcode::kRecvDone) {
    return "recv-done";
  } else if (hlo_opcode == xla::HloOpcode::kInfeed) {
    return "infeed";
  } else if (hlo_opcode == xla::HloOpcode::kOutfeed) {
    return "outfeed";
  } else {
    return "delay";
  }
  return "delay";
}

xla::HloCostAnalysis::Properties GetPropertiesForInstruction(
    const int64_t id,
    const absl::flat_hash_map<int64_t, xla::HloCostAnalysis::Properties>&
        properties_map) {
  auto key_value = properties_map.find(id);
  if (key_value == properties_map.end()) {
    xla::HloCostAnalysis::Properties hlo_property = {};
    return hlo_property;
  } else {
    return key_value->second;
  }
}

float GetProperty(
    const std::string& key,
    const xla::HloCostAnalysis::Properties& properties,
    const float default_value) {
  auto key_value = properties.find(key);
  return key_value == properties.end() ? default_value : key_value->second;
}

void UpdateInstructionProtoProperties(
    paragraph::InstructionProto* instruction,
    const ComputeCostAnalysis::Properties& properties) {
  instruction->set_ops(GetProperty(
      ComputeCostAnalysis::kFlopsKey, properties, 0.0));
  instruction->set_transcendentals(GetProperty(
      ComputeCostAnalysis::kTranscendentalsKey, properties, 0.0));
  instruction->set_bytes_in(GetProperty(
      ComputeCostAnalysis::kOperandBytesAccessedKey, properties, 0.0));
  instruction->set_bytes_out(GetProperty(
      ComputeCostAnalysis::kOutputBytesAccessedKey, properties, 0.0));
  instruction->set_seconds(GetProperty(
      xla::HloCostAnalysis::kOptimalSecondsKey, properties, 0.0));
}

xla::StatusOr<paragraph::InstructionProto> HloInstructionToInstructionProto(
    const xla::HloInstruction* instruction,
    const ComputeCostAnalysis* cost_analysis,
    int64_t new_id) {
  paragraph::InstructionProto proto;
  proto.set_name(instruction->name());

  std::string new_opcode = HloToParagraphOpcodeString(instruction->opcode());
  // Paragraph delay instructions should be leaf nodes, of instructions have
  // inner subroutines, they should be mapped to call instruction
  if (new_opcode == "delay" && !instruction->called_computations().empty()) {
    proto.set_opcode("call");
  } else {
    proto.set_opcode(new_opcode);
  }

  // Set instruction performance properties
  proto.set_ops(cost_analysis->GetPropertyForHlo(
      *instruction, ComputeCostAnalysis::kFlopsKey));
  proto.set_transcendentals(cost_analysis->GetPropertyForHlo(
      *instruction, ComputeCostAnalysis::kTranscendentalsKey));
  proto.set_bytes_in(cost_analysis->GetPropertyForHlo(
      *instruction, ComputeCostAnalysis::kOperandBytesAccessedKey));
  proto.set_bytes_out(cost_analysis->GetPropertyForHlo(
      *instruction, ComputeCostAnalysis::kOutputBytesAccessedKey));
  proto.set_seconds(cost_analysis->GetPropertyForHlo(
      *instruction, xla::HloCostAnalysis::kOptimalSecondsKey));

  // Set inner subroutines passing new id counter to the function
  for (const auto& computation : instruction->called_computations()) {
    TF_ASSIGN_OR_RETURN(auto subroutine_proto,
                     HloComputationToSubroutineProto(computation,
                                                     cost_analysis,
                                                     new_id));
    // Set inner subroutines execution properties
    if (proto.opcode() == "conditional") {
      subroutine_proto.set_execution_probability(
          1.0 / instruction->called_computations().size());
    } else {
      subroutine_proto.set_execution_probability(1.0);
    }
    if (proto.opcode() == "while") {
      subroutine_proto.set_execution_count(1);
    } else {
      subroutine_proto.set_execution_count(1);
    }
    proto.add_inner_subroutines()->Swap(&subroutine_proto);
  }

  // Set communication group ids
  if (proto.opcode() == "all-gather" ||
      proto.opcode() == "all-reduce" ||
      proto.opcode() == "all-to-all") {
    for (auto& group : instruction->replica_groups()) {
      paragraph::CommunicationGroupProto comm_group;
      for (auto id : group.replica_ids()) {
        comm_group.add_group_ids(id);
      }
      proto.add_communication_groups()->Swap(&comm_group);
    }
  }
  if (proto.opcode() == "send-start" ||
      proto.opcode() == "send-done" ||
      proto.opcode() == "recv-start" ||
      proto.opcode() == "recv-done") {
    return xla::Unimplemented(
        "Translation for Send/Recv instructions is not available.");
  }
  // All-Reduce performance reporting is flawed in HLO, as instructions in
  // subroutines (HLO computations) don't have performance parameters (HLO
  // computations and their inner instructions could be reused).
  if (proto.opcode() == "all-reduce") {
    TF_RET_CHECK(instruction->called_computations().size() == 1);
    int64_t root_ind = 0;
    for (int64_t i = 0;
         i < proto.inner_subroutines(0).instructions_size();
         ++i) {
      if (proto.inner_subroutines(0).instructions(i).instruction_id() ==
          proto.inner_subroutines(0).subroutine_root_id()) {
        root_ind = i;
      }
    }
    proto.mutable_inner_subroutines(0)->mutable_instructions(
        root_ind)->set_ops(proto.ops());
    (*proto.mutable_inner_subroutines(0)).mutable_instructions(
        root_ind)->set_transcendentals(proto.transcendentals());
    (*proto.mutable_inner_subroutines(0)).mutable_instructions(
        root_ind)->set_bytes_in(proto.bytes_in());
    (*proto.mutable_inner_subroutines(0)).mutable_instructions(
        root_ind)->set_bytes_out(proto.bytes_out());
    (*proto.mutable_inner_subroutines(0)).mutable_instructions(
        root_ind)->set_seconds(proto.seconds());
  }
  return proto;
}

xla::StatusOr<paragraph::SubroutineProto> HloComputationToSubroutineProto(
    const xla::HloComputation* computation,
    const ComputeCostAnalysis* cost_analysis,
    int64_t new_id) {
  paragraph::SubroutineProto proto;

  // Create an id map that maps old instruction ids to new ones to maintain
  // unique instruction ids for every instruction, including instructions from
  // the repeated subroutines
  absl::flat_hash_map<int64_t, int64_t> instruction_ids_map;
  if (new_id != 0) {
    for (const xla::HloInstruction* instr
             : computation->MakeInstructionPostOrder()) {
      int64_t old_instr_id = instr->unique_id();
      int64_t new_instr_id = new_id++;
      TF_RET_CHECK(instruction_ids_map.find(old_instr_id) ==
                   instruction_ids_map.end());
      instruction_ids_map[old_instr_id] = new_instr_id;
    }
  }

  proto.set_name(computation->name());
  for (const xla::HloInstruction* instr
           : computation->MakeInstructionPostOrder()) {
    TF_ASSIGN_OR_RETURN(paragraph::InstructionProto instr_proto,
                        HloInstructionToInstructionProto(instr,
                                                         cost_analysis,
                                                         new_id));
    // If new_id == 0 then we keep ids from HLO instruction, otherwise we create
    // new unique ids for each instruction
    int64_t old_instr_id = instr->unique_id();
    if (new_id == 0) {
      instr_proto.set_instruction_id(old_instr_id);
    } else {
      TF_RET_CHECK(instruction_ids_map.find(old_instr_id) !=
                   instruction_ids_map.end());
      instr_proto.set_instruction_id(instruction_ids_map.at(old_instr_id));
    }
    // Check root id for current computation and set it if root is found
    if (old_instr_id == computation->root_instruction()->unique_id()) {
      proto.set_subroutine_root_id(instr_proto.instruction_id());
    }
    // Connect instruction with its operands
    for (const xla::HloInstruction* operand : instr->operands()) {
      TF_RET_CHECK(instruction_ids_map.find(operand->unique_id()) !=
                   instruction_ids_map.end());
      instr_proto.add_operand_ids(instruction_ids_map.at(operand->unique_id()));
    }
    proto.add_instructions()->Swap(&instr_proto);
  }
  // Check we found root for the subroutine
  TF_RET_CHECK(proto.subroutine_root_id() != 0);
  return proto;
}

xla::StatusOr<paragraph::GraphProto> HloModuleToGraphProto(
    const xla::HloModule* module,
    const ComputeCostAnalysis* cost_analysis,
    bool reset_ids) {
  paragraph::GraphProto proto;
  proto.set_name(module->name());
  proto.set_processor_id(-1);

  // If next_unique_id set to 0, we would pick the old ids from Hlo module.
  // Setting it to 1 we reset all ids in the Hlo
  int64_t next_unique_id = 0;
  if (reset_ids) {
    next_unique_id = 1;
  }

  TF_ASSIGN_OR_RETURN(
      auto entry_subroutine_proto,
      HloComputationToSubroutineProto(module->entry_computation(),
                                      cost_analysis,
                                      next_unique_id));
  entry_subroutine_proto.set_execution_probability(1.0);
  entry_subroutine_proto.set_execution_count(1);
  *proto.mutable_entry_subroutine() = entry_subroutine_proto;
  return proto;
}

xla::StatusOr<std::unique_ptr<ComputeCostAnalysis>> CreateAndRunCostAnalysis(
        const xla::HloModule* module,
        const ComputeCostAnalysis::Properties per_second_rates) {
  auto cost_analysis = absl::make_unique<ComputeCostAnalysis>(
      ShapeSize, per_second_rates);
  xla::HloComputation* hlo_computation = module->entry_computation();
  TF_CHECK_OK(hlo_computation->Accept(cost_analysis.get()));
  TF_CHECK_OK(cost_analysis->UpdateInstructionProperties());
  return cost_analysis;
}

xla::StatusOr<paragraph::GraphProto> HloConverter(
    const xla::HloModule* module,
    const ComputeCostAnalysis::Properties& per_second_rates) {
  TF_ASSIGN_OR_RETURN(auto cost_analysis,
                      CreateAndRunCostAnalysis(module, per_second_rates));
  TF_ASSIGN_OR_RETURN(auto paragraph_proto,
                      HloModuleToGraphProto(module,
                                            cost_analysis.get(),
                                            true));
  return paragraph_proto;
}
