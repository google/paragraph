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
#include "paragraph/graph/graph.h"

#include <algorithm>
#include <filesystem>
#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"

namespace paragraph {

Graph::Graph(const std::string& name, int64_t processor_id)
    : name_(name),
      processor_id_(processor_id),
      next_unique_id_(1),
      entry_subroutine_(nullptr),
      implicit_entry_instruction_(Instruction(Opcode::kNull, name, nullptr)) {
  implicit_entry_instruction_.SetId(0);
}

Subroutine* Graph::GetEntrySubroutine() const {
  CHECK_NE(nullptr, entry_subroutine_);
  return entry_subroutine_.get();
}

void Graph::SetEntrySubroutine(std::unique_ptr<Subroutine> subroutine) {
  CHECK_EQ(nullptr, entry_subroutine_);
  entry_subroutine_ = std::move(subroutine);
  entry_subroutine_->SetCallingInstruction(&implicit_entry_instruction_);
  // Entry subroutine will have ID=0 until it has root that sets its ID
  if (entry_subroutine_->GetRootInstruction() == nullptr) {
    entry_subroutine_->SetId(0);
  }
}

const std::vector<Subroutine*> Graph::Subroutines() const {
  std::vector<Subroutine*> subroutines;
  CHECK_NE(entry_subroutine_, nullptr);
  subroutines = entry_subroutine_->MakeEmbeddedSubroutinesVector();
  subroutines.push_back(entry_subroutine_.get());
  return subroutines;
}

const std::vector<Instruction*> Graph::InstructionsPostOrder() const {
  CHECK_NE(entry_subroutine_, nullptr);
  absl::flat_hash_map<Instruction*, bool> visited;
  std::vector<Instruction*> postorder;
  entry_subroutine_->PostOrderHelper(&visited, &postorder);
  return postorder;
}

void Graph::ApplyCommunicationTags() {
  absl::flat_hash_map<uint64_t, uint64_t> tag_counters;
  for (Instruction* instruction : InstructionsPostOrder()) {
    Opcode opcode = instruction->GetOpcode();
    // Gets the src->dst ID, returns if we can ignore this instruction.
    int64_t src, dst;
    switch (opcode) {
      case Opcode::kRecvStart:
      case Opcode::kRecvDone:
        src = instruction->PeerId().value();
        dst = instruction->GetGraph()->GetProcessorId();
        break;
      case Opcode::kSendStart:
      case Opcode::kSendDone:
        src = instruction->GetGraph()->GetProcessorId();
        dst = instruction->PeerId().value();
        break;
      default:
        continue;
        break;
    }
    CHECK(src >= 0 && src < 1ll << 32);
    CHECK(dst >= 0 && dst < 1ll << 32);
    uint64_t srcdst = ((static_cast<uint64_t>(src) << 32) |
                       static_cast<uint64_t>(dst));
    uint64_t comm_tag;
    switch (opcode) {
      case Opcode::kRecvStart:
      case Opcode::kSendStart:
        comm_tag = ++tag_counters[srcdst];
        break;
      case Opcode::kRecvDone:
      case Opcode::kSendDone:
        comm_tag = instruction->GetBondedInstruction()->GetCommunicationTag();
        break;
      default:
        CHECK(false);
        break;
    }
    instruction->SetCommunicationTag(comm_tag);
  }
}

void Graph::PostOrderEnforcer() {
  for (auto& subroutine : Subroutines()) {
    if (subroutine->Instructions().size() > 1) {
      std::vector<Instruction*> postorder =
          subroutine->InstructionsPostOrder(true);
      CHECK_EQ(subroutine->Instructions().size(), postorder.size());
      for (size_t instr_ind = 1; instr_ind < postorder.size(); ++instr_ind) {
        postorder.at(instr_ind)->AddOperand(postorder.at(instr_ind - 1));
      }
    }
  }
}

void Graph::SetInstructionId(Instruction* instr_ptr) {
  int64_t result = next_unique_id_;
  next_unique_id_++;
  instr_ptr->SetId(result);
}

int64_t Graph::SubroutineCount() const {
  if (entry_subroutine_ == nullptr) {
    return 0;
  }
  return Subroutines().size();
}

int64_t Graph::InstructionCount() const {
  int64_t n = 0;
  if (entry_subroutine_ == nullptr) {
    return n;
  }
  for (const auto& subroutine : Subroutines()) {
    n += subroutine->InstructionCount();
  }
  return n;
}

absl::flat_hash_map<int64_t, const Instruction*> Graph::InstructionIdMap()
    const {
  absl::flat_hash_map<int64_t, const Instruction*> mapping;
  for (const Subroutine* subroutine : Subroutines()) {
    for (const auto& instruction : subroutine->Instructions()) {
      mapping[instruction->GetId()] = instruction.get();
    }
  }
  return mapping;
}

bool Graph::HasConsecutiveNaturalProcessorIds() const {
  for (size_t i = 0; i < comm_set_.size(); ++i) {
    if (comm_set_.find(i) == comm_set_.end()) {
      return false;
    }
  }
  return true;
}

std::vector<const Instruction*> Graph::FindByName(const std::string& name)
    const {
  std::vector<const Instruction*> matches;
  for (const Subroutine* subroutine : Subroutines()) {
    for (const auto& instruction : subroutine->Instructions()) {
      if (instruction->GetName() == name) {
        matches.push_back(instruction.get());
      }
    }
  }
  return matches;
}

bool Graph::CommunicationSetHasProcessor(int64_t processor_id) const {
  return comm_set_.find(processor_id) != comm_set_.end();
}

shim::StatusOr<GraphProto> Graph::ToProto() const {
  GraphProto proto;
  proto.set_name(name_);
  proto.set_processor_id(processor_id_);

  RETURN_IF_FALSE(entry_subroutine_ != nullptr,
                  absl::InternalError) << "Can't serialize graph without " <<
      "entry subroutine.";
  ASSIGN_OR_RETURN(SubroutineProto entry_subroutine_proto,
                   entry_subroutine_->ToProto());
  *proto.mutable_entry_subroutine() = entry_subroutine_proto;
  return proto;
}

shim::StatusOr<std::unique_ptr<Graph>> Graph::CreateFromProto(
      const GraphProto& proto, bool reset_ids) {
  auto graph = absl::make_unique<Graph>(proto.name(), proto.processor_id());
  ASSIGN_OR_RETURN(auto entry_subroutine,
                   Subroutine::CreateFromProto(proto.entry_subroutine(),
                                               graph.get(), reset_ids));
  graph->SetEntrySubroutine(std::move(entry_subroutine));
  RETURN_IF_TRUE(graph->entry_subroutine_ == nullptr,
                  absl::InternalError) <<
                  "Graph entry subroutine not found.";
  graph->next_unique_id_ = graph->InstructionCount() + 1;
  for (auto& subroutine : graph->Subroutines()) {
    for (auto& instruction : subroutine->Instructions()) {
      graph->next_unique_id_ = std::max(graph->next_unique_id_,
                                        instruction->GetId() + 1);
    }
  }
  if (graph->IsIndividualized()) {
    RETURN_IF_ERROR(graph->ValidateIndividualized());
  } else {
    RETURN_IF_ERROR(graph->ValidateComposite());
  }
  return graph;
}

shim::StatusOr<std::unique_ptr<Graph>> Graph::ReadFromFile(
      const std::string& filename) {
  std::fstream input(filename, std::ios::in | std::ios::binary);
  if (!input) {
    return absl::NotFoundError(
        "File '" + filename + "' could not be opened.");
  }

  GraphProto graph_proto;
  std::string extension = std::filesystem::path(filename).extension();
  if (extension == Graph::kBinaryProtoExtension) {
    if (!graph_proto.ParseFromIstream(&input)) {
      return absl::InvalidArgumentError(
          "Failed to parse graph from binary protobuf file: " + filename);
    }
  } else if (extension == Graph::kTextProtoExtension) {
    google::protobuf::io::IstreamInputStream iis(&input);
    if (!google::protobuf::TextFormat::Parse(&iis, &graph_proto)) {
      return absl::InvalidArgumentError(
          "Failed to parse graph from text protobuf file: " + filename);
    }
  } else {
    return absl::InvalidArgumentError("Invalid extension: " + filename);
  }
  ASSIGN_OR_RETURN(auto graph,
                   Graph::CreateFromProto(graph_proto));
  return graph;
}

absl::Status Graph::WriteToFile(const std::string& filename) const {
  ASSIGN_OR_RETURN(auto graph_proto,
                   ToProto());
  std::fstream output(filename,
                      std::ios::out | std::ios::trunc | std::ios::binary);
  if (!output) {
    return absl::NotFoundError(
        "File '" + filename + "' could not be written.");
  }

  std::string extension = std::filesystem::path(filename).extension();
  if (extension == Graph::kBinaryProtoExtension) {
    if (!graph_proto.SerializeToOstream(&output)) {
      return absl::InvalidArgumentError(
          "Failed to write graph to binary protobuf file: " + filename);
    }
  } else if (extension == Graph::kTextProtoExtension) {
    google::protobuf::io::OstreamOutputStream oos(&output);
    if (!google::protobuf::TextFormat::Print(graph_proto, &oos)) {
      return absl::InvalidArgumentError(
          "Failed to write graph to text protobuf file: " + filename);
    }
  } else {
    return absl::InvalidArgumentError("Invalid extension: " + filename);
  }
  return absl::OkStatus();
}

shim::StatusOr<std::unique_ptr<Graph>> Graph::Clone(
    const std::string& name_suffix, bool reset_ids) {
  ASSIGN_OR_RETURN(GraphProto proto, ToProto());
  ASSIGN_OR_RETURN(std::unique_ptr<Graph> new_graph,
                   CreateFromProto(proto, reset_ids));
  new_graph->SetName(absl::StrCat(name_, name_suffix));
  return new_graph;
}

shim::StatusOr<std::unique_ptr<Graph>> Graph::Individualize(
    int64_t processor_id) const {
  RETURN_IF_ERROR(ValidateComposite());
  ASSIGN_OR_RETURN(GraphProto proto, ToProto());
  ASSIGN_OR_RETURN(std::unique_ptr<Graph> new_graph,
                   CreateFromProto(proto));
  new_graph->SetName(absl::StrCat(name_, "_",  processor_id));
  new_graph->SetProcessorId(processor_id);
  new_graph->comm_set_.clear();
  for (auto& subroutine : new_graph->Subroutines()) {
    for (auto& instruction : subroutine->Instructions()) {
      if (!OpcodeIsControlFlow(instruction->GetOpcode()) &&
          !OpcodeIsGeneralPurpose(instruction->GetOpcode())) {
        if (instruction->CommunicationGroupVectorHasProcessor(processor_id)) {
          ASSIGN_OR_RETURN(auto coord,
                           instruction->GetProcessorCoordinates(processor_id));
          auto comm_group =
              instruction->GetCommunicationGroupVector().at(coord.group);
          if (instruction->GetOpcode() == Opcode::kSend ||
              instruction->GetOpcode() == Opcode::kSendStart ||
              instruction->GetOpcode() == Opcode::kSendDone) {
            RETURN_IF_FALSE(comm_group.size() == 2,
                            absl::InternalError) << "Instruction " <<
                instruction->GetName() <<
                " should have CommunicationGroup of size 2.";
            instruction->ClearCommunicationGroupVector();
            // If the current processor is the sender, then keep the send and
            // put its peer to the communication group.
            // If the current processor is the receiver, swap the send to
            // null, as there should be a matching recv in the graph
            RETURN_IF_FALSE(comm_group.at(0) == processor_id,
                            absl::InternalError) << "Processor with the ID " <<
                processor_id << " is not sender of the instruction " <<
                instruction->GetName() << ", can't procede individualization.";
            instruction->AppendCommunicationGroup({comm_group.at(1)});
          }
          if (instruction->GetOpcode() == Opcode::kRecv ||
              instruction->GetOpcode() == Opcode::kRecvStart ||
              instruction->GetOpcode() == Opcode::kRecvDone) {
            RETURN_IF_FALSE(comm_group.size() == 2,
                            absl::InternalError) << "Instruction " <<
                instruction->GetName() <<
                " should have CommunicationGroup of size 2.";
            instruction->ClearCommunicationGroupVector();
            // If the current processor is the receiver, then keep the recv and
            // put its peer to the communication group.
            // If the current processor is the sender, swap the recv to
            // null, as there should be a matching send in the graph
            RETURN_IF_FALSE(comm_group.at(1) == processor_id,
                            absl::InternalError) << "Processor with the ID " <<
                processor_id << " is not receiver of the instruction " <<
                instruction->GetName() << ", can't procede individualization.";
            instruction->AppendCommunicationGroup({comm_group.at(0)});
          }
          if (instruction->GetOpcode() == Opcode::kSendRecv) {
            RETURN_IF_FALSE(comm_group.size() == 3,
                            absl::InternalError) << "Instruction " <<
                instruction->GetName() <<
                " should have CommunicationGroup of size 3.";
            RETURN_IF_FALSE(comm_group.at(1) == processor_id,
                            absl::InternalError) << "Processor with the ID " <<
                processor_id << " is not middle point of the instruction " <<
                instruction->GetName() << ", can't procede individualization.";
            instruction->ClearCommunicationGroupVector();
            instruction->AppendCommunicationGroup(
                {comm_group.at(0), comm_group.at(2)});
          }
          if (OpcodeIsCollectiveCommunication(instruction->GetOpcode()) &&
              comm_group.size() > 1) {
            // Just drop the communication groups that don't include current
            // processor if communication size is greater than one.
            // Otherwise, there is no real communication involved, and it can
            // be swapped to Null instruction
            instruction->ClearCommunicationGroupVector();
            instruction->AppendCommunicationGroup(comm_group);
          }
        } else {
          // If the processor does not participate anyhow in this communication,
          // swap it to the null node
          instruction->ClearCommunicationGroupVector();
          instruction->opcode_ = Opcode::kNull;
        }
      }
    }
  }
  return new_graph;
}

absl::Status Graph::ValidateCommon() const {
  // Sets to check ID uniqueness
  absl::flat_hash_set<int64_t> instruction_ids;
  RETURN_IF_FALSE(entry_subroutine_ != nullptr,
                  absl::InternalError) <<
      "Graph should have an entry subroutine.";
  for (const auto& subroutine : Subroutines()) {
    // Check that all subroutines have the same parent graph
    RETURN_IF_FALSE(subroutine->GetGraph() == this,
                    absl::InternalError) << "Subroutine " <<
        subroutine->GetName() << " points to a different graph.";
    RETURN_IF_ERROR(subroutine->ValidateCommon());
    for (const auto& instruction : subroutine->Instructions()) {
      // Check instruction ID uniqueness
      RETURN_IF_FALSE(instruction_ids.find(instruction->GetId())
                      == instruction_ids.end(), absl::InternalError) <<
          "Instruction " << instruction->GetName() << " ID = "
          << instruction->GetId() << " is not unique.";
      instruction_ids.insert(instruction->GetId());
    }
  }
  return absl::OkStatus();
}

absl::Status Graph::ValidateComposite() const {
  RETURN_IF_ERROR(ValidateCommon());
  for (const auto& subroutine : Subroutines()) {
    RETURN_IF_ERROR(subroutine->ValidateComposite());
  }
  return absl::OkStatus();
}

absl::Status Graph::ValidateIndividualized() const {
  RETURN_IF_ERROR(ValidateCommon());
  RETURN_IF_FALSE(processor_id_ >= 0,
                  absl::InternalError) << "Individualized graph shuold " <<
      "have a valid Processor Id.";
  for (const auto& subroutine : Subroutines()) {
    RETURN_IF_ERROR(subroutine->ValidateIndividualized());
  }
  return absl::OkStatus();
}

}  // namespace paragraph
