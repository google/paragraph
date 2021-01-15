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
#include "paragraph/graph/subroutine.h"

#include <algorithm>
#include <utility>

#include "absl/container/flat_hash_set.h"

namespace paragraph {

constexpr int64_t Subroutine::kDefaultId;

Subroutine::Subroutine(const std::string& name, Graph* parent)
    : name_(name),
      id_(kDefaultId),
      execution_probability_(1.0),
      execution_count_(1),
      root_instruction_(nullptr),
      calling_instruction_(nullptr),
      graph_(parent) {}

absl::Status Subroutine::AddInstruction(
    std::unique_ptr<Instruction> instruction) {
  instruction->SetParent(this);
  auto instr_ptr = instruction.get();
  RETURN_IF_FALSE(instruction_iterators_.find(
      instr_ptr) == instruction_iterators_.end(), absl::InvalidArgumentError)
      << "Can't insert instruction " << instruction->GetName()
      << " because it already exists in the subroutine.";
  instruction_iterators_[instr_ptr] = instructions_.insert(
      instructions_.end(), std::move(instruction));
  return absl::OkStatus();
}

absl::Status Subroutine::SetRootInstruction(Instruction* instr) {
  if (PREDICT_TRUE(instr != nullptr)) {
    RETURN_IF_FALSE(instruction_iterators_.find(instr) !=
                    instruction_iterators_.end(),
                    absl::InvalidArgumentError) << "Instruction " <<
        instr->GetName() << " not found in the subroutine and can't be set" <<
        " as root for subroutine " << name_ << ".";
    RETURN_IF_FALSE(instr->GetId() != Instruction::kDefaultId,
                    absl::InvalidArgumentError) << "Instruction " <<
        instr->GetName() << " has Default ID = " << Instruction::kDefaultId <<
        "  and can't be set as root for subroutine " << name_ << ".";
    id_ = instr->GetId();
  }
  root_instruction_ = instr;
  return absl::OkStatus();
}

void Subroutine::PostOrderHelper(
    absl::flat_hash_map<Instruction*, bool>* visited,
    std::vector<Instruction*>* postorder,
    bool skip_inner_subroutines) {
  CHECK_NE(root_instruction_, nullptr);
  root_instruction_->PostOrderHelper(
      visited, postorder, skip_inner_subroutines);
  // Check that all instructions in the subroutine are in post order.
  // If some instructions were not found via DFS from root instruction, we need
  // to pop last instruction, add all the missed instructions, insert root again
  postorder->pop_back();
  for (auto& instruction : instructions_) {
    if (visited->find(instruction.get()) == visited->end()) {
      postorder->push_back(instruction.get());
    }
  }
  postorder->push_back(root_instruction_);
}

std::vector<Instruction*> Subroutine::InstructionsPostOrder(
      bool skip_inner_subroutines) {
  absl::flat_hash_map<Instruction*, bool> visited;
  std::vector<Instruction*> postorder;
  PostOrderHelper(&visited, &postorder, skip_inner_subroutines);
  return postorder;
}

std::vector<Subroutine*> Subroutine::MakeEmbeddedSubroutinesVector() const {
  std::vector<Subroutine*> subroutines;
  EmbeddedSubroutines(&subroutines);
  return subroutines;
}

void Subroutine::EmbeddedSubroutines(
    std::vector<Subroutine*>* subroutines) const {
  for (const auto& instr : instructions_) {
    for (const auto& subroutine : instr->InnerSubroutines()) {
      subroutine->EmbeddedSubroutines(subroutines);
      CHECK_NE(subroutine.get(), nullptr);
      subroutines->push_back(subroutine.get());
    }
  }
}

void Subroutine::RemoveInstruction(Instruction* instruction) {
  CHECK(GetRootInstruction() != instruction);
  CHECK_EQ(instruction->UserCount(), 0);

  auto inst_it = instruction_iterators_.find(instruction);
  CHECK(inst_it != instruction_iterators_.end());
  CHECK_EQ(instruction, (*inst_it->second).get());
  // Release subroutines
  for (auto& subroutine : instruction->InnerSubroutines()) {
    instruction->RemoveInnerSubroutine(subroutine.get());
  }
  instruction->SetParent(nullptr);
  instruction_iterators_.erase(inst_it);
  instructions_.erase(inst_it->second);
  inst_it->second->release();
}

absl::Status Subroutine::ReplaceInstructionWithInstructionList(
    Instruction* target_instr_ptr,
    InstructionList* instructions) {
  RETURN_IF_FALSE(root_instruction_ != nullptr,
                  absl::InternalError) <<
      "Replaced instruction does not have root subroutine.";

  // Move operands
  for (auto& operand : target_instr_ptr->operands_) {
    RETURN_IF_ERROR(operand->RemoveUser(target_instr_ptr));
    instructions->front()->AddOperand(operand);
  }

  // Move users
  for (auto user : target_instr_ptr->users_) {
    std::replace(user->operands_.begin(), user->operands_.end(),
                 target_instr_ptr, instructions->back().get());
    instructions->back()->AddUser(user);
  }

  if (root_instruction_ == target_instr_ptr) {
    root_instruction_ = instructions->back().get();
  }

  // Replace instruction with instructions from vector
  for (auto & instr : *instructions) {
    RETURN_IF_ERROR(AddInstruction(std::move(instr)));
  }

  // Cleanup local containers
  target_instr_ptr->users_.clear();
  target_instr_ptr->user_map_.clear();
  instructions_.erase(instruction_iterators_.at(target_instr_ptr));
  instruction_iterators_.erase(target_instr_ptr);
  return absl::OkStatus();
}

void Subroutine::ScalePerformance(double scale) {
  for (auto& instr : instructions_) {
    instr->ScalePerformance(scale);
  }
}

shim::StatusOr<SubroutineProto> Subroutine::ToProto() const {
  SubroutineProto proto;
  RETURN_IF_FALSE(id_ != -1, absl::InternalError) <<
      "Subroutine should have ID != -1, and parent graph before" <<
      " serialization to protobuf.";
  proto.set_subroutine_root_id(id_);
  proto.set_name(name_);
  proto.set_execution_probability(execution_probability_);
  proto.set_execution_count(execution_count_);

  for (auto&& instr : instructions_) {
    ASSIGN_OR_RETURN(InstructionProto instr_proto, instr->ToProto());
    proto.add_instructions()->Swap(&instr_proto);
  }
  return proto;
}

shim::StatusOr<std::unique_ptr<Subroutine>> Subroutine::CreateFromProto(
    const SubroutineProto& proto, Graph* graph, bool reset_ids) {
  // Create a new subroutine base on protobuf data
  std::unique_ptr<Subroutine> subroutine = absl::make_unique<Subroutine>(
      proto.name(), graph);
  subroutine->id_ = proto.subroutine_root_id();
  subroutine->execution_probability_ = proto.execution_probability();
  subroutine->execution_count_ = proto.execution_count();

  // Iterate over all instructions that belongs to subroutine and create them
  // from protobufs inside the Subroutine protobuf
  // Add all the instructions to the map to connect them together later
  absl::flat_hash_map<int64_t, Instruction*> instruction_map;
  for (const InstructionProto& instruction_proto : proto.instructions()) {
    ASSIGN_OR_RETURN(std::unique_ptr<Instruction> instruction,
                     Instruction::CreateFromProto(instruction_proto,
                                                  subroutine.get(),
                                                  reset_ids));

    // Add instruction to the instruction map checking its id wasn't used before
    int64_t instruction_id = instruction_proto.instruction_id();
    RETURN_IF_FALSE(instruction_map.find(instruction_id) ==
                    instruction_map.end(), absl::InvalidArgumentError) <<
        "Instruction with ID " << instruction_id << " cannot be added to " <<
        "instruction map, this instruction ID was used before.";
    instruction_map[instruction_id] = instruction.get();

    // Check if instructions is root
    if (instruction_id == proto.subroutine_root_id()) {
      subroutine->root_instruction_ = instruction.get();
      if (reset_ids) {
        subroutine->id_ = instruction->GetId();
      } else {
        subroutine->id_ = proto.subroutine_root_id();
      }
    }
    RETURN_IF_ERROR(subroutine->AddInstruction(std::move(instruction)));
  }
  RETURN_IF_FALSE(subroutine->root_instruction_ != nullptr,
                  absl::InternalError) <<
      "Root instruction with ID = " << proto.subroutine_root_id() <<
      " not found.";

  // Connect all instructions in the new subroutine with its users and operands
  for (const InstructionProto& instruction_proto : proto.instructions()) {
    // First check for bonded instructions
    int64_t instruction_id = instruction_proto.instruction_id();
    RETURN_IF_FALSE(instruction_map.find(instruction_id) !=
                    instruction_map.end(), absl::InvalidArgumentError) <<
        "Instruction with ID " << instruction_id << " is not found in the " <<
        "instruction map. Make sure you create instruction map from protobuf.";
    auto instruction = instruction_map.at(instruction_id);

    // Connect instruction with its bonded instruction if protobuf has one
    int64_t bonded_instruction_id = instruction_proto.bonded_instruction_id();
    if (bonded_instruction_id != 0) {
      RETURN_IF_FALSE(instruction_map.find(bonded_instruction_id) !=
                      instruction_map.end(), absl::InvalidArgumentError) <<
          "Instruction with ID " << bonded_instruction_id << " is not found "
          "in the instruction map and can't be bonded with instructions with "
          "ID " << instruction_id << ". Make sure you create instruction map "
          "from protobuf.";
      auto bonded_instruction = instruction_map.at(bonded_instruction_id);
      instruction->BondWith(bonded_instruction);
    }
    // Connect instruction with its operands and users
    for (const int64_t operand_id : instruction_proto.operand_ids()) {
      RETURN_IF_FALSE(instruction_map.find(operand_id) != instruction_map.end(),
                      absl::InvalidArgumentError) << "Operand ID = " <<
          operand_id << " not found.";
      instruction->AddOperand(instruction_map.at(operand_id));
    }
  }
  return subroutine;
}

shim::StatusOr<std::unique_ptr<Subroutine>> Subroutine::Clone(
    const std::string& name_suffix, bool reset_ids) {
  ASSIGN_OR_RETURN(SubroutineProto proto, ToProto());
  ASSIGN_OR_RETURN(std::unique_ptr<Subroutine> subroutine,
                   CreateFromProto(proto, graph_, reset_ids));
  subroutine->name_ = absl::StrCat(name_, name_suffix);
  std::vector<Subroutine*> all_new_subs =
      subroutine->MakeEmbeddedSubroutinesVector();
  all_new_subs.push_back(subroutine.get());
  for (auto& subr : all_new_subs) {
    for (auto& instr : subr->Instructions()) {
      instr->SetName(absl::StrCat(instr->GetName(), name_suffix));
    }
  }
  return subroutine;
}

absl::Status Subroutine::ValidateCommon() const {
  // Check that subroutine has root instruction and its ID is equal to root ID
  RETURN_IF_FALSE(root_instruction_ != nullptr, absl::InternalError) <<
      "Subroutine " << name_ << " does not have a root instruction.";
  RETURN_IF_FALSE(root_instruction_->GetId() == id_, absl::InternalError) <<
      "Subroutine " << name_ << " should have the same ID as its root.";
  // Make sure that root instruction of each subroutine has no users, and can
  // yield to calling instruction not interrupted
  RETURN_IF_FALSE(root_instruction_->Users().empty(), absl::InternalError) <<
      "Subroutine " << name_ << " root instruction " <<
      root_instruction_->GetName() << " should have no users.";
  // Check that Instructions size is equal to InstructionCount that reflects
  // the size of InstructionIterator
  RETURN_IF_FALSE(instructions_.size() == instruction_iterators_.size(),
                  absl::InternalError) << "Subroutine returns a list of  " <<
      instructions_.size() << " instructions, but InstructionCount() = " <<
      instruction_iterators_.size() << ".";
  for (const auto& instruction : instructions_) {
    // Check that all instructions have the same parent subroutine
    RETURN_IF_FALSE(instruction->GetParent() == this,
                    absl::InternalError) << "Instruction " <<
        instruction->GetName() << " does not point to the parent subroutine "
        << name_ << ".";
    RETURN_IF_ERROR(instruction->ValidateCommon());
  }
  // Check that subroutine executes once if it's not called from Conditional
  if (calling_instruction_->GetOpcode() != Opcode::kConditional) {
    RETURN_IF_FALSE(execution_probability_ == 1.0,
                    absl::InternalError) << "Subroutine " << name_
        << " should have execution probability 1.0.";
  }
  // Check that each subroutine except if  called from While
  // executes exactly once
  if (calling_instruction_->GetOpcode() != Opcode::kWhile) {
    RETURN_IF_FALSE(execution_count_ == 1,
                    absl::InternalError) << "Subroutine " << name_
        << " should have execution count 1.";
  } else {
    RETURN_IF_FALSE(execution_count_ >= 1,
                    absl::InternalError) << "Subroutine " << name_
        << " called from 'while' instruction " << calling_instruction_->name_
        << " should have execution count >= 1.";
  }
  return absl::OkStatus();
}

absl::Status Subroutine::ValidateComposite() const {
  RETURN_IF_ERROR(ValidateCommon());
  for (const auto& instruction : instructions_) {
    RETURN_IF_ERROR(instruction->ValidateComposite());
  }
  return absl::OkStatus();
}

absl::Status Subroutine::ValidateIndividualized() const {
  RETURN_IF_ERROR(ValidateCommon());
  for (const auto& instruction : instructions_) {
    RETURN_IF_ERROR(instruction->ValidateIndividualized());
  }
  return absl::OkStatus();
}

}  // namespace paragraph
