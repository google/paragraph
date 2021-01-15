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
#include "paragraph/graph/instruction.h"

#include <cstdlib>
#include <utility>

#include "paragraph/graph/graph.h"
#include "paragraph/graph/opcode.h"
#include "paragraph/graph/subroutine.h"

namespace paragraph {

constexpr int64_t Instruction::kDefaultId;
constexpr double Instruction::kFloatTolerance;

Instruction::Instruction(Opcode opcode, const std::string& name,
                         Subroutine* parent)
    : name_(name),
      id_(kDefaultId),
      opcode_(opcode),
      ops_(0.0),
      transcendentals_(0.0),
      bytes_in_(0.0),
      bytes_out_(0.0),
      seconds_(0.0),
      bonded_instruction_(nullptr),
      parent_(parent),
      communication_tag_(0) {
}

shim::StatusOr<Instruction*> Instruction::Create(
      Opcode opcode,
      const std::string& name,
      Subroutine* parent,
      bool is_root) {
  RETURN_IF_FALSE(parent != nullptr, absl::InvalidArgumentError)
      << "Can't create instruction " << name << " without parent.";
  RETURN_IF_FALSE(parent->GetGraph() != nullptr, absl::InvalidArgumentError)
      << "Can't create instruction " << name << " with parent subroutine "
      << parent->GetName() << " that does not point to a graph.";
  // Using `new` and WrapUnique to access a non-public constructor.
  auto instruction = absl::WrapUnique(new Instruction(opcode, name, parent));
  auto instr_ptr = instruction.get();
  instruction->GetGraph()->SetInstructionId(instruction.get());
  RETURN_IF_ERROR(parent->AddInstruction(std::move(instruction)));
  if (is_root) {
    RETURN_IF_ERROR(parent->SetRootInstruction(instr_ptr));
  }
  return instr_ptr;
}

void Instruction::SetId(int64_t id) {
  CHECK_GE(id, 0);
  if (parent_ != nullptr && parent_->GetRootInstruction() == this) {
    parent_->SetId(id);
  }
  id_ = id;
}

bool Instruction::AddOperand(Instruction* operand) {
  if (operand->AddUser(this)) {
    operands_.push_back(operand);
    return true;
  }
  return false;
}

bool Instruction::AddUser(Instruction* user) {
  if (user_map_.find(user) == user_map_.end()) {
    user_map_.emplace(user, users_.size());
    users_.push_back(user);
    return true;
  }
  return false;
}

absl::Status Instruction::RemoveUser(Instruction* user) {
  auto map_it = user_map_.find(user);
  RETURN_IF_FALSE(map_it != user_map_.end(),
                  absl::InternalError) <<
      "User to be removed with ID = " << user->GetId() << " not found.";

  const int64_t index = map_it->second;
  RETURN_IF_FALSE(users_.at(index) == user,
                  absl::InternalError) <<
      "Wrong user with ID = " << user->GetId() << " found during removing.";

  // Move the last user into the position of the removed user.
  users_[index] = users_.back();
  user_map_[users_.back()] = index;

  // Remove the user from the map and drop the last slot from the vector what
  // have been moved to the position of the original user.
  user_map_.erase(map_it);
  users_.pop_back();
  return absl::OkStatus();
}

int64_t Instruction::UserId(Instruction* user) const {
  auto result = user_map_.find(user);
  CHECK(result != user_map_.end());
  return result->second;
}

void Instruction::PostOrderHelper(
    absl::flat_hash_map<Instruction*, bool>* visited,
    std::vector<Instruction*>* postorder,
    bool skip_inner_subroutines) {
  // Three situations are possible when we consider instruction:
  // 1) We have not visited the instruction, so we need to mark it visited, but
  // not finished
  // 2) We have already seen this instruction, but not finished processing it,
  // which means there is a cycle in the graph.
  // 3) We have already seen this instruction (i.e. as an operand of another
  // instruction we processed before), and finished processing it, so it is
  // already in post order vector
  if (visited->find(this) == visited->end()) {
    visited->emplace(this, false);
  } else {
    // If there is cycle, we are in case 2) and CHECK fails
    CHECK(visited->at(this));
    return;
  }
  // First traverse all the operands of the instruction
  for (auto& operand : operands_) {
    operand->PostOrderHelper(visited, postorder, skip_inner_subroutines);
  }
  // Then traverse all the inner subroutines of the instruction if flag
  // skip_inner_subroutines is not set
  if (!skip_inner_subroutines) {
    for (auto& subroutine : inner_subroutines_) {
      subroutine->PostOrderHelper(visited, postorder, skip_inner_subroutines);
    }
  }
  visited->at(this) = true;
  postorder->push_back(this);
}

Instruction* Instruction::GetBondedInstruction() {
  return bonded_instruction_;
}

const Instruction* Instruction::GetBondedInstruction() const {
  return bonded_instruction_;
}

void Instruction::BondWith(Instruction* instruction) {
  bonded_instruction_ = instruction;
  instruction->bonded_instruction_ = this;
}

void Instruction::ScalePerformance(double scale) {
  ops_ *= scale;
  transcendentals_ *= scale;
  bytes_in_ *= scale;
  bytes_out_ *= scale;
  seconds_ *= scale;
  for (auto& subroutine : inner_subroutines_) {
    subroutine->ScalePerformance(scale);
  }
}

void Instruction::AppendInnerSubroutine(
    std::unique_ptr<Subroutine> subroutine) {
  // Check that all instruction has a proper number of inner subrouines
  if (opcode_ != Opcode::kConditional && opcode_ != Opcode::kCall) {
    if (opcode_ == Opcode::kWhile) {
      CHECK_LT(inner_subroutines_.size(), 2) << "While instruction " <<
          name_ << " should have exactly 2 subroutines, can't add any more.";
    } else {
      CHECK(inner_subroutines_.empty()) << "Instruction " << name_ <<
          " should be empty before adding new subroutines.";
    }
  }
  subroutine->SetCallingInstruction(this);
  inner_subroutines_.push_back(std::move(subroutine));
}

void Instruction::RemoveInnerSubroutine(Subroutine* subroutine) {
  CHECK_OK(subroutine->SetRootInstruction(nullptr));
  while (!subroutine->Instructions().empty()) {
    subroutine->RemoveInstruction(
        (*subroutine->Instructions().begin()).get());
  }
  subroutine->SetGraph(nullptr);
  auto subroutine_it = std::find_if(
      inner_subroutines_.begin(), inner_subroutines_.begin(),
      [&](const std::unique_ptr<Subroutine>& subr) {
        return subr.get() == subroutine; });
  CHECK(subroutine_it != inner_subroutines_.end());
  CHECK_EQ(subroutine, (*subroutine_it).get());
  inner_subroutines_.erase(
      std::remove_if(inner_subroutines_.begin(), inner_subroutines_.end(),
                     [&](const std::unique_ptr<Subroutine>& subr) {
                       return subr.get() == subroutine;
                     }),
      inner_subroutines_.end());
}

absl::Status Instruction::ReplaceInnerSubroutine(
    Subroutine* target_subroutine, std::unique_ptr<Subroutine> new_subroutine) {
  auto subroutine_it = std::find_if(
      inner_subroutines_.begin(), inner_subroutines_.begin(),
      [&](const std::unique_ptr<Subroutine>& subr) {
        return subr.get() == target_subroutine; });
  RETURN_IF_FALSE(subroutine_it != inner_subroutines_.end(),
                  absl::InvalidArgumentError) << "Target subroutine " <<
      target_subroutine->GetName() << " is not found in instruction's " <<
      name_ << " inner subroutines.";
  RETURN_IF_FALSE(target_subroutine == (*subroutine_it).get(),
                  absl::InternalError) << "Target subroutine " <<
      target_subroutine->GetName() << "internal pointer is broken.";
  new_subroutine->SetCallingInstruction(this);
  (*subroutine_it).swap(new_subroutine);
  return absl::OkStatus();
}

const Graph* Instruction::GetGraph() const {
  if (parent_ != nullptr) {
    return parent_->GetGraph();
  }
  return nullptr;
}

Graph* Instruction::GetGraph() {
  if (parent_ != nullptr) {
    return parent_->GetGraph();
  }
  return nullptr;
}

void Instruction::ClearCommunicationGroupVector() {
  comm_group_vector_.clear();
  comm_group_vector_.shrink_to_fit();
  processor_id_to_index_map_.clear();
}

const CommunicationGroup& Instruction::GetCommunicationGroup() const {
  CHECK_EQ(comm_group_vector_.size(), 1);
  return comm_group_vector_.at(0);
}

void Instruction::AppendCommunicationGroup(const CommunicationGroup& group) {
  // For collective graph, we are adding each processor id for the matching
  // instruction only once. That means:
  // 1. For Collectives, we add all processor ids from the group
  // 2. For Send we add only sender, as we expect receiver to have matching Recv
  // or SendRecv instruction in the graph
  // 3. For Recv we add only receiver, as we expect sender to have matching Send
  // or SendRecv instruction in the graph
  // 4. For SendRecv, we add only middle node that orchestrates communication,
  // meaning receives from sender and sends to receiver
  // For individualized graph, we add all the processor ids in the communication
  // group, which should be single for each instruction
  if (OpcodeIsCollectiveCommunication(opcode_) ||
      ((opcode_ == Opcode::kSendRecv) && (group.size() == 2)) ||
      ((OpcodeIsProtocolLevelCommunication(opcode_) ||
      (opcode_ == Opcode::kSend) || (opcode_ == Opcode::kRecv)) &&
      (group.size() == 1))) {
    for (uint64_t processor_ind = 0;
         processor_ind < group.size();
         processor_ind++) {
      int64_t processor_id = group.at(processor_ind);
      if (opcode_ != Opcode::kSendRecv) {
        // processor id can't appear twice in the same comm group, except for
        // SendRecv which could send to and recv from the same node
        CHECK(processor_id_to_index_map_.find(processor_id) ==
              processor_id_to_index_map_.end());
      }
      ProcessorCoordinates coordinates;
      coordinates.group = comm_group_vector_.size();
      coordinates.offset = processor_ind;
      processor_id_to_index_map_[processor_id] = coordinates;
      CHECK_NE(GetGraph(), nullptr);
      GetGraph()->comm_set_.insert(processor_id);
    }
  } else if (OpcodeIsIndividualCommunication(opcode_) ||
             OpcodeIsProtocolLevelCommunication(opcode_)) {
    uint64_t processor_ind = UINT64_MAX;
    if (opcode_ == Opcode::kSend ||
        opcode_ == Opcode::kSendStart ||
        opcode_ == Opcode::kSendDone) {
      processor_ind = 0;
    } else if (opcode_ == Opcode::kRecv ||
        opcode_ == Opcode::kRecvStart ||
        opcode_ == Opcode::kRecvDone ||
        opcode_ == Opcode::kSendRecv) {
      processor_ind = 1;
    } else {
      CHECK(false) << "Unknown communication opcode";
    }
    int64_t processor_id = group.at(processor_ind);
    CHECK(processor_id_to_index_map_.find(processor_id) ==
          processor_id_to_index_map_.end());
    ProcessorCoordinates coordinates;
    coordinates.group = comm_group_vector_.size();
    coordinates.offset = processor_ind;
    processor_id_to_index_map_[processor_id] = coordinates;
    CHECK_NE(GetGraph(), nullptr);
    GetGraph()->comm_set_.insert(processor_id);
  } else {
    CHECK(false)
        << "only communication instructions can have communication groups";
  }
  comm_group_vector_.push_back(group);
}

shim::StatusOr<int64_t> Instruction::PeerId() const {
  // For Send/Recv instructions PeerId should be peer processor id, coming from
  // CommunicationGroup of size 1 that encodes it.
  RETURN_IF_FALSE(opcode_ == Opcode::kSend ||
                  opcode_ == Opcode::kSendStart ||
                  opcode_ == Opcode::kSendDone ||
                  opcode_ == Opcode::kRecv ||
                  opcode_ == Opcode::kRecvStart ||
                  opcode_ == Opcode::kRecvDone,
                  absl::InternalError) <<
      "Only Send/Recv(Done) instructions can return PeerId; current" <<
      " opcode is " << OpcodeToString(opcode_) << ".";
  RETURN_IF_FALSE(comm_group_vector_.size() == 1,
                  absl::InternalError) <<
      "There should be only a single CommunicationGroup.";
  const CommunicationGroup& group = comm_group_vector_.at(0);
  RETURN_IF_FALSE(group.size() == 1,
                  absl::InternalError) <<
      " CommunicationGroup size should be equal to 1.";
  return group.at(0);
}

shim::StatusOr<ProcessorCoordinates> Instruction::GetProcessorCoordinates(
    int64_t processor_id) const {
  if (!this->CommunicationGroupVectorHasProcessor(processor_id)) {
    return absl::InvalidArgumentError(
        "Processor index for supplied ProcessorId " +
        std::to_string(processor_id) +
        " not found in instruction " + name_ + "\n");
  }
  return processor_id_to_index_map_.at(processor_id);
}

int64_t Instruction::GetProcessorIndex(int64_t processor_id) const {
  auto processor_coord = GetProcessorCoordinates(processor_id);
  CHECK_OK(processor_coord.status());
  return processor_coord.value().offset;
}

bool Instruction::CommunicationGroupVectorHasProcessor(
    int64_t processor_id) const {
  return processor_id_to_index_map_.find(processor_id) !=
      processor_id_to_index_map_.end();
}

shim::StatusOr<InstructionProto> Instruction::ToProto() const {
  InstructionProto proto;
  RETURN_IF_FALSE(id_ != -1, absl::InternalError) <<
      "Instruction " << name_ << " should have ID != -1, parent subroutine, "
      "and be part of a graph before serialization to protobuf.";
  proto.set_instruction_id(id_);
  proto.set_name(name_);
  proto.set_opcode(OpcodeToString(opcode_));
  if (bonded_instruction_ != nullptr) {
    proto.set_bonded_instruction_id(bonded_instruction_->GetId());
  }

  proto.set_ops(ops_);
  proto.set_bytes_in(bytes_in_);
  proto.set_bytes_out(bytes_out_);
  proto.set_transcendentals(transcendentals_);
  proto.set_seconds(seconds_);

  for (const Instruction* operand : operands_) {
    proto.add_operand_ids(operand->GetId());
  }
  for (const auto& subroutine : inner_subroutines_) {
    ASSIGN_OR_RETURN(SubroutineProto subroutine_proto,
                     subroutine->ToProto());
    proto.add_inner_subroutines()->Swap(&subroutine_proto);
  }

  for (const CommunicationGroup& group : comm_group_vector_) {
    CommunicationGroupProto proto_group;
    for (int64_t processor_id : group) {
      proto_group.add_group_ids(processor_id);
    }
    proto.add_communication_groups()->Swap(&proto_group);
  }

  proto.set_communication_tag(communication_tag_);
  return proto;
}

shim::StatusOr<std::unique_ptr<Instruction>> Instruction::CreateFromProto(
    const InstructionProto& proto, Subroutine* parent, bool reset_ids) {
  // Extract from protobuf all necessary data to create a new instruction
  ASSIGN_OR_RETURN(Opcode opcode, StringToOpcode(proto.opcode()));
  // Using `new` and WrapUnique to access a non-public constructor.
  auto instruction = absl::WrapUnique(new Instruction(
      opcode, proto.name(), parent));
  if (reset_ids) {
    RETURN_IF_FALSE(instruction->GetGraph() != nullptr,
                    absl::InvalidArgumentError) <<
        "No graph to create instruction with reset IDs.";
    instruction->GetGraph()->SetInstructionId(instruction.get());
  } else {
    instruction->id_ = proto.instruction_id();
  }

  // Add performance properties
  instruction->SetOps(proto.ops());
  instruction->SetBytesIn(proto.bytes_in());
  instruction->SetBytesOut(proto.bytes_out());
  instruction->SetTranscendentals(proto.transcendentals());
  instruction->SetSeconds(proto.seconds());

  // Add CommunicationGroups
  for (const CommunicationGroupProto& group : proto.communication_groups()) {
    CommunicationGroup new_group;
    for (int64_t processor_id : group.group_ids()) {
      new_group.push_back(processor_id);
    }
    instruction->AppendCommunicationGroup(new_group);
  }

  // Connect instruction with its inner subroutines
  for (const SubroutineProto& subroutine_proto : proto.inner_subroutines()) {
    ASSIGN_OR_RETURN(std::unique_ptr<Subroutine> subroutine,
                     Subroutine::CreateFromProto(subroutine_proto,
                                                 instruction->GetGraph(),
                                                 reset_ids));
    subroutine->SetCallingInstruction(instruction.get());
    instruction->inner_subroutines_.push_back(std::move(subroutine));
  }

  instruction->SetCommunicationTag(proto.communication_tag());
  return instruction;
}

shim::StatusOr<std::unique_ptr<Instruction>> Instruction::Clone(
    const std::string& name_suffix, bool reset_ids) {
  ASSIGN_OR_RETURN(InstructionProto proto, ToProto());
  ASSIGN_OR_RETURN(std::unique_ptr<Instruction> instruction,
                   CreateFromProto(proto, parent_, reset_ids));
  instruction->name_ = absl::StrCat(name_, name_suffix);
  if (reset_ids) {
    instruction->GetGraph()->SetInstructionId(instruction.get());
  }
  for (auto& inner : instruction->InnerSubroutines()) {
    std::vector<Subroutine*> all_new_subs =
        inner->MakeEmbeddedSubroutinesVector();
    all_new_subs.push_back(inner.get());
    for (auto& subr : all_new_subs) {
      for (auto& instr : subr->Instructions()) {
        instr->SetName(absl::StrCat(instr->GetName(), name_suffix));
      }
    }
  }
  return instruction;
}

absl::Status Instruction::ValidateCommon() const {
  // Check performance parameters of the instruction
  RETURN_IF_FALSE(ops_ >=0, absl::InternalError) << "Instruction "
      << name_ << " should have non-negative number of arithmetic operations.";
  RETURN_IF_FALSE(transcendentals_ >=0, absl::InternalError) << "Instruction "
      << name_ << " should have non-negative number of transcendental "
      "operations.";
  RETURN_IF_FALSE(bytes_in_ >=0, absl::InternalError) << "Instruction "
      << name_ << " should have non-negative number of bytes read.";
  RETURN_IF_FALSE(bytes_in_ >=0, absl::InternalError) << "Instruction "
      << name_ << " should have non-negative number of bytes written.";
  RETURN_IF_FALSE(seconds_ >=0, absl::InternalError) << "Instruction "
      << name_ << " should have non-negative execution time.";

  // Check instruction ID is not default
  RETURN_IF_TRUE(id_ == kDefaultId, absl::InternalError) <<
      "Instruction should have non-default ID, currently ID = -1.";
  // Check instruction parent is not default nullptr
  RETURN_IF_TRUE(parent_ == nullptr, absl::InternalError) <<
      "Instruction should have parent subroutine.";
  // Check instruction is connected to the graph
  RETURN_IF_TRUE(GetGraph() == nullptr, absl::InternalError) <<
      "Instruction should be linked to graph.";
  // Check that Users size is equal to user_map_ size - an internal map that
  // maps all the users of instruction to their IDs
  RETURN_IF_FALSE(users_.size() == user_map_.size(), absl::InternalError) <<
      "Instruction returns a list of  " << users_.size()
      << " users, but only gave internal mapping for " <<
      user_map_.size() << ".";
  // Check that all inner subroutine for an instruction have the instruction
  // as their calling instruction and have current graph as their parent
  for (const auto& inner_sub : InnerSubroutines()) {
    RETURN_IF_ERROR(inner_sub->ValidateCommon());
    RETURN_IF_FALSE(inner_sub->GetCallingInstruction() == this,
                    absl::InternalError) << "Inner subroutine " <<
        inner_sub->GetName() << " does not point to the calling instruction "
        << name_ << ".";
    RETURN_IF_FALSE(inner_sub->GetGraph() == GetGraph(),
                    absl::InternalError) << "Inner subroutine "
        << inner_sub->GetName() << " does not point to the current graph.";
  }
  // Check that all instruction has a proper number of inner subrouines
  if (opcode_ == Opcode::kConditional) {
    RETURN_IF_FALSE(inner_subroutines_.size() >= 2,
                    absl::InternalError) << "Conditional instruction " <<
        name_ << " should have at least 2 subroutines.";
  } else if (opcode_ == Opcode::kWhile) {
    RETURN_IF_FALSE(inner_subroutines_.size() == 2,
                    absl::InternalError) << "While instruction " <<
        name_ << " should have exactly 2 subroutines.";
  } else if (opcode_ == Opcode::kCall) {
    RETURN_IF_FALSE(inner_subroutines_.size() >= 1,
                    absl::InternalError) << "Call instruction " <<
        name_ << " should have at least 1 subroutines.";
  } else if (opcode_ == Opcode::kDelay) {
    RETURN_IF_FALSE(inner_subroutines_.size() == 0,
                    absl::InternalError) << "Delay instruction " <<
        name_ << " should have no subroutines.";
  } else {
    RETURN_IF_FALSE(inner_subroutines_.size() <= 1,
                    absl::InternalError) << "Instruction " <<
        name_ <<
        " should have not more than 1 subroutine.";
  }
  return absl::OkStatus();
}

absl::Status Instruction::ValidateComposite() const {
  RETURN_IF_ERROR(ValidateCommon());
  // Check number of peers in composite graph
  for (const auto& group : comm_group_vector_) {
    // Check that Send/Recv has only two peer in each CommunicationGroup
    if ((OpcodeIsIndividualCommunication(opcode_) ||
        OpcodeIsProtocolLevelCommunication(opcode_)) &&
        (opcode_ != Opcode::kSendRecv)) {
      RETURN_IF_FALSE(group.size() == 2,
                      absl::InternalError) <<
          "Instruction " << name_ <<
          " each CommunicationGroup size should be equal to 2.";
    }
    // Check that SednRecv has exactly 3 processors in each CommunicationGroup
    if (opcode_ == Opcode::kSendRecv) {
      RETURN_IF_FALSE(group.size() == 3,
                      absl::InternalError) <<
          "Instruction " << name_ <<
          " each CommunicationGroup size should be equal to 3.";
    }
  }
  // Check that inner subroutines' execution probabilities add up to 1.0
  double inner_sub_probabilities = 0.0;
  for (const auto& inner_sub : InnerSubroutines()) {
    inner_sub_probabilities += inner_sub->GetExecutionProbability();
    RETURN_IF_ERROR(inner_sub->ValidateComposite());
  }
  // Implement subroutine probability check only for Conditional, as all other
  // subroutines (single subroutine for non-control flow instructions, single
  // subroutine for Call, or body and condition subroutines for While) are
  // always executed.
  if (opcode_ == Opcode::kConditional) {
    RETURN_IF_FALSE(fabs(inner_sub_probabilities - 1.0) < kFloatTolerance,
                    absl::InternalError) << "Instruction " << name_
        << " with opcode 'conditional' has inner subroutines with execution"
        " probabilities that don't add up to 1.0.";
  }
  return absl::OkStatus();
}

absl::Status Instruction::ValidateIndividualized() const {
  RETURN_IF_ERROR(ValidateCommon());
  if (OpcodeIsCollectiveCommunication(opcode_)) {
    RETURN_IF_FALSE(CommunicationGroupVectorHasProcessor(
        GetGraph()->GetProcessorId()), absl::InternalError)
        << "ProcessorId corresponding to the " << "instruction "
        << name_ << " is not found in CommunicationGroupVector.";
  }
  if (OpcodeIsCollectiveCommunication(opcode_) ||
      OpcodeIsIndividualCommunication(opcode_) ||
      OpcodeIsProtocolLevelCommunication(opcode_)) {
    RETURN_IF_FALSE(GetGraph()->IsIndividualized(),
                    absl::InternalError) << "Graph associated with the" <<
        " instruction " << name_ << " has default ProcessorId = " <<
        GetGraph()->GetProcessorId() << ".";
    RETURN_IF_FALSE(comm_group_vector_.size() == 1, absl::InternalError) <<
        "Instruction " << name_ << " has CommunicationGroupVector size = " <<
        comm_group_vector_.size() << ", should be 1.";
  }
  // Check that processor it in its comm_group for collectives, and is not for
  // other communcation instructions
  if (OpcodeIsCollectiveCommunication(opcode_)) {
    RETURN_IF_FALSE(CommunicationGroupVectorHasProcessor(
        GetGraph()->GetProcessorId()),
                    absl::InternalError) << "Instruction " << name_ <<
        " should have its own ID in the CommunicationGroup.";
  }
  if (OpcodeIsIndividualCommunication(opcode_) ||
      OpcodeIsProtocolLevelCommunication(opcode_)) {
    RETURN_IF_TRUE(CommunicationGroupVectorHasProcessor(
        GetGraph()->GetProcessorId()),
                   absl::InternalError) << "Instruction " << name_ <<
        " should NOT have its own ID in the CommunicationGroup.";
  }
  // Check that Send/Recv has only single peer in a single CommunicationGroup
  if ((OpcodeIsIndividualCommunication(opcode_) ||
      OpcodeIsProtocolLevelCommunication(opcode_)) &&
      (opcode_ != Opcode::kSendRecv)) {
    RETURN_IF_FALSE(GetCommunicationGroup().size() == 1,
                    absl::InternalError) <<
        "Instruction " << name_ <<
        " CommunicationGroup size should be equal to 1.";
    ASSIGN_OR_RETURN(int64_t peer, PeerId());
    RETURN_IF_FALSE(peer != GetGraph()->GetProcessorId(),
                    absl::InternalError) <<
        "Instruction " << name_ << " should not have itself as peer.";
  }
  // Check that protocol level instructions has bonded instructions set
  if (OpcodeIsProtocolLevelCommunication(opcode_)) {
    RETURN_IF_FALSE(bonded_instruction_ != nullptr,
                    absl::InternalError) <<
        "All protocol level instructions,i.e. {Send,Recv}{Start,Done}, "
        "should have bonded instructions.";
  }
  // Check that SednRecv has exactly 2 processors in the CommunicationGroup
  if (opcode_ == Opcode::kSendRecv) {
    RETURN_IF_FALSE(GetCommunicationGroup().size() == 2,
                    absl::InternalError) <<
        "Instruction " << name_ <<
        " CommunicationGroup size should be equal to 2.";
  }
  for (const auto& inner_sub : InnerSubroutines()) {
    RETURN_IF_ERROR(inner_sub->ValidateIndividualized());
  }
  return absl::OkStatus();
}

}  // namespace paragraph
