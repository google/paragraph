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
#ifndef PARAGRAPH_GRAPH_INSTRUCTION_H_
#define PARAGRAPH_GRAPH_INSTRUCTION_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "paragraph/graph/graph.pb.h"
#include "paragraph/graph/opcode.h"
#include "paragraph/shim/macros.h"
#include "paragraph/shim/statusor.h"

namespace paragraph {

class Graph;
class Subroutine;

using CommunicationGroup = std::vector<int64_t>;

// ProcessorCoordinate helps to extract position of processor_id in the
// comm_group_vector_ private member of Instruction class.
// First field group_vector_offset stores the offset of a CommunicationGroup
// that contains processor_id in the comm_group_vector_ vector of
// CommunicationGroup.
// Second field comm_group_offset stores the offset of processor_id in
// CommunicationGroup.
struct ProcessorCoordinates {
  int64_t group;
  int64_t offset;
};

class Instruction {
  friend class Graph;
  friend class Subroutine;

 public:
  virtual ~Instruction() = default;

  static shim::StatusOr<Instruction*> Create(
      Opcode opcode,
      const std::string& name,
      Subroutine* parent,
      bool is_root = false);

  // Default id for the instruction
  // As each valid instruction must have dedicated id, this is a
  // default invalid id
  static constexpr int64_t kDefaultId = -1;

  // Tolerance for float to int comparison
  static constexpr double kFloatTolerance = 1e-6;

  // Gets/sets the string identifier for this instruction.
  const std::string& GetName() const { return name_; }
  void SetName(const std::string& name) { name_ = name; }

  // Clear the unique ID of the instruction so that it can be re-assigned, such
  // as for the purpose of compacting the instruction unique IDs.
  void ClearId() { id_ = kDefaultId; }

  // Check if id is Default (non-unique), meaning equal to fDefaultId = -1
  bool HasDefaultId() const { return id_ == kDefaultId; }

  // Return the unique ID assigned to this processor via SetUniqueId (or -1
  // if no id has been assigned yet).
  int64_t GetId() const { return id_; }

  // Set the unique id for this instruction to "id"
  void SetId(int64_t id);

  // Returns the opcode for this instruction.
  const Opcode& GetOpcode() const { return opcode_; }

  // Returns the number of operands to this instruction.
  int64_t OperandCount() const { return operands_.size(); }

  // Ensures the specified instruction is listed as an operand to this
  // instruction.
  bool AddOperand(Instruction* operand);

  // Returns the vector of operands of this instruction.
  const std::vector<Instruction*>& Operands() const { return operands_; }

  // Returns the users of this instruction.
  const std::vector<Instruction*>& Users() const { return users_; }

  // Returns the number of users of this instruction.
  int64_t UserCount() const { return users_.size(); }

  // Adds/Removes user to/from instruction
  bool AddUser(Instruction* user);
  absl::Status RemoveUser(Instruction* user);

  // Returns the index of the user in the users() vector.
  // Precondition: `user` is a user of the instruction.
  int64_t UserId(Instruction* user) const;

  // Returns true if this instruction is a user of 'instruction'.
  bool IsUserOf(const Instruction* instruction) const {
    return instruction->user_map_.find(this) !=
      instruction->user_map_.end();
  }

  // Returns instruction that's bonded with this one
  Instruction* GetBondedInstruction();
  const Instruction* GetBondedInstruction() const;

  // Bonds current instructions with another one
  void BondWith(Instruction* instruction);

  // Set/get the subroutine containing this instruction. SetParent should only
  // be called by Subroutine methods that add/remove instructions to subroutine.
  void SetParent(Subroutine* subroutine) { parent_ = subroutine; }
  const Subroutine* GetParent() const { return parent_; }
  Subroutine* GetParent() { return parent_; }

  // Gets the inner subroutines in this instruction.
  const std::vector<std::unique_ptr<Subroutine>>&
      InnerSubroutines() const { return inner_subroutines_; }

  // Appends a subroutine to inner subroutines
  void AppendInnerSubroutine(std::unique_ptr<Subroutine> subroutine);

  // Removes inner subroutine with all its instruction from the instruction
  // (inner_subroutines_) vector and releases its unique_ptr
  void RemoveInnerSubroutine(Subroutine* subroutine);

  // Replaces inner subroutine  with a new subroutine
  absl::Status ReplaceInnerSubroutine(
    Subroutine* target_subroutine, std::unique_ptr<Subroutine> new_subroutine);

  // Sets/gets Ops (operations) that instruction takes to complete
  // Theoretically, may be used to descripe FLOPs, Arithmetic Ops
  // in case of accelerator uses INT8, FixedPoint, or similar math format
  void SetOps(double ops) { ops_ = ops; }
  double GetOps() const { return ops_; }

  // Sets/Gets number of transcendental ops instruction takes to complete.
  // Transcendentals (sin/cos, exp/log, sigmoid, etc.) are often takes more time
  // than regular FLOP, and may have different arithmetic throughput, or may be
  // implemented using separate LUT mechanism
  void SetTranscendentals(double trans) { transcendentals_ = trans; }
  double GetTranscendentals() const { return transcendentals_; }

  // Sets/gets amount of data accessed by instruction to Read/Write
  void SetBytesIn(double bytes) { bytes_in_ = bytes; }
  double GetBytesIn() const { return bytes_in_; }
  void SetBytesOut(double bytes) { bytes_out_ = bytes; }
  double GetBytesOut() const { return bytes_out_; }

  // Set/gets estimated time for instruction to complete
  void SetSeconds(double seconds) { seconds_ = seconds; }
  double GetSeconds() const { return seconds_; }

  // Scales performance cost of instruction, including all subroutines.
  void ScalePerformance(double scale);

  // Returns the graph for this instruction.
  const Graph* GetGraph() const;
  Graph* GetGraph();

  // Returns the vector of all CommunicationGroups that instruction has. In
  // individualized graph should return a vector of size 1
  const std::vector<CommunicationGroup>& GetCommunicationGroupVector() const {
    return comm_group_vector_;
  }
  // Properly clears the vector of CommunicationGroups and all the information
  // about the processors associated with CommunicationGroups that instruction
  // keeps
  void ClearCommunicationGroupVector();

  // Gets a single CommunicationGroup if instruction only has one - is useful
  // for individualized graph
  const CommunicationGroup& GetCommunicationGroup() const;

  // Addds a new CommunicationGroup to a vector of CommunicationGroups
  void AppendCommunicationGroup(const CommunicationGroup& group);

  // Returns peer processor_id for Individualized Send/Recv instructions
  shim::StatusOr<int64_t> PeerId() const;
  // Returns processor coordinates in the CommunicationGroup vector for a
  // processor_id
  shim::StatusOr<ProcessorCoordinates> GetProcessorCoordinates(
      int64_t processor_id) const;
  int64_t GetProcessorIndex(int64_t processor_id) const;

  // Checks if processor with processor_id exists in the vector of
  // CommunicationGroups associated with this instruction
  bool CommunicationGroupVectorHasProcessor(int64_t processor_id) const;

  // Getter/setter for communication tag
  uint64_t GetCommunicationTag() const { return communication_tag_; }
  void SetCommunicationTag(uint64_t tag) { communication_tag_ = tag; }

  // Returns a serialized representation of this instruction.
  shim::StatusOr<InstructionProto> ToProto() const;

  // Creates an instruction from the given proto. Arguments:
  // proto: the proto to convert from.
  // parent: pointer to the parent subroutine that contains instruction
  static shim::StatusOr<std::unique_ptr<Instruction>>
      CreateFromProto(const InstructionProto& proto, Subroutine* parent,
                      bool reset_ids = false);

  // Clones instruction with its subroutines recursively
  shim::StatusOr<std::unique_ptr<Instruction>> Clone(
      const std::string& name_suffix, bool reset_ids = false);

  // Checks that all implicit requirements for a valid instruction are met
  absl::Status ValidateComposite() const;
  absl::Status ValidateIndividualized() const;

 private:
  // Private constructor to ensure that every instruction is created linked to
  // it parent subroutine and graph via Instruction::Create() call
  Instruction(Opcode opcode, const std::string& name, Subroutine* parent);

  // String identifier for instruction.
  std::string name_;

  // Instruction ID Unique to this Instruction within a Graph
  int64_t id_;

  // Opcode for this instruction.
  Opcode opcode_;

  // Performance related parameters
  double ops_;
  double transcendentals_;
  double bytes_in_;
  double bytes_out_;
  double seconds_;

  // Instruction operands.
  std::vector<Instruction*> operands_;

  // Instruction that's bonded to share some parameters, i.e.
  // SendStart and SendRDone to share sequence number
  Instruction* bonded_instruction_;

  // The users of this instruction. Users are other instructions where this
  // instruction is an operand. The vector users_ and the map user_map_
  // contain identical members. The map enables fast membership testing and
  // the vector enables fast, stable iteration. The value in the map contains
  // the index of the instruction in the vector what enables fast removal.
  std::vector<Instruction*> users_;
  absl::flat_hash_map<const Instruction*, int64_t> user_map_;

  // The Subroutine in which this instruction is contained.
  Subroutine* parent_;

  // Inner subroutines called by this instruction.
  std::vector<std::unique_ptr<Subroutine>> inner_subroutines_;

  // Gets the instructions in this subroutine in post order from root.
  // If skip_inner_subroutines set to true, return post order fo instructions in
  // this subroutine.
  // If skip_inner_subroutines set to false (default), returns
  // post order for this subroutine and all recurcive inner subroutines.
  void PostOrderHelper(absl::flat_hash_map<Instruction*, bool>* visited,
                       std::vector<Instruction*>* postorder,
                       bool skip_inner_subroutines = false);

  // Vector that encodes all communication directions for the instruction
  // For Send/Recv should have size one since it's a direct communication
  std::vector<CommunicationGroup> comm_group_vector_;
  // Keeps mapping between ProcessorId and its index position in
  // comm_group_vector_
  absl::flat_hash_map<int64_t, ProcessorCoordinates> processor_id_to_index_map_;

  // communication_tag is used to match Send instruction to Recv instruction.
  // To be matched in the different graphs, both instructions should have
  // matching id which should be unique per source-destination pair.
  // Only makes sense for SendStart, SendDone, RecvStart, RecvDone instructions
  uint64_t communication_tag_;

  // Validates common properties of Composite and Individualized graphs
  absl::Status ValidateCommon() const;
};

}  // namespace paragraph

#endif  // PARAGRAPH_GRAPH_INSTRUCTION_H_
