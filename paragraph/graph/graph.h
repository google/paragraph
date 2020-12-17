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
#ifndef PARAGRAPH_GRAPH_GRAPH_H_
#define PARAGRAPH_GRAPH_GRAPH_H_

#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "paragraph/graph/subroutine.h"

namespace paragraph {

// Communication group keeps correspondence between ProcessorIds of other
// processors with which this processor is communicating, and pointers to
// CommunicationGroup that contain the other processor ProcessorId
using CommunicationSet = absl::flat_hash_set<int64_t>;

// Graph contains a single Paragraph hierarchical graph.
// Graph consists of subroutines, which consist of instructions,
// which could call other subroutines, hence creating a hierarchical structure.
// Paragraph::Graph may describe a single distributed application running
// on multiple processor, in which case it mainly contains high order
// communication primitives, such as AllReduce, AllGather, AlltoAll. At the same
// time, Paragraph::Graph may represent a graph lowered to a single processor,
// in which case it should contain only primitive Send/Recv communication
// instructions. Graph contains unique_ptr to each instruction or
// subroutine, so it cannot share instructions or subroutine with other graph
class Graph {
  friend class Instruction;

 public:
  explicit Graph(const std::string& name, int64_t processor_id = -1);
  virtual ~Graph() {}

  const std::string& GetName() const { return name_; }
  void SetName(std::string name) { name_ = std::move(name); }

  int64_t GetProcessorId() const { return processor_id_; }
  void SetProcessorId(int64_t processor_id) { processor_id_ = processor_id; }

  // Checks whether the graph is individualized, meaning has instructions as
  // they will be executed on the processor with ptocessor_id
  bool IsIndividualized() const { return processor_id_ >= 0; }

  void SetInstructionId(Instruction* instr_ptr);

  // Return a pointer to the entry subroutine of the graph.
  Subroutine* GetEntrySubroutine() const;
  bool HasEntrySubroutine() const {
    return entry_subroutine_ != nullptr;
  }

  // Sets the entry subroutine of the graph.
  void SetEntrySubroutine(std::unique_ptr<Subroutine> subroutine);

  // Gets the subroutines in this graph.
  const std::vector<Subroutine*> Subroutines() const;

  // Gets the instructions in the graph in post order from root instruction of
  // the entry subroutine.
  const std::vector<Instruction*> InstructionsPostOrder() const;

  // Applies communication tags to the {Recv,Send}{Start,Done} instructions.
  void ApplyCommunicationTags();

  // Creates additional dependencies in the graph to enforce postorder during
  // the scheduling
  void PostOrderEnforcer();

  // Gets the number of subroutines in this graph.
  int64_t SubroutineCount() const;

  // Gets the number of instructions in this graph.
  int64_t InstructionCount() const;

  // Iterator over all the communication peers (ProcessorIds) in the graph
  const CommunicationSet& GetCommunicationSet() const { return comm_set_; }

  // Returns a mapping between instruction ID and instruction pointer.
  // Results are only valid until the graph is modified.
  absl::flat_hash_map<int64_t, const Instruction*> InstructionIdMap() const;

  // Checks that processor ids in the graph start with 0 and are consecutive
  bool HasConsecutiveNaturalProcessorIds() const;

  // Returns the instructions with a matching name.
  std::vector<const Instruction*> FindByName(const std::string& name) const;

  bool CommunicationSetHasProcessor(int64_t processor_id) const;

  // Convert an Graph to or from a proto.
  shim::StatusOr<GraphProto> ToProto() const;
  static shim::StatusOr<std::unique_ptr<Graph>> CreateFromProto(
      const GraphProto& proto, bool reset_ids = false);

  // Graph IO interface
  // Filenames that end in .pb are treated as binary proto files.
  // Filenames that end in .textproto are treated as text proto files.
  static constexpr absl::string_view kBinaryProtoExtension = ".pb";
  static constexpr absl::string_view kTextProtoExtension = ".textproto";
  static shim::StatusOr<std::unique_ptr<Graph>> ReadFromFile(
      const std::string& filename);
  absl::Status WriteToFile(const std::string& filename) const;

  // Creates a clone of this graph.
  // 'reset_ids' allows the IDs within the graph to be reset.
  shim::StatusOr<std::unique_ptr<Graph>> Clone(
      const std::string& name_suffix, bool reset_ids = false);

  // Creates an individualized graph from a composite graph. New graph
  // corresponds to the graph executed on the processor processor_id
  shim::StatusOr<std::unique_ptr<Graph>>
      Individualize(int64_t processor_id) const;

  // Checks that all implicit requirements for a valid graph are met
  absl::Status ValidateComposite() const;
  absl::Status ValidateIndividualized() const;

 private:
  std::string name_;
  int64_t processor_id_;
  int64_t next_unique_id_;

  // Entry subroutine for the whole graph. ID is set to 0 when assigned, and
  // its ID is changed to RootID when root is added.
  std::unique_ptr<Subroutine> entry_subroutine_;
  // Fake entry instruction that is set to be calling instruction for the
  // entry subroutine so it has all the same properties as other subroutines
  Instruction implicit_entry_instruction_;

  // Set that contains all the ProcessorIds of the processors that participate
  // in communication in this graph. For not lowered graph it corresponds to
  // all the processors graph is running on.
  // Needs for fast iterating over the communication peer processors during
  // lowering
  CommunicationSet comm_set_;

  // Validates common properties of Composite and Individualized graphs
  absl::Status ValidateCommon() const;
};

}  // namespace paragraph

#endif  // PARAGRAPH_GRAPH_GRAPH_H_
