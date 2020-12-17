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
#ifndef PARAGRAPH_GRAPH_SUBROUTINE_H_
#define PARAGRAPH_GRAPH_SUBROUTINE_H_

#include <list>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

// When flat_hash_map works again, we can drop unordered_map
// #include "absl/container/flat_hash_map.h"
#include "paragraph/graph/instruction.h"

namespace paragraph {

class Graph;
using InstructionList = std::list<std::unique_ptr<Instruction>>;

class Subroutine {
  friend class Graph;
  friend class Instruction;

 public:
  Subroutine(const std::string& name, Graph* graph);

  virtual ~Subroutine() = default;

  // Default id for the subroutine
  static constexpr int64_t kDefaultId = -1;

  // Gets the string identifier for this subroutine.
  const std::string& GetName() const { return name_; }

  // Clear the unique ID of the subroutine so that it can be re-assigned, such
  // as for the purpose of compacting the subroutine unique IDs.
  void ClearId() { id_ = kDefaultId; }

  // Check if id is Default (non-unique), meaning equal to fDefaultId = -1
  bool HasDefaultId() const { return id_ == kDefaultId; }

  // Return the unique ID assigned to this processor via SetUniqueId (or -1
  // if no id has been assigned yet).
  int GetId() const { return id_; }

  // Set the unique id for this subroutine to "id"
  void SetId(int id) {
    CHECK_GE(id, 0);
    id_ = id;
  }

  // Getter/setter for subroutine execution probability
  double GetExecutionProbability() const {
    return execution_probability_;
  }
  void SetExecutionProbability(double prob) {
    execution_probability_ = prob;
  }

  // Getter/setters for subroutine execution count
  int64_t GetExecutionCount() const {
    return execution_count_;
  }
  void SetExecutionCount(int64_t count) {
    execution_count_ = count;
  }

  // Gets/Sets the root instruction of the subroutine.
  const Instruction* GetRootInstruction() const { return root_instruction_; }
  absl::Status SetRootInstruction(Instruction* instr);

  // Return the calling instruction of the subroutine. The calling instruction
  // is the instruction that calls the subroutine.
  const Instruction* GetCallingInstruction() const {
    return calling_instruction_;
  }
  void SetCallingInstruction(Instruction* instr) {
    calling_instruction_ = instr;
  }

  // Gets the instructions in this subroutine.
  const std::list<std::unique_ptr<Instruction>>&
      Instructions() const { return instructions_; }
  std::list<std::unique_ptr<Instruction>>&
      Instructions() { return instructions_; }

  // Get the iterators for the instructions in this subroutine
  const InstructionList::iterator
      InstructionIterator(Instruction* instruction) const {
    return instruction_iterators_.at(instruction);
  }
  InstructionList::iterator
      InstructionIterator(Instruction* instruction) {
    return instruction_iterators_.at(instruction);
  }

  // Recursively returns all the embedded subroutines - inner subroutines of
  // instructions in the current subroutine - via subroutines set
  // Subroutines are returned in PostOrder, meaning that first the inner-most
  // subroutines are returned in the order they appear in InnerSubroutines()
  // method of calling instruction
  // Useful for Graph::Subroutines() implementation
  std::vector<Subroutine*> MakeEmbeddedSubroutinesVector() const;

  // Gets the instructions in this subroutine in post order from root.
  std::vector<Instruction*> InstructionsPostOrder(
      bool skip_inner_subroutines = false);

  // Removes an instruction from the subroutine instructions list
  void RemoveInstruction(Instruction* instruction);

  // Replaces target instruction in a subroutine with and vector of new
  // instructions. Instructions in the vector are suposed to be wired with each
  // other, the first instruction in the vector will inherit operands from
  // target instruction, the last instruction inherits target's users and root
  // status if target is the root of subroutine
  // WARNING: if !OkStatus() returned, the graph might be corrupted with some
  // users deleted from the target instruction. Fail in this function should
  // cause calling abort() and terminating execution
  absl::Status ReplaceInstructionWithInstructionList(
      Instruction* target_instr_ptr,
      InstructionList* instructions);

  // Returns the number of instruction in the subroutine
  int64_t InstructionCount() const { return instruction_iterators_.size(); }

  // Recursively scales performance cost of subroutine, including
  // all instructions in it and their inner subroutines.
  void ScalePerformance(double scale);

  // Set/get the graph containing this computation.
  void SetGraph(Graph* graph) { graph_ = graph; }
  const Graph* GetGraph() const { return graph_; }
  Graph* GetGraph() { return graph_; }

  // Clones subroutine with its subroutines recursively
  shim::StatusOr<std::unique_ptr<Subroutine>> Clone(
      const std::string& name_suffix, bool reset_ids = false);

  // Returns a serialized representation of this computation.
  shim::StatusOr<SubroutineProto> ToProto() const;

  // Creates a subroutine from the given proto. Arguments:
  // proto: the proto to convert from.
  // graph: pointer to the graph that contains instruction
  static shim::StatusOr<std::unique_ptr<Subroutine>>
      CreateFromProto(const SubroutineProto& proto, Graph* graph,
                      bool reset_ids = false);

  // Checks that all implicit requirements for a valid subroutine are met
  absl::Status ValidateComposite() const;
  absl::Status ValidateIndividualized() const;

 private:
  // Adds a single instruction to the subroutine instructions list
  absl::Status AddInstruction(std::unique_ptr<Instruction> instruction);

  // String identifier for subroutine.
  std::string name_;

  // Subroutine ID that is equal to its root instruction ID
  // Root ID is unique among instructions within the parent graph
  int id_;

  // Execution count and probability for current subroutine
  double execution_probability_;
  int64_t execution_count_;

  // The root instruction is the one that produces the output of the subroutine.
  Instruction* root_instruction_;
  Instruction* calling_instruction_;

  // Store instructions in std::list as they can be added and removed
  // arbitrarily and we want a stable iteration order. Keep a map from
  // instruction pointer to location in the list for fast lookup.
  InstructionList instructions_;
  // TODO(misaev) flat_hash_map returns error, but should work fine,
  // Test with next absl release and uncomment if fixed
  // absl::flat_hash_map<const Instruction*, InstructionList::iterator>
  std::unordered_map<const Instruction*, InstructionList::iterator>
      instruction_iterators_;

  // Gets the instructions in this subroutine in post order from root.
  void PostOrderHelper(absl::flat_hash_map<Instruction*, bool>* visited,
                       std::vector<Instruction*>* postorder,
                       bool skip_inner_subroutines = false);

  // Module containing this computation.
  Graph* graph_;

  // Recursively returns all the embedded subroutines - inner subroutines of
  // instructions in the current subroutine - via subroutines set
  // Subroutines are returned in PostOrder, meaning that first the inner-most
  // subroutines are returned in the order they appear in InnerSubroutines()
  // method of calling instruction
  // Useful for Graph::Subroutines() implementation
  void EmbeddedSubroutines(std::vector<Subroutine*>* subroutines) const;

  // Validates common properties of Composite and Individualized graphs
  absl::Status ValidateCommon() const;
};

}  // namespace paragraph

#endif  // PARAGRAPH_GRAPH_SUBROUTINE_H_
