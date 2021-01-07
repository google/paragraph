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
#ifndef PARAGRAPH_SCHEDULING_GRAPH_SCHEDULER_H_
#define PARAGRAPH_SCHEDULING_GRAPH_SCHEDULER_H_

#include <memory>
#include <queue>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "paragraph/graph/graph.h"
#include "paragraph/scheduling/instruction_fsm.h"
#include "paragraph/scheduling/subroutine_fsm.h"
#include "paragraph/shim/macros.h"
#include "paragraph/shim/statusor.h"

namespace paragraph {

class GraphScheduler {
  friend class InstructionFsm;
  friend class SubroutineFsm;

 public:
  ~GraphScheduler() = default;

  // Creates a new scheduler and verifies the that graph is ready to be
  // scheduled
  // Part of Public API with simulators
  static shim::StatusOr<std::unique_ptr<GraphScheduler>> Create(
      Graph* graph);

  // Provides all instructions ready for scheduling to Simulator
  // Part of Public API with simulators
  std::vector<Instruction*> GetReadyInstructions();
  // NOLINTNEXTLINE(runtime/references)
  void GetReadyInstructions(std::queue<Instruction*>& queue);

  // Marks instruction as Finished in Simulator
  // Part of Public API with simulators
  void InstructionFinished(Instruction* instruction);

  // Seeds internal PRNG
  // The scheduler makes decisions about the order in which some subroutines are
  // called using a PRNG. Setting the seed of the PRNG guarantees a
  // deterministic execution.
  void SeedRandom(uint64_t seed);

  // Interface that returns instruction/subroutine FSM by its pointer
  InstructionFsm& GetFsm(const Instruction* instruction);
  SubroutineFsm& GetFsm(const Subroutine* subroutine);

 private:
  // Private constructor to ensure that graph is valid before it gets scheduled,
  // and also to link scheduler with instructions/subroutines FSMs
  explicit GraphScheduler(Graph* graph);

  // Graph to be scheduled
  Graph* graph_;

  // Scheduler instruction queue
  std::vector<Instruction*> ready_to_schedule_;

  // Interface to get instructions be added to the scheduler queue
  void EnqueueToScheduler(Instruction* instruction);

  // Maps to lookup instruction/subroutine FSM by corresponding pointer
  absl::flat_hash_map<const Instruction*, InstructionFsm>
      instruction_state_map_;
  absl::flat_hash_map<const Subroutine*, SubroutineFsm>
      subroutine_state_map_;

  // Random number engine based on Mersenne Twister algorithm
  std::mt19937_64 prng_;
};

}  // namespace paragraph

#endif  // PARAGRAPH_SCHEDULING_GRAPH_SCHEDULER_H_
