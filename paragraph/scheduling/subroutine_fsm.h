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
#ifndef PARAGRAPH_SCHEDULING_SUBROUTINE_FSM_H_
#define PARAGRAPH_SCHEDULING_SUBROUTINE_FSM_H_

#include <string>

#include "absl/container/flat_hash_set.h"
#include "paragraph/graph/subroutine.h"
#include "paragraph/scheduling/instruction_fsm.h"
#include "paragraph/shim/macros.h"
#include "paragraph/shim/statusor.h"

namespace paragraph {
class GraphScheduler;

#define SUBROUTINE_STATE_LIST(V)                                       \
  V(kBlocked, "blocked")                                               \
  V(kScheduled, "scheduled")                                           \
  V(kFinished, "finished")

class SubroutineFsm {
  friend class GraphScheduler;
  friend class InstructionFsm;

 public:
  enum class State {
#define DECLARE_ENUM(enum_name, state_name) enum_name,
    SUBROUTINE_STATE_LIST(DECLARE_ENUM)
#undef DECLARE_ENUM
  };

  SubroutineFsm(GraphScheduler* scheduler,
                const Subroutine* subroutine);
  ~SubroutineFsm() = default;

  static std::string SubroutineStateToString(SubroutineFsm::State state);

  static shim::StatusOr<SubroutineFsm::State> StringToSubroutineState(
      const std::string& state_name);

  // Getters and setters for subroutine scheduler state
  bool IsBlocked();
  void SetBlocked();
  bool IsScheduled();
  void SetScheduled();
  bool IsFinished();
  void SetFinished();

  // Getter/setters for subroutine execution count
  int64_t GetExecutionCount() const;
  void DecrementExecutionCount();

  // Resets subroutine state
  void Reset(bool reset_exec_count = true);

  // Prepares to schedule subroutine
  absl::Status PrepareToSchedule();

  // Updates the state of the subroutine if one of the instructions is finished
  absl::Status InstructionFinished(const Instruction* instruction);

 private:
  // State of the subrotine
  State state_;

  // Subrotine associated with this state
  const Subroutine* subroutine_;

  // Pointer to the graph scheduler that keeps scheduling information about
  // all subroutines and instructions in the graph
  GraphScheduler* scheduler_;

  // Current execution count (how many times subroutines is left to be
  // finished) and how many instructios are left to be finished in subroutine
  // Describes the state of subroutine "in flight"
  int64_t current_execution_count_;
  absl::flat_hash_set<const Instruction*> instructions_to_execute_;
};

}  // namespace paragraph

#endif  // PARAGRAPH_SCHEDULING_SUBROUTINE_FSM_H_
