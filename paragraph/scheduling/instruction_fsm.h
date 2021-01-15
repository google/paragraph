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
#ifndef PARAGRAPH_SCHEDULING_INSTRUCTION_FSM_H_
#define PARAGRAPH_SCHEDULING_INSTRUCTION_FSM_H_

#include <string>

#include "paragraph/graph/instruction.h"
#include "paragraph/shim/macros.h"
#include "paragraph/shim/statusor.h"

namespace paragraph {
class GraphScheduler;
class SubroutineStateFsm;

#define INSTRUCTION_STATE_LIST(V)                                      \
  V(kBlocked, "blocked")                                               \
  V(kReady, "ready")                                                   \
  V(kScheduled, "scheduled")                                           \
  V(kExecuting, "executing")                                           \
  V(kFinished, "finished")

class InstructionFsm {
  friend class GraphScheduler;
  friend class SubroutineFsm;

 public:
  enum class State {
#define DECLARE_ENUM(enum_name, state_name) enum_name,
    INSTRUCTION_STATE_LIST(DECLARE_ENUM)
#undef DECLARE_ENUM
  };

  InstructionFsm(GraphScheduler* scheduler,
                 Instruction* instruction);
  ~InstructionFsm() = default;

  static std::string InstructionStateToString(InstructionFsm::State state);

  static shim::StatusOr<InstructionFsm::State> StringToInstructionState(
      const std::string& state_name);

  // Getters and setters for instruction scheduler state
  bool IsBlocked();
  void SetBlocked();
  bool IsReady();
  void SetReady();
  bool IsScheduled();
  void SetScheduled();
  bool IsExecuting();
  void SetExecuting();
  bool IsFinished();
  void SetFinished();

  // Resets instruction state setting it to Blocked or Ready depending on
  // its operands and their state
  void Reset();

  // Checks whether instruction is blocked by its operands
  bool IsUnblockedByOperands();

  // Prepares to schedule either the instruction or its inner subroutine
  absl::Status PrepareToSchedule();

  // Picks the next subroutine to schedule among instruction's inner subroutines
  shim::StatusOr<Subroutine*> PickSubroutine();

  // Getters/Setters for instruction timings
  double GetTimeReady();
  void SetTimeReady(double current_time);
  double GetTimeStarted();
  void SetTimeStarted(double current_time);
  double GetTimeFinished();
  void SetTimeFinished(double current_time);

 private:
  // State of the instruction
  State state_;

  // Instruction associated with this state
  Instruction* instruction_;

  // Pointer to the graph scheduler that keeps scheduling information about
  // all subroutines and instructions in the graph
  GraphScheduler* scheduler_;

  // Instruction timings
  double time_ready_;
  double time_started_;
  double time_finished_;
};

}  // namespace paragraph

#endif  // PARAGRAPH_SCHEDULING_INSTRUCTION_FSM_H_
