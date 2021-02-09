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
#include "paragraph/scheduling/subroutine_fsm.h"

#include "paragraph/scheduling/graph_scheduler.h"

namespace paragraph {

std::string SubroutineFsm::SubroutineStateToString(
    SubroutineFsm::State state) {
  switch (state) {
#define CASE_STATE_STRING(enum_name, state_name)     \
  case SubroutineFsm::State::enum_name:              \
    return state_name;
    SUBROUTINE_STATE_LIST(CASE_STATE_STRING)
#undef CASE_STATE_STRING
  default:
    return "UNKNOWN";
  }
}

shim::StatusOr<SubroutineFsm::State> SubroutineFsm::StringToSubroutineState(
    const std::string& state_name) {
  static auto* state_map = new absl::flat_hash_map<
      std::string, SubroutineFsm::State>({
#define STRING_TO_STATE_ENTRY(enum_name, state_name) \
  {state_name, SubroutineFsm::State::enum_name},
      SUBROUTINE_STATE_LIST(STRING_TO_STATE_ENTRY)
#undef STRING_TO_STATE_ENTRY
  });
  auto it = state_map->find(state_name);
  if (it == state_map->end()) {
    return absl::InvalidArgumentError(
        "Unknown subroutine FSM state " + state_name +
        " in StringToSubroutineState conversion.");
  }
  return it->second;
}

SubroutineFsm::SubroutineFsm(GraphScheduler* scheduler,
                             const Subroutine* subroutine)
    : subroutine_(subroutine),
      scheduler_(scheduler) {}

bool SubroutineFsm::IsBlocked() const {
  return state_ == State::kBlocked;
}

void SubroutineFsm::SetBlocked() { state_ = State::kBlocked; }

bool SubroutineFsm::IsScheduled() const {
  return state_ == State::kScheduled;
}

void SubroutineFsm::SetScheduled() { state_ = State::kScheduled; }

bool SubroutineFsm::IsFinished() const {
  return state_ == State::kFinished;
}

void SubroutineFsm::SetFinished() { state_ = State::kFinished; }

int64_t SubroutineFsm::GetExecutionCount() const {
  return current_execution_count_;
}

void SubroutineFsm::DecrementExecutionCount() {
  current_execution_count_--;
}

void SubroutineFsm::Reset(bool reset_exec_count) {
  if (reset_exec_count) {
    current_execution_count_ = subroutine_->GetExecutionCount();
  }
  state_ = State::kBlocked;
  for (auto& instruction : subroutine_->Instructions()) {
    scheduler_->GetFsm(instruction.get()).Reset();
  }
}

absl::Status SubroutineFsm::InstructionFinished(
    const Instruction* instruction) {
  RETURN_IF_FALSE(instructions_to_execute_.erase(instruction) == 1,
                  absl::InternalError) << "Could not find instruction "
      << instruction->GetName() << " in subroutine " << subroutine_->GetName()
      << " scheduler.";
  if (instructions_to_execute_.empty()) {
    current_execution_count_--;
    SetFinished();
    RETURN_IF_FALSE(subroutine_->GetCallingInstruction() != nullptr,
                    absl::InternalError) << "Subroutine " <<
        subroutine_->GetName() <<
        " has registered calling instruction missing.";
    // Entry subroutine needs special treatment because it does not have a real
    // calling instruction
    if (subroutine_ != subroutine_->GetGraph()->GetEntrySubroutine()) {
      RETURN_IF_ERROR(scheduler_->GetFsm(
          subroutine_->GetCallingInstruction()).PrepareToSchedule());
    }
  }
  return absl::OkStatus();
}

// Subroutine scheduling handling
absl::Status SubroutineFsm::PrepareToSchedule() {
  SetScheduled();
  for (const auto& instr : subroutine_->Instructions()) {
    RETURN_IF_FALSE(instructions_to_execute_.emplace(instr.get()).second,
                    absl::InternalError) << "Could not insert instruction "
        << instr->GetName() << " into subroutine " << subroutine_->GetName()
        << " scheduler because it was already there.";
  }
  for (const auto& instruction : subroutine_->Instructions()) {
    if (scheduler_->GetFsm(instruction.get()).IsReady()) {
      RETURN_IF_ERROR(scheduler_->GetFsm(
          instruction.get()).PrepareToSchedule());
    }
  }
  return absl::OkStatus();
}

}  // namespace paragraph
