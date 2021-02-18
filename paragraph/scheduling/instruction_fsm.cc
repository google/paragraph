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
#include "paragraph/scheduling/instruction_fsm.h"

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "paragraph/scheduling/graph_scheduler.h"
#include "paragraph/shim/macros.h"

namespace paragraph {

std::string InstructionFsm::InstructionStateToString(
    InstructionFsm::State state) {
  switch (state) {
#define CASE_STATE_STRING(enum_name, state_name)     \
    case InstructionFsm::State::enum_name:           \
    return state_name;
    INSTRUCTION_STATE_LIST(CASE_STATE_STRING)
#undef CASE_STATE_STRING
  default:
    return "UNKNOWN";
  }
}

shim::StatusOr<InstructionFsm::State> InstructionFsm::StringToInstructionState(
    const std::string& state_name) {
  static auto* state_map = new absl::flat_hash_map<
      std::string, InstructionFsm::State>({
#define STRING_TO_STATE_ENTRY(enum_name, state_name) \
  {state_name, InstructionFsm::State::enum_name},
      INSTRUCTION_STATE_LIST(STRING_TO_STATE_ENTRY)
#undef STRING_TO_STATE_ENTRY
  });
  auto it = state_map->find(state_name);
  if (it == state_map->end()) {
    return absl::InvalidArgumentError(
        "Unknown instruction FSM state " + state_name +
        " in StringToInstructionState conversion.");
  }
  return it->second;
}

InstructionFsm::InstructionFsm(GraphScheduler* scheduler,
                               Instruction* instruction)
    : instruction_(instruction),
      scheduler_(scheduler),
      time_ready_(0.0),
      time_started_(0.0),
      time_finished_(0.0) {}

bool InstructionFsm::IsUnblockedByOperands() const {
  bool unblocked = true;
  for (auto& operand : instruction_->Operands()) {
    unblocked &= scheduler_->GetFsm(operand).IsFinished();
  }
  return unblocked;
}

bool InstructionFsm::IsBlocked() const {
  return state_ == State::kBlocked;
}

void InstructionFsm::SetBlocked() { state_ = State::kBlocked; }

bool InstructionFsm::IsReady() const { return state_ == State::kReady; }

void InstructionFsm::SetReady() { state_ = State::kReady; }

bool InstructionFsm::IsScheduled() const {
  return state_ == State::kScheduled;
}

void InstructionFsm::SetScheduled() { state_ = State::kScheduled; }

bool InstructionFsm::IsExecuting() const {
  return state_ == State::kExecuting;
}

void InstructionFsm::SetExecuting() { state_ = State::kExecuting; }

bool InstructionFsm::IsFinished() const {
  return state_ == State::kFinished;
}

void InstructionFsm::SetFinished() { state_ = State::kFinished; }

double InstructionFsm::GetTimeReady() const {
  return time_ready_;
}

void InstructionFsm::SetTimeReady(double current_time) {
  time_ready_ = current_time;
}

double InstructionFsm::GetTimeStarted() const {
  return time_started_;
}

void InstructionFsm::SetTimeStarted(double current_time) {
  time_started_ = current_time;
}

double InstructionFsm::GetTimeFinished() const {
  return time_finished_;
}

void InstructionFsm::SetTimeFinished(double current_time) {
  time_finished_ = current_time;
}

void InstructionFsm::Reset() {
  if (instruction_->Operands().empty()) {
    state_ = State::kReady;
  } else {
    state_ = State::kBlocked;
  }
  for (auto& subroutine : instruction_->InnerSubroutines()) {
    scheduler_->GetFsm(subroutine.get()).Reset();
  }
}

const Instruction* InstructionFsm::GetInstruction() const {
  return instruction_;
}

absl::Status InstructionFsm::PrepareToSchedule() {
  // If instruction has inner subroutines, we don't schedule it directly,
  // instead we schedule inner subroutines and its instructionss
  if (!instruction_->InnerSubroutines().empty()) {
    SetScheduled();
    ASSIGN_OR_RETURN(Subroutine* subroutine, PickSubroutine());
    // Check if subroutine is not finished yet
    if (scheduler_->GetFsm(subroutine).GetExecutionCount() > 0 &&
        !scheduler_->GetFsm(subroutine).IsFinished()) {
      RETURN_IF_ERROR(scheduler_->GetFsm(subroutine).PrepareToSchedule());
    } else {
      // If we picked this subroutine, and it is finished, it means that
      // all subroutines are finished and parent instruction can be
      // marked finished
      scheduler_->InstructionFinished(instruction_,
                                      scheduler_->GetCurrentTime());
      return absl::OkStatus();
    }
  } else {
    // If we see Null opcode, which is ParaGraph internal opcode, we do not
    // schedule it through scheduler and just instantly execute it
    if (instruction_->GetOpcode() == Opcode::kNull) {
      scheduler_->InstructionFinished(instruction_,
                                      scheduler_->GetCurrentTime());
    } else {
      SetReady();
      SetTimeReady(scheduler_->GetCurrentTime());
      scheduler_->EnqueueToScheduler(instruction_);
    }
  }
  return absl::OkStatus();
}

shim::StatusOr<Subroutine*> InstructionFsm::PickSubroutine() {
  Subroutine* picked_subroutine;
  if (instruction_->GetOpcode() == Opcode::kConditional) {
    RETURN_IF_FALSE(instruction_->InnerSubroutines().size() >= 2,
                    absl::InternalError) << "Conditional instruction " <<
        instruction_->GetName() << " should have at least 2 subroutines.";

    // Setup the weights for distribution
    std::vector<float> weights;
    for (auto& subroutine : instruction_->InnerSubroutines()) {
      weights.push_back(subroutine->GetExecutionProbability());
    }
    // Create the distribution with the weights above
    std::discrete_distribution<uint64_t> dist(weights.begin(), weights.end());
    picked_subroutine = instruction_->InnerSubroutines().at(
        dist(scheduler_->prng_)).get();
  } else if (instruction_->GetOpcode() == Opcode::kWhile) {
    RETURN_IF_FALSE(instruction_->InnerSubroutines().size() == 2,
                    absl::InternalError) << "While instruction " <<
        instruction_->GetName() << " should have exactly 2 subroutines.";
    // Reset state if iteration of loop cycle is finished, next available
    auto body_subroutine = instruction_->InnerSubroutines().at(0).get();
    auto condition_subroutine = instruction_->InnerSubroutines().at(1).get();
    if (scheduler_->GetFsm(body_subroutine).IsFinished() &&
        scheduler_->GetFsm(condition_subroutine).IsFinished() &&
        scheduler_->GetFsm(condition_subroutine).GetExecutionCount() > 0) {
      scheduler_->GetFsm(body_subroutine).Reset(false);
      scheduler_->GetFsm(condition_subroutine).Reset(false);
    }
    if (!scheduler_->GetFsm(body_subroutine).IsFinished()) {
      picked_subroutine = body_subroutine;
    } else {
      picked_subroutine = condition_subroutine;
    }
  } else if (instruction_->GetOpcode() == Opcode::kCall) {
    for (auto& subroutine : instruction_->InnerSubroutines()) {
      picked_subroutine = subroutine.get();
      if (!scheduler_->GetFsm(picked_subroutine).IsFinished() &&
          scheduler_->GetFsm(picked_subroutine).GetExecutionCount() > 0) {
        break;
      }
    }
  } else {
    // Only kConditional, kWhile, and kCall should have more than one subroutine
    RETURN_IF_FALSE(instruction_->InnerSubroutines().size() <= 1,
                    absl::InternalError) << "Instruction " <<
        instruction_->GetName() <<
        " should have not more than 1 subroutine.";
    picked_subroutine = instruction_->InnerSubroutines().at(0).get();
  }
  return picked_subroutine;
}

}  // namespace paragraph
