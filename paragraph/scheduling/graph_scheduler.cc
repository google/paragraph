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
#include "paragraph/scheduling/graph_scheduler.h"

#include <algorithm>

namespace paragraph {

GraphScheduler::GraphScheduler(Graph* graph)
    : graph_(graph),
      current_time_(0.0),
      initialized_(false) {
  logger_ = nullptr;
}

shim::StatusOr<std::unique_ptr<GraphScheduler>> GraphScheduler::Create(
    Graph* graph, std::unique_ptr<Logger> logger) {
  // Check that graph is valid and can be scheduled
  RETURN_IF_ERROR(graph->ValidateIndividualized());
  // Using `new` and WrapUnique to access a non-public constructor.
  auto scheduler = absl::WrapUnique(new GraphScheduler(graph));
  if (logger != nullptr) {
    scheduler->logger_ = std::move(logger);
  }

  // Create all FSMs for instructions and subroutines
  for (const auto& subroutine : scheduler->graph_->Subroutines()) {
    RETURN_IF_FALSE(scheduler->subroutine_state_map_.find(subroutine) ==
                    scheduler->subroutine_state_map_.end(),
                    absl::InternalError) << "subroutine "
        << subroutine->GetName() << " is already in internal scheduler map."
        << " Graph might not be valid.";
    scheduler->subroutine_state_map_.emplace(
        subroutine, SubroutineFsm(scheduler.get(), subroutine));
    for (const auto& instruction : subroutine->Instructions()) {
      RETURN_IF_FALSE(
          scheduler->instruction_state_map_.find(instruction.get()) ==
          scheduler->instruction_state_map_.end(),
          absl::InternalError) << "Instruction " << instruction->GetName()
          << " is not found in internal scheduler map."
          << "Graph might not be valid.";
      scheduler->instruction_state_map_.emplace(
          instruction.get(),
          InstructionFsm(scheduler.get(), instruction.get()));
    }
  }
  return scheduler;
}

absl::Status GraphScheduler::Initialize(double current_time) {
  CHECK_LE(current_time_, current_time);
  current_time_ = current_time;
  initialized_ = true;
  // Reset all FSMs recursively starting from entry subroutine
  GetFsm(graph_->GetEntrySubroutine()).Reset();
  // Moves available to be scheduled instructions from entry subroutine to the
  // scheduling queue
  RETURN_IF_ERROR(
      GetFsm(graph_->GetEntrySubroutine()).PrepareToSchedule());
  return absl::OkStatus();
}

bool GraphScheduler::HasLogger() {
  return logger_ != nullptr;
}

void GraphScheduler::SetLogger(std::unique_ptr<Logger> logger) {
  logger_ = std::move(logger);
}

void GraphScheduler::InstructionStarted(
    Instruction* instruction, double current_time) {
  CHECK(initialized_);
  // CHECK_LE(current_time_, current_time);
  current_time_ = current_time;
  GetFsm(instruction).SetTimeStarted(current_time);
  GetFsm(instruction).SetExecuting();
}

void GraphScheduler::InstructionFinished(
    Instruction* instruction, double current_time) {
  CHECK_LE(current_time_, current_time);
  current_time_ = current_time;
  GetFsm(instruction).SetFinished();
  GetFsm(instruction).SetTimeFinished(current_time);
  if (instruction->InnerSubroutines().empty()) {
    GetFsm(instruction).SetClockTime(current_time -
                                     GetFsm(instruction).GetTimeStarted());
    GetFsm(instruction).SetWallTime(current_time -
                                    GetFsm(instruction).GetTimeStarted());
  }
  // When we finish instruction that has inner subroutines, we need to find
  // when the first instruction in inner subroutines has started. That marks the
  // start time of this instruction as its execution happens in scheduler, not
  // in simulator, and stays in Scheduled state.
  // Also updates instruction execution time to be a sum of nested instructions
  // execution time.
  if (instruction->InnerSubroutines().size() > 0) {
    double start_time = current_time;
    for (auto& subroutine : instruction->InnerSubroutines()) {
      for (auto& nested_instr : subroutine->Instructions()) {
        start_time = std::min(start_time,
                              GetFsm(nested_instr.get()).GetTimeStarted());
        GetFsm(instruction).SetClockTime(
            GetFsm(instruction).GetClockTime() +
            GetFsm(nested_instr.get()).GetClockTime());
      }
    }
    // We consider while instruction separately as we need to set start timer
    // only once and not set it every loop iteration
    if (instruction->GetOpcode() != Opcode::kWhile) {
      GetFsm(instruction).SetTimeStarted(start_time);
      GetFsm(instruction).SetWallTime(current_time - start_time);
    }
  }
  // Log instruction
  if (HasLogger()) {
    CHECK_OK(logger_->LogInstruction(GetFsm(instruction)));
  }
  // Unblock all the users of this instruction if they don't have any other
  // dependencies
  CHECK_OK(GetFsm(instruction->GetParent()).InstructionFinished(instruction));
  for (auto& user : instruction->Users()) {
    if (GetFsm(user).IsUnblockedByOperands()) {
      CHECK_OK(GetFsm(user).PrepareToSchedule());
    }
  }
}

void GraphScheduler::EnqueueToScheduler(Instruction* instruction) {
  CHECK(initialized_);
  ready_to_schedule_.push_back(instruction);
}

std::vector<Instruction*> GraphScheduler::GetReadyInstructions() {
  CHECK(initialized_);
  std::vector<Instruction *> rtn;
  for (auto& instruction : ready_to_schedule_) {
    GetFsm(instruction).SetScheduled();
    rtn.push_back(instruction);
  }
  ready_to_schedule_.clear();
  return rtn;
}

void GraphScheduler::GetReadyInstructions(std::queue<Instruction*>& queue) {
  CHECK(initialized_);
  for (auto& instruction : ready_to_schedule_) {
    GetFsm(instruction).SetScheduled();
    queue.push(instruction);
  }
  ready_to_schedule_.clear();
}

InstructionFsm& GraphScheduler::GetFsm(const Instruction* instruction) {
  // CHECK that instruction exists in internal map
  // Not Status bacause we believe at this point graph is valid
  CHECK(instruction_state_map_.find(instruction) !=
        instruction_state_map_.end()) << "Instruction "
      << instruction->GetName() << " is not found in internal scheduler map."
      << " Graph might not be valid.";
  return instruction_state_map_.at(instruction);
}

SubroutineFsm& GraphScheduler::GetFsm(const Subroutine* subroutine) {
  // CHECK that subroutine exists in internal map
  // Not Status bacause we believe at this point graph is valid
  CHECK(subroutine_state_map_.find(subroutine) !=
        subroutine_state_map_.end()) << "subroutine "
      << subroutine->GetName() << " is not found in internal scheduler map."
      << " Graph might not be valid.";
  return subroutine_state_map_.at(subroutine);
}

double GraphScheduler::GetCurrentTime() const { return current_time_; }

void GraphScheduler::SeedRandom(uint64_t seed) {
  std::seed_seq seq = {(uint32_t)((seed >> 32) & 0xFFFFFFFFlu),
                       (uint32_t)((seed >>  0) & 0xFFFFFFFFlu)};
  prng_.seed(seq);
}

}  // namespace paragraph
