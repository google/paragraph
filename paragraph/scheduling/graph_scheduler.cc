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

namespace paragraph {

GraphScheduler::GraphScheduler(Graph* graph)
    : graph_(graph),
      current_time_(0.0) {}

shim::StatusOr<std::unique_ptr<GraphScheduler>> GraphScheduler::Create(
    Graph* graph) {
  // Check that graph is valid and can be scheduled
  RETURN_IF_ERROR(graph->ValidateIndividualized());
  // Using `new` and WrapUnique to access a non-public constructor.
  auto scheduler = absl::WrapUnique(new GraphScheduler(graph));

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

absl::Status GraphScheduler::Init(double seconds) {
  current_time_ = seconds;
  // Reset all FSMs recursively starting from entry subroutine
  GetFsm(graph_->GetEntrySubroutine()).Reset();
  // Moves available to be scheduled instructions from entry subroutine to the
  // scheduling queue
  RETURN_IF_ERROR(
      GetFsm(graph_->GetEntrySubroutine()).PrepareToSchedule());
  return absl::OkStatus();
}

void GraphScheduler::InstructionStarted(
    Instruction* instruction, double seconds) {
  current_time_ = seconds;
  GetFsm(instruction).SetTimeStarted(seconds);
}

void GraphScheduler::InstructionFinished(
    Instruction* instruction, double seconds) {
  current_time_ = seconds;
  GetFsm(instruction).SetFinished();
  GetFsm(instruction).SetTimeFinished(seconds);
  CHECK_OK(GetFsm(instruction->GetParent()).InstructionFinished(instruction));
  for (auto& user : instruction->Users()) {
    if (GetFsm(user).IsUnblockedByOperands()) {
      CHECK_OK(GetFsm(user).PrepareToSchedule());
    }
  }
}

void GraphScheduler::EnqueueToScheduler(Instruction* instruction) {
  ready_to_schedule_.push_back(instruction);
}

std::vector<Instruction*> GraphScheduler::GetReadyInstructions() {
  std::vector<Instruction *> rtn;
  for (auto& instruction : ready_to_schedule_) {
    GetFsm(instruction).SetScheduled();
    rtn.push_back(instruction);
  }
  ready_to_schedule_.clear();
  return rtn;
}

void GraphScheduler::GetReadyInstructions(std::queue<Instruction*>& queue) {
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

double GraphScheduler::GetCurrentTime() { return current_time_; }
void GraphScheduler::SetCurrentTime(double seconds) { current_time_ = seconds; }

void GraphScheduler::SeedRandom(uint64_t seed) {
  std::seed_seq seq = {(uint32_t)((seed >> 32) & 0xFFFFFFFFlu),
                       (uint32_t)((seed >>  0) & 0xFFFFFFFFlu)};
  prng_.seed(seq);
}

}  // namespace paragraph
