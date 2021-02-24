/* Copyright 2021 Google LLC
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
#include "paragraph/simulator/simple_sim.h"

#include <iostream>
#include <memory>
#include <queue>
#include <utility>

#include "paragraph/graph/graph.h"
#include "paragraph/scheduling/graph_scheduler.h"
#include "paragraph/scheduling/logger.h"

namespace paragraph {

shim::StatusOr<std::unique_ptr<SimpleSim>> SimpleSim::Create(
    std::unique_ptr<Graph> graph,
    const PerformanceParameters& processor_parameters,
    std::unique_ptr<Logger> logger) {
  auto simulator = absl::WrapUnique(new SimpleSim(processor_parameters));
  simulator->graph_ = std::move(graph);
  ASSIGN_OR_RETURN(simulator->scheduler_,
                   GraphScheduler::Create(simulator->graph_.get(),
                                          std::move(logger)));
  return simulator;
}

absl::Status SimpleSim::StartSimulation(double start_time) {
  time_ = start_time;
  scheduler_->Initialize(time_);
  RETURN_IF_ERROR(FetchAndExecute());
  return absl::OkStatus();
}

absl::Status SimpleSim::FetchAndExecute() {
  InstructionFetch();
  while (!fetched_instructions_.empty()) {
    Instruction* executing_instruction = fetched_instructions_.front();
    fetched_instructions_.pop();
    scheduler_->InstructionStarted(executing_instruction, time_);
    Opcode opcode = executing_instruction->GetOpcode();
    RETURN_IF_TRUE(OpcodeIsCollectiveCommunication(opcode),
                    absl::InternalError) << "Bad opcode "
        << OpcodeToString(opcode) << ": simulator expects collectives to be "
        << "translated into SendStart-SendDone and RecvStart-RecvDone "
        << "sequences.";
    RETURN_IF_TRUE(OpcodeIsIndividualCommunication(opcode),
                    absl::InternalError) << "Bad opcode "
        << OpcodeToString(opcode) << ": simulator expexts communication "
        << "instructions to be passed after the protocol translation as a "
        << "sequence of SendStart-SendDone and RecvStart-RecvDone "
        << "instructions.";
    RETURN_IF_TRUE(OpcodeIsControlFlow(opcode),
                    absl::InternalError) << "Bad opcode "
        << OpcodeToString(opcode) << ": simulator expexts only leaf "
        << "instructions that don't have subroutines. Control flow "
        << "instructions are expected to be visible only in the scheduler.";
    if (OpcodeIsGeneralPurpose(opcode)) {
      // We expect performance model from the graph source already populated
      // the 'seconds' member in the instruction in the corresponding bridge
      time_ += executing_instruction->GetSeconds();
      scheduler_->InstructionFinished(executing_instruction, time_);
    } else if (opcode == Opcode::kSendStart) {
      // We take into account network delay only for SendStart instructions
      time_ += executing_instruction->GetBytesIn() /
          processor_parameters_.network_bandwidth_;
      scheduler_->InstructionFinished(executing_instruction, time_);
    } else if ((opcode == Opcode::kRecvStart) ||
               (opcode == Opcode::kSendDone) ||
               (opcode == Opcode::kRecvDone)) {
      // We model SendDone and RecvStart as instant so network instructions
      // don't block execution unless there is a packet transfer
      // We model RecvDone to be instant as we expect that data laready was
      // transfered, and to avoid accounting network delay twice for SendStart
      // and RecvDone corresponding to the same communication
      scheduler_->InstructionFinished(executing_instruction, time_);
    } else if (OpcodeIsCollectiveCommunication(opcode) ||
               OpcodeIsIndividualCommunication(opcode) ||
               OpcodeIsProtocolLevelCommunication(opcode) ||
               OpcodeIsControlFlow(opcode)) {
      // We are supposed to cover all of that already
      return absl::InternalError("Unexpected opcode " + OpcodeToString(opcode));
    } else {
      return absl::InternalError("Unexpected opcode " + OpcodeToString(opcode));
    }
    InstructionFetch();
  }
  return absl::OkStatus();
}

SimpleSim::SimpleSim(const PerformanceParameters& processor_parameters)
  : time_(0),
    processor_parameters_(processor_parameters) {}

void SimpleSim::InstructionFetch() {
  // Fetches all available instructions from the scheduler
  for (Instruction* instruction : scheduler_->GetReadyInstructions()) {
    fetched_instructions_.push(instruction);
  }
}

}  // namespace paragraph
