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

#include <memory>
#include <queue>
#include <utility>

#include "paragraph/graph/graph.h"
#include "paragraph/scheduling/graph_scheduler.h"
#include "paragraph/scheduling/logger.h"

namespace paragraph {

shim::StatusOr<std::unique_ptr<SimpleSim>> SimpleSim::Create(
    std::unique_ptr<Graph> graph,
    const PerformanceParameters& performance_parameters,
    std::unique_ptr<Logger> logger) {
  auto simulator = absl::WrapUnique(new SimpleSim(performance_parameters));
  simulator->graph_ = std::move(graph);
  ASSIGN_OR_RETURN(simulator->scheduler_,
                   GraphScheduler::Create(simulator->graph_.get(),
                                          std::move(logger)));
  return simulator;
}

absl::Status SimpleSim::Simulate(double start_time) {
  processor_time_ = start_time;
  nic_tx_time_ = start_time;
  nic_rx_time_ = start_time;
  RETURN_IF_ERROR(scheduler_->Initialize(start_time));
  RETURN_IF_ERROR(FetchAndExecute());
  return absl::OkStatus();
}

absl::Status SimpleSim::FetchAndExecute() {
  InstructionFetch();
  while (!fetched_instructions_.empty()) {
    Instruction* executing_instruction = fetched_instructions_.front();
    fetched_instructions_.pop();
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
      double current_time = std::max(
          scheduler_->GetFsm(executing_instruction).GetTimeReady(),
          processor_time_);
      scheduler_->InstructionStarted(executing_instruction, current_time);
      current_time += executing_instruction->GetSeconds();
      scheduler_->InstructionFinished(executing_instruction, current_time);
      processor_time_ = current_time;
    } else if ((opcode == Opcode::kSendStart) ||
               (opcode == Opcode::kSendDone)) {
      double current_time = std::max(
          scheduler_->GetFsm(executing_instruction).GetTimeReady(),
          nic_tx_time_);
      scheduler_->InstructionStarted(executing_instruction, current_time);
      if (opcode == Opcode::kSendStart) {
        // We take into account network delay on tx_link between current
        // processor and its peer for SendStart instructions, while we model
        // SendDone as instant
        current_time += executing_instruction->GetBytesIn() /
            performance_parameters_.network_bandwidth_;
      }
      scheduler_->InstructionFinished(executing_instruction, current_time);
      nic_tx_time_ = current_time;
    } else if ((opcode == Opcode::kRecvStart) ||
               (opcode == Opcode::kRecvDone)) {
      double current_time = std::max(
          scheduler_->GetFsm(executing_instruction).GetTimeReady(),
          nic_rx_time_);
      scheduler_->InstructionStarted(executing_instruction, current_time);
      if (opcode == Opcode::kRecvDone) {
        // We take into account network delay on rx_link between current
        // processor and its peer for RecvDone instructions. We model RecvStart
        // as instant, as we believe that all memory is allocated a priori
        current_time += executing_instruction->GetBytesOut() /
            performance_parameters_.network_bandwidth_;
      }
      scheduler_->InstructionFinished(executing_instruction, current_time);
      nic_rx_time_ = current_time;
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

SimpleSim::SimpleSim(const PerformanceParameters& performance_parameters)
  : processor_time_(0),
    nic_tx_time_(0),
    nic_rx_time_(0),
    performance_parameters_(performance_parameters) {}

void SimpleSim::InstructionFetch() {
  // Fetches all available instructions from the scheduler
  for (Instruction* instruction : scheduler_->GetReadyInstructions()) {
    fetched_instructions_.push(instruction);
  }
}

double SimpleSim::GetProcessorTime() const {
  return processor_time_;
}

double SimpleSim::GetNicTxTime() const {
  return nic_tx_time_;
}

double SimpleSim::GetNicRxTime() const {
  return nic_rx_time_;
}

}  // namespace paragraph
