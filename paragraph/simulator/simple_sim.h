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
#ifndef PARAGRAPH_SIMULATOR_SIMPLE_SIM_H_
#define PARAGRAPH_SIMULATOR_SIMPLE_SIM_H_

#include <memory>
#include <queue>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "paragraph/graph/graph.h"
#include "paragraph/scheduling/graph_scheduler.h"
#include "paragraph/scheduling/logger.h"

namespace paragraph {

// Simple simulator is an in-order scalar processor that fetches and executes a
// single instruction at a time. It either simply increment time by the
// instruction modeling time, or it executes a simple roof line model, taking
// time corresponding to whatever is the longest - computation defined by flops,
// memory access defined by memory bandwidth, or network operation defined by
// network banwidth. Network operation assumes zero overhead ideal network -
// all-to-all connection, zero wire latency, no congestions, and time equal
// message size devided by link bandwidth.
class SimpleSim {
 public:
  // Execution state of instruction executing in a simulator. Used by simulator
  // Priority Queue that executes next instruction
  struct PerformanceParameters {
    double flops_;
    double memory_bandwidth_;
    double network_bandwidth_;
    PerformanceParameters(double flops,
                          double memory_bandwidth,
                          double network_bandwidth)
        : flops_(flops),
          memory_bandwidth_(memory_bandwidth),
          network_bandwidth_(network_bandwidth){}
  };

  ~SimpleSim() = default;

  // Creates a new simulator given the graph and performance parameters of the
  // simulates system. It verifies that graph is valid, creates a scheduler,
  // and provides it a logger if it was created
  static shim::StatusOr<std::unique_ptr<SimpleSim>> Create(
      std::unique_ptr<Graph> graph,
      const PerformanceParameters& performance_parameters,
      std::unique_ptr<Logger> logger = nullptr);

  // Starts simulation at a given time. Simulation fetches instructions
  // immediately and is going to finish only when there are no new instructions
  // to fetch, and no instructions are left to execute
  absl::Status Simulate(double start_time = 0);

  // Fetches instructions from the scheduler and executes it incrementing
  // simulation time, a single instrction at a time
  absl::Status FetchAndExecute();

  // Returns internal simulation timer in seconds
  double GetProcessorTime() const;
  double GetNicTxTime() const;
  double GetNicRxTime() const;

 private:
  // Private constructor that is used by Create factory
  explicit SimpleSim(const PerformanceParameters& performance_parameters);

  // Helper function that fetches a single instruction from
  // 'fetched_instructions_' but does not execute it
  void InstructionFetch();

  // Graph that's going to be simulated
  std::unique_ptr<Graph> graph_;

  // Simulator time. We separately model processor time for non-communicaation
  // instruction and NIC time.
  // Processor can execute single non-communication instruction at every time,
  // NIC can provide full injection bandwidth to a single communication
  // instruction at every time.
  double processor_time_;
  double nic_tx_time_;
  double nic_rx_time_;

  // Performance parameters of simulated system, such as FLOPs, Mem BW, Net BW
  PerformanceParameters performance_parameters_;

  // Scheduler for the graph that provides instructions to fetch
  std::unique_ptr<GraphScheduler> scheduler_;

  // Queue with all the instructions that were fetched from the scheduler but
  // not yet executed
  std::queue<Instruction*> fetched_instructions_;
};

}  // namespace paragraph

#endif  // PARAGRAPH_SIMULATOR_SIMPLE_SIM_H_
