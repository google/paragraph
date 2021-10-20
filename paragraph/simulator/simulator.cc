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
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "paragraph/graph/graph.h"
#include "paragraph/simulator/simple_sim.h"
#include "paragraph/shim/macros.h"

ABSL_FLAG(std::string, input_graph, "",
          "Input graph file.");
ABSL_FLAG(std::string, log_file, "",
          "Output log file.");
ABSL_FLAG(bool, skip_zeros, true,
          "Skip logging instruction with zero execution time.");
// defaults from V100 - 125 TF/s, 900 GB/s, NVLink 6 x 200 gbps
// TPUv2 - 46/2 TF/s per core, 700 GB/s, ICI - 4 x 496 Gbps
// TPUv3 - 123/2 TF/s per core, 900 GB/s, ICI - 6 x 656 Gbps
// A100 - 312 TF/s, 1555/2039 GB/s, NVLink 600 gbps
// data - https://cacm.acm.org/magazines/2020/7/245702-a-domain-specific-supercomputer-for-training-deep-neural-networks/fulltext
ABSL_FLAG(double, estimate_mem_gibps, 900,
          "Estimate for GiB/s for processing unit's memory bandwidth.");
ABSL_FLAG(double, estimate_tflops, 125.0,
          "Estimate for TFLOPs/s for processing unit's performance.");
ABSL_FLAG(double, estimate_net_gbit, 656.0,
          "Estimate for network bandwidth in gbps per link.");

int32_t main(int32_t argc, char** argv) {
  // Parsing flags
  absl::ParseCommandLine(argc, argv);

  std::string graph_path = absl::GetFlag(FLAGS_input_graph);
  CHECK_NE(graph_path, "");
  std::string log_file = absl::GetFlag(FLAGS_log_file);
  CHECK_NE(log_file, "");

  double flops = absl::GetFlag(FLAGS_estimate_tflops) * 1e12;
  double mem_bps = absl::GetFlag(FLAGS_estimate_mem_gibps) * 1024 * 1024 * 1024;
  double net_bit_ps = absl::GetFlag(FLAGS_estimate_net_gbit) * 1024 * 1024
      * 1024 / 8.0;
  const auto perf_parameters =
      paragraph::SimpleSim::PerformanceParameters(flops, mem_bps, net_bit_ps);

  // Reading input graph
  auto graph_statusor = paragraph::Graph::ReadFromFile(graph_path);
  CHECK_OK(graph_statusor.status());
  std::unique_ptr<paragraph::Graph> input_graph =
      std::move(graph_statusor.value());
  std::unique_ptr<paragraph::Graph> graph;
  CHECK(input_graph->IsIndividualized());

  bool skip_zeros = absl::GetFlag(FLAGS_skip_zeros);
  CHECK_OK_AND_ASSIGN(auto logger, paragraph::Logger::Create(log_file,
                                                             skip_zeros));
  CHECK_OK_AND_ASSIGN(auto sim, paragraph::SimpleSim::Create(
      std::move(input_graph), perf_parameters, std::move(logger)));
  std::cout << "Simulation parameters:" << std::endl;
  std::cout << "\tFlops: " << absl::GetFlag(FLAGS_estimate_tflops) << " TFLOPs"
            << std::endl;
  std::cout << "\tMem BW: " << absl::GetFlag(FLAGS_estimate_mem_gibps)
            << " GiB/s" << std::endl;
  std::cout << "\tNet BW: " << absl::GetFlag(FLAGS_estimate_net_gbit)
            << " gbps" << std::endl;
  std::cout << "Starting simlator with simulation starting time 0.0s."
            << std::endl;
  CHECK_OK(sim->Simulate(0.0));
  std::cout << "Exiting simlator. Simulation is finished at simulation time "
            << sim->GetProcessorTime() << std::endl;
  return 0;
}
