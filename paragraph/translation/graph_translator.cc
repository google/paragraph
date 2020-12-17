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
#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_cat.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "paragraph/graph/graph.h"
#include "paragraph/shim/macros.h"
#include "paragraph/translation/translation.h"

ABSL_FLAG(std::string, input_graph, "",
          "Input graph file.");
ABSL_FLAG(std::string, translation_config, "",
          "Translation configuration file (JSON).");
ABSL_FLAG(std::string, output_dir, "",
          "Output directory.");
ABSL_FLAG(std::string, output_ext, "auto",
          "File extension of output graphs ('auto' for same as input')");
ABSL_FLAG(bool, check_consecutive_natural_processor_ids, false,
          "Checks that Processor IDs in the graph start with 0 and "
          "are consecutive");

int32_t main(int32_t argc, char** argv) {
  // Parsing flags
  absl::ParseCommandLine(argc, argv);
  std::filesystem::path input_graph = absl::GetFlag(FLAGS_input_graph);
  std::filesystem::path translation_config =
      absl::GetFlag(FLAGS_translation_config);
  std::filesystem::path output_dir = absl::GetFlag(FLAGS_output_dir);
  std::filesystem::path output_ext = absl::GetFlag(FLAGS_output_ext);
  bool check_processor_ids = absl::GetFlag(
      FLAGS_check_consecutive_natural_processor_ids);

  // Reading composite graph from the file
  auto graph_statusor = paragraph::Graph::ReadFromFile(input_graph);
  CHECK_OK(graph_statusor.status());
  std::unique_ptr<paragraph::Graph> composite_graph =
      std::move(graph_statusor.value());
  if (check_processor_ids) {
    CHECK(composite_graph->HasConsecutiveNaturalProcessorIds());
  }

  // Reading translation config from the file
  std::fstream config_input(translation_config,
                            std::ios::in | std::ios::binary);
  CHECK(config_input) << "File '" << translation_config <<
      "' could not be opened.";
  nlohmann::json translation_json;
  config_input >> translation_json;

  // Translate graph for all the processors into its communication set
  std::vector<std::unique_ptr<paragraph::Graph>> translated_graphs;
  auto translation_statusor = paragraph::IndividualizeAndTranslate(
      composite_graph.get(), translation_json);
  CHECK_OK(translation_statusor.status());
  translated_graphs = std::move(translation_statusor.value());

  // Write individualized graphs into output_dir
  std::filesystem::path input_path(input_graph);
  std::filesystem::path basename = input_path.stem();
  std::filesystem::path extension = output_ext;
  if (output_ext == "auto") {
    extension = input_path.extension();
  }
  for (auto& individualized_graph : translated_graphs) {
    std::filesystem::path filename = basename;
    filename += ".";
    filename += std::to_string(individualized_graph->GetProcessorId());
    filename += extension;
    std::filesystem::path output_path = output_dir / filename;
    CHECK_OK(individualized_graph->WriteToFile(output_path));
  }
  return 0;
}
