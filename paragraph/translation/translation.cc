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
#include "paragraph/translation/translation.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "paragraph/translation/translation_map.h"

namespace paragraph {

shim::StatusOr<std::vector<std::unique_ptr<Graph>>> IndividualizeAndTranslate(
    const Graph* composite_graph, nlohmann::json translation_config) {
  std::vector <std::unique_ptr<Graph>> individualized_graphs;
  RETURN_IF_ERROR(composite_graph->ValidateComposite());

  // Create translator configs
  nlohmann::json collective_config;
  nlohmann::json protocol_config;
  for (auto it = translation_config.begin();
       it != translation_config.end();
       ++it) {
    if (it.key() == "collective") {
      collective_config = translation_config["collective"];
    } else if (it.key() == "protocol") {
      protocol_config = translation_config["protocol"];
    } else {
      return absl::InvalidArgumentError("Invalid field " + it.key() +
                                        " in the translation config.");
    }
  }

  // Create translators
  ASSIGN_OR_RETURN(auto collective_translators,
                   CreateTranslators(TranslatorType::kCollective,
                                     collective_config));
  ASSIGN_OR_RETURN(auto protocol_translators,
                   CreateTranslators(TranslatorType::kProtocol,
                                     protocol_config));

  // Create an ordered processor ID vector
  const absl::flat_hash_set<int64_t>& comm_set =
      composite_graph->GetCommunicationSet();
  std::vector<int64_t> processor_ids(comm_set.begin(), comm_set.end());
  std::sort(processor_ids.begin(), processor_ids.end());

  // Create individualized graphs and translate them
  for (int64_t processor : processor_ids) {
    ASSIGN_OR_RETURN(std::unique_ptr<Graph> individualized_graph,
                     composite_graph->Individualize(processor));
    // Collective translation
    for (auto& subroutine : individualized_graph->Subroutines()) {
      for (auto& instruction : subroutine->Instructions()) {
        if (collective_translators.find(OpcodeToString(
            instruction->GetOpcode())) != collective_translators.end()) {
          RETURN_IF_ERROR(collective_translators[OpcodeToString(
              instruction->GetOpcode())]->Translate(instruction.get()));
        }
      }
    }
    // Protocol translation
    for (auto& subroutine : individualized_graph->Subroutines()) {
      for (auto& instruction : subroutine->Instructions()) {
        if (protocol_translators.find(OpcodeToString(
            instruction->GetOpcode())) != protocol_translators.end()) {
          RETURN_IF_ERROR(protocol_translators[OpcodeToString(
              instruction->GetOpcode())]->Translate(instruction.get()));
        }
      }
    }
    // Communication tags
    individualized_graph->ApplyCommunicationTags();

    RETURN_IF_ERROR(individualized_graph->ValidateIndividualized());
    individualized_graphs.push_back(std::move(individualized_graph));
  }
  return individualized_graphs;
}

}  // namespace paragraph
