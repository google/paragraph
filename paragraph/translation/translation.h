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
#ifndef PARAGRAPH_TRANSLATION_TRANSLATION_H_
#define PARAGRAPH_TRANSLATION_TRANSLATION_H_

#include <memory>
#include <vector>

#include "paragraph/graph/graph.h"
#include "nlohmann/json.hpp"
#include "paragraph/shim/statusor.h"

namespace paragraph {

// Takes a composite graph and translation configuration and returns one
// individualized and translated graph per processor. If the composite graph has
// natural consecutive processor IDs, the resulting vector indices will match
// the processor IDs, otherwise they will be in ascending order.
shim::StatusOr<std::vector<std::unique_ptr<Graph>>> IndividualizeAndTranslate(
    const Graph* composite_graph, nlohmann::json translation_config);

}  // namespace paragraph

#endif  // PARAGRAPH_TRANSLATION_TRANSLATION_H_
