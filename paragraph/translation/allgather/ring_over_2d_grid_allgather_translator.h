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
#ifndef PARAGRAPH_TRANSLATION_ALLGATHER_RING_OVER_2D_GRID_ALLGATHER_TRANSLATOR_H_
#define PARAGRAPH_TRANSLATION_ALLGATHER_RING_OVER_2D_GRID_ALLGATHER_TRANSLATOR_H_

#include <memory>
#include <string>
#include <vector>

#include "paragraph/translation/allgather/allgather_translator.h"

namespace paragraph {

/* RingOver2dGridAllGatherTranslator models a all-gather collective on the
 * 2D-Grid topology as if it was a ring topology. In this context, 2D Grid means
 * 2D Mesh or 2D Torus with all processors placed on an integer coordinate grid.
 * Communication group is changed so all communication happens on a ring
 * swizzled through the 2D Grid. In this translation, communication happens only
 * between two neighbors of each processor, even though there are four neighbors
 * for each processor on the mesh (except for edge and corner processors).
 * On the mesh of size M x N, each step the algorithm moves 1 / (M x N)-th of
 * total data. In case of non-trivial concentration, the concentrated nodes
 * become neighbors on the ring and consume concentrator bandwidth
 * (i.e. on-chip bw for multi-core processors).
 */
class RingOver2dGridAllGatherTranslator : public AllGatherTranslator {
 public:
  explicit RingOver2dGridAllGatherTranslator(nlohmann::json config);
  ~RingOver2dGridAllGatherTranslator() = default;

  shim::StatusOr<std::unique_ptr<Subroutine>> GetSubroutine(
      const std::string& name_prefix,
      Instruction* calling_instruction,
      int64_t processor_id,
      int64_t processor_index,
      const CommunicationGroup& comm_group,
      double comm_size) const final;

 private:
  std::unique_ptr<AllGatherTranslator> allgather_translator_;

  // Sizes of each dimension in 2D mesh
  std::vector<uint64_t> dimension_sizes_;
  // Number of processors per mesh node
  uint64_t concentration_;
};

}  // namespace paragraph

#endif  // PARAGRAPH_TRANSLATION_ALLGATHER_RING_OVER_2D_GRID_ALLGATHER_TRANSLATOR_H_
