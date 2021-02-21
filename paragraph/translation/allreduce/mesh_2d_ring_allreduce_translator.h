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
#ifndef PARAGRAPH_TRANSLATION_ALLREDUCE_MESH_2D_RING_ALLREDUCE_TRANSLATOR_H_
#define PARAGRAPH_TRANSLATION_ALLREDUCE_MESH_2D_RING_ALLREDUCE_TRANSLATOR_H_

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "paragraph/translation/allreduce/allreduce_translator.h"

namespace paragraph {

/* Mesh2dRingAllReduceTranslator models an all-reduce collective on the
 * 2D-Mesh topology as if it was a ring topology. Communication group is changed
 * so all communication happens on a ring swizzled through the 2D Mesh. In this
 * translation, communication happens only between two neighbors of each
 * processor, even though there are four neighbors for each processor on the
 * mesh (except for edge and corner processors). On the mesh of size M x N, each
 * step the algorithm moves 1 / (M x N)-th of total data. In case of non-trivial
 * concentration, the concentrated nodes become neighbors on the ring and
 * consume concentrator bandwidth (i.e. on-chip bw for multi-core processors).
 */
class Mesh2dRingAllReduceTranslator : public AllReduceTranslator {
 public:
  explicit Mesh2dRingAllReduceTranslator(nlohmann::json config);
  ~Mesh2dRingAllReduceTranslator() = default;

  shim::StatusOr<std::unique_ptr<Subroutine>> GetSubroutine(
      Subroutine* reduction_subroutine,
      const std::string& name_prefix,
      Instruction* calling_instruction,
      int64_t processor_id,
      int64_t processor_index,
      const CommunicationGroup& comm_group,
      double comm_size) const final;

 private:
  std::unique_ptr<AllReduceTranslator> allreduce_translator_;

  // Sizes of each dimension in 2D mesh
  std::vector<uint64_t> dimension_sizes_;
  // Number of processors per mesh node
  uint64_t concentration_;
};

}  // namespace paragraph

#endif  // PARAGRAPH_TRANSLATION_ALLREDUCE_MESH_2D_RING_ALLREDUCE_TRANSLATOR_H_
