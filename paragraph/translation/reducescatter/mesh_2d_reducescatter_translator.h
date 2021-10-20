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
#ifndef PARAGRAPH_TRANSLATION_REDUCESCATTER_MESH_2D_REDUCESCATTER_TRANSLATOR_H_
#define PARAGRAPH_TRANSLATION_REDUCESCATTER_MESH_2D_REDUCESCATTER_TRANSLATOR_H_

#include <memory>
#include <string>
#include <vector>

#include "paragraph/translation/reducescatter/reducescatter_translator.h"

namespace paragraph {

/* Mesh2dReduceScatterTranslator models a reduce-scatter collective on the
 * 2D-Mesh topology as a two consecutive communications on 1D-Mesh topology
 * of the dimensions of 2D-Mesh. In this topology, all two neighbors in each
 * dimension are connected together in a linear sequence except for termainating
 * nodes (left and right, up and bottom). On the mesh of size M x N, the
 * algorithm splits reduce-scatter into two phases. First it reduces data from
 * the processors across the first dimentsion of size M. Second it reduces
 * partially reduced data across the 2nd dimenstion. In case mesh has
 * concentration (more than one processor corresponding to a single mesh node),
 * local communication for concentration neighbors is resolved first. All the
 * communication in all the phases should have size 1 / (M x N)-th of total
 * data.
 * If nodes don't have messages to propagate (i.e. termination
 * nodes only propagate messages ones), nodes don't send anything, and their
 * peers don't expect any messages from them.
 */
class Mesh2dReduceScatterTranslator : public ReduceScatterTranslator {
 public:
  explicit Mesh2dReduceScatterTranslator(nlohmann::json config);
  ~Mesh2dReduceScatterTranslator() = default;

  shim::StatusOr<std::unique_ptr<Subroutine>> GetSubroutine(
      Subroutine* reduction_subroutine,
      const std::string& name_prefix,
      Instruction* calling_instruction,
      int64_t processor_id,
      int64_t processor_index,
      const CommunicationGroup& comm_group,
      double comm_size) const final;

 private:
  std::unique_ptr<ReduceScatterTranslator> reducescatter_translator_;

  // Sizes of each dimension in 2D mesh
  std::vector<uint64_t> dimension_sizes_;
  // Number of processors per mesh node
  uint64_t concentration_;
  // concentrators
  bool integrated_local_exchange_;
};

}  // namespace paragraph

#endif  // PARAGRAPH_TRANSLATION_REDUCESCATTER_MESH_2D_REDUCESCATTER_TRANSLATOR_H_
