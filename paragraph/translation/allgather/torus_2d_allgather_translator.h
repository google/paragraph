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
#ifndef PARAGRAPH_TRANSLATION_ALLGATHER_TORUS_2D_ALLGATHER_TRANSLATOR_H_
#define PARAGRAPH_TRANSLATION_ALLGATHER_TORUS_2D_ALLGATHER_TRANSLATOR_H_

#include <memory>
#include <string>
#include <vector>

#include "paragraph/translation/allgather/allgather_translator.h"

namespace paragraph {

/* Torus2dAllGatherTranslator models a all-gather collective on the
 * 2D-Torus topology as a two consecutive communications on BiDirectional Rings
 * corresponding ot each torus dimension. In this topology, all two neighbors
 * in each dimension are connected together in a ring. On the torus of size
 * M x N, the algorithm splits all-gather into two phases. First it gathers data
 * from the processors across the first dimentsion of size M. Second it gathers
 * partially gathered data across the 2nd dimenstion. In case torus has
 * concentration (more than one processor corresponding to a single torus node),
 * local communication for concentration neighbors is resolved first. All the
 * communication in all the phases should have size 1 / (M x N)-th of total
 * data.
 */
class Torus2dAllGatherTranslator : public AllGatherTranslator {
 public:
  explicit Torus2dAllGatherTranslator(nlohmann::json config);
  ~Torus2dAllGatherTranslator() = default;

  shim::StatusOr<std::unique_ptr<Subroutine>> GetSubroutine(
      const std::string& name_prefix,
      Instruction* calling_instruction,
      int64_t processor_id,
      int64_t processor_index,
      const CommunicationGroup& comm_group,
      double comm_size) const final;

 private:
  std::unique_ptr<AllGatherTranslator> allgather_translator_;

  // Sizes of each dimension in 2D torus
  std::vector<uint64_t> dimension_sizes_;
  // Number of processors per torus node
  uint64_t concentration_;
  // concentrators
  bool integrated_local_exchange_;
};

}  // namespace paragraph

#endif  // PARAGRAPH_TRANSLATION_ALLGATHER_TORUS_2D_ALLGATHER_TRANSLATOR_H_
