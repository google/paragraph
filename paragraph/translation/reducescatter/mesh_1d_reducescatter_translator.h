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
#ifndef PARAGRAPH_TRANSLATION_REDUCESCATTER_MESH_1D_REDUCESCATTER_TRANSLATOR_H_
#define PARAGRAPH_TRANSLATION_REDUCESCATTER_MESH_1D_REDUCESCATTER_TRANSLATOR_H_

#include <memory>
#include <string>

#include "paragraph/translation/reducescatter/reducescatter_translator.h"
#include "paragraph/translation/barrier/barrier_translator.h"

namespace paragraph {

/* Mesh1dReduceScatterTranslator models a reduce-scatter collective on the
 * 1D-Mesh topology. In this topology, all two neighbors are connected together
 * in a linear sequence except for termainating nodes (left and right).
 * The algorithm splits reduce-scatter into N-1 phases, where N is the size of
 * the mesh. Each phase, 1/N-th of data is gathered from the left neighbor
 * ("i-1"-th processor) and then is sent to the right neighbor
 * ("i+1"-th processor) and wise versa. That effectively performs data shift
 * over the mesh. If nodes don't have messages to propagate (i.e. termination
 * nodes only propagate messages ones), nodes don't send anything, and their
 * peers don't expect any messages from them.
 */
class Mesh1dReduceScatterTranslator : public ReduceScatterTranslator {
 public:
  explicit Mesh1dReduceScatterTranslator(nlohmann::json config);
  ~Mesh1dReduceScatterTranslator() = default;

  shim::StatusOr<std::unique_ptr<Subroutine>> GetSubroutine(
      Subroutine* reduction_subroutine,
      const std::string& name_prefix,
      Instruction* calling_instruction,
      int64_t processor_id,
      int64_t processor_index,
      const CommunicationGroup& comm_group,
      double comm_size) const final;

 private:
  std::unique_ptr<BarrierTranslator> barrier_translator_;

  // Helper functions that tell wether processor will send message cw/ccw
  static bool WillSendCw(int64_t processor_index,
                         int64_t iteration,
                         int64_t communication_size);
  static bool WillSendCcw(int64_t processor_index,
                          int64_t iteration,
                          int64_t communication_size);
};

}  // namespace paragraph

#endif  // PARAGRAPH_TRANSLATION_REDUCESCATTER_MESH_1D_REDUCESCATTER_TRANSLATOR_H_
