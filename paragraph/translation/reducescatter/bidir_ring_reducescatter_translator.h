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
#ifndef PARAGRAPH_TRANSLATION_REDUCESCATTER_BIDIR_RING_REDUCESCATTER_TRANSLATOR_H_
#define PARAGRAPH_TRANSLATION_REDUCESCATTER_BIDIR_RING_REDUCESCATTER_TRANSLATOR_H_

#include <memory>
#include <string>

#include "paragraph/translation/reducescatter/reducescatter_translator.h"
#include "paragraph/translation/barrier/barrier_translator.h"

namespace paragraph {

/* BiDirRingReduceScatterTranslator models a reduce-scatter collective on the
 * bi-directional ring topology. This algorithm works with any phisycal
 * topology that could be mapped to a logical bidirectional ring. The algorithm
 * splits reduce-scatter into two uniderectional rings - clock wise and counter
 * clock wise. Acros each unidirectional ring, 1/2N-th of data is reduces in
 * N-1 phases, where N is the size of the ring. Each phase, every processors
 * sends and receives data to/from both his cw and ccw neighbors. The algorithm
 * effectively works like two concurrent unidirring algorithms.
 */
class BiDirRingReduceScatterTranslator : public ReduceScatterTranslator {
 public:
  explicit BiDirRingReduceScatterTranslator(nlohmann::json config);
  ~BiDirRingReduceScatterTranslator() = default;

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
  std::unique_ptr<ReduceScatterTranslator> reducescatter_translator_;
};

}  // namespace paragraph

#endif  // PARAGRAPH_TRANSLATION_REDUCESCATTER_BIDIR_RING_REDUCESCATTER_TRANSLATOR_H_

