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
#ifndef PARAGRAPH_TRANSLATION_ALLREDUCE_TORUS_2D_ALLREDUCE_TRANSLATOR_H_
#define PARAGRAPH_TRANSLATION_ALLREDUCE_TORUS_2D_ALLREDUCE_TRANSLATOR_H_

#include <memory>
#include <string>

#include "paragraph/translation/allgather/allgather_translator.h"
#include "paragraph/translation/allreduce/allreduce_translator.h"
#include "paragraph/translation/barrier/barrier_translator.h"
#include "paragraph/translation/reducescatter/reducescatter_translator.h"

namespace paragraph {

/* Torus2dAllReduceTranslator models an all-reduce collective on the
 * 2D-Torus topology as a two consequitive communications on a BiDirectional
 * Ring corresponding  to torus dimensions. The algorithm splits all-reduce
 * into 2 phases. First it implements ReduceScatter algorithm over 2D torus,
 * and then AllGather algorithm over 2D torus. If a barrier is passed to
 * all-reduce, it is passed down to both reduce-scatter and all-gather
 * 2D torus algorithms.
 */
class Torus2dAllReduceTranslator : public AllReduceTranslator {
 public:
  explicit Torus2dAllReduceTranslator(nlohmann::json config);
  ~Torus2dAllReduceTranslator() = default;

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
  std::unique_ptr<AllGatherTranslator> allgather_translator_;
};

}  // namespace paragraph

#endif  // PARAGRAPH_TRANSLATION_ALLREDUCE_TORUS_2D_ALLREDUCE_TRANSLATOR_H_
