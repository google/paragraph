/* Copyright 2021 Nic McDonald
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
#ifndef PARAGRAPH_TRANSLATION_ALLREDUCE_DISSEMINATION_ALLREDUCE_TRANSLATOR_H_
#define PARAGRAPH_TRANSLATION_ALLREDUCE_DISSEMINATION_ALLREDUCE_TRANSLATOR_H_

#include <memory>
#include <string>

#include "paragraph/translation/allreduce/allreduce_translator.h"

namespace paragraph {

/* DisseminationAllReduceTranslator models an all-reduce collective on
 * following the dissemination algorithm proposed by Hensgen et. al. in
 * "Two algorithms for barrier synchronization" (1988) and outlined by
 * Mellor-Crummey et. al. in "Algorithms for scalable synchronization on
 * shared-memory multiprocessors" (1991).
 */
class DisseminationAllReduceTranslator : public AllReduceTranslator {
 public:
  explicit DisseminationAllReduceTranslator(nlohmann::json config);
  ~DisseminationAllReduceTranslator() = default;

  shim::StatusOr<std::unique_ptr<Subroutine>> GetSubroutine(
      Subroutine* reduction_subroutine,
      const std::string& name_prefix,
      Instruction* calling_instruction,
      int64_t processor_id,
      int64_t processor_index,
      const CommunicationGroup& comm_group,
      double comm_size) const final;
};

}  // namespace paragraph

#endif  // PARAGRAPH_TRANSLATION_ALLREDUCE_DISSEMINATION_ALLREDUCE_TRANSLATOR_H_
