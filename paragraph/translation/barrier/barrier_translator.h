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
#ifndef PARAGRAPH_TRANSLATION_BARRIER_BARRIER_TRANSLATOR_H_
#define PARAGRAPH_TRANSLATION_BARRIER_BARRIER_TRANSLATOR_H_

#include <string>
#include <memory>

#include "nlohmann/json.hpp"
#include "paragraph/translation/translator.h"

namespace paragraph {

class BarrierTranslator : public Translator {
 public:
  virtual ~BarrierTranslator() = default;

  // Creates particular Barrier algorithm based on Json config
  static shim::StatusOr<std::unique_ptr<BarrierTranslator>>
      Create(nlohmann::json config);

  // Create a translation for a given instruction (part of high-level API)
  absl::Status Translate(Instruction* instruction) const final;

  // Creates a Subroutine that expands the instruction to be translated
  // given all the arguments needed to construct the instructions.
  // Part of low level API that could be called in another translator.
  virtual shim::StatusOr<std::unique_ptr<Subroutine>> GetSubroutine(
      const std::string& name_prefix,
      Instruction* calling_instruction,
      int64_t processor_id,
      int64_t processor_index,
      const CommunicationGroup& comm_group) const = 0;
};

}  // namespace paragraph

#endif  // PARAGRAPH_TRANSLATION_BARRIER_BARRIER_TRANSLATOR_H_
