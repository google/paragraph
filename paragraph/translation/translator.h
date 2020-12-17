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
#ifndef PARAGRAPH_TRANSLATION_TRANSLATOR_H_
#define PARAGRAPH_TRANSLATION_TRANSLATOR_H_

#include "paragraph/graph/graph.h"
#include "paragraph/shim/statusor.h"

namespace paragraph {

class Translator {
 public:
  virtual ~Translator() = default;

  // Abstract API to create a translation of any instruction that has an
  // opcode-specific Translator (defined in translation_map).
  // Translation would create a subroutine with a detailed low-level (lowered)
  // representation of the same instruction.
  virtual absl::Status Translate(Instruction* instruction) const = 0;
};

}  // namespace paragraph

#endif  // PARAGRAPH_TRANSLATION_TRANSLATOR_H_
