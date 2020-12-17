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
#include "paragraph/translation/translation_map.h"

#include <memory>

#include "absl/strings/str_cat.h"
#include "paragraph/graph/opcode.h"
#include "paragraph/shim/macros.h"

namespace paragraph {

shim::StatusOr<absl::flat_hash_map<std::string, std::unique_ptr<Translator>>>
    CreateTranslators(TranslatorType type, nlohmann::json config) {
  // Check if json config asks to translate valid instructions
  static auto* opcode_map = new absl::flat_hash_set<Opcode>({
#define TRANSLATED_OPCODES(opcode, translator) \
    opcode,
       TRANSLATOR_LIST(TRANSLATED_OPCODES)
#undef TRANSLATED_OPCODES
  });
  for (auto it = config.begin(); it != config.end(); ++it) {
    ASSIGN_OR_RETURN(auto opcode, StringToOpcode(it.key()));
    RETURN_IF_FALSE(opcode_map->find(opcode) != opcode_map->end(),
                    absl::InternalError) << "Instruction with opcode " <<
        it.key() << " has no translation available.";
    if (type == TranslatorType::kCollective) {
      RETURN_IF_FALSE(OpcodeIsCollectiveCommunication(opcode),
                      absl::InvalidArgumentError) << "Opcode "
          <<  it.key() << " is not valid for collective translation.";
    }
    if (type == TranslatorType::kProtocol) {
      RETURN_IF_FALSE(OpcodeIsIndividualCommunication(opcode),
                      absl::InvalidArgumentError) << "Opcode "
          << it.key() << " is not valid for protocol translation.";
    }
  }
  absl::flat_hash_map<std::string, std::unique_ptr<Translator>> translation_map;
#define OPCODE_TO_TRANSLATOR_ENTRY(opcode, translator)                         \
  if (config.find(OpcodeToString(opcode)) != config.end()) {                   \
    ASSIGN_OR_RETURN(translation_map[OpcodeToString(opcode)],                  \
                     translator::Create(config[OpcodeToString(opcode)]));      \
  }
    TRANSLATOR_LIST(OPCODE_TO_TRANSLATOR_ENTRY)
#undef OPCODE_TO_TRANSLATOR_ENTRY
  return translation_map;
}

}  // namespace paragraph
