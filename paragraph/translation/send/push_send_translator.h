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
#ifndef PARAGRAPH_TRANSLATION_SEND_PUSH_SEND_TRANSLATOR_H_
#define PARAGRAPH_TRANSLATION_SEND_PUSH_SEND_TRANSLATOR_H_

#include <memory>
#include <string>

#include "paragraph/translation/send/send_translator.h"

namespace paragraph {

/* PushSendTranslator models a send instruction in a simple push-based
 * protocol. Every Send is substituted with SendStart and SendDone instructions.
 * SendStart sends data in a non-blocking way, while SendDone operation blocks
 * further execution until send is finished (i.e. data is received on the
 * receiver side, or sender's data buffer can be reused). Together
 * SendStart/SendDone can model blocking send sequence.
 */
class PushSendTranslator : public SendTranslator {
 public:
  explicit PushSendTranslator(nlohmann::json config);
  ~PushSendTranslator() = default;

  shim::StatusOr<std::unique_ptr<Subroutine>> GetSubroutine(
      const std::string& name_prefix,
      Instruction* calling_instruction,
      int64_t processor_id,
      const CommunicationGroup& comm_group,
      double comm_size) const final;

 private:
};

}  // namespace paragraph

#endif  // PARAGRAPH_TRANSLATION_SEND_PUSH_SEND_TRANSLATOR_H_
