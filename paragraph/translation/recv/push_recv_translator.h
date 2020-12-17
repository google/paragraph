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
#ifndef PARAGRAPH_TRANSLATION_RECV_PUSH_RECV_TRANSLATOR_H_
#define PARAGRAPH_TRANSLATION_RECV_PUSH_RECV_TRANSLATOR_H_

#include <memory>
#include <string>

#include "paragraph/translation/recv/recv_translator.h"

namespace paragraph {

/* PushRecvTranslator models a Recv instruction in a simple push-based
 * protocol. Every Recv is substituted with RecvStart and RecvDone instructions.
 * RecvStart allocated memory buffer, after which receiver is physically ready
 * to receive a message. RecvDone is executed when data is physically received
 * by the receiver. Together RecvStart/RecvDone can model blocking recv sequence.
 */
class PushRecvTranslator : public RecvTranslator {
 public:
  explicit PushRecvTranslator(nlohmann::json config);
  ~PushRecvTranslator() = default;

  shim::StatusOr<std::unique_ptr<Subroutine>> GetSubroutine(
      const std::string& name_prefix,
      Instruction* calling_instruction,
      int64_t processor_id,
      const CommunicationGroup& comm_group,
      double comm_size) const final;

 private:
};

}  // namespace paragraph

#endif  // PARAGRAPH_TRANSLATION_RECV_PUSH_RECV_TRANSLATOR_H_
