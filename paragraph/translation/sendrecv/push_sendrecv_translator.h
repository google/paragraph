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
#ifndef PARAGRAPH_TRANSLATION_SENDRECV_PUSH_SENDRECV_TRANSLATOR_H_
#define PARAGRAPH_TRANSLATION_SENDRECV_PUSH_SENDRECV_TRANSLATOR_H_

#include <memory>
#include <string>

#include "paragraph/translation/sendrecv/sendrecv_translator.h"

namespace paragraph {

/* PushSendRecvTranslator models a SendRecv instruction in a simple push-based
 * protocol. It unwraps it into a chain of SendStart, SendDone, RecvStart,
 * RecvDone instruction and a Root instruciton that helps to inforce a set of
 * dependenicies required by protocol to avoid potential circular deadlock.
 * The dependencies are following:
 * RecvStart happens before SendStart
 * SendStart happens before RecvDone
 * RecvStart happens before RecvDone
 * SendStart happens before SendDone
 * You can visualise the dependenices as such:
 * +---------+         +---------+
 * |RecvStart+--------->SendStart|
 * +--+------+         +--+---+--+
 *    |                   |   |
 *    |    +--------------+   |
 *    |    |                  |
 *    |    |                  |
 * +--v----v-+         +------v--+
 * |RecvDone |         |SendDone |
 * +----+----+         +----+----+
 *      |                   |
 *      +-------+   +-------+
 *              |   |
 *           +--v---v--+
 *           |  Null   |
 *           +---------+
 * */
class PushSendRecvTranslator : public SendRecvTranslator {
 public:
  explicit PushSendRecvTranslator(nlohmann::json config);
  ~PushSendRecvTranslator() = default;

  shim::StatusOr<std::unique_ptr<Subroutine>> GetSubroutine(
      const std::string& name_prefix,
      Instruction* calling_instruction,
      int64_t processor_id,
      const CommunicationGroup& comm_group,
      double comm_size_send,
      double comm_size_recv) const final;

 private:
};

}  // namespace paragraph

#endif  // PARAGRAPH_TRANSLATION_SENDRECV_PUSH_SENDRECV_TRANSLATOR_H_
