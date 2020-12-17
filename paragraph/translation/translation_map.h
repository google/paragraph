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
#ifndef PARAGRAPH_TRANSLATION_TRANSLATION_MAP_H_
#define PARAGRAPH_TRANSLATION_TRANSLATION_MAP_H_

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "nlohmann/json.hpp"
#include "paragraph/shim/statusor.h"
#include "paragraph/translation/translator.h"
#include "paragraph/translation/allgather/allgather_translator.h"
#include "paragraph/translation/allreduce/allreduce_translator.h"
#include "paragraph/translation/barrier/barrier_translator.h"
#include "paragraph/translation/recv/recv_translator.h"
#include "paragraph/translation/reducescatter/reducescatter_translator.h"
#include "paragraph/translation/send/send_translator.h"
#include "paragraph/translation/sendrecv/sendrecv_translator.h"

namespace paragraph {

#define TRANSLATOR_LIST(V)                                                     \
  V(Opcode::kAllGather, AllGatherTranslator)                                   \
  V(Opcode::kAllReduce, AllReduceTranslator)                                   \
  V(Opcode::kBarrier, BarrierTranslator)                                       \
  V(Opcode::kReduceScatter, ReduceScatterTranslator)                           \
  V(Opcode::kSend, SendTranslator)                                             \
  V(Opcode::kRecv, RecvTranslator)                                             \
  V(Opcode::kSendRecv, SendRecvTranslator)

enum class TranslatorType {
  kCollective,
  kProtocol,
};

shim::StatusOr<absl::flat_hash_map<std::string, std::unique_ptr<Translator>>>
    CreateTranslators(TranslatorType type, nlohmann::json config);

}  // namespace paragraph

#endif  // PARAGRAPH_TRANSLATION_TRANSLATION_MAP_H_
