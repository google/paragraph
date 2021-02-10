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
#include "paragraph/translation/allreduce/mesh_1d_allreduce_translator.h"

#include <memory>
#include <string>
#include <utility>

#include "factory/ObjectFactory.h"

namespace paragraph {

Mesh1dAllReduceTranslator::Mesh1dAllReduceTranslator(
    nlohmann::json config) {
  barrier_translator_ = nullptr;
  if (config.find("barrier") != config.end()) {
    auto maybe_translator = BarrierTranslator::Create(config["barrier"]);
    CHECK_OK(maybe_translator.status());
    barrier_translator_ = std::move(maybe_translator.value());
  }
  // Create json config for internal reduce-scatter and all-gather
  nlohmann::json implicit_config = R"(
    {
      "all-gather": {
        "algorithm": "mesh-1d"
      },
      "reduce-scatter": {
        "algorithm": "mesh-1d"
      }
    }
  )"_json;
  auto maybe_reducescatter_translator = ReduceScatterTranslator::Create(
                       implicit_config["reduce-scatter"]);
  CHECK_OK(maybe_reducescatter_translator.status());
  reducescatter_translator_ = std::move(maybe_reducescatter_translator.value());
  auto maybe_allgather_translator = AllGatherTranslator::Create(
                       implicit_config["all-gather"]);
  CHECK_OK(maybe_allgather_translator.status());
  allgather_translator_ = std::move(maybe_allgather_translator.value());
}

shim::StatusOr<std::unique_ptr<Subroutine>>
    Mesh1dAllReduceTranslator::GetSubroutine(
        Subroutine* reduction_subroutine,
        const std::string& name_prefix,
        Instruction* calling_instruction,
        int64_t processor_id,
        int64_t processor_index,
        const CommunicationGroup& comm_group,
        double comm_size) const {
  auto graph = calling_instruction->GetGraph();
  auto allreduce_subroutine = absl::make_unique<Subroutine>(
      absl::StrCat(name_prefix, "_mesh-1d"), graph);
  auto allreduce_sub_ptr = allreduce_subroutine.get();
  RETURN_IF_FALSE(comm_group.at(processor_index) == processor_id,
                  absl::InvalidArgumentError) <<
      "Processor index points to the wrong Processor ID.";
  Instruction* previous_instruction = nullptr;
  // Check if there is barrier in config, and if so then instantiate it and
  // translate using specific translator from config
  if (barrier_translator_ != nullptr) {
    ASSIGN_OR_RETURN(auto barrier, Instruction::Create(
        Opcode::kBarrier,
        absl::StrCat(name_prefix,
                     "_mesh-1d_barrier"),
        allreduce_sub_ptr));
    previous_instruction = barrier;
    barrier->AppendCommunicationGroup(comm_group);
    RETURN_IF_ERROR(barrier_translator_->Translate(barrier));
  }
  // Create ReduceScatter instruction
  ASSIGN_OR_RETURN(auto reducescatter, Instruction::Create(
      Opcode::kReduceScatter,
      absl::StrCat(name_prefix,
                   "_mesh-1d_reduce-scatter"),
      allreduce_sub_ptr));
  reducescatter->SetBytesOut(comm_size);
  reducescatter->AppendCommunicationGroup(comm_group);
  ASSIGN_OR_RETURN(auto new_reduction_subroutine,
                   reduction_subroutine->Clone(
                       "",
                       /*reset_ids*/ false));
  reducescatter->AppendInnerSubroutine(std::move(new_reduction_subroutine));
  // If there is a barrier, we need to add it as control dependency to the
  // first communication after it
  if (previous_instruction != nullptr) {
    reducescatter->AddOperand(previous_instruction);
  }
  RETURN_IF_ERROR(reducescatter_translator_->Translate(reducescatter));
  // Create AllGather instruction
  ASSIGN_OR_RETURN(auto allgather, Instruction::Create(
      Opcode::kAllGather,
      absl::StrCat(name_prefix,
                   "_mesh-1d_all-gather"),
      allreduce_sub_ptr, true));
  allgather->SetBytesOut(comm_size);
  allgather->AppendCommunicationGroup(comm_group);
  allgather->AddOperand(reducescatter);
  RETURN_IF_ERROR(allgather_translator_->Translate(allgather));
  return allreduce_subroutine;
}

registerWithObjectFactory(
    "mesh-1d",
    AllReduceTranslator,
    Mesh1dAllReduceTranslator,
    nlohmann::json);

}  // namespace paragraph
