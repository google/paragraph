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
#include "paragraph/translation/allreduce/mesh_2d_allreduce_translator.h"

#include <memory>
#include <string>
#include <utility>

#include "factory/ObjectFactory.h"

namespace paragraph {

Mesh2dAllReduceTranslator::Mesh2dAllReduceTranslator(
    nlohmann::json config) {
  CHECK_NE(config.find("dimension_widths"), config.end()) <<
      "2D Mesh should have field 'dimension_widths' as an array of size 2.";
  CHECK(config["dimension_widths"].is_array()) <<
      "2D Mesh config field 'dimension_widths' should be an array.";
  CHECK_EQ(config["dimension_widths"].size(), 2) <<
      "2D Mesh config field 'dimension_widths' should should have size 2.";
  // Create json config for internal reduce-scatter and all-gather
  nlohmann::json implicit_config = R"(
    {
      "all-gather": {
        "algorithm": "mesh-2d"
      },
      "reduce-scatter": {
        "algorithm": "mesh-2d"
      }
    }
  )"_json;
  implicit_config["all-gather"]["dimension_widths"] =
      config["dimension_widths"];
  implicit_config["reduce-scatter"]["dimension_widths"] =
      config["dimension_widths"];
  // Extract concentration (number of processors per mesh node) from config
  if (config.find("concentration") != config.end()) {
    implicit_config["all-gather"]["concentration"] =
        config["concentration"];
    implicit_config["reduce-scatter"]["concentration"] =
        config["concentration"];
  }
  // If we have a barrier in 2D Mesh, we need to pass it to both
  // reduce-scatter and all-gather
  if (config.find("barrier") != config.end()) {
    implicit_config["all-gather"]["barrier"] = config["barrier"];
    implicit_config["reduce-scatter"]["barrier"] = config["barrier"];
  }
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
    Mesh2dAllReduceTranslator::GetSubroutine(
        Subroutine* reduction_subroutine,
        const std::string& name_prefix,
        Instruction* calling_instruction,
        int64_t processor_id,
        int64_t processor_index,
        const CommunicationGroup& comm_group,
        double comm_size) const {
  auto graph = calling_instruction->GetGraph();
  auto allreduce_subroutine = absl::make_unique<Subroutine>(
      absl::StrCat(name_prefix, "_mesh-2d"), graph);
  auto allreduce_sub_ptr = allreduce_subroutine.get();
  RETURN_IF_FALSE(comm_group.at(processor_index) == processor_id,
                  absl::InvalidArgumentError) <<
      "Processor index points to the wrong Processor ID.";
  // Create ReduceScatter instruction
  ASSIGN_OR_RETURN(auto reducescatter, Instruction::Create(
      Opcode::kReduceScatter,
      absl::StrCat(name_prefix,
                   "_mesh-2d_reduce-scatter"),
      allreduce_sub_ptr));
  reducescatter->SetBytesOut(comm_size);
  reducescatter->AppendCommunicationGroup(comm_group);
  ASSIGN_OR_RETURN(auto new_reduction_subroutine,
                   reduction_subroutine->Clone(
                       "",
                       /*reset_ids*/ false));
  reducescatter->AppendInnerSubroutine(std::move(new_reduction_subroutine));
  RETURN_IF_ERROR(reducescatter_translator_->Translate(reducescatter));
  // Create AllGather instruction
  ASSIGN_OR_RETURN(auto allgather, Instruction::Create(
      Opcode::kAllGather,
      absl::StrCat(name_prefix,
                   "_mesh-2d_all-gather"),
      allreduce_sub_ptr, true));
  allgather->SetBytesOut(comm_size);
  allgather->AppendCommunicationGroup(comm_group);
  allgather->AddOperand(reducescatter);
  RETURN_IF_ERROR(allgather_translator_->Translate(allgather));
  return allreduce_subroutine;
}

registerWithObjectFactory(
    "mesh-2d",
    AllReduceTranslator,
    Mesh2dAllReduceTranslator,
    nlohmann::json);

}  // namespace paragraph
