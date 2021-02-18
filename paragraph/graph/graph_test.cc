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
#include "paragraph/graph/graph.h"

#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <tuple>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "paragraph/shim/test_macros.h"
#include "paragraph/translation/translation.h"

// Tests construction, name and processor id return
// Test constructs several applications with various names and processor ids,
// checks that processor ids and names are the ones that were passed to
// constructor
TEST(Graph, Construction) {
  std::string graph_0_name = "app0";
  auto graph_0 = absl::make_unique<paragraph::Graph>(graph_0_name);
  EXPECT_EQ(graph_0_name, graph_0->GetName());
  EXPECT_EQ(graph_0->GetProcessorId(), -1);

  std::string graph_1_name = "app1";
  int64_t graph_1_processor = 7;
  auto graph_1 = absl::make_unique<paragraph::Graph>(
      graph_1_name, graph_1_processor);
  EXPECT_EQ(graph_1_name, graph_1->GetName());
  EXPECT_EQ(graph_1_processor, graph_1->GetProcessorId());

  std::string graph_2_name = "app2";
  int64_t graph_2_processor = 42;
  auto graph_2 = absl::make_unique<paragraph::Graph>(
      graph_2_name, graph_2_processor);
  EXPECT_EQ(graph_2_name, graph_2->GetName());
  EXPECT_EQ(graph_2_processor, graph_2->GetProcessorId());
}

// Tests SetName() graph name setter
TEST(Graph, SetName) {
  std::string graph_name = "graph";
  int64_t graph_processor = 7;
  auto graph = absl::make_unique<paragraph::Graph>(
      graph_name, graph_processor);
  EXPECT_EQ(graph_name, graph->GetName());

  graph->SetName("new_name");
  EXPECT_EQ("new_name", graph->GetName());
}

// Tests SetProcessorId() graph processor id setter
TEST(Graph, SetProcessorId) {
  std::string graph_name = "graph";
  int64_t graph_processor = 7;
  auto graph = absl::make_unique<paragraph::Graph>(
      graph_name, graph_processor);
  EXPECT_EQ(graph_name, graph->GetName());

  graph->SetProcessorId(42);
  EXPECT_EQ(42, graph->GetProcessorId());
}

// Tests setter and getter for new unique id that graph provides for
// subroutines and instructions
TEST(Graph, SetInstructionId) {
  std::string graph_name = "graph";
  int64_t graph_processor = 7;
  auto graph = absl::make_unique<paragraph::Graph>(
      graph_name, graph_processor);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "test_instr", sub_ptr));
  EXPECT_EQ(instr->GetId(), 1);
  graph->SetInstructionId(instr);
  EXPECT_EQ(instr->GetId(), 2);

  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "test_instr_2", sub_ptr));
  EXPECT_EQ(instr_2->GetId(), 3);
  graph->SetInstructionId(instr_2);
  EXPECT_EQ(instr_2->GetId(), 4);
}

// Tests adding and checking Entry Subroutine
TEST(Graph, EntrySubroutine) {
  std::string graph_name = "graph";
  int64_t graph_processor = 7;
  auto graph = absl::make_unique<paragraph::Graph>(
      graph_name, graph_processor);
  EXPECT_EQ(graph->HasEntrySubroutine(), false);
  EXPECT_DEATH(graph->GetEntrySubroutine(), "");

  auto subroutine_1 = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine_1", graph.get());
  auto subroutine_1_ptr = subroutine_1.get();
  graph->SetEntrySubroutine(std::move(subroutine_1));
  EXPECT_EQ(graph->HasEntrySubroutine(), true);
  EXPECT_EQ(graph->GetEntrySubroutine(), subroutine_1_ptr);

  auto subroutine_2 = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine_2", graph.get());
  EXPECT_DEATH(graph->SetEntrySubroutine(std::move(subroutine_2)), "");
  EXPECT_EQ(graph->HasEntrySubroutine(), true);
  EXPECT_EQ(graph->GetEntrySubroutine(), subroutine_1_ptr);
  EXPECT_EQ(graph->GetEntrySubroutine()->GetId(), 0);
}

// Tests adding subroutines to graph.
// Also tests how SubroutineCount() handle new subroutines addition
TEST(Graph, AddingSubroutine) {
  std::string graph_name = "graph";
  int64_t graph_processor = 7;
  auto graph = absl::make_unique<paragraph::Graph>(
      graph_name, graph_processor);
  EXPECT_EQ(graph->SubroutineCount(), 0);

  auto subroutine = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = subroutine.get();
  graph->SetEntrySubroutine(std::move(subroutine));
  EXPECT_EQ(graph->SubroutineCount(), 1);
  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_1", sub_ptr));
  instr_1->SetOps(4);
  EXPECT_EQ(graph->SubroutineCount(), 1);

  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_2", sub_ptr));
  instr_2->SetOps(4);
  auto subroutine_1 = absl::make_unique<paragraph::Subroutine>(
      "subroutine_1", graph.get());
  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_3", sub_ptr));
  instr_3->SetOps(4);
  instr_2->AppendInnerSubroutine(std::move(subroutine_1));
  EXPECT_EQ(graph->SubroutineCount(), 2);
}

// Tests adding and checking Entry Subroutine
// Also tests how SubroutineCount() handle subroutines removal
TEST(Graph, RemovingSubroutine) {
  std::string graph_name = "graph";
  int64_t graph_processor = 7;
  auto graph = absl::make_unique<paragraph::Graph>(
      graph_name, graph_processor);

  auto subroutine = absl::make_unique<paragraph::Subroutine>(
      "subroutine", graph.get());
  auto subroutine_ptr = subroutine.get();
  graph->SetEntrySubroutine(std::move(subroutine));
  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_1", subroutine_ptr));
  instr_1->SetOps(4);
  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_2", subroutine_ptr));
  instr_2->SetOps(4);
  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_3", subroutine_ptr, true));
  instr_3->SetOps(4);

  EXPECT_EQ(graph->SubroutineCount(), 1);
  EXPECT_EQ(graph->InstructionCount(), 3);

  subroutine_ptr->RemoveInstruction(instr_1);
  EXPECT_EQ(graph->SubroutineCount(), 1);
  EXPECT_EQ(graph->InstructionCount(), 2);

  subroutine_ptr->RemoveInstruction(instr_2);
  EXPECT_EQ(graph->SubroutineCount(), 1);
  EXPECT_EQ(graph->InstructionCount(), 1);

  EXPECT_OK(subroutine_ptr->SetRootInstruction(nullptr));
  subroutine_ptr->RemoveInstruction(instr_3);
  EXPECT_EQ(graph->SubroutineCount(), 1);
  EXPECT_EQ(graph->InstructionCount(), 0);
}

// Tests instruction and subroutine counts in graph
TEST(Graph, CountSubroutinesAndInstructions) {
  std::string graph_name = "graph";
  int64_t graph_processor = 7;
  auto graph = absl::make_unique<paragraph::Graph>(
      graph_name, graph_processor);

  auto subroutine = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = subroutine.get();
  graph->SetEntrySubroutine(std::move(subroutine));
  EXPECT_EQ(graph->SubroutineCount(), 1);
  EXPECT_EQ(graph->InstructionCount(), 0);

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_1", sub_ptr));
  instr_1->SetOps(4);
  ASSERT_OK_AND_ASSIGN(auto instr_1a, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_1a", sub_ptr));
  instr_1a->SetOps(4);
  EXPECT_EQ(graph->SubroutineCount(), 1);
  EXPECT_EQ(graph->InstructionCount(), 2);

  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_2", sub_ptr));
  instr_2->SetOps(4);
  auto subroutine_1 = absl::make_unique<paragraph::Subroutine>(
      "subroutine_1", graph.get());
  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_3", subroutine_1.get()));
  instr_3->SetOps(4);
  instr_2->AppendInnerSubroutine(std::move(subroutine_1));
  EXPECT_EQ(graph->SubroutineCount(), 2);
  EXPECT_EQ(graph->InstructionCount(), 4);
}

// Tests InstructionsPostOrder() method
TEST(Graph, InstructionsPostOrder) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto subroutine = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "1", subroutine.get(), true));
  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kCall, "2", subroutine.get()));
  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "3", subroutine.get()));
  ASSERT_OK_AND_ASSIGN(auto instr_4, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "4", subroutine.get()));
  ASSERT_OK_AND_ASSIGN(auto instr_5, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "5", subroutine.get()));
  ASSERT_OK_AND_ASSIGN(auto instr_6, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "6", subroutine.get()));
  instr_1->AddOperand(instr_2);
  instr_1->AddOperand(instr_5);
  instr_2->AddOperand(instr_3);
  instr_2->AddOperand(instr_4);
  instr_5->AddOperand(instr_6);
  instr_1->AddOperand(instr_6);
  graph->SetEntrySubroutine(std::move(subroutine));

  auto sub_1 = absl::make_unique<paragraph::Subroutine>(
      "inner_subroutine_1", graph.get());
  ASSERT_OK_AND_ASSIGN(auto instr_7, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "7", sub_1.get(), true));
  ASSERT_OK_AND_ASSIGN(auto instr_8, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "8", sub_1.get()));
  ASSERT_OK_AND_ASSIGN(auto instr_9, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "9", sub_1.get()));
  instr_7->AddOperand(instr_8);
  instr_7->AddOperand(instr_9);
  auto sub_2 = absl::make_unique<paragraph::Subroutine>(
      "inner_subroutine_2", graph.get());
  ASSERT_OK_AND_ASSIGN(auto instr_10, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "10", sub_2.get(), true));
  ASSERT_OK_AND_ASSIGN(auto instr_11, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "11", sub_2.get()));
  ASSERT_OK_AND_ASSIGN(auto instr_12, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "12", sub_2.get()));
  instr_10->AddOperand(instr_11);
  instr_10->AddOperand(instr_12);
  instr_2->AppendInnerSubroutine(std::move(sub_1));
  instr_2->AppendInnerSubroutine(std::move(sub_2));

  std::vector<std::string> instr_names = {
    "3", "4", "8", "9", "7", "11", "12", "10", "2", "6", "5", "1"};
  auto instr_postorder = graph->InstructionsPostOrder();
  EXPECT_EQ(instr_postorder.size(), graph->InstructionCount());
  for (size_t i = 0; i < instr_postorder.size(); ++i) {
    EXPECT_EQ(instr_postorder.at(i)->GetName(), instr_names.at(i));
  }
}

// Tests PostOrderEnforcer() method
TEST(Graph, PostOrderEnforcer) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto subroutine = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "1", subroutine.get(), true));
  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kCall, "2", subroutine.get()));
  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "3", subroutine.get()));
  ASSERT_OK_AND_ASSIGN(auto instr_4, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "4", subroutine.get()));
  ASSERT_OK_AND_ASSIGN(auto instr_5, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "5", subroutine.get()));
  ASSERT_OK_AND_ASSIGN(auto instr_6, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "6", subroutine.get()));
  instr_1->AddOperand(instr_2);
  instr_1->AddOperand(instr_5);
  instr_2->AddOperand(instr_3);
  instr_2->AddOperand(instr_4);
  instr_5->AddOperand(instr_6);
  instr_1->AddOperand(instr_6);
  graph->SetEntrySubroutine(std::move(subroutine));

  auto sub_1 = absl::make_unique<paragraph::Subroutine>(
      "inner_subroutine_1", graph.get());
  ASSERT_OK_AND_ASSIGN(auto instr_7, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "7", sub_1.get(), true));
  ASSERT_OK_AND_ASSIGN(auto instr_8, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "8", sub_1.get()));
  ASSERT_OK_AND_ASSIGN(auto instr_9, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "9", sub_1.get()));
  instr_7->AddOperand(instr_8);
  instr_7->AddOperand(instr_9);
  auto sub_2 = absl::make_unique<paragraph::Subroutine>(
      "inner_subroutine_2", graph.get());
  ASSERT_OK_AND_ASSIGN(auto instr_10, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "10", sub_2.get(), true));
  ASSERT_OK_AND_ASSIGN(auto instr_11, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "11", sub_2.get()));
  ASSERT_OK_AND_ASSIGN(auto instr_12, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "12", sub_2.get()));
  instr_10->AddOperand(instr_11);
  instr_10->AddOperand(instr_12);
  instr_2->AppendInnerSubroutine(std::move(sub_1));
  instr_2->AppendInnerSubroutine(std::move(sub_2));

  graph->PostOrderEnforcer();
  EXPECT_FALSE(instr_4->AddOperand(instr_3));
  EXPECT_FALSE(instr_9->AddOperand(instr_8));
  EXPECT_FALSE(instr_6->AddOperand(instr_2));
  EXPECT_FALSE(instr_12->AddOperand(instr_11));
  EXPECT_TRUE(instr_8->AddOperand(instr_4));
  EXPECT_TRUE(instr_11->AddOperand(instr_7));
  EXPECT_TRUE(instr_2->AddOperand(instr_10));
}

// Tests ApplyCommunicationTags()
TEST(Util, ApplyCommunicationTags) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 2);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  sub_ptr->SetId(3);
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "first_instruction", sub_ptr));
  instr_1->SetOps(4);

  ASSERT_OK_AND_ASSIGN(auto allreduce, paragraph::Instruction::Create(
      paragraph::Opcode::kAllReduce, "all-reduce", sub_ptr));
  allreduce->SetBytesOut(48);
  allreduce->AddOperand(instr_1);
  paragraph::CommunicationGroup allreduce_group = {0, 1, 2};
  allreduce->AppendCommunicationGroup(allreduce_group);

  auto reduction_sub = absl::make_unique<paragraph::Subroutine>(
      "reduction_subroutine", graph.get());
  auto reduction_ptr = reduction_sub.get();
  ASSERT_OK_AND_ASSIGN(auto op1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "op1", reduction_ptr));
  op1->SetBytesOut(48);
  ASSERT_OK_AND_ASSIGN(auto op2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "op2", reduction_ptr));
  op2->SetBytesOut(48);
  ASSERT_OK_AND_ASSIGN(auto sum_op, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "sum", reduction_ptr, true));
  sum_op->SetOps(96);
  sum_op->AddOperand(op1);
  sum_op->AddOperand(op2);
  allreduce->AppendInnerSubroutine(std::move(reduction_sub));

  ASSERT_OK_AND_ASSIGN(auto barrier, paragraph::Instruction::Create(
      paragraph::Opcode::kBarrier, "barrier", sub_ptr));
  barrier->AppendCommunicationGroup(allreduce_group);
  barrier->AddOperand(instr_1);

  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "last_instruction", sub_ptr, true));
  instr_3->SetOps(4);
  instr_3->AddOperand(allreduce);
  instr_3->AddOperand(barrier);

  nlohmann::json translation_config = R"(
    {
      "collective": {
        "all-reduce": {
          "algorithm": "unidir-ring"
        },
        "barrier": {
          "algorithm": "centralized"
        }
      },
      "protocol": {
        "sendrecv": {
          "algorithm": "push"
        },
        "send": {
          "algorithm": "push"
        },
        "recv": {
          "algorithm": "push"
        }
      }
    }
  )"_json;

  // IndividualizeAndTranslate() calls ApplyCommunicationTags on each graph
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<paragraph::Graph>> graphs,
      paragraph::IndividualizeAndTranslate(
          graph.get(), translation_config));

  // Creates ID maps for each graph for correspondance checking,
  absl::flat_hash_map<
    int64_t, absl::flat_hash_map<
      int64_t, const paragraph::Instruction*>> id_maps;
  for (auto& graph : graphs) {
    id_maps[graph->GetProcessorId()] = graph->InstructionIdMap();
  }

  // This tracks the used tags found in the following testing
  absl::flat_hash_map<int64_t, absl::flat_hash_map<
    int64_t, absl::flat_hash_set<uint64_t>>> used_tags;  // [src][dst]->tags

  // Checks communication tags of the all-reduce instructions
  const char ar_fmt[] =
      "all-reduce_unidir-ring_reduce-scatter_unidir-ring_sendrecv_%d_%s%s";
  const std::vector<std::tuple<int64_t, int64_t>> ar_send_pairs = {
    std::make_tuple(0, 1),
    std::make_tuple(1, 2),
    std::make_tuple(2, 0)};
  for (int64_t phase = 1; phase <= 2; phase++) {
    for (const auto& [src, dst] : ar_send_pairs) {
      uint64_t comm_tag = graphs[dst]->FindByName(
          absl::StrFormat(ar_fmt, phase, "recv", "start")).at(0)
                          ->GetCommunicationTag();
      EXPECT_GT(comm_tag, 0);
      EXPECT_TRUE(used_tags[src][dst].insert(comm_tag).second);
      EXPECT_EQ(graphs[dst]->FindByName(
          absl::StrFormat(ar_fmt, phase, "recv", "done")).at(0)
                ->GetCommunicationTag(),
                comm_tag);
      EXPECT_EQ(graphs[src]->FindByName(
          absl::StrFormat(ar_fmt, phase, "send", "start")).at(0)
                ->GetCommunicationTag(),
                comm_tag);
      EXPECT_EQ(graphs[src]->FindByName(
          absl::StrFormat(ar_fmt, phase, "send", "done")).at(0)
                ->GetCommunicationTag(),
                comm_tag);
    }
  }

  // Checks communication tags of the barrier instructions
  const char bar_fmt[] = "barrier_centralized%s_%s_%s_%d_%s%s";
  const int64_t bar_coordinator = 0;
  const std::vector<int64_t> bar_waiters = {1, 2};
  for (int64_t bar_waiter : bar_waiters) {
    int64_t src = bar_waiter;
    int64_t dst = bar_coordinator;
    uint64_t comm_tag = graphs[bar_coordinator]->FindByName(
        absl::StrFormat(bar_fmt, "_coordinator", "recv", "from", bar_waiter,
                        "recv", "start")).at(0)->GetCommunicationTag();
    EXPECT_GT(comm_tag, 0);
    EXPECT_TRUE(used_tags[src][dst].insert(comm_tag).second);
    EXPECT_EQ(graphs[bar_coordinator]->FindByName(
        absl::StrFormat(bar_fmt, "_coordinator", "recv", "from", bar_waiter,
                        "recv", "done")).at(0)->GetCommunicationTag(),
              comm_tag);
    EXPECT_EQ(graphs[bar_waiter]->FindByName(
        absl::StrFormat(bar_fmt, "", "send", "to", bar_coordinator, "send",
                        "start")).at(0)->GetCommunicationTag(),
              comm_tag);
    EXPECT_EQ(graphs[bar_waiter]->FindByName(
        absl::StrFormat(bar_fmt, "", "send", "to", bar_coordinator, "send",
                        "done")).at(0)->GetCommunicationTag(),
              comm_tag);
  }
  for (int64_t bar_waiter : bar_waiters) {
    int64_t src = bar_coordinator;
    int64_t dst = bar_waiter;
    uint64_t comm_tag = graphs[bar_waiter]->FindByName(
        absl::StrFormat(bar_fmt, "", "recv", "from", bar_coordinator,
                        "recv", "start")).at(0)->GetCommunicationTag();
    EXPECT_GT(comm_tag, 0);
    EXPECT_TRUE(used_tags[src][dst].insert(comm_tag).second);
    EXPECT_EQ(graphs[bar_waiter]->FindByName(
        absl::StrFormat(bar_fmt, "", "recv", "from", bar_coordinator,
                        "recv", "done")).at(0)->GetCommunicationTag(),
              comm_tag);
    EXPECT_EQ(graphs[bar_coordinator]->FindByName(
        absl::StrFormat(bar_fmt, "_coordinator", "send", "to", bar_waiter,
                        "send", "start")).at(0)->GetCommunicationTag(),
              comm_tag);
    EXPECT_EQ(graphs[bar_coordinator]->FindByName(
        absl::StrFormat(bar_fmt, "_coordinator", "send", "to", bar_waiter,
                        "send", "done")).at(0)->GetCommunicationTag(),
              comm_tag);
  }
}

// Tests InstructionIdMap() and FindByName() methods
TEST(Graph, InstructionIdMapAndFindByName) {
  auto graph = absl::make_unique<paragraph::Graph>(
      "test_graph", 7);

  auto subroutine = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = subroutine.get();
  graph->SetEntrySubroutine(std::move(subroutine));
  EXPECT_EQ(graph->SubroutineCount(), 1);
  EXPECT_EQ(graph->InstructionCount(), 0);

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_1", sub_ptr));
  instr_1->SetOps(4);
  ASSERT_OK_AND_ASSIGN(auto instr_1a, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_1a", sub_ptr));
  instr_1a->SetOps(4);
  EXPECT_EQ(graph->SubroutineCount(), 1);
  EXPECT_EQ(graph->InstructionCount(), 2);

  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_2", sub_ptr));
  instr_2->SetOps(4);
  auto subroutine_1 = absl::make_unique<paragraph::Subroutine>(
      "subroutine_1", graph.get());
  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_3", subroutine_1.get()));
  instr_3->SetOps(4);
  instr_2->AppendInnerSubroutine(std::move(subroutine_1));

  absl::flat_hash_map<int64_t, const paragraph::Instruction*> id_map =
      graph->InstructionIdMap();
  EXPECT_EQ(id_map.size(), 4);
  EXPECT_EQ(id_map[1]->GetName(), "comp_1");
  EXPECT_EQ(graph->FindByName("comp_1").size(), 1);
  EXPECT_EQ(graph->FindByName("comp_1")[0]->GetId(), 1);
  EXPECT_EQ(id_map[2]->GetName(), "comp_1a");
  EXPECT_EQ(graph->FindByName("comp_1a").size(), 1);
  EXPECT_EQ(graph->FindByName("comp_1a")[0]->GetId(), 2);
  EXPECT_EQ(id_map[3]->GetName(), "comp_2");
  EXPECT_EQ(graph->FindByName("comp_2").size(), 1);
  EXPECT_EQ(graph->FindByName("comp_2")[0]->GetId(), 3);
  EXPECT_EQ(id_map[4]->GetName(), "comp_3");
  EXPECT_EQ(graph->FindByName("comp_3").size(), 1);
  EXPECT_EQ(graph->FindByName("comp_3")[0]->GetId(), 4);
  EXPECT_EQ(graph->FindByName("comp_100").size(), 0);
}

// Tests GetCommunicationSet() method
TEST(Graph, CommunicationSet) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 42);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kSend, "test_send", sub_ptr));
  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kAllReduce, "test_allreduce", sub_ptr));
  EXPECT_EQ(graph->GetCommunicationSet().size(), 0);
  EXPECT_EQ(graph->CommunicationSetHasProcessor(13), false);

  paragraph::CommunicationGroup group;
  group.push_back(13);
  instr_1->AppendCommunicationGroup(group);
  EXPECT_EQ(graph->GetCommunicationSet().size(), 1);
  EXPECT_EQ(graph->CommunicationSetHasProcessor(13), true);

  paragraph::CommunicationGroup group_1 = {1, 7};
  instr_2->AppendCommunicationGroup(group_1);
  paragraph::CommunicationGroup group_2 = {3, 13};
  instr_2->AppendCommunicationGroup(group_2);
  EXPECT_EQ(graph->GetCommunicationSet().size(), 4);
  EXPECT_EQ(graph->CommunicationSetHasProcessor(1), true);
  EXPECT_EQ(graph->CommunicationSetHasProcessor(3), true);
  EXPECT_EQ(graph->CommunicationSetHasProcessor(7), true);
}

// Tests Graph::HasConsecutiveNaturalProcessorIds() method
TEST(Graph, HasConsecutiveNaturalProcessorIds) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 0);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kAllReduce, "test_allreduce", sub_ptr));
  paragraph::CommunicationGroup group = {0, 1, 2, 3};
  instr_1->AppendCommunicationGroup(group);
  EXPECT_EQ(graph->Graph::HasConsecutiveNaturalProcessorIds(), true);

  paragraph::CommunicationGroup group_2 = {5, 6, 7, 8};
  instr_1->AppendCommunicationGroup(group_2);
  EXPECT_EQ(graph->Graph::HasConsecutiveNaturalProcessorIds(), false);
}

paragraph::GraphProto create_test_proto() {
  paragraph::GraphProto test_graph_proto;
  std::string test_graph_str =
      R"proto(
  name: "test_graph"
  processor_id: 1
  entry_subroutine {
    name: "test_subroutine"
    subroutine_root_id: 8
    execution_probability: 1
    execution_count: 1
    instructions {
      name: "compute_pred1"
      opcode: "infeed"
      instruction_id: 1
      bytes_out: 36.5
    }
    instructions {
      name: "reduction_operand1"
      opcode: "delay"
      instruction_id: 2
      ops: 128
      operand_ids: 1
    }
    instructions {
      name: "reduction"
      opcode: "all-reduce"
      instruction_id: 3
      communication_groups {
        group_ids: 1
        group_ids: 7
        group_ids: 42
      }
      operand_ids: 2
    }
    instructions {
      name: "test"
      opcode: "while"
      instruction_id: 4
      inner_subroutines {
        name: "body_subroutine"
        subroutine_root_id: 5
        execution_probability: 1
        execution_count: 1
        instructions {
          name: "body_compute"
          opcode: "delay"
          instruction_id: 5
          transcendentals: 111
        }
      }
      inner_subroutines {
        name: "cond_subroutine"
        subroutine_root_id: 7
        execution_probability: 1
        execution_count: 1
        instructions {
          name: "cond_call"
          opcode: "call"
          instruction_id: 7
          inner_subroutines {
            name: "call_subroutine"
            subroutine_root_id: 6
            execution_probability: 1
            execution_count: 1
            instructions {
              name: "call_func"
              opcode: "delay"
              instruction_id: 6
              seconds: 0.001
            }
          }
        }
      }
    }
    instructions {
      name: "send"
      opcode: "send"
      instruction_id: 8
      bytes_in: 8
      communication_groups {
        group_ids: 42
      }
    }
  }
      )proto";
  google::protobuf::TextFormat::ParseFromString(test_graph_str,
                                                &test_graph_proto);
  return test_graph_proto;
}

// Tests proto serialization - ToProto() method
TEST(Graph, StoreToProto) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kInfeed, "compute_pred1", sub_ptr));
  instr_1->SetBytesOut(36.5);

  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "reduction_operand1", sub_ptr));
  instr_2->SetOps(128.);
  instr_2->AddOperand(instr_1);

  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kAllReduce, "reduction", sub_ptr));
  paragraph::CommunicationGroup group_1 = {1, 7, 42};
  instr_3->AppendCommunicationGroup(group_1);
  instr_3->AddOperand(instr_2);


  ASSERT_OK_AND_ASSIGN(auto while_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kWhile, "test", sub_ptr));
  while_instr->SetId(4);
  auto body_sub = absl::make_unique<paragraph::Subroutine>(
      "body_subroutine", graph.get());
  auto body_sub_ptr = body_sub.get();
  ASSERT_OK_AND_ASSIGN(auto body_compute, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "body_compute", body_sub_ptr, true));
  body_compute->SetId(5);
  body_compute->SetTranscendentals(111.);
  while_instr->AppendInnerSubroutine(std::move(body_sub));

  auto call_sub = absl::make_unique<paragraph::Subroutine>(
      "call_subroutine", graph.get());
  auto call_sub_ptr = call_sub.get();
  ASSERT_OK_AND_ASSIGN(auto call_func, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "call_func", call_sub_ptr, true));
  call_func->SetId(6);
  call_func->SetSeconds(0.001);

  auto cond_sub = absl::make_unique<paragraph::Subroutine>(
      "cond_subroutine", graph.get());
  auto cond_sub_ptr = cond_sub.get();
  ASSERT_OK_AND_ASSIGN(auto cond_call, paragraph::Instruction::Create(
      paragraph::Opcode::kCall, "cond_call", cond_sub_ptr, true));
  cond_call->SetId(7);
  cond_call->AppendInnerSubroutine(std::move(call_sub));
  while_instr->AppendInnerSubroutine(std::move(cond_sub));

  ASSERT_OK_AND_ASSIGN(auto send_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kSend, "send", sub_ptr, true));
  paragraph::CommunicationGroup send_group = {42};
  send_instr->SetBytesIn(8.);
  send_instr->AppendCommunicationGroup(send_group);
  send_instr->SetId(8);

  google::protobuf::util::MessageDifferencer diff;
  EXPECT_TRUE(diff.Compare(graph->ToProto().value(), create_test_proto()));
}

// Tests proto serialization - CreateFromProto() method
TEST(Graph, LoadFromProto) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<paragraph::Graph> graph,
      paragraph::Graph::CreateFromProto(create_test_proto()));
  EXPECT_EQ(graph->GetName(), "test_graph");
  EXPECT_EQ(graph->GetProcessorId(), 1);
  EXPECT_EQ(graph->GetEntrySubroutine()->GetName(), "test_subroutine");
  EXPECT_EQ(graph->InstructionCount(), 8);
  EXPECT_EQ(graph->SubroutineCount(), 4);
}

// Tests graph cloning
TEST(Graph, Clone) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<paragraph::Graph> graph,
      paragraph::Graph::CreateFromProto(create_test_proto()));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<paragraph::Graph> graph_clone,
      graph->Clone("_copy"));
  EXPECT_EQ(graph_clone->GetName(), "test_graph_copy");
  EXPECT_EQ(graph_clone->GetEntrySubroutine()->GetName(),
            "test_subroutine");
  EXPECT_EQ(graph_clone->InstructionCount(), 8);
  EXPECT_EQ(graph_clone->SubroutineCount(), 4);
  google::protobuf::util::MessageDifferencer diff;
  ASSERT_OK_AND_ASSIGN(auto graph_proto, graph->ToProto());
  diff.IgnoreField(graph_proto.GetDescriptor()->FindFieldByName("name"));
  EXPECT_TRUE(diff.Compare(graph_proto,
                           graph_clone->ToProto().value()));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<paragraph::Graph> graph_new,
      graph->Clone("_new_id", true));
  EXPECT_EQ(graph_new->GetName(), "test_graph_new_id");
  EXPECT_EQ(graph_new->GetEntrySubroutine()->GetName(),
            "test_subroutine");
  EXPECT_EQ(graph_new->InstructionCount(), 8);
  EXPECT_EQ(graph_new->SubroutineCount(), 4);
}

std::string get_testfile_name(const std::string& basename) {
  std::string f = std::getenv("TEST_TMPDIR");
  if (f.back() != '/') {
    f += '/';
  }
  f += basename;
  return f;
}

// Tests graph writing to and reading from file
TEST(Graph, FileIO) {
  for (const std::string& ext : {".pb", ".textproto"}) {
    std::filesystem::remove(get_testfile_name("graph_test" + ext));
    EXPECT_FALSE(std::filesystem::exists(get_testfile_name(
        "graph_test" + ext)));
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<paragraph::Graph> graph_1,
        paragraph::Graph::CreateFromProto(create_test_proto()));
    EXPECT_OK(graph_1->WriteToFile(get_testfile_name("graph_test" + ext)));
    EXPECT_TRUE(std::filesystem::exists(get_testfile_name("graph_test" + ext)));

    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<paragraph::Graph> graph_2,
        paragraph::Graph::ReadFromFile(get_testfile_name("graph_test" + ext)));
    google::protobuf::util::MessageDifferencer diff;
    EXPECT_TRUE(diff.Compare(graph_2->ToProto().value(), create_test_proto()));
  }
}

// Tests graph individulization
TEST(Graph, Individualize) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", -1);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kAllReduce, "reduction", sub_ptr));
  paragraph::CommunicationGroup group_1 = {1, 7, 13, 42};
  paragraph::CommunicationGroup group_2 = {2, 8, 14, 43};
  instr_1->AppendCommunicationGroup(group_1);
  instr_1->AppendCommunicationGroup(group_2);

  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kSendRecv, "sendrecv", sub_ptr));
  paragraph::CommunicationGroup group_3 = {42, 1, 2};
  paragraph::CommunicationGroup group_4 = {2, 8, 43};
  instr_2->AppendCommunicationGroup(group_3);
  instr_2->AppendCommunicationGroup(group_4);

  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kSend, "send", sub_ptr));
  paragraph::CommunicationGroup group_5 = {1, 2};
  paragraph::CommunicationGroup group_6 = {8, 43};
  instr_3->AppendCommunicationGroup(group_5);
  instr_3->AppendCommunicationGroup(group_6);

  ASSERT_OK_AND_ASSIGN(auto instr_4, paragraph::Instruction::Create(
      paragraph::Opcode::kRecv, "recv", sub_ptr));
  paragraph::CommunicationGroup group_7 = {42, 1};
  paragraph::CommunicationGroup group_8 = {2, 8};
  instr_4->AppendCommunicationGroup(group_7);
  instr_4->AppendCommunicationGroup(group_8);

  ASSERT_OK_AND_ASSIGN(auto instr_5, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "compute", sub_ptr, true));
  instr_5->AddOperand(instr_4);

  EXPECT_OK(graph->ValidateComposite());
  ASSERT_OK_AND_ASSIGN(auto graph_1, graph->Individualize(1));
  EXPECT_OK(graph_1->ValidateIndividualized());

  ASSERT_OK_AND_ASSIGN(auto graph_2, graph->Individualize(7));
  EXPECT_OK(graph_2->ValidateIndividualized());

  ASSERT_OK_AND_ASSIGN(auto graph_3, graph->Individualize(8));
  EXPECT_OK(graph_3->ValidateIndividualized());

  ASSERT_OK_AND_ASSIGN(auto graph_4, graph->Individualize(5));
  EXPECT_OK(graph_4->ValidateIndividualized());
}

// Tests ValidateComposite() and ValidateIndividualized() methods
TEST(Graph, Validate) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto graph_ptr = graph.get();
  EXPECT_EQ(graph_ptr->ValidateComposite(),
            absl::InternalError("Graph should have an entry subroutine."));

  auto graph_2 = absl::make_unique<paragraph::Graph>("test_graph", 2);
  auto graph_2_ptr = graph_2.get();
  auto alien = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph_2.get());
  graph->SetEntrySubroutine(std::move(alien));
  EXPECT_EQ(graph_ptr->ValidateComposite(),
            absl::InternalError("Subroutine test_subroutine points to "
                                "a different graph."));

  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph_2_ptr);
  auto sub_ptr = sub.get();
  graph_2->SetEntrySubroutine(std::move(sub));
  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "test", sub_ptr, true));
  instr_1->SetOps(4);
  EXPECT_OK(graph_2_ptr->ValidateComposite());
  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "test_2", sub_ptr));
  instr_2->SetOps(4);
  instr_2->SetId(1);
  EXPECT_EQ(graph_2_ptr->ValidateComposite(),
            absl::InternalError("Instruction test_2 ID = 1 is not unique."));

  instr_2->SetId(2);
  EXPECT_OK(graph_2_ptr->ValidateComposite());

  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "test", sub_ptr));
  instr_3->SetOps(4);
  EXPECT_EQ(graph_2_ptr->ValidateComposite(), absl::OkStatus());

  auto graph_3 = absl::make_unique<paragraph::Graph>("test_graph", 3);
  auto graph_3_ptr = graph_3.get();
  auto while_sub = absl::make_unique<paragraph::Subroutine>(
      "while_subroutine", graph_3_ptr);
  auto while_sub_ptr = while_sub.get();
  graph_3->SetEntrySubroutine(std::move(while_sub));
  ASSERT_OK_AND_ASSIGN(auto while_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kWhile, "test", while_sub_ptr));
  auto cond_sub = absl::make_unique<paragraph::Subroutine>(
      "cond_subroutine", graph_3_ptr);
  auto cond_sub_ptr = cond_sub.get();
  ASSERT_OK_AND_ASSIGN(auto cond_compute, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "cond_compute", cond_sub_ptr, true));
  cond_compute->SetOps(4);
  while_instr->AppendInnerSubroutine(std::move(cond_sub));
  EXPECT_OK(while_sub_ptr->SetRootInstruction(while_instr));
  EXPECT_EQ(graph_3_ptr->ValidateComposite(),
            absl::InternalError("While instruction test should have "
                                "exactly 2 subroutines."));
}
