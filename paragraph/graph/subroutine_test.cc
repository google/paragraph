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
#include "paragraph/graph/subroutine.h"

#include <list>
#include <string>
#include <utility>

#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "paragraph/graph/graph.h"
#include "gtest/gtest.h"
#include "paragraph/shim/test_macros.h"

// Tests construction, name and root instruction return
// Test constructs several subroutines with various names and root
// instructions, checks that names and root instructions are
// the ones that were passed to constructor
TEST(Subroutine, Construction) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  std::string subroutine_name = "test_subroutine";
  auto subroutine = absl::make_unique<paragraph::Subroutine>(
      subroutine_name, graph.get());
  EXPECT_EQ(subroutine_name, subroutine->GetName());
  EXPECT_EQ(subroutine->GetRootInstruction(), nullptr);

  std::string subroutine_1_name = "Subroutine_1";
  auto subroutine_1 = absl::make_unique<paragraph::Subroutine>(
      subroutine_1_name, graph.get());
  paragraph::Opcode instr_1_opcode = paragraph::Opcode::kDelay;
  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      instr_1_opcode, "Compute_1", subroutine_1.get()));
  instr_1->SetId(1);
  EXPECT_EQ(subroutine_1_name, subroutine_1->GetName());
  EXPECT_EQ(subroutine_1->GetRootInstruction(), nullptr);

  std::string subroutine_2_name = "Subroutine_2";
  auto subroutine_2 = absl::make_unique<paragraph::Subroutine>(
      subroutine_2_name, graph.get());
  paragraph::Opcode instr_2_opcode = paragraph::Opcode::kAllReduce;
  ASSERT_OK_AND_ASSIGN(instr_1,  paragraph::Instruction::Create(
      instr_2_opcode, "AllReduce", subroutine_2.get()));
  EXPECT_EQ(subroutine_2_name, subroutine_2->GetName());
  EXPECT_EQ(subroutine_2->GetRootInstruction(), nullptr);
}

// Tests GetId(), SetId(), ClearID(), and HasDefaultId() members
// Tests that default unique_id is -1, ids can be set/cleared
TEST(Subroutine, IdHandling) {
  EXPECT_EQ(paragraph::Subroutine::kDefaultId, -1);

  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  std::string subroutine_name = "test_subroutine";
  auto subroutine = absl::make_unique<paragraph::Subroutine>(
      subroutine_name, graph.get());
  std::string instr_1_name = "Compute_1";
  paragraph::Opcode instr_1_opcode = paragraph::Opcode::kDelay;
  auto instr_1 = paragraph::Instruction::Create(
      instr_1_opcode, instr_1_name, subroutine.get());
  EXPECT_EQ(subroutine->GetId(), -1);
  EXPECT_EQ(subroutine->HasDefaultId(), true);

  subroutine->SetId(42);
  EXPECT_EQ(subroutine->GetId(), 42);
  EXPECT_EQ(subroutine->HasDefaultId(), false);

  subroutine->SetId(777);
  EXPECT_EQ(subroutine->GetId(), 777);
  EXPECT_EQ(subroutine->HasDefaultId(), false);

  subroutine->ClearId();
  EXPECT_EQ(subroutine->GetId(), -1);
  EXPECT_EQ(subroutine->HasDefaultId(), true);
}

// Tests adding instructions to a subroutine,
TEST(Subroutine, AddingInstruction) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto sub_unique = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto subroutine = sub_unique.get();

  graph->SetEntrySubroutine(std::move(sub_unique));
  EXPECT_EQ(subroutine->InstructionCount(), 0);

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_middle", subroutine));
  instr_1->SetOps(4);
  EXPECT_EQ(subroutine->InstructionCount(), 1);

  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp1_beginning", subroutine));
  instr_2->SetOps(4);
  EXPECT_EQ(subroutine->InstructionCount(), 2);

  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp1_end", subroutine));
  instr_3->SetOps(4);
  EXPECT_EQ(subroutine->InstructionCount(), 3);
}

// Tests EmbeddedSubroutines() method
TEST(Subroutine, EmbeddedSubroutines) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto subroutine = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_1", subroutine.get()));
  instr_1->SetOps(4);
  EXPECT_TRUE(subroutine->MakeEmbeddedSubroutinesVector().empty());

  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_2", subroutine.get()));
  instr_2->SetOps(4);
  auto subroutine_1 = absl::make_unique<paragraph::Subroutine>(
      "subroutine_1", graph.get());
  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_3", subroutine_1.get()));
  instr_3->SetOps(4);
  instr_2->AppendInnerSubroutine(std::move(subroutine_1));
  EXPECT_EQ(subroutine->MakeEmbeddedSubroutinesVector().size(), 1);

  ASSERT_OK_AND_ASSIGN(auto instr_4, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_4", subroutine.get()));
  instr_4->SetOps(4);
  auto subroutine_2 = absl::make_unique<paragraph::Subroutine>(
      "subroutine_2", graph.get());
  ASSERT_OK_AND_ASSIGN(auto instr_5, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_5", subroutine_2.get()));
  instr_5->SetOps(4);
  instr_4->AppendInnerSubroutine(std::move(subroutine_2));
  auto subroutine_3 = absl::make_unique<paragraph::Subroutine>(
      "subroutine_3", graph.get());
  ASSERT_OK_AND_ASSIGN(auto instr_6, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_6", subroutine_3.get()));
  instr_6->SetOps(4);
  instr_5->AppendInnerSubroutine(std::move(subroutine_3));
  EXPECT_EQ(subroutine->MakeEmbeddedSubroutinesVector().size(), 3);
}

// Tests InstructionsPostOrder() method
TEST(Subroutine, InstructionsPostOrder) {
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
  auto instr_postorder = graph->GetEntrySubroutine()->InstructionsPostOrder();
  EXPECT_EQ(instr_postorder.size(), graph->InstructionCount());
  for (size_t i = 0; i < instr_postorder.size(); ++i) {
    EXPECT_EQ(instr_postorder.at(i)->GetName(), instr_names.at(i));
  }

  instr_names = {
    "3", "4", "2", "6", "5", "1"};
  instr_postorder = graph->GetEntrySubroutine()->InstructionsPostOrder(
      true);
  EXPECT_EQ(instr_postorder.size(),
            graph->GetEntrySubroutine()->Instructions().size());
  for (size_t i = 0; i < instr_postorder.size(); ++i) {
    EXPECT_EQ(instr_postorder.at(i)->GetName(), instr_names.at(i));
  }
}

// Tests that SetRootInstruction updates root of the subroutine
TEST(Subroutine, RootInstructionChange) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  std::string subroutine_name = "Subroutine_1";
  auto subroutine = absl::make_unique<paragraph::Subroutine>(
      subroutine_name, graph.get());
  std::string instr_1_name = "Compute_1";
  paragraph::Opcode instr_1_opcode = paragraph::Opcode::kDelay;
  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      instr_1_opcode, instr_1_name, subroutine.get()));
  EXPECT_EQ(nullptr, subroutine->GetRootInstruction());

  EXPECT_OK(subroutine->SetRootInstruction(instr_1));
  EXPECT_EQ(instr_1, subroutine->GetRootInstruction());

  std::string instr_2_name = "AllReduce";
  paragraph::Opcode instr_2_opcode = paragraph::Opcode::kAllReduce;
  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      instr_2_opcode, instr_2_name, subroutine.get()));
  EXPECT_OK(subroutine->SetRootInstruction(instr_2));
  EXPECT_EQ(instr_2, subroutine->GetRootInstruction());
}

// Tests removing instructions from diffeernt positions in a subroutine,
// Tests how RemoveInstruction handles instruction removing and how it affects
// InstructionCount() as well as iterators Instructions() and InstructionIt()
TEST(Subroutine, RemovingInstruction) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto subroutine = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_beginning", subroutine.get()));
  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_pre_middle", subroutine.get()));
  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_middle", subroutine.get()));
  ASSERT_OK_AND_ASSIGN(auto instr_4, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_end", subroutine.get()));

  subroutine->RemoveInstruction(instr_2);
  auto instructions_it_0 = subroutine->Instructions().begin();
  auto instructions_it_1 = subroutine->Instructions().begin();
  std::advance(instructions_it_1, 1);
  auto instructions_it_2 = subroutine->Instructions().begin();
  std::advance(instructions_it_2, 2);
  auto instr_1_it = subroutine->InstructionIterator(instr_1);
  auto instr_3_it = subroutine->InstructionIterator(instr_3);
  auto instr_4_it = subroutine->InstructionIterator(instr_4);
  EXPECT_EQ(subroutine->InstructionCount(), 3);
  EXPECT_EQ(instructions_it_0, instr_1_it);
  EXPECT_EQ(instructions_it_1, instr_3_it);
  EXPECT_EQ(instructions_it_2, instr_4_it);
  EXPECT_EQ((*instr_1_it).get(), instr_1);
  EXPECT_EQ((*instr_3_it).get(), instr_3);
  EXPECT_EQ((*instr_4_it).get(), instr_4);
  EXPECT_EQ(instr_1_it, subroutine->Instructions().begin());

  subroutine->RemoveInstruction(instr_4);
  instructions_it_0 = subroutine->Instructions().begin();
  instructions_it_1 = subroutine->Instructions().begin();
  std::advance(instructions_it_1, 1);
  instr_1_it = subroutine->InstructionIterator(instr_1);
  instr_3_it = subroutine->InstructionIterator(instr_3);
  EXPECT_EQ(subroutine->InstructionCount(), 2);
  EXPECT_EQ(instructions_it_0, instr_1_it);
  EXPECT_EQ(instructions_it_1, instr_3_it);
  EXPECT_EQ((*instr_1_it).get(), instr_1);
  EXPECT_EQ((*instr_3_it).get(), instr_3);
  EXPECT_EQ(instr_1_it, subroutine->Instructions().begin());

  subroutine->RemoveInstruction(instr_1);
  instructions_it_0 = subroutine->Instructions().begin();
  instr_3_it = subroutine->InstructionIterator(instr_3);
  EXPECT_EQ(subroutine->InstructionCount(), 1);
  EXPECT_EQ(instructions_it_0, instr_3_it);
  EXPECT_EQ((*instr_3_it).get(), instr_3);
}

// Tests ReplaceInstructionWithInstructionList() member
TEST(Subroutine, ReplaceInstructionWithInstructionList) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto subroutine = absl::make_unique<paragraph::Subroutine>(
      "subroutine", graph.get());
  ASSERT_OK_AND_ASSIGN(auto first_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "first", subroutine.get()));
  auto first_instr_it = subroutine->InstructionIterator(first_instr);
  ASSERT_OK_AND_ASSIGN(auto target_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kAllReduce, "target", subroutine.get()));
  auto target_instr_it = subroutine->InstructionIterator(target_instr);
  ASSERT_OK_AND_ASSIGN(auto last_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "last", subroutine.get(), true));
  auto last_instr_it = subroutine->InstructionIterator(last_instr);
  EXPECT_EQ(subroutine->InstructionCount(), 3);
  EXPECT_EQ((*first_instr_it).get(), first_instr);
  EXPECT_EQ((*target_instr_it).get(), target_instr);
  EXPECT_EQ((*last_instr_it).get(), last_instr);

  auto dummy = absl::make_unique<paragraph::Subroutine>(
      "dummy", graph.get());
  std::list<std::unique_ptr<paragraph::Instruction>> new_instr;
  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp1", dummy.get()));
  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp2", dummy.get()));
  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp3", dummy.get()));
  for (auto& instr : dummy->Instructions()) {
    new_instr.push_back(std::move(instr));
  }

  EXPECT_OK(subroutine->ReplaceInstructionWithInstructionList(
      target_instr, &new_instr));
  auto instr_1_it = subroutine->InstructionIterator(instr_1);
  auto instr_2_it = subroutine->InstructionIterator(instr_2);
  auto instr_3_it = subroutine->InstructionIterator(instr_3);
  EXPECT_EQ(subroutine->InstructionCount(), 5);
  EXPECT_EQ((*first_instr_it).get(), first_instr);
  EXPECT_EQ((*instr_1_it).get(), instr_1);
  EXPECT_EQ((*instr_2_it).get(), instr_2);
  EXPECT_EQ((*instr_3_it).get(), instr_3);
  EXPECT_EQ((*last_instr_it).get(), last_instr);

  auto dummy2 = absl::make_unique<paragraph::Subroutine>(
      "dummy2", graph.get());
  std::list<std::unique_ptr<paragraph::Instruction>> new_beginning;
  ASSERT_OK_AND_ASSIGN(auto instr_0, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp0", dummy2.get()));
  for (auto& instr : dummy2->Instructions()) {
    new_beginning.push_back(std::move(instr));
  }
  EXPECT_OK(subroutine->ReplaceInstructionWithInstructionList(
      first_instr, &new_beginning));
  auto instr_0_it = subroutine->InstructionIterator(instr_0);
  EXPECT_EQ(subroutine->InstructionCount(), 5);
  EXPECT_EQ((*instr_0_it).get(), instr_0);
  EXPECT_EQ((*instr_1_it).get(), instr_1);
  EXPECT_EQ((*instr_2_it).get(), instr_2);
  EXPECT_EQ((*instr_3_it).get(), instr_3);
  EXPECT_EQ((*last_instr_it).get(), last_instr);

  auto dummy3 = absl::make_unique<paragraph::Subroutine>(
      "dummy3", graph.get());
  std::list<std::unique_ptr<paragraph::Instruction>> new_ending;
  ASSERT_OK_AND_ASSIGN(auto instr_4, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp4", dummy3.get()));
  ASSERT_OK_AND_ASSIGN(auto instr_5, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp5", dummy3.get()));
  for (auto& instr : dummy3->Instructions()) {
    new_ending.push_back(std::move(instr));
  }
  EXPECT_EQ(subroutine->GetRootInstruction(), last_instr);
  EXPECT_OK(subroutine->ReplaceInstructionWithInstructionList(
      last_instr, &new_ending));
  auto instr_4_it = subroutine->InstructionIterator(instr_4);
  auto instr_5_it = subroutine->InstructionIterator(instr_5);
  EXPECT_EQ(subroutine->InstructionCount(), 6);
  EXPECT_EQ((*instr_0_it).get(), instr_0);
  EXPECT_EQ((*instr_1_it).get(), instr_1);
  EXPECT_EQ((*instr_2_it).get(), instr_2);
  EXPECT_EQ((*instr_3_it).get(), instr_3);
  EXPECT_EQ((*instr_4_it).get(), instr_4);
  EXPECT_EQ((*instr_5_it).get(), instr_5);
  EXPECT_EQ(subroutine->GetRootInstruction(), instr_5);
}

// Tests st/get calling instructions members
TEST(Subroutine, CallingInstruction) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto subroutine = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  EXPECT_EQ(subroutine->GetCallingInstruction(), nullptr);

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_1", subroutine.get(), true));
  subroutine->SetCallingInstruction(instr_1);
  EXPECT_EQ(subroutine->GetCallingInstruction(), instr_1);

  subroutine->SetCallingInstruction(nullptr);
  EXPECT_EQ(subroutine->GetCallingInstruction(), nullptr);
}

// Tests that RemoveInstruction also removes instructions from
// InnerSubroutines of instructions.
TEST(Subroutine, NestedRemoveInstruction) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto main_subroutine = absl::make_unique<paragraph::Subroutine>(
      "main_subroutine", graph.get());
  ASSERT_OK_AND_ASSIGN(auto main_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "main_instr", main_subroutine.get()));

  auto inner_subroutine = absl::make_unique<paragraph::Subroutine>(
      "inner_subroutine", graph.get());
  auto inner_subroutine_ptr = inner_subroutine.get();
  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_1", inner_subroutine_ptr));
  instr_1->SetOps(4);
  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_2", inner_subroutine_ptr));
  instr_2->SetOps(4);
  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp_3", inner_subroutine_ptr, true));
  instr_3->SetOps(4);
  main_instr->AppendInnerSubroutine(std::move(inner_subroutine));

  EXPECT_EQ(main_subroutine->InstructionCount(), 1);
  EXPECT_EQ(inner_subroutine_ptr->InstructionCount(), 3);
  main_subroutine->RemoveInstruction(main_instr);
  EXPECT_EQ(main_subroutine->InstructionCount(), 0);
  EXPECT_EQ(inner_subroutine_ptr->InstructionCount(), 0);
}

// Tests ScalePerformance() method
TEST(Subroutine, ScalePerformance) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto subroutine_1 = absl::make_unique<paragraph::Subroutine>(
      "subroutine_1", graph.get());
  auto subroutine_1_ptr = subroutine_1.get();
  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp1", subroutine_1_ptr));
  instr_1->SetBytesOut(12);
  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp2", subroutine_1_ptr, true));
  instr_2->SetSeconds(0.3);
  instr_2->SetId(2);
  subroutine_1->ScalePerformance(0.5);
  EXPECT_EQ(instr_1->GetBytesOut(), 6);
  EXPECT_FLOAT_EQ(instr_2->GetSeconds(), 0.15);
}

// Tests getting/setting parent subroutine
TEST(Subroutine, Graph) {
  auto subroutine = absl::make_unique<paragraph::Subroutine>(
      "test_Subroutine", nullptr);
  auto instr_1 = paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp1", subroutine.get());
  EXPECT_EQ(subroutine->GetGraph(), nullptr);

  std::string graph_name = "graph";
  int64_t graph_processor = 7;
  auto graph = absl::make_unique<paragraph::Graph>(
      graph_name, graph_processor);
  auto graph_ptr = graph.get();
  EXPECT_EQ(subroutine->GetGraph(), nullptr);
  subroutine->SetGraph(graph_ptr);
  EXPECT_EQ(subroutine->GetGraph(), graph_ptr);
}

// Tests proto serialization - ToProto() method
TEST(Subroutine, StoreToProto) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kAllReduce, "reduction", sub_ptr));
  paragraph::CommunicationGroup group_1 = {1, 7, 42};
  instr_1->AppendCommunicationGroup(group_1);
  instr_1->SetId(3);

  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "reduction_operand1", sub_ptr));
  instr_2->SetId(2);
  instr_2->SetOps(128.);
  instr_1->AddOperand(instr_2);

  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kInfeed, "compute_pred1", sub_ptr));
  instr_3->SetId(1);
  instr_3->SetBytesOut(36.5);
  instr_2->AddOperand(instr_3);

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

  paragraph::SubroutineProto body_sub_proto;
  std::string body_sub_str =
      R"proto(
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
      )proto";
  google::protobuf::TextFormat::ParseFromString(body_sub_str, &body_sub_proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      body_sub_ptr->ToProto().value(), body_sub_proto));

  paragraph::SubroutineProto call_sub_proto;
  std::string call_sub_str =
      R"proto(
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
      )proto";
  google::protobuf::TextFormat::ParseFromString(call_sub_str, &call_sub_proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      call_sub_ptr->ToProto().value(), call_sub_proto));

  paragraph::SubroutineProto cond_sub_proto;
  std::string cond_sub_str =
      R"proto(
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
      )proto";
  google::protobuf::TextFormat::ParseFromString(cond_sub_str, &cond_sub_proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      cond_sub_ptr->ToProto().value(), cond_sub_proto));

  paragraph::SubroutineProto sub_proto;
  std::string sub_str =
      R"proto(
        name: "test_subroutine"
        subroutine_root_id: 8
        execution_probability: 1
        execution_count: 1
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
          name: "reduction_operand1"
          opcode: "delay"
          instruction_id: 2
          ops: 128
          operand_ids: 1
        }
        instructions {
          name: "compute_pred1"
          opcode: "infeed"
          instruction_id: 1
          bytes_out: 36.5
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
      )proto";
  google::protobuf::TextFormat::ParseFromString(sub_str, &sub_proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      sub_ptr->ToProto().value(), sub_proto));
}

// Tests proto serialization - CreateFromProto() method
TEST(Subroutine, LoadFromProto) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  paragraph::SubroutineProto subroutine_proto;
  std::string subroutine_str =
      R"proto(
        name: "test_subroutine"
        subroutine_root_id: 8
        execution_probability: 1
        execution_count: 1
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
          name: "reduction_operand1"
          opcode: "delay"
          instruction_id: 2
          ops: 128
          operand_ids: 1
        }
        instructions {
          name: "compute_pred1"
          opcode: "infeed"
          instruction_id: 1
          bytes_out: 36.5
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
      )proto";
  google::protobuf::TextFormat::ParseFromString(subroutine_str,
                                                &subroutine_proto);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<paragraph::Subroutine> subroutine,
      paragraph::Subroutine::CreateFromProto(subroutine_proto,
                                             graph.get()));
  EXPECT_EQ(subroutine->GetName(), "test_subroutine");
  EXPECT_EQ(subroutine->GetId(), 8);
  EXPECT_EQ(subroutine->GetRootInstruction()->GetName(), "send");
  EXPECT_EQ(subroutine->GetRootInstruction()->GetId(), 8);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      subroutine->ToProto().value(), subroutine_proto));
}

// Tests subroutine cloning
TEST(Subroutine, Clone) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kAllReduce, "reduction", sub_ptr, true));
  paragraph::CommunicationGroup group_1 = {1, 7, 42};
  instr_1->AppendCommunicationGroup(group_1);
  instr_1->SetBytesIn(8);

  auto call_sub = absl::make_unique<paragraph::Subroutine>(
      "call_subroutine", graph.get());
  auto call_sub_ptr = call_sub.get();
  ASSERT_OK_AND_ASSIGN(auto call_func, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "call_func", call_sub_ptr, true));
  call_func->SetSeconds(0.001);

  auto cond_sub = absl::make_unique<paragraph::Subroutine>(
      "cond_subroutine", graph.get());
  auto cond_sub_ptr = cond_sub.get();
  ASSERT_OK_AND_ASSIGN(auto cond_call, paragraph::Instruction::Create(
      paragraph::Opcode::kCall, "cond_call", cond_sub_ptr, true));
  cond_call->AppendInnerSubroutine(std::move(call_sub));
  instr_1->AppendInnerSubroutine(std::move(cond_sub));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<paragraph::Subroutine>
      cloned_subroutine, sub_ptr->Clone("_clone"));
  EXPECT_EQ(cloned_subroutine->GetName(), "test_subroutine_clone");
  google::protobuf::util::MessageDifferencer diff;
  ASSERT_OK_AND_ASSIGN(auto subroutine_proto, sub_ptr->ToProto());
  diff.IgnoreField(subroutine_proto.GetDescriptor()->FindFieldByName("name"));
  for (size_t i = 0; i < sub_ptr->MakeEmbeddedSubroutinesVector().size(); i++) {
    EXPECT_EQ(sub_ptr->MakeEmbeddedSubroutinesVector().at(i)->GetId(),
        cloned_subroutine->MakeEmbeddedSubroutinesVector().at(i)->GetId());
    EXPECT_EQ(sub_ptr->MakeEmbeddedSubroutinesVector().at(i)->GetName(),
        cloned_subroutine->MakeEmbeddedSubroutinesVector().at(i)->GetName());
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<paragraph::Subroutine>
      new_subroutine, sub_ptr->Clone("_new_id", true));
  EXPECT_EQ(new_subroutine->GetName(), "test_subroutine_new_id");
  for (size_t i = 0; i < sub_ptr->MakeEmbeddedSubroutinesVector().size(); i++) {
    EXPECT_NE(sub_ptr->MakeEmbeddedSubroutinesVector().at(i)->GetId(),
        new_subroutine->MakeEmbeddedSubroutinesVector().at(i)->GetId());
    EXPECT_EQ(sub_ptr->MakeEmbeddedSubroutinesVector().at(i)->GetName(),
        new_subroutine->MakeEmbeddedSubroutinesVector().at(i)->GetName());
  }
}

// Tests ValidateComposite() and ValidateIndividualized() methods
TEST(Subroutine, Validate) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "test", sub_ptr));
  EXPECT_EQ(sub_ptr->ValidateComposite(),
            absl::InternalError("Subroutine test_subroutine does not "
                                "have a root instruction."));
  EXPECT_OK(sub_ptr->SetRootInstruction(instr_1));
  EXPECT_OK(sub_ptr->ValidateComposite());
  sub_ptr->SetId(13);
  EXPECT_EQ(sub_ptr->ValidateComposite(),
            absl::InternalError("Subroutine test_subroutine should "
                                "have the same ID as its root."));
  sub_ptr->SetId(1);
  EXPECT_OK(sub_ptr->ValidateComposite());

  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "test_2", sub_ptr));
  instr_2->AddOperand(instr_1);
  EXPECT_EQ(sub_ptr->ValidateComposite(),
            absl::InternalError("Subroutine test_subroutine root "
                                "instruction test should have no users."));

  auto body_sub = absl::make_unique<paragraph::Subroutine>(
      "body_subroutine", graph.get());
  auto body_ptr = body_sub.get();
  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "test_3", body_ptr, true));
  instr_3->SetId(3);
  ASSERT_OK_AND_ASSIGN(auto instr_4, paragraph::Instruction::Create(
      paragraph::Opcode::kInfeed, "alien_instruction", body_ptr));
  instr_4->SetId(4);
  instr_4->SetParent(sub_ptr);

  EXPECT_EQ(body_sub->ValidateComposite(),
            absl::InternalError("Instruction alien_instruction does not "
                                "point to the parent subroutine "
                                "body_subroutine."));

  auto while_sub = absl::make_unique<paragraph::Subroutine>(
      "while_subroutine", graph.get());
  auto while_sub_ptr = while_sub.get();
  ASSERT_OK_AND_ASSIGN(auto while_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kWhile, "test", while_sub_ptr, true));
  auto cond_sub = absl::make_unique<paragraph::Subroutine>(
      "cond_subroutine", graph.get());
  auto cond_sub_ptr = cond_sub.get();
  ASSERT_OK_AND_ASSIGN(auto cond_compute, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "cond_compute", cond_sub_ptr, true));
  cond_compute->SetId(6);
  while_instr->AppendInnerSubroutine(std::move(cond_sub));
  EXPECT_EQ(while_sub_ptr->ValidateComposite(),
            absl::InternalError("While instruction test should have "
                                "exactly 2 subroutines."));
}
