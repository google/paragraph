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
#include "paragraph/graph/instruction.h"

#include <memory>
#include <string>
#include <utility>

#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "paragraph/graph/graph.h"
#include "gtest/gtest.h"
#include "paragraph/shim/test_macros.h"

std::unique_ptr<paragraph::Graph> get_graph() {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  graph->SetEntrySubroutine(std::move(sub));
  return graph;
}

// Tests construction, name and opcode return
// Test constructs several instructions with various names and opcodes,
// checks that opcodes and names are the ones that were passed to constructor
TEST(Instruction, Construction) {
  auto graph = get_graph();
  auto sub_ptr = graph->GetEntrySubroutine();

  std::string instr_1_name = "Compute_1";
  paragraph::Opcode instr_1_opcode = paragraph::Opcode::kDelay;
  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      instr_1_opcode, instr_1_name, sub_ptr));
  EXPECT_EQ(instr_1_name, instr_1->GetName());
  EXPECT_EQ(instr_1_opcode, instr_1->GetOpcode());

  std::string instr_2_name = "AllReduce";
  paragraph::Opcode instr_2_opcode = paragraph::Opcode::kAllReduce;
  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      instr_2_opcode, instr_2_name, sub_ptr));
  EXPECT_EQ(instr_2_name, instr_2->GetName());
  EXPECT_EQ(instr_2_opcode, instr_2->GetOpcode());

  std::string instr_3_name = "send_instruction.42";
  paragraph::Opcode instr_3_opcode = paragraph::Opcode::kSend;
  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      instr_3_opcode, instr_3_name, sub_ptr));
  EXPECT_EQ(instr_3_name, instr_3->GetName());
  EXPECT_EQ(instr_3_opcode, instr_3->GetOpcode());
}

// Tests id(), set_id(), ClearID(), and HasDefaultId() members
// Tests that default GetId is -1, ids can be set/cleared
TEST(Instruction, IdHandling) {
  auto graph = get_graph();
  auto sub_ptr = graph->GetEntrySubroutine();

  EXPECT_EQ(paragraph::Instruction::kDefaultId, -1);
  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "test", sub_ptr));
  EXPECT_EQ(instr_1->GetId(), 1);
  EXPECT_EQ(instr_1->HasDefaultId(), false);

  instr_1->SetId(42);
  EXPECT_EQ(instr_1->GetId(), 42);
  EXPECT_EQ(instr_1->HasDefaultId(), false);

  instr_1->SetId(777);
  EXPECT_EQ(instr_1->GetId(), 777);
  EXPECT_EQ(instr_1->HasDefaultId(), false);

  instr_1->ClearId();
  EXPECT_EQ(instr_1->GetId(), -1);
  EXPECT_EQ(instr_1->HasDefaultId(), true);
}

// Tests adding user(s) to an instruction,
// Tests how Users(), UserCount(), IsUserOf() members hand user addition
TEST(Instruction, AddingUser) {
  auto graph = get_graph();
  auto sub_ptr = graph->GetEntrySubroutine();

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp1", sub_ptr));
  EXPECT_EQ(instr_1->Users().empty(), true);
  EXPECT_EQ(instr_1->UserCount(), 0);

  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp1_user1", sub_ptr));
  EXPECT_EQ(instr_2->IsUserOf(instr_1), false);
  EXPECT_TRUE(instr_1->AddUser(instr_2));
  EXPECT_EQ(instr_1->Users().empty(), false);
  EXPECT_EQ(instr_1->UserCount(), 1);
  EXPECT_EQ(instr_2->IsUserOf(instr_1), true);

  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp1_user2", sub_ptr));
  EXPECT_EQ(instr_3->IsUserOf(instr_1), false);
  EXPECT_TRUE(instr_1->AddUser(instr_3));
  EXPECT_EQ(instr_1->Users().empty(), false);
  EXPECT_EQ(instr_1->UserCount(), 2);
  EXPECT_EQ(instr_2->IsUserOf(instr_1), true);
  EXPECT_EQ(instr_3->IsUserOf(instr_1), true);
}

// Tests removing user from different positions,
// Tests how UserID(), IsUserOf() members handle user removing
TEST(Instruction, RemovingUser) {
  auto graph = get_graph();
  auto sub_ptr = graph->GetEntrySubroutine();

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp1", sub_ptr));

  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp1_user1", sub_ptr));
  instr_1->AddUser(instr_2);
  EXPECT_EQ(instr_1->UserId(instr_2), 0);

  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp1_user2", sub_ptr));
  instr_1->AddUser(instr_3);
  EXPECT_EQ(instr_1->UserId(instr_3), 1);

  ASSERT_OK_AND_ASSIGN(auto instr_4, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp1_user3", sub_ptr));
  instr_1->AddUser(instr_4);
  EXPECT_EQ(instr_1->UserId(instr_4), 2);

  EXPECT_EQ(instr_1->UserCount(), 3);
  EXPECT_OK(instr_1->RemoveUser(instr_3));
  EXPECT_EQ(instr_1->UserCount(), 2);
  EXPECT_EQ(instr_2->IsUserOf(instr_1), true);
  EXPECT_EQ(instr_1->UserId(instr_2), 0);
  EXPECT_EQ(instr_3->IsUserOf(instr_1), false);
  EXPECT_EQ(instr_4->IsUserOf(instr_1), true);
  EXPECT_EQ(instr_1->UserId(instr_4), 1);

  EXPECT_OK(instr_1->RemoveUser(instr_4));
  EXPECT_EQ(instr_1->UserCount(), 1);
  EXPECT_EQ(instr_2->IsUserOf(instr_1), true);
  EXPECT_EQ(instr_1->UserId(instr_2), 0);
  EXPECT_EQ(instr_3->IsUserOf(instr_1), false);
  EXPECT_EQ(instr_4->IsUserOf(instr_1), false);

  ASSERT_OK_AND_ASSIGN(auto instr_5, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp1_user4", sub_ptr));
  instr_1->AddUser(instr_5);
  EXPECT_EQ(instr_1->UserId(instr_5), 1);
  EXPECT_EQ(instr_1->UserCount(), 2);

  EXPECT_OK(instr_1->RemoveUser(instr_2));
  EXPECT_EQ(instr_1->UserCount(), 1);
  EXPECT_EQ(instr_2->IsUserOf(instr_1), false);
  EXPECT_EQ(instr_2->IsUserOf(instr_1), false);
  EXPECT_EQ(instr_4->IsUserOf(instr_1), false);
  EXPECT_EQ(instr_5->IsUserOf(instr_1), true);
  EXPECT_EQ(instr_1->UserId(instr_5), 0);
}

// Tests Adding an operand to an instruction
// Tests AddOperand(), Operands() and OperandCount() members behavior
// when operand is added.
// Tests that adding operand to an instructions adds it to the operand's users
TEST(Instruction, AddingOperand) {
  auto graph = get_graph();
  auto sub_ptr = graph->GetEntrySubroutine();

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp1", sub_ptr));
  EXPECT_EQ(instr_1->OperandCount(), 0);

  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp1_operand1", sub_ptr));
  EXPECT_EQ(instr_2->UserCount(), 0);
  instr_1->AddOperand(instr_2);
  EXPECT_EQ(instr_1->OperandCount(), 1);
  EXPECT_EQ(instr_1->Operands().at(0), instr_2);
  EXPECT_EQ(instr_2->UserCount(), 1);
  EXPECT_EQ(instr_1->IsUserOf(instr_2), true);
  EXPECT_EQ(instr_2->UserId(instr_1), 0);

  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp1_operand2", sub_ptr));
  instr_1->AddOperand(instr_3);
  EXPECT_EQ(instr_1->OperandCount(), 2);
  EXPECT_EQ(instr_1->Operands().at(1), instr_3);
  EXPECT_EQ(instr_3->UserCount(), 1);
  EXPECT_EQ(instr_1->IsUserOf(instr_3), true);
  EXPECT_EQ(instr_3->UserId(instr_1), 0);
}

// Tests bonding instructions
TEST(Instruction, Bonding) {
  auto graph = get_graph();
  auto sub_ptr = graph->GetEntrySubroutine();

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kSendStart, "start", sub_ptr));
  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kSendDone, "finish", sub_ptr, true));

  EXPECT_EQ(instr_1->GetBondedInstruction(), nullptr);
  EXPECT_EQ(instr_2->GetBondedInstruction(), nullptr);

  instr_1->BondWith(instr_2);
  EXPECT_EQ(instr_1->GetBondedInstruction(), instr_2);
  EXPECT_EQ(instr_2->GetBondedInstruction(), instr_1);
}

// Tests getting/setting parent subroutine
TEST(Instruction, ParentSubroutine) {
  auto graph = get_graph();
  auto sub_ptr = graph->GetEntrySubroutine();

  auto subroutine = absl::make_unique<paragraph::Subroutine>(
      "test_Subroutine", nullptr);
  auto subroutine_ptr = subroutine.get();
  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp1", sub_ptr));
  EXPECT_EQ(instr_1->GetParent(), sub_ptr);

  instr_1->SetParent(subroutine_ptr);
  EXPECT_EQ(instr_1->GetParent(), subroutine_ptr);
}

// Tests accessing and changin inner subroutines
TEST(Instruction, InnerSubroutines) {
  auto graph = get_graph();
  auto sub_ptr = graph->GetEntrySubroutine();

  ASSERT_OK_AND_ASSIGN(auto main_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kWhile, "test_instr", sub_ptr));

  auto subroutine_1 = absl::make_unique<paragraph::Subroutine>(
      "subroutine_1", graph.get());
  auto subroutine_1_ptr = subroutine_1.get();
  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp1", subroutine_1_ptr));
  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp2", subroutine_1_ptr));
  instr_2->SetId(2);
  EXPECT_OK(subroutine_1->SetRootInstruction(instr_2));

  auto subroutine_2 = absl::make_unique<paragraph::Subroutine>(
      "subroutine_2", graph.get());
  auto subroutine_2_ptr = subroutine_2.get();
  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp3", subroutine_2_ptr));
  ASSERT_OK_AND_ASSIGN(auto instr_4, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp4", subroutine_2_ptr));
  instr_4->SetId(4);
  EXPECT_OK(subroutine_2->SetRootInstruction(instr_4));

  EXPECT_EQ(main_instr->InnerSubroutines().size(), 0);
  main_instr->AppendInnerSubroutine(std::move(subroutine_1));
  EXPECT_EQ(main_instr->InnerSubroutines().size(), 1);
  auto& main_sub_1 = main_instr->InnerSubroutines().at(0);
  EXPECT_EQ(main_sub_1.get(), subroutine_1_ptr);
  EXPECT_EQ(main_sub_1->Instructions().size(), 2);
  auto sub_1_it = main_sub_1->Instructions().begin();
  EXPECT_EQ((*sub_1_it).get(), instr_1);
  std::advance(sub_1_it, 1);
  EXPECT_EQ((*sub_1_it).get(), instr_2);

  main_instr->AppendInnerSubroutine(std::move(subroutine_2));
  EXPECT_EQ(main_instr->InnerSubroutines().size(), 2);
  EXPECT_EQ(main_instr->InnerSubroutines().at(0).get(),
            subroutine_1_ptr);

  auto& main_sub_2 = main_instr->InnerSubroutines().at(1);
  EXPECT_EQ(main_sub_2.get(), subroutine_2_ptr);
  EXPECT_EQ(main_sub_2->Instructions().size(), 2);
  auto sub_2_it = main_sub_2->Instructions().begin();
  EXPECT_EQ((*sub_2_it).get(), instr_3);
  std::advance(sub_2_it, 1);
  EXPECT_EQ((*sub_2_it).get(), instr_4);
}

// Tests removing inner subroutines
TEST(Instruction, RemoveInnerSubroutines) {
  auto graph = get_graph();
  auto sub_ptr = graph->GetEntrySubroutine();

  ASSERT_OK_AND_ASSIGN(auto main_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kWhile, "test_instr", sub_ptr));

  auto subroutine_1 = absl::make_unique<paragraph::Subroutine>(
      "subroutine_1", graph.get());
  auto subroutine_1_ptr = subroutine_1.get();
  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp1", subroutine_1_ptr));
  instr_1->SetOps(4);
  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp2", subroutine_1_ptr, true));
  instr_2->SetOps(4);

  auto subroutine_2 = absl::make_unique<paragraph::Subroutine>(
      "subroutine_2", graph.get());
  auto subroutine_2_ptr = subroutine_2.get();
  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp3", subroutine_2_ptr));
  instr_3->SetId(4);
  ASSERT_OK_AND_ASSIGN(auto instr_4, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp4", subroutine_2_ptr, true));
  instr_4->SetOps(4);

  main_instr->AppendInnerSubroutine(std::move(subroutine_1));
  main_instr->AppendInnerSubroutine(std::move(subroutine_2));
  EXPECT_EQ(main_instr->InnerSubroutines().size(), 2);
  EXPECT_EQ(main_instr->InnerSubroutines().at(0).get(), subroutine_1_ptr);
  EXPECT_EQ(main_instr->InnerSubroutines().at(1).get(), subroutine_2_ptr);

  main_instr->RemoveInnerSubroutine(subroutine_1_ptr);
  EXPECT_EQ(main_instr->InnerSubroutines().size(), 1);
  EXPECT_EQ(main_instr->InnerSubroutines().at(0).get(), subroutine_2_ptr);
  auto& main_sub_1 = main_instr->InnerSubroutines().at(0);
  EXPECT_EQ(main_sub_1.get(), subroutine_2_ptr);
  EXPECT_EQ(main_sub_1->Instructions().size(), 2);
  auto sub_1_it = main_sub_1->Instructions().begin();
  EXPECT_EQ((*sub_1_it).get(), instr_3);
  std::advance(sub_1_it, 1);
  EXPECT_EQ((*sub_1_it).get(), instr_4);
}
// Tests getters and setters for performance properties methods
// FLOPs, Bytes, Transcendental operations, and projected time cost
TEST(Instruction, PerformanceProperties) {
  auto graph = get_graph();
  auto sub_ptr = graph->GetEntrySubroutine();

  ASSERT_OK_AND_ASSIGN(auto instr, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "test", sub_ptr));
  EXPECT_EQ(instr->GetOps(), 0);
  EXPECT_EQ(instr->GetBytesIn(), 0);
  EXPECT_EQ(instr->GetBytesOut(), 0);
  EXPECT_EQ(instr->GetTranscendentals(), 0);
  EXPECT_EQ(instr->GetSeconds(), 0);

  instr->SetOps(42.);
  EXPECT_EQ(instr->GetOps(), 42.);
  EXPECT_EQ(instr->GetBytesIn(), 0);
  EXPECT_EQ(instr->GetBytesOut(), 0);
  EXPECT_EQ(instr->GetTranscendentals(), 0);
  EXPECT_EQ(instr->GetSeconds(), 0);

  instr->SetTranscendentals(123456.);
  EXPECT_EQ(instr->GetOps(), 42.);
  EXPECT_EQ(instr->GetBytesIn(), 0);
  EXPECT_EQ(instr->GetBytesOut(), 0);
  EXPECT_EQ(instr->GetTranscendentals(), 123456.);
  EXPECT_EQ(instr->GetSeconds(), 0);

  instr->SetBytesIn(8);
  EXPECT_EQ(instr->GetOps(), 42.);
  EXPECT_EQ(instr->GetBytesIn(), 8);
  EXPECT_EQ(instr->GetBytesOut(), 0);
  EXPECT_EQ(instr->GetTranscendentals(), 123456.);
  EXPECT_EQ(instr->GetSeconds(), 0);

  instr->SetBytesOut(16);
  EXPECT_EQ(instr->GetOps(), 42.);
  EXPECT_EQ(instr->GetBytesIn(), 8);
  EXPECT_EQ(instr->GetBytesOut(), 16);
  EXPECT_EQ(instr->GetTranscendentals(), 123456.);
  EXPECT_EQ(instr->GetSeconds(), 0);

  instr->SetSeconds(0.123);
  EXPECT_EQ(instr->GetOps(), 42.);
  EXPECT_EQ(instr->GetBytesIn(), 8);
  EXPECT_EQ(instr->GetBytesOut(), 16);
  EXPECT_EQ(instr->GetTranscendentals(), 123456.);
  EXPECT_EQ(instr->GetSeconds(), 0.123);
}

// Tests ScalePerformance() method
TEST(Instruction, ScalePerformance) {
  auto graph = get_graph();
  auto sub_ptr = graph->GetEntrySubroutine();

  ASSERT_OK_AND_ASSIGN(auto instr, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "test", sub_ptr));
  instr->SetOps(42.);
  instr->SetTranscendentals(123456.);
  instr->SetBytesIn(8);
  instr->SetBytesOut(16);
  instr->SetSeconds(0.123);
  instr->ScalePerformance(0.5);
  EXPECT_EQ(instr->GetOps(), 21);
  EXPECT_EQ(instr->GetBytesIn(), 4);
  EXPECT_EQ(instr->GetBytesOut(), 8);
  EXPECT_EQ(instr->GetTranscendentals(), 61728);
  EXPECT_EQ(instr->GetSeconds(), 0.0615);

  auto subroutine_1 = absl::make_unique<paragraph::Subroutine>(
      "subroutine_1", graph.get());
  auto subroutine_1_ptr = subroutine_1.get();
  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp1", subroutine_1_ptr));
  instr_1->SetBytesOut(12);
  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "comp2", subroutine_1_ptr));
  instr_2->SetSeconds(0.3);
  instr_2->SetId(2);
  EXPECT_OK(subroutine_1->SetRootInstruction(instr_2));
  ASSERT_OK_AND_ASSIGN(auto main_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "main", sub_ptr));
  main_instr->SetOps(42.);
  main_instr->AppendInnerSubroutine(std::move(subroutine_1));
  main_instr->ScalePerformance(0.3333333333333333);
  EXPECT_FLOAT_EQ(main_instr->GetOps(), 14);
  EXPECT_FLOAT_EQ(instr_1->GetBytesOut(), 4);
  EXPECT_FLOAT_EQ(instr_2->GetSeconds(), 0.1);
}

// Tests GetGraph() member
TEST(Instruction, ParentGraph) {
  auto graph = get_graph();
  auto sub_ptr = graph->GetEntrySubroutine();

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "test", sub_ptr));
  EXPECT_EQ(instr_1->GetGraph(), graph.get());
}

// Tests GetCommunicationGroupVector(), AppendCommunicationGroup() methods
TEST(Instruction, CommunicationGroup) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 42);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));
  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kAllReduce, "test", sub_ptr));
  EXPECT_EQ(instr_1->GetCommunicationGroupVector().empty(), true);

  paragraph::CommunicationGroup group_1 = {1, 7, 42};
  instr_1->AppendCommunicationGroup(group_1);
  EXPECT_EQ(instr_1->GetCommunicationGroupVector().size(), 1);
  EXPECT_EQ(instr_1->GetCommunicationGroupVector().at(0).size(), 3);

  paragraph::CommunicationGroup group_2 = {3, 43};
  instr_1->AppendCommunicationGroup(group_2);
  EXPECT_EQ(instr_1->GetCommunicationGroupVector().size(), 2);
  EXPECT_EQ(instr_1->GetCommunicationGroupVector().at(1).size(), 2);

  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kSendRecv, "sendrecv", sub_ptr));
  EXPECT_EQ(instr_2->GetCommunicationGroupVector().empty(), true);

  paragraph::CommunicationGroup group_3 = {2, 2};
  instr_2->AppendCommunicationGroup(group_3);
  EXPECT_EQ(instr_2->GetCommunicationGroupVector().size(), 1);
  EXPECT_EQ(instr_2->GetCommunicationGroupVector().at(0).size(), 2);
}

// Tests CommunicationGroupVector(), CommunicationGroup(),
// GetProcessorCordinates(), PeerId() and
// CommunicationGroupVectorHasProcessor(processor) methods
TEST(Instruction, ProcessorLocation) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 42);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));
  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kSend, "test_send", sub_ptr));
  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kAllReduce, "test_allreduce", sub_ptr));

  EXPECT_EQ(instr_1->GetGraph()->GetProcessorId(), 42);
  EXPECT_EQ(instr_1->GetProcessorCoordinates(13).status(),
            absl::InvalidArgumentError(
                "Processor index for supplied ProcessorId 13 not found in "
                "instruction test_send\n"));

  paragraph::CommunicationGroup group = {13};
  instr_1->AppendCommunicationGroup(group);
  EXPECT_OK(instr_1->GetProcessorCoordinates(13).status());
  EXPECT_OK(instr_1->PeerId().status());
  EXPECT_EQ(instr_1->PeerId().value(), 13);
  EXPECT_EQ(instr_1->GetCommunicationGroupVector().size(), 1);
  EXPECT_EQ(instr_1->GetCommunicationGroup().size(), 1);
  EXPECT_EQ(instr_1->GetCommunicationGroupVector().at(0),
            instr_1->GetCommunicationGroup());
  EXPECT_EQ(instr_1->GetCommunicationGroup().at(0), 13);
  EXPECT_TRUE(instr_1->CommunicationGroupVectorHasProcessor(13));
  EXPECT_FALSE(instr_1->CommunicationGroupVectorHasProcessor(14));
  paragraph::ProcessorCoordinates test_coord = {0, 0};
  EXPECT_EQ(instr_1->GetProcessorCoordinates(13).value().group,
            test_coord.group);
  EXPECT_EQ(instr_1->GetProcessorCoordinates(13).value().offset,
            test_coord.offset);

  paragraph::CommunicationGroup group_1 = {1, 7};
  instr_2->AppendCommunicationGroup(group_1);
  paragraph::CommunicationGroup group_2 = {3, 13};
  instr_2->AppendCommunicationGroup(group_2);
  EXPECT_EQ(instr_2->GetGraph()->GetProcessorId(), 42);
  EXPECT_OK(instr_2->GetProcessorCoordinates(13).status());
  test_coord = {1, 1};
  EXPECT_EQ(instr_2->GetProcessorCoordinates(13).value().group,
            test_coord.group);
  EXPECT_EQ(instr_2->GetProcessorCoordinates(13).value().offset,
            test_coord.offset);
  EXPECT_EQ(instr_2->GetCommunicationGroupVector().size(), 2);
  EXPECT_EQ(instr_2->GetCommunicationGroupVector().at(0).size(), 2);
  EXPECT_EQ(instr_2->GetCommunicationGroupVector().at(0).at(0), 1);
  EXPECT_EQ(instr_2->GetCommunicationGroupVector().at(0).at(1), 7);
  EXPECT_EQ(instr_2->GetCommunicationGroupVector().at(1).size(), 2);
  EXPECT_EQ(instr_2->GetCommunicationGroupVector().at(1).at(0), 3);
  EXPECT_EQ(instr_2->GetCommunicationGroupVector().at(1).at(1), 13);
  EXPECT_TRUE(instr_2->CommunicationGroupVectorHasProcessor(1));
  EXPECT_TRUE(instr_2->CommunicationGroupVectorHasProcessor(3));
  EXPECT_TRUE(instr_2->CommunicationGroupVectorHasProcessor(7));
  EXPECT_TRUE(instr_2->CommunicationGroupVectorHasProcessor(13));
  EXPECT_FALSE(instr_2->CommunicationGroupVectorHasProcessor(14));
  EXPECT_EQ(instr_2->PeerId().status(), absl::InternalError(
      "Only Send/Recv(Done) instructions can return PeerId;"
      " current opcode is all-reduce."));
  EXPECT_DEATH(instr_2->GetCommunicationGroup(), "");
}

// Tests communication tag and related methods
TEST(Instruction, CommunicationTag) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 42);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));
  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kSend, "test_send", sub_ptr));

  EXPECT_EQ(instr_1->GetCommunicationTag(), 0);

  instr_1->SetCommunicationTag(13);
  EXPECT_EQ(instr_1->GetCommunicationTag(), 13);

  instr_1->SetCommunicationTag(42);
  EXPECT_EQ(instr_1->GetCommunicationTag(), 42);
}

// Tests proto serialization - ToProto() method
TEST(Instruction, StoreToProto) {
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
  send_instr->SetCommunicationTag(12345678);

  paragraph::InstructionProto instr_1_proto;
  std::string instr_1_str =
      R"proto(
        name: "reduction"
        opcode: "all-reduce"
        instruction_id: 3
        communication_groups {
          group_ids: 1
          group_ids: 7
          group_ids: 42
        }
        operand_ids: 2
      )proto";
  google::protobuf::TextFormat::ParseFromString(instr_1_str, &instr_1_proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      instr_1->ToProto().value(), instr_1_proto));

  paragraph::InstructionProto instr_2_proto;
  std::string instr_2_str =
      R"proto(
        name: "reduction_operand1"
        opcode: "delay"
        instruction_id: 2
        ops: 128
        operand_ids: 1
      )proto";
  google::protobuf::TextFormat::ParseFromString(instr_2_str, &instr_2_proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      instr_2->ToProto().value(), instr_2_proto));

  paragraph::InstructionProto instr_3_proto;
  std::string instr_3_str =
      R"proto(
        name: "compute_pred1"
        opcode: "infeed"
        instruction_id: 1
        bytes_out: 36.5
      )proto";
  google::protobuf::TextFormat::ParseFromString(instr_3_str, &instr_3_proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      instr_3->ToProto().value(), instr_3_proto));

  paragraph::InstructionProto while_proto;
  std::string while_str =
      R"proto(
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
      )proto";
  google::protobuf::TextFormat::ParseFromString(while_str, &while_proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      while_instr->ToProto().value(), while_proto));

  paragraph::InstructionProto body_proto;
  std::string body_str =
      R"proto(
        name: "body_compute"
        opcode: "delay"
        instruction_id: 5
        transcendentals: 111
      )proto";
  google::protobuf::TextFormat::ParseFromString(body_str, &body_proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      body_compute->ToProto().value(), body_proto));

  paragraph::InstructionProto cond_proto;
  std::string cond_str =
      R"proto(
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
      )proto";
  google::protobuf::TextFormat::ParseFromString(cond_str, &cond_proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      cond_call->ToProto().value(), cond_proto));

  paragraph::InstructionProto call_proto;
  std::string call_str =
      R"proto(
        name: "call_func"
        opcode: "delay"
        instruction_id: 6
        seconds: 0.001
      )proto";
  google::protobuf::TextFormat::ParseFromString(call_str, &call_proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      call_func->ToProto().value(), call_proto));

  paragraph::InstructionProto send_proto;
  std::string send_str =
      R"proto(
        name: "send"
        opcode: "send"
        instruction_id: 8
        bytes_in: 8
        communication_groups {
          group_ids: 42
        }
        communication_tag: 12345678
      )proto";
  google::protobuf::TextFormat::ParseFromString(send_str, &send_proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      send_instr->ToProto().value(), send_proto));
}

// Tests proto serialization - CreateFromProto() method
TEST(Instruction, LoadFromProto) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));

  paragraph::InstructionProto instruction_proto;
  std::string instruction_str =
      R"proto(
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
      )proto";
  google::protobuf::TextFormat::ParseFromString(instruction_str,
                                                &instruction_proto);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<paragraph::Instruction> instruction,
      paragraph::Instruction::CreateFromProto(instruction_proto,
                                              sub_ptr));
  EXPECT_EQ(instruction->GetName(), "test");
  EXPECT_EQ(instruction->GetId(), 4);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      instruction->ToProto().value(), instruction_proto));
}

// Tests instruction cloning
TEST(Instruction, Clone) {
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<paragraph::Instruction>
      cloned_instruction, instr_1->Clone("_clone"));
  EXPECT_EQ(cloned_instruction->GetName(), "reduction_clone");
  EXPECT_EQ(cloned_instruction->InnerSubroutines().size(), 1);
  EXPECT_EQ(cloned_instruction->InnerSubroutines().at(0)->GetName(),
            "cond_subroutine");
  google::protobuf::util::MessageDifferencer diff;
  ASSERT_OK_AND_ASSIGN(auto instruction_proto, instr_1->ToProto());
  diff.IgnoreField(instruction_proto.GetDescriptor()->FindFieldByName("name"));
  EXPECT_TRUE(diff.Compare(instruction_proto,
                           cloned_instruction->ToProto().value()));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<paragraph::Instruction>
      new_instruction, instr_1->Clone("_new_id", true));
  EXPECT_EQ(new_instruction->GetName(), "reduction_new_id");
  EXPECT_EQ(new_instruction->InnerSubroutines().size(), 1);
  EXPECT_EQ(new_instruction->InnerSubroutines().at(0)->GetName(),
            "cond_subroutine");
  EXPECT_EQ(new_instruction->GetId(), 7);
}

// Tests non-consecutive instruction ids
TEST(Instruction, InconsecutiveIds) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "dummy_1", sub_ptr));
  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "dummy_2", sub_ptr, true));
  instr_2->AddOperand(instr_1);
  instr_2->SetId(4);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<paragraph::Graph>
      new_graph, graph->Clone("_clone"));
  ASSERT_OK_AND_ASSIGN(auto new_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "new_dummy",
      new_graph->GetEntrySubroutine()));

  EXPECT_EQ(new_instr->GetId(), 5);
}

// Tests instruction replacing inner subroutine
TEST(Instruction, ReplaceInnerSubroutine) {
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<paragraph::Subroutine> new_cond_sub,
                       cond_sub_ptr->Clone("_clone"));
  new_cond_sub->ScalePerformance(0.5);
  EXPECT_OK(instr_1->ReplaceInnerSubroutine(cond_sub_ptr,
                                                std::move(new_cond_sub)));
  EXPECT_EQ(instr_1->InnerSubroutines().size(), 1);
  EXPECT_EQ(instr_1->InnerSubroutines().at(0)->GetName(),
              "cond_subroutine_clone");
  paragraph::Instruction* new_cond_call_ptr =
      instr_1->InnerSubroutines().at(0)->Instructions().begin()->get();
  EXPECT_EQ(new_cond_call_ptr->GetName(), "cond_call_clone");
  EXPECT_EQ(new_cond_call_ptr->InnerSubroutines().size(), 1);
  paragraph::Instruction* new_call_func_ptr =
      new_cond_call_ptr->InnerSubroutines().at(
          0)->Instructions().begin()->get();
  EXPECT_EQ(new_call_func_ptr->GetName(), "call_func_clone");
  EXPECT_EQ(new_call_func_ptr->GetSeconds(), 0.0005);
}

// Tests ValidateComposite() and ValidateIndividualized() methods
TEST(Instruction, Validate) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kAllReduce, "reduction", sub_ptr));
  instr_1->SetOps(-1);
  EXPECT_EQ(instr_1->ValidateIndividualized(),
            absl::InternalError("Instruction reduction should have "
                                "non-negative number of arithmetic "
                                "operations."));
  instr_1->SetOps(1);
  EXPECT_OK(instr_1->ValidateComposite());
  EXPECT_EQ(instr_1->ValidateIndividualized(),
            absl::InternalError("ProcessorId corresponding to the instruction "
                                "reduction is not found in "
                                "CommunicationGroupVector."));
  paragraph::CommunicationGroup group_1 = {1, 7, 42};
  instr_1->AppendCommunicationGroup(group_1);
  paragraph::CommunicationGroup group_2 = {2, 8, 43};
  instr_1->AppendCommunicationGroup(group_2);
  EXPECT_EQ(instr_1->ValidateIndividualized(),
            absl::InternalError("Instruction reduction has "
                                "CommunicationGroupVector "
                                "size = 2, should be 1."));
  EXPECT_OK(instr_1->ValidateComposite());

  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kSend, "test_send", sub_ptr));
  instr_2->SetId(4);
  instr_2->SetParent(sub_ptr);
  paragraph::CommunicationGroup group_3 = {1};
  instr_2->AppendCommunicationGroup(group_3);
  EXPECT_EQ(instr_2->ValidateComposite(),
            absl::InternalError("Instruction test_send each "
                                "CommunicationGroup size "
                                "should be equal to 2."));
  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kSendRecv, "test_sendrecv", sub_ptr));
  instr_3->SetId(5);
  instr_3->SetParent(sub_ptr);
  paragraph::CommunicationGroup group_4 = {2, 3};
  instr_3->AppendCommunicationGroup(group_4);
  EXPECT_EQ(instr_3->ValidateComposite(),
            absl::InternalError("Instruction test_sendrecv each "
                                "CommunicationGroup "
                                "size should be equal to 3."));

  ASSERT_OK_AND_ASSIGN(auto while_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kWhile, "test", sub_ptr));
  auto body_sub = absl::make_unique<paragraph::Subroutine>(
      "body_subroutine", graph.get());
  auto body_sub_ptr = body_sub.get();
  ASSERT_OK_AND_ASSIGN(auto body_compute, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "body_compute", body_sub_ptr, true));
  body_compute->SetOps(4);
  while_instr->AppendInnerSubroutine(std::move(body_sub));
  EXPECT_EQ(while_instr->ValidateComposite(),
            absl::InternalError("While instruction test should have "
                                "exactly 2 subroutines."));

  auto graph_2 = absl::make_unique<paragraph::Graph>("test_graph", 2);
  auto cond_sub = absl::make_unique<paragraph::Subroutine>(
      "cond_subroutine", graph_2.get());
  auto cond_sub_ptr = cond_sub.get();
  ASSERT_OK_AND_ASSIGN(auto cond_call, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "cond_call", cond_sub_ptr, true));
  cond_call->SetOps(4);
  while_instr->AppendInnerSubroutine(std::move(cond_sub));
  EXPECT_EQ(while_instr->ValidateComposite(),
            absl::InternalError("Inner subroutine cond_subroutine "
                                "does not point to the current graph."));
}
