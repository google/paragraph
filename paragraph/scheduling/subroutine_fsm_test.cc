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
#include "paragraph/scheduling/subroutine_fsm.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paragraph/graph/graph.h"
#include "gtest/gtest.h"
#include "paragraph/scheduling/graph_scheduler.h"
#include "paragraph/shim/test_macros.h"

// Tests StringToSubroutineState() method
TEST(SubroutineFsm, StringToStateConversion) {
  std::string state_str_1 = "blocked";
  ASSERT_OK_AND_ASSIGN(paragraph::SubroutineFsm::State state_1,
                       paragraph::SubroutineFsm::StringToSubroutineState(
                           state_str_1));
  EXPECT_EQ(state_1, paragraph::SubroutineFsm::State::kBlocked);

  std::string state_str_3 = "scheduled";
  ASSERT_OK_AND_ASSIGN(paragraph::SubroutineFsm::State state_3,
                       paragraph::SubroutineFsm::StringToSubroutineState(
                           state_str_3));
  EXPECT_EQ(state_3, paragraph::SubroutineFsm::State::kScheduled);

  std::string state_str_4 = "executed";
  ASSERT_OK_AND_ASSIGN(paragraph::SubroutineFsm::State state_4,
                       paragraph::SubroutineFsm::StringToSubroutineState(
                           state_str_4));
  EXPECT_EQ(state_4, paragraph::SubroutineFsm::State::kExecuted);
}

// Tests SubroutineStateToString() method
TEST(SubroutineFsm, StateToStringConversion) {
  paragraph::SubroutineFsm::State state_1 =
      paragraph::SubroutineFsm::State::kBlocked;
  EXPECT_EQ(paragraph::SubroutineFsm::SubroutineStateToString(state_1),
            "blocked");

  paragraph::SubroutineFsm::State state_3 =
      paragraph::SubroutineFsm::State::kScheduled;
  EXPECT_EQ(paragraph::SubroutineFsm::SubroutineStateToString(state_3),
            "scheduled");

  paragraph::SubroutineFsm::State state_4 =
      paragraph::SubroutineFsm::State::kExecuted;
  EXPECT_EQ(paragraph::SubroutineFsm::SubroutineStateToString(state_4),
            "executed");
}

// Tests subroutine FSM state setters and getters
TEST(SubroutineFsm, StateTransition) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));
  ASSERT_OK_AND_ASSIGN(auto instr, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "dummy", sub_ptr, true));
  instr->SetOps(4);

  ASSERT_OK_AND_ASSIGN(auto scheduler,
                       paragraph::GraphScheduler::Create(graph.get()));
  EXPECT_TRUE(scheduler->GetFsm(sub_ptr).IsScheduled());

  scheduler->GetFsm(sub_ptr).SetBlocked();
  EXPECT_TRUE(scheduler->GetFsm(sub_ptr).IsBlocked());

  scheduler->GetFsm(sub_ptr).SetScheduled();
  EXPECT_TRUE(scheduler->GetFsm(sub_ptr).IsScheduled());

  scheduler->GetFsm(sub_ptr).SetExecuted();
  EXPECT_TRUE(scheduler->GetFsm(sub_ptr).IsExecuted());
}

// Tests subroutine FSM execution count setting and decrementing
TEST(SubroutineFsm, ExecutionCount) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));
  ASSERT_OK_AND_ASSIGN(auto while_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kWhile, "while", sub_ptr, true));
  auto body_sub = absl::make_unique<paragraph::Subroutine>(
      "body_subroutine", graph.get());
  body_sub->SetExecutionCount(3);
  auto body_ptr = body_sub.get();
  ASSERT_OK_AND_ASSIGN(auto body_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "body", body_sub.get()));
  ASSERT_OK_AND_ASSIGN(auto body_root, paragraph::Instruction::Create(
      paragraph::Opcode::kNull, "body_root", body_sub.get(), true));
  body_root->AddOperand(body_instr);
  while_instr->AppendInnerSubroutine(std::move(body_sub));
  auto cond_sub = absl::make_unique<paragraph::Subroutine>(
      "cond_subroutine", graph.get());
  cond_sub->SetExecutionCount(3);
  auto cond_ptr = cond_sub.get();
  ASSERT_OK_AND_ASSIGN(auto cond_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "condition", cond_sub.get(), true));
  cond_instr->SetOps(4);
  while_instr->AppendInnerSubroutine(std::move(cond_sub));

  ASSERT_OK_AND_ASSIGN(auto scheduler,
                       paragraph::GraphScheduler::Create(graph.get()));
  EXPECT_EQ(scheduler->GetFsm(sub_ptr).GetExecutionCount(), 1);
  EXPECT_EQ(scheduler->GetFsm(body_ptr).GetExecutionCount(), 3);
  EXPECT_EQ(scheduler->GetFsm(cond_ptr).GetExecutionCount(), 3);

  scheduler->GetFsm(body_ptr).DecrementExecutionCount();
  EXPECT_EQ(scheduler->GetFsm(sub_ptr).GetExecutionCount(), 1);
  EXPECT_EQ(scheduler->GetFsm(body_ptr).GetExecutionCount(), 2);
  EXPECT_EQ(scheduler->GetFsm(cond_ptr).GetExecutionCount(), 3);

  scheduler->GetFsm(cond_ptr).DecrementExecutionCount();
  EXPECT_EQ(scheduler->GetFsm(sub_ptr).GetExecutionCount(), 1);
  EXPECT_EQ(scheduler->GetFsm(body_ptr).GetExecutionCount(), 2);
  EXPECT_EQ(scheduler->GetFsm(cond_ptr).GetExecutionCount(), 2);
}

// Tests Reset() method
TEST(SubroutineFsm, ResetState) {
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

  ASSERT_OK_AND_ASSIGN(auto scheduler,
                       paragraph::GraphScheduler::Create(graph.get()));

  EXPECT_TRUE(scheduler->GetFsm(instr_1).IsReady());
  EXPECT_TRUE(scheduler->GetFsm(instr_2).IsBlocked());
  EXPECT_TRUE(scheduler->GetFsm(sub_ptr).IsScheduled());

  scheduler->GetFsm(instr_1).SetExecuted();
  scheduler->GetFsm(instr_2).SetExecuted();
  scheduler->GetFsm(sub_ptr).SetExecuted();
  EXPECT_TRUE(scheduler->GetFsm(instr_1).IsExecuted());
  EXPECT_TRUE(scheduler->GetFsm(instr_2).IsExecuted());
  EXPECT_TRUE(scheduler->GetFsm(sub_ptr).IsExecuted());

  scheduler->GetFsm(sub_ptr).Reset();
  EXPECT_TRUE(scheduler->GetFsm(instr_1).IsReady());
  EXPECT_TRUE(scheduler->GetFsm(instr_2).IsBlocked());
  EXPECT_TRUE(scheduler->GetFsm(sub_ptr).IsBlocked());
}

// Tests PrepareToSchedule() method
TEST(SubroutineFsm, PrepareToSchedule) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "dummy", sub_ptr));
  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "root", sub_ptr, true));
  instr_2->AddOperand(instr_1);

  ASSERT_OK_AND_ASSIGN(auto while_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kWhile, "while", sub_ptr));
  auto body_sub = absl::make_unique<paragraph::Subroutine>(
      "body_subroutine", graph.get());
  auto body_sub_ptr = body_sub.get();
  ASSERT_OK_AND_ASSIGN(auto body_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "body", body_sub.get()));
  ASSERT_OK_AND_ASSIGN(auto body_root, paragraph::Instruction::Create(
      paragraph::Opcode::kNull, "body_root", body_sub.get(), true));
  body_root->AddOperand(body_instr);
  while_instr->AppendInnerSubroutine(std::move(body_sub));
  auto cond_sub = absl::make_unique<paragraph::Subroutine>(
      "cond_subroutine", graph.get());
  ASSERT_OK_AND_ASSIGN(auto cond_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "condition", cond_sub.get(), true));
  cond_instr->SetOps(4);
  while_instr->AppendInnerSubroutine(std::move(cond_sub));
  while_instr->AddOperand(instr_1);

  ASSERT_OK_AND_ASSIGN(auto scheduler,
                       paragraph::GraphScheduler::Create(graph.get()));
  // Consume first available instruction, which is dummy
  auto consumed_instructions = scheduler->GetReadyInstructions();

  EXPECT_OK(scheduler->GetFsm(body_sub_ptr).PrepareToSchedule());
  consumed_instructions = scheduler->GetReadyInstructions();
  EXPECT_EQ(consumed_instructions.size(), 1);
  EXPECT_EQ(consumed_instructions.at(0)->GetName(), "body");
}

// Tests subroutine FSM state change when instruction is executed
TEST(SubroutineFsm, InstructionExecuted) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));
  ASSERT_OK_AND_ASSIGN(auto instr, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "dummy", sub_ptr, true));
  instr->SetOps(4);

  ASSERT_OK_AND_ASSIGN(auto scheduler,
                       paragraph::GraphScheduler::Create(graph.get()));
  EXPECT_EQ(scheduler->GetFsm(sub_ptr).GetExecutionCount(), 1);
  EXPECT_TRUE(scheduler->GetFsm(sub_ptr).IsScheduled());
  // Consume first available instruction, which is dummy
  auto consumed_instructions = scheduler->GetReadyInstructions();

  EXPECT_OK(scheduler->GetFsm(sub_ptr).InstructionExecuted(
      consumed_instructions.at(0)));
  EXPECT_EQ(scheduler->GetFsm(sub_ptr).GetExecutionCount(), 0);
  EXPECT_TRUE(scheduler->GetFsm(sub_ptr).IsExecuted());
  consumed_instructions = scheduler->GetReadyInstructions();
  EXPECT_EQ(consumed_instructions.size(), 0);
}
