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
#include "paragraph/scheduling/instruction_fsm.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paragraph/graph/graph.h"
#include "gtest/gtest.h"
#include "paragraph/scheduling/graph_scheduler.h"
#include "paragraph/shim/test_macros.h"

// Tests StringToInstructionState() method
TEST(InstructionFsm, StringToStateConversion) {
  std::string state_str_1 = "blocked";
  ASSERT_OK_AND_ASSIGN(paragraph::InstructionFsm::State state_1,
                       paragraph::InstructionFsm::StringToInstructionState(
                           state_str_1));
  EXPECT_EQ(state_1, paragraph::InstructionFsm::State::kBlocked);

  std::string state_str_2 = "ready";
  ASSERT_OK_AND_ASSIGN(paragraph::InstructionFsm::State state_2,
                       paragraph::InstructionFsm::StringToInstructionState(
                           state_str_2));
  EXPECT_EQ(state_2, paragraph::InstructionFsm::State::kReady);

  std::string state_str_3 = "scheduled";
  ASSERT_OK_AND_ASSIGN(paragraph::InstructionFsm::State state_3,
                       paragraph::InstructionFsm::StringToInstructionState(
                           state_str_3));
  EXPECT_EQ(state_3, paragraph::InstructionFsm::State::kScheduled);

  std::string state_str_4 = "executed";
  ASSERT_OK_AND_ASSIGN(paragraph::InstructionFsm::State state_4,
                       paragraph::InstructionFsm::StringToInstructionState(
                           state_str_4));
  EXPECT_EQ(state_4, paragraph::InstructionFsm::State::kExecuted);
}

// Tests InstructionStateToString() method
TEST(InstructionFsm, StateToStringConversion) {
  paragraph::InstructionFsm::State state_1 =
      paragraph::InstructionFsm::State::kBlocked;
  EXPECT_EQ(paragraph::InstructionFsm::InstructionStateToString(state_1),
            "blocked");

  paragraph::InstructionFsm::State state_2 =
      paragraph::InstructionFsm::State::kReady;
  EXPECT_EQ(paragraph::InstructionFsm::InstructionStateToString(state_2),
            "ready");

  paragraph::InstructionFsm::State state_3 =
      paragraph::InstructionFsm::State::kScheduled;
  EXPECT_EQ(paragraph::InstructionFsm::InstructionStateToString(state_3),
            "scheduled");

  paragraph::InstructionFsm::State state_4 =
      paragraph::InstructionFsm::State::kExecuted;
  EXPECT_EQ(paragraph::InstructionFsm::InstructionStateToString(state_4),
            "executed");
}

// Tests instruction FSM state setters and getters
TEST(InstructionFsm, StateTransition) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));
  ASSERT_OK_AND_ASSIGN(auto instr, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "dummy", sub_ptr, true));

  ASSERT_OK_AND_ASSIGN(auto scheduler,
                       paragraph::GraphScheduler::Create(graph.get()));
  auto instr_fsm = scheduler->GetFsm(instr);
  EXPECT_TRUE(instr_fsm.IsReady());

  instr_fsm.SetBlocked();
  EXPECT_TRUE(instr_fsm.IsBlocked());

  instr_fsm.SetReady();
  EXPECT_TRUE(instr_fsm.IsReady());

  instr_fsm.SetScheduled();
  EXPECT_TRUE(instr_fsm.IsScheduled());

  instr_fsm.SetExecuted();
  EXPECT_TRUE(instr_fsm.IsExecuted());
}

// Tests IsUnblockedByOperands() method
TEST(InstructionFsm, CheckUnblocked) {
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

  EXPECT_TRUE(scheduler->GetFsm(instr_1).IsUnblockedByOperands());
  EXPECT_FALSE(scheduler->GetFsm(instr_2).IsUnblockedByOperands());

  scheduler->GetFsm(instr_1).SetExecuted();
  EXPECT_TRUE(scheduler->GetFsm(instr_2).IsUnblockedByOperands());
}

// Tests Reset() method
TEST(InstructionFsm, ResetState) {
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

  scheduler->GetFsm(instr_1).SetExecuted();
  scheduler->GetFsm(instr_2).SetExecuted();
  EXPECT_TRUE(scheduler->GetFsm(instr_1).IsExecuted());
  EXPECT_TRUE(scheduler->GetFsm(instr_2).IsExecuted());

  scheduler->GetFsm(instr_2).Reset();
  EXPECT_TRUE(scheduler->GetFsm(instr_1).IsExecuted());
  EXPECT_TRUE(scheduler->GetFsm(instr_2).IsBlocked());

  scheduler->GetFsm(instr_1).Reset();
  EXPECT_TRUE(scheduler->GetFsm(instr_1).IsReady());
  EXPECT_TRUE(scheduler->GetFsm(instr_2).IsBlocked());
}

// Tests PrepareToSchedule() method
TEST(InstructionFsm, PrepareToSchedule) {
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

  EXPECT_OK(scheduler->GetFsm(while_instr).PrepareToSchedule());
  consumed_instructions = scheduler->GetReadyInstructions();
  EXPECT_EQ(consumed_instructions.size(), 1);
  EXPECT_EQ(consumed_instructions.at(0)->GetName(), "body");
}

// Tests PickSubroutine() method
TEST(InstructionFsm, PickSubroutine) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto conditional, paragraph::Instruction::Create(
      paragraph::Opcode::kConditional, "conditional", sub_ptr));
  auto cond1_sub = absl::make_unique<paragraph::Subroutine>(
      "cond1_subroutine", graph.get());
  cond1_sub->SetExecutionProbability(0.25);
  ASSERT_OK_AND_ASSIGN(auto cond1_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "cond1", cond1_sub.get(), true));
  conditional->AppendInnerSubroutine(std::move(cond1_sub));
  cond1_instr->SetOps(4);
  auto cond2_sub = absl::make_unique<paragraph::Subroutine>(
      "cond2_subroutine", graph.get());
  cond2_sub->SetExecutionProbability(0.5);
  ASSERT_OK_AND_ASSIGN(auto cond2_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "cond2", cond2_sub.get(), true));
  conditional->AppendInnerSubroutine(std::move(cond2_sub));
  cond2_instr->SetOps(4);
  auto cond3_sub = absl::make_unique<paragraph::Subroutine>(
      "cond3_subroutine", graph.get());
  cond3_sub->SetExecutionProbability(0.25);
  ASSERT_OK_AND_ASSIGN(auto cond3_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "cond3", cond3_sub.get(), true));
  conditional->AppendInnerSubroutine(std::move(cond3_sub));
  cond3_instr->SetOps(4);

  ASSERT_OK_AND_ASSIGN(auto allreduce, paragraph::Instruction::Create(
      paragraph::Opcode::kCall, "allreduce", sub_ptr, true));
  auto reduction_sub = absl::make_unique<paragraph::Subroutine>(
      "reduction_subroutine", graph.get());
  ASSERT_OK_AND_ASSIGN(auto reduce_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "reduce", reduction_sub.get(), true));
  reduce_instr->SetOps(4);
  allreduce->AppendInnerSubroutine(std::move(reduction_sub));

  ASSERT_OK_AND_ASSIGN(auto while_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kWhile, "while", sub_ptr));
  auto body_sub = absl::make_unique<paragraph::Subroutine>(
      "body_subroutine", graph.get());
  ASSERT_OK_AND_ASSIGN(auto body_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "body", body_sub.get()));
  ASSERT_OK_AND_ASSIGN(auto body_root, paragraph::Instruction::Create(
      paragraph::Opcode::kNull, "body_root", body_sub.get(), true));
  body_root->AddOperand(body_instr);
  auto body_ptr = body_sub.get();
  while_instr->AppendInnerSubroutine(std::move(body_sub));
  auto cond_sub = absl::make_unique<paragraph::Subroutine>(
      "cond_subroutine", graph.get());
  ASSERT_OK_AND_ASSIGN(auto cond_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "condition", cond_sub.get(), true));
  cond_instr->SetOps(4);
  while_instr->AppendInnerSubroutine(std::move(cond_sub));

  ASSERT_OK_AND_ASSIGN(auto scheduler,
                       paragraph::GraphScheduler::Create(graph.get()));

  paragraph::Subroutine* picked_subroutine;
  ASSERT_OK_AND_ASSIGN(picked_subroutine,
                       scheduler->GetFsm(allreduce).PickSubroutine());
  EXPECT_EQ(picked_subroutine->GetName(), "reduction_subroutine");

  ASSERT_OK_AND_ASSIGN(picked_subroutine,
                       scheduler->GetFsm(while_instr).PickSubroutine());
  EXPECT_EQ(picked_subroutine->GetName(), "body_subroutine");

  scheduler->GetFsm(body_ptr).SetExecuted();
  ASSERT_OK_AND_ASSIGN(picked_subroutine,
                       scheduler->GetFsm(while_instr).PickSubroutine());
  EXPECT_EQ(picked_subroutine->GetName(), "cond_subroutine");

  const int64_t TRIALS = 10000;
  std::vector<int64_t> counter = {0, 0, 0};
  for (int64_t i = 0; i < TRIALS; ++i) {
    ASSERT_OK_AND_ASSIGN(picked_subroutine,
                         scheduler->GetFsm(conditional).PickSubroutine());
    if (picked_subroutine->GetName() == "cond1_subroutine") {
      counter.at(0)++;
    } else if (picked_subroutine->GetName() == "cond2_subroutine") {
      counter.at(1)++;
    } else if (picked_subroutine->GetName() == "cond3_subroutine") {
      counter.at(2)++;
    } else {
      ASSERT_TRUE(false);
    }
  }
  EXPECT_LE(abs(counter.at(0) - counter.at(2)), 20);
  EXPECT_LE(abs(counter.at(0) + counter.at(2) - counter.at(1)), 200);
}

// Tests PickSubroutine() method with Call instruction
TEST(InstructionFsm, PickSubroutineCall) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));
  ASSERT_OK_AND_ASSIGN(auto call, paragraph::Instruction::Create(
      paragraph::Opcode::kCall, "call", sub_ptr, true));

  auto sub_1 = absl::make_unique<paragraph::Subroutine>(
      "subroutine_1", graph.get());
  auto sub_1_ptr = sub_1.get();
  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "dummy_1", sub_1.get(), true));
  auto sub_2 = absl::make_unique<paragraph::Subroutine>(
      "subroutine_2", graph.get());
  auto sub_2_ptr = sub_2.get();
  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "dummy_2", sub_2.get(), true));
  auto sub_3 = absl::make_unique<paragraph::Subroutine>(
      "subroutine_3", graph.get());
  auto sub_3_ptr = sub_3.get();
  ASSERT_OK_AND_ASSIGN(auto instr_3, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "dummy_3", sub_3.get(), true));

  call->AppendInnerSubroutine(std::move(sub_1));
  call->AppendInnerSubroutine(std::move(sub_2));
  call->AppendInnerSubroutine(std::move(sub_3));

  ASSERT_OK_AND_ASSIGN(auto scheduler,
                       paragraph::GraphScheduler::Create(graph.get()));
  EXPECT_TRUE(scheduler->GetFsm(sub_1_ptr).IsScheduled());
  EXPECT_TRUE(scheduler->GetFsm(sub_2_ptr).IsBlocked());
  EXPECT_TRUE(scheduler->GetFsm(sub_3_ptr).IsBlocked());

  paragraph::Subroutine* picked_subroutine;
  ASSERT_OK_AND_ASSIGN(picked_subroutine,
                       scheduler->GetFsm(call).PickSubroutine());
  EXPECT_EQ(picked_subroutine->GetName(), "subroutine_1");
  scheduler->InstructionExecuted(instr_1);

  ASSERT_OK_AND_ASSIGN(picked_subroutine,
                       scheduler->GetFsm(call).PickSubroutine());
  EXPECT_EQ(picked_subroutine->GetName(), "subroutine_2");
  scheduler->InstructionExecuted(instr_2);

  ASSERT_OK_AND_ASSIGN(picked_subroutine,
                       scheduler->GetFsm(call).PickSubroutine());
  EXPECT_EQ(picked_subroutine->GetName(), "subroutine_3");
  scheduler->InstructionExecuted(instr_3);

  EXPECT_TRUE(scheduler->GetFsm(call).IsExecuted());
}
