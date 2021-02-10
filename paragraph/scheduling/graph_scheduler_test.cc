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
#include "paragraph/scheduling/graph_scheduler.h"

#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "gtest/gtest.h"
#include "paragraph/graph/graph.h"
#include "paragraph/shim/test_macros.h"

std::string testfile_name(const std::string& basename) {
  std::string f = std::getenv("TEST_TMPDIR");
  if (f.back() != '/') {
    f += '/';
  }
  f += basename;
  return f;
}

// Tests scheduler creation and access to instructions/subroutines FSMs
TEST(Scheduler, Timing) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kInfeed, "compute_pred1", sub_ptr, true));

  ASSERT_OK_AND_ASSIGN(auto scheduler,
                       paragraph::GraphScheduler::Create(graph.get()));
  EXPECT_EQ(scheduler->GetCurrentTime(), 0.0);

  CHECK_OK(scheduler->Initialize(10.0));
  EXPECT_EQ(scheduler->GetCurrentTime(), 10.0);

  scheduler->InstructionStarted(instr_1, 30.0);
  EXPECT_EQ(scheduler->GetCurrentTime(), 30.0);

  scheduler->InstructionFinished(instr_1, 40.0);
  EXPECT_EQ(scheduler->GetCurrentTime(), 40.0);
}

// Tests scheduler creation and access to instructions/subroutines FSMs
TEST(Scheduler, Creation) {
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
  auto body_sub = absl::make_unique<paragraph::Subroutine>(
      "body_subroutine", graph.get());
  auto body_sub_ptr = body_sub.get();
  ASSERT_OK_AND_ASSIGN(auto body_compute, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "body_compute", body_sub_ptr, true));
  body_compute->SetTranscendentals(111.);
  while_instr->AppendInnerSubroutine(std::move(body_sub));

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
  while_instr->AppendInnerSubroutine(std::move(cond_sub));

  ASSERT_OK_AND_ASSIGN(auto send_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kSend, "send", sub_ptr, true));
  paragraph::CommunicationGroup send_group = {42};
  send_instr->SetBytesIn(8.);
  send_instr->AppendCommunicationGroup(send_group);

  ASSERT_OK_AND_ASSIGN(auto scheduler,
                       paragraph::GraphScheduler::Create(graph.get()));
  CHECK_OK(scheduler->Initialize(0.0));

  EXPECT_TRUE(scheduler->GetFsm(sub_ptr).IsScheduled());
  EXPECT_TRUE(scheduler->GetFsm(body_sub_ptr).IsScheduled());
  EXPECT_TRUE(scheduler->GetFsm(cond_sub_ptr).IsBlocked());
  EXPECT_TRUE(scheduler->GetFsm(instr_1).IsReady());
  EXPECT_TRUE(scheduler->GetFsm(instr_2).IsBlocked());
  EXPECT_TRUE(scheduler->GetFsm(instr_3).IsBlocked());
  EXPECT_TRUE(scheduler->GetFsm(while_instr).IsScheduled());
  EXPECT_TRUE(scheduler->GetFsm(body_compute).IsReady());
  EXPECT_TRUE(scheduler->GetFsm(cond_call).IsReady());
  EXPECT_TRUE(scheduler->GetFsm(call_func).IsReady());
  EXPECT_TRUE(scheduler->GetFsm(send_instr).IsReady());
}

// Tests scheduler creation and access to instructions/subroutines FSMs
// for a graph with While instruction that executes its subroutines 2 times, and
// also has some nested subroutines
TEST(Scheduler, WhileInstruction) {
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
  auto body_sub = absl::make_unique<paragraph::Subroutine>(
      "body_subroutine", graph.get());
  auto body_sub_ptr = body_sub.get();
  body_sub_ptr->SetExecutionCount(2);
  ASSERT_OK_AND_ASSIGN(auto body_compute, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "body_compute", body_sub_ptr, true));
  body_compute->SetTranscendentals(111.);
  while_instr->AppendInnerSubroutine(std::move(body_sub));

  auto call_sub = absl::make_unique<paragraph::Subroutine>(
      "call_subroutine", graph.get());
  auto call_sub_ptr = call_sub.get();
  ASSERT_OK_AND_ASSIGN(auto call_func, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "call_func", call_sub_ptr, true));
  call_func->SetSeconds(0.001);

  auto cond_sub = absl::make_unique<paragraph::Subroutine>(
      "cond_subroutine", graph.get());
  auto cond_sub_ptr = cond_sub.get();
  cond_sub_ptr->SetExecutionCount(2);
  ASSERT_OK_AND_ASSIGN(auto cond_call, paragraph::Instruction::Create(
      paragraph::Opcode::kCall, "cond_call", cond_sub_ptr, true));
  cond_call->AppendInnerSubroutine(std::move(call_sub));
  while_instr->AppendInnerSubroutine(std::move(cond_sub));

  ASSERT_OK_AND_ASSIGN(auto send_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kSend, "send", sub_ptr, true));
  paragraph::CommunicationGroup send_group = {42};
  send_instr->SetBytesIn(8.);
  send_instr->AppendCommunicationGroup(send_group);

  ASSERT_OK_AND_ASSIGN(auto scheduler,
                       paragraph::GraphScheduler::Create(graph.get()));
  CHECK_OK(scheduler->Initialize(0.0));

  auto consumed_instructions = scheduler->GetReadyInstructions();
  EXPECT_EQ(consumed_instructions.size(), 3);
  EXPECT_EQ(consumed_instructions.at(0)->GetName(), "compute_pred1");
  EXPECT_EQ(consumed_instructions.at(1)->GetName(), "body_compute");
  EXPECT_EQ(consumed_instructions.at(2)->GetName(), "send");

  scheduler->InstructionStarted(consumed_instructions.at(2), 0.0);
  scheduler->InstructionFinished(consumed_instructions.at(2), 0.0);
  EXPECT_EQ(scheduler->GetReadyInstructions().size(), 0);

  scheduler->InstructionStarted(consumed_instructions.at(0), 0.0);
  scheduler->InstructionFinished(consumed_instructions.at(0), 0.0);
  auto reduction_line = scheduler->GetReadyInstructions();
  EXPECT_EQ(reduction_line.size(), 1);
  EXPECT_EQ(reduction_line.at(0)->GetName(), "reduction_operand1");

  scheduler->InstructionStarted(reduction_line.at(0), 0.0);
  scheduler->InstructionFinished(reduction_line.at(0), 0.0);
  reduction_line = scheduler->GetReadyInstructions();
  EXPECT_EQ(reduction_line.size(), 1);
  EXPECT_EQ(reduction_line.at(0)->GetName(), "reduction");

  scheduler->InstructionStarted(reduction_line.at(0), 0.0);
  scheduler->InstructionFinished(reduction_line.at(0), 0.0);
  reduction_line = scheduler->GetReadyInstructions();
  EXPECT_EQ(reduction_line.size(), 0);

  scheduler->InstructionStarted(consumed_instructions.at(1), 0.0);
  scheduler->InstructionFinished(consumed_instructions.at(1), 0.0);
  consumed_instructions = scheduler->GetReadyInstructions();
  EXPECT_EQ(consumed_instructions.size(), 1);
  EXPECT_EQ(consumed_instructions.at(0)->GetName(), "call_func");

  scheduler->InstructionStarted(consumed_instructions.at(0), 0.0);
  scheduler->InstructionFinished(consumed_instructions.at(0), 0.0);
  consumed_instructions = scheduler->GetReadyInstructions();
  EXPECT_EQ(consumed_instructions.size(), 1);
  EXPECT_EQ(consumed_instructions.at(0)->GetName(), "body_compute");

  scheduler->InstructionStarted(consumed_instructions.at(0), 0.0);
  scheduler->InstructionFinished(consumed_instructions.at(0), 0.0);
  consumed_instructions = scheduler->GetReadyInstructions();
  EXPECT_EQ(consumed_instructions.size(), 1);
  EXPECT_EQ(consumed_instructions.at(0)->GetName(), "call_func");

  scheduler->InstructionStarted(consumed_instructions.at(0), 0.0);
  scheduler->InstructionFinished(consumed_instructions.at(0), 0.0);
  consumed_instructions = scheduler->GetReadyInstructions();
  EXPECT_EQ(consumed_instructions.size(), 0);
}

// Tests scheduler creation and access to instructions/subroutines FSMs
// for a graph with Null instruction
TEST(Scheduler, NullInstruction) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "first_instruction", sub_ptr));
  instr_1->SetOps(4);

  ASSERT_OK_AND_ASSIGN(auto allreduce, paragraph::Instruction::Create(
      paragraph::Opcode::kAllReduce, "all-reduce", sub_ptr));
  allreduce->SetBytesOut(48);
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
  ASSERT_OK_AND_ASSIGN(auto root, paragraph::Instruction::Create(
      paragraph::Opcode::kNull, "root", reduction_ptr, true));
  root->AddOperand(op1);
  root->AddOperand(op2);
  allreduce->AppendInnerSubroutine(std::move(reduction_sub));

  ASSERT_OK_AND_ASSIGN(auto last, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "last_instruction", sub_ptr, true));
  last->AddOperand(allreduce);

  ASSERT_OK_AND_ASSIGN(auto scheduler,
                       paragraph::GraphScheduler::Create(graph.get()));
  CHECK_OK(scheduler->Initialize(0.0));

  auto consumed_instructions = scheduler->GetReadyInstructions();
  EXPECT_EQ(consumed_instructions.size(), 3);
  EXPECT_EQ(consumed_instructions.at(0)->GetName(), "first_instruction");
  EXPECT_EQ(consumed_instructions.at(1)->GetName(), "op1");
  EXPECT_EQ(consumed_instructions.at(2)->GetName(), "op2");

  scheduler->InstructionStarted(consumed_instructions.at(0), 0.0);
  scheduler->InstructionFinished(consumed_instructions.at(0), 0.0);
  EXPECT_EQ(scheduler->GetReadyInstructions().size(), 0);

  scheduler->InstructionStarted(consumed_instructions.at(1), 0.0);
  scheduler->InstructionFinished(consumed_instructions.at(1), 0.0);
  EXPECT_EQ(scheduler->GetReadyInstructions().size(), 0);

  scheduler->InstructionStarted(consumed_instructions.at(2), 0.0);
  scheduler->InstructionFinished(consumed_instructions.at(2), 0.0);
  consumed_instructions = scheduler->GetReadyInstructions();
  EXPECT_EQ(consumed_instructions.size(), 1);
  EXPECT_EQ(consumed_instructions.at(0)->GetName(), "last_instruction");
}

// Tests scheduler creation and access to instructions/subroutines FSMs
// for a graph with instructions that have subroutines that could be scheduled
// in parallel
TEST(Scheduler, ParallelSubroutines) {
  paragraph::GraphProto test_graph_proto;
  std::string test_graph_str =
      R"proto(
name: "test_graph"
entry_subroutine {
  name: "test_subroutine"
  subroutine_root_id: 7
  execution_probability: 1
  execution_count: 1
  instructions {
    name: "first_instruction"
    opcode: "delay"
    instruction_id: 1
    ops: 4
    seconds: 0.000001
  }
  instructions {
    name: "my-red"
    opcode: "all-reduce"
    instruction_id: 2
    bytes_out: 48
    operand_ids: 1
    communication_groups {
      group_ids: 0
      group_ids: 1
    }
    inner_subroutines {
      name: "reduction_subroutine"
      subroutine_root_id: 5
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "op1"
        opcode: "delay"
        instruction_id: 3
        bytes_out: 16
        seconds: 0
      }
      instructions {
        name: "op2"
        opcode: "delay"
        instruction_id: 4
        bytes_out: 16
        seconds: 0
      }
      instructions {
        name: "sum"
        opcode: "delay"
        instruction_id: 5
        ops: 32
        operand_ids: 3
        operand_ids: 4
        seconds: 0.00001
      }
    }
  }
  instructions {
    name: "call_func"
    opcode: "call"
    instruction_id: 6
    operand_ids: 1
    inner_subroutines {
      name: "call_subroutine"
      subroutine_root_id: 8
      execution_probability: 1
      execution_count: 1
      instructions {
        name: "called_func"
        opcode: "delay"
        instruction_id: 8
        seconds: 0.001
      }
    }
  }
  instructions {
    name: "last_instruction"
    opcode: "delay"
    instruction_id: 7
    operand_ids: 2
    operand_ids: 6
    ops: 4
    seconds: 0.0000001
  }
}
      )proto";
  google::protobuf::TextFormat::ParseFromString(test_graph_str,
                                                &test_graph_proto);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<paragraph::Graph> graph,
      paragraph::Graph::CreateFromProto(test_graph_proto));

  ASSERT_OK_AND_ASSIGN(auto scheduler,
                       paragraph::GraphScheduler::Create(graph.get()));
  CHECK_OK(scheduler->Initialize(0.0));

  auto consumed_instructions = scheduler->GetReadyInstructions();
  EXPECT_EQ(consumed_instructions.size(), 1);
  EXPECT_EQ(consumed_instructions.at(0)->GetName(), "first_instruction");
  scheduler->InstructionStarted(consumed_instructions.at(0), 0.0);
  scheduler->InstructionFinished(consumed_instructions.at(0), 0.0);

  consumed_instructions = scheduler->GetReadyInstructions();
  EXPECT_EQ(consumed_instructions.size(), 3);
  EXPECT_EQ(consumed_instructions.at(0)->GetName(), "op1");
  EXPECT_EQ(consumed_instructions.at(1)->GetName(), "op2");
  EXPECT_EQ(consumed_instructions.at(2)->GetName(), "called_func");

  scheduler->InstructionStarted(consumed_instructions.at(0), 0.0);
  scheduler->InstructionFinished(consumed_instructions.at(0), 0.0);
  EXPECT_EQ(scheduler->GetReadyInstructions().size(), 0);

  scheduler->InstructionStarted(consumed_instructions.at(2), 0.0);
  scheduler->InstructionFinished(consumed_instructions.at(2), 0.0);
  EXPECT_EQ(scheduler->GetReadyInstructions().size(), 0);

  scheduler->InstructionStarted(consumed_instructions.at(1), 0.0);
  scheduler->InstructionFinished(consumed_instructions.at(1), 0.0);
  consumed_instructions = scheduler->GetReadyInstructions();
  EXPECT_EQ(consumed_instructions.size(), 1);
  EXPECT_EQ(consumed_instructions.at(0)->GetName(), "sum");

  scheduler->InstructionStarted(consumed_instructions.at(0), 0.0);
  scheduler->InstructionFinished(consumed_instructions.at(0), 0.0);
  consumed_instructions = scheduler->GetReadyInstructions();
  EXPECT_EQ(consumed_instructions.size(), 1);
  EXPECT_EQ(consumed_instructions.at(0)->GetName(), "last_instruction");
}

// Tests the queue-based GetReadyInstructions
TEST(Scheduler, GetReadyInstructionQueue) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "first_instruction", sub_ptr));
  instr_1->SetOps(4);

  ASSERT_OK_AND_ASSIGN(auto allreduce, paragraph::Instruction::Create(
      paragraph::Opcode::kAllReduce, "all-reduce", sub_ptr));
  allreduce->SetBytesOut(48);
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
  ASSERT_OK_AND_ASSIGN(auto root, paragraph::Instruction::Create(
      paragraph::Opcode::kNull, "root", reduction_ptr, true));
  root->AddOperand(op1);
  root->AddOperand(op2);
  allreduce->AppendInnerSubroutine(std::move(reduction_sub));

  ASSERT_OK_AND_ASSIGN(auto last, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "last_instruction", sub_ptr, true));
  last->AddOperand(allreduce);

  ASSERT_OK_AND_ASSIGN(auto scheduler,
                       paragraph::GraphScheduler::Create(graph.get()));
  CHECK_OK(scheduler->Initialize(0.0));

  std::queue<paragraph::Instruction*> queue;
  scheduler->GetReadyInstructions(queue);
  EXPECT_EQ(queue.size(), 3);
  EXPECT_EQ(queue.front()->GetName(), "first_instruction");
  queue.pop();
  EXPECT_EQ(queue.front()->GetName(), "op1");
  queue.pop();
  EXPECT_EQ(queue.front()->GetName(), "op2");
  queue.pop();
}

// Tests interaction with logger
TEST(Scheduler, LoggerIO) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));
  ASSERT_OK_AND_ASSIGN(auto instr, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "dummy", sub_ptr, true));

  ASSERT_OK_AND_ASSIGN(auto scheduler,
                       paragraph::GraphScheduler::Create(graph.get()));
  CHECK_OK(scheduler->Initialize(0.0));

  auto instr_fsm = scheduler->GetFsm(instr);
  instr_fsm.SetTimeReady(1.1);
  instr_fsm.SetTimeStarted(2.2);
  instr_fsm.SetTimeFinished(3.123456789012345);

  std::filesystem::remove(testfile_name("logger_test.csv"));
  EXPECT_FALSE(std::filesystem::exists(testfile_name("logger_test.csv")));
  ASSERT_OK_AND_ASSIGN(auto logger, paragraph::Logger::Create(
      testfile_name("logger_test.csv")));
  scheduler->SetLogger(std::move(logger));

  scheduler->InstructionStarted(instr, 20.2);
  scheduler->InstructionFinished(instr, 30.123456789012345);

  EXPECT_TRUE(std::filesystem::exists(testfile_name("logger_test.csv")));
  std::ifstream testfile(testfile_name("logger_test.csv"));
  std::string header, line_1, dummy;

  EXPECT_TRUE(getline(testfile, header).good());
  EXPECT_EQ(header,
            "processor_id,instruction_name,opcode,ready,started,finished");
  EXPECT_TRUE(getline(testfile, line_1).good());
  EXPECT_EQ(line_1,
            "1,dummy,delay,0.000000000000,20.200000000000,30.123456789012");
  EXPECT_FALSE(getline(testfile, dummy).good());
}

// Tests logger change
TEST(GraphScheduler, LoggerChange) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph", 1);
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));
  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "dummy_1", sub_ptr, true));
  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "dummy_2", sub_ptr, true));
  instr_2->AddOperand(instr_1);

  std::filesystem::remove(testfile_name("logger_test_1.csv"));
  EXPECT_FALSE(std::filesystem::exists(testfile_name("logger_test_1.csv")));
  ASSERT_OK_AND_ASSIGN(auto logger, paragraph::Logger::Create(
      testfile_name("logger_test_1.csv")));

  ASSERT_OK_AND_ASSIGN(auto scheduler,
                       paragraph::GraphScheduler::Create(graph.get(),
                                                         std::move(logger)));
  CHECK_OK(scheduler->Initialize(1.1));

  scheduler->InstructionStarted(instr_1, 2.2);
  scheduler->InstructionFinished(instr_1, 3.123456789012345);

  EXPECT_TRUE(std::filesystem::exists(testfile_name("logger_test_1.csv")));
  std::ifstream testfile_1(testfile_name("logger_test_1.csv"));
  std::string header, line_1, line_2, dummy;

  EXPECT_TRUE(getline(testfile_1, header).good());
  EXPECT_EQ(header,
            "processor_id,instruction_name,opcode,ready,started,finished");
  EXPECT_TRUE(getline(testfile_1, line_1).good());
  EXPECT_EQ(line_1,
            "1,dummy_1,delay,1.100000000000,2.200000000000,3.123456789012");
  EXPECT_FALSE(getline(testfile_1, dummy).good());

  std::filesystem::remove(testfile_name("logger_test_2.csv"));
  EXPECT_FALSE(std::filesystem::exists(testfile_name("logger_test_2.csv")));
  ASSERT_OK_AND_ASSIGN(auto new_logger, paragraph::Logger::Create(
      testfile_name("logger_test_2.csv")));
  scheduler->SetLogger(std::move(new_logger));

  scheduler->InstructionStarted(instr_2, 20.2);
  scheduler->InstructionFinished(instr_2, 30.123456789012345);

  EXPECT_TRUE(std::filesystem::exists(testfile_name("logger_test_2.csv")));
  std::ifstream testfile_2(testfile_name("logger_test_2.csv"));

  EXPECT_TRUE(getline(testfile_2, header).good());
  EXPECT_EQ(header,
            "processor_id,instruction_name,opcode,ready,started,finished");
  EXPECT_TRUE(getline(testfile_2, line_2).good());
  EXPECT_EQ(line_2,
            "1,dummy_2,delay,3.123456789012,20.200000000000,30.123456789012");
  EXPECT_FALSE(getline(testfile_2, dummy).good());
}
