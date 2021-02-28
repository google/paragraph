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
#include "paragraph/simulator/simple_sim.h"

#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "paragraph/graph/graph.h"
#include "paragraph/scheduling/graph_scheduler.h"
#include "paragraph/shim/test_macros.h"
#include "paragraph/translation/translation.h"
#include "paragraph/translation/translation_map.h"

std::string log_filename(const std::string& basename) {
  std::string f = std::getenv("TEST_TMPDIR");
  if (f.back() != '/') {
    f += '/';
  }
  f += basename;
  return f;
}

// Tests simulator on a simple while loop
TEST(SimpleSim, WhileLoop) {
  auto graph = absl::make_unique<paragraph::Graph>("test_graph");
  auto sub = absl::make_unique<paragraph::Subroutine>(
      "test_subroutine", graph.get());
  auto sub_ptr = sub.get();
  graph->SetEntrySubroutine(std::move(sub));

  ASSERT_OK_AND_ASSIGN(auto instr_1, paragraph::Instruction::Create(
      paragraph::Opcode::kInfeed, "compute_pred1", sub_ptr));
  instr_1->SetBytesOut(36.5);
  instr_1->SetSeconds(0.01);

  ASSERT_OK_AND_ASSIGN(auto instr_2, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "reduction_operand1", sub_ptr));
  instr_2->SetOps(128.);
  instr_2->SetSeconds(0.05);
  instr_2->AddOperand(instr_1);

  ASSERT_OK_AND_ASSIGN(auto while_instr, paragraph::Instruction::Create(
      paragraph::Opcode::kWhile, "test", sub_ptr));
  auto body_sub = absl::make_unique<paragraph::Subroutine>(
      "body_subroutine", graph.get());
  auto body_sub_ptr = body_sub.get();
  body_sub_ptr->SetExecutionCount(2);
  ASSERT_OK_AND_ASSIGN(auto body_compute, paragraph::Instruction::Create(
      paragraph::Opcode::kDelay, "body_compute", body_sub_ptr, true));
  body_compute->SetTranscendentals(111.);
  body_compute->SetSeconds(0.004);
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
  paragraph::CommunicationGroup send_group = {1, 42};
  send_instr->SetBytesIn(10);
  send_instr->SetSeconds(0.07);
  send_instr->AppendCommunicationGroup(send_group);

  nlohmann::json translation_config = R"(
    {
      "protocol": {
        "send": {
          "algorithm": "push"
        }
      }
    }
  )"_json;

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<paragraph::Graph>> translated_graphs,
      paragraph::IndividualizeAndTranslate(graph.get(), translation_config));


  std::filesystem::remove(log_filename("logger_test.csv"));
  EXPECT_FALSE(std::filesystem::exists(log_filename("logger_test.csv")));
  ASSERT_OK_AND_ASSIGN(auto logger, paragraph::Logger::Create(
      log_filename("logger_test.csv")));

  auto perf = paragraph::SimpleSim::PerformanceParameters(1, 1, 1000);
  ASSERT_OK_AND_ASSIGN(auto sim, paragraph::SimpleSim::Create(
      std::move(translated_graphs.at(0)), perf, std::move(logger)));
  EXPECT_OK(sim->StartSimulation(1));

  EXPECT_TRUE(std::filesystem::exists(log_filename("logger_test.csv")));
  std::ifstream testfile(log_filename("logger_test.csv"));
  std::string header, line;

  EXPECT_TRUE(getline(testfile, header).good());
  EXPECT_EQ(header,
            "processor_id,instruction_name,opcode,"
            "ready,started,finished,clock,wall");
  std::vector<std::string> test_results = {
    "1,compute_pred1,infeed,1.000000000000,1.000000000000,"
        "1.010000000000,0.010000000000,0.010000000000",
    "1,body_compute,delay,1.000000000000,1.010000000000,"
        "1.014000000000,0.004000000000,0.004000000000",
    "1,send_sendstart,send-start,1.000000000000,1.000000000000,"
        "1.010000000000,0.010000000000,0.010000000000",
    "1,reduction_operand1,delay,1.010000000000,1.014000000000,"
        "1.064000000000,0.050000000000,0.050000000000",
    "1,call_func,delay,1.014000000000,1.064000000000,"
        "1.065000000000,0.001000000000,0.001000000000",
    "1,cond_call,call,1.014000000000,1.064000000000,"
        "1.065000000000,0.001000000000,0.001000000000",
    "1,send_senddone,send-done,1.010000000000,1.010000000000,"
        "1.010000000000,0.000000000000,0.000000000000",
    "1,send,send,1.000000000000,1.000000000000,"
        "1.010000000000,0.010000000000,0.010000000000",
    "1,body_compute,delay,1.065000000000,1.065000000000,"
        "1.069000000000,0.004000000000,0.004000000000",
    "1,call_func,delay,1.069000000000,1.069000000000,"
        "1.070000000000,0.001000000000,0.001000000000",
    "1,cond_call,call,1.069000000000,1.069000000000,"
        "1.070000000000,0.001000000000,0.001000000000",
    "1,test,while,1.000000000000,1.010000000000,"
        "1.070000000000,0.010000000000,0.060000000000"
  };

  int counter = 0;
  while (getline(testfile, line)) {
    EXPECT_EQ(line, test_results.at(counter));
    counter++;
  }
  EXPECT_DOUBLE_EQ(sim->GetProcessorTime(), 1.07);
}
