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
#include "paragraph/scheduling/logger.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "paragraph/graph/graph.h"
#include "paragraph/scheduling/graph_scheduler.h"
#include "paragraph/shim/test_macros.h"

std::string get_testfile_name(const std::string& basename) {
  std::string f = std::getenv("TEST_TMPDIR");
  if (f.back() != '/') {
    f += '/';
  }
  f += basename;
  return f;
}

// Tests logger creation with empty and non empty filename
TEST(Logger, Create) {
  std::filesystem::remove(get_testfile_name("logger_test.csv"));
  ASSERT_OK_AND_ASSIGN(auto logger, paragraph::Logger::Create(
      get_testfile_name("logger_test.csv")));
  EXPECT_TRUE(std::filesystem::exists(get_testfile_name("logger_test.csv")));
  std::ifstream testfile(get_testfile_name("logger_test.csv"));
  std::string header;
  EXPECT_TRUE(getline(testfile, header).good());
  EXPECT_EQ(header,
            "processor_id,instruction_name,opcode,ready,started,finished");
}

// Tests graph writing to and reading from file
TEST(Logger, FileIO) {
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

  std::filesystem::remove(get_testfile_name("logger_test.csv"));
  EXPECT_FALSE(std::filesystem::exists(get_testfile_name("logger_test.csv")));
  ASSERT_OK_AND_ASSIGN(auto logger, paragraph::Logger::Create(
      get_testfile_name("logger_test.csv")));
  EXPECT_OK(logger->AddToLog(instr_fsm));

  instr_fsm.SetTimeReady(10.1);
  instr_fsm.SetTimeStarted(20.2);
  instr_fsm.SetTimeFinished(30.123456789012345);
  EXPECT_OK(logger->AddToLog(instr_fsm));

  EXPECT_TRUE(std::filesystem::exists(get_testfile_name("logger_test.csv")));
  std::ifstream testfile(get_testfile_name("logger_test.csv"));
  std::string header, line_1, line_2, dummy;

  EXPECT_TRUE(getline(testfile, header).good());
  EXPECT_EQ(header,
            "processor_id,instruction_name,opcode,ready,started,finished");
  EXPECT_TRUE(getline(testfile, line_1).good());
  EXPECT_EQ(line_1,
            "1,dummy,delay,1.100000000000,2.200000000000,3.123456789012");
  EXPECT_TRUE(getline(testfile, line_2).good());
  EXPECT_EQ(line_2,
            "1,dummy,delay,10.100000000000,20.200000000000,30.123456789012");
  EXPECT_FALSE(getline(testfile, dummy).good());
}
