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
#include "paragraph/scheduling/log_entry.h"

#include <memory>
#include <string>
#include <utility>

#include "paragraph/graph/graph.h"
#include "gtest/gtest.h"
#include "paragraph/scheduling/graph_scheduler.h"
#include "paragraph/shim/test_macros.h"

// Test Log Entry timing setters and getters
TEST(LogEntry, Timing) {
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

  auto log_entry = paragraph::LogEntry(instr);
  EXPECT_EQ(log_entry.GetTimeReady(), 0.0);
  EXPECT_EQ(log_entry.GetTimeStarted(), 0.0);
  EXPECT_EQ(log_entry.GetTimeFinished(), 0.0);

  log_entry.SetTimeReady(1.0);
  EXPECT_EQ(log_entry.GetTimeReady(), 1.0);
  EXPECT_EQ(log_entry.GetTimeStarted(), 0.0);
  EXPECT_EQ(log_entry.GetTimeFinished(), 0.0);

  log_entry.SetTimeStarted(2.0);
  EXPECT_EQ(log_entry.GetTimeReady(), 1.0);
  EXPECT_EQ(log_entry.GetTimeStarted(), 2.0);
  EXPECT_EQ(log_entry.GetTimeFinished(), 0.0);

  log_entry.SetTimeFinished(3.0);
  EXPECT_EQ(log_entry.GetTimeReady(), 1.0);
  EXPECT_EQ(log_entry.GetTimeStarted(), 2.0);
  EXPECT_EQ(log_entry.GetTimeFinished(), 3.0);
}

// Test Log Entry creation
TEST(LogEntry, ToString) {
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

  auto log_entry = paragraph::LogEntry(instr);
  log_entry.SetTimeReady(1.1);
  log_entry.SetTimeStarted(2.2);
  log_entry.SetTimeFinished(3.3);

  EXPECT_EQ(log_entry.ToString(),
            "1,dummy,delay,1.1,2.2,3.3");
}
