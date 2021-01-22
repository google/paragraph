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

#include <string>

#include "absl/strings/str_cat.h"
#include "paragraph/shim/macros.h"

namespace paragraph {

LogEntry::LogEntry(Instruction* instruction)
    : name_(instruction->GetName()),
      opcode_(OpcodeToString(instruction->GetOpcode())),
      time_ready_(0.0),
      time_started_(0.0),
      time_finished_(0.0) {
  CHECK(instruction->GetGraph() != nullptr);
  processor_id_ = instruction->GetGraph()->GetProcessorId();
}

double LogEntry::GetTimeReady() const {
  return time_ready_;
}

void LogEntry::SetTimeReady(double current_time) {
  time_ready_ = current_time;
}

double LogEntry::GetTimeStarted() const {
  return time_started_;
}

void LogEntry::SetTimeStarted(double current_time) {
  time_started_ = current_time;
}

double LogEntry::GetTimeFinished() const {
  return time_finished_;
}

void LogEntry::SetTimeFinished(double current_time) {
  time_finished_ = current_time;
}

std::string LogEntry::ToString(const std::string & delimeter) const {
  std::string log_entry_str = absl::StrCat(
      processor_id_, delimeter,
      name_, delimeter,
      opcode_, delimeter,
      absl::SixDigits(time_ready_), delimeter,
      absl::SixDigits(time_started_), delimeter,
      absl::SixDigits(time_finished_));
  return log_entry_str;
}

}  // namespace paragraph
