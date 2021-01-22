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
#ifndef PARAGRAPH_SCHEDULING_LOG_ENTRY_H_
#define PARAGRAPH_SCHEDULING_LOG_ENTRY_H_

#include <string>

#include "paragraph/graph/graph.h"

namespace paragraph {

class LogEntry {
 public:
  explicit LogEntry(Instruction* instruction);
  ~LogEntry() = default;

  // Getters/Setters for instruction timings
  double GetTimeReady() const;
  void SetTimeReady(double current_time);
  double GetTimeStarted() const;
  void SetTimeStarted(double current_time);
  double GetTimeFinished() const;
  void SetTimeFinished(double current_time);

  std::string ToString(const std::string & delimeter = ",") const;

 private:
  // String identifier for instruction.
  std::string name_;

  // Opcode for this instruction.
  std::string opcode_;

  // ID of processor that executes instruction
  int64_t processor_id_;

  // Instruction timings
  double time_ready_;
  double time_started_;
  double time_finished_;
};

}  // namespace paragraph

#endif  // PARAGRAPH_SCHEDULING_LOG_ENTRY_H_
