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
#ifndef PARAGRAPH_SCHEDULING_LOGGER_H_
#define PARAGRAPH_SCHEDULING_LOGGER_H_

#include <fstream>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "paragraph/scheduling/instruction_fsm.h"

namespace paragraph {

class Logger {
 public:
  // Logger factory that creates a Logger object given a filename with default
  // empty string. If non-empty file name is given, Logger opens file to append
  // to it. If it fails, it return fail status.
  static shim::StatusOr<std::unique_ptr<Logger>> Create(
      const std::string& filename = "");
  ~Logger() = default;

  // Flushes log to file and closes file
  void FlushToFile();

  // Checks if the logger was created with a non-empty filename and can try
  // to store logs to CSV file
  bool IsAvailable();

  // Set filename for the log
  absl::Status SetFilename(const std::string& filename);

  // Write log to CSV file
  absl::Status AppendToCsv(const InstructionFsm& instruction_fsm);

 private:
  // Private constructor so we can handle file access errors during Logger
  // creation
  explicit Logger(const std::string& filename = "");

  // Name of the CSV file to append log to
  std::string filename_;

  // Filestream for the CSV log
  std::fstream ostream_;

  // Checks if Logger file is open and if not tries to open it
  absl::Status OpenFile();

  // Initializes CSV file according to logger format
  absl::Status InitializeCsv();
};

}  // namespace paragraph

#endif  // PARAGRAPH_SCHEDULING_LOGGER_H_
