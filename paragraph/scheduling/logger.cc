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
#include <sstream>
#include <string>

#include "paragraph/graph/graph.h"
#include "paragraph/scheduling/instruction_fsm.h"

namespace paragraph {

shim::StatusOr<std::unique_ptr<Logger>> Logger::Create(
    const std::string& filename) {
  auto logger = absl::WrapUnique(new Logger(filename));
  if (filename != "") {
    RETURN_IF_ERROR(logger->OpenFile());
    RETURN_IF_ERROR(logger->InitializeCsv());
  }
  return logger;
}

void Logger::FlushToFile() {
  if (filename_ != "") {
    if (ostream_.is_open()) {
      ostream_.close();
    }
  }
}

bool Logger::IsAvailable() {
  if (ostream_.is_open()) {
    return true;
  } else if (filename_ != "") {
    return false;
  } else if (OpenFile().ok()) {
    return true;
  } else {
    return false;
  }
}

absl::Status Logger::SetFilename(const std::string& filename) {
  FlushToFile();
  filename_ = filename;
  RETURN_IF_ERROR(OpenFile());
  RETURN_IF_ERROR(InitializeCsv());
  return absl::OkStatus();
}

absl::Status Logger::AppendToCsv(const InstructionFsm& instruction_fsm) {
  RETURN_IF_ERROR(OpenFile());
  ostream_ << MakeCsvLine(instruction_fsm) << std::endl;
  if (ostream_.fail() || ostream_.bad()) {
    return absl::InternalError(
        "Failed to write trace to CSV file: " + filename_);
  }
  return absl::OkStatus();
}

Logger::Logger(const std::string& filename)
    : filename_(filename) {
  std::fstream ostream_;
}

absl::Status Logger::OpenFile() {
  if (!ostream_.is_open()) {
    if (filename_ == "") {
      return absl::InvalidArgumentError(
          "File '" + filename_ + "' could not be opened.");
    }
    ostream_.open(filename_, std::ios::out | std::ios::trunc);
    RETURN_IF_FALSE(ostream_.is_open(), absl::InternalError) <<
        "File '" << filename_ << "' could not be opened.";
  }
  return absl::OkStatus();
}

absl::Status Logger::InitializeCsv() {
  if (ostream_.is_open()) {
    ostream_ << "processor_id,instruction_name,opcode,ready,started,finished";
    ostream_ << std::endl;
  } else {
    return absl::InternalError("Can't initialize CSV File '" + filename_ +
                               "': file could not be opened.");
  }
  return absl::OkStatus();
}

std::string Logger::MakeCsvLine(const InstructionFsm& fsm,
                                const std::string& delimiter) {
  CHECK(fsm.GetInstruction()->GetGraph() != nullptr);
  std::ostringstream str_stream;
  // We store 12 decimal digits of precision, which is equivalent to
  // 1 picosecond precision for each second of time
  str_stream << std::fixed << std::setprecision(12);
  str_stream << fsm.GetTimeReady();
  std::string time_ready_str = str_stream.str();
  str_stream.str("");
  str_stream << fsm.GetTimeStarted();
  std::string time_started_str = str_stream.str();
  str_stream.str("");
  str_stream << fsm.GetTimeFinished();
  std::string time_finished_str = str_stream.str();
  std::string log_entry_str = absl::StrCat(
      fsm.GetInstruction()->GetGraph()->GetProcessorId(), delimiter,
      fsm.GetInstruction()->GetName(), delimiter,
      OpcodeToString(fsm.GetInstruction()->GetOpcode()), delimiter,
      time_ready_str, delimiter,
      time_started_str, delimiter,
      time_finished_str);
  return log_entry_str;
}

}  // namespace paragraph