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
#include "paragraph/graph/opcode.h"

#include <memory>
#include <string>
#include <utility>

#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "paragraph/graph/instruction.h"
#include "gtest/gtest.h"
#include "paragraph/shim/test_macros.h"

// Tests StringToOpcode() function
TEST(Opcode, StringToOpcodeConversion) {
  std::string opcode_str_1 = "delay";
  shim::StatusOr<paragraph::Opcode> opcode_1 =
      paragraph::StringToOpcode(opcode_str_1);
  EXPECT_TRUE(opcode_1.ok());
  EXPECT_EQ(opcode_1.value(), paragraph::Opcode::kDelay);

  std::string opcode_str_2 = "while";
  shim::StatusOr<paragraph::Opcode> opcode_2 =
      paragraph::StringToOpcode(opcode_str_2);
  EXPECT_TRUE(opcode_2.ok());
  EXPECT_EQ(opcode_2.value(), paragraph::Opcode::kWhile);

  std::string opcode_str_3 = "all-to-all";
  shim::StatusOr<paragraph::Opcode> opcode_3 =
      paragraph::StringToOpcode(opcode_str_3);
  EXPECT_TRUE(opcode_3.ok());
  EXPECT_EQ(opcode_3.value(), paragraph::Opcode::kAllToAll);

  std::string opcode_str_4 = "recv-done";
  shim::StatusOr<paragraph::Opcode> opcode_4 =
      paragraph::StringToOpcode(opcode_str_4);
  EXPECT_TRUE(opcode_4.ok());
  EXPECT_EQ(opcode_4.value(), paragraph::Opcode::kRecvDone);

  std::string opcode_str_5 = "outfeed";
  shim::StatusOr<paragraph::Opcode> opcode_5 =
      paragraph::StringToOpcode(opcode_str_5);
  EXPECT_TRUE(opcode_5.ok());
  EXPECT_EQ(opcode_5.value(), paragraph::Opcode::kOutfeed);

  std::string opcode_str_6 = "garbage";
  shim::StatusOr<paragraph::Opcode> opcode_6 =
      paragraph::StringToOpcode(opcode_str_6);
  absl::Status badstatus(absl::StatusCode::kInvalidArgument,
      "Unknown opcode garbage in StringToOpcode conversion");
  EXPECT_FALSE(opcode_6.ok());
  EXPECT_EQ(opcode_6.status(), badstatus);
}

// Tests OpcodeToString() function
TEST(Opcode, OpcodeToStringConversion) {
  paragraph::Opcode opcode_1 = paragraph::Opcode::kDelay;
  std::string opcode_str_1 = paragraph::OpcodeToString(opcode_1);
  EXPECT_EQ(opcode_str_1, "delay");

  paragraph::Opcode opcode_2 = paragraph::Opcode::kWhile;
  std::string opcode_str_2 = paragraph::OpcodeToString(opcode_2);
  EXPECT_EQ(opcode_str_2, "while");

  paragraph::Opcode opcode_3 = paragraph::Opcode::kAllToAll;
  std::string opcode_str_3 = paragraph::OpcodeToString(opcode_3);
  EXPECT_EQ(opcode_str_3, "all-to-all");

  paragraph::Opcode opcode_4 = paragraph::Opcode::kRecvDone;
  std::string opcode_str_4 = paragraph::OpcodeToString(opcode_4);
  EXPECT_EQ(opcode_str_4, "recv-done");

  paragraph::Opcode opcode_5 = paragraph::Opcode::kOutfeed;
  std::string opcode_str_5 = paragraph::OpcodeToString(opcode_5);
  EXPECT_EQ(opcode_str_5, "outfeed");
}

// Tests OpcodeIsCollectiveCommunication() function
TEST(Opcode, IsCollectiveCommunication) {
  paragraph::Opcode test_opcode;

  test_opcode = paragraph::Opcode::kAllGather;
  EXPECT_TRUE(paragraph::OpcodeIsCollectiveCommunication(test_opcode));

  test_opcode = paragraph::Opcode::kSend;
  EXPECT_FALSE(paragraph::OpcodeIsCollectiveCommunication(test_opcode));

  test_opcode = paragraph::Opcode::kRecvDone;
  EXPECT_FALSE(paragraph::OpcodeIsCollectiveCommunication(test_opcode));

  test_opcode = paragraph::Opcode::kWhile;
  EXPECT_FALSE(paragraph::OpcodeIsCollectiveCommunication(test_opcode));

  test_opcode = paragraph::Opcode::kNull;
  EXPECT_FALSE(paragraph::OpcodeIsCollectiveCommunication(test_opcode));

  test_opcode = paragraph::Opcode::kInfeed;
  EXPECT_FALSE(paragraph::OpcodeIsCollectiveCommunication(test_opcode));
}

// Tests OpcodeIsProtocolLevelCommunication() function
TEST(Opcode, IsProtocolLevelCommunication) {
  paragraph::Opcode test_opcode;

  test_opcode = paragraph::Opcode::kAllGather;
  EXPECT_FALSE(paragraph::OpcodeIsProtocolLevelCommunication(test_opcode));

  test_opcode = paragraph::Opcode::kSend;
  EXPECT_FALSE(paragraph::OpcodeIsProtocolLevelCommunication(test_opcode));

  test_opcode = paragraph::Opcode::kRecvDone;
  EXPECT_TRUE(paragraph::OpcodeIsProtocolLevelCommunication(test_opcode));

  test_opcode = paragraph::Opcode::kWhile;
  EXPECT_FALSE(paragraph::OpcodeIsProtocolLevelCommunication(test_opcode));

  test_opcode = paragraph::Opcode::kNull;
  EXPECT_FALSE(paragraph::OpcodeIsProtocolLevelCommunication(test_opcode));

  test_opcode = paragraph::Opcode::kInfeed;
  EXPECT_FALSE(paragraph::OpcodeIsProtocolLevelCommunication(test_opcode));
}

// Tests OpcodeIsIndividualCommunication() function
TEST(Opcode, IsIndividualCommunication) {
  paragraph::Opcode test_opcode;

  test_opcode = paragraph::Opcode::kAllGather;
  EXPECT_FALSE(paragraph::OpcodeIsIndividualCommunication(test_opcode));

  test_opcode = paragraph::Opcode::kSend;
  EXPECT_TRUE(paragraph::OpcodeIsIndividualCommunication(test_opcode));

  test_opcode = paragraph::Opcode::kRecvDone;
  EXPECT_FALSE(paragraph::OpcodeIsIndividualCommunication(test_opcode));

  test_opcode = paragraph::Opcode::kWhile;
  EXPECT_FALSE(paragraph::OpcodeIsIndividualCommunication(test_opcode));

  test_opcode = paragraph::Opcode::kNull;
  EXPECT_FALSE(paragraph::OpcodeIsIndividualCommunication(test_opcode));

  test_opcode = paragraph::Opcode::kInfeed;
  EXPECT_FALSE(paragraph::OpcodeIsIndividualCommunication(test_opcode));
}

// Tests OpcodeIsControlFlow() function
TEST(Opcode, IsControlFlow) {
  paragraph::Opcode test_opcode;

  test_opcode = paragraph::Opcode::kAllGather;
  EXPECT_FALSE(paragraph::OpcodeIsControlFlow(test_opcode));

  test_opcode = paragraph::Opcode::kSend;
  EXPECT_FALSE(paragraph::OpcodeIsControlFlow(test_opcode));

  test_opcode = paragraph::Opcode::kRecvDone;
  EXPECT_FALSE(paragraph::OpcodeIsControlFlow(test_opcode));

  test_opcode = paragraph::Opcode::kWhile;
  EXPECT_TRUE(paragraph::OpcodeIsControlFlow(test_opcode));

  test_opcode = paragraph::Opcode::kNull;
  EXPECT_FALSE(paragraph::OpcodeIsControlFlow(test_opcode));

  test_opcode = paragraph::Opcode::kInfeed;
  EXPECT_FALSE(paragraph::OpcodeIsControlFlow(test_opcode));
}

// Tests OpcodeIsGeneralPurpose() function
TEST(Opcode, IsGeneralPurpose) {
  paragraph::Opcode test_opcode;

  test_opcode = paragraph::Opcode::kAllGather;
  EXPECT_FALSE(paragraph::OpcodeIsGeneralPurpose(test_opcode));

  test_opcode = paragraph::Opcode::kSend;
  EXPECT_FALSE(paragraph::OpcodeIsGeneralPurpose(test_opcode));

  test_opcode = paragraph::Opcode::kRecvDone;
  EXPECT_FALSE(paragraph::OpcodeIsGeneralPurpose(test_opcode));

  test_opcode = paragraph::Opcode::kWhile;
  EXPECT_FALSE(paragraph::OpcodeIsGeneralPurpose(test_opcode));

  test_opcode = paragraph::Opcode::kNull;
  EXPECT_TRUE(paragraph::OpcodeIsGeneralPurpose(test_opcode));

  test_opcode = paragraph::Opcode::kInfeed;
  EXPECT_TRUE(paragraph::OpcodeIsGeneralPurpose(test_opcode));
}
