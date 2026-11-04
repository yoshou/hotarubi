#include "sparsedrive_parts.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>

#include "isa.hpp"
#include "program.hpp"

namespace hotarubi::compiler::sparsedrive {
namespace {

constexpr uint64_t kBase = 0x38700000;

TEST(SparseDrivePartsTest, TokenSizesFollowTheRowFormat) {
  EXPECT_EQ(TokenBytes(Mem::kAbuf, 1), 48U);
  EXPECT_EQ(TokenBytes(Mem::kAbuf, 8), 272U);
  EXPECT_EQ(TokenBytes(Mem::kAbuf, 16), 528U);
  EXPECT_EQ(TokenBytes(Mem::kAbuf, 17), 576U);
  EXPECT_EQ(TokenBytes(Mem::kFbuf, 2), 256U);
}

TEST(SparseDrivePartsTest, ListEntriesAreAddressesInSixteenByteUnits) {
  EXPECT_EQ(Entry(kBase, 0x1230), 0x3870123U);
  EXPECT_EQ(Entry(0, 0), 0U);
}

TEST(SparseDrivePartsDeathTest, ListEntriesRejectUnalignedAddresses) {
  EXPECT_DEATH(static_cast<void>(Entry(kBase, 0x1238)),
               "is not a 16-byte address");
}

TEST(SparseDrivePartsTest, TokenTransfersUseTheRowFormat) {
  Program program(kBase);
  const size_t index =
      program.Load(Tokens(0x2000, Mem::kAbuf, 8, 12, 8, true), "tokens");
  const Instruction& instruction = program.Instructions()[index];
  EXPECT_EQ(instruction.Get(120, 8), 1U);
  EXPECT_EQ(instruction.Get(112, 4), 3U);
  EXPECT_EQ(instruction.Get(80, 32), kBase + 0x2000);
  EXPECT_EQ(instruction.Get(77, 3), 0U);
  EXPECT_EQ(instruction.Get(57, 20), 8U);
  EXPECT_EQ(instruction.Get(45, 12), 1U);
  EXPECT_EQ(instruction.Get(33, 12), 12U);
  EXPECT_EQ(instruction.Get(0, 7), 8U);
}

TEST(SparseDrivePartsTest, WordTransfersAreOneRawRow) {
  Program program(kBase);
  const size_t index = program.Store(Words(0x3000, Mem::kFbuf, 40, 5), "words");
  const Instruction& instruction = program.Instructions()[index];
  EXPECT_EQ(instruction.Get(120, 8), 2U);
  EXPECT_EQ(instruction.Get(112, 4), 0U);
  EXPECT_EQ(instruction.Get(77, 3), 2U);
  EXPECT_EQ(instruction.Get(57, 20), 40U);
  EXPECT_EQ(instruction.Get(45, 12), 1U);
  EXPECT_EQ(instruction.Get(33, 12), 5U);
  EXPECT_EQ(instruction.Get(7, 2), 0U);
}

TEST(SparseDrivePartsTest, RequiredReturnsThePresentValue) {
  const std::optional<int> present = 3;
  EXPECT_EQ(Required(present, "value"), 3);
}

TEST(SparseDrivePartsDeathTest, RequiredRejectsAMissingValue) {
  const std::optional<int> missing;
  EXPECT_DEATH(static_cast<void>(Required(missing, "value")),
               "value is missing");
}

}  // namespace
}  // namespace hotarubi::compiler::sparsedrive
