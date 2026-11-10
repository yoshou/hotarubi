#include "sparsedrive_layers.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "isa.hpp"
#include "program.hpp"
#include "sparsedrive_parts.hpp"

namespace hotarubi::compiler::sparsedrive {
namespace {

constexpr uint64_t kBase = 0x38700000;
constexpr uint32_t kOpDmaLoad = 1;
constexpr uint32_t kOpGemm = 4;
constexpr uint32_t kOpVec = 5;
constexpr uint64_t kRegion = 0x10000;

AttentionPlan SmallPlan() {
  return {.q = kRegion,
          .k = 2 * kRegion,
          .v = 3 * kRegion,
          .o = 4 * kRegion,
          .queries = 96,
          .embed = 256,
          .hd = 64,
          .keys = 64,
          .valid = 60};
}

std::vector<Instruction> Emitted(const AttentionPlan& plan) {
  Program program(kBase);
  EmitAttention(program, plan, "attention");
  return program.Instructions();
}

uint32_t Opcode(const Instruction& instruction) {
  return static_cast<uint32_t>(instruction.Get(120, 8));
}

constexpr size_t kChunks = 3;

TEST(SparseDriveLayersTest, TilesCoverTheRows) {
  const std::vector<Tile> tiles = Tiles(100, 32);
  ASSERT_EQ(tiles.size(), 4U);
  EXPECT_EQ(tiles[0].i0, 0U);
  EXPECT_EQ(tiles[0].m, 32U);
  EXPECT_EQ(tiles[3].t, 3U);
  EXPECT_EQ(tiles[3].i0, 96U);
  EXPECT_EQ(tiles[3].m, 4U);
}

TEST(SparseDriveLayersTest, OnChipTakesConsecutiveWords) {
  OnChip area(100);
  EXPECT_EQ(area.Take(30), 0U);
  EXPECT_EQ(area.Take(70), 30U);
}

TEST(SparseDriveLayersDeathTest, OnChipRejectsMoreThanItsSize) {
  OnChip area(100);
  area.Take(60);
  EXPECT_DEATH(area.Take(41), "cannot take 41");
}

TEST(SparseDriveLayersTest, RowsAreFp32Tokens) {
  const Rows rows{.name = "rows", .n = 10, .cols = 256, .off = 4096};
  EXPECT_EQ(rows.Tok(), 8U);
  EXPECT_EQ(rows.Tb(), TokenBytes(Mem::kFbuf, rows.Tok()));
  EXPECT_EQ(rows.Addr(3), 4096U + (3U * 1024U));
}

TEST(SparseDriveLayersTest, AttentionRunsEachW8ProductAsHiThenAccumulatingLo) {
  const std::vector<Instruction> instructions = Emitted(SmallPlan());
  std::vector<size_t> gemms;
  for (size_t i = 0; i < instructions.size(); ++i) {
    if (Opcode(instructions[i]) == kOpGemm) {
      gemms.push_back(i);
    }
  }
  ASSERT_EQ(gemms.size(), size_t{4} * kHeads * kChunks);
  for (size_t p = 0; p < gemms.size(); p += 2) {
    const Instruction& hi = instructions[gemms[p]];
    const Instruction& lo = instructions[gemms[p + 1]];
    EXPECT_EQ(gemms[p + 1], gemms[p] + 1);
    EXPECT_EQ(hi.Get(2, 2), 1U);
    EXPECT_EQ(hi.Get(1, 1), 0U);
    EXPECT_EQ(lo.Get(2, 2), 2U);
    EXPECT_EQ(lo.Get(1, 1), 1U);
    EXPECT_EQ(lo.Get(4, 20), hi.Get(4, 20));
    EXPECT_EQ(lo.Get(92, 20), hi.Get(92, 20));
  }
}

TEST(SparseDriveLayersTest, AttentionLoadsKThroughSplit8AndVThroughT32Split8) {
  const std::vector<Instruction> instructions = Emitted(SmallPlan());
  size_t keys = 0;
  size_t values = 0;
  size_t queries = 0;
  for (const Instruction& instruction : instructions) {
    if (Opcode(instruction) != kOpDmaLoad) {
      continue;
    }
    const uint64_t offset = (instruction.Get(88, 28) * 16) - kBase;
    const uint64_t region = offset / kRegion;
    const uint64_t mem = instruction.Get(85, 3);
    const uint64_t kind = instruction.Get(25, 3);
    if (region == 2) {
      ++keys;
      EXPECT_EQ(mem, 1U);
      EXPECT_EQ(kind, 3U);
    } else if (region == 3) {
      ++values;
      EXPECT_EQ(mem, 1U);
      EXPECT_EQ(kind, 4U);
    } else {
      ++queries;
      EXPECT_EQ(region, 1U);
      EXPECT_EQ(mem, 0U);
      EXPECT_EQ(kind, 0U);
    }
  }
  EXPECT_EQ(keys, kHeads);
  EXPECT_EQ(values, kHeads);
  EXPECT_EQ(queries, kHeads);
}

TEST(SparseDriveLayersTest, AttentionSoftmaxMasksKeysPastTheValidCount) {
  for (const bool chunk_local : {false, true}) {
    AttentionPlan plan = SmallPlan();
    plan.chunk_local = chunk_local;
    const uint64_t valid = chunk_local ? 128 : plan.valid;
    size_t softmaxes = 0;
    for (const Instruction& instruction : Emitted(plan)) {
      if (Opcode(instruction) == kOpVec &&
          instruction.Get(108, 8) == static_cast<uint32_t>(VecOp::kSoftmax)) {
        ++softmaxes;
        EXPECT_EQ(instruction.Get(32, 11), valid);
        EXPECT_EQ(instruction.Get(107, 1), 1U);
      }
    }
    EXPECT_EQ(softmaxes, kHeads * kChunks);
  }
}

TEST(SparseDriveLayersDeathTest, AttentionRejectsTooFewQueryChunks) {
  AttentionPlan plan = SmallPlan();
  plan.queries = 64;
  EXPECT_DEATH(Emitted(plan), "attention over 64 queries is not supported");
}

}  // namespace
}  // namespace hotarubi::compiler::sparsedrive
