#include "kernels.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "hotarubi/formats.hpp"
#include "hotarubi/isa.hpp"
#include "isa.hpp"
#include "program.hpp"

namespace hotarubi::compiler {
namespace {

constexpr uint64_t kBase = 0x38700000;
constexpr uint64_t kRegion = 0x100000;
constexpr uint64_t kWordF = kWordFbufBytes;

uint32_t Opcode(const Instruction& instruction) {
  return static_cast<uint32_t>(instruction.Get(120, 8));
}

bool IsVec(const Instruction& instruction, VecOp op) {
  return Opcode(instruction) == isa::kOpVec &&
         instruction.Get(108, 8) == static_cast<uint32_t>(op);
}

Attention SmallAttention() {
  return {.q = kRegion,
          .k = 2 * kRegion,
          .v = 3 * kRegion,
          .o = 4 * kRegion,
          .heads = 8,
          .kv_heads = 8,
          .rows = 96,
          .embed = 256,
          .kv_blocks = 8,
          .hd = 64,
          .keys = 64,
          .valid = {{.quad = 0, .keys = 60}}};
}

std::vector<Instruction> Emitted(const Attention& attention) {
  Program program(kBase);
  EmitAttention(program, attention, "attention");
  program.Finalize({.window = 1});
  return program.Instructions();
}

TEST(KernelsTest, AttentionRunsEachW8ProductAsHiThenAccumulatingLo) {
  const std::vector<Instruction> instructions = Emitted(SmallAttention());
  std::vector<size_t> gemms;
  for (size_t i = 0; i < instructions.size(); ++i) {
    if (Opcode(instructions[i]) == isa::kOpGemm) {
      gemms.push_back(i);
    }
  }
  ASSERT_EQ(gemms.size(), size_t{4} * 8 * 3);
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

TEST(KernelsTest, AttentionLoadsKThroughSplit8AndVThroughT32Split8) {
  for (const uint32_t kv_heads : {8U, 2U}) {
    Attention attention = SmallAttention();
    attention.kv_heads = kv_heads;
    size_t keys = 0;
    size_t values = 0;
    size_t queries = 0;
    for (const Instruction& instruction : Emitted(attention)) {
      if (Opcode(instruction) != isa::kOpDmaLoad) {
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
    EXPECT_EQ(keys, kv_heads);
    EXPECT_EQ(values, kv_heads);
    EXPECT_EQ(queries, 8U);
  }
}

TEST(KernelsTest, AttentionSoftmaxMasksTheKeysOfEachRange) {
  Attention attention = SmallAttention();
  attention.valid = {{.quad = 0, .keys = 60}, {.quad = 12, .keys = 64}};
  std::vector<uint64_t> masked;
  for (const Instruction& instruction : Emitted(attention)) {
    if (IsVec(instruction, VecOp::kSoftmax)) {
      masked.push_back(instruction.Get(32, 11));
      EXPECT_EQ(instruction.Get(107, 1), 1U);
    }
  }
  ASSERT_EQ(masked.size(), size_t{8} * 4);
  const std::vector<uint64_t> head = {60, 60, 64, 64};
  for (size_t i = 0; i < masked.size(); ++i) {
    EXPECT_EQ(masked[i], head[i % 4]);
  }
}

TEST(KernelsTest, ChunkLocalAttentionLoadsTheKeysAndMaskOfEachChunk) {
  Attention attention = SmallAttention();
  attention.chunk_local = true;
  attention.mask = 5 * kRegion;
  attention.mask_stride = 16;
  std::vector<uint64_t> masks;
  size_t keys = 0;
  for (const Instruction& instruction : Emitted(attention)) {
    if (Opcode(instruction) != isa::kOpDmaLoad) {
      continue;
    }
    const uint64_t offset = (instruction.Get(88, 28) * 16) - kBase;
    if (offset / kRegion == 5) {
      masks.push_back(offset - (5 * kRegion));
    } else if (offset / kRegion == 2) {
      ++keys;
    }
  }
  EXPECT_EQ(keys, size_t{8} * 3);
  ASSERT_EQ(masks.size(), 3U);
  EXPECT_EQ(masks[0], 0U);
  EXPECT_EQ(masks[1], 16 * kWordF);
  EXPECT_EQ(masks[2], 32 * kWordF);
}

TEST(KernelsDeathTest, AttentionRejectsRowsOutsideWholeChunks) {
  Attention attention = SmallAttention();
  attention.rows = 100;
  EXPECT_DEATH(Emitted(attention), "attention of 100 rows");
}

TEST(KernelsTest, QuantizeLoadsTilesAndRunsTheHookBeforeEachQuantization) {
  Program program(kBase);
  const Tile out = program.Alloc(Mem::kAbuf, 20 * 8);
  std::vector<uint32_t> calls;
  EmitQuantize(program,
               {.x = kRegion,
                .quads = 20,
                .words = 32,
                .tile_quads = 8,
                .tokens = true,
                .dst = out.At()},
               "q", [&](const Tile& rows, uint32_t quad, uint32_t quads) {
                 calls.push_back(quad);
                 calls.push_back(quads);
                 program.Vec({.op = VecOp::kRelu,
                              .src = rows.At(),
                              .dst = rows.At(),
                              .rows_q = quads,
                              .n8 = 32});
               });
  program.Finalize({.window = 1});
  EXPECT_EQ(calls, (std::vector<uint32_t>{0, 8, 8, 8, 16, 4}));
  const std::vector<Instruction>& instructions = program.Instructions();
  ASSERT_EQ(instructions.size(), 9U);
  for (size_t t = 0; t < 3; ++t) {
    const Instruction& load = instructions[3 * t];
    EXPECT_EQ(Opcode(load), isa::kOpDmaLoad);
    EXPECT_EQ(load.Get(88, 28) * 16, kBase + kRegion + (t * 8 * 32 * kWordF));
    EXPECT_EQ(load.Get(25, 3), 1U);
    EXPECT_TRUE(IsVec(instructions[(3 * t) + 1], VecOp::kRelu));
    const Instruction& quantize = instructions[(3 * t) + 2];
    EXPECT_TRUE(IsVec(quantize, VecOp::kCopy));
    EXPECT_EQ(quantize.Get(107, 1), 1U);
    EXPECT_EQ(program.Writes((3 * t) + 2)[0].lo,
              program.Address(out) + (t * 8 * 8));
  }
}

TEST(KernelsTest, LinearLoadsEachWeightTileOnceAndAccumulatesThePieces) {
  Program program(kBase);
  const Tile a = program.Alloc(Mem::kAbuf, 16 * 4);
  const Tile b = program.Alloc(Mem::kAbuf, 16 * 2);
  program.Load(
      {.ddr_offset = 0, .mem = Mem::kAbuf, .onchip = a.At(), .row_words = 64});
  program.Load(
      {.ddr_offset = 0, .mem = Mem::kAbuf, .onchip = b.At(), .row_words = 32});
  std::vector<LinearTile> tiles;
  EmitLinear(program,
             {.planes = {{.mode = WeightMode::kW8Hi, .w = kRegion},
                         {.mode = WeightMode::kW8Lo, .w = 2 * kRegion}},
              .kb_total = 6,
              .pieces = {{.a = a.At(), .a_stride = 4, .kb = 4, .kb_offset = 0},
                         {.a = b.At(), .a_stride = 2, .kb = 2, .kb_offset = 4}},
              .quads = 16,
              .chunk_quads = 8,
              .groups = 12,
              .tile_groups = 8},
             "linear", [&](const LinearTile& tile) {
               tiles.push_back(tile);
               program.Store({.ddr_offset = 3 * kRegion,
                              .mem = Mem::kFbuf,
                              .onchip = tile.c.At(),
                              .row_words = tile.quads * tile.groups});
             });
  program.Finalize({.window = 1});
  ASSERT_EQ(tiles.size(), 4U);
  EXPECT_EQ(tiles[2].first, 8U);
  EXPECT_EQ(tiles[2].groups, 4U);
  EXPECT_EQ(tiles[3].quad, 8U);
  size_t weights = 0;
  std::vector<uint64_t> acc;
  for (const Instruction& instruction : program.Instructions()) {
    if (Opcode(instruction) == isa::kOpDmaLoad &&
        instruction.Get(85, 3) == 1U) {
      ++weights;
    }
    if (Opcode(instruction) == isa::kOpGemm) {
      acc.push_back(instruction.Get(1, 1));
    }
  }
  EXPECT_EQ(weights, size_t{2} * 2 * 2);
  ASSERT_EQ(acc.size(), size_t{4} * 4);
  for (size_t i = 0; i < acc.size(); ++i) {
    EXPECT_EQ(acc[i], i % 4 == 0 ? 0U : 1U);
  }
}

}  // namespace
}  // namespace hotarubi::compiler
