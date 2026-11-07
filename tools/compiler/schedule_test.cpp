#include "schedule.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "isa.hpp"
#include "program.hpp"

namespace hotarubi::compiler {
namespace {

constexpr uint64_t kBase = 0x30000000U;

uint64_t WaitOf(const Program& program, size_t index) {
  return program.Instructions().at(index).Get(116, 4);
}

GemmArgs SmallGemm() {
  return {.a_base = 0,
          .a_stride = 1,
          .kb = 1,
          .m = 32,
          .w_base = 0,
          .w_stride = 1,
          .n8 = 1,
          .c_base = 0};
}

std::vector<size_t> Identity(size_t count) {
  std::vector<size_t> order(count);
  for (size_t i = 0; i < count; ++i) {
    order[i] = i;
  }
  return order;
}

TEST(ScheduleTest, IssuesAReadyCommandAheadOfOneWaitingForItsEngine) {
  Program program(kBase);
  program.Load({.ddr_offset = 0x1000, .mem = Mem::kAbuf, .rows = 8});
  program.Load({.ddr_offset = 0x2000, .mem = Mem::kWbuf, .rows = 1});
  program.Load({.ddr_offset = 0x3000, .mem = Mem::kFbuf, .onchip = 100});
  program.Store({.ddr_offset = 0x100000,
                 .mem = Mem::kFbuf,
                 .onchip = 2048,
                 .rows = 64,
                 .row_words = 8,
                 .ddr_stride = 1024});
  program.Gemm(SmallGemm());
  program.Halt();
  const std::vector<size_t> expected = {0, 1, 2, 4, 3, 5};
  EXPECT_EQ(ScheduleOrder(program, {.window = 8}), expected);
  Schedule(program, {.window = 8});
  EXPECT_EQ(program.EngineOf(3), Engine::kGemm);
  EXPECT_EQ(WaitOf(program, 3), 0U);
  EXPECT_EQ(WaitOf(program, 5), kWaitDma | kWaitGemm | kWaitVec);
}

TEST(ScheduleTest, KeepsDependentCommandsInOrderAndRecomputesWaits) {
  Program program(kBase);
  program.Load({.ddr_offset = 0x1000, .mem = Mem::kAbuf, .rows = 8});
  program.Load({.ddr_offset = 0x2000, .mem = Mem::kWbuf, .rows = 1});
  program.Gemm(SmallGemm());
  program.Vec({.op = VecOp::kCopy, .src = 0, .dst = 100, .rows_q = 8, .n8 = 1});
  program.Store(
      {.ddr_offset = 0x4000, .mem = Mem::kFbuf, .onchip = 100, .rows = 8});
  program.Halt();
  EXPECT_EQ(ScheduleOrder(program, {.window = 8}), Identity(6));
  Schedule(program, {.window = 8});
  EXPECT_EQ(WaitOf(program, 2), kWaitDma);
  EXPECT_EQ(WaitOf(program, 3), kWaitGemm);
  EXPECT_EQ(WaitOf(program, 4), kWaitVec);
}

TEST(ScheduleTest, DdrTransfersConflictOnlyThroughStores) {
  Program program(kBase);
  program.Store({.ddr_offset = 0x1000,
                 .mem = Mem::kFbuf,
                 .rows = 2,
                 .row_words = 2,
                 .ddr_stride = 256});
  program.Load({.ddr_offset = 0x1100, .mem = Mem::kFbuf, .onchip = 100});
  program.Load({.ddr_offset = 0x2000, .mem = Mem::kFbuf, .onchip = 200});
  program.Load({.ddr_offset = 0x3000, .mem = Mem::kFbuf, .onchip = 300});
  program.Store({.ddr_offset = 0x3000, .mem = Mem::kFbuf, .onchip = 400});
  program.Load(
      {.ddr_offset = 0x5000, .mem = Mem::kFbuf, .onchip = 500, .index = true});
  EXPECT_TRUE(Conflicts(program, 0, 1));
  EXPECT_FALSE(Conflicts(program, 0, 2));
  EXPECT_FALSE(Conflicts(program, 1, 2));
  EXPECT_TRUE(Conflicts(program, 3, 4));
  EXPECT_FALSE(Conflicts(program, 2, 4));
  EXPECT_TRUE(Conflicts(program, 4, 5));
  EXPECT_TRUE(Conflicts(program, 0, 5));
  EXPECT_FALSE(Conflicts(program, 3, 5));
}

TEST(ScheduleTest, ControlInstructionsGatherCommandsAndProgramsAreOrdered) {
  Program program(kBase);
  program.Load({.ddr_offset = 0x1000, .mem = Mem::kFbuf});
  program.Signal(1, 2);
  program.Load({.ddr_offset = 0x2000, .mem = Mem::kFbuf, .onchip = 100});
  program.Gacc({.list_offset = 0x3000, .dst = 200, .slots = 4, .blocks = 2});
  program.Gacc({.list_offset = 0x4000, .dst = 300, .slots = 4, .blocks = 2});
  program.Vec({.op = VecOp::kCopy, .src = 0, .dst = 400, .rows_q = 1, .n8 = 1});
  program.Run({.pc = 256,
               .src = 500,
               .dst = 600,
               .rows_q = 1,
               .n8 = 1,
               .reads = std::vector<Range>{{Mem::kFbuf, 500, 501}},
               .writes = std::vector<Range>{{Mem::kFbuf, 600, 601}}});
  program.Vec({.op = VecOp::kCopy, .src = 1, .dst = 700, .rows_q = 1, .n8 = 1});
  EXPECT_TRUE(Conflicts(program, 0, 1));
  EXPECT_TRUE(Conflicts(program, 1, 2));
  EXPECT_TRUE(Conflicts(program, 3, 4));
  EXPECT_FALSE(Conflicts(program, 2, 3));
  EXPECT_TRUE(Conflicts(program, 5, 6));
  EXPECT_TRUE(Conflicts(program, 6, 7));
  EXPECT_FALSE(Conflicts(program, 5, 7));
}

TEST(ScheduleTest, ReorderRejectsAnythingButAPermutation) {
  Program program(kBase);
  program.Load({.ddr_offset = 0x1000, .mem = Mem::kFbuf});
  program.Halt();
  const std::vector<size_t> duplicate = {0, 0};
  const std::vector<size_t> shorter = {1};
  EXPECT_DEATH(program.Reorder(duplicate), "not a permutation");
  EXPECT_DEATH(program.Reorder(shorter), "not a permutation");
}

}  // namespace
}  // namespace hotarubi::compiler
