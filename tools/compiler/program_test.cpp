#include "program.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "hotarubi/isa.hpp"
#include "hotarubi/json.hpp"
#include "isa.hpp"

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

TEST(ProgramTest, WaitsForTheEngineThatWroteTheOperand) {
  Program program(kBase);
  program.Load({.ddr_offset = 0x1000, .mem = Mem::kAbuf, .rows = 8});
  program.Load({.ddr_offset = 0x2000, .mem = Mem::kWbuf, .rows = 1});
  const size_t first = program.Gemm(SmallGemm());
  const size_t second = program.Gemm(SmallGemm());
  EXPECT_EQ(WaitOf(program, first), kWaitDma);
  EXPECT_EQ(WaitOf(program, second), 0U);
  const size_t vec = program.Vec(
      {.op = VecOp::kCopy, .src = 0, .dst = 100, .rows_q = 8, .n8 = 1});
  EXPECT_EQ(WaitOf(program, vec), kWaitGemm);
  EXPECT_EQ(program.WaitCount(), 2U);
}

TEST(ProgramTest, WaitsBeforeOverwritingDataAnotherEngineReads) {
  Program program(kBase);
  program.Load({.ddr_offset = 0, .mem = Mem::kAbuf, .rows = 8});
  program.Load({.ddr_offset = 0, .mem = Mem::kWbuf, .rows = 1});
  program.Gemm(SmallGemm());
  const size_t untouched =
      program.Load({.ddr_offset = 0, .mem = Mem::kAbuf, .onchip = 100});
  const size_t overwrite =
      program.Load({.ddr_offset = 0, .mem = Mem::kAbuf, .onchip = 3});
  EXPECT_EQ(WaitOf(program, untouched), 0U);
  EXPECT_EQ(WaitOf(program, overwrite), kWaitGemm);
}

TEST(ProgramTest, ControlInstructionsWaitForAllEnginesButWaitHostDoesNot) {
  Program program(kBase);
  program.Load({.ddr_offset = 0, .mem = Mem::kFbuf});
  const size_t wait = program.WaitHost(3);
  const size_t signal = program.Signal(1, 2);
  const size_t halt = program.Halt();
  EXPECT_EQ(WaitOf(program, wait), 0U);
  EXPECT_EQ(WaitOf(program, signal), kWaitDma | kWaitGemm | kWaitVec);
  EXPECT_EQ(WaitOf(program, halt), kWaitDma | kWaitGemm | kWaitVec);
  EXPECT_EQ(program.EngineCount(Engine::kCtrl), 2U);
  EXPECT_EQ(program.EngineCount(Engine::kWait), 1U);
}

TEST(ProgramTest, AddsTheBaseToDdrOffsets) {
  Program program(kBase);
  program.Load({.ddr_offset = 0x1230, .mem = Mem::kTbuf, .rows = 2});
  EXPECT_EQ(program.Instructions()[0].Get(88, 28), (kBase + 0x1230U) / 16);
}

TEST(ProgramTest, SetDdrOffsetRewritesOnlyTheAddressOfATransfer) {
  Program program(kBase);
  const size_t index = program.Load(
      {.ddr_offset = 0x1000, .mem = Mem::kFbuf, .onchip = 7, .rows = 3});
  Instruction expected = program.Instructions()[index];
  expected.Set(88, 28, (kBase + 0x2340) / 16);
  program.SetDdrOffset(index, 0x2340);
  EXPECT_EQ(program.Instructions()[index].words, expected.words);
}

TEST(ProgramDeathTest,
     SetDdrOffsetRejectsOtherInstructionsAndUnalignedOffsets) {
  Program program(kBase);
  const size_t load = program.Load({.ddr_offset = 0, .mem = Mem::kFbuf});
  const size_t halt = program.Halt();
  EXPECT_DEATH(program.SetDdrOffset(halt, 0), "is not a DMA transfer");
  EXPECT_DEATH(program.SetDdrOffset(load, 8), "is not a DMA transfer");
  EXPECT_DEATH(program.SetDdrOffset(halt + 1, 0), "is not a DMA transfer");
}

TEST(ProgramTest, QuantizedVecWritesAbufThroughItsScratch) {
  Program program(kBase);
  const size_t index = program.Vec({.op = VecOp::kSoftmax,
                                    .src = 0,
                                    .dst = 40,
                                    .rows_q = 2,
                                    .n8 = 4,
                                    .q = true,
                                    .scratch = 512});
  const Instruction& instruction = program.Instructions()[index];
  EXPECT_EQ(instruction.Get(108, 8), isa::library::kSoftmax);
  EXPECT_EQ(instruction.Get(107, 1), 1U);
  EXPECT_EQ(instruction.Get(0, 32), 512U);
  const size_t gemm = program.Gemm({.a_base = 40,
                                    .a_stride = 1,
                                    .kb = 1,
                                    .m = 32,
                                    .w_base = 0,
                                    .w_stride = 1,
                                    .n8 = 1,
                                    .c_base = 2048});
  EXPECT_EQ(WaitOf(program, gemm), kWaitVec);
}

TEST(ProgramTest, ListsAndSerializesInstructions) {
  Program program(kBase);
  program.Load({.ddr_offset = 0x10, .mem = Mem::kFbuf}, "load");
  program.Halt();
  EXPECT_EQ(program.Listing(),
            "    0 DMA  w=0 01030000014000001001000000000000 load\n"
            "    1 CTRL w=7 07700000000000000000000000000000 halt\n");
  const std::vector<std::byte> bytes = program.Bytes();
  ASSERT_EQ(bytes.size(), 32U);
  EXPECT_EQ(bytes[0], std::byte{0x00});
  EXPECT_EQ(bytes[6], std::byte{0x01});
  EXPECT_EQ(bytes[7], std::byte{0x10});
  EXPECT_EQ(bytes[10], std::byte{0x40});
  EXPECT_EQ(bytes[11], std::byte{0x01});
  EXPECT_EQ(bytes[14], std::byte{0x03});
  EXPECT_EQ(bytes[15], std::byte{0x01});
  EXPECT_EQ(bytes[30], std::byte{0x70});
  EXPECT_EQ(bytes[31], std::byte{0x07});
}

TEST(ProgramTest, ScopesTagInstructionsAndTheJsonListingHoldsRanges) {
  Program program(kBase);
  program.Load({.ddr_offset = 0x10, .mem = Mem::kFbuf, .rows = 2}, "ld");
  {
    const Program::Scope vision(program, "vision");
    const Program::Scope layer(program, "L0");
    program.Vec(
        {.op = VecOp::kCopy, .src = 0, .dst = 100, .rows_q = 2, .n8 = 1},
        "copy \"x\"");
  }
  program.Halt();
  EXPECT_EQ(program.Tag(0), "");
  EXPECT_EQ(program.Tag(1), "vision/L0");
  EXPECT_EQ(program.Tag(2), "");
  EXPECT_EQ(program.Note(1), "copy \"x\"");
  ASSERT_EQ(program.Writes(0).size(), 1U);
  EXPECT_EQ(program.Writes(0)[0].mem, Mem::kFbuf);
  EXPECT_EQ(program.Writes(0)[0].lo, 0U);
  EXPECT_EQ(program.Writes(0)[0].hi, 2U);
  ASSERT_EQ(program.Reads(1).size(), 1U);
  EXPECT_EQ(program.Reads(1)[0].hi, 2U);
  EXPECT_EQ(program.Writes(1)[0].lo, 100U);
  const auto listing = Json::Parse(program.ListingJson());
  ASSERT_TRUE(listing.has_value()) << listing.error();
  EXPECT_EQ(listing->Find("base")->Integer(), static_cast<int64_t>(kBase));
  const Json::Array& tags = listing->Find("tags")->Items();
  ASSERT_EQ(tags.size(), 3U);
  EXPECT_EQ(tags[1].String(), "vision");
  EXPECT_EQ(tags[2].String(), "vision/L0");
  const Json::Array& instructions = listing->Find("instructions")->Items();
  ASSERT_EQ(instructions.size(), 3U);
  EXPECT_EQ(instructions[1].Find("engine")->String(), "VEC");
  EXPECT_EQ(instructions[1].Find("opcode")->Integer(), 5);
  EXPECT_EQ(instructions[1].Find("wait")->Integer(), 1);
  EXPECT_EQ(instructions[1].Find("tag")->Integer(), 2);
  EXPECT_EQ(instructions[1].Find("note")->String(), "copy \"x\"");
  EXPECT_EQ(instructions[1].Find("hex")->String(),
            program.Instructions()[1].Hex());
  const Json::Array& reads = instructions[1].Find("reads")->Items();
  ASSERT_EQ(reads.size(), 1U);
  EXPECT_EQ(reads[0].Items()[0].String(), "FBUF");
  EXPECT_EQ(reads[0].Items()[1].Integer(), 0);
  EXPECT_EQ(reads[0].Items()[2].Integer(), 2);
  EXPECT_EQ(instructions[2].Find("engine")->String(), "CTRL");
  EXPECT_EQ(instructions[2].Find("wait")->Integer(), 0x7);
  EXPECT_TRUE(instructions[2].Find("reads")->Items().empty());
}

TEST(ProgramDeathTest, RejectsAGemmOutsideTheSpecifiedRanges) {
  Program program(kBase);
  GemmArgs gemm = SmallGemm();
  gemm.m = 40;
  EXPECT_DEATH(program.Gemm(gemm), "bad GEMM");
}

}  // namespace
}  // namespace hotarubi::compiler
