#include "program.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <format>
#include <string>
#include <vector>

#include "hotarubi/isa.hpp"
#include "hotarubi/json.hpp"
#include "isa.hpp"

namespace hotarubi::compiler {
namespace {

constexpr uint64_t kBase = 0x30000000U;
constexpr ScheduleOptions kInOrder{.window = 1};

uint64_t WaitOf(const Program& program, size_t index) {
  return program.Instructions().at(index).Get(116, 4);
}

GemmArgs SmallGemm(const Tile& a, const Tile& w, const Tile& c) {
  return {.a_base = a.At(),
          .a_stride = 1,
          .kb = 1,
          .m = 32,
          .w_base = w.At(),
          .w_stride = 1,
          .n8 = 1,
          .c_base = c.At()};
}

bool Overlap(const Range& a, const Range& b) {
  return a.mem == b.mem && a.lo < b.hi && b.lo < a.hi;
}

TEST(ProgramTest, WaitsForTheEngineThatWroteTheOperand) {
  Program program(kBase);
  const Tile a = program.Alloc(Mem::kAbuf, 8);
  const Tile w = program.Alloc(Mem::kWbuf, 1);
  const Tile c = program.Alloc(Mem::kFbuf, 8);
  const Tile d = program.Alloc(Mem::kFbuf, 8);
  program.Load(
      {.ddr_offset = 0x1000, .mem = Mem::kAbuf, .onchip = a.At(), .rows = 8});
  program.Load(
      {.ddr_offset = 0x2000, .mem = Mem::kWbuf, .onchip = w.At(), .rows = 1});
  program.Gemm(SmallGemm(a, w, c));
  program.Gemm(SmallGemm(a, w, c));
  program.Vec(
      {.op = VecOp::kCopy, .src = c.At(), .dst = d.At(), .rows_q = 8, .n8 = 1});
  program.Store(
      {.ddr_offset = 0x3000, .mem = Mem::kFbuf, .onchip = d.At(), .rows = 8});
  program.Finalize(kInOrder);
  EXPECT_EQ(WaitOf(program, 2), kWaitDma);
  EXPECT_EQ(WaitOf(program, 3), 0U);
  EXPECT_EQ(WaitOf(program, 4), kWaitGemm);
  EXPECT_EQ(WaitOf(program, 5), kWaitVec);
  EXPECT_EQ(program.WaitCount(), 3U);
}

TEST(ProgramTest, WaitsBeforeOverwritingATileAnotherEngineReads) {
  Program program(kBase);
  const Tile a = program.Alloc(Mem::kAbuf, 8);
  const Tile w = program.Alloc(Mem::kWbuf, 1);
  const Tile c = program.Alloc(Mem::kFbuf, 8);
  program.Load(
      {.ddr_offset = 0, .mem = Mem::kAbuf, .onchip = a.At(), .rows = 8});
  program.Load({.ddr_offset = 0, .mem = Mem::kWbuf, .onchip = w.At()});
  program.Gemm(SmallGemm(a, w, c));
  program.Load({.ddr_offset = 0, .mem = Mem::kAbuf, .onchip = a.At(3)});
  program.Gemm(SmallGemm(a, w, c));
  program.Finalize(kInOrder);
  EXPECT_EQ(WaitOf(program, 3), kWaitGemm);
}

TEST(ProgramTest, LiveTilesNeverShareWords) {
  Program program(kBase);
  std::vector<Tile> tiles;
  for (uint32_t i = 0; i < 6; ++i) {
    tiles.push_back(program.Alloc(Mem::kFbuf, 600));
    program.Load({.ddr_offset = uint64_t{0x1000} * i,
                  .mem = Mem::kFbuf,
                  .onchip = tiles.back().At(),
                  .row_words = 600});
  }
  for (const Tile& tile : tiles) {
    program.Store({.ddr_offset = 0x100000,
                   .mem = Mem::kFbuf,
                   .onchip = tile.At(),
                   .row_words = 600});
  }
  program.Halt();
  program.Finalize();
  for (size_t i = 0; i < tiles.size(); ++i) {
    for (size_t j = i + 1; j < tiles.size(); ++j) {
      const uint32_t a = program.Address(tiles[i]);
      const uint32_t b = program.Address(tiles[j]);
      EXPECT_TRUE(a + 600 <= b || b + 600 <= a) << i << " " << j;
    }
  }
}

TEST(ProgramTest, ReusesTheWordsOfDeadTiles) {
  Program program(kBase);
  for (uint32_t i = 0; i < 8; ++i) {
    const Tile tile = program.Alloc(Mem::kFbuf, 3000);
    program.Load({.ddr_offset = uint64_t{0x10000} * i,
                  .mem = Mem::kFbuf,
                  .onchip = tile.At(),
                  .row_words = 3000},
                 std::format("ld {}", i));
    program.Store({.ddr_offset = 0x200000U + (uint64_t{0x10000} * i),
                   .mem = Mem::kFbuf,
                   .onchip = tile.At(),
                   .row_words = 3000},
                  std::format("st {}", i));
  }
  program.Halt();
  program.Finalize();
  ASSERT_EQ(program.Size(), 17U);
  for (size_t i = 0; i + 1 < program.Size(); ++i) {
    if (program.Note(i).starts_with("st")) {
      continue;
    }
    const std::string store = std::format("st{}", program.Note(i).substr(2));
    size_t j = i + 1;
    for (; program.Note(j) != store; ++j) {
      if (!program.Writes(j).empty()) {
        EXPECT_FALSE(Overlap(program.Writes(j)[0], program.Writes(i)[0]))
            << program.Note(j) << " overwrites " << program.Note(i);
      }
    }
    EXPECT_EQ(program.Reads(j)[0].lo, program.Writes(i)[0].lo);
  }
}

TEST(ProgramTest, ControlInstructionsWaitForAllEnginesButWaitHostDoesNot) {
  Program program(kBase);
  const Tile tile = program.Alloc(Mem::kFbuf, 1);
  program.Load({.ddr_offset = 0, .mem = Mem::kFbuf, .onchip = tile.At()});
  program.WaitHost(3);
  program.Signal(1, 2);
  program.Halt();
  program.Finalize(kInOrder);
  EXPECT_EQ(WaitOf(program, 1), 0U);
  EXPECT_EQ(WaitOf(program, 2), kWaitDma | kWaitGemm | kWaitVec);
  EXPECT_EQ(WaitOf(program, 3), kWaitDma | kWaitGemm | kWaitVec);
  EXPECT_EQ(program.EngineCount(Engine::kCtrl), 2U);
  EXPECT_EQ(program.EngineCount(Engine::kWait), 1U);
}

TEST(ProgramTest, AddsTheBaseToDdrOffsets) {
  Program program(kBase);
  program.Load({.ddr_offset = 0x1230,
                .mem = Mem::kTbuf,
                .onchip = Place::Fixed(0),
                .rows = 2});
  program.Finalize();
  EXPECT_EQ(program.Instructions()[0].Get(88, 28), (kBase + 0x1230U) / 16);
}

TEST(ProgramTest, SetDdrOffsetRewritesOnlyTheAddressOfATransfer) {
  Program reference(kBase);
  Program program(kBase);
  for (Program* p : {&reference, &program}) {
    const Tile tile = p->Alloc(Mem::kFbuf, 3);
    p->Load({.ddr_offset = p == &reference ? 0x2340U : 0x1000U,
             .mem = Mem::kFbuf,
             .onchip = tile.At(),
             .rows = 3});
  }
  program.SetDdrOffset(0, 0x2340);
  reference.Finalize();
  program.Finalize();
  EXPECT_EQ(program.Instructions()[0].words, reference.Instructions()[0].words);
}

TEST(ProgramDeathTest,
     SetDdrOffsetRejectsOtherInstructionsAndUnalignedOffsets) {
  Program program(kBase);
  const Tile tile = program.Alloc(Mem::kFbuf, 1);
  const size_t load =
      program.Load({.ddr_offset = 0, .mem = Mem::kFbuf, .onchip = tile.At()});
  const size_t halt = program.Halt();
  EXPECT_DEATH(program.SetDdrOffset(halt, 0), "is not a DMA transfer");
  EXPECT_DEATH(program.SetDdrOffset(load, 8), "is not a DMA transfer");
  EXPECT_DEATH(program.SetDdrOffset(halt + 1, 0), "is not a DMA transfer");
}

TEST(ProgramTest, QuantizedVecWritesAbufThroughItsScratch) {
  Program program(kBase);
  const Tile x = program.Alloc(Mem::kFbuf, 32);
  const Tile q = program.Alloc(Mem::kAbuf, 8);
  const Tile scratch = program.Alloc(Mem::kFbuf, 32);
  const Tile w = program.Alloc(Mem::kWbuf, 1);
  const Tile c = program.Alloc(Mem::kFbuf, 8);
  program.Load(
      {.ddr_offset = 0, .mem = Mem::kFbuf, .onchip = x.At(), .rows = 32});
  program.Vec({.op = VecOp::kSoftmax,
               .src = x.At(),
               .dst = q.At(),
               .rows_q = 8,
               .n8 = 4,
               .q = true,
               .scratch = scratch.At()});
  program.Load({.ddr_offset = 0, .mem = Mem::kWbuf, .onchip = w.At()});
  program.Gemm({.a_base = q.At(),
                .a_stride = 1,
                .kb = 1,
                .m = 32,
                .w_base = w.At(),
                .w_stride = 1,
                .n8 = 1,
                .c_base = c.At()});
  program.Finalize(kInOrder);
  const Instruction& instruction = program.Instructions()[1];
  EXPECT_EQ(instruction.Get(108, 8), isa::library::kSoftmax);
  EXPECT_EQ(instruction.Get(107, 1), 1U);
  EXPECT_EQ(instruction.Get(0, 32), program.Address(scratch));
  EXPECT_EQ(WaitOf(program, 3), kWaitDma | kWaitVec);
}

TEST(ProgramTest, RunResolvesItsPlacesAndPacksThemIntoTheImmediate) {
  Program program(kBase);
  const Tile src = program.Alloc(Mem::kFbuf, 4);
  const Tile work = program.Alloc(Mem::kFbuf, 64);
  const Tile out = program.Alloc(Mem::kFbuf, 4);
  program.Load(
      {.ddr_offset = 0, .mem = Mem::kFbuf, .onchip = src.At(), .row_words = 4});
  program.Run({.pc = 256,
               .src = src.At(),
               .dst = out.At(),
               .rows_q = 1,
               .n8 = 1,
               .imm = 1U << 30,
               .imm_places = {{.place = work.At(8), .shift = 12}},
               .reads = {Whole(src), Whole(work)},
               .writes = {Whole(work), Whole(out)}});
  program.Store({.ddr_offset = 0x1000,
                 .mem = Mem::kFbuf,
                 .onchip = out.At(),
                 .row_words = 4});
  program.Finalize(kInOrder);
  const Instruction& run = program.Instructions()[1];
  EXPECT_EQ(run.Get(0, 32), (1U << 30) | ((program.Address(work) + 8) << 12));
  ASSERT_EQ(program.Reads(1).size(), 3U);
  EXPECT_EQ(program.Reads(1)[2].mem, Mem::kCbuf);
  EXPECT_EQ(program.Writes(1)[1].lo, program.Address(out));
  EXPECT_EQ(WaitOf(program, 1), kWaitDma);
  EXPECT_EQ(WaitOf(program, 2), kWaitVec);
}

TEST(ProgramTest, ListsAndSerializesInstructions) {
  Program program(kBase);
  const Tile tile = program.Alloc(Mem::kFbuf, 1);
  program.Load({.ddr_offset = 0x10, .mem = Mem::kFbuf, .onchip = tile.At()},
               "load");
  program.Halt();
  program.Finalize();
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
  const Tile x = program.Alloc(Mem::kFbuf, 2);
  program.Load(
      {.ddr_offset = 0x10, .mem = Mem::kFbuf, .onchip = x.At(), .rows = 2},
      "ld");
  {
    const Program::Scope vision(program, "vision");
    const Program::Scope layer(program, "L0");
    program.Vec({.op = VecOp::kCopy,
                 .src = x.At(),
                 .dst = x.At(),
                 .rows_q = 2,
                 .n8 = 1},
                "copy \"x\"");
  }
  program.Halt();
  program.Finalize(kInOrder);
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
  EXPECT_EQ(program.Writes(1)[0].lo, 0U);
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
  const Tile a = program.Alloc(Mem::kAbuf, 8);
  const Tile w = program.Alloc(Mem::kWbuf, 1);
  const Tile c = program.Alloc(Mem::kFbuf, 8);
  GemmArgs gemm = SmallGemm(a, w, c);
  gemm.m = 40;
  EXPECT_DEATH(program.Gemm(gemm), "bad GEMM");
}

TEST(ProgramDeathTest, RejectsAccessesOutsideTheirTilesAndMemories) {
  Program program(kBase);
  const Tile tile = program.Alloc(Mem::kFbuf, 4);
  EXPECT_DEATH(program.Load({.ddr_offset = 0,
                             .mem = Mem::kFbuf,
                             .onchip = tile.At(2),
                             .row_words = 4}),
               "bad range");
  EXPECT_DEATH(
      program.Load(
          {.ddr_offset = 0, .mem = Mem::kFbuf, .onchip = Place::Fixed(0)}),
      "outside its tiles");
  EXPECT_DEATH(
      program.Load({.ddr_offset = 0, .mem = Mem::kAbuf, .onchip = tile.At()}),
      "outside its tiles");
  EXPECT_DEATH(
      program.Load({.ddr_offset = 0, .mem = Mem::kTbuf, .onchip = tile.At()}),
      "addressed directly");
  EXPECT_DEATH(static_cast<void>(program.Alloc(Mem::kFbuf, 4097)), "bad tile");
  EXPECT_DEATH(static_cast<void>(program.Alloc(Mem::kTbuf, 1)), "bad tile");
}

TEST(ProgramDeathTest, RejectsALoadedProgramNamingAnUndeclaredTile) {
  Program program(kBase);
  const Tile src = program.Alloc(Mem::kFbuf, 4);
  const Tile work = program.Alloc(Mem::kFbuf, 4);
  EXPECT_DEATH(program.Run({.pc = 256,
                            .src = src.At(),
                            .imm_places = {{.place = work.At(), .shift = 12}},
                            .reads = {Whole(src)},
                            .writes = {}}),
               "names a tile it does not declare");
}

TEST(ProgramDeathTest, RejectsUseBeforeFinalizeAndAfterIt) {
  Program program(kBase);
  program.Halt();
  EXPECT_DEATH(static_cast<void>(program.Bytes()), "not finalized");
  program.Finalize();
  EXPECT_DEATH(program.Halt(), "after the program is finalized");
  EXPECT_DEATH(program.Finalize(), "finalized twice");
}

}  // namespace
}  // namespace hotarubi::compiler
