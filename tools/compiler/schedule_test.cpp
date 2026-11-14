#include "schedule.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <format>
#include <string>
#include <string_view>
#include <vector>

#include "isa.hpp"
#include "program.hpp"

namespace hotarubi::compiler {
namespace {

constexpr uint64_t kBase = 0x30000000U;

std::vector<std::string> Order(const Program& program) {
  std::vector<std::string> notes;
  notes.reserve(program.Size());
  for (size_t i = 0; i < program.Size(); ++i) {
    notes.emplace_back(program.Note(i));
  }
  return notes;
}

size_t Position(const Program& program, std::string_view note) {
  for (size_t i = 0; i < program.Size(); ++i) {
    if (program.Note(i) == note) {
      return i;
    }
  }
  return program.Size();
}

class Fixture {
 public:
  explicit Fixture(Program& program) : program_(program) {}

  Tile Loaded(Mem mem, uint32_t words, uint64_t ddr, std::string_view note) {
    const Tile tile = program_.Alloc(mem, words);
    program_.Load({.ddr_offset = ddr,
                   .mem = mem,
                   .onchip = tile.At(),
                   .row_words = words},
                  note);
    return tile;
  }

  void Slow(const Tile& src, const Tile& dst, std::string_view note) {
    program_.Vec({.op = VecOp::kLayerNorm,
                  .src = src.At(),
                  .dst = dst.At(),
                  .rows_q = src.Words() / 32,
                  .n8 = 32,
                  .p_base = params_.At()},
                 note);
  }

  void Params() { params_ = Loaded(Mem::kPbuf, 16, 0x8000, "ld params"); }

 private:
  Program& program_;
  Tile params_;
};

TEST(ScheduleTest, IssuesAReadyCommandAheadOfOneWaitingForAnotherEngine) {
  Program program(kBase);
  Fixture fixture(program);
  fixture.Params();
  const Tile x = fixture.Loaded(Mem::kFbuf, 2048, 0x10000, "ld x");
  const Tile y = program.Alloc(Mem::kFbuf, 2048);
  fixture.Slow(x, y, "norm");
  program.Store({.ddr_offset = 0x100000,
                 .mem = Mem::kFbuf,
                 .onchip = y.At(),
                 .row_words = 2048},
                "st y");
  fixture.Loaded(Mem::kAbuf, 64, 0x20000, "ld z");
  program.Halt();
  program.Finalize({.window = 8});
  EXPECT_LT(Position(program, "ld z"), Position(program, "st y"));
  EXPECT_LT(Position(program, "norm"), Position(program, "st y"));
  EXPECT_EQ(program.Note(program.Size() - 1), "halt");
}

TEST(ScheduleTest, AWindowOfOneKeepsTheProgramOrder) {
  Program program(kBase);
  Fixture fixture(program);
  fixture.Params();
  const Tile x = fixture.Loaded(Mem::kFbuf, 2048, 0x10000, "ld x");
  const Tile y = program.Alloc(Mem::kFbuf, 2048);
  fixture.Slow(x, y, "norm");
  program.Store({.ddr_offset = 0x100000,
                 .mem = Mem::kFbuf,
                 .onchip = y.At(),
                 .row_words = 2048},
                "st y");
  fixture.Loaded(Mem::kAbuf, 64, 0x20000, "ld z");
  program.Halt();
  program.Finalize({.window = 1});
  const std::vector<std::string> expected = {"ld params", "ld x", "norm",
                                             "st y",      "ld z", "halt"};
  EXPECT_EQ(Order(program), expected);
}

TEST(ScheduleTest, LoadsOfStoredDdrWaitForTheStore) {
  Program program(kBase);
  Fixture fixture(program);
  fixture.Params();
  const Tile x = fixture.Loaded(Mem::kFbuf, 2048, 0x10000, "ld x");
  const Tile y = program.Alloc(Mem::kFbuf, 2048);
  fixture.Slow(x, y, "norm");
  program.Store({.ddr_offset = 0x100000,
                 .mem = Mem::kFbuf,
                 .onchip = y.At(),
                 .row_words = 2048},
                "st y");
  fixture.Loaded(Mem::kAbuf, 64, 0x100800, "ld stored");
  fixture.Loaded(Mem::kAbuf, 64, 0x200000, "ld other");
  program.Halt();
  program.Finalize({.window = 16});
  EXPECT_LT(Position(program, "st y"), Position(program, "ld stored"));
  EXPECT_LT(Position(program, "ld other"), Position(program, "st y"));
}

TEST(ScheduleTest, GathersAndIndexedLoadsStayBehindEveryStore) {
  Program program(kBase);
  Fixture fixture(program);
  fixture.Params();
  const Tile x = fixture.Loaded(Mem::kFbuf, 2048, 0x10000, "ld x");
  const Tile y = program.Alloc(Mem::kFbuf, 2048);
  fixture.Slow(x, y, "norm");
  program.Store({.ddr_offset = 0x100000,
                 .mem = Mem::kFbuf,
                 .onchip = y.At(),
                 .row_words = 2048},
                "st y");
  const Tile g = program.Alloc(Mem::kFbuf, 8);
  program.Gacc(
      {.list_offset = 0x300000, .dst = g.At(), .slots = 4, .blocks = 2},
      "gacc");
  const Tile r = program.Alloc(Mem::kFbuf, 4);
  program.Load({.ddr_offset = 0x400000,
                .mem = Mem::kFbuf,
                .onchip = r.At(),
                .row_words = 4,
                .kind = DmaKind::kRowIndexed,
                .tok = 1},
               "gather rows");
  program.Halt();
  program.Finalize({.window = 16});
  EXPECT_LT(Position(program, "st y"), Position(program, "gacc"));
  EXPECT_LT(Position(program, "gacc"), Position(program, "gather rows"));
}

TEST(ScheduleTest, LoadedProgramsStayInOrderWithOtherVecCommands) {
  Program program(kBase);
  Fixture fixture(program);
  fixture.Params();
  const Tile a = fixture.Loaded(Mem::kFbuf, 64, 0x10000, "ld a");
  const Tile b = fixture.Loaded(Mem::kFbuf, 64, 0x20000, "ld b");
  const Tile c = program.Alloc(Mem::kFbuf, 64);
  program.Load({.ddr_offset = 0x30000,
                .mem = Mem::kCbuf,
                .onchip = Place::Fixed(256),
                .row_words = 8},
               "ld program");
  program.Vec(
      {.op = VecOp::kCopy, .src = a.At(), .dst = a.At(), .rows_q = 8, .n8 = 8},
      "copy a");
  program.Run({.pc = 256,
               .src = b.At(),
               .dst = c.At(),
               .rows_q = 8,
               .n8 = 8,
               .imm_places = {},
               .reads = {Whole(b)},
               .writes = {Whole(c)}},
              "run");
  program.Vec(
      {.op = VecOp::kCopy, .src = b.At(), .dst = b.At(), .rows_q = 8, .n8 = 8},
      "copy b");
  program.Store({.ddr_offset = 0x100000,
                 .mem = Mem::kFbuf,
                 .onchip = c.At(),
                 .row_words = 64},
                "st c");
  program.Halt();
  program.Finalize({.window = 16});
  EXPECT_LT(Position(program, "ld program"), Position(program, "run"));
  EXPECT_LT(Position(program, "copy a"), Position(program, "run"));
  EXPECT_LT(Position(program, "run"), Position(program, "copy b"));
}

TEST(ScheduleTest, ControlInstructionsAreBarriers) {
  Program program(kBase);
  Fixture fixture(program);
  fixture.Params();
  fixture.Loaded(Mem::kFbuf, 2048, 0x10000, "before");
  program.Signal(1, 1, "signal");
  fixture.Loaded(Mem::kAbuf, 64, 0x20000, "after");
  program.Halt();
  program.Finalize({.window = 16});
  EXPECT_LT(Position(program, "before"), Position(program, "signal"));
  EXPECT_LT(Position(program, "signal"), Position(program, "after"));
}

TEST(ScheduleTest, CompletesWhenTheTilesNeedTheWholeMemoryInTurn) {
  Program program(kBase);
  Fixture fixture(program);
  fixture.Params();
  for (uint32_t i = 0; i < 6; ++i) {
    const Tile x = fixture.Loaded(Mem::kFbuf, 2048, uint64_t{0x10000} * (i + 1),
                                  std::format("ld x{}", i));
    const Tile y = program.Alloc(Mem::kFbuf, 2048);
    fixture.Slow(x, y, std::format("norm {}", i));
    program.Store({.ddr_offset = 0x100000U + (uint64_t{0x10000} * i),
                   .mem = Mem::kFbuf,
                   .onchip = y.At(),
                   .row_words = 2048},
                  std::format("st y{}", i));
  }
  program.Halt();
  program.Finalize({.window = 64});
  for (uint32_t i = 0; i < 6; ++i) {
    const size_t load = Position(program, std::format("ld x{}", i));
    const size_t norm = Position(program, std::format("norm {}", i));
    const size_t store = Position(program, std::format("st y{}", i));
    EXPECT_LT(load, norm);
    EXPECT_LT(norm, store);
    EXPECT_EQ(program.Reads(norm)[0].lo, program.Writes(load)[0].lo);
    EXPECT_EQ(program.Reads(store)[0].lo, program.Writes(norm)[0].lo);
  }
}

TEST(ScheduleTest, KeepsTheNextTileOutOfTheWordsAnEngineStillReads) {
  Program program(kBase);
  Fixture fixture(program);
  fixture.Params();
  const Tile x = fixture.Loaded(Mem::kFbuf, 1024, 0x10000, "ld x");
  const Tile y = program.Alloc(Mem::kFbuf, 1024);
  fixture.Slow(x, y, "norm");
  fixture.Loaded(Mem::kFbuf, 1024, 0x20000, "ld next");
  program.Store({.ddr_offset = 0x100000,
                 .mem = Mem::kFbuf,
                 .onchip = y.At(),
                 .row_words = 1024},
                "st y");
  program.Halt();
  program.Finalize({.window = 16});
  const size_t norm = Position(program, "norm");
  const size_t next = Position(program, "ld next");
  EXPECT_FALSE(program.Writes(next)[0].lo < program.Reads(norm)[0].hi &&
               program.Reads(norm)[0].lo < program.Writes(next)[0].hi);
}

TEST(ScheduleDeathTest, RejectsTilesLargerThanTheirMemory) {
  const std::vector<TileSize> tiles = {{.mem = Mem::kPbuf, .words = 513}};
  EXPECT_DEATH(static_cast<void>(ScheduleStream({}, tiles, {})),
               "does not fit its memory");
}

}  // namespace
}  // namespace hotarubi::compiler
