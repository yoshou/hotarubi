#include "vec_engine_bench.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "encoding.hpp"
#include "hotarubi/exact.hpp"
#include "hotarubi/isa.hpp"
#include "memories.hpp"
#include "reference.hpp"
#include "timing.hpp"
#include "vec_engine.hpp"
#include "word.hpp"

namespace hotarubi::tb {
namespace {

using iss::encoding::PassFields;

constexpr uint32_t kProgram = isa::vec::kRomWords;

struct Case {
  std::string name;
  iss::VecCommand command;
  std::vector<iss::Word128> program;
};

class VecEngineBenchTest : public testing::Test {
 protected:
  uint32_t Random(uint32_t low, uint32_t high) {
    const uint32_t exponent =
        std::uniform_int_distribution<uint32_t>(low, high)(generator_);
    const uint32_t fraction =
        std::uniform_int_distribution<uint32_t>(0, 0x7FFFFF)(generator_);
    const uint32_t sign =
        std::uniform_int_distribution<uint32_t>(0, 1)(generator_);
    return (sign << 31) | (exponent << 23) | fraction;
  }

  void Fill() {
    for (uint32_t w = 0; w < 1024; ++w) {
      for (uint32_t& element : memories_.fbuf.at(w)) {
        element = Random(118, 132);
      }
    }
    for (uint32_t w = 0; w < 64; ++w) {
      for (uint32_t& element : memories_.pbuf.at(w)) {
        element = Random(122, 130);
      }
    }
    for (uint32_t i = 0; i < 512; ++i) {
      const uint32_t fraction =
          iss::reference::Round(exact::Real::Integer(512 + i).Scaled(-9) +
                                exact::Real::Integer(i % 7).Scaled(-30));
      const uint32_t activation =
          iss::reference::Round(exact::Real::Integer(256 + (i / 2)).Scaled(-9));
      const uint32_t slope = Random(117, 119) & 0x7FFFFFFFU;
      memories_.tbuf.at(i) = (uint64_t{slope} << 32) | fraction;
      memories_.tbuf.at(512 + i) = (uint64_t{slope} << 32) | activation;
    }
  }

  void Check(const Case& c, uint32_t deny_percent) {
    Fill();
    for (size_t i = 0; i < c.program.size(); ++i) {
      memories_.cbuf.at(kProgram + i) = c.program.at(i);
    }
    iss::Memories simulated = memories_;
    iss::VecEngine engine;
    engine.Execute(c.command, simulated);
    VecEngineBench bench(deny_percent, 7);
    const VecEngineRun run = bench.Run(c.command, memories_);
    EXPECT_EQ(memories_.fbuf, simulated.fbuf) << c.name;
    EXPECT_EQ(memories_.abuf, simulated.abuf) << c.name;
    EXPECT_EQ(run.passes, engine.Activity().passes) << c.name;
    if (deny_percent == 0) {
      const iss::CostModel model;
      const iss::Cost cost = model.Vec(engine.Activity());
      const auto predicted = static_cast<double>(cost.cycles);
      const auto measured = static_cast<double>(run.cycles);
      EXPECT_LT(std::abs(measured - predicted), (0.1 * predicted) + 50)
          << c.name << " measured " << run.cycles << " predicted "
          << cost.cycles;
    }
  }

  iss::Memories memories_;
  std::mt19937 generator_{3};
};

std::vector<Case> LibraryPrograms() {
  return {
      {"add",
       {.entry = isa::library::kAdd,
        .src = 0,
        .src2 = 100,
        .dst = 200,
        .rows_q = 3,
        .n8 = 5},
       {}},
      {"addp",
       {.entry = isa::library::kAddP,
        .src = 0,
        .src2 = 0,
        .dst = 200,
        .rows_q = 3,
        .n8 = 5,
        .p_base = 4},
       {}},
      {"max",
       {.entry = isa::library::kMax,
        .src = 0,
        .src2 = 100,
        .dst = 200,
        .rows_q = 2,
        .n8 = 8},
       {}},
      {"copy+q",
       {.entry = isa::library::kCopy,
        .quantize = true,
        .src = 0,
        .dst = 700,
        .rows_q = 2,
        .n8 = 8,
        .imm = 300},
       {}},
      {"layernorm",
       {.entry = isa::library::kLayerNorm,
        .src = 0,
        .src2 = 900,
        .dst = 400,
        .rows_q = 2,
        .n8 = 8,
        .p_base = 8,
        .aux = 64,
        .imm = 0x358637BD},
       {}},
      {"layernorm+q",
       {.entry = isa::library::kLayerNorm,
        .quantize = true,
        .src = 0,
        .src2 = 900,
        .dst = 500,
        .rows_q = 2,
        .n8 = 8,
        .p_base = 8,
        .aux = 64,
        .imm = 0x358637BD},
       {}},
      {"rmsnorm+q",
       {.entry = isa::library::kRmsNorm,
        .quantize = true,
        .src = 0,
        .src2 = 900,
        .dst = 500,
        .rows_q = 2,
        .n8 = 8,
        .p_base = 8,
        .aux = 64,
        .imm = 0x358637BD},
       {}},
      {"softmax",
       {.entry = isa::library::kSoftmax,
        .src = 0,
        .dst = 400,
        .rows_q = 2,
        .n8 = 8,
        .aux = 50},
       {}},
      {"softmax+q",
       {.entry = isa::library::kSoftmax,
        .quantize = true,
        .src = 0,
        .dst = 500,
        .rows_q = 2,
        .n8 = 8,
        .aux = 64,
        .imm = 900},
       {}},
      {"act+q",
       {.entry = isa::library::kAct,
        .quantize = true,
        .src = 0,
        .dst = 500,
        .rows_q = 2,
        .n8 = 4,
        .imm = 900},
       {}},
      {"act_mul",
       {.entry = isa::library::kActMul,
        .src = 0,
        .src2 = 100,
        .dst = 400,
        .rows_q = 2,
        .n8 = 4},
       {}},
      {"rope",
       {.entry = isa::library::kRope,
        .src = 0,
        .dst = 400,
        .rows_q = 2,
        .n8 = 16,
        .p_base = 16,
        .aux = 2},
       {}},
  };
}

Case DependentChain() {
  PassFields first{.rx = 1,
                   .rd = 11,
                   .rq = 4,
                   .rnw = 5,
                   .rn8 = 5,
                   .c0 = 4,
                   .wr = 1,
                   .ialu = 1};
  first.rimm = 8;
  PassFields second = first;
  second.rx = 11;
  second.rd = 12;
  PassFields third = first;
  third.rx = 12;
  third.rx2 = 1;
  third.rd = 3;
  third.x2_en = 1;
  third.c0 = 1;
  PassFields reduce{.rx = 3,
                    .rd = 13,
                    .rq = 4,
                    .rnw = 5,
                    .rn8 = 5,
                    .red = 1,
                    .wr = 1,
                    .sf = 1};
  PassFields scaled{
      .rx = 13, .rd = 14, .rq = 4, .rnw = 5, .rn8 = 5, .b = 3, .wr = 1};
  return {.name = "run chain",
          .command = {.entry = kProgram / isa::vec::kEntryWords,
                      .src = 0,
                      .dst = 300,
                      .rows_q = 2,
                      .n8 = 4,
                      .imm = 0x00030001},
          .program = {iss::encoding::SaluImm(0, 11, 0, 100),
                      iss::encoding::SaluImm(0, 12, 0, 200),
                      iss::encoding::SaluImm(0, 13, 0, 400),
                      iss::encoding::SaluImm(0, 14, 0, 500),
                      iss::encoding::Pass(first), iss::encoding::Pass(second),
                      iss::encoding::Pass(third), iss::encoding::Pass(reduce),
                      iss::encoding::Pass(scaled), iss::encoding::Ret()}};
}

TEST_F(VecEngineBenchTest, LibraryProgramsMatchTheSimulator) {
  for (const Case& c : LibraryPrograms()) {
    Check(c, 0);
  }
}

TEST_F(VecEngineBenchTest, ProgramsMatchTheSimulator) {
  Check(DependentChain(), 0);
}

TEST_F(VecEngineBenchTest, OutputsHoldUnderBackpressure) {
  for (const Case& c : LibraryPrograms()) {
    Check(c, 40);
  }
  Check(DependentChain(), 40);
}

}  // namespace
}  // namespace hotarubi::tb
