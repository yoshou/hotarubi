#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "fp32_ref.hpp"
#include "gemm_ref.hpp"
#include "npu_bench.hpp"
#include "npu_data.hpp"
#include "npu_isa.hpp"

namespace hotarubi::tb {
namespace {

constexpr uint32_t kAbufArea = Ddr::kBase + 0x100000U;
constexpr uint32_t kWbufArea = Ddr::kBase + 0x300000U;
constexpr uint32_t kFbufInArea = Ddr::kBase + 0x500000U;
constexpr uint32_t kFbufOutArea = Ddr::kBase + 0x700000U;
constexpr uint32_t kGuard = 0x3F123456U;

struct GemmCase {
  GemmArgs args;
  std::vector<AbufWord> abuf;
  std::vector<WbufWord> wbuf;
  std::vector<FbufWord> fbuf;
};

uint32_t OutputWords(const GemmArgs& args) { return (args.m / 4) * args.n8; }

std::vector<AbufWord> RandomAbuf(std::mt19937& rng, uint32_t count) {
  std::vector<AbufWord> words(count);
  for (AbufWord& word : words) {
    for (int8_t& q : word.q) {
      q = static_cast<int8_t>(static_cast<int>(rng() % 256) - 128);
    }
    for (uint8_t& scale : word.scale) {
      scale = static_cast<uint8_t>(118 + (rng() % 18));
    }
  }
  return words;
}

std::vector<WbufWord> RandomWbuf(std::mt19937& rng, uint32_t count) {
  std::vector<WbufWord> words(count);
  for (WbufWord& word : words) {
    for (uint8_t& nibble : word.nibble) {
      nibble = static_cast<uint8_t>(rng() % 16);
    }
    for (uint8_t& scale : word.scale) {
      scale = static_cast<uint8_t>(118 + (rng() % 18));
    }
  }
  return words;
}

std::vector<FbufWord> RandomFbuf(std::mt19937& rng, uint32_t count) {
  std::vector<FbufWord> words(count);
  for (FbufWord& word : words) {
    for (uint32_t& value : word) {
      value = RandomFp32(rng, -6, 6);
    }
  }
  return words;
}

std::vector<FbufWord> RunOnNpu(const std::vector<GemmCase>& passes,
                               const std::vector<FbufWord>& fbuf_init,
                               uint32_t out_base, uint32_t out_words,
                               uint32_t seed) {
  NpuBench bench(seed);
  Ddr& ddr = bench.Memory();
  std::vector<Instruction> program;
  uint32_t a_area = kAbufArea;
  uint32_t w_area = kWbufArea;
  for (const GemmCase& pass : passes) {
    for (uint32_t i = 0; i < pass.abuf.size(); ++i) {
      WriteAbuf(ddr, a_area + (i * kAbufWordBytes), pass.abuf[i]);
    }
    for (uint32_t i = 0; i < pass.wbuf.size(); ++i) {
      WriteWbuf(ddr, w_area + (i * kWbufWordBytes), pass.wbuf[i]);
    }
    a_area += 0x80000U;
    w_area += 0x80000U;
  }
  for (uint32_t i = 0; i < fbuf_init.size(); ++i) {
    WriteFbuf(ddr, kFbufInArea + (i * kFbufWordBytes), fbuf_init[i]);
  }
  program.push_back(
      DmaLoad({.ddr_addr = kFbufInArea,
               .mem = Mem::kFbuf,
               .onchip = 0,
               .rows = 1,
               .row_words = static_cast<uint32_t>(fbuf_init.size())}));
  a_area = kAbufArea;
  w_area = kWbufArea;
  for (const GemmCase& pass : passes) {
    program.push_back(
        DmaLoad({.ddr_addr = a_area,
                 .mem = Mem::kAbuf,
                 .onchip = 0,
                 .rows = 1,
                 .row_words = static_cast<uint32_t>(pass.abuf.size()),
                 .wait = kWaitGemm}));
    program.push_back(
        DmaLoad({.ddr_addr = w_area,
                 .mem = Mem::kWbuf,
                 .onchip = 0,
                 .rows = 1,
                 .row_words = static_cast<uint32_t>(pass.wbuf.size())}));
    GemmArgs args = pass.args;
    args.wait = kWaitDma;
    program.push_back(Gemm(args));
    a_area += 0x80000U;
    w_area += 0x80000U;
  }
  program.push_back(DmaStore({.ddr_addr = kFbufOutArea,
                              .mem = Mem::kFbuf,
                              .onchip = out_base,
                              .rows = 1,
                              .row_words = out_words,
                              .wait = kWaitGemm}));
  program.push_back(Halt());
  EXPECT_TRUE(bench.Run(program));
  EXPECT_EQ(bench.ReadReg(0x04) & 0x6U, 0x2U);
  EXPECT_TRUE(bench.Errors().empty());
  std::vector<FbufWord> out;
  out.reserve(out_words);
  for (uint32_t i = 0; i < out_words; ++i) {
    out.push_back(ReadFbuf(ddr, kFbufOutArea + (i * kFbufWordBytes)));
  }
  return out;
}

void CheckGemm(const std::vector<GemmCase>& passes, uint32_t seed) {
  std::mt19937 rng(seed);
  uint32_t fbuf_words = 0;
  for (const GemmCase& pass : passes) {
    fbuf_words =
        std::max(fbuf_words, pass.args.c_base + OutputWords(pass.args) + 1);
  }
  std::vector<FbufWord> fbuf = RandomFbuf(rng, fbuf_words);
  for (uint32_t& value : fbuf.back()) {
    value = kGuard;
  }
  const std::vector<FbufWord> init = fbuf;
  for (const GemmCase& pass : passes) {
    ReferenceGemm(pass.args, pass.abuf, pass.wbuf, fbuf);
  }
  const std::vector<FbufWord> out = RunOnNpu(passes, init, 0, fbuf_words, seed);
  ASSERT_EQ(out.size(), fbuf.size());
  for (uint32_t i = 0; i < fbuf.size(); ++i) {
    for (uint32_t e = 0; e < 32; ++e) {
      ASSERT_EQ(out[i][e], fbuf[i][e]) << "word " << i << " element " << e;
    }
  }
}

GemmCase MakeCase(std::mt19937& rng, const GemmArgs& args) {
  GemmCase gemm{.args = args, .abuf = {}, .wbuf = {}, .fbuf = {}};
  gemm.abuf = RandomAbuf(rng, args.a_base + ((args.m / 4) * args.a_stride));
  gemm.wbuf = RandomWbuf(rng, args.w_base + (args.n8 * args.w_stride));
  return gemm;
}

TEST(GemmTest, SmallestFp4Tile) {
  std::mt19937 rng(1);
  CheckGemm({MakeCase(rng, {.a_base = 0,
                            .a_stride = 1,
                            .kb = 1,
                            .m = 32,
                            .w_base = 0,
                            .w_stride = 1,
                            .n8 = 1,
                            .c_base = 0,
                            .wmode = kWmodeFp4,
                            .acc = false})},
            1);
}

TEST(GemmTest, Fp4WithStridesAndOffsets) {
  std::mt19937 rng(2);
  CheckGemm({MakeCase(rng, {.a_base = 7,
                            .a_stride = 5,
                            .kb = 4,
                            .m = 64,
                            .w_base = 3,
                            .w_stride = 6,
                            .n8 = 3,
                            .c_base = 9,
                            .wmode = kWmodeFp4,
                            .acc = false})},
            2);
}

TEST(GemmTest, Fp4LongReduction) {
  std::mt19937 rng(3);
  CheckGemm({MakeCase(rng, {.a_base = 0,
                            .a_stride = 19,
                            .kb = 19,
                            .m = 96,
                            .w_base = 0,
                            .w_stride = 19,
                            .n8 = 2,
                            .c_base = 4,
                            .wmode = kWmodeFp4,
                            .acc = false})},
            3);
}

TEST(GemmTest, AccumulatesOntoExistingOutput) {
  std::mt19937 rng(4);
  CheckGemm({MakeCase(rng, {.a_base = 2,
                            .a_stride = 3,
                            .kb = 3,
                            .m = 32,
                            .w_base = 1,
                            .w_stride = 3,
                            .n8 = 5,
                            .c_base = 1,
                            .wmode = kWmodeFp4,
                            .acc = true})},
            4);
}

TEST(GemmTest, W8HighThenLowPass) {
  std::mt19937 rng(5);
  GemmCase high = MakeCase(rng, {.a_base = 0,
                                 .a_stride = 2,
                                 .kb = 2,
                                 .m = 64,
                                 .w_base = 0,
                                 .w_stride = 2,
                                 .n8 = 4,
                                 .c_base = 2,
                                 .wmode = kWmodeHigh,
                                 .acc = false});
  GemmCase low = high;
  low.args.wmode = kWmodeLow;
  low.args.acc = true;
  low.wbuf = RandomWbuf(rng, static_cast<uint32_t>(high.wbuf.size()));
  CheckGemm({high, low}, 5);
}

TEST(GemmTest, ManyColumnGroups) {
  std::mt19937 rng(6);
  CheckGemm({MakeCase(rng, {.a_base = 0,
                            .a_stride = 1,
                            .kb = 1,
                            .m = 32,
                            .w_base = 0,
                            .w_stride = 1,
                            .n8 = 37,
                            .c_base = 0,
                            .wmode = kWmodeFp4,
                            .acc = false})},
            6);
}

}  // namespace
}  // namespace hotarubi::tb
