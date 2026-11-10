#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "fp32_ref.hpp"
#include "npu_bench.hpp"
#include "npu_data.hpp"
#include "npu_isa.hpp"

namespace hotarubi::tb {
namespace {

constexpr uint32_t kSrcArea = Ddr::kBase + 0x100000U;
constexpr uint32_t kListArea = Ddr::kBase + 0x300000U;
constexpr uint32_t kDstArea = Ddr::kBase + 0x500000U;
constexpr uint32_t kAuxArea = Ddr::kBase + 0x700000U;

FbufWord RandomFbufWord(std::mt19937& rng) {
  FbufWord word{};
  for (uint32_t& value : word) {
    value = RandomFp32(rng, -20, 20);
  }
  return word;
}

AbufWord RandomAbufWord(std::mt19937& rng) {
  AbufWord word;
  for (int8_t& q : word.q) {
    q = static_cast<int8_t>(static_cast<int>(rng() % 256) - 128);
  }
  for (uint8_t& scale : word.scale) {
    scale = static_cast<uint8_t>(100 + (rng() % 50));
  }
  return word;
}

void FillGarbage(Ddr& ddr, uint32_t address, uint32_t bytes,
                 std::mt19937& rng) {
  for (uint32_t i = 0; i < bytes; ++i) {
    ddr.Write8(address + i, static_cast<uint8_t>(rng()));
  }
}

void RunProgram(NpuBench& bench, const std::vector<Instruction>& program) {
  ASSERT_TRUE(bench.Run(program));
  EXPECT_EQ(bench.ReadReg(0x04) & 0x6U, 0x2U);
  EXPECT_TRUE(bench.Errors().empty());
}

TEST(DmaTest, FbufRaw2dLoadAndStore) {
  std::mt19937 rng(21);
  NpuBench bench(21);
  Ddr& ddr = bench.Memory();
  constexpr uint32_t kLoadStride = 0x1040;
  constexpr uint32_t kStoreStride = 0x1F0;
  std::vector<FbufWord> words;
  for (uint32_t r = 0; r < 3; ++r) {
    for (uint32_t w = 0; w < 5; ++w) {
      words.push_back(RandomFbufWord(rng));
      WriteFbuf(ddr,
                kSrcArea + 0xF80 + (r * kLoadStride) + (w * kFbufWordBytes),
                words.back());
    }
  }
  RunProgram(bench, {DmaLoad({.ddr_addr = kSrcArea + 0xF80,
                              .mem = Mem::kFbuf,
                              .onchip = 100,
                              .rows = 3,
                              .row_words = 5,
                              .ddr_stride = kLoadStride}),
                     DmaStore({.ddr_addr = kDstArea,
                               .mem = Mem::kFbuf,
                               .onchip = 100,
                               .rows = 5,
                               .row_words = 3,
                               .ddr_stride = kStoreStride}),
                     Halt()});
  for (uint32_t i = 0; i < words.size(); ++i) {
    const uint32_t address =
        kDstArea + ((i / 3) * kStoreStride) + ((i % 3) * kFbufWordBytes);
    EXPECT_EQ(ReadFbuf(ddr, address), words[i]) << "word " << i;
  }
}

TEST(DmaTest, AbufRawLoadAndStoreWritesPadding) {
  std::mt19937 rng(22);
  NpuBench bench(22);
  Ddr& ddr = bench.Memory();
  std::vector<AbufWord> words;
  for (uint32_t i = 0; i < 8; ++i) {
    words.push_back(RandomAbufWord(rng));
    WriteAbuf(ddr, kSrcArea + ((i / 4) * 0x400) + ((i % 4) * kAbufWordBytes),
              words.back());
  }
  FillGarbage(ddr, kDstArea, 8 * kAbufWordBytes, rng);
  RunProgram(bench, {DmaLoad({.ddr_addr = kSrcArea,
                              .mem = Mem::kAbuf,
                              .onchip = 4000,
                              .rows = 2,
                              .row_words = 4,
                              .ddr_stride = 0x400}),
                     DmaStore({.ddr_addr = kDstArea,
                               .mem = Mem::kAbuf,
                               .onchip = 4000,
                               .rows = 1,
                               .row_words = 8}),
                     Halt()});
  for (uint32_t i = 0; i < words.size(); ++i) {
    const uint32_t address = kDstArea + (i * kAbufWordBytes);
    EXPECT_EQ(ReadAbuf(ddr, address), words[i]) << "word " << i;
    for (uint32_t b = 132; b < kAbufWordBytes; ++b) {
      EXPECT_EQ(ddr.Read8(address + b), 0) << "word " << i << " byte " << b;
    }
  }
}

TEST(DmaTest, FbufRowFormatIndexListWithNullTokens) {
  std::mt19937 rng(23);
  NpuBench bench(23);
  Ddr& ddr = bench.Memory();
  constexpr uint32_t kTok = 2;
  constexpr uint32_t kTokenBytes = 128 * kTok;
  constexpr uint32_t kTokens = 12;
  constexpr uint32_t kWords = kTokens * kTok;
  std::vector<std::vector<uint32_t>> tokens;
  for (uint32_t e = 0; e < kTokens; ++e) {
    std::vector<uint32_t> token(size_t{32} * kTok, 0);
    if (e == 3 || e == 10) {
      ddr.Write32(kListArea + (4 * e), 0xFFFFFFFFU);
    } else {
      const uint32_t address = kSrcArea + (((e * 37) % 100) * kTokenBytes);
      for (uint32_t i = 0; i < 32 * kTok; ++i) {
        token[i] = RandomFp32(rng, -10, 10);
        ddr.Write32(address + (4 * i), token[i]);
      }
      ddr.Write32(kListArea + (4 * e), address / 16);
    }
    tokens.push_back(token);
  }
  FillGarbage(ddr, kAuxArea, kWords * kFbufWordBytes, rng);
  RunProgram(bench, {DmaLoad({.ddr_addr = kAuxArea,
                              .mem = Mem::kFbuf,
                              .onchip = 20,
                              .rows = 1,
                              .row_words = kWords}),
                     DmaLoad({.ddr_addr = kListArea,
                              .mem = Mem::kFbuf,
                              .onchip = 20,
                              .rows = 3,
                              .row_words = 4,
                              .tok = kTok,
                              .flags = kDmaFlagIndex | kDmaFlagRow}),
                     DmaStore({.ddr_addr = kDstArea,
                               .mem = Mem::kFbuf,
                               .onchip = 20,
                               .rows = 1,
                               .row_words = kWords}),
                     Halt()});
  for (uint32_t i = 0; i < kTokens; ++i) {
    for (uint32_t ch = 0; ch < 32 * kTok; ++ch) {
      const uint32_t word = ((i / 4) * 4 * kTok) + (ch / 8);
      const uint32_t element = ((i % 4) * 8) + (ch % 8);
      ASSERT_EQ(ddr.Read32(kDstArea + (word * kFbufWordBytes) + (4 * element)),
                tokens[i][ch])
          << "token " << i << " channel " << ch;
    }
  }
}

TEST(DmaTest, StoreIgnoresTheIndexFlag) {
  std::mt19937 rng(24);
  NpuBench bench(24);
  Ddr& ddr = bench.Memory();
  constexpr uint32_t kStride = 0x400;
  std::vector<FbufWord> words;
  for (uint32_t e = 0; e < 6; ++e) {
    words.push_back(RandomFbufWord(rng));
    WriteFbuf(ddr, kSrcArea + (e * kFbufWordBytes), words.back());
  }
  RunProgram(bench, {DmaLoad({.ddr_addr = kSrcArea,
                              .mem = Mem::kFbuf,
                              .onchip = 50,
                              .rows = 1,
                              .row_words = 6}),
                     DmaStore({.ddr_addr = kDstArea,
                               .mem = Mem::kFbuf,
                               .onchip = 50,
                               .rows = 2,
                               .row_words = 3,
                               .ddr_stride = kStride,
                               .flags = kDmaFlagIndex}),
                     Halt()});
  for (uint32_t e = 0; e < words.size(); ++e) {
    EXPECT_EQ(ReadFbuf(ddr, kDstArea + ((e / 3) * kStride) +
                                ((e % 3) * kFbufWordBytes)),
              words[e])
        << "word " << e;
  }
}

TEST(DmaTest, LoadIgnoresTheIndexFlagWithoutRowFormat) {
  std::mt19937 rng(25);
  NpuBench bench(25);
  Ddr& ddr = bench.Memory();
  std::vector<AbufWord> words;
  for (uint32_t e = 0; e < 2; ++e) {
    words.push_back(RandomAbufWord(rng));
    WriteAbuf(ddr, kSrcArea + (e * kAbufWordBytes), words.back());
  }
  RunProgram(bench, {DmaLoad({.ddr_addr = kSrcArea,
                              .mem = Mem::kAbuf,
                              .onchip = 10,
                              .rows = 1,
                              .row_words = 2,
                              .flags = kDmaFlagIndex}),
                     DmaStore({.ddr_addr = kDstArea,
                               .mem = Mem::kAbuf,
                               .onchip = 10,
                               .rows = 1,
                               .row_words = 2}),
                     Halt()});
  for (uint32_t e = 0; e < words.size(); ++e) {
    EXPECT_EQ(ReadAbuf(ddr, kDstArea + (e * kAbufWordBytes)), words[e])
        << "word " << e;
  }
}

uint32_t MxTokenBytes(uint32_t tok) {
  return (32 * tok) + (16 * ((tok + 15) / 16));
}

struct MxToken {
  std::vector<int8_t> q;
  std::vector<uint8_t> scale;
};

MxToken RandomMxToken(std::mt19937& rng, uint32_t tok) {
  MxToken token;
  for (uint32_t i = 0; i < 32 * tok; ++i) {
    token.q.push_back(static_cast<int8_t>(static_cast<int>(rng() % 256) - 128));
  }
  for (uint32_t b = 0; b < tok; ++b) {
    token.scale.push_back(static_cast<uint8_t>(100 + (rng() % 50)));
  }
  return token;
}

void WriteMxToken(Ddr& ddr, uint32_t address, const MxToken& token,
                  uint32_t tok) {
  for (uint32_t i = 0; i < 32 * tok; ++i) {
    ddr.Write8(address + i, static_cast<uint8_t>(token.q[i]));
  }
  for (uint32_t i = 32 * tok; i < MxTokenBytes(tok); ++i) {
    const uint32_t b = i - (32 * tok);
    ddr.Write8(address + i, b < tok ? token.scale[b] : 0);
  }
}

std::vector<AbufWord> PackMxTokens(const std::vector<MxToken>& tokens,
                                   uint32_t tok) {
  std::vector<AbufWord> words((tokens.size() / 4) * tok);
  for (uint32_t i = 0; i < tokens.size(); ++i) {
    const uint32_t quad = i / 4;
    const uint32_t lane = i % 4;
    for (uint32_t b = 0; b < tok; ++b) {
      AbufWord& word = words[(quad * tok) + b];
      for (uint32_t k = 0; k < 32; ++k) {
        word.q.at((lane * 32) + k) = tokens[i].q[(b * 32) + k];
      }
      word.scale.at(lane) = tokens[i].scale[b];
    }
  }
  return words;
}

TEST(DmaTest, AbufRowFormatLoadPacksTokensIntoQuads) {
  std::mt19937 rng(26);
  NpuBench bench(26);
  Ddr& ddr = bench.Memory();
  constexpr uint32_t kTok = 3;
  constexpr uint32_t kRows = 2;
  constexpr uint32_t kTokensPerRow = 6;
  constexpr uint32_t kStride = 0x2C0;
  std::vector<MxToken> tokens;
  for (uint32_t r = 0; r < kRows; ++r) {
    for (uint32_t w = 0; w < kTokensPerRow; ++w) {
      tokens.push_back(RandomMxToken(rng, kTok));
      WriteMxToken(ddr, kSrcArea + (r * kStride) + (w * MxTokenBytes(kTok)),
                   tokens.back(), kTok);
    }
  }
  const std::vector<AbufWord> expected = PackMxTokens(tokens, kTok);
  RunProgram(bench,
             {DmaLoad({.ddr_addr = kSrcArea,
                       .mem = Mem::kAbuf,
                       .onchip = 33,
                       .rows = kRows,
                       .row_words = kTokensPerRow,
                       .ddr_stride = kStride,
                       .tok = kTok,
                       .flags = kDmaFlagRow}),
              DmaStore({.ddr_addr = kDstArea,
                        .mem = Mem::kAbuf,
                        .onchip = 33,
                        .rows = 1,
                        .row_words = static_cast<uint32_t>(expected.size())}),
              Halt()});
  for (uint32_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(ReadAbuf(ddr, kDstArea + (i * kAbufWordBytes)), expected[i])
        << "word " << i;
  }
}

TEST(DmaTest, AbufRowFormatStoreUnpacksQuads) {
  std::mt19937 rng(27);
  NpuBench bench(27);
  Ddr& ddr = bench.Memory();
  constexpr uint32_t kTok = 17;
  constexpr uint32_t kTokens = 8;
  std::vector<MxToken> tokens;
  tokens.reserve(kTokens);
  for (uint32_t i = 0; i < kTokens; ++i) {
    tokens.push_back(RandomMxToken(rng, kTok));
  }
  const std::vector<AbufWord> words = PackMxTokens(tokens, kTok);
  for (uint32_t i = 0; i < words.size(); ++i) {
    WriteAbuf(ddr, kSrcArea + (i * kAbufWordBytes), words[i]);
  }
  const uint32_t token_bytes = MxTokenBytes(kTok);
  FillGarbage(ddr, kDstArea, kTokens * token_bytes, rng);
  RunProgram(bench,
             {DmaLoad({.ddr_addr = kSrcArea,
                       .mem = Mem::kAbuf,
                       .onchip = 0,
                       .rows = 1,
                       .row_words = static_cast<uint32_t>(words.size())}),
              DmaStore({.ddr_addr = kDstArea,
                        .mem = Mem::kAbuf,
                        .onchip = 0,
                        .rows = 1,
                        .row_words = kTokens,
                        .tok = kTok,
                        .flags = kDmaFlagRow}),
              Halt()});
  for (uint32_t i = 0; i < kTokens; ++i) {
    const uint32_t base = kDstArea + (i * token_bytes);
    for (uint32_t k = 0; k < 32 * kTok; ++k) {
      ASSERT_EQ(static_cast<int8_t>(ddr.Read8(base + k)), tokens[i].q[k])
          << "token " << i << " element " << k;
    }
    for (uint32_t b = 0; b < token_bytes - (32 * kTok); ++b) {
      const uint8_t expected = b < kTok ? tokens[i].scale[b] : 0;
      ASSERT_EQ(ddr.Read8(base + (32 * kTok) + b), expected)
          << "token " << i << " scale byte " << b;
    }
  }
}

TEST(DmaTest, FbufRowFormatLoadAndStore) {
  std::mt19937 rng(28);
  NpuBench bench(28);
  Ddr& ddr = bench.Memory();
  constexpr uint32_t kTok = 2;
  constexpr uint32_t kRows = 3;
  constexpr uint32_t kTokensPerRow = 4;
  constexpr uint32_t kTokenBytes = 128 * kTok;
  constexpr uint32_t kStride = 0x500;
  std::vector<std::vector<uint32_t>> tokens;
  for (uint32_t r = 0; r < kRows; ++r) {
    for (uint32_t w = 0; w < kTokensPerRow; ++w) {
      std::vector<uint32_t> token;
      for (uint32_t i = 0; i < 32 * kTok; ++i) {
        token.push_back(RandomFp32(rng, -10, 10));
        ddr.Write32(kSrcArea + (r * kStride) + (w * kTokenBytes) + (4 * i),
                    token.back());
      }
      tokens.push_back(token);
    }
  }
  RunProgram(bench, {DmaLoad({.ddr_addr = kSrcArea,
                              .mem = Mem::kFbuf,
                              .onchip = 20,
                              .rows = kRows,
                              .row_words = kTokensPerRow,
                              .ddr_stride = kStride,
                              .tok = kTok,
                              .flags = kDmaFlagRow}),
                     DmaStore({.ddr_addr = kDstArea,
                               .mem = Mem::kFbuf,
                               .onchip = 20,
                               .rows = 1,
                               .row_words = 24}),
                     DmaStore({.ddr_addr = kAuxArea,
                               .mem = Mem::kFbuf,
                               .onchip = 20,
                               .rows = 2,
                               .row_words = 6,
                               .ddr_stride = 0x800,
                               .tok = kTok,
                               .flags = kDmaFlagRow}),
                     Halt()});
  for (uint32_t i = 0; i < tokens.size(); ++i) {
    const uint32_t quad = i / 4;
    const uint32_t lane = i % 4;
    for (uint32_t ch = 0; ch < 32 * kTok; ++ch) {
      const uint32_t word = (quad * 4 * kTok) + (ch / 8);
      const uint32_t element = (lane * 8) + (ch % 8);
      ASSERT_EQ(ddr.Read32(kDstArea + (word * kFbufWordBytes) + (4 * element)),
                tokens[i][ch])
          << "token " << i << " channel " << ch;
      const uint32_t stored =
          kAuxArea + ((i / 6) * 0x800) + ((i % 6) * kTokenBytes) + (4 * ch);
      ASSERT_EQ(ddr.Read32(stored), tokens[i][ch])
          << "token " << i << " channel " << ch;
    }
  }
}

TEST(DmaTest, AbufRowFormatIndexListWithNullToken) {
  std::mt19937 rng(29);
  NpuBench bench(29);
  Ddr& ddr = bench.Memory();
  constexpr uint32_t kTok = 2;
  std::vector<MxToken> tokens;
  for (uint32_t e = 0; e < 8; ++e) {
    if (e == 5) {
      MxToken null_token;
      null_token.q.assign(size_t{32} * kTok, 0);
      null_token.scale.assign(kTok, 127);
      tokens.push_back(null_token);
      ddr.Write32(kListArea + (4 * e), 0xFFFFFFFFU);
      continue;
    }
    tokens.push_back(RandomMxToken(rng, kTok));
    const uint32_t address = kSrcArea + ((rng() % 1000) * MxTokenBytes(kTok));
    WriteMxToken(ddr, address, tokens.back(), kTok);
    ddr.Write32(kListArea + (4 * e), address / 16);
  }
  const std::vector<AbufWord> expected = PackMxTokens(tokens, kTok);
  RunProgram(bench,
             {DmaLoad({.ddr_addr = kListArea,
                       .mem = Mem::kAbuf,
                       .onchip = 64,
                       .rows = 2,
                       .row_words = 4,
                       .tok = kTok,
                       .flags = kDmaFlagIndex | kDmaFlagRow}),
              DmaStore({.ddr_addr = kDstArea,
                        .mem = Mem::kAbuf,
                        .onchip = 64,
                        .rows = 1,
                        .row_words = static_cast<uint32_t>(expected.size())}),
              Halt()});
  for (uint32_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(ReadAbuf(ddr, kDstArea + (i * kAbufWordBytes)), expected[i])
        << "word " << i;
  }
}

TEST(DmaTest, FbufStoreWhileGemmAccumulatesInFbuf) {
  constexpr uint32_t kRows = 32;
  for (uint32_t seed = 40; seed < 52; ++seed) {
    std::mt19937 rng(seed);
    NpuBench bench(seed);
    Ddr& ddr = bench.Memory();
    const uint32_t row_words = 1 + (seed % 3);
    const uint32_t store_stride = (row_words + 1) * kFbufWordBytes;
    std::vector<FbufWord> words;
    for (uint32_t i = 0; i < kRows * row_words; ++i) {
      words.push_back(RandomFbufWord(rng));
      WriteFbuf(ddr, kSrcArea + (i * kFbufWordBytes), words.back());
    }
    FillGarbage(ddr, kDstArea, kRows * store_stride, rng);
    std::vector<Instruction> program = {
        DmaLoad({.ddr_addr = kSrcArea,
                 .mem = Mem::kFbuf,
                 .onchip = 0,
                 .rows = 1,
                 .row_words = kRows * row_words}),
        Gemm({.a_base = 0,
              .a_stride = 1,
              .kb = 1,
              .m = 32,
              .w_base = 0,
              .w_stride = 1,
              .n8 = 4,
              .c_base = 2000,
              .acc = true,
              .wait = kWaitDma}),
        DmaStore({.ddr_addr = kDstArea,
                  .mem = Mem::kFbuf,
                  .onchip = 0,
                  .rows = kRows,
                  .row_words = row_words,
                  .ddr_stride = store_stride})};
    for (uint32_t i = 0; i < 200; ++i) {
      program.push_back(Gemm({.a_base = 0,
                              .a_stride = 1,
                              .kb = 1 + static_cast<uint32_t>(rng() % 3),
                              .m = 32,
                              .w_base = 0,
                              .w_stride = 1,
                              .n8 = 1 + static_cast<uint32_t>(rng() % 12),
                              .c_base = 2000,
                              .acc = true}));
    }
    program.push_back(Halt());
    RunProgram(bench, program);
    for (uint32_t i = 0; i < words.size(); ++i) {
      const uint32_t address = kDstArea + ((i / row_words) * store_stride) +
                               ((i % row_words) * kFbufWordBytes);
      EXPECT_EQ(ReadFbuf(ddr, address), words[i])
          << "seed " << seed << " word " << i;
    }
  }
}

}  // namespace
}  // namespace hotarubi::tb
