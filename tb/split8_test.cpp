#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include "gemm_ref.hpp"
#include "npu_bench.hpp"
#include "npu_data.hpp"
#include "npu_isa.hpp"

namespace hotarubi::tb {
namespace {

constexpr uint32_t kSrcArea = Ddr::kBase + 0x100000U;
constexpr uint32_t kAArea = Ddr::kBase + 0x300000U;
constexpr uint32_t kOutArea = Ddr::kBase + 0x500000U;
constexpr uint32_t kSrcStride = 0x400;

using Matrix = std::vector<std::vector<AbufWord>>;

Matrix RandomMatrix(std::mt19937& rng, uint32_t rows, uint32_t row_words) {
  Matrix matrix(rows, std::vector<AbufWord>(row_words));
  for (std::vector<AbufWord>& row : matrix) {
    for (AbufWord& word : row) {
      for (uint32_t j = 0; j < 4; ++j) {
        const bool zero_token = rng() % 7 == 0;
        for (uint32_t k = 0; k < 32; ++k) {
          word.q.at((j * 32) + k) = static_cast<int8_t>(
              zero_token ? 0 : static_cast<int>(rng() % 256) - 128);
        }
        word.scale.at(j) = static_cast<uint8_t>(120 + (rng() % 16));
      }
    }
  }
  return matrix;
}

void WriteMatrix(Ddr& ddr, const Matrix& matrix) {
  for (uint32_t r = 0; r < matrix.size(); ++r) {
    for (uint32_t w = 0; w < matrix[r].size(); ++w) {
      WriteAbuf(ddr, kSrcArea + (r * kSrcStride) + (w * kAbufWordBytes),
                matrix[r][w]);
    }
  }
}

void SetNibbles(WbufWord& high, WbufWord& low, uint32_t index, int value) {
  high.nibble.at(index) = static_cast<uint8_t>((value >> 4) & 0xF);
  low.nibble.at(index) = static_cast<uint8_t>(value & 0xF);
}

std::vector<WbufWord> ExpectedSplit8(const Matrix& matrix, uint32_t dst) {
  const auto rows = static_cast<uint32_t>(matrix.size());
  const auto row_words = static_cast<uint32_t>(matrix[0].size());
  const uint32_t plane = rows * row_words / 2;
  std::vector<WbufWord> wbuf(dst + (2 * plane));
  for (uint32_t p = 0; p < rows / 2; ++p) {
    for (uint32_t w = 0; w < row_words; ++w) {
      WbufWord& high = wbuf.at(dst + (p * row_words) + w);
      WbufWord& low = wbuf.at(dst + plane + (p * row_words) + w);
      for (uint32_t c = 0; c < 8; ++c) {
        const AbufWord& source = matrix[(2 * p) + (c / 4)][w];
        const uint32_t token = c % 4;
        for (uint32_t k = 0; k < 32; ++k) {
          SetNibbles(high, low, (c * 32) + k, source.q.at((token * 32) + k));
        }
        high.scale.at(c) = source.scale.at(token);
        low.scale.at(c) = source.scale.at(token);
      }
    }
  }
  return wbuf;
}

int Element(const AbufWord& word, uint32_t index) {
  const auto raw = static_cast<uint8_t>(word.q.at(index));
  return raw >= 128 ? static_cast<int>(raw) - 256 : static_cast<int>(raw);
}

int RoundShift(int q, int shift) {
  if (shift >= 9) {
    return 0;
  }
  if (shift <= 0) {
    return q;
  }
  const int magnitude = q < 0 ? -q : q;
  int kept = magnitude >> shift;
  const int rest = magnitude & ((1 << shift) - 1);
  const int half = 1 << (shift - 1);
  if (rest > half || (rest == half && (kept & 1) != 0)) {
    ++kept;
  }
  return q < 0 ? -kept : kept;
}

std::vector<WbufWord> ExpectedT32Split8(const Matrix& matrix, uint32_t dst) {
  const auto rows = static_cast<uint32_t>(matrix.size());
  const auto row_words = static_cast<uint32_t>(matrix[0].size());
  const uint32_t plane = rows * row_words / 2;
  const uint32_t groups = rows / 8;
  std::vector<WbufWord> wbuf(dst + (2 * plane));
  for (uint32_t r8 = 0; r8 < groups; ++r8) {
    for (uint32_t w = 0; w < row_words; ++w) {
      int emax = 0;
      bool any = false;
      for (uint32_t t = 0; t < 32; ++t) {
        const AbufWord& source = matrix[(8 * r8) + (t / 4)][w];
        const uint32_t j = t % 4;
        const bool nonzero =
            std::ranges::any_of(std::span(source.q).subspan(size_t{32} * j, 32),
                                [](int8_t q) { return q != 0; });
        const int e = static_cast<int>(source.scale.at(j)) - 127;
        if (nonzero && (!any || e > emax)) {
          emax = e;
          any = true;
        }
      }
      for (uint32_t cg = 0; cg < 4; ++cg) {
        const uint32_t index = (((4 * w) + cg) * groups) + r8;
        WbufWord& high = wbuf.at(dst + index);
        WbufWord& low = wbuf.at(dst + plane + index);
        for (uint32_t c = 0; c < 8; ++c) {
          high.scale.at(c) = static_cast<uint8_t>(emax + 127);
          low.scale.at(c) = static_cast<uint8_t>(emax + 127);
          for (uint32_t t = 0; t < 32; ++t) {
            const AbufWord& source = matrix[(8 * r8) + (t / 4)][w];
            const uint32_t j = t % 4;
            const int e = static_cast<int>(source.scale.at(j)) - 127;
            const int q = Element(source, (j * 32) + (8 * cg) + c);
            SetNibbles(high, low, (c * 32) + t, RoundShift(q, emax - e));
          }
        }
      }
    }
  }
  return wbuf;
}

std::vector<AbufWord> RandomA(std::mt19937& rng, uint32_t count) {
  std::vector<AbufWord> words(count);
  for (AbufWord& word : words) {
    for (int8_t& q : word.q) {
      q = static_cast<int8_t>(static_cast<int>(rng() % 256) - 128);
    }
    for (uint8_t& scale : word.scale) {
      scale = static_cast<uint8_t>(120 + (rng() % 16));
    }
  }
  return words;
}

void CheckThroughGemm(std::mt19937& rng, uint32_t seed, const Matrix& matrix,
                      DmaKind kind, const std::vector<WbufWord>& expected,
                      GemmArgs high) {
  const auto rows = static_cast<uint32_t>(matrix.size());
  const auto row_words = static_cast<uint32_t>(matrix[0].size());
  const uint32_t plane = rows * row_words / 2;
  const std::vector<AbufWord> a = RandomA(rng, high.kb * (high.m / 4));
  GemmArgs low = high;
  low.wmode = kWmodeLow;
  low.w_base = high.w_base + plane;
  low.acc = true;
  const uint32_t out_words = (high.m / 4) * high.n8;
  std::vector<FbufWord> reference(out_words);
  ReferenceGemm(high, a, expected, reference);
  ReferenceGemm(low, a, expected, reference);

  NpuBench bench(seed);
  Ddr& ddr = bench.Memory();
  WriteMatrix(ddr, matrix);
  for (uint32_t i = 0; i < a.size(); ++i) {
    WriteAbuf(ddr, kAArea + (i * kAbufWordBytes), a[i]);
  }
  high.wait = kWaitDma;
  const std::vector<Instruction> program = {
      DmaLoad({.ddr_addr = kSrcArea,
               .mem = Mem::kWbuf,
               .onchip = high.w_base,
               .rows = rows,
               .row_words = row_words,
               .ddr_stride = kSrcStride,
               .kind = kind}),
      DmaLoad({.ddr_addr = kAArea,
               .mem = Mem::kAbuf,
               .onchip = 0,
               .rows = 1,
               .row_words = static_cast<uint32_t>(a.size())}),
      Gemm(high),
      Gemm(low),
      DmaStore({.ddr_addr = kOutArea,
                .mem = Mem::kFbuf,
                .onchip = 0,
                .rows = 1,
                .row_words = out_words,
                .wait = kWaitGemm}),
      Halt()};
  ASSERT_TRUE(bench.Run(program));
  EXPECT_EQ(bench.ReadReg(0x04) & 0x6U, 0x2U);
  EXPECT_TRUE(bench.Errors().empty());
  for (uint32_t i = 0; i < out_words; ++i) {
    ASSERT_EQ(ReadFbuf(ddr, kOutArea + (i * kFbufWordBytes)), reference[i])
        << "word " << i;
  }
}

TEST(Split8Test, SplitsRowPairsIntoHighAndLowPlanes) {
  std::mt19937 rng(41);
  constexpr uint32_t kRows = 6;
  constexpr uint32_t kRowWords = 3;
  constexpr uint32_t kDst = 5;
  const Matrix matrix = RandomMatrix(rng, kRows, kRowWords);
  CheckThroughGemm(rng, 41, matrix, DmaKind::kSplit8,
                   ExpectedSplit8(matrix, kDst),
                   {.a_base = 0,
                    .a_stride = kRowWords,
                    .kb = kRowWords,
                    .m = 32,
                    .w_base = kDst,
                    .w_stride = kRowWords,
                    .n8 = kRows / 2,
                    .c_base = 0,
                    .wmode = kWmodeHigh,
                    .acc = false});
}

TEST(Split8Test, T32TransposesAndAlignsScales) {
  std::mt19937 rng(42);
  constexpr uint32_t kRows = 16;
  constexpr uint32_t kRowWords = 2;
  constexpr uint32_t kDst = 3;
  const Matrix matrix = RandomMatrix(rng, kRows, kRowWords);
  CheckThroughGemm(rng, 42, matrix, DmaKind::kT32Split8,
                   ExpectedT32Split8(matrix, kDst),
                   {.a_base = 0,
                    .a_stride = kRows / 8,
                    .kb = kRows / 8,
                    .m = 32,
                    .w_base = kDst,
                    .w_stride = kRows / 8,
                    .n8 = 4 * kRowWords,
                    .c_base = 0,
                    .wmode = kWmodeHigh,
                    .acc = false});
}

}  // namespace
}  // namespace hotarubi::tb
