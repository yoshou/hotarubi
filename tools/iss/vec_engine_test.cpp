#include "vec_engine.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include "encoding.hpp"
#include "hotarubi/exact.hpp"
#include "memories.hpp"
#include "reference.hpp"
#include "word.hpp"

namespace hotarubi::iss {
namespace {

constexpr uint32_t kLog2e = 0x3FB8AA3B;
constexpr uint32_t kEpsilon = 0x3727C5AC;

enum Op : uint8_t {
  kAdd = 0,
  kMul = 1,
  kAddP = 2,
  kMulP = 3,
  kScale = 4,
  kCopy = 5,
  kLayerNorm = 6,
  kRmsNorm = 7,
  kGelu = 8,
  kSilu = 9,
  kSiluMul = 10,
  kSoftmax = 11,
  kRope = 12,
  kQuant = 13,
  kRelu = 14,
  kMax = 15,
};

uint32_t FlipSign(uint32_t bits) { return bits ^ 0x80000000U; }

bool Negative(uint32_t bits) { return (bits >> 31) != 0; }

class VecEngineTest : public testing::Test {
 protected:
  uint32_t Random(uint32_t low_exponent, uint32_t high_exponent) {
    const uint32_t exponent = std::uniform_int_distribution<uint32_t>(
        low_exponent, high_exponent)(generator_);
    const uint32_t fraction =
        std::uniform_int_distribution<uint32_t>(0, 0x7FFFFF)(generator_);
    const uint32_t sign =
        std::uniform_int_distribution<uint32_t>(0, 1)(generator_);
    return (sign << 31) | (exponent << 23) | fraction;
  }

  void FillFbuf(uint32_t first, uint32_t count, uint32_t low, uint32_t high) {
    for (uint32_t w = first; w < first + count; ++w) {
      for (uint32_t& element : memories_.fbuf.at(w)) {
        element = Random(low, high);
      }
    }
  }

  void FillPbuf(uint32_t first, uint32_t count, uint32_t low, uint32_t high) {
    for (uint32_t w = first; w < first + count; ++w) {
      for (uint32_t& element : memories_.pbuf.at(w)) {
        element = Random(low, high);
      }
    }
  }

  void FillTable(uint32_t first, bool fraction_table) {
    for (uint32_t i = 0; i < 512; ++i) {
      const uint32_t v =
          fraction_table
              ? reference::Round(exact::Real::Integer(512 + i).Scaled(-9) +
                                 exact::Real::Integer(i % 7).Scaled(-30))
              : reference::Round(
                    exact::Real::Integer(256 + (i / 2)).Scaled(-9));
      const uint32_t s = (Random(117, 119) & 0x7FFFFFFFU);
      memories_.tbuf.at(first + i) = (uint64_t{s} << 32) | v;
    }
  }

  static uint32_t Element(const std::vector<FbufWord>& words, uint32_t base,
                          uint32_t n8, uint32_t row, uint32_t column) {
    return words.at(base + ((row / kTokens) * n8) + (column / kLanes))
        .at(((row % kTokens) * kLanes) + (column % kLanes));
  }

  [[nodiscard]] uint32_t Output(uint32_t base, uint32_t n8, uint32_t row,
                                uint32_t column) const {
    return Element(memories_.fbuf, base, n8, row, column);
  }

  [[nodiscard]] uint32_t RowVector(uint32_t base, uint32_t column) const {
    return memories_.pbuf.at(base + (column / 32)).at(column % 32);
  }

  [[nodiscard]] static std::vector<std::array<uint32_t, 8>> Groups(
      const std::vector<uint32_t>& row) {
    std::vector<std::array<uint32_t, 8>> groups(row.size() / kLanes);
    for (size_t c = 0; c < row.size(); ++c) {
      groups.at(c / kLanes).at(c % kLanes) = row.at(c);
    }
    return groups;
  }

  void ExpectQuantized(uint32_t base, uint32_t n8, uint32_t quads,
                       const std::vector<std::vector<uint32_t>>& rows) const {
    for (uint32_t q = 0; q < quads; ++q) {
      for (uint32_t b = 0; b < n8 / 4; ++b) {
        const AbufWord& word = memories_.abuf.at(base + (q * (n8 / 4)) + b);
        for (uint32_t t = 0; t < kTokens; ++t) {
          std::array<uint32_t, 32> block{};
          for (uint32_t k = 0; k < 32; ++k) {
            block.at(k) = rows.at((q * kTokens) + t).at((b * 32) + k);
          }
          const reference::Mxint8Block expected =
              reference::EncodeMxint8(block);
          for (uint32_t k = 0; k < 32; ++k) {
            ASSERT_EQ(word.at((t * 32) + k), expected.q.at(k))
                << "quad " << q << " block " << b << " token " << t;
          }
          ASSERT_EQ(word.at(kAbufScaleOffset + t), expected.scale);
        }
      }
    }
  }

  void ExpectTransposed(uint32_t dst, uint32_t tiles, bool zip4) const {
    for (uint32_t q = 0; q < tiles; ++q) {
      for (uint32_t row = 0; row < 8; ++row) {
        const FbufWord& out = memories_.fbuf.at(dst + q + (row * tiles));
        for (uint32_t t = 0; t < 4; ++t) {
          for (uint32_t column = 0; column < 8; ++column) {
            const uint32_t source_row =
                zip4 ? (column / 2) + (4 * (column % 2)) : column;
            ASSERT_EQ(out.at((t * 8) + column),
                      memories_.fbuf.at((q * 8) + source_row).at((t * 8) + row))
                << "tile " << q << " row " << row;
          }
        }
      }
    }
  }

  Memories memories_;
  VecEngine engine_;
  std::mt19937 generator_{7};
};

TEST_F(VecEngineTest, TransposeProgramPermutesFourMatricesPerTile) {
  constexpr uint32_t kTiles = 3;
  constexpr uint32_t kPc = 300;
  constexpr uint32_t kDst = 2048;
  for (uint32_t word = 0; word < 8 * kTiles; ++word) {
    for (uint32_t i = 0; i < 32; ++i) {
      memories_.fbuf.at(word).at(i) = generator_();
    }
  }
  for (const bool zip4 : {false, true}) {
    Word128 transpose;
    encoding::Set(transpose, 124, 4, 7);
    encoding::Set(transpose, 119, 5, 1);
    encoding::Set(transpose, 114, 5, 11);
    encoding::Set(transpose, 109, 5, 3);
    encoding::Set(transpose, 104, 5, 4);
    encoding::Set(transpose, 99, 5, 11);
    encoding::Set(transpose, 94, 5, 5);
    encoding::Set(transpose, 89, 5, 4);
    encoding::Set(transpose, 88, 1, zip4 ? 1 : 0);
    memories_.cbuf.at(kPc) = encoding::SaluImm(12, 11, 0, 1);
    memories_.cbuf.at(kPc + 1) = transpose;
    memories_.cbuf.at(kPc + 2) = encoding::Ret();
    engine_.Execute({.op = kVecRun,
                     .src = 0,
                     .dst = kDst,
                     .rows_q = kTiles,
                     .n8 = 8,
                     .aux = kPc},
                    memories_);
    ExpectTransposed(kDst, kTiles, zip4);
    EXPECT_EQ(engine_.Activity().transposes, 1U);
    EXPECT_EQ(engine_.Activity().transpose_tiles, kTiles);
    EXPECT_EQ(engine_.Activity().salu, 1U);
  }
}

TEST_F(VecEngineTest, ElementwiseOperations) {
  constexpr uint32_t kQuads = 3;
  constexpr uint32_t kN8 = 5;
  constexpr uint32_t kSrc = 100;
  constexpr uint32_t kSrc2 = 300;
  constexpr uint32_t kDst = 500;
  constexpr uint32_t kPBase = 20;
  const uint32_t imm = Random(120, 130);
  for (const uint32_t op :
       {kAdd, kMul, kAddP, kMulP, kScale, kCopy, kRelu, kMax}) {
    FillFbuf(kSrc, kQuads * kN8, 110, 140);
    FillFbuf(kSrc2, kQuads * kN8, 110, 140);
    FillPbuf(kPBase, 2, 110, 140);
    const std::vector<FbufWord> before = memories_.fbuf;
    engine_.Execute({.op = op,
                     .quantize = false,
                     .src = kSrc,
                     .src2 = kSrc2,
                     .dst = kDst,
                     .rows_q = kQuads,
                     .n8 = kN8,
                     .p_base = kPBase,
                     .aux = 0,
                     .imm = imm},
                    memories_);
    for (uint32_t row = 0; row < kQuads * kTokens; ++row) {
      for (uint32_t column = 0; column < kN8 * kLanes; ++column) {
        const uint32_t x = Element(before, kSrc, kN8, row, column);
        const uint32_t x2 = Element(before, kSrc2, kN8, row, column);
        const uint32_t p = RowVector(kPBase, column);
        uint32_t expected = 0;
        switch (op) {
          case kAdd:
            expected = reference::Add(x, x2);
            break;
          case kMul:
            expected = reference::Mul(x, x2);
            break;
          case kAddP:
            expected = reference::Add(x, p);
            break;
          case kMulP:
            expected = reference::Mul(x, p);
            break;
          case kScale:
            expected = reference::Mul(x, imm);
            break;
          case kRelu:
            expected = Negative(x) ? 0 : x;
            break;
          case kMax:
            expected = reference::RowMax(std::vector<uint32_t>{x, x2});
            break;
          default:
            expected = x;
            break;
        }
        ASSERT_EQ(Output(kDst, kN8, row, column), expected)
            << "op " << op << " row " << row << " column " << column;
      }
    }
  }
}

TEST_F(VecEngineTest, QuantizesToMxint8) {
  constexpr uint32_t kQuads = 2;
  constexpr uint32_t kN8 = 8;
  FillFbuf(40, kQuads * kN8, 100, 140);
  for (uint32_t w = 52; w < 56; ++w) {
    memories_.fbuf.at(w).fill(0);
  }
  for (uint32_t& element : memories_.fbuf.at(40 + 5)) {
    element = 0;
  }
  memories_.fbuf.at(40 + 6).at(3) = 0x00000001;
  std::vector<std::vector<uint32_t>> rows(size_t{kQuads} * kTokens);
  for (uint32_t row = 0; row < rows.size(); ++row) {
    for (uint32_t column = 0; column < kN8 * kLanes; ++column) {
      rows.at(row).push_back(Output(40, kN8, row, column));
    }
  }
  engine_.Execute({.op = kQuant,
                   .quantize = true,
                   .src = 40,
                   .src2 = 0,
                   .dst = 7000,
                   .rows_q = kQuads,
                   .n8 = kN8,
                   .p_base = 0,
                   .aux = 0,
                   .imm = 3000},
                  memories_);
  ExpectQuantized(7000, kN8, kQuads, rows);
}

class LayerNormTest : public VecEngineTest,
                      public testing::WithParamInterface<bool> {};

TEST_P(LayerNormTest, NormalizesEachRow) {
  const bool quantize = GetParam();
  constexpr uint32_t kQuads = 2;
  constexpr uint32_t kN8 = 8;
  constexpr uint32_t kSrc = 64;
  constexpr uint32_t kScratch = 900;
  constexpr uint32_t kPBase = 30;
  const uint32_t dst = quantize ? 5000 : 640;
  FillFbuf(kSrc, kQuads * kN8, 120, 132);
  FillPbuf(kPBase, 4, 124, 128);
  const std::vector<FbufWord> before = memories_.fbuf;
  engine_.Execute({.op = kLayerNorm,
                   .quantize = quantize,
                   .src = kSrc,
                   .src2 = kScratch,
                   .dst = dst,
                   .rows_q = kQuads,
                   .n8 = kN8,
                   .p_base = kPBase,
                   .aux = 0,
                   .imm = kEpsilon},
                  memories_);
  const uint32_t n = kN8 * kLanes;
  const uint32_t rn = reference::Reciprocal(reference::FromInt(n));
  std::vector<std::vector<uint32_t>> rows(size_t{kQuads} * kTokens);
  for (uint32_t row = 0; row < rows.size(); ++row) {
    std::vector<uint32_t> x(n);
    for (uint32_t c = 0; c < n; ++c) {
      x.at(c) = Element(before, kSrc, kN8, row, c);
    }
    const uint32_t negative_mean =
        reference::Negate(reference::Mul(reference::RowSum(Groups(x)), rn));
    std::vector<uint32_t> d(n);
    std::vector<uint32_t> squares(n);
    for (uint32_t c = 0; c < n; ++c) {
      d.at(c) = reference::Add(x.at(c), negative_mean);
      squares.at(c) = reference::Mul(d.at(c), d.at(c));
    }
    const uint32_t rstd = reference::ReciprocalSqrt(reference::Add(
        reference::Mul(reference::RowSum(Groups(squares)), rn), kEpsilon));
    for (uint32_t c = 0; c < n; ++c) {
      const uint32_t scaled =
          reference::Mul(reference::Mul(d.at(c), rstd), RowVector(kPBase, c));
      rows.at(row).push_back(
          reference::Add(scaled, RowVector(kPBase + (kN8 / 4), c)));
      if (!quantize) {
        ASSERT_EQ(Output(dst, kN8, row, c), rows.at(row).back())
            << "row " << row << " column " << c;
      }
    }
  }
  if (quantize) {
    ExpectQuantized(dst, kN8, kQuads, rows);
  }
}

INSTANTIATE_TEST_SUITE_P(VecEngine, LayerNormTest, testing::Bool());

TEST_F(VecEngineTest, RmsNormNormalizesEachRow) {
  constexpr uint32_t kQuads = 3;
  constexpr uint32_t kN8 = 4;
  constexpr uint32_t kSrc = 10;
  constexpr uint32_t kDst = 60;
  constexpr uint32_t kPBase = 3;
  constexpr uint32_t kCount = 29;
  FillFbuf(kSrc, kQuads * kN8, 118, 134);
  FillPbuf(kPBase, 1, 124, 128);
  const std::vector<FbufWord> before = memories_.fbuf;
  engine_.Execute({.op = kRmsNorm,
                   .quantize = false,
                   .src = kSrc,
                   .src2 = 0,
                   .dst = kDst,
                   .rows_q = kQuads,
                   .n8 = kN8,
                   .p_base = kPBase,
                   .aux = kCount,
                   .imm = kEpsilon},
                  memories_);
  const uint32_t n = kN8 * kLanes;
  const uint32_t rn = reference::Reciprocal(reference::FromInt(kCount));
  for (uint32_t row = 0; row < kQuads * kTokens; ++row) {
    std::vector<uint32_t> squares(n);
    for (uint32_t c = 0; c < n; ++c) {
      const uint32_t x = Element(before, kSrc, kN8, row, c);
      squares.at(c) = reference::Mul(x, x);
    }
    const uint32_t rstd = reference::ReciprocalSqrt(reference::Add(
        reference::Mul(reference::RowSum(Groups(squares)), rn), kEpsilon));
    for (uint32_t c = 0; c < n; ++c) {
      const uint32_t x = Element(before, kSrc, kN8, row, c);
      ASSERT_EQ(Output(kDst, kN8, row, c),
                reference::Mul(reference::Mul(x, rstd), RowVector(kPBase, c)))
          << "row " << row << " column " << c;
    }
  }
}

TEST_F(VecEngineTest, SoftmaxOverTheValidColumns) {
  constexpr uint32_t kQuads = 2;
  constexpr uint32_t kN8 = 8;
  constexpr uint32_t kSrc = 200;
  constexpr uint32_t kDst = 400;
  constexpr uint32_t kValid = 50;
  FillFbuf(kSrc, kQuads * kN8, 120, 131);
  FillTable(0, true);
  const std::vector<FbufWord> before = memories_.fbuf;
  engine_.Execute({.op = kSoftmax,
                   .quantize = false,
                   .src = kSrc,
                   .src2 = 0,
                   .dst = kDst,
                   .rows_q = kQuads,
                   .n8 = kN8,
                   .p_base = 0,
                   .aux = kValid,
                   .imm = 0},
                  memories_);
  const uint32_t n = kN8 * kLanes;
  for (uint32_t row = 0; row < kQuads * kTokens; ++row) {
    std::vector<uint32_t> x(kValid);
    for (uint32_t c = 0; c < kValid; ++c) {
      x.at(c) = Element(before, kSrc, kN8, row, c);
    }
    const uint32_t negative_max = reference::Negate(reference::RowMax(x));
    std::vector<uint32_t> e(n);
    for (uint32_t c = 0; c < kValid; ++c) {
      e.at(c) = reference::Pow2(
          reference::Mul(reference::Add(x.at(c), negative_max), kLog2e),
          memories_.tbuf);
    }
    const uint32_t inverse =
        reference::Reciprocal(reference::RowSum(Groups(e)));
    for (uint32_t c = 0; c < n; ++c) {
      ASSERT_EQ(Output(kDst, kN8, row, c),
                c < kValid ? reference::Mul(e.at(c), inverse) : 0)
          << "row " << row << " column " << c;
    }
  }
}

class ActivationTest : public VecEngineTest,
                       public testing::WithParamInterface<uint32_t> {};

TEST_P(ActivationTest, MultipliesByTheTabulatedFunction) {
  const uint32_t op = GetParam();
  constexpr uint32_t kQuads = 2;
  constexpr uint32_t kN8 = 3;
  constexpr uint32_t kSrc = 0;
  constexpr uint32_t kUp = 20;
  constexpr uint32_t kDst = 40;
  FillFbuf(kSrc, kQuads * kN8, 110, 132);
  FillFbuf(kUp, kQuads * kN8, 120, 130);
  memories_.fbuf.at(kSrc).at(0) = 0x41800000;
  memories_.fbuf.at(kSrc).at(1) = 0xC1800001;
  memories_.fbuf.at(kSrc).at(2) = 0x417FFFFF;
  FillTable(512, false);
  const std::vector<FbufWord> before = memories_.fbuf;
  engine_.Execute({.op = op,
                   .quantize = false,
                   .src = kSrc,
                   .src2 = kUp,
                   .dst = kDst,
                   .rows_q = kQuads,
                   .n8 = kN8,
                   .p_base = 0,
                   .aux = 0,
                   .imm = 0},
                  memories_);
  for (uint32_t row = 0; row < kQuads * kTokens; ++row) {
    for (uint32_t c = 0; c < kN8 * kLanes; ++c) {
      const uint32_t x = Element(before, kSrc, kN8, row, c);
      const uint32_t product =
          reference::Mul(x, reference::Sigma(x, memories_.tbuf));
      uint32_t expected =
          Negative(x) ? reference::Add(FlipSign(product), x) : product;
      if (op == kSiluMul) {
        expected = reference::Mul(expected, Element(before, kUp, kN8, row, c));
      }
      ASSERT_EQ(Output(kDst, kN8, row, c), expected)
          << "row " << row << " column " << c;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(VecEngine, ActivationTest,
                         testing::Values(kGelu, kSilu, kSiluMul));

TEST_F(VecEngineTest, RopeRotatesHalfPairs) {
  constexpr uint32_t kQuads = 2;
  constexpr uint32_t kHeads = 2;
  constexpr uint32_t kN8 = 16;
  constexpr uint32_t kSrc = 100;
  constexpr uint32_t kDst = 300;
  constexpr uint32_t kPBase = 40;
  FillFbuf(kSrc, kQuads * kN8, 118, 132);
  FillPbuf(kPBase, 2 * kQuads * 4, 120, 127);
  const std::vector<FbufWord> before = memories_.fbuf;
  engine_.Execute({.op = kRope,
                   .quantize = false,
                   .src = kSrc,
                   .src2 = 0,
                   .dst = kDst,
                   .rows_q = kQuads,
                   .n8 = kN8,
                   .p_base = kPBase,
                   .aux = kHeads,
                   .imm = 0},
                  memories_);
  for (uint32_t q = 0; q < kQuads; ++q) {
    for (uint32_t h = 0; h < kHeads; ++h) {
      for (uint32_t g = 0; g < 4; ++g) {
        const uint32_t first = kSrc + (q * kN8) + (h * 8) + g;
        const uint32_t out = kDst + (q * kN8) + (h * 8) + g;
        const FbufWord& cosines = memories_.pbuf.at(kPBase + (q * 4) + g);
        const FbufWord& sines =
            memories_.pbuf.at(kPBase + (kQuads * 4) + (q * 4) + g);
        for (uint32_t e = 0; e < kFbufElements; ++e) {
          const uint32_t x1 = before.at(first).at(e);
          const uint32_t x2 = before.at(first + 4).at(e);
          ASSERT_EQ(memories_.fbuf.at(out).at(e),
                    reference::Add(reference::Mul(x1, cosines.at(e)),
                                   FlipSign(reference::Mul(x2, sines.at(e)))));
          ASSERT_EQ(memories_.fbuf.at(out + 4).at(e),
                    reference::Add(reference::Mul(x2, cosines.at(e)),
                                   reference::Mul(x1, sines.at(e))));
        }
      }
    }
  }
}

}  // namespace
}  // namespace hotarubi::iss
