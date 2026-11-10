#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "encoding.hpp"
#include "hotarubi/exact.hpp"
#include "memories.hpp"
#include "reference.hpp"
#include "vec_engine.hpp"
#include "word.hpp"

namespace hotarubi::iss {
namespace {

using encoding::PassFields;

constexpr uint32_t kProgram = 256;
constexpr uint32_t kRun = 16;
constexpr uint32_t kNull = 0xFFFFFFFF;

enum Register : uint8_t {
  kSrc = 1,
  kSrc2 = 2,
  kDst = 3,
  kQuads = 4,
  kN8 = 5,
  kPBase = 6,
  kImm = 8,
  kR11 = 11,
  kR12 = 12,
  kR13 = 13,
  kR14 = 14,
};

PassFields Copy() {
  return {
      .rx = kSrc, .rd = kDst, .rq = kQuads, .rnw = kN8, .rn8 = kN8, .wr = 1};
}

int Order(uint32_t a, uint32_t b) {
  const exact::Real difference = reference::Value(a) - reference::Value(b);
  if (difference.IsZero()) {
    return 0;
  }
  return difference.Negative() ? -1 : 1;
}

bool Relation(uint32_t cmp, int order) {
  switch (cmp) {
    case 1:
      return order < 0;
    case 2:
      return order <= 0;
    case 3:
      return order == 0;
    case 4:
      return order != 0;
    case 5:
      return order > 0;
    case 6:
      return order >= 0;
    default:
      return true;
  }
}

int IntegerOrder(uint32_t a, uint32_t b) {
  const auto left = static_cast<int32_t>(a);
  const auto right = static_cast<int32_t>(b);
  if (left == right) {
    return 0;
  }
  return left < right ? -1 : 1;
}

struct CompareCase {
  uint32_t cmp = 0;
  uint32_t source = 0;
  uint32_t integer = 0;
  uint32_t sel = 0;
  uint32_t imm2 = 0;
};

bool Holds(const CompareCase& c, uint32_t x, uint32_t x2, uint32_t position) {
  uint32_t left = x;
  if (c.source == 3) {
    left = x2;
  } else if (c.source == 2) {
    left = c.integer != 0 ? position : position % 8;
  }
  const uint32_t right = c.source == 1 ? x2 : c.imm2;
  if (c.integer != 0 || c.source == 2) {
    return Relation(c.cmp, IntegerOrder(left, right));
  }
  return Relation(c.cmp, Order(left, right));
}

uint32_t PostExpected(uint32_t c1, bool negate, uint32_t x, uint32_t x2) {
  if (c1 == 1) {
    return reference::Add(x, negate ? x2 ^ 0x80000000U : x2);
  }
  return reference::Add(reference::RowMax(std::vector<uint32_t>{
                            reference::Add(x, 0), negate ? 0U : x2}),
                        0);
}

struct RowExpectation {
  uint32_t sum = 0;
  uint32_t squares = 0;
  uint32_t max = 0;
};

class VecPassTest : public testing::Test {
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

  void Fill(uint32_t first, uint32_t count) {
    for (uint32_t w = first; w < first + count; ++w) {
      for (uint32_t& element : memories_.fbuf.at(w)) {
        element = Random(115, 135);
      }
    }
  }

  void Run(const std::vector<Word128>& program,
           std::pair<uint32_t, uint32_t> src, uint32_t dst, uint32_t quads,
           uint32_t n8, uint32_t imm = 0, uint32_t p_base = 0) {
    for (size_t i = 0; i < program.size(); ++i) {
      memories_.cbuf.at(kProgram + i) = program.at(i);
    }
    before_ = memories_.fbuf;
    engine_.Execute({.op = kRun,
                     .quantize = false,
                     .src = src.first,
                     .src2 = src.second,
                     .dst = dst,
                     .rows_q = quads,
                     .n8 = n8,
                     .p_base = p_base,
                     .aux = kProgram,
                     .imm = imm},
                    memories_);
  }

  void RunMarkAndSkip(uint32_t quads, uint32_t invisible, bool overlap) {
    Fill(0, quads);
    for (uint32_t q = 0; q < invisible; ++q) {
      memories_.fbuf.at(q).fill(0);
    }
    PassFields mark = Copy();
    mark.wr = 0;
    mark.ialu = 4;
    mark.rm = 2;
    PassFields skip = Copy();
    skip.rm = 1;
    std::vector<Word128> program = {encoding::Pass(mark), encoding::Pass(skip)};
    if (overlap) {
      PassFields second = skip;
      second.rd = kSrc2;
      program.push_back(encoding::Pass(second));
    }
    program.push_back(encoding::Ret());
    Run(program, {0, 400}, 200, quads, 1);
  }

  [[nodiscard]] uint32_t Before(uint32_t word, uint32_t element) const {
    return before_.at(word).at(element);
  }

  [[nodiscard]] uint32_t After(uint32_t word, uint32_t element) const {
    return memories_.fbuf.at(word).at(element);
  }

  void CheckSelect(const CompareCase& c,
                   const std::array<uint32_t, kTokens>& rs1) const {
    for (uint32_t w = 0; w < 4; ++w) {
      for (uint32_t e = 0; e < kFbufElements; ++e) {
        const uint32_t x = Before(w, e);
        const uint32_t x2 = Before(10 + w, e);
        const std::array<uint32_t, 6> alternatives = {
            0, 0, x2, 0x5A5A5A5A, kNull, rs1.at(e / 8)};
        ASSERT_EQ(After(100 + w, e), Holds(c, x, x2, e)
                                         ? reference::Add(x, 0)
                                         : alternatives.at(c.sel))
            << "cmp " << c.cmp << " source " << c.source << " integer "
            << c.integer << " word " << w << " element " << e;
      }
    }
  }

  [[nodiscard]] RowExpectation ExpectRow(uint32_t t, uint32_t words,
                                         uint32_t valid_columns) const {
    std::vector<std::array<uint32_t, 8>> sums(words);
    std::vector<std::array<uint32_t, 8>> squares(words);
    std::vector<uint32_t> valid;
    for (uint32_t w = 0; w < words; ++w) {
      for (uint32_t l = 0; l < kLanes; ++l) {
        const uint32_t x = reference::Add(Before(w, (t * 8) + l), 0);
        const bool in = (w * 8) + l < valid_columns;
        sums.at(w).at(l) = in ? x : 0;
        squares.at(w).at(l) = in ? reference::Mul(x, x) : 0;
        if (in) {
          valid.push_back(x);
        }
      }
    }
    return {.sum = reference::RowSum(sums),
            .squares = reference::RowSum(squares),
            .max = reference::RowMax(valid)};
  }

  void CheckMasked(uint32_t sc, uint32_t words, uint32_t valid_columns) const {
    for (uint32_t w = 0; w < words; ++w) {
      for (uint32_t e = 0; e < kFbufElements; ++e) {
        const uint32_t x = Before(w, e);
        const uint32_t y =
            sc == 2 ? reference::Mul(x, x) : reference::Add(x, 0);
        ASSERT_EQ(After(100 + w, e), (w * 8) + (e % 8) < valid_columns ? y : 0U)
            << "sc " << sc;
      }
    }
  }

  Memories memories_;
  VecEngine engine_;
  std::vector<FbufWord> before_;
  std::mt19937 generator_{11};
};

TEST_F(VecPassTest, PermutesTokensAndColumns) {
  const auto source = [](uint32_t px, uint32_t position) -> uint32_t {
    const uint32_t t = position / 8;
    const uint32_t l = position % 8;
    switch (px) {
      case 1:
        return (t * 8) + 5;
      case 2:
        return ((t ^ 1U) * 8) + l;
      case 3:
        return ((t ^ 2U) * 8) + l;
      case 4:
        return ((position % 4) * 8) + (position / 4);
      case 5:
        return (((position / 4) % 4) * 8) + (position % 4) +
               (4 * (position / 16));
      default:
        return position;
    }
  };
  for (uint32_t px = 0; px < 8; ++px) {
    Fill(0, 6);
    PassFields pass = Copy();
    pass.px = px;
    pass.pcol = 5;
    Run({encoding::Pass(pass), encoding::Ret()}, {0, 0}, 100, 2, 3);
    for (uint32_t w = 0; w < 6; ++w) {
      for (uint32_t e = 0; e < kFbufElements; ++e) {
        ASSERT_EQ(After(100 + w, e), Before(w, source(px < 6 ? px : 0, e)))
            << "px " << px << " word " << w << " element " << e;
      }
    }
  }
}

TEST_F(VecPassTest, ComparesAndSelects) {
  Fill(0, 4);
  Fill(10, 4);
  const uint32_t threshold = memories_.fbuf.at(2).at(9);
  memories_.fbuf.at(0).at(0) = 0x80000000;
  memories_.fbuf.at(0).at(1) = 0x00000005;
  memories_.fbuf.at(1).at(4) = threshold;
  memories_.fbuf.at(10).at(4) = memories_.fbuf.at(0).at(4);
  const std::array<uint32_t, kTokens> rs1 = {0x11111111, 0x22222222, 0x33333333,
                                             0x44444444};
  for (uint32_t t = 0; t < kTokens; ++t) {
    for (uint32_t l = 0; l < kLanes; ++l) {
      memories_.fbuf.at(50).at((t * kLanes) + l) = rs1.at(t);
    }
  }
  PassFields row_max = Copy();
  row_max.rx = kR13;
  row_max.rnw = kR14;
  row_max.rn8 = kR14;
  row_max.wr = 0;
  row_max.red = 2;
  row_max.sc = 5;
  for (uint32_t cmp = 1; cmp <= 7; ++cmp) {
    for (uint32_t source = 0; source < 4; ++source) {
      for (uint32_t integer = 0; integer < 2; ++integer) {
        const CompareCase c{.cmp = cmp,
                            .source = source,
                            .integer = integer,
                            .sel = 1 + ((cmp + source + integer) % 5),
                            .imm2 = source == 2 ? 13 : threshold};
        PassFields pass = Copy();
        pass.rx2 = kSrc2;
        pass.x2_en = 1;
        pass.rimm = kR12;
        pass.rimm2 = kR11;
        pass.cmp = c.cmp;
        pass.cmp_src = c.source;
        pass.cmp_int = c.integer;
        pass.sel = c.sel;
        Run({encoding::SaluImm(12, kR11, 0, c.imm2),
             encoding::SaluImm(12, kR12, 0, 0x5A5A5A5A),
             encoding::SaluImm(12, kR13, 0, 50),
             encoding::SaluImm(12, kR14, 0, 1), encoding::Pass(row_max),
             encoding::Pass(pass), encoding::Ret()},
            {0, 10}, 100, 1, 4);
        CheckSelect(c, rs1);
      }
    }
  }
}

TEST_F(VecPassTest, ZeroesTheAddendOfFalseLanes) {
  Fill(0, 2);
  const uint32_t threshold = memories_.fbuf.at(1).at(17);
  const uint32_t imm = Random(126, 128);
  PassFields pass = Copy();
  pass.rimm = kImm;
  pass.rimm2 = kR11;
  pass.c0 = 4;
  pass.c0_pred = 1;
  pass.cmp = 5;
  Run({encoding::SaluImm(12, kR11, 0, threshold), encoding::Pass(pass),
       encoding::Ret()},
      {0, 0}, 100, 1, 2, imm);
  for (uint32_t w = 0; w < 2; ++w) {
    for (uint32_t e = 0; e < kFbufElements; ++e) {
      const uint32_t x = Before(w, e);
      const bool predicate = Order(x, threshold) > 0;
      ASSERT_EQ(After(100 + w, e), reference::Add(x, predicate ? imm : 0));
    }
  }
}

TEST_F(VecPassTest, IntegerPathBypassesTheFloatingPointUnits) {
  Fill(0, 2);
  Fill(10, 2);
  memories_.fbuf.at(0).at(0) = kNull;
  for (uint32_t ialu = 1; ialu < 8; ++ialu) {
    for (uint32_t c0 = 1; c0 <= 4; c0 += 3) {
      PassFields pass = Copy();
      pass.rx2 = kSrc2;
      pass.x2_en = 1;
      pass.rimm = kImm;
      pass.c0 = c0;
      pass.ialu = ialu;
      Run({encoding::Pass(pass), encoding::Ret()}, {0, 10}, 100, 1, 2,
          0x00F0F00F);
      for (uint32_t w = 0; w < 2; ++w) {
        for (uint32_t e = 0; e < kFbufElements; ++e) {
          const uint32_t x = Before(w, e);
          const uint32_t operand = c0 == 1 ? Before(10 + w, e) : 0x00F0F00FU;
          const std::array<uint32_t, 8> results = {
              0,           x + operand, x - operand, x & operand,
              x | operand, x ^ operand, x >> 1,      operand - x};
          ASSERT_EQ(After(100 + w, e), results.at(ialu))
              << "ialu " << ialu << " c0 " << c0;
        }
      }
    }
  }
}

TEST_F(VecPassTest, MapsWordIndicesAndStrides) {
  struct Case {
    uint32_t wx;
    uint32_t wd;
    uint32_t x_s1;
    uint32_t dst_s1;
  };
  const auto map = [](uint32_t mapping, uint32_t w) -> uint32_t {
    const uint32_t h = (w >> 2) & 0xFU;
    const uint32_t gg = w & 3U;
    switch (mapping) {
      case 1:
        return (h * 8) + gg;
      case 2:
        return (h * 8) + 4 + gg;
      case 3:
        return 4 * w;
      default:
        return w;
    }
  };
  for (const Case c : {Case{.wx = 1, .wd = 2, .x_s1 = 0, .dst_s1 = 0},
                       Case{.wx = 2, .wd = 1, .x_s1 = 1, .dst_s1 = 0},
                       Case{.wx = 3, .wd = 0, .x_s1 = 0, .dst_s1 = 1},
                       Case{.wx = 0, .wd = 3, .x_s1 = 1, .dst_s1 = 1}}) {
    Fill(0, 200);
    PassFields pass = Copy();
    pass.rnw = kR11;
    pass.wx = c.wx;
    pass.wd = c.wd;
    pass.x_s1 = c.x_s1;
    pass.dst_s1 = c.dst_s1;
    Run({encoding::SaluImm(12, kR11, 0, 8), encoding::Pass(pass),
         encoding::Ret()},
        {0, 0}, 1000, 3, 40);
    std::vector<FbufWord> expected = before_;
    for (uint32_t q = 0; q < 3; ++q) {
      for (uint32_t w = 0; w < 8; ++w) {
        const uint32_t in = (q * (c.x_s1 != 0 ? 1 : 40)) + map(c.wx, w);
        const uint32_t out =
            1000 + (q * (c.dst_s1 != 0 ? 1 : 40)) + map(c.wd, w);
        expected.at(out) = before_.at(in);
      }
    }
    for (uint32_t w = 0; w < kFbufDepth; ++w) {
      ASSERT_EQ(memories_.fbuf.at(w), expected.at(w))
          << "wx " << c.wx << " wd " << c.wd << " word " << w;
    }
  }
}

TEST_F(VecPassTest, ReadsPbufRowVectorsAndTiles) {
  Fill(0, 8);
  for (uint32_t w = 0; w < 16; ++w) {
    for (uint32_t& element : memories_.pbuf.at(w)) {
      element = Random(120, 130);
    }
  }
  PassFields add = Copy();
  add.rpb = kPBase;
  add.pa = 1;
  add.c0 = 2;
  PassFields scale = Copy();
  scale.rpb = kPBase;
  scale.pa = 1;
  scale.b = 2;
  PassFields tile = scale;
  tile.pa = 2;
  PassFields tile_add = add;
  tile_add.pa = 2;
  for (const PassFields& pass : {add, scale, tile, tile_add}) {
    Run({encoding::Pass(pass), encoding::Ret()}, {0, 0}, 100, 2, 4, 0, 3);
    for (uint32_t q = 0; q < 2; ++q) {
      for (uint32_t g = 0; g < 4; ++g) {
        for (uint32_t e = 0; e < kFbufElements; ++e) {
          const uint32_t x = Before((q * 4) + g, e);
          const uint32_t column = (g * 8) + (e % 8);
          const uint32_t value =
              pass.pa == 2
                  ? memories_.pbuf.at(3 + (q * 4) + g).at(e)
                  : memories_.pbuf.at(3 + (column / 32)).at(column % 32);
          const uint32_t expected =
              pass.b == 2 ? reference::Mul(x, value) : reference::Add(x, value);
          ASSERT_EQ(After(100 + (q * 4) + g, e), expected)
              << "pa " << pass.pa << " b " << pass.b << " quad " << q
              << " word " << g;
        }
      }
    }
  }
}

TEST_F(VecPassTest, PostAddsAndTakesMaxima) {
  Fill(0, 2);
  Fill(10, 2);
  memories_.fbuf.at(0).at(0) = 0x80000000;
  memories_.fbuf.at(10).at(0) = 0x00000000;
  for (uint32_t c1 = 1; c1 < 3; ++c1) {
    for (uint32_t negate = 0; negate < 2; ++negate) {
      PassFields pass = Copy();
      pass.rx2 = kSrc2;
      pass.x2_en = 1;
      pass.c1 = c1;
      pass.neg_c1 = negate;
      Run({encoding::Pass(pass), encoding::Ret()}, {0, 10}, 100, 1, 2);
      for (uint32_t w = 0; w < 2; ++w) {
        for (uint32_t e = 0; e < kFbufElements; ++e) {
          ASSERT_EQ(
              After(100 + w, e),
              PostExpected(c1, negate != 0, Before(w, e), Before(10 + w, e)))
              << "c1 " << c1 << " negate " << negate;
        }
      }
    }
  }
}

TEST_F(VecPassTest, EvaluatesPowersOfTwoAndFractions) {
  for (uint32_t i = 0; i < 512; ++i) {
    const uint32_t v =
        reference::Round(exact::Real::Integer(512 + i).Scaled(-9));
    const uint32_t s = Random(116, 118) & 0x7FFFFFFFU;
    memories_.tbuf.at(i) = (uint64_t{s} << 32) | v;
  }
  Fill(0, 2);
  const std::array<uint32_t, 10> special = {
      0x43800000, 0xC3800000, 0x437FFFFF, 0xC37FFFFF, 0x42FE0000,
      0xC2FC0000, 0x80000000, 0x00000001, 0x3F7FFFFF, 0xBF800000};
  for (size_t i = 0; i < special.size(); ++i) {
    memories_.fbuf.at(0).at(i) = special.at(i);
  }
  for (uint32_t e = 0; e < kFbufElements; ++e) {
    memories_.fbuf.at(1).at(e) =
        reference::Round(exact::Real::Ratio(e * 37 % 1000, 1000));
  }
  PassFields pass = Copy();
  pass.nl = 2;
  Run({encoding::Pass(pass), encoding::Ret()}, {0, 0}, 100, 1, 2);
  for (uint32_t w = 0; w < 2; ++w) {
    for (uint32_t e = 0; e < kFbufElements; ++e) {
      ASSERT_EQ(After(100 + w, e),
                reference::Pow2(Before(w, e), memories_.tbuf))
          << "word " << w << " element " << e;
    }
  }
  pass.nl = 3;
  Run({encoding::Pass(pass), encoding::Ret()}, {1, 0}, 100, 1, 1);
  for (uint32_t e = 0; e < kFbufElements; ++e) {
    ASSERT_EQ(After(100, e),
              reference::Fraction01(Before(1, e), memories_.tbuf));
  }
}

TEST_F(VecPassTest, ReducesMaskedRowsIntoRowScalars) {
  constexpr uint32_t kWords = 11;
  constexpr uint32_t kValid = 77;
  Fill(0, kWords);
  memories_.fbuf.at(4).at(3) = 0x80000000;
  const uint32_t epsilon = Random(100, 110) & 0x7FFFFFFFU;
  std::array<std::array<uint32_t, kTokens>, 6> rs0{};
  std::array<std::array<uint32_t, kTokens>, 6> rs1{};
  for (uint32_t sc = 1; sc < 6; ++sc) {
    PassFields pass = Copy();
    pass.rimm = kR12;
    pass.rimm2 = kR11;
    pass.mask_en = 1;
    pass.red = (sc == 3 || sc == 5) ? 2 : 1;
    pass.sc = sc;
    if (sc == 2) {
      pass.b = 5;
    }
    Run({encoding::SaluImm(12, kR11, 0, kValid),
         encoding::SaluImm(12, kR12, 0, epsilon),
         encoding::SaluImm(12, kR13, 0, 70),
         encoding::Smem({.op = 2, .ra = kR13}), encoding::Pass(pass),
         encoding::Ret()},
        {0, 0}, 100, 1, kWords);
    rs0.at(sc) = engine_.Rows(0).rs0;
    rs1.at(sc) = engine_.Rows(0).rs1;
    CheckMasked(sc, kWords, kValid);
  }
  const uint32_t rn = reference::Reciprocal(reference::FromInt(70));
  for (uint32_t t = 0; t < kTokens; ++t) {
    const RowExpectation row = ExpectRow(t, kWords, kValid);
    EXPECT_EQ(rs0.at(1).at(t), reference::Negate(reference::Mul(row.sum, rn)));
    EXPECT_EQ(rs1.at(2).at(t), reference::ReciprocalSqrt(reference::Add(
                                   reference::Mul(row.squares, rn), epsilon)));
    EXPECT_EQ(rs0.at(3).at(t), reference::Negate(row.max));
    EXPECT_EQ(rs1.at(4).at(t), reference::Reciprocal(row.sum));
    EXPECT_EQ(rs1.at(5).at(t), row.max);
  }
}

TEST_F(VecPassTest, QuantizedWritesFollowTheRowStride) {
  Fill(0, 16);
  PassFields pass = Copy();
  pass.wr = 2;
  Run({encoding::Pass(pass), encoding::Ret()}, {0, 0}, 3000, 2, 8);
  for (uint32_t q = 0; q < 2; ++q) {
    for (uint32_t b = 0; b < 2; ++b) {
      for (uint32_t t = 0; t < kTokens; ++t) {
        std::array<uint32_t, 32> block{};
        for (uint32_t k = 0; k < 32; ++k) {
          block.at(k) = Before((q * 8) + (b * 4) + (k / 8), (t * 8) + (k % 8));
        }
        const reference::Mxint8Block expected = reference::EncodeMxint8(block);
        const AbufWord& word = memories_.abuf.at(3000 + (q * 2) + b);
        for (uint32_t k = 0; k < 32; ++k) {
          ASSERT_EQ(word.at((t * 32) + k), expected.q.at(k));
        }
        ASSERT_EQ(word.at(kAbufScaleOffset + t), expected.scale);
      }
    }
  }
}

TEST_F(VecPassTest, ScalarInstructions) {
  const std::vector<std::pair<uint32_t, uint32_t>> cases = {
      {0, 0x12345678U + 0x0000FFFFU},
      {1, 0x12345678U - 0x0000FFFFU},
      {2, 0x12345678U & 0x0000FFFFU},
      {3, 0x12345678U | 0x0000FFFFU},
      {4, 0x12345678U ^ 0x0000FFFFU},
      {5, 0x12345678U << 31},
      {6, 0x12345678U >> 31},
      {7, 0},
      {8, 0x12345678U * 0x0000FFFFU},
      {9, 0},
      {10, 0},
      {11, 0},
      {12, 0x0000FFFFU},
      {20, 0x0000FFFFU},
  };
  std::vector<Word128> program = {
      encoding::SaluImm(12, kR11, 0, 0x12345678),
      encoding::SaluImm(12, kR12, 0, 0x0000FFFF),
      encoding::SaluImm(0, kR13, 0, 0x8000001F),
  };
  for (size_t i = 0; i < cases.size(); ++i) {
    const uint32_t op = cases.at(i).first;
    const uint32_t b = op == 5 || op == 6 || op == 7 ? kR13 : kR12;
    program.push_back(
        encoding::Salu(op, static_cast<uint32_t>(15 + i), kR11, b));
  }
  program.push_back(encoding::SaluImm(7, 30, kR13, 4));
  program.push_back(encoding::SaluImm(9, 31, kR13, 0));
  program.push_back(encoding::SaluImm(0, 0, kR11, 1));
  program.push_back(encoding::Ret());
  Run(program, {0, 0}, 0, 1, 1);
  for (size_t i = 0; i < cases.size(); ++i) {
    EXPECT_EQ(engine_.Register(static_cast<uint32_t>(15 + i)),
              cases.at(i).second)
        << "op " << cases.at(i).first;
  }
  EXPECT_EQ(engine_.Register(30), 0xF8000001U);
  EXPECT_EQ(engine_.Register(31), 1U);
  EXPECT_EQ(engine_.Register(0), 0U);
  EXPECT_EQ(engine_.Register(kSrc), 0U);
  EXPECT_EQ(engine_.Register(10), 0U);
}

TEST_F(VecPassTest, ScalarMemoryInstructions) {
  Fill(0, 4);
  memories_.fbuf.at(1).at(14) = 0x00000003;
  memories_.fbuf.at(1).at(22) = 0x80000000;
  PassFields negmax = Copy();
  negmax.wr = 0;
  negmax.red = 2;
  negmax.sc = 3;
  PassFields row_max = negmax;
  row_max.sc = 5;
  Run({encoding::SaluImm(12, kR11, 0, 4095), encoding::SaluImm(12, kR12, 0, 9),
       encoding::Smem({.op = 0, .rd = 20, .ra = kR11, .rb = kR12, .imm = 3}),
       encoding::Smem(
           {.op = 0, .rd = 21, .ra = kR11, .col_imm = 1, .col = 31, .imm = 1}),
       encoding::Pass(negmax),
       encoding::Smem({.op = 1, .rd = 22, .col_imm = 1, .col = 3, .which = 1}),
       encoding::Pass(row_max),
       encoding::Smem({.op = 1, .rd = 23, .col_imm = 1, .col = 7}),
       encoding::Smem({.op = 2, .ra = 0}),
       encoding::Smem({.op = 1, .rd = 24, .col_imm = 1, .col = 0}),
       encoding::Smem({.op = 3, .rd = 25, .ra = kR11}), encoding::Ret()},
      {0, 0}, 0, 1, 1);
  EXPECT_EQ(engine_.Register(20), Before(2, 9));
  EXPECT_EQ(engine_.Register(21), Before(0, 31));
  for (uint32_t t = 0; t < kTokens; ++t) {
    std::array<uint32_t, kLanes> lanes{};
    for (uint32_t l = 0; l < kLanes; ++l) {
      lanes.at(l) = Before(0, (t * kLanes) + l);
    }
    const uint32_t largest = reference::RowMax(lanes);
    EXPECT_EQ(engine_.Rows(0).rs0.at(t), largest ^ 0x80000000U);
    EXPECT_EQ(engine_.Rows(0).rs1.at(t), largest);
  }
  EXPECT_EQ(engine_.Register(22), engine_.Rows(0).rs0.at(3));
  EXPECT_EQ(engine_.Register(23), engine_.Rows(0).rs1.at(3));
  EXPECT_EQ(engine_.Register(24), engine_.Rows(0).rs1.at(0));
  EXPECT_EQ(engine_.Rn(), reference::Reciprocal(0x3F800000));
  EXPECT_EQ(engine_.Register(25), 0U);
}

TEST_F(VecPassTest, BranchesLoopAndSkip) {
  Fill(0, 1);
  PassFields pass = Copy();
  pass.rq = kR12;
  pass.rnw = kR12;
  pass.rn8 = kR12;
  pass.rx = kSrc;
  pass.rd = kDst;
  Run({encoding::SaluImm(12, kR11, 0, 5), encoding::SaluImm(12, kR12, 0, 1),
       encoding::SaluImm(0, kR13, kR13, 7), encoding::SaluImm(1, kR11, kR11, 1),
       encoding::Branch(2, kR11, kProgram + 2),
       encoding::Branch(1, kR11, kProgram + 7),
       encoding::SaluImm(12, kR13, 0, 0), encoding::Branch(3, 0, kProgram + 6),
       Word128{.low = 0, .high = uint64_t{9} << 60}, encoding::Pass(pass),
       encoding::Branch(0, 0, kProgram + 12), encoding::SaluImm(12, kR13, 0, 0),
       encoding::Ret()},
      {0, 0}, 100, 1, 1);
  EXPECT_EQ(engine_.Register(kR13), 35U);
  EXPECT_EQ(memories_.fbuf.at(100), before_.at(0));
}

TEST_F(VecPassTest, RegistersHoldTheCommandFields) {
  Run({encoding::Ret()}, {0x1234, 0x0567}, 0x1ABC, 0x1F3, 0xEE, 0xCAFEF00D,
      0x3FF);
  EXPECT_EQ(engine_.Register(1), 0x234U);
  EXPECT_EQ(engine_.Register(2), 0x567U);
  EXPECT_EQ(engine_.Register(3), 0x1ABCU);
  EXPECT_EQ(engine_.Register(4), 0x1F3U);
  EXPECT_EQ(engine_.Register(5), 0xEEU);
  EXPECT_EQ(engine_.Register(6), 0x3FFU);
  EXPECT_EQ(engine_.Register(7), 0U);
  EXPECT_EQ(engine_.Register(8), 0xCAFEF00DU);
  EXPECT_EQ(engine_.Register(9), 0xABCU);
  EXPECT_EQ(engine_.Register(10), 0U);
  engine_.Execute({.op = 5,
                   .quantize = true,
                   .src = 1,
                   .src2 = 2,
                   .dst = 3,
                   .rows_q = 1,
                   .n8 = 4,
                   .p_base = 0,
                   .aux = 0x7AB,
                   .imm = 0x00000ABC},
                  memories_);
  EXPECT_EQ(engine_.Register(7), 0x7ABU);
  EXPECT_EQ(engine_.Register(9), 0xABCU);
  EXPECT_EQ(engine_.Register(10), 1U);
  memories_.cbuf.at(kProgram) = encoding::Ret();
  engine_.Execute({.op = 21,
                   .quantize = false,
                   .src = 1,
                   .src2 = 2,
                   .dst = 3,
                   .rows_q = 1,
                   .n8 = 4,
                   .p_base = 0,
                   .aux = kProgram,
                   .imm = 0},
                  memories_);
  EXPECT_EQ(engine_.Register(7), 0U);
  EXPECT_EQ(engine_.Register(9), 3U);
}

TEST_F(VecPassTest, RejectsProgramsWithUndefinedResults) {
  PassFields shifted = Copy();
  shifted.rd = kSrc2;
  EXPECT_DEATH(Run({encoding::SaluImm(0, kSrc2, kSrc, 1),
                    encoding::Pass(shifted), encoding::Ret()},
                   {10, 0}, 0, 1, 4),
               "after writing it");
  PassFields empty = Copy();
  empty.rq = 0;
  EXPECT_DEATH(
      Run({encoding::Pass(empty), encoding::Ret()}, {10, 50}, 30, 1, 4),
      "0 quads");
}

TEST_F(VecPassTest, RowMaskSkipsTheRowsAMarkingPassFoundZero) {
  constexpr uint32_t kQuads = 8;
  constexpr uint32_t kWords = 2;
  Fill(0, kQuads * kWords);
  Fill(100, kQuads * kWords);
  for (const size_t q : {0, 1, 3, 4, 5, 6, 7}) {
    for (size_t w = 0; w < kWords; ++w) {
      memories_.fbuf.at((q * kWords) + w).fill(0);
    }
  }
  memories_.fbuf.at(size_t{2} * kWords).fill(0);
  PassFields mark = Copy();
  mark.wr = 0;
  mark.ialu = 4;
  mark.rm = 2;
  PassFields skip = Copy();
  skip.ialu = 4;
  skip.rm = 1;
  Run({encoding::Pass(mark), encoding::Pass(skip), encoding::Ret()}, {0, 0},
      100, kQuads, kWords);
  for (uint32_t q = 0; q < kQuads; ++q) {
    for (uint32_t w = 0; w < kWords; ++w) {
      const uint32_t word = (q * kWords) + w;
      for (uint32_t e = 0; e < kFbufElements; ++e) {
        ASSERT_EQ(After(100 + word, e),
                  q == 2 ? Before(word, e) : Before(100 + word, e))
            << "row " << q << " word " << w << " element " << e;
      }
    }
  }
  EXPECT_EQ(engine_.Activity().skipped_rows, kQuads - 1);
  EXPECT_EQ(engine_.Activity().skip_cycles, 1U);
  EXPECT_EQ(engine_.Activity().words, (kQuads * kWords) + kWords);
  EXPECT_EQ(engine_.Activity().drains, 2U);
  EXPECT_EQ(engine_.Activity().overlaps, 0U);
}

TEST_F(VecPassTest, SkipRunsAfterAnIssuedWordCostHalfTheirQuadsAndTheReads) {
  for (const uint32_t x2 : {0U, 1U}) {
    for (const uint32_t wr : {0U, 1U}) {
      for (const uint32_t k : {1U, 2U, 3U, 6U, 10U, 21U}) {
        const uint32_t quads = k + 2;
        Fill(0, quads);
        Fill(50, quads);
        for (uint32_t q = 1; q <= k; ++q) {
          memories_.fbuf.at(q).fill(0);
        }
        PassFields mark = Copy();
        mark.wr = 0;
        mark.ialu = 4;
        mark.rm = 2;
        PassFields skip = Copy();
        skip.rx2 = kSrc2;
        skip.x2_en = x2;
        skip.c0 = x2;
        skip.wr = wr;
        skip.rm = 1;
        Run({encoding::Pass(mark), encoding::Pass(skip), encoding::Ret()},
            {0, 50}, 100, quads, 1);
        const uint32_t reads = 1 + x2;
        EXPECT_EQ(engine_.Activity().skipped_rows, k);
        EXPECT_EQ(engine_.Activity().skip_cycles,
                  std::max((k / 2) + reads, kWordCycles) - kWordCycles)
            << "x2 " << x2 << " wr " << wr << " run " << k;
      }
    }
  }
}

TEST_F(VecPassTest, TrailingSkipRunsAreHiddenOnlyByADrain) {
  constexpr auto kDrainHidden =
      static_cast<uint32_t>(kOverlapLatencyWords * kWordCycles);
  for (const uint32_t run : {5U, kDrainHidden + 7, (2 * kDrainHidden) + 15}) {
    for (const bool overlap : {false, true}) {
      const uint32_t quads = run + 1;
      Fill(0, quads);
      for (uint32_t q = 1; q < quads; ++q) {
        memories_.fbuf.at(q).fill(0);
      }
      PassFields mark = Copy();
      mark.wr = 0;
      mark.ialu = 4;
      mark.rm = 2;
      PassFields skip = Copy();
      skip.rm = 1;
      std::vector<Word128> program = {encoding::Pass(mark),
                                      encoding::Pass(skip)};
      if (overlap) {
        PassFields second = skip;
        second.rd = kSrc2;
        program.push_back(encoding::Pass(second));
      }
      program.push_back(encoding::Ret());
      Run(program, {0, 300}, 100, quads, 1);
      const uint32_t tail = (run - 1) / 2;
      const uint32_t drained = std::max(tail, kDrainHidden) - kDrainHidden;
      EXPECT_EQ(engine_.Activity().skip_cycles,
                overlap ? tail + drained : drained)
          << "run " << run << " overlap " << overlap;
    }
  }
}

TEST_F(VecPassTest, SkipRunsAtTheStartOfAPassCostHalfTheirQuads) {
  for (const uint32_t run : {1U, 2U, 5U, 8U, 101U}) {
    for (const bool overlap : {false, true}) {
      RunMarkAndSkip(run + 1, run, overlap);
      const uint32_t start = (run + 1) / 2;
      EXPECT_EQ(engine_.Activity().skipped_rows,
                overlap ? 2 * uint64_t{run} : run);
      EXPECT_EQ(engine_.Activity().skip_cycles,
                overlap ? (2 * start) - 1 : start)
          << "run " << run << " overlap " << overlap;
    }
  }
}

TEST_F(VecPassTest,
       PassesWithEveryQuadSkippedCostTheStartRunUnlessADrainHidesIt) {
  constexpr auto kDrainHidden =
      static_cast<uint32_t>(kOverlapLatencyWords * kWordCycles);
  for (const uint32_t run : {1U, 2U, 5U, 100U, 101U}) {
    for (const bool overlap : {false, true}) {
      RunMarkAndSkip(run, run, overlap);
      const uint32_t start = (run + 1) / 2;
      const uint32_t last = overlap ? start - 1 : start;
      EXPECT_EQ(engine_.Activity().skipped_rows,
                overlap ? 2 * uint64_t{run} : run);
      EXPECT_EQ(
          engine_.Activity().skip_cycles,
          (overlap ? start : 0) + (std::max(last, kDrainHidden) - kDrainHidden))
          << "run " << run << " overlap " << overlap;
    }
  }
}

TEST_F(VecPassTest, OverlapsDependentPassesAndCountsTheStalledWords) {
  Fill(0, 16);
  PassFields first = Copy();
  PassFields second = Copy();
  second.rx = kDst;
  second.rd = kSrc2;
  Run({encoding::Pass(first), encoding::Pass(second), encoding::Ret()}, {0, 50},
      100, 1, 4);
  EXPECT_EQ(engine_.Activity().passes, 2U);
  EXPECT_EQ(engine_.Activity().drains, 1U);
  EXPECT_EQ(engine_.Activity().overlaps, 1U);
  EXPECT_EQ(engine_.Activity().overlap_stall_words, kOverlapLatencyWords - 4);
  for (uint32_t w = 0; w < 4; ++w) {
    for (uint32_t e = 0; e < kFbufElements; ++e) {
      ASSERT_EQ(After(50 + w, e), reference::Add(Before(w, e), 0));
    }
  }
  Run({encoding::Pass(first), encoding::Pass(second), encoding::Ret()}, {0, 50},
      100, 1, 16);
  EXPECT_EQ(engine_.Activity().overlaps, 1U);
  EXPECT_EQ(engine_.Activity().overlap_stall_words, 0U);
  Run({encoding::Pass(first), encoding::Pass(first), encoding::Ret()}, {0, 50},
      100, 1, 4);
  EXPECT_EQ(engine_.Activity().overlaps, 1U);
  EXPECT_EQ(engine_.Activity().overlap_stall_words, 0U);
  Run({encoding::Pass(first), encoding::SaluImm(0, kR11, 0, 1),
       encoding::Pass(second), encoding::Ret()},
      {0, 50}, 100, 1, 4);
  EXPECT_EQ(engine_.Activity().drains, 2U);
  EXPECT_EQ(engine_.Activity().overlaps, 0U);
  PassFields reduce = Copy();
  reduce.red = 1;
  reduce.sc = 4;
  PassFields scaled = second;
  scaled.b = 3;
  Run({encoding::Pass(reduce), encoding::Pass(scaled), encoding::Ret()},
      {0, 50}, 100, 1, 4);
  EXPECT_EQ(engine_.Activity().drains, 1U);
  EXPECT_EQ(engine_.Activity().overlaps, 1U);
  EXPECT_EQ(engine_.Activity().context_stall_words,
            kReductionLatencyWords +
                ((ScalarProgramCycles(kScalarPrograms.at(4)) + 3) / 4));
  scaled.ctx = 1;
  Run({encoding::Pass(reduce), encoding::Pass(scaled), encoding::Ret()},
      {0, 50}, 100, 1, 4);
  EXPECT_EQ(engine_.Activity().context_stall_words, 0U);
  PassFields quantized = Copy();
  quantized.wr = 2;
  Run({encoding::Pass(quantized), encoding::Pass(first), encoding::Ret()},
      {0, 50}, 100, 1, 4);
  EXPECT_EQ(engine_.Activity().drains, 1U);
  EXPECT_EQ(engine_.Activity().drains_quantized, 0U);
  EXPECT_EQ(engine_.Activity().overlaps, 1U);
  PassFields multiplier = Copy();
  multiplier.pa = 1;
  multiplier.b = 2;
  PassFields pre_adder = Copy();
  pre_adder.rx = kDst;
  pre_adder.rd = kSrc2;
  pre_adder.pa = 1;
  pre_adder.c0 = 2;
  for (const bool multiplier_first : {true, false}) {
    Run(multiplier_first
            ? std::vector{encoding::Pass(multiplier), encoding::Pass(pre_adder),
                          encoding::Ret()}
            : std::vector{encoding::Pass(pre_adder), encoding::Pass(multiplier),
                          encoding::Ret()},
        {multiplier_first ? 0U : 100U, 50}, 100, 1, 4);
    EXPECT_EQ(engine_.Activity().drains, 1U);
    EXPECT_EQ(engine_.Activity().overlaps, 1U);
  }
}

}  // namespace
}  // namespace hotarubi::iss
