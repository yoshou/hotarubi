#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <map>
#include <random>
#include <span>
#include <vector>

#include "fp32_ref.hpp"
#include "npu_bench.hpp"
#include "npu_data.hpp"
#include "npu_isa.hpp"
#include "vec_ref.hpp"

namespace hotarubi::tb {
namespace {

constexpr uint32_t kFbufArea = Ddr::kBase + 0x100000U;
constexpr uint32_t kPbufArea = Ddr::kBase + 0x300000U;
constexpr uint32_t kTbufArea = Ddr::kBase + 0x400000U;
constexpr uint32_t kOutArea = Ddr::kBase + 0x500000U;

constexpr uint32_t kOpAdd = 0;
constexpr uint32_t kOpMul = 1;
constexpr uint32_t kOpAddP = 2;
constexpr uint32_t kOpMulP = 3;
constexpr uint32_t kOpScale = 4;
constexpr uint32_t kOpCopy = 5;
constexpr uint32_t kOpLayerNorm = 6;
constexpr uint32_t kOpRmsNorm = 7;
constexpr uint32_t kOpGelu = 8;
constexpr uint32_t kOpSilu = 9;
constexpr uint32_t kOpSiluMul = 10;
constexpr uint32_t kOpSoftmax = 11;
constexpr uint32_t kOpRope = 12;
constexpr uint32_t kOpQuant = 13;
constexpr uint32_t kOpRelu = 14;
constexpr uint32_t kOpMax = 15;

using Rows = std::vector<std::vector<uint32_t>>;

Rows RandomRows(std::mt19937& rng, uint32_t tokens, uint32_t columns,
                int min_exponent, int max_exponent) {
  Rows rows(tokens, std::vector<uint32_t>(columns));
  for (std::vector<uint32_t>& row : rows) {
    for (uint32_t& value : row) {
      value = RandomFp32(rng, min_exponent, max_exponent);
    }
  }
  return rows;
}

std::vector<FbufWord> ToWords(const Rows& rows) {
  const auto n8 = static_cast<uint32_t>(rows[0].size() / 8);
  std::vector<FbufWord> words((rows.size() / 4) * n8);
  for (uint32_t t = 0; t < rows.size(); ++t) {
    for (uint32_t ch = 0; ch < rows[t].size(); ++ch) {
      words.at(((t / 4) * n8) + (ch / 8)).at(((t % 4) * 8) + (ch % 8)) =
          rows[t][ch];
    }
  }
  return words;
}

Rows FromWords(const std::vector<FbufWord>& words, uint32_t n8) {
  Rows rows((words.size() / n8) * 4, std::vector<uint32_t>(size_t{8} * n8));
  for (uint32_t t = 0; t < rows.size(); ++t) {
    for (uint32_t ch = 0; ch < 8 * n8; ++ch) {
      rows[t][ch] =
          words.at(((t / 4) * n8) + (ch / 8)).at(((t % 4) * 8) + (ch % 8));
    }
  }
  return rows;
}

std::vector<FbufWord> RowVectorWords(std::span<const uint32_t> vector) {
  std::vector<FbufWord> words((vector.size() + 31) / 32);
  for (uint32_t ch = 0; ch < vector.size(); ++ch) {
    words.at(ch / 32).at(ch % 32) = vector[ch];
  }
  return words;
}

std::vector<AbufWord> Quantize(const Rows& rows) {
  const auto blocks = static_cast<uint32_t>(rows[0].size() / 32);
  std::vector<AbufWord> words((rows.size() / 4) * blocks);
  for (uint32_t t = 0; t < rows.size(); ++t) {
    for (uint32_t b = 0; b < blocks; ++b) {
      const MxBlock block = QuantizeMxint8(
          std::span(rows[t]).subspan(size_t{32} * b).first<32>());
      AbufWord& word = words.at(((t / 4) * blocks) + b);
      for (uint32_t k = 0; k < 32; ++k) {
        word.q.at(((t % 4) * 32) + k) = block.q.at(k);
      }
      word.scale.at(t % 4) = block.scale;
    }
  }
  return words;
}

struct VecRun {
  VecArgs args;
  std::map<uint32_t, std::vector<FbufWord>> fbuf;
  std::vector<FbufWord> pbuf;
  const Tables* tables = nullptr;
};

class VecBench {
 public:
  explicit VecBench(uint32_t seed) : bench_(seed) {}

  void Run(const VecRun& run) {
    Ddr& ddr = bench_.Memory();
    std::vector<Instruction> program;
    uint32_t area = kFbufArea;
    for (const auto& [address, words] : run.fbuf) {
      for (uint32_t i = 0; i < words.size(); ++i) {
        WriteFbuf(ddr, area + (i * kFbufWordBytes), words[i]);
      }
      program.push_back(
          DmaLoad({.ddr_addr = area,
                   .mem = Mem::kFbuf,
                   .onchip = address,
                   .rows = 1,
                   .row_words = static_cast<uint32_t>(words.size())}));
      area += 0x10000U;
    }
    if (!run.pbuf.empty()) {
      for (uint32_t i = 0; i < run.pbuf.size(); ++i) {
        WriteFbuf(ddr, kPbufArea + (i * kFbufWordBytes), run.pbuf[i]);
      }
      program.push_back(
          DmaLoad({.ddr_addr = kPbufArea,
                   .mem = Mem::kPbuf,
                   .onchip = run.args.p_base,
                   .rows = 1,
                   .row_words = static_cast<uint32_t>(run.pbuf.size())}));
    }
    if (run.tables != nullptr) {
      for (uint32_t i = 0; i < 1024; ++i) {
        WriteTbuf(ddr, kTbufArea + (i * kTbufWordBytes), run.tables->v.at(i),
                  run.tables->s.at(i));
      }
      program.push_back(DmaLoad({.ddr_addr = kTbufArea,
                                 .mem = Mem::kTbuf,
                                 .onchip = 0,
                                 .rows = 1,
                                 .row_words = 1024}));
    }
    VecArgs args = run.args;
    args.wait = kWaitDma;
    program.push_back(Vec(args));
    const uint32_t words = args.rows_q * args.n8;
    program.push_back(DmaStore({.ddr_addr = kOutArea,
                                .mem = args.q ? Mem::kAbuf : Mem::kFbuf,
                                .onchip = args.dst,
                                .rows = 1,
                                .row_words = args.q ? words / 4 : words,
                                .wait = kWaitVec}));
    program.push_back(Halt());
    ASSERT_TRUE(bench_.Run(program));
    EXPECT_EQ(bench_.ReadReg(0x04) & 0x6U, 0x2U);
    EXPECT_TRUE(bench_.Errors().empty());
  }

  Rows FbufResult(uint32_t rows_q, uint32_t n8) {
    std::vector<FbufWord> words;
    words.reserve(size_t{rows_q} * n8);
    for (uint32_t i = 0; i < rows_q * n8; ++i) {
      words.push_back(
          ReadFbuf(bench_.Memory(), kOutArea + (i * kFbufWordBytes)));
    }
    return FromWords(words, n8);
  }

  std::vector<AbufWord> AbufResult(uint32_t count) {
    std::vector<AbufWord> words;
    words.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
      words.push_back(
          ReadAbuf(bench_.Memory(), kOutArea + (i * kAbufWordBytes)));
    }
    return words;
  }

 private:
  NpuBench bench_;
};

void ExpectRows(const Rows& actual, const Rows& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (uint32_t t = 0; t < expected.size(); ++t) {
    for (uint32_t ch = 0; ch < expected[t].size(); ++ch) {
      ASSERT_EQ(actual[t][ch], expected[t][ch])
          << "token " << t << " column " << ch;
    }
  }
}

void ExpectAbuf(const std::vector<AbufWord>& actual,
                const std::vector<AbufWord>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (uint32_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(actual[i], expected[i]) << "word " << i;
  }
}

Rows Map(const Rows& x,
         const std::function<uint32_t(uint32_t, uint32_t, uint32_t)>& f) {
  Rows y = x;
  for (uint32_t t = 0; t < x.size(); ++t) {
    for (uint32_t ch = 0; ch < x[t].size(); ++ch) {
      y[t][ch] = f(t, ch, x[t][ch]);
    }
  }
  return y;
}

void CheckOutput(VecBench& bench, const VecArgs& args, const Rows& expected) {
  if (args.q) {
    ExpectAbuf(bench.AbufResult(args.rows_q * args.n8 / 4), Quantize(expected));
  } else {
    ExpectRows(bench.FbufResult(args.rows_q, args.n8), expected);
  }
}

struct BinaryCase {
  uint32_t op;
  bool q;
};

class VecBinaryTest : public testing::TestWithParam<BinaryCase> {};

TEST_P(VecBinaryTest, MatchesElementwiseReference) {
  const BinaryCase param = GetParam();
  std::mt19937 rng(100 + param.op + (param.q ? 50 : 0));
  constexpr uint32_t kRowsQ = 2;
  constexpr uint32_t kN8 = 8;
  const Rows x = RandomRows(rng, 4 * kRowsQ, 8 * kN8, -6, 6);
  const Rows x2 = RandomRows(rng, 4 * kRowsQ, 8 * kN8, -6, 6);
  std::vector<uint32_t> p(size_t{8} * kN8);
  for (uint32_t& value : p) {
    value = RandomFp32(rng, -3, 3);
  }
  const uint32_t imm = RandomFp32(rng, -2, 2);
  const VecArgs args{.op = param.op,
                     .src = 16,
                     .src2 = 300,
                     .dst = param.q ? 40U : 600U,
                     .rows_q = kRowsQ,
                     .n8 = kN8,
                     .p_base = 7,
                     .aux = 0,
                     .imm = imm,
                     .q = param.q};
  VecBench bench(200 + param.op);
  bench.Run({.args = args,
             .fbuf = {{16, ToWords(x)}, {300, ToWords(x2)}},
             .pbuf = RowVectorWords(p),
             .tables = nullptr});
  const Rows expected = Map(x, [&](uint32_t t, uint32_t ch, uint32_t a) {
    const uint32_t b = x2[t][ch];
    switch (param.op) {
      case kOpAdd:
        return Fp32Add(a, b);
      case kOpMul:
        return Fp32Mul(a, b);
      case kOpAddP:
        return Fp32Add(a, p[ch]);
      case kOpMulP:
        return Fp32Mul(a, p[ch]);
      case kOpScale:
        return Fp32Mul(a, imm);
      case kOpRelu:
        return Fp32Less(kFp32Zero, a) ? a : kFp32Zero;
      case kOpMax:
        return Fp32Max(a, b);
      default:
        return Fp32Add(a, kFp32Zero);
    }
  });
  CheckOutput(bench, args, expected);
}

INSTANTIATE_TEST_SUITE_P(
    Ops, VecBinaryTest,
    testing::Values(BinaryCase{kOpAdd, false}, BinaryCase{kOpMul, false},
                    BinaryCase{kOpAddP, false}, BinaryCase{kOpMulP, false},
                    BinaryCase{kOpScale, false}, BinaryCase{kOpCopy, false},
                    BinaryCase{kOpRelu, false}, BinaryCase{kOpMax, false},
                    BinaryCase{kOpAdd, true}, BinaryCase{kOpMulP, true},
                    BinaryCase{kOpScale, true}, BinaryCase{kOpCopy, true},
                    BinaryCase{kOpQuant, true}, BinaryCase{kOpMax, true}));

uint32_t ElementCount(const VecArgs& args) {
  return args.aux == 0 ? 8 * args.n8 : args.aux;
}

Rows LayerNorm(const Rows& x, const VecArgs& args,
               std::span<const uint32_t> gamma,
               std::span<const uint32_t> beta) {
  const uint32_t rn = Fp32Recip(Fp32FromInt(ElementCount(args), 0));
  Rows y = x;
  for (uint32_t t = 0; t < x.size(); ++t) {
    const uint32_t negative_mean = Fp32Neg(Fp32Mul(RowSum(x[t]), rn));
    std::vector<uint32_t> centered(x[t].size());
    std::vector<uint32_t> squares(x[t].size());
    for (uint32_t ch = 0; ch < x[t].size(); ++ch) {
      centered[ch] = Fp32Add(x[t][ch], negative_mean);
      squares[ch] = Fp32Mul(centered[ch], centered[ch]);
    }
    const uint32_t rstd =
        Fp32Rsqrt(Fp32Add(Fp32Mul(RowSum(squares), rn), args.imm));
    for (uint32_t ch = 0; ch < x[t].size(); ++ch) {
      y[t][ch] =
          Fp32Add(Fp32Mul(Fp32Mul(centered[ch], rstd), gamma[ch]), beta[ch]);
    }
  }
  return y;
}

Rows RmsNorm(const Rows& x, const VecArgs& args,
             std::span<const uint32_t> gamma) {
  const uint32_t rn = Fp32Recip(Fp32FromInt(ElementCount(args), 0));
  Rows y = x;
  for (uint32_t t = 0; t < x.size(); ++t) {
    std::vector<uint32_t> squares(x[t].size());
    for (uint32_t ch = 0; ch < x[t].size(); ++ch) {
      squares[ch] = Fp32Mul(x[t][ch], x[t][ch]);
    }
    const uint32_t rstd =
        Fp32Rsqrt(Fp32Add(Fp32Mul(RowSum(squares), rn), args.imm));
    for (uint32_t ch = 0; ch < x[t].size(); ++ch) {
      y[t][ch] = Fp32Mul(Fp32Mul(x[t][ch], rstd), gamma[ch]);
    }
  }
  return y;
}

class VecNormTest : public testing::TestWithParam<bool> {};

TEST_P(VecNormTest, LayerNorm) {
  const bool q = GetParam();
  std::mt19937 rng(q ? 301 : 302);
  constexpr uint32_t kRowsQ = 2;
  constexpr uint32_t kN8 = 8;
  const Rows x = RandomRows(rng, 4 * kRowsQ, 8 * kN8, -3, 3);
  std::vector<uint32_t> gamma(size_t{8} * kN8);
  std::vector<uint32_t> beta(size_t{8} * kN8);
  for (uint32_t ch = 0; ch < 8 * kN8; ++ch) {
    gamma[ch] = RandomFp32(rng, -1, 1);
    beta[ch] = RandomFp32(rng, -2, 0);
  }
  std::vector<uint32_t> params = gamma;
  params.insert(params.end(), beta.begin(), beta.end());
  const VecArgs args{.op = kOpLayerNorm,
                     .src = 0,
                     .src2 = 500,
                     .dst = q ? 3U : 900U,
                     .rows_q = kRowsQ,
                     .n8 = kN8,
                     .p_base = 11,
                     .aux = 0,
                     .imm = 0x3727C5ACU,
                     .q = q};
  VecBench bench(q ? 303 : 304);
  bench.Run({.args = args,
             .fbuf = {{0, ToWords(x)}},
             .pbuf = RowVectorWords(params),
             .tables = nullptr});
  CheckOutput(bench, args, LayerNorm(x, args, gamma, beta));
}

TEST_P(VecNormTest, RmsNorm) {
  const bool q = GetParam();
  std::mt19937 rng(q ? 311 : 312);
  constexpr uint32_t kRowsQ = 3;
  constexpr uint32_t kN8 = 12;
  const Rows x = RandomRows(rng, 4 * kRowsQ, 8 * kN8, -3, 3);
  std::vector<uint32_t> gamma(size_t{8} * kN8);
  for (uint32_t& value : gamma) {
    value = RandomFp32(rng, -1, 1);
  }
  const VecArgs args{.op = kOpRmsNorm,
                     .src = 20,
                     .src2 = 700,
                     .dst = q ? 100U : 1000U,
                     .rows_q = kRowsQ,
                     .n8 = kN8,
                     .p_base = 0,
                     .aux = 0,
                     .imm = 0x358637BDU,
                     .q = q};
  VecBench bench(q ? 313 : 314);
  bench.Run({.args = args,
             .fbuf = {{20, ToWords(x)}},
             .pbuf = RowVectorWords(gamma),
             .tables = nullptr});
  CheckOutput(bench, args, RmsNorm(x, args, gamma));
}

INSTANTIATE_TEST_SUITE_P(Quantized, VecNormTest, testing::Bool());

uint32_t SignFix(uint32_t x, uint32_t a) {
  const uint32_t product = Fp32Mul(x, a);
  return (x & kFp32SignBit) != 0 ? Fp32Add(Fp32Neg(product), x) : product;
}

struct ActCase {
  uint32_t op;
  bool q;
};

class VecActTest : public testing::TestWithParam<ActCase> {};

TEST_P(VecActTest, MatchesTableReference) {
  const ActCase param = GetParam();
  std::mt19937 rng(400 + param.op + (param.q ? 50 : 0));
  constexpr uint32_t kRowsQ = 2;
  constexpr uint32_t kN8 = 8;
  const Tables tables = MakeTables(param.op == kOpGelu ? ActFunction::kGeluPhi
                                                       : ActFunction::kSigmoid);
  Rows x = RandomRows(rng, 4 * kRowsQ, 8 * kN8, -8, 4);
  x[0][0] = 0x41800000U;
  x[0][1] = 0xC1A00000U;
  x[0][2] = 0x41700000U;
  const Rows up = RandomRows(rng, 4 * kRowsQ, 8 * kN8, -3, 3);
  const VecArgs args{.op = param.op,
                     .src = 64,
                     .src2 = 128,
                     .dst = param.q ? 0U : 256U,
                     .rows_q = kRowsQ,
                     .n8 = kN8,
                     .p_base = 0,
                     .aux = 0,
                     .imm = 2000,
                     .q = param.q};
  VecBench bench(500 + param.op);
  bench.Run({.args = args,
             .fbuf = {{64, ToWords(x)}, {128, ToWords(up)}},
             .pbuf = {},
             .tables = &tables});
  const Rows expected = Map(x, [&](uint32_t t, uint32_t ch, uint32_t value) {
    const uint32_t y = SignFix(value, TableAct(tables, value));
    return param.op == kOpSiluMul ? Fp32Mul(y, up[t][ch]) : y;
  });
  CheckOutput(bench, args, expected);
}

INSTANTIATE_TEST_SUITE_P(Ops, VecActTest,
                         testing::Values(ActCase{kOpGelu, false},
                                         ActCase{kOpSilu, false},
                                         ActCase{kOpSiluMul, false},
                                         ActCase{kOpSilu, true},
                                         ActCase{kOpSiluMul, true}));

Rows Softmax(const Rows& x, const Tables& tables, uint32_t valid) {
  Rows y = x;
  for (uint32_t t = 0; t < x.size(); ++t) {
    const uint32_t max = RowMax(std::span(x[t]).first(valid));
    std::vector<uint32_t> e(x[t].size(), kFp32Zero);
    for (uint32_t ch = 0; ch < valid; ++ch) {
      e[ch] = TableExp2(tables,
                        Fp32Mul(Fp32Add(x[t][ch], Fp32Neg(max)), kFp32Log2e));
    }
    const uint32_t recip = Fp32Recip(RowSum(e));
    for (uint32_t ch = 0; ch < x[t].size(); ++ch) {
      y[t][ch] = ch < valid ? Fp32Mul(e[ch], recip) : kFp32Zero;
    }
  }
  return y;
}

struct SoftmaxCase {
  uint32_t valid;
  bool q;
};

class VecSoftmaxTest : public testing::TestWithParam<SoftmaxCase> {};

TEST_P(VecSoftmaxTest, MatchesReference) {
  const SoftmaxCase param = GetParam();
  std::mt19937 rng(600 + param.valid);
  constexpr uint32_t kRowsQ = 2;
  constexpr uint32_t kN8 = 8;
  const Tables tables = MakeTables(ActFunction::kSigmoid);
  const Rows x = RandomRows(rng, 4 * kRowsQ, 8 * kN8, -4, 3);
  const VecArgs args{.op = kOpSoftmax,
                     .src = 1,
                     .src2 = 0,
                     .dst = param.q ? 9U : 2000U,
                     .rows_q = kRowsQ,
                     .n8 = kN8,
                     .p_base = 0,
                     .aux = param.valid,
                     .imm = 3000,
                     .q = param.q};
  VecBench bench(700 + param.valid);
  bench.Run(
      {.args = args, .fbuf = {{1, ToWords(x)}}, .pbuf = {}, .tables = &tables});
  CheckOutput(bench, args, Softmax(x, tables, param.valid));
}

INSTANTIATE_TEST_SUITE_P(Columns, VecSoftmaxTest,
                         testing::Values(SoftmaxCase{64, false},
                                         SoftmaxCase{37, false},
                                         SoftmaxCase{1, false},
                                         SoftmaxCase{50, true}));

class VecRopeTest : public testing::TestWithParam<bool> {};

TEST_P(VecRopeTest, RotatesHalvesWithTiles) {
  const bool q = GetParam();
  std::mt19937 rng(q ? 801 : 802);
  constexpr uint32_t kRowsQ = 2;
  constexpr uint32_t kHeads = 2;
  constexpr uint32_t kN8 = 8 * kHeads;
  const Rows x = RandomRows(rng, 4 * kRowsQ, 8 * kN8, -3, 3);
  std::vector<FbufWord> tiles(size_t{2} * kRowsQ * 4);
  for (FbufWord& word : tiles) {
    for (uint32_t& value : word) {
      value = RandomFp32(rng, -2, 0);
    }
  }
  const VecArgs args{.op = kOpRope,
                     .src = 32,
                     .src2 = 0,
                     .dst = q ? 0U : 400U,
                     .rows_q = kRowsQ,
                     .n8 = kN8,
                     .p_base = 5,
                     .aux = kHeads,
                     .imm = 1500,
                     .q = q};
  VecBench bench(q ? 803 : 804);
  bench.Run({.args = args,
             .fbuf = {{32, ToWords(x)}},
             .pbuf = tiles,
             .tables = nullptr});
  Rows expected = x;
  for (uint32_t t = 0; t < x.size(); ++t) {
    const uint32_t quad = t / 4;
    for (uint32_t h = 0; h < kHeads; ++h) {
      for (uint32_t i = 0; i < 32; ++i) {
        const uint32_t gg = i / 8;
        const uint32_t element = ((t % 4) * 8) + (i % 8);
        const uint32_t cos = tiles.at((quad * 4) + gg).at(element);
        const uint32_t sin =
            tiles.at((kRowsQ * 4) + (quad * 4) + gg).at(element);
        const uint32_t x1 = x[t][(64 * h) + i];
        const uint32_t x2 = x[t][(64 * h) + 32 + i];
        expected[t][(64 * h) + i] = Fp32Sub(Fp32Mul(x1, cos), Fp32Mul(x2, sin));
        expected[t][(64 * h) + 32 + i] =
            Fp32Add(Fp32Mul(x2, cos), Fp32Mul(x1, sin));
      }
    }
  }
  CheckOutput(bench, args, expected);
}

INSTANTIATE_TEST_SUITE_P(Quantized, VecRopeTest, testing::Bool());

}  // namespace
}  // namespace hotarubi::tb
