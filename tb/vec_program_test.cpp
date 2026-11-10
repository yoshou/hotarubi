#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <random>
#include <vector>

#include "fp32_ref.hpp"
#include "npu_bench.hpp"
#include "npu_data.hpp"
#include "npu_isa.hpp"
#include "vec_ref.hpp"

namespace hotarubi::tb {
namespace {

constexpr uint32_t kFbufArea = Ddr::kBase + 0x100000U;
constexpr uint32_t kCbufArea = Ddr::kBase + 0x300000U;
constexpr uint32_t kTbufArea = Ddr::kBase + 0x340000U;
constexpr uint32_t kPbufArea = Ddr::kBase + 0x380000U;
constexpr uint32_t kOutArea = Ddr::kBase + 0x400000U;
constexpr uint32_t kProgramStart = 256;
constexpr uint32_t kOpRun = 16;

constexpr uint32_t kRegSrc = 1;
constexpr uint32_t kRegSrc2 = 2;
constexpr uint32_t kRegDst = 3;
constexpr uint32_t kRegRowsQ = 4;
constexpr uint32_t kRegN8 = 5;
constexpr uint32_t kRegPBase = 6;
constexpr uint32_t kRegImm = 8;
constexpr uint32_t kRegQ = 10;

constexpr uint32_t kSaluAdd = 0;
constexpr uint32_t kSaluSub = 1;
constexpr uint32_t kSaluAnd = 2;
constexpr uint32_t kSaluOr = 3;
constexpr uint32_t kSaluXor = 4;
constexpr uint32_t kSaluShl = 5;
constexpr uint32_t kSaluShr = 6;
constexpr uint32_t kSaluSra = 7;
constexpr uint32_t kSaluMul = 8;
constexpr uint32_t kSaluSlt = 9;
constexpr uint32_t kSaluSltu = 10;
constexpr uint32_t kSaluSeq = 11;
constexpr uint32_t kSaluMov = 12;

struct Pass {
  uint32_t rx = 0;
  uint32_t rx2 = 0;
  uint32_t rd = 0;
  uint32_t rq = 0;
  uint32_t rnw = 0;
  uint32_t rn8 = 0;
  uint32_t rpb = 0;
  uint32_t rimm = 0;
  uint32_t rimm2 = 0;
  bool x2_en = false;
  uint32_t mg = 0;
  uint32_t ms = 0;
  uint32_t mx = 0;
  uint32_t mx2 = 0;
  uint32_t md = 0;
  uint32_t pa = 0;
  uint32_t c0 = 0;
  uint32_t b = 0;
  uint32_t nl = 0;
  uint32_t po = 0;
  bool pv = false;
  bool mask_en = false;
  uint32_t red = 0;
  uint32_t wr = 1;
  uint32_t sf = 0;
  bool st = false;
  bool srn = false;
  bool seps = false;
  uint32_t px = 0;
  uint32_t cmp = 0;
  uint32_t cl = 0;
  bool cr = false;
  bool cmp_int = false;
  uint32_t sel = 0;
  uint32_t ialu = 0;
  bool dst_s1 = false;
  bool c0_pred = false;
  bool x_s1 = false;
  uint32_t ctx = 0;
  uint32_t rm = 0;
};

Instruction Encode(const Pass& p) {
  Instruction word;
  word.Set(124, 4, 0)
      .Set(119, 5, p.rx)
      .Set(114, 5, p.rx2)
      .Set(109, 5, p.rd)
      .Set(104, 5, p.rq)
      .Set(99, 5, p.rnw)
      .Set(94, 5, p.rn8)
      .Set(89, 5, p.rpb)
      .Set(84, 5, p.rimm)
      .Set(79, 5, p.rimm2)
      .Set(78, 1, p.x2_en ? 1 : 0)
      .Set(76, 2, p.mg)
      .Set(74, 2, p.ms)
      .Set(72, 2, p.mx)
      .Set(70, 2, p.mx2)
      .Set(68, 2, p.md)
      .Set(66, 2, p.pa)
      .Set(63, 3, p.c0)
      .Set(60, 3, p.b)
      .Set(58, 2, p.nl)
      .Set(56, 2, p.po)
      .Set(55, 1, p.pv ? 1 : 0)
      .Set(54, 1, p.mask_en ? 1 : 0)
      .Set(52, 2, p.red)
      .Set(50, 2, p.wr)
      .Set(47, 3, p.sf)
      .Set(46, 1, p.st ? 1 : 0)
      .Set(45, 1, p.srn ? 1 : 0)
      .Set(44, 1, p.seps ? 1 : 0)
      .Set(24, 20, p.px)
      .Set(21, 3, p.cmp)
      .Set(19, 2, p.cl)
      .Set(18, 1, p.cr ? 1 : 0)
      .Set(17, 1, p.cmp_int ? 1 : 0)
      .Set(14, 3, p.sel)
      .Set(11, 3, p.ialu)
      .Set(10, 1, p.c0_pred ? 1 : 0)
      .Set(9, 1, p.x_s1 ? 1 : 0)
      .Set(8, 1, p.dst_s1 ? 1 : 0)
      .Set(6, 2, p.ctx)
      .Set(4, 2, p.rm);
  return word;
}

Instruction Salu(uint32_t op, uint32_t rd, uint32_t ra, uint32_t rb) {
  Instruction word;
  word.Set(124, 4, 1)
      .Set(119, 5, op)
      .Set(114, 5, rd)
      .Set(109, 5, ra)
      .Set(104, 5, rb);
  return word;
}

Instruction SaluImm(uint32_t op, uint32_t rd, uint32_t ra, uint32_t imm) {
  Instruction word;
  word.Set(124, 4, 1)
      .Set(119, 5, op)
      .Set(114, 5, rd)
      .Set(109, 5, ra)
      .Set(103, 1, 1)
      .Set(0, 32, imm);
  return word;
}

struct Smem {
  uint32_t op = 0;
  uint32_t rd = 0;
  uint32_t ra = 0;
  uint32_t rb = 0;
  bool col_imm = true;
  uint32_t col = 0;
  uint32_t which = 0;
  uint32_t ctx = 0;
  uint32_t imm = 0;
};

Instruction Encode(const Smem& s) {
  Instruction word;
  word.Set(124, 4, 2)
      .Set(121, 3, s.op)
      .Set(116, 5, s.rd)
      .Set(111, 5, s.ra)
      .Set(106, 5, s.rb)
      .Set(105, 1, s.col_imm ? 1 : 0)
      .Set(100, 5, s.col)
      .Set(99, 1, s.which)
      .Set(96, 2, s.ctx)
      .Set(0, 32, s.imm);
  return word;
}

Instruction Branch(uint32_t cond, uint32_t ra, uint32_t target) {
  Instruction word;
  word.Set(124, 4, 3).Set(122, 2, cond).Set(117, 5, ra).Set(0, 16, target);
  return word;
}

Instruction Ret() {
  Instruction word;
  word.Set(124, 4, 4);
  return word;
}

constexpr uint32_t kSmemSld = 0;
constexpr uint32_t kSmemRsmov = 1;
constexpr uint32_t kSmemRn = 2;
constexpr uint32_t kRs1 = 0;
constexpr uint32_t kRs0 = 1;

constexpr uint32_t kRegZero = 0;
constexpr uint32_t kRegOne = 21;
constexpr uint32_t kRegOut = 20;
constexpr uint32_t kRegValue = 11;

std::vector<Instruction> Emit(uint32_t value_register) {
  return {Encode(Pass{.rx = kRegSrc2,
                      .rd = kRegOut,
                      .rq = kRegOne,
                      .rnw = kRegOne,
                      .rn8 = kRegOne,
                      .rimm = value_register,
                      .c0 = 4,
                      .ialu = 4}),
          SaluImm(kSaluAdd, kRegOut, kRegOut, 1)};
}

struct ProgramRun {
  std::vector<Instruction> program;
  VecArgs args;
  std::map<uint32_t, std::vector<FbufWord>> fbuf;
  std::vector<FbufWord> pbuf;
  const Tables* tables = nullptr;
  uint32_t out_base = 0;
  uint32_t out_words = 0;
};

std::vector<FbufWord> RunProgramOn(NpuBench& bench, const ProgramRun& run) {
  Ddr& ddr = bench.Memory();
  std::vector<Instruction> stream;
  for (uint32_t i = 0; i < run.program.size(); ++i) {
    for (uint32_t j = 0; j < 4; ++j) {
      ddr.Write32(kCbufArea + (16 * i) + (4 * j), run.program[i].words.at(j));
    }
  }
  stream.push_back(
      DmaLoad({.ddr_addr = kCbufArea,
               .mem = Mem::kCbuf,
               .onchip = kProgramStart,
               .rows = 1,
               .row_words = static_cast<uint32_t>(run.program.size())}));
  uint32_t area = kFbufArea;
  for (const auto& [address, words] : run.fbuf) {
    for (uint32_t i = 0; i < words.size(); ++i) {
      WriteFbuf(ddr, area + (i * kFbufWordBytes), words[i]);
    }
    stream.push_back(
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
    stream.push_back(
        DmaLoad({.ddr_addr = kPbufArea,
                 .mem = Mem::kPbuf,
                 .onchip = 0,
                 .rows = 1,
                 .row_words = static_cast<uint32_t>(run.pbuf.size())}));
  }
  if (run.tables != nullptr) {
    for (uint32_t i = 0; i < 1024; ++i) {
      WriteTbuf(ddr, kTbufArea + (i * kTbufWordBytes), run.tables->v.at(i),
                run.tables->s.at(i));
    }
    stream.push_back(DmaLoad({.ddr_addr = kTbufArea,
                              .mem = Mem::kTbuf,
                              .onchip = 0,
                              .rows = 1,
                              .row_words = 1024}));
  }
  VecArgs args = run.args;
  args.op = kOpRun;
  args.aux = kProgramStart;
  args.wait = kWaitDma;
  stream.push_back(Vec(args));
  stream.push_back(DmaStore({.ddr_addr = kOutArea,
                             .mem = Mem::kFbuf,
                             .onchip = run.out_base,
                             .rows = 1,
                             .row_words = run.out_words,
                             .wait = kWaitVec}));
  stream.push_back(Halt());
  EXPECT_TRUE(bench.Run(stream));
  EXPECT_EQ(bench.ReadReg(0x04) & 0x6U, 0x2U);
  EXPECT_TRUE(bench.Errors().empty());
  std::vector<FbufWord> out;
  out.reserve(run.out_words);
  for (uint32_t i = 0; i < run.out_words; ++i) {
    out.push_back(ReadFbuf(ddr, kOutArea + (i * kFbufWordBytes)));
  }
  return out;
}

std::vector<FbufWord> RunProgram(const ProgramRun& run, uint32_t seed) {
  NpuBench bench(seed);
  return RunProgramOn(bench, run);
}

FbufWord Splat(uint32_t value) {
  FbufWord word{};
  word.fill(value);
  return word;
}

uint32_t SaluReference(uint32_t op, uint32_t a, uint32_t b) {
  switch (op) {
    case kSaluAdd:
      return a + b;
    case kSaluSub:
      return a - b;
    case kSaluAnd:
      return a & b;
    case kSaluOr:
      return a | b;
    case kSaluXor:
      return a ^ b;
    case kSaluShl:
      return a << (b & 31U);
    case kSaluShr:
      return a >> (b & 31U);
    case kSaluSra: {
      const uint32_t shift = b & 31U;
      const uint32_t fill =
          (a & kFp32SignBit) != 0 && shift != 0 ? ~(~0U >> shift) : 0U;
      return (a >> shift) | fill;
    }
    case kSaluMul:
      return a * b;
    case kSaluSlt: {
      const auto sa = static_cast<int64_t>(a) -
                      ((a & kFp32SignBit) != 0 ? (int64_t{1} << 32) : 0);
      const auto sb = static_cast<int64_t>(b) -
                      ((b & kFp32SignBit) != 0 ? (int64_t{1} << 32) : 0);
      return sa < sb ? 1 : 0;
    }
    case kSaluSltu:
      return a < b ? 1 : 0;
    case kSaluSeq:
      return a == b ? 1 : 0;
    default:
      return b;
  }
}

TEST(VecProgramTest, ScalarAluAndRegisterLoads) {
  constexpr uint32_t kOut = 100;
  const VecArgs args{.op = 0,
                     .src = 37,
                     .src2 = 0,
                     .dst = kOut,
                     .rows_q = 5,
                     .n8 = 9,
                     .p_base = 3,
                     .aux = 0,
                     .imm = 0x80001234U,
                     .q = false};
  std::vector<Instruction> program = {SaluImm(kSaluMov, kRegOut, 0, kOut),
                                      SaluImm(kSaluMov, kRegOne, 0, 1)};
  std::vector<uint32_t> expected;
  const auto emit = [&](uint32_t reg, uint32_t value) {
    for (const Instruction& word : Emit(reg)) {
      program.push_back(word);
    }
    expected.push_back(value);
  };
  emit(kRegSrc, 37);
  emit(kRegSrc2, 0);
  emit(kRegDst, kOut);
  emit(kRegRowsQ, 5);
  emit(kRegN8, 9);
  emit(kRegPBase, 3);
  emit(kRegImm, 0x80001234U);
  emit(kRegQ, 0);
  std::mt19937 rng(71);
  constexpr uint32_t kRegA = 12;
  constexpr uint32_t kRegB = 13;
  for (uint32_t op = kSaluAdd; op <= kSaluMov; ++op) {
    for (int variant = 0; variant < 3; ++variant) {
      uint32_t a = rng();
      uint32_t b = rng();
      if (op >= kSaluShl && op <= kSaluSra) {
        b &= 31U;
      }
      if (variant == 2 && op >= kSaluSlt) {
        b = a;
      }
      program.push_back(SaluImm(kSaluMov, kRegA, 0, a));
      program.push_back(SaluImm(kSaluMov, kRegB, 0, b));
      if (variant == 0) {
        program.push_back(Salu(op, kRegValue, kRegA, kRegB));
      } else {
        program.push_back(SaluImm(op, kRegValue, kRegA, b));
      }
      emit(kRegValue, SaluReference(op, a, b));
    }
  }
  program.push_back(SaluImm(kSaluMov, 0, 0, 0x55));
  emit(kRegZero, 0);
  program.push_back(Ret());
  const std::vector<FbufWord> out =
      RunProgram({.program = program,
                  .args = args,
                  .fbuf = {{0, {FbufWord{}}}},
                  .pbuf = {},
                  .tables = nullptr,
                  .out_base = kOut,
                  .out_words = static_cast<uint32_t>(expected.size())},
                 71);
  for (uint32_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(out[i], Splat(expected[i])) << "result " << i;
  }
}

TEST(VecProgramTest, BranchLoopAndScalarLoads) {
  std::mt19937 rng(72);
  constexpr uint32_t kSrc = 40;
  constexpr uint32_t kOut = 200;
  std::vector<FbufWord> data(6);
  for (FbufWord& word : data) {
    for (uint32_t& value : word) {
      value = rng();
    }
  }
  constexpr uint32_t kRegCount = 12;
  constexpr uint32_t kRegPointer = 14;
  constexpr uint32_t kRegColumn = 15;
  const uint32_t loop = kProgramStart + 5;
  std::vector<Instruction> program = {
      SaluImm(kSaluMov, kRegOut, 0, kOut),
      SaluImm(kSaluMov, kRegOne, 0, 1),
      SaluImm(kSaluMov, kRegCount, 0, 4),
      SaluImm(kSaluAdd, kRegPointer, kRegSrc, 1),
      Branch(0, 0, loop),
      Encode(Smem{.op = kSmemSld,
                  .rd = kRegValue,
                  .ra = kRegPointer,
                  .col_imm = true,
                  .col = 5,
                  .imm = 0}),
  };
  for (const Instruction& word : Emit(kRegValue)) {
    program.push_back(word);
  }
  program.push_back(SaluImm(kSaluAdd, kRegPointer, kRegPointer, 1));
  program.push_back(SaluImm(kSaluSub, kRegCount, kRegCount, 1));
  program.push_back(Branch(2, kRegCount, loop));
  program.push_back(Branch(
      1, kRegCount, static_cast<uint32_t>(kProgramStart + program.size() + 2)));
  program.push_back(Ret());
  program.push_back(SaluImm(kSaluMov, kRegColumn, 0, 0xFFFFFF00U | 27U));
  program.push_back(Encode(Smem{.op = kSmemSld,
                                .rd = kRegValue,
                                .ra = kRegPointer,
                                .rb = kRegColumn,
                                .col_imm = false,
                                .imm = 0xFFFFFFFFU}));
  for (const Instruction& word : Emit(kRegValue)) {
    program.push_back(word);
  }
  program.push_back(Ret());
  const VecArgs args{.op = 0,
                     .src = kSrc + 1,
                     .src2 = 0,
                     .dst = 0,
                     .rows_q = 1,
                     .n8 = 1,
                     .p_base = 0,
                     .aux = 0,
                     .imm = 0,
                     .q = false};
  const std::vector<FbufWord> out =
      RunProgram({.program = program,
                  .args = args,
                  .fbuf = {{0, {FbufWord{}}}, {kSrc, data}},
                  .pbuf = {},
                  .tables = nullptr,
                  .out_base = kOut,
                  .out_words = 5},
                 72);
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT_EQ(out[i], Splat(data[2 + i][5])) << "iteration " << i;
  }
  EXPECT_EQ(out[4], Splat(data[5][27]));
}

constexpr uint32_t kCmpLt = 1;
constexpr uint32_t kCmpLe = 2;
constexpr uint32_t kCmpEq = 3;
constexpr uint32_t kCmpNe = 4;
constexpr uint32_t kCmpGt = 5;
constexpr uint32_t kCmpGe = 6;
constexpr uint32_t kLeftX2 = 1;
constexpr uint32_t kLeftColumn = 2;
constexpr uint32_t kLeftPosition = 3;
constexpr uint32_t kSelZero = 1;
constexpr uint32_t kSelX2 = 2;
constexpr uint32_t kSelImm = 3;
constexpr uint32_t kSelNull = 4;
constexpr uint32_t kSelRs1 = 5;
constexpr uint32_t kC0X2 = 1;
constexpr uint32_t kC0Imm = 4;
constexpr uint32_t kBRs1 = 3;
constexpr uint32_t kBPre = 5;
constexpr uint32_t kRedMax = 2;
constexpr uint32_t kSfIdentity = 1;

bool Relation(uint32_t cmp, bool integer, uint32_t a, uint32_t b) {
  bool less = false;
  bool equal = false;
  if (integer) {
    const auto sa = static_cast<int64_t>(a) -
                    ((a & kFp32SignBit) != 0 ? (int64_t{1} << 32) : 0);
    const auto sb = static_cast<int64_t>(b) -
                    ((b & kFp32SignBit) != 0 ? (int64_t{1} << 32) : 0);
    less = sa < sb;
    equal = sa == sb;
  } else {
    less = Fp32Less(a, b);
    equal = !less && !Fp32Less(b, a);
  }
  switch (cmp) {
    case kCmpLt:
      return less;
    case kCmpLe:
      return less || equal;
    case kCmpEq:
      return equal;
    case kCmpNe:
      return !equal;
    case kCmpGt:
      return !less && !equal;
    case kCmpGe:
      return !less;
    default:
      return true;
  }
}

struct LaneCase {
  Pass pass;
  std::function<uint32_t(uint32_t, uint32_t, uint32_t, uint32_t)> expected;
};

TEST(VecProgramTest, IntegerPathPredicatesAndSelection) {
  std::mt19937 rng(73);
  constexpr uint32_t kX = 10;
  constexpr uint32_t kX2 = 20;
  constexpr uint32_t kOut = 300;
  constexpr uint32_t kWords = 2;
  std::vector<FbufWord> x(kWords);
  std::vector<FbufWord> x2(kWords);
  for (uint32_t w = 0; w < kWords; ++w) {
    for (uint32_t e = 0; e < 32; ++e) {
      x[w][e] = RandomFp32(rng, -3, 3);
      x2[w][e] = RandomFp32(rng, -3, 3);
    }
  }
  constexpr uint32_t kImm = 0x00010003U;
  constexpr uint32_t kImmHalf = 0x3F000000U;
  const uint32_t equal_value = x2[0][3];
  x2[1][17] = equal_value;
  constexpr uint32_t kRs1Value = 0xC0000000U;
  constexpr uint32_t kRs1Word = 40;
  constexpr uint32_t kRegX = 12;
  constexpr uint32_t kRegX2 = 13;
  constexpr uint32_t kRegWords = 16;
  constexpr uint32_t kRegImmValue = 18;
  constexpr uint32_t kRegHalf = 19;
  constexpr uint32_t kRegEqual = 22;
  constexpr uint32_t kRegThirteen = 23;
  constexpr uint32_t kRegFour = 24;
  constexpr uint32_t kRegRs1Word = 26;
  constexpr uint32_t kRegZeroFp = 27;
  const Pass base{.rx = kRegX,
                  .rx2 = kRegX2,
                  .rd = kRegOut,
                  .rq = kRegOne,
                  .rnw = kRegWords,
                  .rn8 = kRegWords,
                  .rimm = kRegImmValue,
                  .rimm2 = kRegHalf};
  const auto with = [&](const std::function<void(Pass&)>& edit) {
    Pass pass = base;
    edit(pass);
    return pass;
  };
  const std::vector<LaneCase> cases = {
      {with([](Pass& p) {
         p.c0 = kC0Imm;
         p.ialu = 1;
       }),
       [](uint32_t a, uint32_t, uint32_t, uint32_t) { return a + kImm; }},
      {with([](Pass& p) {
         p.c0 = kC0Imm;
         p.ialu = 2;
       }),
       [](uint32_t a, uint32_t, uint32_t, uint32_t) { return a - kImm; }},
      {with([](Pass& p) {
         p.c0 = kC0Imm;
         p.ialu = 3;
       }),
       [](uint32_t a, uint32_t, uint32_t, uint32_t) { return a & kImm; }},
      {with([](Pass& p) {
         p.c0 = kC0Imm;
         p.ialu = 5;
       }),
       [](uint32_t a, uint32_t, uint32_t, uint32_t) { return a ^ kImm; }},
      {with([](Pass& p) {
         p.c0 = kC0Imm;
         p.ialu = 6;
       }),
       [](uint32_t a, uint32_t, uint32_t, uint32_t) { return a >> 1; }},
      {with([](Pass& p) {
         p.c0 = kC0Imm;
         p.ialu = 7;
       }),
       [](uint32_t a, uint32_t, uint32_t, uint32_t) { return kImm - a; }},
      {with([](Pass& p) {
         p.cmp = kCmpLt;
         p.sel = kSelZero;
       }),
       [](uint32_t a, uint32_t, uint32_t, uint32_t) {
         return Relation(kCmpLt, false, a, kImmHalf) ? a : 0U;
       }},
      {with([](Pass& p) {
         p.x2_en = true;
         p.cmp = kCmpGe;
         p.cr = true;
         p.sel = kSelX2;
       }),
       [](uint32_t a, uint32_t b, uint32_t, uint32_t) {
         return Relation(kCmpGe, false, a, b) ? a : b;
       }},
      {with([](Pass& p) {
         p.x2_en = true;
         p.rimm2 = kRegEqual;
         p.cmp = kCmpEq;
         p.cl = kLeftX2;
         p.cmp_int = true;
         p.sel = kSelNull;
         p.ialu = 4;
       }),
       [equal_value](uint32_t a, uint32_t b, uint32_t, uint32_t) {
         return b == equal_value ? a : 0xFFFFFFFFU;
       }},
      {with([](Pass& p) {
         p.rimm2 = kRegThirteen;
         p.cmp = kCmpLt;
         p.cl = kLeftPosition;
         p.sel = kSelImm;
       }),
       [](uint32_t a, uint32_t, uint32_t e, uint32_t) {
         return e < 13 ? a : kImm;
       }},
      {with([](Pass& p) {
         p.rimm2 = kRegFour;
         p.cmp = kCmpGt;
         p.cl = kLeftColumn;
         p.sel = kSelRs1;
       }),
       [](uint32_t a, uint32_t, uint32_t e, uint32_t) {
         return e % 8 > 4 ? a : kRs1Value;
       }},
      {with([](Pass& p) {
         p.rimm2 = kRegEqual;
         p.cmp = kCmpNe;
         p.cmp_int = true;
         p.sel = kSelRs1;
         p.x2_en = true;
         p.c0 = kC0X2;
       }),
       [](uint32_t a, uint32_t b, uint32_t, uint32_t) {
         return Fp32Add(a, b);
       }},
      {with([](Pass& p) {
         p.rimm2 = kRegZeroFp;
         p.c0 = kC0Imm;
         p.c0_pred = true;
         p.cmp = kCmpLe;
         p.ialu = 1;
       }),
       [](uint32_t a, uint32_t, uint32_t, uint32_t) {
         return a + (Relation(kCmpLe, false, a, kFp32Zero) ? kImm : 0U);
       }},
      {with([](Pass& p) {
         p.rimm2 = kRegThirteen;
         p.mask_en = true;
       }),
       [](uint32_t a, uint32_t, uint32_t e, uint32_t w) {
         return (8 * w) + (e % 8) >= 13 ? 0U : a;
       }},
  };
  std::vector<Instruction> program = {
      SaluImm(kSaluMov, kRegOut, 0, kOut),
      SaluImm(kSaluMov, kRegOne, 0, 1),
      SaluImm(kSaluMov, kRegX, 0, kX),
      SaluImm(kSaluMov, kRegX2, 0, kX2),
      SaluImm(kSaluMov, kRegWords, 0, kWords),
      SaluImm(kSaluMov, kRegImmValue, 0, kImm),
      SaluImm(kSaluMov, kRegHalf, 0, kImmHalf),
      SaluImm(kSaluMov, kRegEqual, 0, equal_value),
      SaluImm(kSaluMov, kRegThirteen, 0, 13),
      SaluImm(kSaluMov, kRegFour, 0, 4),
      SaluImm(kSaluMov, kRegRs1Word, 0, kRs1Word),
      SaluImm(kSaluMov, kRegZeroFp, 0, kFp32Zero),
      Encode(Pass{.rx = kRegRs1Word,
                  .rq = kRegOne,
                  .rnw = kRegOne,
                  .rn8 = kRegOne,
                  .red = kRedMax,
                  .wr = 0,
                  .sf = kSfIdentity}),
  };
  FbufWord rs1_word{};
  rs1_word.fill(kRs1Value);
  for (const LaneCase& lane : cases) {
    program.push_back(Encode(lane.pass));
    program.push_back(SaluImm(kSaluAdd, kRegOut, kRegOut, kWords));
  }
  program.push_back(Ret());
  const VecArgs args{.op = 0,
                     .src = 0,
                     .src2 = 0,
                     .dst = 0,
                     .rows_q = 1,
                     .n8 = 1,
                     .p_base = 0,
                     .aux = 0,
                     .imm = 0,
                     .q = false};
  const std::vector<FbufWord> out =
      RunProgram({.program = program,
                  .args = args,
                  .fbuf = {{kX, x}, {kX2, x2}, {kRs1Word, {rs1_word}}},
                  .pbuf = {},
                  .tables = nullptr,
                  .out_base = kOut,
                  .out_words = static_cast<uint32_t>(cases.size() * kWords)},
                 73);
  for (uint32_t i = 0; i < cases.size(); ++i) {
    for (uint32_t w = 0; w < kWords; ++w) {
      for (uint32_t e = 0; e < 32; ++e) {
        ASSERT_EQ(out[(i * kWords) + w][e],
                  cases[i].expected(x[w][e], x2[w][e], e, w))
            << "case " << i << " word " << w << " element " << e;
      }
    }
  }
}

uint32_t PermutationCode(uint32_t px, uint32_t pcol) {
  switch (px) {
    case 1:
      return (((pcol >> 2) & 1U) != 0 ? 7U : 6U) << 8 |
             (((pcol >> 1) & 1U) != 0 ? 7U : 6U) << 4 |
             ((pcol & 1U) != 0 ? 7U : 6U);
    case 2:
      return 8U << 12;
    case 3:
      return 8U << 16;
    case 4:
      return (4U << 16) | (5U << 12) | (1U << 8) | (2U << 4) | 3U;
    case 5:
      return (2U << 16) | (3U << 12) | (1U << 8);
    default:
      return 0;
  }
}

uint32_t PermutedSource(uint32_t px, uint32_t pcol, uint32_t e) {
  const uint32_t t = e / 8;
  const uint32_t c = e % 8;
  switch (px) {
    case 1:
      return (t * 8) + pcol;
    case 2:
      return ((t ^ 1U) * 8) + c;
    case 3:
      return ((t ^ 2U) * 8) + c;
    case 4:
      return ((e % 4) * 8) + (e / 4);
    case 5:
      return (((e / 4) % 4) * 8) + (e % 4) + (4 * (e / 16));
    default:
      return e;
  }
}

TEST(VecProgramTest, PermutationsAndWordMappings) {
  std::mt19937 rng(74);
  constexpr uint32_t kA = 0;
  constexpr uint32_t kB = 40;
  constexpr uint32_t kOut = 500;
  std::vector<FbufWord> a(32);
  std::vector<FbufWord> b(32);
  for (uint32_t w = 0; w < 32; ++w) {
    for (uint32_t e = 0; e < 32; ++e) {
      a[w][e] = RandomFp32(rng, -3, 3);
      b[w][e] = RandomFp32(rng, -3, 3);
    }
  }
  constexpr uint32_t kRegA = 12;
  constexpr uint32_t kRegB = 13;
  constexpr uint32_t kRegTwo = 14;
  constexpr uint32_t kRegFour = 15;
  constexpr uint32_t kRegSixteen = 16;
  std::vector<Instruction> program = {
      SaluImm(kSaluMov, kRegOut, 0, kOut),   SaluImm(kSaluMov, kRegOne, 0, 1),
      SaluImm(kSaluMov, kRegA, 0, kA),       SaluImm(kSaluMov, kRegB, 0, kB),
      SaluImm(kSaluMov, kRegTwo, 0, 2),      SaluImm(kSaluMov, kRegFour, 0, 4),
      SaluImm(kSaluMov, kRegSixteen, 0, 16),
  };
  std::vector<FbufWord> expected;
  for (uint32_t px = 1; px <= 5; ++px) {
    const uint32_t pcol = px == 1 ? 5 : 0;
    program.push_back(Encode(Pass{.rx = kRegA,
                                  .rd = kRegOut,
                                  .rq = kRegOne,
                                  .rnw = kRegTwo,
                                  .rn8 = kRegTwo,
                                  .px = PermutationCode(px, pcol)}));
    program.push_back(SaluImm(kSaluAdd, kRegOut, kRegOut, 2));
    for (uint32_t w = 0; w < 2; ++w) {
      FbufWord word{};
      for (uint32_t e = 0; e < 32; ++e) {
        word[e] = a[w][PermutedSource(px, pcol, e)];
      }
      expected.push_back(word);
    }
  }
  program.push_back(Encode(Pass{.rx = kRegA,
                                .rd = kRegOut,
                                .rq = kRegTwo,
                                .rnw = kRegFour,
                                .rn8 = kRegSixteen,
                                .mg = 2,
                                .ms = 3,
                                .mx = 2}));
  program.push_back(SaluImm(kSaluAdd, kRegOut, kRegOut, 32));
  std::vector<FbufWord> mapped(32);
  for (uint32_t q = 0; q < 2; ++q) {
    for (uint32_t w = 0; w < 4; ++w) {
      mapped[(q * 16) + w] = a[(q * 16) + ((w / 4) * 8) + 4 + (w % 4)];
    }
  }
  program.push_back(Encode(Pass{.rx = kRegA,
                                .rx2 = kRegB,
                                .rd = kRegOut,
                                .rq = kRegTwo,
                                .rnw = kRegFour,
                                .rn8 = kRegSixteen,
                                .x2_en = true,
                                .ms = 2,
                                .md = 1,
                                .c0 = kC0X2,
                                .dst_s1 = true,
                                .x_s1 = true}));
  std::vector<FbufWord> strided(32);
  for (uint32_t q = 0; q < 2; ++q) {
    for (uint32_t w = 0; w < 4; ++w) {
      FbufWord& word = strided[q + (4 * w)];
      for (uint32_t e = 0; e < 32; ++e) {
        word[e] = Fp32Add(a[q + w][e], b[q + ((w / 4) * 8) + (w % 4)][e]);
      }
    }
  }
  program.push_back(Ret());
  const VecArgs args{.op = 0,
                     .src = 0,
                     .src2 = 0,
                     .dst = 0,
                     .rows_q = 1,
                     .n8 = 1,
                     .p_base = 0,
                     .aux = 0,
                     .imm = 0,
                     .q = false};
  std::vector<FbufWord> zero_out(80);
  const std::vector<FbufWord> out =
      RunProgram({.program = program,
                  .args = args,
                  .fbuf = {{kA, a}, {kB, b}, {kOut, zero_out}},
                  .pbuf = {},
                  .tables = nullptr,
                  .out_base = kOut,
                  .out_words = 80},
                 74);
  for (uint32_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(out[i], expected[i]) << "permutation word " << i;
  }
  for (uint32_t q = 0; q < 2; ++q) {
    for (uint32_t w = 0; w < 4; ++w) {
      EXPECT_EQ(out[10 + (q * 16) + w], mapped[(q * 16) + w])
          << "mapped q " << q << " w " << w;
      EXPECT_EQ(out[42 + q + (4 * w)], strided[q + (4 * w)])
          << "strided q " << q << " w " << w;
    }
  }
}

TEST(VecProgramTest, RowScalarsAndReductions) {
  std::mt19937 rng(75);
  constexpr uint32_t kX = 8;
  constexpr uint32_t kOut = 600;
  constexpr uint32_t kWords = 3;
  constexpr uint32_t kCount = 24;
  std::vector<FbufWord> x(kWords);
  for (FbufWord& word : x) {
    for (uint32_t& value : word) {
      value = RandomFp32(rng, -2, 2);
    }
  }
  constexpr uint32_t kRegX = 12;
  constexpr uint32_t kRegWords = 13;
  constexpr uint32_t kRegCount = 14;
  constexpr uint32_t kRegEps = 15;
  constexpr uint32_t kEps = 0x3A83126FU;
  std::vector<Instruction> program = {
      SaluImm(kSaluMov, kRegOut, 0, kOut),
      SaluImm(kSaluMov, kRegOne, 0, 1),
      SaluImm(kSaluMov, kRegX, 0, kX),
      SaluImm(kSaluMov, kRegWords, 0, kWords),
      SaluImm(kSaluMov, kRegCount, 0, kCount),
      SaluImm(kSaluMov, kRegEps, 0, kEps),
      Encode(Smem{.op = kSmemRn, .ra = kRegCount}),
  };
  const auto emit_scalars = [&](uint32_t which) {
    for (uint32_t t = 0; t < 4; ++t) {
      program.push_back(Encode(
          Smem{.op = kSmemRsmov, .rd = kRegValue, .col = t, .which = which}));
      for (const Instruction& word : Emit(kRegValue)) {
        program.push_back(word);
      }
    }
  };
  const Pass reduce{.rx = kRegX,
                    .rd = 0,
                    .rq = kRegOne,
                    .rnw = kRegWords,
                    .rn8 = kRegWords,
                    .rimm = kRegEps,
                    .red = 1,
                    .wr = 0};
  const auto pass_with = [&](uint32_t red, uint32_t sc, uint32_t b) {
    struct Scalar {
      uint32_t sf;
      bool st;
      bool srn;
      bool seps;
    };
    const std::array<Scalar, 6> scalars = {
        {{.sf = 0, .st = false, .srn = false, .seps = false},
         {.sf = 2, .st = true, .srn = true, .seps = false},
         {.sf = 4, .st = false, .srn = true, .seps = true},
         {.sf = 2, .st = true, .srn = false, .seps = false},
         {.sf = 3, .st = false, .srn = false, .seps = false},
         {.sf = 1, .st = false, .srn = false, .seps = false}}};
    Pass pass = reduce;
    pass.red = red;
    pass.sf = scalars.at(sc).sf;
    pass.st = scalars.at(sc).st;
    pass.srn = scalars.at(sc).srn;
    pass.seps = scalars.at(sc).seps;
    pass.b = b;
    return Encode(pass);
  };
  program.push_back(pass_with(1, 1, 0));
  emit_scalars(kRs0);
  program.push_back(pass_with(1, 2, kBPre));
  emit_scalars(kRs1);
  program.push_back(pass_with(2, 3, 0));
  emit_scalars(kRs0);
  program.push_back(pass_with(1, 4, 0));
  emit_scalars(kRs1);
  program.push_back(pass_with(2, 5, 0));
  emit_scalars(kRs1);
  program.push_back(Encode(Pass{.rx = kRegX,
                                .rd = kRegOut,
                                .rq = kRegOne,
                                .rnw = kRegWords,
                                .rn8 = kRegWords,
                                .b = kBRs1}));
  program.push_back(Ret());
  const VecArgs args{.op = 0,
                     .src = 0,
                     .src2 = 0,
                     .dst = 0,
                     .rows_q = 1,
                     .n8 = 1,
                     .p_base = 0,
                     .aux = 0,
                     .imm = 0,
                     .q = false};
  const std::vector<FbufWord> out = RunProgram({.program = program,
                                                .args = args,
                                                .fbuf = {{kX, x}},
                                                .pbuf = {},
                                                .tables = nullptr,
                                                .out_base = kOut,
                                                .out_words = (5 * 4) + kWords},
                                               75);
  const uint32_t rn = Fp32Recip(Fp32FromInt(kCount, 0));
  for (uint32_t t = 0; t < 4; ++t) {
    std::vector<uint32_t> row;
    std::vector<uint32_t> squares;
    for (uint32_t w = 0; w < kWords; ++w) {
      for (uint32_t c = 0; c < 8; ++c) {
        row.push_back(x[w][(t * 8) + c]);
        squares.push_back(Fp32Mul(row.back(), row.back()));
      }
    }
    const uint32_t sum = RowSum(row);
    const uint32_t max = RowMax(row);
    const std::vector<uint32_t> scalars = {
        Fp32Neg(Fp32Mul(sum, rn)),
        Fp32Rsqrt(Fp32Add(Fp32Mul(RowSum(squares), rn), kEps)),
        Fp32Neg(max),
        Fp32Recip(sum),
        max,
    };
    for (uint32_t k = 0; k < scalars.size(); ++k) {
      EXPECT_EQ(out[(k * 4) + t], Splat(scalars[k]))
          << "scalar " << k << " token " << t;
    }
    for (uint32_t w = 0; w < kWords; ++w) {
      for (uint32_t c = 0; c < 8; ++c) {
        EXPECT_EQ(out[20 + w][(t * 8) + c],
                  Fp32Mul(x[w][(t * 8) + c], scalars[4]))
            << "scaled token " << t << " word " << w;
      }
    }
  }
}

TEST(VecProgramTest, FractionTable) {
  std::mt19937 rng(76);
  constexpr uint32_t kX = 4;
  constexpr uint32_t kOut = 700;
  const Tables tables = MakeTables(ActFunction::kSigmoid);
  std::vector<FbufWord> x(2);
  for (FbufWord& word : x) {
    for (uint32_t& value : word) {
      value = RandomFp32(rng, -6, 2);
    }
  }
  constexpr uint32_t kRegX = 12;
  constexpr uint32_t kRegTwo = 13;
  const std::vector<Instruction> program = {SaluImm(kSaluMov, kRegOut, 0, kOut),
                                            SaluImm(kSaluMov, kRegOne, 0, 1),
                                            SaluImm(kSaluMov, kRegX, 0, kX),
                                            SaluImm(kSaluMov, kRegTwo, 0, 2),
                                            Encode(Pass{.rx = kRegX,
                                                        .rd = kRegOut,
                                                        .rq = kRegOne,
                                                        .rnw = kRegTwo,
                                                        .rn8 = kRegTwo,
                                                        .nl = 3}),
                                            Ret()};
  const VecArgs args{.op = 0,
                     .src = 0,
                     .src2 = 0,
                     .dst = 0,
                     .rows_q = 1,
                     .n8 = 1,
                     .p_base = 0,
                     .aux = 0,
                     .imm = 0,
                     .q = false};
  const std::vector<FbufWord> out = RunProgram({.program = program,
                                                .args = args,
                                                .fbuf = {{kX, x}},
                                                .pbuf = {},
                                                .tables = &tables,
                                                .out_base = kOut,
                                                .out_words = 2},
                                               76);
  for (uint32_t w = 0; w < 2; ++w) {
    for (uint32_t e = 0; e < 32; ++e) {
      const uint32_t value = x[w][e];
      const uint32_t field = (value >> 23) & 0xFFU;
      const uint64_t mantissa = (value & 0x7FFFFFU) | 0x800000U;
      const int shift = static_cast<int>(field) - 125;
      const uint64_t magnitude =
          shift >= 0 ? mantissa << shift : mantissa >> -shift;
      const bool inexact =
          shift < 0 && (mantissa & ((uint64_t{1} << -shift) - 1)) != 0;
      const bool negative = (value & kFp32SignBit) != 0;
      const uint64_t fixed =
          negative ? (uint64_t{0} - magnitude - (inexact ? 1U : 0U))
                   : magnitude;
      const auto f = static_cast<uint32_t>(fixed & 0x1FFFFFFU);
      EXPECT_EQ(out[w][e], Interpolate(tables, f >> 16, f & 0xFFFFU))
          << "word " << w << " element " << e;
    }
  }
}

uint32_t RotatedColumn(uint32_t column, uint32_t rotation) {
  return ((column >> rotation) | (column << (3 - rotation))) & 7U;
}

void CheckTransposeProgram(uint32_t rows, uint32_t rotation) {
  std::mt19937 rng(913 + rows);
  std::vector<FbufWord> input(size_t{8} * rows);
  for (auto& word : input) {
    for (auto& value : word) {
      value = static_cast<uint32_t>(rng());
    }
  }
  Instruction transpose;
  transpose.Set(124, 4, 7)
      .Set(119, 5, kRegSrc)
      .Set(114, 5, 11)
      .Set(109, 5, kRegDst)
      .Set(104, 5, kRegRowsQ)
      .Set(99, 5, 11)
      .Set(94, 5, kRegN8)
      .Set(89, 5, kRegRowsQ)
      .Set(87, 2, rotation);
  const auto output =
      RunProgram({.program = {SaluImm(kSaluMov, 11, 0, 1), transpose, Ret()},
                  .args = {.src = 0, .dst = 2048, .rows_q = rows, .n8 = 8},
                  .fbuf = {{0, input}},
                  .pbuf = {},
                  .out_base = 2048,
                  .out_words = 8 * rows},
                 913 + rows);
  ASSERT_EQ(output.size(), 8 * rows);
  for (uint32_t sample = 0; sample < 8; ++sample) {
    for (uint32_t q = 0; q < rows; ++q) {
      for (uint32_t t = 0; t < 4; ++t) {
        for (uint32_t field = 0; field < 8; ++field) {
          const uint32_t row = RotatedColumn(field, rotation);
          EXPECT_EQ(output[(sample * rows) + q][(t * 8) + field],
                    input[(q * 8) + row][(t * 8) + sample]);
        }
      }
    }
  }
}

TEST(VecProgramTest, Transpose8PreservesEveryBitInEveryRotation) {
  for (const uint32_t rotation : {0U, 1U, 2U}) {
    for (const uint32_t rows : {1U, 3U, 32U}) {
      CheckTransposeProgram(rows, rotation);
    }
  }
}

void CheckStridedTransposeProgram(uint32_t rotation) {
  std::mt19937 rng(947);
  std::vector<FbufWord> input(377);
  for (auto& word : input) {
    for (auto& value : word) {
      value = static_cast<uint32_t>(rng());
    }
  }
  std::vector<FbufWord> expected(393);
  for (auto& word : expected) {
    word.fill(0xDADADADAU);
  }
  const auto initial = expected;
  for (uint32_t q = 0; q < 2; ++q) {
    for (uint32_t row = 0; row < 8; ++row) {
      for (uint32_t t = 0; t < 4; ++t) {
        for (uint32_t column = 0; column < 8; ++column) {
          const uint32_t source_row = RotatedColumn(column, rotation);
          expected.at((q * 259) + (row * 19)).at((t * 8) + column) =
              input.at((q * 257) + (source_row * 17)).at((t * 8) + row);
        }
      }
    }
  }
  Instruction transpose;
  transpose.Set(124, 4, 7)
      .Set(119, 5, 1)
      .Set(114, 5, 11)
      .Set(109, 5, 3)
      .Set(104, 5, 4)
      .Set(99, 5, 12)
      .Set(94, 5, 14)
      .Set(89, 5, 13)
      .Set(87, 2, rotation);
  const auto output = RunProgram(
      {.program = {SaluImm(kSaluMov, 11, 0, 17), SaluImm(kSaluMov, 12, 0, 259),
                   SaluImm(kSaluMov, 13, 0, 19), SaluImm(kSaluMov, 14, 0, 257),
                   transpose, Ret()},
       .args = {.src = 0, .dst = 2048, .rows_q = 2},
       .fbuf = {{0, input}, {2048, initial}},
       .pbuf = {},
       .out_base = 2048,
       .out_words = 393},
      947);
  EXPECT_EQ(output, expected);
}

TEST(VecProgramTest, Transpose8UsesIndependentTwelveBitStrides) {
  CheckStridedTransposeProgram(0);
  CheckStridedTransposeProgram(1);
  CheckStridedTransposeProgram(2);
}

TEST(VecProgramTest, Transpose8ChangesRotationBetweenConsecutiveInstructions) {
  std::mt19937 generator(967);
  std::vector<FbufWord> input(24);
  for (auto& word : input) {
    for (auto& value : word) {
      value = static_cast<uint32_t>(generator());
    }
  }
  auto expected = input;
  for (uint32_t word = 0; word < 24; ++word) {
    for (uint32_t t = 0; t < 4; ++t) {
      for (uint32_t column = 0; column < 8; ++column) {
        expected[word][(t * 8) + column] =
            input[word][(t * 8) + RotatedColumn(column, 1)];
      }
    }
  }
  const auto transpose = [](uint32_t src, uint32_t dst, uint32_t rotation) {
    Instruction instruction;
    instruction.Set(124, 4, 7)
        .Set(119, 5, src)
        .Set(114, 5, 11)
        .Set(109, 5, dst)
        .Set(104, 5, 4)
        .Set(99, 5, 12)
        .Set(94, 5, 12)
        .Set(89, 5, 11)
        .Set(87, 2, rotation);
    return instruction;
  };
  const auto output = RunProgram(
      {.program = {SaluImm(kSaluMov, 11, 0, 1), SaluImm(kSaluMov, 12, 0, 8),
                   SaluImm(kSaluMov, 13, 0, 1024), transpose(1, 13, 0),
                   transpose(13, 3, 1), Ret()},
       .args = {.src = 0, .dst = 2048, .rows_q = 3},
       .fbuf = {{0, input}},
       .pbuf = {},
       .out_base = 2048,
       .out_words = 24},
      967);
  EXPECT_EQ(output, expected);
}

constexpr uint32_t kRegVecCycles = 0x2C;
constexpr uint32_t kC0Pbuf = 2;
constexpr uint32_t kBPbuf = 2;
constexpr uint32_t kPaRowvec = 1;
constexpr uint32_t kPaTile = 2;

struct ChainProgram {
  uint32_t words = 1;
  uint32_t passes = 2;
  bool reduce = false;
  bool in_place = false;
};

constexpr uint32_t kChainA = 10;
constexpr uint32_t kChainB = 100;
constexpr uint32_t kChainC = 200;
constexpr uint32_t kChainImm = 0x00030001U;
constexpr uint32_t kRegChainA = 12;
constexpr uint32_t kRegChainB = 13;
constexpr uint32_t kRegChainC = 14;
constexpr uint32_t kRegChainWords = 15;
constexpr uint32_t kRegChainImm = 16;

std::vector<Instruction> ChainPrologue(uint32_t words) {
  return {SaluImm(kSaluMov, kRegOne, 0, 1),
          SaluImm(kSaluMov, kRegChainA, 0, kChainA),
          SaluImm(kSaluMov, kRegChainB, 0, kChainB),
          SaluImm(kSaluMov, kRegChainC, 0, kChainC),
          SaluImm(kSaluMov, kRegChainWords, 0, words),
          SaluImm(kSaluMov, kRegChainImm, 0, kChainImm)};
}

std::vector<Instruction> Chain(const ChainProgram& chain) {
  std::vector<Instruction> program = ChainPrologue(chain.words);
  for (uint32_t k = 0; k < chain.passes; ++k) {
    const bool even = k % 2 == 0;
    program.push_back(
        Encode(Pass{.rx = chain.in_place || even ? kRegChainA : kRegChainB,
                    .rd = chain.in_place || !even ? kRegChainA : kRegChainB,
                    .rq = kRegOne,
                    .rnw = kRegChainWords,
                    .rn8 = kRegChainWords,
                    .rimm = kRegChainImm,
                    .c0 = kC0Imm,
                    .red = chain.reduce ? 1U : 0U,
                    .ialu = 1}));
  }
  program.push_back(Ret());
  return program;
}

std::vector<FbufWord> RandomWords(std::mt19937& rng, uint32_t words) {
  std::vector<FbufWord> out(words);
  for (FbufWord& word : out) {
    for (uint32_t& element : word) {
      element = RandomFp32(rng, -3, 3);
    }
  }
  return out;
}

std::vector<FbufWord> Shifted(const std::vector<FbufWord>& words,
                              uint32_t offset) {
  std::vector<FbufWord> out = words;
  for (FbufWord& word : out) {
    for (uint32_t& element : word) {
      element += offset;
    }
  }
  return out;
}

TEST(VecProgramTest, RowMaskSkipsTheRowsAMarkingPassFoundZero) {
  constexpr uint32_t kRows = 8;
  constexpr uint32_t kWords = 2;
  constexpr uint32_t kOut = 64;
  constexpr size_t kTotal = size_t{kRows} * kWords;
  std::mt19937 generator(91);
  std::vector<FbufWord> in(kTotal);
  std::vector<FbufWord> before(kTotal);
  for (uint32_t i = 0; i < in.size(); ++i) {
    for (uint32_t e = 0; e < 32; ++e) {
      in[i].at(e) = static_cast<uint32_t>(generator()) | 1U;
      before[i].at(e) = static_cast<uint32_t>(generator());
    }
  }
  for (const size_t q : {size_t{1}, size_t{4}, size_t{5}}) {
    in[q * kWords].fill(0);
    in[(q * kWords) + 1].fill(0);
  }
  in[size_t{6} * kWords].fill(0);
  const Pass mark{.rx = kRegSrc,
                  .rq = kRegRowsQ,
                  .rnw = kRegN8,
                  .rn8 = kRegN8,
                  .wr = 0,
                  .ialu = 4,
                  .rm = 2};
  const Pass skip{.rx = kRegSrc,
                  .rd = kRegDst,
                  .rq = kRegRowsQ,
                  .rnw = kRegN8,
                  .rn8 = kRegN8,
                  .ialu = 4,
                  .rm = 1};
  const ProgramRun run{
      .program = {Encode(mark), Encode(skip), Ret()},
      .args = {.src = 0, .dst = kOut, .rows_q = kRows, .n8 = kWords},
      .fbuf = {{0, in}, {kOut, before}},
      .pbuf = {},
      .tables = nullptr,
      .out_base = kOut,
      .out_words = kRows * kWords};
  const std::vector<FbufWord> out = RunProgram(run, 92);
  for (uint32_t q = 0; q < kRows; ++q) {
    const bool skipped = q == 1 || q == 4 || q == 5;
    for (uint32_t w = 0; w < kWords; ++w) {
      const uint32_t i = (q * kWords) + w;
      ASSERT_EQ(out[i], skipped ? before[i] : in[i])
          << "row " << q << " word " << w;
    }
  }
}

void CheckSkippedRowsWaitForPendingWrites(uint32_t rows, uint32_t period) {
  constexpr uint32_t kOne = 22;
  constexpr uint32_t kTwo = 23;
  constexpr uint32_t kMid = 100;
  constexpr uint32_t kOut = 200;
  std::mt19937 generator(93 + rows);
  std::vector<FbufWord> in(rows);
  std::vector<FbufWord> before(rows);
  std::vector<bool> visible(rows);
  for (uint32_t q = 0; q < rows; ++q) {
    visible[q] = q % period == period - 1;
    for (uint32_t e = 0; e < 32; ++e) {
      in[q].at(e) =
          visible[q] ? (static_cast<uint32_t>(generator()) << 2) | 4U : 0U;
      before[q].at(e) = static_cast<uint32_t>(generator());
    }
  }
  const Pass mark{.rx = kRegSrc,
                  .rq = kRegRowsQ,
                  .rnw = kRegOne,
                  .rn8 = kRegOne,
                  .wr = 0,
                  .ialu = 4,
                  .rm = 2};
  const Pass first{.rx = kRegSrc,
                   .rd = kRegSrc2,
                   .rq = kRegRowsQ,
                   .rnw = kRegOne,
                   .rn8 = kRegOne,
                   .rimm = kOne,
                   .c0 = 4,
                   .ialu = 4,
                   .rm = 1};
  const Pass second{.rx = kRegSrc2,
                    .rd = kRegDst,
                    .rq = kRegRowsQ,
                    .rnw = kRegOne,
                    .rn8 = kRegOne,
                    .rimm = kTwo,
                    .c0 = 4,
                    .ialu = 4,
                    .rm = 1};
  const ProgramRun run{
      .program = {SaluImm(kSaluAdd, kRegOne, kRegZero, 1),
                  SaluImm(kSaluAdd, kOne, kRegZero, 1),
                  SaluImm(kSaluAdd, kTwo, kRegZero, 2), Encode(mark),
                  Encode(first), Encode(second), Ret()},
      .args = {.src = 0, .src2 = kMid, .dst = kOut, .rows_q = rows, .n8 = 1},
      .fbuf = {{0, in}, {kMid, before}, {kOut, before}},
      .pbuf = {},
      .tables = nullptr,
      .out_base = kOut,
      .out_words = rows};
  const std::vector<FbufWord> out = RunProgram(run, 94 + rows);
  for (uint32_t q = 0; q < rows; ++q) {
    FbufWord expected = before[q];
    if (visible[q]) {
      for (uint32_t e = 0; e < 32; ++e) {
        expected.at(e) = in[q].at(e) | 3U;
      }
    }
    ASSERT_EQ(out[q], expected)
        << "rows " << rows << " period " << period << " row " << q;
  }
}

TEST(VecProgramTest, SkippedRowsDoNotBypassThePendingWriteCheck) {
  for (const uint32_t rows : {6U, 10U, 12U, 18U}) {
    for (const uint32_t period : {2U, 5U, 6U}) {
      CheckSkippedRowsWaitForPendingWrites(rows, period);
    }
  }
}

TEST(VecProgramTest, PassesSkipToTheirLastQuadOrSkipEveryQuad) {
  for (const uint32_t rows : {2U, 3U, 4U, 5U, 18U}) {
    CheckSkippedRowsWaitForPendingWrites(rows, rows);
    CheckSkippedRowsWaitForPendingWrites(rows, rows + 1);
  }
}

TEST(VecProgramTest, DependentPassesOverlapTheDrain) {
  std::mt19937 rng(91);
  const std::vector<ChainProgram> chains = {
      {.words = 1, .passes = 10},
      {.words = 16, .passes = 8},
      {.words = 5, .passes = 6, .in_place = true},
      {.words = 3, .passes = 4, .in_place = true}};
  for (const ChainProgram& chain : chains) {
    const std::vector<FbufWord> x = RandomWords(rng, chain.words);
    const std::vector<FbufWord> out = RunProgram({.program = Chain(chain),
                                                  .args = {.rows_q = 1},
                                                  .fbuf = {{kChainA, x}},
                                                  .pbuf = {},
                                                  .out_base = kChainA,
                                                  .out_words = chain.words},
                                                 1 + chain.words);
    EXPECT_EQ(out, Shifted(x, chain.passes * kChainImm))
        << "words " << chain.words << " passes " << chain.passes;
  }
  const std::vector<FbufWord> x = RandomWords(rng, 16);
  std::array<uint32_t, 2> cycles{};
  for (uint32_t reduce = 0; reduce < 2; ++reduce) {
    NpuBench bench(55);
    const std::vector<FbufWord> out = RunProgramOn(
        bench,
        {.program = Chain({.words = 16, .passes = 8, .reduce = reduce != 0}),
         .args = {.rows_q = 1},
         .fbuf = {{kChainA, x}},
         .pbuf = {},
         .out_base = kChainA,
         .out_words = 16});
    EXPECT_EQ(out, Shifted(x, 8 * kChainImm));
    cycles.at(reduce) = bench.ReadReg(kRegVecCycles);
  }
  EXPECT_GE(cycles[1], cycles[0] + (8 * 30));
}

TEST(VecProgramTest, OverlappedPassesOrderPbufAndSecondOperandReads) {
  std::mt19937 rng(92);
  constexpr uint32_t kWords = 4;
  constexpr uint32_t kOne = 0x3F800000U;
  const std::vector<FbufWord> x = RandomWords(rng, kWords);
  std::vector<Instruction> program = ChainPrologue(kWords);
  program.push_back(Encode(Pass{.rx = kRegChainA,
                                .rd = kRegChainB,
                                .rq = kRegOne,
                                .rnw = kRegChainWords,
                                .rn8 = kRegChainWords,
                                .pa = kPaRowvec,
                                .b = kBPbuf}));
  program.push_back(Encode(Pass{.rx = kRegChainB,
                                .rd = kRegChainA,
                                .rq = kRegOne,
                                .rnw = kRegChainWords,
                                .rn8 = kRegChainWords,
                                .pa = kPaRowvec,
                                .c0 = kC0Pbuf,
                                .ialu = 1}));
  program.push_back(Encode(Pass{.rx = kRegChainA,
                                .rx2 = kRegChainB,
                                .rd = kRegChainC,
                                .rq = kRegOne,
                                .rnw = kRegChainWords,
                                .rn8 = kRegChainWords,
                                .x2_en = true,
                                .c0 = kC0X2,
                                .ialu = 1}));
  program.push_back(Ret());
  const std::vector<FbufWord> out = RunProgram({.program = program,
                                                .args = {.rows_q = 1},
                                                .fbuf = {{kChainA, x}},
                                                .pbuf = {Splat(kOne)},
                                                .out_base = kChainC,
                                                .out_words = kWords},
                                               93);
  std::vector<FbufWord> expected = x;
  for (FbufWord& word : expected) {
    for (uint32_t& element : word) {
      element = (2 * element) + kOne;
    }
  }
  EXPECT_EQ(out, expected);
}

TEST(VecProgramTest, PbufTilesFeedThePreAdderAndTheMultiplier) {
  std::mt19937 rng(94);
  constexpr uint32_t kWords = 4;
  const std::vector<FbufWord> x = RandomWords(rng, kWords);
  const std::vector<FbufWord> tiles = RandomWords(rng, kWords);
  std::vector<Instruction> program = ChainPrologue(kWords);
  program.push_back(SaluImm(kSaluMov, kRegChainC, 0, kChainB + kWords));
  program.push_back(Encode(Pass{.rx = kRegChainA,
                                .rd = kRegChainB,
                                .rq = kRegOne,
                                .rnw = kRegChainWords,
                                .rn8 = kRegChainWords,
                                .pa = kPaTile,
                                .c0 = kC0Pbuf}));
  program.push_back(Encode(Pass{.rx = kRegChainA,
                                .rd = kRegChainC,
                                .rq = kRegOne,
                                .rnw = kRegChainWords,
                                .rn8 = kRegChainWords,
                                .pa = kPaTile,
                                .b = kBPbuf}));
  program.push_back(Ret());
  const std::vector<FbufWord> out = RunProgram({.program = program,
                                                .args = {.rows_q = 1},
                                                .fbuf = {{kChainA, x}},
                                                .pbuf = tiles,
                                                .out_base = kChainB,
                                                .out_words = 2 * kWords},
                                               95);
  for (uint32_t w = 0; w < kWords; ++w) {
    for (uint32_t e = 0; e < 32; ++e) {
      EXPECT_EQ(out[w][e], Fp32Add(x[w][e], tiles[w][e]))
          << "sum word " << w << " element " << e;
      EXPECT_EQ(out[kWords + w][e], Fp32Mul(x[w][e], tiles[w][e]))
          << "product word " << w << " element " << e;
    }
  }
}

}  // namespace
}  // namespace hotarubi::tb
