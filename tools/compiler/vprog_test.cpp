#include "vprog.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "isa.hpp"

namespace hotarubi::compiler::vprog {
namespace {

TEST(VprogTest, PassPlacesEveryFieldAsSpecified) {
  const Instruction word = EncodePass({.rx = 1,
                                       .rx2 = 2,
                                       .rd = 3,
                                       .rq = 4,
                                       .rnw = 5,
                                       .rn8 = 6,
                                       .rpb = 7,
                                       .rimm = 8,
                                       .rimm2 = 9,
                                       .x2_en = true,
                                       .wx = WordMap::kX1,
                                       .wx2 = WordMap::kX2,
                                       .wd = WordMap::kS4,
                                       .pa = Pbuf::kTile,
                                       .pa_at_mul = true,
                                       .c0 = C0::kImm,
                                       .b = B::kLog2e,
                                       .nl = Nl::kTab01,
                                       .c1 = C1::kSignfix,
                                       .neg_c1 = true,
                                       .mask_en = true,
                                       .red = Reduce::kMax,
                                       .wr = Write::kFinal,
                                       .sc = RowScalar::kMax,
                                       .px = Perm::kT4b,
                                       .pcol = 6,
                                       .cmp = Cmp::kGe,
                                       .cmp_src = CmpSrc::kX2Imm2,
                                       .cmp_int = true,
                                       .sel = Sel::kRs1,
                                       .ialu = Ialu::kRsub,
                                       .dst_s1 = true,
                                       .c0_pred = true,
                                       .nodrain = true,
                                       .x_s1 = true});
  EXPECT_EQ(word.Get(124, 4), 0U);
  EXPECT_EQ(word.Get(119, 5), 1U);
  EXPECT_EQ(word.Get(114, 5), 2U);
  EXPECT_EQ(word.Get(109, 5), 3U);
  EXPECT_EQ(word.Get(104, 5), 4U);
  EXPECT_EQ(word.Get(99, 5), 5U);
  EXPECT_EQ(word.Get(94, 5), 6U);
  EXPECT_EQ(word.Get(89, 5), 7U);
  EXPECT_EQ(word.Get(84, 5), 8U);
  EXPECT_EQ(word.Get(79, 5), 9U);
  EXPECT_EQ(word.Get(78, 1), 1U);
  EXPECT_EQ(word.Get(76, 2), 1U);
  EXPECT_EQ(word.Get(74, 2), 2U);
  EXPECT_EQ(word.Get(72, 2), 3U);
  EXPECT_EQ(word.Get(70, 2), 2U);
  EXPECT_EQ(word.Get(69, 1), 1U);
  EXPECT_EQ(word.Get(66, 3), 4U);
  EXPECT_EQ(word.Get(63, 3), 6U);
  EXPECT_EQ(word.Get(61, 2), 3U);
  EXPECT_EQ(word.Get(59, 2), 3U);
  EXPECT_EQ(word.Get(58, 1), 1U);
  EXPECT_EQ(word.Get(57, 1), 1U);
  EXPECT_EQ(word.Get(55, 2), 2U);
  EXPECT_EQ(word.Get(53, 2), 3U);
  EXPECT_EQ(word.Get(50, 3), 6U);
  EXPECT_EQ(word.Get(47, 3), 5U);
  EXPECT_EQ(word.Get(44, 3), 0U);
  EXPECT_EQ(word.Get(41, 3), 6U);
  EXPECT_EQ(word.Get(38, 3), 6U);
  EXPECT_EQ(word.Get(36, 2), 3U);
  EXPECT_EQ(word.Get(35, 1), 1U);
  EXPECT_EQ(word.Get(32, 3), 6U);
  EXPECT_EQ(word.Get(29, 3), 7U);
  EXPECT_EQ(word.Get(28, 1), 1U);
  EXPECT_EQ(word.Get(27, 1), 1U);
  EXPECT_EQ(word.Get(26, 1), 1U);
  EXPECT_EQ(word.Get(25, 1), 1U);
  EXPECT_EQ(word.Get(0, 25), 0U);
}

TEST(VprogTest, ScalarInstructionsPlaceTheirFieldsAsSpecified) {
  const Instruction alu = EncodeSalu(Salu::kSra, 31, 17, 9, 0x80000001U);
  EXPECT_EQ(alu.Get(124, 4), 1U);
  EXPECT_EQ(alu.Get(119, 5), 7U);
  EXPECT_EQ(alu.Get(114, 5), 31U);
  EXPECT_EQ(alu.Get(109, 5), 17U);
  EXPECT_EQ(alu.Get(104, 5), 9U);
  EXPECT_EQ(alu.Get(103, 1), 1U);
  EXPECT_EQ(alu.Get(0, 32), 0x80000001U);
  EXPECT_EQ(EncodeSalu(Salu::kAdd, 1, 2, 3, std::nullopt).Get(103, 1), 0U);
  const Instruction smem = EncodeSmem(Smem::kRsld, {.rd = 4,
                                                    .ra = 5,
                                                    .rb = 6,
                                                    .col = 19,
                                                    .which = true,
                                                    .neg = true,
                                                    .imm = 0xFFFFFFF0U});
  EXPECT_EQ(smem.Get(124, 4), 2U);
  EXPECT_EQ(smem.Get(121, 3), 2U);
  EXPECT_EQ(smem.Get(116, 5), 4U);
  EXPECT_EQ(smem.Get(111, 5), 5U);
  EXPECT_EQ(smem.Get(106, 5), 6U);
  EXPECT_EQ(smem.Get(105, 1), 1U);
  EXPECT_EQ(smem.Get(100, 5), 19U);
  EXPECT_EQ(smem.Get(99, 1), 1U);
  EXPECT_EQ(smem.Get(98, 1), 1U);
  EXPECT_EQ(smem.Get(0, 32), 0xFFFFFFF0U);
  const Instruction branch = EncodeBranch(Cond::kNz, 13, 700);
  EXPECT_EQ(branch.Get(124, 4), 3U);
  EXPECT_EQ(branch.Get(122, 2), 2U);
  EXPECT_EQ(branch.Get(117, 5), 13U);
  EXPECT_EQ(branch.Get(0, 16), 700U);
  EXPECT_EQ(EncodeRet().Get(124, 4), 4U);
  EXPECT_EQ(EncodeNop().Get(124, 4), 5U);
}

TEST(VprogTest, AssemblerResolvesLabelsAgainstTheLoadAddress) {
  Assembler program;
  program.LoadImmediate(1, 3);
  program.Label("loop");
  program.Alu(Salu::kSub, 1, 1, 0, 1);
  program.Branch(Cond::kNz, 1, "loop");
  program.Ret();
  const std::vector<Instruction> words = program.Words(300);
  ASSERT_EQ(words.size(), 4U);
  EXPECT_EQ(words[2].Get(0, 16), 301U);
  EXPECT_EQ(program.Size(), 4U);
}

TEST(VprogTest, AssemblerEnablesX2AndRawCopies) {
  Assembler raw(true);
  raw.Emit({.rx = 1, .rx2 = 2, .rd = 3, .wr = Write::kFbuf});
  raw.Emit({.rx = 1, .rd = 3, .c0 = C0::kImm, .wr = Write::kFbuf});
  const std::vector<Instruction> words = raw.Words();
  EXPECT_EQ(words[0].Get(78, 1), 1U);
  EXPECT_EQ(words[0].Get(29, 3), static_cast<uint64_t>(Ialu::kOr));
  EXPECT_EQ(words[1].Get(78, 1), 0U);
  EXPECT_EQ(words[1].Get(29, 3), 0U);
  Assembler plain;
  plain.Emit({.rx = 1, .rd = 3, .wr = Write::kFbuf});
  EXPECT_EQ(plain.Words()[0].Get(29, 3), 0U);
}

TEST(VprogTest, PassPlacesTheContextAndRowMask) {
  const Instruction word =
      EncodePass({.wr = Write::kNone, .ctx = 1, .rm = RowMask::kSet});
  EXPECT_EQ(word.Get(23, 2), 1U);
  EXPECT_EQ(word.Get(21, 2), 2U);
  EXPECT_EQ(word.Get(0, 21), 0U);
}

TEST(VprogTest, AssemblerSkipsMaskedRowsInsideARegion) {
  Assembler a;
  a.Emit({.rx = 1, .rd = 3, .wr = Write::kFbuf});
  a.SkipMaskedRows(true);
  a.Emit({.rx = 1, .rd = 3, .wr = Write::kFbuf});
  a.Emit({.rx = 1, .wr = Write::kNone, .rm = RowMask::kSet});
  a.SkipMaskedRows(false);
  a.Emit({.rx = 1, .rd = 3, .wr = Write::kFbuf});
  const std::vector<Instruction> words = a.Words();
  EXPECT_EQ(words[0].Get(21, 2), 0U);
  EXPECT_EQ(words[1].Get(21, 2), static_cast<uint64_t>(RowMask::kSkip));
  EXPECT_EQ(words[2].Get(21, 2), static_cast<uint64_t>(RowMask::kSet));
  EXPECT_EQ(words[3].Get(21, 2), 0U);
}

TEST(VprogTest, Transpose8PlacesItsRegistersAndPackBit) {
  Assembler a(true);
  a.Transpose8(11, 3, 4, 5, 12, 13, 4, TransposePack::kZip4);
  a.Transpose8(1, 2, 3, 4, 5, 6, 7);
  const auto words = a.Words(0);
  ASSERT_EQ(words.size(), 2U);
  EXPECT_EQ(words[0].Get(124, 4), 7U);
  EXPECT_EQ(words[0].Get(119, 5), 11U);
  EXPECT_EQ(words[0].Get(114, 5), 12U);
  EXPECT_EQ(words[0].Get(109, 5), 3U);
  EXPECT_EQ(words[0].Get(104, 5), 4U);
  EXPECT_EQ(words[0].Get(99, 5), 13U);
  EXPECT_EQ(words[0].Get(94, 5), 5U);
  EXPECT_EQ(words[0].Get(89, 5), 4U);
  EXPECT_EQ(words[0].Get(88, 1), 1U);
  EXPECT_EQ(words[1].Get(88, 1), 0U);
  EXPECT_EQ(words[1].Get(0, 64), 0U);
  EXPECT_EQ(words[1].Get(64, 24), 0U);
}

TEST(VprogDeathTest, RejectsAnUndefinedLabel) {
  Assembler program;
  program.Branch(Cond::kAlways, 0, "missing");
  EXPECT_DEATH(static_cast<void>(program.Words()), "undefined label");
}

}  // namespace
}  // namespace hotarubi::compiler::vprog
