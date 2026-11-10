#include "vprog.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "isa.hpp"

namespace hotarubi::compiler::vprog {
namespace {

TEST(VprogTest, PassPlacesEveryFieldAsSpecified) {
  const Instruction word = EncodePass(
      {.rx = 1,
       .rx2 = 2,
       .rd = 3,
       .rq = 4,
       .rnw = 5,
       .rn8 = 6,
       .rpb = 7,
       .rimm = 8,
       .rimm2 = 9,
       .x2_en = true,
       .map = {.group = 2,
               .stride = 3,
               .x = Map::kGroup,
               .x2 = Map::kUpper,
               .dst = Map::kGroup},
       .pa = Pbuf::kTile,
       .c0 = C0::kImm,
       .b = B::kPre,
       .nl = Nl::kTab01,
       .post = Post::kSignfix,
       .post_x2 = true,
       .mask_en = true,
       .red = Reduce::kMax,
       .wr = Write::kFinal,
       .sc = {.fn = ScalarFn::kRsqrt, .rs0 = true, .rn = true, .eps = true},
       .px = 0xABCDE,
       .cmp = Cmp::kGe,
       .cmp_left = CmpLeft::kPosition,
       .cmp_right_x2 = true,
       .cmp_int = true,
       .sel = Sel::kRs1,
       .ialu = Ialu::kRsub,
       .dst_s1 = true,
       .c0_pred = true,
       .x_s1 = true,
       .ctx = 1,
       .rm = RowMask::kSet});
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
  EXPECT_EQ(word.Get(76, 2), 2U);
  EXPECT_EQ(word.Get(74, 2), 3U);
  EXPECT_EQ(word.Get(72, 2), 1U);
  EXPECT_EQ(word.Get(70, 2), 2U);
  EXPECT_EQ(word.Get(68, 2), 1U);
  EXPECT_EQ(word.Get(66, 2), 2U);
  EXPECT_EQ(word.Get(63, 3), 4U);
  EXPECT_EQ(word.Get(60, 3), 5U);
  EXPECT_EQ(word.Get(58, 2), 3U);
  EXPECT_EQ(word.Get(56, 2), 3U);
  EXPECT_EQ(word.Get(55, 1), 1U);
  EXPECT_EQ(word.Get(54, 1), 1U);
  EXPECT_EQ(word.Get(52, 2), 2U);
  EXPECT_EQ(word.Get(50, 2), 3U);
  EXPECT_EQ(word.Get(47, 3), 4U);
  EXPECT_EQ(word.Get(46, 1), 1U);
  EXPECT_EQ(word.Get(45, 1), 1U);
  EXPECT_EQ(word.Get(44, 1), 1U);
  EXPECT_EQ(word.Get(24, 20), 0xABCDEU);
  EXPECT_EQ(word.Get(21, 3), 6U);
  EXPECT_EQ(word.Get(19, 2), 3U);
  EXPECT_EQ(word.Get(18, 1), 1U);
  EXPECT_EQ(word.Get(17, 1), 1U);
  EXPECT_EQ(word.Get(14, 3), 5U);
  EXPECT_EQ(word.Get(11, 3), 7U);
  EXPECT_EQ(word.Get(10, 1), 1U);
  EXPECT_EQ(word.Get(9, 1), 1U);
  EXPECT_EQ(word.Get(8, 1), 1U);
  EXPECT_EQ(word.Get(6, 2), 1U);
  EXPECT_EQ(word.Get(4, 2), 2U);
  EXPECT_EQ(word.Get(0, 4), 0U);
}

TEST(VprogTest, PermutationHelpersSelectTheSourceBits) {
  EXPECT_EQ(Bcast(5), (kPermOne << 8) | (kPermZero << 4) | kPermOne);
  EXPECT_EQ(TokenXor(1), (kPermSame | kPermInvert) << 12);
  EXPECT_EQ(TokenXor(3), ((kPermSame | kPermInvert) << 12) |
                             ((kPermSame | kPermInvert) << 16));
  EXPECT_EQ(kTokensToLanes,
            (4U << 16) | (5U << 12) | (1U << 8) | (2U << 4) | 3U);
  EXPECT_EQ(kTokenPairsToLanes, (2U << 16) | (3U << 12) | (1U << 8));
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
  const Instruction smem = EncodeSmem(Smem::kRsmov, {.rd = 4,
                                                     .ra = 5,
                                                     .rb = 6,
                                                     .col = 19,
                                                     .which = true,
                                                     .ctx = 1,
                                                     .imm = 0xFFFFFFF0U});
  EXPECT_EQ(smem.Get(124, 4), 2U);
  EXPECT_EQ(smem.Get(121, 3), 1U);
  EXPECT_EQ(smem.Get(116, 5), 4U);
  EXPECT_EQ(smem.Get(111, 5), 5U);
  EXPECT_EQ(smem.Get(106, 5), 6U);
  EXPECT_EQ(smem.Get(105, 1), 1U);
  EXPECT_EQ(smem.Get(100, 5), 19U);
  EXPECT_EQ(smem.Get(99, 1), 1U);
  EXPECT_EQ(smem.Get(98, 1), 0U);
  EXPECT_EQ(smem.Get(96, 2), 1U);
  EXPECT_EQ(smem.Get(0, 32), 0xFFFFFFF0U);
  EXPECT_EQ(EncodeSmem(Smem::kRn, {}).Get(121, 3), 2U);
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
  EXPECT_EQ(words[0].Get(11, 3), static_cast<uint64_t>(Ialu::kOr));
  EXPECT_EQ(words[1].Get(78, 1), 0U);
  EXPECT_EQ(words[1].Get(11, 3), 0U);
  Assembler plain;
  plain.Emit({.rx = 1, .rd = 3, .wr = Write::kFbuf});
  EXPECT_EQ(plain.Words()[0].Get(11, 3), 0U);
}

TEST(VprogTest, PassPlacesTheContextAndRowMask) {
  const Instruction word =
      EncodePass({.wr = Write::kNone, .ctx = 1, .rm = RowMask::kSet});
  EXPECT_EQ(word.Get(6, 2), 1U);
  EXPECT_EQ(word.Get(4, 2), 2U);
  EXPECT_EQ(word.Get(0, 4), 0U);
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
  EXPECT_EQ(words[0].Get(4, 2), 0U);
  EXPECT_EQ(words[1].Get(4, 2), static_cast<uint64_t>(RowMask::kSkip));
  EXPECT_EQ(words[2].Get(4, 2), static_cast<uint64_t>(RowMask::kSet));
  EXPECT_EQ(words[3].Get(4, 2), 0U);
}

TEST(VprogTest, Transpose8PlacesItsRegistersAndRotation) {
  Assembler a(true);
  a.Transpose8(11, 3, 4, 5, 12, 13, 4, 2);
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
  EXPECT_EQ(words[0].Get(87, 2), 2U);
  EXPECT_EQ(words[1].Get(87, 2), 0U);
  EXPECT_EQ(words[1].Get(0, 64), 0U);
  EXPECT_EQ(words[1].Get(64, 23), 0U);
}

TEST(VprogDeathTest, RejectsAnUndefinedLabel) {
  Assembler program;
  program.Branch(Cond::kAlways, 0, "missing");
  EXPECT_DEATH(static_cast<void>(program.Words()), "undefined label");
}

TEST(VprogDeathTest, RejectsAWordMapStrideBelowItsGroup) {
  Assembler program;
  program.Emit({.map = {.group = 1, .stride = 1, .x = Map::kGroup}});
  program.Emit({.map = {.group = 1, .stride = 2, .x2 = Map::kUpper}});
  program.Emit({.map = {.group = 3, .stride = 0}});
  EXPECT_EQ(program.Size(), 3U);
  EXPECT_DEATH(
      program.Emit({.map = {.group = 2, .stride = 1, .x = Map::kGroup}}),
      "word map stride");
  EXPECT_DEATH(
      program.Emit({.map = {.group = 1, .stride = 1, .dst = Map::kUpper}}),
      "word map stride");
}

}  // namespace
}  // namespace hotarubi::compiler::vprog
