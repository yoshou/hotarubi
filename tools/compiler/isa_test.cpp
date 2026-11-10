#include "isa.hpp"

#include <gtest/gtest.h>

#include <cstdint>

namespace hotarubi::compiler {
namespace {

TEST(IsaTest, DmaLoadPlacesEveryFieldAsSpecified) {
  const Instruction instruction = EncodeDmaLoad({.ddr_addr = 0xA345670U,
                                                 .mem = Mem::kCbuf,
                                                 .onchip = 0x1BCD,
                                                 .rows = 0x7FF,
                                                 .row_words = 0x123,
                                                 .ddr_stride = 0x65432,
                                                 .kind = DmaKind::kT32Split8,
                                                 .tok = 0x25,
                                                 .acc = true,
                                                 .flush = false},
                                                kWaitVec | kWaitDma);
  EXPECT_EQ(instruction.Get(120, 8), 1U);
  EXPECT_EQ(instruction.Get(116, 4), 0x5U);
  EXPECT_EQ(instruction.Get(88, 28), 0xA345670U);
  EXPECT_EQ(instruction.Get(85, 3), 5U);
  EXPECT_EQ(instruction.Get(72, 13), 0x1BCDU);
  EXPECT_EQ(instruction.Get(60, 12), 0x7FFU);
  EXPECT_EQ(instruction.Get(48, 12), 0x123U);
  EXPECT_EQ(instruction.Get(28, 20), 0x65432U);
  EXPECT_EQ(instruction.Get(25, 3), 4U);
  EXPECT_EQ(instruction.Get(19, 6), 0x25U);
  EXPECT_EQ(instruction.Get(18, 1), 1U);
  EXPECT_EQ(instruction.Get(17, 1), 0U);
  EXPECT_EQ(instruction.Get(0, 17), 0U);
}

TEST(IsaTest, DmaStoreHasItsOpcode) {
  const Instruction instruction = EncodeDmaStore(
      {.ddr_addr = 0x40, .mem = Mem::kAbuf, .kind = DmaKind::kRow}, 0);
  EXPECT_EQ(instruction.Get(120, 8), 2U);
  EXPECT_EQ(instruction.Get(25, 3), 1U);
  EXPECT_EQ(instruction.Get(85, 3), 0U);
}

TEST(IsaTest, GemmPlacesEveryFieldAsSpecified) {
  const Instruction instruction = EncodeGemm({.a_base = 0xFEDCB,
                                              .a_stride = 0x3FF,
                                              .kb = 0x2AA,
                                              .m = 0x3E0,
                                              .w_base = 0x12345,
                                              .w_stride = 0x155,
                                              .n8 = 0xFF,
                                              .c_base = 0xA5A5A,
                                              .wmode = WeightMode::kW8Lo,
                                              .acc = true},
                                             kWaitGemm);
  EXPECT_EQ(instruction.Get(120, 8), 4U);
  EXPECT_EQ(instruction.Get(116, 4), 0x2U);
  EXPECT_EQ(instruction.Get(92, 20), 0xFEDCBU);
  EXPECT_EQ(instruction.Get(82, 10), 0x3FFU);
  EXPECT_EQ(instruction.Get(72, 10), 0x2AAU);
  EXPECT_EQ(instruction.Get(62, 10), 0x3E0U);
  EXPECT_EQ(instruction.Get(42, 20), 0x12345U);
  EXPECT_EQ(instruction.Get(32, 10), 0x155U);
  EXPECT_EQ(instruction.Get(24, 8), 0xFFU);
  EXPECT_EQ(instruction.Get(4, 20), 0xA5A5AU);
  EXPECT_EQ(instruction.Get(2, 2), 2U);
  EXPECT_EQ(instruction.Get(1, 1), 1U);
  EXPECT_EQ(instruction.Get(0, 1), 0U);
}

TEST(IsaTest, VecPlacesEveryFieldAsSpecified) {
  const Instruction instruction = EncodeVec({.entry = 0xC7,
                                             .src = 0xABC,
                                             .src2 = 0x123,
                                             .dst = 0x1F00,
                                             .rows_q = 0x1AB,
                                             .n8 = 0xCD,
                                             .p_base = 0x1EF,
                                             .aux = 0x456,
                                             .imm = 0xDEADBEEFU,
                                             .q = true},
                                            0);
  EXPECT_EQ(instruction.Get(120, 8), 5U);
  EXPECT_EQ(instruction.Get(108, 8), 0xC7U);
  EXPECT_EQ(instruction.Get(107, 1), 1U);
  EXPECT_EQ(instruction.Get(95, 12), 0xABCU);
  EXPECT_EQ(instruction.Get(83, 12), 0x123U);
  EXPECT_EQ(instruction.Get(70, 13), 0x1F00U);
  EXPECT_EQ(instruction.Get(61, 9), 0x1ABU);
  EXPECT_EQ(instruction.Get(53, 8), 0xCDU);
  EXPECT_EQ(instruction.Get(52, 1), 0U);
  EXPECT_EQ(instruction.Get(43, 9), 0x1EFU);
  EXPECT_EQ(instruction.Get(32, 11), 0x456U);
  EXPECT_EQ(instruction.Get(0, 32), 0xDEADBEEFU);
}

TEST(IsaTest, ControlInstructionsCarryTheirOperands) {
  const Instruction signal = EncodeSignal(0xA7, 0x89ABCDEFU, kWaitAll);
  EXPECT_EQ(signal.Get(120, 8), 6U);
  EXPECT_EQ(signal.Get(116, 4), 0x7U);
  EXPECT_EQ(signal.Get(104, 8), 0xA7U);
  EXPECT_EQ(signal.Get(32, 64), 0U);
  EXPECT_EQ(signal.Get(96, 8), 0U);
  EXPECT_EQ(signal.Get(0, 32), 0x89ABCDEFU);
  const Instruction halt = EncodeHalt(kWaitAll);
  EXPECT_EQ(halt.Get(120, 8), 7U);
  EXPECT_EQ(halt.Get(0, 64), 0U);
  EXPECT_EQ(halt.Get(64, 48), 0U);
  const Instruction wait = EncodeWaitHost(42, 0);
  EXPECT_EQ(wait.Get(120, 8), 8U);
  EXPECT_EQ(wait.Get(0, 32), 42U);
}

TEST(IsaTest, HexPrintsTheMostSignificantWordFirst) {
  Instruction instruction;
  instruction.words = {0x00000004U, 0x00000003U, 0x00000002U, 0x07000001U};
  EXPECT_EQ(instruction.Hex(), "07000001000000020000000300000004");
}

TEST(IsaDeathTest, RejectsAValueWiderThanItsField) {
  EXPECT_DEATH(EncodeGemm({.n8 = 256}, 0), "does not fit");
}

}  // namespace
}  // namespace hotarubi::compiler
