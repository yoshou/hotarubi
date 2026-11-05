#include "hotarubi/formats.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace hotarubi {
namespace {

uint32_t BitsAt(const std::vector<std::byte>& bytes, size_t offset) {
  uint32_t bits = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    bits |= static_cast<uint32_t>(bytes.at(offset + i)) << (8 * i);
  }
  return bits;
}

TEST(FormatsTest, FbufWordsHoldFourTokensOfEightColumns) {
  Matrix matrix{.rows = 8, .cols = 16, .bits = std::vector<uint32_t>(128)};
  for (uint32_t i = 0; i < 128; ++i) {
    matrix.bits[i] = 0x3F800000U + i;
  }
  const std::vector<std::byte> bytes = PackFbuf(matrix);
  ASSERT_EQ(bytes.size(), 4U * kWordFbufBytes);
  for (uint32_t r = 0; r < 8; ++r) {
    for (uint32_t c = 0; c < 16; ++c) {
      const size_t word = ((r / 4) * 2) + (c / 8);
      const size_t element = ((r % 4) * 8) + (c % 8);
      EXPECT_EQ(BitsAt(bytes, (word * kWordFbufBytes) + (element * 4)),
                matrix.At(r, c));
    }
  }
}

TEST(FormatsTest, AbufWordsHoldFourTokensOfABlockAndTheirScales) {
  Mxint8Matrix matrix{.rows = 4,
                      .cols = 64,
                      .q = std::vector<int8_t>(256),
                      .e = {-3, 5, 0, 1, 2, -1, 7, -127}};
  for (uint32_t i = 0; i < 256; ++i) {
    matrix.q[i] = static_cast<int8_t>(static_cast<int32_t>(i % 200) - 100);
  }
  const std::vector<std::byte> bytes = PackAbuf(matrix);
  ASSERT_EQ(bytes.size(), 2U * kWordAbufBytes);
  for (uint32_t kb = 0; kb < 2; ++kb) {
    for (uint32_t j = 0; j < 4; ++j) {
      for (uint32_t k = 0; k < 32; ++k) {
        EXPECT_EQ(
            static_cast<int8_t>(bytes.at((kb * kWordAbufBytes) + (j * 32) + k)),
            matrix.q[(j * 64) + (kb * 32) + k]);
      }
      EXPECT_EQ(static_cast<int32_t>(bytes.at((kb * kWordAbufBytes) + 128 + j)),
                matrix.e[(j * 2) + kb] + 127);
    }
    for (uint32_t b = 132; b < kWordAbufBytes; ++b) {
      EXPECT_EQ(bytes.at((kb * kWordAbufBytes) + b), std::byte{0});
    }
  }
}

TEST(FormatsTest, Mxint8UsesTheExponentFieldAndRoundsHalfToEven) {
  Matrix matrix{.rows = 2, .cols = 40, .bits = std::vector<uint32_t>(80, 0)};
  matrix.bits[0] = 0x3F7FFFFFU;
  matrix.bits[1] = 0x3F010000U;
  matrix.bits[2] = 0xBE810000U;
  matrix.bits[3] = 0x3A800000U;
  matrix.bits[4] = 0x3F030000U;
  matrix.bits[5] = 0x80000000U;
  matrix.bits[33] = 0x40400000U;
  const Mxint8Matrix encoded = EncodeMxint8(matrix);
  ASSERT_EQ(encoded.cols, 64U);
  EXPECT_EQ(encoded.e[0], -1);
  EXPECT_EQ(encoded.q[0], 127);
  EXPECT_EQ(encoded.q[1], 64);
  EXPECT_EQ(encoded.q[2], -32);
  EXPECT_EQ(encoded.q[3], 0);
  EXPECT_EQ(encoded.q[4], 66);
  EXPECT_EQ(encoded.q[5], 0);
  EXPECT_EQ(encoded.e[1], 1);
  EXPECT_EQ(encoded.q[33], 96);
  EXPECT_EQ(encoded.e[2], 0);
  EXPECT_EQ(encoded.e[3], 0);
  EXPECT_EQ(encoded.q[64 + 5], 0);
  EXPECT_EQ(PadToBlock(720), 736U);
}

TEST(FormatsTest, UnpackFbufInvertsPackFbuf) {
  Matrix matrix{.rows = 8, .cols = 16, .bits = std::vector<uint32_t>(128)};
  for (uint32_t i = 0; i < 128; ++i) {
    matrix.bits[i] = 0x40000000U + (i * 3);
  }
  const Matrix unpacked = UnpackFbuf(PackFbuf(matrix), 8, 16);
  EXPECT_EQ(unpacked.rows, 8U);
  EXPECT_EQ(unpacked.cols, 16U);
  EXPECT_EQ(unpacked.bits, matrix.bits);
}

TEST(FormatsDeathTest, UnpackFbufRejectsAMismatchedSize) {
  const std::vector<std::byte> bytes(100);
  EXPECT_DEATH(static_cast<void>(UnpackFbuf(bytes, 8, 16)),
               "are not an FBUF matrix");
}

}  // namespace
}  // namespace hotarubi
