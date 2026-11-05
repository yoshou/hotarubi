#include "tensors.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "hotarubi/exact.hpp"

namespace hotarubi::compiler {
namespace {

TEST(TensorsTest, WbufWordsPackColumnsAsNibblesThenScales) {
  Mxfp4Matrix matrix{.rows = 8,
                     .cols = 32,
                     .code = std::vector<uint8_t>(256),
                     .e = {0, 1, 2, 3, 4, 5, 6, -7}};
  for (uint32_t i = 0; i < 256; ++i) {
    matrix.code[i] = static_cast<uint8_t>((i * 7) % 16);
  }
  const std::vector<std::byte> bytes = PackWbuf(matrix);
  ASSERT_EQ(bytes.size(), kWordWbufBytes);
  for (uint32_t c = 0; c < 8; ++c) {
    for (uint32_t k = 0; k < 32; ++k) {
      const uint32_t bit = (c * 128) + (4 * k);
      const uint32_t nibble =
          (static_cast<uint32_t>(bytes.at(bit / 8)) >> (bit % 8)) & 0xFU;
      EXPECT_EQ(nibble, matrix.code[(c * 32) + k]);
    }
    EXPECT_EQ(static_cast<int32_t>(bytes.at(128 + c)), matrix.e[c] + 127);
  }
}

TEST(TensorsTest, TablesFollowTheirDefinition) {
  const std::vector<uint64_t> silu = MakeTbuf(Activation::kSilu);
  const std::vector<uint64_t> gelu = MakeTbuf(Activation::kGelu);
  ASSERT_EQ(silu.size(), kTbufWords);
  const auto v = [](uint64_t entry) { return static_cast<uint32_t>(entry); };
  const auto s = [](uint64_t entry) {
    return static_cast<uint32_t>(entry >> 32);
  };
  EXPECT_EQ(v(silu[0]), 0x3F800000U);
  EXPECT_EQ(v(silu[256]), 0x3FB504F3U);
  EXPECT_EQ(silu[100], gelu[100]);
  EXPECT_EQ(v(silu[512]), 0x3F000000U);
  EXPECT_EQ(v(gelu[512]), 0x3F000000U);
  const exact::Real one = exact::Real::Integer(1);
  EXPECT_EQ(v(silu[512 + 32]),
            (one / (one + exact::Exp(one.Negated()))).ToBinary32());
  for (uint32_t i = 0; i + 1 < kTbufWords; ++i) {
    if (i == 511) {
      continue;
    }
    EXPECT_EQ(s(silu[i]), (exact::Real::Binary32(v(silu[i + 1])) -
                           exact::Real::Binary32(v(silu[i])))
                              .ToBinary32())
        << i;
  }
  EXPECT_EQ(PackTbuf(silu).size(), kTbufWords * kWordTbufBytes);
}

}  // namespace
}  // namespace hotarubi::compiler
