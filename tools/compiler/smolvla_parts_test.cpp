#include "smolvla_parts.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "hotarubi/exact.hpp"
#include "tensors.hpp"

namespace hotarubi::compiler::smolvla {
namespace {

constexpr uint32_t kOne = 0x3F800000U;
constexpr uint32_t kCosOne = 0x3F0A5140U;
constexpr uint32_t kSinOne = 0x3F576AA4U;

uint32_t TableEntry(const std::vector<std::byte>& tables, uint32_t rows,
                    uint32_t table, uint32_t row, uint32_t pair) {
  const size_t word = ((static_cast<size_t>(row / 4) * 4) + (pair / 8)) +
                      (static_cast<size_t>(table) * (rows / 4) * 4);
  const size_t at =
      (word * kWordFbufBytes) + ((((size_t{row} % 4) * 8) + (pair % 8)) * 4);
  uint32_t bits = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    bits |= static_cast<uint32_t>(tables.at(at + i)) << (8 * i);
  }
  return bits;
}

TEST(SmolVlaPartsTest, PrefixHoldsImagesLanguageAndState) {
  const Dims one{.cameras = 1, .lang = 12, .steps = 10};
  EXPECT_EQ(one.Image(), 64U);
  EXPECT_EQ(one.Context(), 76U);
  EXPECT_EQ(one.Prefix(), 77U);
  EXPECT_EQ(one.Tokens(), 128U);
  const Dims two{.cameras = 2, .lang = 48, .steps = 10};
  EXPECT_EQ(two.Image(), 128U);
  EXPECT_EQ(two.Prefix(), 177U);
  EXPECT_EQ(two.Tokens(), 192U);
}

TEST(SmolVlaPartsTest, DecimalConstantsRoundThroughBinary64) {
  const exact::Real x = exact::Real::Integer(1) +
                        exact::Real::Ratio(1, int64_t{1} << 24) +
                        exact::Real::Ratio(1, int64_t{1} << 60);
  EXPECT_EQ(x.ToBinary32(), kOne + 1);
  EXPECT_EQ(Fp32OfFp64(x), kOne);
  EXPECT_EQ(Fp32OfFp64(exact::Real::Ratio(3, 4)), 0x3F400000U);
}

TEST(SmolVlaPartsTest, RopeTablesHoldCosinesThenSines) {
  const std::vector<uint32_t> positions = {0, 1, 1, 0};
  const std::vector<std::byte> tables = RopeTables(positions);
  ASSERT_EQ(tables.size(), 2U * 4U * kWordFbufBytes);
  for (uint32_t pair = 0; pair < kHeadPairs; ++pair) {
    EXPECT_EQ(TableEntry(tables, 4, 0, 0, pair), kOne);
    EXPECT_EQ(TableEntry(tables, 4, 1, 0, pair), 0U);
    EXPECT_EQ(TableEntry(tables, 4, 0, 1, pair),
              TableEntry(tables, 4, 0, 2, pair));
    EXPECT_EQ(TableEntry(tables, 4, 1, 1, pair),
              TableEntry(tables, 4, 1, 2, pair));
  }
  EXPECT_EQ(TableEntry(tables, 4, 0, 1, 0), kCosOne);
  EXPECT_EQ(TableEntry(tables, 4, 1, 1, 0), kSinOne);
}

}  // namespace
}  // namespace hotarubi::compiler::smolvla
