#include "fp32.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>

#include "reference.hpp"

namespace hotarubi::iss {
namespace {

constexpr int kRandomCases = 20000;

class RandomFloats {
 public:
  explicit RandomFloats(uint32_t seed) : generator_(seed) {}

  uint32_t Make(uint32_t exponent) {
    return (Bit() << 31) | (exponent << 23) | Fraction();
  }

  uint32_t Exponent(uint32_t low, uint32_t high) {
    return std::uniform_int_distribution<uint32_t>(low, high)(generator_);
  }

  uint32_t Near(uint32_t exponent, uint32_t spread) {
    const uint32_t low = exponent > spread + 1 ? exponent - spread : 1;
    const uint32_t high = exponent + spread < 254 ? exponent + spread : 254;
    return Make(Exponent(low, high));
  }

 private:
  uint32_t Bit() {
    return std::uniform_int_distribution<uint32_t>(0, 1)(generator_);
  }
  uint32_t Fraction() {
    const uint32_t kind =
        std::uniform_int_distribution<uint32_t>(0, 7)(generator_);
    const uint32_t value =
        std::uniform_int_distribution<uint32_t>(0, 0x7FFFFF)(generator_);
    if (kind == 0) {
      return value & 0x7F0000U;
    }
    if (kind == 1) {
      return value | 0x7FFF00U;
    }
    return value;
  }

  std::mt19937 generator_;
};

TEST(Fp32Test, AddRoundsTheExactSumToNearestEven) {
  RandomFloats random(1);
  for (int i = 0; i < kRandomCases; ++i) {
    const uint32_t a = random.Make(random.Exponent(1, 254));
    const uint32_t b = random.Near((a >> 23) & 0xFFU, i % 2 == 0 ? 2 : 40);
    ASSERT_EQ(Fp32Add(a, b), reference::Add(a, b))
        << std::hex << a << " + " << b;
  }
}

TEST(Fp32Test, AddCancelsCloseOperands) {
  RandomFloats random(2);
  for (int i = 0; i < kRandomCases; ++i) {
    const uint32_t a = random.Make(random.Exponent(1, 254));
    const uint32_t b = (a ^ 0x80000000U) + static_cast<uint32_t>(i % 7) - 3;
    if (((b >> 23) & 0xFFU) == 0xFFU) {
      continue;
    }
    ASSERT_EQ(Fp32Add(a, b), reference::Add(a, b))
        << std::hex << a << " + " << b;
  }
}

TEST(Fp32Test, AddFlushesSubnormalsAndNegativeZero) {
  EXPECT_EQ(Fp32Add(0x00000001, 0x3F800000), 0x3F800000U);
  EXPECT_EQ(Fp32Add(0x807FFFFF, 0x00000000), 0x00000000U);
  EXPECT_EQ(Fp32Add(0x80000000, 0x80000000), 0x00000000U);
  EXPECT_EQ(Fp32Add(0x3FC00000, 0xBFC00000), 0x00000000U);
  EXPECT_EQ(Fp32Add(0x00800001, 0x80800000), 0x00000000U);
  EXPECT_EQ(Fp32Add(0x01000001, 0x81000000), 0x00000000U);
  EXPECT_EQ(Fp32Add(0x01000000, 0x80800000), 0x00800000U);
}

TEST(Fp32Test, AddBreaksTiesToEven) {
  EXPECT_EQ(Fp32Add(0x3F800000, 0x33800000), 0x3F800000U);
  EXPECT_EQ(Fp32Add(0x3F800001, 0x33800000), 0x3F800002U);
  EXPECT_EQ(Fp32Add(0x3F800000, 0x33800001), 0x3F800001U);
  EXPECT_EQ(Fp32Add(0x3F800000, 0xB3000000), 0x3F800000U);
  EXPECT_EQ(Fp32Add(0x3F800000, 0xB3000001), 0x3F7FFFFFU);
}

TEST(Fp32Test, AddOverflowsToInfinity) {
  EXPECT_EQ(Fp32Add(0x7F7FFFFF, 0x7F7FFFFF), 0x7F800000U);
  EXPECT_EQ(Fp32Add(0xFF7FFFFF, 0xFF7FFFFF), 0xFF800000U);
  EXPECT_EQ(Fp32Add(0x7F7FFFFF, 0x73000000), 0x7F800000U);
  EXPECT_EQ(Fp32Add(0x7F7FFFFF, 0x72FFFFFF), 0x7F7FFFFFU);
}

TEST(Fp32Test, MulRoundsTheExactProductToNearestEven) {
  RandomFloats random(3);
  for (int i = 0; i < kRandomCases; ++i) {
    const uint32_t a = random.Make(random.Exponent(1, 254));
    const uint32_t target = random.Exponent(0, 255) + 127;
    const uint32_t exponent = (a >> 23) & 0xFFU;
    const uint32_t b = random.Near(
        target > exponent ? std::min<uint32_t>(target - exponent, 254) : 1, 3);
    ASSERT_EQ(Fp32Mul(a, b), reference::Mul(a, b))
        << std::hex << a << " * " << b;
  }
}

TEST(Fp32Test, MulFlushesSubnormalsAndUnderflow) {
  EXPECT_EQ(Fp32Mul(0x00400000, 0x7F000000), 0x00000000U);
  EXPECT_EQ(Fp32Mul(0x80000000, 0x3F800000), 0x00000000U);
  EXPECT_EQ(Fp32Mul(0x00800000, 0x3F000000), 0x00000000U);
  EXPECT_EQ(Fp32Mul(0x80800000, 0x3F000000), 0x00000000U);
  EXPECT_EQ(Fp32Mul(0x00800000, 0x3F7FFFFF), 0x00000000U);
  EXPECT_EQ(Fp32Mul(0x1FFFFFFF, 0x20000001), 0x00800000U);
  EXPECT_EQ(Fp32Mul(0x9FFFFFFF, 0x20000001), 0x80800000U);
}

TEST(Fp32Test, MulOverflowsToInfinity) {
  EXPECT_EQ(Fp32Mul(0x7F000000, 0x40000000), 0x7F800000U);
  EXPECT_EQ(Fp32Mul(0x7F000000, 0xC0000000), 0xFF800000U);
  EXPECT_EQ(Fp32Mul(0x7F7FFFFF, 0x3F800001), 0x7F800000U);
}

TEST(Fp32Test, MulFractionRoundsTheScaledProduct) {
  RandomFloats random(4);
  std::mt19937 generator(5);
  std::uniform_int_distribution<uint32_t> fraction(0, 0xFFFF);
  for (int i = 0; i < kRandomCases; ++i) {
    const uint32_t s = random.Make(random.Exponent(i % 3 == 0 ? 1 : 100, 160));
    const uint32_t f = i % 5 == 0 ? (1U << (i % 16)) : fraction(generator);
    ASSERT_EQ(Fp32MulFraction(s, f), reference::MulFraction(s, f))
        << std::hex << s << " * " << f;
  }
  EXPECT_EQ(Fp32MulFraction(0x3F800000, 0), 0x00000000U);
  EXPECT_EQ(Fp32MulFraction(0x00400000, 0x8000), 0x00000000U);
  EXPECT_EQ(Fp32MulFraction(0x00800000, 0x8000), 0x00000000U);
  EXPECT_EQ(Fp32MulFraction(0xBF800000, 0x8000), 0xBF000000U);
}

}  // namespace
}  // namespace hotarubi::iss
