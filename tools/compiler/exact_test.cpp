#include "exact.hpp"

#include <gtest/gtest.h>

#include <cstdint>

namespace hotarubi::compiler::exact {
namespace {

TEST(ExactTest, ConstantsRoundToTheirKnownValues) {
  EXPECT_EQ(Pi().ToBinary64(), 0x400921FB54442D18U);
  EXPECT_EQ(Pi().ToBinary32(), 0x40490FDBU);
  EXPECT_EQ(Ln2().ToBinary64(), 0x3FE62E42FEFA39EFU);
  EXPECT_EQ(Ln2().ToBinary32(), 0x3F317218U);
  EXPECT_EQ(Exp(Real::Integer(1)).ToBinary64(), 0x4005BF0A8B145769U);
  EXPECT_EQ(Exp(Real::Integer(1)).ToBinary32(), 0x402DF854U);
  EXPECT_EQ(Exp(Real::Integer(-1)).ToBinary64(), 0x3FD78B56362CEF38U);
  EXPECT_EQ(Sqrt(Real::Integer(2)).ToBinary64(), 0x3FF6A09E667F3BCDU);
  EXPECT_EQ(Sqrt(Real::Integer(2)).ToBinary32(), 0x3FB504F3U);
  EXPECT_EQ(Sin(Real::Integer(1)).ToBinary64(), 0x3FEAED548F090CEEU);
  EXPECT_EQ(Cos(Real::Integer(1)).ToBinary64(), 0x3FE14A280FB5068CU);
  EXPECT_EQ(Log(Real::Integer(10)).ToBinary64(), 0x40026BB1BBB55516U);
}

TEST(ExactTest, FunctionsSatisfyTheirIdentities) {
  const uint64_t half = 0x3FE0000000000000U;
  EXPECT_EQ(Sin(Pi().DividedBy(6)).ToBinary64(), half);
  EXPECT_EQ(Cos(Pi().DividedBy(3)).ToBinary64(), half);
  EXPECT_EQ(Tanh(Log(Real::Integer(3)).Scaled(-1)).ToBinary64(), half);
  EXPECT_EQ(Pow(Real::Integer(10000), Real::Ratio(1, 2)).ToBinary64(),
            0x4059000000000000U);
  EXPECT_EQ(Exp(Log(Real::Integer(7))).ToBinary64(), 0x401C000000000000U);
  EXPECT_EQ(Exp(Real()).ToBinary32(), 0x3F800000U);
  for (const int64_t x : {-3, 511, 1000, 1571}) {
    const Real s = Sin(Real::Integer(x));
    const Real c = Cos(Real::Integer(x));
    EXPECT_EQ(((s * s) + (c * c)).ToBinary64(), 0x3FF0000000000000U) << x;
  }
  EXPECT_EQ(Sin(Pi().Scaled(-1).Negated()).ToBinary64(), 0xBFF0000000000000U);
  EXPECT_EQ(Tanh(Real::Integer(400)).ToBinary32(), 0x3F800000U);
}

TEST(ExactTest, ExactValuesRoundHalfToEven) {
  EXPECT_EQ(Real::Integer(16777217).ToBinary32(), 0x4B800000U);
  EXPECT_EQ(Real::Integer(16777219).ToBinary32(), 0x4B800002U);
  EXPECT_EQ((Real::Integer(16777217) + Real::Ratio(1, 1024)).ToBinary32(),
            0x4B800001U);
  EXPECT_EQ(Real::Integer(-3).ToBinary32(), 0xC0400000U);
  EXPECT_EQ(Real::Binary32(0x3DCCCCCDU).ToBinary64(), 0x3FB99999A0000000U);
  EXPECT_EQ(Real::Binary64(0x3FB999999999999AU).ToBinary32(), 0x3DCCCCCDU);
  EXPECT_EQ(
      (Real::Binary32(0x3F800001U) - Real::Binary32(0x3F800000U)).ToBinary32(),
      0x34000000U);
  EXPECT_EQ(Real::Ratio(7, 2).Floor(), 3);
  EXPECT_EQ(Real::Ratio(-7, 2).Floor(), -4);
  EXPECT_EQ(Real::Ratio(5, 2).Nearest(), 3);
  EXPECT_TRUE(Real::Ratio(3, 8).Exact());
  EXPECT_FALSE(Real::Ratio(1, 3).Exact());
}

TEST(ExactDeathTest, RefusesToRoundAnAmbiguousValue) {
  const Real midpoint =
      (Real::Integer(1) + Real::Integer(1).Scaled(-53)).Inexact();
  EXPECT_DEATH(static_cast<void>(midpoint.ToBinary64()), "cannot round");
}

}  // namespace
}  // namespace hotarubi::compiler::exact
