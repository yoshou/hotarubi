#include "gemm_array.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "hotarubi/exact.hpp"
#include "memories.hpp"
#include "reference.hpp"

namespace hotarubi::iss {
namespace {

int64_t E2M1Times2(uint32_t code) {
  const uint32_t exponent = (code >> 1) & 3U;
  const uint32_t mantissa = code & 1U;
  if (exponent == 0) {
    return mantissa;
  }
  return (2 + mantissa) << (exponent - 1);
}

int64_t Weight(uint32_t nibble, uint32_t wmode) {
  if (wmode == 0) {
    const int64_t magnitude = E2M1Times2(nibble & 7U);
    return (nibble & 8U) != 0 ? -magnitude : magnitude;
  }
  if (wmode == 1) {
    return (nibble & 8U) != 0 ? static_cast<int64_t>(nibble) - 16 : nibble;
  }
  return nibble;
}

int64_t ScaleBias(uint32_t wmode) {
  if (wmode == 0) {
    return -7;
  }
  return wmode == 1 ? -8 : -12;
}

class GemmArrayTest : public testing::Test {
 protected:
  void Fill(uint32_t seed) {
    std::mt19937 generator(seed);
    std::uniform_int_distribution<uint32_t> byte(0, 255);
    std::uniform_int_distribution<uint32_t> scale(118, 136);
    for (AbufWord& word : memories_.abuf) {
      for (size_t i = 0; i < kAbufScaleOffset; ++i) {
        word.at(i) = static_cast<uint8_t>(byte(generator));
      }
      for (size_t t = 0; t < kTokens; ++t) {
        word.at(kAbufScaleOffset + t) = static_cast<uint8_t>(scale(generator));
      }
    }
    for (WbufWord& word : memories_.wbuf) {
      for (size_t i = 0; i < kWbufScaleOffset; ++i) {
        word.at(i) = static_cast<uint8_t>(byte(generator));
      }
      for (size_t c = 0; c < kLanes; ++c) {
        word.at(kWbufScaleOffset + c) = static_cast<uint8_t>(scale(generator));
      }
    }
    for (FbufWord& word : memories_.fbuf) {
      for (uint32_t& element : word) {
        element = 0x3F000000U + (byte(generator) << 16);
      }
    }
  }

  [[nodiscard]] uint32_t Expected(const GemmCommand& command, uint32_t token,
                                  uint32_t column, uint32_t initial) const {
    const uint32_t q = token / kTokens;
    const uint32_t g = column / kLanes;
    uint32_t acc = initial;
    for (uint32_t b = 0; b < command.kb; ++b) {
      const AbufWord& a = memories_.abuf.at(
          (command.a_base + (q * command.a_stride) + b) % kAbufDepth);
      const WbufWord& w = memories_.wbuf.at(
          (command.w_base + (g * command.w_stride) + b) % kWbufDepth);
      int64_t sum = 0;
      for (uint32_t k = 0; k < kBlock; ++k) {
        const uint32_t nibble_index = ((column % kLanes) * kBlock) + k;
        const uint32_t nibble =
            (w.at(nibble_index / 2) >> (4 * (nibble_index % 2))) & 0xFU;
        sum += static_cast<int64_t>(static_cast<int8_t>(
                   a.at(((token % kTokens) * kBlock) + k))) *
               Weight(nibble, command.wmode);
      }
      const int64_t exponent =
          (static_cast<int64_t>(a.at(kAbufScaleOffset + (token % kTokens))) -
           127) +
          (static_cast<int64_t>(w.at(kWbufScaleOffset + (column % kLanes))) -
           127) +
          ScaleBias(command.wmode);
      acc = reference::Add(
          acc, reference::Round(exact::Real::Integer(sum).Scaled(exponent)));
    }
    return acc;
  }

  void Check(const GemmCommand& command, const std::vector<FbufWord>& before) {
    for (uint32_t token = 0; token < command.m; ++token) {
      for (uint32_t column = 0; column < command.n8 * kLanes; ++column) {
        const uint32_t address =
            (command.c_base + ((token / kTokens) * command.n8) +
             (column / kLanes)) %
            kFbufDepth;
        const uint32_t element =
            ((token % kTokens) * kLanes) + (column % kLanes);
        const uint32_t initial =
            command.acc ? before.at(address).at(element) : 0;
        ASSERT_EQ(memories_.fbuf.at(address).at(element),
                  Expected(command, token, column, initial))
            << "token " << token << " column " << column;
      }
    }
  }

  Memories memories_;
};

TEST(GemmWeightTest, DecodesTheFourBitValues) {
  for (uint32_t nibble = 0; nibble < 16; ++nibble) {
    EXPECT_EQ(WeightValue(nibble, WeightMode::kFp4), Weight(nibble, 0));
    EXPECT_EQ(WeightValue(nibble, WeightMode::kW8Hi), Weight(nibble, 1));
    EXPECT_EQ(WeightValue(nibble, WeightMode::kW8Lo), Weight(nibble, 2));
  }
  EXPECT_EQ(DecodeWeightMode(3), WeightMode::kW8Lo);
}

TEST(GemmWeightTest, ConvertsBlockSumsExactly) {
  EXPECT_EQ(BlockValue(0, 127, 127, WeightMode::kFp4), 0U);
  EXPECT_EQ(BlockValue(128, 127, 127, WeightMode::kFp4), 0x3F800000U);
  EXPECT_EQ(BlockValue(-3, 128, 127, WeightMode::kW8Hi), 0xBCC00000U);
  EXPECT_EQ(BlockValue(61440, 127, 127, WeightMode::kW8Lo),
            reference::FromInt(15));
  EXPECT_EQ(BlockValue(1, 1, 1, WeightMode::kFp4), 0U);
  EXPECT_EQ(BlockValue(1, 255, 255, WeightMode::kFp4), 0x7F800000U);
  EXPECT_EQ(BlockValue(-1, 255, 255, WeightMode::kFp4), 0xFF800000U);
}

TEST_F(GemmArrayTest, ComputesEveryWeightMode) {
  for (uint32_t wmode = 0; wmode < 3; ++wmode) {
    Fill(10 + wmode);
    const GemmCommand command{.a_base = 100,
                              .a_stride = 4,
                              .kb = 3,
                              .m = 64,
                              .w_base = 7,
                              .w_stride = 5,
                              .n8 = 3,
                              .c_base = 200,
                              .wmode = wmode,
                              .acc = false};
    const std::vector<FbufWord> before = memories_.fbuf;
    ExecuteGemm(command, memories_);
    Check(command, before);
  }
}

TEST_F(GemmArrayTest, AccumulatesOntoTheExistingOutput) {
  Fill(20);
  const GemmCommand command{.a_base = 8100,
                            .a_stride = 2,
                            .kb = 2,
                            .m = 32,
                            .w_base = 2000,
                            .w_stride = 2,
                            .n8 = 2,
                            .c_base = 4000,
                            .wmode = 1,
                            .acc = true};
  const std::vector<FbufWord> before = memories_.fbuf;
  ExecuteGemm(command, memories_);
  Check(command, before);
}

TEST_F(GemmArrayTest, LeavesOtherWordsUnchanged) {
  Fill(30);
  const std::vector<FbufWord> before = memories_.fbuf;
  ExecuteGemm({.a_base = 0,
               .a_stride = 1,
               .kb = 1,
               .m = 32,
               .w_base = 0,
               .w_stride = 1,
               .n8 = 1,
               .c_base = 64,
               .wmode = 0,
               .acc = false},
              memories_);
  for (uint32_t address = 0; address < kFbufDepth; ++address) {
    if (address < 64 || address >= 72) {
      EXPECT_EQ(memories_.fbuf.at(address), before.at(address));
    }
  }
}

TEST(GemmCommandDeathTest, RejectsTokenCountsThatAreNotWholeGroups) {
  Memories memories;
  EXPECT_DEATH(ExecuteGemm({.a_base = 0,
                            .a_stride = 1,
                            .kb = 1,
                            .m = 16,
                            .w_base = 0,
                            .w_stride = 1,
                            .n8 = 1,
                            .c_base = 0,
                            .wmode = 0,
                            .acc = false},
                           memories),
               "m = 16");
}

}  // namespace
}  // namespace hotarubi::iss
