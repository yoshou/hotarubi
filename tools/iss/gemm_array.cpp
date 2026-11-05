#include "gemm_array.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <vector>

#include "fp32.hpp"
#include "hotarubi/require.hpp"
#include "memories.hpp"

namespace hotarubi::iss {
namespace {

constexpr uint32_t kAddressMask = (1U << 20) - 1;
constexpr std::array<int32_t, 8> kFp4Magnitude = {0, 1, 2, 3, 4, 6, 8, 12};

struct WeightBlock {
  std::array<std::array<int32_t, kBlock>, kLanes> values{};
  std::array<uint32_t, kLanes> scales{};
};

int32_t ScaleBias(WeightMode mode) {
  switch (mode) {
    case WeightMode::kFp4:
      return -7;
    case WeightMode::kW8Hi:
      return -8;
    case WeightMode::kW8Lo:
      return -12;
  }
  return 0;
}

WeightBlock DecodeWeights(const WbufWord& word, WeightMode mode) {
  WeightBlock block;
  for (uint32_t c = 0; c < kLanes; ++c) {
    for (uint32_t k = 0; k < kBlock; ++k) {
      const uint32_t index = (c * kBlock) + k;
      const uint32_t byte = word.at(index / 2);
      block.values.at(c).at(k) =
          WeightValue(index % 2 == 0 ? byte & 0xFU : byte >> 4, mode);
    }
    block.scales.at(c) = word.at(kWbufScaleOffset + c);
  }
  return block;
}

void AccumulateBlock(const AbufWord& a, const WeightBlock& weights,
                     WeightMode mode, FbufWord& acc) {
  for (uint32_t t = 0; t < kTokens; ++t) {
    std::array<int32_t, kBlock> activations{};
    for (uint32_t k = 0; k < kBlock; ++k) {
      activations.at(k) = SignedByte(a.at((t * kBlock) + k));
    }
    for (uint32_t c = 0; c < kLanes; ++c) {
      const std::array<int32_t, kBlock>& column = weights.values.at(c);
      int32_t sum = 0;
      for (uint32_t k = 0; k < kBlock; ++k) {
        sum += activations.at(k) * column.at(k);
      }
      uint32_t& element = acc.at((t * kLanes) + c);
      element = Fp32Add(element, BlockValue(sum, a.at(kAbufScaleOffset + t),
                                            weights.scales.at(c), mode));
    }
  }
}

}  // namespace

WeightMode DecodeWeightMode(uint32_t wmode) {
  switch (wmode & 3U) {
    case 0:
      return WeightMode::kFp4;
    case 1:
      return WeightMode::kW8Hi;
    default:
      return WeightMode::kW8Lo;
  }
}

int32_t WeightValue(uint32_t nibble, WeightMode mode) {
  const auto value = static_cast<int32_t>(nibble & 0xFU);
  const bool negative = (nibble & 0x8U) != 0;
  switch (mode) {
    case WeightMode::kFp4:
      return negative ? -kFp4Magnitude.at(nibble & 0x7U)
                      : kFp4Magnitude.at(nibble & 0x7U);
    case WeightMode::kW8Hi:
      return negative ? value - 16 : value;
    case WeightMode::kW8Lo:
      return value;
  }
  return 0;
}

uint32_t BlockValue(int32_t sum, uint32_t a_scale, uint32_t w_scale,
                    WeightMode mode) {
  if (sum == 0) {
    return 0;
  }
  const auto magnitude = static_cast<uint32_t>(std::abs(sum));
  const int32_t msb = std::bit_width(magnitude) - 1;
  const int32_t exponent = static_cast<int32_t>(a_scale + w_scale) - 254 +
                           ScaleBias(mode) + msb + 127;
  return Fp32Pack(sum < 0 ? 1U : 0U, exponent,
                  (magnitude << (23 - msb)) & 0x7FFFFFU);
}

void ExecuteGemm(const GemmCommand& command, Memories& memories) {
  Require(command.m >= 32 && command.m <= 992 && command.m % 32 == 0,
          std::format("GEMM m = {} is not a multiple of 32 from 32 to 992",
                      command.m));
  Require(command.kb >= 1 && command.kb <= command.a_stride,
          std::format("GEMM kb = {} is not in [1, a_stride = {}]", command.kb,
                      command.a_stride));
  Require(command.n8 >= 1, "GEMM n8 = 0");
  const WeightMode mode = DecodeWeightMode(command.wmode);
  const uint32_t quads = command.m / kTokens;
  std::vector<WeightBlock> weights(command.kb);
  for (uint32_t g = 0; g < command.n8; ++g) {
    for (uint32_t kb = 0; kb < command.kb; ++kb) {
      const uint32_t address =
          (command.w_base + (g * command.w_stride) + kb) & kAddressMask;
      weights.at(kb) =
          DecodeWeights(memories.wbuf.at(address % kWbufDepth), mode);
    }
    for (uint32_t q = 0; q < quads; ++q) {
      const uint32_t c_address =
          ((command.c_base + (q * command.n8) + g) & kAddressMask) % kFbufDepth;
      FbufWord acc = command.acc ? memories.fbuf.at(c_address) : FbufWord{};
      for (uint32_t kb = 0; kb < command.kb; ++kb) {
        const uint32_t a_address =
            (command.a_base + (q * command.a_stride) + kb) & kAddressMask;
        AccumulateBlock(memories.abuf.at(a_address % kAbufDepth),
                        weights.at(kb), mode, acc);
      }
      memories.fbuf.at(c_address) = acc;
    }
  }
}

}  // namespace hotarubi::iss
