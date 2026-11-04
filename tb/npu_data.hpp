#ifndef HOTARUBI_TB_NPU_DATA_HPP_
#define HOTARUBI_TB_NPU_DATA_HPP_

#include <array>
#include <cstdint>
#include <random>
#include <span>

#include "npu_bench.hpp"

namespace hotarubi::tb {

inline constexpr uint32_t kAbufWordBytes = 144;
inline constexpr uint32_t kWbufWordBytes = 144;
inline constexpr uint32_t kFbufWordBytes = 128;
inline constexpr uint32_t kTbufWordBytes = 16;
inline constexpr uint32_t kCbufWordBytes = 16;

struct AbufWord {
  std::array<int8_t, 128> q{};
  std::array<uint8_t, 4> scale{};

  bool operator==(const AbufWord&) const = default;
};

struct WbufWord {
  std::array<uint8_t, 256> nibble{};
  std::array<uint8_t, 8> scale{};
};

using FbufWord = std::array<uint32_t, 32>;

struct MxBlock {
  std::array<int8_t, 32> q{};
  uint8_t scale = 0;
};

void WriteAbuf(Ddr& ddr, uint32_t address, const AbufWord& word);
AbufWord ReadAbuf(const Ddr& ddr, uint32_t address);
void WriteWbuf(Ddr& ddr, uint32_t address, const WbufWord& word);
void WriteFbuf(Ddr& ddr, uint32_t address, const FbufWord& word);
FbufWord ReadFbuf(const Ddr& ddr, uint32_t address);
void WriteTbuf(Ddr& ddr, uint32_t address, uint32_t value, uint32_t slope);

MxBlock QuantizeMxint8(std::span<const uint32_t, 32> values);
uint32_t RandomFp32(std::mt19937& rng, int min_exponent, int max_exponent);

}  // namespace hotarubi::tb

#endif  // HOTARUBI_TB_NPU_DATA_HPP_
