#include "npu_data.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>

#include "fp32_ref.hpp"

namespace hotarubi::tb {

void WriteAbuf(Ddr& ddr, uint32_t address, const AbufWord& word) {
  for (uint32_t i = 0; i < 128; ++i) {
    ddr.Write8(address + i, static_cast<uint8_t>(word.q.at(i)));
  }
  for (uint32_t i = 0; i < 4; ++i) {
    ddr.Write8(address + 128 + i, word.scale.at(i));
  }
  for (uint32_t i = 132; i < kAbufWordBytes; ++i) {
    ddr.Write8(address + i, 0);
  }
}

AbufWord ReadAbuf(const Ddr& ddr, uint32_t address) {
  AbufWord word;
  for (uint32_t i = 0; i < 128; ++i) {
    word.q.at(i) = static_cast<int8_t>(ddr.Read8(address + i));
  }
  for (uint32_t i = 0; i < 4; ++i) {
    word.scale.at(i) = ddr.Read8(address + 128 + i);
  }
  return word;
}

void WriteWbuf(Ddr& ddr, uint32_t address, const WbufWord& word) {
  for (uint32_t i = 0; i < 128; ++i) {
    const auto low =
        static_cast<uint32_t>(word.nibble.at(size_t{2} * i) & 0xFU);
    const auto high =
        static_cast<uint32_t>(word.nibble.at((size_t{2} * i) + 1) & 0xFU);
    ddr.Write8(address + i, static_cast<uint8_t>(low | (high << 4)));
  }
  for (uint32_t i = 0; i < 8; ++i) {
    ddr.Write8(address + 128 + i, word.scale.at(i));
  }
  for (uint32_t i = 136; i < kWbufWordBytes; ++i) {
    ddr.Write8(address + i, 0);
  }
}

void WriteFbuf(Ddr& ddr, uint32_t address, const FbufWord& word) {
  for (uint32_t i = 0; i < 32; ++i) {
    ddr.Write32(address + (4 * i), word.at(i));
  }
}

FbufWord ReadFbuf(const Ddr& ddr, uint32_t address) {
  FbufWord word{};
  for (uint32_t i = 0; i < 32; ++i) {
    word.at(i) = ddr.Read32(address + (4 * i));
  }
  return word;
}

void WriteTbuf(Ddr& ddr, uint32_t address, uint32_t value, uint32_t slope) {
  ddr.Write32(address, value);
  ddr.Write32(address + 4, slope);
  ddr.Write32(address + 8, 0);
  ddr.Write32(address + 12, 0);
}

MxBlock QuantizeMxint8(std::span<const uint32_t, 32> values) {
  uint32_t max_field = 0;
  for (const uint32_t value : values) {
    const uint32_t field = (value >> 23) & 0xFFU;
    max_field = std::max(max_field, field);
  }
  MxBlock block;
  if (max_field == 0) {
    block.scale = 127;
    return block;
  }
  block.scale = static_cast<uint8_t>(max_field);
  for (uint32_t i = 0; i < 32; ++i) {
    const uint32_t value = values[i];
    const uint32_t field = (value >> 23) & 0xFFU;
    if (field == 0) {
      continue;
    }
    const uint64_t mantissa = (value & 0x7FFFFFU) | 0x800000U;
    const uint32_t shift = 17 + max_field - field;
    int64_t rounded = 0;
    if (shift < 40) {
      const uint64_t kept = mantissa >> shift;
      const uint64_t rest = mantissa & ((uint64_t{1} << shift) - 1);
      const uint64_t half = uint64_t{1} << (shift - 1);
      rounded = static_cast<int64_t>(kept);
      if (rest > half || (rest == half && (kept & 1U) != 0)) {
        ++rounded;
      }
    }
    if ((value & kFp32SignBit) != 0) {
      rounded = -rounded;
    }
    block.q.at(i) =
        static_cast<int8_t>(std::clamp<int64_t>(rounded, -128, 127));
  }
  return block;
}

uint32_t RandomFp32(std::mt19937& rng, int min_exponent, int max_exponent) {
  const auto span = static_cast<uint32_t>(max_exponent - min_exponent + 1);
  const auto exponent = static_cast<uint32_t>(min_exponent + 127 +
                                              static_cast<int>(rng() % span));
  const uint32_t sign = rng() & 1U;
  return (sign << 31) | (exponent << 23) | (rng() & 0x7FFFFFU);
}

}  // namespace hotarubi::tb
