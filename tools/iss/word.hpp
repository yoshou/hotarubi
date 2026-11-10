#ifndef HOTARUBI_TOOLS_ISS_WORD_HPP_
#define HOTARUBI_TOOLS_ISS_WORD_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#include "hotarubi/isa.hpp"

namespace hotarubi::iss {

struct Word128 {
  uint64_t low = 0;
  uint64_t high = 0;

  [[nodiscard]] constexpr uint32_t Field(uint32_t lsb, uint32_t width) const {
    uint64_t value = 0;
    if (lsb >= 64) {
      value = high >> (lsb - 64);
    } else if (lsb == 0) {
      value = low;
    } else {
      value = (low >> lsb) | (high << (64 - lsb));
    }
    return static_cast<uint32_t>(value & ((uint64_t{1} << width) - 1));
  }

  [[nodiscard]] constexpr uint32_t Field(isa::Field field) const {
    return Field(field.lsb, field.width);
  }

  friend constexpr bool operator==(const Word128&, const Word128&) = default;
};

constexpr uint64_t LoadLittle(std::span<const std::byte> bytes) {
  uint64_t value = 0;
  for (size_t i = bytes.size(); i > 0; --i) {
    value = (value << 8) | std::to_integer<uint64_t>(bytes[i - 1]);
  }
  return value;
}

constexpr void StoreLittle(uint64_t value, std::span<std::byte> bytes) {
  for (std::byte& byte : bytes) {
    byte = static_cast<std::byte>(value & 0xFFU);
    value >>= 8;
  }
}

constexpr Word128 LoadWord128(std::span<const std::byte, 16> bytes) {
  return {.low = LoadLittle(bytes.first<8>()),
          .high = LoadLittle(bytes.last<8>())};
}

}  // namespace hotarubi::iss

#endif  // HOTARUBI_TOOLS_ISS_WORD_HPP_
