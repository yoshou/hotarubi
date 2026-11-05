#include "ddr.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <span>

#include "hotarubi/require.hpp"
#include "word.hpp"

namespace hotarubi::iss {

void Ddr::Read(uint32_t address, std::span<std::byte> out) const {
  const size_t offset = Offset(address, out.size());
  std::ranges::copy(bytes_.subspan(offset, out.size()), out.begin());
}

void Ddr::Write(uint32_t address, std::span<const std::byte> in) {
  const size_t offset = Offset(address, in.size());
  std::ranges::copy(in, bytes_.subspan(offset, in.size()).begin());
}

uint32_t Ddr::Read32(uint32_t address) const {
  std::array<std::byte, 4> bytes{};
  Read(address, bytes);
  return static_cast<uint32_t>(LoadLittle(bytes));
}

uint64_t Ddr::Read64(uint32_t address) const {
  std::array<std::byte, 8> bytes{};
  Read(address, bytes);
  return LoadLittle(bytes);
}

size_t Ddr::Offset(uint32_t address, size_t size) const {
  if (address < base_ || address - base_ > bytes_.size() ||
      size > bytes_.size() - (address - base_)) {
    Fail(
        std::format("DDR access of {} bytes at {:#x} is outside [{:#x}, {:#x})",
                    size, address, base_, base_ + bytes_.size()));
  }
  return static_cast<size_t>(address - base_);
}

}  // namespace hotarubi::iss
