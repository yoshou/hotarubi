#ifndef HOTARUBI_TOOLS_ISS_DDR_HPP_
#define HOTARUBI_TOOLS_ISS_DDR_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

namespace hotarubi::iss {

class Ddr {
 public:
  Ddr(std::span<std::byte> bytes, uint64_t base) : bytes_(bytes), base_(base) {}

  void Read(uint32_t address, std::span<std::byte> out) const;
  void Write(uint32_t address, std::span<const std::byte> in);
  [[nodiscard]] uint32_t Read32(uint32_t address) const;
  [[nodiscard]] uint64_t Read64(uint32_t address) const;

 private:
  [[nodiscard]] size_t Offset(uint32_t address, size_t size) const;

  std::span<std::byte> bytes_;
  uint64_t base_;
};

}  // namespace hotarubi::iss

#endif  // HOTARUBI_TOOLS_ISS_DDR_HPP_
