#ifndef HOTARUBI_RUNTIME_HPP_
#define HOTARUBI_RUNTIME_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "hotarubi/driver.hpp"

namespace hotarubi {

struct Region {
  uint32_t offset = 0;
  uint32_t size = 0;
};

struct Program {
  uint64_t base = 0;
  std::vector<std::byte> image;
  uint32_t instruction_offset = 0;
  uint32_t instruction_count = 0;
  std::map<std::string, Region, std::less<>> regions;
};

class Runtime {
 public:
  Runtime(Device& device, std::span<std::byte> memory,
          uint64_t physical_address)
      : device_(device), memory_(memory), physical_address_(physical_address) {}

  [[nodiscard]] std::expected<void, std::string> Load(Program program);
  [[nodiscard]] std::expected<void, std::string> Write(
      std::string_view region, std::span<const std::byte> data);
  [[nodiscard]] std::expected<void, std::string> Read(
      std::string_view region, std::span<std::byte> data) const;
  [[nodiscard]] std::expected<Counters, std::string> Run(
      std::chrono::milliseconds timeout, const SignalHandler& on_signal = {});

 private:
  [[nodiscard]] std::expected<Region, std::string> Find(std::string_view region,
                                                        size_t size) const;

  Device& device_;
  std::span<std::byte> memory_;
  uint64_t physical_address_;
  Program program_;
  bool loaded_ = false;
};

}  // namespace hotarubi

#endif  // HOTARUBI_RUNTIME_HPP_
