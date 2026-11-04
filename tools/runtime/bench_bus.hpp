#ifndef HOTARUBI_TOOLS_RUNTIME_BENCH_BUS_HPP_
#define HOTARUBI_TOOLS_RUNTIME_BENCH_BUS_HPP_

#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "hotarubi/driver.hpp"
#include "npu_bench.hpp"

namespace hotarubi {

class BenchBus final : public Bus {
 public:
  static constexpr uint64_t kCyclesPerMillisecond = 250000;

  explicit BenchBus(tb::NpuBench& bench) : bench_(bench) {}

  uint32_t Read(Register reg) override {
    return bench_.ReadReg(static_cast<uint32_t>(reg));
  }

  void Write(Register reg, uint32_t value) override {
    bench_.WriteReg(static_cast<uint32_t>(reg), value);
  }

  std::expected<void, std::string> EnableInterrupt() override {
    enabled_ = true;
    return {};
  }

  std::expected<bool, std::string> WaitInterrupt(
      std::chrono::milliseconds timeout) override {
    const uint64_t cycles =
        static_cast<uint64_t>(timeout.count()) * kCyclesPerMillisecond;
    for (uint64_t cycle = 0; cycle <= cycles; ++cycle) {
      if (enabled_ && bench_.Irq()) {
        enabled_ = false;
        return true;
      }
      bench_.Tick();
    }
    return false;
  }

 private:
  tb::NpuBench& bench_;
  bool enabled_ = false;
};

inline std::span<std::byte> BenchMemory(tb::NpuBench& bench) {
  return std::as_writable_bytes(bench.Memory().Bytes());
}

inline std::vector<std::byte> Serialize(
    std::span<const tb::Instruction> program) {
  std::vector<std::byte> bytes;
  for (const tb::Instruction& instruction : program) {
    for (const uint32_t word : instruction.words) {
      for (uint32_t i = 0; i < 4; ++i) {
        bytes.push_back(static_cast<std::byte>(word >> (8 * i)));
      }
    }
  }
  return bytes;
}

inline std::vector<std::byte> FloatBytes(std::span<const float> values) {
  std::vector<std::byte> bytes;
  for (const float value : values) {
    const auto bits = std::bit_cast<uint32_t>(value);
    for (uint32_t i = 0; i < 4; ++i) {
      bytes.push_back(static_cast<std::byte>(bits >> (8 * i)));
    }
  }
  return bytes;
}

}  // namespace hotarubi

#endif  // HOTARUBI_TOOLS_RUNTIME_BENCH_BUS_HPP_
