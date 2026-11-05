#ifndef HOTARUBI_TOOLS_RUNTIME_SCRIPTED_BUS_HPP_
#define HOTARUBI_TOOLS_RUNTIME_SCRIPTED_BUS_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <utility>
#include <vector>

#include "hotarubi/driver.hpp"

namespace hotarubi {

class ScriptedBus final : public Bus {
 public:
  static constexpr uint32_t kIrqSignal = 0x1;
  static constexpr uint32_t kIrqHalt = 0x2;

  explicit ScriptedBus(std::vector<Signal> signals)
      : signals_(std::move(signals)) {}

  uint32_t Read(Register reg) override {
    switch (reg) {
      case Register::kIrqStatus:
        return irq_;
      case Register::kSignal:
        return signal_.id;
      case Register::kSignalValue:
        return signal_.value;
      default:
        return 0;
    }
  }

  void Write(Register reg, uint32_t value) override {
    switch (reg) {
      case Register::kCtrl:
        if ((value & 1U) != 0) {
          next_ = 0;
          ++starts_;
          Advance();
        }
        break;
      case Register::kIrqStatus:
        irq_ &= ~value;
        break;
      case Register::kHostGo:
        host_go_.push_back(value);
        Advance();
        break;
      default:
        break;
    }
  }

  std::expected<void, std::string> EnableInterrupt() override { return {}; }

  std::expected<bool, std::string> WaitInterrupt(
      [[maybe_unused]] std::chrono::milliseconds timeout) override {
    return irq_ != 0;
  }

  [[nodiscard]] uint32_t Starts() const { return starts_; }
  [[nodiscard]] const std::vector<uint32_t>& HostGo() const { return host_go_; }

 private:
  void Advance() {
    if (next_ < signals_.size()) {
      signal_ = signals_[next_++];
      irq_ |= kIrqSignal;
    } else {
      irq_ |= kIrqHalt;
    }
  }

  std::vector<Signal> signals_;
  Signal signal_;
  size_t next_ = 0;
  uint32_t irq_ = 0;
  uint32_t starts_ = 0;
  std::vector<uint32_t> host_go_;
};

}  // namespace hotarubi

#endif  // HOTARUBI_TOOLS_RUNTIME_SCRIPTED_BUS_HPP_
