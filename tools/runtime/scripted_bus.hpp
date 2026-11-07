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
      case Register::kTraceCtrl:
        return trace_ctrl_ | (trace_idle_ ? 2U : 0U);
      case Register::kTraceCount:
        return trace_count_;
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
      case Register::kTraceBase:
        trace_base_ = value;
        break;
      case Register::kTraceCtrl:
        trace_ctrl_ = value & 5U;
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
  [[nodiscard]] uint32_t TraceBase() const { return trace_base_; }
  [[nodiscard]] uint32_t TraceCtrl() const { return trace_ctrl_; }
  void SetTraceIdle(bool idle) { trace_idle_ = idle; }
  void SetTraceCount(uint32_t count) { trace_count_ = count; }

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
  uint32_t trace_base_ = 0;
  uint32_t trace_ctrl_ = 0;
  uint32_t trace_count_ = 0;
  bool trace_idle_ = true;
};

}  // namespace hotarubi

#endif  // HOTARUBI_TOOLS_RUNTIME_SCRIPTED_BUS_HPP_
