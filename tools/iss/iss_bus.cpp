#include "iss_bus.hpp"

#include <chrono>
#include <cstdint>
#include <expected>
#include <string>

#include "hotarubi/driver.hpp"
#include "machine.hpp"

namespace hotarubi::iss {

uint32_t IssBus::Read(Register reg) {
  return machine_.ReadRegister(static_cast<uint32_t>(reg));
}

void IssBus::Write(Register reg, uint32_t value) {
  machine_.WriteRegister(static_cast<uint32_t>(reg), value);
}

std::expected<void, std::string> IssBus::EnableInterrupt() {
  enabled_ = true;
  return {};
}

std::expected<bool, std::string> IssBus::WaitInterrupt(
    [[maybe_unused]] std::chrono::milliseconds timeout) {
  bool blocked = false;
  while (true) {
    if (enabled_ && machine_.Interrupt()) {
      enabled_ = false;
      return true;
    }
    if (blocked) {
      return false;
    }
    const Stop stop = machine_.Run();
    blocked = stop == Stop::kIdle || stop == Stop::kWaitHost ||
              stop == Stop::kStarved;
  }
}

}  // namespace hotarubi::iss
