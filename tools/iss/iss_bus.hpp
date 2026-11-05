#ifndef HOTARUBI_TOOLS_ISS_ISS_BUS_HPP_
#define HOTARUBI_TOOLS_ISS_ISS_BUS_HPP_

#include <chrono>
#include <cstdint>
#include <expected>
#include <string>

#include "hotarubi/driver.hpp"
#include "machine.hpp"

namespace hotarubi::iss {

class IssBus final : public Bus {
 public:
  explicit IssBus(Machine& machine) : machine_(machine) {}

  uint32_t Read(Register reg) override;
  void Write(Register reg, uint32_t value) override;
  std::expected<void, std::string> EnableInterrupt() override;
  std::expected<bool, std::string> WaitInterrupt(
      std::chrono::milliseconds timeout) override;

 private:
  Machine& machine_;
  bool enabled_ = false;
};

}  // namespace hotarubi::iss

#endif  // HOTARUBI_TOOLS_ISS_ISS_BUS_HPP_
