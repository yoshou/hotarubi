#ifndef HOTARUBI_DRIVER_HPP_
#define HOTARUBI_DRIVER_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "hotarubi/isa.hpp"

namespace hotarubi {

enum class Register : uint8_t {
  kCtrl = 0x00,
  kStatus = 0x04,
  kInstrBase = 0x08,
  kInstrCount = 0x0C,
  kIrqStatus = 0x10,
  kIrqEnable = 0x14,
  kSignal = 0x18,
  kSignalValue = 0x1C,
  kIssued = 0x20,
  kCycles = 0x24,
  kDmaCycles = 0x28,
  kGemmCycles = 0x2C,
  kVecCycles = 0x30,
  kStallCycles = 0x34,
  kStarveCycles = 0x38,
  kTraceBase = 0x3C,
  kTraceCtrl = 0x40,
  kTraceCount = 0x44,
  kHostGo = 0x48,
};

constexpr uint32_t Offset(Register reg) { return static_cast<uint32_t>(reg); }

inline constexpr uint32_t kRegisterAddressBits = 8;
inline constexpr uint32_t kRegisterBytes = 4;

inline constexpr uint32_t kCtrlStart = 1U << 0;
inline constexpr uint32_t kCtrlSoftReset = 1U << 1;

inline constexpr uint32_t kStatusRunning = 1U << 0;
inline constexpr uint32_t kStatusHalted = 1U << 1;
inline constexpr uint32_t kStatusError = 1U << 2;
inline constexpr uint32_t kStatusWaitingHost = 1U << 3;
inline constexpr uint32_t kStatusBusyShift = 4;
inline constexpr uint32_t kStatusIbufEmpty =
    1U << (kStatusBusyShift + isa::kEngines);

constexpr uint32_t StatusBusy(uint32_t engine) {
  return 1U << (kStatusBusyShift + engine);
}

inline constexpr uint32_t kIrqSignal = 1U << 0;
inline constexpr uint32_t kIrqHalt = 1U << 1;
inline constexpr uint32_t kIrqIllegal = 1U << 2;
inline constexpr uint32_t kIrqAxiError = 1U << 3;
inline constexpr uint32_t kIrqAll =
    kIrqSignal | kIrqHalt | kIrqIllegal | kIrqAxiError;

inline constexpr isa::Field kSignalId{.lsb = 0, .width = 8};
inline constexpr isa::Field kSignalCount{.lsb = 16, .width = 16};

inline constexpr uint32_t kTraceEnable = 1U << 0;
inline constexpr uint32_t kTraceIdle = 1U << 1;
inline constexpr uint32_t kTraceStatistics = 1U << 2;

struct Counters {
  uint32_t issued = 0;
  uint32_t cycles = 0;
  uint32_t gemm_cycles = 0;
  uint32_t vec_cycles = 0;
  uint32_t dma_cycles = 0;
  uint32_t stall_cycles = 0;
  uint32_t starve_cycles = 0;
  uint32_t trace_count = 0;
};

struct TraceOptions {
  std::optional<uint32_t> base;
  bool statistics = false;
};

struct Signal {
  uint32_t id = 0;
  uint32_t value = 0;
};

using SignalHandler = std::function<void(const Signal&)>;

class Bus {
 public:
  Bus() = default;
  Bus(const Bus&) = delete;
  Bus& operator=(const Bus&) = delete;
  Bus(Bus&&) = delete;
  Bus& operator=(Bus&&) = delete;
  virtual ~Bus() = default;

  virtual uint32_t Read(Register reg) = 0;
  virtual void Write(Register reg, uint32_t value) = 0;
  virtual std::expected<void, std::string> EnableInterrupt() = 0;
  virtual std::expected<bool, std::string> WaitInterrupt(
      std::chrono::milliseconds timeout) = 0;
};

class UioBus final : public Bus {
 public:
  static constexpr size_t kMapSize = 0x1000;

  static std::expected<std::unique_ptr<UioBus>, std::string> Open(
      const std::filesystem::path& device);

  UioBus(const UioBus&) = delete;
  UioBus& operator=(const UioBus&) = delete;
  UioBus(UioBus&&) = delete;
  UioBus& operator=(UioBus&&) = delete;
  ~UioBus() override;

  uint32_t Read(Register reg) override;
  void Write(Register reg, uint32_t value) override;
  std::expected<void, std::string> EnableInterrupt() override;
  std::expected<bool, std::string> WaitInterrupt(
      std::chrono::milliseconds timeout) override;

 private:
  UioBus(int fd, volatile uint32_t* registers);

  int fd_;
  volatile uint32_t* registers_;
};

std::expected<std::filesystem::path, std::string> FindUio(
    const std::filesystem::path& class_dir, std::string_view name);

class SharedBuffer {
 public:
  static std::expected<SharedBuffer, std::string> Open(
      const std::filesystem::path& class_dir,
      const std::filesystem::path& device);

  SharedBuffer(const SharedBuffer&) = delete;
  SharedBuffer& operator=(const SharedBuffer&) = delete;
  SharedBuffer(SharedBuffer&& other) noexcept;
  SharedBuffer& operator=(SharedBuffer&& other) noexcept;
  ~SharedBuffer();

  [[nodiscard]] uint64_t PhysicalAddress() const { return physical_address_; }
  [[nodiscard]] std::span<std::byte> Bytes() const { return {data_, size_}; }

 private:
  SharedBuffer(int fd, std::byte* data, size_t size, uint64_t physical_address);
  void Release();

  int fd_;
  std::byte* data_;
  size_t size_;
  uint64_t physical_address_;
};

class Device {
 public:
  explicit Device(Bus& bus) : bus_(bus) {}

  [[nodiscard]] Counters ReadCounters() const;
  [[nodiscard]] std::expected<Counters, std::string> Run(
      uint32_t address, uint32_t count, std::chrono::milliseconds timeout,
      const SignalHandler& on_signal = {}, const TraceOptions& trace = {});

 private:
  using Deadline = std::chrono::steady_clock::time_point;

  [[nodiscard]] std::expected<Counters, std::string> Finish(
      uint32_t pending, const TraceOptions& trace, Deadline deadline);
  [[nodiscard]] std::expected<uint32_t, std::string> FinishTrace(
      const TraceOptions& trace, uint32_t issued, Deadline deadline);

  Bus& bus_;
};

}  // namespace hotarubi

#endif  // HOTARUBI_DRIVER_HPP_
