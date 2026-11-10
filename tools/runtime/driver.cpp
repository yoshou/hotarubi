#include "hotarubi/driver.hpp"

#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace hotarubi {
namespace {

constexpr uint32_t kIrqEnd = kIrqHalt | kIrqIllegal | kIrqAxiError;
constexpr uint64_t kAddressLimit = uint64_t{1} << 32;

std::string ErrnoMessage(std::string_view what,
                         const std::filesystem::path& path) {
  return std::format("{} {}: {}", what, path.string(),
                     std::error_code(errno, std::generic_category()).message());
}

std::expected<std::string, std::string> ReadLine(
    const std::filesystem::path& path) {
  std::ifstream file(path);
  std::string line;
  if (!file || !std::getline(file, line)) {
    return std::unexpected(std::format("cannot read {}", path.string()));
  }
  return line;
}

std::expected<uint64_t, std::string> ParseNumber(
    const std::filesystem::path& path) {
  auto line = ReadLine(path);
  if (!line) {
    return std::unexpected(line.error());
  }
  std::string_view text = *line;
  int base = 10;
  if (text.starts_with("0x") || text.starts_with("0X")) {
    text.remove_prefix(2);
    base = 16;
  }
  uint64_t value = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value, base);
  if (error != std::errc() || end != text.data() + text.size() ||
      text.empty()) {
    return std::unexpected(
        std::format("cannot parse {}: {}", path.string(), *line));
  }
  return value;
}

}  // namespace

std::expected<std::unique_ptr<UioBus>, std::string> UioBus::Open(
    const std::filesystem::path& device) {
  const int fd = open(device.c_str(), O_RDWR | O_SYNC | O_CLOEXEC);
  if (fd < 0) {
    return std::unexpected(ErrnoMessage("cannot open", device));
  }
  void* map =
      mmap(nullptr, kMapSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (map == MAP_FAILED) {
    std::string message = ErrnoMessage("cannot map", device);
    close(fd);
    return std::unexpected(std::move(message));
  }
  return std::unique_ptr<UioBus>(
      new UioBus(fd, static_cast<volatile uint32_t*>(map)));
}

UioBus::UioBus(int fd, volatile uint32_t* registers)
    : fd_(fd), registers_(registers) {}

UioBus::~UioBus() {
  munmap(const_cast<uint32_t*>(registers_), kMapSize);
  close(fd_);
}

uint32_t UioBus::Read(Register reg) {
  return registers_[static_cast<uint32_t>(reg) / sizeof(uint32_t)];
}

void UioBus::Write(Register reg, uint32_t value) {
  registers_[static_cast<uint32_t>(reg) / sizeof(uint32_t)] = value;
}

std::expected<void, std::string> UioBus::EnableInterrupt() {
  const uint32_t enable = 1;
  if (write(fd_, &enable, sizeof(enable)) !=
      static_cast<ssize_t>(sizeof(enable))) {
    return std::unexpected(
        std::format("cannot enable the interrupt: {}",
                    std::error_code(errno, std::generic_category()).message()));
  }
  return {};
}

std::expected<bool, std::string> UioBus::WaitInterrupt(
    std::chrono::milliseconds timeout) {
  pollfd request{.fd = fd_, .events = POLLIN, .revents = 0};
  const int ready = poll(&request, 1, static_cast<int>(timeout.count()));
  if (ready < 0) {
    return std::unexpected(
        std::format("cannot wait for the interrupt: {}",
                    std::error_code(errno, std::generic_category()).message()));
  }
  if (ready == 0) {
    return false;
  }
  uint32_t count = 0;
  if (read(fd_, &count, sizeof(count)) != static_cast<ssize_t>(sizeof(count))) {
    return std::unexpected(
        std::format("cannot read the interrupt count: {}",
                    std::error_code(errno, std::generic_category()).message()));
  }
  return true;
}

std::expected<std::filesystem::path, std::string> FindUio(
    const std::filesystem::path& class_dir, std::string_view name) {
  std::error_code error;
  for (std::filesystem::directory_iterator it(class_dir, error);
       !error && it != std::filesystem::directory_iterator();
       it.increment(error)) {
    const auto line = ReadLine(it->path() / "name");
    if (line && *line == name) {
      return std::filesystem::path("/dev") / it->path().filename();
    }
  }
  if (error) {
    return std::unexpected(
        std::format("cannot list {}: {}", class_dir.string(), error.message()));
  }
  return std::unexpected(
      std::format("no UIO device named {} in {}", name, class_dir.string()));
}

std::expected<SharedBuffer, std::string> SharedBuffer::Open(
    const std::filesystem::path& class_dir,
    const std::filesystem::path& device) {
  const auto physical_address = ParseNumber(class_dir / "phys_addr");
  if (!physical_address) {
    return std::unexpected(physical_address.error());
  }
  const auto size = ParseNumber(class_dir / "size");
  if (!size) {
    return std::unexpected(size.error());
  }
  if (*size == 0 || *physical_address + *size > kAddressLimit) {
    return std::unexpected(std::format(
        "the buffer at {:#x} with {} bytes is not inside the 32-bit address "
        "space",
        *physical_address, *size));
  }
  const int fd = open(device.c_str(), O_RDWR | O_SYNC | O_CLOEXEC);
  if (fd < 0) {
    return std::unexpected(ErrnoMessage("cannot open", device));
  }
  void* map = mmap(nullptr, *size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (map == MAP_FAILED) {
    std::string message = ErrnoMessage("cannot map", device);
    close(fd);
    return std::unexpected(std::move(message));
  }
  return SharedBuffer(fd, static_cast<std::byte*>(map), *size,
                      *physical_address);
}

SharedBuffer::SharedBuffer(int fd, std::byte* data, size_t size,
                           uint64_t physical_address)
    : fd_(fd), data_(data), size_(size), physical_address_(physical_address) {}

SharedBuffer::SharedBuffer(SharedBuffer&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)),
      data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      physical_address_(std::exchange(other.physical_address_, 0)) {}

SharedBuffer& SharedBuffer::operator=(SharedBuffer&& other) noexcept {
  if (this != &other) {
    Release();
    fd_ = std::exchange(other.fd_, -1);
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
    physical_address_ = std::exchange(other.physical_address_, 0);
  }
  return *this;
}

SharedBuffer::~SharedBuffer() { Release(); }

void SharedBuffer::Release() {
  if (data_ != nullptr) {
    munmap(data_, size_);
  }
  if (fd_ >= 0) {
    close(fd_);
  }
}

Counters Device::ReadCounters() const {
  return {.issued = bus_.Read(Register::kIssued),
          .cycles = bus_.Read(Register::kCycles),
          .gemm_cycles = bus_.Read(Register::kGemmCycles),
          .vec_cycles = bus_.Read(Register::kVecCycles),
          .dma_cycles = bus_.Read(Register::kDmaCycles),
          .stall_cycles = bus_.Read(Register::kStallCycles),
          .starve_cycles = bus_.Read(Register::kStarveCycles),
          .trace_count = bus_.Read(Register::kTraceCount)};
}

std::expected<uint32_t, std::string> Device::FinishTrace(
    const TraceOptions& trace, uint32_t issued, Deadline deadline) {
  while ((bus_.Read(Register::kTraceCtrl) & kTraceIdle) == 0) {
    if (std::chrono::steady_clock::now() >= deadline) {
      bus_.Write(Register::kTraceCtrl, 0);
      return std::unexpected(
          std::format("the trace did not drain: {} records written",
                      bus_.Read(Register::kTraceCount)));
    }
  }
  const uint32_t count = bus_.Read(Register::kTraceCount);
  bus_.Write(Register::kTraceCtrl, 0);
  const uint32_t most = trace.statistics ? 2 * issued : issued;
  if (count < issued || count > most) {
    return std::unexpected(std::format(
        "the trace holds {} records for {} instructions", count, issued));
  }
  return count;
}

std::expected<Counters, std::string> Device::Finish(uint32_t pending,
                                                    const TraceOptions& trace,
                                                    Deadline deadline) {
  bus_.Write(Register::kIrqStatus, pending & kIrqEnd);
  bus_.Write(Register::kIrqEnable, 0);
  if ((pending & kIrqIllegal) != 0) {
    return std::unexpected(
        std::format("illegal instruction after {} instructions",
                    bus_.Read(Register::kIssued)));
  }
  if ((pending & kIrqAxiError) != 0) {
    return std::unexpected("AXI error response");
  }
  Counters counters = ReadCounters();
  if (trace.base) {
    const auto recorded = FinishTrace(trace, counters.issued, deadline);
    if (!recorded) {
      return std::unexpected(recorded.error());
    }
    counters.trace_count = *recorded;
  }
  return counters;
}

std::expected<Counters, std::string> Device::Run(
    uint32_t address, uint32_t count, std::chrono::milliseconds timeout,
    const SignalHandler& on_signal, const TraceOptions& trace) {
  const uint32_t enabled_irqs = on_signal ? kIrqAll : kIrqEnd;
  bus_.Write(Register::kIrqStatus, kIrqAll);
  bus_.Write(Register::kIrqEnable, enabled_irqs);
  bus_.Write(Register::kInstrBase, address);
  bus_.Write(Register::kInstrCount, count);
  if (trace.base) {
    bus_.Write(Register::kTraceBase, *trace.base);
    bus_.Write(Register::kTraceCtrl,
               kTraceEnable | (trace.statistics ? kTraceStatistics : 0U));
  } else {
    bus_.Write(Register::kTraceCtrl, 0);
  }
  if (auto enabled = bus_.EnableInterrupt(); !enabled) {
    return std::unexpected(enabled.error());
  }
  bus_.Write(Register::kCtrl, kCtrlStart);
  const Deadline deadline = std::chrono::steady_clock::now() + timeout;
  while (true) {
    const auto remaining = std::chrono::ceil<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now());
    const auto interrupted = bus_.WaitInterrupt(
        std::max(remaining, std::chrono::milliseconds::zero()));
    if (!interrupted) {
      return std::unexpected(interrupted.error());
    }
    if (!*interrupted) {
      bus_.Write(Register::kIrqEnable, 0);
      bus_.Write(Register::kTraceCtrl, 0);
      return std::unexpected(std::format(
          "the instruction stream did not halt within {} ms (status {:#x}, "
          "{} instructions issued)",
          timeout.count(), bus_.Read(Register::kStatus),
          bus_.Read(Register::kIssued)));
    }
    const uint32_t pending = bus_.Read(Register::kIrqStatus) & enabled_irqs;
    if ((pending & kIrqSignal) != 0) {
      const Signal signal{
          .id = (bus_.Read(Register::kSignal) >> kSignalId.lsb) &
                ((1U << kSignalId.width) - 1),
          .value = bus_.Read(Register::kSignalValue)};
      bus_.Write(Register::kIrqStatus, kIrqSignal);
      on_signal(signal);
      bus_.Write(Register::kHostGo, signal.value);
    }
    if ((pending & kIrqEnd) != 0) {
      return Finish(pending, trace, deadline);
    }
    if (auto enabled = bus_.EnableInterrupt(); !enabled) {
      return std::unexpected(enabled.error());
    }
  }
}

}  // namespace hotarubi
