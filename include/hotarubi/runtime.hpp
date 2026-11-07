#ifndef HOTARUBI_RUNTIME_HPP_
#define HOTARUBI_RUNTIME_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
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

struct Stream {
  uint32_t offset = 0;
  uint32_t count = 0;
};

struct TraceRequest {
  bool enabled = false;
  bool statistics = false;
};

struct Timing {
  std::chrono::nanoseconds inputs{0};
  std::chrono::nanoseconds run{0};
  std::chrono::nanoseconds signals{0};
  std::chrono::nanoseconds outputs{0};
};

inline constexpr std::string_view kTraceRegion = "trace";
inline constexpr uint32_t kTraceRecordBytes = 16;

struct Program {
  uint64_t base = 0;
  std::vector<std::byte> image;
  uint32_t instruction_offset = 0;
  uint32_t instruction_count = 0;
  std::map<std::string, Stream, std::less<>> streams;
  std::map<std::string, std::string, std::less<>> listings;
  std::map<std::string, Region, std::less<>> regions;
  std::map<std::string, int64_t, std::less<>> parameters;
};

std::expected<Program, std::string> LoadProgram(
    const std::filesystem::path& directory);

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
      std::chrono::milliseconds timeout, const SignalHandler& on_signal = {},
      const TraceRequest& trace = {});
  [[nodiscard]] std::expected<Counters, std::string> Run(
      std::string_view stream, std::chrono::milliseconds timeout,
      const SignalHandler& on_signal = {}, const TraceRequest& trace = {});
  [[nodiscard]] std::expected<std::vector<std::byte>, std::string> ReadTrace(
      uint32_t count) const;
  [[nodiscard]] std::expected<Region, std::string> FindRegion(
      std::string_view name) const;
  [[nodiscard]] std::expected<int64_t, std::string> Parameter(
      std::string_view name) const;

 private:
  [[nodiscard]] std::expected<Region, std::string> Find(std::string_view region,
                                                        size_t size) const;
  [[nodiscard]] std::expected<TraceOptions, std::string> TraceOptionsFor(
      const TraceRequest& trace, uint32_t count) const;

  Device& device_;
  std::span<std::byte> memory_;
  uint64_t physical_address_;
  Program program_;
  bool loaded_ = false;
};

}  // namespace hotarubi

#endif  // HOTARUBI_RUNTIME_HPP_
