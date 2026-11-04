#include "hotarubi/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "hotarubi/driver.hpp"

namespace hotarubi {
namespace {

constexpr uint64_t kAddressLimit = uint64_t{1} << 32;
constexpr uint64_t kInstructionBytes = 16;
constexpr uint64_t kInstructionAlignment = 256;
constexpr uint64_t kBeatBytes = 16;

}  // namespace

std::expected<void, std::string> Runtime::Load(Program program) {
  loaded_ = false;
  if (physical_address_ + memory_.size() > kAddressLimit) {
    return std::unexpected(
        "the shared memory is not inside the 32-bit address space");
  }
  if (program.base != physical_address_) {
    return std::unexpected(std::format(
        "the program was built for {:#x} but the shared memory is at {:#x}",
        program.base, physical_address_));
  }
  if (program.image.size() > memory_.size()) {
    return std::unexpected(
        std::format("the image has {} bytes but the shared memory has {}",
                    program.image.size(), memory_.size()));
  }
  const uint64_t stream_end =
      uint64_t{program.instruction_offset} +
      (uint64_t{program.instruction_count} * kInstructionBytes);
  if ((program.base + program.instruction_offset) % kInstructionAlignment !=
          0 ||
      program.instruction_count == 0 || stream_end > program.image.size()) {
    return std::unexpected(
        std::format("the instruction stream at offset {} with {} "
                    "instructions is not inside the image or not aligned",
                    program.instruction_offset, program.instruction_count));
  }
  for (const auto& [name, region] : program.regions) {
    if (region.offset % kBeatBytes != 0 ||
        uint64_t{region.offset} + region.size > memory_.size()) {
      return std::unexpected(std::format(
          "region {} at offset {} with {} bytes is not aligned or not inside "
          "the shared memory",
          name, region.offset, region.size));
    }
  }
  std::ranges::copy(program.image, memory_.begin());
  program_ = std::move(program);
  loaded_ = true;
  return {};
}

std::expected<Region, std::string> Runtime::Find(std::string_view region,
                                                 size_t size) const {
  if (!loaded_) {
    return std::unexpected("no program is loaded");
  }
  const auto it = program_.regions.find(region);
  if (it == program_.regions.end()) {
    return std::unexpected(std::format("no region named {}", region));
  }
  if (it->second.size != size) {
    return std::unexpected(std::format("region {} has {} bytes, not {}", region,
                                       it->second.size, size));
  }
  return it->second;
}

std::expected<void, std::string> Runtime::Write(
    std::string_view region, std::span<const std::byte> data) {
  const auto found = Find(region, data.size());
  if (!found) {
    return std::unexpected(found.error());
  }
  std::ranges::copy(data, memory_.subspan(found->offset, found->size).begin());
  return {};
}

std::expected<void, std::string> Runtime::Read(
    std::string_view region, std::span<std::byte> data) const {
  const auto found = Find(region, data.size());
  if (!found) {
    return std::unexpected(found.error());
  }
  std::ranges::copy(memory_.subspan(found->offset, found->size), data.begin());
  return {};
}

std::expected<Counters, std::string> Runtime::Run(
    std::chrono::milliseconds timeout, const SignalHandler& on_signal) {
  if (!loaded_) {
    return std::unexpected("no program is loaded");
  }
  return device_.Run(
      static_cast<uint32_t>(program_.base + program_.instruction_offset),
      program_.instruction_count, timeout, on_signal);
}

}  // namespace hotarubi
