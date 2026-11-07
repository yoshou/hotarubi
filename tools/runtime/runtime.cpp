#include "hotarubi/runtime.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "hotarubi/driver.hpp"
#include "hotarubi/json.hpp"

namespace hotarubi {
namespace {

constexpr uint64_t kAddressLimit = uint64_t{1} << 32;
constexpr uint64_t kInstructionBytes = 16;
constexpr uint64_t kInstructionAlignment = 256;
constexpr uint64_t kBeatBytes = 16;
constexpr uint64_t kWordLimit = 0xFFFFFFFFU;
constexpr std::array<std::string_view, 4> kFields = {"base", "instr_off",
                                                     "n_instr", "image_bytes"};

std::expected<uint64_t, std::string> Field(const Json& meta,
                                           std::string_view name,
                                           uint64_t limit) {
  const Json* field = meta.Find(name);
  if (field == nullptr) {
    return std::unexpected(std::format("meta.json has no {}", name));
  }
  const auto value = field->Integer();
  if (!value || *value < 0 || std::cmp_greater(*value, limit)) {
    return std::unexpected(std::format("meta.json: bad {}", name));
  }
  return static_cast<uint64_t>(*value);
}

std::expected<std::string, std::string> ReadFile(
    const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return std::unexpected(std::format("cannot open {}", path.string()));
  }
  return std::string(std::istreambuf_iterator<char>(file),
                     std::istreambuf_iterator<char>());
}

bool StreamFits(const Program& program, uint64_t offset, uint64_t count) {
  return (program.base + offset) % kInstructionAlignment == 0 && count != 0 &&
         offset + (count * kInstructionBytes) <= program.image.size();
}

std::expected<std::pair<uint32_t, uint32_t>, std::string> ParsePair(
    const Json& value, std::string_view what) {
  if (!value.IsArray() || value.Items().size() != 2) {
    return std::unexpected(std::format("meta.json: bad {}", what));
  }
  const auto first = value.Items()[0].Integer();
  const auto second = value.Items()[1].Integer();
  if (!first || !second || *first < 0 || *second < 0 ||
      std::cmp_greater(*first + *second, kWordLimit)) {
    return std::unexpected(std::format("meta.json: bad {}", what));
  }
  return std::pair{static_cast<uint32_t>(*first),
                   static_cast<uint32_t>(*second)};
}

std::expected<std::map<std::string, Stream, std::less<>>, std::string>
ParseStreams(const Json& meta) {
  std::map<std::string, Stream, std::less<>> parsed;
  const Json* streams = meta.Find("streams");
  if (streams == nullptr) {
    return parsed;
  }
  if (!streams->IsObject()) {
    return std::unexpected("meta.json: streams is not an object");
  }
  for (const auto& [name, value] : streams->Members()) {
    const auto pair = ParsePair(value, std::format("stream {}", name));
    if (!pair) {
      return std::unexpected(pair.error());
    }
    parsed[name] = {.offset = pair->first, .count = pair->second};
  }
  return parsed;
}

std::expected<std::map<std::string, std::string, std::less<>>, std::string>
ParseListings(const Json& meta) {
  std::map<std::string, std::string, std::less<>> parsed;
  const Json* listings = meta.Find("listings");
  if (listings == nullptr) {
    return parsed;
  }
  if (!listings->IsObject()) {
    return std::unexpected("meta.json: listings is not an object");
  }
  for (const auto& [name, value] : listings->Members()) {
    if (!value.IsString()) {
      return std::unexpected(std::format("meta.json: bad listing {}", name));
    }
    parsed[name] = value.String();
  }
  return parsed;
}

std::map<std::string, int64_t, std::less<>> ParseParameters(const Json& meta) {
  std::map<std::string, int64_t, std::less<>> parsed;
  for (const auto& [name, value] : meta.Members()) {
    if (!value.IsNumber() || std::ranges::contains(kFields, name)) {
      continue;
    }
    if (const auto integer = value.Integer()) {
      parsed[name] = *integer;
    }
  }
  return parsed;
}

std::expected<std::map<std::string, Region, std::less<>>, std::string>
ParseRegions(const Json& meta) {
  const Json* regions = meta.Find("regions");
  if (regions == nullptr || !regions->IsObject()) {
    return std::unexpected("meta.json has no regions");
  }
  std::map<std::string, Region, std::less<>> parsed;
  for (const auto& [name, value] : regions->Members()) {
    const auto pair = ParsePair(value, std::format("region {}", name));
    if (!pair) {
      return std::unexpected(pair.error());
    }
    parsed[name] = {.offset = pair->first, .size = pair->second};
  }
  return parsed;
}

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
  if (!StreamFits(program, program.instruction_offset,
                  program.instruction_count)) {
    return std::unexpected(
        std::format("the instruction stream at offset {} with {} "
                    "instructions is not inside the image or not aligned",
                    program.instruction_offset, program.instruction_count));
  }
  for (const auto& [name, stream] : program.streams) {
    if (!StreamFits(program, stream.offset, stream.count)) {
      return std::unexpected(std::format(
          "instruction stream {} is not inside the image or not aligned",
          name));
    }
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

std::expected<TraceOptions, std::string> Runtime::TraceOptionsFor(
    const TraceRequest& trace, uint32_t count) const {
  if (!trace.enabled) {
    return TraceOptions{};
  }
  const auto region = FindRegion(kTraceRegion);
  if (!region) {
    return std::unexpected("the program has no trace region");
  }
  const uint64_t needed =
      uint64_t{count} * kTraceRecordBytes * (trace.statistics ? 2 : 1);
  if (region->size < needed) {
    return std::unexpected(
        std::format("the trace region holds {} bytes but {} instructions "
                    "need {}",
                    region->size, count, needed));
  }
  return TraceOptions{
      .base = static_cast<uint32_t>(physical_address_ + region->offset),
      .statistics = trace.statistics};
}

std::expected<Counters, std::string> Runtime::Run(
    std::chrono::milliseconds timeout, const SignalHandler& on_signal,
    const TraceRequest& trace) {
  if (!loaded_) {
    return std::unexpected("no program is loaded");
  }
  const auto options = TraceOptionsFor(trace, program_.instruction_count);
  if (!options) {
    return std::unexpected(options.error());
  }
  return device_.Run(
      static_cast<uint32_t>(program_.base + program_.instruction_offset),
      program_.instruction_count, timeout, on_signal, *options);
}

std::expected<Counters, std::string> Runtime::Run(
    std::string_view stream, std::chrono::milliseconds timeout,
    const SignalHandler& on_signal, const TraceRequest& trace) {
  if (!loaded_) {
    return std::unexpected("no program is loaded");
  }
  const auto it = program_.streams.find(stream);
  if (it == program_.streams.end()) {
    return std::unexpected(
        std::format("no instruction stream named {}", stream));
  }
  const auto options = TraceOptionsFor(trace, it->second.count);
  if (!options) {
    return std::unexpected(options.error());
  }
  return device_.Run(static_cast<uint32_t>(program_.base + it->second.offset),
                     it->second.count, timeout, on_signal, *options);
}

std::expected<std::vector<std::byte>, std::string> Runtime::ReadTrace(
    uint32_t count) const {
  const auto region = FindRegion(kTraceRegion);
  if (!region) {
    return std::unexpected(region.error());
  }
  const uint64_t bytes = uint64_t{count} * kTraceRecordBytes;
  if (bytes > region->size) {
    return std::unexpected(
        std::format("{} trace records exceed the trace region of {} bytes",
                    count, region->size));
  }
  const auto source = memory_.subspan(region->offset, bytes);
  return std::vector<std::byte>(source.begin(), source.end());
}

std::expected<Region, std::string> Runtime::FindRegion(
    std::string_view name) const {
  if (!loaded_) {
    return std::unexpected("no program is loaded");
  }
  const auto it = program_.regions.find(name);
  if (it == program_.regions.end()) {
    return std::unexpected(std::format("no region named {}", name));
  }
  return it->second;
}

std::expected<int64_t, std::string> Runtime::Parameter(
    std::string_view name) const {
  if (!loaded_) {
    return std::unexpected("no program is loaded");
  }
  const auto it = program_.parameters.find(name);
  if (it == program_.parameters.end()) {
    return std::unexpected(
        std::format("the program has no parameter {}", name));
  }
  return it->second;
}

std::expected<Program, std::string> LoadProgram(
    const std::filesystem::path& directory) {
  const auto text = ReadFile(directory / "meta.json");
  if (!text) {
    return std::unexpected(text.error());
  }
  const auto meta = Json::Parse(*text);
  if (!meta || !meta->IsObject()) {
    return std::unexpected(meta ? "meta.json is not an object" : meta.error());
  }
  const auto base = Field(*meta, "base", kAddressLimit - 1);
  const auto offset = Field(*meta, "instr_off", kWordLimit);
  const auto count = Field(*meta, "n_instr", kWordLimit);
  const auto bytes = Field(*meta, "image_bytes", kWordLimit);
  for (const auto* field : {&base, &offset, &count, &bytes}) {
    if (!*field) {
      return std::unexpected(field->error());
    }
  }
  auto streams = ParseStreams(*meta);
  if (!streams) {
    return std::unexpected(streams.error());
  }
  auto regions = ParseRegions(*meta);
  if (!regions) {
    return std::unexpected(regions.error());
  }
  auto listings = ParseListings(*meta);
  if (!listings) {
    return std::unexpected(listings.error());
  }
  const auto image = ReadFile(directory / "image.bin");
  if (!image) {
    return std::unexpected(image.error());
  }
  if (image->size() != *bytes) {
    return std::unexpected(std::format(
        "image.bin has {} bytes but meta.json says {}", image->size(), *bytes));
  }
  Program program{.base = *base,
                  .image = std::vector<std::byte>(image->size()),
                  .instruction_offset = static_cast<uint32_t>(*offset),
                  .instruction_count = static_cast<uint32_t>(*count),
                  .streams = std::move(*streams),
                  .listings = std::move(*listings),
                  .regions = std::move(*regions),
                  .parameters = ParseParameters(*meta)};
  std::ranges::transform(*image, program.image.begin(), [](char c) {
    return static_cast<std::byte>(static_cast<unsigned char>(c));
  });
  return program;
}

}  // namespace hotarubi
