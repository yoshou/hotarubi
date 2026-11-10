#include "output.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <system_error>

#include "ddr_image.hpp"
#include "hotarubi/isa.hpp"
#include "hotarubi/json.hpp"
#include "hotarubi/require.hpp"
#include "program.hpp"

namespace hotarubi::compiler {
namespace {

Json Number(uint64_t value) { return Json(static_cast<double>(value)); }

std::expected<void, std::string> WriteFile(const std::filesystem::path& path,
                                           std::span<const std::byte> bytes) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!file) {
    return std::unexpected(std::format("cannot write {}", path.string()));
  }
  return {};
}

}  // namespace

uint64_t TraceRegionBytes(const std::vector<Stream>& streams) {
  uint64_t instructions = 0;
  for (const Stream& stream : streams) {
    instructions = std::max<uint64_t>(instructions, stream.program.Size());
  }
  return instructions * isa::trace::kRecordBytes * kTraceRecordsPerInstruction;
}

uint64_t ReserveTraceRegion(DdrImage& image,
                            const std::vector<Stream>& streams) {
  Require(!streams.empty(), "the trace region needs at least one stream");
  return image.Alloc(kTraceRegion, TraceRegionBytes(streams));
}

std::string ListingFileName(std::string_view stream) {
  return std::format("{}.listing.json", stream);
}

std::string MetaJson(const Compiled& compiled) {
  Json::Object regions;
  for (const auto& [name, region] : compiled.image.Regions()) {
    regions.emplace_back(
        name, Json(Json::Array{Number(region.offset), Number(region.size)}));
  }
  const Stream& first = compiled.streams.front();
  Json::Object meta = {
      {"base", Number(first.program.Base())},
      {"instr_off", Number(first.offset)},
      {"n_instr", Number(first.program.Size())},
      {"image_bytes", Number(compiled.image.End())},
  };
  for (const auto& [name, value] : compiled.parameters) {
    meta.emplace_back(name, Number(value));
  }
  if (compiled.streams.size() > 1) {
    Json::Object streams;
    for (const Stream& stream : compiled.streams) {
      streams.emplace_back(stream.name,
                           Json(Json::Array{Number(stream.offset),
                                            Number(stream.program.Size())}));
    }
    meta.emplace_back("streams", Json(std::move(streams)));
  }
  Json::Object listings;
  for (const Stream& stream : compiled.streams) {
    listings.emplace_back(stream.name, Json(ListingFileName(stream.name)));
  }
  meta.emplace_back("listings", Json(std::move(listings)));
  meta.emplace_back("regions", Json(std::move(regions)));
  return Json(std::move(meta)).Dump();
}

std::expected<void, std::string> WriteCompiled(
    const std::filesystem::path& directory, const Compiled& compiled) {
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  if (error) {
    return std::unexpected(std::format("cannot create {}: {}",
                                       directory.string(), error.message()));
  }
  if (auto written = WriteFile(directory / "image.bin", compiled.image.Bytes());
      !written) {
    return written;
  }
  for (const Stream& stream : compiled.streams) {
    const std::string listing = stream.program.ListingJson();
    if (auto written = WriteFile(directory / ListingFileName(stream.name),
                                 std::as_bytes(std::span<const char>(listing)));
        !written) {
      return written;
    }
  }
  const std::string meta = MetaJson(compiled) + "\n";
  return WriteFile(directory / "meta.json",
                   std::as_bytes(std::span<const char>(meta)));
}

}  // namespace hotarubi::compiler
