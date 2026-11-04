#include "output.hpp"

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

#include "hotarubi/json.hpp"

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
  const std::string meta = MetaJson(compiled) + "\n";
  return WriteFile(directory / "meta.json",
                   std::as_bytes(std::span<const char>(meta)));
}

}  // namespace hotarubi::compiler
