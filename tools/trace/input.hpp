#ifndef HOTARUBI_TOOLS_TRACE_INPUT_HPP_
#define HOTARUBI_TOOLS_TRACE_INPUT_HPP_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "hotarubi/driver.hpp"
#include "hotarubi/json.hpp"
#include "hotarubi/runtime.hpp"
#include "trace_record.hpp"
#include "word.hpp"

namespace hotarubi::trace {

struct Instruction {
  uint32_t index = 0;
  iss::Word128 word;
  uint32_t opcode = 0;
  iss::Source source = iss::Source::kSequencer;
  uint32_t wait = 0;
  uint32_t flags = 0;
  std::string tag;
  std::string note;
};

struct Stream {
  std::string name;
  uint32_t offset = 0;
  uint32_t count = 0;
  std::vector<Instruction> instructions;
};

struct LoadedProgram {
  Program program;
  std::filesystem::path directory;
};

struct CountersFile {
  std::string stream;
  bool statistics = false;
  Counters counters;
};

std::expected<LoadedProgram, std::string> LoadCompiled(
    const std::filesystem::path& directory);
std::expected<Stream, std::string> LoadStream(const LoadedProgram& loaded,
                                              std::string_view name);
std::expected<std::vector<iss::TraceRecord>, std::string> LoadRecords(
    const std::filesystem::path& path);
std::vector<iss::TraceRecord> DecodeRecords(std::span<const std::byte> bytes);
std::expected<CountersFile, std::string> LoadCounters(
    const std::filesystem::path& path);
std::expected<std::string, std::string> ReadText(
    const std::filesystem::path& path);
std::expected<void, std::string> WriteText(const std::filesystem::path& path,
                                           std::string_view text);

}  // namespace hotarubi::trace

#endif  // HOTARUBI_TOOLS_TRACE_INPUT_HPP_
