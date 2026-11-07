#include "input.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hotarubi/driver.hpp"
#include "hotarubi/json.hpp"
#include "hotarubi/runtime.hpp"
#include "trace_record.hpp"
#include "word.hpp"

namespace hotarubi::trace {
namespace {

constexpr uint32_t kInstructionBytes = 16;

std::expected<std::vector<std::byte>, std::string> ReadBytes(
    const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return std::unexpected(std::format("cannot open {}", path.string()));
  }
  std::vector<char> text((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
  std::vector<std::byte> bytes(text.size());
  for (size_t i = 0; i < text.size(); ++i) {
    bytes[i] = static_cast<std::byte>(static_cast<unsigned char>(text[i]));
  }
  return bytes;
}

constexpr std::array<std::string_view, 6> kMemNames = {"ABUF", "WBUF", "FBUF",
                                                       "PBUF", "TBUF", "CBUF"};

std::optional<Range> ParseRange(const Json& item) {
  if (!item.IsArray() || item.Items().size() != 3) {
    return std::nullopt;
  }
  const Json& name = item.Items()[0];
  const Json& lo = item.Items()[1];
  const Json& hi = item.Items()[2];
  if (!name.IsString() || !lo.Integer() || !hi.Integer()) {
    return std::nullopt;
  }
  const auto mem = ParseMem(name.String());
  if (!mem || *lo.Integer() < 0 || *hi.Integer() < *lo.Integer()) {
    return std::nullopt;
  }
  return Range{.mem = *mem,
               .lo = static_cast<uint32_t>(*lo.Integer()),
               .hi = static_cast<uint32_t>(*hi.Integer())};
}

bool ParseRanges(const Json& entry, std::string_view name,
                 std::vector<Range>& ranges) {
  const Json* list = entry.Find(name);
  if (list == nullptr) {
    return true;
  }
  if (!list->IsArray()) {
    return false;
  }
  for (const Json& item : list->Items()) {
    const auto range = ParseRange(item);
    if (!range) {
      return false;
    }
    ranges.push_back(*range);
  }
  return true;
}

std::expected<void, std::string> ApplyListing(
    const std::filesystem::path& path, std::vector<Instruction>& instructions) {
  const auto text = ReadText(path);
  if (!text) {
    return {};
  }
  const auto listing = Json::Parse(*text);
  if (!listing || !listing->IsObject()) {
    return std::unexpected(
        std::format("{} is not a JSON object", path.string()));
  }
  const Json* tags = listing->Find("tags");
  const Json* entries = listing->Find("instructions");
  if (tags == nullptr || !tags->IsArray() || entries == nullptr ||
      !entries->IsArray() || entries->Items().size() != instructions.size()) {
    return std::unexpected(std::format("{} does not describe {} instructions",
                                       path.string(), instructions.size()));
  }
  for (size_t i = 0; i < instructions.size(); ++i) {
    const Json& entry = entries->Items()[i];
    const Json* hex = entry.Find("hex");
    const Json* tag = entry.Find("tag");
    const Json* note = entry.Find("note");
    if (hex == nullptr || !hex->IsString() || tag == nullptr ||
        !tag->Integer() || note == nullptr || !note->IsString()) {
      return std::unexpected(std::format("{}: bad entry {}", path.string(), i));
    }
    const auto tag_index = static_cast<size_t>(*tag->Integer());
    if (tag_index >= tags->Items().size()) {
      return std::unexpected(
          std::format("{}: bad tag of entry {}", path.string(), i));
    }
    const std::string expected = std::format(
        "{:016x}{:016x}", instructions[i].word.high, instructions[i].word.low);
    if (hex->String() != expected) {
      return std::unexpected(std::format(
          "{}: instruction {} is {} in the listing but {} in the image",
          path.string(), i, hex->String(), expected));
    }
    instructions[i].tag = tags->Items()[tag_index].String();
    instructions[i].note = note->String();
    for (const auto& [name, ranges] :
         {std::pair{"reads", &instructions[i].reads},
          std::pair{"writes", &instructions[i].writes}}) {
      if (!ParseRanges(entry, name, *ranges)) {
        return std::unexpected(
            std::format("{}: bad {} of entry {}", path.string(), name, i));
      }
    }
  }
  return {};
}

std::expected<uint32_t, std::string> Field(const Json& object,
                                           std::string_view name) {
  const Json* value = object.Find(name);
  if (value == nullptr || !value->Integer() || *value->Integer() < 0 ||
      std::cmp_greater(*value->Integer(), UINT32_MAX)) {
    return std::unexpected(std::format("counters.json has no {}", name));
  }
  return static_cast<uint32_t>(*value->Integer());
}

}  // namespace

std::string_view MemName(iss::Mem mem) {
  return kMemNames.at(static_cast<size_t>(mem));
}

std::optional<iss::Mem> ParseMem(std::string_view name) {
  for (size_t i = 0; i < kMemNames.size(); ++i) {
    if (kMemNames[i] == name) {
      return static_cast<iss::Mem>(i);
    }
  }
  return std::nullopt;
}

std::expected<std::string, std::string> ReadText(
    const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return std::unexpected(std::format("cannot open {}", path.string()));
  }
  return std::string(std::istreambuf_iterator<char>(file),
                     std::istreambuf_iterator<char>());
}

std::expected<void, std::string> WriteText(const std::filesystem::path& path,
                                           std::string_view text) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(text.data(), static_cast<std::streamsize>(text.size()));
  if (!file) {
    return std::unexpected(std::format("cannot write {}", path.string()));
  }
  return {};
}

std::expected<LoadedProgram, std::string> LoadCompiled(
    const std::filesystem::path& directory) {
  auto program = LoadProgram(directory);
  if (!program) {
    return std::unexpected(program.error());
  }
  return LoadedProgram{.program = std::move(*program), .directory = directory};
}

std::expected<Stream, std::string> LoadStream(const LoadedProgram& loaded,
                                              std::string_view name) {
  const Program& program = loaded.program;
  Stream stream{.name = std::string(name),
                .offset = program.instruction_offset,
                .count = program.instruction_count,
                .instructions = {}};
  if (!program.streams.empty()) {
    const auto found = program.streams.find(name);
    if (found == program.streams.end()) {
      return std::unexpected(
          std::format("the program has no stream named {}", name));
    }
    stream.offset = found->second.offset;
    stream.count = found->second.count;
  } else if (!program.listings.empty() && !program.listings.contains(name)) {
    return std::unexpected(
        std::format("the program has no stream named {}", name));
  }
  std::filesystem::path listing =
      loaded.directory / std::format("{}.listing.json", name);
  if (const auto found = program.listings.find(name);
      found != program.listings.end()) {
    listing = loaded.directory / found->second;
  }
  stream.instructions.reserve(stream.count);
  for (uint32_t i = 0; i < stream.count; ++i) {
    std::array<std::byte, kInstructionBytes> bytes{};
    const size_t at = size_t{stream.offset} + (size_t{i} * kInstructionBytes);
    for (size_t j = 0; j < kInstructionBytes; ++j) {
      bytes.at(j) = program.image.at(at + j);
    }
    const iss::Word128 word = iss::LoadWord128(bytes);
    const uint32_t opcode = word.Field(120, 8);
    stream.instructions.push_back({.index = i,
                                   .word = word,
                                   .opcode = opcode,
                                   .source = iss::SourceOf(opcode),
                                   .wait = word.Field(116, 4),
                                   .flags = word.Field(112, 4),
                                   .tag = {},
                                   .note = {},
                                   .reads = {},
                                   .writes = {}});
  }
  if (auto applied = ApplyListing(listing, stream.instructions); !applied) {
    return std::unexpected(applied.error());
  }
  return stream;
}

std::vector<iss::TraceRecord> DecodeRecords(std::span<const std::byte> bytes) {
  std::vector<iss::TraceRecord> records;
  records.reserve(bytes.size() / iss::kTraceRecordBytes);
  for (size_t at = 0; at + iss::kTraceRecordBytes <= bytes.size();
       at += iss::kTraceRecordBytes) {
    records.push_back(iss::DecodeRecord(
        iss::LoadWord128(bytes.subspan(at).first<iss::kTraceRecordBytes>())));
  }
  return records;
}

std::expected<std::vector<iss::TraceRecord>, std::string> LoadRecords(
    const std::filesystem::path& path) {
  const auto bytes = ReadBytes(path);
  if (!bytes) {
    return std::unexpected(bytes.error());
  }
  if (bytes->size() % iss::kTraceRecordBytes != 0) {
    return std::unexpected(
        std::format("{} is not a whole number of records", path.string()));
  }
  return DecodeRecords(*bytes);
}

std::expected<CountersFile, std::string> LoadCounters(
    const std::filesystem::path& path) {
  const auto text = ReadText(path);
  if (!text) {
    return std::unexpected(text.error());
  }
  const auto json = Json::Parse(*text);
  if (!json || !json->IsObject()) {
    return std::unexpected(
        std::format("{} is not a JSON object", path.string()));
  }
  const Json* counters = json->Find("counters");
  const Json* stream = json->Find("stream");
  const Json* statistics = json->Find("trace_statistics");
  if (counters == nullptr || !counters->IsObject() || stream == nullptr ||
      !stream->IsString() || statistics == nullptr || !statistics->IsBool()) {
    return std::unexpected(std::format(
        "{} lacks stream, trace_statistics or counters", path.string()));
  }
  CountersFile file{.stream = stream->String(),
                    .statistics = statistics->Bool(),
                    .counters = {}};
  const std::array<std::pair<std::string_view, uint32_t*>, 8> fields = {
      {{"issued", &file.counters.issued},
       {"cycles", &file.counters.cycles},
       {"gemm_cycles", &file.counters.gemm_cycles},
       {"vec_cycles", &file.counters.vec_cycles},
       {"dma_cycles", &file.counters.dma_cycles},
       {"stall_cycles", &file.counters.stall_cycles},
       {"starve_cycles", &file.counters.starve_cycles},
       {"trace_count", &file.counters.trace_count}}};
  for (const auto& [name, out] : fields) {
    const auto value = Field(*counters, name);
    if (!value) {
      return std::unexpected(value.error());
    }
    *out = *value;
  }
  return file;
}

}  // namespace hotarubi::trace
