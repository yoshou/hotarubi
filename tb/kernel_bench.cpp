#include "kernel_bench.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <map>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "hotarubi/driver.hpp"
#include "hotarubi/json.hpp"
#include "hotarubi/runtime.hpp"
#include "input.hpp"
#include "npu_bench.hpp"
#include "npu_isa.hpp"
#include "predict.hpp"

namespace hotarubi::tb {
namespace {

constexpr uint32_t kRegStatus = 0x04;
constexpr uint32_t kRegIssued = 0x20;
constexpr uint32_t kRegCycles = 0x24;
constexpr uint32_t kRegGemmCycles = 0x28;
constexpr uint32_t kRegVecCycles = 0x2C;
constexpr uint32_t kRegDmaCycles = 0x30;
constexpr uint32_t kRegStallCycles = 0x34;
constexpr uint32_t kRegStarveCycles = 0x38;
constexpr uint32_t kRegTraceBase = 0x3C;
constexpr uint32_t kRegTraceCtrl = 0x40;
constexpr uint32_t kRegTraceCount = 0x44;
constexpr uint32_t kStatusHalted = 0x2;
constexpr uint32_t kStatusError = 0x4;
constexpr uint32_t kTraceIdle = 0x2;
constexpr uint32_t kStreamAlignment = 256;
constexpr uint32_t kInstructionBytes = 16;
constexpr uint32_t kSlack = 4096;
constexpr std::array<std::string_view, 4> kEngineNames = {"DMA", "GEMM", "VEC",
                                                          "CTRL"};

Json Number(uint64_t value) { return Json(static_cast<double>(value)); }

std::expected<std::vector<trace::Instruction>, std::string> Select(
    const trace::Stream& stream, const KernelOptions& options) {
  if (options.first > options.last || options.last >= stream.count) {
    return std::unexpected(std::format(
        "the window {}..{} is not inside the {} instructions of stream {}",
        options.first, options.last, stream.count, stream.name));
  }
  std::vector<trace::Instruction> selected;
  for (const uint32_t index : options.preamble) {
    if (index >= stream.count) {
      return std::unexpected(
          std::format("preamble instruction {} is beyond the stream", index));
    }
    selected.push_back(stream.instructions.at(index));
  }
  for (uint32_t index = options.first; index <= options.last; ++index) {
    selected.push_back(stream.instructions.at(index));
  }
  const trace::Instruction& halt = stream.instructions.back();
  if (halt.opcode != kOpHalt) {
    return std::unexpected("the stream does not end with HALT");
  }
  selected.push_back(halt);
  return selected;
}

Instruction ToBench(const trace::Instruction& instruction) {
  Instruction out;
  out.words = {static_cast<uint32_t>(instruction.word.low),
               static_cast<uint32_t>(instruction.word.low >> 32),
               static_cast<uint32_t>(instruction.word.high),
               static_cast<uint32_t>(instruction.word.high >> 32)};
  return out;
}

Json Ranges(const std::vector<trace::Range>& ranges) {
  Json::Array items;
  for (const trace::Range& range : ranges) {
    items.emplace_back(Json::Array{Json(std::string(trace::MemName(range.mem))),
                                   Number(range.lo), Number(range.hi)});
  }
  return Json(std::move(items));
}

std::string ListingJson(uint64_t base,
                        const std::vector<trace::Instruction>& selected) {
  std::vector<std::string> tags;
  std::map<std::string, uint32_t> tag_index;
  Json::Array entries;
  for (size_t i = 0; i < selected.size(); ++i) {
    const trace::Instruction& instruction = selected[i];
    auto found = tag_index.find(instruction.tag);
    if (found == tag_index.end()) {
      found =
          tag_index.emplace(instruction.tag, static_cast<uint32_t>(tags.size()))
              .first;
      tags.push_back(instruction.tag);
    }
    entries.emplace_back(Json::Object{
        {"index", Number(i)},
        {"engine", Json(std::string(kEngineNames.at(
                       static_cast<size_t>(instruction.source))))},
        {"opcode", Number(instruction.opcode)},
        {"wait", Number(instruction.wait)},
        {"flags", Number(instruction.flags)},
        {"hex", Json(std::format("{:016x}{:016x}", instruction.word.high,
                                 instruction.word.low))},
        {"tag", Number(found->second)},
        {"note", Json(instruction.note)},
        {"reads", Ranges(instruction.reads)},
        {"writes", Ranges(instruction.writes)}});
  }
  Json::Array tag_array;
  for (const std::string& tag : tags) {
    tag_array.emplace_back(tag);
  }
  return Json(Json::Object{{"base", Number(base)},
                           {"tags", Json(std::move(tag_array))},
                           {"instructions", Json(std::move(entries))}})
             .Dump() +
         "\n";
}

std::string MetaJson(const Program& program, uint64_t stream_offset,
                     uint32_t count, uint64_t image_bytes) {
  Json::Object regions;
  for (const auto& [name, region] : program.regions) {
    regions.emplace_back(
        name, Json(Json::Array{Number(region.offset), Number(region.size)}));
  }
  const std::string listing = std::format("{}.listing.json", kKernelStream);
  return Json(Json::Object{
                  {"base", Number(program.base)},
                  {"instr_off", Number(stream_offset)},
                  {"n_instr", Number(count)},
                  {"image_bytes", Number(image_bytes)},
                  {"listings", Json(Json::Object{{std::string(kKernelStream),
                                                  Json(listing)}})},
                  {"regions", Json(std::move(regions))}})
             .Dump() +
         "\n";
}

std::expected<void, std::string> WriteBytes(const std::filesystem::path& path,
                                            std::span<const uint8_t> bytes) {
  return trace::WriteText(
      path, std::string_view(reinterpret_cast<const char*>(bytes.data()),
                             bytes.size()));
}

}  // namespace

std::expected<Counters, std::string> RunKernel(const KernelOptions& options) {
  const auto loaded = trace::LoadCompiled(options.program);
  if (!loaded) {
    return std::unexpected(loaded.error());
  }
  const Program& program = loaded->program;
  const auto stream = trace::LoadStream(*loaded, options.stream);
  if (!stream) {
    return std::unexpected(stream.error());
  }
  const auto selected = Select(*stream, options);
  if (!selected) {
    return std::unexpected(selected.error());
  }
  const auto trace_region = program.regions.find(kTraceRegion);
  if (trace_region == program.regions.end()) {
    return std::unexpected("the program has no trace region");
  }
  const auto count = static_cast<uint32_t>(selected->size());
  const uint64_t needed =
      uint64_t{count} * kTraceRecordBytes * (options.statistics ? 2 : 1);
  if (trace_region->second.size < needed) {
    return std::unexpected(
        std::format("the trace region holds {} bytes but the kernel needs {}",
                    trace_region->second.size, needed));
  }
  const uint64_t stream_offset = (program.image.size() + kStreamAlignment - 1) /
                                 kStreamAlignment * kStreamAlignment;
  const uint64_t size =
      stream_offset + (uint64_t{count} * kInstructionBytes) + kSlack;
  if (program.base + size > (uint64_t{1} << 32)) {
    return std::unexpected("the image and the kernel stream exceed 32 bits");
  }
  NpuBench bench(options.seed, static_cast<uint32_t>(program.base),
                 static_cast<uint32_t>(size));
  std::span<uint8_t> memory = bench.Memory().Bytes();
  for (size_t i = 0; i < program.image.size(); ++i) {
    memory[i] = std::to_integer<uint8_t>(program.image[i]);
  }
  std::vector<Instruction> instructions;
  instructions.reserve(count);
  for (const trace::Instruction& instruction : *selected) {
    instructions.push_back(ToBench(instruction));
  }
  const auto stream_address =
      static_cast<uint32_t>(program.base + stream_offset);
  bench.LoadProgram(instructions, stream_address);
  std::error_code error;
  std::filesystem::create_directories(options.out, error);
  if (error) {
    return std::unexpected(std::format("cannot create {}: {}",
                                       options.out.string(), error.message()));
  }
  if (auto written = WriteBytes(options.out / "image.bin", memory); !written) {
    return std::unexpected(written.error());
  }
  if (auto written =
          trace::WriteText(options.out / "meta.json",
                           MetaJson(program, stream_offset, count, size));
      !written) {
    return std::unexpected(written.error());
  }
  if (auto written = trace::WriteText(
          options.out / std::format("{}.listing.json", kKernelStream),
          ListingJson(program.base, *selected));
      !written) {
    return std::unexpected(written.error());
  }
  const auto trace_base =
      static_cast<uint32_t>(program.base + trace_region->second.offset);
  bench.WriteReg(kRegTraceBase, trace_base);
  bench.WriteReg(kRegTraceCtrl, options.statistics ? 5U : 1U);
  bench.Start(stream_address, count);
  const bool done = bench.WaitStatus(
      [](uint32_t status) {
        return (status & (kStatusHalted | kStatusError)) != 0;
      },
      options.max_cycles);
  if (!done) {
    return std::unexpected(std::format(
        "the kernel did not halt within {} cycles", options.max_cycles));
  }
  if ((bench.ReadReg(kRegStatus) & kStatusError) != 0) {
    return std::unexpected("the kernel stopped with an error");
  }
  if (!bench.WaitStatus(
          [&](uint32_t) {
            return (bench.ReadReg(kRegTraceCtrl) & kTraceIdle) != 0;
          },
          options.max_cycles)) {
    return std::unexpected("the trace did not drain");
  }
  bench.WriteReg(kRegTraceCtrl, 0);
  const std::vector<std::string> errors = bench.Errors();
  if (!errors.empty()) {
    return std::unexpected(std::format("AXI protocol error: {}", errors[0]));
  }
  const Counters counters{.issued = bench.ReadReg(kRegIssued),
                          .cycles = bench.ReadReg(kRegCycles),
                          .gemm_cycles = bench.ReadReg(kRegGemmCycles),
                          .vec_cycles = bench.ReadReg(kRegVecCycles),
                          .dma_cycles = bench.ReadReg(kRegDmaCycles),
                          .stall_cycles = bench.ReadReg(kRegStallCycles),
                          .starve_cycles = bench.ReadReg(kRegStarveCycles),
                          .trace_count = bench.ReadReg(kRegTraceCount)};
  const uint32_t most =
      options.statistics ? 2 * counters.issued : counters.issued;
  if (counters.trace_count < counters.issued || counters.trace_count > most) {
    return std::unexpected(
        std::format("the trace holds {} records for {} instructions",
                    counters.trace_count, counters.issued));
  }
  const std::span<const uint8_t> records =
      memory.subspan(trace_region->second.offset,
                     size_t{counters.trace_count} * kTraceRecordBytes);
  if (auto written = WriteBytes(
          options.out / std::format("{}.trace.bin", kKernelStream), records);
      !written) {
    return std::unexpected(written.error());
  }
  if (auto written = trace::WriteText(
          options.out / std::format("{}.counters.json", kKernelStream),
          trace::CountersJson(kKernelStream, options.statistics, counters,
                              "rtl"));
      !written) {
    return std::unexpected(written.error());
  }
  return counters;
}

}  // namespace hotarubi::tb
