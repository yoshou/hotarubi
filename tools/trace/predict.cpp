#include "predict.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hotarubi/driver.hpp"
#include "hotarubi/json.hpp"
#include "hotarubi/runtime.hpp"
#include "input.hpp"
#include "machine.hpp"
#include "memories.hpp"
#include "timing.hpp"

namespace hotarubi::trace {
namespace {

constexpr uint32_t kRegCtrl = 0x00;
constexpr uint32_t kRegInstrBase = 0x08;
constexpr uint32_t kRegInstrCount = 0x0C;
constexpr uint32_t kRegSignalValue = 0x1C;
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
constexpr uint32_t kRegHostGo = 0x48;

Json Number(uint64_t value) { return Json(static_cast<double>(value)); }

std::string_view StopName(iss::Stop stop) {
  switch (stop) {
    case iss::Stop::kIllegal:
      return "illegal instruction";
    case iss::Stop::kWaitHost:
      return "waiting for a host value that is never signalled";
    case iss::Stop::kStarved:
      return "end of the stream";
    default:
      return "idle";
  }
}

constexpr uint32_t kRomWords = 256;
constexpr size_t kReportLimit = 10;

iss::AccessSet Declared(const std::vector<Range>& ranges) {
  iss::AccessSet set;
  for (const Range& range : ranges) {
    set.AddRange(range.mem, range.lo, range.hi);
  }
  return set;
}

std::string ExtentsJson(const iss::AccessSet& set) {
  std::string text = "[";
  bool first = true;
  for (const iss::Extent& extent : iss::Extents(set)) {
    text += std::format("{}[\"{}\",{},{}]", first ? "" : ",",
                        MemName(extent.mem), extent.lo, extent.hi);
    first = false;
  }
  return text + "]";
}

class Coverage {
 public:
  explicit Coverage(const Stream& stream, bool accesses)
      : stream_(stream), accesses_(accesses) {}

  void Observe(uint32_t index, const iss::AccessLog& log) {
    if (index >= stream_.instructions.size()) {
      return;
    }
    const Instruction& instruction = stream_.instructions[index];
    iss::AccessSet reads = log.reads;
    for (uint32_t word = 0; word < kRomWords; ++word) {
      reads.cbuf.reset(word);
    }
    Check(index, "reads", reads, Declared(instruction.reads));
    Check(index, "writes", log.writes, Declared(instruction.writes));
    if (accesses_) {
      accesses_text_ +=
          std::format("{{\"index\":{},\"reads\":{},\"writes\":{}}}\n", index,
                      ExtentsJson(log.reads), ExtentsJson(log.writes));
    }
  }

  [[nodiscard]] size_t Violations() const { return violations_; }
  [[nodiscard]] const std::string& Report() const { return report_; }
  [[nodiscard]] std::string TakeAccesses() { return std::move(accesses_text_); }

 private:
  void Check(uint32_t index, std::string_view kind,
             const iss::AccessSet& actual, const iss::AccessSet& declared) {
    const auto outside = iss::FirstOutside(actual, declared);
    if (!outside) {
      return;
    }
    if (violations_ < kReportLimit) {
      report_ += std::format(
          "instruction {} {} {} word {} outside its listed ranges ({})\n",
          index, kind, MemName(outside->mem), outside->address,
          stream_.instructions[index].note);
    }
    ++violations_;
  }

  const Stream& stream_;
  bool accesses_;
  size_t violations_ = 0;
  std::string report_;
  std::string accesses_text_;
};

}  // namespace

std::expected<Prediction, std::string> Predict(
    const LoadedProgram& loaded, std::string_view stream, bool statistics,
    const iss::TimingParameters& parameters, bool accesses) {
  const auto loaded_stream = LoadStream(loaded, stream);
  if (!loaded_stream) {
    return std::unexpected(loaded_stream.error());
  }
  const Program& program = loaded.program;
  const auto region = program.regions.find(kTraceRegion);
  if (region == program.regions.end()) {
    return std::unexpected("the program has no trace region");
  }
  const uint64_t needed =
      uint64_t{loaded_stream->count} * kTraceRecordBytes * (statistics ? 2 : 1);
  if (region->second.size < needed) {
    return std::unexpected(
        std::format("the trace region holds {} bytes but {} are needed",
                    region->second.size, needed));
  }
  std::vector<std::byte> memory = program.image;
  iss::Machine machine(memory, program.base, parameters);
  Coverage coverage(*loaded_stream, accesses);
  machine.Observe([&coverage](uint32_t index, const iss::AccessLog& log) {
    coverage.Observe(index, log);
  });
  machine.WriteRegister(
      kRegInstrBase,
      static_cast<uint32_t>(program.base + loaded_stream->offset));
  machine.WriteRegister(kRegInstrCount, loaded_stream->count);
  machine.WriteRegister(
      kRegTraceBase,
      static_cast<uint32_t>(program.base + region->second.offset));
  machine.WriteRegister(kRegTraceCtrl, statistics ? 5U : 1U);
  machine.WriteRegister(kRegCtrl, 1);
  while (true) {
    const iss::Stop stop = machine.Run();
    if (stop == iss::Stop::kHalt) {
      break;
    }
    if (stop == iss::Stop::kSignal) {
      machine.WriteRegister(kRegHostGo, machine.ReadRegister(kRegSignalValue));
      continue;
    }
    return std::unexpected(std::format(
        "the stream stopped after {} instructions without halting ({})",
        machine.ReadRegister(kRegIssued), StopName(stop)));
  }
  if (!machine.Hazards().empty()) {
    return std::unexpected(std::format("the stream has unordered accesses:\n{}",
                                       DescribeHazards(machine.Hazards(), 10)));
  }
  if (coverage.Violations() != 0) {
    return std::unexpected(std::format(
        "the listing does not cover the accesses of {} instructions:\n{}",
        coverage.Violations(), coverage.Report()));
  }
  const uint32_t count = machine.ReadRegister(kRegTraceCount);
  const auto first =
      memory.begin() + static_cast<std::ptrdiff_t>(region->second.offset);
  return Prediction{
      .records = std::vector<std::byte>(
          first, first + static_cast<std::ptrdiff_t>(uint64_t{count} *
                                                     kTraceRecordBytes)),
      .counters = {.issued = machine.ReadRegister(kRegIssued),
                   .cycles = machine.ReadRegister(kRegCycles),
                   .gemm_cycles = machine.ReadRegister(kRegGemmCycles),
                   .vec_cycles = machine.ReadRegister(kRegVecCycles),
                   .dma_cycles = machine.ReadRegister(kRegDmaCycles),
                   .stall_cycles = machine.ReadRegister(kRegStallCycles),
                   .starve_cycles = machine.ReadRegister(kRegStarveCycles),
                   .trace_count = count},
      .accesses = coverage.TakeAccesses()};
}

std::string CountersJson(std::string_view stream, bool statistics,
                         const Counters& c, std::string_view origin) {
  const Json root(Json::Object{
      {"stream", Json(std::string(stream))},
      {"trace_statistics", Json(statistics)},
      {"counters", Json(Json::Object{{"issued", Number(c.issued)},
                                     {"cycles", Number(c.cycles)},
                                     {"gemm_cycles", Number(c.gemm_cycles)},
                                     {"vec_cycles", Number(c.vec_cycles)},
                                     {"dma_cycles", Number(c.dma_cycles)},
                                     {"stall_cycles", Number(c.stall_cycles)},
                                     {"starve_cycles", Number(c.starve_cycles)},
                                     {"trace_count", Number(c.trace_count)}})},
      {"origin", Json(std::string(origin))}});
  return root.Dump() + "\n";
}

}  // namespace hotarubi::trace
