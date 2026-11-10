#include "analysis.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hotarubi/driver.hpp"
#include "input.hpp"
#include "machine.hpp"
#include "timing.hpp"
#include "trace_record.hpp"

namespace hotarubi::trace {
namespace {

constexpr uint64_t kWrap = uint64_t{1} << 32;
constexpr std::array<uint32_t, kEngines> kWaitBits = {0x1, 0x4, 0x8};
constexpr std::array<std::string_view, 6> kMemNames = {"ABUF", "WBUF", "FBUF",
                                                       "PBUF", "TBUF", "CBUF"};
constexpr std::array<std::string_view, 4> kDmaModes = {"", " SPLIT8",
                                                       " T32_SPLIT8", " GACC"};
constexpr std::array<std::string_view, 3> kWeightModes = {"FP4", "W8_HI",
                                                          "W8_LO"};
constexpr std::array<std::string_view, 17> kVecOps = {
    "ADD",       "MUL",     "ADDP", "MULP", "SCALE",    "COPY",
    "LAYERNORM", "RMSNORM", "GELU", "SILU", "SILU_MUL", "SOFTMAX",
    "ROPE",      "QUANT",   "RELU", "MAX",  "RUN"};

constexpr uint32_t kOpNop = 0;
constexpr uint32_t kOpDmaLoad = 1;
constexpr uint32_t kOpDmaStore = 2;
constexpr uint32_t kOpGemm = 4;
constexpr uint32_t kOpVec = 5;
constexpr uint32_t kOpSignal = 6;
constexpr uint32_t kOpHalt = 7;
constexpr uint32_t kOpWaitHost = 8;

std::string Mask(const std::array<bool, kEngines>& active) {
  std::string name;
  for (int e = 0; e < kEngines; ++e) {
    if (active.at(e)) {
      name += std::format("{}{}", name.empty() ? "" : "+", kSourceNames.at(e));
    }
  }
  return name.empty() ? "Idle" : name;
}

void Add(Aggregate& aggregate, const Event& event) {
  ++aggregate.count;
  aggregate.span += event.Duration();
  aggregate.exclusive += event.exclusive;
  aggregate.streaming += event.streaming;
  aggregate.model += event.model;
  if (event.statistics) {
    ++aggregate.statistics_count;
    aggregate.statistics.a += event.statistics->a;
    aggregate.statistics.b += event.statistics->b;
    aggregate.statistics.c += event.statistics->c;
    aggregate.statistics.d += event.statistics->d;
  }
  if (event.IsEngine()) {
    aggregate.busy.at(static_cast<size_t>(event.source)) += event.Duration();
  }
  aggregate.first = std::min(aggregate.first, event.issue);
  aggregate.last = std::max(aggregate.last, event.completion);
}

struct Unwrapped {
  std::vector<Event> events;
  uint64_t total = 0;
};

std::expected<Unwrapped, std::string> Unwrap(
    const Stream& stream, const std::vector<iss::TraceRecord>& records,
    const iss::CostModel& model) {
  Unwrapped out;
  out.events.resize(stream.count);
  std::vector<bool> seen(stream.count, false);
  for (const iss::TraceRecord& record : records) {
    if (record.index >= stream.count) {
      return std::unexpected(std::format(
          "record index {} is beyond the {} instructions of the stream",
          record.index, stream.count));
    }
    const Instruction& instruction = stream.instructions[record.index];
    Event& event = out.events[record.index];
    if (record.type == iss::RecordType::kStatistics) {
      if (record.source != instruction.source) {
        return std::unexpected(
            std::format("statistics record {} names source {}", record.index,
                        static_cast<int>(record.source)));
      }
      event.statistics = record.statistics;
      continue;
    }
    if (seen[record.index]) {
      return std::unexpected(std::format(
          "instruction {} has two completion records", record.index));
    }
    seen[record.index] = true;
    if (record.opcode != instruction.opcode ||
        record.wait != instruction.wait || record.flags != instruction.flags ||
        record.source != instruction.source) {
      return std::unexpected(std::format(
          "record {} (opcode {}, wait {:#x}, flags {:#x}) does not match "
          "the instruction (opcode {}, wait {:#x}, flags {:#x})",
          record.index, record.opcode, record.wait, record.flags,
          instruction.opcode, instruction.wait, instruction.flags));
    }
    event.index = record.index;
    event.issue = record.issue;
    event.completion = record.completion;
    event.source = instruction.source;
    event.opcode = instruction.opcode;
    event.wait = instruction.wait;
    event.flags = instruction.flags;
    event.category = Category(instruction);
    event.tag = instruction.tag;
    event.note = instruction.note;
    event.streaming = Streaming(instruction);
    event.model = Model(model, instruction);
  }
  for (uint32_t i = 0; i < stream.count; ++i) {
    if (!seen[i]) {
      return std::unexpected(
          std::format("instruction {} has no completion record", i));
    }
  }
  uint64_t epoch = 0;
  uint64_t last = 0;
  for (Event& event : out.events) {
    const uint64_t issue = event.issue;
    if (issue < last) {
      epoch += kWrap;
    }
    last = issue;
    const uint64_t duration = (event.completion - issue) % kWrap;
    event.issue = issue + epoch;
    event.completion = event.issue + duration;
    out.total = std::max(out.total, event.completion + 1);
  }
  return out;
}

std::expected<void, std::string> CheckOrder(const std::vector<Event>& events) {
  std::array<uint64_t, kEngines> end{};
  uint64_t previous = 0;
  for (size_t i = 0; i < events.size(); ++i) {
    const Event& event = events[i];
    if (i > 0 && event.issue <= previous) {
      return std::unexpected(std::format(
          "instruction {} issued at cycle {} before or with instruction {}",
          event.index, event.issue, events[i - 1].index));
    }
    previous = event.issue;
    if (!event.IsEngine()) {
      if (event.completion != event.issue) {
        return std::unexpected(std::format(
            "sequencer instruction {} has a completion after its issue",
            event.index));
      }
      continue;
    }
    const auto engine = static_cast<size_t>(event.source);
    if (event.issue < end.at(engine)) {
      return std::unexpected(std::format(
          "instruction {} overlaps the previous command of its engine",
          event.index));
    }
    end.at(engine) = event.completion;
  }
  return {};
}

std::expected<void, std::string> CheckCounters(const std::vector<Event>& events,
                                               uint64_t total,
                                               const CountersFile& counters,
                                               size_t records) {
  if (counters.counters.issued != events.size()) {
    return std::unexpected(
        std::format("the counters report {} issued instructions for {} records",
                    counters.counters.issued, events.size()));
  }
  if (counters.counters.trace_count != records) {
    return std::unexpected(
        std::format("the counters report {} trace records but the file has {}",
                    counters.counters.trace_count, records));
  }
  if (total % kWrap != counters.counters.cycles) {
    return std::unexpected(std::format(
        "the last record ends at cycle {} but the cycle counter is {}",
        total % kWrap, counters.counters.cycles));
  }
  return {};
}

void Attribute(std::vector<Event>& events, std::map<std::string, uint64_t>& out,
               uint64_t first_issue) {
  std::array<uint64_t, kEngines> end{};
  uint64_t previous = first_issue > 0 ? first_issue - 1 : 0;
  for (Event& event : events) {
    event.gap = event.issue > previous + 1 ? event.issue - previous - 1 : 0;
    if (event.gap > 0) {
      std::string blocker = "Other";
      uint64_t ready = 0;
      for (int e = 0; e < kEngines; ++e) {
        const bool needs =
            (event.wait & kWaitBits.at(e)) != 0 ||
            (event.IsEngine() && static_cast<int>(event.source) == e);
        if (needs && end.at(e) >= ready && end.at(e) > previous) {
          ready = end.at(e);
          blocker = std::string(kSourceNames.at(e));
        }
      }
      event.blocker = blocker;
      out[blocker] += event.gap;
    }
    if (event.IsEngine()) {
      end.at(static_cast<size_t>(event.source)) = event.completion;
    }
    previous = event.issue;
  }
}

void Occupancy(const std::vector<Event>& events, uint64_t total,
               Analysis& out) {
  std::map<uint64_t, std::array<int, kEngines>> changes;
  for (const Event& event : events) {
    if (!event.IsEngine()) {
      continue;
    }
    const auto engine = static_cast<size_t>(event.source);
    changes[event.issue].at(engine) += 1;
    changes[event.completion].at(engine) -= 1;
    out.busy.at(engine) += event.Duration();
  }
  std::array<bool, kEngines> active{};
  uint64_t last = 0;
  for (const auto& [time, delta] : changes) {
    out.occupancy[Mask(active)] += time - last;
    for (int e = 0; e < kEngines; ++e) {
      if (delta.at(e) > 0) {
        active.at(e) = true;
      } else if (delta.at(e) < 0) {
        active.at(e) = false;
      }
    }
    last = time;
  }
  out.occupancy[Mask(active)] += total - last;
}

void Exclusive(std::vector<Event>& events) {
  struct Change {
    uint64_t time;
    int delta;
    Event* event;
  };
  std::vector<Change> changes;
  for (Event& event : events) {
    if (event.IsEngine()) {
      changes.push_back({.time = event.issue, .delta = 1, .event = &event});
      changes.push_back(
          {.time = event.completion, .delta = -1, .event = &event});
    }
  }
  std::ranges::stable_sort(changes, [](const Change& a, const Change& b) {
    return a.time != b.time ? a.time < b.time : a.delta < b.delta;
  });
  std::array<Event*, kEngines> active{};
  int count = 0;
  uint64_t last = 0;
  for (const Change& change : changes) {
    if (count == 1) {
      for (Event* event : active) {
        if (event != nullptr) {
          event->exclusive += change.time - last;
        }
      }
    }
    const auto engine = static_cast<size_t>(change.event->source);
    if (change.delta > 0) {
      active.at(engine) = change.event;
      ++count;
    } else {
      active.at(engine) = nullptr;
      --count;
    }
    last = change.time;
  }
}

}  // namespace

std::string Category(const Instruction& instruction) {
  switch (instruction.opcode) {
    case kOpDmaLoad:
    case kOpDmaStore: {
      const iss::DmaCommand command = iss::DecodeDma(instruction.word);
      const std::string_view mem =
          command.mem < kMemNames.size() ? kMemNames.at(command.mem) : "?";
      if (!command.store && command.mode == 3) {
        return "GACC";
      }
      std::string name =
          std::format("{} {}{}", command.store ? "ST" : "LD", mem,
                      command.store ? "" : kDmaModes.at(command.mode));
      if ((command.flags & 1U) != 0) {
        name += " I";
      }
      if ((command.flags & 2U) != 0) {
        name += " R";
      }
      return name;
    }
    case kOpGemm: {
      const iss::GemmCommand command = iss::DecodeGemm(instruction.word);
      return std::format("GEMM {}",
                         kWeightModes.at(std::min(command.wmode, 2U)));
    }
    case kOpVec: {
      const iss::VecCommand command = iss::DecodeVec(instruction.word);
      if (command.op < kVecOps.size()) {
        return std::format("VEC {}{}", kVecOps.at(command.op),
                           command.quantize && command.op != 13 ? "+Q" : "");
      }
      return std::format("VEC RUN{}", command.op - 16);
    }
    case kOpNop:
      return "NOP";
    case kOpSignal:
      return "SIGNAL";
    case kOpHalt:
      return "HALT";
    case kOpWaitHost:
      return "WAIT_HOST";
    default:
      return "ILLEGAL";
  }
}

uint64_t Streaming(const Instruction& instruction) {
  switch (instruction.opcode) {
    case kOpDmaLoad:
    case kOpDmaStore: {
      const iss::DmaActivity activity =
          iss::EstimateDma(iss::DecodeDma(instruction.word));
      return activity.read_beats + activity.gather_row_beats +
             activity.write_beats;
    }
    case kOpGemm:
      return iss::GemmArrayCycles(iss::DecodeGemm(instruction.word));
    case kOpVec:
      return iss::FixedOperationActivity(iss::DecodeVec(instruction.word))
                 .words *
             4;
    default:
      return 0;
  }
}

uint64_t Model(const iss::CostModel& model, const Instruction& instruction) {
  switch (instruction.opcode) {
    case kOpDmaLoad:
    case kOpDmaStore:
      return model.Dma(iss::EstimateDma(iss::DecodeDma(instruction.word)))
          .cycles;
    case kOpGemm:
      return model.Gemm(iss::DecodeGemm(instruction.word)).cycles;
    case kOpVec: {
      const iss::VecCommand command = iss::DecodeVec(instruction.word);
      if (command.op >= 16) {
        return 0;
      }
      return model.Vec(iss::FixedOperationActivity(command)).cycles;
    }
    default:
      return 0;
  }
}

std::expected<void, std::string> Check(
    const Stream& stream, const std::vector<iss::TraceRecord>& records,
    const std::optional<CountersFile>& counters) {
  const auto unwrapped = Unwrap(stream, records, iss::CostModel());
  if (!unwrapped) {
    return std::unexpected(unwrapped.error());
  }
  if (auto ordered = CheckOrder(unwrapped->events); !ordered) {
    return ordered;
  }
  if (counters) {
    if (counters->stream != stream.name) {
      return std::unexpected(
          std::format("the counters belong to stream {}, not {}",
                      counters->stream, stream.name));
    }
    return CheckCounters(unwrapped->events, unwrapped->total, *counters,
                         records.size());
  }
  return {};
}

std::expected<Analysis, std::string> Analyze(
    const Stream& stream, const std::vector<iss::TraceRecord>& records,
    const std::optional<CountersFile>& counters, const iss::CostModel& model) {
  if (auto checked = Check(stream, records, counters); !checked) {
    return std::unexpected(checked.error());
  }
  auto unwrapped = Unwrap(stream, records, model);
  if (!unwrapped) {
    return std::unexpected(unwrapped.error());
  }
  Analysis analysis;
  analysis.stream = stream.name;
  analysis.events = std::move(unwrapped->events);
  analysis.total_cycles = unwrapped->total;
  analysis.first_issue =
      analysis.events.empty() ? 0 : analysis.events.front().issue;
  Occupancy(analysis.events, analysis.total_cycles, analysis);
  Attribute(analysis.events, analysis.blockers, analysis.first_issue);
  Exclusive(analysis.events);
  for (const Event& event : analysis.events) {
    Add(analysis.categories[event.category], event);
    analysis.has_statistics =
        analysis.has_statistics || event.statistics.has_value();
  }
  return analysis;
}

std::string TagPrefix(std::string_view tag, size_t depth) {
  size_t at = 0;
  for (size_t level = 0; level < depth; ++level) {
    const size_t next = tag.find('/', at);
    if (next == std::string_view::npos) {
      return std::string(tag);
    }
    at = next + 1;
  }
  return std::string(tag.substr(0, at == 0 ? 0 : at - 1));
}

std::map<std::string, Aggregate> ByTag(const Analysis& analysis, size_t depth) {
  std::map<std::string, Aggregate> out;
  for (const Event& event : analysis.events) {
    Add(out[TagPrefix(event.tag, depth)], event);
  }
  return out;
}

}  // namespace hotarubi::trace
