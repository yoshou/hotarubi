#ifndef HOTARUBI_TOOLS_TRACE_ANALYSIS_HPP_
#define HOTARUBI_TOOLS_TRACE_ANALYSIS_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "hotarubi/driver.hpp"
#include "input.hpp"
#include "timing.hpp"
#include "trace_record.hpp"

namespace hotarubi::trace {

inline constexpr int kEngines = 3;
inline constexpr std::array<std::string_view, 4> kSourceNames = {"DMA", "GEMM",
                                                                 "VEC", "CTRL"};

struct Event {
  uint32_t index = 0;
  uint64_t issue = 0;
  uint64_t completion = 0;
  iss::Source source = iss::Source::kSequencer;
  uint32_t opcode = 0;
  uint32_t wait = 0;
  std::string category;
  std::string tag;
  std::string note;
  uint64_t streaming = 0;
  uint64_t model = 0;
  std::optional<iss::Statistics> statistics;
  uint64_t gap = 0;
  uint64_t exclusive = 0;
  std::string blocker;

  [[nodiscard]] uint64_t Duration() const { return completion - issue; }
  [[nodiscard]] bool IsEngine() const {
    return static_cast<int>(source) < kEngines;
  }
};

struct Aggregate {
  uint64_t count = 0;
  uint64_t span = 0;
  uint64_t exclusive = 0;
  uint64_t streaming = 0;
  uint64_t model = 0;
  iss::Statistics statistics;
  uint64_t statistics_count = 0;
  std::array<uint64_t, kEngines> busy{};
  uint64_t first = UINT64_MAX;
  uint64_t last = 0;

  [[nodiscard]] uint64_t Span() const {
    return last > first ? last - first : 0;
  }
};

struct Analysis {
  std::string stream;
  std::vector<Event> events;
  uint64_t total_cycles = 0;
  uint64_t first_issue = 0;
  std::array<uint64_t, kEngines> busy{};
  std::map<std::string, uint64_t> occupancy;
  std::map<std::string, uint64_t> blockers;
  std::map<std::string, Aggregate> categories;
  bool has_statistics = false;
};

std::string Category(const Instruction& instruction);
uint64_t Streaming(const Instruction& instruction);
uint64_t Model(const iss::CostModel& model, const Instruction& instruction);

std::expected<void, std::string> Check(
    const Stream& stream, const std::vector<iss::TraceRecord>& records,
    const std::optional<CountersFile>& counters);
std::expected<Analysis, std::string> Analyze(
    const Stream& stream, const std::vector<iss::TraceRecord>& records,
    const std::optional<CountersFile>& counters,
    const iss::CostModel& model = iss::CostModel());
std::map<std::string, Aggregate> ByTag(const Analysis& analysis, size_t depth);
std::string TagPrefix(std::string_view tag, size_t depth);

}  // namespace hotarubi::trace

#endif  // HOTARUBI_TOOLS_TRACE_ANALYSIS_HPP_
