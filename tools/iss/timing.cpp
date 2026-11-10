#include "timing.hpp"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <expected>
#include <format>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include "dma_engine.hpp"
#include "gemm_array.hpp"
#include "memories.hpp"
#include "trace_record.hpp"
#include "vec_engine.hpp"

namespace hotarubi::iss {
namespace {

constexpr uint32_t kBurstBeats = 16;
constexpr uint32_t kModeRaw = 0;
constexpr uint32_t kModeGather = 3;
constexpr uint32_t kFlagIndex = 1;
constexpr uint32_t kFlagRows = 2;
constexpr uint32_t kGatherEntriesPerSlot = 16;
constexpr uint32_t kGatherSlotGroup = 8;

uint32_t BeatsPerWord(uint32_t mem) {
  switch (mem) {
    case 0:
    case 1:
      return 9;
    case 4:
    case 5:
      return 1;
    default:
      return 8;
  }
}

uint64_t Bursts(uint64_t beats) {
  return (beats + kBurstBeats - 1) / kBurstBeats;
}

uint32_t RowFormatBeats(const DmaCommand& command) {
  const uint32_t tok = command.tok == 0 ? 1 : std::min(command.tok, 32U);
  if (command.mem == 0) {
    return (2 * tok) + (tok > 16 ? 2 : 1);
  }
  return 8 * tok;
}

uint64_t ScaledCycles(uint64_t beats, uint32_t cycles_per_kilobeat) {
  return ((beats * cycles_per_kilobeat) + kBeatsPerKilobeat - 1) /
         kBeatsPerKilobeat;
}

uint32_t Clamp32(uint64_t value) {
  return static_cast<uint32_t>(std::min<uint64_t>(value, UINT32_MAX));
}

}  // namespace

uint64_t GemmArrayCycles(const GemmCommand& command) {
  return uint64_t{command.n8} * (command.m / 32) * command.kb * 8;
}

Cost CostModel::Dma(const DmaActivity& activity) const {
  const TimingParameters& p = parameters_;
  Cost cost;
  cost.cycles =
      p.dma_fixed_cycles +
      ScaledCycles(activity.read_beats, p.dma_read_cycles_per_kilobeat) +
      (activity.read_bursts * p.dma_read_burst_cycles) +
      ScaledCycles(activity.write_beats, p.dma_write_cycles_per_kilobeat) +
      (activity.write_bursts * p.dma_write_burst_cycles) +
      (activity.gather_entries * p.dma_gather_entry_cycles) +
      (activity.gather_rows * p.dma_gather_row_cycles) +
      (activity.gather_fetches * p.dma_gather_fetch_cycles);
  cost.statistics = {
      .a = Clamp32(activity.read_beats + activity.gather_row_beats),
      .b = Clamp32(activity.write_beats),
      .c = 0,
      .d = Clamp32(activity.null_entries)};
  return cost;
}

Cost CostModel::Gemm(const GemmCommand& command) const {
  const uint64_t array = GemmArrayCycles(command);
  return {.cycles = array + parameters_.gemm_fixed_cycles,
          .statistics = {.a = Clamp32(array), .b = 0, .c = 0, .d = 0}};
}

uint64_t CostModel::IssueCycles(const VecActivity& activity) const {
  const uint64_t words = (activity.words + activity.overlap_stall_words) *
                         parameters_.vec_word_cycles;
  const uint64_t tail =
      uint64_t{activity.passes} * parameters_.vec_issue_tail_cycles;
  const uint64_t transposes =
      (uint64_t{activity.transposes} * parameters_.vec_transpose_cycles) +
      (uint64_t{activity.transpose_tiles} *
       parameters_.vec_transpose_tile_cycles);
  return (words > tail ? words - tail : 0) + transposes + activity.skip_cycles;
}

uint64_t CostModel::DrainCycles(const VecActivity& activity) const {
  const TimingParameters& p = parameters_;
  return (uint64_t{activity.drains} * p.vec_drain_cycles) +
         (uint64_t{activity.drains_quantized} * p.vec_drain_quantized_cycles);
}

uint64_t CostModel::ScalarCycles(const VecActivity& activity) const {
  return (activity.context_stall_words * parameters_.vec_word_cycles) +
         (uint64_t{activity.rn} * ScalarProgramCycles(kCountReciprocalProgram));
}

Cost CostModel::Vec(const VecActivity& activity) const {
  const TimingParameters& p = parameters_;
  const uint64_t issue = IssueCycles(activity);
  const uint64_t drain = DrainCycles(activity);
  const uint64_t scalar = ScalarCycles(activity);
  Cost cost;
  cost.cycles = p.vec_fixed_cycles +
                (uint64_t{activity.passes} * p.vec_pass_cycles) + issue +
                drain + scalar + (uint64_t{activity.salu} * p.vec_salu_cycles) +
                (uint64_t{activity.smem} * p.vec_smem_cycles) +
                (uint64_t{activity.branches} * p.vec_branch_cycles);
  cost.statistics = {.a = Clamp32(issue),
                     .b = Clamp32(drain),
                     .c = Clamp32(scalar),
                     .d = activity.passes};
  return cost;
}

DmaActivity EstimateDma(const DmaCommand& command) {
  DmaActivity activity;
  const uint64_t units = uint64_t{command.rows} * command.row_words;
  if (!command.store && command.mode == kModeGather) {
    const uint64_t entries = uint64_t{command.rows} * kGatherEntriesPerSlot;
    const uint64_t groups =
        (command.rows + kGatherSlotGroup - 1) / kGatherSlotGroup;
    const uint64_t row_beats = (2 * command.row_words) + 1;
    activity.gather_entries = entries;
    activity.gather_rows = entries;
    activity.gather_row_beats = entries * row_beats;
    activity.gather_fetches = entries == 0 ? 0 : 1;
    activity.read_beats = 1 + (entries / 2) + (groups * command.row_words * 8);
    activity.read_bursts =
        Bursts(1 + (entries / 2)) + (groups * command.row_words);
    return activity;
  }
  if (units == 0) {
    return activity;
  }
  const bool rows = (command.flags & kFlagRows) != 0;
  const bool indexed =
      !command.store && rows && (command.flags & kFlagIndex) != 0;
  if (!command.store && command.mode != kModeRaw) {
    activity.read_beats = units * 9;
    activity.read_bursts = units;
    return activity;
  }
  const uint64_t unit_beats =
      rows ? RowFormatBeats(command) : BeatsPerWord(command.mem);
  uint64_t& beats = command.store ? activity.write_beats : activity.read_beats;
  uint64_t& bursts =
      command.store ? activity.write_bursts : activity.read_bursts;
  if (indexed) {
    const uint64_t list_beats = (units + 3) / 4;
    activity.read_beats += list_beats;
    activity.read_bursts += Bursts(list_beats);
    beats += units * unit_beats;
    bursts += units * Bursts(unit_beats);
    return activity;
  }
  const uint64_t row_beats = uint64_t{command.row_words} * unit_beats;
  beats += uint64_t{command.rows} * row_beats;
  bursts += uint64_t{command.rows} * Bursts(row_beats);
  return activity;
}

DdrExtent EstimateDdrExtent(const DmaCommand& command) {
  if (!command.store &&
      (command.mode == kModeGather ||
       (command.mode == kModeRaw && (command.flags & kFlagRows) != 0 &&
        (command.flags & kFlagIndex) != 0))) {
    return {.whole = true};
  }
  const uint64_t units = uint64_t{command.rows} * command.row_words;
  if (units == 0) {
    return {.lo = command.ddr_addr, .hi = command.ddr_addr};
  }
  uint64_t unit_beats = BeatsPerWord(command.mem);
  if (!command.store && command.mode != kModeRaw) {
    unit_beats = 9;
  } else if ((command.flags & kFlagRows) != 0) {
    unit_beats = RowFormatBeats(command);
  }
  const uint64_t row_bytes = uint64_t{command.row_words} * unit_beats * 16;
  const uint64_t last_row =
      command.ddr_addr + (uint64_t{command.rows - 1} * command.ddr_stride);
  return {.lo = command.ddr_addr, .hi = last_row + row_bytes};
}

std::expected<TimingParameters, std::string> ParseTiming(
    std::string_view assignments, TimingParameters base) {
  TimingParameters p = base;
  const std::array<std::pair<std::string_view, uint32_t*>, 25> fields = {
      {{"start_cycles", &p.start_cycles},
       {"gemm_fixed_cycles", &p.gemm_fixed_cycles},
       {"dma_fixed_cycles", &p.dma_fixed_cycles},
       {"dma_read_cycles_per_kilobeat", &p.dma_read_cycles_per_kilobeat},
       {"dma_read_burst_cycles", &p.dma_read_burst_cycles},
       {"dma_write_cycles_per_kilobeat", &p.dma_write_cycles_per_kilobeat},
       {"dma_write_burst_cycles", &p.dma_write_burst_cycles},
       {"dma_gather_entry_cycles", &p.dma_gather_entry_cycles},
       {"dma_gather_row_cycles", &p.dma_gather_row_cycles},
       {"dma_gather_fetch_cycles", &p.dma_gather_fetch_cycles},
       {"vec_fixed_cycles", &p.vec_fixed_cycles},
       {"vec_pass_cycles", &p.vec_pass_cycles},
       {"vec_word_cycles", &p.vec_word_cycles},
       {"vec_issue_tail_cycles", &p.vec_issue_tail_cycles},
       {"vec_drain_cycles", &p.vec_drain_cycles},
       {"vec_drain_quantized_cycles", &p.vec_drain_quantized_cycles},
       {"vec_salu_cycles", &p.vec_salu_cycles},
       {"vec_smem_cycles", &p.vec_smem_cycles},
       {"vec_branch_cycles", &p.vec_branch_cycles},
       {"vec_transpose_cycles", &p.vec_transpose_cycles},
       {"vec_transpose_tile_cycles", &p.vec_transpose_tile_cycles}}};
  while (!assignments.empty()) {
    const size_t comma = assignments.find(',');
    const std::string_view item = assignments.substr(0, comma);
    assignments = comma == std::string_view::npos
                      ? std::string_view{}
                      : assignments.substr(comma + 1);
    const size_t equals = item.find('=');
    if (equals == std::string_view::npos) {
      return std::unexpected(std::format("timing item {} has no value", item));
    }
    const std::string_view name = item.substr(0, equals);
    const std::string digits(item.substr(equals + 1));
    uint32_t value = 0;
    const char* const end = digits.data() + digits.size();
    const auto [last, error] = std::from_chars(digits.data(), end, value);
    if (digits.empty() || error != std::errc{} || last != end) {
      return std::unexpected(
          std::format("timing parameter {} needs a number", name));
    }
    bool found = false;
    for (const auto& [field, target] : fields) {
      if (field == name) {
        *target = value;
        found = true;
      }
    }
    if (!found) {
      return std::unexpected(std::format("unknown timing parameter {}", name));
    }
  }
  return p;
}

VecActivity FixedOperationActivity(const VecCommand& command) {
  if (command.op >= kVecRun) {
    return {};
  }
  static thread_local std::unique_ptr<Memories> memories;
  static thread_local std::unique_ptr<VecEngine> engine;
  if (!memories) {
    memories = std::make_unique<Memories>();
    engine = std::make_unique<VecEngine>();
  }
  engine->Execute(command, *memories);
  return engine->Activity();
}

}  // namespace hotarubi::iss
