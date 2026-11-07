#include "timing.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>

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
      (activity.gather_rows * p.dma_gather_row_cycles);
  cost.statistics = {.a = Clamp32(activity.read_beats),
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
  const uint64_t words = activity.words * parameters_.vec_word_cycles;
  const uint64_t tail =
      uint64_t{activity.passes} * parameters_.vec_issue_tail_cycles;
  return words > tail ? words - tail : 0;
}

uint64_t CostModel::DrainCycles(const VecActivity& activity) const {
  const TimingParameters& p = parameters_;
  return (uint64_t{activity.drains} * p.vec_drain_cycles) +
         (uint64_t{activity.drains_reduction} * p.vec_drain_reduction_cycles) +
         (uint64_t{activity.drains_quantized} * p.vec_drain_quantized_cycles);
}

uint64_t CostModel::ScalarCycles(const VecActivity& activity) const {
  const TimingParameters& p = parameters_;
  uint64_t arithmetic = 0;
  uint64_t immediate = 0;
  uint64_t starts = 0;
  for (size_t i = 0; i < kRowScalarKinds; ++i) {
    const uint64_t count = activity.scalars.at(i);
    arithmetic += count * kScalarPrograms.at(i).arithmetic;
    immediate += count * kScalarPrograms.at(i).immediate;
    starts += count;
  }
  arithmetic += uint64_t{activity.rn} * kCountReciprocalProgram.arithmetic;
  immediate += uint64_t{activity.rn} * kCountReciprocalProgram.immediate;
  starts += activity.rn;
  return (arithmetic * p.vec_scalar_arithmetic_cycles) +
         (immediate * p.vec_scalar_immediate_cycles) +
         (starts * p.vec_scalar_start_cycles);
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
    activity.read_beats = 1 + (entries / 2) + (groups * command.row_words * 8) +
                          (entries * row_beats);
    activity.read_bursts = Bursts(1 + (entries / 2)) +
                           (groups * command.row_words) +
                           (entries * Bursts(row_beats));
    return activity;
  }
  if (units == 0) {
    return activity;
  }
  const bool indexed = (command.flags & kFlagIndex) != 0;
  const bool rows = (command.flags & kFlagRows) != 0;
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
