#include "machine.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <string>
#include <vector>

#include "dma_engine.hpp"
#include "gemm_array.hpp"
#include "timeline.hpp"
#include "timing.hpp"
#include "trace_record.hpp"
#include "vec_engine.hpp"
#include "word.hpp"

namespace hotarubi::iss {
namespace {

constexpr uint32_t kOpNop = 0;
constexpr uint32_t kOpDmaLoad = 1;
constexpr uint32_t kOpDmaStore = 2;
constexpr uint32_t kOpGemm = 4;
constexpr uint32_t kOpVec = 5;
constexpr uint32_t kOpSignal = 6;
constexpr uint32_t kOpHalt = 7;
constexpr uint32_t kOpWaitHost = 8;

constexpr uint32_t kRegCtrl = 0x00;
constexpr uint32_t kRegStatus = 0x04;
constexpr uint32_t kRegInstrBase = 0x08;
constexpr uint32_t kRegInstrCount = 0x0C;
constexpr uint32_t kRegIrqStatus = 0x10;
constexpr uint32_t kRegIrqEnable = 0x14;
constexpr uint32_t kRegSignal = 0x18;
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
constexpr uint32_t kRegisterMask = 0xFC;

constexpr uint32_t kCtrlStart = 0x1;
constexpr uint32_t kCtrlSoftReset = 0x2;
constexpr uint32_t kIrqSignal = 0x1;
constexpr uint32_t kIrqHalt = 0x2;
constexpr uint32_t kIrqIllegal = 0x4;
constexpr uint32_t kIrqMask = 0xF;
constexpr uint32_t kTraceEnable = 0x1;
constexpr uint32_t kTraceIdle = 0x2;
constexpr uint32_t kTraceStatistics = 0x4;
constexpr uint32_t kStatusRunning = 0x1;
constexpr uint32_t kStatusHalted = 0x2;
constexpr uint32_t kStatusError = 0x4;
constexpr uint32_t kStatusWaitingHost = 0x8;
constexpr uint32_t kStatusIbufEmpty = 0x80;
constexpr uint32_t kInstructionBytes = 16;
constexpr uint32_t kMemTbuf = 4;
constexpr uint32_t kMemCbuf = 5;

uint32_t Low32(uint64_t value) { return static_cast<uint32_t>(value); }

constexpr std::array<uint32_t, Timeline::kEngines> kWaitBits = {0x1, 0x4, 0x8};
constexpr std::array<std::string_view, 6> kMemNames = {"ABUF", "WBUF", "FBUF",
                                                       "PBUF", "TBUF", "CBUF"};

}  // namespace

std::string DescribeHazards(const std::vector<Hazard>& hazards, size_t limit) {
  std::string text;
  for (size_t i = 0; i < hazards.size() && i < limit; ++i) {
    const Hazard& hazard = hazards[i];
    text += std::format(
        "instructions {} and {} access {} word {} without an ordering wait\n",
        hazard.earlier, hazard.later,
        kMemNames.at(static_cast<size_t>(hazard.conflict.mem)),
        hazard.conflict.address);
  }
  if (hazards.size() > limit) {
    text += std::format("... {} hazards in total\n", hazards.size());
  }
  return text;
}

DmaCommand DecodeDma(const Word128& instruction) {
  return {.store = instruction.Field(120, 8) == kOpDmaStore,
          .ddr_addr = instruction.Field(80, 32),
          .mem = instruction.Field(77, 3),
          .onchip = instruction.Field(57, 20),
          .rows = instruction.Field(45, 12),
          .row_words = instruction.Field(33, 12),
          .ddr_stride = instruction.Field(9, 24),
          .mode = instruction.Field(7, 2),
          .flags = instruction.Field(112, 2),
          .tok = instruction.Field(0, 7)};
}

GemmCommand DecodeGemm(const Word128& instruction) {
  return {.a_base = instruction.Field(92, 20),
          .a_stride = instruction.Field(82, 10),
          .kb = instruction.Field(72, 10),
          .m = instruction.Field(62, 10),
          .w_base = instruction.Field(42, 20),
          .w_stride = instruction.Field(32, 10),
          .n8 = instruction.Field(24, 8),
          .c_base = instruction.Field(4, 20),
          .wmode = instruction.Field(2, 2),
          .acc = instruction.Field(1, 1) != 0};
}

VecCommand DecodeVec(const Word128& instruction) {
  return {.op = instruction.Field(107, 5),
          .quantize = instruction.Field(112, 1) != 0,
          .src = instruction.Field(95, 12),
          .src2 = instruction.Field(83, 12),
          .dst = instruction.Field(70, 13),
          .rows_q = instruction.Field(61, 9),
          .n8 = instruction.Field(53, 8),
          .p_base = instruction.Field(43, 10),
          .aux = instruction.Field(32, 11),
          .imm = instruction.Field(0, 32)};
}

uint64_t EstimateCycles(const Word128& instruction, const CostModel& model) {
  switch (instruction.Field(120, 8)) {
    case kOpDmaLoad:
    case kOpDmaStore:
      return model.Dma(EstimateDma(DecodeDma(instruction))).cycles;
    case kOpGemm:
      return model.Gemm(DecodeGemm(instruction)).cycles;
    case kOpVec: {
      const VecCommand command = DecodeVec(instruction);
      if (command.op < kVecRun) {
        return model.Vec(FixedOperationActivity(command)).cycles;
      }
      return model
          .Vec({.passes = 1,
                .words = uint64_t{command.rows_q} * command.n8,
                .drains = 1})
          .cycles;
    }
    default:
      return 0;
  }
}

uint32_t Machine::ReadRegister(uint32_t offset) const {
  switch (offset & kRegisterMask) {
    case kRegStatus:
      return Status();
    case kRegInstrBase:
      return instr_base_;
    case kRegInstrCount:
      return instr_count_;
    case kRegIrqStatus:
      return irq_status_;
    case kRegIrqEnable:
      return irq_enable_;
    case kRegSignal:
      return (signal_count_ << 16) | signal_id_;
    case kRegSignalValue:
      return signal_value_;
    case kRegIssued:
      return issued_;
    case kRegCycles:
      return Low32(timeline_.Cycles());
    case kRegGemmCycles:
      return Low32(timeline_.Busy(Source::kGemm));
    case kRegVecCycles:
      return Low32(timeline_.Busy(Source::kVec));
    case kRegDmaCycles:
      return Low32(timeline_.Busy(Source::kDma));
    case kRegStallCycles:
      return Low32(timeline_.Stall());
    case kRegStarveCycles:
      return Low32(timeline_.Starve());
    case kRegTraceBase:
      return trace_base_;
    case kRegTraceCtrl:
      return kTraceIdle | (trace_enable_ ? kTraceEnable : 0U) |
             (trace_statistics_ ? kTraceStatistics : 0U);
    case kRegTraceCount:
      return trace_count_;
    case kRegHostGo:
      return host_go_;
    default:
      return 0;
  }
}

void Machine::WriteRegister(uint32_t offset, uint32_t value) {
  switch (offset & kRegisterMask) {
    case kRegCtrl:
      if ((value & kCtrlSoftReset) != 0) {
        running_ = false;
        halted_ = false;
        error_ = false;
      } else if ((value & kCtrlStart) != 0) {
        Start();
      }
      break;
    case kRegInstrBase:
      instr_base_ = value;
      break;
    case kRegInstrCount:
      instr_count_ = value;
      break;
    case kRegIrqStatus:
      irq_status_ &= ~(value & kIrqMask);
      break;
    case kRegIrqEnable:
      irq_enable_ = value;
      break;
    case kRegTraceBase:
      trace_base_ = value;
      break;
    case kRegTraceCtrl:
      trace_enable_ = (value & kTraceEnable) != 0;
      trace_statistics_ = (value & kTraceStatistics) != 0;
      break;
    case kRegHostGo:
      host_go_ = value;
      break;
    default:
      break;
  }
}

Stop Machine::Run() {
  while (running_) {
    if (next_ >= stream_count_) {
      return Stop::kStarved;
    }
    const Word128 instruction = Fetch(next_);
    const uint32_t opcode = instruction.Field(120, 8);
    if (opcode == kOpWaitHost && host_go_ < instruction.Field(0, 32)) {
      return Stop::kWaitHost;
    }
    ++next_;
    ++issued_;
    Issue(opcode, instruction);
    switch (opcode) {
      case kOpSignal:
        return Stop::kSignal;
      case kOpHalt:
        return Stop::kHalt;
      case kOpNop:
      case kOpDmaLoad:
      case kOpDmaStore:
      case kOpGemm:
      case kOpVec:
      case kOpWaitHost:
        break;
      default:
        return Stop::kIllegal;
    }
  }
  return Stop::kIdle;
}

bool Machine::Interrupt() const {
  return (irq_status_ & irq_enable_ & kIrqMask) != 0;
}

void Machine::Start() {
  stream_base_ = instr_base_;
  stream_count_ = instr_count_;
  next_ = 0;
  issued_ = 0;
  trace_count_ = 0;
  host_go_ = 0;
  for (auto& history : history_) {
    history.clear();
  }
  last_.fill(0);
  known_.fill(0);
  hazards_.clear();
  memories_.log.Clear();
  running_ = true;
  halted_ = false;
  error_ = false;
  timeline_.Start();
}

void Machine::Issue(uint32_t opcode, const Word128& instruction) {
  uint64_t earliest_start = 0;
  const Source source = SourceOf(opcode);
  const Cost cost = Execute(opcode, instruction, earliest_start);
  CheckHazards(source, instruction.Field(116, 4), memories_.log);
  if (observer_) {
    observer_(issued_ - 1, memories_.log);
  }
  memories_.log.Clear();
  const Span span = timeline_.Issue(source, instruction.Field(116, 4),
                                    cost.cycles, earliest_start);
  Statistics statistics = cost.statistics;
  if (source == Source::kDma) {
    statistics.c = Low32(span.start - span.issue);
  }
  if (trace_enable_) {
    Record(instruction, span, statistics);
  }
}

void Machine::CheckHazards(Source source, uint32_t wait, const AccessLog& log) {
  for (int e = 0; e < Timeline::kEngines; ++e) {
    if ((wait & kWaitBits.at(e)) != 0) {
      known_.at(e) = std::max(known_.at(e), last_.at(e));
    }
  }
  const auto engine = static_cast<size_t>(source);
  if (engine >= Timeline::kEngines) {
    return;
  }
  known_.at(engine) = std::max(known_.at(engine), last_.at(engine));
  const uint32_t index = issued_ - 1;
  for (size_t e = 0; e < Timeline::kEngines; ++e) {
    std::vector<IssuedAccess>& history = history_.at(e);
    std::erase_if(history, [&](const IssuedAccess& issued) {
      return issued.index + 1 <= known_.at(e);
    });
    if (e == engine) {
      continue;
    }
    const bool tables_ordered = e == static_cast<size_t>(Source::kVec) &&
                                engine == static_cast<size_t>(Source::kDma);
    for (const IssuedAccess& issued : history) {
      AccessLog earlier = issued.log;
      AccessLog later = log;
      if (tables_ordered) {
        earlier.reads.ClearTables();
        later.writes.ClearTables();
      }
      if (const auto conflict = FindConflict(earlier, later)) {
        hazards_.push_back(
            {.earlier = issued.index, .later = index, .conflict = *conflict});
      }
    }
  }
  last_.at(engine) = index + 1;
  history_.at(engine).push_back({.index = index, .log = log});
}

Cost Machine::Execute(uint32_t opcode, const Word128& instruction,
                      uint64_t& earliest_start) {
  switch (opcode) {
    case kOpDmaLoad:
    case kOpDmaStore: {
      const DmaCommand command = DecodeDma(instruction);
      dma_.Execute(command, ddr_, memories_);
      if (!command.store &&
          (command.mem == kMemTbuf || command.mem == kMemCbuf)) {
        earliest_start = timeline_.EngineFree(Source::kVec);
      }
      return model_.Dma(dma_.Activity());
    }
    case kOpGemm: {
      const GemmCommand command = DecodeGemm(instruction);
      ExecuteGemm(command, memories_);
      return model_.Gemm(command);
    }
    case kOpVec:
      vec_.Execute(DecodeVec(instruction), memories_);
      return model_.Vec(vec_.Activity());
    case kOpSignal:
      signal_id_ = instruction.Field(104, 8);
      signal_value_ = instruction.Field(0, 32);
      signal_count_ = (signal_count_ + 1) & 0xFFFFU;
      irq_status_ |= kIrqSignal;
      return {};
    case kOpHalt:
      running_ = false;
      halted_ = true;
      irq_status_ |= kIrqHalt;
      return {};
    case kOpNop:
    case kOpWaitHost:
      return {};
    default:
      running_ = false;
      halted_ = true;
      error_ = true;
      irq_status_ |= kIrqIllegal;
      return {};
  }
}

void Machine::Record(const Word128& instruction, const Span& span,
                     const Statistics& statistics) {
  const uint32_t opcode = instruction.Field(120, 8);
  TraceRecord record{.index = issued_ - 1,
                     .issue = Low32(span.issue),
                     .completion = Low32(span.completion),
                     .source = SourceOf(opcode),
                     .opcode = opcode,
                     .wait = instruction.Field(116, 4),
                     .flags = instruction.Field(112, 4),
                     .type = RecordType::kCompletion,
                     .statistics = {}};
  const auto write = [&](const TraceRecord& r) {
    const Word128 word = EncodeRecord(r);
    std::array<std::byte, kTraceRecordBytes> bytes{};
    StoreLittle(word.low, std::span(bytes).first<8>());
    StoreLittle(word.high, std::span(bytes).last<8>());
    ddr_.Write(trace_base_ + (trace_count_ * kTraceRecordBytes), bytes);
    ++trace_count_;
  };
  write(record);
  if (trace_statistics_ && record.source != Source::kSequencer) {
    record.type = RecordType::kStatistics;
    record.statistics = statistics;
    write(record);
  }
}

Word128 Machine::Fetch(uint32_t index) const {
  std::array<std::byte, kInstructionBytes> bytes{};
  ddr_.Read(stream_base_ + (index * kInstructionBytes), bytes);
  return LoadWord128(bytes);
}

bool Machine::WaitingForHost() const {
  if (!running_ || next_ >= stream_count_) {
    return false;
  }
  const Word128 instruction = Fetch(next_);
  return instruction.Field(120, 8) == kOpWaitHost &&
         host_go_ < instruction.Field(0, 32);
}

uint32_t Machine::Status() const {
  uint32_t status = 0;
  status |= running_ ? kStatusRunning : 0U;
  status |= halted_ ? kStatusHalted : 0U;
  status |= error_ ? kStatusError : 0U;
  status |= WaitingForHost() ? kStatusWaitingHost : 0U;
  status |= running_ && next_ < stream_count_ ? 0U : kStatusIbufEmpty;
  return status;
}

}  // namespace hotarubi::iss
