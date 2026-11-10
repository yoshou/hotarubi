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
#include "hotarubi/driver.hpp"
#include "hotarubi/isa.hpp"
#include "timeline.hpp"
#include "timing.hpp"
#include "trace_record.hpp"
#include "vec_engine.hpp"
#include "word.hpp"

namespace hotarubi::iss {
namespace {

using isa::kOpDmaLoad;
using isa::kOpDmaStore;
using isa::kOpGemm;
using isa::kOpHalt;
using isa::kOpNop;
using isa::kOpSignal;
using isa::kOpVec;
using isa::kOpWaitHost;

constexpr uint32_t kRegCtrl = Offset(Register::kCtrl);
constexpr uint32_t kRegStatus = Offset(Register::kStatus);
constexpr uint32_t kRegInstrBase = Offset(Register::kInstrBase);
constexpr uint32_t kRegInstrCount = Offset(Register::kInstrCount);
constexpr uint32_t kRegIrqStatus = Offset(Register::kIrqStatus);
constexpr uint32_t kRegIrqEnable = Offset(Register::kIrqEnable);
constexpr uint32_t kRegSignal = Offset(Register::kSignal);
constexpr uint32_t kRegSignalValue = Offset(Register::kSignalValue);
constexpr uint32_t kRegIssued = Offset(Register::kIssued);
constexpr uint32_t kRegCycles = Offset(Register::kCycles);
constexpr uint32_t kRegDmaCycles = Offset(Register::kDmaCycles);
constexpr uint32_t kRegGemmCycles = Offset(Register::kGemmCycles);
constexpr uint32_t kRegVecCycles = Offset(Register::kVecCycles);
constexpr uint32_t kRegStallCycles = Offset(Register::kStallCycles);
constexpr uint32_t kRegStarveCycles = Offset(Register::kStarveCycles);
constexpr uint32_t kRegTraceBase = Offset(Register::kTraceBase);
constexpr uint32_t kRegTraceCtrl = Offset(Register::kTraceCtrl);
constexpr uint32_t kRegTraceCount = Offset(Register::kTraceCount);
constexpr uint32_t kRegHostGo = Offset(Register::kHostGo);
constexpr uint32_t kRegisterMask = 0xFC;

constexpr uint32_t kInstructionBytes = 16;

uint32_t Low32(uint64_t value) { return static_cast<uint32_t>(value); }

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
  return {.store = instruction.Field(isa::kOpcode) == kOpDmaStore,
          .ddr_addr = instruction.Field(isa::dma::kDdrAddr),
          .mem = instruction.Field(isa::dma::kMem),
          .onchip = instruction.Field(isa::dma::kAddr),
          .rows = instruction.Field(isa::dma::kRows),
          .row_words = instruction.Field(isa::dma::kRowWords),
          .ddr_stride = instruction.Field(isa::dma::kDdrStride),
          .kind = instruction.Field(isa::dma::kKind),
          .tok = instruction.Field(isa::dma::kTok),
          .acc = instruction.Field(isa::dma::kAcc) != 0,
          .flush = instruction.Field(isa::dma::kFlush) != 0};
}

GemmCommand DecodeGemm(const Word128& instruction) {
  return {.a_base = instruction.Field(isa::gemm::kABase),
          .a_stride = instruction.Field(isa::gemm::kAStride),
          .kb = instruction.Field(isa::gemm::kKb),
          .m = instruction.Field(isa::gemm::kM),
          .w_base = instruction.Field(isa::gemm::kWBase),
          .w_stride = instruction.Field(isa::gemm::kWStride),
          .n8 = instruction.Field(isa::gemm::kN8),
          .c_base = instruction.Field(isa::gemm::kCBase),
          .wmode = instruction.Field(isa::gemm::kWmode),
          .acc = instruction.Field(isa::gemm::kAcc) != 0};
}

VecCommand DecodeVec(const Word128& instruction) {
  return {.entry = instruction.Field(isa::vec::kEntry),
          .quantize = instruction.Field(isa::vec::kQ) != 0,
          .src = instruction.Field(isa::vec::kSrc),
          .src2 = instruction.Field(isa::vec::kSrc2),
          .dst = instruction.Field(isa::vec::kDst),
          .rows_q = instruction.Field(isa::vec::kRowsQ),
          .n8 = instruction.Field(isa::vec::kN8),
          .p_base = instruction.Field(isa::vec::kPBase),
          .aux = instruction.Field(isa::vec::kAux),
          .imm = instruction.Field(isa::vec::kImm)};
}

uint64_t EstimateCycles(const Word128& instruction, const CostModel& model) {
  switch (instruction.Field(isa::kOpcode)) {
    case kOpDmaLoad:
    case kOpDmaStore:
      return model.Dma(EstimateDma(DecodeDma(instruction))).cycles;
    case kOpGemm:
      return model.Gemm(DecodeGemm(instruction)).cycles;
    case kOpVec: {
      const VecCommand command = DecodeVec(instruction);
      if (RunsLibraryProgram(command)) {
        return model.Vec(LibraryActivity(command)).cycles;
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
      return (signal_count_ << kSignalCount.lsb) | signal_id_;
    case kRegSignalValue:
      return signal_value_;
    case kRegIssued:
      return issued_;
    case kRegCycles:
      return Low32(timeline_.Cycles());
    case kRegDmaCycles:
      return Low32(timeline_.Busy(Source::kDma));
    case kRegGemmCycles:
      return Low32(timeline_.Busy(Source::kGemm));
    case kRegVecCycles:
      return Low32(timeline_.Busy(Source::kVec));
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
      irq_status_ &= ~(value & kIrqAll);
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
    const uint32_t opcode = instruction.Field(isa::kOpcode);
    if (opcode == kOpWaitHost &&
        host_go_ < instruction.Field(isa::wait_host::kValue)) {
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
  return (irq_status_ & irq_enable_ & kIrqAll) != 0;
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
  const Source source = SourceOf(opcode);
  const Cost cost = Execute(opcode, instruction);
  CheckHazards(source, instruction.Field(isa::kWait), memories_.log);
  if (observer_) {
    observer_(issued_ - 1, memories_.log);
  }
  memories_.log.Clear();
  const Span span =
      timeline_.Issue(source, instruction.Field(isa::kWait), cost.cycles);
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
    if ((wait & isa::WaitBit(static_cast<uint32_t>(e))) != 0) {
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
    for (const IssuedAccess& issued : history) {
      if (const auto conflict = FindConflict(issued.log, log)) {
        hazards_.push_back(
            {.earlier = issued.index, .later = index, .conflict = *conflict});
      }
    }
  }
  last_.at(engine) = index + 1;
  history_.at(engine).push_back({.index = index, .log = log});
}

Cost Machine::Execute(uint32_t opcode, const Word128& instruction) {
  switch (opcode) {
    case kOpDmaLoad:
    case kOpDmaStore: {
      const DmaCommand command = DecodeDma(instruction);
      dma_.Execute(command, ddr_, memories_);
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
      signal_id_ = instruction.Field(isa::signal::kId);
      signal_value_ = instruction.Field(isa::signal::kValue);
      signal_count_ = (signal_count_ + 1) & ((1U << kSignalCount.width) - 1);
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
  const uint32_t opcode = instruction.Field(isa::kOpcode);
  TraceRecord record{.index = issued_ - 1,
                     .issue = Low32(span.issue),
                     .completion = Low32(span.completion),
                     .source = SourceOf(opcode),
                     .opcode = opcode,
                     .wait = instruction.Field(isa::kWait),
                     .type = RecordType::kCompletion,
                     .statistics = {}};
  const auto write = [&](const TraceRecord& r) {
    const Word128 word = EncodeRecord(r);
    std::array<std::byte, isa::trace::kRecordBytes> bytes{};
    StoreLittle(word.low, std::span(bytes).first<8>());
    StoreLittle(word.high, std::span(bytes).last<8>());
    ddr_.Write(trace_base_ + (trace_count_ * isa::trace::kRecordBytes), bytes);
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
  return instruction.Field(isa::kOpcode) == kOpWaitHost &&
         host_go_ < instruction.Field(isa::wait_host::kValue);
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
