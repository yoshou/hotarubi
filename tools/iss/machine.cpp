#include "machine.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

#include "dma_engine.hpp"
#include "gemm_array.hpp"
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
constexpr uint32_t kRegTraceBase = 0x3C;
constexpr uint32_t kRegTraceCtrl = 0x40;
constexpr uint32_t kRegHostGo = 0x48;
constexpr uint32_t kRegisterMask = 0xFC;

constexpr uint32_t kCtrlStart = 0x1;
constexpr uint32_t kCtrlSoftReset = 0x2;
constexpr uint32_t kIrqSignal = 0x1;
constexpr uint32_t kIrqHalt = 0x2;
constexpr uint32_t kIrqIllegal = 0x4;
constexpr uint32_t kIrqMask = 0xF;
constexpr uint32_t kTraceIdle = 0x2;
constexpr uint32_t kStatusRunning = 0x1;
constexpr uint32_t kStatusHalted = 0x2;
constexpr uint32_t kStatusError = 0x4;
constexpr uint32_t kStatusWaitingHost = 0x8;
constexpr uint32_t kStatusIbufEmpty = 0x80;
constexpr uint32_t kInstructionBytes = 16;

}  // namespace

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
    case kRegTraceBase:
      return trace_base_;
    case kRegTraceCtrl:
      return kTraceIdle | (trace_enable_ ? 1U : 0U);
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
      trace_enable_ = (value & 1U) != 0;
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
  host_go_ = 0;
  running_ = true;
  halted_ = false;
  error_ = false;
}

void Machine::Issue(uint32_t opcode, const Word128& instruction) {
  switch (opcode) {
    case kOpNop:
    case kOpWaitHost:
      break;
    case kOpDmaLoad:
    case kOpDmaStore:
      dma_.Execute(DecodeDma(instruction), ddr_, memories_);
      break;
    case kOpGemm:
      ExecuteGemm(DecodeGemm(instruction), memories_);
      break;
    case kOpVec:
      vec_.Execute(DecodeVec(instruction), memories_);
      break;
    case kOpSignal:
      signal_id_ = instruction.Field(104, 8);
      signal_value_ = instruction.Field(0, 32);
      signal_count_ = (signal_count_ + 1) & 0xFFFFU;
      irq_status_ |= kIrqSignal;
      break;
    case kOpHalt:
      running_ = false;
      halted_ = true;
      irq_status_ |= kIrqHalt;
      break;
    default:
      running_ = false;
      halted_ = true;
      error_ = true;
      irq_status_ |= kIrqIllegal;
      break;
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
