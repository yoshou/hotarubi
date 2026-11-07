#include "vec_engine.hpp"

#include <array>
#include <bitset>
#include <cstdint>
#include <format>
#include <span>

#include "fp32.hpp"
#include "hotarubi/require.hpp"
#include "lane.hpp"
#include "memories.hpp"
#include "rom.hpp"
#include "scalar.hpp"
#include "word.hpp"

namespace hotarubi::iss {
namespace {

constexpr uint32_t kKindPass = 0;
constexpr uint32_t kKindSalu = 1;
constexpr uint32_t kKindSmem = 2;
constexpr uint32_t kKindBranch = 3;
constexpr uint32_t kKindRet = 4;

constexpr uint32_t kSmemLoad = 0;
constexpr uint32_t kSmemRowLoad = 2;
constexpr uint32_t kSmemRowMove = 4;
constexpr uint32_t kSmemMoveRow = 5;
constexpr uint32_t kSmemCount = 6;

constexpr uint32_t kPcMask = kCbufDepth - 1;
constexpr uint32_t kFbufMask = kFbufDepth - 1;
constexpr uint32_t kAbufMask = kAbufDepth - 1;
constexpr uint32_t kPbufMask = kPbufDepth - 1;
constexpr uint32_t kPbufAddressMask = 0x3FF;

uint32_t Salu(uint32_t op, uint32_t a, uint32_t b) {
  switch (op) {
    case 0:
      return a + b;
    case 1:
      return a - b;
    case 2:
      return a & b;
    case 3:
      return a | b;
    case 4:
      return a ^ b;
    case 5:
      return a << (b & 31U);
    case 6:
      return a >> (b & 31U);
    case 7:
      return static_cast<uint32_t>(static_cast<int32_t>(a) >> (b & 31U));
    case 8:
      return a * b;
    case 9:
      return static_cast<int32_t>(a) < static_cast<int32_t>(b) ? 1 : 0;
    case 10:
      return a < b ? 1 : 0;
    case 11:
      return a == b ? 1 : 0;
    default:
      return b;
  }
}

uint32_t PbufAddress(PbufAccess access, uint32_t base, uint32_t tile_row,
                     uint32_t w) {
  switch (access) {
    case PbufAccess::kRowVec:
      return (base + (w >> 2)) & kPbufAddressMask;
    case PbufAccess::kTile:
      return (base + tile_row + (w & 3U)) & kPbufAddressMask;
    case PbufAccess::kNone:
      break;
  }
  return 0;
}

void CheckPbufTiming(const Pass& pass) {
  if (pass.pa == PbufAccess::kNone) {
    return;
  }
  Require(pass.c0 != C0Source::kPbuf || !pass.pa_at_mul,
          "a PASS adds PBUF at the pre-adder with pa_at_mul set");
  Require(pass.b != BSource::kPbuf || pass.pa_at_mul,
          "a PASS multiplies by PBUF without pa_at_mul");
}

struct WordOperands {
  const FbufWord& x;
  const FbufWord& x2;
  const FbufWord& pbuf;
  uint32_t w = 0;
  uint32_t imm = 0;
  uint32_t imm2 = 0;
};

struct WordResult {
  FbufWord out{};
  std::array<std::array<uint32_t, kLanes>, kTokens> reduced{};
};

WordResult ComputeWord(const Pass& pass, const WordOperands& in,
                       const RowState& rows, std::span<const uint64_t> tbuf) {
  const uint32_t slice = in.w & 3U;
  const uint32_t masked_reduction =
      pass.red == Reduction::kMax ? kFp32NegativeInfinity : 0;
  WordResult result;
  for (uint32_t t = 0; t < kTokens; ++t) {
    const uint32_t mul_slice = pass.pa == PbufAccess::kTile ? t : slice;
    for (uint32_t l = 0; l < kLanes; ++l) {
      const uint32_t y =
          LaneOutput(pass,
                     {.x = in.x.at(PermutedIndex(pass.px, pass.pcol, t, l)),
                      .x2 = in.x2.at((t * kLanes) + l),
                      .pbuf_pre = in.pbuf.at((slice * kLanes) + l),
                      .pbuf_mul = in.pbuf.at((mul_slice * kLanes) + l),
                      .t = t,
                      .l = l,
                      .imm = in.imm,
                      .imm2 = in.imm2},
                     rows, tbuf);
      const bool masked =
          pass.mask_en && (in.w * kLanes) + l >= (in.imm2 & 0xFFFFU);
      result.out.at((t * kLanes) + l) = masked ? 0 : y;
      result.reduced.at(t).at(l) = masked ? masked_reduction : y;
    }
  }
  return result;
}

}  // namespace

struct VecEngine::PassState {
  Pass pass{};
  uint32_t src = 0;
  uint32_t src2 = 0;
  uint32_t dst = 0;
  uint32_t abuf = 0;
  uint32_t tile_row = 0;
  uint32_t pb = 0;
  uint32_t imm = 0;
  uint32_t imm2 = 0;
  uint32_t entry = 0;
  uint32_t next_quarter = 0;
  std::array<FbufWord, 4> block{};
  Reductions reductions = ClearedReductions();
  std::bitset<kFbufDepth> written;
};

void VecEngine::Execute(const VecCommand& command, Memories& memories) {
  activity_ = {};
  Initialize(command);
  uint32_t pc = command.op == kVecRun ? command.aux & kPcMask
                                      : kRomStart.at(command.op & 0xFU);
  while (true) {
    memories.log.Read(Mem::kCbuf, pc);
    const Word128 instruction = memories.cbuf.at(pc);
    pc = (pc + 1) & kPcMask;
    switch (instruction.Field(124, 4)) {
      case kKindPass:
        RunPass(instruction, memories);
        break;
      case kKindSalu:
        ++activity_.salu;
        RunSalu(instruction);
        break;
      case kKindSmem:
        ++activity_.smem;
        if (undrained_) {
          ++activity_.drains;
        }
        Drain();
        RunSmem(instruction, memories);
        break;
      case kKindBranch:
        ++activity_.branches;
        if (Taken(instruction)) {
          pc = instruction.Field(0, 10);
        }
        break;
      case kKindRet:
        Require(!undrained_, "a VEC program returns after a nodrain PASS");
        return;
      default:
        break;
    }
  }
}

uint32_t VecEngine::Register(uint32_t index) const {
  return index == 0 ? 0 : registers_.at(index);
}

void VecEngine::Initialize(const VecCommand& command) {
  quantize_ = command.quantize;
  uint32_t scratch = command.dst & kFbufMask;
  if (command.quantize) {
    const bool src2_scratch =
        command.op == 4 || command.op == 6 || command.op == 7;
    scratch = (src2_scratch ? command.src2 : command.imm) & kFbufMask;
  }
  registers_.at(1) = command.src & kFbufMask;
  registers_.at(2) = command.src2 & kFbufMask;
  registers_.at(3) = command.dst & kAbufMask;
  registers_.at(4) = command.rows_q & 0x1FFU;
  registers_.at(5) = command.n8 & 0xFFU;
  registers_.at(6) = command.p_base & kPbufAddressMask;
  registers_.at(7) = command.op == kVecRun ? 0 : command.aux & 0x7FFU;
  registers_.at(8) = command.imm;
  registers_.at(9) = scratch;
  registers_.at(10) = command.quantize ? 1 : 0;
}

void VecEngine::RunPass(const Word128& instruction, Memories& memories) {
  PassState state;
  state.pass = DecodePass(instruction);
  const Pass& pass = state.pass;
  const uint32_t quads = Register(pass.rq) & 0x1FFU;
  const uint32_t words = Register(pass.rnw) & 0x3FFU;
  Require(quads != 0 && words != 0,
          std::format("a PASS has {} quads and {} words", quads, words));
  CheckPbufTiming(pass);
  Require(!undrained_reduction_ || pass.red == Reduction::kNone,
          "a PASS reduces after a nodrain PASS that reduces");
  Require(!undrained_pbuf_at_mul_ || pass.c0 != C0Source::kPbuf,
          "a PASS adds PBUF after a nodrain PASS that reads PBUF at the "
          "multiplier");
  const uint32_t d = Register(pass.rd);
  const uint32_t n8 = Register(pass.rn8) & 0xFFU;
  state.src = Register(pass.rx) & kFbufMask;
  state.src2 = Register(pass.rx2) & kFbufMask;
  state.dst = d & kFbufMask;
  state.abuf = d & kAbufMask;
  state.pb = Register(pass.rpb) & kPbufAddressMask;
  state.imm = Register(pass.rimm);
  state.imm2 = Register(pass.rimm2);
  ++activity_.passes;
  activity_.words += uint64_t{quads} * words;
  for (uint32_t q = 0; q < quads; ++q) {
    for (uint32_t w = 0; w < words; ++w) {
      RunWord(state, w, memories);
    }
    const uint32_t step = pass.x_s1 ? 1 : n8;
    state.src = (state.src + step) & kFbufMask;
    state.src2 = (state.src2 + step) & kFbufMask;
    state.dst = (state.dst + (pass.dst_s1 ? 1 : n8)) & kFbufMask;
    state.abuf = (state.abuf + (n8 >> 2)) & kAbufMask;
    state.tile_row = (state.tile_row + 4) & kPbufAddressMask;
  }
  Require(state.next_quarter == 0,
          "a PASS ends inside a block of 4 quantized words");
  if (pass.nodrain) {
    in_flight_ |= state.written;
    undrained_ = true;
    undrained_reduction_ = undrained_reduction_ || pass.red != Reduction::kNone;
    undrained_pbuf_at_mul_ = undrained_pbuf_at_mul_ || pass.pa_at_mul;
    return;
  }
  ++activity_.drains;
  if (pass.red != Reduction::kNone) {
    ++activity_.drains_reduction;
  }
  if (pass.wr == WriteTarget::kAbuf ||
      (pass.wr == WriteTarget::kFinal && quantize_)) {
    ++activity_.drains_quantized;
  }
  Drain();
  if (pass.sc != RowScalar::kNone) {
    ++activity_.scalars.at(static_cast<size_t>(pass.sc));
  }
  ApplyRowScalar(pass.sc, state.reductions, state.imm, rn_, rows_);
}

void VecEngine::RunWord(PassState& state, uint32_t w, Memories& memories) {
  const Pass& pass = state.pass;
  const FbufWord x =
      Operand(state, (state.src + WordOf(pass.wx, w)) & kFbufMask, memories);
  const FbufWord x2 =
      pass.x2_en
          ? Operand(state, (state.src2 + WordOf(pass.wx2, w)) & kFbufMask,
                    memories)
          : x;
  const uint32_t pbuf_address =
      PbufAddress(pass.pa, state.pb, state.tile_row, w) & kPbufMask;
  if (pass.pa != PbufAccess::kNone) {
    memories.log.Read(Mem::kPbuf, pbuf_address);
  }
  if (pass.nl != Nonlinear::kNone) {
    memories.log.ReadTable(Mem::kTbuf);
  }
  const WordResult result = ComputeWord(pass,
                                        {.x = x,
                                         .x2 = x2,
                                         .pbuf = memories.pbuf.at(pbuf_address),
                                         .w = w,
                                         .imm = state.imm,
                                         .imm2 = state.imm2},
                                        rows_, memories.tbuf);
  for (uint32_t t = 0; t < kTokens; ++t) {
    if (pass.red == Reduction::kSum) {
      uint32_t& sum = state.reductions.sum.at(t).at(state.entry);
      sum = Fp32Add(sum, SumTree(result.reduced.at(t)));
    } else if (pass.red == Reduction::kMax) {
      uint32_t& max = state.reductions.max.at(t);
      max = Fp32Max(max, MaxTree(result.reduced.at(t)));
    }
  }
  state.entry = (state.entry + 1) & 7U;
  Store(state, WordOf(pass.wd, w), result.out, memories);
}

const FbufWord& VecEngine::Operand(const PassState& state, uint32_t address,
                                   Memories& memories) const {
  if (state.written.test(address)) {
    Fail(std::format("a PASS reads FBUF word {} after writing it", address));
  }
  if (in_flight_.test(address)) {
    Fail(std::format(
        "a PASS reads FBUF word {} written by an undrained nodrain PASS",
        address));
  }
  memories.log.Read(Mem::kFbuf, address);
  return memories.fbuf.at(address);
}

void VecEngine::Store(PassState& state, uint32_t word, const FbufWord& out,
                      Memories& memories) const {
  WriteTarget target = state.pass.wr;
  if (target == WriteTarget::kFinal) {
    target = quantize_ ? WriteTarget::kAbuf : WriteTarget::kFbuf;
  }
  if (target == WriteTarget::kFbuf) {
    const uint32_t address = (state.dst + word) & kFbufMask;
    memories.log.Write(Mem::kFbuf, address);
    memories.fbuf.at(address) = out;
    state.written.set(address);
  } else if (target == WriteTarget::kAbuf) {
    const uint32_t quarter = word & 3U;
    if (quarter != state.next_quarter) {
      Fail(std::format("quantized word {} arrives out of order", quarter));
    }
    state.block.at(quarter) = out;
    state.next_quarter = (quarter + 1) & 3U;
    if (quarter == 3) {
      const uint32_t address = (state.abuf + (word >> 2)) & kAbufMask;
      memories.log.Write(Mem::kAbuf, address);
      memories.abuf.at(address) = QuantizeBlock(state.block);
    }
  }
}

void VecEngine::RunSalu(const Word128& instruction) {
  const uint32_t a = Register(instruction.Field(109, 5));
  const uint32_t b = instruction.Field(103, 1) != 0
                         ? instruction.Field(0, 32)
                         : Register(instruction.Field(104, 5));
  Write(instruction.Field(114, 5), Salu(instruction.Field(119, 5), a, b));
}

void VecEngine::RunSmem(const Word128& instruction, Memories& memories) {
  const uint32_t rd = instruction.Field(116, 5);
  const uint32_t a = Register(instruction.Field(111, 5));
  const uint32_t column = instruction.Field(105, 1) != 0
                              ? instruction.Field(100, 5)
                              : Register(instruction.Field(106, 5)) & 31U;
  std::array<uint32_t, kTokens>& scalars =
      instruction.Field(99, 1) != 0 ? rows_.rs0 : rows_.rs1;
  const uint32_t word_address = (a + instruction.Field(0, 32)) & kFbufMask;
  const uint32_t smem_op = instruction.Field(121, 3);
  if (smem_op == kSmemLoad || smem_op == kSmemRowLoad) {
    memories.log.Read(Mem::kFbuf, word_address);
  }
  const FbufWord& word = memories.fbuf.at(word_address);
  switch (smem_op) {
    case kSmemLoad:
      Write(rd, word.at(column));
      break;
    case kSmemRowLoad:
      for (uint32_t t = 0; t < kTokens; ++t) {
        uint32_t value = word.at((t * kLanes) + (column & 7U));
        if (Fp32Exponent(value) == 0) {
          value = 0;
        } else if (instruction.Field(98, 1) != 0) {
          value ^= kFp32SignBit;
        }
        scalars.at(t) = value;
      }
      break;
    case kSmemRowMove:
      Write(rd, scalars.at(column & 3U));
      break;
    case kSmemMoveRow:
      scalars.fill(a);
      break;
    case kSmemCount:
      ++activity_.rn;
      rn_ = CountReciprocal(a);
      break;
    default:
      break;
  }
}

bool VecEngine::Taken(const Word128& instruction) const {
  const uint32_t a = Register(instruction.Field(117, 5));
  switch (instruction.Field(122, 2)) {
    case 0:
      return true;
    case 1:
      return a == 0;
    case 2:
      return a != 0;
    default:
      return false;
  }
}

void VecEngine::Drain() {
  in_flight_.reset();
  undrained_ = false;
  undrained_reduction_ = false;
  undrained_pbuf_at_mul_ = false;
}

void VecEngine::Write(uint32_t index, uint32_t value) {
  if (index != 0) {
    registers_.at(index) = value;
  }
}

}  // namespace hotarubi::iss
