#include "vec_engine.hpp"

#include <algorithm>
#include <array>
#include <bitset>
#include <cstdint>
#include <format>
#include <span>

#include "fp32.hpp"
#include "hotarubi/isa.hpp"
#include "hotarubi/require.hpp"
#include "lane.hpp"
#include "memories.hpp"
#include "scalar.hpp"
#include "word.hpp"

namespace hotarubi::iss {
namespace {

using isa::program::kKindBranch;
using isa::program::kKindPass;
using isa::program::kKindRet;
using isa::program::kKindSalu;
using isa::program::kKindSmem;
using isa::program::kKindTranspose;

constexpr uint32_t kSmemLoad = 0;
constexpr uint32_t kSmemRowMove = 1;
constexpr uint32_t kSmemCount = 2;

constexpr uint32_t kPcMask = kCbufDepth - 1;
constexpr uint32_t kFbufMask = kFbufDepth - 1;
constexpr uint32_t kAbufMask = kAbufDepth - 1;
constexpr uint32_t kPbufMask = kPbufDepth - 1;

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
      return (base + (w >> 2)) & kPbufMask;
    case PbufAccess::kTile:
      return (base + tile_row + (w & 3U)) & kPbufMask;
    case PbufAccess::kNone:
      break;
  }
  return 0;
}

bool Overlaps(const Pass& pass, const Word128& next) {
  return pass.rm != RowMask::kSet &&
         next.Field(isa::program::kKind) == kKindPass;
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
  WordResult result;
  for (uint32_t t = 0; t < kTokens; ++t) {
    const uint32_t pbuf_slice = pass.pa == PbufAccess::kTile ? t : slice;
    for (uint32_t l = 0; l < kLanes; ++l) {
      const uint32_t y =
          LaneOutput(pass,
                     {.x = in.x.at(PermutedIndex(pass.px, t, l)),
                      .x2 = in.x2.at((t * kLanes) + l),
                      .pbuf = in.pbuf.at((pbuf_slice * kLanes) + l),
                      .t = t,
                      .l = l,
                      .w = in.w,
                      .imm = in.imm,
                      .imm2 = in.imm2},
                     rows, tbuf);
      result.out.at((t * kLanes) + l) = y;
      result.reduced.at(t).at(l) = y;
    }
  }
  return result;
}

uint32_t LeadingSkipCycles(uint32_t run, bool follows_pass) {
  if (run == 0) {
    return 0;
  }
  return ((run + 1) / 2) - (follows_pass ? 1 : 0);
}

uint32_t InteriorSkipCycles(uint32_t run, uint32_t reads) {
  if (run == 0) {
    return 0;
  }
  return std::max((run / 2) + reads, kWordCycles) - kWordCycles;
}

uint32_t TrailingSkipCycles(uint32_t run) {
  return run == 0 ? 0 : (run - 1) / 2;
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
  pending_.reset();
  slot_ = 0;
  follows_pass_ = false;
  ready_slot_.fill(0);
  scalar_free_slot_ = 0;
  Initialize(command);
  uint32_t pc = (command.entry * isa::vec::kEntryWords) & kPcMask;
  while (true) {
    memories.log.Read(Mem::kCbuf, pc);
    const Word128 instruction = memories.cbuf.at(pc);
    pc = (pc + 1) & kPcMask;
    const uint32_t kind = instruction.Field(isa::program::kKind);
    if (kind != kKindPass) {
      follows_pass_ = false;
    }
    switch (kind) {
      case kKindPass:
        RunPass(instruction, memories.cbuf.at(pc), memories);
        break;
      case kKindSalu:
        ++activity_.salu;
        RunSalu(instruction);
        break;
      case kKindSmem:
        ++activity_.smem;
        Drain();
        RunSmem(instruction, memories);
        break;
      case kKindBranch:
        ++activity_.branches;
        if (Taken(instruction)) {
          pc = instruction.Field(isa::program::kImm) & kPcMask;
        }
        break;
      case kKindRet:
        return;
      case kKindTranspose:
        Drain();
        RunTranspose(instruction, memories);
        break;
      default:
        break;
    }
  }
}

void VecEngine::RunTranspose(const Word128& instruction, Memories& memories) {
  namespace tr = isa::transpose;
  const uint32_t src = Register(instruction.Field(tr::kRx)) & kFbufMask;
  const uint32_t src_word_stride =
      Register(instruction.Field(tr::kRsw)) & kFbufMask;
  const uint32_t dst = Register(instruction.Field(tr::kRd)) & kFbufMask;
  const uint32_t tiles = Register(instruction.Field(tr::kRt)) & 0x1FFU;
  const uint32_t dst_tile_stride =
      Register(instruction.Field(tr::kRdt)) & kFbufMask;
  const uint32_t src_tile_stride =
      Register(instruction.Field(tr::kRst)) & kFbufMask;
  const uint32_t dst_word_stride =
      Register(instruction.Field(tr::kRdw)) & kFbufMask;
  const uint32_t rotation = instruction.Field(tr::kRor) % 3;
  ++activity_.transposes;
  activity_.transpose_tiles += tiles;
  for (uint32_t q = 0; q < tiles; ++q) {
    std::array<FbufWord, 8> tile;
    for (uint32_t row = 0; row < 8; ++row) {
      const uint32_t address =
          (src + (q * src_tile_stride) + (row * src_word_stride)) & kFbufMask;
      memories.log.Read(Mem::kFbuf, address);
      tile.at(row) = memories.fbuf.at(address);
    }
    for (uint32_t row = 0; row < 8; ++row) {
      FbufWord output;
      for (uint32_t t = 0; t < 4; ++t) {
        for (uint32_t column = 0; column < 8; ++column) {
          const uint32_t source_row =
              ((column >> rotation) | (column << (3 - rotation))) & 7U;
          output.at((t * 8) + column) = tile.at(source_row).at((t * 8) + row);
        }
      }
      const uint32_t address =
          (dst + (q * dst_tile_stride) + (row * dst_word_stride)) & kFbufMask;
      memories.log.Write(Mem::kFbuf, address);
      memories.fbuf.at(address) = output;
    }
  }
}

uint32_t VecEngine::Register(uint32_t index) const {
  return index == 0 ? 0 : registers_.at(index);
}

void VecEngine::Initialize(const VecCommand& command) {
  quantize_ = command.quantize;
  registers_.at(1) = command.src & kFbufMask;
  registers_.at(2) = command.src2 & kFbufMask;
  registers_.at(3) = command.dst & kAbufMask;
  registers_.at(4) = command.rows_q & 0x1FFU;
  registers_.at(5) = command.n8 & 0xFFU;
  registers_.at(6) = command.p_base & kPbufMask;
  registers_.at(7) = command.aux & 0x7FFU;
  registers_.at(8) = command.imm;
  registers_.at(9) = command.quantize ? 1 : 0;
}

void VecEngine::RunPass(const Word128& instruction, const Word128& next,
                        Memories& memories) {
  PassState state;
  state.pass = DecodePass(instruction);
  const Pass& pass = state.pass;
  const uint32_t quads = Register(pass.rq) & 0x1FFU;
  const uint32_t words = Register(pass.rnw) & 0x3FFU;
  Require(quads != 0 && words != 0,
          std::format("a PASS has {} quads and {} words", quads, words));
  const uint32_t d = Register(pass.rd);
  const uint32_t n8 = Register(pass.rn8) & 0xFFU;
  state.src = Register(pass.rx) & kFbufMask;
  state.src2 = Register(pass.rx2) & kFbufMask;
  state.dst = d & kFbufMask;
  state.abuf = d & kAbufMask;
  state.pb = Register(pass.rpb) & kPbufMask;
  state.imm = Register(pass.rimm);
  state.imm2 = Register(pass.rimm2);
  ++activity_.passes;
  const bool reads_rows = pass.c0 == C0Source::kRs0 ||
                          pass.b == BSource::kRs1 || pass.sel == Select::kRs1;
  if (reads_rows || pass.red != Reduction::kNone) {
    AwaitContext(pass.ctx);
  }
  const bool overlaps = Overlaps(pass, next);
  RunQuads(state, quads, words, n8, !overlaps, memories);
  Require(state.next_quarter == 0,
          "a PASS ends inside a block of 4 quantized words");
  if (pass.red != Reduction::kNone) {
    const uint64_t complete = slot_ + kReductionLatencyWords;
    if (pass.sf != ScalarFn::kNone) {
      ++activity_.scalars;
      const uint64_t start = std::max(complete, scalar_free_slot_);
      const uint32_t cycles = ScalarProgramCycles(
          RowScalarProgram(pass.red, pass.sf, pass.srn, pass.seps));
      scalar_free_slot_ = start + ((cycles + 3) / 4);
      ready_slot_.at(pass.ctx) = scalar_free_slot_;
      ApplyRowScalar(pass, state.reductions, state.imm, rn_,
                     rows_.at(pass.ctx));
    } else {
      ready_slot_.at(pass.ctx) = complete;
    }
  }
  follows_pass_ = overlaps;
  if (overlaps) {
    ++activity_.overlaps;
    return;
  }
  ++activity_.drains;
  if (Target(pass.wr) == WriteTarget::kAbuf) {
    ++activity_.drains_quantized;
  }
  Drain();
}

void VecEngine::RunQuads(PassState& state, uint32_t quads, uint32_t words,
                         uint32_t n8, bool drains, Memories& memories) {
  const Pass& pass = state.pass;
  const uint32_t reads = pass.x2_en ? 2 : 1;
  uint64_t skip_cycles = 0;
  uint32_t run = 0;
  bool issued = false;
  for (uint32_t q = 0; q < quads; ++q) {
    if (pass.rm == RowMask::kSkip && !row_mask_.test(q)) {
      ++run;
      ++activity_.skipped_rows;
    } else {
      skip_cycles += issued ? InteriorSkipCycles(run, reads)
                            : LeadingSkipCycles(run, follows_pass_);
      run = 0;
      issued = true;
      activity_.words += words;
      bool nonzero = false;
      for (uint32_t w = 0; w < words; ++w) {
        nonzero = RunWord(state, w, memories) || nonzero;
      }
      if (pass.rm == RowMask::kSet) {
        row_mask_.set(q, nonzero);
      }
    }
    const uint32_t step = pass.x_s1 ? 1 : n8;
    state.src = (state.src + step) & kFbufMask;
    state.src2 = (state.src2 + step) & kFbufMask;
    state.dst = (state.dst + (pass.dst_s1 ? 1 : n8)) & kFbufMask;
    state.abuf = (state.abuf + (n8 >> 2)) & kAbufMask;
    state.tile_row = (state.tile_row + 4) & kPbufMask;
  }
  constexpr auto kDrainHidden =
      static_cast<uint32_t>(kOverlapLatencyWords * kWordCycles);
  const uint32_t tail =
      issued ? TrailingSkipCycles(run) : LeadingSkipCycles(run, follows_pass_);
  skip_cycles += drains ? std::max(tail, kDrainHidden) - kDrainHidden : tail;
  activity_.skip_cycles += skip_cycles;
  slot_ += (skip_cycles + kWordCycles - 1) / kWordCycles;
}

WriteTarget VecEngine::Target(WriteTarget target) const {
  if (target == WriteTarget::kFinal) {
    return quantize_ ? WriteTarget::kAbuf : WriteTarget::kFbuf;
  }
  return target;
}

void VecEngine::AwaitContext(uint32_t ctx) {
  const uint64_t ready = ready_slot_.at(ctx);
  if (ready > slot_) {
    activity_.context_stall_words += ready - slot_;
    slot_ = ready;
  }
}

bool VecEngine::RunWord(PassState& state, uint32_t w, Memories& memories) {
  const Pass& pass = state.pass;
  const FbufWord x = Operand(
      state, (state.src + WordOf(pass, pass.mx, w)) & kFbufMask, memories);
  const FbufWord x2 =
      pass.x2_en
          ? Operand(state, (state.src2 + WordOf(pass, pass.mx2, w)) & kFbufMask,
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
                                        rows_.at(pass.ctx), memories.tbuf);
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
  Store(state, WordOf(pass, pass.md, w), result.out, memories);
  ++slot_;
  return std::ranges::any_of(result.out, [](uint32_t v) { return v != 0; });
}

const FbufWord& VecEngine::Operand(const PassState& state, uint32_t address,
                                   Memories& memories) {
  if (state.written.test(address)) {
    Fail(std::format("a PASS reads FBUF word {} after writing it", address));
  }
  if (pending_.test(address)) {
    const uint64_t ready = pending_slot_.at(address) + kOverlapLatencyWords;
    if (ready > slot_) {
      activity_.overlap_stall_words += ready - slot_;
      slot_ = ready;
    }
  }
  memories.log.Read(Mem::kFbuf, address);
  return memories.fbuf.at(address);
}

void VecEngine::Store(PassState& state, uint32_t word, const FbufWord& out,
                      Memories& memories) {
  const WriteTarget target = Target(state.pass.wr);
  if (target == WriteTarget::kFbuf) {
    const uint32_t address = (state.dst + word) & kFbufMask;
    memories.log.Write(Mem::kFbuf, address);
    memories.fbuf.at(address) = out;
    state.written.set(address);
    pending_.set(address);
    pending_slot_.at(address) = slot_;
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
  namespace pg = isa::program;
  const uint32_t a = Register(instruction.Field(pg::kRa));
  const uint32_t b = instruction.Field(pg::kUseImm) != 0
                         ? instruction.Field(pg::kImm)
                         : Register(instruction.Field(pg::kRb));
  Write(instruction.Field(pg::kRd), Salu(instruction.Field(pg::kOp), a, b));
}

void VecEngine::RunSmem(const Word128& instruction, Memories& memories) {
  namespace pg = isa::program;
  const uint32_t rd = instruction.Field(pg::kRd);
  const uint32_t a = Register(instruction.Field(pg::kRa));
  const uint32_t column = instruction.Field(pg::kUseImm) != 0
                              ? instruction.Field(pg::kSmemCol)
                              : Register(instruction.Field(pg::kRb)) & 31U;
  const uint32_t ctx = instruction.Field(pg::kSmemCtx);
  switch (instruction.Field(pg::kOp)) {
    case kSmemLoad: {
      const uint32_t word_address =
          (a + instruction.Field(pg::kImm)) & kFbufMask;
      memories.log.Read(Mem::kFbuf, word_address);
      Write(rd, memories.fbuf.at(word_address).at(column));
      break;
    }
    case kSmemRowMove: {
      AwaitContext(ctx);
      const RowState& rows = rows_.at(ctx);
      Write(rd, (instruction.Field(pg::kSmemWhich) != 0 ? rows.rs0 : rows.rs1)
                    .at(column & 3U));
      break;
    }
    case kSmemCount:
      ++activity_.rn;
      rn_ = CountReciprocal(a);
      break;
    default:
      break;
  }
}

bool VecEngine::Taken(const Word128& instruction) const {
  const uint32_t a = Register(instruction.Field(isa::program::kRa));
  switch (instruction.Field(isa::program::kOp)) {
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

void VecEngine::Drain() { pending_.reset(); }

void VecEngine::Write(uint32_t index, uint32_t value) {
  if (index != 0) {
    registers_.at(index) = value;
  }
}

}  // namespace hotarubi::iss
