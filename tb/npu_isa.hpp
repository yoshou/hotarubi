#ifndef HOTARUBI_TB_NPU_ISA_HPP_
#define HOTARUBI_TB_NPU_ISA_HPP_

#include <cstdint>

#include "hotarubi/isa.hpp"
#include "npu_bench.hpp"

namespace hotarubi::tb {

using isa::kWaitAll;
using isa::kWaitDma;
using isa::kWaitGemm;
using isa::kWaitVec;

using isa::kOpDmaLoad;
using isa::kOpDmaStore;
using isa::kOpGemm;
using isa::kOpHalt;
using isa::kOpNop;
using isa::kOpSignal;
using isa::kOpVec;
using isa::kOpWaitHost;

using isa::Mem;

enum class DmaKind : uint8_t {
  kRaw = isa::dma::kKindRaw,
  kRow = isa::dma::kKindRow,
  kRowIndexed = isa::dma::kKindRowIndexed,
  kSplit8 = isa::dma::kKindSplit8,
  kT32Split8 = isa::dma::kKindT32Split8,
  kGacc = isa::dma::kKindGacc,
};

struct DmaArgs {
  uint32_t ddr_addr = 0;
  Mem mem = Mem::kFbuf;
  uint32_t onchip = 0;
  uint32_t rows = 1;
  uint32_t row_words = 1;
  uint32_t ddr_stride = 0;
  DmaKind kind = DmaKind::kRaw;
  uint32_t tok = 0;
  bool acc = false;
  bool flush = false;
  uint32_t wait = 0;
};

struct GemmArgs {
  uint32_t a_base = 0;
  uint32_t a_stride = 0;
  uint32_t kb = 0;
  uint32_t m = 0;
  uint32_t w_base = 0;
  uint32_t w_stride = 0;
  uint32_t n8 = 0;
  uint32_t c_base = 0;
  uint32_t wmode = 0;
  bool acc = false;
  uint32_t wait = 0;
};

struct VecArgs {
  uint32_t entry = 0;
  uint32_t src = 0;
  uint32_t src2 = 0;
  uint32_t dst = 0;
  uint32_t rows_q = 0;
  uint32_t n8 = 0;
  uint32_t p_base = 0;
  uint32_t aux = 0;
  uint32_t imm = 0;
  bool q = false;
  uint32_t wait = 0;
};

inline Instruction Header(uint32_t opcode, uint32_t wait) {
  Instruction instruction;
  instruction.Set(isa::kOpcode, opcode).Set(isa::kWait, wait);
  return instruction;
}

inline Instruction Dma(uint32_t opcode, const DmaArgs& args) {
  namespace d = isa::dma;
  Instruction instruction = Header(opcode, args.wait);
  instruction.Set(d::kDdrAddr, args.ddr_addr / 16)
      .Set(d::kMem, static_cast<uint32_t>(args.mem))
      .Set(d::kAddr, args.onchip)
      .Set(d::kRows, args.rows)
      .Set(d::kRowWords, args.row_words)
      .Set(d::kDdrStride, args.ddr_stride / 16)
      .Set(d::kKind, static_cast<uint32_t>(args.kind))
      .Set(d::kTok, args.tok)
      .Set(d::kAcc, args.acc ? 1 : 0)
      .Set(d::kFlush, args.flush ? 1 : 0);
  return instruction;
}

inline Instruction DmaLoad(const DmaArgs& args) {
  return Dma(kOpDmaLoad, args);
}

inline Instruction DmaStore(const DmaArgs& args) {
  return Dma(kOpDmaStore, args);
}

inline Instruction Gemm(const GemmArgs& args) {
  namespace g = isa::gemm;
  Instruction instruction = Header(kOpGemm, args.wait);
  instruction.Set(g::kABase, args.a_base)
      .Set(g::kAStride, args.a_stride)
      .Set(g::kKb, args.kb)
      .Set(g::kM, args.m)
      .Set(g::kWBase, args.w_base)
      .Set(g::kWStride, args.w_stride)
      .Set(g::kN8, args.n8)
      .Set(g::kCBase, args.c_base)
      .Set(g::kWmode, args.wmode)
      .Set(g::kAcc, args.acc ? 1 : 0);
  return instruction;
}

inline Instruction Vec(const VecArgs& args) {
  namespace v = isa::vec;
  Instruction instruction = Header(kOpVec, args.wait);
  instruction.Set(v::kEntry, args.entry)
      .Set(v::kQ, args.q ? 1 : 0)
      .Set(v::kSrc, args.src)
      .Set(v::kSrc2, args.src2)
      .Set(v::kDst, args.dst)
      .Set(v::kRowsQ, args.rows_q)
      .Set(v::kN8, args.n8)
      .Set(v::kPBase, args.p_base)
      .Set(v::kAux, args.aux)
      .Set(v::kImm, args.imm);
  return instruction;
}

inline Instruction Signal(uint32_t id, uint32_t value, uint32_t wait = 0) {
  Instruction instruction = Header(kOpSignal, wait);
  instruction.Set(isa::signal::kId, id).Set(isa::signal::kValue, value);
  return instruction;
}

inline Instruction WaitHost(uint32_t value, uint32_t wait = 0) {
  Instruction instruction = Header(kOpWaitHost, wait);
  instruction.Set(isa::wait_host::kValue, value);
  return instruction;
}

inline Instruction Halt(uint32_t wait = kWaitAll) {
  return Header(kOpHalt, wait);
}

inline Instruction Nop(uint32_t wait = 0) { return Header(kOpNop, wait); }

}  // namespace hotarubi::tb

#endif  // HOTARUBI_TB_NPU_ISA_HPP_
