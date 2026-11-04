#ifndef HOTARUBI_TB_NPU_ISA_HPP_
#define HOTARUBI_TB_NPU_ISA_HPP_

#include <cstdint>

#include "npu_bench.hpp"

namespace hotarubi::tb {

inline constexpr uint32_t kWaitDma = 0x1;
inline constexpr uint32_t kWaitGemm = 0x4;
inline constexpr uint32_t kWaitVec = 0x8;
inline constexpr uint32_t kWaitAll = kWaitDma | kWaitGemm | kWaitVec;

inline constexpr uint32_t kOpNop = 0;
inline constexpr uint32_t kOpDmaLoad = 1;
inline constexpr uint32_t kOpDmaStore = 2;
inline constexpr uint32_t kOpGemm = 4;
inline constexpr uint32_t kOpVec = 5;
inline constexpr uint32_t kOpSignal = 6;
inline constexpr uint32_t kOpHalt = 7;
inline constexpr uint32_t kOpWaitHost = 8;

enum class Mem : uint8_t {
  kAbuf = 0,
  kWbuf = 1,
  kFbuf = 2,
  kPbuf = 3,
  kTbuf = 4,
  kCbuf = 5,
};

enum class DmaMode : uint8_t {
  kRaw = 0,
  kSplit8 = 1,
  kT32Split8 = 2,
  kGacc = 3,
};

inline constexpr uint32_t kDmaFlagIndex = 0x1;
inline constexpr uint32_t kDmaFlagRow = 0x2;
inline constexpr uint32_t kGaccFlagAcc = 0x1;
inline constexpr uint32_t kGaccFlagFlush = 0x2;
inline constexpr uint32_t kVecFlagQ = 0x1;

struct DmaArgs {
  uint32_t ddr_addr = 0;
  Mem mem = Mem::kFbuf;
  uint32_t onchip = 0;
  uint32_t rows = 1;
  uint32_t row_words = 1;
  uint32_t ddr_stride = 0;
  DmaMode mode = DmaMode::kRaw;
  uint32_t tok = 0;
  uint32_t flags = 0;
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
  uint32_t op = 0;
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

inline Instruction Header(uint32_t opcode, uint32_t wait, uint32_t flags) {
  Instruction instruction;
  instruction.Set(120, 8, opcode).Set(116, 4, wait).Set(112, 4, flags);
  return instruction;
}

inline Instruction Dma(uint32_t opcode, const DmaArgs& args) {
  Instruction instruction = Header(opcode, args.wait, args.flags);
  instruction.Set(80, 32, args.ddr_addr)
      .Set(77, 3, static_cast<uint32_t>(args.mem))
      .Set(57, 20, args.onchip)
      .Set(45, 12, args.rows)
      .Set(33, 12, args.row_words)
      .Set(9, 24, args.ddr_stride)
      .Set(7, 2, static_cast<uint32_t>(args.mode))
      .Set(0, 7, args.tok);
  return instruction;
}

inline Instruction DmaLoad(const DmaArgs& args) {
  return Dma(kOpDmaLoad, args);
}

inline Instruction DmaStore(const DmaArgs& args) {
  return Dma(kOpDmaStore, args);
}

inline Instruction Gemm(const GemmArgs& args) {
  Instruction instruction = Header(kOpGemm, args.wait, 0);
  instruction.Set(92, 20, args.a_base)
      .Set(82, 10, args.a_stride)
      .Set(72, 10, args.kb)
      .Set(62, 10, args.m)
      .Set(42, 20, args.w_base)
      .Set(32, 10, args.w_stride)
      .Set(24, 8, args.n8)
      .Set(4, 20, args.c_base)
      .Set(2, 2, args.wmode)
      .Set(1, 1, args.acc ? 1 : 0);
  return instruction;
}

inline Instruction Vec(const VecArgs& args) {
  Instruction instruction = Header(kOpVec, args.wait, args.q ? kVecFlagQ : 0);
  instruction.Set(107, 5, args.op)
      .Set(95, 12, args.src)
      .Set(83, 12, args.src2)
      .Set(70, 13, args.dst)
      .Set(61, 9, args.rows_q)
      .Set(53, 8, args.n8)
      .Set(43, 10, args.p_base)
      .Set(32, 11, args.aux)
      .Set(0, 32, args.imm);
  return instruction;
}

inline Instruction Signal(uint32_t id, uint32_t value, uint32_t wait = 0) {
  Instruction instruction = Header(kOpSignal, wait, 0);
  instruction.Set(104, 8, id).Set(0, 32, value);
  return instruction;
}

inline Instruction WaitHost(uint32_t value, uint32_t wait = 0) {
  Instruction instruction = Header(kOpWaitHost, wait, 0);
  instruction.Set(0, 32, value);
  return instruction;
}

inline Instruction Halt(uint32_t wait = kWaitAll) {
  return Header(kOpHalt, wait, 0);
}

inline Instruction Nop(uint32_t wait = 0) { return Header(kOpNop, wait, 0); }

}  // namespace hotarubi::tb

#endif  // HOTARUBI_TB_NPU_ISA_HPP_
