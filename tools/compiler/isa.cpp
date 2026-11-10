#include "isa.hpp"

#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <string>
#include <vector>

#include "hotarubi/isa.hpp"

namespace hotarubi::compiler {
namespace {

Instruction Header(uint32_t opcode, uint32_t wait) {
  Instruction instruction;
  instruction.Set(isa::kOpcode, opcode).Set(isa::kWait, wait);
  return instruction;
}

Instruction EncodeDma(uint32_t opcode, const DmaFields& fields, uint32_t wait) {
  namespace d = isa::dma;
  Instruction instruction = Header(opcode, wait);
  instruction.Set(d::kDdrAddr, fields.ddr_addr)
      .Set(d::kMem, static_cast<uint32_t>(fields.mem))
      .Set(d::kAddr, fields.onchip)
      .Set(d::kRows, fields.rows)
      .Set(d::kRowWords, fields.row_words)
      .Set(d::kDdrStride, fields.ddr_stride)
      .Set(d::kKind, static_cast<uint32_t>(fields.kind))
      .Set(d::kTok, fields.tok)
      .Set(d::kAcc, fields.acc ? 1 : 0)
      .Set(d::kFlush, fields.flush ? 1 : 0);
  return instruction;
}

}  // namespace

Instruction& Instruction::Set(uint32_t lsb, uint32_t width, uint64_t value) {
  Require(width == 64 || value < (uint64_t{1} << width),
          std::format("value {:#x} does not fit in {} bits at bit {}", value,
                      width, lsb));
  for (uint32_t i = 0; i < width; ++i) {
    const uint32_t position = lsb + i;
    const uint32_t mask = 1U << (position % 32);
    uint32_t& word = words.at(position / 32);
    word = ((value >> i) & 1U) != 0 ? (word | mask) : (word & ~mask);
  }
  return *this;
}

uint64_t Instruction::Get(uint32_t lsb, uint32_t width) const {
  uint64_t value = 0;
  for (uint32_t i = 0; i < width; ++i) {
    const uint32_t position = lsb + i;
    value |=
        static_cast<uint64_t>((words.at(position / 32) >> (position % 32)) & 1U)
        << i;
  }
  return value;
}

std::string Instruction::Hex() const {
  return std::format("{:08x}{:08x}{:08x}{:08x}", words[3], words[2], words[1],
                     words[0]);
}

std::vector<std::byte> Serialize(std::span<const Instruction> instructions) {
  std::vector<std::byte> bytes;
  bytes.reserve(instructions.size() * sizeof(Instruction::words));
  for (const Instruction& instruction : instructions) {
    for (const uint32_t word : instruction.words) {
      for (uint32_t i = 0; i < 4; ++i) {
        bytes.push_back(static_cast<std::byte>(word >> (8 * i)));
      }
    }
  }
  return bytes;
}

Instruction EncodeDmaLoad(const DmaFields& fields, uint32_t wait) {
  return EncodeDma(isa::kOpDmaLoad, fields, wait);
}

Instruction EncodeDmaStore(const DmaFields& fields, uint32_t wait) {
  return EncodeDma(isa::kOpDmaStore, fields, wait);
}

Instruction EncodeGemm(const GemmFields& fields, uint32_t wait) {
  namespace g = isa::gemm;
  Instruction instruction = Header(isa::kOpGemm, wait);
  instruction.Set(g::kABase, fields.a_base)
      .Set(g::kAStride, fields.a_stride)
      .Set(g::kKb, fields.kb)
      .Set(g::kM, fields.m)
      .Set(g::kWBase, fields.w_base)
      .Set(g::kWStride, fields.w_stride)
      .Set(g::kN8, fields.n8)
      .Set(g::kCBase, fields.c_base)
      .Set(g::kWmode, static_cast<uint32_t>(fields.wmode))
      .Set(g::kAcc, fields.acc ? 1 : 0);
  return instruction;
}

Instruction EncodeVec(const VecFields& fields, uint32_t wait) {
  namespace v = isa::vec;
  Instruction instruction = Header(isa::kOpVec, wait);
  instruction.Set(v::kEntry, fields.entry)
      .Set(v::kQ, fields.q ? 1 : 0)
      .Set(v::kSrc, fields.src)
      .Set(v::kSrc2, fields.src2)
      .Set(v::kDst, fields.dst)
      .Set(v::kRowsQ, fields.rows_q)
      .Set(v::kN8, fields.n8)
      .Set(v::kPBase, fields.p_base)
      .Set(v::kAux, fields.aux)
      .Set(v::kImm, fields.imm);
  return instruction;
}

Instruction EncodeSignal(uint32_t id, uint32_t value, uint32_t wait) {
  Instruction instruction = Header(isa::kOpSignal, wait);
  instruction.Set(isa::signal::kId, id).Set(isa::signal::kValue, value);
  return instruction;
}

Instruction EncodeWaitHost(uint32_t value, uint32_t wait) {
  Instruction instruction = Header(isa::kOpWaitHost, wait);
  instruction.Set(isa::wait_host::kValue, value);
  return instruction;
}

Instruction EncodeHalt(uint32_t wait) { return Header(isa::kOpHalt, wait); }

}  // namespace hotarubi::compiler
