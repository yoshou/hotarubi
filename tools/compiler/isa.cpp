#include "isa.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace hotarubi::compiler {
namespace {

constexpr uint32_t kOpDmaLoad = 1;
constexpr uint32_t kOpDmaStore = 2;
constexpr uint32_t kOpGemm = 4;
constexpr uint32_t kOpVec = 5;
constexpr uint32_t kOpSignal = 6;
constexpr uint32_t kOpHalt = 7;
constexpr uint32_t kOpWaitHost = 8;

Instruction Header(uint32_t opcode, uint32_t wait, uint32_t flags) {
  Instruction instruction;
  instruction.Set(120, 8, opcode).Set(116, 4, wait).Set(112, 4, flags);
  return instruction;
}

Instruction EncodeDma(uint32_t opcode, const DmaFields& fields, uint32_t mode,
                      uint32_t wait) {
  Instruction instruction = Header(opcode, wait, fields.flags);
  instruction.Set(80, 32, fields.ddr_addr)
      .Set(77, 3, static_cast<uint32_t>(fields.mem))
      .Set(57, 20, fields.onchip)
      .Set(45, 12, fields.rows)
      .Set(33, 12, fields.row_words)
      .Set(9, 24, fields.ddr_stride)
      .Set(7, 2, mode)
      .Set(0, 7, fields.tok);
  return instruction;
}

}  // namespace

void Fail(std::string_view message) {
  std::println(stderr, "compiler error: {}", message);
  std::abort();
}

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
  return EncodeDma(kOpDmaLoad, fields, static_cast<uint32_t>(fields.mode),
                   wait);
}

Instruction EncodeDmaStore(const DmaFields& fields, uint32_t wait) {
  return EncodeDma(kOpDmaStore, fields, 0, wait);
}

Instruction EncodeGemm(const GemmFields& fields, uint32_t wait) {
  Instruction instruction = Header(kOpGemm, wait, 0);
  instruction.Set(92, 20, fields.a_base)
      .Set(82, 10, fields.a_stride)
      .Set(72, 10, fields.kb)
      .Set(62, 10, fields.m)
      .Set(42, 20, fields.w_base)
      .Set(32, 10, fields.w_stride)
      .Set(24, 8, fields.n8)
      .Set(4, 20, fields.c_base)
      .Set(2, 2, static_cast<uint32_t>(fields.wmode))
      .Set(1, 1, fields.acc ? 1 : 0);
  return instruction;
}

Instruction EncodeVec(const VecFields& fields, uint32_t wait) {
  Instruction instruction = Header(kOpVec, wait, fields.q ? 1 : 0);
  instruction.Set(107, 5, static_cast<uint32_t>(fields.op))
      .Set(95, 12, fields.src)
      .Set(83, 12, fields.src2)
      .Set(70, 13, fields.dst)
      .Set(61, 9, fields.rows_q)
      .Set(53, 8, fields.n8)
      .Set(43, 10, fields.p_base)
      .Set(32, 11, fields.aux)
      .Set(0, 32, fields.imm);
  return instruction;
}

Instruction EncodeSignal(uint32_t id, uint32_t value, uint32_t wait) {
  Instruction instruction = Header(kOpSignal, wait, 0);
  instruction.Set(104, 8, id).Set(0, 32, value);
  return instruction;
}

Instruction EncodeWaitHost(uint32_t value, uint32_t wait) {
  Instruction instruction = Header(kOpWaitHost, wait, 0);
  instruction.Set(0, 32, value);
  return instruction;
}

Instruction EncodeHalt(uint32_t wait) { return Header(kOpHalt, wait, 0); }

}  // namespace hotarubi::compiler
