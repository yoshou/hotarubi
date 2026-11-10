#ifndef HOTARUBI_TOOLS_COMPILER_ISA_HPP_
#define HOTARUBI_TOOLS_COMPILER_ISA_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "hotarubi/isa.hpp"
#include "hotarubi/require.hpp"

namespace hotarubi::compiler {

struct Instruction {
  std::array<uint32_t, 4> words{};

  Instruction& Set(uint32_t lsb, uint32_t width, uint64_t value);
  Instruction& Set(isa::Field field, uint64_t value) {
    return Set(field.lsb, field.width, value);
  }
  [[nodiscard]] uint64_t Get(uint32_t lsb, uint32_t width) const;
  [[nodiscard]] uint64_t Get(isa::Field field) const {
    return Get(field.lsb, field.width);
  }
  [[nodiscard]] std::string Hex() const;
};

using isa::Mem;

inline constexpr int kMemCount = 6;
inline constexpr std::array<uint32_t, kMemCount> kMemDepth = {8192, 2048, 4096,
                                                              512,  1024, 1024};

enum class DmaKind : uint8_t {
  kRaw = isa::dma::kKindRaw,
  kRow = isa::dma::kKindRow,
  kRowIndexed = isa::dma::kKindRowIndexed,
  kSplit8 = isa::dma::kKindSplit8,
  kT32Split8 = isa::dma::kKindT32Split8,
  kGacc = isa::dma::kKindGacc,
};

using isa::gemm::WeightMode;

enum class VecOp : uint8_t {
  kAdd = isa::library::kAdd,
  kMul = isa::library::kMul,
  kMax = isa::library::kMax,
  kAddP = isa::library::kAddP,
  kMulP = isa::library::kMulP,
  kScale = isa::library::kScale,
  kCopy = isa::library::kCopy,
  kRelu = isa::library::kRelu,
  kAct = isa::library::kAct,
  kActMul = isa::library::kActMul,
  kRope = isa::library::kRope,
  kSoftmax = isa::library::kSoftmax,
  kLayerNorm = isa::library::kLayerNorm,
  kRmsNorm = isa::library::kRmsNorm,
};

using isa::kWaitAll;
using isa::kWaitDma;
using isa::kWaitGemm;
using isa::kWaitVec;

struct DmaFields {
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
};

struct GemmFields {
  uint32_t a_base = 0;
  uint32_t a_stride = 0;
  uint32_t kb = 0;
  uint32_t m = 0;
  uint32_t w_base = 0;
  uint32_t w_stride = 0;
  uint32_t n8 = 0;
  uint32_t c_base = 0;
  WeightMode wmode = WeightMode::kFp4;
  bool acc = false;
};

struct VecFields {
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
};

std::vector<std::byte> Serialize(std::span<const Instruction> instructions);

Instruction EncodeDmaLoad(const DmaFields& fields, uint32_t wait);
Instruction EncodeDmaStore(const DmaFields& fields, uint32_t wait);
Instruction EncodeGemm(const GemmFields& fields, uint32_t wait);
Instruction EncodeVec(const VecFields& fields, uint32_t wait);
Instruction EncodeSignal(uint32_t id, uint32_t value, uint32_t wait);
Instruction EncodeWaitHost(uint32_t value, uint32_t wait);
Instruction EncodeHalt(uint32_t wait);

}  // namespace hotarubi::compiler

#endif  // HOTARUBI_TOOLS_COMPILER_ISA_HPP_
