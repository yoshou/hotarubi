#ifndef HOTARUBI_TOOLS_COMPILER_ISA_HPP_
#define HOTARUBI_TOOLS_COMPILER_ISA_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "hotarubi/require.hpp"

namespace hotarubi::compiler {

struct Instruction {
  std::array<uint32_t, 4> words{};

  Instruction& Set(uint32_t lsb, uint32_t width, uint64_t value);
  [[nodiscard]] uint64_t Get(uint32_t lsb, uint32_t width) const;
  [[nodiscard]] std::string Hex() const;
};

enum class Mem : uint8_t {
  kAbuf = 0,
  kWbuf = 1,
  kFbuf = 2,
  kPbuf = 3,
  kTbuf = 4,
  kCbuf = 5,
};

inline constexpr int kMemCount = 6;
inline constexpr std::array<uint32_t, kMemCount> kMemDepth = {8192, 2048, 4096,
                                                              512,  1024, 1024};

enum class DmaMode : uint8_t {
  kRaw = 0,
  kSplit8 = 1,
  kT32Split8 = 2,
  kGacc = 3,
};

enum class WeightMode : uint8_t {
  kFp4 = 0,
  kW8Hi = 1,
  kW8Lo = 2,
};

enum class VecOp : uint8_t {
  kAdd = 0,
  kMul = 1,
  kAddP = 2,
  kMulP = 3,
  kScale = 4,
  kCopy = 5,
  kLayerNorm = 6,
  kRmsNorm = 7,
  kGelu = 8,
  kSilu = 9,
  kSiluMul = 10,
  kSoftmax = 11,
  kRope = 12,
  kQuant = 13,
  kRelu = 14,
  kMax = 15,
  kRun = 16,
};

inline constexpr uint32_t kWaitDma = 0x1;
inline constexpr uint32_t kWaitGemm = 0x4;
inline constexpr uint32_t kWaitVec = 0x8;

struct DmaFields {
  uint32_t ddr_addr = 0;
  Mem mem = Mem::kFbuf;
  uint32_t onchip = 0;
  uint32_t rows = 1;
  uint32_t row_words = 1;
  uint32_t ddr_stride = 0;
  DmaMode mode = DmaMode::kRaw;
  uint32_t tok = 0;
  uint32_t flags = 0;
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
  VecOp op = VecOp::kCopy;
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
