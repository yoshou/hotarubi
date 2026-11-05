#ifndef HOTARUBI_TOOLS_ISS_LANE_HPP_
#define HOTARUBI_TOOLS_ISS_LANE_HPP_

#include <array>
#include <cstdint>
#include <span>

#include "memories.hpp"
#include "word.hpp"

namespace hotarubi::iss {

enum class WordMap : uint8_t { kNorm = 0, kX1 = 1, kX2 = 2, kS4 = 3 };
enum class PbufAccess : uint8_t { kNone = 0, kRowVec = 1, kTile = 2 };
enum class C0Source : uint8_t {
  kZero = 0,
  kX2 = 1,
  kPbuf = 2,
  kRs0 = 3,
  kImm = 4
};
enum class BSource : uint8_t {
  kOne = 0,
  kX2 = 1,
  kPbuf = 2,
  kRs1 = 3,
  kImm = 4,
  kPre = 5,
  kLog2e = 6,
};
enum class Nonlinear : uint8_t { kNone = 0, kAct = 1, kExp2 = 2, kTab01 = 3 };
enum class C1Source : uint8_t { kZero = 0, kX2 = 1, kMax = 2, kSignfix = 3 };
enum class Reduction : uint8_t { kNone = 0, kSum = 1, kMax = 2 };
enum class WriteTarget : uint8_t {
  kNone = 0,
  kFbuf = 1,
  kAbuf = 2,
  kFinal = 3
};
enum class RowScalar : uint8_t {
  kNone = 0,
  kMean = 1,
  kRstd = 2,
  kNegMax = 3,
  kRecipSum = 4,
  kSum = 5,
  kMax = 6,
};
enum class Permutation : uint8_t {
  kNone = 0,
  kBcast = 1,
  kTswap1 = 2,
  kTswap2 = 3,
  kT4 = 4,
  kT4b = 5,
};
enum class Compare : uint8_t {
  kNone = 0,
  kLt = 1,
  kLe = 2,
  kEq = 3,
  kNe = 4,
  kGt = 5,
  kGe = 6,
};
enum class CompareSource : uint8_t {
  kImm2 = 0,
  kX2 = 1,
  kLane = 2,
  kX2Imm2 = 3
};
enum class Select : uint8_t {
  kNone = 0,
  kZero = 1,
  kX2 = 2,
  kImm = 3,
  kNull = 4,
  kRs0 = 5,
  kRs1 = 6,
};
enum class IntegerOp : uint8_t {
  kNone = 0,
  kAdd = 1,
  kSub = 2,
  kAnd = 3,
  kOr = 4,
  kXor = 5,
  kShr1 = 6,
  kRsub = 7,
};

struct Pass {
  uint32_t rx = 0;
  uint32_t rx2 = 0;
  uint32_t rd = 0;
  uint32_t rq = 0;
  uint32_t rnw = 0;
  uint32_t rn8 = 0;
  uint32_t rpb = 0;
  uint32_t rimm = 0;
  uint32_t rimm2 = 0;
  bool x2_en = false;
  WordMap wx = WordMap::kNorm;
  WordMap wx2 = WordMap::kNorm;
  WordMap wd = WordMap::kNorm;
  PbufAccess pa = PbufAccess::kNone;
  bool pa_at_mul = false;
  C0Source c0 = C0Source::kZero;
  BSource b = BSource::kOne;
  Nonlinear nl = Nonlinear::kNone;
  C1Source c1 = C1Source::kZero;
  bool neg_c1 = false;
  bool mask_en = false;
  Reduction red = Reduction::kNone;
  WriteTarget wr = WriteTarget::kNone;
  RowScalar sc = RowScalar::kNone;
  Permutation px = Permutation::kNone;
  uint32_t pcol = 0;
  Compare cmp = Compare::kNone;
  CompareSource cmp_src = CompareSource::kImm2;
  bool cmp_int = false;
  Select sel = Select::kNone;
  IntegerOp ialu = IntegerOp::kNone;
  bool dst_s1 = false;
  bool c0_pred = false;
  bool nodrain = false;
  bool x_s1 = false;
};

Pass DecodePass(const Word128& instruction);

uint32_t WordOf(WordMap map, uint32_t w);
uint32_t PermutedIndex(Permutation px, uint32_t pcol, uint32_t t, uint32_t l);

struct RowState {
  std::array<uint32_t, kTokens> rs0{};
  std::array<uint32_t, kTokens> rs1{};
};

struct LaneInputs {
  uint32_t x = 0;
  uint32_t x2 = 0;
  uint32_t pbuf_pre = 0;
  uint32_t pbuf_mul = 0;
  uint32_t t = 0;
  uint32_t l = 0;
  uint32_t imm = 0;
  uint32_t imm2 = 0;
};

uint32_t LaneOutput(const Pass& pass, const LaneInputs& in,
                    const RowState& rows, std::span<const uint64_t> tbuf);

bool Predicate(const Pass& pass, const LaneInputs& in);
uint32_t PostAdd(uint32_t prod, uint32_t c1, C1Source source, bool negate);
uint32_t Interpolate(uint64_t entry, uint32_t fraction);
uint32_t Activation(uint32_t prod, std::span<const uint64_t> tbuf);
uint32_t Exp2(uint32_t prod, std::span<const uint64_t> tbuf);
uint32_t Table01(uint32_t prod, std::span<const uint64_t> tbuf);
uint32_t MaxKey(uint32_t bits);
uint32_t Fp32Max(uint32_t a, uint32_t b);
uint8_t QuantizeElement(uint32_t value, uint32_t emax);
AbufWord QuantizeBlock(std::span<const FbufWord, 4> words);

}  // namespace hotarubi::iss

#endif  // HOTARUBI_TOOLS_ISS_LANE_HPP_
