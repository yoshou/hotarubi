#ifndef HOTARUBI_TOOLS_ISS_LANE_HPP_
#define HOTARUBI_TOOLS_ISS_LANE_HPP_

#include <array>
#include <cstdint>
#include <span>

#include "memories.hpp"
#include "word.hpp"

namespace hotarubi::iss {

enum class MapMode : uint8_t { kNone = 0, kGroup = 1, kUpper = 2 };
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
};
enum class Nonlinear : uint8_t { kNone = 0, kAct = 1, kExp2 = 2, kTab01 = 3 };
enum class PostOp : uint8_t { kAdd = 0, kSub = 1, kMax = 2, kSignfix = 3 };
enum class Reduction : uint8_t { kNone = 0, kSum = 1, kMax = 2 };
enum class WriteTarget : uint8_t {
  kNone = 0,
  kFbuf = 1,
  kAbuf = 2,
  kFinal = 3
};
enum class ScalarFn : uint8_t {
  kNone = 0,
  kIdentity = 1,
  kNegate = 2,
  kReciprocal = 3,
  kRsqrt = 4,
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
enum class CompareLeft : uint8_t {
  kX = 0,
  kX2 = 1,
  kColumn = 2,
  kPosition = 3
};
enum class Select : uint8_t {
  kNone = 0,
  kZero = 1,
  kX2 = 2,
  kImm = 3,
  kNull = 4,
  kRs1 = 5,
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

enum class RowMask : uint8_t {
  kNone = 0,
  kSkip = 1,
  kSet = 2,
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
  uint32_t mg = 0;
  uint32_t ms = 0;
  MapMode mx = MapMode::kNone;
  MapMode mx2 = MapMode::kNone;
  MapMode md = MapMode::kNone;
  PbufAccess pa = PbufAccess::kNone;
  C0Source c0 = C0Source::kZero;
  BSource b = BSource::kOne;
  Nonlinear nl = Nonlinear::kNone;
  PostOp po = PostOp::kAdd;
  bool pv = false;
  bool mask_en = false;
  Reduction red = Reduction::kNone;
  WriteTarget wr = WriteTarget::kNone;
  ScalarFn sf = ScalarFn::kNone;
  bool st = false;
  bool srn = false;
  bool seps = false;
  uint32_t px = 0;
  Compare cmp = Compare::kNone;
  CompareLeft cl = CompareLeft::kX;
  bool cr = false;
  bool cmp_int = false;
  Select sel = Select::kNone;
  IntegerOp ialu = IntegerOp::kNone;
  bool dst_s1 = false;
  bool c0_pred = false;
  bool x_s1 = false;
  uint32_t ctx = 0;
  RowMask rm = RowMask::kNone;
};

Pass DecodePass(const Word128& instruction);

uint32_t WordOf(const Pass& pass, MapMode mode, uint32_t w);
uint32_t PermutedIndex(uint32_t px, uint32_t t, uint32_t l);

struct RowState {
  std::array<uint32_t, kTokens> rs0{};
  std::array<uint32_t, kTokens> rs1{};
};

struct LaneInputs {
  uint32_t x = 0;
  uint32_t x2 = 0;
  uint32_t pbuf = 0;
  uint32_t t = 0;
  uint32_t l = 0;
  uint32_t imm = 0;
  uint32_t imm2 = 0;
};

uint32_t LaneOutput(const Pass& pass, const LaneInputs& in,
                    const RowState& rows, std::span<const uint64_t> tbuf);

bool Predicate(const Pass& pass, const LaneInputs& in);
uint32_t PostAdd(uint32_t prod, uint32_t c1, PostOp op);
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
