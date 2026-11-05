#include "lane.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>

#include "fp32.hpp"
#include "hotarubi/require.hpp"
#include "memories.hpp"
#include "word.hpp"

namespace hotarubi::iss {
namespace {

constexpr uint32_t kTableHalf = 512;
constexpr uint64_t kMask33 = (uint64_t{1} << 33) - 1;
constexpr uint64_t kMask34 = (uint64_t{1} << 34) - 1;
constexpr uint32_t kNullEntry = 0xFFFFFFFF;

template <typename Enum>
Enum Decode(uint32_t value, Enum last, Enum fallback) {
  return value <= static_cast<uint32_t>(last) ? static_cast<Enum>(value)
                                              : fallback;
}

bool Flag(const Word128& word, uint32_t bit) { return word.Field(bit, 1) != 0; }

uint32_t CompareKey(uint32_t bits) {
  const uint32_t value = Fp32Exponent(bits) == 0 ? 0 : bits;
  return (value >> 31) != 0 ? ~value : value ^ kFp32SignBit;
}

bool Holds(Compare relation, uint32_t left, uint32_t right) {
  switch (relation) {
    case Compare::kLt:
      return left < right;
    case Compare::kLe:
      return left <= right;
    case Compare::kEq:
      return left == right;
    case Compare::kNe:
      return left != right;
    case Compare::kGt:
      return left > right;
    case Compare::kGe:
      return left >= right;
    case Compare::kNone:
      break;
  }
  return true;
}

bool MaxGreaterEqual(uint32_t prod, uint32_t other) {
  const uint32_t prod_magnitude = prod & ~kFp32SignBit;
  const uint32_t other_magnitude = other & ~kFp32SignBit;
  if (prod_magnitude == 0 && other_magnitude == 0) {
    return true;
  }
  const bool prod_negative = (prod >> 31) != 0;
  if (prod_negative != ((other >> 31) != 0)) {
    return !prod_negative;
  }
  return prod_negative ? prod_magnitude <= other_magnitude
                       : prod_magnitude >= other_magnitude;
}

struct FixedPoint {
  uint32_t index = 0;
  uint32_t fraction = 0;
  int32_t exponent = 0;
  bool zero = false;
  bool infinite = false;
};

FixedPoint ExpFixedPoint(uint32_t prod) {
  const uint32_t exponent = Fp32Exponent(prod);
  const uint32_t significand = Fp32Significand(prod);
  const bool negative = (prod >> 31) != 0;
  uint64_t magnitude = 0;
  bool inexact = false;
  if (exponent >= 125) {
    const uint32_t shift = exponent - 125;
    magnitude = shift >= 33 ? 0 : (uint64_t{significand} << shift) & kMask33;
  } else if (exponent >= 101) {
    const uint32_t shift = 125 - exponent;
    magnitude = significand >> shift;
    inexact = (significand & ((1U << shift) - 1U)) != 0;
  } else {
    inexact = significand != 0;
  }
  const uint64_t fixed =
      negative ? (0 - (magnitude + (inexact ? 1U : 0U))) & kMask34 : magnitude;
  const auto high = static_cast<int32_t>((fixed >> 25) & 0x1FFU);
  const bool big = exponent >= 135;
  return {.index = static_cast<uint32_t>((fixed >> 16) & 0x1FFU),
          .fraction = static_cast<uint32_t>(fixed & 0xFFFFU),
          .exponent = high >= 256 ? high - 512 : high,
          .zero = big && negative,
          .infinite = big && !negative};
}

uint32_t C0Value(const Pass& pass, const LaneInputs& in, const RowState& rows) {
  switch (pass.c0) {
    case C0Source::kX2:
      return in.x2;
    case C0Source::kPbuf:
      return in.pbuf_pre;
    case C0Source::kRs0:
      return rows.rs0.at(in.t);
    case C0Source::kImm:
      return in.imm;
    case C0Source::kZero:
      break;
  }
  return 0;
}

uint32_t BValue(const Pass& pass, const LaneInputs& in, const RowState& rows,
                uint32_t pre) {
  switch (pass.b) {
    case BSource::kX2:
      return in.x2;
    case BSource::kPbuf:
      return in.pbuf_mul;
    case BSource::kRs1:
      return rows.rs1.at(in.t);
    case BSource::kImm:
      return in.imm;
    case BSource::kPre:
      return pre;
    case BSource::kLog2e:
      return kFp32Log2e;
    case BSource::kOne:
      break;
  }
  return kFp32One;
}

uint32_t IntegerResult(IntegerOp op, uint32_t x, uint32_t c0) {
  switch (op) {
    case IntegerOp::kAdd:
      return x + c0;
    case IntegerOp::kSub:
      return x - c0;
    case IntegerOp::kAnd:
      return x & c0;
    case IntegerOp::kOr:
      return x | c0;
    case IntegerOp::kXor:
      return x ^ c0;
    case IntegerOp::kShr1:
      return x >> 1;
    case IntegerOp::kRsub:
      return c0 - x;
    case IntegerOp::kNone:
      break;
  }
  return 0;
}

uint32_t FloatResult(const Pass& pass, const LaneInputs& in,
                     const RowState& rows, uint32_t c0, uint32_t c1,
                     std::span<const uint64_t> tbuf) {
  const uint32_t pre = Fp32Add(in.x, c0);
  const uint32_t prod = Fp32Mul(pre, BValue(pass, in, rows, pre));
  switch (pass.nl) {
    case Nonlinear::kAct:
      return Activation(prod, tbuf);
    case Nonlinear::kExp2:
      return Exp2(prod, tbuf);
    case Nonlinear::kTab01:
      return Table01(prod, tbuf);
    case Nonlinear::kNone:
      break;
  }
  return PostAdd(prod, c1, pass.c1, pass.neg_c1);
}

uint32_t SelectValue(Select sel, uint32_t c1, const LaneInputs& in,
                     const RowState& rows) {
  switch (sel) {
    case Select::kX2:
      return c1;
    case Select::kImm:
      return in.imm;
    case Select::kNull:
      return kNullEntry;
    case Select::kRs0:
      return rows.rs0.at(in.t);
    case Select::kRs1:
      return rows.rs1.at(in.t);
    case Select::kNone:
    case Select::kZero:
      break;
  }
  return 0;
}

}  // namespace

Pass DecodePass(const Word128& instruction) {
  const Word128& w = instruction;
  return {
      .rx = w.Field(119, 5),
      .rx2 = w.Field(114, 5),
      .rd = w.Field(109, 5),
      .rq = w.Field(104, 5),
      .rnw = w.Field(99, 5),
      .rn8 = w.Field(94, 5),
      .rpb = w.Field(89, 5),
      .rimm = w.Field(84, 5),
      .rimm2 = w.Field(79, 5),
      .x2_en = Flag(w, 78),
      .wx = static_cast<WordMap>(w.Field(76, 2)),
      .wx2 = static_cast<WordMap>(w.Field(74, 2)),
      .wd = static_cast<WordMap>(w.Field(72, 2)),
      .pa = Decode(w.Field(70, 2), PbufAccess::kTile, PbufAccess::kNone),
      .pa_at_mul = Flag(w, 69),
      .c0 = Decode(w.Field(66, 3), C0Source::kImm, C0Source::kZero),
      .b = Decode(w.Field(63, 3), BSource::kLog2e, BSource::kOne),
      .nl = static_cast<Nonlinear>(w.Field(61, 2)),
      .c1 = static_cast<C1Source>(w.Field(59, 2)),
      .neg_c1 = Flag(w, 58),
      .mask_en = Flag(w, 57),
      .red = Decode(w.Field(55, 2), Reduction::kMax, Reduction::kNone),
      .wr = static_cast<WriteTarget>(w.Field(53, 2)),
      .sc = Decode(w.Field(50, 3), RowScalar::kMax, RowScalar::kRecipSum),
      .px = Decode(w.Field(47, 3), Permutation::kT4b, Permutation::kNone),
      .pcol = w.Field(41, 3),
      .cmp = Decode(w.Field(38, 3), Compare::kGe, Compare::kNone),
      .cmp_src = static_cast<CompareSource>(w.Field(36, 2)),
      .cmp_int = Flag(w, 35),
      .sel = Decode(w.Field(32, 3), Select::kRs1, Select::kZero),
      .ialu = static_cast<IntegerOp>(w.Field(29, 3)),
      .dst_s1 = Flag(w, 28),
      .c0_pred = Flag(w, 27),
      .nodrain = Flag(w, 26),
      .x_s1 = Flag(w, 25),
  };
}

uint32_t WordOf(WordMap map, uint32_t w) {
  switch (map) {
    case WordMap::kX1:
      return (((w >> 2) & 0xFU) << 3) | (w & 3U);
    case WordMap::kX2:
      return (((w >> 2) & 0xFU) << 3) | 4U | (w & 3U);
    case WordMap::kS4:
      return (w & 0xFFU) << 2;
    case WordMap::kNorm:
      break;
  }
  return w;
}

uint32_t PermutedIndex(Permutation px, uint32_t pcol, uint32_t t, uint32_t l) {
  switch (px) {
    case Permutation::kBcast:
      return (t * kLanes) + pcol;
    case Permutation::kTswap1:
      return ((t ^ 1U) * kLanes) + l;
    case Permutation::kTswap2:
      return ((t ^ 2U) * kLanes) + l;
    case Permutation::kT4:
      return ((l & 3U) << 3) | (t << 1) | (l >> 2);
    case Permutation::kT4b:
      return ((t & 1U) << 4) | ((l >> 2) << 3) | ((t >> 1) << 2) | (l & 3U);
    case Permutation::kNone:
      break;
  }
  return (t * kLanes) + l;
}

bool Predicate(const Pass& pass, const LaneInputs& in) {
  if (pass.cmp == Compare::kNone) {
    return true;
  }
  const bool lane = pass.cmp_src == CompareSource::kLane;
  uint32_t left = in.x;
  if (pass.cmp_src == CompareSource::kX2Imm2) {
    left = in.x2;
  } else if (lane) {
    left = pass.cmp_int ? (in.t * kLanes) + in.l : in.l;
  }
  const uint32_t right = pass.cmp_src == CompareSource::kX2 ? in.x2 : in.imm2;
  if (pass.cmp_int || lane) {
    return Holds(pass.cmp, left ^ kFp32SignBit, right ^ kFp32SignBit);
  }
  return Holds(pass.cmp, CompareKey(left), CompareKey(right));
}

uint32_t LaneOutput(const Pass& pass, const LaneInputs& in,
                    const RowState& rows, std::span<const uint64_t> tbuf) {
  const bool predicate = Predicate(pass, in);
  const uint32_t c1 = pass.c1 == C1Source::kSignfix ? in.x : in.x2;
  if (pass.sel != Select::kNone && !predicate) {
    return SelectValue(pass.sel, c1, in, rows);
  }
  const uint32_t c0 = pass.c0_pred && !predicate ? 0 : C0Value(pass, in, rows);
  if (pass.ialu != IntegerOp::kNone) {
    return IntegerResult(pass.ialu, in.x, c0);
  }
  return FloatResult(pass, in, rows, c0, c1, tbuf);
}

uint32_t PostAdd(uint32_t prod, uint32_t c1, C1Source source, bool negate) {
  switch (source) {
    case C1Source::kMax: {
      const uint32_t other = negate ? 0 : c1;
      return Fp32Add(MaxGreaterEqual(prod, other) ? prod : other, 0);
    }
    case C1Source::kSignfix:
      return (c1 >> 31) != 0 ? Fp32Add(prod ^ kFp32SignBit, c1)
                             : Fp32Add(prod, 0);
    case C1Source::kX2:
      return Fp32Add(prod, negate ? c1 ^ kFp32SignBit : c1);
    case C1Source::kZero:
      break;
  }
  return Fp32Add(prod, 0);
}

uint32_t Interpolate(uint64_t entry, uint32_t fraction) {
  return Fp32Add(static_cast<uint32_t>(entry),
                 Fp32MulFraction(static_cast<uint32_t>(entry >> 32), fraction));
}

uint32_t Activation(uint32_t prod, std::span<const uint64_t> tbuf) {
  const uint32_t exponent = Fp32Exponent(prod);
  if (exponent >= 131) {
    return kFp32One;
  }
  const uint32_t significand = Fp32Significand(prod);
  uint32_t fixed = 0;
  if (exponent == 130) {
    fixed = significand << 1;
  } else if (exponent >= 105) {
    fixed = significand >> (129 - exponent);
  }
  return Interpolate(tbuf[kTableHalf + (fixed >> 16)], fixed & 0xFFFFU);
}

uint32_t Exp2(uint32_t prod, std::span<const uint64_t> tbuf) {
  const FixedPoint fixed = ExpFixedPoint(prod);
  const uint32_t y = Interpolate(tbuf[fixed.index], fixed.fraction);
  const int32_t exponent =
      static_cast<int32_t>(Fp32Exponent(y)) + fixed.exponent;
  if (fixed.zero || exponent <= 0 || Fp32Exponent(y) == 0) {
    return 0;
  }
  if (fixed.infinite || exponent >= 255) {
    return kFp32Infinity;
  }
  return (static_cast<uint32_t>(exponent) << 23) | (y & 0x7FFFFFU);
}

uint32_t Table01(uint32_t prod, std::span<const uint64_t> tbuf) {
  const FixedPoint fixed = ExpFixedPoint(prod);
  return Interpolate(tbuf[fixed.index], fixed.fraction);
}

uint32_t MaxKey(uint32_t bits) {
  return (bits >> 31) != 0 ? ~bits : bits ^ kFp32SignBit;
}

uint32_t Fp32Max(uint32_t a, uint32_t b) {
  return MaxKey(a) >= MaxKey(b) ? a : b;
}

uint8_t QuantizeElement(uint32_t value, uint32_t emax) {
  const uint32_t exponent = Fp32Exponent(value);
  const uint32_t significand = Fp32Significand(value);
  if (significand == 0) {
    return 0;
  }
  Require(exponent <= emax, "a quantized element exceeds its block maximum");
  const uint32_t shift = emax - exponent + 17;
  uint32_t magnitude = 0;
  if (shift < 25) {
    const uint32_t integer = significand >> shift;
    const bool guard = ((significand >> (shift - 1)) & 1U) != 0;
    const bool sticky = (significand & ((1U << (shift - 1)) - 1U)) != 0;
    magnitude = integer + (guard && (sticky || (integer & 1U) != 0) ? 1U : 0U);
  }
  if ((value >> 31) != 0) {
    return magnitude >= 128 ? 0x80
                            : static_cast<uint8_t>((256 - magnitude) & 0xFFU);
  }
  return magnitude >= 128 ? 0x7F : static_cast<uint8_t>(magnitude);
}

AbufWord QuantizeBlock(std::span<const FbufWord, 4> words) {
  AbufWord out{};
  for (uint32_t t = 0; t < kTokens; ++t) {
    uint32_t emax = 0;
    for (const FbufWord& word : words) {
      for (uint32_t l = 0; l < kLanes; ++l) {
        emax = std::max(emax, Fp32Exponent(word.at((t * kLanes) + l)));
      }
    }
    for (uint32_t qw = 0; qw < words.size(); ++qw) {
      for (uint32_t l = 0; l < kLanes; ++l) {
        out.at((t * kBlock) + (qw * kLanes) + l) =
            QuantizeElement(words[qw].at((t * kLanes) + l), emax);
      }
    }
    out.at(kAbufScaleOffset + t) = static_cast<uint8_t>(emax == 0 ? 127 : emax);
  }
  return out;
}

}  // namespace hotarubi::iss
