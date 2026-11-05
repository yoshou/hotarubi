#include "reference.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "hotarubi/exact.hpp"

namespace hotarubi::iss::reference {
namespace {

constexpr uint32_t kSign = 0x80000000;
constexpr uint32_t kInfinity = 0x7F800000;
constexpr uint32_t kNegativeInfinity = 0xFF800000;
constexpr uint32_t kOne = 0x3F800000;
constexpr uint32_t kTwo = 0x40000000;
constexpr uint32_t kHalf = 0x3F000000;
constexpr uint32_t kOnePointFive = 0x3FC00000;
constexpr int64_t kSignificandBits = 24;
constexpr int64_t kMinExponent = -126;
constexpr int64_t kMaxExponent = 127;
constexpr int64_t kTableHalf = 512;
constexpr int64_t kFixedBits = 25;
constexpr int kNewtonSteps = 4;

uint32_t Tree(std::span<const uint32_t, 8> v) {
  return Add(Add(Add(v[0], v[1]), Add(v[2], v[3])),
             Add(Add(v[4], v[5]), Add(v[6], v[7])));
}

uint64_t OrderKey(uint32_t bits) {
  const uint64_t magnitude = bits & ~kSign;
  return (bits & kSign) != 0 ? (uint64_t{1} << 31) - 1 - magnitude
                             : (uint64_t{1} << 31) + magnitude;
}

}  // namespace

int64_t NearestEven(const exact::Real& value) {
  int64_t integer = value.Floor();
  const exact::Real excess =
      value - exact::Real::Integer(integer) - exact::Real::Ratio(1, 2);
  if ((!excess.IsZero() && !excess.Negative()) ||
      (excess.IsZero() && integer % 2 != 0)) {
    ++integer;
  }
  return integer;
}

exact::Real Value(uint32_t bits) {
  if (((bits >> 23) & 0xFFU) == 0) {
    return {};
  }
  return exact::Real::Binary32(bits);
}

uint32_t Round(const exact::Real& value) {
  if (value.IsZero()) {
    return 0;
  }
  const uint32_t sign = value.Negative() ? kSign : 0;
  const exact::Real magnitude = value.Negative() ? value.Negated() : value;
  int64_t exponent = magnitude.TopBit();
  if (exponent < kMinExponent - 1) {
    return 0;
  }
  if (exponent > kMaxExponent) {
    return sign | kInfinity;
  }
  const exact::Real scaled = magnitude.Scaled(kSignificandBits - 1 - exponent);
  int64_t significand = scaled.Floor();
  const exact::Real excess =
      scaled - exact::Real::Integer(significand) - exact::Real::Ratio(1, 2);
  const bool above = !excess.IsZero() && !excess.Negative();
  const bool tie_to_odd = excess.IsZero() && significand % 2 != 0;
  if (above || tie_to_odd) {
    ++significand;
  }
  if (significand == (int64_t{1} << kSignificandBits)) {
    significand >>= 1;
    ++exponent;
  }
  if (exponent < kMinExponent) {
    return 0;
  }
  if (exponent > kMaxExponent) {
    return sign | kInfinity;
  }
  return sign | (static_cast<uint32_t>(exponent + 127) << 23) |
         (static_cast<uint32_t>(significand) & 0x7FFFFFU);
}

uint32_t Add(uint32_t a, uint32_t b) { return Round(Value(a) + Value(b)); }

uint32_t Mul(uint32_t a, uint32_t b) { return Round(Value(a) * Value(b)); }

uint32_t Negate(uint32_t bits) {
  return (bits & ~kSign) == 0 ? 0 : bits ^ kSign;
}

uint32_t MulFraction(uint32_t s, uint32_t fraction) {
  return Round(Value(s) * exact::Real::Integer(fraction).Scaled(-16));
}

uint32_t FromInt(int64_t value) { return Round(exact::Real::Integer(value)); }

uint32_t Reciprocal(uint32_t a) {
  uint32_t y = 0x7EF311C7U - a;
  for (int i = 0; i < kNewtonSteps; ++i) {
    y = Mul(y, Add(kTwo, Negate(Mul(a, y))));
  }
  return y;
}

uint32_t ReciprocalSqrt(uint32_t a) {
  uint32_t y = 0x5F375A86U - (a >> 1);
  for (int i = 0; i < kNewtonSteps; ++i) {
    y = Mul(y, Add(kOnePointFive, Negate(Mul(Mul(Mul(a, y), y), kHalf))));
  }
  return y;
}

uint32_t RowSum(std::span<const std::array<uint32_t, 8>> groups) {
  std::array<uint32_t, 8> accumulators{};
  for (size_t j = 0; j < groups.size(); ++j) {
    uint32_t& acc = accumulators.at(j % accumulators.size());
    acc = Add(acc, Tree(groups[j]));
  }
  return Tree(accumulators);
}

uint32_t RowMax(std::span<const uint32_t> values) {
  uint32_t best = kNegativeInfinity;
  for (const uint32_t value : values) {
    if (OrderKey(value) > OrderKey(best)) {
      best = value;
    }
  }
  return best;
}

uint32_t Interpolate(uint64_t entry, uint32_t fraction) {
  return Add(static_cast<uint32_t>(entry),
             MulFraction(static_cast<uint32_t>(entry >> 32), fraction));
}

uint32_t Sigma(uint32_t x, std::span<const uint64_t> table) {
  const exact::Real value = Value(x);
  const exact::Real magnitude = value.Negative() ? value.Negated() : value;
  if (!magnitude.IsZero() && magnitude.TopBit() >= 4) {
    return kOne;
  }
  const int64_t fixed = magnitude.Scaled(21).Floor();
  return Interpolate(table[static_cast<size_t>(kTableHalf + (fixed >> 16))],
                     static_cast<uint32_t>(fixed & 0xFFFF));
}

uint32_t Pow2(uint32_t t, std::span<const uint64_t> table) {
  const exact::Real value = Value(t);
  if (!value.IsZero() && value.TopBit() >= 8) {
    return value.Negative() ? 0 : kInfinity;
  }
  const int64_t fixed = value.Scaled(kFixedBits).Floor();
  const int64_t n = fixed >> kFixedBits;
  const int64_t fraction = fixed - (n << kFixedBits);
  const uint32_t y = Interpolate(table[static_cast<size_t>(fraction >> 16)],
                                 static_cast<uint32_t>(fraction & 0xFFFF));
  const int64_t exponent = static_cast<int64_t>((y >> 23) & 0xFFU) + n;
  if (((y >> 23) & 0xFFU) == 0 || exponent <= 0) {
    return 0;
  }
  if (exponent >= 255) {
    return kInfinity;
  }
  return (static_cast<uint32_t>(exponent) << 23) | (y & 0x7FFFFFU);
}

uint32_t Fraction01(uint32_t t, std::span<const uint64_t> table) {
  const int64_t fixed = Value(t).Scaled(kFixedBits).Floor();
  return Interpolate(table[static_cast<size_t>(fixed >> 16)],
                     static_cast<uint32_t>(fixed & 0xFFFF));
}

Mxint8Block EncodeMxint8(std::span<const uint32_t, 32> values) {
  uint32_t largest = 0;
  for (const uint32_t value : values) {
    largest = std::max(largest, (value >> 23) & 0xFFU);
  }
  const int64_t e = largest == 0 ? 0 : static_cast<int64_t>(largest) - 127;
  Mxint8Block block{.scale = static_cast<uint8_t>(e + 127)};
  for (size_t k = 0; k < values.size(); ++k) {
    const int64_t q = std::clamp<int64_t>(
        NearestEven(Value(values[k]).Scaled(6 - e)), -128, 127);
    block.q.at(k) = static_cast<uint8_t>(q & 0xFF);
  }
  return block;
}

}  // namespace hotarubi::iss::reference
