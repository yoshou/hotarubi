#include "fp32_ref.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <utility>

namespace hotarubi::tb {

namespace {

constexpr uint32_t kExponentMask = 0x7F800000U;
constexpr uint32_t kFractionMask = 0x007FFFFFU;
constexpr uint32_t kHiddenBit = 0x00800000U;
constexpr int kAlignShift = 38;

uint32_t BiasedExponent(uint32_t a) { return (a & kExponentMask) >> 23; }

bool IsZero(uint32_t a) { return BiasedExponent(a) == 0; }

uint64_t Significand(uint32_t a) {
  return static_cast<uint64_t>((a & kFractionMask) | kHiddenBit);
}

bool IsNegative(uint32_t a) { return (a & kFp32SignBit) != 0; }

uint32_t Canonical(uint32_t a) { return IsZero(a) ? kFp32Zero : a; }

}  // namespace

uint32_t Fp32Round(bool negative, uint64_t magnitude, int exponent) {
  if (magnitude == 0) {
    return kFp32Zero;
  }
  int msb = 63 - std::countl_zero(magnitude);
  uint64_t kept = 0;
  if (msb > 23) {
    const int shift = msb - 23;
    kept = magnitude >> shift;
    const uint64_t rest = magnitude & ((uint64_t{1} << shift) - 1);
    const uint64_t half = uint64_t{1} << (shift - 1);
    if (rest > half || (rest == half && (kept & 1) != 0)) {
      ++kept;
    }
    if (kept == (uint64_t{1} << 24)) {
      kept >>= 1;
      ++msb;
    }
  } else {
    kept = magnitude << (23 - msb);
  }
  const int biased = msb + exponent + 127;
  const uint32_t sign = negative ? kFp32SignBit : 0U;
  if (biased <= 0) {
    return kFp32Zero;
  }
  if (biased >= 255) {
    return sign | kExponentMask;
  }
  return sign | (static_cast<uint32_t>(biased) << 23) |
         (static_cast<uint32_t>(kept) & kFractionMask);
}

uint32_t Fp32Add(uint32_t a, uint32_t b) {
  if (IsZero(a)) {
    return Canonical(b);
  }
  if (IsZero(b)) {
    return Canonical(a);
  }
  if ((a & ~kFp32SignBit) < (b & ~kFp32SignBit)) {
    std::swap(a, b);
  }
  const uint32_t diff = BiasedExponent(a) - BiasedExponent(b);
  const uint64_t big = Significand(a) << kAlignShift;
  const uint64_t small_wide = Significand(b) << kAlignShift;
  uint64_t small = 1;
  if (diff < 62) {
    small = small_wide >> diff;
    if ((small_wide & ((uint64_t{1} << diff) - 1)) != 0) {
      small |= 1;
    }
  }
  const uint64_t sum =
      IsNegative(a) == IsNegative(b) ? big + small : big - small;
  return Fp32Round(IsNegative(a), sum,
                   static_cast<int>(BiasedExponent(a)) - 150 - kAlignShift);
}

uint32_t Fp32Sub(uint32_t a, uint32_t b) { return Fp32Add(a, Fp32Neg(b)); }

uint32_t Fp32Mul(uint32_t a, uint32_t b) {
  if (IsZero(a) || IsZero(b)) {
    return kFp32Zero;
  }
  return Fp32Round(
      IsNegative(a) != IsNegative(b), Significand(a) * Significand(b),
      static_cast<int>(BiasedExponent(a) + BiasedExponent(b)) - 300);
}

uint32_t Fp32Neg(uint32_t a) { return a ^ kFp32SignBit; }

uint32_t Fp32FromInt(int64_t value, int exponent) {
  const bool negative = value < 0;
  const uint64_t magnitude = negative
                                 ? uint64_t{0} - static_cast<uint64_t>(value)
                                 : static_cast<uint64_t>(value);
  return Fp32Round(negative, magnitude, exponent);
}

uint32_t Fp32Recip(uint32_t a) {
  uint32_t y = 0x7EF311C7U - a;
  for (int i = 0; i < 4; ++i) {
    y = Fp32Mul(y, Fp32Sub(kFp32Two, Fp32Mul(a, y)));
  }
  return y;
}

uint32_t Fp32Rsqrt(uint32_t a) {
  uint32_t y = 0x5F375A86U - (a >> 1);
  for (int i = 0; i < 4; ++i) {
    y = Fp32Mul(y, Fp32Sub(kFp32OnePointFive,
                           Fp32Mul(Fp32Mul(Fp32Mul(a, y), y), kFp32Half)));
  }
  return y;
}

bool Fp32Less(uint32_t a, uint32_t b) {
  const auto key = [](uint32_t v) {
    const auto magnitude = static_cast<int64_t>(v & ~kFp32SignBit);
    return IsNegative(v) ? -magnitude : magnitude;
  };
  return key(a) < key(b);
}

uint32_t Fp32Max(uint32_t a, uint32_t b) { return Fp32Less(a, b) ? b : a; }

uint32_t Fp32FromDouble(double value) {
  if (value == 0.0) {
    return kFp32Zero;
  }
  int exponent = 0;
  const double fraction = std::frexp(std::fabs(value), &exponent);
  const auto magnitude = static_cast<uint64_t>(std::ldexp(fraction, 53));
  return Fp32Round(value < 0.0, magnitude, exponent - 53);
}

double Fp32ToDouble(uint32_t bits) {
  if (IsZero(bits)) {
    return 0.0;
  }
  const double magnitude =
      std::ldexp(static_cast<double>(Significand(bits)),
                 static_cast<int>(BiasedExponent(bits)) - 150);
  return IsNegative(bits) ? -magnitude : magnitude;
}

}  // namespace hotarubi::tb
