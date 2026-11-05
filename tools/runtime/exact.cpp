#include "hotarubi/exact.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <format>
#include <utility>
#include <vector>

#include "hotarubi/require.hpp"

namespace hotarubi::exact {
namespace {

using Limbs = std::vector<uint32_t>;

constexpr int64_t kLimbBits = 32;
constexpr int64_t kMargin = 200;
constexpr int64_t kSeriesGuard = 16;
constexpr int64_t kExpReduction = 32;
constexpr int kReciprocalSteps = 6;
constexpr int kSqrtSteps = 14;
constexpr int64_t kIntegerLimit = int64_t{1} << 62;

void Trim(Limbs& a) {
  while (!a.empty() && a.back() == 0) {
    a.pop_back();
  }
}

int64_t BitLength(const Limbs& a) {
  if (a.empty()) {
    return 0;
  }
  return (kLimbBits * static_cast<int64_t>(a.size() - 1)) +
         (kLimbBits - std::countl_zero(a.back()));
}

int Compare(const Limbs& a, const Limbs& b) {
  if (a.size() != b.size()) {
    return a.size() < b.size() ? -1 : 1;
  }
  for (size_t i = a.size(); i > 0; --i) {
    if (a[i - 1] != b[i - 1]) {
      return a[i - 1] < b[i - 1] ? -1 : 1;
    }
  }
  return 0;
}

Limbs Add(const Limbs& a, const Limbs& b) {
  Limbs out(std::max(a.size(), b.size()) + 1, 0);
  uint64_t carry = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    const uint64_t sum =
        carry + (i < a.size() ? a[i] : 0U) + (i < b.size() ? b[i] : 0U);
    out[i] = static_cast<uint32_t>(sum);
    carry = sum >> kLimbBits;
  }
  Trim(out);
  return out;
}

Limbs Subtract(const Limbs& a, const Limbs& b) {
  Limbs out(a.size(), 0);
  int64_t borrow = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    int64_t value = static_cast<int64_t>(a[i]) - borrow -
                    static_cast<int64_t>(i < b.size() ? b[i] : 0U);
    borrow = value < 0 ? 1 : 0;
    if (value < 0) {
      value += int64_t{1} << kLimbBits;
    }
    out[i] = static_cast<uint32_t>(value);
  }
  Trim(out);
  return out;
}

Limbs Multiply(const Limbs& a, const Limbs& b) {
  Limbs out(a.size() + b.size(), 0);
  for (size_t i = 0; i < a.size(); ++i) {
    uint64_t carry = 0;
    for (size_t j = 0; j < b.size(); ++j) {
      const uint64_t product =
          (static_cast<uint64_t>(a[i]) * b[j]) + out[i + j] + carry;
      out[i + j] = static_cast<uint32_t>(product);
      carry = product >> kLimbBits;
    }
    out[i + b.size()] = static_cast<uint32_t>(carry);
  }
  Trim(out);
  return out;
}

Limbs ShiftLeft(const Limbs& a, int64_t bits) {
  if (a.empty()) {
    return a;
  }
  const auto limbs = static_cast<size_t>(bits / kLimbBits);
  const auto rest = static_cast<uint32_t>(bits % kLimbBits);
  Limbs out(a.size() + limbs + 1, 0);
  for (size_t i = 0; i < a.size(); ++i) {
    const uint64_t value = static_cast<uint64_t>(a[i]) << rest;
    out[i + limbs] |= static_cast<uint32_t>(value);
    out[i + limbs + 1] |= static_cast<uint32_t>(value >> kLimbBits);
  }
  Trim(out);
  return out;
}

Limbs ShiftRight(const Limbs& a, int64_t bits, bool& lost) {
  const auto limbs = static_cast<size_t>(bits / kLimbBits);
  const auto rest = static_cast<uint32_t>(bits % kLimbBits);
  if (limbs >= a.size()) {
    lost = lost || !a.empty();
    return {};
  }
  for (size_t i = 0; i < limbs; ++i) {
    lost = lost || a[i] != 0;
  }
  if (rest != 0 && (a[limbs] & ((1U << rest) - 1)) != 0) {
    lost = true;
  }
  Limbs out(a.size() - limbs, 0);
  for (size_t i = 0; i < out.size(); ++i) {
    const uint64_t low = a[i + limbs];
    const uint64_t high = i + limbs + 1 < a.size() ? a[i + limbs + 1] : 0U;
    out[i] = static_cast<uint32_t>(((high << kLimbBits) | low) >> rest);
  }
  Trim(out);
  return out;
}

Limbs LowBits(const Limbs& a, int64_t bits) {
  const auto limbs = static_cast<size_t>((bits + kLimbBits - 1) / kLimbBits);
  Limbs out(a.begin(),
            a.begin() + static_cast<std::ptrdiff_t>(std::min(limbs, a.size())));
  if (bits % kLimbBits != 0 && out.size() == limbs) {
    out.back() &= (1U << (bits % kLimbBits)) - 1;
  }
  Trim(out);
  return out;
}

Limbs PowerOfTwo(int64_t bit) {
  Limbs out(static_cast<size_t>(bit / kLimbBits) + 1, 0);
  out.back() = 1U << (bit % kLimbBits);
  return out;
}

bool Negligible(const Real& term, const Real& sum) {
  return term.IsZero() ||
         (!sum.IsZero() &&
          term.TopBit() < sum.TopBit() - (Real::kBits + kSeriesGuard));
}

Real One() { return Real::Integer(1); }

Real Reciprocal(const Real& b) {
  Require(!b.IsZero(), "division by zero");
  const Real seed = One().DividedBy(static_cast<uint32_t>(
      b.Negative() ? b.Negated().Scaled(31 - b.TopBit()).Floor()
                   : b.Scaled(31 - b.TopBit()).Floor()));
  Real x = seed.Scaled(31 - b.TopBit());
  if (b.Negative()) {
    x = x.Negated();
  }
  const Real two = Real::Integer(2);
  for (int i = 0; i < kReciprocalSteps; ++i) {
    x = x * (two - (b * x));
  }
  return x;
}

Real AtanInverse(uint32_t n) {
  Real power = One().DividedBy(n);
  Real sum = power;
  for (uint32_t j = 1;; ++j) {
    power = power.DividedBy(n).DividedBy(n);
    const Real term = power.DividedBy((2 * j) + 1);
    if (Negligible(term, sum)) {
      return sum;
    }
    sum = (j % 2 == 1) ? sum - term : sum + term;
  }
}

Real AtanhSeries(const Real& t) {
  const Real square = t * t;
  Real power = t;
  Real sum = t;
  for (uint32_t j = 1;; ++j) {
    power = power * square;
    const Real term = power.DividedBy((2 * j) + 1);
    if (Negligible(term, sum)) {
      return sum.Scaled(1);
    }
    sum = sum + term;
  }
}

Real SinSeries(const Real& r) {
  if (r.IsZero()) {
    return r;
  }
  const Real square = r * r;
  Real term = r;
  Real sum = r;
  for (uint32_t n = 1;; ++n) {
    term = (term * square).DividedBy((2 * n) * ((2 * n) + 1)).Negated();
    if (Negligible(term, sum)) {
      return sum;
    }
    sum = sum + term;
  }
}

Real CosSeries(const Real& r) {
  const Real square = r * r;
  Real term = One();
  Real sum = One();
  for (uint32_t n = 1;; ++n) {
    term = (term * square).DividedBy(((2 * n) - 1) * (2 * n)).Negated();
    if (Negligible(term, sum)) {
      return sum;
    }
    sum = sum + term;
  }
}

struct Reduced {
  int64_t quadrant = 0;
  Real remainder;
};

Reduced ReduceQuarterTurns(const Real& x) {
  const Real half_pi = Pi().Scaled(-1);
  const int64_t turns = (x / half_pi).Nearest();
  return {.quadrant = ((turns % 4) + 4) % 4,
          .remainder = x - (half_pi * Real::Integer(turns))};
}

}  // namespace

Real Real::Make(bool negative, std::vector<uint32_t> magnitude,
                int64_t exponent, bool exact) {
  Trim(magnitude);
  Real out;
  out.exact_ = exact;
  if (magnitude.empty()) {
    return out;
  }
  const int64_t length = BitLength(magnitude);
  bool lost = false;
  if (length > kBits) {
    magnitude = ShiftRight(magnitude, length - kBits, lost);
    exponent += length - kBits;
  } else if (length < kBits) {
    magnitude = ShiftLeft(magnitude, kBits - length);
    exponent -= kBits - length;
  }
  out.negative_ = negative;
  out.magnitude_ = std::move(magnitude);
  out.exponent_ = exponent;
  out.exact_ = exact && !lost;
  return out;
}

Real Real::Integer(int64_t value) {
  const bool negative = value < 0;
  const uint64_t magnitude = negative ? (~static_cast<uint64_t>(value)) + 1
                                      : static_cast<uint64_t>(value);
  return Make(negative,
              {static_cast<uint32_t>(magnitude),
               static_cast<uint32_t>(magnitude >> kLimbBits)},
              0, true);
}

Real Real::Ratio(int64_t numerator, int64_t denominator) {
  Require(denominator != 0, "ratio with a zero denominator");
  return Integer(numerator) / Integer(denominator);
}

Real Real::Binary32(uint32_t bits) {
  const uint32_t field = (bits >> 23) & 0xFFU;
  Require(field != 0xFFU, "binary32 infinity or NaN");
  const uint32_t fraction = bits & 0x7FFFFFU;
  const uint32_t significand = field == 0 ? fraction : (fraction | 0x800000U);
  const int64_t exponent =
      field == 0 ? -149 : static_cast<int64_t>(field) - 150;
  return Make((bits >> 31) != 0, {significand}, exponent, true);
}

Real Real::Binary64(uint64_t bits) {
  const uint64_t field = (bits >> 52) & 0x7FFU;
  Require(field != 0x7FFU, "binary64 infinity or NaN");
  const uint64_t fraction = bits & ((uint64_t{1} << 52) - 1);
  const uint64_t significand =
      field == 0 ? fraction : (fraction | (uint64_t{1} << 52));
  const int64_t exponent =
      field == 0 ? -1074 : static_cast<int64_t>(field) - 1075;
  return Make((bits >> 63) != 0,
              {static_cast<uint32_t>(significand),
               static_cast<uint32_t>(significand >> kLimbBits)},
              exponent, true);
}

int64_t Real::TopBit() const {
  Require(!IsZero(), "the top bit of zero");
  return exponent_ + kBits - 1;
}

Real Real::Negated() const {
  Real out = *this;
  out.negative_ = !IsZero() && !negative_;
  return out;
}

Real Real::Inexact() const {
  Real out = *this;
  out.exact_ = false;
  return out;
}

Real Real::Scaled(int64_t power_of_two) const {
  Real out = *this;
  if (!IsZero()) {
    out.exponent_ += power_of_two;
  }
  return out;
}

Real Real::DividedBy(uint32_t divisor) const {
  Require(divisor != 0, "division by zero");
  if (IsZero()) {
    return *this;
  }
  const Limbs shifted = ShiftLeft(magnitude_, 2 * kLimbBits);
  Limbs quotient(shifted.size(), 0);
  uint64_t remainder = 0;
  for (size_t i = shifted.size(); i > 0; --i) {
    const uint64_t current = (remainder << kLimbBits) | shifted[i - 1];
    quotient[i - 1] = static_cast<uint32_t>(current / divisor);
    remainder = current % divisor;
  }
  return Make(negative_, std::move(quotient), exponent_ - (2 * kLimbBits),
              exact_ && remainder == 0);
}

int64_t Real::Floor() const {
  if (IsZero()) {
    return 0;
  }
  Require(TopBit() < 62, "value too large for an integer");
  bool lost = false;
  const Limbs integer = exponent_ >= 0
                            ? ShiftLeft(magnitude_, exponent_)
                            : ShiftRight(magnitude_, -exponent_, lost);
  int64_t value = 0;
  for (size_t i = integer.size(); i > 0; --i) {
    value = (value << kLimbBits) | integer[i - 1];
  }
  Require(value < kIntegerLimit, "value too large for an integer");
  if (negative_) {
    return -value - (lost ? 1 : 0);
  }
  return value;
}

int64_t Real::Nearest() const { return (*this + Ratio(1, 2)).Floor(); }

Real::Rounded Real::Round(int significand_bits, int64_t min_exponent,
                          int64_t max_exponent) const {
  int64_t top = TopBit();
  Require(
      top >= min_exponent && top <= max_exponent,
      std::format("value with exponent {} is outside the normal range", top));
  const int64_t shift = kBits - significand_bits;
  bool lost = false;
  const Limbs kept = ShiftRight(magnitude_, shift, lost);
  uint64_t significand = 0;
  for (size_t i = kept.size(); i > 0; --i) {
    significand = (significand << kLimbBits) | kept[i - 1];
  }
  const Limbs remainder = LowBits(magnitude_, shift);
  const Limbs half = PowerOfTwo(shift - 1);
  const int order = Compare(remainder, half);
  if (!exact_) {
    const Limbs distance =
        order >= 0 ? Subtract(remainder, half) : Subtract(half, remainder);
    Require(BitLength(distance) > shift - 1 - kMargin,
            "cannot round: the value is too close to a rounding midpoint");
  }
  if (order > 0 || (order == 0 && (significand & 1U) != 0)) {
    ++significand;
    if (significand == (uint64_t{1} << significand_bits)) {
      significand >>= 1;
      ++top;
    }
  }
  Require(top <= max_exponent, "rounding overflowed the format");
  return {.exponent = top, .significand = significand};
}

uint32_t Real::ToBinary32() const {
  if (IsZero()) {
    return 0;
  }
  const Rounded rounded = Round(24, -126, 127);
  return (negative_ ? 0x80000000U : 0U) |
         (static_cast<uint32_t>(rounded.exponent + 127) << 23) |
         (static_cast<uint32_t>(rounded.significand) & 0x7FFFFFU);
}

uint64_t Real::ToBinary64() const {
  if (IsZero()) {
    return 0;
  }
  const Rounded rounded = Round(53, -1022, 1023);
  return (negative_ ? uint64_t{1} << 63 : 0U) |
         (static_cast<uint64_t>(rounded.exponent + 1023) << 52) |
         (rounded.significand & ((uint64_t{1} << 52) - 1));
}

Real operator+(const Real& a, const Real& b) {
  if (a.IsZero()) {
    Real out = b;
    out.exact_ = a.exact_ && b.exact_;
    return out;
  }
  if (b.IsZero()) {
    Real out = a;
    out.exact_ = a.exact_ && b.exact_;
    return out;
  }
  const bool a_larger = a.exponent_ >= b.exponent_;
  const Real& large = a_larger ? a : b;
  const Real& small = a_larger ? b : a;
  if (large.exponent_ - small.exponent_ > Real::kBits + (2 * kLimbBits)) {
    Real out = large;
    out.exact_ = false;
    return out;
  }
  const Limbs aligned =
      ShiftLeft(large.magnitude_, large.exponent_ - small.exponent_);
  const bool exact = a.exact_ && b.exact_;
  if (large.negative_ == small.negative_) {
    return Real::Make(large.negative_, Add(aligned, small.magnitude_),
                      small.exponent_, exact);
  }
  const int order = Compare(aligned, small.magnitude_);
  if (order == 0) {
    Real out;
    out.exact_ = exact;
    return out;
  }
  return order > 0
             ? Real::Make(large.negative_, Subtract(aligned, small.magnitude_),
                          small.exponent_, exact)
             : Real::Make(small.negative_, Subtract(small.magnitude_, aligned),
                          small.exponent_, exact);
}

Real operator-(const Real& a, const Real& b) { return a + b.Negated(); }

Real operator*(const Real& a, const Real& b) {
  return Real::Make(a.negative_ != b.negative_,
                    Multiply(a.magnitude_, b.magnitude_),
                    a.exponent_ + b.exponent_, a.exact_ && b.exact_);
}

Real operator/(const Real& a, const Real& b) {
  Require(!b.IsZero(), "division by zero");
  if (a.IsZero()) {
    return a;
  }
  if (Compare(b.magnitude_, PowerOfTwo(Real::kBits - 1)) == 0) {
    return Real::Make(a.negative_ != b.negative_, a.magnitude_,
                      a.exponent_ - b.TopBit(), a.exact_ && b.exact_);
  }
  return (a * Reciprocal(b)).Inexact();
}

Real Pi() {
  static const Real kPi = AtanInverse(5).Scaled(4) - AtanInverse(239).Scaled(2);
  return kPi;
}

Real Ln2() {
  static const Real kLn2 = AtanhSeries(One().DividedBy(3));
  return kLn2;
}

Real Sqrt(const Real& x) {
  Require(!x.Negative(), "square root of a negative value");
  if (x.IsZero()) {
    return x;
  }
  Real y = One().Scaled(x.TopBit() / 2);
  for (int i = 0; i < kSqrtSteps; ++i) {
    y = (y + (x / y)).Scaled(-1);
  }
  return y.Inexact();
}

Real Exp(const Real& x) {
  if (x.IsZero()) {
    return One();
  }
  const int64_t turns = (x / Ln2()).Nearest();
  const Real reduced =
      (x - (Ln2() * Real::Integer(turns))).Scaled(-kExpReduction);
  Real term = One();
  Real sum = One();
  for (uint32_t i = 1;; ++i) {
    term = (term * reduced).DividedBy(i);
    if (Negligible(term, sum)) {
      break;
    }
    sum = sum + term;
  }
  for (int64_t i = 0; i < kExpReduction; ++i) {
    sum = sum * sum;
  }
  return sum.Scaled(turns).Inexact();
}

Real Log(const Real& x) {
  Require(!x.IsZero() && !x.Negative(), "logarithm of a non-positive value");
  const int64_t exponent = x.TopBit();
  const Real mantissa = x.Scaled(-exponent);
  const Real t = (mantissa - One()) / (mantissa + One());
  return (AtanhSeries(t) + (Ln2() * Real::Integer(exponent))).Inexact();
}

Real Pow(const Real& x, const Real& y) {
  if (y.IsZero()) {
    return One();
  }
  return Exp(y * Log(x));
}

Real Sin(const Real& x) {
  if (x.IsZero()) {
    return x;
  }
  const Reduced reduced = ReduceQuarterTurns(x);
  switch (reduced.quadrant) {
    case 0:
      return SinSeries(reduced.remainder).Inexact();
    case 1:
      return CosSeries(reduced.remainder).Inexact();
    case 2:
      return SinSeries(reduced.remainder).Negated().Inexact();
    default:
      return CosSeries(reduced.remainder).Negated().Inexact();
  }
}

Real Cos(const Real& x) {
  if (x.IsZero()) {
    return One();
  }
  const Reduced reduced = ReduceQuarterTurns(x);
  switch (reduced.quadrant) {
    case 0:
      return CosSeries(reduced.remainder).Inexact();
    case 1:
      return SinSeries(reduced.remainder).Negated().Inexact();
    case 2:
      return CosSeries(reduced.remainder).Negated().Inexact();
    default:
      return SinSeries(reduced.remainder).Inexact();
  }
}

Real Tanh(const Real& x) {
  if (x.IsZero()) {
    return x;
  }
  const Real magnitude = x.Negative() ? x.Negated() : x;
  const Real e = Exp(magnitude.Scaled(1));
  const Real value = (e - One()) / (e + One());
  return x.Negative() ? value.Negated() : value;
}

}  // namespace hotarubi::exact
