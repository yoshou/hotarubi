#include "vec_ref.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <span>

#include "fp32_ref.hpp"

namespace hotarubi::tb {

namespace {

constexpr uint32_t kNegativeInfinity = 0xFF800000U;
constexpr uint32_t kPositiveInfinity = 0x7F800000U;

uint32_t Field(uint32_t value) { return (value >> 23) & 0xFFU; }

uint64_t Mantissa(uint32_t value) {
  return Field(value) == 0 ? 0 : ((value & 0x7FFFFFU) | 0x800000U);
}

double ActValue(ActFunction function, double x) {
  if (function == ActFunction::kSigmoid) {
    return 1.0 / (1.0 + std::exp(-x));
  }
  const double inner =
      std::sqrt(2.0 / std::numbers::pi) * (x + (0.044715 * x * x * x));
  return (1.0 + std::tanh(inner)) / 2.0;
}

uint32_t Tree(std::span<const uint32_t, 8> l) {
  return Fp32Add(Fp32Add(Fp32Add(l[0], l[1]), Fp32Add(l[2], l[3])),
                 Fp32Add(Fp32Add(l[4], l[5]), Fp32Add(l[6], l[7])));
}

}  // namespace

Tables MakeTables(ActFunction function) {
  Tables tables;
  for (uint32_t i = 0; i < 512; ++i) {
    tables.v.at(i) = Fp32FromDouble(std::exp2(static_cast<double>(i) / 512.0));
    tables.v.at(512 + i) =
        Fp32FromDouble(ActValue(function, static_cast<double>(i) / 32.0));
  }
  for (uint32_t i = 0; i < 512; ++i) {
    const uint32_t next_exp = i == 511 ? kFp32Two : tables.v.at(i + 1);
    const uint32_t next_act = i == 511
                                  ? Fp32FromDouble(ActValue(function, 16.0))
                                  : tables.v.at(512 + i + 1);
    tables.s.at(i) = Fp32Sub(next_exp, tables.v.at(i));
    tables.s.at(512 + i) = Fp32Sub(next_act, tables.v.at(512 + i));
  }
  return tables;
}

uint32_t Interpolate(const Tables& tables, uint32_t index, uint32_t frac) {
  const uint32_t slope = tables.s.at(index);
  uint32_t part = kFp32Zero;
  if (Field(slope) != 0 && frac != 0) {
    part = Fp32Round((slope & kFp32SignBit) != 0, Mantissa(slope) * frac,
                     static_cast<int>(Field(slope)) - 166);
  }
  return Fp32Add(tables.v.at(index), part);
}

uint32_t TableAct(const Tables& tables, uint32_t x) {
  const uint32_t field = Field(x);
  if (field >= 131) {
    return kFp32One;
  }
  const uint64_t mantissa = Mantissa(x);
  const int shift = static_cast<int>(field) - 129;
  uint64_t fixed = 0;
  if (shift >= 0) {
    fixed = mantissa << shift;
  } else if (-shift < 64) {
    fixed = mantissa >> -shift;
  }
  return Interpolate(tables, 512 + static_cast<uint32_t>(fixed >> 16),
                     static_cast<uint32_t>(fixed & 0xFFFFU));
}

uint32_t TableExp2(const Tables& tables, uint32_t t) {
  const uint32_t field = Field(t);
  const bool negative = (t & kFp32SignBit) != 0;
  if (field >= 135) {
    return negative ? kFp32Zero : kPositiveInfinity;
  }
  const uint64_t mantissa = Mantissa(t);
  const int shift = static_cast<int>(field) - 125;
  uint64_t magnitude = 0;
  bool inexact = false;
  if (shift >= 0) {
    magnitude = mantissa << shift;
  } else if (-shift < 64) {
    magnitude = mantissa >> -shift;
    inexact = (mantissa & ((uint64_t{1} << -shift) - 1)) != 0;
  } else {
    inexact = mantissa != 0;
  }
  const int64_t fixed =
      negative ? -static_cast<int64_t>(magnitude + (inexact ? 1 : 0))
               : static_cast<int64_t>(magnitude);
  const int64_t n =
      fixed >= 0 ? fixed / (int64_t{1} << 25)
                 : -((-fixed + (int64_t{1} << 25) - 1) / (int64_t{1} << 25));
  const auto f = static_cast<uint64_t>(fixed - (n * (int64_t{1} << 25)));
  const uint32_t base = Interpolate(tables, static_cast<uint32_t>(f >> 16),
                                    static_cast<uint32_t>(f & 0xFFFFU));
  if (Field(base) == 0) {
    return kFp32Zero;
  }
  const int64_t exponent = static_cast<int64_t>(Field(base)) + n;
  if (exponent <= 0) {
    return kFp32Zero;
  }
  if (exponent >= 255) {
    return kPositiveInfinity;
  }
  return (static_cast<uint32_t>(exponent) << 23) | (base & 0x7FFFFFU);
}

uint32_t RowSum(std::span<const uint32_t> values) {
  std::array<uint32_t, 8> acc{};
  for (size_t g = 0; g < values.size() / 8; ++g) {
    uint32_t& slot = acc.at(g % 8);
    slot = Fp32Add(slot, Tree(values.subspan(8 * g).first<8>()));
  }
  return Tree(acc);
}

uint32_t RowMax(std::span<const uint32_t> values) {
  uint32_t max = kNegativeInfinity;
  for (const uint32_t value : values) {
    max = Fp32Max(max, value);
  }
  return max;
}

}  // namespace hotarubi::tb
