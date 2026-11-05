#ifndef HOTARUBI_TOOLS_ISS_FP32_HPP_
#define HOTARUBI_TOOLS_ISS_FP32_HPP_

#include <bit>
#include <cstdint>

namespace hotarubi::iss {

inline constexpr uint32_t kFp32One = 0x3F800000;
inline constexpr uint32_t kFp32Two = 0x40000000;
inline constexpr uint32_t kFp32Half = 0x3F000000;
inline constexpr uint32_t kFp32OnePointFive = 0x3FC00000;
inline constexpr uint32_t kFp32Log2e = 0x3FB8AA3B;
inline constexpr uint32_t kFp32Infinity = 0x7F800000;
inline constexpr uint32_t kFp32NegativeInfinity = 0xFF800000;
inline constexpr uint32_t kFp32SignBit = 0x80000000;

constexpr uint32_t Fp32Exponent(uint32_t bits) { return (bits >> 23) & 0xFFU; }

constexpr uint32_t Fp32Significand(uint32_t bits) {
  return Fp32Exponent(bits) == 0 ? 0 : ((bits & 0x7FFFFFU) | 0x800000U);
}

constexpr uint32_t Fp32Pack(uint32_t sign, int32_t exponent,
                            uint32_t fraction) {
  if (exponent <= 0) {
    return 0;
  }
  if (exponent >= 255) {
    return (sign << 31) | kFp32Infinity;
  }
  return (sign << 31) | (static_cast<uint32_t>(exponent) << 23) | fraction;
}

constexpr uint32_t Fp32Round(uint32_t sign, int32_t exponent,
                             uint32_t significand, bool guard, bool sticky) {
  const bool round_up = guard && (sticky || (significand & 1U) != 0);
  uint32_t rounded = significand + (round_up ? 1U : 0U);
  if ((rounded >> 24) != 0) {
    ++exponent;
    rounded >>= 1;
  }
  return Fp32Pack(sign, exponent, rounded & 0x7FFFFFU);
}

constexpr uint32_t Fp32Add(uint32_t a, uint32_t b) {
  const uint32_t ea = Fp32Exponent(a);
  const uint32_t eb = Fp32Exponent(b);
  const uint32_t ma = Fp32Significand(a);
  const uint32_t mb = Fp32Significand(b);
  const bool a_big = ((ea << 24) | ma) >= ((eb << 24) | mb);
  const uint32_t e_big = a_big ? ea : eb;
  const uint32_t distance = a_big ? ea - eb : eb - ea;
  const uint32_t m_big = a_big ? ma : mb;
  const uint32_t m_small = a_big ? mb : ma;
  uint32_t shifted = 0;
  if (m_small != 0 && distance > 26) {
    shifted = 1;
  } else if (m_small != 0) {
    const uint32_t extended = m_small << 3;
    const uint32_t lost = extended & ((1U << distance) - 1U);
    shifted = (extended >> distance) | (lost != 0 ? 1U : 0U);
  }
  const bool subtract = ((a ^ b) >> 31) != 0;
  const uint32_t sum =
      subtract ? (m_big << 3) - shifted : (m_big << 3) + shifted;
  if (sum == 0) {
    return 0;
  }
  const int32_t lead_zeros = 28 - std::bit_width(sum);
  const uint32_t norm = sum << lead_zeros;
  return Fp32Round((a_big ? a : b) >> 31,
                   static_cast<int32_t>(e_big) + 1 - lead_zeros, norm >> 4,
                   ((norm >> 3) & 1U) != 0, (norm & 7U) != 0);
}

constexpr uint32_t Fp32Mul(uint32_t a, uint32_t b) {
  if (Fp32Exponent(a) == 0 || Fp32Exponent(b) == 0) {
    return 0;
  }
  const uint64_t product =
      uint64_t{Fp32Significand(a)} * uint64_t{Fp32Significand(b)};
  const bool carry = (product >> 47) != 0;
  const int32_t shift = carry ? 23 : 22;
  const uint64_t kept = product >> shift;
  const bool sticky = (product & ((uint64_t{1} << shift) - 1)) != 0;
  const int32_t exponent =
      static_cast<int32_t>(Fp32Exponent(a) + Fp32Exponent(b)) -
      (carry ? 126 : 127);
  return Fp32Round((a ^ b) >> 31, exponent, static_cast<uint32_t>(kept >> 1),
                   (kept & 1U) != 0, sticky);
}

constexpr uint32_t Fp32MulFraction(uint32_t s, uint32_t fraction) {
  if (Fp32Exponent(s) == 0 || fraction == 0) {
    return 0;
  }
  const uint64_t product = uint64_t{Fp32Significand(s)} * fraction;
  const int32_t lead_zeros = 40 - std::bit_width(product);
  const uint64_t norm = product << lead_zeros;
  return Fp32Round(s >> 31, static_cast<int32_t>(Fp32Exponent(s)) - lead_zeros,
                   static_cast<uint32_t>(norm >> 16), ((norm >> 15) & 1U) != 0,
                   (norm & 0x7FFFU) != 0);
}

}  // namespace hotarubi::iss

#endif  // HOTARUBI_TOOLS_ISS_FP32_HPP_
