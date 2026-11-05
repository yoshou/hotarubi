#ifndef HOTARUBI_TOOLS_ISS_REFERENCE_HPP_
#define HOTARUBI_TOOLS_ISS_REFERENCE_HPP_

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "hotarubi/exact.hpp"

namespace hotarubi::iss::reference {

exact::Real Value(uint32_t bits);
uint32_t Round(const exact::Real& value);
uint32_t Add(uint32_t a, uint32_t b);
uint32_t Mul(uint32_t a, uint32_t b);
uint32_t Negate(uint32_t bits);
uint32_t MulFraction(uint32_t s, uint32_t fraction);
uint32_t FromInt(int64_t value);

uint32_t Reciprocal(uint32_t a);
uint32_t ReciprocalSqrt(uint32_t a);
uint32_t RowSum(std::span<const std::array<uint32_t, 8>> groups);
uint32_t RowMax(std::span<const uint32_t> values);

uint32_t Interpolate(uint64_t entry, uint32_t fraction);
uint32_t Sigma(uint32_t x, std::span<const uint64_t> table);
uint32_t Pow2(uint32_t t, std::span<const uint64_t> table);
uint32_t Fraction01(uint32_t t, std::span<const uint64_t> table);

struct Mxint8Block {
  std::array<uint8_t, 32> q{};
  uint8_t scale = 0;
};

int64_t NearestEven(const exact::Real& value);
Mxint8Block EncodeMxint8(std::span<const uint32_t, 32> values);

}  // namespace hotarubi::iss::reference

#endif  // HOTARUBI_TOOLS_ISS_REFERENCE_HPP_
