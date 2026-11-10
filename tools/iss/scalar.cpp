#include "scalar.hpp"

#include <bit>
#include <cstdint>
#include <span>

#include "fp32.hpp"
#include "lane.hpp"
#include "memories.hpp"

namespace hotarubi::iss {
namespace {

constexpr uint32_t kReciprocalMagic = 0x7EF311C7;
constexpr uint32_t kReciprocalSqrtMagic = 0x5F375A86;
constexpr int kNewtonSteps = 4;

}  // namespace

Reductions ClearedReductions() {
  Reductions reductions;
  reductions.max.fill(kFp32NegativeInfinity);
  return reductions;
}

uint32_t SumTree(std::span<const uint32_t, kLanes> values) {
  return Fp32Add(
      Fp32Add(Fp32Add(values[0], values[1]), Fp32Add(values[2], values[3])),
      Fp32Add(Fp32Add(values[4], values[5]), Fp32Add(values[6], values[7])));
}

uint32_t MaxTree(std::span<const uint32_t, kLanes> values) {
  return Fp32Max(
      Fp32Max(Fp32Max(values[0], values[1]), Fp32Max(values[2], values[3])),
      Fp32Max(Fp32Max(values[4], values[5]), Fp32Max(values[6], values[7])));
}

uint32_t Fp32Negate(uint32_t bits) {
  return (bits & ~kFp32SignBit) == 0 ? 0 : bits ^ kFp32SignBit;
}

uint32_t Reciprocal(uint32_t bits) {
  uint32_t y = kReciprocalMagic - bits;
  for (int i = 0; i < kNewtonSteps; ++i) {
    const uint32_t error = Fp32Add(kFp32Two, Fp32Negate(Fp32Mul(bits, y)));
    y = Fp32Mul(y, error);
  }
  return y;
}

uint32_t ReciprocalSqrt(uint32_t bits) {
  uint32_t y = kReciprocalSqrtMagic - (bits >> 1);
  for (int i = 0; i < kNewtonSteps; ++i) {
    const uint32_t square = Fp32Mul(Fp32Mul(bits, y), y);
    const uint32_t error =
        Fp32Add(kFp32OnePointFive, Fp32Negate(Fp32Mul(square, kFp32Half)));
    y = Fp32Mul(y, error);
  }
  return y;
}

uint32_t CountReciprocal(uint32_t count) {
  const uint32_t n = count & 0x3FFFU;
  const int top = n == 0 ? 0 : std::bit_width(n) - 1;
  const uint32_t fraction = (n << (23 - top)) & 0x7FFFFFU;
  return Reciprocal((static_cast<uint32_t>(127 + top) << 23) | fraction);
}

void ApplyRowScalar(RowScalar program, const Reductions& reductions,
                    uint32_t epsilon, uint32_t rn, RowState& rows) {
  for (uint32_t t = 0; t < kTokens; ++t) {
    const std::array<uint32_t, kLanes>& sums = reductions.sum.at(t);
    switch (program) {
      case RowScalar::kMean:
        rows.rs0.at(t) = Fp32Negate(Fp32Mul(SumTree(sums), rn));
        break;
      case RowScalar::kRstd:
        rows.rs1.at(t) =
            ReciprocalSqrt(Fp32Add(Fp32Mul(SumTree(sums), rn), epsilon));
        break;
      case RowScalar::kNegMax:
        rows.rs0.at(t) = Fp32Negate(reductions.max.at(t));
        break;
      case RowScalar::kRecipSum:
        rows.rs1.at(t) = Reciprocal(SumTree(sums));
        break;
      case RowScalar::kMax:
        rows.rs1.at(t) = reductions.max.at(t);
        break;
      case RowScalar::kNone:
        break;
    }
  }
}

}  // namespace hotarubi::iss
