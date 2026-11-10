#ifndef HOTARUBI_TOOLS_ISS_SCALAR_HPP_
#define HOTARUBI_TOOLS_ISS_SCALAR_HPP_

#include <array>
#include <cstdint>
#include <span>

#include "lane.hpp"
#include "memories.hpp"

namespace hotarubi::iss {

struct Reductions {
  std::array<std::array<uint32_t, kLanes>, kTokens> sum{};
  std::array<uint32_t, kTokens> max{};
};

Reductions ClearedReductions();
uint32_t SumTree(std::span<const uint32_t, kLanes> values);
uint32_t MaxTree(std::span<const uint32_t, kLanes> values);
uint32_t Fp32Negate(uint32_t bits);
uint32_t Reciprocal(uint32_t bits);
uint32_t ReciprocalSqrt(uint32_t bits);
uint32_t CountReciprocal(uint32_t count);
void ApplyRowScalar(const Pass& pass, const Reductions& reductions,
                    uint32_t epsilon, uint32_t rn, RowState& rows);

}  // namespace hotarubi::iss

#endif  // HOTARUBI_TOOLS_ISS_SCALAR_HPP_
