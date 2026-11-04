#ifndef HOTARUBI_TB_VEC_REF_HPP_
#define HOTARUBI_TB_VEC_REF_HPP_

#include <array>
#include <cstdint>
#include <span>

namespace hotarubi::tb {

struct Tables {
  std::array<uint32_t, 1024> v{};
  std::array<uint32_t, 1024> s{};
};

enum class ActFunction : uint8_t {
  kSigmoid,
  kGeluPhi,
};

Tables MakeTables(ActFunction function);
uint32_t Interpolate(const Tables& tables, uint32_t index, uint32_t frac);
uint32_t TableAct(const Tables& tables, uint32_t x);
uint32_t TableExp2(const Tables& tables, uint32_t t);
uint32_t RowSum(std::span<const uint32_t> values);
uint32_t RowMax(std::span<const uint32_t> values);

}  // namespace hotarubi::tb

#endif  // HOTARUBI_TB_VEC_REF_HPP_
