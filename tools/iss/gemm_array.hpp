#ifndef HOTARUBI_TOOLS_ISS_GEMM_ARRAY_HPP_
#define HOTARUBI_TOOLS_ISS_GEMM_ARRAY_HPP_

#include <cstdint>

#include "memories.hpp"

namespace hotarubi::iss {

enum class WeightMode : uint8_t {
  kFp4 = 0,
  kW8Hi = 1,
  kW8Lo = 2,
};

struct GemmCommand {
  uint32_t a_base = 0;
  uint32_t a_stride = 0;
  uint32_t kb = 0;
  uint32_t m = 0;
  uint32_t w_base = 0;
  uint32_t w_stride = 0;
  uint32_t n8 = 0;
  uint32_t c_base = 0;
  uint32_t wmode = 0;
  bool acc = false;
};

WeightMode DecodeWeightMode(uint32_t wmode);
int32_t WeightValue(uint32_t nibble, WeightMode mode);
uint32_t BlockValue(int32_t sum, uint32_t a_scale, uint32_t w_scale,
                    WeightMode mode);
void ExecuteGemm(const GemmCommand& command, Memories& memories);

}  // namespace hotarubi::iss

#endif  // HOTARUBI_TOOLS_ISS_GEMM_ARRAY_HPP_
