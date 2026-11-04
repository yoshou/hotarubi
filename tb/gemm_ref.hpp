#ifndef HOTARUBI_TB_GEMM_REF_HPP_
#define HOTARUBI_TB_GEMM_REF_HPP_

#include <cstdint>
#include <vector>

#include "npu_data.hpp"
#include "npu_isa.hpp"

namespace hotarubi::tb {

inline constexpr uint32_t kWmodeFp4 = 0;
inline constexpr uint32_t kWmodeHigh = 1;
inline constexpr uint32_t kWmodeLow = 2;

int WeightValue(uint32_t wmode, uint8_t nibble);
uint32_t BlockValue(const AbufWord& a, uint32_t token, const WbufWord& w,
                    uint32_t column, uint32_t wmode);
void ReferenceGemm(const GemmArgs& args, const std::vector<AbufWord>& abuf,
                   const std::vector<WbufWord>& wbuf,
                   std::vector<FbufWord>& fbuf);

}  // namespace hotarubi::tb

#endif  // HOTARUBI_TB_GEMM_REF_HPP_
