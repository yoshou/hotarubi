#include "gemm_ref.hpp"

#include <array>
#include <cstdint>
#include <vector>

#include "fp32_ref.hpp"

namespace hotarubi::tb {

namespace {

int ScaleBias(uint32_t wmode) {
  if (wmode == kWmodeFp4) {
    return -7;
  }
  return wmode == kWmodeHigh ? -8 : -12;
}

}  // namespace

int WeightValue(uint32_t wmode, uint8_t nibble) {
  if (wmode == kWmodeFp4) {
    constexpr std::array<int, 8> kMagnitude = {0, 1, 2, 3, 4, 6, 8, 12};
    const int magnitude = kMagnitude.at(nibble & 0x7U);
    return (nibble & 0x8U) != 0 ? -magnitude : magnitude;
  }
  if (wmode == kWmodeHigh) {
    return (nibble & 0x8U) != 0 ? static_cast<int>(nibble) - 16 : nibble;
  }
  return nibble;
}

uint32_t BlockValue(const AbufWord& a, uint32_t token, const WbufWord& w,
                    uint32_t column, uint32_t wmode) {
  int64_t sum = 0;
  for (uint32_t k = 0; k < 32; ++k) {
    sum += static_cast<int64_t>(a.q.at((token * 32) + k)) *
           WeightValue(wmode, w.nibble.at((column * 32) + k));
  }
  const int exponent = (static_cast<int>(a.scale.at(token)) - 127) +
                       (static_cast<int>(w.scale.at(column)) - 127) +
                       ScaleBias(wmode);
  return Fp32FromInt(sum, exponent);
}

void ReferenceGemm(const GemmArgs& args, const std::vector<AbufWord>& abuf,
                   const std::vector<WbufWord>& wbuf,
                   std::vector<FbufWord>& fbuf) {
  for (uint32_t q = 0; q < args.m / 4; ++q) {
    for (uint32_t g = 0; g < args.n8; ++g) {
      FbufWord& out = fbuf.at(args.c_base + (q * args.n8) + g);
      for (uint32_t element = 0; element < 32; ++element) {
        const uint32_t token = element / 8;
        const uint32_t column = element % 8;
        uint32_t acc = args.acc ? out.at(element) : kFp32Zero;
        for (uint32_t b = 0; b < args.kb; ++b) {
          acc = Fp32Add(
              acc,
              BlockValue(abuf.at(args.a_base + (q * args.a_stride) + b), token,
                         wbuf.at(args.w_base + (g * args.w_stride) + b), column,
                         args.wmode));
        }
        out.at(element) = acc;
      }
    }
  }
}

}  // namespace hotarubi::tb
