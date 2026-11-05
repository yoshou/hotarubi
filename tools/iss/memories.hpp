#ifndef HOTARUBI_TOOLS_ISS_MEMORIES_HPP_
#define HOTARUBI_TOOLS_ISS_MEMORIES_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "word.hpp"

namespace hotarubi::iss {

inline constexpr uint32_t kTokens = 4;
inline constexpr uint32_t kLanes = 8;
inline constexpr uint32_t kBlock = 32;
inline constexpr uint32_t kFbufElements = kTokens * kLanes;
inline constexpr uint32_t kAbufWordBytes = 132;
inline constexpr uint32_t kWbufWordBytes = 144;
inline constexpr uint32_t kAbufScaleOffset = 128;
inline constexpr uint32_t kWbufScaleOffset = 128;

inline constexpr uint32_t kAbufDepth = 8192;
inline constexpr uint32_t kWbufDepth = 2048;
inline constexpr uint32_t kFbufDepth = 4096;
inline constexpr uint32_t kPbufDepth = 512;
inline constexpr uint32_t kTbufDepth = 1024;
inline constexpr uint32_t kCbufDepth = 1024;

constexpr int32_t SignedByte(uint32_t byte) {
  return byte >= 128 ? static_cast<int32_t>(byte) - 256
                     : static_cast<int32_t>(byte);
}

using AbufWord = std::array<uint8_t, kAbufWordBytes>;
using WbufWord = std::array<uint8_t, kWbufWordBytes>;
using FbufWord = std::array<uint32_t, kFbufElements>;

enum class Mem : uint8_t {
  kAbuf = 0,
  kWbuf = 1,
  kFbuf = 2,
  kPbuf = 3,
  kTbuf = 4,
  kCbuf = 5,
};

struct Memories {
  Memories();

  std::vector<AbufWord> abuf;
  std::vector<WbufWord> wbuf;
  std::vector<FbufWord> fbuf;
  std::vector<FbufWord> pbuf;
  std::vector<uint64_t> tbuf;
  std::vector<Word128> cbuf;
};

}  // namespace hotarubi::iss

#endif  // HOTARUBI_TOOLS_ISS_MEMORIES_HPP_
