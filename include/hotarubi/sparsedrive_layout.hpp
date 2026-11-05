#ifndef HOTARUBI_SPARSEDRIVE_LAYOUT_HPP_
#define HOTARUBI_SPARSEDRIVE_LAYOUT_HPP_

#include <cstdint>

namespace hotarubi::sparsedrive {

inline constexpr uint32_t kCameras = 6;
inline constexpr uint32_t kImageRows = 256;
inline constexpr uint32_t kImageColumns = 704;
inline constexpr uint32_t kImageChannels = 3;
inline constexpr uint32_t kStemPhases = 2;
inline constexpr uint32_t kStemPadding = 3;
inline constexpr uint32_t kStemRowFloats = 720;
inline constexpr uint64_t kStemImageBytes =
    uint64_t{kStemPhases} * kImageChannels * kImageRows * kStemRowFloats * 4;
inline constexpr uint32_t kKpWidthLane = 12;
inline constexpr uint32_t kKpHeightLane = 13;
inline constexpr uint32_t kKpProjectionLane = 16;
inline constexpr uint32_t kMotionDoneSignal = 1;

}  // namespace hotarubi::sparsedrive

#endif  // HOTARUBI_SPARSEDRIVE_LAYOUT_HPP_
