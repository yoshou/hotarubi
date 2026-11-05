#ifndef HOTARUBI_TOOLS_ISS_ROM_HPP_
#define HOTARUBI_TOOLS_ISS_ROM_HPP_

#include <array>
#include <cstdint>

#include "word.hpp"

namespace hotarubi::iss {

inline constexpr uint32_t kRomWords = 256;
inline constexpr uint32_t kFixedOps = 16;

extern const std::array<uint32_t, kFixedOps> kRomStart;
extern const std::array<Word128, kRomWords> kRom;

}  // namespace hotarubi::iss

#endif  // HOTARUBI_TOOLS_ISS_ROM_HPP_
