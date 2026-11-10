#ifndef HOTARUBI_TOOLS_ISS_ROM_HPP_
#define HOTARUBI_TOOLS_ISS_ROM_HPP_

#include <array>
#include <cstdint>

#include "word.hpp"

namespace hotarubi::iss {

inline constexpr uint32_t kRomWords = 256;

extern const std::array<Word128, kRomWords> kRom;

}  // namespace hotarubi::iss

#endif  // HOTARUBI_TOOLS_ISS_ROM_HPP_
