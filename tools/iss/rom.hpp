#ifndef HOTARUBI_TOOLS_ISS_ROM_HPP_
#define HOTARUBI_TOOLS_ISS_ROM_HPP_

#include <array>
#include <cstdint>

#include "hotarubi/isa.hpp"
#include "word.hpp"

namespace hotarubi::iss {

using isa::vec::kRomWords;

extern const std::array<Word128, kRomWords> kRom;

}  // namespace hotarubi::iss

#endif  // HOTARUBI_TOOLS_ISS_ROM_HPP_
