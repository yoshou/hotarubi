#ifndef HOTARUBI_TOOLS_COMPILER_VEC_ROM_HPP_
#define HOTARUBI_TOOLS_COMPILER_VEC_ROM_HPP_

#include <array>
#include <cstdint>
#include <string>

#include "isa.hpp"
#include "vprog.hpp"

namespace hotarubi::compiler::vec_rom {

struct Rom {
  std::array<Instruction, vprog::kRomWords> words{};
};

Rom Build();
std::string Package(const Rom& rom);

}  // namespace hotarubi::compiler::vec_rom

#endif  // HOTARUBI_TOOLS_COMPILER_VEC_ROM_HPP_
