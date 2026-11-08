#ifndef HOTARUBI_TOOLS_COMPILER_VEC_ROM_HPP_
#define HOTARUBI_TOOLS_COMPILER_VEC_ROM_HPP_

#include <array>
#include <cstdint>
#include <string>

#include "isa.hpp"
#include "vprog.hpp"

namespace hotarubi::compiler::vec_rom {

inline constexpr uint32_t kOperations = 16;

struct Rom {
  std::array<uint32_t, kOperations> starts{};
  std::array<Instruction, vprog::kRomWords> words{};
};

Rom Build();
std::string Package(const Rom& rom);

}  // namespace hotarubi::compiler::vec_rom

#endif  // HOTARUBI_TOOLS_COMPILER_VEC_ROM_HPP_
