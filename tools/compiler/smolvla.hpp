#ifndef HOTARUBI_TOOLS_COMPILER_SMOLVLA_HPP_
#define HOTARUBI_TOOLS_COMPILER_SMOLVLA_HPP_

#include <cstdint>

#include "hotarubi/safetensors.hpp"
#include "output.hpp"

namespace hotarubi::compiler {

struct SmolVlaOptions {
  uint32_t cameras = 2;
  uint32_t lang_tokens = 0;
  uint32_t steps = 10;
  uint64_t base = 0;
  uint32_t schedule_window = 64;
};

Compiled CompileSmolVla(const SmolVlaOptions& options, const SafeTensors& quant,
                        const SafeTensors& model);

}  // namespace hotarubi::compiler

#endif  // HOTARUBI_TOOLS_COMPILER_SMOLVLA_HPP_
