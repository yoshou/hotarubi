#ifndef HOTARUBI_TOOLS_COMPILER_SPARSEDRIVE_HPP_
#define HOTARUBI_TOOLS_COMPILER_SPARSEDRIVE_HPP_

#include <cstdint>

#include "checkpoint.hpp"
#include "hotarubi/safetensors.hpp"
#include "output.hpp"

namespace hotarubi::compiler {

struct SparseDriveOptions {
  uint64_t base = 0;
};

Compiled CompileSparseDrive(const SparseDriveOptions& options,
                            const SafeTensors& quant,
                            const Checkpoint& checkpoint);

}  // namespace hotarubi::compiler

#endif  // HOTARUBI_TOOLS_COMPILER_SPARSEDRIVE_HPP_
