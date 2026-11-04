#ifndef HOTARUBI_TOOLS_COMPILER_WEIGHTS_HPP_
#define HOTARUBI_TOOLS_COMPILER_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "safetensors.hpp"
#include "tensors.hpp"

namespace hotarubi::compiler {

struct W8Planes {
  Mxfp4Matrix hi;
  Mxfp4Matrix lo;
};

Mxfp4Matrix CanonicalMxfp4(const Mxfp4Matrix& matrix, int32_t exponent_shift,
                           uint32_t rows);
W8Planes SplitW8(const Mxint8Matrix& matrix);
Matrix PadRows(const Matrix& matrix, uint32_t rows);
Matrix Columns(const Matrix& matrix, uint32_t begin, uint32_t end);
std::vector<uint32_t> PadValues(std::span<const uint32_t> values, size_t size);

Mxfp4Matrix ReadMxfp4(const SafeTensors& tensors, std::string_view module);
Mxint8Matrix ReadMxint8(const SafeTensors& tensors, std::string_view name);
Matrix ReadMatrix(const SafeTensors& tensors, std::string_view name);

}  // namespace hotarubi::compiler

#endif  // HOTARUBI_TOOLS_COMPILER_WEIGHTS_HPP_
