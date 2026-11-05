#ifndef HOTARUBI_TOOLS_COMPILER_TENSORS_HPP_
#define HOTARUBI_TOOLS_COMPILER_TENSORS_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "hotarubi/formats.hpp"

namespace hotarubi::compiler {

inline constexpr uint32_t kWordWbufBytes = 144;
inline constexpr uint32_t kWordTbufBytes = 16;
inline constexpr uint32_t kTbufWords = 1024;
inline constexpr uint32_t kTableSegments = 512;

struct Mxfp4Matrix {
  uint32_t rows = 0;
  uint32_t cols = 0;
  std::vector<uint8_t> code;
  std::vector<int32_t> e;
};

enum class Activation : uint8_t {
  kSilu,
  kGelu,
};

std::vector<std::byte> PackWbuf(const Mxfp4Matrix& matrix);
std::vector<std::byte> PackPbuf(std::span<const uint32_t> bits);
std::vector<std::byte> PackTbuf(std::span<const uint64_t> entries);

std::vector<uint64_t> MakeTbuf(Activation activation);

}  // namespace hotarubi::compiler

#endif  // HOTARUBI_TOOLS_COMPILER_TENSORS_HPP_
