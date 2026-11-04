#ifndef HOTARUBI_TOOLS_COMPILER_TENSORS_HPP_
#define HOTARUBI_TOOLS_COMPILER_TENSORS_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace hotarubi::compiler {

inline constexpr uint32_t kBlock = 32;
inline constexpr uint32_t kWordAbufBytes = 144;
inline constexpr uint32_t kWordWbufBytes = 144;
inline constexpr uint32_t kWordFbufBytes = 128;
inline constexpr uint32_t kWordTbufBytes = 16;
inline constexpr uint32_t kTbufWords = 1024;
inline constexpr uint32_t kTableSegments = 512;

struct Matrix {
  uint32_t rows = 0;
  uint32_t cols = 0;
  std::vector<uint32_t> bits;

  [[nodiscard]] uint32_t At(uint32_t r, uint32_t c) const {
    return bits[(static_cast<size_t>(r) * cols) + c];
  }
};

struct Mxint8Matrix {
  uint32_t rows = 0;
  uint32_t cols = 0;
  std::vector<int8_t> q;
  std::vector<int32_t> e;
};

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

uint32_t PadToBlock(uint32_t k);
int32_t ExponentField(uint32_t bits);

std::vector<std::byte> PackFbuf(const Matrix& matrix);
std::vector<std::byte> PackAbuf(const Mxint8Matrix& matrix);
std::vector<std::byte> PackWbuf(const Mxfp4Matrix& matrix);
std::vector<std::byte> PackPbuf(std::span<const uint32_t> bits);
std::vector<std::byte> PackTbuf(std::span<const uint64_t> entries);

Mxint8Matrix EncodeMxint8(const Matrix& matrix);
std::vector<uint64_t> MakeTbuf(Activation activation);

}  // namespace hotarubi::compiler

#endif  // HOTARUBI_TOOLS_COMPILER_TENSORS_HPP_
