#ifndef HOTARUBI_FORMATS_HPP_
#define HOTARUBI_FORMATS_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace hotarubi {

inline constexpr uint32_t kBlock = 32;
inline constexpr uint32_t kWordAbufBytes = 144;
inline constexpr uint32_t kWordFbufBytes = 128;

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

uint32_t PadToBlock(uint32_t k);
int32_t ExponentField(uint32_t bits);
std::byte ScaleByte(int32_t e);

std::vector<std::byte> PackFbuf(const Matrix& matrix);
Matrix UnpackFbuf(std::span<const std::byte> bytes, uint32_t rows,
                  uint32_t cols);
std::vector<std::byte> PackAbuf(const Mxint8Matrix& matrix);
Mxint8Matrix EncodeMxint8(const Matrix& matrix);

}  // namespace hotarubi

#endif  // HOTARUBI_FORMATS_HPP_
