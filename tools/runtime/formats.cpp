#include "hotarubi/formats.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <vector>

#include "hotarubi/require.hpp"

namespace hotarubi {
namespace {

constexpr int32_t kScaleBias = 127;
constexpr int32_t kInt8Fraction = 6;
constexpr uint32_t kSignMask = 0x80000000U;
constexpr uint32_t kMagnitudeMask = 0x7FFFFFFFU;
constexpr size_t kScaleOffset = static_cast<size_t>(4) * kBlock;

void AppendBits(std::vector<std::byte>& out, uint32_t bits) {
  for (uint32_t i = 0; i < 4; ++i) {
    out.push_back(static_cast<std::byte>(bits >> (8 * i)));
  }
}

uint32_t LoadBits(std::span<const std::byte> bytes, size_t offset) {
  uint32_t bits = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    bits |= static_cast<uint32_t>(bytes[offset + i]) << (8 * i);
  }
  return bits;
}

int8_t QuantizeInt8(uint32_t bits, int32_t e) {
  const uint32_t field = (bits >> 23) & 0xFFU;
  Require(field != 0xFFU, "MXINT8 input is infinite or NaN");
  const uint64_t significand =
      field == 0 ? (bits & 0x7FFFFFU) : ((bits & 0x7FFFFFU) | 0x800000U);
  const int64_t shift =
      (field == 0 ? -149 : static_cast<int64_t>(field) - 150) + kInt8Fraction -
      e;
  int64_t magnitude = 0;
  if (shift >= 0) {
    magnitude = shift > 8 ? 256 : static_cast<int64_t>(significand << shift);
  } else if (shift > -64) {
    const uint64_t kept = significand >> (-shift);
    const uint64_t rest = significand & ((uint64_t{1} << (-shift)) - 1);
    const uint64_t half = uint64_t{1} << (-shift - 1);
    const bool up = rest > half || (rest == half && (kept & 1U) != 0);
    magnitude = static_cast<int64_t>(kept + (up ? 1 : 0));
  }
  const bool negative = (bits & kSignMask) != 0;
  const int64_t value = negative ? -magnitude : magnitude;
  return static_cast<int8_t>(std::clamp<int64_t>(value, -128, 127));
}

}  // namespace

uint32_t PadToBlock(uint32_t k) { return (k + kBlock - 1) / kBlock * kBlock; }

int32_t ExponentField(uint32_t bits) {
  return static_cast<int32_t>((bits >> 23) & 0xFFU) - kScaleBias;
}

std::byte ScaleByte(int32_t e) {
  Require(e >= -kScaleBias && e <= kScaleBias,
          std::format("block exponent {} is outside E8M0", e));
  return static_cast<std::byte>(e + kScaleBias);
}

std::vector<std::byte> PackFbuf(const Matrix& matrix) {
  Require(matrix.rows % 4 == 0 && matrix.cols % 8 == 0,
          std::format("FBUF matrix {}x{} is not a multiple of 4x8", matrix.rows,
                      matrix.cols));
  std::vector<std::byte> out;
  out.reserve(static_cast<size_t>(matrix.rows) * matrix.cols * 4);
  for (uint32_t q = 0; q < matrix.rows / 4; ++q) {
    for (uint32_t g = 0; g < matrix.cols / 8; ++g) {
      for (uint32_t j = 0; j < 4; ++j) {
        for (uint32_t c = 0; c < 8; ++c) {
          AppendBits(out, matrix.At((4 * q) + j, (8 * g) + c));
        }
      }
    }
  }
  return out;
}

Matrix UnpackFbuf(std::span<const std::byte> bytes, uint32_t rows,
                  uint32_t cols) {
  Require(rows % 4 == 0 && cols % 8 == 0 &&
              bytes.size() == static_cast<size_t>(rows) * cols * 4,
          std::format("{} bytes are not an FBUF matrix of {}x{}", bytes.size(),
                      rows, cols));
  Matrix out{.rows = rows,
             .cols = cols,
             .bits = std::vector<uint32_t>(static_cast<size_t>(rows) * cols)};
  size_t offset = 0;
  for (uint32_t q = 0; q < rows / 4; ++q) {
    for (uint32_t g = 0; g < cols / 8; ++g) {
      for (uint32_t j = 0; j < 4; ++j) {
        for (uint32_t c = 0; c < 8; ++c) {
          out.bits[(static_cast<size_t>((4 * q) + j) * cols) + (size_t{8} * g) +
                   c] = LoadBits(bytes, offset);
          offset += 4;
        }
      }
    }
  }
  return out;
}

std::vector<std::byte> PackAbuf(const Mxint8Matrix& matrix) {
  Require(matrix.rows % 4 == 0 && matrix.cols % kBlock == 0,
          std::format("ABUF matrix {}x{} is not a multiple of 4x32",
                      matrix.rows, matrix.cols));
  const size_t blocks = matrix.cols / kBlock;
  std::vector<std::byte> out(static_cast<size_t>(matrix.rows / 4) * blocks *
                             kWordAbufBytes);
  for (size_t q = 0; q < matrix.rows / 4; ++q) {
    for (size_t kb = 0; kb < blocks; ++kb) {
      const size_t word = ((q * blocks) + kb) * kWordAbufBytes;
      for (size_t j = 0; j < 4; ++j) {
        const size_t row = (4 * q) + j;
        for (size_t k = 0; k < kBlock; ++k) {
          out[word + (j * kBlock) + k] =
              static_cast<std::byte>(static_cast<uint8_t>(
                  matrix.q[(row * matrix.cols) + (kb * kBlock) + k]));
        }
        out[word + kScaleOffset + j] = ScaleByte(matrix.e[(row * blocks) + kb]);
      }
    }
  }
  return out;
}

Mxint8Matrix EncodeMxint8(const Matrix& matrix) {
  const uint32_t cols = PadToBlock(matrix.cols);
  const uint32_t blocks = cols / kBlock;
  Mxint8Matrix out{
      .rows = matrix.rows,
      .cols = cols,
      .q = std::vector<int8_t>(static_cast<size_t>(matrix.rows) * cols),
      .e = std::vector<int32_t>(static_cast<size_t>(matrix.rows) * blocks)};
  for (uint32_t r = 0; r < matrix.rows; ++r) {
    for (uint32_t kb = 0; kb < blocks; ++kb) {
      const uint32_t end = std::min(matrix.cols, (kb + 1) * kBlock);
      uint32_t amax = 0;
      for (uint32_t c = kb * kBlock; c < end; ++c) {
        amax = std::max(amax, matrix.At(r, c) & kMagnitudeMask);
      }
      const int32_t e = amax != 0 ? ExponentField(amax) : 0;
      out.e[(static_cast<size_t>(r) * blocks) + kb] = e;
      for (uint32_t c = kb * kBlock; c < end; ++c) {
        out.q[(static_cast<size_t>(r) * cols) + c] =
            QuantizeInt8(matrix.At(r, c), e);
      }
    }
  }
  return out;
}

}  // namespace hotarubi
