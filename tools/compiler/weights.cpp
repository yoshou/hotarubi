#include "weights.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "hotarubi/safetensors.hpp"
#include "isa.hpp"
#include "tensors.hpp"

namespace hotarubi::compiler {
namespace {

constexpr std::array<uint32_t, 8> kHalfUnits = {0, 1, 2, 3, 4, 6, 8, 12};
constexpr std::array<int32_t, 8> kFloorLog2 = {0, -1, 0, 0, 1, 1, 2, 2};
constexpr uint8_t kSign = 8;
constexpr uint8_t kMagnitude = 7;
constexpr int32_t kScaleBias = 127;
constexpr int32_t kTopExponent = 2;

uint8_t ScaledCode(uint8_t code, int32_t shift) {
  const uint32_t half = kHalfUnits[code & kMagnitude] << shift;
  const auto* const found = std::ranges::find(kHalfUnits, half);
  Require(found != kHalfUnits.end(),
          std::format("E2M1 code {} scaled by 2^{} is not representable", code,
                      shift));
  const auto index = static_cast<uint8_t>(found - kHalfUnits.begin());
  return index != 0 && (code & kSign) != 0 ? index | kSign : index;
}

}  // namespace

Mxfp4Matrix CanonicalMxfp4(const Mxfp4Matrix& matrix, int32_t exponent_shift,
                           uint32_t rows) {
  Require(rows >= matrix.rows && matrix.cols % kBlock == 0,
          std::format("cannot pad a {}x{} MXFP4 matrix to {} rows", matrix.rows,
                      matrix.cols, rows));
  const size_t cols = matrix.cols;
  const size_t blocks = cols / kBlock;
  Mxfp4Matrix out{.rows = rows,
                  .cols = matrix.cols,
                  .code = std::vector<uint8_t>(rows * cols),
                  .e = std::vector<int32_t>(rows * blocks)};
  for (size_t r = 0; r < matrix.rows; ++r) {
    for (size_t kb = 0; kb < blocks; ++kb) {
      const size_t first = (r * cols) + (kb * kBlock);
      const auto block = std::span(matrix.code).subspan(first, kBlock);
      uint8_t top = 0;
      for (const uint8_t code : block) {
        top = std::max<uint8_t>(top, code & kMagnitude);
      }
      if (top == 0) {
        continue;
      }
      out.e[(r * blocks) + kb] = matrix.e[(r * blocks) + kb] + kFloorLog2[top] -
                                 kTopExponent + exponent_shift;
      for (size_t k = 0; k < kBlock; ++k) {
        out.code[first + k] =
            ScaledCode(block[k], kTopExponent - kFloorLog2[top]);
      }
    }
  }
  return out;
}

W8Planes SplitW8(const Mxint8Matrix& matrix) {
  W8Planes planes{
      .hi = {.rows = matrix.rows,
             .cols = matrix.cols,
             .code = {},
             .e = matrix.e},
      .lo = {
          .rows = matrix.rows, .cols = matrix.cols, .code = {}, .e = matrix.e}};
  planes.hi.code.reserve(matrix.q.size());
  planes.lo.code.reserve(matrix.q.size());
  for (const int8_t q : matrix.q) {
    const auto byte = static_cast<uint8_t>(q);
    planes.hi.code.push_back(static_cast<uint8_t>(byte >> 4));
    planes.lo.code.push_back(static_cast<uint8_t>(byte & 0xFU));
  }
  return planes;
}

Matrix PadRows(const Matrix& matrix, uint32_t rows) {
  Require(rows >= matrix.rows,
          std::format("cannot pad {} rows to {}", matrix.rows, rows));
  Matrix out{.rows = rows, .cols = matrix.cols, .bits = matrix.bits};
  out.bits.resize(static_cast<size_t>(rows) * matrix.cols);
  return out;
}

Matrix Columns(const Matrix& matrix, uint32_t begin, uint32_t end) {
  Require(begin <= end && end <= matrix.cols,
          std::format("columns [{}, {}) of a matrix with {} columns", begin,
                      end, matrix.cols));
  Matrix out{.rows = matrix.rows, .cols = end - begin, .bits = {}};
  out.bits.reserve(static_cast<size_t>(out.rows) * out.cols);
  for (uint32_t r = 0; r < matrix.rows; ++r) {
    for (uint32_t c = begin; c < end; ++c) {
      out.bits.push_back(matrix.At(r, c));
    }
  }
  return out;
}

std::vector<uint32_t> PadValues(std::span<const uint32_t> values, size_t size) {
  Require(values.size() <= size,
          std::format("cannot pad {} values to {}", values.size(), size));
  std::vector<uint32_t> out(values.begin(), values.end());
  out.resize(size);
  return out;
}

Mxfp4Matrix ReadMxfp4(const SafeTensors& tensors, std::string_view module) {
  const std::string codes = std::format("{}.weight.mxfp4", module);
  const std::string scales = std::format("{}.weight.scale", module);
  const TensorInfo& info = tensors.Info(codes);
  Require(info.shape.size() == 2 && info.shape[1] % kBlock == 0,
          std::format("{} is not an MXFP4 matrix", codes));
  Mxfp4Matrix out{.rows = static_cast<uint32_t>(info.shape[0]),
                  .cols = static_cast<uint32_t>(info.shape[1]),
                  .code = tensors.Bytes(codes),
                  .e = {}};
  const std::vector<uint8_t> scale = tensors.Bytes(scales);
  Require(scale.size() == static_cast<size_t>(out.rows) * (out.cols / kBlock),
          std::format("{} does not match {}", scales, codes));
  out.e.reserve(scale.size());
  for (const uint8_t byte : scale) {
    out.e.push_back(static_cast<int32_t>(byte) - kScaleBias);
  }
  return out;
}

Mxint8Matrix ReadMxint8(const SafeTensors& tensors, std::string_view name) {
  const std::string values = std::format("{}.mxint8", name);
  const std::string scales = std::format("{}.scale", name);
  const TensorInfo& info = tensors.Info(values);
  Require(info.shape.size() == 2 && info.shape[1] % kBlock == 0,
          std::format("{} is not an MXINT8 matrix", values));
  Mxint8Matrix out{.rows = static_cast<uint32_t>(info.shape[0]),
                   .cols = static_cast<uint32_t>(info.shape[1]),
                   .q = tensors.Int8s(values),
                   .e = {}};
  const std::vector<uint8_t> scale = tensors.Bytes(scales);
  Require(scale.size() == static_cast<size_t>(out.rows) * (out.cols / kBlock),
          std::format("{} does not match {}", scales, values));
  out.e.reserve(scale.size());
  for (const uint8_t byte : scale) {
    out.e.push_back(static_cast<int32_t>(byte) - kScaleBias);
  }
  return out;
}

Matrix ReadMatrix(const SafeTensors& tensors, std::string_view name) {
  const TensorInfo& info = tensors.Info(name);
  Require(!info.shape.empty(), std::format("{} is a scalar", name));
  uint64_t cols = 1;
  for (size_t i = 1; i < info.shape.size(); ++i) {
    cols *= info.shape[i];
  }
  return {.rows = static_cast<uint32_t>(info.shape[0]),
          .cols = static_cast<uint32_t>(cols),
          .bits = tensors.Float32Bits(name)};
}

}  // namespace hotarubi::compiler
