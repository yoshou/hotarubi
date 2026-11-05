#include "tensors.hpp"

#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <vector>

#include "hotarubi/exact.hpp"
#include "isa.hpp"

namespace hotarubi::compiler {
namespace {

constexpr int64_t kGeluCubicNumerator = 44715;
constexpr int64_t kGeluCubicDenominator = 1000000;
constexpr int64_t kActivationSegmentsPerUnit = 32;
constexpr size_t kScaleOffset = static_cast<size_t>(4) * kBlock;

void AppendBits(std::vector<std::byte>& out, uint32_t bits) {
  for (uint32_t i = 0; i < 4; ++i) {
    out.push_back(static_cast<std::byte>(bits >> (8 * i)));
  }
}

uint32_t ToBits(const exact::Real& value) { return value.ToBinary32(); }

void AppendTable(const std::vector<uint32_t>& v, std::vector<uint64_t>& out) {
  for (uint32_t i = 0; i < kTableSegments; ++i) {
    const uint32_t slope =
        ToBits(exact::Real::Binary32(v[i + 1]) - exact::Real::Binary32(v[i]));
    out.push_back((static_cast<uint64_t>(slope) << 32) | v[i]);
  }
}

std::vector<uint32_t> Exp2Values() {
  std::vector<uint32_t> v(kTableSegments + 1);
  for (uint32_t i = 0; i <= kTableSegments; ++i) {
    v[i] = ToBits(
        exact::Exp(exact::Real::Ratio(i, kTableSegments) * exact::Ln2()));
  }
  return v;
}

std::vector<uint32_t> ActivationValues(Activation activation) {
  const exact::Real one = exact::Real::Integer(1);
  const exact::Real cubic =
      exact::Real::Ratio(kGeluCubicNumerator, kGeluCubicDenominator);
  const exact::Real coefficient =
      exact::Sqrt(exact::Real::Integer(2) / exact::Pi());
  std::vector<uint32_t> v(kTableSegments + 1);
  for (uint32_t i = 0; i <= kTableSegments; ++i) {
    const exact::Real x = exact::Real::Ratio(i, kActivationSegmentsPerUnit);
    if (activation == Activation::kSilu) {
      v[i] = ToBits(one / (one + exact::Exp(x.Negated())));
    } else {
      const exact::Real inner = coefficient * (x + (cubic * x * x * x));
      v[i] = ToBits((one + exact::Tanh(inner)).Scaled(-1));
    }
  }
  return v;
}

}  // namespace

std::vector<std::byte> PackWbuf(const Mxfp4Matrix& matrix) {
  Require(matrix.rows % 8 == 0 && matrix.cols % kBlock == 0,
          std::format("WBUF matrix {}x{} is not a multiple of 8x32",
                      matrix.rows, matrix.cols));
  const size_t blocks = matrix.cols / kBlock;
  std::vector<std::byte> out(static_cast<size_t>(matrix.rows / 8) * blocks *
                             kWordWbufBytes);
  for (size_t g = 0; g < matrix.rows / 8; ++g) {
    for (size_t kb = 0; kb < blocks; ++kb) {
      const size_t word = ((g * blocks) + kb) * kWordWbufBytes;
      for (size_t c = 0; c < 8; ++c) {
        const size_t row = (8 * g) + c;
        const size_t base = (row * matrix.cols) + (kb * kBlock);
        for (size_t m = 0; m < kBlock / 2; ++m) {
          const uint32_t low = matrix.code[base + (2 * m)] & 0xFU;
          const uint32_t high = matrix.code[base + (2 * m) + 1] & 0xFU;
          out[word + (c * (kBlock / 2)) + m] =
              static_cast<std::byte>(low | (high << 4));
        }
        out[word + kScaleOffset + c] = ScaleByte(matrix.e[(row * blocks) + kb]);
      }
    }
  }
  return out;
}

std::vector<std::byte> PackPbuf(std::span<const uint32_t> bits) {
  Require(bits.size() % kBlock == 0,
          std::format("PBUF vector of {} values is not a multiple of 32",
                      bits.size()));
  std::vector<std::byte> out;
  out.reserve(bits.size() * 4);
  for (const uint32_t value : bits) {
    AppendBits(out, value);
  }
  return out;
}

std::vector<std::byte> PackTbuf(std::span<const uint64_t> entries) {
  std::vector<std::byte> out(entries.size() * kWordTbufBytes);
  for (size_t i = 0; i < entries.size(); ++i) {
    for (uint32_t b = 0; b < 8; ++b) {
      out[(i * kWordTbufBytes) + b] =
          static_cast<std::byte>(entries[i] >> (8 * b));
    }
  }
  return out;
}

std::vector<uint64_t> MakeTbuf(Activation activation) {
  std::vector<uint64_t> out;
  out.reserve(kTbufWords);
  AppendTable(Exp2Values(), out);
  AppendTable(ActivationValues(activation), out);
  return out;
}

}  // namespace hotarubi::compiler
