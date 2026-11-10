#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ddr_image.hpp"
#include "hotarubi/exact.hpp"
#include "isa.hpp"
#include "program.hpp"
#include "smolvla_parts.hpp"
#include "tensors.hpp"
#include "weights.hpp"

namespace hotarubi::compiler::smolvla {
namespace {

constexpr int64_t kRopeBase = 10000;
constexpr uint64_t kStrideLimit = uint64_t{1} << 24;

}  // namespace

size_t Ld(Program& program, uint64_t ddr, Mem mem, uint32_t onchip,
          uint32_t rows, uint32_t row_words, uint64_t stride,
          std::string_view note, DmaKind kind) {
  Require(stride < kStrideLimit,
          std::format("DMA stride {} is too large", stride));
  return program.Load({.ddr_offset = ddr,
                       .mem = mem,
                       .onchip = onchip,
                       .rows = rows,
                       .row_words = row_words,
                       .ddr_stride = static_cast<uint32_t>(stride),
                       .kind = kind},
                      note);
}

size_t St(Program& program, uint64_t ddr, Mem mem, uint32_t onchip,
          uint32_t rows, uint32_t row_words, uint64_t stride,
          std::string_view note) {
  Require(stride < kStrideLimit,
          std::format("DMA stride {} is too large", stride));
  return program.Store({.ddr_offset = ddr,
                        .mem = mem,
                        .onchip = onchip,
                        .rows = rows,
                        .row_words = row_words,
                        .ddr_stride = static_cast<uint32_t>(stride)},
                       note);
}

uint64_t Place(DdrImage& image, std::string_view name,
               std::span<const std::byte> data) {
  const uint64_t offset = image.Alloc(name, data.size());
  image.Put(offset, data);
  return offset;
}

uint32_t Fp32OfFp64(const exact::Real& value) {
  return exact::Real::Binary64(value.ToBinary64()).ToBinary32();
}

uint32_t RmsEps() {
  static const uint32_t kEps = Fp32OfFp64(exact::Real::Ratio(1, 100000));
  return kEps;
}

std::vector<std::byte> RopeTables(std::span<const uint32_t> positions) {
  const auto rows = static_cast<uint32_t>(positions.size());
  Matrix cos{
      .rows = rows,
      .cols = kHeadPairs,
      .bits = std::vector<uint32_t>(static_cast<size_t>(rows) * kHeadPairs)};
  Matrix sin = cos;
  for (uint32_t k = 0; k < kHeadPairs; ++k) {
    const exact::Real timescale =
        exact::Real::Binary32(exact::Pow(exact::Real::Integer(kRopeBase),
                                         exact::Real::Ratio(k, kHeadPairs))
                                  .ToBinary32());
    for (uint32_t i = 0; i < rows; ++i) {
      const exact::Real radians = exact::Real::Binary32(
          (exact::Real::Integer(positions[i]) / timescale).ToBinary32());
      const size_t at = (static_cast<size_t>(i) * kHeadPairs) + k;
      cos.bits[at] = exact::Cos(radians).ToBinary32();
      sin.bits[at] = exact::Sin(radians).ToBinary32();
    }
  }
  std::vector<std::byte> out = PackFbuf(cos);
  const std::vector<std::byte> packed_sin = PackFbuf(sin);
  out.insert(out.end(), packed_sin.begin(), packed_sin.end());
  return out;
}

W8Region PlaceW8(DdrImage& image, std::string_view name, const Matrix& weights,
                 uint32_t rows) {
  const W8Planes planes = SplitW8(EncodeMxint8(PadRows(weights, rows)));
  const uint64_t hi =
      Place(image, std::format("{}_hi", name), PackWbuf(planes.hi));
  const uint64_t lo =
      Place(image, std::format("{}_lo", name), PackWbuf(planes.lo));
  return {.hi = hi, .lo = lo};
}

}  // namespace hotarubi::compiler::smolvla
