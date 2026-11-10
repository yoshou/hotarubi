#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ddr_image.hpp"
#include "hotarubi/exact.hpp"
#include "isa.hpp"
#include "program.hpp"
#include "sparsedrive_parts.hpp"
#include "tensors.hpp"
#include "weights.hpp"

namespace hotarubi::compiler::sparsedrive {
namespace {

constexpr uint64_t kSlackInt8 = 1024;
constexpr uint64_t kSlackFloat = 256;
constexpr uint64_t kEntryBytes = 16;

}  // namespace

uint64_t TokenBytes(Mem mem, uint32_t tok) {
  if (mem == Mem::kAbuf) {
    return (uint64_t{tok} * 32) + ((uint64_t{tok} + 15) / 16 * 16);
  }
  return uint64_t{tok} * 128;
}

Arena::Arena(DdrImage& image, std::string_view name, uint64_t size)
    : base_(image.Alloc(name, size)), size_(size) {}

uint64_t Arena::Alloc(uint64_t size, uint64_t align) {
  const uint64_t offset = (used_ + align - 1) / align * align;
  Require(offset + size <= size_,
          std::format("arena of {} bytes cannot hold {} more at {}", size_,
                      size, offset));
  used_ = offset + size;
  return base_ + offset;
}

FMap MakeMap(std::string name, uint32_t h, uint32_t w, uint32_t c,
             MapKind kind) {
  Require(h % 4 == 0 && c % kBlock == 0 && (c <= kPlane || c % kPlane == 0),
          std::format("feature map {} of {}x{}x{} is not supported", name, h, w,
                      c));
  FMap map;
  map.name = std::move(name);
  map.h = h;
  map.w = w;
  map.c = c;
  map.kind = kind;
  if (c >= kPlane) {
    map.planes.assign(c / kPlane, kPlane);
  } else {
    map.planes.push_back(c);
  }
  for (const uint32_t channels : map.planes) {
    map.tok.push_back(channels / kBlock);
    map.tb.push_back(TokenBytes(map.Memory(), channels / kBlock));
  }
  return map;
}

FMap& PlaceMap(FMap& map, DdrImage& image, std::optional<uint64_t> slack,
               Arena* arena) {
  map.alloc_tokens =
      map.Tokens() +
      slack.value_or(map.kind == MapKind::kInt8 ? kSlackInt8 : kSlackFloat);
  map.off.clear();
  for (size_t p = 0; p < map.planes.size(); ++p) {
    const uint64_t bytes = map.alloc_tokens * map.tb[p];
    map.off.push_back(
        arena != nullptr
            ? arena->Alloc(bytes)
            : image.Alloc(std::format("{}.p{}", map.name, p), bytes));
  }
  return map;
}

FMap& AliasMap(FMap& map, const FMap& other) {
  const uint64_t slack = map.kind == MapKind::kInt8 ? kSlackInt8 : kSlackFloat;
  Require(other.kind == map.kind && other.planes == map.planes &&
              map.Tokens() + slack <= other.alloc_tokens,
          std::format("{} cannot reuse {}", map.name, other.name));
  map.off = other.off;
  map.alloc_tokens = other.alloc_tokens;
  return map;
}

size_t Lists::Add(std::vector<uint32_t> list) {
  Require(list.size() % 4 == 0, "an index list is not a multiple of 4");
  lists_.push_back(std::move(list));
  return lists_.size() - 1;
}

void Lists::Place(DdrImage& image, std::string_view name) {
  uint64_t total = 0;
  for (const auto& list : lists_) {
    total += list.size() * 4;
  }
  uint64_t at = image.Alloc(name, std::max<uint64_t>(total, 16));
  offsets_.clear();
  for (const auto& list : lists_) {
    offsets_.push_back(at);
    std::vector<std::byte> bytes;
    bytes.reserve(list.size() * 4);
    for (const uint32_t entry : list) {
      for (uint32_t i = 0; i < 4; ++i) {
        bytes.push_back(static_cast<std::byte>(entry >> (8 * i)));
      }
    }
    image.Put(at, bytes);
    at += bytes.size();
  }
}

Conv ReadConv(const Sources& sources, std::string_view module, uint32_t kh,
              bool folded) {
  const Mxfp4Matrix raw = ReadMxfp4(sources.quant, module);
  Require(raw.cols % (kh * kh) == 0,
          std::format("{} does not have {}x{} taps", module, kh, kh));
  const std::string bias = std::format("{}.bias", module);
  return {.w = CanonicalMxfp4(raw, 0, raw.rows),
          .kh = kh,
          .cin = raw.cols / (kh * kh),
          .bias = folded ? sources.quant.Float32Bits(bias)
                         : sources.checkpoint.Float32Bits(bias)};
}

Mxfp4Matrix ConvTile(const Conv& conv, uint32_t n0, uint32_t n, uint32_t tap,
                     uint32_t c0, uint32_t c) {
  const uint32_t first = (tap * conv.cin) + c0;
  Require(n0 + n <= conv.w.rows && first + c <= conv.w.cols &&
              first % kBlock == 0 && c % kBlock == 0,
          "conv weight tile is outside the matrix");
  const uint32_t blocks = conv.w.cols / kBlock;
  Mxfp4Matrix tile{.rows = n, .cols = c, .code = {}, .e = {}};
  tile.code.reserve(static_cast<size_t>(n) * c);
  tile.e.reserve(static_cast<size_t>(n) * (c / kBlock));
  for (uint32_t r = n0; r < n0 + n; ++r) {
    const size_t row = static_cast<size_t>(r) * conv.w.cols;
    tile.code.insert(
        tile.code.end(),
        conv.w.code.begin() + static_cast<std::ptrdiff_t>(row + first),
        conv.w.code.begin() + static_cast<std::ptrdiff_t>(row + first + c));
    for (uint32_t b = 0; b < c / kBlock; ++b) {
      tile.e.push_back(
          conv.w.e[(static_cast<size_t>(r) * blocks) + (first / kBlock) + b]);
    }
  }
  return tile;
}

std::vector<uint32_t> AddBits(const std::vector<uint32_t>& a,
                              const std::vector<uint32_t>& b) {
  Require(a.size() == b.size(), "cannot add vectors of different sizes");
  std::vector<uint32_t> out(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    out[i] = (exact::Real::Binary32(a[i]) + exact::Real::Binary32(b[i]))
                 .ToBinary32();
  }
  return out;
}

std::vector<uint32_t> PaddedBias(const std::vector<uint32_t>& bias) {
  return PadValues(bias, PadToBlock(static_cast<uint32_t>(bias.size())));
}

uint32_t Entry(uint64_t base, uint64_t offset) {
  const uint64_t entry = (base + offset) / kEntryBytes;
  Require((base + offset) % kEntryBytes == 0 && entry < kNull,
          std::format("index list entry for {:#x} is not a 16-byte address",
                      base + offset));
  return static_cast<uint32_t>(entry);
}

Transfer Words(uint64_t ddr, Mem mem, uint32_t onchip, uint32_t words) {
  return {.ddr_offset = ddr,
          .mem = mem,
          .onchip = onchip,
          .rows = 1,
          .row_words = words};
}

Transfer Tokens(uint64_t ddr, Mem mem, uint32_t onchip, uint32_t tokens,
                uint32_t tok, bool index) {
  return {.ddr_offset = ddr,
          .mem = mem,
          .onchip = onchip,
          .rows = 1,
          .row_words = tokens,
          .ddr_stride = 0,
          .kind = index ? DmaKind::kRowIndexed : DmaKind::kRow,
          .tok = tok};
}

}  // namespace hotarubi::compiler::sparsedrive
