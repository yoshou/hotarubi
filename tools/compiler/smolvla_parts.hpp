#ifndef HOTARUBI_TOOLS_COMPILER_SMOLVLA_PARTS_HPP_
#define HOTARUBI_TOOLS_COMPILER_SMOLVLA_PARTS_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "ddr_image.hpp"
#include "hotarubi/exact.hpp"
#include "hotarubi/safetensors.hpp"
#include "isa.hpp"
#include "program.hpp"
#include "tensors.hpp"

namespace hotarubi::compiler::smolvla {

inline constexpr uint64_t kWordF = kWordFbufBytes;
inline constexpr uint64_t kWordA = kWordAbufBytes;
inline constexpr uint64_t kWordT = kWordTbufBytes;
inline constexpr uint32_t kImageTokens = 64;
inline constexpr uint32_t kMaxLanguage = 48;
inline constexpr uint32_t kLmWords = 120;
inline constexpr uint32_t kLmBlocks = 30;
inline constexpr uint32_t kKvBlocks = 10;
inline constexpr uint32_t kKvGroups = 40;
inline constexpr uint32_t kQueryHeads = 15;
inline constexpr uint32_t kKvHeads = 5;
inline constexpr uint32_t kHeadPairs = 32;
inline constexpr uint32_t kSuffix = 64;
inline constexpr uint32_t kTbufWordsLoaded = 1024;

struct Dims {
  uint32_t cameras = 1;
  uint32_t lang = 0;
  uint32_t steps = 10;

  [[nodiscard]] uint32_t Image() const { return kImageTokens * cameras; }
  [[nodiscard]] uint32_t Context() const { return Image() + lang; }
  [[nodiscard]] uint32_t Prefix() const { return Context() + 1; }
  [[nodiscard]] uint32_t Tokens() const {
    return (Image() + kMaxLanguage + 1 + kBlock - 1) / kBlock * kBlock;
  }
};

struct VisionLayer {
  std::array<uint64_t, 6> weights{};
  uint64_t pbuf = 0;
};

struct Vision {
  uint64_t tbuf = 0;
  uint64_t patch_hi = 0;
  uint64_t patch_lo = 0;
  uint64_t patch_pbuf = 0;
  uint64_t patch_pos = 0;
  uint64_t patch_in = 0;
  uint64_t state_hi = 0;
  uint64_t state_lo = 0;
  uint64_t state_pbuf = 0;
  uint64_t state_in = 0;
  std::vector<VisionLayer> layers;
  uint64_t connector = 0;
  uint64_t connector_pbuf = 0;
  uint64_t q = 0;
  uint64_t k = 0;
  uint64_t v = 0;
  uint64_t o = 0;
  uint64_t h = 0;
  std::array<uint64_t, 4> x{};
};

struct DecoderLayer {
  std::array<uint64_t, 7> weights{};
  uint64_t pbuf = 0;
};

struct Lm {
  uint64_t rope = 0;
  uint64_t tbuf = 0;
  std::vector<DecoderLayer> layers;
  uint64_t o = 0;
  std::vector<uint64_t> k;
  std::vector<uint64_t> v;
  std::array<uint64_t, 4> x{};
};

struct W8Region {
  uint64_t hi = 0;
  uint64_t lo = 0;
};

struct Expert {
  std::array<W8Region, 5> heads{};
  uint64_t pb_heads = 0;
  uint64_t rope = 0;
  uint64_t tbuf = 0;
  uint64_t mask = 0;
  std::vector<uint64_t> time;
  std::vector<DecoderLayer> layers;
  uint64_t o = 0;
  std::vector<uint64_t> kc;
  std::vector<uint64_t> vc;
  std::array<uint64_t, 3> x{};
  std::vector<uint64_t> xt;
  std::vector<uint64_t> vt;
};

size_t Ld(Program& program, uint64_t ddr, Mem mem, uint32_t onchip,
          uint32_t rows, uint32_t row_words, uint64_t stride,
          std::string_view note, DmaKind kind = DmaKind::kRaw);
size_t St(Program& program, uint64_t ddr, Mem mem, uint32_t onchip,
          uint32_t rows, uint32_t row_words, uint64_t stride,
          std::string_view note);
uint64_t Place(DdrImage& image, std::string_view name,
               std::span<const std::byte> data);
uint32_t Fp32OfFp64(const exact::Real& value);
uint32_t RmsEps();
std::vector<std::byte> RopeTables(std::span<const uint32_t> positions);
W8Region PlaceW8(DdrImage& image, std::string_view name, const Matrix& weights,
                 uint32_t rows);

Vision PlaceVision(DdrImage& image, const SafeTensors& quant,
                   const SafeTensors& model, const Dims& dims);
void EmitVision(Program& program, const Vision& vision, const Dims& dims,
                uint64_t lm_input);
Lm PlaceLm(DdrImage& image, const SafeTensors& quant, const SafeTensors& model,
           const Dims& dims);
void EmitLm(Program& program, const Lm& lm, const Dims& dims);
Expert PlaceExpert(DdrImage& image, const SafeTensors& quant,
                   const SafeTensors& model, const Dims& dims);
void EmitExpert(Program& program, const Expert& expert, const Lm& lm,
                const Dims& dims);

}  // namespace hotarubi::compiler::smolvla

#endif  // HOTARUBI_TOOLS_COMPILER_SMOLVLA_PARTS_HPP_
