#ifndef HOTARUBI_TOOLS_COMPILER_SPARSEDRIVE_PARTS_HPP_
#define HOTARUBI_TOOLS_COMPILER_SPARSEDRIVE_PARTS_HPP_

#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "checkpoint.hpp"
#include "ddr_image.hpp"
#include "hotarubi/safetensors.hpp"
#include "hotarubi/sparsedrive_layout.hpp"
#include "isa.hpp"
#include "program.hpp"
#include "tensors.hpp"

namespace hotarubi::compiler::sparsedrive {

inline constexpr uint64_t kWordF = kWordFbufBytes;
inline constexpr uint64_t kWordA = kWordAbufBytes;
inline constexpr uint64_t kWordT = kWordTbufBytes;
inline constexpr uint64_t kWordC = 16;
inline constexpr uint32_t kPlane = 256;
inline constexpr uint32_t kNull = 0xFFFFFFFFU;
inline constexpr uint32_t kAbufWords = kMemDepth[0];
inline constexpr uint32_t kWbufWords = kMemDepth[1];
inline constexpr uint32_t kFbufWords = kMemDepth[2];
inline constexpr uint32_t kPbufWords = kMemDepth[3];
using ::hotarubi::sparsedrive::kCameras;

enum class MapKind : uint8_t {
  kInt8,
  kFloat,
};

uint64_t TokenBytes(Mem mem, uint32_t tok);

template <typename T>
const T& Required(const std::optional<T>& value, std::string_view what) {
  if (!value.has_value()) {
    Fail(std::format("{} is missing", what));
  }
  return *value;
}

class Arena {
 public:
  Arena(DdrImage& image, std::string_view name, uint64_t size);
  void Reset() { used_ = 0; }
  uint64_t Alloc(uint64_t size, uint64_t align = DdrImage::kPage);

 private:
  uint64_t base_;
  uint64_t size_;
  uint64_t used_ = 0;
};

struct FMap {
  std::string name;
  uint32_t h = 0;
  uint32_t w = 0;
  uint32_t c = 0;
  MapKind kind = MapKind::kInt8;
  std::vector<uint32_t> planes;
  std::vector<uint32_t> tok;
  std::vector<uint64_t> tb;
  std::vector<uint64_t> off;
  uint64_t alloc_tokens = 0;

  [[nodiscard]] uint32_t Tokens() const { return h * w; }
  [[nodiscard]] Mem Memory() const {
    return kind == MapKind::kInt8 ? Mem::kAbuf : Mem::kFbuf;
  }
  [[nodiscard]] uint64_t Addr(size_t p, uint64_t token) const {
    return off[p] + (token * tb[p]);
  }
  [[nodiscard]] uint32_t QuadWords(size_t p) const {
    return kind == MapKind::kInt8 ? tok[p] : 4 * tok[p];
  }
};

FMap MakeMap(std::string name, uint32_t h, uint32_t w, uint32_t c,
             MapKind kind);
FMap& PlaceMap(FMap& map, DdrImage& image,
               std::optional<uint64_t> slack = std::nullopt,
               Arena* arena = nullptr);
FMap& AliasMap(FMap& map, const FMap& other);

class Lists {
 public:
  size_t Add(std::vector<uint32_t> list);
  void Place(DdrImage& image, std::string_view name);
  [[nodiscard]] uint64_t Offset(size_t id) const { return offsets_.at(id); }

 private:
  std::vector<std::vector<uint32_t>> lists_;
  std::vector<uint64_t> offsets_;
};

struct Sources {
  const SafeTensors& quant;
  const Checkpoint& checkpoint;
};

struct Conv {
  Mxfp4Matrix w;
  uint32_t kh = 1;
  uint32_t cin = 0;
  std::vector<uint32_t> bias;
};

Conv ReadConv(const Sources& sources, std::string_view module, uint32_t kh,
              bool folded);
Mxfp4Matrix ConvTile(const Conv& conv, uint32_t n0, uint32_t n, uint32_t tap,
                     uint32_t c0, uint32_t c);
std::vector<uint32_t> AddBits(const std::vector<uint32_t>& a,
                              const std::vector<uint32_t>& b);
std::vector<uint32_t> PaddedBias(const std::vector<uint32_t>& bias);
uint32_t Entry(uint64_t base, uint64_t offset);

Transfer Words(uint64_t ddr, Mem mem, uint32_t onchip, uint32_t words);
Transfer Tokens(uint64_t ddr, Mem mem, uint32_t onchip, uint32_t tokens,
                uint32_t tok, bool index = false);

struct Backbone {
  Mxint8Matrix stem;
  std::vector<uint32_t> stem_bias;
  std::map<std::string, Conv, std::less<>> convs;
};

struct CameraInput {
  uint64_t image = 0;
  std::string lists;
  std::vector<size_t>* gathers = nullptr;
};

std::vector<std::byte> StemListBytes(uint64_t base, uint64_t image);

using ::hotarubi::sparsedrive::kStemImageBytes;

Backbone ReadBackbone(const Sources& sources);
std::vector<FMap> EmitEncoder(Program& program, DdrImage& image, uint64_t base,
                              const Backbone& backbone,
                              const CameraInput& input, uint32_t camera);

std::vector<Instruction> StemQuantKernel();
void EmitSingleConv(Program& program, DdrImage& image, uint64_t base,
                    const std::string& name, const FMap& x, const Conv& conv,
                    uint32_t stride, const FMap* out_f32, const FMap* out_i8,
                    bool relu);

}  // namespace hotarubi::compiler::sparsedrive

#endif  // HOTARUBI_TOOLS_COMPILER_SPARSEDRIVE_PARTS_HPP_
