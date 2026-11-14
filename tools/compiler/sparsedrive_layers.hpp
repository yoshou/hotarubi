#ifndef HOTARUBI_TOOLS_COMPILER_SPARSEDRIVE_LAYERS_HPP_
#define HOTARUBI_TOOLS_COMPILER_SPARSEDRIVE_LAYERS_HPP_

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "checkpoint.hpp"
#include "ddr_image.hpp"
#include "hotarubi/safetensors.hpp"
#include "isa.hpp"
#include "program.hpp"
#include "sparsedrive_kernels.hpp"
#include "tensors.hpp"

namespace hotarubi::compiler::sparsedrive {

inline constexpr uint32_t kD = 256;
inline constexpr uint32_t kHeads = 8;
inline constexpr uint32_t kKbD = kD / 32;
inline constexpr uint32_t kNwD = kD / 8;

struct Band {
  uint32_t t = 0;
  uint32_t i0 = 0;
  uint32_t m = 0;
};

std::vector<Band> Bands(uint32_t n, uint32_t mt);

struct Rows {
  std::string name;
  uint32_t n = 0;
  uint32_t cols = 0;
  uint64_t off = 0;

  [[nodiscard]] uint32_t Tok() const { return cols / 32; }
  [[nodiscard]] uint64_t Tb() const { return uint64_t{cols} * 4; }
  [[nodiscard]] uint64_t Addr(uint64_t j = 0) const { return off + (j * Tb()); }
};

Rows PlaceRows(DdrImage& image, std::string name, uint32_t n, uint32_t cols,
               uint64_t slack = 128);
void PutRows(DdrImage& image, const Rows& rows, uint32_t src_rows,
             uint32_t src_cols, const std::vector<uint32_t>& values);
void PutWords(DdrImage& image, uint64_t offset, const std::vector<Word>& words);
void PutEntries(DdrImage& image, uint64_t offset,
                const std::vector<uint32_t>& entries);
std::vector<std::byte> WordBytes(const std::vector<Word>& words);
std::vector<std::byte> EntryBytes(const std::vector<uint32_t>& entries);

struct LayerNorm {
  std::vector<uint32_t> g;
  std::vector<uint32_t> b;
};

struct Piece {
  Place a{};
  uint32_t kb_offset = 0;
  uint32_t kb = 0;
};

class Lin {
 public:
  Lin() = default;
  static Lin Fp4(const Mxfp4Matrix& w, const std::vector<uint32_t>& bias,
                 uint32_t n_pad);
  static Lin W8(const Mxint8Matrix& w, const std::vector<uint32_t>& bias,
                uint32_t n_pad, uint32_t k_shift);

  void Place(DdrImage& image, std::string_view name);
  [[nodiscard]] Tile Load(Program& program, std::string_view note) const;
  [[nodiscard]] std::vector<Piece> Input(compiler::Place a) const;
  void Gemm(Program& program, const std::vector<Piece>& pieces, uint32_t row,
            uint32_t a_stride, uint32_t m, compiler::Place c, const Tile& w,
            std::string_view note, bool acc = false) const;

  [[nodiscard]] uint32_t N() const { return n_; }
  [[nodiscard]] uint32_t Rows() const { return rows_; }
  [[nodiscard]] uint32_t Kb() const { return kb_; }
  [[nodiscard]] uint32_t Groups() const { return rows_ / 8; }
  [[nodiscard]] uint32_t Words() const { return Groups() * kb_; }
  [[nodiscard]] uint32_t TotalWords() const {
    return Words() * static_cast<uint32_t>(planes_.size());
  }
  [[nodiscard]] uint64_t Offset(size_t plane) const { return off_.at(plane); }
  [[nodiscard]] const std::vector<uint32_t>& Bias() const { return bias_; }

 private:
  uint32_t n_ = 0;
  uint32_t rows_ = 0;
  uint32_t kb_ = 0;
  std::vector<uint32_t> bias_;
  std::vector<std::vector<std::byte>> planes_;
  std::vector<WeightMode> modes_;
  std::vector<uint64_t> off_;
};

class Pbuf {
 public:
  uint32_t Add(const std::vector<uint32_t>& values);
  uint32_t Norm(const LayerNorm& norm);
  void Place(DdrImage& image, std::string_view name);
  [[nodiscard]] Tile Load(Program& program, std::string_view note) const;
  [[nodiscard]] uint64_t Offset() const { return off_; }
  [[nodiscard]] uint32_t Words() const { return words_; }

 private:
  std::vector<uint32_t> values_;
  uint32_t words_ = 0;
  uint64_t off_ = 0;
};

using Keys = std::map<std::string, uint32_t, std::less<>>;

class Params {
 public:
  Params(const Tile& tile, const Keys& keys) : tile_(tile), keys_(&keys) {}

  [[nodiscard]] compiler::Place At(std::string_view key) const;

 private:
  Tile tile_;
  const Keys* keys_;
};

class HeadWeights {
 public:
  HeadWeights(const SafeTensors& quant, const Checkpoint& checkpoint)
      : quant_(quant), checkpoint_(checkpoint) {}

  [[nodiscard]] Lin Linear(std::string_view module, uint32_t n_pad = 8,
                           uint32_t k_shift = 0) const;
  [[nodiscard]] std::vector<Lin> Attention(std::string_view module,
                                           uint32_t embed, bool scale_q) const;
  [[nodiscard]] LayerNorm Norm(std::string_view module) const;
  [[nodiscard]] std::vector<uint32_t> Values(std::string_view name) const;
  [[nodiscard]] Mxfp4Matrix Fp4Rows(std::string_view module) const;
  [[nodiscard]] Mxint8Matrix W8Rows(std::string_view name) const;
  [[nodiscard]] bool IsW8(std::string_view module) const;
  [[nodiscard]] const Checkpoint& Ckpt() const { return checkpoint_; }

 private:
  const SafeTensors& quant_;
  const Checkpoint& checkpoint_;
};

Mxfp4Matrix SelectRows(const Mxfp4Matrix& matrix,
                       const std::vector<int64_t>& rows);
Mxint8Matrix SelectRows(const Mxint8Matrix& matrix,
                        const std::vector<int64_t>& rows);
std::vector<uint32_t> SelectValues(const std::vector<uint32_t>& values,
                                   const std::vector<int64_t>& rows,
                                   uint32_t fill);
uint32_t NegativeHugeBits();
uint32_t LnEpsBits();

}  // namespace hotarubi::compiler::sparsedrive

#endif  // HOTARUBI_TOOLS_COMPILER_SPARSEDRIVE_LAYERS_HPP_
