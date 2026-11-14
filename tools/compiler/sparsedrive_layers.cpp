#include "sparsedrive_layers.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
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

std::string_view ModeName(WeightMode mode) {
  switch (mode) {
    case WeightMode::kFp4:
      return "FP4";
    case WeightMode::kW8Hi:
      return "W8_HI";
    case WeightMode::kW8Lo:
      return "W8_LO";
  }
  return {};
}

}  // namespace

std::vector<Band> Bands(uint32_t n, uint32_t mt) {
  std::vector<Band> bands;
  for (uint32_t t = 0; t * mt < n; ++t) {
    bands.push_back({.t = t, .i0 = t * mt, .m = std::min(mt, n - (t * mt))});
  }
  return bands;
}

Rows PlaceRows(DdrImage& image, std::string name, uint32_t n, uint32_t cols,
               uint64_t slack) {
  Require(cols % 32 == 0 && n % 4 == 0,
          std::format("rows {} of {}x{} are not supported", name, n, cols));
  Rows rows{.name = std::move(name), .n = n, .cols = cols, .off = 0};
  rows.off = image.Alloc(rows.name, (n + slack) * rows.Tb());
  return rows;
}

void PutRows(DdrImage& image, const Rows& rows, uint32_t src_rows,
             uint32_t src_cols, const std::vector<uint32_t>& values) {
  Require(src_rows <= rows.n && src_cols <= rows.cols &&
              values.size() == static_cast<size_t>(src_rows) * src_cols,
          std::format("values do not fit rows {}", rows.name));
  std::vector<uint32_t> all(static_cast<size_t>(rows.n) * rows.cols);
  for (uint32_t r = 0; r < src_rows; ++r) {
    for (uint32_t c = 0; c < src_cols; ++c) {
      all[(static_cast<size_t>(r) * rows.cols) + c] =
          values[(static_cast<size_t>(r) * src_cols) + c];
    }
  }
  PutEntries(image, rows.off, all);
}

std::vector<std::byte> EntryBytes(const std::vector<uint32_t>& entries) {
  std::vector<std::byte> bytes;
  bytes.reserve(entries.size() * 4);
  for (const uint32_t entry : entries) {
    for (uint32_t i = 0; i < 4; ++i) {
      bytes.push_back(static_cast<std::byte>(entry >> (8 * i)));
    }
  }
  return bytes;
}

std::vector<std::byte> WordBytes(const std::vector<Word>& words) {
  std::vector<uint32_t> entries;
  entries.reserve(words.size() * 32);
  for (const Word& word : words) {
    entries.insert(entries.end(), word.begin(), word.end());
  }
  return EntryBytes(entries);
}

void PutWords(DdrImage& image, uint64_t offset,
              const std::vector<Word>& words) {
  image.Put(offset, WordBytes(words));
}

void PutEntries(DdrImage& image, uint64_t offset,
                const std::vector<uint32_t>& entries) {
  image.Put(offset, EntryBytes(entries));
}

Lin Lin::Fp4(const Mxfp4Matrix& w, const std::vector<uint32_t>& bias,
             uint32_t n_pad) {
  Lin lin;
  lin.n_ = w.rows;
  lin.rows_ = (w.rows + n_pad - 1) / n_pad * n_pad;
  lin.kb_ = w.cols / 32;
  lin.bias_ = PadValues(bias, lin.rows_);
  lin.planes_.push_back(PackWbuf(CanonicalMxfp4(w, 0, lin.rows_)));
  lin.modes_.push_back(WeightMode::kFp4);
  return lin;
}

Lin Lin::W8(const Mxint8Matrix& w, const std::vector<uint32_t>& bias,
            uint32_t n_pad, uint32_t k_shift) {
  Lin lin;
  lin.n_ = w.rows;
  lin.rows_ = (w.rows + n_pad - 1) / n_pad * n_pad;
  lin.kb_ = w.cols / 32;
  lin.bias_ = PadValues(bias, lin.rows_);
  Mxint8Matrix padded{
      .rows = lin.rows_,
      .cols = w.cols,
      .q = std::vector<int8_t>(static_cast<size_t>(lin.rows_) * w.cols),
      .e = std::vector<int32_t>(static_cast<size_t>(lin.rows_) * lin.kb_)};
  for (uint32_t r = 0; r < w.rows; ++r) {
    for (uint32_t c = 0; c < w.cols; ++c) {
      const int8_t q = w.q[(static_cast<size_t>(r) * w.cols) + c];
      if (c + k_shift < w.cols) {
        padded.q[(static_cast<size_t>(r) * w.cols) + c + k_shift] = q;
      } else {
        Require(q == 0, "shifted W8 columns leave the block");
      }
    }
    for (uint32_t b = 0; b < lin.kb_; ++b) {
      padded.e[(static_cast<size_t>(r) * lin.kb_) + b] =
          w.e[(static_cast<size_t>(r) * lin.kb_) + b];
    }
  }
  const W8Planes planes = SplitW8(padded);
  lin.planes_.push_back(PackWbuf(planes.hi));
  lin.planes_.push_back(PackWbuf(planes.lo));
  lin.modes_ = {WeightMode::kW8Hi, WeightMode::kW8Lo};
  return lin;
}

void Lin::Place(DdrImage& image, std::string_view name) {
  off_.clear();
  for (size_t i = 0; i < planes_.size(); ++i) {
    const uint64_t offset =
        image.Alloc(std::format("{}.w{}", name, i), planes_[i].size());
    image.Put(offset, planes_[i]);
    off_.push_back(offset);
  }
}

Tile Lin::Load(Program& program, std::string_view note) const {
  const Tile w = program.Alloc(Mem::kWbuf, TotalWords());
  for (size_t i = 0; i < off_.size(); ++i) {
    program.Load(
        sparsedrive::Words(off_[i], Mem::kWbuf,
                           w.At(static_cast<uint32_t>(i) * this->Words()),
                           this->Words()),
        std::format("{} ld W{}", note, i));
  }
  return w;
}

std::vector<Piece> Lin::Input(compiler::Place a) const {
  return {{.a = a, .kb_offset = 0, .kb = kb_}};
}

void Lin::Gemm(Program& program, const std::vector<Piece>& pieces, uint32_t row,
               uint32_t a_stride, uint32_t m, compiler::Place c, const Tile& w,
               std::string_view note, bool acc) const {
  for (size_t i = 0; i < pieces.size(); ++i) {
    for (size_t p = 0; p < modes_.size(); ++p) {
      program.Gemm({.a_base = pieces[i].a + row,
                    .a_stride = a_stride,
                    .kb = pieces[i].kb,
                    .m = m,
                    .w_base = w.At((static_cast<uint32_t>(p) * this->Words()) +
                                   pieces[i].kb_offset),
                    .w_stride = kb_,
                    .n8 = Groups(),
                    .c_base = c,
                    .wmode = modes_[p],
                    .acc = acc || i > 0 || p > 0},
                   std::format("{} gemm p{}{}", note, i, ModeName(modes_[p])));
    }
  }
}

uint32_t Pbuf::Add(const std::vector<uint32_t>& values) {
  const auto words = static_cast<uint32_t>((values.size() + 31) / 32);
  const uint32_t at = words_;
  values_.insert(values_.end(), values.begin(), values.end());
  values_.resize(static_cast<size_t>(words_ + words) * 32);
  words_ += words;
  return at;
}

uint32_t Pbuf::Norm(const LayerNorm& norm) {
  const uint32_t at = words_;
  Add(norm.g);
  Add(norm.b);
  return at;
}

void Pbuf::Place(DdrImage& image, std::string_view name) {
  Require(words_ <= kPbufWords,
          std::format("PBUF image {} of {} words is too large", name, words_));
  const std::vector<std::byte> bytes = PackPbuf(values_);
  off_ = image.Alloc(name, bytes.size());
  image.Put(off_, bytes);
}

Tile Pbuf::Load(Program& program, std::string_view note) const {
  const Tile tile = program.Alloc(Mem::kPbuf, words_);
  program.Load(sparsedrive::Words(off_, Mem::kPbuf, tile.At(), words_),
               std::format("{} ld PBUF", note));
  return tile;
}

compiler::Place Params::At(std::string_view key) const {
  const auto found = keys_->find(key);
  Require(found != keys_->end(), std::format("no parameter {}", key));
  return tile_.At(found->second);
}

Lin HeadWeights::Linear(std::string_view module, uint32_t n_pad,
                        uint32_t k_shift) const {
  const std::string bias_name = std::format("{}.bias", module);
  const std::vector<uint32_t> bias = checkpoint_.Contains(bias_name)
                                         ? checkpoint_.Float32Bits(bias_name)
                                         : std::vector<uint32_t>{};
  if (IsW8(module)) {
    return Lin::W8(ReadMxint8(quant_, std::format("{}.weight", module)), bias,
                   n_pad, k_shift);
  }
  Require(k_shift == 0, std::format("{} is MXFP4 with a column shift", module));
  return Lin::Fp4(ReadMxfp4(quant_, module), bias, n_pad);
}

std::vector<Lin> HeadWeights::Attention(std::string_view module, uint32_t embed,
                                        bool scale_q) const {
  const Mxint8Matrix in =
      ReadMxint8(quant_, std::format("{}.in_proj_weight", module));
  const std::vector<uint32_t> bias =
      checkpoint_.Float32Bits(std::format("{}.in_proj_bias", module));
  Require(in.rows == 3 * embed && bias.size() == size_t{3} * embed,
          std::format("{} is not a {}-wide attention", module, embed));
  std::vector<Lin> lins;
  for (uint32_t part = 0; part < 3; ++part) {
    std::vector<int64_t> rows(embed);
    for (uint32_t r = 0; r < embed; ++r) {
      rows[r] = (part * embed) + r;
    }
    Mxint8Matrix w = SelectRows(in, rows);
    std::vector<uint32_t> b = SelectValues(bias, rows, 0);
    if (part == 0 && scale_q) {
      const uint32_t blocks = w.cols / 32;
      for (uint32_t r = 0; r < w.rows; ++r) {
        for (uint32_t kb = 0; kb < blocks; ++kb) {
          const auto first =
              w.q.begin() +
              static_cast<std::ptrdiff_t>((static_cast<size_t>(r) * w.cols) +
                                          (size_t{kb} * 32));
          if (std::any_of(first, first + 32, [](int8_t q) { return q != 0; })) {
            w.e[(static_cast<size_t>(r) * blocks) + kb] -= 3;
          }
        }
      }
      for (uint32_t& value : b) {
        value = exact::Real::Binary32(value).Scaled(-3).ToBinary32();
      }
    }
    lins.push_back(Lin::W8(w, b, 8, 0));
  }
  lins.push_back(Linear(std::format("{}.out_proj", module)));
  return lins;
}

LayerNorm HeadWeights::Norm(std::string_view module) const {
  return {.g = checkpoint_.Float32Bits(std::format("{}.weight", module)),
          .b = checkpoint_.Float32Bits(std::format("{}.bias", module))};
}

std::vector<uint32_t> HeadWeights::Values(std::string_view name) const {
  return checkpoint_.Float32Bits(name);
}

Mxfp4Matrix HeadWeights::Fp4Rows(std::string_view module) const {
  return ReadMxfp4(quant_, module);
}

Mxint8Matrix HeadWeights::W8Rows(std::string_view name) const {
  return ReadMxint8(quant_, name);
}

bool HeadWeights::IsW8(std::string_view module) const {
  return quant_.Contains(std::format("{}.weight.mxint8", module));
}

namespace {

template <typename Matrix, typename Element>
Matrix Select(const Matrix& matrix, const std::vector<int64_t>& rows,
              std::vector<Element> Matrix::* values) {
  const uint32_t blocks = matrix.cols / 32;
  Matrix out;
  out.rows = static_cast<uint32_t>(rows.size());
  out.cols = matrix.cols;
  (out.*values).assign(rows.size() * matrix.cols, Element{});
  out.e.assign(rows.size() * blocks, 0);
  for (size_t i = 0; i < rows.size(); ++i) {
    if (rows[i] < 0) {
      continue;
    }
    const auto r = static_cast<size_t>(rows[i]);
    Require(r < matrix.rows, "row selection is outside the matrix");
    std::copy_n(
        (matrix.*values).begin() + static_cast<std::ptrdiff_t>(r * matrix.cols),
        matrix.cols,
        (out.*values).begin() + static_cast<std::ptrdiff_t>(i * matrix.cols));
    std::copy_n(matrix.e.begin() + static_cast<std::ptrdiff_t>(r * blocks),
                blocks,
                out.e.begin() + static_cast<std::ptrdiff_t>(i * blocks));
  }
  return out;
}

}  // namespace

Mxfp4Matrix SelectRows(const Mxfp4Matrix& matrix,
                       const std::vector<int64_t>& rows) {
  return Select(matrix, rows, &Mxfp4Matrix::code);
}

Mxint8Matrix SelectRows(const Mxint8Matrix& matrix,
                        const std::vector<int64_t>& rows) {
  return Select(matrix, rows, &Mxint8Matrix::q);
}

std::vector<uint32_t> SelectValues(const std::vector<uint32_t>& values,
                                   const std::vector<int64_t>& rows,
                                   uint32_t fill) {
  std::vector<uint32_t> out;
  out.reserve(rows.size());
  for (const int64_t r : rows) {
    out.push_back(r < 0 ? fill : values.at(static_cast<size_t>(r)));
  }
  return out;
}

uint32_t NegativeHugeBits() {
  constexpr int64_t kHalfDigits = 1000000000000000;
  return exact::Real::Binary64((exact::Real::Integer(kHalfDigits) *
                                exact::Real::Integer(kHalfDigits))
                                   .Negated()
                                   .ToBinary64())
      .ToBinary32();
}

uint32_t LnEpsBits() {
  return exact::Real::Binary64(exact::Real::Ratio(1, 100000).ToBinary64())
      .ToBinary32();
}

}  // namespace hotarubi::compiler::sparsedrive
