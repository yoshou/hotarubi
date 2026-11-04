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
#include "exact.hpp"
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

struct W8Plane {
  WeightMode mode;
  std::string_view name;
  uint32_t k_slot;
  uint32_t v_slot;
  bool acc;
};

constexpr std::array<W8Plane, 2> kW8Planes = {{{.mode = WeightMode::kW8Hi,
                                                .name = "hi",
                                                .k_slot = 0,
                                                .v_slot = 2,
                                                .acc = false},
                                               {.mode = WeightMode::kW8Lo,
                                                .name = "lo",
                                                .k_slot = 1,
                                                .v_slot = 3,
                                                .acc = true}}};

constexpr uint32_t kChunkQuads = 8;
constexpr uint32_t kChunkLocalKeys = 128;

uint32_t PlaneWords(uint32_t keys, uint32_t blocks) {
  return keys * blocks / 8;
}

class AttentionEmitter {
 public:
  AttentionEmitter(Program& program, const AttentionPlan& plan,
                   std::string_view note);
  void Emit();

 private:
  void LoadKv(uint32_t h, uint32_t c, bool key);
  void LoadQ(uint32_t h);
  void Prefetch(uint32_t g);
  [[nodiscard]] uint32_t Slot(uint32_t g) const;
  void Scores(uint32_t g);
  void Softmax(uint32_t g);
  void Values(uint32_t g);
  void Store(uint32_t g);

  Program& program_;
  const AttentionPlan& plan_;
  std::string_view note_;
  uint32_t kbq_;
  uint32_t nwq_;
  uint32_t hb_;
  uint32_t nkc_;
  uint32_t kbk_;
  uint32_t kw_;
  uint32_t chunks_;
  uint64_t stride_;
  std::array<uint32_t, 2> qh_{};
  std::array<uint32_t, 2> pp_{};
  std::array<uint32_t, 2> sc_{};
  uint32_t scratch_ = 0;
  std::array<uint32_t, 2> oo_{};
  uint32_t mk_ = 0;
  std::array<uint32_t, 2> kv_{};
  uint32_t scale_ = 0;
};

AttentionEmitter::AttentionEmitter(Program& program, const AttentionPlan& plan,
                                   std::string_view note)
    : program_(program),
      plan_(plan),
      note_(note),
      kbq_(plan.embed / 32),
      nwq_(plan.embed / 8),
      hb_(plan.hd / 32),
      nkc_(plan.chunk_local ? kChunkLocalKeys : plan.keys),
      kbk_(nkc_ / 32),
      kw_(PlaneWords(nkc_, hb_)),
      chunks_(plan.queries / 32),
      stride_(uint64_t{kbq_} * kWordA) {
  Require(
      plan.queries % 32 == 0 && (plan.chunk_local || chunks_ >= 3),
      std::format("attention over {} queries is not supported", plan.queries));
  OnChip a(kAbufWords);
  qh_ = {a.Take(plan.queries / 4 * hb_), a.Take(plan.queries / 4 * hb_)};
  pp_ = {a.Take(8 * kbk_), a.Take(8 * kbk_)};
  OnChip f(kFbufWords);
  sc_ = {f.Take(nkc_), f.Take(nkc_)};
  scratch_ = f.Take(nkc_);
  oo_ = {f.Take(plan.hd), f.Take(plan.hd)};
  if (plan.masked) {
    mk_ = f.Take(nkc_);
  }
  kv_ = {0, 4 * kw_};
  Require(8 * kw_ <= kWbufWords, "attention keys exceed WBUF");
  scale_ = exact::Real::Binary64((exact::Real::Integer(1) /
                                  exact::Sqrt(exact::Real::Integer(plan.hd)))
                                     .ToBinary64())
               .ToBinary32();
}

void AttentionEmitter::Emit() {
  const uint32_t total = kHeads * chunks_;
  LoadKv(0, 0, true);
  LoadKv(0, 0, false);
  LoadQ(0);
  for (uint32_t g = 0; g < total; ++g) {
    if (g >= 1) {
      Softmax(g - 1);
    }
    Scores(g);
    if (g >= 2) {
      Store(g - 2);
    }
    if (g >= 1) {
      Values(g - 1);
    }
    Prefetch(g);
  }
  Softmax(total - 1);
  Store(total - 2);
  Values(total - 1);
  Store(total - 1);
}

void AttentionEmitter::LoadKv(uint32_t h, uint32_t c, bool key) {
  const uint32_t slot =
      plan_.chunk_local ? kv_[((h * chunks_) + c) % 2] : kv_[h % 2];
  const uint64_t chunk =
      plan_.chunk_local ? uint64_t{32} * c * kbq_ * kWordA : 0;
  const uint64_t column = uint64_t{hb_} * h * kWordA;
  program_.Load({.ddr_offset = (key ? plan_.k : plan_.v) + chunk + column,
                 .mem = Mem::kWbuf,
                 .onchip = slot + (key ? 0 : 2 * kw_),
                 .rows = nkc_ / 4,
                 .row_words = hb_,
                 .ddr_stride = static_cast<uint32_t>(stride_),
                 .mode = key ? DmaMode::kSplit8 : DmaMode::kT32Split8},
                std::format("{} ld {} h{} c{}", note_, key ? "K" : "V", h, c));
}

void AttentionEmitter::LoadQ(uint32_t h) {
  program_.Load({.ddr_offset = plan_.q + (uint64_t{hb_} * h * kWordA),
                 .mem = Mem::kAbuf,
                 .onchip = qh_[h % 2],
                 .rows = plan_.queries / 4,
                 .row_words = hb_,
                 .ddr_stride = static_cast<uint32_t>(stride_)},
                std::format("{} ld Q head {}", note_, h));
}

void AttentionEmitter::Prefetch(uint32_t g) {
  if (plan_.chunk_local) {
    const uint32_t next = g + 1;
    if (next < kHeads * chunks_) {
      LoadKv(next / chunks_, next % chunks_, true);
      LoadKv(next / chunks_, next % chunks_, false);
      if (next % chunks_ == 0) {
        LoadQ(next / chunks_);
      }
    }
    return;
  }
  const uint32_t h = g / chunks_;
  const uint32_t c = g % chunks_;
  if (c >= 3 || h + 1 >= kHeads) {
    return;
  }
  if (c < 2) {
    LoadKv(h + 1, 0, c == 0);
  } else {
    LoadQ(h + 1);
  }
}

uint32_t AttentionEmitter::Slot(uint32_t g) const {
  return plan_.chunk_local ? kv_[g % 2] : kv_[(g / chunks_) % 2];
}

void AttentionEmitter::Scores(uint32_t g) {
  const uint32_t h = g / chunks_;
  const uint32_t c = g % chunks_;
  for (const W8Plane& plane : kW8Planes) {
    program_.Gemm({.a_base = qh_[h % 2] + (c * kChunkQuads * hb_),
                   .a_stride = hb_,
                   .kb = hb_,
                   .m = kChunkQuads * 4,
                   .w_base = Slot(g) + (plane.k_slot * kw_),
                   .w_stride = hb_,
                   .n8 = nkc_ / 8,
                   .c_base = sc_[g % 2],
                   .wmode = plane.mode,
                   .acc = plane.acc},
                  std::format("{} S {} h{} c{}", note_, plane.name, h, c));
  }
  if (!plan_.masked) {
    return;
  }
  program_.Load(Words(plan_.mask + (uint64_t{c} * plan_.mask_cs * kWordF),
                      Mem::kFbuf, mk_, kChunkQuads * (nkc_ / 8)),
                std::format("{} ld mask c{}", note_, c));
  program_.Vec({.op = VecOp::kAdd,
                .src = sc_[g % 2],
                .src2 = mk_,
                .dst = sc_[g % 2],
                .rows_q = kChunkQuads,
                .n8 = nkc_ / 8},
               std::format("{} + mask h{} c{}", note_, h, c));
}

void AttentionEmitter::Softmax(uint32_t g) {
  const uint32_t h = g / chunks_;
  const uint32_t c = g % chunks_;
  if (plan_.hd != 64) {
    program_.Vec({.op = VecOp::kScale,
                  .src = sc_[g % 2],
                  .dst = sc_[g % 2],
                  .rows_q = kChunkQuads,
                  .n8 = nkc_ / 8,
                  .imm = scale_},
                 std::format("{} scale h{} c{}", note_, h, c));
  }
  program_.Vec({.op = VecOp::kSoftmax,
                .src = sc_[g % 2],
                .dst = pp_[g % 2],
                .rows_q = kChunkQuads,
                .n8 = nkc_ / 8,
                .aux = plan_.chunk_local ? nkc_ : plan_.valid,
                .q = true,
                .scratch = scratch_},
               std::format("{} softmax+Q h{} c{}", note_, h, c));
}

void AttentionEmitter::Values(uint32_t g) {
  for (const W8Plane& plane : kW8Planes) {
    program_.Gemm({.a_base = pp_[g % 2],
                   .a_stride = kbk_,
                   .kb = kbk_,
                   .m = kChunkQuads * 4,
                   .w_base = Slot(g) + (plane.v_slot * kw_),
                   .w_stride = kbk_,
                   .n8 = plan_.hd / 8,
                   .c_base = oo_[g % 2],
                   .wmode = plane.mode,
                   .acc = plane.acc},
                  std::format("{} PV {} h{} c{}", note_, plane.name,
                              g / chunks_, g % chunks_));
  }
}

void AttentionEmitter::Store(uint32_t g) {
  const uint32_t h = g / chunks_;
  const uint32_t c = g % chunks_;
  program_.Store({.ddr_offset = plan_.o + (((uint64_t{c} * kChunkQuads * nwq_) +
                                            (uint64_t{h} * (plan_.hd / 8))) *
                                           kWordF),
                  .mem = Mem::kFbuf,
                  .onchip = oo_[g % 2],
                  .rows = kChunkQuads,
                  .row_words = plan_.hd / 8,
                  .ddr_stride = static_cast<uint32_t>(uint64_t{nwq_} * kWordF)},
                 std::format("{} st O h{} c{}", note_, h, c));
}

}  // namespace

std::vector<Tile> Tiles(uint32_t n, uint32_t mt) {
  std::vector<Tile> tiles;
  for (uint32_t t = 0; t * mt < n; ++t) {
    tiles.push_back({.t = t, .i0 = t * mt, .m = std::min(mt, n - (t * mt))});
  }
  return tiles;
}

uint32_t OnChip::Take(uint32_t n) {
  Require(used_ + n <= size_,
          std::format("on-chip area of {} words cannot take {} at {}", size_, n,
                      used_));
  const uint32_t at = used_;
  used_ += n;
  return at;
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

void Lin::Load(Program& program, uint32_t slot, std::string_view note) const {
  Require(slot + TotalWords() <= kWbufWords,
          std::format("{}: weights of {} words do not fit at {}", note,
                      TotalWords(), slot));
  for (size_t i = 0; i < off_.size(); ++i) {
    program.Load(
        sparsedrive::Words(off_[i], Mem::kWbuf,
                           slot + (static_cast<uint32_t>(i) * this->Words()),
                           this->Words()),
        std::format("{} ld W{}", note, i));
  }
}

void Lin::Gemm(Program& program, uint32_t a_base, uint32_t a_stride, uint32_t m,
               uint32_t c_base, uint32_t slot, std::string_view note,
               const std::vector<Piece>& pieces, bool acc) const {
  const std::vector<Piece> whole = {{.a_offset = 0, .kb_offset = 0, .kb = kb_}};
  const std::vector<Piece>& parts = pieces.empty() ? whole : pieces;
  for (size_t i = 0; i < parts.size(); ++i) {
    for (size_t p = 0; p < modes_.size(); ++p) {
      program.Gemm(
          {.a_base = a_base + parts[i].a_offset,
           .a_stride = a_stride,
           .kb = parts[i].kb,
           .m = m,
           .w_base = slot + (static_cast<uint32_t>(p) * this->Words()) +
                     parts[i].kb_offset,
           .w_stride = kb_,
           .n8 = Groups(),
           .c_base = c_base,
           .wmode = modes_[p],
           .acc = acc || i > 0 || p > 0},
          std::format("{} gemm p{}{}", note, i, ModeName(modes_[p])));
    }
  }
}

void EmitAttention(Program& program, const AttentionPlan& plan,
                   std::string_view note) {
  AttentionEmitter(program, plan, note).Emit();
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

void Pbuf::Load(Program& program, std::string_view note) const {
  program.Load(sparsedrive::Words(off_, Mem::kPbuf, 0, words_),
               std::format("{} ld PBUF", note));
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
