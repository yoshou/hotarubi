#include "kernels.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hotarubi/exact.hpp"
#include "hotarubi/formats.hpp"
#include "hotarubi/require.hpp"
#include "isa.hpp"
#include "program.hpp"

namespace hotarubi::compiler {
namespace {

constexpr uint64_t kWordA = kWordAbufBytes;
constexpr uint64_t kWordF = kWordFbufBytes;
constexpr uint32_t kKeptMaskWords = kMemDepth[2] / 4;

struct W8Plane {
  WeightMode mode;
  std::string_view name;
  uint32_t index;
  bool acc;
};

constexpr std::array<W8Plane, 2> kW8Planes = {
    {{.mode = WeightMode::kW8Hi, .name = "hi", .index = 0, .acc = false},
     {.mode = WeightMode::kW8Lo, .name = "lo", .index = 1, .acc = true}}};

bool MultiPass(VecOp op) {
  return op == VecOp::kLayerNorm || op == VecOp::kRmsNorm ||
         op == VecOp::kAct || op == VecOp::kActMul || op == VecOp::kSoftmax ||
         op == VecOp::kRope;
}

uint32_t InverseSqrtBits(uint32_t value) {
  return exact::Real::Binary64((exact::Real::Integer(1) /
                                exact::Sqrt(exact::Real::Integer(value)))
                                   .ToBinary64())
      .ToBinary32();
}

class AttentionEmitter {
 public:
  AttentionEmitter(Program& program, const Attention& plan,
                   std::string_view note)
      : program_(program),
        plan_(plan),
        note_(note),
        hb_(plan.hd / kBlock),
        kbk_(plan.keys / kBlock),
        chunks_(plan.rows / (4 * plan.chunk_quads)),
        score_words_(plan.chunk_quads * (plan.keys / 8)) {
    Require(plan.chunk_quads % 8 == 0 &&
                plan.rows % (4 * plan.chunk_quads) == 0 &&
                plan.keys % kBlock == 0 && plan.hd % kBlock == 0 &&
                plan.kv_heads >= 1 && plan.heads % plan.kv_heads == 0 &&
                (!plan.chunk_local || plan.kv_heads == plan.heads) &&
                !plan.valid.empty() && plan.valid.front().quad == 0,
            std::format("attention of {} rows in chunks of {} quads over {} "
                        "keys with {} heads on {} is not supported",
                        plan.rows, plan.chunk_quads, plan.keys, plan.heads,
                        plan.kv_heads));
    scratch_ = program.Alloc(Mem::kFbuf, score_words_);
    if (plan.mask && chunks_ * score_words_ <= kKeptMaskWords) {
      masks_.resize(chunks_);
    }
  }

  void Emit() {
    const uint32_t share = plan_.heads / plan_.kv_heads;
    for (uint32_t h = 0; h < plan_.heads; ++h) {
      const Place q = Queries(h);
      for (uint32_t c = 0; c < chunks_; ++c) {
        if (plan_.chunk_local || (c == 0 && h % share == 0)) {
          LoadWeights(h / share, c);
        }
        const Tile s = Scores(q, h, c);
        const Tile p = Softmax(s, h, c);
        Values(p, h, c);
      }
    }
  }

 private:
  [[nodiscard]] uint32_t PlaneWords() const { return plan_.keys * hb_ / 8; }

  Place Queries(uint32_t h) {
    const uint32_t words = plan_.rows / 4 * hb_;
    if (plan_.queries) {
      return plan_.queries->At(h * words);
    }
    const Tile q = program_.Alloc(Mem::kAbuf, words);
    program_.Load(
        {.ddr_offset = plan_.q + (uint64_t{hb_} * h * kWordA),
         .mem = Mem::kAbuf,
         .onchip = q.At(),
         .rows = plan_.rows / 4,
         .row_words = hb_,
         .ddr_stride = static_cast<uint32_t>(plan_.embed / kBlock * kWordA)},
        std::format("{} ld Q h{}", note_, h));
    return q.At();
  }

  void LoadWeights(uint32_t h, uint32_t c) {
    const uint64_t stride = plan_.kv_blocks * kWordA;
    const uint64_t chunk =
        plan_.chunk_local ? uint64_t{c} * (plan_.keys / 4) * stride : 0;
    const uint64_t offset = chunk + (uint64_t{hb_} * h * kWordA);
    k_ = program_.Alloc(Mem::kWbuf, 2 * PlaneWords());
    v_ = program_.Alloc(Mem::kWbuf, 2 * PlaneWords());
    for (const bool key : {true, false}) {
      program_.Load(
          {.ddr_offset = (key ? plan_.k : plan_.v) + offset,
           .mem = Mem::kWbuf,
           .onchip = key ? k_.At() : v_.At(),
           .rows = plan_.keys / 4,
           .row_words = hb_,
           .ddr_stride = static_cast<uint32_t>(stride),
           .kind = key ? DmaKind::kSplit8 : DmaKind::kT32Split8},
          std::format("{} ld {} h{} c{}", note_, key ? "K" : "V", h, c));
    }
  }

  Tile Mask(uint32_t c) {
    if (!masks_.empty()) {
      const std::optional<Tile>& kept = masks_[c];
      if (kept.has_value()) {
        return *kept;
      }
    }
    const Tile mask = program_.Alloc(Mem::kFbuf, score_words_);
    program_.Load({.ddr_offset = plan_.mask.value_or(0) +
                                 (uint64_t{c} * plan_.mask_stride * kWordF),
                   .mem = Mem::kFbuf,
                   .onchip = mask.At(),
                   .rows = 1,
                   .row_words = score_words_},
                  std::format("{} ld mask c{}", note_, c));
    if (!masks_.empty()) {
      masks_[c] = mask;
    }
    return mask;
  }

  Tile Scores(Place q, uint32_t h, uint32_t c) {
    const Tile s = program_.Alloc(Mem::kFbuf, score_words_);
    for (const W8Plane& plane : kW8Planes) {
      program_.Gemm({.a_base = q + (c * plan_.chunk_quads * hb_),
                     .a_stride = hb_,
                     .kb = hb_,
                     .m = plan_.chunk_quads * 4,
                     .w_base = k_.At(plane.index * PlaneWords()),
                     .w_stride = hb_,
                     .n8 = plan_.keys / 8,
                     .c_base = s.At(),
                     .wmode = plane.mode,
                     .acc = plane.acc},
                    std::format("{} S {} h{} c{}", note_, plane.name, h, c));
    }
    if (plan_.mask) {
      program_.Vec({.op = VecOp::kAdd,
                    .src = s.At(),
                    .src2 = Mask(c).At(),
                    .dst = s.At(),
                    .rows_q = plan_.chunk_quads,
                    .n8 = plan_.keys / 8},
                   std::format("{} + mask h{} c{}", note_, h, c));
    }
    if (plan_.scale) {
      program_.Vec({.op = VecOp::kScale,
                    .src = s.At(),
                    .dst = s.At(),
                    .rows_q = plan_.chunk_quads,
                    .n8 = plan_.keys / 8,
                    .imm = InverseSqrtBits(plan_.hd)},
                   std::format("{} scale h{} c{}", note_, h, c));
    }
    return s;
  }

  Tile Softmax(const Tile& s, uint32_t h, uint32_t c) {
    const Tile p = program_.Alloc(Mem::kAbuf, plan_.chunk_quads * kbk_);
    const uint32_t first = c * plan_.chunk_quads;
    const uint32_t last = first + plan_.chunk_quads;
    for (size_t i = 0; i < plan_.valid.size(); ++i) {
      const uint32_t begin = std::max(first, plan_.valid[i].quad);
      const uint32_t end = i + 1 < plan_.valid.size()
                               ? std::min(last, plan_.valid[i + 1].quad)
                               : last;
      if (begin >= end) {
        continue;
      }
      program_.Vec({.op = VecOp::kSoftmax,
                    .src = s.At((begin - first) * (plan_.keys / 8)),
                    .dst = p.At((begin - first) * kbk_),
                    .rows_q = end - begin,
                    .n8 = plan_.keys / 8,
                    .aux = plan_.valid[i].keys,
                    .q = true,
                    .scratch = scratch_.At()},
                   std::format("{} softmax+Q h{} c{} keys<{}", note_, h, c,
                               plan_.valid[i].keys));
    }
    return p;
  }

  void Values(const Tile& p, uint32_t h, uint32_t c) {
    const uint32_t words = plan_.hd / 8;
    const uint32_t row = plan_.embed / 8;
    const Tile o = program_.Alloc(Mem::kFbuf, plan_.chunk_quads * words);
    for (const W8Plane& plane : kW8Planes) {
      program_.Gemm({.a_base = p.At(),
                     .a_stride = kbk_,
                     .kb = kbk_,
                     .m = plan_.chunk_quads * 4,
                     .w_base = v_.At(plane.index * PlaneWords()),
                     .w_stride = kbk_,
                     .n8 = words,
                     .c_base = o.At(),
                     .wmode = plane.mode,
                     .acc = plane.acc},
                    std::format("{} PV {} h{} c{}", note_, plane.name, h, c));
    }
    program_.Store(
        {.ddr_offset = plan_.o + (((uint64_t{c} * plan_.chunk_quads * row) +
                                   (uint64_t{h} * words)) *
                                  kWordF),
         .mem = Mem::kFbuf,
         .onchip = o.At(),
         .rows = plan_.chunk_quads,
         .row_words = words,
         .ddr_stride = static_cast<uint32_t>(row * kWordF)},
        std::format("{} st O h{} c{}", note_, h, c));
  }

  Program& program_;
  const Attention& plan_;
  std::string_view note_;
  uint32_t hb_;
  uint32_t kbk_;
  uint32_t chunks_;
  uint32_t score_words_;
  Tile scratch_;
  Tile k_;
  Tile v_;
  std::vector<std::optional<Tile>> masks_;
};

}  // namespace

void EmitAttention(Program& program, const Attention& attention,
                   std::string_view note) {
  AttentionEmitter(program, attention, note).Emit();
}

void EmitQuantize(Program& program, const Quantize& quantize,
                  std::string_view note, const QuantizeHook& before) {
  Require(quantize.words % 4 == 0 && quantize.tile_quads >= 1 &&
              quantize.quads >= 1,
          std::format("{}: rows of {} words are not quantized", note,
                      quantize.words));
  std::optional<Tile> scratch;
  if (MultiPass(quantize.op)) {
    scratch = program.Alloc(Mem::kFbuf, quantize.tile_quads * quantize.words);
  }
  for (uint32_t quad = 0; quad < quantize.quads; quad += quantize.tile_quads) {
    const uint32_t count = std::min(quantize.tile_quads, quantize.quads - quad);
    const uint32_t tile = quad / quantize.tile_quads;
    const Tile x = program.Alloc(Mem::kFbuf, count * quantize.words);
    const uint64_t ddr =
        quantize.x + (uint64_t{quad} * quantize.words * kWordF);
    if (quantize.tokens) {
      program.Load({.ddr_offset = ddr,
                    .mem = Mem::kFbuf,
                    .onchip = x.At(),
                    .rows = 1,
                    .row_words = 4 * count,
                    .kind = DmaKind::kRow,
                    .tok = quantize.words / 4},
                   std::format("{} ld t{}", note, tile));
    } else {
      program.Load(
          {.ddr_offset = ddr,
           .mem = Mem::kFbuf,
           .onchip = x.At(),
           .rows = count,
           .row_words = quantize.words,
           .ddr_stride = static_cast<uint32_t>(quantize.words * kWordF)},
          std::format("{} ld t{}", note, tile));
    }
    if (before) {
      before(x, quad, count);
    }
    program.Vec({.op = quantize.op,
                 .src = x.At(),
                 .dst = quantize.dst + (quad * (quantize.words / 4)),
                 .rows_q = count,
                 .n8 = quantize.words,
                 .p_base = quantize.p_base,
                 .aux = quantize.aux,
                 .imm = quantize.imm,
                 .q = true,
                 .scratch = scratch ? std::optional<Place>(scratch->At())
                                    : std::nullopt},
                std::format("{} Q t{}", note, tile));
  }
}

LinearEmitter::LinearEmitter(Program& program, Linear linear,
                             std::string_view note)
    : program_(program),
      linear_(std::move(linear)),
      note_(note),
      weights_(linear_.pieces.size() * linear_.planes.size()) {
  Require(!linear_.planes.empty() && !linear_.pieces.empty() &&
              linear_.chunk_quads >= 8 && linear_.chunk_quads % 8 == 0 &&
              linear_.quads >= 8 && linear_.quads % 8 == 0 &&
              linear_.tile_groups >= 1 && linear_.groups >= 1,
          std::format("{}: a product of {} quads in chunks of {} by {} "
                      "column groups in tiles of {} is not supported",
                      note, linear_.quads, linear_.chunk_quads, linear_.groups,
                      linear_.tile_groups));
}

uint32_t LinearEmitter::Tiles() const {
  return (linear_.groups + linear_.tile_groups - 1) / linear_.tile_groups;
}

uint32_t LinearEmitter::Chunks() const {
  return (linear_.quads + linear_.chunk_quads - 1) / linear_.chunk_quads;
}

LinearTile LinearEmitter::Emit(uint32_t tile, uint32_t chunk,
                               const std::optional<Tile>& onto) {
  if (tile_ != tile) {
    std::ranges::fill(weights_, std::nullopt);
    tile_ = tile;
  }
  const uint32_t first = tile * linear_.tile_groups;
  const uint32_t groups = std::min(linear_.tile_groups, linear_.groups - first);
  const uint32_t quad = chunk * linear_.chunk_quads;
  const uint32_t quads = std::min(linear_.chunk_quads, linear_.quads - quad);
  const LinearTile out{
      .c = onto ? *onto : program_.Alloc(Mem::kFbuf, quads * groups),
      .tile = tile,
      .chunk = chunk,
      .first = first,
      .groups = groups,
      .quad = quad,
      .quads = quads};
  for (size_t p = 0; p < linear_.pieces.size(); ++p) {
    const LinearPiece& piece = linear_.pieces[p];
    for (size_t l = 0; l < linear_.planes.size(); ++l) {
      const LinearPlane& plane = linear_.planes[l];
      std::optional<Tile>& w = weights_[(p * linear_.planes.size()) + l];
      if (!w) {
        w = program_.Alloc(Mem::kWbuf, out.groups * piece.kb);
        program_.Load(
            {.ddr_offset =
                 plane.w +
                 (((uint64_t{out.first} * linear_.kb_total) + piece.kb_offset) *
                  kWordA),
             .mem = Mem::kWbuf,
             .onchip = w->At(),
             .rows = out.groups,
             .row_words = piece.kb,
             .ddr_stride = static_cast<uint32_t>(linear_.kb_total * kWordA)},
            std::format("{} ld W t{} p{} l{}", note_, tile, p, l));
      }
      program_.Gemm(
          {.a_base = piece.a + (out.quad * piece.a_stride),
           .a_stride = piece.a_stride,
           .kb = piece.kb,
           .m = out.quads * 4,
           .w_base = w->At(),
           .w_stride = piece.kb,
           .n8 = out.groups,
           .c_base = out.c.At(),
           .wmode = plane.mode,
           .acc = onto.has_value() || p > 0 || l > 0},
          std::format("{} gemm t{} c{} p{} l{}", note_, tile, chunk, p, l));
    }
  }
  return out;
}

void EmitLinear(Program& program, const Linear& linear, std::string_view note,
                const std::function<void(const LinearTile&)>& post) {
  LinearEmitter emitter(program, linear, note);
  for (uint32_t tile = 0; tile < emitter.Tiles(); ++tile) {
    for (uint32_t chunk = 0; chunk < emitter.Chunks(); ++chunk) {
      post(emitter.Emit(tile, chunk));
    }
  }
}

}  // namespace hotarubi::compiler
