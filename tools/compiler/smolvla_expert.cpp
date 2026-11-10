#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ddr_image.hpp"
#include "hotarubi/exact.hpp"
#include "hotarubi/safetensors.hpp"
#include "isa.hpp"
#include "program.hpp"
#include "smolvla_parts.hpp"
#include "tensors.hpp"
#include "weights.hpp"

namespace hotarubi::compiler::smolvla {
namespace {

constexpr uint32_t kLayers = 16;
constexpr uint32_t kWidth = 720;
constexpr uint32_t kPadded = 736;
constexpr uint32_t kHidden = 2048;
constexpr uint32_t kActionWidth = 32;
constexpr uint32_t kKb = kPadded / kBlock;
constexpr uint32_t kKbQuery = 30;
constexpr uint32_t kKbHidden = kHidden / kBlock;
constexpr uint32_t kWords = kPadded / 8;
constexpr uint32_t kQueryWords = 120;
constexpr uint32_t kActionWords = kActionWidth / 8;
constexpr uint32_t kQuads = kSuffix / 4;
constexpr uint32_t kHeadWords = 8;
constexpr uint32_t kGateGroups = 16;
constexpr uint32_t kGateTiles = kHidden / (8 * kGateGroups);
constexpr uint32_t kGateBlocks = kGateGroups / 4;
constexpr uint32_t kPbLn1 = 0;
constexpr uint32_t kPbLn2 = 23;
constexpr uint32_t kPbCosSelf = 48;
constexpr uint32_t kPbCosCross = 176;
constexpr uint32_t kPbIn = 304;
constexpr uint32_t kPbMlpIn = 327;
constexpr uint32_t kPbMlpOut = 350;
constexpr uint32_t kPbFinal = 373;
constexpr uint32_t kPbOut = 396;
constexpr uint32_t kPbEnd = 397;
constexpr uint32_t kRopeWords = 256;
constexpr std::array<uint32_t, 4> kTileGroups = {28, 28, 28, 8};
constexpr std::array<uint32_t, 4> kTileFirst = {0, 28, 56, 84};
constexpr uint32_t kWeightPlane = 1024;
constexpr uint32_t kKvPlane = 400;
constexpr uint32_t kKvOut = 1024;
constexpr uint32_t kFrequencies = kWidth / 2;
constexpr int64_t kMinPeriodPerMille = 4;
constexpr int64_t kMaxPeriod = 4;
constexpr int64_t kMaskMagnitude = 1000000000000000;
constexpr size_t kO = 3;
constexpr size_t kGate = 4;
constexpr size_t kUp = 5;
constexpr size_t kDown = 6;
constexpr size_t kHeadIn = 0;
constexpr size_t kHeadMlpInAction = 1;
constexpr size_t kHeadMlpInTime = 2;
constexpr size_t kHeadMlpOut = 3;
constexpr size_t kHeadOut = 4;
constexpr std::array<std::string_view, 7> kLinearModules = {
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj",    "mlp.up_proj",
    "mlp.down_proj"};
constexpr std::array<std::string_view, 7> kLinearTags = {
    "q", "k", "v", "o", "gate", "up", "down"};
constexpr std::string_view kExpertModel = "model.vlm_with_expert.lm_expert";

using Hook = std::function<void(uint32_t)>;

bool IsCross(uint32_t layer) { return layer % 2 == 1; }

uint64_t CacheRow(uint32_t prefix_slots) {
  return uint64_t{prefix_slots / 4} * kKvBlocks;
}

uint32_t ScoreWords(const Dims& dims) {
  return kQuads * (dims.Tokens() + kSuffix) / 8;
}

struct Plane {
  WeightMode mode = WeightMode::kFp4;
  uint64_t offset = 0;
};

struct Piece {
  uint32_t a_base = 0;
  uint32_t a_stride = 0;
  uint32_t kb = 0;
  uint32_t kb_offset = 0;
};

struct Tiles {
  std::vector<Plane> planes;
  uint32_t kb_total = 0;
  std::vector<Piece> pieces;
  std::vector<uint32_t> groups;
  std::vector<uint32_t> first;
  uint32_t c_base = 0;
  bool accumulate = false;
};

exact::Real Rn64(const exact::Real& value) {
  return exact::Real::Binary64(value.ToBinary64());
}

std::vector<uint32_t> TimeEmbedding(uint32_t t) {
  const exact::Real one = exact::Real::Integer(1);
  const exact::Real step = Rn64(exact::Real::Ratio(1, kFrequencies - 1));
  const exact::Real min_period =
      Rn64(exact::Real::Ratio(kMinPeriodPerMille, 1000));
  const exact::Real ratio = Rn64(exact::Real::Integer(kMaxPeriod) / min_period);
  const exact::Real pi = Rn64(exact::Pi());
  const exact::Real time = exact::Real::Binary32(t);
  std::vector<uint32_t> out(kWidth);
  for (uint32_t i = 0; i < kFrequencies; ++i) {
    const exact::Real fraction =
        i + 1 == kFrequencies ? one : Rn64(exact::Real::Integer(i) * step);
    const exact::Real period =
        Rn64(min_period * Rn64(exact::Pow(ratio, fraction)));
    const exact::Real scale = Rn64(Rn64(one / period).Scaled(1) * pi);
    const exact::Real x = Rn64(scale * time);
    out[i] = Fp32OfFp64(exact::Sin(x));
    out[kFrequencies + i] = Fp32OfFp64(exact::Cos(x));
  }
  return out;
}

std::vector<std::byte> TimeInput(uint32_t step, uint32_t steps) {
  const exact::Real dt = Rn64(exact::Real::Ratio(-1, steps));
  const exact::Real t =
      Rn64(exact::Real::Integer(1) + Rn64(exact::Real::Integer(step) * dt));
  const std::vector<uint32_t> row =
      PadValues(TimeEmbedding(t.ToBinary32()), kPadded);
  Matrix input{.rows = kSuffix, .cols = kPadded, .bits = {}};
  for (uint32_t r = 0; r < kSuffix; ++r) {
    input.bits.insert(input.bits.end(), row.begin(), row.end());
  }
  return PackAbuf(EncodeMxint8(input));
}

std::vector<std::byte> SelfMask(const Dims& dims) {
  const uint32_t prefix_slots = dims.Tokens();
  const uint32_t keys = prefix_slots + kSuffix;
  const uint32_t negative = Fp32OfFp64((exact::Real::Integer(kMaskMagnitude) *
                                        exact::Real::Integer(kMaskMagnitude))
                                           .Negated());
  Matrix mask{.rows = kSuffix,
              .cols = keys,
              .bits = std::vector<uint32_t>(static_cast<size_t>(kSuffix) * keys,
                                            negative)};
  for (uint32_t q = 0; q < kSuffix; ++q) {
    for (uint32_t key = 0; key < keys; ++key) {
      if (key < dims.Prefix() ||
          (key >= prefix_slots && key <= prefix_slots + q)) {
        mask.bits[(static_cast<size_t>(q) * keys) + key] = 0;
      }
    }
  }
  return PackFbuf(mask);
}

std::vector<std::byte> ExpertRope(const Dims& dims) {
  std::vector<uint32_t> self(kSuffix);
  std::vector<uint32_t> cross(kSuffix);
  for (uint32_t i = 0; i < kSuffix; ++i) {
    self[i] = dims.Prefix() + i;
    cross[i] = i;
  }
  std::vector<std::byte> out = RopeTables(self);
  const std::vector<std::byte> tail = RopeTables(cross);
  out.insert(out.end(), tail.begin(), tail.end());
  return out;
}

void PlaceHeads(DdrImage& image, const SafeTensors& model, Expert& out) {
  const Matrix in = ReadMatrix(model, "model.action_in_proj.weight");
  const Matrix mlp_in = ReadMatrix(model, "model.action_time_mlp_in.weight");
  const Matrix mlp_out = ReadMatrix(model, "model.action_time_mlp_out.weight");
  const Matrix action_out = ReadMatrix(model, "model.action_out_proj.weight");
  out.heads[kHeadIn] = PlaceW8(image, "ex.in", in, kPadded);
  out.heads[kHeadMlpInAction] =
      PlaceW8(image, "ex.mlp_in_a", Columns(mlp_in, 0, kWidth), kPadded);
  out.heads[kHeadMlpInTime] = PlaceW8(
      image, "ex.mlp_in_t", Columns(mlp_in, kWidth, 2 * kWidth), kPadded);
  out.heads[kHeadMlpOut] = PlaceW8(image, "ex.mlp_out", mlp_out, kPadded);
  out.heads[kHeadOut] = PlaceW8(image, "ex.out", action_out, action_out.rows);
  std::vector<uint32_t> pbuf;
  for (const std::string& name :
       {std::string("model.action_in_proj.bias"),
        std::string("model.action_time_mlp_in.bias"),
        std::string("model.action_time_mlp_out.bias"),
        std::format("{}.norm.weight", kExpertModel)}) {
    const std::vector<uint32_t> values =
        PadValues(model.Float32Bits(name), kPadded);
    pbuf.insert(pbuf.end(), values.begin(), values.end());
  }
  const std::vector<uint32_t> bias =
      model.Float32Bits("model.action_out_proj.bias");
  pbuf.insert(pbuf.end(), bias.begin(), bias.end());
  out.pb_heads = Place(image, "ex.pb_heads", PackPbuf(pbuf));
}

void PlaceLayer(DdrImage& image, const SafeTensors& quant,
                const SafeTensors& model, uint32_t layer, DecoderLayer& out) {
  const std::string prefix = std::format("{}.layers.{}", kExpertModel, layer);
  const std::string tag = std::format("E{}", layer);
  for (size_t i = 0; i < kLinearModules.size(); ++i) {
    const Mxfp4Matrix raw =
        ReadMxfp4(quant, std::format("{}.{}", prefix, kLinearModules[i]));
    const uint32_t rows = i == kO || i == kDown ? kPadded : raw.rows;
    out.weights[i] =
        Place(image, std::format("{}.w{}", tag, kLinearTags[i]),
              PackWbuf(CanonicalMxfp4(raw, i == 0 ? -3 : 0, rows)));
  }
  std::vector<uint32_t> pbuf = PadValues(
      model.Float32Bits(std::format("{}.input_layernorm.weight", prefix)),
      kPadded);
  const std::vector<uint32_t> post =
      PadValues(model.Float32Bits(
                    std::format("{}.post_attention_layernorm.weight", prefix)),
                kPadded);
  pbuf.insert(pbuf.end(), post.begin(), post.end());
  out.pbuf = Place(image, std::format("{}.pbuf", tag), PackPbuf(pbuf));
}

void LinearTiles(Program& program, const Tiles& tiles, std::string_view note,
                 const Hook& after_first, const Hook& post) {
  struct Step {
    uint32_t nt = 0;
    uint32_t p = 0;
    uint32_t pl = 0;
  };
  std::vector<Step> steps;
  for (uint32_t nt = 0; nt < tiles.groups.size(); ++nt) {
    for (uint32_t p = 0; p < tiles.pieces.size(); ++p) {
      for (uint32_t pl = 0; pl < tiles.planes.size(); ++pl) {
        steps.push_back({.nt = nt, .p = p, .pl = pl});
      }
    }
  }
  const auto load_w = [&](size_t wt) {
    const Step& step = steps[wt];
    const Piece& piece = tiles.pieces[step.p];
    Require(tiles.groups[step.nt] * piece.kb <= kWeightPlane,
            "weight tile exceeds its WBUF plane");
    Ld(program,
       tiles.planes[step.pl].offset +
           (((uint64_t{tiles.first[step.nt]} * tiles.kb_total) +
             piece.kb_offset) *
            kWordA),
       Mem::kWbuf, static_cast<uint32_t>(wt % 2) * kWeightPlane,
       tiles.groups[step.nt], piece.kb, tiles.kb_total * kWordA,
       std::format("{} ld W nt{} p{} pl{}", note, step.nt, step.p, step.pl));
  };
  load_w(0);
  for (size_t wt = 0; wt < steps.size(); ++wt) {
    const Step& step = steps[wt];
    const Piece& piece = tiles.pieces[step.p];
    const bool first = step.p == 0 && step.pl == 0;
    if (first && step.nt >= 1 && post) {
      post(step.nt - 1);
    }
    program.Gemm(
        {.a_base = piece.a_base,
         .a_stride = piece.a_stride,
         .kb = piece.kb,
         .m = kSuffix,
         .w_base = static_cast<uint32_t>(wt % 2) * kWeightPlane,
         .w_stride = piece.kb,
         .n8 = tiles.groups[step.nt],
         .c_base = tiles.c_base + (kQuads * tiles.first[step.nt]),
         .wmode = tiles.planes[step.pl].mode,
         .acc = tiles.accumulate || !first},
        std::format("{} gemm nt{} p{} pl{}", note, step.nt, step.p, step.pl));
    if (first && after_first) {
      after_first(step.nt);
    }
    if (wt + 1 < steps.size()) {
      load_w(wt + 1);
    }
  }
  if (post) {
    post(static_cast<uint32_t>(tiles.groups.size() - 1));
  }
}

Tiles PaddedTiles(std::vector<Plane> planes, uint32_t kb_total,
                  std::vector<Piece> pieces, bool accumulate) {
  return {.planes = std::move(planes),
          .kb_total = kb_total,
          .pieces = std::move(pieces),
          .groups = {kTileGroups.begin(), kTileGroups.end()},
          .first = {kTileFirst.begin(), kTileFirst.end()},
          .c_base = 0,
          .accumulate = accumulate};
}

std::vector<Piece> TiledPieces(uint32_t a_base) {
  std::vector<Piece> pieces;
  for (size_t i = 0; i < kTileGroups.size(); ++i) {
    const uint32_t blocks = kTileGroups[i] / 4;
    pieces.push_back({.a_base = a_base + (kQuads * (kTileFirst[i] / 4)),
                      .a_stride = blocks,
                      .kb = blocks,
                      .kb_offset = kTileFirst[i] / 4});
  }
  return pieces;
}

std::vector<Plane> W8Planes(const W8Region& region) {
  return {{.mode = WeightMode::kW8Hi, .offset = region.hi},
          {.mode = WeightMode::kW8Lo, .offset = region.lo}};
}

void RmsNormQ(Program& program, uint64_t x, uint32_t pb, uint32_t dst,
              std::string_view note) {
  constexpr uint32_t kScratch = 1472;
  Ld(program, x, Mem::kFbuf, 0, kQuads, kWords, kWords * kWordF,
     std::format("{} ld x", note));
  program.Vec({.op = VecOp::kRmsNorm,
               .src = 0,
               .dst = dst,
               .rows_q = kQuads,
               .n8 = kWords,
               .p_base = pb,
               .aux = kWidth,
               .imm = RmsEps(),
               .q = true,
               .scratch = kScratch},
              std::format("{} RMSNorm+Q", note));
}

class Qkv {
 public:
  Qkv(Program& program, const DecoderLayer& layer, const Lm& lm, uint32_t l,
      uint32_t prefix_slots, std::string_view note)
      : program_(program),
        layer_(layer),
        k_cache_(lm.k[l]),
        v_cache_(lm.v[l]),
        cross_(IsCross(l)),
        row0_(CacheRow(prefix_slots)),
        note_(note) {
    for (size_t linear = 0; linear < (cross_ ? 1U : 3U); ++linear) {
      const uint32_t count = linear == 0 ? kQueryHeads : kKvHeads;
      for (uint32_t h = 0; h < count; ++h) {
        heads_.push_back({linear, h});
      }
    }
  }

  void Emit(uint32_t a_base, uint32_t q_abuf) {
    LoadWeight(0);
    for (size_t j = 0; j < heads_.size(); ++j) {
      if (j >= 1) {
        PostVec(j - 1, q_abuf);
      }
      program_.Gemm({.a_base = a_base,
                     .a_stride = kKb,
                     .kb = kKb,
                     .m = kSuffix,
                     .w_base = kWt[j % 2],
                     .w_stride = kKb,
                     .n8 = kHeadWords,
                     .c_base = kCb[j % 2]},
                    std::format("{} gemm {}", note_, Name(j)));
      if (j + 1 < heads_.size()) {
        LoadWeight(j + 1);
      }
      if (j >= 1) {
        PostStore(j - 1);
      }
    }
    PostVec(heads_.size() - 1, q_abuf);
    PostStore(heads_.size() - 1);
  }

 private:
  struct Head {
    size_t linear = 0;
    uint32_t index = 0;
  };

  static constexpr std::array<uint32_t, 2> kWt = {0, 184};
  static constexpr std::array<uint32_t, 2> kCb = {0, 128};
  static constexpr uint32_t kScratch = 256;
  static constexpr std::array<uint32_t, 2> kKt = {1792, 1824};
  static constexpr std::array<uint32_t, 2> kVt = {1856, 1888};

  [[nodiscard]] std::string Name(size_t j) const {
    return std::format("{}{}", kLinearTags[heads_[j].linear], heads_[j].index);
  }

  void LoadWeight(size_t j) {
    Ld(program_,
       layer_.weights[heads_[j].linear] +
           (uint64_t{heads_[j].index} * kHeadWords * kKb * kWordA),
       Mem::kWbuf, kWt[j % 2], kHeadWords, kKb, kKb * kWordA,
       std::format("{} ld W {}", note_, Name(j)));
  }

  void PostVec(size_t j, uint32_t q_abuf) {
    const Head& head = heads_[j];
    if (head.linear == 2) {
      program_.Vec({.op = VecOp::kCopy,
                    .src = kCb[j % 2],
                    .dst = kVt[j % 2],
                    .rows_q = kQuads,
                    .n8 = kHeadWords,
                    .q = true},
                   std::format("{} quant {}", note_, Name(j)));
      return;
    }
    program_.Vec({.op = VecOp::kRope,
                  .src = kCb[j % 2],
                  .dst = head.linear == 0 ? q_abuf + (head.index * 2 * kQuads)
                                          : kKt[j % 2],
                  .rows_q = kQuads,
                  .n8 = kHeadWords,
                  .p_base = cross_ ? kPbCosCross : kPbCosSelf,
                  .aux = 1,
                  .q = true,
                  .scratch = kScratch},
                 std::format("{} rope+Q {}", note_, Name(j)));
  }

  void PostStore(size_t j) {
    const Head& head = heads_[j];
    if (head.linear == 0) {
      return;
    }
    const bool key = head.linear == 1;
    St(program_,
       (key ? k_cache_ : v_cache_) +
           ((row0_ + (uint64_t{2} * head.index)) * kWordA),
       Mem::kAbuf, key ? kKt[j % 2] : kVt[j % 2], kQuads, 2, kKvBlocks * kWordA,
       std::format("{} st {} suffix {}", note_, key ? "K" : "V", Name(j)));
  }

  Program& program_;
  const DecoderLayer& layer_;
  uint64_t k_cache_;
  uint64_t v_cache_;
  bool cross_;
  uint64_t row0_;
  std::string note_;
  std::vector<Head> heads_;
};

class Attention {
 public:
  Attention(Program& program, const Expert& expert, const Lm& lm, uint32_t l,
            const Dims& dims, std::string_view note)
      : program_(program),
        expert_(expert),
        cross_(IsCross(l)),
        k_(cross_ ? expert.kc[l] : lm.k[l]),
        v_(cross_ ? expert.vc[l] : lm.v[l]),
        keys_(cross_ ? dims.Tokens() : dims.Tokens() + kSuffix),
        valid_(cross_ ? dims.Prefix() : keys_),
        s_(ScoreWords(dims)),
        note_(note) {}

  void Emit(uint32_t q_abuf) {
    const uint32_t rows = keys_ / 4;
    for (uint32_t g = 0; g < kKvHeads; ++g) {
      const uint64_t column = uint64_t{2} * g * kWordA;
      Ld(program_, k_ + column, Mem::kWbuf, g * 4 * rows, rows, 2,
         kKvBlocks * kWordA, std::format("{} ld K group {}", note_, g),
         DmaKind::kSplit8);
      Ld(program_, v_ + column, Mem::kWbuf, (g * 4 * rows) + (2 * rows), rows,
         2, kKvBlocks * kWordA, std::format("{} ld V group {}", note_, g),
         DmaKind::kT32Split8);
    }
    if (!cross_) {
      Ld(program_, expert_.mask, Mem::kFbuf, 2 * s_, kQuads, keys_ / 8,
         uint64_t{keys_ / 8} * kWordF, std::format("{} ld mask", note_));
    }
    for (uint32_t h = 0; h < kQueryHeads; ++h) {
      if (h >= 1) {
        Softmax(h - 1);
      }
      Scores(h, q_abuf);
      if (h >= 2) {
        StoreOutput(h - 2);
      }
      if (h >= 1) {
        Values(h - 1);
      }
    }
    Softmax(kQueryHeads - 1);
    StoreOutput(kQueryHeads - 2);
    Values(kQueryHeads - 1);
    StoreOutput(kQueryHeads - 1);
  }

 private:
  static constexpr std::array<uint32_t, 2> kP = {1536, 1664};

  [[nodiscard]] uint32_t Group(uint32_t h) const { return (h / 3) * keys_; }

  [[nodiscard]] uint32_t O(uint32_t h) const {
    return (4 * s_) + ((h % 2) * kQuads * kHeadWords);
  }

  void Scores(uint32_t h, uint32_t q_abuf) {
    for (const bool lo : {false, true}) {
      program_.Gemm({.a_base = q_abuf + (h * 2 * kQuads),
                     .a_stride = 2,
                     .kb = 2,
                     .m = kSuffix,
                     .w_base = Group(h) + (lo ? keys_ / 4 : 0),
                     .w_stride = 2,
                     .n8 = keys_ / 8,
                     .c_base = (h % 2) * s_,
                     .wmode = lo ? WeightMode::kW8Lo : WeightMode::kW8Hi,
                     .acc = lo},
                    std::format("{} S {} h{}", note_, lo ? "lo" : "hi", h));
    }
  }

  void Softmax(uint32_t h) {
    const uint32_t s = (h % 2) * s_;
    if (!cross_) {
      program_.Vec({.op = VecOp::kAdd,
                    .src = s,
                    .src2 = 2 * s_,
                    .dst = s,
                    .rows_q = kQuads,
                    .n8 = keys_ / 8},
                   std::format("{} S + mask h{}", note_, h));
    }
    program_.Vec({.op = VecOp::kSoftmax,
                  .src = s,
                  .dst = kP[h % 2],
                  .rows_q = kQuads,
                  .n8 = keys_ / 8,
                  .aux = valid_,
                  .q = true,
                  .scratch = 3 * s_},
                 std::format("{} softmax+Q h{}", note_, h));
  }

  void Values(uint32_t h) {
    const uint32_t blocks = keys_ / kBlock;
    for (const bool lo : {false, true}) {
      program_.Gemm({.a_base = kP[h % 2],
                     .a_stride = blocks,
                     .kb = blocks,
                     .m = kSuffix,
                     .w_base = Group(h) + ((lo ? 3 : 2) * (keys_ / 4)),
                     .w_stride = blocks,
                     .n8 = kHeadWords,
                     .c_base = O(h),
                     .wmode = lo ? WeightMode::kW8Lo : WeightMode::kW8Hi,
                     .acc = lo},
                    std::format("{} PV {} h{}", note_, lo ? "lo" : "hi", h));
    }
  }

  void StoreOutput(uint32_t h) {
    St(program_, expert_.o + (uint64_t{h} * kHeadWords * kWordF), Mem::kFbuf,
       O(h), kQuads, kHeadWords, kQueryWords * kWordF,
       std::format("{} st O h{}", note_, h));
  }

  Program& program_;
  const Expert& expert_;
  bool cross_;
  uint64_t k_;
  uint64_t v_;
  uint32_t keys_;
  uint32_t valid_;
  uint32_t s_;
  std::string note_;
};

void ProjResidual(Program& program, std::vector<Piece> pieces, uint64_t w,
                  uint32_t kb_total, const std::array<uint64_t, 2>& x,
                  std::string_view note) {
  constexpr std::array<uint32_t, 2> kXr = {1536, 1984};
  const auto after_first = [&](uint32_t nt) {
    Ld(program, x[0] + (uint64_t{kTileFirst[nt]} * kWordF), Mem::kFbuf,
       kXr[nt % 2], kQuads, kTileGroups[nt], kWords * kWordF,
       std::format("{} ld residual nt{}", note, nt));
  };
  const auto post = [&](uint32_t nt) {
    const uint32_t c = kQuads * kTileFirst[nt];
    program.Vec({.op = VecOp::kAdd,
                 .src = c,
                 .src2 = kXr[nt % 2],
                 .dst = c,
                 .rows_q = kQuads,
                 .n8 = kTileGroups[nt]},
                std::format("{} residual nt{}", note, nt));
    St(program, x[1] + (uint64_t{kTileFirst[nt]} * kWordF), Mem::kFbuf, c,
       kQuads, kTileGroups[nt], kWords * kWordF,
       std::format("{} st x nt{}", note, nt));
  };
  LinearTiles(program,
              PaddedTiles({{.mode = WeightMode::kFp4, .offset = w}}, kb_total,
                          std::move(pieces), false),
              note, after_first, post);
}

void GateUp(Program& program, const DecoderLayer& layer, uint32_t a_base,
            uint32_t h_abuf, std::string_view note) {
  constexpr std::array<uint32_t, 2> kWg = {0, 736};
  constexpr std::array<uint32_t, 2> kWu = {368, 1104};
  constexpr std::array<uint32_t, 2> kG = {0, 256};
  constexpr std::array<uint32_t, 2> kU = {512, 768};
  constexpr uint32_t kScratch = 1024;
  const auto load_w = [&](uint32_t j) {
    const uint64_t offset = uint64_t{j} * kGateGroups * kKb * kWordA;
    Ld(program, layer.weights[kGate] + offset, Mem::kWbuf, kWg[j % 2],
       kGateGroups, kKb, kKb * kWordA, std::format("{} ld W gate {}", note, j));
    Ld(program, layer.weights[kUp] + offset, Mem::kWbuf, kWu[j % 2],
       kGateGroups, kKb, kKb * kWordA, std::format("{} ld W up {}", note, j));
  };
  const auto post = [&](uint32_t j) {
    program.Vec({.op = VecOp::kActMul,
                 .src = kG[j % 2],
                 .src2 = kU[j % 2],
                 .dst = h_abuf + (j * kQuads * kGateBlocks),
                 .rows_q = kQuads,
                 .n8 = kGateGroups,
                 .q = true,
                 .scratch = kScratch},
                std::format("{} silu_mul+Q {}", note, j));
  };
  load_w(0);
  for (uint32_t j = 0; j < kGateTiles; ++j) {
    if (j >= 1) {
      post(j - 1);
    }
    for (const bool up : {false, true}) {
      program.Gemm({.a_base = a_base,
                    .a_stride = kKb,
                    .kb = kKb,
                    .m = kSuffix,
                    .w_base = up ? kWu[j % 2] : kWg[j % 2],
                    .w_stride = kKb,
                    .n8 = kGateGroups,
                    .c_base = up ? kU[j % 2] : kG[j % 2]},
                   std::format("{} gemm {} {}", note, up ? "up" : "gate", j));
    }
    if (j + 1 < kGateTiles) {
      load_w(j + 1);
    }
  }
  post(kGateTiles - 1);
}

void Layer(Program& program, const Expert& expert, const Lm& lm,
           const Dims& dims, uint32_t l, const std::array<uint64_t, 3>& x,
           std::string_view tag) {
  constexpr uint32_t kA = 0;
  constexpr uint32_t kQa = 512;
  constexpr uint32_t kAo = 1024;
  constexpr uint32_t kH = 2048;
  const DecoderLayer& layer = expert.layers[l];
  Ld(program, layer.pbuf, Mem::kPbuf, 0, 1, 2 * kKb, uint64_t{2} * kKb * kWordF,
     std::format("{} ld PBUF (gamma x 2)", tag));
  RmsNormQ(program, x[0], kPbLn1, kA, std::format("{} RMSNorm1", tag));
  Qkv(program, layer, lm, l, dims.Tokens(), std::format("{} qkv", tag))
      .Emit(kA, kQa);
  Attention(program, expert, lm, l, dims, std::format("{} attn", tag))
      .Emit(kQa);
  Ld(program, expert.o, Mem::kFbuf, 0, kQuads, kQueryWords,
     kQueryWords * kWordF, std::format("{} quant O ld O", tag));
  program.Vec({.op = VecOp::kCopy,
               .src = 0,
               .dst = kAo,
               .rows_q = kQuads,
               .n8 = kQueryWords,
               .q = true},
              std::format("{} quant O", tag));
  ProjResidual(
      program,
      {{.a_base = kAo, .a_stride = kKbQuery, .kb = kKbQuery, .kb_offset = 0}},
      layer.weights[kO], kKbQuery, {x[0], x[1]}, std::format("{} o_proj", tag));
  RmsNormQ(program, x[1], kPbLn2, kA, std::format("{} RMSNorm2", tag));
  GateUp(program, layer, kA, kH, std::format("{} mlp", tag));
  std::vector<Piece> pieces;
  pieces.reserve(kGateTiles);
  for (uint32_t j = 0; j < kGateTiles; ++j) {
    pieces.push_back({.a_base = kH + (j * kQuads * kGateBlocks),
                      .a_stride = kGateBlocks,
                      .kb = kGateBlocks,
                      .kb_offset = j * kGateBlocks});
  }
  ProjResidual(program, std::move(pieces), layer.weights[kDown], kKbHidden,
               {x[1], x[2]}, std::format("{} down", tag));
}

void CrossKv(Program& program, const Expert& expert, const Lm& lm,
             const Dims& dims, uint32_t l) {
  const uint32_t quads = dims.Tokens() / 4;
  const uint32_t a_words = quads * kKvBlocks;
  const uint32_t c_words = quads * kKvGroups;
  const std::array<uint64_t, 2> sources = {lm.k[l], lm.v[l]};
  const std::array<uint64_t, 2> outputs = {expert.kc[l], expert.vc[l]};
  const std::array<uint64_t, 2> weights = {expert.layers[l].weights[1],
                                           expert.layers[l].weights[2]};
  for (uint32_t i = 0; i < 2; ++i) {
    const std::string note =
        std::format("E{} cross kv {}", l, i == 0 ? "k" : "v");
    Ld(program, sources[i], Mem::kAbuf, i * a_words, quads, kKvBlocks,
       kKvBlocks * kWordA, std::format("{} ld cache", note));
    Ld(program, weights[i], Mem::kWbuf, i * kKvPlane, kKvGroups, kKvBlocks,
       kKvBlocks * kWordA, std::format("{} ld W", note));
    program.Gemm({.a_base = i * a_words,
                  .a_stride = kKvBlocks,
                  .kb = kKvBlocks,
                  .m = dims.Tokens(),
                  .w_base = i * kKvPlane,
                  .w_stride = kKvBlocks,
                  .n8 = kKvGroups,
                  .c_base = i * c_words},
                 std::format("{} gemm", note));
    program.Vec({.op = VecOp::kCopy,
                 .src = i * c_words,
                 .dst = kKvOut + (i * a_words),
                 .rows_q = quads,
                 .n8 = kKvGroups,
                 .q = true},
                std::format("{} quant", note));
    St(program, outputs[i], Mem::kAbuf, kKvOut + (i * a_words), quads,
       kKvBlocks, kKvBlocks * kWordA, std::format("{} st", note));
  }
}

class Suffix {
 public:
  Suffix(Program& program, const Expert& expert, std::string_view tag)
      : program_(program), expert_(expert), tag_(tag) {}

  void Emit(uint32_t step, uint64_t xt, uint64_t x0) {
    Ld(program_, expert_.time[step], Mem::kAbuf, kAt, kQuads, kKb, kKb * kWordA,
       std::format("{} ld time embedding (MXINT8)", tag_));
    Ld(program_, xt, Mem::kFbuf, kXt, kQuads, kActionWords,
       kActionWords * kWordF, std::format("{} ld x_t", tag_));
    program_.Vec({.op = VecOp::kCopy,
                  .src = kXt,
                  .dst = kAx,
                  .rows_q = kQuads,
                  .n8 = kActionWords,
                  .q = true},
                 std::format("{} quant x_t", tag_));
    LinearTiles(
        program_,
        PaddedTiles(W8Planes(expert_.heads[kHeadIn]), 1,
                    {{.a_base = kAx, .a_stride = 1, .kb = 1, .kb_offset = 0}},
                    false),
        std::format("{} action_in", tag_), {}, [&](uint32_t nt) { BiasQ(nt); });
    LinearTiles(program_,
                PaddedTiles(W8Planes(expert_.heads[kHeadMlpInAction]), kKb,
                            TiledPieces(kAa), false),
                std::format("{} mlp_in(action)", tag_), {}, {});
    LinearTiles(
        program_,
        PaddedTiles(
            W8Planes(expert_.heads[kHeadMlpInTime]), kKb,
            {{.a_base = kAt, .a_stride = kKb, .kb = kKb, .kb_offset = 0}},
            true),
        std::format("{} mlp_in(time)", tag_), {},
        [&](uint32_t nt) { BiasSilu(nt); });
    LinearTiles(program_,
                PaddedTiles(W8Planes(expert_.heads[kHeadMlpOut]), kKb,
                            TiledPieces(kAh), false),
                std::format("{} mlp_out", tag_), {},
                [&](uint32_t nt) { BiasStore(nt, x0); });
  }

 private:
  static constexpr uint32_t kAx = 0;
  static constexpr uint32_t kAt = 512;
  static constexpr uint32_t kAa = 1024;
  static constexpr uint32_t kAh = 1536;
  static constexpr uint32_t kXt = 2048;
  static constexpr uint32_t kScratch = 2560;

  void Bias(uint32_t nt, uint32_t pb, uint32_t dst, bool q) {
    program_.Vec({.op = VecOp::kAddP,
                  .src = kQuads * kTileFirst[nt],
                  .dst = dst,
                  .rows_q = kQuads,
                  .n8 = kTileGroups[nt],
                  .p_base = pb + (kTileFirst[nt] / 4),
                  .q = q},
                 std::format("{} bias{} nt{}", tag_, q ? "+Q" : "", nt));
  }

  void BiasQ(uint32_t nt) {
    Bias(nt, kPbIn, kAa + (kQuads * (kTileFirst[nt] / 4)), true);
  }

  void BiasSilu(uint32_t nt) {
    const uint32_t c = kQuads * kTileFirst[nt];
    Bias(nt, kPbMlpIn, c, false);
    program_.Vec({.op = VecOp::kAct,
                  .src = c,
                  .dst = kAh + (kQuads * (kTileFirst[nt] / 4)),
                  .rows_q = kQuads,
                  .n8 = kTileGroups[nt],
                  .q = true,
                  .scratch = kScratch},
                 std::format("{} silu+Q nt{}", tag_, nt));
  }

  void BiasStore(uint32_t nt, uint64_t x0) {
    const uint32_t c = kQuads * kTileFirst[nt];
    Bias(nt, kPbMlpOut, c, false);
    St(program_, x0 + (uint64_t{kTileFirst[nt]} * kWordF), Mem::kFbuf, c,
       kQuads, kTileGroups[nt], kWords * kWordF,
       std::format("{} st x0 nt{}", tag_, nt));
  }

  Program& program_;
  const Expert& expert_;
  std::string tag_;
};

void StepEpilogue(Program& program, const Expert& expert, uint32_t dt,
                  const std::array<uint64_t, 4>& x, std::string_view tag) {
  constexpr uint32_t kC = 3072;
  constexpr uint32_t kXt = 3136;
  RmsNormQ(program, x[0], kPbFinal, 0, std::format("{} final norm", tag));
  Ld(program, x[1], Mem::kFbuf, kXt, kQuads, kActionWords,
     kActionWords * kWordF, std::format("{} ld x_t", tag));
  const auto post = [&](uint32_t) {
    const VecArgs whole{.op = VecOp::kAddP,
                        .src = kC,
                        .dst = kC,
                        .rows_q = kQuads,
                        .n8 = kActionWords,
                        .p_base = kPbOut};
    program.Vec(whole, std::format("{} action_out bias", tag));
    St(program, x[2], Mem::kFbuf, kC, kQuads, kActionWords,
       kActionWords * kWordF, std::format("{} st v_t", tag));
    program.Vec({.op = VecOp::kScale,
                 .src = kC,
                 .dst = kC,
                 .rows_q = kQuads,
                 .n8 = kActionWords,
                 .imm = dt},
                std::format("{} dt * v_t", tag));
    program.Vec({.op = VecOp::kAdd,
                 .src = kC,
                 .src2 = kXt,
                 .dst = kC,
                 .rows_q = kQuads,
                 .n8 = kActionWords},
                std::format("{} x_t + dt * v_t", tag));
    St(program, x[3], Mem::kFbuf, kC, kQuads, kActionWords,
       kActionWords * kWordF, std::format("{} st x_t next", tag));
  };
  LinearTiles(
      program,
      {.planes = W8Planes(expert.heads[kHeadOut]),
       .kb_total = kKb,
       .pieces = {{.a_base = 0, .a_stride = kKb, .kb = kKb, .kb_offset = 0}},
       .groups = {kActionWords},
       .first = {0},
       .c_base = kC,
       .accumulate = false},
      std::format("{} action_out", tag), {}, post);
}

}  // namespace

Expert PlaceExpert(DdrImage& image, const SafeTensors& quant,
                   const SafeTensors& model, const Dims& dims) {
  Expert expert;
  PlaceHeads(image, model, expert);
  expert.rope = Place(image, "ex.rope", ExpertRope(dims));
  expert.tbuf = Place(image, "ex.tbuf", PackTbuf(MakeTbuf(Activation::kSilu)));
  expert.mask = Place(image, "ex.mask", SelfMask(dims));
  for (uint32_t k = 0; k < dims.steps; ++k) {
    expert.time.push_back(
        Place(image, std::format("ex.time{}", k), TimeInput(k, dims.steps)));
  }
  expert.layers.resize(kLayers);
  for (uint32_t l = 0; l < kLayers; ++l) {
    PlaceLayer(image, quant, model, l, expert.layers[l]);
  }
  expert.o = image.Alloc("ex.O", uint64_t{kQuads} * kQueryWords * kWordF);
  const uint64_t cache = uint64_t{dims.Tokens() / 4} * kKvBlocks * kWordA;
  expert.kc.resize(kLayers);
  expert.vc.resize(kLayers);
  for (uint32_t l = 1; l < kLayers; l += 2) {
    expert.kc[l] = image.Alloc(std::format("ex.Kc{}", l), cache);
    expert.vc[l] = image.Alloc(std::format("ex.Vc{}", l), cache);
  }
  const uint64_t x_bytes = uint64_t{kQuads} * kWords * kWordF;
  for (uint32_t i = 0; i < expert.x.size(); ++i) {
    expert.x[i] = image.Alloc(std::format("xe.{}", i), x_bytes);
  }
  const uint64_t action_bytes = uint64_t{kQuads} * kActionWords * kWordF;
  for (uint32_t k = 0; k <= dims.steps; ++k) {
    expert.xt.push_back(image.Alloc(std::format("x_t.{}", k), action_bytes));
  }
  for (uint32_t k = 0; k < dims.steps; ++k) {
    expert.vt.push_back(image.Alloc(std::format("v_t.{}", k), action_bytes));
  }
  return expert;
}

void EmitExpert(Program& program, const Expert& expert, const Lm& lm,
                const Dims& dims) {
  const Program::Scope expert_scope(program, "expert");
  const uint32_t dt = Fp32OfFp64(exact::Real::Ratio(-1, dims.steps));
  {
    const Program::Scope scope(program, "setup");
    Ld(program, expert.tbuf, Mem::kTbuf, 0, 1, kTbufWordsLoaded,
       kTbufWordsLoaded * kWordT, "ex ld TBUF (exp2 + silu)");
    Ld(program, expert.rope, Mem::kPbuf, kPbCosSelf, 1, kRopeWords,
       kRopeWords * kWordF, "ex ld PBUF cos/sin tiles (self / cross)");
    Ld(program, expert.pb_heads, Mem::kPbuf, kPbIn, 1, kPbEnd - kPbIn,
       uint64_t{kPbEnd - kPbIn} * kWordF,
       "ex ld PBUF head biases and final norm gamma");
  }
  for (uint32_t l = 1; l < kLayers; l += 2) {
    const Program::Scope scope(program, std::format("cross_kv/E{}", l));
    CrossKv(program, expert, lm, dims, l);
  }
  for (uint32_t k = 0; k < dims.steps; ++k) {
    const Program::Scope step(program, std::format("S{}", k));
    {
      const Program::Scope scope(program, "suffix");
      Suffix(program, expert, std::format("S{} suffix", k))
          .Emit(k, expert.xt[k], expert.x[0]);
    }
    for (uint32_t l = 0; l < kLayers; ++l) {
      const Program::Scope scope(program, std::format("E{}", l));
      Layer(program, expert, lm, dims, l,
            {expert.x[l % 2], expert.x[2], expert.x[(l + 1) % 2]},
            std::format("S{} E{}", k, l));
    }
    const Program::Scope scope(program, "out");
    StepEpilogue(
        program, expert, dt,
        {expert.x[kLayers % 2], expert.xt[k], expert.vt[k], expert.xt[k + 1]},
        std::format("S{} out", k));
  }
}

}  // namespace hotarubi::compiler::smolvla
