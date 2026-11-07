#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <string>
#include <string_view>
#include <vector>

#include "ddr_image.hpp"
#include "hotarubi/safetensors.hpp"
#include "isa.hpp"
#include "program.hpp"
#include "smolvla_parts.hpp"
#include "tensors.hpp"
#include "weights.hpp"

namespace hotarubi::compiler::smolvla {
namespace {

constexpr uint32_t kLayers = 16;
constexpr uint32_t kHidden = 2560;
constexpr uint32_t kKbHidden = kHidden / kBlock;
constexpr uint32_t kTileQuads = 8;
constexpr uint32_t kGateGroups = 16;
constexpr uint32_t kGateTiles = kHidden / (8 * kGateGroups);
constexpr uint32_t kGateBlocks = kGateGroups / 4;
constexpr uint32_t kHeadWords = 8;
constexpr uint32_t kPbLn1 = 0;
constexpr uint32_t kPbLn2 = kLmBlocks;
constexpr uint32_t kPbCos = 64;
constexpr uint32_t kAbufBank = 1024;
constexpr size_t kO = 3;
constexpr size_t kGate = 4;
constexpr size_t kUp = 5;
constexpr size_t kDown = 6;
constexpr std::array<std::string_view, 7> kLinearModules = {
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj",    "mlp.up_proj",
    "mlp.down_proj"};
constexpr std::array<std::string_view, 7> kLinearTags = {
    "q", "k", "v", "o", "gate", "up", "down"};

uint32_t Bank(uint32_t address) {
  return (address + kAbufBank - 1) / kAbufBank * kAbufBank;
}

struct Plan {
  uint32_t tokens = 0;
  uint32_t quads = 0;
  uint32_t context = 0;
  uint32_t head = 0;
  uint32_t rows = 0;
  uint32_t qa = 0;
  std::array<uint32_t, 2> kt{};
  std::array<uint32_t, 2> vt{};
  std::array<uint32_t, 2> p{};
  uint32_t ao = 0;
  uint32_t h = 0;
  uint32_t chunk = 0;
  uint32_t groups = 0;
};

Plan MakePlan(const Dims& dims) {
  Plan plan;
  plan.tokens = dims.Tokens();
  plan.quads = plan.tokens / 4;
  plan.context = dims.Context();
  Require(dims.lang % 4 == 0 && dims.lang <= kMaxLanguage,
          std::format("{} language tokens: needs a multiple of 4 up to {}",
                      dims.lang, kMaxLanguage));
  plan.head = 2 * plan.quads;
  plan.rows = kLmBlocks * plan.quads;
  for (plan.chunk = plan.quads; plan.chunk >= kTileQuads;
       plan.chunk -= kTileQuads) {
    const uint32_t s = plan.chunk * plan.tokens / 8;
    if (plan.quads % plan.chunk == 0 &&
        (3 * s) + (2 * plan.chunk * kHeadWords) <= kMemDepth[2]) {
      break;
    }
  }
  Require(plan.chunk >= kTileQuads, "LM attention does not fit in FBUF");
  for (plan.groups = kLmWords; plan.groups >= 1; --plan.groups) {
    if (kLmWords % plan.groups == 0 &&
        4 * plan.quads * plan.groups <= kMemDepth[2]) {
      break;
    }
  }
  const uint32_t p_words = plan.chunk * plan.tokens / kBlock;
  plan.qa = Bank(plan.rows);
  const uint32_t temps = Bank(plan.qa + (kQueryHeads * plan.head));
  plan.kt = {temps, temps + plan.head};
  plan.vt = {temps + (2 * plan.head), temps + (3 * plan.head)};
  plan.p = {temps + (4 * plan.head), temps + (4 * plan.head) + p_words};
  plan.ao = Bank(plan.p[1] + p_words);
  const uint32_t h_words = kGateTiles * plan.quads * kGateBlocks;
  plan.h = Bank(plan.ao + plan.rows);
  if (plan.h + h_words > kMemDepth[0]) {
    plan.h = Bank(plan.rows);
  }
  Require(
      plan.ao + plan.rows <= kMemDepth[0] && plan.h + h_words <= kMemDepth[0],
      "LM activations do not fit in ABUF");
  return plan;
}

uint32_t ScoreWords(const Plan& plan) { return plan.chunk * plan.tokens / 8; }

uint64_t RowBytes(const Plan& plan) {
  return uint64_t{plan.quads} * kLmWords * kWordF;
}

uint64_t CacheBytes(const Plan& plan) {
  return uint64_t{(plan.tokens + kSuffix) / 4} * kKvBlocks * kWordA;
}

void PlaceLayer(DdrImage& image, const SafeTensors& quant,
                const SafeTensors& model, uint32_t layer, DecoderLayer& out) {
  const std::string prefix = std::format(
      "model.vlm_with_expert.vlm.model.text_model.layers.{}", layer);
  const std::string tag = std::format("T{}", layer);
  for (size_t i = 0; i < kLinearModules.size(); ++i) {
    const Mxfp4Matrix raw =
        ReadMxfp4(quant, std::format("{}.{}", prefix, kLinearModules[i]));
    out.weights[i] =
        Place(image, std::format("{}.w{}", tag, kLinearTags[i]),
              PackWbuf(CanonicalMxfp4(raw, i == 0 ? -3 : 0, raw.rows)));
  }
  std::vector<uint32_t> pbuf =
      model.Float32Bits(std::format("{}.input_layernorm.weight", prefix));
  for (const uint32_t value : model.Float32Bits(
           std::format("{}.post_attention_layernorm.weight", prefix))) {
    pbuf.push_back(value);
  }
  out.pbuf = Place(image, std::format("{}.pbuf", tag), PackPbuf(pbuf));
}

void RmsNorm(Program& program, const Plan& plan, uint64_t x, uint32_t pb,
             uint32_t dst, std::string_view note) {
  constexpr std::array<uint32_t, 2> kXt = {0, 960};
  constexpr uint32_t kScratch = 1920;
  for (uint32_t t = 0; t < plan.quads / kTileQuads; ++t) {
    Ld(program, x + (uint64_t{t} * kTileQuads * kLmWords * kWordF), Mem::kFbuf,
       kXt[t % 2], kTileQuads, kLmWords, kLmWords * kWordF,
       std::format("{} ld x tile {}", note, t));
    program.Vec({.op = VecOp::kRmsNorm,
                 .src = kXt[t % 2],
                 .dst = dst + (t * kTileQuads * kLmBlocks),
                 .rows_q = kTileQuads,
                 .n8 = kLmWords,
                 .p_base = pb,
                 .imm = RmsEps(),
                 .q = true,
                 .scratch = kScratch},
                std::format("{} RMSNorm tile {}", note, t));
  }
}

void QuantRows(Program& program, const Plan& plan, uint64_t x, uint32_t dst,
               std::string_view note) {
  constexpr std::array<uint32_t, 2> kXt = {0, 960};
  for (uint32_t t = 0; t < plan.quads / kTileQuads; ++t) {
    Ld(program, x + (uint64_t{t} * kTileQuads * kLmWords * kWordF), Mem::kFbuf,
       kXt[t % 2], kTileQuads, kLmWords, kLmWords * kWordF,
       std::format("{} ld tile {}", note, t));
    program.Vec({.op = VecOp::kQuant,
                 .src = kXt[t % 2],
                 .dst = dst + (t * kTileQuads * kLmBlocks),
                 .rows_q = kTileQuads,
                 .n8 = kLmWords},
                std::format("{} quant tile {}", note, t));
  }
}

struct Head {
  size_t linear = 0;
  uint32_t index = 0;
};

class Qkv {
 public:
  Qkv(Program& program, const Plan& plan, const DecoderLayer& layer,
      uint64_t k_cache, uint64_t v_cache, std::string_view note)
      : program_(program),
        plan_(plan),
        layer_(layer),
        k_cache_(k_cache),
        v_cache_(v_cache),
        note_(note) {
    for (size_t linear = 0; linear < 3; ++linear) {
      const uint32_t count = linear == 0 ? kQueryHeads : kKvHeads;
      for (uint32_t h = 0; h < count; ++h) {
        heads_.push_back({.linear = linear, .index = h});
      }
    }
  }

  void Emit() {
    LoadWeight(0);
    for (size_t j = 0; j < heads_.size(); ++j) {
      if (j >= 1) {
        PostVec(j - 1);
      }
      program_.Gemm({.a_base = 0,
                     .a_stride = kLmBlocks,
                     .kb = kLmBlocks,
                     .m = plan_.tokens,
                     .w_base = kWt[j % 2],
                     .w_stride = kLmBlocks,
                     .n8 = kHeadWords,
                     .c_base = C(j)},
                    std::format("{} gemm {}", note_, Name(j)));
      if (j + 1 < heads_.size()) {
        LoadWeight(j + 1);
      }
      if (j >= 1) {
        PostStore(j - 1);
      }
    }
    PostVec(heads_.size() - 1);
    PostStore(heads_.size() - 1);
  }

 private:
  static constexpr std::array<uint32_t, 2> kWt = {0, 240};

  [[nodiscard]] uint32_t C(size_t j) const {
    return static_cast<uint32_t>(j % 2) * plan_.quads * kHeadWords;
  }

  [[nodiscard]] std::string Name(size_t j) const {
    return std::format("{}{}", kLinearTags[heads_[j].linear], heads_[j].index);
  }

  void LoadWeight(size_t j) {
    Ld(program_,
       layer_.weights[heads_[j].linear] +
           (uint64_t{heads_[j].index} * kHeadWords * kLmBlocks * kWordA),
       Mem::kWbuf, kWt[j % 2], kHeadWords, kLmBlocks, kLmBlocks * kWordA,
       std::format("{} ld W {}", note_, Name(j)));
  }

  void PostVec(size_t j) {
    const Head& head = heads_[j];
    const uint32_t scratch = 2 * plan_.quads * kHeadWords;
    if (head.linear == 2) {
      program_.Vec({.op = VecOp::kQuant,
                    .src = C(j),
                    .dst = plan_.vt[j % 2],
                    .rows_q = plan_.quads,
                    .n8 = kHeadWords},
                   std::format("{} quant {}", note_, Name(j)));
      return;
    }
    const uint32_t dst = head.linear == 0 ? plan_.qa + (head.index * plan_.head)
                                          : plan_.kt[j % 2];
    program_.Vec({.op = VecOp::kRope,
                  .src = C(j),
                  .dst = dst,
                  .rows_q = plan_.quads,
                  .n8 = kHeadWords,
                  .p_base = kPbCos,
                  .aux = 1,
                  .q = true,
                  .scratch = scratch},
                 std::format("{} rope+Q {}", note_, Name(j)));
  }

  void PostStore(size_t j) {
    const Head& head = heads_[j];
    if (head.linear == 0) {
      return;
    }
    const bool key = head.linear == 1;
    St(program_,
       (key ? k_cache_ : v_cache_) + (uint64_t{2} * head.index * kWordA),
       Mem::kAbuf, key ? plan_.kt[j % 2] : plan_.vt[j % 2], plan_.quads, 2,
       kKvBlocks * kWordA,
       std::format("{} st {} cache {}", note_, key ? "K" : "V", Name(j)));
  }

  Program& program_;
  const Plan& plan_;
  const DecoderLayer& layer_;
  uint64_t k_cache_;
  uint64_t v_cache_;
  std::string note_;
  std::vector<Head> heads_;
};

class Attention {
 public:
  Attention(Program& program, const Plan& plan, uint64_t k_cache,
            uint64_t v_cache, uint64_t o, std::string_view note)
      : program_(program),
        plan_(plan),
        k_cache_(k_cache),
        v_cache_(v_cache),
        o_(o),
        note_(note),
        chunks_(plan.quads / plan.chunk),
        s_(ScoreWords(plan)) {}

  void Emit() {
    const uint32_t group = 4 * plan_.quads;
    for (uint32_t g = 0; g < kKvHeads; ++g) {
      const uint64_t column = uint64_t{2} * g * kWordA;
      Ld(program_, k_cache_ + column, Mem::kWbuf, g * group, plan_.quads, 2,
         kKvBlocks * kWordA, std::format("{} ld K group {}", note_, g),
         DmaMode::kSplit8);
      Ld(program_, v_cache_ + column, Mem::kWbuf, (g * group) + plan_.head,
         plan_.quads, 2, kKvBlocks * kWordA,
         std::format("{} ld V group {}", note_, g), DmaMode::kT32Split8);
    }
    const uint32_t total = kQueryHeads * chunks_;
    for (uint32_t g = 0; g < total; ++g) {
      if (g >= 1) {
        Softmax(g - 1);
      }
      Scores(g);
      if (g >= 2) {
        StoreOutput(g - 2);
      }
      if (g >= 1) {
        Values(g - 1);
      }
    }
    Softmax(total - 1);
    StoreOutput(total - 2);
    Values(total - 1);
    StoreOutput(total - 1);
  }

 private:
  [[nodiscard]] uint32_t Group(uint32_t g) const {
    return (g / chunks_ / 3) * 4 * plan_.quads;
  }

  [[nodiscard]] uint32_t S(uint32_t g) const { return (g % 2) * s_; }

  [[nodiscard]] uint32_t O(uint32_t g) const {
    return (3 * s_) + ((g % 2) * plan_.chunk * kHeadWords);
  }

  void Scores(uint32_t g) {
    const uint32_t h = g / chunks_;
    const uint32_t c = g % chunks_;
    for (const bool lo : {false, true}) {
      program_.Gemm(
          {.a_base = plan_.qa + (h * plan_.head) + (c * plan_.chunk * 2),
           .a_stride = 2,
           .kb = 2,
           .m = plan_.chunk * 4,
           .w_base = Group(g) + (lo ? plan_.quads : 0),
           .w_stride = 2,
           .n8 = plan_.tokens / 8,
           .c_base = S(g),
           .wmode = lo ? WeightMode::kW8Lo : WeightMode::kW8Hi,
           .acc = lo},
          std::format("{} S {} h{} c{}", note_, lo ? "lo" : "hi", h, c));
    }
  }

  void Softmax(uint32_t g) {
    const uint32_t first = (g % chunks_) * plan_.chunk;
    const uint32_t last = first + plan_.chunk;
    const uint32_t state = plan_.context / 4;
    const std::array<std::array<uint32_t, 3>, 2> parts = {
        {{first, std::min(last, state), plan_.context},
         {std::max(first, state), last, plan_.context + 1}}};
    for (const auto& [begin, end, valid] : parts) {
      if (begin >= end) {
        continue;
      }
      program_.Vec(
          {.op = VecOp::kSoftmax,
           .src = S(g) + ((begin - first) * plan_.tokens / 8),
           .dst = plan_.p[g % 2] + ((begin - first) * plan_.tokens / kBlock),
           .rows_q = end - begin,
           .n8 = plan_.tokens / 8,
           .aux = valid,
           .q = true,
           .scratch = 2 * s_},
          std::format("{} softmax+Q h{} c{} keys<{}", note_, g / chunks_,
                      g % chunks_, valid));
    }
  }

  void Values(uint32_t g) {
    const uint32_t blocks = plan_.tokens / kBlock;
    for (const bool lo : {false, true}) {
      program_.Gemm({.a_base = plan_.p[g % 2],
                     .a_stride = blocks,
                     .kb = blocks,
                     .m = plan_.chunk * 4,
                     .w_base = Group(g) + plan_.head + (lo ? plan_.quads : 0),
                     .w_stride = blocks,
                     .n8 = kHeadWords,
                     .c_base = O(g),
                     .wmode = lo ? WeightMode::kW8Lo : WeightMode::kW8Hi,
                     .acc = lo},
                    std::format("{} PV {} h{} c{}", note_, lo ? "lo" : "hi",
                                g / chunks_, g % chunks_));
    }
  }

  void StoreOutput(uint32_t g) {
    const uint32_t h = g / chunks_;
    const uint32_t c = g % chunks_;
    St(program_,
       o_ + (((uint64_t{c} * plan_.chunk * kLmWords) +
              (uint64_t{h} * kHeadWords)) *
             kWordF),
       Mem::kFbuf, O(g), plan_.chunk, kHeadWords, kLmWords * kWordF,
       std::format("{} st O h{} c{}", note_, h, c));
  }

  Program& program_;
  const Plan& plan_;
  uint64_t k_cache_;
  uint64_t v_cache_;
  uint64_t o_;
  std::string note_;
  uint32_t chunks_;
  uint32_t s_;
};

struct Piece {
  uint32_t a_offset = 0;
  uint32_t kb_offset = 0;
  uint32_t kb = 0;
};

class ProjResidual {
 public:
  ProjResidual(Program& program, const Plan& plan, std::string_view note)
      : program_(program),
        plan_(plan),
        note_(note),
        tiles_(kLmWords / plan.groups),
        c_words_(plan.quads * plan.groups) {}

  void Emit(uint32_t a_base, uint32_t a_stride,
            const std::vector<Piece>& pieces, uint64_t w, uint32_t w_total,
            const std::array<uint64_t, 2>& x) {
    const auto count = static_cast<uint32_t>(tiles_ * pieces.size());
    const auto load_w = [&](uint32_t wt) {
      const uint32_t nt = wt / pieces.size();
      const Piece& piece = pieces[wt % pieces.size()];
      Ld(program_,
         w + (((uint64_t{nt} * plan_.groups * w_total) + piece.kb_offset) *
              kWordA),
         Mem::kWbuf, (wt % 2) * plan_.groups * kLmBlocks, plan_.groups,
         piece.kb, uint64_t{w_total} * kWordA,
         std::format("{} ld W nt{} p{}", note_, nt, wt % pieces.size()));
    };
    load_w(0);
    for (uint32_t wt = 0; wt < count; ++wt) {
      const uint32_t nt = wt / pieces.size();
      const uint32_t p = wt % pieces.size();
      if (p == 0 && nt >= 1) {
        Post(nt - 1, x[1]);
      }
      program_.Gemm({.a_base = a_base + pieces[p].a_offset,
                     .a_stride = a_stride,
                     .kb = pieces[p].kb,
                     .m = plan_.tokens,
                     .w_base = (wt % 2) * plan_.groups * kLmBlocks,
                     .w_stride = pieces[p].kb,
                     .n8 = plan_.groups,
                     .c_base = (nt % 2) * c_words_,
                     .acc = p > 0},
                    std::format("{} gemm nt{} p{}", note_, nt, p));
      if (p == 0) {
        Ld(program_, x[0] + (uint64_t{nt} * plan_.groups * kWordF), Mem::kFbuf,
           (2 + (nt % 2)) * c_words_, plan_.quads, plan_.groups,
           kLmWords * kWordF, std::format("{} ld residual nt{}", note_, nt));
      }
      if (wt + 1 < count) {
        load_w(wt + 1);
      }
    }
    Post(tiles_ - 1, x[1]);
  }

 private:
  void Post(uint32_t nt, uint64_t x_out) {
    const uint32_t c = (nt % 2) * c_words_;
    program_.Vec({.op = VecOp::kAdd,
                  .src = c,
                  .src2 = (2 + (nt % 2)) * c_words_,
                  .dst = c,
                  .rows_q = plan_.quads,
                  .n8 = plan_.groups},
                 std::format("{} residual nt{}", note_, nt));
    St(program_, x_out + (uint64_t{nt} * plan_.groups * kWordF), Mem::kFbuf, c,
       plan_.quads, plan_.groups, kLmWords * kWordF,
       std::format("{} st x nt{}", note_, nt));
  }

  Program& program_;
  const Plan& plan_;
  std::string note_;
  uint32_t tiles_;
  uint32_t c_words_;
};

void GateUp(Program& program, const Plan& plan, const DecoderLayer& layer,
            std::string_view note) {
  constexpr std::array<uint32_t, 2> kWg = {0, 960};
  constexpr std::array<uint32_t, 2> kWu = {480, 1440};
  const uint32_t g_words = plan.quads * kGateGroups;
  const auto load_w = [&](uint32_t j) {
    const uint64_t offset = uint64_t{j} * kGateGroups * kLmBlocks * kWordA;
    Ld(program, layer.weights[kGate] + offset, Mem::kWbuf, kWg[j % 2],
       kGateGroups, kLmBlocks, kLmBlocks * kWordA,
       std::format("{} ld W gate {}", note, j));
    Ld(program, layer.weights[kUp] + offset, Mem::kWbuf, kWu[j % 2],
       kGateGroups, kLmBlocks, kLmBlocks * kWordA,
       std::format("{} ld W up {}", note, j));
  };
  const auto post = [&](uint32_t j) {
    program.Vec({.op = VecOp::kSiluMul,
                 .src = (j % 2) * g_words,
                 .src2 = (2 + (j % 2)) * g_words,
                 .dst = plan.h + (j * plan.quads * kGateBlocks),
                 .rows_q = plan.quads,
                 .n8 = kGateGroups,
                 .q = true,
                 .scratch = 4 * g_words},
                std::format("{} silu_mul+Q {}", note, j));
  };
  load_w(0);
  for (uint32_t j = 0; j < kGateTiles; ++j) {
    if (j >= 1) {
      post(j - 1);
    }
    for (const bool up : {false, true}) {
      program.Gemm({.a_base = 0,
                    .a_stride = kLmBlocks,
                    .kb = kLmBlocks,
                    .m = plan.tokens,
                    .w_base = up ? kWu[j % 2] : kWg[j % 2],
                    .w_stride = kLmBlocks,
                    .n8 = kGateGroups,
                    .c_base = ((up ? 2 : 0) + (j % 2)) * g_words},
                   std::format("{} gemm {} {}", note, up ? "up" : "gate", j));
    }
    if (j + 1 < kGateTiles) {
      load_w(j + 1);
    }
  }
  post(kGateTiles - 1);
}

void Layer(Program& program, const Plan& plan, const Lm& lm, uint32_t l,
           const std::array<uint64_t, 3>& x, std::string_view tag) {
  const DecoderLayer& layer = lm.layers[l];
  Ld(program, layer.pbuf, Mem::kPbuf, 0, 1, 2 * kLmBlocks,
     uint64_t{2} * kLmBlocks * kWordF,
     std::format("{} ld PBUF (RMSNorm gamma x 2)", tag));
  RmsNorm(program, plan, x[0], kPbLn1, 0, std::format("{} RMSNorm1", tag));
  Qkv(program, plan, layer, lm.k[l], lm.v[l], std::format("{} qkv", tag))
      .Emit();
  Attention(program, plan, lm.k[l], lm.v[l], lm.o, std::format("{} attn", tag))
      .Emit();
  QuantRows(program, plan, lm.o, plan.ao, std::format("{} quant O", tag));
  ProjResidual(program, plan, std::format("{} o_proj", tag))
      .Emit(plan.ao, kLmBlocks,
            {{.a_offset = 0, .kb_offset = 0, .kb = kLmBlocks}},
            layer.weights[kO], kLmBlocks, {x[0], x[1]});
  RmsNorm(program, plan, x[1], kPbLn2, 0, std::format("{} RMSNorm2", tag));
  GateUp(program, plan, layer, std::format("{} mlp", tag));
  std::vector<Piece> pieces;
  pieces.reserve(kGateTiles);
  for (uint32_t j = 0; j < kGateTiles; ++j) {
    pieces.push_back({.a_offset = j * plan.quads * kGateBlocks,
                      .kb_offset = j * kGateBlocks,
                      .kb = kGateBlocks});
  }
  ProjResidual(program, plan, std::format("{} down", tag))
      .Emit(plan.h, kGateBlocks, pieces, layer.weights[kDown], kKbHidden,
            {x[1], x[2]});
}

}  // namespace

Lm PlaceLm(DdrImage& image, const SafeTensors& quant, const SafeTensors& model,
           const Dims& dims) {
  const Plan plan = MakePlan(dims);
  Lm lm;
  std::vector<uint32_t> positions(plan.tokens);
  for (uint32_t i = 0; i < plan.tokens; ++i) {
    positions[i] = std::min(i, plan.context);
  }
  lm.rope = Place(image, "lm.rope", RopeTables(positions));
  lm.tbuf = Place(image, "lm.tbuf", PackTbuf(MakeTbuf(Activation::kSilu)));
  lm.layers.resize(kLayers);
  for (uint32_t l = 0; l < kLayers; ++l) {
    PlaceLayer(image, quant, model, l, lm.layers[l]);
  }
  lm.o = image.Alloc("lm.O", RowBytes(plan));
  for (uint32_t l = 0; l < kLayers; ++l) {
    lm.k.push_back(image.Alloc(std::format("lm.K{}", l), CacheBytes(plan)));
    lm.v.push_back(image.Alloc(std::format("lm.V{}", l), CacheBytes(plan)));
  }
  for (uint32_t i = 0; i < lm.x.size(); ++i) {
    lm.x[i] = image.Alloc(std::format("xl.{}", i), RowBytes(plan));
  }
  return lm;
}

void EmitLm(Program& program, const Lm& lm, const Dims& dims) {
  const Program::Scope lm_scope(program, "lm");
  const Plan plan = MakePlan(dims);
  const uint32_t rope_words = 2 * plan.quads * 4;
  {
    const Program::Scope scope(program, "setup");
    Ld(program, lm.tbuf, Mem::kTbuf, 0, 1, kTbufWordsLoaded,
       kTbufWordsLoaded * kWordT, "lm ld TBUF (exp2 + silu)");
    Ld(program, lm.rope, Mem::kPbuf, kPbCos, 1, rope_words, rope_words * kWordF,
       "lm ld PBUF cos/sin tiles");
  }
  for (uint32_t l = 0; l < kLayers; ++l) {
    const Program::Scope scope(program, std::format("L{}", l));
    const uint64_t x_in = l == 0 ? lm.x[0] : lm.x[1 + ((l - 1) % 2)];
    Layer(program, plan, lm, l, {x_in, lm.x[3], lm.x[1 + (l % 2)]},
          std::format("T{}", l));
  }
}

}  // namespace hotarubi::compiler::smolvla
