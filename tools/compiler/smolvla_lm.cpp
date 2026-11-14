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
#include "hotarubi/safetensors.hpp"
#include "isa.hpp"
#include "kernels.hpp"
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

struct Plan {
  uint32_t tokens = 0;
  uint32_t quads = 0;
  uint32_t context = 0;
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
  return plan;
}

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
        PlaceData(image, std::format("{}.w{}", tag, kLinearTags[i]),
                  PackWbuf(CanonicalMxfp4(raw, i == 0 ? -3 : 0, raw.rows)));
  }
  std::vector<uint32_t> pbuf =
      model.Float32Bits(std::format("{}.input_layernorm.weight", prefix));
  for (const uint32_t value : model.Float32Bits(
           std::format("{}.post_attention_layernorm.weight", prefix))) {
    pbuf.push_back(value);
  }
  out.pbuf = PlaceData(image, std::format("{}.pbuf", tag), PackPbuf(pbuf));
}

Tile Quantized(Program& program, const Plan& plan, uint64_t x, VecOp op,
               Place gamma, std::string_view note) {
  const Tile a = program.Alloc(Mem::kAbuf, plan.quads * kLmBlocks);
  EmitQuantize(program,
               {.x = x,
                .quads = plan.quads,
                .words = kLmWords,
                .tile_quads = kTileQuads,
                .op = op,
                .p_base = gamma,
                .imm = op == VecOp::kRmsNorm ? RmsEps() : 0,
                .dst = a.At()},
               note);
  return a;
}

void Qkv(Program& program, const Plan& plan, const DecoderLayer& layer,
         const Tile& a, const Tile& q, const Tile& rope, uint64_t k_cache,
         uint64_t v_cache, std::string_view note) {
  const uint32_t head = 2 * plan.quads;
  const Tile scratch = program.Alloc(Mem::kFbuf, plan.quads * kHeadWords);
  for (size_t linear = 0; linear < 3; ++linear) {
    const std::string name = std::format("{} {}", note, kLinearTags[linear]);
    EmitLinear(program,
               {.planes = {{.w = layer.weights[linear]}},
                .kb_total = kLmBlocks,
                .pieces = {{.a = a.At(),
                            .a_stride = kLmBlocks,
                            .kb = kLmBlocks,
                            .kb_offset = 0}},
                .quads = plan.quads,
                .chunk_quads = plan.quads,
                .groups = (linear == 0 ? kQueryHeads : kKvHeads) * kHeadWords,
                .tile_groups = kHeadWords},
               name, [&](const LinearTile& tile) {
                 Place out = q.At(tile.tile * head);
                 if (linear != 0) {
                   out = program.Alloc(Mem::kAbuf, head).At();
                 }
                 if (linear == 2) {
                   program.Vec({.op = VecOp::kCopy,
                                .src = tile.c.At(),
                                .dst = out,
                                .rows_q = plan.quads,
                                .n8 = kHeadWords,
                                .q = true},
                               std::format("{}{} quant", name, tile.tile));
                 } else {
                   program.Vec({.op = VecOp::kRope,
                                .src = tile.c.At(),
                                .dst = out,
                                .rows_q = plan.quads,
                                .n8 = kHeadWords,
                                .p_base = rope.At(),
                                .aux = 1,
                                .q = true,
                                .scratch = scratch.At()},
                               std::format("{}{} rope+Q", name, tile.tile));
                 }
                 if (linear != 0) {
                   St(program,
                      (linear == 1 ? k_cache : v_cache) +
                          (uint64_t{2} * tile.tile * kWordA),
                      Mem::kAbuf, out, plan.quads, 2, kKvBlocks * kWordA,
                      std::format("{}{} st cache", name, tile.tile));
                 }
               });
  }
}

void ProjResidual(Program& program, const Plan& plan,
                  std::vector<LinearPiece> pieces, uint64_t w,
                  uint32_t kb_total, const std::array<uint64_t, 2>& x,
                  std::string_view note) {
  EmitLinear(program,
             {.planes = {{.w = w}},
              .kb_total = kb_total,
              .pieces = std::move(pieces),
              .quads = plan.quads,
              .chunk_quads = plan.quads,
              .groups = kLmWords,
              .tile_groups = plan.groups},
             note, [&](const LinearTile& tile) {
               const uint64_t offset = uint64_t{tile.first} * kWordF;
               const Tile residual =
                   program.Alloc(Mem::kFbuf, tile.quads * tile.groups);
               Ld(program, x[0] + offset, Mem::kFbuf, residual.At(), tile.quads,
                  tile.groups, kLmWords * kWordF,
                  std::format("{} ld residual t{}", note, tile.tile));
               program.Vec({.op = VecOp::kAdd,
                            .src = tile.c.At(),
                            .src2 = residual.At(),
                            .dst = tile.c.At(),
                            .rows_q = tile.quads,
                            .n8 = tile.groups},
                           std::format("{} residual t{}", note, tile.tile));
               St(program, x[1] + offset, Mem::kFbuf, tile.c.At(), tile.quads,
                  tile.groups, kLmWords * kWordF,
                  std::format("{} st x t{}", note, tile.tile));
             });
}

Tile GateUp(Program& program, const Plan& plan, const DecoderLayer& layer,
            const Tile& a, std::string_view note) {
  const Tile h =
      program.Alloc(Mem::kAbuf, kGateTiles * plan.quads * kGateBlocks);
  const Tile scratch = program.Alloc(Mem::kFbuf, plan.quads * kGateGroups);
  const auto product = [&](size_t linear) {
    return Linear{.planes = {{.w = layer.weights[linear]}},
                  .kb_total = kLmBlocks,
                  .pieces = {{.a = a.At(),
                              .a_stride = kLmBlocks,
                              .kb = kLmBlocks,
                              .kb_offset = 0}},
                  .quads = plan.quads,
                  .chunk_quads = plan.quads,
                  .groups = kHidden / 8,
                  .tile_groups = kGateGroups};
  };
  LinearEmitter gate(program, product(kGate), std::format("{} gate", note));
  LinearEmitter up(program, product(kUp), std::format("{} up", note));
  for (uint32_t j = 0; j < kGateTiles; ++j) {
    const LinearTile g = gate.Emit(j, 0);
    const LinearTile u = up.Emit(j, 0);
    program.Vec({.op = VecOp::kActMul,
                 .src = g.c.At(),
                 .src2 = u.c.At(),
                 .dst = h.At(j * plan.quads * kGateBlocks),
                 .rows_q = plan.quads,
                 .n8 = kGateGroups,
                 .q = true,
                 .scratch = scratch.At()},
                std::format("{} silu_mul+Q {}", note, j));
  }
  return h;
}

void Layer(Program& program, const Plan& plan, const Lm& lm, const Tile& rope,
           uint32_t l, const std::array<uint64_t, 3>& x, std::string_view tag) {
  const DecoderLayer& layer = lm.layers[l];
  const Tile pb = program.Alloc(Mem::kPbuf, 2 * kLmBlocks);
  Ld(program, layer.pbuf, Mem::kPbuf, pb.At(), 1, 2 * kLmBlocks,
     uint64_t{2} * kLmBlocks * kWordF,
     std::format("{} ld PBUF (RMSNorm gamma x 2)", tag));
  const Tile q = program.Alloc(Mem::kAbuf, kQueryHeads * 2 * plan.quads);
  const Tile a1 = Quantized(program, plan, x[0], VecOp::kRmsNorm, pb.At(kPbLn1),
                            std::format("{} RMSNorm1", tag));
  Qkv(program, plan, layer, a1, q, rope, lm.k[l], lm.v[l],
      std::format("{} qkv", tag));
  EmitAttention(
      program,
      {.queries = q,
       .k = lm.k[l],
       .v = lm.v[l],
       .o = lm.o,
       .heads = kQueryHeads,
       .kv_heads = kKvHeads,
       .rows = plan.tokens,
       .chunk_quads = plan.chunk,
       .embed = kLmWords * 8,
       .kv_blocks = kKvBlocks,
       .hd = kHeadWords * 8,
       .keys = plan.tokens,
       .valid = {{.quad = 0, .keys = plan.context},
                 {.quad = plan.context / 4, .keys = plan.context + 1}}},
      std::format("{} attn", tag));
  const Tile ao = Quantized(program, plan, lm.o, VecOp::kCopy, Place(),
                            std::format("{} quant O", tag));
  ProjResidual(
      program, plan,
      {{.a = ao.At(), .a_stride = kLmBlocks, .kb = kLmBlocks, .kb_offset = 0}},
      layer.weights[kO], kLmBlocks, {x[0], x[1]},
      std::format("{} o_proj", tag));
  const Tile a2 = Quantized(program, plan, x[1], VecOp::kRmsNorm, pb.At(kPbLn2),
                            std::format("{} RMSNorm2", tag));
  const Tile h = GateUp(program, plan, layer, a2, std::format("{} mlp", tag));
  std::vector<LinearPiece> pieces;
  pieces.reserve(kGateTiles);
  for (uint32_t j = 0; j < kGateTiles; ++j) {
    pieces.push_back({.a = h.At(j * plan.quads * kGateBlocks),
                      .a_stride = kGateBlocks,
                      .kb = kGateBlocks,
                      .kb_offset = j * kGateBlocks});
  }
  ProjResidual(program, plan, std::move(pieces), layer.weights[kDown],
               kKbHidden, {x[1], x[2]}, std::format("{} down", tag));
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
  lm.rope = PlaceData(image, "lm.rope", RopeTables(positions));
  lm.tbuf = PlaceData(image, "lm.tbuf", PackTbuf(MakeTbuf(Activation::kSilu)));
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
  Tile rope;
  {
    const Program::Scope scope(program, "setup");
    Ld(program, lm.tbuf, Mem::kTbuf, Place::Fixed(0), 1, kTbufWordsLoaded,
       kTbufWordsLoaded * kWordT, "lm ld TBUF (exp2 + silu)");
    rope = program.Alloc(Mem::kPbuf, rope_words);
    Ld(program, lm.rope, Mem::kPbuf, rope.At(), 1, rope_words,
       rope_words * kWordF, "lm ld PBUF cos/sin tiles");
  }
  for (uint32_t l = 0; l < kLayers; ++l) {
    const Program::Scope scope(program, std::format("L{}", l));
    const uint64_t x_in = l == 0 ? lm.x[0] : lm.x[1 + ((l - 1) % 2)];
    Layer(program, plan, lm, rope, l, {x_in, lm.x[3], lm.x[1 + (l % 2)]},
          std::format("T{}", l));
  }
}

}  // namespace hotarubi::compiler::smolvla
