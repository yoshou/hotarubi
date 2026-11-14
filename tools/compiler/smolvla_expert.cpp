#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ddr_image.hpp"
#include "hotarubi/exact.hpp"
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
constexpr uint32_t kRopeSelf = 0;
constexpr uint32_t kRopeCross = 128;
constexpr uint32_t kRopeWords = 256;
constexpr uint32_t kBiasIn = 0;
constexpr uint32_t kBiasMlpIn = 23;
constexpr uint32_t kBiasMlpOut = 46;
constexpr uint32_t kFinalGamma = 69;
constexpr uint32_t kBiasOut = 92;
constexpr uint32_t kHeadsWords = 93;
constexpr uint32_t kTileGroups = 28;
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

bool IsCross(uint32_t layer) { return layer % 2 == 1; }

uint64_t CacheRow(uint32_t prefix_slots) {
  return uint64_t{prefix_slots / 4} * kKvBlocks;
}

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
  out.pb_heads = PlaceData(image, "ex.pb_heads", PackPbuf(pbuf));
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
        PlaceData(image, std::format("{}.w{}", tag, kLinearTags[i]),
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
  out.pbuf = PlaceData(image, std::format("{}.pbuf", tag), PackPbuf(pbuf));
}

Linear Product(std::vector<LinearPlane> planes, uint32_t kb_total,
               std::vector<LinearPiece> pieces) {
  return {.planes = std::move(planes),
          .kb_total = kb_total,
          .pieces = std::move(pieces),
          .quads = kQuads,
          .chunk_quads = kQuads,
          .groups = kWords,
          .tile_groups = kTileGroups};
}

std::vector<LinearPiece> TiledPieces(const Tile& a) {
  std::vector<LinearPiece> pieces;
  pieces.reserve((kWords + kTileGroups - 1) / kTileGroups);
  for (uint32_t first = 0; first < kWords; first += kTileGroups) {
    const uint32_t blocks = std::min(kTileGroups, kWords - first) / 4;
    pieces.push_back({.a = a.At(kQuads * (first / 4)),
                      .a_stride = blocks,
                      .kb = blocks,
                      .kb_offset = first / 4});
  }
  return pieces;
}

std::vector<LinearPlane> W8Planes(const W8Region& region) {
  return {{.mode = WeightMode::kW8Hi, .w = region.hi},
          {.mode = WeightMode::kW8Lo, .w = region.lo}};
}

Tile RmsNormQ(Program& program, uint64_t x, Place gamma,
              std::string_view note) {
  const Tile a = program.Alloc(Mem::kAbuf, kQuads * kKb);
  EmitQuantize(program,
               {.x = x,
                .quads = kQuads,
                .words = kWords,
                .tile_quads = kQuads,
                .op = VecOp::kRmsNorm,
                .p_base = gamma,
                .aux = kWidth,
                .imm = RmsEps(),
                .dst = a.At()},
               note);
  return a;
}

void Qkv(Program& program, const DecoderLayer& layer, const Lm& lm, uint32_t l,
         uint32_t prefix_slots, const Tile& a, const Tile& q, const Tile& rope,
         std::string_view note) {
  const bool cross = IsCross(l);
  const uint32_t head = 2 * kQuads;
  const Tile scratch = program.Alloc(Mem::kFbuf, kQuads * kHeadWords);
  for (size_t linear = 0; linear < (cross ? 1U : 3U); ++linear) {
    const std::string name = std::format("{} {}", note, kLinearTags[linear]);
    EmitLinear(
        program,
        {.planes = {{.w = layer.weights[linear]}},
         .kb_total = kKb,
         .pieces = {{.a = a.At(), .a_stride = kKb, .kb = kKb, .kb_offset = 0}},
         .quads = kQuads,
         .chunk_quads = kQuads,
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
                         .rows_q = kQuads,
                         .n8 = kHeadWords,
                         .q = true},
                        std::format("{}{} quant", name, tile.tile));
          } else {
            program.Vec({.op = VecOp::kRope,
                         .src = tile.c.At(),
                         .dst = out,
                         .rows_q = kQuads,
                         .n8 = kHeadWords,
                         .p_base = rope.At(cross ? kRopeCross : kRopeSelf),
                         .aux = 1,
                         .q = true,
                         .scratch = scratch.At()},
                        std::format("{}{} rope+Q", name, tile.tile));
          }
          if (linear != 0) {
            St(program,
               (linear == 1 ? lm.k[l] : lm.v[l]) +
                   ((CacheRow(prefix_slots) + (uint64_t{2} * tile.tile)) *
                    kWordA),
               Mem::kAbuf, out, kQuads, 2, kKvBlocks * kWordA,
               std::format("{}{} st suffix", name, tile.tile));
          }
        });
  }
}

void ProjResidual(Program& program, std::vector<LinearPiece> pieces, uint64_t w,
                  uint32_t kb_total, const std::array<uint64_t, 2>& x,
                  std::string_view note) {
  EmitLinear(
      program, Product({{.w = w}}, kb_total, std::move(pieces)), note,
      [&](const LinearTile& tile) {
        const uint64_t offset = uint64_t{tile.first} * kWordF;
        const Tile residual = program.Alloc(Mem::kFbuf, kQuads * tile.groups);
        Ld(program, x[0] + offset, Mem::kFbuf, residual.At(), kQuads,
           tile.groups, kWords * kWordF,
           std::format("{} ld residual t{}", note, tile.tile));
        program.Vec({.op = VecOp::kAdd,
                     .src = tile.c.At(),
                     .src2 = residual.At(),
                     .dst = tile.c.At(),
                     .rows_q = kQuads,
                     .n8 = tile.groups},
                    std::format("{} residual t{}", note, tile.tile));
        St(program, x[1] + offset, Mem::kFbuf, tile.c.At(), kQuads, tile.groups,
           kWords * kWordF, std::format("{} st x t{}", note, tile.tile));
      });
}

Tile GateUp(Program& program, const DecoderLayer& layer, const Tile& a,
            std::string_view note) {
  const Tile h = program.Alloc(Mem::kAbuf, kGateTiles * kQuads * kGateBlocks);
  const Tile scratch = program.Alloc(Mem::kFbuf, kQuads * kGateGroups);
  const auto product = [&](size_t linear) {
    return Linear{
        .planes = {{.w = layer.weights[linear]}},
        .kb_total = kKb,
        .pieces = {{.a = a.At(), .a_stride = kKb, .kb = kKb, .kb_offset = 0}},
        .quads = kQuads,
        .chunk_quads = kQuads,
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
                 .dst = h.At(j * kQuads * kGateBlocks),
                 .rows_q = kQuads,
                 .n8 = kGateGroups,
                 .q = true,
                 .scratch = scratch.At()},
                std::format("{} silu_mul+Q {}", note, j));
  }
  return h;
}

void Layer(Program& program, const Expert& expert, const Lm& lm,
           const Dims& dims, uint32_t l, const Tile& rope,
           const std::array<uint64_t, 3>& x, std::string_view tag) {
  const DecoderLayer& layer = expert.layers[l];
  const bool cross = IsCross(l);
  const uint32_t keys = cross ? dims.Tokens() : dims.Tokens() + kSuffix;
  const Tile pb = program.Alloc(Mem::kPbuf, 2 * kKb);
  Ld(program, layer.pbuf, Mem::kPbuf, pb.At(), 1, 2 * kKb,
     uint64_t{2} * kKb * kWordF, std::format("{} ld PBUF (gamma x 2)", tag));
  const Tile q = program.Alloc(Mem::kAbuf, kQueryHeads * 2 * kQuads);
  const Tile a1 =
      RmsNormQ(program, x[0], pb.At(kPbLn1), std::format("{} RMSNorm1", tag));
  Qkv(program, layer, lm, l, dims.Tokens(), a1, q, rope,
      std::format("{} qkv", tag));
  EmitAttention(
      program,
      {.queries = q,
       .k = cross ? expert.kc[l] : lm.k[l],
       .v = cross ? expert.vc[l] : lm.v[l],
       .o = expert.o,
       .heads = kQueryHeads,
       .kv_heads = kKvHeads,
       .rows = kSuffix,
       .chunk_quads = kQuads,
       .embed = kQueryWords * 8,
       .kv_blocks = kKvBlocks,
       .hd = kHeadWords * 8,
       .keys = keys,
       .valid = {{.quad = 0, .keys = cross ? dims.Prefix() : keys}},
       .mask = cross ? std::nullopt : std::optional<uint64_t>(expert.mask)},
      std::format("{} attn", tag));
  const Tile ao = program.Alloc(Mem::kAbuf, kQuads * kKbQuery);
  EmitQuantize(program,
               {.x = expert.o,
                .quads = kQuads,
                .words = kQueryWords,
                .tile_quads = kQuads,
                .dst = ao.At()},
               std::format("{} quant O", tag));
  ProjResidual(
      program,
      {{.a = ao.At(), .a_stride = kKbQuery, .kb = kKbQuery, .kb_offset = 0}},
      layer.weights[kO], kKbQuery, {x[0], x[1]}, std::format("{} o_proj", tag));
  const Tile a2 =
      RmsNormQ(program, x[1], pb.At(kPbLn2), std::format("{} RMSNorm2", tag));
  const Tile h = GateUp(program, layer, a2, std::format("{} mlp", tag));
  std::vector<LinearPiece> pieces;
  pieces.reserve(kGateTiles);
  for (uint32_t j = 0; j < kGateTiles; ++j) {
    pieces.push_back({.a = h.At(j * kQuads * kGateBlocks),
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
  const std::array<uint64_t, 2> sources = {lm.k[l], lm.v[l]};
  const std::array<uint64_t, 2> outputs = {expert.kc[l], expert.vc[l]};
  for (uint32_t i = 0; i < 2; ++i) {
    const std::string note =
        std::format("E{} cross kv {}", l, i == 0 ? "k" : "v");
    const Tile a = program.Alloc(Mem::kAbuf, quads * kKvBlocks);
    Ld(program, sources.at(i), Mem::kAbuf, a.At(), quads, kKvBlocks,
       kKvBlocks * kWordA, std::format("{} ld cache", note));
    EmitLinear(program,
               {.planes = {{.w = expert.layers[l].weights[1 + i]}},
                .kb_total = kKvBlocks,
                .pieces = {{.a = a.At(),
                            .a_stride = kKvBlocks,
                            .kb = kKvBlocks,
                            .kb_offset = 0}},
                .quads = quads,
                .chunk_quads = quads,
                .groups = kKvGroups,
                .tile_groups = kKvGroups},
               note, [&](const LinearTile& tile) {
                 const Tile out = program.Alloc(Mem::kAbuf, quads * kKvBlocks);
                 program.Vec({.op = VecOp::kCopy,
                              .src = tile.c.At(),
                              .dst = out.At(),
                              .rows_q = quads,
                              .n8 = kKvGroups,
                              .q = true},
                             std::format("{} quant", note));
                 St(program, outputs.at(i), Mem::kAbuf, out.At(), quads,
                    kKvBlocks, kKvBlocks * kWordA, std::format("{} st", note));
               });
  }
}

void Suffix(Program& program, const Expert& expert, const Tile& heads,
            uint32_t step, uint64_t xt, uint64_t x0, std::string_view tag) {
  const auto bias = [&](const LinearTile& tile, uint32_t offset, Place dst,
                        bool q) {
    program.Vec({.op = VecOp::kAddP,
                 .src = tile.c.At(),
                 .dst = dst,
                 .rows_q = kQuads,
                 .n8 = tile.groups,
                 .p_base = heads.At(offset + (tile.first / 4)),
                 .q = q},
                std::format("{} bias{} t{}", tag, q ? "+Q" : "", tile.tile));
  };
  const Tile time = program.Alloc(Mem::kAbuf, kQuads * kKb);
  Ld(program, expert.time[step], Mem::kAbuf, time.At(), kQuads, kKb,
     kKb * kWordA, std::format("{} ld time embedding (MXINT8)", tag));
  const Tile action = program.Alloc(Mem::kAbuf, kQuads * (kActionWords / 4));
  EmitQuantize(program,
               {.x = xt,
                .quads = kQuads,
                .words = kActionWords,
                .tile_quads = kQuads,
                .dst = action.At()},
               std::format("{} quant x_t", tag));
  const Tile in = program.Alloc(Mem::kAbuf, kQuads * kKb);
  EmitLinear(
      program,
      Product(W8Planes(expert.heads[kHeadIn]), 1,
              {{.a = action.At(), .a_stride = 1, .kb = 1, .kb_offset = 0}}),
      std::format("{} action_in", tag), [&](const LinearTile& tile) {
        bias(tile, kBiasIn, in.At(kQuads * (tile.first / 4)), true);
      });
  const Tile hidden = program.Alloc(Mem::kAbuf, kQuads * kKb);
  const Tile scratch = program.Alloc(Mem::kFbuf, kQuads * kTileGroups);
  LinearEmitter of_action(
      program,
      Product(W8Planes(expert.heads[kHeadMlpInAction]), kKb, TiledPieces(in)),
      std::format("{} mlp_in(action)", tag));
  LinearEmitter of_time(
      program,
      Product(W8Planes(expert.heads[kHeadMlpInTime]), kKb,
              {{.a = time.At(), .a_stride = kKb, .kb = kKb, .kb_offset = 0}}),
      std::format("{} mlp_in(time)", tag));
  for (uint32_t t = 0; t < of_action.Tiles(); ++t) {
    const LinearTile tile = of_action.Emit(t, 0);
    of_time.Emit(t, 0, tile.c);
    bias(tile, kBiasMlpIn, tile.c.At(), false);
    program.Vec({.op = VecOp::kAct,
                 .src = tile.c.At(),
                 .dst = hidden.At(kQuads * (tile.first / 4)),
                 .rows_q = kQuads,
                 .n8 = tile.groups,
                 .q = true,
                 .scratch = scratch.At()},
                std::format("{} silu+Q t{}", tag, t));
  }
  EmitLinear(
      program,
      Product(W8Planes(expert.heads[kHeadMlpOut]), kKb, TiledPieces(hidden)),
      std::format("{} mlp_out", tag), [&](const LinearTile& tile) {
        bias(tile, kBiasMlpOut, tile.c.At(), false);
        St(program, x0 + (uint64_t{tile.first} * kWordF), Mem::kFbuf,
           tile.c.At(), kQuads, tile.groups, kWords * kWordF,
           std::format("{} st x0 t{}", tag, tile.tile));
      });
}

void StepEpilogue(Program& program, const Expert& expert, const Tile& heads,
                  uint32_t dt, const std::array<uint64_t, 4>& x,
                  std::string_view tag) {
  const Tile a = RmsNormQ(program, x[0], heads.At(kFinalGamma),
                          std::format("{} final norm", tag));
  const Tile xt = program.Alloc(Mem::kFbuf, kQuads * kActionWords);
  Ld(program, x[1], Mem::kFbuf, xt.At(), kQuads, kActionWords,
     kActionWords * kWordF, std::format("{} ld x_t", tag));
  EmitLinear(
      program,
      {.planes = W8Planes(expert.heads[kHeadOut]),
       .kb_total = kKb,
       .pieces = {{.a = a.At(), .a_stride = kKb, .kb = kKb, .kb_offset = 0}},
       .quads = kQuads,
       .chunk_quads = kQuads,
       .groups = kActionWords,
       .tile_groups = kActionWords},
      std::format("{} action_out", tag), [&](const LinearTile& tile) {
        program.Vec({.op = VecOp::kAddP,
                     .src = tile.c.At(),
                     .dst = tile.c.At(),
                     .rows_q = kQuads,
                     .n8 = kActionWords,
                     .p_base = heads.At(kBiasOut)},
                    std::format("{} action_out bias", tag));
        St(program, x[2], Mem::kFbuf, tile.c.At(), kQuads, kActionWords,
           kActionWords * kWordF, std::format("{} st v_t", tag));
        program.Vec({.op = VecOp::kScale,
                     .src = tile.c.At(),
                     .dst = tile.c.At(),
                     .rows_q = kQuads,
                     .n8 = kActionWords,
                     .imm = dt},
                    std::format("{} dt * v_t", tag));
        program.Vec({.op = VecOp::kAdd,
                     .src = tile.c.At(),
                     .src2 = xt.At(),
                     .dst = tile.c.At(),
                     .rows_q = kQuads,
                     .n8 = kActionWords},
                    std::format("{} x_t + dt * v_t", tag));
        St(program, x[3], Mem::kFbuf, tile.c.At(), kQuads, kActionWords,
           kActionWords * kWordF, std::format("{} st x_t next", tag));
      });
}

}  // namespace

Expert PlaceExpert(DdrImage& image, const SafeTensors& quant,
                   const SafeTensors& model, const Dims& dims) {
  Expert expert;
  PlaceHeads(image, model, expert);
  expert.rope = PlaceData(image, "ex.rope", ExpertRope(dims));
  expert.tbuf =
      PlaceData(image, "ex.tbuf", PackTbuf(MakeTbuf(Activation::kSilu)));
  expert.mask = PlaceData(image, "ex.mask", SelfMask(dims));
  for (uint32_t k = 0; k < dims.steps; ++k) {
    expert.time.push_back(PlaceData(image, std::format("ex.time{}", k),
                                    TimeInput(k, dims.steps)));
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
  Tile rope;
  Tile heads;
  {
    const Program::Scope scope(program, "setup");
    Ld(program, expert.tbuf, Mem::kTbuf, Place::Fixed(0), 1, kTbufWordsLoaded,
       kTbufWordsLoaded * kWordT, "ex ld TBUF (exp2 + silu)");
    rope = program.Alloc(Mem::kPbuf, kRopeWords);
    Ld(program, expert.rope, Mem::kPbuf, rope.At(), 1, kRopeWords,
       kRopeWords * kWordF, "ex ld PBUF cos/sin tiles (self / cross)");
    heads = program.Alloc(Mem::kPbuf, kHeadsWords);
    Ld(program, expert.pb_heads, Mem::kPbuf, heads.At(), 1, kHeadsWords,
       uint64_t{kHeadsWords} * kWordF,
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
      Suffix(program, expert, heads, k, expert.xt[k], expert.x[0],
             std::format("S{} suffix", k));
    }
    for (uint32_t l = 0; l < kLayers; ++l) {
      const Program::Scope scope(program, std::format("E{}", l));
      Layer(program, expert, lm, dims, l, rope,
            {expert.x[l % 2], expert.x[2], expert.x[(l + 1) % 2]},
            std::format("S{} E{}", k, l));
    }
    const Program::Scope scope(program, "out");
    StepEpilogue(
        program, expert, heads, dt,
        {expert.x[kLayers % 2], expert.xt[k], expert.vt[k], expert.xt[k + 1]},
        std::format("S{} out", k));
  }
}

}  // namespace hotarubi::compiler::smolvla
