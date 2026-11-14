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
#include "hotarubi/safetensors.hpp"
#include "isa.hpp"
#include "kernels.hpp"
#include "program.hpp"
#include "smolvla_parts.hpp"
#include "tensors.hpp"
#include "weights.hpp"

namespace hotarubi::compiler::smolvla {
namespace {

constexpr uint32_t kTokens = 1024;
constexpr uint32_t kWidth = 768;
constexpr uint32_t kHidden = 3072;
constexpr uint32_t kHeads = 12;
constexpr uint32_t kLayers = 12;
constexpr uint32_t kKb = kWidth / kBlock;
constexpr uint32_t kKbHidden = kHidden / kBlock;
constexpr uint32_t kWords = kWidth / 8;
constexpr uint32_t kQuads = kTokens / 4;
constexpr uint32_t kTileQuads = 8;
constexpr uint32_t kChunkQuads = 32;
constexpr uint32_t kQueryQuads = 8;
constexpr uint32_t kTileGroups = 32;
constexpr uint64_t kXBytes = uint64_t{kQuads} * kWords * kWordF;
constexpr uint64_t kPatchBytes = uint64_t{kQuads} * kKb * kWordA;
constexpr uint32_t kPbLn1 = 0;
constexpr uint32_t kPbLn2 = 48;
constexpr std::array<uint32_t, 6> kPbBias = {96, 120, 144, 168, 192, 288};
constexpr uint32_t kPbWords = 312;
constexpr size_t kQ = 0;
constexpr size_t kO = 3;
constexpr size_t kFc1 = 4;
constexpr size_t kFc2 = 5;
constexpr std::array<std::string_view, 6> kLinearModules = {
    "self_attn.q_proj",   "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.out_proj", "mlp.fc1",          "mlp.fc2"};
constexpr std::array<std::string_view, 6> kLinearTags = {"q", "k",   "v",
                                                         "o", "fc1", "fc2"};
constexpr uint32_t kShuffle = 4;
constexpr uint32_t kPieces = kShuffle * kShuffle;
constexpr uint32_t kConnectorQuads = kTokens / kPieces / 4;
constexpr uint32_t kConnectorBlocks = kKb * kPieces;
constexpr uint32_t kConnectorGroups = 30;
constexpr uint32_t kStateQuads = 8;
constexpr uint32_t kStateWords = 4;
constexpr std::string_view kVisionModel =
    "model.vlm_with_expert.vlm.model.vision_model";

uint32_t LnEps() {
  static const uint32_t kEps = Fp32OfFp64(exact::Real::Ratio(1, 1000000));
  return kEps;
}

uint32_t ConnectorScale() {
  static const uint32_t kScale =
      Fp32OfFp64(exact::Sqrt(exact::Real::Integer(int64_t{kLmWords} * 8)));
  return kScale;
}

std::vector<uint32_t> TokenOrder() {
  constexpr uint32_t kSide = 32;
  constexpr uint32_t kSideQuads = kSide / kShuffle;
  std::vector<uint32_t> order(kTokens);
  for (uint32_t hq = 0; hq < kSideQuads; ++hq) {
    for (uint32_t wq = 0; wq < kSideQuads; ++wq) {
      for (uint32_t i = 0; i < kShuffle; ++i) {
        for (uint32_t j = 0; j < kShuffle; ++j) {
          order[(((i * kShuffle) + j) * (kTokens / kPieces)) +
                (hq * kSideQuads) + wq] =
              (((kShuffle * hq) + i) * kSide) + (kShuffle * wq) + j;
        }
      }
    }
  }
  return order;
}

void PlaceLayer(DdrImage& image, const SafeTensors& quant,
                const SafeTensors& model, uint32_t layer, VisionLayer& out) {
  const std::string prefix =
      std::format("{}.encoder.layers.{}", kVisionModel, layer);
  const std::string tag = std::format("L{}", layer);
  std::vector<uint32_t> pbuf(static_cast<size_t>(kPbWords) * kBlock);
  for (size_t i = 0; i < kLinearModules.size(); ++i) {
    const std::string module = std::format("{}.{}", prefix, kLinearModules[i]);
    const Mxfp4Matrix raw = ReadMxfp4(quant, module);
    const int32_t shift = i == kQ ? -3 : 0;
    out.weights[i] =
        PlaceData(image, std::format("{}.w{}", tag, kLinearTags[i]),
                  PackWbuf(CanonicalMxfp4(raw, shift, raw.rows)));
    const std::vector<uint32_t> bias =
        model.Float32Bits(std::format("{}.bias", module));
    for (size_t c = 0; c < bias.size(); ++c) {
      pbuf[(size_t{kPbBias[i]} * kBlock) + c] =
          i == kQ ? exact::Real::Binary32(bias[c]).Scaled(shift).ToBinary32()
                  : bias[c];
    }
  }
  size_t at = 0;
  for (const std::string_view norm : {"layer_norm1", "layer_norm2"}) {
    for (const std::string_view part : {"weight", "bias"}) {
      for (const uint32_t value :
           model.Float32Bits(std::format("{}.{}.{}", prefix, norm, part))) {
        pbuf[at++] = value;
      }
    }
  }
  out.pbuf = PlaceData(image, std::format("{}.pbuf", tag), PackPbuf(pbuf));
}

void PlacePatch(DdrImage& image, const SafeTensors& model, uint32_t cameras,
                Vision& out) {
  const std::string embeddings = std::format("{}.embeddings", kVisionModel);
  const W8Region weights = PlaceW8(
      image, "patch.w",
      ReadMatrix(model, std::format("{}.patch_embedding.weight", embeddings)),
      kWidth);
  out.patch_hi = weights.hi;
  out.patch_lo = weights.lo;
  out.patch_pbuf = PlaceData(image, "patch.pbuf",
                             PackPbuf(model.Float32Bits(std::format(
                                 "{}.patch_embedding.bias", embeddings))));
  const Matrix pos = ReadMatrix(
      model, std::format("{}.position_embedding.weight", embeddings));
  Matrix ordered{.rows = kTokens, .cols = kWidth, .bits = {}};
  ordered.bits.reserve(pos.bits.size());
  for (const uint32_t token : TokenOrder()) {
    for (uint32_t c = 0; c < kWidth; ++c) {
      ordered.bits.push_back(pos.At(token, c));
    }
  }
  out.patch_pos = PlaceData(image, "patch.pos", PackFbuf(ordered));
  out.patch_in = image.Alloc("patch.in", cameras * kPatchBytes);
}

void PlaceState(DdrImage& image, const SafeTensors& model, Vision& out) {
  const W8Region weights =
      PlaceW8(image, "state.w", ReadMatrix(model, "model.state_proj.weight"),
              kLmWords * 8);
  out.state_hi = weights.hi;
  out.state_lo = weights.lo;
  out.state_pbuf =
      PlaceData(image, "state.pbuf",
                PackPbuf(model.Float32Bits("model.state_proj.bias")));
  out.state_in =
      image.Alloc("state.in", uint64_t{kStateQuads} * kStateWords * kWordF);
}

void PlaceConnector(DdrImage& image, const SafeTensors& quant,
                    const SafeTensors& model, Vision& out) {
  const Mxfp4Matrix raw = ReadMxfp4(
      quant,
      "model.vlm_with_expert.vlm.model.connector.modality_projection.proj");
  out.connector =
      PlaceData(image, "conn.w", PackWbuf(CanonicalMxfp4(raw, 0, raw.rows)));
  std::vector<uint32_t> pbuf =
      model.Float32Bits(std::format("{}.post_layernorm.weight", kVisionModel));
  for (const uint32_t value :
       model.Float32Bits(std::format("{}.post_layernorm.bias", kVisionModel))) {
    pbuf.push_back(value);
  }
  out.connector_pbuf = PlaceData(image, "conn.pbuf", PackPbuf(pbuf));
}

Tile LayerNorm(Program& program, uint64_t x, Place gamma,
               std::string_view note) {
  const Tile a = program.Alloc(Mem::kAbuf, kQuads * kKb);
  EmitQuantize(program,
               {.x = x,
                .quads = kQuads,
                .words = kWords,
                .tile_quads = kTileQuads,
                .op = VecOp::kLayerNorm,
                .p_base = gamma,
                .imm = LnEps(),
                .dst = a.At()},
               note);
  return a;
}

Linear Product(std::vector<LinearPlane> planes, const Tile& a,
               uint32_t groups) {
  return {.planes = std::move(planes),
          .kb_total = kKb,
          .pieces = {{.a = a.At(), .a_stride = kKb, .kb = kKb, .kb_offset = 0}},
          .quads = kQuads,
          .chunk_quads = kChunkQuads,
          .groups = groups,
          .tile_groups = kTileGroups};
}

void Projection(Program& program, const Tile& a, uint64_t w, Place bias,
                uint64_t out, bool gelu, std::string_view note) {
  const uint32_t groups = (gelu ? kHidden : kWidth) / 8;
  const Tile scratch =
      gelu ? program.Alloc(Mem::kFbuf, kChunkQuads * kTileGroups) : Tile();
  EmitLinear(
      program, Product({{.w = w}}, a, groups), note,
      [&](const LinearTile& tile) {
        const std::string where =
            std::format("{} t{} c{}", note, tile.tile, tile.chunk);
        const Tile q =
            program.Alloc(Mem::kAbuf, tile.quads * (tile.groups / 4));
        program.Vec({.op = VecOp::kAddP,
                     .src = tile.c.At(),
                     .dst = gelu ? tile.c.At() : q.At(),
                     .rows_q = tile.quads,
                     .n8 = tile.groups,
                     .p_base = bias + (tile.first / 4),
                     .q = !gelu},
                    std::format("{} bias", where));
        if (gelu) {
          program.Vec({.op = VecOp::kAct,
                       .src = tile.c.At(),
                       .dst = q.At(),
                       .rows_q = tile.quads,
                       .n8 = tile.groups,
                       .q = true,
                       .scratch = scratch.At()},
                      std::format("{} GELU+Q", where));
        }
        St(program,
           out + (((uint64_t{tile.quad} * (groups / 4)) + (tile.first / 4)) *
                  kWordA),
           Mem::kAbuf, q.At(), tile.quads, tile.groups / 4,
           (groups / 4) * kWordA, std::format("{} st", where));
      });
}

void BiasResidual(Program& program, const LinearTile& tile, uint32_t quad,
                  Place bias, uint64_t x_res, uint64_t x_out,
                  std::string_view note) {
  const std::string where =
      std::format("{} t{} q{}", note, tile.tile, quad + tile.quad);
  const uint64_t offset =
      ((uint64_t{quad + tile.quad} * kWords) + tile.first) * kWordF;
  const Tile residual = program.Alloc(Mem::kFbuf, tile.quads * tile.groups);
  Ld(program, x_res + offset, Mem::kFbuf, residual.At(), tile.quads,
     tile.groups, kWords * kWordF, std::format("{} ld residual", where));
  program.Vec({.op = VecOp::kAddP,
               .src = tile.c.At(),
               .dst = tile.c.At(),
               .rows_q = tile.quads,
               .n8 = tile.groups,
               .p_base = bias + (tile.first / 4)},
              std::format("{} bias", where));
  program.Vec({.op = VecOp::kAdd,
               .src = tile.c.At(),
               .src2 = residual.At(),
               .dst = tile.c.At(),
               .rows_q = tile.quads,
               .n8 = tile.groups},
              std::format("{} residual", where));
  St(program, x_out + offset, Mem::kFbuf, tile.c.At(), tile.quads, tile.groups,
     kWords * kWordF, std::format("{} st x", where));
}

void Fc2Residual(Program& program, uint64_t h, uint64_t w, Place bias,
                 uint64_t x_res, uint64_t x_out, std::string_view note) {
  for (uint32_t quad = 0; quad < kQuads; quad += kChunkQuads) {
    const Tile a = program.Alloc(Mem::kAbuf, kChunkQuads * kKbHidden);
    Ld(program, h + (uint64_t{quad} * kKbHidden * kWordA), Mem::kAbuf, a.At(),
       kChunkQuads, kKbHidden, kKbHidden * kWordA,
       std::format("{} ld H q{}", note, quad));
    std::vector<LinearPiece> pieces;
    pieces.reserve(kKbHidden / kKb);
    for (uint32_t kb = 0; kb < kKbHidden; kb += kKb) {
      pieces.push_back(
          {.a = a.At(kb), .a_stride = kKbHidden, .kb = kKb, .kb_offset = kb});
    }
    EmitLinear(program,
               {.planes = {{.w = w}},
                .kb_total = kKbHidden,
                .pieces = std::move(pieces),
                .quads = kChunkQuads,
                .chunk_quads = kChunkQuads,
                .groups = kWords,
                .tile_groups = kTileGroups},
               std::format("{} q{}", note, quad), [&](const LinearTile& tile) {
                 BiasResidual(program, tile, quad, bias, x_res, x_out, note);
               });
  }
}

void Layer(Program& program, const Vision& vision, const VisionLayer& layer,
           const std::array<uint64_t, 3>& x, std::string_view tag) {
  const Tile pb = program.Alloc(Mem::kPbuf, kPbWords);
  Ld(program, layer.pbuf, Mem::kPbuf, pb.At(), 1, kPbWords, kPbWords * kWordF,
     std::format("{} ld PBUF", tag));
  {
    Tile a;
    {
      const Program::Scope scope(program, "ln1");
      a = LayerNorm(program, x[0], pb.At(kPbLn1), std::format("{} LN1", tag));
    }
    const std::array<uint64_t, 3> outputs = {vision.q, vision.k, vision.v};
    for (size_t i = 0; i < outputs.size(); ++i) {
      const Program::Scope scope(program,
                                 std::format("{}_proj", kLinearTags[i]));
      Projection(program, a, layer.weights[i], pb.At(kPbBias[i]), outputs[i],
                 false, std::format("{} {}_proj", tag, kLinearTags[i]));
    }
  }
  {
    const Program::Scope scope(program, "attn");
    EmitAttention(program,
                  {.q = vision.q,
                   .k = vision.k,
                   .v = vision.v,
                   .o = vision.o,
                   .heads = kHeads,
                   .kv_heads = kHeads,
                   .rows = kTokens,
                   .chunk_quads = kQueryQuads,
                   .embed = kWidth,
                   .kv_blocks = kKb,
                   .hd = kWidth / kHeads,
                   .keys = kTokens,
                   .valid = {{.quad = 0, .keys = kTokens}}},
                  std::format("{} attn", tag));
  }
  {
    const Tile a = program.Alloc(Mem::kAbuf, kQuads * kKb);
    {
      const Program::Scope scope(program, "quant_o");
      EmitQuantize(program,
                   {.x = vision.o,
                    .quads = kQuads,
                    .words = kWords,
                    .tile_quads = kTileQuads,
                    .dst = a.At()},
                   std::format("{} quant O", tag));
    }
    const Program::Scope scope(program, "out_proj");
    const std::string note = std::format("{} out_proj", tag);
    EmitLinear(program, Product({{.w = layer.weights[kO]}}, a, kWords), note,
               [&](const LinearTile& tile) {
                 BiasResidual(program, tile, 0, pb.At(kPbBias[kO]), x[0], x[1],
                              note);
               });
  }
  {
    Tile a;
    {
      const Program::Scope scope(program, "ln2");
      a = LayerNorm(program, x[1], pb.At(kPbLn2), std::format("{} LN2", tag));
    }
    const Program::Scope scope(program, "fc1");
    Projection(program, a, layer.weights[kFc1], pb.At(kPbBias[kFc1]), vision.h,
               true, std::format("{} fc1", tag));
  }
  const Program::Scope scope(program, "fc2");
  Fc2Residual(program, vision.h, layer.weights[kFc2], pb.At(kPbBias[kFc2]),
              x[1], x[2], std::format("{} fc2", tag));
}

void PatchCamera(Program& program, const Vision& vision, const Tile& bias,
                 uint32_t camera, std::string_view note) {
  const Tile a = program.Alloc(Mem::kAbuf, kQuads * kKb);
  Ld(program, vision.patch_in + (camera * kPatchBytes), Mem::kAbuf, a.At(),
     kQuads, kKb, kKb * kWordA, std::format("{} ld patches (MXINT8)", note));
  EmitLinear(program,
             Product({{.mode = WeightMode::kW8Hi, .w = vision.patch_hi},
                      {.mode = WeightMode::kW8Lo, .w = vision.patch_lo}},
                     a, kWords),
             note, [&](const LinearTile& tile) {
               BiasResidual(program, tile, 0, bias.At(), vision.patch_pos,
                            vision.x[0] + (camera * kXBytes), note);
             });
}

void StateProjection(Program& program, const Vision& vision, const Dims& dims,
                     uint64_t lm_input) {
  const uint32_t row = dims.Prefix() - 1;
  Require(row % 4 == 0,
          std::format("the state token {} does not start a quad", row));
  const Tile pb = program.Alloc(Mem::kPbuf, kLmBlocks);
  Ld(program, vision.state_pbuf, Mem::kPbuf, pb.At(), 1, kLmBlocks,
     kLmBlocks * kWordF, "state ld PBUF (bias)");
  const Tile a = program.Alloc(Mem::kAbuf, kStateQuads * (kStateWords / 4));
  EmitQuantize(program,
               {.x = vision.state_in,
                .quads = kStateQuads,
                .words = kStateWords,
                .tile_quads = kStateQuads,
                .dst = a.At()},
               "state");
  EmitLinear(program,
             {.planes = {{.mode = WeightMode::kW8Hi, .w = vision.state_hi},
                         {.mode = WeightMode::kW8Lo, .w = vision.state_lo}},
              .kb_total = 1,
              .pieces = {{.a = a.At(), .a_stride = 1, .kb = 1, .kb_offset = 0}},
              .quads = kStateQuads,
              .chunk_quads = kStateQuads,
              .groups = kLmWords,
              .tile_groups = kLmWords},
             "state", [&](const LinearTile& tile) {
               program.Vec({.op = VecOp::kAddP,
                            .src = tile.c.At(),
                            .dst = tile.c.At(),
                            .rows_q = kStateQuads,
                            .n8 = kLmWords,
                            .p_base = pb.At()},
                           "state bias");
               St(program, lm_input + (uint64_t{row / 4} * kLmWords * kWordF),
                  Mem::kFbuf, tile.c.At(), 1, kLmWords, kLmWords * kWordF,
                  "state st state token");
             });
}

void Connector(Program& program, const Vision& vision, uint64_t x, uint64_t out,
               std::string_view note) {
  const Tile pb = program.Alloc(Mem::kPbuf, 2 * kKb);
  Ld(program, vision.connector_pbuf, Mem::kPbuf, pb.At(), 1, 2 * kKb,
     uint64_t{2} * kKb * kWordF, std::format("{} ld PBUF (post LN)", note));
  const Tile a =
      LayerNorm(program, x, pb.At(), std::format("{} post LN", note));
  std::vector<LinearPiece> pieces;
  pieces.reserve(kPieces);
  for (uint32_t m = 0; m < kPieces; ++m) {
    pieces.push_back({.a = a.At(m * kConnectorQuads * kKb),
                      .a_stride = kKb,
                      .kb = kKb,
                      .kb_offset = m * kKb});
  }
  EmitLinear(program,
             {.planes = {{.w = vision.connector}},
              .kb_total = kConnectorBlocks,
              .pieces = std::move(pieces),
              .quads = kConnectorQuads,
              .chunk_quads = kConnectorQuads,
              .groups = kLmWords,
              .tile_groups = kConnectorGroups},
             note, [&](const LinearTile& tile) {
               program.Vec({.op = VecOp::kScale,
                            .src = tile.c.At(),
                            .dst = tile.c.At(),
                            .rows_q = tile.quads,
                            .n8 = tile.groups,
                            .imm = ConnectorScale()},
                           std::format("{} scale t{}", note, tile.tile));
               St(program, out + (uint64_t{tile.first} * kWordF), Mem::kFbuf,
                  tile.c.At(), tile.quads, tile.groups, kLmWords * kWordF,
                  std::format("{} st t{}", note, tile.tile));
             });
}

}  // namespace

Vision PlaceVision(DdrImage& image, const SafeTensors& quant,
                   const SafeTensors& model, const Dims& dims) {
  Vision vision;
  vision.tbuf = PlaceData(image, "tbuf", PackTbuf(MakeTbuf(Activation::kGelu)));
  PlacePatch(image, model, dims.cameras, vision);
  PlaceState(image, model, vision);
  vision.layers.resize(kLayers);
  for (uint32_t l = 0; l < kLayers; ++l) {
    PlaceLayer(image, quant, model, l, vision.layers[l]);
  }
  PlaceConnector(image, quant, model, vision);
  const uint64_t qkv = uint64_t{kQuads} * kKb * kWordA;
  vision.q = image.Alloc("work.Q", qkv);
  vision.k = image.Alloc("work.K", qkv);
  vision.v = image.Alloc("work.V", qkv);
  vision.o = image.Alloc("work.O", kXBytes);
  vision.h = image.Alloc("work.H", uint64_t{kQuads} * kKbHidden * kWordA);
  vision.x[0] = image.Alloc("xv.0", dims.cameras * kXBytes);
  for (uint32_t i = 1; i < vision.x.size(); ++i) {
    vision.x[i] = image.Alloc(std::format("xv.{}", i), kXBytes);
  }
  return vision;
}

void EmitVision(Program& program, const Vision& vision, const Dims& dims,
                uint64_t lm_input) {
  const Program::Scope vision_scope(program, "vision");
  Tile patch_bias;
  {
    const Program::Scope scope(program, "setup");
    Ld(program, vision.tbuf, Mem::kTbuf, Place::Fixed(0), 1, kTbufWordsLoaded,
       kTbufWordsLoaded * kWordT, "ld TBUF (exp2 + gelu)");
    patch_bias = program.Alloc(Mem::kPbuf, kKb);
    Ld(program, vision.patch_pbuf, Mem::kPbuf, patch_bias.At(), 1, kKb,
       kKb * kWordF, "patch ld PBUF (bias)");
  }
  for (uint32_t c = 0; c < dims.cameras; ++c) {
    const Program::Scope scope(program, std::format("patch/c{}", c));
    PatchCamera(program, vision, patch_bias, c, std::format("c{} patch", c));
  }
  {
    const Program::Scope scope(program, "state");
    StateProjection(program, vision, dims, lm_input);
  }
  for (uint32_t c = 0; c < dims.cameras; ++c) {
    const Program::Scope camera(program, std::format("c{}", c));
    for (uint32_t l = 0; l < kLayers; ++l) {
      const Program::Scope scope(program, std::format("L{}", l));
      const uint64_t x_in =
          l == 0 ? vision.x[0] + (c * kXBytes) : vision.x[1 + ((l - 1) % 2)];
      Layer(program, vision, vision.layers[l],
            {x_in, vision.x[3], vision.x[1 + (l % 2)]},
            std::format("c{} L{}", c, l));
    }
    const Program::Scope scope(program, "conn");
    Connector(program, vision, vision.x[1 + ((kLayers - 1) % 2)],
              lm_input + (uint64_t{c} * (kImageTokens / 4) * kLmWords * kWordF),
              std::format("c{} conn", c));
  }
}

}  // namespace hotarubi::compiler::smolvla
