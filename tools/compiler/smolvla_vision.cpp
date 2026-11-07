#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
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
constexpr uint32_t kChunks = kQuads / kChunkQuads;
constexpr uint32_t kQueryQuads = 8;
constexpr uint32_t kTileGroups = 32;
constexpr uint32_t kWidthTiles = kWidth / (8 * kTileGroups);
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
constexpr uint32_t kConnectorTiles = kLmWords / kConnectorGroups;
constexpr uint32_t kStateQuads = 8;
constexpr uint32_t kStateWords = 4;
constexpr uint32_t kStateC = 64;
constexpr uint32_t kStateLo = 120;
constexpr uint32_t kPatchLo = 768;
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

uint64_t TileOffset(uint32_t nt, uint32_t mc) {
  return ((uint64_t{mc} * kChunkQuads * kWords) +
          (uint64_t{nt} * kTileGroups)) *
         kWordF;
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
    out.weights[i] = Place(image, std::format("{}.w{}", tag, kLinearTags[i]),
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
  out.pbuf = Place(image, std::format("{}.pbuf", tag), PackPbuf(pbuf));
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
  out.patch_pbuf = Place(image, "patch.pbuf",
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
  out.patch_pos = Place(image, "patch.pos", PackFbuf(ordered));
  out.patch_in = image.Alloc("patch.in", cameras * kPatchBytes);
}

void PlaceState(DdrImage& image, const SafeTensors& model, Vision& out) {
  const W8Region weights =
      PlaceW8(image, "state.w", ReadMatrix(model, "model.state_proj.weight"),
              kLmWords * 8);
  out.state_hi = weights.hi;
  out.state_lo = weights.lo;
  out.state_pbuf = Place(image, "state.pbuf",
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
      Place(image, "conn.w", PackWbuf(CanonicalMxfp4(raw, 0, raw.rows)));
  std::vector<uint32_t> pbuf =
      model.Float32Bits(std::format("{}.post_layernorm.weight", kVisionModel));
  for (const uint32_t value :
       model.Float32Bits(std::format("{}.post_layernorm.bias", kVisionModel))) {
    pbuf.push_back(value);
  }
  out.connector_pbuf = Place(image, "conn.pbuf", PackPbuf(pbuf));
}

void LayerNorm(Program& program, uint64_t x, uint32_t pb, uint32_t dst,
               std::string_view note) {
  constexpr std::array<uint32_t, 2> kXt = {1792, 2560};
  constexpr uint32_t kScratch = 3328;
  for (uint32_t t = 0; t < kQuads / kTileQuads; ++t) {
    Ld(program, x + (uint64_t{t} * kTileQuads * kWords * kWordF), Mem::kFbuf,
       kXt[t % 2], kTileQuads, kWords, kWords * kWordF,
       std::format("{} ld x tile {}", note, t));
    program.Vec({.op = VecOp::kLayerNorm,
                 .src = kXt[t % 2],
                 .dst = dst + (t * kTileQuads * kKb),
                 .rows_q = kTileQuads,
                 .n8 = kWords,
                 .p_base = pb,
                 .imm = LnEps(),
                 .q = true,
                 .scratch = kScratch},
                std::format("{} LN tile {}", note, t));
  }
}

void LoadWeightTile(Program& program, uint64_t w, uint32_t nt, uint32_t at,
                    std::string_view note) {
  Ld(program, w + (uint64_t{nt} * kTileGroups * kKb * kWordA), Mem::kWbuf, at,
     kTileGroups, kKb, kKb * kWordA, std::format("{} ld W tile {}", note, nt));
}

void Projection(Program& program, uint64_t w, uint32_t pb, uint64_t out,
                bool gelu, std::string_view note) {
  constexpr std::array<uint32_t, 2> kCb = {0, 1024};
  constexpr uint32_t kScratch = 2048;
  constexpr std::array<uint32_t, 2> kWt = {0, 768};
  constexpr std::array<uint32_t, 2> kTmp = {6144, 6400};
  const uint32_t n = gelu ? kHidden : kWidth;
  const uint32_t tiles = n / (8 * kTileGroups);
  const uint32_t kbo = n / kBlock;
  const uint32_t count = tiles * kChunks;
  const auto post_vec = [&](uint32_t j) {
    const uint32_t nt = j / kChunks;
    const uint32_t mc = j % kChunks;
    const uint32_t c = kCb[j % 2];
    if (gelu) {
      program.Vec({.op = VecOp::kAddP,
                   .src = c,
                   .dst = c,
                   .rows_q = kChunkQuads,
                   .n8 = kTileGroups,
                   .p_base = pb + (nt * 8)},
                  std::format("{} bias nt{} mc{}", note, nt, mc));
      program.Vec({.op = VecOp::kGelu,
                   .src = c,
                   .dst = kTmp[j % 2],
                   .rows_q = kChunkQuads,
                   .n8 = kTileGroups,
                   .q = true,
                   .scratch = kScratch},
                  std::format("{} GELU+Q nt{} mc{}", note, nt, mc));
    } else {
      program.Vec({.op = VecOp::kAddP,
                   .src = c,
                   .dst = kTmp[j % 2],
                   .rows_q = kChunkQuads,
                   .n8 = kTileGroups,
                   .p_base = pb + (nt * 8),
                   .q = true},
                  std::format("{} bias+Q nt{} mc{}", note, nt, mc));
    }
  };
  const auto post_st = [&](uint32_t j) {
    const uint32_t nt = j / kChunks;
    const uint32_t mc = j % kChunks;
    St(program,
       out +
           (((uint64_t{mc} * kChunkQuads * kbo) + (uint64_t{nt} * 8)) * kWordA),
       Mem::kAbuf, kTmp[j % 2], kChunkQuads, 8, kbo * kWordA,
       std::format("{} st nt{} mc{}", note, nt, mc));
  };
  LoadWeightTile(program, w, 0, kWt[0], note);
  for (uint32_t j = 0; j < count; ++j) {
    const uint32_t nt = j / kChunks;
    const uint32_t mc = j % kChunks;
    if (j >= 1) {
      post_vec(j - 1);
    }
    program.Gemm({.a_base = mc * kChunkQuads * kKb,
                  .a_stride = kKb,
                  .kb = kKb,
                  .m = kChunkQuads * 4,
                  .w_base = kWt[nt % 2],
                  .w_stride = kKb,
                  .n8 = kTileGroups,
                  .c_base = kCb[j % 2]},
                 std::format("{} gemm nt{} mc{}", note, nt, mc));
    if (mc == 0 && nt + 1 < tiles) {
      LoadWeightTile(program, w, nt + 1, kWt[(nt + 1) % 2], note);
    }
    if (j >= 1) {
      post_st(j - 1);
    }
  }
  post_vec(count - 1);
  post_st(count - 1);
}

class Attention {
 public:
  Attention(Program& program, const Vision& vision, std::string_view note)
      : program_(program), vision_(vision), note_(note) {}

  void Emit() {
    for (uint32_t which = 0; which < 3; ++which) {
      Prefetch(0, which);
    }
    for (uint32_t g = 0; g < kTotal; ++g) {
      const uint32_t h = g / kQueryChunks;
      const uint32_t c = g % kQueryChunks;
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
      if (c < 3 && h + 1 < kHeads) {
        Prefetch(h + 1, c);
      }
    }
    Softmax(kTotal - 1);
    StoreOutput(kTotal - 2);
    Values(kTotal - 1);
    StoreOutput(kTotal - 1);
  }

 private:
  static constexpr std::array<uint32_t, 2> kKv = {0, 1024};
  static constexpr std::array<uint32_t, 2> kQh = {0, 512};
  static constexpr std::array<uint32_t, 2> kP = {1024, 1280};
  static constexpr std::array<uint32_t, 2> kS = {0, 1024};
  static constexpr uint32_t kScratch = 2048;
  static constexpr std::array<uint32_t, 2> kO = {3072, 3136};
  static constexpr uint32_t kQueryChunks = kQuads / kQueryQuads;
  static constexpr uint32_t kTotal = kHeads * kQueryChunks;
  static constexpr uint32_t kPlane = 256;

  void Prefetch(uint32_t h, uint32_t which) {
    const uint64_t column = uint64_t{2} * h * kWordA;
    const uint32_t kv = kKv[h % 2];
    if (which == 0) {
      Ld(program_, vision_.k + column, Mem::kWbuf, kv, kQuads, 2, kKb * kWordA,
         std::format("{} ld K head {}", note_, h), DmaMode::kSplit8);
    } else if (which == 1) {
      Ld(program_, vision_.v + column, Mem::kWbuf, kv + (2 * kPlane), kQuads, 2,
         kKb * kWordA, std::format("{} ld V head {}", note_, h),
         DmaMode::kT32Split8);
    } else {
      Ld(program_, vision_.q + column, Mem::kAbuf, kQh[h % 2], kQuads, 2,
         kKb * kWordA, std::format("{} ld Q head {}", note_, h));
    }
  }

  void Scores(uint32_t g) {
    const uint32_t h = g / kQueryChunks;
    const uint32_t c = g % kQueryChunks;
    const uint32_t a = kQh[h % 2] + (c * kQueryQuads * 2);
    for (const bool lo : {false, true}) {
      program_.Gemm(
          {.a_base = a,
           .a_stride = 2,
           .kb = 2,
           .m = kQueryQuads * 4,
           .w_base = kKv[h % 2] + (lo ? kPlane : 0),
           .w_stride = 2,
           .n8 = kTokens / 8,
           .c_base = kS[g % 2],
           .wmode = lo ? WeightMode::kW8Lo : WeightMode::kW8Hi,
           .acc = lo},
          std::format("{} S {} h{} c{}", note_, lo ? "lo" : "hi", h, c));
    }
  }

  void Softmax(uint32_t g) {
    program_.Vec({.op = VecOp::kSoftmax,
                  .src = kS[g % 2],
                  .dst = kP[g % 2],
                  .rows_q = kQueryQuads,
                  .n8 = kTokens / 8,
                  .aux = kTokens,
                  .q = true,
                  .scratch = kScratch},
                 std::format("{} softmax+Q h{} c{}", note_, g / kQueryChunks,
                             g % kQueryChunks));
  }

  void Values(uint32_t g) {
    const uint32_t h = g / kQueryChunks;
    for (const bool lo : {false, true}) {
      program_.Gemm({.a_base = kP[g % 2],
                     .a_stride = kTokens / kBlock,
                     .kb = kTokens / kBlock,
                     .m = kQueryQuads * 4,
                     .w_base = kKv[h % 2] + ((lo ? 3 : 2) * kPlane),
                     .w_stride = kTokens / kBlock,
                     .n8 = 8,
                     .c_base = kO[g % 2],
                     .wmode = lo ? WeightMode::kW8Lo : WeightMode::kW8Hi,
                     .acc = lo},
                    std::format("{} PV {} h{} c{}", note_, lo ? "lo" : "hi", h,
                                g % kQueryChunks));
    }
  }

  void StoreOutput(uint32_t g) {
    const uint32_t h = g / kQueryChunks;
    const uint32_t c = g % kQueryChunks;
    St(program_,
       vision_.o + (((uint64_t{c} * kQueryQuads * kWords) + (uint64_t{h} * 8)) *
                    kWordF),
       Mem::kFbuf, kO[g % 2], kQueryQuads, 8, kWords * kWordF,
       std::format("{} st O h{} c{}", note_, h, c));
  }

  Program& program_;
  const Vision& vision_;
  std::string note_;
};

void QuantToAbuf(Program& program, uint64_t x, uint32_t dst,
                 std::string_view note) {
  constexpr std::array<uint32_t, 2> kXt = {1792, 2560};
  for (uint32_t t = 0; t < kQuads / kTileQuads; ++t) {
    Ld(program, x + (uint64_t{t} * kTileQuads * kWords * kWordF), Mem::kFbuf,
       kXt[t % 2], kTileQuads, kWords, kWords * kWordF,
       std::format("{} ld tile {}", note, t));
    program.Vec({.op = VecOp::kQuant,
                 .src = kXt[t % 2],
                 .dst = dst + (t * kTileQuads * kKb),
                 .rows_q = kTileQuads,
                 .n8 = kWords},
                std::format("{} quant tile {}", note, t));
  }
}

void BiasResidual(Program& program, uint32_t c, uint32_t xr, uint32_t pb,
                  std::string_view where) {
  program.Vec({.op = VecOp::kAddP,
               .src = c,
               .dst = c,
               .rows_q = kChunkQuads,
               .n8 = kTileGroups,
               .p_base = pb},
              std::format("{} bias", where));
  program.Vec({.op = VecOp::kAdd,
               .src = c,
               .src2 = xr,
               .dst = c,
               .rows_q = kChunkQuads,
               .n8 = kTileGroups},
              std::format("{} residual", where));
}

void OutProjResidual(Program& program, uint64_t w, uint32_t pb, uint64_t x_res,
                     uint64_t x_out, std::string_view note) {
  constexpr std::array<uint32_t, 2> kCb = {0, 1024};
  constexpr std::array<uint32_t, 2> kXr = {2048, 3072};
  constexpr std::array<uint32_t, 2> kWt = {0, 768};
  constexpr uint32_t kCount = kWidthTiles * kChunks;
  const auto where = [&](uint32_t j) {
    return std::format("{} nt{} mc{}", note, j / kChunks, j % kChunks);
  };
  LoadWeightTile(program, w, 0, kWt[0], note);
  for (uint32_t j = 0; j < kCount; ++j) {
    const uint32_t nt = j / kChunks;
    const uint32_t mc = j % kChunks;
    if (j >= 1) {
      BiasResidual(program, kCb[(j - 1) % 2], kXr[(j - 1) % 2],
                   pb + (((j - 1) / kChunks) * 8), where(j - 1));
    }
    program.Gemm({.a_base = mc * kChunkQuads * kKb,
                  .a_stride = kKb,
                  .kb = kKb,
                  .m = kChunkQuads * 4,
                  .w_base = kWt[nt % 2],
                  .w_stride = kKb,
                  .n8 = kTileGroups,
                  .c_base = kCb[j % 2]},
                 std::format("{} gemm", where(j)));
    if (mc == 0 && nt + 1 < kWidthTiles) {
      LoadWeightTile(program, w, nt + 1, kWt[(nt + 1) % 2], note);
    }
    Ld(program, x_res + TileOffset(nt, mc), Mem::kFbuf, kXr[j % 2], kChunkQuads,
       kTileGroups, kWords * kWordF, std::format("{} ld residual", where(j)));
    if (j >= 1) {
      St(program, x_out + TileOffset((j - 1) / kChunks, (j - 1) % kChunks),
         Mem::kFbuf, kCb[(j - 1) % 2], kChunkQuads, kTileGroups,
         kWords * kWordF, std::format("{} st x", where(j - 1)));
    }
  }
  BiasResidual(program, kCb[(kCount - 1) % 2], kXr[(kCount - 1) % 2],
               pb + (((kCount - 1) / kChunks) * 8), where(kCount - 1));
  St(program, x_out + TileOffset(kWidthTiles - 1, kChunks - 1), Mem::kFbuf,
     kCb[(kCount - 1) % 2], kChunkQuads, kTileGroups, kWords * kWordF,
     std::format("{} st x", where(kCount - 1)));
}

class Fc2Residual {
 public:
  Fc2Residual(Program& program, uint64_t h, uint64_t w, uint32_t pb,
              std::string_view note)
      : program_(program), h_(h), w_(w), pb_(pb), note_(note) {}

  void Emit(uint64_t x_res, uint64_t x_out) {
    for (uint32_t kc = 0; kc < kSlices; ++kc) {
      LoadHidden(0, kc);
    }
    LoadWeight(0);
    for (uint32_t j = 0; j < kCount; ++j) {
      const uint32_t mc = j / kWidthTiles;
      const uint32_t nt = j % kWidthTiles;
      if (j >= 1) {
        Post(j - 1);
      }
      for (uint32_t kc = 0; kc < kSlices; ++kc) {
        const uint32_t wt = (j * kSlices) + kc;
        program_.Gemm({.a_base = kHa[mc % 2] + (kc * kChunkQuads * kKb),
                       .a_stride = kKb,
                       .kb = kKb,
                       .m = kChunkQuads * 4,
                       .w_base = kWt[wt % 2],
                       .w_stride = kKb,
                       .n8 = kTileGroups,
                       .c_base = kCb[j % 2],
                       .acc = kc > 0},
                      std::format("{} gemm mc{} nt{} kc{}", note_, mc, nt, kc));
        if (wt + 1 < kCount * kSlices) {
          LoadWeight(wt + 1);
        }
        if (nt == 0 && mc + 1 < kChunks) {
          LoadHidden(mc + 1, kc);
        }
      }
      Ld(program_, x_res + TileOffset(nt, mc), Mem::kFbuf, kXr[j % 2],
         kChunkQuads, kTileGroups, kWords * kWordF,
         std::format("{} ld residual mc{} nt{}", note_, mc, nt));
      if (j >= 1) {
        Store(j - 1, x_out);
      }
    }
    Post(kCount - 1);
    Store(kCount - 1, x_out);
  }

 private:
  static constexpr uint32_t kSlices = kHidden / kWidth;
  static constexpr std::array<uint32_t, 2> kHa = {0, 3072};
  static constexpr std::array<uint32_t, 2> kCb = {0, 1024};
  static constexpr std::array<uint32_t, 2> kXr = {2048, 3072};
  static constexpr std::array<uint32_t, 2> kWt = {0, 768};
  static constexpr uint32_t kCount = kChunks * kWidthTiles;

  void LoadWeight(uint32_t wt) {
    const uint32_t nt = (wt / kSlices) % kWidthTiles;
    const uint32_t kc = wt % kSlices;
    Ld(program_,
       w_ + (((uint64_t{nt} * kTileGroups * kKbHidden) + (uint64_t{kc} * kKb)) *
             kWordA),
       Mem::kWbuf, kWt[wt % 2], kTileGroups, kKb, kKbHidden * kWordA,
       std::format("{} ld W nt{} kc{}", note_, nt, kc));
  }

  void LoadHidden(uint32_t mc, uint32_t kc) {
    Ld(program_,
       h_ + (((uint64_t{mc} * kChunkQuads * kKbHidden) + (uint64_t{kc} * kKb)) *
             kWordA),
       Mem::kAbuf, kHa[mc % 2] + (kc * kChunkQuads * kKb), kChunkQuads, kKb,
       kKbHidden * kWordA,
       std::format("{} ld H chunk {} piece {}", note_, mc, kc));
  }

  void Post(uint32_t j) {
    BiasResidual(
        program_, kCb[j % 2], kXr[j % 2], pb_ + ((j % kWidthTiles) * 8),
        std::format("{} mc{} nt{}", note_, j / kWidthTiles, j % kWidthTiles));
  }

  void Store(uint32_t j, uint64_t x_out) {
    const uint32_t mc = j / kWidthTiles;
    const uint32_t nt = j % kWidthTiles;
    St(program_, x_out + TileOffset(nt, mc), Mem::kFbuf, kCb[j % 2],
       kChunkQuads, kTileGroups, kWords * kWordF,
       std::format("{} st x mc{} nt{}", note_, mc, nt));
  }

  Program& program_;
  uint64_t h_;
  uint64_t w_;
  uint32_t pb_;
  std::string note_;
};

void Layer(Program& program, const Vision& vision, const VisionLayer& layer,
           const std::array<uint64_t, 3>& x, std::string_view tag) {
  constexpr uint32_t kA = 0;
  {
    const Program::Scope scope(program, "ln1");
    LayerNorm(program, x[0], kPbLn1, kA, std::format("{} LN1", tag));
  }
  const std::array<uint64_t, 3> outputs = {vision.q, vision.k, vision.v};
  for (size_t i = 0; i < outputs.size(); ++i) {
    const Program::Scope scope(program, std::format("{}_proj", kLinearTags[i]));
    Projection(program, layer.weights[i], kPbBias[i], outputs[i], false,
               std::format("{} {}_proj", tag, kLinearTags[i]));
  }
  {
    const Program::Scope scope(program, "attn");
    Attention(program, vision, std::format("{} attn", tag)).Emit();
  }
  {
    const Program::Scope scope(program, "quant_o");
    QuantToAbuf(program, vision.o, kA, std::format("{} quant O", tag));
  }
  {
    const Program::Scope scope(program, "out_proj");
    OutProjResidual(program, layer.weights[kO], kPbBias[kO], x[0], x[1],
                    std::format("{} out_proj", tag));
  }
  {
    const Program::Scope scope(program, "ln2");
    LayerNorm(program, x[1], kPbLn2, kA, std::format("{} LN2", tag));
  }
  {
    const Program::Scope scope(program, "fc1");
    Projection(program, layer.weights[kFc1], kPbBias[kFc1], vision.h, true,
               std::format("{} fc1", tag));
  }
  const Program::Scope scope(program, "fc2");
  Fc2Residual(program, vision.h, layer.weights[kFc2], kPbBias[kFc2],
              std::format("{} fc2", tag))
      .Emit(x[1], x[2]);
}

void PatchCamera(Program& program, const Vision& vision, uint32_t camera,
                 std::string_view note) {
  constexpr std::array<uint32_t, 2> kCb = {0, 1024};
  constexpr std::array<uint32_t, 2> kXr = {2048, 3072};
  constexpr uint32_t kCount = kWidthTiles * kChunks;
  const uint64_t x_out = vision.x[0] + (camera * kXBytes);
  const auto where = [&](uint32_t j) {
    return std::format("{} nt{} mc{}", note, j / kChunks, j % kChunks);
  };
  const auto post_vec = [&](uint32_t j) {
    program.Vec({.op = VecOp::kAddP,
                 .src = kCb[j % 2],
                 .dst = kCb[j % 2],
                 .rows_q = kChunkQuads,
                 .n8 = kTileGroups,
                 .p_base = (j / kChunks) * 8},
                std::format("{} bias", where(j)));
    program.Vec({.op = VecOp::kAdd,
                 .src = kCb[j % 2],
                 .src2 = kXr[j % 2],
                 .dst = kCb[j % 2],
                 .rows_q = kChunkQuads,
                 .n8 = kTileGroups},
                std::format("{} + pos", where(j)));
  };
  const auto post_st = [&](uint32_t j) {
    St(program, x_out + TileOffset(j / kChunks, j % kChunks), Mem::kFbuf,
       kCb[j % 2], kChunkQuads, kTileGroups, kWords * kWordF,
       std::format("{} st x", where(j)));
  };
  Ld(program, vision.patch_in + (camera * kPatchBytes), Mem::kAbuf, 0, kQuads,
     kKb, kKb * kWordA, std::format("{} ld patches (MXINT8)", note));
  for (uint32_t j = 0; j < kCount; ++j) {
    const uint32_t nt = j / kChunks;
    const uint32_t mc = j % kChunks;
    if (j >= 1) {
      post_vec(j - 1);
    }
    if (mc == 0) {
      LoadWeightTile(program, vision.patch_hi, nt, 0,
                     std::format("{} hi", note));
      LoadWeightTile(program, vision.patch_lo, nt, kPatchLo,
                     std::format("{} lo", note));
    }
    for (const bool lo : {false, true}) {
      program.Gemm({.a_base = mc * kChunkQuads * kKb,
                    .a_stride = kKb,
                    .kb = kKb,
                    .m = kChunkQuads * 4,
                    .w_base = lo ? kPatchLo : 0,
                    .w_stride = kKb,
                    .n8 = kTileGroups,
                    .c_base = kCb[j % 2],
                    .wmode = lo ? WeightMode::kW8Lo : WeightMode::kW8Hi,
                    .acc = lo},
                   std::format("{} gemm {}", where(j), lo ? "lo" : "hi"));
    }
    Ld(program, vision.patch_pos + TileOffset(nt, mc), Mem::kFbuf, kXr[j % 2],
       kChunkQuads, kTileGroups, kWords * kWordF,
       std::format("{} ld pos", where(j)));
    if (j >= 1) {
      post_st(j - 1);
    }
  }
  post_vec(kCount - 1);
  post_st(kCount - 1);
}

void StateProjection(Program& program, const Vision& vision, const Dims& dims,
                     uint64_t lm_input) {
  const uint32_t row = dims.Prefix() - 1;
  Require(row % 4 == 0,
          std::format("the state token {} does not start a quad", row));
  Ld(program, vision.state_pbuf, Mem::kPbuf, 0, 1, kLmBlocks,
     kLmBlocks * kWordF, "state ld PBUF (bias)");
  Ld(program, vision.state_in, Mem::kFbuf, 0, kStateQuads, kStateWords,
     kStateWords * kWordF, "state ld state");
  program.Vec({.op = VecOp::kQuant,
               .src = 0,
               .dst = 0,
               .rows_q = kStateQuads,
               .n8 = kStateWords},
              "state quant");
  Ld(program, vision.state_hi, Mem::kWbuf, 0, kLmWords, 1, kWordA,
     "state ld W hi");
  Ld(program, vision.state_lo, Mem::kWbuf, kStateLo, kLmWords, 1, kWordA,
     "state ld W lo");
  for (const bool lo : {false, true}) {
    program.Gemm({.a_base = 0,
                  .a_stride = 1,
                  .kb = 1,
                  .m = kStateQuads * 4,
                  .w_base = lo ? kStateLo : 0,
                  .w_stride = 1,
                  .n8 = kLmWords,
                  .c_base = kStateC,
                  .wmode = lo ? WeightMode::kW8Lo : WeightMode::kW8Hi,
                  .acc = lo},
                 lo ? "state gemm lo" : "state gemm hi");
  }
  program.Vec({.op = VecOp::kAddP,
               .src = kStateC,
               .dst = kStateC,
               .rows_q = kStateQuads,
               .n8 = kLmWords},
              "state bias");
  St(program, lm_input + (uint64_t{row / 4} * kLmWords * kWordF), Mem::kFbuf,
     kStateC, 1, kLmWords, kLmWords * kWordF, "state st state token");
}

void Connector(Program& program, const Vision& vision, uint64_t x, uint64_t out,
               std::string_view note) {
  constexpr std::array<uint32_t, 2> kWt = {0, 720};
  constexpr std::array<uint32_t, 2> kCb = {0, 480};
  constexpr uint32_t kCount = kConnectorTiles * kPieces;
  Ld(program, vision.connector_pbuf, Mem::kPbuf, 0, 1, 2 * kKb,
     uint64_t{2} * kKb * kWordF, std::format("{} ld PBUF (post LN)", note));
  LayerNorm(program, x, 0, 0, std::format("{} post LN", note));
  const auto load_w = [&](uint32_t wt) {
    const uint32_t nt = wt / kPieces;
    const uint32_t m = wt % kPieces;
    Ld(program,
       vision.connector +
           (((uint64_t{nt} * kConnectorGroups * kConnectorBlocks) +
             (uint64_t{m} * kKb)) *
            kWordA),
       Mem::kWbuf, kWt[wt % 2], kConnectorGroups, kKb,
       kConnectorBlocks * kWordA, std::format("{} ld W nt{} m{}", note, nt, m));
  };
  load_w(0);
  for (uint32_t nt = 0; nt < kConnectorTiles; ++nt) {
    for (uint32_t m = 0; m < kPieces; ++m) {
      const uint32_t wt = (nt * kPieces) + m;
      program.Gemm({.a_base = m * kConnectorQuads * kKb,
                    .a_stride = kKb,
                    .kb = kKb,
                    .m = kConnectorQuads * 4,
                    .w_base = kWt[wt % 2],
                    .w_stride = kKb,
                    .n8 = kConnectorGroups,
                    .c_base = kCb[nt % 2],
                    .acc = m > 0},
                   std::format("{} gemm nt{} m{}", note, nt, m));
      if (wt + 1 < kCount) {
        load_w(wt + 1);
      }
    }
    program.Vec({.op = VecOp::kScale,
                 .src = kCb[nt % 2],
                 .dst = kCb[nt % 2],
                 .rows_q = kConnectorQuads,
                 .n8 = kConnectorGroups,
                 .imm = ConnectorScale()},
                std::format("{} scale nt{}", note, nt));
    St(program, out + (uint64_t{nt} * kConnectorGroups * kWordF), Mem::kFbuf,
       kCb[nt % 2], kConnectorQuads, kConnectorGroups, kLmWords * kWordF,
       std::format("{} st nt{}", note, nt));
  }
}

}  // namespace

Vision PlaceVision(DdrImage& image, const SafeTensors& quant,
                   const SafeTensors& model, const Dims& dims) {
  Vision vision;
  vision.tbuf = Place(image, "tbuf", PackTbuf(MakeTbuf(Activation::kGelu)));
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
  {
    const Program::Scope scope(program, "setup");
    Ld(program, vision.tbuf, Mem::kTbuf, 0, 1, kTbufWordsLoaded,
       kTbufWordsLoaded * kWordT, "ld TBUF (exp2 + gelu)");
    Ld(program, vision.patch_pbuf, Mem::kPbuf, 0, 1, kKb, kKb * kWordF,
       "patch ld PBUF (bias)");
  }
  for (uint32_t c = 0; c < dims.cameras; ++c) {
    const Program::Scope scope(program, std::format("patch/c{}", c));
    PatchCamera(program, vision, c, std::format("c{} patch", c));
  }
  {
    const Program::Scope scope(program, "state");
    StateProjection(program, vision, dims, lm_input);
  }
  for (uint32_t c = 0; c < dims.cameras; ++c) {
    const Program::Scope camera(program, std::format("c{}", c));
    for (uint32_t l = 0; l < kLayers; ++l) {
      const Program::Scope scope(program, std::format("L{}", l));
      const std::string tag = std::format("c{} L{}", c, l);
      Ld(program, vision.layers[l].pbuf, Mem::kPbuf, 0, 1, kPbWords,
         kPbWords * kWordF, std::format("{} ld PBUF", tag));
      const uint64_t x_in =
          l == 0 ? vision.x[0] + (c * kXBytes) : vision.x[1 + ((l - 1) % 2)];
      Layer(program, vision, vision.layers[l],
            {x_in, vision.x[3], vision.x[1 + (l % 2)]}, tag);
    }
    const Program::Scope scope(program, "conn");
    Connector(program, vision, vision.x[1 + ((kLayers - 1) % 2)],
              lm_input + (uint64_t{c} * (kImageTokens / 4) * kLmWords * kWordF),
              std::format("c{} conn", c));
  }
}

}  // namespace hotarubi::compiler::smolvla
