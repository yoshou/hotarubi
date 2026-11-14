#include "sparsedrive_head.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "ddr_image.hpp"
#include "hotarubi/exact.hpp"
#include "isa.hpp"
#include "kernels.hpp"
#include "program.hpp"
#include "sparsedrive_kernels.hpp"
#include "sparsedrive_layers.hpp"
#include "sparsedrive_parts.hpp"
#include "tensors.hpp"
#include "vprog.hpp"
#include "weights.hpp"

namespace hotarubi::compiler::sparsedrive {
namespace {

constexpr uint32_t kDetTile = 116;
constexpr uint32_t kDetGatherQuads = 2;
constexpr uint32_t kQuantQuads = 32;
constexpr uint32_t kPackWords = 512;
constexpr uint32_t kDetLearn = 6;
constexpr uint32_t kGroups = 8;
constexpr int kCbufKeypoints = 1;
constexpr int kCbufBank = 2;
constexpr int kCbufProject = 3;

constexpr HeadConfig kDetConfig{.det = true,
                                .na = 900,
                                .nap = 928,
                                .ntemp = 600,
                                .ntp = 608,
                                .acols = 32,
                                .adim = 11,
                                .dq = 512,
                                .hd = 64,
                                .npt = 13,
                                .scam = 56};
constexpr HeadConfig kMapConfig{.det = false,
                                .na = 100,
                                .nap = 128,
                                .ntemp = 33,
                                .ntp = 64,
                                .acols = 64,
                                .adim = 40,
                                .dq = 256,
                                .hd = 32,
                                .npt = 300,
                                .scam = 1200};

std::vector<Op> HeadOrder(bool det) {
  const std::vector<Op> first = {Op::kGnn, Op::kNorm, Op::kDeformable,
                                 Op::kFfn, Op::kNorm, Op::kRefine};
  const std::vector<Op> later = {Op::kTempGnn,    Op::kGnn, Op::kNorm,
                                 Op::kDeformable, Op::kFfn, Op::kNorm,
                                 Op::kRefine};
  std::vector<Op> order = first;
  for (int i = 0; i < 5; ++i) {
    order.insert(order.end(), later.begin(), later.end());
  }
  if (det) {
    order.erase(order.begin(), order.begin() + 2);
  }
  return order;
}

std::string_view OpName(Op op) {
  switch (op) {
    case Op::kGnn:
      return "gnn";
    case Op::kTempGnn:
      return "temp_gnn";
    case Op::kCrossGnn:
      return "cross_gnn";
    case Op::kNorm:
      return "norm";
    case Op::kDeformable:
      return "deformable";
    case Op::kFfn:
      return "ffn";
    case Op::kRefine:
      return "refine";
  }
  return "";
}

std::vector<int64_t> Inverse(const std::vector<int64_t>& perm, size_t size) {
  std::vector<int64_t> inverse(size, -1);
  for (size_t i = 0; i < perm.size(); ++i) {
    inverse.at(static_cast<size_t>(perm[i])) = static_cast<int64_t>(i);
  }
  return inverse;
}

HeadLayer DeformableLayer(const HeadWeights& weights, const HeadConfig& c,
                          const std::string& module) {
  HeadLayer layer;
  layer.op = Op::kDeformable;
  const std::string lfc = module + ".kps_generator.learnable_fc";
  const std::vector<uint32_t> lfc_bias = weights.Values(lfc + ".bias");
  if (c.det) {
    const std::vector<int64_t> rows = Inverse(LfcPerm(kDetLearn), 24);
    layer.lfc = Lin::W8(SelectRows(weights.W8Rows(lfc + ".weight"), rows),
                        SelectValues(lfc_bias, rows, 0), 32, 0);
  } else {
    const std::vector<int64_t> rows = Inverse(MapLfcPerm(), kMapLfcCols);
    layer.lfc = Lin::Fp4(SelectRows(weights.Fp4Rows(lfc), rows),
                         SelectValues(lfc_bias, rows, 0), 32);
  }
  for (const auto& [linear, norm] : {std::pair{0, 2}, {3, 5}}) {
    layer.cam.emplace_back(
        weights.Linear(std::format("{}.camera_encoder.{}", module, linear)),
        weights.Norm(std::format("{}.camera_encoder.{}", module, norm)));
  }
  const Mxfp4Matrix wfc = weights.Fp4Rows(module + ".weights_fc");
  const std::vector<uint32_t> wfc_bias =
      weights.Values(module + ".weights_fc.bias");
  const uint32_t used = kLevels * c.npt;
  for (uint32_t g = 0; g < kGroups; ++g) {
    std::vector<int64_t> rows(c.scam, -1);
    for (uint32_t j = 0; j < used; ++j) {
      rows[j] = (int64_t{j} * kGroups) + g;
    }
    layer.wfc.push_back(
        Lin::Fp4(SelectRows(wfc, rows),
                 SelectValues(wfc_bias, rows, NegativeHugeBits()), 8));
  }
  layer.oproj = weights.Linear(module + ".output_proj");
  if (c.det) {
    layer.fix_scale = weights.Values(module + ".kps_generator.fix_scale");
  }
  return layer;
}

HeadLayer RefineLayer(const HeadWeights& weights, const HeadConfig& c,
                      const std::string& module) {
  HeadLayer layer;
  layer.op = Op::kRefine;
  for (const int i : {0, 2, 5, 7}) {
    layer.reg.push_back(weights.Linear(std::format("{}.layers.{}", module, i)));
  }
  layer.reg_ln = {weights.Norm(module + ".layers.4"),
                  weights.Norm(module + ".layers.9")};
  layer.reg_out = weights.Linear(module + ".layers.10", 32);
  layer.scale = weights.Values(module + ".layers.11.scale");
  layer.cls = {weights.Linear(module + ".cls_layers.0"),
               weights.Linear(module + ".cls_layers.3")};
  layer.cls_ln = {weights.Norm(module + ".cls_layers.2"),
                  weights.Norm(module + ".cls_layers.5")};
  layer.cls_out = weights.Linear(module + ".cls_layers.6", 32);
  if (c.det) {
    layer.qt = {weights.Linear(module + ".quality_layers.0"),
                weights.Linear(module + ".quality_layers.3")};
    layer.qt_ln = {weights.Norm(module + ".quality_layers.2"),
                   weights.Norm(module + ".quality_layers.5")};
    layer.qt_out = weights.Linear(module + ".quality_layers.6", 32);
  }
  return layer;
}

Word NullWord() {
  Word w{};
  w.fill(kNullEntry);
  return w;
}

}  // namespace

HeadParams ReadHeadParams(const HeadWeights& weights, bool det) {
  const std::string head = det ? "head.det_head" : "head.map_head";
  HeadParams params;
  params.config = det ? kDetConfig : kMapConfig;
  const HeadConfig& c = params.config;
  params.anchor0 = weights.Values(head + ".instance_bank.anchor");
  params.x0 = weights.Values(head + ".instance_bank.instance_feature");
  struct Branch {
    std::string_view name;
    uint32_t lo;
    uint32_t hi;
    uint32_t col0;
  };
  const std::vector<Branch> branches =
      det ? std::vector<Branch>{{.name = "pos", .lo = 0, .hi = 3, .col0 = 0},
                                {.name = "size", .lo = 3, .hi = 6, .col0 = 128},
                                {.name = "yaw", .lo = 6, .hi = 8, .col0 = 160},
                                {.name = "vel", .lo = 8, .hi = 11, .col0 = 192}}
          : std::vector<Branch>{{.name = "pos", .lo = 0, .hi = 40, .col0 = 0}};
  const uint32_t depth = det ? 4 : 2;
  for (const Branch& b : branches) {
    EncoderBranch branch{.name = std::string(b.name),
                         .lo = b.lo,
                         .hi = b.hi,
                         .col0 = b.col0,
                         .chain = {}};
    for (uint32_t i = 0; i < depth; ++i) {
      const std::string fc =
          std::format("{}.anchor_encoder.{}_fc", head, b.name);
      branch.chain.emplace_back(
          weights.Linear(std::format("{}.{}", fc, 3 * i), 8,
                         det && i == 0 ? b.lo : 0),
          weights.Norm(std::format("{}.{}", fc, (3 * i) + 2)));
    }
    params.enc.push_back(std::move(branch));
  }
  if (det) {
    params.fc_before = weights.Linear(head + ".fc_before");
    params.fc_after = weights.Linear(head + ".fc_after");
  }
  const std::vector<Op> order = HeadOrder(det);
  for (size_t i = 0; i < order.size(); ++i) {
    const std::string module = std::format("{}.layers.{}", head, i);
    HeadLayer layer;
    layer.op = order[i];
    switch (order[i]) {
      case Op::kGnn:
      case Op::kTempGnn:
        layer.attn = weights.Attention(module + ".attn", c.dq, c.hd == 64);
        params.layers.push_back(std::move(layer));
        break;
      case Op::kNorm:
        layer.ln = weights.Norm(module);
        params.layers.push_back(std::move(layer));
        break;
      case Op::kDeformable:
        params.layers.push_back(DeformableLayer(weights, c, module));
        break;
      case Op::kFfn:
        layer.ln = weights.Norm(module + ".pre_norm");
        layer.fc1 = weights.Linear(module + ".layers.0.0");
        layer.fc2 = weights.Linear(module + ".layers.1");
        layer.idf = weights.Linear(module + ".identity_fc");
        params.layers.push_back(std::move(layer));
        break;
      case Op::kRefine:
        params.layers.push_back(RefineLayer(weights, c, module));
        break;
      case Op::kCrossGnn:
        Fail("a detection or map head has no cross attention");
    }
  }
  return params;
}

SparseHead::SparseHead(DdrImage& image, uint64_t base, HeadParams params,
                       const std::vector<std::vector<Level>>& levels,
                       uint64_t w_off)
    : p(std::move(params)),
      tag(p.config.det ? "det" : "map"),
      ncam(static_cast<uint32_t>(levels.size())),
      image_(image),
      base_(base) {
  const HeadConfig& cf = C();
  kbq = cf.dq / 32;
  nwq = cf.dq / 8;
  s = ncam * cf.scam;
  for (uint32_t i = 0; i < 3; ++i) {
    x[i] = PlaceRows(image, std::format("{}.x{}", tag, i), cf.nap, kD);
  }
  e = PlaceRows(image, tag + ".E", cf.nap, kD);
  tx = PlaceRows(image, tag + ".temp_x", cf.ntp, kD);
  for (uint32_t i = 0; i < 2; ++i) {
    anc[i] =
        PlaceRows(image, std::format("{}.anchor{}", tag, i), cf.nap, cf.acols);
  }
  tanc = PlaceRows(image, tag + ".temp_anchor", cf.ntp, cf.acols);
  x_init = PlaceRows(image, tag + ".x_init", cf.nap, kD, 0);
  a_init = PlaceRows(image, tag + ".anchor_init", cf.nap, cf.acols, 0);
  PutRows(image, x_init, cf.na, kD, p.x0);
  PutRows(image, a_init, cf.na, cf.adim, p.anchor0);
  const auto deformable =
      std::ranges::find(p.layers, Op::kDeformable, &HeadLayer::op);
  lfc = PlaceRows(image, tag + ".lfc", cf.nap, deformable->lfc.Rows());
  cls = PlaceRows(image, tag + ".cls", cf.nap, 32);
  if (cf.det) {
    qt = PlaceRows(image, tag + ".qt", cf.nap, 32);
  }
  cam_in = PlaceRows(image, tag + ".cam_in", 32, 32, 0);
  cam_vec = PlaceRows(image, tag + ".cam_vec", 32, kD, 0);
  const uint64_t abuf_bytes = uint64_t{cf.nap / 4} * kbq * kWordA;
  qb = image.Alloc(tag + ".Q", abuf_bytes);
  kb = image.Alloc(tag + ".K", abuf_bytes);
  vb = image.Alloc(tag + ".V", abuf_bytes);
  ob = image.Alloc(tag + ".O", uint64_t{cf.nap / 4} * nwq * kWordF);
  cat = image.Alloc(tag + ".cat", uint64_t{cf.nap / 4} * 64 * kWordF);
  nq = (cf.na + 3) / 4;
  wq = s / 8;
  wg = nq * wq;
  w = w_off;
  lbc = 16 + (uint64_t{8} * cf.scam * 16);
  lists_bytes = uint64_t{nq} * ncam * lbc;
  lists = image.Alloc(tag + ".gacc", lists_bytes);
  for (uint32_t i = 0; i < 2; ++i) {
    order[i] =
        image.Alloc(std::format("{}.order{}", tag, i), uint64_t{cf.nap} * 4);
  }
  for (uint32_t i = 0; i < 2; ++i) {
    order_use[i] = image.Alloc(std::format("{}.order_use{}", tag, i),
                               uint64_t{cf.nap} * 4);
  }
  keep = image.Alloc(tag + ".keep", uint64_t{cf.ntp} * 4);
  const std::vector<std::byte> table = PackTbuf(MakeTbuf(Activation::kSilu));
  tb = image.Alloc(tag + ".tbuf", table.size());
  image.Put(tb, table);
  ncls = std::ranges::find_last(p.layers, Op::kRefine, &HeadLayer::op)
             .begin()
             ->cls_out.N();
  if (cf.det) {
    kp.Add("init", DetKeypointInit(kDetTile));
    kp.Add("point", DetKeypointPoint(kDetTile, kDetLearn));
    kp.Add("level", DetKeypointLevel(kDetTile));
  } else {
    kp.Add("init", MapKeypointInit(cf.nap / 4));
    kp.Add("point", MapKeypointPoint(cf.nap / 4));
    kp.Add("level", MapKeypointLevel(cf.nap / 4));
  }
  if (wq > 255) {
    kp.Add("ls", LongSoftmaxKernel());
  }
  const std::vector<std::byte> kp_bytes = Serialize(kp.Words());
  vprog = image.Alloc(tag + ".vprog", kp_bytes.size());
  image.Put(vprog, kp_bytes);
  kp_const = image.Alloc(tag + ".kp_const", uint64_t{ncam} * kWordF);
  PutWords(image, kp_const, std::vector<Word>(ncam, ConstWord()));
  kp_lv = image.Alloc(tag + ".kp_lv", uint64_t{ncam} * kWordF);
  std::vector<Word> level_words;
  level_words.reserve(levels.size());
  for (const auto& camera : levels) {
    level_words.push_back(LevelWord(camera));
  }
  PutWords(image, kp_lv, level_words);
  for (size_t l = 0; l < p.layers.size(); ++l) {
    if (p.layers[l].op == Op::kDeformable && cf.det) {
      kp_fs[l] = image.Alloc(std::format("{}.kp_fs{}", tag, l), kWordF);
      PutWords(image, kp_fs[l], {FixScaleWord(p.layers[l].fix_scale)});
    }
  }
  if (!cf.det) {
    kp_zh = image.Alloc(tag + ".kp_zh", kWordF);
    PutWords(image, kp_zh, {ZhWord()});
  }
  nw = cf.nap / 32;
  ib.Add("update", InstanceBankKernel(cf.nap, cf.na, cf.ntemp,
                                      BankMode::kUpdate, cf.na - cf.ntemp));
  ib.Add("cache", InstanceBankKernel(cf.nap, cf.na, cf.ntemp, BankMode::kCache,
                                     cf.ntemp));
  ib.Add("copy", MaskKernel(false));
  const std::vector<std::byte> ib_bytes = Serialize(ib.Words());
  ibprog = image.Alloc(tag + ".ibprog", ib_bytes.size());
  image.Put(ibprog, ib_bytes);
  ib_idx = image.Alloc(tag + ".ib_idx", uint64_t{nw} * kWordF);
  PutWords(image, ib_idx, LaneIndexWords(nw));
  ib_lane = image.Alloc(tag + ".ib_lane", kWordF);
  PutWords(image, ib_lane, {LaneIdWord()});
  prevconf = image.Alloc(tag + ".prevconf", uint64_t{nw} * kWordF);
  Word ninf{};
  ninf.fill(0xFF800000U);
  PutWords(image, prevconf, std::vector<Word>(nw, ninf));
  selidx = image.Alloc(tag + ".selidx", uint64_t{nw} * kWordF);
  keep_a = image.Alloc(tag + ".keep_a", uint64_t{nw} * kWordF);
  PutWords(image, keep_a, std::vector<Word>(nw, NullWord()));
  tanc_raw = PlaceRows(image, tag + ".temp_anchor_raw", cf.ntp, cf.acols);
  proj_pb = image.Alloc(tag + ".proj_pb", 18 * kWordF);
  proj_dt = image.Alloc(tag + ".proj_dt", kWordF);
  pj.Add("project", AnchorProjectKernel(cf.det, cf.acols / 8));
  const std::vector<std::byte> pj_bytes = Serialize(pj.Words());
  pjprog = image.Alloc(tag + ".pjprog", pj_bytes.size());
  image.Put(pjprog, pj_bytes);
  for (uint32_t k = 0; k < 2; ++k) {
    const Rows& rows = k == 0 ? tx : tanc;
    std::vector<uint32_t> entries(cf.nap, kNullEntry);
    for (uint32_t i = 0; i < cf.ntemp; ++i) {
      entries[i] = Entry(base_, rows.Addr(i));
    }
    PutEntries(image, order[k], entries);
  }
  PutEntries(image, keep, std::vector<uint32_t>(cf.ntp, kNullEntry));
  const auto block = [&](uint32_t q, uint32_t camera, uint32_t slots) {
    std::vector<uint32_t> entries = {
        Entry(base_, w) + (((camera * (cf.scam / 8)) + (q * wq)) * 8), 0, 0, 0};
    for (uint32_t i = 0; i < slots * 16; ++i) {
      entries.push_back(kNullEntry);
      entries.push_back(0);
    }
    return entries;
  };
  std::vector<uint32_t> all;
  for (uint32_t q = 0; q < nq; ++q) {
    for (uint32_t camera = 0; camera < ncam; ++camera) {
      const std::vector<uint32_t> one = block(q, camera, cf.scam);
      all.insert(all.end(), one.begin(), one.end());
    }
  }
  PutEntries(image, lists, all);
  null_list = image.Alloc(tag + ".gacc_null", 16 + (8 * 16));
  PutEntries(image, null_list, block(0, 0, 1));
  PlaceParams();
}

namespace {

void PlaceLayerWeights(DdrImage& image, HeadLayer& layer, const std::string& at,
                       bool det) {
  switch (layer.op) {
    case Op::kGnn:
    case Op::kTempGnn:
      for (size_t i = 0; i < 4; ++i) {
        layer.attn[i].Place(image, at + std::string(1, "qkvo"[i]));
      }
      break;
    case Op::kDeformable:
      layer.lfc.Place(image, at + "lfc");
      for (size_t i = 0; i < layer.cam.size(); ++i) {
        layer.cam[i].first.Place(image, std::format("{}cam{}", at, i));
      }
      for (size_t g = 0; g < layer.wfc.size(); ++g) {
        layer.wfc[g].Place(image, std::format("{}wfc{}", at, g));
      }
      layer.oproj.Place(image, at + "oproj");
      break;
    case Op::kFfn:
      layer.fc1.Place(image, at + "fc1");
      layer.fc2.Place(image, at + "fc2");
      layer.idf.Place(image, at + "idf");
      break;
    case Op::kRefine:
      for (size_t i = 0; i < layer.reg.size(); ++i) {
        layer.reg[i].Place(image, std::format("{}reg{}", at, i));
      }
      layer.reg_out.Place(image, at + "reg_out");
      for (size_t i = 0; i < layer.cls.size(); ++i) {
        layer.cls[i].Place(image, std::format("{}cls{}", at, i));
      }
      layer.cls_out.Place(image, at + "cls_out");
      if (det) {
        for (size_t i = 0; i < layer.qt.size(); ++i) {
          layer.qt[i].Place(image, std::format("{}qt{}", at, i));
        }
        layer.qt_out.Place(image, at + "qt_out");
      }
      break;
    case Op::kNorm:
    case Op::kCrossGnn:
      break;
  }
}

void AddDeformableBiases(const HeadLayer& layer, const std::string& note,
                         Pbuf& pb, Keys& keys) {
  for (size_t i = 0; i < layer.cam.size(); ++i) {
    keys[std::format("{} cam b{}", note, i)] =
        pb.Add(layer.cam[i].first.Bias());
    keys[std::format("{} cam ln{}", note, i)] = pb.Norm(layer.cam[i].second);
  }
  keys[note + " lfc"] = pb.Add(layer.lfc.Bias());
  for (size_t g = 0; g < layer.wfc.size(); ++g) {
    keys[std::format("{} wfc{}", note, g)] = pb.Add(layer.wfc[g].Bias());
  }
  keys[note + " oproj"] = pb.Add(layer.oproj.Bias());
  keys[note + " cam"] = pb.Add(std::vector<uint32_t>(kD, 0));
  std::vector<uint32_t> lanes;
  for (const Word& word : LaneConstantWords()) {
    lanes.insert(lanes.end(), word.begin(), word.end());
  }
  keys[note + " kplane"] = pb.Add(lanes);
}

void AddRefineBiases(const HeadLayer& layer, const std::string& note,
                     const HeadConfig& cf, Pbuf& pb, Keys& keys) {
  for (size_t i = 0; i < layer.reg.size(); ++i) {
    keys[std::format("{} reg{}", note, i)] = pb.Add(layer.reg[i].Bias());
  }
  for (size_t i = 0; i < layer.reg_ln.size(); ++i) {
    keys[std::format("{} reg_ln{}", note, i)] = pb.Norm(layer.reg_ln[i]);
  }
  keys[note + " reg_out"] = pb.Add(layer.reg_out.Bias());
  keys[note + " scale"] = pb.Add(PadValues(layer.scale, cf.acols));
  for (size_t i = 0; i < layer.cls.size(); ++i) {
    keys[std::format("{} cls{}", note, i)] = pb.Add(layer.cls[i].Bias());
  }
  for (size_t i = 0; i < layer.cls_ln.size(); ++i) {
    keys[std::format("{} cls_ln{}", note, i)] = pb.Norm(layer.cls_ln[i]);
  }
  keys[note + " cls_out"] = pb.Add(layer.cls_out.Bias());
  if (cf.det) {
    for (size_t i = 0; i < layer.qt.size(); ++i) {
      keys[std::format("{} qt{}", note, i)] = pb.Add(layer.qt[i].Bias());
    }
    for (size_t i = 0; i < layer.qt_ln.size(); ++i) {
      keys[std::format("{} qt_ln{}", note, i)] = pb.Norm(layer.qt_ln[i]);
    }
    keys[note + " qt_out"] = pb.Add(layer.qt_out.Bias());
  }
}

void AddLayerBiases(const HeadLayer& layer, const std::string& note,
                    const HeadConfig& cf, Pbuf& pb, Keys& keys) {
  switch (layer.op) {
    case Op::kGnn:
    case Op::kTempGnn:
      for (size_t i = 0; i < 4; ++i) {
        keys[std::format("{} {}", note, "qkvo"[i])] =
            pb.Add(layer.attn[i].Bias());
      }
      break;
    case Op::kNorm:
      keys[note] = pb.Norm(layer.ln);
      break;
    case Op::kDeformable:
      AddDeformableBiases(layer, note, pb, keys);
      break;
    case Op::kFfn:
      keys[note + " ln"] = pb.Norm(layer.ln);
      keys[note + " fc1"] = pb.Add(layer.fc1.Bias());
      keys[note + " fc2"] = pb.Add(AddBits(layer.fc2.Bias(), layer.idf.Bias()));
      break;
    case Op::kRefine:
      AddRefineBiases(layer, note, cf, pb, keys);
      break;
    case Op::kCrossGnn:
      break;
  }
}

}  // namespace

void SparseHead::PlaceParams() {
  for (auto& branch : p.enc) {
    for (size_t i = 0; i < branch.chain.size(); ++i) {
      branch.chain[i].first.Place(
          image_, std::format("{}.enc.{}{}", tag, branch.name, i));
    }
  }
  if (p.fc_before.has_value()) {
    p.fc_before->Place(image_, tag + ".fc_before");
  }
  if (p.fc_after.has_value()) {
    p.fc_after->Place(image_, tag + ".fc_after");
  }
  for (size_t l = 0; l < p.layers.size(); ++l) {
    PlaceLayerWeights(image_, p.layers[l], std::format("{}.L{}.", tag, l),
                      C().det);
  }
  for (const auto& branch : p.enc) {
    std::vector<uint32_t> mask(C().acols, 0);
    for (uint32_t col = branch.lo; col < branch.hi; ++col) {
      mask[col] = FloatBits(1);
    }
    keys_enc[std::format("mask.{}", branch.name)] = pb_enc.Add(mask);
    for (size_t i = 0; i < branch.chain.size(); ++i) {
      keys_enc[std::format("enc.{}.b{}", branch.name, i)] =
          pb_enc.Add(branch.chain[i].first.Bias());
      keys_enc[std::format("enc.{}.ln{}", branch.name, i)] =
          pb_enc.Norm(branch.chain[i].second);
    }
  }
  pb_enc.Place(image_, tag + ".pbuf.enc");
  for (size_t l = 0; l < p.layers.size(); ++l) {
    const HeadLayer& layer = p.layers[l];
    const std::string note = std::format("{} L{} {}", tag, l, OpName(layer.op));
    Pbuf pb;
    Keys keys;
    AddLayerBiases(layer, note, C(), pb, keys);
    pb.Place(image_, std::format("{}.pbuf.L{}", tag, l));
    if (layer.op == Op::kRefine && C().det) {
      scale_patch.emplace_back(
          pb.Offset() + (uint64_t{keys.at(note + " scale")} * kWordF),
          layer.scale);
    }
    pb_layers.emplace_back(std::move(pb), std::move(keys), note);
  }
}

void SparseHead::LoadRows(const Rows& rows, uint32_t j, uint32_t m, Place dst,
                          std::string_view note) {
  program_->Load(Tokens(rows.Addr(j), Mem::kFbuf, dst, m, rows.Tok()), note);
}

Tile SparseHead::LoadRows(const Rows& rows, uint32_t j, uint32_t m,
                          std::string_view note) {
  const Tile tile = program_->Alloc(Mem::kFbuf, m / 4 * (rows.cols / 8));
  LoadRows(rows, j, m, tile.At(), note);
  return tile;
}

void SparseHead::StoreRows(const Rows& rows, uint32_t j, uint32_t m, Place src,
                           std::string_view note) {
  program_->Store(Tokens(rows.Addr(j), Mem::kFbuf, src, m, rows.Tok()), note);
}

void SparseHead::StoreColumns(const Rows& rows, uint32_t j, uint32_t m,
                              Place src, uint32_t col0, uint32_t ncols,
                              std::string_view note) {
  Require(ncols % 32 == 0 && col0 % 32 == 0, "unaligned column store");
  program_->Store({.ddr_offset = rows.Addr(j) + (uint64_t{col0} * 4),
                   .mem = Mem::kFbuf,
                   .onchip = src,
                   .rows = m,
                   .row_words = 1,
                   .ddr_stride = static_cast<uint32_t>(rows.Tb()),
                   .kind = DmaKind::kRow,
                   .tok = ncols / 32},
                  note);
}

void SparseHead::StoreColumnsStrided(const Rows& rows, uint32_t j, uint32_t m,
                                     const Tile& src, uint32_t n8_src,
                                     uint32_t col0, uint32_t ncols,
                                     std::string_view note) {
  const uint32_t words = ncols / 8;
  if (n8_src == words && col0 == 0) {
    StoreColumns(rows, j, m, src.At(), col0, ncols, note);
    return;
  }
  const uint32_t per = std::max(1U, kPackWords / words);
  for (uint32_t q0 = 0; q0 < m / 4; q0 += per) {
    const uint32_t count = std::min(per, (m / 4) - q0);
    const Tile pack = program_->Alloc(Mem::kFbuf, count * words);
    for (uint32_t q = 0; q < count; ++q) {
      program_->Vec({.op = VecOp::kCopy,
                     .src = src.At(((q0 + q) * n8_src) + (col0 / 8)),
                     .dst = pack.At(q * words),
                     .rows_q = 1,
                     .n8 = words},
                    std::format("{} pack q{}", note, q0 + q));
    }
    StoreColumns(rows, j + (4 * q0), 4 * count, pack.At(), col0, ncols,
                 std::format("{} q{}", note, q0));
  }
}

void SparseHead::QuantRows(const Rows& rows, uint32_t n, Place abuf,
                           std::string_view note, const QuantizeHook& before) {
  EmitQuantize(*program_,
               {.x = rows.Addr(0),
                .quads = n / 4,
                .words = rows.cols / 8,
                .tile_quads = kQuantQuads,
                .tokens = true,
                .dst = abuf},
               note, before);
}

void SparseHead::QuantRowsAdd(const Rows& rows, const Rows& rows2, uint32_t n,
                              Place abuf, std::string_view note,
                              std::optional<Place> abuf_plain,
                              std::optional<Place> pb_vec) {
  const uint32_t words = rows.cols / 8;
  QuantRows(
      rows, n, abuf, note, [&](const Tile& x, uint32_t quad, uint32_t quads) {
        const uint32_t t = quad / kQuantQuads;
        const Tile second = LoadRows(rows2, 4 * quad, 4 * quads,
                                     std::format("{} ld 2nd t{}", note, t));
        if (abuf_plain.has_value()) {
          program_->Vec({.op = VecOp::kCopy,
                         .src = x.At(),
                         .dst = *abuf_plain + (quad * (words / 4)),
                         .rows_q = quads,
                         .n8 = words,
                         .q = true},
                        std::format("{} Q(plain) t{}", note, t));
        }
        program_->Vec({.op = VecOp::kAdd,
                       .src = x.At(),
                       .src2 = second.At(),
                       .dst = x.At(),
                       .rows_q = quads,
                       .n8 = words},
                      std::format("{} add t{}", note, t));
        if (pb_vec.has_value()) {
          program_->Vec({.op = VecOp::kAddP,
                         .src = x.At(),
                         .dst = x.At(),
                         .rows_q = quads,
                         .n8 = words,
                         .p_base = *pb_vec},
                        std::format("{} addp t{}", note, t));
        }
      });
}

void SparseHead::EncodeAnchors(Program& program, const Rows& anchors,
                               const Rows& out, uint32_t n,
                               std::string_view note) {
  constexpr uint32_t kTile = 224;
  program_ = &program;
  const HeadConfig& cf = C();
  const Params params(pb_enc.Load(*program_, note), keys_enc);
  for (const Band& band : Bands(n, kTile)) {
    const uint32_t rq = band.m / 4;
    const Tile rows = LoadRows(anchors, band.i0, band.m,
                               std::format("{} ld anchors t{}", note, band.t));
    for (const auto& branch : p.enc) {
      Tile a = program_->Alloc(Mem::kAbuf, rq * (cf.acols / 32));
      program_->Vec(
          {.op = VecOp::kMulP,
           .src = rows.At(),
           .dst = a.At(),
           .rows_q = rq,
           .n8 = cf.acols / 8,
           .p_base = params.At("mask." + branch.name),
           .q = true},
          std::format("{} {} select+Q t{}", note, branch.name, band.t));
      for (size_t i = 0; i < branch.chain.size(); ++i) {
        const Lin& lin = branch.chain[i].first;
        const std::string where = std::format("{} {}{}", note, branch.name, i);
        const uint32_t words = lin.Groups();
        const Tile weights = lin.Load(*program_, where);
        const Tile c = program_->Alloc(Mem::kFbuf, rq * words);
        lin.Gemm(*program_, lin.Input(a.At()), 0,
                 i == 0 ? cf.acols / 32 : lin.Kb(), band.m, c.At(), weights,
                 std::format("{} t{}", where, band.t));
        program_->Vec(
            {.op = VecOp::kAddP,
             .src = c.At(),
             .dst = c.At(),
             .rows_q = rq,
             .n8 = words,
             .p_base = params.At(std::format("enc.{}.b{}", branch.name, i))},
            std::format("{} bias t{}", where, band.t));
        program_->Vec({.op = VecOp::kRelu,
                       .src = c.At(),
                       .dst = c.At(),
                       .rows_q = rq,
                       .n8 = words},
                      std::format("{} relu t{}", where, band.t));
        const Place ln = params.At(std::format("enc.{}.ln{}", branch.name, i));
        if (i + 1 < branch.chain.size()) {
          const Tile next = program_->Alloc(Mem::kAbuf, rq * (words / 4));
          const Tile scratch = program_->Alloc(Mem::kFbuf, rq * words);
          program_->Vec({.op = VecOp::kLayerNorm,
                         .src = c.At(),
                         .dst = next.At(),
                         .rows_q = rq,
                         .n8 = words,
                         .p_base = ln,
                         .imm = LnEpsBits(),
                         .q = true,
                         .scratch = scratch.At()},
                        std::format("{} LN+Q t{}", where, band.t));
          a = next;
        } else {
          const Tile normed = program_->Alloc(Mem::kFbuf, rq * words);
          program_->Vec({.op = VecOp::kLayerNorm,
                         .src = c.At(),
                         .dst = normed.At(),
                         .rows_q = rq,
                         .n8 = words,
                         .p_base = ln,
                         .imm = LnEpsBits()},
                        std::format("{} LN t{}", where, band.t));
          StoreColumns(
              out, band.i0, band.m, normed.At(), branch.col0, lin.Rows(),
              std::format("{} {} st E t{}", note, branch.name, band.t));
        }
      }
    }
  }
}

void SparseHead::Attention(const HeadLayer& layer, const Params& pb,
                           std::string_view note) {
  const HeadConfig& cf = C();
  const Rows& xr = x[cur_x];
  const bool is_temp = layer.op == Op::kTempGnn && temp_;
  const bool raw_temp = layer.op == Op::kTempGnn && !temp_;
  const uint32_t keys = is_temp ? cf.ntp : cf.nap;
  const uint32_t valid = is_temp ? cf.ntemp : cf.na;
  const bool dec = cf.det;
  const auto quantized = [&](uint32_t rows) {
    return program_->Alloc(Mem::kAbuf, rows / 4 * kKbD);
  };
  const Tile ax = quantized(cf.nap);
  std::vector<Piece> qp;
  std::optional<Tile> ae;
  if (dec) {
    ae = quantized(cf.nap);
    QuantRows(xr, cf.nap, ax.At(), std::format("{} Q(x)", note));
    QuantRows(e, cf.nap, ae->At(), std::format("{} Q(E)", note));
    qp = {{.a = ax.At(), .kb_offset = 0, .kb = kKbD},
          {.a = ae->At(), .kb_offset = kKbD, .kb = kKbD}};
  } else {
    const Tile aq = quantized(cf.nap);
    QuantRowsAdd(xr, e, cf.nap, aq.At(), std::format("{} Q(x+E)", note),
                 ax.At());
    qp = {{.a = aq.At(), .kb_offset = 0, .kb = kKbD}};
  }
  const auto proj = [&](const Lin& lin, const std::vector<Piece>& pieces,
                        uint32_t astride, uint32_t n, uint64_t out,
                        const std::string& name) {
    const Tile weights = lin.Load(*program_, name);
    for (const Band& band : Bands(n, 64)) {
      const uint32_t q0 = band.i0 / 4;
      const uint32_t rq = band.m / 4;
      const Tile c = program_->Alloc(Mem::kFbuf, rq * nwq);
      lin.Gemm(*program_, pieces, q0 * astride, astride, band.m, c.At(),
               weights, std::format("{} t{}", name, band.t));
      const Tile q = program_->Alloc(Mem::kAbuf, rq * kbq);
      program_->Vec({.op = VecOp::kAddP,
                     .src = c.At(),
                     .dst = q.At(),
                     .rows_q = rq,
                     .n8 = nwq,
                     .p_base = pb.At(name),
                     .q = true},
                    std::format("{} bias+Q t{}", name, band.t));
      program_->Store(Words(out + (uint64_t{q0} * kbq * kWordA), Mem::kAbuf,
                            q.At(), rq * kbq),
                      std::format("{} st t{}", name, band.t));
    }
  };
  proj(layer.attn[0], qp, kKbD, cf.nap, qb, std::format("{} q", note));
  std::vector<Piece> kp_pieces = qp;
  Place vsrc = ax.At();
  if (is_temp) {
    const Tile atx = quantized(cf.ntp);
    vsrc = atx.At();
    if (dec) {
      QuantRows(tx, cf.ntp, atx.At(), std::format("{} Q(temp_x)", note));
      kp_pieces = {
          {.a = atx.At(), .kb_offset = 0, .kb = kKbD},
          {.a = Required(ae, "Q(E)").At(), .kb_offset = kKbD, .kb = kKbD}};
    } else {
      const Tile ak = quantized(cf.ntp);
      QuantRowsAdd(tx, e, cf.ntp, ak.At(), std::format("{} Q(temp_x+E)", note),
                   atx.At());
      kp_pieces = {{.a = ak.At(), .kb_offset = 0, .kb = kKbD}};
    }
  }
  proj(layer.attn[1], kp_pieces, kKbD, keys, kb, std::format("{} k", note));
  if (dec && !raw_temp) {
    const Tile av = program_->Alloc(Mem::kAbuf, keys / 4 * kbq);
    const Lin& before = Required(p.fc_before, "fc_before");
    const Tile weights =
        before.Load(*program_, std::format("{} fc_before", note));
    for (const Band& band : Bands(keys, 64)) {
      const uint32_t q0 = band.i0 / 4;
      const uint32_t rq = band.m / 4;
      const Tile c = program_->Alloc(Mem::kFbuf, rq * nwq);
      before.Gemm(*program_, before.Input(vsrc), q0 * kKbD, kKbD, band.m,
                  c.At(), weights,
                  std::format("{} fc_before t{}", note, band.t));
      program_->Vec({.op = VecOp::kCopy,
                     .src = c.At(),
                     .dst = av.At(q0 * kbq),
                     .rows_q = rq,
                     .n8 = nwq,
                     .q = true},
                    std::format("{} fc_before Q t{}", note, band.t));
    }
    proj(layer.attn[2], {{.a = av.At(), .kb_offset = 0, .kb = kbq}}, kbq, keys,
         vb, std::format("{} v", note));
  } else if (dec) {
    proj(layer.attn[2], qp, kKbD, cf.nap, vb, std::format("{} v", note));
  } else {
    proj(layer.attn[2], {{.a = vsrc, .kb_offset = 0, .kb = kKbD}}, kKbD, keys,
         vb, std::format("{} v", note));
  }
  EmitAttention(*program_,
                {.q = qb,
                 .k = kb,
                 .v = vb,
                 .o = ob,
                 .heads = kHeads,
                 .kv_heads = kHeads,
                 .rows = cf.nap,
                 .chunk_quads = 8,
                 .embed = cf.dq,
                 .kv_blocks = kbq,
                 .hd = cf.hd,
                 .keys = keys,
                 .valid = {{.quad = 0, .keys = valid}},
                 .scale = cf.hd != 64},
                note);
  OutProjection(layer, pb, note);
}

void SparseHead::OutProjection(const HeadLayer& layer, const Params& pb,
                               std::string_view note) {
  const HeadConfig& cf = C();
  const Rows& xr = x[cur_x];
  const Rows& xo = x[(cur_x + 1) % 3];
  const Tile ao = program_->Alloc(Mem::kAbuf, cf.nap / 4 * kbq);
  for (const Band& band : Bands(cf.nap, 128)) {
    const uint32_t q0 = band.i0 / 4;
    const uint32_t rq = band.m / 4;
    const Tile o = program_->Alloc(Mem::kFbuf, rq * nwq);
    program_->Load(
        Words(ob + (uint64_t{q0} * nwq * kWordF), Mem::kFbuf, o.At(), rq * nwq),
        std::format("{} ld O t{}", note, band.t));
    program_->Vec({.op = VecOp::kCopy,
                   .src = o.At(),
                   .dst = ao.At(q0 * kbq),
                   .rows_q = rq,
                   .n8 = nwq,
                   .q = true},
                  std::format("{} Q(O) t{}", note, band.t));
  }
  const Lin& op = layer.attn[3];
  const Tile weights = op.Load(*program_, std::format("{} out_proj", note));
  const Place pbo = pb.At(std::format("{} o", note));
  if (cf.det) {
    const std::array<Tile, 2> af = {
        program_->Alloc(Mem::kAbuf, cf.nap / 4 * kKbD),
        program_->Alloc(Mem::kAbuf, cf.nap / 4 * kKbD)};
    for (uint32_t half = 0; half < 2; ++half) {
      const Rows& res = half == 0 ? xr : e;
      for (const Band& band : Bands(cf.nap, 64)) {
        const uint32_t q0 = band.i0 / 4;
        const uint32_t rq = band.m / 4;
        const Tile c = program_->Alloc(Mem::kFbuf, rq * kNwD);
        program_->Gemm({.a_base = ao.At(q0 * kbq),
                        .a_stride = kbq,
                        .kb = kbq,
                        .m = band.m,
                        .w_base = weights.At(half * 32 * op.Kb()),
                        .w_stride = op.Kb(),
                        .n8 = 32,
                        .c_base = c.At()},
                       std::format("{} out_proj h{} t{}", note, half, band.t));
        const Tile identity =
            LoadRows(res, band.i0, band.m,
                     std::format("{} ld identity h{} t{}", note, half, band.t));
        program_->Vec(
            {.op = VecOp::kAddP,
             .src = c.At(),
             .dst = c.At(),
             .rows_q = rq,
             .n8 = kNwD,
             .p_base = pbo + (half * (kNwD / 4))},
            std::format("{} out_proj bias h{} t{}", note, half, band.t));
        program_->Vec({.op = VecOp::kAdd,
                       .src = c.At(),
                       .src2 = identity.At(),
                       .dst = c.At(),
                       .rows_q = rq,
                       .n8 = kNwD},
                      std::format("{} + identity h{} t{}", note, half, band.t));
        program_->Vec({.op = VecOp::kCopy,
                       .src = c.At(),
                       .dst = af.at(half).At(q0 * kKbD),
                       .rows_q = rq,
                       .n8 = kNwD,
                       .q = true},
                      std::format("{} Q h{} t{}", note, half, band.t));
      }
    }
    const Lin& after = Required(p.fc_after, "fc_after");
    const Tile wa = after.Load(*program_, std::format("{} fc_after", note));
    for (const Band& band : Bands(cf.nap, 128)) {
      const Tile c = program_->Alloc(Mem::kFbuf, band.m / 4 * after.Groups());
      after.Gemm(*program_,
                 {{.a = af[0].At(), .kb_offset = 0, .kb = kKbD},
                  {.a = af[1].At(), .kb_offset = kKbD, .kb = kKbD}},
                 (band.i0 / 4) * kKbD, kKbD, band.m, c.At(), wa,
                 std::format("{} fc_after t{}", note, band.t));
      StoreRows(xo, band.i0, band.m, c.At(),
                std::format("{} st x t{}", note, band.t));
    }
  } else {
    for (const Band& band : Bands(cf.nap, 64)) {
      const uint32_t q0 = band.i0 / 4;
      const uint32_t rq = band.m / 4;
      const Tile c = program_->Alloc(Mem::kFbuf, rq * kNwD);
      op.Gemm(*program_, op.Input(ao.At()), q0 * kbq, kbq, band.m, c.At(),
              weights, std::format("{} out_proj t{}", note, band.t));
      const Tile identity = LoadRows(
          xr, band.i0, band.m, std::format("{} ld identity t{}", note, band.t));
      program_->Vec({.op = VecOp::kAddP,
                     .src = c.At(),
                     .dst = c.At(),
                     .rows_q = rq,
                     .n8 = kNwD,
                     .p_base = pbo},
                    std::format("{} out_proj bias t{}", note, band.t));
      program_->Vec({.op = VecOp::kAdd,
                     .src = c.At(),
                     .src2 = identity.At(),
                     .dst = c.At(),
                     .rows_q = rq,
                     .n8 = kNwD},
                    std::format("{} + identity t{}", note, band.t));
      StoreRows(xo, band.i0, band.m, c.At(),
                std::format("{} st x t{}", note, band.t));
    }
  }
  cur_x = (cur_x + 1) % 3;
}

void SparseHead::Norm(const Params& pb, std::string_view note) {
  const Rows& xr = x[cur_x];
  const Rows& xo = x[(cur_x + 1) % 3];
  for (const Band& band : Bands(C().nap, 128)) {
    const uint32_t rq = band.m / 4;
    const Tile in =
        LoadRows(xr, band.i0, band.m, std::format("{} ld t{}", note, band.t));
    const Tile out = program_->Alloc(Mem::kFbuf, rq * kNwD);
    program_->Vec({.op = VecOp::kLayerNorm,
                   .src = in.At(),
                   .dst = out.At(),
                   .rows_q = rq,
                   .n8 = kNwD,
                   .p_base = pb.At(note),
                   .imm = LnEpsBits()},
                  std::format("{} LN t{}", note, band.t));
    StoreRows(xo, band.i0, band.m, out.At(),
              std::format("{} st t{}", note, band.t));
  }
  cur_x = (cur_x + 1) % 3;
}

void SparseHead::CameraEncoder(const HeadLayer& layer, const Params& pb,
                               std::string_view note) {
  constexpr uint32_t kQuads = 8;
  const Tile in =
      LoadRows(cam_in, 0, 4 * kQuads, std::format("{} ld cam_in", note));
  Tile a = program_->Alloc(Mem::kAbuf, kQuads);
  program_->Vec({.op = VecOp::kCopy,
                 .src = in.At(),
                 .dst = a.At(),
                 .rows_q = kQuads,
                 .n8 = 4,
                 .q = true},
                std::format("{} Q", note));
  for (size_t i = 0; i < layer.cam.size(); ++i) {
    const Lin& lin = layer.cam[i].first;
    const std::string where = std::format("{} {}", note, i);
    const Tile weights = lin.Load(*program_, where);
    const Tile c = program_->Alloc(Mem::kFbuf, kQuads * kNwD);
    lin.Gemm(*program_, lin.Input(a.At()), 0, i == 0 ? 1 : kKbD, 4 * kQuads,
             c.At(), weights, where);
    program_->Vec({.op = VecOp::kAddP,
                   .src = c.At(),
                   .dst = c.At(),
                   .rows_q = kQuads,
                   .n8 = kNwD,
                   .p_base = pb.At(std::format("{} b{}", note, i))},
                  std::format("{} bias {}", note, i));
    program_->Vec({.op = VecOp::kRelu,
                   .src = c.At(),
                   .dst = c.At(),
                   .rows_q = kQuads,
                   .n8 = kNwD},
                  std::format("{} relu {}", note, i));
    const Place ln = pb.At(std::format("{} ln{}", note, i));
    if (i + 1 < layer.cam.size()) {
      const Tile next = program_->Alloc(Mem::kAbuf, kQuads * kKbD);
      const Tile scratch = program_->Alloc(Mem::kFbuf, kQuads * kNwD);
      program_->Vec({.op = VecOp::kLayerNorm,
                     .src = c.At(),
                     .dst = next.At(),
                     .rows_q = kQuads,
                     .n8 = kNwD,
                     .p_base = ln,
                     .imm = LnEpsBits(),
                     .q = true,
                     .scratch = scratch.At()},
                    std::format("{} LN+Q {}", note, i));
      a = next;
    } else {
      const Tile out = program_->Alloc(Mem::kFbuf, kQuads * kNwD);
      program_->Vec({.op = VecOp::kLayerNorm,
                     .src = c.At(),
                     .dst = out.At(),
                     .rows_q = kQuads,
                     .n8 = kNwD,
                     .p_base = ln,
                     .imm = LnEpsBits()},
                    std::format("{} LN {}", note, i));
      StoreRows(cam_vec, 0, 4 * kQuads, out.At(),
                std::format("{} st cam_vec", note));
    }
  }
}

void SparseHead::Deformable(const HeadLayer& layer, size_t index,
                            const Params& pb, std::string_view note) {
  const HeadConfig& cf = C();
  const Rows& xr = x[cur_x];
  const Tile ax = program_->Alloc(Mem::kAbuf, cf.nap / 4 * kKbD);
  const Tile aw = program_->Alloc(Mem::kAbuf, cf.nap / 4 * kKbD);
  const Place cam = pb.At(std::format("{} cam", note));
  QuantRowsAdd(xr, e, cf.nap, aw.At(), std::format("{} Q(x+E+cam)", note),
               ax.At(), cam);
  const Lin& lin = layer.lfc;
  const Tile weights = lin.Load(*program_, std::format("{} lfc", note));
  const uint32_t n8l = lin.Groups();
  const uint32_t mtl =
      std::max(32U, std::min(448U, (kFbufWords / 2 / n8l) * 4 / 32 * 32));
  for (const Band& band : Bands(cf.nap, mtl)) {
    const Tile c = program_->Alloc(Mem::kFbuf, band.m / 4 * n8l);
    lin.Gemm(*program_, lin.Input(ax.At()), (band.i0 / 4) * kKbD, kKbD, band.m,
             c.At(), weights, std::format("{} lfc t{}", note, band.t));
    program_->Vec({.op = VecOp::kAddP,
                   .src = c.At(),
                   .dst = c.At(),
                   .rows_q = band.m / 4,
                   .n8 = n8l,
                   .p_base = pb.At(std::format("{} lfc", note))},
                  std::format("{} lfc bias t{}", note, band.t));
    if (lfc.Tok() <= 8) {
      StoreRows(lfc, band.i0, band.m, c.At(),
                std::format("{} st lfc t{}", note, band.t));
    } else {
      for (uint32_t c0 = 0; c0 < lfc.cols; c0 += 256) {
        StoreColumnsStrided(
            lfc, band.i0, band.m, c, n8l, c0, std::min(256U, lfc.cols - c0),
            std::format("{} st lfc t{} cols {}", note, band.t, c0));
      }
    }
  }
  DeformableWeights(layer, pb, note, aw);
  const Tile ag = program_->Alloc(Mem::kAbuf, cf.nap / 4 * kKbD);
  const uint32_t ring_quads = cf.det ? kDetGatherQuads : 32;
  uint32_t gathered = 0;
  std::optional<Pending> pending;
  Tile slot;
  const auto gather = [&](uint32_t until) {
    for (; gathered < until; ++gathered) {
      Gather(gathered, ag, ring_quads, note, pending, slot);
    }
  };
  if (cf.det) {
    Keypoints(index, pb, note, gather);
  } else {
    KeypointsMap(note);
  }
  gather(cf.nap / 4);
  if (pending.has_value()) {
    program_->Vec(pending->args, pending->note);
  }
  const Lin& opj = layer.oproj;
  const Tile wo = opj.Load(*program_, std::format("{} output_proj", note));
  for (const Band& band : Bands(cf.nap, 128)) {
    const uint32_t q0 = band.i0 / 4;
    const uint32_t rq = band.m / 4;
    const Tile c = program_->Alloc(Mem::kFbuf, rq * kNwD);
    opj.Gemm(*program_, opj.Input(ag.At()), q0 * kKbD, kKbD, band.m, c.At(), wo,
             std::format("{} output_proj t{}", note, band.t));
    program_->Vec({.op = VecOp::kAddP,
                   .src = c.At(),
                   .dst = c.At(),
                   .rows_q = rq,
                   .n8 = kNwD,
                   .p_base = pb.At(std::format("{} oproj", note))},
                  std::format("{} output_proj bias t{}", note, band.t));
    program_->Store({.ddr_offset = cat + (uint64_t{q0} * 64 * kWordF),
                     .mem = Mem::kFbuf,
                     .onchip = c.At(),
                     .rows = rq,
                     .row_words = kNwD,
                     .ddr_stride = 64 * kWordF},
                    std::format("{} st cat[:256] t{}", note, band.t));
    const Tile identity =
        LoadRows(xr, band.i0, band.m, std::format("{} ld x t{}", note, band.t));
    program_->Store(
        {.ddr_offset = cat + (((uint64_t{q0} * 64) + kNwD) * kWordF),
         .mem = Mem::kFbuf,
         .onchip = identity.At(),
         .rows = rq,
         .row_words = kNwD,
         .ddr_stride = 64 * kWordF},
        std::format("{} st cat[256:] t{}", note, band.t));
  }
}

void SparseHead::Gather(uint32_t q, const Tile& ag, uint32_t ring_quads,
                        std::string_view note, std::optional<Pending>& pending,
                        Tile& slot) {
  const HeadConfig& cf = C();
  const uint32_t quads = cf.nap / 4;
  if (q % ring_quads == 0) {
    slot = program_->Alloc(Mem::kFbuf, std::min(ring_quads, quads - q) * kNwD);
  }
  const Place dst = slot.At((q % ring_quads) * kNwD);
  const uint32_t cameras = q < nq ? ncam : 1;
  for (uint32_t camera = 0; camera < cameras; ++camera) {
    if (q < nq) {
      program_->Gacc({.list_offset = ListAddr(q, camera),
                      .dst = dst,
                      .slots = cf.scam,
                      .blocks = kKbD,
                      .coefficient_stride = static_cast<uint32_t>(wg * kWordF),
                      .acc = camera > 0,
                      .flush = camera == ncam - 1},
                     std::format("{} gacc q{} c{}", note, q, camera));
    } else {
      program_->Gacc({.list_offset = null_list,
                      .dst = dst,
                      .slots = 1,
                      .blocks = kKbD,
                      .coefficient_stride = static_cast<uint32_t>(wg * kWordF),
                      .acc = false,
                      .flush = true},
                     std::format("{} gacc pad q{}", note, q));
    }
    if (camera == 0 && pending.has_value()) {
      program_->Vec(pending->args, pending->note);
      pending.reset();
    }
  }
  if (q % ring_quads == ring_quads - 1 || q == quads - 1) {
    const uint32_t first = q - (q % ring_quads);
    pending =
        Pending{.args = {.op = VecOp::kCopy,
                         .src = slot.At(),
                         .dst = ag.At(first * kKbD),
                         .rows_q = (q % ring_quads) + 1,
                         .n8 = kNwD,
                         .q = true},
                .note = std::format("{} Q(gacc) t{}", note, q / ring_quads)};
  }
}

void SparseHead::DeformableWeights(const HeadLayer& layer, const Params& pb,
                                   std::string_view note, Tile aw) {
  const HeadConfig& cf = C();
  const Rows& xr = x[cur_x];
  const uint32_t n8w = cf.scam / 8;
  const uint32_t mt =
      std::max(32U, std::min(448U, (kFbufWords / 4 / n8w) * 4 / 32 * 32));
  const Place cam = pb.At(std::format("{} cam", note));
  for (uint32_t camera = 0; camera < ncam; ++camera) {
    if (camera > 0) {
      program_->Load(Words(cam_vec.Addr(camera), Mem::kPbuf, cam, kNwD / 4),
                     std::format("{} ld cam{} -> PBUF", note, camera));
      aw = program_->Alloc(Mem::kAbuf, cf.nap / 4 * kKbD);
      QuantRowsAdd(xr, e, cf.nap, aw.At(),
                   std::format("{} Q(x+E+cam{})", note, camera), std::nullopt,
                   cam);
    }
    for (uint32_t g = 0; g < kGroups; ++g) {
      const Lin& lin = layer.wfc[g];
      const Tile weights =
          lin.Load(*program_, std::format("{} wfc g{} c{}", note, g, camera));
      for (const Band& band : Bands(cf.nap, mt)) {
        const uint32_t q0 = band.i0 / 4;
        if (q0 >= nq) {
          continue;
        }
        const uint32_t count = std::min(band.m / 4, nq - q0);
        const Tile c = program_->Alloc(Mem::kFbuf, band.m / 4 * n8w);
        lin.Gemm(*program_, lin.Input(aw.At()), q0 * kKbD, kKbD, band.m, c.At(),
                 weights,
                 std::format("{} wfc g{} c{} t{}", note, g, camera, band.t));
        program_->Vec(
            {.op = VecOp::kAddP,
             .src = c.At(),
             .dst = c.At(),
             .rows_q = band.m / 4,
             .n8 = n8w,
             .p_base = pb.At(std::format("{} wfc{}", note, g))},
            std::format("{} wfc bias g{} c{} t{}", note, g, camera, band.t));
        program_->Store(
            {.ddr_offset = w + (((uint64_t{g} * wg) + (uint64_t{q0} * wq) +
                                 (uint64_t{camera} * n8w)) *
                                kWordF),
             .mem = Mem::kFbuf,
             .onchip = c.At(),
             .rows = count,
             .row_words = n8w,
             .ddr_stride = static_cast<uint32_t>(uint64_t{wq} * kWordF)},
            std::format("{} st logits g{} c{} t{}", note, g, camera, band.t));
      }
    }
  }
  WeightSoftmax(note);
}

void SparseHead::WeightSoftmax(std::string_view note) {
  const uint32_t tqs = std::min(32U, kFbufWords / (4 * wq));
  struct Part {
    uint32_t g = 0;
    uint32_t q0 = 0;
    uint32_t count = 0;
    uint64_t addr = 0;
  };
  std::vector<Part> parts;
  for (uint32_t g = 0; g < kGroups; ++g) {
    for (uint32_t q0 = 0; q0 < nq; q0 += tqs) {
      parts.push_back(
          {.g = g,
           .q0 = q0,
           .count = std::min(tqs, nq - q0),
           .addr = w + (((uint64_t{g} * wg) + (uint64_t{q0} * wq)) * kWordF)});
    }
  }
  std::vector<Tile> in(parts.size());
  std::vector<Tile> out(parts.size());
  const auto load = [&](size_t i) {
    const Part& part = parts[i];
    in[i] = program_->Alloc(Mem::kFbuf, part.count * wq);
    program_->Load(Words(part.addr, Mem::kFbuf, in[i].At(), part.count * wq),
                   std::format("{} ld logits g{} q{}", note, part.g, part.q0));
  };
  const auto softmax = [&](size_t i) {
    const Part& part = parts[i];
    out[i] = program_->Alloc(Mem::kFbuf, part.count * wq);
    if (wq <= 255) {
      program_->Vec({.op = VecOp::kSoftmax,
                     .src = in[i].At(),
                     .dst = out[i].At(),
                     .rows_q = part.count,
                     .n8 = wq,
                     .aux = s},
                    std::format("{} softmax g{} q{}", note, part.g, part.q0));
    } else {
      program_->Run({.pc = kp.Pc("ls"),
                     .src = in[i].At(),
                     .src2 = out[i].At(),
                     .dst = out[i].At(),
                     .rows_q = part.count,
                     .imm = wq,
                     .imm_places = {},
                     .reads = {Whole(in[i]), Whole(out[i]),
                               Fixed(Mem::kTbuf, 0, kTbufWords)},
                     .writes = {Whole(out[i])}},
                    std::format("{} softmax g{} q{}", note, part.g, part.q0));
    }
  };
  const auto store = [&](size_t i) {
    const Part& part = parts[i];
    program_->Store(Words(part.addr, Mem::kFbuf, out[i].At(), part.count * wq),
                    std::format("{} st w g{} q{}", note, part.g, part.q0));
  };
  load(0);
  for (size_t i = 0; i < parts.size(); ++i) {
    softmax(i);
    if (i > 0) {
      store(i - 1);
    }
    if (i + 1 < parts.size()) {
      load(i + 1);
    }
  }
  store(parts.size() - 1);
}

void SparseHead::Keypoints(size_t index, const Params& pb,
                           std::string_view note,
                           const std::function<void(uint32_t)>& gather) {
  const HeadConfig& cf = C();
  constexpr uint32_t kTq = kDetTile;
  const uint64_t stride = uint64_t{ncam} * lbc;
  const uint32_t stores = ncam * cf.npt * kLevels;
  const Access kplane{
      .mem = Mem::kPbuf,
      .at = pb.At(std::format("{} kplane", note)),
      .words = static_cast<uint32_t>(LaneConstantWords().size())};
  const Access table = Fixed(Mem::kTbuf, 0, kTbufWords);
  const Rows& anchors = anc[cur_a];
  std::vector<Tile> consts;
  for (uint32_t camera = 0; camera < ncam; ++camera) {
    const Tile cst = program_->Alloc(Mem::kFbuf, 3);
    program_->Load(
        Words(kp_const + (uint64_t{camera} * kWordF), Mem::kFbuf, cst.At(0), 1),
        std::format("{} kp ld const c{}", note, camera));
    program_->Load(Words(kp_fs.at(index), Mem::kFbuf, cst.At(1), 1),
                   std::format("{} kp ld fix_scale c{}", note, camera));
    program_->Load(
        Words(kp_lv + (uint64_t{camera} * kWordF), Mem::kFbuf, cst.At(2), 1),
        std::format("{} kp ld levels c{}", note, camera));
    consts.push_back(cst);
  }
  uint32_t t = 0;
  for (uint32_t q0 = 0; q0 < cf.nap / 4; q0 += kTq, ++t) {
    const uint32_t i0 = 4 * q0;
    const Tile anchor = LoadRows(anchors, i0, 4 * kTq,
                                 std::format("{} kp ld anchor t{}", note, t));
    const Tile learned =
        LoadRows(lfc, i0, 4 * kTq, std::format("{} kp ld lfc t{}", note, t));
    const Tile work = program_->Alloc(Mem::kFbuf, DetKeypointWork(kTq));
    const uint32_t rows = std::min(kTq, nq - q0);
    const uint32_t ready = q0 - std::min(q0, kTq);
    uint32_t k = 0;
    for (uint32_t camera = 0; camera < ncam; ++camera) {
      const Tile& cst = consts[camera];
      const std::vector<Packed> bases = {{.place = cst.At(), .shift = 0},
                                         {.place = work.At(), .shift = 12}};
      if (camera == 0) {
        program_->Run(
            {.pc = kp.Pc("init"),
             .src = anchor.At(),
             .rows_q = kTq,
             .n8 = 0,
             .p_base = kplane.at,
             .imm_places = bases,
             .reads = {Whole(anchor), Whole(cst), Whole(work), kplane, table},
             .writes = {Whole(work)}},
            std::format("{} kp init t{}", note, t));
      }
      for (uint32_t pt = 0; pt < cf.npt; ++pt) {
        program_->Run(
            {.pc = kp.Pc("point"),
             .src = anchor.At(),
             .src2 = learned.At(),
             .rows_q = kTq,
             .n8 = pt,
             .imm_places = bases,
             .reads = {Whole(anchor), Whole(learned), Whole(cst), Whole(work),
                       table},
             .writes = {Whole(work)}},
            std::format("{} kp point t{} c{} p{}", note, t, camera, pt));
        for (uint32_t l = 0; l < kLevels; ++l) {
          const Tile out = program_->Alloc(Mem::kFbuf, kTq);
          ++k;
          program_->Run({.pc = kp.Pc("level"),
                         .dst = out.At(),
                         .rows_q = kTq,
                         .n8 = l,
                         .p_base = kplane.at,
                         .imm_places = bases,
                         .reads = {Whole(cst), Whole(work), kplane},
                         .writes = {Whole(out), Whole(work)}},
                        std::format("{} kp level t{} c{} p{} l{}", note, t,
                                    camera, pt, l));
          program_->Store(
              {.ddr_offset = ListAddr(q0, camera) + 16 +
                             (((uint64_t{l} * cf.npt) + pt) * kWordF),
               .mem = Mem::kFbuf,
               .onchip = out.At(),
               .rows = rows,
               .row_words = 1,
               .ddr_stride = static_cast<uint32_t>(stride)},
              std::format("{} kp st lists t{} c{} p{} l{}", note, t, camera, pt,
                          l));
          gather(ready + ((q0 - ready) * k / stores));
        }
      }
    }
  }
}

void SparseHead::KeypointsMap(std::string_view note) {
  const HeadConfig& cf = C();
  const uint32_t tq = cf.nap / 4;
  const uint64_t stride = uint64_t{ncam} * lbc;
  const Tile anchor =
      LoadRows(anc[cur_a], 0, cf.nap, std::format("{} kp ld anchor", note));
  std::vector<Tile> consts;
  for (uint32_t camera = 0; camera < ncam; ++camera) {
    const Tile cst = program_->Alloc(Mem::kFbuf, 3);
    program_->Load(
        Words(kp_const + (uint64_t{camera} * kWordF), Mem::kFbuf, cst.At(0), 1),
        std::format("{} kp ld const c{}", note, camera));
    program_->Load(
        Words(kp_lv + (uint64_t{camera} * kWordF), Mem::kFbuf, cst.At(1), 1),
        std::format("{} kp ld levels c{}", note, camera));
    program_->Load(Words(kp_zh, Mem::kFbuf, cst.At(2), 1),
                   std::format("{} kp ld z c{}", note, camera));
    consts.push_back(cst);
  }
  const Tile work = program_->Alloc(Mem::kFbuf, MapKeypointWork(tq));
  const Packed work_base{.place = work.At(), .shift = 12};
  program_->Run(
      {.pc = kp.Pc("init"),
       .src = anchor.At(),
       .rows_q = tq,
       .n8 = 0,
       .imm_places = {{.place = consts[0].At(), .shift = 0}, work_base},
       .reads = {Whole(anchor), Whole(consts[0]), Whole(work)},
       .writes = {Whole(work)}},
      std::format("{} kp init", note));
  for (uint32_t gi = 0; gi < kMapGroups; ++gi) {
    const auto [h, lq, gs] = MapGroup(gi);
    const uint32_t col0 = ((h * kMapLearn) + lq) * kMapBlock;
    const Tile learned = program_->Alloc(Mem::kFbuf, tq * (kMapBlock / 8));
    program_->Load({.ddr_offset = lfc.Addr(0) + (uint64_t{col0} * 4),
                    .mem = Mem::kFbuf,
                    .onchip = learned.At(),
                    .rows = cf.nap,
                    .row_words = 1,
                    .ddr_stride = static_cast<uint32_t>(lfc.Tb()),
                    .kind = DmaKind::kRow,
                    .tok = kMapBlock / 32},
                   std::format("{} kp ld lfc blk h{} lq{}", note, h, lq));
    const uint32_t imm = (h << 24) | (lq << 27) | (gs << 29);
    for (uint32_t camera = 0; camera < ncam; ++camera) {
      const Tile& cst = consts[camera];
      const std::vector<Packed> bases = {{.place = cst.At(), .shift = 0},
                                         work_base};
      program_->Run({.pc = kp.Pc("point"),
                     .src2 = learned.At(),
                     .rows_q = tq,
                     .n8 = 0,
                     .imm = imm,
                     .imm_places = bases,
                     .reads = {Whole(learned), Whole(cst), Whole(work)},
                     .writes = {Whole(work)}},
                    std::format("{} kp point g{} c{}", note, gi, camera));
      for (uint32_t l = 0; l < kLevels; ++l) {
        const Tile out = program_->Alloc(Mem::kFbuf, 8 * tq);
        program_->Run(
            {.pc = kp.Pc("level"),
             .dst = out.At(),
             .rows_q = tq,
             .n8 = l,
             .imm = imm,
             .imm_places = bases,
             .reads = {Whole(cst), Whole(work)},
             .writes = {Whole(out), Whole(work)}},
            std::format("{} kp level g{} c{} l{}", note, gi, camera, l));
        for (uint32_t i = 0; i < 8; ++i) {
          const uint32_t sample = (8 * gs) + i;
          if (sample >= kMapSamples) {
            break;
          }
          const uint32_t pt = (((sample * kMapHeights) + h) * kMapLearn) + lq;
          program_->Store(
              {.ddr_offset = ListAddr(0, camera) + 16 +
                             (((uint64_t{l} * cf.npt) + pt) * kWordF),
               .mem = Mem::kFbuf,
               .onchip = out.At(i * tq),
               .rows = nq,
               .row_words = 1,
               .ddr_stride = static_cast<uint32_t>(stride)},
              std::format("{} kp st lists c{} p{} l{}", note, camera, pt, l));
        }
      }
    }
  }
}

void SparseHead::Ffn(const HeadLayer& layer, const Params& pb,
                     std::string_view note) {
  const HeadConfig& cf = C();
  const Rows& xo = x[(cur_x + 1) % 3];
  const Tile aff = program_->Alloc(Mem::kAbuf, cf.nap / 4 * 16);
  for (const Band& band : Bands(cf.nap, 64)) {
    const uint32_t q0 = band.i0 / 4;
    const uint32_t rq = band.m / 4;
    const Tile in = program_->Alloc(Mem::kFbuf, rq * 64);
    program_->Load(
        Words(cat + (uint64_t{q0} * 64 * kWordF), Mem::kFbuf, in.At(), rq * 64),
        std::format("{} ld cat t{}", note, band.t));
    const Tile scratch = program_->Alloc(Mem::kFbuf, rq * 64);
    program_->Vec({.op = VecOp::kLayerNorm,
                   .src = in.At(),
                   .dst = aff.At(q0 * 16),
                   .rows_q = rq,
                   .n8 = 64,
                   .p_base = pb.At(std::format("{} ln", note)),
                   .imm = LnEpsBits(),
                   .q = true,
                   .scratch = scratch.At()},
                  std::format("{} LN+Q t{}", note, band.t));
  }
  for (const Band& band : Bands(cf.nap, 128)) {
    const uint32_t q0 = band.i0 / 4;
    const uint32_t rq = band.m / 4;
    std::array<Tile, 2> ah;
    for (uint32_t nt = 0; nt < 2; ++nt) {
      const uint32_t words = 64 * layer.fc1.Kb();
      const Tile w1 = program_->Alloc(Mem::kWbuf, words);
      program_->Load(
          Words(layer.fc1.Offset(0) + (uint64_t{nt} * words * kWordA),
                Mem::kWbuf, w1.At(), words),
          std::format("{} ld fc1 nt{} t{}", note, nt, band.t));
      const Tile h1 = program_->Alloc(Mem::kFbuf, rq * 64);
      program_->Gemm({.a_base = aff.At(q0 * 16),
                      .a_stride = 16,
                      .kb = 16,
                      .m = band.m,
                      .w_base = w1.At(),
                      .w_stride = layer.fc1.Kb(),
                      .n8 = 64,
                      .c_base = h1.At()},
                     std::format("{} fc1 nt{} t{}", note, nt, band.t));
      program_->Vec({.op = VecOp::kAddP,
                     .src = h1.At(),
                     .dst = h1.At(),
                     .rows_q = rq,
                     .n8 = 64,
                     .p_base = pb.At(std::format("{} fc1", note)) + (nt * 16)},
                    std::format("{} fc1 bias nt{} t{}", note, nt, band.t));
      ah.at(nt) = program_->Alloc(Mem::kAbuf, rq * 16);
      program_->Vec({.op = VecOp::kRelu,
                     .src = h1.At(),
                     .dst = ah.at(nt).At(),
                     .rows_q = rq,
                     .n8 = 64,
                     .q = true},
                    std::format("{} fc1 relu+Q nt{} t{}", note, nt, band.t));
    }
    const Tile w2 =
        layer.fc2.Load(*program_, std::format("{} fc2 t{}", note, band.t));
    const Tile wi =
        layer.idf.Load(*program_, std::format("{} idf t{}", note, band.t));
    const Tile out = program_->Alloc(Mem::kFbuf, rq * kNwD);
    layer.fc2.Gemm(*program_,
                   {{.a = ah[0].At(), .kb_offset = 0, .kb = 16},
                    {.a = ah[1].At(), .kb_offset = 16, .kb = 16}},
                   0, 16, band.m, out.At(), w2,
                   std::format("{} fc2 t{}", note, band.t));
    layer.idf.Gemm(*program_, layer.idf.Input(aff.At()), q0 * 16, 16, band.m,
                   out.At(), wi,
                   std::format("{} identity_fc t{}", note, band.t), true);
    program_->Vec({.op = VecOp::kAddP,
                   .src = out.At(),
                   .dst = out.At(),
                   .rows_q = rq,
                   .n8 = kNwD,
                   .p_base = pb.At(std::format("{} fc2", note))},
                  std::format("{} bias t{}", note, band.t));
    StoreRows(xo, band.i0, band.m, out.At(),
              std::format("{} st x t{}", note, band.t));
  }
  cur_x = (cur_x + 1) % 3;
}

void SparseHead::Refine(const HeadLayer& layer, const Params& pb,
                        std::string_view note) {
  const HeadConfig& cf = C();
  constexpr uint32_t kTile = 224;
  const Rows& xr = x[cur_x];
  const Rows& a_in = anc[cur_a];
  const Rows& a_out = anc[1 - cur_a];
  const auto quantized = [&] {
    return program_->Alloc(Mem::kAbuf, cf.nap / 4 * kKbD);
  };
  const Tile xe = quantized();
  const Tile ax = quantized();
  QuantRows(xr, cf.nap, xe.At(), std::format("{} Q(x+E)", note),
            [&](const Tile& rows, uint32_t quad, uint32_t quads) {
              const uint32_t t = quad / kQuantQuads;
              const Tile et = LoadRows(e, 4 * quad, 4 * quads,
                                       std::format("{} ld E t{}", note, t));
              program_->Vec({.op = VecOp::kCopy,
                             .src = rows.At(),
                             .dst = ax.At(quad * kKbD),
                             .rows_q = quads,
                             .n8 = kNwD,
                             .q = true},
                            std::format("{} Q(x) t{}", note, t));
              program_->Vec({.op = VecOp::kAdd,
                             .src = rows.At(),
                             .src2 = et.At(),
                             .dst = rows.At(),
                             .rows_q = quads,
                             .n8 = kNwD},
                            std::format("{} x+E t{}", note, t));
            });
  const auto chain = [&](const std::vector<Lin>& lins,
                         const std::function<std::string(size_t)>& lns,
                         Tile src, const std::string& bkey,
                         const std::function<bool(size_t)>& relu_ln) {
    for (size_t i = 0; i < lins.size(); ++i) {
      const Tile dst = quantized();
      const std::string name = std::format("{}{}", bkey, i);
      const Tile weights = lins[i].Load(*program_, name);
      for (const Band& band : Bands(cf.nap, kTile)) {
        const uint32_t q0 = band.i0 / 4;
        const uint32_t rq = band.m / 4;
        const Tile c = program_->Alloc(Mem::kFbuf, rq * kNwD);
        lins[i].Gemm(*program_, lins[i].Input(src.At()), q0 * kKbD, kKbD,
                     band.m, c.At(), weights,
                     std::format("{} t{}", name, band.t));
        program_->Vec({.op = VecOp::kAddP,
                       .src = c.At(),
                       .dst = c.At(),
                       .rows_q = rq,
                       .n8 = kNwD,
                       .p_base = pb.At(name)},
                      std::format("{} bias t{}", name, band.t));
        if (!relu_ln(i)) {
          program_->Vec({.op = VecOp::kRelu,
                         .src = c.At(),
                         .dst = dst.At(q0 * kKbD),
                         .rows_q = rq,
                         .n8 = kNwD,
                         .q = true},
                        std::format("{} relu+Q t{}", name, band.t));
        } else {
          program_->Vec({.op = VecOp::kRelu,
                         .src = c.At(),
                         .dst = c.At(),
                         .rows_q = rq,
                         .n8 = kNwD},
                        std::format("{} relu t{}", name, band.t));
          const Tile scratch = program_->Alloc(Mem::kFbuf, rq * kNwD);
          program_->Vec({.op = VecOp::kLayerNorm,
                         .src = c.At(),
                         .dst = dst.At(q0 * kKbD),
                         .rows_q = rq,
                         .n8 = kNwD,
                         .p_base = pb.At(lns(i)),
                         .imm = LnEpsBits(),
                         .q = true,
                         .scratch = scratch.At()},
                        std::format("{} LN+Q t{}", name, band.t));
        }
      }
      src = dst;
    }
    return src;
  };
  const auto head_out =
      [&](const Lin& lin, const Tile& src, const Rows& rows_out,
          const std::string& bkey,
          const std::function<void(const Band&, const Tile&, uint32_t)>& post) {
        const Tile weights = lin.Load(*program_, bkey);
        for (const Band& band : Bands(cf.nap, kTile)) {
          const uint32_t q0 = band.i0 / 4;
          const uint32_t rq = band.m / 4;
          const Tile c = program_->Alloc(Mem::kFbuf, rq * lin.Groups());
          lin.Gemm(*program_, lin.Input(src.At()), q0 * kKbD, kKbD, band.m,
                   c.At(), weights, std::format("{} t{}", bkey, band.t));
          program_->Vec({.op = VecOp::kAddP,
                         .src = c.At(),
                         .dst = c.At(),
                         .rows_q = rq,
                         .n8 = lin.Groups(),
                         .p_base = pb.At(bkey)},
                        std::format("{} bias t{}", bkey, band.t));
          if (post) {
            post(band, c, lin.Groups());
          }
          StoreRows(rows_out, band.i0, band.m, c.At(),
                    std::format("{} st t{}", bkey, band.t));
        }
      };
  Tile src = chain(
      layer.reg,
      [&](size_t i) { return std::format("{} reg_ln{}", note, i / 2); }, xe,
      std::format("{} reg", note), [](size_t i) { return i % 2 == 1; });
  head_out(layer.reg_out, src, a_out, std::format("{} reg_out", note),
           [&](const Band& band, const Tile& c, uint32_t n8) {
             const uint32_t rq = band.m / 4;
             const Tile anchor =
                 LoadRows(a_in, band.i0, band.m,
                          std::format("{} ld anchor t{}", note, band.t));
             program_->Vec({.op = VecOp::kMulP,
                            .src = c.At(),
                            .dst = c.At(),
                            .rows_q = rq,
                            .n8 = n8,
                            .p_base = pb.At(std::format("{} scale", note))},
                           std::format("{} scale t{}", note, band.t));
             program_->Vec({.op = VecOp::kAdd,
                            .src = c.At(),
                            .src2 = anchor.At(),
                            .dst = c.At(),
                            .rows_q = rq,
                            .n8 = n8},
                           std::format("{} + anchor t{}", note, band.t));
           });
  src = chain(
      layer.cls, [&](size_t i) { return std::format("{} cls_ln{}", note, i); },
      ax, std::format("{} cls", note), [](size_t) { return true; });
  head_out(layer.cls_out, src, cls, std::format("{} cls_out", note), {});
  if (cf.det) {
    src = chain(
        layer.qt, [&](size_t i) { return std::format("{} qt_ln{}", note, i); },
        xe, std::format("{} qt", note), [](size_t) { return true; });
    head_out(layer.qt_out, src, Required(qt, "quality rows"),
             std::format("{} qt_out", note), {});
  }
  cur_a = 1 - cur_a;
}

void SparseHead::Reorder(std::string_view note) {
  const Rows& xr = x[cur_x];
  const Rows& xo = x[(cur_x + 1) % 3];
  const Rows& a_in = anc[cur_a];
  const Rows& a_out = anc[1 - cur_a];
  for (const Band& band : Bands(C().nap, 128)) {
    const Tile rows = program_->Alloc(Mem::kFbuf, band.m / 4 * (xr.cols / 8));
    program_->Load(Tokens(order_use[0] + (uint64_t{band.i0} * 4), Mem::kFbuf,
                          rows.At(), band.m, xr.Tok(), true),
                   std::format("{} gather x t{}", note, band.t));
    StoreRows(xo, band.i0, band.m, rows.At(),
              std::format("{} st x t{}", note, band.t));
  }
  const Tile anchors =
      program_->Alloc(Mem::kFbuf, C().nap / 4 * (a_in.cols / 8));
  program_->Load(
      Tokens(order_use[1], Mem::kFbuf, anchors.At(), C().nap, a_in.Tok(), true),
      std::format("{} gather anchors", note));
  StoreRows(a_out, 0, C().nap, anchors.At(),
            std::format("{} st anchors", note));
  cur_x = (cur_x + 1) % 3;
  cur_a = 1 - cur_a;
}

void SparseHead::Keep(std::string_view note) {
  const Rows& xr = x[cur_x];
  for (const Band& band : Bands(C().ntp, 128)) {
    const Tile rows = program_->Alloc(Mem::kFbuf, band.m / 4 * (xr.cols / 8));
    program_->Load(Tokens(keep + (uint64_t{band.i0} * 4), Mem::kFbuf, rows.At(),
                          band.m, xr.Tok(), true),
                   std::format("{} gather keep t{}", note, band.t));
    StoreRows(tx, band.i0, band.m, rows.At(),
              std::format("{} st temp_x t{}", note, band.t));
  }
}

void SparseHead::Init(std::string_view note) {
  for (const Band& band : Bands(C().nap, 128)) {
    const Tile rows = LoadRows(x_init, band.i0, band.m,
                               std::format("{} ld x_init t{}", note, band.t));
    StoreRows(x[0], band.i0, band.m, rows.At(),
              std::format("{} st x t{}", note, band.t));
  }
  const Tile anchors =
      LoadRows(a_init, 0, C().nap, std::format("{} ld anchor_init", note));
  StoreRows(anc[0], 0, C().nap, anchors.At(),
            std::format("{} st anchor", note));
  cur_x = 0;
  cur_a = 0;
}

void SparseHead::KeepAnchor(std::string_view note) {
  const Rows& a = anc[cur_a];
  for (const Band& band : Bands(C().ntp, 128)) {
    const Tile rows = program_->Alloc(Mem::kFbuf, band.m / 4 * (a.cols / 8));
    program_->Load(Tokens(keep_a + (uint64_t{band.i0} * 4), Mem::kFbuf,
                          rows.At(), band.m, a.Tok(), true),
                   std::format("{} gather t{}", note, band.t));
    StoreRows(tanc_raw, band.i0, band.m, rows.At(),
              std::format("{} st t{}", note, band.t));
  }
}

void SparseHead::AnchorProject(std::string_view note) {
  const uint32_t words = C().acols / 8;
  const uint32_t n = C().ntp;
  EnsureCbuf(kCbufProject);
  const Tile params = program_->Alloc(Mem::kPbuf, C().det ? 18 : 9);
  program_->Load(Words(proj_pb, Mem::kPbuf, params.At(), params.Words()),
                 std::format("{} ld PBUF", note));
  const Tile dt = program_->Alloc(Mem::kFbuf, 1);
  program_->Load(Words(proj_dt, Mem::kFbuf, dt.At(), 1),
                 std::format("{} ld dt", note));
  const Tile raw = LoadRows(tanc_raw, 0, n, std::format("{} ld raw", note));
  const Tile out = program_->Alloc(Mem::kFbuf, n / 4 * words);
  program_->Run({.pc = pj.Pc("project"),
                 .src = raw.At(),
                 .src2 = dt.At(),
                 .dst = out.At(),
                 .rows_q = n / 4,
                 .n8 = words,
                 .p_base = params.At(),
                 .imm_places = {},
                 .reads = {Whole(raw), Whole(dt), Whole(out), Whole(params)},
                 .writes = {Whole(out)}},
                std::format("{} run", note));
  StoreRows(tanc, 0, n, out.At(), std::format("{} st temp_anchor", note));
}

void SparseHead::EnsureCbuf(int which) {
  if (cbuf_has_ == which) {
    return;
  }
  const auto load = [&](uint64_t offset, const CbufImage& image,
                        std::string_view what) {
    program_->Load(Words(offset, Mem::kCbuf, Place::Fixed(vprog::kRomWords),
                         static_cast<uint32_t>(image.Words().size())),
                   std::format("{} ld VEC programs ({})", tag, what));
  };
  if (which == kCbufKeypoints) {
    load(vprog, kp, "keypoints");
  } else if (which == kCbufBank) {
    load(ibprog, ib, "instance bank");
  } else {
    load(pjprog, pj, "anchor projection");
  }
  cbuf_has_ = which;
}

void SparseHead::Bank(bool update, std::string_view note) {
  const HeadConfig& cf = C();
  const Rows& xr = x[cur_x];
  const Rows& ar = anc[cur_a];
  EnsureCbuf(kCbufBank);
  const uint32_t work = cf.nap;
  const uint32_t out = work + (6 * nw) + 3;
  const uint32_t cst = out + (5 * nw);
  const uint32_t flag = cst + 1;
  const uint32_t identity = cst + 2;
  const Tile area =
      program_->Alloc(Mem::kFbuf, update ? identity + (2 * nw) : cst + 1);
  const auto key = std::tuple{update, cur_x, cur_a};
  if (!ib_const_.contains(key)) {
    const uint64_t offset =
        image_.Alloc(std::format("{}.ib_const.{}{}{}", tag,
                                 update ? "update" : "cache", cur_x, cur_a),
                     kWordF);
    PutWords(image_, offset,
             {IbankConstWord(
                 Entry(base_, xr.Addr(0)), static_cast<uint32_t>(xr.Tb() / 16),
                 Entry(base_, ar.Addr(0)), static_cast<uint32_t>(ar.Tb() / 16),
                 cf.na, cf.ntemp)});
    ib_const_[key] = offset;
  }
  LoadRows(cls, 0, cf.nap, area.At(), std::format("{} ld cls", note));
  program_->Load(Words(ib_idx, Mem::kFbuf, area.At(work + nw), nw),
                 std::format("{} ld idx", note));
  program_->Load(Words(ib_lane, Mem::kFbuf, area.At(work + (6 * nw) + 2), 1),
                 std::format("{} ld lane", note));
  if (update) {
    program_->Load(Words(order[0], Mem::kFbuf, area.At(out), nw),
                   std::format("{} ld order0", note));
    program_->Load(Words(order[1], Mem::kFbuf, area.At(out + nw), nw),
                   std::format("{} ld order1", note));
  } else {
    program_->Load(Words(keep, Mem::kFbuf, area.At(out), nw),
                   std::format("{} ld keep", note));
    program_->Load(Words(prevconf, Mem::kFbuf, area.At(out + (3 * nw)), nw),
                   std::format("{} ld prevconf", note));
    program_->Load(Words(keep_a, Mem::kFbuf, area.At(out + (4 * nw)), nw),
                   std::format("{} ld keep_a", note));
  }
  program_->Load(Words(ib_const_.at(key), Mem::kFbuf, area.At(cst), 1),
                 std::format("{} ld const", note));
  program_->Run(
      {.pc = ib.Pc(update ? "update" : "cache"),
       .src = area.At(),
       .src2 = area.At(work),
       .dst = area.At(out),
       .rows_q = cf.nap / 4,
       .n8 = ncls,
       .imm_places = {{.place = area.At(cst), .shift = 0}},
       .reads = {Part(area, 0, cst + 1), Fixed(Mem::kTbuf, 0, kTbufWords)},
       .writes = {Part(area, work, cst - work)}},
      std::format("{} run", note));
  if (update) {
    program_->Store(Words(order[0], Mem::kFbuf, area.At(out), nw),
                    std::format("{} st order0", note));
    program_->Store(Words(order[1], Mem::kFbuf, area.At(out + nw), nw),
                    std::format("{} st order1", note));
    const auto id_key = std::pair{cur_x, cur_a};
    if (!order_id_.contains(id_key)) {
      std::vector<uint32_t> ids(2 * static_cast<size_t>(cf.nap), kNullEntry);
      for (uint32_t i = 0; i < cf.na; ++i) {
        ids[i] = Entry(base_, xr.Addr(i));
        ids[cf.nap + i] = Entry(base_, ar.Addr(i));
      }
      const uint64_t offset =
          image_.Alloc(std::format("{}.order_id{}{}", tag, cur_x, cur_a),
                       uint64_t{2} * cf.nap * 4);
      PutEntries(image_, offset, ids);
      order_id_[id_key] = offset;
    }
    program_->Load(Words(fs_flag, Mem::kFbuf, area.At(flag), 1),
                   std::format("{} ld flag", note));
    program_->Load(
        Words(order_id_.at(id_key), Mem::kFbuf, area.At(identity), 2 * nw),
        std::format("{} ld identity", note));
    for (uint32_t i = 0; i < 2; ++i) {
      program_->Run(
          {.pc = ib.Pc("copy"),
           .src = area.At(identity + (i * nw)),
           .src2 = area.At(flag),
           .dst = area.At(out + (i * nw)),
           .rows_q = nw,
           .imm_places = {},
           .reads = {Part(area, identity + (i * nw), nw), Part(area, flag, 1)},
           .writes = {Part(area, out + (i * nw), nw)}},
          std::format("{} identity{} if !mask", note, i));
      program_->Store(
          Words(order_use[i], Mem::kFbuf, area.At(out + (i * nw)), nw),
          std::format("{} st order_use{}", note, i));
    }
  } else {
    program_->Store(Words(keep, Mem::kFbuf, area.At(out), nw),
                    std::format("{} st keep", note));
    program_->Store(Words(prevconf, Mem::kFbuf, area.At(out + nw), nw),
                    std::format("{} st selconf", note));
    program_->Store(Words(selidx, Mem::kFbuf, area.At(out + (2 * nw)), nw),
                    std::format("{} st selidx", note));
    program_->Store(Words(keep_a, Mem::kFbuf, area.At(out + (4 * nw)), nw),
                    std::format("{} st keep_a", note));
  }
}

void SparseHead::EmitFrame(Program& program, bool temp) {
  program_ = &program;
  temp_ = temp;
  cur_x = 0;
  cur_a = 0;
  {
    const Program::Scope scope(program, "init");
    program.Load(
        {.ddr_offset = tb,
         .mem = Mem::kTbuf,
         .onchip = Place::Fixed(0),
         .rows = 1,
         .row_words = kTbufWords,
         .ddr_stride = static_cast<uint32_t>(uint64_t{kTbufWords} * kWordT)},
        std::format("{} ld TBUF", tag));
    cbuf_has_ = 0;
    if (temp) {
      AnchorProject(std::format("{} temp anchor proj", tag));
    }
    Init(std::format("{} init", tag));
    EncodeAnchors(program, anc[cur_a], e, C().nap, std::format("{} E0", tag));
  }
  uint32_t refines = 0;
  for (size_t l = 0; l < p.layers.size(); ++l) {
    const HeadLayer& layer = p.layers[l];
    const auto& [pb, keys, note] = pb_layers[l];
    const Program::Scope scope(program,
                               std::format("L{}.{}", l, OpName(layer.op)));
    const Params params(pb.Load(program, note), keys);
    switch (layer.op) {
      case Op::kGnn:
      case Op::kTempGnn:
        Attention(layer, params, note);
        break;
      case Op::kNorm:
        Norm(params, note);
        break;
      case Op::kDeformable:
        CameraEncoder(layer, params, note + " cam");
        program.Load(Words(cam_vec.Addr(0), Mem::kPbuf,
                           params.At(note + " cam"), kNwD / 4),
                     std::format("{} ld cam -> PBUF", note));
        EnsureCbuf(kCbufKeypoints);
        Deformable(layer, l, params, note);
        break;
      case Op::kFfn:
        Ffn(layer, params, note);
        break;
      case Op::kCrossGnn:
        break;
      case Op::kRefine:
        Refine(layer, params, note);
        ++refines;
        if (refines == 1 && temp) {
          Bank(true, std::format("{} ibank update", tag));
          Reorder(std::format("{} reorder", tag));
        }
        EncodeAnchors(program, anc[cur_a], e, C().nap,
                      std::format("{} E after L{}", tag, l));
        break;
    }
  }
  const Program::Scope scope(program, "end");
  Bank(false, std::format("{} ibank cache", tag));
  Keep(std::format("{} keep", tag));
  KeepAnchor(std::format("{} keep anchor", tag));
}

}  // namespace hotarubi::compiler::sparsedrive
