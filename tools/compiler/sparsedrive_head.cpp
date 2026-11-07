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
  w.fill(kNull);
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
    std::vector<uint32_t> entries(cf.nap, kNull);
    for (uint32_t i = 0; i < cf.ntemp; ++i) {
      entries[i] = Entry(base_, rows.Addr(i));
    }
    PutEntries(image, order[k], entries);
  }
  PutEntries(image, keep, std::vector<uint32_t>(cf.ntp, kNull));
  const auto block = [&](uint32_t camera, uint32_t slots) {
    std::vector<uint32_t> entries = {
        Entry(base_, w) + (camera * (cf.scam / 8) * 8),
        (wg & 0xFFFFU) | ((wq & 0xFFFFU) << 16), 0, 0};
    for (uint32_t i = 0; i < slots * 16; ++i) {
      entries.push_back(kNull);
      entries.push_back(0);
    }
    return entries;
  };
  std::vector<uint32_t> all;
  for (uint32_t q = 0; q < nq; ++q) {
    for (uint32_t camera = 0; camera < ncam; ++camera) {
      const std::vector<uint32_t> one = block(camera, cf.scam);
      all.insert(all.end(), one.begin(), one.end());
    }
  }
  PutEntries(image, lists, all);
  null_list = image.Alloc(tag + ".gacc_null", 16 + (8 * 16));
  PutEntries(image, null_list, block(0, 1));
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

void SparseHead::LoadRows(const Rows& rows, uint32_t j, uint32_t m,
                          uint32_t dst, std::string_view note) {
  program_->Load(Tokens(rows.Addr(j), Mem::kFbuf, dst, m, rows.Tok()), note);
}

void SparseHead::StoreRows(const Rows& rows, uint32_t j, uint32_t m,
                           uint32_t src, std::string_view note) {
  program_->Store(Tokens(rows.Addr(j), Mem::kFbuf, src, m, rows.Tok()), note);
}

void SparseHead::StoreColumns(const Rows& rows, uint32_t j, uint32_t m,
                              uint32_t src, uint32_t col0, uint32_t ncols,
                              std::string_view note) {
  Require(ncols % 32 == 0 && col0 % 32 == 0, "unaligned column store");
  program_->Store({.ddr_offset = rows.Addr(j) + (uint64_t{col0} * 4),
                   .mem = Mem::kFbuf,
                   .onchip = src,
                   .rows = m,
                   .row_words = 1,
                   .ddr_stride = static_cast<uint32_t>(rows.Tb()),
                   .mode = DmaMode::kRaw,
                   .index = false,
                   .row_format = true,
                   .tok = ncols / 32},
                  note);
}

void SparseHead::StoreColumnsStrided(const Rows& rows, uint32_t j, uint32_t m,
                                     uint32_t src, uint32_t n8_src,
                                     uint32_t col0, uint32_t ncols,
                                     std::string_view note) {
  const uint32_t words = ncols / 8;
  if (n8_src == words && col0 == 0) {
    StoreColumns(rows, j, m, src, col0, ncols, note);
    return;
  }
  constexpr uint32_t kTmp = 3584;
  const uint32_t per = std::max(1U, (kFbufWords - kTmp) / words);
  for (uint32_t q0 = 0; q0 < m / 4; q0 += per) {
    const uint32_t count = std::min(per, (m / 4) - q0);
    for (uint32_t q = 0; q < count; ++q) {
      program_->Vec({.op = VecOp::kCopy,
                     .src = src + ((q0 + q) * n8_src) + (col0 / 8),
                     .dst = kTmp + (q * words),
                     .rows_q = 1,
                     .n8 = words},
                    std::format("{} pack q{}", note, q0 + q));
    }
    StoreColumns(rows, j + (4 * q0), 4 * count, kTmp, col0, ncols,
                 std::format("{} q{}", note, q0));
  }
}

void SparseHead::QuantRows(
    const Rows& rows, uint32_t n, uint32_t abuf, std::string_view note,
    const std::function<void(uint32_t, uint32_t, uint32_t, uint32_t)>& extra) {
  constexpr std::array<uint32_t, 2> kSlots = {0, 1024};
  const uint32_t words = rows.cols / 8;
  const uint32_t blocks = rows.cols / 32;
  for (const Tile& tile : Tiles(n, 128)) {
    const uint32_t slot = kSlots[tile.t % 2];
    LoadRows(rows, tile.i0, tile.m, slot,
             std::format("{} ld t{}", note, tile.t));
    if (extra) {
      extra(tile.t, slot, tile.m / 4, tile.i0);
    }
    program_->Vec({.op = VecOp::kQuant,
                   .src = slot,
                   .dst = abuf + ((tile.i0 / 4) * blocks),
                   .rows_q = tile.m / 4,
                   .n8 = words},
                  std::format("{} quant t{}", note, tile.t));
  }
}

void SparseHead::QuantRowsAdd(const Rows& rows, const Rows& rows2, uint32_t n,
                              uint32_t abuf, std::string_view note,
                              std::optional<uint32_t> abuf_plain,
                              std::optional<uint32_t> pb_vec) {
  constexpr std::array<uint32_t, 2> kSecond = {2048, 3072};
  const uint32_t words = rows.cols / 8;
  const uint32_t blocks = rows.cols / 32;
  QuantRows(rows, n, abuf, note,
            [&](uint32_t t, uint32_t slot, uint32_t rq, uint32_t i0) {
              LoadRows(rows2, i0, 4 * rq, kSecond[t % 2],
                       std::format("{} ld 2nd t{}", note, t));
              if (abuf_plain.has_value()) {
                program_->Vec({.op = VecOp::kQuant,
                               .src = slot,
                               .dst = *abuf_plain + ((i0 / 4) * blocks),
                               .rows_q = rq,
                               .n8 = words},
                              std::format("{} Q(plain) t{}", note, t));
              }
              program_->Vec({.op = VecOp::kAdd,
                             .src = slot,
                             .src2 = kSecond[t % 2],
                             .dst = slot,
                             .rows_q = rq,
                             .n8 = words},
                            std::format("{} add t{}", note, t));
              if (pb_vec.has_value()) {
                program_->Vec({.op = VecOp::kAddP,
                               .src = slot,
                               .dst = slot,
                               .rows_q = rq,
                               .n8 = words,
                               .p_base = *pb_vec},
                              std::format("{} addp t{}", note, t));
              }
            });
}

void SparseHead::AnchorEncoder(std::string_view note) {
  constexpr uint32_t kTile = 224;
  constexpr uint32_t kAt = 0;
  constexpr uint32_t kCt = 512;
  constexpr uint32_t kScratch = 2304;
  constexpr uint32_t kA1 = 0;
  const uint32_t a2 = (kTile / 4) * (C().acols / 32);
  const Rows& rows = anc[cur_a];
  pb_enc.Load(*program_, note);
  for (const Tile& tile : Tiles(C().nap, kTile)) {
    const uint32_t rq = tile.m / 4;
    LoadRows(rows, tile.i0, tile.m, kAt,
             std::format("{} ld anchors t{}", note, tile.t));
    for (const auto& branch : p.enc) {
      program_->Vec(
          {.op = VecOp::kMulP,
           .src = kAt,
           .dst = kA1,
           .rows_q = rq,
           .n8 = C().acols / 8,
           .p_base = keys_enc.at("mask." + branch.name),
           .q = true},
          std::format("{} {} select+Q t{}", note, branch.name, tile.t));
      for (size_t i = 0; i < branch.chain.size(); ++i) {
        const Lin& lin = branch.chain[i].first;
        const std::string where = std::format("{} {}{}", note, branch.name, i);
        lin.Load(*program_, 0, where);
        lin.Gemm(*program_, i == 0 ? kA1 : a2,
                 i == 0 ? C().acols / 32 : lin.Kb(), tile.m, kCt, 0,
                 std::format("{} t{}", where, tile.t));
        const uint32_t words = lin.Groups();
        program_->Vec(
            {.op = VecOp::kAddP,
             .src = kCt,
             .dst = kCt,
             .rows_q = rq,
             .n8 = words,
             .p_base = keys_enc.at(std::format("enc.{}.b{}", branch.name, i))},
            std::format("{} bias t{}", where, tile.t));
        program_->Vec({.op = VecOp::kRelu,
                       .src = kCt,
                       .dst = kCt,
                       .rows_q = rq,
                       .n8 = words},
                      std::format("{} relu t{}", where, tile.t));
        const uint32_t ln =
            keys_enc.at(std::format("enc.{}.ln{}", branch.name, i));
        if (i + 1 < branch.chain.size()) {
          program_->Vec({.op = VecOp::kLayerNorm,
                         .src = kCt,
                         .dst = a2,
                         .rows_q = rq,
                         .n8 = words,
                         .p_base = ln,
                         .imm = LnEpsBits(),
                         .q = true,
                         .scratch = kScratch},
                        std::format("{} LN+Q t{}", where, tile.t));
        } else {
          program_->Vec({.op = VecOp::kLayerNorm,
                         .src = kCt,
                         .dst = kScratch,
                         .rows_q = rq,
                         .n8 = words,
                         .p_base = ln,
                         .imm = LnEpsBits()},
                        std::format("{} LN t{}", where, tile.t));
          StoreColumns(
              e, tile.i0, tile.m, kScratch, branch.col0, lin.Rows(),
              std::format("{} {} st E t{}", note, branch.name, tile.t));
        }
      }
    }
  }
}

void SparseHead::Attention(const HeadLayer& layer, const Keys& pb,
                           std::string_view note) {
  const HeadConfig& cf = C();
  const Rows& xr = x[cur_x];
  const bool is_temp = layer.op == Op::kTempGnn && temp_;
  const bool raw_temp = layer.op == Op::kTempGnn && !temp_;
  const uint32_t keys = is_temp ? cf.ntp : cf.nap;
  const uint32_t valid = is_temp ? cf.ntemp : cf.na;
  const bool dec = cf.det;
  OnChip a(kAbufWords);
  std::vector<Piece> qp;
  uint32_t ax = 0;
  uint32_t ae = 0;
  if (dec) {
    ax = a.Take(cf.nap / 4 * kKbD);
    ae = a.Take(cf.nap / 4 * kKbD);
    QuantRows(xr, cf.nap, ax, std::format("{} Q(x)", note));
    QuantRows(e, cf.nap, ae, std::format("{} Q(E)", note));
    qp = {{.a_offset = ax, .kb_offset = 0, .kb = kKbD},
          {.a_offset = ae, .kb_offset = kKbD, .kb = kKbD}};
  } else {
    ax = a.Take(cf.nap / 4 * kKbD);
    const uint32_t aq = a.Take(cf.nap / 4 * kKbD);
    QuantRowsAdd(xr, e, cf.nap, aq, std::format("{} Q(x+E)", note), ax);
    qp = {{.a_offset = aq, .kb_offset = 0, .kb = kKbD}};
  }
  const std::array<uint32_t, 2> tmp = {a.Take(16 * kbq), a.Take(16 * kbq)};
  constexpr std::array<uint32_t, 2> kCb = {0, 1024};
  const auto proj = [&](const Lin& lin, const std::vector<Piece>& pieces,
                        uint32_t astride, uint32_t n, uint64_t out,
                        const std::string& name) {
    lin.Load(*program_, 0, name);
    for (const Tile& tile : Tiles(n, 64)) {
      const uint32_t c = kCb[tile.t % 2];
      const uint32_t q0 = tile.i0 / 4;
      lin.Gemm(*program_, q0 * astride, astride, tile.m, c, 0,
               std::format("{} t{}", name, tile.t), pieces);
      program_->Vec({.op = VecOp::kAddP,
                     .src = c,
                     .dst = tmp[tile.t % 2],
                     .rows_q = tile.m / 4,
                     .n8 = nwq,
                     .p_base = pb.at(name),
                     .q = true},
                    std::format("{} bias+Q t{}", name, tile.t));
      program_->Store(Words(out + (uint64_t{q0} * kbq * kWordA), Mem::kAbuf,
                            tmp[tile.t % 2], (tile.m / 4) * kbq),
                      std::format("{} st t{}", name, tile.t));
    }
  };
  proj(layer.attn[0], qp, kKbD, cf.nap, qb, std::format("{} q", note));
  std::vector<Piece> kp_pieces = qp;
  uint32_t vsrc = ax;
  if (is_temp) {
    if (dec) {
      const uint32_t atx = a.Take(cf.ntp / 4 * kKbD);
      QuantRows(tx, cf.ntp, atx, std::format("{} Q(temp_x)", note));
      kp_pieces = {{.a_offset = atx, .kb_offset = 0, .kb = kKbD},
                   {.a_offset = ae, .kb_offset = kKbD, .kb = kKbD}};
      vsrc = atx;
    } else {
      const uint32_t atx = a.Take(cf.ntp / 4 * kKbD);
      const uint32_t ak = a.Take(cf.ntp / 4 * kKbD);
      QuantRowsAdd(tx, e, cf.ntp, ak, std::format("{} Q(temp_x+E)", note), atx);
      kp_pieces = {{.a_offset = ak, .kb_offset = 0, .kb = kKbD}};
      vsrc = atx;
    }
  }
  proj(layer.attn[1], kp_pieces, kKbD, keys, kb, std::format("{} k", note));
  if (dec && !raw_temp) {
    const uint32_t av = a.Take(keys / 4 * kbq);
    const Lin& before = Required(p.fc_before, "fc_before");
    before.Load(*program_, 0, std::format("{} fc_before", note));
    for (const Tile& tile : Tiles(keys, 64)) {
      const uint32_t c = kCb[tile.t % 2];
      const uint32_t q0 = tile.i0 / 4;
      before.Gemm(*program_, vsrc + (q0 * kKbD), kKbD, tile.m, c, 0,
                  std::format("{} fc_before t{}", note, tile.t));
      program_->Vec({.op = VecOp::kQuant,
                     .src = c,
                     .dst = av + (q0 * kbq),
                     .rows_q = tile.m / 4,
                     .n8 = nwq},
                    std::format("{} fc_before Q t{}", note, tile.t));
    }
    proj(layer.attn[2], {{.a_offset = av, .kb_offset = 0, .kb = kbq}}, kbq,
         keys, vb, std::format("{} v", note));
  } else if (dec) {
    proj(layer.attn[2], qp, kKbD, cf.nap, vb, std::format("{} v", note));
  } else {
    proj(layer.attn[2], {{.a_offset = vsrc, .kb_offset = 0, .kb = kKbD}}, kKbD,
         keys, vb, std::format("{} v", note));
  }
  EmitAttention(*program_,
                {.q = qb,
                 .k = kb,
                 .v = vb,
                 .o = ob,
                 .queries = cf.nap,
                 .embed = cf.dq,
                 .hd = cf.hd,
                 .keys = keys,
                 .valid = valid},
                note);
  OutProjection(layer, pb, note);
}

void SparseHead::OutProjection(const HeadLayer& layer, const Keys& pb,
                               std::string_view note) {
  const HeadConfig& cf = C();
  const Rows& xr = x[cur_x];
  const Rows& xo = x[(cur_x + 1) % 3];
  OnChip a(kAbufWords);
  const uint32_t ao = a.Take(cf.nap / 4 * kbq);
  const std::array<uint32_t, 2> ot = nwq == 64
                                         ? std::array<uint32_t, 2>{0, 2048}
                                         : std::array<uint32_t, 2>{0, 1024};
  for (const Tile& tile : Tiles(cf.nap, 128)) {
    const uint32_t slot = ot[tile.t % 2];
    const uint32_t q0 = tile.i0 / 4;
    program_->Load(Words(ob + (uint64_t{q0} * nwq * kWordF), Mem::kFbuf, slot,
                         (tile.m / 4) * nwq),
                   std::format("{} ld O t{}", note, tile.t));
    program_->Vec({.op = VecOp::kQuant,
                   .src = slot,
                   .dst = ao + (q0 * kbq),
                   .rows_q = tile.m / 4,
                   .n8 = nwq},
                  std::format("{} Q(O) t{}", note, tile.t));
  }
  const Lin& op = layer.attn[3];
  op.Load(*program_, 0, std::format("{} out_proj", note));
  constexpr std::array<uint32_t, 2> kCt = {0, 512};
  constexpr std::array<uint32_t, 2> kXt = {1024, 1536};
  const uint32_t pbo = pb.at(std::format("{} o", note));
  if (cf.det) {
    const std::array<uint32_t, 2> af = {a.Take(cf.nap / 4 * kKbD),
                                        a.Take(cf.nap / 4 * kKbD)};
    for (uint32_t half = 0; half < 2; ++half) {
      const Rows& res = half == 0 ? xr : e;
      for (const Tile& tile : Tiles(cf.nap, 64)) {
        const uint32_t c = kCt[tile.t % 2];
        const uint32_t xt = kXt[tile.t % 2];
        const uint32_t q0 = tile.i0 / 4;
        program_->Gemm({.a_base = ao + (q0 * kbq),
                        .a_stride = kbq,
                        .kb = kbq,
                        .m = tile.m,
                        .w_base = half * 32 * op.Kb(),
                        .w_stride = op.Kb(),
                        .n8 = 32,
                        .c_base = c},
                       std::format("{} out_proj h{} t{}", note, half, tile.t));
        LoadRows(res, tile.i0, tile.m, xt,
                 std::format("{} ld identity h{} t{}", note, half, tile.t));
        program_->Vec(
            {.op = VecOp::kAddP,
             .src = c,
             .dst = c,
             .rows_q = tile.m / 4,
             .n8 = kNwD,
             .p_base = pbo + (half * (kNwD / 4))},
            std::format("{} out_proj bias h{} t{}", note, half, tile.t));
        program_->Vec({.op = VecOp::kAdd,
                       .src = c,
                       .src2 = xt,
                       .dst = c,
                       .rows_q = tile.m / 4,
                       .n8 = kNwD},
                      std::format("{} + identity h{} t{}", note, half, tile.t));
        program_->Vec({.op = VecOp::kQuant,
                       .src = c,
                       .dst = af[half] + (q0 * kKbD),
                       .rows_q = tile.m / 4,
                       .n8 = kNwD},
                      std::format("{} Q h{} t{}", note, half, tile.t));
      }
    }
    const Lin& after = Required(p.fc_after, "fc_after");
    after.Load(*program_, 0, std::format("{} fc_after", note));
    for (const Tile& tile : Tiles(cf.nap, 128)) {
      const uint32_t c = tile.t % 2 == 0 ? 0 : 1024;
      after.Gemm(*program_, (tile.i0 / 4) * kKbD, kKbD, tile.m, c, 0,
                 std::format("{} fc_after t{}", note, tile.t),
                 {{.a_offset = af[0], .kb_offset = 0, .kb = kKbD},
                  {.a_offset = af[1], .kb_offset = kKbD, .kb = kKbD}});
      StoreRows(xo, tile.i0, tile.m, c,
                std::format("{} st x t{}", note, tile.t));
    }
  } else {
    for (const Tile& tile : Tiles(cf.nap, 64)) {
      const uint32_t c = kCt[tile.t % 2];
      const uint32_t xt = kXt[tile.t % 2];
      const uint32_t q0 = tile.i0 / 4;
      op.Gemm(*program_, ao + (q0 * kbq), kbq, tile.m, c, 0,
              std::format("{} out_proj t{}", note, tile.t));
      LoadRows(xr, tile.i0, tile.m, xt,
               std::format("{} ld identity t{}", note, tile.t));
      program_->Vec({.op = VecOp::kAddP,
                     .src = c,
                     .dst = c,
                     .rows_q = tile.m / 4,
                     .n8 = kNwD,
                     .p_base = pbo},
                    std::format("{} out_proj bias t{}", note, tile.t));
      program_->Vec({.op = VecOp::kAdd,
                     .src = c,
                     .src2 = xt,
                     .dst = c,
                     .rows_q = tile.m / 4,
                     .n8 = kNwD},
                    std::format("{} + identity t{}", note, tile.t));
      StoreRows(xo, tile.i0, tile.m, c,
                std::format("{} st x t{}", note, tile.t));
    }
  }
  cur_x = (cur_x + 1) % 3;
}

void SparseHead::Norm(const Keys& pb, std::string_view note) {
  constexpr std::array<uint32_t, 2> kXt = {0, 1024};
  constexpr uint32_t kScratch = 2048;
  const Rows& xr = x[cur_x];
  const Rows& xo = x[(cur_x + 1) % 3];
  for (const Tile& tile : Tiles(C().nap, 128)) {
    const uint32_t slot = kXt[tile.t % 2];
    LoadRows(xr, tile.i0, tile.m, slot, std::format("{} ld t{}", note, tile.t));
    program_->Vec({.op = VecOp::kLayerNorm,
                   .src = slot,
                   .dst = kScratch,
                   .rows_q = tile.m / 4,
                   .n8 = kNwD,
                   .p_base = pb.at(std::string(note)),
                   .imm = LnEpsBits()},
                  std::format("{} LN t{}", note, tile.t));
    StoreRows(xo, tile.i0, tile.m, kScratch,
              std::format("{} st t{}", note, tile.t));
  }
  cur_x = (cur_x + 1) % 3;
}

void SparseHead::CameraEncoder(const HeadLayer& layer, const Keys& pb,
                               std::string_view note) {
  LoadRows(cam_in, 0, 32, 0, std::format("{} ld cam_in", note));
  program_->Vec({.op = VecOp::kQuant, .src = 0, .dst = 0, .rows_q = 8, .n8 = 4},
                std::format("{} Q", note));
  for (size_t i = 0; i < layer.cam.size(); ++i) {
    const Lin& lin = layer.cam[i].first;
    lin.Load(*program_, 0, std::format("{} {}", note, i));
    lin.Gemm(*program_, i == 0 ? 0 : 8, i == 0 ? 1 : kKbD, 32, 0, 0,
             std::format("{} {}", note, i));
    program_->Vec({.op = VecOp::kAddP,
                   .src = 0,
                   .dst = 0,
                   .rows_q = 8,
                   .n8 = kNwD,
                   .p_base = pb.at(std::format("{} b{}", note, i))},
                  std::format("{} bias {}", note, i));
    program_->Vec(
        {.op = VecOp::kRelu, .src = 0, .dst = 0, .rows_q = 8, .n8 = kNwD},
        std::format("{} relu {}", note, i));
    const uint32_t ln = pb.at(std::format("{} ln{}", note, i));
    if (i == 0) {
      program_->Vec({.op = VecOp::kLayerNorm,
                     .src = 0,
                     .dst = 8,
                     .rows_q = 8,
                     .n8 = kNwD,
                     .p_base = ln,
                     .imm = LnEpsBits(),
                     .q = true,
                     .scratch = 256},
                    std::format("{} LN+Q {}", note, i));
    } else {
      program_->Vec({.op = VecOp::kLayerNorm,
                     .src = 0,
                     .dst = 256,
                     .rows_q = 8,
                     .n8 = kNwD,
                     .p_base = ln,
                     .imm = LnEpsBits()},
                    std::format("{} LN {}", note, i));
    }
  }
  StoreRows(cam_vec, 0, 32, 256, std::format("{} st cam_vec", note));
}

void SparseHead::Deformable(const HeadLayer& layer, size_t index,
                            const Keys& pb, std::string_view note) {
  const HeadConfig& cf = C();
  const Rows& xr = x[cur_x];
  OnChip a(kAbufWords);
  const uint32_t ax = a.Take(cf.nap / 4 * kKbD);
  const uint32_t aw = a.Take(cf.nap / 4 * kKbD);
  QuantRowsAdd(xr, e, cf.nap, aw, std::format("{} Q(x+E+cam)", note), ax,
               pb.at(std::format("{} cam", note)));
  const Lin& lin = layer.lfc;
  lin.Load(*program_, 0, std::format("{} lfc", note));
  const uint32_t n8l = lin.Groups();
  const uint32_t mtl =
      std::max(32U, std::min(448U, (kFbufWords / 2 / n8l) * 4 / 32 * 32));
  for (const Tile& tile : Tiles(cf.nap, mtl)) {
    const uint32_t c = tile.t % 2 == 0 ? 0 : (mtl / 4) * n8l;
    lin.Gemm(*program_, ax + ((tile.i0 / 4) * kKbD), kKbD, tile.m, c, 0,
             std::format("{} lfc t{}", note, tile.t));
    program_->Vec({.op = VecOp::kAddP,
                   .src = c,
                   .dst = c,
                   .rows_q = tile.m / 4,
                   .n8 = n8l,
                   .p_base = pb.at(std::format("{} lfc", note))},
                  std::format("{} lfc bias t{}", note, tile.t));
    if (lfc.Tok() <= 8) {
      StoreRows(lfc, tile.i0, tile.m, c,
                std::format("{} st lfc t{}", note, tile.t));
    } else {
      for (uint32_t c0 = 0; c0 < lfc.cols; c0 += 256) {
        StoreColumnsStrided(
            lfc, tile.i0, tile.m, c, n8l, c0, std::min(256U, lfc.cols - c0),
            std::format("{} st lfc t{} cols {}", note, tile.t, c0));
      }
    }
  }
  DeformableWeights(layer, pb, note);
  if (cf.det) {
    Keypoints(index, pb, note);
  } else {
    KeypointsMap(note);
  }
  OnChip a2(kAbufWords);
  const uint32_t ag = a2.Take(cf.nap / 4 * kKbD);
  constexpr std::array<uint32_t, 2> kGb = {0, 1024};
  for (uint32_t q = 0; q < cf.nap / 4; ++q) {
    const uint32_t dst = kGb[(q / 32) % 2] + ((q % 32) * kNwD);
    if (q < nq) {
      for (uint32_t camera = 0; camera < ncam; ++camera) {
        program_->Gacc({.list_offset = ListAddr(q, camera),
                        .dst = dst,
                        .slots = cf.scam,
                        .blocks = kKbD,
                        .quad = q,
                        .acc = camera > 0,
                        .flush = camera == ncam - 1},
                       std::format("{} gacc q{} c{}", note, q, camera));
      }
    } else {
      program_->Gacc({.list_offset = null_list,
                      .dst = dst,
                      .slots = 1,
                      .blocks = kKbD,
                      .quad = 0,
                      .acc = false,
                      .flush = true},
                     std::format("{} gacc pad q{}", note, q));
    }
    if (q % 32 == 31 || q == (cf.nap / 4) - 1) {
      const uint32_t t = q / 32;
      program_->Vec({.op = VecOp::kQuant,
                     .src = kGb[t % 2],
                     .dst = ag + (t * 32 * kKbD),
                     .rows_q = (q % 32) + 1,
                     .n8 = kNwD},
                    std::format("{} Q(gacc) t{}", note, t));
    }
  }
  const Lin& opj = layer.oproj;
  opj.Load(*program_, 0, std::format("{} output_proj", note));
  constexpr std::array<uint32_t, 2> kCt = {2048, 3072};
  constexpr std::array<uint32_t, 2> kXt = {0, 1024};
  for (const Tile& tile : Tiles(cf.nap, 128)) {
    const uint32_t c = kCt[tile.t % 2];
    const uint32_t xt = kXt[tile.t % 2];
    const uint32_t q0 = tile.i0 / 4;
    const uint32_t rq = tile.m / 4;
    opj.Gemm(*program_, ag + (q0 * kKbD), kKbD, tile.m, c, 0,
             std::format("{} output_proj t{}", note, tile.t));
    program_->Vec({.op = VecOp::kAddP,
                   .src = c,
                   .dst = c,
                   .rows_q = rq,
                   .n8 = kNwD,
                   .p_base = pb.at(std::format("{} oproj", note))},
                  std::format("{} output_proj bias t{}", note, tile.t));
    program_->Store({.ddr_offset = cat + (uint64_t{q0} * 64 * kWordF),
                     .mem = Mem::kFbuf,
                     .onchip = c,
                     .rows = rq,
                     .row_words = kNwD,
                     .ddr_stride = 64 * kWordF},
                    std::format("{} st cat[:256] t{}", note, tile.t));
    LoadRows(xr, tile.i0, tile.m, xt, std::format("{} ld x t{}", note, tile.t));
    program_->Store(
        {.ddr_offset = cat + (((uint64_t{q0} * 64) + kNwD) * kWordF),
         .mem = Mem::kFbuf,
         .onchip = xt,
         .rows = rq,
         .row_words = kNwD,
         .ddr_stride = 64 * kWordF},
        std::format("{} st cat[256:] t{}", note, tile.t));
  }
}

void SparseHead::DeformableWeights(const HeadLayer& layer, const Keys& pb,
                                   std::string_view note) {
  const HeadConfig& cf = C();
  const Rows& xr = x[cur_x];
  const uint32_t aw = cf.nap / 4 * kKbD;
  const uint32_t n8w = cf.scam / 8;
  uint32_t mt = std::min(448U, (kFbufWords / 4 / n8w) * 4 / 32 * 32);
  const bool two = mt >= 32;
  if (!two) {
    mt = 32;
  }
  const uint32_t cw = (mt / 4) * n8w;
  const std::array<uint32_t, 2> cb = {0, two ? cw : 0};
  Require(2 * cw <= kFbufWords, "weights tile exceeds FBUF");
  for (uint32_t camera = 0; camera < ncam; ++camera) {
    if (camera > 0) {
      program_->Load(Words(cam_vec.Addr(camera), Mem::kPbuf,
                           pb.at(std::format("{} cam", note)), kNwD / 4),
                     std::format("{} ld cam{} -> PBUF", note, camera));
      QuantRowsAdd(xr, e, cf.nap, aw,
                   std::format("{} Q(x+E+cam{})", note, camera), std::nullopt,
                   pb.at(std::format("{} cam", note)));
    }
    for (uint32_t g = 0; g < kGroups; ++g) {
      const Lin& lin = layer.wfc[g];
      const uint32_t slot = lin.TotalWords() <= 1024 ? (g % 2) * 1024 : 0;
      lin.Load(*program_, slot, std::format("{} wfc g{} c{}", note, g, camera));
      for (const Tile& tile : Tiles(cf.nap, mt)) {
        const uint32_t c = cb[tile.t % 2];
        const uint32_t q0 = tile.i0 / 4;
        if (q0 >= nq) {
          continue;
        }
        const uint32_t count = std::min(tile.m / 4, nq - q0);
        lin.Gemm(*program_, aw + (q0 * kKbD), kKbD, tile.m, c, slot,
                 std::format("{} wfc g{} c{} t{}", note, g, camera, tile.t));
        program_->Vec(
            {.op = VecOp::kAddP,
             .src = c,
             .dst = c,
             .rows_q = tile.m / 4,
             .n8 = n8w,
             .p_base = pb.at(std::format("{} wfc{}", note, g))},
            std::format("{} wfc bias g{} c{} t{}", note, g, camera, tile.t));
        program_->Store(
            {.ddr_offset = w + (((uint64_t{g} * wg) + (uint64_t{q0} * wq) +
                                 (uint64_t{camera} * n8w)) *
                                kWordF),
             .mem = Mem::kFbuf,
             .onchip = c,
             .rows = count,
             .row_words = n8w,
             .ddr_stride = static_cast<uint32_t>(uint64_t{wq} * kWordF)},
            std::format("{} st logits g{} c{} t{}", note, g, camera, tile.t));
      }
    }
  }
  const uint32_t tqs = std::max(1U, std::min(32U, kFbufWords / (2 * wq)));
  const uint32_t sd = tqs * wq;
  for (uint32_t g = 0; g < kGroups; ++g) {
    for (uint32_t q0 = 0; q0 < nq; q0 += tqs) {
      const uint32_t count = std::min(tqs, nq - q0);
      const uint64_t addr =
          w + (((uint64_t{g} * wg) + (uint64_t{q0} * wq)) * kWordF);
      program_->Load(Words(addr, Mem::kFbuf, 0, count * wq),
                     std::format("{} ld logits g{} q{}", note, g, q0));
      if (wq <= 255) {
        program_->Vec({.op = VecOp::kSoftmax,
                       .src = 0,
                       .dst = sd,
                       .rows_q = count,
                       .n8 = wq,
                       .aux = s},
                      std::format("{} softmax g{} q{}", note, g, q0));
      } else {
        program_->Run({.pc = kp.Pc("ls"),
                       .src = 0,
                       .src2 = sd,
                       .dst = sd,
                       .rows_q = count,
                       .imm = wq},
                      std::format("{} softmax g{} q{}", note, g, q0));
      }
      program_->Store(Words(addr, Mem::kFbuf, sd, count * wq),
                      std::format("{} st w g{} q{}", note, g, q0));
    }
  }
}

void SparseHead::Keypoints(size_t index, const Keys& pb,
                           std::string_view note) {
  const HeadConfig& cf = C();
  constexpr uint32_t kTq = kDetTile;
  const uint64_t stride = uint64_t{ncam} * lbc;
  constexpr uint32_t kA = 0;
  constexpr uint32_t kLf = 4 * kTq;
  constexpr uint32_t kConst = 8 * kTq;
  constexpr std::array<uint32_t, 2> kOut = {kConst + 3, kConst + 3 + kTq};
  constexpr uint32_t kWork = kConst + 3 + (2 * kTq);
  Require(kWork + DetKeypointWork(kTq) <= kFbufWords,
          "keypoint work area exceeds FBUF");
  const uint32_t pbv = pb.at(std::format("{} kplane", note));
  const uint32_t imm = kConst | (kWork << 12);
  const Rows& anchors = anc[cur_a];
  const auto run = [&](std::string_view name, uint32_t n8, uint32_t rd,
                       const std::string& what) {
    program_->Run({.pc = kp.Pc(std::string(name)),
                   .src = kA,
                   .src2 = kLf,
                   .dst = rd,
                   .rows_q = kTq,
                   .n8 = n8,
                   .p_base = pbv,
                   .imm = imm},
                  std::format("{} kp {}", note, what));
  };
  uint32_t t = 0;
  for (uint32_t q0 = 0; q0 < cf.nap / 4; q0 += kTq, ++t) {
    const uint32_t i0 = 4 * q0;
    LoadRows(anchors, i0, 4 * kTq, kA,
             std::format("{} kp ld anchor t{}", note, t));
    LoadRows(lfc, i0, 4 * kTq, kLf, std::format("{} kp ld lfc t{}", note, t));
    program_->Load(Words(kp_fs.at(index), Mem::kFbuf, kConst + 1, 1),
                   std::format("{} kp ld fix_scale", note));
    const uint32_t rows = std::min(kTq, nq - q0);
    uint32_t k = 0;
    for (uint32_t camera = 0; camera < ncam; ++camera) {
      program_->Load(
          Words(kp_const + (uint64_t{camera} * kWordF), Mem::kFbuf, kConst, 1),
          std::format("{} kp ld const c{}", note, camera));
      program_->Load(
          Words(kp_lv + (uint64_t{camera} * kWordF), Mem::kFbuf, kConst + 2, 1),
          std::format("{} kp ld levels c{}", note, camera));
      if (camera == 0) {
        run("init", 0, kOut[0], std::format("init t{}", t));
      }
      for (uint32_t pt = 0; pt < cf.npt; ++pt) {
        run("point", pt, kOut[0],
            std::format("point t{} c{} p{}", t, camera, pt));
        for (uint32_t l = 0; l < kLevels; ++l) {
          const uint32_t o = kOut[k % 2];
          ++k;
          run("level", l, o,
              std::format("level t{} c{} p{} l{}", t, camera, pt, l));
          program_->Store(
              {.ddr_offset = ListAddr(q0, camera) + 16 +
                             (((uint64_t{l} * cf.npt) + pt) * kWordF),
               .mem = Mem::kFbuf,
               .onchip = o,
               .rows = rows,
               .row_words = 1,
               .ddr_stride = static_cast<uint32_t>(stride)},
              std::format("{} kp st lists t{} c{} p{} l{}", note, t, camera, pt,
                          l));
        }
      }
    }
  }
}

void SparseHead::KeypointsMap(std::string_view note) {
  const HeadConfig& cf = C();
  const uint32_t tq = cf.nap / 4;
  const uint64_t stride = uint64_t{ncam} * lbc;
  constexpr uint32_t kA = 0;
  const uint32_t lfs = 8 * tq;
  const uint32_t cst = 16 * tq;
  const std::array<uint32_t, 2> out = {cst + 3, cst + 3 + (8 * tq)};
  const uint32_t work = cst + 3 + (16 * tq);
  Require(work + MapKeypointWork(tq) <= kFbufWords,
          "keypoint work area exceeds FBUF");
  const Rows& anchors = anc[cur_a];
  LoadRows(anchors, 0, cf.nap, kA, std::format("{} kp ld anchor", note));
  program_->Load(Words(kp_const, Mem::kFbuf, cst, 1),
                 std::format("{} kp ld const", note));
  program_->Load(Words(kp_zh, Mem::kFbuf, cst + 2, 1),
                 std::format("{} kp ld z", note));
  const uint32_t imm0 = cst | (work << 12);
  const auto run = [&](std::string_view name, uint32_t n8, uint32_t rd,
                       uint32_t imm, const std::string& what) {
    program_->Run({.pc = kp.Pc(std::string(name)),
                   .src = kA,
                   .src2 = lfs,
                   .dst = rd,
                   .rows_q = tq,
                   .n8 = n8,
                   .p_base = 0,
                   .imm = imm},
                  std::format("{} kp {}", note, what));
  };
  run("init", 0, out[0], imm0, "init");
  uint32_t k = 0;
  for (uint32_t gi = 0; gi < kMapGroups; ++gi) {
    const auto [h, lq, gs] = MapGroup(gi);
    const uint32_t col0 = ((h * kMapLearn) + lq) * kMapBlock;
    program_->Load({.ddr_offset = lfc.Addr(0) + (uint64_t{col0} * 4),
                    .mem = Mem::kFbuf,
                    .onchip = lfs,
                    .rows = cf.nap,
                    .row_words = 1,
                    .ddr_stride = static_cast<uint32_t>(lfc.Tb()),
                    .mode = DmaMode::kRaw,
                    .index = false,
                    .row_format = true,
                    .tok = kMapBlock / 32},
                   std::format("{} kp ld lfc blk h{} lq{}", note, h, lq));
    const uint32_t imm = imm0 | (h << 24) | (lq << 27) | (gs << 29);
    for (uint32_t camera = 0; camera < ncam; ++camera) {
      program_->Load(
          Words(kp_const + (uint64_t{camera} * kWordF), Mem::kFbuf, cst, 1),
          std::format("{} kp ld const g{} c{}", note, gi, camera));
      program_->Load(
          Words(kp_lv + (uint64_t{camera} * kWordF), Mem::kFbuf, cst + 1, 1),
          std::format("{} kp ld levels g{} c{}", note, gi, camera));
      run("point", 0, out[0], imm, std::format("point g{} c{}", gi, camera));
      for (uint32_t l = 0; l < kLevels; ++l) {
        const uint32_t o = out[k % 2];
        ++k;
        run("level", l, o, imm,
            std::format("level g{} c{} l{}", gi, camera, l));
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
               .onchip = o + (i * tq),
               .rows = nq,
               .row_words = 1,
               .ddr_stride = static_cast<uint32_t>(stride)},
              std::format("{} kp st lists c{} p{} l{}", note, camera, pt, l));
        }
      }
    }
  }
}

void SparseHead::Ffn(const HeadLayer& layer, const Keys& pb,
                     std::string_view note) {
  const HeadConfig& cf = C();
  const Rows& xo = x[(cur_x + 1) % 3];
  OnChip a(kAbufWords);
  const uint32_t aff = a.Take(cf.nap / 4 * 16);
  const uint32_t ah0 = a.Take(512);
  const uint32_t ah1 = a.Take(512);
  constexpr std::array<uint32_t, 2> kCt = {0, 1024};
  constexpr uint32_t kScratch = 2048;
  for (const Tile& tile : Tiles(cf.nap, 64)) {
    const uint32_t slot = kCt[tile.t % 2];
    const uint32_t q0 = tile.i0 / 4;
    program_->Load(Words(cat + (uint64_t{q0} * 64 * kWordF), Mem::kFbuf, slot,
                         (tile.m / 4) * 64),
                   std::format("{} ld cat t{}", note, tile.t));
    program_->Vec({.op = VecOp::kLayerNorm,
                   .src = slot,
                   .dst = aff + (q0 * 16),
                   .rows_q = tile.m / 4,
                   .n8 = 64,
                   .p_base = pb.at(std::format("{} ln", note)),
                   .imm = LnEpsBits(),
                   .q = true,
                   .scratch = kScratch},
                  std::format("{} LN+Q t{}", note, tile.t));
  }
  constexpr uint32_t kH1 = 0;
  constexpr uint32_t kOut = 2048;
  for (const Tile& tile : Tiles(cf.nap, 128)) {
    const uint32_t q0 = tile.i0 / 4;
    const uint32_t rq = tile.m / 4;
    for (uint32_t nt = 0; nt < 2; ++nt) {
      const uint32_t ah = nt == 0 ? ah0 : ah1;
      program_->Load(Words(layer.fc1.Offset(0) +
                               (uint64_t{nt} * 64 * layer.fc1.Kb() * kWordA),
                           Mem::kWbuf, nt * 1024, 64 * layer.fc1.Kb()),
                     std::format("{} ld fc1 nt{} t{}", note, nt, tile.t));
      program_->Gemm({.a_base = aff + (q0 * 16),
                      .a_stride = 16,
                      .kb = 16,
                      .m = tile.m,
                      .w_base = nt * 1024,
                      .w_stride = layer.fc1.Kb(),
                      .n8 = 64,
                      .c_base = kH1},
                     std::format("{} fc1 nt{} t{}", note, nt, tile.t));
      program_->Vec({.op = VecOp::kAddP,
                     .src = kH1,
                     .dst = kH1,
                     .rows_q = rq,
                     .n8 = 64,
                     .p_base = pb.at(std::format("{} fc1", note)) + (nt * 16)},
                    std::format("{} fc1 bias nt{} t{}", note, nt, tile.t));
      program_->Vec({.op = VecOp::kRelu,
                     .src = kH1,
                     .dst = ah,
                     .rows_q = rq,
                     .n8 = 64,
                     .q = true},
                    std::format("{} fc1 relu+Q nt{} t{}", note, nt, tile.t));
    }
    layer.fc2.Load(*program_, 0, std::format("{} fc2 t{}", note, tile.t));
    layer.idf.Load(*program_, 1024, std::format("{} idf t{}", note, tile.t));
    layer.fc2.Gemm(*program_, 0, 16, tile.m, kOut, 0,
                   std::format("{} fc2 t{}", note, tile.t),
                   {{.a_offset = ah0, .kb_offset = 0, .kb = 16},
                    {.a_offset = ah1, .kb_offset = 16, .kb = 16}});
    layer.idf.Gemm(*program_, aff + (q0 * 16), 16, tile.m, kOut, 1024,
                   std::format("{} identity_fc t{}", note, tile.t), {}, true);
    program_->Vec({.op = VecOp::kAddP,
                   .src = kOut,
                   .dst = kOut,
                   .rows_q = rq,
                   .n8 = kNwD,
                   .p_base = pb.at(std::format("{} fc2", note))},
                  std::format("{} bias t{}", note, tile.t));
    StoreRows(xo, tile.i0, tile.m, kOut,
              std::format("{} st x t{}", note, tile.t));
  }
  cur_x = (cur_x + 1) % 3;
}

void SparseHead::Refine(const HeadLayer& layer, const Keys& pb,
                        std::string_view note) {
  const HeadConfig& cf = C();
  constexpr uint32_t kTile = 224;
  const Rows& xr = x[cur_x];
  const Rows& a_in = anc[cur_a];
  const Rows& a_out = anc[1 - cur_a];
  OnChip a(kAbufWords);
  const uint32_t ar0 = a.Take(cf.nap / 4 * kKbD);
  const uint32_t ar1 = a.Take(cf.nap / 4 * kKbD);
  const uint32_t ax = a.Take(cf.nap / 4 * kKbD);
  const uint32_t ar2 = a.Take(cf.nap / 4 * kKbD);
  constexpr uint32_t kCt = 0;
  constexpr uint32_t kScratch = 1792;
  constexpr uint32_t kAt = 3584;
  constexpr std::array<uint32_t, 2> kEt = {2048, 3072};
  QuantRows(xr, cf.nap, ar0, std::format("{} Q(x+E)", note),
            [&](uint32_t t, uint32_t slot, uint32_t rq, uint32_t i0) {
              LoadRows(e, i0, 4 * rq, kEt[t % 2],
                       std::format("{} ld E t{}", note, t));
              program_->Vec({.op = VecOp::kQuant,
                             .src = slot,
                             .dst = ax + ((i0 / 4) * kKbD),
                             .rows_q = rq,
                             .n8 = kNwD},
                            std::format("{} Q(x) t{}", note, t));
              program_->Vec({.op = VecOp::kAdd,
                             .src = slot,
                             .src2 = kEt[t % 2],
                             .dst = slot,
                             .rows_q = rq,
                             .n8 = kNwD},
                            std::format("{} x+E t{}", note, t));
              if (cf.det) {
                program_->Vec({.op = VecOp::kCopy,
                               .src = slot,
                               .dst = ar2 + ((i0 / 4) * kKbD),
                               .rows_q = rq,
                               .n8 = kNwD,
                               .q = true},
                              std::format("{} Q(x+E) copy t{}", note, t));
              }
            });
  const auto chain = [&](const std::vector<Lin>& lins,
                         const std::function<std::string(size_t)>& lns,
                         uint32_t src, const std::string& bkey,
                         const std::function<bool(size_t)>& relu_ln) {
    for (size_t i = 0; i < lins.size(); ++i) {
      const uint32_t dst = src != ar1 ? ar1 : ar0;
      const std::string name = std::format("{}{}", bkey, i);
      lins[i].Load(*program_, 0, name);
      for (const Tile& tile : Tiles(cf.nap, kTile)) {
        const uint32_t q0 = tile.i0 / 4;
        const uint32_t rq = tile.m / 4;
        lins[i].Gemm(*program_, src + (q0 * kKbD), kKbD, tile.m, kCt, 0,
                     std::format("{} t{}", name, tile.t));
        program_->Vec({.op = VecOp::kAddP,
                       .src = kCt,
                       .dst = kCt,
                       .rows_q = rq,
                       .n8 = kNwD,
                       .p_base = pb.at(name)},
                      std::format("{} bias t{}", name, tile.t));
        if (!relu_ln(i)) {
          program_->Vec({.op = VecOp::kRelu,
                         .src = kCt,
                         .dst = dst + (q0 * kKbD),
                         .rows_q = rq,
                         .n8 = kNwD,
                         .q = true},
                        std::format("{} relu+Q t{}", name, tile.t));
        } else {
          program_->Vec({.op = VecOp::kRelu,
                         .src = kCt,
                         .dst = kCt,
                         .rows_q = rq,
                         .n8 = kNwD},
                        std::format("{} relu t{}", name, tile.t));
          program_->Vec({.op = VecOp::kLayerNorm,
                         .src = kCt,
                         .dst = dst + (q0 * kKbD),
                         .rows_q = rq,
                         .n8 = kNwD,
                         .p_base = pb.at(lns(i)),
                         .imm = LnEpsBits(),
                         .q = true,
                         .scratch = kScratch},
                        std::format("{} LN+Q t{}", name, tile.t));
        }
      }
      src = dst;
    }
    return src;
  };
  const auto head_out =
      [&](const Lin& lin, uint32_t src, const Rows& rows_out,
          const std::string& bkey,
          const std::function<void(uint32_t, uint32_t, uint32_t, uint32_t)>&
              post) {
        lin.Load(*program_, 0, bkey);
        for (const Tile& tile : Tiles(cf.nap, kTile)) {
          const uint32_t q0 = tile.i0 / 4;
          const uint32_t rq = tile.m / 4;
          lin.Gemm(*program_, src + (q0 * kKbD), kKbD, tile.m, kCt, 0,
                   std::format("{} t{}", bkey, tile.t));
          program_->Vec({.op = VecOp::kAddP,
                         .src = kCt,
                         .dst = kCt,
                         .rows_q = rq,
                         .n8 = lin.Groups(),
                         .p_base = pb.at(bkey)},
                        std::format("{} bias t{}", bkey, tile.t));
          if (post) {
            post(tile.t, tile.i0, tile.m, lin.Groups());
          }
          StoreRows(rows_out, tile.i0, tile.m, kCt,
                    std::format("{} st t{}", bkey, tile.t));
        }
      };
  uint32_t src = chain(
      layer.reg,
      [&](size_t i) { return std::format("{} reg_ln{}", note, i / 2); }, ar0,
      std::format("{} reg", note), [](size_t i) { return i % 2 == 1; });
  head_out(layer.reg_out, src, a_out, std::format("{} reg_out", note),
           [&](uint32_t t, uint32_t i0, uint32_t m, uint32_t n8) {
             LoadRows(a_in, i0, m, kAt,
                      std::format("{} ld anchor t{}", note, t));
             program_->Vec({.op = VecOp::kMulP,
                            .src = kCt,
                            .dst = kCt,
                            .rows_q = m / 4,
                            .n8 = n8,
                            .p_base = pb.at(std::format("{} scale", note))},
                           std::format("{} scale t{}", note, t));
             program_->Vec({.op = VecOp::kAdd,
                            .src = kCt,
                            .src2 = kAt,
                            .dst = kCt,
                            .rows_q = m / 4,
                            .n8 = n8},
                           std::format("{} + anchor t{}", note, t));
           });
  src = chain(
      layer.cls, [&](size_t i) { return std::format("{} cls_ln{}", note, i); },
      ax, std::format("{} cls", note), [](size_t) { return true; });
  head_out(layer.cls_out, src, cls, std::format("{} cls_out", note), {});
  if (cf.det) {
    src = chain(
        layer.qt, [&](size_t i) { return std::format("{} qt_ln{}", note, i); },
        ar2, std::format("{} qt", note), [](size_t) { return true; });
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
  for (const Tile& tile : Tiles(C().nap, 128)) {
    const uint32_t slot = tile.t % 2 == 0 ? 0 : 1024;
    program_->Load(Tokens(order_use[0] + (uint64_t{tile.i0} * 4), Mem::kFbuf,
                          slot, tile.m, xr.Tok(), true),
                   std::format("{} gather x t{}", note, tile.t));
    StoreRows(xo, tile.i0, tile.m, slot,
              std::format("{} st x t{}", note, tile.t));
  }
  program_->Load(
      Tokens(order_use[1], Mem::kFbuf, 2048, C().nap, a_in.Tok(), true),
      std::format("{} gather anchors", note));
  StoreRows(a_out, 0, C().nap, 2048, std::format("{} st anchors", note));
  cur_x = (cur_x + 1) % 3;
  cur_a = 1 - cur_a;
}

void SparseHead::Keep(std::string_view note) {
  const Rows& xr = x[cur_x];
  for (const Tile& tile : Tiles(C().ntp, 128)) {
    const uint32_t slot = tile.t % 2 == 0 ? 0 : 1024;
    program_->Load(Tokens(keep + (uint64_t{tile.i0} * 4), Mem::kFbuf, slot,
                          tile.m, xr.Tok(), true),
                   std::format("{} gather keep t{}", note, tile.t));
    StoreRows(tx, tile.i0, tile.m, slot,
              std::format("{} st temp_x t{}", note, tile.t));
  }
}

void SparseHead::Init(std::string_view note) {
  for (const Tile& tile : Tiles(C().nap, 128)) {
    const uint32_t slot = tile.t % 2 == 0 ? 0 : 1024;
    LoadRows(x_init, tile.i0, tile.m, slot,
             std::format("{} ld x_init t{}", note, tile.t));
    StoreRows(x[0], tile.i0, tile.m, slot,
              std::format("{} st x t{}", note, tile.t));
  }
  LoadRows(a_init, 0, C().nap, 2048, std::format("{} ld anchor_init", note));
  StoreRows(anc[0], 0, C().nap, 2048, std::format("{} st anchor", note));
  cur_x = 0;
  cur_a = 0;
}

void SparseHead::KeepAnchor(std::string_view note) {
  const Rows& a = anc[cur_a];
  for (const Tile& tile : Tiles(C().ntp, 128)) {
    const uint32_t slot = tile.t % 2 == 0 ? 0 : 1024;
    program_->Load(Tokens(keep_a + (uint64_t{tile.i0} * 4), Mem::kFbuf, slot,
                          tile.m, a.Tok(), true),
                   std::format("{} gather t{}", note, tile.t));
    StoreRows(tanc_raw, tile.i0, tile.m, slot,
              std::format("{} st t{}", note, tile.t));
  }
}

void SparseHead::AnchorProject(std::string_view note) {
  const uint32_t words = C().acols / 8;
  const uint32_t n = C().ntp;
  program_->Load(Words(pjprog, Mem::kCbuf, vprog::kRomWords,
                       static_cast<uint32_t>(pj.Words().size())),
                 std::format("{} ld VEC program", note));
  cbuf_has_ = kCbufProject;
  program_->Load(Words(proj_pb, Mem::kPbuf, 0, C().det ? 18 : 9),
                 std::format("{} ld PBUF", note));
  program_->Load(Words(proj_dt, Mem::kFbuf, 2 * n * words / 4, 1),
                 std::format("{} ld dt", note));
  LoadRows(tanc_raw, 0, n, 0, std::format("{} ld raw", note));
  program_->Run({.pc = vprog::kRomWords,
                 .src = 0,
                 .src2 = 2 * n * words / 4,
                 .dst = n * words / 4,
                 .rows_q = n / 4,
                 .n8 = words,
                 .p_base = 0},
                std::format("{} run", note));
  StoreRows(tanc, 0, n, n * words / 4, std::format("{} st temp_anchor", note));
}

void SparseHead::EnsureCbuf(int which) {
  if (cbuf_has_ == which) {
    return;
  }
  if (which == kCbufKeypoints) {
    program_->Load(Words(vprog, Mem::kCbuf, vprog::kRomWords,
                         static_cast<uint32_t>(kp.Words().size())),
                   std::format("{} ld VEC programs (keypoints)", tag));
  } else {
    program_->Load(Words(ibprog, Mem::kCbuf, vprog::kRomWords,
                         static_cast<uint32_t>(ib.Words().size())),
                   std::format("{} ld VEC programs (instance bank)", tag));
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
  LoadRows(cls, 0, cf.nap, 0, std::format("{} ld cls", note));
  program_->Load(Words(ib_idx, Mem::kFbuf, work + nw, nw),
                 std::format("{} ld idx", note));
  program_->Load(Words(ib_lane, Mem::kFbuf, work + (6 * nw) + 2, 1),
                 std::format("{} ld lane", note));
  if (update) {
    program_->Load(Words(order[0], Mem::kFbuf, out, nw),
                   std::format("{} ld order0", note));
    program_->Load(Words(order[1], Mem::kFbuf, out + nw, nw),
                   std::format("{} ld order1", note));
  } else {
    program_->Load(Words(keep, Mem::kFbuf, out, nw),
                   std::format("{} ld keep", note));
    program_->Load(Words(prevconf, Mem::kFbuf, out + (3 * nw), nw),
                   std::format("{} ld prevconf", note));
    program_->Load(Words(keep_a, Mem::kFbuf, out + (4 * nw), nw),
                   std::format("{} ld keep_a", note));
  }
  program_->Load(Words(ib_const_.at(key), Mem::kFbuf, cst, 1),
                 std::format("{} ld const", note));
  program_->Run({.pc = ib.Pc(update ? "update" : "cache"),
                 .src = 0,
                 .src2 = work,
                 .dst = out,
                 .rows_q = cf.nap / 4,
                 .n8 = ncls,
                 .p_base = 0,
                 .imm = cst},
                std::format("{} run", note));
  if (update) {
    program_->Store(Words(order[0], Mem::kFbuf, out, nw),
                    std::format("{} st order0", note));
    program_->Store(Words(order[1], Mem::kFbuf, out + nw, nw),
                    std::format("{} st order1", note));
    const auto id_key = std::pair{cur_x, cur_a};
    if (!order_id_.contains(id_key)) {
      std::vector<uint32_t> ids(2 * static_cast<size_t>(cf.nap), kNull);
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
    const uint32_t fl = cst + 1;
    const uint32_t id = cst + 2;
    program_->Load(Words(fs_flag, Mem::kFbuf, fl, 1),
                   std::format("{} ld flag", note));
    program_->Load(Words(order_id_.at(id_key), Mem::kFbuf, id, 2 * nw),
                   std::format("{} ld identity", note));
    for (uint32_t i = 0; i < 2; ++i) {
      program_->Run({.pc = ib.Pc("copy"),
                     .src = id + (i * nw),
                     .src2 = fl,
                     .dst = out + (i * nw),
                     .rows_q = nw},
                    std::format("{} identity{} if !mask", note, i));
      program_->Store(Words(order_use[i], Mem::kFbuf, out + (i * nw), nw),
                      std::format("{} st order_use{}", note, i));
    }
  } else {
    program_->Store(Words(keep, Mem::kFbuf, out, nw),
                    std::format("{} st keep", note));
    program_->Store(Words(prevconf, Mem::kFbuf, out + nw, nw),
                    std::format("{} st selconf", note));
    program_->Store(Words(selidx, Mem::kFbuf, out + (2 * nw), nw),
                    std::format("{} st selidx", note));
    program_->Store(Words(keep_a, Mem::kFbuf, out + (4 * nw), nw),
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
         .onchip = 0,
         .rows = 1,
         .row_words = kTbufWords,
         .ddr_stride = static_cast<uint32_t>(uint64_t{kTbufWords} * kWordT)},
        std::format("{} ld TBUF", tag));
    cbuf_has_ = 0;
    if (temp) {
      AnchorProject(std::format("{} temp anchor proj", tag));
    }
    Init(std::format("{} init", tag));
    AnchorEncoder(std::format("{} E0", tag));
  }
  uint32_t refines = 0;
  for (size_t l = 0; l < p.layers.size(); ++l) {
    const HeadLayer& layer = p.layers[l];
    const auto& [pb, keys, note] = pb_layers[l];
    const Program::Scope scope(program,
                               std::format("L{}.{}", l, OpName(layer.op)));
    pb.Load(program, note);
    switch (layer.op) {
      case Op::kGnn:
      case Op::kTempGnn:
        Attention(layer, keys, note);
        break;
      case Op::kNorm:
        Norm(keys, note);
        break;
      case Op::kDeformable:
        CameraEncoder(layer, keys, note + " cam");
        program.Load(Words(cam_vec.Addr(0), Mem::kPbuf, keys.at(note + " cam"),
                           kNwD / 4),
                     std::format("{} ld cam -> PBUF", note));
        EnsureCbuf(kCbufKeypoints);
        Deformable(layer, l, keys, note);
        break;
      case Op::kFfn:
        Ffn(layer, keys, note);
        break;
      case Op::kCrossGnn:
        break;
      case Op::kRefine:
        Refine(layer, keys, note);
        ++refines;
        if (refines == 1 && temp) {
          Bank(true, std::format("{} ibank update", tag));
          Reorder(std::format("{} reorder", tag));
        }
        AnchorEncoder(std::format("{} E after L{}", tag, l));
        break;
    }
  }
  const Program::Scope scope(program, "end");
  Bank(false, std::format("{} ibank cache", tag));
  Keep(std::format("{} keep", tag));
  KeepAnchor(std::format("{} keep anchor", tag));
}

}  // namespace hotarubi::compiler::sparsedrive
